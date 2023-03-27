import os
import jax
import torch

jax.config.update("jax_enable_x64", True)
from pathlib import Path
from absl import app, flags
from jax import numpy as jnp, random
from jax.numpy import linalg as jla
from tqdm.auto import tqdm, trange
from dictionaries import data_dict, model_dict, criterion_dict, activation_dict
from loss import build_loss
from util import tree_save, load_config
from taylor import find_instability, track_dynamics
import json
from functools import partial
import shutil
from torch.utils.tensorboard import SummaryWriter


class BaseEngine(object):

    def __init__(self, config_path, logger, save_dir):
        # Assign a logger
        self.logger = logger

        # Load configurations
        self.config = load_config(config_path)

        self.model_config = self.config['model']
        self.train_config = self.config['train']
        self.data_config = self.config['data']
        self.computation_config = self.config['computation']

        # Determine which device to use
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.num_devices = torch.cuda.device_count()

        if device == 'cpu':
            self.logger.warn('GPU is not available.')
        else:
            self.logger.warn('GPU is available with {} devices.'.format(self.num_devices))
        self.logger.warn('CPU is available with {} devices.'.format(jax.device_count('cpu')))

        # Load a summary writer
        self.save_dir = save_dir
        log_dir = os.path.join(self.save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def run(self):
        pass


class Engine(BaseEngine):

    def __init__(self, config_path, logger, save_dir):
        super(Engine, self).__init__(config_path, logger, save_dir)

    def _build(self):

        def build_data(data_config):
            return data_dict[data_config['name']](data_config['train_count'])
        self.data = build_data(self.data_config)

        def build_criterion(train_config):
            return criterion_dict[train_config['criterion']]
        self.criterion = build_criterion(self.train_config)

        labels = jax.nn.one_hot(self.data[1], self.model_config['n_classes'])
        mu = labels.sum(axis=0) / len(labels)
        std = jnp.sqrt(((labels - mu) ** 2).mean(axis=0))

        def build_model(model_config):
            width = model_config['width']
            num_classes = model_config['n_classes']
            activation = activation_dict[model_config['activation']]
            name = model_config['name']
            kwargs = {
                'activation': activation,
                'n_classes': num_classes,
                'width': width
            }
            if name.startswith('normalized_'):
                kwargs.update({
                    'normalization_scale': mu,
                    'normalization_bias': std
                })
            return model_dict[name](**kwargs)
        self.model = build_model(self.model_config)

        model_key, self.eig_key = random.split(random.PRNGKey(self.data_config['seed']))
        self.p, loss = build_loss(
            model=self.model,
            data=self.data,
            criterion=self.criterion,
            batch_size=min(self.train_config['ghost_batch_size'], self.data_config['train_count']),
            model_key=model_key,
        )
        dtype_dict = dict(f32=jnp.float32, f64=jnp.float64)
        self.loss = loss._replace(
            D=partial(loss.D, dtype=dtype_dict[self.computation_config.get('deriv_dtype', 'f32')]),
            eig=partial(
                loss.eig,
                tol=self.computation_config.get('solver_tol', 1e-9),
                hvp_dtype=dtype_dict[self.computation_config.get('hvp_dtype', 'f32')],
                solver_dtype=dtype_dict[self.computation_config.get('solver_dtype', 'f32')],
            ),
        )

    def run(self):

        self._build()

        lr = self.train_config['lr']
        p = self.p.astype(jnp.float32)
        U = random.normal(self.eig_key, (len(p), 2), dtype=p.dtype)

        print("Running Until Instability")
        p, U0, prelim = find_instability(p, U[:, :1], lr, self.loss, self.writer)
        tree_save(prelim, self.save_dir + "/prelim.pytree", overwrite=True)
        print("Switching to 64 bit")
        p = p.astype(jnp.float64)
        U = U.astype(jnp.float64).at[:, :1].set(U0)
        _, U = self.loss.eig(p, U)
        print("Tracking Dynamics")
        track_dynamics(
            p,
            U,
            lr,
            self.loss,
            steps=self.computation_config.get('steps', 2000),
            num_proj_steps=self.computation_config.get('num_proj_steps', 3),
            generalized_pred=self.computation_config.get('generalized_pred', 3),
            save_dir=self.save_dir,
            writer=self.writer
        )