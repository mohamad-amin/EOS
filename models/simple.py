from flax import linen as nn
from .torch_layers import *
from typing import Callable


class MLP(nn.Module):
    activation: Callable
    n_classes: int
    width: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = TorchLinear(self.width)(x)
        x = self.activation(x)
        x = TorchLinear(self.width)(x)
        x = self.activation(x)
        x = TorchLinear(self.n_classes)(x)
        return x


class CNN(nn.Module):
    activation: Callable
    n_classes: int
    width: int

    @nn.compact
    def __call__(self, x):
        x = TorchConv(self.width, (3, 3))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, (2, 2), (2, 2), "SAME")
        x = TorchConv(self.width, (3, 3))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, (2, 2), (2, 2), "SAME")
        x = x.reshape(x.shape[0], -1)
        x = TorchLinear(self.n_classes)(x)
        return x


class NormalizedMLP(nn.Module):
    activation: Callable
    n_classes: int
    width: int
    normalization_scale = None
    normalization_bias = None

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = TorchLinear(self.width)(x)
        x = self.activation(x)
        x = TorchLinear(self.width)(x)
        x = self.activation(x)
        x = TorchLinear(self.n_classes)(x)
        x = self.normalization_scale * TorchFixedBN(use_running_average=False)(x) + self.normalization_bias
        return x


class NormalizedCNN(nn.Module):
    activation: Callable
    n_classes: int
    width: int
    normalization_scale: float = 0.0
    normalization_bias: float = 1

    @nn.compact
    def __call__(self, x):
        x = TorchConv(self.width, (3, 3))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, (2, 2), (2, 2), "SAME")
        x = TorchConv(self.width, (3, 3))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, (2, 2), (2, 2), "SAME")
        x = x.reshape(x.shape[0], -1)
        x = TorchLinear(self.n_classes)(x)
        x = self.normalization_scale * TorchFixedBN(use_running_average=False)(x) + self.normalization_bias
        return x
