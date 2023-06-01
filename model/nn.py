import torch.distributions
from torch import nn
import torch
import numpy as np
from torch._C import device
from .fully_connected import get_fully_connected_layer

class NN(nn.Module):
    def __init__(self, nn_args, nn_kwargs):
        super(NN, self).__init__()
        self.model = get_fully_connected_layer(*nn_args, **nn_kwargs)
        self.device = device

    def forward(self, x):
        output = self.model(x)
        return output