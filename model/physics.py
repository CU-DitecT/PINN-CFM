import torch
import math
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F


class IDM(torch.nn.Module):
    def __init__(self, params_value, params_trainable,
                 device=None):

        super(IDM, self).__init__()
        self.torch_params = dict()
        self.params_trainable = params_trainable
        for k, v in params_value.items():
            if params_trainable[k] is True:
                self.torch_params[k] = torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device),
                                                               requires_grad=True,
                                                               )
                self.torch_params[k].retain_grad()
            else:
                self.torch_params[k] = torch.nn.Parameter(torch.tensor(v, dtype=torch.float32),
                                                               requires_grad=False,
                                                               )


    def forward(self, x):
        dx = x[:, 0]
        dv = x[:, 1]
        v = x[:, 2]

        s0 = self.torch_params["s0"]
        v0 = self.torch_params["v0"]
        T = self.torch_params["T"]
        a = self.torch_params["a"]
        b = self.torch_params["b"]

        # idm equation
        s_star = s0 + T*v - v*dv/2/torch.sqrt(a * b)
        acc = a * (1 - torch.pow(v / v0, 4) - torch.pow(s_star / dx, 2))

        return acc










