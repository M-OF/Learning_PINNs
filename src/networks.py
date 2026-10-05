# Network architectures
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn

from typing import Tuple

class PINN(nn.Module):
    def __init__(
                self, 
                 N_IN: int, 
                 N_HID: int, 
                 N_OUT: int,
                 activation = None,
                 init_mode = None,
                 ):
        super().__init__()

        if activation is None:
            activation_cls = torch.nn.Tanh
        elif isinstance(activation,nn.Module):
            activation_cls = activation.__class__
        else:
            activation_cls = activation

        activation_name = activation_cls.__name__.lower()

        if 'relu' in activation_name or 'silu' in activation_name:
            self.init_mode = 'kaiming'

        elif "tanh" in activation_name:
            self.init_mode = "xavier"

        else:
            self.init_mode = 'xavier'
            activation_cls = nn.Tanh

        self.l1 = nn.Linear(N_IN, N_HID)
        self.l2 = nn.Linear(N_HID, N_HID)
        self.out = nn.Linear(N_HID, N_OUT)

        layers = []
        layers.extend([self.l1, activation_cls(), self.l2, activation_cls(), self.out])
        self.network = nn.Sequential(*layers)

        self.initialize_net()

    def initialize_net(self):
        for l in self.network:
            if isinstance(l, nn.Linear):
                if self.init_mode == 'kaiming':
                    nn.init.kaiming_uniform_(l.weight, nonlinearity="relu")
                elif self.init_mode == 'xavier':
                    nn.init.xavier_uniform_(l.weight)


    def forward(self,x):
        return self.network(x)