import torch
import torch.nn as nn
import os
import json

class base_model(nn.Module):
    def __init__(self):
        super(base_model, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265]))
        self.pi_const.requires_grad = False