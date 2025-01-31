'''
From IC3Net models.py, continuous case removed (flatland has discrete action space)
'''

import torch
from torch import Tensor
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

class MLP(nn.Module):
    def __init__(self, args: Namespace, num_inputs: int):
        super().__init__()

        self.args = args
        self.fc1 = nn.Linear(num_inputs, args.hid_size)
        self.fc2 = nn.Linear(args.hid_size, args.hid_size)

        self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
        self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))

        self.value_head = nn.Linear(args.hid_size, 1)


    def forward(self, x: Tensor, info={}):
        x = nn.Tanh(self.fc1(x))
        h = nn.Tanh(sum([self.fc2(x), x]))
        v = self.value_head(h)

        action_mean = self.action_mean(h)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return (action_mean, action_log_std, action_std), v
        