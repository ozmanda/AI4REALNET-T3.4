'''
From IC3Net models.py, continuous case removed (flatland has discrete action space) and only single actions considered.
'''

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, args: Namespace, num_inputs: int) -> None:
        super().__init__()
        """
        Basic MLP network with fully connected layers and action / value heads. The action
        and value heads are synonymous with actor and critic networks, respectively. 
        """

        self.args = args
        self.fc1 = nn.Linear(num_inputs, args.hid_size)
        self.fc2 = nn.Linear(args.hid_size, args.hid_size)
        self.actor = nn.Linear(args.hid_size, args.n_actions)
        self.critic = nn.Linear(args.hid_size, 1)


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        ''' Outputs the log-probability of each action in a Tensor of length n_actions'''
        x = nn.Tanh(self.fc1(x))
        h = nn.Tanh(sum([self.fc2(x), x]))
        v = self.critic(h)
        return F.log_softmax(self.actor(h), dim=-1), v
        