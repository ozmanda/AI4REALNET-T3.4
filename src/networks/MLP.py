'''
From IC3Net models.py, continuous case removed (flatland has discrete action space) and only single actions considered.
'''

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from typing import Tuple, Union

class MLP(nn.Module):
    def __init__(self, args: Namespace, num_inputs: int) -> None:
        super().__init__()
        """
        Basic MLP network with fully connected layers and action / value heads. The action
        and value heads are synonymous with actor and critic networks, respectively. 
        """
        self.multiagent = False
        self.args = args
        self.n_features = num_inputs
        self.fc1 = nn.Linear(self.n_features, args.hid_size)
        self.fc2 = nn.Linear(args.hid_size, args.hid_size)
        self.actor = nn.Linear(args.hid_size, args.n_actions)
        self.actor_target = nn.Linear(args.hid_size, args.n_actions)
        self.critic = nn.Linear(args.hid_size, 1)


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        ''' Outputs the log-probability of each action in a Tensor of length n_actions'''
        x = self.check_tensor_dimensions(x)
        x = nn.Tanh(self.fc1(x))
        h = nn.Tanh(sum([self.fc2(x), x]))
        v = self.critic(h)
        return F.log_softmax(self.actor(h), dim=-1), v
        

    def check_tensor_dimensions(self, input_tensor: Tensor) -> Tensor: 
        """Check the dimensions of the input tensor and adjust for the multi-agent case. This function assumes dimensions of shape (batch_size, n_agents, n_features) or (batch_size, n_features)."""
        if len(input_tensor.shape) == 3:
            self.multiagent = True
            self.n_agents = input_tensor.size(1)
            input_tensor = input_tensor.view(-1, self.n_features)
        if len(input_tensor.size) != 2:
            raise ValueError(f'Input tensor has incorrect dimensions: {input_tensor.size()}')
        return input_tensor
    

    def adjust_output_dimensions(self, output_tensor: Union[Tensor, Tuple]) -> Tensor:
        """ Return the output tensor(s) to the original dimensions for the multi-agent case, also handles multiple outputs in the form of a tuple. """
        if self.multiagent:
            if isinstance(output_tensor, tuple):
                return tuple([output_tensor[i].view(-1, self.n_agents, output_tensor[i].size(1)) for i in range(len(output_tensor))])
            else: 
                return output_tensor.view(-1, self.n_agents, output_tensor.size(1))
        