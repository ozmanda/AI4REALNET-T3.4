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
        self.n_agents = args.n_agents
        self.args = args
        self.n_features = num_inputs
        self.hid_size = args.hid_size
        self.n_actions = args.n_actions
        self.fc1 = nn.Linear(self.n_features, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, self.hid_size)
        self.actor = nn.Linear(self.hid_size, self.n_actions)
        self.actor_target = nn.Linear(self.hid_size, self.n_actions)
        self.critic = nn.Linear(self.hid_size, 1)
        self.tanh = nn.Tanh()


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        ''' Outputs the log-probability of each action in a Tensor of length n_actions'''
        x = self.adjust_input_dimensions(x)
        encoded_x = self.fc1(x)
        x = self.tanh(encoded_x)
        h = self.tanh(sum([self.fc2(x), x]))
        v = self.critic(h)
        output = self.adjust_output_dimensions((F.log_softmax(self.actor(h), dim=-1), v))
        return output
        

    def adjust_input_dimensions(self, input: Union[Tensor, Tuple[Tensor, ...]]) -> Union[Tensor, Tuple]: 
        """
        Check the dimensions of the input tensor and adjust for the multi-agent case. This function assumes dimensions of shape (batch_size, n_agents, n_features) or (batch_size, n_features).
        """

        if isinstance(input, tuple):
            if input[0].dim() == 3:
                self.multiagent = True
                self.n_agents = input[0].size(1)
                input = tuple([input[i].view(-1, input[i].size(-1)) for i in range(len(input))])
            if any([input[i].dim() != 2 for i in range(len(input))]):
                raise ValueError(f'Input tensor has incorrect dimensions: {[input[i].size() for i in range(len(input))]}')
            return input
            
        elif isinstance(input, Tensor):
            if input.dim() != 2:
                if input.dim() == 3:
                    self.multiagent = True
                    self.n_agents = input.size(1)
                    input = input.view(-1, input.size(-1))
                else:
                    raise ValueError(f'Input tensor has incorrect dimensions: {input.size()}')
            return input
        else:
            raise ValueError(f'Input tensor is not a Tensor or Tuple: {type(input)}')
    

    def adjust_output_dimensions(self, output: Union[Tensor, Tuple[Tensor, ...]]) -> Union[Tensor, Tuple]:
        """ Return the output tensor(s) to the original dimensions for the multi-agent case, also handles multiple outputs in the form of a tuple. """            
        if isinstance(output, tuple):
            if any([output[i].dim() != 2 for i in range(len(output))]):
                raise ValueError(f'At least one input tensor has incorrect dimensions: {[output[i].size() for i in range(len(output))]}')
            
            output = tuple([output[i].view(-1, self.n_agents, output[i].size(-1)) for i in range(len(output))])
            
        elif isinstance(output, Tensor):
            if output.dim() == 2:
                    output = output.view(-1, self.n_agents, output.size(-1))
            else:
                raise ValueError(f'Input tensor has incorrect dimensions: {output.size()}')
        else:
            raise ValueError(f'Input tensor is not a Tensor or Tuple: {type(output)}')
        
        return output