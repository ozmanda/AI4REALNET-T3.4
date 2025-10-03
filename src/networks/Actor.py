'''
Adapted from the JBR_HSE Solution to the Flatland 2020 Challenge at NeurIPS
Original Code: https://github.com/jbr-ai-labs/NeurIPS2020-Flatland-Competition-Solution
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List, Dict
from argparse import Namespace

from src.networks.Attention import MultiHeadAttention
from src.networks.Recursive import RecursiveLayer

from src.utils.observation.obs_utils import direction_tensor


class Actor(nn.Module):  
    def __init__(self, config_dict: Dict, n_features: int, n_actions: int): # TODO: change config_dict to DefaultDict, add recursive layer hidden size parameter
        super().__init__()
        self.n_actions = n_actions # TODO: check that this comes from env.
        self.n_features = n_features
        self.input_size = n_features * config_dict['n_nodes']
        self.hid_size: int = config_dict['hidden_size']
        self.layer_sizes: List[int] = config_dict['layer_sizes']
        self.intent_size: int = config_dict['intent_size']
        self.neighbour_depth: int = config_dict['neighbour_depth']
        
        self.state_encoder = nn.Sequential(RecursiveLayer(n_features=n_features, n_nodes=config_dict['n_nodes'], output_size=self.hid_size))
        self.policy_hidden_layer = nn.Linear(self.hid_size, self.intent_size)
        self.intent_encoding = nn.Sequential(nn.Linear(self.hid_size, self.intent_size, bias=False),
                                                  nn.ReLU(inplace=True))
        self.intent_attention = MultiHeadAttention(hid_size=self.hid_size, intent_size=self.intent_size, num_heads=config_dict['n_heads'])
        self.action_decoder = nn.Sequential(nn.Linear((self.intent_size * config_dict['n_heads']) + self.hid_size, 256),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(256, self.n_actions))

        self.direction_tensor: Tensor = direction_tensor(self.neighbour_depth)
        
    
    def encode(self, states: Tensor) -> Tensor:
        return self.state_encoder(states)


    def intent(self, encoded_state: Tensor) -> Tensor:
        return self.intent_encoding(encoded_state)


    def act(self, encoded_states: Tensor, agent_signals: Tensor) -> Tensor:
        """
        Outputs the log_probabilities of the action options

        Parameters: 
            - encoded_state     Tensor      (batch_size, hidden_size)
            - agent_signals     Tensor      (batch_size, n_agents, intention_size)
        """
        # resize to 

        query: Tensor = self.policy_hidden_layer(encoded_states)
        if query.dim() == 2:  
            query = query.unsqueeze(1)
        # key: Tensor = self._add_direction(agent_signals) #! removed direction information
        key = agent_signals.view(agent_signals.size(0), agent_signals.size(1), -1)
        attended_signals: Tensor = self.intent_attention(query=query, key=key, value=key)
        input = torch.cat([encoded_states, attended_signals], dim=1)
        return self.action_decoder(input)


    def _add_direction(self, agent_signals: Tensor) -> Tensor:
        """
        Appends the direction tensor at the current neighbour depth to the signal tensor, adding two columns at dimension 2.
        """
        self.direction_tensor = self.direction_tensor.unsqueeze(0)
        self.direction_tensor = self.direction_tensor.expand(agent_signals.shape[0], -1, -1)
        return torch.cat([agent_signals, self.direction_tensor])
        