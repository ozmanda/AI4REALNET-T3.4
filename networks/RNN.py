'''
From IC3Net models.py, continuous case removed (flatland has discrete action space)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from MLP import MLP
from typing import Tuple

class RNN(MLP):
    def __init__(self, args, num_inputs):
        super().__init__(args, num_inputs)
        self.n_agents: int = self.args.n_agents
        self.hid_size = self.args.hid_size

    def forward(self, observations, prev_hidden_state) -> Tuple[Tensor, Tensor, Tensor]:
        encoded_x: Tensor = self.fc1(observations)
        next_hidden_state: Tensor = F.tanh(self.fc2(prev_hidden_state) + encoded_x)

        v = self.value_head(next_hidden_state)

        return F.log_softmax(self.action_head(next_hidden_state), dim=-1), v, next_hidden_state


    def init_hidden(self, batch_size: int) -> Tensor:
        # TODO: write description
        'dim 0 = num of layers * num of direction'
        return torch.zeros(batch_size * self.n_agents, self.hid_size, requires_grad=True)
    

class LSTM(RNN):
    def __init__(self, args, num_inputs):
        super().__init__(args, num_inputs)
        self.n_agents: int = self.args.n_agents
        self.hid_size: int = self.args.hid_size
        del self.fc2
        self.lstm_unit = nn.LSTMCell(self.hid_size, self.hid_size)

    def forward(self, observations: Tensor, prev_hidden_state: Tensor, prev_cell_state: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # TODO: write description 
        # TODO: add array sizing in comments
        batch_size: int = observations.size(0)

        encoded_x: Tensor = self.fc1(observations)
        encoded_x = encoded_x.view(batch_size * self.n_agents, self.hid_size)

        next_hidden_state, next_cell_state = self.lstm_unit(encoded_x, (prev_hidden_state, prev_cell_state))

        # TODO: check that this can be replaced with next_hidden_state.clone(), next_cell_state.clone()
        # ret = (next_hidden_state.clone(), next_cell_state.clone())
        next_hid = next_hid.view(batch_size, self.n_agents, self.hid_size)

        v = self.value_head(next_hidden_state)

        return F.log_softmax(self.action_head(next_hidden_state), dim=-1), v, next_hidden_state.clone(), next_cell_state.clone()
    

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        # TODO: write description
        'dim 0 = num of layers * num of direction'
        return tuple((torch.zeros(batch_size * self.n_agents, self.hid_size, requires_grad=True),
                      torch.zeros(batch_size * self.n_agents, self.hid_size, requires_grad=True)))