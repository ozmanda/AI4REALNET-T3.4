'''
From IC3Net models.py, continuous case removed (flatland has discrete action space)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from MLP import MLP

class RNN(MLP):
    def __init__(self, args, num_inputs):
        super().__init__(args, num_inputs)
        self.n_agents = self.args.n_agents
        self.hid_size = self.args.hid_size
        if self.args.rnn_type == 'LSTM':
            del self.fc2
            self.lstm_unit = nn.LSTMCell(self.hid_size, self.hid_size)

    def forward(self, x):
        x, prev_hid = x
        encoded_x: Tensor = self.fc1(x)

        if self.args.rnn_type == 'LSTM':
            batch_size: int = encoded_x.size(0)
            encoded_x = encoded_x.view(batch_size * self.n_agents, self.hid_size)
            output = self.lstm_unit(encoded_x, prev_hid)
            next_hid: Tensor = output[0]
            ret = (next_hid.clone(), output[1].clone())
            next_hid = next_hid.view(batch_size, self.n_agents, self.hid_size)
        
        else:
            next_hid = F.tanh(self.fc2(prev_hid) + encoded_x)
            ret = next_hid

        v = self.value_head(next_hid)
        action_mean = self.action_mean(next_hid)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return (action_mean, action_log_std, action_std), v, ret

    def init_hidden(self, batch_size):
        'dim 0 = num of layers * num of direction'
        return tuple((torch.zeros(batch_size * self.n_agents, self.hid_size, requires_grad=True),
                      torch.zeros(batch_size * self.n_agents, self.hid_size, requires_grad=True)))
