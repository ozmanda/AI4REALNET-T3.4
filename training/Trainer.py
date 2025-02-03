'''
From IC3Net trainer.py 
'''

import numpy as np
import torch
from torch import optim, Tensor
import torch.nn as nn
from typing import Tuple

class Trainer():
    def __init__(self, args, policy_net, env): 
        self.args = args
        self.policy_net = policy_net
        self.env = env

        self.optimizer = optim.rmsprop.RMSprop(policy_net.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)
        self.params = [p for p in self.policy_net.parameters()]

    
    def get_episode(self, epoch):
        episode = []
        obs = self.env.reset()

        state = dict()
        info = dict()

        # initialisations
        if self.args.commnet:
            info['comm_action'] = torch.zeros(self.n_agents, dytpe=int)
        if self.args.rnn_type == 'LSTM':
            prev_hid: Tuple[Tensor, Tensor] = self.policy_net.init_hidden(self.n_agents)
        else:
            prev_hid: Tensor = torch.zeros(1, self.n_agents, self.args.hid_size)

        for t in range(self.args.max_steps):
            if self.recurrent: 
                #! the original code combines tuples and tensors, keep them separated here for clarity
                x = [state, prev_hid]
                action_out, value, prev_hid = self.policy_net(x, info) #! this isn't adapted yet. 