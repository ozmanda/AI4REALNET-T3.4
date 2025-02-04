'''
Adapted from IC3Net trainer.py
Two subclasses of Trainer from Trainer.py specifically for LSTM and RNN policies
'''
from Trainer import Trainer
from torch import optim, Tensor
import torch.nn as nn
from typing import Tuple, List
import torch

class LSTMTrainer(Trainer): 
    def __init__(self, args, policy_net, env) -> None:
        super().__init__(args, policy_net, env)
    
    def get_episode(self) -> Tuple:
        episode = []
        observations = self.env.reset()
        state = dict()
        info = dict()

        # declarations
        action: List[Tensor]
        value: Tensor
        prev_hidden_state: Tensor; prev_cell_state: Tensor

        # initialisations
        info['comm_action'] = torch.zeros(self.n_agents, dytpe=int)
        prev_hidden_state, prev_cell_state = self.policy_net.init_hidden(self.n_agents)

        step = 0
        while True: 

            step += 1
            if step >= self.args.max_steps:
                break



class RNNTrainer(Trainer): 
    def __init__(self, args, policy_net, env) -> None:
        super().__init__(args, policy_net, env)
    
    def get_episode(self) -> Tuple:
        episode = []
        observations = self.env.reset()
        state = dict()
        info = dict()

        # declarations
        action: List[Tensor]
        value: Tensor
        prev_hidden_state: Tensor