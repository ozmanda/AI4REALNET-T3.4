'''
Adapted from IC3Net trainer.py to be a base trainer class, and trainer for MLP. The original code considers all network types during training. 
'''

import numpy as np
import torch
from torch import optim, Tensor
import torch.nn as nn
from typing import Tuple, List
from argparse import Namespace
from utils.action_utils import sample_action
from networks.MLP import MLP
from flatland.envs.rail_env import RailEnv
from collections import namedtuple
from dataclasses import dataclass

Transition = namedtuple('Transition', ('state', 'action', 'value', 'reward', 'done'))

@dataclass
class Transition:
    state: dict
    action: dict
    value: float
    reward: dict
    done: dict


class Trainer():
    def __init__(self, args: Namespace, policy_net: MLP, env: RailEnv): 
        self.args: Namespace = args
        self.policy_net: MLP = policy_net
        self.env: RailEnv = env
        if self.obs_type == 'tree': 
            self.max_tree_depth = args.max_tree_depth

        self.stats: dict = dict()

        self.optimizer = optim.rmsprop.RMSprop(policy_net.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)
        self.params = [p for p in self.policy_net.parameters()]


    def get_episode(self, epoch):
        episode: List = []
        observations: Tuple[dict, dict] = self.env.reset()
        state: dict = dict()

        action: List[Tensor]
        value: Tensor
        prev_hidden_state: Tensor = torch.zeros(1, self.args.n_agents, self.args.hid_size)

        step = 0
        while True: 
            action_probs, value = self.policy_net(observations)
            action: Tensor = sample_action(action_probs)
            next_state, reward, done, info = self.env.step(action)

            step += 1
            if step >= self.args.max_steps:
                break


    
    def get_batch(self) -> Tuple[List[Tensor], dict]:
        batch: List[Tensor] = []
        self.stats['num_episodes'] = 0
        
        while len(batch) < self.args.batch_size:
            pass
                       