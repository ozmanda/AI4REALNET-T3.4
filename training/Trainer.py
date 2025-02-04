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

class Trainer():
    def __init__(self, args: Namespace, policy_net, env): 
        self.args: Namespace = args
        self.policy_net = policy_net
        self.env = env

        self.stats: dict = dict()

        self.optimizer = optim.rmsprop.RMSprop(policy_net.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)
        self.params = [p for p in self.policy_net.parameters()]

    
    def get_episode(self, epoch):
        # TODO: keep only base case intialisation in this function
        episode = []
        observations = self.env.reset()
        state = dict()

        action: List[Tensor]
        value: Tensor
        prev_hidden_state: Tensor = torch.zeros(1, self.args.n_agents, self.args.hid_size)

        while True: 
            action_probs, value = self.policy_net(observations)
            action = sample_action(action_probs)
            action = [x.squeeze().data.numpy() for x in action] 
            # translate into env action
            # action = translate_action(self.args, self.env, action)


            step += 1
            if step >= self.args.max_steps:
                break


    
    def get_batch(self) -> Tuple[List[Tensor], dict]:
        batch: List[Tensor] = []
        self.stats['num_episodes'] = 0
        
        while len(batch) < self.args.batch_size:
            pass
                       