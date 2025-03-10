'''
Adapted from IC3Net trainer.py to be a base trainer class, and trainer for MLP. The original code considers all network types during training. 
'''

from argparse import Namespace
import torch
from torch import optim, Tensor
from typing import Tuple, List, Dict
from collections import namedtuple

from src.utils.action_utils import sample_action
from src.networks.MLP import MLP
from flatland.envs.rail_env import RailEnv


Transition = namedtuple('Transition', ('state', 'action', 'action_log_prob', 'hidden_states', 'cell_states', 'value', 'reward', 'done'))

class Trainer():
    def __init__(self, args: Namespace, policy_net: MLP, env: RailEnv): 
        self.args: Namespace = args
        self.policy_net: MLP = policy_net
        self.env: RailEnv = env
        if self.obs_type == 'tree': 
            self.max_tree_depth = args.max_tree_depth

        # TODO: add dict typing
        self.stats: Dict = dict()

        self.optimizer = optim.rmsprop.RMSprop(policy_net.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)
        self.params = [p for p in self.policy_net.parameters()]


    def get_episode(self, epoch):
        episode: List = []
        observations: Tuple[Dict, Dict] = self.env.reset()
        # TODO: add dict typing
        state: Dict = dict()

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


    # TODO: add dict typing
    def get_batch(self) -> Tuple[List[Tensor], Dict]:
        batch: List[Tensor] = []
        self.stats['num_episodes'] = 0
        
        while len(batch) < self.args.batch_size:
            pass
                       