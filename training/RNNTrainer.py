'''
Adapted from IC3Net trainer.py
Two subclasses of Trainer from Trainer.py specifically for LSTM and RNN policies
'''
from Trainer import Trainer
from torch import optim, Tensor
import torch.nn as nn
from typing import Tuple, List, Union
import torch
from networks.RNN import RNN, LSTM
from flatland.envs.rail_env import RailEnv
from argparse import Namespace
from utils.action_utils import sample_action, action_tensor_to_dict
from utils.obs import obs_dict_to_tensor
from flatland.envs.rail_env import RailEnv
from Trainer import Transition

class RNNTrainer(): 
    """ Trainer class for the no-communication recurrent policy networks, generalised to both the RNN and LSTM versions """

    def __init__(self, args: Namespace, policy_net: Union[RNN, LSTM], env: RailEnv) -> None:
        self.args: Namespace = args
        self.policy_net: Union[RNN, LSTM] = policy_net
        self.env: RailEnv = env
        self.n_agents: int = self.args.n_agents
        self.agent_ids: range = env.get_agent_handles()
        self.lstm = True if args.rnn_type == 'lstm' else False

        self.observation_type: str = args.observation_type
        if self.observation_type == 'tree':
            self.max_tree_depth: int = args.max_tree_depth
            self.tree_nodes = int((4 ** (self.max_tree_depth + 1) - 1) / 3) #* geometric progression
            self.obs_features: int = 12

        self.stats: dict = dict()

        self.optimizer = optim.rmsprop.RMSprop(policy_net.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)
        self.params = [p for p in self.policy_net.parameters()]

    
    def get_episode(self) -> List[Transition]:
        """
        Performs one episode in the environment and gathers the transitions
    
        Return: List[Transitions]
            - state     dict[int: dict]
            - action    dict[int: int]
            - value     dict[int: float]
            - reward    dict[int: float]
            - done      dict[int: bool]
        """
        episode: List = []
        output: Tuple[dict, dict] = self.env.reset()
        obs_dict, info_dict = output
        obs_tensor: Tensor = obs_dict_to_tensor(obs_dict, obs_type=self.args.observation_type, )
        stats: dict = dict()

        # declarations
        action: List[Tensor]
        value: Tensor

        # initialisations
        if self.lstm:
            prev_hidden_state, prev_cell_state = self.policy_net.init_hidden(self.n_agents)
        else: 
            prev_hidden_state = self.policy_net.init_hidden(self.n_agents)

        step = 0
        while True: 
            # TODO: check dimension of action_probs
            if self.lstm:
                action_probs, value, next_hidden_state, next_cell_state = self.policy_net(obs_tensor, prev_hidden_state, prev_cell_state)
            else: 
                action_probs, value, next_hidden_state = self.policy_net(obs_tensor, prev_hidden_state)

            if (step + 1) % self.args.detach_gap == 0:
                next_hidden_state = next_hidden_state.detach()
                if self.lstm:
                    next_cell_state = next_cell_state.detach()
            
            # TODO: check dimension of sampled action
            actions_tensor: Tensor = sample_action(action_probs)
            actions_dict: dict = action_tensor_to_dict(actions_tensor, self.agent_ids)
            # TODO: check what datatypes are returned
            next_obs_dict, reward, done_dict, info = self.env.step(actions_dict)

            # create alive mask for agents that are already done to mask reward
            done_mask = [done_dict[agent_id] for agent_id in self.agent_ids]

            transition = Transition(obs_dict, actions_dict, value, reward, done_dict)
            episode.append(transition)
            obs_dict = next_obs_dict

            step += 1
            if step >= self.args.max_steps or all(done_mask):
                break
        
        return episode 
