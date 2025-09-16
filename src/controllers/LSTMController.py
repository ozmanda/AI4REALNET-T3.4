import wandb
import numpy as np
from itertools import chain
from typing import Dict, Union, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from flatland.envs.rail_env import RailEnv

from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer
from src.networks.LSTM import LSTM


class LSTMController(nn.Module):
    """
    Basic Controller for Proximal Policy Optimization (PPO) algorithm.
    Implements simple Feed Forward NNs for the actor and critic networks. 
    """
    def __init__(self, config: Dict):
        super(LSTMController, self).__init__()
        self.config: Dict = config
        self._init_hyperparameters(config)

        self._build_lstm_network()
        self.lstm_network.reset_network(n_agents=self.config['n_agents'])
        self.update_step: int = 0

    def _init_hyperparameters(self, config: Dict) -> None:
        """
        Initialize hyperparameters from the configuration dictionary.
        """
        self.action_size: int = config['action_size']
        self.state_size: int = config['state_size']
        self.gamma: float = config['gamma']
        self.lam: float = config['lam']
        self.gae_horizon: int = config['gae_horizon']
        self.clip_epsilon: float = config['clip_epsilon']
        self.value_loss_coef: float = config['value_loss_coefficient']
        self.entropy_coef: float = config['entropy_coefficient']

    def _build_lstm_network(self) -> None:
        """
        Build the LSTM network architecture, including feature extraction, LSTM unit, actor and critic heads.
        """
        self.lstm_network = LSTM(self.config)

    def init_wandb(self) -> None:
        wandb.watch(self.lstm_network, log='all')

    def update_weights(self, network_params: Dict) -> None:
        """
        Update the weights of the actor and critic networks.

        Parameters:
            - network_params: Tuple containing the actor and critic network parameters
        """
        self.lstm_network.update_weights(network_params)
        print("LSTMController: Weights updated.")

    def get_parameters(self):
        return self.lstm_network.get_parameters()

    def get_state_dict(self) -> Dict:
        return self.lstm_network.get_state_dict()
    
    def update_weights(self, network_params: Dict) -> None:
        self.lstm_network.update_weights(network_params)
    

    def state_values(self, states: Tensor) -> Tensor:
        """
        Get the state values from the critic network for the current and next states.
        
        Parameters:
            - states: Tensor of shape (batch_size, state_size)
            - next_states: Tensor of shape (batch_size, state_size)
        
        Returns:
            - state_values: Tensor of shape (batch_size, 1)
            - next_state_values: Tensor of shape (batch_size, 1)
        """
        state_values = self.lstm_network.state_values(states)
        return state_values
        

    def sample_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get the actions from the actor network based on the current state.
        
        Parameters:
            - state: Tensor of shape (n_agents, state_size)
        
        Returns:
            - actions: Tensor of shape (n_agents, 1)
            - log_probs: Tensor of shape (n_agents, 1)
        """
        actions, log_probs, values, _ = self.lstm_network(state)
        return actions, log_probs, values


    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select the best action based on the current state using the actor network.
        
        Parameters:
            - state: Tensor of shape (batch_size, state_size)
        
        Returns:
            - action: Tensor of shape (batch_size, 1)
            - log_prob: Tensor of shape (batch_size, 1)
        """
        actions, log_probs, values, hidden_states = self.lstm_network(state, select_best_action=True)
        return actions, log_probs
    
    