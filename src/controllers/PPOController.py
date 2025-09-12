import wandb
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, List, Tuple

from flatland.envs.rail_env import RailEnv

from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer
from src.networks.FeedForwardNN import FeedForwardNN


class PPOController(nn.Module):
    """
    Basic Controller for Proximal Policy Optimization (PPO) algorithm.
    Implements simple Feed Forward NNs for the actor and critic networks. 
    """
    def __init__(self, config: Dict, agent_ID: Union[int, str] = None):
        super(PPOController, self).__init__()
        self.config: Dict = config
        if agent_ID:
            self.agent_ID: Union[int, str] = agent_ID
        self._init_hyperparameters(config)

        self._build_actor()
        self._build_critic()
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
        # self.max_grad_norm: float = config['max grad norm']

    def _build_actor(self) -> nn.Module:
        # TODO: make this modular
        self.actor_network = FeedForwardNN(self.state_size, self.config['actor_config']['hidden_size'], self.action_size)

    def _build_critic(self) -> nn.Module:
        # TODO: make this modular
        self.critic_network = FeedForwardNN(self.state_size, self.config['critic_config']['hidden_size'], 1) 

    def _make_logits(self, states: Tensor) -> Tensor:
        """
        Create logits for the action space based on the current state.
        
        Parameters:
            - states: Tensor (batch_size, n_features)
        
        Returns:
            - logits: Tensor (batch_size, n_actions)
        """
        return self.actor_network(states)
    

    def update_weights(self, network_params: Tuple[Dict, Dict]) -> None:
        """
        Update the weights of the actor and critic networks.

        Parameters:
            - network_params: Tuple containing the actor and critic network parameters
        """
        actor_params, critic_params = network_params
        self.update_actor(actor_params)
        self.update_critic(critic_params)

    def update_actor(self, network_params: Dict) -> None:
        """ Update the actor network with the given parameters. """
        self.old_actor_params = self.actor_network.state_dict()
        self.new_actor_params = network_params
        self.actor_network.load_state_dict(self.new_actor_params)

    def update_critic(self, network_params: Dict) -> None:
        """ Update the critic network with the given parameters. """
        self.old_critic_params = self.critic_network.state_dict()
        self.new_critic_params = network_params
        self.critic_network.load_state_dict(self.new_critic_params)

    def get_network_params(self) -> Tuple[Dict, Dict]:
        """
        Get the current parameters of the actor and critic networks.
        
        Returns:
            - actor_params: Dict containing the actor network parameters
            - critic_params: Dict containing the critic network parameters
        """
        actor_params = self.actor_network.state_dict()
        critic_params = self.critic_network.state_dict()
        return actor_params, critic_params
    

    def state_values(self, states: Tensor, next_states: Tensor) -> Tensor:
        """
        Get the state values from the critic network for the current and next states.
        
        Parameters:
            - states: Tensor of shape (batch_size, state_size)
            - next_states: Tensor of shape (batch_size, state_size)
        
        Returns:
            - state_values: Tensor of shape (batch_size, 1)
            - next_state_values: Tensor of shape (batch_size, 1)
        """
        state_values = self.critic_network(states)
        next_state_values = self.critic_network(next_states)
        return state_values, next_state_values
        

    def sample_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get the action from the actor network based on the current state.
        
        Parameters:
            - state: Tensor of shape (batch_size, state_size)
        
        Returns:
            - action: Tensor of shape (batch_size, 1)
            - log_prob: Tensor of shape (batch_size, 1)
        """
        logits = self._make_logits(state)
        action_distribution = torch.distributions.Categorical(logits=logits)
        actions = action_distribution.sample()
        log_prob = action_distribution.log_prob(actions)
        return actions, log_prob
    

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select the best action based on the current state using the actor network.
        
        Parameters:
            - state: Tensor of shape (batch_size, state_size)
        
        Returns:
            - action: Tensor of shape (batch_size, 1)
            - log_prob: Tensor of shape (batch_size, 1)
        """
        with torch.no_grad():
            logits = self._make_logits(state)
            actions = torch.argmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=1)
        return actions, log_probs
    

    def _discounted_rewards(self, rewards: Tensor, dones: Tensor, step: int) -> Tensor:
        """ Compute discounted rewards. """
        discounted_rewards = torch.zeros_like(rewards)
        cumulative_reward = 0.0
        for i in reversed(range(len(rewards))):
            cumulative_reward = rewards[i] + (self.gamma * cumulative_reward * (1 - dones[i]))
            discounted_rewards[i] = cumulative_reward
        return discounted_rewards
    


