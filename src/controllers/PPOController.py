import wandb
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, List, Tuple
from itertools import chain

from flatland.envs.rail_env import RailEnv

from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer
from src.networks.FeedForwardNN import FeedForwardNN


class PPOController(nn.Module):
    """
    Basic Controller for Proximal Policy Optimization (PPO) algorithm.
    Implements simple Feed Forward NNs for the actor and critic networks. 
    """
    def __init__(self, config: Dict, agent_ID: Union[int, str] = ''):
        super(PPOController, self).__init__()
        self.config: Dict = config
        if agent_ID:
            self.agent_ID: Union[int, str] = agent_ID
        self._init_hyperparameters(config)

        self._build_encoder()
        self._build_actor()
        self._build_critic()
        self.update_step: int = 0

    def _init_hyperparameters(self, config: Dict) -> None:
        """
        Initialize hyperparameters from the configuration dictionary.
        """
        self.action_size: int = config['action_size']
        self.state_size: int = config['state_size']

    def _build_encoder(self) -> None:
        self.encoded_state_size = self.config['encoder']['output_size']
        self.encoder_network = FeedForwardNN(self.state_size, self.encoded_state_size, self.config['encoder'])

    def _build_actor(self) -> None:
        self.actor_network = FeedForwardNN(self.encoded_state_size, self.action_size, self.config['actor_config'])

    def _build_critic(self) -> None:
        self.critic_network = FeedForwardNN(self.encoded_state_size, 1, self.config['critic_config'])

    def init_wandb(self) -> None:
        wandb.watch(self.actor_network, log='all')
        wandb.watch(self.critic_network, log='all')

    def _make_logits(self, encoded_states: Tensor) -> Tensor:
        """
        Create logits for the action space based on the current state.
        
        Parameters:
            - encoded_states: Tensor (batch_size, n_features)
        
        Returns:
            - logits: Tensor (batch_size, n_actions)
        """
        return self.actor_network(encoded_states)
    
    def get_parameters(self):
        return chain(self.actor_network.parameters(), self.critic_network.parameters(), self.encoder_network.parameters())

    def update_weights(self, network_params: Tuple[Dict, Dict]) -> None:
        """
        Update the weights of the actor and critic networks.

        Parameters:
            - network_params: Tuple containing the actor and critic network parameters
        """
        encoder_params, actor_params, critic_params = network_params
        self.update_encoder(encoder_params)
        self.update_actor(actor_params)
        self.update_critic(critic_params)

    def update_encoder(self, network_params: Dict) -> None:
        """ Update the feature extraction network with the given parameters. """
        self.old_encoder_params = self.encoder_network.state_dict()
        self.new_encoder_params = network_params
        self.encoder_network.load_state_dict(network_params)

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

    def get_network_params(self) -> Tuple[Dict, Dict, Dict]:
        """
        Get the current parameters of the actor, critic, and encoder networks.

        Returns:
            - actor_params: Dict containing the actor network parameters
            - critic_params: Dict containing the critic network parameters
        """
        actor_params = self.actor_network.state_dict()
        critic_params = self.critic_network.state_dict()
        encoder_params = self.encoder_network.state_dict()
        return encoder_params, actor_params, critic_params

    def state_values(self, states: Tensor, extras: Dict[str, Tensor]) -> Tensor:
        """
        Get the state values from the critic network for the current and next states.
        
        Parameters:
            - states: Tensor of shape (batch_size, state_size)
            - next_states: Tensor of shape (batch_size, state_size)
        
        Returns:
            - state_values: Tensor of shape (batch_size, 1)
        """
        encoded_states = self.encoder_network(states)
        return self.critic_network(encoded_states)


    def sample_action(self, states: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor, None]:
        """
        Get the action from the actor network based on the current state.
        
        Parameters:
            - state: Tensor of shape (batch_size, state_size)
        
        Returns:
            - action: Tensor of shape (batch_size, 1)
            - log_prob: Tensor of shape (batch_size, 1)
            - value: Tensor of shape (batch_size, 1)
        """
        encoded_states = self.encoder_network(states)
        logits = self._make_logits(encoded_states)
        action_distribution = torch.distributions.Categorical(logits=logits)
        actions = action_distribution.sample()
        log_prob = action_distribution.log_prob(actions)
        values = self.critic_network(encoded_states)
        return actions, log_prob, values, None # extras = None (compatibility with other controllers)
    

    def select_action(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Select the best action based on the current state using the actor network.
        
        Parameters:
            - state: Tensor of shape (batch_size, state_size)
        
        Returns:
            - action: Tensor of shape (batch_size, 1)
            - log_prob: Tensor of shape (batch_size, 1)
        """
        # TODO: change this to match sample_action function
        with torch.no_grad():
            encoded_state = self.encoder_network(state)
            logits = self._make_logits(encoded_state)
            actions = torch.argmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=1)
        return actions, log_probs
    

    def evaluate_action(self, states: Tensor, actions: Tensor, extras: Dict) -> Tensor:
        """
        Computes the log-probabilities of the given actions under the current policy.
        """
        encoded_states = self.encoder_network(states)
        logits = self._make_logits(encoded_states)
        action_distribution = torch.distributions.Categorical(logits=logits)
        return action_distribution.log_prob(actions)

    def get_state_dict(self) -> Dict:
        """
        Get the state dictionary of the PPO controller.
        """
        state_dict = {
            'actor_network': self.actor_network.state_dict(),
            'critic_network': self.critic_network.state_dict(),
            'encoder_network': self.encoder_network.state_dict()
        }
        return state_dict
    

    def load_model(self, model_path: str) -> None:
        """
        Load the model from the specified path.
        """
        self.actor_network.load_state_dict(torch.load(f'{model_path}/actor.pth', map_location=torch.device('cpu')))
        self.critic_network.load_state_dict(torch.load(f'{model_path}/critic.pth', map_location=torch.device('cpu')))
        self.encoder_network.load_state_dict(torch.load(f'{model_path}/encoder.pth', map_location=torch.device('cpu')))
