from typing import Dict
from torch import Tensor
import torch.nn as nn

class Controller:
    """ Base class for multi-agent RL Controllers. For use with a single agent, the input sizes should be adjusted accordingly."""
    def __init__(self):
        self.actor_network: nn.Module
        self.critic_network: nn.Module

    def update_weights(self, network_params: Dict) -> None:
        """
        Update the weights of the networks in the controller.
        """
        pass

    def get_state_dict(self) -> Dict:
        """
        Get the state dictionary of the networks in the controller.
        """
        pass

    def sample_action(self, state: Tensor) -> Tensor:
        """
        Sample an action from the distribution provided by the network based on the current state.

        Parameters:
            - states: Tensor    (n_agents, batch_size, n_features)

        Returns:
            - actions: Tensor   (n_agents, batch_size)
            - log_probs: Tensor (n_agents, batch_size)
        """
        pass

    def state_values(self, states: Tensor) -> Tensor:
        """
        Outputs the state values from the critic network for the given state.

        Parameters:
            - states: Tensor    (n_agents, batch_size, n_features)

        Returns:
            - values: Tensor    (n_agents, 1)
        """
        pass

