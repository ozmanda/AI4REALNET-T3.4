from torch import Tensor
from typing import Any, Dict, List, Tuple, Iterable

class Controller(): 
    """ Base class for multi-agent RL Controllers. For use with a single agent, the input sizes should be adjusted accordingly."""
    def __init__(self):
        pass

    def _make_logits(self, states: Tensor) -> Tensor:
        """
        Create logits for the action space based on the current state.
        
        Parameters:
            - states: Tensor (n_agents, batch_size, n_features)
        
        Returns:
            - logits: Tensor (batch_size, n_actions)
        """
        pass

    def select_action(self, states: Tensor, ) -> Tensor:
        """
        Select the best action based on the current state.

        Parameters:
            - states: Tensor (n_agents, batch_size, n_features)

        Returns:
            - actions: Tensor (n_agents, batch_size)
        """
        pass

    def sample_action(self, states: Tensor) -> Tensor:
        """
        Sample an action from the distribution provided by the network based on the current state.

        Parameters:
            - states: Tensor (n_agents, batch_size, n_features)

        Returns:
            - actions: Tensor (n_agents, batch_size)
            - log_probs: Tensor (n_agents, batch_size)
        """
        pass

    def load_controller(self, path: str) -> None:
        """
        Load the controller from a given path.

        Parameters:
            - path: str, the path to the controller file
        """
        pass

    def save_controller(self, path: str) -> None:
        """
        Save the controller to a given path.

        Parameters:
            - path: str, the path to the controller file
        """
        pass

    def get_parameters(self) -> Iterable:
        """
        Get the parameters of the controller that can be optimised.

        Returns:
            - parameters: Dict, the parameters of the controller
        """
        pass