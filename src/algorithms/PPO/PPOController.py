from argparse import Namespace
import torch
from torch import Tensor
from typing import List, Tuple, Dict

class PPOController():
    """
    Controller class for the PPO algorithm. Interacts with the environment, choosing actions according to the current policy.
    """
    def __init__(args: Namespace, device: str) -> None:
        pass


    def _make_logits(self, states: Tensor, neighbour_states: Tensor) -> Tensor:
        """
        Create the logits for the action space based on the current state and neighbour states.

        Parameters:
            - states            Tensor      (batch_size, n_features)
            - neighbour_states  Tensor      (batch_size, n_features)

        Returns:
            - logits            Tensor      (batch_size, n_actions)
        """
        pass


    def select_action(self, handles: List[str], state_dict: Dict[str, Tensor], neighbours_state_dict: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Select an action based on the current state and neighbour states.

        Parameters:
            - handles           List[str]  List of handles for the agents

        Returns:
            - actions           Tensor     (batch_size, n_actions)
            - log_probs         Tensor     (batch_size, n_actions)
        """
        pass


    def update_networks(self, network_parameters: Tuple) -> None:
        """
        Update the networks with the given parameters.

        Parameters:
            - network_parameters  Tuple     Tuple of network parameters
        """
        pass


    def get_network_parameters(self) -> Tuple:
        """
        Get the network parameters.

        Returns:
            - network_parameters  Tuple     Tuple of network parameters
        """
        pass


    def actor_hard_update(self) -> None:
        """
        Perform a hard update of the networks.
        """
        pass


    def actor_soft_update(self) -> None:
        """
        Perform a soft update of the networks.
        """
        pass


    def load_controller(self, path: str) -> None:
        """
        Load the controller from the given path.

        Parameters:
            - path      str       Path to the controller file
        """
        pass


    def save_controller(self, path: str) -> None:
        """
        Save the controller to the given path.

        Parameters:
            - path      str       Path to save the controller file
        """
        pass