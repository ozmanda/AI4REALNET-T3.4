from argparse import Namespace
import torch
from torch import Tensor
from typing import List, Tuple, Dict
from src.networks.Actor import Actor
from src.networks.Critic import Critic
from copy import deepcopy

class PPOController():
    """
    Controller class for the PPO algorithm. Interacts with the environment, choosing actions according to the current policy.
    """
    def __init__(self, args: Namespace, device: str) -> None:
        self.args = args
        self.device = device

        # Hyperparameters
        self.state_size: int = args.state_size
        self.action_size: int = args.action_size
        self.gamma: float = args.gamma
        self.entropy_coefficient: float = args.entropy_coefficient

        # Initialise Networks
        self.actor_network: Actor = Actor(args, n_features=self.state_size, n_actions=self.action_size).to(self.device)
        self.target_actor_network: Actor = deepcopy(self.actor_network).to(self.device)
        self.critic_network: Critic = Critic(args, n_features=self.state_size, layer_sizes=args.critic_hidden_layer_sizes).to(self.device)


    def _make_logits(self, states: Tensor, neighbour_states: Tensor) -> Tensor:
        """
        Create the logits for the action space based on the current state and neighbour states.

        Parameters:
            - states            Tensor      (batch_size, n_features)
            - neighbour_states  Tensor      (n_agents, batch_size, n_features)

        Returns:
            - logits            Tensor      (batch_size, n_actions)
        """
        encoded_states = self.actor_network.encode(states)

        neighbour_encoded_states: Tensor = torch.zeros(neighbour_states.size(0), neighbour_states.size(1), self.args.hidden_size, device=self.device)
        valid_neighbours: Tensor = torch.norm(neighbour_states, dim=2) > 1e-9 #? is this correct? Vector norm over dim 2?

        if torch.any(valid_neighbours): 
            neighbour_encoded_states = self.actor_network.encode(neighbour_states[valid_neighbours])
        
        neighbour_signals: Tensor = self.actor_network.intent(neighbour_encoded_states)
        logits: Tensor = self.actor_network.act(encoded_states, neighbour_signals)
        return logits


    def select_action(self, handles: List[int], state_dict: Dict[str, Tensor], neighbours_state_dict: Dict[str, Tensor]) -> Tuple[Dict, Dict]:
        """
        Select an action based on the current state and neighbour states.

        Parameters:
            - handles           List[int]  List of handles for the agents

        Returns:
            - actions           Dict[handle, int]     
            - log_probs         Dict[handle, Tensor]
        """
        states = torch.stack([state_dict[handle] for handle in handles])
        neighbour_states = torch.stack([torch.stack([neighbours_state_dict[handle]]) for handle in handles])

        with torch.no_grad(): 
            logits = self._make_logits(states, neighbour_states)
            actions = torch.argmax(logits, dim=1)
            log_probs = torch.log_softmax(logits, dim=1)
        
        return actions, log_probs


    def sample_action(self, handles: List[str], state_dict: Dict[str, Tensor], neighbours_state_dict: Dict[str, Tensor]) -> Tuple[Dict, Dict]:
        """
        Sample an action from the distribution provided by the Actor network based on the current and neighbour states

        Parameters: 
            - handles                   List[int]           List of handles for the angets
            - state_dict                Dict[int, Tensor]   Dictionary of states for each agents
            - neighbour_state_dict      Dict[int, Tensor]   Dictionary of states for each valid neighbour agent
        
        Returns: 
            - actions           Dict[handle, int]     
            - log_probs         Dict[handle, Tensor]
        """
        states = torch.stack([state_dict[handle] for handle in handles])
        neighbour_states = torch.stack([torch.stack([neighbours_state_dict[handle]]) for handle in handles])

        with torch.no_grad(): 
            logits = self._make_logits(states, neighbour_states)
            action_distribution = torch.distributions.Categorical(logits=logits)
            actions = action_distribution.sample()
            log_probs = action_distribution.log_prob(actions)
        
        return actions, log_probs


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