import os
import torch
from torch import Tensor
from copy import deepcopy
from typing import List, Tuple, Dict
from src.networks.Actor import Actor
from src.networks.Critic import Critic

class PPOController():
    """
    Controller class for the PPO algorithm. Interacts with the environment, choosing actions according to the current policy.
    """
    def __init__(self, config_dict: Dict, device: str) -> None:
        self.device = device
        self.config: Dict = config_dict
        self.state_size = config_dict['state_size']
        self.action_size = config_dict['action_size'] # TODO: actions need to be taken from environment and not from actor config
        self.neighbour_depth = config_dict['neighbour_depth']
        self.hidden_size = config_dict['actor_config']['hidden_size']
        self.batch_size = config_dict['batch_size']
        self.gae_horizon = config_dict['gae_horizon']
        self.n_epochs_update = config_dict['n_epochs_update']
        self.gamma = config_dict['gamma']
        self.lam = config_dict['lam']
        self.clip_epsilon = config_dict['clip_epsilon']
        self.value_loss_coefficient = config_dict['value_loss_coefficient']
        self.entropy_coefficient = config_dict['entropy_coefficient']
        self.tau = config_dict['tau']
        # TODO: cleanup config

        # Initialise Networks
        self.actor_network: Actor = Actor(config_dict['actor_config'], n_features=config_dict['n_features'], n_actions=self.action_size).to(self.device)
        self.target_actor_network: Actor = deepcopy(self.actor_network).to(self.device)
        self.critic_network: Critic = Critic(n_features=config_dict['n_features'], config=config_dict['critic_config']).to(self.device)


    def _make_logits(self, states: Tensor, neighbour_states: Tensor) -> Tensor:
        """
        Create the logits for the action space based on the current state and neighbour states.

        Parameters:
            - states            Tensor      (n_agents, batch_size, n_features)
            - neighbour_states  Tensor      (n_agents, batch_size, n_features)

        Returns:
            - logits            Tensor      (batch_size, n_actions)
        """
        n_agents = states.size(0)
        encoded_states = self.actor_network.encode(states)

        neighbour_encoded_states: Tensor = torch.zeros(neighbour_states.size(0), neighbour_states.size(1), self.hidden_size, device=self.device) #? necessary?
        valid_neighbours: Tensor = torch.linalg.matrix_norm(neighbour_states) > -1 #? is this correct?

        if torch.any(valid_neighbours): 
            neighbour_encoded_states = self.actor_network.encode(neighbour_states[valid_neighbours]).view(n_agents, -1, self.hidden_size)
        
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
        neighbour_states = torch.stack([torch.stack([neighbours_state_dict[handle]]) for handle in handles]).squeeze()

        with torch.no_grad(): 
            logits = self._make_logits(states, neighbour_states)
            actions = torch.argmax(logits, dim=1)
            log_probs = torch.log_softmax(logits, dim=1)

        actions = actions.view(actions.size(1), -1)
        log_probs = log_probs.view(log_probs.size(1), -1)
        actions = {handle: actions[handle] for handle in handles}
        log_probs = {handle: log_probs[handle] for handle in handles}
        
        return actions, log_probs


    def sample_action(self, handles: List[str], state_dict: Dict[str, Tensor], neighbours_state_dict: Dict[str, Tensor]) -> Tuple[Dict[int, int], Dict[int, Tensor]]:
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
        try:
            neighbour_states = torch.stack([torch.stack(neighbours_state_dict[handle]) for handle in handles])
        except:
            return 0 


        with torch.no_grad(): 
            logits = self._make_logits(states, neighbour_states)
            action_distribution = torch.distributions.Categorical(logits=logits)
            actions = action_distribution.sample() # TODO: same agent handle in twice, find out why!
            log_probs = action_distribution.log_prob(actions)
        
        try:
            actions = {handle: int(actions[idx]) for idx, handle in enumerate(handles)} #* this cancels out the double handle, but still figure out why
            log_probs = {handle: log_probs[idx] for idx, handle in enumerate(handles)}
        except IndexError as e:
            return 0

        return actions, log_probs


    def update_networks(self, network_parameters: Tuple[Dict, ...]) -> None:
        """
        Update the networks with the given parameters.

        Parameters:
            - network_parameters  Tuple     Tuple of network parameters (actor, critic, target_actor)
        """
        actor_state_dict, critic_state_dict, target_actor_state_dict = network_parameters
        self.actor_network.load_state_dict(actor_state_dict)
        self.critic_network.load_state_dict(critic_state_dict)
        self.target_actor_network.load_state_dict(target_actor_state_dict)


    def get_network_parameters(self) -> Tuple:
        """
        Get the network parameters for the actor and critic networks.

        Returns:
            - network_parameters  Tuple     Tuple of network parameters
        """
        actor_state_dict = self.actor_network.state_dict()
        critic_state_dict = self.critic_network.state_dict()
        return actor_state_dict, critic_state_dict


    def actor_hard_update(self) -> None:
        """ Perform a hard update of the actor target network. """
        self.target_actor_network.load_state_dict(self.actor_network.state_dict())


    def actor_soft_update(self) -> None:
        """ Perform a soft update of the networks. """
        for target_parameter, local_parameter in zip(self.actor_network.parameters(), self.target_actor_network.parameters()):
            target_parameter.data.copy_(self.tau * local_parameter.data + (1 - self.tau) * target_parameter.data)


    def load_controller(self, path: str) -> None:
        """
        Load the controller from the given path. Target actor network is initialised with hard actor network update.

        Parameters:
            - path      str       Path to the controller file (.torch file)
        """
        model = torch.load(path)
        self.actor_network.load_state_dict(model['actor'])
        self.critic_network.load_state_dict(model['critic'])
        self.actor_hard_update()


    def save_controller(self, path: str, name: str = 'controller.torch') -> None:
        """
        Save the controller to the given path. Default model name is "controller.torch",

        Parameters:
            - path      str       Path to save the controller file
            - name      str       Name of the controller file (optional)
        """
        torch.save(self.config, os.path.join(path, 'config.torch'))
        actor_state, critic_state = self.get_network_parameters()
        torch.save({'actor': actor_state, 'critic': critic_state}, os.path.join(path, name))