from argparse import Namespace
from src.networks.RNN import RNN
import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor
from typing import Tuple, Dict

class LSTM(nn.Module):
    """
    LSTM network for multi-agent reinforcement learning, adapted from (Hochreiter & Schmidhauber 1997) and made for use with
    Flatland multi-agent simulation environment.
    """
    def __init__(self, config: Dict, n_agents: int) -> None:
        super(LSTM, self).__init__()
        self.n_agents = n_agents
        self._init_config(config)
        self._init_network()
        self.reset_network(n_agents)

    def _init_config(self, config: Dict) -> None:
        self.n_features: int = config['n_features']
        self.n_agents: int = config['n_agents']
        self.n_actions = config['n_actions']
        self.hidsize = config['hidden_size']
        self.lstm_hidsize = config['lstm_hidden_size']
    
    def _init_network(self) -> None:
        """
        Initialise an LSTM-based Deep RL actor network
        """
        self.feature_extraction = nn.Sequential(
            nn.Linear(self.n_features, self.hidsize),
            nn.ReLU(),
            nn.Linear(self.hidsize, self.hidsize),
            nn.ReLU()
        )
        self.lstm_unit = nn.LSTM(self.hidsize, self.lstm_hidsize)
        self.actor_head = nn.Linear(self.lstm_hidsize, self.n_actions)
        self.critic_head = nn.Linear(self.lstm_hidsize, 1)

        self.prev_hidden_state, self.prev_cell_state = self._init_hidden()


    def reset_network(self, n_agents: int = None) -> None:
        """
        Reset the hidden and cell states of the LSTM module for all agents
        """
        self.n_agents = n_agents if n_agents is not None else self.n_agents
        self.prev_hidden_state, self.prev_cell_state = self._init_hidden()


    def forward(self, states: Tensor, hidden: Tuple[Tensor, Tensor] = None, select_best_action = False) -> Tuple[Tensor, Tensor, Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through the LSTM network for multiple time steps.
    
        Input:
            - observations          Tensor (sequence_length, n_agents, num_inputs)
            - hidden                Tuple(Tensor, Tensor) each of shape (n_agents, hid_size)
            - select_best_action    bool - if True, selects the action with the highest log-probability instead of sampling from the distribution
    
        Return: Tuple
            - action_log_probs      Tensor (sequence_length, n_agents, n_actions)
            - value                 Tensor (sequence_length, n_agents, 1)
            - next_hidden_state     Tensor (n_agents, hid_size)
            - next_cell_state       Tensor (n_agents, hid_size)
        """
        hx, hidden_states = self._feature_extraction(states)   # (n_agents, hid_size) # TODO: manage previous hidden states! 
        logits = self.actor_head(hx)
        action_distribution = torch.distributions.Categorical(logits=logits)
        if select_best_action:
            actions = torch.argmax(logits, dim=-1)
            action_log_probs = torch.log_softmax(logits, dim=1)
        else:
            actions = action_distribution.sample()
            action_log_probs = action_distribution.log_prob(actions)
        
        values = self.critic_head(hx)
        return actions, action_log_probs, values, hidden_states
    

    def state_values(self, states: Tensor) -> Tensor:
        """ Get the state values from the critic network for the current states. """
        return self.critic_head(self._feature_extraction(states)[0])

    def _feature_extraction(self, states: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        encoded_x: Tensor = self.feature_extraction(states)     # (n_agents, hid_size)
        encoded_x = encoded_x.unsqueeze(1)              # (n_agents, 1, hid_size) - adding sequence length dimension
        hx, hidden_states = self.lstm_unit(encoded_x, (self.prev_hidden_state, self.prev_cell_state))
        hx = hx[:, -1, :]  # (n_agents, hid_size) - removing sequence length dimension
        self.prev_hidden_state, self.prev_cell_state = hidden_states
        return hx, hidden_states


    def _init_hidden(self) -> None:
        """
        Initialises the hidden and cell states of the LSTM module for all agents

        Input:
            - batch_size        int

        Output: Tuple
            - hidden_state      Tensor (batch_size * n_agents, hid_size)
            - cell_state        Tensor (batch_size * n_agents, hid_size)
        """
        self.hidden_state = torch.zeros(self.n_agents, self.hidsize, requires_grad=True)
        self.cell_state = torch.zeros(self.n_agents, self.hidsize, requires_grad=True)


    def get_state_dict(self) -> Dict:
        """
        Get the state dictionary of the LSTM network.
        """
        state_dict = {
            'feature_extraction': self.feature_extraction.state_dict(),
            'lstm_unit': self.lstm_unit.state_dict(),
            'actor_head': self.actor_head.state_dict(),
            'critic_head': self.critic_head.state_dict()
        }
        return state_dict
    

    def update_weights(self, state_dict: Dict) -> None:
        """
        Update the weights of the LSTM network with the given state dictionary.
        """
        if 'feature_extraction' in state_dict:
            self.feature_extraction.load_state_dict(state_dict['feature_extraction'])
        if 'lstm_unit' in state_dict:
            self.lstm_unit.load_state_dict(state_dict['lstm_unit'])
        if 'actor_head' in state_dict:
            self.actor_head.load_state_dict(state_dict['actor_head'])
        if 'critic_head' in state_dict:
            self.critic_head.load_state_dict(state_dict['critic_head'])
        