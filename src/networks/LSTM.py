from argparse import Namespace
from src.networks.RNN import RNN
import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor
from typing import Tuple, Dict, Iterable
from itertools import chain

class LSTM(nn.Module):
    """
    LSTM network for multi-agent reinforcement learning, adapted from (Hochreiter & Schmidhauber 1997) and made for use with
    Flatland multi-agent simulation environment.
    """
    def __init__(self, config: Dict) -> None:
        super(LSTM, self).__init__()
        self._init_config(config)
        self._init_network()

    def _init_config(self, config: Dict) -> None:
        self.n_features: int = config['n_features']
        self.state_size: int = config['state_size']
        self.n_agents: int = config['n_agents']
        self.n_actions = config['action_size']
        self.hidsize = config['hidden_size']
        self.lstm_hidsize = config['lstm_hidden_size']
    
    def _init_network(self) -> None:
        """
        Initialise an LSTM-based Deep RL actor network
        """
        self.feature_extraction = nn.Sequential(
            nn.Linear(self.state_size, self.hidsize),
            nn.ReLU(),
            nn.Linear(self.hidsize, self.hidsize),
            nn.ReLU()
        )
        self.lstm_unit = nn.LSTM(self.hidsize, self.lstm_hidsize)
        self.actor_head = nn.Linear(self.lstm_hidsize, self.n_actions)
        self.critic_head = nn.Linear(self.lstm_hidsize, 1)

        self._init_hidden()

    
    def get_parameters(self):
        params: Iterable = chain(
            self.feature_extraction.parameters(),
            self.lstm_unit.parameters(),
            self.actor_head.parameters(),
            self.critic_head.parameters()
        )
        return params


    def reset_hidden_states(self, n_agents: int = None) -> None:
        """
        Reset the hidden and cell states of the LSTM module for all agents
        """
        self.n_agents = n_agents if n_agents else self.n_agents
        self._init_hidden()


    def forward(self, states: Tensor, hidden: Tuple[Tensor, Tensor] = None, select_best_action = False) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """
        Forward pass through the LSTM network for multiple time steps.
    
        Input:
            - observations          Tensor (batchsize, n_agents, num_inputs) #! check options
            - hidden                Tuple(Tensor, Tensor) each of shape (n_agents, lstm_hidsize)
            - select_best_action    bool - if True, selects the action with the highest log-probability instead of sampling from the distribution
    
        Return: Tuple
            - actions               Tensor (sequence_length, n_agents)
            - action_log_probs      Tensor (sequence_length, n_agents, n_actions) #! check shape
            - value                 Tensor (sequence_length, n_agents, 1)
            - hidden_states         Tuple[Tensor, Tensor] of size (n_agents, lstm_hidsize)
            - next_cell_state       Tensor (n_agents, lstm_hidsize)
        """
        hx, prev_hidden_states = self._feature_extraction(states)   # (n_agents, hid_size) # TODO: manage previous hidden states! 
        logits = self.actor_head(hx)
        action_distribution = torch.distributions.Categorical(logits=logits)
        if select_best_action:
            actions = torch.argmax(logits, dim=-1)
            action_log_probs = torch.log_softmax(logits, dim=1)
        else:
            actions = action_distribution.sample()
            action_log_probs = action_distribution.log_prob(actions)
        
        values = self.critic_head(hx)
        return actions, action_log_probs, values, prev_hidden_states
    

    # TODO: add extras variable to forward output (contains previous and current hidden states)
    def evaluate(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate the states, actions and next states experienced by the agents using the current policy

        Input:
            - states            Tensor (sequence_length, state_size)
            - actions           Tensor (sequence_length, 1)
            - next_states       Tensor (sequence_length, state_size)

        Output: Tuple
            - log_probs         Tensor (sequence_length, 1)
            - state_values      Tensor (sequence_length, 1)
            - next_state_values Tensor (sequence_length, 1)
        """
        encoded_x = self.feature_extraction(states)   # (sequence_length, n_agents, hid_size)
        hx, _ = self.lstm_unit(encoded_x)                     # (sequence_length, n_agents, lstm_hidsize)
        logits = self.actor_head(hx)
        action_distribution = torch.distributions.Categorical(logits=logits)
        log_probs = action_distribution.log_prob(actions)
        state_values = self.critic_head(hx)
        encoded_next_x = self.feature_extraction(next_states)   # (sequence_length, n_agents, hid_size)
        next_hx, _ = self.lstm_unit(encoded_next_x)   
        next_state_values = self.critic_head(next_hx)                  # (sequence_length, n_agents, 1)
        return log_probs, state_values, next_state_values


    def evaluate_action(self, states: Tensor, actions: Tensor) -> Tensor:
        """
        Computes the log-probabilities of the given actions under the current policy.
        
        Input:
            - states            Tensor (n_agents, state_size)
            - actions           Tensor (n_agents, 1)
            - extras            Dict containing previous hidden and cell states
        Output: Tensor
            - log_probs         Tensor (n_agents, 1)
        """
        encoded_x = self.feature_extraction(states)   # (sequence_length, hid_size)
        logits = self.actor_head(encoded_x)
        action_distribution = torch.distributions.Categorical(logits=logits)
        log_probs = action_distribution.log_prob(actions)
        return log_probs


    def state_values(self, states: Tensor, hidden: Tuple[Tensor, Tensor]) -> Tensor:
        """ 
        Get the state values from the critic network for the current states. 
        
        Input:
            - states            Tensor (n_agents, state_size)
        
        Output: Tensor
            - state_values      Tensor (n_agents, 1)
        """
        hx, _ = self._feature_extraction(states)
        return self.critic_head(hx)


    def _feature_extraction(self, states: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Extract features from the input states using the feature extraction network.
        Input:
            - states            Tensor (seq_len, n_agents, state_size)
            - hidden            Optional Tuple[Tensor, Tensor] (n_agents, lstm_hidsize)

        Output: Tuple
            - hx                Tensor (n_agents, lstm_hidsize)
            - hidden_states     Tuple(Tensor, Tensor) each of shape (n_agents, lstm_hidsize)
        """
        encoded_x: Tensor = self.feature_extraction(states).unsqueeze(0)     # (n_agents, hid_size)
        hx, next_hidden_states = self.lstm_unit(encoded_x, (self.prev_hidden_state, self.prev_cell_state))
        hidden_states = {'prev_hidden_state': self.prev_hidden_state.squeeze(0).detach(), 
                            'prev_cell_state': self.prev_cell_state.squeeze(0).detach(),
                            'next_hidden_state': next_hidden_states[0].squeeze(0).detach(),
                            'next_cell_state': next_hidden_states[1].squeeze(0).detach()}
        self.prev_hidden_state, self.prev_cell_state = next_hidden_states
        hx = hx.squeeze(0)  # (n_agents, hid_size) - removing sequence length dimension
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
        with torch.no_grad():
            self.prev_hidden_state = torch.zeros(1, self.n_agents, self.lstm_hidsize, requires_grad=False)
            self.prev_cell_state = torch.zeros(1, self.n_agents, self.lstm_hidsize, requires_grad=False)


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
        