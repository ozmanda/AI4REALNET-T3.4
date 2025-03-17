'''
From IC3Net models.py, continuous case removed (flatland has discrete action space)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.networks.MLP import MLP
from typing import Tuple
from argparse import Namespace

class RNN(MLP):
    def __init__(self, args: Namespace, num_inputs: int) -> None:
        super().__init__(args, num_inputs)
        self.n_features: int = num_inputs
        self.n_agents: int = self.args.n_agents
        self.hid_size: int = self.args.hid_size
        self.multiagent: bool = False
        self.n_actions = self.args.n_actions


    def forward(self, observations: Tensor, prev_hidden_state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the RNN module.
    
        Input:
            - observations          Tensor (batch_size * n_agents, num_inputs)
            - prev_hidden_state     Tensor (batch_size * n_agents, hid_size)
    
        Return: Tuple
            - action_log_probs      Tensor (batch_size * n_agents, n_actions)
            - value                 Tensor (batch_size * n_agents, 1)
            - next_hidden_state     Tensor (batch_size * n_agents, hid_size)
        """
        observations, prev_hidden_state = self.adjust_input_dimensions((observations, prev_hidden_state))
        encoded_x: Tensor = self.fc1(observations)
        next_hidden_state: Tensor = F.tanh(self.fc2(prev_hidden_state) + encoded_x)

        v = self.critic(next_hidden_state)

        return self.adjust_output_dimensions((F.log_softmax(self.actor(next_hidden_state), dim=-1), v, next_hidden_state))
    

    def forward_target_network(self, observations: Tensor, prev_hidden_state: Tensor) -> Tensor:
        """
        Forward pass through the target actor network. Identical to self.forward, but only outputs the action logprobs. 
        """
        # Handle single observations by adding a singleton dimension
        observations, prev_hidden_state = self.adjust_input_dimensions((observations, prev_hidden_state))
        encoded_x: Tensor = self.fc1(observations)
        next_hidden_state: Tensor = F.tanh(self.fc2(prev_hidden_state) + encoded_x)
        action_log_probs = F.log_softmax(self.actor_target(next_hidden_state), dim=-1)
        return self.adjust_output_dimensions(action_log_probs)


    def init_hidden(self, batch_size: int) -> Tensor:
        """
        Initialises the hidden and cell states of the RNN module. 

        Input:  batch_size        int

        Output: hidden_state      Tensor (batch_size * n_agents, hid_size)
        """
        return torch.zeros(batch_size * self.n_agents, self.hid_size, requires_grad=True)
    

class LSTM(RNN):
    def __init__(self, args: Namespace, num_inputs: int) -> None:
        super().__init__(args, num_inputs)
        self.n_features: int = num_inputs
        self.n_agents: int = args.n_agents
        self.hid_size: int = args.hid_size
        del self.fc2
        self.lstm_unit = nn.LSTMCell(self.hid_size, self.hid_size)
        self.n_actions = self.args.n_actions


    def forward(self, observations: Tensor, prev_hidden_state: Tensor, prev_cell_state: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of the LSTM module.
    
        Input:
            - observations          Tensor (batch_size * n_agents, num_inputs)
            - prev_hidden_state     Tensor (batch_size * n_agents, hid_size)
            - prev_cell_state       Tensor (batch_size * n_agents, hid_size)
    
        Return: Tuple
            - action_log_probs      Tensor (batch_size * n_agents, n_actions)
            - value                 Tensor (batch_size * n_agents, 1)
            - next_hidden_state     Tensor (batch_size * n_agents, hid_size)
            - next_cell_state       Tensor (batch_size * n_agents, hid_size)
        """
        observations, prev_hidden_state, prev_cell_state = self.adjust_input_dimensions((observations, prev_hidden_state, prev_cell_state))

        encoded_x: Tensor = self.fc1(observations)
        next_hidden_state, next_cell_state = self.lstm_unit(encoded_x, (prev_hidden_state, prev_cell_state))

        v = self.critic(next_hidden_state)
        action_log_probs = F.log_softmax(self.actor(next_hidden_state), dim=-1)

        return self.adjust_output_dimensions((action_log_probs, v, next_hidden_state.clone(), next_cell_state.clone()))
    

    def forward_target_network(self, observations: Tensor, prev_hidden_states: Tensor, prev_cell_states: Tensor) -> Tensor:
        """
        Forward pass through the target actor network. Identical to self.forward, but only outputs the action logprobs. 
        """
        observations, prev_hidden_states, prev_cell_states = self.adjust_input_dimensions((observations, prev_hidden_states, prev_cell_states))

        encoded_x: Tensor = self.fc1(observations)
        next_hidden_state, _ = self.lstm_unit(encoded_x, (prev_hidden_states, prev_cell_states))
        action_log_probs = F.log_softmax(self.actor_target(next_hidden_state), dim=-1)
        return self.adjust_output_dimensions(action_log_probs)
    

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Initialises the hidden and cell states of the LSTM module. 

        Input:
            - batch_size        int

        Output: Tuple
            - hidden_state      Tensor (batch_size * n_agents, hid_size)
            - cell_state        Tensor (batch_size * n_agents, hid_size)
        """
        return tuple((torch.zeros(batch_size * self.n_agents, self.hid_size, requires_grad=True),
                      torch.zeros(batch_size * self.n_agents, self.hid_size, requires_grad=True)))