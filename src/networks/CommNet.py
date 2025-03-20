'''
From IC3Net comm.py, multi- and continuous acrion support have been removed, as the flatland environment does not require them. Variable names have been adjusted for clarity and consistency. The original code can be found at 
https://github.com/IC3Net/IC3Net/tree/master
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch import Tensor
from argparse import Namespace
from typing import Tuple, List
from src.networks.MLP import MLP

class CommNet(MLP):
    ''' # TODO: implement input and output adjustment functions from MLP everywhere
    MLP-based communication netowrk that uses a communication vector to communicate information between agents.
    '''
    def __init__(self, args: Namespace, obs_shape: int):
        '''
        Setup internal networks and weights.

        Arguments:
         - args: Namespace: The arguments passed to the model
         - obs_shape: int: The shape of the environment observation for agents
        '''
        super().__init__(args, obs_shape)

        self.args = args
        self.n_features: int = obs_shape
        self.n_agents: int = args.n_agents
        self.n_actions: int = args.n_actions
        self.hid_size: int = args.hid_size
        self.comm_passes: int = args.comm_passes
        self.recurrent: bool = args.recurrent
        self.share_weights: bool = args.share_weights

        # Set the standard deviation of the normal distribution with which initial weights for a linear layer are set
        self.init_std = args.comm_init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        self.comm_mask = torch.ones(self.n_agents, self.n_agents) - torch.eye(self.n_agents)

        # Layers
        self.encoder = nn.Linear(self.n_features, self.hid_size)
        self.actor = nn.Linear(self.hid_size, self.n_actions, dtype=torch.float64)
        if self.recurrent:
            self.LSTM_module = nn.LSTMCell(self.hid_size, self.hid_size, dtype=torch.float64) 
        self.tanh = nn.Tanh()
        self.critic = nn.Linear(self.hid_size, 1, dtype=torch.float64)

        # if weights are shared the linear layer is shared, otherwise one is instatiated for each pass
        if self.share_weights:
            self.C_module = nn.Linear(self.hid_size, self.hid_size, dtype=torch.float64)
            self.C_modules = [self.C_module for _ in range(self.comm_passes)]
        else:
            self.C_modules = [nn.Linear(self.hid_size, self.hid_size, dtype=torch.float64) for _ in range(self.comm_passes)]

        # Communication module weight initialisations
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
    

    def forward(self, state: Tensor, prev_hidden_state: Tensor = None, prev_cell_state: Tensor = None, info={}) -> Tuple[List[Tensor], Tensor, Tuple[Tensor, Tensor]]:
        '''
        Forward function for the CommNet class 

        Arguments:
         - state                tuple       The state of the agents
         - prev_hidden_state    Tensor      Previous hidden state and in the case of LSTM the cell state
         - prev_cell_state      Tensor      previous cell state in the case of LSTM
         - info                 dict        Additional information, e.g., comm_action and comm_

        Returns:
         - tuple:
             - comm_out                 (batchsize, n_agents, #?)
             - action_log_probs         (batchsize, n_agents, n_actions)
             - value                    (batchsize, n_agents, 1)   
             - hidden state tuple
                - next_hidden_state     (batchsize, n_agents, hid_size)  
                - next_cell_state       (batchsize, n_agents, hid_size)      
        '''
        # TODO: define tensor dimensions for this function
        batch_size: int = state.size(0)
        state, prev_hidden_state, prev_cell_state = self.adjust_input_dimensions((state, prev_hidden_state, prev_cell_state))

        prev_hidden_state = prev_hidden_state.to(torch.float64)
        prev_cell_state = prev_cell_state.to(torch.float64)
        encoded_state = self.forward_state_encoder(state)

        num_agents_alive, agent_mask = self._get_agent_mask(batch_size, info)
        self.agent_mask = agent_mask 
        self.n_agent_alive = num_agents_alive
        
        for i in range(self.comm_passes):
            comm_input: Tensor = self._comm_pass(i, prev_hidden_state) 

            if self.args.recurrent:
                # skip connection: input is the combination of comm matrix and encoded state for all agents
                input = encoded_state + comm_input
                hidden_state, cell_state = self.LSTM_module(input, (prev_hidden_state, prev_cell_state))
                prev_cell_state = cell_state

            else: 
                hidden_state: Tensor = sum([encoded_state, 
                                    self.C_modules[i](prev_hidden_state), 
                                    comm_input])
                hidden_state = self.tanh(hidden_state)

            prev_hidden_state = hidden_state

        value: Tensor = self.critic(hidden_state).view(-1, self.n_agents, 1)
        action_log_probs: Tensor = F.softmax(self.actor(hidden_state), dim=-1).view(-1, self.n_agents, self.n_actions)
        hidden_state = hidden_state.view(batch_size, self.n_agents, self.hid_size)

        if self.recurrent: #! cleanup
            return action_log_probs, value, hidden_state.clone(), cell_state.view(batch_size, self.n_agents, self.hid_size).clone()
        else:
            return action_log_probs, value, None, None


    def forward_state_encoder(self, state: Tensor) -> Tensor:
        '''
        Forward pass through the encoder network.

        Arguments:
         - state: torch.Tensor: the current state of the agents (B x N x obs_shape)

        Returns:
         - encoded state: torch.Tensor: The encoded state (B x N x hid_size)
        '''
        state = state.view(-1, self.n_features)
        encoded_state: Tensor = self.encoder(state)
        encoded_state = self.tanh(encoded_state)

        return encoded_state
    

    def forward_target_network(self, state: Tensor, prev_hidden_state: Tensor = None, prev_cell_state: Tensor = None, info={}) -> Tuple[List[Tensor], Tensor, Tuple[Tensor, Tensor]]:
        """
        Performs a forward pass through the target actor network. 

        Arguments:
         - state: tuple: The state of the agents
         - prev_hidden_state: Tensor: previous hidden state and in the case of LSTM the cell state
         - prev_cell_state: Tensor: previous cell state in the case of LSTM
         - info: dict: Additional information to pass through the

        Returns: 
         - action_probs: Tensor: The action probabilities for the agents
        """
        batch_size: int = state.size(0)
        state = state.view(-1, self.n_features)
        prev_hidden_state = prev_hidden_state.view(-1, self.hid_size).to(torch.float64)
        prev_cell_state = prev_cell_state.view(-1, self.hid_size).to(torch.float64)

        encoded_state = self.forward_state_encoder(state)
        num_agents_alive, agent_mask = self._get_agent_mask(batch_size, info)
        self.agent_mask = agent_mask 
        self.n_agent_alive = num_agents_alive

        for i in range(self.comm_passes):
            comm_input: Tensor = self._comm_pass(i, prev_hidden_state) #!

            if self.args.recurrent:
                # skip connection: input is the combination of comm matrix and encoded state for all agents
                input = encoded_state + comm_input
                input = input.view(batch_size * self.n_agents, self.hid_size)
                hidden_state, cell_state = self.LSTM_module(input, (prev_hidden_state, prev_cell_state))
                prev_cell_state = cell_state

            else: 
                hidden_state = sum([encoded_state, self.C_modules[i](prev_hidden_state), comm_input])
                hidden_state = self.tanh(hidden_state)

            prev_hidden_state = hidden_state

        hidden_state = hidden_state.view(batch_size, self.n_agents, self.hid_size)
        action_probs: Tensor = F.softmax(self.actor(hidden_state), dim=-1)

        return action_probs.view(batch_size, self.n_agents, self.n_actions)
    

    def _comm_pass(self, comm_pass: int, prev_hidden_state: Tensor) -> Tensor: 
        """
        Passes the hidden state through the comm network # TODO finish this docstring

        Inputs:
            - comm_pass             int 
            - prev_hidden_state     Tensor      (batchsize * n_agents, hid_size)
        """
        # TODO: cleanup tensor size handling for comm passes 
        prev_hidden_state = prev_hidden_state.view(-1, self.n_agents, self.hid_size) 
        comm = prev_hidden_state.unsqueeze(-2).expand(-1, self.n_agents, self.n_agents, self.hid_size) # (B x N x 1 x hid_size) -> (B x N x N x hid_size)

        mask = self.comm_mask.view(1, self.n_agents, self.n_agents)
        mask = mask.expand(comm.shape[0], self.n_agents, self.n_agents)
        mask = mask.unsqueeze(-1) # (B x N x N x 1)

        mask = mask.expand_as(comm) # (B x N x N x hid_size)
        comm = comm * mask

        if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' and self.n_agents_alive > 1:
            comm = comm / (self.n_agents_alive - 1) #?

        # mask communication from and to dead agents 
        comm = comm * self.agent_mask
        
        # Combine all of C_j for an ith agent
        comm_sum = comm.sum(dim=1) # (B x N x hid_size)
        c = self.C_modules[comm_pass](comm_sum) 
        c = c.view(-1, self.hid_size)
        return c
    

    def _get_agent_mask(self, batch_size: int, info: dict) -> Tuple[int, Tensor]:
        '''
        Get the mask for the alive agents. # TODO: finish this docstring

        Arguments:
         - batch_size: int: The size of the batch
         - info: dict: Additional information to pass through the

        '''
        if 'alive_mask' in info:
            agent_mask: Tensor = torch.from_numpy(info['alive_mask']) if isinstance(info['alive_mask'], np.ndarray) else info['alive_mask'] # TODO: clean this up
            num_agents_alive: int = agent_mask.sum(dim=1)
        else: 
            agent_mask: Tensor = torch.ones(batch_size, self.n_agents)
            num_agents_alive: int = self.n_agents

        agent_mask = agent_mask.view(batch_size, 1, self.n_agents)
        agent_mask = agent_mask.expand(batch_size, self.n_agents, self.n_agents).unsqueeze(-1)

        # Hard Attention -> action whether an agent communications or not #! ????
        if self.args.hard_attention:
            comm_action: Tensor = torch.tensor(info['comm_action']) if isinstance(info['comm_action'], np.ndarray) else info['comm_action']
            comm_action_mask: Tensor = comm_action.expand(batch_size, self.n_agents, self.n_agents).unsqueeze(-1) # (B x N x N x 1)
            agent_mask = comm_action_mask.double() * agent_mask.clone()

        agent_mask_transpose = agent_mask.transpose(1, 2) # (B x N x N x 1) -> mirrors around the diagonal
        agent_mask = agent_mask * agent_mask_transpose    # TODO: check that this change is correct     

        return num_agents_alive, agent_mask

 
    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        '''
        Initialize the hidden state of the network.

        Arguments:
         - batch_size: int: The size of the batch to initialize the hidden state with

        Returns:
         - Tuple[torch.Tensor, torch.Tensor]: The hidden state of the network, with the first element being the hidden state and the second being the cell state. Gradients should be calculated for both during backpropagation.
        '''
        return tuple((torch.zeros(batch_size * self.n_agents, self.hid_size, requires_grad=True),
                      torch.zeros(batch_size * self.n_agents, self.hid_size, requires_grad=True)))
    

    def init_weights(self, m: nn.Linear) -> None:
        ''' Initialize the weights of a nn.Linear layer '''
        # TODO: this function is never called, check original code
        m.weight.data.normal_(0, self.init_std)            
        
