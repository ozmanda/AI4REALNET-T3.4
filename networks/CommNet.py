'''
From IC3Net comm.py, multi- and continuous acrion support have been removed, as the flatland environment does not require them. Variable names have been adjusted for clarity and consistency. The original code can be found at 
https://github.com/IC3Net/IC3Net/tree/master
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from argparse import Namespace
from typing import Tuple, List

class CommNet(nn.Module):
    '''
    MLP-based communication netowrk that uses a communication vector to communicate information between agents.
    '''
    def __init__(self, args: Namespace, obs_shape: int):
        '''
        Setup internal networks and weights.

        Arguments:
         - args: Namespace: The arguments passed to the model
         - obs_shape: int: The shape of the environment observation for agents
        '''
        super().__init__()

        self.args = args
        self.n_agents: int = args.n_agents
        self.hid_size: int = args.hid_size
        self.comm_passes: int = args.comm_passes
        self.recurrent: bool = args.recurrent

        #! this is multi-action support, which is not needed for flatland
        self.heads = nn.ModuleList([nn.Linear(args.hid_size, o) for o in args.naction_heads])

        # Set the standard deviation of the normal distribution with which initial weights for a linear layer are set
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm:
            self.comm_mask = torch.zeros(self.n_agents, self.n_agents)
        else:
            self.comm_mask = torch.ones(self.n_agents, self.n_agents) - torch.eye(self.n_agents)

        # Encoder
        self.encoder = nn.Linear(obs_shape, self.hid_size)

        if self.recurrent:
            self.LSTM_module = nn.LSTMCell(self.hid_size, self.hid_size) 

        # if weights are shared the linear layer is shared, otherwise one is instatiated for each pass
        if self.share_weights:
            self.C_module = nn.Linear(self.hid_size, self.hid_size)
            self.C_modules = [self.C_module for _ in range(self.comm_passes)]
        else:
            self.C_modules = [nn.Linear(self.hid_size, self.hid_size) for _ in range(self.comm_passes)]

        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()
        self.value_head = nn.Linear(self.hid_size, 1)
    

    def forward(self, state: Tensor, prev_hidden_state: Tensor = None, prev_cell_state: Tensor = None, info={}) -> Tuple[List[Tensor], Tensor, Tuple[Tensor, Tensor]]:
        '''
        Forward function for the CommNet class

        Arguments:
         - state: tuple: The state of the agents
         - prev_hidden_state: Tensor: previous hidden state and in the case of LSTM the cell state
         - prev_cell_state: Tensor: previous cell state in the case of LSTM
         - info: dict: Additional information to pass through the

        Returns:
         - tuple: (next_hidden_state, comm_out, action_data, value_head)
             - next_hidden_state: Tensor
             - comm_out: Tensor
             - action_data: data needed to take next action #! data type?
             - value_head: Tensor #! (?)        
        '''
        batch_size: int = state.size()[0]
        encoded_state, hidden_state, cell_state = self.forward_state_encoder(state, prev_hidden_state, prev_cell_state)

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        # Hard Attention -> action whether an agent communications or not #! ????
        if self.args.hard_attention:
            comm_action: Tensor = torch.tensor(info['comm_action'])
            comm_action_mask: Tensor = comm_action.expand(batch_size, self.n_agents, self.n_agents).unsqueeze(-1) # (B x N x N x 1)
            agent_mask *= comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2) # (B x N x N x 1) -> mirrors around the diagonal

        for i in range(self.comm_passes):
            comm = hidden_state.view(batch_size, self.n_agents, self.hid_size) if self.args.recurrent else hidden_state # (B x N x hid_size)
            comm = comm.unsqueeze(-2).expand(-1, self.n_agents, self.n_agents, self.hid_size) # (B x N x 1 x hid_size) -> (B x N x N x hid_size)

            mask = self.comm_mask.view(1, self.n_agents, self.n_agents)
            mask = mask.expand(comm.shape[0], self.n_agents, self.n_agents)
            mask = mask.unsqueeze(-1) # (B x N x N x 1)

            mask = mask.expand_as(comm) # (B x N x N x hid_size)
            comm = comm * mask

            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1) #?

            # mask communication from and to dead agents
            comm = comm * agent_mask
            comm = comm * agent_mask_transpose
            
            # Combine all of C_j for an ith agent -> h_j
            comm_sum = comm.sum(dim=1) # (B x N x hid_size)
            c = self.C_modules[i](comm_sum) 

            if self.args.recurrent:
                # skip connection: input is the combination of comm matrix and encoded state for all agents
                input = encoded_state + c
                input = input.view(batch_size * self.n_agents, self.hid_size)
                hidden_state, cell_state = self.LSTM_module(input, (hidden_state, cell_state))

            else: 
                hidden_state = sum([encoded_state, self.C_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)

        value_head = self.value_head(hidden_state)
        hidden_state = hidden_state.view(batch_size, self.n_agents, self.hid_size)

        action: List[Tensor] = [F.softmax(head(hidden_state), dim=-1) for head in self.heads]

        if self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone())
        else:
            return action, value_head, (None, None)


    def forward_state_encoder(self, state: Tensor, prev_hidden_state: Tensor = None, prev_cell_state: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        '''
        Forward pass through the encoder network.

        Arguments:
         - state: torch.Tensor: the current state of the agents (B x N x obs_shape)
         - hidden_state: torch.Tensor: the hidden state of the network (for recurrent models) -> (B x N x hid_size)

        Returns:
         - encoded state: torch.Tensor: The encoded state (B x N x hid_size)
         - hidden_state: torch.Tensor: The hidden state of the network (B*N x hid_size)
         - cell_state: torch.Tensor: The cell state of the network (B*N x hid_size)
        '''
        encoded_state: Tensor = self.encoder(state)
        cell_state: Tensor = prev_cell_state

        if self.recurrent:
            hidden_state = prev_hidden_state
        else:
            encoded_state = self.tanh(encoded_state)
            hidden_state = encoded_state

        return encoded_state, hidden_state, cell_state
    

    def get_agent_mask(self, batch_size: int, info: dict) -> Tuple[int, Tensor]:
        '''
        Get the mask for the alive agents.

        Arguments:
         - batch_size: int: The size of the batch
         - info: dict: Additional information to pass through the

        '''
        if 'alive_mask' in info:
            agent_mask: Tensor = torch.from_numpy(info['alive_mask'])
            num_agents_alive: int = agent_mask.sum()
        else: 
            agent_mask: Tensor = torch.ones(self.n_agents)
            num_agents_alive: int = self.n_agents

        agent_mask = agent_mask.view(1, 1, self.n_agents)
        agent_mask = agent_mask.expand(batch_size, self.n_agents, self.n_agents).unsqueeze(-1)

        return num_agents_alive, agent_mask

 
    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        '''
        Initialize the hidden state of the network.

        Arguments:
         - batch_size: int: The size of the batch to initialize the hidden state with

        Returns:
         - Tuple[torch.Tensor, torch.Tensor]: The hidden state of the network, with the first element being the hidden state and the second being the cell state. Gradients should be calculated for both during backpropagation.
        '''
        #! the size of this seems wrong to me, as the batch size is not used in the LSTMCell
        return tuple((torch.zeros(batch_size * self.n_agents, self.hid_size, requires_grad=True),
                      torch.zeros(batch_size * self.n_agents, self.hid_size, requires_grad=True)))
    
        #! following text is from the torch.nn.LSTMCell documentation:
        '''
        Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_size)` or `(input_size)`: tensor containing input features
        - **h_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial hidden state
        - **c_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial cell state

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

        Outputs: (h_1, c_1)
        - **h_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next hidden state
        - **c_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next cell state
        '''
    

    def init_weights(self, m: nn.Linear) -> None:
        ''' Initialize the weights of a nn.Linear layer '''
        m.weight.data.normal_(0, self.init_std)            
        
