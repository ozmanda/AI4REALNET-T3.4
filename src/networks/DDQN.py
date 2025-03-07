
import os

from memory.ReplayBuffer import Experience, ReplayBuffer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
import copy


class DuelingQNetwork(nn.Module):
    """ 
    Dueling Q-Network according to https://arxiv.org/pdf/1511.06581.pdf 
     --> this network is used to estimate the Q-value in the Double Dueling DQN
    """

    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super().__init__()

        # value network
        self.fc1_val = nn.Linear(state_size, hidsize1)
        self.fc2_val = nn.Linear(hidsize1, hidsize2)
        self.fc3_val = nn.Linear(hidsize2, 1)

        # advantage network
        self.fc1_adv = nn.Linear(state_size, hidsize1)
        self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc3_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        val = F.relu(self.fc1_val(x))
        val = F.relu(self.fc2_val(val))
        val = self.fc3_val(val)

        adv = F.relu(self.fc1_adv(x))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc3_adv(adv)

        return val + adv - adv.mean()
    

class DDDQNPolicy():
    """ Double Dueling DQN Policy """

    def __init__(self, state_size, action_size, parameters, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode
        self.state_size = state_size
        self.action_size = action_size
        self.hidsize = 1

        # set model parameters if not performing evaluation
        if not evaluation_mode:
            self.hidsize = parameters.hidden_size
            self.buffer_size = parameters.buffer_size
            self.batch_size = parameters.batch_size
            self.update_every = parameters.update_every
            self.learning_rate = parameters.learning_rate
            self.tau = parameters.tau
            self.gamma = parameters.gamma
            self.buffer_min_size = parameters.buffer_min_size

        # Device
        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else: 
            self.device = torch.device("cpu")
        
        # Q-Network
        self.qnet_local = DuelingQNetwork(state_size, action_size, self.hidsize, self.hidsize).to(self.device)

        if not evaluation_mode:
            self.qnet_target = copy.deepcopy(self.qnet_local)
            self.optimiser = optim.Adam(self.qnet_local.parameters(), lr=self.learning_rate)
            self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)

            self.t_step = 0
            self.loss = 0
    
    
    def act(self, state, eps=0.):
        """ 
        Returns actions for given state as per current policy.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # calling eval sets the module in evaluation mode: disables training-specific behaviours like 
        #  dropout, batch noramlization etc.
        self.qnet_local.eval()

        # torch.no_grad() disables gradient calculation when we know we won't call Tensor.backward()
        with torch.no_grad():
            action_values = self.qnet_local(state)

        # set back to training mode
        self.qnet_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))     
        

    def learn(self):
        """ 
        Update value parameters using given batch of experience tuples.
        """
        states, actions, rewards, next_states, dones = self.memory.sample()

        # predicted q-values for actions
        q_pred = self.qnet_local(states).gather(1, actions)

        # compute target q-values
        q_best_actions = self.qnet_local(next_states).max(1)[1]    # best actions for next states
        q_targets_next = self.qnet_target(next_states).gather(1, q_best_actions.unsqueeze(-1))   # target q-values

        # compute targets for current states (1-dones ignores future rewards for terminal states)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(q_pred, q_targets.unsqueeze(-1))

        # Minimise loss
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # soft update target network
        self.soft_update()


    def soft_update(self):
        """ 
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Module.parameters() provides a generator that returns a reference to the model's weights and biases.
        parameter.data accesses the raw tensor data and .copy_ writes directly to the data
        """
        for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


    def save(self, filename):
        """ 
        Save model parameters to file.
        """
        torch.save(self.qnet_local.state_dict(), f'{filename}.local')
        torch.save(self.qnet_target.state_dict(), f'{filename}.target')


    def load(self, filename):
        """ 
        Load model parameters from file.
        """
        if os.path.exists(f'{filename}.local'):
            self.qnet_local.load_state_dict(torch.load(f'{filename}.local'))
        if os.path.exists(f'{filename}.target'):
            self.qnet_target.load_state_dict(torch.load(f'{filename}.target'))


    def test(self):
        """ 
        Test the act and _learn functions
        """
        self.act(np.array([[0] * self.state_size]))
        self.learn()