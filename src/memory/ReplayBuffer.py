import torch
import random
import numpy as np
from collections import Iterable, deque, namedtuple

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer():
    """ 
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, action_size, buffer_size, batch_size, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        if action is None:
            Warning("Action is None")
        experience = Experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.memory.append(experience)  

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(self.vstack_states([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(self.vstack_states([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(self.vstack_states([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(self.vstack_states([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(self.vstack_states([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def vstack_states(self, states):
        """
        Determines the dimensionality of the state, considering states being iterables and then reshapes
        them to a tensor of shape (len(states), statedim). For example:
        
        1D state: [1, 2, 3, 4] -> [[1], [2], [3], [4]]  |  (4,) -> (4, 1)
        2D state: [[1, 2], [3, 4]] -> [[1, 2], [3, 4]]  |  (4, 2) -> (4, 2)  (no change)
        """
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states
    
    def __len__(self):
        return len(self.memory)