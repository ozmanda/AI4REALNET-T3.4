import copy
import torch
from torch.nn import functional as F
import pickle
from torch import optim

from memory.ReplayBuffer import ReplayBuffer
from networks.DDQN import DDDQNPolicy


class DDQNController():

    def __init__(self, state_size, action_size, parameters, evaluation_mode=False):
        self.policy = DDDQNPolicy(state_size, action_size, parameters, evaluation_mode=evaluation_mode)

    
    def env_step():
        pass


    def network_step(self, state, action, reward, next_state, done):
        """ Save experience in replay memory, and initiate learning if applicable. """
        # add experience to ReplayBuffer
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.__len__() > self.buffer_min_size and self.memory.__len__() > self.batch_size:
                self.learn()


    def save_replay_buffer(self, filename):
        """ Save replay buffer to file. """
        with open(filename, 'wb') as f:
            pickle.dump(list(self.memory.memory[-500000:]), f)


    def load_replay_buffer(self, filename):
        """ Load replay buffer from file. """
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)