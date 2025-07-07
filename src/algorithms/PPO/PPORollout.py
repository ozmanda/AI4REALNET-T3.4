import torch
from typing import Tuple, List
from collections import namedtuple
from itertools import chain

PPOTransition = namedtuple('PPOTransition', ('state', 'action', 'log_prob', 'reward', 'next_state', 'done', 'neighbour_states'))
Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'reward', 'next_state', 'done', 'info'))

class PPORollout():
    """ Rollout class for the PPO algorithm. Gathers experience tuples with which the learner then updates the policy. """

    def __init__(self): 
        self.transitions: List[PPOTransition] = []
        self.gae = None


    def append_transition(self, transition: PPOTransition) -> None:
        """ Append a transition to the rollout. """
        self.transitions.append(transition)

    
    def unzip_ppo_transitions(self, device=torch.device('cpu')) -> Tuple:
        """ Unzip the transitions into separate tensors. """
        batch = PPOTransition(*zip(*self.transitions))
        state = torch.stack(batch.state).to(device)
        action = torch.stack(batch.action).to(device)
        log_prob = torch.stack(batch.log_prob).to(device)
        reward = torch.stack(batch.reward).to(device)
        next_state = torch.stack(batch.next_state).to(device)
        done = torch.stack(batch.done).to(device)
        neighbours_states = torch.stack(batch.neighbours_states).to(device)
        actual_len = torch.stack(batch.actual_len).to(device)
        return state, action, log_prob, reward, next_state, done, neighbours_states, actual_len
    

    @staticmethod
    def combine_rollouts(rollouts: List[PPOTransition]):
        """ Combine multiple rollouts into a single rollout. """
        combined_rollout = PPORollout()
        combined_rollout.transitions = list(chain.from_iterable([rollout.transitions for rollout in rollouts]))
        return combined_rollout
    

    def is_empty(self) -> bool:
        """ Check if the rollout is empty. """
        return not self.transitions
