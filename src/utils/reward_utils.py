import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Dict, Union

from flatland.envs.rewards import Rewards


class SimpleReward(Rewards[int]):
    """
    Simple reward function that gives a reward of +1 for reaching the target or intermediate stations, -1 for collisions, and 0 otherwise.
    """
    def __init__(self):
        super(SimpleReward, self).__init__()

    def reset(self):
        pass

    def step_reward(self, handle: int) -> int:
        """
        Get the reward for a specific agent based on its current state.

        Args:
            handle (int): The unique identifier for the agent.

        Returns:
            float: The reward for the agent.
        """
        # TODO: Implement the logic to compute the reward based on the agent's state
        pass

    def end_of_episode_reward(self, agent, distance_map, elapsed_steps):
        """
        Get the reward for a specific agent at the end of an episode.
        """
        pass

    def cumulate(self, *rewards):
        pass


def compute_discounted_reward_per_agent(rewards: Tensor, gamma: float) -> Tensor: 
    """
    Computes the discounted reward / return / reward-to-go for each agent. 

    Input:
        - rewards           Tensor (batch_size, n_agents)      [float] 

    Output:
        - discounted_reward Tensor (batch_size, n_agents)       [float] 
    """
    batch_size = rewards.size(0)
    n_agents = rewards.size(1)
    future_reward = torch.zeros(n_agents)
    discounted_rewards = torch.zeros(batch_size, n_agents)
    advantages = torch.zeros(batch_size, n_agents)

    for sample in reversed(range(batch_size)):
        returns = rewards[sample] + gamma * future_reward # (n_agents)
        discounted_rewards[sample] = returns.clone()
        future_reward = returns.clone()

    return discounted_rewards


