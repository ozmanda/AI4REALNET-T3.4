import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Dict, Union

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


