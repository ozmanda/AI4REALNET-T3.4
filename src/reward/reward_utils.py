import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Dict, Union

from flatland.envs.rewards import Rewards
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.step_utils.env_utils import AgentTransitionData
from flatland.envs.step_utils.states import TrainState


class SimpleReward(Rewards[int]):
    """
    Simple reward function that gives a reward of +1 for reaching the target or intermediate stations, -1 for collisions, and 0 otherwise.
    """
    def __init__(self):
        super(SimpleReward, self).__init__()


    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> int:
        """
        Get the reward for a specific agent based on its current state. Simplified version of the default Flatland reward function: 
         - +1 for reaching the target
         - -1 for collisions
         

        Args:
            handle (int): The unique identifier for the agent.

        Returns:
            float: The reward for the agent.
        """
        reward = 0
        if agent.state == TrainState.DONE:
            reward = 1
        elif agent.state_machine.previous_state == TrainState.MOVING and agent.state == TrainState.STOPPED and not agent_transition_data.state_transition_signal.stop_action_given:
            # collision
            reward = -1
        else:
            reward = 0
        return reward


    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> int:
        """
        Simple end-of-episode reward function: either the agent has reached its target (+1) or not (-1).
        """
        reward = 0
        if agent.state == TrainState.DONE:
            reward = 1
        else:
            reward = -1
        return reward
        

    def cumulate(self, *rewards) -> int:
        return sum(rewards)


    def empty(self) -> int:
        return 0


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