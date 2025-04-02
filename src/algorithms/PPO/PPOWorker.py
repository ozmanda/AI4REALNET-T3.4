import ray
import torch
from torch import Tensor
from typing import Dict, Tuple

from src.algorithms.PPO.PPORunner import PPORunner

@ray.remote
class PPOWorker():
    def __init__(self, worker_handle, env_config, controller_config) -> None: 
        pass

    def run(self, ppo_network_parameters, critic_network_parameters) -> Tuple[Dict]:
        """
        Run a single episode in the environment and collect rollouts.

        Parameters:
            - ppo_network_parameters   Tuple     Tuple of PPO network parameters
            - critic_network_parameters Tuple     Tuple of critic network parameters
        """
        pass

    def _generalised_advantage_estimator(self, state: Tensor, next_state: Tensor, reward: Tensor, done: Tensor, step: int) -> Tensor:
        """
        Calculate the Generalised Advantage Estimator (GAE) for the given state and reward.

        Parameters:
            - state          Tensor      (batch_size, n_features)
            - next_state     Tensor      (batch_size, n_features)
            - reward         Tensor      (batch_size)
            - done           Tensor      (batch_size)
            - step           int         Current step in the episode

        Returns:
            - advantages     Tensor      (batch_size)
        """
        pass