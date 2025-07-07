import ray
import torch
from torch import Tensor
from typing import Dict, Tuple, Union

from src.algorithms.PPO.PPORunner import PPORunner
from src.configs.EnvConfig import FlatlandEnvConfig

@ray.remote
class PPOWorker():
    def __init__(self, worker_handle: Union[int, str], env_config: FlatlandEnvConfig, controller_config, device: str = 'cpu', max_steps: int = 1e10) -> None: 
        self.worker_handle = worker_handle
        self.max_steps = max_steps
        self.env = env_config.create_env()
        self.controller = controller_config.create_controller(torch.device(device=device))
        
        self.gamma = controller_config.gamma
        self.lam = controller_config.lam
        self.gae_horizon = controller_config.gae_horizon

        self.runner = PPORunner(env=self.env, controller=self.controller)


    def run(self, ppo_network_parameters, critic_network_parameters) -> Tuple[Dict]:
        """
        Run a single episode in the environment and collect rollouts.

        Parameters:
            - ppo_network_parameters   Tuple     Tuple of PPO network parameters
            - critic_network_parameters Tuple     Tuple of critic network parameters
        """
        self.controller.update_networks((ppo_network_parameters, critic_network_parameters))

        rollout_dict, info = self.runner.run(self.max_steps)
        info['handle'] = self.worker_handle
        info['shaped_reward'] = 0
        info['env'] = self.env

        for handle, rollout in rollout_dict.items(): #TODO: check that this actually edits the content of the dict
            if rollout.is_empty():
                continue
            state, action, log_prob, reward, next_state, done, neighbours_states, actual_len = rollout.unzip_ppo_transitions(device=self.controller.device) 
            rollout.gae = self._generalised_advantage_estimator(state, next_state, reward, done, neighbours_states, actual_len)
            info['shaped_reward'] += reward.sum().item()
        
        return rollout_dict,

    def _generalised_advantage_estimator(self, state: Tensor, next_state: Tensor, reward: Tensor, done: Tensor, step: int) -> Tensor:
        """
        Calculate the Generalised Advantage Estimator (GAE) for the given state and reward. # TODO: understand the formulae behind this

        Parameters:
            - state          Tensor      (batch_size, n_features)
            - next_state     Tensor      (batch_size, n_features)
            - reward         Tensor      (batch_size)
            - done           Tensor      (batch_size)
            - step           int         Current step in the episode

        Returns:
            - advantages     Tensor      (batch_size)
        """
        with torch.no_grad(): 
            state_values: Tensor = self.controller.critic_network(state).squeeze(1)
            next_state_values: Tensor = self.controller.critic_network(next_state).squeeze(1)
            expected_state_values = (next_state_values * torch.pow(self.gamma, step)) * (1 - done) + reward
            gae = expected_state_values - state_values
            gae_copy = gae.clone()
            for i in reversed(range(len(gae)-1)): 
                gae[i] += gae[i+1] * self.gamma * self.lam
                if i + self.gae_horizon < len(gae): 
                    gae[i] -= gae_copy[i + self.gae_horizon] * (self.lam * self.gamma) ** self.gae_horizon
        return gae