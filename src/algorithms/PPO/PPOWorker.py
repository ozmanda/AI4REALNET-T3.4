from collections import namedtuple
from typing import Tuple, Dict, Union
from flatland.envs.rail_env import RailEnv

import queue
import torch
from torch import Tensor
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event

from src.utils.obs_utils import obs_dict_to_tensor
from src.configs.EnvConfig import FlatlandEnvConfig
from src.algorithms.PPO.PPOController import PPOController
from src.configs.ControllerConfigs import PPOControllerConfig
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer

Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'reward', 'next_state', 'done', 'info'))

class PPOWorker(mp.Process): 
    def __init__(self, worker_id: Union[str, int], env_config: FlatlandEnvConfig, controller_config: PPOControllerConfig, 
                 logging_queue: mp.Queue, rollout_queue: mp.Queue, weights_queue: mp.Queue, done_event: Event,
                 max_steps: Tuple = (10000, 1000), device: str = 'cpu'):
        super().__init__()

        # multiprocessing setup
        self.worker_id: Union[str, int] = worker_id
        self.logging_queue: mp.Queue = logging_queue
        self.rollout_queue: mp.Queue = rollout_queue
        self.weights_queue: mp.Queue = weights_queue
        self.done_event: Event = done_event
        self.device: str = device

        # env and controller setup
        self.env_config: FlatlandEnvConfig = env_config
        self._init_env()
        self.controller_config: PPOControllerConfig = controller_config
        self.controller: PPOController = self.controller_config.create_controller()

        # rollout setup
        self.max_steps: int = max_steps[0]
        self.max_steps_per_episode: int = max_steps[1]
        self.rollout: MultiAgentRolloutBuffer = MultiAgentRolloutBuffer(n_agents=self.env.number_of_agents)


    def _init_env(self) -> RailEnv:
        """
        Initialize the environment based on the provided configuration.
        """
        self.obs_type: str = self.env_config['observation_builder_config']['type']
        self.max_depth: int = self.env_config['observation_builder_config']['max_depth']

        self.env_config = FlatlandEnvConfig(self.env_config)
        self.env: RailEnv = self.env_config.create_env()


    def run(self) -> MultiAgentRolloutBuffer:
        """
        Run a single episode in the environment and collect rollouts.
        """
        self.env.reset()
        self.rollout.reset(agent_handles=self.env.get_agent_handles())
        self._try_refresh_weights()
        # TODO: add actor-critic update after each episode

        episode_step = 0
        current_state_dict, _ = self.env.reset()
        n_agents = self.env.number_of_agents
        current_state_tensor: Tensor = obs_dict_to_tensor(observation=current_state_dict, 
                                                          obs_type=self.obs_type, 
                                                          n_agents=n_agents,
                                                          max_depth=self.max_depth, 
                                                          n_nodes=self.controller.config['n_nodes'])

        while not self.done_event.is_set():
            actions, log_probs = self.controller.sample_action(current_state_tensor)
            actions_dict: Dict[Union[int, str], Tensor] = {}
            for i, agent_handle in enumerate(self.env.get_agent_handles()):
                actions_dict[agent_handle] = actions[i]

            next_state, rewards, dones, infos = self.env.step(actions_dict)
            next_state_tensor: Tensor = obs_dict_to_tensor(observation=next_state, 
                                                           obs_type=self.obs_type, 
                                                           n_agents=n_agents,
                                                           max_depth=self.max_depth, 
                                                           n_nodes=self.controller.config['n_nodes'])

            self.rollout.add_transitions(states=current_state_tensor, actions=actions_dict, log_probs=log_probs, 
                                    rewards=rewards, next_states=next_state_tensor, dones=dones)

            current_state_tensor = next_state_tensor
            episode_step += 1

            if all(dones.values()) or episode_step >= self.max_steps_per_episode:
                # TODO: gather current episode info and pass it to the queue
                self.rollout.end_episode()
                current_state_dict, _ = self.env.reset()
                current_state_tensor = obs_dict_to_tensor(observation=current_state_dict, 
                                                           obs_type=self.obs_type, 
                                                           n_agents=n_agents,
                                                           max_depth=self.max_depth, 
                                                           n_nodes=self.controller.config['n_nodes'])
                episode_step = 0
                self.total_episodes += 1

                self.logging_queue.put({'worker_id': self.worker_id,
                                        'episode': self.total_episodes,
                                        'episode/reward': self.rollout.episodes[-1]['average_episode_reward'],
                                        'episode/average_length': self.rollout.episodes[-1]['average_episode_length'],})            

        # TODO: add IMPALA vtrace correction
        self._generalised_advantage_estimator()
        self.rollout_queue.put(self.rollout)


    def _generalised_advantage_estimator(self) -> Tensor:
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
        with torch.no_grad():
            for idx, episode in enumerate(self.rollout.episodes):
                for agent in self.rollout.agent_handles:
                    states = torch.stack(episode['states'][agent])
                    next_states = torch.stack(episode['next_states'][agent])
                    rewards = torch.tensor(episode['rewards'][agent])
                    dones = torch.tensor(episode['dones'][agent]).float()
                    dones = (dones != 0).float()
                    gaes = torch.zeros(len(states))

                    state_values: Tensor = self.controller.critic_network(states).squeeze(-1)
                    next_state_values: Tensor = self.controller.critic_network(next_states).squeeze(-1)

                    self.rollout.episodes[idx]['state_values'][agent] = state_values
                    self.rollout.episodes[idx]['next_state_values'][agent] = next_state_values

                    deltas = rewards + self.controller.gamma * next_state_values * (1 - dones) - state_values

                    for i in reversed(range(len(deltas))):
                        gaes[i] = deltas[i] + self.controller.gamma * gaes[i + 1] * (1 - dones[i]) if i < len(deltas) - 2 else deltas[i]

                    self.rollout.episodes[idx]['gaes'][agent] = gaes

    def _try_refresh_weights(self):
        updated = False
        while True: 
            try:
                state = self.weights_queue.get_nowait()
                self.controller.update_weights(state)
                updated = True
            except queue.Empty:
                break
        return updated

