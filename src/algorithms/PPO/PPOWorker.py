from collections import namedtuple
from typing import Tuple, Dict, Union
from flatland.envs.rail_env import RailEnv

import queue
import torch
from torch import Tensor
import torch.multiprocessing as mp
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Event

from src.utils.obs_utils import obs_dict_to_tensor
from src.configs.EnvConfig import FlatlandEnvConfig
from src.controllers.PPOController import PPOController
from src.configs.ControllerConfigs import PPOControllerConfig
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer

Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'reward', 'next_state', 'done', 'info'))

class PPOWorker(mp.Process): 
    """
    Worker class that inherits from torch.multiprocessing.Process, meaning that when the .start() method is called on PPOWorker,
    the entry point is the run() function. This 
    """

    def __init__(self, worker_id: Union[str, int], env_config: FlatlandEnvConfig, controller_config: PPOControllerConfig, 
                 logging_queue: mp.Queue, rollout_queue: mp.Queue, barrier, shared_weights, done_event: Event,
                 max_steps: Tuple = (10000, 1000), device: str = 'cpu'):
        super().__init__()

        # multiprocessing setup
        self.worker_id: Union[str, int] = worker_id
        self.logging_queue: mp.Queue = logging_queue
        self.rollout_queue: mp.Queue = rollout_queue
        self.shared_weights: DictProxy = shared_weights
        self.barrier = barrier
        self.done_event: Event = done_event
        self.device: str = device
        self.local_update_step: int = -1

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
        self.obs_type: str = self.env_config.observation_builder_config['type']
        self.max_depth: int = self.env_config.observation_builder_config['max_depth']
        self.env: RailEnv = self.env_config.create_env()


    def run(self) -> MultiAgentRolloutBuffer:
        """
        Entry point for worker.run()
        Run a single episode in the environment and collect rollouts.
        """
        self.env.reset()
        self.rollout.reset(n_agents=self.env.number_of_agents)
        self._wait_for_weights()

        episode_step = 0
        self.total_episodes = 0
        current_state_dict, _ = self.env.reset()
        n_agents = self.env.number_of_agents
        current_state_tensor: Tensor = obs_dict_to_tensor(observation=current_state_dict, 
                                                          obs_type=self.obs_type, 
                                                          n_agents=n_agents,
                                                          max_depth=self.max_depth, 
                                                          n_nodes=self.controller.config['n_nodes'])

        while not self.done_event.is_set():
            actions, log_probs, values, extras = self.controller.sample_action(current_state_tensor)
            actions_dict: Dict[Union[int, str], Tensor] = {}
            
            # TODO: consider agents which have already terminated
            for i in range(self.env.number_of_agents):
                actions_dict[i] = actions[i].detach()

            next_state, rewards, dones, infos = self.env.step(actions_dict)
            next_state_tensor: Tensor = obs_dict_to_tensor(observation=next_state, 
                                                           obs_type=self.obs_type, 
                                                           n_agents=n_agents,
                                                           max_depth=self.max_depth, 
                                                           n_nodes=self.controller.config['n_nodes'])
            # TODO: add observation normalisation when passing to controller!
            next_state_values = self.controller.state_values(next_state_tensor, extras=extras)
            self.rollout.add_transitions(states=current_state_tensor.detach(), 
                                         actions=actions_dict, 
                                         log_probs=log_probs.detach(), 
                                         rewards=rewards, 
                                         next_states=next_state_tensor.detach(), 
                                         state_values=values.squeeze(-1).detach(),
                                         next_state_values=next_state_values.squeeze(-1).detach(),
                                         dones=dones,
                                         extras=extras)

            current_state_tensor = next_state_tensor
            episode_step += 1

            if all(dones.values()) or episode_step >= self.max_steps_per_episode:
                # self._generalised_advantage_estimator()
                self.rollout.end_episode()
                current_state_dict, _ = self.env.reset()
                current_state_tensor = obs_dict_to_tensor(observation=current_state_dict, 
                                                           obs_type=self.obs_type, 
                                                           n_agents=n_agents,
                                                           max_depth=self.max_depth, 
                                                           n_nodes=self.controller.config['n_nodes'])
                episode_step = 0
                self.total_episodes += 1

                self.rollout_queue.put(self.rollout.episodes[-1])
                self.logging_queue.put({'worker_id': self.worker_id,
                                        'episode': self.total_episodes,
                                        'episode/reward': self.rollout.episodes[-1]['average_episode_reward'],
                                        'episode/average_length': self.rollout.episodes[-1]['average_episode_length'],})
                # TODO: add mean and standard deviation information for normalisation (observation)
                if not self.done_event.is_set():
                    self._wait_for_weights()
        


    def _generalised_advantage_estimator(self) -> Tensor:
        """
        Calculate the Generalised Advantage Estimator (GAE) for the current episode (called before rollout.end_episode).

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
            for agent in range(self.env.number_of_agents):
                states = torch.stack(self.rollout.current_episode['states'][agent])
                next_states = torch.stack(self.rollout.current_episode['next_states'][agent])
                rewards = torch.tensor(self.rollout.current_episode['rewards'][agent])
                dones = torch.tensor(self.rollout.current_episode['dones'][agent]).float()
                dones = (dones != 0).float()
                gaes = torch.zeros(len(states))

                state_values: Tensor = self.controller.critic_network(states).squeeze(-1)
                next_state_values: Tensor = self.controller.critic_network(next_states).squeeze(-1)

                self.rollout.current_episode['state_values'][agent] = state_values
                self.rollout.current_episode['next_state_values'][agent] = next_state_values

                deltas = rewards + self.controller.gamma * next_state_values * (1 - dones) - state_values

                for i in reversed(range(len(deltas))):
                    gaes[i] = deltas[i] + self.controller.gamma * gaes[i + 1] * (1 - dones[i]) if i < len(deltas) - 2 else deltas[i]

                self.rollout.current_episode['gaes'][agent] = gaes


    def _wait_for_weights(self):
        """
        Wait for the learner to update weights, blocking until new weights are available.
        This ensures rollouts are synchronous across all workers.
        """
        self.barrier.wait()
        if self.shared_weights['update_step'] > self.local_update_step:
            self.local_update_step = self.shared_weights['update_step']
            self.controller.update_weights(self.shared_weights['controller_state'])