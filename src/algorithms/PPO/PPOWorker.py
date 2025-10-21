from collections import namedtuple
from typing import Tuple, Dict, Union
from flatland.envs.rail_env import RailEnv

import queue
import torch
from torch import Tensor
import torch.multiprocessing as mp
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Event

from src.utils.observation.obs_utils import obs_dict_to_tensor
from src.utils.observation.normalisation import FlatlandNormalisation
from src.utils.observation.RunningMeanStd import RunningMeanStd

from src.configs.EnvConfig import FlatlandEnvConfig
from src.controllers.PPOController import PPOController
from src.configs.ControllerConfigs import ControllerConfig
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer

Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'reward', 'next_state', 'done', 'info'))

class PPOWorker(mp.Process): 
    """
    Worker class that inherits from torch.multiprocessing.Process, meaning that when the .start() method is called on PPOWorker,
    the entry point is the run() function. This 
    """
    def __init__(self, worker_id: Union[str, int], env_config: FlatlandEnvConfig, controller_config: ControllerConfig, logging_queue: mp.Queue, 
                 rollout_queue: mp.Queue, barrier, shared_weights, done_event: Event, max_steps: Tuple = (10000, 1000), device: str = 'cpu'):
        super().__init__()

        # multiprocessing setup
        self.worker_id: Union[str, int] = worker_id
        self.logging_queue: mp.Queue = logging_queue
        self.rollout_queue: mp.Queue = rollout_queue
        # self.observation_queue: mp.Queue = observation_queue
        self.shared_weights: DictProxy = shared_weights
        # self.shared_normalisation: DictProxy = shared_normalisation
        self.barrier = barrier
        self.done_event: Event = done_event
        self.device: str = device
        self.local_update_step: int = -1

        # env and controller setup
        self.env_config: FlatlandEnvConfig = env_config
        self._init_env()
        self.controller_config: ControllerConfig = controller_config
        self.controller: PPOController = self.controller_config.create_controller()

        # rollout setup
        self.max_steps: int = max_steps[0]
        self.max_steps_per_episode: int = max_steps[1]
        self.rollout: MultiAgentRolloutBuffer = MultiAgentRolloutBuffer(n_agents=self.env.number_of_agents)

        # normalisation setup
        self._init_normalisation()


    def _init_env(self) -> None:
        """ Initialize the environment based on the provided configuration. """
        self.obs_type: str = self.env_config.observation_builder_config['type']
        self.max_depth: int = self.env_config.observation_builder_config['max_depth']
        self.env: RailEnv = self.env_config.create_env()
        self.n_agents: int = self.env.number_of_agents  


    def _init_normalisation(self) -> None:
        """ Normalisation setup """
        self.flatland_normalisation: FlatlandNormalisation = FlatlandNormalisation(
            n_nodes=self.controller.config['n_nodes'],
            n_features=self.controller.config['n_features'],
            n_agents=self.env.number_of_agents,
            env_size=(self.env_config.width, self.env_config.height)
        )


    def run(self) -> None:
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
        # self.push_distance_obs(self._get_distance_obs(observation=current_state_tensor))
        # self._wait_for_normalisation()
        current_state_tensor = self.normalise_observation(current_state_tensor)
        dones = {i: False for i in range(n_agents)}

        while not self.done_event.is_set():
            actions, log_probs, values, extras = self.controller.sample_action(current_state_tensor)
            actions_dict: Dict[Union[int, str], Tensor] = {}
            
            # TODO: consider agents which have already terminated
            for i in range(self.env.number_of_agents):
                actions_dict[i] = int(actions[i].detach()) if not dones[i] else None

            # step returns: tuple[Dict[AgentHandle, Any], dict[int, float], dict[str | int, bool], dict[str, Any]]
            next_state, rewards, dones, infos = self.env.step(actions_dict)
            next_state_tensor: Tensor = obs_dict_to_tensor(observation=next_state, 
                                                           obs_type=self.obs_type, 
                                                           n_agents=n_agents,
                                                           max_depth=self.max_depth, 
                                                           n_nodes=self.controller.config['n_nodes'])
            next_state_tensor = self.normalise_observation(next_state_tensor)
            next_state_values = self.controller.state_values(next_state_tensor, extras=extras)
            self.rollout.add_transitions(states=current_state_tensor.detach(), 
                                         actions=actions.detach(), 
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
                # TODO: add running mean and standard deviation information for normalisation
                # self.push_distance_obs(self._get_distance_obs())
                # self._wait_for_normalisation()
                self.rollout.end_episode()
                current_state_dict, _ = self.env.reset()
                current_state_tensor = obs_dict_to_tensor(observation=current_state_dict, 
                                                           obs_type=self.obs_type, 
                                                           n_agents=n_agents,
                                                           max_depth=self.max_depth, 
                                                           n_nodes=self.controller.config['n_nodes'])
                current_state_tensor = self.normalise_observation(current_state_tensor)
                episode_step = 0
                self.total_episodes += 1

                self.rollout_queue.put(self.rollout.episodes[-1])
                self.logging_queue.put({'worker_id': self.worker_id,
                                        'episode': self.total_episodes,
                                        'episode/total_reward': self.rollout.episodes[-1]['total_reward'],
                                        'episode/average_reward': self.rollout.episodes[-1]['average_episode_reward'],
                                        'episode/average_length': self.rollout.episodes[-1]['average_episode_length'],
                                        'episode/completion': sum([dones[agent] for agent in range(self.n_agents)]) / self.n_agents})
                if not self.done_event.is_set():
                    self._wait_for_weights()

    def normalise_observation(self, observation: Tensor) -> Tensor:
        """
        Normalize the observation using the FlatlandNormalisation class.
        """
        normalized_observation = self.flatland_normalisation.normalise(observation.unsqueeze(0))
        return normalized_observation.squeeze(0)

    def _wait_for_weights(self):
        """
        Wait for the learner to update weights, blocking until new weights are available.
        This ensures rollouts are synchronous across all workers.
        """
        self.barrier.wait()
        if self.shared_weights['update_step'] > self.local_update_step:
            self.local_update_step = self.shared_weights['update_step']
            self.controller.update_weights(self.shared_weights['controller_state'])

# TODO: finish implementing running mean and std calculation over all workers
    # def push_distance_obs(self, distance_obs: Tensor) -> None:
    #     """ Push distance observations to the learner. """ 
    #     self.observation_queue.put(distance_obs) 
    #     print(f"Worker {self.worker_id} pushed observations...")
    #     self.barrier.wait()

    # def _get_distance_obs(self, observation: Tensor = None, episode: bool = True) -> Tensor:
    #     """ Extract distance observations from the full observation tensor. """
    #     if episode: 
    #         observation = torch.stack([torch.stack(self.rollout.current_episode['states'][agent]) for agent in range(self.env.number_of_agents)])

    #     if observation.dim() == 2:
    #         obs_view = observation.view(observation.shape[0], self.controller.config['n_nodes'], self.controller.config['n_features']).detach()
    #         distance_obs = obs_view[:, :, :7] 
    #     elif observation.dim() == 3:
    #         obs_view = observation.view(observation.shape[0], self.env.number_of_agents, self.controller.config['n_nodes'], self.controller.config['n_features']).detach()
    #         distance_obs = obs_view[:, :, :, :7]  # Assuming distance features are the first 7 features
    #     else:
    #         raise ValueError("Observation tensor has an unsupported number of dimensions.")
        
    #     return distance_obs.view(-1, 7)  # Flatten to (batch_size * n_agents * n_nodes, 7)

    # def _wait_for_normalisation(self):
    #     """
    #     Wait for the normalisation process to update normalisation statistics.
    #     """
    #     print(f"Worker {self.worker_id} waiting at barrier before normalisation update...")
    #     self.barrier.wait()
    #     distance_rms: RunningMeanStd = self.shared_normalisation.get('distance_rms', None)
    #     if not distance_rms:
    #         raise ValueError("Running mean and std not found in shared normalisation dictionary.")
    #     self.flatland_normalisation.distance_rms = distance_rms
