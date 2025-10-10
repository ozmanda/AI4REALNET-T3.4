from collections import namedtuple
from typing import Tuple, Dict, Union, List
from flatland.envs.rail_env import RailEnv

import queue
import numpy as np
import torch
from torch import Tensor
import torch.multiprocessing as mp
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Event

from src.utils.observation.obs_utils import obs_dict_to_tensor
from src.utils.observation.normalisation import FlatlandNormalisation, IdentityNormalisation
from src.utils.observation.RunningMeanStd import RunningMeanStd

from src.configs.EnvConfig import FlatlandEnvConfig, PettingZooEnvConfig
from src.controllers.PPOController import PPOController
from src.configs.ControllerConfigs import ControllerConfig
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer

Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'reward', 'next_state', 'done', 'info'))

class PPOWorker(mp.Process): 
    """
    Worker class that inherits from torch.multiprocessing.Process, meaning that when the .start() method is called on PPOWorker,
    the entry point is the run() function. This 
    """
    def __init__(self, worker_id: Union[str, int], env_config: Union[FlatlandEnvConfig, PettingZooEnvConfig], controller_config: ControllerConfig, logging_queue: mp.Queue, 
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
        self.env_type: str = getattr(self.env_config, 'env_type', 'flatland')
        self._init_env()
        self.controller_config: ControllerConfig = controller_config
        self.controller: PPOController = self.controller_config.create_controller()

        # rollout setup
        self.max_steps: int = max_steps[0]
        self.max_steps_per_episode: int = max_steps[1]
        self.rollout: MultiAgentRolloutBuffer = MultiAgentRolloutBuffer(n_agents=self.n_agents)

        # normalisation setup
        self._init_normalisation()


    def _init_env(self) -> None:
        """ Initialize the environment based on the provided configuration. """
        self.env: RailEnv = self.env_config.create_env()
        if self.env_type == 'flatland':
            self.obs_type: str = self.env_config.observation_builder_config['type']
            self.max_depth: int = self.env_config.observation_builder_config['max_depth']
            self.agent_ids: List[Union[int, str]] = list(range(self.env_config.get_num_agents()))
        else:
            agent_ids = list(getattr(self.env, 'possible_agents', []))
            if not agent_ids:
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    reset_obs = reset_result[0]
                else:
                    reset_obs = reset_result
                agent_ids = list(reset_obs.keys())
            self.agent_ids = agent_ids
            setattr(self.env_config, 'agent_ids', self.agent_ids)
        self.n_agents = len(self.agent_ids)
        if self.n_agents == 0:
            raise ValueError("Unable to determine agent identifiers for the environment.")


    def _init_normalisation(self) -> None:
        """ Normalisation setup """
        if self.env_type == 'flatland':
            self.normalisation = FlatlandNormalisation(
                n_nodes=self.controller.config['n_nodes'],
                n_features=self.controller.config['n_features'],
                n_agents=self.n_agents,
                env_size=(self.env_config.width, self.env_config.height)
            )
        else:
            self.normalisation = IdentityNormalisation()


    def run(self) -> None:
        """
        Entry point for worker.run()
        Run a single episode in the environment and collect rollouts.
        """
        reset_result = self.env.reset()
        current_state_dict = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        self.rollout.reset(n_agents=self.n_agents)
        self._wait_for_weights()

        episode_step = 0
        self.total_episodes = 0
        current_state_tensor: Tensor = self._observation_to_tensor(current_state_dict)
        current_state_tensor = self.normalise_observation(current_state_tensor)

        while not self.done_event.is_set():
            actions, log_probs, values, extras = self.controller.sample_action(current_state_tensor)
            actions_dict = self._build_action_dict(actions)

            step_results = self.env.step(actions_dict)
            next_state_dict, rewards_dict, dones_dict, infos = self._process_step_output(step_results)
            next_state_tensor: Tensor = self._observation_to_tensor(next_state_dict)
            next_state_tensor = self.normalise_observation(next_state_tensor)
            next_state_values = self.controller.state_values(next_state_tensor, extras=extras).detach()

            rewards_index = self._dict_to_index_dict(rewards_dict, default=0.0)
            dones_index = self._dict_to_index_dict(dones_dict, default=False)

            self.rollout.add_transitions(states=current_state_tensor.detach(),
                                         actions=actions.detach(),
                                         log_probs=log_probs.detach(),
                                         rewards=rewards_index,
                                         next_states=next_state_tensor.detach(),
                                         state_values=values.squeeze(-1).detach(),
                                         next_state_values=next_state_values.squeeze(-1),
                                         dones=dones_index,
                                         extras={})

            current_state_tensor = next_state_tensor
            episode_step += 1

            episode_done = dones_dict.get('__all__', all(dones_dict.get(agent_id, False) for agent_id in self.agent_ids))
            if episode_done or episode_step >= self.max_steps_per_episode:
                self.rollout.end_episode()
                reset_result = self.env.reset()
                current_state_dict = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                current_state_tensor = self._observation_to_tensor(current_state_dict)
                current_state_tensor = self.normalise_observation(current_state_tensor)
                episode_step = 0
                self.total_episodes += 1

                self.rollout_queue.put(self.rollout.episodes[-1])
                self.logging_queue.put({'worker_id': self.worker_id,
                                        'episode': self.total_episodes,
                                        'episode/total_reward': self.rollout.episodes[-1]['total_reward'],
                                        'episode/average_reward': self.rollout.episodes[-1]['average_episode_reward'],
                                        'episode/average_length': self.rollout.episodes[-1]['average_episode_length'],})
                if not self.done_event.is_set():
                    self._wait_for_weights()

    def normalise_observation(self, observation: Tensor) -> Tensor:
        """
        Normalize the observation using the FlatlandNormalisation class.
        """
        normalized_observation = self.normalisation.normalise(observation.unsqueeze(0))
        return normalized_observation.squeeze(0)

    def _observation_to_tensor(self, observation: Dict[Union[int, str], Union[np.ndarray, Tensor]]) -> Tensor:
        """
        Convert environment observations to a tensor with shape (n_agents, state_size).
        """
        if self.env_type == 'flatland':
            tensor = obs_dict_to_tensor(observation=observation,
                                        obs_type=self.obs_type,
                                        n_agents=self.n_agents,
                                        max_depth=self.max_depth,
                                        n_nodes=self.controller.config['n_nodes'])
            return tensor.view(self.n_agents, -1)

        state_shape = getattr(self.env_config, 'state_shape', None)
        if state_shape is None:
            sample_obs = next(iter(observation.values()), None)
            if sample_obs is not None:
                state_shape = np.array(sample_obs).shape
                setattr(self.env_config, 'state_shape', state_shape)
        obs_tensors: List[Tensor] = []
        for agent_id in self.agent_ids:
            agent_obs = observation.get(agent_id)
            if agent_obs is None:
                if state_shape is None:
                    agent_obs = np.zeros(self.controller.config['state_size'], dtype=np.float32)
                else:
                    agent_obs = np.zeros(state_shape, dtype=np.float32)
            agent_array = np.asarray(agent_obs)
            if agent_array.dtype == np.uint8:
                agent_array = agent_array.astype(np.float32) / 255.0
            else:
                agent_array = agent_array.astype(np.float32)
            obs_tensors.append(torch.from_numpy(agent_array).flatten())
        return torch.stack(obs_tensors, dim=0)

    def _build_action_dict(self, actions: Tensor) -> Dict[Union[int, str], int]:
        """
        Convert tensorised actions into the environment-specific dictionary format.
        """
        action_dict: Dict[Union[int, str], int] = {}
        active_agents = getattr(self.env, 'agents', self.agent_ids) if self.env_type != 'flatland' else self.agent_ids
        for idx, agent_id in enumerate(self.agent_ids):
            if self.env_type != 'flatland' and agent_id not in active_agents:
                continue
            action_value = actions[idx].item() if isinstance(actions[idx], Tensor) else actions[idx]
            key = agent_id if self.env_type != 'flatland' else agent_id
            action_dict[key] = int(action_value)
        return action_dict

    def _process_step_output(self, step_results):
        """
        Harmonise environment step outputs across supported environment types.
        """
        if self.env_type == 'flatland':
            next_state, rewards, dones, infos = step_results
            filtered_dones = {agent_id: dones.get(agent_id, False) for agent_id in self.agent_ids}
            filtered_dones['__all__'] = dones.get('__all__', all(filtered_dones.values()))
            return next_state, rewards, filtered_dones, infos

        next_state, rewards, terminations, truncations, infos = step_results
        dones: Dict[Union[int, str], bool] = {}
        for agent_id in self.agent_ids:
            dones[agent_id] = bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
        dones['__all__'] = bool(terminations.get('__all__', all(dones.values())) or truncations.get('__all__', False))
        return next_state, rewards, dones, infos

    def _dict_to_index_dict(self, values: Dict[Union[int, str], Union[float, bool]], default: Union[float, bool] = 0.0) -> Dict[int, Union[float, bool]]:
        """
        Convert dictionaries keyed by agent identifiers to dictionaries keyed by agent index.
        """
        indexed: Dict[int, Union[float, bool]] = {}
        for idx, agent_id in enumerate(self.agent_ids):
            key = agent_id if self.env_type != 'flatland' else agent_id
            if isinstance(values, dict):
                value = values.get(key, default)
            else:
                value = values[idx]
            if isinstance(default, bool):
                indexed[idx] = bool(value)
            else:
                indexed[idx] = float(value)
        return indexed


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
