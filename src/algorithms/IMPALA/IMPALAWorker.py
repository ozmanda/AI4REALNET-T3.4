import queue
from collections import namedtuple
from typing import Tuple, Dict, Union
from flatland.envs.rail_env import RailEnv

import torch
from torch import Tensor
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event

from src.utils.obs_utils import obs_dict_to_tensor
from src.configs.EnvConfig import FlatlandEnvConfig
from src.algorithms.PPO.PPOController import PPOController
from src.configs.ControllerConfigs import PPOControllerConfig
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer

class IMPALAWorker(mp.Process):
    """
    Worker class that inherits from torch.multiprocessing.Process, meaning that when the .start() method is called on PPOWorker,
    the entry point is the run() function. This 
    """

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
        self._try_refresh_weights()

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
            actions, log_probs = self.controller.sample_action(current_state_tensor)
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
            self.rollout.add_transitions(states=current_state_tensor.detach(), 
                                         actions=actions_dict, 
                                         log_probs=log_probs.detach(), 
                                         rewards=rewards, 
                                         next_states=next_state_tensor.detach(), 
                                         dones=dones)

            current_state_tensor = next_state_tensor
            episode_step += 1

            if all(dones.values()) or episode_step >= self.max_steps_per_episode:
                self._generalised_advantage_estimator()
                # TODO: add IMPALA vtrace correction
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

                # TODO: ensure only new weights taken
                self._vtrace_correction
                updated = self._try_refresh_weights()


    def _vtrace_correction(self) -> None:
        """
        Apply v-trace correction to the rollout buffer.
        """
        traj_len = self.rollout.total_steps

        



    def _try_refresh_weights(self):
        updated = False
        while True: 
            try:
                state = self.weights_queue.get_nowait()
                self.controller.update_weights(state)
                updated = True
                print(f'Worker {self.worker_id} updated weights')
                break
            except queue.Empty:
                break
        return updated