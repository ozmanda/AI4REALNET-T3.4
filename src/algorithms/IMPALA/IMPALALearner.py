import os
import sys
import time
import wandb
import queue
import argparse
import numpy as np
from itertools import chain
from typing import List, Dict, Union, Tuple

import torch
from torch import Tensor
import torch.optim as optim
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.managers import DictProxy

from src.algorithms.IMPALA.IMPALAWorker import IMPALAWorker
from src.configs.ControllerConfigs import PPOControllerConfig
from src.controllers.PPOController import PPOController
from src.controllers.LSTMController import LSTMController
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer
from src.algorithms.loss import vtrace

from flatland.envs.rail_env import RailEnv
from src.configs.EnvConfig import FlatlandEnvConfig

class IMPALALearner(): 
    """
    Learner class for the IMPALA Algorithm.
    # TODO: generalise from PPOControllerConfig to base class
    """
    def __init__(self, controller_config: PPOControllerConfig, learner_config: Dict, env_config: FlatlandEnvConfig, device: str = None) -> None:
        # Initialise environment and set controller / learning parameters
        self.env_config = env_config
        self._init_learning_params(learner_config)
        self._init_controller(controller_config)

        # Parallelisation Configuration
        self.n_workers: int = learner_config['n_workers'] 
        self._init_queues()

        # Initialise the optimiser
        self.optimizer: optim.Optimizer = self._build_optimizer(learner_config['optimiser_config'])
        self.update_step: int = 0

        # Initialise wandb for logging
        self._init_wandb(learner_config)


    def _init_controller(self, config: PPOControllerConfig) -> None:
        self.controller_config = config
        self.n_nodes: int = config.config_dict['n_nodes']
        self.state_size: int = config.config_dict['state_size']
        self.entropy_coeff: float = config.config_dict['entropy_coefficient']
        self.value_loss_coeff: float = config.config_dict['value_loss_coefficient']
        self.gamma: float = config.config_dict['gamma']
        self.controller: Union[PPOController, LSTMController] = config.create_controller()


    def _init_learning_params(self, learner_config: Dict) -> None:
        self.max_steps: int = learner_config['max_steps']
        self.max_steps_per_episode: int = learner_config['max_steps_per_episode']
        self.target_updates: int = learner_config['target_updates']
        self.samples_per_update: int = learner_config['samples_per_update']
        self.completed_updates: int = 0
        self.total_steps: int = 0
        self.iterations: int = learner_config['training_iterations']
        self.batch_size: int = learner_config['batch_size']
        self.importance_sampling: bool = learner_config['IS']
        self.episodes_infos: List[Dict] = []
        self.total_episodes: int = 0
        self.c_bar: float = learner_config['c_bar']
        self.rho_bar: float = learner_config['rho_bar']


    def _init_wandb(self, learner_config: Dict) -> None:
        """
        Initialize Weights & Biases for logging.
        """
        wandb.init(project='AI4REALNET-T3.4', entity='CLS-FHNW', config=learner_config, reinit=True)
        wandb.run.define_metric('episodes/*', step_metric='episode')
        wandb.run.define_metric('train/*', step_metric='update_step')
        wandb.run.name = f"{learner_config['run_name']}_IMPALA"
        wandb.run.save()
        self.controller.init_wandb()


    def _init_queues(self) -> None:
        # create queues
        self.logging_queue: mp.Queue = mp.Queue()
        self.rollout_queue: mp.Queue = mp.Queue()
        self.weights_queue: mp.Queue = mp.Queue()
        self.manager = mp.Manager()
        self.shared_weights: DictProxy = self.manager.dict()
        self.barrier = mp.Barrier(self.n_workers + 1)  # +1 for the learner
        self.done_event: Event = mp.Event()


    def _build_optimizer(self, optimiser_config: Dict[str, Union[int, str]]) -> optim.Optimizer:
        # TODO: how to generalise to other controllers?
        if optimiser_config['type'] == 'adam': 
            self.optimiser = optim.Adam(params=self.controller.get_parameters(), lr=float(optimiser_config['learning_rate']))
        else: 
            raise Warning('Only Adam optimiser has been implemented')


    def async_run(self) -> None: 
        """
        Asynchronous IMPALA training run. The learner will continuously collect rollouts from the workers to perform 
        policy updates. The workers will not wait for the new policy to continue rollout generation and are fully
        decoupled. This means that there are never idle workers, however they may be using a multiple update-old policy
        """
        # Broadcast initial weights to all workers, one for each worker, ensuring they start with the same model parameters
        # TODO: check if this is desirable (different initial starting points could be beneficial)
        self._broadcast_controller_state()

        # initialise learning rollout
        self.rollout = MultiAgentRolloutBuffer(n_agents=self.env_config.n_agents)

        # create and start workers
        # TODO: add device specification for the workers
        mp.set_start_method('spawn', force=True)  # parallelisation of rollout gathering - spawn is safer for pytorch
        workers: List[IMPALAWorker] = []
        for worker_id in range(self.n_workers):
            worker = IMPALAWorker(worker_id=worker_id,
                               logging_queue=self.logging_queue,
                               rollout_queue=self.rollout_queue,
                               shared_weights=self.shared_weights,
                               barrier=self.barrier,
                               done_event=self.done_event,
                               env_config=self.env_config,
                               controller_config=self.controller_config,
                               max_steps=(self.max_steps, self.max_steps_per_episode),
                               device='cpu')
            workers.append(worker)
            print(f'Starting worker {worker_id}')
            worker.start()

        # gather rollouts and update when enough data is collected
        while self.completed_updates < self.target_updates:
            # gather rollouts from workers
            try: 
                self._gather_rollouts()
            except queue.Empty:
                print("No rollouts received from workers.")
                continue

            # add the episode to the current rollout buffer
            if self.rollout.total_steps > self.samples_per_update:
                # update the controller with the current rollout
                self._optimise(rollout = self.rollout)
                self.completed_updates += 1
                print(f'\n\nCompleted Updates: {self.completed_updates} / {self.target_updates}\n\n')
                
                wandb.log({
                    'train/update': self.completed_updates,
                    'train/samples_this_update': self.rollout.total_steps
                })

                # reset rollout
                self.rollout.reset(n_agents=self.env_config.n_agents)

                # broadcast updated controller weights
                self._broadcast_controller_state()


        self.done_event.set()
        print("Training complete, waiting for workers to terminate...")
        # self.barrier.wait()  # ensure all workers have finished

        # drain any final episode info
        print("Draining final episode info...")
        self._log_episode_info()

        print('terminating workers...')
        # terminate all workers
        for w in workers:
            if w.is_alive():
                w.terminate()
        
        print("All workers terminated. Finishing WandB run and shutting down manager...")
        wandb.finish()
        self.manager.shutdown()


    def _optimise(self, rollout: MultiAgentRolloutBuffer) -> Dict[str, List[float]]:
        """
        Optimise the model parameters using the collected rollouts.
        """
        total_loss, loss_dict = self._loss(rollout, self.batch_size)

        self.optimiser.zero_grad()
        total_loss.backward()
        self.optimiser.step()
        self.update_step += 1

        # log to wandb
        wandb_log_dict = {'update_step': self.update_step}
        for loss_key in loss_dict.keys():
            wandb_log_dict[loss_key] = loss_dict[loss_key].mean().item()
        wandb.log(wandb_log_dict)
    

    def _loss(self, rollout: MultiAgentRolloutBuffer, batch_size: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the actor and critic loss for the given rollout buffer considering v-trace correction.
        """
        policy_loss_sum = torch.tensor(0.0)
        value_loss_sum = torch.tensor(0.0)
        entropy_sum = torch.tensor(0.0)
        entropy_count = 0
        total_steps = 0

        for episode in rollout.episodes:
            for agent in range(rollout.n_agents):
                states = torch.stack(episode['states'][agent])  # (episode_length, state_size)
                next_states = torch.stack(episode['next_states'][agent])  # (episode_length, state_size)
                actions = torch.stack(episode['actions'][agent])  # (episode_length, 1)
                behaviour_log_probs = torch.stack(episode['log_probs'][agent])  # (episode_length, 1)
                rewards = torch.tensor(episode['rewards'][agent])  # (episode_length,)
                dones = torch.tensor(episode['dones'][agent])  # (episode_length,)

                target_log_probs, target_state_values, target_next_state_values = self.controller.evaluate(states=states, actions=actions, next_states=next_states)

                v_s, pg_adv = vtrace(
                    behaviour_log_probs=behaviour_log_probs,
                    target_log_probs=target_log_probs,
                    rewards=rewards,
                    state_values=target_state_values,
                    next_state_values=target_next_state_values,
                    dones=dones,
                    gamma=self.gamma,
                    rho_bar=self.rho_bar,
                    c_bar=self.c_bar
                )
            
                total_steps += rewards.size(0)

                policy_loss = (-pg_adv * target_log_probs).sum()
                policy_loss_sum += policy_loss

                value_loss = 0.5 * torch.mean((v_s.detach() - target_state_values)**2)
                value_loss_sum += value_loss

                entropy = -(target_log_probs.exp() * target_log_probs).sum(-1)
                entropy_sum += entropy.sum()
                entropy_count += 1

        # normalize losses by total steps
        policy_loss_sum /= total_steps
        value_loss_sum /= total_steps
        entropy_mean = entropy_sum/entropy_count 

        # calculate total training loss
        total_loss = policy_loss_sum + self.value_loss_coeff * value_loss_sum - self.entropy_coeff * entropy_mean

        return total_loss, {"policy_loss": policy_loss_sum,
                            "value_loss": value_loss_sum,
                            "entropy_loss": entropy_mean}


    def _gather_rollouts(self) -> None:
        """ Gather episodes from rollout queue and add to training rollout. """
        for _ in range(self.n_workers):
            try:
                worker_rollout: Dict[str, List] = self.rollout_queue.get(timeout=60)
                self.rollout.add_episode(worker_rollout)
                self._log_episode_info()
            except queue.Empty:
                break
            except MemoryError:
                pass


    def _broadcast_controller_state(self) -> None:
        """ Push the current controller state to the shared dictionary for workers to access. """
        controller_state: Dict = self.controller.get_state_dict()
        self.shared_weights['controller_state'] = controller_state
        self.shared_weights['update_step'] = self.update_step

        
    def _log_episode_info(self):
        """ Log episode information to Weights & Biases. """
        try: 
            while True: 
                log_info = self.logging_queue.get_nowait()
                worker_id = log_info['worker_id']
                wandb.log({
                    f'worker_{worker_id}/episode_reward': log_info['episode/reward'],
                    f'worker_{worker_id}/episode_length': log_info['episode/average_length'],
                    f'worker_{worker_id}/total_reward': log_info['episode/total_reward'],
                    # global aggregated metrics
                    'episode/average_reward': log_info['episode/reward'],
                })
        except queue.Empty:
            pass
