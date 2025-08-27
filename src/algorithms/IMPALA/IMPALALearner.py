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

from src.algorithms.IMPALA.IMPALAWorker import IMPALAWorker
from src.configs.ControllerConfigs import PPOControllerConfig
from src.algorithms.PPO.PPOController import PPOController
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer
from src.training.loss import value_loss, value_loss_with_IS, policy_loss

from flatland.envs.rail_env import RailEnv
from src.configs.EnvConfig import FlatlandEnvConfig

class IMPALALearner(): 
    """
    Learner class for the IMPALA Algorithm.
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
        wandb.watch(self.controller.actor_network, log='all')
        wandb.watch(self.controller.critic_network, log='all')

    def _init_controller(self, config: PPOControllerConfig) -> None:
        self.controller_config = config
        self.n_nodes: int = config.config_dict['n_nodes']
        self.state_size: int = config.config_dict['state_size']
        self.entropy_coeff: float = config.config_dict['entropy_coefficient']
        self.value_loss_coeff: float = config.config_dict['value_loss_coefficient']
        self.gamma: float = config.config_dict['gamma']
        self.controller = config.create_controller()

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

    def _init_wandb(self, learner_config: Dict) -> None:
        """
        Initialize Weights & Biases for logging.
        """
        wandb.init(project='AI4REALNET-T3.4', entity='CLS-FHNW', config=learner_config, reinit=True)
        wandb.run.define_metric('episodes/*', step_metric='episode')
        wandb.run.define_metric('train/*', step_metric='update_step')
        wandb.run.name = f"{learner_config['run_name']}_IMPALA"
        wandb.run.save()
        wandb.watch(self.controller.actor_network, log='all')
        wandb.watch(self.controller.critic_network, log='all')

    def _init_queues(self) -> None:
        # create queues
        self.logging_queue: mp.Queue = mp.Queue()
        self.rollout_queue: mp.Queue = mp.Queue()
        self.weights_queue: mp.Queue = mp.Queue()
        self.done_event: Event = mp.Event()

    def _build_optimizer(self, optimiser_config: Dict[str, Union[int, str]]) -> optim.Optimizer:
        if optimiser_config['type'] == 'adam': 
            self.optimiser = optim.Adam(params=chain(self.controller.actor_network.parameters(), self.controller.critic_network.parameters()), lr=float(optimiser_config['learning_rate']))
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
        controller_state = (self.controller.actor_network.state_dict(),
                            self.controller.critic_network.state_dict())
        for worker in range(self.n_workers):
            self.weights_queue.put(controller_state)
            # TODO: ensure that workers only update when there are new weights

        # initialise learning rollout
        self.rollout = MultiAgentRolloutBuffer(n_agents=self.env_config.n_agents)

        # create and start workers
        # TODO: add device specification for the workers
        mp.set_start_method('spawn')  # parallelisation of rollout gathering - spawn is safer for pytorch
        workers: List[IMPALAWorker] = []
        for worker_id in range(self.n_workers):
            worker = IMPALAWorker(worker_id=worker_id,
                               logging_queue=self.logging_queue,
                               rollout_queue=self.rollout_queue,
                               weights_queue=self.weights_queue,
                               done_event=self.done_event,
                               env_config=self.env_config,
                               controller_config=self.controller_config,
                               max_steps=(self.max_steps, self.max_steps_per_episode),
                               device='cpu')
            workers.append(worker)
            worker.start()

        # gather rollouts and update when enough data is collected
        while self.completed_updates < self.target_updates:
            # check episode logging
            # self._log_episode_info()

            # gather rollouts from workers
            try: 
                self._gather_rollouts()
            except queue.Empty:
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
                controller_state = (self.controller.actor_network.state_dict(),
                                    self.controller.critic_network.state_dict())
                for worker in workers:
                    worker.weights_queue.put(controller_state)


        self.done_event.set()
        time.sleep(1)

        # drain any final episode info
        self._log_episode_info()

        # terminate all workers
        #! TODO: there is something wrong here, the workers never finish!
        for w in workers:
            w.join()
        for w in workers:
            if w.is_alive():
                w.terminate()
        
        wandb.finish()


    def _optimise(self, rollout: MultiAgentRolloutBuffer) -> Dict[str, List[float]]:
        """
        Optimise the model parameters using the collected rollouts.
        """
        losses = {
            'policy_loss': [],
            'value_loss': []
        }

        total_loss, actor_loss, critic_loss = self._loss(rollout, self.batch_size)

        self.optimiser.zero_grad()
        total_loss.backward()
        self.optimiser.step()
        self.update_step += 1

        # add metrics
        losses['policy_loss'].append(actor_loss.mean().item())
        losses['value_loss'].append(critic_loss.mean().item())

        # Log losses to wandb
        wandb.log({
            'update_step': self.update_step,
            'train/policy_loss': np.mean(losses['policy_loss']),
            'train/value_loss': np.mean(losses['value_loss'])
        })
        return losses
    

    def _loss(self, rollout: MultiAgentRolloutBuffer, batch_size: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the actor and critic loss for the given rollout buffer considering v-trace correction. 
        """