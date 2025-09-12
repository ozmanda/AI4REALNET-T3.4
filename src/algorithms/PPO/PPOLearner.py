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

from src.algorithms.PPO.PPOWorker import PPOWorker
from src.configs.ControllerConfigs import PPOControllerConfig
from src.controllers.PPOController import PPOController
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer
from src.algorithms.loss import value_loss, value_loss_with_IS, policy_loss

from flatland.envs.rail_env import RailEnv
from src.configs.EnvConfig import FlatlandEnvConfig

class PPOLearner():
    """
    Learner class for the PPO Algorithm.
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
        self.run_name = learner_config['run_name']
        wandb.init(project='AI4REALNET-T3.4', entity='CLS-FHNW', config=learner_config, reinit=True)
        wandb.run.define_metric('episodes/*', step_metric='episode')
        wandb.run.define_metric('train/*', step_metric='update_step')
        wandb.run.name = f"{self.run_name}_PPO"
        wandb.watch(self.controller.actor_network, log='all')
        wandb.watch(self.controller.critic_network, log='all')

    def _init_queues(self) -> None:
        # create queues
        self.logging_queue: mp.Queue = mp.Queue()
        self.rollout_queue: mp.Queue = mp.Queue()
        # self.weights_queue: mp.Queue = mp.Queue()
        self.barrier = mp.Barrier(self.n_workers + 1)  # +1 for the learner process
        self.manager = mp.Manager()
        self.done_event: Event = mp.Event()


    def sync_run(self) -> None:
        """
        Synchronous PPO training run.
        """
        # Broadcast initial weights to all workers, one for each worker, ensuring they start with the same model parameters
        self.shared_weights: DictProxy = self.manager.dict()
        self._broadcast_controller_state()

        # initialise learning rollout
        self.rollout = MultiAgentRolloutBuffer(n_agents=self.env_config.n_agents)

        # create and start workers
        # TODO: add device specification for the workers
        mp.set_start_method('spawn', force=True)  # parallelisation of rollout gathering - spawn is safer for pytorch
        workers: List[PPOWorker] = []
        for worker_id in range(self.n_workers):
            worker = PPOWorker(worker_id=worker_id,
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
            worker.start()
        self.barrier.wait()  # wait for all workers to be ready

        # gather rollouts and update when enough data is collected
        while self.completed_updates < self.target_updates:
            # reset rollout for next update
            self.rollout.reset(n_agents=self.env_config.n_agents)
            self.barrier.wait()  # wait for workers to finish their rollout
            for _ in range(self.n_workers):
                # gather rollouts from workers
                try: 
                    self._gather_rollout()
                    
                except Exception as e:
                    print(f"Error: {e}")
                    continue

            # add the episode to the current rollout buffer
            if self.rollout.total_steps >= self.samples_per_update:
                # update the controller with the current rollout
                self._optimise(rollout = self.rollout)
                self.completed_updates += 1
                print(f'\n\nCompleted Updates: {self.completed_updates} / {self.target_updates}\n\n')

                wandb.log({'train/average_episode_reward': np.mean([ep['average_episode_reward'] for ep in self.rollout.episodes])})

                # broadcast updated controller weights
                self._broadcast_controller_state()


        # Wait for all workers to finish their last trajectory collections
        self.done_event.set()
        # drain any final episode info
        for _ in range(self.n_workers):
            try:
                self._gather_rollout()
            except queue.Empty:
                continue

        # final controller update
        self._optimise(rollout=self.rollout)
        self.completed_updates += 1
        wandb.log({
            'train/average_episode_reward': np.mean([ep['average_episode_reward'] for ep in self.rollout.episodes]),
        })

        # terminate all workers
        for w in workers:
            w.join()
        for w in workers:
            if w.is_alive():
                w.terminate()
        
        wandb.finish()

        savepath = os.path.join('models', f'{self.run_name}')
        os.makedirs(savepath, exist_ok=True)
        torch.save(self.controller.actor_network.state_dict(), os.path.join(savepath, 'actor_model.pth'))
        torch.save(self.controller.critic_network.state_dict(), os.path.join(savepath, 'critic_model.pth'))


    def _gather_rollout(self) -> None:
        """ Gather episodes from rollout queue and add to training rollout. """
        worker_rollout: Dict[str, List] = self.rollout_queue.get(timeout=60)
        self.rollout.add_episode(worker_rollout)
        self._log_episode_info()


    def _broadcast_controller_state(self) -> None:
        """
        Push the current controller state to the shared dictionary for workers to access.
        """
        controller_state = (self.controller.actor_network.state_dict(),
                            self.controller.critic_network.state_dict())
        self.shared_weights['controller_state'] = controller_state
        self.shared_weights['update_step'] = self.update_step


    def _log_episode_info(self):
        """Log episode information to Weights & Biases. """
        try: 
            while True: 
                log_info = self.logging_queue.get_nowait()
                worker_id = log_info['worker_id']
                wandb.log({
                    f'worker_{worker_id}/episode_reward': log_info['episode/reward'],
                    f'worker_{worker_id}/episode_length': log_info['episode/average_length'],
                    # global aggregated metrics
                    'episode/average_reward': log_info['episode/reward'],
                })
        except queue.Empty:
            pass


    def worker_entry(self, queue: mp.Queue) -> None:
        """
        Entry point for each worker to run the PPOWorker.
        """
        worker = PPOWorker(env_config=self.env_config, 
                           controller=self.controller,
                           max_steps=self.max_steps,
                           device='cpu')
        rollout = worker.run()
        queue.put(rollout)


    def _build_optimizer(self, optimiser_config: Dict[str, Union[int, str]]) -> optim.Optimizer:
        if optimiser_config['type'] == 'adam': 
            self.optimiser = optim.Adam(params=chain(self.controller.actor_network.parameters(), self.controller.critic_network.parameters()), lr=float(optimiser_config['learning_rate']))
        else: 
            raise Warning('Only Adam optimiser has been implemented')


    def _optimise(self, rollout: MultiAgentRolloutBuffer) -> Dict[str, List[float]]:
        """
        Optimise the model parameters using the collected rollouts.
        """
        losses = {
            'policy_loss': [],
            'value_loss': []
        }

        total_loss, actor_loss, critic_loss = self._loss(rollout)

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
    
    
    def _loss(self, rollout: MultiAgentRolloutBuffer) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the loss for the actor and critic networks.
        """
        # forward pass through the actor network to get actions and log probabilities
        transitions: Dict[str, Tensor] = rollout.get_transitions() # TODO: check that this is done correctly everywhere
        new_log_probs, entropy, new_state_values, new_next_state_values = self._evaluate(transitions['states'], transitions['next_states'], transitions['actions'])
        new_state_values = new_state_values.squeeze(-1)  
        new_next_state_values = new_next_state_values.squeeze(-1)

        # Policy loss
        old_log_probs = transitions['log_probs']
        actor_loss = policy_loss(gae=transitions['gaes'],
                                new_log_prob=new_log_probs,
                                old_log_prob=old_log_probs,
                                clip_eps=self.controller.config['clip_epsilon'])
        
        # Value loss
        if self.importance_sampling:
            critic_loss = value_loss_with_IS(state_values=new_state_values,
                                            next_state_values=new_next_state_values,
                                            new_log_prob=new_log_probs,
                                            old_log_prob=old_log_probs,
                                            reward=transitions['rewards'],
                                            done=transitions['dones'],
                                            gamma=self.controller.config['gamma']
                                            )
        else:
            critic_loss = value_loss(state_values=new_state_values,
                                    next_state_values=new_next_state_values,
                                    reward=transitions['rewards'],
                                    done=transitions['dones'],
                                    gamma=self.controller.config['gamma'])
            
        # Entropy bonus
        entropy_loss = -entropy.mean()  # Encourage exploration

        # Total loss & optimisation step        # TODO: controller or learner config?
        total_loss: Tensor = actor_loss + critic_loss * self.controller.config['value_loss_coefficient'] + entropy_loss * self.controller.config['entropy_coefficient']
        return total_loss, actor_loss, critic_loss
        

    def _evaluate(self, states: Tensor, next_states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ 
        Forward pass through the actor network to get the action distribution and log probabilities, to compute the entropy of the policy distribution and the current state values.
        """
        logits = self.controller.actor_network(states)
        action_distribution = torch.distributions.Categorical(logits=logits)
        log_probs = action_distribution.log_prob(actions)
        entropy = action_distribution.entropy()
        state_values = self.controller.critic_network(states)
        next_state_values = self.controller.critic_network(next_states)
        return log_probs, entropy, state_values, next_state_values
