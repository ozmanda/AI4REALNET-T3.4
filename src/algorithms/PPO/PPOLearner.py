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

from src.configs.ControllerConfigs import ControllerConfig
from src.configs.EnvConfig import FlatlandEnvConfig, PettingZooEnvConfig
from src.algorithms.PPO.PPOWorker import PPOWorker
from src.algorithms.loss import value_loss, value_loss_with_IS, policy_loss
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer
from src.utils.observation.RunningMeanStd import RunningMeanStd


class PPOLearner():
    """
    Learner class for the PPO Algorithm.
    """
    def __init__(self, controller_config: ControllerConfig, learner_config: Dict, env_config: Union[FlatlandEnvConfig, PettingZooEnvConfig], device: str = None) -> None:
        # Initialise environment and set controller / learning parameters
        self.env_config = env_config
        self._init_learning_params(learner_config)
        self._init_controller(controller_config)

        # Parallelisation Configuration
        self.n_workers: int = learner_config['n_workers'] 
        self._init_queues()

        # Initialise the optimiser
        self.optimizer: optim.optimizer.Optimizer = self._build_optimizer(learner_config['optimiser_config'])
        self.epochs: int = 0

        # Initialise wandb for logging
        self._init_wandb(learner_config)
        wandb.watch(self.controller.actor_network, log='all')
        wandb.watch(self.controller.critic_network, log='all')

        # Initialise running mean and std for observation normalisation
        self.distance_rms: RunningMeanStd = RunningMeanStd(size=1)

    def _init_controller(self, config: ControllerConfig) -> None:
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
        self.batch_size: int = learner_config['batch_size']
        self.importance_sampling: bool = learner_config['IS']
        self.episodes_infos: List[Dict] = []
        self.total_episodes: int = 0
        self.n_epochs_per_update: int = learner_config['n_epochs_update']

    def _init_wandb(self, learner_config: Dict) -> None:
        """
        Initialize Weights & Biases for logging.
        """
        self.run_name = learner_config['run_name']
        wandb.init(project='PPO_PettingZoo_Testing', entity='CLS-FHNW', config=learner_config, reinit=True)
        wandb.run.define_metric('episodes/*', step_metric='episode')
        wandb.run.define_metric('train/*', step_metric='epoch')
        wandb.run.name = f"{self.run_name}_PPO"
        wandb.watch(self.controller.actor_network, log='all')
        wandb.watch(self.controller.critic_network, log='all')

    def _init_queues(self) -> None:
        # create queues
        self.logging_queue: mp.Queue = mp.Queue()
        self.rollout_queue: mp.Queue = mp.Queue()
        # self.weights_queue: mp.Queue = mp.Queue()
        self.barrier = mp.Barrier(self.n_workers + 1)  # +1 for the learner process
        self.done_event: Event = mp.Event()
        # self.observation_queue: mp.Queue = mp.Queue()
        self.manager = mp.Manager()
        self.shared_weights: DictProxy = self.manager.dict()
        # self.shared_normalisation: DictProxy = self.manager.dict()


    def sync_run(self) -> None:
        """
        Synchronous PPO training run.
        """
        # initialise learning rollout
        n_agents = self.env_config.get_num_agents()
        self.rollout = MultiAgentRolloutBuffer(n_agents=n_agents)

        # create and start workers
        # TODO: add device specification for the workers
        mp.set_start_method('spawn', force=True)  # parallelisation of rollout gathering - spawn is safer for pytorch
        workers: List[PPOWorker] = []
        print('Initialising workers...')
        for worker_id in range(self.n_workers):
            worker = PPOWorker(worker_id=worker_id,
                               logging_queue=self.logging_queue,
                               rollout_queue=self.rollout_queue,
                               shared_weights=self.shared_weights,
                            #    shared_normalisation=self.shared_normalisation,
                            #    observation_queue=self.observation_queue,
                               barrier=self.barrier,
                               done_event=self.done_event,
                               env_config=self.env_config,
                               controller_config=self.controller_config,
                               max_steps=(self.max_steps, self.max_steps_per_episode),
                               device='cpu')
            workers.append(worker)
            worker.start()
        self._broadcast_controller_state() # initial weight broadcast
        # self._initialise_normalisation()

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
            # self._update_normalisation()

            # add the episode to the current rollout buffer
            if self.rollout.total_steps >= self.samples_per_update:
                # update the controller with the current rollout
                self._optimise()
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
        self._optimise()
        self.completed_updates += 1
        wandb.log({
            'train/average_episode_reward': np.mean([ep['average_episode_reward'] for ep in self.rollout.episodes]),
        })

        # terminate all workers
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
        self.shared_weights['update_step'] = self.completed_updates
        self.barrier.wait()  # wait for all workers to acknowledge the new weights


    def _log_episode_info(self):
        """Log episode information to Weights & Biases. """
        try: 
            while True: 
                log_info = self.logging_queue.get_nowait()
                worker_id = log_info['worker_id']
                wandb.log({
                    f'worker_{worker_id}/total_reward': log_info['episode/total_reward'],
                    f'worker_{worker_id}/average_reward': log_info['episode/average_reward'],
                    f'worker_{worker_id}/average_episode_length': log_info['episode/average_length'],
                    # global aggregated metrics
                    'episode/total_reward': log_info['episode/total_reward'],
                })
        except queue.Empty:
            pass


    def _build_optimizer(self, optimiser_config: Dict[str, Union[int, str]]) -> optim.Optimizer:
        if optimiser_config['type'] == 'adam': 
            self.optimiser = optim.Adam(params=chain(self.controller.actor_network.parameters(), self.controller.critic_network.parameters()), lr=float(optimiser_config['learning_rate']))
        else: 
            raise Warning('Only Adam optimiser has been implemented')


    def _optimise(self) -> Dict[str, List[float]]:
        """
        Optimise the model parameters using the collected rollouts.
        """
        losses = {
            'policy_loss': [],
            'value_loss': []
        }

        self._gaes()
        self.rollout.get_transitions(gae=True) # TODO: check that this is done correctly everywhere

        for _ in range(self.n_epochs_per_update):   
            for minibatch in self.rollout.get_minibatches(self.batch_size): # TODO: check that this is the right batchsize
                new_log_probs, entropy, new_state_values, new_next_state_values = self._evaluate(minibatch['states'], minibatch['next_states'], minibatch['actions'])
                minibatch['new_log_probs'] = new_log_probs
                minibatch['new_state_values'] = new_state_values.squeeze(-1)
                minibatch['new_next_state_values'] = new_next_state_values.squeeze(-1)
                minibatch['entropy'] = entropy

                total_loss, actor_loss, critic_loss = self._loss(minibatch)

                self.optimiser.zero_grad()
                total_loss.backward()
                self.optimiser.step()

                # add metrics
                losses['policy_loss'].append(actor_loss.item())
                losses['value_loss'].append(critic_loss.item())
            self.epochs += 1
            wandb.log({
                'epoch': self.epochs,
                'train/policy_loss': np.mean(losses['policy_loss']),
                'train/value_loss': np.mean(losses['value_loss'])
            })

        return losses
    
    
    def _loss(self, minibatch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the loss for the actor and critic networks.
        """
        actor_loss = policy_loss(gae=minibatch['gaes'],
                                new_log_prob=minibatch['new_log_probs'],
                                old_log_prob=minibatch['log_probs'],
                                clip_eps=self.controller.config['clip_epsilon'])

        if self.importance_sampling:
            critic_loss = value_loss_with_IS(state_values=minibatch['new_state_values'],
                                            next_state_values=minibatch['new_next_state_values'],
                                            new_log_prob=minibatch['new_log_probs'],
                                            old_log_prob=minibatch['log_probs'],
                                            reward=minibatch['rewards'],
                                            done=minibatch['dones'],
                                            gamma=self.controller.config['gamma']
                                            )
        else:
            critic_loss = value_loss(state_values=minibatch['new_state_values'],
                                    next_state_values=minibatch['new_next_state_values'],
                                    reward=minibatch['rewards'],
                                    done=minibatch['dones'],
                                    gamma=self.controller.config['gamma'])            
            
        # Entropy bonus
        entropy_loss = -minibatch['entropy'].mean()  # Encourage exploration

        # Total loss & optimisation step        # TODO: controller or learner config?
        total_loss: Tensor = actor_loss + critic_loss * self.controller.config['value_loss_coefficient'] + entropy_loss * self.controller.config['entropy_coefficient']
        return total_loss, actor_loss, critic_loss


    def _gaes(self) -> None:
        # TODO: normalise GAEs
        all_gaes: List[Tensor] = []
        for idx, episode in enumerate(self.rollout.episodes):
            self.rollout.episodes[idx]['gaes'] = [[] for _ in range(self.env_config.n_agents)]
            for agent in range(len(episode['states'])):
                states = torch.stack(episode['states'][agent])
                state_values = torch.stack(episode['state_values'][agent])
                next_states = torch.stack(episode['next_states'][agent])
                next_state_values = torch.stack(episode['next_state_values'][agent])
                rewards = torch.tensor(episode['rewards'][agent])
                dones = torch.tensor(episode['dones'][agent]).float()
                dones = (dones != 0).float()
                gaes = []

                deltas = rewards + self.gamma * next_state_values * (1 - dones) - state_values
                gae = 0
                for t in reversed(range(len(rewards))):
                    gae = deltas[t] + self.gamma * self.controller.config['lam'] * (1 - dones[t]) * gae
                    gaes.insert(0, gae.detach())

                gae_tensor = torch.stack(gaes)
                self.rollout.episodes[idx]['gaes'][agent] = gae_tensor
                all_gaes.append(gae_tensor)

        self._normalise_gaes(all_gaes)

    
    def _normalise_gaes(self, gaes: List[Tensor]) -> None:
        if not gaes:
            return
        stacked_gaes = torch.cat([gae.flatten() for gae in gaes])
        gae_mean = stacked_gaes.mean()
        gae_std = stacked_gaes.std().clamp_min(1e-8)  # avoid division by zero
        for idx, episode in enumerate(self.rollout.episodes):
            for agent in range(len(episode['gaes'])):
                gae_tensor = episode['gaes'][agent]
                if isinstance(gae_tensor, list):
                    gae_tensor = torch.stack(gae_tensor)
                self.rollout.episodes[idx]['gaes'][agent] = (gae_tensor - gae_mean) / gae_std


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
    
    # TODO: finish implementing running mean and std calculation over all workers
    # def _initialise_normalisation(self) -> None:
    #     """
    #     Update the running mean and std for distance metrics and push to the shared dictionary for workers to access.
    #     """
    #     self._gather_observations()
    #     self.shared_normalisation['distance_rms'] = self.distance_rms
    #     print("Learner waiting at barrier after normalsation push...")
    #     self.barrier.wait()
    
    # def _gather_observations(self) -> None:
    #     """
    #     Gather distance observations from all workers to update the running mean and std.
    #     """
    #     print("Learner waiting at barrier before gathering observations...")
    #     self.barrier.wait()  # wait for all workers to push their statistics
    #     for _ in range(self.n_workers): 
    #         worker_obs = self.observation_queue.get()
    #         print("Learner calculating rms update...")
    #         self.distance_rms.update_batch(worker_obs)
        
    # def _update_normalisation(self) -> None:
    #     """
    #     Update the running mean and std for distance metrics and push to the shared dictionary for workers to access.
    #     """
    #     self._gather_observations()
    #     self.shared_normalisation['distance_rms'] = self.distance_rms
    #     print("Learner waiting at barrier after normalsation push...")
    #     self.barrier.wait()
