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

from src.controllers.BaseController import Controller
from src.controllers.PPOController import PPOController
from src.controllers.LSTMController import LSTMController
from src.configs.ControllerConfigs import ControllerConfig
from src.configs.EnvConfig import FlatlandEnvConfig
from src.algorithms.PPO.PPOWorker import PPOWorker
from src.algorithms.loss import value_loss, value_loss_with_IS, policy_loss
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer
from src.utils.observation.RunningMeanStd import RunningMeanStd


class PPOLearner():
    """
    Learner class for the PPO Algorithm.
    """
    def __init__(self, controller_config: ControllerConfig, learner_config: Dict, env_config: FlatlandEnvConfig, device: str = None) -> None:
        # Initialise environment and set controller / learning parameters
        self.env_config = env_config
        self._init_learning_params(learner_config)
        self._init_controller(controller_config)

        # Parallelisation Configuration
        self.n_workers: int = learner_config['n_workers'] 
        self._init_queues()

        # Initialise the optimiser
        self._build_optimiser(learner_config['optimiser_config'])
        self.epochs: int = 0

        # Initialise wandb for logging
        self._init_wandb(learner_config)

        # Initialise running mean and std for observation normalisation
        self.distance_rms: RunningMeanStd = RunningMeanStd(size=1)

    def _init_controller(self, config: ControllerConfig) -> None:
        self.controller_config = config
        self.n_nodes: int = config.config_dict['n_nodes']
        self.state_size: int = config.config_dict['state_size']
        self.controller: Union[PPOController, LSTMController, Controller] = config.create_controller()

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
        self.sgd_iterations: int = learner_config['sgd_iterations']
        self.entropy_coeff: float = learner_config['entropy_coefficient']
        self.value_loss_coeff: float = learner_config['value_loss_coefficient']
        self.gamma: float = learner_config['gamma']
        self.gae_lambda: float = learner_config['lam']
        self.gae_horizon: int = learner_config['gae_horizon']
        self.clip_epsilon: float = learner_config['clip_epsilon']

    def _init_wandb(self, learner_config: Dict) -> None:
        """
        Initialize Weights & Biases for logging.
        """
        self.run_name = learner_config['run_name']
        wandb.init(project='AI4REALNET-T3.4', entity='CLS-FHNW', config=learner_config, reinit=True)
        wandb.run.define_metric('episodes/*', step_metric='episode')
        wandb.run.define_metric('train/*', step_metric='epoch')
        wandb.run.name = f"{self.run_name}_PPO"
        wandb.watch(self.controller.actor_network, log='all')
        wandb.watch(self.controller.critic_network, log='all')
        wandb.watch(self.controller.encoder_network, log='all')

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
        self.n_agents = self.env_config.get_num_agents()
        self.rollout = MultiAgentRolloutBuffer(n_agents=self.n_agents)

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

                # reset rollout for next update
                self.rollout.reset(n_agents=self.n_agents)

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
        torch.save(self.controller.encoder_network.state_dict(), os.path.join(savepath, 'encoder_model.pth'))


    def _gather_rollout(self) -> None:
        """ Gather episodes from rollout queue and add to training rollout. """
        worker_rollout: Dict[str, List] = self.rollout_queue.get(timeout=60)
        self.rollout.add_episode(worker_rollout)
        self._log_episode_info()


    def _broadcast_controller_state(self) -> None:
        """
        Push the current controller state to the shared dictionary for workers to access.
        """
        controller_state = (self.controller.encoder_network.state_dict(),
                            self.controller.actor_network.state_dict(),
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
                    f'worker_{worker_id}/completion': log_info['episode/completion'],
                    # global aggregated metrics
                    'episode/total_reward': log_info['episode/total_reward'],
                })
        except queue.Empty:
            pass


    def _build_optimiser(self, optimiser_config: Dict[str, Union[int, str]]) -> optim.Optimizer:
        if optimiser_config['type'] == 'adam': 
            self.optimiser = optim.Adam(params=chain(self.controller.actor_network.parameters(), self.controller.critic_network.parameters(), self.controller.encoder_network.parameters()), lr=float(optimiser_config['learning_rate']))
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

        gae_stats = self._gaes()
        self.rollout.get_transitions(gae=True) # TODO: check that this is done correctly everywhere

        for _ in range(self.sgd_iterations):  
            entropy_sum: Tensor = Tensor()
            old_log_probs_sum: Tensor = Tensor()
            new_log_probs_sum: Tensor = Tensor()
            actor_delta = []
            critic_delta = []

            for minibatch in self.rollout.get_minibatches(self.batch_size): # TODO: check that this is the right batchsize
                new_log_probs, entropy, new_state_values, new_next_state_values = self._evaluate(minibatch['states'], minibatch['next_states'], minibatch['actions'])
                minibatch['new_log_probs'] = new_log_probs
                minibatch['new_state_values'] = new_state_values.squeeze(-1)
                minibatch['entropy'] = entropy

                total_loss, actor_loss, critic_loss = self._loss(minibatch)
                
                #! delta actor / critic norm calculation
                with torch.no_grad():
                    actor_before = torch.nn.utils.parameters_to_vector(self.controller.actor_network.parameters()).norm().item()
                    critic_before = torch.nn.utils.parameters_to_vector(self.controller.critic_network.parameters()).norm().item()
                    encoder_before = torch.nn.utils.parameters_to_vector(self.controller.encoder_network.parameters()).norm().item()

                self.optimiser.zero_grad()
                total_loss.backward()
                self.optimiser.step()

                #! delta actor / critic norm calculation
                with torch.no_grad():
                    actor_after = torch.nn.utils.parameters_to_vector(self.controller.actor_network.parameters()).norm().item()
                    critic_after = torch.nn.utils.parameters_to_vector(self.controller.critic_network.parameters()).norm().item()
                    encoder_after = torch.nn.utils.parameters_to_vector(self.controller.encoder_network.parameters()).norm().item()
                    actor_delta = np.abs(actor_after - actor_before)
                    critic_delta = np.abs(critic_after - critic_before)
                    encoder_delta = np.abs(encoder_after - encoder_before)

                # add metrics
                losses['policy_loss'].append(actor_loss.item())
                losses['value_loss'].append(critic_loss.item())

                # entropy and log probs for approx KL
                entropy_sum = torch.cat((entropy_sum, minibatch['entropy']))
                old_log_probs_sum = torch.cat((old_log_probs_sum, minibatch['log_probs']))
                new_log_probs_sum = torch.cat((new_log_probs_sum, new_log_probs))

            self.epochs += 1
            wandb.log({
                'epoch': self.epochs,
                'train/policy_loss': np.mean(losses['policy_loss']),
                'train/value_loss': np.mean(losses['value_loss']),
                'train/raw_gae_mean': gae_stats[0],
                'train/raw_gae_std': gae_stats[1],
                'train/mean_entropy': entropy.mean().item(),
                'train/approx_kl': torch.mean(old_log_probs_sum - new_log_probs_sum).item(),
                #! delta actor / critic norm calculation
                'train/actor_delta': actor_delta,
                'train/critic_delta': critic_delta,
                'train/encoder_delta': encoder_delta,
            })

        return losses
    
    
    def _loss(self, minibatch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the loss for the actor and critic networks.
        """
        actor_loss = policy_loss(gae=minibatch['gaes'],
                                new_log_prob=minibatch['new_log_probs'],
                                old_log_prob=minibatch['log_probs'],
                                clip_eps=self.clip_epsilon)

        if self.importance_sampling:
            critic_loss = value_loss_with_IS(state_values=minibatch['new_state_values'],
                                            new_state_values=minibatch['new_state_values'],
                                            new_log_prob=minibatch['new_log_probs'],
                                            old_log_prob=minibatch['log_probs'],
                                            reward=minibatch['rewards'],
                                            done=minibatch['dones'],
                                            gamma=self.gamma
                                            )
        else:
            critic_loss = value_loss(state_values=minibatch['state_values'],
                                    new_state_values=minibatch['new_state_values'],
                                    reward=minibatch['rewards'],
                                    done=minibatch['dones'],
                                    gamma=self.gamma
                                    )

        # Entropy bonus
        entropy_loss = -minibatch['entropy'].mean()  # Encourage exploration

        # Total loss & optimisation step        # TODO: controller or learner config?
        total_loss: Tensor = actor_loss + critic_loss * self.value_loss_coeff + entropy_loss * self.entropy_coeff
        return total_loss, actor_loss, critic_loss


    def _gaes(self) -> Tuple[float, float]:
        # TODO: normalise GAEs
        for idx, episode in enumerate(self.rollout.episodes):
            self.rollout.episodes[idx]['gaes'] = [[] for _ in range(self.env_config.n_agents)]
            for agent in range(len(episode['states'])):
                state_values = torch.stack(episode['state_values'][agent])
                next_state_values = torch.stack(episode['next_state_values'][agent])

                rewards = torch.tensor(episode['rewards'][agent])
                dones = torch.tensor(episode['dones'][agent]).float()
                traj_len = len(rewards)

                gaes = [torch.tensor(0.0) for _ in range(len(rewards))]
                advantage = 0.0
                for t in reversed(range(len(rewards))):
                    delta = rewards[t] + self.gamma * next_state_values[t] * (1 - dones[t]) - state_values[t]
                    advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
                    gaes[t] = advantage

                gae_tensor = torch.stack(gaes)
                self.rollout.episodes[idx]['gaes'][agent] = gae_tensor

        return self._normalise_gaes()

    
    def _normalise_gaes(self) -> Tuple[float, float]:
        n_episodes = len(self.rollout.episodes)

        stacked_gaes = torch.cat([torch.cat(self.rollout.episodes[episode]['gaes']) for episode in range(n_episodes)], dim=0)

        raw_gae_mean = stacked_gaes.mean()
        raw_gae_std = stacked_gaes.std().clamp_min(1e-8)  # avoid division by zero

        for idx in range(n_episodes):
            for agent in range(self.n_agents):
                gae_tensor = self.rollout.episodes[idx]['gaes'][agent]
                self.rollout.episodes[idx]['gaes'][agent] = (gae_tensor - raw_gae_mean) / raw_gae_std
        
        stacked_gaes = torch.cat([torch.cat(self.rollout.episodes[episode]['gaes']) for episode in range(n_episodes)], dim=0)
        gae_mean = stacked_gaes.mean().item()
        gae_std = stacked_gaes.std().item()

        return raw_gae_mean.item(), raw_gae_std.item(), gae_mean, gae_std


    def _evaluate(self, states: Tensor, next_states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ 
        Forward pass through the actor network to get the action distribution and log probabilities, to compute the entropy of the policy distribution and the current state values.
        """
        encoded_states = self.controller.encoder_network(states)
        encoded_next_states = self.controller.encoder_network(next_states)
        logits = self.controller.actor_network(encoded_states) # (batch_size, n_actions)
        action_distribution = torch.distributions.Categorical(logits=logits)
        log_probs = action_distribution.log_prob(actions) # (batch_size,)
        entropy = action_distribution.entropy() # (batch_size,)
        state_values = self.controller.critic_network(encoded_states) # (batch_size, 1)
        next_state_values = self.controller.critic_network(encoded_next_states) # (batch_size, 1)
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
