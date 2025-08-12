import os
import sys
import wandb
import argparse
import numpy as np
from itertools import chain
from typing import List, Dict, Union, Tuple

import torch
from torch import Tensor
import torch.optim as optim
import torch.multiprocessing as mp

from src.algorithms.PPO.PPOWorker import PPOWorker
from src.algorithms.PPO.PPOController import PPOController
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer
from src.training.loss import value_loss, value_loss_with_IS, policy_loss

from flatland.envs.rail_env import RailEnv
from src.configs.EnvConfig import FlatlandEnvConfig

class PPOLearner():
    """
    Learner class for the PPO Controller.
    """
    def __init__(self, controller: PPOController, learner_config: Dict, env_config: Dict) -> None:
        # Initialise environment and controller
        self.env_config = env_config
        self.controller: PPOController = controller
        self.n_nodes: int = controller.config['n_nodes']
        self.state_size: int = controller.config['state_size']
        self.entropy_coeff: float = controller.config['entropy_coefficient']
        self.value_loss_coeff: float = controller.config['value_loss_coefficient']
        self.gamma: float = controller.config['gamma'] # TODO: ensure gamma is in learner_config

        # Learning parameters
        self.max_steps: int = learner_config['max_steps']
        self.max_steps_per_episode: int = learner_config['max_steps_per_episode']
        self.total_steps: int = 0
        self.iterations: int = learner_config['training_iterations']
        self.batch_size: int = learner_config['batch_size']
        self.importance_sampling: bool = learner_config['importance_sampling'] # TODO: add importance sampling to config
        self.episodes_infos: List[Dict] = []
        self.total_episodes: int = 0

        # Parallelisation Configuration
        self.n_workers: int = learner_config['n_workers'] # TODO: add n_workers to learner_config

        # Initialise the optimiser
        self.optimizer: optim.Optimizer = self._build_optimizer(learner_config['optimiser_config']) # TODO: add optimiser_config to learner_config

        # Initialise wandb for logging
        self._init_wandb(learner_config)
        wandb.watch(self.controller.actor_network, log='all')
        wandb.watch(self.controller.critic_network, log='all')


    def _init_wandb(self, learner_config: Dict) -> None:
        """
        Initialize Weights & Biases for logging.
        """
        wandb.init(project='AI4REALNET-T3.4', entity='CLS-FHNW', config=learner_config, reinit=True)
        wandb.run.define_metric('episodes/*', step_metric='episode')
        wandb.run.define_metric('train/*', step_metric='update_step')
        wandb.run.name = learner_config['run_name']
        wandb.run.save()


    def worker_entry(self, worker_id: int, queue: mp.Queue) -> None:
        """
        Entry point for each worker to run the PPOWorker.
        """
        worker = PPOWorker(env_config=self.env_config, 
                           controller=self.controller,
                           max_steps=self.max_steps,
                           device='cpu')
        rollout = worker.run()
        queue.put(rollout)
    

    def gather_rollout(self) -> MultiAgentRolloutBuffer:
        """
        Rollout function to gather experience tuples for the PPO agent.
        
        :return: List of transitions collected during the rollout.
        """
        # parallelisation of rollout gathering
        mp.set_start_method('spawn')
        queue: mp.Queue = mp.Queue()
        processes: List[mp.Process] = []
        for worker_id in range(self.n_workers):
            process = mp.Process(target=self.worker_entry, args=(worker_id, queue))
            processes.append(process)
            process.start()

        rollouts: List[MultiAgentRolloutBuffer] = [queue.get() for _ in range(self.n_workers)]

        for process in processes:
            process.join()

        # Combine rollouts from all workers
        combined_rollout: MultiAgentRolloutBuffer = MultiAgentRolloutBuffer.combine_rollouts(rollouts)

        return combined_rollout


    def _build_optimizer(self, optimiser_config: Dict[str, Union[int, str]]) -> optim.Optimizer:
        if optimiser_config['type'] == 'adam': 
            self.optimiser = optim.Adam(params=chain(self.controller.actor_network.parameters(), self.controller.critic_network.parameters()), lr=float(optimiser_config['learning_rate']))
        else: 
            raise Warning('Only Adam optimiser has been implemented')


    def _optimise(self, rollouts: MultiAgentRolloutBuffer) -> Dict[str, List[float]]:
        """
        Optimise the model parameters using the collected rollouts.
        """
        losses = {
            'policy_loss': [],
            'value_loss': []
        }

        total_loss, actor_loss, critic_loss = self._loss(rollouts)

        self.optimiser.zero_grad()
        total_loss.backward()
        self.optimiser.step()
        self.update_step += 1

        # add metrics
        losses['policy_loss'].append(actor_loss.mean().item())
        losses['value_loss'].append(critic_loss.mean().item())
        wandb.log({
            'update_step': self.update_step,
            'train/policy_loss': np.mean(losses['policy_loss']),
            'train/value_loss': np.mean(losses['value_loss'])
        })
        return losses


    def run(self) -> Dict:
        n_episodes = 0
        metrics: Dict[List] = {
            'rewards': [],
            'policy_loss': [],
            'value_loss': []
        }

        self.total_steps = 0
        for epoch in range(self.iterations):
            print(f'\nEpoch {epoch + 1}/{self.iterations}')
            while self.total_steps < self.max_steps:
                rollout = self.gather_rollout()
                self._calculate_metrics(rollout)
                n_episodes += len(rollout.episodes)
                losses = self._optimise(rollout)
                self.total_steps += rollout.total_steps


                episode_rewards = [episode['average_episode_reward'] for episode in rollout.episodes]
                average_episode_reward = sum(episode_rewards) / len(episode_rewards)
                print(f'\nTotal Steps: {self.total_steps}, Total Episodes: {n_episodes}, Average Episode Reward: {average_episode_reward}')
                
                # TODO: add this to wandb
                # metric information 
                metrics['policy_loss'].extend(losses['policy_loss'])
                metrics['value_loss'].extend(losses['value_loss'])
                metrics['rewards'].extend(episode_rewards)

            self.total_steps = 0


        return metrics
    
    
    def _loss(self, rollout: MultiAgentRolloutBuffer, batch_size: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the loss for the actor and critic networks.
        """
        
        for minibatch in rollout.get_minibatches(batch_size=batch_size, shuffle=False):
            # forward pass through the actor network to get actions and log probabilities
            new_log_probs, entropy, new_state_values, new_next_state_values = self._evaluate(minibatch['states'], minibatch['next_states'], minibatch['actions'])
            new_state_values = new_state_values.squeeze(-1)  
            new_next_state_values = new_next_state_values.squeeze(-1)

            # Policy loss
            old_log_probs = minibatch['log_probs']
            actor_loss = policy_loss(gae=minibatch['gaes'],
                                    new_log_prob=new_log_probs,
                                    old_log_prob=old_log_probs,
                                    clip_eps=self.controller.config['clip_epsilon'])
            
            # Value loss
            if self.importance_sampling:
                critic_loss = value_loss_with_IS(state_values=new_state_values,
                                                next_state_values=new_next_state_values,
                                                new_log_prob=new_log_probs,
                                                old_log_prob=old_log_probs,
                                                reward=minibatch['rewards'],
                                                done=minibatch['dones'],
                                                gamma=self.controller.config['gamma']
                                                )
            else:
                critic_loss = value_loss(state_values=new_state_values,
                                        next_state_values=new_next_state_values,
                                        reward=minibatch['rewards'],
                                        done=minibatch['dones'],
                                        gamma=self.controller.config['gamma'])
                
            # Entropy bonus
            entropy_loss = -entropy.mean()  # Encourage exploration

            # TODO: value_loss_coef and entropy_coef into config
            # Total loss & optimisation step
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
    

    def _calculate_metrics(self, rollout: MultiAgentRolloutBuffer) -> None:
        """
        Calculate metrics from the collected rollout.
        
        :param rollout: The collected rollout buffer.
        :return: A dictionary containing the calculated metrics.
        """
        for episode in rollout.episodes:
            episode_length = episode['episode_length']
            self.episodes_infos.append({
                'episode_length': episode_length,
                'total_steps': rollout.total_steps,
                'n_agents': rollout.n_agents, 
                'episode_reward': episode['average_episode_reward'],
            })