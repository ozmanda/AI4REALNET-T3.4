import os
import sys
import argparse
from torch import Tensor
from typing import List, Dict, Union

import wandb

from src.utils.obs_utils import calculate_state_size, obs_dict_to_tensor

from src.algorithms.PPO.PPOController import PPOController
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer

from flatland.envs.rail_env import RailEnv

class Learner():
    """
    Learner class for the PPO Controller.
    """
    def __init__(self, env: RailEnv, controller: PPOController, learner_config: Dict, env_config: Dict) -> None:
        self._init_wandb(learner_config)
        self.env = env
        self.controller: PPOController = controller
        wandb.watch(self.controller.actor_network, log='all')
        wandb.watch(self.controller.critic_network, log='all')
        self.max_steps: int = learner_config['max_steps']
        self.max_steps_per_episode: int = learner_config['max_steps_per_episode']
        self.total_steps: int = 0
        self.env_config: dict = env_config
        self.learner_config: Dict = learner_config
        self.n_nodes: int = controller.config['n_nodes']
        self.state_size: int = controller.config['state_size']
        self.obs_type: str = env_config['observation_builder_config']['type']
        self.max_depth: int = env_config['observation_builder_config']['max_depth']
        self.iterations: int = learner_config['training_iterations']
        self.batch_size: int = learner_config['batch_size']
        self.episodes_infos: List[Dict] = []
        self.total_episodes: int = 0

    def _init_wandb(self, learner_config: Dict) -> None:
        """
        Initialize Weights & Biases for logging.
        """
        wandb.init(project='AI4REALNET-T3.4', entity='CLS-FHNW', config=learner_config, reinit=True)
        wandb.run.define_metric('episodes/*', step_metric='episode')
        wandb.run.define_metric('train/*', step_metric='update_step')
        wandb.run.name = learner_config['run_name']
        wandb.run.save()

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
                losses = self.controller.update_networks(rollout, self.iterations, self.batch_size, IS=self.learner_config['IS'])
                self.total_steps += rollout.total_steps


                episode_rewards = [episode['average_episode_reward'] for episode in rollout.episodes]
                average_episode_reward = sum(episode_rewards) / len(episode_rewards)
                print(f'\nTotal Steps: {self.total_steps}, Total Episodes: {n_episodes}, Average Episode Reward: {average_episode_reward}')
                
                # metric information
                metrics['policy_loss'].extend(losses['policy_loss'])
                metrics['value_loss'].extend(losses['value_loss'])
                metrics['rewards'].extend(episode_rewards)

            self.total_steps = 0


        return metrics
    

    def gather_rollout(self) -> MultiAgentRolloutBuffer:
        """
        Rollout function to gather experience tuples for the PPO agent.
        
        :return: List of transitions collected during the rollout.
        """
        rollout: MultiAgentRolloutBuffer = MultiAgentRolloutBuffer(n_agents=self.env.number_of_agents)
        rollout.reset(agent_handles=self.env.get_agent_handles())

        episode_step = 0
        current_state_dict, _ = self.env.reset()
        n_agents = self.env.number_of_agents
        current_state_tensor: Tensor = obs_dict_to_tensor(observation=current_state_dict, 
                                                          obs_type=self.obs_type, 
                                                          n_agents=n_agents,
                                                          max_depth=self.max_depth, 
                                                          n_nodes=self.n_nodes)

        while rollout.total_steps < self.max_steps:
            print(f'\rEpisode {rollout.n_episodes + 1} - Step {episode_step + 1}/{self.max_steps_per_episode}', end='', flush=True)
            actions, log_probs = self.controller.sample_action(current_state_tensor)
            actions_dict: Dict[Union[int, str], Tensor] = {}
            for i, agent_handle in enumerate(self.env.get_agent_handles()):
                actions_dict[agent_handle] = actions[i]

            next_state, rewards, dones, infos = self.env.step(actions_dict)
            next_state_tensor: Tensor = obs_dict_to_tensor(observation=next_state, 
                                                           obs_type=self.obs_type, 
                                                           n_agents=n_agents,
                                                           max_depth=self.max_depth, 
                                                           n_nodes=self.n_nodes)

            rollout.add_transitions(states=current_state_tensor, actions=actions_dict, log_probs=log_probs, 
                                    rewards=rewards, next_states=next_state_tensor, dones=dones)

            current_state_tensor = next_state_tensor
            episode_step += 1

            if all(dones.values()) or episode_step >= self.max_steps_per_episode:
                rollout.end_episode()
                current_state_dict, _ = self.env.reset()
                current_state_tensor = obs_dict_to_tensor(observation=current_state_dict, 
                                                           obs_type=self.obs_type, 
                                                           n_agents=n_agents,
                                                           max_depth=self.max_depth, 
                                                           n_nodes=self.n_nodes)
                episode_step = 0
                self.total_episodes += 1

                wandb.log({
                    'episode': self.total_episodes,
                    'episode/reward': rollout.episodes[-1]['average_episode_reward'],
                    'episode/average_length': rollout.episodes[-1]['average_episode_length'],
                })

        return rollout
    

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