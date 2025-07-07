import os
import sys
import argparse
from torch import Tensor
from typing import List, Dict, Union

from src.utils.file_utils import load_config_file
from src.utils.obs_utils import calculate_state_size, obs_dict_to_tensor
from src.configs.EnvConfig import FlatlandEnvConfig
from src.configs.ControllerConfigs import PPOControllerConfig

from src.algorithms.PPO.ppo import PPOAgent
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer, Transition

from flatland.envs.rail_env import RailEnv

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Learner():
    """
    Learner class for the PPO Controller.
    """
    def __init__(self, env: RailEnv, controller: PPOAgent, learner_config: Dict, env_config: Dict) -> None:
        self.env = env
        self.controller: PPOAgent = controller
        self.max_steps: int = learner_config['max_steps']
        self.max_steps_per_episode: int = learner_config['max_steps_per_episode']
        self.total_steps: int = 0
        self.env_config: dict = env_config
        self.learner_config: Dict = learner_config
        self.n_nodes: int = controller.config['n_nodes']
        self.state_size: int = controller.config['state_size']
        self.obs_type: str = env_config['observation_builder_config']['type']
        self.max_depth: int = env_config['observation_builder_config']['max_depth']
        self.epochs: int = learner_config['epochs_per_rollout']
        self.batch_size: int = learner_config['batch_size']
        self.episodes_infos: List[Dict] = []


    def run(self) -> Dict:
        n_episodes = 0
        metrics: Dict = {}
        self.total_steps = 0
        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch + 1}/{self.epochs}')
            while self.total_steps < self.max_steps:
                rollout = self.gather_rollout()
                self._calculate_metrics(rollout)
                n_episodes += len(rollout.episodes)
                self.controller.update_networks(rollout, self.epochs, self.batch_size, IS=self.learner_config['IS'])
                self.total_steps += rollout.total_steps
                average_episode_reward = sum(episode['average_episode_reward'] for episode in rollout.episodes) / len(rollout.episodes)
                print(f'\nTotal Steps: {self.total_steps}, Total Episodes: {n_episodes}, Average Episode Reward: {average_episode_reward}')
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PPO agent')
    parser.add_argument('--config_path', type=str, default='src/configs/ppo_config.yaml', help='Path to the configuration file')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    config = load_config_file(args.config_path)

    # initialise the environment
    env_config = FlatlandEnvConfig(config['environment_config'])
    env: RailEnv = env_config.create_env()
    env.reset()

    # initialise a PPO agent per agent in the environment
    config['controller_config']['n_nodes'], config['controller_config']['state_size'] = calculate_state_size(config['environment_config']['observation_builder_config']['max_depth'])
    controller_config: PPOControllerConfig = PPOControllerConfig(config['controller_config'], device='cpu')
    controller: PPOAgent = PPOAgent(config['controller_config'])

    learner = Learner(env=env, 
                      controller=controller, 
                      learner_config=config['learner_config'], 
                      env_config=config['environment_config']
                      )
    metrics = learner.run()