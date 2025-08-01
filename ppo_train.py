import os
import sys
import argparse
from torch import Tensor
from typing import List, Dict, Union

import wandb

from src.utils.file_utils import load_config_file
from src.utils.obs_utils import calculate_state_size, obs_dict_to_tensor
from src.configs.EnvConfig import FlatlandEnvConfig
from src.configs.ControllerConfigs import PPOControllerConfig

from src.algorithms.PPO.PPOController import PPOController
from src.algorithms.PPO.PPOLearner import Learner
from flatland.envs.rail_env import RailEnv


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
    controller: PPOController = PPOController(config['controller_config'])

    learner = Learner(env=env, 
                      controller=controller, 
                      learner_config=config['learner_config'], 
                      env_config=config['environment_config']
                      )
    metrics = learner.run()