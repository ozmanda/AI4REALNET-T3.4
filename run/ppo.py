import os
import sys
import random
import argparse
import numpy as np
from typing import Dict

import torch

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(src_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.utils.file_utils import load_config_file
from src.utils.observation.obs_utils import calculate_state_size

from src.configs.EnvConfig import FlatlandEnvConfig
from src.configs.ControllerConfigs import PPOControllerConfig
from src.algorithms.PPO.PPOLearner import PPOLearner


def train_ppo(controller_config: PPOControllerConfig, learner_config: Dict, env_config: Dict, device: str) -> None:
    learner = PPOLearner(controller_config=controller_config,
                         learner_config=learner_config,
                         env_config=env_config,
                         device=device)
    learner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a PPO agent')
    parser.add_argument('--config_path', type=str, default='src/configs/PPO_FNN.yaml', help='Path to the configuration file')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the training on (cpu or cuda)')
    args = parser.parse_args()


    # Load config file
    config = load_config_file(args.config_path)

    # prepare environment config
    if args.random_seed:
        config['environment_config']['random_seed'] = args.random_seed
    env_config = FlatlandEnvConfig(config['environment_config'])

    # prepare controller config and setup parallelisation
    learner_config = config['learner_config']

    # prepare controller
    config['controller_config']['n_nodes'], config['controller_config']['state_size'] = calculate_state_size(env_config.observation_builder_config['max_depth'])
    controller_config = PPOControllerConfig(config['controller_config'])


    train_ppo(controller_config = controller_config,
              learner_config = learner_config,
              env_config = env_config, 
              device = args.device)