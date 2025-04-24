import torch
import numpy as np
import torch.multiprocessing as mp
from typing import Dict
import argparse
import random

from src.utils.file_utils import load_config_file
from src.utils.obs_utils import calculate_state_size

from src.configs.EnvConfig import FlatlandEnvConfig
from src.configs.ControllerConfigs import PPOControllerConfig
from src.configs.OptimiserConfig import AdamConfig
from src.algorithms.PPO.PPOLearner import PPOLearner


def train_ppo(n_workers: int, random_seed: int, controller_config: PPOControllerConfig, environment_config: FlatlandEnvConfig, device: str) -> None:
    init_random_seeds(random_seed)

    learner = PPOLearner(controller_config, env_config, device, n_workers)
    learner.rollouts()


def init_random_seeds(random_seed: int, cuda_deterministic: bool = False) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a PPO agent')
    parser.add_argument('--config_path', type=str, default='src/configs/ppo_config.yaml', help='Path to the configuration file')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    config = load_config_file(args.config_path)
    env_config = FlatlandEnvConfig(config['environment_config'])

    config['controller_config']['state_size'] = calculate_state_size(config['environment_config']['max_tree_depth'])
    controller_config = PPOControllerConfig(config['controller_config'])

    train_ppo(config['n_workers'], config['random_seed'], controller_config, 
              env_config, config['device'])