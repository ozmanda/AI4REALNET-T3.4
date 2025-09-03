import os
import sys
import random
import argparse
import numpy as np
from typing import Dict

import torch
import torch.multiprocessing as mp

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(src_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.utils.file_utils import load_config_file
from src.utils.obs_utils import calculate_state_size

from src.configs.EnvConfig import FlatlandEnvConfig
from src.configs.ControllerConfigs import PPOControllerConfig
from src.algorithms.IMPALA.IMPALALearner import IMPALALearner


def train_impala(random_seed: int, controller_config: PPOControllerConfig, learner_config: Dict, env_config: Dict, device: str) -> None:
    init_random_seeds(random_seed)
    learner = IMPALALearner(controller_config=controller_config,
                         learner_config=learner_config,
                         env_config=env_config,
                         device=device)
    learner.async_run()


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
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the training on (cpu or cuda)')
    parser.add_argument('--n_workers', type=int, default=5, help='Number of parallel workers for training')
    args = parser.parse_args()


    # Load config file
    config = load_config_file(args.config_path)

    # prepare environment config
    env_config = FlatlandEnvConfig(config['environment_config'])

    # prepare controller config and setup parallelisation
    learner_config = config['learner_config']
    if args.device == 'cpu' and args.n_workers is None:
        # If no device is specified, use all available CPU cores
        args.n_workers = mp.cpu_count()
    learner_config['n_workers'] = args.n_workers

    # prepare controller
    config['controller_config']['n_nodes'], config['controller_config']['state_size'] = calculate_state_size(env_config.observation_builder_config['max_depth'])
    controller_config = PPOControllerConfig(config['controller_config'])


    train_impala(random_seed = args.random_seed,
              controller_config = controller_config,
              learner_config = learner_config,
              env_config = env_config, 
              device = args.device)