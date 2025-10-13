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
from src.utils.observation.obs_utils import calculate_state_size

from src.configs.EnvConfig import create_env_config
from src.configs.ControllerConfigs import PPOControllerConfig
from src.algorithms.PPO.PPOLearner import PPOLearner


def train_ppo(controller_config: PPOControllerConfig, learner_config: Dict, env_config: Dict, device: str) -> None:
    learner = PPOLearner(controller_config=controller_config,
                         learner_config=learner_config,
                         env_config=env_config,
                         device=device)
    learner.sync_run()


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
    parser.add_argument('--config_path', type=str, default='src/configs/PPO_FNN.yaml', help='Path to the configuration file')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the training on (cpu or cuda)')
    args = parser.parse_args()


    # Load config file
    config = load_config_file(args.config_path)

    # prepare environment config
    if args.random_seed:
        config['environment_config']['random_seed'] = args.random_seed
    env_config = create_env_config(config['environment_config'])

    # prepare controller config and setup parallelisation
    learner_config = config['learner_config']

    # prepare controller
    controller_config_dict = config['controller_config']
    env_type = getattr(env_config, 'env_type', 'flatland')
    if env_type == 'flatland':
        n_nodes, state_size = calculate_state_size(env_config.observation_builder_config['max_depth'])
        controller_config_dict['n_nodes'] = n_nodes
        controller_config_dict['state_size'] = state_size
    else:
        controller_config_dict['state_size'] = getattr(env_config, 'state_size')
        controller_config_dict['action_size'] = getattr(env_config, 'action_size')
        controller_config_dict['n_nodes'] = controller_config_dict.get('n_nodes', 1)
        controller_config_dict['n_features'] = controller_config_dict['state_size']
    controller_config = PPOControllerConfig(controller_config_dict)


    train_ppo(controller_config = controller_config,
              learner_config = learner_config,
              env_config = env_config, 
              device = args.device)
