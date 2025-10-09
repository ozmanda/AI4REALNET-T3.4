import torch
import random
import unittest
import numpy as np

from typing import Dict

from src.algorithms.PPO.PPOLearner import PPOLearner

from src.configs.ControllerConfigs import PPOControllerConfig
from src.configs.EnvConfig import FlatlandEnvConfig

from src.utils.file_utils import load_config_file
from src.utils.observation.obs_utils import calculate_state_size


class PPO_Test(unittest.TestCase):
    def setup(self, config_path: str) -> None:
        config = load_config_file(config_path)

        # prepare environment
        self.env_config = FlatlandEnvConfig(config['environment_config'])

        # prepare learner config - reduce for testing
        self.learner_config = config['learner_config']
        self.learner_config['run_name'] = "test_run"
        self.learner_config['n_workers'] = 2
        self.learner_config['target_updates'] = 2
        self.learner_config['max_steps_per_episode'] = 100

        # prepare controller
        controller_config = config['controller_config']
        controller_config['n_nodes'], controller_config['state_size'] = calculate_state_size(self.env_config.observation_builder_config['max_depth'])

        self.controller_config = PPOControllerConfig(controller_config)


    def train_ppo(self, random_seed: int, controller_config: PPOControllerConfig, learner_config: Dict, env_config: Dict, device: str) -> None:
        self.init_random_seeds(random_seed)
        learner = PPOLearner(controller_config=controller_config,
                            learner_config=learner_config,
                            env_config=env_config,
                            device=device)
        learner.sync_run()


    def init_random_seeds(self, random_seed: int, cuda_deterministic: bool = False) -> None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        if cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    def test_FNN_random_env(self): 
        config_path: str = 'src/configs/PPO_FNN.yaml'
        self.setup(config_path)
        self.train_ppo(random_seed = 42,
                       controller_config = self.controller_config,
                       learner_config = self.learner_config,
                       env_config = self.env_config,
                       device = 'cpu')
        

    def test_FNN_scenario(self): 
        config_path: str = 'src/configs/PPO_FNN_Scenario.yaml'
        self.setup(config_path)
        self.train_ppo(random_seed = 42,
                       controller_config = self.controller_config,
                       learner_config = self.learner_config,
                       env_config = self.env_config,
                       device = 'cpu')
        
    def test_LSTM(self): 
        config_path: str = 'src/configs/PPO_LSTM.yaml'
        self.setup(config_path)
        self.train_ppo(random_seed = 42,
                       controller_config = self.controller_config,
                       learner_config = self.learner_config,
                       env_config = self.env_config,
                       device = 'cpu')