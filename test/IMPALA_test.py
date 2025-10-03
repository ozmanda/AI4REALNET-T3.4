import torch
import random
import unittest
import numpy as np

from typing import Dict

from src.configs.EnvConfig import FlatlandEnvConfig
from src.configs.ControllerConfigs import PPOControllerConfig
from src.configs.ControllerConfigs import LSTMControllerConfig

from src.algorithms.IMPALA.IMPALALearner import IMPALALearner

from src.utils.file_utils import load_config_file
from src.utils.observation.obs_utils import calculate_state_size


class TestIMPALA(unittest.TestCase):
    def setup_controller(self, config_path: str) -> None:
        config = load_config_file(config_path)

        # prepare environment
        self.env_config = FlatlandEnvConfig(config['environment_config'])

        # prepare learner config
        self.learner_config = config['learner_config']
        self.learner_config['n_workers'] = 2
        self.learner_config['target_updates'] = 2

        # prepare controller
        controller_config = config['controller_config']
        controller_config['n_agents'] = self.env_config.n_agents
        controller_config['n_nodes'], controller_config['state_size'] = calculate_state_size(self.env_config.observation_builder_config['max_depth'])

        if controller_config['type'] == 'FNN':
            self.controller_config = PPOControllerConfig(controller_config)
        elif controller_config['type'] == 'LSTM':
            self.controller_config = LSTMControllerConfig(controller_config)


    def train_impala(self, random_seed: int, controller_config: PPOControllerConfig, learner_config: Dict, env_config: Dict, device: str) -> None:
        self.init_random_seeds(random_seed)
        print("Initializing IMPALA learner...")
        learner = IMPALALearner(controller_config=controller_config,
                            learner_config=learner_config,
                            env_config=env_config,
                            device=device)
        print("Starting IMPALA training...")
        learner.async_run()


    def init_random_seeds(self, random_seed: int, cuda_deterministic: bool = False) -> None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        if cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    def test_impala_FNN(self):
        config_path: str = 'src/configs/IMPALA_FNN.yaml'
        self.setup_controller(config_path)
        self.train_impala(random_seed=42,
                          controller_config=self.controller_config,
                          learner_config=self.learner_config,
                          env_config=self.env_config,
                          device='cpu')
        

    def test_impala_LSTM(self):
        config_path: str = 'src/configs/IMPALA_LSTM.yaml'
        self.setup_controller(config_path)
        self.train_impala(random_seed=42,
                          controller_config=self.controller_config,
                          learner_config=self.learner_config,
                          env_config=self.env_config,
                          device='cpu')