import unittest
from typing import Dict, Tuple
import torch
from torch import Tensor

from src.utils.file_utils import load_config_file
from src.utils.obs_utils import calculate_state_size

from src.configs.EnvConfig import FlatlandEnvConfig
from src.configs.ControllerConfigs import PPOControllerConfig
from src.algorithms.PPO.PPORunner import PPORunner

class PPORunner_Test(unittest.TestCase):
    def setup(self) -> Tuple[FlatlandEnvConfig, PPOControllerConfig]:
        config = load_config_file('src/configs/ppo_config.yaml')
        # create flatland env with default settings for testing
        envconfig = FlatlandEnvConfig(config['environment_config'])

        n_nodes, state_size = calculate_state_size(config['environment_config']['observation_builder_config']['max_depth'])
        config['controller_config']['actor_config']['n_nodes'] = n_nodes
        config['controller_config']['critic_config']['n_nodes'] = n_nodes
        config['controller_config']['state_size'] = state_size
        controller = PPOControllerConfig(config['controller_config'], device='cpu')
        return envconfig, controller
    

    def test_run(self):
        # Create a mock environment and controller
        envconfig, controllerconfig = self.setup()
        controller = controllerconfig.create_controller()
        
        # Create a PPORunner instance
        runner = PPORunner(0, envconfig, controller)
        
        # Run the runner for a specified number of steps
        max_steps = 10
        rollouts, stats = runner.run(max_steps)
        
        # Check if the rollouts and stats are not empty
        self.assertIsNotNone(rollouts)
        self.assertIsNotNone(stats)