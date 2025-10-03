import sys
sys.path.append('../src')
import unittest
from argparse import Namespace
import torch
from torch import Tensor
import numpy as np
from src.algorithms.PPO_JBR_HSE.PPOController import PPOController
from src.algorithms.PPO_JBR_HSE.PPORunner import PPORunner
from src.configs.ControllerConfigs import PPOControllerConfig
from flatland.envs.rail_env import RailEnv
from src.utils.file_utils import load_config_file
from src.utils.observation.obs_utils import calculate_state_size
from src.configs.EnvConfig import FlatlandEnvConfig
from typing import Dict, Tuple


class PPOController_Test(unittest.TestCase):
    def setup(self) -> PPOController: 
        env, max_depth = self.setup_small_env()
        n_nodes, state_size = calculate_state_size(max_depth)
        self.test_config_dict = {
            'state_size': state_size,
            'n_features': 12,
            'action_size': 5,
            'neighbour_depth': 3,
            'batch_size': 32,
            'gae_horizon': 20,
            'n_epochs_update': 10,
            'gamma': 0.99,
            'lam': 0.95,
            'tau': 0.99,
            'clip_epsilon': 0.2,
            'value_loss_coefficient': 0.5,
            'entropy_coefficient': 0.01,
            'actor_config': {
                'hidden_size': 64,
                'layer_sizes': [128, 64],
                'thought_size': 32,
                'intent_size': 16,
                'activation_function': 'relu',
                'n_heads': 3,
                'neighbour_depth': 3,
                'n_nodes': n_nodes
            },
            'critic_config': {
                'layer_sizes': [128, 64],
                'n_nodes': n_nodes
            }
        }
        return PPOController(self.test_config_dict, device='cpu')
    

    def setup_small_env(self) -> Tuple[RailEnv, int]:
        env_config = {
            'height': 30,
            'width': 30,
            'n_agents': 8,
            'n_cities': 4,
            'grid_distribution': False,
            'max_rails_between_cities': 2,
            'max_rail_pairs_in_city': 2,
            'observation_builder_config': {'type': 'tree',
                                           'predictor': 'shortest_path',
                                           'max_depth': 3},
            'malfunction_config': {'malfunction_rate': 0.001,
                                   'min_duration': 20,
                                   'max_duration': 50},
            'speed_ratios': {1.0: 0.7,
                             0.5: 0.3},
            'reward_config': 0,
            'random_seed': 42
            }
        envconfig = FlatlandEnvConfig(env_config)
        max_depth = envconfig.observation_builder_config['max_depth']
        return envconfig.create_env(), max_depth

    
    def test_PPOController(self):
        # Create a mock PPOController object
        controller = self.setup()
        n_agents = 5
        handles = list(range(n_agents))
        
        # Create mock input data
        states: Dict[int, Tensor] = {}
        for agent_id in handles:
            states[agent_id] = torch.randn(self.test_config_dict['batch_size'], 
                                           self.test_config_dict['state_size'])

        # Create mock neighbour states for each agent
        neighbour_states: Dict[int, Tensor] = {}
        for agent_id in handles:
            neighbour_states[agent_id] = torch.randn(self.test_config_dict['batch_size'], 
                                                     self.test_config_dict['state_size'])
        
        actions, log_probs = controller.select_action(handles, states, neighbour_states)

        # check actions
        self.assertIsInstance(actions, dict)
        self.assertEqual(len(actions), n_agents)
        for agent_id in handles:
            self.assertIn(agent_id, actions)
            self.assertEqual(actions[agent_id][0].dim(), 0)

        # check log_probs
        self.assertIsInstance(log_probs, dict)
        self.assertEqual(len(log_probs), n_agents)
        for agent_id in handles:
            self.assertIn(agent_id, log_probs)
            self.assertIsInstance(log_probs[agent_id][0], Tensor)


    def test_env_config(self):
        # Test the environment configuration
        env, max_depth = self.setup_small_env()
        obs, info = env.reset()
        self.assertIsInstance(obs, dict)


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