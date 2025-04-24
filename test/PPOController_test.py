import sys
sys.path.append('../src')
import unittest
from argparse import Namespace
import torch
from torch import Tensor
import numpy as np
from src.algorithms.PPO.PPOController import PPOController
from flatland.envs.rail_env import RailEnv
from src.utils.obs_utils import calculate_state_size
from src.configs.EnvConfig import FlatlandEnvConfig
from typing import Dict, Tuple


class PPOController_Test(unittest.TestCase):
    def setup(self) -> PPOController: 
        env, max_depth = self.setup_small_env()
        state_size = calculate_state_size(max_depth)
        self.test_config_dict = {
            'state_size': state_size,
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
                'neighbour_depth': 3
            },
            'critic_layer_sizes': [128, 64]
        }
        return PPOController(self.test_config_dict, device='cpu')
    

    def setup_small_env(self) -> Tuple[RailEnv, int]:
        # create flatland env with default settings for testing
        envconfig = FlatlandEnvConfig(observation_builder_config={'max_depth': 3,
                                                                  'type': 'tree',
                                                                  'predictor': 'shortest_path'},
                                      malfunction_config={'malfunction_rate': 0.0,
                                                          'min_duration': 0.0, 
                                                          'max_duration': 0.0})
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
            self.assertIsInstance(actions[agent_id], int)

        # check log_probs
        self.assertIsInstance(log_probs, dict)
        self.assertEqual(len(log_probs), n_agents)
        for agent_id in handles:
            self.assertIn(agent_id, log_probs)
            self.assertIsInstance(log_probs[agent_id], Tensor)


    def test_env_config(self):
        # Test the environment configuration
        env, max_depth = self.setup_small_env()
        obs, info = env.reset()
        self.assertIsInstance(obs, dict)