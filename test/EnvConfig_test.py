import unittest
from src.utils.file_utils import load_config_file
from src.configs.EnvConfig import FlatlandEnvConfig
from flatland.envs.rail_env import RailEnv

class EnvConfig_Test(unittest.TestCase):
    def load_config(self) -> FlatlandEnvConfig:
        config = load_config_file('src/configs/ppo_config.yaml')
        return config['environment_config']
    
    def test_env_creation(self) -> None:
        env_config = self.load_config()
        env: RailEnv = FlatlandEnvConfig(env_config).create_env()
        self.assertIsNotNone(env, "Environment should be created successfully")
        self.assertTrue(isinstance(env, RailEnv), "Created object should be of type RailEnv")
