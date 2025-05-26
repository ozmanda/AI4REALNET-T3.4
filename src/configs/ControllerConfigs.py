from src.algorithms.PPO.PPOController import PPOController
from typing import Dict, List, Union
from src.utils.obs_utils import calculate_state_size

class PPOControllerConfig(): 
    def __init__(self, config_dict: Dict[str, Union[Dict, str, float, int]], device: str) -> None:
        self.device = device
        self.config_dict = config_dict
        self.config_dict['actor_config']['layer_sizes'] = [int(x) for x in config_dict['actor_config']['layer_sizes']]
        self.config_dict['critic_config']['layer_sizes'] = [int(x) for x in config_dict['critic_config']['layer_sizes']]


    def create_controller(self) -> PPOController:
        return PPOController(self.config_dict, self.device) 