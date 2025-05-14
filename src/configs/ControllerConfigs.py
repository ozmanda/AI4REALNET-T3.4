from src.algorithms.PPO.PPOController import PPOController
from typing import Dict, List, Union

class PPOControllerConfig(): 
    def __init__(self, config_dict: Dict[str, Union[Dict, str, float, int]], device: str) -> None:
        self.device = device
        self.config_dict = config_dict
        self.config_dict['actor_config']['actor_layers_sizes'] = [int(x) for x in config_dict['actor_layers_sizes'].values()]
        self.config_dict['critic_config']['critic_layers_sizes'] = [int(x) for x in config_dict['critic_layers_sizes'].values()]


    def create_controller(self) -> PPOController:
        return PPOController(self.config_dict, self.device) 