from src.controllers.PPOController import PPOController
from src.controllers.LSTMController import LSTMController
from src.controllers.BaseController import Controller
from typing import Dict, List, Union
from src.utils.observation.obs_utils import calculate_state_size

class ControllerConfig():
    def __init__(self, config_dict: Dict) -> None:
        self.config_dict = config_dict
        self.type = config_dict.get('type', 'FNN')

    def create_controller(self) -> Controller:
        raise NotImplementedError("This method should be overridden by subclasses.")

class PPOControllerConfig(ControllerConfig): 
    def __init__(self, config_dict: Dict[str, Union[Dict, str, float, int]]) -> None:
        super().__init__(config_dict)

    def create_controller(self) -> PPOController:
        return PPOController(self.config_dict) 

class LSTMControllerConfig(ControllerConfig): 
    def __init__(self, config_dict: Dict) -> None:
        super().__init__(config_dict)

    def create_controller(self) -> LSTMController:
        return LSTMController(self.config_dict)