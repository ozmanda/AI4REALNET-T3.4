from src.algorithms.PPO.PPOController import PPOController
from typing import Dict, List, Union

class PPOControllerConfig(): 
    def __init__(self, config_dict: Dict[str, Union[Dict, str, float, int]]):
        self.config_dict = config_dict
        self.config_dict['actor_config']['actor_layers_sizes'] = [int(x) for x in config_dict['actor_layers_sizes'].values()]
        self.config_dict['actor_config']['critic_layers_sizes'] = [int(x) for x in config_dict['critic_layers_sizes'].values()]


    def create_controller(self) -> PPOController:
        return PPOController(state_size=self.state_size,
                             action_size=self.action_size,
                             neighbour_depth=self.neighbour_depth,
                             optimiser_config=self.optimiser_config,
                             batch_size=self.batch_size,
                             gae_horizon=self.gae_horizon,
                             n_epochs_update=self.n_epochs_update,
                             gamma=self.gamma,
                             lam=self.lam,
                             clip_epsilon=self.clip_epsilon,
                             value_loss_coefficient=self.value_loss_coefficient,
                             entropy_coefficient=self.entropy_coefficient,
                             actor_layers_sizes=self.actor_layers_sizes,
                             critic_layers_sizes=self.critic_layers_sizes) #TODO: this isn't correct, move to Namespace for laoding in PPOController