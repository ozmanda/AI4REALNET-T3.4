from src.algorithms.PPO.PPOController import PPOController
from typing import Dict, List

class PPOController(): 
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 neighbour_depth: int, 
                 optimiser_config: Dict[str],
                 batch_size: int,
                 gae_horizon: float, 
                 n_epochs_update: int, 
                 gamma: float,
                 lam: float, 
                 clip_epsilon: float,
                 value_loss_coefficient: float, 
                 entropy_coefficient: float,
                 actor_layers_sizes: List[int],
                 critic_layers_sizes: List[int], 
                 ):
        self.state_size = state_size
        self.action_size = action_size
        self.neighbour_depth = neighbour_depth
        self.optimiser_config = optimiser_config
        self.batch_size = batch_size
        self.gae_horizon = gae_horizon
        self.n_epochs_update = n_epochs_update
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.value_loss_coefficient = value_loss_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.actor_layers_sizes = actor_layers_sizes
        self.critic_layers_sizes = critic_layers_sizes

    # TODO: implement config loading from .yaml
        
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
                             critic_layers_sizes=self.critic_layers_sizes)