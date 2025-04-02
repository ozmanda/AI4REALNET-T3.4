import torch
from src.training.loss import value_loss, value_loss_with_IS, policy_loss
from torch import Tensor
from typing import Dict, Tuple, List
from src.algorithms.PPO.PPOController import PPOController
from src.algorithms.PPO.PPORollout import PPORollout
from src.algorithms.PPO.PPORunner import PPORunner

class PPOLearner():
    """
    Learner class for the PPO algorithm, updates the policy based on experiences (rollouts).
    """
    def __init__(self, args, device: str, controller: PPOController) -> None: 
        pass


    def _optimise(self, rollouts_dict: Dict):
        pass

    def _loss(self, state: Tensor, action: int, old_log_prob, reward, next_state, done, gae, neighbours_states, step) -> float:
        pass

    def rollouts(self, max_optim_steps: int, max_episodes:int) -> None:
        """
        Run the PPO algorithm for a given number of episodes and optimisation steps.

        Parameters:
            - max_optim_steps   int     Maximum number of optimisation steps per episode
            - max_episodes      int     Maximum number of episodes to run
        """
        pass

