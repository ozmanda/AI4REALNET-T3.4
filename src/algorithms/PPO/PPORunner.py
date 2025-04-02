import torch
from torch import Tensor
from src.algorithms.PPO.PPORollout import PPORollout, PPOTransition
from src.algorithms.PPO.PPOController import PPOController
from flatland.envs.rail_env import RailEnv
from typing import Dict, Tuple, List

class PPORunner():
    def __init__(self, env: RailEnv, controller: PPOController):
        self.env: RailEnv = env
        self.controller: PPOController = controller 


    def run(self) -> Tuple[PPORollout, Dict]:
        """
        Run a single episode in the environment and collect rollouts.

        Parameters:
            - env           RailEnv         The environment to run the episode in
            - controller    PPOController   The controller to use for action selection

        Returns:
            - rollout       PPORollout      The collected rollouts
            - stats         Dict            Statistics about the episode (e.g., rewards, lengths)
        """
        pass


    def _select_actions(self, state: Dict[int, Tensor]) -> Tuple:
        """
        Select actions for all agents based on their current states.
        """
        pass


    def _save_transitions(self, state_dict: Dict, action_dict: Dict, log_probs: Dict, next_state: Dict, reward: Dict, done: Dict, step: int) -> None:
        """
        Save transitions for each agent in the environment.
        """
        pass


    def _wrap(self, obs: Dict) -> Dict:
        for key, value in obs.items():
            if isinstance(value, Tensor):
                obs[key] = torch.tensor(value, dtype=torch.float64)
        return obs

