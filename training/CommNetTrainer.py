from argparse import Namespace
from torch import optim, Tensor
from networks.CommNet import CommNet
from flatland.envs.rail_env import RailEnv
from training.Trainer import Transition
from typing import List, Tuple, Dict

class CommNetTrainer():
    def __init__(self, args, policy_net: CommNet, env: RailEnv) -> None:
        self.args: Namespace = args
        self.policy_net: CommNet = policy_net
        self.env: RailEnv = env
        self.agent_ids = env.get_agent_handles()
        self.observation_type = args.observation_type
        if self.observation_type == 'tree':
            self.max_tree_depth = args.max_tree_depth
            self.tree_nodes = int((4 ** (self.max_tree_depth + 1) - 1) / 3) #* geometric progression
            self.obs_features: int = 12

        self.stats: dict = dict()

        self.optimizer = optim.rmsprop.RMSprop(policy_net.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)
        self.params = [p for p in self.policy_net.parameters()]

    
    def get_episode(self) -> List[Transition]:
        """
        Forward pass of the LSTM module.
    
        Input:
            - observations          Tensor (batch_size * n_agents, num_inputs)
            - prev_hidden_state     Tensor (batch_size * n_agents, hid_size)
            - prev_cell_state       Tensor (batch_size * n_agents, hid_size)
    
        Return: Tuple
            - episode               List[Transitions]
        """
        pass


    def compute_grad(self, batch: List[Transition]) -> Dict[str: float]:
        """
        Compute the gradients for the policy network.
    
        Input:
            - batch                 List[Transitions]
        """
        pass


    def run_batch(self) -> None: 
        """
        Run one batch through the policy network.
    
        Input:
            - batch                 List[Transitions]
        """
        pass
        pass