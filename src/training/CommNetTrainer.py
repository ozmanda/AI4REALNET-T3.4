from argparse import Namespace
import torch
from torch import optim, Tensor
from torch.nn.parameter import Parameter
from src.networks.CommNet import CommNet
from flatland.envs.rail_env import RailEnv
from src.training.Trainer import Transition
from typing import List, Tuple, Dict, Any
from src.utils.utils import merge_dicts, dict_tuple_to_tensor
from src.utils.action_utils import sample_action, action_tensor_to_dict
from src.utils.observation.obs_utils import obs_dict_to_tensor, _calculate_tree_nodes
from src.reward.reward_utils import compute_discounted_reward_per_agent
from src.training.RNNTrainer import RNNTrainer

class CommNetTrainer(RNNTrainer):
    def __init__(self, args, policy_net: CommNet, env: RailEnv) -> None:
        self.args: Namespace = args
        self.policy_net: CommNet = policy_net
        self.env: RailEnv = env
        self.agent_ids: range = env.get_agent_handles()
        self.observation_type: str = args.observation_type
        self.recurrent = args.recurrent
        if self.observation_type == 'tree':
            self.max_tree_depth: int = args.max_tree_depth
            self.tree_nodes = _calculate_tree_nodes(self.max_tree_depth)
            self.obs_features: int = 12

        self.stats: dict = dict()

        self.optimizer = optim.rmsprop.RMSprop(policy_net.parameters(), lr=args.lr, alpha=args.alpha, eps=args.eps)
        self.params: List[Parameter] = [p for p in self.policy_net.parameters()]

    
    def get_episode(self) -> Tuple[List[Transition], Dict[str, Any]]:
        """
        Performs one episode in the environment and gathers the transitions
    
        Return: 
            1. List[Transitions]
                - state     Dict[int, Dict]
                - action    Dict[int, int]
                - value     Dict[int, float]
                - reward    Dict[int, float]
                - done      Dict[int, bool]

            2. episode_info     Dict
                - num_steps     int
                - sum_reward    float   -> sum of all rewards

        """
        # for global observation: (n_agents, (env_height, env_width, 23))
        # for tree observation: (n_agents, 1)
        output: Tuple[Dict, Dict] = self.env.reset()
        self.args.n_agents = self.env.get_num_agents()
        obs_dict, info_dict = output

        # for global observation: (n_agents, env_height * env_width * 23)
        # for tree observation: (n_agents, n_nodes * obs_features)
        obs_tensor: Tensor = obs_dict_to_tensor(obs_dict, self.observation_type, self.n_agents, self.max_tree_depth, self.tree_nodes) 
        

        # initialisations
        if self.lstm:
            prev_hidden_state, prev_cell_state = self.policy_net.init_hidden(1)
        else: 
            prev_hidden_state = self.policy_net.init_hidden(1)

        episode: List = []
        episode_info: Dict[str, Any] = dict()
        episode_info['num_steps'] = 0
        episode_info['agent_reward'] = [0]*self.n_agents
        episode_info['sum_reward'] = 0

        # initialise comm_action
        episode_info['comm_action'] = torch.zeros(self.n_agents, dtype=int)

        while True: 
            # TODO: check dimension of action_log_probs
            if self.lstm:
                action_log_probs, value, next_hidden_state, next_cell_state = self.policy_net(obs_tensor, prev_hidden_state, prev_cell_state)
            else: 
                action_log_probs, value, next_hidden_state = self.policy_net(obs_tensor, prev_hidden_state)

            if (episode_info['num_steps'] + 1) % self.args.detach_gap == 0:
                next_hidden_state = next_hidden_state.detach()
                if self.lstm:
                    next_cell_state = next_cell_state.detach()

            # TODO: check dimension of sampled action
            actions_tensor: Tensor = sample_action(action_log_probs)
            actions_dict: Dict = action_tensor_to_dict(actions_tensor, self.agent_ids)
            # TODO: check what datatypes are returned
            next_obs_dict, reward, done_dict, info = self.env.step(actions_dict)

            # create alive mask for agents that are already done to mask reward
            done_mask = [done_dict[agent_id] for agent_id in self.agent_ids]

            # gather important stats
            episode_info['sum_reward'] += sum(reward.values())
            episode_info['agent_reward'] += list(reward.values())

            # Define Transition tuple
            if self.lstm:
                transition = Transition(obs_tensor, actions_tensor, action_log_probs, prev_hidden_state, prev_cell_state, value, reward, done_dict)
            else: 
                transition = Transition(obs_tensor, actions_tensor, action_log_probs, prev_hidden_state, None, value, reward, done_dict)
            episode.append(transition)
            obs_dict = next_obs_dict

            episode_info['num_steps'] += 1
            if episode_info['num_steps'] >= self.args.max_steps or all(done_mask):
                break
        
        return episode, episode_info
