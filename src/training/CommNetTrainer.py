from argparse import Namespace
from torch import optim, Tensor
from networks.CommNet import CommNet
from flatland.envs.rail_env import RailEnv
from training.Trainer import Transition
from typing import List, Tuple, Dict
from utils.utils import merge_dicts, dict_tuple_to_tensor
from utils.action_utils import sample_action, action_tensor_to_dict
from utils.obs import obs_dict_to_tensor
from utils.reward_utils import compute_discounted_reward_per_agent

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
        step = 0
        while True: 
            # TODO: check dimension of action_log_probs
            if self.lstm:
                action_log_probs, value, next_hidden_state, next_cell_state = self.policy_net(obs_tensor, prev_hidden_state, prev_cell_state)
            else: 
                action_log_probs, value, next_hidden_state = self.policy_net(obs_tensor, prev_hidden_state)

            if (step + 1) % self.args.detach_gap == 0:
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

            if self.lstm:
                transition = Transition(obs_tensor, actions_tensor, action_log_probs, prev_hidden_state, prev_cell_state, value, reward, done_dict)
            else: 
                transition = Transition(obs_tensor, actions_tensor, action_log_probs, prev_hidden_state, None, value, reward, done_dict)
            episode.append(transition)
            obs_dict = next_obs_dict

            step += 1
            if step >= self.args.max_steps or all(done_mask):
                break
        
        return episode 


    def compute_grad(self, batch: List[Transition]) -> Dict[str, float]:
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