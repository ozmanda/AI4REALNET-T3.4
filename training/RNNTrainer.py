'''
Adapted from IC3Net trainer.py
Two subclasses of Trainer from Trainer.py specifically for LSTM and RNN policies
'''
from training.Trainer import Trainer
from torch import optim, Tensor
import torch.nn as nn
from typing import Tuple, List, Union, Dict
import torch
from networks.RNN import RNN, LSTM
from flatland.envs.rail_env import RailEnv
from argparse import Namespace
from utils.utils import merge_dicts, dict_tuple_to_tensor
from utils.action_utils import sample_action, action_tensor_to_dict
from utils.obs import obs_dict_to_tensor
from flatland.envs.rail_env import RailEnv
from training.Trainer import Transition

from utils.reward_utils import compute_discounted_reward_per_agent

class RNNTrainer(): 
    """ Trainer class for the no-communication recurrent policy networks, generalised to both the RNN and LSTM versions """

    def __init__(self, args: Namespace, policy_net: Union[RNN, LSTM], env: RailEnv) -> None:
        self.args: Namespace = args
        self.policy_net: Union[RNN, LSTM] = policy_net
        self.env: RailEnv = env
        self.n_agents: int = self.args.n_agents
        self.agent_ids: range = env.get_agent_handles()
        self.n_actions: int = len(env.action_space)
        self.lstm: bool = True if args.rnn_type == 'lstm' else False

        self.observation_type: str = args.observation_type
        if self.observation_type == 'tree':
            self.max_tree_depth: int = args.max_tree_depth
            self.tree_nodes: int = int((4 ** (self.max_tree_depth + 1) - 1) / 3) #* geometric progression
            self.obs_features: int = 12
            self.n_obs: int = self.tree_nodes * self.obs_features
        elif self.observation_type == 'global':
            self.n_obs: int = self.n_agents * env.width * env.height * 23

        self.info: dict = dict()

        self.optimizer = optim.RMSprop(policy_net.parameters(), lr=args.learning_rate, alpha=args.alpha, eps=args.eps)
        self.params: List[nn.Parameter] = [p for p in self.policy_net.parameters()]

        # target actor network update frequency
        self.optimiser_steps: int = 0
        self.target_actor_update_freq: int = args.target_update_freq

    
    def get_episode(self) -> List[Transition]:
        """
        Performs one episode in the environment and gathers the transitions
    
        Return: List[Transitions]
            - state     Dict[int, Dict]
            - action    Dict[int, int]
            - value     Dict[int, float]
            - reward    Dict[int, float]
            - done      Dict[int, bool]
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
    

    def run_batch(self) -> Tuple[Transition, Dict]:
        """
        Gathers one batch of training data from multiple episodes. The length of the batch can
        be greater that args.batch_size, as it is determined by the number of steps in the episodes.
        # TODO: is it better to have complete episodes or a fixed batch size?
    
        Return: # TODO: finish listing transition fields
            - batch         Tranisition with fields consisting of lists of length num_steps
                - action    list of Tensors ()
            - batch_info    dict
        """
        batch = []
        batch_info: Dict = dict()
        batch_info['num_episodes'] = 0

        while len(batch) < self.args.batch_size:
            episode: List[Transition] = self.get_episode()
            batch_info['num_episodes'] += 1
            batch.extend(episode)
        
        batch_info['num_steps'] = len(batch)
        batch = Transition(*zip(*batch)) # turns a list of transitions into a transition containing lists
        return batch, batch_info


    def train_batch(self):
        """
        Trains the network on one batch of transitions.
    
        Output:
            - performs a single optimizer step
    
        Return:
            - grad_stats    Dict
        """
        batch, batch_info = self.run_batch()
        self.optimizer.zero_grad()

        grad_info = self.compute_gradient(batch)

        for parameter in self.params:
            if parameter._grad is not None:
                parameter._grad.data /= grad_info['num_steps']
        self.optimizer.step()
        self.optimiser_steps += 1

        # perform target network update
        if self.optimiser_steps % self.target_actor_update_freq == 0:
            self.policy_net.actor_target.load_state_dict(self.policy_net.actor.state_dict())

        return merge_dicts(grad_info, batch_info)


    def compute_gradient(self, batch: Transition) -> Dict[str, float]:
        """
        Computes the PPO loss metric using the following formulae: 
            PPO_loss = E[CLIP_loss - c1 * value_loss + c2 * entropy]
            CLIP_loss = E[min(ratio * advantage, clip(ratio, 1 - epsilon, 1 + epsilon) * advantage)]
            value_loss = E[(value - returns)^2]
            entropy = E[-sum(prob * log(prob))]
    
        Input: batch            Transition containing lists of length batch_size for each field
            - state             Tensor(n_agents, n_obs)      [float] -> observation
            - action            Tensor(n_agents)             [int]   -> action taken
            - action_log_prob   Tensor(n_agents, n_actions)  [float] -> log prob of each action
            - value             Tensor(n_agents)             [float] -> predicted next state value
            - reward            Tensor(n_agents)             [float]
            - done              Tensor(n_agents)             [bool]
    
        Return: gradient_info   dict
            - action_loss
            - value_loss
            - entropy
        """
        gradient_info: Dict = {}
        batch_size = len(batch.state)

        # convert lists to tensors
        observations = torch.stack(batch.state)
        actions = torch.stack(batch.action)
        action_log_probs = torch.stack(batch.action_log_prob) # (batch_size, n_agents, n_actions)
        values = torch.stack(batch.value)
        rewards = dict_tuple_to_tensor(batch.reward)
        # rewards = torch.stack(batch.reward)
        dones = dict_tuple_to_tensor(batch.done)
        # dones = torch.stack(batch.done)
        hidden_states = (torch.stack(batch.hidden_states), torch.stack(batch.cell_states))

        returns: Tensor = torch.zeros(batch_size, self.n_agents)
        advantages: Tensor = torch.zeros(batch_size, self.n_agents)
        future_rewards: Tensor = torch.zeros(self.n_agents)

        # advantage calculation
        #! assumes that all elements in the batch are from the same episode
        discounted_rewards: Tensor = compute_discounted_reward_per_agent(rewards, self.args.gamma) # (batch_size, n_agents)
        advantages: Tensor = discounted_rewards - values # (batch_size, n_agents)
        if self.args.normalise_advantage: 
            advantages = (advantages - advantages.mean()) / advantages.std()

        # clipped surrogate objective
        CLIP_loss: Tensor = torch.zeros(batch_size, self.n_agents)

        # probability ratio # TODO: where to get prev_hidden_state and _cell from?
        target_log_probs = self.policy_net.forward_target_network(observations, hidden_states)
        ratio: Tensor = torch.exp(action_log_probs - target_log_probs) 
        clipped_ratio: Tensor = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon)
        clipped_ratio = clipped_ratio.sum(dim=0)


        # Value Loss calculation
        value_loss: Tensor = (values - discounted_rewards).pow(2)
        value_loss *= ~dones
        value_loss = value_loss.mean() #! original code has sum, but "E[]" is a mean function (?)


        # Entropy calculation
        entropy: Tensor = torch.sum(-torch.exp(action_log_probs) * action_log_probs, dim=1).mean() # (batch_size, n_agents)


        # compute PPO loss
        ppo_loss: Tensor = CLIP_loss - self.args.value_coefficient * value_loss + self.args.entropy_coefficient * entropy

        grad_dict = {'ppo_loss': ppo_loss, 
                     'CLIP_loss': CLIP_loss,
                     'value_loss': value_loss, 
                     'entropy': entropy}
        
        ppo_loss.backward()
        
        return grad_dict