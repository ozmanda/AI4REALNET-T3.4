import wandb
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, List, Tuple
from itertools import chain

from flatland.envs.rail_env import RailEnv

from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer
from src.networks.FeedForwardNN import FeedForwardNN

from src.training.loss import value_loss, value_loss_with_IS, policy_loss

class PPOAgent():
    def __init__(self, config: Dict, agent_ID: Union[int, str] = None):
        self.config: Dict = config
        if agent_ID:
            self.agent_ID: Union[int, str] = agent_ID
        self._init_hyperparameters(config)

        self.actor: nn.Module = self._build_actor()
        self.critic: nn.Module = self._build_critic()
        self.optimizer: optim.Optimizer = self._build_optimizer(config['optimiser_config'])
        self.update_step: int = 0

    def _init_hyperparameters(self, config: Dict) -> None:
        """
        Initialize hyperparameters from the configuration dictionary.
        """
        self.action_size: int = config['action_size']
        self.state_size: int = config['state_size']
        self.gamma: float = config['gamma']
        self.lam: float = config['lam']
        self.gae_horizon: int = config['gae_horizon']
        self.clip_epsilon: float = config['clip_epsilon']
        self.value_loss_coef: float = config['value_loss_coefficient']
        self.entropy_coef: float = config['entropy_coefficient']
        # self.max_grad_norm: float = config['max grad norm']

    def _build_actor(self) -> nn.Module:
        self.actor_network = FeedForwardNN(self.state_size, self.config['actor_config']['hidden_size'], self.action_size)

    def _build_critic(self) -> nn.Module:
        self.critic_network = FeedForwardNN(self.state_size, self.config['critic_config']['hidden_size'], 1) # TODO: add hidden size to critic config

    def _build_optimizer(self, optimiser_config: Dict[str, Union[int, str]]) -> optim.Optimizer:
        if optimiser_config['type'] == 'adam': 
            self.optimiser = optim.Adam(params=chain(self.actor_network.parameters(), self.critic_network.parameters()), lr=float(optimiser_config['learning_rate']))
        else: 
            raise Warning('Only Adam optimiser has been implemented')


    def update_networks(self, rollout: MultiAgentRolloutBuffer, epochs: int, batch_size: int, IS: bool = False) -> None:
        """
        Update the actor and critic networks using the collected transitions from the rollout buffer.
        1. for each agent and episode, compute the GAE
        3. compute the loss
        4. update the actor and critic networks
        """
        rollout = self._generalised_advantage_estimation(rollout)
        losses = {
            'policy_loss': [],
            'value_loss': []
        }

        for minibatch in rollout.get_minibatches(batch_size=batch_size, shuffle=False):
            # forward pass through the actor network to get actions and log probabilities
            new_log_probs, entropy, new_state_values, new_next_state_values = self._evaluate(minibatch['states'], minibatch['next_states'], minibatch['actions'])
            new_state_values = new_state_values.squeeze(-1)  
            new_next_state_values = new_next_state_values.squeeze(-1)

            # Policy loss
            old_log_probs = minibatch['log_probs']
            actor_loss = policy_loss(gae=minibatch['gaes'],
                                    new_log_prob=new_log_probs,
                                    old_log_prob=old_log_probs,
                                    clip_eps=self.clip_epsilon)
            
            # Value loss
            if IS:
                critic_loss = value_loss_with_IS(state_values=new_state_values,
                                                next_state_values=new_next_state_values,
                                                new_log_prob=new_log_probs,
                                                old_log_prob=old_log_probs,
                                                reward=minibatch['rewards'],
                                                done=minibatch['dones'],
                                                gamma=self.gamma
                                                )
            else:
                critic_loss = value_loss(state_values=new_state_values,
                                        next_state_values=new_next_state_values,
                                        reward=minibatch['rewards'],
                                        done=minibatch['dones'],
                                        gamma=self.gamma)
                
            # Entropy bonus
            entropy_loss = -entropy.mean()  # Encourage exploration

            # Total loss & optimisation step
            total_loss = actor_loss + critic_loss * self.value_loss_coef + entropy_loss * self.entropy_coef
            self.optimiser.zero_grad()
            total_loss.backward()
            self.optimiser.step()
            self.update_step += 1

            # add metrics
            losses['policy_loss'].append(actor_loss.mean().item())
            losses['value_loss'].append(critic_loss.mean().item())
            wandb.log({
                'update_step': self.update_step,
                'train/policy_loss': np.mean(losses['policy_loss']),
                'train/value_loss': np.mean(losses['value_loss'])
            })

        return losses


    def state_values(self, states: Tensor, next_states: Tensor) -> Tensor:
        """
        Get the state values from the critic network for the current and next states.
        
        Parameters:
            - states: Tensor of shape (batch_size, state_size)
            - next_states: Tensor of shape (batch_size, state_size)
        
        Returns:
            - state_values: Tensor of shape (batch_size, 1)
            - next_state_values: Tensor of shape (batch_size, 1)
        """
        state_values = self.critic_network(states)
        next_state_values = self.critic_network(next_states)
        return state_values, next_state_values
        

    def sample_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get the action from the actor network based on the current state.
        
        Parameters:
            - state: Tensor of shape (batch_size, state_size)
        
        Returns:
            - action: Tensor of shape (batch_size, action_size)
            - log_prob: Tensor of shape (batch_size, action_size)
        """
        logits = self.actor_network(state)
        action_distribution = torch.distributions.Categorical(logits=logits)
        actions = action_distribution.sample()
        log_prob = action_distribution.log_prob(actions)
        return actions, log_prob
    

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select the best action based on the current state using the actor network.
        
        Parameters:
            - state: Tensor of shape (batch_size, state_size)
        
        Returns:
            - action: Tensor of shape (batch_size, action_size)
            - log_prob: Tensor of shape (batch_size, action_size)
        """
        with torch.no_grad():
            logits = self.actor_network(state)
            actions = torch.argmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=1)
        return actions, log_probs
        

    def _generalised_advantage_estimation(self, rollout: MultiAgentRolloutBuffer, k: int = 1) -> Tensor:
        """ 
        Compute advantages using Generalized Advantage Estimation (GAE). 
        The parameter k allows for a k-step TD error calculation, where k=1 corresponds to the standard TD(0) method.
        The parameter gae_horizon truncates the GAE calculation to a fixed horizon, which can be useful for stabilizing training.
        """
        with torch.no_grad():
            for idx, episode in enumerate(rollout.episodes):
                for agent in rollout.agent_handles:
                    states = torch.stack(episode['states'][agent])
                    next_states = torch.stack(episode['next_states'][agent])
                    rewards = torch.tensor(episode['rewards'][agent])
                    dones = torch.tensor(episode['dones'][agent]).float()
                    dones = (dones != 0).float()
                    gaes = torch.zeros(len(states))

                    state_values: Tensor = self.critic_network(states).squeeze(-1)
                    next_state_values: Tensor = self.critic_network(next_states).squeeze(-1)

                    rollout.episodes[idx]['state_values'][agent] = state_values
                    rollout.episodes[idx]['next_state_values'][agent] = next_state_values

                    deltas = rewards + self.gamma * next_state_values * (1 - dones) - state_values

                    for i in reversed(range(len(deltas))):
                        gaes[i] = deltas[i] + self.gamma * gaes[i + 1] * (1 - dones[i]) if i < len(deltas) - 2 else deltas[i]

                    rollout.episodes[idx]['gaes'][agent] = gaes

        return rollout

    def _discounted_rewards(self, rewards: Tensor, dones: Tensor, step: int) -> Tensor:
        """ Compute discounted rewards. """
        discounted_rewards = torch.zeros_like(rewards)
        cumulative_reward = 0.0
        for i in reversed(range(len(rewards))):
            cumulative_reward = rewards[i] + (self.gamma * cumulative_reward * (1 - dones[i]))
            discounted_rewards[i] = cumulative_reward
        return discounted_rewards
    

    def _evaluate(self, states: Tensor, next_states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ 
        Forward pass through the actor network to get the action distribution and log probabilities, to compute the entropy of the policy distribution and the current state values.
        """
        logits = self.actor_network(states)
        action_distribution = torch.distributions.Categorical(logits=logits)
        log_probs = action_distribution.log_prob(actions)
        entropy = action_distribution.entropy()
        state_values = self.critic_network(states)
        next_state_values = self.critic_network(next_states)
        return log_probs, entropy, state_values, next_state_values
