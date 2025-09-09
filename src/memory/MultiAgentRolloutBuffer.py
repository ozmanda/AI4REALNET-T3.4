import torch
import numpy as np
from torch import Tensor
from typing import List, Dict, Union
from src.algorithms.PPO_JBR_HSE.PPORollout import Transition


class MultiAgentRolloutBuffer:
    """
    A class to manage the rollout buffer for storing transitions during the training of reinforcement learning agents in multi-agent scenarios.
    """

    def __init__(self, n_agents: int = None) -> None:
        # TODO: class for training with dynamic number of agents
        self.n_agents: int = n_agents
        self.episodes: List[Dict] = []
        self.current_episode: Dict[str, List] = {}
        self.n_episodes: int = 0
        self.total_steps: int = 0


    def reset(self, n_agents) -> None:
        """
        Resets the buffer by clearing all stored transitions.
        """
        self.n_agents = n_agents
        self.episodes: List = []
        self._reset_current_episode()


    def _reset_current_episode(self) -> None:
        self.current_episode: Dict[str, List] = {
            'states': [[] for _ in range(self.n_agents)],
            'state_values': [[] for _ in range(self.n_agents)],
            'actions': [[] for _ in range(self.n_agents)],
            'log_probs': [[] for _ in range(self.n_agents)],
            'rewards': [[] for _ in range(self.n_agents)],
            'next_states': [[] for _ in range(self.n_agents)],
            'next_state_values': [[] for _ in range(self.n_agents)],
            'dones': [[] for _ in range(self.n_agents)],
            'gaes': [[] for _ in range(self.n_agents)],
            'episode_length': [0 for _ in range(self.n_agents)],
            'average_episode_length': 0,
            'average_episode_reward': 0.0
        }


    def add_transitions(self, states: Tensor, actions: Dict[Union[int, str], int], log_probs: Dict[Union[int, str], float], 
                        rewards: Dict[Union[int, str], float], next_states: Tensor, dones: Dict[Union[int, str], bool]) -> None:
        """
        Adds a transition to the buffer.
        """
        for agent_handle in range(self.n_agents):
            self.current_episode['states'][agent_handle].append(states[agent_handle])
            self.current_episode['actions'][agent_handle].append(actions[agent_handle])
            self.current_episode['log_probs'][agent_handle].append(log_probs[agent_handle])
            self.current_episode['rewards'][agent_handle].append(rewards[agent_handle])
            self.current_episode['next_states'][agent_handle].append(next_states[agent_handle])
            self.current_episode['dones'][agent_handle].append(dones[agent_handle])
        self.total_steps += 1


    def add_episode(self, episode: Dict[str, List]) -> None:
        """
        Adds an external episode to the buffer
        """
        self.episodes.append(episode)
        self.total_steps += np.sum(episode['episode_length'])
        self.n_episodes += 1


    def end_episode(self, verbose: int = 0) -> None:
        agent_episode_rewards = []
        total_episode_length = 0
        for agent in range(self.n_agents):
            self.current_episode['episode_length'][agent] += len(self.current_episode['states'][agent])
            agent_episode_rewards.append(sum(self.current_episode['rewards'][agent]) / len(self.current_episode['rewards'][agent]))

        self.current_episode['average_episode_length'] = np.sum(self.current_episode['episode_length']) / self.n_agents
        self.current_episode['average_episode_reward'] = sum(agent_episode_rewards) / self.n_agents

        if verbose > 0:
            print(f"\nEpisode {self.n_episodes + 1} - Average Reward: {self.current_episode['average_episode_reward']}\n")

        self.episodes.append(self.current_episode)
        self.total_steps += np.sum(self.current_episode['episode_length'])
        self._reset_current_episode()
        self.n_episodes += 1


    def get_transitions(self, shuffle: bool = False, gaes: bool = True) -> Dict[str, Tensor]:
        """
        Flatten over all episodes agents and return a tensor for each transition value.
        """
        states = []
        next_states = []
        actions = []
        log_probs = []
        rewards = []
        state_values = []
        next_state_values = []
        dones = []
        if gaes:
            gaes = []
        for episode in self.episodes:
            for agent_handle in range(self.n_agents):
                states.extend(episode['states'][agent_handle])
                next_states.extend(episode['next_states'][agent_handle])
                state_values.extend(episode['states'][agent_handle])
                next_state_values.extend(episode['next_states'][agent_handle])
                actions.extend(episode['actions'][agent_handle])
                log_probs.extend(episode['log_probs'][agent_handle])
                rewards.extend(episode['rewards'][agent_handle])
                dones.extend(episode['dones'][agent_handle])
                if gaes:
                    gaes.extend(episode['gaes'][agent_handle])

        if shuffle:
            indices = torch.randperm(len(states))
            states = [states[i] for i in indices]
            next_states = [next_states[i] for i in indices]
            state_values = [state_values[i] for i in indices]
            next_state_values = [next_state_values[i] for i in indices]
            actions = [actions[i] for i in indices]
            log_probs = [log_probs[i] for i in indices]
            rewards = [rewards[i] for i in indices]
            dones = [dones[i] for i in indices]
            if gaes:
                gaes = [gaes[i] for i in indices]

        return_dict: Dict = {'states': torch.stack(states).clone().detach(),
                'next_states': torch.stack(next_states).clone().detach(),
                'state_values': torch.stack(state_values).clone().detach(),
                'next_state_values': torch.stack(next_state_values).clone().detach(),
                'actions': torch.stack(actions).clone().detach(),
                'log_probs': torch.stack(log_probs).clone().detach(),
                'rewards': torch.tensor(rewards).clone().detach(),
                'dones': torch.tensor(dones, dtype=torch.float32).clone().detach()
                }
        if gaes:
            return_dict['gaes'] = torch.stack(gaes).clone().detach()

        return return_dict
    
    
    def get_minibatches(self, batch_size: int, shuffle: bool = False) -> List[Dict[str, Tensor]]:
        """
        Splits the transitions into minibatches of a specified size.
        """
        transitions = self.get_transitions(shuffle)
        minibatches = []
        for i in range(0, transitions['states'].size(0), batch_size):
            minibatch = {key: value[i:i + batch_size] for key, value in transitions.items()}
            minibatches.append(minibatch)
        return minibatches
    

    def add_episode(self, episode) -> None: 
        """
        Adds an external episode to the buffer
        """
        self.episodes.append(episode)
        self.total_steps += np.sum(episode['episode_length'])
        self.n_episodes += 1

    
    def combine_rollouts(self, rollouts: List["MultiAgentRolloutBuffer"]) -> "MultiAgentRolloutBuffer":
        """
        Combines multiple rollouts into a single MultiAgentRolloutBuffer.
        """
        combined_rollout = MultiAgentRolloutBuffer(n_agents=self.n_agents)
        for rollout in rollouts:
            combined_rollout.episodes.extend(rollout.episodes)
            combined_rollout.total_steps += rollout.total_steps
            combined_rollout.n_episodes += rollout.n_episodes
        return combined_rollout

    def _len(self) -> int:
        """ Returns the total number of steps in the buffer """
        return self.total_steps