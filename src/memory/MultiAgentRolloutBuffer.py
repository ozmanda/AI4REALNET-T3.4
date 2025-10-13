import torch
import numpy as np
from torch import Tensor
from typing import List, Dict, Union, Any, Optional

class MultiAgentRolloutBuffer:
    """
    Rollout-Buffer für Multi-Agent RL (PPO).
    Speichert:
      - episodische Returns (Summe der Rewards)
      - Schritt-Mittelwert
      - decision_mask / action_mask in extras
      - success-Flag (1=alle Agenten DONE, 0=Timeout/Abbruch)
    """

    def __init__(self, n_agents: Optional[int] = None) -> None:
        self.n_agents: Optional[int] = n_agents
        self.episodes: List[Dict] = []
        self.current_episode: Dict[str, List] = {}
        self.n_episodes: int = 0
        self.total_steps: int = 0
        self.transitions: Dict[str, Tensor] = {}

    def reset(self, n_agents: int) -> None:
        self.n_agents = n_agents
        self.episodes = []
        self._reset_current_episode()
        self.total_steps = 0
        self.n_episodes = 0
        self.transitions = {}

    def _reset_current_episode(self) -> None:
        assert self.n_agents is not None
        self.current_episode: Dict[str, Any] = {
            'states': [[] for _ in range(self.n_agents)],
            'state_values': [[] for _ in range(self.n_agents)],
            'actions': [[] for _ in range(self.n_agents)],
            'log_probs': [[] for _ in range(self.n_agents)],
            'rewards': [[] for _ in range(self.n_agents)],
            'next_states': [[] for _ in range(self.n_agents)],
            'next_state_values': [[] for _ in range(self.n_agents)],
            'dones': [[] for _ in range(self.n_agents)],
            'extras': {},
            'gaes': [[] for _ in range(self.n_agents)],

            'episode_length': [0 for _ in range(self.n_agents)],
            'episode_return': [0.0 for _ in range(self.n_agents)],
            'episode_reward_mean': [0.0 for _ in range(self.n_agents)],

            # Back-compat / Aggregates
            'episode_reward': [0.0 for _ in range(self.n_agents)],  # = Return (Summe)
            'average_episode_length': 0.0,
            'average_episode_return': 0.0,
            'average_step_reward': 0.0,
            'average_episode_reward': 0.0,  # = average_episode_return
            'total_reward': 0.0,

            # NEW: did this episode end because all agents are DONE?
            'success': 0,
        }

    def add_transitions(
        self,
        states: Tensor,
        actions: Dict[Union[int, str], Tensor],
        log_probs: Tensor,
        rewards: Dict[Union[int, str], float],
        next_states: Tensor,
        state_values: Optional[Tensor],
        next_state_values: Optional[Tensor],
        dones: Dict[Union[int, str], bool],
        extras: Dict[str, Tensor],
    ) -> None:
        assert self.n_agents is not None
        for agent in range(self.n_agents):
            self.current_episode['states'][agent].append(states[agent])
            self.current_episode['actions'][agent].append(actions[agent])
            self.current_episode['log_probs'][agent].append(log_probs[agent])
            self.current_episode['rewards'][agent].append(float(rewards[agent]))
            self.current_episode['next_states'][agent].append(next_states[agent])
            self.current_episode['dones'][agent].append(bool(dones[agent]))

            if state_values is not None and next_state_values is not None:
                self.current_episode['state_values'][agent].append(state_values[agent].squeeze(-1))
                self.current_episode['next_state_values'][agent].append(next_state_values[agent].squeeze(-1))

        if extras:
            if not self.current_episode['extras']:
                self.current_episode['extras'] = {k: [[] for _ in range(self.n_agents)] for k in extras.keys()}
            for k, tensor in extras.items():
                for agent in range(self.n_agents):
                    self.current_episode['extras'][k][agent].append(tensor[agent])

        self.total_steps += 1

    def end_episode(self, verbose: int = 0) -> None:
        assert self.n_agents is not None

        episode_length, episode_return, episode_reward_mean = [], [], []
        for agent in range(self.n_agents):
            ep_len = len(self.current_episode['states'][agent])
            episode_length.append(ep_len)
            if ep_len > 0:
                rew = self.current_episode['rewards'][agent]
                ret = float(np.sum(rew))     # Summe
                mean_step = float(np.mean(rew))
            else:
                ret = 0.0
                mean_step = 0.0
            episode_return.append(ret)
            episode_reward_mean.append(mean_step)

        self.current_episode['episode_length'] = episode_length
        self.current_episode['episode_return'] = episode_return
        self.current_episode['episode_reward_mean'] = episode_reward_mean
        self.current_episode['episode_reward'] = episode_return[:]

        self.current_episode['average_episode_length'] = float(np.mean(episode_length))
        self.current_episode['average_episode_return'] = float(np.mean(episode_return))
        self.current_episode['average_step_reward'] = float(np.mean(episode_reward_mean))
        self.current_episode['total_reward'] = float(np.sum(episode_return))
        self.current_episode['average_episode_reward'] = self.current_episode['average_episode_return']

        if verbose > 0:
            print(
                f"\nEpisode {self.n_episodes + 1} | "
                f"avg_return: {self.current_episode['average_episode_return']:.2f} | "
                f"avg_step_reward: {self.current_episode['average_step_reward']:.3f} | "
                f"avg_len: {self.current_episode['average_episode_length']:.1f} | "
                f"success: {self.current_episode.get('success',0)}\n"
            )

        self.episodes.append(self.current_episode)
        self._reset_current_episode()
        self.n_episodes += 1

    def add_episode(self, episode: Dict[str, Any]) -> None:
        self.episodes.append(episode)
        self.total_steps += int(np.sum(episode['episode_length']))
        self.n_episodes += 1

    def get_transitions(self, gae: bool = True) -> None:
        assert self.n_agents is not None

        states, next_states, actions, log_probs = [], [], [], []
        rewards, dones, state_values, next_state_values = [], [], [], []
        gaes = [] if gae else None

        all_extra_keys = set()
        for ep in self.episodes:
            all_extra_keys.update(ep.get('extras', {}).keys())
        extras_out: Dict[str, List[Tensor]] = {k: [] for k in all_extra_keys}

        for ep in self.episodes:
            for agent in range(self.n_agents):
                states.extend(ep['states'][agent])
                next_states.extend(ep['next_states'][agent])
                actions.extend(ep['actions'][agent])
                log_probs.extend(ep['log_probs'][agent])
                rewards.extend(ep['rewards'][agent])
                dones.extend(ep['dones'][agent])
                state_values.extend(ep['state_values'][agent])
                next_state_values.extend(ep['next_state_values'][agent])
                if gae:
                    gaes.extend(ep['gaes'][agent])

            ep_extras = ep.get('extras', {})
            for k in all_extra_keys:
                if k in ep_extras:
                    for agent in range(self.n_agents):
                        extras_out[k].extend(ep_extras[k][agent])

        transitions_dict: Dict[str, Tensor] = {
            'states': torch.stack(states).detach(),
            'next_states': torch.stack(next_states).detach(),
            'state_values': torch.stack(state_values).detach(),
            'next_state_values': torch.stack(next_state_values).detach(),
            'actions': torch.stack(actions).detach(),
            'log_probs': torch.stack(log_probs).detach(),
            'rewards': torch.tensor(rewards, dtype=torch.float32).detach(),
            'dones': torch.tensor(dones, dtype=torch.float32).detach(),
        }
        if gae:
            transitions_dict['gaes'] = torch.stack(gaes).detach() if len(gaes) > 0 else torch.empty(0)

        if 'decision_mask' in extras_out and len(extras_out['decision_mask']) > 0:
            transitions_dict['decision_mask'] = torch.stack(extras_out['decision_mask']).float()
        if 'action_mask' in extras_out and len(extras_out['action_mask']) > 0:
            transitions_dict['action_mask'] = torch.stack(extras_out['action_mask']).bool()

        transitions_dict['extras'] = extras_out
        self.transitions = transitions_dict

    def shuffle_transitions(self, transitions: Dict[str, Union[Tensor, Dict[str, List[Tensor]]]]) -> Dict[str, Any]:
        n = transitions['states'].shape[0]
        idx = torch.randperm(n)
        out: Dict[str, Any] = {}
        for k, v in transitions.items():
            if isinstance(v, torch.Tensor):
                out[k] = v[idx]
            else:
                out[k] = v
        return out

    def get_minibatches(self, minibatch_size: int) -> List[Dict[str, Tensor]]:
        assert 'states' in self.transitions, "Call get_transitions() before get_minibatches()."
        transitions = dict(self.transitions)
        transitions.pop('extras', None)
        transitions = self.shuffle_transitions(transitions)
        n = transitions['states'].shape[0]
        minibatches: List[Dict[str, Tensor]] = []
        for start in range(0, n, minibatch_size):
            end = start + minibatch_size
            mb = {k: v[start:end] for k, v in transitions.items()}
            minibatches.append(mb)
        return minibatches

    def combine_rollouts(self, rollouts: List["MultiAgentRolloutBuffer"]) -> "MultiAgentRolloutBuffer":
        combined = MultiAgentRolloutBuffer(n_agents=self.n_agents)
        for rb in rollouts:
            combined.episodes.extend(rb.episodes)
            combined.total_steps += rb.total_steps
            combined.n_episodes += rb.n_episodes
        return combined

    def _len(self) -> int:
        return int(self.total_steps)
