import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Union, Tuple, Optional

from itertools import chain
from src.networks.FeedForwardNN import FeedForwardNN


class PPOController(nn.Module):
    """
    Basic Controller for Proximal Policy Optimization (PPO) with
    feed-forward actor/critic and diskretem Action Space (Categorical).
    Shapes:
      - states: (batch, state_dim)
      - logits: (batch, action_size)
      - actions: (batch,)  [dtype: long]
      - log_probs: (batch,)
      - values: (batch, 1)
      - entropy: (batch,)
    """
    def __init__(self, config: Dict, agent_ID: Union[int, str] = None):
        super().__init__()
        self.config: Dict = config
        self.agent_ID: Optional[Union[int, str]] = agent_ID
        self._init_hyperparameters(config)

        self._build_actor()
        self._build_critic()
        self.update_step: int = 0

    # ----------------------------- setup -----------------------------

    def _init_hyperparameters(self, config: Dict) -> None:
        self.action_size: int = int(config['action_size'])
        self.state_size: int = int(config['state_size'])
        self.gamma: float = float(config['gamma'])
        self.lam: float = float(config['lam'])
        self.gae_horizon: int = int(config.get('gae_horizon', 0))
        self.clip_epsilon: float = float(config['clip_epsilon'])
        self.value_loss_coef: float = float(config['value_loss_coefficient'])
        self.entropy_coef: float = float(config['entropy_coefficient'])

    def _build_actor(self) -> None:
        self.actor_network = FeedForwardNN(self.state_size, self.action_size, self.config['actor_config'])

    def _build_critic(self) -> None:
        self.critic_network = FeedForwardNN(self.state_size, 1, self.config['critic_config'])

    def get_parameters(self):
        return chain(self.actor_network.parameters(), self.critic_network.parameters())

    # --------------------------- weights I/O --------------------------

    def update_weights(self, network_params: Tuple[Dict, Dict]) -> None:
        actor_params, critic_params = network_params
        self.actor_network.load_state_dict(actor_params)
        self.critic_network.load_state_dict(critic_params)

    def get_network_params(self) -> Tuple[Dict, Dict]:
        return self.actor_network.state_dict(), self.critic_network.state_dict()

    # --------------------------- forward APIs ------------------------

    @torch.no_grad()
    def state_values(self, states: Tensor, extras: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """
        Critic-Vorwärtspass.
        Returns: values (batch, 1)
        """
        return self.critic_network(states)

    def _make_logits(self, states: Tensor) -> Tensor:
        """
        Returns: logits (batch, action_size)
        """
        return self.actor_network(states)

    @torch.no_grad()
    def sample_action(self, states: Tensor) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict]]:
        """
        Stochastische Action-Selektion.
        Returns:
          actions   (batch,) long
          log_prob  (batch,)
          values    (batch,1)
          extras    dict|None
        """
        logits = self._make_logits(states)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()                    # (batch,)
        log_prob = dist.log_prob(actions)         # (batch,)
        values = self.critic_network(states)      # (batch,1)
        return actions.long(), log_prob, values, None

    @torch.no_grad()
    def select_action(self, states: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Greedy Action-Selektion (argmax). Gibt zusätzlich die zugehörigen Log-Probabilites zurück.
        Returns:
          actions  (batch,) long
          log_prob (batch,)
        """
        logits = self._make_logits(states)        # (batch, action_size)
        actions = torch.argmax(logits, dim=-1)    # (batch,)
        log_probs_all = torch.log_softmax(logits, dim=-1)  # (batch, action_size)
        log_prob = log_probs_all.gather(1, actions.view(-1, 1)).squeeze(1)  # (batch,)
        return actions.long(), log_prob

    def evaluate(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Bewertet gegebene Aktionen unter der aktuellen Policy.
        Returns:
          log_probs (batch,)
          entropy   (batch,)
          values    (batch,1)
        """
        logits = self._make_logits(states)                    # (batch, action_size)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions.long())             # (batch,)
        entropy = dist.entropy()                              # (batch,)
        values = self.critic_network(states)                  # (batch,1)
        return log_probs, entropy, values
