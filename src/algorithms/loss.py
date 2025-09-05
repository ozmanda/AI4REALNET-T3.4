import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple

def value_loss(state_values: Tensor, next_state_values: Tensor, reward: Tensor, done: Tensor, gamma: float, actual_len: int = 1) -> Tensor:
    """
    Calculate the value loss for the critic network. The actual_len parameter is used to indicate the number of steps between state and next_state. Default is 1. 
    """
    expected_state_values = (next_state_values.detach() * gamma ** actual_len * (1 - done)) + reward
    return F.mse_loss(state_values, expected_state_values)


def value_loss_with_IS(state_values: Tensor, next_state_values: Tensor, new_log_prob: Tensor, old_log_prob: Tensor, reward: Tensor, done: Tensor, gamma: float, actual_len: int = 1):
    ''' 
    Value Loss with Importance Sampling
     -> adds importance sampling weights to account for policy changes, stabilising training by reducing the influence of value predictions based on large, potentially unreliable updates to the policy (=stabilisation)
    actual_len indicates the number of steps between state and next_state, default is 1.
    '''
    expected_state_values = (next_state_values.detach() * (gamma ** actual_len) * (1 - done)) + reward
    with torch.no_grad():
        truncated_ratio_log = torch.clamp(new_log_prob - old_log_prob, max=0)
        importance_sample_fix = torch.exp(truncated_ratio_log)
    
    value_loss = (F.mse_loss(expected_state_values, state_values, reduction="none") * importance_sample_fix).mean()
    return value_loss


def policy_loss(gae: Tensor, new_log_prob: Tensor, old_log_prob: Tensor, clip_eps: float):
    unclipped_ratio = torch.exp(new_log_prob - old_log_prob)
    clipped_ratio = torch.clamp(unclipped_ratio, 1 - clip_eps, 1 + clip_eps)
    actor_loss = -torch.min(clipped_ratio * gae, unclipped_ratio * gae).mean()
    return actor_loss


def vtrace(behaviour_log_probs: Tensor, target_log_probs: Tensor, actions: Tensor, rewards: Tensor, values: Tensor, 
           dones: Tensor, gamma: float, rho_bar: float = 1.0, c_bar: float = 1.0) -> Tuple[Tensor, Tensor]: 
    """
    V-trace algorithm for off-policy actor-critic methods like IMPALA. Assumes that the final states already contain the bootstrap values.

    Parameters: 
    - behaviour_log_probs: Log probabilities of the behaviour policy
    - target_log_probs: Log probabilities of the target policy
    - actions: Actions taken by the agent
    - rewards: Rewards received by the agent
    - values: State values predicted by the critic
    - gamma: Discount factor
    - rho_bar: Importance sampling ratio (default: 1.0)
    - c_bar: Clipping parameter (default: 1.0)

    Returns:
    - v_s: corrected value targets
    - pg_adv: policy gradient advantages
    """
    v_s = None
    advantages = None
    trajectory_length, batchsize = actions.shape()

    # gather log probs for actions taken
    target_log_probs = target_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    behaviour_log_probs = behaviour_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    # IS ratios
    rhos = torch.exp(target_log_probs - behaviour_log_probs)
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)

    
    return v_s, advantages