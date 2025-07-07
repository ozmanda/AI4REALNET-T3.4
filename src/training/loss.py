import torch
from torch import Tensor
import torch.nn.functional as F

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
    expected_state_values = (next_state_values.detach() * torch.pow(gamma, actual_len) * (1 - done)) + reward
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
