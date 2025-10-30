import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple

def value_loss(predicted_values: Tensor, target_values: Tensor) -> Tensor:
    """
    Calculate the value loss for the critic network using MSE between predicted and target values.
    When using GAE, target_values should be (GAE + baseline).
    """
    return F.mse_loss(predicted_values, target_values)


def value_loss_with_IS(predicted_values: Tensor, bootstrap_values: Tensor, new_state_values: Tensor, new_log_prob: Tensor, old_log_prob: Tensor, reward: Tensor, done: Tensor, gamma: float, actual_len: int = 1):
    '''
    Value Loss with Importance Sampling
     -> adds importance sampling weights to account for policy changes, stabilising training by reducing the influence of value predictions based on large, potentially unreliable updates to the policy (=stabilisation)
    actual_len indicates the number of steps between state and next_state, default is 1.
    '''
    target = reward + (gamma ** actual_len) * (1 - done) * new_state_values.detach()
    with torch.no_grad():
        truncated_ratio_log = torch.clamp(new_log_prob - old_log_prob, max=0)
        importance_sample_fix = torch.exp(truncated_ratio_log)

    value_loss = (F.mse_loss(predicted_values, target, reduction="none") * importance_sample_fix).mean()
    return value_loss


def policy_loss(gae: Tensor, new_log_prob: Tensor, old_log_prob: Tensor, clip_eps: float):
    unclipped_ratio = torch.exp(new_log_prob - old_log_prob)
    clipped_ratio = torch.clamp(unclipped_ratio, 1 - clip_eps, 1 + clip_eps)
    actor_loss = -torch.min(clipped_ratio * gae, unclipped_ratio * gae).mean()
    return actor_loss


def vtrace(behaviour_log_probs: Tensor, target_log_probs: Tensor, rewards: Tensor, state_values: Tensor, next_state_values: Tensor,
           dones: Tensor, gamma: float, rho_bar: float = 1.0, c_bar: float = 1.0) -> Tuple[Tensor, Tensor]: 
    """
    V-trace algorithm for off-policy actor-critic methods like IMPALA. Assumes that the final states already contain the bootstrap values.

    Parameters: 
    - behaviour_log_probs:      Tensor -> Log probabilities of the behaviour policy at the chosen actions
    - target_log_probs:         Tensor -> Log probabilities of the target policy at the chosen actions
    - actions:                  Tensor -> Actions taken by the agent
    - rewards:                  Tensor -> Rewards received by the agent
    - values:                   Tensor -> State values predicted by the critic
    - gamma:                    float  -> Discount factor
    - rho_bar:                  float  -> Importance sampling ratio (default: 1.0)
    - c_bar:                    float  -> Clipping parameter (default: 1.0)

    Returns:
    - v_s:                      Tensor -> corrected value targets
    - pg_adv:                   Tensor -> policy gradient advantages
    """
    with torch.no_grad():
        discounts = gamma * (1 - dones.long()) #(batchsize)
        rhos = torch.exp(target_log_probs - behaviour_log_probs) #(batchsize)
        clipped_rhos = torch.clamp(rhos, max=rho_bar) # same as min(rho_bar, rhos) -> (batchsize)
        clipped_cs = torch.clamp(rhos, max=c_bar) # same as min(c_bar, rhos) -> (batchsize)

        deltas = clipped_rhos * (rewards + discounts * next_state_values.squeeze(-1) - state_values.squeeze(-1))  #(batchsize)

        # calculate v-trace targets
        vs = torch.zeros_like(state_values)
        next_vs = next_state_values[-1]
        for t in reversed(range(state_values.size(0))):
            vs_t = state_values[t] + deltas[t] + discounts[t] * clipped_cs[t] * (next_vs - next_state_values[t])
            vs[t] = vs_t
            next_vs = vs_t
        
        # calculate policy gradient advantages
        td_target = rewards + discounts * torch.cat([vs[1:], next_state_values[-1].unsqueeze(0)], dim=0)
        pg_advantages = clipped_rhos * (td_target - state_values)
    
    return vs, pg_advantages
