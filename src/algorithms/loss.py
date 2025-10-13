import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple


def _ensure_1d(x: Tensor, name: str) -> Tensor:
    if x.ndim > 1:
        # Erlaube (N,1) und squeeze es
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.squeeze(-1)
        else:
            raise ValueError(f"{name} must be 1D (or Nx1), got shape {tuple(x.shape)}")
    return x


def value_loss(
    state_values: Tensor,
    next_state_values: Tensor,
    reward: Tensor,
    done: Tensor,
    gamma: float,
    actual_len: int = 1,
) -> Tensor:
    """
    TD(1) target:  r + gamma^k * (1-done) * V(s')
    Erwartet 1D-Vektoren (N,), (N,) ...; (N,1) wird automatisch gesqueezed.
    """
    state_values       = _ensure_1d(state_values, "state_values")
    next_state_values  = _ensure_1d(next_state_values, "next_state_values")
    reward             = _ensure_1d(reward, "reward").to(state_values.dtype)
    done               = _ensure_1d(done, "done").to(state_values.dtype)

    if not isinstance(gamma, float):
        gamma = float(gamma)

    target = reward + (gamma ** actual_len) * (1.0 - done) * next_state_values.detach()
    return F.mse_loss(state_values, target)


def value_loss_with_IS(
    state_values: Tensor,
    next_state_values: Tensor,
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    reward: Tensor,
    done: Tensor,
    gamma: float,
    actual_len: int = 1,
) -> Tensor:
    """
    Value-Loss mit (trunkiertem) Importance Weight. Unüblich für PPO, aber ok experimentell.
    Erwartet alle Eingänge als 1D-Vektoren.
    """
    state_values       = _ensure_1d(state_values, "state_values")
    next_state_values  = _ensure_1d(next_state_values, "next_state_values")
    reward             = _ensure_1d(reward, "reward").to(state_values.dtype)
    done               = _ensure_1d(done, "done").to(state_values.dtype)
    new_log_prob       = _ensure_1d(new_log_prob, "new_log_prob")
    old_log_prob       = _ensure_1d(old_log_prob, "old_log_prob")

    if not isinstance(gamma, float):
        gamma = float(gamma)

    target = reward + (gamma ** actual_len) * (1.0 - done) * next_state_values.detach()

    with torch.no_grad():
        # nur Down-weighten (positives Ratio-Log wird auf 0 gecappt)
        truncated_ratio_log = torch.clamp(new_log_prob - old_log_prob, max=0.0)
        is_weight = torch.exp(truncated_ratio_log)

    per_sample_mse = F.mse_loss(state_values, target, reduction="none")
    return (per_sample_mse * is_weight).mean()


def policy_loss(
    gae: Tensor,
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    clip_eps: float,
) -> Tensor:
    """
    PPO-Clip Loss (zu minimieren).
    Erwartet 1D-Vektoren: gae, new_log_prob, old_log_prob.
    Wichtig: gae wird ohne Grad verwendet (detach), wie PPO erwartet.
    """
    gae          = _ensure_1d(gae, "gae").detach()
    new_log_prob = _ensure_1d(new_log_prob, "new_log_prob")
    old_log_prob = _ensure_1d(old_log_prob, "old_log_prob")

    # Ratio r_t
    ratio = torch.exp(new_log_prob - old_log_prob)

    # Clip(r_t, 1-eps, 1+eps)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    # Surrogate: min(r * A, clip(r) * A)
    surr1 = ratio  * gae
    surr2 = clipped * gae

    # Negatives Vorzeichen: wir minimieren den Verlust
    return -torch.min(surr1, surr2).mean()


def vtrace(
    behaviour_log_probs: Tensor,
    target_log_probs: Tensor,
    rewards: Tensor,
    state_values: Tensor,
    next_state_values: Tensor,
    dones: Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """
    V-trace für Off-Policy-Korrekturen (zeitlich geordnet: t=0..T-1).
    Erwartet Sequenzen gleicher Länge; state_values/next_state_values dürfen (T,1) sein.
    """
    # Bringe alles auf 1D, erlaube (T,1)
    behaviour_log_probs = _ensure_1d(behaviour_log_probs, "behaviour_log_probs")
    target_log_probs    = _ensure_1d(target_log_probs, "target_log_probs")
    rewards             = _ensure_1d(rewards, "rewards")
    dones               = _ensure_1d(dones, "dones").to(rewards.dtype)

    # Values dürfen (T,1) bleiben (Form für nachfolgende Ops)
    if state_values.ndim == 1:
        state_values = state_values.unsqueeze(-1)
    if next_state_values.ndim == 1:
        next_state_values = next_state_values.unsqueeze(-1)

    T = state_values.size(0)
    assert next_state_values.size(0) == T, "state/next_state length mismatch"

    with torch.no_grad():
        discounts = float(gamma) * (1.0 - dones)                    # (T,)
        rhos = torch.exp(target_log_probs - behaviour_log_probs)    # (T,)
        clipped_rhos = torch.clamp(rhos, max=rho_bar)               # (T,)
        clipped_cs   = torch.clamp(rhos, max=c_bar)                  # (T,)

        deltas = clipped_rhos.unsqueeze(-1) * (
            rewards.unsqueeze(-1) + discounts.unsqueeze(-1) * next_state_values - state_values
        )  # (T,1)

        vs = torch.zeros_like(state_values)
        next_vs = next_state_values[-1]  # (1,)

        for t in reversed(range(T)):
            vs_t = state_values[t] + deltas[t] + discounts[t] * clipped_cs[t] * (next_vs - next_state_values[t])
            vs[t] = vs_t
            next_vs = vs_t

        # TD targets für Policy-Grad-Advantage
        td_target = rewards.unsqueeze(-1) + discounts.unsqueeze(-1) * torch.cat(
            [vs[1:], next_state_values[-1:].clone()], dim=0
        )
        pg_advantages = clipped_rhos.unsqueeze(-1) * (td_target - state_values)

    return vs, pg_advantages.squeeze(-1)
