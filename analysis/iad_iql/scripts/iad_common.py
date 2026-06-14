"""Shared utilities for IAD-IQL frozen actor extraction."""
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/code/CORL")
from algorithms.offline.iql import (  # noqa: E402
    EXP_ADV_MAX,
    GaussianPolicy,
    ReplayBuffer,
    TwinQ,
    ValueFunction,
    compute_mean_std,
    eval_actor,
    normalize_states,
    set_seed,
    wrap_env,
)
import d4rl  # noqa: E402

VARIANTS = [
    "standard_iql",
    "ensemble_mean",
    "iad_lambda_0.5",
    "iad_lambda_1.0",
    "shuffled_iad_lambda_0.5",
    "shuffled_iad_lambda_1.0",
]


@dataclass
class ValueBundle:
    qf: TwinQ
    vf: ValueFunction
    seed: int
    checkpoint_path: str
    total_it: int


def load_norm_stats(path: str = "/code/analysis/iad_iql/checkpoints/norm_stats.npz"):
    data = np.load(path)
    return data["state_mean"], data["state_std"]


def build_env_and_buffer(
    env_name: str,
    device: str,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    buffer_size: int = 2_000_000,
):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size, device)
    replay_buffer.load_d4rl_dataset(dataset)
    eval_env = wrap_env(gym.make(env_name), state_mean=state_mean, state_std=state_std)
    return env, eval_env, replay_buffer, state_dim, action_dim


def load_value_bundle(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    device: str,
    seed: int,
) -> ValueBundle:
    sd = torch.load(checkpoint_path, map_location=device)
    qf = TwinQ(state_dim, action_dim).to(device)
    vf = ValueFunction(state_dim).to(device)
    qf.load_state_dict(sd["qf"])
    vf.load_state_dict(sd["vf"])
    qf.eval()
    vf.eval()
    qf.requires_grad_(False)
    vf.requires_grad_(False)
    return ValueBundle(
        qf=qf,
        vf=vf,
        seed=seed,
        checkpoint_path=checkpoint_path,
        total_it=int(sd["total_it"]),
    )


def flatten_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in module.parameters()])


@torch.no_grad()
def compute_online_advantages(
    bundle: ValueBundle, states: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    q_min = bundle.qf(states, actions)
    v = bundle.vf(states)
    return q_min - v


def log_domain_weights(
    advantage_term: torch.Tensor,
    disagreement_term: Optional[torch.Tensor] = None,
    lambda_: float = 0.0,
    beta: float = 3.0,
    exp_adv_max: float = EXP_ADV_MAX,
) -> torch.Tensor:
    max_log_weight = math.log(exp_adv_max)
    log_w = beta * advantage_term
    if disagreement_term is not None and lambda_ != 0.0:
        log_w = log_w - lambda_ * disagreement_term
    return torch.exp(torch.clamp(log_w, max=max_log_weight))


def compute_weights(
    a1: torch.Tensor,
    a2: torch.Tensor,
    variant: str,
    beta: float = 3.0,
    exp_adv_max: float = EXP_ADV_MAX,
) -> Dict[str, torch.Tensor]:
    mu_a = (a1 + a2) / 2.0
    u_a = (a1 - a2).abs() / 2.0
    batch_mean_ua = u_a.mean()
    hat_u_a = torch.clamp(u_a / (batch_mean_ua + 1e-6), 0.0, 5.0)

    stats: Dict[str, torch.Tensor] = {
        "a1": a1,
        "a2": a2,
        "mu_a": mu_a,
        "u_a": u_a,
        "hat_u_a": hat_u_a,
    }

    if variant == "standard_iql":
        w = log_domain_weights(a1, beta=beta, exp_adv_max=exp_adv_max)
        stats["w_base"] = w
        stats["w"] = w
        return stats

    if variant == "ensemble_mean":
        w = log_domain_weights(mu_a, beta=beta, exp_adv_max=exp_adv_max)
        stats["w_base"] = w
        stats["w"] = w
        return stats

    if variant in (
        "iad_lambda_0.5",
        "iad_lambda_1.0",
        "shuffled_iad_lambda_0.5",
        "shuffled_iad_lambda_1.0",
    ):
        lam = 0.5 if "0.5" in variant else 1.0
        hat_for_penalty = hat_u_a
        if variant.startswith("shuffled_iad"):
            perm = torch.randperm(hat_u_a.shape[0], device=hat_u_a.device)
            hat_for_penalty = hat_u_a[perm].detach()

        w_base = log_domain_weights(mu_a, beta=beta, exp_adv_max=exp_adv_max)
        w_tilde = log_domain_weights(
            mu_a, disagreement_term=hat_for_penalty, lambda_=lam, beta=beta, exp_adv_max=exp_adv_max
        )
        scale = (w_base.mean() / (w_tilde.mean() + 1e-8)).detach()
        w_matched = w_tilde * scale
        w = torch.clamp(w_matched, max=exp_adv_max)
        stats["w_base"] = w_base
        stats["w_tilde"] = w_tilde
        stats["w_matched"] = w_matched
        stats["w"] = w
        stats["mean_w_base"] = w_base.mean()
        stats["mean_w_tilde"] = w_tilde.mean()
        stats["mean_w_final"] = w.mean()
        stats["clamp_ratio"] = (w >= exp_adv_max - 1e-6).float().mean()
        return stats

    raise ValueError(f"Unknown variant: {variant}")


def compute_ensemble3_weights(
    a0: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    beta: float = 3.0,
    exp_adv_max: float = EXP_ADV_MAX,
) -> Dict[str, torch.Tensor]:
    """3-value-system ensemble: weight = exp(beta * mean(A0,A1,A2)), clipped."""
    mu_3q = (a0 + a1 + a2) / 3.0
    w = log_domain_weights(mu_3q, beta=beta, exp_adv_max=exp_adv_max)
    a_stack = torch.stack([a0, a1, a2], dim=-1)
    std_3q = a_stack.std(dim=-1)
    return {
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "mu_3q": mu_3q,
        "std_3q": std_3q,
        "w": w,
        "w_base": w,
        "clamp_ratio": (w >= exp_adv_max - 1e-6).float().mean(),
    }


def actor_bc_loss(actor: GaussianPolicy, states: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor):
    dist = actor(states)
    log_prob = dist.log_prob(actions).sum(-1)
    return torch.mean(-weights.detach() * log_prob)


def ess(weights: torch.Tensor) -> float:
    w = weights.detach()
    s = w.sum()
    return float((s * s) / (w.pow(2).sum() + 1e-8))


def get_batch_from_indices(replay_buffer: ReplayBuffer, indices: np.ndarray) -> list:
    states = replay_buffer._states[indices]
    actions = replay_buffer._actions[indices]
    rewards = replay_buffer._rewards[indices]
    next_states = replay_buffer._next_states[indices]
    dones = replay_buffer._dones[indices]
    return [states, actions, rewards, next_states, dones]


def resolve_checkpoint(
    seed: int,
    step: int = 999999,
    checkpoints_dir: str = "/code/analysis/iad_iql/checkpoints",
) -> str:
    p = Path(checkpoints_dir) / f"value_seed{seed}" / f"checkpoint_{step}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing checkpoint: {p}")
    return str(p)
