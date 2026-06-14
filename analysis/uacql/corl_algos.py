#!/usr/bin/env python3
"""
CORL-compatible algorithm implementations for offline RL.
Adapted from https://github.com/tinkoff-ai/CORL
All algorithm classes here are self-contained — no d4rl dependency.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

TensorBatch = List[torch.Tensor]

# ══════════════════════════════════════════════════════════════════════
# Common utilities
# ══════════════════════════════════════════════════════════════════════

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


# ══════════════════════════════════════════════════════════════════════
# CQL components (adapted from CORL)
# ══════════════════════════════════════════════════════════════════════

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))
    def forward(self) -> nn.Parameter:
        return self.constant


def init_module_weights(module: nn.Sequential, orthogonal_init: bool = False):
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)
    nn.init.constant_(module[-1].bias, 0.0)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(self, mean, log_std, sample):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(Normal(mean, std), TanhTransform(cache_size=1))
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(self, mean, log_std, deterministic=False):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(Normal(mean, std), TanhTransform(cache_size=1))
        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()
        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)
        return action_sample, log_prob


class CQLTanhGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, log_std_multiplier=1.0,
                 log_std_offset=-1.0, orthogonal_init=False, no_tanh=False):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )
        init_module_weights(self.base_network)
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(self, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state, device="cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class CQLFullyConnectedQFunction(nn.Module):
    def __init__(self, observation_dim, action_dim, orthogonal_init=False, n_hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(observation_dim + action_dim, 256), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))
        self.network = nn.Sequential(*layers)
        init_module_weights(self.network, orthogonal_init)

    def forward(self, observations, actions):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class ContinuousCQL(nn.Module):
    """Continuous CQL trainer — adapted from CORL."""
    def __init__(self, critic_1, critic_1_optimizer, critic_2, critic_2_optimizer,
                 actor, actor_optimizer, target_entropy, discount=0.99, alpha_multiplier=1.0,
                 use_automatic_entropy_tuning=True, backup_entropy=False,
                 policy_lr=3e-4, qf_lr=3e-4, soft_target_update_rate=5e-3,
                 bc_steps=100000, target_update_period=1, cql_n_actions=10,
                 cql_importance_sample=True, cql_lagrange=False,
                 cql_target_action_gap=-1.0, cql_temp=1.0, cql_alpha=5.0,
                 cql_max_target_backup=False, cql_clip_diff_min=-np.inf,
                 cql_clip_diff_max=np.inf, device="cpu"):
        super().__init__()
        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device
        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(device)
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=policy_lr)
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(self.log_alpha_prime.parameters(), lr=qf_lr)

    def update_target_network(self, soft_target_update_rate):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def _alpha_and_alpha_loss(self, observations, log_pi):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(self, observations, actions, new_actions, alpha, log_pi):
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(self, observations, actions, next_observations, rewards, dones, alpha, log_dict):
        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(next_observations, repeat=self.cql_n_actions)
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ), dim=-1)
            next_log_pi = torch.gather(next_log_pi, -1, max_target_indices.unsqueeze(-1)).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL penalty
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(observations, repeat=self.cql_n_actions)
        cql_next_actions, cql_next_log_pis = self.actor(next_observations, repeat=self.cql_n_actions)
        cql_current_actions = cql_current_actions.detach()
        cql_current_log_pis = cql_current_log_pis.detach()
        cql_next_actions = cql_next_actions.detach()
        cql_next_log_pis = cql_next_log_pis.detach()

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        cql_cat_q1 = torch.cat([cql_q1_rand, torch.unsqueeze(q1_predicted, 1),
                                cql_q1_next_actions, cql_q1_current_actions], dim=1)
        cql_cat_q2 = torch.cat([cql_q2_rand, torch.unsqueeze(q2_predicted, 1),
                                cql_q2_next_actions, cql_q2_current_actions], dim=1)
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5 ** action_dim)
            cql_cat_q1 = torch.cat([
                cql_q1_rand - random_density,
                cql_q1_next_actions - cql_next_log_pis.detach(),
                cql_q1_current_actions - cql_current_log_pis.detach(),
            ], dim=1)
            cql_cat_q2 = torch.cat([
                cql_q2_rand - random_density,
                cql_q2_next_actions - cql_next_log_pis.detach(),
                cql_q2_current_actions - cql_current_log_pis.detach(),
            ], dim=1)

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        cql_qf1_diff = torch.clamp(cql_qf1_ood - q1_predicted,
                                   self.cql_clip_diff_min, self.cql_clip_diff_max).mean()
        cql_qf2_diff = torch.clamp(cql_qf2_ood - q2_predicted,
                                   self.cql_clip_diff_min, self.cql_clip_diff_max).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
            cql_min_qf1_loss = alpha_prime * self.cql_alpha * (cql_qf1_diff - self.cql_target_action_gap)
            cql_min_qf2_loss = alpha_prime * self.cql_alpha * (cql_qf2_diff - self.cql_target_action_gap)
            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = observations.new_tensor(0.0)
            alpha_prime = observations.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(dict(
            qf1_loss=qf1_loss.item(), qf2_loss=qf2_loss.item(),
            cql_qf1_diff=cql_qf1_diff.mean().item(), cql_qf2_diff=cql_qf2_diff.mean().item(),
            average_qf1=q1_predicted.mean().item(), average_qf2=q2_predicted.mean().item(),
        ))
        return qf_loss, alpha_prime, alpha_prime_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        observations, actions, rewards, next_observations, dones = batch
        self.total_it += 1
        new_actions, log_pi = self.actor(observations)
        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)
        policy_loss = self._policy_loss(observations, actions, new_actions, alpha, log_pi)
        log_dict = dict(log_pi=log_pi.mean().item(), policy_loss=policy_loss.item(),
                        alpha_loss=alpha_loss.item(), alpha=alpha.item())
        qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
            observations, actions, next_observations, rewards, dones, alpha, log_dict)

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)
        return log_dict


# ══════════════════════════════════════════════════════════════════════
# IQL components (adapted from CORL)
# ══════════════════════════════════════════════════════════════════════

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EXP_ADV_MAX = 100.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(self, dims, activation_fn=nn.ReLU, output_activation_fn=None,
                 squeeze_output=False, dropout=None):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims")
        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class IQLGaussianPolicy(nn.Module):
    def __init__(self, state_dim, act_dim, max_action, hidden_dim=256, n_hidden=2, dropout=None):
        super().__init__()
        self.net = MLP([state_dim, *([hidden_dim] * n_hidden), act_dim], output_activation_fn=nn.Tanh)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state, device="cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class IQLDeterministicPolicy(nn.Module):
    def __init__(self, state_dim, act_dim, max_action, hidden_dim=256, n_hidden=2, dropout=None):
        super().__init__()
        self.net = MLP([state_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation_fn=nn.Tanh, dropout=dropout)
        self.max_action = max_action

    def forward(self, obs):
        return self.net(obs)

    @torch.no_grad()
    def act(self, state, device="cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
                .cpu().data.numpy().flatten())


class IQLTwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))


class IQLValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)


class ImplicitQLearning(nn.Module):
    """Implicit Q-Learning trainer — adapted from CORL."""
    def __init__(self, max_action, actor, actor_optimizer, q_network, q_optimizer,
                 v_network, v_optimizer, iql_tau=0.7, beta=3.0, max_steps=1000000,
                 discount=0.99, tau=0.005, device="cpu"):
        super().__init__()
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau
        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(self, next_v, observations, actions, rewards, terminals, log_dict):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(self, adv, observations, actions, log_dict):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape mismatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        observations, actions, rewards, next_observations, dones = batch
        log_dict = {}
        with torch.no_grad():
            next_v = self.vf(next_observations)
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        self._update_policy(adv, observations, actions, log_dict)
        return log_dict


# ══════════════════════════════════════════════════════════════════════
# TD3+BC components (adapted from CORL)
# ══════════════════════════════════════════════════════════════════════

class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state, device="cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))


class TD3BC(nn.Module):
    """TD3+BC trainer — adapted from CORL."""
    def __init__(self, max_action, actor, actor_optimizer, critic_1, critic_1_optimizer,
                 critic_2, critic_2_optimizer, discount=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, alpha=2.5, device="cpu"):
        super().__init__()
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1
        state, action, reward, next_state, done = batch
        not_done = 1 - done

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            lmbda = self.alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
            log_dict["actor_loss"] = actor_loss.item()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict
