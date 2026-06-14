"""
UA-CQL: Uncertainty-Aware Conservative Q-Learning
Aligned with CORL's standard implementation pattern.

Core innovations:
1. Vectorized ensemble Q-network for efficient epistemic uncertainty estimation
2. Uncertainty-weighted adaptive CQL penalty: higher uncertainty → stronger pessimism
3. Dual adaptive mechanisms: alpha (conservative coefficient) and k (pessimism coefficient)

Reference: CORL CQL (https://github.com/tinkoff-ai/CORL)
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

TensorBatch = List[torch.Tensor]


# ══════════════════════════════════════════════════════════════════════
# Network Modules
# ══════════════════════════════════════════════════════════════════════

class VecEnsembleQ(nn.Module):
    """
    Vectorized ensemble Q-network: uses batched matrix multiplication (bmm)
    for efficient parallel computation across ensemble members.
    Outputs both mean and std across the ensemble as uncertainty estimate.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, ensemble_size: int = 3):
        super().__init__()
        self.E = ensemble_size
        in_d = obs_dim + act_dim
        # Vectorized weights: [E, in_dim, out_dim]
        self.w1 = nn.Parameter(torch.randn(ensemble_size, in_d, hidden_dim) * (2 / in_d) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(ensemble_size, 1, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(ensemble_size, hidden_dim, hidden_dim) * (2 / hidden_dim) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(ensemble_size, 1, hidden_dim))
        self.w3 = nn.Parameter(torch.randn(ensemble_size, hidden_dim, 1) * (2 / hidden_dim) ** 0.5)
        self.b3 = nn.Parameter(torch.zeros(ensemble_size, 1, 1))

    def _fwd(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_d] → [E, B, in_d] → [E, B]
        x = x.unsqueeze(0).expand(self.E, -1, -1)
        x = F.relu(torch.bmm(x, self.w1) + self.b1)
        x = F.relu(torch.bmm(x, self.w2) + self.b2)
        return (torch.bmm(x, self.w3) + self.b3).squeeze(-1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, std) across ensemble."""
        qs = self._fwd(torch.cat([obs, act], -1))
        return qs.mean(0), qs.std(0)

    def all_values(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Returns all ensemble Q-values [E, B]."""
        return self._fwd(torch.cat([obs, act], -1))


class UAGaussianPolicy(nn.Module):
    """Stochastic Gaussian policy with tanh squashing."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        return self.mu(h), self.log_std(h).clamp(-5, 2)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = (dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        return action, log_prob

    @torch.no_grad()
    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self(obs)[0])

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self.deterministic(state).cpu().numpy().flatten()


# ══════════════════════════════════════════════════════════════════════
# UA-CQL Trainer
# ══════════════════════════════════════════════════════════════════════

class UACQL(nn.Module):
    """
    UA-CQL: Uncertainty-Aware Conservative Q-Learning.

    Key differences from standard CQL:
    1. VecEnsembleQ replaces dual Q-networks — provides epistemic uncertainty via ensemble std
    2. Uncertainty-weighted CQL penalty: uw = (sigma_random + sigma_policy) normalized
       → OOD actions with high disagreement get stronger penalty
    3. Adaptive alpha (CQL weight): auto-tuned via gradient descent on CQL loss
    4. Adaptive k (pessimism coefficient): updated online based on Q-uncertainty
       → target = Q_mean - k*Q_std - alpha*log_prob (vs standard Q_mean - alpha*log_prob)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.device

        self.q = VecEnsembleQ(cfg.obs_dim, cfg.act_dim, cfg.hidden_dim, cfg.ensemble_size).to(d)
        self.qt = VecEnsembleQ(cfg.obs_dim, cfg.act_dim, cfg.hidden_dim, cfg.ensemble_size).to(d)
        self.qt.load_state_dict(self.q.state_dict())
        for p in self.qt.parameters():
            p.requires_grad = False

        self.pi = UAGaussianPolicy(cfg.obs_dim, cfg.act_dim, cfg.hidden_dim).to(d)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr)

        # SAC entropy temperature
        self.log_sac_alpha = nn.Parameter(torch.tensor(np.log(cfg.sac_alpha)), requires_grad=True)
        self.sac_opt = torch.optim.Adam([self.log_sac_alpha], lr=cfg.lr)
        self.target_ent = -cfg.act_dim

        # CQL adaptive alpha (conservative coefficient)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(cfg.alpha_init)), requires_grad=True)
        self.al_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        # Adaptive pessimism coefficient k (GPU tensor for async updates)
        self.register_buffer("_k_t", torch.tensor(cfg.k_pessimism))

        self.total_it = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def sac_alpha(self):
        return self.log_sac_alpha.exp()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        obs, act, rew, nobs, done = batch
        B, cfg = obs.shape[0], self.cfg
        K = cfg.num_random_actions
        self.total_it += 1

        # ── 1. Target Q computation ──
        rew = rew.squeeze(-1)
        done = done.squeeze(-1)
        with torch.no_grad():
            na, nlp = self.pi.sample(nobs)
            nm, ns = self.qt(nobs, na)
            tq = rew + cfg.gamma * (1 - done) * (nm - self._k_t * ns - self.sac_alpha * nlp)

        # ── 2. Bellman loss ──
        all_q = self.q.all_values(obs, act)
        bellman_loss = sum(F.mse_loss(all_q[i], tq) for i in range(cfg.ensemble_size)) / cfg.ensemble_size

        # ── 3. UA-CQL penalty (uncertainty-weighted) ──
        # Sample random actions + current policy actions
        ra = torch.FloatTensor(B * K, cfg.act_dim).uniform_(-1, 1).to(cfg.device)
        or_ = obs.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        with torch.no_grad():
            pa, _ = self.pi.sample(or_)

        # Compute Q mean & std for random and policy actions
        qm_cat, qs_cat = self.q(torch.cat([or_, or_], 0), torch.cat([ra, pa], 0))
        rm, pm = qm_cat[:B * K].reshape(B, K), qm_cat[B * K:].reshape(B, K)
        rs, ps = qs_cat[:B * K].reshape(B, K), qs_cat[B * K:].reshape(B, K)

        # Uncertainty weight: normalized total uncertainty (sigma_random + sigma_policy)
        uw = (rs + ps).mean(1).detach()
        uw = (uw / (uw.mean() + 1e-8)).clamp(0.5, 2.0)

        dm, ds = self.q(obs, act)
        cql_penalty = (uw * (torch.logsumexp(torch.cat([rm, pm], 1), 1) - dm)).mean()

        q_loss = bellman_loss + self.alpha.detach() * cql_penalty

        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.q_opt.step()

        # ── 4. Adaptive alpha update (CQL weight) ──
        self.al_opt.zero_grad()
        (-self.log_alpha * cql_penalty.detach()).backward()
        self.al_opt.step()

        # ── 5. Policy update ──
        pa2, lp2 = self.pi.sample(obs)
        pm2, ps2 = self.q(obs, pa2)
        pi_loss = (self.sac_alpha.detach() * lp2 - (pm2 - self._k_t * ps2)).mean()

        self.pi_opt.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        self.pi_opt.step()

        # ── 6. SAC entropy temperature update ──
        self.sac_opt.zero_grad()
        (-(self.log_sac_alpha * (lp2.detach() + self.target_ent))).mean().backward()
        self.sac_opt.step()

        # ── 7. Adaptive k update (online, GPU-side, no .item() sync) ──
        self._k_t = (self._k_t + cfg.k_adapt_rate * (ds.mean().detach() - 0.5)).clamp(0.1, 3.0)

        # ── 8. Soft target update ──
        for p, pt in zip(self.q.parameters(), self.qt.parameters()):
            pt.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)

        return {
            "q_loss": q_loss.item(),
            "bellman_loss": bellman_loss.item(),
            "cql_penalty": cql_penalty.item(),
            "pi_loss": pi_loss.item(),
            "alpha": self.alpha.item(),
            "sac_alpha": self.sac_alpha.item(),
            "k": self._k_t.item(),
            "q_std": ds.mean().item(),
            "uncertainty_weight": uw.mean().item(),
        }
