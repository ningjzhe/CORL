"""
UGH-CQL: Uncertainty-Gated Hybrid Conservative Q-Learning
Aligned with CORL's standard implementation pattern.

Core innovations (building on UA-CQL):
1. Uncertainty-weighted adaptive CQL penalty (from UA-CQL)
2. Dual adaptive alpha/k mechanisms (from UA-CQL)
3. Value network with IQL expectile regression for auxiliary value estimation
4. Uncertainty-gated hybrid policy update:
   - Low uncertainty regions: IQL-style AWR (advantage-weighted regression)
   - High uncertainty regions: CQL-style conservative policy gradient
   → Smooth transition via exp(-std/temperature) gating

This hybrid design combines:
- CQL's conservatism for OOD actions (high uncertainty → CQL)
- IQL's stability for in-distribution actions (low uncertainty → AWR)

Reference: CORL CQL + IQL (https://github.com/tinkoff-ai/CORL)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from ua_cql import VecEnsembleQ, UAGaussianPolicy, UACQL

TensorBatch = List[torch.Tensor]


# ══════════════════════════════════════════════════════════════════════
# Value Network (IQL-style)
# ══════════════════════════════════════════════════════════════════════

class ValueNetwork(nn.Module):
    """State-value network for IQL expectile regression."""
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════
# UGH-CQL Trainer
# ══════════════════════════════════════════════════════════════════════

class UGHCQL(nn.Module):
    """
    UGH-CQL: Uncertainty-Gated Hybrid Conservative Q-Learning.

    Extends UA-CQL with:
    1. ValueNetwork trained via IQL expectile regression on Q-values
       → provides stable value baseline for advantage computation
    2. Uncertainty-gated hybrid policy loss:
       - gate = exp(-Q_std / temperature) ∈ (0, 1]
       - Low Q_std → gate ≈ 1 → use IQL AWR loss (behavior cloning weighted by advantage)
       - High Q_std → gate ≈ 0 → use CQL policy loss (conservative, penalized by uncertainty)
       - Final loss = gate * IQL_loss + (1 - gate) * CQL_loss

    Design rationale:
    - In-distribution data: epistemic uncertainty is low, IQL's AWR provides
      stable policy improvement without excessive conservatism
    - OOD regions: uncertainty is high, CQL's conservative update prevents
      overestimation errors from propagating to the policy
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.device

        # Q-networks (same as UA-CQL)
        self.q = VecEnsembleQ(cfg.obs_dim, cfg.act_dim, cfg.hidden_dim, cfg.ensemble_size).to(d)
        self.qt = VecEnsembleQ(cfg.obs_dim, cfg.act_dim, cfg.hidden_dim, cfg.ensemble_size).to(d)
        self.qt.load_state_dict(self.q.state_dict())
        for p in self.qt.parameters():
            p.requires_grad = False

        # Policy, Value networks
        self.pi = UAGaussianPolicy(cfg.obs_dim, cfg.act_dim, cfg.hidden_dim).to(d)
        self.v = ValueNetwork(cfg.obs_dim, cfg.hidden_dim).to(d)

        # Optimizers
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=cfg.lr)

        # SAC entropy temperature
        self.log_sac_alpha = nn.Parameter(torch.tensor(np.log(cfg.sac_alpha)), requires_grad=True)
        self.sac_opt = torch.optim.Adam([self.log_sac_alpha], lr=cfg.lr)
        self.target_ent = -cfg.act_dim

        # CQL adaptive alpha
        self.log_alpha = nn.Parameter(torch.tensor(np.log(cfg.alpha_init)), requires_grad=True)
        self.al_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        # Adaptive pessimism coefficient k
        self.register_buffer("_k_t", torch.tensor(cfg.k_pessimism))

        self.total_it = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def sac_alpha(self):
        return self.log_sac_alpha.exp()

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Asymmetric L2 loss for expectile regression (IQL-style)."""
        tau = self.cfg.expectile
        weight = torch.where(diff > 0, tau, 1 - tau)
        return (weight * diff.pow(2)).mean()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        obs, act, rew, nobs, done = batch
        B, cfg = obs.shape[0], self.cfg
        K = cfg.num_random_actions
        self.total_it += 1

        # ── 1. Target Q computation (UA-CQL style with adaptive k) ──
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
        ra = torch.FloatTensor(B * K, cfg.act_dim).uniform_(-1, 1).to(cfg.device)
        or_ = obs.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        with torch.no_grad():
            pa, _ = self.pi.sample(or_)

        qm_cat, qs_cat = self.q(torch.cat([or_, or_], 0), torch.cat([ra, pa], 0))
        rm, pm = qm_cat[:B * K].reshape(B, K), qm_cat[B * K:].reshape(B, K)
        rs, ps = qs_cat[:B * K].reshape(B, K), qs_cat[B * K:].reshape(B, K)

        uw = (rs + ps).mean(1).detach()
        uw = (uw / (uw.mean() + 1e-8)).clamp(0.5, 2.0)

        dm, ds = self.q(obs, act)
        cql_penalty = (uw * (torch.logsumexp(torch.cat([rm, pm], 1), 1) - dm)).mean()

        q_loss = bellman_loss + self.alpha.detach() * cql_penalty

        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.q_opt.step()

        # ── 4. Adaptive alpha update ──
        self.al_opt.zero_grad()
        (-self.log_alpha * cql_penalty.detach()).backward()
        self.al_opt.step()

        # ── 5. Value network update (IQL expectile regression on Q) ──
        with torch.no_grad():
            q_val = dm.detach()  # Q-mean as target for value
        v_pred = self.v(obs)
        v_loss = self._expectile_loss(q_val - v_pred)

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        # ── 6. Uncertainty-gated hybrid policy update ──
        # Gate: low uncertainty → IQL AWR, high uncertainty → CQL
        with torch.no_grad():
            gate = torch.exp(-ds.detach() / cfg.unc_temperature)  # [B], ∈ (0, 1]

        # CQL policy loss component
        pa2, lp2 = self.pi.sample(obs)
        pm2, ps2 = self.q(obs, pa2)
        cql_pi = self.sac_alpha.detach() * lp2 - (pm2 - self._k_t * ps2)

        # IQL AWR policy loss component
        with torch.no_grad():
            advantage = (q_val - v_pred.detach()).clamp(-100, 100)
            awr_weight = (cfg.iql_beta * advantage).exp().clamp(max=100.0)
        mu, log_std = self.pi(obs)
        x = torch.atanh(act.clamp(-0.999, 0.999))
        log_prob = torch.distributions.Normal(mu, log_std.exp()).log_prob(x).sum(-1)
        iql_pi = -(awr_weight * log_prob)

        # Hybrid loss: gate-controlled combination
        pi_loss = (gate * iql_pi + (1 - gate) * cql_pi).mean()

        self.pi_opt.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        self.pi_opt.step()

        # ── 7. SAC entropy temperature update ──
        self.sac_opt.zero_grad()
        (-(self.log_sac_alpha * (lp2.detach() + self.target_ent))).mean().backward()
        self.sac_opt.step()

        # ── 8. Adaptive k update ──
        self._k_t = (self._k_t + cfg.k_adapt_rate * (ds.mean().detach() - 0.5)).clamp(0.1, 3.0)

        # ── 9. Soft target update ──
        for p, pt in zip(self.q.parameters(), self.qt.parameters()):
            pt.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)

        return {
            "q_loss": q_loss.item(),
            "bellman_loss": bellman_loss.item(),
            "cql_penalty": cql_penalty.item(),
            "pi_loss": pi_loss.item(),
            "v_loss": v_loss.item(),
            "alpha": self.alpha.item(),
            "sac_alpha": self.sac_alpha.item(),
            "k": self._k_t.item(),
            "q_std": ds.mean().item(),
            "gate_mean": gate.mean().item(),
            "advantage_mean": advantage.mean().item(),
        }
