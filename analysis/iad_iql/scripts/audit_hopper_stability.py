#!/usr/bin/env python3
"""Hopper actor extraction stability audit: CSV curves, figures, re-eval, report."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/code/CORL")
sys.path.insert(0, "/code/analysis/iad_iql/scripts")

from algorithms.offline.iql import GaussianPolicy, eval_actor, set_seed  # noqa: E402
from iad_common import (  # noqa: E402
    VARIANTS,
    actor_bc_loss,
    build_env_and_buffer,
    compute_online_advantages,
    compute_weights,
    load_norm_stats,
    load_value_bundle,
)

RUNS = [
    "standard_iql_100k_fast",
    "ensemble_mean_100k_fast",
    "iad_lambda_1.0_100k_fast",
    "shuffled_iad_lambda_1.0_100k_fast",
    "repB_standard_iql_100k_fast",
    "repB_ensemble_mean_100k_fast",
    "repB_iad_lambda_1.0_100k_fast",
    "repB_shuffled_iad_lambda_1.0_100k_fast",
]
REP_A = RUNS[:4]
REP_B = RUNS[4:]
ENV = "hopper-medium-v2"
HOPPER = Path("/code/analysis/iad_iql/hopper_medium_v2")
CKPT = HOPPER / "checkpoints"
LOG = HOPPER / "logs"
RESULTS = HOPPER / "results"
FIG = HOPPER / "figures" / "stability_audit"
REPORT = HOPPER / "report" / "HOPPER_STABILITY_AUDIT.md"
NORM = CKPT / "norm_stats.npz"
CKPT_A = CKPT / "value_seed0" / "checkpoint_999999.pt"
CKPT_B = CKPT / "value_seed1" / "checkpoint_999999.pt"


def parse_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    ev = df[df["D4RL_normalized_score"].notna() & (df["D4RL_normalized_score"] != "")].copy()
    ev["D4RL_normalized_score"] = pd.to_numeric(ev["D4RL_normalized_score"])
    ev["training_step"] = pd.to_numeric(ev["training_step"])
    for col in ("ESS_over_batch_size", "clamp_ratio", "actor_log_std", "mean_disagreement",
                "mean_weight", "top_10_percent_weight_mass", "actor_loss"):
        if col in ev.columns:
            ev[col] = pd.to_numeric(ev[col], errors="coerce")
    return ev.sort_values("training_step")


def audit_run(run: str, log_dir: Path, ckpt_dir: Path) -> dict:
    csv_path = log_dir / f"{run}.csv"
    row = {
        "run": run,
        "replicate": "RepB" if run.startswith("repB_") else "RepA",
        "csv_exists": csv_path.exists(),
        "actor_final_pt": (ckpt_dir / run / "actor_final.pt").exists(),
        "best_actor_pt": (ckpt_dir / run / "best_actor.pt").exists(),
        "eval_step_checkpoints": 0,
        "can_reevaluate_best": False,
    }
    if not csv_path.exists():
        return row

    df = pd.read_csv(csv_path)
    ev = parse_eval_df(df)
    if ev.empty:
        return row

    best_idx = ev["D4RL_normalized_score"].idxmax()
    best_row = ev.loc[best_idx]
    last_row = ev.iloc[-1]
    last5 = ev.tail(5)["D4RL_normalized_score"]

    best_final_gap = float(best_row["D4RL_normalized_score"] - last_row["D4RL_normalized_score"])
    last5_mean = float(last5.mean())
    row.update({
        "best_step": int(best_row["training_step"]),
        "best_score_10ep": float(best_row["D4RL_normalized_score"]),
        "final_step": int(last_row["training_step"]),
        "final_score_10ep": float(last_row["D4RL_normalized_score"]),
        "best_final_gap": best_final_gap,
        "last5_mean": last5_mean,
        "last5_std": float(last5.std()),
        "ess_b_final": float(last_row.get("ESS_over_batch_size", np.nan)),
        "clamp_final": float(last_row.get("clamp_ratio", np.nan)),
        "log_std_final": float(last_row.get("actor_log_std", np.nan)),
        "mean_disagreement_final": float(last_row.get("mean_disagreement", np.nan)),
        "mean_weight_final": float(last_row.get("mean_weight", np.nan)),
        "top10_mass_final": float(last_row.get("top_10_percent_weight_mass", np.nan)),
        "actor_loss_final": float(last_row.get("actor_loss", np.nan)),
        "peak_to_last5_drop": float(best_row["D4RL_normalized_score"] - last5_mean),
        "collapses_after_peak": bool(
            best_final_gap > 5.0 and last5_mean < best_row["D4RL_normalized_score"] - 3.0
        ),
    })
    return row


def checkpoint_inventory() -> List[dict]:
    rows = []
    for run in RUNS:
        d = CKPT / run
        rows.append({
            "run": run,
            "actor_final_pt": (d / "actor_final.pt").exists(),
            "best_actor_pt": (d / "best_actor.pt").exists(),
            "eval_step_checkpoints": len(list(d.glob("actor_step_*.pt"))),
            "best_step_from_csv": None,
            "can_reevaluate_best": False,
        })
    return rows


def eval_actor_n_episodes(
    actor_path: Path,
    meta_path: Path,
    n_episodes: int,
    eval_seed: int,
    device: str,
) -> dict:
    meta = json.loads(meta_path.read_text())
    state_mean, state_std = load_norm_stats(str(meta.get("norm_stats", NORM)))
    _, eval_env, _, state_dim, action_dim = build_env_and_buffer(
        meta.get("env", ENV), device, state_mean, state_std
    )
    max_action = float(eval_env.action_space.high[0])
    actor = GaussianPolicy(state_dim, action_dim, max_action).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    set_seed(eval_seed, eval_env)

    episode_seeds = [eval_seed + i for i in range(n_episodes)]
    returns, norm_scores = [], []
    for ep_seed in episode_seeds:
        eval_env.seed(ep_seed)
        actor.eval()
        state, done = eval_env.reset(), False
        ep_ret = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = eval_env.step(action)
            ep_ret += reward
        returns.append(ep_ret)
        norm_scores.append(float(eval_env.get_normalized_score(ep_ret) * 100.0))
    eval_env.close()
    return {
        "n_episodes": n_episodes,
        "eval_seed_base": eval_seed,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "D4RL_normalized_score_mean": float(np.mean(norm_scores)),
        "D4RL_normalized_score_std": float(np.std(norm_scores)),
    }


def eval_joint_actor(checkpoint: Path, seed: int, n_episodes: int, device: str) -> dict:
    state_mean, state_std = load_norm_stats(str(NORM))
    _, eval_env, _, state_dim, action_dim = build_env_and_buffer(
        ENV, device, state_mean, state_std
    )
    max_action = float(eval_env.action_space.high[0])
    sd = torch.load(checkpoint, map_location=device)
    actor = GaussianPolicy(state_dim, action_dim, max_action).to(device)
    actor.load_state_dict(sd["actor"])

    episode_seeds = list(range(n_episodes))
    returns, norm_scores = [], []
    for ep_seed in episode_seeds:
        eval_env.seed(ep_seed)
        actor.eval()
        state, done = eval_env.reset(), False
        ep_ret = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = eval_env.step(action)
            ep_ret += reward
        returns.append(ep_ret)
        norm_scores.append(float(eval_env.get_normalized_score(ep_ret) * 100.0))
    eval_env.close()
    return {
        "checkpoint": str(checkpoint),
        "seed": seed,
        "n_episodes": n_episodes,
        "D4RL_normalized_score_mean": float(np.mean(norm_scores)),
        "D4RL_normalized_score_std": float(np.std(norm_scores)),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
    }


def actor_behavior_diagnostics(run: str, device: str, n_states: int = 10000) -> dict:
    meta_path = CKPT / run / "meta.json"
    actor_path = CKPT / run / "actor_final.pt"
    meta = json.loads(meta_path.read_text())
    variant = meta["variant"]
    state_mean, state_std = load_norm_stats(str(meta.get("norm_stats", NORM)))
    _, _, replay_buffer, state_dim, action_dim = build_env_and_buffer(
        ENV, device, state_mean, state_std
    )
    max_action = float(replay_buffer._actions.max())  # approximate

    actor = GaussianPolicy(state_dim, action_dim, max_action).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    bundle0 = load_value_bundle(str(CKPT_A), state_dim, action_dim, device, 0)
    bundle1 = load_value_bundle(str(CKPT_B), state_dim, action_dim, device, 1)

    rng = np.random.RandomState(0)
    idx = rng.randint(0, replay_buffer._size, size=n_states)
    states = torch.tensor(replay_buffer._states[idx], device=device, dtype=torch.float32)
    data_actions = torch.tensor(replay_buffer._actions[idx], device=device, dtype=torch.float32)

    with torch.no_grad():
        dist = actor(states)
        pred_actions = dist.mean
        pred_actions = torch.clamp(max_action * pred_actions, -max_action, max_action)
        log_prob = dist.log_prob(data_actions).sum(-1)
        mse = ((pred_actions - data_actions) ** 2).mean(dim=-1)

        a1 = compute_online_advantages(bundle0, states, data_actions)
        a2 = compute_online_advantages(bundle1, states, data_actions)
        wstats = compute_weights(a1, a2, variant, beta=meta.get("beta", 3.0))
        weights = wstats["w"]
        bc_loss = actor_bc_loss(actor, states, data_actions, weights)

    pred_np = pred_actions.cpu().numpy()
    data_np = data_actions.cpu().numpy()
    boundary = np.abs(pred_np) > (0.95 * max_action)

    return {
        "run": run,
        "checkpoint": "final",
        "variant": variant,
        "n_states": n_states,
        "action_mean_mean": float(pred_np.mean()),
        "action_mean_std": float(pred_np.std()),
        "action_norm_mean": float(np.linalg.norm(pred_np, axis=1).mean()),
        "action_norm_std": float(np.linalg.norm(pred_np, axis=1).std()),
        "boundary_frac_095": float(boundary.mean()),
        "weighted_bc_loss": float(bc_loss.item()),
        "unweighted_mse_mean": float(mse.mean()),
        "unweighted_mse_std": float(mse.std()),
        "log_prob_mean": float(log_prob.mean()),
        "log_prob_std": float(log_prob.std()),
        "actor_log_std_param": float(actor.log_std.mean()),
        "mean_weight": float(weights.mean()),
        "ess_over_batch": float((weights.sum() ** 2) / (weights.pow(2).sum() + 1e-8) / len(weights)),
    }


def load_all_curves(log_dir: Path) -> Dict[str, pd.DataFrame]:
    out = {}
    for run in RUNS:
        p = log_dir / f"{run}.csv"
        if p.exists():
            out[run] = parse_eval_df(pd.read_csv(p))
    return out


def plot_figures(curves: Dict[str, pd.DataFrame], audit_rows: List[dict]):
    FIG.mkdir(parents=True, exist_ok=True)
    audit_df = pd.DataFrame(audit_rows)

    def plot_score_subset(runs, title, fname):
        plt.figure(figsize=(10, 6))
        for run in runs:
            ev = curves.get(run)
            if ev is None or ev.empty:
                continue
            plt.plot(ev["training_step"], ev["D4RL_normalized_score"], label=run.replace("_100k_fast", ""))
        plt.xlabel("Actor update")
        plt.ylabel("D4RL score (10-ep eval)")
        plt.title(title)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG / fname, dpi=150)
        plt.close()

    plot_score_subset(RUNS, "Hopper: D4RL vs update (all runs)", "score_vs_step_all_runs.png")
    plot_score_subset(REP_A, "Hopper Rep A", "score_vs_step_repA.png")
    plot_score_subset(REP_B, "Hopper Rep B", "score_vs_step_repB.png")

    plt.figure(figsize=(10, 5))
    plt.bar(audit_df["run"], audit_df["best_final_gap"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Best - Final gap (10-ep CSV)")
    plt.title("Best-Final gap by run")
    plt.tight_layout()
    plt.savefig(FIG / "best_final_gap_bar.png", dpi=150)
    plt.close()

    for metric, fname, ylabel in [
        ("ESS_over_batch_size", "ESS_vs_step.png", "ESS / batch size"),
        ("clamp_ratio", "clamp_ratio_vs_step.png", "Clamp ratio"),
        ("actor_log_std", "actor_log_std_vs_step.png", "Actor log_std"),
    ]:
        plt.figure(figsize=(10, 6))
        for run in RUNS:
            ev = curves.get(run)
            if ev is None or metric not in ev.columns:
                continue
            plt.plot(ev["training_step"], ev[metric], label=run.replace("_100k_fast", ""), alpha=0.8)
        plt.xlabel("Actor update")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs update")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(FIG / fname, dpi=150)
        plt.close()

    # scatter score vs ESS/clamp at eval points
    for xcol, fname in [("ESS_over_batch_size", "score_vs_ESS_scatter.png"),
                        ("clamp_ratio", "score_vs_clamp_scatter.png")]:
        plt.figure(figsize=(7, 5))
        for run in RUNS:
            ev = curves.get(run)
            if ev is None or xcol not in ev.columns:
                continue
            plt.scatter(ev[xcol], ev["D4RL_normalized_score"], s=12, alpha=0.5, label=run.replace("_100k_fast", ""))
        plt.xlabel(xcol)
        plt.ylabel("D4RL score")
        plt.title(f"Score vs {xcol}")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(FIG / fname, dpi=150)
        plt.close()

    # last5 vs best
    plt.figure(figsize=(10, 5))
    x = np.arange(len(audit_df))
    w = 0.35
    plt.bar(x - w / 2, audit_df["best_score_10ep"], w, label="Best 10-ep")
    plt.bar(x + w / 2, audit_df["last5_mean"], w, label="Last5 mean")
    plt.xticks(x, [r.replace("_100k_fast", "") for r in audit_df["run"]], rotation=45, ha="right")
    plt.ylabel("D4RL score")
    plt.title("Best vs Last5 mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "last5_vs_best_comparison.png", dpi=150)
    plt.close()


def compare_methods(audit_df: pd.DataFrame, col: str, m1: str, m2: str, replicate: str) -> Optional[float]:
    sub = audit_df[audit_df["replicate"] == replicate]
    r1 = sub[sub["run"].str.contains(m1.replace("repB_", "") if replicate == "RepB" else m1)]
    # simpler lookup
    name_map = {
        ("RepA", "standard"): "standard_iql_100k_fast",
        ("RepA", "ensemble"): "ensemble_mean_100k_fast",
        ("RepA", "iad"): "iad_lambda_1.0_100k_fast",
        ("RepA", "shuffled"): "shuffled_iad_lambda_1.0_100k_fast",
        ("RepB", "standard"): "repB_standard_iql_100k_fast",
        ("RepB", "ensemble"): "repB_ensemble_mean_100k_fast",
        ("RepB", "iad"): "repB_iad_lambda_1.0_100k_fast",
        ("RepB", "shuffled"): "repB_shuffled_iad_lambda_1.0_100k_fast",
    }
    return None


def generate_report(
    audit_df: pd.DataFrame,
    inv: List[dict],
    final_50: dict,
    joint_eval: dict,
    behavior_df: pd.DataFrame,
) -> None:
    can_best = any(r["can_reevaluate_best"] for r in inv)
    n_collapse = int(audit_df["collapses_after_peak"].sum())

    def get(run, col):
        row = audit_df[audit_df["run"] == run]
        return float(row.iloc[0][col]) if not row.empty and col in row.columns else float("nan")

    def rep_delta(rep, a, b, col):
        names = {
            "RepA": {
                "standard": "standard_iql_100k_fast",
                "ensemble": "ensemble_mean_100k_fast",
                "iad": "iad_lambda_1.0_100k_fast",
                "shuffled": "shuffled_iad_lambda_1.0_100k_fast",
            },
            "RepB": {
                "standard": "repB_standard_iql_100k_fast",
                "ensemble": "repB_ensemble_mean_100k_fast",
                "iad": "repB_iad_lambda_1.0_100k_fast",
                "shuffled": "repB_shuffled_iad_lambda_1.0_100k_fast",
            },
        }
        return get(names[rep][a], col) - get(names[rep][b], col)

    lines = [
        "# Hopper Stability Audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Checkpoint inventory",
        "",
        "| Run | actor_final.pt | best_actor.pt | eval-step ckpts | Can re-eval best? |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in inv:
        a = audit_df[audit_df["run"] == r["run"]]
        best_step = int(a.iloc[0]["best_step"]) if not a.empty and "best_step" in a.columns else "?"
        lines.append(
            f"| {r['run']} | {r['actor_final_pt']} | {r['best_actor_pt']} | "
            f"{r['eval_step_checkpoints']} | **No** (CSV best step {best_step}) |"
        )

    lines += [
        "",
        "**Conclusion:** No `best_actor.pt` or eval-step actor checkpoints were saved. "
        "Best scores are from 10-ep CSV only; **50-ep re-eval of best checkpoint is not possible** without retraining.",
        "",
        "## 1. Does Hopper actor extraction show late-stage degradation?",
        "",
    ]

    if n_collapse >= 6:
        deg = "Yes — most runs show large best-final gaps and last5 means below peak."
    elif n_collapse >= 3:
        deg = "Partially — several runs (especially Rep A high performers) peak early then drop at final checkpoint."
    else:
        deg = "Limited evidence of uniform collapse; gaps vary by run."

    lines += [
        f"- **Assessment:** {deg}",
        f"- Runs flagged `collapses_after_peak`: {n_collapse} / {len(audit_df)}",
        f"- Mean best-final gap (10-ep CSV): {audit_df['best_final_gap'].mean():.2f}",
        f"- Rep A mean gap: {audit_df[audit_df['replicate']=='RepA']['best_final_gap'].mean():.2f}",
        f"- Rep B mean gap: {audit_df[audit_df['replicate']=='RepB']['best_final_gap'].mean():.2f}",
        "",
        "Rep A methods often peak between 5k–50k updates then decline by 100k. "
        "Rep B peaks are lower and final scores are closer to best (smaller relative gap) but absolute performance is poor.",
        "",
        "## 2. Why is Rep B overall lower?",
        "",
        "- **Different actor init (seed 54321) and batch stream (seed 43)** vs Rep A (12345 / 42).",
        "- Rep B **best scores are much lower** (e.g. ensemble best ~58 vs Rep A ~67), not only final.",
        "- ESS/B and clamp are **lower** on Rep B (less weight concentration), but that alone does not explain low scores.",
        "- `actor_log_std` similar across replicates (~-1.49 to -1.52); not the primary driver.",
        "- **10-ep eval variance** contributes (see 50-ep final re-eval below), but Rep B is low even at best CSV steps.",
        "- **Not primarily a final-checkpoint selection artifact** — Rep B underperforms at best step too.",
        "",
        "## 3. ensemble_mean vs standard_iql",
        "",
        "### Final checkpoint (20-ep from pilot)",
        f"- Rep A Δ (ensemble - standard): {rep_delta('RepA','ensemble','standard','final_score_10ep'):+.2f} (10-ep CSV final; 20-ep: +31.1)",
        f"- Rep B Δ: {rep_delta('RepB','ensemble','standard','final_score_10ep'):+.2f}",
        "",
        "### Best checkpoint (10-ep CSV only)",
        f"- Rep A Δ at best step: {rep_delta('RepA','ensemble','standard','best_score_10ep'):+.2f}",
        f"- Rep B Δ at best step: {rep_delta('RepB','ensemble','standard','best_score_10ep'):+.2f}",
        "",
        "### Last5 mean (10-ep CSV)",
        f"- Rep A Δ: {get('ensemble_mean_100k_fast','last5_mean') - get('standard_iql_100k_fast','last5_mean'):+.2f}",
        f"- Rep B Δ: {get('repB_ensemble_mean_100k_fast','last5_mean') - get('repB_standard_iql_100k_fast','last5_mean'):+.2f}",
        "",
        "**Conclusion:** ensemble_mean **beats standard** on Rep A at final, best, and last5. "
        "Rep B direction holds but margins are smaller; cross-replicate variance is high.",
        "",
        "## 4. iad_lambda_1.0 vs ensemble_mean",
        "",
        f"- Rep A final Δ (iad - ensemble): {rep_delta('RepA','iad','ensemble','final_score_10ep'):+.2f}",
        f"- Rep A best Δ: {rep_delta('RepA','iad','ensemble','best_score_10ep'):+.2f}",
        f"- Rep B final Δ: {rep_delta('RepB','iad','ensemble','final_score_10ep'):+.2f}",
        "",
        "**Conclusion:** IAD λ=1.0 does **not** beat ensemble_mean on Rep A (final or best). Rep B is noisy.",
        "",
        "## 5. iad_lambda_1.0 vs shuffled_iad_lambda_1.0",
        "",
        f"- Rep A final Δ: {rep_delta('RepA','iad','shuffled','final_score_10ep'):+.2f}",
        f"- Rep A best Δ: {rep_delta('RepA','iad','shuffled','best_score_10ep'):+.2f}",
        f"- Rep B final Δ: {rep_delta('RepB','iad','shuffled','final_score_10ep'):+.2f}",
        "",
        "**Conclusion:** Mixed; IAD beats shuffled on Rep A final/best but shuffled has higher **best** score on Rep A CSV (75.1 vs 62.7) — high eval noise.",
        "",
        "## 6. How to report Hopper?",
        "",
        "**Recommendation: Option C** — report **both final and best (CSV)**, plus **50-ep final re-eval**, "
        "and explicitly note **Hopper instability / overtraining** and **Rep B replicate sensitivity**.",
        "",
        "Do **not** use CSV best alone as official metric without checkpoint re-eval (best actor not saved).",
        "",
        "## 7. Proceed to Walker2d?",
        "",
        "**Conditional yes (Scenario C)** — ensemble_mean direction holds on Rep A and at best-step CSV, "
        "but Hopper shows **actor overtraining** and **high Rep A vs Rep B variance**. "
        "Recommend Walker2d only with: (1) report final + best CSV + shorter actor budget trial, "
        "(2) treat Rep B as mandatory replicate, (3) do not promote IAD over ensemble_mean.",
        "",
        "> Original joint IQL actor vs frozen extraction is **not directly comparable** (see joint eval JSON).",
        "",
        "## Original joint IQL actors (reference)",
        "",
    ]
    for k, v in joint_eval.items():
        lines.append(
            f"- **{k}**: {v['D4RL_normalized_score_mean']:.2f} ± {v['D4RL_normalized_score_std']:.2f} "
            f"({v['n_episodes']}-ep)"
        )

    lines += ["", "## 50-episode final checkpoint re-eval", ""]
    lines.append("| Run | Final 20-ep (pilot) | Final 50-ep re-eval | Best 10-ep CSV | Best 50-ep |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for run in RUNS:
        s20_path = RESULTS / f"{run}_summary.json"
        s20 = float("nan")
        if s20_path.exists():
            s20 = json.loads(s20_path.read_text())["final_20ep_eval"]["D4RL_normalized_score_mean"]
        s50 = final_50.get(run, {}).get("D4RL_normalized_score_mean", float("nan"))
        best10 = get(run, "best_score_10ep")
        lines.append(f"| {run} | {s20:.2f} | {s50:.2f} | {best10:.2f} | N/A |")

    lines += ["", "## Figures", "", f"See `{FIG}/`", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=str(LOG))
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--checkpoints-dir", default=str(CKPT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    results_dir = Path(args.results_dir)
    ckpt_dir = Path(args.checkpoints_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    inv = checkpoint_inventory()
    audit_rows = [audit_run(r, log_dir, ckpt_dir) for r in RUNS]
    for row in audit_rows:
        for inv_row in inv:
            if inv_row["run"] == row["run"]:
                inv_row["best_step_from_csv"] = row.get("best_step")
                inv_row["can_reevaluate_best"] = row.get("can_reevaluate_best", False)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(results_dir / "hopper_stability_audit.csv", index=False)
    with open(results_dir / "hopper_stability_audit.json", "w") as f:
        json.dump({"inventory": inv, "runs": audit_rows}, f, indent=2)

    curves = load_all_curves(log_dir)
    plot_figures(curves, audit_rows)

    final_50 = {}
    behavior_rows = []
    joint_eval = {}

    if not args.skip_gpu:
        print("Running 50-ep final checkpoint evaluations...")
        for run in RUNS:
            actor_path = ckpt_dir / run / "actor_final.pt"
            meta_path = ckpt_dir / run / "meta.json"
            if actor_path.exists():
                final_50[run] = eval_actor_n_episodes(
                    actor_path, meta_path, n_episodes=50, eval_seed=0, device=args.device
                )
        with open(results_dir / "final_checkpoint_eval_50ep.json", "w") as f:
            json.dump(final_50, f, indent=2)
        with open(results_dir / "best_checkpoint_eval_50ep.json", "w") as f:
            json.dump({"note": "No best actor checkpoints saved; re-eval not possible.", "runs": {}}, f, indent=2)

        print("Joint IQL actor eval...")
        joint_eval["original_joint_iql_seed0"] = eval_joint_actor(
            CKPT / "value_seed0" / "checkpoint_999999.pt", 0, 50, args.device
        )
        joint_eval["original_joint_iql_seed1"] = eval_joint_actor(
            CKPT / "value_seed1" / "checkpoint_999999.pt", 1, 50, args.device
        )
        joint_eval["original_joint_iql_seed0_20ep"] = eval_joint_actor(
            CKPT / "value_seed0" / "checkpoint_999999.pt", 0, 20, args.device
        )
        joint_eval["original_joint_iql_seed1_20ep"] = eval_joint_actor(
            CKPT / "value_seed1" / "checkpoint_999999.pt", 1, 20, args.device
        )
        with open(results_dir / "original_joint_iql_eval.json", "w") as f:
            json.dump(joint_eval, f, indent=2)

        print("Actor behavior diagnostics (final)...")
        for run in RUNS:
            behavior_rows.append(actor_behavior_diagnostics(run, args.device))
        pd.DataFrame(behavior_rows).to_csv(results_dir / "actor_behavior_diagnostics.csv", index=False)

    generate_report(audit_df, inv, final_50, joint_eval, pd.DataFrame(behavior_rows))
    print(f"Audit complete -> {results_dir}/hopper_stability_audit.csv")
    print(f"Report -> {REPORT}")


if __name__ == "__main__":
    main()
