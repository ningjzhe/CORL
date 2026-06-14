#!/usr/bin/env python3
"""Generate IAD-IQL overnight figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OFFICIAL_METHODS = [
    "standard_iql_100k",
    "ensemble_mean_100k_fast",
    "iad_lambda_0.5_100k_fast",
    "iad_lambda_1.0_100k_fast",
    "shuffled_iad_lambda_0.5_100k_fast",
]
LABELS = {
    "standard_iql_100k": "Standard IQL",
    "ensemble_mean_100k_fast": "Ensemble Mean",
    "iad_lambda_0.5_100k_fast": "IAD λ=0.5",
    "iad_lambda_1.0_100k_fast": "IAD λ=1.0",
    "shuffled_iad_lambda_0.5_100k_fast": "Shuffled IAD λ=0.5",
}


def load_curves(log_dir: Path) -> pd.DataFrame:
    frames = []
    for stem in OFFICIAL_METHODS:
        csv_path = log_dir / f"{stem}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["method"] = stem
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_metric_vs_update(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path):
    plt.figure(figsize=(8, 5))
    for method in OFFICIAL_METHODS:
        sub = df[df["method"] == method]
        if sub.empty or metric not in sub.columns:
            continue
        ev = sub[sub[metric].notna() & (sub[metric] != "")]
        if ev.empty:
            continue
        ev = ev.copy()
        ev[metric] = pd.to_numeric(ev[metric])
        plt.plot(ev["training_step"], ev[metric], label=LABELS.get(method, method))
    plt.xlabel("Actor update")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs update")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_final_scores(results_dir: Path, out_path: Path):
    summaries = []
    for method in OFFICIAL_METHODS:
        p = results_dir / f"{method}_summary.json"
        if not p.exists():
            continue
        with open(p) as f:
            s = json.load(f)
        summaries.append({
            "method": method,
            "final": s.get("final_d4rl_score", float("nan")),
            "best": s.get("best_d4rl_score", float("nan")),
            "final_20ep": s.get("final_20ep_eval", {}).get("D4RL_normalized_score_mean", float("nan")),
        })
    if not summaries:
        return
    sdf = pd.DataFrame(summaries)
    x = np.arange(len(sdf))
    plt.figure(figsize=(9, 5))
    width = 0.25
    plt.bar(x - width, sdf["final"], width, label="Final (10-ep)")
    plt.bar(x, sdf["best"], width, label="Best (10-ep)")
    plt.bar(x + width, sdf["final_20ep"], width, label="Final (20-ep)")
    plt.xticks(x, [LABELS.get(m, m) for m in sdf["method"]], rotation=20, ha="right")
    plt.ylabel("D4RL normalized score")
    plt.title("Final score comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_ess_vs_final(results_dir: Path, out_path: Path):
    xs, ys, labels = [], [], []
    for method in OFFICIAL_METHODS:
        p = results_dir / f"{method}_summary.json"
        if not p.exists():
            continue
        with open(p) as f:
            s = json.load(f)
        xs.append(s.get("ess_over_batch_size", float("nan")))
        ys.append(s.get("final_d4rl_score", float("nan")))
        labels.append(LABELS.get(method, method))
    if not xs:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys)
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), fontsize=8)
    plt.xlabel("ESS / batch_size (final)")
    plt.ylabel("Final D4RL score")
    plt.title("ESS/B vs final score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_compatibility(results_dir: Path, fig_dir: Path):
    compat_path = results_dir / "value_system_compatibility.json"
    if not compat_path.exists():
        return
    with open(compat_path) as f:
        c = json.load(f)
    # A1 vs A2 scatter would need raw data; use histogram of disagreement stats from last actor CSV
    for method in OFFICIAL_METHODS:
        csv_path = Path(f"/code/analysis/iad_iql/logs/{method}.csv")
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "mean_disagreement" not in df.columns:
            continue
        ev = df[df["mean_disagreement"].notna()].tail(1)
        if ev.empty:
            continue
        plt.figure(figsize=(5, 4))
        plt.bar(["mean", "p90", "p99"], [
            float(ev["mean_disagreement"].iloc[0]),
            float(ev["p90_disagreement"].iloc[0]),
            float(ev["p99_disagreement"].iloc[0]),
        ])
        plt.title(f"Disagreement ({LABELS.get(method, method)})")
        plt.tight_layout()
        plt.savefig(fig_dir / f"disagreement_{method}.png", dpi=150)
        plt.close()
        break

    if "pearson_A1_A2" in c:
        plt.figure(figsize=(5, 4))
        plt.bar(["Pearson", "Spearman"], [c["pearson_A1_A2"], c["spearman_A1_A2"]])
        plt.title("A1 vs A2 correlation")
        plt.tight_layout()
        plt.savefig(fig_dir / "a1_vs_a2_correlation.png", dpi=150)
        plt.close()

    d = c.get("disagreement_abs_A1_minus_A2_over_2", {})
    if d:
        plt.figure(figsize=(6, 4))
        plt.bar(list(d.keys()), list(d.values()))
        plt.title("Disagreement distribution (100k transitions)")
        plt.tight_layout()
        plt.savefig(fig_dir / "disagreement_histogram.png", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="/code/analysis/iad_iql/logs")
    parser.add_argument("--results-dir", default="/code/analysis/iad_iql/results")
    parser.add_argument("--fig-dir", default="/code/analysis/iad_iql/figures")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_curves(log_dir)
    if not df.empty:
        plot_metric_vs_update(df, "D4RL_normalized_score", "D4RL score", fig_dir / "d4rl_score_vs_update.png")
        plot_metric_vs_update(df, "ESS_over_batch_size", "ESS/B", fig_dir / "ess_over_b_vs_update.png")
        plot_metric_vs_update(df, "clamp_ratio", "Clamp ratio", fig_dir / "clamp_ratio_vs_update.png")
        plot_metric_vs_update(df, "actor_loss", "Actor loss", fig_dir / "actor_loss_vs_update.png")
        if "corr_A1_A2" in df.columns:
            plot_metric_vs_update(df, "corr_A1_A2", "corr(A1,A2)", fig_dir / "a1_vs_a2_vs_update.png")

    plot_final_scores(results_dir, fig_dir / "final_score_comparison.png")
    plot_ess_vs_final(results_dir, fig_dir / "ess_over_b_vs_final_score.png")
    plot_compatibility(results_dir, fig_dir)
    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
