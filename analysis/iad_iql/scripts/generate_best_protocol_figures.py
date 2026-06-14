#!/usr/bin/env python3
"""Figures for best-checkpoint protocol."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/code/analysis/iad_iql/best_checkpoint_protocol")
FIG = ROOT / "figures"
ENVS = {
    "halfcheetah": ROOT / "halfcheetah_medium_v2",
    "hopper": ROOT / "hopper_medium_v2",
    "walker2d": ROOT / "walker2d_medium_v2",
}


def load_all_summaries():
    rows = []
    for env_label, proj in ENVS.items():
        rdir = proj / "results"
        for p in rdir.glob("*_summary.json"):
            s = json.loads(p.read_text())
            if "smoke" in s["run_name"]:
                continue
            rows.append({
                "env": env_label,
                "method": s["method"],
                "rep": s["replicate"],
                "best_50ep": s["best_50ep_score_mean"],
                "final_50ep": s["final_50ep_score_mean"],
                "best_step": s["best_step"],
                "gap": s["best_final_gap_50ep"],
            })
    return pd.DataFrame(rows)


def load_curves(env_proj):
    frames = []
    log_dir = env_proj / "logs"
    for csv in log_dir.glob("*.csv"):
        if "smoke" in csv.name:
            continue
        df = pd.read_csv(csv)
        ev = df[df["D4RL_normalized_score"].notna() & (df["D4RL_normalized_score"] != "")]
        if ev.empty:
            continue
        ev = ev.copy()
        ev["D4RL_normalized_score"] = pd.to_numeric(ev["D4RL_normalized_score"])
        ev["run"] = csv.stem
        frames.append(ev)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_all_summaries()
    if df.empty:
        print("No summaries for figures.")
        return

    for col, fname in [("best_50ep", "best50_by_env_method.png"), ("final_50ep", "final50_by_env_method.png")]:
        plt.figure(figsize=(10, 5))
        for env in df["env"].unique():
            sub = df[df["env"] == env]
            means = sub.groupby("method")[col].mean()
            stds = sub.groupby("method")[col].std().fillna(0)
            x = range(len(means))
            plt.bar([i + (0 if env == "hopper" else 0.4) for i in x], means, width=0.35,
                    yerr=stds, label=env, alpha=0.8)
        plt.xticks(range(len(means)), means.index, rotation=20, ha="right")
        plt.ylabel("D4RL (50-ep mean)")
        plt.title(col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / fname, dpi=150)
        plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(df["best_step"], bins=20, edgecolor="black")
    plt.xlabel("Best step")
    plt.ylabel("Count")
    plt.title("Best step distribution")
    plt.tight_layout()
    plt.savefig(FIG / "best_step_distribution.png", dpi=150)
    plt.close()

    pivot = df.groupby(["env", "method"])["gap"].mean().unstack(fill_value=0)
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Best - Final gap (50-ep)")
    plt.title("Best-final gap by env/method")
    plt.tight_layout()
    plt.savefig(FIG / "best_final_gap_by_env_method.png", dpi=150)
    plt.close()

    for env_label, proj in ENVS.items():
        curves = load_curves(proj)
        if curves.empty:
            continue
        plt.figure(figsize=(10, 6))
        for run, sub in curves.groupby("run"):
            plt.plot(sub["training_step"], sub["D4RL_normalized_score"], alpha=0.7, label=run)
        plt.xlabel("Update")
        plt.ylabel("D4RL (10-ep)")
        plt.title(f"Score vs step — {env_label}")
        plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(FIG / f"score_vs_step_{env_label}.png", dpi=150)
        plt.close()

    comp = pd.read_csv(ROOT / "results" / "best_protocol_comparison.csv")
    ens = comp[comp["Comparison"].str.contains("ensemble_mean - standard")]
    if not ens.empty:
        plt.figure(figsize=(6, 4))
        plt.bar(ens["Env"], ens["Best 50-ep Delta"])
        plt.ylabel("Delta (Best 50-ep)")
        plt.title("ensemble_mean vs standard_iql")
        plt.tight_layout()
        plt.savefig(FIG / "ensemble_vs_standard_delta.png", dpi=150)
        plt.close()

    iad = comp[comp["Comparison"].str.contains("iad_lambda_1.0 - shuffled")]
    if not iad.empty:
        plt.figure(figsize=(6, 4))
        plt.bar(iad["Env"], iad["Best 50-ep Delta"])
        plt.ylabel("Delta (Best 50-ep)")
        plt.title("IAD vs shuffled λ=1.0")
        plt.tight_layout()
        plt.savefig(FIG / "iad_vs_shuffled_delta.png", dpi=150)
        plt.close()

    print(f"Figures -> {FIG}")


if __name__ == "__main__":
    main()
