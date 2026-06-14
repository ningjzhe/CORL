#!/usr/bin/env python3
"""Walker2d-specific figures for best-checkpoint protocol."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/code/analysis/iad_iql/best_checkpoint_protocol/walker2d_medium_v2")
FIG = ROOT / "figures"
RESULTS = ROOT / "results"
GLOBAL = Path("/code/analysis/iad_iql/best_checkpoint_protocol/results")


def load_summaries():
    rows = []
    for p in RESULTS.glob("*_summary.json"):
        s = json.loads(p.read_text())
        if s.get("status") != "completed" or "smoke" in s["run_name"]:
            continue
        rows.append({
            "method": s["method"],
            "rep": s["replicate"],
            "best_50ep": s["best_50ep_score_mean"],
            "final_50ep": s["final_50ep_score_mean"],
            "best_step": s["best_step"],
            "gap": s["best_final_gap_50ep"],
        })
    return pd.DataFrame(rows)


def load_curves():
    frames = []
    for csv in (ROOT / "logs").glob("*.csv"):
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


def bar_by_method(df, col, title, fname):
    plt.figure(figsize=(8, 5))
    means = df.groupby("method")[col].mean()
    stds = df.groupby("method")[col].std().fillna(0)
    x = range(len(means))
    plt.bar(x, means, yerr=stds, capsize=4, alpha=0.85)
    plt.xticks(x, means.index, rotation=20, ha="right")
    plt.ylabel("D4RL (50-ep mean)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=150)
    plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_summaries()
    if df.empty:
        print("No Walker2d summaries for figures.")
        return

    bar_by_method(df, "best_50ep", "Best 50-ep by method — Walker2d", "best50_by_method_walker2d.png")
    bar_by_method(df, "final_50ep", "Final 50-ep by method — Walker2d", "final50_by_method_walker2d.png")

    plt.figure(figsize=(8, 5))
    means = df.groupby("method")["gap"].mean()
    stds = df.groupby("method")["gap"].std().fillna(0)
    x = range(len(means))
    plt.bar(x, means, yerr=stds, capsize=4, alpha=0.85)
    plt.xticks(x, means.index, rotation=20, ha="right")
    plt.ylabel("Best - Final gap (50-ep)")
    plt.title("Best-final gap — Walker2d")
    plt.tight_layout()
    plt.savefig(FIG / "best_final_gap_walker2d.png", dpi=150)
    plt.close()

    curves = load_curves()
    if not curves.empty:
        plt.figure(figsize=(10, 6))
        for run, sub in curves.groupby("run"):
            plt.plot(sub["training_step"], sub["D4RL_normalized_score"], alpha=0.7, label=run)
        plt.xlabel("Update")
        plt.ylabel("D4RL (10-ep)")
        plt.title("Score vs step — Walker2d")
        plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(FIG / "score_vs_step_walker2d.png", dpi=150)
        plt.close()

    if (GLOBAL / "best_protocol_comparison.csv").exists():
        comp = pd.read_csv(GLOBAL / "best_protocol_comparison.csv")
        w = comp[comp["Env"] == "walker2d"]
        for label, substr, fname in [
            ("ensemble_mean vs standard_iql", "ensemble_mean - standard", "ensemble_vs_standard_delta_walker2d.png"),
            ("IAD vs ensemble_mean", "iad_lambda_1.0 - ensemble_mean", "iad_vs_ensemble_delta_walker2d.png"),
            ("IAD vs shuffled", "iad_lambda_1.0 - shuffled", "iad_vs_shuffled_delta_walker2d.png"),
        ]:
            sub = w[w["Comparison"].str.contains(substr)]
            if sub.empty:
                continue
            plt.figure(figsize=(5, 4))
            plt.bar(["Best 50-ep", "Final 50-ep"], [sub["Best 50-ep Delta"].iloc[0], sub["Final 50-ep Delta"].iloc[0]])
            plt.ylabel("Delta")
            plt.title(f"{label} — Walker2d")
            plt.tight_layout()
            plt.savefig(FIG / fname, dpi=150)
            plt.close()

    print(f"Walker2d figures -> {FIG}")


if __name__ == "__main__":
    main()
