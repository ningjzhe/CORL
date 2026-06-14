#!/usr/bin/env python3
"""Generate Hopper pilot figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REP_A = [
    "standard_iql_100k_fast",
    "ensemble_mean_100k_fast",
    "iad_lambda_1.0_100k_fast",
    "shuffled_iad_lambda_1.0_100k_fast",
]
LABELS = {
    "standard_iql_100k_fast": "Standard IQL",
    "ensemble_mean_100k_fast": "Ensemble Mean",
    "iad_lambda_1.0_100k_fast": "IAD λ=1.0",
    "shuffled_iad_lambda_1.0_100k_fast": "Shuffled IAD λ=1.0",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="/code/analysis/iad_iql/hopper_medium_v2/logs")
    parser.add_argument("--results-dir", default="/code/analysis/iad_iql/hopper_medium_v2/results")
    parser.add_argument("--figures-dir", default="/code/analysis/iad_iql/hopper_medium_v2/figures")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    results_dir = Path(args.results_dir)
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for stem in REP_A:
        csv_path = log_dir / f"{stem}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["method"] = stem
            frames.append(df)
    if frames:
        curves = pd.concat(frames, ignore_index=True)
        plt.figure(figsize=(8, 5))
        for stem in REP_A:
            sub = curves[curves["method"] == stem]
            ev = sub[sub["D4RL_normalized_score"].notna() & (sub["D4RL_normalized_score"] != "")]
            if ev.empty:
                continue
            ev = ev.copy()
            ev["D4RL_normalized_score"] = pd.to_numeric(ev["D4RL_normalized_score"])
            plt.plot(ev["training_step"], ev["D4RL_normalized_score"], label=LABELS.get(stem, stem))
        plt.xlabel("Actor update")
        plt.ylabel("D4RL score (10-ep)")
        plt.title("Hopper Rep A: D4RL vs update")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "hopper_repA_d4rl_vs_update.png", dpi=150)
        plt.close()

    summaries = []
    for stem in REP_A:
        p = results_dir / f"{stem}_summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        summaries.append({
            "method": LABELS.get(stem, stem),
            "final_20ep": s.get("final_20ep_eval", {}).get("D4RL_normalized_score_mean", float("nan")),
        })
    if summaries:
        df = pd.DataFrame(summaries)
        plt.figure(figsize=(7, 4))
        plt.bar(df["method"], df["final_20ep"])
        plt.ylabel("D4RL (20-ep)")
        plt.title("Hopper Rep A final scores")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "hopper_repA_final_20ep_bar.png", dpi=150)
        plt.close()

    cross_path = results_dir / "main_results_replicates_summary.csv"
    if cross_path.exists():
        cross = pd.read_csv(cross_path)
        if "Mean 20-ep" in cross.columns:
            plt.figure(figsize=(7, 4))
            plt.bar(cross["Method"], cross["Mean 20-ep"], yerr=cross.get("Std 20-ep"))
            plt.ylabel("D4RL (20-ep mean ± std)")
            plt.title("Hopper cross-replicate")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / "hopper_cross_replicate_20ep.png", dpi=150)
            plt.close()

    print(f"Figures -> {fig_dir}")


if __name__ == "__main__":
    main()
