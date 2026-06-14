#!/usr/bin/env python3
"""Summarize IAD-IQL experiment results into replicate tables."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REP_A_RUNS = {
    "standard_iql": "standard_iql_100k",
    "ensemble_mean": "ensemble_mean_100k_fast",
    "iad_lambda_0.5": "iad_lambda_0.5_100k_fast",
    "iad_lambda_1.0": "iad_lambda_1.0_100k_fast",
    "shuffled_iad_lambda_0.5": "shuffled_iad_lambda_0.5_100k_fast",
    "shuffled_iad_lambda_1.0": "shuffled_iad_lambda_1.0_100k_fast",
}

REP_B_RUNS = {
    "repB_standard_iql": "repB_standard_iql_100k_fast",
    "repB_ensemble_mean": "repB_ensemble_mean_100k_fast",
    "repB_iad_lambda_1.0": "repB_iad_lambda_1.0_100k_fast",
    "repB_shuffled_iad_lambda_1.0": "repB_shuffled_iad_lambda_1.0_100k_fast",
}

CROSS_REPLICATE = {
    "standard_iql": ("standard_iql_100k", "repB_standard_iql_100k_fast"),
    "ensemble_mean": ("ensemble_mean_100k_fast", "repB_ensemble_mean_100k_fast"),
    "iad_lambda_1.0": ("iad_lambda_1.0_100k_fast", "repB_iad_lambda_1.0_100k_fast"),
    "shuffled_iad_lambda_1.0": (
        "shuffled_iad_lambda_1.0_100k_fast",
        "repB_shuffled_iad_lambda_1.0_100k_fast",
    ),
}

HOPPER_REP_A = {
    "standard_iql": "standard_iql_100k_fast",
    "ensemble_mean": "ensemble_mean_100k_fast",
    "iad_lambda_1.0": "iad_lambda_1.0_100k_fast",
    "shuffled_iad_lambda_1.0": "shuffled_iad_lambda_1.0_100k_fast",
}

HOPPER_REP_B = dict(REP_B_RUNS)


def summarize_run(log_dir: Path, results_dir: Path, stem: str, label: str) -> dict:
    csv_path = log_dir / f"{stem}.csv"
    summary_path = results_dir / f"{stem}_summary.json"
    row = {"Method": label, "Run": stem}

    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        row.update({
            "Final 10-ep Score": s.get("final_d4rl_score"),
            "Final 20-ep Score": s.get("final_20ep_eval", {}).get("D4RL_normalized_score_mean"),
            "Best Score": s.get("best_d4rl_score"),
            "ESS/B": s.get("ess_over_batch_size"),
            "Clamp Ratio": s.get("clamp_ratio"),
            "Actor log_std": s.get("actor_log_std"),
            "QV frozen": s.get("qv_frozen"),
        })
        return row

    if not csv_path.exists():
        row["Status"] = "missing"
        return row

    df = pd.read_csv(csv_path)
    eval_df = df[df["D4RL_normalized_score"].notna() & (df["D4RL_normalized_score"] != "")]
    if eval_df.empty:
        row["Status"] = "no_eval"
        return row
    eval_df = eval_df.copy()
    eval_df["D4RL_normalized_score"] = pd.to_numeric(eval_df["D4RL_normalized_score"])
    last = eval_df.iloc[-1]
    row.update({
        "Final 10-ep Score": float(last["D4RL_normalized_score"]),
        "Best Score": float(eval_df["D4RL_normalized_score"].max()),
        "ESS/B": float(last.get("ESS_over_batch_size", float("nan"))),
        "Clamp Ratio": float(last.get("clamp_ratio", float("nan"))),
        "Actor log_std": float(last.get("actor_log_std", float("nan"))),
        "Status": "completed_no_summary",
    })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="/code/analysis/iad_iql/logs")
    parser.add_argument("--output-dir", default="/code/analysis/iad_iql/results")
    parser.add_argument(
        "--standard-iql-run",
        default="standard_iql_100k",
        help="Run name stem for Rep A standard_iql (Hopper: standard_iql_100k_fast)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rep_a_runs = dict(REP_A_RUNS)
    rep_a_runs["standard_iql"] = args.standard_iql_run
    cross_replicate = dict(CROSS_REPLICATE)
    cross_replicate["standard_iql"] = (args.standard_iql_run, "repB_standard_iql_100k_fast")

    if args.standard_iql_run == "standard_iql_100k_fast":
        rep_a_runs = dict(HOPPER_REP_A)
        rep_b_runs = dict(HOPPER_REP_B)
    else:
        rep_b_runs = dict(REP_B_RUNS)

    rep_a = [summarize_run(log_dir, out_dir, stem, label) for label, stem in rep_a_runs.items()]
    rep_b = [summarize_run(log_dir, out_dir, stem, label) for label, stem in rep_b_runs.items()]

    pd.DataFrame(rep_a).to_csv(out_dir / "main_results_repA.csv", index=False)
    pd.DataFrame(rep_b).to_csv(out_dir / "main_results_repB.csv", index=False)

    # Legacy combined outputs (Rep A primary)
    pd.DataFrame(rep_a).to_csv(out_dir / "main_results.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(rep_a, f, indent=2)

    # Cross-replicate summary on final 20-ep score
    cross_rows = []
    cross_json = {}
    for label, (stem_a, stem_b) in cross_replicate.items():
        scores = []
        for stem in (stem_a, stem_b):
            summary_path = out_dir / f"{stem}_summary.json"
            if summary_path.exists():
                s = json.loads(summary_path.read_text())
                scores.append(s["final_20ep_eval"]["D4RL_normalized_score_mean"])
        if len(scores) == 2:
            cross_rows.append({
                "Method": label,
                "RepA 20-ep": scores[0],
                "RepB 20-ep": scores[1],
                "Mean 20-ep": float(np.mean(scores)),
                "Std 20-ep": float(np.std(scores)),
            })
            cross_json[label] = {
                "repA": scores[0],
                "repB": scores[1],
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
        else:
            cross_rows.append({"Method": label, "Status": f"incomplete ({len(scores)}/2)"})

    pd.DataFrame(cross_rows).to_csv(out_dir / "main_results_replicates_summary.csv", index=False)
    with open(out_dir / "replicate_summary.json", "w") as f:
        json.dump(cross_json, f, indent=2)

    # Training curves for all official stems
    all_stems = list(rep_a_runs.values()) + list(rep_b_runs.values())
    curves = []
    for stem in all_stems:
        csv_path = log_dir / f"{stem}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["method"] = stem
            curves.append(df)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(out_dir / "training_curves.csv", index=False)

    print(f"RepA: {len(rep_a)} methods, RepB: {len(rep_b)} methods -> {out_dir}")


if __name__ == "__main__":
    main()
