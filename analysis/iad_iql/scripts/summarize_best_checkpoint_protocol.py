#!/usr/bin/env python3
"""Summarize best-checkpoint protocol runs across envs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/code/analysis/iad_iql/best_checkpoint_protocol")
ENVS = {
    "halfcheetah-medium-v2": ROOT / "halfcheetah_medium_v2" / "results",
    "hopper-medium-v2": ROOT / "hopper_medium_v2" / "results",
    "walker2d-medium-v2": ROOT / "walker2d_medium_v2" / "results",
}
WALKER_OUT = ROOT / "walker2d_medium_v2" / "results"
OUT = ROOT / "results"
METHODS = ["standard_iql", "ensemble_mean", "iad_lambda_1.0", "shuffled_iad_lambda_1.0"]
COMPARISONS = [
    ("ensemble_mean", "standard_iql", "ensemble_mean - standard_iql"),
    ("iad_lambda_1.0", "ensemble_mean", "iad_lambda_1.0 - ensemble_mean"),
    ("iad_lambda_1.0", "shuffled_iad_lambda_1.0", "iad_lambda_1.0 - shuffled_iad_lambda_1.0"),
]


def load_summaries():
    rows = []
    for env, rdir in ENVS.items():
        if not rdir.exists():
            continue
        for p in sorted(rdir.glob("*_summary.json")):
            s = json.loads(p.read_text())
            if s.get("status") != "completed":
                continue
            if "smoke" in s.get("run_name", p.stem):
                continue
            rows.append({
                "env": env,
                "run_name": s["run_name"],
                "rep": s.get("replicate", "?"),
                "method": s.get("method"),
                "best_step": s.get("best_step"),
                "best_10ep": s.get("best_10ep_score"),
                "best_50ep": s.get("best_50ep_score_mean"),
                "best_50ep_std": s.get("best_50ep_score_std"),
                "final_50ep": s.get("final_50ep_score_mean"),
                "final_50ep_std": s.get("final_50ep_score_std"),
                "gap_50ep": s.get("best_final_gap_50ep"),
                "actor_seed": s.get("actor_seed"),
                "batch_seed": s.get("batch_seed"),
            })
    return pd.DataFrame(rows)


def env_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env in df["env"].unique():
        for method in METHODS:
            sub = df[(df["env"] == env) & (df["method"] == method)]
            if sub.empty:
                continue
            rows.append({
                "Env": env.replace("-medium-v2", ""),
                "Method": method,
                "Best 50-ep Mean ± Std": f"{sub['best_50ep'].mean():.2f} ± {sub['best_50ep'].std():.2f}",
                "Final 50-ep Mean ± Std": f"{sub['final_50ep'].mean():.2f} ± {sub['final_50ep'].std():.2f}",
                "Best Step Mean": f"{sub['best_step'].mean():.0f}",
                "N": len(sub),
            })
    return pd.DataFrame(rows)


def comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env in df["env"].unique():
        for m_high, m_low, label in COMPARISONS:
            sub_h = df[(df["env"] == env) & (df["method"] == m_high)]["best_50ep"]
            sub_l = df[(df["env"] == env) & (df["method"] == m_low)]["best_50ep"]
            sub_hf = df[(df["env"] == env) & (df["method"] == m_high)]["final_50ep"]
            sub_lf = df[(df["env"] == env) & (df["method"] == m_low)]["final_50ep"]
            if len(sub_h) == 0 or len(sub_l) == 0:
                continue
            rows.append({
                "Env": env.replace("-medium-v2", ""),
                "Comparison": label,
                "Best 50-ep Delta": float(sub_h.mean() - sub_l.mean()),
                "Final 50-ep Delta": float(sub_hf.mean() - sub_lf.mean()),
            })
    return pd.DataFrame(rows)


def replicate_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env in df["env"].unique():
        for rep in ["A", "B"]:
            sub = df[(df["env"] == env) & (df["rep"] == rep)]
            if sub.empty:
                continue
            rows.append({
                "Env": env.replace("-medium-v2", ""),
                "Rep": rep,
                "Best 50-ep Mean": float(sub["best_50ep"].mean()),
                "Final 50-ep Mean": float(sub["final_50ep"].mean()),
                "Gap Mean": float(sub["gap_50ep"].mean()),
            })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_summaries()
    if df.empty:
        print("No completed summaries found.")
        return

    table1 = df.rename(columns={
        "rep": "Rep", "method": "Method", "best_step": "Best Step",
        "best_10ep": "Best 10-ep", "best_50ep": "Best 50-ep",
        "final_50ep": "Final 50-ep", "gap_50ep": "Gap",
    })[["env", "Rep", "Method", "Best Step", "Best 10-ep", "Best 50-ep", "Final 50-ep", "Gap"]]
    table1 = table1.rename(columns={"env": "Env"})
    table1["Env"] = table1["Env"].str.replace("-medium-v2", "")

    table2 = env_method_summary(df)
    table3 = comparison_table(df)
    rep_df = replicate_summary(df)

    table1.to_csv(OUT / "best_protocol_all_runs.csv", index=False)
    table2.to_csv(OUT / "best_protocol_env_summary.csv", index=False)
    rep_df.to_csv(OUT / "best_protocol_replicate_summary.csv", index=False)
    table3.to_csv(OUT / "best_protocol_comparison.csv", index=False)

    summary_json = {
        "n_runs": len(df),
        "per_run": df.to_dict(orient="records"),
        "env_method": table2.to_dict(orient="records"),
        "comparison": table3.to_dict(orient="records"),
        "replicate": rep_df.to_dict(orient="records"),
    }
    with open(OUT / "best_protocol_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Summarized {len(df)} runs -> {OUT}")

    walker = df[df["env"] == "walker2d-medium-v2"]
    if not walker.empty:
        WALKER_OUT.mkdir(parents=True, exist_ok=True)
        rep_a = walker[walker["rep"] == "A"].rename(columns={
            "method": "Method", "best_step": "Best Step",
            "best_10ep": "Best 10-ep", "best_50ep": "Best 50-ep",
            "final_50ep": "Final 50-ep", "gap_50ep": "Gap",
        })
        rep_b = walker[walker["rep"] == "B"].rename(columns={
            "method": "Method", "best_step": "Best Step",
            "best_10ep": "Best 10-ep", "best_50ep": "Best 50-ep",
            "final_50ep": "Final 50-ep", "gap_50ep": "Gap",
        })
        rep_a[["Method", "Best Step", "Best 10-ep", "Best 50-ep", "Final 50-ep", "Gap"]].to_csv(
            WALKER_OUT / "main_results_repA.csv", index=False
        )
        rep_b[["Method", "Best Step", "Best 10-ep", "Best 50-ep", "Final 50-ep", "Gap"]].to_csv(
            WALKER_OUT / "main_results_repB.csv", index=False
        )
        rep_df.to_csv(WALKER_OUT / "main_results_replicates_summary.csv", index=False)
        walker_summary = {
            "n_runs": len(walker),
            "per_run": walker.to_dict(orient="records"),
            "env_method": table2[table2["Env"] == "walker2d"].to_dict(orient="records"),
            "comparison": table3[table3["Env"] == "walker2d"].to_dict(orient="records"),
            "replicate": rep_df[rep_df["Env"] == "walker2d"].to_dict(orient="records"),
        }
        with open(WALKER_OUT / "summary.json", "w") as f:
            json.dump(walker_summary, f, indent=2)


if __name__ == "__main__":
    main()
