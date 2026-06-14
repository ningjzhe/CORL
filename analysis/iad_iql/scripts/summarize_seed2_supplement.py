#!/usr/bin/env python3
"""Summarize ensemble_mean_3q supplement runs."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/code/analysis/iad_iql/seed2_supplement")
OUT = ROOT / "results"
ENVS = ["halfcheetah", "hopper", "walker2d"]
BCP = Path("/code/analysis/iad_iql/best_checkpoint_protocol")


def load_3q_summaries():
    rows = []
    for env in ENVS:
        rdir = ROOT / env / "results"
        if not rdir.exists():
            continue
        for p in rdir.glob("*_ensemble_mean_3q_summary.json"):
            s = json.loads(p.read_text())
            if s.get("status") != "completed":
                continue
            rows.append({
                "env": env,
                "run_name": s["run_name"],
                "rep": s["replicate"],
                "method": s["method"],
                "best_step": s["best_step"],
                "best_10ep": s["best_10ep_score"],
                "best_50ep": s["best_50ep_score_mean"],
                "final_50ep": s["final_50ep_score_mean"],
                "gap": s["best_final_gap_50ep"],
            })
    return pd.DataFrame(rows)


def load_2q_baseline(env):
    env_dir = {
        "halfcheetah": BCP / "halfcheetah_medium_v2",
        "hopper": BCP / "hopper_medium_v2",
        "walker2d": BCP / "walker2d_medium_v2",
    }[env]
    rows = []
    for method in ["standard_iql", "ensemble_mean"]:
        for p in (env_dir / "results").glob(f"*_{method}_summary.json"):
            s = json.loads(p.read_text())
            rows.append({"method": method, "rep": s["replicate"],
                         "best_50ep": s["best_50ep_score_mean"],
                         "final_50ep": s["final_50ep_score_mean"],
                         "gap": s["best_final_gap_50ep"]})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_3q_summaries()
    if df.empty:
        print("No 3Q summaries yet.")
        return

    table = df.rename(columns={
        "env": "Env", "rep": "Rep", "method": "Method",
        "best_step": "Best Step", "best_10ep": "Best 10-ep",
        "best_50ep": "Best 50-ep", "final_50ep": "Final 50-ep", "gap": "Gap",
    })
    table.to_csv(OUT / "ensemble3_all_runs.csv", index=False)

    comparisons = []
    for env in df["env"].unique():
        sub3 = df[df["env"] == env]
        base = load_2q_baseline(env)
        b_std = [r for r in base if r["method"] == "standard_iql"]
        b_2q = [r for r in base if r["method"] == "ensemble_mean"]
        if sub3.empty:
            continue
        comparisons.append({
            "env": env,
            "3q_best_mean": float(sub3["best_50ep"].mean()),
            "3q_final_mean": float(sub3["final_50ep"].mean()),
            "3q_gap_mean": float(sub3["gap"].mean()),
            "2q_best_mean": float(sum(r["best_50ep"] for r in b_2q) / len(b_2q)) if b_2q else None,
            "2q_final_mean": float(sum(r["final_50ep"] for r in b_2q) / len(b_2q)) if b_2q else None,
            "2q_gap_mean": float(sum(r["gap"] for r in b_2q) / len(b_2q)) if b_2q else None,
            "std_best_mean": float(sum(r["best_50ep"] for r in b_std) / len(b_std)) if b_std else None,
            "delta_3q_vs_2q_best": float(sub3["best_50ep"].mean() - sum(r["best_50ep"] for r in b_2q) / len(b_2q)) if b_2q else None,
            "delta_3q_vs_std_best": float(sub3["best_50ep"].mean() - sum(r["best_50ep"] for r in b_std) / len(b_std)) if b_std else None,
        })

    summary = {
        "n_3q_runs": len(df),
        "per_run": df.to_dict(orient="records"),
        "comparisons": comparisons,
    }
    with open(OUT / "ensemble3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {OUT / 'ensemble3_all_runs.csv'} ({len(df)} runs)")


if __name__ == "__main__":
    main()
