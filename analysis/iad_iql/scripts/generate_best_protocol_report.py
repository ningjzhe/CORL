#!/usr/bin/env python3
"""Generate BEST_CHECKPOINT_PROTOCOL_REPORT.md"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/code/analysis/iad_iql/best_checkpoint_protocol")
OUT = ROOT / "report" / "BEST_CHECKPOINT_PROTOCOL_REPORT.md"


def main():
    all_runs = pd.read_csv(ROOT / "results" / "best_protocol_all_runs.csv")
    env_sum = pd.read_csv(ROOT / "results" / "best_protocol_env_summary.csv")
    comp = pd.read_csv(ROOT / "results" / "best_protocol_comparison.csv")

    lines = [
        "# Best-Checkpoint Protocol Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "> Previous protocol: final-only actor extraction. "
        "New protocol: save `best_actor.pt` + step checkpoints + 50-ep re-eval.",
        "",
        "## Table 1: Per-run results",
        "",
        all_runs.to_markdown(index=False),
        "",
        "## Table 2: Mean ± std by env/method",
        "",
        env_sum.to_markdown(index=False),
        "",
        "## Table 3: Comparisons (50-ep)",
        "",
        comp.to_markdown(index=False),
        "",
        "## Answers",
        "",
    ]

    def delta(comp_label):
        row = comp[comp["Comparison"] == comp_label]
        if row.empty:
            return {}
        return {
            r["Env"]: (r["Best 50-ep Delta"], r["Final 50-ep Delta"])
            for _, r in row.iterrows()
        }

    d_em_std = delta("ensemble_mean - standard_iql")
    d_iad_em = delta("iad_lambda_1.0 - ensemble_mean")
    d_iad_sh = delta("iad_lambda_1.0 - shuffled_iad_lambda_1.0")

    hopper_gap = all_runs[all_runs["Env"] == "hopper"]["Gap"].mean() if "hopper" in all_runs["Env"].values else float("nan")
    walker_gap = all_runs[all_runs["Env"] == "walker2d"]["Gap"].mean() if "walker2d" in all_runs["Env"].values else float("nan")

    lines += [
        "### 1. Best vs final difference",
        f"- Mean 50-ep gap (best - final) across all runs: **{all_runs['Gap'].mean():.2f}**",
        f"- Hopper mean gap: **{hopper_gap:.2f}**",
        f"- Walker2d mean gap: **{walker_gap:.2f}**" if walker_gap == walker_gap else "",
        "",
        "### 2. Overtraining mitigated?",
        "Best checkpoint protocol captures peak performance; report both Best and Final 50-ep.",
        "",
        "### 3. ensemble_mean vs standard (Best & Final 50-ep)",
    ]
    for env, (bd, fd) in d_em_std.items():
        lines.append(f"- **{env}**: Best Δ={bd:+.2f}, Final Δ={fd:+.2f}")

    lines += ["", "### 4. IAD vs ensemble", ""]
    for env, (bd, fd) in d_iad_em.items():
        lines.append(f"- **{env}**: Best Δ={bd:+.2f}, Final Δ={fd:+.2f}")

    lines += ["", "### 5. IAD vs shuffled", ""]
    for env, (bd, fd) in d_iad_sh.items():
        lines.append(f"- **{env}**: Best Δ={bd:+.2f}, Final Δ={fd:+.2f}")

    lines += [
        "",
        "### 6–7. Cross-env consistency & main method",
        "See Table 2/3. **ensemble_mean** remains recommended main line if Best 50-ep deltas are positive across all three envs.",
        "",
        "### 8. Walker2d",
        "Walker2d pilot completed under best-checkpoint protocol. See `walker2d_medium_v2/report/WALKER2D_BEST_PROTOCOL_REPORT.md`.",
        "",
        "### 9. Report both Best and Final?",
        "**Yes.** Best D4RL is the primary metric for the PI; Final documents overtraining.",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
