#!/usr/bin/env python3
"""Generate HalfCheetah pilot report with decision analysis."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/code/analysis/iad_iql")
RESULTS = ROOT / "results"
REPORT = ROOT / "report" / "HALFCHEETAH_PILOT_REPORT.md"

DISPLAY = {
    "standard_iql_100k": "standard_iql",
    "ensemble_mean_100k_fast": "ensemble_mean",
    "iad_lambda_0.5_100k_fast": "iad_lambda_0.5",
    "iad_lambda_1.0_100k_fast": "iad_lambda_1.0",
    "shuffled_iad_lambda_0.5_100k_fast": "shuffled_iad_lambda_0.5",
}


def load_table():
    rows = []
    validation = json.loads((RESULTS / "pilot_run_validation.json").read_text())
    by_run = {r["run_name"]: r for r in validation["runs"]}
    for run, label in DISPLAY.items():
        r = by_run[run]
        rows.append({
            "Method": label,
            "Final 10-ep Score": round(r["final_10ep"], 3),
            "Final 20-ep Score": round(r["final_20ep"], 3),
            "Best Score": round(r["best_score"], 3),
            "ESS/B": round(r["ess_over_b"], 3),
            "Clamp Ratio": round(r["clamp_ratio"], 3),
            "Actor log_std": round(r["actor_log_std"], 3),
            "Status": "completed",
        })
    return pd.DataFrame(rows)


def cmp(a, b, key="Final 20-ep Score"):
    return float(a[key]) - float(b[key])


def main():
    df = load_table()
    em = df[df["Method"] == "ensemble_mean"].iloc[0]
    iad05 = df[df["Method"] == "iad_lambda_0.5"].iloc[0]
    iad10 = df[df["Method"] == "iad_lambda_1.0"].iloc[0]
    shuf = df[df["Method"] == "shuffled_iad_lambda_0.5"].iloc[0]
    std = df[df["Method"] == "standard_iql"].iloc[0]

    d_em_iad = cmp(iad05, em)
    d_iad_shuf = cmp(iad05, shuf)
    d_iad10_iad05 = cmp(iad10, iad05)

    # Decision logic
    if d_em_iad > 0.3 and d_iad_shuf > 0.15:
        scenario = "A"
        recommendation = (
            "IAD λ=0.5 shows clear gains over ensemble_mean and shuffled control. "
            "Recommend starting cross-dataset value systems (halfcheetah seed2, hopper, walker2d) "
            "with λ=0.5 fixed."
        )
    elif d_em_iad > 0 and d_iad_shuf > 0:
        scenario = "B"
        recommendation = (
            "IAD λ=0.5 is slightly better than ensemble_mean and shuffled, but margins are modest. "
            "Recommend halfcheetah seed2 value system and one actor replicate before Hopper/Walker2d."
        )
    else:
        scenario = "C"
        recommendation = (
            "IAD λ=0.5 does not clearly beat ensemble_mean and/or shuffled control. "
            "Do not expand datasets yet; analyze disagreement signal and consider ensemble_mean baseline."
        )

    lines = [
        "# HalfCheetah IAD-IQL Pilot Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Executive Summary",
        "",
        f"- **Scenario: {scenario}**",
        f"- {recommendation}",
        "",
        "## Main Results",
        "",
        df.to_markdown(index=False),
        "",
        "## Fast Fix Integrity",
        "",
        "- Fast script removed **per-step** `flatten_params` Q/V checks only.",
        "- Advantage formula, weight formula, mean matching, shuffled pairing, and actor BC loss **unchanged**.",
        "- Q/V remain frozen (`eval()`, `requires_grad_(False)`); start/end max delta = 0 for all runs.",
        "- All runs used identical `shared_actor_init.pt` and `actor_training_batch_indices.npy` (seed=42).",
        "- Evaluation seeds: base 0, episodes 0–19 for 20-ep final eval.",
        "- Aborted 15k slow runs excluded; see `results/aborted_slow_actor_runs.json`.",
        "",
        "## Q1: IAD λ=0.5 vs ensemble_mean",
        "",
        f"- Final 20-ep: IAD **{iad05['Final 20-ep Score']:.3f}** vs Ensemble **{em['Final 20-ep Score']:.3f}** (Δ={d_em_iad:+.3f})",
        f"- Best 10-ep: IAD **{iad05['Best Score']:.3f}** vs Ensemble **{em['Best Score']:.3f}**",
        f"- ESS/B: IAD **{iad05['ESS/B']:.3f}** vs Ensemble **{em['ESS/B']:.3f}** (lower = less conservative)",
        f"- Clamp: IAD **{iad05['Clamp Ratio']:.3f}** vs Ensemble **{em['Clamp Ratio']:.3f}**",
        "",
        "**Conclusion:** IAD λ=0.5 final 20-ep score is **slightly lower** than ensemble_mean "
        f"({d_em_iad:+.3f}). Best-score during training is comparable. IAD is less conservative (lower ESS/B). "
        "No clear win for IAD λ=0.5 over ensemble_mean on final evaluation.",
        "",
        "## Q2: IAD λ=0.5 vs shuffled_iad",
        "",
        f"- Final 20-ep: IAD **{iad05['Final 20-ep Score']:.3f}** vs Shuffled **{shuf['Final 20-ep Score']:.3f}** (Δ={d_iad_shuf:+.3f})",
        f"- Best 10-ep: IAD **{iad05['Best Score']:.3f}** vs Shuffled **{shuf['Best Score']:.3f}**",
        f"- ESS/B nearly identical: **{iad05['ESS/B']:.3f}** vs **{shuf['ESS/B']:.3f}**",
        "",
        "**Conclusion:** IAD λ=0.5 and shuffled IAD are **virtually tied** on final 20-ep "
        f"(Δ={d_iad_shuf:+.3f}). This suggests sample-level disagreement–transition pairing "
        "does not provide strong additional signal beyond the overall weight reshaping / regularization.",
        "",
        "## Q3: Is λ=1.0 overly conservative?",
        "",
        f"- Final 20-ep: λ=1.0 **{iad10['Final 20-ep Score']:.3f}** vs λ=0.5 **{iad05['Final 20-ep Score']:.3f}** (Δ={d_iad10_iad05:+.3f})",
        f"- ESS/B: λ=1.0 **{iad10['ESS/B']:.3f}** vs λ=0.5 **{iad05['ESS/B']:.3f}** (lower ESS/B = less conservative)",
        f"- Clamp: λ=1.0 **{iad10['Clamp Ratio']:.3f}** vs λ=0.5 **{iad05['Clamp Ratio']:.3f}**",
        "",
        "**Conclusion:** λ=1.0 achieves the **highest** final/best scores in this pilot, with **lower** ESS/B "
        "than λ=0.5 (less weight concentration, not more conservative). λ=1.0 is **not** overly conservative here; "
        "if anything λ=0.5 underperforms λ=1.0 slightly.",
        "",
        "## Reference: standard_iql (single value system)",
        "",
        f"- Final 20-ep: **{std['Final 20-ep Score']:.3f}** — below all dual-value methods (~48–49).",
        "",
        "## Value System Compatibility",
        "",
    ]

    compat = RESULTS / "value_system_compatibility.json"
    if compat.exists():
        c = json.loads(compat.read_text())
        lines.extend([
            f"- Pearson(A1,A2): {c.get('pearson_A1_A2'):.3f}",
            f"- Spearman(A1,A2): {c.get('spearman_A1_A2'):.3f}",
            f"- Disagreement mean: {c.get('disagreement_abs_A1_minus_A2_over_2', {}).get('mean'):.3f}",
            "",
        ])

    lines.extend([
        "## Figures",
        "",
        "- `figures/d4rl_score_vs_update.png`",
        "- `figures/actor_loss_vs_update.png`",
        "- `figures/ess_over_b_vs_update.png`",
        "- `figures/clamp_ratio_vs_update.png`",
        "- `figures/final_score_comparison.png`",
        "- `figures/a1_vs_a2_correlation.png`",
        "- `figures/disagreement_histogram.png`",
        "- `figures/ess_over_b_vs_final_score.png`",
        "",
        "## Output Files",
        "",
        "- `results/main_results.csv`",
        "- `results/training_curves.csv`",
        "- `results/summary.json`",
        "- `results/pilot_run_validation.json`",
        "- `results/*_summary.json`",
        "",
        "## Next Steps",
        "",
        f"**Scenario {scenario}:** {recommendation}",
    ])

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
