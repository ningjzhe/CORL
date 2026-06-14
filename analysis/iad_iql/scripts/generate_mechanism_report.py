#!/usr/bin/env python3
"""Generate HalfCheetah mechanism validation report (RepA + RepB)."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/code/analysis/iad_iql")
RESULTS = ROOT / "results"
REPORT = ROOT / "report" / "HALFCHEETAH_MECHANISM_REPORT.md"


def load_summary(stem):
    p = RESULTS / f"{stem}_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def score20(stem):
    s = load_summary(stem)
    if not s:
        return float("nan")
    return s["final_20ep_eval"]["D4RL_normalized_score_mean"]


def main():
    rep_a = pd.read_csv(RESULTS / "main_results_repA.csv")
    rep_b = pd.read_csv(RESULTS / "main_results_repB.csv") if (RESULTS / "main_results_repB.csv").exists() else pd.DataFrame()
    cross = pd.read_csv(RESULTS / "main_results_replicates_summary.csv") if (RESULTS / "main_results_replicates_summary.csv").exists() else pd.DataFrame()

    iad_a = score20("iad_lambda_1.0_100k_fast")
    shuf_a = score20("shuffled_iad_lambda_1.0_100k_fast")
    iad_b = score20("repB_iad_lambda_1.0_100k_fast")
    shuf_b = score20("repB_shuffled_iad_lambda_1.0_100k_fast")
    std_b = score20("repB_standard_iql_100k_fast")
    em_b = score20("repB_ensemble_mean_100k_fast")

    d_a = iad_a - shuf_a if pd.notna(iad_a) and pd.notna(shuf_a) else float("nan")
    d_b = iad_b - shuf_b if pd.notna(iad_b) and pd.notna(shuf_b) else float("nan")

    def verdict(delta, tol=0.15):
        if pd.isna(delta):
            return "incomplete"
        if delta > tol:
            return "IAD better — sample-level pairing may help"
        if delta < -tol:
            return "shuffled better — disagreement proxy may penalize wrong samples"
        return "similar — gain likely from weight reshaping, not sample-level pairing"

    v_a = verdict(d_a)
    v_b = verdict(d_b)

    # Scenario A/B/C across replicates
    both_iad_win = (d_a > 0.15 and d_b > 0.15)
    both_similar = (abs(d_a) <= 0.15 and abs(d_b) <= 0.15)
    any_shuf_win = (d_a < -0.15 or d_b < -0.15)

    if both_iad_win:
        scenario = "A"
        main_method = "IAD-IQL λ=1.0"
        expand = "Yes — consider cross-dataset extension after documentation."
    elif both_similar or (abs(d_a) <= 0.15 or abs(d_b) <= 0.15):
        scenario = "B"
        main_method = "ensemble_mean (IAD λ=1.0 as ablation)"
        expand = "No — emphasize independent value-system ensemble policy extraction."
    else:
        scenario = "C"
        main_method = "ensemble_mean"
        expand = "No — do not extend IAD; revisit disagreement proxy."

    if any_shuf_win and not both_iad_win:
        scenario = "C"
        main_method = "ensemble_mean"
        expand = "No — shuffled control matches or beats real IAD."

    lines = [
        "# HalfCheetah Mechanism Validation Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Replicate A Results",
        "",
        rep_a.to_markdown(index=False),
        "",
        "## Replicate B Results",
        "",
        rep_b.to_markdown(index=False) if not rep_b.empty else "_Pending_",
        "",
        "## Cross-Replicate Summary",
        "",
        cross.to_markdown(index=False) if not cross.empty else "_Pending_",
        "",
        "## Key Questions",
        "",
        "### 1. Rep A: iad_lambda_1.0 vs shuffled_iad_lambda_1.0",
        f"- IAD λ=1.0 20-ep: **{iad_a:.3f}**",
        f"- Shuffled λ=1.0 20-ep: **{shuf_a:.3f}**",
        f"- Δ = **{d_a:+.3f}**",
        f"- **{v_a}**",
        "",
        "### 2. Rep B: dual-value methods vs standard_iql",
        f"- repB standard 20-ep: **{std_b:.3f}**",
        f"- repB ensemble_mean 20-ep: **{em_b:.3f}**",
        f"- repB iad λ=1.0 20-ep: **{iad_b:.3f}**",
        f"- Dual-value methods remain above standard if scores ≳ 48 vs ~46–47.",
        "",
        "### 3. Rep B: iad_lambda_1.0 vs shuffled_iad_lambda_1.0",
        f"- Δ = **{d_b:+.3f}**",
        f"- **{v_b}**",
        "",
        "### 4. Consistency across replicates",
        f"- Rep A verdict: {v_a}",
        f"- Rep B verdict: {v_b}",
        f"- Consistent: **{'yes' if (d_a * d_b > 0 or (abs(d_a) <= 0.15 and abs(d_b) <= 0.15)) else 'mixed/incomplete'}**",
        "",
        "### 5. Recommended main method",
        f"- **Scenario {scenario}:** {main_method}",
        "",
        "### 6. Sample-level disagreement evidence",
        f"- {'Some support' if both_iad_win else 'Insufficient' if both_similar else 'Against IAD' if any_shuf_win else 'Inconclusive'}",
        "",
        "### 7. Extend to Hopper/Walker2d?",
        f"- **{expand}**",
        "",
        "## Output Files",
        "",
        "- `results/main_results_repA.csv`",
        "- `results/main_results_repB.csv`",
        "- `results/main_results_replicates_summary.csv`",
        "- `results/replicate_summary.json`",
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
