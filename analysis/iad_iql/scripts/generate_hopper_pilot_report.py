#!/usr/bin/env python3
"""Generate Hopper-medium-v2 pilot report with HalfCheetah comparison."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

HOPPER_METHODS = [
    "standard_iql",
    "ensemble_mean",
    "iad_lambda_1.0",
    "shuffled_iad_lambda_1.0",
]

HC_MEANS = {
    "standard_iql": 46.49,
    "ensemble_mean": 48.58,
    "iad_lambda_1.0": 48.95,
    "shuffled_iad_lambda_1.0": 48.96,
}


def score20(summary_path: Path) -> float:
    return float(json.loads(summary_path.read_text())["final_20ep_eval"]["D4RL_normalized_score_mean"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default="/code/analysis/iad_iql/hopper_medium_v2")
    args = parser.parse_args()

    root = Path(args.project_dir)
    results = root / "results"
    report_path = root / "report" / "HOPPER_PILOT_REPORT.md"

    cross = pd.read_csv(results / "main_results_replicates_summary.csv") if (
        results / "main_results_replicates_summary.csv"
    ).exists() else pd.DataFrame()

    compat = {}
    compat_path = results / "value_system_compatibility.json"
    if compat_path.exists():
        compat = json.loads(compat_path.read_text())

    lines = [
        "# Hopper-medium-v2 Pilot Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Value Systems",
        "",
    ]

    if compat:
        fe = compat.get("final_eval_d4rl_score", {})
        dis = compat.get("disagreement_abs_A1_minus_A2_over_2", {})
        lines += [
            f"- Compatible: **{compat.get('compatible')}**",
            f"- Pearson(A1,A2): {compat.get('pearson_A1_A2', float('nan')):.4f}",
            f"- Spearman(A1,A2): {compat.get('spearman_A1_A2', float('nan')):.4f}",
            f"- Disagreement mean / p90 / p99: "
            f"{dis.get('mean', float('nan')):.4f} / {dis.get('p90', float('nan')):.4f} / {dis.get('p99', float('nan')):.4f}",
            f"- Value seed0 final D4RL: {fe.get('seed0', float('nan')):.3f}",
            f"- Value seed1 final D4RL: {fe.get('seed1', float('nan')):.3f}",
            "",
        ]

    lines += ["## Cross-Replicate Actor Results (20-ep)", ""]
    if cross.empty:
        lines.append("_Actor results pending._")
    else:
        lines.append(cross.to_markdown(index=False))
        lines += ["", "## Replication vs HalfCheetah (20-ep mean ± std)", ""]
        for method in HOPPER_METHODS:
            row = cross[cross["Method"] == method]
            if row.empty or pd.isna(row.iloc[0].get("Mean 20-ep")):
                lines.append(f"- **{method}**: pending")
                continue
            hop_mean = float(row.iloc[0]["Mean 20-ep"])
            hc_mean = HC_MEANS.get(method, float("nan"))
            lines.append(
                f"- **{method}**: Hopper {hop_mean:.2f} vs HalfCheetah {hc_mean:.2f} (Δ {hop_mean - hc_mean:+.2f})"
            )

        lines += ["", "## Pilot Questions", ""]
        std = cross[cross["Method"] == "standard_iql"]
        em = cross[cross["Method"] == "ensemble_mean"]
        iad = cross[cross["Method"] == "iad_lambda_1.0"]
        shuf = cross[cross["Method"] == "shuffled_iad_lambda_1.0"]

        if not std.empty and not em.empty:
            lines.append(
                f"1. ensemble_mean vs standard_iql: "
                f"Δ={float(em.iloc[0]['Mean 20-ep']) - float(std.iloc[0]['Mean 20-ep']):+.2f}"
            )
        if not em.empty and not iad.empty:
            lines.append(
                f"2. iad_lambda_1.0 vs ensemble_mean: "
                f"Δ={float(iad.iloc[0]['Mean 20-ep']) - float(em.iloc[0]['Mean 20-ep']):+.2f}"
            )

        pairs = [
            ("Rep A", "iad_lambda_1.0_100k_fast_summary.json", "shuffled_iad_lambda_1.0_100k_fast_summary.json"),
            ("Rep B", "repB_iad_lambda_1.0_100k_fast_summary.json", "repB_shuffled_iad_lambda_1.0_100k_fast_summary.json"),
        ]
        for label, iad_name, shuf_name in pairs:
            ip = results / iad_name
            sp = results / shuf_name
            if ip.exists() and sp.exists():
                lines.append(f"3/4. {label} iad vs shuffled λ=1.0: Δ={score20(ip) - score20(sp):+.2f}")

        lines += ["", "## Walker2d Recommendation", ""]
        if not std.empty and not em.empty:
            d_em_std = float(em.iloc[0]["Mean 20-ep"]) - float(std.iloc[0]["Mean 20-ep"])
            iad_beats_both = all(
                score20(results / a) > score20(results / b)
                for a, b in [
                    ("iad_lambda_1.0_100k_fast_summary.json", "shuffled_iad_lambda_1.0_100k_fast_summary.json"),
                    ("repB_iad_lambda_1.0_100k_fast_summary.json", "repB_shuffled_iad_lambda_1.0_100k_fast_summary.json"),
                ]
                if (results / a).exists() and (results / b).exists()
            )
            if d_em_std > 1.0:
                rec = "ensemble_mean >> standard on Hopper. Recommend Walker2d with ensemble_mean main line."
            elif d_em_std > 0 and not iad_beats_both:
                rec = (
                    "Dual-system gain replicated; IAD vs shuffled unstable. "
                    "Proceed to Walker2d; keep ensemble_mean as main line."
                )
            else:
                rec = "Mixed signals. Review before Walker2d; ensemble_mean remains default main line."
            lines.append(rec)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
