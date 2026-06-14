#!/usr/bin/env python3
"""Generate overnight markdown report from pipeline status and results."""
import argparse
import json
from datetime import datetime
from pathlib import Path


def load_json(path: Path, default=None):
    if not path.exists():
        return default if default is not None else {}
    with open(path) as f:
        return json.load(f)


def fmt(v, nd=3):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if v != v:
            return "N/A"
        return f"{v:.{nd}f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default="/code/analysis/iad_iql/results/overnight_status.json")
    parser.add_argument("--output", default="/code/analysis/iad_iql/report/OVERNIGHT_REPORT.md")
    args = parser.parse_args()

    status = load_json(Path(args.status), {})
    results_dir = Path("/code/analysis/iad_iql/results")
    compat = load_json(results_dir / "value_system_compatibility.json")
    fairness = load_json(results_dir / "fairness_checks.json")
    summary = load_json(results_dir / "summary.json", [])

    methods = [
        "standard_iql_100k",
        "ensemble_mean_100k",
        "iad_lambda_0.5_100k",
        "iad_lambda_1.0_100k",
        "shuffled_iad_lambda_0.5_100k",
    ]
    summaries = {m: load_json(results_dir / f"{m}_summary.json") for m in methods}

    def score_pair(name):
        s = summaries.get(name, {})
        if not s:
            return "N/A", "N/A"
        return fmt(s.get("final_d4rl_score")), fmt(s.get("best_d4rl_score"))

    std_final, std_best = score_pair("standard_iql_100k")
    em_final, em_best = score_pair("ensemble_mean_100k")
    iad05_final, iad05_best = score_pair("iad_lambda_0.5_100k")
    iad10_final, iad10_best = score_pair("iad_lambda_1.0_100k")
    shuf_final, shuf_best = score_pair("shuffled_iad_lambda_0.5_100k")

    def cmp_better(a, b):
        try:
            return float(a) > float(b)
        except (TypeError, ValueError):
            return None

    iad05_vs_em = cmp_better(iad05_final, em_final)
    iad_vs_shuf = cmp_better(iad05_final, shuf_final)

    conservative = []
    for name in methods:
        s = summaries.get(name, {})
        if s and s.get("clamp_ratio", 0) > 0.3:
            conservative.append(f"{name}: clamp_ratio={fmt(s.get('clamp_ratio'))}, ESS/B={fmt(s.get('ess_over_batch_size'))}")

    lines = [
        "# IAD-IQL Overnight Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Pipeline status",
        "",
        f"- Start: {status.get('pipeline_start_time', 'N/A')}",
        f"- End: {status.get('pipeline_end_time', 'N/A')}",
        f"- standard_iql: **{status.get('standard_status', 'N/A')}**",
        f"- value_seed1: **{status.get('value_seed1_status', 'N/A')}**",
        f"- compatibility: **{status.get('compatibility_status', 'N/A')}**",
        f"- ensemble_mean: **{status.get('ensemble_mean_status', 'N/A')}**",
        f"- iad_0.5: **{status.get('iad_0_5_status', 'N/A')}**",
        f"- iad_1.0: **{status.get('iad_1_0_status', 'N/A')}**",
        f"- shuffled: **{status.get('shuffled_status', 'N/A')}**",
        f"- summary: **{status.get('summary_status', 'N/A')}**",
        "",
        "## 1. Standard IQL actor extraction",
        "",
        f"- Status: {status.get('standard_status', 'N/A')}",
        f"- Final D4RL (10-ep): {std_final}",
        f"- Best D4RL (10-ep): {std_best}",
        "",
        "## 2. Value System seed 1",
        "",
        f"- Status: {status.get('value_seed1_status', 'N/A')}",
        "",
        "## 3. Value checkpoint compatibility",
        "",
        f"- Compatible: {compat.get('compatible', 'N/A')}",
        f"- Pearson(A1,A2): {fmt(compat.get('pearson_A1_A2'))}",
        f"- Spearman(A1,A2): {fmt(compat.get('spearman_A1_A2'))}",
        "",
        "### Disagreement distribution",
        "",
        "```json",
        json.dumps(compat.get("disagreement_abs_A1_minus_A2_over_2", {}), indent=2),
        "```",
        "",
        "## 4. Remaining actor extractions",
        "",
        "| Method | Status | Final | Best |",
        "|--------|--------|-------|------|",
    ]

    status_map = {
        "ensemble_mean_100k": status.get("ensemble_mean_status"),
        "iad_lambda_0.5_100k": status.get("iad_0_5_status"),
        "iad_lambda_1.0_100k": status.get("iad_1_0_status"),
        "shuffled_iad_lambda_0.5_100k": status.get("shuffled_status"),
    }
    for m, st in status_map.items():
        fin, best = score_pair(m)
        lines.append(f"| {m} | {st or 'N/A'} | {fin} | {best} |")

    lines.extend([
        "",
        "## 5. Five-method scores",
        "",
        "| Method | Final | Best | ESS/B | Clamp |",
        "|--------|-------|------|-------|-------|",
    ])
    for m in methods:
        s = summaries.get(m, {})
        lines.append(
            f"| {m} | {fmt(s.get('final_d4rl_score'))} | {fmt(s.get('best_d4rl_score'))} | "
            f"{fmt(s.get('ess_over_batch_size'))} | {fmt(s.get('clamp_ratio'))} |"
        )

    lines.extend([
        "",
        "## 6. Key comparisons",
        "",
        f"- IAD λ=0.5 vs ensemble mean (final): {'IAD better' if iad05_vs_em else 'ensemble better or inconclusive' if iad05_vs_em is False else 'inconclusive'}",
        f"- IAD λ=0.5 vs shuffled IAD (final): {'IAD better' if iad_vs_shuf else 'shuffled better or inconclusive' if iad_vs_shuf is False else 'inconclusive'}",
        "",
        "## 7. Over-conservatism signals",
        "",
    ])
    if conservative:
        lines.extend([f"- {c}" for c in conservative])
    else:
        lines.append("- No high clamp_ratio (>0.3) flagged at final step.")

    lines.extend([
        "",
        "## 8. Failures and anomalies",
        "",
        "```json",
        json.dumps(status.get("errors", []), indent=2),
        "```",
        "",
        "## 9. Recommended next steps",
        "",
        "1. Review compatibility JSON and disagreement plots before cross-dataset runs.",
        "2. If all five HalfCheetah actors completed, compare 20-ep final evals.",
        "3. Only after pilot sign-off: start Hopper/Walker2d value systems.",
        "4. Do not extend to 200k unless all curves still improving.",
        "",
        "## 10. Output paths",
        "",
        "- Status: `/code/analysis/iad_iql/results/overnight_status.json`",
        "- Initial status: `/code/analysis/iad_iql/results/overnight_initial_status.txt`",
        "- Pipeline log: `/code/analysis/iad_iql/logs/overnight_pipeline.log`",
        "- Compatibility: `/code/analysis/iad_iql/results/value_system_compatibility.json`",
        "- Fairness: `/code/analysis/iad_iql/results/fairness_checks.json`",
        "- Summaries: `/code/analysis/iad_iql/results/*_summary.json`",
        "- Main results: `/code/analysis/iad_iql/results/main_results.csv`",
        "- Figures: `/code/analysis/iad_iql/figures/`",
    ])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
