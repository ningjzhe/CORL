#!/usr/bin/env python3
"""Generate SEED2_AND_ENSEMBLE3_SUPPLEMENT_REPORT.md"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/code/analysis/iad_iql/seed2_supplement")
OUT = ROOT / "report" / "SEED2_AND_ENSEMBLE3_SUPPLEMENT_REPORT.md"


def load_json(p):
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    results = ROOT / "results"
    lines = [
        "# Seed2 + Ensemble3 Supplement Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    for env in ["halfcheetah", "hopper", "walker2d"]:
        q = load_json(results / f"{env}_seed2_value_quality.json")
        lines += [f"## {env} seed2 value quality", "",
                  f"- status: **{q.get('status', 'pending')}**",
                  f"- final D4RL: {q.get('seed2_final_D4RL', 'NA')}", ""]

    compat_lines = ["## Pair compatibility", ""]
    for env in ["halfcheetah", "hopper", "walker2d"]:
        c = load_json(results / f"{env}_pair_compatibility.json")
        if not c:
            continue
        compat_lines.append(f"### {env}")
        for pair in ("pair01", "pair02", "pair12"):
            p = c.get(pair, {})
            compat_lines.append(
                f"- {pair}: compatible={p.get('compatible')} pearson={p.get('pearson', 'NA'):.4f}"
                if isinstance(p.get("pearson"), float) else f"- {pair}: pending"
            )
        compat_lines.append("")
    lines += compat_lines

    csv_path = results / "ensemble3_all_runs.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        lines += ["## 3Q actor results", "", df.to_markdown(index=False), ""]

    comp = load_json(results / "ensemble3_summary.json")
    if comp.get("comparisons"):
        lines += ["## Comparisons vs 2Q / standard", ""]
        for c in comp["comparisons"]:
            lines.append(
                f"- **{c['env']}**: 3Q vs 2Q Best Δ={c.get('delta_3q_vs_2q_best', 'NA'):+.2f}, "
                f"3Q vs std Best Δ={c.get('delta_3q_vs_std_best', 'NA'):+.2f}, "
                f"3Q gap={c.get('3q_gap_mean', 'NA')}"
            )
        lines.append("")

    hc = load_json(ROOT / "logs" / "health_check_30min.json")
    if hc:
        lines += ["## 30-min health check", "",
                  f"- recommendation: **{hc.get('recommendation')}**", ""]
        for j in hc.get("jobs", []):
            lines.append(f"- {j['name']}: step={j['step']} sps={j.get('interval_steps_per_sec', j['steps_per_sec'])} gpu={j['gpu']}")
        lines.append("")

    lines += [
        "## Answers",
        "",
        "1. **seed2 Q/V healthy?** See per-env value quality above.",
        "2. **Pair compatibility?** See pair tables; 3Q gate requires pair02 & pair12 compatible.",
        "3. **3Q vs standard?** See comparison deltas.",
        "4. **3Q vs 2Q ensemble?** See comparison deltas.",
        "5. **3Q gap vs 2Q?** Compare gap means in comparisons.",
        "6. **Rep A/B stability?** See per-run table.",
        "7. **3Q in main table?** Recommend supplement only unless 3Q consistently beats 2Q.",
        "8. **Main method?** Keep **ensemble_mean (2Q)** as primary; **ensemble_mean_3q** as supplement.",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
