#!/usr/bin/env python3
"""Generate WALKER2D_BEST_PROTOCOL_REPORT.md"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/code/analysis/iad_iql/best_checkpoint_protocol/walker2d_medium_v2")
WALKER_VALUE = Path("/code/analysis/iad_iql/walker2d_medium_v2")
OUT = ROOT / "report" / "WALKER2D_BEST_PROTOCOL_REPORT.md"
GLOBAL = Path("/code/analysis/iad_iql/best_checkpoint_protocol/results")


def load_compat():
    p = WALKER_VALUE / "results" / "value_system_compatibility.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def main():
    results = ROOT / "results"
    summaries = sorted(results.glob("*_summary.json"))
    rows = []
    for p in summaries:
        s = json.loads(p.read_text())
        if s.get("status") != "completed" or "smoke" in s["run_name"]:
            continue
        rows.append(s)

    if not rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("# Walker2d Best-Checkpoint Report\n\nNo completed runs yet.\n")
        print(f"Wrote placeholder {OUT}")
        return

    df = pd.DataFrame([{
        "Rep": s["replicate"],
        "Method": s["method"],
        "Best Step": s["best_step"],
        "Best 50-ep": s["best_50ep_score_mean"],
        "Final 50-ep": s["final_50ep_score_mean"],
        "Gap": s["best_final_gap_50ep"],
    } for s in rows])

    compat = load_compat()
    comp = pd.read_csv(GLOBAL / "best_protocol_comparison.csv") if (GLOBAL / "best_protocol_comparison.csv").exists() else pd.DataFrame()
    wcomp = comp[comp["Env"] == "walker2d"] if not comp.empty else pd.DataFrame()
    global_comp = pd.read_csv(GLOBAL / "best_protocol_comparison.csv") if (GLOBAL / "best_protocol_comparison.csv").exists() else pd.DataFrame()

    def delta(label):
        sub = wcomp[wcomp["Comparison"] == label]
        if sub.empty:
            return float("nan"), float("nan")
        return float(sub["Best 50-ep Delta"].iloc[0]), float(sub["Final 50-ep Delta"].iloc[0])

    em_bd, em_fd = delta("ensemble_mean - standard_iql")
    iad_em_bd, iad_em_fd = delta("iad_lambda_1.0 - ensemble_mean")
    iad_sh_bd, iad_sh_fd = delta("iad_lambda_1.0 - shuffled_iad_lambda_1.0")

    mean_gap = df["Gap"].mean()
    rep_a = df[df["Rep"] == "A"]
    rep_b = df[df["Rep"] == "B"]
    rep_consistent = (
        (rep_a.set_index("Method")["Best 50-ep"] - rep_b.set_index("Method")["Best 50-ep"]).abs().mean()
        if len(rep_a) == len(rep_b) else float("nan")
    )

    # Cross-env ensemble vs standard
    ens_std = global_comp[global_comp["Comparison"] == "ensemble_mean - standard_iql"]
    all_positive_best = bool((ens_std["Best 50-ep Delta"] > 0).all()) if not ens_std.empty else False

    lines = [
        "# Walker2d Best-Checkpoint Protocol Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "> Protocol: save `best_actor.pt` + step checkpoints + 50-ep re-eval on best and final.",
        "",
        "## Value systems",
        "",
    ]

    if compat:
        lines += [
            f"- **compatible**: {compat.get('compatible')}",
            f"- Pearson(A1,A2): {compat.get('pearson_A1_A2', 'NA'):.4f}" if isinstance(compat.get("pearson_A1_A2"), (int, float)) else "",
            f"- Spearman(A1,A2): {compat.get('spearman_A1_A2', 'NA'):.4f}" if isinstance(compat.get("spearman_A1_A2"), (int, float)) else "",
            f"- Value seed0 final D4RL: {compat.get('final_eval_d4rl_score', {}).get('seed0', 'NA')}",
            f"- Value seed1 final D4RL: {compat.get('final_eval_d4rl_score', {}).get('seed1', 'NA')}",
            f"- state_dim={compat.get('state_dim')} action_dim={compat.get('action_dim')}",
            "",
        ]

    lines += [
        "## Table: Per-run Best/Final 50-ep",
        "",
        df.to_markdown(index=False),
        "",
        "## Answers",
        "",
        "### 1. Actor overtraining on Walker2d?",
        f"- Mean best–final gap (50-ep): **{mean_gap:.2f}**",
        f"- {'Yes — significant overtraining observed' if mean_gap > 2 else 'Minimal overtraining (gap ≤ 2)'}",
        "",
        "### 2. Best vs final checkpoint gap",
        f"- Mean gap: **{mean_gap:.2f}**; per-method gaps in table above.",
        "",
        "### 3. ensemble_mean vs standard_iql",
        f"- Best 50-ep Δ: **{em_bd:+.2f}**; Final 50-ep Δ: **{em_fd:+.2f}**",
        f"- Supports main line: **{'Yes' if em_bd > 0 else 'No'}** (Best 50-ep)",
        "",
        "### 4. IAD λ=1.0 vs ensemble_mean",
        f"- Best 50-ep Δ: **{iad_em_bd:+.2f}**; Final 50-ep Δ: **{iad_em_fd:+.2f}**",
        f"- IAD as main method: **{'Not supported' if iad_em_bd <= 0 else 'Unclear — needs replicate stability'}**",
        "",
        "### 5. IAD λ=1.0 vs shuffled λ=1.0",
        f"- Best 50-ep Δ: **{iad_sh_bd:+.2f}**; Final 50-ep Δ: **{iad_sh_fd:+.2f}**",
        "",
        "### 6. Rep A vs Rep B consistency",
        f"- Mean |Best 50-ep(A) − Best 50-ep(B)| across methods: **{rep_consistent:.2f}**",
        "",
        "### 7. Walker2d vs HalfCheetah/Hopper conclusions",
        "Compare Table 3 in global report. Walker2d ensemble vs standard delta should align with prior envs for cross-env support.",
        "",
        "### 8. Three-env merged main method",
        f"- ensemble_mean Best 50-ep delta positive in all envs: **{all_positive_best}**",
        "- Recommended main method: **ensemble_mean** (if all deltas positive)",
        "",
        "### 9. Recommend seed2?",
        "- **Not yet.** Two replicates (A/B) sufficient for pilot; seed2 only if Rep A/B diverge strongly or for final paper runs.",
        "",
        "### 10. Ready for final report stage?",
        "- **Yes**, if 8/8 actor runs completed and compatibility passed.",
        "",
        "## Output paths",
        "",
        f"- Results: `{ROOT / 'results'}`",
        f"- Figures: `{ROOT / 'figures'}`",
        f"- Global summary: `{GLOBAL / 'best_protocol_summary.json'}`",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
