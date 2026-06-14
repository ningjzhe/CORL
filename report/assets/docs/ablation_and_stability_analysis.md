# Ablation and Stability Analysis (Algorithm Owner)

## 1. ensemble_mean vs standard_iql

**Claim supported:** Independent two-value-system ensemble improves actor extraction over single seed0 value system.

| Env | Best 50-ep Δ | Final 50-ep Δ | Rep A ensemble Best | Rep A standard Best |
|-----|-------------:|--------------:|--------------------:|--------------------:|
| HalfCheetah | +1.79 | +1.96 | 48.94 | 47.13 |
| Hopper | +21.24 | +16.97 | **71.72** | 52.90 |
| Walker2d | +5.06 | +2.24 | **89.69** | 80.85 |

Hopper shows the largest absolute gain — also the env with strongest overtraining, so Best 50-ep comparison is especially important.

---

## 2. IAD λ=1.0 vs ensemble_mean (ablation)

| Env | Best Δ (IAD − ensemble) | Final Δ | Verdict |
|-----|------------------------:|--------:|---------|
| HalfCheetah | +0.05 | +0.24 | ~tie; shuffled slightly higher Best mean |
| Hopper | **−4.95** | −0.74 | IAD does not beat ensemble at Best |
| Walker2d | +0.76 | **−5.86** | IAD Best slightly higher; **Final much worse** |

**Rep-level instability:**

- Hopper Rep A: IAD Best 63.7 vs ensemble 71.7
- Walker2d Rep A: IAD Final 66.7 vs ensemble Final 84.2 (gap 18.9 on IAD)

**Conclusion:** Disagreement penalty does not provide consistent improvement over simple advantage averaging. Keep IAD as ablation only.

---

## 3. IAD vs shuffled_iad (mechanism control)

| Env | Best Δ (IAD − shuffled) | Final Δ |
|-----|------------------------:|--------:|
| HalfCheetah | −0.19 | −0.09 |
| Hopper | +11.36 | +7.94 |
| Walker2d | +0.51 | **−8.47** |

Hopper shows positive IAD − shuffled at aggregate level, but **shuffled Rep A Best 68.9 vs IAD 63.7** — high replicate noise. Walker2d Final strongly favors shuffled over IAD.

**Conclusion:** No clean evidence that structured disagreement penalty beats its shuffled control on Final 50-ep. Mechanism claim weak.

---

## 4. HalfCheetah stability

- All methods: Best ≈ Final (gaps ≤ 0.45)
- Method ranking by Best mean: shuffled (49.08) ≈ IAD (48.89) ≈ ensemble (48.84) > standard (47.05)
- **Main claim still holds:** ensemble_mean > standard_iql with comfortable margin
- Differences among ensemble/IAD/shuffled are **small (~1 point)** — do not promote IAD over ensemble on HC alone

---

## 5. Hopper stability (critical)

### Overtraining

- **8/8** runs in prior final-only audit showed late-stage collapse
- Best-protocol gaps: mean **17.4** across all Hopper runs; max **30.4** (ensemble Rep B)
- ensemble Rep A: Best 71.7 → Final 59.2 (gap 12.5)
- ensemble Rep B: Best 63.4 → Final **33.1** (gap 30.4)

### Rep A vs Rep B

| Method | Rep A Best 50 | Rep B Best 50 | Rep A Final 50 | Rep B Final 50 |
|--------|-------------:|--------------:|---------------:|---------------:|
| ensemble_mean | 71.72 | 63.45 | 59.23 | 33.08 |
| standard_iql | 52.90 | 39.79 | 34.97 | 23.40 |

Rep B is lower at **both** best and final — not only a checkpoint-selection artifact.

### Direction of main claim

ensemble_mean > standard_iql holds on **both** replicates at Best and Final 50-ep.

---

## 6. Walker2d stability

- Mean gap **3.18** (moderate overtraining)
- ensemble_mean: Rep A gap 5.5, Rep B gap 0.2 — Rep B very stable
- IAD Rep A: large gap (18.9) — **avoid promoting IAD**
- shuffled and IAD Rep B: gap 0

Main claim: ensemble Best 86.8 vs standard 81.7 (+5.1); Final +2.2.

---

## 7. Best-checkpoint protocol necessity

| Evidence | Source |
|----------|--------|
| No `best_actor.pt` in final-only Hopper runs | `HOPPER_STABILITY_AUDIT.md` |
| Cannot 50-ep re-eval CSV-best without step checkpoints | Same audit |
| Large best–final gaps under best-protocol | Table D in `main_results_tables_our_side.md` |
| Protocol saves step ckpts + best + final + 50-ep re-eval | `train_frozen_iad_actor_best.py` |

**Recommendation for paper:** Report **Best 50-ep** as primary; **Final 50-ep** and gap as stability diagnostic; cite Hopper as motivating example.

---

## 8. What not to over-claim

- Do **not** claim IAD beats ensemble as main method
- Do **not** use Hopper Rep B alone to dismiss ensemble (Rep A strongly supports main line)
- Do **not** merge our 2-replicate std with experiment owner 3-seed tables without relabeling N
- seed2 / 3Q supplement: **not run successfully** — no ablation on third value system in main paper

---

## 9. Suggested ablation figure narrative

1. **Figure:** score vs step (Hopper/Walker2d) — show peak then decline  
   Path: `best_checkpoint_protocol/figures/score_vs_step_{hopper,walker2d}.png`
2. **Figure:** best–final gap by env/method  
   Path: `best_checkpoint_protocol/figures/best_final_gap_by_env_method.png`
3. **Figure:** ensemble vs standard delta across envs  
   Path: `best_checkpoint_protocol/figures/ensemble_vs_standard_delta.png`
