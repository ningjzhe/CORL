# Final Conclusion Draft (Algorithm Owner Only)

> This draft summarizes claims **supported by our experiments only**. It does **not** compare to CQL, DT, UA_CQL, or UGH_CQL until experiment-owner baselines are integrated (see `baseline_section_placeholder.md`).

---

## Primary conclusion

Our experiments show that **independent value-system ensemble policy extraction** (`ensemble_mean`) **consistently improves** over **standard single-value-system IQL actor extraction** (`standard_iql`) across **HalfCheetah, Hopper, and Walker2d** under the **best-checkpoint protocol**, on both **Best 50-episode** and **Final 50-episode** D4RL normalized scores:

| Environment | Best 50-ep Δ (ensemble − standard) | Final 50-ep Δ |
|-------------|-----------------------------------:|--------------:|
| HalfCheetah | +1.79 | +1.96 |
| Hopper | +21.24 | +16.97 |
| Walker2d | +5.06 | +2.24 |

This supports the main line:

> *Independent value-system ensemble policy extraction improves IQL actor extraction.*

---

## Ablation conclusion (IAD)

**Disagreement-aware weighting (IAD, λ=1.0) does not consistently outperform simple ensemble averaging.** On Hopper, IAD is worse than `ensemble_mean` at Best 50-ep (Δ = −4.95). On Walker2d, IAD Final 50-ep is substantially below ensemble (Δ = −5.86) despite a slightly higher Best on average driven by Rep B.

Against the **shuffled disagreement control**, IAD does not show a robust mechanism advantage on Final 50-ep (Walker2d Δ = −8.47).

Therefore:

> *IAD remains an ablation, not the proposed main method.*

---

## Protocol conclusion (best-checkpoint)

**Hopper and Walker2d exhibit actor overtraining** during 100k-step frozen-Q/V extraction: peak performance occurs before the final checkpoint, and Final 50-ep can be far below Best 50-ep (e.g., Hopper ensemble Rep B: 63.4 → 33.1). HalfCheetah shows minimal gap.

We therefore:

1. Save **`best_actor.pt`** and **`actor_final.pt`** plus step checkpoints  
2. Report **Best 50-ep** (primary) and **Final 50-ep** (stability)  
3. Motivate the best-checkpoint protocol from the earlier Hopper audit where best actors could not be re-evaluated under final-only saving  

> *Reporting both best and final checkpoints is necessary for honest evaluation of actor extraction on unstable environments.*

---

## Replicate / variance caveat

Results use **two actor replicates** (Rep A/B) with fixed init and batch streams per replicate. **Hopper shows high Rep A vs Rep B variance** (especially Rep B final scores). Main directional claims hold on both replicates for `ensemble_mean` vs `standard_iql`, but absolute scores should be reported with replicate spread, not over-interpreted as 3-seed value-system variance.

---

## Scope limits (explicit)

The following are **not** part of our completed evidence base for the main paper:

- seed2 value systems  
- 3-value `ensemble_mean_3q`  
- pair02 / pair12 ensemble ablations  
- medium-replay, new λ, 200k extension  
- External baselines (CQL, DT, UA, UGH)  

Scripts for seed2/3Q supplement exist but **experiments were not successfully executed** — do not include in main results.

---

## Suggested one-paragraph abstract snippet (our section)

We study policy extraction from frozen IQL value functions using two independently trained value systems. We compare standard single-system weighted behavioral cloning, ensemble averaging of advantages (`ensemble_mean`), and disagreement-aware variants (IAD). Across HalfCheetah, Hopper, and Walker2d, ensemble averaging consistently outperforms the single-system baseline under rigorous 50-episode evaluation of best and final actor checkpoints. IAD does not reliably beat ensemble averaging and is treated as an ablation. Hopper and Walker2d reveal substantial actor overtraining, motivating best-checkpoint reporting alongside final-checkpoint scores.

---

## Recommended positioning vs experiment owner section

| Section | Owner |
|---------|-------|
| ensemble_mean method & gains vs standard_iql | **Algorithm owner (us)** |
| CQL / DT / UA / UGH numbers & SOTA comparison | **Experiment owner** |
| Unified training speed / compute | **Experiment owner** (confirm concurrency assumptions) |
| Combined main table | **Joint** after `questions_for_experiment_owner.md` resolved |

---

## Ready for formal writing?

**Algorithm-owner section: yes.** Tables A–D, methods, protocol, ablations, and figures are complete from `best_checkpoint_protocol/`.

**Full paper with baselines: not yet** — blocked on experiment-owner data listed in `baseline_section_placeholder.md`.
