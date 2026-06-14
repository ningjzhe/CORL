# Experimental Protocol (Algorithm Owner)

## 1. Value systems

- **Environments:** halfcheetah-medium-v2, hopper-medium-v2, walker2d-medium-v2 (D4RL Gym MuJoCo v2)
- **Two independent IQL value systems per env:** seed0, seed1
- **Training:** 1,000,000 gradient steps, eval every 5,000, batch 256, β=3.0, τ=0.7, state normalization on, reward normalization off
- **Checkpoints:** `checkpoint_999999.pt` (qf, vf, actor state dicts)
- **Compatibility:** Pairwise forward-pass check before actor extraction (structure, preprocessing, finite advantages)

**Not included in main results:** seed2 value systems; 3-value `ensemble_mean_3q` (supplement not completed).

---

## 2. Frozen Q/V during actor extraction

- Load seed0 and/or seed1 Q/V from checkpoint; `requires_grad=False`
- Verify Q/V parameters unchanged at training start and end (max delta = 0)
- Actor is the **only** trainable module

---

## 3. Actor-only policy extraction

- **Updates:** 100,000
- **Batch size:** 256
- **Optimizer:** Adam, lr = 3e-4
- **Loss:** Advantage-weighted BC (method-specific weights; see `method_description.md`)
- **No** Q/V retraining, **no** per-step Q/V flattening

---

## 4. Replicates (fairness)

| Replicate | Actor init seed | Batch index seed | Shared init file | Batch indices file |
|-----------|----------------:|-----------------:|------------------|-------------------|
| Rep A | 12345 | 42 | `shared_actor_init_seed12345.pt` | `actor_training_batch_indices_seed42.npy` |
| Rep B | 54321 | 43 | `shared_actor_init_seed54321.pt` | `actor_training_batch_indices_seed43.npy` |

**Fixed across methods within a replicate:** same actor initialization and same sequence of training batches. Only the **weighting rule** differs (standard / ensemble / IAD / shuffled).

**N = 2** actor replicates per method per env (not 3 value seeds).

---

## 5. Mid-training evaluation

- **Frequency:** every 5,000 actor updates
- **Episodes:** 10
- **Action:** deterministic mean action (IQL `act()`)
- **Eval seed during training:** 0 (10-ep monitoring)

---

## 6. Best-checkpoint protocol

At each eval step:

1. Save `actor_step_{NNNNNN}.pt`
2. Track best 10-ep D4RL score; save `best_actor.pt` when improved

At end of 100k updates:

3. Save `actor_final.pt` (checkpoint at step 100,000)

**Why:** Hopper/Walker2d show **actor overtraining** — final checkpoint can be far below peak (Hopper ensemble Rep B: Best 50-ep 63.4 → Final 33.1). Prior **final-only** protocol could not re-evaluate best actor because `best_actor.pt` was not saved (documented in Hopper stability audit).

---

## 7. Final evaluation (reporting metrics)

After actor training completes, **fresh 50-episode** evaluation on:

| Checkpoint | Episodes | Eval seeds | Action |
|------------|---------:|------------|--------|
| `best_actor.pt` | 50 | 0–49 | deterministic mean |
| `actor_final.pt` | 50 | 0–49 | deterministic mean |

**Primary metric for claims:** **Best 50-ep** D4RL normalized score  
**Secondary (overtraining diagnostic):** **Final 50-ep** and **Best − Final gap**

---

## 8. Methods run (per env × replicate)

1. `standard_iql`
2. `ensemble_mean`
3. `iad_lambda_1.0`
4. `shuffled_iad_lambda_1.0`

**Total:** 3 envs × 4 methods × 2 replicates = **24 actor runs**

---

## 9. Output layout

| Content | Path |
|---------|------|
| Best-protocol runs | `/code/analysis/iad_iql/best_checkpoint_protocol/{env}_medium_v2/` |
| Aggregated tables | `/code/analysis/iad_iql/best_checkpoint_protocol/results/` |
| Global report | `/code/analysis/iad_iql/best_checkpoint_protocol/report/BEST_CHECKPOINT_PROTOCOL_REPORT.md` |

Legacy final-only pilots under `/code/analysis/iad_iql/results/` and env-specific dirs are **superseded** for main claims.

---

## 10. Excluded from main protocol / results

| Item | Status |
|------|--------|
| seed2 Q/V training | Not completed / not in main table |
| 3Q `ensemble_mean_3q` | Not completed |
| pair02 / pair12 actor runs | Not run |
| CQL, DT, UA_CQL, UGH_CQL | Experiment owner — see baseline placeholder |
| medium-replay, new λ, 200k extension | Not run |

---

## 11. Known artifacts

- Value training may end with `free(): invalid pointer` after successful 1M checkpoint save → marked `completed_with_cleanup_error`; checkpoints valid.
- Hopper high Rep A vs Rep B variance → report both replicates; do not collapse to single seed.
