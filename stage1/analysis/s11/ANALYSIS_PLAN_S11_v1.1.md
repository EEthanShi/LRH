# S1.1 analysis plan v1.1 (PREREG v1.1; descriptive mode)

**Status: FROZEN.** The sha256 is recorded in `STATUS.md` by role s11-select. Written 2026-10-08, after the v1.1 estimator re-selection and before any COCO feature array was read. The S1.1 role may copy this file verbatim to `analysis/s11/` (same sha256).

## What this file is

This file amends the draft `analysis/s11/ANALYSIS_PLAN_S11.md`, as `prereg/LEAD_DECISIONS_v1.1.md` D4.8 requires. The draft is v0.9, sha256 `62fa39c637f9e604a43a08c801d4fc92570b52ec21ce9aa7cebd0fc2e3b659bb`, and is unchanged.

- Every pinned decision of the draft (O1–O12) is adopted **except where this file overrides it**. Overrides are marked **[v1.1]**.
- Authority, highest first:
  1. `prereg/LEAD_DECISIONS_v1.1.md` (sha256 `eaebd687…`);
  2. `hrhv2/PREDICTOR_CHOICE_v1.1.md` (sha256 `f83b13a9…`);
  3. this file;
  4. the draft.

## V0. Lead decisions implemented here

| id | decision | where |
|---|---|---|
| D2 | exact expected discrepancy for k ≤ 3 | V3 |
| D3 | exact ties by the SPEC rule; near-tie fraction reported | O11 (unchanged) |
| D4.1–D4.6 | anchor-local re-selection on plants + S_sel + null gate | `PREDICTOR_CHOICE_v1.1.md` |
| **D4.7** | **no eligible estimator → S1.1 descriptive only; crossing law reported as an unvalidated predictor** | V1, V4 |
| D4.8 | criteria (1)–(3) unchanged; N1 := N0 when its slope ≤ 0; per-draw slopes over m ∈ {256 … 16384}, averaged over draws | V3, V5 |
| D5 | duplicates kept, plus a sensitivity without surplus duplicates; one clamp threshold per model on the 16384 pool | V6 |

## V1. Mode: descriptive only [v1.1, D4.7]

`PREDICTOR_CHOICE_v1.1.md` records **no eligible estimator**:
- No pre-listed anchor-local variant passed both the S_sel eligibility check (D4.4) and the null gate (D4.5).
- The frozen v1.0 P1 failed both.

Consequences:
- **Gate G1 is not run as a confirmatory test.** Criteria (1)–(3) are not evaluated with pass/fail, and no "confirm" or "G1 passed" statement may be made from S1.1.
- **The S1.1 deliverables** are:
  - the observed s_N(m, k) curves;
  - the slope classes;
  - the nulls N0 and N1;
  - the crossing law as an **unvalidated predictor** (V4), reported with the same accuracy statistics as the nulls and labelled "unvalidated (PREREG v1.1 D4.7)".
- **Reading.** PREREG v1.0 states what happens when the crossing law is not confirmed: it is reported as descriptive only, and the paper's lead result shifts to the lattice/link theorems. This plan does not re-decide that consequence.

## V2. Anchors, cells and data use [v1.1, overrides O3 "Prediction"]

**Unchanged from O3.** Draws, halves and nested prefixes; the fit size m_f = 1024.

**Predicted anchors of a draw.**
- At m ≥ 1024: the H2 items of the draw's 1024 prefix (515 in draw 0). This set is fixed, and neighbours are searched in the whole m prefix.
- At m ∈ {256, 512}: the H2 items inside the m prefix.
- Why: under D4.2 an anchor can be predicted only from its own row at m_f.

**Observed s_N(m, k)** is the mean over the predicted anchors of |N_k^A ∩ N_k^B|/k, with the SPEC tie rule on pool position.
- The draft's quantity (all H2 items of each prefix) is also computed, as the descriptive column `obs_allH2`.
- Both are reported. The curves and slope classes of V5 use `obs`, so that every predictor and null is compared with the same anchors.

**What each quantity reads.**
- The unvalidated predictor reads only the predicted anchors' own A- and B-rows at m_f and their A-rows at m.
- N1 reads H1 × H1 of the m_f prefix.
- N0H1 reads the H1 rows of the m_f prefix.
- No B entry at any m ≠ 1024 enters a prediction (`hrhv2/tests/test_predict_v11.py`).

**Cells.**
- (pair, m, k) with m ∈ {256, 512, 1024, 2048, 4096, 8192, 16384} and k ∈ {1, 10, 100}.
- m = 1024 is in-sample for every predictor and null. It is excluded from accuracy statistics and included in slopes (V5).

**Implementation.** One call of `hrhv2.predict_v11.run_unit` per (pair, pre, draw, direction), with:
- `variants = ("iii-LSlog-EB",)` (V4), or `()` for a nulls-only run;
- `modes = ("d2", "mu")`, `m_f = 1024`, `ks = (1, 10, 100)`;
- inputs: the predicted anchors' rows of D_A and D_B over the 16384 pool, in the draw's order, plus the 1024 × 1024 fit block, with D built by the frozen `hrhv2.prep` path.

The draft runner `analysis/s11/s11_run.py` implements the v1.0 protocol and is not used for v1.1. A v1.1 runner must:
- keep its COCO authorization gate;
- check the hashes of `hrhv2/predict_v11.py` (`a031b213…`), `hrhv2/PREDICTOR_CHOICE_v1.1.md` and this file;
- record `D4 = "b-v1.1: no eligible estimator (descriptive)"` in `LEAD_AUTHORIZATION.json`.

## V3. Nulls and computation mode [v1.1, overrides O8's N0/N1 and the draft's N1 default]

**N0** = the observed s_N of the predicted anchors at m_f = 1024, from their own rows. This is the like-for-like fit-m value for the evaluated anchor set. The draft's N0 (H1 anchors) is reported as `N0H1`.

**N1** = hrhv2's N1:
- σ_A = RMS of the global affine residual on H1 × H1 at m_f, divided by the slope.
- It is computed in mode `d2`: the exact expected discrepancy for k ≤ 3 (D2) and the SPEC μ for k ≥ 10.
- **D4.8:** when the slope is ≤ 0 or undefined, N1 := N0. This replaces the draft's chance-level default; `N1c` is not used.
- The μ-law N1 is reported as `N1|mu`.

**N2 and P2** may be reported as exploratory from the frozen v1.0 code on the same anchors. They are not required.

## V4. The unvalidated predictor [v1.1, D4.7]

**Choice.** The crossing law is represented by variant `iii-LSlog-EB` of `hrhv2/predict_v11.py`, computed in modes d2 and mu, in both directions; the primary value is the direction mean (O6).

**How it was chosen.** It is the variant with the lowest plant score among the variants that passed the S_sel eligibility check: (iii) 0.062, (i) 0.067, (ii) 0.257. The S_sel median |log ratios| were 0.063 for (iii) and 0.056 for (i).

**Caveat.** This reporting rule was stated after the selection results. It carries no confirmatory weight, and the variant failed the D4.5 gate.

**Lead's option.** It may be dropped to save compute (V8); the descriptive deliverables of V5 do not need it.

## V5. Descriptive analyses [v1.1, D4.8 slopes; replaces the G1 evaluation of O7–O9]

**Curves.** s_N(m, k) per pair, as the draw mean ± draw sd, for `obs` and `obs_allH2`.

**Slopes (D4.8).**
- For each (pair, k, draw): the OLS slope of ln max(s_N, ε) on ln m over all seven m.
- ε(m, k) = 0.5/(n_anchors(m) · k): the draft's O5, per draw, with the V2 anchor counts.
- (pair, k) slope = the mean of the 5 draw slopes.
- Computed for `obs`, N0 (identically 0), N1 and the unvalidated predictor.

**Slope classes per (pair, k).**

| class | condition | trichotomy reading |
|---|---|---|
| rising | mean slope ≥ +0.1 | conformal-like |
| flat | \|mean slope\| < 0.1 | smooth |
| falling | mean slope ≤ −0.1 | rough |

Counts are reported per pair class and per k. If no unit has \|slope\| ≥ 0.1, record "only smooth class observed".

**Accuracy table (descriptive; no thresholds, no pass/fail).** For N0, N1 and the unvalidated predictor, per pair class and pooled:
- the median |log(X/obs)| over cells with m ≠ 1024 and obs > 2k/(m − 1);
- the MSLE on the units that are not flat;
- the fraction of non-flat units whose slope sign matches the observed sign.

**Secondary (O10, descriptive).**
- Slope ordering seed–seed ≥ cross-vision ≥ cross-modal.
- k-elasticity vs m-elasticity, using the V5 slopes.

## V6. Preprocessing and duplicates [v1.1, D5]

**Primary: `prh_clip+l2`.**
- One clamp threshold per model, computed on all 16384 pool rows and **pooled over all four stored layers** (the draft's `pool-stored`), following D5's wording "pooled over all stored layers and computed on the full 16384-item pool".
- **Recorded conflict.** D5's parenthetical label reads `pool-layer`, which in the draft names the single-layer scope. This plan follows the definition text and runs `pool-layer` as a sensitivity. `LEAD_AUTHORIZATION.json` must name `clip_mode = "pool-stored"` for the primary run, or override this paragraph explicitly.

**Secondary: `l2`.**

**Duplicates** (44 groups / 54 surplus items in the pool) are kept and resolved by the SPEC tie rule. Sensitivity: rerun with the surplus duplicates dropped from the pool, keeping the lowest pool position of each group. Exact-tie counts at the k boundary are recorded per cell (O11).

## V7. Pairs, directions, draws

**Unchanged:** O1 (29 primary pairs; 61 with all 36 seed pairs as secondary), O6 (both directions, mean primary), O3 (5 draws).

## V8. Compute and expected cost

Measured on COCO-size dev plants: M = 16384, 491 predicted anchors, 1 core per unit (`results/S1_1_select/coco_cost_timing.json`, `coco_cost_nulls_timing.json`). These are timings only, on plants with seed prefix `S11selCost`. The real COCO couplings may be more or less saturated; saturation drives the exact k = 1 and μ cost.

**Per unit = (pair, pre, draw, direction):**
- nulls only (N0, N0H1, N1 in d2 + mu, observed): 8–82 s;
- plus the unvalidated predictor (iii): 367–656 s (variant i 462–678 s; ii 150–502 s).

**Per (pair, pre) job, once:**
- two full 16384² D via `hrhv2.prep` with `prh_clip+l2`: 6–15 s per model on synthetic features with d = 768–3200, about 2.1 GB each;
- all-row kNN tables for `obs_allH2`: about 2.4 s per model and m.

**Primary design:** 29 pairs × 2 pre × 5 draws × 2 directions = 580 units.

| configuration | CPU-h | wall at 12 cores |
|---|---|---|
| nulls only | ≈ 1.3–13 (+ D builds ≈ 0.3, + `obs_allH2` kNN ≈ 3) | ≈ 0.4–1.4 h |
| with the unvalidated predictor | ≈ 60–106 | ≈ 5–9 h |
| secondary 32 extra seed pairs | ≈ +110 % of the above | — |

**Memory.** About 5 GB per worker (two full D plus the anchor-row arrays). At most 12 workers by the CPU rule, and fewer if other roles share the 125 GB.

## V9. Unchanged from the draft

- O1 (pairs, classes, direction labels)
- O2 (depth rule; preprocessing labels, with V6's scope choice)
- O3 (draws, halves)
- O5 (log floor, with V2 anchor counts)
- O6 (directions)
- O10 (secondary, descriptive)
- O11 (ties)
- O12 (provenance outputs; plus `predict_v11.py` and `PREDICTOR_CHOICE_v1.1.md` hashes)
