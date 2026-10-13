# Lead decisions D1–D6 and PREREG v1.1 (2026-10-08)

This file amends `prereg/PREREG_stage1.md` v1.0, whose sha256 is `bee0583c…`. The v1.0 file is unchanged. Every analysis already run under v1.0 stays reported under v1.0. All changes below are made **before any COCO metric is computed** (S1.1 has not opened COCO features), and they are justified only by evidence from plants and from S_sel.

## D1 — P-Bern gate (SPEC §7)

**Cause of the failure.** Diagonal float64 noise after L2 normalisation (|K_ii − 1| ≤ 8.9e-16), amplified by β, reorders far-tail pairs. A separate cross-fitted isotonic check failed because one evaluation pair falls below the fitted range and constant extrapolation is applied there. Both are numerical artifacts of how the plant is built, not defects of the metrics. kNN, components and MST are exactly identical in all 27 cells in both implementations.

**Decision.**
- Build P-Bern monotonically: D_B = −2·expm1(β(K_A − 1)), with unit diagonals enforced.
- The gate requires zero **strict** discordances for Δ_O and Δ_E. Float64 ties are reported separately and not counted as failures.
- The isotonic gate is read in-sample. The cross-fitted value is reported, with a note on the extrapolation.

## D2 — P-Gauss at k = 1

The failure comes from the SPEC §6 μ approximation itself; with the true coupling it still misses by 0.047–0.062. The exact expected discrepancy is within 0.019.

**Decision.** Use the exact expected discrepancy for k ≤ 3. It may be used for every k if affordable, and the choice is frozen with the estimator under D4.

## D3 — Vendor identity tolerances

**Decision.**
- The ≤ 1e-6 tolerances of G1′ and SPEC §8 refer to float64 inputs.
- Exact ties, such as the duplicate caption on WIT items 493/531, are resolved by the SPEC tie rule.
- The near-tie fraction is always reported.

## D4 — S1.1 estimator (central decision)

**Diagnosis.**
- The frozen v1.0 P1 pools the coupling across the other half's anchors, on rank-normalised u.
- The crossing law, however, conditions on the anchor x_i. Pooling anchors with different local scales manufactures dispersion.
- That is why v1.0 P1 fails on P-Conf (it predicts collapse where s_N → 0.91) and on S_sel (it predicts ≤ 0.035 where 0.13–0.48 is observed).
- G0 never checked predictor accuracy on the smooth plant classes, which was a gap in SPEC §7.
- The v1.0 P1 result on S_sel is reported as a failure of **that estimator**, not of the theorem.

**Decision (PREREG v1.1 for S1.1).**

1. **Candidates.** The P1 estimator is re-selected among at most 4 pre-listed variants, all **anchor-local**:
   - (i) a per-anchor log-distance location–scale model, i.e. the multiplicative residual model, fitted on candidates outside the predicted rank band;
   - (ii) a per-anchor row-isotonic link plus a residual dispersion model, fitted with candidate-level cross-fitting (even/odd candidate ranks);
   - (iii) variant (i) or (ii) with dispersion shrunk toward the pooled-anchor estimate;
   - (iv) the v1.0 P1, kept as a reference.
2. **Data the estimator may use.** Following SPEC §6 ("the same anchor at a different m"), the predicted anchors' own rows **at the fit m = 1024 only**. Cells at m = 1024 itself are in-sample and are not scored.
3. **Selection rule.** Choose the variant that minimises the maximum, over plant classes, of the class-mean absolute error. Plant classes: P-Gauss, P-Aniso, P-Conf, P-Rough, and a new P-Smooth+noise mixture.
4. **Real-data check (no selection on it).** S_sel (wit4096_sel, 1023 rows) is used as a real-data sanity check: one of 4 ViT–ViT pairs, 4 LM–LM pairs and 4 LM–ViT pairs, with an m-sweep 256 → 1023 by nested prefixes. A variant whose median |log(pred/obs)| on S_sel exceeds log 1.5 is **ineligible**, whatever its plant score.
5. **Additional gate (new).** The selected variant must beat null N0 on P-Conf and P-Aniso, and must beat N1 on P-Rough and P-GaussSat.
6. **Freeze.** The choice is frozen in `hrhv2/PREDICTOR_CHOICE_v1.1.md`, with its sha256 (and that of `predict.py`) appended to STATUS.md before COCO is opened.
7. **If no variant passes (4) and (5).** Record "no eligible estimator". S1.1 is then run descriptively only: observed s_N(m, k) curves, slope classes and nulls. The crossing law is reported as an unvalidated predictor.
8. **Evaluation rule, N1 and slopes.**
   - Confirm criteria (1)–(3) are unchanged from v1.0.
   - When the fitted slope of N1 is ≤ 0 or undefined, N1 is the constant N0.
   - Slopes are OLS slopes of log s_N on log m over m ∈ {256, …, 16384}, per (pair, k, draw), averaged over draws.
   - The pinned decisions in the draft `analysis/s11/ANALYSIS_PLAN_S11.md` are adopted, except where this file overrides them.

## D5 — Duplicates and clip scope

**Decision.**
- Duplicate texts are kept and resolved by the tie rule. A sensitivity run drops the surplus duplicates (44 groups / 54 surplus items in the COCO pool; WIT 493/531).
- Clip threshold:
  - WIT: pooled over all of a model's layers (equal to SOG/ARH).
  - S1.1 nested pools: one threshold per model, pooled over all stored layers and computed on the full 16384-item pool (`pool-layer`).
  - `l2` is the secondary preprocessing.

## D6 — Whose forest in PE2

**Decision.** The forest of A (the LM) is primary, as frozen in `ANALYSIS_PLAN_S12.md`. The forests of B, of the union and of the intersection are reported as sensitivities.

## S1.3a — prediction 1(b)

The verdict "FALSIFIED" stands as the frozen rule's output. The report must state two things:
- **Why the test was ill-posed.** It compared one N = 65536 mean against a narrow CI of the N = ∞ mean; equal means pass with probability ≈ 0.46.
- **What the diagnostics show.** The same-init paired differences are ~1e-4, with CIs containing 0.

Part 1(a) holds: 0 passes in every cell. No re-test is run under a new rule for this cohort.

## Additional analysis authorised (S1.2 secondary, preregistered as "replication only")

**ARH layer-max replication** on the 136-pair subpanel:
- calibrated mKNN (k = 10) and biased CKA, max over all layer pairs, with the same permutation across layers inside every permutation;
- K = 500 if feasible on the GPU, otherwise K = 200, recorded.

It answers whether ARH's loss of the CKA trend comes from layer-max selection. Our fixed-depth result shows that, along the BLOOMZ axis, calibrated CKA keeps its trend (0.70 / 0.92).

## Authorisation

The S1.1 runner may write `analysis/s11/LEAD_AUTHORIZATION.json` and open COCO features **only after**:
- step 6 of D4 is recorded in STATUS.md, or step 7 is recorded (descriptive mode);
- `ANALYSIS_PLAN_S11.md` (amended per D4.8) is frozen, with its sha256 in STATUS.md.

The authorisation JSON must cite this file's sha256.
