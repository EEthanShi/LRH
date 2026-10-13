# S1.2 LLaMA axis, Part B: instantiated predictions from features only (v1)

Role `llama-axis-predict`. This is the B6 output of `analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md` (Part A, sha256 990428e3f29c7ecfba9cf314a45c73c79f0154f86a07520429910d282c0a2fc8), §9. The task text asked for `PREDICTIONS_PART_B.md`; that path is a symlink to this file, which carries the name the frozen plan gives.

## 0. Freeze conditions and provenance

- **No cross-model score existed when this file was written.** This covers every observed or permuted score involving llama-13b or llama-30b, at any depth.
  - `results/S1_2_layermax_llama/pairs/` was absent at the start and at the end of the Part B run (`partB_features.json` → `meta.guard`, `meta.pairs_dir_exists_at_end`; STATUS 23:07:35Z).
  - No llama-axis analysis process has run.
- **Inputs are feature stacks only.**
  - llama-7b `1a727b78…`, llama-13b `8f138d43…`, llama-30b `460517a5…`;
  - the 7 other S1.2 LMs and the 17 ViTs in `features/wit1024/`;
  - all 27 sha256 are in `partB_features.json` → `meta.inputs`.
- **Clip thresholds.** Recomputed with `hrhv2.prep.prh_clip_threshold` on each float64 all-layer stack (D5).
  - All 25 S1.2 models are bitwise equal to `results/S1_2/cache/clip_thresholds.json`.
  - New values: llama-13b 1.7140773002756753, llama-30b 1.5034985621459782.
  - File: `results/S1_2_layermax_llama/partB/clip_thresholds_partB.json`. V-L1, run with the E/I copies, must reproduce these bitwise.
- **Code.**
  - `analysis/s12_layermax_llama/partb_pr.py` (sha256 6103a8be8bddcda65b12c7ddca812dc601189550fb5b3c50009899ed8dd2b9ba) computes B1–B5. It reuses `analysis/s12/common.preprocess_rows`, `hrhv2.prep.center_gram` and `analysis/s12/aggregate._rho_matrix`, all as read-only imports.
  - `analysis/s12_layermax_llama/partb_checks.py` (sha256 b2e55a169532e82330378af30b16bf4931e321837405d57064653346c09397dd) runs sanity checks and retrodictions on the 8 known LMs only.
- **Outputs.**
  - `results/S1_2_layermax_llama/partB/partB_features.json`, sha256 ede0127589a02037427b343e9c3593ee427d430e9746d7f4942b15f76a3031bf (all B1–B5 numbers);
  - `results/S1_2_layermax_llama/partB/partB_retro_checks.json`, sha256 1f599c362518aea66bb998f95877dc03ffbda88969c9779d93e9e35224520948.
- **Implementation error fixed before this file was written.**
  - The B4 test "|Tr| = 1" used exact float equality. Spearman through `_rho_matrix` returns 0.9999999999999998 for a perfectly monotone row, so the first run marked four trends "not material".
  - The test is now ||Tr| − 1| ≤ 1e-12, the `lm_trend` tolerance, plus a direct check that the values are strictly monotone in size for every ViT.
  - The first output is kept as `partB_features.attempt1_materiality_float_bug.json` (sha256 240ca7d8…). A field-by-field comparison shows that only those four `material` flags changed; every number is bitwise identical.
- **Order deviation (logged in STATUS).** S4/V-L2 (kNN tables) has not been run. No Part B quantity uses a kNN table.

## 1. What was computed (plan §9, B1–B5)

- **B1.** PR_u = (tr K̄)² / ‖K̄‖_F², with K̄ = HKH and K = XXᵀ of the 1024 preprocessed rows in float64. This was done for every stored layer of the 10 LMs (332 layers) and the 17 ViTs (352 layers), under both pre.
- **B2.**
  - Entry baselines: B_uw = √(PR_u PR_w)/(m − 1), with m = 1024.
  - Layer-max baseline: B̄ = max_uw B_uw = √(PR*_LM · PR*_ViT)/1023. The factorised value equals the brute-force maximum of the full B matrix exactly (max |diff| = 0).
  - Fixed-depth baselines B_fd(r) at the §3 layers.
- **Metric-specific baselines.** For every pair and layer pair, these do not depend on features:
  - mKNN: 10/1023 = 0.009775;
  - debiased CKA: 0 (plan §8).

## 2. Primary instance: prh_clip+l2

### 2.1 LLaMA PR profiles and the threshold-setting LM layer

| model | L_A | PR* | u* (rel. depth) | runner-up layer, PR*/PR(runner-up) | PR(emb) | PR(last, post-norm) | PR at r = 0.5 (layer) |
|---|---|---|---|---|---|---|---|
| llama-7b | 33 | 257.33 | 28 (0.875) | 27, 1.004 | 131.38 | 40.12 | 119.88 (16) |
| llama-13b | 41 | 308.57 | 34 (0.850) | 33, 1.008 | 156.98 | 49.47 | 204.98 (20) |
| llama-30b | 61 | 359.22 | 48 (0.800) | 47, 1.001 | 162.39 | 99.63 | 272.00 (30) |

- **Where the max baseline comes from.** A late block sets it, at relative depth 0.80–0.88, not the embedding. It sits on a broad plateau: the runner-up is within 0.1–0.8%.
- **Agreement for llama-7b.** This is the layer the post-hoc null profile found (layer 28, §4 R2). The observed llama-7b CKA argmax depth was 0.83 (Part A §0).
- **ViT side.** w* is the last stored block for 15 of 17 ViTs; the exceptions are MAE-L (block 20 of 0–23) and MAE-H (block 29 of 0–31).

### 2.2 Predicted layer-max baseline B̄ (biased CKA)

- **Means over the 17 ViTs:** 7B 0.1847, 13B 0.2023, 30B 0.2182.
- **Direction.** B̄ is strictly increasing in size for **every one of the 17 ViTs**.
  - ρ_v(B̄) = +1 for every ViT, so **Tr_pred = Tr_L(B̄) = +1 and s = +1**.
  - The ratios √(PR*_j′/PR*_j) are common to all ViTs: 13B/7B 1.0950, 30B/13B 1.0790, 30B/7B 1.1815.
- **Gaps** gap_v = |B̄(j′,v) − B̄(j,v)|, as median [min, max] over ViTs, with the number of ViTs where gap ≥ δ_b = 0.005:

| size pair | median gap [min, max] | ViTs with gap ≥ δ_b |
|---|---|---|
| 7B–13B | 0.0189 [0.0065, 0.0237] | 17/17 |
| 13B–30B | 0.0172 [0.0059, 0.0216] | 17/17 |
| 7B–30B | 0.0361 [0.0124, 0.0453] | 17/17 |

- **B4: MATERIAL.** |Tr_pred| = 1, and the smallest median gap (0.0172) is at least 0.005.

Per-ViT values. PR* and w* are for the ViT; each triple is 7B / 13B / 30B.

| ViT | PR* | w* (of 0..L_B−1) | B̄ | B_fd(0.5) | B_fd(1.0) |
|---|---|---|---|---|---|
| ViT-Ti/16 in21k | 70.4 | 11 of 11 | 0.1316 / 0.1441 / 0.1554 | 0.0409 / 0.0534 / 0.0615 | 0.0519 / 0.0577 / 0.0819 |
| ViT-S/16 in21k | 122.6 | 11 of 11 | 0.1737 / 0.1902 / 0.2052 | 0.0471 / 0.0616 / 0.0710 | 0.0686 / 0.0761 / 0.1081 |
| ViT-B/16 in21k | 196.5 | 11 of 11 | 0.2198 / 0.2407 / 0.2597 | 0.0532 / 0.0696 / 0.0802 | 0.0868 / 0.0964 / 0.1368 |
| ViT-L/16 in21k | 253.1 | 23 of 23 | 0.2495 / 0.2732 / 0.2948 | 0.0706 / 0.0923 / 0.1064 | 0.0985 / 0.1094 / 0.1552 |
| MAE-B/16 | 18.9 | 11 of 11 | 0.0682 / 0.0747 / 0.0806 | 0.0306 / 0.0400 / 0.0460 | 0.0269 / 0.0299 / 0.0425 |
| MAE-L/16 | 32.9 | 20 of 23 | 0.0899 / 0.0985 / 0.1063 | 0.0485 / 0.0635 / 0.0731 | 0.0325 / 0.0360 / 0.0511 |
| MAE-H/14 | 39.5 | 29 of 31 | 0.0985 / 0.1079 / 0.1164 | 0.0591 / 0.0773 / 0.0891 | 0.0354 / 0.0393 / 0.0558 |
| DINOv2-S | 126.1 | 11 of 11 | 0.1761 / 0.1928 / 0.2080 | 0.0328 / 0.0429 / 0.0494 | 0.0695 / 0.0772 / 0.1096 |
| DINOv2-B | 192.4 | 11 of 11 | 0.2175 / 0.2381 / 0.2570 | 0.0423 / 0.0553 / 0.0637 | 0.0859 / 0.0954 / 0.1353 |
| DINOv2-L | 239.1 | 23 of 23 | 0.2425 / 0.2655 / 0.2865 | 0.0344 / 0.0450 / 0.0519 | 0.0957 / 0.1063 / 0.1509 |
| DINOv2-g | 194.2 | 39 of 39 | 0.2185 / 0.2393 / 0.2582 | 0.0476 / 0.0622 / 0.0717 | 0.0863 / 0.0958 / 0.1360 |
| CLIP-B/16 laion2b | 128.4 | 11 of 11 | 0.1777 / 0.1945 / 0.2099 | 0.0423 / 0.0554 / 0.0638 | 0.0701 / 0.0779 / 0.1105 |
| CLIP-L/14 laion2b | 142.8 | 23 of 23 | 0.1874 / 0.2052 / 0.2214 | 0.0559 / 0.0730 / 0.0841 | 0.0740 / 0.0822 / 0.1166 |
| CLIP-H/14 laion2b | 160.9 | 31 of 31 | 0.1989 / 0.2178 / 0.2350 | 0.0717 / 0.0937 / 0.1079 | 0.0785 / 0.0872 / 0.1238 |
| CLIP-B/16 ft-in12k | 193.9 | 11 of 11 | 0.2184 / 0.2391 / 0.2580 | 0.0477 / 0.0624 / 0.0719 | 0.0862 / 0.0957 / 0.1359 |
| CLIP-L/14 ft-in12k | 211.2 | 23 of 23 | 0.2279 / 0.2495 / 0.2692 | 0.0562 / 0.0735 / 0.0847 | 0.0900 / 0.0999 / 0.1418 |
| CLIP-H/14 ft-in12k | 242.5 | 31 of 31 | 0.2442 / 0.2674 / 0.2885 | 0.0664 / 0.0868 / 0.1000 | 0.0964 / 0.1071 / 0.1519 |

### 2.3 Fixed-depth entry baselines (used to attribute P-L3)

- **r = 0.5** (LM layers 16 / 20 / 30).
  - Means 0.0498 / 0.0652 / 0.0751, strictly increasing for all 17 ViTs; Tr_L(B_fd) = +1.
  - Median gaps 0.0147 (7B–13B), 0.0095 (13B–30B), 0.0242 (7B–30B).
  - **Material.**
- **r = 1.0** (LM layers 32 / 40 / 60, post-final-norm).
  - Means 0.0725 / 0.0806 / 0.1143, strictly increasing for all 17 ViTs; Tr_L(B_fd) = +1.
  - Median gaps 0.0087 (7B–13B; 14 of 17 ViTs ≥ δ_b), 0.0366 (13B–30B), 0.0452 (7B–30B).
  - **Material.**
- **Consequence.** On this axis the fixed-depth entry baselines also rise with size, at both depths. Their level is lower (0.05–0.11, against 0.18–0.22 for B̄), but their gaps are comparable to the layer-max gaps. So the equal-baseline premise behind the fixed-depth remedy does not hold here.

### 2.4 B5 descriptive extras

- **Pooled association.** Pearson(ARH rank, B̄) over the 170 pooled pairs is +0.377. For comparison: 8-LM panel +0.270, dense ranks +0.378, log num_params +0.406.
- **Mean B̄ by LM:**

| LM | mean B̄ |
|---|---|
| bloomz-560m | 0.1333 |
| bloomz-1b1 | 0.1285 |
| bloomz-1b7 | 0.1793 |
| bloomz-3b | 0.1814 |
| bloomz-7b1 | 0.1818 |
| open_llama_3b | 0.1636 |
| open_llama_7b | 0.1741 |
| llama-7b | 0.1847 |
| llama-13b | 0.2023 |
| llama-30b | 0.2182 |

- **Threshold-setting LM layer u*/(L_A − 1):** llama 0.875 / 0.850 / 0.800. BLOOMZ, for comparison: 0, 1.0, 0, 0, 0.03.
- **BLOOMZ retrodiction** (outcome known, not a test): Tr_BLOOMZ(B̄) = +0.90, equal to the known Tr_BLOOMZ(c_α) = +0.90.

## 3. Secondary instance: l2

- **Threshold-setting layer.** The LM layer that sets the threshold is the **embedding output** (u* = 0) for all three sizes.
  - PR*: 131.38 / 156.98 / 162.39.
  - PR*/runner-up: 1.035 / 1.140 / 3.238.
- **B̄.**
  - Means 0.1181 / 0.1291 / 0.1313, strictly increasing for all 17 ViTs; Tr_pred = +1.
  - Median gaps: 7B–13B 0.0115 (16/17 ViTs ≥ δ_b); 13B–30B 0.0023 (0/17); 7B–30B 0.0138 (16/17).
  - **B4: NOT material**, because the 13B–30B median gap of 0.0023 is below 0.005.
- **B_fd(0.5)** (LM layers 16 / 20 / 30).
  - 0.0132 / 0.0146 / 0.0131, so 13B is highest and the trend is not monotone: ρ_v = −0.5 for every ViT, Tr −0.5.
  - All gaps ≤ 0.0023. **Not material.**
- **B_fd(1.0).**
  - 0.0382 / 0.0477 / 0.0727, increasing for all 17 ViTs; Tr +1.
  - Median gaps 0.0099 / 0.0262 / 0.0362. **Material.**
- **B5.** Pearson(ARH rank, B̄) is +0.265 on the 10-LM panel (+0.280 on the 8-LM panel). BLOOMZ retrodiction: Tr(B̄) +0.90, equal to the known Tr(c_α) +0.90.

## 4. Sanity checks and retrodictions on the 8 known LMs (outcomes known; not tests)

From `partB_retro_checks.json`. Each value is given as prh_clip+l2 [l2].

- **R1: PR reproduction.**
  - The PRs reproduce the stored profiles in `results/S1_2_layermax/indep/diag_null_profile.json` for all 25 S1.2 models, to a maximum relative error of 1.5e-15 [1.4e-15]. They are not bitwise equal because the BLAS thread count differs.
  - In the stored indep run, the entry-wise permutation null means agree with Proposition 5 to a median relative error of 0.46% [2.3%], max 2.6% [6.2%].
- **R2: sanity check against `posthoc_null_structure.json`.**
  - The predicted per-LM null level, mean_v B̄, is compared with the post-hoc profile maximum max_a mean_v max_b E0[S_ab]. The largest |relative error| over the 8 LMs is 1.0e-4 [1.1e-4].
  - The predicted u* equals the post-hoc argmax layer for 8/8 LMs under both pre. Under prh_clip+l2: BLOOMZ 0, 24, 0, 0, 1; open_llama 25, 30; llama-7b 28.
  - BLOOMZ, predicted vs post-hoc:

    | LM | predicted | post-hoc |
    |---|---|---|
    | bloomz-560m | 0.13325 | 0.13324 |
    | bloomz-1b1 | 0.12848 | 0.12848 |
    | bloomz-1b7 | 0.17926 | 0.17928 |
    | bloomz-3b | 0.18140 | 0.18140 |
    | bloomz-7b1 | 0.18182 | 0.18184 |

- **R3: thresholds on the 136 known pairs.**
  - |c_α − B̄|: median 0.0020 [0.0022]; c_α − B̄ ranges over [0.0016, 0.0028] [[0.0017, 0.0036]].
  - E0[max] − B̄: mean 6e-5 [1e-5].
  - Pearson(c_α, B̄) = 1.000.
  - BLOOMZ c_α order agrees with the B̄ order on **111/111 [161/161]** (ViT, size-pair) comparisons with gap ≥ δ_b.
  - llama-7b: mean c_α 0.1866 vs mean B̄ 0.1847 [0.1202 vs 0.1181].
- **R4: mKNN and debiased CKA.**
  - mKNN: c_α mean 0.01389 [0.01478] against the baseline 0.00978, so ζ has mean 0.0041 [0.0050] and max 0.0054 [0.0133].
  - Debiased CKA: c_α mean 0.0051 [0.0061]; E0[max] mean 0.0026 [0.0030]. This comes from multiplicity only, since every entry baseline is 0.
- **Descriptive expectation (not a decision rule).** These suggest c_α ≈ B̄ + 0.002 on the LLaMA axis. That gives means near 0.187 / 0.204 / 0.220 under prh_clip+l2.

## 5. Instantiated predictions — primary: prh_clip+l2, layer-max; s_cal primary, uncensored secondary

**Conventions.**
- ΔTr_L(score) = Tr_L(score) − Tr_L(raw T), for the same protocol, metric and pre.
- Tr_L is defined as in plan §5: 17 ViTs, 3 sizes, exact p over 6 relabellings.
- The smallest one-sided p is 1/6, so none of these outcomes can be significant. Each is a directional, descriptive result on one model family, with ViT-only uncertainty.
- 0.10 is the plan's materiality threshold for Tr_L.

### P-L1 (mKNN, layer-max)

- **Part B input.** B̄ = 10/1023 = 0.009775 for all 51 pairs and every layer pair, so there is no size-dependent baseline.
- **Prediction:** |ΔTr_L(s_cal)| < 0.10 **and** |ΔTr_L(unc)| < 0.10.
- **The only size-dependent term is multiplicity**: 33·L_B, 41·L_B and 61·L_B layer pairs. It can only push Tr lower. On the known panel ζ = c_α − 10/1023 averaged 0.0041 (max 0.0054).
- **If it fails** (|ΔTr| ≥ 0.10 for either score), that counts under plan §10 item 3.
- **Deductive check** (reported; a violation means an implementation error): no (ViT, size pair) whose raw gap exceeds ζ_j + ζ_j′ changes order.

### P-L2 (biased CKA, layer-max): MATERIAL case, s = sign(Tr_pred) = +1

- **Prediction:** calibration moves the LLaMA trend down, sign(ΔTr_L(s_cal)) = −1.

| outcome | condition | how it counts |
|---|---|---|
| confirmed | ΔTr_L(s_cal) ≤ −0.10 | for the prediction |
| consistent-null | \|ΔTr_L(s_cal)\| < 0.10 | neither way; Corollary 2 allows it when raw gaps exceed baseline gaps |
| contradicted | ΔTr_L(s_cal) ≥ +0.10 | plan §10 item 2, reported jointly with P-L2a |
| not testable | Tr_L(raw T) = −1 (within 1e-12) | no move in the predicted direction is possible |

- The same four-way rule is applied to ΔTr_L(uncensored), as a secondary instance.
- **What the numbers imply** (for reading only; this is not an extra rule).
  - Under s_cal, a ViT's concordant raw pair T_j < T_j′ becomes discordant iff T_j′ − T_j < (1 − T_j)(c_j′ − c_j)/(1 − c_j).
  - Take the llama-7b levels (T ≈ 0.43, c ≈ 0.19) and threshold steps equal to the B̄ steps (medians 0.019 for 7B→13B and 0.017 for 13B→30B). Raw gaps below about 0.013 and 0.012 respectively then flip.
  - Under uncensored (T − E0[max], with E0[max] ≈ B̄), a pair flips iff T_j′ − T_j < B̄_j′ − B̄_j, the per-ViT gaps of §2.2.

### P-L2a (premise; plan §10 item 1)

1. The median over the 51 LLaMA-axis pairs of |c_α − B̄| is ≤ 0.01. The retrodiction on the 136 known pairs gives 0.0020.
2. All 51 (ViT, size-pair) comparisons have gap ≥ δ_b (minimum gap 0.0059).
   - The predicted order in each is **c_α(larger LLaMA) > c_α(smaller LLaMA)**.
   - The premise holds if **at least 34 of the 51** agree (≥ 2/3).

**Deductive check** (reported; a violation means an implementation error): every ViT whose c_α increases with size has ρ_v(s_cal) − ρ_v(raw) ≤ 0. Any other monotone c_α order implies the opposite sign.

### P-L3 (biased CKA, fixed depth r = 0.5 and r = 1.0, each entry calibrated against its own null)

- **Applies** at each r where |Tr_L(raw, r)| ≥ 0.10. Let s_r = sign(Tr_L(raw, r)).
- **Prediction: "keeps sign"**, meaning s_r·Tr_L(s_cal, r) ≥ 0.10.
  - "Lost" if |Tr_L(s_cal, r)| < 0.10.
  - "Reversed" if s_r·Tr_L(s_cal, r) ≤ −0.10.
- **Attribution, recorded now.** B_fd(r) increases materially with size at both depths (§2.3).
  - If s_r = +1 and the outcome is "lost" or "reversed", it is attributed to the entry's own baseline trend (Corollary 2 with agg = s_{u0v0}).
  - "Reversed" counts against the remedy claim (plan §10 item 4), not against the corollary.
- **Part B flags this prediction as at risk on this axis when s_r = +1.** The fixed-depth baseline gaps (medians 0.009–0.045) are comparable to the layer-max gaps.

### P-L4 (debiased CKA; entry baseline 0 for every layer pair, plan §8)

- **Predictions.**
  - (a) ΔTr_L(s_cal) ≥ −0.10 **and** ΔTr_L(unc) ≥ −0.10 at layer-max.
  - (b) On the 10-LM pooled panel, the ARH-style Pearson r with ARH rank at layer-max satisfies r_after ≥ r_before − 0.05.
- **Known-panel magnitudes:** debiased c_α ≈ 0.005; E0[max] ≈ 0.0026, from multiplicity only.
- **If it fails**, that counts under plan §10 item 3.

### P-L5 (secondary; calibrate-then-max, biased CKA)

- **Applies** if |Tr_L(raw T, layer-max)| ≥ 0.10. Let s be its sign.
- **Prediction: "keeps sign"**, meaning s·Tr_L(M) ≥ 0.10.
  - "Lost" if |Tr_L(M)| < 0.10.
  - "Reversed" if s·Tr_L(M) ≤ −0.10, which counts under plan §10 item 4.
- **Part B note.** M subtracts each entry's own threshold, so the max-baseline trend of §2.2 does not enter M directly. Only the entry baselines near the observed argmax matter. If that argmax stays on the late-block PR plateau of §2.1 (llama-7b: depth 0.83), those entry baselines also rise with size, because the plateau PR rises from about 257 to 309 to 359.

### Descriptive, unscored (B5)

- Pearson(ARH rank, B̄) is +0.377 on the 10-LM panel (+0.270 on the 8-LM panel).
- By Corollary 2, this positive association points toward r_after(CKA) < r_before(CKA) in the ARH-style reading.
- It is not a plan prediction. The FP§6 reading in plan §7 is applied unchanged.

## 6. Secondary instance: l2 (failures confined to l2 are reported, not counted; plan §10)

- **P-L1, P-L4:** the same rules as §5.
- **P-L2: NOT material** (13B–30B median gap 0.0023), so there is **no directional prediction on Tr_L**.
  - Pairwise flips, for s_cal and uncensored, are reported descriptively for the size pairs whose gap is ≥ δ_b: 7B–13B (16 ViTs) and 7B–30B (16 ViTs).
  - In those pairs, Corollary 2 allows only concordant → discordant moves.
- **P-L2a.**
  - (i) Median |c_α − B̄| ≤ 0.01 (retrodiction: 0.0022).
  - (ii) There are 32 qualifying comparisons (7B–13B 16, 7B–30B 16, 13B–30B 0). The predicted order is c_α(larger) > c_α(smaller), and the premise holds if **at least 22 of the 32** agree.
- **P-L3:** "keeps sign" at both depths. Attribution:
  - At r = 0.5, B_fd is not monotone (7B < 13B > 30B, Tr −0.5) and not material (gaps ≤ 0.0023), so the equal-baseline premise approximately holds.
  - At r = 1.0, B_fd increases materially (0.0382 / 0.0477 / 0.0727).
- **P-L5:** the same rule as §5.

## 7. Summary

| ID | metric / protocol (primary pre) | Part B input | prediction | if it fails (plan §10) |
|---|---|---|---|---|
| P-L1 | mKNN, layer-max | baseline 10/1023 for every pair | \|ΔTr_L(s_cal)\| < 0.10 and \|ΔTr_L(unc)\| < 0.10 | item 3 |
| P-L2 | biased CKA, layer-max | B̄ 0.1847 / 0.2023 / 0.2182, Tr_pred +1, material | ΔTr_L(s_cal) ≤ −0.10 (confirmed); \|·\| < 0.10 consistent-null; ≥ +0.10 contradicted | item 2 (with item 1) |
| P-L2a | biased CKA, layer-max thresholds | 51 comparisons, all gap ≥ 0.005 | median \|c_α − B̄\| ≤ 0.01 and ≥ 34/51 with c_α(larger) > c_α(smaller) | item 1 |
| P-L3 | biased CKA, fixed r = 0.5, 1.0 | B_fd rising, material at both r | keeps sign (at risk: own baseline rises) | item 4 if reversed |
| P-L4 | debiased CKA, layer-max | baseline 0 for every layer pair | ΔTr_L(s_cal), ΔTr_L(unc) ≥ −0.10; pooled r_after ≥ r_before − 0.05 | item 3 |
| P-L5 | biased CKA, calibrate-then-max (secondary) | entry baselines | keeps sign | item 4 if reversed |

## 8. Next steps allowed by the plan

- **S6 and S7** (V-L3–V-L7, then the main runs E and I) may start only after this file's sha256 has been appended to STATUS.md.
- **V-L1 and V-L2 with the E/I copies come first.** V-L1 must reproduce `clip_thresholds_partB.json` bitwise.
- **V-L7 compares** the entry-wise permutation null means with the B_uw built from the PR vectors in `partB_features.json` (`per_pre.<pre>.lm|vit.<stem>.PR`).
