# HRH v2 Stage 0/1 — metric specification (frozen 2026-10-08, v1.0)

This file defines every criterion ("observable + discrepancy") computed in Stage 0/1. Two independent implementations (`hrhv2/` main, `hrhv2_ref/` reference) must follow it exactly and agree to the stated tolerances. Any deviation must be recorded in `SPEC_CHANGELOG.md` with a reason, never silently.

## 0. Inputs and conventions

- A representation on m evaluation items is a float array `X` of shape (m, d). All metric arithmetic is in **float64**.
- **Preprocessing is part of the criterion Q** and is passed explicitly:
  - `pre="l2"`: L2-normalize each row (rows with norm 0 are an error).
  - `pre="prh_clip+l2"`: PRH/ARH outlier clamp, then L2. Compute the per-sample 95th percentile of |x| over coordinates; average these over samples to get one threshold c; clamp every coordinate to [−c, c]; then L2-normalize. This must match the vendored ARH `preprocess.py` (`q_outlier=0.95`). Check this against the code and record the result.
  - `pre="none"`: use X as given (only for synthetic plants and the linear AE).
- Gram `K = X Xᵀ`. Squared distance `D_ij = K_ii + K_jj − 2K_ij` (= 2 − 2K_ij after L2), clipped below at 0. Plain distance `d = sqrt(D)`. Diagonal is excluded everywhere.
- **Tie rule (global).** Order the candidates j ≠ i by the key (D_ij, j) lexicographically, with the smaller index first on exact equality. MST and Prim use the same key on (D_ij, min(i,j), max(i,j)). Report the **near-tie fraction**: the fraction of anchors whose k-th and (k+1)-th distances differ by < 1e-12 × median(D).

## 1. Structural observables (exact-equality discrepancies; all in [0,1])

1. **kNN sets.** N_k(i) = the first k candidates under the tie rule. Mutual overlap `s_N(k) = (1/(m k)) Σ_i |N_k^A(i) ∩ N_k^B(i)|`; discrepancy Δ_N = 1 − s_N.
   - Grid: k ∈ {1, 3, 10, 30, 100} plus k_q = max(1, round(0.01·(m−1))).
   - Chance level: k/(m−1).
2. **kNN-graph components.** Undirected graph with edge {i,j} iff j ∈ N_k(i) or i ∈ N_k(j). Discrepancy = Rand distance: the fraction of unordered pairs whose same-component indicator differs. Also report the number of components in each model and the flag `degenerate = both graphs connected`. k ∈ {1, 3, 10}.
3. **Per-anchor ranking.** For each anchor i, the Kendall distance between the orderings of {D_A(i,j)}_{j≠i} and {D_B(i,j)}_{j≠i}: the fraction of discordant pairs (j,l), with ties counted as ½ (equivalently (1 − τ_b)/2 on rows without ties). Δ_O is the mean over anchors. Use an O(m log m) algorithm per row (scipy.stats.kendalltau with variant "b" is acceptable if the identity Δ = (1−τ_b)/2 is verified on tie-free rows).
4. **Global edge order.** Kendall τ_b and Spearman ρ between the upper-triangular entries of D_A and D_B; Δ_E = (1 − τ_b)/2.
5. **MST.** Minimum spanning tree of the complete graph weighted by D (identical to the tree weighted by d, since the map is monotone). Use dense Prim from node 0 with the tie key above.
   - Report `s_T = |T_A ∩ T_B| / (m−1)`.
   - **Forest/bridge split.** F_A is the set of undirected 1-NN edges of A (the 1-NN forest, ⊆ T_A). Report:
     - `forest_share_A = |F_A| / (m−1)`;
     - `s_T_forest = |T_A ∩ T_B ∩ F_A| / |F_A|`;
     - `s_T_bridge = |(T_A ∖ F_A) ∩ T_B| / |T_A ∖ F_A|`;
     - `s_N(1)`.
   - Sanity check (must hold exactly): `F_A ⊆ T_A`.

## 2. Geometric criteria

6. **Scale-normalized geometry.** ν(D) = mean off-diagonal D. Report `Gs_rms = rms(D_A/ν_A − D_B/ν_B)` and `Gs_max = max|·|` over off-diagonal pairs.
7. **Raw geometry** (only when preprocessing is identical, e.g. both L2). Report `G_rms = rms(D_A − D_B)`, `G_max = max|D_A − D_B|`.
8. **Linear CKA.**
   - Biased: HSIC(K_A, K_B) = tr(K̄_A K̄_B) with K̄ = HKH; CKA = HSIC_AB / sqrt(HSIC_AA · HSIC_BB).
   - Debiased: the unbiased HSIC estimator of Song et al. 2012, which is what ARH calls unbiased CKA. Match the vendored ARH implementation to ≤ 1e-8.
9. **Pearson RSA.** Pearson correlation of the upper-triangular entries of D_A and D_B. This is affine-invariant.
10. **SOG distance-aware overlaps.**
    - d̃ = d / Q_0.9(positive off-diagonal d). Use plain distance d unless the pinned SOG code (commit `be654a37`, github junwon0/what-converges-prh) uses another quantity; record the check.
    - Weight: `w_ij(τ) = exp(−|log d̃_A(i,j) − log d̃_B(i,j)|/τ)`.
    - kNN version (k=10): `(1/(m k)) Σ_i Σ_{j ∈ N_A(i) ∩ N_B(i)} w_ij`.
    - MST version: `(1/(m−1)) Σ_{e ∈ T_A ∩ T_B} w_e`.
    - τ ∈ {1, 0.1, 0.01, 0.001, 0.0001}.

## 3. Quotient (warp) decomposition

Done on the upper-triangular pairs of D_A (x) and D_B (y).

- Total discrepancy: `SS0 = Σ (y − x)²`.
- Quotients, each giving a residual r and an explained fraction `1 − Σr²/SS0`:
  - (a) scale: y ≈ c·x;
  - (b) affine: y ≈ a·x + b;
  - (c) global isotonic: y ≈ g(x), g non-decreasing (`scipy.optimize.isotonic_regression`);
  - (d) row isotonic: for each anchor i, y_i· ≈ g_i(x_i·) over j ≠ i.
- Report the residual RMS and max for each quotient.
- **Cross-fitting.** Split anchors by a fixed hash into halves H1/H2. Fit on pairs with both endpoints in H1; evaluate on pairs with both endpoints in H2, with g extended by constant extrapolation and linear interpolation between fitted knots. Report both in-sample and cross-fitted values.
- Report in both directions (A→B and B→A) and their mean.

## 4. GL-invariant criterion (linear models only)

11. **Projector distance.** `P = X (XᵀX)⁺ Xᵀ` on centered X (m × m). Report `‖P_A − P_B‖_F² / (2 d)` ∈ [0,1]; require d < m. Also report mean squared canonical correlation (CCA) as a cross-check.

## 5. Permutation calibration (ARH)

- Permute the rows of B by π; recompute the score. K = 500 permutations, α = 0.05.
- Threshold: c_α is the ⌈(1−α)(K+1)⌉-th order statistic of {s_obs} ∪ {s_π}.
- `s_cal = max((s_obs − c_α)/(s_max − c_α), 0)`, with s_max = 1.
- Also report the add-one p-value and the **uncensored** quantity `s_obs − mean(s_π)`.
- If a layer-max aggregate is used, the same π is applied to every layer and the max is recomputed inside each permutation.
- Isotonic quotients are refit inside each permutation.
- Seeds are derived from sha256 of a string key; no global RNG state.

## 6. Mechanism readouts (Stage 1 predictors; validated on plants before use on real data)

- **Crossing-law predictor μ̂ (P1).** For anchor i, estimate the conditional coupling of b_l = D_B(i,l) given u_l = D_A(i,l), using ONLY data disjoint from the predicted quantity:
  - pairs (u, b) from other anchors (other half), pooled by rank-normalized u;
  - or the same anchor at a different m.
  
  Then find t* with E X(t*) = E Y(t*) under the estimated conditional law and set μ̂_i = Σ_{j∈N_A(i)} P̂(b_j > t*). Predicted `1 − s_N = mean_i μ̂_i / k`. The exact estimator is chosen on plants (G0) and on the S_sel/fit half, then frozen before the evaluation half.
- **Dispersion law (P2).** 1 − s_N ≈ (p̂/2)·mean_i δ̂_i(r_k), with p̂ the Levina–Bickel MLE intrinsic dimension at scale k (bias calibrated on plants with known p) and δ̂ the relative dispersion of the row-isotonic residual at radius r_k.
- **Null predictors.**
  - N0: s_N constant in m (equal to the fit-m value).
  - N1: independent additive noise, with σ estimated as the global RMS of the affine residual.
  - N2: Gaussian copula with constant ρ_i (known to fail on anisotropic plants; kept as a null only).
- **Readouts.** Also report the forest share and the warp shares from §3.

## 7. Known-answer plants (gate G0; must pass before any real-data analysis)

Point clouds: m ∈ {256, 1024, 4096}, intrinsic dimension p ∈ {2, 4, 8}, embedded in ℝ^64 via a random isometry.

| Plant | Requirement |
|---|---|
| P-Bern: K_B = exp(β(K_A − 1)) entrywise, β ∈ {1, 3, 10} | Every ordinal rung (kNN all k, components, Δ_O, Δ_E, MST) is **identical** (Δ = 0 exactly). CKA follows a monotone decreasing path. Global-isotonic explained fraction ≥ 0.999. |
| P-Aniso: B = A·M, M = diag(1,2,1,…) | Fixed-k s_N approximately flat in m (\|slope\| < 0.03 per doubling). |
| P-Conf: smooth conformal map | Fixed-k s_N rising toward 1 with m. |
| P-Rough: shared + private noise | Fixed-k s_N falling with m. |
| P-Gauss: D_B = D_A + iid N(0, σ²) | μ̂ within ±0.02 of observed 1 − s_N for s_N ∈ [0.05, 0.99]. |
| Known-p clouds | Report the Levina–Bickel bias at each (m, p). |

The implementation must report pass/fail for each plant in `results/G0_report.json`.

## 8. Agreement between implementations

On every plant and on the real features, `hrhv2` and `hrhv2_ref` must agree:
- exact equality for set/graph/tree quantities;
- ≤ 1e-10 relative for floating quantities (CKA, RSA, Gs, quotients);
- ≤ 1e-6 for isotonic fits.

On identical features, `hrhv2` must match the vendored PRH/ARH mKNN and CKA and the SOG reference code to ≤ 1e-6.
