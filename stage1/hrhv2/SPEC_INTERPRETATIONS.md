# hrhv2: interpretations of SPEC v1.0 (harness-main)

SPEC.md is frozen. Where the text leaves a choice open, `hrhv2` makes the choice below. None of these choices changes a SPEC definition; each one is covered by a unit test in `hrhv2/tests/`. A formal deviation would go to SPEC_CHANGELOG.md, which is owned by the lead. The independent reference implementation (`hrhv2_ref`) may have chosen differently. Every item below is therefore a candidate cause if an agreement check under SPEC section 8 fails.

## Section 0: inputs
- **`prh_clip+l2` on stacked layers.** With an (m, L, d) array, the clamp threshold is computed per sample over all L·d coordinates and then averaged, which is what PRH/ARH `remove_outliers` do. `run_pair.py --clip-scope layer` uses the selected layer only. Vendored ARH casts to float32 before computing the threshold; hrhv2 works in float64. On float64 input the match is exact (test_prep).
- **Near-tie fraction.** median(D) is the median of the strict upper triangle.

## Section 1: structural observables
- **Kendall distance with ties.** A pair counts ½ when it is tied in exactly one ordering and 0 when it is tied in both (a joint tie is agreement). On tie-free rows this equals (1 − τ_b)/2 exactly.
- **τ_b.** Computed from exact integer pair counts, with the denominator also formed in integer arithmetic. As a result, identical orderings give τ_b = 1.0 and Δ_E = 0 exactly; plain `sqrt·sqrt` does not guarantee this.
- **Global edge order.** Exact on all m(m−1)/2 pairs by default. `ladder(..., global_order_max_pairs=N)` enables Bernoulli pair subsampling. This is a non-SPEC speed option and is flagged in the output (`globalorder_subsampled`).
- **MST.** Dense Prim from node 0 with the key (D, min, max). `F_A ⊆ T_A` is asserted in every ladder call.

## Section 2: geometric criteria
- **Debiased CKA.** Uses the Song et al. unbiased HSIC. For distance-only inputs (`Geom.from_dist`, used only for plants) the Gram matrix is −½HDH. The unbiased HSIC is invariant to K_ij → K_ij + a_i + a_j, so this equals the Gram-based value (tested). `cka_debiased_arheps` adds ARH's EPS = 1e-8 to the denominator, for comparison with ARH.
- **SOG.** Uses plain distance d = sqrt(D). Q_0.9 is taken over the positive off-diagonal entries of both triangles with linear interpolation; the result equals SOG `_normalize_distance_array` / `_quantile_normalize`. Normalized distances are clamped at 1e-12 before the log, as in the SOG code. The overlaps match the vendored SOG kNN and MST functions to 1e-12 on identical supports.

## Section 3: quotients
- **Row-isotonic explained fraction.** Both sums run over ordered pairs, so each pair counts once from each anchor.
- **Row-isotonic cross-fitting.** SPEC's "pairs with both endpoints in H1 / H2" cannot apply to a per-anchor fit, because the anchor is an endpoint of every pair in its row. For every anchor i, g_i is therefore fitted on the columns j ∈ H1∖{i} and evaluated on the columns l ∈ H2∖{i}. The global quotients follow SPEC literally: fit on H1×H1, evaluate on H2×H2.
- **Cross-fit with too few pairs** (fewer than 2 fit pairs or no evaluation pair) is reported as NaN.

## Section 4: GL criterion
- ‖P_A − P_B‖²_F is divided by (r_A + r_B), which equals 2d when both ranks are full. The rank uses the numpy `matrix_rank` tolerance.

## Section 5: calibration
- **Threshold index.** ⌈(1−α)(K+1)⌉ is computed in exact rational arithmetic from the decimal α; for example K = 19, α = 0.05 gives 19, where floating point would give 20. ARH instead uses `np.quantile(null ∪ {s_obs}, 1−α)` with interpolation; that value is reported as `c_alpha_arh_npquantile`, for reference only.
- **`calibrate_knn`** permutes the precomputed neighbour tables. It equals full recomputation whenever the near-tie fraction is 0 (tested).

## Section 6: predictors
- **P1 data separation.** P1 is fitted only on pairs with both endpoints in H1 at the fit m (H1×H1) and predicts only H2 anchors. Tests check that no other entry of D_B, and no entry of D_A outside H1×H1, changes the fit.
- **"Pooled by rank-normalized u".** This phrase admits more than one implementation; three readings were compared on plants (see PREDICTOR_CHOICE.md). The frozen choice is recorded there.
- **μ is computed exactly as SPEC states**, using the balance threshold t*. The exact expected discrepancy E|N_A∖N_B| under the same fitted coupling is reported separately, as the diagnostic `P1x` (it is not P1). This shows where the μ-law approximation, rather than the coupling estimator, causes error. Theorem 2 allows a gap of up to √(μ/2).
- **N0** is s_N of the H1 anchors at the fit m, using their full rows.
- **N1** is a Gaussian law with σ_A = RMS(affine residual on H1×H1) / slope.
- **N2** is the per-anchor Gaussian copula. ρ_i is fitted by truncated-bulk profile likelihood on a grid.
- **P2.** p̂ is the MacKay–Ghahramani average of the Levina–Bickel MLE at scale max(k, 3). δ is the in-sample row-isotonic residual in A-units, divided by the local band slope. This in-sample residual is biased low; the size of the bias is reported in G0.

## Section 7: plants and gate operationalization (fixed before the G0 run)
- **Point cloud.** Uniform on the flat torus [0,1)^p, Clifford-embedded in R^{2p} with local metric equal to the identity, then mapped into R^64 by a random isometry. Nested pools are prefixes of one m = 4096 cloud. H1/H2 is the hash of the item index (salt `hrhv2-halves`).
- **P-Bern.** Runs on a spherical cap (torus lifted by a constant coordinate, then L2-normalized; max D_A ≈ 0.16). This keeps D_A → D_B strictly increasing in float64 for β ≤ 10; collisions are counted in every cell.
- **P-Aniso.** "M = diag(1,2,1,…)" scales the first intrinsic circle by 2, which is a linear map on the features.
- **P-Conf.** Sphere inversion about a centre at twice the torus radius.
- **P-Rough.** Private Gaussian noise in 32 dimensions with variance s² = 0.3 · median D_(10) at m = 256.
- **P-Gauss.** σ = c · median D_(10) of an m = 1024 reference cloud, with c ∈ {0.02, 0.05, 0.15, 0.4}.
- **Slope gates.** Fixed k ∈ {1, 3, 10}, with s_N averaged over replicate clouds on all anchors. "Approximately flat" means |OLS slope of s_N on log2 m| < 0.03 over m ∈ {256, 1024, 4096}. "Rising" and "falling" mean strict monotonicity over the three values of m.
- **P-Gauss gate.** Every cell with mean observed s_N ∈ [0.05, 0.99], over k ∈ {1, 3, 10, 30, 41, 100}, must satisfy |mean P1 − mean observed| ≤ 0.02. P1 is fitted on H1×H1 at m = 1024 and evaluated on the H2 anchors at each m.
