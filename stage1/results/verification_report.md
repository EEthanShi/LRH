# HRH v2 Stage 0/1: verifier report (adversarial)

2026-10-08 · role: verifier · project `~/hrh_v2_stage1_20261008`

The run was interrupted by a usage limit. On resuming, I found no earlier verifier output on the server, so everything below was produced in this session. Paths in this report are relative to the project directory unless stated otherwise. Scripts and raw outputs are in `results/verification/`.

## 1. Verdict

**YES-with-caveats** for real-data analyses built on the metric ladder (S1.2), once the lead records decisions D1 and D2. **NO** for S1.1 with P1 as currently frozen.

**What I checked.** For every SPEC criterion, hrhv2 and hrhv2_ref were given identical inputs.
- After one numerical fix (F1, below), they agree on 106 of 106 such cases: 14,385 comparisons, 0 outside SPEC §8 tolerance.
- The distance matrices the two implementations build are bitwise identical in every one of these cases.
- A brute-force recomputation of every set and tree quantity agrees with both implementations in all 53 cases where it was run.
- Given float64 inputs, hrhv2 reproduces the pinned PRH, ARH and SOG code. The only exceptions come from one pair of exactly duplicated WIT items, explained in §3.

**What blocks.**
- **G0 still fails as reported**, on P-Bern under the cross-fitted reading and on P-Gauss at k = 1. I confirmed both causes independently.
- **SPEC §7 requires G0 to pass before any real-data analysis.** The lead therefore has to record decisions D1 and D2 (§6) before S1.2 starts.
- **The frozen P1 estimator fails on real data.** I checked it on S_sel, the WIT split the PREREG reserves for estimator choices, at k = 1 and k = 10:
  - on the ViT–ViT and LM–LM pairs, observed s_N is 0.13–0.48 and P1 predicts at most 0.035;
  - on the LM–ViT pair, observed s_N is 0.00–0.085 and P1 predicts at most 0.023.

  SPEC §6 requires the estimator to be chosen on plants *and* on S_sel. Only plants were used.
- **The nulls are also unfit on these pairs.**
  - N1 returns NaN when the global affine slope is ≤ 0.
  - P2 clips to 0 in 52 of 54 cells.

Run as frozen, S1.1 would almost certainly end with "crossing law descriptive only". That outcome is already visible on S_sel.

## 2. Cross-implementation agreement (SPEC §8)

Driver: `agree.py`. Final run, after fix F1: `agree_postF1/` (123 cases) and `agree_summary_postF1.json`.

In every case both implementations received the same input: features with a given `pre`, a Gram, or a squared-distance matrix. They also used the same H1/H2 split, since hrhv2's hash split is passed to ref as `H1=`. For `prh_clip+l2`, each implementation computed the clip threshold with its own code; the two thresholds were bitwise equal.

Tolerances: exact equality for set, graph and tree quantities; ≤ 1e-10 relative (with a 1e-13 floor) for floats; ≤ 1e-6 absolute for isotonic fits.

| Case family | Cases | Result |
|---|---|---|
| G0 plants at m ∈ {256, 1024}, unclipped as G0 uses them: P-Bern (3 p × 3 β), P-Aniso, P-Conf, P-Rough (3 p each), P-Gauss (3 p × 4 c), WarpShare | 62 | 45 identical; the other 17 differ only by convention D1 (below) |
| P-Gauss and WarpShare given the same pre-clipped D | 26 | identical |
| P-Bern on the full sphere at β = 10 (float collisions in D_B) | 2 | identical |
| Random heavy-tailed pairs, each `pre` | 6 | identical |
| (m, 3, d) stacks with the all-layer clip threshold, each `pre` | 6 | identical |
| Near-tie stress: duplicated rows; integer lattices with 81 or 1,000 distinct points; quantised features; m = 60 (exercises ref's brute-force Kendall) | 9 | identical |
| Real WIT-1024 features: LLaMA-7B vs OpenLLaMA-7B, CLIP-B vs DINOv2-B, BLOOMZ-1b7 vs DINOv2-L; relative depth 0.5 and 1.0; `l2` and `prh_clip+l2` | 12 | identical after F1 (2 float failures before) |

The first row includes all 24 P-Gauss and both WarpShare cases. In 17 of them (15 P-Gauss, 2 WarpShare), D_B has negative entries, which the two implementations treat differently. These 17 are excluded from the 106 agreeing cases; the paragraph on convention D1 below explains them.

**Totals for the 106 cases with identical inputs:**
- 5,485 exact comparisons, 0 failures;
- 5,932 float comparisons, 0 failures, largest relative difference 1.4e-11 (ref's own residual error on debiased CKA);
- 2,968 isotonic comparisons, 0 failures.

The brute-force path recomputes everything from scratch:
- kNN sets from Python-sorted (D_ij, j) lists;
- components by BFS;
- the MST by Kruskal on Python-sorted (D, min, max) keys;
- per-row Kendall by O(n²) sign products.

It ran on every case with m ≤ 256 and every stress case (53 in total), and agreed with both implementations in all 53.

**The one real disagreement was resolved by exact arithmetic (fix F1).** On BLOOMZ-1b7 (hidden state 12) vs DINOv2-L (block 11) with `pre="l2"`, `cka_debiased` differed by 1.19e-9 relative. `exact_cka.py` evaluates Song et al.'s formula on the float64 Grams in exact rational arithmetic. Measured against that value:

| | relative error |
|---|---|
| hrhv2 (before F1) | 1.2e-9 |
| hrhv2_ref | 1.4e-11 |

hrhv2 was the inaccurate one; §4 has the fix.

**One convention differs and is now documented (`SPEC_CHANGELOG.md`, item D1).** When a distance matrix is supplied directly, as for the P-Gauss and WarpShare plants:
- hrhv2 keeps negative entries;
- ref clips them to 0.

Affected: 15 P-Gauss cases and 2 WarpShare cases. Given the same pre-clipped D, the implementations agree in all 26 cases. On the unclipped plants, brute force against the unclipped D agrees with hrhv2 in 16 of 16 checked cases. G0 is right to use hrhv2's convention, because the P-Gauss oracle and the P1 model both assume D_B = D_A + σZ without clipping.

**Thread-count determinism (`threads/compare.json`).** I ran the six real pairs with 1 and with 7 BLAS threads.
- K and D were bitwise identical, and so was every kNN set up to k = 100.
- Only floating-point reductions differ, by ≤ 1e-15 relative.

**Test suites.** `hrhv2` passes 100/100 (one test added). `hrhv2_ref` passes 56/56 and was not changed.

## 3. Identity against the pinned PRH, ARH and SOG code

Driver: `vendor_identity.py`; output: `vendor_identity.json`.

Pinned code:
- PRH `metrics.py` (dcd76ba), with functions extracted by `ast` because its imports are not installable here;
- ARH `utils`, `estimators` and `preprocess` (785054b);
- ARH `calibrated_similarity.calibrate`;
- SOG be654a37, imported as a package.

Inputs: 3 real pairs × 2 depths × {`prh_clip+l2`, `l2`}. Each comparison was run two ways:
- the vendored functions fed hrhv2's own float64 rows (the identity test);
- the vendored pipeline exactly as its authors run it, in float32 (the native test).

### 3.1 Answers to the three SOG questions

1. **Does SOG's log-ratio use plain d or squared D?** Plain chord distance d.
   - In code: `layerwise_engine._knn_distance_matrix_from_layer` computes `sqrt(clamp(2 − 2⟨x,y⟩))`, and `topology_alignment.prepare_mst_overlap_cache` uses `torch.cdist`.
   - Numerically, on float64 rows hrhv2 reproduces SOG to ≤ 1.9e-14 on a pair without duplicate items. A squared-D variant is 11–99% lower.
2. **Does SOG's MST use D or d?** d: Prim on `cdist` after quantile normalisation, with SOG's own tie rule (strict-less update, then argmin).
   - The tree under d is the same as the SPEC tree under D whenever there are no exact ties.
   - The edge sets were identical in all 6 real cases, in float64 and in native float32.
3. **Is SOG's preprocessing equal to `prh_clip+l2`?** Yes, provided the clip threshold is computed over **all layers of the model**.
   - `prepare_features(q=0.95)` is applied to the stacked (n, L, d) tensor: one threshold per model, in float32. `_normalize_layers` then applies L2 to each layer.
   - hrhv2 matches PRH and ARH `remove_outliers` to ≤ 2.2e-16 in float64. The float32 threshold differs by ≤ 1.5e-7 relative.
   - The per-layer scope is a different criterion. For LLaMA-7B layer 16 the threshold is 0.52 instead of 1.24.
   - `run_pair` defaults to the all-layer scope, but `hrhv2_ref` defaults to the per-layer one. Every analysis script must pin the scope explicitly.

No change to hrhv2 was needed for any of these. All three are recorded in `SPEC_CHANGELOG.md`.

### 3.2 Results

| Quantity | float64 inputs (identity) | native float32 vendored pipeline |
|---|---|---|
| mKNN, k ∈ {1, 3, 10, 30, 100}, 60 cells | ≤ 6.7e-8 (the vendored code returns float32), except 1 cell at 3.3e-5 (one neighbour), from the duplicate-item tie below | ≤ 3.6e-8 on ViT–ViT; up to 9.8e-5 (one neighbour) in 5 of 40 LM cells |
| CKA biased (PRH +1e-6 and ARH +1e-8 conventions) | ≤ 6.7e-16 | ≤ 3.4e-7 |
| Debiased CKA: ARH `estimators.cka_debiased` (U-centred) | ≤ 2.1e-11 | ≤ 5.8e-8 |
| Debiased CKA: Song formula with PRH +1e-6 or ARH `CKAUnbiased` +1e-8 | ≤ 2.7e-12 | up to 4.5e-4: the float32 Song formula suffers the same cancellation as before F1 |
| ARH `calibrate()`, K = 500, hrhv2 permutations injected: mKNN at k = 10 and biased CKA on 2 pairs | s_cal, c_α and p agree to ≤ 3.5e-8; null values to ≤ 1.9e-9 | n/a |
| SOG kNN-dist and MST-dist overlaps, τ ∈ {1, …, 1e-4} | ≤ 1.9e-14 (ViT–ViT); ≤ 7e-9 (LM–ViT); 1.95e-4 and 9.8e-4 (LM–LM, the duplicate pair) | ≤ 3.9e-6 absolute on ViT–ViT (2.8–3.6% relative at τ = 1e-4) |
| MST edge sets | identical in all 6 cases | identical in all 6 cases |

**Calibration thresholds.** SPEC's threshold is the ⌈(1−α)(K+1)⌉-th order statistic. ARH/SOG's experiment code uses `compute_null_summary`, i.e. `np.quantile` over the null plus the observed value. At K = 500 and α = 0.05 both give the 476th order statistic, and the two agree to ≤ 9.3e-10.

**Duplicate items.** WIT-1024 items 493 and 531 have the identical caption "Solar eclipse map of path on earth". Every LM therefore gives them bitwise-identical features, and D(i, 493) = D(i, 531) exactly for every anchor i.
- *mKNN.* Wherever the pair falls at positions k and k+1 of an anchor's ranking (checked anchors: 34, 558 and 890 at k = 30; 890 and 940 at k = 10), SPEC's tie rule picks item 493. torch `topk` and `argsort` pick arbitrarily. hrhv2 follows SPEC.
- *SOG weight of the pair.*
  - In exact arithmetic d = 0 in both models, so the weight is 1. This is what hrhv2 and hrhv2_ref give.
  - SOG's float pipeline produces a rounding-noise d of about 1e-8, so its weight is ≈ 0.
- *How far this reaches.* It accounts for every float64 discrepancy in the table above. The COCO pools have 3 duplicate captions at m = 2048 and 49 at m = 16384.

**CKA epsilons.** ARH `CKAUnbiased` adds 1e-8, which moves debiased CKA by up to 2.2e-5 on WIT-1024. PRH's +1e-6 moves it by up to 2.2e-3.
- hrhv2's `cka_debiased` is the exact Song value.
- `cka_debiased_arheps` reproduces the ARH convention.
- ARH's `CKAUnbiased` class source is not vendored anywhere on the server. I therefore checked the reconstruction from `utils.hsic_unbiased`, as hrhv2_ref did.

**Gate G1′ ("our mKNN/CKA/calibration match ARH/PRH to ≤ 1e-6, else stop").** On float64 inputs it passes everywhere except the one duplicate-tie cell. The native float32 pipelines cannot meet 1e-6, because single-neighbour flips are worth 1/(mk) ≈ 1e-4. This needs a recorded reading (§6, D3).

## 4. Adversarial review of hrhv2

| # | Area | Finding | Evidence | Action |
|---|---|---|---|---|
| F1 | Debiased CKA numerics | Song's formula on the raw Gram cancelled heavily: the terms exceeded the result by up to 2.7e5, giving up to 1.2e-9 relative error. The error also depended on the BLAS thread count. | `exact_cka.json`, `hsic_variants.json` | **Fixed.** The HSIC is now evaluated on HKH. This is the same estimator in exact arithmetic, by the U-statistic's invariance to K + a1ᵀ + 1aᵀ. Error is now ≤ 3.1e-15. Test added; logged in `hrhv2/FIXES.md`. |
| A1 | Tie rules, self-exclusion, k off-by-one, normalisations | No defect. | Brute force on 53 cases including the lattice, duplicate and quantised stress cases; exact agreement with ref; hand-check of near-tie indexing and the Levina–Bickel sums | none |
| A2 | Cross-fitting leakage | P1, P2, N1 and N2 are unchanged when I perturb any B entry outside H1×H1 of the fit prefix, i.e. every entry of the target anchors' B-rows and everything outside the prefix. **N0 moves.** It is s_N of the H1 anchors at the fit m over all columns, so it reads the H1×H2 entries, which are part of the target anchors' B-rows. | `adversarial.json` | none. The leak is limited to a null, makes the null stronger, and is therefore conservative for P1. It should be noted in the S1.1 write-up. |
| A3 | Silent subsampling | None in SPEC metrics. The global edge order is exact by default; subsampling is opt-in and flagged. The diagnostic `P1x` returns NaN rather than truncating. μ converges: results at 60 iterations equal those at 400 on real couplings. | grep; `mu_balance` test | none |
| A4 | Nondeterminism | D is bitwise reproducible across thread counts; the ladder differs only at ≤ 1e-15. Forking after BLAS start-up triggers a CPython DeprecationWarning, but no hang was seen. | `threads/compare.json` | none |
| A5 | `calibrate_knn` fast path | With exact ties, permuting precomputed neighbour tables is not the SPEC §5 recomputation. On BLOOMZ (duplicate items) the null differs by one neighbour (1/(mk)) in up to 11 of 100 permutations. c_α and s_cal are unchanged; the uncensored value moves by 3e-7. | `adversarial.json` | Use full recomputation when `neartie_* > 0`, or deduplicate. |
| A6 | Predictors on real couplings (S_sel) | Frozen P1 predicts s_N ≤ 0.035 at k = 1 and 10 where the observed s_N is 0.13–0.48 (ViT–ViT and LM–LM), including at the fit m on held-out anchors. N0 stays within 0.08. At ranks 8–64, the fitted residual scale is 2.7–5.5× the anchors' realised in-row residual. Even the realised rank-1 residual is 1.5–210× the gap between the first and second A-distances, so under column independence it would also imply near-total crossing, yet observed s_N(1) is up to 0.40. This fits a smooth, column-correlated coupling, the same failure as on P-Conf (G0 class MAE 0.48). On BLOOMZ-1b7 L12 vs DINOv2-L L11 the global affine slope is −0.008: there N1 = NaN (σ = ∞), and P1's slope floor of 1e-12 inflates its scale to 1e10–1e12. P2 clips to 0 in 52 of 54 cells. | `p1_real_diag_Ssel.json` | **Not changed**, because `predict.py` and P1 are frozen (PREDICTOR_CHOICE.md). Lead decision D4. |
| A7 | Nested pools with `prh_clip+l2` | `run_pair --m/--rows` recomputes the clip threshold on the selected rows, so D for a fixed pair changes with m. The COCO files store only 4 layers, so the "all-layer" threshold there is not ARH's. | code reading | For S1.1 use `l2`, or fix one threshold in advance (§6, D5). |
| A8 | PE2 "forest share of shared MST edges" | Whose 1-NN forest is meant (A's, B's, or their union) is undefined. hrhv2 reports only A's (`shared_forest_fraction_of_shared`); ref reports A, B and union. | code reading | Fix the definition before S1.2 (§6, D6). |
| A9 | `gl_projector_dist` | Can come out as −4e-16 when the true value is 0. Cosmetic. | agreement output | none |

hrhv2_ref was not changed. I found no clear bug in it; the clip difference is a convention (D1).

## 5. G0: re-run after F1, and independent checks of the two failures

**Re-run.** `results/verification/G0_rerun/`, same seeds (`G0f`), frozen variant `P1-med-B-r8`. Comparison with harness-main's report: `G0_rerun_vs_orig.json`.

The verdict is the same as harness-main's: G0 fails on P-Bern and P-Gauss, and the P1 choice is still `P1-med-B-r8`. The run took 3,272 s on 10 cores.

Of 12,523 numeric leaves, differences fall into three groups:
- **CKA keys:** `cka_debiased`, `cka_debiased_arheps` and `hsicu_*` differ by ≤ 1.3e-10 relative. This is F1.
- **Aggregated means:** other aggregated floats differ by ≤ 2.6e-13. Results arrive from the worker pool in a different order, so replicate means are summed in a different order.
- **Everything else** is unchanged, including every ordinal, set and tree value and every pass/fail flag.

The P-Gauss errors by k are unchanged too: maximum |error| 0.0565 at k = 1, 0.016 at k = 3, and ≤ 0.006 for k ≥ 10.

**P-Gauss at k = 1 is a property of the μ-law, not of the estimator.** `pgauss_check.py` uses code independent of `hrhv2.predict` (scipy `brentq` per anchor, and a 400-draw Monte Carlo of the exact expectation under the same independent law). On the G0f seeds, 4 cells × 4 replicates, with the *true* coupling, measured as observed s_N minus predicted:

| | k = 1 | k = 3 |
|---|---|---|
| μ-law | −0.047 to −0.062 | −0.009 to −0.029 |
| Monte Carlo exact expectation | +0.000 to −0.019 | ≤ 0.015 |

This confirms harness-main's attribution. It also shows the μ-law's own bias at k = 3 is about 0.015, which leaves the k = 3 pass (0.016 against a 0.02 bound) little margin.

**P-Bern cross-fitted isotonic fraction.** Reproduced: 0.99779 (AB) and 0.99777 (BA) at p = 8, m = 256, β = 1.
- Exactly one H2×H2 evaluation pair has D_A = 0.00536, below the H1 fit range, which starts at 0.0129.
- Constant extrapolation at that one point produces 99.99999% of the residual sum of squares.
- The in-sample fraction is 1.0 and every ordinal rung is exactly equal.

## 6. Decisions for the lead

- **D1. P-Bern isotonic gate.** Accept the in-sample reading of SPEC §7, or record the cross-fitted failure. The defect is SPEC's constant extrapolation, not the plant.
- **D2. P1 at k = 1.** Either issue a v1.1 that uses the exact expectation (P1x) for k ≤ 3, or drop k = 1 from S1.1's P1 tests. The PREREG says "an estimator that fails its plant is not used".
- **D3. G1′ and SPEC §8 vendor tolerance.** Read them as "on float64 inputs". Exclude, or deduplicate, anchors whose k-boundary is an exact tie. Report native float32 differences separately.
- **D4. P1 and the nulls on real data (A6).** Before S1.1, either:
  - (a) run S1.1 as frozen and accept the near-certain "descriptive only" result; or
  - (b) issue PREDICTOR_CHOICE/PREREG v1.1 that does the S_sel selection SPEC §6 requires, and define N1 and P1 for slope ≤ 0 (for example, chance level) so they cannot return NaN or absurd scales.

  Do not touch COCO until this is decided.
- **D5. Duplicate items and the clip scope.**
  - Deduplicate identical texts, or keep them and report them: WIT items 493/531; COCO 3 at m = 2048, 49 at m = 16384.
  - Pin `--clip-scope all-layers` for WIT.
  - Use `l2`, or a single fixed threshold, for S1.1's nested pools.
- **D6. PE2 forest share.** Fix A's forest, B's, or their union. If B's or the union is chosen, hrhv2 needs that output key added.

## 7. Files

All under `~/hrh_v2_stage1_20261008/`.

- **Report:** `results/verification_report.md` (this file).
- **Fix:** `hrhv2/FIXES.md` and `SPEC_CHANGELOG.md` (created).
- **Code changed:** `hrhv2/geometric.py` (new sha256 `d957964f…`) and `hrhv2/tests/test_geometric.py`; the pre-fix copies are in `results/verification/backup_hrhv2/`.
- **Unchanged and checked:** `hrhv2/predict.py` still has its frozen sha256 `a17964b4…`. `SPEC.md` (`e781e44f…`) and `prereg/PREREG_stage1.md` (`bee0583c…`) match their recorded hashes.
- **Verification scripts and outputs (`results/verification/`):**
  - `agree.py`, `agree_postF1/`, `agree_summary_postF1.json`, and the pre-fix run `agree/`;
  - `summarize_agree.py`;
  - `exact_cka.py` / `.json`, `hsic_variants.py` / `.json`;
  - `vendor_identity.py` / `.json`;
  - `thread_determinism.py`, `threads/`;
  - `adversarial.py` / `.json`;
  - `pgauss_check.py` / `.json`;
  - `p1_real_diag.py`, `p1_real_diag_Ssel.json`;
  - `compare_g0.py`, `G0_rerun/`, `G0_rerun_vs_orig.json`.
- **Logs:** `logs/verify_*.log`, with `.done` markers.

**Data-use disclosure.** All predictor diagnostics ran on S_sel. One μ-convergence check also printed frozen-P1 predictions for 3 WIT-1024 pairs. That is S1.2 data, and P1 is not an S1.2 endpoint; nothing was chosen from it. No COCO-pool predictions or metrics were computed.
