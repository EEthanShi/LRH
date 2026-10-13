# S1.2 debate subpanel: analysis plan (operationalization of PREREG v1.0)

Role: S1.2 analyst. This plan was written before the S1.2 panel was computed.

**Disclosure.** One timing smoke test ran first: LLaMA-7B vs DINOv2-L, r=0.5, K=40, nothing saved. It showed 2 shared A-bridges at m=1024, so the power gate is likely to fire. No choice below depends on it: every choice is either dictated by the PREREG or fixed for the reasons given.

The plan's sha256 is logged in `STATUS.md`. The PREREG (`prereg/PREREG_stage1.md`, sha256 `bee0583c…`) and SPEC (`e781e44f…`) are unchanged. Where the PREREG leaves a choice open, the choice is fixed below. Anything not in the PREREG is labelled **exploratory** in the report.

## 0. Gate status at start

- The verifier's verdict is **YES-with-caveats** for S1.2. The caveats it attaches:
  - **D1 and D2 are not yet recorded by the lead.** D1 is the P-Bern cross-fitted isotonic gate; D2 is P1 at k=1. S1.2 uses no P1 or other §6 predictor as an endpoint.
  - **Consequence of D1 for this plan.** The warp diagnostic below is reported two ways:
    - in-sample (this reading passes P-Bern);
    - cross-fitted (this reading fails P-Bern by constant extrapolation), flagged as such.
  - **D3.** G1′ is read on float64 inputs. Anchors whose k-boundary is an exact tie are reported, not dropped. The only case is WIT items 493 and 531, which have identical captions and therefore identical LM features.
  - **D5.** The clip scope is pinned to all stored layers per (model, dataset). The duplicate pair is kept, because the PREREG says "no run is dropped", and it is reported.
  - **D6.** Whose 1-NN forest PE2 refers to is fixed in §3.
- **G1′.**
  - The verifier's float64 identity evidence (`results/verification/vendor_identity.json`) is reused.
  - It is re-checked end-to-end through this role's pipeline on 3 LM–ViT pairs (`analysis/s12/g1prime.py`).

## 1. Data, models, layers, preprocessing

- **Data.**
  - S = WIT-1024: m = 1024, all rows.
  - S_sel = wit_4096 rows 1024–2047, minus row 953 (it overlaps S by caption): 1023 rows. S_sel is used only for:
    - the PE3 PCA fits;
    - the PE2 power-gate fallback (§3).
- **Models.**
  - 8 LMs, with mean-pooled hidden states.
  - 17 ViTs, with the CLS token at each block output.
  - Both feature types are as stored by the extraction role.
- **Pair orientation.** A = LM, B = ViT. The permutation therefore acts on the ViT rows.
- **Layer rule (task spec).** Stored index = floor(r·(L_stored−1) + 0.5), r ∈ {0.5, 1.0}.
  - LMs store hidden_states[0..L] including the embedding. The rule picks hidden_states[L/2] and the final post-norm state.
  - ViTs store block outputs 0..L−1. The rule picks stored index round((L−1)/2) and the last block. For ViT-B the r=0.5 choice is block 6 of 0..11.
  - The extraction sidecar rule (the verifier's) agrees for LMs and at r=1.0. For ViTs at r=0.5 it is one block earlier. An **exploratory** sensitivity run uses the sidecar rule at r=0.5.
  - Both depths are primary (PREREG), so PE2 and PE3 verdicts are evaluated at each depth. If the two depths disagree, the overall verdict is "depth-dependent".
- **Preprocessing.**
  - Primary `prh_clip+l2`: the clamp threshold is computed over all stored layers of every sample of the dataset or pool (= ARH/PRH/SOG), then L2 per layer.
  - Secondary `l2`.
  - Checked bitwise identical to `hrhv2.prep.Geom.from_features(clip_ref=stack)`.

## 2. Per-pair quantities (136 pairs × 2 depths × 2 pre)

- **Ladder.** Full `hrhv2.ladder` with the SPEC default grids:
  - s_N at k ∈ {1, 3, 10, 30, 100} (k_q = 10);
  - components at k = 1, 3, 10;
  - Δ_O;
  - Kendall τ_b and Spearman of the global edge order (exact);
  - MST with the forest/bridge split;
  - Gs and raw G;
  - CKA, biased and debiased;
  - Pearson RSA;
  - SOG τ-weighted kNN (k=10) and MST overlaps;
  - quotients: scale, affine, global isotonic and row isotonic, each in-sample and cross-fitted (H1/H2 = hash of the row index), in both directions.

  The GL criterion (§4, linear models only) is not computed.
- **Certified fraction of v2 eq. (6)**, at k = 10 and k = 1. Anchor i is certified if γ_i > 2e_i, where:
  - γ_i = D_(k+1) − D_(k) of A's row;
  - e_i = max_j |D_A(i,j) − D_B(i,j)|.

  Variants:
  - symmetrised gap, max(γ_A, γ_B);
  - scale-normalised, D/ν;
  - row-isotonic quotient, e_i^iso = min over increasing f of max_j |D_A − f(D_B)|, in both directions;
  - "any" = certified by at least one variant.
- **Dispersion-law ingredients** (descriptive, in-sample, all anchors, k = 10), computed for both references (A = LM and B = ViT):
  - p̂ = Levina–Bickel MLE (MacKay–Ghahramani average) on the reference model's rows;
  - δ̂_D = mean_i MAD_iso(D_k,i)/D_k,i, from `hrhv2.predict.fit_iso_dispersion` with the fit mask set to all items.
- **PE3 width-equalised CKA.**
  1. For each model, depth and pre, take the top-128 PCs of the preprocessed S_sel rows, centred on the S_sel mean.
  2. Project the preprocessed WIT-1024 rows onto them.
  3. Compute linear CKA (biased and debiased) on the projections, with pre = none.
- **Permutation calibration (SPEC §5).**
  - K = 500, α = 0.05. π_k = `hrhv2.calibrate.permutation(m, "S1.2|<run>|<lm>|<vit>|r=<r>", k)`, the same π for every score and both pre.
  - Scores calibrated:
    - s_N at every k;
    - s_T;
    - s_T_bridge (A's bridges, as in SPEC), s_T_bridge_B, and their mean;
    - CKA biased and debiased, raw and width-equalised.
  - Recorded per score: s_cal, c_α, add-one p, null mean, and uncensored = s_obs − mean null.
  - **Fast exact path.** The kNN tables and MST of permuted B are relabellings of B's. This equals SPEC recomputation whenever every off-diagonal D_B value is distinct. That is checked per configuration; if it fails, every permutation is recomputed. CKA uses P(HKH)Pᵀ.
  - **Spot check.** In every configuration the first 2 permutations are also recomputed from scratch (knn_table, mst_prim, cka_pair on the permuted Geom). The observed fast-path scores must equal the ladder's.

## 3. PE2 (MST = forest + bridges): primary k=10, pre `prh_clip+l2`, at each depth

- **D6 reading, fixed here.**
  - **Primary (SPEC-literal).** F_A is the 1-NN forest of A = LM.
    - Forest share of shared MST edges = |T_A∩T_B∩F_A| / |T_A∩T_B|.
    - Bridge overlap = SPEC `s_T_bridge` = |(T_A∖F_A)∩T_B| / |T_A∖F_A|.
  - **Sensitivity readings**, each with the full rule re-evaluated:
    - B's forest, with `s_T_bridge_B`;
    - the union F_A ∪ F_B, with bridge overlap = the mean of both directions;
    - the intersection F_A ∩ F_B (forest share only).
- **Conditions.**
  - (a) Forest share ≥ 0.70 in ≥ 90% of pairs. A pair with no shared MST edge counts as failing.
  - (b) |s_T − s_N(1)| ≤ 0.03 in ≥ 80% of pairs.
  - (c) The within-family trend Tr of bridge overlap is not significant (Holm-adjusted two-sided exact p ≥ 0.05), or |Tr| ≤ 0.3.
  - (d) Tr ≥ 0.5 with Holm-adjusted one-sided exact p < 0.05.
  - (e) Bridge overlap exceeds its own permutation null (s_obs > c_α, i.e. s_cal > 0) in ≥ 50% of pairs.
- **Verdict.**
  - "SOG-global ≈ 1-NN" if (a), (b) and (c) all hold.
  - "Genuine global convergence" if (d) and (e) both hold.
  - Otherwise "mixed".
- **Power gate.**
  - Statistic: the mean over the 136 pairs of the number of shared bridges |(T_A∖F_A)∩T_B| at that depth, primary pre.
  - **It fires if < 10.**
  - **Deviation.** The PREREG re-run at m = 3072 (wit_4096 minus S_sel) is impossible here:
    - only S (1024 rows) and S_sel (1023 usable rows) are extracted;
    - the wit_4096 rows outside S_sel carry about 1024 S captions;
    - this role must not use the GPU.

    The re-run instead uses the largest available disjoint pool: m = 2047 = S ∪ (S_sel∖{row 953}).
    - The clip threshold is recomputed on that pool.
    - Only the PE2 keys and their calibrations are computed.
    - The PE2 verdict is reported at both m = 1024 and m = 2047. The gate designates m = 2047.

## 4. PE3 (width-equalised CKA)

- **Scores.** Uncensored calibrated scores.
  - Primary CKA: biased linear CKA (PRH/ARH `cka`).
  - Secondary CKA: debiased.
  - Tr_mKNN uses s_N(10).
- **Trend Tr.** For each of the 17 ViTs, take the Spearman ρ between LM size (num_params) and score within BLOOMZ (5 sizes); Tr is the mean over ViTs.
  - Exact within-family permutation p: all 120 relabellings of the BLOOMZ sizes, with the same relabelling applied to every ViT.
  - OpenLLaMA (3B, 7B) is a secondary sign check only.
  - LLaMA-7B is a singleton.
  - These limitations are stated in the report.
- **Verdict.**
  - "Width artifact" if Tr_eq − Tr_raw ≥ 0.3 and Tr_eq ≥ 0.6·Tr_mKNN.
  - "Genuine global loss" if Tr_eq − Tr_raw ≤ 0.1.
  - Otherwise "inconclusive (between branches)".
- **Holm-family test.** One-sided exact p of Tr_eq − Tr_raw > 0.

## 5. Statistics

- **Vision-side trends (secondary).**
  - Statistic: for each LM and each vision family, take the Spearman ρ against ViT num_params; average over the (LM, family) cells. Families: AugReg 4, MAE 3, DINOv2 4, CLIP-laion2b 3, CLIP-ft 3.
  - Exact permutation over all 24·6·6·6·24 within-family relabellings.
- **Holm family.**
  - Members: the PE2 bridge-trend p and the PE3 (Tr_eq − Tr_raw) p, at r = 0.5 and r = 1.0. That is 4 tests, Holm at 0.05.
  - The PE2 p comes from the run the power gate designates.
  - (c) uses two-sided p and (d) uses one-sided p. Each sidedness is adjusted as its own family.
- **CIs.**
  - Pair-level summaries (fractions of pairs, medians, means): two-way cluster bootstrap over LMs × ViTs. LMs and ViTs are resampled independently with replacement over the product grid. 2000 draws, seed sha256("S1.2|boot|<stat>"), percentile 95%.
  - Tr (LM side): resample ViTs, because LM size is the trend axis.
  - Vision-side Tr: resample LMs.
- **Diagnostics (not confirmatory).**
  - **Warp shares:** scale, affine-extra, iso-extra, row-extra and residual, each in-sample and cross-fitted.
  - **Refutation rule:** "global monotone warp explains geometry" is refuted if the median (global iso − affine) explained fraction is < 0.10, applied in each reading. Exploratory addition: the iso gain as a share of the post-affine residual.
  - **Trend decomposition within BLOOMZ, per ViT:** OLS slopes against log(num_params) of log(1−s_N(10)), log p̂ and log δ̂_D, plus the closure residual. Done for both references.
  - **Certified fraction of eq. (6).**

## 6. Not run

- ARH layer-max is not run (secondary, replication only, per the PREREG), for lack of time.
