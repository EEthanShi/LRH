# Stage 1 preregistration (v1.0, 2026-10-08)

This file is frozen before any real-data metric is computed. Its sha256 is recorded in `prereg/PREREG_stage1.sha256`. Any later change requires a new version with a changelog entry; the previous version stays valid for every analysis already run.

## Thesis under test (theory-led)

For one pair of models, every alignment criterion reads the same cross-model coupling through that criterion's invariance class and scale. Structural criteria (kNN at k, components, per-anchor rankings, MST/H0) are blind to monotone warps. They read the residual dispersion relative to their own margins at their own scale; this is the crossing law. Geometric criteria (CKA, scale-normalized distance, SOG τ-weighting) read the warp at zeroth order and the residual only at second order. ARH's and SOG's contrasting conclusions are therefore points of one criterion lattice evaluated on one coupling.

## Gate G0 (Stage 0)

All plant requirements in SPEC §7 pass, and `hrhv2` agrees with `hrhv2_ref` (SPEC §8). An estimator that fails its plant is not used in Stage 1.

## S1.2 — debate subpanel (WIT-1024)

**Models.**
- LMs (8): BLOOMZ 560m, 1b1, 1b7, 3b, 7b1; OpenLLaMA 3B, 7B; LLaMA-7B (huggyllama).
- ViTs (17): the ARH "val" set.

**Data.** S = `minhuh/prh@wit_1024` (revision 0220f2b7…), caption index 0. S_sel = wit_4096 items 1025–2048, used only for layer and estimator choices.

**Layers.**
- Primary: fixed relative depth 0.5 and 1.0 (last block).
- Secondary: ARH layer-max, calibrated inside permutations. Used for replication only.

**Gate G1′.** On identical features, our mKNN/CKA/calibration match the vendored ARH/PRH code to ≤ 1e-6. If it fails, report it and stop.

**Endpoints.**

- **PE2: MST = forest + bridges.** Primary layer, k=10, pre="prh_clip+l2".
  - "SOG-global ≈ 1-NN" if all hold:
    - forest share of shared MST edges ≥ 0.70 in ≥ 90% of pairs;
    - |s_T − s_N(1)| ≤ 0.03 in ≥ 80% of pairs;
    - within-family trend of bridge overlap not significant, or |ρ| ≤ 0.3.
  - "Genuine global convergence" if both hold:
    - bridge-overlap within-family Spearman ≥ 0.5 with exact permutation p < 0.05;
    - bridge overlap exceeds its own permutation null in ≥ 50% of pairs.
  - Otherwise "mixed".
  - Power gate: if the expected number of shared bridges per pair is < 10, re-run at m = 3072 (wit_4096 minus S_sel) and say so.
- **PE3: width-equalized CKA.** Project each model to its top-128 PCs (fit on S_sel); compute calibrated CKA trend Tr_eq (within-family Spearman vs LM size).
  - "Width artifact" if Tr_eq − Tr_raw ≥ 0.3 and Tr_eq ≥ 0.6·Tr_mKNN.
  - "Genuine global loss" if Tr_eq − Tr_raw ≤ 0.1.
- **Diagnostics (not confirmatory).**
  - Warp shares: scale, affine, global isotonic, row isotonic.
  - "Global monotone warp explains geometry" is declared refuted if the median (global-isotonic − affine) explained fraction is < 0.10.
  - Trend decomposition Δlog(1−s_N) ≈ Δlog p̂ + Δlog δ̂(r_k).
  - Certified fraction of v2 eq. (6) (expected ≈ 0).

**Statistics.** Two-way cluster bootstrap over LMs × ViTs (2000 draws); exact within-family permutation for trends; uncensored calibrated scores (s_obs − mean null) for trends; Holm correction across PE2–PE3.

## S1.1 — out-of-sample (m, k) test of the crossing law (kill-or-go gate G1)

**Pool.** `sayakpaul/coco-30-val-2014` (COCO val2014 images with captions), first caption per image, deduplicated by image id. A fixed sha256-hashed permutation of item ids defines nested pools m ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}; 5 nested draws use 5 hash salts. Anchors are split into halves H1/H2 by hash.

**Models** (fixed relative depth 0.5; fp32).
- Vision: DINOv2-B, OpenCLIP ViT-B/16 laion2b, MAE-B, AugReg ViT-B/16 in21k, SigLIP-B/16.
- Text: PolyPythias-410m seeds 1–9, BLOOMZ-1b7, OpenLLaMA-3B.

**Pair classes.**
- Seed–seed: Pythia-410m; 4 disjoint pairs primary, all 36 secondary.
- Cross-objective vision–vision: 10 pairs.
- Cross-modal: 15 pairs.

**Procedure.** Fit the coupling estimator (frozen after G0) on H1 at m = 1024. Predict s_N on H2 at all other m and at k ∈ {1, 10, 100}. Predictors: P1 (μ̂) and P2 (dispersion law); nulls N0, N1, N2.

**Confirm** (all required):
1. Median |log(pred/obs)| ≤ log 1.15 over cells with obs > 2× chance.
2. On pairs with |observed fixed-k log-log slope| ≥ 0.1, MSLE(P1) ≤ 0.5 · min(MSLE(N0), MSLE(N1)).
3. Slope sign correct in ≥ 80% of those pairs.

**Secondary.** Ordering of fixed-k slopes: seed–seed ≥ cross-vision ≥ cross-modal; k-elasticity ≈ −m-elasticity.

**Outcomes.**
- If (1) or (2) fails: the crossing law is reported as descriptive only, and the paper's lead result shifts to the lattice/link theorems.
- If no pair has |slope| ≥ 0.1: record "only smooth class observed".

## S1.3a — certified linear-AE GL instance (CPU)

**Data.** x ∈ ℝ¹⁶ ~ N(0, U Λ Uᵀ), Λ = (4, 3, 2, 1.5, 0.2 × 12), U fixed by hash. Latent d = 4. Probes: m = 256 held-out samples.

**Arms.**
- P-GD: full batch, lr 0.05.
- P-Adam.
- P-SGD: batch 16.
- WD: λ = 0.05·λ₄.
- TIED: W = Eᵀ.
- small-init: σ₀ = 0.01.

All arms use default init N(0, 1/fan_in) unless noted. Training runs to a stationarity certificate: gradient norm ≤ 1e-8 relative, or the PCA-optimal loss gap ≤ 1e-8.

**Sizes.** N ∈ {256, 4096, 65536} plus N = ∞ (exact population covariance, 1000 inits).

**Runs.** 96 runs per (arm, N), i.e. 48 disjoint A/B pairs.

**Criteria.**
- kNN (k=10, ε_N = 0.05), Gs_rms (ε = 0.02), CKA (ε = 0.01), projector distance (ε = 0.01).
- Decision rule: one-sided 95% Clopper–Pearson. Reliable if the lower bound > 0.9; unreliable if the upper bound < 0.8; otherwise inconclusive.

**Frozen predictions.**
1. P-GD: kNN, Gs and CKA unreliable at every N; the largest-N mean is inside the 95% CI of the N = ∞ plateau.
2. WD, TIED and small-init: reliable at the largest N.
3. Projector criterion: reliable in every arm at the largest N.
4. P-SGD and P-Adam: no prediction. The outcome answers whether minibatch SGD self-balances (Ziyin & Chuang 2025); both outcomes are reported.

## Hygiene (all experiments)

- float64 metrics; the SPEC tie rule; fp32 features.
- LMs at batch size 1, so there is no padding.
- No run is dropped. Pilots are excluded and labeled. Seeds are sha256-derived.
- Every number in the paper is recomputed from raw saved features by a script in this directory.
