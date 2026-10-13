# S1.2 secondary: ARH layer-max replication on the 136-pair WIT-1024 subpanel — analysis plan v1

Role: `arh-layermax-early`. Authorisation: `prereg/LEAD_DECISIONS_v1.1.md` (sha256 eaebd687…), section "Additional analysis authorised (S1.2 secondary, preregistered as replication only)". Paths: `analysis/s12_layermax/`, `results/S1_2_layermax/`.

This plan is frozen before any layer-max quantity is computed on real features. Its sha256 is appended to `STATUS.md` before the validation and main runs.

**Disclosure.**
- Before this plan, two GPU timing pilots ran on random tensors only (float64 / float16 batched GEMM and a row/column gather on 1024×1024 matrices). No feature file was opened. They are pilots, excluded, and fixed nothing except the implementation layout.
- I have read the S1.2 fixed-depth report (`results/S1_2/S1_2_report.md`), as the task requires, so the fixed-depth trends (CKA Tr_raw 0.70 / 0.92; mKNN 0.84 / 0.98; vision-axis CKA −0.04 vs mKNN 0.71 at r = 1.0) are known. No layer-max number is known.

## 1. Data, models, layers, preprocessing

- S = WIT-1024, m = 1024, all rows. Items 493 and 531 (same caption, identical LM features) are kept and resolved by the SPEC tie rule (D3/D5).
- 8 LMs × 17 ViTs = 136 pairs; the registry and families are those of `analysis/s12/common.py`. A = LM, B = ViT; permutations act on the ViT rows (as S1.2 and ARH `calibrate(X, Y)` permuting Y).
- **All stored layers.** LMs: hidden_states[0..L] (embedding output, every block, last one post-final-norm). ViTs: every block output, CLS token, no final norm. This is the ARH/PRH layer set. Layer pairs per model pair: L_A·L_B ∈ [300, 1320]; panel total 80,960.
- **Primary preprocessing `prh_clip+l2`**: one clamp threshold per model, pooled over all stored layers of all 1024 rows (D5; = PRH/ARH `remove_outliers`), from `hrhv2.prep.prh_clip_threshold` on the float64 stack; it must equal the S1.2 cache value (`results/S1_2/cache/clip_thresholds.json`, key `<stem>|wit1024`) bitwise. Then per-layer L2 in float64 (arithmetic of `hrhv2.prep.preprocess`).
- **Secondary `l2`**: computed too (cost is small).

## 2. Scores per layer pair (a, b)

- **mKNN**: SPEC s_N(10), neighbours ordered by the key (D_ij, j) on the SPEC squared distance D = K_ii + K_jj − 2K_ij (clipped at 0).
- **Biased linear CKA**: SPEC §2.8, HSIC = ⟨HK_AH, HK_BH⟩_F, no ε. (ARH's estimator adds ε = 1e-8 to the unnormalised denominator; negligible, checked in V4.)
- **Secondary, exploratory**: debiased CKA (Song et al. 2012, evaluated on HKH as `hrhv2.geometric.cka_pair`), because ARH Table 2 shows its largest loss for this estimator (0.65 → −0.01).

## 3. Layer-max calibration (SPEC §5 = ARH Algorithm 2 `calibrate_layers`)

- Observed T = max over all (a, b) of S[a, b].
- K = 500 permutations, π_k = `hrhv2.calibrate.permutation(1024, "S1.2-layermax|<lm>|<vit>", k)`, k = 0..499. The same π_k is applied to every ViT layer, both metrics and both pre. Inside every permutation the full L_A×L_B matrix is recomputed and T_k = its max.
- c_α = the ⌈(1−α)(K+1)⌉ = 476-th order statistic (ascending) of {T} ∪ {T_k}, α = 0.05; s_cal = max((T − c_α)/(1 − c_α), 0); p = (1 + #{T_k ≥ T})/(K+1); uncensored = T − mean(T_k). K is recorded per pair.
- **Implementation (GPU).**
  - CKA in float64: HSIC_π[a,b] = Σ_i Σ_j Kc_A^a[i,j] · Kc_B^b[π(i),π(j)], one batched GEMM per (pair, π) over all layer pairs; denominators are permutation invariant. Debiased terms from the same inner product plus O(m) diagonal / row-sum terms.
  - mKNN as exact integer arithmetic: count_π[a,b] = Σ_ij M_A^a[i,j] M_B^b[π(i),π(j)], M = 0/1 kNN adjacency (row i has the k = 10 SPEC neighbours of i). Entries are 0/1 and per-row counts ≤ 10, so a float16 GEMM with per-row partial sums ≤ 10 and a float32 final sum is exact; checked against integer CPU counts.
  - This relabelling equals SPEC recomputation on the permuted ViT matrix when no ViT row has an exact distance tie straddling the k-boundary. That is checked per ViT layer; flagged layers have their permuted kNN recomputed from the permuted D inside every permutation.
  - Memory cap 40 GB on the GPU (`set_per_process_memory_fraction`), ≤ 6 CPU threads.

## 4. Validation (run before the main run; results reported whatever they are)

- **V1 (kNN tables, all layers).** GPU float64 kNN tables (first 10 neighbours) vs `hrhv2.structural.knn_table` on hrhv2-preprocessed float64 rows, for every stored layer of all 25 models and both pre (1164 tables). Required: identical. Rule fixed now: if any table differs, the hrhv2 table is used for that layer and the case is reported. Also reported per layer: exact boundary ties (D_(10) = D_(11)) and the SPEC near-tie fraction at k = 10.
- **V2 (hrhv2 identity on layer pairs).** ≥ 12 layer pairs spread over models and depths (including the widest LM, d = 4096, and DINOv2-g), observed and 2 permutations each: GPU CKA (biased, debiased) vs `hrhv2.geometric.cka_pair` ≤ 1e-6 absolute (expected ~1e-13); GPU mKNN vs `hrhv2` s_N(10) recomputed from scratch on `B.permuted(π)` exactly equal.
- **V3 (fixed-depth cross-check).** The entries of S at the S1.2 layers (r = 0.5, 1.0) must reproduce S1.2's stored s_obs for all 136 pairs × 2 depths × 2 pre: mKNN exactly, CKA ≤ 1e-10. On 3 pairs, the single-entry calibration with S1.2's own permutation keys must reproduce S1.2's c_α, s_cal and p (≤ 1e-12).
- **V4 (vendored ARH).** With our permutations injected through `perm_fn`:
  - ARH scalar `calibrate()` on one layer pair (mKNN with the pinned PRH `mutual_knn`; biased CKA with ARH `estimators.cka_biased`), vs our single-entry calibration;
  - ARH `calibrate_layers(agg="max")` on one full pair (bloomz-560m × ViT-Ti/16-in21k, 25 × 12 layers) vs our layer-max T, c_α, p, s_cal.
  - Tolerance 1e-6 (ARH stores values in float32). Differences caused by tie-breaking of exact ties (PRH `argsort`, LM duplicates) are reported, not hidden.
- If V1–V4 reveal an implementation error, it is fixed and the validation rerun before the main run; the fix and both validation outputs are kept.

## 5. Trend statistics

All on prh_clip+l2 (primary) and l2 (secondary), for mKNN(10), biased CKA, and debiased CKA (exploratory).

- **(a) ARH-style.** Pearson r across the 136 pairs between the score and the LLM rank.
  - Rank = position in ARH's ordered LLM list. That list is `prh_models.get_models("val")` (ARH 785054b and SOG be654a37 agree) and is the x-axis order of ARH Fig. 5, "ranked according to their language performance (Huh et al. 2024)": bloomz-560m 1, -1b1 2, -1b7 3, -3b 4, -7b1 5, open_llama_3b 6, open_llama_7b 7, [open_llama_13b 8, absent], llama-7b 9.
  - Sensitivity: dense ranks 1..8.
  - Scores: raw T ("before"), s_cal ("after", ARH's statistic), uncensored.
  - Uncertainty: ViT-bootstrap 95% CI (2000 draws, seed sha256("S1.2-layermax|boot|<stat>")); exact one-sided p over all 8! = 40,320 LM relabellings (same relabelling for every ViT).
- **(b) S1.2-style (LM axis).** Tr = mean over the 17 ViTs of the within-BLOOMZ Spearman ρ (num_params vs score), on uncensored layer-max scores (primary), and on s_cal and raw. Exact one-sided p over the 120 BLOOMZ relabellings (same for every ViT). OpenLLaMA 7B > 3B fraction as a sign check.
- **(c) Vision axis.** S1.2's statistic: mean over (LM, vision family) cells of Spearman ρ against ViT num_params; exact p over all 124,416 within-family relabellings.
- **Fixed-depth anchors**, same statistics: the single entries at the S1.2 layers (r = 0.5, 1.0), (i) from S1.2's stored values and (ii) calibrated with this run's permutations.

## 6. Reading of the replication question (fixed now; descriptive, no Holm family, unadjusted p)

Δ_X = r_before(X) − r_after(X), ARH-style (a), primary pre, layer-max.

- **ARH pattern reappears** if Δ_CKA − Δ_mKNN ≥ 0.20 and r_after(CKA) ≤ r_after(mKNN) − 0.20. (ARH Table 2: Δ_CKA − Δ_mKNN = 0.41 − 0.01 = 0.40; r_after 0.45 vs 0.84.)
- **ARH pattern absent** if Δ_CKA − Δ_mKNN ≤ 0.10 or r_after(CKA) ≥ r_after(mKNN) − 0.10.
- Otherwise **partial**.
- **S1.2 axis.** "Layer-max loses the CKA trend on the BLOOMZ axis" if Tr_unc(CKA, layer-max) ≤ 0.40 while Tr_unc(mKNN, layer-max) ≥ 0.60; "no loss" if Tr_unc(CKA, layer-max) ≥ 0.60; otherwise "intermediate".

The same readings are reported for debiased CKA (exploratory) and for l2 (secondary).

## 7. Attribution (pre-specified diagnostics, reported whatever the outcome)

- **A1 Selection.** The argmax (a*, b*) per pair, metric and pre, as relative depths a*/(L_A − 1), b*/(L_B − 1); the number of tied maxima. Tabulated by LM and by ViT. Spearman of the selected LM depth with LLM rank (across pairs) and with BLOOMZ size; agreement of the CKA and mKNN selections. The modal location of the null maxima (argmax of each T_k).
- **A2 Null inflation vs number of layer pairs.**
  - Across pairs: Spearman of c_α and of mean(T_k) with L_A·L_B.
  - Within pairs, nested depth grids G_n: layers round-half-up(r·(L − 1)) for r = j/(n − 1), j = 0..n − 1, on both sides, n ∈ {2, 3, 5, 9, 17} (duplicates removed), plus the full grid; single entries at r = 0.5 and r = 1.0; one-sided maxima (all LM layers × last ViT layer; last LM layer × all ViT layers).
  - For each subset: T, c_α, p, s_cal, uncensored, null mean, and every trend statistic of §5, plotted against the subset size.
- **A3 Counterfactual decomposition of the trend change** (ARH-style r and Tr on):
  1. T (raw max);
  2. T − c_α (shift only, no ceiling);
  3. s_cal = (T − c_α)/(1 − c_α), censored (ARH);
  4. T − c_α^entry(a*, b*), the single-entry threshold of the selected pair (selection kept, max inflation removed);
  5. T − mean null of the selected entry;
  6. uncensored T − mean(T_k);

  plus the trends of c_α itself and of the inflation c_α − c_α^entry(a*, b*) against LLM rank.
- **Descriptive attribution rule.**
  - If 2 already reproduces the loss of 3, the cause is the size-dependent null shift.
  - If 2 and 3 differ, the ceiling division and censoring contribute.
  - If 4 recovers the trend, the cause is max-inflation, with null inflation growing with layer count.
  - If the fixed-depth entries keep the trend while the selected depths move with size, the cause is selection.

## 8. Reporting rules

- No pair, layer or permutation is dropped. Failures are reported.
- Pilots are excluded and labelled.
- Every number in the report comes from `results/S1_2_layermax/layermax_summary.json`, which is written by `aggregate.py`. Per-pair raw outputs are in `results/S1_2_layermax/pairs/`.
- `results/S1_2_layermax/DONE` is written last.
