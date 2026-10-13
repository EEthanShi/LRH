# S1.2 secondary: ARH layer-max replication, independent implementation (role `arh-layermax`)

**Scope.**
- Panel: 136 pairs, 8 LMs × 17 ViTs, WIT-1024 (m = 1024).
- Layers: all stored layers. LMs: embedding plus every block, the last post-norm. ViTs: every block output, CLS token. That gives 80,960 layer pairs.
- Metrics: mKNN at k = 10, biased linear CKA, and debiased CKA (exploratory). All float64.
- Calibration: K = 500 for every pair and pre. The same π is applied to every ViT layer, and the max is recomputed inside each permutation.
- Preprocessing: `prh_clip+l2` (primary, all-layer clip per D5) and `l2` (secondary).

**Status.** This is replication only (LEAD_DECISIONS_v1.1). All p-values are unadjusted.

**Plan.** `analysis/s12_layermax/indep/PLAN_INDEP.md` (sha256 085746a8…, fixed before any computation) adopts the definitions of the early role's frozen `ANALYSIS_PLAN_LAYERMAX.md` (dd189721…) and uses the same permutation keys. The two codebases were written independently, and the early role's code was not read.

**Exploratory items.** Anything marked *exploratory* was not pre-specified, in particular the participation-ratio diagnostics in §4.

## 0. Answer

**Does ARH's "CKA trend lost after calibration" reappear under layer-max, given that our fixed-depth CKA trend persisted?**

### On the S1.2 axis (within-BLOOMZ Tr): yes

| | Tr before (raw max T) | Tr after (ARH s_cal) | Tr uncensored |
|---|---|---|---|
| Layer-max CKA | 0.89 (p = 0.042) | −0.02 (p = 0.51) | −0.29 (p = 0.68) |
| Layer-max mKNN | 1.00 | 1.00 | 0.99 (p = 0.008) |

- For the same calibrated statistic at fixed depth, CKA reaches 0.69 at r = 0.5 and 0.92 at r = 1.0.
- The pre-set rule reads **"layer-max loses the CKA trend on the BLOOMZ axis"**. The rule needs Tr_unc(CKA) ≤ 0.40 and Tr_unc(mKNN) ≥ 0.60.

### On ARH's own pooled statistic: only weakly

ARH's statistic is the Pearson r between score and LLM rank across all 136 pairs.

| | before | after (s_cal) | uncensored | ARH Table 2 (204 pairs) |
|---|---|---|---|---|
| CKA | 0.49 | 0.43 | 0.33 | 0.86 → 0.45 |
| mKNN | 0.52 | 0.52 | — | 0.85 → 0.84 |

- ΔΔ = 0.07, so the pre-set rule reads **"ARH pattern absent"**.
- This subpanel cannot reproduce ARH's magnitude. ARH's pooled "before" r of 0.86 relies on four LMs that are not in our panel: open_llama_13b and llama-13b/30b/65b, ARH ranks 8 and 10–12. Our "before" r is only 0.49.

### Attribution of the BLOOMZ-axis loss

**It is neither a layer-selection effect nor a count-of-layer-pairs effect.** It comes from **heterogeneous null baselines across layers**. The layer-max threshold c_α is set by a layer pair other than the selected one. In larger BLOOMZ models those other layers have a higher null CKA.

1. **The threshold is set elsewhere.** c_α of the layer max equals the largest single-entry threshold over all layer pairs: Spearman 0.99999, mean difference 5e-5.
2. **That entry is not the selected one.** For BLOOMZ it is an *early* LM layer (embedding or block 1) × the last ViT block. The observed maximum sits at the *top* LM layer, at relative depth 0.93–0.99 for every BLOOMZ size.
3. **Why early layers have a high null (exploratory).** A single entry's permutation-null mean is exactly tr(Kc_A)·tr(Kc_B)/((m−1)‖Kc_A‖‖Kc_B‖) = √(PR_A·PR_B)/(m−1). Here PR is the participation ratio of the centred Gram, and the formula is confirmed to MC error (median 0.5%).
4. **That null grows with BLOOMZ size.** The early-layer PR of BLOOMZ jumps with width, from 134 / 118 (560m, 1b1) to 242 / 248 / 249 (1b7, 3b, 7b1). The top-layer PR stays at about 110–140.
5. **So c_α rises with size and absorbs the trend.** c_α goes 0.135, 0.131, 0.181, 0.183, 0.184. The selected entry's own threshold is nearly flat: 0.109, 0.108, 0.107, 0.110, 0.121.
6. **Checks.**
   - Subtracting the selected entry's own threshold (T − c_entry) keeps the trend: Tr = 0.76.
   - Restricting the max to the upper half of LM layers restores it (exploratory): Tr_unc = 0.89, p = 0.017.
   - A 2 × 2 grid (first/last × first/last layers) already loses it: Tr_unc = −0.25.
   - c_α jumps between bloomz-1b1 and bloomz-1b7, which have the same L = 25.

**Bottom line.** The S1.2 fixed-depth result is not contradicted. The two analyses calibrate different statistics. Under layer-max, ARH's s_cal is a valid significance test of "max over layers > null". As an effect size for a scale trend, however, it subtracts the null of the most isotropic layer pair from the similarity of a different, best-aligned layer pair.

## 1. Validation (all checks pass; I4 mKNN `calibrate_layers` completed 2026-10-08T12:04Z, recorded in the 2026-10-09 addendum)

| check | result |
|---|---|
| I1 clip thresholds recomputed from float64 stacks vs S1.2 cache | 25/25 bitwise equal |
| I1 exact k = 10 boundary ties on the permuted (ViT) side | 0 of 352 layers in either pre, so the relabelling path equals SPEC recomputation for every π |
| I1 exact boundary ties on the LM side (A, never permuted) | 100 / 67 layers (prh / l2), all from the duplicate captions 493/531, resolved by the SPEC lower-index rule. Max near-tie fraction: LM 0.0059, ViT 0 |
| I2 hrhv2 identity: 12 layer pairs × 2 pre × {identity, π₀, π₁, π₄₉₉}; hrhv2 recomputes kNN and CKA from scratch on `B.permuted(π)` | mKNN identical in 96/96. max \|ΔCKA\| biased 2.5e-16, debiased 8.3e-16 (tol 1e-6) |
| I3 fixed-depth entries vs S1.2 stored s_obs (136 × 2 depths × 2 pre) | mKNN exactly equal, 544/544. CKA biased ≤ 3.1e-15, debiased ≤ 3.4e-15 |
| I4 vendored ARH `calibrate()`, our π injected, one layer pair | mKNN \|Δ\| ≤ 1.0e-9. CKA \|Δ\| ≤ 7.1e-9 |
| I4 vendored ARH `calibrate_layers(agg="max")` on bloomz-560m × ViT-Ti (25 × 12) | **CKA: PASS.** \|Δ\| s_cal 3.8e-9, c_α 7.9e-10, p 3e-12. The observed 25 × 12 matrix vs ARH `cka_biased` agrees to ≤ 1.6e-11 (`validation_arh_cka_b.json`). **mKNN: PASS** (finished 2026-10-08T12:04Z after 3554 s; 150k PRH `mutual_knn` calls on a GPU shared with Stage 2). \|Δ\| s_cal 1.1e-9, c_α 0 (exact), p 3e-12. The observed 25 × 12 matrix vs ARH `mutual_knn` agrees to ≤ 6.0e-9 with 0 entries > 1e-6; the residual is ARH's float32 arithmetic (`validation_arh.json`, marker `I4.done`). Our values for the pair (s_cal 0.074356, c_α 0.013672, p 0.0020) also equal the vendored output recorded by `arh-layermax-early` V4 |
| I5 cross-implementation vs `arh-layermax-early`, all 136 pairs × 2 pre × 3 metrics × 10 layer subsets | Null kNN counts identical (136 × 500 × L_A × L_B). argmax identical 408/408 per pre. Every s_obs, c_α, s_cal, p, uncensored and null mean agrees to ≤ 1e-14 (p exactly) |

**Compute.**
- GPU run: 384 s (prh) and 425 s (l2) wall on the shared RTX PRO 6000, using 8 CPU threads and ≤ 8 GB of GPU memory.
- K = 500 everywhere. No pair or permutation was dropped.
- Code provenance of I4: the combined I4 job (`launch_i4.sh`, launched 10:38Z) ran the `validate.py` in place at launch. At 11:34Z `validate.py` was replaced by the content of `validate_i4b.py` (sha256 bd8a697a…), which only adds the `--arh-metric` option; the two files are now identical. The CKA part of I4 was run under both versions (`validation_arh.json`, `validation_arh_cka_b.json`) and agrees bitwise. The pre-11:34Z file itself was not kept, so its hash is not recorded.
- The full observed and null L_A × L_B matrices are stored, so every sub-grid maximum below is exact.

## 2. Main tables (`prh_clip+l2`, primary)

### 2a. ARH-style trend

Pearson r across 136 pairs vs ARH LLM rank. ARH order is `get_models("val")` position: bloomz 1–5, open_llama 6–7, llama-7b 9. Brackets give the ViT-bootstrap 95% CI. p is exact and one-sided over 8! relabellings.

| metric | aggregate | before (raw T) | after (s_cal) | uncensored | frac s_cal > 0 |
|---|---|---|---|---|---|
| mKNN(10) | layer-max | 0.52 [0.43, 0.69] (p < 0.001) | 0.52 [0.43, 0.68] (p < 0.001) | 0.52 | 1.00 |
| mKNN(10) | fixed r = 0.5 | 0.47 | 0.47 | 0.47 | 1.00 |
| mKNN(10) | fixed r = 1.0 | 0.11 | 0.11 | 0.11 | 1.00 |
| CKA biased | layer-max | 0.49 [0.37, 0.69] (p < 0.001) | 0.43 [0.35, 0.56] (p = 0.022) | 0.33 [0.27, 0.44] (p = 0.058) | 1.00 |
| CKA biased | fixed r = 0.5 | −0.04 | −0.06 | −0.07 | 1.00 |
| CKA biased | fixed r = 1.0 | −0.36 | −0.25 | −0.22 | 1.00 |
| CKA debiased (expl.) | layer-max | 0.40 | 0.40 | 0.40 | 1.00 |

Readings under the pre-set rule:
- **CKA biased: ARH pattern absent.** Δ_CKA − Δ_mKNN = 0.07. The gap in r_after (mKNN − CKA) is 0.09.
- **Debiased CKA: absent.** ΔΔ = 0.00.

Sensitivity of the CKA (before / after) values:
- dense ranks 1..8: 0.50 / 0.43;
- log num_params: 0.44 / 0.30.

The fixed-depth pooled CKA r is negative mainly because of llama-7b's last layer (post-norm), whose CKA is 0.21 against about 0.34 for the other LMs. Layer-max avoids that layer: its selected depth for llama-7b is 0.83.

### 2b. S1.2-style trend

Tr is the mean over 17 ViTs of the within-BLOOMZ Spearman. p is exact and one-sided over 120 relabellings.

| metric | aggregate | Tr uncensored [CI] | Tr s_cal | Tr raw T | vision-axis Tr (unc), p | OpenLLaMA 7B > 3B |
|---|---|---|---|---|---|---|
| mKNN(10) | layer-max | 0.99 [0.98, 1.00] (p = 0.008) | 1.00 | 1.00 | 0.76 (0.002) | 1.00 |
| mKNN(10) | fixed r = 0.5 | 0.84 (0.025) | 0.84 | 0.85 | 0.20 | 0.35 |
| mKNN(10) | fixed r = 1.0 | 0.99 (0.008) | 0.98 | 0.99 | 0.71 (0.005) | 0.12 |
| **CKA biased** | **layer-max** | **−0.29 [−0.41, −0.18] (p = 0.683)** | **−0.02 (0.508)** | **0.89 (0.042)** | 0.11 (0.355) | 0.82 |
| CKA biased | fixed r = 0.5 | 0.69 (0.100) | 0.70 | 0.88 | 0.09 | 1.00 |
| CKA biased | fixed r = 1.0 | 0.92 (0.017) | 0.92 | 0.76 | −0.04 | 0.29 |
| CKA debiased (expl.) | layer-max | 0.84 (0.033) | 0.84 | 0.83 | 0.18 | 1.00 |

Two further observations:
- The fixed-depth rows reproduce S1.2: on S1.2's own stored values and permutations, Tr_unc is 0.70 / 0.92 for CKA and 0.84 / 0.98 for mKNN.
- On the vision axis the local/global contrast persists under layer-max: mKNN 0.76 (p = 0.002) against CKA 0.11.

## 3. Attribution (plan §7)

### A1 Selection

Mean over the 17 ViTs of the relative depth of the observed argmax:

| LM | L_A | ARH rank | mKNN LM depth | CKA LM depth | CKA ViT depth | CKA T | CKA c_α (layer-max) | CKA c_entry (selected pair) | CKA s_cal | CKA uncensored |
|---|---|---|---|---|---|---|---|---|---|---|
| bloomz-560m | 25 | 1 | 0.91 | 0.97 | 0.95 | 0.336 | 0.135 | 0.109 | 0.232 | 0.203 |
| bloomz-1b1 | 25 | 2 | 0.93 | 0.99 | 0.94 | 0.359 | 0.131 | 0.108 | 0.263 | 0.230 |
| bloomz-1b7 | 25 | 3 | 0.95 | 0.98 | 0.95 | 0.355 | 0.181 | 0.107 | 0.211 | 0.175 |
| bloomz-3b | 31 | 4 | 0.92 | 0.94 | 0.95 | 0.366 | 0.183 | 0.110 | 0.223 | 0.185 |
| bloomz-7b1 | 31 | 5 | 0.92 | 0.93 | 0.95 | 0.384 | 0.184 | 0.121 | 0.245 | 0.202 |
| open_llama_3b | 27 | 6 | 0.88 | 0.96 | 0.95 | 0.413 | 0.166 | 0.146 | 0.298 | 0.250 |
| open_llama_7b | 33 | 7 | 0.83 | 0.87 | 0.96 | 0.426 | 0.176 | 0.153 | 0.304 | 0.252 |
| llama-7b | 33 | 9 | 0.73 | 0.83 | 0.96 | 0.427 | 0.187 | 0.170 | 0.296 | 0.242 |

- **Within BLOOMZ, selection does not move with size.** The CKA selection stays at the top LM layer: within-BLOOMZ Tr of the selected depth is −0.24.
- **Across families, deeper-ranked LMs select earlier layers.** Correlation of selected depth with ARH rank: CKA −0.76, mKNN −0.74.
- **The CKA and mKNN selections differ.** They pick the same layer pair in 7% of pairs, with a mean |Δ depth| of 0.06 on the LM side.
- **Null maxima sit elsewhere.**
  - CKA: LM depth 0.44 on average, ViT 0.99. They coincide with the observed argmax in 8.6% of permutations.
  - mKNN: (0.37, 0.47), coinciding in 0.2% of permutations.

### A2 Null inflation and the number of layer pairs

Across pairs, Spearman correlations with L_A·L_B:

| metric | c_α | mean T_k | inflation c_α − c_entry |
|---|---|---|---|
| CKA | 0.30 | 0.30 | 0.06 |
| mKNN | 0.48 | 0.70 | 0.33 |

Mean inflation is 0.040 for CKA (0.168 vs 0.128) and 0.0023 for mKNN (0.0139 vs 0.0116).

**Nested grids** (same π). Column abbreviations: **r b / a** = ARH-style r before and after calibration (raw T / s_cal); **Tr unc** and **Tr s_cal** = within-BLOOMZ Tr of the uncensored and the s_cal score.

| subset (entries/pair) | CKA r b / a | CKA Tr unc | CKA Tr s_cal | CKA mean c_α | mKNN Tr unc |
|---|---|---|---|---|---|
| fixed r=1.0 (1) | −0.36 / −0.25 | 0.92 | 0.92 | 0.120 | 0.99 |
| grid2: {first, last}² (4) | −0.34 / −0.37 | −0.25 | −0.16 | 0.153 | 0.99 |
| grid5 (25) | 0.45 / 0.41 | −0.05 | 0.06 | 0.159 | 0.99 |
| grid17 (≤ 249; 17 LM layers for every LM) | 0.49 / 0.42 | −0.29 | −0.05 | 0.167 | 1.00 |
| LM all × ViT last (29) | 0.48 / 0.41 | −0.28 | −0.00 | 0.167 | 0.99 |
| LM last × ViT all (21) | −0.39 / −0.29 | 0.92 | 0.92 | 0.121 | 0.99 |
| *expl.:* full minus LM embedding layer | 0.49 / 0.41 | −0.29 | −0.01 | 0.164 | 0.99 |
| *expl.:* LM upper half × ViT all | 0.49 / 0.39 | **0.89 (p = 0.017)** | 0.90 | 0.149 | 0.99 |
| full (595) | 0.49 / 0.43 | −0.29 | −0.02 | 0.168 | 0.99 |

**The number of layer pairs is not the driver:**
- 4 entries already reproduce the loss;
- equalising every LM to 17 layers does not remove it;
- c_α jumps between two BLOOMZ models that have the same L = 25.

**What matters is whether the LM's early layers enter the max.** Dropping only the embedding layer is not enough, because blocks 1–2 have the same high PR.

### A3 Counterfactual decomposition (full grid)

| score | CKA pooled r | CKA Tr | mKNN Tr |
|---|---|---|---|
| 1 raw T | 0.49 | 0.89 | 1.00 |
| 2 T − c_α (shift only) | 0.34 | −0.29 | 1.00 |
| 3 s_cal (ARH) | 0.43 | −0.02 | 1.00 |
| 3b uncapped (T − c_α)/(1 − c_α) | 0.43 | −0.02 | 1.00 |
| 4 T − c_entry(a*, b*) (selection kept, max-inflation removed) | 0.21 | **0.76** | 0.99 |
| 5 T − entry null mean | 0.21 | 0.76 | 0.99 |
| 6 uncensored T − mean(T_k) | 0.33 | −0.29 | 0.99 |
| c_α itself | 0.27 | **0.90** | 0.57 |
| inflation c_α − c_entry | −0.16 | 0.55 | 0.47 |

Applying the plan §7 rule to the BLOOMZ axis:
- (2) already reproduces the loss, so the cause is a **size-dependent null shift** (c_α Tr = 0.90).
- (2) and (3) agree on direction, and censoring never binds (all s_cal > 0), so the ceiling and censoring are not the cause.
- (4) recovers the trend, so the cause is **max-inflation**: the max-null rises with BLOOMZ size beyond the selected entry's null.
- The selected depths do not move with BLOOMZ size, so it is **not selection**.

**The pooled cross-family axis behaves differently.**
- The entry-wise threshold of the selected pair itself rises with ARH rank. The selected late layers of OpenLLaMA and LLaMA have PR 200–250, against about 110–140 at the BLOOMZ top layers.
- This is ARH's width confound at the selected layers. It lowers the pooled r more under entry-wise calibration (0.21) than under layer-max (0.43).
- In our subpanel this effect is modest. Whether it explains ARH's full-panel drop (with the 13B–65B LLaMAs) cannot be tested here.

## 4. Mechanism (exploratory; `diag_null_profile.json`, `diag_no_embedding.json`)

**The null formula.** For centred Grams, the permutation expectation is E_π⟨Kc_A, P Kc_B Pᵀ⟩ = tr(Kc_A)·tr(Kc_B)/(m−1). So the null CKA baseline of a layer pair is √(PR_A·PR_B)/(m−1).
- Observed agreement with the 500-permutation null means: median relative error 0.5% (prh) and 2.3% (l2). The identity is exact in expectation, so the residual is the Monte Carlo error of a 500-draw mean.
- The layer-max c_α equals max over entries of the entry-wise c_α to within 4.4e-4.

**Participation ratio by layer** (`prh_clip+l2`; layers 0, 1, 2, L/4, L/2, 3L/4, L−2, L−1):

| model | PR at those layers |
|---|---|
| bloomz-560m | 134, 133, 127, 77, 67, 74, 76, 128 |
| bloomz-1b7 | 242, 185, 180, 110, 107, 111, 101, 119 |
| bloomz-3b | 248, 237, 218, 132, 109, 125, 112, 120 |
| bloomz-7b1 | 189, 249, 249, 170, 141, 153, 132, 111 |
| ViT final blocks | up to 250 (e.g. ViT-L in21k 253, DINOv2-L 239, CLIP-H-ft 243); the MAE models stay at PR ≤ 38 at every depth. The max-null entry sits at mean ViT relative depth 0.99 |

**What this means.** Under layer-max calibration, the larger BLOOMZ models are charged with the null of their wide early layers, even though their alignment peak is at the top layer. That top layer's own null does not grow with size.

**OpenLLaMA and LLaMA have the opposite profile.**
- Their PR is low in blocks 1 to L/4 (11–36).
- The embedding layer is moderate (131–180).
- PR peaks late, at 202–257 at relative depth 0.88–0.96.

For them the max-null entry is near the selected layer, and c_α − c_entry is about 0.02. For BLOOMZ ≥ 1b7 the gap is 0.06–0.07.

## 5. Secondary: `l2`

The **direction is the same as primary** on the BLOOMZ axis: layer-max CKA has Tr raw 0.43, s_cal −0.05 and uncensored −0.09, while mKNN has Tr_unc 0.69.

**Pooled ARH r:**
- CKA 0.72 → 0.66 (uncensored 0.63);
- mKNN 0.50 → 0.51;
- reading: "absent".

**Under `l2` the BLOOMZ mid layers are degenerate.** Their PR is ≈ 2–3, dominated by outlier dimensions, and the CKA selection is unstable: bloomz-560m and bloomz-1b7 select LM layers 3 and 2. The null max again sits at early LM layers × the last ViT block, at PR 80–157.

**Fixed-depth `l2` trends are negative at r = 0.5**, as S1.2 reported. Debiased CKA under `l2`: Tr = 0.34 at layer-max and 0.63 at r = 1.0.

## 6. Interpretation and limits

1. **Layer-max reproduces ARH's qualitative pattern on the BLOOMZ axis.** It is an artefact of how the layer-max null is built: the per-layer null baselines √(PR_A·PR_B)/(m−1) are heterogeneous. It is not evidence that the global similarity of the best-aligned layers stops growing with BLOOMZ size, since the entry-wise calibrated trend at the selected layer is 0.76 and at fixed last depth 0.92.
2. **The ARH layer-max s_cal remains a valid level-α test** of "some layer pair aligns beyond chance". It should not be used as a per-model effect size in scale-trend regressions when per-layer null baselines differ. The fixed-depth or entry-wise threshold is the appropriate scale for a trend claim.
3. **Debiased CKA (exploratory) shows no loss.** Layer-max Tr is 0.84, and the pooled r is 0.40 before and after. Its null is ≈ 0 for every layer (c_α ≈ 0.005), so no baseline heterogeneity exists, consistent with the mechanism. ARH's reported collapse of unbiased CKA (0.65 → −0.01) is **not reproduced** on this subpanel.
4. **Limits.**
   - The BLOOMZ axis has 5 sizes, so the smallest exact p is 1/120. The calibrated-trend loss is a point-estimate change from 0.89 (p = 0.042) to −0.29 (p = 0.68), not a significance test of the change.
   - The subpanel lacks ARH's four largest LMs, so the pooled ARH magnitude (0.86 → 0.45) is untestable here.
   - The PR mechanism is diagnosed post hoc (exploratory). It is exact as a statement about null means, but its link to BLOOMZ width is observed on 5 models only.
   - The permutation draws are shared with `arh-layermax-early` by design. The two implementations are independent in code, not in Monte Carlo draws.

## 7. Files

- **Code** (`analysis/s12_layermax/indep/`):
  - `PLAN_INDEP.md`, `lmx_common.py`, `run_layermax.py` (GPU);
  - `validate.py` (I1–I4) and `validate_i4b.py` (the same, plus a per-metric I4 option);
  - `lmx_aggregate.py`, `lmx_report.py`, `lmx_i5.py`;
  - `lmx_diag.py` and `lmx_diag2.py` (exploratory);
  - `launch_*.sh`.
- **Results** (`results/S1_2_layermax/indep/`):
  - `<pre>/pairs/*.npz`: full observed and null matrices;
  - `<pre>/prep_meta.json`;
  - `layermax_indep_summary.json`: every number in §2–3;
  - `tables.md`: the full tables, both pre;
  - `validation.json`, `validation_i5.json`, `validation_arh*.json`;
  - `diag_null_profile.json`, `diag_no_embedding.json`;
  - `DONE`.
- **Logs**: `logs/lmx_indep_*.log`.


## 8. Addendum (2026-10-09, role `arh-layermax`, resumed after a usage-limit cut-off)

- The vendored ARH `calibrate_layers(agg="max")` mKNN check completed and passes (§1). Every validation item I1–I5 now passes; no number in §0–§6 changed.
- Nothing was recomputed: `RUN_main.done`, `DONE`, `I4cka.done` and `I4.done` all exist, and `layermax_indep_summary.json` is unchanged (sha256 40380e80…).
- The report hash recorded in STATUS at 2026-10-08T11:35:06Z (efcb0b95…) belongs to an earlier save of this file. It was saved again 14 s later; the hash of this version is recorded in the 2026-10-09 STATUS event.
- The 2026-10-08 version (sha256 ecf9cacb…) is kept unchanged as `LAYERMAX_INDEP_REPORT.v20261008.md`.
- A follow-up extension to ARH's larger LMs (llama-13b/30b) is being run by separate roles under `analysis/s12_layermax_llama/`. It is not part of this report.
