# S1.2 LLaMA axis: layer-max replication, implementation I (role `llama-axis-run-indep`)

Panel: llama-7b / 13b / 30b × the 17 ARH `val` ViTs on WIT-1024 (51 pairs). Pre-processing: `prh_clip+l2` (primary) and `l2` (secondary). Calibration: K = 500 permutations, keys `S1.2-layermax|<lm>|<vit>`. Definitions, rules and predictions come from `ANALYSIS_PLAN_LLAMA.md` (Part A) and `PARTB_PREDICTIONS.md` (Part B, frozen before any cross-model score). Every number below is read from `results/S1_2_layermax_llama/indep/summary.json` or the validation files next to it.

**Reading guide.**
- Tr_L is the mean over the 17 ViTs of the Spearman correlation between LLaMA size (7B < 13B < 30B) and the score.
- With 3 sizes, the smallest one-sided exact p is 1/6. No LLaMA-axis result can be significant: each is a directional, descriptive result on one model family, with ViT-only uncertainty.
- 0.10 is the plan's materiality threshold for changes in Tr_L. It equals 4 ViTs changing by one adjacent swap.

## 1. Part B verdicts (primary: prh_clip+l2, layer-max)

| ID | what was predicted | observed | outcome |
|---|---|---|---|
| P-L1 | mKNN: \|ΔTr_L(s_cal)\| < 0.10 and \|ΔTr_L(unc)\| < 0.10 | Tr_L raw +0.41 → s_cal +0.38 / unc +0.38; ΔTr -0.029 / -0.029 | **holds** |
| P-L2 | biased CKA (material, s = +1): ΔTr_L(s_cal) ≤ −0.10 "confirmed" | Tr_L raw +0.68 → s_cal -0.85; ΔTr -1.529 (exact p one-sided decrease 0.167) | **confirmed** |
| P-L2 (unc, secondary) | same rule on uncensored | Tr_L unc -1.00; ΔTr -1.676 | confirmed |
| P-L2a | premise: median \|c_α − B̄\| ≤ 0.01 and ≥ 34/51 comparisons with c_α(larger) > c_α(smaller) | median \|c_α − B̄\| 0.0018; order agrees 51/51 | **holds** |
| P-L3 (r0.5) | biased CKA fixed depth: keeps sign | Tr_L raw +1.00 → s_cal +0.97 | **keeps sign** |
| P-L3 (r1.0) | biased CKA fixed depth: keeps sign | Tr_L raw +1.00 → s_cal +1.00 | **keeps sign** |
| P-L4 | debiased CKA: ΔTr_L(s_cal), ΔTr_L(unc) ≥ −0.10; pooled r_after ≥ r_before − 0.05 | ΔTr +0.000 / +0.000; pooled r +0.368 → +0.364 | **holds** |
| P-L5 | calibrate-then-max (secondary): keeps sign | Tr_L raw +0.68 → Tr_L(M) -0.88 | **reversed** |

**Plan §10 (primary pre only).**
- item1 premise failure P-L2a: **no**
- item2 P-L2 contradicted: **no**
- item3 P-L1 or P-L4 fails: **no**
- item4 P-L3 or P-L5 reversed: **YES**
- Deductive checks, where a violation would mean an implementation error: P-L1 0 violations, P-L2a 0 violations.
- Item 4 is triggered by: P-L5. Under plan §10 this counts against the remedy claim (fixed layers, or calibration before the max, avoid the interaction), not against Corollary 2 itself, which allows a reversal whenever the baselines of the entries that set the score rise faster than the score.
- Items 1 and 2 are not triggered: nothing here counts against using Proposition 5 + Corollary 2 as a predictor of calibration effects from features.

## 2. Validation (plan §4)

| item | check | result |
|---|---|---|
| V-L1 | clip thresholds recomputed in the run = reference, bitwise (S1.2 cache for llama-7b and the ViTs; frozen `clip_thresholds_partB.json` for llama-13b/30b) | 20 models, all bitwise equal: yes → **PASS** |
| V-L2 | kNN tables = `hrhv2.structural.knn_table` (I uses it by construction); ViT and llama-7b tables = frozen layer-max caches; ViT boundary ties = 0; observed count matrices = overlap of these tables | frozen tables identical yes (36 model×pre); ViT exact boundary-tie rows 0; observed counts identical on all layer pairs yes → **PASS** |
| V-L3 | 12 fixed layer pairs × 4 perms × 2 pre vs hrhv2 from scratch | 96 checks; mKNN all equal yes; max \|ΔCKA\| biased 3.3e-16, debiased 3.9e-16 → **PASS** |
| V-L4a | llama-7b fixed-depth entries (17 pairs × 2 depths × 2 pre) = S1.2 panel s_obs | 68 configs; mKNN equal yes; max \|ΔCKA\| 1.4e-15 / 1.5e-15 → **PASS** |
| V-L4b | llama-7b layer-max outputs = frozen `results/S1_2_layermax/indep` and `.../pairs` (early); ≤ 1e-12 relative, p and null counts identical | p, argmax and mKNN null counts identical; raw matrices max rel 2.4e-15. Strict 1e-12 relative exceeded only by: early:cka_d.null_mean (see note). Under the V-L5 mixed tolerance: **PASS**; strict rule: **FAIL (1 near-zero field)** |
| V-L7 | Proposition 5: entry-wise null means vs √(PR_u PR_v)/(m−1) (biased CKA), 0 (debiased), 10/1023 (mKNN) on the 34 new pairs | primary: median rel. err. 0.0007, \|z\|>4 fractions 0.0000 / 0.0000 / 0.0000; l2: median 0.0024 → **PASS** |
| V-L6 | vendored ARH `calibrate` / `calibrate_layers(agg="max")`, our π injected; tolerance 1e-6 | max \|Δ\| over s_cal, p, τ: 2.0e-08; observed-matrix diffs > 1e-6: mKNN 0/108, CKA 0/492 → **PASS** |
| V-L1/2/4b/7 recheck | the four entries of `validation_vl.json` recomputed with the current `vl_checks.py` (sha256 5b857a19…), nothing overwritten | identical to stored: V-L1 yes, V-L4B yes, V-L7 yes, V-L2 yes → **PASS** |
| self-check | BLOOMZ Tr and 8-LM ARH r recomputed by the new aggregation = frozen 8-LM indep summary | max \|Δ\| Tr 0.0e+00, r 1.6e-15 → **PASS** |

**Note on V-L4b.** The only field above 1e-12 relative is the debiased-CKA null mean against the early implementation's frozen outputs. Its worst case is vit_huge_patch14_224.mae|fixed_r1.0: \|value\| 7.7e-07, absolute difference 6.7e-18. The exact expectation of this quantity is 0 (plan §8; V-L7 confirms within MC error), so a relative tolerance is ill-conditioned there. The absolute difference is round-off, not an implementation difference. Against the frozen independent run, the same field agrees to 1.3e-14 relative. The first, strict-only output is kept as `validation_vl.attempt1_vl4b_strict_only.json`.

## 3. LLaMA-axis trends, prh_clip+l2 (primary)

Tr_L [exact one-sided p over 6 relabellings]; ΔTr_L = Tr_L(score) − Tr_L(raw T) for the same protocol.

| metric | protocol | Tr_L raw T | Tr_L s_cal | Tr_L uncensored | ΔTr s_cal | ΔTr unc | Tr_L c_α | Tr_L E0[max] |
|---|---|---|---|---|---|---|---|---|
| mKNN | layer-max | +0.41 [0.333] | +0.38 [0.333] | +0.38 [0.333] | -0.029 | -0.029 | +0.30 | +0.71 |
| mKNN | fixed r=0.5 | +0.82 [0.167] | +0.82 [0.167] | +0.82 [0.167] | +0.000 | +0.000 | -0.36 | -0.09 |
| mKNN | fixed r=1.0 | +0.91 [0.167] | +0.91 [0.167] | +0.91 [0.167] | +0.000 | +0.000 | -0.13 | -0.09 |
| mKNN | calibrate-then-max | +0.41 [0.333] | +0.38 [0.333] | +0.38 [0.333] | -0.029 | -0.029 | – | – |
| CKA (biased) | layer-max | +0.68 [0.167] | -0.85 [1.000] | -1.00 [1.000] | -1.529 | -1.676 | +1.00 | +1.00 |
| CKA (biased) | fixed r=0.5 | +1.00 [0.167] | +0.97 [0.167] | +0.65 [0.333] | -0.029 | -0.353 | +1.00 | +1.00 |
| CKA (biased) | fixed r=1.0 | +1.00 [0.167] | +1.00 [0.167] | +1.00 [0.167] | +0.000 | +0.000 | +1.00 | +1.00 |
| CKA (biased) | calibrate-then-max | +0.68 [0.167] | -0.88 [1.000] | -0.88 [1.000] | -1.559 | -1.559 | – | – |
| CKA (debiased) | layer-max | -0.38 [1.000] | -0.38 [1.000] | -0.38 [1.000] | +0.000 | +0.000 | -0.03 | +0.15 |
| CKA (debiased) | fixed r=0.5 | +1.00 [0.167] | +1.00 [0.167] | +1.00 [0.167] | +0.000 | +0.000 | -0.24 | -0.12 |
| CKA (debiased) | fixed r=1.0 | +1.00 [0.167] | +1.00 [0.167] | +1.00 [0.167] | +0.000 | +0.000 | -0.15 | -0.03 |
| CKA (debiased) | calibrate-then-max | -0.38 [1.000] | -0.38 [1.000] | -0.38 [1.000] | +0.000 | +0.000 | – | – |

All score trends use 17/17 ViTs. Only mKNN c_α rows can be constant across the three sizes (c_α is a count / 10240); such ViTs are excluded from that Tr_L and counted in `n_vit_constant` in summary.json.

Means over the 17 ViTs (layer-max):

| metric | LM | T | c_α | s_cal | uncensored | E0[max] |
|---|---|---|---|---|---|---|
| mKNN | 7B | 0.1438 | 0.0138 | 0.1318 | 0.1312 | 0.0126 |
| mKNN | 13B | 0.1430 | 0.0138 | 0.1310 | 0.1304 | 0.0126 |
| mKNN | 30B | 0.1447 | 0.0139 | 0.1327 | 0.1320 | 0.0127 |
| CKA (biased) | 7B | 0.4268 | 0.1866 | 0.2962 | 0.2421 | 0.1847 |
| CKA (biased) | 13B | 0.4331 | 0.2041 | 0.2888 | 0.2309 | 0.2023 |
| CKA (biased) | 30B | 0.4408 | 0.2200 | 0.2843 | 0.2226 | 0.2182 |
| CKA (debiased) | 7B | 0.3327 | 0.0056 | 0.3290 | 0.3299 | 0.0028 |
| CKA (debiased) | 13B | 0.3301 | 0.0056 | 0.3263 | 0.3273 | 0.0028 |
| CKA (debiased) | 30B | 0.3281 | 0.0056 | 0.3244 | 0.3252 | 0.0029 |

Layer subsets, biased CKA (A2; grid17 equalises the LM side to 17 layers; noLMemb / LMdeep / LMemb_only are the pre-specified descriptive subsets):

| subset | Tr_L raw | Tr_L s_cal | Tr_L unc | Tr_L c_α | mean c_α 7B / 13B / 30B |
|---|---|---|---|---|---|
| full | +0.68 | -0.85 | -1.00 | +1.00 | 0.1866 / 0.2041 / 0.2200 |
| grid2 | +1.00 | +1.00 | +1.00 | +1.00 | 0.1333 / 0.1456 / 0.1480 |
| grid3 | +1.00 | +1.00 | +0.94 | +1.00 | 0.1334 / 0.1659 / 0.1908 |
| grid5 | +0.47 | -0.79 | -1.00 | +1.00 | 0.1783 / 0.1990 / 0.2175 |
| grid9 | +0.56 | -0.32 | -1.00 | +1.00 | 0.1864 / 0.2028 / 0.2182 |
| grid17 | +0.68 | -0.91 | -1.00 | +1.00 | 0.1866 / 0.2033 / 0.2195 |
| lmall_x_vitlast | +0.68 | -0.56 | -1.00 | +1.00 | 0.1856 / 0.2030 / 0.2188 |
| lmlast_x_vitall | +1.00 | +1.00 | +1.00 | +1.00 | 0.0752 / 0.0832 / 0.1170 |
| noLMemb | +0.68 | -0.85 | -1.00 | +1.00 | 0.1866 / 0.2041 / 0.2200 |
| LMdeep | +0.68 | -0.85 | -1.00 | +1.00 | 0.1866 / 0.2041 / 0.2200 |
| LMemb_only | +1.00 | +1.00 | +1.00 | +1.00 | 0.1341 / 0.1464 / 0.1488 |

Readings (plan §7):

| metric | ARH-style, pooled 10 LMs: r before → after (CKA) vs (mKNN) | verdict | 8-LM reference verdict | LLaMA-axis reading |
|---|---|---|---|---|
| CKA (biased) (primary) | +0.51 → +0.40 vs +0.52 → +0.52 | partial (ΔΔ +0.12) | ARH pattern absent | intermediate |
| CKA (debiased) (exploratory (debiased CKA)) | +0.37 → +0.36 vs +0.52 → +0.52 | ARH pattern absent (ΔΔ +0.00) | ARH pattern absent | no raw trend to lose |

Pooled 10-LM panel, ARH-style Pearson r with LLM rank (layer-max) [exact one-sided p over 10! relabellings; ViT-bootstrap 95% CI]:

| metric | score | r (ARH rank) | p_one | CI | r (dense rank) | r (log params) | r on 8-LM panel |
|---|---|---|---|---|---|---|---|
| mKNN | raw T | +0.517 | 5.5e-07 | [+0.43, +0.68] | +0.524 | +0.467 | +0.519 |
| mKNN | s_cal | +0.518 | 5.5e-07 | [+0.43, +0.68] | +0.525 | +0.467 | +0.520 |
| mKNN | uncensored | +0.517 | 5.5e-07 | [+0.43, +0.69] | +0.524 | +0.467 | +0.520 |
| CKA (biased) | raw T | +0.512 | 8.3e-07 | [+0.38, +0.71] | +0.519 | +0.476 | +0.494 |
| CKA (biased) | s_cal | +0.398 | 1.1e-02 | [+0.31, +0.56] | +0.405 | +0.317 | +0.425 |
| CKA (biased) | uncensored | +0.266 | 6.7e-02 | [+0.20, +0.38] | +0.274 | +0.179 | +0.334 |
| CKA (debiased) | raw T | +0.368 | 4.6e-04 | [+0.30, +0.49] | +0.379 | +0.344 | +0.401 |
| CKA (debiased) | s_cal | +0.364 | 4.5e-04 | [+0.30, +0.48] | +0.375 | +0.341 | +0.398 |
| CKA (debiased) | uncensored | +0.365 | 4.6e-04 | [+0.30, +0.49] | +0.377 | +0.342 | +0.399 |

Joint within-family trend Tr_BL = (Tr_BLOOMZ + Tr_L)/2, exact p over 720 relabellings (descriptive; the BLOOMZ half was already known):

| metric | score | Tr_BLOOMZ | Tr_L | Tr_BL | p_one |
|---|---|---|---|---|---|
| mKNN | s_obs | +1.00 | +0.41 | +0.71 | 0.0083 |
| mKNN | s_cal | +1.00 | +0.38 | +0.69 | 0.0125 |
| mKNN | uncensored | +0.99 | +0.38 | +0.69 | 0.0111 |
| CKA (biased) | s_obs | +0.89 | +0.68 | +0.78 | 0.0069 |
| CKA (biased) | s_cal | -0.02 | -0.85 | -0.44 | 0.8472 |
| CKA (biased) | uncensored | -0.29 | -1.00 | -0.65 | 0.9306 |
| CKA (debiased) | s_obs | +0.83 | -0.38 | +0.22 | 0.2403 |
| CKA (debiased) | s_cal | +0.84 | -0.38 | +0.23 | 0.2306 |
| CKA (debiased) | uncensored | +0.84 | -0.38 | +0.23 | 0.2306 |

Vision axis restricted to the LLaMA rows (descriptive), layer-max:

- mKNN: Tr_vision raw +0.74 (p_one 0.005), s_cal +0.74, unc +0.74. OpenLLaMA 7B > 3B fraction (raw): 1.00
- CKA (biased): Tr_vision raw +0.90 (p_one 0.000), s_cal +0.58, unc +0.25. OpenLLaMA 7B > 3B fraction (raw): 1.00
- CKA (debiased): Tr_vision raw +0.26 (p_one 0.195), s_cal +0.26, unc +0.26. OpenLLaMA 7B > 3B fraction (raw): 1.00

Attribution diagnostics (plan §6), biased CKA:

- **A1, selection.** Mean relative depth of the selected LM layer: 7B 0.83, 13B 0.81, 30B 0.82. Mean relative depth of the selected ViT layer: 7B 0.96, 13B 0.97, 30B 0.98. Extra tied maxima: 0. CKA and mKNN select the same layer pair in 0.06 of pairs. Mean relative LM depth of the null maxima: 7B 0.87, 13B 0.85, 30B 0.80.
- **Highest-null LM layer.** The argmax over LM layers of the mean over ViTs of max_v E0[S_uv] is: 7B layer 28 (rel 0.875, level 0.1847), 13B layer 34 (rel 0.850, level 0.2023), 30B layer 48 (rel 0.800, level 0.2182).
- **A2, inflation.** Spearman(c_α, number of layer pairs) is +0.37 on the 51 LLaMA pairs and +0.36 on the 170 pooled pairs. Mean inflation c_α − c_α^entry(a*, b*): 7B 0.0169, 13B 0.0117, 30B 0.0043.
- **A3, counterfactual decomposition.** Tr_L and pooled-10 r (ARH rank), with the mean by LLaMA size:

| quantity | Tr_L | pooled r | 7B | 13B | 30B |
|---|---|---|---|---|---|
| 1_T_raw | +0.68 | +0.51 | 0.4268 | 0.4331 | 0.4408 |
| 2_T_minus_calpha | -1.00 | +0.27 | 0.2402 | 0.2291 | 0.2208 |
| 3_s_cal_ARH | -0.85 | +0.40 | 0.2962 | 0.2888 | 0.2843 |
| 3b_uncapped_ratio | -0.85 | +0.40 | 0.2962 | 0.2888 | 0.2843 |
| 4_T_minus_c_entry | -0.91 | -0.02 | 0.2571 | 0.2408 | 0.2251 |
| 5_T_minus_entry_null_mean | -0.91 | -0.02 | 0.2590 | 0.2426 | 0.2269 |
| 6_uncensored_T_minus_mean_Tk | -1.00 | +0.27 | 0.2421 | 0.2309 | 0.2226 |
| c_alpha | +1.00 | +0.38 | 0.1866 | 0.2041 | 0.2200 |
| inflation_calpha_minus_c_entry | -0.50 | -0.33 | 0.0169 | 0.0117 | 0.0043 |
| mean_Tk | +1.00 | +0.38 | 0.1847 | 0.2023 | 0.2182 |
| level_minus_multiplicity | +0.34 | -0.58 | 0.0000 | 0.0000 | 0.0000 |
| max_entry_null_mean | +1.00 | +0.38 | 0.1847 | 0.2023 | 0.2182 |

- **Plug-in check.** Tr_L(T − B̄_pred) = -1.00 against Tr_L(uncensored) = -1.00; \|diff\| 0.000, within 0.10: yes.

## 4. Part B details (primary)

- **Threshold vs predicted max baseline.**
  - Mean c_α by size: 0.1866 / 0.2041 / 0.2200; mean B̄: 0.1847 / 0.2023 / 0.2182.
  - c_α − B̄ ranges over [0.0015, 0.0021] (mean 0.0018); E0[max] − B̄ mean -3.8e-06; Pearson(c_α, B̄) 1.0000.
  - Qualifying comparisons per size pair: {'huggyllama__llama-7b|huggyllama__llama-13b': 17, 'huggyllama__llama-7b|huggyllama__llama-30b': 17, 'huggyllama__llama-13b|huggyllama__llama-30b': 17}. They match the frozen Part B counts: yes.
- **P-L2a deductive check.** c_α increases with size for 17 ViTs and decreases for 0. Violations: 0. Uncensored analogue (E0[max] order): 0 violations.
- **Pairwise flips under s_cal** (qualifying comparisons, n = 51):
  - concordant → discordant (the predicted direction): 38;
  - discordant → concordant (against): 0;
  - concordant → tie: 0; unchanged: 13.
- **Pairwise flips under uncensored** (qualifying comparisons, n = 51):
  - concordant → discordant (the predicted direction): 42;
  - discordant → concordant (against): 0;
  - concordant → tie: 0; unchanged: 9.
- **P-L1 deductive check.**
  - The check is vacuous here: no (ViT, size pair) has a raw mKNN gap larger than ζ_j + ζ_j′, so Corollary 2 guarantees no order and none is tested.
  - s_cal: 0 violations among 0 eligible comparisons;
  - uncensored: 0 violations among 0;
  - mean ζ = \|c_α − 10/1023\| is 0.0041 (max 0.0046).
- **P-L3 r0.5.** Tr_L(c_α) +1.00. The Part B B_fd trend is +1.00 (material: yes). Uncensored, descriptive: keeps sign.
- **P-L3 r1.0.** Tr_L(c_α) +1.00. The Part B B_fd trend is +1.00 (material: yes). Uncensored, descriptive: keeps sign.
- **P-L5.** Tr_L of the uncensored analogue: -0.88 (reversed).

## 5. l2 Part B outcomes (secondary; failures confined to l2 are reported, not counted)

- **P-L1:** holds (ΔTr +0.000 / +0.000).
- **P-L2:** no directional prediction (B-bar trend not material). Tr_L raw -0.50 → s_cal -0.56 (ΔTr -0.059).
  - Pairwise flips on the 32 qualifying comparisons:
    - s_cal: 2 in the predicted direction, 0 against;
    - uncensored: 7 in the predicted direction, 0 against.
- **P-L2a:** holds. Median \|c_α − B̄\| is 0.0021; order agrees 32/32 (needed 22).
- **P-L3:** r0.5 not applicable (|Tr_L(raw)| < 0.10) (Tr +0.03 → -0.09); r1.0 keeps sign (Tr +1.00 → +1.00).
- **P-L4:** holds. ΔTr +0.000 / +0.000; pooled r +0.638 → +0.637.
- **P-L5:** keeps sign (Tr_L(M) -0.50).

## 6. LLaMA-axis trends, l2 (secondary)

Tr_L [exact one-sided p over 6 relabellings]; ΔTr_L = Tr_L(score) − Tr_L(raw T) for the same protocol.

| metric | protocol | Tr_L raw T | Tr_L s_cal | Tr_L uncensored | ΔTr s_cal | ΔTr unc | Tr_L c_α | Tr_L E0[max] |
|---|---|---|---|---|---|---|---|---|
| mKNN | layer-max | -0.97 [1.000] | -0.97 [1.000] | -0.97 [1.000] | +0.000 | +0.000 | +0.83 | +1.00 |
| mKNN | fixed r=0.5 | -0.74 [0.833] | -0.76 [1.000] | -0.74 [0.833] | -0.029 | +0.000 | +0.75 | +0.26 |
| mKNN | fixed r=1.0 | +1.00 [0.167] | +1.00 [0.167] | +1.00 [0.167] | +0.000 | +0.000 | +0.18 | +0.18 |
| mKNN | calibrate-then-max | -0.97 [1.000] | -0.97 [1.000] | -0.97 [1.000] | +0.000 | +0.000 | – | – |
| CKA (biased) | layer-max | -0.50 [0.833] | -0.56 [0.833] | -0.71 [0.833] | -0.059 | -0.206 | +1.00 | +1.00 |
| CKA (biased) | fixed r=0.5 | +0.03 [0.500] | -0.09 [0.667] | -0.15 [0.667] | -0.118 | -0.176 | -0.32 | -0.50 |
| CKA (biased) | fixed r=1.0 | +1.00 [0.167] | +1.00 [0.167] | +1.00 [0.167] | +0.000 | +0.000 | +1.00 | +1.00 |
| CKA (biased) | calibrate-then-max | -0.50 [0.833] | -0.50 [0.833] | -0.50 [0.833] | +0.000 | +0.000 | – | – |
| CKA (debiased) | layer-max | -0.50 [0.833] | -0.50 [0.833] | -0.50 [0.833] | +0.000 | +0.000 | +0.79 | +1.00 |
| CKA (debiased) | fixed r=0.5 | -0.06 [0.667] | +0.03 [0.500] | +0.00 [0.667] | +0.088 | +0.059 | -0.21 | -0.21 |
| CKA (debiased) | fixed r=1.0 | +1.00 [0.167] | +1.00 [0.167] | +1.00 [0.167] | +0.000 | +0.000 | -0.44 | -0.18 |
| CKA (debiased) | calibrate-then-max | -0.50 [0.833] | -0.50 [0.833] | -0.50 [0.833] | +0.000 | +0.000 | – | – |

All score trends use 17/17 ViTs. Only mKNN c_α rows can be constant across the three sizes (c_α is a count / 10240); such ViTs are excluded from that Tr_L and counted in `n_vit_constant` in summary.json.

Means over the 17 ViTs (layer-max):

| metric | LM | T | c_α | s_cal | uncensored | E0[max] |
|---|---|---|---|---|---|---|
| mKNN | 7B | 0.0945 | 0.0143 | 0.0813 | 0.0816 | 0.0128 |
| mKNN | 13B | 0.0883 | 0.0145 | 0.0749 | 0.0754 | 0.0129 |
| mKNN | 30B | 0.0834 | 0.0147 | 0.0697 | 0.0704 | 0.0131 |
| CKA (biased) | 7B | 0.3222 | 0.1202 | 0.2299 | 0.2041 | 0.1181 |
| CKA (biased) | 13B | 0.3340 | 0.1312 | 0.2336 | 0.2049 | 0.1291 |
| CKA (biased) | 30B | 0.2358 | 0.1334 | 0.1176 | 0.1045 | 0.1313 |
| CKA (debiased) | 7B | 0.2527 | 0.0060 | 0.2482 | 0.2498 | 0.0030 |
| CKA (debiased) | 13B | 0.2639 | 0.0063 | 0.2593 | 0.2608 | 0.0032 |
| CKA (debiased) | 30B | 0.1860 | 0.0066 | 0.1805 | 0.1826 | 0.0034 |

Layer subsets, biased CKA (A2; grid17 equalises the LM side to 17 layers; noLMemb / LMdeep / LMemb_only are the pre-specified descriptive subsets):

| subset | Tr_L raw | Tr_L s_cal | Tr_L unc | Tr_L c_α | mean c_α 7B / 13B / 30B |
|---|---|---|---|---|---|
| full | -0.50 | -0.56 | -0.71 | +1.00 | 0.1202 / 0.1312 / 0.1334 |
| grid2 | +1.00 | +1.00 | +1.00 | +1.00 | 0.1197 / 0.1307 / 0.1329 |
| grid3 | +1.00 | +1.00 | +1.00 | +1.00 | 0.1197 / 0.1307 / 0.1329 |
| grid5 | +1.00 | +1.00 | +1.00 | +1.00 | 0.1198 / 0.1307 / 0.1329 |
| grid9 | +1.00 | +1.00 | +1.00 | +1.00 | 0.1201 / 0.1312 / 0.1333 |
| grid17 | +1.00 | +1.00 | +1.00 | +1.00 | 0.1202 / 0.1312 / 0.1334 |
| lmall_x_vitlast | -0.50 | -0.56 | -0.71 | +1.00 | 0.1198 / 0.1307 / 0.1329 |
| lmlast_x_vitall | +1.00 | +1.00 | +1.00 | +1.00 | 0.0408 / 0.0502 / 0.0752 |
| noLMemb | -0.50 | -0.50 | -0.50 | -0.50 | 0.1182 / 0.1230 / 0.0752 |
| LMdeep | -0.50 | -0.50 | -0.50 | -0.50 | 0.1182 / 0.1230 / 0.0752 |
| LMemb_only | +1.00 | +1.00 | +1.00 | +1.00 | 0.1202 / 0.1312 / 0.1334 |

Readings (plan §7):

| metric | ARH-style, pooled 10 LMs: r before → after (CKA) vs (mKNN) | verdict | 8-LM reference verdict | LLaMA-axis reading |
|---|---|---|---|---|
| CKA (biased) (secondary (l2)) | +0.63 → +0.57 vs +0.40 → +0.41 | ARH pattern absent (ΔΔ +0.07) | ARH pattern absent | no raw trend to lose |
| CKA (debiased) (exploratory (debiased CKA)) | +0.64 → +0.64 vs +0.40 → +0.41 | ARH pattern absent (ΔΔ +0.01) | ARH pattern absent | no raw trend to lose |

Pooled 10-LM panel, ARH-style Pearson r with LLM rank (layer-max) [exact one-sided p over 10! relabellings; ViT-bootstrap 95% CI]:

| metric | score | r (ARH rank) | p_one | CI | r (dense rank) | r (log params) | r on 8-LM panel |
|---|---|---|---|---|---|---|---|
| mKNN | raw T | +0.402 | 3.5e-03 | [+0.32, +0.55] | +0.411 | +0.357 | +0.497 |
| mKNN | s_cal | +0.411 | 4.0e-03 | [+0.33, +0.56] | +0.420 | +0.358 | +0.507 |
| mKNN | uncensored | +0.405 | 3.8e-03 | [+0.32, +0.55] | +0.414 | +0.356 | +0.501 |
| CKA (biased) | raw T | +0.632 | 5.2e-03 | [+0.57, +0.71] | +0.624 | +0.499 | +0.723 |
| CKA (biased) | s_cal | +0.571 | 2.1e-02 | [+0.54, +0.62] | +0.549 | +0.396 | +0.660 |
| CKA (biased) | uncensored | +0.547 | 2.6e-02 | [+0.52, +0.59] | +0.525 | +0.373 | +0.631 |
| CKA (debiased) | raw T | +0.638 | 7.7e-03 | [+0.60, +0.68] | +0.638 | +0.546 | +0.711 |
| CKA (debiased) | s_cal | +0.637 | 7.9e-03 | [+0.60, +0.68] | +0.637 | +0.545 | +0.711 |
| CKA (debiased) | uncensored | +0.637 | 7.9e-03 | [+0.60, +0.68] | +0.637 | +0.545 | +0.711 |

Joint within-family trend Tr_BL = (Tr_BLOOMZ + Tr_L)/2, exact p over 720 relabellings (descriptive; the BLOOMZ half was already known):

| metric | score | Tr_BLOOMZ | Tr_L | Tr_BL | p_one |
|---|---|---|---|---|---|
| mKNN | s_obs | +0.71 | -0.97 | -0.13 | 0.6139 |
| mKNN | s_cal | +0.66 | -0.97 | -0.16 | 0.6306 |
| mKNN | uncensored | +0.69 | -0.97 | -0.14 | 0.6194 |
| CKA (biased) | s_obs | +0.43 | -0.50 | -0.04 | 0.5264 |
| CKA (biased) | s_cal | -0.05 | -0.56 | -0.31 | 0.7431 |
| CKA (biased) | uncensored | -0.09 | -0.71 | -0.40 | 0.8375 |
| CKA (debiased) | s_obs | +0.34 | -0.50 | -0.08 | 0.5653 |
| CKA (debiased) | s_cal | +0.34 | -0.50 | -0.08 | 0.5653 |
| CKA (debiased) | uncensored | +0.34 | -0.50 | -0.08 | 0.5653 |

Vision axis restricted to the LLaMA rows (descriptive), layer-max:

- mKNN: Tr_vision raw +0.77 (p_one 0.002), s_cal +0.77, unc +0.77. OpenLLaMA 7B > 3B fraction (raw): 0.00
- CKA (biased): Tr_vision raw +0.87 (p_one 0.000), s_cal +0.09, unc -0.12. OpenLLaMA 7B > 3B fraction (raw): 0.76
- CKA (debiased): Tr_vision raw +0.17 (p_one 0.281), s_cal +0.17, unc +0.17. OpenLLaMA 7B > 3B fraction (raw): 0.00

Attribution diagnostics (plan §6), biased CKA:

- **A1, selection.** Mean relative depth of the selected LM layer: 7B 0.97, 13B 0.97, 30B 0.71. Mean relative depth of the selected ViT layer: 7B 0.95, 13B 0.95, 30B 0.96. Extra tied maxima: 0. CKA and mKNN select the same layer pair in 0.45 of pairs. Mean relative LM depth of the null maxima: 7B 0.07, 13B 0.00, 30B 0.00.
- **Highest-null LM layer.** The argmax over LM layers of the mean over ViTs of max_v E0[S_uv] is: 7B layer 0 (rel 0.000, level 0.1181), 13B layer 0 (rel 0.000, level 0.1291), 30B layer 0 (rel 0.000, level 0.1313).
- **A2, inflation.** Spearman(c_α, number of layer pairs) is +0.33 on the 51 LLaMA pairs and +0.32 on the 170 pooled pairs. Mean inflation c_α − c_α^entry(a*, b*): 7B 0.0186, 13B 0.0255, 30B 0.0438.
- **A3, counterfactual decomposition.** Tr_L and pooled-10 r (ARH rank), with the mean by LLaMA size:

| quantity | Tr_L | pooled r | 7B | 13B | 30B |
|---|---|---|---|---|---|
| 1_T_raw | -0.50 | +0.63 | 0.3222 | 0.3340 | 0.2358 |
| 2_T_minus_calpha | -0.71 | +0.55 | 0.2020 | 0.2028 | 0.1024 |
| 3_s_cal_ARH | -0.56 | +0.57 | 0.2299 | 0.2336 | 0.1176 |
| 3b_uncapped_ratio | -0.56 | +0.57 | 0.2299 | 0.2336 | 0.1176 |
| 4_T_minus_c_entry | -0.50 | +0.53 | 0.2206 | 0.2283 | 0.1462 |
| 5_T_minus_entry_null_mean | -0.50 | +0.53 | 0.2227 | 0.2304 | 0.1484 |
| 6_uncensored_T_minus_mean_Tk | -0.71 | +0.55 | 0.2041 | 0.2049 | 0.1045 |
| c_alpha | +1.00 | +0.26 | 0.1202 | 0.1312 | 0.1334 |
| inflation_calpha_minus_c_entry | +0.56 | +0.08 | 0.0186 | 0.0255 | 0.0438 |
| mean_Tk | +1.00 | +0.26 | 0.1181 | 0.1291 | 0.1313 |
| level_minus_multiplicity | -0.85 | +0.09 | 0.0001 | 0.0000 | 0.0000 |
| max_entry_null_mean | +1.00 | +0.26 | 0.1181 | 0.1291 | 0.1313 |

- **Plug-in check.** Tr_L(T − B̄_pred) = -0.71 against Tr_L(uncensored) = -0.71; \|diff\| 0.000, within 0.10: yes.

## 7. Limits

- **Small family.** There are only 3 LLaMA sizes, so the smallest one-sided p is 1/6. Every Tr_L carries ViT-sampling uncertainty only; the uncertainty from having 3 LMs is not quantified.
- **Size and layer count are confounded.** L_A is 33 / 41 / 61. grid17 equalises the LM side.
- **Shared LM features.** The 17 ViT rows share the same three LM feature sets, so they are not independent replicates.
- **Pooled 10-LM panel.** It reuses the frozen 8-LM independent run for the 7 non-LLaMA LMs, with the same definitions and keys, and drops nothing.
- **Scope.** Every number above is implementation I. Its comparison with the early implementation E (V-L5) is in `validation_cross.json`.

## 7b. Cross-implementation comparison with E (V-L5; written after this report's numbers were final)

- **Outcome on plan-level quantities: PASS** (V-L5 per pair, clip thresholds, and every Tr_L / ΔTr_L / pooled r / reading / Part B input and outcome that both report). Descriptive diagnostics: 6 mismatching values (l2|trends_L|mknn|full|inflation, prh_clip+l2|trends_L|mknn|full|inflation, prh_clip+l2|trends_L|mknn|full|shift). Details are in `validation_cross.json`.
- **V-L5 per pair** (102 pair records = 51 pairs × 2 pre; 3 metrics; 13 subsets plus calibrate-then-max):
  - floats (s_obs, c_α, s_cal, uncensored, null mean, null sd, selected-entry threshold and null mean, M, M_unc) within 1e-12·max + 1e-15: 25704 values, 0 outside; largest |Δ| 4.6e-15, at most 0.71× the tolerance;
  - exact items (p, argmax, tied maxima, entry counts, calibrate-then-max argmax): 17238 checks, 0 differ;
  - mKNN null counts (500 × L_A × L_B per pair) identical in 102/102 pair records; observed counts identical in 102/102.
- **Clip thresholds (V-L1, E vs I):** 20/20 bitwise equal.
- **Entry-level arrays (beyond the plan items, informational):** values outside the mixed tolerance: {'cka_b.entry_obs': 4, 'cka_d.entry_obs': 53, 'cka_d.entry_c_alpha': 35, 'cka_d.full_null_max': 146}; largest |Δ| 9.5e-15. Exact items with differences: none. E stores CKA nulls in float32: 572 of 95040000 values differ from float32(I), max relative |Δ| 2.3e-05.
- **Summary level:** 4127 quantities that E also reports (every Tr_L with its exact p and per-ViT ρ, ΔTr_L and its p, Tr_BL, pooled r and exact p, vision trend, means by size, readings, Part B outcomes and their inputs, plan §10 items): 6 mismatches (tolerance 1e-12 absolute; categorical outcomes exact). Bootstrap CIs whose rng key strings differ between the two codes are listed as informational (254 values), not compared.
- **Mismatches:** prh_clip+l2|trends_L|mknn|full|shift|Tr (I 0.3745, E 0.3824); prh_clip+l2|trends_L|mknn|full|inflation|Tr (I 0.198, E 0.1686); l2|trends_L|mknn|full|inflation|Tr (I 0.6392, E 0.6176). The per-ViT ρ vectors of the same quantities differ too.
- **mKNN exact-tie check.** mKNN scores are counts / 10240. Recomputing every LLaMA-axis mKNN Tr_L from the integer counts with exact rational arithmetic (142 trends: 13 subsets × 5 fields, calibrate-then-max × 2 fields and 4 derived differences, × 2 pre) gives the exact values. I differs from them in 2 and E in 3.
  - prh_clip+l2, full, inflation: exact +0.1901 (10 ViTs with exact ties); I +0.1980; E +0.1686.
  - l2, full, inflation: exact +0.6313 (5 ViTs with exact ties); I +0.6392; E +0.6176.
  - prh_clip+l2, full, shift: exact +0.3745 (1 ViTs with exact ties); I +0.3745; E +0.3824.
  - Cause: a difference of two count/10240 floats (T − c_α, c_α − c_α^entry) is not always equal for equal integer differences, so exact ties between sizes are broken by round-off in one or both codes. Only these descriptive A3 differences are affected; every Tr_L of T, c_α, s_cal, uncensored, E0[max], M and M_unc, and so P-L1, equals the exact value in both implementations.

## 8. Files

- Summary: `results/S1_2_layermax_llama/indep/summary.json`. Code sha256 values are listed in `provenance.code_sha256`.
- Validation: `validation.json` (V-L3, V-L4a, ties), `validation_vl.json` (V-L1, V-L2, V-L4b, V-L7), `validation_vl6_arh.json` (V-L6), `validation_vl_recheck.json` (provenance recheck), `validation_cross.json` (I vs E; written after this report's numbers were final).
- `REPORT.pre_cross.md` is this report as it stood before the comparison with E; sections 1–7 are unchanged since.
- Raw matrices: `<pre>/pairs/<lm>__<vit>.npz`, holding the observed and 500 null L_A × L_B matrices.
