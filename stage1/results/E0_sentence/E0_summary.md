# E0 summary: sentence encoders as the text side of the S1.1 COCO pools

Role `e0-run`, closing step of plan §11 step 7; outputs as plan §14. Written 2026-10-10T14:57:58Z.
Plan `analysis/e0_sentence/ANALYSIS_PLAN_E0.md` v1.0 (`e859c6ea…`) with `ANALYSIS_PLAN_E0_ADDENDUM_1.md` (`dcf0ecfc…`; implementation fix only, no rule or label changed).
Preconditions: STATUS line 341 `[e0-indep] AGREE` and line 339 `[e0-review] REVIEW`.
Every number is read from the files in §9 at the hashes listed there. This role recomputed no statistic; it only rechecked the caption-source fact by a count (§5.1). Readings follow plan §13, which binds this summary; paper wording stays the author's.

## 1. Confirmatory labels (plan order)

| prediction | label | statistic against its frozen threshold |
|---|---|---|
| **P1** agreement at fixed k/m stays stable, k ∈ {10, 100} | **holds** | 15/15 pairs informative (e_m from -0.602 to -0.390; needs \|e_m\| ≥ 0.10). Median e_k + e_m = 0.0060 (holds if \|median\| ≤ 0.03). 15/15 pairs have \|sum\| ≤ 0.05 (holds if ≥ 80%); sums run from -0.0086 to 0.0155. Per encoder: contriever 0.0060, mpnet 0.0115, qwen3emb06 0.0058. Holds in each of the 5 draws. |
| **P2b** between classes, the slope follows the level, not the vision model | **uninformative** | Guard: median \|g\| = 0.0921 < 2·s_res = 0.1129. Per encoder: contriever 0.0850, mpnet 0.0928, qwen3emb06 0.0921; none informative. New median level L(10, 1024) = 0.0594, against 0.0430 for the S1.1 LM-token pairs. |
| **P2c** within a pair, averaging 1 to 5 captions moves the slope along the curve | **uninformative** | Guard: median ln(L5/L1) = 0.178 < 0.3. Per encoder: contriever 0.236, mpnet 0.148, qwen3emb06 0.146; none informative. |

**The three labels come out the same in three implementations.**
- `VERDICTS_E0.json` (`5b04e460…`): frozen evaluation code through the ADDENDUM 1 wrapper.
- e0-review recompute (`414fa020…`): own code, no E0, S1.1 or hrhv2 import. Labels equal; statistics within 2.2e-15. The frozen line, s_res and ViT competitors re-derived from S1.1 within 4.4e-16. The overlap step re-derived from the stored neighbour lists in 9,450 cells, difference 0.0.
- e0-indep V-f (`1bcc4733…`, `fc998a32…`): own neighbour tables from the stored embeddings and ViT features. Overlap counts are equal in 9,450 of 9,450 cells (arms P, D1–D5), for both `obs` and `obs_allH2`. Tie counts are equal, and no anchor needed the near-tie clause. 934 statistics agree within 9.1e-15 (tolerance 1e-12), and all 41 labels are identical. Limit: the check works on per-cell total overlap counts, because the obs files hold no per-anchor lists.
- Frozen `e0_guard.py audit`: PASS. All 87 new-pair files were born after the GUARD CLEAN line (14:06:50Z).

## 2. Binding readings (plan §13)

1. **P1 holds (§13.1).** Agreement at fixed k/m stays stable at k ≥ 10 for sentence-encoder text sides too (median sum 0.0060). The evaluation size acts as a scale parameter for this family as well.
2. **P2b × P2c = uninformative × uninformative (§13.2).** Nothing new about the level–slope regularity. Under §1.4 no remark goes into `app:scale`. Neither guard passed, so neither label says anything for or against the regularity. The P2b guard behaved as its frozen operating characteristics predicted: at a median level near 0.06 it returns "uninformative" in 84–88% of runs.
3. **Theorem 1** is exact; no E0 outcome changes it. The P1-fails modifier does not apply. The **B6 tension row** is not triggered, because P2b did not fail.
4. **Level wording (§13.3): "between"** in both comparison arms. Arm P: sentence encoders 0.059, LM tokens 0.043, vision–vision 0.189. Arm C: 0.066, 0.046, 0.196. Consequence: the Limitations paragraph says that our vision–language levels describe language-model token features with our protocol, citing Schnaus et al. and E0.
5. **R1 clip caveat (§13.3): not triggered.** The encoders' own-10-NN overlap of `clip` against `l2` is 0.937 (contriever), 0.959 (mpnet) and 0.964 (qwen3emb06), all ≥ 0.8. The clip does move the three S1.1 LMs (0.51, 0.64, 0.76), but the rule does not apply to them; this is context only.
6. **Without Q1 (§13.3).** No sentence may attribute a level difference between LM-token and sentence-encoder pairs in the AVG arms to the encoder rather than to the caption average. Only the C1 arms (P, C) compare encoders cleanly.
7. **§13.4 draft sentence, filled in** (a proposal; the wording stays the author's). Neither P2b bracket applies:
   > With three sentence encoders as the text side, the vision--language median at $k=10$ and $m=1024$ is $0.059$, against $0.043$ with language-model token features, and agreement at fixed $k/m$ again stays close to stable for $k\ge10$ (median sum of elasticities $0.006$).
   The §13.4 Limitations draft ("sentence encoders reach higher agreement …, as Schnaus et al. report") can be read as if the encoders explained the gap to Schnaus et al. E0 shows that they close only a small part of it (§3.5). The review's replacement wording is post hoc (§6).

## 3. Secondary and descriptive results (no confirmatory weight)

Plan §8 applies: an "uninformative" confirmatory label is reported as such, and no secondary cell replaces it. The full code-produced table is `eval/E0_tables.md` (plan B8.5).

### 3.1 Every secondary label

| analysis | cell | label | statistic |
|---|---|---|---|
| P1 | ctrold | holds | median sum 0.0108; informative 25 |
| P1 | D1 | holds | median sum 0.0122; informative 15 |
| P1 | D2 | holds | median sum 0.0091; informative 15 |
| P1 | D3 | holds | median sum 0.0110; informative 15 |
| P1 | D4 | holds | median sum 0.0116; informative 15 |
| P1 | A | holds | median sum 0.0119; informative 15 |
| P1 | Aall | holds | median sum 0.0120; informative 15 |
| P1 | C | holds | median sum 0.0068; informative 15 |
| P1 | CA | holds | median sum 0.0151; informative 15 |
| P1 | CAall | holds | median sum 0.0151; informative 15 |
| P1 | L | holds | median sum 0.0061; informative 15 |
| P1 | Pdd | holds | median sum 0.0045; informative 15 |
| P2b | P\|k1 | uninformative | median \|g\| 0.0758 (guard 0.2208); λ̂ — |
| P2b | P\|k100 | uninformative | median \|g\| 0.0655 (guard 0.0895); λ̂ — |
| P2b | P\|mref2048 | uninformative | median \|g\| 0.0858 (guard 0.1003); λ̂ — |
| P2b | P\|obs_allH2 | uninformative | median \|g\| 0.0921 (guard 0.1129); λ̂ — |
| P2b | P\|obs_clean | uninformative | median \|g\| 0.0922 (guard 0.1129); λ̂ — |
| P2b | A | holds | median \|g\| 0.1215 (guard 0.1129); λ̂ 0.957 |
| P2b | Aall | holds | median \|g\| 0.1216 (guard 0.1129); λ̂ 0.956 |
| P2b | Pdd | uninformative | median \|g\| 0.0923 (guard 0.1129); λ̂ — |
| P2b | L | inconclusive | median \|g\| 0.1080 (guard 0.0953); λ̂ 0.409 |
| P2b | C | uninformative | median \|g\| 0.0981 (guard 0.1201); λ̂ — |
| P2b | CA | uninformative | median \|g\| 0.1199 (guard 0.1201); λ̂ — |
| P2b | CAall | uninformative | median \|g\| 0.1197 (guard 0.1201); λ̂ — |
| P2c (descriptive, §7.4) | k1 | holds (if the rule were applied) | median γ 0.279 |
| P2c (descriptive, §7.4) | k100 | uninformative (if the rule were applied) | median γ 0.285 |
| P2c (descriptive, §7.4) | obs_allH2 | uninformative (if the rule were applied) | median γ 0.437 |
| P2c (descriptive, §7.4) | obs_clean | uninformative (if the rule were applied) | median γ 0.432 |

Secondary arm A (AVG5, `clip`) passes the P2b guard by a small margin and gives "holds" (λ̂ 0.957). Arm L gives "inconclusive" (λ̂ 0.409). Every other P2b cell is "uninformative". None of these changes the primary P2b label.

### 3.2 Statistics the plan says are always reported
- **P2b mixing weight (descriptive).** λ̂ = 0.4376. Per encoder: λ_e = 0.3011 (contriever), 0.4955 (mpnet), 0.4750 (qwen3emb06). These values come from V-f and the review (which agree within 1e-12), since VERDICTS leaves them empty (§5.2). Leave-one-encoder-out: 0.403–0.485; leave-MPNet-out: 0.403. Leave-one-ViT-out: 0.392–0.558. The descriptive Wilcoxon test gives W+ = 54, p = 0.640 against the ViT-specific competitor, and W+ = 62, p = 0.467 against the constant one. Median residual against the line: -0.034; median b(10) minus the ViT competitor: +0.053.
- **P2c (descriptive).** Median γ = 0.435. Per encoder: 0.316 (contriever), 0.346 (mpnet), 0.516 (qwen3emb06); leave-one-encoder-out 0.325–0.448. For comparison, the cross-class line has 0.228 and the S1.1 within-class coefficient is 0.080. Ladder medians over the 15 pairs for n = 1…5 captions: L = 0.0595, 0.0645, 0.0656, 0.0667, 0.0668; b(10) = -0.509, -0.482, -0.457, -0.446, -0.432.
- **P1 (reported with it, §7.1).** The median sum over all k (1, 10, 100) is 0.058; on k ∈ {1, 10} it is 0.091. corr(e_k, −e_m) over the 30 VL pairs of arm P is 0.9940. Per-draw median sums run from -0.0108 to 0.0257.

### 3.3 Levels and slopes per arm (class medians; L = s̄ at k = 10, m = 1024)

| arm | class | n | L(10,1024) median [min, max] | L(100,8192) | median b(10) | median sum k∈{10,100} | falling at k = 1 / 10 / 100 |
|---|---|---|---|---|---|---|---|
| ctrold | cross-modal | 15 | 0.0462 [0.0361, 0.0866] | 0.0529 | -0.564 | -0.0007 | 15 / 15 / 15 |
| ctrold | cross-vision | 10 | 0.1963 [0.1290, 0.2823] | 0.2119 | -0.208 | 0.0138 | 3 / 10 / 10 |
| ctrold | seed-seed | 4 | 0.4100 [0.3816, 0.7157] | 0.4226 | -0.004 | 0.0058 | 0 / 0 / 1 |
| P | cross-modal-sentence | 15 | 0.0594 [0.0460, 0.1375] | 0.0671 | -0.525 | 0.0060 | 15 / 15 / 15 |
| D1 | cross-modal-sentence | 15 | 0.0595 [0.0458, 0.1320] | 0.0675 | -0.509 | 0.0122 | 15 / 15 / 15 |
| D2 | cross-modal-sentence | 15 | 0.0645 [0.0517, 0.1492] | 0.0724 | -0.482 | 0.0091 | 15 / 15 / 15 |
| D3 | cross-modal-sentence | 15 | 0.0656 [0.0534, 0.1554] | 0.0739 | -0.457 | 0.0110 | 15 / 15 / 15 |
| D4 | cross-modal-sentence | 15 | 0.0667 [0.0536, 0.1578] | 0.0747 | -0.446 | 0.0116 | 13 / 15 / 15 |
| A | cross-modal-sentence | 15 | 0.0668 [0.0536, 0.1607] | 0.0752 | -0.432 | 0.0119 | 12 / 15 / 15 |
| Aall | cross-modal-sentence | 15 | 0.0667 [0.0536, 0.1608] | 0.0752 | -0.432 | 0.0120 | 12 / 15 / 15 |
| C | cross-modal-sentence | 15 | 0.0662 [0.0498, 0.1454] | 0.0762 | -0.521 | 0.0068 | 15 / 15 / 15 |
| CA | cross-modal-sentence | 15 | 0.0740 [0.0586, 0.1687] | 0.0846 | -0.433 | 0.0151 | 14 / 15 / 15 |
| CAall | cross-modal-sentence | 15 | 0.0739 [0.0586, 0.1686] | 0.0846 | -0.432 | 0.0151 | 14 / 15 / 15 |
| L | cross-modal-sentence | 15 | 0.0591 [0.0459, 0.1296] | 0.0675 | -0.523 | 0.0061 | 15 / 15 / 15 |
| Pdd | cross-modal-sentence | 15 | 0.0593 [0.0459, 0.1371] | 0.0672 | -0.524 | 0.0045 | 15 / 15 / 15 |
| S1.1-clip | cross-modal | 15 | 0.0430 [0.0327, 0.0981] | 0.0491 | -0.571 | 0.0044 | 15 / 15 / 15 |
| S1.1-clip | cross-vision | 10 | 0.1885 [0.1288, 0.2656] | 0.2049 | -0.214 | 0.0080 | 3 / 10 / 10 |
| S1.1-clip | seed-seed | 4 | 0.5253 [0.3601, 0.6918] | 0.5384 | -0.014 | 0.0049 | 0 / 0 / 1 |
| S1.1-l2 | cross-modal | 15 | 0.0405 [0.0295, 0.0763] | 0.0453 | -0.566 | -0.0020 | 15 / 15 / 15 |
| S1.1-l2 | cross-vision | 10 | 0.1852 [0.1260, 0.2670] | 0.2005 | -0.221 | 0.0096 | 3 / 10 / 10 |
| S1.1-l2 | seed-seed | 4 | 0.5262 [0.3833, 0.6966] | 0.5435 | -0.012 | 0.0056 | 0 / 0 / 1 |

- **Within each ViT, every encoder is above every LM** (arm P vs S1.1 `clip`):
  - augreg_b: LMs 0.048–0.060, encoders 0.071–0.078.
  - clip_b: LMs 0.033–0.039, encoders 0.046–0.050.
  - dinov2_b: LMs 0.041–0.050, encoders 0.056–0.061.
  - mae_b: LMs 0.033–0.039, encoders 0.047–0.049.
  - siglip_b: LMs 0.074–0.098, encoders 0.116–0.137.
- **By encoder (arm P).** mpnet L 0.0613, b(10) -0.512; contriever 0.0561, -0.546; qwen3emb06 0.0594, -0.522.
- **Class median at k = 10 as m grows.** Sentence encoders: 0.105 (m = 256), 0.059 (1024), 0.0132 (16,384). LM tokens: 0.091, 0.043, 0.0085.
- **Slope order equals level order (P3f).** At k = 1, 10 and 100 both orders are seed > vision > sentence-encoder VL > LM-token VL. At k = 10 the median slopes are −0.014, −0.214, −0.525 and −0.571.
- **P2 (descriptive).** Spearman ρ(L, b(10)) over the 30 VL pairs is 0.790 (within-ViT permutation p = 1.0e-05, R = 100,000). Over the 6 text models ρ = 0.943 (exact p = 0.0083). As the plan anticipated (§7.2, B5), P2 cannot separate "the level sets the slope" from "the level partly sets the slope".

### 3.4 Caption averaging and preprocessing (P3b, P3c; medians of ln-ratios)
- AVG5 − C1: ln L +0.156 (×1.168) under `clip`, +0.132 (×1.141) under `ctr`. Δb(10) is +0.101 and +0.099.
- AVG5 − N1 (`clip`): ln L +0.178 (×1.195), Δb(10) +0.081. ALL − AVG5: +0.0000 (`clip`), -0.0002 (`ctr`).
- `ctr` − `clip`: ln L +0.089 for the new pairs. For the old pairs it is +0.068 (VL), +0.046 (vision) and -0.090 (seed). `l2` − `clip`: -0.006 for the new pairs.
- Shared-k/m profile (P3g): the k = 1 cells are 0.837–0.874 of the k = 10 cells at the same k/m. The k = 100 cells are 0.989–1.055.

### 3.5 Comparison with Schnaus et al. (P3h; arm CA = `ctr` × AVG5, interpolated to m = 10,000)
- **DINOv2-B × MPNet in arm CA:** 0.077 at k = 100 and 0.028 at k = 10. Arm P gives 0.061 and 0.018. Schnaus et al. report about 0.38 and 0.07–0.19; these are the critic's readings of their figures, approximate, and were disclosed before the freeze.
- **Arm CA at k = 100:** median 0.077, maximum 0.170. Our vision–vision pairs in the same cell under `ctr` reach a median of 0.203 and a maximum of 0.294. That is still below their vision–language value.
- **Fixed-m ratio s̄(100, m)/s̄(10, m) for DINOv2-B × MPNet:** 2.74 and 2.64 in arm CA (m = 8,192 and 16,384), 3.42 and 3.56 in arm P. This is the k-dependence at fixed m that P1 implies (e_k ≈ −e_m).
- **Reading (descriptive).** The encoder, caption averaging and centering together close only a small part of a roughly fivefold gap. Which protocol difference explains the rest is open (§6); X3 is the run that would test it.

## 4. Verification record
- **V-f (e0-indep), STATUS line 341: AGREE.**
  - The design (draws, halves, anchors) was rebuilt bitwise.
  - Distance matrices for 23 inputs (5 ViTs and 3 encoders × {C1, N1–N5}) were built through `hrhv2.prep`, with V-f's own caption average.
  - Each H2 row of each prefix was fully argsorted with the key (D, position).
  - Exact text-side ties at the k boundary come from duplicate captions: 1,765 anchor-cells in P, 7,645 in D1 and 100 in D2, equal in both implementations. No anchor was accepted under the near-tie clause.
  - Computing the statistics from V-f's own obs instead of E0's obs changes them by at most 2.4e-15.
- **Review (e0-review), STATUS lines 339 and 340.**
  - Labels and statistics were recomputed with independent code.
  - The review's own order audit passes. The STATUS prefix hashes check out and the milestones are in order.
  - The 15 named hashes and the 22 manifest files are unchanged.
  - Feature, cache and calibration files were all born before the GUARD CLEAN line, and the new-pair files after it.
  - Limit: birth times order the files in the project directory only. They cannot rule out a computation done elsewhere, and no sign of one was found.

## 5. Corrections and findings
1. **Caption source (found by the review; rechecked here by count).** C1 is the S1.1 caption, the `text` column of `items.parquet`. It equals the **last** official caption of its image by annotation id (= last in file order) for 16,384/16,384 pool images, and the first (lowest-id) caption for 0/16,384.
   - **What this corrects.** v4 `app:exp-s11` says "the first caption", which is wrong. PROTOCOL §2 and plan R.3(6)/§13.5 say "sampled at random by the dataset author". The data rule that out: a random pick would make about one image in five carry its first caption, and none does.
   - **Duplicates.** In 6 images C1 also equals an earlier duplicate caption. That is why the review's first-match rank histogram shows ranks 1 and 3 for two images.
   - **Effect on results: none.** Arm D1 (N1, the lowest-id caption) has median L 0.0595; arm P (C1) has 0.0594.
   - **Wording.** The §13.5 sentence cannot be used with its stated premise. The review's proposal E reads: "For each of 16,384 COCO validation images, the pool holds the caption stored by coco-30-val-2014, which is the last caption of the image in the COCO annotation file."
2. **λ̂ reporting gap (found by V-f).** Plan §7.3 says λ̂ and every λ_e are always reported. For arm P, `VERDICTS_E0.json`, `summary.json` and `E0_tables.md` give `lam None, lam_e {}`, because the guard returned uninformative. The values in §3.2 (λ̂ 0.4376) close the gap. They are descriptive and do not affect the label.
3. **STATUS line 334 lists two hashes in swapped order** (found here). Line 334 lists the ctrold eval tables {cells, drawslopes, slopes, elasticity} as 9c95c59e, 93c1b332, a28a80dd, 675edb30. On disk, slopes = 675edb30 and elasticity = a28a80dd; the VERDICTS `addendum_1` record has them right. All four files are unchanged.
4. **`VERDICTS_E0.json` is writable** (mode 664; the obs, eval, diag and guard files are 664 too). Its sha256 still equals the logged 5b04e460. This role modifies no existing file, so the mode was left as it is; the lead may set it to 444.
5. **V-f disclosures.**
   - V-f read the STATUS tail, which carries the labels, before its own numbers existed, in order to copy the line format. It opened no E0 code, report, review or VERDICTS file before `vf_obs.parquet` existed.
   - One file lies outside the plan paths: `/tmp/vf_designcheck_1049305/vf_design_check.json`. It was not deleted, per the rules.
   - The docstring of `vf_rules.py` mentions a scipy cross-check that never runs. The Wilcoxon p comes from full enumeration of the 2^15 sign patterns.
6. **"+14–17%" mixes two arms.** AVG5 against C1 raises L by 17% under `clip` and 14% under `ctr`. Both are medians of the ln ratio.
7. **Process notes logged earlier.**
   - The COCO archive came through the http fallback the plan allows; its size and md5 were verified.
   - ADDENDUM 1, a fix to the write step, was logged before the guard.
   - The first `e0_calib` attempt wrote nothing because `eval/` was missing; it was rerun unchanged.
   - The CTR CALIBRATION milestone was logged a second time, in the exact form the guard parses, with the same hash.

## 6. Exploratory and post hoc items (none enters a label)
1. P1's restriction to k ≥ 10 was chosen after seeing S1.1 (2026-10-10), before any E0 score (plan §7.1 disclosure).
2. The e0-review exploratory file, labelled EXPLORATORY and post hoc, holds three analyses. (a) P1 on m ≥ 2048 only: median sum 0.0046, with the k = 100 cells at ≥ 2.25× chance. (b) The k = 1 comparison with S1.1: median sum on k ∈ {1, 10} 0.091 against 0.055, and over all k 0.058 against 0.030. (c) A per-ViT P2b decomposition for arms P and A. The new-pair sums themselves are pre-specified (plan §7.1); the comparison with S1.1 and its reading (k = 1 is where Theorem 1's square-root term is relatively largest) are post hoc.
3. Some readings of descriptive numbers were made after the labels were known. The new slopes sit about halfway between the level line and the ViT values (λ̂ about 0.44). Caption averaging flattens the slope faster than the level alone predicts (median γ 0.435 against 0.228). Arm A's λ̂ of 0.957 averages ViT-level deviations of opposite sign.
4. The level gap to Schnaus et al. is roughly fivefold. Attributing it to the vision features (mid-depth CLS here, pooled output tokens there) is a hypothesis; X3 would test it and has not run. Schnaus et al.'s values are readings of their figures.
5. Paper wording. The e0-review proposals A–F (OLD → NEW; v4 unedited) use wording that plan §13 does not fix, except where they fill §13.4. They are post hoc and, if used, are labelled so in the manuscript source comments (plan B8.3). Proposal D departs from the §13.4 Limitations draft, because that draft reads as if the encoders explained the gap; proposal E replaces §13.5.
6. The caption-source fact was checked after the results (by the review, and rechecked here). It is a fact about the data, not a statistic, and it changes no result.

## 7. Runs not done
- **X1.** Exploratory (plan §7.6). S1.1's null N1 on the 15 arm-P pairs through the frozen hrhv2.predict_v11. Code not written.
- **X2.** Exploratory (plan §7.6). Levina–Bickel intrinsic dimension per model and caption count n, correlated with the P2b residuals and with γ. Code not written.
- **X3.** Exploratory (plan §7.6). The protocol bridge: DINOv2-B with rel-1.0 CLS, rel-1.0 patch mean and rel-0.5 patch mean, each with the 3 encoders under ctr × N5 (9 pairs, all to be reported). It needs new DINOv2 features on the GPU. It is the direct test of the vision-feature explanation for the level gap.
- **Q1.** Conditional (plan §15). E0-LMavg: the 3 S1.1 LMs re-embedded on all official captions (about 0.6–1.5 GPU-h, local snapshots, no download). It needs a STATUS line 'E0 Q1 APPROVED' that quotes the author, and none exists. If it is approved now, every Q1 output carries 'decision to run made after the guard'.
- **B7.** Conditional (plan §15). The MPNet membership check, a 23.9 MB text download of sentence-transformers/coco-captions. It needs 'E0 B7 APPROVED', and none exists.
- **X4, X5.** Computed by the frozen evaluation as exploratory columns (VERDICTS descriptive.x4_exploratory; slope_obs_cc in eval/E0_slopes_<arm>.parquet); not interpreted.

## 8. Compute (actual use; no GPU cap was set)
- **GPU: 0.38 GPU-h for all of E0.** All of it was the e0-extract embedding run: 0.368 h in-loop, 0.38 h process wall including loading and the 0.7-min smoke. Every other role used 0. The plan estimated 0.4–1.0 GPU-h (central 0.6).
- **CPU as logged per role:**
  - e0-critic: about 1 core-min.
  - e0-freeze: 22 s for the constants, plus about 0.4 core-h for code freeze and validation.
  - e0-extract: pinned to 6 cores; not logged separately.
  - e0-run: about 0.5 core-h for the caches; line 337 gives "E0 total so far about 1 core-h".
  - e0-review: about 3 core-min.
  - e0-indep: 0.45 core-h.
  - This closing step: under 1 CPU-min in total (this script, its dry runs and read-only checks; 1 thread, taskset 0-5, nice 10); GPU 0.
- **E0 CPU total: about 1.5 core-h.** It is 1.9 core-h if the line-337 total leaves out the freeze role's 0.4. Never more than 6 cores were used (taskset 0-5); that limit is the task rule, not an E0 cap.
- **Disk at closing:** `features/coco_e0` 1.9 GB, `extract/e0_sentence/hf_home` 2.0 GB, `results/E0_sentence/cache` 2.9 GB, `analysis/e0_sentence/indep` 2.2 GB.

## 9. Sources (sha256, verified against STATUS before reading)

| file | sha256 |
|---|---|
| `analysis/e0_sentence/ANALYSIS_PLAN_E0.md` | `e859c6ea52f941b18ca27b83150f489dded0085a4d99218d909b36c19603f0ee` |
| `analysis/e0_sentence/ANALYSIS_PLAN_E0_ADDENDUM_1.md` | `dcf0ecfc9b3e79063782a9e0d51af7db3e91c4719dee5ab8cec9490cb2509aad` |
| `analysis/e0_sentence/CODE_MANIFEST_E0.json` | `6b0111c85aa5416b1d7c97b7f75fe2fe413ce8e2741bdb4e852e43e707664b13` |
| `analysis/e0_sentence/addendum1/e0_evaluate_add1.py` | `d0c1ce4af6cff66b099ed9ab1c5dd074bca9f7f2d99324b4d3d0c2805e521da8` |
| `results/E0_sentence/extraction/EXTRACTION_E0.json` | `27bdcb149f1b386be886b9e5539b2b72b352c54be1867a4a42f9a87c6416f8de` |
| `results/E0_sentence/eval/ctr_calibration.json` | `33d1e796fa1a80185b053e237fd93b58e1ccf3f72e84506ecedb484948d63da2` |
| `results/E0_sentence/guard/GUARD_CHECK_20261010T140640Z.json` | `f5099bd2b48b3b4d6d31a235710c99b81913cceee4d0e201feb5a28e7c20af7d` |
| `results/E0_sentence/diag/diag_singlemodel.json` | `e60bf07c222312f6073f818211423270d3edf29656ada576e85d7863d6a35c7f` |
| `results/E0_sentence/VERDICTS_E0.json` | `5b04e46098e39056f6e9b2d534d53fa54777b1670ccfa68442cb7ef3fd1568a7` |
| `results/E0_sentence/eval/E0_tables.md` | `22b6b04b7b400b75a6c936306525d32bf90256dceb275a9a606d3a455e268a66` |
| `results/E0_sentence/summary.json` | `b1d6fcddeb706ba607246207d8d70b6ebc215d8f412f76877955306f1ea822b2` |
| `results/E0_sentence/REPORT.md` | `a7ec345754e034a30bbf89f87678678169ec2bb74974e0994d4e75c6f8f7cdb4` |
| `analysis/e0_sentence/review/e0_review_recompute.py` | `c0fcf8d243e2c18450ecc0522fb746350c5b2ff56459afcf0bb725b4c51a7527` |
| `analysis/e0_sentence/review/review_recompute_20261010T141910Z.json` | `414fa020230f0e84f6f3cffb2ac2cbb7b7600387570b79b4af2950b1be344d9a` |
| `analysis/e0_sentence/review/audit_20261010T142040Z.json` | `17e837e2f44fbf9ed95dd42151cf2b28bbe09a11ad9409fcb3812f3dcd487e58` |
| `analysis/e0_sentence/review/e0_review_explore.py` | `583cd61ae372c183c05e523fac011759f843927dad67bd041bd1df900d2d262e` |
| `analysis/e0_sentence/review/review_explore_20261010T142122Z.json` | `abbe5cf2c2395f1b18636a27071410d3b9c787ee2eeac05e682cabda3d397e1b` |
| `analysis/e0_sentence/review/e0_review_paper_numbers.py` | `0be7d4099d90e292487aea4bdd1b40ac3539e9e74fcbd4dcdf9045e89e3c7068` |
| `analysis/e0_sentence/review/review_paper_numbers_20261010T142548Z.json` | `4bbe71cd0206c612ff58bd1c4c4b23ce065c971c4a15cb934e4091a706843286` |
| `analysis/e0_sentence/indep/run_20261010T143557Z/INDEP_MANIFEST.json` | `2f636640ef8f539e3ee1cc44130b37cff694a212cefa5567f80669aae89f933e` |
| `analysis/e0_sentence/indep/run_20261010T143557Z/vf_obs.parquet` | `0f153b4f778c4eb9d4d8929674bec36358528da97c7ded8806e41a6aa2f5ce37` |
| `analysis/e0_sentence/indep/run_20261010T143557Z/vf_compare_obs.json` | `fc998a323a73aa37f7bc85b1e8e75a5653d96c984e67e039d04beab4933d7de9` |
| `analysis/e0_sentence/indep/run_20261010T143557Z/vf_stats.json` | `3aaca1d8438bf21543e52918edfbf84f7376599c537bc19cd78e0a24d036367c` |
| `analysis/e0_sentence/indep/run_20261010T143557Z/vf_compare_verdicts.json` | `1bcc47331298f8030b8ae5ba2161cb145d6f1508b445a8c1ebc3687e419a00aa` |
| `analysis/e0_sentence/indep/run_20261010T143557Z/vf_compare_extra.json` | `b865827e161f727aac5f910db9ccc76d34168d442041779cc2ce9ac27b78c5a4` |
| `features/coco_e0/captions_index.parquet` | `ee979cc81d7d69be888438b72c2cf921489f1475db1b07c23c747caec60ed76b` |
| `results/E0_sentence/obs/obs_ctrold.parquet` | `f08e6699d1b23b27ed2ea262573fe3cf2c497eecdee2a798e357ac0d6f882a03` |
| `results/E0_sentence/obs/obs_P.parquet` | `1e13c7df10571a250de0faf44f9767ca4a835f7d9aff960a96f5b51ea68c6cbc` |
| `results/E0_sentence/obs/obs_D1.parquet` | `408a01b18a89cb13aa6ff0d5b23ccd328c91ea926e59314c753543f574b332fa` |
| `results/E0_sentence/obs/obs_D2.parquet` | `5c29e8c7c74a969b29c679677b866b9a41a343681267f1baf36370b8eabe9fc7` |
| `results/E0_sentence/obs/obs_D3.parquet` | `767d31459bd322e365aebf0bf5764157885efd21fff7558ddda865f271901c79` |
| `results/E0_sentence/obs/obs_D4.parquet` | `684444c9d90b92ac57891beccc5b724638f1f7ed6dc4b860244ba13c0e94763a` |
| `results/E0_sentence/obs/obs_A.parquet` | `b72e237cdc556d9be3d4e9a6fe1bd351a047ce79dc04544a734511d59b9f40b5` |
| `results/E0_sentence/obs/obs_Aall.parquet` | `ce87493df67957c92d8ea91b26103ef09b37b867fd70c18312d25d56398abda3` |
| `results/E0_sentence/obs/obs_C.parquet` | `50238287636ba56dac5bda4806ad0c1491da06fed7b61b0c5bfb7832ccc0ce83` |
| `results/E0_sentence/obs/obs_CA.parquet` | `1fc081920d822a307c879f4cafd071b54e740235ca69f51fbc9279a38ca44118` |
| `results/E0_sentence/obs/obs_CAall.parquet` | `431aeed6705f98bc4837097d5453c73547acb511813224916e11d06622f5351d` |
| `results/E0_sentence/obs/obs_L.parquet` | `cdb7be449c64bfc141055289ca601a47aeb39d09ef6c777892a9456e80065e90` |
| `results/E0_sentence/obs/obs_Pdd.parquet` | `969d6561c652d7ef55661f8be96548683d674fb73311287bd20d9ce9dabb1a8d` |
| `results/E0_sentence/eval/E0_cells_ctrold.parquet` | `9c95c59ed7c8adfb4266a3ba176b529307d515ba24cafdf5def93fe521ea34cc` |
| `results/E0_sentence/eval/E0_drawslopes_ctrold.parquet` | `93c1b3328873ee07a1b383673e8d403fc69f74ffb273966a780d8f03b0064f27` |
| `results/E0_sentence/eval/E0_slopes_ctrold.parquet` | `675edb30ed3d8b380c003acbf4343f4ae1e37b178bc75813add0d2708bcbfd41` |
| `results/E0_sentence/eval/E0_elasticity_ctrold.parquet` | `a28a80dd2d94facc92af345b039470bc3eb35dae98142404ab348e3f5fb83fd2` |
| `features/coco_pool/items.parquet` | `2e287dbcbd56af25dad6d8b331b3f82f331d10f02d2368f246df83f72c234292` |
| notes/v2_assessment_20261007/stage1_results/e0_sentence/review.md (author's notes, e0-review report) | `c20b4f7d937b987a4183380eaca2246ebce8151e8a79291ceafa707b3e8b661a` |
