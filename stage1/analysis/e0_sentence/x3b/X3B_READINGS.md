# X3b (last-block vision features, CPU only): readings fixed before any X3b cross-model score

- **Role:** `x3b-run`. **Written:** 2026-10-10, about 21:20 UTC. No single-model cache and no cross-model score with a last-block vision feature had been computed by this role when this file was written. The only last-block scores that existed are X3's DINOv2-B rows (ctr x N5), which were read before writing (X3_REPORT.md, X3_results.json).
- **Status:** EXPLORATORY in full. Follow-up to X3 (ANALYSIS_PLAN_E0.md v1.0 section 7.6; X3_READINGS.md sha256 `0151bf9a...`). Nothing here is confirmatory, and no S1.1, E0 or X3 label changes.
- **Approval:** the author approved this check on 2026-10-10 ("做", reply to "X3 的复核建议再查一项，只用 CPU、几分"). CPU only; no GPU process is started. Limits from the task, not from this role: at most 6 cores (`taskset -c 0-5`), `nice 10`, 1 BLAS thread per worker. This role sets no budget cap of its own; actual use is reported.
- **What was read before writing:** the stored-feature sidecars, the S1.1 and E0 frozen code (s11copy, e0_*.py), X3's files, and the frozen mid-depth outputs that give the reference medians in section 4 (S1.1 cells and slopes for `clip` and `l2`, E0 cells, slopes and elasticities for `ctrold` and `CA`).

## 1. Inventory of stored features (STEP 0)

All on the COCO pool of 16,384 images (`features/coco_pool/`, `items.parquet` sha256 `2e287dbc...`), stored at 4 relative depths {0.25, 0.5, 0.75, 1.0}, float32, sha256 checked against each sidecar before use.

| model key | stored file | layers on the stored axis (axis 0..3) | token / pooling | used by S1.1, E0 |
|---|---|---|---|---|
| dinov2_b, clip_b, mae_b, augreg_b | `<slug>.npy` (16384, 4, 768) | outputs of blocks 2, 5, 8, 11 (`blocks.i.add_1`), no final norm | CLS token (index 0) | axis 1 (block 5) |
| siglip_b | `<slug>.npy` (16384, 4, 768) | blocks 2, 5, 8, 11, no final norm | no CLS token: mean of all patch tokens, no MAP head | axis 1 |
| the 5 ViTs | `<slug>.tokmean.npy` | blocks 2, 5, 8, 11 | mean of patch tokens | X3 only (DINOv2-B); **not used here** |
| bloomz_1b7 | `bigscience__bloomz-1b7.npy` (16384, 4, 2048) | hidden_states 6, 12, 18, 24 (24 = after the final norm) | mean over all tokens of the S1.1 caption | axis 1 (hidden_states 12) |
| openllama_3b | `openlm-research__open_llama_3b.npy` (16384, 4, 3200) | hidden_states 7, 13, 20, 26 (26 = after the final norm) | same | axis 1 (13) |
| pythia410m_s1..s9 | `EleutherAI__pythia-410m-seed<s>.npy` (16384, 4, 1024) | hidden_states 6, 12, 18, 24 (24 = after the final norm) | same | axis 1 (12); only s1 is in the vision-language pairs |
| the 11 LMs | `<slug>.lm_aux.npz` key `last_prenorm` | last layer before the final norm | same | **not scored here** (an auxiliary variant of the rel-1.0 axis) |
| mpnet, contriever, qwen3emb06 | `features/coco_e0/<enc>.caps.npy` + `captions_index.parquet` | one output embedding per caption | E0 five-caption average N5 | E0 arm CA; E0 ctr caches `<enc>.n5` reused read only |

- **Captions.** The language models saw one caption per image, the `text` column of `items.parquet` (the "S1.1 caption"). The sentence encoders use the first five captions per image by annotation id, averaged (N5), as in E0 arm CA and X3.
- **The "last block"** is the stored rel-1.0 axis (axis 3): the output of block 11 of 12, before the final norm, with the same token as S1.1 (CLS for four ViTs, patch mean for SigLIP). For DINOv2-B this is exactly X3's `x3_cls_r10`. Only the depth changes against S1.1/E0.
- **Existing single-model neighbour caches reused (read only):** S1.1 `results/S1_1/v11/cache/{clip,l2}/<model>` (all models at rel 0.5); E0 `results/E0_sentence/cache/ctr/<model>` (ViTs and LMs at rel 0.5, `<enc>.n5`); X3 `results/E0_sentence/x3/cache/ctr/x3_cls_r10` (DINOv2-B at rel 1.0 under ctr). No cache at any other depth exists in the project.

## 2. Pairs and protocols

Design `e0_common.design("clip")` (= S1.1 design): 5 nested draws, m in {256, 512, ..., 16384}, k in {1, 10, 100}, S1.1 anchors (column `obs`). Preprocessing keys:
- `clip`: S1.1 primary, prh_clip+l2 with ONE clamp threshold per model computed over all 4 stored layers on all 16,384 rows (`pool-stored`), so the threshold is the same whichever depth is used; D through `s11_common.model_D(name, "prh_clip+l2", "pool-stored", rel=r)`.
- `l2`: S1.1 secondary, `s11_common.model_D(name, "l2", None, rel=r)`.
- `ctr`: E0, l2 -> subtract the float64 pool mean -> l2 (`e0_common.ctr_features`), D = `features_to_D(., "none")`.

| set | question | preprocessing | vision side | text side | pairs | mid-depth reference (same pairs at vision rel 0.5) |
|---|---|---|---|---|---|---|
| **A1** | **Q-a, binding** | clip | 5 ViTs, rel 1.0 | bloomz_1b7, openllama_3b, pythia410m_s1, rel 0.5 (S1.1 caches) | 15 | S1.1 `obs_clip` cross-modal rows (median L(10,1024) 0.04295) |
| A2 | Q-a, other LM layers, reported separately | clip | 5 ViTs at rel 1.0 **and** at rel 0.5 | the 3 LMs at rel 0.25, 0.75, 1.0 | 3 layers x 2 depths x 15 = 90 | the rel-0.5 vision rows of the same LM layer (new; no S1.1 score exists for these LM layers) |
| A3 | Q-a sensitivity, descriptive | l2 | 5 ViTs, rel 1.0 | 3 LMs rel 0.5 (S1.1 l2 caches) | 15 | S1.1 `obs_l2` cross-modal rows |
| A4 | Q-a sensitivity, descriptive (same preprocessing as Q-b) | ctr | 5 ViTs, rel 1.0 (the B caches) | 3 LMs rel 0.5 (E0 ctr caches) | 15 | E0 `obs_ctrold` cross-modal rows |
| **B** | **Q-b, binding** | ctr | 5 ViTs, rel 1.0 (DINOv2-B: X3 cache reused) | mpnet, contriever, qwen3emb06, N5 (E0 ctr caches) | 15 | E0 `obs_CA` rows |

- The 3 DINOv2-B pairs of B are X3's `x3_cls_r10` pairs, already known (L(100,10000) 0.4289 / MPNet). They are counted in B as the task asks; the share over the 12 other pairs is reported beside it.
- New single-model caches: clip ViT rel 1.0 (5), clip LM rel 0.25/0.75/1.0 (9), l2 ViT rel 1.0 (5), ctr ViT rel 1.0 for the four ViTs other than DINOv2-B (4): 23 caches, written under `results/E0_sentence/x3b/cache/<pre>/<model>.r<rel>/` by the frozen `e0_prep.build_cache`.

## 3. Quantities (frozen E0 code, imported, never edited)

Cells from the frozen S1.1 `pair_rows`; draw means, slopes, elasticities and levels from the frozen `e0_evaluate.arm_tables`, `elasticities`, `levels`. For every pair, column `obs`:
- levels L(10, 1024), L(100, 8192), and **L(100, 10000)** (k = m/100 at m = 10,000) by the E0/X3 interpolation L = exp(ln s(8192) + (ln 10000 - ln 8192)/ln 2 x (ln s(16384) - ln s(8192))), each with the eps floor;
- slopes b(k), k in {1, 10, 100} (`slope_obs`, mean over draws of the OLS slope of ln s on ln m over the 7 m);
- elasticities e_k, e_m and their sum on k in {10, 100}; "informative" if |e_m| >= 0.10;
- depth ratio R(cell) = L_last(cell) / L_mid(cell) for the same text model, preprocessing and LM layer.
`obs_allH2` versions are reported as descriptive only.

## 4. Reference values (frozen mid-depth outputs, fixed now)

Class medians at vision rel 0.5 (computed with the frozen `e0_evaluate` functions from the frozen cells):

| source | class (n) | L(10,1024) | L(100,8192) | L(100,10000) | b(1) | b(10) | b(100) | median sum k in {10,100} | abs sum <= 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| S1.1 clip | vision-language, LM tokens (15) | **0.04295** | 0.04910 | 0.04379 | -0.522 | -0.571 | -0.598 | 0.0044 | 15/15 |
| S1.1 clip | vision (10) | 0.18852 | 0.20494 | 0.19596 | -0.076 | -0.214 | -0.278 | 0.0080 | 10/10 |
| S1.1 clip | seed (4) | 0.52535 | 0.53841 | 0.53859 | 0.009 | -0.014 | -0.055 | 0.0049 | 4/4 |
| S1.1 l2 | vision-language (15) | 0.04049 | 0.04534 | 0.04026 | -0.461 | -0.566 | -0.619 | -0.0020 | 15/15 |
| E0 ctrold | vision-language (15) | 0.04623 | 0.05288 | 0.04727 | -0.508 | -0.565 | -0.581 | -0.0007 | 15/15 |
| E0 CA | vision x sentence encoder (15) | 0.07398 | 0.08457 | 0.07715 | -0.170 | -0.433 | -0.473 | 0.0151 | 15/15 |

## 5. Readings (fixed now; all exploratory)

### 5.1 Q-a: is the 0.043 level a mid-depth effect? (set A1, binding)
- Per pair R = L_last(10,1024) / L_mid(10,1024); median over the 15 pairs:

| median R | reading |
|---|---|
| >= 2 | "the level is mostly a depth effect" |
| 1.25 to 2 | "the level is partly a depth effect" |
| < 1.25 | "the level is not a depth effect" |

- If the median R is below 0.8, the report adds "the last block lowers the level" (the label stays "not a depth effect").
- Reported beside it (descriptive): the median L_last(10,1024) against 0.04295 and against E0 CA's 0.07398; per-ViT medians of R; R at L(100,8192) and L(100,10000).

### 5.2 Q-a: does the fall with m remain? (A1, binding)
- Median b(10) over the 15 pairs < -0.1: "still falls with m"; otherwise "no longer falls clearly with m". Also reported: the count with b(10) <= -0.1, and b(1), b(100).

### 5.3 Q-a: stable at fixed k/m? (A1, binding)
- |e_k + e_m| on k in {10, 100} <= 0.05 in at least 80% of the 15 pairs (>= 12): "stays close to stable at fixed k/m"; otherwise "not stable at fixed k/m". Also reported: the median sum, the informative count, E0 P1's thresholds applied descriptively (|median| <= 0.03 and >= 80% within 0.05), and sums on {1, 10, 100}.

### 5.4 Other LM layers (A2), other preprocessing (A3, A4)
- Rules 5.1-5.3 are applied unchanged to each LM layer of A2 (15 pairs per layer, R against vision rel 0.5 at the same LM layer) and to A3 and A4. These labels are **descriptive** and reported separately; they do not change the A1 reading.

### 5.5 Q-b: is the depth effect specific to DINOv2-B? (set B, binding)
- Binding cell L(100, 10000), the X3 cell. Share = number of the 15 pairs with L_last > L_mid (strictly):

| share | reading |
|---|---|
| >= 12/15 | "the rise with depth is not specific to DINOv2-B" |
| 6/15 to 11/15 | "the rise with depth holds for some ViTs only" |
| <= 5/15 | "the rise with depth is largely specific to DINOv2-B" |

- Size (descriptive, bands fixed now): median R at L(100,10000) over the 12 non-DINOv2 pairs, read as ">= 2: a large rise, as for DINOv2-B"; "1.25 to 2: a moderate rise"; "< 1.25: little or no rise". Per-ViT median R is reported.
- Also reported (descriptive): the share and median R at L(10,1024) and L(100,8192); the share over the 12 non-DINOv2 pairs.

### 5.6 Q-b: stability and fall with m (B)
- Stability: the rule of 5.3 on the 15 B pairs (binding for Q-b).
- Fall with m: the rule of 5.2, descriptive.

### 5.7 Cross-reading (descriptive, no label)
- Per ViT, median R of A4 (ctr, LM tokens) against median R of B (ctr, sentence encoders): same vision features and preprocessing, different text side.

## 6. Wrapper validation (STOP rule), before any new cross-model score
The wrapper imports the frozen code and never edits it. Before any X3b score:
1. **S1.1 cell.** Rebuild the `clip` caches of dinov2_b and pythia410m_s1 at rel 0.5 through the wrapper's depth-general path into `results/E0_sentence/x3b/validation/cache/`. Each must equal the S1.1 cache bitwise (every array of every `d<draw>_knn.npz`, every `median_D`, `n_zero_offdiag_pool`). The wrapper's rows for `dinov2_b__pythia410m_s1` from the rebuilt caches must equal the `obs_clip.parquet` rows bitwise in every column.
2. **X3 cell.** Rebuild the `ctr` cache of dinov2_b at rel 1.0 through the wrapper's ctr path. It must equal X3's `cache/ctr/x3_cls_r10` bitwise. The wrapper's rows for `x3_cls_r10__mpnet` (rebuilt cache x E0 `mpnet.n5`) must equal the `obs_X3.parquet` rows bitwise in every column.
3. **References.** The wrapper's rows for the four mid-depth reference sets (the 15 vision-language pairs of `obs_clip`, `obs_l2` and `obs_ctrold`, and the 15 pairs of `obs_CA`), recomputed from the existing caches, must equal the frozen rows bitwise.
- If any check fails: STOP, write no X3b score, report.

## 7. Outputs (all new; nothing existing is edited or deleted)
- `analysis/e0_sentence/x3b/`: this file, `x3b_scores.py`, `x3b_report.py`.
- `results/E0_sentence/x3b/`: `cache/`, `validation/` (`VALIDATION_X3B.json`), `obs/` (`obs_X3B.parquet`, `obs_X3B_ref.parquet`, `obs_X3B.json`), `eval/`, `X3B_results.json`, `X3B_REPORT.md`, `logs/`.
- STATUS lines (append only): `[x3b-run] READINGS FIXED`, `[x3b-run] SCORES`, `[x3b-run] REPORT`, each with sha256 and CPU core-hours.
