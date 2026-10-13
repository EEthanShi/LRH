# X3 (protocol bridge): readings fixed before any X3 cross-model score

- **Role:** `x3-run`. **Written:** 2026-10-10, about 20:20 UTC, before any X3 feature was extracted and before any cross-model score involving an X3 feature existed.
- **Status:** EXPLORATORY in full (ANALYSIS_PLAN_E0.md v1.0 §7.6 X3, sha256 `e859c6ea52f941b18ca27b83150f489dded0085a4d99218d909b36c19603f0ee`). X3 makes no confirmatory claim and changes no E0 label.
- **Approval:** the author approved running X3 on 2026-10-10 ("跑"). No GPU or CPU cap is set by this role. Actual use is reported.
- **What the readings use.** Everything below was fixed after reading three things: the frozen plan, E0's finished arm-CA outputs for the three `dinov2_b` pairs (the reference values in §4), and the Schnaus et al. text (paper.txt, pp. 21, 25–27 and 46–48). No X3 feature or score had been computed.

## 1. Pair count (a correction to the task text)
- The frozen X3 is **3 vision features × 3 encoders = 9 pairs**: rel-1.0 CLS, rel-1.0 patch mean and rel-0.5 patch mean.
- This file adds **1 vision feature × 3 encoders = 3 pairs**, labelled as an addition beyond the frozen X3 (§2, `x3add_schnaus224`).
- That makes **12 X3 pairs in total**.
- The computed task text said "10 vision variants × 3 encoders = 30 pairs". That count reads the 9 frozen *pairs* as 9 vision *variants*. No vision variant is invented to reach 30.
- **Validation pairs (not X3 results).** The E0 reference feature (rel-0.5 CLS) is rebuilt through the X3 wrapper, giving 3 more pairs. The rebuild must reproduce E0's existing results bitwise (§6).

## 2. Vision features (DINOv2-B, timm `vit_base_patch14_dinov2.lvd142m`)
- **Pinned weights:** HF snapshot `4685c99dabffe5affac90bd99dbffd25801ae58d`, `model.safetensors` 346,334,872 B, sha256 `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b`. This is the S1.1 pin, loaded from the local cache file with its sha256 verified.

| id | feature | image | source used for scoring |
|---|---|---|---|
| `x3_cls_r10` (frozen) | CLS token (index 0) of the output of block 11 (`blocks.11.add_1`), no final norm | 518 px, S1.1 transform | S1.1 stored `features/coco_pool/vit_base_patch14_dinov2.lvd142m.npy`, axis 3 (sha256 `11a5e377…6b6e`, checked against the sidecar) |
| `x3_tok_r10` (frozen) | mean of the 1369 patch tokens (`[num_prefix_tokens:]`) of the block-11 output, no final norm | 518 px | S1.1 stored `….tokmean.npy`, axis 3 (sha256 `9406da25…3284`, checked) |
| `x3_tok_r05` (frozen) | mean of the patch tokens of the block-5 output, no final norm | 518 px | S1.1 stored `….tokmean.npy`, axis 1 |
| `x3add_schnaus224` (**ADDED, beyond the frozen X3**) | mean over **all 257 output tokens** of `forward_features`, i.e. after the final LayerNorm: the CLS token and 256 patch tokens. This model has no register tokens, so none are included. | **224 px**: resize the short side to 256 (bicubic), center crop to 224, ImageNet mean and std. Position embeddings are resampled from 37×37 to 16×16 by bicubic interpolation (`align_corners=False`, no antialias, as HF `Dinov2Model.interpolate_pos_encoding`), in float32, once at load. | new extraction, `features/coco_e0/x3/` |
| reference `dinov2_b` (E0) | CLS token of the block-5 output, no final norm | 518 px | E0 cache `results/E0_sentence/cache/ctr/dinov2_b` (read only) |

### Why the added variant
- Schnaus et al. describe their vision features on p. 26 (App. B): the models are "used at a resolution of 224 pixels except DINOv3", and all are "mean pooled over the class, register and patch tokens". The App. E setup (p. 46) names "DINOv2 ViT-B/14 (Oquab et al., 2024)". Registers are mentioned only for the RAE generation encoder (p. 27), so the no-register model is taken.
- **None of the 9 frozen features matches that protocol.** Each of them differs from it in at least three ways: no final norm, 518 px, and either CLS only or patch tokens only.
- **The final norm is not stated** in the paper. "Output tokens" is read as the post-LayerNorm sequence, which is HF `last_hidden_state` and timm `forward_features`.

### Differences that stay open (not replicated; listed in the report)
- whether a register model was used;
- the library and its position-embedding interpolation (timm's antialiased resample is checked on 64 images as a side note);
- bf16 storage with ℓ2 normalization (p. 21);
- centering by the mean of a disjoint training half for the aligner (p. 25), against "each side centered" on the 10,000 metric samples (p. 26);
- their text side averages all of the up to 7 captions (p. 25), not the first 5;
- MPNet is the same model, but Qwen3-Embedding is their 8B and our 0.6B;
- their queries are all points of a random 10,000 subsample (pp. 26 and 46), not S1.1's anchors in a nested prefix.

### Re-extraction at the S1.1 settings (a check, not a scoring input)
- The new extraction run also recomputes the three frozen features with the S1.1 code path: 518 px, batch 11, the same hooks, fp32 without TF32.
- Requirement: max |re-extracted − stored| ≤ 1e-3 × max |stored| for each of the three arrays. Otherwise STOP and report before any score. For scale, S1.1's own rerun check gave a max abs diff of 5.1e-5 against a max abs value of 18.5.
- A determinism check is also run: the same 64 images are passed twice in the same batch layout, and the max abs diff is reported.
- A by-product is stored but not scored: the 518 px mean of all output tokens after the final norm. It is left for a later resolution-versus-pooling split if the author wants one.

## 3. Pairs, preprocessing and quantities (E0 arm CA, unchanged)
- **Vision side, `ctr`:** ℓ2-normalize the rows, subtract the float64 mean of the 16,384 pool rows, ℓ2-normalize again (`e0_common.ctr_features`). D = `s11_common.features_to_D(·, "none")`. The cache is written by the frozen `e0_prep.build_cache` (imported, not edited) into `results/E0_sentence/x3/cache/ctr/<id>/`.
- **Text side:** E0's existing `ctr` caches `results/E0_sentence/cache/ctr/<enc>.n5` (N5 = five-caption average), read only. Their sha256 values are recorded. This is exactly arm CA's text side.
- **Design:** `e0_common.design("clip")`, which gives the 5 nested draws, m ∈ {256, …, 16384} and k ∈ {1, 10, 100}.
- **Cells and summaries:**
  - Cells come from the frozen S1.1 `pair_rows`. Draw means, slopes and elasticities come from the frozen `e0_evaluate.arm_tables`, `elasticities` and `levels`. All of these are imported, not edited.
  - The primary column is `obs` (S1.1's anchors). `obs_allH2` is reported next to it as the column closer to "all points are queries".
- **Cells reported for every X3 pair:**
  - the full draw-mean curves s̄(m, k) for k ∈ {1, 10, 100} over all 7 m, with the draw sd and chance k/(m − 1);
  - L(100, 8192) and L(10, 1024);
  - **L(100, 10000)**, which is k = m/100 at m = 10,000, and L(10, 10000). Both use the plan §6 formula, the same one that gave E0's 0.077: L = exp(ln s̄(8192) + (ln 10000 − ln 8192)/ln 2 · (ln s̄(16384) − ln s̄(8192))), with the ε̄ floor;
  - the slopes b(k) for k ∈ {1, 10, 100};
  - the fixed-m ratios s̄(100, m)/s̄(10, m) at m ∈ {8192, 16384}.

## 4. Reference values (fixed now)
- **E0 arm CA**, rel-0.5 CLS × N5 under `ctr`, column `obs`:

| pair | L(100, 10000) | L(10, 10000) | L(100, 8192) | L(10, 1024) | e_k + e_m on {10, 100} | b(10) |
|---|---|---|---|---|---|---|
| dinov2_b × mpnet | 0.07715427364361675 | 0.028468971940399428 | 0.08457053594964746 | 0.07396597754992014 | 0.015452 | −0.421282 |
| dinov2_b × contriever | 0.07633979414898837 | 0.02840697483165157 | 0.08392362078737889 | 0.07472484821165516 | 0.009590 | −0.427675 |
| dinov2_b × qwen3emb06 | 0.07717300965111769 | 0.02961578325318391 | 0.08465748036030873 | 0.07397754861172179 | 0.015080 | −0.412938 |

- **Schnaus et al.:** about **0.38** for DINOv2-B × MPNet on MS COCO, mutual k-NN with k = ⌊n/100⌋ = 100 at n = 10,000. This is the dotted "full space" line of Fig. 23 (p. 48), read from the rendered figure by the E0 critic. It is approximate, and about 0.38–0.39 also holds for Qwen3-Embedding-8B. At k = 10 on 10,000 COCO points their Fig. 14b gives about 0.07–0.19. That second reading is only descriptive and has no rule.

## 5. Reading rule for the gap (binding for the X3 wording)
- The rule uses L = L(100, 10000), column `obs`, for the pair DINOv2 feature v × MPNet under `ctr` × N5, with L_E0 = 0.07715427364361675.
- **Share of the log-gap closed:** share(v) = ln(L_v / L_E0) / ln(0.38 / L_E0). The denominator is 1.594364.

| share | reading | equivalent L(100, 10000) |
|---|---|---|
| ≥ 1 | "closes the whole gap (reaches or passes Schnaus' reading)" | ≥ 0.380 |
| 0.5 to 1 | "closes most of the gap" | 0.1712 to 0.380 |
| 0.2 to 0.5 | "closes part of the gap" | 0.1061 to 0.1712 |
| 0 to 0.2 | "closes little of the gap" | 0.0772 to 0.1061 |
| < 0 | "widens the gap" | < 0.0772 |

- **Reading-error band (descriptive):** the share is also reported with 0.37 and 0.39 in place of 0.38. The label always uses 0.38.
- **Other encoders (descriptive):** the same share is computed for the Contriever and Qwen3-Embedding-0.6B pairs, each against its own E0 reference, and labelled descriptive. Schnaus et al. did not use Contriever, and their Qwen embedder is 8B. The `obs_allH2` version of the share is also descriptive.
- **No attribution beyond the label.** The added variant changes the final norm, the token set and the resolution at once. The report will not split its share among those three. Among the frozen features, rel-1.0 CLS against the reference changes depth only. rel-0.5 patch mean against the reference changes the token set only. rel-1.0 patch mean changes both. Those contrasts are reported as differences in ln L, with no further label.

## 6. Wrapper validation (STOP rule)
- The wrapper rebuilds the reference feature: E0's `dinov2_b`, the S1.1 axis-1 CLS token, under `ctr`. It writes this into `results/E0_sentence/x3/validation/cache/ctr/dinov2_b_ref/`.
- The rebuild must reproduce every array of each `d<draw>_knn.npz` in E0's `results/E0_sentence/cache/ctr/dinov2_b/` bitwise, and every `median_D`.
- The wrapper's obs rows for the three reference pairs must equal E0's `obs_CA.parquet` rows for those pairs, bitwise in every numeric column.
- If any of these differs: STOP and report. No X3 result is then reported as comparable to E0.

## 7. P1-type scale reading (descriptive, every X3 pair)
- For each pair: e_k, e_m and their sum on K = {10, 100} (`e0_evaluate.elasticities`). The pair is "close to stable at fixed k/m" if |sum| ≤ 0.05, and "informative" if |e_m| ≥ 0.10.
- Per feature, and over all 12 pairs, the report gives:
  - the median sum and the count with |sum| ≤ 0.05;
  - the plan's P1 thresholds applied descriptively: |median| ≤ 0.03 and at least 80% within 0.05.
- Also reported: the sums on {1, 10, 100} and {1, 10}.
- All of this is descriptive. P1's confirmatory label stays the E0 label.

## 8. Also reported (descriptive, fixed now)
- b(10) against L(10, 1024) for each X3 pair, set beside the E0 `ctr` calibration line at k = 10, m = 1024 (0.15928637788145117 + 0.2361611718219916 ln L, from `ctr_calibration.json`). This is orientation only.
- Everything that fails, including any check of §2 or §6, is reported in place.
- **Compute:** GPU hours as the wall time of the GPU process. CPU core-hours are summed over the pinned processes, with at most 6 cores (`taskset -c 0-5`).

## 9. Output paths (all new; nothing existing is edited or deleted)
- `analysis/e0_sentence/x3/`: this file, `x3_scores.py` and `x3_report.py`.
- `extract/e0_sentence/x3/`: `x3_extract.py` and its logs.
- `features/coco_e0/x3/`: X3 feature arrays, the sidecar and `DONE`, all chmod 444.
- `results/E0_sentence/x3/`:
  - `extraction/EXTRACTION_X3.json`
  - `cache/ctr/<id>/`
  - `validation/`
  - `obs/obs_X3.parquet`
  - `X3_results.json`
  - `X3_REPORT.md`
