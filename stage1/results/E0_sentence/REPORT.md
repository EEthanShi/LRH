# E0 run report (role e0-run, 2026-10-10 UTC): sentence encoders as the text side of the S1.1 COCO pools

Plan `analysis/e0_sentence/ANALYSIS_PLAN_E0.md` v1.0 (e859c6ea...) + `ANALYSIS_PLAN_E0_ADDENDUM_1.md` (dcf0ecfc..., implementation fix only, logged before the guard). Verdicts `VERDICTS_E0.json` (5b04e460...), tables `eval/E0_tables.md`, run-role summary `summary.json` (b1d6fcdd...). V-f (e0-indep) and the review (e0-review) have not run yet; this is not the plan's `E0_summary.{json,md}`.

## 1. Confirmatory labels (plan order)

| prediction | label | key numbers |
|---|---|---|
| **P1** fixed k/m stays stable (k in {10, 100}) | **holds** | 15/15 pairs informative (e_m from -0.39 to -0.60); median e_k + e_m = 0.006; 15/15 with \|sum\| <= 0.05 (max 0.0155); holds in each of the 5 draws |
| **P2b** between classes, slope follows level | **uninformative** | median \|g\| = 0.092 < guard 0.113: new levels (median 0.059) are too close to the old ones (0.043) for the level line and the ViT competitor to make different predictions |
| **P2c** within pair, caption averaging moves the slope along the curve | **uninformative** | median ln(L5/L1) = 0.178 < 0.3; no encoder informative (0.146-0.236) |

Binding readings (plan §13):
- **P1 holds.** Agreement at fixed k/m stays stable at k >= 10 for sentence-encoder text sides too (median sum 0.006). The evaluation size acts as a scale parameter for this family as well.
- **P2b x P2c = uninformative x uninformative.** Nothing new about the level-slope regularity, so no app:scale remark (§1.4).
- **B6 tension row:** not triggered.
- **Level wording (§13.3):** "between", i.e. above LM tokens and below vision-vision, in both arms P and C.
- **R1 clip caveat:** none (encoder clip-vs-l2 own 10-NN overlap 0.937-0.964).

## 2. Levels (L = s_N at k = 10, m = 1024; class medians)

| arm | sentence encoders | LM tokens (S1.1 / old under ctr) | vision-vision |
|---|---|---|---|
| P (clip, C1) | 0.059 [0.046, 0.138] | 0.043 | 0.189 |
| C (ctr, C1) | 0.066 [0.050, 0.145] | 0.046 | 0.196 |
| CA (ctr, AVG5) | 0.074 [0.059, 0.169] | - | - |

- **Within every ViT, each sentence encoder is above each LM.** For siglip_b the LMs give 0.074-0.098 and the encoders 0.116-0.138. For clip_b the LMs give 0.033-0.039 and the encoders 0.046-0.050.
- **By encoder (arm P medians):** mpnet 0.061, qwen3emb06 0.059, contriever 0.056.
- **Caption averaging (P3b):** AVG5 - C1 raises ln L by 0.156 (clip) and 0.132 (ctr), i.e. about +14-17%. ALL vs AVG5 changes nothing (98% of images have exactly 5 captions).
- **Preprocessing (P3c):** ctr - clip adds +0.089 to ln L for new pairs (+0.068 for old VL pairs). l2 - clip changes it by -0.006.
- **Schnaus-matched cell (P3h).** This is arm CA (centred, 5-caption average) at m = 10,000 by interpolation.
  - dinov2_b x mpnet gives 0.077 at k = 100 and 0.028 at k = 10. Schnaus et al. report about 0.38 and 0.07-0.19 (the critic's readings of their figures).
  - Encoder, caption average and centering together explain little of this roughly fivefold gap.
  - The remaining protocol differences are the vision features (mid-depth CLS here, pooled output tokens there) and the anchors. The bridge X3 is exploratory and has not been run.
- **Fixed-m ratio** s(100, m)/s(10, m) for dinov2_b x mpnet is 2.6-3.6 (k-elasticity about 0.42-0.55). This matches the direction of Schnaus' k-dependence at fixed m, and it is exactly what P1 implies (e_k = -e_m).

## 3. Slopes (descriptive; no confirmatory weight)
- **Class medians b(10):** seed -0.014, vision -0.214, sentence-encoder VL -0.525, LM-token VL -0.571. The slope order equals the level order across the four classes at k = 1, 10 and 100 (P3f).
- **P2 (descriptive).** Spearman(L, b(10)) over the 30 VL pairs is 0.79 (within-ViT permutation p = 1e-5). Over the 6 text models rho = 0.94 (exact p = 0.008). As the plan anticipated (§7.2, B5), this does not separate "level sets the slope" from "partly set by the level".
- **P2b statistics, computed anyway.** Leave-one-encoder-out lambda is 0.40-0.49 and leave-one-ViT-out 0.39-0.56, so the new slopes sit roughly halfway between the level line and the ViT values.
  - Secondary arm A (AVG5, clip) passes the guard (median |g| 0.122) and gets "holds" (lambda 0.957; per encoder 0.90/0.95/1.02).
  - Arm L gets "inconclusive" (0.41). Every other P2b cell is "uninformative".
  - These are secondary results only.
- **P2c ladder** (medians over 15 pairs, n = 1..5 captions):
  - L: 0.0595, 0.0645, 0.0656, 0.0667, 0.0668.
  - b(10): -0.509, -0.482, -0.457, -0.446, -0.432.
  - Median gamma is 0.435 (per encoder 0.32-0.52), above the line's 0.228. The guard failed, so no label beyond "uninformative".

## 4. Process record
- **Extraction** (STATUS lines from role e0-extract).
  - COCO annotations were verified by size and md5. The https request raised a connection error, so the http fallback the plan allows was used.
  - Caption check V-e passes: all 16,384 images were found, and C1 equals an official caption for 100% of items.
  - **C1 is never the lowest-id caption (0.0%).** This supports the §13.5 correction of v4 l.3659: "first caption" is wrong.
  - Model check V-d passes for all three encoders: pinned files verified, no missing or unexpected keys, determinism 0.0, Qwen card scores reproduced to 2.6e-7 in fp32, every Qwen input ends with 151643, no truncation (max 63 tokens).
- **ADDENDUM 1.** The frozen `e0_evaluate.main` would have stopped on files already written by `e0_calib`; a pre-guard dry run on synthetic obs found this. The wrapper reuses the 4 ctrold tables only after checking that they are content-identical. No statistic changed.
- **First calib attempt.** The first `e0_calib` attempt wrote nothing because `eval/` did not exist. The directory was created and calib was rerun unchanged.
- **STATUS tag.** The first CTR CALIBRATION STATUS line used a tag the guard does not parse. Because STATUS is append-only, an exact-form line with the same hash was appended.
- **Guard.** GUARD CLEAN was logged before the first cross-model score. A read-only `e0_guard.py audit` gives PASS: all 87 new-pair files were born after the GUARD CLEAN line.
- **Compute.**
  - GPU: 0.38 GPU-h process wall (0.368 in-loop), against a plan estimate of 0.4-1.0 (central 0.6). No cap was set.
  - CPU: about 1 core-h, never more than 6 cores (all processes pinned with taskset).
  - Disk: features 2.0 GB; caches are in results/E0_sentence/cache.

## 5. Not done / needs a decision
- **Remaining plan steps:** the V-f independent implementation (e0-indep), the review and audit file (e0-review), then `E0_summary.{json,md}` and `DONE`.
- **Exploratory X1-X3:** not run. X3 is the most informative next step for the level gap to Schnaus et al., but it needs new DINOv2 features (GPU).
- **Q1 and B7:** Q1 (LM re-extraction on AVG5) and B7 (MPNet membership check) were not approved and not run. Without Q1, the AVG arms cannot separate the caption effect from the encoder effect for LM pairs. The C1 arms (P, C) compare encoders cleanly.

## 6. 作者须知（中文）
1. **结果是什么？** 确认性预测 P1 成立。
   - 把文本侧换成 3 个句向量模型后，15 对全部随 m 下降，又随 k 等量上升。
   - k 弹性加 m 弹性的中位数是 0.006，15 对全在 0.05 以内。
   - 所以"固定 k/m 时一致性不变"这条尺度解读推广到了句向量。
2. **P2b 和 P2c 为什么没有结论？** 两条都是"无信息"。
   - 句向量把视觉-语言一致性只从 0.043 提到 0.059，离原来太近。按水平预测的斜率和按 ViT 预测的斜率，差距不到守卫门槛（0.092 < 0.113）。
   - 从 1 条到 5 条描述平均，一致性只提高约 19%（ln 比中位数 0.178），门槛要求 35%（ln 比 0.3）。
   - 按冻结规则，关于"斜率由水平决定"，E0 没有给出新东西，不加 remark。
3. **和 Schnaus 等人差在哪？**
   - 在他们的设置附近（中心化、5 条描述平均、m=10,000、k=100），DINOv2 × MPNet 是 0.077，他们约 0.38。
   - 编码器、描述平均、中心化加起来只解释差距的一小部分。
   - 最可能的剩余差别是视觉特征：我们用中间层 CLS，他们用输出层池化。检验它的是探索性的 X3，还没跑。
4. **论文怎么写？**
   - 按 §13.3 是"介于两者之间"。局限段应写明，我们的视觉-语言水平描述的是"语言模型 token 特征加我们的协议"，并引用 Schnaus 等人和 E0。
   - §13.4 的句子可以填数：句向量 0.059，token 特征 0.043，弹性和中位数 0.006。
   - 另外，"pool holds the first caption" 这句是错的：C1 从来不是编号最小的那条（0%）。
