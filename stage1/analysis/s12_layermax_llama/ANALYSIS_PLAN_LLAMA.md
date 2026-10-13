# S1.2 secondary, second size axis: layer-max replication on LLaMA-7B/13B/30B x 17 ViTs (WIT-1024). Analysis plan, Part A, v1

Role `llama-axis-plan`. Authorisation: the user approved this run on 2026-10-08 (relayed by the lead); it extends the replication authorised in `prereg/LEAD_DECISIONS_v1.1.md` (sha256 eaebd687…). Paths: `analysis/s12_layermax_llama/`, `results/S1_2_layermax_llama/`, `features/wit1024_llama_axis/`, `extract/registry_llama_axis.py`, `extract/llama_axis/`.

This plan extends two frozen plans and changes nothing in them:
- `analysis/s12_layermax/ANALYSIS_PLAN_LAYERMAX.md` v1, sha256 dd1897214d449aaa0e48fb12d5ba43364e015bb16b67508afa221b143431a594 (the "frozen plan"; its sections are cited as FP§n);
- `analysis/s12_layermax/indep/PLAN_INDEP.md`, sha256 085746a823b23bfe19f9a3c1d85b354a6beec6dd71c6cca802c7cf98a0ab8817.

Every definition not restated here is taken verbatim from those plans. Part A (this file) is frozen and hashed in `STATUS.md` before any model download starts. Part B (§9) is a rule: it is instantiated from features only and frozen before any cross-model score on a new pair is computed.

## 0. Disclosure

- **Read before writing:** both frozen plans, both layer-max reports (`results/S1_2_layermax/S1_2_layermax_report.md`, `results/S1_2_layermax/indep/LAYERMAX_INDEP_REPORT.md`), `extract/PROTOCOL.md` (sha256 e013f98b…), `extract/registry.py` (sha256 5e95fe65…), the trend code in `analysis/s12/aggregate.py` and `analysis/s12_layermax/{lm_core,aggregate}.py`, and Proposition 5 and Corollary 2 of `arxiv_v3.tex`.
- **Already known that involves LLaMA-7B** (prh_clip+l2, biased CKA, means over 17 ViTs): layer-max T 0.427, c_α 0.187, s_cal 0.296, uncensored 0.242, single-entry threshold of the selected pair 0.170; mKNN T 0.144, c_α 0.0138. The selected LM depth is 0.83 for CKA. In the 8-LM panel, the PR profile of OpenLLaMA/LLaMA is low in blocks 1 to L/4 (11–36), moderate at the embedding (131–180) and peaks late (202–257 at relative depth 0.88–0.96). Under `l2` the embedding is the highest-null LLaMA-7B layer (0.118).
- **Not known:** any feature, PR value or score of LLaMA-13B or LLaMA-30B.
- **Hub metadata.** Queried on 2026-10-08 with `HfApi.model_info(..., token=False)`; this is metadata only, and no file was downloaded. All three repos report `gated: False` and `private: False`. Revisions and LFS sha256 are pinned in §1.
- This role ran no pilot.

## 1. Models, panel, ranks

| model | revision (pinned) | blocks L | hidden d | stored layers | weights to download (safetensors only) | expected num_params |
|---|---|---|---|---|---|---|
| huggyllama/llama-7b | 4782ad278652c7c71b72204d462d6d01eaaf7549 (already extracted, `features/wit1024/huggyllama__llama-7b.*`) | 32 | 4096 | 33 | none | 6,738,415,616 (sidecar) |
| huggyllama/llama-13b | bf57045473f207bb1de1ed035ace226f4d9f9bba | 40 | 5120 | 41 | 3 shards, 26.03 GB: 48118f78…d065, 6ad4cf0a…0af6, 534f1a58…acb3d | ≈ 13.0 B |
| huggyllama/llama-30b | 2b1edcdb3c7ced7bce6c1aa75c94545777c3118b | 60 | 6656 | 61 | 7 shards, 65.06 GB: 154e2b45…f832, f141f0fa…b77f, 8dbc39c9…b0f6, 5bb79933…9045, acb4f57b…53d0, ffc901be…9ab, 28d1f0c0…350 | ≈ 32.5 B |

- **Files.** The full LFS sha256 of every shard is in the `model_info` output and is written into `extract/registry_llama_axis.py`. The non-weight files to download are `config.json`, `generation_config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer.model` (sha256 9e556afd…d347, the same in all three repos), `tokenizer_config.json`, `special_tokens_map.json`, `LICENSE` and `README.md`. The `pytorch_model*.bin` duplicates are excluded with `allow_patterns`.
- **Exclusions.**
  - LLaMA-65B is skipped: it does not fit one GPU in fp16, and quantisation is not allowed.
  - `open_llama_13b` (ARH rank 8) is not added, because it is not in the authorised scope.
  - Qwen3-14B is never used.
- **Panels.**
  - *LLaMA axis*: {llama-7b, llama-13b, llama-30b} × the 17 ARH `val` ViTs = 51 pairs. LM = A, ViT = B, and permutations act on the ViT rows (FP§1).
  - *Pooled ARH panel*: the 8 S1.2 LMs plus llama-13b and llama-30b, × 17 ViTs = 170 pairs.
- **ARH LLM rank** is the `prh_models.get_models("val")` order (FP§5a): bloomz 1–5, open_llama_3b 6, open_llama_7b 7, llama-7b 9, llama-13b 10, llama-30b 11. Dense ranks 1..10 are a sensitivity check, and log num_params a second one.
- **Family label** for the trend code: `LLaMA` for the three huggyllama models.
- **Layer pairs per pre.** ViT stored layers total 352. LLaMA layer pairs: 33·352 + 41·352 + 61·352 = 47,520.

## 2. Extraction of llama-13b and llama-30b

The protocol is `extract/PROTOCOL.md` §1, §3–§7 for LMs, unchanged except where this section says otherwise.

**Fixed conventions.**
- `AutoTokenizer.from_pretrained(<own snapshot>, use_fast=True)`, with ValueError falling back to slow.
- WIT text `text[0]` unchanged, no prompt, default special tokens (BOS), no truncation.
- Batch 1, no padding, `attn_implementation="sdpa"`.
- TF32 disabled (`fp32_precision="ieee"`).
- Mean over all tokens of every `hidden_states[0..L]`, accumulated in float64 and stored float32. The last state is post-final-norm. `lm_aux` holds n_tokens, nll_mean, bpb and last_prenorm.
- Items come from `features/wit1024/items.parquet`; its sha256 must equal `items_manifest.json` (7716bf4d…).

**Dtype.** Weights and activations are fp32, as for every Stage-1 LM. The checkpoints are stored in 16-bit, about 2 bytes per parameter, and E1 records the safetensors dtype. Upcasting to fp32 is therefore exact.
- llama-13b: a standard fp32 load of about 52 GB on the GPU.
- llama-30b: a full fp32 copy (about 130 GB) does not fit in 96 GB. It is run as **streamed fp32**. The 16-bit weights stay on the GPU, and each decoder layer is upcast to fp32 just before its forward and released after. Embedding, final norm, lm_head and rotary buffers are kept in fp32. Every arithmetic operation then runs in fp32 on the same fp32 weight values as a full fp32 model.
- If E3 fails, llama-30b is **not** extracted in fp16 activations. That path would be a protocol deviation, and the role stops and reports so the lead can decide.

**Outputs and code.**
- Features go to `features/wit1024_llama_axis/<slug>.{npy,json,lm_aux.npz,done}`, a new directory. Nothing is written to `features/wit1024/`.
- Code changes are made on copies in `extract/llama_axis/`, for example `extract_core.py` with the streamed path. Each copy's sha256 and the sha256 of its original go into STATUS.

**Extraction validation (E1–E7, reported whatever the outcome).**
- **E1.** The downloaded blob sha256 equals the LFS sha256 above for every shard. Record the safetensors dtype (F16/BF16) and the snapshot hash.
- **E2.** Re-extract llama-7b with the copied pipeline into `features/_llama_axis_check/`. It must reproduce the stored `huggyllama__llama-7b.npy` (sha256 1a727b78…e0a1). Bitwise equality is expected. Up to 1e-6 maximum relative difference is accepted and reported; above that, the role stops (environment drift).
- **E3.** The streamed-fp32 path equals the standard fp32 path on llama-7b and on llama-13b, over all items and layers. Bitwise equality is expected; ≤ 1e-6 relative is required.
- **E4.** Hidden-state semantics are checked on item 0 as in PROTOCOL §4: hs[0] is the input to layer 0; hs[i] is the output of layer i−1; hs[L] is final_norm(last layer output).
- **E5.** Tokeniser:
  - For each model, `sp_check` against `sentencepiece.encode`.
  - Token ids must be identical across the three repos for all 1024 captions. The llama-7b `tokenizer_config.json` (2284 B, with an added chat template) differs from the 13b/30b one (700 B).
  - If any caption's ids differ, the primary features use each repo's own tokeniser (the ARH/PRH convention), and the count is reported. A sensitivity extraction of 13b and 30b with the llama-7b tokeniser files is then run, and every LLaMA-axis trend is also reported on it. Predictions are evaluated on the primary features only.
- **E6.** Rerun check: 3 items extracted twice, bitwise equal. Array checks: values finite, no zero rows.
- **E7 (descriptive, not a gate).** `num_params` from the loaded model; mean NLL and bpb per model.

**STOP conditions.** Any 401/403, gating prompt, licence or terms acceptance, or token request during download stops the run. Licences are never accepted and tokens are never created.

## 3. Preprocessing, layers, scores, calibration

All of this is unchanged from FP§1–§3 and PLAN_INDEP.

**Preprocessing.**
- Primary `prh_clip+l2`: one clamp threshold per model from `hrhv2.prep.prh_clip_threshold` on the float64 stack of all stored layers (D5), then per-layer L2 in float64.
- Secondary `l2`.

**Layers.**
- Every stored layer is used. LM: hidden_states[0..L]. ViT: every block output, CLS token, no final norm.
- The fixed depths r ∈ {0.5, 1.0} use the S1.2 primary rule: stored index floor(r·(L_stored − 1) + 0.5).
  - llama-7b: 16 and 32.
  - llama-13b: 20 and 40.
  - llama-30b: 30 and 60.
  - At r = 1.0 the LM layer is the post-final-norm state.

**Scores per layer pair.**
- mKNN s_N(10), SPEC tie rule.
- Biased linear CKA (primary).
- Debiased CKA (Song et al.) evaluated on HKH, as `hrhv2.geometric.cka_pair` (exploratory/secondary).

**Calibration.**
- K = 500.
- π_k = `hrhv2.calibrate.permutation(1024, "S1.2-layermax|<lm stem>|<vit stem>", k)`, k = 0..499, with stems such as `huggyllama__llama-13b`. These are the same key format and the same function as the frozen plan. The same π_k is applied to every ViT layer, every metric and both pre.
- The max is recomputed inside every permutation.
- c_α is the 476th order statistic (ascending) of {T} ∪ {T_k}, with α = 0.05.
- s_cal = max((T − c_α)/(1 − c_α), 0).
- p = (1 + #{T_k ≥ T})/501.
- uncensored = T − mean(T_k).
- The null mean is also recorded.

**Protocols per pair, pre and metric.**
- (i) Layer-max over the full L_A × L_B grid.
- (ii) Fixed depth at r = 0.5 and r = 1.0. This is a single entry calibrated against its own permutation null from the same π_k.
- (iii) The FP§7 A2 subsets: grid2, grid3, grid5, grid9 and grid17 (from `lm_core.grid_layers`), LMall×ViTlast and LMlast×ViTall. Added now as pre-specified, descriptive subsets: noLMemb, LMdeep (relative depth ≥ 0.5) and LMemb_only.
- (iv) **Calibrate-then-max (new, secondary).** M = max over (u,v) of the entry-wise s_cal,uv, where each entry is calibrated against its own c_α,uv from the same 500 π. This is the first remedy named after Corollary 2. Its uncensored analogue is max over (u,v) of (S_uv − mean_k S_uv^(k)).

**Implementations.**
- Copy of the early role's code in `analysis/s12_layermax_llama/early/` (implementation **E**).
- Copy of the independent role's code in `analysis/s12_layermax_llama/indep/` (implementation **I**). I stores the full observed and null L_A × L_B matrices.
- Each copy is changed only in its model list, paths and family label. The sha256 of each copy and of its original go into STATUS.
- llama-7b is recomputed by both implementations, as a validation (V-L4).
- For the 7 non-LLaMA LMs, the pooled panel reads the frozen `results/S1_2_layermax/` outputs read-only. Their definitions and permutation keys are identical.

## 4. Validation (before any main-run output is inspected; reported whatever the outcome)

- **V-L1 (clip).** Thresholds recomputed by E and by I are bitwise equal. llama-7b equals the S1.2 cache (`results/S1_2/cache/clip_thresholds.json`) bitwise. The new values go to `results/S1_2_layermax_llama/cache/clip_thresholds.json`.
- **V-L2 (kNN tables).** Every stored layer of llama-13b and llama-30b, both pre, is checked against `hrhv2.structural.knn_table`: tables must be identical. If one differs, the hrhv2 table is used for that layer and the case is reported (FP§4 V1).
  - Exact k = 10 boundary ties and SPEC near-tie fractions are reported per layer.
  - The ViT side must have 0 boundary ties, as before. ViT tables are recomputed and must equal the frozen caches.
- **V-L3 (hrhv2 identity).** 12 layer pairs, fixed now: (llama-30b, DINOv2-g), (llama-13b, CLIP-H ft-in12k), (llama-30b, ViT-Ti in21k) and (llama-13b, MAE-H), each at (first, first), (middle, middle) and (last, last) stored layers. Each is checked at the identity and π_0, π_1, π_499, for both pre:
  - CKA (biased and debiased) vs `cka_pair` on `B.permuted(π)`: ≤ 1e-6 absolute (expected ~1e-13);
  - mKNN: exactly equal to hrhv2 recomputed from scratch.
- **V-L4 (llama-7b reproduction).**
  - (a) The fixed-depth entries for 17 pairs × 2 depths × 2 pre reproduce the S1.2 stored s_obs: mKNN exactly, CKA ≤ 1e-10.
  - (b) The full llama-7b layer-max outputs of E and of I reproduce `results/S1_2_layermax/pairs/` and `results/S1_2_layermax/indep/`. The fields are T, c_α, s_cal, p, uncensored, null mean, argmax and every subset. Tolerance is ≤ 1e-12 relative; p and the mKNN null counts must be identical.
- **V-L5 (E vs I on the 34 new pairs).** Both pre, all metrics and subsets:
  - mKNN null counts identical;
  - argmax identical;
  - s_obs, c_α, s_cal, uncensored and null mean within |a − b| ≤ 1e-12·max(|a|, |b|) + 1e-15;
  - p exactly equal.
- **V-L6 (vendored ARH, our π injected through `perm_fn`).** Tolerance 1e-6, because ARH stores float32.
  - Scalar `calibrate()` on (llama-30b last, ViT-Ti/16-in21k last) for mKNN (PRH `mutual_knn`) and for biased CKA (ARH `cka_biased`).
  - `calibrate_layers(agg="max")` for biased CKA on the full llama-13b × ViT-Ti/16-in21k grid (41 × 12).
  - `calibrate_layers(agg="max")` for mKNN on llama-30b LM layers `grid_layers(61, 9)` × all 12 ViT-Ti layers, compared with I's exact sub-grid maximum.
  - Tie-rule differences are reported, not hidden.
- **V-L7 (Proposition 5, numerically, on the new pairs).** Compare the entry-wise permutation null means (500 π) with √(PR_u·PR_v)/(m−1) from §9 B1, for biased CKA:
  - report the median relative error (expected ≤ 3%, as 0.5% on the 8-LM panel) and the fraction of entries with |z| > 4 against the MC standard error (expected ≤ 1%);
  - debiased-CKA null means are expected to be within MC error of 0 (§9);
  - mKNN null means are expected within MC error of 10/1023.
  - The identities are exact, so a failure means an implementation error. It is investigated before any reading.

**Order.** If V-L1–V-L7 reveal an error, it is fixed and the validation is rerun; both outputs are kept. The main run may start before V-L6 finishes, which is slow. No main-run output except completion status is opened until V-L1–V-L7 pass. Any deviation from that order is logged.

## 5. Trend statistics

These apply to both pre, all three metrics, and the scores raw T, s_cal and uncensored, for protocols (i), (ii), (iv) and the subsets of (iii).

- **LLaMA-axis trend Tr_L.** This is the BLOOMZ Tr definition (`analysis/s12/aggregate.py::lm_trend`) with family `LLaMA`.
  - Tr_L = mean over the 17 ViTs of ρ_v, where ρ_v is the Spearman correlation between the num_params ranks (7B < 13B < 30B) and the three scores of ViT v. Average ranks are used for ties. A ViT with three equal scores is excluded and counted.
  - Exact p over the 3! = 6 relabellings of the sizes, with the same relabelling for every ViT: p_one = #{Tr(σ) ≥ Tr_obs − 1e-12}/6, and p_two the same on |·|.
  - **Smallest attainable p:** one-sided 1/6 ≈ 0.167, two-sided 1/3. No LLaMA-axis trend can be significant at 0.05.
  - The 17 ViT rows share the same three LM feature sets, so a binomial count over ViTs would overstate the evidence. It is not used.
  - ViT-bootstrap 95% CI: 2000 draws, rng key `S1.2|bootvit|layermax-llama|<stat>`. It reflects ViT sampling only. Uncertainty from having only 3 LMs is not quantified.
  - With 3 sizes, ρ_v ∈ {−1, −0.5, 0.5, 1} without ties, and ρ_v is a monotone function of Kendall's τ, i.e. of the number of concordant size pairs. One adjacent swap in one ViT moves Tr_L by 0.5/17 ≈ 0.029.
  - **Materiality for changes in Tr_L is fixed at 0.10.** That is at least 4 ViTs changing by one adjacent swap in the same direction.
- **ΔTr_L(score)** = Tr_L(score) − Tr_L(raw T), for the same protocol. Its exact p uses the same 6 relabellings (`lm_trend_diff`); the minimum is 1/6.
- **Joint within-family trend (secondary, descriptive).** Tr_BL = (Tr_BLOOMZ + Tr_L)/2, with exact p over the 120 × 6 = 720 joint relabellings. The smallest one-sided p is 1/720. The BLOOMZ half is already known, so Tr_BL carries no prediction.
- **ARH-style pooled r (FP§5a) on the 10-LM panel.**
  - Pearson r across the 170 pairs between the score and the ARH LLM rank, for raw T ("before"), s_cal ("after") and uncensored.
  - Sensitivity: dense ranks 1..10, and log num_params.
  - ViT-bootstrap CI: 2000 draws, key `S1.2-layermax-llama|boot|arh_r|<stat>`.
  - Exact one-sided p over all 10! = 3,628,800 LM relabellings (same for every ViT; r is linear in the LM row sums). The smallest p is 1/10!.
  - Also reported on the 8-LM panel, so the two can be compared.
- **Vision axis (descriptive).** S1.2's statistic restricted to the (LLaMA LM, vision family) cells, with exact within-family relabellings.
- **OpenLLaMA 7B > 3B fraction:** kept for the pooled panel summary.

All p-values are unadjusted, and there is no Holm family. This is a replication and a set of directional predictions. Nothing here is a confirmatory test.

## 6. Attribution diagnostics (pre-specified, descriptive, LLaMA axis)

- **FP§7 A1:** selection. Argmax depths, ties, CKA/mKNN agreement and the location of the null maxima.
- **FP§7 A2:** null inflation against the number of layer pairs and the nested grids. Here L_A grows with size (33, 41, 61), so the layer count and size are confounded on this axis. grid17 equalises the LM side to 17 layers.
- **FP§7 A3:** counterfactual decomposition (1)–(6), plus the trends of c_α and of the inflation c_α − c_α^entry(a*, b*).
- **Added (the 8-LM post-hoc diagnostics, now pre-specified):**
  - level vs multiplicity: E0[max] − max_uv E0[S_uv];
  - per-LM-layer null level (max over ViT layers);
  - the LM layer holding the highest null;
  - the leave-layers-out maxima of §3 (iii);
  - T − c_α^entry(a*, b*) trends.
- **Plug-in check (secondary).** Compute Tr_L of T − B̄_pred, with B̄_pred from §9 B2. It is expected to agree with Tr_L(uncensored) to within 0.10, because the two differ only by the multiplicity term.

## 7. Replication readings (FP§6 rules, applied unchanged)

- **ARH-style, 10-LM panel** (primary pre, biased CKA, layer-max). Δ_X = r_before(X) − r_after(X).
  - "ARH pattern reappears" if Δ_CKA − Δ_mKNN ≥ 0.20 and r_after(CKA) ≤ r_after(mKNN) − 0.20.
  - "Absent" if Δ_CKA − Δ_mKNN ≤ 0.10 or r_after(CKA) ≥ r_after(mKNN) − 0.10.
  - Otherwise "partial".
  - Reference: ARH Table 2 on 12 LMs / 204 pairs gives linear CKA 0.86 → 0.45, unbiased CKA 0.65 → −0.01 and mKNN 0.85 → 0.84. Our panel lacks open_llama_13b and llama-65b.
- **LLaMA axis.** This reading is applied only if Tr_L(raw T, CKA, layer-max) ≥ 0.60; otherwise the outcome is "no raw trend to lose".
  - "Layer-max loses the CKA trend on the LLaMA axis" if Tr_L,unc(CKA) ≤ 0.40 while Tr_L,unc(mKNN) ≥ 0.60.
  - "No loss" if Tr_L,unc(CKA) ≥ 0.60.
  - Otherwise "intermediate".
- Debiased CKA and `l2` get the same readings, labelled exploratory and secondary respectively.

## 8. Theory used by Part B

**Proposition 5** gives the random-pairing baselines:
- mKNN: E_π s_N = k/(m−1) = 10/1023 for every layer pair. This needs no ties on the permuted side, which V-L2 checks.
- Biased CKA: E_π CKA(K_u, K_v^π) = √(PR(K̄_u)·PR(K̄_v))/(m−1), with PR(K̄) = (tr K̄)²/‖K̄‖_F² the participation ratio of the centred Gram.

**Debiased CKA baseline.** This is derived here; it is not stated in the manuscript.
- Let K̃ = K − diag(K) and S_K = 1ᵀK̃1. Song's estimator is

  HSIC_u = [tr(K̃L̃) + S_K·S_L/((m−1)(m−2)) − (2/(m−2))·1ᵀK̃L̃1] / (m(m−3)).

- Under a uniform π: E tr(K̃L̃^π) = S_K·S_L/(m(m−1)) and E 1ᵀK̃L̃^π1 = S_K·S_L/m.
- So E_π HSIC_u ∝ S_K·S_L·[1/(m(m−1)) + 1/((m−1)(m−2)) − 2/(m(m−2))] = 0.
- The denominators of debiased CKA are permutation invariant, so E_π dCKA = 0 for every layer pair.
- V-L7 checks this numerically.

**Corollary 2** (aggregation before calibration):
- Let s̄_j be the layer-max score of model pair j, s̄⁰_j = max_uv of its baselines, and c_j its threshold, with |c_j − s̄⁰_j| ≤ ζ_j. Then s_cal,j is bracketed by [s̄_j − s̄⁰_j ∓ ζ_j]_+/(1 − s̄⁰_j ∓ ζ_j).
- With equal max baselines, the order of two pairs whose raw scores differ by more than ζ_1 + ζ_2 is preserved.
- A reversal occurs when the max baseline grows faster than s̄ (the corollary's last condition).
- Since c ↦ (T − c)/(1 − c) is non-increasing for T ≤ 1, for two LLaMA sizes j < j' and one ViT:
  - if c_j < c_j', calibration can only turn a concordant pair (T_j < T_j') into a discordant one, never the reverse;
  - if c_j > c_j', only the reverse can happen.
  - So when the thresholds are monotone in size, every ρ_v moves weakly in the direction opposite to the threshold trend.
- **What is actually tested** is therefore not the corollary, which is a proven statement. It is:
  - (a) whether the threshold order is predicted from features alone by the PR-based max baseline. This is the premise "c_α lies within a small ζ of the largest baseline", stated after Corollary 2.
  - (b) whether that predicted direction shows up in the calibrated trend.

## 9. Part B: the rule (features only; instantiated and frozen before any cross-model score on a new pair)

Part B is computed after extraction (§2) and the single-model checks V-L1 and V-L2. It is computed before V-L3–V-L7 and before the main run. No observed or permuted cross-model score of llama-13b or llama-30b, at any depth, may exist when it is frozen.
- `results/S1_2_layermax_llama/pairs/` must not exist at freeze time; this is checked and logged.
- The Part B script `analysis/s12_layermax_llama/partb_pr.py` opens only per-model feature stacks and the clip thresholds.

**Computations, for both pre:**
- **B1. PRs.** For every stored layer u of the 10 LMs and every stored layer w of the 17 ViTs: PR_u = (tr K̄)²/‖K̄‖_F², where K̄ = HKH and K = X Xᵀ of the 1024 preprocessed rows, in float64. These are single-model quantities.
- **B2. Predicted baselines.** For every pair (i, v):
  - B_uw = √(PR_u·PR_w)/(m−1), with m = 1024;
  - the predicted layer-max baseline B̄(i,v) = max_uw B_uw = √(PR*_i·PR*_v)/(m−1), with PR*_i = max_u PR_u, and its argmax (u*, w*);
  - the fixed-depth baselines B_fd(i,v; r) at the §3 layers, r ∈ {0.5, 1.0}.
- **B3. LLaMA-axis predicted trends.**
  - Tr_pred = Tr_L(B̄). Because the ViT factor is common to the three LMs, ρ_v(B̄) is the same for every ViT and equals Spearman(size, PR*_LM), which takes a value in {±1, ±0.5}.
  - Also Tr_L(B_fd(r)) for r ∈ {0.5, 1.0}.
  - The pairwise gaps gap_v(j, j') = |B̄(j,v) − B̄(j',v)|.
- **B4. Materiality** of the max-baseline trend. The trend is "material" if |Tr_pred| = 1 (B̄ monotone in size) and, for each of the 3 size pairs, the median over ViTs of gap_v(j, j') is ≥ δ_b = 0.005.
  - Why 0.005: on the 8-LM panel, c_α − E0[max] ≈ 0.002 and E0[max] − max E0 ≈ 6e-5 for biased CKA. A gap of 0.005 is therefore 2.5× the largest expected offset between c_α and B̄.
  - The same rule applies to B_fd(r).
- **B5. Descriptive extras.**
  - Pearson(ARH rank, B̄) across the 170 pooled pairs.
  - Location of the threshold-setting LM layer u*/(L_A − 1) per LLaMA model.
  - A retrodiction on BLOOMZ: Tr_BLOOMZ(B̄) against the known c_α trend 0.90. This is labelled "retrodiction, outcome known, not a test".
- **B6. Freeze.** Write `analysis/s12_layermax_llama/PARTB_PREDICTIONS.md` (the instantiated predictions below, with signs and materiality flags filled in) and `results/S1_2_layermax_llama/partB/partB_features.json` (all B1–B5 numbers). Append the sha256 of both files and of `partb_pr.py` to STATUS before the next step.

**Prediction templates.** The primary instance is prh_clip+l2, layer-max. The scores are s_cal (primary) and uncensored (secondary). The same rules are applied to `l2` with its own B quantities and reported as secondary.

- **P-L1 (mKNN; B̄ = 10/1023 for every pair).**
  - Prediction: |ΔTr_L(s_cal)| < 0.10 and |ΔTr_L(unc)| < 0.10. In words, calibration neither reverses nor materially changes the raw LLaMA mKNN trend.
  - The only size-dependent term left is multiplicity, since c_α grows with L_A·L_B (33 → 41 → 61 LM layers), which pushes only toward lower Tr. The bet is that it is negligible; on BLOOMZ c_α was 0.0138–0.0140.
  - Deductive check, reported: no size pair whose raw gap exceeds ζ_j + ζ_j' (ζ = |c_α − 10/1023|) changes order. A violation means an implementation error.
- **P-L2 (biased CKA, layer-max).** Let s = sign(Tr_pred).
  - **If material**, the prediction is sign(ΔTr_L(s_cal)) = −s:
    - "confirmed" if −s·ΔTr ≥ 0.10;
    - "consistent-null" if |ΔTr| < 0.10. Corollary 2 allows this when raw gaps exceed baseline gaps, so it does not count for or against;
    - "contradicted" if s·ΔTr ≥ 0.10;
    - "not testable" if Tr_L(raw) = −s already, so no move in the predicted direction is possible.
  - **P-L2a (premise).** (i) The median over the 51 pairs of |c_α − B̄| is ≤ 0.01. (ii) Over the (ViT, size-pair) comparisons with gap ≥ δ_b, the order of c_α agrees with the order of B̄ in ≥ 2/3 of cases.
  - Deductive check, reported: every ViT whose c_α order is monotone has ρ_v(s_cal) − ρ_v(raw) of sign −(sign of the c_α trend) or zero.
  - **If not material**: no directional prediction on Tr_L. The pairwise flips for size pairs with gap ≥ δ_b are reported descriptively, in the predicted direction and against it.
- **P-L3 (biased CKA, fixed depth r ∈ {0.5, 1.0}, each entry calibrated against its own null).** Apply only where |Tr_L(raw, r)| ≥ 0.10, with s_r = sign(Tr_L(raw, r)).
  - "Keeps sign" if s_r·Tr_L(s_cal, r) ≥ 0.10.
  - "Lost" if |Tr_L(s_cal, r)| < 0.10.
  - "Reversed" if s_r·Tr_L(s_cal, r) ≤ −0.10.
  - The prediction is "keeps sign". Part B records Tr_L(B_fd(r)) and its materiality, so that any loss can be attributed to the entry's own baseline trend.
- **P-L4 (debiased CKA; baseline 0 for every pair).**
  - ΔTr_L(s_cal) ≥ −0.10 and ΔTr_L(unc) ≥ −0.10 at layer-max (no calibration-induced loss).
  - On the 10-LM pooled panel, r_after ≥ r_before − 0.05.
- **P-L5 (secondary; calibrate-then-max, biased CKA).** The same three-way rule as P-L3, with s = sign(Tr_L(raw T, layer-max)), applied to Tr_L(M) when |Tr_L(raw T, layer-max)| ≥ 0.10. The prediction is "keeps sign".

## 10. What would count against Corollary 2, and what would not

Primary instance: prh_clip+l2, biased CKA, layer-max, s_cal.

**Against the corollary's use as a predictor of calibration effects from features (Prop. 5 + Cor. 2):**
1. **Premise failure (P-L2a fails).** The layer-max threshold does not track the PR-predicted max baseline: median |c_α − B̄| > 0.01, or c_α order agreement < 2/3 on material comparisons. The approximation stated after Corollary 2 ("the calibration threshold of a layer-aggregated score lies within ζ_j of the largest baseline, regardless of the layer pair attaining the observed maximum", with ζ small) would then be wrong for these models.
2. **P-L2 "contradicted"** in the material case: calibration moves the LLaMA trend in the same direction as the max-baseline trend by ≥ 0.10. Given §8, this can happen only if the actual threshold order departs from the B̄ order, so it is reported jointly with item 1.

**Against the claims stated after Corollary 2, not the corollary itself:**
3. **P-L1 or P-L4 fails.** A calibration-induced |ΔTr| ≥ 0.10 under equal baselines would show that terms the corollary leaves in ζ (multiplicity growing with layer count) move calibrated trends in practice. It would contradict the "mKNN and debiased CKA are immune" reading. It would not contradict the corollary, because those flips involve raw gaps below ζ_1 + ζ_2.
4. **P-L3 or P-L5 "reversed".** It would contradict the remedy claim that fixed layers, or calibration before the max, avoid the interaction. The corollary itself, applied with agg = s_{u0v0}, allows a reversal whenever the entry's own baseline grows faster than its score.

**Not counted:**
- "consistent-null" and "not testable" outcomes;
- failures confined to `l2` (reported as secondary);
- debiased-CKA readings beyond P-L4;
- any p-value. With 3 sizes no p < 1/6 is possible, so every LLaMA-axis outcome is a directional, descriptive result on one model family with ViT-only uncertainty.

**Implementation errors, never evidence:** a violation of a deductive check (P-L1, P-L2), or a failure of V-L7. These are investigated and fixed first.

## 11. Order of operations, compute and reporting

**Steps.** Each one gets a STATUS line `<UTC ISO>Z [llama-axis-<role>] …` with file hashes.
- **S0.** Part A frozen.
- **S1.** Write `extract/registry_llama_axis.py` (revisions, shard sha256, family, ranks) and the code copies.
- **S2.** Download safetensors and the small files only into the existing HF cache; STOP conditions per §2.
- **S3.** Extraction and E1–E7.
- **S4.** Single-model precompute and V-L1–V-L2.
- **S5.** Part B frozen.
- **S6.** V-L3–V-L7.
- **S7.** Main runs E and I (K = 500, 51 pairs × 2 pre).
- **S8.** Aggregate, readings, prediction outcomes and report. `results/S1_2_layermax_llama/DONE` is written last.

**Compute.**
- At most 6 CPU threads in total across all llama-axis processes at any time (OMP, MKL and OPENBLAS thread counts and `torch.set_num_threads`).
- Use the GPU freely. Extraction of llama-30b needs about 65 GB of 16-bit weights plus one fp32 layer. Analysis processes use a per-process GPU memory fraction ≤ 0.40.
- Never kill or renice other jobs.
- Disk: about 91 GB of weights plus about 3 GB of features and results, against 214 GB free.

**Frozen files.** Never edited, moved or deleted: SPEC, prereg/*, both frozen plans, `registry.py`, existing features and existing results. Everything new goes under the paths listed at the top.

**Reporting.**
- No pair, layer or permutation is dropped. Failures are reported plainly.
- Every number in the report comes from `results/S1_2_layermax_llama/layermax_llama_summary.json`, which aggregate writes.
- The report states, for each of P-L1–P-L5 and for §10, the outcome category reached, including when it goes against the paper.
