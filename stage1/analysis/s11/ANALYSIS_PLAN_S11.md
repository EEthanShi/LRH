# S1.1 analysis plan: operationalization of PREREG v1.0 §S1.1

**Status: DRAFT v0.9, not frozen.** Role s11, 2026-10-08.

This plan was written before any COCO feature array was read; only the item table (ids and texts) and the sidecar JSONs were opened. It does not change `prereg/PREREG_stage1.md` (sha256 `bee0583c…`), `SPEC.md` or `hrhv2/PREDICTOR_CHOICE.md`. It pins the PREREG wording that the analysis code would otherwise have to choose silently.

**Freezing.** The lead freezes the plan by recording its sha256 in `analysis/s11/LEAD_AUTHORIZATION.json` (format in `s11_run.py`) and in a STATUS line. Until that file exists, `s11_run.py` refuses to open any COCO feature file. Any change after COCO features are touched goes into a versioned addendum, never into this file.

## 0. Preconditions (lead decisions from `results/verification_report.md` §6)

| id | decision needed before S1.1 | default assumed by this plan |
|---|---|---|
| D1, D2 | G0 failed on P-Bern (cross-fitted isotonic) and P-Gauss at k = 1. SPEC §7 says G0 must pass before real-data analysis, and PREREG says an estimator that fails its plant is not used. | none; the lead must record both |
| D4 | (a) run S1.1 with the frozen P1 (`P1-med-B-r8`) and accept the likely "descriptive only" outcome; or (b) issue PREDICTOR_CHOICE/PREREG v1.1 with the S_sel selection SPEC §6 requires. | none. The code takes the variant as a parameter and checks it against the authorization. |
| D2 / k = 1 | if D2 drops k = 1 from the P1 tests, criteria (1)–(3) run on k ∈ {10, 100} | k ∈ {1, 10, 100} (PREREG) |
| D3 | exact ties at the k-boundary (duplicate captions) | keep, using the SPEC tie rule; report a tie-free sensitivity (O11) |
| D5 | clamp scope for `prh_clip+l2` on nested pools; duplicates | one threshold fixed on the whole 16384 pool (O2); duplicates kept (O11) |

## O1. Pairs, classes, models

- **Models.** The models are the 17 COCO extractions in `features/coco_pool/`. Checkpoints, revisions and file hashes are in the sidecars.
  - Vision: DINOv2-B, OpenCLIP B/16 laion2b, MAE-B, AugReg B/16 in21k, SigLIP-B/16 (`.webli`).
  - Text: Pythia-410m seeds 1–9, BLOOMZ-1b7, OpenLLaMA-3B.
- **Primary set (29 pairs).**
  - seed–seed: Pythia-410m seeds (1,2), (3,4), (5,6), (7,8);
  - cross-vision: all 10 ViT pairs;
  - cross-modal: 5 ViTs × {BLOOMZ-1b7, OpenLLaMA-3B, Pythia-410m-seed1} = 15.
- **Secondary set (61 pairs).** As above, with all 36 seed pairs.
- **Direction labels.** "A" is the lower seed, the earlier ViT in the order listed above, or the ViT in a cross-modal pair. The labels matter only for the asymmetric predictors (O6).

## O2. Features and preprocessing

- **Depth.** Relative depth 0.5. The residual index is ℓ(r) = floor(r·L + 0.5), as in extract/PROTOCOL.md §5. ViT arrays store block ℓ−1; LM arrays store hidden_states[ℓ]. The axis position is looked up in the sidecar's `layer_axis_*_indices`, which gives axis 1 of the 4 stored depths for every model; a test checks this.
  - The generic rule round(rel·(n_stored − 1)) is **not** used. On the 4-depth COCO arrays it gives axis 2, which is relative depth 0.75.
  - The last block (rel 1.0) is not part of the PREREG S1.1. If it is run, it is labelled exploratory.
- **Primary preprocessing: `prh_clip+l2`.** One clamp threshold per model, computed once on all 16384 pool rows. Every nested prefix and every draw therefore sees bit-identical D entries; this avoids verifier finding A7.
  - Default scope `pool-layer`: the threshold uses the selected layer's coordinates.
  - Alternative `pool-stored`: all 4 stored depths. This is still not ARH's all-layer threshold, which needs every layer.
  - The threshold uses no B data and no labels, but it does use rows outside smaller prefixes. This is recorded.
- **Secondary preprocessing: `l2`.**
- **Arithmetic.** Distances are float64 via `hrhv2.prep` (SPEC §0).

## O3. Nested draws, halves, cells

- **Draws.**
  - Draw 0 is the stored pool order, i.e. the extraction's own nested prefixes, ascending sha256("hrh-v2-coco-pool-v1:" + image_id).
  - Draws 1–4 use `hrhv2.util.hash_order(image_ids, "hrh-v2-s11-draw<d>")`.
  - All draws permute **within** the extracted 16384 pool, as extract/PROTOCOL.md §2 requires. Consequences: at m = 16384 the five draws contain the same items and differ only in the m = 1024 fit set; at m = 8192 two draws share about half their items.
- **Halves.** `hrhv2.util.hash_half(str(image_id), "hrhv2-halves")`, decided per item. It is therefore identical in every draw and every prefix.
- **Fit.** P1 coupling, N1, N2 and the P2 dispersion are fitted on H1 × H1 of each draw's m = 1024 prefix. A test checks that no H2 row and no other m is read.
- **Prediction.** At each m ∈ {256, …, 16384}, every H2 item of the prefix is an anchor. Neighbours are searched among all items of the prefix.
- **Equivalence with G0.** The code reproduces `hrhv2.predict.crossfit_predict`, the protocol frozen in G0, bit for bit; tested.
- **Cells.** A cell is (pair, m, k) with m ∈ {256, 512, 2048, 4096, 8192, 16384} (the fit m = 1024 is excluded, per "all other m") and k ∈ {1, 10, 100}. Observed s_N and every prediction are averaged over the 5 draws; draw-level rows are saved.

## O4. Fixed-k slope and steep units

- **Slope.** For each (pair, k), the OLS slope of ln s on ln m over the 6 evaluation m, using draw-mean values. This is the log–log slope, an elasticity, so it does not depend on the log base.
- **Steep units.** A (pair, k) unit is steep if |observed slope| ≥ 0.1. "Pairs" in criteria (2) and (3) means steep (pair, k) units.
- **Alternative reading** (a pair is steep at k = 10 only): reported as a sensitivity if the lead asks; it is not the default.

## O5. Log floor

ε(m, k) = 0.5 / (n_draws · n_H2(m) · k), half the resolution of a draw-mean s_N. It applies to observed values and predictions inside slopes and MSLE, i.e. ln max(x, ε). Criterion (1) uses no floor (O7).

## O6. Directions

- P1, P2, N1 and N2 are computed A→B (coupling of D_B given D_A; predicts from D_A rows) and B→A.
- **Primary prediction:** the arithmetic mean of the two directions, since s_N is symmetric.
- **Sensitivities:** A→B only and B→A only.
- N0 has no direction.

## O7. Criterion (1)

- **Cells.** Primary-set cells whose observed draw-mean s_N > 2·k/(m − 1).
- **Statistic.** |ln(P1/obs)|, with P1 ≤ 0 counted as +∞.
- **Pass** if the median ≤ ln 1.15 = 0.1398.

## O8. Nulls and criterion (2)

- **N0.** hrhv2's definition: s_N of the H1 anchors at the fit m, using their full rows. It reads H1 × H2 entries of B (verifier finding A2). This leak can only help the null, so it is conservative for P1.
- **N1.** hrhv2's definition: σ_A = RMS(global affine residual on H1 × H1) / slope. When the slope is ≤ 0, hrhv2 sets σ_A = ∞ and N1 is undefined.
  - **Default** (lead sign-off needed; D4(b) names this option): the σ → ∞ limit, which is chance level k/(m − 1).
  - **Sensitivity:** undefined N1 is dropped from the min.
- **Criterion (2).** MSLE(X) = mean over cells of steep units of (ln max(X, ε) − ln max(obs, ε))². Pass if MSLE(P1) ≤ 0.5 · min(MSLE(N0), MSLE(N1)).
- **N2 and P2** are reported. PREREG names P2 as a second predictor, but the confirm criteria are stated for P1. P2 and N2 are put through the same criteria as exploratory rows.

## O9. Criterion (3) and the outcome rule

- **Criterion (3).** The sign of P1's fixed-k slope, computed as in O4 on P1's draw-mean predictions, equals the sign of the observed slope in ≥ 80% of steep units.
- **Outcome**, in PREREG wording:
  - If (1) or (2) fails: "the crossing law is reported as descriptive only, and the paper's lead result shifts to the lattice/link theorems".
  - If no unit is steep: record "only smooth class observed". (2) and (3) are then vacuous, and the outcome follows (1).
  - If (1) and (2) pass and (3) fails: not confirmed, and the descriptive-only trigger is not met. The lead decides.
  - If (1), (2) and (3) all pass: G1 passes and the crossing law stays the lead mechanism.

## O10. Secondary analyses (PREREG "Secondary")

- **Slope ordering.** Per k, the median observed fixed-k slope per class on the primary set (and with all 36 seed pairs). Check seed–seed ≥ cross-vision ≥ cross-modal. "Cross-modal negative" comes from final_plan, not the PREREG, and is labelled as such.
- **k-elasticity ≈ −m-elasticity.** Per pair, fit ln s = a + e_k ln k + e_m ln m by OLS over all 21 (m, k) cells, including m = 1024. Report e_k, e_m, e_k + e_m, class medians, and corr(e_k, −e_m). This is descriptive; no threshold was preregistered.

## O11. Ties and duplicates

- The SPEC tie rule applies: on exact equality the lower position in the draw's order wins.
- Each cell records the number of H2 anchors with an exact tie between the k-th and (k+1)-th candidate, for each model.
- **Sensitivity:** observed s_N on the anchors with no exact boundary tie. Predictions are unchanged in this sensitivity.
- Duplicate captions (49 at m = 16384) are kept unless the lead decides otherwise (D5).

## O12. Computation and provenance

- **Jobs.** One job is (pair, pre) and covers 5 draws and 2 directions. Defaults: 3 workers × 4 BLAS threads, at most 12 cores, about 11 GB RSS per worker at m = 16384.
- **Outputs.**
  - Per job: `results/S1_1/raw/<pre>/<pair>.{parquet,json,done}`. The JSON records code hashes, the predict.py hash, the authorization record, clamp thresholds, layer names, timings and per-draw fit diagnostics.
  - Evaluation: `results/S1_1/S1_1_summary_<pre>.{md,json}` plus cell, slope and per-pair parquet files.
- **Runtime.** See `results/S1_1/S1_1_status.json` (dry-run timings). On smooth, column-correlated couplings, P1's μ balance touches most columns, which makes it O(anchors × m) per iteration. The full design may then need tens of CPU-hours. If the lead wants a cheaper run, the only change this plan would accept is a hash-selected subset of H2 anchors at m ≥ 4096, used for **both** the observed values and the predictions. That needs an addendum before any COCO run.
