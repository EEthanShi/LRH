# P1 crossing-law predictor: v1.1 re-selection (PREREG v1.1, LEAD_DECISIONS_v1.1 D4)

Owner: s11-select. Authority: `prereg/LEAD_DECISIONS_v1.1.md` (sha256 `eaebd68787a7111f87bdac87a58ff565c0364bae64416dc9203eff913d2f12c3`), decisions D2 and D4 (steps 1–7). This file versions, and does not edit, `hrhv2/PREDICTOR_CHOICE.md` (v1.0, sha256 `75e1ba3d…`) and `hrhv2/predict.py` (v1.0, sha256 `a17964b4…`); both stay frozen and valid for every analysis already run under v1.0. The implementation is the new module `hrhv2/predict_v11.py`, which imports `mu_balance`, `exact_loss`, `fit_coupling`, `_column_laws` and `fit_n1` from the frozen `predict.py`.

No COCO feature file was opened for anything in this file.

## Part A: written before the formal plant and S_sel runs

### A1. Protocol (D4.2: "the same anchor at a different m")

- **Nested pools** are prefixes of one item order. Fit size m_f = 1024 (plants; COCO). On S_sel only 1023 usable rows exist, so there m_f = 512 and the predicted sizes are m ∈ {256, 1023} (the task's instruction; m = 512 is in-sample).
- **Halves** H1/H2 by `hrhv2.util.hash_half` (salt `hrhv2-halves`), per item, hence identical in every prefix.
- **Predicted anchors** are the H2 items of the m_f prefix (≈ 512 at m_f = 1024); at m < m_f only those inside the m prefix. This is forced by D4.2: an anchor can be predicted only if its own row at m_f exists. For m > m_f the anchor set is therefore fixed at the m_f-prefix anchors, and neighbours are searched in the whole m prefix.
- **What a v1.1 estimator reads** for anchor i: its own A- and B-row at m_f (distances to the other m_f − 1 items), pooled summaries of those same rows over the predicted anchors (variant iii's prior), and, at the prediction size m, its A-row only. No B entry at any m ≠ m_f is read (`test_no_B_entry_outside_fit_rows_is_read`).
- **Observed** s_N(m, k) = mean over the predicted anchors of |N_k^A(i) ∩ N_k^B(i)|/k, SPEC tie rule (D_ij, pool position). Cells at m = m_f are in-sample and never scored.

### A2. Predicted rank band and fit window of cell (m, k)

With n_f = m_f − 1 columns per fit row, the k-NN boundary at size m sits near fit-rank ρ = k·n_f/(m − 1) (1-based ranks in the fit row sorted by the tie rule).
- Band(m, k) = {r : ρ/2 ≤ r ≤ 2ρ}; it may be empty when ρ < ½ (e.g. m ≥ 2048 at k = 1, where the in-sample overlap is at most m_f/m of the columns).
- Window(m, k) = usable ranks in [ρ/8, 8ρ], extended upward by whole ranks (then downward) until it holds N_MIN = 64 usable ranks. Usable = outside the band for variants i and iii; all ranks for variant ii.

### A3. The four pre-listed variants (D4.1)

| id | name in code | definition |
|---|---|---|
| (i) | `i-LSlog` | Per-anchor multiplicative residual model, fitted on Window \ Band: log b = x + a_i + β_i(x − x̄_i) + exp(c_i + γ_i(x − x̄_i))·ζ with x = log u. (a, β) by OLS of log(b/u) on x; (c, γ) by OLS of log\|residual\| on x (Harvey's multiplicative-heteroscedasticity estimator). ζ: the anchor's own standardized residuals, as a variance-corrected Gaussian KDE (≤ 101 quantile atoms, Silverman bandwidth, support ± 7h). Below the window, location and log-scale are extrapolated linearly in x; above it the deviation terms are held at the window top. β ∈ [−5, 5], γ ∈ [−4, 4]. Non-positive distances (exact duplicates; the additive P-Gauss plant) enter the logs at half the row's smallest positive value. |
| (ii) | `ii-IsoXF` | Per-anchor row-isotonic link with even/odd candidate-level cross-fitting: g_E = isotonic fit of b on u over even fit-ranks, g_O over odd fit-ranks; residual of an odd-rank column b − g_E(u), of an even-rank column b − g_O(u). Dispersion in B units by the same Harvey model on Window (no band exclusion; cross-fitting is this variant's guard); b = g(u) + exp(c + γ(x − x̄))·ζ, ζ as in (i). At prediction a fit-row column is centred by the other-parity link, a new column by the mean of both links (`test_isoxf_centre_does_not_use_own_b`). Links interpolate linearly; below the smallest fitted u they extrapolate linearly with the non-negative slope of the first 8 fitted points; above, they are constant. |
| (iii) | `iii-LSlog-EB` | Variant (i) with (c_i, γ_i) shrunk toward their precision-weighted mean over the predicted anchors (normal–normal empirical Bayes, method-of-moments prior variance, each parameter separately); ζ is re-standardized with the shrunk scale. **Base = (i)**, fixed a priori: (i)'s dispersion is dimensionless (log domain), so pooling it across anchors does not mix local scales, which is the v1.0 failure D4 diagnoses; (ii)'s dispersion is in B units. |
| (iv) | `iv-v10` | The frozen v1.0 P1 (`P1-med-B-r8`): coupling pooled over H1 × H1 of the m_f prefix, predicted on the same anchors. Reference; it is in the candidate set like the others. |

### A4. D2: computation mode (decided here, on cost and on plant evidence)

- **Mode `d2` (primary):** exact expected discrepancy E\|N_A \ N_B\| for **k ≤ 3** (k = 1: closed product form `exact_loss_k1`, identical to the frozen `exact_loss` at k = 1, tested to 1e-12 and against Monte Carlo; k = 2, 3: the frozen `exact_loss`), SPEC balance-threshold μ for k > 3.
- **Not extended to all k.** Cost: the exact computation is O(G·(L·k + k³)) per anchor, with G = 400 grid cells and L the number of columns that can cross, which approaches m in the saturated regime (L ≈ 16 000 at m = 16384). At k = 100 that is ≥ 6·10⁸ operations per anchor; for ≈ 500 anchors × 6 sizes × 2 directions × 5 draws × 29 pairs it is infeasible. Plant evidence: on P-Gauss the μ-law's own error is ≤ 0.006 at k = 10 and ≤ 0.003 at k ≥ 30 (G0 report, oracle μ-law column), against 0.047–0.062 at k = 1. As a further check, the exact value at k = 10 is computed for variants i, ii, iii and the oracle on replicate 0 of every plant at m ∈ {256, 512, 2048}, and reported.
- **Mode `mu`:** SPEC μ for every k (the literal v1.0 functional), reported for every predictor.
- The same mode is applied to every predictor **and** to the model-based nulls (N1, oracle).

### A5. Nulls

- **N0** (v1.1): observed s_N of the predicted anchors at m_f, from their own rows, i.e. the like-for-like "fit-m value" for the evaluated anchor set. It is stronger than the v1.0 N0 (H1 anchors at m_f), which shares no anchor with the evaluation; the v1.0 version is reported as `N0H1`.
- **N1**: independent additive Gaussian noise in A units with σ_A = RMS of the global affine residual on H1 × H1 at m_f / slope (`hrhv2.predict.fit_n1`), computed in the same mode as P1 (`d2`). The exact expected discrepancy is the expected s_N of N1's own model; the μ-law is an approximation that is biased by ≈ 0.05 at k = 1, so computing P1 exactly and N1 by the μ-law would credit P1 with a computational, not an estimation, advantage. The μ-law N1 (v1.0 computation) is reported as `N1|mu`. D4.8: if the affine slope is ≤ 0 or undefined, N1 := N0.
- **oracle** (P-Gauss plants only): the crossing law with the true σ, both modes.

### A6. Plants (D4.3, D4.5)

Base cloud and generators as in `hrhv2.plants` (torus of dimension p in ℝ⁶⁴); nested pools m ∈ {256, 512, 1024, 2048, 4096, 8192, 16384} are prefixes of one generated cloud; fit m_f = 1024; k ∈ {1, 10, 100}; p ∈ {2, 4, 8}; direction A → B (as in G0); seed prefix **`S11sel`** (fresh). Only the predicted anchors' rows (≈ 512 × 16384) and the 1024 × 1024 fit block are formed (`analysis/s11_select/sel_common.py`; the P-Gauss noise is the exact `gauss_noise_matrix` stream, tested).

| class | definition | role |
|---|---|---|
| P-Gauss | D_B = D_A + E, E iid N(0, σ²), σ = c·median D_(10) (m = 1024 reference), c ∈ {0.02, 0.05, 0.15} | selection |
| P-Aniso | first intrinsic circle scaled by 2 (`aniso_features`) | selection; gate vs N0 |
| P-Conf | sphere inversion (`conf_features`) | selection; gate vs N0 |
| P-Rough | shared + private noise, κ = 0.3 (`rough_features`, as G0) | selection; gate vs N1 |
| P-Smooth+noise (new) | B = sphere inversion of the torus with its first circle scaled by 1.5 (smooth, neither linear nor conformal); both models carry independent private noise in 32 dims with variance s_a² = 0.03 × median D_(10) of that model's smooth part (256-point reference cloud, one map per p) | selection |
| P-GaussSat (new in the set) | P-Gauss with c = 0.4 (G0's most saturated setting; the s11 dry run's P-GaussSat) | gate vs N1 |

Replicates: **4** per (class, p[, c]) (as G0's slope plants): 36 P-Gauss units, 12 units for each other class, 96 units in all; replicate 0 also carries the k = 10 exact diagnostic (A4).

### A7. S_sel real-data check (D4.4; no selection on it)

- Data: `features/wit4096_sel`, 1024 rows minus the flagged S-overlap row 953 (1023 rows). Item id `wit4096:<src_row>`.
- Features: relative depth 0.5 with the S1.2 layer rule (stored index floor(0.5·(L_stored − 1) + 0.5)); `prh_clip+l2` with one all-layer clamp threshold per model computed on the 1023 usable rows (checked against the S1.2 cache); float64 D.
- Pairs (fixed by rule, `sel_common.SSEL_PAIRS`): models shared with the COCO design where extracted on WIT.
  - ViT–ViT: (DINOv2-B, CLIP-B), (DINOv2-B, MAE-B), (DINOv2-B, AugReg-B), (CLIP-B, MAE-B) — the first 4 combinations of the COCO ViT order (SigLIP is not extracted on WIT).
  - LM–ViT: {BLOOMZ-1b7, OpenLLaMA-3B} × {DINOv2-B, CLIP-B}.
  - LM–LM: (BLOOMZ-1b7, OpenLLaMA-3B); (BLOOMZ-1b7, BLOOMZ-3b), (OpenLLaMA-3B, OpenLLaMA-7B) and (LLaMA-7B, OpenLLaMA-7B) stand in for the Pythia seed pairs (not extracted on WIT); the last is the verifier's documented failure pair.
- 5 nested draws (`hash_order` salts `hrh-v2-s11sel-draw0..4`), prefixes m ∈ {256, 512, 1023}, m_f = 512, k ∈ {1, 10, 100}, both directions; the prediction of a cell is the mean of the two directions (COCO plan O6); draws are averaged per cell (O3).
- Statistic: median of \|log(pred/obs)\| over the pooled cells of all 12 pairs with draw-mean obs > 2k/(m − 1) and m ≠ 512 (the criterion-(1) cell rule); pred ≤ 0 or missing counts as +∞. **Ineligible if > log 1.5 = 0.405.**

### A8. Selection rule (D4.3–D4.7; implemented in `analysis/s11_select/select_v11.py`)

1. **Plant score** of a variant = max over the five selection classes (P-Gauss, P-Aniso, P-Conf, P-Rough, P-Smooth+noise) of the class-mean absolute error \|pred − obs\| over eligible cells: replicate-mean obs ∈ [0.05, 0.99], m ≠ m_f; predictions and obs averaged over replicates first (as G0); mode `d2`.
2. **Eligibility (D4.4)**: the S_sel statistic of A7 ≤ log 1.5.
3. **Null gate (D4.5)**: class MAE strictly below N0's on P-Conf and on P-Aniso, and strictly below N1's on P-Rough and on P-GaussSat (same cells, same mode).
4. Among variants passing 2 and 3: minimal plant score. Variants within 0.002 of the minimum are tied and resolved by the a-priori order (i) > (iii) > (ii) > (iv) (simplest first).
5. If no variant passes 2 and 3: record **"no eligible estimator"** (D4.7); S1.1 then runs descriptively only.

Reported, never governing: the gate with `N0H1` instead of N0 and with `N1|mu` instead of N1; the whole rule in mode `mu`; S_sel per direction and per pair class; the exact-vs-μ check at k = 10.

### A9. Code at the time of Part A

- `hrhv2/predict_v11.py`: `a031b2136fb651340ff0589b7ed584e553b5b42c1de35fd64d9deb53ca5cd31e`
- `hrhv2/tests/test_predict_v11.py`: `c3f6f7fd9870341458eeafe2e1e32cccf90b3b2a1a336657c807757f3f7160f3`
- `analysis/s11_select/sel_common.py`: `1f7a7be31835178d11b5a437028950c92032d6ea40739899c9c73dd8db045ea1`
- `analysis/s11_select/run_plants.py`: `6397095a3ff0cb9fd107b6bb875b75d39c8a7a4d79d91d07e6df7e4d5566a5bd`
- `analysis/s11_select/run_ssel.py`: `edb268344a04bdf55e9e3b7b128415182827ed7e5f9a47348cf75e4f548dda54`
- `analysis/s11_select/select_v11.py`: `56b270bff72f5b661703e42dfbe04754ab7b9b5bfc97169bc72df468d954675b`
- `analysis/s11_select/test_select.py`: `a19f60dc73d091b6bdb3ad4da7819b06dcbe61350ee53fbc1c684017c5f82fc0`
- `hrhv2/predict.py`: `a17964b4dc4a04673eda1a0798fa5d27afb3ea5c61ea8a9b251a2a2d5777b74c`

Tests at Part A: hrhv2 122 passed (100 pre-existing + 22 new), analysis/s11_select/test_select.py 5 passed.

### A10. Development disclosure

- Unit tests only: `hrhv2/tests/test_predict_v11.py` (22 tests) on toy couplings (seed keys `v11test|…`), and `analysis/s11_select/test_select.py` (5 tests). The recovery test uses a coupling that variant (i) specifies exactly (D_B = D_A·exp(σZ), p = 4, m_f = 512); at σ = 0.06 variants i and iii were within 0.008 of observed s_N on the out-of-sample cells m ∈ {256, 1024, 2048}, k ∈ {1, 10, 100} (N1 up to 0.63 off). This was inspected once as a sanity check of the implementation. No plant class of A6 and no S_sel pair was evaluated before this Part A.
- One timing-only smoke ran on dev seeds (prefix `S11selDev`, units P-GaussSat p = 2, P-Rough p = 2, P-Conf p = 8, P-Smooth+noise p = 2; output `results/S1_1_select/dev_timing_smoke/`). Only wall times and the absence of NaN/crashes were inspected, in order to fix REPS; no prediction or error from it was looked at, and nothing in A1–A8 was changed after it.
- No constant in A2–A5 (band and window widths, N_MIN, KDE settings, clips, exact-k cut) was tuned on any plant or real data; all were set when the module was written.

## Part B: result (filled after the formal runs)

Formal runs:
- **Plants:** 2026-10-08T07:03Z–10:15Z, seed prefix `S11sel`, 96 units, 0 failed, 31.1 CPU-h on 10 workers × 1 BLAS thread.
- **S_sel:** 07:03Z–08:26Z, 120 units (12 pairs × 5 draws × 2 directions), 0 failed, 2.7 CPU-h.

The code is unchanged since Part A: every sha256 in A9 was re-verified after the runs. The rule was applied by `analysis/s11_select/select_v11.py` (sha256 `56b270bf…`). Outputs: `results/S1_1_select/selection_summary.json`, `selection_report.md`, `plant_cells.parquet`, `ssel_cells.parquet` and `d2_exact_k10_diag.parquet`.

### B1. Outcome per variant (mode d2)

| variant | plant score (max class MAE) | worst class | S_sel median \|log(pred/obs)\| (≤ 0.405) | P-Conf vs N0 | P-Aniso vs N0 | P-Rough vs N1 | P-GaussSat vs N1 | passes (2)+(3) |
|---|---|---|---|---|---|---|---|---|
| (i) `i-LSlog` | 0.067 | P-Rough | 0.056 eligible | 0.014 < 0.043 ok | 0.021 > 0.014 **fail** | 0.067 > 0.051 **fail** | 0.062 > 0.003 **fail** | no |
| (ii) `ii-IsoXF` | 0.257 | P-Aniso | 0.119 eligible | 0.116 > 0.043 **fail** | 0.257 > 0.014 **fail** | 0.032 < 0.051 ok | 0.026 > 0.003 **fail** | no |
| (iii) `iii-LSlog-EB` | 0.062 | P-Rough | 0.063 eligible | 0.007 < 0.043 ok | 0.025 > 0.014 **fail** | 0.062 > 0.051 **fail** | 0.056 > 0.003 **fail** | no |
| (iv) `iv-v10` | 0.521 | P-Conf | 1.818 **ineligible** | 0.521 > 0.043 **fail** | 0.060 > 0.014 **fail** | 0.038 < 0.051 ok | 0.004 > 0.003 **fail** | no |

### B2. Plant class-mean absolute error

Eligible cells: replicate-mean obs ∈ [0.05, 0.99] and m ≠ 1024.

| predictor | P-Gauss (146 cells) | P-Aniso (54) | P-Conf (51) | P-Rough (52) | P-Smooth+noise (54) | P-GaussSat (53) | score |
|---|---|---|---|---|---|---|---|
| (i) | 0.030 | 0.021 | 0.014 | 0.067 | 0.039 | 0.062 | 0.067 |
| (ii) | 0.039 | 0.257 | 0.116 | 0.032 | 0.202 | 0.026 | 0.257 |
| (iii) | 0.029 | 0.025 | 0.007 | 0.062 | 0.036 | 0.056 | 0.062 |
| (iv) | 0.005 | 0.060 | 0.521 | 0.038 | 0.487 | 0.004 | 0.521 |
| N0 (own anchors) | 0.088 | 0.014 | 0.043 | 0.146 | 0.069 | 0.146 | 0.146 |
| N0H1 (v1.0) | 0.088 | 0.014 | 0.044 | 0.146 | 0.071 | 0.145 | 0.146 |
| N1 (d2) | 0.004 | 0.447 | 0.664 | 0.051 | 0.548 | 0.003 | 0.664 |
| N1\|mu (v1.0) | 0.012 | 0.445 | 0.664 | 0.052 | 0.548 | 0.005 | 0.664 |
| oracle (d2 / mu) | 0.004 / 0.012 | – | – | – | – | 0.003 / 0.005 | – |

### B3. S_sel

Fit m = 512. Cells: m ∈ {256, 1023} with draw-mean obs > 2× chance (57 cells).

| predictor | pooled | ViT–ViT | LM–LM | LM–ViT | A→B only | B→A only |
|---|---|---|---|---|---|---|
| (i) | 0.056 | 0.053 | 0.016 | 0.782 | 0.085 | 0.081 |
| (ii) | 0.119 | 0.121 | 0.055 | 1.198 | 0.126 | 0.107 |
| (iii) | 0.063 | 0.057 | 0.026 | 0.617 | 0.079 | 0.080 |
| (iv) | 1.818 | 3.775 | 0.606 | 2.445 | 1.497 | 3.223 |
| N0 | 0.141 | 0.190 | 0.043 | 0.349 | – | – |
| N1 | 0.781 | 0.943 | 0.404 | 1.160 | 0.715 | 0.897 |

### B4. Sensitivities (reported; none changes the outcome)

- **N0H1 instead of N0 in the gate:** no variant passes.
- **μ-law N1 (the v1.0 computation) instead of N1 in mode d2:** no variant passes.
- **Whole rule in mode `mu` (SPEC literal):** scores are (i) 0.062, (ii) 0.266, (iii) 0.058, (iv) 0.530. S_sel eligibility is unchanged, and no variant passes the gate.
- **D2 check:** at k = 10 (replicate 0, m ∈ {256, 512, 2048}), |μ − exact| ≤ 0.0070 for (i), 0.0053 for (iii), 0.0045 for (ii) and 0.0049 for the oracle. This supports restricting the exact computation to k ≤ 3.

### B5. FROZEN

**Outcome: no eligible estimator (D4.7).**

- No variant passes both the S_sel eligibility check (D4.4) and the null gate (D4.5). Variants (i), (ii) and (iii) pass D4.4, but each fails at least two gate classes. Variant (iv) fails both D4.4 and D4.5.
- No v1.1 P1 estimator is frozen.
- S1.1 is run **descriptively only**: observed s_N(m, k) curves, slope classes and the nulls. The crossing law is reported as an **unvalidated predictor**.
- `hrhv2/predict_v11.py` is frozen at sha256 `a031b2136fb651340ff0589b7ed584e553b5b42c1de35fd64d9deb53ca5cd31e`, as the code of the descriptive S1.1 run (protocol, nulls N0/N1 and the unvalidated predictor).

This outcome is not changed without a new version of this file and a STATUS entry.

### B6. Reading (diagnostic; nothing here was used to select)

**The failures are class-specific, and they are not where v1.0 failed.**
- The multiplicative variants (i) and (iii) are accurate on the smooth classes: P-Conf 0.014 and 0.007, P-Smooth+noise 0.039 and 0.036. Variant (iv) scores 0.521 and 0.487 there.
- (i) and (iii) fail on the additive-noise classes P-Rough and P-GaussSat, mostly at p ≤ 4 and k = 1, where they predict less collapse at large m than occurs. The cause: the log-domain model floors non-positive distances, so it cannot represent an additive residual larger than the distance itself. Example: P-GaussSat, p = 2, m = 16384, k = 1 gives obs ≈ 0.03 against (i) ≈ 0.30.
- (i) and (iii) also lose narrowly to N0 on P-Aniso: 0.021 and 0.025 against 0.014. The loss is mainly at p = 8 (0.033 against 0.011), where s_N is flat in m by construction, so N0 is nearly exact.
- Variant (ii) shows the reverse pattern. It beats N1 on P-Rough but fails on every smooth class, because its isotonic link, fitted on half a row, is too noisy at small ranks.
- On P-GaussSat, N1 is the generating model and sits at the oracle (0.003). No anchor-local estimator approaches it: each one fits its dispersion on only ≥ 64 columns of a single row.

**Real data (S_sel) confirms the anchor-local diagnosis of D4.**
- (i) and (iii) predict s_N out of sample with a median |log ratio| of 0.056 and 0.063, i.e. a typical error of about 6%. For comparison: N0 0.141, N1 0.78, frozen v1.0 P1 1.82.
- Example: DINOv2-B vs AugReg-B at m = 1023, k = 10 has obs 0.322; (iii) predicts 0.321 and v1.0 predicts 0.030.
- The weak class is LM–ViT. Its observed values are near chance (0.01–0.06 at k ≤ 10), and every variant over-predicts there: (iii) scores 0.62 against N0's 0.35.

**What decided the outcome.** The binding constraints are two gate clauses: beat N1 on P-GaussSat (N1's own model), and beat N0 on P-Aniso (flat by construction). Under the rule stated before the runs, this is a failure, and it is recorded as one.
