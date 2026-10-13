# OpenLLaMA axis: the verdict contract between E, I, R, `forecast_ol.py` and `joint_summary.py` (S0b, v1)

Written at step S0b by role `oll-axis-extract` (2026-10-09, UTC), before any OpenLLaMA-13B byte exists on ds2213. It is hashed in the S0b `CODE FROZEN` line together with the code.

**What this file is.** It is an implementation contract for the frozen plan `ANALYSIS_PLAN_OPENLLAMA.md` (sha256 `4f18d710beda45492b1e191f74580bfc4ef53f6dafe791ad245ff1cf335f3240`).
- It names every quantity that V-O5 (E vs I) and R must compare, so that the comparisons are mechanical.
- It fixes the operational details that the plan leaves to code: tie handling, tails of p, key strings of the rngs.
- It changes no threshold, band, rule, prediction or category of the plan.
- Where the plan's text and this file disagree, the plan wins. The disagreement is then an implementation error, fixed as a logged deviation.
- Choices this file makes that the plan does not state are marked **[op]** (operationalisation).

## 0. Files and format

| writer | file | sections |
|---|---|---|
| E aggregate (`early/aggregate_ol.py`) | `results/S1_2_layermax_openllama/early/verdicts_ol.json` | A, B, C, D, G |
| I aggregate (`indep/lmx_ol_aggregate.py`) | `results/S1_2_layermax_openllama/indep/verdicts_ol.json` | A, B, C, D, G |
| `forecast_ol.py --impl E` / `--impl I` | `results/S1_2_layermax_openllama/forecast/{E,I}/verdicts_forecast.json` | F |
| `joint_summary.py --impl E` / `--impl I` | `results/S1_2_layermax_openllama/joint/{E,I}/verdicts_joint.json` | J |
| R (`review/ol_axis_review.py`) | `results/S1_2_layermax_openllama/review/{E,I}/verdicts_review.json` | A–D, F, G, J recomputed from the pair files of E (resp. I) |

- Each file is JSON: `{"schema": "OL-VERDICTS v1", "impl": "<E|I|R-E|R-I>", "inputs": {path: sha256}, "v": {key: value}}`.
- Values: JSON numbers (float or int), strings, booleans, or `null` (undefined, e.g. every ViT constant).
- Keys use `|` as separator. Keys starting with `x|` are implementation-specific extras and are never compared.
- `verdict_keys.py` lists the required keys of each section (`required_keys(section)`), and `compare_verdicts.py` compares two files:
  - same required keys present;
  - strings, booleans and ints exactly equal;
  - floats with |a − b| ≤ 1e-12·max(|a|, |b|) + 1e-15;
  - `null` = `null`, NaN = NaN.
- V-O5 = `compare_verdicts.py` on E vs I (sections A–D, G), on `forecast/E` vs `forecast/I`, and on `joint/E` vs `joint/I`. 0 mismatches are required.
- R compares its own recomputation with each implementation's files: `review/E` vs E and `review/I` vs I.

## 1. Names

- `P` ∈ {`prh_clip+l2`, `l2`}. `M` ∈ {`mknn`, `cka_biased`, `cka_debiased`}.
- Sizes `3b`, `7b`, `13b` = `openlm-research__open_llama_{3b,7b,13b}`, with size ranks 1, 2, 3 (num_params order). ViTs: the 17 stems of `analysis/s12/common.py::VITS`, in that order.
- Protocols `Q`:
  - `full` (layer-max over all stored layers), `fixed_r0.5`, `fixed_r1.0`;
  - `grid2`, `grid3`, `grid5`, `grid9`, `grid17`;
  - `LMall_ViTlast`, `LMlast_ViTall`, `noLMemb`, `LMdeep`, `LMemb_only`;
  - and `ctm` (calibrate-then-max).
  - Layer sets are exactly those of `analysis/s12_layermax_llama/early/lm_core.py::subsets`.
- Scores `X`:
  - for every Q except `ctm`: `raw` (T = subset maximum of the observed matrix), `s_cal`, `unc`;
  - for `ctm`: `M`, `M_unc`, `TM` (the observed score at the M-argmax; first maximum in row-major order, lower LM layer first).
- Pooled panels (ARH order of `prh_models.get_models("val")`):
  - `11`: the 5 BLOOMZ, 3b, 7b, 13b, llama-7b, llama-13b, llama-30b. ARH ranks 1..11; these equal the dense ranks.
  - `10`: `11` without 13b. ARH ranks 1–7 and 9–11; dense ranks 1..10.
  - `8`: the frozen 8-LM panel.
- Sources:
  - 3b/7b/13b: this implementation's own reruns;
  - BLOOMZ and llama-7b: frozen `results/S1_2_layermax/` (E: `pairs/`, I: `indep/`);
  - llama-13b/30b: the implementation's own LLaMA-axis pair files (plan §3).
- `s_O(P)`, `material(P)`, `s_late(P)`, `material_Blate(P)` and the B̄, B_late and B_fd values are read from `partB_features_ol.json` (§8 below). They are never recomputed by E, I or R from features.

## 2. Exactness (plan F2)

- **mKNN.** Every quantity is computed from the integer counts out of m·k = 10,240 as `fractions.Fraction`:
  - T and the null counts;
  - c_α (the 476th ascending order statistic of {T} ∪ {T_k}, as a count);
  - the null mean (Σ counts / 500);
  - s_cal = max((T − c)/(1 − c), 0); uncensored = T − null mean;
  - entry-wise c_α and null mean for `ctm`; M, M_unc and TM.
  - Order relations, ranks and ties use the exact values.
  - Values are written to JSON as `float(Fraction)`.
- **CKA.** Ties only on exact float64 equality.
  - **Near-tie audit:** a size pair with |a − b| ≤ 1e-12·max(|a|, |b|) is counted in `neartie|P|M|Q|X`. The decision uses the exact values.
- **p-values and comparisons with tolerance 1e-9.**
  - Relabelling counts compare Tr(σ) with Tr_obs at 1e-9.
  - "|Tr| = 1" means ||Tr| − 1| ≤ 1e-9.
  - "Tr(raw) = −s" means |Tr + s| ≤ 1e-9.
  - Category thresholds (0.10, 0.20, 0.40, 0.60) are compared without tolerance. **[op]**

## 3. Section A: trends (all P, M, Q, X)

- **ρ_v.** The Spearman correlation between the size ranks (1, 2, 3) and the three scores of ViT v. Average ranks are used for ties (`scipy.stats.rankdata`), with the Pearson formula on ranks.
- **Excluded ViTs.** A ViT with three exactly equal scores is excluded. A ViT with any non-finite score is excluded too. The count is `tr_nconst|P|M|Q|X`.
- **Tr.** The mean of ρ_v over the remaining ViTs: `tr|P|M|Q|X`. It is null if none remain.
- **Exact p over the 6 relabellings σ.** σ permutes the size ranks, the same σ for every ViT. With tolerance 1e-9:
  - `tr_pge|…` = #{Tr(σ) ≥ Tr_obs − 1e-9}/6;
  - `tr_ple|…` = #{Tr(σ) ≤ Tr_obs + 1e-9}/6;
  - `tr_ptwo|…` = #{|Tr(σ)| ≥ |Tr_obs| − 1e-9}/6.
- **Differences.** `dtr|P|M|Q|X` = Tr(X) − Tr(base).
  - For Q ≠ `ctm`, X ∈ {`s_cal`, `unc`} and base = `raw` of the same Q.
  - For Q = `ctm`, X ∈ {`M`, `M_unc`, `TM`} and base = `raw` of `full`.
  - The same σ is applied to both terms. `dtr_pge`, `dtr_ple` and `dtr_ptwo` are defined as for Tr.
- **D3 decomposition** (plan §6), for every P and M:
  - `d3cal|P|M` = Tr(`ctm`,`M`) − Tr(`ctm`,`TM`);
  - `d3sel|P|M` = Tr(`ctm`,`TM`) − Tr(`full`,`raw`).
- **Near ties.** `neartie|P|M|Q|X` (int) for M ≠ `mknn` and for X = `raw` as well.
- **ViT-bootstrap CIs of Tr** (key `S1.2|bootvit|layermax-openllama|<stat>`) are written as `x|` extras. They are not compared, because the plan does not fix `<stat>`.

## 4. Section B: order changes (all P, M, and every Q including `ctm`)

- **Scores.** X ∈ {`s_cal`, `unc`} for Q ≠ `ctm`, and X ∈ {`M`, `M_unc`} for `ctm`. The raw score is `raw` of the same Q (of `full` for `ctm`).
- **Order signs.** For a comparison (v, a < b): d_raw = sign(raw_b − raw_a) and d_X = sign(X_b − X_a), each in {−1, 0, +1} (exact for mKNN).
- **New comparisons (34).** a ∈ {3b, 7b} and b = 13b, over the 17 ViTs.
  - `chg|P|M|Q|X|new|down` = #{d_X < d_raw};
  - `chg|P|M|Q|X|new|up` = #{d_X > d_raw};
  - `chg|…|new|c2d` = #{d_raw = +1, d_X = −1};
  - `chg|…|new|d2c` = #{d_raw = −1, d_X = +1}.
- **Pilot comparisons (17).** a = 3b and b = 7b, with the same four counts under `pilot`.
- **F_new** (plan §5.2), with s = `s_O(P)`:
  - f₋ = `down` if s = +1, else `up`; f₊ = the other one;
  - `fnew|P|M|Q|X|F` = (f₋ − f₊)/34;
  - `fnew|…|fminus` = f₋ and `fnew|…|fplus` = f₊;
  - `fnew|…|cp_lo`, `fnew|…|cp_hi` = the 95% Clopper–Pearson bounds of f₋/34 (§9).

## 5. Section C: Part B decision rules (plan §9, §10)

Unless stated otherwise: M = `cka_biased`, Q = `full`, and every key exists for both P.

- **P-O1** (M = `mknn`, Q = `full`).
  - `PO1|P|holds` = |dtr(s_cal)| < 0.10 and |dtr(unc)| < 0.10.
  - **ζ check** over all 51 comparisons (3 size pairs × 17 ViTs), in exact rationals:
    - for X = `s_cal`, ζ_j = |c_α,j − 10/1023|; for X = `unc`, ζ_j = |null mean_j − 10/1023| **[op]**;
    - a comparison qualifies if |raw_b − raw_a| > ζ_a + ζ_b;
    - `PO1|P|zeta|X|nq` counts qualifying comparisons, `…|nrev` those whose X order is the strict reverse of the raw order, and `…|ntie` those tied after;
    - a nonzero `nrev` is an implementation error.
- **P-O2.** For X ∈ {`s_cal`, `unc`}, with s = `s_O(P)` and Δ = dtr(`full`, X):
  - `PO2|P|X|cat` is decided in this order:
    1. not `material(P)` → `"no category (not material)"`;
    2. |Tr(raw) + s| ≤ 1e-9 → `"not testable"`;
    3. s·Δ ≤ −0.10 → `"loss reproduced"`;
    4. s·Δ ≥ +0.10 → `"moved with the baseline"`;
    5. |Δ| < 0.10 → `"trend kept"`.
  - `PO2|P|X|dist` = min(|s·Δ + 0.10|, |s·Δ − 0.10|).
  - `PO2|P|X|kedge` = dist ≤ 1/17 + 1e-9 **[op: one order change moves Tr by at most 1/17]**.
- **P-O2a(i).**
  - `PO2ai|P|median` = the median over the 17 new pairs (13b, v) of c_α − B̄(13b, v).
  - `PO2ai|P|in_band` = 0.0009 ≤ median ≤ 0.0029.
  - `PO2ai|P|n` = 17; `PO2ai|P|x` = #pairs with offset in [0.0009, 0.0029]; `cp_lo` and `cp_hi` as in §9.
  - `PO2ai|P|lp_median_abs` = the median of |c_α − B̄|; `PO2ai|P|lp_le_0.01` = lp_median_abs ≤ 0.01.
  - `PO2ai|P|offset|<vit>` = the realised offset of each new pair.
- **P-O2a(ii).**
  - Over the 34 new comparisons with |B̄(13b, v) − B̄(a, v)| ≥ 0.005: `n`.
  - `x` = #{sign(c_α,13b − c_α,a) = sign(B̄_13b − B̄_a)}.
  - `need` = ⌈2n/3⌉.
  - `PO2aii|P|verdict` = `"not evaluable"` if n < 6, `"holds"` if x ≥ need, else `"fails"`.
  - Keys: `PO2aii|P|n`, `|x`, `|need`, `|verdict`, `|cp_lo`, `|cp_hi`.
- **P-O3.** For r ∈ {`0.5`, `1.0`} and X ∈ {`s_cal`, `unc`}, with Q = `fixed_r<r>` and s_r = sign(Tr(raw)):
  - `PO3|P|r|X|desc` = `"not applied"` if |Tr(raw)| < 0.10;
  - else `"keeps sign"` if s_r·Tr(X) ≥ 0.10;
  - else `"lost"` if |Tr(X)| < 0.10;
  - else `"reversed"` if s_r·Tr(X) ≤ −0.10.
- **Keep identity** (plan §8; all M, all Q ≠ `ctm`, all 51 comparisons).
  - Comparisons where neither score is censored (s_cal > 0 for both), for X = `s_cal`:
    - the predicted keep is T_b − T_a ≥ (1 − T_a)(c_b − c_a)/(1 − c_a);
    - for X = `unc` it is T_b − T_a ≥ μ_b − μ_a (for unc, all comparisons).
  - The observed keep is X_b − X_a ≥ 0.
  - `keepid|P|M|Q|X|nviol` counts comparisons where predicted and observed keep differ, excluding |lhs − rhs| ≤ 1e-12 for CKA. It must be 0.
- **Sign rule** (plan §8), checked per comparison **[op: the per-comparison form of "can turn a concordant pair discordant, never the reverse", which also holds with censoring at 0]**.
  - Scope: all M, all Q ≠ `ctm`, all 51 comparisons (a < b).
  - The threshold t is c_α for X = `s_cal` and the null mean for X = `unc`.
  - A comparison violates the rule if:
    - t_b > t_a, d_raw ≤ 0 and d_X = +1; or
    - t_b < t_a, d_raw ≥ 0 and d_X = −1; or
    - t_b = t_a and d_X = −d_raw ≠ 0.
  - `signrule|P|M|Q|X|nviol` counts the violations. It must be 0.
- **P-O4** (M = `cka_debiased`, Q = `full`).
  - `PO4|P|X|holds` = dtr(X) ≥ −0.10, for X ∈ {`s_cal`, `unc`}.
  - `PO4|P|arh|holds` = r(11, P, `cka_debiased`, `s_cal`, `arh`) ≥ r(…, `raw`, …) − 0.05.
- **P-O5.** For X ∈ {`M`, `M_unc`}, with s = `s_late(P)` and Δ = dtr(`ctm`, X):
  - `PO5|P|X|pred_holds` = s·Δ ≤ 0. It is null if B_late is not material.
  - `PO5|P|X|cat` = the P-O2 categories with s_late, decided in this order:
    1. not `material_Blate(P)` → `"no category (not material)"`;
    2. P-O5a verdict `"fails"` → `"premise failed"`;
    3. |Tr(`full`,`raw`) + s| ≤ 1e-9 → `"not testable"`;
    4. otherwise as for P-O2.
- **P-O5a.**
  - Over the new comparisons with |B_late(13b, v) − B_late(a, v)| ≥ 0.005: `n`.
  - `x` = #{sign(c^M_13b − c^M_a) = sign(B_late,13b − B_late,a)}, where c^M is the entry c_α at the M-argmax.
  - `PO5a|P|n`, `|x`, `|need` = ⌈2n/3⌉, and `|verdict` (`"not evaluable"` if n < 6 **[op: as P-O2a(ii)]**).
- **D3 deductive check.** `D3|P|nviol` = #ViTs with c^M strictly rising in size and ρ_v(M) − ρ_v(TM) > 0. It must be 0.
- **Plug-in.**
  - `plugin|P|tr` = Tr over the 17 ViTs of (T − B̄) (Q = `full`, raw T).
  - `plugin|P|diff` = plugin tr − Tr(`full`, `unc`).
  - `plugin|P|within` = |diff| ≤ 0.10.
- **§7 OpenLLaMA-axis reading.** For M ∈ {`cka_biased`, `cka_debiased`}, `readO|P|M` is:
  - `"not applied"` if Tr(`full`, `raw`, M) < 0.60;
  - else `"loses"` if Tr(`full`, `unc`, M) ≤ 0.40 and Tr(`full`, `unc`, `mknn`) ≥ 0.60;
  - else `"no loss"` if Tr(`full`, `unc`, M) ≥ 0.60;
  - else `"intermediate"`.
- **§5.6 curves.**
  - **Materiality τ.** For τ ∈ {0.03, 0.06, 0.10, 0.15, 0.20}:
    - `curve|P|PO2|X|tau=<τ>` = the P-O2 category with τ in place of 0.10;
    - `curve|P|PO1|tau=<τ>`;
    - `curve|P|PO5|X|tau=<τ>`.
  - **δ_b.** For δ ∈ {0.0025, 0.005, 0.0075, 0.01}:
    - `curve|P|material|db=<δ>` = (|Tr_pred| = 1 and the median B̄ gap ≥ δ for all 3 size pairs), computed from the per-ViT B̄ of Part B;
    - `curve|P|PO2aii|db=<δ>|n`, `…|x`, `…|verdict`.
  - Numbers are formatted with Python `repr(float)`, e.g. `tau=0.1`, `db=0.0025`.
- **D1** (all M). `D1|P|M|<size>` = the median over ViTs of (null mean of the layer-max − max_uv entry null mean).
- **D2.**
  - `D2|P|M|<size>|argmax_rel` = the median over ViTs of argmax_A/(L_stored − 1) of the observed layer-max.
  - `D2|P|M|<size>|c_minus_centry` = the median over ViTs of c_α − c_α^entry(a*, b*).
- **7B > 3B.** `frac7gt3|P|M` = the fraction of ViTs with T_7b > T_3b (`full`, `raw`).

## 6. Section D: pooled ARH panel (plan §5.3, §7)

- **r.** `arh|<panel>|P|M|X|<kind>|r` = the Pearson r across all pairs of the panel between the score and the LM rank.
  - X ∈ {`raw`, `s_cal`, `unc`} of `full`.
  - kind ∈ {`arh`, `dense`, `logp`}, where logp is the natural log of num_params.
  - The rank vector is each LM's value repeated over its 17 ViTs. r = `np.corrcoef` of the ravelled (LM × ViT) grid, as `aggregate_llama.py::pear`.
- **Exact p** (panel `11`, kind `arh` only).
  - `arh|11|P|M|X|arh|pge` = #{r(σ) ≥ r_obs − 1e-9}/11! over all 11! LM relabellings. r(σ) is linear in the LM row sums, as `aggregate_llama.py::arh_test`.
- **CI** (panel `11`, every kind).
  - `arh|11|P|M|X|<kind>|ci_lo` and `…|ci_hi`: 2000 draws.
  - rng `util.rng_from_key("S1.2-layermax-openllama|boot|arh_r|" + f"{P}|{M}|{X}|{kind}")`.
  - Draw b: `j = rng.integers(0, 17, 17)`, value pear(G[:, j], x). Bounds `np.nanpercentile(bs, [2.5, 97.5])`.
- **ΔΔ.** For M ∈ {`cka_biased`, `cka_debiased`} and every panel:
  - `dd|<panel>|P|M` = (r_raw − r_scal)_M − (r_raw − r_scal)_mknn, with kind `arh`.
- **ΔΔ CI** (panel `11`). `dd|11|P|M|ci_lo` and `…|ci_hi` come from exactly `aggregate_llama.py::delta_boot`, with the key and the panel changed:
  - rng `util.rng_from_key(f"S1.2-layermax-openllama|boot|arh_delta|{P}|{M}")`;
  - 2000 draws; `j = rng.integers(0, 17, 17)`, applied to all four 11-LM grids (M raw, M s_cal, mKNN raw, mKNN s_cal);
  - `np.nanpercentile` 2.5 and 97.5.
- **ΔΔ exact p** (panel `11`). `dd|11|P|M|pge` = #{ΔΔ(σ) ≥ ΔΔ_obs − 1e-9}/11!, with the same σ for all four grids.
- **§7(a) rule** (M = `cka_biased`, the CI just defined). `rule7a|P` is:
  - `"replicated the direction"` if ci_lo > 0;
  - `"direction not established"` if ci_lo ≤ 0 ≤ ci_hi;
  - `"opposite direction"` if ci_hi < 0.
  - `size_not_reproduced|P` = 0.40 > ci_hi, where 0.40 is ARH's ΔΔ.
- **FP§6 category** (panel `11`, M = `cka_biased`, kind `arh`).
  - `fp6|P|raw_cat` is:
    - `"ARH pattern reappears"` if ΔΔ ≥ 0.20 and r_after(CKA) ≤ r_after(mKNN) − 0.20;
    - else `"absent"` if ΔΔ ≤ 0.10 or r_after(CKA) ≥ r_after(mKNN) − 0.10;
    - else `"partial"`.
  - **Gap CI [op].** `fp6|P|gap_ci_lo` and `…|gap_ci_hi` are the 95% CI of r_after(mKNN) − r_after(CKA):
    - rng `util.rng_from_key(f"S1.2-layermax-openllama|boot|arh_gap|{P}")`;
    - 2000 draws of `j = rng.integers(0, 17, 17)`; nanpercentile 2.5 and 97.5.
  - `fp6|P|kedge` = (dd ci_lo ≤ 0.10 ≤ dd ci_hi) or (gap_ci_lo ≤ 0.10 ≤ gap_ci_hi).
  - `fp6|P|cat` = `"not decidable at this panel size (knife edge)"` if kedge, else raw_cat.

## 6b. Section G: provenance checks (each aggregate)

- `G|llama_pairs_equal_merged_summary` (bool): the llama-13b/30b layer-max values this implementation reads from its own LLaMA-axis pair files equal the values embedded in `results/S1_2_layermax_llama/layermax_llama_summary.json` (`db9242f9…`), at 1e-12 relative (plan R.3.7).
- `G|rerun_equals_frozen_3b7b` (bool): V-O4b.
  - The rerun 3b/7b layer-max outputs equal the frozen 8-LM files to ≤ 1e-12 relative.
  - p and the mKNN null counts are identical.
  - If false, the plan §3 source rule applies, and the `x|frozen|…` extras carry every OpenLLaMA-axis verdict recomputed on the frozen values.
- `G|n_pairs_read` (int): the number of (LM, ViT, pre) records read for the 11-LM panel, which is 11·17·2 = 374.

## 7. Sections F and J (single scripts run once per implementation)

### Section F: P-O2b (`forecast_ol.py --impl E|I`)

The script reads `partB_features_ol.json` (boundaries, frozen 3b/7b values) and, from the implementation's pair files, only the 13b observed T, s_cal, null mean and the 3b/7b reruns needed for the observed orders.

For P and X ∈ {`s_cal`, `unc`}, and each new comparison (a ∈ {3b, 7b}, v):
- **Forecast** `PO2b|P|X|dec|<a>|<vit>`:
  - `"below"` if T_13b,v < lo_a,v;
  - `"above"` if T_13b,v > hi_a,v;
  - else `"undecided"`.
  - lo and hi are the Part B boundaries (κ^lo and κ^hi for `s_cal`; T_a,v + B̄_13b,v ∓ 0.0002 − μ_a,v for `unc`).
  - If the frozen s_cal of a is 0, the s_cal forecast is `"undecided"` **[op: 13b cannot rank strictly below a censored score]**.
- **Observed** `PO2b|P|X|obs|<a>|<vit>`: `"below"` if X_13b < X_a, `"above"` if X_13b > X_a, `"tie"` otherwise. The orders come from this implementation's reruns.
- **Correct** `PO2b|P|X|ok|<a>|<vit>` = dec = obs. It is null if undecided.
- **Totals:**
  - `PO2b|P|X|n_decided`, `PO2b|P|X|x_correct`, `PO2b|P|X|n_undecided`;
  - `PO2b|P|X|cp_lo` and `…|cp_hi` of x/n.
- **Verdict** `PO2b|P|X|verdict`: `"fails"` if x < n; else `"not evaluable"` if n < 10; else `"holds"` **[op: a wrong forecast is reported as "fails" even when n < 10]**.
- **h curve** (X = `s_cal`). For h ∈ {0.0003, 0.0005, 0.001, 0.0015, 0.002}, with band [0.0019 − h, 0.0019 + h]:
  - `PO2b|P|s_cal|h=<h>|n_decided`;
  - `PO2b|P|s_cal|h=<h>|x_correct`.

### Section J: joint summary J, EXPLORATORY (`joint_summary.py --impl E|I`)

Primary pre, M = `cka_biased`, Q = `full`, for X ∈ {`s_cal`, `unc`}.
- **ΔTr per family.** ΔTr_f(σ_f) for:
  - B = BLOOMZ (120 relabellings; frozen 8-LM files of the implementation's format);
  - L = LLaMA-1 (6; the implementation's LLaMA-axis files);
  - O = OpenLLaMA (6; this run).
- **Directions.** s_B = s_L = +1, and s_O = `s_O_J(P)` from Part B.
- **Keys:**
  - `J|X|sO`;
  - `J|X|Jmin` = max_f s_f·ΔTr_f(id), with `J|X|pmin` = #{σ : J_min(σ) ≤ J_min(id) + 1e-9}/4320;
  - `J|X|pmin_product` = n_B(t)·n_L(t)·n_O(t)/4320 with t = J_min(id) and n_f(t) = #{σ_f : s_f·ΔTr_f(σ_f) ≤ t + 1e-9}. It must equal pmin;
  - `J|X|Jmean` = mean_f s_f·ΔTr_f(id), with `J|X|pmean` = #{σ : J_mean(σ) ≤ J_mean(id) + 1e-9}/4320;
  - `J|X|nO` = #{σ_O : s_O·ΔTr_O(σ_O) ≤ s_O·ΔTr_O(id) + 1e-9};
  - `J|X|cond_size` (below);
  - `J|X|dTr|B`, `J|X|dTr|L`, `J|X|dTr|O` (identity values).
- **Conditional size.** `J|X|cond_size` = #{σ_O : v0's rule rejects when σ_O is taken as the observed labelling}/6.
  - Under σ_O, the model labelled 13b is the "new" one. Its new comparisons are counted with the order changes of §4 (f₋ for s_O), under the relabelled size order.
  - v0's rule rejects iff all three hold:
    - J_min p ≤ 0.05, recomputed with that σ_O as identity;
    - s_O·ΔTr_O(σ_O) ≤ −0.10;
    - f₋ ≥ 3.
- **Labels.** `J|label` = `"EXPLORATORY"`, and `J|provenance` = `"BLOOMZ hypothesis-generating; BLOOMZ fails B4; BLOOMZ and LLaMA known before this plan"`.

## 8. `partB_features_ol.json` (written by `partb_ol.py` at S5; read by E, I, R and forecast)

Required fields. `partb_ol.py` may add more.

```
{"schema": "OL-PARTB v1", "m": 1024, "delta_b": 0.005,
 "band": {"o": 0.0019, "o_lo": 0.0009, "o_hi": 0.0029, "h": 0.001}, "mu_band": [-0.0002, 0.0002],
 "ol_lms": [3 slugs], "vits": [17 stems], "pooled11": [11 slugs, ARH order],
 "num_params": {slug: int}, "L_stored": {stem: int},
 "per_pre": {P: {
    "clip_c": {stem: float or null},
    "PR": {stem: [float per stored layer]}, "PR_star": {stem: float}, "u_star": {stem: int}, "u_star_rel": {stem: float},
    "Bbar": {lm: {vit: float}}, "Bbar_argmax": {lm: {vit: [u, w]}},
    "Bfd": {"0.5": {ol_lm: {vit: float}}, "1.0": {...}},
    "Blate": {ol_lm: {vit: float}},
    "gaps": {"Bbar"|"Bfd_0.5"|"Bfd_1.0"|"Blate": {"3b|7b"|"7b|13b"|"3b|13b": {"per_vit": [17 floats, B(larger) - B(smaller)], "median": float}}},
    "Tr_pred": float, "material": bool, "s_O": +1|-1, "s_O_J": +1|-1,
    "Tr_Bfd": {"0.5": float, "1.0": float}, "material_Bfd": {"0.5": bool, "1.0": bool},
    "Tr_Blate": float, "material_Blate": bool, "s_late": +1|-1,
    "known": {"T"|"c_alpha"|"null_mean"|"s_cal": {"3b"|"7b": {vit: float}}},
    "boundaries": {"s_cal": {"lo": {"3b"|"7b": {vit: float}}, "hi": {...}},
                   "unc": {"lo": {...}, "hi": {...}},
                   "s_cal_h": {"<h>": {"lo": {...}, "hi": {...}}}}}}}
```

- `known` holds the frozen 8-LM layer-max values of 3b/7b for biased CKA, Q = `full`, read at Part B (plan B6).
- `Tr_pred` = Tr_O(B̄). `s_O` = sign(Tr_pred); if Tr_pred = 0, `s_O` = +1 and this is flagged. `s_O_J` = `s_O` if material, else +1.
- `s_late` = sign(Tr_Blate), with +1 if 0.
- B_late(i, v) = the maximum over LM states with relative depth u/(L_stored − 1) ≥ 0.60, and over all ViT layers, of √(PR_u·PR_w)/1023.
- `clip_thresholds_partB_ol.json` = `{"clip": {stem: {"c": float}}}` (the format of `clip_thresholds_partB.json`).

## 9. Shared small definitions

- **Clopper–Pearson 95% (x of n).**
  - lo = 0 if x = 0, else `scipy.stats.beta.ppf(0.025, x, n − x + 1)`;
  - hi = 1 if x = n, else `beta.ppf(0.975, x + 1, n − x)`;
  - null if n = 0.
- **Median.** `numpy.median` of float64 values. For mKNN, the median of the `float(Fraction)` values.
- **sign(0)** = 0. A strictly monotone sequence has no equal neighbours.

## 10. Test mode (S0b only)

- Every gated script calls `guard.require_partb_frozen(out_root)` first. Before the CODE FROZEN line exists, the S0b unit tests may run it with `OL_S0B_TEST=1` and an out_root under `~/oll_s0b_scratch/`.
- Test inputs are either synthetic, or known files relabelled. For example, the LLaMA-axis pair files of llama-7b/13b/30b can stand in for 3b/7b/13b, under the slug `synth__ol13b` for the stand-in 13b.
- No test file name contains `open_llama_13b`.
- After CODE FROZEN, `OL_S0B_TEST=1` makes the guard exit 3.
