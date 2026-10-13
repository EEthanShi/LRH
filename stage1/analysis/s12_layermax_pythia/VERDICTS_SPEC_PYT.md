# Pythia axis: the verdict contract between E, I, R and `forecast_pyt.py` (S0b, v1)

Written at step S0b by role `pyt-code-infra` (2026-10-10, UTC), before any Pythia byte exists on ds2213. Adapted from `analysis/s12_layermax_openllama/VERDICTS_SPEC.md` (sha256 `61da992e…`; pristine copy `orig_copy/VERDICTS_SPEC.md`, diff `VERDICTS_SPEC_PYT.md.diff`). It is hashed in the S0b `CODE FROZEN` line together with the code.

**Amended 2026-10-10 by §15 (Amendment A1: lead rulings R1 and R2, recorded by `pyt-code-infra`).** §15 is additive; no earlier line of this file was changed. Where §15 and an earlier section differ, §15 governs. Version before A1: sha256 `2d5332ebfc35c3c98d800ea627cc2914994ca19f43ac749d89d1e5b149a4b4e1`.

**What this file is.** An implementation contract for the frozen plan `ANALYSIS_PLAN_PYTHIA.md` (sha256 `6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b`).
- It names every quantity that V-P5 (E vs I) and R must compare, so that the comparisons are mechanical.
- It fixes the operational details that the plan leaves to code: tie handling, tails of p, rng draw order, key strings, file names.
- It changes no threshold, band, rule, prediction or category of the plan.
- Where the plan's text and this file disagree, the plan wins. The disagreement is then an implementation error, fixed as a logged deviation (a hashed amendment, plan §7).
- Choices this file makes that the plan does not state are marked **[op]** (operationalisation). Places where the plan cannot be implemented literally are marked **[plan-gap]** and are reported to the lead.
- Key names come from `verdict_keys.py` (`required_keys(section)`); this file defines their values.

## 0. Files and format

| writer | file | sections |
|---|---|---|
| E aggregate (`early/aggregate_pyt.py`) | `results/S1_2_layermax_pythia/early/verdicts_pyt.json` | A, B, C, H, R, D, G |
| I aggregate (`indep/lmx_pyt_aggregate.py`) | `results/S1_2_layermax_pythia/indep/verdicts_pyt.json` | A, B, C, H, R, D, G |
| `forecast_pyt.py --impl E` / `--impl I` | `results/S1_2_layermax_pythia/forecast/{E,I}/verdicts_forecast.json` | F, K |
| R (`review/…`) | `results/S1_2_layermax_pythia/review/{E,I}/verdicts_review.json` | A–D, G, H, R, F, K recomputed from the pair files of E (resp. I) and Part B |

- Each file is JSON: `{"schema": "PYT-VERDICTS v1", "impl": "<E|I|R-E|R-I>", "inputs": {path: sha256}, "v": {key: value}}`.
- Values: JSON numbers (float or int), strings, booleans, or `null` (undefined, e.g. a trend with no valid ViT, or every key of a pre that was not run).
- Keys use `|` as separator. Keys starting with `x|` are implementation-specific extras and are never compared.
- `compare_verdicts.py` compares two files:
  - the required keys of the selected sections must be present on both sides;
  - strings, booleans and ints exactly equal; floats with |a − b| ≤ 1e-12·max(|a|, |b|) + 1e-15; `null` = `null`, NaN = NaN.
  - **It prints and writes the number of mismatches and the names of mismatching or missing keys only, never their values** (plan §11 V-P5, critique B6.4).
- **V-P5** = `compare_verdicts.py` on E vs I (`--sections EI`) and on `forecast/E` vs `forecast/I` (`--sections FK`). 0 mismatches are required.
- **R** compares its own recomputation with each implementation's files: `review/E` vs E and vs `forecast/E`, `review/I` vs I and vs `forecast/I`.
- **[op] K is written by `forecast_pyt.py`.** The §7 category needs P-P2b, which exists only after `forecast_pyt.py` has run (plan §11 S8 order: aggregates, cross-validation, `forecast_pyt.py`, R). So `forecast_pyt.py --impl X` reads implementation X's aggregate file (sections C, H, R) and Part B, writes F, and then K from them. It recomputes nothing of C, H or R; it checks their internal consistency (`K|…|h_consistent`).
- **Legacy mode (V-P8 only).** The generalised aggregators accept `--legacy`; with n = 3 on the frozen OpenLLaMA pair files they write a file with schema `OL-VERDICTS v1` and the OpenLLaMA key names (`verdict_keys.legacy_required_keys("A"|"B"|"C")`, which imports the frozen `analysis/s12_layermax_openllama/verdict_keys.py` read-only after checking its sha256). Comparison: `compare_verdicts.py <legacy file> results/S1_2_layermax_openllama/layermax_openllama_summary.json --b-sub embedded.verdicts.E --legacy --sections A,B,C` (and `.I`). Legacy outputs go only to the S0b scratch root (§10).

## 1. Names

- `P` ∈ {`prh_clip+l2`, `l2`}. `M` ∈ {`mknn`, `cka_biased`, `cka_debiased`}. `x` ∈ {`s_cal`, `unc`} (the primary and the uncensored version of every confirmatory statistic).
- Sizes `70m`, `160m`, `410m`, `1b`, `1.4b`, `2.8b`, `6.9b` = `EleutherAI__pythia-<size>`, with size ranks 1..7 (num_params order). Size pairs are written `<smaller>|<larger>` (21 pairs, `verdict_keys.SIZE_PAIRS`). A comparison is (j < j′, v): 21 × 17 = 357.
- ViTs: the 17 stems of `verdict_keys.VITS`, in that order (= `analysis/s12/common.py::VITS`). ViT families for §5.5(d) (plan §1, four families): `in21k` (4 AugReg), `MAE` (3), `DINOv2` (4), `CLIP` (the 3 LAION-2B and their 3 `ft_in12k` versions) = `verdict_keys.VIT_FAMILY4`. (`analysis/s12/common.py` splits CLIP into two labels; the plan's four families govern.)
- Protocols `Q`: `full`, `fixed_r0.5`, `fixed_r1.0`, `grid2`, `grid3`, `grid5`, `grid9`, `grid17`, `LMall_ViTlast`, `LMlast_ViTall`, `noLMemb`, `LMdeep`, `LMemb_only`, and `ctm`. Layer sets are exactly those of the copied `lm_core.py::subsets` (`grid_layers` deduplicates with a set; grid5 has 5 LM layers for every Pythia model; `grid9` for 70m and `grid17` for 70m/160m are saturated).
- Scores `X`: for every Q except `ctm`: `raw` (T = subset maximum of the observed matrix), `s_cal`, `unc`; for `ctm`: `M`, `M_unc`, `TM` (the observed score at the M-argmax; first maximum in row-major order, lower LM layer first).
- Sources:
  - Pythia: this implementation's own runs (E: `results/S1_2_layermax_pythia/early/pairs/<P>/<lm>__<vit>.json`; I: `results/S1_2_layermax_pythia/indep/<P>/pairs/<lm>__<vit>.npz`).
  - Cross-family (section D only): BLOOMZ, llama-7b, open_llama_3b/7b from `results/S1_2_layermax/` (E: `pairs/`, I: `indep/`); llama-13b/30b from `results/S1_2_layermax_llama/{early,indep}/`; open_llama_13b from `results/S1_2_layermax_openllama/{early,indep}/`; each implementation reads its own counterpart (plan §3).
- Part B values (`s`, `s_late`, membership, H2 branch, B̄, B_late, B_fd, bands, extrapolation flags, materiality curves, width, depth, num_params) are read from `partB_features_pyt.json` (§8). E, I and R never recompute them from features.

## 2. Exactness (plan F2)

- **mKNN.** Every quantity is computed from the integer counts out of m·k = 10,240 as `fractions.Fraction`: T and the null counts; c_α (the 476th ascending order statistic of {T} ∪ {T_k}); the null mean (Σ counts / 500); s_cal = max((T − c)/(1 − c), 0); uncensored = T − null mean; entry-wise c_α and null mean for `ctm`; M, M_unc and TM. Order relations, ranks and ties use the exact values. Values are written to JSON as `float(Fraction)`.
- **CKA.** Ties only on exact float64 equality. **Near-tie audit:** `neartie|P|M|Q|X` counts the comparisons (of 357) with |a − b| ≤ 1e-12·max(|a|, |b|) (exact equality included, as the OpenLLaMA spec). Decisions use the exact values.
- **Spearman ρ_v** = Pearson correlation of the size ranks (1..n under the relabelling) with the average ranks (`scipy.stats.rankdata`, method `average`) of the n scores of ViT v, computed as Σ(x−x̄)(r−r̄)/√(Σ(x−x̄)²·Σ(r−r̄)²) in float64.
- **Relabelling counts** compare with tolerance 1e-9 (plan F2): `p_le` = #{σ : stat(σ) ≤ stat(id) + 1e-9}/N, `p_ge` = #{σ : stat(σ) ≥ stat(id) − 1e-9}/N, `p_two` = #{σ : |stat(σ)| ≥ |stat(id)| − 1e-9}/N, with N = 5040 (7 sizes), 720 (6) or 120 (5). σ permutes the size ranks; the same σ is applied to every ViT and to both terms of a difference. Written as `count / N` (Python float division).
- **Bootstrap counts [op]** compare with tolerance 1e-9 in the direction that keeps the p conservative: p_mag counts D*_b ≥ −m − 1e-9; TOST p_lo counts Δ*_b ≤ −m + 1e-9 and p_hi counts Δ*_b ≥ m − 1e-9. (E and I compute the same draws; the tolerance removes any dependence on float rounding of the mean.)
- **Category and decision thresholds** (0.10 and τ, 0.40, 0.60, 0.75, 0.05, 0.05/2, 0.05/3, ±m) are compared without tolerance **[op]**, as in the OpenLLaMA spec. p-values are compared as `count / N` floats with the literals 0.05, 0.05/2 and 0.05/3.
- **Exact zero of Δρ_v [op].** "Δρ_v exactly 0" (degenerate TOST, plan §5.4 I1) is decided in exact arithmetic: with integer size ranks and half-integer average ranks, ρ = num/√(S_x·S_r) with num and S_r rational; ρ_X = ρ_base iff sign(num_X) = sign(num_base) and num_X²·S_r,base = num_base²·S_r,X (`fractions.Fraction`).

## 3. Section A: trends (all P, M, Q, X)

- **Valid ViT for one score.** Its 7 scores are finite and not all exactly equal. `tr_nconst|P|M|Q|X` = 17 − #valid.
- **Tr.** `tr|P|M|Q|X` = mean of ρ_v over the valid ViTs; `null` if none.
- **Exact p over 5040 relabellings:** `tr_pge`, `tr_ple`, `tr_ptwo` (§2).
- **Sensitivity ρ_v := 0** (plan §5.1, §5.5(e)). `tr0|P|M|Q|X` = mean over the ViTs with 7 finite scores of ρ_v, with ρ_v := 0 for a constant vector (normally all 17 ViTs).
- **Differences on the common ViT set (plan §5.1, critique B2.2).** For X ∈ `dscores(Q)` and base = `raw` of the same Q (for Q = `ctm`, base = `raw` of `full`):
  - V(base, X) = the ViTs valid for both scores; `dtr_nv|P|M|Q|X` = |V|;
  - `dtr|P|M|Q|X` = mean over V of [ρ_v(X) − ρ_v(base)]; `null` if V is empty;
  - `dtr_pge`, `dtr_ple`, `dtr_ptwo`: §2, with Tr_V(X; σ) − Tr_V(base; σ) for every σ (same σ for both terms, V fixed);
  - `dtr0|P|M|Q|X` = `tr0`(X) − `tr0`(base), over the ViTs with finite scores for both (ρ_v := 0 for constant vectors).
  - With 3 sizes and no fully censored ViT this equals the OpenLLaMA definition (Tr(X) − Tr(base) over per-score sets); V-P8(b) lists the keys where it does not.
- **D3 decomposition** (plan §6, as OP§6): `d3cal|P|M` = mean over V(M, TM) of ρ_v(`ctm`,`M`) − ρ_v(`ctm`,`TM`); `d3sel|P|M` = mean over V(TM, raw full) of ρ_v(`ctm`,`TM`) − ρ_v(`full`,`raw`).
- **Near ties:** `neartie|P|M|Q|X` (int, M ≠ `mknn`, X = `raw` included).
- **ViT-bootstrap CIs of Tr** (key `S1.2|bootvit|layermax-pythia|<stat>`) may be written as `x|` extras; they are not compared.

## 4. Section B: order changes over the 357 comparisons (all P, M, every Q including `ctm`)

- X ∈ {`s_cal`, `unc`} for Q ≠ `ctm`, X ∈ {`M`, `M_unc`} for `ctm`; the raw score is `raw` of the same Q (of `full` for `ctm`).
- For a comparison (v, j < j′): d_raw = sign(raw_j′ − raw_j), d_X = sign(X_j′ − X_j), in {−1, 0, +1} (exact for mKNN).
- `chg|P|M|Q|X|all|down` = #{d_X < d_raw}; `…|up` = #{d_X > d_raw}; `…|c2d` = #{d_raw = +1, d_X = −1}; `…|d2c` = #{d_raw = −1, d_X = +1}; `…|c2t` = #{d_raw = +1, d_X = 0}; `…|t2c` = #{d_raw = 0, d_X = +1}; `…|d2t` = #{d_raw = −1, d_X = 0}; `…|t2d` = #{d_raw = 0, d_X = −1}.
- **Plan §5.3 counts.** `fcnt|…|n` = 357; f₋ = `fcnt|…|fminus` = c2d (concordant with size under T, discordant under the score); f₊ = `fcnt|…|fplus` = d2c; `fcnt|…|F` = (f₋ − f₊)/357; `fm_cp_lo/hi` and `fp_cp_lo/hi` = the 95% Clopper–Pearson bounds of f₋/357 and f₊/357 (§9). Tied outcomes are reported by the `chg` counts. **[op]** "Concordant" means d = +1, "discordant" d = −1.

## 5. Section C: Part B rules, implementation checks, attribution (plan §6, §8, §9)

Unless stated otherwise: M = `cka_biased`, Q = `full`, and every key exists for both P and both x. For x = `s_cal` the threshold t is c_α; for x = `unc` it is the null mean μ ("same rules, secondary", plan §9).

- **P-P1** (M = `mknn`, Q = `full`; exact rationals). ζ_i = |t_i − 10/1023| **[op: μ for x = `unc`, as the OpenLLaMA spec]**. A comparison qualifies if |T_j′ − T_j| > ζ_j + ζ_j′. `PP1|P|x|nq` = #qualifying; `…|nrev` = #qualifying with d_x = −d_raw ≠ 0; `…|ntie` = #qualifying with d_x = 0 and d_raw ≠ 0; `PP1|P|x|holds` = (nrev = 0); `PP1|P|holds` = both x hold. A reversal is an implementation error (plan §9 P-P1). **[op]** A tie created by censoring is reported, not counted as a change of order.
- **P-P2a(i)** per model i (plan §9; band from Part B, never recomputed):
  - offset_{i,v} = t(i, v) − B̄(i, v); `PP2ai|P|x|offset|<size>|<vit>`;
  - `PP2ai|P|x|median|<size>` = `numpy.median` over the 17 ViTs;
  - `PP2ai|P|x|lo|<size>`, `…|hi|<size>` = the band: for x = `s_cal` Part B `per_pre.P.band.<size>.lo/hi` (f(PR*_i) ∓ h); for x = `unc` the uncensored band `mu_band` = [−0.0003, +0.0008] for every model and both P **[op: plan B6 gives one uncensored band "for every model"]**;
  - `…|in_band|<size>` = lo ≤ median ≤ hi; `…|x|<size>` = #ViTs with lo ≤ offset ≤ hi; `…|cp_lo|<size>`, `…|cp_hi|<size>` of x/17;
  - `PP2ai|P|x|n_models_in_band`; `PP2ai|P|x|holds` = n_models_in_band ≥ 6; `PP2ai|P|x|failing` = the sizes not in band, comma-joined in size order (`""` if none).
  - Flat-band sensitivity (x = `s_cal` only): Part B `band_flat` ([0.0010, 0.0040] for prh_clip+l2, [0.0010, 0.0045] for l2) in place of band_i: `PP2ai|P|s_cal|flat|in_band|<size>`, `…|flat|n_models_in_band`, `…|flat|holds`.
- **P-P2a(ii)** (consistency check). Over the comparisons with |B̄(j′, v) − B̄(j, v)| ≥ δ_b = 0.005: `PP2aii|P|x|n`; `…|x` = #{sign(t_j′ − t_j) = sign(B̄_j′ − B̄_j)}; `…|need` = ⌈0.9n⌉; `…|need67` = ⌈2n/3⌉; `…|verdict` = `"not evaluable"` if n < 6, `"holds"` if x ≥ need, else `"fails"`; `…|verdict67` the same with need67; `…|cp_lo`, `…|cp_hi` of x/n.
- **P-P2a.** `PP2a|P|x|verdict` = `"fails"` if `PP2ai|P|x|holds` is false or `PP2aii|P|x|verdict` = `"fails"`; `"holds"` if (i) holds and (ii) holds; otherwise `"not evaluable"`.
- **P-P3** (descriptive). For r ∈ {`0.5`, `1.0`}, Q = `fixed_r<r>`, s_r = sign Tr(raw): `PP3|P|r|x|desc` = `"not applied"` if |Tr(raw)| < 0.10; else `"keeps sign"` if s_r·Tr(x) ≥ 0.10; else `"lost"` if |Tr(x)| < 0.10; else `"reversed"`. Tr values are the single-score trends of section A.
- **P-P5a** (CTM premise). Over the comparisons with |B_late(j′, v) − B_late(j, v)| ≥ δ_b: `PP5a|P|x|n`; `…|x` = #{sign(c^M_j′ − c^M_j) = sign(B_late,j′ − B_late,j)}, where c^M is the entry c_α at the M-argmax (x = `s_cal`) or the entry null mean at the M_unc-argmax (x = `unc`) **[op]**; `…|need` = ⌈2n/3⌉; `…|verdict` (`"not evaluable"` if n < 6 **[op]**, `"holds"` if x ≥ need, else `"fails"`); `…|cp_lo`, `…|cp_hi`.
- **Keep identity** (plan §8; all M, every Q ≠ `ctm`, x ∈ {`s_cal`, `unc`}, all 357 comparisons), as the OpenLLaMA spec: for x = `s_cal` over the comparisons with s_cal > 0 for both, predicted keep T_j′ − T_j ≥ (1 − T_j)(c_j′ − c_j)/(1 − c_j); for x = `unc` over all comparisons, T_j′ − T_j ≥ μ_j′ − μ_j; observed keep x_j′ − x_j ≥ 0. `keepid|P|M|Q|x|nviol` counts disagreements, excluding |lhs − rhs| ≤ 1e-12 for CKA. Must be 0.
- **Sign rule per comparison** (plan §8), as the OpenLLaMA spec: a comparison violates it if t_j′ > t_j, d_raw ≤ 0 and d_x = +1; or t_j′ < t_j, d_raw ≥ 0 and d_x = −1; or t_j′ = t_j and d_x = −d_raw ≠ 0. `signrule|P|M|Q|x|nviol`; must be 0.
- **Sign rule per ViT** (plan §8, §7). For each ViT with valid raw and x vectors:
  - `signvit|P|M|Q|x|nmono_up` = #ViTs with t non-decreasing in size over all 7 sizes (t_1 ≤ … ≤ t_7, exact) and no censoring tie (below); `…|nviol_up` = #those with ρ_v(x) > ρ_v(raw) + 1e-12;
  - `…|nmono_down`, `…|nviol_down`: the same for t non-increasing, with ρ_v(x) < ρ_v(raw) − 1e-12 **[op: the mirror case, reported]**;
  - `…|nexcl_tie` = #ViTs with t monotone (either direction) that are excluded because two sizes are tied under x but not under raw.
  - **[plan-gap]** The plan states "if c_α is non-decreasing in size over all 7 sizes for a ViT, ρ_v(score) ≤ ρ_v(raw)". This is false when censoring at 0 ties two sizes whose raw order is discordant: a discordant pair that becomes a tie raises Spearman's ρ (example: raw ranks (2, 1, 3, 4, 5, 6, 7) against size, the two smallest both censored, ρ rises from 0.964 to 0.991, with any non-decreasing thresholds). Without ties the inequality holds (a superset of inversions raises Σd²). The check is therefore applied only to ViTs without censoring ties; excluded ViTs are counted, never treated as errors.
- **D3 deductive check.** `D3|P|nviol` = #ViTs with c^M strictly rising in size, no censoring tie in M that is not a tie in TM, and ρ_v(M) − ρ_v(TM) > 1e-12; `D3|P|nexcl_tie` = #ViTs with c^M strictly rising excluded for a censoring tie **[op, same reason]**. Must be 0.
- **Plug-in** (plan §6). `plugin|P|tr` = single-score Tr of (T − B̄) (Q = `full`, raw T); `plugin|P|diff` = mean over V(T − B̄, `unc`) of ρ_v(T − B̄) − ρ_v(`unc`) **[op: common ViT set]**; `plugin|P|nv` = |V|; `plugin|P|within` = |diff| ≤ 0.10.
- **OP-style reading** (plan §7, continuity). For M ∈ {`cka_biased`, `cka_debiased`}: `readO|P|M` = `"not applied"` if Tr(`full`, `raw`, M) < 0.60; else `"loses"` if Tr(`full`, `unc`, M) ≤ 0.40 and Tr(`full`, `unc`, `mknn`) ≥ 0.60; else `"no loss"` if Tr(`full`, `unc`, M) ≥ 0.60; else `"intermediate"` (single-score trends).
- **D1** (all M). `D1|P|M|<size>` = median over ViTs of (null mean of the layer-max − max_uv entry null mean).
- **D2.** `D2|P|M|<size>|argmax_rel` = median over ViTs of argmax_A/(L_stored − 1) of the observed layer-max; `D2|P|M|<size>|c_minus_centry` = median over ViTs of c_α − c_α^entry(a*, b*).
- **D4** (plan §6). Spearman (§2 ρ formula) over the 7 sizes between a quantity and log num_params, width or depth (number of blocks), widths and depths from Part B: `D4|P|<qty>|logparams|width|depth` with qty ∈ {`PRstar` (Part B PR*), `Bbar` (mean over the 17 ViTs of B̄), `calpha` (mean over ViTs of c_α), `T` (mean over ViTs of raw T)}; `D4|P|<qty>|step|<a>|<b>` = qty(b) − qty(a) for the contrasts 410m→1b, 1b→1.4b, 2.8b→6.9b.
- **D5.** For each adjacent size step: `D5|P|<a>|<b>|raw_gain` = median over ViTs of T(b) − T(a); `…|thr_gain` = median of c_α(b) − c_α(a). `D5|P|g` = median over ViTs of [T(6.9b) − T(70m)] / log₂(N_6.9b/N_70m); `D5|P|h` the same for c_α **[op: the "{g}, {h} per doubling" placeholders of plan §13]**.
- **D6** (M ∈ {`mknn`, `cka_debiased`}, Q = `full`). `D6|P|M|<size>|c_minus_base` = median over ViTs of c_α − baseline (10/1023 for mKNN, 0 for debiased CKA); `D6|P|M|spearman_L` = Spearman over the 7 sizes between c_minus_base and L_stored; `D6|P|M|step|410m|1b` = c_minus_base(1b) − c_minus_base(410m).

## 6. Section H: the confirmatory family (plan §5.4)

Computed for every P and x. **Confirmatory only for P = `prh_clip+l2`, x = `s_cal`**; the other three versions are descriptive and labelled so in every report. All bootstrap draws use `hrhv2.util.rng_from_key(key)` with the key below, P substituted, and the suffix `|unc` appended for x = `unc` **[op]**; the ViTs of the relevant common set V are taken in `verdict_keys.VITS` order, n = |V|, and **draw b is `j = rng.integers(0, n, n)` for b = 0..1999, one call per draw** (the OpenLLaMA convention; one call of shape (2000, n) gives different draws). The statistic of a draw is computed from the fixed per-ViT Δρ_v (`numpy.mean` over `dr[j]`).

- **H1.**
  - `H|P|x|H1|member` = Part B `per_pre.P.material_Bbar`; `…|s` = Part B `s` (sign Tr_P(B̄), +1 if 0).
  - V = V(T, x) for M = `cka_biased`, Q = `full`; `…|nV` = |V|; `…|R` = mean over V of ρ_v(raw); `…|C` = mean over V of ρ_v(x); `…|D` = D₁ = s·(C − R).
  - `…|p_perm` = #{σ : s·ΔTr_V(σ) ≤ D₁ + 1e-9}/5040.
  - Draws D*_b = s·mean(Δρ[j]), key `S1.2-layermax-pythia|boot|H1mag|cka_biased|full|<P>`. `…|p_mag` = (1 + #{b : D*_b ≥ −0.10 − 1e-9})/2001; `…|p_mag|m=<m>` with m ∈ {0.05, 0.1, 0.15} from the same draws.
  - `…|p` = max(p_perm, p_mag); `…|confirmed` = (p ≤ 0.05 and D₁ ≤ −0.10) if member, else `null`. Every other H1 key is computed whether or not H1 is a member (needed for the δ_b curve of section K).
  - `…|ci_lo`, `…|ci_hi` = `numpy.percentile(D*, [2.5, 97.5])` (linear).
  - `…|kedge` **[op]** = min(|D₁ + 0.10|, |D₁ − 0.10|) ≤ 0.0021 + 1e-9, or ci_lo ≤ −0.10 ≤ ci_hi, or ci_lo ≤ +0.10 ≤ ci_hi (plan §5.6: within one adjacent swap, or inside the CI of a boundary).
- **H2.** `…|H2|branch` = Part B `H2_branch` (`"H2a"` iff B_late material, else `"H2b"`); `…|s_late` = Part B `s_late`.
  - V = V(T_full, X) with X = `M` (x = `s_cal`) or `M_unc` (x = `unc`) of `ctm`; Δ₂ = mean over V of ρ_v(X) − ρ_v(raw, full); `…|nV`; `…|D` = D₂ = s_late·Δ₂ (both branches).
  - `…|p_perm` = #{σ : s_late·Δ₂(σ) ≤ D₂ + 1e-9}/5040; `…|p_mag` from draws s_late·mean(Δρ[j]) with key `S1.2-layermax-pythia|boot|H2mag|cka_biased|ctm|<P>`, margins as H1.
  - `…|p_lo`, `…|p_hi`, `…|p_tost|m=<m>`, `…|degenerate`, `…|ranks_identical` = the TOST values of (`cka_biased`, `ctm`) below.
  - `…|p` = max(p_perm, p_mag) for H2a; the TOST p of (`cka_biased`, `ctm`) for H2b.
- **H3** = the TOST of (`mknn`, `full`); **H4** = the TOST of (`cka_debiased`, `full`): `…|D` (= Δ, unsigned), `…|nV`, `…|p_lo`, `…|p_hi`, `…|p`, `…|degenerate`, `…|ranks_identical`, `…|p_tost|m=<m>`.
- **TOST (every metric, Q ∈ {`full`, `ctm`}; descriptive except the members).** `TOST|P|x|M|Q|…`. V = V(base, X) as section A (X = x for `full`, `M`/`M_unc` for `ctm`; base = raw `full`); Δ = mean over V of Δρ_v; draws Δ*_b = mean(Δρ[j]) with key `S1.2-layermax-pythia|boot|tost|<M>|<Q>|<P>`; `p_lo` = (1 + #{Δ*_b ≤ −0.10 + 1e-9})/2001, `p_hi` = (1 + #{Δ*_b ≥ 0.10 − 1e-9})/2001; `p` = max(p_lo, p_hi) if −0.10 < Δ < 0.10, else 1.0. `degenerate` = every Δρ_v over V is exactly 0 (§2); `ranks_identical` = for every v in V the average-rank vectors of X and base are equal. `null` for all if V is empty. If degenerate, every draw is 0 and p = 1/2001 automatically.
- **Gate and Holm** (plan §5.4 procedure).
  - `H|P|x|gate` = `"open (H1 confirmed)"` if H1 is a member and confirmed; `"open (H1 not a member)"` if H1 is not a member; else `"closed"`.
  - If the gate is open: the three members H2 (its branch), H3, H4 with their `p`. Sort ascending by p, ties broken by the order H2, H3, H4 **[op]**; `H|P|x|holm_order` = the sorted names comma-joined (e.g. `"H3,H4,H2"`). Step-down: for i = 0, 1, 2 reject the i-th while p_(i) ≤ 0.05/(3 − i) (that is 0.05/3, 0.05/2, 0.05), stop at the first non-rejection. Adjusted `p_holm` of the i-th = min(1, max_{k ≤ i} (3 − k)·p_(k)).
  - `H|P|x|<h>|tested` = gate open; `…|p_holm`, `…|holm_reject` = `null` if not tested; `…|confirmed` = `null` if not tested, else for H2a holm_reject and D₂ ≤ −0.10, for H2b, H3, H4 holm_reject. `holm_order` = `null` if not tested.

## 7. Section R: robustness (plan §5.5; descriptive, wording modifiers only)

For every P and x, with s the 7-size Part B sign **[op]** and M = `cka_biased`, Q = `full` unless stated.
- **(a) Leave-one-size-out.** For each size z: the 6 remaining sizes keep their order (ranks 1..6); validity of a ViT is decided on the 6-size vectors **[op]**. `R|P|x|loo|z|D1` = s·ΔTr on V₆(T, x); `…|p_perm` over 720 relabellings; `…|p_mag` with key `<H1mag key>|loo|<z>` (suffix after the `|unc` suffix if any) and draws over V₆; `…|p` = max; `…|dtr_mknn`, `…|dtr_dcka` = the 6-size ΔTr for mKNN and debiased CKA (common sets). `R|P|x|fragile` = some z has (D1₆ ≤ −0.10) ≠ (D₁ ≤ −0.10) or (p₆ ≤ 0.05) ≠ (p_H1 ≤ 0.05); `R|P|x|fragile_sizes` = those z, comma-joined in size order (`""` if none).
- **(b) Five-size check** (410m, 1b, 1.4b, 2.8b, 6.9b): `R|P|x|five|D1`, `…|p_perm` over 120; `…|carried` = sign(D1₅) ≠ sign(D₁) or (D1₅ ≤ −0.10) ≠ (D₁ ≤ −0.10); `…|carried_D` = sign differs or (D1₅ ≥ 0.10) ≠ (D₁ ≥ 0.10) (used by category D). sign(0) = 0.
- **(c) grid5.** `R|P|x|grid5|D1` = s·`dtr|P|cka_biased|grid5|x`; `…|dtr_mknn` = `dtr|P|mknn|grid5|x`; `…|dtr_dcka` = `dtr|P|cka_debiased|grid5|x`.
- **(d) Leave-one-ViT-family-out.** For f ∈ {`in21k`, `MAE`, `DINOv2`, `CLIP`}: mean over V minus the ViTs of f: `R|P|x|lofo|f|D1` (s·ΔTr), `…|dtr_mknn`, `…|dtr_dcka`.
- **(e)** `R|P|x|rho0|D1` = s·`dtr0|P|cka_biased|full|x`.
- **M4 inputs** (x-independent): `R|P|n_scal_zero` = #{(i, v) of the 119 pairs with s_cal = 0 (exact)}; `R|P|median_scal|<size>` = median over ViTs of s_cal.

## 8. Section D: cross-family table and 18-LM panel (descriptive)

- `famtr|P|M|X|<fam>` (X ∈ {`raw`, `s_cal`, `unc`}, Q = `full`) = the single-score Tr over the family's own sizes (BLOOMZ 5, LLaMA 3 = llama-7b/13b/30b, OpenLLaMA 3, Pythia 7; size ranks by num_params); `famdtr|P|M|x|<fam>` = ΔTr on the common set.
- `panel18|P|M|X|r` = `numpy.corrcoef` of the ravelled 18 × 17 grid (rows `verdict_keys.PANEL18`, columns `VITS`) against log num_params repeated over ViTs; `…|ci_lo`, `…|ci_hi`: 2000 draws, rng `rng_from_key("S1.2|bootvit|layermax-pythia|panel18|" + f"{P}|{M}|{X}")`, draw b `j = rng.integers(0, 17, 17)`, r on G[:, j], `numpy.nanpercentile([2.5, 97.5])`. Labelled "not ARH's panel"; never used in paper text.

## 9. Section G: provenance (each aggregate)

- `G|n_pairs_read` (int) = Pythia (LM, ViT, pre) records read (238; 119 if l2 was not run).
- `G|n_known_records` (int) = cross-family records read for section D (11·17 per pre run).
- `G|l2_run` (bool). **[op]** If l2 was not run (budget, plan §11), every key with P = `l2` in every section is `null`.
- `G|partB_sha256` (string) = sha256 of the `partB_features_pyt.json` read.
- `G|clip_equal_partB` (bool) = V-P1: the 7 clip thresholds used equal `clip_thresholds_partB_pyt.json` bitwise.

## 10. Sections F and K (`forecast_pyt.py --impl E|I`)

### Section F: P-P2b (plan §9 B6)

Inputs: `partB_features_pyt.json` (B̄, bands, flags) and, from the implementation's pair files, only T, s_cal and the null mean of the layer-max (`cka_biased`, `full`) of every Pythia model. For P, x ∈ {`s_cal`, `unc`} and each comparison (j < j′, v):
- **x = `s_cal`.** S(T; c) = max((T − c)/(1 − c), 0). With [o_lo,i, o_hi,i] = band_i: the largest possible calibrated score of i is S(T_i; B̄_i + o_lo,i), the smallest S(T_i; B̄_i + o_hi,i). `PP2b|P|s_cal|dec|j|j′|v` = `"below"` iff S(T_j′; B̄_j′ + o_lo,j′) < S(T_j; B̄_j + o_hi,j); else `"above"` iff S(T_j′; B̄_j′ + o_hi,j′) > S(T_j; B̄_j + o_lo,j); else `"tie"` iff S(T_j′; B̄_j′ + o_lo,j′) = 0 and S(T_j; B̄_j + o_lo,j) = 0 (both 0 at every offset in the bands); else `"undecided"`.
- **x = `unc`.** U = T − μ with μ − B̄ ∈ `mu_band` = [u_lo, u_hi]: `"below"` iff T_j′ − B̄_j′ − u_lo < T_j − B̄_j − u_hi; `"above"` iff T_j′ − B̄_j′ − u_hi > T_j − B̄_j − u_lo; else `"undecided"` (no ties).
- **Observed** `…|obs|j|j′|v` = `"below"` if x_j′ < x_j, `"above"` if x_j′ > x_j, else `"tie"` (this implementation's s_cal or uncensored). **Correct** `…|ok|j|j′|v` = (dec = obs), `null` if undecided.
- **Totals.** `n_decided` = #{below, above, tie} **[op: a forecast tie is a decided forecast, correct iff both observed s_cal are equal; the plan lists "decided/undecided/ties" and does not say whether ties count toward the 30]**; `x_correct`; `n_wrong`; `n_undecided`; `n_tie_fc`; `n_flip_fc_rawconc` = #{T_j′ > T_j and dec = `"below"`}; `cp_lo`, `cp_hi` of x/n.
- **Verdict** `PP2b|P|x|verdict` = `"fails"` if n_wrong > 0; else `"not evaluable"` if n_decided < 30; else `"holds"` (plan §9).
- `wrong_all_extrap` = `null` if n_wrong = 0, else every wrong comparison has j or j′ extrapolated (Part B flags of P); `wrong_sizes` = the sizes in wrong comparisons, comma-joined in size order (`""` if none).
- **h curve** (x = `s_cal`): for h ∈ {0.0004, 0.0006, 0.0008, 0.001, 0.0015}, band f(PR*_i) ± h (Part B `band_h_curve`): `PP2b|P|s_cal|h=<h>|n_decided`, `…|x_correct`.
- **Flat band** (x = `s_cal`): Part B `band_flat` for every model: `PP2b|P|s_cal|flat|n_decided`, `…|x_correct`, `…|verdict`.

### Section K: category, modifiers, readings (plan §7)

For every P and x (confirmatory only for prh_clip+l2, s_cal). Inputs: this implementation's H, C, R keys, its F keys, and Part B.
- **Category** `K|P|x|category`, applied in this order (plan §7 table; thresholds without tolerance):
  1. H1 not a member → `"E"`;
  2. D₁ ≥ +0.10 → `"D"`;
  3. |D₁| < 0.10 → `"C"`;
  4. D₁ ≤ −0.10 and H1 not confirmed → `"F"`; `K|P|x|F_reason` = `"not consistent across the seven sizes"` if p_perm > 0.05, `"size of the drop not established"` if p_mag > 0.05, both joined with `"; "` if both (else `null`);
  5. H1 confirmed and (`PP2a|P|x|verdict` = `"fails"` or `PP2b|P|x|verdict` = `"fails"`) → `"B"`;
  6. H1 confirmed otherwise: `"A"` if R ≥ 0.60 and C ≤ 0.40; `"A-att"` if R ≥ 0.60 and C > 0.40; `"A-prime"` if R < 0.60. `K|P|x|premise` = `"verified"` if both P-P2a and P-P2b hold, else `"unverified"` (neither fails and at least one is not evaluable); `null` outside A, A-att, A-prime.
  - `K|P|x|R`, `…|C`, `…|nV` = the H1 values.
- **Modifiers** (reported whatever the category):
  - `M1` = `R|P|x|fragile`; `M1_sizes` = `R|P|x|fragile_sizes`.
  - `M2` = not (`R|P|x|grid5|D1` ≤ −0.10) for A, A-att, A-prime, else `null`; `M2_x` = `R|P|x|grid5|D1`; `M2_H3` = |`R|P|x|grid5|dtr_mknn`| ≥ 0.10 if the H3 reading is one of the three "kept" readings or "no comparison changed order", else `null`; `M2_H4` the same with `dtr_dcka`.
  - `M3` = `R|P|x|five|carried_D` for category D, else `R|P|x|five|carried`.
  - `M4` (x = `s_cal` only; `null` for `unc` **[op]**) = `R|P|n_scal_zero` > 119/3 or some size has `R|P|median_scal|<size>` = 0; `M4_sizes` = the sizes with median s_cal = 0, comma-joined; `M4_n_zero` = `R|P|n_scal_zero`.
  - `M5_extrap_sizes` = Part B extrapolated sizes of P, comma-joined. `M5_B` (category B only, else `null`) **[op]** = P-P2a(ii) did not fail, every failing model of P-P2a(i) is extrapolated, and `PP2b|P|x|wrong_all_extrap` is not false. `M5_A` (A, A-att, A-prime only, else `null`) = some extrapolated model is in band (`PP2ai|P|x|in_band|<size>`); `M5_A_sizes` = those sizes.
- **Readings.**
  - `H3_reading` (and `H4_reading`), in this order: |D| ≥ 0.10 → `"moved"`; degenerate → `"no comparison changed order"`; not tested → `"kept, not tested (gate closed)"`; Holm rejects → `"equivalence confirmed"`; else `"kept, equivalence not established"` **[op: the degenerate reading is placed before the gate, since it is deductive]**.
  - `H2_reading`, in this order: `PP5a|P|x|verdict` = `"fails"` → `"premise failed"`; confirmed and H2a → `"follows the late baselines"`; confirmed and H2b → `"kept as predicted"`; otherwise the C/D/F analogues on D₂ = s_late·Δ₂: D₂ ≥ 0.10 → `"moved with the late baselines"`; |D₂| < 0.10 → `"kept"`; D₂ ≤ −0.10 → `"drop not confirmed"`; each analogue gets the suffix `" (not tested, gate closed)"` if H2 was not tested.
  - `H2_same_as_H1` = Part B `H2_same_as_H1` (B_late = B̄ for ≥ 6 of 7 models).
  - Counted as confirmed predictions in the paper: `H2_counted` = H2 confirmed and not H2_same_as_H1; `H3_counted` = H3 reading `"equivalence confirmed"`; `H4_counted` likewise.
- **Knife edge [op].** `kedge` = `H|P|x|H1|kedge`, or (category in A, A-att, A-prime and (|R − 0.60| ≤ 0.0021 + 1e-9 or |C − 0.40| ≤ 0.0021 + 1e-9)); `kedge_which` = the triggering names among `D1`, `R`, `C`, comma-joined (`""` if none).
- **Curves** (plan §5.6). `K|P|x|curve|tau=<τ>` = the category with τ in place of 0.10 in steps 2–4 and in the H1 conjunction "D₁ ≤ −τ" (p_H1 unchanged) **[op]**; `K|P|x|curve|db=<δ>` = the category with H1 membership taken from Part B `material_curve_db.Bbar.<δ>` (confirmed then = p ≤ 0.05 and D₁ ≤ −0.10 from the H values, computed for non-members too).
- `h_consistent` = the H inputs are internally consistent: p = max(p_perm, p_mag); confirmed as defined; the gate string matches member/confirmed; the Holm decisions match the p's.

## 11. `partB_features_pyt.json` (written by `partb_pyt.py` at S5; read by E, I, R and `forecast_pyt.py`)

Required fields; `partb_pyt.py` may add more.

```
{"schema": "PYT-PARTB v1", "m": 1024, "delta_b": 0.005,
 "sizes": [7 labels], "pyt_lms": [7 slugs, size order], "vits": [17 stems], "vit_family4": {stem: family},
 "num_params": {slug: int}, "width": {slug: int}, "depth_blocks": {slug: int}, "L_stored": {stem: int},
 "band_constants": {P: {"a": float, "b": float, "h": float, "fit_min_PR": float, "flat": [lo, hi]}},
 "mu_band": [-0.0003, 0.0008],
 "per_pre": {P: {
    "clip_c": {stem: float or null}, "PR": {stem: [float per stored layer]}, "PR_star": {stem: float},
    "u_star": {stem: int}, "u_star_rel": {stem: float},
    "Bbar": {lm: {vit: float}}, "Bbar_argmax": {lm: {vit: [u, w]}},
    "Bfd": {"0.5": {lm: {vit: float}}, "1.0": {...}}, "Blate": {lm: {vit: float}},
    "gaps": {"Bbar"|"Bfd_0.5"|"Bfd_1.0"|"Blate": {"<a>|<b>": {"per_vit": [17 floats, B(b) - B(a)], "median": float}}},
    "Tr_Bbar": float, "Tr_Bbar_5size": float, "Tr_Bfd": {"0.5": float, "1.0": float}, "Tr_Blate": float,
    "materiality": {"Bbar"|"Bfd_0.5"|"Bfd_1.0"|"Blate": {"abs_tr": float, "median_end_gap": float, "material": bool}},
    "material_Bbar": bool, "material_Blate": bool, "material_curve_db": {"Bbar"|"Blate": {"<db>": bool}},
    "s": +1|-1, "s_flag_zero": bool, "s_late": +1|-1, "s_late_flag_zero": bool,
    "members": ["H1"?, "H2a"|"H2b", "H3", "H4"], "H2_branch": "H2a"|"H2b",
    "Blate_equals_Bbar": {size: bool}, "n_Blate_equals_Bbar": int, "H2_same_as_H1": bool,
    "band": {size: {"f": float, "lo": float, "hi": float}}, "band_h_curve": {"<h>": {size: [lo, hi]}},
    "band_flat": [lo, hi], "extrapolated": {size: bool},
    "qualifying": {"PP2aii_n": int, "PP2aii_need": int, "PP2aii_need67": int, "PP5a_n": int, "PP5a_need": int}}}}
```

- B̄(i, v) = √(PR*_i·PR*_v)/1023; B_fd(i, v; r) at the plan §3 layers; B_late(i, v) = max over LM states with relative depth u/(L_stored − 1) ≥ 0.60 and all ViT layers of √(PR_u·PR_w)/1023. Because these factorise, the size order of each is common to all ViTs and Tr is a single Spearman over the 7 sizes.
- Materiality (plan B4): |Tr| ≥ 0.75 − 1e-9 **[op tolerance]** and the median over ViTs of s_B·[B(6.9b, v) − B(70m, v)] ≥ 3δ_b = 0.015; the curve uses 3δ for δ ∈ {0.0025, 0.005, 0.0075, 0.01}.
- `s` = sign Tr(B̄) (+1 and flagged if 0); `s_late` = sign Tr(B_late) (+1 and flagged if 0).
- band_i = [f(PR*_i) − h, f(PR*_i) + h], f = a + b/√PR*_i with the frozen constants (prh_clip+l2 a = 0.001069, b = 0.012920, h = 0.0008; l2 a = 0.001551, b = 0.006843, h = 0.0018); extrapolated iff PR*_i < 124.5 (prh_clip+l2) or < 79.6 (l2).
- `clip_thresholds_partB_pyt.json` = `{"clip": {stem: {"c": float}}}` (the format of `clip_thresholds_partB_ol.json`).

## 12. Shared small definitions

- **Clopper–Pearson 95% (x of n):** lo = 0 if x = 0, else `scipy.stats.beta.ppf(0.025, x, n − x + 1)`; hi = 1 if x = n, else `beta.ppf(0.975, x + 1, n − x)`; `null` if n = 0.
- **Median:** `numpy.median` of float64 values; for mKNN, of the `float(Fraction)` values.
- **sign(0) = 0.** ⌈·⌉ = `math.ceil` on exact rationals (⌈0.9n⌉ = ⌈9n/10⌉).

## 13. Test mode (S0b only) and the stand-in fixture

- Every gated script calls `guard.require_partb_frozen(out_root)` first. Before the CODE FROZEN line exists, S0b tests may run with `PYT_S0B_TEST=1` (the variable the extraction side uses) and an out_root under `guard.SCRATCH` = `analysis/s12_layermax_pythia/s0b_scratch/` (inside the axis's own tree, plan "Paths"; excluded from the code manifest). After CODE FROZEN, `PYT_S0B_TEST=1` makes the guard exit 3.
- Sub-roots: `s0b_scratch/PB/` (Part B and forecast tests), `s0b_scratch/E/`, `s0b_scratch/I/`, `s0b_scratch/R/`, `s0b_scratch/VP9/` (the shared fixture).
- No test file name contains `EleutherAI__pythia-`. Stand-in slugs are `synth__pyt-<size>`.
- **V-P9 7-size stand-in** (plan §11): sizes 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b ← `bigscience__bloomz-560m`, `-1b1`, `-1b7`, `-3b`, `-7b1`, `huggyllama__llama-13b`, `huggyllama__llama-30b` (num_params increasing). Features: `features/wit1024/` and `features/wit1024_llama_axis/`. Pair files: E `results/S1_2_layermax/pairs/<P>/` and `results/S1_2_layermax_llama/early/pairs/<P>/`; I `results/S1_2_layermax/indep/<P>/pairs/` and `results/S1_2_layermax_llama/indep/<P>/pairs/`. The stand-in Part B is `partb_pyt.py --standin vp9` (test mode), written to `s0b_scratch/PB/partB_vp9/`. Every output of the fixture is labelled "stand-in".

## 14. S0b records read by `s0b_freeze.py`

- `--require-pass FILE`: a JSON file with top-level `"pass": true` (e.g. each implementation's V-P8 and V-P9 record, and the infrastructure test report `s0b_scratch/PB/tests/<UTC>/report.json`). Its sha256 enters the manifest and the `CODE FROZEN` line.
- `--test-log FILE`: any record of S0b test runs, failed runs included (hashed only).
- `--dryruns FILE`: the JSON list `[[label, script, args...], ...]` of every gated command; each must exit 3 and write nothing with `PYT_S0B_TEST` unset. The infrastructure part is `s0b_dryruns_infra.json`; E, I and R add their own gated commands.
- G0: `results/S1_2_layermax_pythia/logs/gpu_gate_G0.json` (written by `gpu_gate_pyt.py g0`) must exist, or `--g0-later` records that the freeze precedes G0.

## 15. Amendment A1 (2026-10-10, written at 2026-10-10T03:19:13Z by `pyt-code-infra`): lead rulings R1 and R2

**Why.** The S0b code freeze was blocked (STATUS line `2026-10-10T02:56:56Z [pyt-axis-code-freeze] S0b CODE FREEZE NOT DONE`): the cross-implementation fixture X found R and `forecast_pyt.py` (E and I sides) disagreeing on 4 keys, `K|prh_clip+l2|unc|M4_n_zero`, `K|prh_clip+l2|unc|M4_sizes`, `K|l2|unc|M4_n_zero`, `K|l2|unc|M4_sizes`, because the M4 bullet of §10 did not say whether "(x = `s_cal` only; `null` for `unc`)" covers M4 alone or the whole bullet. Fixture X also needed an exclusion for `G|clip_equal_partB`. The lead ruled on both at about 2026-10-10T03:05Z, under the author's delegation. This section records the rulings verbatim and states their consequences for every key.

**What A1 does not change.** No key is added or removed (`verdict_keys.py` unchanged, sha256 `6d20e04d65ee592673865b0258843a620d04857568cf1fe41c391bbf24ed8213`). No threshold, band, rule, prediction, category or confirmatory definition of the plan changes. No earlier line of this file is edited.

### 15.1 Ruling R1: a variant-restricted key is `null` outside its domain

> (R1) GENERAL RULE: a key that the contract defines only for some variants (e.g. a modifier defined only for s_cal) is null in every other variant, in every implementation; so under x=unc all three M4 keys (M4, M4_n_zero, M4_sizes) are null.

1. **Who.** E, I, R (its recomputation of every section, F and K included) and `forecast_pyt.py --impl E|I`.
2. **What "null" means.** JSON `null`, never a placeholder such as `0`, `""`, `false` or `"n/a"`. A placeholder outside the domain is a mismatch under §0.
3. **M4 (the blocking item).** For both P, `K|P|unc|M4`, `K|P|unc|M4_sizes` and `K|P|unc|M4_n_zero` are `null`. The parenthetical "(x = `s_cal` only; `null` for `unc`)" of §10 governs the whole M4 bullet. Under x = `s_cal` the three keys stay as §10 defines them: M4 a bool; M4_sizes a string, `""` if no size has median s_cal = 0; M4_n_zero the int `R|P|n_scal_zero`. The x-independent inputs `R|P|n_scal_zero` and `R|P|median_scal|<size>` (§7) are not restricted.
4. **Domain table.** Every key of this contract whose definition carries a restriction is listed below. Outside the stated domain the value is `null` in every implementation. "Text" means the restriction is written in the cited section. "R1" means R1 extends a restriction written on the head key of a §10 bullet to the sub-keys of that bullet, as for M4. R1 adds no other restriction.

   - `K|P|x|M4`, `…|M4_sizes`, `…|M4_n_zero`: non-null only when x = `s_cal`. Source: §10 text (M4); R1 (M4_sizes, M4_n_zero).
   - `K|P|x|M5_A`, `…|M5_A_sizes`: non-null only when category ∈ {`A`, `A-att`, `A-prime`}. Source: §10 text (M5_A); R1 (M5_A_sizes).
   - `K|P|x|M5_B`: non-null only when category = `B`. Source: §10 text.
   - `K|P|x|M2`: non-null only when category ∈ {`A`, `A-att`, `A-prime`}. Source: §10 text.
   - `K|P|x|M2_H3` (`…|M2_H4`): non-null only when the H3 (H4) reading is not `"moved"`, i.e. one of the three "kept" readings or `"no comparison changed order"`. Source: §10 text.
   - `K|P|x|premise`: non-null only when category ∈ {`A`, `A-att`, `A-prime`}. Source: §10 text.
   - `K|P|x|F_reason`: non-null only when category = `F`. Source: §10 text (step 4).
   - `PP2b|P|x|wrong_all_extrap`: non-null only when n_wrong > 0. Source: §10 text.
   - `PP2b|P|x|ok|j|j′|v`: non-null only when dec ≠ `"undecided"`. Source: §10 text.
   - `H|P|x|H1|confirmed`: non-null only when H1 is a member. Source: §6 text.
   - `H|P|x|<h>|p_holm`, `…|holm_reject`, `…|confirmed` (h ∈ H2, H3, H4); `H|P|x|holm_order`: non-null only when the gate is open (`tested`). Source: §6 text.
   - every key with P = `l2`, in every section: non-null only when `G|l2_run` is true. Source: §9 text.
   - `tr`, `dtr`, TOST and H values over a ViT set; Clopper–Pearson bounds: non-null only when the set is non-empty; n > 0. Source: §3, §6, §12 text.

5. **Keys without a restriction are written in every variant**; R1 does not touch them. Examples: `K|P|x|M1`, `M1_sizes`, `M2_x`, `M3`, `M5_extrap_sizes`, `kedge`, `kedge_which`, the readings, the `…_counted` keys, `H2_same_as_H1`, `h_consistent`, the curves, and `K|P|x|R`, `C`, `nV`. In particular `M2_x` (= `R|P|x|grid5|D1`) is written whatever the category: in §10 the A-family restriction is part of M2's own value definition, and M2_H3/M2_H4 carry their own conditions. A count that one variant's rule can never raise is 0, not `null`. Example: `PP2b|P|unc|n_tie_fc` is 0, since the uncensored rule (§10) produces no forecast ties and the totals are defined for both x. These are not new rulings. In fixture X (categories E and F) R and `forecast_pyt.py` already agreed on every one of these keys.
6. **Part B.** `partB_features_pyt.json` has no x dimension. Its one variant-restricted field is `per_pre.l2.clip_c`, which is `null` for every stem because l2 has no clipping. That already follows R1.
7. **Disagreements.** Per ruling R3 (§15.3): a further disagreement that R1 resolves is fixed by the role that owns the code; one that R1 does not resolve is reported to the lead, not improvised.

### 15.2 Ruling R2: `G|clip_equal_partB` in S0b fixture mode

> (R2) G|clip_equal_partB is a fixture-mode artefact of I (V-P1 is not evaluated in fixture mode by design): accept it as a pre-registered fixture exclusion of type 2, disclosed as POST HOC (added after fixture round 1), and it must be evaluated in the real run.

1. **Scope.** S0b test mode only (`PYT_S0B_TEST=1`, before the code-freeze line, outputs under `guard.SCRATCH`). It applies to the V-P9 stand-in fixture X (`tests_freeze/xfixture_pyt.py`, where type 2 is called "class 2") and to any comparison there with implementation I's aggregate on one side (E vs I; R-I vs I).
2. **Rule.** When I's value of `G|clip_equal_partB` is `null` in that scope, the mismatch is listed as fixture exclusion type 2 and does not fail the fixture. No other key is covered by type 2. For reference, type 1 was fixed before fixture round 1: mismatches on keys naming `noLMemb`, `LMdeep` or `LMemb_only`, because the frozen BLOOMZ E-format pair files store no summary of these three subsets.
3. **Disclosure.** Type 2 was added after fixture round 1 (run `20261010T022655Z`), so it is POST HOC. Every record that applies it, and the code-freeze line, says so.
4. **Real run.** From S6 to S8, with `PYT_S0B_TEST` unset, type 2 does not exist. E and I each write a bool `G|clip_equal_partB` (§9). V-P1 passes only if both are `true` (plan §11 V-P1), and no category is read before V-P1 passes (plan §7 checks). A `null` in the real run is a V-P1 failure, not an exclusion.
5. **How a real-run `null` is caught.** `compare_verdicts.py` counts `null` against a bool as a mismatch (§0) and knows nothing about type 2. A `null` on one side of V-P5 is therefore a mismatch. R evaluates V-P1 itself and compares `review/E` with E and `review/I` with I, so a `null` on both sides of V-P5 is still flagged by R. Only the fixture harness applies the type-2 exclusion.

### 15.3 Ruling R3, recorded for reference

> (R3) Add a synthetic H1-positive fixture variant so that categories A/A-att/A'/B and modifiers M2, M5_A, M5_B and the premise are exercised, and compare R with forecast (both E and I sides) on every variant; any further disagreement that R1 resolves is fixed by the owning role; any disagreement R1 does not resolve is reported, not improvised.

The fixture variant belongs to the fixture harness, not to this file. On the infrastructure side, the S0b test `t16_R1_domains_section_K` (`tests_infra/test_infra_pyt.py`) drives `forecast_pyt.section_K` through every category (A, A-att, A-prime, B, C, D, E, F) for both x, with hand-made consistent inputs. It checks the domain table of §15.1 for every K key.

### 15.4 Code consequences (as checked when A1 was written)

- `forecast_pyt.py` (sha256 `d4f3bd6af5d48f0f7dcf77610d42728ee5fbdf9a86f5039d3a6076994d173f38`, unchanged) already writes the §15.1 table. It writes `null` for the three M4 keys under x = `unc`, `null` outside the domain for M2, M2_H3, M2_H4, M5_A, M5_A_sizes, M5_B, premise, F_reason, `wrong_all_extrap` and `ok`, and `null` for every l2 key when l2 was not run. No change.
- `partb_pyt.py` (sha256 `f304677ab9545980bdea8eea1363c77bd1ab5dd920687d7fd8613c742c5ef4de`, unchanged): see §15.1 item 6. No change.
- R writes `0` and `""` for `K|P|unc|M4_n_zero` and `K|P|unc|M4_sizes` (code freeze report, 02:56:56Z). R's owner changes this. This file names no line of E, I or R code.
