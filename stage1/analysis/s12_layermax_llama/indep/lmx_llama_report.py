#!/usr/bin/env python
"""LLaMA axis, implementation I (role llama-axis-run-indep): REPORT.md from summary.json
and the validation files only (every number in the report is read from them)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lmx_common as L  # noqa: E402

OUT = L.OUT
S = json.load(open(os.path.join(OUT, "summary.json")))
V = json.load(open(os.path.join(OUT, "validation.json")))
VL = json.load(open(os.path.join(OUT, "validation_vl.json")))
V6 = json.load(open(os.path.join(OUT, "validation_vl6_arh.json")))
_rc = os.path.join(OUT, "validation_vl_recheck.json")
RC = json.load(open(_rc)) if os.path.exists(_rc) else None
_xc = os.path.join(OUT, "validation_cross.json")
XC = json.load(open(_xc)) if os.path.exists(_xc) else None
LL = S["llama_order"]
SHORT = {"huggyllama__llama-7b": "7B", "huggyllama__llama-13b": "13B", "huggyllama__llama-30b": "30B"}
MET = {"mknn": "mKNN", "cka_b": "CKA (biased)", "cka_d": "CKA (debiased)"}
PRI = "prh_clip+l2"


def f(x, n=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    return f"{x:+.{n}f}" if n and abs(x) < 10 else f"{x:.{n}g}"


def g(x, n=4):
    return "n/a" if x is None else f"{x:.{n}f}"


def e(x):
    return "n/a" if x is None else f"{x:.1e}"


def blk(pre, met, sub):
    return S["per_pre"][pre]["blocks"][met][sub]


lines = []
w = lines.append
pb = S["per_pre"][PRI]["partB"]
s10 = S["section10_primary"]

w("# S1.2 LLaMA axis: layer-max replication, implementation I (role `llama-axis-run-indep`)")
w("")
w("Panel: llama-7b / 13b / 30b × the 17 ARH `val` ViTs on WIT-1024 (51 pairs). Pre-processing: `prh_clip+l2` (primary) and `l2` (secondary). Calibration: K = 500 permutations, keys `S1.2-layermax|<lm>|<vit>`. Definitions, rules and predictions come from `ANALYSIS_PLAN_LLAMA.md` (Part A) and `PARTB_PREDICTIONS.md` (Part B, frozen before any cross-model score). Every number below is read from `results/S1_2_layermax_llama/indep/summary.json` or the validation files next to it.")
w("")
w("**Reading guide.**")
w("- Tr_L is the mean over the 17 ViTs of the Spearman correlation between LLaMA size (7B < 13B < 30B) and the score.")
w("- With 3 sizes, the smallest one-sided exact p is 1/6. No LLaMA-axis result can be significant: each is a directional, descriptive result on one model family, with ViT-only uncertainty.")
w("- 0.10 is the plan's materiality threshold for changes in Tr_L. It equals 4 ViTs changing by one adjacent swap.")
w("")

# ------------------------------------------------------------------ 1. headline
w("## 1. Part B verdicts (primary: prh_clip+l2, layer-max)")
w("")
w("| ID | what was predicted | observed | outcome |")
w("|---|---|---|---|")
p = pb["P-L1"]
w(f"| P-L1 | mKNN: \\|ΔTr_L(s_cal)\\| < 0.10 and \\|ΔTr_L(unc)\\| < 0.10 | Tr_L raw {f(p['Tr_raw'],2)} → s_cal {f(p['Tr_scal'],2)} / unc {f(p['Tr_unc'],2)}; ΔTr {f(p['dTr_scal'],3)} / {f(p['dTr_unc'],3)} | **{p['outcome']}** |")
p = pb["P-L2"]
w(f"| P-L2 | biased CKA (material, s = +1): ΔTr_L(s_cal) ≤ −0.10 \"confirmed\" | Tr_L raw {f(p['Tr_raw'],2)} → s_cal {f(p['Tr_scal'],2)}; ΔTr {f(p['dTr_scal'],3)} (exact p one-sided decrease {g(p['dTr_scal_p_one_decrease'],3)}) | **{p['outcome_scal']}** |")
w(f"| P-L2 (unc, secondary) | same rule on uncensored | Tr_L unc {f(p['Tr_unc'],2)}; ΔTr {f(p['dTr_unc'],3)} | {p['outcome_unc_secondary']} |")
p = pb["P-L2a"]
w(f"| P-L2a | premise: median \\|c_α − B̄\\| ≤ 0.01 and ≥ {p['n_needed']}/{p['n_qualifying']} comparisons with c_α(larger) > c_α(smaller) | median \\|c_α − B̄\\| {g(p['median_abs_calpha_minus_Bbar'])}; order agrees {p['n_agree']}/{p['n_qualifying']} | **{p['outcome']}** |")
for r in ("r0.5", "r1.0"):
    p = pb["P-L3"][r]
    w(f"| P-L3 ({r}) | biased CKA fixed depth: keeps sign | Tr_L raw {f(p['Tr_raw'],2)} → s_cal {f(p['Tr_scal'],2)} | **{p['outcome']}** |")
p = pb["P-L4"]
w(f"| P-L4 | debiased CKA: ΔTr_L(s_cal), ΔTr_L(unc) ≥ −0.10; pooled r_after ≥ r_before − 0.05 | ΔTr {f(p['dTr_scal'],3)} / {f(p['dTr_unc'],3)}; pooled r {f(p['pooled10_r_before'],3)} → {f(p['pooled10_r_after'],3)} | **{p['outcome']}** |")
p = pb["P-L5"]
w(f"| P-L5 | calibrate-then-max (secondary): keeps sign | Tr_L raw {f(p['Tr_raw_layermax'],2)} → Tr_L(M) {f(p['Tr_M'],2)} | **{p['outcome']}** |")
w("")
w("**Plan §10 (primary pre only).**")
for k, v in s10["items"].items():
    w(f"- {k.replace('_', ' ')}: **{'YES' if v else 'no'}**")
w(f"- Deductive checks, where a violation would mean an implementation error: P-L1 {s10['implementation_error_flags']['P-L1_deductive_violations']} violations, P-L2a {s10['implementation_error_flags']['P-L2a_deductive_violations']} violations.")
_rev = [k for k, v in pb["P-L3"].items() if v.get("outcome") == "reversed"] + (["P-L5"] if pb["P-L5"]["outcome"] == "reversed" else [])
if _rev:
    w(f"- Item 4 is triggered by: {', '.join(('P-L3 ' + k) if k.startswith('r') else k for k in _rev)}. Under plan §10 this counts against the remedy claim (fixed layers, or calibration before the max, avoid the interaction), not against Corollary 2 itself, which allows a reversal whenever the baselines of the entries that set the score rise faster than the score.")
if s10["any_against_corollary_use"]:
    w("- Items 1 or 2 are triggered: this counts against using Proposition 5 + Corollary 2 as a predictor of calibration effects from features.")
else:
    w("- Items 1 and 2 are not triggered: nothing here counts against using Proposition 5 + Corollary 2 as a predictor of calibration effects from features.")
w("")

# ------------------------------------------------------------------ 2. validation
w("## 2. Validation (plan §4)")
w("")
w("| item | check | result |")
w("|---|---|---|")
v1 = VL["V-L1"]
w(f"| V-L1 | clip thresholds recomputed in the run = reference, bitwise (S1.2 cache for llama-7b and the ViTs; frozen `clip_thresholds_partB.json` for llama-13b/30b) | {v1['n_models']} models, all bitwise equal: {f(v1['all_bitwise_equal'])} → **{'PASS' if v1['pass'] else 'FAIL'}** |")
v2 = VL["V-L2"]["summary"]
w(f"| V-L2 | kNN tables = `hrhv2.structural.knn_table` (I uses it by construction); ViT and llama-7b tables = frozen layer-max caches; ViT boundary ties = 0; observed count matrices = overlap of these tables | frozen tables identical {f(v2['frozen_tables_all_identical'])} ({v2['n_frozen_compared']} model×pre); ViT exact boundary-tie rows {v2['vit_total_exact_boundary_tie_rows']}; observed counts identical on all layer pairs {f(v2['obs_counts_all_identical'])} → **{'PASS' if VL['V-L2']['pass'] else 'FAIL'}** |")
i2 = V["I2"]
w(f"| V-L3 | 12 fixed layer pairs × 4 perms × 2 pre vs hrhv2 from scratch | {i2['n_checks']} checks; mKNN all equal {f(i2['mknn_all_equal'])}; max \\|ΔCKA\\| biased {e(i2['max_cka_b_absdiff'])}, debiased {e(i2['max_cka_d_absdiff'])} → **{'PASS' if i2['pass'] else 'FAIL'}** |")
i3 = V["I3"]
w(f"| V-L4a | llama-7b fixed-depth entries (17 pairs × 2 depths × 2 pre) = S1.2 panel s_obs | {i3['n_configs']} configs; mKNN equal {f(i3['mknn_all_equal'])}; max \\|ΔCKA\\| {e(i3['max_cka_b_absdiff'])} / {e(i3['max_cka_d_absdiff'])} → **{'PASS' if i3['pass'] else 'FAIL'}** |")
v4 = VL["V-L4B"]
exc = sorted({x for pre in L.PRES for x in v4[pre]["strict_relative_1e-12_exceeded_by"]})
w(f"| V-L4b | llama-7b layer-max outputs = frozen `results/S1_2_layermax/indep` and `.../pairs` (early); ≤ 1e-12 relative, p and null counts identical | p, argmax and mKNN null counts identical; raw matrices max rel {e(max(v4[pre]['raw_vs_frozen_indep']['max_rel'][k] for pre in L.PRES for k in v4[pre]['raw_vs_frozen_indep']['max_rel']))}. Strict 1e-12 relative exceeded only by: {', '.join(exc) if exc else 'none'} (see note). Under the V-L5 mixed tolerance: **{'PASS' if v4['pass_mixed_tolerance_V-L5_form'] else 'FAIL'}**; strict rule: **{'PASS' if v4['pass'] else 'FAIL (1 near-zero field)'}** |")
v7 = VL["V-L7"]
w(f"| V-L7 | Proposition 5: entry-wise null means vs √(PR_u PR_v)/(m−1) (biased CKA), 0 (debiased), 10/1023 (mKNN) on the 34 new pairs | primary: median rel. err. {g(v7[PRI]['new_34']['cka_b']['median_rel_err'])}, \\|z\\|>4 fractions {g(v7[PRI]['new_34']['cka_b']['frac_abs_z_gt_4'])} / {g(v7[PRI]['new_34']['cka_d']['frac_abs_z_gt_4'])} / {g(v7[PRI]['new_34']['mknn']['frac_abs_z_gt_4'])}; l2: median {g(v7['l2']['new_34']['cka_b']['median_rel_err'])} → **{'PASS' if v7['pass'] else 'FAIL'}** |")
w(f"| V-L6 | vendored ARH `calibrate` / `calibrate_layers(agg=\"max\")`, our π injected; tolerance 1e-6 | max \\|Δ\\| over s_cal, p, τ: {e(V6['max_absdiff'])}; observed-matrix diffs > 1e-6: mKNN {V6['obsmatrix_mknn_llama30b_grid9']['n_diff_gt_1e-6']}/{V6['obsmatrix_mknn_llama30b_grid9']['n_entries']}, CKA {V6['obsmatrix_cka_b_llama13b_full']['n_diff_gt_1e-6']}/{V6['obsmatrix_cka_b_llama13b_full']['n_entries']} → **{'PASS' if V6['pass_1e-6'] else 'FAIL'}** |")
sc = S["self_checks"]
if RC is not None:
    w(f"| V-L1/2/4b/7 recheck | the four entries of `validation_vl.json` recomputed with the current `vl_checks.py` (sha256 {RC['vl_checks_sha256'][:8]}…), nothing overwritten | identical to stored: {', '.join(k + ' ' + f(v['identical_to_stored']) for k, v in RC['items'].items())} → **{'PASS' if RC['all_identical'] else 'FAIL'}** |")
w(f"| self-check | BLOOMZ Tr and 8-LM ARH r recomputed by the new aggregation = frozen 8-LM indep summary | max \\|Δ\\| Tr {e(max(sc[p]['max_abs_diff_Tr_bloomz'] for p in L.PRES))}, r {e(max(sc[p]['max_abs_diff_r8'] for p in L.PRES))} → **{'PASS' if all(sc[p]['pass'] for p in L.PRES) else 'FAIL'}** |")
w("")
wh = v4[PRI]["summary_vs_frozen_early"]["where_max_rel"].get("cka_d.null_mean", {})
w(f"**Note on V-L4b.** The only field above 1e-12 relative is the debiased-CKA null mean against the early implementation's frozen outputs. Its worst case is {wh.get('where', 'n/a')}: \\|value\\| {e(wh.get('a'))}, absolute difference {e(wh.get('absdiff'))}. The exact expectation of this quantity is 0 (plan §8; V-L7 confirms within MC error), so a relative tolerance is ill-conditioned there. The absolute difference is round-off, not an implementation difference. Against the frozen independent run, the same field agrees to {e(v4[PRI]['summary_vs_frozen_indep']['max_rel'].get('cka_d.null_mean'))} relative. The first, strict-only output is kept as `validation_vl.attempt1_vl4b_strict_only.json`.")
w("")

# ------------------------------------------------------------------ 3. llama axis trends
for pre in L.PRES:
    w(f"## {'3' if pre == PRI else '6'}. LLaMA-axis trends, {pre}{' (primary)' if pre == PRI else ' (secondary)'}")
    w("")
    w("Tr_L [exact one-sided p over 6 relabellings]; ΔTr_L = Tr_L(score) − Tr_L(raw T) for the same protocol.")
    w("")
    w("| metric | protocol | Tr_L raw T | Tr_L s_cal | Tr_L uncensored | ΔTr s_cal | ΔTr unc | Tr_L c_α | Tr_L E0[max] |")
    w("|---|---|---|---|---|---|---|---|---|")
    for met in ("mknn", "cka_b", "cka_d"):
        for sub, lab in (("full", "layer-max"), ("fixed_r0.5", "fixed r=0.5"), ("fixed_r1.0", "fixed r=1.0"), ("ctm", "calibrate-then-max")):
            b = blk(pre, met, sub)
            la = b["llama_axis"]
            ca = f(la["c_alpha"]["Tr"], 2) if "c_alpha" in la else "–"
            nm = f(la["null_mean"]["Tr"], 2) if "null_mean" in la else "–"
            w(f"| {MET[met]} | {lab} | {f(la['s_obs']['Tr'],2)} [{g(la['s_obs']['p_one'],3)}] | {f(la['s_cal']['Tr'],2)} [{g(la['s_cal']['p_one'],3)}] | "
              f"{f(la['uncensored']['Tr'],2)} [{g(la['uncensored']['p_one'],3)}] | {f(b['delta']['s_cal']['dTr'],3)} | {f(b['delta']['uncensored']['dTr'],3)} | {ca} | {nm} |")
    w("")
    w("All score trends use 17/17 ViTs. Only mKNN c_α rows can be constant across the three sizes (c_α is a count / 10240); such ViTs are excluded from that Tr_L and counted in `n_vit_constant` in summary.json.")
    w("")
    w("Means over the 17 ViTs (layer-max):")
    w("")
    w("| metric | LM | T | c_α | s_cal | uncensored | E0[max] |")
    w("|---|---|---|---|---|---|---|")
    for met in ("mknn", "cka_b", "cka_d"):
        b = blk(pre, met, "full")["mean_by_lm"]
        for lm in LL:
            w(f"| {MET[met]} | {SHORT[lm]} | {g(b['s_obs'][lm])} | {g(b['c_alpha'][lm])} | {g(b['s_cal'][lm])} | {g(b['uncensored'][lm])} | {g(b['null_mean'][lm])} |")
    w("")
    w("Layer subsets, biased CKA (A2; grid17 equalises the LM side to 17 layers; noLMemb / LMdeep / LMemb_only are the pre-specified descriptive subsets):")
    w("")
    w("| subset | Tr_L raw | Tr_L s_cal | Tr_L unc | Tr_L c_α | mean c_α 7B / 13B / 30B |")
    w("|---|---|---|---|---|---|")
    for sub in ("full", "grid2", "grid3", "grid5", "grid9", "grid17", "lmall_x_vitlast", "lmlast_x_vitall", "noLMemb", "LMdeep", "LMemb_only"):
        b = blk(pre, "cka_b", sub)
        la = b["llama_axis"]
        mc = " / ".join(g(b["mean_by_lm"]["c_alpha"][lm]) for lm in LL)
        w(f"| {sub} | {f(la['s_obs']['Tr'],2)} | {f(la['s_cal']['Tr'],2)} | {f(la['uncensored']['Tr'],2)} | {f(la['c_alpha']['Tr'],2)} | {mc} |")
    w("")
    R = S["per_pre"][pre]["readings"]
    w("Readings (plan §7):")
    w("")
    w("| metric | ARH-style, pooled 10 LMs: r before → after (CKA) vs (mKNN) | verdict | 8-LM reference verdict | LLaMA-axis reading |")
    w("|---|---|---|---|---|")
    for cka in ("cka_b", "cka_d"):
        r = R[cka]
        i = r["inputs"]
        w(f"| {MET[cka]} ({r['label']}) | {f(i['r_before_cka'],2)} → {f(i['r_after_cka'],2)} vs {f(i['r_before_mknn'],2)} → {f(i['r_after_mknn'],2)} | "
          f"{r['arh_pooled10']['verdict']} (ΔΔ {f(r['arh_pooled10']['dDelta'],2)}) | {r['arh_pooled8_reference']['verdict']} | {r['llama_axis']} |")
    w("")
    w("Pooled 10-LM panel, ARH-style Pearson r with LLM rank (layer-max) [exact one-sided p over 10! relabellings; ViT-bootstrap 95% CI]:")
    w("")
    w("| metric | score | r (ARH rank) | p_one | CI | r (dense rank) | r (log params) | r on 8-LM panel |")
    w("|---|---|---|---|---|---|---|---|")
    for met in ("mknn", "cka_b", "cka_d"):
        b = blk(pre, met, "full")
        for fl, lab in (("s_obs", "raw T"), ("s_cal", "s_cal"), ("uncensored", "uncensored")):
            pr = b["pooled_r"][fl]
            ci = pr["arh_rank"].get("ci")
            w(f"| {MET[met]} | {lab} | {f(pr['arh_rank']['r'],3)} | {e(pr['arh_rank']['p_one'])} | "
              f"{'[' + f(ci[0],2) + ', ' + f(ci[1],2) + ']' if ci else '–'} | {f(pr['dense_rank']['r'],3)} | {f(pr['log_params']['r'],3)} | {f(b['pooled_r_8lm'][fl]['r'],3)} |")
    w("")
    w("Joint within-family trend Tr_BL = (Tr_BLOOMZ + Tr_L)/2, exact p over 720 relabellings (descriptive; the BLOOMZ half was already known):")
    w("")
    w("| metric | score | Tr_BLOOMZ | Tr_L | Tr_BL | p_one |")
    w("|---|---|---|---|---|---|")
    for met in ("mknn", "cka_b", "cka_d"):
        for fl in ("s_obs", "s_cal", "uncensored"):
            j = blk(pre, met, "full")["joint_BL"][fl]
            w(f"| {MET[met]} | {fl} | {f(j['Tr_BLOOMZ'],2)} | {f(j['Tr_L'],2)} | {f(j['Tr_BL'],2)} | {g(j['p_one'],4)} |")
    w("")
    w("Vision axis restricted to the LLaMA rows (descriptive), layer-max:")
    w("")
    for met in ("mknn", "cka_b", "cka_d"):
        va = blk(pre, met, "full")["vision_axis_llama"]
        w(f"- {MET[met]}: Tr_vision raw {f(va['s_obs']['Tr'],2)} (p_one {g(va['s_obs']['p_one'],3)}), s_cal {f(va['s_cal']['Tr'],2)}, unc {f(va['uncensored']['Tr'],2)}. OpenLLaMA 7B > 3B fraction (raw): {g(blk(pre, met, 'full')['openllama_7b_gt_3b_frac']['s_obs'],2)}")
    w("")
    D = S["per_pre"][pre]["diagnostics"]
    w("Attribution diagnostics (plan §6), biased CKA:")
    w("")
    sel = D["cka_b"]["selection"]
    w(f"- **A1, selection.** Mean relative depth of the selected LM layer: " + ", ".join(f"{SHORT[lm]} {g(sel['rel_a_by_lm'][lm],2)}" for lm in LL)
      + f". Mean relative depth of the selected ViT layer: " + ", ".join(f"{SHORT[lm]} {g(sel['rel_b_by_lm'][lm],2)}" for lm in LL)
      + f". Extra tied maxima: {sel['n_tied_max_extra']}. CKA and mKNN select the same layer pair in {g(sel.get('sel_agree_cka_b_mknn_frac_llama'),2)} of pairs. Mean relative LM depth of the null maxima: "
      + ", ".join(f"{SHORT[lm]} {g(sel['null_argmax_rel_a_by_lm'][lm],2)}" for lm in LL) + ".")
    prof = D["cka_b"]["null_profile"]
    w(f"- **Highest-null LM layer.** The argmax over LM layers of the mean over ViTs of max_v E0[S_uv] is: " + ", ".join(
        f"{SHORT[lm]} layer {prof[lm]['argmax_layer']} (rel {g(prof[lm]['argmax_rel'],3)}, level {g(prof[lm]['max'])})" for lm in LL) + ".")
    infl = D["cka_b"]["inflation"]
    w(f"- **A2, inflation.** Spearman(c_α, number of layer pairs) is {f(infl['spearman_calpha_vs_nlayerpairs_llama51'],2)} on the 51 LLaMA pairs and {f(infl['spearman_calpha_vs_nlayerpairs_pooled170'],2)} on the 170 pooled pairs. Mean inflation c_α − c_α^entry(a*, b*): "
      + ", ".join(f"{SHORT[lm]} {g(infl['mean_inflation_by_llama'][lm])}" for lm in LL) + ".")
    cf = D["cka_b"]["counterfactual"]
    w("- **A3, counterfactual decomposition.** Tr_L and pooled-10 r (ARH rank), with the mean by LLaMA size:")
    w("")
    w("| quantity | Tr_L | pooled r | 7B | 13B | 30B |")
    w("|---|---|---|---|---|---|")
    for k, v in cf.items():
        w(f"| {k} | {f(v['Tr_L'],2)} | {f(v['pooled10_arh_r'],2)} | " + " | ".join(g(v['mean_by_llama'][lm]) for lm in LL) + " |")
    w("")
    pc = S["per_pre"][pre]["partB"]["plug_in_check"]
    w(f"- **Plug-in check.** Tr_L(T − B̄_pred) = {f(pc['Tr_T_minus_Bbar'],2)} against Tr_L(uncensored) = {f(pc['Tr_unc'],2)}; \\|diff\\| {g(pc['abs_diff'],3)}, within 0.10: {f(pc['within_0.10'])}.")
    w("")
    if pre == PRI:
        w("## 4. Part B details (primary)")
        w("")
        p2a = pb["P-L2a"]
        w(f"- **Threshold vs predicted max baseline.**")
        w(f"  - Mean c_α by size: " + " / ".join(g(p2a['mean_calpha_by_lm'][lm]) for lm in LL) + "; mean B̄: " + " / ".join(g(p2a['mean_Bbar_by_lm'][lm]) for lm in LL) + ".")
        w(f"  - c_α − B̄ ranges over [{g(p2a['calpha_minus_Bbar']['min'])}, {g(p2a['calpha_minus_Bbar']['max'])}] (mean {g(p2a['calpha_minus_Bbar']['mean'])}); E0[max] − B̄ mean {e(p2a['E0max_minus_Bbar']['mean'])}; Pearson(c_α, B̄) {g(p2a['pearson_calpha_Bbar'],4)}.")
        w(f"  - Qualifying comparisons per size pair: {p2a['n_qualifying_by_size_pair']}. They match the frozen Part B counts: {f(p2a['n_qualifying_matches_frozen'])}.")
        dc = p2a["deductive_check"]
        w(f"- **P-L2a deductive check.** c_α increases with size for {dc['n_vit_calpha_increasing']} ViTs and decreases for {dc['n_vit_calpha_decreasing']}. Violations: {dc['n_violations']}. Uncensored analogue (E0[max] order): {dc['uncensored_analogue']['n_violations']} violations.")
        pf = pb["P-L2"]["pairwise_flips_qualifying"]
        for sname in ("s_cal", "uncensored"):
            x = pf[sname]
            w(f"- **Pairwise flips under {sname}** (qualifying comparisons, n = {x['n']}):")
            w(f"  - concordant → discordant (the predicted direction): {x['conc_to_disc']};")
            w(f"  - discordant → concordant (against): {x['disc_to_conc']};")
            w(f"  - concordant → tie: {x['conc_to_tie']}; unchanged: {x['unchanged']}.")
        d1 = pb["P-L1"]["deductive_check"]
        w(f"- **P-L1 deductive check.**")
        if d1['s_cal']['n_eligible'] == 0 and d1['uncensored']['n_eligible'] == 0:
            w("  - The check is vacuous here: no (ViT, size pair) has a raw mKNN gap larger than ζ_j + ζ_j′, so Corollary 2 guarantees no order and none is tested.")
        w(f"  - s_cal: {d1['s_cal']['n_violations']} violations among {d1['s_cal']['n_eligible']} eligible comparisons;")
        w(f"  - uncensored: {d1['uncensored']['n_violations']} violations among {d1['uncensored']['n_eligible']};")
        w(f"  - mean ζ = \\|c_α − 10/1023\\| is {g(pb['P-L1']['zeta_mean'])} (max {g(pb['P-L1']['zeta_max'])}).")
        for r in ("r0.5", "r1.0"):
            x = pb["P-L3"][r]
            w(f"- **P-L3 {r}.** Tr_L(c_α) {f(x['Tr_calpha'],2)}. The Part B B_fd trend is {f(x['Bfd_Tr_partB'],2)} (material: {f(x['Bfd_material_partB'])}). Uncensored, descriptive: {x.get('unc_outcome_descriptive', 'n/a')}.")
        w(f"- **P-L5.** Tr_L of the uncensored analogue: {f(pb['P-L5']['Tr_M_unc'],2)} ({pb['P-L5'].get('unc_outcome_descriptive', 'n/a')}).")
        w("")
        w("## 5. l2 Part B outcomes (secondary; failures confined to l2 are reported, not counted)")
        w("")
        q = S["per_pre"]["l2"]["partB"]
        w(f"- **P-L1:** {q['P-L1']['outcome']} (ΔTr {f(q['P-L1']['dTr_scal'],3)} / {f(q['P-L1']['dTr_unc'],3)}).")
        w(f"- **P-L2:** {q['P-L2']['outcome_scal']}. Tr_L raw {f(q['P-L2']['Tr_raw'],2)} → s_cal {f(q['P-L2']['Tr_scal'],2)} (ΔTr {f(q['P-L2']['dTr_scal'],3)}).")
        pf = q["P-L2"]["pairwise_flips_qualifying"]
        w(f"  - Pairwise flips on the {pf['s_cal']['n']} qualifying comparisons:")
        w(f"    - s_cal: {pf['s_cal']['conc_to_disc']} in the predicted direction, {pf['s_cal']['disc_to_conc']} against;")
        w(f"    - uncensored: {pf['uncensored']['conc_to_disc']} in the predicted direction, {pf['uncensored']['disc_to_conc']} against.")
        w(f"- **P-L2a:** {q['P-L2a']['outcome']}. Median \\|c_α − B̄\\| is {g(q['P-L2a']['median_abs_calpha_minus_Bbar'])}; order agrees {q['P-L2a']['n_agree']}/{q['P-L2a']['n_qualifying']} (needed {q['P-L2a']['n_needed']}).")
        w(f"- **P-L3:** r0.5 {q['P-L3']['r0.5']['outcome']} (Tr {f(q['P-L3']['r0.5']['Tr_raw'],2)} → {f(q['P-L3']['r0.5']['Tr_scal'],2)}); r1.0 {q['P-L3']['r1.0']['outcome']} (Tr {f(q['P-L3']['r1.0']['Tr_raw'],2)} → {f(q['P-L3']['r1.0']['Tr_scal'],2)}).")
        w(f"- **P-L4:** {q['P-L4']['outcome']}. ΔTr {f(q['P-L4']['dTr_scal'],3)} / {f(q['P-L4']['dTr_unc'],3)}; pooled r {f(q['P-L4']['pooled10_r_before'],3)} → {f(q['P-L4']['pooled10_r_after'],3)}.")
        w(f"- **P-L5:** {q['P-L5']['outcome']} (Tr_L(M) {f(q['P-L5']['Tr_M'],2)}).")
        w("")

w("## 7. Limits")
w("")
w("- **Small family.** There are only 3 LLaMA sizes, so the smallest one-sided p is 1/6. Every Tr_L carries ViT-sampling uncertainty only; the uncertainty from having 3 LMs is not quantified.")
w("- **Size and layer count are confounded.** L_A is 33 / 41 / 61. grid17 equalises the LM side.")
w("- **Shared LM features.** The 17 ViT rows share the same three LM feature sets, so they are not independent replicates.")
w("- **Pooled 10-LM panel.** It reuses the frozen 8-LM independent run for the 7 non-LLaMA LMs, with the same definitions and keys, and drops nothing.")
w("- **Scope.** Every number above is implementation I. Its comparison with the early implementation E (V-L5) is in `validation_cross.json`.")
if XC is not None:
    w("")
    w("## 7b. Cross-implementation comparison with E (V-L5; written after this report's numbers were final)")
    w("")
    for line in XC.get("report_lines", []):
        w(line)
w("")
w("## 8. Files")
w("")
w(f"- Summary: `results/S1_2_layermax_llama/indep/summary.json`. Code sha256 values are listed in `provenance.code_sha256`.")
w("- Validation: `validation.json` (V-L3, V-L4a, ties), `validation_vl.json` (V-L1, V-L2, V-L4b, V-L7), `validation_vl6_arh.json` (V-L6), `validation_vl_recheck.json` (provenance recheck), `validation_cross.json` (I vs E; written after this report's numbers were final).")
w("- `REPORT.pre_cross.md` is this report as it stood before the comparison with E; sections 1–7 are unchanged since.")
w("- Raw matrices: `<pre>/pairs/<lm>__<vit>.npz`, holding the observed and 500 null L_A × L_B matrices.")
open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(lines) + "\n")
print("wrote", os.path.join(OUT, "REPORT.md"), len(lines), "lines")
