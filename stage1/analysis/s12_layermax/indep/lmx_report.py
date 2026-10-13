#!/usr/bin/env python
"""Tables for the independent layer-max report, printed from the summary JSON
(results/S1_2_layermax/indep/layermax_indep_summary.json). Interpretation
prose is written by hand in the report file; every number comes from here."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lmx_common as L  # noqa: E402

S = json.load(open(os.path.join(L.OUT, "layermax_indep_summary.json")))
SHORT = {"bigscience__bloomz-560m": "bloomz-560m", "bigscience__bloomz-1b1": "bloomz-1b1",
         "bigscience__bloomz-1b7": "bloomz-1b7", "bigscience__bloomz-3b": "bloomz-3b",
         "bigscience__bloomz-7b1": "bloomz-7b1", "openlm-research__open_llama_3b": "open_llama_3b",
         "openlm-research__open_llama_7b": "open_llama_7b", "huggyllama__llama-7b": "llama-7b"}
MN = {"mknn": "mKNN(10)", "cka_b": "CKA biased", "cka_d": "CKA debiased (expl.)"}


def f(x, n=2):
    if x is None:
        return "n/a"
    return f"{x:.{n}f}"


def rcell(t, ci=True):
    a = t["arh"]["arh_rank"]
    if a.get("r") is None:
        return "n/a"
    s = f"{a['r']:.2f}"
    if ci and "ci" in a:
        s += f" [{a['ci'][0]:.2f}, {a['ci'][1]:.2f}]"
    return s + f" (p={a['p_one']:.3f})"


def trcell(t):
    s = t["s12"]
    return f"{s['Tr']:.2f} (p={s['p_one']:.3f})"


out = []
P = S["per_pre"]
for pre in L.PRES:
    R = P[pre]
    T = R["trends"]
    out.append(f"\n### pre = {pre}\n")
    out.append("ARH-style Pearson r (score vs ARH LLM rank, 136 pairs) [ViT-bootstrap 95% CI] (exact one-sided p, 8! relabellings)\n")
    out.append("| metric | aggregate | before (raw T) | after (s_cal, ARH) | uncensored | Δ = before − after | frac s_cal > 0 |")
    out.append("|---|---|---|---|---|---|---|")
    for met in ("mknn", "cka_b", "cka_d"):
        for sub, lab in (("full", "layer-max"), ("fixed_r0.5", "fixed r=0.5"), ("fixed_r1.0", "fixed r=1.0")):
            t = T[met][sub]
            rb, ra = t["before_raw"]["arh"]["arh_rank"]["r"], t["after_scal"]["arh"]["arh_rank"]["r"]
            d = (rb - ra) if (rb is not None and ra is not None) else None
            out.append(f"| {MN[met]} | {lab} | {rcell(t['before_raw'])} | {rcell(t['after_scal'])} | {rcell(t['uncensored'])} | {f(d)} | {f(t['frac_s_cal_pos'])} |")
    out.append("\nS1.2-style Tr (mean over 17 ViTs of within-BLOOMZ Spearman; exact one-sided p over 120 relabellings)\n")
    out.append("| metric | aggregate | Tr uncensored [CI] | Tr s_cal | Tr raw T | vision-axis Tr unc (p) | OpenLLaMA 7B>3B (unc) |")
    out.append("|---|---|---|---|---|---|---|")
    for met in ("mknn", "cka_b", "cka_d"):
        for sub, lab in (("full", "layer-max"), ("fixed_r0.5", "fixed r=0.5"), ("fixed_r1.0", "fixed r=1.0")):
            t = T[met][sub]
            u = t["uncensored"]["s12"]
            out.append(f"| {MN[met]} | {lab} | {u['Tr']:.2f} [{u['ci'][0]:.2f}, {u['ci'][1]:.2f}] (p={u['p_one']:.3f}) | {trcell(t['after_scal'])} | {trcell(t['before_raw'])} | "
                       f"{t['uncensored']['vision']['Tr']:.2f} ({t['uncensored']['vision']['p_one']:.3f}) | {f(u['openllama_frac_larger_higher'])} |")
    fx = R["s12_stored_fixed_depth"]
    out.append("\nCross-check: same statistics on S1.2's stored fixed-depth values (S1.2 permutations)\n")
    out.append("| metric | r | ARH r before | ARH r after | Tr unc | Tr s_cal |")
    out.append("|---|---|---|---|---|---|")
    for k, t in fx.items():
        met, r = k.split("|")
        out.append(f"| {MN[met]} | {r} | {f(t['before_raw']['arh']['arh_rank']['r'])} | {f(t['after_scal']['arh']['arh_rank']['r'])} | "
                   f"{f(t['uncensored']['s12']['Tr'])} | {f(t['after_scal']['s12']['Tr'])} |")
    out.append("\nReadings (PLAN §6):\n")
    for cka, rd in R["readings"].items():
        out.append(f"- {MN[cka]}: layer-max ARH reading = **{rd['arh']['verdict']}** (ΔΔ = {rd['arh']['dDelta']:.2f}, r_after gap mKNN − CKA = {rd['arh']['r_after_gap']:.2f}); "
                   f"S1.2 axis = **{rd['s12_axis']}**; fixed r=0.5: {rd['arh_fixed_r0.5']['verdict']} / {rd['s12_axis_fixed_r0.5']}; "
                   f"fixed r=1.0: {rd['arh_fixed_r1.0']['verdict']} / {rd['s12_axis_fixed_r1.0']}")
    # selection
    out.append("\nSelection: mean relative depth of the observed argmax over the 17 ViTs (LM depth a*/(L_A−1), ViT depth b*/(L_B−1)); modal LM layer index\n")
    out.append("| LM | L_A | ARH rank | mKNN LM depth | mKNN ViT depth | mKNN modal a* | CKA LM depth | CKA ViT depth | CKA modal a* | CKA layer-max T | CKA c_α | CKA s_cal | CKA unc | mKNN s_cal |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    pairs = R["pairs"]
    for lm in L.LM_STEMS:
        LA = pairs[f"{lm}|{L.VIT_STEMS[0]}"]["LA"]
        sk, sc = T["mknn"]["selection"], T["cka_b"]["selection"]
        ic = T["cka_b"]["inflation"]
        ik = T["mknn"]["inflation"]
        out.append(f"| {SHORT[lm]} | {LA} | {L.ARH_RANK[lm]} | {sk['rel_a_by_lm'][lm]:.2f} | {sk['rel_b_by_lm'][lm]:.2f} | {sk['a_by_lm_mode'][lm]} | "
                   f"{sc['rel_a_by_lm'][lm]:.2f} | {sc['rel_b_by_lm'][lm]:.2f} | {sc['a_by_lm_mode'][lm]} | {ic['T_by_lm'][lm]:.3f} | {ic['calpha_by_lm'][lm]:.3f} | "
                   f"{ic['s_cal_by_lm'][lm]:.3f} | {ic['unc_by_lm'][lm]:.3f} | {ik['s_cal_by_lm'][lm]:.3f} |")
    for met in ("mknn", "cka_b", "cka_d"):
        s = T[met]["selection"]
        out.append(f"- {MN[met]}: selected LM depth vs ARH rank r = {f(s['rel_a_trend']['arh_r'].get('r'))} (p={f(s['rel_a_trend']['arh_r'].get('p_one'), 3)}), "
                   f"within-BLOOMZ Tr = {f(s['rel_a_trend']['s12']['Tr'])}; selected ViT depth vs rank r = {f(s['rel_b_trend']['arh_r'].get('r'))}; "
                   f"extra tied maxima = {s['n_tied_max_total']}; null argmax mean rel depth (LM, ViT) = ({s['null_argmax_rel_a_mean']:.2f}, {s['null_argmax_rel_b_mean']:.2f}); "
                   f"null argmax = observed argmax in {s['null_argmax_same_as_obs_mean']:.3f} of permutations")
    out.append("\nNull inflation (layer-max c_α vs single-entry c_α at the selected pair)\n")
    out.append("| metric | mean c_α (max) | mean c_α (entry) | mean inflation | Spearman(c_α, L_A·L_B) | Spearman(mean T_k, L_A·L_B) | Spearman(inflation, L_A·L_B) | ARH r of c_α | Tr of c_α |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for met in ("mknn", "cka_b", "cka_d"):
        i = T[met]["inflation"]
        ca = T[met]["full"]["c_alpha"]
        out.append(f"| {MN[met]} | {T[met]['full']['mean_c_alpha']:.4f} | {i['mean_c_entry']:.4f} | {i['mean_inflation_calpha_minus_c_entry']:.4f} | "
                   f"{i['spearman_calpha_vs_nlayerpairs']:.2f} | {i['spearman_meanTk_vs_nlayerpairs']:.2f} | {i['spearman_inflation_vs_nlayerpairs']:.2f} | "
                   f"{f(ca['arh']['arh_rank']['r'])} | {f(ca['s12']['Tr'])} |")
    out.append("\nCounterfactual decomposition (PLAN §7 A3): ARH-style r vs rank, and S1.2 Tr\n")
    out.append("| metric | " + " | ".join(["1 T", "2 T−c_α", "3 s_cal", "3b uncapped", "4 T−c_entry", "5 T−null_entry", "6 unc", "c_α", "infl."]) + " |")
    out.append("|---|" + "---|" * 9)
    keys = ["1_T_raw", "2_T_minus_calpha", "3_s_cal_ARH", "3b_uncapped_ratio", "4_T_minus_c_entry", "5_T_minus_entry_null_mean",
            "6_uncensored_T_minus_mean_Tk", "c_alpha", "inflation_calpha_minus_c_entry"]
    for met in ("mknn", "cka_b", "cka_d"):
        cf = T[met]["counterfactual"]
        out.append(f"| {MN[met]} r | " + " | ".join(f(cf[k]["arh_r"].get("r")) for k in keys) + " |")
        out.append(f"| {MN[met]} Tr | " + " | ".join(f(cf[k]["s12"]["Tr"]) for k in keys) + " |")
    out.append("\nNested layer grids (PLAN §7 A2): ARH r before / after, S1.2 Tr (uncensored), mean c_α, mean entries per pair\n")
    out.append("| metric | subset | entries | r before | r after | Tr unc | Tr s_cal | mean c_α | mean T |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for met in ("mknn", "cka_b", "cka_d"):
        for sub in ("fixed_r0.5", "fixed_r1.0", "grid2", "grid3", "grid5", "grid9", "grid17", "lmall_x_vitlast", "lmlast_x_vitall", "full"):
            t = T[met][sub]
            out.append(f"| {MN[met]} | {sub} | {t['mean_entries']:.0f} | {f(t['before_raw']['arh']['arh_rank']['r'])} | {f(t['after_scal']['arh']['arh_rank']['r'])} | "
                       f"{f(t['uncensored']['s12']['Tr'])} | {f(t['after_scal']['s12']['Tr'])} | {t['mean_c_alpha']:.4f} | {t['mean_s_obs']:.4f} |")
    out.append("\nSensitivity of the ARH-style r to the LLM covariate (layer-max): before / after / uncensored\n")
    out.append("| metric | ARH rank (1..7, 9) | dense rank (1..8) | log num_params |")
    out.append("|---|---|---|---|")
    for met in ("mknn", "cka_b", "cka_d"):
        t = T[met]["full"]
        cells = []
        for c in ("arh_rank", "dense_rank", "log_params"):
            cells.append(" / ".join(f(t[x]["arh"][c].get("r")) for x in ("before_raw", "after_scal", "uncensored")))
        out.append(f"| {MN[met]} | " + " | ".join(cells) + " |")
print("\n".join(out))
