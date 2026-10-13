#!/usr/bin/env python
"""OPENLLAMA-AXIS COPY of analysis/s12_layermax_llama/indep/lmx_llama_report.py (sub-role I of oll-axis-extract,
S0b): writes results/S1_2_layermax_openllama/indep/REPORT.md from indep/summary.json (verdicts_ol.json
content) and the validation files next to it only; every number is read from them. Labels: every
OpenLLaMA-axis result is descriptive (no confirmatory endpoint, plan R.1.2); J lives in joint/ and is
EXPLORATORY (R.1.3); P-O2b lives in forecast/. Part B gate first (plan F1(a)).
usage: launch_ol.sh ol_I_report 1 -- venv/bin/python analysis/s12_layermax_openllama/indep/lmx_ol_report.py
       (after lmx_ol_aggregate.py; rerun after cross_validate.py to add the V-O5 section)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ol_gate  # noqa: E402
OUT_ROOT = ol_gate.gate(__file__)   # F1(a): exits 3 unless Part B is frozen (S0b test mode: scratch root)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import lmx_common as L  # noqa: E402

OUT = L.OUT
PRI = "prh_clip+l2"
PRES = ("prh_clip+l2", "l2")
METS = (("mknn", "mKNN"), ("cka_biased", "CKA (biased)"), ("cka_debiased", "CKA (debiased, exploratory)"))


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, n=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, str):
        return x
    return f"{x:+.{n}f}"


def e(x):
    return "n/a" if x is None else f"{x:.1e}"


def main():
    ap = argparse.ArgumentParser(description="REPORT.md of implementation I")
    ap.add_argument("--sens-e2", action="store_true", help="report on indep/sens_e2/")
    args = ap.parse_args()
    global OUT
    if args.sens_e2:
        OUT = L.OUT_SENS
    S = load("summary.json")
    if S is None:
        raise SystemExit("summary.json missing")
    V = S["verdicts"]
    VA = load("validation.json") if not args.sens_e2 else None
    VL = load("validation_vl.json") if not args.sens_e2 else None
    V6 = load("validation_vo6_arh.json") if not args.sens_e2 else None
    XC = load("validation_cross.json") if not args.sens_e2 else None
    lines = []
    w = lines.append
    w("# OpenLLaMA axis (S1.2, third size axis): layer-max replication, implementation I")
    w("")
    w("**Status of every result below: descriptive. This axis has no confirmatory endpoint (plan R.1.2).** "
      "Every OpenLLaMA-only exact p has smallest attainable value 1/6 (one-sided). The joint summary J is EXPLORATORY "
      "(plan R.1.3) and is written by `joint_summary.py` (not by this implementation); P-O2b, the primary decision rule, "
      "is written by `forecast_ol.py` from these pair files.")
    w("")
    w(f"Panel: open_llama_3b / 7b / 13b × the 17 ARH `val` ViTs on WIT-1024 (51 pairs), both pre (`prh_clip+l2` primary, "
      f"`l2` secondary). K = {S['K']} permutations, keys `S1.2-layermax|<lm>|<vit>`. Every number is read from "
      f"`{os.path.relpath(os.path.join(OUT, 'summary.json'), L.PROJ)}` (its `verdicts` block = `verdicts_ol.json`) or the "
      f"validation files next to it. Test mode: {S.get('test_mode')}. S-E2 sensitivity run: {S.get('sens_e2')}. "
      f"Missing required verdict keys: {len(S['missing_required_keys'])} of {S['n_required_keys']}.")
    w("")
    # ---------------------------------------------------------------- validation
    w("## 1. Validation (plan section 11)")
    w("")
    if VL:
        v1 = VL.get("V-L1", {})
        w(f"- **V-O1** clip thresholds recomputed = reference bitwise: {f(v1.get('all_bitwise_equal'))} "
          f"({v1.get('n_models', 'n/a')} models); pass {f(v1.get('pass'))}.")
        v2 = VL.get("V-L2", {})
        if v2:
            sm = v2.get("summary", {})
            w(f"- **V-O2** kNN tables: ViT exact boundary-tie rows {sm.get('vit_total_exact_boundary_tie_rows', 'n/a')}; "
              f"frozen tables identical {f(sm.get('frozen_tables_all_identical'))} ({sm.get('n_frozen_compared', 'n/a')} compared); "
              f"observed counts = table overlaps {f(sm.get('obs_counts_all_identical'))}; pass {f(v2.get('pass'))}.")
        v4 = VL.get("V-L4B", {})
        if v4:
            w(f"- **V-O4b** rerun 3b/7b vs frozen (I outputs strict 1e-12 relative, E outputs in the V-O5 form; p, argmax, "
              f"null counts identical): pass {f(v4.get('pass_V-O4b_I'))} (all-strict form: {f(v4.get('pass'))}).")
        v7 = VL.get("V-L7", {})
        if v7:
            for p in PRES:
                if p in v7:
                    r = v7[p]["new_17"]
                    w(f"- **V-O7(a)** {p}, new pairs: biased-CKA null mean vs √(PR·PR)/1023 median relative error "
                      f"{r['cka_b']['median_rel_err']:.4f}, |z| > 4 in {r['cka_b']['frac_abs_z_gt_4']:.4f}; debiased |z| > 4 "
                      f"{r['cka_d']['frac_abs_z_gt_4']:.4f}; mKNN |z| > 4 {r['mknn']['frac_abs_z_gt_4']:.4f}; pass {f(r['pass'])}.")
        v7c = VL.get("V-L7C", {})
        if v7c:
            w(f"- **V-O7(c)** m = 8 plant over all 8! permutations: max |mean debiased CKA| {e(v7c.get('max_abs_mean_dcka'))}; "
              f"pass {f(v7c.get('pass'))}.")
    if VA:
        if "I2" in VA:
            i2 = VA["I2"]
            w(f"- **V-O3** {i2['n_checks']} checks: mKNN all equal {f(i2['mknn_all_equal'])}, max |ΔCKA| biased "
              f"{e(i2['max_cka_b_absdiff'])}, debiased {e(i2['max_cka_d_absdiff'])}; pass {f(i2['pass'])}.")
            b = i2.get("V-O7b", {})
            w(f"- **V-O7(b)** closed form on the V-O3 entries: max |sum|/largest term {e(b.get('max_rel_to_largest'))}; "
              f"pass {f(b.get('pass'))}.")
        if "I3" in VA:
            i3 = VA["I3"]
            w(f"- **V-O4a** fixed-depth entries of 3b/7b vs S1.2 stored s_obs ({i3['n_configs']} configs): mKNN equal "
              f"{f(i3['mknn_all_equal'])}, max |ΔCKA| {e(i3['max_cka_b_absdiff'])}; pass {f(i3['pass'])}.")
    if V6:
        w(f"- **V-O6** (reduced, vendored ARH, our π): max |Δ| {e(V6.get('max_absdiff'))}; pass at 1e-6 {f(V6.get('pass_1e-6'))}.")
    w(f"- **Section G:** llama-13b/30b pair files = merged LLaMA summary: {f(V.get('G|llama_pairs_equal_merged_summary'))}; "
      f"rerun 3b/7b = frozen 8-LM files: {f(V.get('G|rerun_equals_frozen_3b7b'))}; records read for the 11-LM panel: "
      f"{V.get('G|n_pairs_read')}.")
    if V.get("G|rerun_equals_frozen_3b7b") is False:
        w("  - The plan section 3 source rule applies: the reruns stay primary; every verdict on the frozen 3b/7b values is "
          "in the `x|frozen|…` keys of `verdicts_ol.json`.")
    w("")
    # ---------------------------------------------------------------- trends
    w("## 2. OpenLLaMA-axis trends Tr_O (exact p over the 6 relabellings; descriptive)")
    w("")
    for p in PRES:
        w(f"**{p}{' (primary)' if p == PRI else ' (secondary)'}.** Tr_O (one-sided p upper / lower); ΔTr_O vs raw.")
        w("")
        w("| metric | protocol | raw | s_cal | uncensored | ΔTr s_cal | ΔTr unc |")
        w("|---|---|---|---|---|---|---|")
        for m, mn in METS:
            for q in ("full", "fixed_r0.5", "fixed_r1.0", "grid17"):
                cells = []
                for x in ("raw", "s_cal", "unc"):
                    k = f"{p}|{m}|{q}|{x}"
                    cells.append(f"{f(V[f'tr|{k}'])} ({f(V[f'tr_pge|{k}'], 2)}/{f(V[f'tr_ple|{k}'], 2)})")
                cells += [f(V[f"dtr|{p}|{m}|{q}|s_cal"]), f(V[f"dtr|{p}|{m}|{q}|unc"])]
                w(f"| {mn} | {q} | " + " | ".join(cells) + " |")
            k = f"{p}|{m}|ctm"
            w(f"| {mn} | calibrate-then-max | TM {f(V[f'tr|{k}|TM'])} | M {f(V[f'tr|{k}|M'])} | M_unc {f(V[f'tr|{k}|M_unc'])} | "
              f"{f(V[f'dtr|{k}|M'])} | {f(V[f'dtr|{k}|M_unc'])} |")
        w("")
    # ---------------------------------------------------------------- Part B rules
    w("## 3. Part B decision rules and categories (plan sections 9, 10)")
    w("")
    for p in PRES:
        w(f"**{p}{' (primary)' if p == PRI else ' (secondary)'}.**")
        w(f"- P-O1 (mKNN; near-identity): holds {f(V[f'PO1|{p}|holds'])}; ζ check s_cal {V[f'PO1|{p}|zeta|s_cal|nq']} qualifying, "
          f"{V[f'PO1|{p}|zeta|s_cal|nrev']} reversed; unc {V[f'PO1|{p}|zeta|unc|nq']} qualifying, {V[f'PO1|{p}|zeta|unc|nrev']} reversed.")
        for x in ("s_cal", "unc"):
            w(f"- P-O2 ({x}): **{V[f'PO2|{p}|{x}|cat']}**; distance to the nearest boundary {f(V[f'PO2|{p}|{x}|dist'])}, "
              f"knife edge {f(V[f'PO2|{p}|{x}|kedge'])}; F_new {f(V[f'fnew|{p}|cka_biased|full|{x}|F'])} "
              f"(f₋ {V[f'fnew|{p}|cka_biased|full|{x}|fminus']}, f₊ {V[f'fnew|{p}|cka_biased|full|{x}|fplus']}, "
              f"CP of f₋/34 [{f(V[f'fnew|{p}|cka_biased|full|{x}|cp_lo'])}, {f(V[f'fnew|{p}|cka_biased|full|{x}|cp_hi'])}]); "
              f"pilot 3B–7B order changes down/up {V[f'chg|{p}|cka_biased|full|{x}|pilot|down']}/{V[f'chg|{p}|cka_biased|full|{x}|pilot|up']}.")
        w(f"- P-O2a(i): median offset c_α − B̄ over the 17 new pairs {V[f'PO2ai|{p}|median']:.5f}; in [0.0009, 0.0029] "
          f"{f(V[f'PO2ai|{p}|in_band'])}; {V[f'PO2ai|{p}|x']}/17 offsets in the band (CP [{f(V[f'PO2ai|{p}|cp_lo'])}, "
          f"{f(V[f'PO2ai|{p}|cp_hi'])}]); LP comparison median |offset| ≤ 0.01: {f(V[f'PO2ai|{p}|lp_le_0.01'])}.")
        w(f"- P-O2a(ii): {V[f'PO2aii|{p}|x']}/{V[f'PO2aii|{p}|n']} (need {V[f'PO2aii|{p}|need']}): **{V[f'PO2aii|{p}|verdict']}**.")
        w(f"- P-O3 (fixed depth; descriptive on this axis): r 0.5 s_cal {V[f'PO3|{p}|0.5|s_cal|desc']}, unc {V[f'PO3|{p}|0.5|unc|desc']}; "
          f"r 1.0 s_cal {V[f'PO3|{p}|1.0|s_cal|desc']}, unc {V[f'PO3|{p}|1.0|unc|desc']}.")
        w(f"- P-O4 (debiased CKA; near-identity): s_cal {f(V[f'PO4|{p}|s_cal|holds'])}, unc {f(V[f'PO4|{p}|unc|holds'])}, "
          f"11-LM r {f(V[f'PO4|{p}|arh|holds'])}.")
        w(f"- P-O5a: {V[f'PO5a|{p}|x']}/{V[f'PO5a|{p}|n']} (need {V[f'PO5a|{p}|need']}): {V[f'PO5a|{p}|verdict']}. "
          f"P-O5 M: {V[f'PO5|{p}|M|cat']} (prediction holds {f(V[f'PO5|{p}|M|pred_holds'])}); M_unc: {V[f'PO5|{p}|M_unc|cat']}.")
        w(f"- D3: calibration term {f(V[f'd3cal|{p}|cka_biased'])}, selection term {f(V[f'd3sel|{p}|cka_biased'])}; "
          f"deductive violations {V[f'D3|{p}|nviol']}.")
        w(f"- Plug-in Tr_O(T − B̄) {f(V[f'plugin|{p}|tr'])} vs Tr_O(unc): difference {f(V[f'plugin|{p}|diff'])}, within 0.10 "
          f"{f(V[f'plugin|{p}|within'])}.")
        w(f"- Section 7 OpenLLaMA reading: biased CKA {V[f'readO|{p}|cka_biased']}; debiased (exploratory) {V[f'readO|{p}|cka_debiased']}.")
        nk = sum(v for k, v in V.items() if k.startswith(f"keepid|{p}|") and isinstance(v, int))
        ns = sum(v for k, v in V.items() if k.startswith(f"signrule|{p}|") and isinstance(v, int))
        w(f"- Deductive checks (implementation errors if nonzero): keep identity {nk}, sign rule {ns}, P-O1 ζ reversals "
          f"{V[f'PO1|{p}|zeta|s_cal|nrev'] + V[f'PO1|{p}|zeta|unc|nrev']}, D3 {V[f'D3|{p}|nviol']}.")
        w(f"- Threshold curves (τ = 0.03 … 0.20), P-O2 s_cal: "
          + "; ".join(f"{t}: {V[f'curve|{p}|PO2|s_cal|tau={t}']}" for t in ("0.03", "0.06", "0.1", "0.15", "0.2")) + ".")
        w(f"- Materiality by δ_b: " + "; ".join(f"{d}: {f(V[f'curve|{p}|material|db={d}'])}" for d in ("0.0025", "0.005", "0.0075", "0.01")) + ".")
        w("")
    # ---------------------------------------------------------------- pooled
    w("## 4. ARH-style pooled panel (11 LMs, 187 pairs; plan sections 5.3, 7)")
    w("")
    w("| pre | metric | r raw | r s_cal | r unc | exact p raw (11!) | 95% ViT CI raw |")
    w("|---|---|---|---|---|---|---|")
    for p in PRES:
        for m, mn in METS:
            k = f"arh|11|{p}|{m}"
            w(f"| {p} | {mn} | {f(V[f'{k}|raw|arh|r'])} | {f(V[f'{k}|s_cal|arh|r'])} | {f(V[f'{k}|unc|arh|r'])} | "
              f"{e(V[f'{k}|raw|arh|pge'])} | [{f(V[f'{k}|raw|arh|ci_lo'])}, {f(V[f'{k}|raw|arh|ci_hi'])}] |")
    w("")
    for p in PRES:
        w(f"- **{p}:** ΔΔ (biased CKA) 11 LMs {f(V[f'dd|11|{p}|cka_biased'])}, ViT-bootstrap CI "
          f"[{f(V[f'dd|11|{p}|cka_biased|ci_lo'])}, {f(V[f'dd|11|{p}|cka_biased|ci_hi'])}], exact relabelling p "
          f"{e(V[f'dd|11|{p}|cka_biased|pge'])} (descriptive: 10 of 11 LMs known); 10 LMs {f(V[f'dd|10|{p}|cka_biased'])}; "
          f"8 LMs {f(V[f'dd|8|{p}|cka_biased'])}.")
        w(f"  - §7(a) rule{' (decides the wording of l.2752)' if p == PRI else ' (secondary pre; not used for l.2752)'}: "
          f"**{V[f'rule7a|{p}']}**; ARH size 0.40 above the upper bound: {f(V[f'size_not_reproduced|{p}'])}.")
        w(f"  - FP§6 category (continuity): raw {V[f'fp6|{p}|raw_cat']}; gap CI [{f(V[f'fp6|{p}|gap_ci_lo'])}, "
          f"{f(V[f'fp6|{p}|gap_ci_hi'])}]; reported: **{V[f'fp6|{p}|cat']}**.")
        w(f"  - Debiased CKA (exploratory): ΔΔ {f(V[f'dd|11|{p}|cka_debiased'])}, CI [{f(V[f'dd|11|{p}|cka_debiased|ci_lo'])}, "
          f"{f(V[f'dd|11|{p}|cka_debiased|ci_hi'])}].")
    if V.get(f"rule7a|{PRI}") == "opposite direction":
        w("- **The CI lies wholly below 0: not covered by the author's pre-commitment; goes to the author before any wording.**")
    w("")
    # ---------------------------------------------------------------- cross validation
    w("## 5. Cross-implementation check (V-O5, I side)")
    w("")
    if XC:
        w(f"- Order: I's summary.json / verdicts_ol.json final before E was opened (recorded {XC['order']['E_first_opened_after_utc']}).")
        for ln in XC.get("report_lines", []):
            w(ln)
        w(f"- **V-O5 pass (I side): {f(XC.get('pass_V-O5'))}.**")
    else:
        w("- Not yet run (cross_validate.py runs after this report's first version; rerun this script afterwards).")
    w("")
    w("## 6. Labels")
    w("")
    w("- Every OpenLLaMA-axis number here is descriptive; no item is a hypothesis test (plan section 5).")
    w("- Debiased-CKA readings are exploratory; `l2` readings are secondary (plan section 7).")
    w("- J (joint 3-family relabelling p) is EXPLORATORY and is reported by `joint_summary.py` with its labels.")
    w("- Interpretation written after the numbers were seen must be labelled post hoc (plan F8); this file contains none.")
    w("")
    w(f"Written {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by lmx_ol_report.py from summary.json "
      f"(finished {S.get('finished_utc')}).")
    fn = os.path.join(OUT, "REPORT.md")
    with open(fn + ".tmp", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(fn + ".tmp", fn)
    print(f"wrote {fn} ({len(lines)} lines)", flush=True)


if __name__ == "__main__":
    main()
