"""P0 bookkeeping for the S1.2 LLaMA axis (plan ANALYSIS_PLAN_LLAMA.md S8, left open).

Writes results/S1_2_layermax_llama/layermax_llama_summary.json and then
results/S1_2_layermax_llama/DONE (last).

No score is recomputed. Every value is COPIED from files that already exist and
whose sha256 was logged in STATUS.md by the role that wrote them:
  E summary, I summary, I cross-validation, the review's recomputed verdicts.
The script aborts (exit 2) if any input hash differs from the logged value, or
if either output already exists.

Role: [oll-axis-freeze]. Run: venv/bin/python -I -B <this file> [--dry]  (1 thread, CPU; --dry writes nothing).
"""
import hashlib
import json
import math
import os
import sys
import time

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
R = os.path.join(PROJ, "results", "S1_2_layermax_llama")
OUT = os.path.join(R, "layermax_llama_summary.json")
DONE = os.path.join(R, "DONE")

# path (relative to PROJ) -> (expected sha256, STATUS line that logged it, what it is)
SOURCES = {
    "results/S1_2_layermax_llama/early/summary.json":
        ("cdbd3819ab82808c9779e10dd474fed0a2cb4c655fa45b4a4e42d0ba41d7d3b5", "2026-10-09T01:33:44Z [llama-axis-run-early]", "implementation E aggregate"),
    "results/S1_2_layermax_llama/indep/summary.json":
        ("5ee3ed87c7ed5db7fbcbc34d4505f49a191d0893221fc4d9d71364d5d521b695", "2026-10-09T01:35:37Z [llama-axis-run-indep]", "implementation I aggregate (final before E opened)"),
    "results/S1_2_layermax_llama/indep/validation_cross.json":
        ("b89c2a876e27965efee746dfd33cd3cc53f2c28dad1c4300046927ce51fe34f2", "2026-10-09T01:43:40Z [llama-axis-run-indep]", "I vs E cross-validation incl. exact-rational mKNN trends"),
    "results/S1_2_layermax_llama/review/recomputed_verdicts.json":
        ("4ed53698aaf9ddd9861738f48c582c974c29a9df4a5e9ed44e4ce4d86b37dcfe", "2026-10-09T01:58:12Z [llama-axis-review]", "review: every verdict recomputed from the frozen rules with new code"),
    "results/S1_2_layermax_llama/early/REPORT.md":
        ("44132085ee389495ecadf094aadff551c951e994b2a9d4ac356b6593f4308f3e", "2026-10-09T01:33:44Z [llama-axis-run-early]", "E report"),
    "results/S1_2_layermax_llama/indep/REPORT.pre_cross.md":
        ("6610024e87f3e2e37f0b148c56a67d52353cf20895274221fa91ab47800294a7", "2026-10-09T01:35:37Z [llama-axis-run-indep]", "I report before cross-validation"),
    "results/S1_2_layermax_llama/indep/REPORT.md":
        ("2e85f70ef5fe708152b798f395a8ad633ced0f0655fa63aa30bd458bad575574", "2026-10-09T01:43:40Z [llama-axis-run-indep]", "I report (sections 1-7 = pre_cross, + 7b)"),
    "results/S1_2_layermax_llama/early/validation/VL5_E_vs_I.json":
        ("6f918d73bab2352af8f858466ac48659ddca954179d634142a1834883c9912f3", "2026-10-09T01:29:06Z [llama-axis-run-early]", "V-L5 from the E side"),
    "results/S1_2_layermax_llama/early/hashes.json":
        ("82f407d601b66f4955f178ecbf1dcb7901d97557fa5535433ed7d2047df8c11a", "validation_cross.json inputs", "E code and output hashes"),
    "analysis/s12_layermax_llama/early/interpretation.txt":
        ("3937342e7f50e78fe29dc133beee96f6f2929ab78ba3340aa4d067a6cb9e44d5", "2026-10-09T01:33:44Z [llama-axis-run-early]", "E section-0 interpretation (post hoc)"),
    "analysis/s12_layermax_llama/review/llama_axis_review.py":
        ("a96576195e69d2cb42ab28db3da509da348e5b37f9832f1f5843374c307dc4ef", "2026-10-09T01:58:12Z [llama-axis-review]", "review code"),
    "analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md":
        ("990428e3f29c7ecfba9cf314a45c73c79f0154f86a07520429910d282c0a2fc8", "2026-10-08T22:19:40Z", "frozen plan (Part A)"),
    "analysis/s12_layermax_llama/PARTB_PREDICTIONS.md":
        ("f9afa919957d01eed8373253e710baa6b7d22c3d613099cabc9f6107f8bec5c4", "2026-10-08T23:18:46Z [llama-axis-predict]", "frozen Part B"),
    "results/S1_2_layermax_llama/partB/partB_features.json":
        ("ede0127589a02037427b343e9c3593ee427d430e9746d7f4942b15f76a3031bf", "2026-10-08T23:18:46Z [llama-axis-predict]", "frozen Part B features"),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def norm(v):
    """Normalise outcome encodings: E stores some outcomes as booleans named
    'holds', I as the string 'holds'."""
    if v is True:
        return "holds"
    if v is False:
        return "fails"
    # category labels: E, I and the review add different parenthetical reasons to
    # the same category; compare the category only (raw strings are kept in the file)
    if isinstance(v, str):
        for cat in ("not applicable", "no directional prediction"):
            if v.startswith(cat):
                return cat
    return v


def same(a, b):
    a, b = norm(a), norm(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return abs(a - b) <= 1e-12 * max(abs(a), abs(b)) + 1e-15
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


# (headline field, E path under per_pre[pre], I path under per_pre[pre])
FIELDS = [
    ("P-L1.Tr_raw", ["predictions", "P-L1", "Tr_raw"], ["partB", "P-L1", "Tr_raw"]),
    ("P-L1.Tr_scal", ["predictions", "P-L1", "Tr_scal"], ["partB", "P-L1", "Tr_scal"]),
    ("P-L1.Tr_unc", ["predictions", "P-L1", "Tr_unc"], ["partB", "P-L1", "Tr_unc"]),
    ("P-L1.dTr_scal", ["predictions", "P-L1", "dTr_scal"], ["partB", "P-L1", "dTr_scal"]),
    ("P-L1.dTr_unc", ["predictions", "P-L1", "dTr_unc"], ["partB", "P-L1", "dTr_unc"]),
    ("P-L1.outcome", ["predictions", "P-L1", "outcome"], ["partB", "P-L1", "outcome"]),
    ("P-L2.Bbar_material", ["predictions", "P-L2", "material"], ["partB", "Bbar_material"]),
    ("P-L2.s", ["predictions", "P-L2", "s"], ["partB", "s"]),
    ("P-L2.Tr_pred", ["predictions", "P-L2", "Tr_pred"], ["partB", "Bbar_Tr_pred"]),
    ("P-L2.Tr_raw", ["predictions", "P-L2", "Tr_raw"], ["partB", "P-L2", "Tr_raw"]),
    ("P-L2.Tr_scal", ["predictions", "P-L2", "Tr_scal"], ["partB", "P-L2", "Tr_scal"]),
    ("P-L2.Tr_unc", ["predictions", "P-L2", "Tr_unc"], ["partB", "P-L2", "Tr_unc"]),
    ("P-L2.dTr_scal", ["predictions", "P-L2", "dTr_scal"], ["partB", "P-L2", "dTr_scal"]),
    ("P-L2.dTr_unc", ["predictions", "P-L2", "dTr_unc"], ["partB", "P-L2", "dTr_unc"]),
    ("P-L2.p_one_decrease_scal", ["predictions", "P-L2", "p_dTr_scal", "p_lower"], ["partB", "P-L2", "dTr_scal_p_one_decrease"]),
    ("P-L2.p_one_decrease_unc", ["predictions", "P-L2", "p_dTr_unc", "p_lower"], ["partB", "P-L2", "dTr_unc_p_one_decrease"]),
    ("P-L2.outcome_scal", ["predictions", "P-L2", "outcome_scal"], ["partB", "P-L2", "outcome_scal"]),
    ("P-L2.outcome_unc_secondary", ["predictions", "P-L2", "outcome_unc_secondary"], ["partB", "P-L2", "outcome_unc_secondary"]),
    ("P-L2.flips_scal_conc_to_disc", ["predictions", "P-L2", "flips_scal", "conc_to_disc"], ["partB", "P-L2", "pairwise_flips_qualifying", "s_cal", "conc_to_disc"]),
    ("P-L2.flips_unc_conc_to_disc", ["predictions", "P-L2", "flips_unc", "conc_to_disc"], ["partB", "P-L2", "pairwise_flips_qualifying", "uncensored", "conc_to_disc"]),
    ("P-L2a.median_calpha_minus_Bbar", ["predictions", "P-L2a", "median_abs_calpha_minus_Bbar"], ["partB", "P-L2a", "median_abs_calpha_minus_Bbar"]),
    ("P-L2a.n_qualifying", ["predictions", "P-L2a", "n_qualifying"], ["partB", "P-L2a", "n_qualifying"]),
    ("P-L2a.n_agree", ["predictions", "P-L2a", "n_agree_larger_higher"], ["partB", "P-L2a", "n_agree"]),
    ("P-L2a.n_needed", ["predictions", "P-L2a", "n_needed"], ["partB", "P-L2a", "n_needed"]),
    ("P-L2a.outcome", ["predictions", "P-L2a", "holds"], ["partB", "P-L2a", "outcome"]),
    ("P-L3.r0.5.Tr_raw", ["predictions", "P-L3", "0.5", "Tr_raw"], ["partB", "P-L3", "r0.5", "Tr_raw"]),
    ("P-L3.r0.5.Tr_scal", ["predictions", "P-L3", "0.5", "Tr_scal"], ["partB", "P-L3", "r0.5", "Tr_scal"]),
    ("P-L3.r0.5.Tr_unc", ["predictions", "P-L3", "0.5", "Tr_unc"], ["partB", "P-L3", "r0.5", "Tr_unc"]),
    ("P-L3.r0.5.outcome", ["predictions", "P-L3", "0.5", "outcome"], ["partB", "P-L3", "r0.5", "outcome"]),
    ("P-L3.r1.0.Tr_raw", ["predictions", "P-L3", "1.0", "Tr_raw"], ["partB", "P-L3", "r1.0", "Tr_raw"]),
    ("P-L3.r1.0.Tr_scal", ["predictions", "P-L3", "1.0", "Tr_scal"], ["partB", "P-L3", "r1.0", "Tr_scal"]),
    ("P-L3.r1.0.Tr_unc", ["predictions", "P-L3", "1.0", "Tr_unc"], ["partB", "P-L3", "r1.0", "Tr_unc"]),
    ("P-L3.r1.0.outcome", ["predictions", "P-L3", "1.0", "outcome"], ["partB", "P-L3", "r1.0", "outcome"]),
    ("P-L4.Tr_raw", ["predictions", "P-L4", "Tr_raw"], ["partB", "P-L4", "Tr_raw"]),
    ("P-L4.dTr_scal", ["predictions", "P-L4", "dTr_scal"], ["partB", "P-L4", "dTr_scal"]),
    ("P-L4.dTr_unc", ["predictions", "P-L4", "dTr_unc"], ["partB", "P-L4", "dTr_unc"]),
    ("P-L4.pooled10_r_before", ["predictions", "P-L4", "pooled_r_before"], ["partB", "P-L4", "pooled10_r_before"]),
    ("P-L4.pooled10_r_after", ["predictions", "P-L4", "pooled_r_after"], ["partB", "P-L4", "pooled10_r_after"]),
    ("P-L4.outcome", ["predictions", "P-L4", "outcome"], ["partB", "P-L4", "outcome"]),
    ("P-L5.Tr_raw_layermax", ["predictions", "P-L5", "Tr_raw_layermax"], ["partB", "P-L5", "Tr_raw_layermax"]),
    ("P-L5.Tr_M", ["predictions", "P-L5", "Tr_M"], ["partB", "P-L5", "Tr_M"]),
    ("P-L5.Tr_M_unc", ["predictions", "P-L5", "Tr_M_unc"], ["partB", "P-L5", "Tr_M_unc"]),
    ("P-L5.outcome", ["predictions", "P-L5", "outcome"], ["partB", "P-L5", "outcome"]),
    ("ARH10.cka_b.r_before_cka", ["readings", "cka_biased", "r_before_cka"], ["readings", "cka_b", "inputs", "r_before_cka"]),
    ("ARH10.cka_b.r_after_cka", ["readings", "cka_biased", "r_after_cka"], ["readings", "cka_b", "inputs", "r_after_cka"]),
    ("ARH10.cka_b.r_before_mknn", ["readings", "cka_biased", "r_before_mknn"], ["readings", "cka_b", "inputs", "r_before_mknn"]),
    ("ARH10.cka_b.r_after_mknn", ["readings", "cka_biased", "r_after_mknn"], ["readings", "cka_b", "inputs", "r_after_mknn"]),
    ("ARH10.cka_b.dDelta", ["readings", "cka_biased", "delta_diff"], ["readings", "cka_b", "arh_pooled10", "dDelta"]),
    ("ARH10.cka_b.r_after_gap_mknn_minus_cka", ["readings", "cka_biased", "r_after_gap_mknn_minus_cka"], ["readings", "cka_b", "arh_pooled10", "r_after_gap"]),
    ("ARH10.cka_b.verdict", ["readings", "cka_biased", "verdict_arh_style_10LM"], ["readings", "cka_b", "arh_pooled10", "verdict"]),
    ("ARH8.cka_b.dDelta_reference", ["readings", "cka_biased", "delta_diff_8LM"], ["readings", "cka_b", "arh_pooled8_reference", "dDelta"]),
    ("ARH8.cka_b.verdict_reference", ["readings", "cka_biased", "verdict_arh_style_8LM_frozen"], ["readings", "cka_b", "arh_pooled8_reference", "verdict"]),
    ("LLaMA_axis.cka_b.verdict", ["readings", "cka_biased", "verdict_llama_axis"], ["readings", "cka_b", "llama_axis"]),
    ("ARH10.cka_d.dDelta", ["readings", "cka_debiased", "delta_diff"], ["readings", "cka_d", "arh_pooled10", "dDelta"]),
    ("ARH10.cka_d.verdict", ["readings", "cka_debiased", "verdict_arh_style_10LM"], ["readings", "cka_d", "arh_pooled10", "verdict"]),
    ("LLaMA_axis.cka_d.verdict", ["readings", "cka_debiased", "verdict_llama_axis"], ["readings", "cka_d", "llama_axis"]),
]
E_ONLY = [
    ("ARH10.cka_b.dDelta_ci95_vit_bootstrap", ["readings", "cka_biased", "delta_diff_ci"]),
    ("ARH10.cka_d.dDelta_ci95_vit_bootstrap", ["readings", "cka_debiased", "delta_diff_ci"]),
]
# review verdict list index -> headline field it must equal
REVIEW_MAP = [
    ("P-L1", 3, "P-L1.outcome"),
    ("P-L2", 3, "P-L2.outcome_scal"),
    ("P-L2", 5, "P-L2.outcome_unc_secondary"),
    ("P-L2a", 2, "P-L2a.outcome"),
    ("P-L5", 3, "P-L5.outcome"),
    ("P-L4", 4, "P-L4.outcome"),
    ("ARH_reading_cka_b", 6, "ARH10.cka_b.verdict"),
    ("ARH_reading_cka_d", 6, "ARH10.cka_d.verdict"),
    ("LLaMA_reading", 3, "LLaMA_axis.cka_b.verdict"),
]


def verdict_equiv(review_val, summary_val):
    a, b = norm(review_val), norm(summary_val)
    if a == b:
        return True
    # E/I write "ARH pattern absent"; the review abbreviates it as "absent"
    return isinstance(a, str) and isinstance(b, str) and a == "absent" and b == "ARH pattern absent"


def main():
    t0 = time.time()
    for p in (OUT, DONE):
        if os.path.exists(p):
            print(f"STOP: {p} already exists; nothing written")
            sys.exit(2)
    got = {}
    bad = []
    for rel, (exp, logged, what) in SOURCES.items():
        h = sha(os.path.join(PROJ, rel))
        got[rel] = {"sha256": h, "expected_sha256": exp, "logged_in": logged, "what": what, "match": h == exp}
        if h != exp:
            bad.append(rel)
    if bad:
        print("STOP: input hash mismatch:", bad)
        sys.exit(2)

    E = json.load(open(os.path.join(R, "early", "summary.json")))
    I = json.load(open(os.path.join(R, "indep", "summary.json")))
    X = json.load(open(os.path.join(R, "indep", "validation_cross.json")))
    V = json.load(open(os.path.join(R, "review", "recomputed_verdicts.json")))

    headline, mism, n_cmp = {}, [], 0
    for pre in ("prh_clip+l2", "l2"):
        h = {}
        for name, pe, pi in FIELDS:
            ve, vi = get(E["per_pre"][pre], pe), get(I["per_pre"][pre], pi)
            eq = None
            if ve is not None and vi is not None:
                eq = same(ve, vi)
                n_cmp += 1
                if not eq:
                    mism.append({"pre": pre, "field": name, "E": ve, "I": vi})
            h[name] = {"E": ve, "I": vi, "E_equals_I": eq}
        for name, pe in E_ONLY:
            h[name] = {"E": get(E["per_pre"][pre], pe), "I": "not computed by I", "E_equals_I": None}
        rv = {}
        for impl in ("E", "I"):
            vv = V["verdicts"].get(f"{pre}|{impl}", {})
            for key, idx, field in REVIEW_MAP:
                rval = vv.get(key, [None] * 8)[idx] if isinstance(vv.get(key), list) and len(vv[key]) > idx else None
                sval = h[field][impl]
                ok = verdict_equiv(rval, sval) if (rval is not None and sval is not None) else None
                rv[f"{impl}|{key}[{idx}]"] = {"review": rval, "summary": sval, "equal": ok}
                if ok is False:
                    mism.append({"pre": pre, "field": f"review {impl} {key}[{idx}]", "review": rval, "summary": sval})
            for r_key, r_field in (("fixed_r0.5", "P-L3.r0.5.outcome"), ("fixed_r1.0", "P-L3.r1.0.outcome")):
                pl3 = vv.get("P-L3", {}).get(r_key)
                rval = pl3[2] if isinstance(pl3, list) and len(pl3) > 2 else None
                sval = h[r_field][impl]
                # review writes "not applicable" where B_fd is not material; E/I write their own label
                ok = verdict_equiv(rval, sval) if (rval is not None and sval is not None) else None
                rv[f"{impl}|P-L3.{r_key}"] = {"review": rval, "summary": sval, "equal": ok}
        review_E_eq_I = V["verdicts"].get(f"{pre}|E") == V["verdicts"].get(f"{pre}|I")
        headline[pre] = {"values": h, "review_vs_summary": rv, "review_E_equals_review_I": review_E_eq_I}

    review_mism = [(pre, k, v) for pre in headline for k, v in headline[pre]["review_vs_summary"].items() if v["equal"] is False]

    out = {
        "what": ("Merged S1.2 LLaMA-axis summary (ANALYSIS_PLAN_LLAMA.md S8: 'results/S1_2_layermax_llama/DONE is written last'; "
                 "§11: 'Every number in the report comes from results/S1_2_layermax_llama/layermax_llama_summary.json'). "
                 "Written late, as P0 of the OpenLLaMA-axis plan. NO score recomputed: every value below is copied from the "
                 "sources listed, whose sha256 equal the values logged in STATUS.md."),
        "written_by_role": "oll-axis-freeze",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": got,
        "source_rule": [
            "Plan-level numbers: E and I agree on every plan-level quantity (validation_cross.json pass_plan_level = true; "
            "headline_agreement below re-checks the copied headline values at 1e-12 rel + 1e-15).",
            "Verdicts: the review's recomputed_verdicts.json (new code, frozen rules) is the reference; review_vs_summary checks "
            "that it equals both E and I.",
            "Descriptive mKNN A3 trends (shift and inflation) where E and I differ by float round-off of exact ties: the "
            "exact-rational values in cross_validation.part4_mknn_exact_rational are primary; neither E's nor I's float value "
            "is used for those three quantities.",
            "ViT-bootstrap CI of the ARH-style dDelta exists only in E (I did not compute it); it is copied from E.",
            "Every other number: the embedded E and I summaries (embedded.early_summary, embedded.indep_summary), unchanged.",
        ],
        "headline": headline,
        "headline_agreement": {"n_compared_E_vs_I": n_cmp, "n_mismatch": len(mism), "mismatches": mism,
                               "n_review_vs_summary_mismatch": len(review_mism)},
        "verdicts_review": V.get("verdicts"),
        "review_diagnostics": V.get("diagnostics"),
        "section10": {"I_section10_primary": I.get("section10_primary"),
                      "E_section10_by_pre": {pre: E["per_pre"][pre].get("section10") for pre in E["per_pre"]}},
        "cross_validation": {
            "pass_plan_level": X.get("pass_plan_level"), "pass_all": X.get("pass_all"),
            "part3_summary_level": {k: X["part3_summary_level"].get(k) for k in ("n_compared", "n_mismatch", "mismatches")},
            "part4_mknn_exact_rational": X.get("part4_mknn_exact_rational"),
            "V-L1_clip_E_vs_I": X.get("V-L1_clip_E_vs_I"),
            "finished_utc": X.get("finished_utc"),
        },
        "deviations_recorded_by_review": [
            "Part B guard checked results/S1_2_layermax_llama/pairs/, a path neither run used (E wrote early/pairs/<pre>/, "
            "I wrote indep/<pre>/pairs/); the order Part B -> scores rests on file birth times (STATUS 2026-10-09T01:58:12Z).",
            "Plan §11 thread cap exceeded: nominal 12 threads 23:31-23:35Z and 7 until 00:48Z; E and I ran V-L6 concurrently "
            "(STATUS 2026-10-09T01:58:12Z).",
            "[arh-layermax] edited the existing results/S1_2_layermax/indep/LAYERMAX_INDEP_REPORT.md at 01:24Z (numbers "
            "unchanged, old copy kept) (STATUS 2026-10-09T01:58:12Z); open question for the author.",
            "Extraction: an E1 checker bug (one-dtype rule vs legacy fp32 rotary inv_freq buffers) was found and fixed; "
            "implementation error, not data (STATUS 2026-10-08T22:46:01Z).",
            "Merged summary and DONE were left by both run roles to the lead and are written only now (this file).",
        ],
        "embedded": {"early_summary": E, "indep_summary": I},
    }
    if "--dry" in sys.argv:
        print(json.dumps({"DRY": True, "n_compared": n_cmp, "n_mismatch": len(mism), "mismatches": mism,
                          "n_review_vs_summary_mismatch": len(review_mism), "review_mismatches": review_mism,
                          "review_E_eq_I": {p: headline[p]["review_E_equals_review_I"] for p in headline},
                          "headline_prh": {k: v for k, v in headline["prh_clip+l2"]["values"].items()}},
                         indent=1, default=str)[:12000])
        return
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, sort_keys=False, allow_nan=True)
    os.replace(tmp, OUT)
    hs = sha(OUT)
    with open(DONE, "w") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} layermax_llama_summary.json sha256 {hs}\n")
    print(json.dumps({"summary_sha256": hs, "done_sha256": sha(DONE), "n_compared": n_cmp, "n_mismatch": len(mism),
                      "mismatches": mism, "n_review_vs_summary_mismatch": len(review_mism),
                      "review_mismatch_examples": review_mism[:10],
                      "review_E_eq_I": {p: headline[p]["review_E_equals_review_I"] for p in headline},
                      "size_bytes": os.path.getsize(OUT), "secs": round(time.time() - t0, 2)}, indent=1, default=str))


if __name__ == "__main__":
    main()
