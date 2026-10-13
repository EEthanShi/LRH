#!/usr/bin/env python
"""Pythia axis, role pyt-review: merged summary and DONE (plan ANALYSIS_PLAN_PYTHIA.md 6de89603, section 4 F7 and
section 11 step S8). NEW script of the review role (not in the S0b manifest; logged as such). NO score is recomputed:
every value is copied from files whose sha256 equals the value logged in STATUS, and E = I is checked again on every
EI and FK key with the tolerance of VERDICTS_SPEC_PYT.md section 0.

GATE (plan sections 7 and 10): V-P5 (E vs I) fails by the letter on float tolerance (163 pair-level float fields,
3 descriptive section-D verdict keys). The plan reads no category and writes no DONE before V-P5 has 0 mismatches or
the lead has ruled under a hashed amendment. This script therefore WRITES NOTHING unless it is given
--vp5-ruling-line N, the 1-based number of a STATUS.md line that records that ruling (it must contain "V-P5" and
"RULING" or "AMENDMENT", be later than the review role's lines 321-323, not be a [pyt-axis-review] line, and must not say REJECT/RERUN/NOT ACCEPT).
The line is copied verbatim into the summary. If the ruling instead asks for a fix and rerun, do not use this script.

Writes, in this order and never overwriting:
  results/S1_2_layermax_pythia/layermax_pythia_summary.json  (chmod 444)
  results/S1_2_layermax_pythia/DONE                           (chmod 444, last)

usage: venv/bin/python -I analysis/s12_layermax_pythia/review_role/merge_pyt_summary.py --dry
       venv/bin/python -I analysis/s12_layermax_pythia/review_role/merge_pyt_summary.py --vp5-ruling-line N
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
RES = os.path.join(PROJ, "results", "S1_2_layermax_pythia")
R = "results/S1_2_layermax_pythia/"
A = "analysis/s12_layermax_pythia/"
OUTF = os.path.join(RES, "layermax_pythia_summary.json")
DONE = os.path.join(RES, "DONE")
REVIEW_START_LINE = 323  # the ruling must come after the review role's own lines 321-323
KNOWN_EI = ["D2|prh_clip+l2|cka_debiased|2.8b|c_minus_centry", "D2|prh_clip+l2|cka_debiased|70m|c_minus_centry",
            "D5|prh_clip+l2|70m|160m|raw_gain"]
EXPECT_CATEGORY = {"prh_clip+l2|s_cal": "C"}
INPUTS = {  # path: sha256 logged in STATUS (lines 216, 247, 255, 282, 285, 301, 304, 312-320 and the review lines)
    A + "ANALYSIS_PLAN_PYTHIA.md": "6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b",
    A + "CRITIQUE.md": "e452b318f35946d36e1b6d8566f8dd0cf2467cb1603462d9571284df3a7d7115",
    A + "S0B_CODE_MANIFEST.json": "ca600c80b94277328da6f9c2ee8e26785776bbfd0d80f8669b0a050098dcb237",
    A + "VERDICTS_SPEC_PYT.md": "96dce74e27aa69b0867326bd525c6cbc1666733dbfa0627e2f00d12afde5bfd7",
    A + "PARTB_PREDICTIONS_PYT.md": "64a5b83a5ae67413168d612bfe447bcd43e20fd6c5afd2d7e662b8489699d600",
    A + "g0_smoke/g0_smoke_pyt.py": "7c9e9595f03dc02593214002258e0d045268be7fdeacff4c00bb05d46bdc13ad",
    R + "partB/partB_features_pyt.json": "a8d6282945d7eeaa64095bf5771100edc21d0441ede18b84a0410f0fd9cd7b6a",
    R + "partB/clip_thresholds_partB_pyt.json": "5de400480d892f9243a78452d239dacf9aa0bd3d6ea7e136ed703dee01375454",
    R + "partB/partB_retro_checks_pyt.json": "2552056125203281c35ee7ed1ec24a8ec0aa9ceb52344c808cd4eca0d5eecb41",
    R + "extraction/EXTRACTION_SUMMARY.json": "f4183615cf8c8fc7c2f0c7ca2fb018b1fd878a3111fe0c82c32e5095ba2fda11",
    R + "extraction/S1_registry.json": "915f7fc2d580d9cd505801bc1811a7dfdd970c5a4a3d098fdb2a076b632033e5",
    R + "extraction/e1prov_2.8b.json": "ba77fce7ef8ac6b80d193fd3d1fb62225ae90356628a984a475397261c69a04e",
    R + "feasibility/feasibility_pyt.json": "e493871abf4e52589eb92b64a39845705ff3a9e6531006f6eac7602d78cb3642",
    R + "logs/gpu_gate_G0.json": "8d078889d67efe94939a7214945e0e9d0d30ef67c5b93591cf7df44ad3f109a2",
    R + "cache/clip_thresholds.json": "a778bf56eff6cd9aff1f095346e306e7be42ab054623618818c8bfd312653a49",
    R + "early/summary.json": "abdc0d069aeb471fab67edccc90e36a567afb028c3e5791e412513c5a6b76143",
    R + "early/verdicts_pyt.json": "99e0c54c84b9f439f45ad13f04d3f521ba11b97c2b40360b202fea5f66f626cb",
    R + "early/REPORT.md": "61cf2a6ad3787982df9056a67109feee5c3773a4cd15c8ee5df7b9037cb6d93f",
    R + "early/hashes.json": "8c374def6168c77362e691b0602bd662837e8c816267d2163d5a2e34bce1cac9",
    R + "early/validation/VP1_VP2_knn_tables.json": "86259394c1d96cfff768b6c7dbc3200d50db819b6294ba93ce88ee44bd3c6e2f",
    R + "early/validation/VP3.json": "a0dad5ff7704e631c6d745cc8f5910ede61fa52fb8a94a9173e7575d7330b73a",
    R + "early/validation/VP4_known_pairs.json": "a94104910679d50d69d5905a10f5085e30ffbd22527f1b8f24a5ea40fe825c0d",
    R + "early/validation/VP5_E_vs_I.json": "f97aa474b5807dea65b58dddf7b2d50faf48f6658e681ef3b94050e4fabf174c",
    R + "early/validation/VP7_prop5.json": "ee536bf1112725dbd13f1b8441fcee6d12ecbe2532eadfa280ad7f8c9aa979f1",
    R + "early/validation/VP8_generalisation.json": "ec246f8cae5bcd416b384d267f53416958bd92f84c04a03bac2cc535bf37c20f",
    R + "early/validation/VP9_known_answers.json": "49190c2cd3315192ea1d2c61602a3ed5eaee3647ce176d79936677760c507d54",
    R + "forecast/E/verdicts_forecast.json": "d0f7146bf5429aebaf5532683c44b736005b97daf6c67044b0425bda6839516b",
    R + "forecast/E/forecast_detail.json": "123546671236f5c469d009636805ffe1b3df8bbfa3d69badd4ddd3863dc11596",
    R + "indep/summary.json": "8733c8c1e699988cfaa1539de628142d36893b1b530768ceae96122cb32153bf",
    R + "indep/verdicts_pyt.json": "797934c2a80d84ad0b1d262f6bb9864496468a93e0f8d5cfbab243895cbcf4d9",
    R + "indep/REPORT.md": "c637dd82b8e431583d48a7698dc379918de0a548604ee45ffcd0cce756b410d0",
    R + "indep/validation.json": "f0fed1524d0bac72c09cb1cb961629f921429616519c1c3f402a3bdc082ab038",
    R + "indep/validation_vl.json": "1ca2f0b7a7460123af406b54644acfe802df2012dcabf19fee9cfca05ea72d52",
    R + "indep/validation_vp6_arh.json": "9a2e5a800d133c27adb65ee825b9b10cb224e13a3c2b7d9ee28d1c4df38797e2",
    R + "indep/validation_cross_1-2-3_prh_clip+l2+l2.json": "0632dbe68791fff5ef262151b2b04637d9dd7aacbb935ccc68ba4a10627a9786",
    R + "indep/validation_cross.json": "a229e77679ffd95fe7b194761db2a468e74a0f3abf6b8ba37d94a75d93040ae0",
    R + "indep/cross_EI.json": "433b323dd435cda94b958b594d8bc03fdd18826b1bfc652f8ff5f3b0cca77ee9",
    R + "indep/cross_FK.json": "6ffedb5fd97686d5fa7e03c0887b6a1fda4105ac8b0de7ac1c63b0b9aec41bcb",
    R + "indep/vp5_diag_I.json": "dd0ecbf2196cb857d9eacfbbdc3a1dc06498da9fff74f25c4ac0158e9c832cc6",
    R + "forecast/I/verdicts_forecast.json": "d78d0b2b10d71b0cdb0156f28394cc90973990ebe7a4a99755f3d836b585b1d7",
    R + "forecast/I/forecast_detail.json": "a6f8b1c92e4a9d4839a0f78887c1e505c2461ea8e8ca61a17bbc0714ebacc9d9",
    # review role (STATUS [pyt-axis-review] lines from 321 on)
    R + "review/E/verdicts_review.json": "13322147a87e645e182af0041d4e0e9b0b3493fa05ea8eaa4ae7e964075f4d78",
    R + "review/I/verdicts_review.json": "ada669480a22b6dd3c83e87e66b467da63b8991ad09fa255966fa114ee6d6687",
    R + "review/recomputed_verdicts.json": "c2f106b74dfe23308609536c49233a91411c2ee8ae9e66c8a2170f7fba5053ca",
    R + "review/REVIEW_REPORT.md": "73f0b6f10cf9da8f79a1f6fc17f71447cd2f9e24a40cbb2e761505b45527b7f6",
    R + "review/audit_F1c.json": "15f72fb92b8933a89607d62a274899a6210e6d5f12657f9e6627fe04cdb3b006",
    R + "review/cmp/RE_vs_RI_ALL.json": "2aeec2fac4fda18b86e6989175aed244f5e05113cc80bc28746d35b87beb832c",
    R + "review/role/order_audit.json": "9d34a1bee0bd6ced94fefc8a92361456c8f98c9f1704a581c9772b144d443d75",
    R + "review/role/reviewer_endpoints.json": "64c5f8159ca17edfe20b48f86fccbadc3bbc6c8f596ff9efd90a33791a5cbe20",
    R + "review/role/reviewer_endpoints_vs_R.json": "fd4b3e6b0a88b70fb4f0eb7aff84987845fee9d799195834d5a632a53b14549e",
}
RECORD_ONLY = [R + "logs/gpu_ledger.tsv"]  # hashed and copied, not enforced (a later step may append)
PRES = ("prh_clip+l2", "l2")
SIZES = ("70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def same(a, b):  # VERDICTS_SPEC_PYT.md section 0
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return math.isnan(fa) and math.isnan(fb)
        return abs(fa - fb) <= 1e-12 * max(abs(fa), abs(fb)) + 1e-15
    return type(a) is type(b) and a == b


def sentences(v, P="prh_clip+l2", x="s_cal"):
    """Plan section 13, frozen wording for the realised category, with numbers read from the headline values."""
    cat = v[f"K|{P}|{x}|category"]
    raw, cal = v[f"H|{P}|{x}|H1|R"], v[f"H|{P}|{x}|H1|C"]
    n7 = v[f"PP2ai|{P}|{x}|n_models_in_band"]
    g, h = v[f"D5|{P}|g"], v[f"D5|{P}|h"]
    words = {7: "seven", 6: "six", 5: "five", 4: "four", 3: "three", 2: "two", 1: "one", 0: "none"}
    num = {"raw": f"{raw:.2f}", "cal": f"{cal:.2f}", "raw_signed": f"{raw:+.2f}", "cal_signed": f"{cal:+.2f}", "n7": words.get(n7, str(n7)), "g": f"{g:.3f}", "h": f"{h:.4f}",
           "pH1": v[f"H|{P}|{x}|H1|p"], "pperm_count": round(v[f"H|{P}|{x}|H1|p_perm"] * 5040),
           "pmag": v[f"H|{P}|{x}|H1|p_mag"], "TrB": None}
    out = {"category": cat, "numbers": num,
           "rule": "plan section 13: frozen sentence of the realised category; wording that departs from it is labelled post hoc"}
    if cat != "C":
        out["note"] = "only the category-C sentences are spelled out by this script; see plan section 13 for the others"
        return out
    out["abstract_clause_C"] = (
        "On seven Pythia models, where a confirmatory test was fixed before any Pythia weight was downloaded, "
        f"calibration kept the trend, which moved from ${num['raw_signed']}$ to ${num['cal_signed']}$.")
    out["sec_exp_aggregation_paragraph_C"] = (
        "Three sizes leave each of these outcomes at $p\\ge1/6$, so we added a confirmatory test on the seven Pythia "
        "models from 70M to 6.9B, trained on the same data in the same order \\citep{biderman2023pythia}. Single-model "
        "features again predicted thresholds rising with size, and the thresholds lay in the predicted band for "
        f"{num['n7']} of the seven models. The raw scores rose faster, by a median ${num['g']}$ per doubling of "
        f"parameters against ${num['h']}$ for the thresholds, so calibration moved the layer-max CKA trend only from "
        f"${num['raw']}$ to ${num['cal']}$, an outcome allowed by Corollary~\\ref{{cor:aggregation}}.")
    out["limitations_C"] = (
        "Each size trend rests on one draw of evaluation inputs, and the one confirmatory family, Pythia, keeps the "
        "layer-max CKA trend under calibration, while the other families give exact one-sided $p$ of at least $1/120$ "
        "and $1/6$.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--vp5-ruling-line", type=int, default=None)
    a = ap.parse_args()
    if not a.dry and (os.path.exists(OUTF) or os.path.exists(DONE)):
        sys.exit("refusing: merged summary or DONE already exists")
    status = open(os.path.join(PROJ, "STATUS.md"), encoding="utf-8").read().splitlines()
    ruling = None
    if a.vp5_ruling_line is not None:
        n = a.vp5_ruling_line
        if not (REVIEW_START_LINE < n <= len(status)):
            sys.exit(f"refusing: ruling line {n} is not after the review ROLE START line {REVIEW_START_LINE}")
        ln = status[n - 1]
        if not re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \[", ln) or "[pyt-axis-review]" in ln[:40] or "V-P5" not in ln or not (
                "RULING" in ln or "AMENDMENT" in ln) or re.search(r"REJECT|RERUN|NOT ACCEPT", ln):
            sys.exit(f"refusing: STATUS line {n} is not a V-P5 ruling that accepts the float differences")
        ruling = {"status_line": n, "text": ln, "sha256_of_line": hashlib.sha256(ln.encode()).hexdigest()}
    elif not a.dry:
        print("refusing (exit 3): no --vp5-ruling-line; plan sections 7 and 10 need the lead's hashed ruling on V-P5 first",
              file=sys.stderr)
        sys.exit(3)
    ins = {}
    for rel, ref in INPUTS.items():
        got = sha(os.path.join(PROJ, rel))
        if got != ref:
            sys.exit(f"ABORT: {rel} sha256 {got} != logged {ref}")
        ins[rel] = got
    rec_only = {rel: sha(os.path.join(PROJ, rel)) for rel in RECORD_ONLY}
    L = lambda rel: json.load(open(os.path.join(PROJ, rel)))  # noqa: E731
    VE, VI = L(R + "early/verdicts_pyt.json")["v"], L(R + "indep/verdicts_pyt.json")["v"]
    FE, FI = L(R + "forecast/E/verdicts_forecast.json")["v"], L(R + "forecast/I/verdicts_forecast.json")["v"]
    sys.path.insert(0, os.path.join(PROJ, A))
    import verdict_keys as VK  # frozen contract (read only)
    diff_ei = [k for k in VK.required_keys("EI") if not same(VE.get(k), VI.get(k))]
    diff_fk = [k for k in VK.required_keys("FK") if not same(FE.get(k), FI.get(k))]
    if sorted(diff_ei) != sorted(KNOWN_EI) or diff_fk:
        sys.exit(f"ABORT: E != I differs from the known V-P5 picture: EI {diff_ei} FK {diff_fk}")
    rv = L(R + "review/recomputed_verdicts.json")
    au = L(R + "review/audit_F1c.json")
    oa = L(R + "review/role/order_audit.json")
    ec = L(R + "review/role/reviewer_endpoints_vs_R.json")
    rr = L(R + "review/cmp/RE_vs_RI_ALL.json")
    if not (rv["required_pass"] and au["pass"] and oa["pass"] and ec["pass"]) or sorted(rr["mismatching_keys"]) != sorted(KNOWN_EI):
        sys.exit("ABORT: a review check did not pass")
    v = dict(VI)
    v.update(FI)  # headline values of I (= E on every key within the contract tolerance except the 3 known D keys)
    for P, cat in EXPECT_CATEGORY.items():
        p, x = P.split("|")
        if v[f"K|{p}|{x}|category"] != cat:
            sys.exit(f"ABORT: category {v[f'K|{p}|{x}|category']} != {cat} recorded by the review role")
    HK = []
    for P in PRES:
        for x in ("s_cal", "unc"):
            HK += [f"K|{P}|{x}|{k}" for k in ("category", "F_reason", "premise", "R", "C", "nV", "M1", "M1_sizes", "M2", "M2_x",
                                              "M2_H3", "M2_H4", "M3", "M4", "M4_sizes", "M4_n_zero", "M5_extrap_sizes", "M5_A",
                                              "M5_A_sizes", "M5_B", "H2_reading", "H3_reading", "H4_reading", "H2_same_as_H1",
                                              "H2_counted", "H3_counted", "H4_counted", "kedge", "kedge_which", "h_consistent")]
            HK += [f"K|{P}|{x}|curve|tau={t}" for t in ("0.03", "0.06", "0.1", "0.15", "0.2")]
            HK += [f"H|{P}|{x}|H1|{k}" for k in ("member", "s", "D", "nV", "R", "C", "p_perm", "p_mag", "p", "confirmed", "ci_lo", "ci_hi", "kedge")]
            HK += [f"H|{P}|{x}|{h}|{k}" for h in ("H2", "H3", "H4") for k in ("D", "nV", "p", "p_lo", "p_hi", "degenerate", "tested", "p_holm", "holm_reject", "confirmed")]
            HK += [f"H|{P}|{x}|H2|{k}" for k in ("branch", "s_late", "p_perm", "p_mag")]
            HK += [f"H|{P}|{x}|{k}" for k in ("gate", "holm_order")]
            HK += [f"PP2a|{P}|{x}|verdict", f"PP2b|{P}|{x}|verdict", f"PP2b|{P}|{x}|n_decided", f"PP2b|{P}|{x}|x_correct",
                   f"PP2b|{P}|{x}|n_wrong", f"PP2b|{P}|{x}|n_undecided", f"PP2b|{P}|{x}|n_flip_fc_rawconc",
                   f"PP2ai|{P}|{x}|n_models_in_band", f"PP2ai|{P}|{x}|holds", f"PP2aii|{P}|{x}|n", f"PP2aii|{P}|{x}|x",
                   f"PP2aii|{P}|{x}|need", f"PP2aii|{P}|{x}|verdict", f"PP5a|{P}|{x}|n", f"PP5a|{P}|{x}|x", f"PP5a|{P}|{x}|verdict",
                   f"PP1|{P}|{x}|holds", f"fcnt|{P}|cka_biased|full|{x}|fminus", f"fcnt|{P}|cka_biased|full|{x}|fplus"]
            HK += [f"PP2ai|{P}|{x}|median|{s}" for s in SIZES] + [f"PP2ai|{P}|{x}|in_band|{s}" for s in SIZES]
            HK += [f"R|{P}|{x}|{k}" for k in ("fragile", "fragile_sizes", "five|D1", "five|p_perm", "five|carried", "grid5|D1",
                                              "grid5|dtr_mknn", "grid5|dtr_dcka", "rho0|D1")]
        HK += [f"tr|{P}|{M}|{Q}|{X}" for M in ("mknn", "cka_biased", "cka_debiased") for Q in ("full", "fixed_r0.5", "fixed_r1.0")
               for X in ("raw", "s_cal", "unc")]
        HK += [f"tr|{P}|cka_biased|ctm|{X}" for X in ("M", "M_unc", "TM")]
        HK += [f"D5|{P}|g", f"D5|{P}|h"] + [f"R|{P}|median_scal|{s}" for s in SIZES] + [f"R|{P}|n_scal_zero"]
        HK += [f"PP3|{P}|{r}|s_cal|desc" for r in ("0.5", "1.0")] + [f"readO|{P}|cka_biased", f"D3|{P}|nviol"]
        HK += [f"famtr|{P}|cka_biased|{X}|{f}" for X in ("raw", "s_cal", "unc") for f in ("BLOOMZ", "LLaMA", "OpenLLaMA", "Pythia")]
        HK += [f"famdtr|{P}|{M}|{x}|{f}" for M in ("mknn", "cka_debiased") for x in ("s_cal", "unc") for f in ("BLOOMZ", "LLaMA", "OpenLLaMA", "Pythia")]
        HK += [f"panel18|{P}|cka_biased|{X}|{k}" for X in ("raw", "s_cal") for k in ("r", "ci_lo", "ci_hi")]
    HK += ["G|n_pairs_read", "G|l2_run", "G|clip_equal_partB", "G|partB_sha256"]
    head = {k: v.get(k) for k in HK}
    miss = [k for k in HK if k not in v]
    if miss:
        sys.exit(f"ABORT: headline keys missing {miss[:10]}")
    imm = max(abs(v[f"famdtr|prh_clip+l2|{M}|{x}|{f}"]) for M in ("mknn", "cka_debiased") for x in ("s_cal", "unc")
              for f in ("BLOOMZ", "LLaMA", "OpenLLaMA", "Pythia"))
    vp5 = L(R + "early/validation/VP5_E_vs_I.json")
    xc = L(R + "indep/validation_cross_1-2-3_prh_clip+l2+l2.json")
    doc = {
        "schema": "PYT-AXIS MERGED SUMMARY v1",
        "what": "Pythia 70M-6.9B x 17 ViTs layer-max axis (S1.2 fourth size axis): merged summary, plan F7 / S8",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "writer": {"script": A + "review_role/merge_pyt_summary.py", "sha256": sha(os.path.abspath(__file__)), "role": "pyt-review",
                   "note": "NEW script of the review role (not in the S0b manifest); copies only, no score recomputed"},
        "labels": {
            "confirmatory": "H1 (gatekeeper, alpha 0.05) and, behind it, Holm over {H2a, H3, H4}, for prh_clip+l2 / biased CKA / layer-max / s_cal ONLY (plan 5.4)",
            "secondary": "l2 and uncensored versions of every statistic: secondary, descriptive; everything outside plan 5.4 is descriptive",
            "vp5": "V-P5 fails by the letter on float tolerance only; the lead's ruling is copied under 'vp5_ruling'",
            "posthoc": "any paper wording departing from plan section 13 is post hoc"},
        "vp5_ruling": ruling,
        "vp5_status": {
            "pair_level": {"checks": xc.get("part1_pair_level", {}).get("n_checks"), "fail": xc.get("part1_pair_level", {}).get("n_fail"),
                           "fields": "cka_debiased.c_alpha 121/3094, cka_debiased.null_mean 5/3094, cka_debiased.full.entry_c_alpha_at_argmax 18/238, cka_debiased.ctm.c_alpha_at_argmax 18/238, cka_biased.c_alpha 1/3094 (tolerance 1e-12 rel + 1e-15 abs); 0 in mKNN, p, argmax, ties",
                           "E_side_record": R + "early/validation/VP5_E_vs_I.json", "E_side_pass": vp5.get("pass")},
            "verdict_level_EI_mismatching_keys": diff_ei, "FK_mismatching_keys": diff_fk,
            "decision_effect": "0 order, censoring or tie changes over 357 comparisons x 3 metrics x 5 quantities x 2 pre (review/role/reviewer_endpoints.json vp5_effect); R-E vs R-I differ on the same 3 keys only"},
        "inputs_sha256": ins, "record_only_sha256": rec_only,
        "headline": head,
        "headline_source": "values of implementation I (forecast/I for F, K); E equal on every EI and FK key within 1e-12 rel + 1e-15 except the 3 known section-D keys (checked here)",
        "paper_sentences_s13": sentences(v),
        "immunity_max_four_families_prh": imm,
        "checks": {
            "R_review": {"required_pass": rv["required_pass"],
                         "comparisons": [{k: c.get(k) for k in ("a", "b", "sections", "n_compared", "n_mismatch", "pass", "required")} for c in rv["comparisons"]]},
            "F1c_audit": {k: au.get(k) for k in ("pass", "n_files", "n_born_before_partB", "min_birth", "partB_status_utc")},
            "order_audit": {"pass": oa["pass"], "checks": {k: c["ok"] for k, c in oa["checks"].items()}},
            "reviewer_endpoints_vs_R": {k: ec[k] for k in ("n_keys", "mismatch_vs_R_I", "mismatch_vs_R_E", "pass")}},
        "deviations_and_notes": [
            "V-P5 (E vs I) fails by the letter on float tolerance only; ruled as copied under vp5_ruling. The I role read its own REPORT.md, with the category, before V-P5 ran, and the I and review reports gave the category to the lead before the ruling, so the ruling was made with the category known.",
            "Amendment A-G0 (STATUS line 249): a new post-freeze G0 smoke driver analysis/s12_layermax_pythia/g0_smoke/g0_smoke_pyt.py 7c9e9595, recorded with its hash before its first run; f0 was measured on the bloomz-560m E2 workload (nominal 0.5 GPU-min) instead of the plan's E/I GPU smoke (nominal 3.0), which the frozen code could no longer run. Part B line 285 noted it still awaited the lead.",
            "E's S6 GPU steps (09:21:24Z-09:22:44Z) ran while the CPU-only S2.3 amendment-A3 analysis ran (09:20:47Z-09:25:17Z); no S2.3 GPU process existed (A2 GPU work ended 09:09:34Z, 11 min of no compute process before). A3 failed rc 5 on its own analysis error, fixed by A4 (P5b zero MC denominator).",
            "2.8B was loaded from the step143000 file because the main .bin differs (E1-PROV fail, frozen fallback; STATUS lines 267, 270).",
            "E and I each wrote a primary-only summary before l2 (kept as *.prev-*); E's were copied aside at 09:55:32Z (birth = copy time, mtime 09:28:29Z, hashes logged in line 304 at 09:28:55Z, before any I pair file).",
            "I left a 0-byte /tmp/.x_none on ds2213 (no-delete rule); I's STATUS line 315 has the typo 'prep_meta 0b fac07707'; I's integer supplement (indep/validation_cross.json) shows pass=false only through three over-specified float-encoding checks (<= 2 ulp).",
            "R flags float32 copies of null means in 9 of 238 E pair files (relative <= 7.3e-6 against the inherited 1e-6 flag) and 129,193 mKNN entries stored as count/10240 to one ulp; no decision reads either.",
            "Helper scripts outside the S0b manifest (none computes a verdict): analysis/s12_layermax_pythia/run_I/*.py (I role), review_role/*.py (this role; reviewer_endpoints_pyt.py v1 failed with a TypeError, launcher pyt_review_endpoints_1 rc 1, and v2 ran; order_audit_R.py v1 dated E's copied primary files by birth and was superseded by v2); E's V-P5 magnitude diagnostic ran from stdin.",
            "E and I outputs were left at mode 0664; their sha256 are logged in STATUS and re-verified by the review role at start and end.",
            "Threshold curve (plan 5.6): at tau = 0.03 the primary category would be F, at tau >= 0.06 it is C; the frozen tau is 0.10, and the knife-edge flag is not set (D1 CI [-0.046, -0.023])."],
        "embedded": {"E_summary": L(R + "early/summary.json"), "I_summary": L(R + "indep/summary.json"),
                     "verdicts": {"E": VE, "I": VI, "forecast_E": FE, "forecast_I": FI,
                                  "review_E": L(R + "review/E/verdicts_review.json")["v"], "review_I": L(R + "review/I/verdicts_review.json")["v"]}},
    }
    if a.dry:
        print(json.dumps({"vp5_ruling": ruling, "category": head["K|prh_clip+l2|s_cal|category"],
                          "H1": {k: head[f"H|prh_clip+l2|s_cal|H1|{k}"] for k in ("R", "C", "D", "p_perm", "p_mag", "p", "confirmed")},
                          "EI_diff": diff_ei, "FK_diff": diff_fk, "immunity_max": imm,
                          "sentences": doc["paper_sentences_s13"]}, indent=1))
        print("DRY: nothing written")
        return 0
    tmp = OUTF + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    os.replace(tmp, OUTF)
    os.chmod(OUTF, 0o444)
    hs = sha(OUTF)
    with open(DONE, "w") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "role": "pyt-review",
                            "layermax_pythia_summary.json": hs, "vp5_ruling_status_line": ruling["status_line"],
                            "statement": "Pythia axis S8 complete: V-P1-V-P4 and V-P6-V-P9 pass; V-P5 float-only, ruled at the cited "
                                         "STATUS line; R = E and R = I (0 mismatches); reviewer endpoints equal R-I; order audit pass. "
                                         "Primary instance category " + head["K|prh_clip+l2|s_cal|category"] + "."}) + "\n")
    os.chmod(DONE, 0o444)
    print("summary", os.path.relpath(OUTF, PROJ), hs, os.path.getsize(OUTF))
    print("DONE", os.path.relpath(DONE, PROJ), sha(DONE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
