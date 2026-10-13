#!/usr/bin/env python
"""OpenLLaMA axis, role oll-review: merged summary and DONE (plan ANALYSIS_PLAN_OPENLLAMA.md 4f18d710, section 4 F7
and section 11 step S8, which the run roles left open). NEW script of the review role, written after R passed (not in
the S0b manifest; logged as such). NO score is recomputed: every value is copied from files whose sha256 equals the
value logged in STATUS by the role that wrote it, and E = I is checked again on the headline keys with the tolerance of
VERDICTS_SPEC.md section 0. Writes, in this order and never overwriting:
  results/S1_2_layermax_openllama/layermax_openllama_summary.json  (chmod 444)
  results/S1_2_layermax_openllama/DONE                              (chmod 444, last)

usage: venv/bin/python -I analysis/s12_layermax_openllama/review_role/merge_ol_summary.py [--dry]
"""
import hashlib
import json
import os
import sys
import time

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
RES = os.path.join(PROJ, "results", "S1_2_layermax_openllama")
R = "results/S1_2_layermax_openllama/"
A = "analysis/s12_layermax_openllama/"
OUTF = os.path.join(RES, "layermax_openllama_summary.json")
DONE = os.path.join(RES, "DONE")
INPUTS = {  # path: sha256 logged in STATUS (full where STATUS gives it)
    A + "ANALYSIS_PLAN_OPENLLAMA.md": "4f18d710beda45492b1e191f74580bfc4ef53f6dafe791ad245ff1cf335f3240",
    A + "PARTB_PREDICTIONS_OL.md": "f3527d09fedf59270180e64af7265d26ff59039031725e7b8d5e38e671208865",
    A + "S0B_CODE_MANIFEST.json": "b2b553622a97e042488ace043adde592c1b6f1128bddf252f8a865bc9dafed4d",
    R + "partB/partB_features_ol.json": "e1104bd71e5eff9ced65ed402a61c187590bce5dee306c25ee2f64b18fdd7e51",
    R + "partB/clip_thresholds_partB_ol.json": "f9805138afef0bba302bec863387e611ca3b57e5fe8ae2c8cbcaa91879c0c908",
    R + "extraction/EXTRACTION_SUMMARY.json": "a343f3a55ad3165e5ef18fd877943b8a23068040726461735201014835a6ee26",
    R + "early/summary.json": "b302119c4fff9ead88f7418fa4e01126f2b621f653691a7cc6cf2e328925890d",
    R + "early/verdicts_ol.json": "0c7f34b59937a6ed6ed154570126c76f726d559f8962c3e2fc1d9aa8f0bf1f8f",
    R + "early/REPORT.md": "85a5f2c83cf02b2d168083dfd3b18141ee043f8822434f61712a09e81a815ed5",
    R + "indep/summary.json": "7b60cf8c8ca646a263da658220f3f3eaf7477dc28b96e18b5df95d6bd7dd0fd9",
    R + "indep/verdicts_ol.json": "58fcc80b6ffca7cecc44cf1951e0bbf6d6e611fe6305b28c4a04ade78ade6967",
    R + "indep/REPORT.md": "4a6bdc4e10114b61125d369e9c6e49a6f2a4793fd0205738a9787815529ba389",
    R + "forecast/E/verdicts_forecast.json": "54055939adfc22dd9bf0e89fe3fa138b4b143522a786e5db50ff7bb38ba82263",
    R + "forecast/I/verdicts_forecast.json": "87fc0ab8aa80cf6f2620329a8e159524daa836961889f893ec1b294856097b36",
    R + "joint/E/verdicts_joint.json": "be2e6ee59f5187d7665bae9e52b3248a66ba5a6bada6a8d805831517d888fe5e",
    R + "joint/I/verdicts_joint.json": "204ed4acab9ed39c143bd668de8d8a827c6ca08a4b1e2b71060694dc777ec00a",
    R + "early/validation/VO1_VO2_knn_tables.json": "ad794dce63134f6a361a00537df84393feb83dd56e0c965c2f1c2e8713a2e842",
    R + "early/validation/VO3_VO4a.json": "996c442f6c7f0f98e00533ee151e868d9dd26c69323e08483e693f082600b6be",
    R + "early/validation/VO4b_3b7b.json": "0d680fa310a5d6f4ff1a4ada63d6b331dfaf5ffac6d16f2a9ade595a5368e09f",
    R + "early/validation/VO7_prop5.json": "acdea068b81e6771a97bd668e0d3e7c0f34f47f4be189b2430396b41b5cfed58",
    R + "early/validation/VO5_pairs.json": "db8103696e4f6c95ab0c37ec09621a2fa262b90655559c74ba74d51ff89c57aa",
    R + "early/validation/VO5_verdicts.json": "6c9f9af35dbffa9f9d87ecfe80ee5f8a4be16f1569606321865f7eee6ad5a4c9",
    R + "indep/validation_vl.json": "d15f96af6cfc3e36199e247cf0bb34b0cd6cf2b4e4d73c3dde13d152c86bbb6f",
    R + "indep/validation.json": "e553a37a47646a7db2002a6ba22f179187e2f9c7036593f5ba19d24fb840cf8e",
    R + "indep/validation_vo6_arh.json": "9763236bfa9ceb28ff100461b94520adeb4dedeebd2e236fa716640a357f2f45",
    R + "indep/validation_cross.json": "de95bdb181882fdbddcad1b4eb9331df847f8bc7f9f59176f50ff8063798ae41",
    R + "indep/exact_mknn_vs_E.json": "b0fbd361596a4bfc81cf9ab57014d883dc62f423b1ea503c5b8bf77f2ac02fbd",
    # review role outputs (logged in the [oll-axis-review] STATUS lines)
    R + "review/E/verdicts_review.json": "07cc6b00c59a1b058e5254de5bd1e5204a74b6049c6b35e6d74d864e8d24c65f",
    R + "review/I/verdicts_review.json": "f4e988042f35a8c9b8b5846f99b6b61deafbf4deedb09350d1bae983b6dcd28b",
    R + "review/recomputed_verdicts.json": "ec750fdebd260711355e61343a56930decbe363192d9b2c3caf2079dcffe697f",
    R + "review/REVIEW_REPORT.md": "5bb1b3ee2d1d4712dcbf397b5f59e1c4703086f5720e53fd5352ee63d74f7a1c",
    R + "review/audit_F1c.json": "74d61641f61d4a745000c4810b4ecc04c6cd487e2220bf845f8e924771d1a0cd",
    R + "review/role/order_audit.json": "a0cef151dadb8614c8e06531c1eacb722f0f804c2eac86559b4b7dd4b710dc46",
    R + "review/role/endpoints_review.json": "2f6fc6eabcd6d00f0371dbeeb1b16edddb081c2009b0fb51fb3630520c892b28",
    R + "review/role/rehash_start.json": "361b3a86b50768782dd93964dd6e01ddddb85926a0b712f16e6b4dd978c4bc1d",
    "results/S1_2_layermax_llama/layermax_llama_summary.json": "db9242f9c5734434c8e5eb0d23204b9e755c4d49cd6636f1d7da58b7e8fc7e7e",
}
HEADLINE = [  # (section file, key); primary instance first, then l2 (secondary)
    *[("EI", k) for k in (
        "tr|prh_clip+l2|cka_biased|full|raw", "tr|prh_clip+l2|cka_biased|full|s_cal", "tr|prh_clip+l2|cka_biased|full|unc",
        "dtr|prh_clip+l2|cka_biased|full|s_cal", "dtr_ple|prh_clip+l2|cka_biased|full|s_cal",
        "dtr|prh_clip+l2|cka_biased|full|unc", "dtr_ple|prh_clip+l2|cka_biased|full|unc",
        "chg|prh_clip+l2|cka_biased|full|s_cal|new|c2d", "chg|prh_clip+l2|cka_biased|full|s_cal|new|d2c",
        "chg|prh_clip+l2|cka_biased|full|s_cal|pilot|c2d",
        "fnew|prh_clip+l2|cka_biased|full|s_cal|F", "fnew|prh_clip+l2|cka_biased|full|unc|F",
        "PO1|prh_clip+l2|holds", "tr|prh_clip+l2|mknn|full|raw", "tr|prh_clip+l2|mknn|full|s_cal", "tr|prh_clip+l2|mknn|full|unc",
        "PO2|prh_clip+l2|s_cal|cat", "PO2|prh_clip+l2|s_cal|kedge", "PO2|prh_clip+l2|unc|cat",
        "PO2ai|prh_clip+l2|median", "PO2ai|prh_clip+l2|in_band", "PO2ai|prh_clip+l2|x", "PO2ai|prh_clip+l2|n",
        "PO2aii|prh_clip+l2|n", "PO2aii|prh_clip+l2|x", "PO2aii|prh_clip+l2|need", "PO2aii|prh_clip+l2|verdict",
        "PO3|prh_clip+l2|0.5|s_cal|desc", "PO3|prh_clip+l2|1.0|s_cal|desc",
        "dtr|prh_clip+l2|cka_debiased|full|s_cal", "dtr|prh_clip+l2|cka_debiased|full|unc",
        "PO4|prh_clip+l2|s_cal|holds", "PO4|prh_clip+l2|unc|holds", "PO4|prh_clip+l2|arh|holds",
        "arh|11|prh_clip+l2|cka_debiased|raw|arh|r", "arh|11|prh_clip+l2|cka_debiased|s_cal|arh|r",
        "dtr|prh_clip+l2|cka_biased|ctm|M", "dtr|prh_clip+l2|cka_biased|ctm|M_unc",
        "PO5a|prh_clip+l2|n", "PO5a|prh_clip+l2|x", "PO5a|prh_clip+l2|verdict", "PO5|prh_clip+l2|M|cat",
        "d3cal|prh_clip+l2|cka_biased", "d3sel|prh_clip+l2|cka_biased", "D3|prh_clip+l2|nviol",
        "plugin|prh_clip+l2|diff", "plugin|prh_clip+l2|within", "readO|prh_clip+l2|cka_biased",
        "arh|11|prh_clip+l2|cka_biased|raw|arh|r", "arh|11|prh_clip+l2|cka_biased|s_cal|arh|r",
        "arh|11|prh_clip+l2|mknn|raw|arh|r", "arh|11|prh_clip+l2|mknn|s_cal|arh|r",
        "dd|11|prh_clip+l2|cka_biased", "dd|11|prh_clip+l2|cka_biased|ci_lo", "dd|11|prh_clip+l2|cka_biased|ci_hi",
        "dd|11|prh_clip+l2|cka_biased|pge", "dd|10|prh_clip+l2|cka_biased", "dd|8|prh_clip+l2|cka_biased",
        "rule7a|prh_clip+l2", "size_not_reproduced|prh_clip+l2", "fp6|prh_clip+l2|raw_cat", "fp6|prh_clip+l2|cat",
        "G|llama_pairs_equal_merged_summary", "G|rerun_equals_frozen_3b7b", "G|n_pairs_read",
        "tr|l2|cka_biased|full|raw", "dtr|l2|cka_biased|full|s_cal", "dtr|l2|cka_biased|full|unc",
        "PO1|l2|holds", "PO2|l2|s_cal|cat", "PO2ai|l2|median", "PO2ai|l2|x", "PO2aii|l2|verdict", "PO4|l2|s_cal|holds",
        "PO5|l2|M|cat", "dd|11|l2|cka_biased", "dd|11|l2|cka_biased|ci_lo", "dd|11|l2|cka_biased|ci_hi", "rule7a|l2",
        "fp6|l2|cat")],
    *[("F", k) for k in (
        "PO2b|prh_clip+l2|s_cal|n_decided", "PO2b|prh_clip+l2|s_cal|x_correct", "PO2b|prh_clip+l2|s_cal|n_undecided",
        "PO2b|prh_clip+l2|s_cal|cp_lo", "PO2b|prh_clip+l2|s_cal|verdict",
        "PO2b|prh_clip+l2|unc|n_decided", "PO2b|prh_clip+l2|unc|x_correct", "PO2b|prh_clip+l2|unc|verdict",
        "PO2b|prh_clip+l2|s_cal|h=0.0003|n_decided", "PO2b|prh_clip+l2|s_cal|h=0.0003|x_correct",
        "PO2b|prh_clip+l2|s_cal|h=0.002|n_decided", "PO2b|prh_clip+l2|s_cal|h=0.002|x_correct",
        "PO2b|l2|s_cal|n_decided", "PO2b|l2|s_cal|x_correct", "PO2b|l2|s_cal|verdict", "PO2b|l2|unc|verdict")],
    *[("J", k) for k in (
        "J|label", "J|provenance", "J|s_cal|dTr|B", "J|s_cal|dTr|L", "J|s_cal|dTr|O", "J|s_cal|Jmin", "J|s_cal|pmin",
        "J|s_cal|pmin_product", "J|s_cal|Jmean", "J|s_cal|pmean", "J|s_cal|nO", "J|s_cal|cond_size",
        "J|unc|dTr|B", "J|unc|dTr|L", "J|unc|dTr|O", "J|unc|Jmin", "J|unc|pmin", "J|unc|nO", "J|unc|cond_size")],
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    return abs(a - b) <= 1e-12 * max(abs(a), abs(b)) + 1e-15


def passes(d, depth=0, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if "pass" in str(k).lower() and not isinstance(v, (dict, list)):
                out[pre + str(k)] = v
            elif depth < 2 and isinstance(v, dict):
                out.update(passes(v, depth + 1, pre + str(k) + "."))
    return out


def main():
    dry = "--dry" in sys.argv
    if not dry and (os.path.exists(OUTF) or os.path.exists(DONE)):
        sys.exit("refusing: merged summary or DONE already exists")
    ins = {}
    for rel, ref in INPUTS.items():
        got = sha(os.path.join(PROJ, rel))
        if got != ref:
            sys.exit(f"ABORT: {rel} sha256 {got} != logged {ref}")
        ins[rel] = got
    L = lambda rel: json.load(open(os.path.join(PROJ, rel)))  # noqa: E731
    VE, VI = L(R + "early/verdicts_ol.json")["v"], L(R + "indep/verdicts_ol.json")["v"]
    FE, FI = L(R + "forecast/E/verdicts_forecast.json")["v"], L(R + "forecast/I/verdicts_forecast.json")["v"]
    JE, JI = L(R + "joint/E/verdicts_joint.json")["v"], L(R + "joint/I/verdicts_joint.json")["v"]
    src = {"EI": (VE, VI), "F": (FE, FI), "J": (JE, JI)}
    head, diff = {}, []
    for sec, k in HEADLINE:
        a, b = src[sec][0][k], src[sec][1][k]
        if not same(a, b):
            diff.append([k, a, b])
        head[k] = b  # value of I (equal to E within the VERDICTS_SPEC tolerance)
    if diff:
        sys.exit(f"ABORT: E != I on headline keys {diff}")
    rv = L(R + "review/recomputed_verdicts.json")
    er = L(R + "review/role/endpoints_review.json")
    oa = L(R + "review/role/order_audit.json")
    if not (rv["required_pass"] and er["pass"] and oa["pass"]):
        sys.exit("ABORT: review checks did not all pass")
    vfiles = ["early/validation/VO1_VO2_knn_tables.json", "early/validation/VO3_VO4a.json", "early/validation/VO4b_3b7b.json",
              "early/validation/VO7_prop5.json", "early/validation/VO5_pairs.json", "early/validation/VO5_verdicts.json",
              "indep/validation_vl.json", "indep/validation.json", "indep/validation_vo6_arh.json", "indep/validation_cross.json",
              "indep/exact_mknn_vs_E.json"]
    validation = {f: passes(L(R + f)) for f in vfiles}
    for f in ("early/validation/VO1_VO2_knn_tables.json", "early/validation/VO3_VO4a.json"):
        validation[f]["summary"] = L(R + f).get("summary")
    doc = {
        "schema": "OL-AXIS MERGED SUMMARY v1",
        "what": "OpenLLaMA 3B/7B/13B x 17 ViTs layer-max axis (S1.2 third size axis): merged summary, plan F7 / S8",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "writer": {"script": A + "review_role/merge_ol_summary.py", "sha256": sha(os.path.abspath(__file__)),
                   "role": "oll-review", "note": "NEW script of the review role (not in the S0b manifest); copies only, no score recomputed"},
        "labels": {
            "confirmatory": "NONE on this axis (author decision R.1.2); every OpenLLaMA-axis result is descriptive; smallest one-sided p of any OpenLLaMA-only statistic is 1/6",
            "primary_instance": "prh_clip+l2, biased CKA, layer-max, s_cal (uncensored secondary)",
            "secondary": "l2 readings are secondary; debiased-CKA readings are exploratory",
            "J": "EXPLORATORY; printed with n_O/6, the conditional size and the labels 'BLOOMZ hypothesis-generating; BLOOMZ fails B4; BLOOMZ and LLaMA known before this plan'",
            "posthoc": "reviewer sensitivity numbers under 'reviewer_posthoc' are POST HOC and decide nothing"},
        "inputs_sha256": ins,
        "headline": head,
        "headline_source": "values of implementation I; E equal on every headline key within 1e-12 rel + 1e-15 (checked here)",
        "manuscript_rule_7a": {
            "rule": "plan section 7(a), author pre-commitment R.1.5: ViT-CI lower bound of dDelta > 0 -> 'replicated the direction'; interval contains 0 -> 'direction not established'; U < 0 goes to the author",
            "dDelta_11LM": head["dd|11|prh_clip+l2|cka_biased"],
            "ci": [head["dd|11|prh_clip+l2|cka_biased|ci_lo"], head["dd|11|prh_clip+l2|cka_biased|ci_hi"]],
            "outcome": head["rule7a|prh_clip+l2"],
            "size_not_reproduced": head["size_not_reproduced|prh_clip+l2"],
            "panel_clause": "a panel that still lacks LLaMA-65B (OpenLLaMA-13B is now in it), plan section 7(a)",
            "secondary_l2_not_deciding": head["rule7a|l2"]},
        "checks": {
            "E_vs_I_V-O5": validation["early/validation/VO5_verdicts.json"],
            "R_review": {"required_pass": rv["required_pass"],
                         "comparisons": [{k: c.get(k) for k in ("a", "b", "sections", "n_compared", "n_mismatch", "pass", "required")}
                                         for c in rv["comparisons"]],
                         "F1c_audit": {k: rv["audit"].get(k) for k in ("pass", "n_files", "n_born_before_partB", "min_birth", "partB_status_utc")}},
            "reviewer_endpoints": {k: er[k] for k in ("n_compared", "n_mismatch", "pass", "pair_values_I_summary_vs_E_json")},
            "order_audit": {"pass": oa["pass"], "status_times": oa["status_times"],
                            "checks": [{"check": c["check"], "pass": c["pass"]} for c in oa["checks"]]},
            "validation_pass_flags": validation},
        "reviewer_posthoc": {k: er.get(k) for k in ("means", "PO2ai_range", "PO2b_table", "PO2b_h_curve", "desc_7b_to_13b",
                                                     "boot_frac_le_0", "posthoc_sensitivity")},
        "deviations_and_notes": [
            "E saw the EXPLORATORY J values in the joint_summary log before V-O5 passed (STATUS 19:55:46Z); no value changed, J carries no decision.",
            "I misstated the mKNN tie split in STATUS 20:01:22Z ('13 + 2'); corrected at 20:01:36Z: 3 cells (5 comparisons) prh_clip+l2, 12 cells (18 comparisons) l2, none in full or ctm.",
            "Helper scripts written after CODE FROZEN (read-only checks, hashed in STATUS, not in the S0b manifest): partB_verify/verify_partb_ol.py (8eca2017), indep_run/{hashcheck_I.py af2fdd7f, exact_mknn_check_I.py 4592e809, exact_mknn_vs_E.py 38561cc3}, early/interpretation.txt (post hoc text, f4be32a4), review_role/{rehash_review.py, order_audit.py, endpoints_review.py, merge_ol_summary.py}.",
            "Part B file name PARTB_PREDICTIONS_OL.md (plan and guard) used instead of the lead's PARTB_PREDICTIONS.md.",
            "Under l2, B_fd(1.0) came out material (Tr -1, knife edge on the delta_b curve) although plan B4 said B_fd cannot be material (computed for prh_clip+l2); P-O3 stays descriptive (plan sections 9, 10).",
            "E outputs (early/summary.json, verdicts_ol.json, REPORT.md, hashes.json, forecast/E, joint/E) were left at mode 0664, not 0444; their sha256 are logged in STATUS and re-verified equal by the review role at start and end.",
            "E2 was run under the lead's stricter stop rule (bitwise) instead of the plan's 1e-6; it was bitwise equal, so S-E2 was not triggered.",
            "Seven GPU steps (2.24 GPU-min) ran before CODE FROZEN as S0b tests on stand-in data (llama-13b features under a synthetic slug); no OpenLLaMA-13B byte existed then (HF cache born 19:23:25Z, after CODE FROZEN 19:21:07Z).",
            "R recorded l2|kappa_recomputed_vs_partB_maxabs 9.1e-5: R's informational recomputation of the l2 boundaries used the primary band o = 0.0019, while Part B uses l2's own band o = 0.002 [0.001, 0.003]; all decisions read the frozen Part B boundaries, and the reviewer's recomputation with each instance's own band gives 0.",
            "R noted float32-rounded null means in some E pair json files (max rel 1.75e-6 / 3.95e-6); no decision reads them (E decisions use the npz arrays; E = I at 1e-12 on every verdict key).",
            "Post hoc (reviewer): over 50 other bootstrap keys the 11-LM dDelta lower bound stays <= 0 (max -0.0004); every leave-one-ViT-out dDelta is positive (0.088 to 0.161); about 4% of pinned bootstrap draws are <= 0."],
        "embedded": {
            "E_summary": L(R + "early/summary.json"), "I_summary": L(R + "indep/summary.json"),
            "verdicts": {"E": VE, "I": VI, "forecast_E": FE, "forecast_I": FI, "joint_E": JE, "joint_I": JI}},
    }
    if dry:
        print(json.dumps({k: doc[k] for k in ("manuscript_rule_7a",)}, indent=1))
        print(json.dumps({k: v for k, v in list(head.items())[:20]}, indent=1))
        print("DRY: nothing written")
        return 0
    tmp = OUTF + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    os.replace(tmp, OUTF)
    os.chmod(OUTF, 0o444)
    hs = sha(OUTF)
    with open(DONE, "w") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "role": "oll-review",
                            "layermax_openllama_summary.json": hs,
                            "statement": "OpenLLaMA axis S8 complete: V-O1..V-O7 pass, E = I (V-O5), R = E = I (0 mismatches), "
                                         "reviewer endpoints 0 mismatches, order audit pass. No confirmatory endpoint; "
                                         "section 7(a): " + head["rule7a|prh_clip+l2"] + "."}) + "\n")
    os.chmod(DONE, 0o444)
    print("summary", os.path.relpath(OUTF, PROJ), hs, os.path.getsize(OUTF))
    print("DONE", os.path.relpath(DONE, PROJ), sha(DONE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
