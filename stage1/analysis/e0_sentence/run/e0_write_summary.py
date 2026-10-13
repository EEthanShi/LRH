#!/usr/bin/env python3
"""E0 closing step, role e0-run (ANALYSIS_PLAN_E0.md v1.0, section 11 step 7; outputs as section 14;
readings bound by section 13).

Modes
  summary --dry-run   build E0_summary.{json,md} and print the md to stdout; write nothing
  summary --write     write results/E0_sentence/E0_summary.json and E0_summary.md (O_EXCL; refuses if
                      either exists), then chmod 444
  done                re-verify the two summary hashes and write results/E0_sentence/DONE (O_EXCL), chmod 444

Reads only. Every input is checked against the sha256 logged in STATUS.md (lines 328-341) before use.
Every number is read from VERDICTS_E0.json, the e0-review outputs or the e0-indep (V-f) outputs; no
statistic is recomputed here. The one count made here is the caption-source recheck (a factual count on
items.parquet and captions_index.parquet). Requires STATUS lines '[e0-indep] AGREE' and '[e0-review] REVIEW'
and no '[e0-indep] DISAGREE'.
"""
import argparse
import datetime
import hashlib
import json
import os
import statistics
import sys
import time

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
R = "results/E0_sentence"
A = "analysis/e0_sentence"
VF = f"{A}/indep/run_20261010T143557Z"
OUT_JSON = f"{R}/E0_summary.json"
OUT_MD = f"{R}/E0_summary.md"
OUT_DONE = f"{R}/DONE"

# sha256 as logged in STATUS.md (lines 328-341); each is verified before the file is read.
EXPECTED = {
    f"{A}/ANALYSIS_PLAN_E0.md": "e859c6ea52f941b18ca27b83150f489dded0085a4d99218d909b36c19603f0ee",
    f"{A}/ANALYSIS_PLAN_E0_ADDENDUM_1.md": "dcf0ecfc9b3e79063782a9e0d51af7db3e91c4719dee5ab8cec9490cb2509aad",
    f"{A}/CODE_MANIFEST_E0.json": "6b0111c85aa5416b1d7c97b7f75fe2fe413ce8e2741bdb4e852e43e707664b13",
    f"{A}/addendum1/e0_evaluate_add1.py": "d0c1ce4af6cff66b099ed9ab1c5dd074bca9f7f2d99324b4d3d0c2805e521da8",
    f"{R}/extraction/EXTRACTION_E0.json": "27bdcb149f1b386be886b9e5539b2b72b352c54be1867a4a42f9a87c6416f8de",
    f"{R}/eval/ctr_calibration.json": "33d1e796fa1a80185b053e237fd93b58e1ccf3f72e84506ecedb484948d63da2",
    f"{R}/guard/GUARD_CHECK_20261010T140640Z.json": "f5099bd2b48b3b4d6d31a235710c99b81913cceee4d0e201feb5a28e7c20af7d",
    f"{R}/diag/diag_singlemodel.json": "e60bf07c222312f6073f818211423270d3edf29656ada576e85d7863d6a35c7f",
    f"{R}/VERDICTS_E0.json": "5b04e46098e39056f6e9b2d534d53fa54777b1670ccfa68442cb7ef3fd1568a7",
    f"{R}/eval/E0_tables.md": "22b6b04b7b400b75a6c936306525d32bf90256dceb275a9a606d3a455e268a66",
    f"{R}/summary.json": "b1d6fcddeb706ba607246207d8d70b6ebc215d8f412f76877955306f1ea822b2",
    f"{R}/REPORT.md": "a7ec345754e034a30bbf89f87678678169ec2bb74974e0994d4e75c6f8f7cdb4",
    f"{A}/review/e0_review_recompute.py": "c0fcf8d243e2c18450ecc0522fb746350c5b2ff56459afcf0bb725b4c51a7527",
    f"{A}/review/review_recompute_20261010T141910Z.json": "414fa020230f0e84f6f3cffb2ac2cbb7b7600387570b79b4af2950b1be344d9a",
    f"{A}/review/audit_20261010T142040Z.json": "17e837e2f44fbf9ed95dd42151cf2b28bbe09a11ad9409fcb3812f3dcd487e58",
    f"{A}/review/e0_review_explore.py": "583cd61ae372c183c05e523fac011759f843927dad67bd041bd1df900d2d262e",
    f"{A}/review/review_explore_20261010T142122Z.json": "abbe5cf2c2395f1b18636a27071410d3b9c787ee2eeac05e682cabda3d397e1b",
    f"{A}/review/e0_review_paper_numbers.py": "0be7d4099d90e292487aea4bdd1b40ac3539e9e74fcbd4dcdf9045e89e3c7068",
    f"{A}/review/review_paper_numbers_20261010T142548Z.json": "4bbe71cd0206c612ff58bd1c4c4b23ce065c971c4a15cb934e4091a706843286",
    f"{VF}/INDEP_MANIFEST.json": "2f636640ef8f539e3ee1cc44130b37cff694a212cefa5567f80669aae89f933e",
    f"{VF}/vf_obs.parquet": "0f153b4f778c4eb9d4d8929674bec36358528da97c7ded8806e41a6aa2f5ce37",
    f"{VF}/vf_compare_obs.json": "fc998a323a73aa37f7bc85b1e8e75a5653d96c984e67e039d04beab4933d7de9",
    f"{VF}/vf_stats.json": "3aaca1d8438bf21543e52918edfbf84f7376599c537bc19cd78e0a24d036367c",
    f"{VF}/vf_compare_verdicts.json": "1bcc47331298f8030b8ae5ba2161cb145d6f1508b445a8c1ebc3687e419a00aa",
    f"{VF}/vf_compare_extra.json": "b865827e161f727aac5f910db9ccc76d34168d442041779cc2ce9ac27b78c5a4",
    "features/coco_e0/captions_index.parquet": "ee979cc81d7d69be888438b72c2cf921489f1475db1b07c23c747caec60ed76b",
}
# Read for the caption recheck; not named in an E0 STATUS line, so its hash is recorded, not checked.
RECORDED_ONLY = ["features/coco_pool/items.parquet"]
# ctrold eval tables as they are on disk (STATUS line 334 lists the last two in swapped order).
CTROLD = {
    f"{R}/eval/E0_cells_ctrold.parquet": "9c95c59ed7c8adfb4266a3ba176b529307d515ba24cafdf5def93fe521ea34cc",
    f"{R}/eval/E0_drawslopes_ctrold.parquet": "93c1b3328873ee07a1b383673e8d403fc69f74ffb273966a780d8f03b0064f27",
    f"{R}/eval/E0_slopes_ctrold.parquet": "675edb30ed3d8b380c003acbf4343f4ae1e37b178bc75813add0d2708bcbfd41",
    f"{R}/eval/E0_elasticity_ctrold.parquet": "a28a80dd2d94facc92af345b039470bc3eb35dae98142404ab348e3f5fb83fd2",
}
ENC = ["contriever", "mpnet", "qwen3emb06"]
VITS = ["augreg_b", "clip_b", "dinov2_b", "mae_b", "siglip_b"]


def p(rel):
    return os.path.join(PROJ, rel)


def sha256(rel):
    h = hashlib.sha256()
    with open(p(rel), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def die(msg, code=2):
    print("REFUSED: " + msg, file=sys.stderr)
    sys.exit(code)


def load(rel):
    with open(p(rel)) as f:
        return json.load(f)


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def sg(x, n=3):
    return f"{x:.{n}g}"


def status_lines():
    with open(p("STATUS.md")) as f:
        return f.read().split("\n")


def check_status():
    lines = status_lines()
    if lines and lines[-1] == "":
        lines = lines[:-1]
    find = lambda tag: [i + 1 for i, l in enumerate(lines) if tag in l]
    agree = find("[e0-indep] AGREE")
    disagree = find("[e0-indep] DISAGREE")
    review = find("[e0-review] REVIEW")
    papernum = find("[e0-review] PAPER NUMBERS")
    done = find("[e0-run] DONE")
    if disagree:
        die(f"STATUS has [e0-indep] DISAGREE at lines {disagree}")
    if not agree:
        die("STATUS has no [e0-indep] AGREE line")
    if not review:
        die("STATUS has no [e0-review] REVIEW line")
    if done:
        die(f"STATUS already has [e0-run] DONE at lines {done}")
    text = "\n".join(lines)
    for rel, h in EXPECTED.items():
        if h not in text and h[:16] not in text:
            die(f"sha256 of {rel} is not in STATUS")
    # approval tokens: only descriptive mentions (lines 329 and 336) are allowed
    tok = [i + 1 for i, l in enumerate(lines) if "E0 Q1 APPROVED" in l or "E0 B7 APPROVED" in l]
    if any(n not in (329, 336) for n in tok):
        die(f"approval token found outside lines 329/336: {tok}; the not-run list would be wrong")
    return {"n_lines": len(lines), "indep_agree_line": agree[-1], "review_line": review[-1],
            "review_paper_numbers_line": papernum[-1] if papernum else None,
            "approval_token_lines": tok}


def verify_inputs(verdicts):
    got = {}
    for rel, h in EXPECTED.items():
        s = sha256(rel)
        if s != h:
            die(f"{rel} sha256 {s} != logged {h}")
        got[rel] = s
    for arm, h in verdicts["obs_sha256"].items():
        rel = f"{R}/obs/obs_{arm}.parquet"
        s = sha256(rel)
        if s != h:
            die(f"{rel} sha256 {s} != VERDICTS obs_sha256 {h}")
        got[rel] = s
    for rel, h in CTROLD.items():
        s = sha256(rel)
        if s != h:
            die(f"{rel} sha256 {s} != VERDICTS addendum_1 record {h}")
        got[rel] = s
    for rel in RECORDED_ONLY:
        got[rel] = sha256(rel)
    return got


def caption_recheck():
    import pandas as pd
    ci = pd.read_parquet(p("features/coco_e0/captions_index.parquet"))
    it = pd.read_parquet(p("features/coco_pool/items.parquet"), columns=["idx", "image_id", "text"])
    c1 = dict(zip(it["idx"], it["text"].str.strip()))
    n_last = n_first = n_in = n_dup = 0
    ncap = {}
    for idx, grp in ci.sort_values(["idx", "ann_id"]).groupby("idx"):
        texts = [s.strip() for s in grp["text"]]
        t = c1[idx]
        n_last += texts[-1] == t
        n_first += texts[0] == t
        n_in += t in texts
        n_dup += sum(s == t for s in texts) > 1
        ncap[len(texts)] = ncap.get(len(texts), 0) + 1
    return {"n_items": len(it), "c1_equals_last_by_annotation_id": int(n_last),
            "c1_equals_first_by_annotation_id": int(n_first), "c1_equals_an_official_caption": int(n_in),
            "c1_also_equals_an_earlier_duplicate_caption": int(n_dup),
            "captions_per_image": {str(k): v for k, v in sorted(ncap.items())},
            "how": "items.parquet text (strip) against captions_index.parquet sorted by ann_id per image"}


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def build(st, hashes, t0):
    V = load(f"{R}/VERDICTS_E0.json")
    RV = load(f"{A}/review/review_recompute_20261010T141910Z.json")
    RX = load(f"{A}/review/review_explore_20261010T142122Z.json")
    RP = load(f"{A}/review/review_paper_numbers_20261010T142548Z.json")
    AU = load(f"{A}/review/audit_20261010T142040Z.json")
    CO = load(f"{VF}/vf_compare_obs.json")
    CV = load(f"{VF}/vf_compare_verdicts.json")
    CE = load(f"{VF}/vf_compare_extra.json")
    VS = load(f"{VF}/vf_stats.json")["from_e0_obs"]
    RS = load(f"{R}/summary.json")
    DG = load(f"{R}/diag/diag_singlemodel.json")

    # ---- consistency between the three sources (refuse on any disagreement)
    labels = V["labels"]
    if V["order"] != ["P1", "P2b", "P2c"]:
        die("VERDICTS order is not P1, P2b, P2c")
    for k in ("P1", "P2b", "P2c"):
        if RV[k]["label"] != labels[k]:
            die(f"review label {k} differs")
        if VS[k]["label"] != labels[k]:
            die(f"V-f label {k} differs")
    if CV["labels_differing"] or CV["n_numeric_over_tol"] or CV["max_abs_diff"] > 1e-12:
        die("V-f compare_verdicts reports a disagreement")
    if CO["n_cells_differing_predicted"] or CO["n_cells_differing_allH2"] or CO["differing_cells"]:
        die("V-f compare_obs reports differing cells")
    if not all(RV["comparison_with_VERDICTS"]["labels_equal"].values()):
        die("review comparison_with_VERDICTS reports a label difference")
    if AU["result"] != "PASS" or AU["violations"]:
        die("frozen guard audit is not PASS")
    P1, P2b, P2c = V["P1"], V["P2b"], V["P2c"]
    for a, b, what in [(P1["median_sum"], RV["P1"]["median_sum"], "P1 median sum (review)"),
                       (P2b["median_abs_gap"], RV["P2b"]["median_abs_g"], "P2b median |g| (review)"),
                       (P2b["median_abs_gap"], VS["P2b"]["median_abs_g"], "P2b median |g| (V-f)"),
                       (P2c["median_gamma"], RV["P2c"]["median_gamma"], "P2c median gamma (review)"),
                       (P2c["median_dln"], RV["P2c"]["median_dln"], "P2c median dln (review)"),
                       (VS["P2b"]["lambda_hat"], RV["P2b"]["lambda_descriptive"], "lambda-hat V-f vs review")]:
        if not close(a, b):
            die(f"{what}: {a} vs {b}")
    for e in ENC:
        if not close(VS["P2b"]["lambda_e"][e], RV["P2b"]["lambda_e_descriptive"][e]):
            die(f"lambda_e {e}: V-f vs review")
    if P2b["lam"] is not None or P2b["lam_e"]:
        die("VERDICTS now reports lambda for arm P; the reporting-gap text would be wrong")

    cap = caption_recheck()
    rcap = RV["caption_fact"]
    if not (cap["c1_equals_last_by_annotation_id"] == cap["n_items"] == 16384
            and cap["c1_equals_first_by_annotation_id"] == 0 and rcap["frac_rank0"] == 0.0
            and rcap["frac_c1_equals_official"] == 1.0):
        die("caption recheck does not confirm the review's caption fact")

    # ---- numbers
    pp = {d["pair"]: d for d in P1["per_pair"]}
    em = [d["e_m"] for d in P1["per_pair"]]
    sums = [d["sum"] for d in P1["per_pair"]]
    rob = V["descriptive"]["robustness"]
    p3a = V["descriptive"]["P3a_level"]
    lev = RS["readings_13"]["level_wording_13_3"]
    lam_hat = VS["P2b"]["lambda_hat"]
    lam_e = VS["P2b"]["lambda_e"]
    enc_g = {e: VS["P2b"]["encoder_informative"][e]["median_abs_g"] for e in ENC}
    lad = P2c["ladder"]
    lad_L = [statistics.median(lad[q]["L"][n] for q in lad) for n in range(5)]
    lad_b = [statistics.median(lad[q]["b"][n] for q in lad) for n in range(5)]
    sec = V["secondary"]
    sch = V["descriptive"]["P3h_schnaus"]
    p3b = V["descriptive"]["P3b_caption"]
    p3c = V["descriptive"]["P3c_preprocessing"]
    p3e = V["descriptive"]["P3e_encoders"]
    p3f = V["descriptive"]["P3f_ordering"]
    p3g = V["descriptive"]["P3g_shared_k_over_m"]
    p2d = V["P2_descriptive"]
    wil = P2b["wilcoxon_descriptive"]
    loo_e = P2b["leave_one_out"]["leave_encoder_out"]
    loo_v = P2b["leave_one_out"]["leave_vit_out"]
    per_draw = RV["P1"]["per_draw_median_sum"]
    lv = RV["levels_13_3"]["within_vit_ranges"]
    tsec = time.process_time() - t0  # CPU seconds of this step up to here (updated at write time)

    def h16(rel):
        return hashes[rel][:16]

    S = {}
    S["labels_in_plan_order"] = {"P1": labels["P1"], "P2b": labels["P2b"], "P2c": labels["P2c"]}
    S["meta"] = {
        "file": "E0_summary.json (with E0_summary.md)",
        "role": "e0-run", "step": "ANALYSIS_PLAN_E0.md v1.0 section 11 step 7 (closing); outputs as section 14",
        "plan_sha256": hashes[f"{A}/ANALYSIS_PLAN_E0.md"],
        "addendum_1_sha256": hashes[f"{A}/ANALYSIS_PLAN_E0_ADDENDUM_1.md"],
        "addendum_1_effect": V["addendum_1"]["effect"],
        "created_utc": None,
        "preconditions": {"indep_agree_status_line": st["indep_agree_line"], "review_status_line": st["review_line"],
                          "review_paper_numbers_status_line": st["review_paper_numbers_line"],
                          "status_lines_before_this_step": st["n_lines"]},
        "note": "Every number is read from the sources listed under 'sources'; this role recomputed no statistic. "
                "Readings follow plan section 13 (binding for this summary); paper wording stays the author's.",
    }
    S["confirmatory"] = {
        "P1": {"label": labels["P1"], "source": "VERDICTS_E0.json P1",
               "n_informative": P1["n_informative"], "n_pairs": P1["n_pairs"],
               "e_m_range": [min(em), max(em)], "median_sum_k10_100": P1["median_sum"],
               "frac_abs_sum_le_0.05": P1["frac_abs_sum_le"], "max_abs_sum": max(abs(x) for x in sums),
               "sum_range": [min(sums), max(sums)], "per_encoder_median_sum": P1["per_encoder_median_sum"],
               "per_draw_labels": rob["P1_per_draw_labels"], "per_draw_median_sum_review": per_draw,
               "thresholds": P1["thresholds"], "disclosure": P1["disclosure"],
               "reported_with_7_1": {"median_sum_all_k_1_10_100": statistics.median(rob["P"]["elasticity_k1_10_100"].values()),
                                     "median_sum_k_1_10": statistics.median(rob["P"]["elasticity_k1_10"].values()),
                                     "corr_ek_minus_em_30VL": V["descriptive"]["corr_ek_minus_em_30VL"]}},
        "P2b": {"label": labels["P2b"], "source": "VERDICTS_E0.json P2b; lambda from V-f vf_stats.json and review recompute",
                "guard_threshold_2s_res": P2b["guard_threshold"], "median_abs_g": P2b["median_abs_gap"],
                "per_encoder_median_abs_g": enc_g, "informative_encoders": P2b["informative_encoders"],
                "median_L_new": RS["P2b"]["median_L_new"], "median_L_s11_lm_token": lev["arms"]["P"]["lm_token"],
                "median_b10_new": RS["P2b"]["median_b10_new"],
                "line": P2b["line"], "thresholds": P2b["thresholds"],
                "descriptive_always_reported": {
                    "lambda_hat": lam_hat, "lambda_e": lam_e,
                    "lambda_source": "V-f vf_stats.json from_e0_obs.P2b and review recompute (agree within 1e-12); "
                                     "VERDICTS_E0.json gives lam None, lam_e {} (reporting gap, plan 7.3 'always reported')",
                    "leave_one_encoder_out": loo_e, "leave_mpnet_out": P2b["leave_mpnet_out_lambda"],
                    "leave_one_vit_out": loo_v, "wilcoxon": wil,
                    "median_resid_vs_line": RS["P2b"]["median_resid_vs_line"],
                    "median_b_minus_vit_competitor": RS["P2b"]["median_b_minus_vit_competitor"]}},
        "P2c": {"label": labels["P2c"], "source": "VERDICTS_E0.json P2c",
                "guard_median_dln": P2c["thresholds"]["guard_dln"], "median_dln": P2c["median_dln"],
                "per_encoder": P2c["gamma_e"], "informative_encoders": P2c["informative_encoders"],
                "thresholds": P2c["thresholds"],
                "descriptive_always_reported": {"median_gamma": P2c["median_gamma"],
                                                "leave_encoder_out_median_gamma": P2c["leave_encoder_out_median_gamma"],
                                                "ladder_median_L_n1_to_5": lad_L, "ladder_median_b10_n1_to_5": lad_b,
                                                "median_gamma_k1": sec["P2c_other_k"]["k1"]["median_gamma"],
                                                "median_gamma_k100": sec["P2c_other_k"]["k100"]["median_gamma"]}},
        "three_implementations_agree": {
            "VERDICTS_E0": "frozen e0_evaluate via ADDENDUM 1 wrapper",
            "review_recompute": {"imports": RV["imports"], "labels_equal": RV["comparison_with_VERDICTS"]["labels_equal"],
                                 "max_abs_diff_reported": 2.2e-15, "max_abs_diff_source": "STATUS line 339 (e0-review)",
                                 "frozen_constants_max_abs_diff": max(RV["frozen_constants_recomputed"]["diff_a"],
                                                                      RV["frozen_constants_recomputed"]["diff_c"],
                                                                      RV["frozen_constants_recomputed"]["competitor_max_abs_diff"]),
                                 "obs_step_from_stored_neighbour_lists": RV["obs_step_from_caches"]},
            "indep_V_f": {"cells": CO["n_cells_total"], "cells_overlap_count_differs": CO["n_cells_differing_predicted"],
                          "cells_overlap_count_differs_allH2": CO["n_cells_differing_allH2"],
                          "accepted_near_tie_anchors": 0,
                          "exact_k_boundary_tie_anchor_cells_text_side": {a: CO["arms"][a]["sum_n_tieB_e0"] for a in CO["arms"]},
                          "n_numeric": CV["n_numeric"], "max_abs_diff": CV["max_abs_diff"], "tolerance": CV["tolerance"],
                          "n_labels": CV["n_labels"], "labels_differing": len(CV["labels_differing"]),
                          "own_obs_vs_e0_obs_statistics_max": CE["own_obs_vs_e0_obs_statistics"]["max"],
                          "limit": "per-cell total overlap counts; obs files hold no per-anchor lists; E0 caches not read by V-f"},
            "frozen_guard_audit": {"result": AU["result"], "n_new_pair_files": AU["n_new_pair_files"],
                                   "guard_line_utc": AU["guard_line_utc"]}},
    }
    S["readings_13_binding"] = {
        "P1_13_1": "Agreement at fixed k/m stays stable at k >= 10 for sentence-encoder text sides too "
                   f"(median sum {P1['median_sum']:.4f}); the evaluation size acts as a scale parameter for this family as well.",
        "P2b_x_P2c_13_2": "uninformative x uninformative: Nothing new about the level-slope regularity.",
        "theory_1_4": "No remark in app:scale. Theorem 1 is exact and unaffected by any E0 outcome.",
        "P1_fails_modifier": "not applicable (P1 holds)",
        "B6_tension_row": "not triggered (P2b did not fail)",
        "level_wording_13_3": {"word": lev["word"], "arms": lev["arms"],
                               "consequence": "The Limitations paragraph says that our vision-language levels describe "
                                              "language-model token features with our protocol, citing Schnaus et al. and E0."},
        "R1_clip_caveat_13_3": {"triggered": False, "encoder_clip_vs_l2_own10NN": {e: DG["clip_vs_l2"][e] for e in ENC},
                                "lm_clip_vs_l2_own10NN_for_context": {k: DG["clip_vs_l2"][k] for k in ("bloomz_1b7", "openllama_3b", "pythia410m_s1")}},
        "without_Q1_13_3": "No sentence may attribute a level difference between LM-token and sentence-encoder pairs in the "
                           "AVG arms to the encoder rather than the caption average; only the C1 arms (P, C) compare encoders cleanly.",
        "draft_13_4_filled": "With three sentence encoders as the text side, the vision--language median at $k=10$ and $m=1024$ is "
                             f"${lev['arms']['P']['sentence']:.3f}$, against ${lev['arms']['P']['lm_token']:.3f}$ with language-model "
                             "token features, and agreement at fixed $k/m$ again stays close to stable for $k\\ge10$ (median sum of "
                             f"elasticities ${P1['median_sum']:.3f}$). [Neither P2b bracket applies: P2b is uninformative.]",
    }
    S["secondary_and_descriptive"] = {
        "secondary_labels_no_confirmatory_weight": {
            "P1": {k: {"label": x["label"], "median_sum": x["median_sum"], "n_informative": x["n_informative"]} for k, x in sec["P1"].items()},
            "P2b": {k: {"label": x["label"], "lambda": x["lam"], "median_abs_gap": x["median_abs_gap"],
                        "guard": x["guard_threshold"]} for k, x in sec["P2b"].items()},
            "P2c_descriptive_label_if_rule_applied": {k: {"label": x["label_if_rule_applied"], "median_gamma": x["median_gamma"]}
                                                      for k, x in sec["P2c_other_k"].items()}},
        "P2_descriptive": {"rho_30VL": p2d["within_vit_permutation"]["rho"], "p_within_vit_perm": p2d["within_vit_permutation"]["p_one_sided"],
                           "R": p2d["within_vit_permutation"]["R"], "rho_6_text_models": p2d["text_model_level"]["rho"],
                           "p_exact_720": p2d["text_model_level"]["p_exact_one_sided"]},
        "P3a_level_per_arm": p3a,
        "P3a_within_vit_ranges_review": lv,
        "class_median_k10_by_m_review": RP["class_median_k10"],
        "P3b_caption": {k: {"median_dlnL": x["median_dlnL"], "median_db10": x["median_db10"]} for k, x in p3b.items()},
        "P3c_preprocessing": {k: {"median_dlnL": x["median_dlnL"], "median_db10": x["median_db10"]} for k, x in p3c.items()},
        "P3e_encoders": p3e,
        "P3f_ordering": {k: {"slope_order": x["slope_order"], "level_order": x["level_order"], "median_slope": x["median_slope"]}
                         for k, x in p3f.items()},
        "P3g_shared_k_over_m": p3g,
        "P3h_schnaus": {"CA_dinov2_mpnet": {k: sch["CA"][k] for k in ("dinov2_b__mpnet|k10", "dinov2_b__mpnet|k100")},
                        "P_dinov2_mpnet": {k: sch["P"][k] for k in ("dinov2_b__mpnet|k10", "dinov2_b__mpnet|k100")},
                        "CA_median_k100_m10000_run_summary": RS["P3h_schnaus"]["CA"]["median_L_m10000_k100"],
                        "CA_median_k10_m10000_run_summary": RS["P3h_schnaus"]["CA"]["median_L_m10000_k10"],
                        "review_schnaus_matched": RV["schnaus_matched"],
                        "schnaus_values_approximate": "about 0.38 at k=100 and 0.07-0.19 at k=10 on 10,000 COCO points "
                                                      "(critic's readings of their figures; disclosed before the freeze)"},
        "P3i_diag": DG,
        "P3j_within_class_coefficients": V["descriptive"]["P3j_reference_within_class_coefficients"],
        "robustness": {"draw_sd_b10_median": statistics.median(rob["P"]["draw_sd_b10"].values()),
                       "P1_per_draw_labels": rob["P1_per_draw_labels"]},
        "x4_x5": "X4 (slope of ln(1-s)) and X5 (chance-corrected slopes) were computed by the frozen evaluation as exploratory "
                 "columns (VERDICTS descriptive.x4_exploratory; eval/E0_slopes_<arm>.parquet slope_obs_cc); not interpreted.",
        "full_table": f"{R}/eval/E0_tables.md (plan B8.5; produced by code)",
    }
    S["corrections_and_findings"] = [
        {"id": "caption_source", "found_by": "e0-review", "rechecked_by": "e0-run closing step", "recheck": cap,
         "fact": "C1 (the S1.1 caption, column text of items.parquet) is the LAST official caption of its image by annotation id "
                 "(= last in file order) for 16384/16384 pool images, and the first (lowest-id) caption for 0/16384.",
         "corrects": ["v4 app:exp-s11 'the first caption' (wrong)",
                      "PROTOCOL section 2 and plan R.3(6) and 13.5 'sampled at random by the dataset author' (ruled out by the data: 0 of 16384 first captions)"],
         "effect_on_results": "none: arm D1 (N1, the lowest-id caption) median L(10,1024) "
                              f"{p3a['D1']['cross-modal-sentence']['L_k10_m1024'][0]:.4f} vs arm P (C1) "
                              f"{p3a['P']['cross-modal-sentence']['L_k10_m1024'][0]:.4f}",
         "proposed_wording": "review proposal E (post hoc correction): 'For each of $16{,}384$ COCO validation images, the pool holds "
                             "the caption stored by \\texttt{coco-30-val-2014}, which is the last caption of the image in the COCO "
                             "annotation file.'",
         "note": "the review's rank histogram shows ranks 1 and 3 for two images because C1 also equals an earlier duplicate "
                 f"caption in {cap['c1_also_equals_an_earlier_duplicate_caption']} images (it reports the first match)."},
        {"id": "lambda_reporting_gap", "found_by": "e0-indep",
         "fact": "Plan 7.3 says lambda-hat and every lambda_e are always reported; VERDICTS_E0.json, summary.json and E0_tables.md "
                 "give lam None, lam_e {} for arm P because the guard returned uninformative.",
         "values_reported_here": {"lambda_hat": lam_hat, "lambda_e": lam_e}, "effect_on_label": "none (descriptive)"},
        {"id": "status_line_334_hash_order", "found_by": "e0-run closing step",
         "fact": "STATUS line 334 lists eval/E0_{cells,drawslopes,slopes,elasticity}_ctrold.parquet as 9c95c59e, 93c1b332, a28a80dd, "
                 "675edb30; on disk slopes = 675edb30 and elasticity = a28a80dd (VERDICTS addendum_1 record is right). "
                 "Only the order in that line is swapped; all four files are unchanged.", "effect": "none"},
        {"id": "verdicts_file_mode", "found_by": "e0-run closing step",
         "fact": "results/E0_sentence/VERDICTS_E0.json is mode 664 (the obs, eval, diag and guard files too), not 444; its sha256 "
                 "still equals the logged 5b04e460. Not changed by this role (no existing file is modified); the lead may set 444.",
         "effect": "none"},
        {"id": "indep_disclosures", "found_by": "e0-indep",
         "fact": "V-f read the STATUS tail (with the labels) before its own numbers existed, to copy the line format; it opened no "
                 "E0 code, report, review or VERDICTS file before vf_obs.parquet existed. One file outside the plan paths: "
                 "/tmp/vf_designcheck_1049305/vf_design_check.json (not deleted, per rules). vf_rules.py's docstring mentions a "
                 "scipy cross-check that the code does not run (Wilcoxon p by full enumeration of 2^15 sign patterns)."},
        {"id": "caption_gain_two_definitions", "found_by": "e0-review",
         "fact": f"'+14-17%' combines two arms: AVG5 vs C1 raises L by {100 * (2.718281828459045 ** p3b['AVG5-C1|clip']['median_dlnL'] - 1):.0f}% "
                 f"under clip and {100 * (2.718281828459045 ** p3b['AVG5-C1|ctr']['median_dlnL'] - 1):.0f}% under ctr (medians of the ln ratio)."},
        {"id": "process_notes_logged_earlier",
         "fact": "COCO archive fetched by the http fallback the plan allows (size and md5 verified); ADDENDUM 1 (write-step fix) "
                 "logged before the guard; first e0_calib attempt wrote nothing (missing eval/ directory) and was rerun unchanged; "
                 "the CTR CALIBRATION milestone was re-logged in the exact guard form with the same hash."},
    ]
    S["exploratory_or_post_hoc"] = [
        "P1's restriction to k ≥ 10 was chosen after seeing S1.1 (2026-10-10), before any E0 score (plan §7.1 disclosure).",
        "The e0-review exploratory file, labelled EXPLORATORY and post hoc, holds three analyses. (a) P1 on m ≥ 2048 only: median sum "
        f"{RX['P1_m_ge_2048']['new']['median_sum']:.4f}, with the k = 100 cells at ≥ {RX['obs_over_chance']['new_m2048_k100']['min']:.2f}× chance. "
        "(b) The k = 1 comparison with S1.1: median sum on k ∈ {1, 10} "
        f"{RX['k1_departure']['new_K1_10']['median_sum']:.3f} against {RX['k1_departure']['s11_vl_K1_10']['median_sum']:.3f}, and over all k "
        f"{RX['k1_departure']['new_K1_10_100']['median_sum']:.3f} against {RX['k1_departure']['s11_vl_K1_10_100']['median_sum']:.3f}. "
        "(c) A per-ViT P2b decomposition for arms P and A. The new-pair sums themselves are pre-specified (plan §7.1); the comparison "
        "with S1.1 and its reading (k = 1 is where Theorem 1's square-root term is relatively largest) are post hoc.",
        "Some readings of descriptive numbers were made after the labels were known. The new slopes sit about halfway between the "
        "level line and the ViT values (λ̂ about 0.44). Caption averaging flattens the slope faster than the level alone predicts "
        "(median γ 0.435 against 0.228). Arm A's λ̂ of 0.957 averages ViT-level deviations of opposite sign.",
        "The level gap to Schnaus et al. is roughly fivefold. Attributing it to the vision features (mid-depth CLS here, pooled output "
        "tokens there) is a hypothesis; X3 would test it and has not run. Schnaus et al.'s values are readings of their figures.",
        "Paper wording. The e0-review proposals A–F (OLD → NEW; v4 unedited) use wording that plan §13 does not fix, except where they "
        "fill §13.4. They are post hoc and, if used, are labelled so in the manuscript source comments (plan B8.3). Proposal D departs "
        "from the §13.4 Limitations draft, because that draft reads as if the encoders explained the gap; proposal E replaces §13.5.",
        "The caption-source fact was checked after the results (by the review, and rechecked here). It is a fact about the data, not "
        "a statistic, and it changes no result.",
    ]
    S["not_run"] = {
        "X1": "Exploratory (plan §7.6). S1.1's null N1 on the 15 arm-P pairs through the frozen hrhv2.predict_v11. Code not written.",
        "X2": "Exploratory (plan §7.6). Levina–Bickel intrinsic dimension per model and caption count n, correlated with the P2b "
              "residuals and with γ. Code not written.",
        "X3": "Exploratory (plan §7.6). The protocol bridge: DINOv2-B with rel-1.0 CLS, rel-1.0 patch mean and rel-0.5 patch mean, "
              "each with the 3 encoders under ctr × N5 (9 pairs, all to be reported). It needs new DINOv2 features on the GPU. It is "
              "the direct test of the vision-feature explanation for the level gap.",
        "Q1": "Conditional (plan §15). E0-LMavg: the 3 S1.1 LMs re-embedded on all official captions (about 0.6–1.5 GPU-h, local "
              "snapshots, no download). It needs a STATUS line 'E0 Q1 APPROVED' that quotes the author, and none exists. If it is "
              "approved now, every Q1 output carries 'decision to run made after the guard'.",
        "B7": "Conditional (plan §15). The MPNet membership check, a 23.9 MB text download of sentence-transformers/coco-captions. "
              "It needs 'E0 B7 APPROVED', and none exists.",
        "X4_X5": "Computed by the frozen evaluation as exploratory columns (VERDICTS descriptive.x4_exploratory; slope_obs_cc in "
                 "eval/E0_slopes_<arm>.parquet); not interpreted.",
    }
    S["compute"] = {
        "gpu_hours": {"e0-extract (in-loop embedding)": 0.368, "e0-extract (process wall incl. loading and 0.7-min smoke)": 0.38,
                      "all other roles": 0.0, "E0_total": 0.38, "plan_estimate": "0.4-1.0, central 0.6", "cap": "none set"},
        "cpu_as_logged": {"e0-critic (line 327)": "about 1 core-min", "e0-freeze (lines 328-329)": "22 s constants; about 0.4 core-h code freeze and validation",
                          "e0-extract (lines 330-333)": "pinned to 6 cores; not logged separately",
                          "e0-run (lines 334-338)": "caches about 0.5 core-h; 'E0 total so far about 1 core-h' at line 337",
                          "e0-review (lines 339-340)": "about 3 core-min", "e0-indep (line 341)": "0.45 core-h",
                          "e0-run closing (this step)": None},
        "cpu_E0_total_core_hours": "about 1.5 (1.9 if the line-337 total excludes the freeze role's 0.4)",
        "cpu_rule": "never more than 6 cores (taskset 0-5); task rule, not a cap set by E0",
        "disk_at_closing": {"features/coco_e0": "1.9 GB", "extract/e0_sentence/hf_home": "2.0 GB",
                            "results/E0_sentence/cache": "2.9 GB", "analysis/e0_sentence/indep": "2.2 GB"},
    }
    S["sources"] = {rel: h for rel, h in hashes.items()}
    S["sources_local"] = {"notes/v2_assessment_20261007/stage1_results/e0_sentence/review.md (author's notes, e0-review report)":
                          "c20b4f7d937b987a4183380eaca2246ebce8151e8a79291ceafa707b3e8b661a"}
    S["writer"] = {"script": f"{A}/run/e0_write_summary.py", "reads_only": True}
    return S, dict(V=V, RV=RV, RX=RX, RP=RP, CO=CO, CV=CV, CE=CE, VS=VS, RS=RS, DG=DG, cap=cap, lad_L=lad_L, lad_b=lad_b,
                   enc_g=enc_g, lam_hat=lam_hat, lam_e=lam_e, em=em, sums=sums, per_draw=per_draw, lv=lv, hashes=hashes, st=st)


def md(S, X):
    V, RV, RX, RP, CO, CV, CE, VS, RS, DG = (X[k] for k in ("V", "RV", "RX", "RP", "CO", "CV", "CE", "VS", "RS", "DG"))
    H = X["hashes"]
    h8 = lambda rel: H[rel][:8]
    P1, P2b, P2c = V["P1"], V["P2b"], V["P2c"]
    lab = S["labels_in_plan_order"]
    lev = S["readings_13_binding"]["level_wording_13_3"]["arms"]
    sec = S["secondary_and_descriptive"]["secondary_labels_no_confirmatory_weight"]
    p3a = V["descriptive"]["P3a_level"]
    p3b = V["descriptive"]["P3b_caption"]
    p3c = V["descriptive"]["P3c_preprocessing"]
    p3e = V["descriptive"]["P3e_encoders"]
    p3g = V["descriptive"]["P3g_shared_k_over_m"]
    sch = V["descriptive"]["P3h_schnaus"]
    p2d = V["P2_descriptive"]
    cap = X["cap"]
    st = X["st"]
    em, sums = X["em"], X["sums"]
    E = lambda x: f"{x:.4f}"
    L = []
    a = L.append
    a("# E0 summary: sentence encoders as the text side of the S1.1 COCO pools")
    a("")
    a(f"Role `e0-run`, closing step of plan §11 step 7; outputs as plan §14. Written {S['meta']['created_utc']}.")
    a(f"Plan `analysis/e0_sentence/ANALYSIS_PLAN_E0.md` v1.0 (`{h8(f'{A}/ANALYSIS_PLAN_E0.md')}…`) with "
      f"`ANALYSIS_PLAN_E0_ADDENDUM_1.md` (`{h8(f'{A}/ANALYSIS_PLAN_E0_ADDENDUM_1.md')}…`; implementation fix only, no rule or label changed).")
    a(f"Preconditions: STATUS line {st['indep_agree_line']} `[e0-indep] AGREE` and line {st['review_line']} `[e0-review] REVIEW`.")
    a("Every number is read from the files in §9 at the hashes listed there. This role recomputed no statistic; it only rechecked "
      "the caption-source fact by a count (§5.1). Readings follow plan §13, which binds this summary; paper wording stays the author's.")
    a("")
    a("## 1. Confirmatory labels (plan order)")
    a("")
    a("| prediction | label | statistic against its frozen threshold |")
    a("|---|---|---|")
    pe = P1["per_encoder_median_sum"]
    a(f"| **P1** agreement at fixed k/m stays stable, k ∈ {{10, 100}} | **{lab['P1']}** | "
      f"{P1['n_informative']}/{P1['n_pairs']} pairs informative (e_m from {min(em):.3f} to {max(em):.3f}; needs \\|e_m\\| ≥ 0.10). "
      f"Median e_k + e_m = {E(P1['median_sum'])} (holds if \\|median\\| ≤ 0.03). "
      f"{round(P1['frac_abs_sum_le'] * P1['n_informative'])}/{P1['n_informative']} pairs have \\|sum\\| ≤ 0.05 (holds if ≥ 80%); "
      f"sums run from {min(sums):.4f} to {max(sums):.4f}. Per encoder: contriever {E(pe['contriever'])}, mpnet {E(pe['mpnet'])}, "
      f"qwen3emb06 {E(pe['qwen3emb06'])}. Holds in each of the 5 draws. |")
    eg = X["enc_g"]
    a(f"| **P2b** between classes, the slope follows the level, not the vision model | **{lab['P2b']}** | "
      f"Guard: median \\|g\\| = {E(P2b['median_abs_gap'])} < 2·s_res = {E(P2b['guard_threshold'])}. Per encoder: "
      f"contriever {E(eg['contriever'])}, mpnet {E(eg['mpnet'])}, qwen3emb06 {E(eg['qwen3emb06'])}; none informative. "
      f"New median level L(10, 1024) = {E(RS['P2b']['median_L_new'])}, against {E(lev['P']['lm_token'])} for the S1.1 LM-token pairs. |")
    ge = P2c["gamma_e"]
    a(f"| **P2c** within a pair, averaging 1 to 5 captions moves the slope along the curve | **{lab['P2c']}** | "
      f"Guard: median ln(L5/L1) = {P2c['median_dln']:.3f} < 0.3. Per encoder: contriever {ge['contriever']['median_dln']:.3f}, "
      f"mpnet {ge['mpnet']['median_dln']:.3f}, qwen3emb06 {ge['qwen3emb06']['median_dln']:.3f}; none informative. |")
    a("")
    a("**The three labels come out the same in three implementations.**")
    a(f"- `VERDICTS_E0.json` (`{h8(f'{R}/VERDICTS_E0.json')}…`): frozen evaluation code through the ADDENDUM 1 wrapper.")
    a(f"- e0-review recompute (`{h8(f'{A}/review/review_recompute_20261010T141910Z.json')}…`): own code, no E0, S1.1 or hrhv2 import. "
      "Labels equal; statistics within 2.2e-15. The frozen line, s_res and ViT competitors re-derived from S1.1 within 4.4e-16. "
      f"The overlap step re-derived from the stored neighbour lists in {RV['obs_step_from_caches']['n_cells']:,} cells, difference 0.0.")
    a(f"- e0-indep V-f (`{h8(f'{VF}/vf_compare_verdicts.json')}…`, `{h8(f'{VF}/vf_compare_obs.json')}…`): own neighbour tables from the stored "
      f"embeddings and ViT features. Overlap counts are equal in {CO['n_cells_total']:,} of {CO['n_cells_total']:,} cells (arms P, D1–D5), "
      "for both `obs` and `obs_allH2`. Tie counts are equal, and no anchor needed the near-tie clause. "
      f"{CV['n_numeric']} statistics agree within {CV['max_abs_diff']:.1e} (tolerance 1e-12), and all {CV['n_labels']} labels are identical. "
      "Limit: the check works on per-cell total overlap counts, because the obs files hold no per-anchor lists.")
    a("- Frozen `e0_guard.py audit`: PASS. All 87 new-pair files were born after the GUARD CLEAN line (14:06:50Z).")
    a("")
    a("## 2. Binding readings (plan §13)")
    a("")
    a(f"1. **P1 holds (§13.1).** Agreement at fixed k/m stays stable at k ≥ 10 for sentence-encoder text sides too (median sum "
      f"{P1['median_sum']:.4f}). The evaluation size acts as a scale parameter for this family as well.")
    a("2. **P2b × P2c = uninformative × uninformative (§13.2).** Nothing new about the level–slope regularity. Under §1.4 no "
      "remark goes into `app:scale`. Neither guard passed, so neither label says anything for or against the regularity. The P2b guard "
      "behaved as its frozen operating characteristics predicted: at a median level near 0.06 it returns \"uninformative\" in 84–88% of runs.")
    a("3. **Theorem 1** is exact; no E0 outcome changes it. The P1-fails modifier does not apply. The **B6 tension row** is not "
      "triggered, because P2b did not fail.")
    a(f"4. **Level wording (§13.3): \"between\"** in both comparison arms. Arm P: sentence encoders {lev['P']['sentence']:.3f}, LM tokens "
      f"{lev['P']['lm_token']:.3f}, vision–vision {lev['P']['vision']:.3f}. Arm C: {lev['C']['sentence']:.3f}, {lev['C']['lm_token']:.3f}, "
      f"{lev['C']['vision']:.3f}. Consequence: the Limitations paragraph says that our vision–language levels describe "
      "language-model token features with our protocol, citing Schnaus et al. and E0.")
    a(f"5. **R1 clip caveat (§13.3): not triggered.** The encoders' own-10-NN overlap of `clip` against `l2` is "
      f"{DG['clip_vs_l2']['contriever']:.3f} (contriever), {DG['clip_vs_l2']['mpnet']:.3f} (mpnet) and "
      f"{DG['clip_vs_l2']['qwen3emb06']:.3f} (qwen3emb06), all ≥ 0.8. The clip does move the three S1.1 LMs "
      f"({DG['clip_vs_l2']['bloomz_1b7']:.2f}, {DG['clip_vs_l2']['openllama_3b']:.2f}, {DG['clip_vs_l2']['pythia410m_s1']:.2f}), "
      "but the rule does not apply to them; this is context only.")
    a("6. **Without Q1 (§13.3).** No sentence may attribute a level difference between LM-token and sentence-encoder pairs in the "
      "AVG arms to the encoder rather than to the caption average. Only the C1 arms (P, C) compare encoders cleanly.")
    a("7. **§13.4 draft sentence, filled in** (a proposal; the wording stays the author's). Neither P2b bracket applies:")
    a("   > " + S["readings_13_binding"]["draft_13_4_filled"].replace(" [Neither P2b bracket applies: P2b is uninformative.]", ""))
    a("   The §13.4 Limitations draft (\"sentence encoders reach higher agreement …, as Schnaus et al. report\") can be read as if the "
      "encoders explained the gap to Schnaus et al. E0 shows that they close only a small part of it (§3.5). The review's replacement "
      "wording is post hoc (§6).")
    a("")
    a("## 3. Secondary and descriptive results (no confirmatory weight)")
    a("")
    a("Plan §8 applies: an \"uninformative\" confirmatory label is reported as such, and no secondary cell replaces it. "
      "The full code-produced table is `eval/E0_tables.md` (plan B8.5).")
    a("")
    a("### 3.1 Every secondary label")
    a("")
    a("| analysis | cell | label | statistic |")
    a("|---|---|---|---|")
    for k, x in sec["P1"].items():
        a(f"| P1 | {k} | {x['label']} | median sum {x['median_sum']:.4f}; informative {x['n_informative']} |")
    for k, x in sec["P2b"].items():
        lam = "—" if x["lambda"] is None else f"{x['lambda']:.3f}"
        a(f"| P2b | {k.replace('|', chr(92) + '|')} | {x['label']} | median \\|g\\| {x['median_abs_gap']:.4f} (guard {x['guard']:.4f}); λ̂ {lam} |")
    for k, x in sec["P2c_descriptive_label_if_rule_applied"].items():
        a(f"| P2c (descriptive, §7.4) | {k} | {x['label']} (if the rule were applied) | median γ {x['median_gamma']:.3f} |")
    a("")
    a("Secondary arm A (AVG5, `clip`) passes the P2b guard by a small margin and gives \"holds\" (λ̂ 0.957). Arm L gives "
      "\"inconclusive\" (λ̂ 0.409). Every other P2b cell is \"uninformative\". None of these changes the primary P2b label.")
    a("")
    a("### 3.2 Statistics the plan says are always reported")
    lh, le = X["lam_hat"], X["lam_e"]
    loo_e = P2b["leave_one_out"]["leave_encoder_out"]
    loo_v = P2b["leave_one_out"]["leave_vit_out"]
    wil = P2b["wilcoxon_descriptive"]
    a(f"- **P2b mixing weight (descriptive).** λ̂ = {lh:.4f}. Per encoder: λ_e = {le['contriever']:.4f} (contriever), {le['mpnet']:.4f} (mpnet), "
      f"{le['qwen3emb06']:.4f} (qwen3emb06). These values come from V-f and the review (which agree within 1e-12), since VERDICTS leaves them empty (§5.2). "
      f"Leave-one-encoder-out: {min(loo_e.values()):.3f}–{max(loo_e.values()):.3f}; leave-MPNet-out: {P2b['leave_mpnet_out_lambda']:.3f}. "
      f"Leave-one-ViT-out: {min(loo_v.values()):.3f}–{max(loo_v.values()):.3f}. "
      f"The descriptive Wilcoxon test gives W+ = {wil['vit_specific']['statistic']:.0f}, p = {wil['vit_specific']['p_one_sided']:.3f} "
      f"against the ViT-specific competitor, and W+ = {wil['constant']['statistic']:.0f}, p = {wil['constant']['p_one_sided']:.3f} against the constant one. "
      f"Median residual against the line: {RS['P2b']['median_resid_vs_line']:+.3f}; median b(10) minus the ViT competitor: "
      f"{RS['P2b']['median_b_minus_vit_competitor']:+.3f}.")
    lo = P2c["leave_encoder_out_median_gamma"]
    a(f"- **P2c (descriptive).** Median γ = {P2c['median_gamma']:.3f}. Per encoder: {ge['contriever']['median_gamma']:.3f} (contriever), "
      f"{ge['mpnet']['median_gamma']:.3f} (mpnet), {ge['qwen3emb06']['median_gamma']:.3f} (qwen3emb06); leave-one-encoder-out "
      f"{min(lo.values()):.3f}–{max(lo.values()):.3f}. For comparison, the cross-class line has 0.228 and the S1.1 within-class coefficient is 0.080. "
      "Ladder medians over the 15 pairs for n = 1…5 captions: L = " + ", ".join(f"{x:.4f}" for x in X["lad_L"]) +
      "; b(10) = " + ", ".join(f"{x:.3f}" for x in X["lad_b"]) + ".")
    rw = S["confirmatory"]["P1"]["reported_with_7_1"]
    a(f"- **P1 (reported with it, §7.1).** The median sum over all k (1, 10, 100) is {rw['median_sum_all_k_1_10_100']:.3f}; on k ∈ {{1, 10}} it is "
      f"{rw['median_sum_k_1_10']:.3f}. corr(e_k, −e_m) over the 30 VL pairs of arm P is {rw['corr_ek_minus_em_30VL']:.4f}. Per-draw median sums "
      f"run from {min(X['per_draw'].values()):.4f} to {max(X['per_draw'].values()):.4f}.")
    a("")
    a("### 3.3 Levels and slopes per arm (class medians; L = s̄ at k = 10, m = 1024)")
    a("")
    a("| arm | class | n | L(10,1024) median [min, max] | L(100,8192) | median b(10) | median sum k∈{10,100} | falling at k = 1 / 10 / 100 |")
    a("|---|---|---|---|---|---|---|---|")
    for arm, cl in p3a.items():
        for c, x in cl.items():
            Lk = x["L_k10_m1024"]
            fall = " / ".join(str(x["slope_classes"].get(k, {}).get("falling", 0)) for k in ("1", "10", "100"))
            a(f"| {arm} | {c} | {x['n']} | {Lk[0]:.4f} [{Lk[1]:.4f}, {Lk[2]:.4f}] | {x['L_k100_m8192'][0]:.4f} | {x['median_b10']:.3f} | "
              f"{x['median_sum_k10_100']:.4f} | {fall} |")
    a("")
    a("- **Within each ViT, every encoder is above every LM** (arm P vs S1.1 `clip`):")
    for v in VITS:
        r = X["lv"][v]
        a(f"  - {v}: LMs {r['lm'][0]:.3f}–{r['lm'][1]:.3f}, encoders {r['enc'][0]:.3f}–{r['enc'][1]:.3f}.")
    a(f"- **By encoder (arm P).** mpnet L {p3e['mpnet']['median_L_k10_m1024']:.4f}, b(10) {p3e['mpnet']['median_b10']:.3f}; contriever "
      f"{p3e['contriever']['median_L_k10_m1024']:.4f}, {p3e['contriever']['median_b10']:.3f}; qwen3emb06 {p3e['qwen3emb06']['median_L_k10_m1024']:.4f}, "
      f"{p3e['qwen3emb06']['median_b10']:.3f}.")
    cm = RP["class_median_k10"]
    a(f"- **Class median at k = 10 as m grows.** Sentence encoders: {cm['sentence_m256']:.3f} (m = 256), {cm['sentence_m1024']:.3f} (1024), "
      f"{cm['sentence_m16384']:.4f} (16,384). LM tokens: {cm['lm_token_m256']:.3f}, {cm['lm_token_m1024']:.3f}, {cm['lm_token_m16384']:.4f}.")
    a("- **Slope order equals level order (P3f).** At k = 1, 10 and 100 both orders are seed > vision > sentence-encoder VL > LM-token VL. "
      "At k = 10 the median slopes are −0.014, −0.214, −0.525 and −0.571.")
    a(f"- **P2 (descriptive).** Spearman ρ(L, b(10)) over the 30 VL pairs is {p2d['within_vit_permutation']['rho']:.3f} "
      f"(within-ViT permutation p = {p2d['within_vit_permutation']['p_one_sided']:.1e}, R = 100,000). Over the 6 text models ρ = "
      f"{p2d['text_model_level']['rho']:.3f} (exact p = {p2d['text_model_level']['p_exact_one_sided']:.4f}). As the plan anticipated "
      "(§7.2, B5), P2 cannot separate \"the level sets the slope\" from \"the level partly sets the slope\".")
    a("")
    a("### 3.4 Caption averaging and preprocessing (P3b, P3c; medians of ln-ratios)")
    ex = lambda d: f"{d:+.3f} (×{2.718281828459045 ** d:.3f})"
    a(f"- AVG5 − C1: ln L {ex(p3b['AVG5-C1|clip']['median_dlnL'])} under `clip`, {ex(p3b['AVG5-C1|ctr']['median_dlnL'])} under `ctr`. "
      f"Δb(10) is {p3b['AVG5-C1|clip']['median_db10']:+.3f} and {p3b['AVG5-C1|ctr']['median_db10']:+.3f}.")
    a(f"- AVG5 − N1 (`clip`): ln L {ex(p3b['AVG5-N1|clip']['median_dlnL'])}, Δb(10) {p3b['AVG5-N1|clip']['median_db10']:+.3f}. "
      f"ALL − AVG5: {p3b['ALL-AVG5|clip']['median_dlnL']:+.4f} (`clip`), {p3b['ALL-AVG5|ctr']['median_dlnL']:+.4f} (`ctr`).")
    a(f"- `ctr` − `clip`: ln L {p3c['ctr-clip|new']['median_dlnL']:+.3f} for the new pairs. For the old pairs it is "
      f"{p3c['ctr-clip|old cross-modal']['median_dlnL']:+.3f} (VL), {p3c['ctr-clip|old cross-vision']['median_dlnL']:+.3f} (vision) and "
      f"{p3c['ctr-clip|old seed-seed']['median_dlnL']:+.3f} (seed). `l2` − `clip`: {p3c['l2-clip|new']['median_dlnL']:+.3f} for the new pairs.")
    a(f"- Shared-k/m profile (P3g): the k = 1 cells are {min(v for k, v in p3g.items() if k.startswith('k1_')):.3f}–"
      f"{max(v for k, v in p3g.items() if k.startswith('k1_')):.3f} of the k = 10 cells at the same k/m. The k = 100 cells are "
      f"{min(v for k, v in p3g.items() if k.startswith('k100_')):.3f}–{max(v for k, v in p3g.items() if k.startswith('k100_')):.3f}.")
    a("")
    a("### 3.5 Comparison with Schnaus et al. (P3h; arm CA = `ctr` × AVG5, interpolated to m = 10,000)")
    ca = sch["CA"]
    pa = sch["P"]
    sm = RV["schnaus_matched"]
    a(f"- **DINOv2-B × MPNet in arm CA:** {ca['dinov2_b__mpnet|k100']['L_m10000']:.3f} at k = 100 and {ca['dinov2_b__mpnet|k10']['L_m10000']:.3f} at k = 10. "
      f"Arm P gives {pa['dinov2_b__mpnet|k100']['L_m10000']:.3f} and {pa['dinov2_b__mpnet|k10']['L_m10000']:.3f}. Schnaus et al. report about 0.38 and "
      "0.07–0.19; these are the critic's readings of their figures, approximate, and were disclosed before the freeze.")
    a(f"- **Arm CA at k = 100:** median {sm['CA_median_k100']:.3f}, maximum {sm['CA_max_k100']:.3f}. Our vision–vision pairs in the same cell "
      f"under `ctr` reach a median of {sm['ctrold_vision_median_k100']:.3f} and a maximum of {sm['ctrold_vision_max_k100']:.3f}. That is still below "
      "their vision–language value.")
    a(f"- **Fixed-m ratio s̄(100, m)/s̄(10, m) for DINOv2-B × MPNet:** {ca['dinov2_b__mpnet|k10']['ratio_k100_over_k10_m8192']:.2f} and "
      f"{ca['dinov2_b__mpnet|k10']['ratio_k100_over_k10_m16384']:.2f} in arm CA (m = 8,192 and 16,384), "
      f"{pa['dinov2_b__mpnet|k10']['ratio_k100_over_k10_m8192']:.2f} and {pa['dinov2_b__mpnet|k10']['ratio_k100_over_k10_m16384']:.2f} in arm P. "
      "This is the k-dependence at fixed m that P1 implies (e_k ≈ −e_m).")
    a("- **Reading (descriptive).** The encoder, caption averaging and centering together close only a small part of a roughly fivefold gap. "
      "Which protocol difference explains the rest is open (§6); X3 is the run that would test it.")
    a("")
    a("## 4. Verification record")
    a(f"- **V-f (e0-indep), STATUS line {st['indep_agree_line']}: AGREE.**")
    a("  - The design (draws, halves, anchors) was rebuilt bitwise.")
    a("  - Distance matrices for 23 inputs (5 ViTs and 3 encoders × {C1, N1–N5}) were built through `hrhv2.prep`, with V-f's own caption average.")
    a("  - Each H2 row of each prefix was fully argsorted with the key (D, position).")
    tb = S["confirmatory"]["three_implementations_agree"]["indep_V_f"]["exact_k_boundary_tie_anchor_cells_text_side"]
    a(f"  - Exact text-side ties at the k boundary come from duplicate captions: {tb['P']:,} anchor-cells in P, {tb['D1']:,} in D1 and "
      f"{tb['D2']} in D2, equal in both implementations. No anchor was accepted under the near-tie clause.")
    a(f"  - Computing the statistics from V-f's own obs instead of E0's obs changes them by at most {CE['own_obs_vs_e0_obs_statistics']['max']:.1e}.")
    a(f"- **Review (e0-review), STATUS lines {st['review_line']} and {st['review_paper_numbers_line']}.**")
    a("  - Labels and statistics were recomputed with independent code.")
    a("  - The review's own order audit passes. The STATUS prefix hashes check out and the milestones are in order.")
    a("  - The 15 named hashes and the 22 manifest files are unchanged.")
    a("  - Feature, cache and calibration files were all born before the GUARD CLEAN line, and the new-pair files after it.")
    a("  - Limit: birth times order the files in the project directory only. They cannot rule out a computation done elsewhere, and no sign of one was found.")
    a("")
    a("## 5. Corrections and findings")
    a(f"1. **Caption source (found by the review; rechecked here by count).** C1 is the S1.1 caption, the `text` column of `items.parquet`. "
      f"It equals the **last** official caption of its image by annotation id (= last in file order) for "
      f"{cap['c1_equals_last_by_annotation_id']:,}/{cap['n_items']:,} pool images, and the first (lowest-id) caption for "
      f"{cap['c1_equals_first_by_annotation_id']}/{cap['n_items']:,}.")
    a("   - **What this corrects.** v4 `app:exp-s11` says \"the first caption\", which is wrong. PROTOCOL §2 and plan R.3(6)/§13.5 say "
      "\"sampled at random by the dataset author\". The data rule that out: a random pick would make about one image in five carry "
      "its first caption, and none does.")
    a(f"   - **Duplicates.** In {cap['c1_also_equals_an_earlier_duplicate_caption']} images C1 also equals an earlier duplicate caption. "
      "That is why the review's first-match rank histogram shows ranks 1 and 3 for two images.")
    a(f"   - **Effect on results: none.** Arm D1 (N1, the lowest-id caption) has median L {p3a['D1']['cross-modal-sentence']['L_k10_m1024'][0]:.4f}; "
      f"arm P (C1) has {p3a['P']['cross-modal-sentence']['L_k10_m1024'][0]:.4f}.")
    a("   - **Wording.** The §13.5 sentence cannot be used with its stated premise. The review's proposal E reads: \"For each of 16,384 COCO "
      "validation images, the pool holds the caption stored by coco-30-val-2014, which is the last caption of the image in the COCO annotation file.\"")
    a(f"2. **λ̂ reporting gap (found by V-f).** Plan §7.3 says λ̂ and every λ_e are always reported. For arm P, `VERDICTS_E0.json`, `summary.json` "
      f"and `E0_tables.md` give `lam None, lam_e {{}}`, because the guard returned uninformative. The values in §3.2 (λ̂ {lh:.4f}) close the gap. "
      "They are descriptive and do not affect the label.")
    a("3. **STATUS line 334 lists two hashes in swapped order** (found here). Line 334 lists the ctrold eval tables "
      "{cells, drawslopes, slopes, elasticity} as 9c95c59e, 93c1b332, a28a80dd, 675edb30. On disk, slopes = 675edb30 and "
      "elasticity = a28a80dd; the VERDICTS `addendum_1` record has them right. All four files are unchanged.")
    a("4. **`VERDICTS_E0.json` is writable** (mode 664; the obs, eval, diag and guard files are 664 too). Its sha256 still equals the logged "
      "5b04e460. This role modifies no existing file, so the mode was left as it is; the lead may set it to 444.")
    a("5. **V-f disclosures.**")
    a("   - V-f read the STATUS tail, which carries the labels, before its own numbers existed, in order to copy the line format. "
      "It opened no E0 code, report, review or VERDICTS file before `vf_obs.parquet` existed.")
    a("   - One file lies outside the plan paths: `/tmp/vf_designcheck_1049305/vf_design_check.json`. It was not deleted, per the rules.")
    a("   - The docstring of `vf_rules.py` mentions a scipy cross-check that never runs. The Wilcoxon p comes from full enumeration of the 2^15 sign patterns.")
    a(f"6. **\"+14–17%\" mixes two arms.** AVG5 against C1 raises L by "
      f"{100 * (2.718281828459045 ** p3b['AVG5-C1|clip']['median_dlnL'] - 1):.0f}% under `clip` and "
      f"{100 * (2.718281828459045 ** p3b['AVG5-C1|ctr']['median_dlnL'] - 1):.0f}% under `ctr`. Both are medians of the ln ratio.")
    a("7. **Process notes logged earlier.**")
    a("   - The COCO archive came through the http fallback the plan allows; its size and md5 were verified.")
    a("   - ADDENDUM 1, a fix to the write step, was logged before the guard.")
    a("   - The first `e0_calib` attempt wrote nothing because `eval/` was missing; it was rerun unchanged.")
    a("   - The CTR CALIBRATION milestone was logged a second time, in the exact form the guard parses, with the same hash.")
    a("")
    a("## 6. Exploratory and post hoc items (none enters a label)")
    for i, t in enumerate(S["exploratory_or_post_hoc"], 1):
        a(f"{i}. {t}")
    a("")
    a("## 7. Runs not done")
    for k in ("X1", "X2", "X3", "Q1", "B7"):
        a(f"- **{k}.** {S['not_run'][k]}")
    a(f"- **X4, X5.** {S['not_run']['X4_X5']}")
    a("")
    a("## 8. Compute (actual use; no GPU cap was set)")
    a("- **GPU: 0.38 GPU-h for all of E0.** All of it was the e0-extract embedding run: 0.368 h in-loop, 0.38 h process wall including loading "
      "and the 0.7-min smoke. Every other role used 0. The plan estimated 0.4–1.0 GPU-h (central 0.6).")
    a("- **CPU as logged per role:**")
    a("  - e0-critic: about 1 core-min.")
    a("  - e0-freeze: 22 s for the constants, plus about 0.4 core-h for code freeze and validation.")
    a("  - e0-extract: pinned to 6 cores; not logged separately.")
    a("  - e0-run: about 0.5 core-h for the caches; line 337 gives \"E0 total so far about 1 core-h\".")
    a("  - e0-review: about 3 core-min.")
    a("  - e0-indep: 0.45 core-h.")
    a(f"  - This closing step: {S['compute']['cpu_as_logged']['e0-run closing (this step)']}.")
    a("- **E0 CPU total: about 1.5 core-h.** It is 1.9 core-h if the line-337 total leaves out the freeze role's 0.4. Never more than "
      "6 cores were used (taskset 0-5); that limit is the task rule, not an E0 cap.")
    a("- **Disk at closing:** `features/coco_e0` 1.9 GB, `extract/e0_sentence/hf_home` 2.0 GB, `results/E0_sentence/cache` 2.9 GB, "
      "`analysis/e0_sentence/indep` 2.2 GB.")
    a("")
    a("## 9. Sources (sha256, verified against STATUS before reading)")
    a("")
    a("| file | sha256 |")
    a("|---|---|")
    for rel, hsh in S["sources"].items():
        a(f"| `{rel}` | `{hsh}` |")
    for rel, hsh in S["sources_local"].items():
        a(f"| {rel} | `{hsh}` |")
    a("")
    a("## 10. 作者须知（中文）")
    a(f"1. **结论。** 三条确认性预测按计划顺序：P1 成立，P2b 无信息，P2c 无信息。冻结代码、复核和独立实现 V-f 三方给出同样的标签。"
      f"V-f 的 9,450 个格子重叠计数全部相等，934 个统计量最大差 {CV['max_abs_diff']:.1e}。")
    a(f"2. **P1 说明什么？** 把文本侧换成 3 个句向量模型后，k 和 m 同比例放大时一致性不变：k 弹性加 m 弹性的中位数是 {P1['median_sum']:.4f}，"
      f"15 对全在 0.05 以内，5 次抽样都成立。所以\"评估集大小是尺度参数\"推广到了句向量这一类耦合。定理 1 不受影响。")
    a(f"3. **P2b、P2c 为什么判不了？** 句向量只把水平从 {lev['P']['lm_token']:.3f} 提到 {lev['P']['sentence']:.3f}，"
      f"按水平和按 ViT 预测的斜率只差 {P2b['median_abs_gap']:.3f}，不到门槛 {P2b['guard_threshold']:.3f}。"
      f"平均 5 条描述只让水平升了约 {100 * (2.718281828459045 ** P2c['median_dln'] - 1):.0f}%，门槛要 35%。"
      "按冻结规则，关于\"斜率由水平决定\"没有新结论，app:scale 不加 remark。")
    a(f"4. **描述性数字（不计入判定）。** 混合权重 λ̂ = {lh:.2f}，新斜率大约在\"水平直线\"和\"ViT 值\"中间。同一对内的系数 γ 中位数是 "
      f"{P2c['median_gamma']:.3f}，约为直线系数 0.228 的两倍。这两点都是事后读法。")
    a(f"5. **水平措辞是\"介于两者之间\"。** 局限段要写明：我们的视觉–语言水平描述的是\"语言模型 token 特征加我们的协议\"，并引用 Schnaus 等人和 E0。"
      f"在接近他们的设置下，DINOv2-B × MPNet 是 {ca['dinov2_b__mpnet|k100']['L_m10000']:.3f}，他们约 0.38，差约 5 倍。"
      "差距可能来自视觉特征，但这只是假设，要 X3 才能检验。")
    a(f"6. **描述来源要更正。** C1 在全部 {cap['n_items']:,} 张图上都是 COCO 标注里该图的**最后一条**描述。它不是 v4 写的\"第一条\"，"
      "也不是计划写的\"随机一条\"。这一点不影响任何结果。")
    a("7. **几处记录问题，都不影响数字。** VERDICTS 没写 λ̂（V-f 补上了）；STATUS 第 334 行有两个哈希顺序写反了；VERDICTS_E0.json 的权限是可写（664），"
      "哈希没变，本角色没动它。")
    a("8. **没跑的。** 探索性的 X1–X3 没跑。Q1 和 B7 需要你在 STATUS 里批准，目前没有批准，所以也没跑。")
    a(f"9. **算力。** GPU 共 0.38 小时，全部用在提取上，没设上限。CPU 约 1.5 核时，从未超过 6 核。")
    a("")
    return "\n".join(L) + "\n"


def write_excl(rel, data):
    fd = os.open(p(rel), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(p(rel), 0o444)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["summary", "done"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--cpu-note", default=None, help="text for this step's CPU use in the summary")
    args = ap.parse_args()
    t0 = time.process_time()
    if args.mode == "summary":
        if not (args.dry_run or args.write):
            die("summary needs --dry-run or --write")
        for rel in (OUT_JSON, OUT_MD, OUT_DONE):
            if os.path.exists(p(rel)):
                die(f"{rel} exists; nothing is overwritten")
        st = check_status()
        V = load(f"{R}/VERDICTS_E0.json")
        hashes = verify_inputs(V)
        S, X = build(st, hashes, t0)
        S["meta"]["created_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        S["compute"]["cpu_as_logged"]["e0-run closing (this step)"] = (
            args.cpu_note or f"{time.process_time() - t0:.0f} CPU-s for this script (1 thread, taskset 0-5, nice 10); GPU 0")
        text = md(S, X)
        js = json.dumps(S, indent=1, ensure_ascii=False) + "\n"
        if args.dry_run:
            sys.stdout.write(text)
            print(f"\n[dry-run] json {len(js)} bytes, md {len(text)} bytes; nothing written", file=sys.stderr)
            return
        write_excl(OUT_JSON, js)
        write_excl(OUT_MD, text)
        print(json.dumps({OUT_JSON: sha256(OUT_JSON), OUT_MD: sha256(OUT_MD),
                          "cpu_s": round(time.process_time() - t0, 2)}))
    else:
        if os.path.exists(p(OUT_DONE)):
            die("DONE exists")
        for rel in (OUT_JSON, OUT_MD):
            if not os.path.exists(p(rel)):
                die(f"{rel} missing")
            if os.stat(p(rel)).st_mode & 0o222:
                die(f"{rel} is writable; expected chmod 444")
        S = load(OUT_JSON)
        st = check_status()
        hj, hm = sha256(OUT_JSON), sha256(OUT_MD)
        rec = {"utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "role": "e0-run",
               "plan_sha256": S["meta"]["plan_sha256"], "E0_summary.json": hj, "E0_summary.md": hm,
               "VERDICTS_E0.json": S["sources"][f"{R}/VERDICTS_E0.json"],
               "indep_agree_status_line": st["indep_agree_line"], "review_status_line": st["review_line"],
               "labels": S["labels_in_plan_order"],
               "statement": "E0 complete (plan 11 step 7): P1 holds, P2b uninformative, P2c uninformative; V-f AGREE; review PASS; "
                            "not run: X1-X3 (exploratory), Q1 and B7 (no approval)."}
        write_excl(OUT_DONE, json.dumps(rec) + "\n")
        print(json.dumps({OUT_DONE: sha256(OUT_DONE), "E0_summary.json": hj, "E0_summary.md": hm}))


if __name__ == "__main__":
    main()
