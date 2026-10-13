"""E0 run-role summary (role e0-run; not frozen code). Reads the frozen outputs and applies the binding readings of
ANALYSIS_PLAN_E0.md v1.0 §13 mechanically. Writes results/E0_sentence/summary.json (refuses to overwrite).
This is NOT the plan's E0_summary.{json,md}, which follows V-f (e0-indep) and the review (e0-review).

    venv/bin/python -I analysis/e0_sentence/run/e0_run_summary.py
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/hrh_v2_stage1_20261008")
AN = os.path.join(ROOT, "analysis", "e0_sentence")
sys.path.insert(0, AN)
import e0_common as E  # noqa: E402

P1_READ = {
    "holds": "Agreement at fixed k/m stays stable at k >= 10 for sentence-encoder text sides too (median sum {med}); "
             "the evaluation size acts as a scale parameter for this family as well.",
    "partly holds": "Close to stable for most pairs; the deviating pairs and encoders are named: {dev}.",
    "fails": "The k/m reading does not transfer to sentence-encoder couplings at k >= 10. Section 5.2's 'acts as a scale "
             "parameter' is limited to the S1.1 panel; the Limitations paragraph says so with E0's sums.",
    "uninformative": "Fewer than 8 pairs fall measurably with m; the k/m reading is untested for these pairs (their e_m are reported).",
}
P2_READ = {
    ("holds", "holds"): "One curve shape across four classes, and lowering caption noise moves a pair along it: remark in app:scale (heuristic, R10) with lambda-hat and gamma. The contribution may say 'falls at a rate set by the level of agreement'.",
    ("holds", "inconclusive"): "The cross-class regularity extends to sentence encoders; the within-pair evidence is inconclusive. Stated as an empirical regularity; no remark.",
    ("holds", "fails"): "The level predicts the slope across classes but not within a pair: the regularity reflects model class (d_int or residual shape) more than sigma. No remark (R10).",
    ("holds", "uninformative"): "The cross-class regularity extends; within-pair untestable (averaging hardly changes the level). Empirical statement only.",
    ("inconclusive", "holds"): "Within a pair the slope follows the level as predicted; between classes the new slopes lie between the level line and the ViT values (lambda-hat stated). The slope has a sigma component and a class component.",
    ("inconclusive", "inconclusive"): "Neither reading is supported; lambda-hat and gamma are stated.",
    ("inconclusive", "fails"): "Between classes mixed, within pair no sigma effect: the slope is mostly a property of the pairing.",
    ("inconclusive", "uninformative"): "Mixed between classes; within pair untestable.",
    ("fails", "holds"): "Between classes the ViT/modality value predicts the slope; within a pair the level still moves it as predicted. The S1.1 ordering is not explained by level alone; the slope has a pair-level component (X2).",
    ("fails", "inconclusive"): "The slope stays near the ViT value at higher levels; within pair inconclusive. The appendix sentence 'mirrors the ordering of agreement levels' is weakened.",
    ("fails", "fails"): "The slope is a property of the vision model and the modality pairing, not of the level. The modality reading of the contribution sentence is strengthened; the appendix sentence is weakened.",
    ("fails", "uninformative"): "The slope is a property of the vision model and the modality pairing, not of the level (as the fails/fails cell), without within-pair evidence.",
    ("uninformative", "holds"): "New levels lie where the two between-class predictions coincide; nothing new between classes. Within pair: the slope follows the level as predicted (as the first row).",
    ("uninformative", "inconclusive"): "Nothing new.",
    ("uninformative", "fails"): "Within pair no sigma effect.",
    ("uninformative", "uninformative"): "Nothing new about the level-slope regularity.",
}


def main():
    out_path = os.path.join(E.OUT, "summary.json")
    if os.path.exists(out_path):
        raise FileExistsError(out_path)
    V = json.load(open(E.VERDICTS))
    diag = json.load(open(os.path.join(E.DIAG, "diag_singlemodel.json")))
    D = V["descriptive"]
    lab = V["labels"]
    S = {"role": "e0-run", "created_utc": E.utcnow(), "plan": "ANALYSIS_PLAN_E0.md v1.0 + ADDENDUM 1 (implementation fix only)",
         "labels_in_plan_order": {k: lab[k] for k in ("P1", "P2b", "P2c")}}
    # ---------------- confirmatory statistics
    p1, p2b, p2c = V["P1"], V["P2b"], V["P2c"]
    dev = [r["pair"] for r in p1["per_pair"] if r["pair"] in (p1["informative_pairs"] or []) and abs(r["sum"]) > 0.05]
    S["P1"] = {"label": p1["label"], "n_informative": p1["n_informative"], "median_sum": p1["median_sum"],
               "frac_abs_sum_le_0.05": p1["frac_abs_sum_le"], "per_encoder_median_sum": p1["per_encoder_median_sum"],
               "deviating_informative_pairs_abs_sum_gt_0.05": dev,
               "per_pair": p1["per_pair"]}
    S["P2b"] = {"label": p2b["label"], "lambda_hat": p2b["lam"],
                "lambda_e": {e: v["lam"] for e, v in p2b["lam_e"].items()},
                "encoder_informative": {e: v["informative"] for e, v in p2b["lam_e"].items()},
                "median_abs_gap": p2b["median_abs_gap"], "guard_threshold": p2b["guard_threshold"],
                "leave_one_out": p2b["leave_one_out"], "wilcoxon_descriptive": p2b.get("wilcoxon_descriptive"),
                "median_b10_new": float(np.median([r["b"] for r in p2b["per_pair"]])),
                "median_L_new": float(np.median([r["L"] for r in p2b["per_pair"]])),
                "median_resid_vs_line": float(np.median([r["resid_vs_line"] for r in p2b["per_pair"]])),
                "median_b_minus_vit_competitor": float(np.median([r["y"] for r in p2b["per_pair"]]))}
    S["P2c"] = {"label": p2c["label"], "median_gamma": p2c["median_gamma"], "median_dln": p2c["median_dln"],
                "gamma_e": p2c["gamma_e"], "leave_encoder_out_median_gamma": p2c["leave_encoder_out_median_gamma"]}
    # ---------------- binding readings (§13)
    R = {"P1": P1_READ[p1["label"]].format(med=None if p1["median_sum"] is None else round(p1["median_sum"], 4),
                                           dev=", ".join(dev) or "none"),
         "P2b_x_P2c": P2_READ[(p2b["label"], p2c["label"])]}
    if p1["label"] == "fails":
        R["P1_fails_modifier"] = "Every P2b/P2c reading is stated as a regularity at fixed k, without the scale-parameter interpretation."
    medL = S["P2b"]["median_L_new"]
    R["B6_tension_row_applies"] = bool(p1["label"] == "holds" and p2b["label"] == "fails" and medL >= 0.2)
    if R["B6_tension_row_applies"]:
        R["B6"] = ("This matches the prior evidence of Schnaus et al. (k-elasticity of at least about 0.3 at fixed m for "
                   "sentence-encoder pairs), which was disclosed before the freeze.")
    # §13.3 level wording
    lv = D["P3a_level"]

    def med(arm, cls):
        return lv[arm][cls]["L_k10_m1024"][0]
    arms = {"P": {"sentence": med("P", E.NEW_CLASS), "lm_token": med("S1.1-clip", "cross-modal"),
                  "vision": med("S1.1-clip", "cross-vision")},
            "C": {"sentence": med("C", E.NEW_CLASS), "lm_token": med("ctrold", "cross-modal"),
                  "vision": med("ctrold", "cross-vision")}}
    above_v = all(a["sentence"] > a["vision"] for a in arms.values())
    above_lm = all(a["sentence"] > a["lm_token"] for a in arms.values())
    not_above_lm = all(a["sentence"] <= a["lm_token"] for a in arms.values())
    between = all(a["lm_token"] < a["sentence"] <= a["vision"] for a in arms.values())
    word = ("above vision-vision" if above_v else "between" if between else "not above LM tokens" if not_above_lm else "mixed")
    R["level_wording_13_3"] = {"word": word, "arms": arms}
    R["R1_clip_caveat_encoders"] = [e for e in diag.get("flag_below_0.8", []) if e in E.ENCODERS]
    S["readings_13"] = R
    # ---------------- descriptive pieces used in the report
    S["P3a_level"] = lv
    S["P3b_caption"] = {k: {"median_dlnL": v["median_dlnL"], "median_db10": v["median_db10"]} for k, v in D["P3b_caption"].items()}
    S["P3c_preprocessing"] = {k: {"median_dlnL": v["median_dlnL"], "median_db10": v["median_db10"]}
                              for k, v in D["P3c_preprocessing"].items()}
    S["P3e_encoders"] = D["P3e_encoders"]
    S["P3f_ordering"] = D["P3f_ordering"]
    S["P3g_shared_k_over_m"] = D["P3g_shared_k_over_m"]
    sch = D["P3h_schnaus"]
    S["P3h_schnaus"] = {arm: {"dinov2_b__mpnet": {k: v for k, v in sch[arm].items() if k.startswith("dinov2_b__mpnet|")},
                              "median_L_m10000_k100": float(np.median([v["L_m10000"] for k, v in sch[arm].items()
                                                                       if k.endswith("|k100") and "L_m10000" in v])),
                              "median_L_m10000_k10": float(np.median([v["L_m10000"] for k, v in sch[arm].items()
                                                                      if k.endswith("|k10") and "L_m10000" in v]))}
                        for arm in sch}
    S["P2_descriptive"] = {k: V["P2_descriptive"][k] for k in ("within_vit_permutation", "text_model_level")}
    S["secondary_labels"] = {n: {c: r.get("label", r.get("label_if_rule_applied")) for c, r in d.items()}
                             for n, d in V["secondary"].items()}
    S["secondary_P2b_lambda"] = {c: r.get("lam") for c, r in V["secondary"]["P2b"].items()}
    S["secondary_P2c_median_gamma"] = {c: r.get("median_gamma") for c, r in V["secondary"]["P2c_other_k"].items()}
    S["secondary_P1_median_sum"] = {c: r.get("median_sum") for c, r in V["secondary"]["P1"].items()}
    S["robustness_P1_per_draw_labels"] = D["robustness"].get("P1_per_draw_labels")
    S["corr_ek_minus_em_30VL"] = D.get("corr_ek_minus_em_30VL")
    S["diag_P3i"] = diag
    S["files_sha256"] = {os.path.relpath(p, ROOT): E.sha256_file(p) for p in
                         [E.VERDICTS, os.path.join(E.EVAL, "E0_tables.md"), E.CTR_CALIB, E.EXTRACTION_MANIFEST,
                          os.path.join(E.DIAG, "diag_singlemodel.json")]}
    S["pending"] = ["V-f independent implementation (role e0-indep)", "review + e0_guard.py audit (role e0-review)",
                    "E0_summary.{json,md} and DONE (after V-f and review)", "exploratory X1-X3 (X4, X5 computed in eval)",
                    "conditional Q1 and B7 (need the author's logged approval)"]
    h = E.write_json_new(out_path, S)
    print("summary.json", h)
    print(json.dumps(E.jsonable({"labels": S["labels_in_plan_order"], "readings": R}), indent=1))


if __name__ == "__main__":
    main()
