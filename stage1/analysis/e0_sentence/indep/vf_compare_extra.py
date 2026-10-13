"""V-f step 6 (role e0-indep): the two P1 companions that E0 records only in the
run-role results/E0_sentence/summary.json (not in VERDICTS_E0.json):
corr(e_k, -e_m) over the 30 VL pairs of arm P and the per-draw P1 labels.
Also the own-obs vs E0-obs statistics consistency (vf_stats.json).

Usage: python vf_compare_extra.py <outdir> <vf_stats.json>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf_common as C  # noqa: E402

SUMM = os.path.join(C.ROOT, "results", "E0_sentence", "summary.json")


def walk(x, y, p, out):
    if isinstance(x, dict):
        for k in x:
            walk(x[k], y[k], p + "/" + k, out)
    elif isinstance(x, list):
        for i, (u, v) in enumerate(zip(x, y)):
            walk(u, v, p + f"[{i}]", out)
    elif isinstance(x, bool) or isinstance(x, str):
        if x != y:
            out["str_diffs"].append([p, x, y])
    elif isinstance(x, (int, float)) and y is not None:
        d = abs(x - y)
        out["n"] += 1
        if d > out["max"]:
            out["max"], out["where"] = d, p


def main(outdir, stats_path):
    st = json.load(open(stats_path))
    a = st["from_e0_obs"]
    s = json.load(open(SUMM))
    res = dict(summary=dict(path=SUMM, sha256=C.sha256_file(SUMM)),
               vf_stats=dict(path=stats_path, sha256=C.sha256_file(stats_path)))
    res["corr_ek_minus_em_30VL"] = dict(mine=a["P1"]["corr_ek_minus_em_30VL"], e0=s["corr_ek_minus_em_30VL"],
                                        absdiff=abs(a["P1"]["corr_ek_minus_em_30VL"] - s["corr_ek_minus_em_30VL"]))
    res["P1_per_draw_labels"] = dict(mine={d: v["label"] for d, v in a["P1"]["per_draw"].items()},
                                     e0=s["robustness_P1_per_draw_labels"])
    res["P1_per_draw_labels"]["equal"] = res["P1_per_draw_labels"]["mine"] == res["P1_per_draw_labels"]["e0"]
    res["e0_summary_lambda_hat"] = s.get("lambda_hat")
    out = dict(n=0, max=0.0, where=None, str_diffs=[])
    walk(st["from_e0_obs"], st["from_vf_obs"], "", out)
    res["own_obs_vs_e0_obs_statistics"] = out
    sha = C.write_new_json(os.path.join(outdir, "vf_compare_extra.json"), res)
    print(json.dumps(res, indent=1, default=str))
    print("vf_compare_extra.json", sha)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
