#!/usr/bin/env python3
"""E0 review: numbers proposed for the v4 sentences (descriptive; own code of e0_review_recompute.py).
    OMP_NUM_THREADS=1 taskset -c 0-5 venv/bin/python -I analysis/e0_sentence/review/e0_review_paper_numbers.py --out <json>
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e0_review_recompute as R
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
if os.path.exists(a.out):
    sys.exit("refusing to overwrite")
cell = {arm: R.drawmean_cells(pd.read_parquet(os.path.join(R.OBSD, f"obs_{arm}.parquet"))) for arm in ("P", "A", "C")}
s11 = pd.read_parquet(os.path.join(R.S11, "obs/obs_clip.parquet")); s11 = s11[s11.primary == True]
cV = R.drawmean_cells(s11[s11.cls == "cross-modal"])
out = {"label": "DESCRIPTIVE (e0-review)"}
out["class_median_k10"] = {f"sentence_m{m}": float(cell["P"][(cell["P"].k == 10) & (cell["P"].m == m)].s.median()) for m in R.M_GRID}
out["class_median_k10"].update({f"lm_token_m{m}": float(cV[(cV.k == 10) & (cV.m == m)].s.median()) for m in R.M_GRID})
def lv(c):
    return c[(c.k == 10) & (c.m == 1024)].set_index("pair").s
out["median_ratio_L10_1024"] = {"A_over_P": float((lv(cell["A"]) / lv(cell["P"])).median()), "C_over_P": float((lv(cell["C"]) / lv(cell["P"])).median())}
b, _ = R.draw_slopes(pd.read_parquet(os.path.join(R.OBSD, "obs_P.parquet")))
out["sentence_median_b10"] = float(b[b.k == 10].b.median())
el = R.elasticities(cell["P"], [10, 100])
out["sentence_sum_range_k10_100"] = [float(min(v["sum"] for v in el.values())), float(max(v["sum"] for v in el.values()))]
json.dump(out, open(a.out, "x"), indent=1)
print(json.dumps(out, indent=1))
