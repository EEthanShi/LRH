#!/usr/bin/env python3
"""E0 review, EXPLORATORY (post hoc, written after the verdicts; no confirmatory weight).
Reuses the own-code functions of e0_review_recompute.py (same folder). Questions:
 (1) does P1 survive when cells near chance are removed (m >= 2048 only)?
 (2) how close to chance are the k = 100, m = 256 cells of the new pairs?
 (3) the k = 1 departure: medians of e_k + e_m on K = {1,10} and {1,10,100}, new pairs vs S1.1 VL pairs.
 (4) per-ViT medians of observed slope, line prediction and competitor in arms P and A.
    OMP_NUM_THREADS=1 taskset -c 0-5 venv/bin/python -I analysis/e0_sentence/review/e0_review_explore.py --out <json>
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e0_review_recompute as R  # own review code only

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
if os.path.exists(a.out):
    sys.exit("refusing to overwrite")
out = {"label": "EXPLORATORY (e0-review, post hoc)"}
P = pd.read_parquet(os.path.join(R.OBSD, "obs_P.parquet"))
s11 = pd.read_parquet(os.path.join(R.S11, "obs/obs_clip.parquet")); s11 = s11[s11.primary == True]
vl = s11[s11.cls == "cross-modal"]
cP, cV = R.drawmean_cells(P), R.drawmean_cells(vl)
def med_sum(c, K, mmin=0):
    el = R.elasticities(c[c.m >= mmin].assign(), K) if mmin == 0 else None
    if mmin:
        cc = c[c.m >= mmin]
        el = {}
        for p, g in cc[cc.k.isin(K)].groupby("pair"):
            X = np.column_stack([np.ones(len(g)), np.log(g.k.astype(float)), np.log(g.m.astype(float))])
            b, *_ = np.linalg.lstsq(X, g.lns.values, rcond=None)
            el[p] = {"e_k": b[1], "e_m": b[2], "sum": b[1] + b[2]}
    s = np.array([v["sum"] for v in el.values()])
    return {"median_sum": float(np.median(s)), "max_abs_sum": float(np.max(np.abs(s))),
            "frac_abs_le_0.05": float(np.mean(np.abs(s) <= 0.05)), "median_e_m": float(np.median([v["e_m"] for v in el.values()]))}
out["P1_m_ge_2048"] = {"new": med_sum(cP, [10, 100], 2048), "s11_vl": med_sum(cV, [10, 100], 2048)}
out["P1_k10_100_all_m"] = {"new": med_sum(cP, [10, 100]), "s11_vl": med_sum(cV, [10, 100])}
out["k1_departure"] = {"new_K1_10": med_sum(cP, [1, 10]), "s11_vl_K1_10": med_sum(cV, [1, 10]),
                       "new_K1_10_100": med_sum(cP, [1, 10, 100]), "s11_vl_K1_10_100": med_sum(cV, [1, 10, 100])}
def ratio_chance(c, m, k):
    x = c[(c.m == m) & (c.k == k)]
    r = x.s / (k / (m - 1))
    return {"median": float(r.median()), "min": float(r.min()), "max": float(r.max())}
out["obs_over_chance"] = {f"new_m{m}_k{k}": ratio_chance(cP, m, k) for m, k in [(256, 100), (2048, 100), (16384, 10), (1024, 10)]}
out["obs_over_chance"].update({f"s11vl_m{m}_k{k}": ratio_chance(cV, m, k) for m, k in [(256, 100), (16384, 10)]})
for arm in ("P", "A"):
    X = pd.read_parquet(os.path.join(R.OBSD, f"obs_{arm}.parquet"))
    b, _ = R.draw_slopes(X); c = R.drawmean_cells(X); L, lnL = R.level(c)
    b10 = b[b.k == 10].set_index("pair").b.to_dict()
    per = {}
    for v in R.VITS:
        ps = [p for p in b10 if p.startswith(v + "__")]
        per[v] = {"median_L": float(np.median([L[p] for p in ps])), "median_b10": float(np.median([b10[p] for p in ps])),
                  "line": float(np.median([R.bhat_star(lnL[p]) for p in ps])), "competitor": R.COMP[v]}
    out[f"per_vit_{arm}"] = per
json.dump(out, open(a.out, "x"), indent=1)
print(json.dumps(out, indent=1))
