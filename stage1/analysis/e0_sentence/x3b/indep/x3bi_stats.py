"""X3b independent check: statistics, readings and comparison with the run.

From cells (mine, and separately the run's obs parquets) compute draw-mean
cells, per-draw slopes, elasticities, levels, per-pair records, set
summaries and the readings of X3B_READINGS.md section 5; compare with
results/E0_sentence/x3b/X3B_results.json (numbers <= 1e-12 absolute, labels
and integers identical) and with the run's obs/ and eval/ parquets.

Usage: python x3bi_stats.py <outdir>
EXPLORATORY.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/hrh_v2_stage1_20261008"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x3bi_common as C  # noqa: E402

TOL = 1e-12
LN_INTERP = (math.log(10000.0) - math.log(8192.0)) / math.log(2.0)
LAST_SETS = ("A1", "A3", "A4", "B", "A2_r025", "A2_r075", "A2_r10")
MID_SET = {"A1": "A1ref", "A3": "A3ref", "A4": "A4ref", "B": "Bref",
           "A2_r025": "A2_r025", "A2_r075": "A2_r075", "A2_r10": "A2_r10"}
CELLS3 = {"L10_1024": (10, 1024), "L100_8192": (100, 8192), "L100_10000": (100, 10000)}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def drawmean(cells: pd.DataFrame) -> pd.DataFrame:
    g = cells.groupby(["set", "pair", "m", "k"], sort=True)
    out = g.agg(obs=("obs", "mean"), obs_draw_sd=("obs", lambda x: float(np.std(x.to_numpy(), ddof=1))),
                obs_allH2=("obs_allH2", "mean"),
                obs_allH2_draw_sd=("obs_allH2", lambda x: float(np.std(x.to_numpy(), ddof=1))),
                n_anchors=("n_anchors", "mean"), n_allH2=("n_allH2", "mean"),
                n_draws=("obs", "size")).reset_index()
    out["eps"] = 0.5 / (out["n_draws"] * out["n_anchors"] * out["k"])
    out["eps_allH2"] = 0.5 / (out["n_draws"] * out["n_allH2"] * out["k"])
    out["chance"] = out["k"] / (out["m"] - 1)
    return out


def ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def drawslopes(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, p, k, d), g in cells.groupby(["set", "pair", "k", "draw"], sort=True):
        g = g.sort_values("m")
        assert len(g) == 7
        x = np.log(g["m"].to_numpy(float))
        y = np.log(np.maximum(g["obs"].to_numpy(), 0.5 / (g["n_anchors"].to_numpy() * k)))
        ya = np.log(np.maximum(g["obs_allH2"].to_numpy(), 0.5 / (g["n_allH2"].to_numpy() * k)))
        rows.append({"set": s, "pair": p, "k": k, "draw": d, "slope_obs": ols_slope(x, y),
                     "slope_obs_allH2": ols_slope(x, ya)})
    return pd.DataFrame(rows)


def slope_class(b):
    return "rising" if b >= 0.1 else ("falling" if b <= -0.1 else "flat")


def slopes(ds: pd.DataFrame) -> pd.DataFrame:
    g = ds.groupby(["set", "pair", "k"], sort=True)
    out = g.agg(slope_obs=("slope_obs", "mean"), slope_obs_allH2=("slope_obs_allH2", "mean"),
                slope_obs_draw_sd=("slope_obs", lambda x: float(np.std(x.to_numpy(), ddof=1))),
                n_draws=("slope_obs", "size")).reset_index()
    out["slope_class"] = out["slope_obs"].map(slope_class)
    out["slope_class_allH2"] = out["slope_obs_allH2"].map(slope_class)
    return out


def elasticity(dm: pd.DataFrame, kset) -> pd.DataFrame:
    rows = []
    for (s, p), g in dm.groupby(["set", "pair"], sort=True):
        g = g[g.k.isin(kset)]
        y = np.log(np.maximum(g["obs"].to_numpy(), g["eps"].to_numpy()))
        X = np.column_stack([np.ones(len(g)), np.log(g["k"].to_numpy(float)), np.log(g["m"].to_numpy(float))])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        rows.append({"set": s, "pair": p, "e_k": float(beta[1]), "e_m": float(beta[2]),
                     "sum": float(beta[1] + beta[2]), "rmse": float(np.sqrt(np.mean(res ** 2))),
                     "n_cells": int(len(g)), "k_set": "k" + "_".join(str(k) for k in kset)})
    return pd.DataFrame(rows)


def levels(dm: pd.DataFrame, col: str, epscol: str) -> pd.DataFrame:
    rows = []
    for (s, p, k), g in dm.groupby(["set", "pair", "k"], sort=True):
        g = g.set_index("m")
        r = {"set": s, "pair": p, "k": k}
        for m in C.M_GRID:
            r[f"L_m{m}"] = float(g.loc[m, col])
        a = math.log(max(g.loc[8192, col], g.loc[8192, epscol]))
        b = math.log(max(g.loc[16384, col], g.loc[16384, epscol]))
        r["L_m10000"] = math.exp(a + LN_INTERP * (b - a))
        rows.append(r)
    L = pd.DataFrame(rows)
    for m in (8192, 16384):
        piv = L.pivot_table(index=["set", "pair"], columns="k", values=f"L_m{m}")
        rat = (piv[100] / piv[10]).rename(f"ratio_k100_over_k10_m{m}").reset_index()
        rat["k"] = 10
        L = L.merge(rat, on=["set", "pair", "k"], how="left")
    return L


# ---------------------------------------------------------------------------
# per-pair records and readings
# ---------------------------------------------------------------------------

def pair_records(dm, sl, el, lv, lva):
    """dict (set, pair) -> record of the per-pair quantities."""
    recs = {}
    E = {kk: el[kk].set_index(["set", "pair"]) for kk in el}
    S = sl.set_index(["set", "pair", "k"])
    LV = lv.set_index(["set", "pair", "k"])
    LA = lva.set_index(["set", "pair", "k"])
    for (s, p) in sorted(set(zip(dm["set"], dm["pair"]))):
        r = {"L10_1024": LV.loc[(s, p, 10), "L_m1024"], "L100_8192": LV.loc[(s, p, 100), "L_m8192"],
             "L100_10000": LV.loc[(s, p, 100), "L_m10000"], "L10_10000": LV.loc[(s, p, 10), "L_m10000"],
             "allH2_L10_1024": LA.loc[(s, p, 10), "L_m1024"],
             "allH2_L100_10000": LA.loc[(s, p, 100), "L_m10000"]}
        for k in C.KS:
            r[f"b{k}"] = S.loc[(s, p, k), "slope_obs"]
        e = E["k10_100"].loc[(s, p)]
        r.update({"e_k": e["e_k"], "e_m": e["e_m"], "sum": e["sum"], "rmse": e["rmse"],
                  "sum_k1_10_100": E["k1_10_100"].loc[(s, p), "sum"], "sum_k1_10": E["k1_10"].loc[(s, p), "sum"]})
        recs[(s, p)] = {kk: float(v) for kk, v in r.items()}
    return recs


def parse_pair(p):
    a, b = p.split("__")
    v, tv = a.split(".")
    t, tt = b.split(".")
    return v, tv, t, tt


REL_OF = {"r025": "0.25", "r05": "0.5", "r075": "0.75", "r10": "1.0", "n5": "n5"}
PRE_OF_SET = {"A1": "clip", "A3": "l2", "A4": "ctr", "B": "ctr", "A2_r025": "clip", "A2_r075": "clip", "A2_r10": "clip"}


def per_pair_tables(recs):
    out = {}
    for s in LAST_SETS:
        lst = []
        lasts = sorted([p for (ss, p) in recs if ss == s and parse_pair(p)[1] == "r10"],
                       key=lambda p: (C.VITS.index(parse_pair(p)[0]),
                                      (C.LMS + C.ENCS).index(parse_pair(p)[2])))
        assert len(lasts) == 15, (s, len(lasts))
        for p in lasts:
            v, tv, t, tt = parse_pair(p)
            pm = f"{v}.r05__{t}.{tt}"
            sm = MID_SET[s]
            r = {"set": s, "pair": p, "pre": PRE_OF_SET[s], "vit": v, "rel_v": "1.0", "text": t, "rel_t": REL_OF[tt]}
            r.update(recs[(s, p)])
            r.update({"set_mid": sm, "pair_mid": pm, "pre_mid": PRE_OF_SET[s], "rel_v_mid": "0.5", "rel_t_mid": REL_OF[tt]})
            for kk, vv in recs[(sm, pm)].items():
                r[kk + "_mid"] = vv
            for c in CELLS3:
                r["R_" + c] = r[c] / r[c + "_mid"]
            r["_allH2_R_L10_1024"] = r["allH2_L10_1024"] / r["allH2_L10_1024_mid"]
            r["_allH2_R_L100_10000"] = r["allH2_L100_10000"] / r["allH2_L100_10000_mid"]
            lst.append(r)
        out[s] = lst
    return out


def stability(rows, suffix=""):
    s = np.array([r["sum" + suffix] for r in rows])
    em = np.array([r["e_m" + suffix] for r in rows])
    s3 = np.array([r["sum_k1_10_100" + suffix] for r in rows])
    n = len(rows)
    nin = int((np.abs(s) <= 0.05).sum())
    inf = np.abs(em) >= 0.10
    si = s[inf]
    p1 = bool(inf.sum() >= 8 and abs(np.median(si)) <= 0.03 and (np.abs(si) <= 0.05).mean() >= 0.8)
    return {"n": n, "n_abs_sum_le_0.05": nin, "frac": nin / n, "median_sum": float(np.median(s)),
            "range_sum": [float(s.min()), float(s.max())], "n_informative_abs_em_ge_0.10": int(inf.sum()),
            "label": "stays close to stable at fixed k/m" if nin >= math.ceil(0.8 * n) else "not stable at fixed k/m",
            "E0_P1_thresholds_descriptive": p1,
            "median_sum_k1_10_100": float(np.median(s3)),
            "n_abs_sum_k1_10_100_le_0.05": int((np.abs(s3) <= 0.05).sum())}


def depth_label(R):
    if R >= 2:
        return "the level is mostly a depth effect"
    if R >= 1.25:
        return "the level is partly a depth effect"
    return "the level is not a depth effect"


def set_summary(s, rows):
    out = {"set": s, "n": len(rows)}
    for c in CELLS3:
        R = np.array([r["R_" + c] for r in rows])
        out[f"median_last_{c}"] = float(np.median([r[c] for r in rows]))
        out[f"median_mid_{c}"] = float(np.median([r[c + "_mid"] for r in rows]))
        out[f"median_R_{c}"] = float(np.median(R))
        out[f"range_R_{c}"] = [float(R.min()), float(R.max())]
        out[f"n_last_gt_mid_{c}"] = int(sum(r[c] > r[c + "_mid"] for r in rows))
    out["per_vit_median_R"] = {v: {c: float(np.median([r["R_" + c] for r in rows if r["vit"] == v])) for c in CELLS3}
                               for v in C.VITS}
    out["per_vit_median_last"] = {v: {c: float(np.median([r[c] for r in rows if r["vit"] == v])) for c in CELLS3}
                                  for v in C.VITS}
    out["median_b"] = {f"b{k}": float(np.median([r[f"b{k}"] for r in rows])) for k in C.KS}
    out["median_b_mid"] = {f"b{k}": float(np.median([r[f"b{k}_mid"] for r in rows])) for k in C.KS}
    out["n_b10_le_-0.1"] = int(sum(r["b10"] <= -0.1 for r in rows))
    out["fall_label"] = "still falls with m" if out["median_b"]["b10"] < -0.1 else "no longer falls clearly with m"
    out["stability"] = stability(rows)
    out["stability_mid"] = stability(rows, "_mid")
    out["depth_label_at_L10_1024"] = depth_label(out["median_R_L10_1024"])
    out["median_allH2_R_L10_1024"] = float(np.median([r["_allH2_R_L10_1024"] for r in rows]))
    out["median_allH2_R_L100_10000"] = float(np.median([r["_allH2_R_L100_10000"] for r in rows]))
    return out


def qb(rows, stab_label, fall_label):
    nd = [r for r in rows if r["vit"] != "dinov2_b"]

    def size_label(x):
        return "a large rise, as for DINOv2-B" if x >= 2 else ("a moderate rise" if x >= 1.25 else "little or no rise")

    n = int(sum(r["L100_10000"] > r["L100_10000_mid"] for r in rows))
    lab = ("the rise with depth is not specific to DINOv2-B" if n >= 12 else
           ("the rise with depth holds for some ViTs only" if n >= 6 else
            "the rise with depth is largely specific to DINOv2-B"))
    Rnd = np.array([r["R_L100_10000"] for r in nd])
    out = {"cell": "L(100, 10000)", "n_last_gt_mid": n, "share": n / len(rows), "label": lab,
           "n_last_gt_mid_non_dinov2": int(sum(r["L100_10000"] > r["L100_10000_mid"] for r in nd)),
           "n_non_dinov2": len(nd), "median_R_non_dinov2": float(np.median(Rnd)),
           "range_R_non_dinov2": [float(Rnd.min()), float(Rnd.max())],
           "size_label_descriptive": size_label(float(np.median(Rnd))),
           "descriptive_other_cells": {}, "stability_label": stab_label, "fall_label_descriptive": fall_label}
    for c in ("L10_1024", "L100_8192"):
        out["descriptive_other_cells"][c] = {
            "n_last_gt_mid": int(sum(r[c] > r[c + "_mid"] for r in rows)),
            "n_last_gt_mid_non_dinov2": int(sum(r[c] > r[c + "_mid"] for r in nd)),
            "median_R_non_dinov2": float(np.median([r["R_" + c] for r in nd]))}
    return out


def everything(cells: pd.DataFrame):
    dm = drawmean(cells)
    ds = drawslopes(cells)
    sl = slopes(ds)
    el = {"k10_100": elasticity(dm, (10, 100)), "k1_10_100": elasticity(dm, (1, 10, 100)),
          "k1_10": elasticity(dm, (1, 10))}
    lv = levels(dm, "obs", "eps")
    lva = levels(dm, "obs_allH2", "eps_allH2")
    recs = pair_records(dm, sl, el, lv, lva)
    pp = per_pair_tables(recs)
    sets = {s: set_summary(s, pp[s]) for s in LAST_SETS}
    res = {"sets": sets,
           "per_pair": {s: [{k: v for k, v in r.items() if not k.startswith("_")} for r in pp[s]] for s in LAST_SETS},
           "Q_a": {"binding_set": "A1", "depth_label": sets["A1"]["depth_label_at_L10_1024"],
                   "fall_label": sets["A1"]["fall_label"], "stability_label": sets["A1"]["stability"]["label"],
                   "median_last_L10_1024_vs_0.04295": [sets["A1"]["median_last_L10_1024"], sets["A1"]["median_mid_L10_1024"]]},
           "Q_b": qb(pp["B"], sets["B"]["stability"]["label"], sets["B"]["fall_label"])}
    refs = {"S1.1 clip VL L(10,1024)": float(np.median([recs[k]["L10_1024"] for k in recs if k[0] == "A1ref"])),
            "E0 CA L(10,1024)": float(np.median([recs[k]["L10_1024"] for k in recs if k[0] == "Bref"]))}
    vv = [recs[k]["L10_1024"] for k in recs if k[0] == "VVref"]
    if vv:
        refs["S1.1 clip vision L(10,1024)"] = float(np.median(vv))
    res["references_mid_depth"] = refs
    # cross-reading 5.7 (descriptive): per ViT median R, A4 vs B
    res["cross_reading"] = {v: {"A4": sets["A4"]["per_vit_median_R"][v], "B": sets["B"]["per_vit_median_R"][v]} for v in C.VITS}
    return res, {"dm": dm, "ds": ds, "sl": sl, "el": el, "lv": lv, "lva": lva}


# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------

def flatten(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, p + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flatten(v, p + f"[{i}]")
    else:
        yield p, o


def index_per_pair(res):
    """Re-key per_pair lists by pair name so that order cannot matter."""
    out = dict(res)
    out["per_pair"] = {s: {r["pair"]: r for r in lst} for s, lst in res["per_pair"].items()}
    return out


def compare_json(theirs, mine):
    skip = ("/role", "/status", "/readings_sha256", "/readings_status_utc", "/provenance_sha256", "/validation",
            "/reference_consistency", "/compute", "/code_sha256", "/created_utc")
    T = dict(flatten(index_per_pair(theirs)))
    M = dict(flatten(index_per_pair(mine)))
    n_num = n_other = 0
    maxd = 0.0
    worst = None
    bad, missing = [], []
    for p, tv in T.items():
        if p.startswith(skip):
            continue
        if p not in M:
            missing.append(p)
            continue
        mv = M[p]
        if isinstance(tv, (bool, str)) or tv is None:
            n_other += 1
            if type(tv) is not type(mv) or tv != mv:
                bad.append((p, tv, mv))
        elif isinstance(tv, int):
            n_other += 1
            if isinstance(mv, bool) or not isinstance(mv, (int, np.integer)) or int(mv) != tv:
                bad.append((p, tv, mv))
        else:
            n_num += 1
            dlt = abs(float(tv) - float(mv))
            if not (dlt <= TOL):
                bad.append((p, tv, mv))
            if dlt > maxd:
                maxd, worst = dlt, p
    extra = [p for p in M if p not in T and not p.startswith(("/cross_reading", "/references_mid_depth"))]
    return {"n_numeric": n_num, "n_label_bool_int": n_other, "max_abs_diff_numeric": maxd, "worst_field": worst,
            "n_bad": len(bad), "bad": [list(map(str, b)) for b in bad[:200]], "n_missing_in_mine": len(missing),
            "missing": missing[:100], "n_extra_in_mine": len(extra), "extra": extra[:50]}


def compare_frames(a, b, keys, cols, tol=TOL):
    m = a.merge(b, on=keys, how="outer", suffixes=("_m", "_r"), indicator=True)
    out = {"rows_mine": len(a), "rows_run": len(b), "rows_matched": int((m["_merge"] == "both").sum())}
    mm = m[m["_merge"] == "both"]
    for c in cols:
        x, y = mm[c + "_m"], mm[c + "_r"]
        if x.dtype == object or y.dtype == object or str(x.dtype).startswith(("str", "string")):
            out[c] = {"n_unequal": int((x.astype(str) != y.astype(str)).sum())}
        else:
            xa, ya = x.to_numpy(float), y.to_numpy(float)
            both_nan = np.isnan(xa) & np.isnan(ya)
            d = np.where(both_nan, 0.0, np.abs(xa - ya))
            d = np.where(np.isnan(d), np.inf, d)
            out[c] = {"max_abs_diff": float(d.max()) if len(d) else 0.0, "n_gt_tol": int((d > tol).sum())}
    return out


def main():
    outdir = sys.argv[1]
    t0 = time.time()
    run = os.path.join(C.ROOT, "results/E0_sentence/x3b")
    theirs = json.load(open(os.path.join(run, "X3B_results.json")))
    cells = pd.read_parquet(os.path.join(outdir, "obs", "x3bi_cells.parquet"))
    mine, T = everything(cells)
    # run-side cells through my statistics (separates cell-level from statistic-level differences)
    o1 = pd.read_parquet(os.path.join(run, "obs/obs_X3B.parquet"))
    o2 = pd.read_parquet(os.path.join(run, "obs/obs_X3B_ref.parquet"))
    rc = pd.concat([o1, o2.drop(columns=["orig_config"])], ignore_index=True).rename(columns={"config": "set"})
    mine_from_run, TR = everything(rc)
    cmp_main = compare_json(theirs, mine)
    cmp_runcells = compare_json(theirs, mine_from_run)
    # cell-level comparison against the run's obs parquets
    myc = cells[cells.set != "VVref"].copy()
    myc["obs_sd"] = myc["obs_sd1"]
    myc["obs_allH2_sd"] = myc["obs_allH2_sd1"]
    cc = compare_frames(myc, rc, ["set", "pair", "draw", "m", "k"],
                        ["obs", "obs_allH2", "n_anchors", "n_allH2", "chance", "obs_sd", "obs_allH2_sd",
                         "n_tieA", "n_tieB", "n_clean", "obs_clean", "neartie_A", "neartie_B",
                         "neartie_allH2_A", "neartie_allH2_B", "is_fit_m"])
    myc["obs_sd"] = myc["obs_sd0"]
    myc["obs_allH2_sd"] = myc["obs_allH2_sd0"]
    cc_sd0 = compare_frames(myc, rc, ["set", "pair", "draw", "m", "k"], ["obs_sd", "obs_allH2_sd"])
    myc["n_tieA"], myc["n_tieB"] = myc["n_tieA_allH2"], myc["n_tieB_allH2"]
    cc_tie_allH2 = compare_frames(myc, rc, ["set", "pair", "draw", "m", "k"], ["n_tieA", "n_tieB"])
    # eval parquets
    ev = {}
    E = lambda f: pd.read_parquet(os.path.join(run, "eval", f))  # noqa: E731
    dmm = T["dm"][T["dm"].set != "VVref"]
    ev["cells"] = compare_frames(dmm, E("X3B_cells.parquet"), ["set", "pair", "m", "k"],
                                 ["obs", "obs_draw_sd", "obs_allH2", "obs_allH2_draw_sd", "n_anchors", "n_allH2",
                                  "eps", "chance"])
    ev["drawslopes"] = compare_frames(T["ds"][T["ds"].set != "VVref"], E("X3B_drawslopes.parquet"),
                                      ["set", "pair", "k", "draw"], ["slope_obs", "slope_obs_allH2"])
    ev["slopes"] = compare_frames(T["sl"][T["sl"].set != "VVref"], E("X3B_slopes.parquet"), ["set", "pair", "k"],
                                  ["slope_obs", "slope_obs_allH2", "slope_obs_draw_sd", "slope_class", "slope_class_allH2"])
    el_all = pd.concat(list(T["el"].values()), ignore_index=True)
    el_all = el_all[el_all.set != "VVref"]
    ev["elasticity"] = compare_frames(el_all, E("X3B_elasticity.parquet"), ["set", "pair", "k_set"],
                                      ["e_k", "e_m", "sum", "rmse", "n_cells"])
    lcols = [f"L_m{m}" for m in C.M_GRID] + ["L_m10000", "ratio_k100_over_k10_m8192", "ratio_k100_over_k10_m16384"]
    ev["levels"] = compare_frames(T["lv"][T["lv"].set != "VVref"], E("X3B_levels.parquet"), ["set", "pair", "k"], lcols)
    ev["levels_allH2"] = compare_frames(T["lva"][T["lva"].set != "VVref"], E("X3B_levels_allH2.parquet"),
                                        ["set", "pair", "k"], lcols)
    # mid-depth references against frozen S1.1 / E0 outputs (independent of the run)
    fz = {}
    frozen = {"A1ref": "results/S1_1/v11/S1_1_v11_cells_clip.parquet",
              "VVref": "results/S1_1/v11/S1_1_v11_cells_clip.parquet",
              "A3ref": "results/S1_1/v11/S1_1_v11_cells_l2.parquet",
              "A4ref": "results/E0_sentence/eval/E0_cells_ctrold.parquet",
              "Bref": "results/E0_sentence/eval/E0_cells_CA.parquet"}
    for st, f in frozen.items():
        fr = pd.read_parquet(os.path.join(C.ROOT, f))
        mine_s = T["dm"][T["dm"].set == st].copy()
        mine_s["pair"] = mine_s["pair"].map(lambda p: "__".join(x.split(".")[0] for x in p.split("__")))
        fz[st] = {"file": f, **compare_frames(mine_s, fr[fr.pair.isin(mine_s.pair.unique())], ["pair", "m", "k"],
                                              ["obs", "obs_allH2", "n_anchors", "n_allH2", "eps", "obs_draw_sd"])}
    out = {"compare_json_from_my_cells": cmp_main, "compare_json_from_run_cells": cmp_runcells,
           "cells_vs_run_obs": cc, "cells_vs_run_obs_sd_ddof0": cc_sd0, "cells_vs_run_obs_ties_allH2": cc_tie_allH2,
           "eval_vs_run": ev, "frozen_refs": fz, "seconds": time.time() - t0}
    os.makedirs(os.path.join(outdir, "stats"), exist_ok=True)
    with open(os.path.join(outdir, "stats", "x3bi_stats.json"), "w") as fh:
        json.dump(mine, fh, indent=1)
    with open(os.path.join(outdir, "stats", "x3bi_stats_from_run_cells.json"), "w") as fh:
        json.dump(mine_from_run, fh, indent=1)
    with open(os.path.join(outdir, "stats", "x3bi_compare.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    for nm, t in T.items():
        if isinstance(t, dict):
            for kk, df in t.items():
                df.to_parquet(os.path.join(outdir, "stats", f"x3bi_{nm}_{kk}.parquet"), index=False)
        else:
            t.to_parquet(os.path.join(outdir, "stats", f"x3bi_{nm}.parquet"), index=False)
    print(json.dumps({k: (v if k.startswith("compare") else "...") for k, v in out.items()}, indent=1, default=str)[:6000])
    print("STATS DONE", time.time() - t0)


if __name__ == "__main__":
    main()
