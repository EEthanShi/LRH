"""x3-indep step 3: statistics, interpolation and readings from the cells,
then comparison with X3_results.json.

usage: x3i_stats.py <run_dir>

Definitions from ANALYSIS_PLAN_E0.md section 6 and X3_READINGS.md sections
3-8. Computed twice: from this role's own cells (x3i_cells.parquet) and,
as a second check of the statistics layer alone, from the run side's obs
cells (obs_X3.parquet + E0 obs_CA rows of the 3 reference pairs).
"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x3i_common as C  # noqa: E402

VIS = ["dinov2_b", "x3_cls_r10", "x3_tok_r10", "x3_tok_r05", "x3add_schnaus224"]
X3V = VIS[1:]
SCHNAUS = 0.38
BAND = [0.37, 0.39]
L_E0 = {"mpnet": 0.07715427364361675, "contriever": 0.07633979414898837,
        "qwen3emb06": 0.07717300965111769}            # X3_READINGS.md section 4
CTR_A, CTR_B = 0.15928637788145117, 0.2361611718219916  # ctr line, k = 10, m = 1024
KSETS = {"k10_100": [10, 100], "k1_10_100": [1, 10, 100], "k1_10": [1, 10]}
TOL = 1e-12


def label(share: float) -> str:
    if share >= 1:
        return "closes the whole gap (reaches or passes Schnaus' reading)"
    if share >= 0.5:
        return "closes most of the gap"
    if share >= 0.2:
        return "closes part of the gap"
    if share >= 0:
        return "closes little of the gap"
    return "widens the gap"


def slope_class(b: float) -> str:
    if b >= 0.1:
        return "rising"
    if b <= -0.1:
        return "falling"
    return "flat"


def ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    return float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))


def pair_tables(c: pd.DataFrame, col: str, ncol: str):
    """draw-mean s, draw sd, floor, per-draw slopes for one pair and one column."""
    sbar, sd, eps, nbar = {}, {}, {}, {}
    for k in C.KS:
        for m in C.M_GRID:
            x = c[(c.k == k) & (c.m == m)].sort_values("draw")
            if len(x) != C.N_DRAWS:
                raise RuntimeError("draw count")
            v = x[col].to_numpy(float)
            sbar[(m, k)] = float(np.mean(v))
            sd[(m, k)] = float(np.std(v, ddof=1))
            nbar[(m, k)] = float(np.mean(x[ncol].to_numpy(float)))
            eps[(m, k)] = 0.5 / (C.N_DRAWS * nbar[(m, k)] * k)
    dslopes = {}
    for k in C.KS:
        bs = []
        for d in range(C.N_DRAWS):
            x = c[(c.k == k) & (c.draw == d)].sort_values("m")
            s = x[col].to_numpy(float)
            e = 0.5 / (x[ncol].to_numpy(float) * k)
            bs.append(ols_slope(np.log(x.m.to_numpy(float)), np.log(np.maximum(s, e))))
        dslopes[k] = bs
    return sbar, sd, eps, nbar, dslopes


def lnL(sbar, eps, m, k):
    return math.log(max(sbar[(m, k)], eps[(m, k)]))


def L10000(sbar, eps, k):
    a = lnL(sbar, eps, 8192, k)
    b = lnL(sbar, eps, 16384, k)
    return math.exp(a + (math.log(10000) - math.log(8192)) / math.log(2) * (b - a))


def elasticity(sbar, eps, K):
    X, y = [], []
    for k in K:
        for m in C.M_GRID:
            X.append([1.0, math.log(k), math.log(m)])
            y.append(lnL(sbar, eps, m, k))
    X, y = np.asarray(X), np.asarray(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return float(beta[1]), float(beta[2]), float(np.sqrt(np.mean(res ** 2)))


def compute(cells: pd.DataFrame) -> dict:
    P = {}
    raw = {}
    for v in VIS:
        for enc in C.ENCODERS:
            pair = f"{v}__{enc}"
            c = cells[cells.pair == pair]
            sb, sd, ep, nb, ds = pair_tables(c, "obs", "n_anchors")
            sbH, sdH, epH, nbH, dsH = pair_tables(c, "obs_allH2", "n_allH2")
            raw[pair] = (sb, sd, ep, nb, ds, sbH, epH, nbH, dsH)
            r = {"pair": pair, "feature": v, "enc": enc,
                 "L_10_1024": sb[(1024, 10)], "L_100_8192": sb[(8192, 100)],
                 "L_10_10000": L10000(sb, ep, 10), "L_100_10000": L10000(sb, ep, 100),
                 "L_100_10000_allH2": L10000(sbH, epH, 100), "L_10_10000_allH2": L10000(sbH, epH, 10),
                 "ratio_k100_over_k10_m8192": sb[(8192, 100)] / sb[(8192, 10)],
                 "ratio_k100_over_k10_m16384": sb[(16384, 100)] / sb[(16384, 10)]}
            for k in C.KS:
                r[f"b{k}"] = float(np.mean(ds[k]))
                r[f"b{k}_draw_sd"] = float(np.std(ds[k], ddof=1))
                r[f"b{k}_class"] = slope_class(r[f"b{k}"])
                r[f"b{k}_allH2"] = float(np.mean(dsH[k]))
            for nm, K in KSETS.items():
                ek, em, rmse = elasticity(sb, ep, K)
                r[f"e_k_{nm}"], r[f"e_m_{nm}"], r[f"sum_{nm}"], r[f"rmse_{nm}"] = ek, em, ek + em, rmse
            r["p1_stable_abs_sum_le_0.05"] = abs(r["sum_k10_100"]) <= 0.05
            r["p1_informative_abs_em_ge_0.10"] = abs(r["e_m_k10_100"]) >= 0.10
            r["ratio_to_schnaus_0.38"] = r["L_100_10000"] / SCHNAUS
            pred = CTR_A + CTR_B * math.log(max(sb[(1024, 10)], ep[(1024, 10)]))
            r["ctr_line_pred_b10"] = pred
            r["ctr_line_pred_b10_capped"] = min(pred, 0.0)
            r["ctr_line_resid_b10"] = r["b10"] - pred
            P[pair] = r
    # reference-dependent quantities
    for v in VIS:
        for enc in C.ENCODERS:
            r = P[f"{v}__{enc}"]
            ref = P[f"dinov2_b__{enc}"]
            Le0 = L_E0[enc]
            r["share_0.38"] = math.log(r["L_100_10000"] / Le0) / math.log(SCHNAUS / Le0)
            r["share_0.37"] = math.log(r["L_100_10000"] / Le0) / math.log(BAND[0] / Le0)
            r["share_0.39"] = math.log(r["L_100_10000"] / Le0) / math.log(BAND[1] / Le0)
            LH = ref["L_100_10000_allH2"]
            r["share_allH2_0.38"] = math.log(r["L_100_10000_allH2"] / LH) / math.log(SCHNAUS / LH)
            r["share_allH2_0.38_vs_obs_ref"] = math.log(r["L_100_10000_allH2"] / Le0) / math.log(SCHNAUS / Le0)
            if v == "dinov2_b":
                # the reference against itself: share 0 by construction (readings section 5 table);
                # the raw float (recomputed L over the logged constant) is kept as a diagnostic
                r["share_0.38_raw_vs_constant"] = r["share_0.38"]
                for kk in ["share_0.38", "share_0.37", "share_0.39"]:
                    r[kk] = 0.0
            r["gap_label"] = label(r["share_0.38"])
            r["gap_label_binding"] = (enc == "mpnet" and v != "dinov2_b")
            r["gap_label_0.37"] = label(r["share_0.37"])
            r["gap_label_0.39"] = label(r["share_0.39"])
            r["gap_label_allH2"] = label(r["share_allH2_0.38"])
            r["ln_L100_10000_minus_ref"] = math.log(r["L_100_10000"]) - math.log(ref["L_100_10000"])
            r["ln_L10_1024_minus_ref"] = math.log(r["L_10_1024"]) - math.log(ref["L_10_1024"])
    out = {"pairs": P}
    out["reference_cell_vs_readings"] = {enc: {"mine": P[f"dinov2_b__{enc}"]["L_100_10000"], "readings": L_E0[enc],
                                               "abs_diff": abs(P[f"dinov2_b__{enc}"]["L_100_10000"] - L_E0[enc])}
                                         for enc in C.ENCODERS}
    out["binding_gap_reading_mpnet"] = {
        v: {"L_100_10000": P[f"{v}__mpnet"]["L_100_10000"], "share_0.38": P[f"{v}__mpnet"]["share_0.38"],
            "share_0.37": P[f"{v}__mpnet"]["share_0.37"], "share_0.39": P[f"{v}__mpnet"]["share_0.39"],
            "share_allH2_0.38": P[f"{v}__mpnet"]["share_allH2_0.38"], "gap_label": P[f"{v}__mpnet"]["gap_label"]}
        for v in X3V}
    con = {}
    for enc in C.ENCODERS:
        g = lambda v: math.log(P[f"{v}__{enc}"]["L_100_10000"])  # noqa: E731
        con[enc] = {"depth only: x3_cls_r10 vs ref (ln L(100,10000))": g("x3_cls_r10") - g("dinov2_b"),
                    "token set only: x3_tok_r05 vs ref": g("x3_tok_r05") - g("dinov2_b"),
                    "depth and token set: x3_tok_r10 vs ref": g("x3_tok_r10") - g("dinov2_b"),
                    "added (norm + all tokens + 224 px): x3add_schnaus224 vs ref": g("x3add_schnaus224") - g("dinov2_b"),
                    "added vs x3_tok_r10 (norm + CLS in the mean + 224 px)": g("x3add_schnaus224") - g("x3_tok_r10")}
    out["contrasts_ln_L_100_10000"] = con
    p1 = {}
    sums12 = [P[f"{v}__{e}"]["sum_k10_100"] for v in X3V for e in C.ENCODERS]
    ems12 = [P[f"{v}__{e}"]["e_m_k10_100"] for v in X3V for e in C.ENCODERS]
    med = float(np.median(sums12))
    p1["all_12"] = {"median_sum": med, "n_abs_le_0.05": int(sum(abs(s) <= 0.05 for s in sums12)),
                    "n_informative": int(sum(abs(e) >= 0.10 for e in ems12)), "n": len(sums12),
                    "max_abs_sum": float(max(abs(s) for s in sums12)),
                    "plan_thresholds_descriptive": {
                        "abs_median_le_0.03": abs(med) <= 0.03,
                        "frac_abs_le_0.05_ge_0.8": sum(abs(s) <= 0.05 for s in sums12) / len(sums12) >= 0.8}}
    for v in X3V:
        s = [P[f"{v}__{e}"]["sum_k10_100"] for e in C.ENCODERS]
        p1[v] = {"median_sum": float(np.median(s)), "n_abs_le_0.05": int(sum(abs(x) <= 0.05 for x in s)), "n": 3,
                 "sums": {e: P[f"{v}__{e}"]["sum_k10_100"] for e in C.ENCODERS}}
    out["p1_type_descriptive"] = p1
    curves, shape = {}, {}
    for pair, (sb, sd, ep, nb, ds, sbH, epH, nbH, dsH) in raw.items():
        curves[pair], shape[pair] = {}, {}
        for k in C.KS:
            curves[pair][str(k)] = [{"m": m, "s": sb[(m, k)], "draw_sd": sd[(m, k)], "chance": k / (m - 1),
                                     "s_allH2": sbH[(m, k)], "n_anchors_mean": nb[(m, k)]} for m in C.M_GRID]
            s = np.array([sb[(m, k)] for m in C.M_GRID])
            shape[pair][str(k)] = {"argmax_m": C.M_GRID[int(np.argmax(s))], "max_s": float(s.max()),
                                   "non_increasing_in_m": bool(np.all(np.diff(s) <= 0))}
    out["curves"] = curves
    out["curve_shape"] = shape
    # floors: does any cell fall below its floor (which would make the floor matter)?
    out["n_cells_below_floor"] = int(sum(sb[key] < ep[key] for (sb, sd, ep, *_rest) in raw.values() for key in sb))
    return out


# ----------------------------------------------------------------------------
# comparison with X3_results.json
# ----------------------------------------------------------------------------

def compare(mine: dict, ref: dict) -> dict:
    res = {"n_numeric": 0, "max_abs_diff": 0.0, "argmax": None, "n_over_tol": 0, "over_tol": [],
           "n_exact_fields": 0, "exact_mismatch": [], "missing_in_mine": []}

    def num(path, a, b):
        res["n_numeric"] += 1
        dif = abs(float(a) - float(b))
        if dif > res["max_abs_diff"]:
            res["max_abs_diff"], res["argmax"] = dif, path
        if not dif <= TOL:
            res["n_over_tol"] += 1
            res["over_tol"].append({"path": path, "mine": a, "run": b, "diff": dif})

    def exact(path, a, b):
        res["n_exact_fields"] += 1
        if a != b:
            res["exact_mismatch"].append({"path": path, "mine": a, "run": b})

    def walk(path, a, b):
        if isinstance(b, dict):
            for kk, vv in b.items():
                if a is None or kk not in a:
                    res["missing_in_mine"].append(f"{path}/{kk}")
                    continue
                walk(f"{path}/{kk}", a[kk], vv)
        elif isinstance(b, list):
            if len(a) != len(b):
                exact(path + "#len", len(a), len(b))
            for i, (x, y) in enumerate(zip(a, b)):
                walk(f"{path}[{i}]", x, y)
        elif isinstance(b, bool) or isinstance(b, str):
            exact(path, a, b)
        elif isinstance(b, (int, float)):
            if isinstance(a, bool):
                exact(path, a, b)
            elif isinstance(b, int) and path.endswith("argmax_m") or path.endswith("/m") or "n_abs" in path \
                    or path.endswith("/n") or "n_informative" in path:
                exact(path, int(a), int(b))
            else:
                num(path, a, b)
        elif b is None:
            exact(path, a, b)

    mp = {p["pair"]: p for p in ref["pairs"]}
    for pair, rp in mp.items():
        sub = {kk: vv for kk, vv in rp.items() if kk not in ("feature_text", "role")}
        walk(f"pairs/{pair}", mine["pairs"].get(pair), sub)
    for sec in ["binding_gap_reading_mpnet", "contrasts_ln_L_100_10000", "p1_type_descriptive",
                "curves", "curve_shape"]:
        walk(sec, mine[sec], ref[sec])
    rc = ref["reference_check_vs_readings"]
    for enc in C.ENCODERS:
        num(f"reference_check_vs_readings/{enc}/E0_levels_CA", mine["reference_cell_vs_readings"][enc]["mine"],
            rc[enc]["E0_levels_CA"])
    res["over_tol"] = res["over_tol"][:200]
    return res


def main():
    run_dir = sys.argv[1]
    t0, c0 = time.time(), time.process_time()
    cells = pd.read_parquet(os.path.join(run_dir, "obs", "x3i_cells.parquet"))
    mine = compute(cells)
    ox = pd.read_parquet(C.P("results/E0_sentence/x3/obs/obs_X3.parquet"))
    oca = pd.read_parquet(C.P("results/E0_sentence/obs/obs_CA.parquet"))
    run_cells = pd.concat([ox, oca[oca["pair"].isin([f"dinov2_b__{e}" for e in C.ENCODERS])]], ignore_index=True)
    from_run = compute(run_cells)
    xr_path = C.P("results/E0_sentence/x3/X3_results.json")
    ref = json.load(open(xr_path))
    cmp_mine = compare(mine, ref)
    cmp_run = compare(from_run, ref)
    # labels: all gap labels and slope classes identical, including the 0.37/0.39 band and allH2
    lab = []
    for pair, r in mine["pairs"].items():
        for kk in ["gap_label", "gap_label_0.37", "gap_label_0.39", "gap_label_allH2",
                   "b1_class", "b10_class", "b100_class",
                   "p1_stable_abs_sum_le_0.05", "p1_informative_abs_em_ge_0.10"]:
            lab.append((pair, kk, r[kk]))
    band_ok = all(r["gap_label"] == r["gap_label_0.37"] == r["gap_label_0.39"] == r["gap_label_allH2"]
                  for p, r in mine["pairs"].items() if not p.startswith("dinov2_b"))
    agree = (cmp_mine["n_over_tol"] == 0 and not cmp_mine["exact_mismatch"] and not cmp_mine["missing_in_mine"]
             and cmp_run["n_over_tol"] == 0 and not cmp_run["exact_mismatch"])
    out = {"role": "x3-indep", "tolerance": TOL,
           "compare_own_cells_vs_X3_results": cmp_mine,
           "compare_run_cells_vs_X3_results": cmp_run,
           "labels_unchanged_over_band_and_allH2": band_ok,
           "stats_agree": agree,
           "X3_results_sha256": C.sha256_file(xr_path),
           "own_results": mine,
           "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
    os.makedirs(os.path.join(run_dir, "stats"), exist_ok=True)
    C.write_json_new(os.path.join(run_dir, "stats", "x3i_stats.json"), out)
    print(json.dumps({k: (v if k != "over_tol" else v[:5]) for k, v in cmp_mine.items()}, default=C._json_default)[:2500])
    print(json.dumps({k: (v if k != "over_tol" else v[:5]) for k, v in cmp_run.items()}, default=C._json_default)[:1500])
    print("band_ok", band_ok, "agree", agree, "floors", mine["n_cells_below_floor"])


if __name__ == "__main__":
    main()
