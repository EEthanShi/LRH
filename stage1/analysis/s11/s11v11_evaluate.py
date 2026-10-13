"""S1.1 v1.1 evaluation (descriptive mode; analysis/s11/ANALYSIS_PLAN_S11_v1.1.md V1-V7).

Gate G1 is NOT run as a confirmatory test (PREREG v1.1 D4.7, plan V1): the three
v1.0 criterion statistics are computed as a descriptive accuracy table with no
thresholds and no pass/fail (plan V5).

    venv/bin/python analysis/s11/s11v11_evaluate.py [--configs clip,l2,cliplayer,dedup]

Inputs: results/S1_1/v11/obs/obs_<config>.parquet (observed s_N), unit JSONs
results/S1_1/v11/units/<config>/{nulls,iii}/*.json. Outputs: results/S1_1/v11/
S1_1_v11_{cells,drawslopes,slopes,curves}_<config>.parquet, S1_1_v11_summary.json.

Definitions (plan V5, O5, O6, O10, D4.8)
  direction mean (O6) for N1, N1|mu, the unvalidated predictor, P2, N2;
  per-draw slope of ln max(s, eps_d) on ln m over all 7 m, eps_d(m,k) = 0.5/(n_anchors(m) k)
  (V2 anchor counts of that draw); (pair, k) slope = mean of the 5 draw slopes;
  class: rising >= +0.1 (conformal-like), flat |slope| < 0.1 (smooth), falling <= -0.1 (rough);
  draw-mean cells: mean over draws (NaN propagates); eps_bar = 0.5/(n_draws n_anchors k) (O5);
  accuracy: (a) median |ln(X/obs)| over draw-mean cells with m != 1024 and obs > 2k/(m-1)
            (X <= 0 -> +inf); (b) MSLE over the m != 1024 cells of non-flat units, floor eps_bar;
            (c) fraction of non-flat units whose mean slope sign equals the observed sign.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s11_common as C  # noqa: E402
import s11v11_common as V  # noqa: E402

FLAT = 0.1
SLOPE_ZERO_TOL = 1e-12
CLASS_ORDER = ("seed-seed", "cross-vision", "cross-modal")
SLOPE_CLASSES = ("rising", "flat", "falling")
TRICHOTOMY = {"rising": "conformal-like", "flat": "smooth (anisotropic)", "falling": "rough"}
U = V.UNVALIDATED
# predictor columns after direction averaging; (label, column, primary?)
PRED_MAIN = (("N0", "N0"), ("N1", "N1"), (f"{U} (unvalidated)", U))
PRED_EXPL = (("N1|mu", "N1|mu"), (f"{U}|mu", f"{U}|mu"), ("P2 (v1.0, expl.)", "P2"), ("N2 (v1.0, expl.)", "N2"),
             ("N0H1 (v1.0 N0)", "N0H1"))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_units(config, kind):
    rows = []
    files = sorted(glob.glob(os.path.join(V.UNITS, config, kind, "*.json")))
    for f in files:
        with open(f) as fh:
            u = json.load(fh)
        for c in u["cells"]:
            r = {"pair": u["pair"], "draw": int(u["draw"]), "dir": u["direction"], "m": int(c["m"]),
                 "k": int(c["k"])}
            for key, val in c.items():
                if key in ("m", "k"):
                    continue
                r[key] = np.nan if val is None else val
            rows.append(r)
    return pd.DataFrame(rows), len(files)


def to_float(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


NULL_COLS = ("obs", "N0", "N0H1", "N1", "N1|mu", "P2", "P2_raw", "P2_phat", "N2", "N1_defined")


def build_cells(config, checks):
    """Draw-level table (pair, draw, m, k): observed + direction-mean predictors.
    Nulls (N0, N0H1, N1, N1|mu, P2, N2) come from the 'nulls' unit when it exists,
    else from the 'iii' unit (which computes them identically; equality is checked
    where both exist)."""
    obs = pd.read_parquet(os.path.join(V.OBS, f"obs_{config}.parquet"))
    key = ["pair", "draw", "m", "k"]
    df = obs.copy()
    units = {}
    for kind in ("nulls", "iii"):
        u, nf = load_units(config, kind)
        checks[f"{config}/{kind}_unit_files"] = nf
        units[kind] = u.set_index(key + ["dir"]) if len(u) else None
    un, ui = units["nulls"], units["iii"]
    if un is not None and ui is not None:
        both = un.index.intersection(ui.index)
        for c in ("obs", "N0", "N0H1", "N1", "N1|mu", "P2", "N2"):
            if c in un and c in ui and len(both):
                a = to_float(un.loc[both, c])
                b = to_float(ui.loc[both, c])
                ok = a.notna() & b.notna()
                checks[f"{config}/{c}_nulls_vs_iii_maxdiff"] = float((a[ok] - b[ok]).abs().max()) if ok.any() else None
    if un is not None and ui is not None:
        nul = pd.concat([un[[c for c in NULL_COLS if c in un]],
                         ui.loc[ui.index.difference(un.index), [c for c in NULL_COLS if c in ui]]])
    else:
        src = un if un is not None else ui
        nul = src[[c for c in NULL_COLS if c in src]] if src is not None else None
    cols_out = []
    if nul is not None:
        nul = nul.reset_index()
        for d in ("AB", "BA"):
            nd = nul[nul["dir"] == d].drop(columns=["dir"]).set_index(key)
            nd.columns = [f"{c}__{d}" for c in nd.columns]
            df = df.merge(nd.reset_index(), on=key, how="left")
        for d in ("AB", "BA"):
            ok = df[f"obs__{d}"].notna()
            checks[f"{config}/unit_obs_equal_obsstep_{d}"] = {
                "n": int(ok.sum()), "max_abs_diff": float((df.loc[ok, f"obs__{d}"] - df.loc[ok, "obs"]).abs().max())
                if ok.any() else None}
        for c in ("N0", "N0H1"):
            a, b = to_float(df[f"{c}__AB"]), to_float(df[f"{c}__BA"])
            checks[f"{config}/{c}_AB_eq_BA_maxdiff"] = float((a - b).abs().max())
            df[c] = a
        for c in ("N1", "N1|mu", "P2", "P2_raw", "N2", "P2_phat"):
            df[f"{c}_AB"] = to_float(df[f"{c}__AB"])
            df[f"{c}_BA"] = to_float(df[f"{c}__BA"])
            df[c] = 0.5 * (df[f"{c}_AB"] + df[f"{c}_BA"])
        df["N1_defined_AB"] = df["N1_defined__AB"]
        df["N1_defined_BA"] = df["N1_defined__BA"]
    if ui is not None:
        ii = ui[[U, f"{U}|mu"]].reset_index()
        for d in ("AB", "BA"):
            nd = ii[ii["dir"] == d].drop(columns=["dir"]).set_index(key)
            nd.columns = [f"{c}_{d}" for c in nd.columns]
            df = df.merge(nd.reset_index(), on=key, how="left")
        for c in (U, f"{U}|mu"):
            df[f"{c}_AB"] = to_float(df[f"{c}_AB"])
            df[f"{c}_BA"] = to_float(df[f"{c}_BA"])
            df[c] = 0.5 * (df[f"{c}_AB"] + df[f"{c}_BA"])
    df = df.drop(columns=[c for c in df.columns if "__" in c])
    # N0 (plan V3) = observed s_N of the predicted anchors at m_f: taken from the obs step for
    # every pair; the units' own N0 is checked against it where available.
    fitobs = df[df["m"] == V.M_FIT].set_index(["pair", "draw", "k"])["obs"].rename("N0_obs")
    df = df.merge(fitobs.reset_index(), on=["pair", "draw", "k"], how="left")
    if "N0" in df:
        ok = df["N0"].notna()
        checks[f"{config}/N0_units_eq_obs_at_mf_maxdiff"] = float((df.loc[ok, "N0"] - df.loc[ok, "N0_obs"]).abs().max()) \
            if ok.any() else None
    df["N0"] = df["N0_obs"]
    df = df.drop(columns=["N0_obs"])
    df["eps_d"] = 0.5 / (df["n_anchors"] * df["k"])
    # EXPLORATORY (not in plan V5): chance-corrected overlap (s - c)/(1 - c), c = k/(m-1)
    df["obs_cc"] = (df["obs"] - df["chance"]) / (1.0 - df["chance"])
    return df


def predictor_cols(df):
    cols = [c for _, c in PRED_MAIN + PRED_EXPL if c in df]
    for c in list(cols):
        for d in ("AB", "BA"):
            if f"{c}_{d}" in df and c in (U, "N1"):
                cols.append(f"{c}_{d}")
    return cols


def ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if not np.all(np.isfinite(y)):
        return float("nan")
    xc = x - x.mean()
    return float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))


def draw_slopes(df, cols, obs_cols=("obs", "obs_allH2", "obs_clean", "obs_cc")):
    """D4.8 per-draw slopes over all 7 m (eps_d floor with the V2 counts; obs_allH2
    uses its own counts)."""
    rows = []
    for (pair, k, d), g in df.groupby(["pair", "k", "draw"]):
        g = g.sort_values("m")
        x = np.log(g["m"].to_numpy(float))
        r = {"pair": pair, "k": int(k), "draw": int(d), "cls": g["cls"].iloc[0], "primary": bool(g["primary"].iloc[0]),
             "n_m": int(len(g))}
        eps = g["eps_d"].to_numpy(float)
        for c in obs_cols:
            e = 0.5 / (g["n_allH2"].to_numpy(float) * k) if c == "obs_allH2" else eps
            r[f"slope_{c}"] = ols_slope(x, np.log(np.maximum(g[c].to_numpy(float), e)))
        for c in cols:
            y = g[c].to_numpy(float)
            r[f"slope_{c}"] = ols_slope(x, np.log(np.maximum(y, eps))) if np.all(np.isfinite(y)) else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def mean_slopes(ds):
    num = [c for c in ds.columns if c.startswith("slope_")]
    g = ds.groupby(["pair", "k", "cls", "primary"])
    S = g[num].agg(lambda s: s.mean(skipna=False)).reset_index()
    S["n_draws"] = g["draw"].nunique().to_numpy()
    S["slope_obs_draw_sd"] = g["slope_obs"].std().to_numpy()
    S["slope_class"] = np.where(S["slope_obs"] >= FLAT, "rising", np.where(S["slope_obs"] <= -FLAT, "falling", "flat"))
    S["slope_class_allH2"] = np.where(S["slope_obs_allH2"] >= FLAT, "rising",
                                      np.where(S["slope_obs_allH2"] <= -FLAT, "falling", "flat"))
    S["slope_class_cc"] = np.where(S["slope_obs_cc"] >= FLAT, "rising",
                                   np.where(S["slope_obs_cc"] <= -FLAT, "falling", "flat"))
    S["slope_class_clean"] = np.where(S["slope_obs_clean"] >= FLAT, "rising",
                                      np.where(S["slope_obs_clean"] <= -FLAT, "falling", "flat"))
    return S


def drawmean_cells(df, cols):
    key = ["pair", "cls", "primary", "m", "k"]
    agg = {"obs": ("obs", "mean"), "obs_draw_sd": ("obs", "std"), "obs_allH2": ("obs_allH2", "mean"),
           "obs_allH2_draw_sd": ("obs_allH2", "std"), "obs_clean": ("obs_clean", "mean"),
           "n_draws": ("draw", "nunique"), "n_anchors": ("n_anchors", "mean"), "n_allH2": ("n_allH2", "mean"),
           "chance": ("chance", "first"), "n_tieA": ("n_tieA", "mean"), "n_tieB": ("n_tieB", "mean"),
           "neartie_A": ("neartie_A", "mean"), "neartie_B": ("neartie_B", "mean")}
    g = df.groupby(key, sort=True)
    c = g.agg(**agg).reset_index()
    for X in cols:
        v = g[X].agg(lambda s: s.mean(skipna=False)).reset_index(drop=True)
        c[X] = v.to_numpy()
    c["eps"] = 0.5 / (c["n_draws"] * c["n_anchors"] * c["k"])
    c["is_fit_m"] = c["m"] == V.M_FIT
    return c


def complete_pairs(c, X, pairs):
    """Pairs of the set for which X exists in every cell (all draws, both directions)."""
    g = c[c["pair"].isin(pairs)].groupby("pair")[X].apply(lambda s: bool(s.notna().all()))
    return sorted(g[g].index)


def accuracy(c, S, X, pairs):
    """Descriptive accuracy of predictor column X on a pair set (plan V5), over the
    pairs of the set with X available in every cell."""
    n_req = len(pairs)
    pairs = complete_pairs(c, X, pairs)
    cc = c[c["pair"].isin(pairs) & ~c["is_fit_m"]]
    el = cc[cc["obs"] > 2.0 * cc["k"] / (cc["m"] - 1)]
    x = el[X].to_numpy(float)
    o = el["obs"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(x > 0, np.abs(np.log(x / o)), np.inf)
    lr = np.where(np.isfinite(x), lr, np.nan)
    med = float(np.nanmedian(lr)) if np.any(~np.isnan(lr)) else float("nan")
    Ss = S[S["pair"].isin(pairs)]
    nf = Ss[np.abs(Ss["slope_obs"]) >= FLAT]
    keys = set(zip(nf["pair"], nf["k"]))
    sc = cc[[(p, k) in keys for p, k in zip(cc["pair"], cc["k"])]]
    if len(sc):
        xs = sc[X].to_numpy(float)
        msle = float(np.mean((np.log(np.maximum(xs, sc["eps"])) - np.log(np.maximum(sc["obs"], sc["eps"]))) ** 2)) \
            if np.all(np.isfinite(xs)) else float("nan")
    else:
        msle = float("nan")
    if len(nf) and f"slope_{X}" in nf:
        ps = nf[f"slope_{X}"].to_numpy(float)
        # |slope| < 1e-12 is a zero slope (a constant series such as N0 gives +-1e-17 from rounding)
        sg = np.where(np.abs(ps) < SLOPE_ZERO_TOL, 0.0, np.sign(ps))
        agree = (sg == np.sign(nf["slope_obs"].to_numpy(float))) & np.isfinite(ps)
        sign = float(agree.mean())
        n_sign = int(agree.sum())
    else:
        sign, n_sign = float("nan"), 0
    return {"n_pairs": len(pairs), "n_pairs_requested": n_req,
            "median_abs_log_ratio": med, "n_cells_eligible": int(len(el)),
            "frac_within_15pct": float(np.mean(lr <= math.log(1.15))) if len(lr) else float("nan"),
            "n_inf": int(np.isinf(lr).sum()), "msle_nonflat": msle, "n_nonflat_units": int(len(nf)),
            "n_nonflat_cells": int(len(sc)), "sign_frac": sign, "n_sign_correct": n_sign}


def slope_class_counts(S, pairs, col="slope_class"):
    out = {}
    Ss = S[S["pair"].isin(pairs)]
    for cl in CLASS_ORDER + ("all",):
        g = Ss if cl == "all" else Ss[Ss["cls"] == cl]
        if not len(g):
            continue
        out[cl] = {}
        for k in V.KS:
            gk = g[g["k"] == k]
            out[cl][int(k)] = {sc: int((gk[col] == sc).sum()) for sc in SLOPE_CLASSES}
            out[cl][int(k)]["n"] = int(len(gk))
            scol = {"slope_class": "slope_obs", "slope_class_allH2": "slope_obs_allH2",
                    "slope_class_clean": "slope_obs_clean", "slope_class_cc": "slope_obs_cc"}[col]
            out[cl][int(k)]["median_slope"] = float(gk[scol].median()) if len(gk) else float("nan")
    return out


def secondary(c, S, pairs):
    Ss = S[S["pair"].isin(pairs)]
    order = {}
    for k, g in Ss.groupby("k"):
        med = {cl: float(np.median(g.loc[g["cls"] == cl, "slope_obs"])) for cl in CLASS_ORDER if (g["cls"] == cl).any()}
        holds = bool(med["seed-seed"] >= med["cross-vision"] >= med["cross-modal"]) if len(med) == 3 else None
        order[int(k)] = {"median_slope_by_class": med, "ordering_holds": holds}
    el = []
    for pair, g in c[c["pair"].isin(pairs)].groupby("pair"):
        X = np.column_stack([np.ones(len(g)), np.log(g["k"].to_numpy(float)), np.log(g["m"].to_numpy(float))])
        y = np.log(np.maximum(g["obs"].to_numpy(float), g["eps"].to_numpy(float)))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        el.append({"pair": pair, "cls": g["cls"].iloc[0], "e_k": float(beta[1]), "e_m": float(beta[2]),
                   "sum": float(beta[1] + beta[2]), "rmse": float(np.sqrt(np.mean(resid ** 2))),
                   "e_m_V5_mean_over_k": float(Ss.loc[Ss["pair"] == pair, "slope_obs"].mean())})
    el = pd.DataFrame(el)
    summ = {cl: {"median_e_k": float(g["e_k"].median()), "median_e_m": float(g["e_m"].median()),
                 "median_sum": float(g["sum"].median()), "median_rmse": float(g["rmse"].median()), "n": int(len(g))}
            for cl, g in el.groupby("cls")}
    corr = float(np.corrcoef(el["e_k"], -el["e_m"])[0, 1]) if len(el) > 2 else float("nan")
    return {"slope_ordering": order, "elasticity_by_class": summ, "corr_ek_minus_em": corr,
            "elasticity_per_pair": el.to_dict("records")}


def curves(c, pairs):
    g = c[c["pair"].isin(pairs)].groupby(["cls", "k", "m"])
    out = g["obs"].agg(["median", "min", "max", "count"]).reset_index()
    out["median_chance"] = g["chance"].first().to_numpy()
    out["median_obs_allH2"] = g["obs_allH2"].median().to_numpy()
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def f3(x, nd=3):
    if x is None:
        return "-"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(x):
        return "n/a"
    if math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def evaluate_config(config, checks):
    df = build_cells(config, checks)
    cols = predictor_cols(df)
    ds = draw_slopes(df, cols)
    S = mean_slopes(ds)
    c = drawmean_cells(df, cols)
    prim = sorted(c.loc[c["primary"], "pair"].unique())
    allp = sorted(c["pair"].unique())
    R = {"config": config, "spec": V.CONFIGS[config], "n_pairs": len(allp), "n_primary": len(prim),
         "draws_per_cell": sorted(set(int(x) for x in c["n_draws"])),
         "units_available": {X: int(c[X].notna().sum()) if X in c else 0 for X in cols}}
    sets = {"primary": prim}
    if len(allp) > len(prim):
        sets["all (36 seed pairs)"] = allp
    R["slope_classes"] = {name: slope_class_counts(S, ps) for name, ps in sets.items()}
    R["slope_classes_allH2"] = {name: slope_class_counts(S, ps, "slope_class_allH2") for name, ps in sets.items()}
    R["slope_classes_clean"] = {name: slope_class_counts(S, ps, "slope_class_clean") for name, ps in sets.items()}
    R["slope_classes_chance_corrected_EXPLORATORY"] = {name: slope_class_counts(S, ps, "slope_class_cc")
                                                       for name, ps in sets.items()}
    R["only_smooth_class_observed"] = {name: bool((np.abs(S.loc[S["pair"].isin(ps), "slope_obs"]) < FLAT).all())
                                       for name, ps in sets.items()}
    acc = {}
    for name, ps in sets.items():
        acc[name] = {}
        for label, X in PRED_MAIN + PRED_EXPL + ((f"{U} A->B only", f"{U}_AB"), (f"{U} B->A only", f"{U}_BA"),
                                                  ("N1 A->B only", "N1_AB"), ("N1 B->A only", "N1_BA")):
            if X not in c or not complete_pairs(c, X, ps):
                continue
            acc[name][label] = {"pooled": accuracy(c, S, X, ps)}
            for cl in CLASS_ORDER:
                pc = [p for p in ps if C.PAIRS[p]["cls"] == cl]
                if pc and complete_pairs(c, X, pc):
                    acc[name][label][cl] = accuracy(c, S, X, pc)
    R["accuracy"] = acc
    R["secondary"] = {name: secondary(c, S, ps) for name, ps in sets.items()}
    if "N1_defined_AB" in df:
        nd = df.groupby(["pair", "draw"])[["N1_defined_AB", "N1_defined_BA"]].first().reset_index()
        und = []
        for r in nd.itertuples(index=False):
            for d, val in (("AB", r[2]), ("BA", r[3])):
                if val is not None and not (isinstance(val, float) and np.isnan(val)) and not bool(val):
                    und.append(f"{r[0]}@d{r[1]}:{d}")
        R["n1_undefined_units"] = und
    R["ties"] = {"max_n_tieA": float(c["n_tieA"].max()), "max_n_tieB": float(c["n_tieB"].max()),
                 "cells_with_ties": int(((c["n_tieA"] > 0) | (c["n_tieB"] > 0)).sum()),
                 "max_neartie_frac": float(np.nanmax(c[["neartie_A", "neartie_B"]].to_numpy()))}
    out = os.path.join(V.OUT)
    df.to_parquet(os.path.join(out, f"S1_1_v11_drawcells_{config}.parquet"), index=False)
    c.to_parquet(os.path.join(out, f"S1_1_v11_cells_{config}.parquet"), index=False)
    ds.to_parquet(os.path.join(out, f"S1_1_v11_drawslopes_{config}.parquet"), index=False)
    S.to_parquet(os.path.join(out, f"S1_1_v11_slopes_{config}.parquet"), index=False)
    cv = curves(c, allp)
    cv.to_parquet(os.path.join(out, f"S1_1_v11_curves_{config}.parquet"), index=False)
    return R, c, S, cv


def md_config(R, c, S, cv, primary_cfg):
    L = []
    cfg = R["config"]
    L += [f"## Configuration `{cfg}` ({R['spec']['role']}): pre = {R['spec']['pre']}"
          + (f", clip = {R['spec']['clip_mode']}" if R['spec']['clip_mode'] else "")
          + (", surplus duplicates dropped" if R['spec']['dedup'] else ""), "",
          f"{R['n_pairs']} pairs ({R['n_primary']} primary); draws per cell {R['draws_per_cell']}.", ""]
    for name, sc in R["slope_classes"].items():
        L += [f"### Fixed-k slope classes, {name} pair set (D4.8 slopes; rising = conformal-like, flat = smooth, falling = rough)", "",
              "| class | k | rising | flat | falling | n | median slope |", "|---|---|---|---|---|---|---|"]
        for cl, byk in sc.items():
            for k, v in byk.items():
                L.append(f"| {cl} | {k} | {v['rising']} | {v['flat']} | {v['falling']} | {v['n']} | {f3(v['median_slope'])} |")
        L.append("")
        L.append(f"Only smooth class observed: {R['only_smooth_class_observed'][name]}.")
        L.append("")
    for title, key in (("all H2 items of each prefix (draft quantity obs_allH2)", "slope_classes_allH2"),
                       ("anchors without an exact k-boundary tie (O11)", "slope_classes_clean"),
                       ("EXPLORATORY, not in the plan: chance-corrected (s - c)/(1 - c)",
                        "slope_classes_chance_corrected_EXPLORATORY")):
        sc = R[key].get("primary")
        if not sc:
            continue
        L += [f"Slope classes, primary pairs, {title}: " + "; ".join(
            f"{cl} k={k}: {v['rising']}/{v['flat']}/{v['falling']} (median {f3(v['median_slope'])})"
            for cl, byk in sc.items() if cl != "all" for k, v in byk.items()), ""]
    for name, acc in R["accuracy"].items():
        L += [f"### Descriptive accuracy, {name} pair set (no thresholds, no pass/fail; plan V5)", "",
              "| predictor | class | pairs | median abs log(X/obs) (n cells) | MSLE non-flat (units/cells) | slope sign agreement |",
              "|---|---|---|---|---|---|"]
        for label, byc in acc.items():
            for cl in ("pooled",) + CLASS_ORDER:
                if cl not in byc:
                    continue
                a = byc[cl]
                L.append(f"| {label} | {cl} | {a['n_pairs']}/{a['n_pairs_requested']} | {f3(a['median_abs_log_ratio'])} ({a['n_cells_eligible']}) | "
                         f"{f3(a['msle_nonflat'])} ({a['n_nonflat_units']}/{a['n_nonflat_cells']}) | "
                         f"{a['n_sign_correct']}/{a['n_nonflat_units']} = {f3(a['sign_frac'])} |")
        L.append("")
    for name, sec in R["secondary"].items():
        L += [f"### Secondary (O10, descriptive), {name} pair set", ""]
        for k, v in sec["slope_ordering"].items():
            med = v["median_slope_by_class"]
            L.append(f"- k = {k}: median fixed-k slope seed-seed {f3(med.get('seed-seed'))}, cross-vision "
                     f"{f3(med.get('cross-vision'))}, cross-modal {f3(med.get('cross-modal'))}; "
                     f"seed >= vision >= modal: {v['ordering_holds']}")
        L += ["", "| class | median e_k | median e_m | median e_k + e_m | median fit rmse | n |", "|---|---|---|---|---|---|"]
        for cl, v in sec["elasticity_by_class"].items():
            L.append(f"| {cl} | {f3(v['median_e_k'])} | {f3(v['median_e_m'])} | {f3(v['median_sum'])} | {f3(v['median_rmse'])} | {v['n']} |")
        L += ["", f"corr(e_k, -e_m) across pairs = {f3(sec['corr_ek_minus_em'])}", ""]
    if cfg == primary_cfg:
        L += ["### Observed s_N(m, k) curves per class (median over pairs of the draw-mean; [min, max])", ""]
        ms = sorted(cv["m"].unique())
        for k in V.KS:
            L += [f"k = {k}:", "", "| class | " + " | ".join(f"m={m}" for m in ms) + " |",
                  "|---|" + "---|" * len(ms)]
            for cl in CLASS_ORDER:
                g = cv[(cv["cls"] == cl) & (cv["k"] == k)].set_index("m")
                if not len(g):
                    continue
                L.append(f"| {cl} | " + " | ".join(f"{f3(g.loc[m, 'median'])} [{f3(g.loc[m, 'min'], 2)}, {f3(g.loc[m, 'max'], 2)}]"
                                                    if m in g.index else "-" for m in ms) + " |")
            chance = [k / (m - 1) for m in ms]
            L.append("| chance k/(m-1) | " + " | ".join(f"{x:.4f}" for x in chance) + " |")
            L.append("")
        L += ["### Per pair (draw-mean obs on the predicted anchors; D4.8 slopes)", "",
              "| pair | class | s_N k=1 m=256 / 1024 / 16384 | s_N k=10 m=256 / 1024 / 16384 | s_N k=100 m=256 / 1024 / 16384 | slope k=1 / 10 / 100 | class k=1/10/100 (+ rising, 0 flat, - falling) |",
              "|---|---|---|---|---|---|---|"]
        for cl in CLASS_ORDER:
            for pair in sorted(c.loc[c["cls"] == cl, "pair"].unique()):
                g = c[c["pair"] == pair].set_index(["k", "m"])
                s = S[S["pair"] == pair].set_index("k")
                mx = int(c.loc[c["pair"] == pair, "m"].max())
                cells = []
                for k in V.KS:
                    cells.append(" / ".join(f3(g.loc[(k, m), "obs"]) if (k, m) in g.index else "-" for m in (256, 1024, mx)))
                sl = " / ".join(f3(s.loc[k, "slope_obs"]) for k in V.KS)
                cls_ = "/".join({"rising": "+", "flat": "0", "falling": "-"}[s.loc[k, "slope_class"]] for k in V.KS)
                L.append(f"| {pair}{'' if C.PAIRS[pair]['primary'] else ' (sec.)'} | {cl} | {cells[0]} | {cells[1]} | {cells[2]} | {sl} | {cls_} |")
        L.append("")
    return L


def config_deltas(configs):
    """Slope differences of each configuration vs the primary 'clip' on shared (pair, k) units."""
    p0 = os.path.join(V.OUT, "S1_1_v11_slopes_clip.parquet")
    if not os.path.exists(p0):
        return {}
    S0 = pd.read_parquet(p0).set_index(["pair", "k"])
    out = {}
    for cfg in configs:
        if cfg == "clip":
            continue
        S1 = pd.read_parquet(os.path.join(V.OUT, f"S1_1_v11_slopes_{cfg}.parquet")).set_index(["pair", "k"])
        j = S0[["slope_obs", "slope_class", "cls", "primary"]].join(S1[["slope_obs", "slope_class"]], rsuffix="_cfg",
                                                                     how="inner")
        j = j[j["primary"]]
        d = (j["slope_obs_cfg"] - j["slope_obs"])
        out[cfg] = {"n_units": int(len(j)), "median_abs_dslope": float(d.abs().median()),
                    "max_abs_dslope": float(d.abs().max()), "mean_dslope": float(d.mean()),
                    "n_class_changes": int((j["slope_class"] != j["slope_class_cfg"]).sum()),
                    "class_changes": [f"{p}@k={k}: {a}->{b}" for (p, k), a, b in
                                      zip(j.index, j["slope_class"], j["slope_class_cfg"]) if a != b],
                    "by_class": {cl: {"median_dslope": float(d[j["cls"] == cl].median()),
                                      "max_abs_dslope": float(d[j["cls"] == cl].abs().max())}
                                 for cl in CLASS_ORDER if (j["cls"] == cl).any()}}
    return out


def md_deltas(D):
    if not D:
        return []
    L = ["## Sensitivity of the observed fixed-k slopes to the configuration (primary pairs, vs `clip`)", "",
         "| configuration | (pair,k) units | median abs dslope | max abs dslope | mean dslope | slope-class changes |",
         "|---|---|---|---|---|---|"]
    for cfg, v in D.items():
        L.append(f"| {cfg} | {v['n_units']} | {f3(v['median_abs_dslope'])} | {f3(v['max_abs_dslope'])} | "
                 f"{f3(v['mean_dslope'])} | {v['n_class_changes']}" + (f" ({'; '.join(v['class_changes'][:8])})" if v['class_changes'] else "") + " |")
    L.append("")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="clip,l2,cliplayer,dedup")
    args = ap.parse_args(argv)
    auth = V.check_authorization()
    checks = {}
    allR = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": "descriptive (PREREG v1.1 D4.7)",
            "G1": "NOT RUN as a confirmatory test: no eligible estimator (PREREG v1.1 D4.7; plan V1)",
            "authorization_utc": auth["utc"], "frozen_hashes": V.check_hashes(), "configs": {}}
    md = ["# S1.1 on COCO under PREREG v1.1: descriptive results", "",
          "Gate G1 is NOT run as a confirmatory test: no eligible estimator (PREREG v1.1 D4.7; analysis plan V1). "
          f"The crossing law is reported as an unvalidated predictor ({U}).", ""]
    for config in args.configs.split(","):
        if not os.path.exists(os.path.join(V.OBS, f"obs_{config}.parquet")):
            continue
        R, c, S, cv = evaluate_config(config, checks)
        allR["configs"][config] = R
        md += md_config(R, c, S, cv, "clip")
    allR["config_deltas_vs_clip"] = config_deltas(list(allR["configs"]))
    md += md_deltas(allR["config_deltas_vs_clip"])
    allR["consistency_checks"] = checks
    with open(os.path.join(V.OUT, "S1_1_v11_summary.json"), "w") as f:
        json.dump(V.jsonable(allR), f, indent=1)
    md += ["## Consistency checks", "", "```", json.dumps(V.jsonable(checks), indent=1), "```"]
    with open(os.path.join(V.OUT, "S1_1_v11_tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", os.path.join(V.OUT, "S1_1_v11_summary.json"), os.path.join(V.OUT, "S1_1_v11_tables.md"))


if __name__ == "__main__":
    main()
