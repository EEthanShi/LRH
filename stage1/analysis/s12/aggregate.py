"""S1.2 aggregation: panel tables (parquet), PE2 / PE3 decision rules exactly
as preregistered (operationalized in ANALYSIS_PLAN_S12.md), exact within-
family permutation tests, Holm, two-way cluster bootstrap CIs, diagnostics.

Outputs (results/S1_2/):
  panel_<run>.parquet      one row per (LM, ViT, r, pre) with every scalar
  s12_summary.json         every number used in the report
  (the markdown report is written by report.py from s12_summary.json)
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/hrh_v2_stage1_20261008"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import common as C  # noqa: E402
from hrhv2 import util  # noqa: E402

LM_IDX = {s: i for i, (s, _) in enumerate(C.LMS)}
VIT_IDX = {s: j for j, (s, _) in enumerate(C.VITS)}
LM_FAM = {s: f for s, f in C.LMS}
VIT_FAM = {s: f for s, f in C.VITS}
NB = 2000
CALIB_FIELDS = ("s_obs", "s_cal", "c_alpha", "p_value", "null_mean", "null_sd", "uncensored", "null_q95")


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_run(run: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(os.path.join(C.RES, run, "pairs", "*.json")))
    if not files:
        return None
    rows = []
    for f in files:
        rec = json.load(open(f))
        for pre, d in rec["per_pre"].items():
            row = {"lm": rec["lm"], "vit": rec["vit"], "r": float(rec["r"]), "pre": pre, "run": run,
                   "K": rec["K"], "perm_key": rec["perm_key"]}
            for k, v in d["ladder"].items():
                if not isinstance(v, (dict, list)):
                    row[f"L.{k}"] = v
            for k, v in d["mst_readings"].items():
                row[f"M.{k}"] = v
            for sec in ("eq6", "law_A_ref", "law_B_ref", "cka_eq"):
                for k, v in d.get(sec, {}).items():
                    row[f"{sec}.{k}"] = v
            for n, s in d["calib"].items():
                for k in CALIB_FIELDS:
                    if k in s:
                        row[f"cal.{n}.{k}"] = s[k]
            for side in ("A", "B"):
                mt = d[f"meta_{side}"]
                row[f"layer_{side}"] = mt["layer"]
                row[f"n_stored_{side}"] = mt["n_stored"]
                row[f"d_{side}"] = mt["d"]
                row[f"clip_c_{side}"] = mt["clip_c"]
                row["m"] = mt["m"]
            row["chk.fast_vs_ladder"] = d["fastpath_obs_vs_ladder_maxabs"]
            sp = d.get("calib_spotcheck_maxabs") or {}
            row["chk.spot_maxabs"] = max(sp.values()) if sp else np.nan
            row["chk.exact_relabel"] = d["calib_exact_relabel"]
            row["law_error"] = d.get("law_error")
            for k, v in d["timing_s"].items():
                row[f"t.{k}"] = v
            rows.append(row)
    df = pd.DataFrame(rows)
    df["lm_family"] = df["lm"].map(LM_FAM)
    df["vit_family"] = df["vit"].map(VIT_FAM)
    df["lm_params"] = df["lm"].map(C.n_params)
    df["vit_params"] = df["vit"].map(C.n_params)
    return df


def grid(df: pd.DataFrame, col: str, r: float, pre: str) -> np.ndarray:
    """(8 LMs, 17 ViTs) array in registry order; NaN where missing."""
    G = np.full((len(C.LMS), len(C.VITS)), np.nan)
    sub = df[(df.r == r) & (df.pre == pre)]
    for lm, vit, v in zip(sub.lm, sub.vit, pd.to_numeric(sub[col], errors="coerce")):
        G[LM_IDX[lm], VIT_IDX[vit]] = v
    return G


# ----------------------------------------------------------------------------
# exact within-family permutation trends
# ----------------------------------------------------------------------------

def _rank_cols(Y):
    """Average ranks down each column, centred; NaN-free input assumed."""
    R = np.apply_along_axis(rankdata, 0, Y)
    return R - R.mean(axis=0, keepdims=True)


def _rho_matrix(X_perms, Y):
    """Spearman rho for every permutation (rows of X_perms = size ranks) and
    every column of Y (members x cells). NaN for constant columns."""
    Ry = _rank_cols(Y)
    Xc = X_perms - X_perms.mean(axis=1, keepdims=True)
    num = Xc @ Ry                                   # (P, cells)
    den = np.sqrt((Xc ** 2).sum(axis=1))[:, None] * np.sqrt((Ry ** 2).sum(axis=0))[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(den > 0, num / den, np.nan)
    return rho


def size_ranks(stems):
    return rankdata([C.n_params(s) for s in stems])


def lm_trend(G: np.ndarray) -> dict:
    """Tr = mean over ViTs of Spearman(BLOOMZ size, score); exact p over all
    120 relabellings of the BLOOMZ sizes (same relabelling for every ViT)."""
    bl = [i for i, (s, f) in enumerate(C.LMS) if f == "BLOOMZ"]
    Y = G[bl, :]
    ok = np.all(np.isfinite(Y), axis=0)
    Y = Y[:, ok]
    x = size_ranks([C.LMS[i][0] for i in bl])
    P = np.array([np.array(p) for p in itertools.permutations(x)])
    rho = _rho_matrix(P, Y)                          # (120, n_vit)
    obs_idx = int(np.flatnonzero(np.all(P == x, axis=1))[0])
    T = np.nanmean(rho, axis=1)
    t_obs = float(T[obs_idx])
    rho_v = rho[obs_idx]
    eps = 1e-12
    out = {"Tr": t_obs, "rho_per_vit": [None if not np.isfinite(v) else float(v) for v in rho_v],
           "n_vit_used": int(np.sum(np.isfinite(rho_v))), "n_vit_constant": int(np.sum(~np.isfinite(rho_v)) + np.sum(~ok)),
           "p_one": float(np.mean(T >= t_obs - eps)), "p_two": float(np.mean(np.abs(T) >= abs(t_obs) - eps)),
           "n_perm": int(P.shape[0]), "null_T": T, "rho_all": rho, "obs_idx": obs_idx}
    # OpenLLaMA sign check (secondary): fraction of ViTs with score(7B) > score(3B)
    ol = [i for i, (s, f) in enumerate(C.LMS) if f == "OpenLLaMA"]
    ol = sorted(ol, key=lambda i: C.n_params(C.LMS[i][0]))
    d = G[ol[1]] - G[ol[0]]
    out["openllama_frac_larger_higher"] = float(np.mean(d[np.isfinite(d)] > 0)) if np.isfinite(d).any() else None
    return out


def lm_trend_diff(G1: np.ndarray, G2: np.ndarray) -> dict:
    """Tr(G1) - Tr(G2) under the same BLOOMZ relabelling; exact one-sided p."""
    a, b = lm_trend(G1), lm_trend(G2)
    D = a["null_T"] - b["null_T"]
    obs = float(D[a["obs_idx"]])
    eps = 1e-12
    return {"diff": obs, "p_one": float(np.mean(D >= obs - eps)), "p_two": float(np.mean(np.abs(D) >= abs(obs) - eps))}


def vision_trend(G: np.ndarray) -> dict:
    """Mean over (LM, vision family) cells of Spearman(ViT size, score); exact
    p over all within-family relabellings (product over the 5 families)."""
    fams = {}
    for j, (s, f) in enumerate(C.VITS):
        fams.setdefault(f, []).append(j)
    S_list, n_cells, obs_sum = [], 0, 0.0
    for f, idx in fams.items():
        x = size_ranks([C.VITS[j][0] for j in idx])
        P = np.array([np.array(p) for p in itertools.permutations(x)])
        Y = G[:, idx].T                               # members x LMs
        ok = np.all(np.isfinite(Y), axis=0)
        rho = _rho_matrix(P, Y[:, ok])                # (P, LMs)
        valid = np.isfinite(rho[0])
        n_cells += int(valid.sum())
        S = np.nansum(rho[:, valid], axis=1)          # (P,)
        oi = int(np.flatnonzero(np.all(P == x, axis=1))[0])
        obs_sum += float(S[oi])
        S_list.append(S)
    tot = S_list[0]
    for S in S_list[1:]:
        tot = np.add.outer(tot, S).ravel()
    T = tot / n_cells
    t_obs = obs_sum / n_cells
    eps = 1e-12
    return {"Tr": t_obs, "p_one": float(np.mean(T >= t_obs - eps)), "p_two": float(np.mean(np.abs(T) >= abs(t_obs) - eps)),
            "n_cells": n_cells, "n_perm": int(T.size)}


# ----------------------------------------------------------------------------
# bootstrap and Holm
# ----------------------------------------------------------------------------

def boot_two_way(stat, grids: list[np.ndarray], key: str, B: int = NB):
    """Two-way cluster bootstrap over LMs x ViTs (independent resampling of
    rows and columns of the 8 x 17 grid). Returns (estimate, lo, hi)."""
    rng = util.rng_from_key(f"S1.2|boot|{key}")
    est = float(stat(*grids))
    vals = np.empty(B)
    nl, nv = grids[0].shape
    for b in range(B):
        li = rng.integers(0, nl, nl)
        vj = rng.integers(0, nv, nv)
        vals[b] = stat(*[g[np.ix_(li, vj)] for g in grids])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return est, float(lo), float(hi)


def boot_vits(vec: np.ndarray, key: str, B: int = NB, vec2: np.ndarray | None = None):
    """Bootstrap over ViTs of mean(vec) (or mean(vec - vec2))."""
    rng = util.rng_from_key(f"S1.2|bootvit|{key}")
    v = np.asarray(vec, dtype=float) if vec2 is None else np.asarray(vec, float) - np.asarray(vec2, float)
    n = v.size
    vals = np.array([np.nanmean(v[rng.integers(0, n, n)]) for _ in range(B)])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(np.nanmean(v)), float(lo), float(hi)


def holm(p: dict) -> dict:
    keys = sorted(p, key=lambda k: p[k])
    n = len(keys)
    adj, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (n - i) * p[k]))
        adj[k] = run
    return adj


def frac(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x))


# ----------------------------------------------------------------------------
# PE2
# ----------------------------------------------------------------------------

READINGS = {
    # reading: (forest-share column, bridge-overlap calib name)
    "A (SPEC-literal, F_A = LM forest)": ("M.forest_share_shared_A", "s_T_bridge"),
    "B (F_B = ViT forest)": ("M.forest_share_shared_B", "s_T_bridge_B"),
    "union (F_A u F_B)": ("M.forest_share_shared_union", "s_T_bridge_mean"),
    "intersection (F_A n F_B)": ("M.forest_share_shared_inter", "s_T_bridge_mean"),
}
PRIMARY_READING = "A (SPEC-literal, F_A = LM forest)"


def pe2_raw(df, r, pre, reading):
    fcol, bname = READINGS[reading]
    FS = grid(df, fcol, r, pre)
    cond_a_pair = np.where(np.isfinite(FS), FS >= 0.70, False).astype(float)
    sT = grid(df, "L.s_T", r, pre)
    sN1 = grid(df, "L.s_N_k1", r, pre)
    cond_b_pair = (np.abs(sT - sN1) <= 0.03).astype(float)
    U = grid(df, f"cal.{bname}.uncensored", r, pre)
    scal = grid(df, f"cal.{bname}.s_cal", r, pre)
    pv = grid(df, f"cal.{bname}.p_value", r, pre)
    cond_e_pair = (scal > 0).astype(float)
    tr = lm_trend(U)
    vt = vision_trend(U)
    key = f"pe2|{df.run.iloc[0]}|{r}|{pre}|{reading}"
    out = {
        "frac_forest_ge_0.70": boot_two_way(frac, [cond_a_pair], key + "|a"),
        "median_forest_share": boot_two_way(lambda g: float(np.nanmedian(g)), [FS], key + "|fs"),
        "n_pairs_no_shared_edge": int(np.sum(~np.isfinite(FS))),
        "frac_abs_sT_minus_sN1_le_0.03": boot_two_way(frac, [cond_b_pair], key + "|b"),
        "median_abs_sT_minus_sN1": boot_two_way(lambda a, b: float(np.median(np.abs(a - b))), [sT, sN1], key + "|bm"),
        "Tr_bridge": {k: tr[k] for k in ("Tr", "p_one", "p_two", "n_vit_used", "n_vit_constant", "n_perm",
                                         "openllama_frac_larger_higher")},
        "Tr_bridge_ci_vits": boot_vits(np.array([np.nan if v is None else v for v in tr["rho_per_vit"]]), key + "|tr"),
        "Tr_bridge_vision_side": vt,
        "frac_bridge_exceeds_null": boot_two_way(frac, [cond_e_pair], key + "|e"),
        "frac_bridge_p_le_0.05": frac(pv <= 0.05),
        "median_bridge_overlap_obs": float(np.nanmedian(grid(df, f"cal.{bname}.s_obs", r, pre))),
        "median_bridge_null_mean": float(np.nanmedian(grid(df, f"cal.{bname}.null_mean", r, pre))),
        "mean_shared_bridges_A": float(np.nanmean(grid(df, "M.n_shared_bridges_A", r, pre))),
        "mean_shared_bridges_B": float(np.nanmean(grid(df, "M.n_shared_bridges_B", r, pre))),
        "mean_shared_bridges_pure": float(np.nanmean(grid(df, "M.n_shared_bridges_pure", r, pre))),
        "mean_bridges_A": float(np.nanmean(grid(df, "M.n_bridges_A", r, pre))),
        "mean_shared_edges": float(np.nanmean(grid(df, "M.n_shared_edges", r, pre))),
        "median_s_T": float(np.nanmedian(sT)), "median_s_N1": float(np.nanmedian(sN1)),
        "frac_s_T_exceeds_null": frac(grid(df, "cal.s_T.s_cal", r, pre) > 0),
    }
    return out


def pe2_verdict(raw, p_two_holm, p_one_holm):
    a = raw["frac_forest_ge_0.70"][0] >= 0.90
    b = raw["frac_abs_sT_minus_sN1_le_0.03"][0] >= 0.80
    tr = raw["Tr_bridge"]["Tr"]
    c = (p_two_holm >= 0.05) or (abs(tr) <= 0.3)
    d = (tr >= 0.5) and (p_one_holm < 0.05)
    e = raw["frac_bridge_exceeds_null"][0] >= 0.50
    if a and b and c:
        v = "SOG-global ≈ 1-NN"
    elif d and e:
        v = "Genuine global convergence"
    else:
        v = "mixed"
    return {"a_forest": a, "b_sT_vs_sN1": b, "c_trend_null": c, "d_trend_genuine": d, "e_exceeds_null": e,
            "p_two_holm": p_two_holm, "p_one_holm": p_one_holm, "verdict": v}


# ----------------------------------------------------------------------------
# PE3
# ----------------------------------------------------------------------------

def pe3_raw(df, r, pre, cka="biased"):
    Graw = grid(df, f"cal.cka_{cka}.uncensored", r, pre)
    Geq = grid(df, f"cal.cka_eq_{cka}.uncensored", r, pre)
    Gk = grid(df, "cal.s_N_k10.uncensored", r, pre)
    tr_raw, tr_eq, tr_k = lm_trend(Graw), lm_trend(Geq), lm_trend(Gk)
    diff = lm_trend_diff(Geq, Graw)
    key = f"pe3|{df.run.iloc[0]}|{r}|{pre}|{cka}"
    rv = lambda t: np.array([np.nan if v is None else v for v in t["rho_per_vit"]])  # noqa: E731
    out = {
        "Tr_raw": {k: tr_raw[k] for k in ("Tr", "p_one", "p_two", "n_vit_used", "openllama_frac_larger_higher")},
        "Tr_eq": {k: tr_eq[k] for k in ("Tr", "p_one", "p_two", "n_vit_used", "openllama_frac_larger_higher")},
        "Tr_mKNN": {k: tr_k[k] for k in ("Tr", "p_one", "p_two", "n_vit_used", "openllama_frac_larger_higher")},
        "Tr_raw_ci": boot_vits(rv(tr_raw), key + "|raw"),
        "Tr_eq_ci": boot_vits(rv(tr_eq), key + "|eq"),
        "Tr_mKNN_ci": boot_vits(rv(tr_k), key + "|knn"),
        "Tr_eq_minus_raw": diff,
        "Tr_eq_minus_raw_ci": boot_vits(rv(tr_eq), key + "|diff", vec2=rv(tr_raw)),
        "vision_side": {"raw": vision_trend(Graw), "eq": vision_trend(Geq), "mKNN": vision_trend(Gk)},
        "trend_of_uncalibrated_s_obs": {"raw": lm_trend(grid(df, f"cal.cka_{cka}.s_obs", r, pre))["Tr"],
                                        "eq": lm_trend(grid(df, f"cal.cka_eq_{cka}.s_obs", r, pre))["Tr"],
                                        "mKNN": lm_trend(grid(df, "cal.s_N_k10.s_obs", r, pre))["Tr"]},
        "frac_pairs_s_cal_pos": {"raw": frac(grid(df, f"cal.cka_{cka}.s_cal", r, pre) > 0),
                                 "eq": frac(grid(df, f"cal.cka_eq_{cka}.s_cal", r, pre) > 0),
                                 "mKNN": frac(grid(df, "cal.s_N_k10.s_cal", r, pre) > 0)},
        "mean_uncensored": {"raw": boot_two_way(lambda g: float(np.mean(g)), [Graw], key + "|mraw"),
                            "eq": boot_two_way(lambda g: float(np.mean(g)), [Geq], key + "|meq"),
                            "mKNN": boot_two_way(lambda g: float(np.mean(g)), [Gk], key + "|mk")},
    }
    return out


def pe3_verdict(raw):
    tr_eq, tr_raw, tr_k = raw["Tr_eq"]["Tr"], raw["Tr_raw"]["Tr"], raw["Tr_mKNN"]["Tr"]
    d = tr_eq - tr_raw
    if d >= 0.3 and tr_eq >= 0.6 * tr_k:
        v = "Width artifact"
    elif d <= 0.1:
        v = "Genuine global loss"
    else:
        v = "inconclusive (between branches)"
    return {"Tr_eq_minus_Tr_raw": d, "Tr_eq_ge_0.6_Tr_mKNN": bool(tr_eq >= 0.6 * tr_k), "verdict": v}


# ----------------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------------

WARP = ("scale", "affine_extra", "iso_extra", "row_extra", "residual")


def warp_diag(df, r, pre):
    out = {}
    for suf in ("expl", "expl_cf"):
        d = {}
        for w in WARP:
            G = grid(df, f"L.warpshare_{suf}_{w}", r, pre)
            d[w] = boot_two_way(lambda g: float(np.median(g)), [G], f"warp|{r}|{pre}|{suf}|{w}")
        iso = grid(df, f"L.warpshare_{suf}_iso_extra", r, pre)
        aff = grid(df, f"L.quot_mean_affine_{suf}", r, pre)
        d["refuted_median_iso_minus_affine_lt_0.10"] = bool(d["iso_extra"][0] < 0.10)
        d["exploratory_iso_gain_share_of_post_affine_residual"] = boot_two_way(
            lambda a, b: float(np.median(a / (1 - b))), [iso, aff], f"warp|{r}|{pre}|{suf}|isoshare")
        d["frac_pairs_iso_extra_ge_0.10"] = frac(iso >= 0.10)
        out[suf] = d
    return out


def ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    return float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))


def trend_decomp(df, r, pre):
    """Within BLOOMZ, per ViT: slopes vs log(params) of log(1-s_N(10)),
    log p-hat and log delta-hat (P2 law ingredients), both references."""
    bl = [i for i, (s, f) in enumerate(C.LMS) if f == "BLOOMZ"]
    x = np.log([C.n_params(C.LMS[i][0]) for i in bl])
    y = np.log(1 - grid(df, "L.s_N_k10", r, pre))[bl]
    out = {}
    for ref in ("law_A_ref", "law_B_ref"):
        lp = np.log(grid(df, f"{ref}.p_hat_mg", r, pre))[bl]
        ld = np.log(grid(df, f"{ref}.delta_D_mean", r, pre))[bl]
        by = np.array([ols_slope(x, y[:, v]) for v in range(y.shape[1])])
        bp = np.array([ols_slope(x, lp[:, v]) for v in range(y.shape[1])])
        bd = np.array([ols_slope(x, ld[:, v]) for v in range(y.shape[1])])
        res = by - bp - bd
        bs = np.array([ols_slope(x, np.log(grid(df, "L.s_N_k10", r, pre))[bl][:, v]) for v in range(y.shape[1])])
        key = f"decomp|{r}|{pre}|{ref}"
        out[ref] = {
            "slope_log_1msN": boot_vits(by, key + "|y"), "slope_log_phat": boot_vits(bp, key + "|p"),
            "slope_log_delta": boot_vits(bd, key + "|d"), "closure_residual": boot_vits(res, key + "|res"),
            "median_slopes": {"log_1msN": float(np.median(by)), "log_phat": float(np.median(bp)),
                              "log_delta": float(np.median(bd)), "residual": float(np.median(res))},
            "exploratory_slope_log_sN": boot_vits(bs, key + "|ls"),
            "mean_slopes": {"log_1msN": float(np.mean(by)), "log_phat": float(np.mean(bp)),
                            "log_delta": float(np.mean(bd)), "residual": float(np.mean(res)), "log_sN": float(np.mean(bs))},
            "frac_vits_delta_slope_neg": float(np.mean(bd < 0)), "frac_vits_phat_slope_neg": float(np.mean(bp < 0)),
            "frac_vits_1msN_slope_neg": float(np.mean(by < 0)),
        }
        # across all 136 pairs: does the P2 law track the level of 1 - s_N? (exploratory)
        P2 = grid(df, f"{ref}.P2_one_minus_sN", r, pre).ravel()
        Y = (1 - grid(df, "L.s_N_k10", r, pre)).ravel()
        ok = np.isfinite(P2) & (P2 > 0)
        out[ref]["exploratory_corr_log1msN_logP2_all_pairs"] = float(np.corrcoef(np.log(Y[ok]), np.log(P2[ok]))[0, 1])
        out[ref]["median_P2_one_minus_sN"] = float(np.nanmedian(P2))
    out["median_one_minus_sN10"] = float(np.nanmedian(1 - grid(df, "L.s_N_k10", r, pre)))
    return out


def eq6_diag(df, r, pre):
    out = {}
    for k in (10, 1):
        for var in ("raw_A", "raw_sym", "scaled_sym", "iso_A", "iso_B", "any"):
            G = grid(df, f"eq6.eq6_cert_{var}_k{k}", r, pre)
            out[f"k{k}_{var}"] = {"mean": float(np.nanmean(G)), "max": float(np.nanmax(G)),
                                  "n_pairs_any_certified": int(np.sum(G > 0))}
        out[f"k{k}_median_gap_over_2e_raw_A"] = float(np.nanmedian(grid(df, f"eq6.eq6_median_gap_over_2e_raw_A_k{k}", r, pre)))
        out[f"k{k}_median_gap_over_2e_iso_A"] = float(np.nanmedian(grid(df, f"eq6.eq6_median_gap_over_2e_iso_A_k{k}", r, pre)))
    out["observed_median_1_minus_sN10"] = float(np.nanmedian(1 - grid(df, "L.s_N_k10", r, pre)))
    return out


DESC_COLS = [
    ("s_N(1)", "L.s_N_k1"), ("s_N(10)", "L.s_N_k10"), ("s_N(100)", "L.s_N_k100"), ("s_T", "L.s_T"),
    ("forest share A", "L.forest_share_A"), ("s_T_forest", "L.s_T_forest"), ("s_T_bridge", "L.s_T_bridge"),
    ("comp Rand k=3", "L.comp_rand_k3"), ("Δ_O", "L.Delta_O"), ("global τ_b", "L.globalorder_tau_b"),
    ("global Spearman", "L.globalorder_spearman"), ("Gs_rms", "L.Gs_rms"), ("CKA biased", "L.cka_biased"),
    ("CKA debiased", "L.cka_debiased"), ("CKA eq (128 PC) biased", "cka_eq.cka_biased"), ("RSA Pearson", "L.rsa_pearson"),
    ("SOG kNN τ=1", "L.sog_knn10_tau1"), ("SOG kNN τ=0.01", "L.sog_knn10_tau0p01"), ("SOG MST τ=1", "L.sog_mst_tau1"),
    ("SOG MST τ=0.01", "L.sog_mst_tau0p01"),
    ("expl scale", "L.quot_mean_scale_expl"), ("expl affine", "L.quot_mean_affine_expl"),
    ("expl iso", "L.quot_mean_iso_expl"), ("expl rowiso", "L.quot_mean_rowiso_expl"),
    ("expl iso (cf)", "L.quot_mean_iso_expl_cf"), ("expl rowiso (cf)", "L.quot_mean_rowiso_expl_cf"),
    ("mKNN(10) s_cal", "cal.s_N_k10.s_cal"), ("mKNN(10) uncensored", "cal.s_N_k10.uncensored"),
    ("CKA biased s_cal", "cal.cka_biased.s_cal"), ("CKA biased uncensored", "cal.cka_biased.uncensored"),
    ("s_T uncensored", "cal.s_T.uncensored"), ("s_T_bridge uncensored", "cal.s_T_bridge.uncensored"),
]


def describe(df, r, pre):
    out = {}
    for name, col in DESC_COLS:
        G = grid(df, col, r, pre)
        if not np.isfinite(G).any():
            continue
        q = np.nanpercentile(G, [25, 50, 75])
        out[name] = {"median": float(q[1]), "q25": float(q[0]), "q75": float(q[2]),
                     "mean": float(np.nanmean(G)), "per_lm_mean": {C.LMS[i][0]: float(np.nanmean(G[i])) for i in range(len(C.LMS))}}
    return out


def checks(df):
    out = {"n_rows": int(len(df)), "n_pairs": int(df.groupby(["lm", "vit"]).ngroups),
           "all_exact_relabel": bool(df["chk.exact_relabel"].all()),
           "max_fast_vs_ladder": float(df["chk.fast_vs_ladder"].max()),
           "max_spotcheck": float(np.nanmax(df["chk.spot_maxabs"])) if df["chk.spot_maxabs"].notna().any() else None,
           "law_errors": int(df["law_error"].notna().sum()) if "law_error" in df else 0}
    for k in (1, 10, 100):
        c = f"L.neartie_A_k{k}"
        if c in df:
            out[f"max_neartie_A_k{k}"] = float(df[c].max())
            out[f"max_neartie_B_k{k}"] = float(df[f"L.neartie_B_k{k}"].max())
    if "L.forest_in_mst_A" in df:
        out["forest_in_mst_all"] = bool(df["L.forest_in_mst_A"].all() and df["L.forest_in_mst_B"].all())
    out["layers"] = {s: {str(r): int(df[(df.lm == s) & (df.r == r)].layer_A.iloc[0]) for r in sorted(df.r.unique())}
                     for s, _ in C.LMS if (df.lm == s).any()}
    out["layers"].update({s: {str(r): int(df[(df.vit == s) & (df.r == r)].layer_B.iloc[0]) for r in sorted(df.r.unique())}
                          for s, _ in C.VITS if (df.vit == s).any()})
    return out


def main():
    summary = {"plan": "analysis/s12/ANALYSIS_PLAN_S12.md"}
    runs = {}
    for run in ("main", "pool2047", "sidecar_rule"):
        df = load_run(run)
        if df is None:
            continue
        runs[run] = df
        df.to_parquet(os.path.join(C.RES, f"panel_{run}.parquet"), index=False)
        summary[f"checks_{run}"] = checks(df)
    main_df = runs["main"]
    depths = sorted(main_df.r.unique())

    # descriptive
    summary["describe"] = {f"r={r}|{pre}": describe(main_df, r, pre) for r in depths for pre in C.PRES}

    # PE2 raw (m=1024 and the gate fallback)
    pe2 = {}
    for r in depths:
        for pre in C.PRES:
            for reading in READINGS:
                pe2[f"m1024|r={r}|{pre}|{reading}"] = pe2_raw(main_df, r, pre, reading)
            if "pool2047" in runs and pre in runs["pool2047"].pre.unique() and r in runs["pool2047"].r.unique():
                for reading in READINGS:
                    pe2[f"m2047|r={r}|{pre}|{reading}"] = pe2_raw(runs["pool2047"], r, pre, reading)
    gate = {}
    for r in depths:
        g = pe2[f"m1024|r={r}|{C.PRIMARY_PRE}|{PRIMARY_READING}"]["mean_shared_bridges_A"]
        fires = g < 10
        designated = "m2047" if fires and f"m2047|r={r}|{C.PRIMARY_PRE}|{PRIMARY_READING}" in pe2 else "m1024"
        gate[str(r)] = {"mean_shared_bridges_A_m1024": g, "fires": bool(fires), "designated": designated,
                        "fallback_available": f"m2047|r={r}|{C.PRIMARY_PRE}|{PRIMARY_READING}" in pe2}
        if f"m2047|r={r}|{C.PRIMARY_PRE}|{PRIMARY_READING}" in pe2:
            gate[str(r)]["mean_shared_bridges_A_m2047"] = pe2[f"m2047|r={r}|{C.PRIMARY_PRE}|{PRIMARY_READING}"]["mean_shared_bridges_A"]
    summary["pe2_power_gate"] = gate

    # PE3 raw
    pe3 = {f"r={r}|{pre}|{cka}": pe3_raw(main_df, r, pre, cka) for r in depths for pre in C.PRES
           for cka in ("biased", "debiased")}

    # Holm families (primary pre; PE2 on the designated run, primary reading; PE3 primary CKA)
    holm_out = {}
    for side in ("p_one", "p_two"):
        fam = {}
        for r in depths:
            des = gate[str(r)]["designated"]
            fam[f"PE2|r={r}|{des}"] = pe2[f"{des}|r={r}|{C.PRIMARY_PRE}|{PRIMARY_READING}"]["Tr_bridge"][side]
            fam[f"PE3|r={r}"] = pe3[f"r={r}|{C.PRIMARY_PRE}|biased"]["Tr_eq_minus_raw"][side]
        holm_out[side] = {"raw": fam, "holm": holm(fam)}
    summary["holm"] = holm_out

    # verdicts: primary (designated run, primary reading, primary pre) and sensitivities
    verdicts = {"PE2": {}, "PE3": {}}
    for r in depths:
        for key, raw in pe2.items():
            mtag, rr, pre, reading = key.split("|", 3)
            if rr != f"r={r}":
                continue
            primary_cell = (pre == C.PRIMARY_PRE and reading == PRIMARY_READING and mtag == gate[str(r)]["designated"])
            if primary_cell:
                ptwo = holm_out["p_two"]["holm"][f"PE2|r={r}|{mtag}"]
                pone = holm_out["p_one"]["holm"][f"PE2|r={r}|{mtag}"]
                adj = "Holm"
            else:  # sensitivity cells: unadjusted exact p (stated)
                ptwo, pone, adj = raw["Tr_bridge"]["p_two"], raw["Tr_bridge"]["p_one"], "unadjusted"
            v = pe2_verdict(raw, ptwo, pone)
            v["p_adjustment"] = adj
            v["primary"] = primary_cell
            verdicts["PE2"][key] = v
        for pre in C.PRES:
            for cka in ("biased", "debiased"):
                v = pe3_verdict(pe3[f"r={r}|{pre}|{cka}"])
                v["primary"] = (pre == C.PRIMARY_PRE and cka == "biased")
                verdicts["PE3"][f"r={r}|{pre}|{cka}"] = v
    prim2 = {r: verdicts["PE2"][f"{gate[str(r)]['designated']}|r={r}|{C.PRIMARY_PRE}|{PRIMARY_READING}"]["verdict"] for r in depths}
    prim3 = {r: verdicts["PE3"][f"r={r}|{C.PRIMARY_PRE}|biased"]["verdict"] for r in depths}
    verdicts["PE2_overall"] = prim2[depths[0]] if len(set(prim2.values())) == 1 else "depth-dependent: " + "; ".join(f"r={r}: {v}" for r, v in prim2.items())
    verdicts["PE3_overall"] = prim3[depths[0]] if len(set(prim3.values())) == 1 else "depth-dependent: " + "; ".join(f"r={r}: {v}" for r, v in prim3.items())
    # robustness: the same Holm family and primary-cell rule if the m=1024 run were designated
    alt = {}
    for side in ("p_one", "p_two"):
        fam = {}
        for r in depths:
            fam[f"PE2|r={r}|m1024"] = pe2[f"m1024|r={r}|{C.PRIMARY_PRE}|{PRIMARY_READING}"]["Tr_bridge"][side]
            fam[f"PE3|r={r}"] = pe3[f"r={r}|{C.PRIMARY_PRE}|biased"]["Tr_eq_minus_raw"][side]
        alt[side] = {"raw": fam, "holm": holm(fam)}
    summary["holm_alt_m1024"] = alt
    verdicts["PE2_alt_m1024_holm"] = {
        str(r): pe2_verdict(pe2[f"m1024|r={r}|{C.PRIMARY_PRE}|{PRIMARY_READING}"],
                            alt["p_two"]["holm"][f"PE2|r={r}|m1024"], alt["p_one"]["holm"][f"PE2|r={r}|m1024"])
        for r in depths}
    summary["verdicts"] = verdicts
    summary["pe2"] = pe2
    summary["pe3"] = pe3

    # diagnostics
    summary["warp"] = {f"r={r}|{pre}": warp_diag(main_df, r, pre) for r in depths for pre in C.PRES}
    summary["trend_decomposition"] = {f"r={r}|{pre}": trend_decomp(main_df, r, pre) for r in depths for pre in C.PRES}
    summary["eq6"] = {f"r={r}|{pre}": eq6_diag(main_df, r, pre) for r in depths for pre in C.PRES}

    # exploratory: sidecar layer rule at r=0.5
    if "sidecar_rule" in runs:
        sd = runs["sidecar_rule"]
        ex = {}
        for pre in C.PRES:
            ex[pre] = {"pe3_biased": pe3_raw(sd, 0.5, pre, "biased"),
                       "pe2_A": pe2_raw(sd, 0.5, pre, PRIMARY_READING)}
            ex[pre]["pe3_verdict"] = pe3_verdict(ex[pre]["pe3_biased"])
            ex[pre]["pe2_verdict_unadjusted"] = pe2_verdict(ex[pre]["pe2_A"], ex[pre]["pe2_A"]["Tr_bridge"]["p_two"],
                                                            ex[pre]["pe2_A"]["Tr_bridge"]["p_one"])
        summary["exploratory_sidecar_rule_r0.5"] = ex

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if k not in ("null_T", "rho_all", "obs_idx")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return C.jsonable(o)
    json.dump(clean(summary), open(os.path.join(C.RES, "s12_summary.json"), "w"), indent=1, allow_nan=False)
    print(json.dumps(clean({"gate": gate, "verdicts": {k: v for k, v in verdicts.items() if k.endswith("overall")},
                            "holm": holm_out}), indent=1))


if __name__ == "__main__":
    main()
