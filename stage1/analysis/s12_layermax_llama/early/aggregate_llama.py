"""LLAMA-AXIS (new file, implementation E, role llama-axis-run-early): aggregation.

Plan: analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md (sections 5-7, 10) and the frozen
Part B instance analysis/s12_layermax_llama/PARTB_PREDICTIONS.md (section 5-6 rules).
Replaces the copied aggregate.py (whose trend code is tied to the 8-LM BLOOMZ layout);
the trend arithmetic is the S1.2 one (analysis/s12/aggregate.py, read-only import:
_rho_matrix, boot_vits, vision_trend), with family LLaMA, and is self-checked
against A12.lm_trend on the frozen BLOOMZ rows.

Inputs (read-only except the E outputs):
  results/S1_2_layermax_llama/early/pairs/<pre>/*.{json,npz}        (E, 51 pairs x 2 pre)
  results/S1_2_layermax/pairs/<pre>/*.{json,npz}                    (frozen 8-LM run; 7 non-LLaMA
                                                                     LMs for the pooled panel, BLOOMZ
                                                                     half of Tr_BL, comparison)
  results/S1_2_layermax/layermax_summary.json                       (frozen 8-LM summary, comparison)
  results/S1_2_layermax_llama/partB/partB_features.json             (frozen Part B quantities)

Output: results/S1_2_layermax_llama/early/summary.json
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr, pearsonr  # noqa: E402

import lm_core as LC  # noqa: E402
from hrhv2 import util  # noqa: E402

_spec = importlib.util.spec_from_file_location("s12_aggregate", os.path.join(LC.PROJ, "analysis", "s12", "aggregate.py"))
A12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A12)

PRES = LC.PRES
METRICS = ("mknn", "cka_biased", "cka_debiased")
LLAMA = [s for s, _ in LC.LMS]
VITS = [s for s, _ in LC.VITS]
NV = len(VITS)
FROZEN_LMS = [s for s, _ in LC.C12.LMS]                    # 8-LM order of the frozen run (= ARH order)
POOLED = FROZEN_LMS + ["huggyllama__llama-13b", "huggyllama__llama-30b"]   # 10-LM panel, ARH order
FAM = dict(LC.C12.LMS)
FAM.update({s: "LLaMA" for s in LLAMA})
ARH_RANK10 = np.array([LC.ARH_LLM_ORDER.index(s.replace("__", "/")) + 1 for s in POOLED], dtype=float)
DENSE10 = np.arange(1, 11, dtype=float)
LOGP10 = np.log(np.array([LC.n_params(s) for s in POOLED], dtype=float))
SUB_STD = ["fixed_r0.5", "fixed_r1.0", "grid2", "grid3", "grid5", "grid9", "grid17", "LMall_ViTlast",
           "LMlast_ViTall", "full"]
SUB_NEW = ["noLMemb", "LMdeep", "LMemb_only"]
COLS = ("s_obs", "s_cal", "uncensored")
DECOMP = ("s_obs", "shift", "s_cal", "sel_entry_thr", "sel_entry_mean", "uncensored", "c_alpha", "inflation",
          "null_mean", "entry_c_alpha_at_argmax")
EPS = 1e-12
NB = 2000
MAT = 0.10                 # plan section 5 materiality for Tr_L changes
DELTA_B = 0.005            # Part B delta_b
BASE_MKNN = LC.KNN_K / (LC.M_ITEMS - 1)


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def _load(base_dir, pre, lm, vit):
    with open(os.path.join(base_dir, "pairs", pre, f"{lm}__{vit}.json")) as f:
        return json.load(f)


REC = {(pre, lm, vit): _load(LC.RES, pre, lm, vit) for pre in PRES for lm in LLAMA for vit in VITS}
FRZ = {(pre, lm, vit): _load(LC.FROZEN_RES, pre, lm, vit) for pre in PRES for lm in FROZEN_LMS for vit in VITS}
PB = json.load(open(os.path.join(LC.PARTB, "partB_features.json")))
FRZ_SUMMARY = json.load(open(os.path.join(LC.FROZEN_RES, "layermax_summary.json")))


def ctm_from_npz(base_dir, pre, lm, vit, met):
    """Protocol (iv) from stored float64 entry arrays (numpy version of run_layermax.calib_then_max)."""
    z = np.load(os.path.join(base_dir, "pairs", pre, f"{lm}__{vit}.npz"))
    obs, c, nm = z[f"{met}.entry_obs"], z[f"{met}.entry_c_alpha"], z[f"{met}.entry_null_mean"]
    den = 1.0 - c
    scal = np.where(den > 0, np.maximum((obs - c) / np.where(den > 0, den, 1.0), 0.0), 0.0)
    unc = obs - nm
    j = int(np.argmax(scal.reshape(-1)))
    return {"M": float(scal.reshape(-1)[j]), "argmax_A": j // obs.shape[1], "argmax_B": j % obs.shape[1],
            "M_unc": float(unc.max())}


FRZ_CTM = {(pre, lm, vit, met): ctm_from_npz(LC.FROZEN_RES, pre, lm, vit, met)
           for pre in PRES for lm in FROZEN_LMS for vit in VITS for met in METRICS}


def field(rec, met, subset, col, pre=None, lm=None, vit=None):
    if subset == "ctm":
        d = rec["calib_then_max"][met] if "calib_then_max" in rec else FRZ_CTM[(pre, lm, vit, met)]
        return float(d[col])
    s = rec["summaries"][met][subset]
    if col in s:
        return float(s[col])
    if col == "shift":
        return s["s_obs"] - s["c_alpha"]
    if col == "sel_entry_thr":
        return s["s_obs"] - s["entry_c_alpha_at_argmax"]
    if col == "sel_entry_mean":
        return s["s_obs"] - s["entry_null_mean_at_argmax"]
    if col == "inflation":
        return s["c_alpha"] - s["entry_c_alpha_at_argmax"]
    if col == "rel_A":
        return s["argmax_A"] / (rec["L_A"] - 1)
    if col == "rel_B":
        return s["argmax_B"] / (rec["L_B"] - 1)
    if col == "minus_Bbar":                      # plug-in check: T - Bbar_pred (biased CKA)
        return s["s_obs"] - PB["per_pre"][pre]["Bbar"][lm][vit]
    raise KeyError(col)


def gridL(pre, met, subset, col):
    G = np.full((3, NV), np.nan)
    for i, lm in enumerate(LLAMA):
        for j, vit in enumerate(VITS):
            G[i, j] = field(REC[(pre, lm, vit)], met, subset, col, pre, lm, vit)
    return G


def grid8(pre, met, subset, col):
    G = np.full((8, NV), np.nan)
    for i, lm in enumerate(FROZEN_LMS):
        for j, vit in enumerate(VITS):
            G[i, j] = field(FRZ[(pre, lm, vit)], met, subset, col, pre, lm, vit)
    return G


def grid10(pre, met, subset, col):
    """10-LM pooled panel (ARH order): 7 non-LLaMA LMs from the frozen run, the 3 LLaMA from E."""
    G = np.full((10, NV), np.nan)
    G8 = grid8(pre, met, subset, col)
    GL = gridL(pre, met, subset, col)
    G[:7] = G8[:7]
    G[7:] = GL
    return G


# ----------------------------------------------------------------------------
# LLaMA-axis trend (S1.2 lm_trend arithmetic, family LLaMA), differences, joint
# ----------------------------------------------------------------------------

_XL = rankdata([LC.n_params(s) for s in LLAMA])
_PL = np.array([np.array(p) for p in itertools.permutations(_XL)])
_OBS_L = int(np.flatnonzero(np.all(_PL == _XL, axis=1))[0])


def tr_core(G3):
    Y = G3
    ok = np.all(np.isfinite(Y), axis=0)
    Yo = Y[:, ok]
    with np.errstate(invalid="ignore"):
        rho = A12._rho_matrix(_PL, Yo)
        T = np.nanmean(rho, axis=1) if np.isfinite(rho).any() else np.full(_PL.shape[0], np.nan)
    rho_full = np.full((_PL.shape[0], NV), np.nan)
    rho_full[:, ok] = rho
    return T, rho_full, int(np.sum(~ok))


def tr_L(G3, key=None):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        T, rho, n_nan = tr_core(G3)
    t = float(T[_OBS_L])
    rv = rho[_OBS_L]
    out = {"Tr": t if math.isfinite(t) else None,
           "rho_per_vit": [None if not np.isfinite(v) else float(v) for v in rv],
           "n_vit_used": int(np.sum(np.isfinite(rv))), "n_vit_constant": int(np.sum(~np.isfinite(rv))),
           "p_one": float(np.mean(T >= t - EPS)) if math.isfinite(t) else None,
           "p_two": float(np.mean(np.abs(T) >= abs(t) - EPS)) if math.isfinite(t) else None, "n_relabel": 6}
    if key is not None and math.isfinite(t):
        est, lo, hi = A12.boot_vits(np.where(np.isfinite(rv), rv, np.nan), f"layermax-llama|{key}")
        out["ci"] = [lo, hi]
    out["_null_T"] = T
    return out


def tr_diff(G_score, G_raw):
    a, b = tr_L(G_score), tr_L(G_raw)
    D = a["_null_T"] - b["_null_T"]
    obs = float(D[_OBS_L])
    if not math.isfinite(obs):
        return {"diff": None}
    return {"diff": obs, "p_upper": float(np.mean(D >= obs - EPS)), "p_lower": float(np.mean(D <= obs + EPS)),
            "p_two": float(np.mean(np.abs(D) >= abs(obs) - EPS))}


def clean(t):
    return {k: v for k, v in t.items() if not k.startswith("_")}


def tr_bl(G8, G3):
    """Joint within-family trend (Tr_BLOOMZ + Tr_L)/2; exact p over 120 x 6 joint relabellings."""
    b = A12.lm_trend(G8)
    lt = tr_L(G3)
    J = (b["null_T"][:, None] + lt["_null_T"][None, :]) / 2.0
    obs = (b["Tr"] + lt["Tr"]) / 2.0 if lt["Tr"] is not None else float("nan")
    if not math.isfinite(obs):
        return {"Tr_BL": None}
    return {"Tr_BL": obs, "Tr_BLOOMZ": b["Tr"], "Tr_L": lt["Tr"], "p_one": float(np.mean(J >= obs - EPS)),
            "p_two": float(np.mean(np.abs(J) >= abs(obs) - EPS)), "n_relabel": int(J.size)}


# ----------------------------------------------------------------------------
# ARH-style pooled r (10-LM panel), exact relabelling p, ViT bootstrap
# ----------------------------------------------------------------------------

print("building 10! relabellings", flush=True)
_PERMS10 = np.array(list(itertools.permutations(range(10))), dtype=np.int8)
_XCP = {}


def _xcp(kind):
    if kind not in _XCP:
        x = {"arh": ARH_RANK10, "dense": DENSE10, "logp": LOGP10}[kind]
        _XCP[kind] = (x - x.mean())[_PERMS10]           # (10!, 10): value given to LM i under relabelling
    return _XCP[kind]


def pear(G, x):
    xx = np.repeat(x[:, None], G.shape[1], axis=1).ravel()
    y = G.ravel()
    if np.nanstd(y) == 0 or not np.all(np.isfinite(y)):
        return float("nan")
    return float(np.corrcoef(xx, y)[0, 1])


def arh_test(G, kind, key, exact=True):
    x = {"arh": ARH_RANK10, "dense": DENSE10, "logp": LOGP10}[kind]
    r = pear(G, x)
    out = {"r": r}
    if not math.isfinite(r):
        return out
    if exact:
        y = G - G.mean()
        rows = y.sum(axis=1)
        den = np.sqrt(G.shape[1] * np.sum((x - x.mean()) ** 2)) * np.sqrt(np.sum(y ** 2))
        R = (_xcp(kind) @ rows) / den
        out["p_one"] = float(np.mean(R >= r - EPS))
        out["n_relabel"] = int(R.size)
    rng = util.rng_from_key(f"S1.2-layermax-llama|boot|arh_r|{key}")
    bs = np.empty(NB)
    for b in range(NB):
        j = rng.integers(0, G.shape[1], G.shape[1])
        bs[b] = pear(G[:, j], x)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    out["ci"] = [float(lo), float(hi)]
    return out


def arh8(G8):
    x = np.array([LC.ARH_LLM_ORDER.index(s.replace("__", "/")) + 1 for s in FROZEN_LMS], dtype=float)
    return pear(G8, x)


def delta_boot(G1b, G1a, G2b, G2a, key):
    x = ARH_RANK10
    f = lambda a, b, c, d: (pear(a, x) - pear(b, x)) - (pear(c, x) - pear(d, x))  # noqa: E731
    est = f(G1b, G1a, G2b, G2a)
    rng = util.rng_from_key(f"S1.2-layermax-llama|boot|arh_delta|{key}")
    bs = np.empty(NB)
    for b in range(NB):
        j = rng.integers(0, NV, NV)
        bs[b] = f(G1b[:, j], G1a[:, j], G2b[:, j], G2a[:, j])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"est": float(est), "ci": [float(lo), float(hi)]}


def openllama_frac(G10):
    d = G10[6] - G10[5]
    return float(np.mean(d > 0))


# ----------------------------------------------------------------------------
# helpers for Part B rules
# ----------------------------------------------------------------------------

PAIRS3 = [(0, 1), (1, 2), (0, 2)]
PAIRKEY = {(0, 1): "huggyllama__llama-7b|huggyllama__llama-13b", (1, 2): "huggyllama__llama-13b|huggyllama__llama-30b",
           (0, 2): "huggyllama__llama-7b|huggyllama__llama-30b"}


def sgn(x, tol=0.0):
    return 0 if abs(x) <= tol else (1 if x > 0 else -1)


def cat_pl2(d, raw_tr, s):
    """P-L2 four-way rule (Part B section 5), for a material trend with sign s."""
    if d is None:
        return "undefined"
    if raw_tr is not None and abs(raw_tr - (-s)) <= EPS:
        return "not testable"
    if -s * d >= MAT:
        return "confirmed"
    if s * d >= MAT:
        return "contradicted"
    if abs(d) < MAT:
        return "consistent-null"
    return "undefined"


def cat_keep(tr_score, s):
    """P-L3 / P-L5 three-way rule."""
    if tr_score is None:
        return "undefined (all ViTs constant)"
    if s * tr_score >= MAT:
        return "keeps sign"
    if abs(tr_score) < MAT:
        return "lost"
    if s * tr_score <= -MAT:
        return "reversed"
    return "undefined"


def flips(Graw, Gsc, Gbase=None, gap_min=None, pre=None):
    """Per (ViT, size pair): order changes from raw to score. Concordant = larger LLaMA scores higher.
    If gap_min is given, only size pairs with Part B gap >= gap_min are counted (per ViT)."""
    res = {"n_compared": 0, "conc_to_disc": 0, "conc_to_tie": 0, "disc_to_conc": 0, "disc_to_tie": 0,
           "tie_raw": 0, "unchanged": 0, "per_size_pair": {}}
    for (a, b) in PAIRS3:
        pk = PAIRKEY[(a, b)]
        gaps = PB["per_pre"][pre]["llama_axis"]["Bbar"]["gaps"][pk]["gap_per_vit"] if gap_min is not None else None
        sp = {"n": 0, "conc_to_disc": 0, "disc_to_conc": 0, "conc_to_tie": 0, "disc_to_tie": 0}
        for v in range(NV):
            if gaps is not None and gaps[v] < gap_min:
                continue
            r0 = sgn(Graw[b, v] - Graw[a, v])
            r1 = sgn(Gsc[b, v] - Gsc[a, v])
            res["n_compared"] += 1
            sp["n"] += 1
            if r0 == 0:
                res["tie_raw"] += 1
            elif r0 == r1:
                res["unchanged"] += 1
            elif r0 > 0 and r1 < 0:
                res["conc_to_disc"] += 1
                sp["conc_to_disc"] += 1
            elif r0 > 0 and r1 == 0:
                res["conc_to_tie"] += 1
                sp["conc_to_tie"] += 1
            elif r0 < 0 and r1 > 0:
                res["disc_to_conc"] += 1
                sp["disc_to_conc"] += 1
            elif r0 < 0 and r1 == 0:
                res["disc_to_tie"] += 1
                sp["disc_to_tie"] += 1
        res["per_size_pair"][pk] = sp
    return res


def deductive_order(Graw, Gscore, Gthr, base, kind):
    """Corollary 2 / Part B deductive check: if |T_j - T_j'| > zeta_j + zeta_j' (zeta = |thr - base|),
    the order of the score must not reverse (ties from censoring at 0 are reported separately)."""
    n_q, n_rev, n_tie = 0, 0, 0
    viol = []
    for (a, b) in PAIRS3:
        for v in range(NV):
            za, zb = abs(Gthr[a, v] - base[a, v]), abs(Gthr[b, v] - base[b, v])
            gap = Graw[b, v] - Graw[a, v]
            if abs(gap) > za + zb:
                n_q += 1
                s1 = sgn(Gscore[b, v] - Gscore[a, v])
                if s1 == -sgn(gap):
                    n_rev += 1
                    viol.append(f"{VITS[v]}|{LLAMA[a]}<{LLAMA[b]}")
                elif s1 == 0:
                    n_tie += 1
    return {"kind": kind, "n_qualifying": n_q, "n_reversed": n_rev, "n_tied_after": n_tie, "violations": viol,
            "pass": n_rev == 0}


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def per_pre(pre):
    P = {"trends_L": {}, "delta_L": {}, "joint_BL": {}, "pooled": {}, "vision": {}, "per_lm": {}, "readings": {},
         "predictions": {}, "attribution": {}, "means": {}}
    pbp = PB["per_pre"][pre]
    # ---------------- 5: LLaMA-axis trends of every metric / protocol / score
    for met in METRICS:
        for subset in SUB_STD + SUB_NEW:
            for col in COLS:
                k = f"{met}|{subset}|{col}"
                P["trends_L"][k] = clean(tr_L(gridL(pre, met, subset, col), f"{pre}|{k}"))
            for col in ("s_cal", "uncensored"):
                P["delta_L"][f"{met}|{subset}|{col}"] = tr_diff(gridL(pre, met, subset, col),
                                                               gridL(pre, met, subset, "s_obs"))
        for col in ("M", "M_unc"):
            k = f"{met}|ctm|{col}"
            P["trends_L"][k] = clean(tr_L(gridL(pre, met, "ctm", col), f"{pre}|{k}"))
            P["delta_L"][k] = tr_diff(gridL(pre, met, "ctm", col), gridL(pre, met, "full", "s_obs"))
        for col in DECOMP[1:]:
            if col in COLS:
                continue
            k = f"{met}|full|{col}"
            P["trends_L"][k] = clean(tr_L(gridL(pre, met, "full", col), f"{pre}|{k}"))
    k = "cka_biased|full|minus_Bbar"
    P["trends_L"][k] = clean(tr_L(gridL(pre, "cka_biased", "full", "minus_Bbar"), f"{pre}|{k}"))
    # ---------------- joint Tr_BL (descriptive)
    for met in METRICS:
        for subset in ("full", "fixed_r0.5", "fixed_r1.0"):
            for col in COLS:
                P["joint_BL"][f"{met}|{subset}|{col}"] = tr_bl(grid8(pre, met, subset, col), gridL(pre, met, subset, col))
        P["joint_BL"][f"{met}|ctm|M"] = tr_bl(grid8(pre, met, "ctm", "M"), gridL(pre, met, "ctm", "M"))
    # ---------------- ARH-style pooled r (10 LMs, 170 pairs) + 8-LM comparison
    for met in METRICS:
        # (the plan-section-3(iii) subsets added for this axis are not in the frozen 8-LM pair files,
        #  so they have no pooled r)
        for subset in ("full", "fixed_r0.5", "fixed_r1.0", "grid17", "ctm"):
            cols = ("M", "M_unc") if subset == "ctm" else COLS
            for col in cols:
                k = f"{met}|{subset}|{col}"
                G10 = grid10(pre, met, subset, col)
                exact = subset in ("full", "fixed_r0.5", "fixed_r1.0", "ctm")
                blk = {"arh": arh_test(G10, "arh", f"{pre}|{k}", exact=exact),
                       "dense": pear(G10, DENSE10), "logp": pear(G10, LOGP10),
                       "r_8LM": arh8(grid8(pre, met, subset, col)),
                       "openllama_7b_gt_3b_frac": openllama_frac(G10), "mean": float(np.mean(G10))}
                if exact:
                    blk["dense_test"] = arh_test(G10, "dense", f"{pre}|{k}|dense", exact=True)
                P["pooled"][k] = blk
    # ---------------- vision axis restricted to LLaMA rows (descriptive)
    for met in METRICS:
        for col in COLS:
            for subset in ("full", "fixed_r0.5", "fixed_r1.0"):
                t = A12.vision_trend(gridL(pre, met, subset, col))
                P["vision"][f"{met}|{subset}|{col}"] = {"Tr": t["Tr"], "p_one": t["p_one"], "p_two": t["p_two"],
                                                        "n_cells": t["n_cells"]}
    # ---------------- per-LM means
    for met in METRICS:
        P["means"][met] = {}
        for subset in ("full", "fixed_r0.5", "fixed_r1.0"):
            for col in ("s_obs", "c_alpha", "s_cal", "uncensored", "null_mean", "p_value"):
                G = gridL(pre, met, subset, col)
                P["means"][met][f"{subset}|{col}"] = {lm: float(G[i].mean()) for i, lm in enumerate(LLAMA)}
        for col in ("M", "M_unc"):
            G = gridL(pre, met, "ctm", col)
            P["means"][met][f"ctm|{col}"] = {lm: float(G[i].mean()) for i, lm in enumerate(LLAMA)}
        G = gridL(pre, met, "full", "s_cal")
        P["means"][met]["full|frac_s_cal_zero"] = {lm: float(np.mean(G[i] <= 0)) for i, lm in enumerate(LLAMA)}
        G = gridL(pre, met, "full", "p_value")
        P["means"][met]["full|frac_p_le_0.05"] = {lm: float(np.mean(G[i] <= 0.05)) for i, lm in enumerate(LLAMA)}

    def T(met, subset="full", col="s_obs"):
        return P["trends_L"][f"{met}|{subset}|{col}"]["Tr"]

    def D(met, subset="full", col="s_cal"):
        return P["delta_L"][f"{met}|{subset}|{col}"]["diff"]

    # ---------------- Part B predictions
    pr = {}
    # P-L1 (mKNN, layer-max)
    Graw = gridL(pre, "mknn", "full", "s_obs")
    base = np.full((3, NV), BASE_MKNN)
    d1s, d1u = D("mknn", "full", "s_cal"), D("mknn", "full", "uncensored")
    pr["P-L1"] = {"Tr_raw": T("mknn"), "Tr_scal": T("mknn", col="s_cal"), "Tr_unc": T("mknn", col="uncensored"),
                  "dTr_scal": d1s, "dTr_unc": d1u,
                  "holds": bool(d1s is not None and d1u is not None and abs(d1s) < MAT and abs(d1u) < MAT),
                  "zeta_calpha_mean_by_lm": {lm: float(np.mean(np.abs(gridL(pre, "mknn", "full", "c_alpha")[i] - BASE_MKNN)))
                                             for i, lm in enumerate(LLAMA)},
                  "deductive_scal": deductive_order(Graw, gridL(pre, "mknn", "full", "s_cal"),
                                                    gridL(pre, "mknn", "full", "c_alpha"), base, "s_cal vs c_alpha"),
                  "deductive_unc": deductive_order(Graw, gridL(pre, "mknn", "full", "uncensored"),
                                                   gridL(pre, "mknn", "full", "null_mean"), base, "uncensored vs E0[max]"),
                  "flips_scal": flips(Graw, gridL(pre, "mknn", "full", "s_cal"), pre=pre),
                  "flips_unc": flips(Graw, gridL(pre, "mknn", "full", "uncensored"), pre=pre)}
    pr["P-L1"]["outcome"] = "holds" if pr["P-L1"]["holds"] else "fails (plan section 10 item 3)"
    # P-L2 / P-L2a (biased CKA, layer-max)
    mat = bool(pbp["llama_axis"]["Bbar"]["material"])
    s = 1 if pbp["llama_axis"]["Bbar"]["strictly_increasing_in_size_every_vit"] else (
        -1 if pbp["llama_axis"]["Bbar"]["strictly_decreasing_in_size_every_vit"] else 0)
    Graw = gridL(pre, "cka_biased", "full", "s_obs")
    Gc = gridL(pre, "cka_biased", "full", "c_alpha")
    Gnm = gridL(pre, "cka_biased", "full", "null_mean")
    Bbar = np.array([[pbp["Bbar"][lm][vit] for vit in VITS] for lm in LLAMA])
    pl2 = {"material": mat, "s": s, "Tr_pred": pbp["llama_axis"]["Bbar"]["trend"]["Tr"],
           "Tr_raw": T("cka_biased"), "Tr_scal": T("cka_biased", col="s_cal"), "Tr_unc": T("cka_biased", col="uncensored"),
           "dTr_scal": D("cka_biased"), "dTr_unc": D("cka_biased", col="uncensored"),
           "p_dTr_scal": P["delta_L"]["cka_biased|full|s_cal"], "p_dTr_unc": P["delta_L"]["cka_biased|full|uncensored"]}
    if mat:
        pl2["outcome_scal"] = cat_pl2(pl2["dTr_scal"], pl2["Tr_raw"], s)
        pl2["outcome_unc_secondary"] = cat_pl2(pl2["dTr_unc"], pl2["Tr_raw"], s)
        pl2["flips_scal"] = flips(Graw, gridL(pre, "cka_biased", "full", "s_cal"), pre=pre)
        pl2["flips_unc"] = flips(Graw, gridL(pre, "cka_biased", "full", "uncensored"), pre=pre)
    else:
        pl2["outcome_scal"] = "no directional prediction (B-bar trend not material)"
        pl2["outcome_unc_secondary"] = pl2["outcome_scal"]
        pl2["flips_scal_gap_ge_delta_b"] = flips(Graw, gridL(pre, "cka_biased", "full", "s_cal"), gap_min=DELTA_B, pre=pre)
        pl2["flips_unc_gap_ge_delta_b"] = flips(Graw, gridL(pre, "cka_biased", "full", "uncensored"), gap_min=DELTA_B,
                                                pre=pre)
    pr["P-L2"] = pl2
    absdiff = np.abs(Gc - Bbar)
    comps, agree = 0, 0
    per_sp = {}
    for (a, b) in PAIRS3:
        gaps = pbp["llama_axis"]["Bbar"]["gaps"][PAIRKEY[(a, b)]]["gap_per_vit"]
        n_q, n_a = 0, 0
        for v in range(NV):
            if gaps[v] >= DELTA_B:
                n_q += 1
                n_a += int(Gc[b, v] > Gc[a, v])
        per_sp[PAIRKEY[(a, b)]] = {"n_qualifying": n_q, "n_calpha_larger_higher": n_a}
        comps += n_q
        agree += n_a
    need = int(math.ceil(2 * comps / 3)) if comps else 0
    # deductive check: monotone c_alpha order -> rho_v(s_cal) - rho_v(raw) has the opposite sign (or 0)
    rho_raw = P["trends_L"]["cka_biased|full|s_obs"]["rho_per_vit"]
    rho_sc = P["trends_L"]["cka_biased|full|s_cal"]["rho_per_vit"]
    rho_un = P["trends_L"]["cka_biased|full|uncensored"]["rho_per_vit"]
    ded = {"n_vit_calpha_increasing": 0, "n_vit_calpha_decreasing": 0, "violations_scal": [], "violations_unc": [],
           "n_vit_nullmean_increasing": 0}
    for v in range(NV):
        inc = Gc[0, v] < Gc[1, v] < Gc[2, v]
        dec = Gc[0, v] > Gc[1, v] > Gc[2, v]
        ded["n_vit_calpha_increasing"] += int(inc)
        ded["n_vit_calpha_decreasing"] += int(dec)
        if (inc or dec) and rho_raw[v] is not None and rho_sc[v] is not None:
            dd = rho_sc[v] - rho_raw[v]
            if (inc and dd > EPS) or (dec and dd < -EPS):
                cens = bool(np.any(gridL(pre, "cka_biased", "full", "s_cal")[:, v] <= 0))
                ded["violations_scal"].append({"vit": VITS[v], "d_rho": dd, "censoring_involved": cens})
        incm = Gnm[0, v] < Gnm[1, v] < Gnm[2, v]
        decm = Gnm[0, v] > Gnm[1, v] > Gnm[2, v]
        ded["n_vit_nullmean_increasing"] += int(incm)
        if (incm or decm) and rho_raw[v] is not None and rho_un[v] is not None:
            dd = rho_un[v] - rho_raw[v]
            if (incm and dd > EPS) or (decm and dd < -EPS):
                ded["violations_unc"].append({"vit": VITS[v], "d_rho": dd})
    ded["pass"] = not [x for x in ded["violations_scal"] if not x["censoring_involved"]] and not ded["violations_unc"]
    pl2a = {"median_abs_calpha_minus_Bbar": float(np.median(absdiff)), "max_abs_calpha_minus_Bbar": float(absdiff.max()),
            "mean_calpha_minus_Bbar": float(np.mean(Gc - Bbar)),
            "range_calpha_minus_Bbar": [float((Gc - Bbar).min()), float((Gc - Bbar).max())],
            "mean_E0max_minus_Bbar": float(np.mean(Gnm - Bbar)),
            "pearson_calpha_Bbar": float(np.corrcoef(Gc.ravel(), Bbar.ravel())[0, 1]),
            "calpha_mean_by_lm": {lm: float(Gc[i].mean()) for i, lm in enumerate(LLAMA)},
            "Bbar_mean_by_lm": {lm: float(Bbar[i].mean()) for i, lm in enumerate(LLAMA)},
            "Tr_L_calpha": T("cka_biased", col="c_alpha"),
            "n_qualifying": comps, "n_agree_larger_higher": agree, "n_needed": need, "per_size_pair": per_sp,
            "i_holds": bool(np.median(absdiff) <= 0.01), "ii_holds": bool(comps > 0 and agree >= need),
            "deductive": ded}
    pl2a["holds"] = bool(pl2a["i_holds"] and pl2a["ii_holds"])
    pr["P-L2a"] = pl2a
    # P-L3 (biased CKA, fixed depth)
    pl3 = {}
    for r in ("0.5", "1.0"):
        sub = f"fixed_r{r}"
        tr_raw, tr_sc, tr_un = T("cka_biased", sub), T("cka_biased", sub, "s_cal"), T("cka_biased", sub, "uncensored")
        blk = {"Tr_raw": tr_raw, "Tr_scal": tr_sc, "Tr_unc": tr_un, "dTr_scal": D("cka_biased", sub),
               "Bfd_material": bool(pbp["llama_axis"]["Bfd"][r]["material"]),
               "Bfd_mean_by_lm": pbp["llama_axis"]["Bfd"][r]["mean_over_vits_by_model"],
               "calpha_mean_by_lm": P["means"]["cka_biased"][f"{sub}|c_alpha"],
               "flips_scal": flips(gridL(pre, "cka_biased", sub, "s_obs"), gridL(pre, "cka_biased", sub, "s_cal"), pre=pre)}
        if tr_raw is not None and abs(tr_raw) >= MAT:
            s_r = 1 if tr_raw > 0 else -1
            blk["s_r"] = s_r
            blk["applies"] = True
            blk["outcome"] = cat_keep(tr_sc, s_r)
            blk["outcome_unc_descriptive"] = cat_keep(tr_un, s_r)
        else:
            blk["applies"] = False
            blk["outcome"] = "not applicable (abs Tr_L(raw, r) < 0.10)"
        pl3[r] = blk
    pr["P-L3"] = pl3
    # P-L4 (debiased CKA)
    d4s, d4u = D("cka_debiased"), D("cka_debiased", col="uncensored")
    rb = P["pooled"]["cka_debiased|full|s_obs"]["arh"]["r"]
    ra = P["pooled"]["cka_debiased|full|s_cal"]["arh"]["r"]
    pl4 = {"Tr_raw": T("cka_debiased"), "Tr_scal": T("cka_debiased", col="s_cal"),
           "Tr_unc": T("cka_debiased", col="uncensored"), "dTr_scal": d4s, "dTr_unc": d4u,
           "a_holds": bool(d4s is not None and d4u is not None and d4s >= -MAT and d4u >= -MAT),
           "pooled_r_before": rb, "pooled_r_after": ra,
           "b_holds": bool(math.isfinite(ra) and math.isfinite(rb) and ra >= rb - 0.05),
           "flips_scal": flips(gridL(pre, "cka_debiased", "full", "s_obs"), gridL(pre, "cka_debiased", "full", "s_cal"),
                               pre=pre)}
    pl4["holds"] = bool(pl4["a_holds"] and pl4["b_holds"])
    pl4["outcome"] = "holds" if pl4["holds"] else "fails (plan section 10 item 3)"
    pr["P-L4"] = pl4
    # P-L5 (calibrate-then-max)
    tr_raw = T("cka_biased")
    pl5 = {"Tr_raw_layermax": tr_raw, "Tr_M": P["trends_L"]["cka_biased|ctm|M"]["Tr"],
           "Tr_M_unc": P["trends_L"]["cka_biased|ctm|M_unc"]["Tr"],
           "M_mean_by_lm": P["means"]["cka_biased"]["ctm|M"]}
    if tr_raw is not None and abs(tr_raw) >= MAT:
        s5 = 1 if tr_raw > 0 else -1
        pl5.update({"applies": True, "s": s5, "outcome": cat_keep(pl5["Tr_M"], s5),
                    "outcome_unc_descriptive": cat_keep(pl5["Tr_M_unc"], s5)})
    else:
        pl5.update({"applies": False, "outcome": "not applicable (abs Tr_L(raw T, layer-max) < 0.10)"})
    pr["P-L5"] = pl5
    P["predictions"] = pr
    # ---------------- plan section 10 summary
    it = {}
    it["item1_premise_failure"] = not pl2a["holds"]
    it["item2_PL2_contradicted"] = pl2["outcome_scal"] == "contradicted"
    it["item3_PL1_or_PL4_fails"] = (not pr["P-L1"]["holds"]) or (not pl4["holds"])
    it["item4_PL3_or_PL5_reversed"] = any(v.get("outcome") == "reversed" for v in pl3.values()) or \
        pl5.get("outcome") == "reversed"
    it["deductive_checks_pass"] = bool(pr["P-L1"]["deductive_scal"]["pass"] and pr["P-L1"]["deductive_unc"]["pass"]
                                       and ded["pass"])
    P["section10"] = it
    # ---------------- plan section 7 readings
    rd = {}
    for cka in ("cka_biased", "cka_debiased"):
        rr = lambda met, col: P["pooled"][f"{met}|full|{col}"]["arh"]["r"]  # noqa: E731
        d_c = rr(cka, "s_obs") - rr(cka, "s_cal")
        d_m = rr("mknn", "s_obs") - rr("mknn", "s_cal")
        gap_after = rr("mknn", "s_cal") - rr(cka, "s_cal")
        if not all(math.isfinite(v) for v in (d_c, d_m, gap_after)):
            v = "undefined (constant scores)"
        elif (d_c - d_m) >= 0.20 and gap_after >= 0.20:
            v = "ARH pattern reappears"
        elif (d_c - d_m) <= 0.10 or gap_after <= 0.10:
            v = "ARH pattern absent"
        else:
            v = "partial"
        boot = delta_boot(grid10(pre, cka, "full", "s_obs"), grid10(pre, cka, "full", "s_cal"),
                          grid10(pre, "mknn", "full", "s_obs"), grid10(pre, "mknn", "full", "s_cal"), f"{pre}|{cka}")
        trc_raw = T(cka)
        trc, trm = T(cka, col="uncensored"), T("mknn", col="uncensored")
        if trc_raw is None or trc_raw < 0.60 or trc is None or trm is None:
            v2 = "no raw trend to lose"
        elif trc <= 0.40 and trm >= 0.60:
            v2 = "layer-max loses the CKA trend on the LLaMA axis"
        elif trc >= 0.60:
            v2 = "no loss"
        else:
            v2 = "intermediate"
        rd[cka] = {"r_before_cka": rr(cka, "s_obs"), "r_after_cka": rr(cka, "s_cal"),
                   "r_before_mknn": rr("mknn", "s_obs"), "r_after_mknn": rr("mknn", "s_cal"),
                   "delta_cka": d_c, "delta_mknn": d_m, "delta_diff": d_c - d_m, "delta_diff_ci": boot["ci"],
                   "r_after_gap_mknn_minus_cka": gap_after, "verdict_arh_style_10LM": v,
                   "Tr_L_raw_cka": trc_raw, "Tr_L_unc_cka": trc, "Tr_L_unc_mknn": trm, "verdict_llama_axis": v2}
        # 8-LM comparison (frozen run, recomputed here from the frozen pair files)
        d_c8 = arh8(grid8(pre, cka, "full", "s_obs")) - arh8(grid8(pre, cka, "full", "s_cal"))
        d_m8 = arh8(grid8(pre, "mknn", "full", "s_obs")) - arh8(grid8(pre, "mknn", "full", "s_cal"))
        rd[cka]["r8_before_cka"] = arh8(grid8(pre, cka, "full", "s_obs"))
        rd[cka]["r8_after_cka"] = arh8(grid8(pre, cka, "full", "s_cal"))
        rd[cka]["r8_after_mknn"] = arh8(grid8(pre, "mknn", "full", "s_cal"))
        rd[cka]["delta_diff_8LM"] = d_c8 - d_m8
        rd[cka]["verdict_arh_style_8LM_frozen"] = FRZ_SUMMARY["per_pre"][pre]["reading"][cka]["verdict_arh_style"]
    P["readings"] = rd
    # ---------------- attribution diagnostics (plan section 6), LLaMA axis
    at = {"selection": {}, "null_vs_size": {}, "subset_curve": {}, "decomp": {}, "level_vs_multiplicity": {},
          "null_profile": {}, "plugin_check": {}}
    for met in METRICS:
        GA = gridL(pre, met, "full", "rel_A")
        GB = gridL(pre, met, "full", "rel_B")
        GAa = gridL(pre, met, "full", "argmax_A")
        nt = gridL(pre, met, "full", "n_tied_max")
        sel = {"rel_A_mean_by_lm": {lm: float(GA[i].mean()) for i, lm in enumerate(LLAMA)},
               "rel_A_median_by_lm": {lm: float(np.median(GA[i])) for i, lm in enumerate(LLAMA)},
               "argmax_A_mode_by_lm": {lm: int(np.bincount(GAa[i].astype(int)).argmax()) for i, lm in enumerate(LLAMA)},
               "rel_B_mean_by_lm": {lm: float(GB[i].mean()) for i, lm in enumerate(LLAMA)},
               "frac_lm_embedding_selected": float(np.mean(GAa == 0)),
               "n_tied_max_gt1": int(np.sum(nt > 1)),
               "tr_L_relA": clean(tr_L(GA))}
        nA = {lm: [] for lm in LLAMA}
        for i, lm in enumerate(LLAMA):
            for vit in VITS:
                z = np.load(os.path.join(LC.RES, "pairs", pre, f"{lm}__{vit}.npz"))
                nA[lm].append(z[f"{met}.full_null_argmax_A"] / (LC.n_stored(lm) - 1))
        sel["null_rel_A_mean_by_lm"] = {lm: float(np.concatenate(v).mean()) for lm, v in nA.items()}
        sel["null_frac_lm_embedding_by_lm"] = {lm: float(np.mean(np.concatenate(v) == 0)) for lm, v in nA.items()}
        at["selection"][met] = sel
        Gc = gridL(pre, met, "full", "c_alpha")
        npairs = np.array([[REC[(pre, lm, vit)]["L_A"] * REC[(pre, lm, vit)]["L_B"] for vit in VITS] for lm in LLAMA])
        at["null_vs_size"][met] = {"spearman_calpha_vs_nlayerpairs": float(spearmanr(Gc.ravel(), npairs.ravel()).statistic),
                                   "calpha_mean_by_lm": {lm: float(Gc[i].mean()) for i, lm in enumerate(LLAMA)},
                                   "inflation_mean_by_lm": {lm: float(gridL(pre, met, "full", "inflation")[i].mean())
                                                            for i, lm in enumerate(LLAMA)}}
        cur = {}
        for subset in SUB_STD + SUB_NEW:
            cur[subset] = {"mean_n_layer_pairs": float(gridL(pre, met, subset, "n_layer_pairs").mean()),
                           "mean_s_obs_by_lm": {lm: float(gridL(pre, met, subset, "s_obs")[i].mean()) for i, lm in enumerate(LLAMA)},
                           "mean_c_alpha_by_lm": {lm: float(gridL(pre, met, subset, "c_alpha")[i].mean()) for i, lm in enumerate(LLAMA)},
                           "tr_raw": P["trends_L"][f"{met}|{subset}|s_obs"]["Tr"],
                           "tr_scal": P["trends_L"][f"{met}|{subset}|s_cal"]["Tr"],
                           "tr_unc": P["trends_L"][f"{met}|{subset}|uncensored"]["Tr"],
                           "tr_calpha": clean(tr_L(gridL(pre, met, subset, "c_alpha")))["Tr"]}
        at["subset_curve"][met] = cur
        at["decomp"][met] = {col: P["trends_L"][f"{met}|full|{col}"]["Tr"] for col in DECOMP}
        # level vs multiplicity and per-LM-layer null profile (from stored f64 entry null means)
        lvm, prof = [], {}
        for i, lm in enumerate(LLAMA):
            profs, argl = [], []
            for vit in VITS:
                z = np.load(os.path.join(LC.RES, "pairs", pre, f"{lm}__{vit}.npz"))
                nm = z[f"{met}.entry_null_mean"]
                profs.append(nm.max(axis=1))
                a0, _ = np.unravel_index(np.argmax(nm), nm.shape)
                argl.append(int(a0))
                lvm.append(REC[(pre, lm, vit)]["summaries"][met]["full"]["null_mean"] - float(nm.max()))
            Mp = np.mean(np.stack(profs), axis=0)
            prof[lm] = {"null_level_by_layer": [float(x) for x in Mp], "argmax_layer": int(np.argmax(Mp)),
                        "argmax_rel": float(np.argmax(Mp) / (Mp.size - 1)), "max": float(Mp.max()),
                        "emb": float(Mp[0]), "last": float(Mp[-1]),
                        "frac_vits_argmax_at_emb": float(np.mean(np.array(argl) == 0)),
                        "partB_u_star": pbp["llama_axis"]["threshold_setting_LM_layer"][lm]["u_star"]
                        if met == "cka_biased" else None}
        at["level_vs_multiplicity"][met] = {"mean_E0max_minus_maxE0": float(np.mean(lvm)),
                                            "max_E0max_minus_maxE0": float(np.max(lvm))}
        at["null_profile"][met] = prof
    tu = P["trends_L"]["cka_biased|full|uncensored"]["Tr"]
    tb = P["trends_L"]["cka_biased|full|minus_Bbar"]["Tr"]
    at["plugin_check"] = {"Tr_L_T_minus_Bbar": tb, "Tr_L_uncensored": tu,
                          "abs_diff": abs(tb - tu) if (tb is not None and tu is not None) else None,
                          "within_0.10": bool(tb is not None and tu is not None and abs(tb - tu) <= MAT)}
    P["attribution"] = at
    return P


def selfchecks():
    """Trend code vs S1.2 (BLOOMZ rows of the frozen run) and protocol (iv) torch vs numpy on llama-7b."""
    out = {}
    G8 = grid8("prh_clip+l2", "cka_biased", "full", "uncensored")
    a = A12.lm_trend(G8)
    # the LLaMA-axis trend code applied to the BLOOMZ rows with the BLOOMZ sizes must equal A12.lm_trend
    bl = [i for i, s in enumerate(FROZEN_LMS) if FAM[s] == "BLOOMZ"]
    x = rankdata([LC.n_params(FROZEN_LMS[i]) for i in bl])
    Pm = np.array([np.array(p) for p in itertools.permutations(x)])
    rho = A12._rho_matrix(Pm, G8[bl])
    oi = int(np.flatnonzero(np.all(Pm == x, axis=1))[0])
    out["trend_code_vs_A12_bloomz"] = {"A12": a["Tr"], "ours": float(np.nanmean(rho, axis=1)[oi]),
                                       "equal": bool(abs(a["Tr"] - float(np.nanmean(rho, axis=1)[oi])) == 0.0)}
    fz = FRZ_SUMMARY["per_pre"]["prh_clip+l2"]["trends"]["cka_biased"]["full|uncensored"]["tr_lm"]["Tr"]
    out["frozen_summary_bloomz_Tr_unc_cka"] = {"frozen": fz, "recomputed": a["Tr"], "equal": bool(fz == a["Tr"])}
    md = 0.0
    for pre in PRES:
        for vit in VITS:
            for met in METRICS:
                e = REC[(pre, "huggyllama__llama-7b", vit)]["calib_then_max"][met]
                f = FRZ_CTM[(pre, "huggyllama__llama-7b", vit, met)]
                md = max(md, abs(e["M"] - f["M"]), abs(e["M_unc"] - f["M_unc"]))
    out["ctm_llama7b_E_torch_vs_frozen_npz_numpy_maxabs"] = md
    r8 = arh8(grid8("prh_clip+l2", "cka_biased", "full", "s_obs"))
    out["arh_r_8LM_recomputed_vs_frozen"] = {
        "recomputed": r8, "frozen": FRZ_SUMMARY["per_pre"]["prh_clip+l2"]["trends"]["cka_biased"]["full|s_obs"]["arh_r"]["r"]}
    # llama-7b rows of the pooled panel: E vs frozen (bitwise expected after V-L4(b))
    mx = 0.0
    for pre in PRES:
        for met in METRICS:
            for col in COLS:
                mx = max(mx, float(np.max(np.abs(gridL(pre, met, "full", col)[0] - grid8(pre, met, "full", col)[7]))))
    out["llama7b_E_vs_frozen_full_maxabs"] = mx
    return out


def main():
    out = {"role": "llama-axis-run-early", "implementation": "E", "plan": "analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md",
           "partB": "analysis/s12_layermax_llama/PARTB_PREDICTIONS.md",
           "panel": {"llama": LLAMA, "vits": VITS, "pooled_lms": POOLED, "arh_rank10": ARH_RANK10.tolist(),
                     "n_pairs_llama": len(LLAMA) * NV, "n_pairs_pooled": 10 * NV},
           "checks": {"n_records": len(REC), "K_values": sorted({r["K"] for r in REC.values()}),
                      "perm_keys_ok": all(r["perm_key"] == LC.perm_key(r["lm"], r["vit"]) for r in REC.values())},
           "selfchecks": selfchecks(), "per_pre": {}}
    for pre in PRES:
        print("aggregating", pre, flush=True)
        out["per_pre"][pre] = per_pre(pre)
    out["finished_utc"] = LC.utc()
    with open(os.path.join(LC.RES, "summary.json"), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    print("written", os.path.join(LC.RES, "summary.json"), flush=True)


if __name__ == "__main__":
    main()
