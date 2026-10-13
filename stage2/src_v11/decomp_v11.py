"""S2.2 v1.1 decompositions (PREREG_S2_2_v1.1.md section 7; extends v1.0 review a2_gauge_decomposition.py).

Per pair and criterion Q in R_cf:
  X := D(u) - D(y);  X = S + M with S := D(u^c) - D(y), M := D(u) - D(u^c);
  X = B + I with B := D(u_bal) - D(y), I := D(u) - D(u_bal);
  2x2: S_bal := D(u_bal,c) - D(y); M_bal := B - S_bal; I_S := S - S_bal; I_M := M - M_bal.
  (identities: X = S + M = B + I, B = S_bal + M_bal, I = I_S + I_M.)
u^c = Gbar^{1/2} y (Gbar = log-Euclidean mean of det-normalised (W W^T)^{-1} over the arm's PILOT runs at that N,
frozen with the eps file); u^c_pair (secondary) = v1.0 definition, D(u^c_pair) := (D(common_A) + D(common_B)) / 2;
u_bal = N^{-1/4} y; u_bal,c = Gbar_bal^{1/2} y (Gbar_bal = log-Euclidean mean of det-normalised N^{-1/2}).
Also: the I4 decision map (section 10.2), used on FLOW and in the KA5 plant test.
"""
from __future__ import annotations

import numpy as np

import common_v11 as CM
import gaugemath_v11 as GM

PARTS = ("X", "S", "M", "B", "I", "S_bal", "M_bal", "I_S", "I_M", "S_pair", "M_pair")
DELTA_EQ = 0.5          # I4 equivalence margin in sigma_y units (author decision 2026-10-09)
I4_LEVEL = 1 - 0.05 / 4  # simultaneous 98.75% (Bonferroni over the 4 (Q, N) cells)


def gbars(runs):
    """(Gbar, Gbar_bal, n) from a list of loaded runs (pilot runs of one arm and N). Runs that are not ok, or whose
    G = (W W^T)^{-1} or N = M M^T is not numerically SPD (resolution R-NUM), are skipped; (None, None, 0) if none is left."""
    Gs, Gb = [], []
    for run in runs:
        if not CM.run_ok(run):
            continue
        P, W = np.asarray(run["feats"]["P"], np.float64), np.asarray(run["feats"]["W"], np.float64)
        try:
            if not (GM.spd_ok(GM.G_of(W)) and GM.spd_ok(GM.N_of(P, W))):
                continue
            Gs.append(GM.norm_metric(W))
            Gb.append(GM.det_normalize(GM.Gbal_of(P, W)))
        except (np.linalg.LinAlgError, ValueError):
            continue
    if not Gs:
        return None, None, 0
    return GM.log_euclid_mean(Gs), GM.log_euclid_mean(Gb), len(Gs)


def decompose(d):
    """d: {tag: deltas dict} for one pair with tags y, u, uc, ucpA, ucpB, ubal, ubalc. Returns {Q: {part: value}}."""
    out = {}
    for Q in CM.R_CF:
        g = lambda t: float(d[t].get(Q, np.nan)) if t in d else float("nan")
        y, u, uc, ub, ubc = g("y"), g("u"), g("uc"), g("ubal"), g("ubalc")
        ucp = 0.5 * (g("ucpA") + g("ucpB"))
        X, S, M = u - y, uc - y, u - uc
        B, I = ub - y, u - ub
        S_bal = ubc - y
        M_bal = B - S_bal
        out[Q] = {"X": X, "S": S, "M": M, "B": B, "I": I, "S_bal": S_bal, "M_bal": M_bal, "I_S": S - S_bal,
                  "I_M": M - M_bal, "S_pair": ucp - y, "M_pair": u - ucp, "y": y, "u": u, "uc": uc, "ubal": ub,
                  "ubalc": ubc, "ucp": ucp, "y_plus_M": y + M, "y_plus_I": y + I}
    return out


def pair_jobs(lroot, cohort, pairs, Gbar, Gbar_bal, tags=("y", "u", "uc", "ucpA", "ucpB", "ubal", "ubalc")):
    gb = None if Gbar is None else np.asarray(Gbar).tolist()
    gbb = None if Gbar_bal is None else np.asarray(Gbar_bal).tolist()
    return [(lroot, cohort, t, a, b, gb if t == "uc" else None, gbb if t == "ubalc" else None)
            for (a, b) in pairs for t in tags]


def cell_parts(res, pairs):
    """res from CM.run_ladders; returns list (per pair) of decompose() outputs."""
    out = []
    for a, b in pairs:
        d = {t: res[(t, a, b)] for t in ("y", "u", "uc", "ucpA", "ucpB", "ubal", "ubalc") if (t, a, b) in res}
        out.append(decompose(d))
    return out


def sigma_y(res, pairs, Q):
    v = np.array([res[("y", a, b)].get(Q, np.nan) for a, b in pairs], float)
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1)) if v.size > 1 else float("nan")


def summarize_parts(parts_list, sig, key):
    """Mean / median / 95% bootstrap CI / share > 0 of every part, raw and in sigma_y units, per Q."""
    out = {}
    for Q in CM.R_CF:
        o = {}
        for p in PARTS:
            v = [pp[Q][p] for pp in parts_list]
            b = CM.boot_mean_ci(v, f"{key}|{Q}|{p}")
            b["sigma_units"] = {"mean": b["mean"] / sig[Q] if sig.get(Q) else float("nan"),
                                "ci": [c / sig[Q] for c in b["ci"]] if sig.get(Q) else [float("nan")] * 2}
            o[p] = b
        X = o["X"]["mean"]
        o["S_over_X"] = o["S"]["mean"] / X if X else float("nan")
        o["I_over_X"] = o["I"]["mean"] / X if X else float("nan")
        o["sigma_y"] = sig.get(Q)
        out[Q] = o
    return out


def i4_map(parts_by_N, sig_by_N, key, nboot=10000):
    """I4 decision map. parts_by_N: {N: list of per-pair decompose() outputs}; sig_by_N: {N: {Q: sigma_y}}.
    Intervals: simultaneous 98.75% percentile bootstrap CIs of mean I and mean I_M, Q in {G_s, 1-CKA}, in sigma_y units.
    (i) realized: lower bounds of I_Gs and I_M,Gs > 0 at both N.
    (ii) not detectable: CIs of I_Gs and I_CKA inside [-0.5, +0.5] sigma_y at both N.
    (i) and (ii) together: reported as (i), size inside the equivalence margin (resolution R-I4, prereg section 17).
    (iii) otherwise."""
    ci = {}
    for N, pl in parts_by_N.items():
        for Q in ("Gs", "dCKA"):
            for p in ("I", "I_M"):
                v = [pp[Q][p] for pp in pl]
                b = CM.boot_mean_ci(v, f"{key}|I4|{N}|{Q}|{p}", level=I4_LEVEL, n=nboot)
                s = sig_by_N[N][Q]
                ci[f"{N}|{Q}|{p}"] = {"mean_raw": b["mean"], "ci_raw": b["ci"], "mean_sigma": b["mean"] / s,
                                      "ci_sigma": [b["ci"][0] / s, b["ci"][1] / s], "n": b["n"], "sigma_y": s}
    Ns = sorted(parts_by_N)
    realized = all(ci[f"{N}|Gs|I"]["ci_sigma"][0] > 0 and ci[f"{N}|Gs|I_M"]["ci_sigma"][0] > 0 for N in Ns)
    within = all(-DELTA_EQ <= ci[f"{N}|{Q}|I"]["ci_sigma"][0] and ci[f"{N}|{Q}|I"]["ci_sigma"][1] <= DELTA_EQ
                 for N in Ns for Q in ("Gs", "dCKA"))
    if realized and within:
        outcome = "(i) realized, size within the 0.5 sigma_y equivalence margin"
    elif realized:
        outcome = "(i) realized"
    elif within:
        outcome = "(ii) not detectable"
    else:
        outcome = "(iii) estimates only"
    return {"outcome": outcome, "realized": realized, "within_margin": within, "intervals": ci,
            "level": I4_LEVEL, "delta_eq_sigma": DELTA_EQ}
