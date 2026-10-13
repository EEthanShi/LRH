"""S2.2 v1.1 KA5 plants (PREREG_S2_2_v1.1.md section 14, step 3): known gauges planted on v1.0 runs (READ-ONLY).

A plant keeps each run's function y and end-to-end map M = W P (both from the v1.0 run) and replaces the head by a
planted factorization (P', W') with W' P' = M, so that u' = W'^{-1} y has a KNOWN metric G' = (W' W'^T)^{-1}:
  common          G'_a = Gbar_ref                                   (one common gauge for every run)
  common+run<s>   G'_a = Gbar_ref^{1/2} expm(s Z_a) Gbar_ref^{1/2}  (common gauge + run-specific perturbation)
  balanced        W'_a = N_a^{1/4}  (so C' = P'P'^T - W'^T W' = 0 and u' = u_bal: I = 0 by construction)
  balanced+run<s> G'_a = N_a^{-1/4} expm(s Z_a) N_a^{-1/4}           (run-specific imbalance around the balanced head)
with W'_a = G'_a^{-1/2} (symmetric root), P'_a = W'_a^{-1} M_a, Z_a a symmetric traceless Gaussian matrix with
||Z_a||_F = 1 drawn from sha256("S2.2v11|KA5|Z|<run_id>"), and Gbar_ref the log-Euclidean mean of the det-normalised
v1.0 noWD (W W^T)^{-1} at the same N (a real, anisotropic common gauge). Expected (planted order):
  common: S > 0 and M = 0 (u^c = u' exactly);  common+run0.15, common+run0.30: S > 0, 0 < M(0.15) < M(0.30);
  balanced: I = 0 and X = B (to rounding);     balanced+run0.30: I > 0 and I_M > 0 on G_s (I4 'realized').
"""
from __future__ import annotations

import os

import numpy as np

import common_v11 as CM
import crit as C
import gaugemath_v11 as GM

PLANTS = {"common": ("common", 0.0), "common+run0.15": ("common", 0.15), "common+run0.30": ("common", 0.30),
          "balanced": ("balanced", 0.0), "balanced+run0.30": ("balanced", 0.30)}


def z_of(rid):
    rng = CM.key_rng(f"S2.2v11|KA5|Z|{rid}")
    A = rng.standard_normal((9, 9))
    Z = 0.5 * (A + A.T)
    Z -= np.trace(Z) / 9 * np.eye(9)
    return Z / np.linalg.norm(Z)


def planted_head(P, W, plant, rid, Gref=None):
    base, s = PLANTS[plant]
    P = np.asarray(P, np.float64)
    W = np.asarray(W, np.float64)
    M = W @ P
    if base == "common":
        R = GM.sqrtm_spd(Gref)
        G = R @ GM.expm_sym(s * z_of(rid)) @ R if s > 0 else np.array(Gref, np.float64)
    else:
        if s == 0:
            Wp = GM.powm_spd(M @ M.T, 0.25)
            return np.linalg.solve(Wp, M), Wp
        R = GM.powm_spd(M @ M.T, -0.25)
        G = R @ GM.expm_sym(s * z_of(rid)) @ R
    Wp = GM.powm_spd(G, -0.5)
    return np.linalg.solve(Wp, M), Wp


def virtual_run(run, plant, rid, Gref=None):
    f = run["feats"]
    Pp, Wp = planted_head(f["P"], f["W"], plant, rid, Gref)
    return {"meta": {"diverged": False, "finite": True, "plant": plant},
            "feats": {"y": np.asarray(f["y"], np.float64), "P": Pp, "W": Wp}}


def gbar_of_plant(runs_rids, plant, Gref):
    """(Gbar', Gbar_bal') of the planted heads of a list of (run, rid)."""
    Gs, Gb = [], []
    for run, rid in runs_rids:
        Pp, Wp = planted_head(run["feats"]["P"], run["feats"]["W"], plant, rid, Gref)
        Gs.append(GM.norm_metric(Wp))
        Gb.append(GM.det_normalize(GM.Gbal_of(Pp, Wp)))
    return GM.log_euclid_mean(Gs), GM.log_euclid_mean(Gb)


def plant_job(args):
    """(lroot, cohort, plant, tag, ra, rb, Gref, Gbar, Gbar_bal) -> R_cf deltas of the planted pair (cached)."""
    import json
    lroot, cohort, plant, tag, ra, rb, Gref, Gbar, Gbal = args
    path = os.path.join(lroot, plant, tag, f"{ra}__{rb}.json")
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    A, B = CM.load_run(cohort, ra), CM.load_run(cohort, rb)
    Gref = None if Gref is None else np.array(Gref)
    VA, VB = virtual_run(A, plant, ra, Gref), virtual_run(B, plant, rb, Gref)
    XA, XB = CM.pair_features(VA, VB, tag, None if Gbar is None else np.array(Gbar),
                              None if Gbal is None else np.array(Gbal))
    row = CM.reduced_ladder(C.geom(XA, "none"), C.geom(XB, "none"))
    res = {"deltas": {k: v for k, v in C.deltas(row).items() if k in CM.R_CF}, "A": ra, "B": rb, "tag": tag,
           "plant": plant}
    CM.jdump(res, path)
    return res
