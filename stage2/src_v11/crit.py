"""Shared Stage-2 criterion definitions (PREREG_stage2.md §2.4 / §3.3): run loading, pair
ladders via the frozen hrhv2 harness, and the discrepancy map Delta_Q used by gates.py and analyze.py.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
sys.path.insert(0, os.path.expanduser("~/hrh_v2_stage1_20261008"))
from hrhv2.prep import Geom  # noqa: E402
from hrhv2.ladder import ladder  # noqa: E402
from hrhv2 import structural as st  # noqa: E402

S_IDX = np.load(os.path.join(ROOT, "data", "S_idx.npy"))
ITEM_IDS = [f"part5:{i}" for i in S_IDX]
BERN_BETA = 5.0  # 1 / tau_A


def run_dir(cohort, run_id):
    return os.path.join(ROOT, "runs", cohort, run_id)


def load_run(cohort, run_id):
    d = run_dir(cohort, run_id)
    if not os.path.exists(os.path.join(d, ".done")):
        return None
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    feats = dict(np.load(os.path.join(d, "feats.npz")))
    return {"meta": meta, "feats": feats}


def run_ok(run, rep):
    if run is None:
        return False
    X = run["feats"].get(rep)
    return bool(not run["meta"].get("diverged", False) and X is not None and np.all(np.isfinite(X)))


def geom(X, pre):
    X = np.asarray(X, dtype=np.float64)
    return Geom.from_features(X, pre)


def bern_geom(G: Geom, beta=BERN_BETA):
    """Bernstein warp exp(beta(K-1)) of a unit-norm (l2) Gram."""
    K = np.exp(beta * (G.K - 1.0))
    np.fill_diagonal(K, 1.0)
    return Geom.from_gram(K, pre="bern")


def pair_ladder(GA: Geom, GB: Geom, n_jobs=2):
    return ladder(GA, GB, item_ids=ITEM_IDS, n_jobs=n_jobs)


def deltas(row: dict) -> dict:
    """Delta_Q (discrepancies, smaller = more agreement) from one hrhv2 ladder row."""
    g = lambda k: row.get(k, float("nan"))
    out = {}
    for k in (1, 3, 10, 30, 100):
        out[f"dN_k{k}"] = 1.0 - g(f"s_N_k{k}")
    out["dN_kq"] = 1.0 - g("s_N_kq")
    for k in (1, 3, 10):
        out[f"dC_k{k}"] = g(f"comp_rand_k{k}")
    out["dO"] = g("Delta_O")
    out["dE"] = g("Delta_E")
    out["dMST"] = 1.0 - g("s_T")
    out["dMST_forest"] = 1.0 - g("s_T_forest")
    out["dMST_bridge"] = 1.0 - g("s_T_bridge")
    out["Gs"] = g("Gs_rms")
    out["Gs_max"] = g("Gs_max")
    out["dAff"] = 1.0 - g("rsa_pearson")
    for t in ("1", "0p1", "0p01"):
        out[f"dSOG_t{t}"] = 1.0 - g(f"sog_knn10_tau{t}")
        out[f"dSOGmst_t{t}"] = 1.0 - g(f"sog_mst_tau{t}")
    out["dCKA"] = 1.0 - g("cka_biased")
    out["dCKAd"] = 1.0 - g("cka_debiased")
    out["rawG"] = g("G_rms") if row.get("raw_G_applicable") else float("nan")
    nuA, nuB = g("nu_A"), g("nu_B")
    out["dIso"] = 0.5 * (g("quot_AB_iso_rms_cf") / nuB + g("quot_BA_iso_rms_cf") / nuA)
    om = []
    for d in ("AB", "BA"):
        ia, aa = g(f"quot_{d}_iso_expl_cf"), g(f"quot_{d}_affine_expl_cf")
        om.append(1.0 - (1.0 - ia) / (1.0 - aa) if (1.0 - aa) > 0 else float("nan"))
    out["omega"] = float(np.mean(om))
    out["omega_insample"] = float(np.mean([
        1.0 - (1.0 - g(f"quot_{d}_iso_expl")) / (1.0 - g(f"quot_{d}_affine_expl")) for d in ("AB", "BA")]))
    out["rowiso_share_cf"] = g("warpshare_expl_cf_row_extra")
    out["dPi"] = g("gl_projector_dist")
    return out


# Delta names that carry a tolerance decision
TOL_KEYS = ["dN_k1", "dN_k3", "dN_k10", "dN_k30", "dN_k100", "dN_kq", "dC_k1", "dC_k3", "dC_k10", "dO", "dE",
            "dMST", "Gs", "Gs_max", "dAff", "dSOG_t1", "dSOG_t0p1", "dSOG_t0p01", "dSOGmst_t0p1", "dCKA", "dCKAd",
            "rawG", "dIso", "dPi"]
FAIL = float("inf")  # Delta of a pair containing a diverged / non-finite run


def cp_bounds(x: int, n: int, alpha=0.05):
    from scipy.stats import beta
    lo = beta.ppf(alpha, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(1 - alpha, x + 1, n - x) if x < n else 1.0
    return float(lo), float(hi)


def decide(x: int, n: int):
    lo, hi = cp_bounds(x, n)
    if n == 0:
        return "none", lo, hi
    if lo > 0.9:
        return "reliable", lo, hi
    if hi < 0.8:
        return "unreliable", lo, hi
    return "inconclusive", lo, hi
