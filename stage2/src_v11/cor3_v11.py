"""S2.2 v1.1 K-block (Corollary cor:gauge-neighbors, prospectively; PREREG_S2_2_v1.1.md section 10.4).

Copied from v1.0 review cor3_A_gaugenet.py (sqdist, knn, overlap_counts, set_changed, certificate_check,
gen_eig_ratio) and cor3_graded.py (graded bound), extended with the imbalance-only exchange (u_A vs u_A^bal,
kappa = D_A) and the Lemma N1 / K3 checks. Per pair, both directions, k in {1, 10, 30}:
  pure exchange      u_A = W_A^{-1} y_A  vs  W_B^{-1} y_A, kappa = kappa_AB = kappa(G_A, G_B), G = (W W^T)^{-1};
  imbalance-only     u_A                 vs  u_A^bal = N_A^{-1/4} y_A, kappa = D_A = kappa(G_A, G_A^bal);
  K1 zero violations among certified anchors; K2 graded bound >= observed slot change (per anchor and per pair);
  K3 kappa_AB / kappa^bal_AB in [1/(D_A D_B), D_A D_B]; K4(a) anchors with gap ratio in [kappa^{1/2}, kappa)
  that keep their set; K6 real cross-run u_A vs u_B against the function-only comparison in one gauge.
"""
from __future__ import annotations

import os
import time
from multiprocessing import get_context

import numpy as np
import scipy.linalg as sla

import common_v11 as CM
import gaugemath_v11 as GM

KS = (1, 10, 30)
KMAX = 32


def sqdist(X):
    """Exact squared Euclidean distances by coordinate differences (no Gram cancellation) (v1.0 copy)."""
    X = np.asarray(X, np.float64)
    m = X.shape[0]
    D = np.zeros((m, m))
    for c in range(X.shape[1]):
        diff = X[:, c][:, None] - X[:, c][None, :]
        D += diff * diff
    return D


def knn(D, K=KMAX):
    """First K neighbours (self excluded) sorted by (D_ij, j) (v1.0 copy)."""
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    part = np.argpartition(Dm, K, axis=1)[:, :K + 1]
    vals = np.take_along_axis(Dm, part, 1)
    order = np.lexsort((part, vals), axis=1)
    idx = np.take_along_axis(part, order, 1)
    vals = np.take_along_axis(vals, order, 1)
    ties = int(np.sum(vals[:, 1:] == vals[:, :-1]))
    return idx[:, :K], vals[:, :K], ties


def overlap_counts(iA, iB, k):
    return (iA[:, :k, None] == iB[:, None, :k]).any(2).sum(1)


def set_changed(iA, iB, k):
    return ~np.all(np.sort(iA[:, :k], 1) == np.sort(iB[:, :k], 1), axis=1)


def gen_eig_ratio(GA, GB):
    w = sla.eigh(GB, GA, eigvals_only=True)
    return float(w.max() / w.min()), float(w.min()), float(w.max())


def exchange_stats(DR, ref, cf, kappa, k):
    """Certificate (v1.0 certificate_check) + graded bound (v1.0 cor3_graded) + K4(a) for one direction."""
    iR, vR = ref
    iC = cf[0]
    m = iR.shape[0]
    cert = (vR[:, k - 1] > 0) & (vR[:, k] > kappa * vR[:, k - 1])
    ch = set_changed(iR, iC, k)
    ov = overlap_counts(iR, iC, k)
    dk, dk1 = vR[:, k - 1], vR[:, k]
    inner = np.sum(vR[:, :k] >= (dk1 / kappa)[:, None], axis=1)
    outer = np.sum(DR <= (kappa * dk)[:, None], axis=1) - 1 - k
    b = np.minimum(np.minimum(inner, outer), k)
    sw = k - ov
    gap = vR[:, k] / np.maximum(vR[:, k - 1], 1e-300)
    band = (gap >= np.sqrt(kappa)) & (gap < kappa)
    return {"n_cert": int(cert.sum()), "cert_frac": float(cert.mean()), "changed_frac": float(ch.mean()),
            "violations": int(np.sum(cert & ch)), "obs_dN": float(sw.sum() / (m * k)), "bound_dN": float(b.sum() / (m * k)),
            "anchor_bound_violations": int(np.sum(sw > b)), "graded_cert_frac": float(np.mean(b == 0)),
            "k4a_band_n": int(band.sum()), "k4a_band_kept": int(np.sum(band & ~ch))}


def pair_job(args):
    cohort, cell, pid, ra, rb = args
    A, B = CM.load_run(cohort, ra), CM.load_run(cohort, rb)
    out = {"cell": cell, "pair": pid, "A": ra, "B": rb}
    if not (CM.run_ok(A) and CM.run_ok(B)):
        out["failed"] = True
        return out
    for r in (A, B):   # resolution R-NUM: a numerically degenerate head fails the pair (counted)
        P, W = np.asarray(r["feats"]["P"], np.float64), np.asarray(r["feats"]["W"], np.float64)
        if not (GM.spd_ok(GM.G_of(W)) and GM.spd_ok(GM.N_of(P, W))):
            out.update(failed=True, numerical_failure=True)
            return out
    try:
        return _pair_job(out, A, B)
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as e:
        out.update(failed=True, numerical_failure=repr(e)[:300])
        return out


def _pair_job(out, A, B):
    yA, yB = (np.asarray(r["feats"]["y"], np.float64) for r in (A, B))
    WA, WB = (np.asarray(r["feats"]["W"], np.float64) for r in (A, B))
    PA, PB = (np.asarray(r["feats"]["P"], np.float64) for r in (A, B))
    WAi, WBi = np.linalg.inv(WA), np.linalg.inv(WB)
    GA, GB = WAi.T @ WAi, WBi.T @ WBi
    GbA, GbB = GM.Gbal_of(PA, WA), GM.Gbal_of(PB, WB)
    kap, lo, hi = gen_eig_ratio(GA, GB)
    kap_bal = gen_eig_ratio(GbA, GbB)[0]
    DA_, DB_ = gen_eig_ratio(GA, GbA)[0], gen_eig_ratio(GB, GbB)[0]
    reps = {"uA": yA @ WAi.T, "uAp": yA @ WBi.T, "uB": yB @ WBi.T, "uBp": yB @ WAi.T,
            "uAbal": GM.rep_bal(yA, PA, WA), "uBbal": GM.rep_bal(yB, PB, WB)}
    D = {k: sqdist(v) for k, v in reps.items()}
    NN = {}
    ties = {}
    for k, v in D.items():
        i_, v_, t_ = knn(v)
        NN[k] = (i_, v_)
        ties[k] = t_
    out.update({"kappa": kap, "gen_eig_lo": lo, "gen_eig_hi": hi, "kappa_bal": kap_bal, "D_A": DA_, "D_B": DB_,
                "K3_lo": 1.0 / (DA_ * DB_), "K3_hi": DA_ * DB_,
                "K3_ok": bool(1.0 / (DA_ * DB_) * (1 - 1e-9) <= kap / kap_bal <= DA_ * DB_ * (1 + 1e-9)),
                "ties": ties})
    m = D["uA"].shape[0]
    for k in KS:
        out[f"pureA_k{k}"] = exchange_stats(D["uA"], NN["uA"], NN["uAp"], kap, k)
        out[f"pureB_k{k}"] = exchange_stats(D["uB"], NN["uB"], NN["uBp"], kap, k)
        out[f"imbA_k{k}"] = exchange_stats(D["uA"], NN["uA"], NN["uAbal"], DA_, k)
        out[f"imbB_k{k}"] = exchange_stats(D["uB"], NN["uB"], NN["uBbal"], DB_, k)
        ov = lambda a, b: overlap_counts(NN[a][0], NN[b][0], k)
        dn = lambda o: float(1.0 - o.sum() / (m * k))
        o_uu, o_fB, o_fA = ov("uA", "uB"), ov("uAp", "uB"), ov("uA", "uBp")
        out[f"cross_k{k}"] = {"dN_uu": dn(o_uu), "dN_fun_inBgauge": dn(o_fB), "dN_fun_inAgauge": dn(o_fA),
                              "excess_uu_minus_fun_mean": dn(o_uu) - 0.5 * (dn(o_fB) + dn(o_fA))}
    return out


def run_pairs(cohort, jobs, n_jobs, label=""):
    t = time.time()
    res = []
    with get_context("fork").Pool(n_jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(pair_job, jobs, chunksize=2)):
            res.append(r)
            if (i + 1) % 100 == 0:
                print(f"  K-block {label}: {i + 1}/{len(jobs)} ({time.time() - t:.0f} s)", flush=True)
    res.sort(key=lambda r: (r["cell"], str(r["pair"])))
    return res


def summarize_cell(R):
    ok = [r for r in R if not r.get("failed")]
    o = {"n_pairs": len(R), "n_ok": len(ok)}
    for k in KS:
        for ex in ("pure", "imb"):
            rows = [r[f"{ex}A_k{k}"] for r in ok] + [r[f"{ex}B_k{k}"] for r in ok]
            if not rows:
                o[f"{ex}_k{k}"] = {"violations_total": 0, "certified_anchors_total": 0, "cert_frac_mean": float("nan"),
                                   "changed_frac_mean": float("nan"), "obs_dN_mean": float("nan"), "obs_dN_ci95": [float("nan")] * 2,
                                   "bound_dN_mean": float("nan"), "anchor_bound_violations_total": 0, "pairs_bound_ge_obs": 0,
                                   "n_dir": 0, "bound_over_obs_median": None, "k4a_band_n": 0, "k4a_band_kept": 0}
                continue
            ratios = [x["bound_dN"] / x["obs_dN"] for x in rows if x["obs_dN"] > 0]
            o[f"{ex}_k{k}"] = {
                "violations_total": int(sum(x["violations"] for x in rows)),
                "certified_anchors_total": int(sum(x["n_cert"] for x in rows)),
                "cert_frac_mean": float(np.mean([x["cert_frac"] for x in rows])),
                "changed_frac_mean": float(np.mean([x["changed_frac"] for x in rows])),
                "obs_dN_mean": float(np.mean([x["obs_dN"] for x in rows])),
                "obs_dN_ci95": CM.boot_mean_ci([x["obs_dN"] for x in rows], f"K|{ex}|{k}|{len(rows)}")["ci"],
                "bound_dN_mean": float(np.mean([x["bound_dN"] for x in rows])),
                "anchor_bound_violations_total": int(sum(x["anchor_bound_violations"] for x in rows)),
                "pairs_bound_ge_obs": int(sum(x["bound_dN"] >= x["obs_dN"] - 1e-15 for x in rows)), "n_dir": len(rows),
                "bound_over_obs_median": float(np.median(ratios)) if ratios else None,
                "k4a_band_n": int(sum(x["k4a_band_n"] for x in rows)), "k4a_band_kept": int(sum(x["k4a_band_kept"] for x in rows))}
        o[f"cross_k{k}_excess_mean"] = float(np.mean([r[f"cross_k{k}"]["excess_uu_minus_fun_mean"] for r in ok])) if ok else None
    if ok:
        o["kappa_median"] = float(np.median([r["kappa"] for r in ok]))
        o["kappa_bal_median"] = float(np.median([r["kappa_bal"] for r in ok]))
        o["D_median"] = float(np.median([r["D_A"] for r in ok] + [r["D_B"] for r in ok]))
        o["K3_violations"] = int(sum(not r["K3_ok"] for r in ok))
        o["kappa_over_kappa_bal_median"] = float(np.median([r["kappa"] / r["kappa_bal"] for r in ok]))
        o["exact_ties_total"] = int(sum(sum(r["ties"].values()) for r in ok))
    return o
