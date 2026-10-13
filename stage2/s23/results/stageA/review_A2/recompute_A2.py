#!/usr/bin/env python3
"""s23-review-A2: independent recompute of S2.3 Stage A after A2 (completion), A3 (memo fix), A4 (P5b zero MC).

Imports NO s23 / hrhv2 / crit / A1-A4 code: numpy, scipy (stats.beta/binom, optimize.minimize) and the
standard library only. Every metric is re-implemented from its PREREG v1.0 / SPEC definition:
  D_ij = K_ii + K_jj - 2K_ij clipped at 0 (this evaluation order); tie rule (D_ij, j) for neighbours and
  (D_ij, min, max) for the MST (Kruskal here, Prim in the frozen code; a strict edge order gives one tree);
  dN_k10 = 1 - s_N(10); dO = mean per-anchor Kendall distance (one-sided ties 1/2) by sign matrices;
  dEk = global Kendall distance (one-sided ties 1/2) by a block inversion count of my own; dMST = 1 - s_T;
  Gs / Gs_max = rms / max of D_A/nu_A - D_B/nu_B (nu = mean off-diagonal D); dCKA = 1 - biased CKA of H K H;
  iso_inf (A->B) = 1/2 max_g (prefix max - suffix min) over x-tie groups; dIso = cross-fitted isotonic rms
  (hash halves 'hrhv2-halves|i', own weighted PAVA); T_E on S_ex8 = equality of the 28x28 sign matrices;
  F_mu = unit-diagonal normalization of the Frobenius projection of exp(K/t)/Z onto {K psd, diag <= mu}
  by its dual (own objective, scipy L-BFGS-B).
Statistics (CP, Holm, block bootstrap of Pi, median-ratio bootstrap, TOST) are re-implemented with review-own
keyed generators (keys 's23-review-A2|...'), 20,000 resamples. Inputs read: runs/*.npy and *.json (cert_pass,
psd_active), results/prestep/frozen_predictions.json (frozen constants), prestep_arrays.npz (K*, K*^(t)).
The frozen outputs (stageA_results.json, results/stageA/cache/*.json, cache/F/*.npy) are read only AFTER the
recompute, for comparison. CPU only, at most 4 worker processes with one BLAS thread each.

    python recompute_A2.py [--workers 4]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import math
import sys
import time
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta, binom

HOME = os.path.expanduser("~")
S23 = os.path.join(HOME, "hrh_v2_stage2_20261008", "s23")
RUNS = os.path.join(S23, "runs")
OUT = os.path.join(S23, "results", "stageA", "review_A2")
FDIR = os.path.join(OUT, "F_review")
FR = json.load(open(os.path.join(S23, "results", "prestep", "frozen_predictions.json")))
T = float(FR["cfg"]["t"])
ZP = float(FR["instance"]["Z"])
S8 = np.array(FR["probes"]["S_ex8"])
GRID = list(FR["grid"]["grid"])
L0, LTOP = int(FR["grid"]["log2N0"]), int(FR["grid"]["log2N_top"])
EPS = FR["eps"]
TC = FR["cfg"]["tests"]
CONF = list(FR["p5_rungs"]["confirmatory"])
ORDER = ["dN_k10", "dO", "dEk", "dMST"]
M = 256
IU = np.triu_indices(M, 1)
IU_I, IU_J = IU[0].astype(np.int64), IU[1].astype(np.int64)
IU8 = np.triu_indices(8, 1)
NB_RUNS = 192
NBOOT = 20000
PMIN = 1.0 / NBOOT
H1 = np.array([(int.from_bytes(hashlib.sha256(f"hrhv2-halves|{i}".encode()).digest()[:8], "little") & 1) == 0
               for i in range(M)])


# ============================================================================ runs
def rid(arm, L, l2, rep):
    return f"{arm}_{L}_l{l2}_r{rep:03d}"


def rbase(arm, L, l2, rep):
    return os.path.join(RUNS, arm, f"l{l2}", L, rid(arm, L, l2, rep))


def run_info(arm, L, l2, rep):
    b = rbase(arm, L, l2, rep)
    ok_files = os.path.exists(b + ".npy") and os.path.exists(b + ".done") and os.path.exists(b + ".json")
    j = json.load(open(b + ".json")) if os.path.exists(b + ".json") else {}
    return {"failed": not (ok_files and bool(j.get("cert_pass", False))), "psd_active": bool(j.get("psd_active", False)),
            "proj_active": bool(j.get("proj_active", False))}


def load(spec):
    arm, L, l2, rep, tr = spec
    K = np.load(rbase(arm, L, l2, rep) + ".npy").astype(np.float64)
    if tr is None:
        return K
    if tr == "phi":
        P = np.exp((K - 1.0) / T)
        np.fill_diagonal(P, 1.0)
        return P
    if tr == "half":
        P = 0.5 * K + 0.5
        np.fill_diagonal(P, 1.0)
        return P
    if tr in ("F1", "F4"):
        return np.load(os.path.join(FDIR, f"{rid(arm, L, l2, rep)}__{tr}.npy"))
    raise ValueError(tr)


# ============================================================================ metrics (own implementations)
def dist(K):
    dg = np.ascontiguousarray(np.diag(K), dtype=np.float64)
    D = (dg[:, None] + dg[None, :]) - 2.0 * K
    np.maximum(D, 0.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def knn_idx(D, k):
    Dc = D.copy()
    np.fill_diagonal(Dc, np.inf)
    return np.argsort(Dc, axis=1, kind="stable")[:, :k]


def s_N(nA, nB):
    k = nA.shape[1]
    return float((nA[:, :, None] == nB[:, None, :]).any(axis=2).sum()) / (nA.shape[0] * k)


def d_O(DA, DB):
    m = DA.shape[0]
    off = ~np.eye(m, dtype=bool)
    XA = DA[off].reshape(m, m - 1)
    XB = DB[off].reshape(m, m - 1)
    per = np.empty(m)
    for lo in range(0, m, 32):
        a, b = XA[lo:lo + 32], XB[lo:lo + 32]
        sa = np.sign(a[:, :, None] - a[:, None, :])
        sb = np.sign(b[:, :, None] - b[:, None, :])
        per[lo:lo + 32] = np.abs(sa - sb).sum(axis=(1, 2)) / 2.0 / ((m - 1) * (m - 2))
    return float(per.mean())


def tie_pairs(v):
    _, c = np.unique(v, return_counts=True)
    c = c.astype(np.int64)
    return int((c * (c - 1) // 2).sum())


def kendall_own(x, y):
    """(tot, strictly discordant, x-tie pairs, y-tie pairs, joint-tie pairs) by a block inversion count."""
    n = x.size
    tot = n * (n - 1) // 2
    o = np.lexsort((y, x))
    ys = y[o]
    dis = 0
    prev = np.empty(0)
    B = 2048
    for lo in range(0, n, B):
        c = ys[lo:lo + B]
        if prev.size:
            dis += int(prev.size * c.size - np.searchsorted(prev, c, side="right").sum())
        dis += int(np.triu(c[:, None] > c[None, :], 1).sum())
        prev = np.sort(np.concatenate([prev, c]))
    xt, yt = tie_pairs(x), tie_pairs(y)
    _, cj = np.unique(np.stack([x, y], 1), axis=0, return_counts=True)
    cj = cj.astype(np.int64)
    nt = int((cj * (cj - 1) // 2).sum())
    return tot, dis, xt, yt, nt


def dEk_own(ua, ub):
    tot, dis, xt, yt, nt = kendall_own(ua, ub)
    return (dis + 0.5 * ((xt - nt) + (yt - nt))) / tot, xt, yt


def mst_codes(D):
    w = D[IU]
    o = np.lexsort((IU_J, IU_I, w))
    parent = list(range(M))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = IU_I[o].tolist(), IU_J[o].tolist()
    codes = set()
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            codes.add(a * M + b)
            if len(codes) == M - 1:
                break
    return codes


def center(K):
    rm = K.mean(axis=1)
    cm = K.mean(axis=0)
    return K - rm[:, None] - cm[None, :] + rm.mean()


def cka(KA, KB):
    A, Bm = center(KA), center(KB)
    return float(np.vdot(A, Bm) / math.sqrt(float(np.vdot(A, A)) * float(np.vdot(Bm, Bm))))


def gs_pair(ua, ub):
    d = ua / ua.mean() - ub / ub.mean()
    return float(math.sqrt(float(np.mean(d * d)))), float(np.max(np.abs(d)))


def iso_inf(ua, ub):
    y = ub / ub.mean()
    o = np.lexsort((y, ua))
    xs, ys = ua[o], y[o]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    gmax = np.maximum.reduceat(ys, starts)
    gmin = np.minimum.reduceat(ys, starts)
    pref = np.maximum.accumulate(gmax)
    suf = np.minimum.accumulate(gmin[::-1])[::-1]
    return 0.5 * max(0.0, float(np.max(pref - suf)))


def pava(y, w):
    vals, wts, cnts = [], [], []
    for yi, wi in zip(y.tolist(), w.tolist()):
        vals.append(yi); wts.append(wi); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            wn = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / wn); wts.append(wn); cnts.append(c1 + c2)
    return np.repeat(np.array(vals), cnts)


def upper_masked(D, mask):
    idx = np.flatnonzero(mask)
    sub = D[np.ix_(idx, idx)]
    return sub[np.triu_indices(idx.size, 1)]


def iso_cf_rms(x1, y1, x2, y2):
    ux, inv, cnt = np.unique(x1, return_inverse=True, return_counts=True)
    ym = np.bincount(inv, weights=y1) / cnt
    g = pava(ym, cnt.astype(np.float64))
    r = y2 - np.interp(x2, ux, g)
    return float(math.sqrt(float(np.mean(r * r))))


def d_iso(DA, DB, ua, ub):
    A1, B1, A2, B2 = upper_masked(DA, H1), upper_masked(DB, H1), upper_masked(DA, ~H1), upper_masked(DB, ~H1)
    return 0.5 * (iso_cf_rms(A1, B1, A2, B2) / ub.mean() + iso_cf_rms(B1, A1, B2, A2) / ua.mean())


def te8(DA, DB):
    a = DA[np.ix_(S8, S8)][IU8]
    b = DB[np.ix_(S8, S8)][IU8]
    return bool(np.array_equal(np.sign(a[:, None] - a[None, :]), np.sign(b[:, None] - b[None, :])))


def omega_self(K):
    u = dist(K)[IU]
    G = 2.0 * (1.0 - np.exp(-u / (2.0 * T)))
    return float(np.max(np.abs(u / u.mean() - G / G.mean())))


def inv_link_cf(KA, KB):
    ub = KB[IU]
    if np.any(ub <= 0):
        return float("inf"), float(np.mean(ub <= 0))
    DB = -2.0 * T * np.log(np.where(KB > 0, KB, 1.0))
    np.fill_diagonal(DB, 0.0)
    DB = 0.5 * (DB + DB.T)
    return gs_pair(dist(KA)[IU], DB[IU])[1], 0.0


def pair_metrics(KA, KB, level, extra=()):
    DA, DB = dist(KA), dist(KB)
    ua, ub = DA[IU], DB[IU]
    out = {"maxdiff": float(np.max(np.abs(KA - KB)))}
    nA, nB = knn_idx(DA, 10), knn_idx(DB, 10)
    out["s_N_k10"] = s_N(nA, nB)
    out["dN_k10"] = 1.0 - out["s_N_k10"]
    out["TE8"] = te8(DA, DB)
    out["Gs"], out["Gs_max"] = gs_pair(ua, ub)
    if level == "light":
        return out
    out["dO"] = d_O(DA, DB)
    out["dEk"], out["ties_A"], out["ties_B"] = dEk_own(ua, ub)
    TA, TB = mst_codes(DA), mst_codes(DB)
    out["dMST"] = 1.0 - len(TA & TB) / (M - 1)
    out["dCKA"] = 1.0 - cka(KA, KB)
    out["iso_inf_AB"] = iso_inf(ua, ub)
    if "dIso" in extra:
        out["dIso"] = d_iso(DA, DB, ua, ub)
    if "cf" in extra:
        out["c_invlink"], out["frac_KB_nonpos"] = inv_link_cf(KA, KB)
    return out


# ============================================================================ plants: own projection
def _psd_parts(M):
    ev, V = np.linalg.eigh(M)
    lp = np.maximum(ev, 0.0)
    return ev, V, lp, (V * lp) @ V.T


def _omega(ev):
    lp = np.maximum(ev, 0.0)
    dl = ev[:, None] - ev[None, :]
    num = lp[:, None] - lp[None, :]
    same = np.abs(dl) <= 1e-14 * max(1.0, float(np.max(np.abs(ev))))
    with np.errstate(divide="ignore", invalid="ignore"):
        Om = np.where(same, np.where((ev[:, None] > 0) & (ev[None, :] > 0), 1.0, 0.0), num / np.where(same, 1.0, dl))
    return Om


def ball_projection(W, mu):
    """argmin ||K - W||_F over {K psd, diag K <= mu}. Dual: min_y>=0 1/2||P+(W - Diag y)||^2 + mu sum y.
    Stage 1 L-BFGS-B on the dual; stage 2 semismooth Newton (generalized Jacobian of P+, CG) on the active set
    with an active-set check of the KKT conditions."""
    n = W.shape[0]

    def fg(y):
        ev, V, lp, Kp = _psd_parts(W - np.diag(y))
        return 0.5 * float(np.dot(lp, lp)) + mu * float(y.sum()), mu - np.diag(Kp)

    y0 = np.maximum(np.diag(W) - mu, 0.0)
    r = minimize(fg, y0, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * n,
                 options={"maxiter": 3000, "ftol": 1e-16, "gtol": 1e-12, "maxcor": 30})
    y = np.array(r.x, float)
    ev, V, lp, K = _psd_parts(W - np.diag(y))
    A = (y > 1e-10) | (np.diag(K) > mu - 1e-10)
    newton_it = 0
    for outer in range(20):
        for it in range(60):
            ev, V, lp, K = _psd_parts(W - np.diag(y))
            G = mu - np.diag(K)
            G[~A] = 0.0
            g0 = float(np.max(np.abs(G)))
            if g0 <= 1e-14:
                break
            Om = _omega(ev)
            idx = np.flatnonzero(A)

            def Jv(h):
                hf = np.zeros(n); hf[idx] = h
                X = V.T @ (hf[:, None] * V)
                return (((V @ (Om * X)) * V).sum(axis=1))[idx] + 1e-13 * h

            b = -G[idx]
            d = np.zeros(idx.size); res = b.copy(); pdir = res.copy(); rs = float(res @ res)
            for _ in range(300):
                Ap_ = Jv(pdir)
                alpha = rs / float(pdir @ Ap_)
                d += alpha * pdir; res -= alpha * Ap_
                rs_new = float(res @ res)
                if math.sqrt(rs_new) <= 1e-15 + 1e-12 * math.sqrt(float(b @ b)):
                    break
                pdir = res + (rs_new / rs) * pdir; rs = rs_new
            step = 1.0
            while step > 1e-6:
                yt = y.copy(); yt[idx] += step * d
                Gt = mu - np.diag(_psd_parts(W - np.diag(yt))[3]); Gt[~A] = 0.0
                if float(np.max(np.abs(Gt))) < g0:
                    y = yt
                    break
                step *= 0.5
            newton_it += 1
            if step <= 1e-6:
                break
        ev, V, lp, K = _psd_parts(W - np.diag(y))
        dg = np.diag(K)
        drop = A & (y < 0)
        add = (~A) & (dg > mu + 1e-13)
        if not drop.any() and not add.any():
            break
        y[drop] = 0.0
        A = (A & ~drop) | add
    K = 0.5 * (K + K.T)
    dg = np.diag(K)
    primal = 0.5 * float(np.sum((K - W) ** 2))
    dual = -(0.5 * float(np.dot(lp, lp)) + mu * float(y.sum())) + 0.5 * float(np.sum(W * W))
    return K, {"lbfgs_nit": int(r.nit), "newton_it": newton_it, "feas": float(max(0.0, np.max(dg - mu))),
               "min_y": float(np.min(y)), "comp": float(np.max(np.abs(y * (dg - mu)))), "gap": primal - dual,
               "lam_min_K": float(np.min(np.linalg.eigvalsh(K))), "n_active": int(np.sum(A))}


def task_F(spec_mu):
    (arm, L, l2, rep), mu = spec_mu
    K = load((arm, L, l2, rep, None))
    W = np.exp(K / T) / ZP
    Kp, cert = ball_projection(W, float(mu))
    d = np.sqrt(np.diag(Kp))
    Fn = Kp / d[:, None] / d[None, :]
    np.fill_diagonal(Fn, 1.0)
    np.save(os.path.join(FDIR, f"{rid(arm, L, l2, rep)}__F{mu}.npy"), Fn)
    return rid(arm, L, l2, rep), mu, cert


# ============================================================================ pair tasks
def task_pair(t):
    name, j, sa, sb, level, extra = t
    KA, KB = load(sa), load(sb)
    out = pair_metrics(KA, KB, level, extra)
    if name.endswith("link_self"):
        out["omega_self_A"] = omega_self(KA)
    return name, j, out


def task_err(t):
    arm, L, l2, rep = t
    K = load((arm, L, l2, rep, None))
    Kinf = KSTAR if L == "A" else KBT
    return (arm, L, l2, rep), float(np.max(np.abs(K - Kinf)))


def _init():
    global KSTAR, KBT
    arr = np.load(os.path.join(S23, "results", "prestep", "prestep_arrays.npz"))
    KSTAR, KBT = arr["Kstar"].astype(np.float64), arr["KBt"].astype(np.float64)


def build_tasks():
    tasks = []
    A = lambda l2, i, tr=None: ("0", "A", l2, i, tr)
    Bq = lambda l2, i, tr=None: ("0", "B", l2, i, tr)
    Ap = lambda l2, i, tr=None: ("Rm", "Ap", l2, i, tr)
    for l2 in GRID:
        ex = ("dIso",) if l2 == 20 else (("cf",) if l2 in (L0, LTOP) else ())
        for j in range(96):
            tasks.append((f"l{l2}_wA", j, A(l2, 2 * j), A(l2, 2 * j + 1), "full", ex if l2 == 20 else ()))
            tasks.append((f"l{l2}_wB", j, Bq(l2, 2 * j), Bq(l2, 2 * j + 1), "full", ex if l2 == 20 else ()))
            tasks.append((f"l{l2}_AB", j, A(l2, 2 * j), Bq(l2, 2 * j), "full", ex))
            tasks.append((f"l{l2}_link_self", j, A(l2, 2 * j), A(l2, 2 * j, "phi"), "light", ()))
    for l2 in (L0, LTOP):
        for j in range(96):
            tasks.append((f"l{l2}_link_peer", j, A(l2, 2 * j), A(l2, 2 * j + 1, "phi"), "full", ()))
            tasks.append((f"l{l2}_w_phi", j, A(l2, 2 * j, "phi"), A(l2, 2 * j + 1, "phi"), "full", ()))
            tasks.append((f"l{l2}_rescale_self", j, A(l2, 2 * j), A(l2, 2 * j, "half"), "full", ()))
    for mu in (1, 4):
        for j in range(96):
            tasks.append((f"l{LTOP}_plant{mu}", j, A(LTOP, 2 * j), A(LTOP, 2 * j + 1, f"F{mu}"), "full", ()))
            tasks.append((f"l{LTOP}_w_F{mu}", j, A(LTOP, 2 * j, f"F{mu}"), A(LTOP, 2 * j + 1, f"F{mu}"), "full", ()))
    for l2 in (20, 22):
        for j in range(96):
            tasks.append((f"rm_l{l2}_cross", j, A(l2, 2 * j), Ap(l2, 2 * j), "full", ()))
            tasks.append((f"rm_l{l2}_wAp", j, Ap(l2, 2 * j), Ap(l2, 2 * j + 1), "full", ()))
    return tasks


# ============================================================================ statistics (own)
def rng(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f"s23-review-A2|{key}".encode()).digest()[:8], "little"))


def cp(x, n, a=0.05):
    lo = float(beta.ppf(a, x, n - x + 1)) if x > 0 else 0.0
    hi = float(beta.ppf(1 - a, x + 1, n - x)) if x < n else 1.0
    return lo, hi


def decide(vals, failed, eps, predict):
    v = np.array(vals, float)
    bad = np.array(failed, bool) | ~np.isfinite(v)
    v[bad] = np.inf if predict == "reliable" else -np.inf
    n = v.size
    x = int(np.sum(v <= eps))
    lo, hi = cp(x, n)
    dec = "reliable" if lo > 0.9 else ("unreliable" if hi < 0.8 else "inconclusive")
    p = float(binom.sf(x - 1, n, 0.9)) if predict == "reliable" else float(binom.cdf(x, n, 0.8))
    return {"x": x, "n": n, "cp_lo": lo, "cp_hi": hi, "decision": dec, "p": p, "holds": dec == predict,
            "n_failed": int(bad.sum()), "eps": float(eps)}


def pi_test(ab, aa, bb, fab, faa, fbb, key, predict):
    def adj(v, f, worst_high):
        v = np.array(v, float)
        bad = np.array(f, bool) | ~np.isfinite(v)
        ok = v[~bad]
        if bad.any():
            v[bad] = (ok.max() if worst_high else ok.min()) if ok.size else (np.inf if worst_high else 0.0)
        return v
    hi_ = predict == "no_plateau"
    ab, aa, bb = adj(ab, fab, hi_), adj(aa, faa, not hi_), adj(bb, fbb, not hi_)
    nb = ab.size
    idx = rng(key).integers(0, nb, size=(NBOOT, nb))
    pis = ab[idx].mean(1) - (aa[idx].mean(1) + bb[idx].mean(1))
    mab, maa, mbb = float(ab.mean()), float(aa.mean()), float(bb.mean())
    lo, hi = (float(q) for q in np.quantile(pis, [0.025, 0.975]))
    allzero = mab == 0.0 and maa == 0.0 and mbb == 0.0
    den = 0.5 * (maa + mbb)
    r = {"mean_AB": mab, "mean_AA": maa, "mean_BB": mbb, "Pi": mab - maa - mbb, "ci_lo": lo, "ci_hi": hi,
         "plateau": bool(lo > 0), "no_plateau": bool(hi < 0 or allzero), "rho": mab / den if den > 0 else float("nan"),
         "p_noplateau": PMIN if allzero else max(PMIN, float(np.mean(pis >= 0))),
         "p_plateau": max(PMIN, float(np.mean(pis <= 0))), "n_failed": int(np.sum(fab) + np.sum(faa) + np.sum(fbb)),
         "n_blocks": nb, "predict": predict}
    r["p"] = r["p_noplateau"] if predict == "no_plateau" else r["p_plateau"]
    r["holds"] = r["no_plateau"] if predict == "no_plateau" else r["plateau"]
    return r


def ratio_test(e_hi, f_hi, e_lo, f_lo, key):
    e_hi = np.array(e_hi, float); e_lo = np.array(e_lo, float)
    e_hi[np.array(f_hi, bool) | ~np.isfinite(e_hi)] = np.inf
    e_lo[np.array(f_lo, bool) | ~np.isfinite(e_lo)] = 0.0
    g = rng(key)
    ih = g.integers(0, e_hi.size, size=(NBOOT, e_hi.size))
    il = g.integers(0, e_lo.size, size=(NBOOT, e_lo.size))
    num, den = np.median(e_hi[ih], axis=1), np.median(e_lo[il], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.where(den > 0, num / den, np.inf)
    lo, hi = (float(q) for q in np.quantile(rr, [0.025, 0.975]))
    return {"ratio": float(np.median(e_hi) / np.median(e_lo)), "ci_lo": lo, "ci_hi": hi, "holds": bool(hi < TC["conv"]),
            "plateau": bool(lo > TC["plat"]), "p": max(PMIN, float(np.mean(rr >= TC["conv"])))}


def mean_boot(v, key):
    v = np.array(v, float)
    idx = rng(key).integers(0, v.size, size=(NBOOT, v.size))
    return v[idx].mean(1)


def holm(pv, alpha=0.05):
    items = sorted(pv.items(), key=lambda kv: kv[1])
    out, stop = {}, False
    for i, (k, p) in enumerate(items):
        thr = alpha / (len(items) - i)
        rej = (not stop) and p <= thr
        stop = stop or not rej
        out[k] = {"p": p, "thr": thr, "rejected": rej}
    return out


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    nw = min(4, max(1, a.workers))
    os.makedirs(FDIR, exist_ok=True)
    t0 = time.time()
    _init()
    ctx = get_context("fork")
    info = {}
    for arm, L, l2s in (("0", "A", GRID), ("0", "B", GRID), ("Rm", "Ap", [20, 22])):
        for l2 in l2s:
            info[(arm, L, l2)] = [run_info(arm, L, l2, r) for r in range(NB_RUNS)]
    fails = {f"{k[0]}_{k[1]}_l{k[2]}": int(sum(x["failed"] for x in v)) for k, v in info.items()}
    print("certificate failures per cell:", fails, flush=True)

    # ---- phase 1: plant projections F_mu for every arm-0 A run at N_top
    fspecs = [(("0", "A", LTOP, r), mu) for mu in (1, 4) for r in range(NB_RUNS)]
    fspecs = [s for s in fspecs if not os.path.exists(os.path.join(FDIR, f"{rid(*s[0])}__F{s[1]}.npy"))]
    fcert = []
    if fspecs:
        with ctx.Pool(nw, initializer=_init) as pool:
            for k, res in enumerate(pool.imap_unordered(task_F, fspecs, chunksize=2)):
                fcert.append(res)
                if k % 64 == 0:
                    print(f"F {k}/{len(fspecs)} {time.time() - t0:.0f}s {res}", flush=True)
    json.dump([{"id": r, "mu": m, **c} for r, m, c in fcert], open(os.path.join(OUT, "F_review_certs.json"), "w"), indent=0)

    # ---- phase 2: pairs and recovery errors
    tasks = build_tasks()
    rows = {}
    with ctx.Pool(nw, initializer=_init) as pool:
        for k, (name, j, out) in enumerate(pool.imap_unordered(task_pair, tasks, chunksize=4)):
            rows.setdefault(name, {})[j] = out
            if k % 500 == 0:
                print(f"pairs {k}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
        errs = {}
        et = [("0", L, l2, r) for L in ("A", "B") for l2 in GRID for r in range(NB_RUNS)]
        for key, e in pool.imap_unordered(task_err, et, chunksize=16):
            errs[key] = e
    rows = {n: [d[j] for j in range(96)] for n, d in rows.items()}
    json.dump(rows, open(os.path.join(OUT, "pair_rows_review.json"), "w"))
    print(f"computed {len(tasks)} pairs in {time.time() - t0:.0f}s", flush=True)

    R = {"version": "s23-review-A2 recompute v1", "certificate_failures": fails}

    def V(n, k):
        return [r_[k] for r_ in rows[n]]

    def TEv(n):
        return [0.0 if r_["TE8"] else 1.0 for r_ in rows[n]]

    def pfail(n, la, lb):
        # failed flag of each pair from its two runs: la/lb = (arm, L, l2, offset) with run index 2j+offset
        out = []
        for j in range(96):
            fa = info[la[:3]][2 * j + la[3]]["failed"]
            fb = info[lb[:3]][2 * j + lb[3]]["failed"]
            out.append(fa or fb)
        return out

    FA = lambda l2: pfail(None, ("0", "A", l2, 0), ("0", "A", l2, 1))
    FB = lambda l2: pfail(None, ("0", "B", l2, 0), ("0", "B", l2, 1))
    FX = lambda l2: pfail(None, ("0", "A", l2, 0), ("0", "B", l2, 0))
    FAA = lambda l2: pfail(None, ("0", "A", l2, 0), ("0", "A", l2, 1))   # link_peer / w_phi / plants use A runs only
    FS = lambda l2: pfail(None, ("0", "A", l2, 0), ("0", "A", l2, 0))
    # ---- population constants re-derived from the frozen arrays
    Kd = dist(KSTAR)[IU]; Gd = dist(KBT)[IU]
    R["population_check"] = {
        "KBt_equals_exp_link": float(np.max(np.abs(np.where(np.eye(M, dtype=bool), 1.0, np.exp((KSTAR - 1) / T)) - KBT))),
        "omega_psi": gs_pair(Kd, Gd)[1], "eps_geo_frozen": EPS["eps_geo"],
        "one_minus_cka_inf": 1 - cka(KSTAR, KBT), "frozen": EPS["one_minus_cka_inf"],
        "Gs_inf": gs_pair(Kd, Gd)[0], "eps_cka_frozen": EPS["eps_cka"]}

    # ---- F0
    F0 = {}
    eA = {l2: [errs[("0", "A", l2, r)] for r in range(NB_RUNS)] for l2 in GRID}
    eB = {l2: [errs[("0", "B", l2, r)] for r in range(NB_RUNS)] for l2 in GRID}
    fA = {l2: [x["failed"] for x in info[("0", "A", l2)]] for l2 in GRID}
    fB = {l2: [x["failed"] for x in info[("0", "B", l2)]] for l2 in GRID}
    F0["P1_A"] = ratio_test(eA[LTOP], fA[LTOP], eA[L0], fA[L0], "P1_A")
    F0["P1_B"] = ratio_test(eB[LTOP], fB[LTOP], eB[L0], fB[L0], "P1_B")
    gte = {L: decide(TEv(f"l{LTOP}_w{L}"), (FA if L == "A" else FB)(LTOP), 0.0, "reliable") for L in ("A", "B")}
    gte_ok = all(v["holds"] for v in gte.values())
    ggeo = {}
    for l2 in (L0, LTOP):
        g = {}
        for L in ("A", "B"):
            ff = (FA if L == "A" else FB)(l2)
            g[f"Gs_max_{L}"] = decide(V(f"l{l2}_w{L}", "Gs_max"), ff, EPS["eps_geo"], "reliable")
            g[f"dCKA_{L}"] = decide(V(f"l{l2}_w{L}", "dCKA"), ff, EPS["eps_cka"], "reliable")
        g["holds"] = all(v["holds"] for v in g.values())
        ggeo[l2] = g
    F0["P2"] = decide(TEv(f"l{LTOP}_AB"), FX(LTOP), 0.0, "reliable")
    p3a = [decide(V(f"l{l2}_AB", "Gs_max"), FX(l2), EPS["eps_geo"], "unreliable") for l2 in (L0, LTOP)]
    F0["P3a"] = {"N0": p3a[0], "Ntop": p3a[1], "p": max(p3a[0]["p"], p3a[1]["p"]), "holds": p3a[0]["holds"] and p3a[1]["holds"]}
    band = (EPS["one_minus_cka_inf"] - TC["cka_band"], EPS["one_minus_cka_inf"] + TC["cka_band"])
    cab = np.array(V(f"l{LTOP}_AB", "dCKA"), float); cab[np.array(FX(LTOP))] = band[1] + 1.0
    mb = mean_boot(cab, "P3b_tost")
    tost_p = max(max(PMIN, float(np.mean(mb <= band[0]))), max(PMIN, float(np.mean(mb >= band[1]))))
    wa = np.array(V(f"l{LTOP}_wA", "dCKA"), float); wa[np.array(FA(LTOP))] = 1.0
    wb = np.array(V(f"l{LTOP}_wB", "dCKA"), float); wb[np.array(FB(LTOP))] = 1.0
    pa = max(PMIN, float(np.mean(mean_boot(wa, "P3b_wA") >= TC["within_cka"])))
    pb = max(PMIN, float(np.mean(mean_boot(wb, "P3b_wB") >= TC["within_cka"])))
    pP3b = max(tost_p, pa, pb)
    F0["P3b"] = {"cross_mean": float(cab.mean()), "band": band, "ci90_cross": [float(np.quantile(mb, 0.05)), float(np.quantile(mb, 0.95))],
                 "within_A_mean": float(wa.mean()), "within_B_mean": float(wb.mean()), "p_tost": tost_p, "p_wA": pa, "p_wB": pb,
                 "p": pP3b, "holds": pP3b < TC["alpha"]}
    c0, ct = V(f"l{L0}_AB", "c_invlink"), V(f"l{LTOP}_AB", "c_invlink")
    F0["P3c"] = ratio_test(ct, FX(LTOP), c0, FX(L0), "P3c")
    F0["P3c"]["frac_KB_nonpos"] = {"N0": float(np.mean(V(f"l{L0}_AB", "frac_KB_nonpos"))), "Ntop": float(np.mean(V(f"l{LTOP}_AB", "frac_KB_nonpos")))}
    F0["P3c"]["median_c"] = {"N0": float(np.median(c0)), "Ntop": float(np.median(ct))}
    F0["P4a"] = decide(V(f"l{LTOP}_AB", "iso_inf_AB"), FX(LTOP), EPS["eps_geo"], "reliable")
    pis = {r_: pi_test(V(f"l{LTOP}_AB", r_), V(f"l{LTOP}_wA", r_), V(f"l{LTOP}_wB", r_), FX(LTOP), FA(LTOP), FB(LTOP),
                       f"P5|{r_}", "no_plateau") for r_ in ORDER}
    falsified = [r_ for r_ in ORDER if pis[r_]["plateau"]]
    F0["P5_order"] = {"rungs": pis, "p": max(pis[r_]["p"] for r_ in CONF),
                      "holds": bool(CONF and all(pis[r_]["no_plateau"] for r_ in CONF) and not falsified), "falsified_on": falsified}
    pg = {r_: pi_test(V(f"l{LTOP}_AB", r_), V(f"l{LTOP}_wA", r_), V(f"l{LTOP}_wB", r_), FX(LTOP), FA(LTOP), FB(LTOP),
                      f"P5g|{r_}", "plateau") for r_ in ("Gs", "dCKA")}
    F0["P5_geom"] = {"rungs": pg, "p": max(p["p"] for p in pg.values()), "holds": all(p["holds"] for p in pg.values())}
    guard_of = {"P2": gte_ok, "P3a": ggeo[L0]["holds"] and ggeo[LTOP]["holds"], "P4a": ggeo[LTOP]["holds"]}
    pv = {k: (v["p"] if guard_of.get(k, True) else 1.0) for k, v in F0.items()}
    H = holm(pv)
    for k in F0:
        F0[k]["guard_ok"] = guard_of.get(k, True)
        F0[k]["holm"] = H[k]
        F0[k]["confirmed"] = bool(F0[k]["holds"] and H[k]["rejected"] and guard_of.get(k, True))
    R["F0"] = F0
    R["guards"] = {"TE_Ntop": gte, "TE_ok": gte_ok, "geo": {str(k): v for k, v in ggeo.items()}}

    # ---- calibration and plants
    cal = {}
    for l2 in GRID:
        rr = rows[f"l{l2}_link_self"]
        x = sum(1 for r_ in rr if r_["TE8"])
        om = max(abs(r_["Gs_max"] - r_["omega_self_A"]) for r_ in rr)
        cal[f"link_self_l{l2}"] = {"x": x, "n": 96, "omega_err_max": om, "ok": x == 96 and om <= 1e-9}
    for l2 in (L0, LTOP):
        lp, wphi, wA_ = f"l{l2}_link_peer", f"l{l2}_w_phi", f"l{l2}_wA"
        fl = FAA(l2)
        p2l = decide(TEv(lp), fl, 0.0, "reliable")
        p3l = decide(V(lp, "Gs_max"), fl, EPS["eps_geo"], "unreliable")
        p4l = decide(V(lp, "iso_inf_AB"), fl, EPS["eps_geo"], "reliable")
        p5l = {r_: pi_test(V(lp, r_), V(wA_, r_), V(wphi, r_), fl, FA(l2), fl, f"cal|lp|{l2}|{r_}", "no_plateau") for r_ in ORDER}
        p5g = {r_: pi_test(V(lp, r_), V(wA_, r_), V(wphi, r_), fl, FA(l2), fl, f"cal|lpg|{l2}|{r_}", "plateau") for r_ in ("Gs", "dCKA")}
        ok = bool(p3l["holds"] and all(p5l[r_]["no_plateau"] for r_ in CONF) and not any(v["plateau"] for v in p5l.values())
                  and all(v["plateau"] for v in p5g.values()) and p2l["holds"] and p4l["holds"])
        cal[f"link_peer_l{l2}"] = {"P2": p2l, "P3a": p3l, "P4a": p4l, "P5_order": p5l, "P5_geom": p5g, "ok_raw": ok,
                                    "ok": ok if l2 == LTOP else True, "judged": l2 == LTOP}
        rs = rows[f"l{l2}_rescale_self"]
        worst = max(max(abs(r_["Gs_max"]), abs(r_["Gs"]), abs(r_["dCKA"]), max(abs(r_[k]) for k in ORDER)) for r_ in rs)
        cal[f"rescale_self_l{l2}"] = {"max_abs": worst, "ok": worst <= 1e-9}
    plants = {}
    for mu in (1, 4):
        cr, wF, wA_ = f"l{LTOP}_plant{mu}", f"l{LTOP}_w_F{mu}", f"l{LTOP}_wA"
        fl = FAA(LTOP)
        pp = {r_: pi_test(V(cr, r_), V(wA_, r_), V(wF, r_), fl, FA(LTOP), fl, f"plant{mu}|{r_}", "plateau") for r_ in ORDER}
        wte = decide(TEv(wF), fl, 0.0, "unreliable")
        broken = bool(any(p["plateau"] for p in pp.values()) or wte["decision"] == "unreliable")
        lad = FR["poplim"][f"i_mu{mu}"]["ladder"]
        dinf = {r_: float(lad[r_]) for r_ in ORDER}
        expect = "broken" if any(v >= TC["dep_hi"] for v in dinf.values()) else ("intact" if all(v <= TC["dep_lo"] for v in dinf.values()) else "none")
        match = expect == "none" or (expect == "broken" and broken) or (expect == "intact" and not broken)
        plants[f"plant_break_mu{mu}"] = {"pi": pp, "withinF_TE": wte, "broken": broken, "expected": expect, "match": match,
                                         "delta_inf": dinf}
    cal_ok = all(v["ok"] for v in cal.values()) and all(v["match"] for v in plants.values())
    R["calibration"], R["plants"], R["calibration_ok"] = cal, plants, cal_ok

    # ---- verdict
    lw, lw_ok = {}, True
    for L in ("A", "B"):
        for r_ in ORDER:
            m0 = float(np.median(V(f"l{L0}_w{L}", r_))); mt = float(np.median(V(f"l{LTOP}_w{L}", r_)))
            good = (m0 == 0.0 and mt == 0.0) or (m0 > 0 and mt / m0 < TC["conv"])
            lw[f"{L}_{r_}"] = {"median_N0": m0, "median_Ntop": mt, "ratio": mt / m0 if m0 > 0 else float("nan"), "ok": good}
            lw_ok = lw_ok and good
    l_cross = bool(CONF and all(pis[r_]["no_plateau"] for r_ in CONF) and not falsified)
    if lw_ok and gte_ok and l_cross:
        verdict = "PRESENT-order"
    elif lw_ok and gte_ok and falsified:
        verdict = "BROKEN-cross " + ",".join(falsified)
    elif not (lw_ok and gte_ok):
        verdict = "BROKEN-within" + (" + BROKEN-cross " + ",".join(falsified) if falsified else "")
    else:
        verdict = "INCONCLUSIVE"
    R["verdict"] = {"verdict": verdict, "L_cross": l_cross, "L_within": lw, "L_within_ok": lw_ok, "TE_guard": gte_ok}

    # ---- stop rule
    reasons = [f"F0 {k} not confirmed" for k, v in F0.items() if not v["confirmed"]]
    if not gte_ok:
        reasons.append("S_ex8 within T_E guard failed")
    if verdict != "PRESENT-order":
        reasons.append(f"order verdict {verdict}")
    if not cal_ok:
        reasons.append("calibration / planted-control mismatch")
    R["stop_rule"] = {"arm0_pattern_present": not reasons, "reasons": reasons,
                      "action": "report to the author; Stage B needs a separate approval" if not reasons else "STOP"}

    # ---- N_match and P7
    tab = {str(l2): float(np.mean(V(f"l{l2}_wA", "s_N_k10"))) for l2 in GRID}
    best = min(tab, key=lambda k: (abs(tab[k] - TC["nmatch_target"]), int(k)))
    lm = int(best)
    p7 = {"table_within_A_sN10": tab, "log2N_match": lm, "Pi": {}}
    for r_ in ORDER + ["dIso"]:
        p7["Pi"][r_] = pi_test(V(f"l{lm}_AB", r_), V(f"l{lm}_wA", r_), V(f"l{lm}_wB", r_), FX(lm), FA(lm), FB(lm), f"P7|{r_}", "no_plateau")
    if any(p7["Pi"][r_]["plateau"] for r_ in ORDER):
        p7["rule"] = "plateau at matched noise: sample-size qualifier required"
    elif all(p7["Pi"][r_]["ci_hi"] < 0 for r_ in ORDER):
        p7["rule"] = "inside the model at matched noise, cross-objective order disagreement does not exceed the within-objective level"
    else:
        p7["rule"] = "neither clause applies"
    p7["within_B_sN10"] = float(np.mean(V(f"l{lm}_wB", "s_N_k10")))
    p7["cross_sN10"] = float(np.mean(V(f"l{lm}_AB", "s_N_k10")))
    R["P7"] = p7

    # ---- P8 (R- small N)
    p8 = {}
    for l2 in (20, 22):
        fx = pfail(None, ("0", "A", l2, 0), ("Rm", "Ap", l2, 0))
        fp = pfail(None, ("Rm", "Ap", l2, 0), ("Rm", "Ap", l2, 1))
        e = {"Pi": {}}
        for r_ in ORDER + ["Gs", "dCKA"]:
            e["Pi"][r_] = pi_test(V(f"rm_l{l2}_cross", r_), V(f"l{l2}_wA", r_), V(f"rm_l{l2}_wAp", r_), fx, FA(l2), fp, f"P8|{l2}|{r_}", "no_plateau")
        e["frac_psd_active_A"] = float(np.mean([x["psd_active"] for x in info[("0", "A", l2)]]))
        e["frac_psd_active_Ap"] = float(np.mean([x["psd_active"] for x in info[("Rm", "Ap", l2)]]))
        e["n_failed_Ap"] = int(sum(x["failed"] for x in info[("Rm", "Ap", l2)]))
        e["failed_Ap_reps"] = [r for r, x in enumerate(info[("Rm", "Ap", l2)]) if x["failed"]]
        p8[str(l2)] = e
    R["P8"] = p8

    # ---- descriptive per N
    desc = {}
    for l2 in GRID:
        d = {s_: {k: float(np.mean(V(f"l{l2}_{n_}", k))) for k in ORDER + ["Gs", "Gs_max", "dCKA", "s_N_k10", "iso_inf_AB"]}
             for s_, n_ in (("AB", "AB"), ("AA", "wA"), ("BB", "wB"))}
        d["TE8_agree"] = {s_: int(sum(r_["TE8"] for r_ in rows[f"l{l2}_{n_}"])) for s_, n_ in (("AB", "AB"), ("AA", "wA"), ("BB", "wB"))}
        d["median_eA"], d["median_eB"] = float(np.median(eA[l2])), float(np.median(eB[l2]))
        d["frac_psd_active_A"] = float(np.mean([x["psd_active"] for x in info[("0", "A", l2)]]))
        d["frac_proj_active_B"] = float(np.mean([x["proj_active"] for x in info[("0", "B", l2)]]))
        d["n_failed_A"], d["n_failed_B"] = int(sum(fA[l2])), int(sum(fB[l2]))
        if l2 == 20:
            d["dIso"] = {s_: float(np.mean(V(f"l20_{n_}", "dIso"))) for s_, n_ in (("AB", "AB"), ("AA", "wA"), ("BB", "wB"))}
        desc[str(l2)] = d
    R["descriptive"] = desc
    R["runtime_s"] = time.time() - t0

    # ============================================================== comparison with the frozen outputs (read only now)
    fz = json.load(open(os.path.join(S23, "results", "stageA", "stageA_results.json")))
    cmpv = {}
    for k in F0:
        a_, b_ = F0[k], fz["F0"][k]
        cmpv[k] = {"confirmed_review": a_["confirmed"], "confirmed_frozen": b_["confirmed"], "holds_review": a_["holds"],
                   "holds_frozen": b_["holds"]}
        for f in ("ratio", "ci_lo", "ci_hi", "x", "n"):
            if f in a_ and f in b_:
                cmpv[k][f] = [a_[f], b_[f]]
    for r_ in ORDER:
        a_, b_ = F0["P5_order"]["rungs"][r_], fz["F0"]["P5_order"]["rungs"][r_]
        cmpv[f"P5_order.{r_}"] = {f: [a_[f], b_[f]] for f in ("mean_AB", "mean_AA", "mean_BB", "Pi", "ci_lo", "ci_hi", "rho")}
    for r_ in ("Gs", "dCKA"):
        a_, b_ = F0["P5_geom"]["rungs"][r_], fz["F0"]["P5_geom"]["rungs"][r_]
        cmpv[f"P5_geom.{r_}"] = {f: [a_[f], b_[f]] for f in ("mean_AB", "mean_AA", "mean_BB", "Pi", "ci_lo", "ci_hi", "rho")}
    cmpv["P3a"].update({"x_N0": [F0["P3a"]["N0"]["x"], fz["F0"]["P3a"]["N0"]["x"]], "x_Ntop": [F0["P3a"]["Ntop"]["x"], fz["F0"]["P3a"]["Ntop"]["x"]]})
    cmpv["P3b"].update({"cross_mean": [F0["P3b"]["cross_mean"], fz["F0"]["P3b"]["tost"]["mean"]],
                        "within_A": [F0["P3b"]["within_A_mean"], fz["F0"]["P3b"]["within_A"]["mean"]],
                        "within_B": [F0["P3b"]["within_B_mean"], fz["F0"]["P3b"]["within_B"]["mean"]]})
    cmpv["verdict"] = [verdict, fz["verdict"]["verdict"]]
    cmpv["stop_rule"] = [R["stop_rule"]["action"], fz["stop_rule"]["action"]]
    cmpv["calibration_ok"] = [cal_ok, fz["calibration_ok"]]
    cmpv["plants"] = {k: [plants[k]["broken"], fz["plants"][k]["broken"]] for k in plants}
    cmpv["N_match"] = [lm, fz["N_match"]["log2N_match"]]
    cmpv["N_match_table"] = {k: [tab[k], fz["N_match"]["table"][k]["mean_sN10_withinA"]] for k in tab}
    cmpv["P7_rule"] = [p7["rule"], fz["secondary"]["P7"]["rule"]]
    for r_ in ORDER + ["dIso"]:
        a_, b_ = p7["Pi"][r_], fz["secondary"]["P7"]["Pi"][r_]
        cmpv[f"P7.{r_}"] = {f: [a_[f], b_[f]] for f in ("mean_AB", "mean_AA", "mean_BB", "Pi", "ci_lo", "ci_hi", "rho")}
    for l2 in ("20", "22"):
        for r_ in ORDER + ["Gs", "dCKA"]:
            a_, b_ = p8[l2]["Pi"][r_], fz["secondary"]["P8"][l2]["Pi"][r_]
            cmpv[f"P8.{l2}.{r_}"] = {f: [a_[f], b_[f]] for f in ("mean_AB", "mean_AA", "mean_BB", "Pi", "ci_lo", "ci_hi", "rho", "n_failed")}
        cmpv[f"P8.{l2}.psd"] = [[p8[l2]["frac_psd_active_A"], p8[l2]["frac_psd_active_Ap"]],
                                [fz["secondary"]["P8"][l2]["frac_psd_active_A"], fz["secondary"]["P8"][l2]["frac_psd_active_Ap"]]]
    cmpv["L_within"] = {k: [lw[k]["ratio"], fz["verdict"]["L_within"][k]["ratio"]] for k in lw}
    R["compare_frozen"] = cmpv

    # per-pair comparison with the frozen pair-row cache
    def fnum(v):
        if isinstance(v, str):
            return float(v.replace("Infinity", "inf"))
        return float(v)
    cache = os.path.join(S23, "results", "stageA", "cache")
    mapping = {f"l{l2}_{w}": f"arm0_l{l2}_{w}_n96.json" for l2 in GRID for w in ("wA", "wB", "AB", "link_self")}
    mapping.update({f"l{l2}_{w}": f"arm0_l{l2}_{w}_n96.json" for l2 in (L0, LTOP) for w in ("link_peer", "w_phi", "rescale_self")})
    mapping.update({f"l{LTOP}_plant1": f"arm0_l{LTOP}_plant1_n96.json", f"l{LTOP}_w_F1": f"arm0_l{LTOP}_w_F1_n96.json",
                    f"l{LTOP}_plant4": f"arm0_l{LTOP}_plant4_n96.json", f"l{LTOP}_w_F4": f"arm0_l{LTOP}_w_F4_n96.json"})
    mapping.update({f"rm_l{l2}_{w}": f"rm_l{l2}_{w}_n96.json" for l2 in (20, 22) for w in ("cross", "wAp")})
    pc = {}
    for mine, fn in mapping.items():
        p = os.path.join(cache, fn)
        if not os.path.exists(p) or mine not in rows:
            pc[mine] = {"missing": fn}
            continue
        fro = json.load(open(p))
        ent = {"n": len(fro)}
        keys = ["s_N_k10", "Gs_max"] + ([] if mine.endswith("link_self") else ORDER + ["Gs", "dCKA", "iso_inf_AB"])
        if mine.startswith("l20_"):
            keys = keys + (["dIso"] if not mine.endswith("link_self") else [])
        for k in keys:
            diffs = []
            for a_, b_ in zip(rows[mine], fro):
                fk = "dN_k10" if k == "dN_k10" else k
                bv = fnum(b_["d"].get(fk, "nan"))
                av = a_[k]
                diffs.append(0.0 if (av == bv or (math.isnan(av) and math.isnan(bv))) else abs(av - bv))
            ent[k] = {"max_abs_diff": float(np.max(diffs)), "n_exact": int(sum(1 for x in diffs if x == 0.0))}
        ent["TE8_agree_mismatch"] = int(sum(1 for a_, b_ in zip(rows[mine], fro) if bool(a_["TE8"]) != bool(b_["ex8"].get("T_E"))))
        pc[mine] = ent
    R["compare_pairs"] = pc
    # F projections vs the frozen cache/F
    fd = []
    for mu in (1, 4):
        for r in range(NB_RUNS):
            a_ = np.load(os.path.join(FDIR, f"{rid('0', 'A', LTOP, r)}__F{mu}.npy"))
            b_ = np.load(os.path.join(cache, "F", f"{rid('0', 'A', LTOP, r)}__F{mu}.npy"))
            fd.append(float(np.max(np.abs(a_ - b_))))
    R["compare_F"] = {"max_abs_diff": float(np.max(fd)), "median": float(np.median(fd))}
    json.dump(R, open(os.path.join(OUT, "recompute_A2.json"), "w"), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print(json.dumps({"verdict": verdict, "stop": R["stop_rule"], "N_match": lm, "P7": p7["rule"],
                      "F0": {k: [v["holds"], v["confirmed"], v["p"]] for k, v in F0.items()}, "cal_ok": cal_ok,
                      "compare_F": R["compare_F"], "runtime_s": R["runtime_s"]}, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
