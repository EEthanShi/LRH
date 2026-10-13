"""SPEC section 1: structural observables (exact-equality discrepancies).

All functions take squared-distance matrices D (m, m) float64 with zero
diagonal and apply the global tie rule:

* neighbours of i: candidates j != i ordered by (D_ij, j);
* MST / Prim: edges ordered by (D_ij, min(i,j), max(i,j)).

Under these keys every set/graph/tree quantity is a deterministic function of
D, so two implementations of the SPEC must agree exactly.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata

from . import util

try:  # exact O(n log n) discordant-pair count (Cython, used by scipy.kendalltau)
    from scipy.stats._stats import _kendall_dis as _scipy_kendall_dis
except Exception:  # pragma: no cover
    _scipy_kendall_dis = None


# ----------------------------------------------------------------------------
# kNN sets
# ----------------------------------------------------------------------------

def knn_table(D: np.ndarray, kmax: int, *, chunk: int = 512):
    """First ``kmax`` neighbours of every item under the tie rule (D_ij, j).

    Returns (idx, val): int64 (m, kmax) neighbour indices sorted by the key,
    and float64 (m, kmax) the corresponding D values. Exact under ties: among
    candidates whose distance equals the kmax-th value, the smallest indices
    are taken.
    """
    m = D.shape[0]
    K = int(min(kmax, m - 1))
    if K < 1:
        raise ValueError("kmax must be >= 1 and m >= 2")
    idx_out = np.empty((m, K), dtype=np.int64)
    val_out = np.empty((m, K), dtype=np.float64)
    for lo in range(0, m, chunk):
        hi = min(m, lo + chunk)
        n = hi - lo
        Dc = np.array(D[lo:hi], dtype=np.float64, copy=True)
        r = np.arange(n)
        Dc[r, lo + r] = np.inf
        v = np.partition(Dc, K - 1, axis=1)[:, K - 1]
        lt = Dc < v[:, None]
        eq = Dc == v[:, None]
        need = K - lt.sum(axis=1)
        sel = lt | (eq & (np.cumsum(eq, axis=1) <= need[:, None]))
        cols = np.nonzero(sel)[1].reshape(n, K)          # row-major -> ascending j
        vals = np.take_along_axis(Dc, cols, axis=1)
        order = np.lexsort((cols, vals), axis=-1)         # primary vals, then j
        idx_out[lo:hi] = np.take_along_axis(cols, order, axis=1)
        val_out[lo:hi] = np.take_along_axis(vals, order, axis=1)
    return idx_out, val_out


def overlap_counts(nA: np.ndarray, nB: np.ndarray, k: int) -> np.ndarray:
    """Per-anchor |N_k^A(i) & N_k^B(i)| (int64, length m)."""
    m = nA.shape[0]
    a = nA[:, :k]
    b = nB[:, :k]
    out = np.empty(m, dtype=np.int64)
    step = max(1, min(m, int(2e7 // max(k * k, 1))))  # bound the (step,k,k) temp
    for lo in range(0, m, step):
        hi = min(m, lo + step)
        out[lo:hi] = (a[lo:hi, :, None] == b[lo:hi, None, :]).any(axis=2).sum(axis=1)
    return out


def knn_overlap(nA: np.ndarray, nB: np.ndarray, k: int) -> float:
    """s_N(k) = (1/(m k)) sum_i |N_k^A(i) & N_k^B(i)|."""
    c = overlap_counts(nA, nB, k)
    return float(c.sum()) / (nA.shape[0] * k)


def default_k_grid(m: int) -> list[int]:
    """{1, 3, 10, 30, 100} plus k_q = max(1, round(0.01 (m-1))), restricted to k < m."""
    kq = max(1, int(round(0.01 * (m - 1))))
    ks = sorted({k for k in (1, 3, 10, 30, 100, kq) if k <= m - 1})
    return ks


def k_q(m: int) -> int:
    return max(1, int(round(0.01 * (m - 1))))


def near_tie_fraction(vals: np.ndarray, k: int, median_D: float) -> float:
    """Fraction of anchors whose k-th and (k+1)-th distances differ by
    < 1e-12 * median(D). ``vals`` from knn_table with kmax >= k+1."""
    if vals.shape[1] < k + 1:
        return float("nan")
    gap = vals[:, k] - vals[:, k - 1]
    return float(np.mean(gap < 1e-12 * median_D))


# ----------------------------------------------------------------------------
# kNN-graph components
# ----------------------------------------------------------------------------

def knn_components(nbrs: np.ndarray, k: int):
    """Components of the undirected kNN graph ({i,j} iff j in N_k(i) or i in N_k(j)).
    Returns (n_components, labels)."""
    m = nbrs.shape[0]
    rows = np.repeat(np.arange(m), k)
    cols = nbrs[:, :k].ravel()
    G = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(m, m)).tocsr()
    ncomp, labels = connected_components(G, directed=True, connection="weak")
    return int(ncomp), labels


def _pairs_same(counts: np.ndarray) -> int:
    c = counts.astype(object)
    return int(sum(int(x) * (int(x) - 1) // 2 for x in c if x > 1))


def rand_distance(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Fraction of unordered pairs whose same-component indicator differs
    (exact integer arithmetic)."""
    m = lab1.size
    tot = m * (m - 1) // 2
    sameA = _pairs_same(np.bincount(lab1))
    sameB = _pairs_same(np.bincount(lab2))
    joint = lab1.astype(np.int64) * (int(lab2.max()) + 1) + lab2.astype(np.int64)
    sameAB = _pairs_same(np.unique(joint, return_counts=True)[1])
    return (sameA + sameB - 2 * sameAB) / tot


# ----------------------------------------------------------------------------
# Kendall distance (ties counted 1/2)
# ----------------------------------------------------------------------------

def _tie_pairs(dense_ranks: np.ndarray) -> int:
    cnt = np.bincount(dense_ranks).astype(np.int64)
    cnt = cnt[cnt > 1]
    return int((cnt * (cnt - 1) // 2).sum())


def kendall_counts(x: np.ndarray, y: np.ndarray) -> dict:
    """Exact pair counts between two orderings (same algorithm as
    scipy.stats.kendalltau): tot, dis (strictly discordant), xtie, ytie
    (pairs tied in x / in y, joint ties included) and ntie (joint ties)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    tot = n * (n - 1) // 2
    perm = np.argsort(y, kind="mergesort")
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)
    perm = np.argsort(x, kind="mergesort")
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)
    if _scipy_kendall_dis is not None:
        dis = int(_scipy_kendall_dis(x, y))
    else:  # pragma: no cover - O(n^2) fallback, only for tiny inputs
        # within equal-x runs y is ascending, so strict inversions are exactly
        # the strictly discordant pairs
        dis = int(sum(int(np.sum(y[i + 1:] < y[i])) for i in range(n)))
    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype(np.int64)
    ntie = int((cnt * (cnt - 1) // 2).sum())
    return {"tot": tot, "dis": dis, "xtie": _tie_pairs(x), "ytie": _tie_pairs(y), "ntie": ntie}


def kendall_distance_from_counts(c: dict) -> float:
    """(discordant + 1/2 * one-sided ties) / total pairs. Joint ties count 0."""
    one_sided = (c["xtie"] - c["ntie"]) + (c["ytie"] - c["ntie"])
    return (c["dis"] + 0.5 * one_sided) / c["tot"]


def tau_b_from_counts(c: dict) -> float:
    """Kendall tau_b from exact integer pair counts. The denominator is formed
    in integer arithmetic, so identical orderings give exactly 1.0 (and
    Delta_E = (1 - tau_b)/2 exactly 0)."""
    tot, xt, yt, nt, dis = (int(c[k]) for k in ("tot", "xtie", "ytie", "ntie", "dis"))
    if xt == tot or yt == tot:
        return float("nan")
    cmd = tot - xt - yt + nt - 2 * dis
    a, b = tot - xt, tot - yt
    if a == b:
        return float(cmd / a) if abs(cmd) < 2 ** 53 else float(Fraction(cmd, a))
    return float(cmd / math.sqrt(a * b))


def kendall_distance(x: np.ndarray, y: np.ndarray) -> float:
    return kendall_distance_from_counts(kendall_counts(x, y))


def _row_kendall_worker(lo: int, hi: int):
    sh = util.shared()
    DA, DB = sh["DA"], sh["DB"]
    m = DA.shape[0]
    out = np.empty(hi - lo, dtype=np.float64)
    nt = np.zeros(hi - lo, dtype=np.int64)
    for r, i in enumerate(range(lo, hi)):
        x = np.concatenate((DA[i, :i], DA[i, i + 1:]))
        y = np.concatenate((DB[i, :i], DB[i, i + 1:]))
        c = kendall_counts(x, y)
        out[r] = kendall_distance_from_counts(c)
        nt[r] = c["xtie"] + c["ytie"]
    return out, nt


def row_kendall(DA: np.ndarray, DB: np.ndarray, *, n_jobs: int | None = None):
    """Per-anchor Kendall distance between rows of D_A and D_B (j != i).
    Returns (per_anchor array, number of rows containing any tie)."""
    m = DA.shape[0]
    res = util.map_rows(_row_kendall_worker, m, n_jobs=n_jobs, share={"DA": DA, "DB": DB})
    per = np.concatenate([r[0] for r in res])
    ties = np.concatenate([r[1] for r in res])
    return per, int(np.sum(ties > 0))


# ----------------------------------------------------------------------------
# Global edge order
# ----------------------------------------------------------------------------

GLOBAL_ORDER_MAX_PAIRS = None        # default: exact on all pairs (SPEC 1.4)
GLOBAL_ORDER_SUBSAMPLE = 20_000_000   # expected subsample size if max_pairs is set


def _upper_sample(DA, DB, frac, key):
    m = DA.shape[0]
    rng = util.rng_from_key(key)
    xs, ys = [], []
    for i in range(m - 1):
        sel = rng.random(m - 1 - i) < frac
        xs.append(DA[i, i + 1:][sel])
        ys.append(DB[i, i + 1:][sel])
    return np.concatenate(xs), np.concatenate(ys)


def global_order(DA: np.ndarray, DB: np.ndarray, *, max_pairs: int | None = GLOBAL_ORDER_MAX_PAIRS,
                 subsample: int = GLOBAL_ORDER_SUBSAMPLE, key: str = "hrhv2|global_order") -> dict:
    """Kendall tau_b and Spearman rho between the upper-triangular entries of
    D_A and D_B; Delta_E = (1 - tau_b)/2.

    Exact by default. Only if ``max_pairs`` is given and the number of pairs
    exceeds it (a non-SPEC speed option, flagged in the output), each pair is kept
    independently with probability subsample/n_pairs (Bernoulli, seeded by
    sha256(key)); the result reports ``globalorder_subsampled`` and the number
    of pairs used.
    """
    m = DA.shape[0]
    npairs = util.n_pairs(m)
    if max_pairs is not None and npairs > max_pairs:
        frac = subsample / npairs
        x, y = _upper_sample(DA, DB, frac, key)
        sub = True
    else:
        x, y = util.upper_values(DA), util.upper_values(DB)
        sub = False
    c = kendall_counts(x, y)
    tau = tau_b_from_counts(c)
    rx = rankdata(x)
    ry = rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    rho = float(np.dot(rx, ry) / np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    return {
        "globalorder_tau_b": tau,
        "globalorder_spearman": rho,
        "Delta_E": (1.0 - tau) / 2.0,
        "globalorder_kendall_dist_halfties": kendall_distance_from_counts(c),
        "globalorder_discordant": c["dis"],
        "globalorder_ties_A": c["xtie"],
        "globalorder_ties_B": c["ytie"],
        "globalorder_npairs_used": int(x.size),
        "globalorder_npairs_total": int(npairs),
        "globalorder_subsampled": bool(sub),
        "globalorder_subsample_key": key if sub else "",
    }


# ----------------------------------------------------------------------------
# MST (dense Prim with the SPEC tie key) and forest/bridge split
# ----------------------------------------------------------------------------

def mst_prim(D: np.ndarray):
    """Minimum spanning tree of the complete graph weighted by D.

    Dense Prim from node 0, O(m^2) time, O(m) extra memory. Edges are compared
    by the key (D_ij, min(i,j), max(i,j)); this is a strict total order on
    edges, so the returned tree is the unique minimum tree under that order.
    Returns int64 edge codes ``min*m + max`` (sorted) .
    """
    m = D.shape[0]
    if m < 2:
        return np.empty(0, dtype=np.int64)
    INF = np.inf
    notin = np.ones(m, dtype=bool)
    notin[0] = False
    best = np.array(D[0], dtype=np.float64, copy=True)
    best[0] = INF
    par = np.zeros(m, dtype=np.int64)
    codes = np.empty(m - 1, dtype=np.int64)
    for t in range(m - 1):
        mn = best.min()
        cand = np.flatnonzero(best == mn)
        if cand.size == 1:
            v = int(cand[0])
        else:
            pc = par[cand]
            lo_ = np.minimum(cand, pc)
            hi_ = np.maximum(cand, pc)
            v = int(cand[np.lexsort((hi_, lo_))[0]])
        u = int(par[v])
        a, b = (u, v) if u < v else (v, u)
        codes[t] = a * m + b
        notin[v] = False
        best[v] = INF
        row = D[v]
        upd = (row < best) & notin
        eqm = (row == best) & notin
        if eqm.any():
            w = np.flatnonzero(eqm)
            pw = par[w]
            new_lo = np.minimum(v, w)
            new_hi = np.maximum(v, w)
            old_lo = np.minimum(pw, w)
            old_hi = np.maximum(pw, w)
            better = (new_lo < old_lo) | ((new_lo == old_lo) & (new_hi < old_hi))
            upd[w[better]] = True
        best[upd] = row[upd]
        par[upd] = v
    return np.sort(codes)


def nn1_forest(nn1: np.ndarray) -> np.ndarray:
    """Undirected 1-NN edges {i, NN(i)} as sorted unique codes min*m+max."""
    m = nn1.size
    i = np.arange(m)
    a = np.minimum(i, nn1)
    b = np.maximum(i, nn1)
    return np.unique(a.astype(np.int64) * m + b)


def mst_compare(TA: np.ndarray, TB: np.ndarray, FA: np.ndarray, FB: np.ndarray | None = None) -> dict:
    """MST overlap and the forest/bridge split (SPEC 1.5)."""
    m1 = TA.size  # m - 1
    shared = np.intersect1d(TA, TB, assume_unique=True)
    FA_in_TA = np.isin(FA, TA, assume_unique=True).all()
    bridgesA = np.setdiff1d(TA, FA, assume_unique=True)
    shared_forest = np.intersect1d(shared, FA, assume_unique=True)
    shared_bridge = np.intersect1d(bridgesA, TB, assume_unique=True)
    out = {
        "s_T": shared.size / m1,
        "forest_share_A": FA.size / m1,
        "s_T_forest": shared_forest.size / FA.size if FA.size else float("nan"),
        "s_T_bridge": shared_bridge.size / bridgesA.size if bridgesA.size else float("nan"),
        "n_bridges_A": int(bridgesA.size),
        "n_shared_bridges": int(shared_bridge.size),
        "forest_in_mst_A": bool(FA_in_TA),
        "shared_forest_fraction_of_shared": (shared_forest.size / shared.size) if shared.size else float("nan"),
    }
    if FB is not None:
        bridgesB = np.setdiff1d(TB, FB, assume_unique=True)
        out.update({
            "forest_share_B": FB.size / m1,
            "s_T_forest_B": np.intersect1d(shared, FB, assume_unique=True).size / FB.size if FB.size else float("nan"),
            "s_T_bridge_B": np.intersect1d(bridgesB, TA, assume_unique=True).size / bridgesB.size if bridgesB.size else float("nan"),
            "forest_in_mst_B": bool(np.isin(FB, TB, assume_unique=True).all()),
        })
    return out
