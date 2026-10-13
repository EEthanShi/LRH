"""SPEC §1: structural (ordinal) observables.

Algorithms (deliberately different from the "obvious" ones):
  * kNN sets: full lexsort on (D_ij, j) (core.neighbor_order); overlap counted by
    sorting the concatenated neighbour lists and counting adjacent duplicates.
  * kNN-graph components: union-find; Rand distance via the contingency-table
    pair-count identity (brute force in tests).
  * Per-anchor Kendall distance: Fagin et al. (2006) K^(1/2) — pairs strictly
    ordered in both and in opposite directions count 1, pairs tied in exactly one
    ranking count 1/2, pairs tied in both count 0 — computed by brute-force O(n^2)
    sign products for small m, otherwise by a vectorised bottom-up merge-sort
    inversion count (Knight 1966) applied to all rows at once.
  * Global edge order: Kendall tau_b by the same merge-sort counter; Spearman as
    Pearson of mid-ranks.
  * MST: Kruskal on the strict total order (D, min(i,j), max(i,j)) with
    union-find.  A dense Prim with the same key is provided for the cross-check
    (it must give the identical tree because the key is a strict total order);
    the SOG-paper Prim (strict-improvement parent update, lowest vertex index on
    equal minima) is also provided to document that it can differ under ties.
"""
from __future__ import annotations

import numpy as np

from .core import neighbor_order, upper


# --------------------------------------------------------------------------
# kNN overlap
# --------------------------------------------------------------------------

def knn_overlap_counts(NA, NB):
    """Per-anchor |N_A(i) ∩ N_B(i)| for (m,k) index arrays with distinct entries per row."""
    C = np.sort(np.concatenate([NA, NB], axis=1), axis=1)
    return (C[:, 1:] == C[:, :-1]).sum(axis=1)


def s_N(NA, NB):
    m, k = NA.shape
    return float(knn_overlap_counts(NA, NB).sum()) / float(m * k)


def near_tie_fraction(D, order, k, rel=1e-12):
    """Fraction of anchors whose k-th and (k+1)-th smallest distances differ by
    < rel * median(off-diagonal D)."""
    m = D.shape[0]
    if k + 1 > m - 1:
        return float("nan")
    med = float(np.median(upper(D)))
    rows = np.arange(m)
    dk = D[rows, order[:, k - 1]]
    dk1 = D[rows, order[:, k]]
    return float(np.mean((dk1 - dk) < rel * med))


# --------------------------------------------------------------------------
# union-find and components
# --------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]  # path halving
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]
        return True

    def labels(self):
        n = len(self.p)
        lab = np.empty(n, dtype=np.int64)
        for i in range(n):
            lab[i] = self.find(i)
        return lab


def knn_components(N):
    """Component labels of the undirected graph {i,j}: j in N_k(i) or i in N_k(j)."""
    m, k = N.shape
    uf = UnionFind(m)
    for i, row in enumerate(N.tolist()):
        for j in row:
            uf.union(i, j)
    return uf.labels()


def _same_pairs(labels):
    _, cnt = np.unique(labels, return_counts=True)
    cnt = cnt.astype(object)  # python ints, no overflow
    return int(sum(c * (c - 1) // 2 for c in cnt))


def rand_distance(la, lb):
    """Fraction of unordered pairs whose same-component indicator differs."""
    la = np.asarray(la)
    lb = np.asarray(lb)
    m = la.shape[0]
    _, ia = np.unique(la, return_inverse=True)
    _, ib = np.unique(lb, return_inverse=True)
    joint = ia.astype(np.int64) * (int(ib.max()) + 1) + ib.astype(np.int64)
    sa, sb, sab = _same_pairs(ia), _same_pairs(ib), _same_pairs(joint)
    return (sa + sb - 2 * sab) / (m * (m - 1) / 2)


def n_components(labels):
    return int(np.unique(labels).size)


# --------------------------------------------------------------------------
# rank helpers and merge-sort inversion counting
# --------------------------------------------------------------------------

def dense_rank_rows(V):
    """Dense integer ranks (0..) per row, equal values share a rank."""
    V = np.atleast_2d(V)
    o = np.argsort(V, axis=1, kind="stable")
    s = np.take_along_axis(V, o, axis=1)
    new = np.zeros(s.shape, dtype=np.int64)
    new[:, 1:] = (s[:, 1:] != s[:, :-1])
    r_sorted = np.cumsum(new, axis=1)
    R = np.empty_like(r_sorted)
    np.put_along_axis(R, o, r_sorted, axis=1)
    return R


def tied_pairs_sorted_rows(*cols_sorted):
    """Number of tied pairs per row, given rows already sorted lexicographically
    by the provided columns (ties = equal in all provided columns)."""
    first = cols_sorted[0]
    R, n = first.shape
    if n <= 1:
        return np.zeros(R, dtype=np.int64)
    eq = np.ones((R, n - 1), dtype=bool)
    for c in cols_sorted:
        eq &= (c[:, 1:] == c[:, :-1])
    pos = np.arange(n, dtype=np.int64)
    start_marker = np.zeros((R, n), dtype=np.int64)
    start_marker[:, 1:] = np.where(eq, 0, pos[1:])
    gs = np.maximum.accumulate(start_marker, axis=1)
    return (pos[None, :] - gs).sum(axis=1)


def count_inversions_rows(V):
    """Number of strict inversions (p<q, V[p] > V[q]) in each row of an integer
    array with values in [0, n).  Vectorised bottom-up merge sort: at each level
    the left blocks are sorted, so for every element of a right block the number
    of strictly larger left-block elements is w - searchsorted(left, x, 'right'),
    done for all blocks of all rows at once with block offsets."""
    V = np.atleast_2d(np.asarray(V, dtype=np.int64))
    R, n = V.shape
    inv = np.zeros(R, dtype=np.int64)
    if n <= 1:
        return inv
    npad = 1 << (n - 1).bit_length()
    W = np.full((R, npad), n, dtype=np.int64)  # sentinel n > every rank, at the end
    W[:, :n] = V
    base = npad + 1
    w = 1
    while w < npad:
        nb = npad // (2 * w)
        X = W.reshape(R, nb, 2, w)
        blk = (np.arange(R, dtype=np.int64)[:, None] * nb + np.arange(nb, dtype=np.int64)[None, :])
        off = (blk * base)[:, :, None]
        Lg = (X[:, :, 0, :] + off).ravel()
        Rg = (X[:, :, 1, :] + off).ravel()
        pos = np.searchsorted(Lg, Rg, side="right").reshape(R, nb, w)
        le = pos - (blk * w)[:, :, None]
        inv += (w - le).sum(axis=(1, 2))
        W = np.sort(W.reshape(R, nb, 2 * w), axis=2).reshape(R, npad)
        w *= 2
    return inv


def _pair_stats_rows(a, b):
    """For each row: (n_discordant_strict, ties_a, ties_b, ties_ab, n_pairs)."""
    R, n = a.shape
    o = np.lexsort((b, a), axis=-1)  # primary a, secondary b
    a_s = np.take_along_axis(a, o, axis=1)
    b_s = np.take_along_axis(b, o, axis=1)
    rb = dense_rank_rows(b)
    rb_s = np.take_along_axis(rb, o, axis=1)
    nd = count_inversions_rows(rb_s)
    tA = tied_pairs_sorted_rows(a_s)
    tAB = tied_pairs_sorted_rows(a_s, b_s)
    b_sorted = np.sort(b, axis=1)
    tB = tied_pairs_sorted_rows(b_sorted)
    npairs = n * (n - 1) // 2
    return nd, tA, tB, tAB, npairs


def _offdiag_rows(Mx, rows):
    m = Mx.shape[0]
    out = np.empty((len(rows), m - 1), dtype=Mx.dtype)
    for t, i in enumerate(rows):
        out[t, :i] = Mx[i, :i]
        out[t, i:] = Mx[i, i + 1:]
    return out


def kendall_rows_merge(DA, DB, chunk=256, return_detail=False):
    """Per-anchor Fagin K^(1/2) Kendall distance, merge-sort counting.
    With return_detail=True also returns per-row (strict-discordant fraction,
    A-only tie fraction, B-only tie fraction)."""
    m = DA.shape[0]
    res = np.empty(m, dtype=np.float64)
    det = np.empty((m, 3), dtype=np.float64)
    for s in range(0, m, chunk):
        rows = np.arange(s, min(m, s + chunk))
        a = _offdiag_rows(DA, rows)
        b = _offdiag_rows(DB, rows)
        nd, tA, tB, tAB, npairs = _pair_stats_rows(a, b)
        res[rows] = (nd + 0.5 * ((tA - tAB) + (tB - tAB))) / npairs
        det[rows, 0] = nd / npairs
        det[rows, 1] = (tA - tAB) / npairs
        det[rows, 2] = (tB - tAB) / npairs
    if return_detail:
        return res, det
    return res


def kendall_pair_counts(x, y):
    """Global counts: dict(n0, n_disc_strict, ties_A_only, ties_B_only, ties_both)."""
    x = np.asarray(x, dtype=np.float64)[None, :]
    y = np.asarray(y, dtype=np.float64)[None, :]
    nd, tA, tB, tAB, n0 = _pair_stats_rows(x, y)
    nd, tA, tB, tAB = int(nd[0]), int(tA[0]), int(tB[0]), int(tAB[0])
    return {"n0": n0, "n_disc_strict": nd, "ties_A_only": tA - tAB, "ties_B_only": tB - tAB,
            "ties_both": tAB}


def kendall_rows_brute(DA, DB):
    """Per-anchor Fagin K^(1/2) Kendall distance by explicit O(n^2) pair signs."""
    m = DA.shape[0]
    res = np.empty(m, dtype=np.float64)
    n = m - 1
    iu, ju = np.triu_indices(n, 1)
    for i in range(m):
        a = np.delete(DA[i], i)
        b = np.delete(DB[i], i)
        sa = np.sign(a[iu] - a[ju])
        sb = np.sign(b[iu] - b[ju])
        disc = np.count_nonzero(sa * sb < 0)
        one_tie = np.count_nonzero((sa == 0) ^ (sb == 0))
        res[i] = (disc + 0.5 * one_tie) / len(iu)
    return res


def tau_b_from_counts(c):
    n0, nd = c["n0"], c["n_disc_strict"]
    tA = c["ties_A_only"] + c["ties_both"]
    tB = c["ties_B_only"] + c["ties_both"]
    nc = n0 - tA - tB + c["ties_both"] - nd
    den = float(n0 - tA) * float(n0 - tB)
    if den <= 0:
        return float("nan")
    return float(nc - nd) / np.sqrt(den)


def kendall_tau_b(x, y):
    """Kendall tau_b of two 1-D samples via merge-sort counting (exact integer counts)."""
    return tau_b_from_counts(kendall_pair_counts(x, y))


def midranks(v):
    v = np.asarray(v, dtype=np.float64)
    o = np.argsort(v, kind="stable")
    s = v[o]
    new = np.empty(s.size, dtype=bool)
    new[0] = True
    new[1:] = s[1:] != s[:-1]
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], s.size)
    avg = (starts + ends - 1) / 2.0 + 1.0
    gid = np.cumsum(new) - 1
    r = np.empty(v.size, dtype=np.float64)
    r[o] = avg[gid]
    return r


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    yc = y - y.mean()
    den = np.sqrt(np.dot(xc, xc) * np.dot(yc, yc))
    if den == 0:
        return float("nan")
    return float(np.dot(xc, yc) / den)


def spearman(x, y):
    return pearson(midranks(x), midranks(y))


# --------------------------------------------------------------------------
# MST
# --------------------------------------------------------------------------

def mst_kruskal(D):
    """Kruskal MST under the strict total order (D_ij, min(i,j), max(i,j)).
    Returns sorted int64 array of edge codes i*m + j (i<j)."""
    m = D.shape[0]
    if m < 2:
        return np.zeros(0, dtype=np.int64)
    I, J = np.triu_indices(m, 1)
    w = D[I, J]
    order = np.lexsort((J, I, w))
    uf = UnionFind(m)
    edges = []
    need = m - 1
    CH = 1 << 16
    for s in range(0, order.size, CH):
        oi = order[s:s + CH]
        for i, j in zip(I[oi].tolist(), J[oi].tolist()):
            if uf.union(i, j):
                edges.append(i * m + j)
                if len(edges) == need:
                    return np.array(sorted(edges), dtype=np.int64)
    raise RuntimeError("graph not connected?")


def _key_less(d1, lo1, hi1, d2, lo2, hi2):
    return (d1 < d2) | ((d1 == d2) & ((lo1 < lo2) | ((lo1 == lo2) & (hi1 < hi2))))


def mst_prim_tiekey(D, root=0):
    """Dense Prim from `root` comparing crossing edges by the full SPEC key
    (D, min, max) both in the parent update and in the vertex selection."""
    m = D.shape[0]
    in_tree = np.zeros(m, dtype=bool)
    in_tree[root] = True
    best_d = D[root].astype(np.float64).copy()
    best_u = np.full(m, root, dtype=np.int64)
    edges = []
    for _ in range(m - 1):
        cand = np.flatnonzero(~in_tree)
        bu = best_u[cand]
        lo = np.minimum(bu, cand)
        hi = np.maximum(bu, cand)
        sel = np.lexsort((hi, lo, best_d[cand]))[0]
        v = int(cand[sel])
        u = int(best_u[v])
        edges.append(min(u, v) * m + max(u, v))
        in_tree[v] = True
        rest = np.flatnonzero(~in_tree)
        if rest.size == 0:
            break
        nd = D[v, rest]
        nlo = np.minimum(v, rest)
        nhi = np.maximum(v, rest)
        cu = best_u[rest]
        better = _key_less(nd, nlo, nhi, best_d[rest], np.minimum(cu, rest), np.maximum(cu, rest))
        upd = rest[better]
        best_d[upd] = nd[better]
        best_u[upd] = v
    return np.array(sorted(edges), dtype=np.int64)


def mst_prim_sog(D, root=0):
    """Dense Prim as described in SOG App. C.3: parent updated only on strict
    improvement of D; among equal minima the lowest vertex index is selected.
    Provided only to document that it can differ from the SPEC key under ties."""
    m = D.shape[0]
    in_tree = np.zeros(m, dtype=bool)
    in_tree[root] = True
    best_d = D[root].astype(np.float64).copy()
    best_d[root] = np.inf
    best_u = np.full(m, root, dtype=np.int64)
    edges = []
    for _ in range(m - 1):
        bd = np.where(in_tree, np.inf, best_d)
        v = int(np.argmin(bd))  # first (lowest index) among equal minima
        u = int(best_u[v])
        edges.append(min(u, v) * m + max(u, v))
        in_tree[v] = True
        improve = (~in_tree) & (D[v] < best_d)
        best_d[improve] = D[v][improve]
        best_u[improve] = v
    return np.array(sorted(edges), dtype=np.int64)


def forest_1nn(order, m):
    """Undirected 1-NN edges (codes i*m+j, i<j) under the SPEC tie rule."""
    nn = order[:, 0]
    i = np.arange(m, dtype=np.int64)
    lo = np.minimum(i, nn)
    hi = np.maximum(i, nn)
    return np.unique(lo * m + hi)
