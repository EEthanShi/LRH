"""SPEC section 1: structural observables, tie rule, exact invariances."""
import itertools

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import kendalltau, spearmanr

from hrhv2 import structural as st
from hrhv2 import util
from conftest import int_dist, rand_dist, tied_dist

# Hand example (see docstrings of the tests):
#   D_A: 0-2 = 1, 1-3 = 1, 0-1 = 0-3 = 2, 1-2 = 2-3 = 3  (ties at 2 and 3)
#   D_B: points 0,1,2,3 on a line, D_B(i,j) = |i - j|
DA_HAND = np.array([[0, 2, 1, 2],
                    [2, 0, 3, 1],
                    [1, 3, 0, 3],
                    [2, 1, 3, 0]], dtype=float)
DB_HAND = np.abs(np.subtract.outer(np.arange(4), np.arange(4))).astype(float)


def brute_knn(D, k):
    m = D.shape[0]
    out = []
    for i in range(m):
        cand = [(D[i, j], j) for j in range(m) if j != i]
        cand.sort()
        out.append([j for _, j in cand[:k]])
    return np.array(out)


def brute_mst(D):
    """Kruskal with the SPEC key (D_ij, min(i,j), max(i,j)) and union-find."""
    m = D.shape[0]
    edges = sorted((D[i, j], i, j) for i in range(m) for j in range(i + 1, m))
    par = list(range(m))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    T = []
    for _, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            par[ri] = rj
            T.append(i * m + j)
    return np.sort(np.array(T, dtype=np.int64))


def brute_kendall_dist(x, y):
    n = len(x)
    s = 0.0
    for a, b in itertools.combinations(range(n), 2):
        dx, dy = np.sign(x[a] - x[b]), np.sign(y[a] - y[b])
        if dx == 0 and dy == 0:
            continue
        if dx == 0 or dy == 0:
            s += 0.5
        elif dx != dy:
            s += 1.0
    return s / (n * (n - 1) / 2)


# ----------------------------------------------------------------------------
# kNN
# ----------------------------------------------------------------------------

def test_knn_hand_example_tie_rule():
    # row 0: j=1 (2), j=2 (1), j=3 (2) -> order 2, 1, 3 (tie 1 vs 3 -> smaller index)
    idx, val = st.knn_table(DA_HAND, 3)
    assert idx.tolist() == [[2, 1, 3], [3, 0, 2], [0, 1, 3], [1, 0, 2]]
    assert val.tolist() == [[1, 2, 2], [1, 2, 3], [1, 3, 3], [1, 2, 3]]
    idxB, _ = st.knn_table(DB_HAND, 3)
    assert idxB.tolist() == [[1, 2, 3], [0, 2, 3], [1, 3, 0], [2, 1, 0]]


def test_knn_overlap_hand():
    nA, _ = st.knn_table(DA_HAND, 3)
    nB, _ = st.knn_table(DB_HAND, 3)
    # k=1: A = 2,3,0,1 ; B = 1,0,1,2 -> no overlap
    assert st.knn_overlap(nA, nB, 1) == 0.0
    # k=2: |{2,1}&{1,2}|=2, |{3,0}&{0,2}|=1, |{0,1}&{1,3}|=1, |{1,0}&{2,1}|=1 -> 5/8
    assert st.knn_overlap(nA, nB, 2) == 5 / 8
    assert st.overlap_counts(nA, nB, 2).tolist() == [2, 1, 1, 1]


def test_constant_matrix_tie_rule():
    m = 7
    D = np.ones((m, m)) - np.eye(m)
    idx, _ = st.knn_table(D, 3)
    for i in range(m):
        assert idx[i].tolist() == [j for j in range(m) if j != i][:3]
    # Prim with key (D, min, max) on equal weights -> star at node 0
    T = st.mst_prim(D)
    assert T.tolist() == [0 * m + j for j in range(1, m)]
    assert np.array_equal(T, brute_mst(D))


@pytest.mark.parametrize("key,levels", [("t1", 2), ("t2", 3), ("t3", 5)])
def test_knn_matches_bruteforce_with_ties(key, levels):
    D = tied_dist(23, key, levels)
    for k in (1, 3, 7, 22):
        idx, _ = st.knn_table(D, k)
        assert np.array_equal(idx, brute_knn(D, k))


def test_knn_chunk_independent_and_deterministic():
    D = tied_dist(40, "chunk", 3)
    ref, rv = st.knn_table(D, 10)
    for chunk in (1, 3, 17, 512):
        idx, val = st.knn_table(D, 10, chunk=chunk)
        assert np.array_equal(idx, ref) and np.array_equal(val, rv)


def test_default_k_grid():
    assert st.default_k_grid(1024) == [1, 3, 10, 30, 100]  # k_q = round(10.23) = 10
    assert st.k_q(1024) == 10 and st.k_q(16384) == 164 and st.k_q(256) == 3
    assert st.default_k_grid(16384) == [1, 3, 10, 30, 100, 164]
    assert st.default_k_grid(50) == [1, 3, 10, 30]


def test_near_tie_fraction_hand():
    vals = np.array([[1.0, 2.0, 2.0], [1.0, 2.0, 3.0]])
    assert st.near_tie_fraction(vals, 2, median_D=1.0) == 0.5
    assert st.near_tie_fraction(vals, 1, median_D=1.0) == 0.0
    assert np.isnan(st.near_tie_fraction(vals, 3, median_D=1.0))


# ----------------------------------------------------------------------------
# components
# ----------------------------------------------------------------------------

def test_components_and_rand_hand():
    nA, _ = st.knn_table(DA_HAND, 1)
    nB, _ = st.knn_table(DB_HAND, 1)
    ca, la = st.knn_components(nA, 1)   # edges {0,2},{1,3}
    cb, lb = st.knn_components(nB, 1)   # edges {0,1},{1,2},{2,3}
    assert (ca, cb) == (2, 1)
    # pairs same in A: (0,2),(1,3); same in B: all 6 -> 4 of 6 differ
    assert st.rand_distance(la, lb) == pytest.approx(4 / 6, abs=0)


def test_rand_distance_bruteforce():
    rng = util.rng_from_key("rand")
    for _ in range(5):
        a = rng.integers(0, 4, 30)
        b = rng.integers(0, 3, 30)
        diff = sum((a[i] == a[j]) != (b[i] == b[j]) for i in range(30) for j in range(i + 1, 30))
        assert st.rand_distance(a, b) == diff / (30 * 29 / 2)


# ----------------------------------------------------------------------------
# Kendall
# ----------------------------------------------------------------------------

def test_row_kendall_hand():
    per, rows_with_ties = st.row_kendall(DA_HAND, DB_HAND)
    # row 0: A=(2,1,2) B=(1,2,3): (1,2) disc, (1,3) A-tie 1/2, (2,3) conc -> 1.5/3
    # row 1: A=(2,3,1) B=(1,1,2): (0,2) B-tie 1/2, (0,3) disc, (2,3) disc -> 2.5/3
    # row 2: A=(1,3,3) B=(2,1,1): (0,1) disc, (0,3) disc, (1,3) joint tie 0 -> 2/3
    # row 3: A=(2,1,3) B=(3,2,1): (0,1) conc, (0,2) disc, (1,2) disc -> 2/3
    assert per.tolist() == pytest.approx([1.5 / 3, 2.5 / 3, 2 / 3, 2 / 3], abs=1e-15)
    assert per.mean() == pytest.approx(2 / 3, abs=1e-15)
    assert rows_with_ties == 3


def test_kendall_distance_bruteforce_with_ties():
    rng = util.rng_from_key("kend")
    for t in range(20):
        n = 15
        x = rng.integers(0, 4, n).astype(float)
        y = rng.integers(0, 4, n).astype(float)
        assert st.kendall_distance(x, y) == pytest.approx(brute_kendall_dist(x, y), abs=1e-14)


def test_kendall_tau_b_identity_and_scipy():
    rng = util.rng_from_key("kend2")
    x = rng.standard_normal(500)
    y = x + rng.standard_normal(500)
    c = st.kendall_counts(x, y)
    tau = st.tau_b_from_counts(c)
    assert tau == pytest.approx(kendalltau(x, y, variant="b").statistic, abs=1e-14)
    # tie-free: Delta = (1 - tau_b)/2 exactly
    assert st.kendall_distance_from_counts(c) == pytest.approx((1 - tau) / 2, abs=1e-15)
    # with ties tau_b still matches scipy
    xt, yt = np.round(x), np.round(y)
    assert st.tau_b_from_counts(st.kendall_counts(xt, yt)) == pytest.approx(
        kendalltau(xt, yt, variant="b").statistic, abs=1e-13)


def test_row_kendall_bruteforce_parallel_equal():
    D1 = tied_dist(25, "rk1", 4)
    D2 = tied_dist(25, "rk2", 4)
    per1, _ = st.row_kendall(D1, D2, n_jobs=1)
    per2, _ = st.row_kendall(D1, D2, n_jobs=3)
    assert np.array_equal(per1, per2)
    for i in range(25):
        x = np.delete(D1[i], i)
        y = np.delete(D2[i], i)
        assert per1[i] == pytest.approx(brute_kendall_dist(x, y), abs=1e-14)


def test_global_order_vs_scipy():
    DA = rand_dist(60, "go1")
    DB = DA + rand_dist(60, "go2") * 0.3
    g = st.global_order(DA, DB)
    x, y = util.upper_values(DA), util.upper_values(DB)
    assert g["globalorder_tau_b"] == pytest.approx(kendalltau(x, y).statistic, abs=1e-13)
    assert g["globalorder_spearman"] == pytest.approx(spearmanr(x, y).statistic, abs=1e-13)
    assert g["Delta_E"] == pytest.approx((1 - g["globalorder_tau_b"]) / 2, abs=0)
    assert not g["globalorder_subsampled"] and g["globalorder_npairs_used"] == 60 * 59 // 2
    # tie handling
    T1, T2 = tied_dist(40, "go3", 3), tied_dist(40, "go4", 3)
    g = st.global_order(T1, T2)
    xt, yt = util.upper_values(T1), util.upper_values(T2)
    assert g["globalorder_tau_b"] == pytest.approx(kendalltau(xt, yt).statistic, abs=1e-13)
    assert g["globalorder_spearman"] == pytest.approx(spearmanr(xt, yt).statistic, abs=1e-13)
    # optional subsampling is flagged
    gs = st.global_order(DA, DB, max_pairs=100, subsample=500)
    assert gs["globalorder_subsampled"] and gs["globalorder_npairs_used"] < 60 * 59 // 2


# ----------------------------------------------------------------------------
# MST and forest/bridge split
# ----------------------------------------------------------------------------

def test_mst_hand_and_split():
    TA = st.mst_prim(DA_HAND)
    TB = st.mst_prim(DB_HAND)
    m = 4
    assert TA.tolist() == sorted([0 * m + 2, 1 * m + 3, 0 * m + 1])
    assert TB.tolist() == sorted([0 * m + 1, 1 * m + 2, 2 * m + 3])
    nA, _ = st.knn_table(DA_HAND, 1)
    nB, _ = st.knn_table(DB_HAND, 1)
    FA, FB = st.nn1_forest(nA[:, 0]), st.nn1_forest(nB[:, 0])
    assert FA.tolist() == [0 * m + 2, 1 * m + 3]
    r = st.mst_compare(TA, TB, FA, FB)
    assert r["s_T"] == 1 / 3
    assert r["forest_share_A"] == 2 / 3
    assert r["s_T_forest"] == 0.0 and r["s_T_bridge"] == 1.0
    assert r["n_bridges_A"] == 1 and r["n_shared_bridges"] == 1
    assert r["forest_in_mst_A"] and r["forest_in_mst_B"]


def test_rectangle_mst_differs_with_identical_rankings():
    """theory_mechanism.md §3: T_MST is not below T_O (rectangle example)."""
    a, b, eps = 2.0, 1.0, 0.05

    def D_of(P):
        P = np.asarray(P, float)
        return ((P[:, None] - P[None]) ** 2).sum(-1)
    D1 = D_of([(0, 0), (a + eps, 0), (a, b), (0, b)])
    D2 = D_of([(0, 0), (a - eps, 0), (a, b), (0, b)])
    per, _ = st.row_kendall(D1, D2)
    assert per.max() == 0.0
    assert not np.array_equal(st.mst_prim(D1), st.mst_prim(D2))


@pytest.mark.parametrize("seed", range(4))
def test_mst_matches_scipy_tie_free(seed):
    D = rand_dist(80, f"mst{seed}")
    T = st.mst_prim(D)
    M = minimum_spanning_tree(np.triu(D, 1)).tocoo()
    ref = np.sort(np.minimum(M.row, M.col).astype(np.int64) * 80 + np.maximum(M.row, M.col))
    assert np.array_equal(T, ref)


@pytest.mark.parametrize("key,levels", [("mt1", 2), ("mt2", 3), ("mt3", 6)])
def test_mst_matches_bruteforce_kruskal_with_ties(key, levels):
    D = tied_dist(30, key, levels)
    assert np.array_equal(st.mst_prim(D), brute_mst(D))


@pytest.mark.parametrize("maker,key", [(rand_dist, "f1"), (rand_dist, "f2"), (tied_dist, "f3"),
                                       (tied_dist, "f4"), (int_dist, "f5")])
def test_forest_subset_of_mst(maker, key):
    D = maker(70, key)
    nn, _ = st.knn_table(D, 1)
    F = st.nn1_forest(nn[:, 0])
    T = st.mst_prim(D)
    assert np.isin(F, T).all()
    assert st.mst_compare(T, T, F)["forest_in_mst_A"]


# ----------------------------------------------------------------------------
# exact invariance under strictly increasing maps
# ----------------------------------------------------------------------------

MAPS = {
    "log": np.log,
    "sqrt": np.sqrt,
    "cube_plus": lambda x: x ** 3 + x,
    "exp_scaled": lambda x: np.exp(x / x.max() * 5.0),
    "bernstein_like": lambda x: 2.0 - 2.0 * np.exp(-0.5 * x / x.max() * 3.0),
}


@pytest.mark.parametrize("name", list(MAPS))
def test_ordinal_rungs_exactly_invariant_under_increasing_maps(name):
    m = 60
    D = int_dist(m, "inv")
    f = MAPS[name]
    vals = np.unique(util.upper_values(D))
    fv = f(vals)
    assert np.all(np.diff(fv) > 0), "map must stay strictly increasing in float64"
    D2 = np.zeros_like(D)
    iu = np.triu_indices(m, 1)
    D2[iu] = f(D[iu])
    D2 = D2 + D2.T
    n1, _ = st.knn_table(D, 20)
    n2, _ = st.knn_table(D2, 20)
    assert np.array_equal(n1, n2)
    for k in (1, 3, 10):
        assert st.rand_distance(st.knn_components(n1, k)[1], st.knn_components(n2, k)[1]) == 0.0
    per, _ = st.row_kendall(D, D2)
    assert per.max() == 0.0
    g = st.global_order(D, D2)
    assert g["Delta_E"] == 0.0 and g["globalorder_tau_b"] == 1.0
    assert np.array_equal(st.mst_prim(D), st.mst_prim(D2))


def test_row_monotone_maps_preserve_knn_and_rankings_only():
    m = 50
    D = int_dist(m, "rowinv")
    D2 = D.copy()
    for i in range(m):  # a different strictly increasing map per anchor row
        D2[i] = (i + 1.0) * D[i] ** (1.0 + i / m)
    n1, _ = st.knn_table(D, 15)
    n2, _ = st.knn_table(D2, 15)
    assert np.array_equal(n1, n2)
    per, _ = st.row_kendall(D, D2)
    assert per.max() == 0.0
    # a global (cross-anchor) observable does change
    Ds = 0.5 * (D2 + D2.T)
    assert st.global_order(D, Ds)["Delta_E"] > 0
