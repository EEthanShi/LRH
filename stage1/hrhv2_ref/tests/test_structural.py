import itertools

import numpy as np
import pytest

from hrhv2_ref import structural as st
from hrhv2_ref.core import gram_sym, neighbor_order, sqdist_from_gram


def D_line(xs):
    X = np.asarray(xs, dtype=float)[:, None]
    return sqdist_from_gram(gram_sym(X))


def fagin(a, b):
    nd, tA, tB, tAB, n0 = st._pair_stats_rows(np.atleast_2d(a).astype(float), np.atleast_2d(b).astype(float))
    return float((nd[0] + 0.5 * ((tA[0] - tAB[0]) + (tB[0] - tAB[0]))) / n0)


# ---------------------------------------------------------------- kNN ----

def test_s_N_hand():
    NA = np.array([[1, 2], [0, 2], [0, 1]])
    NB = np.array([[2, 1], [2, 3], [3, 4]])
    # overlaps 2, 1, 0 -> 3 / (3*2)
    assert st.s_N(NA, NB) == pytest.approx(0.5)


def test_near_tie_fraction_hand():
    # x = 0, 1, -1, 5 : anchor 0 has 1st and 2nd distances equal (1, 1)
    D = D_line([0.0, 1.0, -1.0, 5.0])
    o = neighbor_order(D)
    # k=1: anchors whose 1st and 2nd distances tie: anchor 0 only -> 1/4
    assert st.near_tie_fraction(D, o, 1) == pytest.approx(0.25)


# --------------------------------------------------------- components ----

def test_rand_distance_hand():
    assert st.rand_distance([0, 0, 1, 1], [0, 0, 0, 1]) == pytest.approx(0.5)
    assert st.rand_distance([5, 5, 7], [1, 1, 2]) == 0.0


def test_rand_distance_vs_bruteforce():
    rng = np.random.default_rng(3)
    for _ in range(20):
        la = rng.integers(0, 4, size=30)
        lb = rng.integers(0, 5, size=30)
        tot = diff = 0
        for i, j in itertools.combinations(range(30), 2):
            tot += 1
            diff += (la[i] == la[j]) != (lb[i] == lb[j])
        assert st.rand_distance(la, lb) == pytest.approx(diff / tot, abs=1e-15)


def test_knn_components_hand():
    # 1-NN: 0<->1, 2<->3, 4->3  => {0,1}, {2,3,4}
    D = D_line([0.0, 1.0, 10.0, 11.0, 30.0])
    o = neighbor_order(D)
    lab = st.knn_components(o[:, :1])
    assert st.n_components(lab) == 2
    assert lab[0] == lab[1] and lab[2] == lab[3] == lab[4] and lab[0] != lab[2]


# ------------------------------------------------------------ Kendall ----

def test_fagin_hand():
    # a strict, b ties (0,1): pair(0,1) -> 1/2 ; others concordant  => 1/6
    assert fagin([1, 2, 3], [1, 1, 3]) == pytest.approx(1 / 6)
    # tie in both for (0,1) -> 0 ; (0,2),(1,2) discordant -> 2/3
    assert fagin([1, 1, 3], [2, 2, 0]) == pytest.approx(2 / 3)
    # reversed order -> 1
    assert fagin([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(1.0)


def test_inversions_vs_bruteforce():
    rng = np.random.default_rng(4)
    for n in (1, 2, 3, 5, 8, 13, 64, 100):
        V = rng.integers(0, max(1, n // 2), size=(7, n))
        got = st.count_inversions_rows(V)
        for r in range(7):
            ref = sum(1 for p in range(n) for q in range(p + 1, n) if V[r, p] > V[r, q])
            assert got[r] == ref


def test_kendall_rows_merge_equals_brute_with_ties():
    rng = np.random.default_rng(5)
    for m in (5, 17, 40):
        A = rng.integers(0, 6, size=(m, m)).astype(float)
        B = rng.integers(0, 6, size=(m, m)).astype(float)
        A = A + A.T
        B = B + B.T
        np.fill_diagonal(A, 0)
        np.fill_diagonal(B, 0)
        assert np.allclose(st.kendall_rows_merge(A, B, chunk=7), st.kendall_rows_brute(A, B), atol=1e-15)


def test_kendall_row_identity_tie_free():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(6)
    X = rng.normal(size=(30, 5))
    Y = X + 0.5 * rng.normal(size=(30, 5))
    DA = sqdist_from_gram(gram_sym(X))
    DB = sqdist_from_gram(gram_sym(Y))
    rows = st.kendall_rows_merge(DA, DB)
    for i in range(30):
        a = np.delete(DA[i], i)
        b = np.delete(DB[i], i)
        tb = scipy_stats.kendalltau(a, b, variant="b").statistic
        assert rows[i] == pytest.approx((1 - tb) / 2, abs=1e-13)


def test_kendall_tau_b_and_spearman_vs_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)
    x = rng.integers(0, 20, size=500).astype(float)
    y = x + rng.integers(0, 15, size=500)
    assert st.kendall_tau_b(x, y) == pytest.approx(scipy_stats.kendalltau(x, y, variant="b").statistic, abs=1e-12)
    assert st.spearman(x, y) == pytest.approx(scipy_stats.spearmanr(x, y).statistic, abs=1e-12)


# ---------------------------------------------------------------- MST ----

def test_mst_forest_bridge_hand():
    D = D_line([0.0, 1.0, 5.0, 6.0])
    m = 4
    T = st.mst_kruskal(D)
    assert sorted(T.tolist()) == [0 * m + 1, 1 * m + 2, 2 * m + 3]
    F = st.forest_1nn(neighbor_order(D), m)
    assert sorted(F.tolist()) == [0 * m + 1, 2 * m + 3]
    assert np.isin(F, T).all()


def random_tied_D(rng, m, levels=4):
    A = rng.integers(1, levels + 1, size=(m, m)).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    return A


def test_prim_tiekey_equals_kruskal_under_heavy_ties():
    rng = np.random.default_rng(8)
    for m in (3, 4, 6, 10, 25, 60):
        for _ in range(10):
            D = random_tied_D(rng, m)
            assert np.array_equal(st.mst_prim_tiekey(D), st.mst_kruskal(D))
            # 1-NN forest under the SPEC tie rule is inside the MST
            F = st.forest_1nn(neighbor_order(D), m)
            assert np.isin(F, st.mst_kruskal(D)).all()


def test_sog_prim_can_differ_under_ties():
    # edges: 0-2:1, 0-1:2, 1-3:3, 2-3:3, 1-2:5, 0-3:9
    D = np.zeros((4, 4))
    for (i, j), w in {(0, 2): 1, (0, 1): 2, (1, 3): 3, (2, 3): 3, (1, 2): 5, (0, 3): 9}.items():
        D[i, j] = D[j, i] = w
    m = 4
    spec = st.mst_kruskal(D)
    assert sorted(spec.tolist()) == [0 * m + 1, 0 * m + 2, 1 * m + 3]
    assert np.array_equal(st.mst_prim_tiekey(D), spec)
    sog = st.mst_prim_sog(D)
    assert sorted(sog.tolist()) == [0 * m + 1, 0 * m + 2, 2 * m + 3]


def test_mst_vs_scipy_tie_free():
    csgraph = pytest.importorskip("scipy.sparse.csgraph")
    rng = np.random.default_rng(9)
    X = rng.normal(size=(80, 6))
    D = sqdist_from_gram(gram_sym(X))
    T = st.mst_kruskal(D)
    S = csgraph.minimum_spanning_tree(D).tocoo()
    ref = sorted(min(i, j) * 80 + max(i, j) for i, j in zip(S.row, S.col))
    assert T.tolist() == ref
