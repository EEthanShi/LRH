import itertools

import numpy as np
import pytest

from hrhv2_ref import geometric as geo
from hrhv2_ref.core import gram_sym, sqdist_from_gram


def test_cka_biased_hand_1d():
    # 1-D linear CKA = squared Pearson correlation: x=[1,2,3], y=[1,2,4] -> 27/28
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[1.0], [2.0], [4.0]])
    assert geo.cka_biased(gram_sym(X), gram_sym(Y)) == pytest.approx(27 / 28, abs=1e-14)


def song_bruteforce(K, L):
    """Song et al. 2012 Thm 2: U-statistic over ordered 4-tuples of distinct indices,
    h = (1/4!) sum over permutations (s,t,u,v) of K_st L_st + K_st L_uv - 2 K_st L_su."""
    m = K.shape[0]
    tot = 0.0
    cnt = 0
    for tup in itertools.permutations(range(m), 4):
        h = 0.0
        for s, t, u, v in itertools.permutations(tup):
            h += K[s, t] * L[s, t] + K[s, t] * L[u, v] - 2 * K[s, t] * L[s, u]
        tot += h / 24.0
        cnt += 1
    return tot / cnt


def test_hsic_song_equals_ustatistic_definition():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(6, 3))
    Y = rng.normal(size=(6, 2)) + X[:, :2]
    K, L = gram_sym(X), gram_sym(Y)
    assert geo.hsic_song(K, L) == pytest.approx(song_bruteforce(K, L), rel=1e-12)


def test_hsic_song_equals_ucentering():
    # Szekely-Rizzo U-centred inner product equals Song's estimator
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 5))
    Y = rng.normal(size=(30, 4))
    K, L = gram_sym(X), gram_sym(Y)
    m = 30

    def ucenter(A):
        A = A.copy()
        np.fill_diagonal(A, 0)
        r = A.sum(1) / (m - 2)
        g = A.sum() / ((m - 1) * (m - 2))
        U = A - r[:, None] - r[None, :] + g
        np.fill_diagonal(U, 0)
        return U

    ref = np.sum(ucenter(K) * ucenter(L)) / (m * (m - 3))
    assert geo.hsic_song(K, L) == pytest.approx(ref, rel=1e-12)


def test_cka_invariances():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(40, 6))
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    Y = 3.0 * X @ Q
    K, L = gram_sym(X), gram_sym(Y)
    assert geo.cka_biased(K, L) == pytest.approx(1.0, abs=1e-12)
    assert geo.cka_debiased(K, L) == pytest.approx(1.0, abs=1e-12)
    # debiased CKA of linear kernels is invariant to a feature mean shift
    Z = rng.normal(size=(40, 4))
    a = geo.cka_debiased(K, gram_sym(Z))
    b = geo.cka_debiased(K, gram_sym(Z + 5.0))
    assert a == pytest.approx(b, abs=1e-10)


def test_rsa_and_geometry_hand():
    DA = np.array([[0, 1, 4], [1, 0, 9], [4, 9, 0]], dtype=float)
    DB = 3 * DA + (1 - np.eye(3))
    assert geo.rsa_pearson(DA, DB) == pytest.approx(1.0)
    gs = geo.scale_normalised_geometry(DA, 7 * DA)
    assert gs[0] == pytest.approx(0.0, abs=1e-15) and gs[1] == pytest.approx(0.0, abs=1e-15)
    g = geo.raw_geometry(DA, DA + 2 * (1 - np.eye(3)))
    assert g == pytest.approx((2.0, 2.0))


def test_sog_weights_and_quantile():
    assert geo.quantile_linear([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 0.9) == pytest.approx(10.0)
    assert geo.quantile_linear([1, 2, 3, 4], 0.9) == pytest.approx(3.7)
    a, b = np.array([1.0, 2.0, 0.0, 0.0]), np.array([1.0, 2 * np.e, 0.0, 1.0])
    w = geo.sog_weights(a, b, 1.0, 1.0, 1.0, eps=0.0)  # v0.1 rule: one zero -> 0
    assert w.tolist() == pytest.approx([1.0, np.exp(-1.0), 1.0, 0.0], rel=1e-15, abs=0)
    w = geo.sog_weights(a, b, 1.0, 1.0, 1.0)  # SOG rule: log(max(d~, 1e-12))
    assert w[:3] == pytest.approx([1.0, np.exp(-1.0), 1.0]) and w[3] == pytest.approx(1e-12, rel=1e-9)
    # doubled-multiset quantile == numpy quantile of the explicitly doubled array
    rng = np.random.default_rng(3)
    for n in (1, 2, 5, 37, 1000):
        s = np.sort(rng.random(n))
        for q in (0.0, 0.1, 0.5, 0.9, 0.95, 1.0):
            assert geo.quantile_linear_doubled(s, q) == pytest.approx(np.quantile(np.repeat(s, 2), q), rel=1e-14, abs=0)
    # sog_scale 'ordered' == SOG's np.quantile(D[D > 0], 0.9) over the full matrix of d
    X = rng.normal(size=(40, 5))
    X[7] = X[3]  # one duplicated point -> an off-diagonal zero distance
    D = sqdist_from_gram(gram_sym(X))
    d = np.sqrt(D)
    assert geo.sog_scale(D) == pytest.approx(float(np.quantile(d[d > 0], 0.9)), rel=1e-15)
    iu = np.triu_indices(40, 1)
    du = d[iu]
    assert geo.sog_scale(D, mode="unordered") == pytest.approx(float(np.quantile(du[du > 0], 0.9)), rel=1e-15)
    # normalisation: scaling B by c and its quantile by c leaves w unchanged
    w2 = geo.sog_weights(np.array([2.0]), np.array([6.0]), 1.0, 3.0, 0.1)
    assert w2[0] == pytest.approx(1.0)
    w3 = geo.sog_weights(np.array([2.0]), np.array([3.0]), 1.0, 3.0, 0.1)
    assert w3[0] == pytest.approx(np.exp(-np.log(2.0) / 0.1))


def test_projector_hand():
    XA = np.array([[1.0], [-1.0], [0.0], [0.0]])
    XB = np.array([[0.0], [0.0], [1.0], [-1.0]])
    pd, cca, rA, rB = geo.projector_distance(XA, XB)
    assert (rA, rB) == (1, 1)
    assert pd == pytest.approx(1.0, abs=1e-14) and cca == pytest.approx(0.0, abs=1e-14)
    rng = np.random.default_rng(13)
    X = rng.normal(size=(50, 4))
    M = rng.normal(size=(4, 4))
    pd, cca, _, _ = geo.projector_distance(X, X @ M + 2.0)
    assert pd == pytest.approx(0.0, abs=1e-12) and cca == pytest.approx(1.0, abs=1e-12)
    Y = rng.normal(size=(50, 4))
    pd, cca, _, _ = geo.projector_distance(X, Y)
    assert pd == pytest.approx(1.0 - cca, abs=1e-12)  # identity when both full rank d
