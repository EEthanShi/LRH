import numpy as np
import pytest

from hrhv2_ref import quotient as quo
from hrhv2_ref.core import gram_sym, sqdist_from_gram


def test_pava_hand():
    assert quo.pava([1, 3, 2, 4], [1, 1, 1, 1]).tolist() == [1, 2.5, 2.5, 4]
    assert quo.pava([3, 1], [1, 3]).tolist() == [1.5, 1.5]
    assert quo.pava([5, 4, 3, 2, 1], [1] * 5).tolist() == [3.0] * 5
    assert quo.pava([1, 2, 3], [1, 1, 1]).tolist() == [1, 2, 3]


def minmax_isotonic(y, w):
    """g_i = max_{s<=i} min_{t>=i} weighted mean(y_s..y_t)  (O(n^3) brute force)."""
    n = len(y)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    g = np.empty(n)
    for i in range(n):
        best = -np.inf
        for s in range(i + 1):
            inner = min(np.dot(w[s:t + 1], y[s:t + 1]) / w[s:t + 1].sum() for t in range(i, n))
            best = max(best, inner)
        g[i] = best
    return g


def test_pava_vs_minmax_and_scipy():
    opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(20)
    for n in (1, 2, 5, 12, 30):
        y = rng.normal(size=n)
        w = rng.uniform(0.5, 2.0, size=n)
        got = quo.pava(y, w)
        assert np.allclose(got, minmax_isotonic(y, w), atol=1e-12)
        assert np.allclose(got, opt.isotonic_regression(y, weights=w).x, atol=1e-12)


def test_iso_fit_tie_pooling_and_predict():
    kx, kg = quo.iso_fit([1, 1, 2], [3, 1, 2])
    assert kx.tolist() == [1, 2] and kg.tolist() == [2, 2]
    kx, kg = quo.iso_fit([0, 1], [0, 2])
    assert quo.iso_predict(kx, kg, np.array([-1.0, 0.0, 0.5, 1.0, 3.0])).tolist() == [0, 0, 1, 2, 2]


def test_quotients_exact_warps():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(25, 4))
    DA = sqdist_from_gram(gram_sym(X))
    H1 = np.zeros(25, bool)
    H1[:12] = True
    q = quo.quotients_one_direction(DA, 2.0 * DA, H1)
    assert q["scale"][0] == pytest.approx(1.0) and q["scale"][1] == pytest.approx(0.0, abs=1e-12)
    assert q["scale_xfit"][0] == pytest.approx(1.0)
    off = 1 - np.eye(25)
    q = quo.quotients_one_direction(DA, 2.0 * DA + off, H1)
    assert q["affine"][0] == pytest.approx(1.0) and q["scale"][0] < 0.999
    q = quo.quotients_one_direction(DA, np.sqrt(DA) * off, H1)  # monotone, non-affine
    assert q["iso"][0] == pytest.approx(1.0, abs=1e-12)
    assert q["rowiso"][0] == pytest.approx(1.0, abs=1e-12)
    assert q["affine"][0] < 0.999
    # cross-fitted iso: interpolation error only
    assert q["iso_xfit"][0] > 0.99
