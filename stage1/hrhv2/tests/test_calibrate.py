"""SPEC section 5: permutation calibration."""
import numpy as np
import pytest

from hrhv2 import calibrate as cal
from hrhv2 import structural as st
from hrhv2.prep import Geom
from conftest import rand_features


def test_order_index_exact_rational():
    assert cal.order_index(19, 0.05) == 19      # (0.95)(20) = 19 exactly
    assert cal.order_index(500, 0.05) == 476    # ceil(475.95)
    assert cal.order_index(99, 0.01) == 99
    assert cal.order_index(9, 0.1) == 9
    assert cal.order_index(10, 0.05) == 11      # ceil(10.45) -> the max of K+1 values
    assert cal.order_index(1000, 0.05) == 951   # ceil(950.95)


def test_threshold_and_summary_hand():
    null = np.arange(19) / 19 * 0.4          # 19 values < 0.4
    s_obs = 0.5
    c = cal.threshold(s_obs, null, 0.05)     # 19th of 20 sorted values = largest null value
    assert c == null[18]
    r = cal.summarize(s_obs, null, 0.05)
    assert r["c_alpha"] == c
    assert r["s_cal"] == pytest.approx((0.5 - c) / (1 - c), abs=1e-15)
    assert r["p_value"] == pytest.approx(1 / 20, abs=1e-15)
    assert r["uncensored"] == pytest.approx(0.5 - null.mean(), abs=1e-15)
    # censoring: s_obs below the threshold -> 0
    r2 = cal.summarize(0.1, null, 0.05)
    assert r2["s_cal"] == 0.0
    assert r2["p_value"] == pytest.approx((1 + np.sum(null >= 0.1)) / 20, abs=1e-15)
    # K = 500: threshold is the 476th order statistic of the 501 values
    rng = np.random.default_rng(0)
    nul = rng.random(500)
    comb = np.sort(np.append(nul, 0.3))
    assert cal.threshold(0.3, nul, 0.05) == comb[475]


def test_permutations_are_key_deterministic():
    p1 = cal.permutation(100, "keyA", 3)
    assert np.array_equal(p1, cal.permutation(100, "keyA", 3))
    assert not np.array_equal(p1, cal.permutation(100, "keyA", 4))
    assert not np.array_equal(p1, cal.permutation(100, "keyB", 3))
    assert np.array_equal(np.sort(p1), np.arange(100))


def _score_knn(k):
    def f(A, B):
        nA, _ = st.knn_table(A.D, k)
        nB, _ = st.knn_table(B.D, k)
        return st.knn_overlap(nA, nB, k)
    return f


def test_calibrate_knn_fast_path_equals_recomputation():
    XA = rand_features(70, 5, "cal1")
    XB = XA + 0.5 * rand_features(70, 5, "cal2")
    A, B = Geom.from_features(XA, "l2"), Geom.from_features(XB, "l2")
    k = 5
    slow = cal.calibrate(_score_knn(k), A, B, key="t|knn", K=40, return_null=True)
    nA, _ = st.knn_table(A.D, k)
    nB, _ = st.knn_table(B.D, k)
    fast = cal.calibrate_knn(nA, nB, k, key="t|knn", K=40)
    for f in ("s_obs", "c_alpha", "s_cal", "p_value", "null_mean", "uncensored"):
        assert fast[f] == slow[f], f
    # repeat -> identical (no global RNG)
    again = cal.calibrate(_score_knn(k), A, B, key="t|knn", K=40, return_null=True)
    assert np.array_equal(again["null"], slow["null"])


def test_layer_max_uses_same_permutation_for_all_layers():
    m = 40
    XA = [rand_features(m, 4, f"cl|A{l}") for l in range(2)]
    XB = [XA[0] + 0.3 * rand_features(m, 4, f"cl|B{l}") for l in range(3)]
    As = [Geom.from_features(x, "l2") for x in XA]
    Bs = [Geom.from_features(x, "l2") for x in XB]
    score = _score_knn(3)
    r = cal.calibrate_layers(score, As, Bs, key="t|layers", K=15)
    manual = []
    for kk in range(15):
        pi = cal.permutation(m, "t|layers", kk)
        manual.append(max(score(a, b.permuted(pi)) for a in As for b in Bs))
    expect = cal.summarize(max(score(a, b) for a in As for b in Bs), np.array(manual), 0.05)
    assert r["s_obs"] == expect["s_obs"] and r["c_alpha"] == expect["c_alpha"]
    assert r["null_mean"] == pytest.approx(expect["null_mean"], abs=0)


def test_isotonic_quotient_refit_inside_permutation():
    """A fitted-quotient score receives the permuted Geom and refits."""
    from hrhv2 import quotients as qu
    from hrhv2 import util
    m = 30
    A = Geom.from_features(rand_features(m, 3, "cq1"), "none")
    B = Geom.from_features(rand_features(m, 3, "cq1") ** 3, "none")

    def iso_expl(a, b):
        x, y = util.upper_values(a.D), util.upper_values(b.D)
        _, _, fit = qu.iso_fit(x, y)
        return 1 - np.sum((y - fit) ** 2) / np.sum((y - x) ** 2)
    r = cal.calibrate(iso_expl, A, B, key="t|iso", K=10, return_null=True)
    pi = cal.permutation(m, "t|iso", 0)
    assert r["null"][0] == iso_expl(A, B.permuted(pi))
    assert r["s_obs"] > r["null"].max()
