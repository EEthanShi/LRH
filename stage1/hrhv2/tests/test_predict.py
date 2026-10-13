"""SPEC section 6: predictors (crossing law, dispersion law, nulls)."""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import ndtr

from hrhv2 import plants, predict, structural as st, util


def _row_mu_brentq(u, s, k):
    """Reference: scalar root of E X(t) = E Y(t) for b ~ N(u, s^2), all columns."""
    uA, sA, uN, sN = u[:k], s[:k], u[k:], s[k:]
    g = lambda t: (1 - ndtr((t - uA) / sA)).sum() - ndtr((t - uN) / sN).sum()
    t = brentq(g, u.min() - 50 * s.max(), u.max() + 50 * s.max(), xtol=1e-14, rtol=1e-14)
    return (1 - ndtr((t - uA) / sA)).sum()


def test_mu_balance_matches_brentq_and_truncation_is_exact():
    rng = util.rng_from_key("mu1")
    U = np.sort(rng.random((12, 300)) ** 0.5, axis=1)
    S = 0.01 + 0.03 * rng.random((12, 300))
    for k in (1, 3, 10, 50):
        mu = predict.mu_balance(U, S, k, predict._gauss_F, predict.GAUSS_SD_SUPPORT)
        ref = np.array([_row_mu_brentq(U[i], S[i], k) for i in range(12)])
        assert np.max(np.abs(mu - ref)) < 1e-9
    # no noise -> no crossings
    mu0 = predict.mu_balance(U, np.full_like(U, 1e-12), 5, predict._gauss_F, predict.GAUSS_SD_SUPPORT)
    assert np.max(mu0) < 1e-6
    # k >= number of columns -> 0
    assert np.all(predict.mu_balance(U[:, :5], S[:, :5], 5, predict._gauss_F, predict.GAUSS_SD_SUPPORT) == 0)


def test_mu_law_two_sided_bound_monte_carlo():
    """Thm 2: |E|N_A \\ N_B| - mu| <= sqrt(mu/2) under independent laws."""
    rng = util.rng_from_key("mu2")
    u = np.sort(rng.random(400))
    s = np.full_like(u, 0.004)
    for k in (1, 5, 20):
        mu = predict.mu_balance(u[None], s[None], k, predict._gauss_F, predict.GAUSS_SD_SUPPORT)[0]
        lost = []
        for _ in range(3000):
            b = u + s * rng.standard_normal(u.size)
            lost.append(k - np.sum(np.argsort(b, kind="stable")[:k] < k))
        e = np.mean(lost)
        assert abs(e - mu) <= np.sqrt(mu / 2) + 3 * np.std(lost) / np.sqrt(3000)


def test_mu_invariant_to_common_shift_and_scale():
    rng = util.rng_from_key("mu3")
    U = np.sort(rng.random((5, 200)), axis=1)
    S = 0.02 * np.ones_like(U)
    m1 = predict.mu_balance(U, S, 7, predict._gauss_F, predict.GAUSS_SD_SUPPORT)
    m2 = predict.mu_balance(3 * U + 10, 3 * S, 7, predict._gauss_F, predict.GAUSS_SD_SUPPORT)
    assert np.max(np.abs(m1 - m2)) < 1e-9


def test_levina_bickel_on_known_dimension():
    for p in (2, 4):
        D = plants.feature_dists(plants.base_cloud(3000, p, f"lb{p}"))  # flat torus, no boundary
        _, v = st.knn_table(D, 20)
        lb = predict.lb_dimension(v, 20)
        assert lb["mg"] == pytest.approx(p, rel=0.15)
        assert lb["n_used"] == 3000
    # hand example: one anchor with T = (1, 2, 4), k = 3 ->
    # [ (log(4/1) + log(4/2)) / 2 ]^{-1} = 2 / log 8
    v = np.array([[1.0, 4.0, 16.0]])
    assert predict.lb_dimension(v, 3)["lb"] == pytest.approx(2 / np.log(8), abs=1e-15)


def test_parse_variant():
    assert predict.parse_variant("P1-u-E") == ("u", "emp", "all", False)
    assert predict.parse_variant("P1-med-B-r8") == ("med", "bin", "r8", False)
    assert predict.parse_variant("P1-med-B-difff") == ("med", "bin", "diff", True)
    with pytest.raises(ValueError):
        predict.parse_variant("P1-x-E")
    with pytest.raises(ValueError):
        predict.parse_variant("P1-u-E-zz")


def _gauss_pair(m, p, c, key):
    DA = plants.feature_dists(plants.base_cloud(m, p, key))
    sig = c * plants.knn_d_median(DA, 10)
    return DA, DA + plants.gauss_noise_matrix(m, sig, key), sig


@pytest.mark.parametrize("variant", ["P1-u-E-all", "P1-med-B-r8", "P1-u-B-diff", "P1-q-E-r8"])
def test_fit_coupling_has_no_leakage(variant):
    """Only pairs with BOTH endpoints in the fit mask may influence the fit."""
    m = 240
    DA, DB, _ = _gauss_pair(m, 3, 0.1, "leak")
    h1 = util.hash_half([str(i) for i in range(m)], "hrhv2-halves")
    base = predict.fit_coupling(DA, DB, h1, variant=variant)
    P = DB.copy()
    out = ~np.outer(h1, h1)          # every pair touching H2
    P[out] = util.rng_from_key("leak2").random(out.sum()) * 5
    P = np.triu(P, 1)
    P = P + P.T
    P[np.outer(h1, h1)] = DB[np.outer(h1, h1)]
    DA2 = DA.copy()
    DA2[out] = DA2[out] * 0 + 7.0     # A outside H1xH1 is irrelevant too
    np.fill_diagonal(DA2, 0)
    pert = predict.fit_coupling(DA2, P, h1, variant=variant)
    for f in ("xc", "loc", "mad", "cnt", "zq", "cdf_table", "shape_xc"):
        a, b = getattr(base, f), getattr(pert, f)
        assert (a is None and b is None) or np.array_equal(a, b), f
    assert base.lo_slope == pert.lo_slope and base.hi_slope == pert.hi_slope
    # the nulls / P2 fits obey the same restriction
    assert predict.fit_n1(DA, DB, h1) == predict.fit_n1(DA2, P, h1)
    assert np.array_equal(predict.fit_n2(DA, DB, h1), predict.fit_n2(DA2, P, h1))
    d1, d2 = predict.fit_iso_dispersion(DA, DB, h1), predict.fit_iso_dispersion(DA2, P, h1)
    assert np.array_equal(d1.mad, d2.mad) and np.array_equal(d1.xc, d2.xc)


def test_crossfit_predict_observed_and_no_B_rows_of_targets():
    m = 512
    DA, DB, sig = _gauss_pair(m, 3, 0.1, "cf")
    h1 = util.hash_half([str(i) for i in range(m)], "hrhv2-halves")
    res = predict.crossfit_predict(DA, DB, h1, m_fit=256, m_list=[256, 512], ks=[3, 10],
                                   variants=["P1-u-E-r8"], oracle_sigma=sig)
    anchors = np.flatnonzero(~h1[:512])
    nA, _ = st.knn_table(DA, 10)
    nB, _ = st.knn_table(DB, 10)
    assert res[512][10]["obs"] == pytest.approx(st.overlap_counts(nA[anchors], nB[anchors], 10).mean() / 10)
    assert res[512][10]["n_anchors"] == anchors.size
    # predictions do not depend on any B entry outside H1 x H1 of the fit prefix
    P = DB.copy()
    fitpairs = np.zeros((m, m), dtype=bool)
    fitpairs[:256, :256] = np.outer(h1[:256], h1[:256])
    sel = np.triu(~fitpairs, 1)
    P[sel] += 0.5 * sig
    P = np.triu(P, 1)
    P = P + P.T
    res2 = predict.crossfit_predict(DA, P, h1, m_fit=256, m_list=[256, 512], ks=[3, 10],
                                    variants=["P1-u-E-r8"], oracle_sigma=sig)
    for mm in (256, 512):
        for k in (3, 10):
            assert res2[mm][k]["P1-u-E-r8"] == res[mm][k]["P1-u-E-r8"]
            assert res2[mm][k]["N1"] == res[mm][k]["N1"]
            assert res2[mm][k]["P2"] == res[mm][k]["P2"]


def test_p1_close_to_oracle_on_small_gauss_plant():
    m = 1024
    DA, DB, sig = _gauss_pair(m, 4, 0.15, "small")
    h1 = util.hash_half([str(i) for i in range(m)], "hrhv2-halves")
    res = predict.crossfit_predict(DA, DB, h1, m_fit=1024, m_list=[1024], ks=[10, 30],
                                   variants=["P1-u-E-r8", "P1-med-B-r8"], oracle_sigma=sig,
                                   with_nulls=False)
    for k in (10, 30):
        c = res[1024][k]
        for v in ("P1-u-E-r8", "P1-med-B-r8"):
            assert abs(c[v] - c["oracle"]) < 0.02
        assert abs(c["oracle"] - c["obs"]) < 0.03


def test_n1_recovers_sigma_on_gauss_plant():
    m = 600
    DA, DB, sig = _gauss_pair(m, 3, 0.2, "n1")
    h1 = util.hash_half([str(i) for i in range(m)], "hrhv2-halves")
    n1 = predict.fit_n1(DA, DB, h1)
    assert n1["slope"] == pytest.approx(1.0, abs=0.02)
    assert n1["sigma_A"] == pytest.approx(sig, rel=0.05)


def test_exact_loss_matches_monte_carlo():
    rng = util.rng_from_key("ex1")
    u = np.sort(rng.random(250))
    s = np.full_like(u, 0.005)
    for k in (1, 3, 10):
        ex = predict.exact_loss(u[None], s[None], k, predict._gauss_F, predict.GAUSS_SD_SUPPORT)[0]
        mu = predict.mu_balance(u[None], s[None], k, predict._gauss_F, predict.GAUSS_SD_SUPPORT)[0]
        lost = []
        for _ in range(4000):
            b = u + s * rng.standard_normal(u.size)
            lost.append(k - np.sum(np.argsort(b, kind="stable")[:k] < k))
        se = np.std(lost) / np.sqrt(4000)
        assert abs(ex - np.mean(lost)) <= 4 * se + 0.005, (k, ex, np.mean(lost), mu)


def test_exact_loss_k1_closed_form():
    # two candidates: P(b_1 > b_2) for b_j ~ N(u_j, s^2) = Phi(-(u2-u1)/(s sqrt2))
    u = np.array([[0.0, 1.0]])
    s = np.array([[0.7, 0.7]])
    ex = predict.exact_loss(u, s, 1, predict._gauss_F, predict.GAUSS_SD_SUPPORT, n_grid=4000)[0]
    assert ex == pytest.approx(ndtr(-1.0 / (0.7 * np.sqrt(2))), abs=2e-6)
