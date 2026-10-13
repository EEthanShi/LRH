"""SPEC section 2: geometric criteria."""
import numpy as np
import pytest
from scipy.stats import ortho_group

from hrhv2 import geometric as geo
from hrhv2 import structural as st
from hrhv2 import util
from hrhv2.prep import Geom, center_gram
from conftest import rand_features, rand_dist


def D3(a01, a02, a12):
    return np.array([[0, a01, a02], [a01, 0, a12], [a02, a12, 0]], dtype=float)


def test_gs_and_raw_g_hand():
    DA, DB = D3(1, 2, 3), D3(1, 1, 4)
    g = geo.gs(DA, DB)
    # nu_A = nu_B = 2 ; normalized (0.5,1,1.5) vs (0.5,0.5,2): diffs (0, .5, -.5)
    assert g["nu_A"] == 2.0 and g["nu_B"] == 2.0
    assert g["Gs_rms"] == pytest.approx(np.sqrt(0.5 / 3), abs=1e-15)
    assert g["Gs_max"] == 0.5
    r = geo.raw_g(DA, DB)
    assert r["G_rms"] == pytest.approx(np.sqrt(2 / 3), abs=1e-15) and r["G_max"] == 1.0
    # scale invariance of Gs
    assert geo.gs(DA, 7.0 * DB)["Gs_rms"] == pytest.approx(g["Gs_rms"], abs=1e-15)


def _cka_explicit(XA, XB):
    KA, KB = XA @ XA.T, XB @ XB.T
    m = KA.shape[0]
    H = np.eye(m) - 1.0 / m
    h = lambda K, L: np.trace(K @ H @ L @ H)
    return h(KA, KB) / np.sqrt(h(KA, KA) * h(KB, KB))


def _hsic_u_explicit(K, L):
    m = K.shape[0]
    Kt, Lt = K.copy(), L.copy()
    np.fill_diagonal(Kt, 0)
    np.fill_diagonal(Lt, 0)
    one = np.ones(m)
    return (np.trace(Kt @ Lt) + one @ Kt @ one * (one @ Lt @ one) / ((m - 1) * (m - 2))
            - 2.0 / (m - 2) * one @ Kt @ Lt @ one) / (m * (m - 3))


def test_cka_biased_and_unbiased_explicit():
    XA = rand_features(40, 6, "cka1")
    XB = XA @ rand_features(6, 9, "cka2") + 0.5 * rand_features(40, 9, "cka3")
    A, B = Geom.from_features(XA, "none"), Geom.from_features(XB, "none")
    r = geo.cka_pair(A, B)
    assert r["cka_biased"] == pytest.approx(_cka_explicit(XA, XB), abs=1e-13)
    KA, KB = A.K, B.K
    hab = _hsic_u_explicit(KA, KB)
    assert r["hsicu_AB"] == pytest.approx(hab, rel=1e-12)
    deb = hab / np.sqrt(_hsic_u_explicit(KA, KA) * _hsic_u_explicit(KB, KB))
    assert r["cka_debiased"] == pytest.approx(deb, abs=1e-12)


def test_cka_matches_vendored_prh_and_arh(prh, arh, torch):
    XA = rand_features(64, 12, "ckav1")
    XB = np.tanh(XA @ rand_features(12, 20, "ckav2")) + 0.3 * rand_features(64, 20, "ckav3")
    A, B = Geom.from_features(XA, "l2"), Geom.from_features(XB, "l2")
    r = geo.cka_pair(A, B)
    tA, tB = torch.tensor(A.X), torch.tensor(B.X)
    K, L = tA @ tA.T, tB @ tB.T
    for ns in (prh, arh):
        hu = lambda a, b: float(ns["hsic_unbiased"](a, b))
        hb = lambda a, b: float(ns["hsic_biased"](a, b))
        assert r["hsicu_AB"] == pytest.approx(hu(K, L), rel=1e-10, abs=1e-14)
        deb_ref = hu(K, L) / np.sqrt(hu(K, K) * hu(L, L))
        assert abs(r["cka_debiased"] - deb_ref) <= 1e-8
        bia_ref = hb(K, L) / np.sqrt(hb(K, K) * hb(L, L))
        assert abs(r["cka_biased"] - bia_ref) <= 1e-10
    # PRH's own cka() adds eps=1e-6 to the denominator; ARH adds 1e-8
    prh_deb = prh["cka"](tA, tB, unbiased=True)
    ours_prh_eps = r["hsicu_AB"] / (np.sqrt(r["hsicu_AA"] * r["hsicu_BB"]) + 1e-6)
    assert prh_deb == pytest.approx(ours_prh_eps, rel=1e-10)
    assert abs(r["cka_debiased"] - r["cka_debiased_arheps"]) < 1e-6


def test_cka_orthogonal_and_scale_invariance():
    XA = rand_features(50, 8, "ckai1")
    XB = XA[:, :5] ** 2 + rand_features(50, 5, "ckai2")
    Q = ortho_group.rvs(8, random_state=3)
    base = geo.cka_pair(Geom.from_features(XA, "none"), Geom.from_features(XB, "none"))
    rot = geo.cka_pair(Geom.from_features(XA @ Q * 3.7, "none"), Geom.from_features(0.01 * XB, "none"))
    for key in ("cka_biased", "cka_debiased"):
        assert rot[key] == pytest.approx(base[key], abs=1e-12)
    self_ = geo.cka_pair(Geom.from_features(XA, "none"), Geom.from_features(XA @ Q, "none"))
    assert self_["cka_biased"] == pytest.approx(1.0, abs=1e-13)
    assert self_["cka_debiased"] == pytest.approx(1.0, abs=1e-12)


def test_unbiased_hsic_dist_input_equivalence():
    """For distance-only inputs the unbiased HSIC uses -1/2 H D H; it equals the
    value from the Gram (U-statistic invariance to K_ij + a_i + a_j)."""
    XA = rand_features(30, 4, "hd1")
    XB = rand_features(30, 3, "hd2") + XA[:, :3]
    A, B = Geom.from_features(XA, "none"), Geom.from_features(XB, "none")
    Ad = Geom.from_dist(A.D)
    assert geo.hsic_unbiased(Ad.hsic_gram(), B.K) == pytest.approx(geo.hsic_unbiased(A.K, B.K), rel=1e-10)
    assert geo.cka_pair(Ad, B)["cka_debiased"] == pytest.approx(geo.cka_pair(A, B)["cka_debiased"], abs=1e-12)
    assert geo.cka_pair(Ad, B)["cka_biased"] == pytest.approx(geo.cka_pair(A, B)["cka_biased"], abs=1e-12)


def test_participation_ratio_limit():
    Kc = center_gram(np.eye(6))
    assert geo.participation_ratio(Kc) == pytest.approx(5.0, abs=1e-12)


def test_pearson_rsa_and_affine_invariance():
    DA, DB = rand_dist(45, "rsa1"), rand_dist(45, "rsa2") + 0.4 * rand_dist(45, "rsa1")
    x, y = util.upper_values(DA), util.upper_values(DB)
    assert geo.pearson_rsa(DA, DB) == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-13)
    DB2 = 3.0 * DB + 11.0
    np.fill_diagonal(DB2, 0.0)
    assert geo.pearson_rsa(DA, DB2) == pytest.approx(geo.pearson_rsa(DA, DB), abs=1e-13)


# ----------------------------------------------------------------------------
# SOG
# ----------------------------------------------------------------------------

def test_sog_scale_matches_full_matrix_quantile(sog, torch):
    D = rand_dist(37, "sog1")
    d = np.sqrt(D)
    full = d[d > 0]
    assert geo.sog_scale_from_D(D) == pytest.approx(np.quantile(full, 0.9), rel=1e-14)
    assert geo.sog_scale(d) == pytest.approx(np.quantile(full, 0.9), rel=1e-14)
    t = torch.tensor(d)
    ref = sog["_quantile_normalize"](t, q=0.9).numpy()
    assert np.allclose(d / geo.sog_scale_from_D(D), ref, rtol=1e-13, atol=0)
    nd = sog["_normalize_distance_array"](d, mode="quantile", q=0.9)
    assert np.allclose(d / geo.sog_scale_from_D(D), nd, rtol=1e-13, atol=0)


def test_sog_weight_hand():
    w = geo.sog_weights(np.array([1.0, 2.0]), np.array([np.e, 2.0]), 1.0, 1.0, tau=0.5)
    assert w.tolist() == pytest.approx([np.exp(-2.0), 1.0], abs=1e-15)


def _sog_brute(DA, DB, k, tau):
    m = DA.shape[0]
    dA, dB = np.sqrt(DA), np.sqrt(DB)
    sA, sB = np.quantile(dA[dA > 0], 0.9), np.quantile(dB[dB > 0], 0.9)
    nA, _ = st.knn_table(DA, k)
    nB, _ = st.knn_table(DB, k)
    s = 0.0
    for i in range(m):
        for j in set(nA[i]) & set(nB[i]):
            s += np.exp(-abs(np.log(dA[i, j] / sA) - np.log(dB[i, j] / sB)) / tau)
    knn = s / (m * k)
    TA, TB = st.mst_prim(DA), st.mst_prim(DB)
    s = 0.0
    for e in set(TA.tolist()) & set(TB.tolist()):
        i, j = divmod(e, m)
        s += np.exp(-abs(np.log(dA[i, j] / sA) - np.log(dB[i, j] / sB)) / tau)
    return knn, s / (m - 1)


def test_sog_overlaps_bruteforce_and_vendored(sog, torch):
    m, k = 40, 5
    DA = rand_dist(m, "sog2")
    DB = DA * np.exp(0.2 * rand_dist(m, "sog3") / rand_dist(m, "sog3").max())
    DB = 0.5 * (DB + DB.T)
    np.fill_diagonal(DB, 0)
    nA, _ = st.knn_table(DA, k)
    nB, _ = st.knn_table(DB, k)
    TA, TB = st.mst_prim(DA), st.mst_prim(DB)
    taus = (1.0, 0.1, 0.01)
    r = geo.sog_overlaps(DA, DB, nA, nB, TA, TB, k=k, taus=taus)
    dA, dB = np.sqrt(DA), np.sqrt(DB)
    nda = dA / geo.sog_scale_from_D(DA)
    ndb = dB / geo.sog_scale_from_D(DB)
    for tau in taus:
        tag = geo._tau_tag(tau)
        knn_b, mst_b = _sog_brute(DA, DB, k, tau)
        assert r[f"sog_knn{k}_tau{tag}"] == pytest.approx(knn_b, rel=1e-12)
        assert r[f"sog_mst_tau{tag}"] == pytest.approx(mst_b, rel=1e-12)
        ref_mst = sog["_mst_distance_overlap_similarity_from_cached"](nda, TA, ndb, TB, n=m, distance_tau=tau)
        assert r[f"sog_mst_tau{tag}"] == pytest.approx(ref_mst, rel=1e-12)
        rows = np.arange(m)[:, None]
        la = torch.tensor(np.log(np.maximum(nda[rows, nA], 1e-12)))
        lb = torch.tensor(np.log(np.maximum(ndb[rows, nB], 1e-12)))
        ref_knn = float(sog["_knn_distance_overlap_from_log_values"](
            torch.tensor(nA), torch.tensor(nB), la, lb, topk=k, tau=tau))
        assert r[f"sog_knn{k}_tau{tag}"] == pytest.approx(ref_knn, rel=1e-12)
    # identical representations: every weight is 1
    rr = geo.sog_overlaps(DA, DA, nA, nA, TA, TA, k=k, taus=taus)
    assert all(rr[f"sog_knn{k}_tau{geo._tau_tag(t)}"] == pytest.approx(1.0, abs=1e-15) for t in taus)
    assert all(rr[f"sog_mst_tau{geo._tau_tag(t)}"] == pytest.approx(1.0, abs=1e-15) for t in taus)


def _song_exact(K, L):
    """Song et al. unbiased HSIC in exact rational arithmetic (float64 inputs are dyadic)."""
    from fractions import Fraction
    m = K.shape[0]
    Kf = [[Fraction(float(K[i, j])) if i != j else Fraction(0) for j in range(m)] for i in range(m)]
    Lf = [[Fraction(float(L[i, j])) if i != j else Fraction(0) for j in range(m)] for i in range(m)]
    t1 = sum(Kf[i][j] * Lf[i][j] for i in range(m) for j in range(m))
    rK = [sum(r) for r in Kf]
    rL = [sum(r) for r in Lf]
    t3 = sum(a * b for a, b in zip(rK, rL))
    return (t1 + sum(rK) * sum(rL) / ((m - 1) * (m - 2)) - Fraction(2, m - 2) * t3) / (m * (m - 3))


def test_cka_debiased_accurate_on_ill_conditioned_gram():
    """Verifier fix F1: rows sharing a large common mean make the raw-Gram Song
    terms cancel heavily; evaluating on H K H keeps the unbiased HSIC and the
    debiased CKA accurate to <= 1e-12 relative vs exact rational arithmetic
    (the raw-Gram evaluation misses this by orders of magnitude)."""
    mu = 30.0 * rand_features(1, 10, "illc0")
    XA = rand_features(40, 10, "illc1") + mu
    XB = 0.4 * XA + rand_features(40, 10, "illc2") + 2 * mu
    A, B = Geom.from_features(XA, "none"), Geom.from_features(XB, "none")
    r = geo.cka_pair(A, B)
    hab = _song_exact(A.K, B.K)
    haa = _song_exact(A.K, A.K)
    hbb = _song_exact(B.K, B.K)
    assert abs(r["hsicu_AB"] - float(hab)) <= 1e-12 * abs(float(hab))
    deb = float(hab) / np.sqrt(float(haa) * float(hbb))
    assert abs(r["cka_debiased"] - deb) <= 1e-12 * abs(deb)
    # the old raw-Gram evaluation is far less accurate on this input
    raw = geo.hsic_unbiased(A.K, B.K)
    assert abs(raw - float(hab)) > 100 * abs(r["hsicu_AB"] - float(hab))
