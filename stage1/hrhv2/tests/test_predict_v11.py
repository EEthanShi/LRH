"""Tests for hrhv2.predict_v11 (PREREG v1.1 anchor-local P1 variants, D2 mode, nulls)."""
import numpy as np
import pytest

from hrhv2 import plants, util
from hrhv2 import predict as p10
from hrhv2 import predict_v11 as v


# ----------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------

def _mult_pair(M, p, key, sigma):
    """Torus cloud; D_B = D_A * exp(sigma * Zsym) (symmetric iid N(0,1) Zsym):
    given an anchor, b_l = u_l exp(sigma zeta_l) with independent columns, i.e.
    exactly variant (i)'s model (beta = gamma = 0, Gaussian zeta)."""
    DA = plants.feature_dists(plants.base_cloud(M, p, key))
    Z = plants.gauss_noise_matrix(M, 1.0, key + "|Z")
    DB = DA * np.exp(sigma * Z)
    return DA, DB


def _unit(DA, DB, m_f, salt="hrhv2-halves"):
    M = DA.shape[0]
    h1 = util.hash_half([str(i) for i in range(M)], salt)
    a = np.flatnonzero(~h1[:m_f])
    return dict(DA_rows=DA[a].copy(), DB_rows=DB[a].copy(), anchor_pos=a,
                DA_fit=DA[:m_f, :m_f].copy(), DB_fit=DB[:m_f, :m_f].copy(), h1_fit=h1[:m_f].copy())


PRED_KEYS = ("i-LSlog", "ii-IsoXF", "iii-LSlog-EB", "iv-v10", "N1", "i-LSlog|mu", "ii-IsoXF|mu",
             "iii-LSlog-EB|mu", "iv-v10|mu", "N1|mu", "N0", "N0H1")


@pytest.fixture(scope="module")
def small_unit():
    DA, DB = _mult_pair(512, 3, "v11test|mult", 0.15)
    return DA, DB, _unit(DA, DB, 256)


def _run(u, **kw):
    args = dict(m_grid=(128, 256, 512), m_f=256, ks=(1, 10), modes=("d2", "mu"))
    args.update(kw)
    return v.run_unit(u["DA_rows"], u["DB_rows"], u["anchor_pos"], u["DA_fit"], u["DB_fit"], u["h1_fit"], **args)


def _preds(res):
    return {(c["m"], c["k"]): {key: c[key] for key in PRED_KEYS if key in c} for c in res["cells"]}


# ----------------------------------------------------------------------------
# exact discrepancy
# ----------------------------------------------------------------------------

def test_exact_k1_matches_frozen_exact_loss_gauss():
    rng = util.rng_from_key("v11test|exact_gauss")
    U = np.sort(rng.random((7, 60)) * 3.0, axis=1)
    c, s, F, aux, csup = v.gauss_laws(U, 0.2)
    a = v.exact_loss_k1(c, s, F, aux, csup)
    b = p10.exact_loss(c, s, 1, F, None, aux=aux, col_support=csup, n_grid=v.EXACT_GRID)
    assert np.allclose(a, b, atol=1e-12, rtol=0)


def test_exact_k1_matches_frozen_exact_loss_shape_tables():
    rng = util.rng_from_key("v11test|exact_shape")
    n_a, n = 6, 80
    Z = rng.standard_t(4, size=(n_a, 150))
    shp = v.build_shape(Z)
    U = np.sort(rng.random((n_a, n)) * 2.0, axis=1)
    s = 0.05 + 0.1 * rng.random((n_a, n))
    aux = (np.broadcast_to(np.arange(n_a)[:, None], U.shape),)
    csup = (np.broadcast_to(shp.zlo[:, None], U.shape), np.broadcast_to(shp.zhi[:, None], U.shape))
    a = v.exact_loss_k1(U, s, shp.F, aux, csup)
    b = p10.exact_loss(U, s, 1, shp.F, None, aux=aux, col_support=csup, n_grid=v.EXACT_GRID)
    assert np.allclose(a, b, atol=1e-12, rtol=0)


def test_exact_k1_monte_carlo():
    """Exact k=1 discrepancy agrees with a Monte Carlo of the same independent law."""
    rng = util.rng_from_key("v11test|mc")
    U = np.sort(rng.random((3, 40)), axis=1)
    sig = 0.03
    c, s, F, aux, csup = v.gauss_laws(U, sig)
    ex = v.exact_loss_k1(c, s, F, aux, csup)
    n_mc = 40000
    b = U[:, None, :] + sig * rng.standard_normal((3, n_mc, 40))
    mc = (np.argmin(b, axis=2) != 0).mean(axis=1)
    assert np.all(np.abs(ex - mc) < 0.01)


# ----------------------------------------------------------------------------
# shape tables, EB, windows, tie rule
# ----------------------------------------------------------------------------

def test_shape_cdf_monotone_and_variance_preserving():
    rng = util.rng_from_key("v11test|shape")
    Z = rng.standard_normal((5, 400)) * np.array([[1.0], [0.5], [2.0], [1.0], [0.1]]) + 3.0
    shp = v.build_shape(Z)
    assert np.all(np.diff(shp.T, axis=1) >= 0)
    assert np.all(shp.T[:, 0] == 0.0) and np.all(shp.T[:, -1] == 1.0)
    mu, var = shp.moments()
    # atoms are 101 quantiles of the sample; KDE keeps their mean and variance
    pos = (np.arange(101) + 0.5) / 101 * 400 - 0.5
    Zs = np.sort(Z, axis=1)
    atoms = np.array([np.interp(pos, np.arange(400), z) for z in Zs])
    assert np.allclose(mu, atoms.mean(1), atol=2e-3 * atoms.std(1).max())
    assert np.allclose(var, atoms.var(1), rtol=0.02)
    aux = (np.arange(5),)
    assert np.allclose(shp.F(shp.zlo - 1.0, aux), 0.0) and np.allclose(shp.F(shp.zhi + 1.0, aux), 1.0)


def test_eb_shrink_full_when_no_heterogeneity():
    rng = util.rng_from_key("v11test|eb")
    se = np.full(200, 0.5)
    theta = 1.0 + se * rng.standard_normal(200)     # pure sampling noise around 1
    out, info = v.eb_shrink(theta, se)
    assert info["tau2"] < 0.05
    assert np.std(out) < 0.25 * np.std(theta)
    theta2 = np.concatenate((np.zeros(100), np.full(100, 5.0))) + 0.01 * rng.standard_normal(200)
    out2, info2 = v.eb_shrink(theta2, np.full(200, 0.01))
    assert info2["mean_w"] > 0.99 and np.allclose(out2, theta2, atol=1e-3)


@pytest.mark.parametrize("m,k", [(256, 1), (256, 10), (256, 100), (512, 1), (512, 100),
                                 (2048, 1), (2048, 10), (16384, 1), (16384, 100)])
def test_band_and_window(m, k):
    n_f = 1023
    rho = v.cell_rho(k, m, 1024)
    band = v.band_mask(rho, n_f)
    w = v.window_mask(rho, n_f, exclude=band)
    assert not np.any(w & band)
    assert w.sum() >= v.N_MIN
    r = np.arange(1, n_f + 1)
    if band.any():
        assert r[band].min() >= rho / 2 and r[band].max() <= 2 * rho
    w2 = v.window_mask(rho, n_f)
    assert w2.sum() >= v.N_MIN


def test_sort_rows_and_knn_tie_rule():
    D = np.array([[0.0, 1.0, 0.5, 0.5, 2.0, 0.5],
                  [1.0, 0.0, 3.0, 3.0, 3.0, 0.1]])
    U, COL, _ = v.sort_rows(D, np.array([0, 1]), 6)
    assert COL[0].tolist() == [2, 3, 5, 1, 4]
    assert COL[1].tolist() == [5, 0, 2, 3, 4]
    nn = v.knn_rows(D, np.array([0, 1]), 6, 3)
    assert nn[0].tolist() == [2, 3, 5] and nn[1].tolist() == [5, 0, 2]


# ----------------------------------------------------------------------------
# data-use guarantees (D4.2)
# ----------------------------------------------------------------------------

def test_no_B_entry_outside_fit_rows_is_read(small_unit):
    """Every predictor is unchanged when every B entry the v1.1 protocol may not
    read is perturbed: the anchors' B-rows beyond m_f, and DB_fit outside H1 x H1
    (variant iv / N1 read only H1 x H1; N0H1 reads H1 rows and is excluded)."""
    DA, DB, u = small_unit
    base = _preds(_run(u))
    u2 = {k: (val.copy() if isinstance(val, np.ndarray) else val) for k, val in u.items()}
    rng = util.rng_from_key("v11test|perturb")
    u2["DB_rows"][:, 256:] = rng.random(u2["DB_rows"][:, 256:].shape) * 5.0
    h1 = u["h1_fit"]
    outside = ~(h1[:, None] & h1[None, :])
    P = rng.random(outside.shape) * 5.0
    P = 0.5 * (P + P.T)
    u2["DB_fit"] = np.where(outside, P, u2["DB_fit"])
    np.fill_diagonal(u2["DB_fit"], 0.0)
    pert = _preds(_run(u2))
    for cell, vals in base.items():
        for key, val in vals.items():
            if key == "N0H1":
                continue
            assert pert[cell][key] == val, (cell, key)


def test_band_exclusion_variants_i_iii(small_unit):
    """For cell (m, k), variants i and iii ignore the B values of each anchor's
    band columns at the fit size."""
    DA, DB, u = small_unit
    m, k, m_f = 128, 10, 256
    base = _run(u, m_grid=(m,), ks=(k,), variants=("i-LSlog", "iii-LSlog-EB"), modes=("d2",))
    Fr = v.make_fit_rows(u["DA_rows"][:, :m_f], u["DB_rows"][:, :m_f], u["anchor_pos"], m_f)
    band = v.band_mask(v.cell_rho(k, m, m_f), Fr.n_f)
    assert band.sum() > 5
    u2 = dict(u)
    DBr = u["DB_rows"].copy()
    rng = util.rng_from_key("v11test|band")
    for r in range(DBr.shape[0]):
        cols = Fr.COL[r, band]
        DBr[r, cols] = DBr[r, cols] * np.exp(rng.standard_normal(cols.size))
    u2["DB_rows"] = DBr
    pert = _run(u2, m_grid=(m,), ks=(k,), variants=("i-LSlog", "iii-LSlog-EB"), modes=("d2",))
    c0, c1 = base["cells"][0], pert["cells"][0]
    assert c0["i-LSlog"] == c1["i-LSlog"] and c0["iii-LSlog-EB"] == c1["iii-LSlog-EB"]
    assert c0["obs"] != c1["obs"]


def test_isoxf_centre_does_not_use_own_b(small_unit):
    DA, DB, u = small_unit
    m_f = 256
    Fr = v.make_fit_rows(u["DA_rows"][:, :m_f], u["DB_rows"][:, :m_f], u["anchor_pos"], m_f)
    links = v.fit_iso_links(Fr)
    ri = np.arange(Fr.U.shape[0])
    cen = v.iso_centres(links, ri, Fr.U, Fr.COL)
    for t in (0, 1, 7, 100):
        B2 = Fr.B.copy()
        B2[:, t] += 10.0
        Fr2 = v.FitRows(U=Fr.U, B=B2, COL=Fr.COL, logU=Fr.logU, logB=v.pos_floor_log(B2), m_f=m_f)
        cen2 = v.iso_centres(v.fit_iso_links(Fr2), ri, Fr.U, Fr.COL)
        assert np.array_equal(cen[:, t], cen2[:, t])


def test_n1_undefined_falls_back_to_n0():
    DA = plants.feature_dists(plants.base_cloud(300, 2, "v11test|neg"))
    DB = DA.max() - DA                       # global affine slope < 0
    np.fill_diagonal(DB, 0.0)
    u = _unit(DA, DB, 200)
    res = v.run_unit(u["DA_rows"], u["DB_rows"], u["anchor_pos"], u["DA_fit"], u["DB_fit"], u["h1_fit"],
                     m_grid=(150, 300), m_f=200, ks=(1, 10), variants=("i-LSlog",), modes=("d2", "mu"))
    assert res["diag"]["N1_defined"] is False
    for c in res["cells"]:
        assert c["N1"] == c["N0"] and c["N1|mu"] == c["N0"]


def test_fit_m_obs_equals_N0(small_unit):
    DA, DB, u = small_unit
    res = _run(u, variants=("i-LSlog",))
    for c in res["cells"]:
        if c["m"] == 256:
            assert c["is_fit_m"] and c["obs"] == c["N0"]


def test_mu_and_d2_identical_above_exact_kmax(small_unit):
    DA, DB, u = small_unit
    res = _run(u, variants=("i-LSlog",))
    for c in res["cells"]:
        if c["k"] > v.EXACT_K_MAX:
            assert c["i-LSlog"] == c["i-LSlog|mu"]


# ----------------------------------------------------------------------------
# recovery on a coupling that variant (i) specifies exactly
# ----------------------------------------------------------------------------

def test_variant_i_recovers_multiplicative_coupling():
    DA, DB = _mult_pair(2048, 4, "v11test|recover", 0.06)
    u = _unit(DA, DB, 512)
    res = v.run_unit(u["DA_rows"], u["DB_rows"], u["anchor_pos"], u["DA_fit"], u["DB_fit"], u["h1_fit"],
                     m_grid=(256, 2048), m_f=512, ks=(1, 10), variants=("i-LSlog", "iii-LSlog-EB"),
                     modes=("d2",))
    for c in res["cells"]:
        se = c["obs_sd"] / np.sqrt(c["n_anchors"])
        for key in ("i-LSlog", "iii-LSlog-EB"):
            assert abs(c[key] - c["obs"]) < 0.03 + 3 * se, (c["m"], c["k"], key, c[key], c["obs"])
