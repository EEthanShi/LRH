"""SPEC section 3: quotient (warp) decomposition and cross-fitting."""
import numpy as np
import pytest

from hrhv2 import quotients as qu
from hrhv2 import util
from conftest import rand_dist


def offdiag(f, D):
    out = f(D)
    np.fill_diagonal(out, 0.0)
    return out


def _h1(m):
    return util.hash_half([str(i) for i in range(m)], "hrhv2-halves")


def test_isotonic_hand_with_ties():
    x = np.array([1.0, 1.0, 2.0, 3.0])
    y = np.array([2.0, 0.0, 3.0, 1.0])
    # tie pooling: x=1 -> mean 1 (w=2); x=2 -> 3; x=3 -> 1; PAVA pools the last two -> 2
    kx, kg, fit = qu.iso_fit(x, y)
    assert fit.tolist() == [1.0, 1.0, 2.0, 2.0]
    assert kx.tolist() == [1.0, 2.0, 3.0] and kg.tolist() == [1.0, 2.0, 2.0]
    # unsorted input order is respected
    o = np.array([3, 0, 2, 1])
    _, _, fit2 = qu.iso_fit(x[o], y[o])
    assert fit2.tolist() == fit[o].tolist()


def test_iso_predict_interp_and_constant_extrapolation():
    kx, kg = np.array([0.0, 1.0]), np.array([0.0, 2.0])
    assert qu.iso_predict(kx, kg, np.array([-1.0, 0.5, 2.0])).tolist() == [0.0, 1.0, 2.0]


def test_planted_quotients_explained_exactly():
    m = 60
    DA = rand_dist(m, "q1")
    h1 = _h1(m)
    cases = {
        "scale": offdiag(lambda D: 3.0 * D, DA),
        "affine": offdiag(lambda D: 2.0 * D + 5.0, DA),
        "iso": offdiag(lambda D: D ** 3 + np.exp(D / D.max()), DA),
    }
    order = ["scale", "affine", "iso", "rowiso"]
    for name, DB in cases.items():
        r = qu.quotient_ladder(DA, DB, h1)
        lvl = order.index(name)
        for q in order[lvl:]:  # the planted quotient and every coarser one
            for d in ("AB", "BA") if name != "affine" else ("AB",):
                assert r[f"quot_{d}_{q}_expl"] == pytest.approx(1.0, abs=1e-12), (name, q, d)
        for q in order[:lvl]:  # finer quotients cannot explain everything
            assert r[f"quot_AB_{q}_expl"] < 0.999, (name, q)
        assert r["quot_AB_iso_expl_cf"] > 0.98
        assert r["quot_AB_iso_rms"] == pytest.approx(0.0, abs=1e-9)


def test_row_isotonic_planted_row_warps():
    m = 40
    DA = rand_dist(m, "q2")
    DB = DA.copy()
    for i in range(m):
        DB[i] = (1.0 + i) * DA[i] ** (1.0 + 2.0 * i / m)
    h1 = _h1(m)
    r = qu.row_isotonic(DA, DB, h1)
    assert r["AB"]["rowiso_expl"] == pytest.approx(1.0, abs=1e-12)
    assert r["AB"]["rowiso_max"] == pytest.approx(0.0, abs=1e-9)
    # the global isotonic quotient cannot absorb row-specific warps
    xa, xb = util.upper_values(DA), util.upper_values(DB)
    _, _, fit = qu.iso_fit(xa, xb)
    assert 1 - np.sum((xb - fit) ** 2) / np.sum((xb - xa) ** 2) < 0.99
    # parallel == serial
    r2 = qu.row_isotonic(DA, DB, h1, n_jobs=3)
    for d in ("AB", "BA"):
        for key, v in r[d].items():
            assert r2[d][key] == pytest.approx(v, rel=1e-12, abs=1e-15)


def test_explained_fraction_definition_hand():
    # 3 points: x = (1,2,3), y = (2,1,3) on pairs (01,02,12)
    DA = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], float)
    DB = np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0]], float)
    x, y = np.array([1., 2., 3.]), np.array([2., 1., 3.])
    h1 = np.array([True, True, False])
    g = qu.global_quotients_dir(x, y, x, y, x, y)
    ss0 = np.sum((y - x) ** 2)                       # = 2
    c = np.dot(x, y) / np.dot(x, x)                  # = 13/14
    assert g["ss0"] == ss0
    assert g["scale_expl"] == pytest.approx(1 - np.sum((y - c * x) ** 2) / ss0, abs=1e-15)
    # isotonic: (2,1) pooled to 1.5 -> residuals (.5,-.5,0) -> 1 - 0.5/2
    assert g["iso_expl"] == pytest.approx(0.75, abs=1e-15)
    r = qu.quotient_ladder(DA, DB, h1)
    assert r["quot_AB_iso_expl"] == pytest.approx(0.75, abs=1e-15)
    assert np.isnan(r["quot_AB_iso_expl_cf"])  # H2 x H2 is empty here


def test_crossfit_uses_only_H1H1_for_fit_and_H2H2_for_evaluation():
    m = 50
    DA = rand_dist(m, "q3")
    DB = offdiag(lambda D: np.sqrt(D) + 0.3 * rand_dist(m, "q4"), DA)
    DB = 0.5 * (DB + DB.T)
    h1 = _h1(m)
    base = qu.quotient_ladder(DA, DB, h1, with_row=False)
    cf_keys = [k for k in base if k.endswith("_cf") and "rowiso" not in k]
    assert cf_keys
    # perturb every mixed (H1,H2) pair: cross-fitted global quotients unchanged
    P = DB.copy()
    mixed = np.outer(h1, ~h1) | np.outer(~h1, h1)
    P[mixed] *= 1.7
    pert = qu.quotient_ladder(DA, P, h1, with_row=False)
    for k in cf_keys:
        assert pert[k] == base[k], k
    # perturbing an H2xH2 pair changes the evaluation, an H1xH1 pair the fit
    i2 = np.flatnonzero(~h1)[:2]
    P2 = DB.copy(); P2[i2[0], i2[1]] += 1.0; P2[i2[1], i2[0]] += 1.0
    assert qu.quotient_ladder(DA, P2, h1, with_row=False)["quot_AB_affine_expl_cf"] != base["quot_AB_affine_expl_cf"]
    i1 = np.flatnonzero(h1)[:2]
    P1 = DB.copy(); P1[i1[0], i1[1]] += 1.0; P1[i1[1], i1[0]] += 1.0
    assert qu.quotient_ladder(DA, P1, h1, with_row=False)["quot_AB_affine_expl_cf"] != base["quot_AB_affine_expl_cf"]


def test_row_crossfit_columns_split():
    """Row-isotonic cross-fit (interpretation in SPEC_INTERPRETATIONS.md): g_i
    is fitted on columns in H1, evaluated on columns in H2. Changing B on
    H2-columns of a row must not change that row's fit, only its evaluation."""
    m = 30
    DA = rand_dist(m, "q5")
    DB = offdiag(lambda D: D ** 2, DA) + 0.01 * rand_dist(m, "q6")
    h1 = _h1(m)
    # manual computation for direction AB
    ss, ss0 = 0.0, 0.0
    for i in range(m):
        cols1 = [j for j in range(m) if j != i and h1[j]]
        cols2 = [j for j in range(m) if j != i and not h1[j]]
        kx, kg, _ = qu.iso_fit(DA[i, cols1], DB[i, cols1])
        r = DB[i, cols2] - qu.iso_predict(kx, kg, DA[i, cols2])
        ss += np.dot(r, r)
        d = DB[i, cols2] - DA[i, cols2]
        ss0 += np.dot(d, d)
    out = qu.row_isotonic(DA, DB, h1)
    assert out["AB"]["rowiso_expl_cf"] == pytest.approx(1 - ss / ss0, abs=1e-12)


def test_directions_swap():
    m = 35
    DA, DB = rand_dist(m, "q7"), rand_dist(m, "q8")
    h1 = _h1(m)
    r1 = qu.quotient_ladder(DA, DB, h1)
    r2 = qu.quotient_ladder(DB, DA, h1)
    for q in ("scale", "affine", "iso", "rowiso"):
        assert r1[f"quot_AB_{q}_expl"] == pytest.approx(r2[f"quot_BA_{q}_expl"], abs=1e-14)
        assert r1[f"quot_mean_{q}_expl"] == pytest.approx(r2[f"quot_mean_{q}_expl"], abs=1e-14)
    # warp shares telescope to 1
    tot = sum(r1[f"warpshare_expl_{p}"] for p in ("scale", "affine_extra", "iso_extra", "row_extra", "residual"))
    assert tot == pytest.approx(1.0, abs=1e-12)
