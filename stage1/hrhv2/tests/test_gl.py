"""SPEC section 4: GL-invariant projector distance."""
import numpy as np
import pytest

from hrhv2 import gl
from conftest import rand_features


def _proj(X):
    Xc = X - X.mean(axis=0)
    return Xc @ np.linalg.pinv(Xc.T @ Xc) @ Xc.T


def test_gl_invariance_exact():
    XA = rand_features(80, 4, "gl1")
    G = rand_features(4, 4, "gl2") + 3 * np.eye(4)
    r = gl.projector_distance(XA, XA @ G + 7.0)  # + column shift (centering removes it)
    assert r["gl_projector_dist"] == pytest.approx(0.0, abs=1e-12)
    assert r["gl_cca_mean_sq"] == pytest.approx(1.0, abs=1e-12)


def test_orthogonal_subspaces_distance_one():
    Z = rand_features(60, 6, "gl3")
    Z = Z - Z.mean(axis=0)
    Q, _ = np.linalg.qr(Z)
    r = gl.projector_distance(Q[:, :3], Q[:, 3:6])
    assert r["gl_projector_dist"] == pytest.approx(1.0, abs=1e-12)
    assert r["gl_cca_mean_sq"] == pytest.approx(0.0, abs=1e-12)


def test_matches_explicit_projectors():
    XA = rand_features(50, 5, "gl4")
    XB = XA[:, :3] @ rand_features(3, 5, "gl5") + 0.7 * rand_features(50, 5, "gl6")
    r = gl.projector_distance(XA, XB)
    PA, PB = _proj(XA), _proj(XB)
    expect = np.sum((PA - PB) ** 2) / (2 * 5)
    assert r["gl_projector_dist"] == pytest.approx(expect, abs=1e-12)
    # equal full ranks: distance = 1 - mean squared canonical correlation
    assert r["gl_projector_dist"] == pytest.approx(1 - r["gl_cca_mean_sq"], abs=1e-12)
    assert r["gl_rank_A"] == 5 and r["gl_rank_B"] == 5


def test_requires_d_less_than_m():
    r = gl.projector_distance(rand_features(5, 5, "gl7"), rand_features(5, 3, "gl8"))
    assert np.isnan(r["gl_projector_dist"]) and "d >= m" in r["gl_note"]
