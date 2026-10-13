"""Tests for the s11-select plant/S_sel helpers (no COCO, no S_sel metrics)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sel_common as S  # noqa: E402
from hrhv2 import plants  # noqa: E402


def test_gauss_E_rows_match_full_matrix():
    M, m_f, sigma, key = 300, 128, 0.7, "selTest|gE"
    E = plants.gauss_noise_matrix(M, sigma, key)
    rows = np.array([1, 5, 6, 60, 127])
    Er, Ef = S.gauss_E_rows(M, sigma, key, rows, m_f)
    assert np.array_equal(Er, E[rows])
    assert np.array_equal(Ef, E[:m_f, :m_f])


def test_dist_rows_close_to_full():
    X = plants.base_cloud(400, 3, "selTest|dr")
    D = plants.feature_dists(X)
    rows = np.array([0, 7, 399])
    Dr = S.dist_rows(X, rows)
    assert np.allclose(Dr, D[rows], atol=1e-13)
    assert np.all(Dr[np.arange(3), rows] == 0.0)


def test_plant_unit_shapes_and_halves():
    u = S.plant_unit("P-Smooth+noise", 2, 0, "selTest", M=2048, m_f=512)
    a = u["anchor_pos"]
    assert u["DA_rows"].shape == (a.size, 2048) and u["DA_fit"].shape == (512, 512)
    assert not u["h1_fit"][a].any() and np.all(a < 512)
    assert np.allclose(u["DA_rows"][:, :512], u["DA_fit"][a], atol=1e-12)
    assert np.allclose(u["DB_rows"][:, :512], u["DB_fit"][a], atol=1e-12)
    g = S.plant_unit("P-GaussSat", 2, 0, "selTest", M=1024, m_f=512)
    assert np.allclose(g["DB_rows"][:, :512] - g["DA_rows"][:, :512],
                       g["DB_fit"][g["anchor_pos"]] - g["DA_fit"][g["anchor_pos"]], atol=1e-12)


def test_plant_prefix_nested():
    u1 = S.plant_unit("P-Rough", 4, 1, "selTest", M=2048, m_f=512)
    u2 = S.plant_unit("P-Rough", 4, 1, "selTest", M=4096, m_f=512)
    assert np.allclose(u1["DA_rows"], u2["DA_rows"][:, :2048], atol=1e-12)
    assert np.allclose(u1["DB_rows"], u2["DB_rows"][:, :2048], atol=1e-12)


def test_ssel_pair_rule():
    assert sum(len(v) for v in S.SSEL_PAIRS.values()) == 12
    assert all(len(v) == 4 for v in S.SSEL_PAIRS.values())
    assert S.ssel_layer_index({"shape": [1023, 12, 768]}) == 6
    assert S.ssel_layer_index({"shape": [1023, 25, 2048]}) == 12
