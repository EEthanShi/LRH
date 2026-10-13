"""SPEC section 7 plant generators, the ladder, and the run_pair CLI."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from hrhv2 import geometric as geo
from hrhv2 import plants, quotients as qu, structural as st, util
from hrhv2.ladder import ladder
from hrhv2.prep import Geom
from conftest import ROOT, rand_features


def test_clifford_torus_closed_form_and_isometry():
    z = plants.torus_z(50, 3, "pl1")
    D = plants.feature_dists(plants.embed(plants.clifford(z), "pl1"))
    assert np.allclose(D, plants.torus_dist(z), atol=1e-14)
    Q = plants.random_isometry(6, "pl2")
    assert np.allclose(Q.T @ Q, np.eye(6), atol=1e-14) and Q.shape == (64, 6)


def test_nested_prefix_pools():
    big = plants.base_cloud(512, 4, "nest")
    small = plants.base_cloud(128, 4, "nest")
    assert np.array_equal(big[:128], small)
    h_big = util.hash_half(plants.nested_ids(512), "hrhv2-halves")
    h_small = util.hash_half(plants.nested_ids(128), "hrhv2-halves")
    assert np.array_equal(h_big[:128], h_small)
    assert 0.4 < h_big.mean() < 0.6


@pytest.mark.parametrize("p", [2, 4])
def test_bernstein_plant_ordinal_identity_and_cka_path(p):
    m = 200
    KA, KBs, info = plants.bern_grams(m, p, (1, 3, 10), f"bern|p={p}")
    A = Geom.from_gram(KA)
    ckas = []
    for beta in (1, 3, 10):
        assert info[beta]["float_collisions_D"] == 0 and info[beta]["float_collisions_K"] == 0
        B = Geom.from_gram(KBs[beta])
        r = ladder(A, B, with_gl=False)
        for k in st.default_k_grid(m):
            assert r[f"Delta_N_k{k}"] == 0.0
        for k in (1, 3, 10):
            assert r[f"comp_rand_k{k}"] == 0.0
        assert r["Delta_O"] == 0.0 and r["Delta_E"] == 0.0 and r["s_T"] == 1.0
        assert r["quot_mean_iso_expl"] >= 0.999 and r["quot_mean_iso_expl_cf"] >= 0.999
        ckas.append((r["cka_biased"], r["cka_debiased"]))
    assert info["float_collisions_DA"] == 0
    for j in (0, 1):
        assert ckas[0][j] > ckas[1][j] > ckas[2][j]


def test_ladder_identical_pair():
    X = rand_features(120, 10, "lad1")
    A = Geom.from_features(X, "l2")
    r = ladder(A, A)
    for k in st.default_k_grid(120):
        assert r[f"s_N_k{k}"] == 1.0
    assert r["Delta_O"] == 0.0 and r["Delta_E"] == 0.0 and r["s_T"] == 1.0
    assert r["cka_biased"] == pytest.approx(1.0, abs=1e-12) and r["Gs_rms"] == 0.0
    assert r["rsa_pearson"] == pytest.approx(1.0, abs=1e-12)
    assert r["sog_knn10_tau0p01"] == pytest.approx(1.0, abs=1e-15)
    assert r["gl_projector_dist"] == pytest.approx(0.0, abs=1e-12)
    assert r["forest_in_mst_A"] and r["n_H1"] == int(util.hash_half([str(i) for i in range(120)], "hrhv2-halves").sum())


def test_warpshare_plant_recovered():
    DA, DB, planted = plants.warpshare_plant(300, 3, 0.15, "ws")
    assert 0.0 < planted < 1.0
    xa, xb = util.upper_values(DA), util.upper_values(DB)
    _, _, fit = qu.iso_fit(xa, xb)
    est = 1 - np.sum((xb - fit) ** 2) / np.sum((xb - xa) ** 2)
    assert abs(est - planted) < 0.05


def test_run_pair_cli(tmp_path):
    m, L, d = 90, 3, 12
    XA = rand_features(m * L, d, "cli1").reshape(m, L, d) ** 3
    XB = rand_features(m * L, d, "cli2").reshape(m, L, d) + XA
    pa, pb = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(pa, XA.astype(np.float32))
    np.save(pb, XB.astype(np.float32))
    out = tmp_path / "pair.json"
    cmd = [sys.executable, "-m", "hrhv2.run_pair", "--a", str(pa), "--b", str(pb), "--pre", "prh_clip+l2",
           "--layer-a", "1", "--layer-b", "-1", "--k-grid", "1,3,10", "--calib-k", "3", "--calib-K", "20",
           "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT})
    res = json.loads(out.read_text())
    A = Geom.from_features(XA.astype(np.float32)[:, 1, :].astype(np.float64), "prh_clip+l2",
                           clip_ref=XA.astype(np.float32).astype(np.float64))
    B = Geom.from_features(XB.astype(np.float32)[:, -1, :].astype(np.float64), "prh_clip+l2",
                           clip_ref=XB.astype(np.float32).astype(np.float64))
    r = ladder(A, B, k_grid=[1, 3, 10])
    for key in ("s_N_k1", "s_N_k3", "s_N_k10", "Delta_O", "Delta_E", "s_T", "cka_debiased",
                "quot_mean_iso_expl_cf", "sog_mst_tau0p01"):
        assert res["ladder"][key] == pytest.approx(r[key], abs=1e-14), key
    assert res["inputs"]["A"]["layer"] == 1 and res["inputs"]["B"]["layer"] == -1
    assert res["calibration_knn"]["k3"]["K"] == 20
    assert res["inputs"]["A"]["sha256"] and res["ladder"]["spec_version"] == "1.0"


def test_run_pair_rows_and_prefix(tmp_path):
    m, d = 70, 8
    XA = rand_features(m, d, "cli3")
    XB = XA + 0.5 * rand_features(m, d, "cli4")
    pa, pb = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(pa, XA)
    np.save(pb, XB)
    ids = tmp_path / "ids.txt"
    ids.write_text("\n".join(f"item{i}" for i in range(m)))
    rows = np.arange(m)[::-1][:50].copy()
    np.save(tmp_path / "rows.npy", rows)
    from hrhv2 import run_pair
    out = run_pair.main(["--a", str(pa), "--b", str(pb), "--pre", "l2", "--rows", str(tmp_path / "rows.npy"),
                         "--item-ids", str(ids), "--k-grid", "1,3", "--no-gl", "--out", str(tmp_path / "o.json")])
    A, B = Geom.from_features(XA[rows], "l2"), Geom.from_features(XB[rows], "l2")
    r = ladder(A, B, k_grid=[1, 3], item_ids=[f"item{i}" for i in rows], with_gl=False)
    assert out["ladder"]["s_N_k3"] == r["s_N_k3"] and out["ladder"]["quot_mean_iso_expl_cf"] == r["quot_mean_iso_expl_cf"]
    assert out["inputs"]["A"]["m_used"] == 50
    out2 = run_pair.main(["--a", str(pa), "--b", str(pb), "--pre", "l2", "--m", "40", "--k-grid", "1",
                          "--no-gl", "--no-quotients", "--out", str(tmp_path / "o2.json")])
    assert out2["ladder"]["m"] == 40
