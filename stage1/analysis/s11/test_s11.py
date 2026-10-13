"""Tests for the S1.1 pipeline (synthetic data and metadata only; no feature array is read).

    cd analysis/s11 && ../../venv/bin/python -m pytest -q test_s11.py
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import s11_common as C  # noqa: E402
from s11_core import knn_rows, run_pair_draw  # noqa: E402
import s11_evaluate as E  # noqa: E402
from hrhv2 import plants, predict as pr, structural as st, util  # noqa: E402


def _rough(m=1536, p=4, rep=0):
    key = f"S11test|rough|{rep}"
    base = plants.feature_dists(plants.base_cloud(256, p, f"{key}|ref"))
    XA, XB = plants.rough_features(m, p, key, 0.3 * plants.knn_d_median(base, 10))
    return plants.feature_dists(XA), plants.feature_dists(XB)


def test_knn_rows_matches_knn_table_with_ties():
    rng = util.rng_from_key("S11test|ties")
    X = np.round(rng.standard_normal((300, 3)) * 2) / 2          # quantised -> many exact ties
    X[17] = X[5]                                                  # an exact duplicate item
    D = plants.feature_dists(X)
    nT, vT = st.knn_table(D, 31)
    rows = np.array([0, 5, 17, 18, 299, 150])
    nR, vR = knn_rows(D, rows, 31, chunk=4)
    assert np.array_equal(nR, nT[rows]) and np.array_equal(vR, vT[rows])


@pytest.mark.parametrize("ident", [True, False])
def test_matches_crossfit_predict(ident):
    """The S1.1 core reproduces hrhv2.predict.crossfit_predict (the protocol G0
    froze) exactly, for the identity order and for a hash-permuted draw."""
    DA, DB = _rough()
    m_all = DA.shape[0]
    ids = [str(i) for i in range(m_all)]
    h1 = util.hash_half(ids, C.HALVES_SALT)
    order = np.arange(m_all) if ident else util.hash_order(ids, "S11test|draw")
    m_list, m_fit, ks = (256, 512, 1536), 512, (1, 10, 100)
    v = pr.FROZEN_P1_VARIANT
    rows, _ = run_pair_draw(DA, DB, order, h1, m_grid=m_list, m_fit=m_fit, ks=ks, variant=v,
                            directions=("AB", "BA"))
    ref = pr.crossfit_predict(DA[np.ix_(order, order)], DB[np.ix_(order, order)], h1[order],
                              m_fit=m_fit, m_list=m_list, ks=ks, variants=(v,))
    refBA = pr.crossfit_predict(DB[np.ix_(order, order)], DA[np.ix_(order, order)], h1[order],
                                m_fit=m_fit, m_list=m_list, ks=ks, variants=(v,))
    assert len(rows) == 9
    for r in rows:
        cell = ref[r["m"]][r["k"]]
        cellBA = refBA[r["m"]][r["k"]]
        assert r["obs"] == cell["obs"] and r["obs_sd"] == cell["obs_sd"]
        assert r["n_anchors"] == cell["n_anchors"]
        assert r["P1_AB"] == cell[v] and r["P1_BA"] == cellBA[v]
        assert r["N0"] == cell["N0"] == cellBA["N0"]
        assert r["N1_AB"] == cell["N1"] and r["N1_BA"] == cellBA["N1"]
        assert r["N2_AB"] == cell["N2"] and r["P2_AB"] == cell["P2"]
        assert r["N2_BA"] == cellBA["N2"] and r["P2_BA"] == cellBA["P2"]
        assert r["obs"] == cellBA["obs"]


def test_fit_never_reads_h2_or_other_m():
    """Perturbing every D_B entry outside H1 x H1 of the fit prefix changes no prediction."""
    DA, DB = _rough(m=1024)
    ids = [str(i) for i in range(1024)]
    h1 = util.hash_half(ids, C.HALVES_SALT)
    order = np.arange(1024)
    kw = dict(m_grid=(256, 1024), m_fit=512, ks=(1, 10), variant=pr.FROZEN_P1_VARIANT, directions=("AB",))
    rows0, _ = run_pair_draw(DA, DB, order, h1, **kw)
    DB2 = DB.copy()
    keep = np.zeros(1024, bool)
    keep[:512] = h1[:512]
    out = ~(keep[:, None] & keep[None, :])
    noise = util.rng_from_key("S11test|perturb").random((1024, 1024))
    noise = 0.5 * (noise + noise.T)
    DB2[out] += noise[out]
    np.fill_diagonal(DB2, 0.0)
    rows1, _ = run_pair_draw(DA, DB2, order, h1, **kw)
    for a, b in zip(rows0, rows1):
        for key in ("P1_AB", "N1_AB", "N2_AB", "P2_AB"):
            assert a[key] == b[key], key


def test_design_draws_and_halves():
    ids = list(range(1000, 1300))
    orders = C.draw_orders(ids)
    assert len(orders) == C.N_DRAWS
    assert np.array_equal(orders[0], np.arange(300))
    for o in orders:
        assert np.array_equal(np.sort(o), np.arange(300))
    assert len({tuple(o[:50]) for o in orders}) == C.N_DRAWS
    h1 = C.h1_mask(ids)
    assert 0.4 < h1.mean() < 0.6
    assert np.array_equal(C.h1_mask(ids[:100]), h1[:100])        # per-item -> nested-consistent


def test_pair_list():
    P = C.pair_list()
    cls = pd.Series([p["cls"] for p in P]).value_counts().to_dict()
    assert cls == {"seed-seed": 36, "cross-vision": 10, "cross-modal": 15}
    prim_seed = sorted(p["pair"] for p in P if p["cls"] == "seed-seed" and p["primary"])
    assert prim_seed == [f"pythia410m_s{a}__pythia410m_s{b}" for a, b in C.PRIMARY_SEED_PAIRS]
    for p in P:
        assert p["a"] in C.MODELS and p["b"] in C.MODELS


def test_layer_axis_from_sidecars():
    """Metadata only: rel depth 0.5 is axis 1 for every S1.1 model, and the rule
    l(r) = floor(r L + 0.5) matches the sidecar mapping."""
    for name, slug in C.MODELS.items():
        meta = C.sidecar(slug)
        assert meta["kind"] == C.KIND[name]
        ax = C.layer_axis(meta)
        assert ax == 1, (name, ax)
        L = meta.get("n_blocks") or meta.get("n_layers")
        assert int(meta["rel_depths_to_residual_index"]["0.5"]) == math.floor(0.5 * L + 0.5)
        assert len(meta["layer_names"]) == 4


def test_refuses_coco_without_authorization(monkeypatch, tmp_path):
    import s11_run
    monkeypatch.setattr(s11_run, "AUTH_PATH", str(tmp_path / "LEAD_AUTHORIZATION.json"))

    def boom(*a, **k):
        raise AssertionError("a feature file was opened")
    monkeypatch.setattr(np, "load", boom)
    with pytest.raises(SystemExit) as e:
        s11_run.main(["--pair", "dinov2_b__clip_b", "--pre", "l2", "--out-root", str(tmp_path)])
    assert "REFUSED" in str(e.value)


def _fake_raw(slopes, n_anchor=4000, draws=5):
    """Synthetic raw rows with s_N = s0 * (m/1024)^slope (exact power laws)."""
    rows = []
    for pair, (cls, sl_obs, sl_pred, s0) in slopes.items():
        for d in range(draws):
            for m in C.M_GRID:
                for k in C.KS:
                    obs = s0 * (m / 1024) ** sl_obs
                    pred = s0 * (m / 1024) ** sl_pred
                    r = {"pair": pair, "cls": cls, "primary": True, "pre": "l2", "draw": d, "m": m, "k": k,
                         "is_fit_m": m == 1024, "n_anchors": n_anchor, "chance": k / (m - 1), "obs": obs,
                         "obs_sd": 0.0, "n_tieA": 0, "n_tieB": 0, "n_clean": n_anchor, "obs_clean": obs,
                         "N0": s0}
                    for dd in ("AB", "BA"):
                        r.update({f"P1_{dd}": pred, f"P1_mu_{dd}": 0.0, f"N1_{dd}": s0 * 0.5, f"N1c_{dd}": s0 * 0.5,
                                  f"N2_{dd}": s0, f"P2_{dd}": pred, f"P2_raw_{dd}": pred, f"P2_phat_{dd}": 4.0})
                    rows.append(r)
    return pd.DataFrame(rows)


def test_evaluator_known_answers():
    df = _fake_raw({"a": ("seed-seed", 0.3, 0.3, 0.5), "b": ("cross-vision", -0.2, -0.2, 0.4),
                    "c": ("cross-modal", 0.0, 0.0, 0.3)})
    c = E.cells(E.combine_directions(df, "mean", "chance"))
    S = E.slopes_table(c)
    s = S.set_index(["pair", "k"])["slope_obs"]
    assert abs(s[("a", 10)] - 0.3) < 1e-12 and abs(s[("b", 1)] + 0.2) < 1e-12
    res = E.evaluate(c, S, lambda t: t["primary"])
    assert res["c1"]["median_abs_log_ratio"] < 1e-12 and res["c1"]["pass"]
    assert res["n_steep"] == 6 and res["c3"]["frac"] == 1.0
    assert res["c2"]["msle"]["P1"] < 1e-20 and res["c2"]["pass"]
    assert E.outcome(res)["confirm"]
    # wrong sign everywhere -> (3) fails; a constant predictor equal to N0 fails (2)
    df2 = _fake_raw({"a": ("seed-seed", 0.3, -0.3, 0.5), "b": ("cross-vision", -0.2, 0.2, 0.4)})
    c2 = E.cells(E.combine_directions(df2, "mean", "chance"))
    r2 = E.evaluate(c2, E.slopes_table(c2), lambda t: t["primary"])
    assert r2["c3"]["frac"] == 0.0 and not r2["c3"]["pass"]
    assert not E.outcome(r2)["confirm"]
    # no steep unit -> only smooth class observed
    df3 = _fake_raw({"c": ("cross-modal", 0.05, 0.05, 0.3)})
    c3 = E.cells(E.combine_directions(df3, "mean", "chance"))
    r3 = E.evaluate(c3, E.slopes_table(c3), lambda t: t["primary"])
    assert r3["only_smooth"] and "only smooth class observed" in E.outcome(r3)["text"]


def test_evaluator_zero_prediction_is_infinite_log_error():
    df = _fake_raw({"a": ("seed-seed", 0.3, 0.3, 0.5)})
    for d in ("AB", "BA"):
        df[f"P1_{d}"] = 0.0
    c = E.cells(E.combine_directions(df, "mean", "chance"))
    res = E.evaluate(c, E.slopes_table(c), lambda t: t["primary"])
    assert math.isinf(res["c1"]["median_abs_log_ratio"]) and not res["c1"]["pass"]
    assert not res["c2"]["pass"]


def test_duplicate_items_zero_distances():
    """Exact duplicates (identical LM captions -> D = 0) must not break any predictor."""
    key = "S11test|dups"
    XA, XB = plants.aniso_features(800, 4, key)
    for src, dst in ((5, 400), (17, 18), (300, 301), (2, 700)):
        XA[dst] = XA[src]
        XB[dst] = XB[src]
    DA, DB = plants.feature_dists(XA), plants.feature_dists(XB)
    ids = [str(i) for i in range(800)]
    h1 = util.hash_half(ids, C.HALVES_SALT)
    rows, _ = run_pair_draw(DA, DB, np.arange(800), h1, m_grid=(256, 800), m_fit=400, ks=(1, 10, 100),
                            variant=pr.FROZEN_P1_VARIANT, directions=("AB", "BA"))
    for r in rows:
        for key in ("P1_AB", "P1_BA", "N1_AB", "N2_AB", "P2_AB", "N0", "obs"):
            assert np.isfinite(r[key]), (key, r)
    assert sum(r["n_tieA"] for r in rows) >= 0
