"""Tests for the S1.1 v1.1 entry points (synthetic data and COCO metadata only;
no COCO feature array is read).

    cd analysis/s11 && ../../venv/bin/python -m pytest -q test_s11v11.py
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import s11_common as C  # noqa: E402
import s11v11_common as V  # noqa: E402
import s11v11_prep as PR  # noqa: E402
import s11v11_units as UN  # noqa: E402
import s11v11_obs as OB  # noqa: E402
from hrhv2 import plants, predict as p10, predict_v11 as v11, structural as st, util  # noqa: E402


# ---------------------------------------------------------------------------
# guard
# ---------------------------------------------------------------------------

def _no_feature_reads(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a feature file was opened")
    monkeypatch.setattr(np, "load", boom)


def test_guard_refuses_without_authorization(monkeypatch, tmp_path):
    monkeypatch.setattr(V, "AUTH_PATH", str(tmp_path / "LEAD_AUTHORIZATION.json"))
    _no_feature_reads(monkeypatch)
    with pytest.raises(SystemExit) as e:
        V.check_authorization("clip")
    assert "REFUSED" in str(e.value)


def test_guard_refuses_wrong_hash_or_clip_mode(monkeypatch, tmp_path):
    if not os.path.exists(V.AUTH_PATH):
        pytest.skip("no authorization file")
    auth = json.load(open(V.AUTH_PATH))
    _no_feature_reads(monkeypatch)
    for key, bad in (("plan_sha256", "0" * 64), ("predict_v11_py_sha256", "1" * 64), ("clip_mode", "pool-layer"),
                     ("D4", "a-frozen"), ("lead_decisions_v1_1_sha256", "2" * 64)):
        a = dict(auth)
        a[key] = bad
        p = tmp_path / f"auth_{key}.json"
        p.write_text(json.dumps(a))
        monkeypatch.setattr(V, "AUTH_PATH", str(p))
        with pytest.raises(SystemExit) as e:
            V.check_authorization("clip")
        assert "REFUSED" in str(e.value), key


def test_guard_accepts_current_authorization():
    if not os.path.exists(V.AUTH_PATH):
        pytest.skip("no authorization file")
    a = V.check_authorization("clip")
    assert a["D4"] == V.D4_V11 and a["clip_mode"] == "pool-stored"
    for c in V.CONFIGS:
        V.check_authorization(c)


def test_v10_runner_refuses_under_v11_authorization(monkeypatch, tmp_path):
    """The v1.0 runner must not run under the v1.1 authorization (plan hash differs)."""
    if not os.path.exists(V.AUTH_PATH):
        pytest.skip("no authorization file")
    import s11_run
    _no_feature_reads(monkeypatch)
    with pytest.raises(SystemExit) as e:
        s11_run.main(["--pair", "dinov2_b__clip_b", "--pre", "l2", "--out-root", str(tmp_path)])
    assert "REFUSED" in str(e.value)


# ---------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------

def test_surplus_duplicates_keeps_lowest_position():
    texts = ["a", "b", "a", "c", "b", "a", "d"]
    assert V.surplus_duplicates(texts).tolist() == [2, 4, 5]


def test_design_from_item_table():
    ids, texts = C.coco_item_ids()
    full = V.design("clip", ids, texts)
    dd = V.design("dedup", ids, texts)
    assert dd["dropped_rows"].size == 54
    assert len({t for t in np.asarray(texts, dtype=object)[np.setdiff1d(np.arange(len(ids)), dd["dropped_rows"])]}) \
        == len(ids) - 54
    for a, b in zip(full["draws"], dd["draws"]):
        assert a["M"] == 16384 and b["M"] == 16330
        assert a["m_grid"] == V.M_GRID and b["m_grid"][-1] == 16330 and b["m_grid"][:-1] == V.M_GRID[:-1]
        keep = ~np.isin(a["order"], dd["dropped_rows"])
        assert np.array_equal(a["order"][keep], b["order"])        # same draw order, surplus removed
        for d_ in (a, b):
            assert np.all(d_["anchor_pos"] < V.M_FIT) and not d_["h1"][d_["anchor_pos"]].any()
            assert d_["anchor_pos"].size == int((~d_["h1"][:V.M_FIT]).sum())
    assert full["draws"][0]["anchor_pos"].size == 515            # plan V2 (draw 0)


def test_pair_sets():
    assert len(V.pairs_for("clip")) == 61 and len(V.pairs_for("l2")) == 61
    assert len(V.pairs_for("dedup")) == 29 and len(V.pairs_for("cliplayer")) == 29
    assert len(V.pairs_for("clip", "secondary")) == 32
    assert len(V.models_for("clip")) == 16 and len(V.models_for("dedup")) == 15
    assert "pythia410m_s9" not in V.models_for("dedup")


# ---------------------------------------------------------------------------
# kNN tables, cached unit inputs, P2 proxy
# ---------------------------------------------------------------------------

def _synthetic_pool(M=2048, p=3, key="s11v11test"):
    XA, XB = plants.conf_features(M, p, key)
    for src, dst in ((30, M // 3), (31, M // 2), (5, M - 7), (6, M - 9)):   # exact duplicates (ties)
        XA[dst] = XA[src]
    for src, dst in ((2, M - 3), (31, M // 2), (40, M // 5)):
        XB[dst] = XB[src]
    return plants.feature_dists(XA), plants.feature_dists(XB)


def test_knn_prefix_rows_matches_knn_table():
    DA, _ = _synthetic_pool(M=900)
    DA = np.round(DA * 4) / 4                                     # many exact ties
    np.fill_diagonal(DA, 0.0)
    ids = [f"t{i}" for i in range(900)]
    o = np.asarray(util.hash_order(ids, "s11v11test|o"), dtype=np.int64)
    for m in (256, 600, 900):
        Dm = DA[np.ix_(o[:m], o[:m])]
        nT, vT = st.knn_table(Dm, 101)
        pos = np.array([0, 3, 17, m - 1, m // 2])
        nR, vR = PR.knn_prefix_rows(DA, o, pos, m, 101, chunk=2)
        assert np.array_equal(nR, nT[pos]) and np.array_equal(vR, vT[pos])


def test_median_offdiag_items():
    DA, _ = _synthetic_pool(M=300)
    items = np.array([5, 1, 7, 100, 299, 42, 43])
    sub = DA[np.ix_(items, items)]
    off = sub[~np.eye(items.size, dtype=bool)]
    assert PR.median_offdiag_items(DA, items) == float(np.median(off))
    off_all = DA[~np.eye(300, dtype=bool)]
    assert PR.median_offdiag_items(DA, np.arange(300)) == float(np.median(off_all))


def _fake_design(M, m_grid, salt):
    ids = [f"syn{i}" for i in range(M)]
    o = np.asarray(util.hash_order(ids, salt), dtype=np.int64)
    h1 = util.hash_half(ids, V.C.HALVES_SALT)[o]
    anc = np.flatnonzero(~h1[:V.M_FIT])
    return {"draw": 0, "order": o, "h1": h1, "anchor_pos": anc, "m_grid": m_grid, "M": M}


@pytest.mark.parametrize("kind", ["nulls", "iii"])
def test_cached_unit_equals_direct_run_unit(kind):
    """Phase-A arrays + run_one == run_unit on the permuted full matrices (run_ssel.py
    convention), both directions; P2 through RowsAsSquare == frozen predict_p2 on
    the full prefix matrix."""
    M = 2048
    DA, DB = _synthetic_pool(M=M)
    dd = _fake_design(M, (256, 512, 1024, 2048), "s11v11test|draw")
    o, anc = dd["order"], dd["anchor_pos"]
    rowsA, fitA = PR.unit_arrays(DA, o, anc, V.M_FIT)
    rowsB, fitB = PR.unit_arrays(DB, o, anc, V.M_FIT)
    DAo, DBo = DA[np.ix_(o, o)], DB[np.ix_(o, o)]
    assert np.array_equal(rowsA, DAo[anc]) and np.array_equal(fitB, DBo[:V.M_FIT, :V.M_FIT])
    variants = () if kind == "nulls" else (V.UNVALIDATED,)
    for direction, (X, Y) in (("AB", (DAo, DBo)), ("BA", (DBo, DAo))):
        res, _ = UN.run_one("test", kind, None, dd, direction, rowsA, fitA, rowsB, fitB)
        ref = v11.run_unit(X[anc], Y[anc], anc, X[:V.M_FIT, :V.M_FIT], Y[:V.M_FIT, :V.M_FIT], dd["h1"][:V.M_FIT],
                           m_grid=dd["m_grid"], m_f=V.M_FIT, ks=V.KS, variants=variants, modes=v11.MODES)
        assert len(res["cells"]) == len(ref["cells"])
        for a, b in zip(res["cells"], ref["cells"]):
            for key, val in b.items():
                if key == "seconds":
                    continue
                assert (a[key] == val) or (isinstance(val, float) and np.isnan(val) and np.isnan(a[key])), key
        if True:                                    # P2/N2 are added in both kinds
            disp = p10.fit_iso_dispersion(X[:V.M_FIT, :V.M_FIT], Y[:V.M_FIT, :V.M_FIT], dd["h1"][:V.M_FIT])
            rhos = p10.fit_n2(X[:V.M_FIT, :V.M_FIT], Y[:V.M_FIT, :V.M_FIT], dd["h1"][:V.M_FIT])
            for cell in res["cells"]:
                m, k = cell["m"], cell["k"]
                pos = anc[anc < m]
                p2 = p10.predict_p2(X[:m, :m], pos, [k], disp)[k]
                assert cell["P2"] == p2["s_pred"] and cell["P2_phat"] == p2["p_hat"]
                assert cell["N2"] == p10.predict_n2(rhos, m, [k])[k]["s_pred"]


def test_obs_rows_match_run_unit_obs():
    """Observed s_N from the phase-A kNN tables == run_unit's own obs (bitwise),
    obs_allH2 == obs for m < m_f, and tie counts are consistent."""
    M = 2048
    DA, DB = _synthetic_pool(M=M)
    dd = _fake_design(M, (256, 512, 1024, 2048), "s11v11test|draw2")
    o = dd["order"]
    cache = {}
    for name, D in (("A", DA), ("B", DB)):
        tabs = {}
        for m in dd["m_grid"]:
            h2pos = np.flatnonzero(~dd["h1"][:m])
            idx, vals = PR.knn_prefix_rows(D, o, h2pos, m, V.KMAX_TAB)
            tabs[f"pos_m{m}"] = h2pos.astype(np.int32)
            tabs[f"idx_m{m}"] = idx.astype(np.int32)
            tabs[f"vals_m{m}"] = vals[:, V.VAL_COLS]
        cache[name] = {0: tabs}
    med = {0: {m: 1.0 for m in dd["m_grid"]}}
    P = {"pair": "synA__synB", "cls": "cross-vision", "primary": True}
    rows = OB.pair_rows("test", P, {"draws": [dd]}, cache["A"], cache["B"], med, med)
    rowsA, fitA = PR.unit_arrays(DA, o, dd["anchor_pos"], V.M_FIT)
    rowsB, fitB = PR.unit_arrays(DB, o, dd["anchor_pos"], V.M_FIT)
    res, _ = UN.run_one("test", "nulls", None, dd, "AB", rowsA, fitA, rowsB, fitB)
    ref = {(c["m"], c["k"]): c for c in res["cells"]}
    for r in rows:
        c = ref[(r["m"], r["k"])]
        assert r["obs"] == c["obs"] and r["n_anchors"] == c["n_anchors"]
        if r["m"] < V.M_FIT:
            assert r["obs_allH2"] == r["obs"]
        if r["m"] == V.M_FIT:
            assert c["N0"] == r["obs"]
    assert any(r["n_tieA"] > 0 or r["n_tieB"] > 0 for r in rows)        # the planted duplicates tie
