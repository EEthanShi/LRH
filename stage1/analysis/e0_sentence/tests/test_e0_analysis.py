"""pytest for the E0 analysis code (ANALYSIS_PLAN_E0.md v1.0 §10): constants, caption average, ctr, design copy,
cache layout on synthetic data, every decision rule on planted inputs (one plant per outcome label), and the
evaluation path end to end on planted s_N worlds. Synthetic data and frozen S1.1 outputs only.

    OMP_NUM_THREADS=4 venv/bin/python -m pytest -q analysis/e0_sentence/tests
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import e0_common as E  # noqa: E402
import e0_evaluate as EV  # noqa: E402
import e0_guard as G  # noqa: E402
import e0_prep as EP  # noqa: E402
import e0_rules as R  # noqa: E402
import s11v11_obs as SO  # noqa: E402

FC = json.load(open(E.FREEZE_CONSTANTS))


# ---------------------------------------------------------------------------------------------- constants
def test_constants_equal_freeze_json_bitwise():
    for (pre, k, m), (a, b, s) in R.LINES.items():
        d = FC["calibration"][f"{pre}|k{k}|m{m}"]
        assert (a, b, s) == (d["a"], d["b"], d["resid_sd"])
    for (pre, k), comp in R.COMPETITOR.items():
        assert comp == FC["competitor_vit"][f"{pre}|k{k}"]["beta_M_v"]
    assert R.BETA_M_CONST == FC["competitor_vit"]["clip|k10"]["vl_median_slope"]
    wc = FC["within_class"]["clip|k10"]
    assert R.WITHIN_CLASS_K10["pooled"] == wc["pooled_within_class"]
    assert R.WITHIN_CLASS_K10["cross-modal"] == wc["cross-modal"]


def test_freeze_constants_file_hash():
    assert E.sha256_file(E.FREEZE_CONSTANTS) == "144d5aef8a322f687f0159efd5566e1d6e89fc39b592ebffa64e8ba565a5410d"


def test_plan_frozen_in_status():
    m = E.milestone("plan")
    assert m is not None and m["file_ok"]


# ---------------------------------------------------------------------------------------------- inputs
def _index(n_pool, counts, rng):
    rows = []
    for i in range(n_pool):
        for r in range(counts[i]):
            rows.append({"idx": i, "rank": r})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)   # unordered on purpose
    df.insert(0, "cap_row", np.arange(len(df)))
    return df


def test_caption_average_matches_definition():
    rng = np.random.default_rng(0)
    n_pool, d = 7, 5
    counts = [5, 5, 6, 7, 5, 5, 5]
    idx = _index(n_pool, counts, rng)
    caps = rng.normal(size=(len(idx), d)).astype(np.float32)
    for n in (1, 3, 5, None):
        X, cnt = E.caption_average(caps, idx, n, n_pool=n_pool)
        for i in range(n_pool):
            sel = idx[(idx["idx"] == i) & ((idx["rank"] < n) if n else True)].sort_values("rank")
            Z = caps[sel["cap_row"]].astype(np.float64)
            Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            v = Z.mean(axis=0)
            v /= np.linalg.norm(v)
            assert np.allclose(X[i], v, atol=1e-14, rtol=0)
            assert cnt[i] == (min(n, counts[i]) if n else counts[i])
        assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-14)


def test_caption_average_rejects_missing_rows():
    rng = np.random.default_rng(1)
    idx = _index(3, [2, 2, 2], rng)
    idx = idx[idx["idx"] != 1]
    with pytest.raises(ValueError):
        E.caption_average(rng.normal(size=(6, 3)).astype(np.float32), idx, None, n_pool=3)


def test_ctr_features_definition():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 8)) * 3 + 5
    Y = E.ctr_features(X)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Z = Xn - Xn.mean(axis=0)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    assert np.allclose(Y, Z, atol=1e-15)
    with pytest.raises(ValueError):
        E.ctr_features(np.vstack([X, np.zeros(8)]))


# ---------------------------------------------------------------------------------------------- design / lists
def test_design_copy_bitwise():
    import importlib.util
    spec = importlib.util.spec_from_file_location("s11v11_common_orig_t",
                                                  os.path.join(E.PROJECT, "analysis", "s11", "s11v11_common.py"))
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    for kind in ("clip", "dedup"):
        a, b = E.design(kind), orig.design(kind)
        for da, db in zip(a["draws"], b["draws"]):
            for x in ("order", "h1", "anchor_pos"):
                assert np.array_equal(da[x], db[x])
            assert da["m_grid"] == db["m_grid"]
    for f in ("s11_common.py", "s11v11_common.py", "s11v11_prep.py", "s11v11_obs.py", "s11v11_evaluate.py"):
        assert E.sha256_file(os.path.join(E.S11COPY, f)) == E.sha256_file(os.path.join(E.PROJECT, "analysis", "s11", f))


def test_lists_and_pairs():
    L = E.new_cache_list()
    assert len(L) == 51 and len(set(L)) == 51
    assert sum(1 for p, _ in L if p == "clip") == 21 and sum(1 for p, _ in L if p == "ctr") == 24
    P = E.new_pairs("P")
    assert len(P) == 15 and P[0]["b"].endswith(".c1") and all(p["cls"] == E.NEW_CLASS for p in P)
    assert [p["b"].split(".")[1] for p in E.new_pairs("D3")] == ["n3"] * 15
    assert E.cache_dir("clip", "dinov2_b").startswith(E.S11_CACHE)
    assert E.cache_dir("clip-dedup", "dinov2_b") == os.path.join(E.S11_CACHE, "dedup", "dinov2_b")
    assert E.cache_dir("ctr", "dinov2_b").startswith(E.CACHE)
    assert E.cache_dir("clip", "mpnet.c1") == os.path.join(E.CACHE, "clip", "mpnet.c1")
    assert len(E.old_pairs()) == 29


def test_cpu_budget():
    E.check_cpu_budget(2, 3)
    with pytest.raises(SystemExit):
        E.check_cpu_budget(2, 4)


# ---------------------------------------------------------------------------------------------- caches on synthetic data
def small_design(n_pool=1536, seed=3):
    rng = np.random.default_rng(seed)
    draws = []
    for d in range(2):
        o = np.arange(n_pool) if d == 0 else rng.permutation(n_pool)
        h1 = rng.random(n_pool) < 0.5
        draws.append({"draw": d, "order": o.astype(np.int64), "h1": h1, "anchor_pos": np.flatnonzero(~h1[:1024]),
                      "m_grid": (256, 512, 1024, 1536), "M": n_pool})
    return {"config": "syn", "n_pool": n_pool, "dropped_rows": np.zeros(0, np.int64), "draws": draws}


def syn_D(X):
    from hrhv2 import prep
    return prep.Geom.from_features(X, "l2", keep_X=False).D


def test_cache_builder_and_obs_on_synthetic(tmp_path):
    rng = np.random.default_rng(4)
    n = 1536
    Z = rng.normal(size=(n, 6))
    XA = Z @ rng.normal(size=(6, 20))
    XB = Z @ rng.normal(size=(6, 16)) + 0.3 * rng.normal(size=(n, 16))
    des = small_design(n)
    caches, meds = {}, {}
    for name, X in (("A", XA), ("B", XB)):
        out = str(tmp_path / name)
        EP.build_cache("l2", name, out, save_rows=True, D=syn_D(X), info={"syn": True}, design_fn=lambda k: des)
        assert os.path.exists(os.path.join(out, "DONE"))
        assert set(os.listdir(out)) >= {"meta.json", "DONE", "d0_knn.npz", "d1_knn.npz", "d0_rows.npy", "d0_fit.npy"}
        with pytest.raises(FileExistsError):
            EP.build_cache("l2", name, out, save_rows=False, D=syn_D(X), info={}, design_fn=lambda k: des)
        import e0_obs
        caches[name], meds[name], _ = e0_obs.load_cache(out, des)
    P = {"pair": "A__B", "cls": "syn", "primary": True}
    rows = SO.pair_rows("syn", P, des, caches["A"], caches["B"], meds["A"], meds["B"])
    df = pd.DataFrame(rows)
    assert set(df["m"]) == {256, 512, 1024, 1536} and set(df["k"]) == {1, 10, 100}
    # brute-force check of one cell: draw 0, m = 512, k = 10
    DA, DB = syn_D(XA), syn_D(XB)
    o = des["draws"][0]["order"][:512]
    anc = des["draws"][0]["anchor_pos"]
    anc = anc[anc < 512]
    tot = 0.0
    for i in anc:
        def nn(D):
            r = D[o[i], o].copy()
            r[i] = np.inf
            return set(np.lexsort((np.arange(512), r))[:10])
        tot += len(nn(DA) & nn(DB)) / 10
    cell = df[(df["draw"] == 0) & (df["m"] == 512) & (df["k"] == 10)]["obs"].iloc[0]
    assert abs(cell - tot / anc.size) < 1e-15
    dfx, ds, S, c = EV.arm_tables(df)
    assert len(S) == 3 and np.all(np.isfinite(S["slope_obs"]))


# ---------------------------------------------------------------------------------------------- P1 plants
def test_p1_all_labels():
    em = np.full(15, -0.4)
    assert R.p1_rule(np.full(15, 0.01), em)["label"] == "holds"
    s = np.array([0.04] * 10 + [0.08] * 5)          # median 0.04, 10/15 within 0.05
    assert R.p1_rule(s, em)["label"] == "partly holds"
    assert R.p1_rule(np.full(15, 0.07), em)["label"] == "fails"
    s = np.array([0.0] * 7 + [0.2] * 8)              # median 0.2 -> fails
    assert R.p1_rule(s, em)["label"] == "fails"
    s = np.array([0.02] * 7 + [-0.09] * 8)           # |median| 0.09 > 0.06
    assert R.p1_rule(s, em)["label"] == "fails"
    s = np.array([0.0] * 7 + [0.055] * 8)            # |median| 0.055 <= 0.06 but only 7/15 within 0.05
    assert R.p1_rule(s, em)["label"] == "fails"
    em2 = np.array([-0.05] * 8 + [-0.4] * 7)         # 7 informative < 8
    assert R.p1_rule(np.zeros(15), em2)["label"] == "uninformative"
    # flat pairs are excluded from the statistic
    em3 = np.array([-0.05] * 5 + [-0.4] * 10)
    s3 = np.array([0.5] * 5 + [0.0] * 10)
    assert R.p1_rule(s3, em3)["label"] == "holds"


# ---------------------------------------------------------------------------------------------- P2b plants
VITS = list(E.VITS)
ENCS = list(E.ENCODERS)


def plant_p2b(levels_by_enc, lam_by_enc, line=R.LINES[("clip", 10, 1024)], comp=R.COMPETITOR[("clip", 10)]):
    L, b, v, e = [], [], [], []
    for vit in VITS:
        for enc in ENCS:
            l = levels_by_enc[enc]
            bh = R.bhat_star(np.log(l), line)
            bm = comp[vit]
            L.append(l)
            b.append(bm + lam_by_enc[enc] * (bh - bm))
            v.append(vit)
            e.append(enc)
    return np.array(L), np.array(b), v, e


def test_p2b_all_labels():
    lv = {"mpnet": 0.3, "contriever": 0.15, "qwen3emb06": 0.25}
    assert R.p2b_rule(*plant_p2b(lv, {e: 1.0 for e in ENCS}))["label"] == "holds"
    r = R.p2b_rule(*plant_p2b(lv, {e: 1.0 for e in ENCS}))
    assert abs(r["lam"] - 1.0) < 1e-12
    assert R.p2b_rule(*plant_p2b(lv, {e: 0.0 for e in ENCS}))["label"] == "fails"
    assert R.p2b_rule(*plant_p2b(lv, {e: 0.5 for e in ENCS}))["label"] == "inconclusive"
    assert R.p2b_rule(*plant_p2b(lv, {e: 1.5 for e in ENCS}))["label"] == "inconclusive"       # overshoot
    assert R.p2b_rule(*plant_p2b(lv, {"mpnet": 1.0, "contriever": 0.3, "qwen3emb06": 1.0}))["label"] == "inconclusive"
    near = {e: 0.045 for e in ENCS}                                                            # predictions coincide
    assert R.p2b_rule(*plant_p2b(near, {e: 1.0 for e in ENCS}))["label"] == "uninformative"
    one = {"mpnet": 0.3, "contriever": 0.045, "qwen3emb06": 0.046}                             # 1 informative encoder
    r = R.p2b_rule(*plant_p2b(one, {e: 1.0 for e in ENCS}))
    assert r["label"] in ("inconclusive", "uninformative")
    two = {"mpnet": 0.3, "contriever": 0.045, "qwen3emb06": 0.3}
    r = R.p2b_rule(*plant_p2b(two, {e: 1.0 for e in ENCS}))
    assert r["label"] == "holds" and r["informative_encoders"] == ["mpnet", "qwen3emb06"]


def test_bhat_star_cap():
    line = R.LINES[("clip", 10, 1024)]
    assert R.bhat_star(np.log(0.9), line) == 0.0
    assert abs(R.bhat_star(np.log(0.2), line) - (line[0] + line[1] * math.log(0.2))) < 1e-15


def test_p2b_leave_one_out_and_wilcoxon():
    L, b, v, e = plant_p2b({"mpnet": 0.3, "contriever": 0.15, "qwen3emb06": 0.25}, {x: 1.0 for x in ENCS})
    loo = R.p2b_leave_one_out(L, b, v, e)
    assert all(abs(x - 1.0) < 1e-12 for x in loo["leave_encoder_out"].values())
    w = R.wilcoxon_descriptive(L, b + 0.001 * np.arange(15), v)
    assert w["vit_specific"]["p_one_sided"] < 0.01


# ---------------------------------------------------------------------------------------------- P2c plants
def plant_p2c(gamma_by_enc, dln_by_enc):
    gam, dln, enc = [], [], []
    for vit in VITS:
        for e in ENCS:
            gam.append(gamma_by_enc[e])
            dln.append(dln_by_enc[e])
            enc.append(e)
    return np.array(gam), np.array(dln), enc


def test_p2c_all_labels():
    d = {e: 0.5 for e in ENCS}
    assert R.p2c_rule(*plant_p2c({e: 0.228 for e in ENCS}, d))["label"] == "holds"
    assert R.p2c_rule(*plant_p2c({e: 0.08 for e in ENCS}, d))["label"] == "fails"
    assert R.p2c_rule(*plant_p2c({e: 0.15 for e in ENCS}, d))["label"] == "inconclusive"
    assert R.p2c_rule(*plant_p2c({e: 0.40 for e in ENCS}, d))["label"] == "inconclusive"      # overshoot
    assert R.p2c_rule(*plant_p2c({e: 0.228 for e in ENCS}, {e: 0.2 for e in ENCS}))["label"] == "uninformative"
    mixed = {"mpnet": 0.228, "contriever": 0.05, "qwen3emb06": 0.228}
    assert R.p2c_rule(*plant_p2c(mixed, d))["label"] == "inconclusive"
    r = R.p2c_rule(*plant_p2c({e: 0.228 for e in ENCS}, {"mpnet": 0.5, "contriever": 0.2, "qwen3emb06": 0.2}))
    assert r["label"] == "uninformative"
    r = R.p2c_rule(*plant_p2c({e: 0.228 for e in ENCS}, {"mpnet": 0.5, "contriever": 0.5, "qwen3emb06": 0.2}))
    assert r["label"] == "holds" and r["informative_encoders"] == ["contriever", "mpnet"]


def test_gamma_pairs():
    Ln = np.exp(np.array([[-2.0, -1.8, -1.7, -1.65, -1.6]]))
    bn = 0.3 + 0.228 * np.log(Ln)
    g, dl = R.gamma_pairs(Ln, bn)
    assert abs(g[0] - 0.228) < 1e-12 and abs(dl[0] - 0.4) < 1e-12


# ---------------------------------------------------------------------------------------------- P2 descriptive
def test_p2_permutation_deterministic_and_sane():
    rng = np.random.default_rng(5)
    Lm = rng.random((5, 6))
    Sm = Lm + 0.01 * rng.normal(size=(5, 6))
    a = R.p2_within_vit_permutation(Lm, Sm, R=2000)
    b = R.p2_within_vit_permutation(Lm, Sm, R=2000)
    assert a == b and a["rho"] > 0.95 and a["p_one_sided"] < 0.01
    t = R.p2_text_model_level(Lm, Sm)
    assert t["n_perms"] == 720 and 0 < t["p_exact_one_sided"] <= 1


def test_p2_permutation_matches_loop_definition():
    rng = np.random.default_rng(6)
    Lm = rng.random((5, 6))
    Sm = rng.random((5, 6))
    R_ = 300
    got = R.p2_within_vit_permutation(Lm, Sm, R=R_)
    g = np.random.Generator(np.random.PCG64(R.P2_SEED))
    rho = R.spearman(Lm.ravel(), Sm.ravel())
    cnt = 0
    for _ in range(R_):
        Sp = np.stack([Sm[i, g.permutation(6)] for i in range(5)])
        cnt += R.spearman(Lm.ravel(), Sp.ravel()) >= rho - 1e-12
    assert got["p_one_sided"] == (1 + cnt) / (1 + R_)


# ---------------------------------------------------------------------------------------------- evaluation end to end
def planted_obs(arm, level_fn, slope_fn, ksum=0.0):
    """obs rows: s(m, k) = L * ((k/m)/(10/1024))^{-b} * (k/10)^{ksum} for every draw (ksum != 0 breaks k/m)."""
    rows = []
    for p in E.new_pairs(arm):
        L, b = level_fn(p), slope_fn(p)
        for d in range(5):
            for m in (256, 512, 1024, 2048, 4096, 8192, 16384):
                for k in (1, 10, 100):
                    s = L * ((k / m) / (10 / 1024)) ** (-b) * (k / 10) ** ksum
                    s = max(s, 1e-4)          # pure power law (values above 1 are allowed in this plant)
                    rows.append({"config": arm, "pair": p["pair"], "cls": p["cls"], "primary": True, "draw": d, "m": m,
                                 "k": k, "is_fit_m": m == 1024, "chance": k / (m - 1), "n_anchors": 515, "obs": s,
                                 "obs_sd": 0.0, "n_allH2": 8000, "obs_allH2": s, "obs_allH2_sd": 0.0, "n_tieA": 0,
                                 "n_tieB": 0, "n_clean": 515, "obs_clean": s, "neartie_A": 0.0, "neartie_B": 0.0,
                                 "neartie_allH2_A": 0.0, "neartie_allH2_B": 0.0, "vit": p["vit"], "enc": p["enc"]})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def s11():
    return {pre: EV.load_s11(pre, FC["inputs_sha256"]) for pre in ("clip", "l2")}


LV = {"mpnet": 0.30, "contriever": 0.14, "qwen3emb06": 0.22}
NFAC = (1.0, 1.25, 1.45, 1.55, 1.65)                       # ln(L5/L1) = 0.50


def world(lam, gamma, ksum):
    line = R.LINES[("clip", 10, 1024)]
    comp = R.COMPETITOR[("clip", 10)]

    def slope(p, L):
        bh = float(R.bhat_star(math.log(L), line))
        return comp[p["vit"]] + lam * (bh - comp[p["vit"]])
    obs = {"P": planted_obs("P", lambda p: LV[p["enc"]], lambda p: slope(p, LV[p["enc"]]), ksum)}
    for n, arm in enumerate(E.P2C_ARMS):
        def lev(p, n=n):
            return 0.8 * LV[p["enc"]] * NFAC[n]

        def slp(p, n=n):
            base = slope(p, 0.8 * LV[p["enc"]])
            return base + gamma * math.log(NFAC[n])
        obs[arm] = planted_obs(arm, lev, slp, ksum)
    return obs


def test_evaluate_planted_world_holds(s11):
    V = EV.evaluate(world(1.0, 0.228, 0.0), s11, None)
    assert V["labels"] == {"P1": "holds", "P2b": "holds", "P2c": "holds"}
    prof = V["descriptive"]["P3g_shared_k_over_m"]
    assert len(prof) == 6 and all(abs(v - 1.0) < 1e-9 for v in prof.values())   # pure k/m world: ratio 1
    assert abs(V["P2b"]["lam"] - 1.0) < 1e-6
    assert abs(V["P2c"]["median_gamma"] - 0.228) < 1e-6
    assert V["P2_descriptive"]["within_vit_permutation"]["R"] == R.P2_R


def test_evaluate_planted_world_fails(s11):
    V = EV.evaluate(world(0.0, 0.0, 0.2), s11, None)
    assert V["labels"] == {"P1": "fails", "P2b": "fails", "P2c": "fails"}


def test_evaluate_planted_world_inconclusive(s11):
    V = EV.evaluate(world(0.5, 0.15, 0.04), s11, None)
    assert V["labels"] == {"P1": "partly holds", "P2b": "inconclusive", "P2c": "inconclusive"}


def test_tables_md_renders(s11):
    obs = world(1.0, 0.228, 0.0)
    V = EV.evaluate(obs, s11, None)
    T = {a: {"slopes": EV.arm_tables(o)[2], "levels": EV.levels(EV.arm_tables(o)[3]),
             "el": {"k10_100": EV.elasticities(EV.arm_tables(o)[3], (10, 100))}} for a, o in obs.items()}
    md = EV.tables_md(V, T)
    assert "| P1 | holds |" in md and "| P2b | holds |" in md and "| P2c | holds |" in md


# ---------------------------------------------------------------------------------------------- guard pieces
def test_guard_negative_scan(tmp_path):
    out = tmp_path
    os.makedirs(out / "obs")
    os.makedirs(out / "eval")
    (out / "obs" / "obs_ctrold.parquet").write_text("x")
    (out / "eval" / "ctr_calibration.json").write_text("{}")
    assert G.forbidden_files(str(out)) == []
    (out / "obs" / "obs_P.parquet").write_text("x")
    (out / "VERDICTS_E0.json").write_text("{}")
    bad = G.forbidden_files(str(out))
    assert len(bad) == 2


def test_guard_births(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("x")
    b = G.births([str(p)])
    assert str(p) in b


def test_require_milestones_synthetic(tmp_path):
    h = E.sha256_file(E.PLAN)
    st = tmp_path / "STATUS.md"
    st.write_text(f"2026-10-10T12:58:52Z [e0-freeze] PLAN FROZEN: analysis/e0_sentence/ANALYSIS_PLAN_E0.md sha256 {h} x\n")
    assert E.require_milestones("plan", path=str(st))["plan"]["file_ok"]
    with pytest.raises(SystemExit):
        E.require_milestones("plan", "code", path=str(st))
    st.write_text(f"2026-10-10T12:58:52Z [e0-freeze] PLAN FROZEN: analysis/e0_sentence/ANALYSIS_PLAN_E0.md sha256 {'0' * 64}\n")
    with pytest.raises(SystemExit):
        E.require_milestones("plan", path=str(st))
