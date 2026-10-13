"""Integration test of the E0 analysis path at full pool size on SYNTHETIC text features (random Gaussian vectors
standing in for the three encoders; no sentence encoder exists or is read). Uses the real design, the real S1.1
ViT caches (read only) and temporary directories for every output. About 3-4 CPU minutes at <= 6 threads.

    OMP_NUM_THREADS=6 venv/bin/python -m pytest -q -p no:cacheprovider analysis/e0_sentence/tests/test_e0_integration.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import e0_calib as CAL  # noqa: E402
import e0_common as E  # noqa: E402
import e0_diag as DG  # noqa: E402
import e0_evaluate as EV  # noqa: E402
import e0_guard as G  # noqa: E402
import e0_obs as EO  # noqa: E402
import e0_prep as EP  # noqa: E402
import e0_rules as R  # noqa: E402

N = 16384


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    root = tmp_path_factory.mktemp("e0syn")
    feat = root / "features"
    cache = root / "cache"
    os.makedirs(feat)
    rng = np.random.default_rng(20261010)
    rows = []
    for i in range(N):
        for r in range(5 + (i % 7 == 0)):
            rows.append({"idx": i, "image_id": 100000 + i, "ann_id": 10 * i + r, "rank": r, "text": f"t{i}_{r}"})
    idx = pd.DataFrame(rows)
    idx.insert(0, "cap_row", np.arange(len(idx)))
    idx.to_parquet(feat / "captions_index.parquet", index=False)
    Z = rng.normal(size=(N, 12))
    for j, enc in enumerate(E.ENCODERS):
        d = 32 + 8 * j
        W = rng.normal(size=(12, d))
        c1 = (Z @ W + (0.5 + j) * rng.normal(size=(N, d))).astype(np.float32)
        caps = (Z[idx["idx"].to_numpy()] @ W + (1.0 + j) * rng.normal(size=(len(idx), d))).astype(np.float32)
        if enc != "contriever":
            c1 /= np.linalg.norm(c1, axis=1, keepdims=True)
            caps /= np.linalg.norm(caps, axis=1, keepdims=True)
        np.save(feat / f"{enc}.c1.npy", c1)
        np.save(feat / f"{enc}.caps.npy", caps)
    for pre, inp in (("clip", "c1"), ("l2", "c1"), ("clip", "n5"), ("ctr", "c1")):
        for enc in E.ENCODERS:
            if inp == "n5" and enc != "mpnet":
                continue
            if pre == "ctr" and enc != "mpnet":
                continue
            m = f"{enc}.{inp}"
            EP.build_cache(pre, m, E.cache_dir(pre, m, str(cache)), save_rows=(pre == "clip" and inp == "c1"),
                           feat_dir=str(feat))
    return {"feat": str(feat), "cache": str(cache), "index": idx}


def test_text_matrix_full_size(sandbox):
    X, info = E.text_matrix("mpnet.n5", sandbox["feat"])
    assert X.shape[0] == N and np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-12)
    assert set(info["captions_per_row"]) == {5}
    Xa, infoa = E.text_matrix("mpnet.all", sandbox["feat"])
    assert set(infoa["captions_per_row"]) == {5, 6}
    with pytest.raises(RuntimeError):
        E.text_matrix("mpnet.c1", sandbox["feat"], expected_sha={})


def test_cache_meta_and_layout(sandbox):
    d = E.cache_dir("clip", "mpnet.c1", sandbox["cache"])
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["info"]["clip_threshold"] > 0 and meta["design"]["kind"] == "clip"
    assert set(meta["draws"]) == {"0", "1", "2", "3", "4"}
    k = np.load(os.path.join(d, "d0_knn.npz"))
    assert k["idx_m16384"].shape[1] == 101 and k["vals_m16384"].shape[1] == len(E.V.VAL_COLS) == 6
    assert os.path.exists(os.path.join(d, "d0_rows.npy"))
    assert not os.path.exists(os.path.join(E.cache_dir("l2", "mpnet.c1", sandbox["cache"]), "d0_rows.npy"))


def test_obs_arm_P_and_evaluate(sandbox):
    df, files = EO.obs_rows("P", cache_root=sandbox["cache"])
    assert df["pair"].nunique() == 15 and len(df) == 15 * 5 * 7 * 3
    assert set(df["vit"]) == set(E.VITS) and set(df["enc"]) == set(E.ENCODERS)
    assert any("S1_1/v11/cache/clip/dinov2_b" in f for f in files)
    s11 = {pre: EV.load_s11(pre, json.load(open(E.FREEZE_CONSTANTS))["inputs_sha256"]) for pre in ("clip", "l2")}
    V = EV.evaluate({"P": df}, s11, None)
    assert V["labels"]["P1"] in R.LABELS["P1"] and V["labels"]["P2b"] in R.LABELS["P2b"]
    assert "P2c" not in V["labels"]
    assert len(V["P1"]["per_pair"]) == 15


def test_diag_on_sandbox(sandbox):
    res = DG.run(feat_dir=sandbox["feat"], cache_root=sandbox["cache"])
    assert set(res["clip_vs_l2"]) == set(E.ENCODERS) | set(E.OLD_TEXT)
    assert all(0 <= v <= 1 for v in res["clip_vs_l2"].values())
    assert 0 <= res["contriever_dot_vs_cosine"] <= 1


def test_calibrate_reproduces_frozen_clip_lines():
    """e0_calib on S1.1's clip obs (old pairs) must give the frozen clip lines and competitors exactly."""
    obs = pd.read_parquet(os.path.join(E.S11_OUT, "obs", "obs_clip.parquet"))
    obs = obs[obs["primary"]].reset_index(drop=True)
    out, tabs = CAL.calibrate(obs)
    for k in (1, 10, 100):
        for m in (1024, 2048):
            a, b, s = R.LINES[("clip", k, m)]
            got = out["lines"][f"k{k}|m{m}"]
            assert abs(got["a"] - a) < 1e-13 and abs(got["b"] - b) < 1e-13 and abs(got["resid_sd"] - s) < 1e-13
        for v, x in R.COMPETITOR[("clip", k)].items():
            assert abs(out["competitor"][f"k{k}"][v] - x) < 1e-13
    assert out["P1_old_pairs_ctr_descriptive"] == {"cross-modal": "holds", "cross-vision": "holds",
                                                   "seed-seed": "uninformative"}


def test_guard_check_before_code_freeze_is_violation():
    res = G.check()
    assert res["result"] == "VIOLATION"
