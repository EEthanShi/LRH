"""X3b independent check (role x3b-indep): shared definitions.

Written without reading analysis/e0_sentence/x3b/x3b_scores.py or x3b_report.py
(nor e0_*.py, s11copy/, s11*.py, x3*.py). Definitions come from
ANALYSIS_PLAN_E0.md (2.3, 2.4, 3.1, 6), ANALYSIS_PLAN_S11.md O2/O3,
ANALYSIS_PLAN_S11_v1.1.md V2/V5/V6, SPEC.md section 0 and X3B_READINGS.md.
Imports only numpy, pandas, hrhv2.prep and hrhv2.util.
EXPLORATORY.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/hrh_v2_stage1_20261008")
POOL = os.path.join(ROOT, "features/coco_pool")
E0F = os.path.join(ROOT, "features/coco_e0")

M_GRID = (256, 512, 1024, 2048, 4096, 8192, 16384)
KS = (1, 10, 100)
N_DRAWS = 5
M_FIT = 1024
NCOL = 101                       # neighbours kept per anchor (k <= 100, plus k+1)
VAL_COLS = (0, 1, 9, 10, 99, 100)  # sorted-D columns kept (k-1 and k for each k)

VITS = ("dinov2_b", "clip_b", "mae_b", "augreg_b", "siglip_b")
LMS = ("bloomz_1b7", "openllama_3b", "pythia410m_s1")
ENCS = ("mpnet", "contriever", "qwen3emb06")
SLUG = {
    "dinov2_b": "vit_base_patch14_dinov2.lvd142m",
    "clip_b": "vit_base_patch16_clip_224.laion2b",
    "mae_b": "vit_base_patch16_224.mae",
    "augreg_b": "vit_base_patch16_224.augreg_in21k",
    "siglip_b": "vit_base_patch16_siglip_224.webli",
    "bloomz_1b7": "bigscience__bloomz-1b7",
    "openllama_3b": "openlm-research__open_llama_3b",
    "pythia410m_s1": "EleutherAI__pythia-410m-seed1",
}
REL_AXIS = {"0.25": 0, "0.5": 1, "0.75": 2, "1.0": 3}
REL_TAG = {"0.25": "r025", "0.5": "r05", "0.75": "r075", "1.0": "r10"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Design (S1.1 O3 / V2), rebuilt with hrhv2.util
# ---------------------------------------------------------------------------

def build_design():
    from hrhv2.util import hash_half, hash_order
    items = pd.read_parquet(os.path.join(POOL, "items.parquet"))
    ids = items["image_id"].to_numpy()
    assert len(ids) == 16384
    orders = np.empty((N_DRAWS, len(ids)), dtype=np.int64)
    orders[0] = np.arange(len(ids))
    for d in range(1, N_DRAWS):
        orders[d] = hash_order([int(i) for i in ids], f"hrh-v2-s11-draw{d}")
    h1 = hash_half([str(int(i)) for i in ids], "hrhv2-halves")
    return orders, h1, ids


# ---------------------------------------------------------------------------
# Features -> D
# ---------------------------------------------------------------------------

def _check_sidecar(name: str):
    sc = json.load(open(os.path.join(POOL, SLUG[name] + ".json")))
    want = sc["files_sha256"]["primary"]["sha256"]
    path = os.path.join(POOL, SLUG[name] + ".npy")
    got = sha256_file(path)
    if got != want:
        raise RuntimeError(f"sha256 mismatch {path}: {got} != {want}")
    # layer axis sanity: rel -> residual index list must be increasing, 4 depths
    rel_map = sc["rel_depths_to_residual_index"]
    assert list(rel_map.keys()) == ["0.25", "0.5", "0.75", "1.0"], rel_map
    return path, got, sc


def my_l2(X: np.ndarray) -> np.ndarray:
    """Own row normalisation (float64), zero row is an error."""
    X = np.array(X, dtype=np.float64, copy=True)
    n = np.sqrt((X * X).sum(axis=1))
    if np.any(n == 0):
        raise ValueError("zero row")
    return X / n[:, None]


def my_ctr(X: np.ndarray) -> np.ndarray:
    """ANALYSIS_PLAN_E0 2.4 ctr: l2 rows -> subtract float64 column mean of the
    16,384 pool rows -> l2 rows."""
    Y = my_l2(X)
    mu = Y.sum(axis=0) / Y.shape[0]
    Y = Y - mu[None, :]
    return my_l2(Y)


def my_n5(enc: str):
    """ANALYSIS_PLAN_E0 2.3: first five official captions by ascending
    annotation id; each card embedding (float32 -> float64) divided by its l2
    norm; arithmetic mean in ascending annotation-id order; divided by its l2
    norm."""
    sc = json.load(open(os.path.join(E0F, enc + ".json")))
    cpath = os.path.join(E0F, enc + ".caps.npy")
    ipath = os.path.join(E0F, "captions_index.parquet")
    got_c = sha256_file(cpath)
    got_i = sha256_file(ipath)
    if got_c != sc["files_sha256"][enc + ".caps.npy"]:
        raise RuntimeError("caps sha256 mismatch " + enc)
    if got_i != sc["captions_index_sha256"]:
        raise RuntimeError("captions_index sha256 mismatch")
    caps = np.load(cpath)
    ci = pd.read_parquet(ipath)
    items = pd.read_parquet(os.path.join(POOL, "items.parquet"))
    n_pool = len(items)
    # own ordering by (pool idx, annotation id); check image ids
    ci = ci.sort_values(["idx", "ann_id"], kind="stable").reset_index(drop=True)
    assert (ci.groupby("idx").image_id.nunique() == 1).all()
    img_of_idx = ci.groupby("idx").image_id.first()
    assert np.array_equal(img_of_idx.index.to_numpy(), np.arange(n_pool))
    assert np.array_equal(img_of_idx.to_numpy(), items["image_id"].to_numpy())
    ci["myrank"] = ci.groupby("idx").cumcount()
    rank_agrees = bool((ci["myrank"].to_numpy() == ci["rank"].to_numpy()).all())
    cnt = ci.groupby("idx").size().to_numpy()
    sel = ci[ci.myrank < 5]
    n_sel = sel.groupby("idx").size().to_numpy()
    X = np.zeros((n_pool, caps.shape[1]), dtype=np.float64)
    for r in range(5):
        rows = sel[sel.myrank == r]
        v = caps[rows.cap_row.to_numpy()].astype(np.float64)
        v = v / np.sqrt((v * v).sum(axis=1))[:, None]
        X[rows.idx.to_numpy()] += v
    X = X / n_sel[:, None]
    X = X / np.sqrt((X * X).sum(axis=1))[:, None]
    info = {"caps_sha256": got_c, "captions_index_sha256": got_i,
            "rank_column_agrees_with_own_ann_id_rank": rank_agrees,
            "captions_per_image_min": int(cnt.min()), "captions_per_image_max": int(cnt.max()),
            "n_selected_hist": {str(int(a)): int(b) for a, b in zip(*np.unique(n_sel, return_counts=True))}}
    return X, info


def model_D(spec: dict):
    """spec: {pre: clip|l2|ctr, name, rel (str) or input 'n5'} -> (D, info)."""
    from hrhv2.prep import Geom, prh_clip_threshold
    pre, name = spec["pre"], spec["name"]
    info = {"pre": pre, "name": name}
    if name in ENCS:
        assert pre == "ctr" and spec.get("rel") == "n5"
        X, inf2 = my_n5(name)
        info.update(inf2)
        Xp = my_ctr(X)
        G = Geom.from_features(Xp, "none", keep_X=False)
    else:
        path, sh, sc = _check_sidecar(name)
        info["npy_sha256"] = sh
        A = np.load(path, mmap_mode="r")
        ax = REL_AXIS[spec["rel"]]
        info["axis"] = ax
        info["layer_name"] = sc["layer_names"][ax]
        Xl = np.asarray(A[:, ax, :])
        if pre == "clip":
            c = prh_clip_threshold(np.asarray(A))      # pool-stored: all 4 stored depths, all 16,384 rows
            info["clip_threshold"] = c
            Xc = np.clip(Xl.astype(np.float64), -c, c)
            # SPEC prh_clip+l2 = clamp then l2; pass the clamped array with pre="l2"
            # and also check against hrhv2.prep's own path on the same threshold
            G = Geom.from_features(Xc, "l2", keep_X=False)
        elif pre == "l2":
            G = Geom.from_features(Xl, "l2", keep_X=False)
        elif pre == "ctr":
            Xp = my_ctr(Xl)
            G = Geom.from_features(Xp, "none", keep_X=False)
        else:
            raise ValueError(pre)
    D = G.D
    G.K = None
    del G
    return D, info


def model_key(spec: dict) -> str:
    tag = "n5" if spec.get("rel") == "n5" else REL_TAG[spec["rel"]]
    return f"{spec['pre']}/{spec['name']}.{tag}"


def all_model_specs():
    specs = []
    for pre in ("clip", "l2", "ctr"):
        for v in VITS:
            for rel in ("0.5", "1.0"):
                specs.append({"pre": pre, "name": v, "rel": rel})
    for t in LMS:
        for rel in ("0.25", "0.5", "0.75", "1.0"):
            specs.append({"pre": "clip", "name": t, "rel": rel})
        specs.append({"pre": "l2", "name": t, "rel": "0.5"})
        specs.append({"pre": "ctr", "name": t, "rel": "0.5"})
    for e in ENCS:
        specs.append({"pre": "ctr", "name": e, "rel": "n5"})
    return specs


# ---------------------------------------------------------------------------
# Pairs (X3B_READINGS.md section 2)
# ---------------------------------------------------------------------------

def all_pairs():
    """List of dicts: set, pair (run-side naming), pre, vit, rel_v, text, rel_t,
    key_a, key_b. Order: ViT outer, text inner (as in X3B_READINGS)."""
    out = []

    def add(st, pre, v, rv, t, rt):
        tv = REL_TAG[rv]
        tt = "n5" if rt == "n5" else REL_TAG[rt]
        out.append({"set": st, "pair": f"{v}.{tv}__{t}.{tt}", "pre": pre, "vit": v,
                    "rel_v": rv, "text": t, "rel_t": rt,
                    "key_a": f"{pre}/{v}.{tv}", "key_b": f"{pre}/{t}.{tt}"})

    for v in VITS:
        for t in LMS:
            add("A1", "clip", v, "1.0", t, "0.5")
            add("A1ref", "clip", v, "0.5", t, "0.5")
            add("A3", "l2", v, "1.0", t, "0.5")
            add("A3ref", "l2", v, "0.5", t, "0.5")
            add("A4", "ctr", v, "1.0", t, "0.5")
            add("A4ref", "ctr", v, "0.5", t, "0.5")
            for rt, st in (("0.25", "A2_r025"), ("0.75", "A2_r075"), ("1.0", "A2_r10")):
                add(st, "clip", v, "1.0", t, rt)
                add(st, "clip", v, "0.5", t, rt)
        for e in ENCS:
            add("B", "ctr", v, "1.0", e, "n5")
            add("Bref", "ctr", v, "0.5", e, "n5")
    return out
