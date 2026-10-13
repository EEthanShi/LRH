"""V-f (role e0-indep): shared helpers, written from the text of
ANALYSIS_PLAN_E0.md (frozen) sections 2.3, 2.4, 2.5, 3.1, 5, 6, 10 (V-f) and
the S1.1 plan text it cites (O3, V2). It imports numpy, pandas and hrhv2 only.
No e0_* module, no s11copy module and no s11v11 module is imported or read.

Nothing here writes outside the directory passed by the caller, and every
writer refuses an existing target.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/hrh_v2_stage1_20261008")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hrhv2 import prep as HP  # noqa: E402  (plan V-f: hrhv2 may be imported)
from hrhv2 import util as HU  # noqa: E402

M_GRID = [256, 512, 1024, 2048, 4096, 8192, 16384]
K_GRID = [1, 10, 100]
M_FIT = 1024
N_POOL = 16384
N_DRAWS = 5
KMAX = 101  # first 101 neighbours (k = 100 plus the (k+1)-th for the boundary gap)

# ViT tag -> stored slug (plan 2.2; the five ViTs of the S1.1 VL pairs)
VITS = {
    "augreg_b": "vit_base_patch16_224.augreg_in21k",
    "clip_b": "vit_base_patch16_clip_224.laion2b",
    "dinov2_b": "vit_base_patch14_dinov2.lvd142m",
    "mae_b": "vit_base_patch16_224.mae",
    "siglip_b": "vit_base_patch16_siglip_224.webli",
}
ENCS = ["contriever", "mpnet", "qwen3emb06"]
INPUTS = ["c1", "n1", "n2", "n3", "n4", "n5"]
# arm -> text input (plan 2.5): P = C1; D1..D4 = N1..N4; D5 (= arm A) = N5; all under `clip`
ARMS = {"P": "c1", "D1": "n1", "D2": "n2", "D3": "n3", "D4": "n4", "D5": "n5"}
E0_OBS_FILE = {"P": "obs_P", "D1": "obs_D1", "D2": "obs_D2", "D3": "obs_D3",
               "D4": "obs_D4", "D5": "obs_A"}

FEAT_POOL = os.path.join(ROOT, "features", "coco_pool")
FEAT_E0 = os.path.join(ROOT, "features", "coco_e0")
EXTRACTION_JSON = os.path.join(ROOT, "results", "E0_sentence", "extraction", "EXTRACTION_E0.json")


def sha256_file(path: str, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_new_bytes(path: str, data: bytes) -> str:
    """Write a new file; refuse if it exists. Returns sha256."""
    with open(path, "xb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def write_new_json(path: str, obj) -> str:
    return write_new_bytes(path, (json.dumps(obj, indent=1, sort_keys=True, default=_jdefault) + "\n").encode())


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


# ---------------------------------------------------------------------------
# Design (plan 3.1 = S1.1 O3 + V2), rebuilt from hrhv2.util and items.parquet
# ---------------------------------------------------------------------------

def design():
    """Return image_ids (16384,), orders (5, 16384) pool indices in draw order,
    h1 (16384,) bool (True = H1)."""
    items = pd.read_parquet(os.path.join(FEAT_POOL, "items.parquet"))
    assert len(items) == N_POOL
    assert np.array_equal(items["idx"].to_numpy(), np.arange(N_POOL))
    # draw 0 = stored order: ascending sha256("hrh-v2-coco-pool-v1:" + image_id)
    ids = items["image_id"].to_numpy().astype(np.int64)
    keys = [hashlib.sha256(f"hrh-v2-coco-pool-v1:{int(i)}".encode()).hexdigest() for i in ids]
    o0 = np.array(sorted(range(N_POOL), key=lambda r: keys[r]), dtype=np.int64)
    assert np.array_equal(o0, np.arange(N_POOL)), "stored order is not the sha256 order"
    orders = [np.arange(N_POOL, dtype=np.int64)]
    idl = [int(i) for i in ids]
    for d in range(1, N_DRAWS):
        orders.append(np.asarray(HU.hash_order(idl, f"hrh-v2-s11-draw{d}"), dtype=np.int64))
    h1 = np.asarray(HU.hash_half([str(i) for i in idl], "hrhv2-halves"), dtype=bool)
    return ids, np.stack(orders), h1


def predicted_anchors(order: np.ndarray, h1: np.ndarray, m: int) -> np.ndarray:
    """S1.1 V2: at m >= 1024 the H2 items of the draw's 1024 prefix; at m < 1024
    the H2 items inside the m prefix. Returned as pool indices in draw order."""
    base = order[:min(m, M_FIT)]
    return base[~h1[base]]


# ---------------------------------------------------------------------------
# Inputs (plan 2.2, 2.3) and preprocessing `clip` (plan 2.4 = SPEC prh_clip+l2)
# ---------------------------------------------------------------------------

def _vit_paths(tag: str):
    slug = VITS[tag]
    side = json.load(open(os.path.join(FEAT_POOL, slug + ".json")))
    prim = side["files_sha256"]["primary"]
    return os.path.join(FEAT_POOL, prim["name"]), prim["sha256"], side


def _extraction_hashes():
    return json.load(open(EXTRACTION_JSON))["files_sha256"]


def caption_rank_table():
    """(16384, n_max) array of caption rows (-1 = none), column r = caption of
    rank r by ascending annotation id (own ranking from ann_id)."""
    ci = pd.read_parquet(os.path.join(FEAT_E0, "captions_index.parquet"))
    assert np.array_equal(ci["cap_row"].to_numpy(), np.arange(len(ci)))
    ci = ci.sort_values(["idx", "ann_id"], kind="mergesort")
    ci["myrank"] = ci.groupby("idx").cumcount()
    nmax = int(ci["myrank"].max()) + 1
    tab = -np.ones((N_POOL, nmax), dtype=np.int64)
    tab[ci["idx"].to_numpy(), ci["myrank"].to_numpy()] = ci["cap_row"].to_numpy()
    assert (tab[:, 0] >= 0).all()
    return tab, len(ci)


def my_caption_average(caps32: np.ndarray, tab: np.ndarray, n: int):
    """Own caption average (plan 2.3): float32 -> float64; each caption row / its
    l2 norm; arithmetic mean over the image's first n captions (ascending
    annotation id), summed in that order; / l2 norm. Zero row -> error.
    Returns (X (16384, d) float64, counts)."""
    d = caps32.shape[1]
    acc = np.zeros((N_POOL, d), dtype=np.float64)
    cnt = np.zeros(N_POOL, dtype=np.int64)
    for r in range(n):
        if r >= tab.shape[1]:
            break
        sel = np.flatnonzero(tab[:, r] >= 0)
        E = caps32[tab[sel, r]].astype(np.float64)
        nr = np.sqrt(np.sum(E * E, axis=1))
        if np.any(nr == 0):
            raise ValueError("zero caption embedding")
        acc[sel] += E / nr[:, None]
        cnt[sel] += 1
    if np.any(cnt == 0):
        raise ValueError("image without caption")
    mean = acc / cnt[:, None]
    nm = np.sqrt(np.sum(mean * mean, axis=1))
    if np.any(nm == 0):
        raise ValueError("zero averaged row")
    return mean / nm[:, None], cnt


def model_names():
    out = list(VITS)
    for e in ENCS:
        for i in INPUTS:
            out.append(f"{e}.{i}")
    return out


def load_preprocessed(name: str, verify: bool = True):
    """Preprocessed (`clip` = SPEC prh_clip+l2) float64 features of one model,
    plus a provenance dict."""
    prov = {"name": name}
    if name in VITS:
        path, want, side = _vit_paths(name)
        if verify:
            got = sha256_file(path)
            assert got == want, (path, got, want)
            prov["file_sha256"] = got
        names = side["layer_names"]
        rel = side["rel_depths_to_residual_index"]
        # stored layers = outputs of blocks layer_axis_block_indices; rel 0.5 -> residual
        # index rel["0.5"] -> stored block index rel["0.5"] - 1
        blk = side["layer_axis_block_indices"]
        li = blk.index(int(rel["0.5"]) - 1)
        prov.update(layer_index=li, layer_name=names[li])
        A = np.load(path)
        assert A.shape == (N_POOL, 4, A.shape[2]) and A.dtype == np.float32
        X = A[:, li, :]
        c = HP.prh_clip_threshold(A)  # pool-stored: all 4 stored layers, all 16384 rows
        Xp = HP.preprocess(X, "prh_clip+l2", clip_ref=A)
        prov.update(clip_threshold=c, clip_scope="pool-stored (4 layers, 16384 rows)")
        del A
    else:
        enc, inp = name.split(".")
        hashes = _extraction_hashes()
        if inp == "c1":
            path = os.path.join(FEAT_E0, f"{enc}.c1.npy")
            if verify:
                got = sha256_file(path)
                assert got == hashes[f"features/coco_e0/{enc}.c1.npy"], path
                prov["file_sha256"] = got
            X = np.load(path)
            assert X.shape[0] == N_POOL and X.dtype == np.float32
            prov["caption_counts"] = None
        else:
            n = int(inp[1])
            path = os.path.join(FEAT_E0, f"{enc}.caps.npy")
            cpath = os.path.join(FEAT_E0, "captions_index.parquet")
            if verify:
                got = sha256_file(path)
                assert got == hashes[f"features/coco_e0/{enc}.caps.npy"], path
                gotc = sha256_file(cpath)
                assert gotc == hashes["features/coco_e0/captions_index.parquet"], cpath
                prov["file_sha256"] = got
                prov["captions_index_sha256"] = gotc
            caps = np.load(path)
            tab, ncap = caption_rank_table()
            assert caps.shape[0] == ncap and caps.dtype == np.float32
            X, cnt = my_caption_average(caps, tab, n)
            prov["caption_counts"] = {int(v): int(c) for v, c in zip(*np.unique(cnt, return_counts=True))}
            del caps
        c = HP.prh_clip_threshold(X)  # one threshold per (encoder, input) on all 16384 rows
        Xp = HP.preprocess(X, "prh_clip+l2")
        prov.update(clip_threshold=c, clip_scope="16384 rows of this input")
    return Xp, prov


def overlap_counts(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise |set(a_r) & set(b_r)| for rows of unique entries."""
    c = np.concatenate([a, b], axis=1)
    c.sort(axis=1)
    return np.sum(c[:, 1:] == c[:, :-1], axis=1)
