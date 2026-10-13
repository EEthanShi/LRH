"""S1.1 (PREREG_stage1.md v1.0, section S1.1): shared design constants and data access.

Role s11. Everything here is a fixed, data-independent design choice. The
operationalizations of open PREREG wording are listed in
ANALYSIS_PLAN_S11.md (DRAFT until the lead signs it off; see LEAD_AUTHORIZATION
in s11_run.py).

Nothing in this module reads a feature array at import time.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys

import numpy as np

PROJECT = os.path.expanduser("~/hrh_v2_stage1_20261008")
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from hrhv2 import util  # noqa: E402
from hrhv2 import prep  # noqa: E402

ANALYSIS_DIR = os.path.join(PROJECT, "analysis", "s11")
RESULTS_DIR = os.path.join(PROJECT, "results", "S1_1")
COCO_DIR = os.path.join(PROJECT, "features", "coco_pool")

# ---------------------------------------------------------------------------
# PREREG S1.1 design
# ---------------------------------------------------------------------------
M_GRID = (256, 512, 1024, 2048, 4096, 8192, 16384)
M_FIT = 1024
M_EVAL = tuple(m for m in M_GRID if m != M_FIT)   # "predict ... at all other m"
KS = (1, 10, 100)
N_DRAWS = 5
REL_DEPTH = "0.5"                                   # fixed relative depth (PREREG)
HALVES_SALT = "hrhv2-halves"                        # hrhv2.util.hash_half default (also used by G0)
POOL_SALT = "hrh-v2-coco-pool-v1"                   # extraction order = draw 0
DRAW_SALTS = (None,) + tuple(f"hrh-v2-s11-draw{d}" for d in range(1, N_DRAWS))
PRES = ("prh_clip+l2", "l2")                        # primary, secondary
CLIP_MODES = ("pool-layer", "pool-stored")          # single threshold fixed on the 16384 pool

VISION = {
    "dinov2_b": "vit_base_patch14_dinov2.lvd142m",
    "clip_b": "vit_base_patch16_clip_224.laion2b",
    "mae_b": "vit_base_patch16_224.mae",
    "augreg_b": "vit_base_patch16_224.augreg_in21k",
    "siglip_b": "vit_base_patch16_siglip_224.webli",
}
PYTHIA = {f"pythia410m_s{s}": f"EleutherAI__pythia-410m-seed{s}" for s in range(1, 10)}
OTHER_LM = {
    "bloomz_1b7": "bigscience__bloomz-1b7",
    "openllama_3b": "openlm-research__open_llama_3b",
}
MODELS = {**VISION, **PYTHIA, **OTHER_LM}
KIND = {**{k: "vit" for k in VISION}, **{k: "lm" for k in PYTHIA}, **{k: "lm" for k in OTHER_LM}}

CLASSES = ("seed-seed", "cross-vision", "cross-modal")
PRIMARY_SEED_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8))


def pair_list():
    """All 61 pairs: 36 seed-seed (4 primary), 10 cross-vision, 15 cross-modal.

    Direction convention for the asymmetric predictors: 'A' is the first model
    listed (lower seed; earlier ViT in VISION order; the ViT for cross-modal).
    Both directions are always computed (ANALYSIS_PLAN_S11.md, O6)."""
    out = []
    for s, t in itertools.combinations(range(1, 10), 2):
        out.append({"pair": f"pythia410m_s{s}__pythia410m_s{t}", "a": f"pythia410m_s{s}",
                    "b": f"pythia410m_s{t}", "cls": "seed-seed",
                    "primary": (s, t) in PRIMARY_SEED_PAIRS})
    for a, b in itertools.combinations(list(VISION), 2):
        out.append({"pair": f"{a}__{b}", "a": a, "b": b, "cls": "cross-vision", "primary": True})
    for a in VISION:
        for b in ("bloomz_1b7", "openllama_3b", "pythia410m_s1"):
            out.append({"pair": f"{a}__{b}", "a": a, "b": b, "cls": "cross-modal", "primary": True})
    assert len(out) == 61 and sum(p["primary"] for p in out) == 29
    return out


PAIRS = {p["pair"]: p for p in pair_list()}


# ---------------------------------------------------------------------------
# Item design (ids only; no features)
# ---------------------------------------------------------------------------

def draw_orders(item_ids):
    """Five nested orders of the pool rows (each a permutation of 0..n-1).

    draw 0 = the stored row order (ascending sha256('hrh-v2-coco-pool-v1:' + id),
    i.e. the extraction's own nested prefixes idx < m); draws 1..4 =
    hrhv2.util.hash_order(ids, 'hrh-v2-s11-draw<d>'), i.e. ascending
    sha256('hrh-v2-s11-draw<d>|' + id). All draws permute WITHIN the extracted
    pool (extract/PROTOCOL.md section 2), so at m = 16384 every draw has the
    same items."""
    n = len(item_ids)
    out = []
    for salt in DRAW_SALTS:
        if salt is None:
            out.append(np.arange(n, dtype=np.int64))
        else:
            out.append(np.asarray(util.hash_order([str(i) for i in item_ids], salt), dtype=np.int64))
    return out


def h1_mask(item_ids):
    """H1 membership per item (True = H1), hrhv2.util.hash_half with salt
    'hrhv2-halves' on str(item id). Per-item, hence identical in every draw and
    every nested prefix."""
    return util.hash_half([str(i) for i in item_ids], HALVES_SALT)


def coco_item_ids():
    """image_id of every pool row, in stored row order (metadata only)."""
    import pandas as pd
    t = pd.read_parquet(os.path.join(COCO_DIR, "items.parquet"), columns=["idx", "image_id", "text"])
    t = t.sort_values("idx")
    assert np.array_equal(t["idx"].to_numpy(), np.arange(len(t)))
    return t["image_id"].astype(np.int64).to_numpy(), t["text"].tolist()


# ---------------------------------------------------------------------------
# Layer selection (documented rule) and features -> D
# ---------------------------------------------------------------------------

def sidecar(slug, ds_dir=COCO_DIR):
    with open(os.path.join(ds_dir, slug + ".json")) as f:
        return json.load(f)


def layer_axis(meta, rel=REL_DEPTH):
    """Index on the stored layer axis for relative depth ``rel``.

    Rule (extract/PROTOCOL.md section 5): residual-stream index
    l(r) = floor(r*L + 0.5) = meta['rel_depths_to_residual_index'][rel];
    ViT arrays store block outputs, block index l-1, listed in
    meta['layer_axis_block_indices']; LM arrays store hidden_states[l], listed
    in meta['layer_axis_hidden_state_indices']. The axis position is looked up
    in that list (for COCO, rel 0.5 is always axis 1 of the 4 stored depths
    {0.25, 0.5, 0.75, 1.0}). round(rel*(n_stored-1)) is NOT used: on the 4-depth
    COCO arrays it gives axis 2 (= rel 0.75)."""
    ell = int(meta["rel_depths_to_residual_index"][rel])
    if meta["kind"] == "vit":
        return int(meta["layer_axis_block_indices"].index(ell - 1))
    return int(meta["layer_axis_hidden_state_indices"].index(ell))


def features_to_D(X, pre, *, clip_ref=None):
    """SPEC section 0 squared distances (float64) via hrhv2.prep."""
    G = prep.Geom.from_features(X, pre, clip_ref=clip_ref, keep_X=False)
    return G.D


def model_D(name, pre, clip_mode, *, ds_dir=COCO_DIR, rel=REL_DEPTH):
    """D of one model at relative depth ``rel`` on the whole stored pool (pool order).

    pre='prh_clip+l2' uses ONE clamp threshold computed on all stored pool rows
    (so every nested prefix and every draw sees the same D entries):
      clip_mode='pool-layer'  : threshold over the selected layer's coordinates;
      clip_mode='pool-stored' : over all stored layers (4 on COCO; NOT the
                                ARH all-layer threshold, which needs every layer).
    Returns (D, info)."""
    slug = MODELS[name]
    meta = sidecar(slug, ds_dir)
    ax = layer_axis(meta, rel)
    arr = np.load(os.path.join(ds_dir, slug + ".npy"), mmap_mode="r")
    X = np.asarray(arr[:, ax, :], dtype=np.float64)
    clip_ref = None
    thr = None
    if pre == "prh_clip+l2":
        if clip_mode == "pool-layer":
            clip_ref = X
        elif clip_mode == "pool-stored":
            clip_ref = np.asarray(arr, dtype=np.float64)
        else:
            raise ValueError(f"clip_mode must be one of {CLIP_MODES} for prh_clip+l2")
        thr = prep.prh_clip_threshold(clip_ref)
    D = features_to_D(X, pre, clip_ref=clip_ref)
    del clip_ref
    info = {"model": name, "slug": slug, "kind": meta["kind"], "axis": ax,
            "residual_index": int(meta["rel_depths_to_residual_index"][rel]),
            "layer_name": meta["layer_names"][ax], "d": int(X.shape[1]), "pre": pre,
            "clip_mode": clip_mode if pre == "prh_clip+l2" else None, "clip_threshold": thr,
            "npy_sha256_sidecar": meta["files_sha256"]["primary"]["sha256"]}
    return D, info


def file_sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ols_slope(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    return float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))
