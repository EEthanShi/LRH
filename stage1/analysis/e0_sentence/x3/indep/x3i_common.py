"""X3 independent check (role x3-indep): shared inputs.

Written from ANALYSIS_PLAN_E0.md v1.0 (sections 2.3, 2.4, 3.1, 6), the
S1.1 plan O3 / V2 text and X3_READINGS.md only. The X3 run role's code
(x3_scores.py, x3_report.py), the E0 code (e0_*.py) and s11copy/ are
neither imported nor read. Only numpy, pandas, hrhv2.prep and hrhv2.util
are used, as in E0's V-f.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

from hrhv2 import prep, util

ROOT = os.path.expanduser("~/hrh_v2_stage1_20261008")
M_GRID = [256, 512, 1024, 2048, 4096, 8192, 16384]
KS = [1, 10, 100]
NB = 101                      # neighbours kept per anchor (k_max + 1)
VALCOLS = [0, 1, 9, 10, 99, 100]   # sorted columns k-1, k for k in KS
N_POOL = 16384
N_DRAWS = 5
FIT_M = 1024                  # V2: at m >= 1024 the anchors are the H2 items of the 1024 prefix

DINO = "features/coco_pool/vit_base_patch14_dinov2.lvd142m.npy"
DINO_TOK = "features/coco_pool/vit_base_patch14_dinov2.lvd142m.tokmean.npy"
SCHNAUS = "features/coco_e0/x3/x3add_schnaus224.npy"

# id -> (file, axis or None)
VISION = {
    "dinov2_b": (DINO, 1),             # E0 reference: CLS block 5
    "x3_cls_r10": (DINO, 3),           # CLS block 11
    "x3_tok_r10": (DINO_TOK, 3),       # patch mean block 11
    "x3_tok_r05": (DINO_TOK, 1),       # patch mean block 5
    "x3add_schnaus224": (SCHNAUS, None),
}
ENCODERS = ["mpnet", "contriever", "qwen3emb06"]
CAPS_INDEX = "features/coco_e0/captions_index.parquet"

# sha256 values logged before this role (sidecar / STATUS / EXTRACTION_E0.json)
EXPECTED_SHA = {
    DINO: "11a5e377f2cd1ddde31c2350bb753ff544060793c2acba578e7caa4c5d2d6b6e",
    DINO_TOK: "9406da2544fe71c3700218f48aa5ba01c6aedfa30fb1cab4b789364d7d173284",
    SCHNAUS: "374f087c9e5780ec6546bbb66301d5f3db7f4c5504dfcb9eea35fc9d78bafa5e",
    CAPS_INDEX: "ee979cc81d7d69be888438b72c2cf921489f1475db1b07c23c747caec60ed76b",
    "features/coco_e0/mpnet.caps.npy": "a7c4700c822ff36d2cc648c65ba79bea7b81d1eb9d05b1d22ea978d932fd974e",
    "features/coco_e0/contriever.caps.npy": "105053fdc8e1f14031e3810b65dad7318c2d1debcff61171cb3f926bf9bb4916",
    "features/coco_e0/qwen3emb06.caps.npy": "02d6df89f7e4cac756da35f417bec81f694fbffb9c2446409a78561dc8917191",
    "features/coco_pool/items.parquet": "2e287dbcbd56af25dad6d8b331b3f82f331d10f02d2368f246df83f72c234292",
}


def P(rel: str) -> str:
    return os.path.join(ROOT, rel)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def check_sha(rel: str) -> str:
    s = sha256_file(P(rel))
    exp = EXPECTED_SHA.get(rel)
    if exp is not None and s != exp:
        raise RuntimeError(f"sha256 mismatch for {rel}: {s} != {exp}")
    return s


def write_json_new(path: str, obj) -> None:
    """Refuse to overwrite; write via a temp file in the same directory."""
    if os.path.exists(path):
        raise FileExistsError(path)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False, default=_json_default)
    os.rename(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ----------------------------------------------------------------------------
# Design (S1.1 O3; rebuilt here from hrhv2.util, then compared with design.npz)
# ----------------------------------------------------------------------------

def build_design() -> dict:
    items = pd.read_parquet(P("features/coco_pool/items.parquet"))
    ids = [int(x) for x in items["image_id"].to_numpy()]
    if len(ids) != N_POOL:
        raise RuntimeError("pool size")
    orders = np.empty((N_DRAWS, N_POOL), dtype=np.int64)
    orders[0] = np.arange(N_POOL)                     # draw 0 = stored order
    for d in range(1, N_DRAWS):
        orders[d] = util.hash_order(ids, f"hrh-v2-s11-draw{d}")
    h1 = util.hash_half([str(i) for i in ids], "hrhv2-halves")
    # draw 0 stored order should be ascending sha256("hrh-v2-coco-pool-v1:" + id)
    keys = [hashlib.sha256(f"hrh-v2-coco-pool-v1:{i}".encode()).hexdigest() for i in ids]
    stored_sorted = all(keys[j] < keys[j + 1] for j in range(len(keys) - 1))
    ref = np.load(P("results/S1_1/design/design.npz"))
    chk = {
        "orders_equal_design_npz": bool(np.array_equal(orders, ref["orders"])),
        "h1_equal_design_npz": bool(np.array_equal(h1, ref["h1"])),
        "image_id_equal_design_npz": bool(np.array_equal(np.asarray(ids, dtype=np.int64), ref["image_id"])),
        "draw0_stored_order_is_sha256_ascending": bool(stored_sorted),
        "design_npz_sha256": sha256_file(P("results/S1_1/design/design.npz")),
    }
    return {"orders": orders, "h1": h1, "image_id": np.asarray(ids, dtype=np.int64), "checks": chk}


def anchor_sets(order: np.ndarray, h1: np.ndarray, m: int):
    """Return (allH2 prefix positions, obs-anchor mask over them).

    allH2: every H2 item of the m prefix (positions in the draw order).
    obs (S1.1 V2): at m >= 1024 the H2 items of the 1024 prefix; at
    m in {256, 512} the H2 items of the m prefix.
    """
    pos = np.flatnonzero(~h1[order[:m]])
    lim = FIT_M if m >= FIT_M else m
    return pos, pos < lim


# ----------------------------------------------------------------------------
# Features and ctr (plan 2.3, 2.4)
# ----------------------------------------------------------------------------

def _l2rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1)
    if np.any(n == 0):
        raise ValueError("zero row")
    return X / n[:, None]


def ctr(X: np.ndarray) -> np.ndarray:
    """l2-normalize rows, subtract the float64 column mean of the pool rows,
    l2-normalize again."""
    X = np.asarray(X, dtype=np.float64)
    X = _l2rows(X)
    mu = X.mean(axis=0)
    X = X - mu[None, :]
    return _l2rows(X)


def load_vision(vid: str) -> np.ndarray:
    rel, axis = VISION[vid]
    check_sha(rel)
    A = np.load(P(rel), mmap_mode="r")
    X = np.array(A[:, axis, :] if axis is not None else A, dtype=np.float64)
    if X.shape[0] != N_POOL:
        raise RuntimeError("rows")
    return X


def caption_rows_n5() -> np.ndarray:
    """(16384, 5) rows of <enc>.caps.npy: per pool item, its captions sorted by
    ascending annotation id, the first five."""
    check_sha(CAPS_INDEX)
    ci = pd.read_parquet(P(CAPS_INDEX))
    if not np.array_equal(ci["cap_row"].to_numpy(), np.arange(len(ci))):
        raise RuntimeError("cap_row is not the row number")
    ci = ci.sort_values(["idx", "ann_id"], kind="stable")
    ci["r"] = ci.groupby("idx").cumcount()
    sel = ci[ci["r"] < 5]
    cnt = sel.groupby("idx").size()
    if len(cnt) != N_POOL or cnt.min() < 5:
        raise RuntimeError("an image has fewer than 5 captions")
    sel = sel.sort_values(["idx", "r"], kind="stable")
    rows = sel["cap_row"].to_numpy().reshape(N_POOL, 5)
    if not np.array_equal(sel["idx"].to_numpy().reshape(N_POOL, 5)[:, 0], np.arange(N_POOL)):
        raise RuntimeError("idx order")
    return rows


def load_text_n5(enc: str) -> np.ndarray:
    rel = f"features/coco_e0/{enc}.caps.npy"
    check_sha(rel)
    C = np.load(P(rel), mmap_mode="r")
    rows = caption_rows_n5()
    E = np.array(C[rows.reshape(-1)], dtype=np.float64)          # card embeddings -> float64
    E = _l2rows(E).reshape(N_POOL, 5, -1)
    S = E[:, 0, :].copy()
    for j in range(1, 5):                                        # ascending annotation-id order
        S += E[:, j, :]
    S /= 5.0
    return _l2rows(S)


def model_features(mid: str) -> np.ndarray:
    if mid in VISION:
        return ctr(load_vision(mid))
    enc = mid.split(".")[0]
    return ctr(load_text_n5(enc))


# ----------------------------------------------------------------------------
# Neighbour tables (full stable argsort; key (D, position in the draw order))
# ----------------------------------------------------------------------------

def knn_tables(Xc: np.ndarray, design: dict, chunk: int = 512) -> dict:
    G = prep.Geom.from_features(Xc, "none", keep_X=False)
    D = G.D
    G.K = None
    del G
    out = {"median_D": {}}
    med_full = prep.median_offdiag(D)        # m = 16384 prefix = the whole pool in every draw
    for d in range(N_DRAWS):
        o = design["orders"][d]
        for m in M_GRID:
            pos, is_obs = anchor_sets(o, design["h1"], m)
            cols = o[:m]
            n = len(pos)
            idx = np.empty((n, NB), dtype=np.int32)
            vals = np.empty((n, len(VALCOLS)), dtype=np.float64)
            exact_tie = np.zeros((n, len(KS)), dtype=bool)
            for lo in range(0, n, chunk):
                pc = pos[lo:lo + chunk]
                R = D[np.ix_(o[pc], cols)]
                R[np.arange(len(pc)), pc] = -1.0          # self first (D >= 0 elsewhere)
                S = np.argsort(R, axis=1, kind="stable")  # ties -> lower draw position first
                if not np.array_equal(S[:, 0], pc):
                    raise RuntimeError("self not first")
                nb = S[:, 1:NB + 1]
                sv = np.take_along_axis(R, nb, axis=1)
                idx[lo:lo + len(pc)] = nb
                vals[lo:lo + len(pc)] = sv[:, VALCOLS]
                for j, k in enumerate(KS):
                    exact_tie[lo:lo + len(pc), j] = sv[:, k] == sv[:, k - 1]
            if m == N_POOL:
                med = med_full
            else:
                med = prep.median_offdiag(D[np.ix_(cols, cols)])
            out[(d, m)] = {"pos": pos.astype(np.int32), "is_obs": is_obs, "idx": idx,
                           "vals": vals, "exact_tie": exact_tie}
            out["median_D"][(d, m)] = float(med)
    out["n_zero_offdiag"] = int(np.count_nonzero(D == 0) - N_POOL)
    del D
    return out


def overlaps(ia: np.ndarray, ib: np.ndarray, k: int) -> np.ndarray:
    """|first k of A  intersect  first k of B| per row (lists hold distinct ids)."""
    C = np.concatenate([ia[:, :k], ib[:, :k]], axis=1)
    C.sort(axis=1)
    return (C[:, 1:] == C[:, :-1]).sum(axis=1).astype(np.int32)
