"""s11-select: shared design of the PREREG v1.1 (D4) estimator re-selection.

Plants (anchor rows only; nested pools = prefixes of one generated cloud), the
S_sel real-data panel, and the selection constants. Nothing here reads a COCO
file. See hrhv2/PREDICTOR_CHOICE_v1.1.md (Part A) for the pre-stated rule.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

PROJECT = os.path.expanduser("~/hrh_v2_stage1_20261008")
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from hrhv2 import plants, util  # noqa: E402
from hrhv2.prep import gram_to_dist, prh_clip_threshold  # noqa: E402

RESULTS = os.path.join(PROJECT, "results", "S1_1_select")
HALVES_SALT = "hrhv2-halves"

# ----------------------------------------------------------------------------
# plants
# ----------------------------------------------------------------------------
PLANT_M_GRID = (256, 512, 1024, 2048, 4096, 8192, 16384)
PLANT_M_FIT = 1024
PLANT_KS = (1, 10, 100)
P_LIST = (2, 4, 8)
GAUSS_C = (0.02, 0.05, 0.15)          # P-Gauss class (G0's settings below saturation)
GAUSS_SAT_C = 0.4                     # P-GaussSat (G0's most saturated setting; s11 dry run)
ROUGH_KAPPA = 0.3                     # P-Rough, as G0
SMOOTH_KAPPA = 0.03                   # P-Smooth+noise private-noise level (10x below P-Rough)
SMOOTH_ANISO = 1.5                    # P-Smooth+noise: first circle scaled by 1.5, then inversion
SELECTION_CLASSES = ("P-Gauss", "P-Aniso", "P-Conf", "P-Rough", "P-Smooth+noise")
GATE_CLASSES = {"P-Conf": "N0", "P-Aniso": "N0", "P-Rough": "N1", "P-GaussSat": "N1"}
ELIGIBLE = (0.05, 0.99)               # observed s_N range of a scored plant cell (G0 / SPEC 7)


def halves(M: int) -> np.ndarray:
    return util.hash_half(plants.nested_ids(M), HALVES_SALT)


def dist_rows(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Squared distances of the given rows to all items (SPEC 0 construction on
    raw features: K_ii + K_jj - 2 K_ij, clipped at 0, zero self-distance)."""
    X = np.asarray(X, dtype=np.float64)
    dg = np.einsum("ij,ij->i", X, X)
    D = (dg[rows, None] + dg[None, :]) - 2.0 * (X[rows] @ X.T)
    np.maximum(D, 0.0, out=D)
    D[np.arange(rows.size), rows] = 0.0
    return D


def gauss_E_rows(M: int, sigma: float, key: str, rows: np.ndarray, m_f: int):
    """Entries of hrhv2.plants.gauss_noise_matrix(M, sigma, key) for the given
    rows (all columns) and for the m_f x m_f prefix block, without forming the
    M x M matrix (same RNG stream; tested against the full matrix)."""
    rng = util.rng_from_key(f"{key}|gauss_E")
    rows = np.asarray(rows, dtype=np.int64)
    idx_of = np.full(M, -1, dtype=np.int64)
    idx_of[rows] = np.arange(rows.size)
    Er = np.zeros((rows.size, M))
    Ef = np.zeros((m_f, m_f))
    for i in range(M - 1):
        e = rng.standard_normal(M - 1 - i) * sigma
        j = idx_of[i]
        if j >= 0:
            Er[j, i + 1:] = e
        gt = rows[np.searchsorted(rows, i, side="right"):]
        if gt.size:
            Er[idx_of[gt], i] = e[gt - i - 1]
        if i < m_f - 1:
            n = m_f - 1 - i
            Ef[i, i + 1:] = e[:n]
            Ef[i + 1:, i] = e[:n]
    return Er, Ef


def gauss_sigma(prefix: str, p: int, c: float) -> float:
    ref = plants.feature_dists(plants.base_cloud(1024, p, f"{prefix}|P-Gauss|p={p}|sigma_ref"))
    return c * plants.knn_d_median(ref, 10)


def _inversion(Y: np.ndarray, c0: np.ndarray) -> np.ndarray:
    W = Y - c0
    return c0 + W / np.sum(W * W, axis=1, keepdims=True)


def smooth_noise_features(m: int, p: int, key: str, map_key: str, kappa: float = SMOOTH_KAPPA,
                          aniso: float = SMOOTH_ANISO, d_priv: int = 32):
    """P-Smooth+noise: B = sphere inversion of the anisotropically scaled
    Clifford torus (first circle x aniso) -- smooth, neither conformal nor
    linear -- and both models carry small independent private noise
    eps_a ~ N(0, s_a^2 I / d_priv), s_a^2 = kappa * median D_(10) of that
    model's smooth part on a 256-point reference cloud (shared per p via
    map_key). Small noise: s_N follows the smooth map at small m and must
    eventually fall at large m (the crossing law's rough tail)."""
    z = plants.torus_z(m, p, key)
    radii = np.ones(p)
    radii[0] = aniso
    rho0 = math.sqrt(p) / (2.0 * math.pi)
    vdir = util.rng_from_key(f"{map_key}|conf_dir").standard_normal(2 * p)
    c0 = 2.0 * rho0 * vdir / np.linalg.norm(vdir)
    YA = plants.clifford(z)
    YB = _inversion(plants.clifford(z, radii), c0)
    zr = plants.torus_z(256, p, f"{map_key}|ref")
    s2A = kappa * plants.knn_d_median(plants.feature_dists(plants.clifford(zr)), 10)
    s2B = kappa * plants.knn_d_median(plants.feature_dists(_inversion(plants.clifford(zr, radii), c0)), 10)
    eA = util.rng_from_key(f"{key}|privA").standard_normal((m, d_priv)) * math.sqrt(s2A / d_priv)
    eB = util.rng_from_key(f"{key}|privB").standard_normal((m, d_priv)) * math.sqrt(s2B / d_priv)
    XA = plants.embed(np.concatenate([YA, eA], axis=1), f"{key}|A")
    XB = plants.embed(np.concatenate([YB, eB], axis=1), f"{key}|B")
    return XA, XB, {"s2A": s2A, "s2B": s2B}


def plant_unit(plant: str, p: int, rep: int, prefix: str, c: float | None = None,
               M: int = PLANT_M_GRID[-1], m_f: int = PLANT_M_FIT) -> dict:
    """Inputs of predict_v11.run_unit for one plant replicate (direction A -> B)."""
    h1 = halves(M)
    a = np.flatnonzero(~h1[:m_f])
    info = {"plant": plant, "p": p, "rep": rep, "c": c, "prefix": prefix}
    sigma = None
    if plant in ("P-Gauss", "P-GaussSat"):
        cc = GAUSS_SAT_C if plant == "P-GaussSat" else c
        sigma = gauss_sigma(prefix, p, cc)
        key = f"{prefix}|P-Gauss|p={p}|c={cc}|rep={rep}"
        X = plants.base_cloud(M, p, key)
        DA_rows = dist_rows(X, a)
        DA_fit = plants.feature_dists(X[:m_f])
        Er, Ef = gauss_E_rows(M, sigma, key, a, m_f)
        DB_rows, DB_fit = DA_rows + Er, DA_fit + Ef
        info.update({"c": cc, "sigma": sigma, "key": key,
                     "frac_negative_DB_rows": float((DB_rows < 0).mean())})
    else:
        key = f"{prefix}|{plant}|p={p}|rep={rep}"
        if plant == "P-Aniso":
            XA, XB = plants.aniso_features(M, p, key)
        elif plant == "P-Conf":
            XA, XB = plants.conf_features(M, p, key)
        elif plant == "P-Rough":
            ref = plants.feature_dists(plants.base_cloud(256, p, f"{prefix}|P-Rough|p={p}|ref"))
            s2 = ROUGH_KAPPA * plants.knn_d_median(ref, 10)
            XA, XB = plants.rough_features(M, p, key, s2)
            info["s2"] = s2
        elif plant == "P-Smooth+noise":
            XA, XB, extra = smooth_noise_features(M, p, key, f"{prefix}|P-Smooth+noise|p={p}")
            info.update(extra)
        else:
            raise ValueError(plant)
        DA_rows, DB_rows = dist_rows(XA, a), dist_rows(XB, a)
        DA_fit, DB_fit = plants.feature_dists(XA[:m_f]), plants.feature_dists(XB[:m_f])
        info["key"] = key
    return {"DA_rows": DA_rows, "DB_rows": DB_rows, "anchor_pos": a, "DA_fit": DA_fit, "DB_fit": DB_fit,
            "h1_fit": h1[:m_f].copy(), "oracle_sigma": sigma, "info": info}


def plant_units(n_reps: int):
    """(plant, p, c, rep) of the selection design."""
    out = []
    for p in P_LIST:
        for rep in range(n_reps):
            for plant in ("P-Aniso", "P-Conf", "P-Rough", "P-Smooth+noise", "P-GaussSat"):
                out.append((plant, p, None, rep))
            for c in GAUSS_C:
                out.append(("P-Gauss", p, c, rep))
    return out


def unit_name(plant, p, c, rep):
    cc = "" if c is None else f"_c{c}"
    return f"{plant}_p{p}{cc}_r{rep}"


# ----------------------------------------------------------------------------
# S_sel (wit4096_sel minus the flagged S-overlap row)
# ----------------------------------------------------------------------------
SSEL_DIR = os.path.join(PROJECT, "features", "wit4096_sel")
SSEL_DROP_ROWS = (953,)
SSEL_M_GRID = (256, 512, 1023)
SSEL_M_FIT = 512
SSEL_KS = (1, 10, 100)
SSEL_DRAWS = 5
SSEL_DRAW_SALTS = tuple(f"hrh-v2-s11sel-draw{d}" for d in range(SSEL_DRAWS))
SSEL_REL_DEPTH = 0.5
SSEL_PRE = "prh_clip+l2"
SSEL_ELIG_LOG = math.log(1.5)

# Pair rule (fixed before any S_sel metric): models shared with the COCO S1.1
# design wherever extracted on WIT (base ViTs in COCO order DINOv2-B, CLIP-B,
# MAE-B, AugReg-B; SigLIP and Pythia are not extracted on WIT).
#   ViT-ViT : the first 4 pairs of itertools.combinations over that ViT order.
#   LM-ViT  : {BLOOMZ-1b7, OpenLLaMA-3B} x {DINOv2-B, CLIP-B}; A = the ViT (COCO O1).
#   LM-LM   : BLOOMZ-1b7 x OpenLLaMA-3B (the two COCO LMs), and three
#             same-family pairs standing in for the Pythia seed pairs:
#             BLOOMZ-1b7 x BLOOMZ-3b, OpenLLaMA-3B x OpenLLaMA-7B, and the
#             verifier's documented failure pair LLaMA-7B x OpenLLaMA-7B.
_DINO, _CLIP, _MAE, _AUG = ("vit_base_patch14_dinov2.lvd142m", "vit_base_patch16_clip_224.laion2b",
                            "vit_base_patch16_224.mae", "vit_base_patch16_224.augreg_in21k")
_B17, _B3, _OL3, _OL7, _LL7 = ("bigscience__bloomz-1b7", "bigscience__bloomz-3b", "openlm-research__open_llama_3b",
                               "openlm-research__open_llama_7b", "huggyllama__llama-7b")
SSEL_PAIRS = {
    "vit-vit": [(_DINO, _CLIP), (_DINO, _MAE), (_DINO, _AUG), (_CLIP, _MAE)],
    "lm-lm": [(_B17, _OL3), (_B17, _B3), (_OL3, _OL7), (_LL7, _OL7)],
    "lm-vit": [(_DINO, _B17), (_CLIP, _B17), (_DINO, _OL3), (_CLIP, _OL3)],
}


def ssel_items():
    import pandas as pd
    t = pd.read_parquet(os.path.join(SSEL_DIR, "items.parquet"), columns=["idx", "src_row", "overlap_S_caption0",
                                                                           "overlap_S_anycaption", "overlap_S_image"])
    t = t.sort_values("idx")
    flagged = np.flatnonzero(t[["overlap_S_caption0", "overlap_S_anycaption", "overlap_S_image"]].any(axis=1).to_numpy())
    assert flagged.tolist() == list(SSEL_DROP_ROWS), flagged
    keep = np.ones(len(t), dtype=bool)
    keep[list(SSEL_DROP_ROWS)] = False
    rows = np.flatnonzero(keep)
    ids = [f"wit4096:{int(s)}" for s in t["src_row"].to_numpy()[rows]]
    return rows, ids


def ssel_layer_index(meta: dict, r: float = SSEL_REL_DEPTH) -> int:
    """S1.2 layer rule: floor(r (L_stored - 1) + 0.5) over the stored layers."""
    L = int(meta["shape"][1])
    return int(math.floor(r * (L - 1) + 0.5))


def ssel_model_D(stem: str, rows: np.ndarray):
    """prh_clip+l2 squared distances (1023 x 1023, stored row order of the
    usable rows) at relative depth 0.5; one all-layer clamp threshold per model
    computed on the usable rows (ARH/PRH/SOG scope, SPEC_CHANGELOG C1, D5)."""
    with open(os.path.join(SSEL_DIR, stem + ".json")) as f:
        meta = json.load(f)
    arr = np.load(os.path.join(SSEL_DIR, stem + ".npy"), mmap_mode="r")
    stack = np.asarray(arr[rows], dtype=np.float64)
    thr = prh_clip_threshold(stack)
    li = ssel_layer_index(meta)
    X = np.array(stack[:, li, :], dtype=np.float64, copy=True)
    del stack
    np.clip(X, -thr, thr, out=X)
    X /= np.sqrt(np.einsum("ij,ij->i", X, X))[:, None]
    D = gram_to_dist(X @ X.T)
    return D, {"stem": stem, "kind": meta["kind"], "layer_index": li, "n_stored": int(meta["shape"][1]),
               "clip_threshold": thr, "d": int(X.shape[1]),
               "npy_sha256_sidecar": (meta.get("files_sha256") or {}).get("primary", {}).get("sha256")
               if isinstance(meta.get("files_sha256"), dict) else None}


def file_sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else (None if math.isnan(f) else ("inf" if f > 0 else "-inf"))
    return v
