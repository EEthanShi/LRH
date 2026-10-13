"""OPENLLAMA-AXIS COPY of analysis/s12_layermax_llama/indep/common_llama.py (itself a copy of
analysis/s12/common.py), sub-role I of oll-axis-extract (S0b, 2026-10-09).
Changed only: model list (LMS = openlm-research/open_llama_{3b,7b,13b}, family "OpenLLaMA";
LMS_POOLED = the 11-LM ARH panel: 5 BLOOMZ, OpenLLaMA 3b/7b/13b, llama-7b/13b/30b), the
feature path of open_llama_13b (features/wit1024_openllama_axis/), the clip-threshold source of
open_llama_13b (results/S1_2_layermax_openllama/partB/clip_thresholds_partB_ol.json, frozen Part B),
the S-E2 sensitivity source of 3b/7b (features/wit1024_openllama_axis/_e2_check/, plan section 6),
and the S0b test hooks (active only if guard.test_mode(): the 13b slug becomes synth__ol13b, whose
features are a stand-in stack, default the LLaMA-axis llama-13b features, and whose Part B clip
threshold is read from ~/oll_s0b_scratch/I/partB/). The llama-13b/30b paths of the LLaMA copy stay
(their sidecars give num_params for the pooled panel).

S1.2 debate subpanel: model registry, layer rule, feature loading.

Layer rule (task specification, documented in ANALYSIS_PLAN_S12.md):
    stored index = floor(r * (L_stored - 1) + 0.5)       (round half up)
over the layers stored in the feature file (LMs: hidden_states[0..L], i.e.
embedding output + every block, the last one post-final-norm; ViTs: the
output of every block 0..L-1, CLS token, no final norm). For every model in
the panel round-half-up and round-half-even give the same index.

The extraction sidecar / verifier rule (``sidecar_layer_index``) coincides
with this one for LMs (r*n_layers is an integer for all 8 LMs) and for r=1.0,
but for ViTs at r=0.5 it selects one block earlier (block output ell-1 with
ell = floor(r*L+0.5)). It is used only for the exploratory sensitivity check.

Preprocessing ``prh_clip+l2``: one clamp threshold per (model, dataset/pool)
computed over ALL stored layers of every sample (PRH/ARH/SOG behaviour,
SPEC_CHANGELOG C1, ``--clip-scope all-layers``), then per-layer L2.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
FEAT = os.path.join(PROJ, "features")
RES = os.path.join(PROJ, "results", "S1_2")
CACHE = os.path.join(RES, "cache")
# LLAMA-AXIS: feature directory of the two LLaMA-axis models (sidecars read for num_params)
FEAT_LLAMA = os.path.join(FEAT, "wit1024_llama_axis")
LLAMA_AXIS_STEMS = ("huggyllama__llama-13b", "huggyllama__llama-30b")
# OPENLLAMA-AXIS: the new model, its feature directory, its frozen Part B clip threshold, S-E2 source
import sys  # noqa: E402
sys.path.insert(0, os.path.join(PROJ, "analysis", "s12_layermax_openllama", "indep"))
import ol_gate  # noqa: E402  (guard.test_mode() and the S0b scratch root)
TEST = ol_gate.test_mode()
SLUG3, SLUG7 = "openlm-research__open_llama_3b", "openlm-research__open_llama_7b"
SLUG13 = "synth__ol13b" if TEST else "openlm-research__open_llama_13b"
FEAT_OL = os.path.join(FEAT, "wit1024_openllama_axis")
E2_DIR = os.path.join(FEAT_OL, "_e2_check")
NEW_STEMS = (SLUG13,)
CLIP_PARTB = (os.path.join(os.environ.get("OL_I_TEST_PARTB", os.path.join(ol_gate.SCRATCH_I, "partB")),
                           "clip_thresholds_partB_ol.json") if TEST else
              os.path.join(PROJ, "results", "S1_2_layermax_openllama", "partB", "clip_thresholds_partB_ol.json"))
TEST_FEAT13 = os.environ.get("OL_I_TEST_FEAT13", os.path.join(FEAT_LLAMA, "huggyllama__llama-13b"))  # S0b only
SENS_E2 = False   # set True by run_layermax.py --sens-e2: 3b/7b features from E2_DIR (plan section 6, S-E2)
_E2_CLIP = {}


def _feat_path(stem: str, ds: str, ext: str) -> str:
    if stem in NEW_STEMS or stem in LLAMA_AXIS_STEMS:
        if ds != "wit1024":
            raise ValueError(f"{stem} has only wit1024 features")
        if TEST and stem == SLUG13:
            return TEST_FEAT13 + ext
        return os.path.join(FEAT_OL if stem in NEW_STEMS else FEAT_LLAMA, stem + ext)
    if SENS_E2 and stem in (SLUG3, SLUG7) and ds == "wit1024":
        return os.path.join(E2_DIR, stem + ext)
    return os.path.join(FEAT, ds, stem + ext)

LMS_POOLED = [  # (feature stem, family)  -- OPENLLAMA-AXIS: the 11-LM pooled panel, ARH order
    ("bigscience__bloomz-560m", "BLOOMZ"),
    ("bigscience__bloomz-1b1", "BLOOMZ"),
    ("bigscience__bloomz-1b7", "BLOOMZ"),
    ("bigscience__bloomz-3b", "BLOOMZ"),
    ("bigscience__bloomz-7b1", "BLOOMZ"),
    (SLUG3, "OpenLLaMA"),
    (SLUG7, "OpenLLaMA"),
    (SLUG13, "OpenLLaMA"),
    ("huggyllama__llama-7b", "LLaMA"),
    ("huggyllama__llama-13b", "LLaMA"),
    ("huggyllama__llama-30b", "LLaMA"),
]
LMS = [x for x in LMS_POOLED if x[1] == "OpenLLaMA"]   # OPENLLAMA-AXIS: the run panel (3 OpenLLaMA)
VITS = [  # (feature stem, family) -- PRH/ARH "val" vision set (17)
    ("vit_tiny_patch16_224.augreg_in21k", "AugReg-in21k"),
    ("vit_small_patch16_224.augreg_in21k", "AugReg-in21k"),
    ("vit_base_patch16_224.augreg_in21k", "AugReg-in21k"),
    ("vit_large_patch16_224.augreg_in21k", "AugReg-in21k"),
    ("vit_base_patch16_224.mae", "MAE"),
    ("vit_large_patch16_224.mae", "MAE"),
    ("vit_huge_patch14_224.mae", "MAE"),
    ("vit_small_patch14_dinov2.lvd142m", "DINOv2"),
    ("vit_base_patch14_dinov2.lvd142m", "DINOv2"),
    ("vit_large_patch14_dinov2.lvd142m", "DINOv2"),
    ("vit_giant_patch14_dinov2.lvd142m", "DINOv2"),
    ("vit_base_patch16_clip_224.laion2b", "CLIP-laion2b"),
    ("vit_large_patch14_clip_224.laion2b", "CLIP-laion2b"),
    ("vit_huge_patch14_clip_224.laion2b", "CLIP-laion2b"),
    ("vit_base_patch16_clip_224.laion2b_ft_in12k", "CLIP-laion2b-ft-in12k"),
    ("vit_large_patch14_clip_224.laion2b_ft_in12k", "CLIP-laion2b-ft-in12k"),
    ("vit_huge_patch14_clip_224.laion2b_ft_in12k", "CLIP-laion2b-ft-in12k"),
]
DEPTHS = (0.5, 1.0)
PRES = ("prh_clip+l2", "l2")
PRIMARY_PRE = "prh_clip+l2"
K_PERM = 500
ALPHA = 0.05
PC_DIM = 128
SSEL_DROP_ROWS = (953,)   # wit4096_sel row overlapping S (caption); items_manifest overlap_rows


def sidecar(stem: str, ds: str = "wit1024") -> dict:
    with open(_feat_path(stem, ds, ".json")) as f:
        return json.load(f)


def n_stored(stem: str, ds: str = "wit1024") -> int:
    return int(sidecar(stem, ds)["shape"][1])


def layer_index(stem: str, r: float, ds: str = "wit1024") -> int:
    """Primary rule: floor(r * (L_stored - 1) + 0.5) over the stored layers."""
    L = n_stored(stem, ds)
    return int(math.floor(r * (L - 1) + 0.5))


def sidecar_layer_index(stem: str, r: float, ds: str = "wit1024") -> int:
    """Extraction-sidecar / verifier rule (sensitivity only)."""
    meta = sidecar(stem, ds)
    if meta["kind"] == "lm":
        return int(math.floor(r * int(meta["n_layers"]) + 0.5))
    L = int(meta["shape"][1])
    return int(math.floor(r * L + 0.5)) - 1


def n_params(stem: str) -> int:
    return int(sidecar(stem)["num_params"])


def rows_for(ds: str) -> np.ndarray:
    """Usable rows of a dataset (S_sel minus the flagged S-overlap row)."""
    if ds == "wit1024":
        return np.arange(1024)
    if ds == "wit4096_sel":
        keep = np.ones(1024, dtype=bool)
        keep[list(SSEL_DROP_ROWS)] = False
        return np.flatnonzero(keep)
    raise ValueError(ds)


def load_stack(stem: str, ds: str) -> np.ndarray:
    """Memory-mapped (m, L, d) float32 stack of a dataset (all rows)."""
    return np.load(_feat_path(stem, ds, ".npy"), mmap_mode="r")


def clip_threshold(stem: str, ds: str, pool: str | None = None) -> float:
    """All-layer PRH/ARH threshold for (model, pool), cached in
    cache/clip_thresholds.json. pool None = the dataset's usable rows;
    pool 'S+Ssel' = wit1024 rows followed by usable wit4096_sel rows."""
    from hrhv2.prep import prh_clip_threshold
    if stem in NEW_STEMS and pool is None and ds == "wit1024":   # LLAMA-AXIS
        with open(CLIP_PARTB) as f:
            return float(json.load(f)["clip"][stem]["c"])
    if SENS_E2 and stem in (SLUG3, SLUG7) and pool is None and ds == "wit1024":
        # S-E2: re-extracted features; threshold recomputed on their float64 all-layer stack (D5)
        if stem not in _E2_CLIP:
            _E2_CLIP[stem] = float(prh_clip_threshold(np.asarray(load_stack(stem, ds), dtype=np.float64)))
        return _E2_CLIP[stem]
    key = f"{stem}|{pool or ds}"
    path = os.path.join(CACHE, "clip_thresholds.json")
    tab = {}
    if os.path.exists(path):
        with open(path) as f:
            tab = json.load(f)
    if key in tab:
        return float(tab[key])
    raise KeyError(f"clip threshold {key} not precomputed (run prep.py)")


def preprocess_rows(X: np.ndarray, pre: str, c: float | None) -> np.ndarray:
    """Same arithmetic as hrhv2.prep.preprocess(X, pre, clip_ref=stack) when c is
    that stack's prh_clip_threshold: clamp to [-c, c] (prh_clip+l2 only), then
    divide by the Euclidean row norm. Returns a float64 copy."""
    X = np.array(X, dtype=np.float64, copy=True)
    if pre == "prh_clip+l2":
        np.clip(X, -c, c, out=X)
    elif pre != "l2":
        raise ValueError(pre)
    norms = np.sqrt(np.einsum("ij,ij->i", X, X))
    if np.any(norms == 0):
        raise ValueError("zero row")
    X /= norms[:, None]
    return X


def features(stem: str, r: float, pre: str, ds: str = "wit1024", *, rule: str = "primary",
             pool: str | None = None) -> tuple[np.ndarray, dict]:
    """Preprocessed float64 rows (m, d) of one layer, plus provenance."""
    if pool == "S+Ssel":
        li = layer_index(stem, r) if rule == "primary" else sidecar_layer_index(stem, r)
        a = np.asarray(load_stack(stem, "wit1024")[:, li, :], dtype=np.float64)
        b = np.asarray(load_stack(stem, "wit4096_sel")[rows_for("wit4096_sel"), li, :], dtype=np.float64)
        X = np.concatenate([a, b], axis=0)
        c = clip_threshold(stem, "wit1024", pool="S+Ssel") if pre == "prh_clip+l2" else None
    else:
        li = layer_index(stem, r, ds) if rule == "primary" else sidecar_layer_index(stem, r, ds)
        X = np.asarray(load_stack(stem, ds)[rows_for(ds), li, :], dtype=np.float64)
        c = clip_threshold(stem, ds) if pre == "prh_clip+l2" else None
    meta = {"stem": stem, "ds": pool or ds, "r": r, "layer": li, "n_stored": n_stored(stem),
            "pre": pre, "clip_c": c, "rule": rule, "d": int(X.shape[1]), "m": int(X.shape[0])}
    return preprocess_rows(X, pre, c), meta


def geom(Xp: np.ndarray, pre_label: str):
    """Geom from already-preprocessed rows (output of preprocess_rows). Same
    arithmetic as hrhv2.prep.Geom.from_features: K = Xp Xp^T, D = gram_to_dist(K)."""
    from hrhv2.prep import Geom, gram_to_dist
    K = Xp @ Xp.T
    D = gram_to_dist(K)
    return Geom(D=D, K=K, X=Xp, pre=pre_label, source="features", meta={"d": int(Xp.shape[1])})


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
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
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    return v
