"""PYT-AXIS COPY of analysis/s12_layermax_openllama/indep/common_ol.py (sha256 2b253cd9...; itself a copy of
analysis/s12_layermax_llama/indep/common_llama.py and analysis/s12/common.py), sub-role pyt-code-I (S0b, 2026-10-10).
Changed only (all marked PYT-AXIS): the run panel (LMS = EleutherAI/pythia-{70m,160m,410m,1b,1.4b,2.8b,6.9b}, family
"Pythia", plan section 1), the feature directory of the 7 Pythia models (features/wit1024_pythia_axis/, plan section 2),
their clip-threshold source (results/S1_2_layermax_pythia/partB/clip_thresholds_partB_pyt.json, frozen Part B, plan
section 9 B1 / V-P1), the four known models of V-P4 and the 11-LM pooled list (read-only, for V-P4 and the descriptive
18-LM panel) with their feature directories and frozen clip sources, the removal of the S-E2 sensitivity (E2 is
descriptive on this axis, plan R.4), and the S0b test hooks (active only if pyt_gate.test_mode(): the 7 Pythia slugs
become synth__pyt-<size>, whose features, sidecars and pair-file sources are 7 known models (default the 5 BLOOMZ,
llama-13b, llama-30b in size order, plan V-P9 stand-in), and whose Part B clip thresholds are read from
the shared stand-in analysis/s12_layermax_pythia/s0b_scratch/PB/partB_vp9/, VERDICTS_SPEC_PYT.md section 13).
No test name contains "EleutherAI__pythia-".

S1.2 debate subpanel: model registry, layer rule, feature loading.

Layer rule (task specification, documented in ANALYSIS_PLAN_S12.md):
    stored index = floor(r * (L_stored - 1) + 0.5)       (round half up)
over the layers stored in the feature file (LMs: hidden_states[0..L], i.e.
embedding output + every block, the last one post-final-norm; ViTs: the
output of every block 0..L-1, CLS token, no final norm).

The extraction sidecar / verifier rule (``sidecar_layer_index``) is kept for reference only.

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
import sys  # noqa: E402
sys.path.insert(0, os.path.join(PROJ, "analysis", "s12_layermax_pythia", "indep"))   # PYT-AXIS
import pyt_gate  # noqa: E402  PYT-AXIS (test_mode() and the S0b test root)
TEST = pyt_gate.test_mode()   # PYT-AXIS
# PYT-AXIS: feature directories of the non-wit1024 models (read-only)
FEAT_PYT = os.path.join(FEAT, "wit1024_pythia_axis")
FEAT_LLAMA = os.path.join(FEAT, "wit1024_llama_axis")
FEAT_OL = os.path.join(FEAT, "wit1024_openllama_axis")
SIZES = ("70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b")   # PYT-AXIS: size order = num_params order (plan 1)
REAL_SLUGS = tuple(f"EleutherAI__pythia-{s}" for s in SIZES)   # PYT-AXIS
TEST_STANDIN_DEFAULT = ("bigscience__bloomz-560m", "bigscience__bloomz-1b1", "bigscience__bloomz-1b7",
                        "bigscience__bloomz-3b", "bigscience__bloomz-7b1", "huggyllama__llama-13b",
                        "huggyllama__llama-30b")   # PYT-AXIS: plan V-P9 7-size stand-in fixture
if TEST:   # PYT-AXIS
    _src = tuple(os.environ.get("PYT_I_TEST_STANDIN", ",".join(TEST_STANDIN_DEFAULT)).split(","))
    if len(_src) != len(SIZES):
        raise SystemExit("PYT_I_TEST_STANDIN must list 7 known stems")
    SLUGS = tuple(f"synth__pyt-{s}" for s in SIZES)
    STANDIN = dict(zip(SLUGS, _src))          # synth slug -> source stem (features, sidecar, pair files)
else:
    SLUGS = REAL_SLUGS
    STANDIN = {}
SIZE_OF = dict(zip(SLUGS, SIZES))   # PYT-AXIS
NEW_STEMS = SLUGS   # PYT-AXIS: every Pythia model is new (plan R.4: 7 sizes, all new)
LLAMA_AXIS_STEMS = ("huggyllama__llama-13b", "huggyllama__llama-30b")
OL_AXIS_STEMS = ("openlm-research__open_llama_13b",)   # PYT-AXIS
TEST_ROOT = pyt_gate.TEST_ROOT   # PYT-AXIS
CLIP_PARTB = (os.path.join(os.environ.get("PYT_I_TEST_PARTB", os.path.join(  # test: the shared stand-in (spec 13)
    os.path.dirname(TEST_ROOT), "PB", "partB_vp9")), "clip_thresholds_partB_pyt.json") if TEST else
              os.path.join(PROJ, "results", "S1_2_layermax_pythia", "partB", "clip_thresholds_partB_pyt.json"))  # PYT-AXIS
CLIP_PARTB_LLAMA = os.path.join(PROJ, "results", "S1_2_layermax_llama", "partB", "clip_thresholds_partB.json")  # PYT-AXIS
CLIP_PARTB_OL = os.path.join(PROJ, "results", "S1_2_layermax_openllama", "partB", "clip_thresholds_partB_ol.json")  # PYT-AXIS


def source_stem(stem: str) -> str:   # PYT-AXIS: the known model a test stand-in reads (identity otherwise)
    return STANDIN.get(stem, stem)


def _feat_path(stem: str, ds: str, ext: str) -> str:
    stem = source_stem(stem)   # PYT-AXIS
    special = {**{s: FEAT_PYT for s in REAL_SLUGS}, **{s: FEAT_LLAMA for s in LLAMA_AXIS_STEMS},
               **{s: FEAT_OL for s in OL_AXIS_STEMS}}   # PYT-AXIS
    if stem in special:
        if ds != "wit1024":
            raise ValueError(f"{stem} has only wit1024 features")
        return os.path.join(special[stem], stem + ext)
    return os.path.join(FEAT, ds, stem + ext)


LMS_POOLED = [  # (feature stem, family)  -- the 11-LM ARH panel (read-only; descriptive 18-LM panel, plan 1)
    ("bigscience__bloomz-560m", "BLOOMZ"),
    ("bigscience__bloomz-1b1", "BLOOMZ"),
    ("bigscience__bloomz-1b7", "BLOOMZ"),
    ("bigscience__bloomz-3b", "BLOOMZ"),
    ("bigscience__bloomz-7b1", "BLOOMZ"),
    ("openlm-research__open_llama_3b", "OpenLLaMA"),
    ("openlm-research__open_llama_7b", "OpenLLaMA"),
    ("openlm-research__open_llama_13b", "OpenLLaMA"),
    ("huggyllama__llama-7b", "LLaMA"),
    ("huggyllama__llama-13b", "LLaMA"),
    ("huggyllama__llama-30b", "LLaMA"),
]
LMS = [(s, "Pythia") for s in SLUGS]   # PYT-AXIS: the run panel (7 Pythia, size order)
LMS_18 = LMS_POOLED + LMS              # PYT-AXIS: descriptive 18-LM panel (not ARH's panel)
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
# PYT-AXIS: the 4 pretraining families of plan section 1 (used only by section 5.5(d), leave-one-ViT-family-out), with
# the labels of VERDICTS_SPEC_PYT.md section 1 (in21k, MAE, DINOv2, CLIP; the two CLIP rows above form one family)
VIT_FAMILY4 = {s: {"AugReg-in21k": "in21k", "MAE": "MAE", "DINOv2": "DINOv2"}.get(f, "CLIP") for s, f in VITS}
# PYT-AXIS: plan V-P4, the 4 known pairs each implementation reruns (both pre)
KNOWN_VP4 = (("bigscience__bloomz-560m", "vit_tiny_patch16_224.augreg_in21k"),
             ("openlm-research__open_llama_3b", "vit_giant_patch14_dinov2.lvd142m"),
             ("huggyllama__llama-13b", "vit_huge_patch14_224.mae"),
             ("openlm-research__open_llama_13b", "vit_huge_patch14_clip_224.laion2b_ft_in12k"))
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


def width(stem: str) -> int:   # PYT-AXIS: d of the stored stack (plan D4)
    return int(sidecar(stem)["shape"][2])


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


def _partb_clip(path: str, stem: str) -> float:   # PYT-AXIS: {"clip": {stem: {"c": float}}} (format of the frozen files)
    with open(path) as f:
        tab = json.load(f)["clip"]
    v = tab[stem]
    return float(v["c"] if isinstance(v, dict) else v)


def clip_threshold(stem: str, ds: str, pool: str | None = None) -> float:
    """All-layer PRH/ARH threshold for (model, pool). PYT-AXIS: the 7 Pythia models (test: stand-ins) read the frozen
    Part B file; llama-13b/30b and open_llama_13b read their frozen Part B files; every other model reads the S1.2 cache
    (cache/clip_thresholds.json)."""
    if pool is None and ds == "wit1024":   # PYT-AXIS
        if stem in NEW_STEMS:
            return _partb_clip(CLIP_PARTB, stem)
        if stem in LLAMA_AXIS_STEMS:
            return _partb_clip(CLIP_PARTB_LLAMA, stem)
        if stem in OL_AXIS_STEMS:
            return _partb_clip(CLIP_PARTB_OL, stem)
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
