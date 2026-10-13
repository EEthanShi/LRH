"""LLAMA-AXIS COPY of analysis/s12_layermax/indep/lmx_common.py (role
llama-axis-run-indep). Changed only: registry module (common_llama, a copy of
analysis/s12/common.py with the LLaMA-axis model list and paths), output path
(results/S1_2_layermax_llama/indep), ARH ranks of llama-13b (10) / llama-30b (11),
the pooled LM list, and the path of the frozen 8-LM indep outputs.

Independent ARH layer-max replication (role arh-layermax): shared helpers.

Registry, layer conventions and preprocessing arithmetic are imported from the
S1.2 module analysis/s12/common.py (read-only import). Per-layer CPU prep uses
hrhv2 float64 arithmetic only.
"""
from __future__ import annotations

import json
import math
import os
import sys

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "analysis", "s12"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # LLAMA-AXIS: common_llama

import numpy as np  # noqa: E402

import common_llama as C  # noqa: E402  (LLAMA-AXIS copy of the S1.2 registry / preprocessing)
from hrhv2 import structural as st, geometric as geo, calibrate as cal  # noqa: E402
from hrhv2.prep import prh_clip_threshold, gram_to_dist, center_gram, median_offdiag  # noqa: E402

OUT = os.path.join(PROJ, "results", "S1_2_layermax_llama", "indep")
OLD_INDEP = os.path.join(PROJ, "results", "S1_2_layermax", "indep")   # frozen 8-LM run (read-only)
K_PERM = 500
ALPHA = 0.05
KNN = 10
M = 1024
PRES = ("prh_clip+l2", "l2")

# ARH ordering = position in aristotelian.prh.prh_models.get_models("val") (1-based);
# open_llama_13b (8) and llama-13b/30b/65b (10-12) are absent from our panel.
ARH_RANK = {
    "bigscience__bloomz-560m": 1, "bigscience__bloomz-1b1": 2, "bigscience__bloomz-1b7": 3,
    "bigscience__bloomz-3b": 4, "bigscience__bloomz-7b1": 5,
    "openlm-research__open_llama_3b": 6, "openlm-research__open_llama_7b": 7,
    "huggyllama__llama-7b": 9, "huggyllama__llama-13b": 10, "huggyllama__llama-30b": 11,
}
LM_STEMS = [s for s, _ in C.LMS]                 # LLAMA-AXIS: llama-7b, -13b, -30b
POOLED_LM_STEMS = [s for s, _ in C.LMS_POOLED]   # LLAMA-AXIS: 10-LM pooled panel
VIT_STEMS = [s for s, _ in C.VITS]


def perm_key(lm: str, vit: str) -> str:
    return f"S1.2-layermax|{lm}|{vit}"


def perms(lm: str, vit: str, K: int = K_PERM) -> np.ndarray:
    return np.stack([cal.permutation(M, perm_key(lm, vit), k) for k in range(K)])


def load_stack64(stem: str) -> np.ndarray:
    return np.asarray(C.load_stack(stem, "wit1024"), dtype=np.float64)


def clip_c(stack64: np.ndarray, stem: str) -> tuple[float, float]:
    """(recomputed all-layer threshold, S1.2 cached value)."""
    return prh_clip_threshold(stack64), C.clip_threshold(stem, "wit1024")


def prep_layer(X: np.ndarray, pre: str, c: float | None):
    """hrhv2 arithmetic for one layer: preprocessed rows -> K, D, kNN table,
    centred Gram and its self-HSICs."""
    Xp = C.preprocess_rows(X, pre, c)
    K = Xp @ Xp.T
    D = gram_to_dist(K)
    idx, vals = st.knn_table(D, KNN + 1)
    Kc = center_gram(K)
    nrm2 = float(np.vdot(Kc.ravel(), Kc.ravel()))
    hu = float(geo.hsic_unbiased(Kc, Kc))
    med = median_offdiag(D)
    exact_bnd_ties = int(np.sum(vals[:, KNN] == vals[:, KNN - 1]))
    near = float(st.near_tie_fraction(vals, KNN, med))
    return {"Kc": Kc, "nbr": idx[:, :KNN].copy(), "nrm2": nrm2, "hu": hu,
            "exact_boundary_ties": exact_bnd_ties, "near_tie_frac": near}


def summarize(T: float, null: np.ndarray, alpha: float = ALPHA) -> dict:
    return cal.summarize(float(T), np.asarray(null, dtype=np.float64), alpha)


def jsonable(v):
    return C.jsonable(v)
