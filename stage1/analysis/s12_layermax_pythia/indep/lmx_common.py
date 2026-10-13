"""PYT-AXIS COPY of analysis/s12_layermax_openllama/indep/lmx_common.py (sha256 d0cc9b81...), sub-role pyt-code-I
(S0b, 2026-10-10). Changed only (marked PYT-AXIS): registry module (common_pyt, the Pythia-axis copy), output path
(results/S1_2_layermax_pythia/indep via pyt_gate.out_root(); S0b test mode: analysis/s12_layermax_pythia/
s0b_scratch/I/out), the frozen indep output roots read for V-P4 and the descriptive 18-LM panel (8-LM run, LLaMA axis,
OpenLLaMA axis; read-only), the thread variable PYT_THREADS (launch_pyt.sh), the size ranks of the Pythia axis, and the
removal of the S-E2 sensitivity root. The ARH ranks of the 11 known LMs are kept for the descriptive panel only:
Pythia is not in ARH's ranked list (plan R.4, section 7 "ARH pooled reading: not applicable").
(The OpenLLaMA copy changed: registry module, output path, ARH rank of open_llama_13b, the LLaMA-axis indep path.)

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

sys.dont_write_bytecode = True   # PYT-AXIS: no __pycache__ in the frozen code tree
PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "analysis", "s12"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # PYT-AXIS: common_pyt, pyt_gate

import numpy as np  # noqa: E402

import common_pyt as C  # noqa: E402  (PYT-AXIS copy of the S1.2 registry / preprocessing)
import pyt_gate  # noqa: E402  PYT-AXIS
from hrhv2 import structural as st, geometric as geo, calibrate as cal  # noqa: E402
from hrhv2.prep import prh_clip_threshold, gram_to_dist, center_gram, median_offdiag  # noqa: E402

OUT = pyt_gate.out_root()   # PYT-AXIS: results/S1_2_layermax_pythia/indep (S0b test: s0b_scratch/I/out)
OLD_INDEP = os.path.join(PROJ, "results", "S1_2_layermax", "indep")   # frozen 8-LM run (read-only)
LLAMA_INDEP = os.path.join(PROJ, "results", "S1_2_layermax_llama", "indep")   # LLaMA-axis I run (read-only)
OL_INDEP = os.path.join(PROJ, "results", "S1_2_layermax_openllama", "indep")   # PYT-AXIS: OpenLLaMA-axis I (read-only)
THREADS = int(os.environ.get("PYT_THREADS", "1"))   # PYT-AXIS: set by launch_pyt.sh
K_PERM = 500
ALPHA = 0.05
KNN = 10
M = 1024
PRES = ("prh_clip+l2", "l2")

# ARH ordering = position in aristotelian.prh.prh_models.get_models("val") (1-based); 11 known LMs only
# (PYT-AXIS: used by the descriptive 18-LM panel's labels; no Pythia model has an ARH rank)
ARH_RANK = {
    "bigscience__bloomz-560m": 1, "bigscience__bloomz-1b1": 2, "bigscience__bloomz-1b7": 3,
    "bigscience__bloomz-3b": 4, "bigscience__bloomz-7b1": 5,
    "openlm-research__open_llama_3b": 6, "openlm-research__open_llama_7b": 7, "openlm-research__open_llama_13b": 8,
    "huggyllama__llama-7b": 9, "huggyllama__llama-13b": 10, "huggyllama__llama-30b": 11,
}
LM_STEMS = [s for s, _ in C.LMS]                 # PYT-AXIS: the 7 Pythia models, size order (test: stand-ins)
SIZE_RANK = {s: i + 1 for i, s in enumerate(LM_STEMS)}   # PYT-AXIS: plan section 1 "Ranks and labels"
POOLED_LM_STEMS = [s for s, _ in C.LMS_POOLED]   # 11 known LMs (read-only)
VIT_STEMS = [s for s, _ in C.VITS]


def frozen_indep_root(stem: str) -> str:   # PYT-AXIS: the frozen I counterpart of a known LM (plan section 3)
    stem = C.source_stem(stem)
    if stem in C.LLAMA_AXIS_STEMS:
        return LLAMA_INDEP
    if stem in C.OL_AXIS_STEMS:
        return OL_INDEP
    return OLD_INDEP


def perm_key(lm: str, vit: str) -> str:
    return f"S1.2-layermax|{lm}|{vit}"


def perms(lm: str, vit: str, K: int = K_PERM) -> np.ndarray:
    return np.stack([cal.permutation(M, perm_key(lm, vit), k) for k in range(K)])


def load_stack64(stem: str) -> np.ndarray:
    return np.asarray(C.load_stack(stem, "wit1024"), dtype=np.float64)


def clip_c(stack64: np.ndarray, stem: str) -> tuple[float, float]:
    """(recomputed all-layer threshold, reference value: frozen Part B file or S1.2 cache)."""
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


def keep_aside(path: str) -> str | None:   # PYT-AXIS: no-delete rule; an output that would be replaced is renamed
    """If path exists, rename it to <path>.prev-<UTC> (never removed) and return the new name."""
    if not os.path.exists(path):
        return None
    import time as _t
    new = f"{path}.prev-{_t.strftime('%Y%m%dT%H%M%SZ', _t.gmtime())}"
    k = 1
    while os.path.exists(new):
        new = f"{path}.prev-{_t.strftime('%Y%m%dT%H%M%SZ', _t.gmtime())}-{k}"
        k += 1
    os.rename(path, new)
    return new


def write_json(path: str, obj, **kw) -> None:   # PYT-AXIS: tmp file, older output kept aside, then rename
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, **kw)
    keep_aside(path)
    os.replace(tmp, path)


def jsonable(v):
    return C.jsonable(v)
