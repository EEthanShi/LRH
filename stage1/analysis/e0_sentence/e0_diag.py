"""E0 single-model diagnostics P3i (ANALYSIS_PLAN_E0.md v1.0 §7.5; critique R1, R7). Before the guard.

For each sentence encoder (input C1) and each S1.1 language model: the mean own-10-NN overlap between its
`clip` and `l2` geometries at m = 1024 over draw 0's predicted anchors (from the single-model caches).
For Contriever: the own-10-NN overlap between the raw dot product (its card's score) and cosine at m = 1024,
same anchors. No cross-model quantity. Writes results/E0_sentence/diag/diag_singlemodel.json.

    venv/bin/python -I analysis/e0_sentence/e0_diag.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e0_common as E  # noqa: E402
from hrhv2 import structural as st  # noqa: E402

K = 10
M = 1024


def own_overlap(cache_a: str, cache_b: str, k: int = K, m: int = M) -> float:
    A = np.load(os.path.join(cache_a, "d0_knn.npz"))
    B = np.load(os.path.join(cache_b, "d0_knn.npz"))
    if not np.array_equal(A[f"pos_m{m}"], B[f"pos_m{m}"]):
        raise RuntimeError("anchor positions differ")
    return float(st.overlap_counts(A[f"idx_m{m}"], B[f"idx_m{m}"], k).mean() / k)


def topk_by_score(S: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
    """Neighbours of ``rows`` among 0..n-1 by descending score, ties by ascending position, self excluded."""
    out = np.empty((rows.size, k), np.int64)
    for i, r in enumerate(rows):
        s = S[r].astype(np.float64).copy()
        s[r] = -np.inf
        order = np.lexsort((np.arange(s.size), -s))
        out[i] = order[:k]
    return out


def dot_vs_cosine(X: np.ndarray, rows: np.ndarray, k: int = K) -> float:
    X = np.asarray(X, np.float64)
    dot = X @ X.T
    Xn = E.l2_rows(X)
    cos = Xn @ Xn.T
    a = topk_by_score(dot, rows, k)
    b = topk_by_score(cos, rows, k)
    return float(st.overlap_counts(a, b, k).mean() / k)


def run(feat_dir: str = E.FEAT, cache_root: str = E.CACHE) -> dict:
    out = {"plan": "ANALYSIS_PLAN_E0.md v1.0 §7.5 P3i", "k": K, "m": M, "draw": 0, "clip_vs_l2": {}, "created_utc": E.utcnow()}
    for enc in E.ENCODERS:
        out["clip_vs_l2"][enc] = own_overlap(E.cache_dir("clip", f"{enc}.c1", cache_root), E.cache_dir("l2", f"{enc}.c1", cache_root))
    for lm in E.OLD_TEXT:
        out["clip_vs_l2"][lm] = own_overlap(E.cache_dir("clip", lm), E.cache_dir("l2", lm))
    des = E.design("clip")
    d0 = des["draws"][0]
    rows_pos = d0["anchor_pos"]                       # positions in draw 0 order (prefix 1024)
    order = d0["order"][:M]
    X = np.load(os.path.join(feat_dir, "contriever.c1.npy"))[order]
    out["contriever_dot_vs_cosine"] = dot_vs_cosine(X, rows_pos)
    out["flag_below_0.8"] = sorted(n for n, v in out["clip_vs_l2"].items() if v < 0.8)
    return out


def main():
    E.require_milestones("plan", "code", "extract")
    res = run()
    h = E.write_json_new(os.path.join(E.DIAG, "diag_singlemodel.json"), res)
    print("diag_singlemodel.json", h, res["clip_vs_l2"], res["contriever_dot_vs_cosine"])


if __name__ == "__main__":
    main()
