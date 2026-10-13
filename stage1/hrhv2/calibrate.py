"""SPEC section 5: permutation calibration (ARH).

    s_pi   = score(A, B permuted by pi), pi_k = permutation from sha256(key|perm|k)
    c_alpha = ceil((1 - alpha)(K + 1))-th order statistic (1-based, ascending)
              of {s_obs} U {s_pi}
    s_cal  = max((s_obs - c_alpha) / (s_max - c_alpha), 0), s_max = 1
    p      = (1 + #{s_pi >= s_obs}) / (K + 1)
    uncensored = s_obs - mean(s_pi)

Layer-max aggregates: the SAME pi is applied to every layer of B and the max
is recomputed inside each permutation (``calibrate_layers``). Isotonic or any
other fitted quotient must be refit inside ``score`` (the caller's score
function receives the permuted Geom, so refitting happens automatically).
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, Sequence

import numpy as np

from . import util
from .prep import Geom
from . import structural


def permutation(m: int, key: str, k: int) -> np.ndarray:
    return util.rng_from_key(f"{key}|perm|{k}").permutation(m)


def order_index(K: int, alpha: float) -> int:
    """1-based rank ceil((1 - alpha)(K + 1)), computed in exact rational
    arithmetic (alpha is read through its shortest decimal repr, so
    alpha=0.05, K=19 gives exactly 19, not 20 from float rounding)."""
    a = Fraction(repr(float(alpha)))
    r = math.ceil((1 - a) * (K + 1))
    return int(max(1, min(r, K + 1)))


def threshold(s_obs: float, null: np.ndarray, alpha: float = 0.05) -> float:
    """c_alpha = the ceil((1-alpha)(K+1))-th order statistic (1-based,
    ascending) of {s_obs} U {s_pi} (SPEC 5)."""
    null = np.asarray(null, dtype=np.float64)
    comb = np.sort(np.concatenate(([float(s_obs)], null)))
    return float(comb[order_index(null.size, alpha) - 1])


def summarize(s_obs: float, null: np.ndarray, alpha: float = 0.05, s_max: float = 1.0) -> dict:
    null = np.asarray(null, dtype=np.float64)
    c = threshold(s_obs, null, alpha)
    denom = s_max - c
    s_cal = max((s_obs - c) / denom, 0.0) if denom > 0 else 0.0
    return {
        "s_obs": float(s_obs),
        "c_alpha": c,
        "s_cal": float(s_cal),
        "p_value": float((1.0 + np.sum(null >= s_obs)) / (null.size + 1)),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)) if null.size > 1 else float("nan"),
        "uncensored": float(s_obs - null.mean()),
        # reference only (not the SPEC threshold): ARH's compute_null_summary
        # uses np.quantile(null U {s_obs}, 1 - alpha) with linear interpolation
        "c_alpha_arh_npquantile": float(np.quantile(np.append(null, s_obs), 1.0 - alpha)),
        "K": int(null.size),
        "alpha": alpha,
    }


def calibrate(score: Callable[[Geom, Geom], float], A: Geom, B: Geom, *, key: str,
              K: int = 500, alpha: float = 0.05, s_max: float = 1.0,
              return_null: bool = False) -> dict:
    """Generic calibration: ``score(A, B.permuted(pi))`` for K sha256-seeded pi."""
    m = A.m
    s_obs = float(score(A, B))
    null = np.empty(K)
    for k in range(K):
        pi = permutation(m, key, k)
        null[k] = score(A, B.permuted(pi))
    out = summarize(s_obs, null, alpha, s_max)
    out["key"] = key
    if return_null:
        out["null"] = null
    return out


def calibrate_layers(score: Callable[[Geom, Geom], float], As: Sequence[Geom],
                     Bs: Sequence[Geom], *, key: str, K: int = 500, alpha: float = 0.05,
                     s_max: float = 1.0, agg=np.max) -> dict:
    """Aggregation-aware calibration: same pi for every layer of B; the
    aggregate (default max) is recomputed inside each permutation."""
    m = As[0].m
    S = np.array([[score(a, b) for b in Bs] for a in As])
    T_obs = float(agg(S))
    null = np.empty(K)
    for k in range(K):
        pi = permutation(m, key, k)
        Bp = [b.permuted(pi) for b in Bs]
        null[k] = float(agg(np.array([[score(a, b) for b in Bp] for a in As])))
    out = summarize(T_obs, null, alpha, s_max)
    out["key"] = key
    out["argmax"] = [int(v) for v in np.unravel_index(np.argmax(S), S.shape)]
    return out


def calibrate_knn(nA: np.ndarray, nB: np.ndarray, k: int, *, key: str, K: int = 500,
                  alpha: float = 0.05) -> dict:
    """Fast exact-path calibration of s_N(k) from precomputed neighbour tables.

    For B permuted by pi (new item i = old item pi[i]) the neighbour list of i
    is pi^{-1}[N_B(pi[i])]. This equals recomputing kNN on the permuted matrix
    whenever there are no exact distance ties at the k-boundary (the tie rule
    breaks ties by index, which a relabelling changes); callers should check
    the near-tie fraction (reported by the ladder) is 0.
    """
    m = nA.shape[0]
    s_obs = structural.knn_overlap(nA, nB, k)
    a = nA[:, :k]
    null = np.empty(K)
    for kk in range(K):
        pi = permutation(m, key, kk)
        inv = np.empty(m, dtype=np.int64)
        inv[pi] = np.arange(m)
        bp = inv[nB[pi, :k]]
        null[kk] = structural.knn_overlap(a, bp, k)
    out = summarize(s_obs, null, alpha)
    out["key"] = key
    return out
