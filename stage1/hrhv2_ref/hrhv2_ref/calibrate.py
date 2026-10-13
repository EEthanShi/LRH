"""SPEC §5: permutation calibration (ARH Algorithm 1 / Algorithm 2).

* B's rows are permuted: B_pi = B[pi] (row i of B_pi is row pi(i) of B) — the
  ARH convention `sim(X, Y[pi])`.  For Gram inputs (gram=True) K_B -> K_B[pi][:, pi].
* c_alpha = the r-th smallest (1-based) of {s_obs} ∪ {s_pi}, r = ceil((1-alpha)(K+1))
  computed in exact decimal rational arithmetic (Fraction(str(alpha))).
* s_cal = max((s_obs - c_alpha)/(s_max - c_alpha), 0); if c_alpha >= s_max the
  score is 0 (ARH clamps the denominator to 1e-12, which gives the same 0 unless
  s_obs > s_max).
* p = (1 + #{s_pi >= s_obs}) / (K + 1);  uncensored = s_obs - mean(s_pi).
* Layer-max: if X_A / X_B are lists of layers, the score is max over all layer
  pairs and the same pi is applied to every layer of B inside each permutation.
* Seeds: permutation k is numpy PCG64 seeded with the first 8 bytes (big-endian)
  of sha256(f"{key}|perm|{k}"); no global RNG state.  SPEC does not pin the
  derivation, so pass `perms=` to reproduce another implementation's nulls.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .core import sha256_int


def perm_from_key(key: str, k: int, m: int):
    rng = np.random.Generator(np.random.PCG64(sha256_int(f"{key}|perm|{k}")))
    return rng.permutation(m)


def critical_rank(K: int, alpha: float) -> int:
    r = math.ceil((1 - Fraction(str(alpha))) * (K + 1))
    return max(1, min(int(r), K + 1))


def _apply_perm(XB, pi, gram):
    if gram:
        return XB[np.ix_(pi, pi)]
    return XB[pi]


def _score(XA, XB, score_fn, layered):
    if not layered:
        return float(score_fn(XA, XB))
    return max(float(score_fn(a, b)) for a in XA for b in XB)


def calibrate_ref(X_A, X_B, score_fn, *, K=500, alpha=0.05, s_max=1.0,
                  key="hrhv2_ref", perms=None, gram=False, return_null=True):
    layered = isinstance(X_A, (list, tuple))
    if layered:
        XA = [np.asarray(x, dtype=np.float64) for x in X_A]
        XB = [np.asarray(x, dtype=np.float64) for x in X_B]
        m = XA[0].shape[0]
    else:
        XA = np.asarray(X_A, dtype=np.float64)
        XB = np.asarray(X_B, dtype=np.float64)
        m = XA.shape[0]
    if perms is None:
        perms = [perm_from_key(key, k, m) for k in range(K)]
    else:
        perms = [np.asarray(p, dtype=np.int64) for p in perms]
        K = len(perms)
    s_obs = _score(XA, XB, score_fn, layered)
    null = np.empty(K, dtype=np.float64)
    for k, pi in enumerate(perms):
        if layered:
            XBp = [_apply_perm(b, pi, gram) for b in XB]
        else:
            XBp = _apply_perm(XB, pi, gram)
        null[k] = _score(XA, XBp, score_fn, layered)
    allv = np.sort(np.concatenate([[s_obs], null]))
    r = critical_rank(K, alpha)
    c = float(allv[r - 1])
    if s_max is None:
        s_cal = max(s_obs - c, 0.0)
    elif c >= s_max:
        s_cal = 0.0
    else:
        s_cal = max((s_obs - c) / (s_max - c), 0.0)
    out = {
        "s_obs": s_obs,
        "c_alpha": c,
        "rank_r": r,
        "s_cal": s_cal,
        "p_value": (1.0 + float(np.sum(null >= s_obs))) / (K + 1),
        "uncensored": s_obs - float(np.mean(null)),
        "K": K,
        "alpha": alpha,
    }
    if return_null:
        out["null"] = null
    return out
