"""SPEC section 4: GL-invariant criterion (linear models only).

Projector P = Xc (Xc^T Xc)^+ Xc^T on column-centered X (m x m, never formed).
With U_A, U_B orthonormal bases of the column spaces (ranks r_A, r_B):

    ||P_A - P_B||_F^2 = r_A + r_B - 2 ||U_A^T U_B||_F^2

reported divided by (r_A + r_B) (= 2d when both have full rank d), in [0, 1].
Mean squared canonical correlation = ||U_A^T U_B||_F^2 / min(r_A, r_B) (the
singular values of U_A^T U_B are the canonical correlations); with equal full
ranks, projector distance = 1 - mean rho^2 exactly.
"""
from __future__ import annotations

import numpy as np


def _basis(X: np.ndarray, rtol: float | None = None):
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    if rtol is None:
        rtol = max(Xc.shape) * np.finfo(np.float64).eps
    r = int(np.sum(s > rtol * (s[0] if s.size else 0.0)))
    return U[:, :r], r


def projector_distance(XA: np.ndarray, XB: np.ndarray) -> dict:
    m = XA.shape[0]
    if XB.shape[0] != m:
        raise ValueError("row mismatch")
    if XA.shape[1] >= m or XB.shape[1] >= m:
        return {"gl_projector_dist": float("nan"), "gl_cca_mean_sq": float("nan"),
                "gl_note": "d >= m: GL criterion undefined (SPEC 4 requires d < m)"}
    UA, rA = _basis(XA)
    UB, rB = _basis(XB)
    C = UA.T @ UB
    s2 = float(np.sum(C * C))
    rho = np.linalg.svd(C, compute_uv=False)
    return {
        "gl_projector_dist": (rA + rB - 2.0 * s2) / (rA + rB),
        "gl_projector_frob2": rA + rB - 2.0 * s2,
        "gl_cca_mean_sq": s2 / min(rA, rB),
        "gl_cca_min": float(rho.min()) if rho.size else float("nan"),
        "gl_rank_A": rA, "gl_rank_B": rB,
    }
