"""S2.2 v1.1 gauge mathematics (numpy, float64, CPU). PREREG_S2_2_v1.1.md sections 2 and 7.

Sign convention (paper): C := P P^T - W^T W  (v1.0 code logged the opposite sign; every v1.1 quantity
that depends on the sign of C is computed here with the paper's sign; mu_C, ||C|| and rho do not depend on it).
Notation: M := W P (9 x 256), N := M M^T, Y := W W^T, s := sigma_min(M), G := (W W^T)^{-1}, G_bal := N^{-1/2},
u_bal := N^{-1/4} y, kappa(X, Z) := lambda_max / lambda_min of X^{-1/2} Z X^{-1/2}.
Row-vector convention for features: y is (m, 9); u = y W^{-T}; the u-metric in y-coordinates is G.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


# ----------------------------------------------------------------------------- SPD helpers
def sym(A):
    return 0.5 * (A + A.T)


def eig_fun(M, f):
    w, V = np.linalg.eigh(sym(np.asarray(M, np.float64)))
    return sym((V * f(w)) @ V.T)


def sqrtm_spd(M):
    return eig_fun(M, lambda w: np.sqrt(np.maximum(w, 0.0)))


def powm_spd(M, p):
    return eig_fun(M, lambda w: np.power(w, p))


def logm_spd(M):
    return eig_fun(M, np.log)


def expm_sym(S):
    return eig_fun(S, np.exp)


def spd_ok(M, max_cond=1e12):
    """Numerically SPD: finite, smallest eigenvalue > 0 and condition number <= max_cond (resolution R-NUM)."""
    M = np.asarray(M, np.float64)
    if not np.all(np.isfinite(M)):
        return False
    w = np.linalg.eigvalsh(sym(M))
    return bool(w[0] > 0 and w[-1] / w[0] <= max_cond)


def det_normalize(M):
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        raise ValueError("matrix not positive definite")
    return M / np.exp(logdet / M.shape[0])


def log_euclid_mean(Ms):
    return expm_sym(np.mean([logm_spd(M) for M in Ms], axis=0))


def d_ai(A, B):
    """Affine-invariant distance (v1.0 a4_mechanism_eps.py definition)."""
    w, V = np.linalg.eigh(sym(A))
    Ai = (V / np.sqrt(w)) @ V.T
    return float(np.sqrt(np.sum(np.log(np.linalg.eigvalsh(sym(Ai @ B @ Ai))) ** 2)))


def kappa(X, Z):
    """lambda_max / lambda_min of the pencil (Z, X) = cond of X^{-1/2} Z X^{-1/2}."""
    w = sla.eigh(sym(Z), sym(X), eigvals_only=True)
    return float(w.max() / w.min())


# ----------------------------------------------------------------------------- head quantities
def C_of(P, W):
    P = np.asarray(P, np.float64)
    W = np.asarray(W, np.float64)
    return P @ P.T - W.T @ W


def rho_of(P, W):
    """rho = ||C||_F / (||W^T W||_F + ||P P^T||_F) (v1.0 imb_rel; sign-free)."""
    P = np.asarray(P, np.float64)
    W = np.asarray(W, np.float64)
    A, B = W.T @ W, P @ P.T
    return float(np.linalg.norm(B - A) / (np.linalg.norm(A) + np.linalg.norm(B)))


def delta_C(P0, W0, P1, W1):
    """delta_C := ||C_end - C_0||_F / (||P_end P_end^T||_F + ||W_end^T W_end||_F)."""
    P1 = np.asarray(P1, np.float64)
    W1 = np.asarray(W1, np.float64)
    num = np.linalg.norm(C_of(P1, W1) - C_of(P0, W0))
    return float(num / (np.linalg.norm(P1 @ P1.T) + np.linalg.norm(W1.T @ W1)))


def mu_C(C0, C1):
    """mu_C := <C0, C_end>_F / (||C0||_F ||C_end||_F)."""
    return float(np.sum(C0 * C1) / (np.linalg.norm(C0) * np.linalg.norm(C1)))


def norm_metric(W):
    """det-normalized (W W^T)^{-1} (v1.0 a4 / a2 definition)."""
    W = np.asarray(W, np.float64)
    return det_normalize(np.linalg.inv(W @ W.T))


def G_of(W):
    W = np.asarray(W, np.float64)
    return np.linalg.inv(W @ W.T)


def N_of(P, W):
    M = np.asarray(W, np.float64) @ np.asarray(P, np.float64)
    return M @ M.T


def Gbal_of(P, W):
    return powm_spd(N_of(P, W), -0.5)


def s_of(P, W):
    M = np.asarray(W, np.float64) @ np.asarray(P, np.float64)
    return float(np.linalg.svd(M, compute_uv=False)[-1])


def lemmaN1_g(r):
    """g(r) = (1 + r) / (1 - r/2), valid for r < 2."""
    return (1.0 + r) / (1.0 - r / 2.0) if r < 2.0 else float("inf")


def run_gauge(P, W, P0=None, W0=None):
    """Per-run gauge-level quantities (section 7)."""
    P = np.asarray(P, np.float64)
    W = np.asarray(W, np.float64)
    C1 = C_of(P, W)
    s = s_of(P, W)
    c_end = float(np.linalg.norm(C1, 2))
    out = {"C_end_op": c_end, "C_end_fro": float(np.linalg.norm(C1)), "s": s,
           "rho_end": rho_of(P, W),
           "cond_W": float(np.linalg.cond(W)), "cond_M": float(np.linalg.cond(W @ P)),
           "PPt_fro": float(np.linalg.norm(P @ P.T)), "WtW_fro": float(np.linalg.norm(W.T @ W))}
    if P0 is not None:
        C0 = C_of(P0, W0)
        c0 = float(np.linalg.norm(C0, 2))
        dC = C1 - C0
        out.update({"C0_op": c0, "C0_fro": float(np.linalg.norm(C0)), "rho_0": rho_of(P0, W0),
                    "delta_C": delta_C(P0, W0, P, W), "dC_fro": float(np.linalg.norm(dC)),
                    "dC_op": float(np.linalg.norm(dC, 2)),
                    "growth_rel": float(np.linalg.norm(dC) / np.linalg.norm(C0)) if np.linalg.norm(C0) > 0 else float("inf"),
                    "mu_C": mu_C(C0, C1) if np.linalg.norm(C0) > 0 and np.linalg.norm(C1) > 0 else float("nan"),
                    "cond_W0": float(np.linalg.cond(np.asarray(W0, np.float64)))})
    # metric-dependent quantities: only for a numerically SPD G and N (resolution R-NUM); otherwise recorded
    G, Nm = G_of(W), N_of(P, W)
    if not (spd_ok(G) and spd_ok(Nm)):
        out["gauge_numerical_failure"] = "G or N not numerically SPD (cond > 1e12 or non-positive eigenvalue)"
        return out
    out["D"] = kappa(G, Gbal_of(P, W))
    r = c_end / s
    out["N1_arg_end"] = r
    out["N1_bound_end"] = lemmaN1_g(r)
    out["N1_vacuous_end"] = bool(r >= 2.0)
    out["N1_ok_end"] = bool(r >= 2.0 or out["D"] <= out["N1_bound_end"] * (1 + 1e-9))
    if P0 is not None:
        r2 = (out["C0_op"] + out["dC_op"]) / s
        out["N1_arg_flow"] = r2
        out["N1_bound_flow"] = lemmaN1_g(r2)
        out["N1_vacuous_flow"] = bool(r2 >= 2.0)
        out["N1_ok_flow"] = bool(r2 >= 2.0 or out["D"] <= out["N1_bound_flow"] * (1 + 1e-9))
    return out


# ----------------------------------------------------------------------------- representations
def rep_u(y, W):
    return np.asarray(y, np.float64) @ np.linalg.inv(np.asarray(W, np.float64)).T


def rep_metric(y, Gm):
    """Features whose squared distances are dy^T Gm dy: y Gm^{1/2} (Gm SPD)."""
    return np.asarray(y, np.float64) @ sqrtm_spd(Gm)


def rep_bal(y, P, W):
    """u_bal = N^{-1/4} y (row form y N^{-1/4})."""
    return np.asarray(y, np.float64) @ powm_spd(N_of(P, W), -0.25)


def balanced_factor(P, W, Q):
    """Balanced factorization of M = W P with rotation Q: W_b = N^{1/4} Q, P_b = W_b^{-1} M, so that
    P_b P_b^T = Q^T N^{-1/4} N N^{-1/4} Q = Q^T N^{1/2} Q = W_b^T W_b (C_b = 0) and W_b^{-1} y = Q^T u_bal."""
    M = np.asarray(W, np.float64) @ np.asarray(P, np.float64)
    Wb = powm_spd(M @ M.T, 0.25) @ Q
    Pb = np.linalg.solve(Wb, M)
    return Pb, Wb


def haar_orthogonal(rng, n=9):
    Z = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(Z)
    return Q * np.sign(np.diag(R))[None, :]
