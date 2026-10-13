"""SPEC §2: geometric criteria, and SPEC §4 projector.

CKA is computed from the explicit HSIC formulas in the papers:
  * biased  — Gretton et al. (2005): HSIC_b = (m-1)^-2 tr(K H L H), H = I - 11^T/m
    (explicit centring matrix; the (m-1)^-2 factor cancels in CKA);
  * debiased — Song et al. (2012, JMLR 13, eq. 5):
    HSIC_u = [tr(K~ L~) + 1'K~1 1'L~1 / ((m-1)(m-2)) - 2/(m-2) 1'K~L~1] / (m(m-3)),
    K~ = K with zero diagonal.  No epsilon is added to the denominator (ARH adds
    1e-8, PRH adds 1e-6; see `cka_debiased_eps`).
"""
from __future__ import annotations

import numpy as np

from .core import upper


# ---------------------------------------------------------------- Gs / G ----

def scale_normalised_geometry(DA, DB):
    a = upper(DA)
    b = upper(DB)
    nuA = a.mean()
    nuB = b.mean()
    r = a / nuA - b / nuB
    return float(np.sqrt(np.mean(r * r))), float(np.max(np.abs(r)))


def raw_geometry(DA, DB):
    r = upper(DA) - upper(DB)
    return float(np.sqrt(np.mean(r * r))), float(np.max(np.abs(r)))


# ------------------------------------------------------------------- CKA ----

EXPLICIT_H_MAX_M = 6000


def hsic_biased(K, L):
    """Gretton et al. 2005 biased HSIC (unnormalised: tr(KHLH))."""
    m = K.shape[0]
    if m <= EXPLICIT_H_MAX_M:
        H = np.eye(m) - np.full((m, m), 1.0 / m)
        Kc = H @ K @ H
        Lc = H @ L @ H
    else:  # algebraically identical double centring, to bound O(m^3) cost
        Kc = K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()
        Lc = L - L.mean(0, keepdims=True) - L.mean(1, keepdims=True) + L.mean()
    return float(np.sum(Kc * Lc))  # = tr(Kc Lc) for symmetric matrices


def hsic_song(K, L):
    """Song et al. 2012 unbiased HSIC estimator (eq. 5)."""
    m = K.shape[0]
    if m < 4:
        raise ValueError("unbiased HSIC needs m >= 4")
    Kt = np.array(K, dtype=np.float64, copy=True)
    Lt = np.array(L, dtype=np.float64, copy=True)
    np.fill_diagonal(Kt, 0.0)
    np.fill_diagonal(Lt, 0.0)
    t1 = float(np.sum(Kt * Lt.T))
    t2 = float(Kt.sum()) * float(Lt.sum()) / ((m - 1) * (m - 2))
    t3 = 2.0 / (m - 2) * float(Kt.sum(axis=1) @ Lt.sum(axis=0))  # 1'K~L~1
    return (t1 + t2 - t3) / (m * (m - 3))


def cka_from_hsic(hab, haa, hbb, eps=0.0):
    prod = haa * hbb
    if prod <= 0:
        return float("nan")
    return float(hab / (np.sqrt(prod) + eps))


def cka_biased(KA, KB):
    return cka_from_hsic(hsic_biased(KA, KB), hsic_biased(KA, KA), hsic_biased(KB, KB))


def cka_debiased(KA, KB, eps=0.0):
    return cka_from_hsic(hsic_song(KA, KB), hsic_song(KA, KA), hsic_song(KB, KB), eps=eps)


# ------------------------------------------------------------------- RSA ----

def rsa_pearson(DA, DB):
    a = upper(DA)
    b = upper(DB)
    ac = a - a.mean()
    bc = b - b.mean()
    den = np.sqrt(np.sum(ac * ac) * np.sum(bc * bc))
    return float(np.sum(ac * bc) / den) if den > 0 else float("nan")


# ------------------------------------------------------------------- SOG ----

def quantile_linear(v, q):
    s = np.sort(np.asarray(v, dtype=np.float64))
    n = s.size
    pos = q * (n - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    return float(s[lo] + (pos - lo) * (s[hi] - s[lo]))


def quantile_linear_doubled(s_sorted, q):
    """numpy-'linear' q-quantile of the multiset in which every element of the
    sorted array s appears exactly twice (= the ordered off-diagonal entries of a
    symmetric matrix whose unordered entries are s), without materialising it:
    N = 2n, pos = q (N - 1), element t_j = s[j // 2]."""
    s = np.asarray(s_sorted, dtype=np.float64)
    N = 2 * s.size
    pos = q * (N - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, N - 1)
    tl = s[lo // 2]
    th = s[hi // 2]
    return float(tl + (pos - lo) * (th - tl))


SOG_Q_MODES = ("ordered", "unordered")


def sog_scale(D, q=0.9, mode="ordered"):
    """Q_q of the positive off-diagonal plain distances d = sqrt(D).

    mode="ordered" (default): quantile over the off-diagonal *matrix entries*, i.e.
      every unordered pair counted twice.  This is the literal SPEC reading and is
      exactly what the pinned SOG code does (be654a37, prh/topology_alignment.py
      `_normalize_distance_array`: np.quantile(D[D > 0], q) on the full matrix;
      the kNN path `_normalize_distance_matrix` uses torch.quantile(D[offdiag], q),
      which is the same set when no off-diagonal distance is exactly 0).
    mode="unordered": quantile over the m(m-1)/2 unordered pairs (v0.1 behaviour).
    The two differ by ~half a sorted-index step; at tau = 1e-4 this changes the
    SOG weights at the 1e-3..1e-2 level, so the convention matters for SPEC §8."""
    d = np.sqrt(upper(D))
    d = d[d > 0]
    if mode == "ordered":
        return quantile_linear_doubled(np.sort(d), q)
    if mode == "unordered":
        return quantile_linear(d, q)
    raise ValueError(f"unknown SOG quantile mode {mode!r}; choose from {SOG_Q_MODES}")


SOG_EPS = 1e-12


def sog_weights(dA, dB, QA, QB, tau, eps=SOG_EPS):
    """w = exp(-|log max(dA/QA, eps) - log max(dB/QB, eps)| / tau), eps = 1e-12.

    The eps clamp of the *normalised* distance before the log is the pinned SOG
    convention (layerwise_engine._knn_log_distance_values_from_layer:
    log(D.clamp_min(1e-12)); topology_alignment._mst_distance_overlap_similarity_from_cached:
    log(max(D, 1e-12))).  Consequences: both distances zero -> w = 1; exactly one
    zero -> w = exp(-|log eps - log d~|/tau) (tiny but > 0).  Pass eps=0 to get the
    v0.1 rule (one zero -> w = 0)."""
    dA = np.asarray(dA, dtype=np.float64)
    dB = np.asarray(dB, dtype=np.float64)
    if eps > 0:
        la = np.log(np.maximum(dA / QA, eps))
        lb = np.log(np.maximum(dB / QB, eps))
        return np.exp(-np.abs(la - lb) / tau)
    w = np.zeros(dA.shape, dtype=np.float64)
    both0 = (dA == 0) & (dB == 0)
    pos = (dA > 0) & (dB > 0)
    lr = np.abs(np.log(dA[pos] / QA) - np.log(dB[pos] / QB))
    w[pos] = np.exp(-lr / tau)
    w[both0] = 1.0
    return w


# ------------------------------------------------------------- projector ----

def _numerical_rank(Xc):
    s = np.linalg.svd(Xc, compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0
    tol = s[0] * max(Xc.shape) * np.finfo(np.float64).eps
    return int(np.sum(s > tol))


def projector(X):
    """P = Xc (Xc^T Xc)^+ Xc^T for column-centred X (literal SPEC formula)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    G = Xc.T @ Xc
    Gp = np.linalg.pinv(G, hermitian=True)
    P = Xc @ Gp @ Xc.T
    return 0.5 * (P + P.T), _numerical_rank(Xc)


def projector_distance(XA, XB):
    """||P_A - P_B||_F^2 / (r_A + r_B)  (= /(2d) when both have full rank d).
    Also returns the mean squared canonical correlation: singular values of
    U_A^T U_B, with U the left singular vectors of the centred X truncated to rank."""
    m = XA.shape[0]
    if XA.shape[1] >= m or XB.shape[1] >= m:
        return float("nan"), float("nan"), None, None
    PA, rA = projector(XA)
    PB, rB = projector(XB)
    Dp = PA - PB
    pd = float(np.sum(Dp * Dp) / (rA + rB))
    # CCA cross-check: canonical correlations = singular values of QA^T QB
    XAc = XA - XA.mean(0, keepdims=True)
    XBc = XB - XB.mean(0, keepdims=True)
    UA, sA, _ = np.linalg.svd(XAc, full_matrices=False)
    UB, sB, _ = np.linalg.svd(XBc, full_matrices=False)
    QA = UA[:, :rA]
    QB = UB[:, :rB]
    rho = np.linalg.svd(QA.T @ QB, compute_uv=False)
    rho = np.clip(rho, 0.0, 1.0)
    cca_msq = float(np.mean(rho[: min(rA, rB)] ** 2))
    return pd, cca_msq, rA, rB
