"""SPEC §0: inputs, preprocessing, Gram, squared distances, tie rule.

Reference implementation (hrhv2_ref), written independently of hrhv2/.
All arithmetic in float64.

Choices made where SPEC v1.0 is silent (see ladder.py module docstring for the
full list):
  * Gram is computed as X @ X.T and then made *bitwise* symmetric by mirroring
    the upper triangle.  D_ij = (K_ii + K_jj) - 2 K_ij exactly as written in
    SPEC (general formula, not 2 - 2K after L2), clipped below at 0, diagonal
    set to 0.  Because (K_ii + K_jj) is commutative in IEEE arithmetic, D is
    bitwise symmetric.
  * prh_clip threshold: per-sample linear-interpolated quantile of |x| over the
    d coordinates of *this* (m, d) array (torch.quantile / numpy 'linear'
    convention), averaged over samples.  ARH/PRH apply it to the stacked
    (n, L, d) tensor of *all layers*, i.e. one threshold per model; to
    reproduce that, pass `clip_threshold=` computed on the stacked tensor
    (helper: prh_clip_threshold on X.reshape(n, -1)).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

PRE_CHOICES = ("l2", "prh_clip+l2", "none")


def as_f64_2d(X, name="X"):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains non-finite values")
    return X


def l2_rows(X):
    """Divide each row by its Euclidean norm.  Zero rows are an error (SPEC §0)."""
    nrm = np.sqrt(np.einsum("ij,ij->i", X, X))
    if np.any(nrm == 0.0):
        bad = np.flatnonzero(nrm == 0.0)
        raise ValueError(f"rows with zero norm cannot be L2-normalised: {bad[:10].tolist()}")
    return X / nrm[:, None]


def _quantile_linear_rows(A, q):
    """Per-row quantile with linear interpolation (numpy 'linear' == torch default).

    position = q * (n - 1); value = s[lo] + (pos - lo) * (s[hi] - s[lo]).
    """
    s = np.sort(A, axis=1)
    n = s.shape[1]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[:, lo] + frac * (s[:, hi] - s[:, lo])


def prh_clip_threshold(X, q=0.95):
    """PRH/ARH `remove_outliers(exact=False)`: mean over samples of the per-sample
    q-quantile of |x|.  For an (n, L, d) stack pass X.reshape(n, L*d)."""
    X = np.asarray(X, dtype=np.float64)
    X2 = X.reshape(X.shape[0], -1)
    per = _quantile_linear_rows(np.abs(X2), q)
    return float(np.mean(per))


def preprocess(X, pre, clip_threshold=None, q=0.95):
    """Apply criterion preprocessing.  Returns (X_pre, info dict)."""
    X = as_f64_2d(X)
    info = {}
    if pre == "none":
        return X, info
    if pre == "l2":
        return l2_rows(X), info
    if pre == "prh_clip+l2":
        c = prh_clip_threshold(X, q) if clip_threshold is None else float(clip_threshold)
        info["clip_c"] = c
        Xc = np.minimum(np.maximum(X, -c), c)
        return l2_rows(Xc), info
    raise ValueError(f"unknown pre={pre!r}; choose from {PRE_CHOICES}")


def gram_sym(X):
    """K = X X^T, made bitwise symmetric by mirroring the upper triangle."""
    K = X @ X.T
    U = np.triu(K)
    return U + np.triu(K, 1).T


def sqdist_from_gram(K):
    """D_ij = (K_ii + K_jj) - 2 K_ij, clipped below at 0, zero diagonal."""
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("Gram must be square")
    if not np.array_equal(K, K.T):
        # enforce exact symmetry from the upper triangle (documented choice)
        K = np.triu(K) + np.triu(K, 1).T
    dg = np.diag(K).copy()
    D = (dg[:, None] + dg[None, :]) - 2.0 * K
    np.maximum(D, 0.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def upper(Mat):
    """Upper-triangular (i<j) entries, row-major order."""
    m = Mat.shape[0]
    iu, ju = np.triu_indices(m, 1)
    return Mat[iu, ju]


def neighbor_order(D, kmax=None, chunk=1024):
    """For each anchor i, candidates j != i ordered by the SPEC key (D_ij, j).

    Deliberately a *full* lexicographic argsort with explicit (value, index)
    keys (np.lexsort, primary key = value, secondary = column index); self is
    removed by pushing the diagonal to +inf.  Returns int64 array (m, kmax)
    (kmax defaults to m-1).
    """
    m = D.shape[0]
    if kmax is None:
        kmax = m - 1
    kmax = min(kmax, m - 1)
    out = np.empty((m, kmax), dtype=np.int64)
    cols = np.arange(m, dtype=np.int64)
    for s in range(0, m, chunk):
        e = min(m, s + chunk)
        V = D[s:e].copy()
        V[np.arange(e - s), np.arange(s, e)] = np.inf
        IDX = np.broadcast_to(cols, V.shape)
        o = np.lexsort((IDX, V), axis=-1)
        out[s:e] = o[:, :kmax]
    return out


def sha256_int(key: str, nbytes: int = 8) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:nbytes], "big")


def hash_halves(m, item_ids=None, salt="hrhv2_ref/crossfit/v1"):
    """Fixed hash split of anchors into halves.

    key_i = sha256(f"{salt}|{item_id_i}"); anchors sorted by (key, index);
    the first floor(m/2) form H1, the rest H2.  Returns boolean mask of H1.
    (SPEC §3 does not pin the hash; pass an explicit mask to `ladder_ref(H1=...)`
    to reproduce another implementation's split.)
    """
    ids = list(range(m)) if item_ids is None else list(item_ids)
    if len(ids) != m:
        raise ValueError("item_ids length must equal m")
    keys = [hashlib.sha256(f"{salt}|{iid}".encode("utf-8")).digest() for iid in ids]
    order = sorted(range(m), key=lambda i: (keys[i], i))
    H1 = np.zeros(m, dtype=bool)
    H1[order[: m // 2]] = True
    return H1
