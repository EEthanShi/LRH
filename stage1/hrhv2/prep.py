"""SPEC section 0: preprocessing, Gram and distance construction, tie rule.

A representation enters the ladder as a ``Geom``:

* ``Geom.from_features(X, pre)``  features (m, d) -> preprocessing -> K = X X^T
* ``Geom.from_gram(K)``           a Gram matrix given directly (plants)
* ``Geom.from_dist(D)``           a squared-distance matrix given directly
                                  (only for synthetic plants such as P-Gauss,
                                  where D_B is not Euclidean; no clipping)

For features and Grams, ``D_ij = K_ii + K_jj - 2 K_ij`` evaluated in exactly
this order, clipped below at 0, with an exactly-zero diagonal.

Tie rule (global): candidates j != i are ordered by the key (D_ij, j); MST /
Prim use (D_ij, min(i,j), max(i,j)). See ``structural.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PRE_CHOICES = ("l2", "prh_clip+l2", "none")


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------

def prh_clip_threshold(X: np.ndarray, q: float = 0.95) -> float:
    """PRH/ARH outlier threshold c.

    For each sample, the q-quantile (linear interpolation, identical to the
    default of ``torch.quantile``) of |x| over all coordinates of that sample;
    c is the mean of these over samples. ``X`` may be (m, d) or (m, L, d); in
    the latter case all coordinates of a sample (all layers) are pooled, which
    is what the vendored ARH ``remove_outliers`` does on its stacked
    (n, L, d) tensor (``flatten(start_dim=1)``).
    """
    Xa = np.abs(np.asarray(X, dtype=np.float64).reshape(X.shape[0], -1))
    return float(np.mean(np.quantile(Xa, q, axis=1)))


def preprocess(X: np.ndarray, pre: str, *, clip_ref: np.ndarray | None = None,
               q: float = 0.95) -> np.ndarray:
    """Apply preprocessing ``pre`` and return a float64 copy.

    pre="l2"          : divide each row by its Euclidean norm (zero row -> error)
    pre="prh_clip+l2" : clamp every coordinate to [-c, c] with
                        c = prh_clip_threshold(clip_ref if given else X, q),
                        then L2-normalize.
    pre="none"        : float64 copy.
    """
    if pre not in PRE_CHOICES:
        raise ValueError(f"pre must be one of {PRE_CHOICES}, got {pre!r}")
    X = np.array(X, dtype=np.float64, copy=True)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (m, d)")
    if pre == "none":
        return X
    if pre == "prh_clip+l2":
        c = prh_clip_threshold(X if clip_ref is None else clip_ref, q)
        np.clip(X, -c, c, out=X)
    norms = np.sqrt(np.einsum("ij,ij->i", X, X))
    if np.any(norms == 0):
        raise ValueError("pre=%r: a row has norm 0" % pre)
    X /= norms[:, None]
    return X


# ----------------------------------------------------------------------------
# Gram / distance construction
# ----------------------------------------------------------------------------

def gram_to_dist(K: np.ndarray, *, chunk: int = 1024) -> np.ndarray:
    """D_ij = K_ii + K_jj - 2 K_ij (this evaluation order), clipped at 0,
    diagonal set to exactly 0. Row-chunked to bound temporaries."""
    m = K.shape[0]
    dg = np.ascontiguousarray(np.diag(K), dtype=np.float64)
    D = np.empty((m, m), dtype=np.float64)
    for lo in range(0, m, chunk):
        hi = min(m, lo + chunk)
        D[lo:hi] = (dg[lo:hi, None] + dg[None, :]) - 2.0 * K[lo:hi]
    np.maximum(D, 0.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def center_gram(K: np.ndarray) -> np.ndarray:
    """K - row_mean - col_mean + grand_mean (new array)."""
    rm = K.mean(axis=1)
    cm = K.mean(axis=0)
    gm = rm.mean()
    out = K - rm[:, None]
    out -= cm[None, :]
    out += gm
    return out


def dist_to_centered_gram(D: np.ndarray) -> np.ndarray:
    """-1/2 H D H (classical MDS Gram). Equals H K H when D comes from K."""
    return -0.5 * center_gram(D)


@dataclass
class Geom:
    """One representation evaluated on m items.

    D       (m, m) float64 squared distances, zero diagonal (always present)
    K       (m, m) Gram or None (``from_dist``)
    X       preprocessed features or None (only ``from_features``)
    pre     preprocessing label (part of the criterion)
    source  'features' | 'gram' | 'dist'
    """

    D: np.ndarray
    K: np.ndarray | None = None
    X: np.ndarray | None = None
    pre: str = "none"
    source: str = "dist"
    meta: dict = field(default_factory=dict)

    # -- constructors --------------------------------------------------------
    @classmethod
    def from_features(cls, X: np.ndarray, pre: str = "l2", *, clip_ref=None,
                      keep_X: bool = True) -> "Geom":
        Xp = preprocess(X, pre, clip_ref=clip_ref)
        K = Xp @ Xp.T
        D = gram_to_dist(K)
        return cls(D=D, K=K, X=Xp if keep_X else None, pre=pre, source="features",
                   meta={"d": int(Xp.shape[1])})

    @classmethod
    def from_gram(cls, K: np.ndarray, pre: str = "none") -> "Geom":
        K = np.asarray(K, dtype=np.float64)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("K must be square")
        return cls(D=gram_to_dist(K), K=K, X=None, pre=pre, source="gram")

    @classmethod
    def from_dist(cls, D: np.ndarray, pre: str = "none") -> "Geom":
        D = np.array(D, dtype=np.float64, copy=True)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("D must be square")
        if not np.all(np.isfinite(D)):
            raise ValueError("D has non-finite entries")
        if not np.array_equal(D, D.T):
            raise ValueError("D must be exactly symmetric")
        np.fill_diagonal(D, 0.0)
        return cls(D=D, K=None, X=None, pre=pre, source="dist")

    # -- accessors -----------------------------------------------------------
    @property
    def m(self) -> int:
        return int(self.D.shape[0])

    def centered_gram(self) -> np.ndarray:
        """H K H if a Gram is available, else -1/2 H D H."""
        if self.K is not None:
            return center_gram(self.K)
        return dist_to_centered_gram(self.D)

    def hsic_gram(self) -> np.ndarray:
        """Gram used by the unbiased HSIC: K, or -1/2 H D H for dist inputs
        (documented convention; the unbiased HSIC depends on the diagonal-free
        part of K, which a distance matrix does not determine)."""
        if self.K is not None:
            return self.K
        return dist_to_centered_gram(self.D)

    def subset(self, idx: np.ndarray) -> "Geom":
        """Restrict to items ``idx`` (in that order). Distances are sliced, not
        recomputed, so nested pools see bit-identical entries."""
        idx = np.asarray(idx)
        ix = np.ix_(idx, idx)
        return Geom(D=np.ascontiguousarray(self.D[ix]),
                    K=None if self.K is None else np.ascontiguousarray(self.K[ix]),
                    X=None if self.X is None else self.X[idx],
                    pre=self.pre, source=self.source, meta=dict(self.meta))

    def permuted(self, perm: np.ndarray) -> "Geom":
        """Rows (items) permuted by perm: new item i is old item perm[i]."""
        return self.subset(perm)


def median_offdiag(D: np.ndarray) -> float:
    """Median of the strict upper triangle of D (= of all off-diagonal)."""
    from .util import upper_values
    return float(np.median(upper_values(D)))


def same_pre(a: Geom, b: Geom) -> bool:
    """Raw geometry (SPEC 2.7) is reported only for identical preprocessing."""
    return a.pre == b.pre
