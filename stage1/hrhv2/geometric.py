"""SPEC section 2: geometric criteria.

Gs (scale-normalized D), raw G, linear CKA (biased and debiased / unbiased
HSIC of Song et al. 2012), Pearson RSA, and the SOG distance-aware overlaps.
All sums are float64 and O(m^2); nothing is O(m^3).
"""
from __future__ import annotations

import numpy as np

from .prep import Geom, center_gram
from . import util

ARH_EPS = 1e-8  # EPS added to the CKA denominator in vendored ARH (785054b) utils.py


# ----------------------------------------------------------------------------
# Gs and raw G
# ----------------------------------------------------------------------------

def mean_offdiag(D: np.ndarray) -> float:
    m = D.shape[0]
    return float(D.sum() / (m * (m - 1)))  # diagonal is exactly 0


def _rms_max_offdiag(A: np.ndarray, B: np.ndarray, a: float, b: float, chunk: int = 1024):
    """rms and max of (a*A - b*B) over off-diagonal entries (diag of A,B is 0)."""
    m = A.shape[0]
    ss = 0.0
    mx = 0.0
    for lo in range(0, m, chunk):
        hi = min(m, lo + chunk)
        diff = a * A[lo:hi] - b * B[lo:hi]
        ss += float(np.einsum("ij,ij->", diff, diff))
        mx = max(mx, float(np.max(np.abs(diff))))
    return float(np.sqrt(ss / (m * (m - 1)))), mx


def gs(DA: np.ndarray, DB: np.ndarray) -> dict:
    """Scale-normalized geometry: D/nu(D), nu = mean off-diagonal D."""
    nA, nB = mean_offdiag(DA), mean_offdiag(DB)
    rms, mx = _rms_max_offdiag(DA, DB, 1.0 / nA, 1.0 / nB)
    return {"Gs_rms": rms, "Gs_max": mx, "nu_A": nA, "nu_B": nB}


def raw_g(DA: np.ndarray, DB: np.ndarray) -> dict:
    rms, mx = _rms_max_offdiag(DA, DB, 1.0, 1.0)
    return {"G_rms": rms, "G_max": mx}


# ----------------------------------------------------------------------------
# CKA
# ----------------------------------------------------------------------------

def cka_biased_from_centered(KcA: np.ndarray, KcB: np.ndarray) -> float:
    """HSIC = tr(Kc_A Kc_B) = <Kc_A, Kc_B>_F (both symmetric)."""
    hab = float(np.vdot(KcA.ravel(), KcB.ravel()))
    haa = float(np.vdot(KcA.ravel(), KcA.ravel()))
    hbb = float(np.vdot(KcB.ravel(), KcB.ravel()))
    return hab / np.sqrt(haa * hbb)


def hsic_unbiased(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC (Song et al. 2012, eq. 5), O(m^2), for symmetric K, L.

    With K~, L~ the matrices with zeroed diagonal:
      [ <K~,L~> + 1'K~1 1'L~1/((m-1)(m-2)) - 2/(m-2) 1'K~L~1 ] / (m(m-3)).
    """
    m = K.shape[0]
    if m < 4:
        raise ValueError("unbiased HSIC needs m >= 4")
    dK = np.diag(K).astype(np.float64)
    dL = np.diag(L).astype(np.float64)
    t1 = float(np.vdot(K.ravel(), L.ravel())) - float(np.dot(dK, dL))
    cK = K.sum(axis=0) - dK      # column sums of K~
    rL = L.sum(axis=1) - dL      # row sums of L~
    sK = float(cK.sum())
    sL = float(rL.sum())
    t2 = sK * sL / ((m - 1) * (m - 2))
    t3 = 2.0 * float(np.dot(cK, rL)) / (m - 2)
    return (t1 + t2 - t3) / (m * (m - 3))


def cka_pair(A: Geom, B: Geom) -> dict:
    """Biased and debiased linear CKA (plus the ARH-EPS variant of debiased).

    Both HSICs are evaluated on the double-centred Grams H K H. The biased
    HSIC is <HKH, HLH>_F by definition. The Song et al. (2012) unbiased HSIC
    is invariant to K_ij -> K_ij + a_i + a_j, and H K H = K + a 1' + 1 a' for
    symmetric K, so evaluating it on H K H is the same estimator in exact
    arithmetic; it avoids the large row/column-mean terms of a raw Gram that
    otherwise cancel (verifier fix F1, hrhv2/FIXES.md: on raw unit-norm Grams
    the cancelling terms exceeded the result by up to 3e5 on WIT-1024
    features, costing up to 1.2e-9 relative accuracy; on H K H the error vs
    exact rational arithmetic is <= 3e-15)."""
    KcA = A.centered_gram()
    KcB = B.centered_gram()
    biased = cka_biased_from_centered(KcA, KcB)
    hab = hsic_unbiased(KcA, KcB)
    haa = hsic_unbiased(KcA, KcA)
    hbb = hsic_unbiased(KcB, KcB)
    del KcA, KcB
    deb = hab / np.sqrt(haa * hbb) if haa > 0 and hbb > 0 else float("nan")
    deb_arh = hab / (np.sqrt(haa * hbb) + ARH_EPS) if haa * hbb >= 0 else float("nan")
    return {
        "cka_biased": float(biased),
        "cka_debiased": float(deb),
        "cka_debiased_arheps": float(deb_arh),
        "hsicu_AB": hab, "hsicu_AA": haa, "hsicu_BB": hbb,
    }


def participation_ratio(Kc: np.ndarray) -> float:
    """PR of a centered Gram: (tr Kc)^2 / ||Kc||_F^2."""
    tr = float(np.trace(Kc))
    fr = float(np.vdot(Kc.ravel(), Kc.ravel()))
    return tr * tr / fr


# ----------------------------------------------------------------------------
# Pearson RSA
# ----------------------------------------------------------------------------

def pearson_rsa(DA: np.ndarray, DB: np.ndarray) -> float:
    """Pearson correlation of the upper-triangular entries of D_A and D_B.

    Computed on the full symmetric matrices (each pair twice) with the
    diagonal contribution removed exactly; centered sums for accuracy."""
    m = DA.shape[0]
    n_off = m * (m - 1)
    xa = DA.sum() / n_off
    xb = DB.sum() / n_off
    sab = saa = sbb = 0.0
    for lo in range(0, m, 1024):
        hi = min(m, lo + 1024)
        a = DA[lo:hi] - xa
        b = DB[lo:hi] - xb
        sab += float(np.einsum("ij,ij->", a, b))
        saa += float(np.einsum("ij,ij->", a, a))
        sbb += float(np.einsum("ij,ij->", b, b))
    # remove the m diagonal terms, each (0 - xa)(0 - xb)
    sab -= m * xa * xb
    saa -= m * xa * xa
    sbb -= m * xb * xb
    return float(sab / np.sqrt(saa * sbb))


# ----------------------------------------------------------------------------
# SOG distance-aware overlaps
# ----------------------------------------------------------------------------

def sog_scale(d: np.ndarray, q: float = 0.9) -> float:
    """Q_q of the positive off-diagonal entries of the plain-distance matrix d,
    over BOTH triangles (as SOG's ``D[D > 0]``), linear interpolation (numpy /
    torch default). Uses the doubled-multiset identity on the upper triangle
    to avoid materializing m^2 values."""
    u = util.upper_values(d)
    u = u[u > 0]
    N = u.size
    if N == 0:
        return 1.0
    pos = q * (2 * N - 1)                 # position in the doubled sorted array
    lo = int(np.floor(pos))
    frac = pos - lo
    i0, i1 = lo // 2, min(lo + 1, 2 * N - 1) // 2
    kth = np.partition(u, sorted({i0, i1}))
    v0, v1 = kth[i0], kth[i1]
    return float(v0 + frac * (v1 - v0))


def sog_weights(dA_vals: np.ndarray, dB_vals: np.ndarray, sA: float, sB: float,
                tau: float, eps: float = 1e-12) -> np.ndarray:
    """w = exp(-|log dA~ - log dB~| / tau) with d~ = max(d/Q0.9, eps)."""
    a = np.maximum(dA_vals / sA, eps)
    b = np.maximum(dB_vals / sB, eps)
    return np.exp(-np.abs(np.log(a) - np.log(b)) / tau)


def sog_overlaps(DA: np.ndarray, DB: np.ndarray, nA: np.ndarray, nB: np.ndarray,
                 TA: np.ndarray, TB: np.ndarray, *, k: int = 10,
                 taus=(1.0, 0.1, 0.01, 0.001, 0.0001)) -> dict:
    """SOG kNN (k) and MST distance-aware overlaps for each tau.

    Plain distance d = sqrt(D). kNN version: (1/(m k)) sum_i sum_{j in
    N_A(i) & N_B(i)} w_ij. MST version: (1/(m-1)) sum_{e in T_A & T_B} w_e.
    """
    m = DA.shape[0]
    sA = sog_scale_from_D(DA)
    sB = sog_scale_from_D(DB)
    out = {"sog_scale_A": sA, "sog_scale_B": sB}
    # shared kNN pairs
    a = nA[:, :k]
    b = nB[:, :k]
    rows, cols = [], []
    for lo in range(0, m, 4096):
        hi = min(m, lo + 4096)
        hit = (a[lo:hi, :, None] == b[lo:hi, None, :]).any(axis=2)
        r, c = np.nonzero(hit)
        rows.append(r + lo)
        cols.append(a[lo:hi][r, c])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    dAk = np.sqrt(DA[rows, cols])
    dBk = np.sqrt(DB[rows, cols])
    shared = np.intersect1d(TA, TB, assume_unique=True)
    u, v = shared // m, shared % m
    dAt = np.sqrt(DA[u, v])
    dBt = np.sqrt(DB[u, v])
    for tau in taus:
        tag = _tau_tag(tau)
        out[f"sog_knn{k}_tau{tag}"] = float(sog_weights(dAk, dBk, sA, sB, tau).sum() / (m * k))
        out[f"sog_mst_tau{tag}"] = float(sog_weights(dAt, dBt, sA, sB, tau).sum() / (m - 1))
    return out


def sog_scale_from_D(D: np.ndarray, q: float = 0.9) -> float:
    """Q_q of positive off-diagonal plain distances sqrt(D) (sqrt is monotone,
    so the order statistics are taken on D and square-rooted after
    interpolation is done in d units)."""
    u = util.upper_values(D)
    u = u[u > 0]
    N = u.size
    if N == 0:
        return 1.0
    pos = q * (2 * N - 1)
    lo = int(np.floor(pos))
    frac = pos - lo
    i0, i1 = lo // 2, min(lo + 1, 2 * N - 1) // 2
    kth = np.partition(u, sorted({i0, i1}))
    v0, v1 = np.sqrt(kth[i0]), np.sqrt(kth[i1])
    return float(v0 + frac * (v1 - v0))


def _tau_tag(tau: float) -> str:
    return ("%g" % tau).replace(".", "p").replace("-", "m")
