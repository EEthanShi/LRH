#!/usr/bin/env python
"""Verifier: resolve CKA disagreements by EXACT arithmetic.

For the float64 Gram matrices K_A, K_B that both implementations use (K = Xp Xp^T
from the identical preprocessed features), compute the Song et al. (2012)
unbiased HSIC and the biased HSIC exactly (every float64 is a dyadic rational;
all sums/products are done in Python integers), then CKA with 60-digit Decimal
arithmetic, and report each implementation's error against this exact value.

Usage: python exact_cka.py OUT.json
"""
from __future__ import annotations

import json
import os
import sys
from decimal import Decimal, getcontext
from fractions import Fraction

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "hrhv2_ref"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from agree import REAL_PAIRS, layer_index, FEAT  # noqa: E402
from hrhv2.prep import Geom  # noqa: E402
from hrhv2 import geometric as geo  # noqa: E402
from hrhv2_ref import geometric as rgeo  # noqa: E402
import hrhv2_ref.core as rcore  # noqa: E402

getcontext().prec = 60


def dyadic(v):
    """float64 array -> (int64 mantissas M, int64 exponents E), v = M * 2^E exactly."""
    mant, ex = np.frexp(np.asarray(v, dtype=np.float64))
    M = np.ldexp(mant, 53).astype(np.int64)
    E = ex.astype(np.int64) - 53
    E = np.where(M == 0, 0, E)
    return M, E


def exact_sum(v) -> Fraction:
    M, E = dyadic(v)
    E0 = int(E.min())
    tot = 0
    for e in np.unique(E):
        s = int(M[E == e].astype(object).sum())
        tot += s << int(e - E0)
    return Fraction(tot) * (Fraction(2) ** E0)


def exact_dot(a, b) -> Fraction:
    Ma, Ea = dyadic(a)
    Mb, Eb = dyadic(b)
    P = Ma.astype(object) * Mb.astype(object)
    E = Ea + Eb
    E0 = int(E.min())
    tot = 0
    for e in np.unique(E):
        s = int(P[E == e].sum())
        tot += s << int(e - E0)
    return Fraction(tot) * (Fraction(2) ** E0)


def song_parts(K):
    """Exact zero-diagonal row sums (as Fractions) of K~."""
    m = K.shape[0]
    rows = []
    for i in range(m):
        r = np.concatenate((K[i, :i], K[i, i + 1:]))
        rows.append(exact_sum(r))
    return rows


def offdiag(K):
    m = K.shape[0]
    mask = ~np.eye(m, dtype=bool)
    return K[mask]


def hsic_song_exact(K, L, rK=None, rL=None):
    m = K.shape[0]
    t1 = exact_dot(offdiag(K), offdiag(L))
    rK = song_parts(K) if rK is None else rK
    rL = song_parts(L) if rL is None else rL
    sK = sum(rK, Fraction(0))
    sL = sum(rL, Fraction(0))
    t3 = sum((a * b for a, b in zip(rK, rL)), Fraction(0))
    val = (t1 + sK * sL / ((m - 1) * (m - 2)) - Fraction(2, m - 2) * t3) / (m * (m - 3))
    return val, rK, rL


def hsic_biased_exact(K, L):
    """tr(HKHL) = <HKH, HLH>_F computed exactly from K, L: sum_ij K_ij L_ij
    - (2/m) sum_i rK_i rL_i + (1/m^2) sK sL with full (diagonal-included) sums."""
    m = K.shape[0]
    t1 = exact_dot(K.ravel(), L.ravel())
    rK = [exact_sum(K[i]) for i in range(m)]
    rL = [exact_sum(L[i]) for i in range(m)]
    sK, sL = sum(rK, Fraction(0)), sum(rL, Fraction(0))
    t2 = sum((a * b for a, b in zip(rK, rL)), Fraction(0))
    return t1 - Fraction(2, m) * t2 + sK * sL / (m * m)


def cka_dec(ab, aa, bb):
    return Decimal(ab.numerator) / Decimal(ab.denominator) / (
        (Decimal(aa.numerator) / Decimal(aa.denominator)) * (Decimal(bb.numerator) / Decimal(bb.denominator))).sqrt()


def run_one(name_a, name_b, r, pre):
    arrA = np.load(os.path.join(FEAT, name_a + ".npy"), mmap_mode="r")
    arrB = np.load(os.path.join(FEAT, name_b + ".npy"), mmap_mode="r")
    la, lb = layer_index(name_a, r), layer_index(name_b, r)
    XA = np.asarray(arrA[:, la, :], dtype=np.float64)
    XB = np.asarray(arrB[:, lb, :], dtype=np.float64)
    clipA = np.asarray(arrA, dtype=np.float64) if pre == "prh_clip+l2" else None
    clipB = np.asarray(arrB, dtype=np.float64) if pre == "prh_clip+l2" else None
    A = Geom.from_features(XA, pre, clip_ref=clipA)
    B = Geom.from_features(XB, pre, clip_ref=clipB)
    KA, KB = A.K, B.K
    main = geo.cka_pair(A, B)
    ref_ab, ref_aa, ref_bb = rgeo.hsic_song(KA, KB), rgeo.hsic_song(KA, KA), rgeo.hsic_song(KB, KB)
    ref_deb = rgeo.cka_from_hsic(ref_ab, ref_aa, ref_bb)
    ref_bia = rgeo.cka_from_hsic(rgeo.hsic_biased(KA, KB), rgeo.hsic_biased(KA, KA), rgeo.hsic_biased(KB, KB))
    hab, rKA, rKB = hsic_song_exact(KA, KB)
    haa, _, _ = hsic_song_exact(KA, KA, rKA, rKA)
    hbb, _, _ = hsic_song_exact(KB, KB, rKB, rKB)
    deb = cka_dec(hab, haa, hbb)
    bab, baa, bbb = hsic_biased_exact(KA, KB), hsic_biased_exact(KA, KA), hsic_biased_exact(KB, KB)
    bia = cka_dec(bab, baa, bbb)
    out = {"pair": [name_a, name_b], "r": r, "pre": pre, "layers": [la, lb],
           "exact_cka_debiased": float(deb), "exact_cka_biased": float(bia),
           "exact_hsicu_AB": float(hab),
           "main_cka_debiased": main["cka_debiased"], "ref_cka_debiased": ref_deb,
           "main_cka_biased": main["cka_biased"], "ref_cka_biased": ref_bia,
           "main_hsicu_AB": main["hsicu_AB"], "ref_hsicu_AB": ref_ab}
    for impl in ("main", "ref"):
        for q in ("cka_debiased", "cka_biased"):
            ex = float(deb) if q == "cka_debiased" else float(bia)
            out[f"relerr_{impl}_{q}"] = abs(out[f"{impl}_{q}"] - ex) / abs(ex)
        out[f"relerr_{impl}_hsicu_AB"] = abs(out[f"{impl}_hsicu_AB"] - float(hab)) / abs(float(hab))
    # conditioning: magnitude of the cancelling Song terms relative to the result
    t1 = exact_dot(offdiag(KA), offdiag(KB))
    out["song_cancellation_ratio_AB"] = float(abs(t1) / (abs(hab) * KA.shape[0] * (KA.shape[0] - 3)))
    return out


def main():
    res = []
    for a, b in REAL_PAIRS:
        for r in (0.5, 1.0):
            for pre in ("l2", "prh_clip+l2"):
                o = run_one(a, b, r, pre)
                print(json.dumps({k: o[k] for k in o if k.startswith(("pair", "r", "pre", "relerr", "exact_cka", "song"))}), flush=True)
                res.append(o)
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
