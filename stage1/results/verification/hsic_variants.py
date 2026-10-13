#!/usr/bin/env python
"""Accuracy of float64 unbiased-HSIC / debiased-CKA formulations against the exact value
(exact_cka.py) on the real-feature Grams: current hrhv2 (raw K), Song formula on the
double-centred Gram HKH (identical in exact arithmetic), explicit U-centring."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from exact_cka import hsic_song_exact, cka_dec, REAL_PAIRS, layer_index, FEAT  # noqa: E402
from hrhv2.prep import Geom, center_gram  # noqa: E402
from hrhv2 import geometric as geo  # noqa: E402


def ucenter(K):
    m = K.shape[0]
    Kt = K.copy()
    np.fill_diagonal(Kt, 0.0)
    r = Kt.sum(axis=1)
    s = r.sum()
    U = Kt - r[:, None] / (m - 2) - r[None, :] / (m - 2) + s / ((m - 1) * (m - 2))
    np.fill_diagonal(U, 0.0)
    return U


def hsic_u(UK, UL):
    m = UK.shape[0]
    return float(np.vdot(UK, UL)) / (m * (m - 3))


res = []
for a, b in REAL_PAIRS:
    for r in (0.5, 1.0):
        for pre in ("l2", "prh_clip+l2"):
            arrA = np.load(os.path.join(FEAT, a + ".npy"), mmap_mode="r")
            arrB = np.load(os.path.join(FEAT, b + ".npy"), mmap_mode="r")
            la, lb = layer_index(a, r), layer_index(b, r)
            cA = np.asarray(arrA, dtype=np.float64) if pre != "l2" else None
            cB = np.asarray(arrB, dtype=np.float64) if pre != "l2" else None
            A = Geom.from_features(np.asarray(arrA[:, la, :], dtype=np.float64), pre, clip_ref=cA)
            B = Geom.from_features(np.asarray(arrB[:, lb, :], dtype=np.float64), pre, clip_ref=cB)
            KA, KB = A.K, B.K
            hab, rA, rB = hsic_song_exact(KA, KB)
            haa, _, _ = hsic_song_exact(KA, KA, rA, rA)
            hbb, _, _ = hsic_song_exact(KB, KB, rB, rB)
            ex = float(cka_dec(hab, haa, hbb))
            cur = geo.cka_pair(A, B)["cka_debiased"]
            KcA, KcB = center_gram(KA), center_gram(KB)
            h = [geo.hsic_unbiased(KcA, KcB), geo.hsic_unbiased(KcA, KcA), geo.hsic_unbiased(KcB, KcB)]
            hkh = h[0] / np.sqrt(h[1] * h[2])
            UA, UB = ucenter(KA), ucenter(KB)
            u = [hsic_u(UA, UB), hsic_u(UA, UA), hsic_u(UB, UB)]
            uc = u[0] / np.sqrt(u[1] * u[2])
            o = {"pair": [a, b], "r": r, "pre": pre, "exact": ex,
                 "relerr_current": abs(cur - ex) / abs(ex), "relerr_song_on_HKH": abs(hkh - ex) / abs(ex),
                 "relerr_ucentred": abs(uc - ex) / abs(ex)}
            print(json.dumps(o), flush=True)
            res.append(o)
json.dump(res, open(sys.argv[1], "w"), indent=1)
