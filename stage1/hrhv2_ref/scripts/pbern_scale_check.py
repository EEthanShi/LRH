"""Reference-side P-Bern check + timing at SPEC plant sizes.

Plant construction here is the reference's own (Gaussian N(0, I_p) cloud, embedded
in R^64 by a sha256-seeded random isometry, then L2-normalised); it is NOT the G0
plant builder of hrhv2.  Purpose: (i) does the reference reproduce the SPEC §7
P-Bern requirement exactly in float64, (ii) timing per part at m in {256,1024,4096}.

Usage: python pbern_scale_check.py OUT_JSON [m ...]
"""
import json
import sys
import time
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hrhv2_ref import ladder_ref  # noqa: E402
from hrhv2_ref.core import gram_sym, preprocess, sha256_int, sqdist_from_gram, upper  # noqa: E402
from hrhv2_ref import structural as st  # noqa: E402


def plant_cloud(m, p, d=64, key="hrhv2_ref/plant"):
    rng = np.random.Generator(np.random.PCG64(sha256_int(f"{key}|m={m}|p={p}")))
    Z = rng.normal(size=(m, p))
    Q, _ = np.linalg.qr(rng.normal(size=(d, p)))
    return Z @ Q.T


def main():
    out_path = sys.argv[1]
    ms = [int(a) for a in sys.argv[2:]] or [256, 1024, 4096]
    res = []
    for m in ms:
        for p in (2, 4, 8):
            X = plant_cloud(m, p)
            Xl, _ = preprocess(X, "l2")
            KA = gram_sym(Xl)
            for beta in (1.0, 3.0, 10.0):
                KB = np.exp(beta * (KA - 1.0))
                rec = {"m": m, "p": p, "beta": beta, "time": {}}
                full = {}
                for part in ("knn", "comp", "rank", "edge", "mst", "geom", "cka", "rsa", "sog", "quot"):
                    t0 = time.time()
                    o = ladder_ref(KA, KB, gram=True, parts={part})
                    rec["time"][part] = round(time.time() - t0, 2)
                    full.update(o)
                # floating-point merges created by exp saturation (ties in B absent in A)
                DA = sqdist_from_gram(KA)
                DB = sqdist_from_gram(KB)
                a, b = upper(DA), upper(DB)
                nA = np.unique(a).size
                nB = np.unique(b).size
                rec["distinct_D_A"] = int(nA)
                rec["distinct_D_B"] = int(nB)
                keep = [k for k in full if any(k.startswith(s) for s in (
                    "s_N@", "delta_N@", "comp_rand@", "comp_n_", "delta_O", "delta_E", "E_", "rows_with_ties",
                    "kendall_global", "spearman_global",
                    "s_T", "forest_share", "n_bridges", "n_shared", "F_A_", "F_B_", "prim_eq", "cka_",
                    "rsa_", "Gs_", "quot_iso_expl", "quot_rowiso_expl", "quot_affine_expl",
                    "quot_iso_xfit", "near_tie", "k_q", "sog_"))]
                rec.update({k: full[k] for k in keep})
                res.append(rec)
                print(json.dumps({k: rec[k] for k in ("m", "p", "beta", "time", "distinct_D_A", "distinct_D_B")}),
                      "s_N@all1:", all(rec[k] == 1.0 for k in rec if k.startswith("s_N@") and rec[k] == rec[k]),
                      "dO", rec["delta_O"], "dE", rec["delta_E"], "sT", rec["s_T"],
                      "iso", rec["quot_iso_expl"], flush=True)
                with open(out_path, "w") as f:
                    json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
