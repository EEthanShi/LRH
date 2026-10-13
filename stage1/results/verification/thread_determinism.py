#!/usr/bin/env python
"""Verifier: is the hrhv2 ladder bitwise reproducible across BLAS thread counts?

Run twice with different OPENBLAS_NUM_THREADS (and HRHV2 n_jobs); each run saves the
Gram/D hashes and the ladder; a third call compares.
Usage: python thread_determinism.py run OUT.npz NJOBS | compare A.npz B.npz OUT.json
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/hrh_v2_stage1_20261008"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from agree import layer_index, FEAT, REAL_PAIRS  # noqa: E402
from hrhv2.prep import Geom  # noqa: E402
from hrhv2.ladder import ladder  # noqa: E402
from hrhv2 import structural as st  # noqa: E402


def run(out, njobs):
    res = {}
    arrs = {}
    for a, b in REAL_PAIRS:
        for r in (0.5, 1.0):
            key = f"{a}|{b}|{r}"
            XA = np.asarray(np.load(os.path.join(FEAT, a + ".npy"), mmap_mode="r")[:, layer_index(a, r), :], dtype=np.float64)
            XB = np.asarray(np.load(os.path.join(FEAT, b + ".npy"), mmap_mode="r")[:, layer_index(b, r), :], dtype=np.float64)
            A, B = Geom.from_features(XA, "l2"), Geom.from_features(XB, "l2")
            lad = ladder(A, B, n_jobs=njobs)
            nA, _ = st.knn_table(A.D, 101)
            nB, _ = st.knn_table(B.D, 101)
            res[key] = {k: v for k, v in lad.items() if not k.startswith("time_") and isinstance(v, (int, float, bool))}
            arrs[key + "|DA"] = A.D
            arrs[key + "|DB"] = B.D
            arrs[key + "|nA"] = nA
            arrs[key + "|nB"] = nB
    np.savez(out, **arrs)
    json.dump(res, open(out + ".json", "w"))


def compare(fa, fb, out):
    A, B = np.load(fa), np.load(fb)
    ja, jb = json.load(open(fa + ".json")), json.load(open(fb + ".json"))
    rep = {}
    for key in ja:
        d = {}
        for t in ("DA", "DB"):
            x, y = A[key + "|" + t], B[key + "|" + t]
            d[f"{t}_bitwise_equal"] = bool(np.array_equal(x, y))
            d[f"{t}_max_absdiff"] = float(np.max(np.abs(x - y)))
            d[f"{t}_n_entries_diff"] = int(np.sum(x != y))
        for t in ("nA", "nB"):
            x, y = A[key + "|" + t], B[key + "|" + t]
            d[f"{t}_k100_rows_differ"] = int(np.sum(np.any(np.sort(x, 1) != np.sort(y, 1), axis=1)))
        diffs = {k: (ja[key][k], jb[key][k]) for k in ja[key] if k in jb[key] and ja[key][k] != jb[key][k]}
        d["ladder_keys_differ"] = {k: v for k, v in list(diffs.items())[:40]}
        d["n_ladder_keys_differ"] = len(diffs)
        rep[key] = d
    json.dump(rep, open(out, "w"), indent=1)
    for k, v in rep.items():
        print(k, {x: y for x, y in v.items() if x != "ladder_keys_differ"})
        print("   ", list(v["ladder_keys_differ"].items())[:12])


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]))
    else:
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
