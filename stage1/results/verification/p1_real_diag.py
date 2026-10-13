#!/usr/bin/env python
"""Verifier: behaviour of the FROZEN P1 (and nulls) on real features, on S_sel
(wit_4096 items 1025-2048; PREREG: the split reserved for layer/estimator choices),
so that no S1.1 evaluation data (COCO pool) and no S1.2 data (WIT-1024) is used.

For 3 model pairs x 2 relative depths x pre=l2: crossfit_predict with the frozen
variant, m_fit = 1024, nested prefixes m in {256, 512, 1024}, k in {1, 10, 100}.
Diagnostic of WHY P1 errs: at the predicted anchors' A-neighbour ranks, compare the
fitted model's residual scale s(u) with (a) the A-distance spacing around rank k and
(b) the anchors' own realised |b - warp(u)| (in-row band-affine residual, A units).
Usage: python p1_real_diag.py OUT.json
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/hrh_v2_stage1_20261008"))
import numpy as np  # noqa: E402

from hrhv2 import predict, util  # noqa: E402
from hrhv2.prep import Geom  # noqa: E402

SEL = os.path.expanduser("~/hrh_v2_stage1_20261008/features/wit4096_sel")
PAIRS = [("vit_base_patch16_clip_224.laion2b", "vit_base_patch14_dinov2.lvd142m"),
         ("huggyllama__llama-7b", "openlm-research__open_llama_7b"),
         ("bigscience__bloomz-1b7", "vit_large_patch14_dinov2.lvd142m")]


def layer_index(name, r):
    meta = json.load(open(os.path.join(SEL, name + ".json")))
    arr = np.load(os.path.join(SEL, name + ".npy"), mmap_mode="r")
    if meta.get("kind") == "lm":
        return int(math.floor(r * int(meta["n_layers"]) + 0.5))
    return int(math.floor(r * arr.shape[1] + 0.5)) - 1


def main():
    out = []
    for a, b in PAIRS:
        for r in (0.5, 1.0):
            XA = np.asarray(np.load(os.path.join(SEL, a + ".npy"), mmap_mode="r")[:, layer_index(a, r), :], dtype=np.float64)
            XB = np.asarray(np.load(os.path.join(SEL, b + ".npy"), mmap_mode="r")[:, layer_index(b, r), :], dtype=np.float64)
            A, B = Geom.from_features(XA, "l2"), Geom.from_features(XB, "l2")
            m = A.m
            h1 = util.hash_half([f"sel{i}" for i in range(m)], "hrhv2-halves")
            cf = predict.crossfit_predict(A.D, B.D, h1, m_fit=m, m_list=[256, 512, m], ks=[1, 10, 100],
                                          variants=[predict.FROZEN_P1_VARIANT], with_nulls=True)
            fit = cf.pop("_fit")
            rec = {"pair": [a, b], "r": r, "cells": {}}
            for mm, cell in cf.items():
                for k, c in cell.items():
                    rec["cells"][f"m={mm}|k={k}"] = {x: c[x] for x in ("obs", "chance", predict.FROZEN_P1_VARIANT, "N0", "N1", "N2", "P2")}
            # diagnostic at m = 1024 on H2 anchors
            model = predict.fit_coupling(A.D, B.D, h1, variant=predict.FROZEN_P1_VARIANT)
            anchors = np.flatnonzero(~h1)
            U, Bs = predict.sorted_rows(A.D[np.ix_(np.arange(m), np.arange(m))], B.D)
            U, Bs = U[anchors], Bs[anchors]
            R, C = predict.coupling_residuals(U, Bs)
            X, unit = predict.pool_coords(U, model.pool, m)
            s_model = (unit * model.scale(X))  # A units
            diag = {}
            for k in (1, 10, 100):
                j = k - 1
                spacing = np.median(U[:, j + 1] - U[:, j])
                diag[k] = {"median_A_distance_at_rank_k": float(np.median(U[:, j])),
                           "median_A_spacing_rank_k_to_k+1": float(spacing),
                           "median_model_scale_at_rank_k": float(np.median(s_model[:, j])),
                           "median_inrow_abs_residual_at_rank_k": float(np.median(np.abs(R[:, j]))),
                           "median_inrow_abs_residual_ranks_8_64": float(np.median(np.abs(R[:, 7:64]))),
                           "median_model_scale_ranks_8_64": float(np.median(s_model[:, 7:64]))}
            rec["diag_m1024_H2"] = diag
            rec["fit_summary"] = {k: v for k, v in fit[predict.FROZEN_P1_VARIANT].items() if isinstance(v, (int, float, str, list))}
            out.append(rec)
            print(json.dumps({"pair": rec["pair"], "r": r,
                              "cells": {kk: {x: round(v, 4) for x, v in c.items()} for kk, c in rec["cells"].items() if kk.startswith("m=1024") or kk.startswith("m=256")},
                              "diag": diag}), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
