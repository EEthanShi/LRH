"""Runtime benchmark of the full ladder (and of P1 cross-fitted prediction)
on synthetic d=768 features.

    OMP_NUM_THREADS=8 python -m hrhv2.bench_ladder --m 1024 16384 --jobs 8 --out logs/bench.json
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np

from . import plants, predict, util
from .ladder import ladder
from .prep import Geom
from .run_pair import jsonable


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[1024, 16384])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    variant = args.variant or predict.FROZEN_P1_VARIANT or predict.P1_CANDIDATES[0]
    res = {}
    for m in args.m:
        t0 = time.perf_counter()
        XA, XB = plants.bench_features(m, f"bench|m={m}")
        A = Geom.from_features(XA, "prh_clip+l2")
        B = Geom.from_features(XB, "prh_clip+l2")
        t_prep = time.perf_counter() - t0
        r = ladder(A, B, n_jobs=args.jobs)
        times = {k: v for k, v in r.items() if k.startswith("time_")}
        t1 = time.perf_counter()
        h1 = util.hash_half([str(i) for i in range(m)], "hrhv2-halves")
        mf = min(1024, m)
        model = predict.fit_coupling(A.D[:mf, :mf], B.D[:mf, :mf], h1[:mf], variant=variant)
        t_fit = time.perf_counter() - t1
        t2 = time.perf_counter()
        anchors = np.flatnonzero(~h1)
        pr = predict.predict_p1(A.D, anchors, [1, 10, 100], model)
        t_pred = time.perf_counter() - t2
        res[m] = {"prep_s": t_prep, "ladder_s": r["time_total_s"], **times,
                  "p1_fit_s_at_m1024": t_fit, "p1_predict_s_k1_10_100": t_pred, "p1_variant": variant,
                  "s_N_k10": r["s_N_k10"], "p1_k10": pr[10]["s_pred"],
                  "maxrss_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
        print(json.dumps(jsonable({m: res[m]})), flush=True)
        del A, B, r
    if args.out:
        with open(args.out, "w") as f:
            json.dump(jsonable(res), f, indent=1)


if __name__ == "__main__":
    main()
