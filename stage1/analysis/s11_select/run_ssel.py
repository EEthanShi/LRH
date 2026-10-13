"""s11-select: S_sel real-data eligibility check (D4.4), no selection on it.

    python run_ssel.py --jobs 12 --out results/S1_1_select/ssel

12 pairs (sel_common.SSEL_PAIRS) x 5 nested draws x 2 directions, prefixes
m in {256, 512, 1023}, fit m = 512, k in {1, 10, 100}, prh_clip+l2 (all-layer
threshold on the 1023 usable rows), relative depth 0.5 (S1.2 layer rule).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sel_common as S  # noqa: E402
from hrhv2 import predict_v11 as v  # noqa: E402
from hrhv2 import util  # noqa: E402

_D = {}


def short(stem):
    return stem.split("__")[-1].replace("vit_", "").replace("_patch14", "").replace("_patch16", "")


def work(task):
    cls, a, b, d, direction, out_dir = task
    name = f"{cls}__{short(a)}__{short(b)}__d{d}__{direction}"
    path = os.path.join(out_dir, name + ".json")
    if os.path.exists(path):
        return name, "exists", 0.0
    t0 = time.perf_counter()
    try:
        rows, ids = S.ssel_items()
        DA, DB = _D[a][0], _D[b][0]
        if direction == "BA":
            DA, DB = DB, DA
        order = np.asarray(util.hash_order(ids, S.SSEL_DRAW_SALTS[d]), dtype=np.int64)
        h1 = util.hash_half(ids, S.HALVES_SALT)[order]
        DAo, DBo = DA[np.ix_(order, order)], DB[np.ix_(order, order)]
        m_f = S.SSEL_M_FIT
        anc = np.flatnonzero(~h1[:m_f])
        res = v.run_unit(DAo[anc], DBo[anc], anc, DAo[:m_f, :m_f], DBo[:m_f, :m_f], h1[:m_f],
                         m_grid=S.SSEL_M_GRID, m_f=m_f, ks=S.SSEL_KS, variants=v.VARIANTS, modes=v.MODES)
        rec = {"unit": name, "cls": cls, "a": a if direction == "AB" else b, "b": b if direction == "AB" else a,
               "pair": f"{short(a)}__{short(b)}", "draw": d, "direction": direction,
               "a_info": _D[a][1], "b_info": _D[b][1], "cells": res["cells"], "diag": res["diag"],
               "total_s": time.perf_counter() - t0}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(S.jsonable(rec), f)
        os.replace(tmp, path)
        return name, "ok", time.perf_counter() - t0
    except Exception:
        with open(os.path.join(out_dir, name + ".failed"), "w") as f:
            f.write(traceback.format_exc())
        return name, "FAILED", time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    for ext in ("RUN.done", "RUN.failed"):
        if os.path.exists(os.path.join(out, ext)):
            os.remove(os.path.join(out, ext))
    rows, ids = S.ssel_items()
    stems = sorted({s for prs in S.SSEL_PAIRS.values() for pr in prs for s in pr})
    t0 = time.perf_counter()
    cache_path = os.path.join(S.PROJECT, "results", "S1_2", "cache", "clip_thresholds.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    thr_check = {}
    for s in stems:
        D, info = S.ssel_model_D(s, rows)
        _D[s] = (D, info)
        ref = cache.get(f"{s}|wit4096_sel")
        thr_check[s] = {"ours": info["clip_threshold"], "s12_cache": ref,
                        "rel_diff": (abs(info["clip_threshold"] - ref) / ref) if ref else None,
                        "layer_index": info["layer_index"], "n_stored": info["n_stored"]}
        print(f"D {s}: layer {info['layer_index']}/{info['n_stored']} thr {info['clip_threshold']:.6f} "
              f"(S1.2 cache {ref})", flush=True)
    tasks = []
    for cls, prs in S.SSEL_PAIRS.items():
        for a, b in prs:
            for d in range(S.SSEL_DRAWS):
                for direction in ("AB", "BA"):
                    tasks.append((cls, a, b, d, direction, out))
    meta = {"n_units": len(tasks), "pairs": S.SSEL_PAIRS, "m_grid": S.SSEL_M_GRID, "m_fit": S.SSEL_M_FIT,
            "ks": S.SSEL_KS, "draw_salts": S.SSEL_DRAW_SALTS, "pre": S.SSEL_PRE, "rel_depth": S.SSEL_REL_DEPTH,
            "dropped_rows": S.SSEL_DROP_ROWS, "n_rows": int(rows.size), "clip_threshold_check": thr_check,
            "items_sha256": S.file_sha256(os.path.join(S.SSEL_DIR, "items.parquet")),
            "predict_v11_sha256": S.file_sha256(os.path.join(S.PROJECT, "hrhv2", "predict_v11.py")),
            "sel_common_sha256": S.file_sha256(os.path.join(HERE, "sel_common.py")),
            "run_ssel_sha256": S.file_sha256(os.path.abspath(__file__)),
            "D_build_s": time.perf_counter() - t0,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out, "RUN_meta.json"), "w") as f:
        json.dump(S.jsonable(meta), f, indent=1)
    n_fail = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(args.jobs) as pool:
        for name, status, sec in pool.imap_unordered(work, tasks):
            n_fail += status == "FAILED"
            print(f"[{time.strftime('%H:%M:%S')}] {name} {status} {sec:.0f}s", flush=True)
    meta["wall_s"] = time.perf_counter() - t0
    meta["n_failed"] = n_fail
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(out, "RUN_meta.json"), "w") as f:
        json.dump(S.jsonable(meta), f, indent=1)
    with open(os.path.join(out, "RUN.failed" if n_fail else "RUN.done"), "w") as f:
        f.write(meta["finished_utc"] + "\n")
    print("done", meta["wall_s"], "failed", n_fail, flush=True)


if __name__ == "__main__":
    main()
