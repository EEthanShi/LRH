"""s11-select: plant runs of the v1.1 anchor-local protocol (direction A -> B).

    python run_plants.py --prefix S11sel --reps 4 --jobs 12 --out results/S1_1_select/plants

One JSON per (plant, p, c, rep) unit; <out>/RUN.done or RUN.failed at the end.
Single-threaded BLAS per worker (thread env vars set before numpy import).
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sel_common as S  # noqa: E402
from hrhv2 import predict_v11 as v  # noqa: E402

EXACT_DIAG = {"ks": (10,), "m": (256, 512, 2048), "variants": ("i-LSlog", "iii-LSlog-EB", "ii-IsoXF", "oracle")}


def work(task):
    plant, p, c, rep, prefix, out_dir, m_grid = task
    name = S.unit_name(plant, p, c, rep)
    path = os.path.join(out_dir, name + ".json")
    if os.path.exists(path):
        return name, "exists", 0.0
    t0 = time.perf_counter()
    try:
        u = S.plant_unit(plant, p, rep, prefix, c=c, M=max(m_grid))
        t1 = time.perf_counter()
        res = v.run_unit(u["DA_rows"], u["DB_rows"], u["anchor_pos"], u["DA_fit"], u["DB_fit"], u["h1_fit"],
                         m_grid=m_grid, m_f=S.PLANT_M_FIT, ks=S.PLANT_KS, variants=v.VARIANTS, modes=v.MODES,
                         oracle_sigma=u["oracle_sigma"], exact_diag=EXACT_DIAG if rep == 0 else None)
        rec = {"unit": name, "info": u["info"], "cells": res["cells"], "diag": res["diag"],
               "gen_s": t1 - t0, "total_s": time.perf_counter() - t0}
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
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma list of unit names (dev/smoke)")
    ap.add_argument("--m-grid", default=",".join(map(str, S.PLANT_M_GRID)))
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    for ext in ("RUN.done", "RUN.failed"):
        if os.path.exists(os.path.join(out, ext)):
            os.remove(os.path.join(out, ext))
    m_grid = tuple(int(x) for x in args.m_grid.split(","))
    units = S.plant_units(args.reps)
    if args.only:
        keep = set(args.only.split(","))
        units = [u for u in units if S.unit_name(*u) in keep]
    # costly units first (saturated / small p)
    cost = {"P-GaussSat": 0, "P-Rough": 1, "P-Gauss": 2, "P-Smooth+noise": 3, "P-Conf": 4, "P-Aniso": 5}
    units.sort(key=lambda t: (t[1], cost[t[0]]))
    tasks = [(pl, p, c, r, args.prefix, out, m_grid) for pl, p, c, r in units]
    meta = {"prefix": args.prefix, "reps": args.reps, "n_units": len(tasks), "jobs": args.jobs, "m_grid": m_grid,
            "predict_v11_sha256": S.file_sha256(os.path.join(S.PROJECT, "hrhv2", "predict_v11.py")),
            "sel_common_sha256": S.file_sha256(os.path.join(HERE, "sel_common.py")),
            "run_plants_sha256": S.file_sha256(os.path.abspath(__file__)),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out, "RUN_meta.json"), "w") as f:
        json.dump(S.jsonable(meta), f, indent=1)
    t0 = time.perf_counter()
    n_fail = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(args.jobs, maxtasksperchild=1) as pool:
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
