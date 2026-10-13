"""S1.1 v1.1, phases B/C: one unit = (configuration, pair, draw, direction, kind).

kind 'nulls' : hrhv2.predict_v11.run_unit with variants=() and modes=('d2','mu')
               (observed s_N on the predicted anchors, N0, N0H1, N1 [d2], N1|mu;
               D4.8 N1 := N0 when the affine slope <= 0), plus the exploratory
               v1.0 P2 and N2 (frozen hrhv2.predict: fit on H1 x H1 of the m_f
               prefix, P2 predicted on the same anchors; plan V3).
kind 'iii'   : run_unit with variants=('iii-LSlog-EB',) -- the crossing law as
               the UNVALIDATED predictor (plan V4; PREREG v1.1 D4.7). run_unit always
               computes obs, N0, N0H1 and N1 as well, so an 'iii' unit is a superset of a
               'nulls' unit (P2/N2 are added too); 'nulls' units are only run for pairs
               and configurations without 'iii' units (compute, see STATUS).

Inputs come from the phase-A cache (s11v11_prep.py): the predicted anchors' D rows
over the draw order and the m_f block, which are exactly the arrays run_ssel.py
passes (tested in test_s11v11.py). Direction 'AB' predicts with A = the pair's
first model (s11_common.pair_list convention); 'BA' swaps the roles.

    setsid nohup venv/bin/python analysis/s11/s11v11_units.py --kind nulls \
        --configs clip,l2,cliplayer,dedup --workers 12 > results/S1_1/v11/logs/units_nulls.log 2>&1 &
"""
from __future__ import annotations

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
import s11_common as C  # noqa: E402
import s11v11_common as V  # noqa: E402
from hrhv2 import predict as p10  # noqa: E402
from hrhv2 import predict_v11 as v11  # noqa: E402

KINDS = ("nulls", "iii")
COST_RANK = {"cross-modal": 0, "cross-vision": 1, "seed-seed": 2}


class RowsAsSquare:
    """Read-only stand-in for an m x m distance matrix of which only the rows of
    the predicted anchors are available: ``obj[i]`` is the D row of the anchor at
    position i over positions 0..m-1, ``obj.shape == (m, m)``. This is all that
    hrhv2.predict.anchor_rows (hence predict_p2) reads (tested)."""

    def __init__(self, rows: np.ndarray, pos: np.ndarray, m: int):
        self._rows = rows
        self._ix = {int(p): r for r, p in enumerate(np.asarray(pos))}
        self._m = int(m)
        self.shape = (self._m, self._m)

    def __getitem__(self, i):
        return self._rows[self._ix[int(i)], :self._m]


def p2_n2(DX_rows, anchor_pos, DX_fit, DY_fit, h1_fit, m_grid, ks):
    """Exploratory v1.0 P2 and N2 (frozen hrhv2.predict) on the v1.1 anchors.
    Fit: H1 x H1 of the m_f prefix (as v1.0). P2 is predicted from the predicted
    anchors' X-rows at every m; N2 depends on m and k only through the fitted rho_i."""
    t0 = time.perf_counter()
    disp = p10.fit_iso_dispersion(DX_fit, DY_fit, h1_fit)
    rhos = p10.fit_n2(DX_fit, DY_fit, h1_fit)
    out = {}
    for m in m_grid:
        pos = anchor_pos[anchor_pos < m]
        kk = [int(k) for k in ks if k <= m - 2]
        if not kk or pos.size == 0:
            continue
        p2 = p10.predict_p2(RowsAsSquare(DX_rows, anchor_pos, m), pos, kk, disp, lb_bias=None)
        n2 = p10.predict_n2(rhos, m, kk)
        for k in kk:
            out[(int(m), k)] = {"P2": p2[k]["s_pred"], "P2_raw": p2[k]["s_pred_raw"], "P2_phat": p2[k]["p_hat"],
                                "P2_deltaD": p2[k]["delta_D_mean"], "N2": n2[k]["s_pred"]}
    diag = {"P2_disp": {"n_bins": int(disp.xc.size), "lo_slope": float(disp.lo_slope),
                        "hi_slope": float(disp.hi_slope)},
            "N2_rho_median": float(np.median(rhos)),
            "N2_rho_iqr": [float(np.quantile(rhos, 0.25)), float(np.quantile(rhos, 0.75))],
            "p2n2_s": time.perf_counter() - t0}
    return out, diag


def load_inputs(config, model, draw):
    cd = V.cache_dir(config, model)
    rows = np.load(os.path.join(cd, f"d{draw}_rows.npy"))
    fit = np.load(os.path.join(cd, f"d{draw}_fit.npy"))
    return rows, fit


_DES = {}


def design_cached(config):
    if config not in _DES:
        _DES[config] = V.design(config)
    return _DES[config]


def unit_path(config, kind, pair, draw, direction):
    return os.path.join(V.UNITS, config, kind, f"{pair}__d{draw}__{direction}.json")


def run_one(config, kind, P, dd, direction, rowsA, fitA, rowsB, fitB):
    """Compute one unit from in-memory arrays (also used by the tests)."""
    if direction == "AB":
        DXr, DYr, DXf, DYf = rowsA, rowsB, fitA, fitB
    elif direction == "BA":
        DXr, DYr, DXf, DYf = rowsB, rowsA, fitB, fitA
    else:
        raise ValueError(direction)
    h1_fit = dd["h1"][:V.M_FIT]
    variants = () if kind == "nulls" else (V.UNVALIDATED,)
    res = v11.run_unit(DXr, DYr, dd["anchor_pos"], DXf, DYf, h1_fit, m_grid=dd["m_grid"], m_f=V.M_FIT,
                       ks=V.KS, variants=variants, modes=v11.MODES)
    pn, extra_diag = p2_n2(DXr, dd["anchor_pos"], DXf, DYf, h1_fit, dd["m_grid"], V.KS)
    for c in res["cells"]:
        c.update(pn.get((int(c["m"]), int(c["k"])), {}))
    return res, extra_diag


def work(task):
    config, kind, pair, draw, direction = task
    path = unit_path(config, kind, pair, draw, direction)
    if os.path.exists(path):
        return task, "exists", 0.0
    t0 = time.perf_counter()
    try:
        P = C.PAIRS[pair]
        dd = design_cached(config)["draws"][draw]
        rowsA, fitA = load_inputs(config, P["a"], draw)
        rowsB, fitB = load_inputs(config, P["b"], draw)
        if rowsA.shape != (dd["anchor_pos"].size, dd["M"]) or rowsB.shape != rowsA.shape:
            raise RuntimeError("cache shape does not match the design")
        res, extra = run_one(config, kind, P, dd, direction, rowsA, fitA, rowsB, fitB)
        rec = {"config": config, "kind": kind, "pair": pair, "cls": P["cls"], "primary": P["primary"],
               "a": P["a"], "b": P["b"], "draw": draw, "direction": direction,
               "x_model": P["a"] if direction == "AB" else P["b"],
               "m_grid": list(dd["m_grid"]), "cells": res["cells"], "diag": {**res["diag"], **extra},
               "seconds": time.perf_counter() - t0, "pid": os.getpid(),
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(V.jsonable(rec), f)
        os.replace(tmp, path)
        return task, "ok", time.perf_counter() - t0
    except Exception:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path[:-5] + ".failed", "w") as f:
            f.write(traceback.format_exc())
        return task, "FAILED", time.perf_counter() - t0


def task_list(kind, configs, which):
    tasks = []
    for config in configs:
        for P in V.pairs_for(config, which):
            for d in range(V.N_DRAWS):
                for direction in ("AB", "BA"):
                    tasks.append((config, kind, P["pair"], d, direction))
    # expensive classes first inside each configuration (shorter tail); configuration order kept
    corder = {c: i for i, c in enumerate(configs)}
    tasks.sort(key=lambda t: (corder[t[0]], COST_RANK[C.PAIRS[t[2]]["cls"]], t[2], t[3], t[4]))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--configs", required=True, help="comma list, run in this order")
    ap.add_argument("--which", default="all", choices=("all", "primary", "secondary"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.workers > 12:
        raise SystemExit("role limit: <= 12 cores")
    configs = args.configs.split(",")
    for c in configs:
        V.check_authorization(c)
        missing = [m for m in V.models_for(c) if not os.path.exists(os.path.join(V.cache_dir(c, m), "DONE"))]
        if missing:
            raise SystemExit(f"cache missing for {c}: {missing}")
    tag = args.tag or f"{args.kind}_{'-'.join(configs)}_{args.which}"
    marker = os.path.join(V.OUT, f"UNITS_{tag}")
    for ext in (".done", ".failed"):
        if os.path.exists(marker + ext):
            os.remove(marker + ext)
    tasks = task_list(args.kind, configs, args.which)
    todo = [t for t in tasks if not os.path.exists(unit_path(t[0], t[1], t[2], t[3], t[4]))]
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {tag}: {len(tasks)} units, {len(todo)} to run, "
          f"{args.workers} workers x 1 thread", flush=True)
    n_fail, n_done = 0, 0
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=20) as pool:
        for task, status, sec in pool.imap_unordered(work, todo, chunksize=1):
            n_fail += status == "FAILED"
            n_done += 1
            print(f"[{time.strftime('%H:%M:%S')}] {n_done}/{len(todo)} {'/'.join(map(str, task))} {status} {sec:.0f}s",
                  flush=True)
    with open(marker + (".failed" if n_fail else ".done"), "w") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} n={len(todo)} n_failed={n_fail} "
                f"wall={time.perf_counter() - t0:.0f}s\n")
    print("finished", tag, "failed:", n_fail, flush=True)


if __name__ == "__main__":
    main()
