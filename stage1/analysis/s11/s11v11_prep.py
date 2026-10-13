"""S1.1 v1.1, phase A: per (model, D source) build the 16384 x 16384 float64 D once
(hrhv2.prep via s11_common.model_D) and cache, for every configuration and draw
that uses it:

  d<draw>_rows.npy   (n_a, M)  D rows of the predicted anchors (H2 items of the
                               draw's m_f = 1024 prefix) to pool positions 0..M-1
                               in the draw's order (plan V2)
  d<draw>_fit.npy    (1024, 1024) the m_f prefix block (N1, N0H1, P2/N2 fits)
  d<draw>_knn.npz    for every m of the grid: first 101 neighbours (positions in
                     the m prefix, SPEC tie rule (D_ij, position)) of ALL H2 items
                     of the prefix (obs_allH2, plan V2) and their D values at
                     columns k-1, k for k in {1, 10, 100} (O11 tie bookkeeping)
  meta.json          clamp threshold, layer, near-tie medians median(D) per prefix,
                     exact-zero counts, timings, npy sha256 check

    setsid nohup venv/bin/python analysis/s11/s11v11_prep.py --workers 3 --threads 4 \
        > results/S1_1/v11/logs/prep.log 2>&1 &

Refuses to run without the v1.1 authorization (s11v11_common.check_authorization).
"""
from __future__ import annotations

import argparse
import os
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--workers", type=int, default=3)
_ap.add_argument("--threads", type=int, default=4)
_ap.add_argument("--configs", default="clip,l2,cliplayer,dedup")
_ap.add_argument("--models", default=None)
_ARGS = _ap.parse_args() if __name__ == "__main__" else None
if _ARGS is not None:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = str(_ARGS.threads)

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s11_common as C  # noqa: E402
import s11v11_common as V  # noqa: E402


def knn_prefix_rows(D: np.ndarray, order: np.ndarray, pos_rows: np.ndarray, m: int, kmax: int,
                    chunk: int = 256):
    """First kmax neighbours of the items at positions ``pos_rows`` (< m) of the
    draw order, among positions 0..m-1, key (D, position), self excluded.
    Same algorithm as s11_core.knn_rows / hrhv2.structural.knn_table, applied to
    D[order[:m]][:, order[:m]] without forming that matrix.
    Returns (idx int64 positions, vals float64), shape (len(pos_rows), min(kmax, m-1))."""
    order = np.asarray(order, dtype=np.int64)
    pos_rows = np.asarray(pos_rows, dtype=np.int64)
    cols = order[:m]
    K = int(min(kmax, m - 1))
    idx_out = np.empty((pos_rows.size, K), dtype=np.int64)
    val_out = np.empty((pos_rows.size, K), dtype=np.float64)
    for lo in range(0, pos_rows.size, chunk):
        hi = min(pos_rows.size, lo + chunk)
        n = hi - lo
        r = pos_rows[lo:hi]
        Dc = D[np.ix_(order[r], cols)].astype(np.float64, copy=True)
        Dc[np.arange(n), r] = np.inf
        v = np.partition(Dc, K - 1, axis=1)[:, K - 1]
        lt = Dc < v[:, None]
        eq = Dc == v[:, None]
        need = K - lt.sum(axis=1)
        sel = lt | (eq & (np.cumsum(eq, axis=1) <= need[:, None]))
        cc = np.nonzero(sel)[1].reshape(n, K)
        vals = np.take_along_axis(Dc, cc, axis=1)
        o = np.lexsort((cc, vals), axis=-1)
        idx_out[lo:hi] = np.take_along_axis(cc, o, axis=1)
        val_out[lo:hi] = np.take_along_axis(vals, o, axis=1)
    return idx_out, val_out


def median_offdiag_items(D: np.ndarray, items: np.ndarray) -> float:
    """Median of the off-diagonal entries of D restricted to ``items``
    (D >= 0 with a zero diagonal: the m diagonal zeros are the m smallest
    order statistics, so the off-diagonal order statistic j is the overall
    order statistic j + m)."""
    items = np.asarray(items, dtype=np.int64)
    m = items.size
    if m == D.shape[0] and np.array_equal(items, np.arange(m)):
        flat = np.array(D, dtype=np.float64, copy=True).ravel()
    else:
        flat = D[np.ix_(items, items)].ravel()
    n_off = m * (m - 1)
    i1, i2 = m + n_off // 2 - 1, m + n_off // 2
    part = np.partition(flat, (i1, i2))
    return float(0.5 * (part[i1] + part[i2]))


def unit_arrays(D: np.ndarray, order: np.ndarray, anchor_pos: np.ndarray, m_f: int):
    """(rows, fit): the predicted anchors' rows over the whole draw order and the
    m_f prefix block -- equal to DAo[anchor_pos], DAo[:m_f, :m_f] with
    DAo = D[order][:, order] (run_ssel.py convention)."""
    order = np.asarray(order, dtype=np.int64)
    rows = np.ascontiguousarray(D[np.ix_(order[anchor_pos], order)])
    fit = np.ascontiguousarray(D[np.ix_(order[:m_f], order[:m_f])])
    return rows, fit


def dsrc_args(dsrc: str):
    return {"clip-stored": ("prh_clip+l2", "pool-stored"), "clip-layer": ("prh_clip+l2", "pool-layer"),
            "l2": ("l2", None)}[dsrc]


def work(task):
    dsrc, model, configs = task
    t0 = time.perf_counter()
    tag = f"{dsrc}/{model}"
    try:
        todo = [c for c in configs if not os.path.exists(os.path.join(V.cache_dir(c, model), "DONE"))]
        if not todo:
            return tag, "exists", 0.0
        V.check_authorization()
        pre, clip_mode = dsrc_args(dsrc)
        slug = C.MODELS[model]
        meta_sc = C.sidecar(slug)
        npy = os.path.join(C.COCO_DIR, slug + ".npy")
        npy_sha = V.file_sha256(npy)
        if npy_sha != meta_sc["files_sha256"]["primary"]["sha256"]:
            raise RuntimeError(f"{npy} sha256 mismatch with sidecar")
        D, info = C.model_D(model, pre, clip_mode)
        info["npy_sha256_checked"] = npy_sha
        t_build = time.perf_counter() - t0
        n_zero_offdiag_full = int((D == 0).sum() - D.shape[0])
        med_full = median_offdiag_items(D, np.arange(D.shape[0]))
        for config in todo:
            out = V.cache_dir(config, model)
            os.makedirs(out, exist_ok=True)
            des = V.design(config)
            cm = {"config": config, "model": model, "dsrc": dsrc, "info": info, "t_build_D_s": t_build,
                  "n_zero_offdiag_pool": n_zero_offdiag_full, "draws": {}}
            med_cache = {}
            for dd in des["draws"]:
                d, o, h1o, anc, M = dd["draw"], dd["order"], dd["h1"], dd["anchor_pos"], dd["M"]
                td = time.perf_counter()
                rows, fit = unit_arrays(D, o, anc, V.M_FIT)
                np.save(os.path.join(out, f"d{d}_rows.npy"), rows)
                np.save(os.path.join(out, f"d{d}_fit.npy"), fit)
                tabs = {}
                dm = {"M": M, "n_anchors": int(anc.size), "median_D": {}, "n_h2": {}, "n_zero_nn1_h2": {}}
                for m in dd["m_grid"]:
                    h2pos = np.flatnonzero(~h1o[:m])
                    idx, vals = knn_prefix_rows(D, o, h2pos, m, V.KMAX_TAB)
                    tabs[f"pos_m{m}"] = h2pos.astype(np.int32)
                    tabs[f"idx_m{m}"] = idx.astype(np.int32)
                    tabs[f"vals_m{m}"] = vals[:, V.VAL_COLS]
                    if m == D.shape[0]:
                        med = med_full                            # whole pool (every draw)
                    elif m == M:
                        if "full_set" not in med_cache:           # same item set in every draw (dedup)
                            med_cache["full_set"] = median_offdiag_items(D, o[:m])
                        med = med_cache["full_set"]
                    else:
                        med = median_offdiag_items(D, o[:m])
                    dm["median_D"][m] = med
                    dm["n_h2"][m] = int(h2pos.size)
                    dm["n_zero_nn1_h2"][m] = int((vals[:, 0] == 0).sum())
                tabs["val_cols"] = np.asarray(V.VAL_COLS, dtype=np.int64)
                np.savez(os.path.join(out, f"d{d}_knn.npz"), **tabs)
                dm["seconds"] = time.perf_counter() - td
                dm["rows_sha256"] = V.file_sha256(os.path.join(out, f"d{d}_rows.npy"))
                cm["draws"][d] = dm
            cm["design"] = {"n_dropped": int(des["dropped_rows"].size), "dropped_rows": des["dropped_rows"]}
            cm["seconds_total"] = time.perf_counter() - t0
            cm["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(os.path.join(out, "meta.json"), "w") as f:
                json.dump(V.jsonable(cm), f, indent=1)
            with open(os.path.join(out, "DONE"), "w") as f:
                f.write(cm["finished_utc"] + "\n")
        del D
        return tag, "ok", time.perf_counter() - t0
    except Exception:
        os.makedirs(V.LOGS, exist_ok=True)
        with open(os.path.join(V.LOGS, f"prep_{dsrc}_{model}.failed"), "w") as f:
            f.write(traceback.format_exc())
        return tag, "FAILED", time.perf_counter() - t0


def tasks_for(configs, models=None):
    by = {}
    for c in configs:
        ds = V.CONFIGS[c]["dsrc"]
        for mdl in V.models_for(c):
            if models and mdl not in models:
                continue
            by.setdefault((ds, mdl), []).append(c)
    size = {m: C.sidecar(C.MODELS[m])["shape"][2] for _, m in by}
    return sorted(((ds, m, cs) for (ds, m), cs in by.items()), key=lambda t: (-size[t[1]], t[0], t[1]))


def main():
    args = _ARGS
    V.check_authorization()
    os.makedirs(V.LOGS, exist_ok=True)
    marker = os.path.join(V.OUT, "PREP")
    for ext in (".done", ".failed"):
        if os.path.exists(marker + ext):
            os.remove(marker + ext)
    configs = args.configs.split(",")
    tasks = tasks_for(configs, args.models.split(",") if args.models else None)
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} prep: {len(tasks)} D builds, "
          f"{args.workers} workers x {args.threads} threads", flush=True)
    n_fail = 0
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=1) as pool:
        for tag, status, sec in pool.imap_unordered(work, tasks):
            n_fail += status == "FAILED"
            print(f"[{time.strftime('%H:%M:%S')}] {tag} {status} {sec:.0f}s", flush=True)
    with open(marker + (".failed" if n_fail else ".done"), "w") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} n_failed={n_fail} wall={time.perf_counter() - t0:.0f}s\n")
    print("prep finished, failed:", n_fail, flush=True)


if __name__ == "__main__":
    main()
