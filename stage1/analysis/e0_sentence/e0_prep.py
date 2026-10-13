"""E0 single-model kNN caches (ANALYSIS_PLAN_E0.md v1.0 §5). CPU only; no cross-model quantity.

Per (pre key, model): build the 16384 x 16384 float64 D once (e0_common.model_D) and write, for every draw of
the design, the S1.1 cache layout with the byte-identical S1.1 functions (s11copy/s11v11_prep.py):
  d<draw>_knn.npz  first 101 neighbours of all H2 items of each prefix + D values at columns k-1, k
  d<draw>_rows.npy, d<draw>_fit.npy  only with --save-rows (arm-P sentence models, for X1; and V-b)
  meta.json, DONE

    venv/bin/python -I analysis/e0_sentence/e0_prep.py --workers 2 --threads 3            # all 51 new caches
    venv/bin/python -I analysis/e0_sentence/e0_prep.py --only ctr:dinov2_b --workers 1 --threads 6

Refuses unless STATUS shows PLAN FROZEN and E0 CODE FROZEN (and EXTRACTION DONE for text models), with the
file hashes. Never overwrites: an existing DONE is skipped, a partial cache directory stops the run.
"""
from __future__ import annotations

import argparse
import os
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--workers", type=int, default=2)
_ap.add_argument("--threads", type=int, default=3)
_ap.add_argument("--only", default=None, help="comma list of pre:model")
_ap.add_argument("--set", default="new", choices=("new", "q1"))
_ARGS = _ap.parse_args() if __name__ == "__main__" else None
if _ARGS is not None:
    if _ARGS.workers < 1 or _ARGS.threads < 1 or _ARGS.workers * _ARGS.threads > 6:
        raise SystemExit("REFUSED: workers x threads exceeds the 6-thread task rule")
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = str(_ARGS.threads)

import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e0_common as E  # noqa: E402
import s11v11_prep as SP  # noqa: E402  (byte-identical copy in s11copy/)

ROWS_MODELS = tuple(f"{e}.c1" for e in E.ENCODERS)      # arm-P sentence models (plan §5)


def build_cache(pre_key: str, model: str, out: str, *, save_rows: bool, feat_dir: str = E.FEAT,
                expected_sha: dict | None = None, design_fn=None, D=None, info=None) -> dict:
    """Write one cache directory ``out`` (must not exist or be empty). Mirrors s11v11_prep.work for one
    configuration; the kNN, median and anchor-row functions are the byte-identical S1.1 ones."""
    t0 = time.perf_counter()
    if os.path.exists(os.path.join(out, "DONE")):
        raise FileExistsError(f"{out} already complete")
    if os.path.isdir(out) and os.listdir(out):
        raise FileExistsError(f"{out} is a partial cache; inspect it (E0 code deletes nothing)")
    os.makedirs(out, exist_ok=True)
    if D is None:
        D, info = E.model_D(pre_key, model, feat_dir, expected_sha)
    t_build = time.perf_counter() - t0
    des = (design_fn or E.design)(E.design_of_pre(pre_key))
    n_zero_offdiag_full = int((D == 0).sum() - D.shape[0])
    med_full = SP.median_offdiag_items(D, np.arange(D.shape[0]))
    cm = {"pre_key": pre_key, "model": model, "info": info, "t_build_D_s": t_build,
          "n_zero_offdiag_pool": n_zero_offdiag_full, "draws": {}, "plan": "ANALYSIS_PLAN_E0.md v1.0 §5"}
    med_cache = {}
    for dd in des["draws"]:
        d, o, h1o, anc, M = dd["draw"], dd["order"], dd["h1"], dd["anchor_pos"], dd["M"]
        td = time.perf_counter()
        dm = {"M": M, "n_anchors": int(anc.size), "median_D": {}, "n_h2": {}, "n_zero_nn1_h2": {}}
        if save_rows:
            rows, fit = SP.unit_arrays(D, o, anc, SP.V.M_FIT)
            E.save_npy_new(os.path.join(out, f"d{d}_rows.npy"), rows)
            E.save_npy_new(os.path.join(out, f"d{d}_fit.npy"), fit)
            del rows, fit
        tabs = {}
        for m in dd["m_grid"]:
            h2pos = np.flatnonzero(~h1o[:m])
            idx, vals = SP.knn_prefix_rows(D, o, h2pos, m, SP.V.KMAX_TAB)
            tabs[f"pos_m{m}"] = h2pos.astype(np.int32)
            tabs[f"idx_m{m}"] = idx.astype(np.int32)
            tabs[f"vals_m{m}"] = vals[:, SP.V.VAL_COLS]
            if m == D.shape[0]:
                med = med_full
            elif m == M:
                if "full_set" not in med_cache:
                    med_cache["full_set"] = SP.median_offdiag_items(D, o[:m])
                med = med_cache["full_set"]
            else:
                med = SP.median_offdiag_items(D, o[:m])
            dm["median_D"][m] = med
            dm["n_h2"][m] = int(h2pos.size)
            dm["n_zero_nn1_h2"][m] = int((vals[:, 0] == 0).sum())
        tabs["val_cols"] = np.asarray(SP.V.VAL_COLS, dtype=np.int64)
        npz = os.path.join(out, f"d{d}_knn.npz")
        if os.path.exists(npz):
            raise FileExistsError(npz)
        tmp = os.path.join(out, f"d{d}_knn.tmp.npz")
        np.savez(tmp, **tabs)
        os.replace(tmp, npz)
        dm["seconds"] = time.perf_counter() - td
        dm["knn_sha256"] = E.sha256_file(npz)
        if save_rows:
            dm["rows_sha256"] = E.sha256_file(os.path.join(out, f"d{d}_rows.npy"))
        cm["draws"][d] = dm
    cm["design"] = {"kind": E.design_of_pre(pre_key), "n_dropped": int(des["dropped_rows"].size)}
    cm["seconds_total"] = time.perf_counter() - t0
    cm["finished_utc"] = E.utcnow()
    E.write_json_new(os.path.join(out, "meta.json"), cm)
    E.write_new(os.path.join(out, "DONE"), cm["finished_utc"] + "\n")
    return cm


def work(task):
    pre_key, model = task
    out = E.cache_dir(pre_key, model)
    tag = f"{pre_key}/{model}"
    t0 = time.perf_counter()
    try:
        if os.path.exists(os.path.join(out, "DONE")):
            return tag, "exists", 0.0
        exp = E.extraction_hashes() if model not in E.C.MODELS else None
        build_cache(pre_key, model, out, save_rows=(pre_key == "clip" and model in ROWS_MODELS), expected_sha=exp)
        return tag, "ok", time.perf_counter() - t0
    except Exception:
        os.makedirs(os.path.join(E.OUT, "logs"), exist_ok=True)
        p = os.path.join(E.OUT, "logs", f"prep_{pre_key}_{model}_{E.utcnow()}.failed")
        with open(p, "w") as f:
            f.write(traceback.format_exc())
        return tag, "FAILED", time.perf_counter() - t0


def main():
    args = _ARGS
    tasks = E.new_cache_list() if args.set == "new" else E.q1_cache_list()
    if args.only:
        want = [tuple(x.split(":", 1)) for x in args.only.split(",")]
        bad = [w for w in want if w not in tasks]
        if bad:
            raise SystemExit(f"not in the plan's cache list: {bad}")
        tasks = want
    need = ["plan", "code"]
    if any(m not in E.C.MODELS for _, m in tasks):
        need.append("extract")
    E.require_milestones(*need)
    if args.set == "q1" and not E.approved("E0 Q1 APPROVED"):
        raise SystemExit("REFUSED: conditional arm Q1 needs a STATUS line 'E0 Q1 APPROVED' (plan §15)")
    os.makedirs(os.path.join(E.OUT, "logs"), exist_ok=True)
    print(f"{E.utcnow()} e0_prep: {len(tasks)} caches, {args.workers} workers x {args.threads} threads", flush=True)
    n_fail = 0
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=1) as pool:
        for tag, status, sec in pool.imap_unordered(work, tasks):
            n_fail += status == "FAILED"
            print(f"[{time.strftime('%H:%M:%S')}] {tag} {status} {sec:.0f}s", flush=True)
    print("e0_prep finished, failed:", n_fail, flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
