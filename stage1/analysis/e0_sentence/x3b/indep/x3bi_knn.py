"""X3b independent check: single-model neighbour tables from my own D.

For every model (51) and every draw (5) and prefix m: all H2 items of the
prefix are anchors; the candidates are all other items of the prefix; full
stable argsort of the anchor's D row in draw order, i.e. the SPEC key
(D, position in the draw order); the first 101 neighbours (positions within
the prefix) and the sorted D at columns 0,1,9,10,99,100 are kept, plus the
median off-diagonal D of the prefix (hrhv2.prep.median_offdiag).

Usage: python x3bi_knn.py <outdir> <n_workers>
EXPLORATORY.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.expanduser("~/hrh_v2_stage1_20261008"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import x3bi_common as C  # noqa: E402

CH = 512


def knn_one_draw(D, order, h1, med_full):
    out, med = {}, {}
    for m in C.M_GRID:
        P = order[:m]
        h2pos = np.flatnonzero(~h1[P]).astype(np.int32)
        n = len(h2pos)
        idx = np.empty((n, C.NCOL), dtype=np.int32)
        vals = np.empty((n, len(C.VAL_COLS)), dtype=np.float64)
        rr = np.arange(CH)
        for c0 in range(0, n, CH):
            c1 = min(n, c0 + CH)
            R = D[np.ix_(P[h2pos[c0:c1]], P)]
            R[rr[: c1 - c0], h2pos[c0:c1]] = np.inf          # exclude the anchor itself
            o = np.argsort(R, axis=1, kind="stable")[:, : C.NCOL]  # ties -> lower position
            idx[c0:c1] = o
            vals[c0:c1] = np.take_along_axis(R, o[:, list(C.VAL_COLS)], axis=1)
        if m == len(order):
            med[m] = med_full
        else:
            from hrhv2.prep import median_offdiag
            med[m] = median_offdiag(D[np.ix_(P, P)])
        out[f"pos_m{m}"] = h2pos
        out[f"idx_m{m}"] = idx
        out[f"vals_m{m}"] = vals
    return out, med


def build(spec, outdir, orders, h1):
    from hrhv2.prep import median_offdiag
    key = C.model_key(spec)
    mdir = os.path.join(outdir, "knn", key.replace("/", "__"))
    os.makedirs(mdir, exist_ok=False)
    t0 = time.time()
    D, info = C.model_D(spec)
    tD = time.time() - t0
    nz = int((D == 0).sum()) - D.shape[0]
    med_full = median_offdiag(D)
    meta = {"key": key, "spec": spec, "info": info, "t_build_D_s": tD,
            "n_zero_offdiag_ordered": nz, "draws": {}}
    for d in range(C.N_DRAWS):
        t1 = time.time()
        arrs, med = knn_one_draw(D, orders[d], h1, med_full)
        f = os.path.join(mdir, f"d{d}_knn.npz")
        np.savez(f, **arrs)
        meta["draws"][str(d)] = {"median_D": {str(m): float(v) for m, v in med.items()},
                                 "n_h2": {str(m): int(len(arrs[f"pos_m{m}"])) for m in C.M_GRID},
                                 "sha256": C.sha256_file(f), "seconds": time.time() - t1}
    meta["seconds_total"] = time.time() - t0
    with open(os.path.join(mdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    for fn in os.listdir(mdir):
        os.chmod(os.path.join(mdir, fn), 0o444)
    return key, meta["seconds_total"]


def _worker(args):
    spec, outdir = args
    orders, h1 = _DESIGN
    return build(spec, outdir, orders, h1)


_DESIGN = None


def main():
    global _DESIGN
    outdir, nw = sys.argv[1], int(sys.argv[2])
    orders, h1, ids = C.build_design()
    # design check against S1.1 design.npz (bitwise)
    z = np.load(os.path.join(C.ROOT, "results/S1_1/design/design.npz"))
    dchk = {"orders_equal": bool(np.array_equal(z["orders"], orders)),
            "h1_equal": bool(np.array_equal(z["h1"], h1)),
            "image_id_equal": bool(np.array_equal(z["image_id"], ids))}
    with open(os.path.join(outdir, "design_check.json"), "w") as fh:
        json.dump(dchk, fh, indent=1)
    print("design check", dchk, flush=True)
    if not all(dchk.values()):
        raise SystemExit("design mismatch: STOP")
    _DESIGN = (orders, h1)
    specs = C.all_model_specs()
    # big-d models first
    def cost(s):
        return {"openllama_3b": 0, "bloomz_1b7": 1, "pythia410m_s1": 2, "qwen3emb06": 3}.get(s["name"], 5)
    specs.sort(key=cost)
    t0 = time.time()
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=nw, maxtasksperchild=1) as pool:
        for key, sec in pool.imap_unordered(_worker, [(s, outdir) for s in specs]):
            print(f"{time.strftime('%H:%M:%S')} done {key} {sec:.1f}s", flush=True)
    tm = os.times()
    rec = {"wall_s": time.time() - t0, "children_user_s": tm.children_user,
           "children_sys_s": tm.children_system, "self_user_s": tm.user, "self_sys_s": tm.system,
           "n_models": len(specs)}
    with open(os.path.join(outdir, "knn_compute.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print("KNN DONE", rec, flush=True)


if __name__ == "__main__":
    main()
