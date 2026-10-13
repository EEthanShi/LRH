"""Precompute + validation V1 (plan section 4).

LLAMA-AXIS COPY (implementation E, role llama-axis-run-early): copied from
analysis/s12_layermax/precompute.py (sha256 e8f2cdc3...). Changes, marked
"LLAMA-AXIS": feature loader (lm_core.load_stack); V-L1 also compares with the
frozen Part B thresholds and writes/compares the shared llama-axis cache
results/S1_2_layermax_llama/cache/clip_thresholds.json; V-L2 also compares the
final tables of the 18 previously processed models (llama-7b + 17 ViTs) with
the frozen caches results/S1_2_layermax/cache/knn_final/ (must be identical).

Phase 1 (GPU): clip thresholds (hrhv2, all-layer) checked against the S1.2
cache; GPU float64 kNN tables (11 columns, key (D_ij, j)) for every stored
layer of all 25 models and both preprocessings.

Phase 2 (CPU, 5 worker processes x 1 BLAS thread): hrhv2 reference tables
(hrhv2.prep.gram_to_dist + hrhv2.structural.knn_table on the same float64
rows), comparison, exact boundary ties and SPEC near-tie fraction at k=10.
Final tables = GPU table, or the hrhv2 table where they differ (rule fixed in
the plan).

Outputs: results/S1_2_layermax/cache/{clip.json, knn_gpu/, knn_hrhv2/, knn_final/}
         results/S1_2_layermax/validation/V1_knn_tables.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402

KMAX = LC.KNN_K + 1


def _models():
    return [(s, "A") for s, _ in LC.LMS] + [(s, "B") for s, _ in LC.VITS]


def phase1():
    import torch
    LC.setup_torch(threads=1)
    os.makedirs(os.path.join(LC.CACHE, "knn_gpu"), exist_ok=True)
    clip = {}
    t0 = time.time()
    for stem, _ in _models():
        c = LC.clip_c(stem)
        c12 = LC.s12_clip_c(stem)
        clip[stem] = {"c": c, "c_s12_cache": c12, "bitwise_equal": bool(c == c12)}
        for pre in LC.PRES:
            out = os.path.join(LC.CACHE, "knn_gpu", f"{stem}__{pre}.npz")
            if os.path.exists(out):
                continue
            L = LC.n_stored(stem)
            nn = np.empty((LC.M_ITEMS, L, KMAX), dtype=np.int64)
            nv = np.empty((LC.M_ITEMS, L, KMAX), dtype=np.float64)
            st = LC.load_stack(stem)   # LLAMA-AXIS: loader
            for l in range(L):
                X = LC.C12.preprocess_rows(np.asarray(st[:, l, :]), pre, c if pre == "prh_clip+l2" else None)
                g = LC.gpu_layer(X)
                nn[:, l, :] = g["nn"].cpu().numpy()
                nv[:, l, :] = g["nv"].cpu().numpy()
                del g
            np.savez(out + ".tmp.npz", nn=nn, nv=nv)
            os.replace(out + ".tmp.npz", out)
        print(f"[{time.time() - t0:6.0f}s] gpu tables {stem} (clip equal to S1.2 cache: {clip[stem]['bitwise_equal']})",
              flush=True)
        torch.cuda.empty_cache()
    with open(os.path.join(LC.CACHE, "clip.json"), "w") as f:
        json.dump(clip, f, indent=1)
    # LLAMA-AXIS: V-L1 shared cache (plan section 4): create it if absent, otherwise compare bitwise
    shared = os.path.join(LC.PROJ, "results", "S1_2_layermax_llama", "cache", "clip_thresholds.json")
    os.makedirs(os.path.dirname(shared), exist_ok=True)
    mine = {f"{s}|wit1024": clip[s]["c"] for s in clip}
    try:
        fd = os.open(shared, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as f:
            json.dump(mine, f, indent=1)
        clip["_shared_cache"] = {"action": "created_by_E", "path": shared}
    except FileExistsError:
        other = json.load(open(shared))
        clip["_shared_cache"] = {"action": "compared", "path": shared,
                                 "bitwise_equal_all": all(other.get(k) == v for k, v in mine.items()),
                                 "keys_missing": [k for k in mine if k not in other]}
    return clip


def v1_worker(args):
    stem, pre, c = args
    from hrhv2.prep import gram_to_dist, median_offdiag
    from hrhv2 import structural as st_
    gpu = np.load(os.path.join(LC.CACHE, "knn_gpu", f"{stem}__{pre}.npz"))
    gnn = gpu["nn"]
    L = gnn.shape[1]
    ref = np.empty_like(gnn)
    rows = []
    stck = LC.load_stack(stem)   # LLAMA-AXIS: loader
    for l in range(L):
        X = LC.C12.preprocess_rows(np.asarray(stck[:, l, :]), pre, c)
        D = gram_to_dist(X @ X.T)
        idx, vals = st_.knn_table(D, KMAX)
        ref[:, l, :] = idx
        k = LC.KNN_K
        ordered_eq = bool(np.array_equal(idx[:, :k], gnn[:, l, :k]))
        set_eq_rows = int(sum(set(idx[i, :k]) == set(gnn[i, l, :k]) for i in range(idx.shape[0])))
        rows.append({"layer": l, "ordered_identical": ordered_eq, "set_identical_rows": set_eq_rows,
                     "set_identical": set_eq_rows == idx.shape[0],
                     "exact_boundary_tie_rows": int(np.sum(vals[:, k] == vals[:, k - 1])),
                     "near_tie_frac_k10": float(st_.near_tie_fraction(vals, k, median_offdiag(D)))})
    os.makedirs(os.path.join(LC.CACHE, "knn_hrhv2"), exist_ok=True)
    np.save(os.path.join(LC.CACHE, "knn_hrhv2", f"{stem}__{pre}.npy"), ref)
    return stem, pre, rows


def phase2(clip):
    tasks = []
    for stem, _ in _models():
        for pre in LC.PRES:
            tasks.append((stem, pre, clip[stem]["c"] if pre == "prh_clip+l2" else None))
    tasks.sort(key=lambda t: -LC.n_stored(t[0]))   # LLAMA-AXIS: longest (llama-30b) first
    ctx = mp.get_context("spawn")
    res = {}
    t0 = time.time()
    with ctx.Pool(5) as pool:
        for stem, pre, rows in pool.imap_unordered(v1_worker, tasks):
            res[f"{stem}|{pre}"] = rows
            print(f"[{time.time() - t0:6.0f}s] V1 {stem} {pre}: identical {sum(r['set_identical'] for r in rows)}/{len(rows)}",
                  flush=True)
    os.makedirs(os.path.join(LC.CACHE, "knn_final"), exist_ok=True)
    summ = {"n_tables": 0, "n_ordered_identical": 0, "n_set_identical": 0, "replaced_by_hrhv2": [],
            "max_near_tie_frac_k10": {"A": 0.0, "B": 0.0}, "boundary_tie_layers": {"A": [], "B": []}}
    side = {s: sd for s, sd in _models()}
    for stem, _ in _models():
        for pre in LC.PRES:
            rows = res[f"{stem}|{pre}"]
            gnn = np.load(os.path.join(LC.CACHE, "knn_gpu", f"{stem}__{pre}.npz"))["nn"]
            ref = np.load(os.path.join(LC.CACHE, "knn_hrhv2", f"{stem}__{pre}.npy"))
            fin = gnn.copy()
            for r in rows:
                summ["n_tables"] += 1
                summ["n_ordered_identical"] += int(r["ordered_identical"])
                summ["n_set_identical"] += int(r["set_identical"])
                if not r["ordered_identical"]:
                    fin[:, r["layer"], :] = ref[:, r["layer"], :]
                    summ["replaced_by_hrhv2"].append(f"{stem}|{pre}|L{r['layer']}")
                sd = side[stem]
                summ["max_near_tie_frac_k10"][sd] = max(summ["max_near_tie_frac_k10"][sd], r["near_tie_frac_k10"])
                if r["exact_boundary_tie_rows"] > 0:
                    summ["boundary_tie_layers"][sd].append(
                        f"{stem}|{pre}|L{r['layer']}|rows={r['exact_boundary_tie_rows']}")
            np.save(os.path.join(LC.CACHE, "knn_final", f"{stem}__{pre}.npy"), fin)
    # LLAMA-AXIS: V-L2, final tables of previously processed models vs the frozen S1.2 layer-max cache
    frozen_cmp = {}
    for stem, _ in _models():
        for pre in LC.PRES:
            fz = os.path.join(LC.FROZEN_RES, "cache", "knn_final", f"{stem}__{pre}.npy")
            if not os.path.exists(fz):
                continue
            mine_t = np.load(os.path.join(LC.CACHE, "knn_final", f"{stem}__{pre}.npy"))
            frozen_cmp[f"{stem}|{pre}"] = bool(np.array_equal(mine_t, np.load(fz)))
    summ["frozen_cache_compared"] = len(frozen_cmp)
    summ["frozen_cache_identical"] = int(sum(frozen_cmp.values()))
    summ["frozen_cache_differs"] = [k for k, v in frozen_cmp.items() if not v]
    summ["clip_bitwise_equal_reference_all"] = all(v["bitwise_equal"] for k, v in clip.items() if not k.startswith("_"))
    out = {"summary": summ, "clip": clip, "per_table": res, "frozen_cache_cmp": frozen_cmp, "finished_utc": LC.utc()}
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    with open(os.path.join(LC.RES, "validation", "V1_knn_tables.json"), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    print("V1 summary", json.dumps(LC.jsonable(summ))[:2000], flush=True)


if __name__ == "__main__":
    clip = phase1()
    phase2(clip)
