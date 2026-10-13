"""Precompute + validation V-O1 / V-O2 (plan section 11; single-model only, NOT gated: plan section 4 F1(a)).

OPENLLAMA-AXIS COPY (implementation E, S0b sub-role E of oll-axis-extract): copied from
analysis/s12_layermax_llama/early/precompute.py (sha256 7106c4b5...). Changes, marked "OL-AXIS":
model list and paths from lm_core (3 OpenLLaMA LMs + 17 ViTs); V-O1 reference = the frozen Part B
threshold clip_thresholds_partB_ol.json for open_llama_13b and the S1.2 cache for 3B, 7B and the ViTs;
the shared single-model cache results/S1_2_layermax_openllama/cache/clip_thresholds.json is created if
absent and compared bitwise otherwise; V-O2 compares the final tables of 3B, 7B and the 17 ViTs with the
frozen results/S1_2_layermax/cache/knn_final/; the hrhv2 reference phase uses OL_THREADS worker processes
with 1 BLAS thread each (plan F5 thread cap); explicit pass flags; --models (debug subset).
S-E2 mode (OL_SENS_E2=1): only open_llama_3b/7b are processed (from the _e2_check features), outputs under
early/sens_e2/cache/ and early/sens_e2/validation/; the shared cache is not touched.

Run (step S6, after the Part B FROZEN line). Two launcher calls, so that only the GPU phase is charged to the
axis GPU ledger (plan section 11):
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_openllama/launch_ol.sh ol_E_precompute_gpu 1 --gpu 3 -- \
      venv/bin/python analysis/s12_layermax_openllama/early/precompute.py --phase gpu
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_openllama/launch_ol.sh ol_E_precompute_cpu 4 -- \
      venv/bin/python analysis/s12_layermax_openllama/early/precompute.py --phase cpu
  (--phase all runs both in one GPU process. S-E2 only: OL_SENS_E2=1 with the names ol_E_e2_precompute_gpu/_cpu.)

Phase 1 (GPU, 1 thread): clip thresholds (hrhv2, all-layer) checked against the reference; GPU float64 kNN
tables (11 columns, key (D_ij, j)) for every stored layer of all 20 models and both preprocessings.

Phase 2 (CPU, OL_THREADS worker processes x 1 BLAS thread): hrhv2 reference tables
(hrhv2.prep.gram_to_dist + hrhv2.structural.knn_table on the same float64
rows), comparison, exact boundary ties and SPEC near-tie fraction at k=10.
Final tables = GPU table, or the hrhv2 table where they differ (rule fixed in
the plan).

Outputs: <CACHE>/{clip.json, knn_gpu/, knn_hrhv2/, knn_final/}   (CACHE = results/S1_2_layermax_openllama/early/cache)
         results/S1_2_layermax_openllama/cache/clip_thresholds.json   (shared; created by the first of E/I)
         <RES>/validation/VO1_VO2_knn_tables.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402

KMAX = LC.KNN_K + 1
ONLY = None   # --models (debug subset), set in main()


def _models():
    out = [(s, "A") for s, _ in LC.LMS] + [(s, "B") for s, _ in LC.VITS]
    if LC.SENS_E2:   # S-E2: only the re-extracted models
        out = [(s, sd) for s, sd in out if s in LC.E2_AFFECTED]
    if ONLY:
        out = [(s, sd) for s, sd in out if s in ONLY]
    return out


def phase1():
    import torch
    LC.setup_torch(threads=1)
    torch.set_num_threads(1)
    os.makedirs(os.path.join(LC.CACHE, "knn_gpu"), exist_ok=True)
    clip = {}
    t0 = time.time()
    for stem, _ in _models():
        c = LC.clip_c(stem)
        c12 = LC.s12_clip_c(stem)
        clip[stem] = {"c": c, "c_reference": c12, "bitwise_equal": bool(c == c12),
                      "reference": "partB clip_thresholds_partB_ol.json" if stem in LC.NEW_LMS
                      else "S1.2 cache results/S1_2/cache/clip_thresholds.json"}
        for pre in LC.PRES:
            out = os.path.join(LC.CACHE, "knn_gpu", f"{stem}__{pre}.npz")
            if os.path.exists(out):
                continue
            L = LC.n_stored(stem)
            nn = np.empty((LC.M_ITEMS, L, KMAX), dtype=np.int64)
            nv = np.empty((LC.M_ITEMS, L, KMAX), dtype=np.float64)
            st = LC.load_stack(stem)
            for l in range(L):
                X = LC.C12.preprocess_rows(np.asarray(st[:, l, :]), pre, c if pre == "prh_clip+l2" else None)
                g = LC.gpu_layer(X)
                nn[:, l, :] = g["nn"].cpu().numpy()
                nv[:, l, :] = g["nv"].cpu().numpy()
                del g
            np.savez(out + ".tmp.npz", nn=nn, nv=nv)
            os.replace(out + ".tmp.npz", out)
        print(f"[{time.time() - t0:6.0f}s] gpu tables {stem} (clip bitwise equal to reference: {clip[stem]['bitwise_equal']})",
              flush=True)
        torch.cuda.empty_cache()
    old = {}
    if os.path.exists(os.path.join(LC.CACHE, "clip.json")):   # rerun / --models subsets: keep other models
        old = {k: v for k, v in json.load(open(os.path.join(LC.CACHE, "clip.json"))).items() if not k.startswith("_")}
    old.update(clip)
    clip = old
    # V-O1 shared cache (plan section 11): create it if absent, otherwise compare bitwise (not in S-E2 mode)
    if not LC.SENS_E2:
        shared = os.path.join(LC.SHARED_CACHE, "clip_thresholds.json")
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
                                     "bitwise_equal_all": all(other.get(k) == v for k, v in mine.items() if k in other),
                                     "keys_missing": [k for k in mine if k not in other]}
    with open(os.path.join(LC.CACHE, "clip.json"), "w") as f:
        json.dump(clip, f, indent=1)
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
    stck = LC.load_stack(stem)
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
    tasks.sort(key=lambda t: -LC.n_stored(t[0]))   # longest first
    nproc = LC.n_threads(2)
    saved = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                            "NUMEXPR_NUM_THREADS", "OL_THREADS")}
    for v in saved:   # workers: 1 BLAS thread each, nproc workers <= OL_THREADS (plan F5)
        os.environ[v] = "1"
    ctx = mp.get_context("spawn")
    res = {}
    t0 = time.time()
    with ctx.Pool(nproc) as pool:
        for stem, pre, rows in pool.imap_unordered(v1_worker, tasks):
            res[f"{stem}|{pre}"] = rows
            print(f"[{time.time() - t0:6.0f}s] V-O2 {stem} {pre}: identical {sum(r['set_identical'] for r in rows)}/{len(rows)}",
                  flush=True)
    for v, x in saved.items():
        if x is None:
            os.environ.pop(v, None)
        else:
            os.environ[v] = x
    os.makedirs(os.path.join(LC.CACHE, "knn_final"), exist_ok=True)
    summ = {"n_tables": 0, "n_ordered_identical": 0, "n_set_identical": 0, "replaced_by_hrhv2": [],
            "max_near_tie_frac_k10": {"A": 0.0, "B": 0.0}, "boundary_tie_layers": {"A": [], "B": []},
            "n_worker_processes": nproc}
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
    # V-O2, final tables of previously processed models (3B, 7B, 17 ViTs) vs the frozen S1.2 layer-max cache
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
    cl = {k: v for k, v in clip.items() if not k.startswith("_") and k in side}
    summ["clip_bitwise_equal_reference_all"] = all(v["bitwise_equal"] for v in cl.values())
    summ["new_lm_tables"] = {f"{s}|{pre}": {"n_layers": len(res[f"{s}|{pre}"]),
                                            "n_set_identical": sum(r["set_identical"] for r in res[f"{s}|{pre}"])}
                             for s in LC.NEW_LMS if s in side for pre in LC.PRES}
    sh = clip.get("_shared_cache", {})
    summ["VO1_pass"] = bool(summ["clip_bitwise_equal_reference_all"]
                            and (sh.get("action") != "compared" or sh.get("bitwise_equal_all", False)))
    summ["VO2_pass"] = bool(summ["n_set_identical"] == summ["n_tables"] and not summ["boundary_tie_layers"]["B"]
                            and summ["frozen_cache_identical"] == summ["frozen_cache_compared"])
    out = {"plan_items": "V-O1 (clip thresholds bitwise: 13B = clip_thresholds_partB_ol.json, 3B/7B/ViTs = S1.2 "
                         "cache) and V-O2 (kNN tables = hrhv2.structural.knn_table on every layer, both pre; ViT side "
                         "0 boundary ties; frozen caches equal)",
           "mode": {"test": LC.TEST, "sens_e2": LC.SENS_E2, "models_subset": sorted(ONLY) if ONLY else None},
           "summary": summ, "clip": clip, "per_table": res, "frozen_cache_cmp": frozen_cmp, "finished_utc": LC.utc()}
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    name = "VO1_VO2_knn_tables.json" if not ONLY else "VO1_VO2_knn_tables.subset.json"
    with open(os.path.join(LC.RES, "validation", name), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    print("V-O1/V-O2 summary", json.dumps(LC.jsonable(summ))[:2000], flush=True)
    print("V-O1", "PASS" if summ["VO1_pass"] else "FAIL", "V-O2", "PASS" if summ["VO2_pass"] else "FAIL", flush=True)


def main():
    global ONLY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None,
                    help="debug only: comma list of stems or 3b/7b/13b (the output file then gets the suffix .subset)")
    ap.add_argument("--phase", choices=("all", "gpu", "cpu"), default="all",
                    help="gpu = phase 1 (GPU tables, clip thresholds); cpu = phase 2 (hrhv2 reference, V-O1/V-O2 "
                         "file; reads the phase-1 clip.json); all = both")
    args = ap.parse_args()
    if args.models:
        lab = {v: k for k, v in LC.SIZE_LABEL.items()}
        ONLY = {lab.get(x.strip(), x.strip()) for x in args.models.split(",") if x.strip()}
    if args.phase in ("all", "gpu"):
        clip = phase1()
    else:
        clip = json.load(open(os.path.join(LC.CACHE, "clip.json")))
        missing = [s for s, _ in _models() if s not in clip]
        if missing:
            sys.exit(f"phase cpu: clip.json lacks {missing}; run --phase gpu first")
    if args.phase in ("all", "cpu"):
        phase2(clip)


if __name__ == "__main__":
    main()
