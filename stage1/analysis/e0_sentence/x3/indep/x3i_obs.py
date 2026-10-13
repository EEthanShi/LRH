"""x3-indep step 2: per-anchor overlaps and cells for the 15 pairs
(12 X3 pairs + 3 E0 reference pairs), then comparison with the run side.

usage: x3i_obs.py <run_dir>

Run-side files are read as data only: neighbour caches (npz), meta.json,
obs_X3.parquet, E0 obs_CA.parquet. No run-side code is imported or read.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x3i_common as C  # noqa: E402

VIS = ["dinov2_b", "x3_cls_r10", "x3_tok_r10", "x3_tok_r05", "x3add_schnaus224"]
TXT = [f"{e}.n5" for e in C.ENCODERS]


def run_cache_dir(mid: str) -> str:
    if mid == "dinov2_b":
        return C.P("results/E0_sentence/cache/ctr/dinov2_b")
    if mid in C.VISION:
        return C.P(f"results/E0_sentence/x3/cache/ctr/{mid}")
    return C.P(f"results/E0_sentence/cache/ctr/{mid}")


def main():
    run_dir = sys.argv[1]
    t0, c0 = time.time(), time.process_time()
    design = C.build_design()
    mine = {mid: np.load(os.path.join(run_dir, "knn", f"{mid}.npz")) for mid in VIS + TXT}
    theirs = {}
    in_sha = {}
    for mid in VIS + TXT:
        dd = run_cache_dir(mid)
        theirs[mid] = {d: np.load(os.path.join(dd, f"d{d}_knn.npz")) for d in range(C.N_DRAWS)}
        for d in range(C.N_DRAWS):
            in_sha[os.path.relpath(os.path.join(dd, f"d{d}_knn.npz"), C.ROOT)] = C.sha256_file(os.path.join(dd, f"d{d}_knn.npz"))
        in_sha[os.path.relpath(os.path.join(dd, "meta.json"), C.ROOT)] = C.sha256_file(os.path.join(dd, "meta.json"))

    # ---- (a) neighbour tables, per model ----------------------------------
    knn_cmp = {}
    for mid in VIS + TXT:
        meta_t = json.load(open(os.path.join(run_cache_dir(mid), "meta.json")))
        r = {"pos_equal": True, "idx_rows_equal": 0, "idx_rows": 0,
             "firstk_set_rows_differ": {str(k): 0 for k in C.KS},
             "vals_max_abs_diff": 0.0, "median_D_max_abs_diff": 0.0,
             "n_exact_tie_at_k_mine": [0, 0, 0]}
        for d in range(C.N_DRAWS):
            z = theirs[mid][d]
            for m in C.M_GRID:
                key = f"d{d}_m{m}"
                pm, im, vm = mine[mid][key + "_pos"], mine[mid][key + "_idx"], mine[mid][key + "_vals"]
                pt, it, vt = z[f"pos_m{m}"], z[f"idx_m{m}"], z[f"vals_m{m}"]
                if not np.array_equal(pm.astype(np.int64), pt.astype(np.int64)):
                    r["pos_equal"] = False
                    continue
                r["idx_rows"] += len(pm)
                r["idx_rows_equal"] += int(np.all(im.astype(np.int64) == it.astype(np.int64), axis=1).sum())
                for k in C.KS:
                    a = np.sort(im[:, :k].astype(np.int64), axis=1)
                    b = np.sort(it[:, :k].astype(np.int64), axis=1)
                    r["firstk_set_rows_differ"][str(k)] += int(np.any(a != b, axis=1).sum())
                if not np.array_equal(z["val_cols"], np.asarray(C.VALCOLS)):
                    raise RuntimeError("val_cols layout differs")
                r["vals_max_abs_diff"] = max(r["vals_max_abs_diff"], float(np.max(np.abs(vm - vt))))
                md_t = meta_t["draws"][str(d)]["median_D"][str(m)]
                md_m = json.load(open(os.path.join(run_dir, "knn", f"{mid}.json")))["median_D"][key]
                r["median_D_max_abs_diff"] = max(r["median_D_max_abs_diff"], abs(md_t - md_m))
                tie = mine[mid][key + "_tie"]
                r["n_exact_tie_at_k_mine"] = (np.asarray(r["n_exact_tie_at_k_mine"]) + tie.sum(axis=0)).tolist()
        knn_cmp[mid] = r

    # ---- (b) per-anchor overlaps and cells --------------------------------
    rows = []
    per_anchor = {}
    ov_cmp = {"n_anchor_cells": 0, "n_anchor_counts_differ": 0, "differing": []}
    for v in VIS:
        for enc in C.ENCODERS:
            pair = f"{v}__{enc}"
            t = f"{enc}.n5"
            for d in range(C.N_DRAWS):
                for m in C.M_GRID:
                    key = f"d{d}_m{m}"
                    pos = mine[v][key + "_pos"]
                    if not np.array_equal(pos, mine[t][key + "_pos"]):
                        raise RuntimeError("anchor positions differ between models")
                    is_obs = mine[v][key + "_isobs"]
                    ia, ib = mine[v][key + "_idx"], mine[t][key + "_idx"]
                    ja, jb = theirs[v][d][f"idx_m{m}"], theirs[t][d][f"idx_m{m}"]
                    for k in C.KS:
                        ov = C.overlaps(ia, ib, k)
                        ov_t = C.overlaps(ja.astype(np.int64), jb.astype(np.int64), k)
                        per_anchor[f"{pair}|d{d}|m{m}|k{k}"] = ov
                        ov_cmp["n_anchor_cells"] += len(ov)
                        bad = np.flatnonzero(ov != ov_t)
                        ov_cmp["n_anchor_counts_differ"] += len(bad)
                        for b in bad[:50]:
                            ov_cmp["differing"].append({"pair": pair, "draw": d, "m": m, "k": k,
                                                        "prefix_pos": int(pos[b]), "mine": int(ov[b]),
                                                        "theirs": int(ov_t[b])})
                        rows.append({
                            "pair": pair, "vit": v, "enc": enc, "draw": d, "m": m, "k": k,
                            "chance": k / (m - 1),
                            "n_anchors": int(is_obs.sum()), "obs": float(ov[is_obs].sum()) / (is_obs.sum() * k),
                            "n_allH2": int(len(ov)), "obs_allH2": float(ov.sum()) / (len(ov) * k),
                        })
    cells = pd.DataFrame(rows)

    # ---- (c) cells against obs_X3.parquet and E0 obs_CA.parquet -----------
    ox = pd.read_parquet(C.P("results/E0_sentence/x3/obs/obs_X3.parquet"))
    oca = pd.read_parquet(C.P("results/E0_sentence/obs/obs_CA.parquet"))
    in_sha["results/E0_sentence/x3/obs/obs_X3.parquet"] = C.sha256_file(C.P("results/E0_sentence/x3/obs/obs_X3.parquet"))
    in_sha["results/E0_sentence/obs/obs_CA.parquet"] = C.sha256_file(C.P("results/E0_sentence/obs/obs_CA.parquet"))
    ref = oca[oca["pair"].isin([f"dinov2_b__{e}" for e in C.ENCODERS])]
    run = pd.concat([ox, ref], ignore_index=True)
    key = ["pair", "draw", "m", "k"]
    mg = cells.merge(run, on=key, how="outer", suffixes=("_i", "_r"), indicator=True)
    cell_cmp = {"n_cells_mine": int(len(cells)), "n_cells_run": int(len(run)),
                "n_unmatched": int((mg["_merge"] != "both").sum())}
    for col in ["obs", "obs_allH2", "chance"]:
        dif = np.abs(mg[col + "_i"].to_numpy(float) - mg[col + "_r"].to_numpy(float))
        cell_cmp[col + "_max_abs_diff"] = float(np.nanmax(dif))
        cell_cmp[col + "_n_bitwise_equal"] = int((mg[col + "_i"].to_numpy(float) == mg[col + "_r"].to_numpy(float)).sum())
    for col in ["n_anchors", "n_allH2"]:
        cell_cmp[col + "_all_equal"] = bool((mg[col + "_i"].to_numpy() == mg[col + "_r"].to_numpy()).all())

    os.makedirs(os.path.join(run_dir, "obs"), exist_ok=True)
    cells_path = os.path.join(run_dir, "obs", "x3i_cells.parquet")
    if os.path.exists(cells_path):
        raise FileExistsError(cells_path)
    cells.to_parquet(cells_path, index=False)
    pa_path = os.path.join(run_dir, "obs", "x3i_per_anchor_overlaps.npz")
    np.savez_compressed(pa_path + ".tmp.npz", **per_anchor)
    os.rename(pa_path + ".tmp.npz", pa_path)
    out = {"knn_vs_run_caches": knn_cmp, "per_anchor_overlap_vs_run_caches": ov_cmp,
           "cells_vs_run_obs": cell_cmp, "design_checks": design["checks"],
           "inputs_sha256": in_sha,
           "outputs_sha256": {"obs/x3i_cells.parquet": C.sha256_file(cells_path),
                              "obs/x3i_per_anchor_overlaps.npz": C.sha256_file(pa_path)},
           "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
    C.write_json_new(os.path.join(run_dir, "obs", "x3i_obs_compare.json"), out)
    print(json.dumps({k: out[k] for k in ["per_anchor_overlap_vs_run_caches", "cells_vs_run_obs"]}, default=C._json_default)[:3000])
    for mid, r in knn_cmp.items():
        print(mid, r)


if __name__ == "__main__":
    main()
