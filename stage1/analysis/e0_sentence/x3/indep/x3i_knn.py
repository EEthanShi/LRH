"""x3-indep step 1: neighbour tables of one model under ctr.

usage: x3i_knn.py <run_dir> <model_id>
model_id: dinov2_b | x3_cls_r10 | x3_tok_r10 | x3_tok_r05 | x3add_schnaus224 |
          mpnet.n5 | contriever.n5 | qwen3emb06.n5
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x3i_common as C  # noqa: E402


def main():
    run_dir, mid = sys.argv[1], sys.argv[2]
    t0 = time.time()
    c0 = time.process_time()
    design = C.build_design()
    if not all(v for k, v in design["checks"].items() if k.endswith("design_npz")):
        raise RuntimeError(f"design differs from design.npz: {design['checks']}")
    Xc = C.model_features(mid)
    feat_sha = C.sha256_file(C.P(C.VISION[mid][0])) if mid in C.VISION else None
    tabs = C.knn_tables(Xc, design)
    arrs = {}
    meta = {"model": mid, "d": int(Xc.shape[1]), "median_D": {}, "n_h2": {}, "n_obs": {},
            "n_exact_tie_at_k": {}, "n_zero_offdiag": tabs["n_zero_offdiag"],
            "feature_sha256": feat_sha, "design_checks": design["checks"],
            "omp": os.environ.get("OPENBLAS_NUM_THREADS")}
    for d in range(C.N_DRAWS):
        for m in C.M_GRID:
            t = tabs[(d, m)]
            key = f"d{d}_m{m}"
            arrs[key + "_pos"] = t["pos"]
            arrs[key + "_isobs"] = t["is_obs"]
            arrs[key + "_idx"] = t["idx"]
            arrs[key + "_vals"] = t["vals"]
            arrs[key + "_tie"] = t["exact_tie"]
            meta["median_D"][key] = tabs["median_D"][(d, m)]
            meta["n_h2"][key] = int(len(t["pos"]))
            meta["n_obs"][key] = int(t["is_obs"].sum())
            meta["n_exact_tie_at_k"][key] = t["exact_tie"].sum(axis=0).tolist()
    os.makedirs(os.path.join(run_dir, "knn"), exist_ok=True)
    out = os.path.join(run_dir, "knn", f"{mid}.npz")
    if os.path.exists(out):
        raise FileExistsError(out)
    np.savez(out + ".tmp.npz", **arrs)
    os.rename(out + ".tmp.npz", out)
    meta["npz_sha256"] = C.sha256_file(out)
    meta["wall_s"] = time.time() - t0
    meta["cpu_s"] = time.process_time() - c0
    C.write_json_new(os.path.join(run_dir, "knn", f"{mid}.json"), meta)
    print(mid, "done", round(meta["wall_s"], 1), "s", meta["npz_sha256"][:16], flush=True)


if __name__ == "__main__":
    main()
