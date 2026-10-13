"""SPEC §2.10 / §8: hrhv2_ref SOG overlaps vs the pinned SOG code (be654a37).

For a realistic non-monotone pair (same generator as scipy_crosscheck.py) at the
given m, report for every tau:
  ref        : ladder_ref default (Q over ordered positive entries, eps clamp)  [v0.2]
  ref_unord  : Q over unordered pairs, one-zero -> 0                           [v0.1]
  sog_f64    : SOG functions fed float64 L2 rows / our float64 d
  sog_f32    : SOG's own pipeline dtype (float32 features, torch.topk kNN,
               float32 cdist + Prim for the MST support, float32 values cast to f64)
and the kNN-set / MST-edge agreement of the float32 pipeline with the ref.
Usage: python sog_crosscheck.py OUT_JSON [m ...]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, HERE)
import torch  # noqa: E402

from hrhv2_ref import ladder_ref  # noqa: E402
from hrhv2_ref import structural as st  # noqa: E402
from hrhv2_ref.core import gram_sym, neighbor_order, preprocess, sqdist_from_gram  # noqa: E402
from scipy_crosscheck import pair  # noqa: E402
from sog_vendor_loader import load_sog, sog_head_commit  # noqa: E402

TAUS = (1.0, 0.1, 0.01, 0.001, 0.0001)


def sog_values(ns, XA, XB, NA, NB, dA, dB, m):
    """SOG kNN + MST distance-aware overlaps.  XA/XB: L2 rows as torch tensors
    (dtype decides the kNN-path precision).  NA/NB: kNN index tensors.
    dA/dB: float64 numpy plain-distance matrices for the MST value path."""
    lA = ns["_knn_log_distance_values_from_layer"](XA, NA, q=0.9)
    lB = ns["_knn_log_distance_values_from_layer"](XB, NB, q=0.9)
    vA = ns["_normalize_distance_array"](dA, q=0.9)
    vB = ns["_normalize_distance_array"](dB, q=0.9)
    return lA, lB, vA, vB


def main():
    out_path = sys.argv[1]
    ms = [int(a) for a in sys.argv[2:]] or [1024]
    ns = load_sog()
    res = {"sog_commit": sog_head_commit(), "torch": torch.__version__}
    for m in ms:
        t0 = time.time()
        A, B = pair(m, f"hrhv2_ref/xcheck|{m}")
        Ap, _ = preprocess(A, "l2")
        Bp, _ = preprocess(B, "l2")
        ref = ladder_ref(A, B, "l2", parts={"sog"})
        ref_u = ladder_ref(A, B, "l2", parts={"sog"}, sog_q_mode="unordered", sog_eps=0.0)
        DA, DB = sqdist_from_gram(gram_sym(Ap)), sqdist_from_gram(gram_sym(Bp))
        oA, oB = neighbor_order(DA, 10), neighbor_order(DB, 10)
        TA, TB = st.mst_kruskal(DA), st.mst_kruskal(DB)
        dA, dB = np.sqrt(DA), np.sqrt(DB)
        rec = {"m": m, "sog_Q_A_ordered": ref["sog_Q_A"], "sog_Q_A_unordered": ref_u["sog_Q_A"],
               "sog_Q_B_ordered": ref["sog_Q_B"], "sog_Q_B_unordered": ref_u["sog_Q_B"]}
        # --- SOG fed float64 ------------------------------------------------
        t64A, t64B = torch.from_numpy(Ap), torch.from_numpy(Bp)
        nA, nB = torch.from_numpy(oA), torch.from_numpy(oB)
        lA, lB, vA, vB = sog_values(ns, t64A, t64B, nA, nB, dA, dB, m)
        eA = np.sort(ns["_mst_edge_codes_numpy"](vA))
        eB = np.sort(ns["_mst_edge_codes_numpy"](vB))
        rec["f64_mst_edges_equal_ref"] = bool(np.array_equal(eA, TA) and np.array_equal(eB, TB))
        # --- SOG native float32 pipeline -------------------------------------
        f32A = torch.nn.functional.normalize(torch.from_numpy(A).float(), p=2, dim=-1)
        f32B = torch.nn.functional.normalize(torch.from_numpy(B).float(), p=2, dim=-1)
        kA = ns["_compute_knn_indices"](f32A, 10)
        kB = ns["_compute_knn_indices"](f32B, 10)
        rec["f32_knn_sets_equal_ref_A"] = int(np.sum(np.all(np.sort(kA.numpy(), 1) == np.sort(oA, 1), axis=1)))
        rec["f32_knn_sets_equal_ref_B"] = int(np.sum(np.all(np.sort(kB.numpy(), 1) == np.sort(oB, 1), axis=1)))
        DrA = ns["_pairwise_distances"](f32A, metric="euclidean")
        DrB = ns["_pairwise_distances"](f32B, metric="euclidean")
        sA = ns["_quantile_normalize"](DrA, q=0.9).numpy().astype(np.float64)
        sB = ns["_quantile_normalize"](DrB, q=0.9).numpy().astype(np.float64)
        e32A = np.sort(ns["_mst_edge_codes_numpy"](sA))
        e32B = np.sort(ns["_mst_edge_codes_numpy"](sB))
        rec["f32_mst_shared_with_ref_A"] = int(np.intersect1d(e32A, TA).size)
        rec["f32_mst_shared_with_ref_B"] = int(np.intersect1d(e32B, TB).size)
        l32A, l32B, v32A, v32B = sog_values(ns, f32A, f32B, kA, kB, DrA.numpy().astype(np.float64),
                                            DrB.numpy().astype(np.float64), m)
        rows = []
        for t in TAUS:
            lab = f"{t:g}"
            sk64 = float(ns["_knn_distance_overlap_from_log_values"](nA, nB, lA, lB, topk=10, tau=t))
            sm64 = ns["_mst_distance_overlap_similarity_from_cached"](vA, eA, vB, eB, n=m, distance_tau=t)
            sk32 = float(ns["_knn_distance_overlap_from_log_values"](kA, kB, l32A, l32B, topk=10, tau=t))
            sm32 = ns["_mst_distance_overlap_similarity_from_cached"](v32A, e32A, v32B, e32B, n=m, distance_tau=t)
            r = {"tau": t,
                 "knn_ref": ref[f"sog_knn@{lab}"], "knn_ref_unord": ref_u[f"sog_knn@{lab}"],
                 "knn_sog_f64": sk64, "knn_sog_f32": sk32,
                 "mst_ref": ref[f"sog_mst@{lab}"], "mst_ref_unord": ref_u[f"sog_mst@{lab}"],
                 "mst_sog_f64": sm64, "mst_sog_f32": sm32}
            for v in ("knn", "mst"):
                base = r[f"{v}_ref"]
                for other in ("ref_unord", "sog_f64", "sog_f32"):
                    r[f"{v}_absdiff_ref_vs_{other}"] = abs(base - r[f"{v}_{other}"])
            rows.append(r)
            print(m, json.dumps(r), flush=True)
        rec["by_tau"] = rows
        rec["seconds"] = round(time.time() - t0, 1)
        res[str(m)] = rec
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
