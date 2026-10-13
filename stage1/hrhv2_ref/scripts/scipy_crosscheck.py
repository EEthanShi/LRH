"""Cross-check reference algorithms against library implementations at realistic m.

Pairs: noisy, non-monotone (A = cloud, B = A + noise, then L2), m in {1024, 4096}.
Reports max abs differences; the reference must agree with:
  scipy.stats.kendalltau(variant='b') / spearmanr (global, tie-free data),
  scipy per-row kendalltau on a row sample ((1 - tau_b)/2),
  scipy.optimize.isotonic_regression (global iso, in-sample, x sorted, no ties),
  scipy.sparse.csgraph.minimum_spanning_tree (tie-free),
  PRH-style kNN (argsort of cosine similarity) and ARH feature-space linear CKA,
  brute-force Rand distance.
Usage: python scipy_crosscheck.py OUT_JSON [m ...]
"""
import json
import os
import sys
import time

import numpy as np
import scipy
from scipy import optimize, stats
from scipy.sparse import csgraph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hrhv2_ref import ladder_ref  # noqa: E402
from hrhv2_ref import geometric as geo  # noqa: E402
from hrhv2_ref import quotient as quo  # noqa: E402
from hrhv2_ref import structural as st  # noqa: E402
from hrhv2_ref.core import gram_sym, neighbor_order, preprocess, sha256_int, sqdist_from_gram, upper  # noqa: E402


def pair(m, key):
    rng = np.random.Generator(np.random.PCG64(sha256_int(key)))
    Z = rng.normal(size=(m, 6))
    Q, _ = np.linalg.qr(rng.normal(size=(64, 6)))
    A = Z @ Q.T + 0.05 * rng.normal(size=(m, 64))
    B = np.tanh(A @ rng.normal(size=(64, 48)) / 4) + 0.1 * rng.normal(size=(m, 48))
    return A, B


def main():
    out_path = sys.argv[1]
    ms = [int(a) for a in sys.argv[2:]] or [1024]
    res = {"scipy": scipy.__version__, "numpy": np.__version__}
    for m in ms:
        r = {}
        t0 = time.time()
        A, B = pair(m, f"hrhv2_ref/xcheck|{m}")
        Ap, _ = preprocess(A, "l2")
        Bp, _ = preprocess(B, "l2")
        KA, KB = gram_sym(Ap), gram_sym(Bp)
        DA, DB = sqdist_from_gram(KA), sqdist_from_gram(KB)
        a, b = upper(DA), upper(DB)
        r["distinct_A"] = int(np.unique(a).size)
        r["distinct_B"] = int(np.unique(b).size)
        r["n_pairs"] = int(a.size)
        out = ladder_ref(A, B, "l2", parts={"knn", "comp", "rank", "edge", "mst", "cka", "quot"})
        r["ladder_seconds"] = round(time.time() - t0, 1)
        # global Kendall / Spearman
        r["tau_b_ref"] = out["kendall_global"]
        r["tau_b_scipy"] = float(stats.kendalltau(a, b, variant="b").statistic)
        r["rho_ref"] = out["spearman_global"]
        r["rho_scipy"] = float(stats.spearmanr(a, b).statistic)
        # per-row Kendall on a sample of rows
        rows = st.kendall_rows_merge(DA, DB)
        diffs = []
        for i in range(0, m, max(1, m // 64)):
            ai = np.delete(DA[i], i)
            bi = np.delete(DB[i], i)
            diffs.append(abs(rows[i] - (1 - stats.kendalltau(ai, bi, variant="b").statistic) / 2))
        r["row_kendall_maxdiff"] = float(max(diffs))
        r["delta_O_ref"] = out["delta_O"]
        # global isotonic in-sample
        o = np.argsort(a, kind="stable")
        sp_fit = optimize.isotonic_regression(b[o]).x
        ref_fit = quo.iso_insample(a, b)[o]
        r["iso_maxdiff_vs_scipy"] = float(np.max(np.abs(sp_fit - ref_fit)))
        # MST vs csgraph (tie-free)
        T = st.mst_kruskal(DA)
        S = csgraph.minimum_spanning_tree(DA).tocoo()
        ref = np.sort(np.minimum(S.row, S.col).astype(np.int64) * m + np.maximum(S.row, S.col))
        r["mst_equal_csgraph"] = bool(np.array_equal(T, ref))
        r["prim_eq_kruskal"] = bool(out["prim_eq_kruskal_A"] and out["prim_eq_kruskal_B"])
        r["F_subset_T"] = bool(out["F_A_subset_T_A"] and out["F_B_subset_T_B"])
        # kNN vs PRH-style cosine argsort
        oA = neighbor_order(DA, 101)
        S_ = KA.copy()
        np.fill_diagonal(S_, -1e8)
        prh = np.argsort(-S_, axis=1, kind="stable")[:, :100]
        r["knn100_sets_equal_prh_style"] = bool(all(set(oA[i, :100]) == set(prh[i]) for i in range(m)))
        # CKA biased vs ARH feature-space formula
        Xc = Ap - Ap.mean(0)
        Yc = Bp - Bp.mean(0)
        arh_lin = np.linalg.norm(Yc.T @ Xc) ** 2 / (np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc))
        r["cka_biased_ref"] = out["cka_biased"]
        r["cka_biased_featurespace"] = float(arh_lin)
        r["cka_debiased"] = out["cka_debiased"]
        r["cka_debiased_minus_eps8"] = out["cka_debiased"] - out["cka_debiased_eps8"]
        r["cka_debiased_minus_eps6"] = out["cka_debiased"] - out["cka_debiased_eps6"]
        r["hsic_u_AA"], r["hsic_u_BB"] = out["hsic_u_AA"], out["hsic_u_BB"]
        # Rand distance brute force (vectorised over all pairs)
        for k in (1, 3, 10):
            la = st.knn_components(oA[:, :k])
            lb = st.knn_components(neighbor_order(DB, k)[:, :k])
            iu, ju = np.triu_indices(m, 1)
            brute = float(np.mean((la[iu] == la[ju]) != (lb[iu] == lb[ju])))
            r[f"rand@{k}_absdiff"] = abs(brute - out[f"comp_rand@{k}"])
            r[f"comp_n@{k}"] = (out[f"comp_n_A@{k}"], out[f"comp_n_B@{k}"])
        r["summary"] = {k: out[k] for k in ("s_N@1", "s_N@10", "s_N@100", "s_T", "s_T_forest", "s_T_bridge",
                                            "forest_share_A", "forest_share_shared_A", "delta_O", "delta_E",
                                            "quot_affine_expl", "quot_iso_expl", "quot_rowiso_expl",
                                            "quot_iso_expl_xfit", "quot_rowiso_expl_xfit")}
        res[str(m)] = r
        print(json.dumps({str(m): r}, default=float), flush=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
