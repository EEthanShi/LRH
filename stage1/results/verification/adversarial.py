#!/usr/bin/env python
"""Verifier, task 3: targeted adversarial probes of hrhv2.

A. Cross-fit data separation of every predictor (P1, P2, N0, N1, N2): perturb, inside
   the fit prefix, ONLY the B entries of pairs (H1 x H2) -- these are entries of the
   predicted (H2) anchors' own B-rows -- and see which predictions move.
B. Same with the B entries of pairs (H2 x H2) and of items beyond the fit prefix.
C. calibrate_knn fast path vs SPEC-5 full recomputation (kNN recomputed on the
   permuted matrix with the SPEC tie rule) on WIT-1024 LM features that contain an
   exact duplicate item pair (493, 531), k in {10, 30}.
Usage: python adversarial.py OUT.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/hrh_v2_stage1_20261008"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from hrhv2 import plants, predict, util, structural as st, calibrate as cal  # noqa: E402
from hrhv2.prep import Geom  # noqa: E402
from agree import layer_index, FEAT  # noqa: E402


def sym_perturb(DB, mask, amount, key):
    P = DB.copy()
    noise = util.rng_from_key(key).random(DB.shape) * amount
    noise = np.triu(noise, 1)
    noise = noise + noise.T
    P[mask] = P[mask] + noise[mask]
    return P


def probe_separation():
    m_max, m_fit = 768, 384
    DA = plants.feature_dists(plants.base_cloud(m_max, 3, "VERIF1|adv"))
    sig = 0.15 * plants.knn_d_median(DA, 10)
    DB = DA + plants.gauss_noise_matrix(m_max, sig, "VERIF1|adv")
    h1 = util.hash_half(plants.nested_ids(m_max), "hrhv2-halves")
    kw = dict(m_fit=m_fit, m_list=[256, 384, 768], ks=[1, 10], variants=[predict.FROZEN_P1_VARIANT], oracle_sigma=sig)
    base = predict.crossfit_predict(DA, DB, h1, **kw)
    inpref = np.zeros((m_max, m_max), dtype=bool)
    inpref[:m_fit, :m_fit] = True
    H1, H2 = np.outer(h1, h1), np.outer(~h1, ~h1)
    cross = ~(H1 | H2)
    masks = {
        "H1xH2_in_fit_prefix(target B-rows)": cross & inpref,
        "H2xH2_in_fit_prefix(target B-rows)": H2 & inpref,
        "outside_fit_prefix": ~inpref,
    }
    out = {}
    for name, msk in masks.items():
        P = sym_perturb(DB, msk, 0.5 * sig, "VERIF1|adv|" + name)
        r = predict.crossfit_predict(DA, P, h1, **kw)
        moved = {}
        for mm in (256, 384, 768):
            for k in (1, 10):
                for pred in (predict.FROZEN_P1_VARIANT, "N0", "N1", "N2", "P2"):
                    if r[mm][k][pred] != base[mm][k][pred]:
                        moved.setdefault(pred, []).append([mm, k, base[mm][k][pred], r[mm][k][pred]])
        out[name] = {"predictors_that_moved": sorted(moved), "examples": {k: v[:2] for k, v in moved.items()}}
    return out


def recompute_null(DA, DB, k, key, K):
    nA, _ = st.knn_table(DA, k)
    vals = []
    for kk in range(K):
        pi = cal.permutation(DA.shape[0], key, kk)
        nB, _ = st.knn_table(DB[np.ix_(pi, pi)], k)
        vals.append(st.knn_overlap(nA, nB, k))
    return np.array(vals)


def probe_calib_ties(K=100):
    res = {}
    for name, r in (("bigscience__bloomz-1b7", 1.0), ("bigscience__bloomz-1b7", 0.5)):
        other = "vit_large_patch14_dinov2.lvd142m"
        XA = np.asarray(np.load(os.path.join(FEAT, name + ".npy"), mmap_mode="r")[:, layer_index(name, r), :], dtype=np.float64)
        XB = np.asarray(np.load(os.path.join(FEAT, other + ".npy"), mmap_mode="r")[:, layer_index(other, r), :], dtype=np.float64)
        # put the LM on the PERMUTED side (B) so its duplicate-item ties are re-broken by new indices
        A, B = Geom.from_features(XB, "l2"), Geom.from_features(XA, "l2")
        for k in (10, 30):
            key = f"VERIF1|calibties|{name}|{r}|{k}"
            nA, _ = st.knn_table(A.D, k)
            nB, _ = st.knn_table(B.D, k)
            # fast-path null re-derived explicitly (same code path as calibrate_knn)
            fnull = []
            for kk in range(K):
                pi = cal.permutation(A.m, key, kk)
                inv = np.empty(A.m, dtype=np.int64)
                inv[pi] = np.arange(A.m)
                fnull.append(st.knn_overlap(nA, inv[nB[pi, :k]], k))
            fnull = np.array(fnull)
            rnull = recompute_null(A.D, B.D, k, key, K)
            sobs = st.knn_overlap(nA, nB, k)
            sf, sr = cal.summarize(sobs, fnull), cal.summarize(sobs, rnull)
            res[f"{name}|r={r}|k={k}"] = {
                "n_perms_differ": int(np.sum(fnull != rnull)), "max_abs_null_diff": float(np.max(np.abs(fnull - rnull))),
                "unit_1_over_mk": 1.0 / (A.m * k),
                "c_alpha_fast": sf["c_alpha"], "c_alpha_recompute": sr["c_alpha"],
                "s_cal_fast": sf["s_cal"], "s_cal_recompute": sr["s_cal"],
                "uncensored_fast": sf["uncensored"], "uncensored_recompute": sr["uncensored"], "K": K}
            print(json.dumps({f"{name}|r={r}|k={k}": res[f"{name}|r={r}|k={k}"]}), flush=True)
    return res


def main():
    out = {"separation": probe_separation()}
    print(json.dumps(out["separation"], indent=1), flush=True)
    out["calib_ties"] = probe_calib_ties()
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
