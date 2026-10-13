#!/usr/bin/env python
"""LLaMA axis, implementation I (role llama-axis-run-indep): plan V-L6.
Adapted from analysis/s12_layermax/indep/validate.py::i4 (same vendored ARH entry
points, same permutation injection through perm_fn = PermFeeder with our keys).

  (1) scalar calibrate() on (llama-30b last, ViT-Ti/16-in21k last), mKNN (PRH
      mutual_knn) and biased CKA (ARH cka_biased);
  (2) calibrate_layers(agg="max"), biased CKA, full llama-13b x ViT-Ti/16-in21k grid (41 x 12);
  (3) calibrate_layers(agg="max"), mKNN, llama-30b LM layers grid_layers(61, 9) x all 12
      ViT-Ti layers, against this run's exact sub-grid maximum.
pre = prh_clip+l2 (primary). Tolerance 1e-6 (ARH stores float32). Tie-rule differences
are reported (observed-matrix comparison per layer pair).
Output: results/S1_2_layermax_llama/indep/validation_vl6_arh.json
"""
from __future__ import annotations

import os

_T = os.environ.get("LMX_THREADS", "3")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T
os.environ["VERIF_TORCH_THREADS"] = _T

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import lmx_common as L  # noqa: E402
from lmx_common import C  # noqa: E402

VIT = "vit_tiny_patch16_224.augreg_in21k"
PRE = "prh_clip+l2"


def rhu(x):
    import math
    return int(math.floor(x + 0.5))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import torch
    torch.set_num_threads(int(_T))
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, os.path.join(L.PROJ, "results", "verification"))
    import vendor_identity as vi  # verifier helpers: pinned PRH mutual_knn, ARH estimators/calibrate
    from calibrated_similarity.calibration import calibrate_layers as arh_calibrate_layers
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.30, 0)
    sims = {"mknn": (lambda X, Y: vi.prh_mknn(X, Y, L.KNN)), "cka_b": (lambda X, Y: vi.arh_est.cka_biased(X, Y))}

    def feats(stem):
        S = L.load_stack64(stem)
        c = C.clip_threshold(stem, "wit1024")
        return [torch.from_numpy(C.preprocess_rows(S[:, l, :], PRE, c)).to(dev) for l in range(S.shape[1])]

    def mats(lm):
        z = np.load(os.path.join(L.OUT, PRE, "pairs", f"{lm}__{VIT}.npz"))
        return z, {"mknn": (z["obs_cnt"] / (L.M * L.KNN), z["null_cnt"].astype(np.float64) / (L.M * L.KNN)),
                   "cka_b": (z["obs_cka_b"], z["null_cka_b"])}

    def ours(S_obs, S_null, ra=None, rb=None, a=None, b=None):
        if a is not None:
            T, null = float(S_obs[a, b]), S_null[:, a, b]
        else:
            Ss = S_obs[np.ix_(ra, rb)]
            T, null = float(Ss.max()), S_null[:, ra][:, :, rb].reshape(S_null.shape[0], -1).max(axis=1)
        return L.summarize(T, null)

    def cmp(arh, o):
        sc, p, tau = (float(x) for x in arh)
        return {"arh": {"s_cal": sc, "p": p, "tau": tau},
                "ours": {k: o[k] for k in ("s_obs", "s_cal", "p_value", "c_alpha")},
                "absdiff": {"s_cal": abs(sc - o["s_cal"]), "p": abs(p - o["p_value"]), "tau": abs(tau - o["c_alpha"])}}

    out = {"pre": PRE, "K": L.K_PERM, "tolerance": 1e-6}
    XB = feats(VIT)
    # (1) scalar, llama-30b last x ViT-Ti last
    lm = "huggyllama__llama-30b"
    XA30 = feats(lm)
    z30, M30 = mats(lm)
    a1, b1 = len(XA30) - 1, len(XB) - 1
    key30 = L.perm_key(lm, VIT)
    for name, sim in sims.items():
        t0 = time.time()
        feeder = vi.PermFeeder(L.M, key30, L.K_PERM)
        res = vi.arh_calibrate(XA30[a1], XB[b1], sim, K=L.K_PERM, alpha=L.ALPHA, smax=1.0, perm_fn=feeder)
        out[f"scalar_{name}"] = {"pair": [lm, VIT], "layers": [a1, b1], **cmp(res, ours(*M30[name], a=a1, b=b1)),
                                 "secs": time.time() - t0}
        log(f"V-L6 scalar {name} {out[f'scalar_{name}']['absdiff']}")
    # (3) mKNN calibrate_layers on grid_layers(61, 9) x all ViT-Ti layers
    ra = np.unique([rhu(j / 8 * (len(XA30) - 1)) for j in range(9)])
    rb = np.arange(len(XB))
    t0 = time.time()
    feeder = vi.PermFeeder(L.M, key30, L.K_PERM)
    res = arh_calibrate_layers([XA30[a] for a in ra], XB, sims["mknn"], agg="max", K=L.K_PERM, alpha=L.ALPHA,
                               smax=1.0, perm_fn=feeder)
    out["layers_mknn_llama30b_grid9"] = {"pair": [lm, VIT], "lm_layers": ra.tolist(),
                                         **cmp(res, ours(*M30["mknn"], ra=ra, rb=rb)), "secs": time.time() - t0}
    log(f"V-L6 layers mknn grid9 {out['layers_mknn_llama30b_grid9']['absdiff']} {time.time() - t0:.0f}s")
    S = np.array([[sims["mknn"](XA30[a], y) for y in XB] for a in ra])
    d = np.abs(S - M30["mknn"][0][np.ix_(ra, rb)])
    out["obsmatrix_mknn_llama30b_grid9"] = {"maxabs": float(d.max()), "n_diff_gt_1e-6": int((d > 1e-6).sum()),
                                            "n_entries": int(d.size)}
    del XA30
    torch.cuda.empty_cache()
    # (2) CKA calibrate_layers on the full llama-13b x ViT-Ti grid
    lm = "huggyllama__llama-13b"
    XA13 = feats(lm)
    z13, M13 = mats(lm)
    key13 = L.perm_key(lm, VIT)
    t0 = time.time()
    feeder = vi.PermFeeder(L.M, key13, L.K_PERM)
    res = arh_calibrate_layers(XA13, XB, sims["cka_b"], agg="max", K=L.K_PERM, alpha=L.ALPHA, smax=1.0, perm_fn=feeder)
    out["layers_cka_b_llama13b_full"] = {"pair": [lm, VIT], "grid": [len(XA13), len(XB)],
                                         **cmp(res, ours(*M13["cka_b"], ra=np.arange(len(XA13)), rb=rb)),
                                         "secs": time.time() - t0}
    log(f"V-L6 layers cka_b full {out['layers_cka_b_llama13b_full']['absdiff']} {time.time() - t0:.0f}s")
    S = np.array([[sims["cka_b"](x, y) for y in XB] for x in XA13])
    d = np.abs(S - M13["cka_b"][0])
    out["obsmatrix_cka_b_llama13b_full"] = {"maxabs": float(d.max()), "n_diff_gt_1e-6": int((d > 1e-6).sum()),
                                            "n_entries": int(d.size)}
    diffs = [v for k, dd in out.items() if isinstance(dd, dict) and "absdiff" in dd for v in dd["absdiff"].values()]
    out["max_absdiff"] = float(max(diffs))
    out["pass_1e-6"] = bool(out["max_absdiff"] <= 1e-6)
    json.dump(L.jsonable(out), open(os.path.join(L.OUT, "validation_vl6_arh.json"), "w"), indent=1)
    log(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
