#!/usr/bin/env python
"""OPENLLAMA-AXIS COPY of analysis/s12_layermax_llama/indep/vl6_arh.py (sub-role I of oll-axis-extract,
S0b): plan V-O6 (reduced; once, in I, alone, <= 4 threads, at S8). Same vendored ARH entry points and
the same permutation injection (perm_fn = PermFeeder with our keys S1.2-layermax|<lm>|<vit>).

  (1) scalar calibrate() on (open_llama_13b last, ViT-Ti/16-in21k last), mKNN (PRH mutual_knn) and
      biased CKA (ARH cka_biased);
  (2) calibrate_layers(agg="max"), biased CKA, on grid_layers(41, 5) x grid_layers(12, 5) of
      open_llama_13b x ViT-Ti/16-in21k, against this run's exact sub-grid maximum.
  The full 41 x 12 grid and the mKNN grid of the LLaMA copy are dropped (plan V-O6).
pre = prh_clip+l2 (primary). Tolerance 1e-6. Tie-rule differences are reported (observed-matrix
comparison on the sub-grid). Part B gate first (plan F1(a)).
usage: launch_ol.sh ol_I_vo6 4 --gpu 8 -- venv/bin/python analysis/s12_layermax_openllama/indep/vl6_arh.py
       [--device cuda|cpu]   (default cuda, as the LLaMA copy; GPU memory fraction 0.30)
Output: results/S1_2_layermax_openllama/indep/validation_vo6_arh.json
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ol_gate  # noqa: E402
OUT_ROOT = ol_gate.gate(__file__)   # F1(a): exits 3 unless Part B is frozen (S0b test mode: scratch root)

_T = os.environ.get("OL_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T
os.environ["VERIF_TORCH_THREADS"] = _T

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

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
    ap = argparse.ArgumentParser(description="plan V-O6 (reduced), implementation I")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--pairs-root", default=None, help="S0b test mode only")
    ap.add_argument("--K-test", type=int, default=None, help="S0b test mode only: first K permutations on both sides")
    args = ap.parse_args()
    if (args.pairs_root or args.K_test) and not ol_gate.test_mode():
        raise SystemExit("--pairs-root / --K-test are S0b test options")
    root = args.pairs_root or L.OUT
    KP = args.K_test or L.K_PERM
    import torch
    torch.set_num_threads(int(_T))
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, os.path.join(L.PROJ, "results", "verification"))
    import vendor_identity as vi  # verifier helpers: pinned PRH mutual_knn, ARH estimators/calibrate
    from calibrated_similarity.calibration import calibrate_layers as arh_calibrate_layers
    dev = torch.device(args.device)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.30, 0)
    sims = {"mknn": (lambda X, Y: vi.prh_mknn(X, Y, L.KNN)), "cka_b": (lambda X, Y: vi.arh_est.cka_biased(X, Y))}

    def feats(stem):
        S = L.load_stack64(stem)
        c = C.clip_threshold(stem, "wit1024")
        return [torch.from_numpy(C.preprocess_rows(S[:, l, :], PRE, c)).to(dev) for l in range(S.shape[1])]

    def mats(lm):
        z = np.load(os.path.join(root, PRE, "pairs", f"{lm}__{VIT}.npz"))
        return z, {"mknn": (z["obs_cnt"] / (L.M * L.KNN), z["null_cnt"][:KP].astype(np.float64) / (L.M * L.KNN)),
                   "cka_b": (z["obs_cka_b"], z["null_cka_b"][:KP])}

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

    out = {"pre": PRE, "K": KP, "tolerance": 1e-6, "device": args.device, "pairs_root": root}
    XB = feats(VIT)
    # (1) scalar, open_llama_13b last x ViT-Ti last
    lm = C.SLUG13
    XA30 = feats(lm)
    z30, M30 = mats(lm)
    a1, b1 = len(XA30) - 1, len(XB) - 1
    key30 = L.perm_key(lm, VIT)
    for name, sim in sims.items():
        t0 = time.time()
        feeder = vi.PermFeeder(L.M, key30, KP)
        res = vi.arh_calibrate(XA30[a1], XB[b1], sim, K=KP, alpha=L.ALPHA, smax=1.0, perm_fn=feeder)
        out[f"scalar_{name}"] = {"pair": [lm, VIT], "layers": [a1, b1], **cmp(res, ours(*M30[name], a=a1, b=b1)),
                                 "secs": time.time() - t0}
        log(f"V-O6 scalar {name} {out[f'scalar_{name}']['absdiff']}")
    # (2) biased-CKA calibrate_layers on grid_layers(L_A, 5) x grid_layers(L_B, 5) (41 x 12 -> 5 x 5)
    ra = np.unique([rhu(j / 4 * (len(XA30) - 1)) for j in range(5)])
    rb = np.unique([rhu(j / 4 * (len(XB) - 1)) for j in range(5)])
    t0 = time.time()
    feeder = vi.PermFeeder(L.M, key30, KP)
    res = arh_calibrate_layers([XA30[a] for a in ra], [XB[b] for b in rb], sims["cka_b"], agg="max", K=KP,
                               alpha=L.ALPHA, smax=1.0, perm_fn=feeder)
    out["layers_cka_b_grid5x5"] = {"pair": [lm, VIT], "lm_layers": ra.tolist(), "vit_layers": rb.tolist(),
                                   **cmp(res, ours(*M30["cka_b"], ra=ra, rb=rb)), "secs": time.time() - t0}
    log(f"V-O6 layers cka_b grid5x5 {out['layers_cka_b_grid5x5']['absdiff']} {time.time() - t0:.0f}s")
    S = np.array([[sims["cka_b"](XA30[a], XB[b]) for b in rb] for a in ra])
    d = np.abs(S - M30["cka_b"][0][np.ix_(ra, rb)])
    out["obsmatrix_cka_b_grid5x5"] = {"maxabs": float(d.max()), "n_diff_gt_1e-6": int((d > 1e-6).sum()),
                                      "n_entries": int(d.size)}
    S = np.array([[sims["mknn"](XA30[a], XB[b]) for b in rb] for a in ra])
    d = np.abs(S - M30["mknn"][0][np.ix_(ra, rb)])
    out["obsmatrix_mknn_grid5x5_informational"] = {"maxabs": float(d.max()), "n_diff_gt_1e-6": int((d > 1e-6).sum()),
                                                   "n_entries": int(d.size)}
    diffs = [v for k, dd in out.items() if isinstance(dd, dict) and "absdiff" in dd for v in dd["absdiff"].values()]
    out["max_absdiff"] = float(max(diffs))
    out["pass_1e-6"] = bool(out["max_absdiff"] <= 1e-6)
    fn = os.path.join(L.OUT, "validation_vo6_arh.json")
    json.dump(L.jsonable(out), open(fn + ".tmp", "w"), indent=1)
    os.replace(fn + ".tmp", fn)
    log(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
