#!/usr/bin/env python
"""PYT-AXIS COPY of analysis/s12_layermax_openllama/indep/vl6_arh.py (sha256 e75f5fb3...), sub-role pyt-code-I (S0b,
2026-10-10): plan V-P6 (vendored ARH, reduced; once, in I, <= 4 threads, at S8). Same vendored ARH entry points and
the same permutation injection (perm_fn = PermFeeder with our keys S1.2-layermax|<lm>|<vit>).

  (1) scalar calibrate() on (pythia-6.9b last, ViT-Ti/16-in21k last), mKNN (PRH mutual_knn) and biased CKA
      (ARH cka_biased);
  (2) calibrate_layers(agg="max"), biased CKA, on grid_layers(33, 5) x grid_layers(12, 5) of pythia-6.9b x
      ViT-Ti/16-in21k, against this run's exact sub-grid maximum. Under the GPU budget rule of plan 11 the check
      shrinks to (1): option --scalar-only (PYT-AXIS).
pre = prh_clip+l2 (primary). Tolerance 1e-6. Tie-rule differences are reported (observed-matrix comparison on the
sub-grid). Order gate first (plan F1(a)). Changes (marked PYT-AXIS): the model (6.9b; test mode: its stand-in),
PYT_THREADS, --scalar-only, outputs kept aside.
usage: launch_pyt.sh pyt_I_vp6 4 --gpu 2 -- venv/bin/python analysis/s12_layermax_pythia/indep/vl6_arh.py
       [--device cuda|cpu] [--scalar-only]   (GPU memory fraction 0.30)
Output: results/S1_2_layermax_pythia/indep/validation_vp6_arh.json
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True   # PYT-AXIS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyt_gate  # noqa: E402  PYT-AXIS
OUT_ROOT = pyt_gate.gate(__file__)   # PYT-AXIS F1(a): exits 3 unless the plan's gates hold (S0b test mode: test root)

_T = os.environ.get("PYT_THREADS", "1")   # PYT-AXIS
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
LM = dict(zip(C.SIZES, C.SLUGS))["6.9b"]   # PYT-AXIS


def rhu(x):
    import math
    return int(math.floor(x + 0.5))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="plan V-P6 (reduced), implementation I")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--scalar-only", action="store_true", help="plan 11 budget rule: only the scalar check")  # PYT-AXIS
    ap.add_argument("--pairs-root", default=None, help="S0b test mode only")
    ap.add_argument("--K-test", type=int, default=None, help="S0b test mode only: first K permutations on both sides")
    ap.add_argument("--lm", default=None, help="S0b test mode only: another LM (e.g. a V-P4 known model)")
    args = ap.parse_args()
    if (args.pairs_root or args.K_test or args.lm) and not pyt_gate.test_mode():
        raise SystemExit("--pairs-root / --K-test / --lm are S0b test options")
    root = args.pairs_root or L.OUT
    KP = args.K_test or L.K_PERM
    lm = args.lm or LM
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

    def mats(lm_):
        z = np.load(os.path.join(root, PRE, "pairs", f"{lm_}__{VIT}.npz"))
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

    out = {"pre": PRE, "K": KP, "tolerance": 1e-6, "device": args.device, "pairs_root": root, "lm": lm,
           "scalar_only": bool(args.scalar_only), "test_mode": bool(pyt_gate.test_mode())}
    XB = feats(VIT)
    XA = feats(lm)
    zA, MA = mats(lm)
    a1, b1 = len(XA) - 1, len(XB) - 1
    keyA = L.perm_key(lm, VIT)
    for name, sim in sims.items():
        t0 = time.time()
        feeder = vi.PermFeeder(L.M, keyA, KP)
        res = vi.arh_calibrate(XA[a1], XB[b1], sim, K=KP, alpha=L.ALPHA, smax=1.0, perm_fn=feeder)
        out[f"scalar_{name}"] = {"pair": [lm, VIT], "layers": [a1, b1], **cmp(res, ours(*MA[name], a=a1, b=b1)),
                                 "secs": time.time() - t0}
        log(f"V-P6 scalar {name} {out[f'scalar_{name}']['absdiff']}")
    if not args.scalar_only:   # PYT-AXIS
        ra = np.unique([rhu(j / 4 * (len(XA) - 1)) for j in range(5)])
        rb = np.unique([rhu(j / 4 * (len(XB) - 1)) for j in range(5)])
        t0 = time.time()
        feeder = vi.PermFeeder(L.M, keyA, KP)
        res = arh_calibrate_layers([XA[a] for a in ra], [XB[b] for b in rb], sims["cka_b"], agg="max", K=KP,
                                   alpha=L.ALPHA, smax=1.0, perm_fn=feeder)
        out["layers_cka_b_grid5x5"] = {"pair": [lm, VIT], "lm_layers": ra.tolist(), "vit_layers": rb.tolist(),
                                       **cmp(res, ours(*MA["cka_b"], ra=ra, rb=rb)), "secs": time.time() - t0}
        log(f"V-P6 layers cka_b grid5x5 {out['layers_cka_b_grid5x5']['absdiff']} {time.time() - t0:.0f}s")
        S = np.array([[sims["cka_b"](XA[a], XB[b]) for b in rb] for a in ra])
        d = np.abs(S - MA["cka_b"][0][np.ix_(ra, rb)])
        out["obsmatrix_cka_b_grid5x5"] = {"maxabs": float(d.max()), "n_diff_gt_1e-6": int((d > 1e-6).sum()),
                                          "n_entries": int(d.size)}
        S = np.array([[sims["mknn"](XA[a], XB[b]) for b in rb] for a in ra])
        d = np.abs(S - MA["mknn"][0][np.ix_(ra, rb)])
        out["obsmatrix_mknn_grid5x5_informational"] = {"maxabs": float(d.max()), "n_diff_gt_1e-6": int((d > 1e-6).sum()),
                                                       "n_entries": int(d.size)}
    diffs = [v for k, dd in out.items() if isinstance(dd, dict) and "absdiff" in dd for v in dd["absdiff"].values()]
    out["max_absdiff"] = float(max(diffs))
    out["pass_1e-6"] = bool(out["max_absdiff"] <= 1e-6)
    L.write_json(os.path.join(L.OUT, "validation_vp6_arh.json"), L.jsonable(out))   # PYT-AXIS
    log(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
