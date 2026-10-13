"""Validation V4 (plan section 4): vendored ARH calibration code vs ours, with
our permutations injected through perm_fn.

  (i)  ARH scalar calibrate() on one layer pair (bloomz-560m L24 x ViT-Ti L11)
  (ii) ARH calibrate_layers(agg="max") on the full 25 x 12 grid of the same pair

Similarities fed to ARH: pinned PRH `mutual_knn` (k=10) and ARH
`estimators.cka_biased` (+1e-8), both on our float64 preprocessed rows on the
GPU (loaded through the verifier's read-only helper module
results/verification/vendor_identity.py, as S1.2's G1' check did).
ARH stores scores in float32, hence the 1e-6 tolerance.

Output: results/S1_2_layermax/validation/V4_arh.json
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VERIF_TORCH_THREADS", "6")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lm_core as LC  # noqa: E402
import run_layermax as RL  # noqa: E402
from hrhv2 import calibrate as cal  # noqa: E402

sys.path.insert(0, os.path.join(LC.PROJ, "results", "verification"))
import vendor_identity as vi  # noqa: E402
from calibrated_similarity.calibration import calibrate_layers as arh_calibrate_layers  # noqa: E402

LM = "bigscience__bloomz-560m"
VIT = "vit_tiny_patch16_224.augreg_in21k"
LA_SCALAR, LB_SCALAR = 24, 11
PRE = "prh_clip+l2"
TOL = 1e-6


def main():
    LC.setup_torch(threads=6)
    clip = json.load(open(os.path.join(LC.CACHE, "clip.json")))
    t0 = time.time()
    # our implementation (same code path as the main run)
    mods = {}
    for stem, side in ((LM, "A"), (VIT, "B")):
        tab = np.load(os.path.join(LC.CACHE, "knn_final", f"{stem}__{PRE}.npy"))
        mods[stem] = LC.ModelGPU(stem, PRE, side, clip[stem]["c"], knn_tables=tab)
    A, B = mods[LM], mods[VIT]
    key = LC.perm_key(LM, VIT)
    R = LC.run_pair(A, B, LC.perms_for(LM, VIT))
    js, _ = RL.summarize_pair(R, A.L, B.L)
    ours_scalar = {}
    for met in ("mknn", "cka_biased"):
        T = R[met][:, LA_SCALAR, LB_SCALAR].cpu().numpy()
        ours_scalar[met] = cal.summarize(float(T[0]), T[1:], LC.ALPHA)
    print(f"[{time.time() - t0:5.0f}s] ours done", flush=True)
    # rows for the vendored code: same float64 preprocessed rows, on the GPU
    XA = [torch.from_numpy(LC.layer_rows(LM, l, PRE, clip[LM]["c"])).to(LC.DEV) for l in range(A.L)]
    XB = [torch.from_numpy(LC.layer_rows(VIT, l, PRE, clip[VIT]["c"])).to(LC.DEV) for l in range(B.L)]
    sims = {"mknn": lambda X, Y: vi.prh_mknn(X, Y, LC.KNN_K),
            "cka_biased": lambda X, Y: vi.arh_est.cka_biased(X, Y)}
    out = {"pair": [LM, VIT], "pre": PRE, "K": LC.K_PERM, "perm_key": key, "tolerance": TOL,
           "scalar": {}, "layers": {}}
    for met, sim in sims.items():
        # (i) scalar calibrate()
        feeder = vi.PermFeeder(LC.M_ITEMS, key, LC.K_PERM)
        sc, p, tau = vi.arh_calibrate(XA[LA_SCALAR], XB[LB_SCALAR], sim, K=LC.K_PERM, alpha=LC.ALPHA, smax=1.0,
                                      perm_fn=feeder)
        o = ours_scalar[met]
        out["scalar"][met] = {"layer_A": LA_SCALAR, "layer_B": LB_SCALAR,
                              "arh": {"s_cal": float(sc), "p_value": float(p), "c_alpha": float(tau)},
                              "ours": {"s_cal": o["s_cal"], "p_value": o["p_value"], "c_alpha": o["c_alpha"],
                                       "s_obs": o["s_obs"]},
                              "absdiff": {"s_cal": abs(float(sc) - o["s_cal"]), "p_value": abs(float(p) - o["p_value"]),
                                          "c_alpha": abs(float(tau) - o["c_alpha"])}}
        print(f"[{time.time() - t0:5.0f}s] scalar {met}: {out['scalar'][met]['absdiff']}", flush=True)
        # (ii) calibrate_layers(agg='max')
        feeder = vi.PermFeeder(LC.M_ITEMS, key, LC.K_PERM)
        Tc, pa, ta = arh_calibrate_layers(XA, XB, sim, agg="max", K=LC.K_PERM, alpha=LC.ALPHA, smax=1.0,
                                          perm_fn=feeder)
        o = js[met]["full"]
        out["layers"][met] = {"n_layer_pairs": A.L * B.L,
                              "arh": {"s_cal": float(Tc), "p_value": float(pa), "c_alpha": float(ta)},
                              "ours": {"s_cal": o["s_cal"], "p_value": o["p_value"], "c_alpha": o["c_alpha"],
                                       "s_obs": o["s_obs"], "argmax": [o["argmax_A"], o["argmax_B"]]},
                              "absdiff": {"s_cal": abs(float(Tc) - o["s_cal"]), "p_value": abs(float(pa) - o["p_value"]),
                                          "c_alpha": abs(float(ta) - o["c_alpha"])}}
        print(f"[{time.time() - t0:5.0f}s] layers {met}: {out['layers'][met]['absdiff']}", flush=True)
    worst = max(v for sec in ("scalar", "layers") for d in out[sec].values() for v in d["absdiff"].values())
    out["max_absdiff"] = worst
    out["pass"] = bool(worst <= TOL)
    out["finished_utc"] = LC.utc()
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    with open(os.path.join(LC.RES, "validation", "V4_arh.json"), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    print("V4", "PASS" if out["pass"] else "FAIL", worst, flush=True)


if __name__ == "__main__":
    main()
