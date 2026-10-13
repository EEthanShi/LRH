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

LLAMA-AXIS COPY (implementation E, role llama-axis-run-early) = plan V-L6:
copied from analysis/s12_layermax/v4_arh.py (sha256 74fdbd33...). Changes,
marked "LLAMA-AXIS": the cases fixed in ANALYSIS_PLAN_LLAMA section 4 V-L6:
  (i)   scalar calibrate() on (llama-30b last, ViT-Ti/16-in21k last), mKNN (PRH
        mutual_knn) and biased CKA (ARH cka_biased);
  (ii)  calibrate_layers(agg="max"), biased CKA, full llama-13b x ViT-Ti grid (41 x 12);
  (iii) calibrate_layers(agg="max"), mKNN, llama-30b LM layers grid_layers(61, 9) x all
        12 ViT-Ti layers, against our exact sub-grid maximum (same pi).
Ours = the main-run code path (lm_core.run_pair + run_layermax.summarize_pair /
subset_summary), recomputed here (no main-run output is read). Output
results/S1_2_layermax_llama/early/validation/V4_arh.json (lm_core.RES).
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

# LLAMA-AXIS: V-L6 cases
LM_S = "huggyllama__llama-30b"       # (i) scalar, last x last; (iii) mKNN sub-grid
LM_L = "huggyllama__llama-13b"       # (ii) CKA full grid
VIT = "vit_tiny_patch16_224.augreg_in21k"
PRE = "prh_clip+l2"
TOL = 1e-6


def _rows(stem, clip, L):
    return [torch.from_numpy(LC.layer_rows(stem, l, PRE, clip[stem]["c"])).to(LC.DEV) for l in range(L)]


def _cmp(arh, ours):
    return {k: abs(float(arh[k]) - float(ours[k])) for k in ("s_cal", "p_value", "c_alpha")}


def main():
    # LLAMA-AXIS: rewritten case list (plan V-L6); the comparison logic is the original one
    LC.setup_torch(threads=6)
    clip = json.load(open(os.path.join(LC.CACHE, "clip.json")))
    t0 = time.time()
    mods = {}
    for stem, side in ((LM_S, "A"), (LM_L, "A"), (VIT, "B")):
        tab = np.load(os.path.join(LC.CACHE, "knn_final", f"{stem}__{PRE}.npy"))
        mods[stem] = LC.ModelGPU(stem, PRE, side, clip[stem]["c"], knn_tables=tab)
    B = mods[VIT]
    sims = {"mknn": lambda X, Y: vi.prh_mknn(X, Y, LC.KNN_K),
            "cka_biased": lambda X, Y: vi.arh_est.cka_biased(X, Y)}
    out = {"pre": PRE, "K": LC.K_PERM, "tolerance": TOL, "scalar": {}, "layers": {}}
    XB = _rows(VIT, clip, B.L)
    # ---------------- (i) scalar calibrate() on (llama-30b last, ViT-Ti last)
    A = mods[LM_S]
    key = LC.perm_key(LM_S, VIT)
    R = LC.run_pair(A, B, LC.perms_for(LM_S, VIT))
    la, lb = A.L - 1, B.L - 1
    XA = _rows(LM_S, clip, A.L)
    for met, sim in sims.items():
        T = R[met][:, la, lb].cpu().numpy()
        o = cal.summarize(float(T[0]), T[1:], LC.ALPHA)
        feeder = vi.PermFeeder(LC.M_ITEMS, key, LC.K_PERM)
        sc, p, tau = vi.arh_calibrate(XA[la], XB[lb], sim, K=LC.K_PERM, alpha=LC.ALPHA, smax=1.0, perm_fn=feeder)
        arh = {"s_cal": float(sc), "p_value": float(p), "c_alpha": float(tau)}
        out["scalar"][met] = {"pair": [LM_S, VIT], "perm_key": key, "layer_A": la, "layer_B": lb, "arh": arh,
                              "ours": {k: o[k] for k in ("s_cal", "p_value", "c_alpha", "s_obs")},
                              "absdiff": _cmp(arh, o)}
        print(f"[{time.time() - t0:5.0f}s] (i) scalar {met}: {out['scalar'][met]['absdiff']}", flush=True)
    # ---------------- (iii) mKNN sub-grid: llama-30b grid_layers(61, 9) x all ViT-Ti layers
    ia, ib = LC.grid_layers(A.L, 9), list(range(B.L))
    E = RL.entry_stats(R["mknn"])
    o, _ = RL.subset_summary(R["mknn"], ia, ib, E)
    del R
    feeder = vi.PermFeeder(LC.M_ITEMS, key, LC.K_PERM)
    Tc, pa, ta = arh_calibrate_layers([XA[a] for a in ia], XB, sims["mknn"], agg="max", K=LC.K_PERM,
                                      alpha=LC.ALPHA, smax=1.0, perm_fn=feeder)
    arh = {"s_cal": float(Tc), "p_value": float(pa), "c_alpha": float(ta)}
    out["layers"]["mknn_subgrid_llama30b"] = {
        "pair": [LM_S, VIT], "perm_key": key, "layers_A": ia, "layers_B": ib, "n_layer_pairs": len(ia) * len(ib),
        "arh": arh, "ours": {"s_cal": o["s_cal"], "p_value": o["p_value"], "c_alpha": o["c_alpha"], "s_obs": o["s_obs"],
                             "argmax": [o["argmax_A"], o["argmax_B"]]}, "absdiff": _cmp(arh, o)}
    print(f"[{time.time() - t0:5.0f}s] (iii) mKNN sub-grid: {out['layers']['mknn_subgrid_llama30b']['absdiff']}", flush=True)
    del XA
    torch.cuda.empty_cache()
    # ---------------- (ii) biased CKA full grid: llama-13b x ViT-Ti (41 x 12)
    A = mods[LM_L]
    key = LC.perm_key(LM_L, VIT)
    R = LC.run_pair(A, B, LC.perms_for(LM_L, VIT))
    js, _ = RL.summarize_pair(R, A.L, B.L)
    del R
    XA = _rows(LM_L, clip, A.L)
    feeder = vi.PermFeeder(LC.M_ITEMS, key, LC.K_PERM)
    Tc, pa, ta = arh_calibrate_layers(XA, XB, sims["cka_biased"], agg="max", K=LC.K_PERM, alpha=LC.ALPHA, smax=1.0,
                                      perm_fn=feeder)
    o = js["cka_biased"]["full"]
    arh = {"s_cal": float(Tc), "p_value": float(pa), "c_alpha": float(ta)}
    out["layers"]["cka_biased_full_llama13b"] = {
        "pair": [LM_L, VIT], "perm_key": key, "n_layer_pairs": A.L * B.L, "arh": arh,
        "ours": {"s_cal": o["s_cal"], "p_value": o["p_value"], "c_alpha": o["c_alpha"], "s_obs": o["s_obs"],
                 "argmax": [o["argmax_A"], o["argmax_B"]]}, "absdiff": _cmp(arh, o)}
    print(f"[{time.time() - t0:5.0f}s] (ii) CKA full grid: {out['layers']['cka_biased_full_llama13b']['absdiff']}", flush=True)
    worst = max(v for sec in ("scalar", "layers") for d in out[sec].values() for v in d["absdiff"].values())
    out["max_absdiff"] = worst
    out["pass"] = bool(worst <= TOL)
    out["elapsed_s"] = time.time() - t0
    out["finished_utc"] = LC.utc()
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    with open(os.path.join(LC.RES, "validation", "V4_arh.json"), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    print("V-L6", "PASS" if out["pass"] else "FAIL", worst, flush=True)


if __name__ == "__main__":
    main()
