"""Validation V-P3 (hrhv2 identity on 12 layer pairs of the Pythia axis, observed + 3 permutations). Plan section 11.
Uses the same GPU code path as the main run (lm_core.ModelGPU / pair_scores / run_pair). GATED (plan F1(a)).

PYTHIA-AXIS COPY (implementation E, S0b sub-role pyt-code-E): copied from
analysis/s12_layermax_openllama/early/validate.py (sha256 958d1c94...). Changes, marked "PYT-AXIS":
  * the E gate lm_core.gate() is the first action after argument parsing;
  * V-P3: the 12 layer pairs fixed in plan section 11 ((6.9b, DINOv2-g), (6.9b, CLIP-H ft-in12k),
    (70m, ViT-Ti in21k), (70m, MAE-H), each at (first, first), (middle, middle), (last, last) stored layers;
    middle = floor(0.5 (L-1) + 0.5)), at the identity and pi_0, pi_1, pi_499, both pre;
    CKA biased/debiased vs hrhv2 cka_pair <= 1e-6 absolute, mKNN exactly equal;
  * removed: V-O4(a) (fixed-depth entries of open_llama_3b/7b vs S1.2) and the S1.2-key single-entry calibration
    check, which have no Pythia counterpart (no Pythia model is in S1.2; the known-pair check of this axis is V-P4,
    vl4_known_pyt.py);
  * only the models needed are built; --vits (debug) restricts V-P3 to the given ViTs;
  * output <RES>/validation/VP3.json.

Run (step S6, after precompute.py; GPU):
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_pythia/launch_pyt.sh pyt_E_validate 2 --gpu 2 -- \\
      venv/bin/python analysis/s12_layermax_pythia/early/validate.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import lm_core as LC  # noqa: E402  (constants and functions only; reads no data at import)
import run_layermax as RL  # noqa: E402
from hrhv2 import structural as st, geometric as geo  # noqa: E402

# V-P3 cases fixed in ANALYSIS_PLAN_PYTHIA section 11  PYT-AXIS
_VL3_PAIRS = [  # PYT-AXIS
    (LC.SLUG["6.9b"], "vit_giant_patch14_dinov2.lvd142m"),  # PYT-AXIS
    (LC.SLUG["6.9b"], "vit_huge_patch14_clip_224.laion2b_ft_in12k"),  # PYT-AXIS
    (LC.SLUG["70m"], "vit_tiny_patch16_224.augreg_in21k"),  # PYT-AXIS
    (LC.SLUG["70m"], "vit_huge_patch14_224.mae"),  # PYT-AXIS
]


def _vl3_cases():
    out = []
    for lm, vit in _VL3_PAIRS:
        LA, LB = LC.n_stored(lm), LC.n_stored(vit)
        for la, lb in ((0, 0), (LC.fixed_layer(LA, 0.5), LC.fixed_layer(LB, 0.5)), (LA - 1, LB - 1)):
            out.append((lm, vit, la, lb))
    return out


VL3_PERM_IDX = (0, 1, 499)   # pi_0, pi_1, pi_499 (plus the identity)


def hrhv2_geom(stem, layer, pre, c):
    X = LC.layer_rows(stem, layer, pre, c)
    return LC.C12.geom(X, pre)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vits", default=None, help="debug: restrict V-P3 to these ViT stems (comma list)")  # PYT-AXIS
    args = ap.parse_args()
    LC.gate("early/validate.py")   # PYT-AXIS: plan F1(a), first action
    LC.setup_torch(threads=2)
    only = {x.strip() for x in args.vits.split(",")} if args.vits else None
    vits = [v for v, _ in LC.VITS if only is None or v in only]
    cases = [c for c in _vl3_cases() if c[1] in vits]   # (lm, vit, LM layer, ViT layer)
    out = {"V2": [], "started_utc": LC.utc(), "names": {"V2": "V-P3"},  # PYT-AXIS
           "vits_subset": sorted(only) if only else None, "device": LC.DEV}  # PYT-AXIS
    t0 = time.time()
    for pre in LC.PRES:
        need = {c[0] for c in cases} | {c[1] for c in cases}   # PYT-AXIS
        models = RL.load_models(pre, need)
        cget = (lambda s: LC.clip_value(s)) if pre == "prh_clip+l2" else (lambda s: None)
        # ---------------- V-P3 (all 12 cases under both pre)  PYT-AXIS
        for lm, vit, la, lb in cases:
            A, B = models[lm], models[vit]
            perms = LC.perms_for(lm, vit, K=500)[list(VL3_PERM_IDX)]   # LLAMA-AXIS
            GA = hrhv2_geom(lm, la, pre, cget(lm))
            GB = hrhv2_geom(vit, lb, pre, cget(vit))
            nA, _ = st.knn_table(GA.D, LC.KNN_K)
            rec = {"pre": pre, "lm": lm, "vit": vit, "layer_A": la, "layer_B": lb, "checks": []}
            for k in range(len(VL3_PERM_IDX) + 1):   # LLAMA-AXIS
                pi = None if k == 0 else perms[k - 1]
                pit = torch.arange(LC.M_ITEMS, device=LC.DEV) if pi is None else torch.from_numpy(pi).to(LC.DEV)
                s = LC.pair_scores(A, B, pit)
                gpu = {"cka_biased": float(s["cka_biased"][la, lb]), "cka_debiased": float(s["cka_debiased"][la, lb]),
                       "mknn": float(s["mknn_count"][la, lb]) / (LC.M_ITEMS * LC.KNN_K)}
                GBp = GB if pi is None else GB.permuted(pi)
                c = geo.cka_pair(GA, GBp)
                nB, _ = st.knn_table(GBp.D, LC.KNN_K)          # recomputed from scratch on the permuted matrix
                ref = {"cka_biased": c["cka_biased"], "cka_debiased": c["cka_debiased"],
                       "mknn": st.knn_overlap(nA, nB, LC.KNN_K)}
                rec["checks"].append({"perm": "identity" if pi is None else f"k={VL3_PERM_IDX[k - 1]}", "gpu": gpu, "hrhv2": ref,
                                      "absdiff": {m: abs(gpu[m] - ref[m]) for m in gpu}})
            rec["max_absdiff_cka"] = max(max(ch["absdiff"]["cka_biased"], ch["absdiff"]["cka_debiased"])
                                         for ch in rec["checks"])
            rec["mknn_exact"] = all(ch["absdiff"]["mknn"] == 0.0 for ch in rec["checks"])
            out["V2"].append(rec)
            print(f"[{time.time() - t0:5.0f}s] V-P3 {pre} {lm} L{la} x {vit} L{lb}: CKA maxabs {rec['max_absdiff_cka']:.2e}, "
                  f"mKNN exact {rec['mknn_exact']}", flush=True)  # PYT-AXIS
        del models
        if LC.DEV == "cuda":  # PYT-AXIS
            torch.cuda.empty_cache()
    v2c = max((r["max_absdiff_cka"] for r in out["V2"]), default=float("nan"))
    out["summary"] = {
        "V2_n_cases": len(out["V2"]), "V2_max_absdiff_cka": v2c, "V2_cka_pass": bool(v2c <= 1e-6),
        "V2_mknn_all_exact": all(r["mknn_exact"] for r in out["V2"]),
    }
    sm = out["summary"]
    sm["VP3_pass"] = bool(sm["V2_n_cases"] == 24 and sm["V2_cka_pass"] and sm["V2_mknn_all_exact"])  # PYT-AXIS
    out["finished_utc"] = LC.utc()
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    name = "VP3.json" if only is None else "VP3.subset.json"  # PYT-AXIS
    with open(os.path.join(LC.RES, "validation", name), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    print("SUMMARY", json.dumps(LC.jsonable(out["summary"])), flush=True)


if __name__ == "__main__":
    main()
