"""Validation V-O3 (hrhv2 identity on 12 layer pairs of open_llama_13b, observed + 3 permutations) and
V-O4(a) (fixed-depth entries of open_llama_3b/7b vs the S1.2 stored values), plus the S1.2-key single-entry
calibration check (open_llama_7b x MAE-H, r = 0.5). Plan section 11. Uses the same GPU code path as the
main run (lm_core.ModelGPU / pair_scores / run_pair). GATED (plan F1(a)).

OPENLLAMA-AXIS COPY (implementation E, S0b sub-role E of oll-axis-extract): copied from
analysis/s12_layermax_llama/early/validate.py (sha256 0ad3fb86...). Changes, marked "OL-AXIS":
  * the Part B gate guard.require_partb_frozen(lm_core.RES) is the first action after argument parsing;
  * V-O3: the 12 layer pairs fixed in plan section 11 ((13B, DINOv2-g), (13B, CLIP-H ft-in12k),
    (13B, ViT-Ti in21k), (13B, MAE-H), each at (first, first), (middle, middle), (last, last) stored
    layers; middle = floor(0.5 (L-1) + 0.5)), at the identity and pi_0, pi_1, pi_499, both pre;
    CKA biased/debiased vs hrhv2 cka_pair <= 1e-6 absolute, mKNN exactly equal;
  * V-O4(a): fixed-depth entries of open_llama_3b and open_llama_7b (2 LMs x 17 ViTs x 2 depths x 2 pre)
    vs the stored S1.2 values: mKNN exact, CKA <= 1e-10;
  * only the models needed are built; --vits (debug) restricts both checks to the given ViTs;
  * output <RES>/validation/VO3_VO4a.json.

Run (step S6, after precompute.py; GPU):
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_openllama/launch_ol.sh ol_E_validate 2 --gpu 4 -- \
      venv/bin/python analysis/s12_layermax_openllama/early/validate.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lm_core as LC  # noqa: E402  (constants and functions only; reads no data at import)
import guard  # noqa: E402  (analysis/s12_layermax_openllama/guard.py, on sys.path via lm_core)
import run_layermax as RL  # noqa: E402
from hrhv2 import calibrate as cal, structural as st, geometric as geo  # noqa: E402

# V-O3 cases fixed in ANALYSIS_PLAN_OPENLLAMA section 11
_VL3_PAIRS = [
    (LC.SLUG_13B, "vit_giant_patch14_dinov2.lvd142m"),
    (LC.SLUG_13B, "vit_huge_patch14_clip_224.laion2b_ft_in12k"),
    (LC.SLUG_13B, "vit_tiny_patch16_224.augreg_in21k"),
    (LC.SLUG_13B, "vit_huge_patch14_224.mae"),
]
VO4A_LMS = (LC.SLUG_3B, LC.SLUG_7B)   # the S1.2-processed LMs of this axis


def _vl3_cases():
    out = []
    for lm, vit in _VL3_PAIRS:
        LA, LB = LC.n_stored(lm), LC.n_stored(vit)
        for la, lb in ((0, 0), (LC.fixed_layer(LA, 0.5), LC.fixed_layer(LB, 0.5)), (LA - 1, LB - 1)):
            out.append((lm, vit, la, lb))
    return out


VL3_PERM_IDX = (0, 1, 499)   # pi_0, pi_1, pi_499 (plus the identity)
V3_CALIB_PAIRS = [
    (LC.SLUG_7B, "vit_huge_patch14_224.mae", 0.5),
]


def hrhv2_geom(stem, layer, pre, c):
    X = LC.layer_rows(stem, layer, pre, c)
    return LC.C12.geom(X, pre)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vits", default=None, help="debug: restrict V-O3 and V-O4(a) to these ViT stems (comma list)")
    args = ap.parse_args()
    guard.require_partb_frozen(LC.RES, script="early/validate.py")   # plan F1(a): first action
    LC.setup_torch(threads=2)
    only = {x.strip() for x in args.vits.split(",")} if args.vits else None
    vits = [v for v, _ in LC.VITS if only is None or v in only]
    cases = [c for c in _vl3_cases() if c[1] in vits]   # (lm, vit, LM layer, ViT layer)
    calib_pairs = [c for c in V3_CALIB_PAIRS if c[1] in vits]
    out = {"V2": [], "V3_obs": {}, "V3_calib": [], "started_utc": LC.utc(),
           "names": {"V2": "V-O3", "V3_obs": "V-O4(a)", "V3_calib": "S1.2-key single-entry calibration"},
           "vits_subset": sorted(only) if only else None}
    t0 = time.time()
    for pre in LC.PRES:
        need = {LC.SLUG_13B, *VO4A_LMS, *vits}
        models = RL.load_models(pre, need)
        cget = (lambda s: LC.clip_value(s)) if pre == "prh_clip+l2" else (lambda s: None)
        # ---------------- V-O3 (all 12 cases under both pre)
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
            print(f"[{time.time() - t0:5.0f}s] V2 {pre} {lm} L{la} x {vit} L{lb}: CKA maxabs {rec['max_absdiff_cka']:.2e}, "
                  f"mKNN exact {rec['mknn_exact']}", flush=True)
        # ---------------- V3 observed entries vs S1.2 stored values (all 136 pairs, both depths)
        worst = {"mknn": 0.0, "cka_biased": 0.0, "cka_debiased": 0.0}
        n_cmp = 0
        ident = torch.arange(LC.M_ITEMS, device=LC.DEV)
        for lm in VO4A_LMS:   # V-O4(a): open_llama_3b and open_llama_7b
            for vit in vits:
                s = LC.pair_scores(models[lm], models[vit], ident)
                for r in (0.5, 1.0):
                    rec = json.load(open(os.path.join(LC.PROJ, "results", "S1_2", "main", "pairs",
                                                      f"{lm}__{vit}__r{r}.json")))["per_pre"][pre]
                    la, lb = rec["meta_A"]["layer"], rec["meta_B"]["layer"]
                    assert la == LC.fixed_layer(models[lm].L, r) and lb == LC.fixed_layer(models[vit].L, r)
                    mine = {"mknn": float(s["mknn_count"][la, lb]) / (LC.M_ITEMS * LC.KNN_K),
                            "cka_biased": float(s["cka_biased"][la, lb]),
                            "cka_debiased": float(s["cka_debiased"][la, lb])}
                    theirs = {"mknn": rec["ladder"]["s_N_k10"], "cka_biased": rec["ladder"]["cka_biased"],
                              "cka_debiased": rec["ladder"]["cka_debiased"]}
                    for m in worst:
                        worst[m] = max(worst[m], abs(mine[m] - theirs[m]))
                    n_cmp += 1
        out["V3_obs"][pre] = {"n_pair_depths": n_cmp, "max_absdiff": worst,
                              "pass": bool(worst["mknn"] == 0.0 and worst["cka_biased"] <= 1e-10
                                           and worst["cka_debiased"] <= 1e-10)}
        print(f"[{time.time() - t0:5.0f}s] V3 obs {pre}: {out['V3_obs'][pre]}", flush=True)
        # ---------------- V3 calibration with S1.2's own permutation keys (primary pre only)
        if pre == "prh_clip+l2":
            for lm, vit, r in calib_pairs:
                rec = json.load(open(os.path.join(LC.PROJ, "results", "S1_2", "main", "pairs",
                                                  f"{lm}__{vit}__r{r}.json")))
                key, K = rec["perm_key"], rec["K"]
                d = rec["per_pre"][pre]
                la, lb = d["meta_A"]["layer"], d["meta_B"]["layer"]
                R = LC.run_pair(models[lm], models[vit], LC.perms_for(lm, vit, K=K, key=key))
                row = {"lm": lm, "vit": vit, "r": r, "key": key, "metrics": {}}
                for met, name in (("mknn", "s_N_k10"), ("cka_biased", "cka_biased"), ("cka_debiased", "cka_debiased")):
                    T = R[met][:, la, lb].cpu().numpy()
                    s = cal.summarize(float(T[0]), T[1:], LC.ALPHA)
                    ref = d["calib"][name]
                    row["metrics"][met] = {k: {"mine": s[k], "s12": ref[k], "absdiff": abs(s[k] - ref[k])}
                                           for k in ("s_obs", "c_alpha", "s_cal", "p_value", "null_mean")}
                row["max_absdiff"] = max(v["absdiff"] for mm in row["metrics"].values() for v in mm.values())
                out["V3_calib"].append(row)
                print(f"[{time.time() - t0:5.0f}s] V3 calib {lm} x {vit} r={r}: max absdiff {row['max_absdiff']:.2e}",
                      flush=True)
                del R
        del models
        torch.cuda.empty_cache()
    v2c = max((r["max_absdiff_cka"] for r in out["V2"]), default=float("nan"))
    v3c = max((r["max_absdiff"] for r in out["V3_calib"]), default=0.0)
    out["summary"] = {
        "V2_n_cases": len(out["V2"]), "V2_max_absdiff_cka": v2c, "V2_cka_pass": bool(v2c <= 1e-6),
        "V2_mknn_all_exact": all(r["mknn_exact"] for r in out["V2"]),
        "V3_obs_pass": all(v["pass"] for v in out["V3_obs"].values()),
        "V3_calib_max_absdiff": v3c,
        "V3_calib_pass": bool(v3c <= 1e-12),
    }
    sm = out["summary"]
    sm["VO3_pass"] = bool(sm["V2_n_cases"] == 24 and sm["V2_cka_pass"] and sm["V2_mknn_all_exact"])
    sm["VO4a_pass"] = bool(sm["V3_obs_pass"] and all(v["n_pair_depths"] == 68 for v in out["V3_obs"].values()))
    out["finished_utc"] = LC.utc()
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    name = "VO3_VO4a.json" if only is None else "VO3_VO4a.subset.json"
    with open(os.path.join(LC.RES, "validation", name), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    print("SUMMARY", json.dumps(LC.jsonable(out["summary"])), flush=True)


if __name__ == "__main__":
    main()
