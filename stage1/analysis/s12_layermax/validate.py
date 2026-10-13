"""Validation V2 (hrhv2 identity on layer pairs, observed + 2 permutations) and
V3 (fixed-depth entries vs S1.2 stored values; S1.2-key single-entry
calibration on 3 pairs). Plan section 4. Uses the same GPU code path as the
main run (lm_core.ModelGPU / pair_scores / run_pair).

Output: results/S1_2_layermax/validation/V2_V3.json
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lm_core as LC  # noqa: E402
import run_layermax as RL  # noqa: E402
from hrhv2 import calibrate as cal, structural as st, geometric as geo  # noqa: E402

V2_CASES = [  # (lm, vit, LM layer, ViT layer)
    ("bigscience__bloomz-560m", "vit_tiny_patch16_224.augreg_in21k", 0, 0),
    ("bigscience__bloomz-560m", "vit_tiny_patch16_224.augreg_in21k", 24, 11),
    ("bigscience__bloomz-1b7", "vit_large_patch14_dinov2.lvd142m", 12, 11),
    ("bigscience__bloomz-7b1", "vit_huge_patch14_clip_224.laion2b", 30, 31),
    ("huggyllama__llama-7b", "vit_giant_patch14_dinov2.lvd142m", 0, 39),
    ("huggyllama__llama-7b", "vit_giant_patch14_dinov2.lvd142m", 16, 20),
    ("huggyllama__llama-7b", "vit_huge_patch14_224.mae", 32, 31),
    ("openlm-research__open_llama_7b", "vit_large_patch16_224.augreg_in21k", 5, 3),
    ("openlm-research__open_llama_3b", "vit_base_patch16_clip_224.laion2b_ft_in12k", 26, 11),
    ("bigscience__bloomz-3b", "vit_small_patch14_dinov2.lvd142m", 1, 5),
    ("bigscience__bloomz-1b1", "vit_large_patch14_clip_224.laion2b", 24, 0),
    ("openlm-research__open_llama_7b", "vit_giant_patch14_dinov2.lvd142m", 32, 0),
]
V3_CALIB_PAIRS = [
    ("bigscience__bloomz-560m", "vit_base_patch16_clip_224.laion2b", 0.5),
    ("openlm-research__open_llama_7b", "vit_giant_patch14_dinov2.lvd142m", 1.0),
    ("huggyllama__llama-7b", "vit_huge_patch14_224.mae", 0.5),
]


def hrhv2_geom(stem, layer, pre, c):
    X = LC.layer_rows(stem, layer, pre, c)
    return LC.C12.geom(X, pre)


def main():
    LC.setup_torch(threads=6)
    clip = json.load(open(os.path.join(LC.CACHE, "clip.json")))
    out = {"V2": [], "V3_obs": {}, "V3_calib": [], "started_utc": LC.utc()}
    t0 = time.time()
    for pre in LC.PRES:
        models = RL.load_models(pre)
        cget = (lambda s: clip[s]["c"]) if pre == "prh_clip+l2" else (lambda s: None)
        # ---------------- V2
        cases = V2_CASES if pre == "prh_clip+l2" else V2_CASES[::3]
        for lm, vit, la, lb in cases:
            A, B = models[lm], models[vit]
            perms = LC.perms_for(lm, vit, K=2)
            GA = hrhv2_geom(lm, la, pre, cget(lm))
            GB = hrhv2_geom(vit, lb, pre, cget(vit))
            nA, _ = st.knn_table(GA.D, LC.KNN_K)
            rec = {"pre": pre, "lm": lm, "vit": vit, "layer_A": la, "layer_B": lb, "checks": []}
            for k in range(3):
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
                rec["checks"].append({"perm": "identity" if pi is None else f"k={k - 1}", "gpu": gpu, "hrhv2": ref,
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
        for lm, _ in LC.LMS:
            for vit, _ in LC.VITS:
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
            for lm, vit, r in V3_CALIB_PAIRS:
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
    v2c = max(r["max_absdiff_cka"] for r in out["V2"])
    out["summary"] = {
        "V2_n_cases": len(out["V2"]), "V2_max_absdiff_cka": v2c, "V2_cka_pass": bool(v2c <= 1e-6),
        "V2_mknn_all_exact": all(r["mknn_exact"] for r in out["V2"]),
        "V3_obs_pass": all(v["pass"] for v in out["V3_obs"].values()),
        "V3_calib_max_absdiff": max(r["max_absdiff"] for r in out["V3_calib"]),
        "V3_calib_pass": bool(max(r["max_absdiff"] for r in out["V3_calib"]) <= 1e-12),
    }
    out["finished_utc"] = LC.utc()
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    with open(os.path.join(LC.RES, "validation", "V2_V3.json"), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    print("SUMMARY", json.dumps(LC.jsonable(out["summary"])), flush=True)


if __name__ == "__main__":
    main()
