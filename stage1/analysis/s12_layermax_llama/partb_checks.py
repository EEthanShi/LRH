#!/usr/bin/env python
"""S1.2 LLaMA axis, Part B sanity checks and retrodictions (role llama-axis-predict).

RETRODICTION / SANITY ONLY, outcomes known, not tests. Reads
  results/S1_2_layermax_llama/partB/partB_features.json   (features-only Part B numbers)
and, read-only, the frozen 8-LM layer-max outputs (pairs that do NOT involve
llama-13b or llama-30b):
  results/S1_2_layermax/indep/diag_null_profile.json       (PR profiles, entry-null check)
  results/S1_2_layermax/posthoc_null_structure.json        (null-level profiles)
  results/S1_2_layermax/pairs/<pre>/<lm>__<vit>.json       (layer-max c_alpha, null mean)
No llama-13b / llama-30b cross-model score exists or is computed here.

Checks:
  R1  recomputed per-layer PRs vs the stored PR profiles of the 25 S1.2 models
  R2  predicted per-LM null level mean_v Bbar vs posthoc profile max (= max_a mean_v max_b E0[S_ab]),
      and predicted threshold-setting LM layer u* vs the posthoc argmax layer
  R3  on the 136 known pairs: c_alpha - Bbar and E0[max] - Bbar (the P-L2a(i) quantity),
      BLOOMZ c_alpha order vs Bbar order on (ViT, size-pair) comparisons with gap >= delta_b
      (the P-L2a(ii) quantity), Tr_BLOOMZ(c_alpha) vs Tr_BLOOMZ(Bbar)
  R4  mKNN: c_alpha - 10/1023 (zeta of P-L1); debiased CKA c_alpha (baseline 0)
Output: results/S1_2_layermax_llama/partB/partB_retro_checks.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "6"

import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import partb_pr as PB  # noqa: E402  (functions/constants only; main() not run)

RES0 = os.path.join(PB.PROJ, "results", "S1_2_layermax")
OUT = os.path.join(PB.OUTDIR, "partB_retro_checks.json")
LM8 = [s for s, _, _ in PB.LMS][:8]
VITS = [s for s, _, _ in PB.VITS]


def main():
    F = json.load(open(os.path.join(PB.OUTDIR, "partB_features.json")))
    diag = json.load(open(os.path.join(RES0, "indep", "diag_null_profile.json")))
    post = json.load(open(os.path.join(RES0, "posthoc_null_structure.json")))
    out = {"label": "retrodiction / sanity, outcomes known, not tests; no llama-13b/30b cross-model score used",
           "per_pre": {}}
    for pre in PB.PRES:
        Fp = F["per_pre"][pre]
        R = {}
        # R1 PR reproduction
        errs = {}
        for s in LM8 + VITS:
            a = np.array(Fp["lm"][s]["PR"] if s in Fp["lm"] else Fp["vit"][s]["PR"])
            b = np.array(diag[pre]["PR_profile"][s])
            errs[s] = float(np.max(np.abs(a - b) / np.abs(b)))
        R["R1_PR_vs_indep_diag_max_rel_err"] = {"max": max(errs.values()), "per_model": errs,
                                                "bitwise_all": all(v == 0.0 for v in errs.values())}
        R["R1_indep_entry_null_mean_vs_prop5"] = {
            "median_rel_err": diag[pre]["median_rel_err_pred_vs_entry_null_mean"],
            "max_rel_err": diag[pre]["max_rel_err_pred_vs_entry_null_mean"]}
        # R2 null-level profile (biased CKA)
        prof = post["per_pre"][pre]["profile"]["cka_biased"]
        r2 = {}
        for s in LM8:
            pred_level = float(np.mean([Fp["Bbar"][s][v] for v in VITS]))
            r2[s] = {"pred_mean_v_Bbar": pred_level, "posthoc_profile_max": prof[s]["max"],
                     "rel_err": (pred_level - prof[s]["max"]) / prof[s]["max"],
                     "pred_u_star": Fp["lm"][s]["argmax"], "posthoc_argmax_layer": prof[s]["argmax_layer"],
                     "u_star_equal": Fp["lm"][s]["argmax"] == prof[s]["argmax_layer"]}
        R["R2_null_level_profile"] = r2
        # R3 c_alpha vs Bbar on known pairs
        rows = []
        for s in LM8:
            for v in VITS:
                js = json.load(open(os.path.join(RES0, "pairs", pre, f"{s}__{v}.json")))
                fb = js["summaries"]["cka_biased"]["full"]
                rows.append({"lm": s, "vit": v, "c_alpha": fb["c_alpha"], "E0_max": fb["null_mean"],
                             "Bbar": Fp["Bbar"][s][v], "s_obs": fb["s_obs"],
                             "c_alpha_mknn": js["summaries"]["mknn"]["full"]["c_alpha"],
                             "c_alpha_deb": js["summaries"]["cka_debiased"]["full"]["c_alpha"],
                             "E0_max_deb": js["summaries"]["cka_debiased"]["full"]["null_mean"]})
        ca = np.array([r["c_alpha"] for r in rows])
        e0 = np.array([r["E0_max"] for r in rows])
        bb = np.array([r["Bbar"] for r in rows])
        R["R3_calpha_minus_Bbar_136"] = {
            "median_abs": float(np.median(np.abs(ca - bb))), "mean": float(np.mean(ca - bb)),
            "min": float(np.min(ca - bb)), "max": float(np.max(ca - bb)),
            "E0max_minus_Bbar_mean": float(np.mean(e0 - bb)), "E0max_minus_Bbar_maxabs": float(np.max(np.abs(e0 - bb))),
            "calpha_minus_E0max_mean": float(np.mean(ca - e0)),
            "spearman_calpha_Bbar": float(np.corrcoef(PB.rankdata(ca), PB.rankdata(bb))[0, 1]),
            "pearson_calpha_Bbar": float(np.corrcoef(ca, bb)[0, 1]),
            "by_lm_mean": {s: {"c_alpha": float(np.mean([r["c_alpha"] for r in rows if r["lm"] == s])),
                               "Bbar": float(np.mean([r["Bbar"] for r in rows if r["lm"] == s])),
                               "E0_max": float(np.mean([r["E0_max"] for r in rows if r["lm"] == s]))} for s in LM8}}
        # BLOOMZ order agreement and trends
        idx = {(r["lm"], r["vit"]): r for r in rows}
        agree = tot = 0
        agree_all = tot_all = 0
        for x, y in itertools.combinations(PB.BLOOMZ, 2):
            for v in VITS:
                dB = idx[(y, v)]["Bbar"] - idx[(x, v)]["Bbar"]
                dc = idx[(y, v)]["c_alpha"] - idx[(x, v)]["c_alpha"]
                tot_all += 1
                agree_all += int(np.sign(dB) == np.sign(dc))
                if abs(dB) >= PB.DELTA_B:
                    tot += 1
                    agree += int(np.sign(dB) == np.sign(dc))
        Gc = np.array([[idx[(s, v)]["c_alpha"] for v in VITS] for s in PB.BLOOMZ])
        Gb = np.array([[idx[(s, v)]["Bbar"] for v in VITS] for s in PB.BLOOMZ])
        sizes = [F["meta"]["inputs"][s]["num_params"] for s in PB.BLOOMZ]
        R["R3_bloomz"] = {"order_agreement_gap_ge_delta_b": {"agree": agree, "n": tot,
                                                            "frac": (agree / tot) if tot else None},
                          "order_agreement_all": {"agree": agree_all, "n": tot_all, "frac": agree_all / tot_all},
                          "Tr_BLOOMZ_calpha": PB.family_trend(Gc, sizes)["Tr"],
                          "Tr_BLOOMZ_Bbar": PB.family_trend(Gb, sizes)["Tr"]}
        # LLaMA-7B only (known)
        l7 = [r for r in rows if r["lm"] == "huggyllama__llama-7b"]
        R["R3_llama7b"] = {"mean_c_alpha": float(np.mean([r["c_alpha"] for r in l7])),
                           "mean_Bbar": float(np.mean([r["Bbar"] for r in l7])),
                           "median_abs_calpha_minus_Bbar": float(np.median([abs(r["c_alpha"] - r["Bbar"]) for r in l7]))}
        # R4 mKNN zeta, debiased
        cm = np.array([r["c_alpha_mknn"] for r in rows])
        cd = np.array([r["c_alpha_deb"] for r in rows])
        R["R4_mknn"] = {"baseline": PB.KNN_K / (PB.M - 1), "c_alpha_mean": float(cm.mean()),
                        "c_alpha_min": float(cm.min()), "c_alpha_max": float(cm.max()),
                        "zeta_mean": float(np.mean(cm - PB.KNN_K / (PB.M - 1))),
                        "zeta_max": float(np.max(np.abs(cm - PB.KNN_K / (PB.M - 1))))}
        R["R4_debiased"] = {"baseline": 0.0, "c_alpha_mean": float(cd.mean()), "c_alpha_max": float(cd.max()),
                            "E0max_mean": float(np.mean([r["E0_max_deb"] for r in rows]))}
        out["per_pre"][pre] = R
        print(f"== {pre}: R1 PR max rel err {R['R1_PR_vs_indep_diag_max_rel_err']['max']:.2e} "
              f"(bitwise {R['R1_PR_vs_indep_diag_max_rel_err']['bitwise_all']}) | "
              f"R2 u* equal {sum(d['u_star_equal'] for d in r2.values())}/8, max|rel err| "
              f"{max(abs(d['rel_err']) for d in r2.values()):.2e} | R3 median|c-B| {R['R3_calpha_minus_Bbar_136']['median_abs']:.4f} "
              f"mean c-B {R['R3_calpha_minus_Bbar_136']['mean']:+.4f} | BLOOMZ agree {agree}/{tot} "
              f"Tr c {R['R3_bloomz']['Tr_BLOOMZ_calpha']:+.2f} Tr B {R['R3_bloomz']['Tr_BLOOMZ_Bbar']:+.2f} | "
              f"7b c {R['R3_llama7b']['mean_c_alpha']:.4f} B {R['R3_llama7b']['mean_Bbar']:.4f} | "
              f"mKNN zeta mean {R['R4_mknn']['zeta_mean']:.5f} max {R['R4_mknn']['zeta_max']:.5f}", flush=True)
    with open(OUT, "w") as f:
        json.dump(PB.jsonable(out), f, indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
