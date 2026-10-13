#!/usr/bin/env python
"""S1.2 OpenLLaMA axis, Part B sanity checks and retrodictions (role oll-axis-predict, step S5, after partb_ol.py).  # OL-AXIS
Copy of analysis/s12_layermax_llama/partb_checks.py (sha256 b2e55a169532e82330378af30b16bf4931e321837405d57064653346c09397dd);  # OL-AXIS
every changed or added line carries the marker OL-AXIS; diff: partb_checks_ol.py.diff.

RETRODICTION / SANITY ONLY, outcomes known, not tests. NOT GATED: it opens no score involving open_llama_13b.  # OL-AXIS
Reads  # OL-AXIS
  results/S1_2_layermax_openllama/partB/partB_features_ol.json  (features-only Part B numbers)  # OL-AXIS
and, read-only, the frozen layer-max outputs of KNOWN pairs only:  # OL-AXIS
  results/S1_2_layermax/indep/diag_null_profile.json       (PR profiles, entry-null check)
  results/S1_2_layermax/posthoc_null_structure.json        (null-level profiles)
  results/S1_2_layermax/pairs/<pre>/<lm>__<vit>.json       (8-LM layer-max c_alpha, null mean, fixed depths)  # OL-AXIS
  results/S1_2_layermax_llama/early/pairs/<pre>/<lm>__<vit>.json  (llama-13b/30b layer-max c_alpha)  # OL-AXIS
No open_llama_13b cross-model score exists or is computed here.  # OL-AXIS

Checks:
  R1  recomputed per-layer PRs vs the stored PR profiles of the 25 S1.2 models
  R2  predicted per-LM null level mean_v Bbar vs posthoc profile max, and u* vs the posthoc argmax layer  # OL-AXIS
  R3  on the 136 known 8-LM pairs (+ the 34 llama-13b/30b pairs): c_alpha - Bbar, E0[max] - Bbar, BLOOMZ c_alpha  # OL-AXIS
      order vs Bbar order, Tr_BLOOMZ(c_alpha) vs Tr_BLOOMZ(Bbar)  # OL-AXIS
  R4  mKNN: c_alpha - 10/1023 (zeta of P-O1); debiased CKA c_alpha (baseline 0)  # OL-AXIS
  R5  OpenLLaMA pilot pairs (3b, 7b): offsets c_alpha - Bbar against the band, null mean - Bbar, fixed-depth  # OL-AXIS
      offsets c_fd - B_fd (plan B4: 0.0020-0.0025), LLaMA offsets inside the band  # OL-AXIS
  R6  retro-forecast of the 17 pilot comparisons 3B < 7B with the P-O2b rule (7B in the role of the new model,  # OL-AXIS
      kappa from 3B's T, c_alpha and Bbar_7B + o), scored against the known s_cal order (sanity of the machinery)  # OL-AXIS
Output: results/S1_2_layermax_openllama/partB/partB_retro_checks_ol.json  # OL-AXIS
# OL-AXIS
RUN (step S5, after partb_ol.py; CPU, seconds):  # OL-AXIS
  analysis/s12_layermax_openllama/launch_ol.sh partb_checks_ol 2 -- venv/bin/python analysis/s12_layermax_openllama/partb_checks_ol.py  # OL-AXIS
S0b test mode (OL_S0B_TEST=1): reads/writes ~/oll_s0b_scratch/PB/partB/ (or --partb-dir, test only).  # OL-AXIS
"""
from __future__ import annotations

import os

_TH = os.environ.get("OL_THREADS", "1")  # OL-AXIS: thread count from launch_ol.sh (plan F5)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _TH  # OL-AXIS

import argparse  # noqa: E402  # OL-AXIS
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import partb_ol as PB  # noqa: E402  # OL-AXIS (functions/constants only; main() not run)

RES0 = os.path.join(PB.PROJ, "results", "S1_2_layermax")
LM8 = [s for s, _, _ in PB.KNOWN_LMS][:8]  # OL-AXIS
VITS = [s for s, _, _ in PB.VITS]
OL37 = [PB.OL3, PB.OL7]  # OL-AXIS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # OL-AXIS
    ap.add_argument("--partb-dir", default=None, help="S0b test mode only")  # OL-AXIS
    args = ap.parse_args()  # OL-AXIS
    test = PB.guard.test_mode()  # OL-AXIS
    if args.partb_dir and not test:  # OL-AXIS
        raise SystemExit("--partb-dir is an S0b test option")  # OL-AXIS
    pdir = args.partb_dir or (os.path.join(PB.SCRATCH_PB, "partB") if test else PB.OUTDIR_REAL)  # OL-AXIS
    OUT = os.path.join(pdir, "partB_retro_checks_ol.json")  # OL-AXIS
    F = json.load(open(os.path.join(pdir, "partB_features_ol.json")))  # OL-AXIS
    diag = json.load(open(os.path.join(RES0, "indep", "diag_null_profile.json")))
    post = json.load(open(os.path.join(RES0, "posthoc_null_structure.json")))
    out = {"label": "retrodiction / sanity, outcomes known, not tests; no open_llama_13b cross-model score used",  # OL-AXIS
           "partB_features_ol_sha256": PB.sha256_file(os.path.join(pdir, "partB_features_ol.json")),  # OL-AXIS
           "test_mode": test, "per_pre": {}}  # OL-AXIS
    for pre in PB.PRES:
        Fp = F["per_pre"][pre]
        R = {}
        # R1 PR reproduction
        errs = {}
        for s in LM8 + VITS:
            a = np.array(Fp["PR"][s])  # OL-AXIS
            b = np.array(diag[pre]["PR_profile"][s])
            errs[s] = float(np.max(np.abs(a - b) / np.abs(b)))
        R["R1_PR_vs_indep_diag_max_rel_err"] = {"max": max(errs.values()), "per_model": errs,
                                                "bitwise_all": all(v == 0.0 for v in errs.values())}
        R["R1_indep_entry_null_mean_vs_prop5"] = {
            "median_rel_err": diag[pre]["median_rel_err_pred_vs_entry_null_mean"],
            "max_rel_err": diag[pre]["max_rel_err_pred_vs_entry_null_mean"]}
        R["R1_recheck_27_vs_llama_partB"] = F["meta"]["recheck_27"]  # OL-AXIS
        # R2 null-level profile (biased CKA)
        prof = post["per_pre"][pre]["profile"]["cka_biased"]
        r2 = {}
        for s in LM8:
            pred_level = float(np.mean([Fp["Bbar"][s][v] for v in VITS]))
            r2[s] = {"pred_mean_v_Bbar": pred_level, "posthoc_profile_max": prof[s]["max"],
                     "rel_err": (pred_level - prof[s]["max"]) / prof[s]["max"],
                     "pred_u_star": Fp["u_star"][s], "posthoc_argmax_layer": prof[s]["argmax_layer"],  # OL-AXIS
                     "u_star_equal": Fp["u_star"][s] == prof[s]["argmax_layer"]}  # OL-AXIS
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
                             "E0_max_deb": js["summaries"]["cka_debiased"]["full"]["null_mean"],  # OL-AXIS
                             "s_cal": fb["s_cal"],  # OL-AXIS
                             "fd": {r: {k: js["summaries"]["cka_biased"][f"fixed_r{r}"][k] for k in ("c_alpha", "s_obs")}  # OL-AXIS
                                    for r in ("0.5", "1.0")}})  # OL-AXIS
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
        sizes = [F["num_params"][s] for s in PB.BLOOMZ]  # OL-AXIS
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
        # llama-13b/30b (LLaMA axis, known): c_alpha - Bbar  # OL-AXIS
        offL = []  # OL-AXIS
        for s in PB.LLAMA:  # OL-AXIS
            for v in VITS:  # OL-AXIS
                js = json.load(open(os.path.join(PB.LLAMA_E, pre, f"{s}__{v}.json")))  # OL-AXIS
                offL.append(js["summaries"]["cka_biased"]["full"]["c_alpha"] - Fp["Bbar"][s][v])  # OL-AXIS
        offL = np.array(offL)  # OL-AXIS
        bd = Fp["band"]  # OL-AXIS
        R["R3_llama_axis_51"] = {"median": float(np.median(offL)), "min": float(offL.min()), "max": float(offL.max()),  # OL-AXIS
                                 "n_in_band": int(np.sum((offL >= bd["o_lo"]) & (offL <= bd["o_hi"]))), "n": int(offL.size)}  # OL-AXIS
        # R4 mKNN zeta, debiased
        cm = np.array([r["c_alpha_mknn"] for r in rows])
        cd = np.array([r["c_alpha_deb"] for r in rows])
        R["R4_mknn"] = {"baseline": PB.KNN_K / (PB.M - 1), "c_alpha_mean": float(cm.mean()),
                        "c_alpha_min": float(cm.min()), "c_alpha_max": float(cm.max()),
                        "zeta_mean": float(np.mean(cm - PB.KNN_K / (PB.M - 1))),
                        "zeta_max": float(np.max(np.abs(cm - PB.KNN_K / (PB.M - 1))))}
        R["R4_debiased"] = {"baseline": 0.0, "c_alpha_mean": float(cd.mean()), "c_alpha_max": float(cd.max()),
                            "E0max_mean": float(np.mean([r["E0_max_deb"] for r in rows]))}
        # R5 OpenLLaMA pilot pairs (the real 3b/7b frozen files)  # OL-AXIS
        pil = [r for r in rows if r["lm"] in OL37]  # OL-AXIS
        po = np.array([r["c_alpha"] - r["Bbar"] for r in pil])  # OL-AXIS
        pm = np.array([r["E0_max"] - r["Bbar"] for r in pil])  # OL-AXIS
        fdo = {}  # OL-AXIS
        for r_ in ("0.5", "1.0"):  # OL-AXIS
            if all(s in Fp["Bfd"][r_] for s in OL37):  # OL-AXIS
                x_ = np.array([r["fd"][r_]["c_alpha"] - Fp["Bfd"][r_][r["lm"]][r["vit"]] for r in pil])  # OL-AXIS
                fdo[r_] = {"median": float(np.median(x_)), "min": float(x_.min()), "max": float(x_.max())}  # OL-AXIS
        R["R5_openllama_pilot"] = {"n": int(po.size), "calpha_minus_Bbar_median": float(np.median(po)),  # OL-AXIS
                                   "min": float(po.min()), "max": float(po.max()),  # OL-AXIS
                                   "n_in_band": int(np.sum((po >= bd["o_lo"]) & (po <= bd["o_hi"]))), "band": bd,  # OL-AXIS
                                   "null_mean_minus_Bbar_median": float(np.median(pm)), "null_mean_minus_Bbar_min": float(pm.min()),  # OL-AXIS
                                   "null_mean_minus_Bbar_max": float(pm.max()),  # OL-AXIS
                                   "n_null_mean_in_mu_band": int(np.sum((pm >= Fp["mu_band"][0]) & (pm <= Fp["mu_band"][1]))),  # OL-AXIS
                                   "fixed_depth_c_minus_Bfd": fdo}  # OL-AXIS
        # R6 retro-forecast of the 17 pilot comparisons 3B < 7B (P-O2b machinery; 7B as the 'new' model)  # OL-AXIS
        rr = {"n_decided": 0, "x_correct": 0, "n_undecided": 0, "rows": []}  # OL-AXIS
        if all(s in Fp["Bbar"] for s in OL37):  # OL-AXIS
            for v in VITS:  # OL-AXIS
                a3, a7 = idx[(PB.OL3, v)], idx[(PB.OL7, v)]  # OL-AXIS
                kap = [a3["s_obs"] + (1 - a3["s_obs"]) * (a7["Bbar"] + o - a3["c_alpha"]) / (1 - a3["c_alpha"])  # OL-AXIS
                       for o in (bd["o_lo"], bd["o_hi"])]  # OL-AXIS
                dec = ("undecided" if a3["s_cal"] == 0 else "below" if a7["s_obs"] < kap[0]  # OL-AXIS
                       else "above" if a7["s_obs"] > kap[1] else "undecided")  # OL-AXIS
                obs = "below" if a7["s_cal"] < a3["s_cal"] else "above" if a7["s_cal"] > a3["s_cal"] else "tie"  # OL-AXIS
                if dec == "undecided":  # OL-AXIS
                    rr["n_undecided"] += 1  # OL-AXIS
                else:  # OL-AXIS
                    rr["n_decided"] += 1  # OL-AXIS
                    rr["x_correct"] += int(dec == obs)  # OL-AXIS
                rr["rows"].append({"vit": v, "T7": a7["s_obs"], "kappa_lo": kap[0], "kappa_hi": kap[1], "dec": dec, "obs": obs})  # OL-AXIS
        R["R6_pilot_retro_forecast"] = rr  # OL-AXIS
        out["per_pre"][pre] = R
        print(f"== {pre}: R1 PR max rel err {R['R1_PR_vs_indep_diag_max_rel_err']['max']:.2e} "
              f"(bitwise {R['R1_PR_vs_indep_diag_max_rel_err']['bitwise_all']}) | "
              f"R2 u* equal {sum(d['u_star_equal'] for d in r2.values())}/8, max|rel err| "
              f"{max(abs(d['rel_err']) for d in r2.values()):.2e} | R3 median|c-B| {R['R3_calpha_minus_Bbar_136']['median_abs']:.4f} "
              f"mean c-B {R['R3_calpha_minus_Bbar_136']['mean']:+.4f} | BLOOMZ agree {agree}/{tot} "
              f"Tr c {R['R3_bloomz']['Tr_BLOOMZ_calpha']:+.2f} Tr B {R['R3_bloomz']['Tr_BLOOMZ_Bbar']:+.2f} | "
              f"7b c {R['R3_llama7b']['mean_c_alpha']:.4f} B {R['R3_llama7b']['mean_Bbar']:.4f} | "
              f"mKNN zeta mean {R['R4_mknn']['zeta_mean']:.5f} max {R['R4_mknn']['zeta_max']:.5f} | "  # OL-AXIS
              f"OL pilot c-B median {R['R5_openllama_pilot']['calpha_minus_Bbar_median']:.5f} in band "  # OL-AXIS
              f"{R['R5_openllama_pilot']['n_in_band']}/34 | LLaMA in band {R['R3_llama_axis_51']['n_in_band']}/51 | "  # OL-AXIS
              f"R6 retro-forecast {rr['x_correct']}/{rr['n_decided']} (undecided {rr['n_undecided']})", flush=True)  # OL-AXIS
    with open(OUT, "w") as f:
        json.dump(PB.jsonable(out), f, indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
