#!/usr/bin/env python
"""S1.2 Pythia axis, Part B sanity checks and retrodictions (role pyt-axis-predict, step S5, after partb_pyt.py).  # PYT-AXIS
Copy of analysis/s12_layermax_openllama/partb_checks_ol.py (sha256 48d91f9eac0915ce8a5712af438435f100f85f9c767be503514da5a2ae71774f),  # PYT-AXIS
adapted at S0b by role pyt-code-infra (2026-10-10); every changed or added line carries the marker PYT-AXIS;  # PYT-AXIS
diff: partb_checks_pyt.py.diff.  # PYT-AXIS

RETRODICTION / SANITY ONLY, outcomes known, not tests. NOT GATED: it opens no score involving a Pythia model.  # PYT-AXIS
Reads  # PYT-AXIS
  results/S1_2_layermax_pythia/partB/partB_features_pyt.json  (features-only Part B numbers)  # PYT-AXIS
and, read-only, the frozen layer-max outputs of KNOWN pairs of other families only:  # PYT-AXIS
  results/S1_2_layermax/pairs/<pre>/<lm>__<vit>.json                 (BLOOMZ, OpenLLaMA 3B/7B, llama-7b)  # PYT-AXIS
  results/S1_2_layermax_llama/early/pairs/<pre>/<lm>__<vit>.json     (llama-13b/30b)  # PYT-AXIS
  results/S1_2_layermax_openllama/early/pairs/<pre>/<lm>__<vit>.json (open_llama_13b)  # PYT-AXIS
  results/S1_2_layermax_llama/partB/partB_features.json, results/S1_2_layermax_openllama/partB/partB_features_ol.json  # PYT-AXIS
No Pythia cross-model score exists or is computed here.  # PYT-AXIS

Checks:
  K1  internal consistency of the Part B JSON: Bbar factorised = brute-force maximum (<= 1e-15), Tr_P(Bbar) =  # PYT-AXIS
      Spearman(size, PR*) exactly, rho_v common to all ViTs, materiality flags recomputed from the stored numbers,  # PYT-AXIS
      membership, H2 branch, band_i = a + b/sqrt(PR*_i) +- h with the frozen constants, extrapolation flags, the  # PYT-AXIS
      P-P2a(ii)/P-P5a counts and needs recomputed from the stored per-ViT baselines  # PYT-AXIS
  K2  B1 recheck block (3 known LMs + 17 ViTs <= 1e-14, clips bitwise) and the clip file equal to meta  # PYT-AXIS
  K3  offset retrodiction on the 187 known pairs (all inside the PR*-indexed band by construction) and the 11-LM  # PYT-AXIS
      refit against the frozen constants (difference > 1e-6 logged; the frozen constants apply regardless)  # PYT-AXIS
  K4  the P-P2b machinery (forecast_pyt.forecast_scal) applied to the known families BLOOMZ (5 sizes), LLaMA-1  # PYT-AXIS
      (7b/13b/30b) and OpenLLaMA (3b/7b/13b) with their own PR*-indexed bands and known T and s_cal: decided /  # PYT-AXIS
      correct / undecided per family (retrodiction; sanity of the machinery, outcomes known)  # PYT-AXIS
  K5  mKNN: c_alpha - 10/1023 (zeta of P-P1) and debiased-CKA c_alpha on the known pairs  # PYT-AXIS
  K6  D7: PR* of the _tok1 stacks finite and positive; Tr_P(Bbar_tok1) against Tr_P(Bbar)  # PYT-AXIS
Output: results/S1_2_layermax_pythia/partB/partB_retro_checks_pyt.json; exit 0 if K1, K2, K3 (all 187 inside) and K6 pass,  # PYT-AXIS
else 5 (K4 and K5 are descriptive).  # PYT-AXIS

RUN (step S5, after partb_pyt.py; CPU, seconds):  # PYT-AXIS
  analysis/s12_layermax_pythia/launch_pyt.sh partb_checks_pyt 2 -- venv/bin/python analysis/s12_layermax_pythia/partb_checks_pyt.py  # PYT-AXIS
S0b test mode (PYT_S0B_TEST=1): reads/writes <guard.SCRATCH>/PB/partB_vp9/ (or --partb-dir, test only).  # PYT-AXIS
"""
from __future__ import annotations

import os

_TH = os.environ.get("PYT_THREADS", "1")  # PYT-AXIS: thread count from launch_pyt.sh (plan F5)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _TH  # PYT-AXIS

import argparse  # noqa: E402  # PYT-AXIS
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402  # PYT-AXIS
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import partb_pyt as PB  # noqa: E402  # PYT-AXIS (functions/constants only; main() not run)
import forecast_pyt as FC  # noqa: E402  # PYT-AXIS (forecast_scal only; main() not run, no gate needed: known data)

FAMS = {"BLOOMZ": [("bigscience__bloomz-560m", PB.FROZEN8), ("bigscience__bloomz-1b1", PB.FROZEN8),  # PYT-AXIS
                   ("bigscience__bloomz-1b7", PB.FROZEN8), ("bigscience__bloomz-3b", PB.FROZEN8),  # PYT-AXIS
                   ("bigscience__bloomz-7b1", PB.FROZEN8)],  # PYT-AXIS
        "LLaMA": [("huggyllama__llama-7b", PB.FROZEN8), ("huggyllama__llama-13b", PB.LLAMA_E),  # PYT-AXIS
                  ("huggyllama__llama-30b", PB.LLAMA_E)],  # PYT-AXIS
        "OpenLLaMA": [("openlm-research__open_llama_3b", PB.FROZEN8), ("openlm-research__open_llama_7b", PB.FROZEN8),  # PYT-AXIS
                      ("openlm-research__open_llama_13b", PB.OL_E)]}  # PYT-AXIS


def _known_pr(pre: str, lm: str, LLPB: dict, OLPB: dict) -> float:  # PYT-AXIS
    if lm == "openlm-research__open_llama_13b":  # PYT-AXIS
        return float(OLPB["per_pre"][pre]["lm"][lm]["PR_star"])  # PYT-AXIS
    return float(LLPB["per_pre"][pre]["lm"][lm]["PR_star"])  # PYT-AXIS


def k1(F: dict, pre: str) -> dict:  # PYT-AXIS
    Q = F["per_pre"][pre]  # PYT-AXIS
    sizes, pyt, vits = F["sizes"], F["pyt_lms"], F["vits"]  # PYT-AXIS
    bc = F["band_constants"][pre]  # PYT-AXIS
    pr = [Q["PR_star"][k] for k in pyt]  # PYT-AXIS
    sumd2 = PB.spearman_exact_sumd2(pr)  # PYT-AXIS
    tr_ex = None if sumd2 is None else 1 - 6 * sumd2 / (7 * 48)  # PYT-AXIS
    G = {k: np.array([[Q[k][lm][v] for v in vits] for lm in pyt]) for k in ("Bbar", "Blate")}  # PYT-AXIS
    G["Bfd_0.5"] = np.array([[Q["Bfd"]["0.5"][lm][v] for v in vits] for lm in pyt])  # PYT-AXIS
    G["Bfd_1.0"] = np.array([[Q["Bfd"]["1.0"][lm][v] for v in vits] for lm in pyt])  # PYT-AXIS
    mats = {k: PB.materiality(G[k], [], PB.trend_rows(G[k])["Tr"]) for k in G}  # PYT-AXIS
    bands_ok = all(abs(Q["band"][s]["f"] - (bc["a"] + bc["b"] / math.sqrt(Q["PR_star"][k]))) <= 1e-15  # PYT-AXIS
                   and abs(Q["band"][s]["lo"] - (Q["band"][s]["f"] - bc["h"])) <= 1e-15  # PYT-AXIS
                   and abs(Q["band"][s]["hi"] - (Q["band"][s]["f"] + bc["h"])) <= 1e-15 for s, k in zip(sizes, pyt))  # PYT-AXIS
    ex_ok = all(Q["extrapolated"][s] == (Q["PR_star"][k] < bc["fit_min_PR"]) for s, k in zip(sizes, pyt))  # PYT-AXIS
    comps = [(a, b, j) for a, b in itertools.combinations(range(7), 2) for j in range(len(vits))]  # PYT-AXIS
    n2 = sum(abs(G["Bbar"][b, j] - G["Bbar"][a, j]) >= PB.DELTA_B for a, b, j in comps)  # PYT-AXIS
    n5 = sum(abs(G["Blate"][b, j] - G["Blate"][a, j]) >= PB.DELTA_B for a, b, j in comps)  # PYT-AXIS
    qu = Q["qualifying"]  # PYT-AXIS
    res = {"Bbar_factorised_vs_bruteforce_maxabs": Q["Bbar_factorised_vs_bruteforce_maxabs"],  # PYT-AXIS
           "Blate_factorised_vs_bruteforce_maxabs": Q["Blate_factorised_vs_bruteforce_maxabs"],  # PYT-AXIS
           "Tr_Bbar": Q["Tr_Bbar"], "Tr_Bbar_exact": tr_ex,  # PYT-AXIS
           "tr_equals_exact": (tr_ex is not None and abs(Q["Tr_Bbar"] - tr_ex) <= 1e-12),  # PYT-AXIS
           "rho_common": Q["rho_common_to_all_vits"],  # PYT-AXIS
           "materiality_recomputed_equal": all(mats[k]["material"] == Q["materiality"][k]["material"] for k in G),  # PYT-AXIS
           "material_flags_equal": (Q["material_Bbar"] == mats["Bbar"]["material"]  # PYT-AXIS
                                    and Q["material_Blate"] == mats["Blate"]["material"]),  # PYT-AXIS
           "members_ok": Q["members"] == (["H1"] if Q["material_Bbar"] else []) + [Q["H2_branch"], "H3", "H4"]  # PYT-AXIS
                                         and Q["H2_branch"] == ("H2a" if Q["material_Blate"] else "H2b"),  # PYT-AXIS
           "s_ok": Q["s"] == (1 if Q["Tr_Bbar"] >= 0 else -1) and Q["s_late"] == (1 if Q["Tr_Blate"] >= 0 else -1),  # PYT-AXIS
           "bands_ok": bool(bands_ok), "extrap_ok": bool(ex_ok),  # PYT-AXIS
           "qualifying_ok": (n2 == qu["PP2aii_n"] and n5 == qu["PP5a_n"] and qu["PP2aii_need"] == -(-9 * n2 // 10)  # PYT-AXIS
                             and qu["PP2aii_need67"] == -(-2 * n2 // 3) and qu["PP5a_need"] == -(-2 * n5 // 3)),  # PYT-AXIS
           "H2_same_ok": Q["H2_same_as_H1"] == (sum(Q["Blate_equals_Bbar"].values()) >= 6)}  # PYT-AXIS
    res["pass"] = bool(res["Bbar_factorised_vs_bruteforce_maxabs"] <= 1e-15 and res["Blate_factorised_vs_bruteforce_maxabs"]  # PYT-AXIS
                       <= 1e-15 and (res["tr_equals_exact"] or tr_ex is None) and all(res["rho_common"].values())  # PYT-AXIS
                       and res["materiality_recomputed_equal"] and res["material_flags_equal"] and res["members_ok"]  # PYT-AXIS
                       and res["s_ok"] and bands_ok and ex_ok and res["qualifying_ok"] and res["H2_same_ok"])  # PYT-AXIS
    return res  # PYT-AXIS


def k4(pre: str, F: dict, LLPB: dict, OLPB: dict) -> dict:  # PYT-AXIS: P-P2b machinery on known families
    bc = F["band_constants"][pre]  # PYT-AXIS
    vits = F["vits"]  # PYT-AXIS
    vit_pr = {v: LLPB["per_pre"][pre]["vit"][v]["PR_star"] for v in vits}  # PYT-AXIS
    out = {}  # PYT-AXIS
    for fam, lms in FAMS.items():  # PYT-AXIS
        T, S, Bb, bd = {}, {}, {}, {}  # PYT-AXIS
        for lm, base in lms:  # PYT-AXIS
            ps = _known_pr(pre, lm, LLPB, OLPB)  # PYT-AXIS
            f = bc["a"] + bc["b"] / math.sqrt(ps)  # PYT-AXIS
            bd[lm] = (f - bc["h"], f + bc["h"])  # PYT-AXIS
            for v in vits:  # PYT-AXIS
                js, _ = PB.read_pair_json(base, pre, lm, v)  # PYT-AXIS
                fb = js["summaries"]["cka_biased"]["full"]  # PYT-AXIS
                T[(lm, v)], S[(lm, v)] = float(fb["s_obs"]), float(fb["s_cal"])  # PYT-AXIS
                Bb[(lm, v)] = math.sqrt(ps * vit_pr[v]) / (PB.M - 1)  # PYT-AXIS
        nd = xc = nu = 0  # PYT-AXIS
        for (la, _), (lb, _) in itertools.combinations(lms, 2):  # PYT-AXIS
            for v in vits:  # PYT-AXIS
                dec = FC.forecast_scal(T[(la, v)], T[(lb, v)], Bb[(la, v)], Bb[(lb, v)], bd[la], bd[lb])  # PYT-AXIS
                if dec == "undecided":  # PYT-AXIS
                    nu += 1  # PYT-AXIS
                    continue  # PYT-AXIS
                nd += 1  # PYT-AXIS
                xc += int(dec == FC.obs_order(S[(lb, v)], S[(la, v)]))  # PYT-AXIS
        out[fam] = {"n_comparisons": nd + nu, "n_decided": nd, "x_correct": xc, "n_undecided": nu}  # PYT-AXIS
    out["label"] = "retrodiction on known families (outcomes known, not a test; sanity of the P-P2b machinery)"  # PYT-AXIS
    return out  # PYT-AXIS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # PYT-AXIS
    ap.add_argument("--partb-dir", default=None, help="S0b test mode only")  # PYT-AXIS
    args = ap.parse_args()  # PYT-AXIS
    test = PB.guard.test_mode()  # PYT-AXIS
    if args.partb_dir and not test:  # PYT-AXIS
        raise SystemExit("--partb-dir is an S0b test option")  # PYT-AXIS
    pdir = args.partb_dir or (os.path.join(PB.SCRATCH_PB, "partB_vp9") if test else PB.OUTDIR_REAL)  # PYT-AXIS
    OUT = os.path.join(pdir, "partB_retro_checks_pyt.json")  # PYT-AXIS
    if not os.path.isfile(os.path.join(pdir, "partB_features_pyt.json")):  # PYT-AXIS: order gate (after partb_pyt.py)
        print(f"[partb_checks_pyt] REFUSED (exit 3): {pdir}/partB_features_pyt.json missing", file=sys.stderr)  # PYT-AXIS
        raise SystemExit(3)  # PYT-AXIS
    if os.path.exists(OUT):  # PYT-AXIS: no-delete rule
        if not test:  # PYT-AXIS
            raise SystemExit(f"{OUT} exists (no-delete rule)")  # PYT-AXIS
        os.replace(OUT, OUT + ".prev-" + str(os.getpid()))  # PYT-AXIS
    F = json.load(open(os.path.join(pdir, "partB_features_pyt.json")))  # PYT-AXIS
    clipf = json.load(open(os.path.join(pdir, "clip_thresholds_partB_pyt.json")))  # PYT-AXIS
    for p_, want in ((PB.LLAMA_PARTB, PB.LLAMA_PARTB_SHA), (PB.OL_PARTB, PB.OL_PARTB_SHA)):  # PYT-AXIS
        if PB.sha256_file(p_) != want:  # PYT-AXIS
            raise SystemExit(f"frozen input changed: {p_}")  # PYT-AXIS
    LLPB, OLPB = json.load(open(PB.LLAMA_PARTB)), json.load(open(PB.OL_PARTB))  # PYT-AXIS
    out = {"label": "retrodiction / sanity, outcomes known, not tests; no Pythia cross-model score used",  # PYT-AXIS
           "partB_features_pyt_sha256": PB.sha256_file(os.path.join(pdir, "partB_features_pyt.json")),  # PYT-AXIS
           "test_mode": test, "per_pre": {}}  # PYT-AXIS
    ok_all = True  # PYT-AXIS
    k2 = {"recheck": F["meta"]["recheck"],  # PYT-AXIS
          "clip_file_equals_meta": all(clipf["clip"][k]["c"] == F["meta"]["clip_thresholds"][k] for k in F["pyt_lms"])}  # PYT-AXIS
    k2["pass"] = bool(k2["recheck"]["pass"] and k2["clip_file_equals_meta"])  # PYT-AXIS
    out["K2"] = k2  # PYT-AXIS
    ok_all &= k2["pass"]  # PYT-AXIS
    for pre in PB.PRES:
        R = {}
        R["K1"] = k1(F, pre)  # PYT-AXIS
        used: dict = {}  # PYT-AXIS
        ro = PB.known_offsets(pre, LLPB, OLPB, used)  # PYT-AXIS
        stored = F["per_pre"][pre]["B5"]["offset_retrodiction_187"]  # PYT-AXIS
        R["K3"] = {"n": ro["n"], "n_in_band": ro["n_in_band"], "refit": ro["refit"],  # PYT-AXIS
                   "refit_diff_gt_1e-6": bool(ro["refit"]["abs_diff_a"] > 1e-6 or ro["refit"]["abs_diff_b"] > 1e-6),  # PYT-AXIS
                   "equals_partB_stored": (ro["n_in_band"] == stored["n_in_band"]  # PYT-AXIS
                                           and ro["refit"]["a"] == stored["refit"]["a"]),  # PYT-AXIS
                   "pass": bool(ro["n_in_band"] == ro["n"] == 187)}  # PYT-AXIS
        R["K4"] = k4(pre, F, LLPB, OLPB)  # PYT-AXIS
        cm, cd = [], []  # PYT-AXIS: K5
        for lm, base in PB.KNOWN11:  # PYT-AXIS
            for v in F["vits"]:  # PYT-AXIS
                js, _ = PB.read_pair_json(base, pre, lm, v)  # PYT-AXIS
                cm.append(float(js["summaries"]["mknn"]["full"]["c_alpha"]))  # PYT-AXIS
                cd.append(float(js["summaries"]["cka_debiased"]["full"]["c_alpha"]))  # PYT-AXIS
        cm, cd = np.array(cm), np.array(cd)  # PYT-AXIS
        R["K5"] = {"mknn_baseline": PB.KNN_K / (PB.M - 1), "mknn_c_alpha_mean": float(cm.mean()),  # PYT-AXIS
                   "zeta_mean": float(np.mean(cm - PB.KNN_K / (PB.M - 1))),  # PYT-AXIS
                   "zeta_max": float(np.max(np.abs(cm - PB.KNN_K / (PB.M - 1)))),  # PYT-AXIS
                   "debiased_c_alpha_mean": float(cd.mean()), "debiased_c_alpha_max": float(cd.max())}  # PYT-AXIS
        d7 = F["per_pre"][pre]["D7"]  # PYT-AXIS
        R["K6"] = {"Tr_Bbar_tok1": d7["Tr_Bbar_tok1"], "Tr_Bbar": d7["Tr_Bbar"],  # PYT-AXIS
                   "pass": bool(all(np.isfinite(x) and x > 0 for x in d7["PR_star_tok1"].values()))}  # PYT-AXIS
        ok_all &= R["K1"]["pass"] and R["K3"]["pass"] and R["K6"]["pass"]  # PYT-AXIS
        out["per_pre"][pre] = R
        print(f"== {pre}: K1 pass {R['K1']['pass']} | K3 in band {R['K3']['n_in_band']}/{R['K3']['n']} refit a "  # PYT-AXIS
              f"{R['K3']['refit']['a']:.6f} b {R['K3']['refit']['b']:.6f} (diff>1e-6 {R['K3']['refit_diff_gt_1e-6']}) | "  # PYT-AXIS
              "K4 " + "; ".join(f"{f} {d['x_correct']}/{d['n_decided']} (undecided {d['n_undecided']})"  # PYT-AXIS
                                for f, d in R["K4"].items() if f != "label") +  # PYT-AXIS
              f" | K5 zeta max {R['K5']['zeta_max']:.5f} | K6 pass {R['K6']['pass']}", flush=True)  # PYT-AXIS
    out["pass"] = bool(ok_all)  # PYT-AXIS
    with open(OUT, "w") as f:
        json.dump(PB.jsonable(out), f, indent=1)
    print("wrote", OUT, "pass", out["pass"], flush=True)  # PYT-AXIS
    sys.exit(0 if out["pass"] else 5)  # PYT-AXIS


if __name__ == "__main__":
    main()
