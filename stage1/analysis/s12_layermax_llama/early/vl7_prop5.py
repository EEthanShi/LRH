"""LLAMA-AXIS (new file, implementation E, role llama-axis-run-early): plan V-L7.

Proposition 5, numerically, on the 51 LLaMA-axis pairs (both pre). Compares the
entry-wise permutation null means (500 pi, float64, stored by the main run as
<met>.entry_null_mean with <met>.entry_null_sd) with the exact random-pairing
baselines:
  * biased CKA:   B_uw = sqrt(PR_u PR_w)/(m-1), PR from the frozen Part B file
                  (partB_features.json -> per_pre.<pre>.lm|vit.<stem>.PR);
  * debiased CKA: 0 (plan section 8);
  * mKNN(10):     10/1023.
Reports the median relative error (CKA biased) and, for every metric, the
fraction of entries with |z| > 4, z = (null mean - baseline)/(null sd/sqrt(K)).
Only null statistics are read (no observed score).
Expected (plan): median rel. error <= 3%; |z| > 4 fraction <= 1%.

Output: results/S1_2_layermax_llama/early/validation/VL7_prop5.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402

M = LC.M_ITEMS
BASE_MKNN = LC.KNN_K / (M - 1)


def main():
    pb = json.load(open(os.path.join(LC.PARTB, "partB_features.json")))
    out = {"K": LC.K_PERM, "per_pre": {}}
    for pre in LC.PRES:
        P = pb["per_pre"][pre]
        acc = {met: {"rel": [], "z": [], "n_sd0": 0} for met in ("cka_biased", "cka_debiased", "mknn")}
        per_lm = {}
        for lm, _ in LC.LMS:
            prA = np.array(P["lm"][lm]["PR"], dtype=np.float64)
            rl, zl = [], []
            for vit, _ in LC.VITS:
                prB = np.array(P["vit"][vit]["PR"], dtype=np.float64)
                z = np.load(os.path.join(LC.RES, "pairs", pre, f"{lm}__{vit}.npz"))
                for met in acc:
                    nm = z[f"{met}.entry_null_mean"]
                    sd = z[f"{met}.entry_null_sd"]
                    if met == "cka_biased":
                        base = np.sqrt(prA[:, None] * prB[None, :]) / (M - 1)
                        assert base.shape == nm.shape, (lm, vit, base.shape, nm.shape)
                        r = np.abs(nm - base) / base
                        acc[met]["rel"].append(r.ravel())
                        rl.append(r.ravel())
                    else:
                        base = np.full_like(nm, 0.0 if met == "cka_debiased" else BASE_MKNN)
                    se = sd / np.sqrt(LC.K_PERM)
                    good = se > 0
                    acc[met]["n_sd0"] += int((~good).sum())
                    zz = np.where(good, (nm - base) / np.where(good, se, 1.0), np.nan)
                    acc[met]["z"].append(zz.ravel())
                    if met == "cka_biased":
                        zl.append(zz.ravel())
            rl, zl = np.concatenate(rl), np.concatenate(zl)
            per_lm[lm] = {"cka_biased_median_rel_err": float(np.median(rl)), "cka_biased_max_rel_err": float(rl.max()),
                          "cka_biased_frac_absz_gt4": float(np.nanmean(np.abs(zl) > 4))}
        R = {"per_lm": per_lm}
        for met, a in acc.items():
            zz = np.concatenate(a["z"])
            fin = np.isfinite(zz)
            d = {"n_entries": int(zz.size), "n_entries_null_sd_zero": a["n_sd0"],
                 "frac_absz_gt4": float(np.mean(np.abs(zz[fin]) > 4)), "max_absz": float(np.max(np.abs(zz[fin]))),
                 "mean_z": float(np.mean(zz[fin])), "sd_z": float(np.std(zz[fin]))}
            if a["rel"]:
                rr = np.concatenate(a["rel"])
                d.update({"median_rel_err": float(np.median(rr)), "p95_rel_err": float(np.percentile(rr, 95)),
                          "max_rel_err": float(rr.max())})
            R[met] = d
        R["pass"] = bool(R["cka_biased"]["median_rel_err"] <= 0.03
                         and all(R[m]["frac_absz_gt4"] <= 0.01 for m in ("cka_biased", "cka_debiased", "mknn")))
        out["per_pre"][pre] = R
        print(pre, json.dumps(LC.jsonable({m: R[m] for m in acc})), "PASS" if R["pass"] else "FAIL", flush=True)
    out["pass"] = all(v["pass"] for v in out["per_pre"].values())
    out["finished_utc"] = LC.utc()
    with open(os.path.join(LC.RES, "validation", "VL7_prop5.json"), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    print("V-L7", "PASS" if out["pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
