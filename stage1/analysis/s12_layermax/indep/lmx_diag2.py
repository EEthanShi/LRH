#!/usr/bin/env python
"""EXPLORATORY (not in PLAN_INDEP.md): layer-max with the LM input-embedding
layer (hidden_states[0]) excluded from the max, for every metric and pre.
Uses the stored null matrices (same permutations); no recomputation.
Writes results/S1_2_layermax/indep/diag_no_embedding.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
import json  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import lmx_aggregate as A  # noqa: E402  (functions only; main() not run)
import lmx_common as L  # noqa: E402

SUBS = {
    "full": lambda LA, LB: (np.arange(LA), np.arange(LB)),
    "no_lm_emb": lambda LA, LB: (np.arange(1, LA), np.arange(LB)),
    "no_lm_emb_no_vit_first": lambda LA, LB: (np.arange(1, LA), np.arange(1, LB)),
    "lm_upper_half": lambda LA, LB: (np.arange((LA - 1) // 2, LA), np.arange(LB)),
}


def main():
    m, k = L.M, L.KNN
    out = {}
    for pre in L.PRES:
        G = {s: {met: {f: np.empty((A.NL, A.NV)) for f in ("s_obs", "s_cal", "uncensored", "c_alpha", "rel_a")}
                 for met in A.METRICS} for s in SUBS}
        for i, lm in enumerate(L.LM_STEMS):
            for j, vit in enumerate(L.VIT_STEMS):
                z = np.load(os.path.join(L.OUT, pre, "pairs", f"{lm}__{vit}.npz"))
                mats = {"mknn": (z["obs_cnt"] / (m * k), z["null_cnt"].astype(np.float64) / (m * k)),
                        "cka_b": (z["obs_cka_b"], z["null_cka_b"]), "cka_d": (z["obs_cka_d"], z["null_cka_d"])}
                for met, (S, N) in mats.items():
                    LA, LB = S.shape
                    for s, fn in SUBS.items():
                        ra, rb = fn(LA, LB)
                        Ss = S[np.ix_(ra, rb)]
                        Ns = N[:, ra][:, :, rb]
                        T = float(Ss.max())
                        sm = L.summarize(T, Ns.reshape(Ns.shape[0], -1).max(axis=1))
                        a, _ = np.unravel_index(int(np.argmax(Ss)), Ss.shape)
                        for f in ("s_obs", "s_cal", "uncensored", "c_alpha"):
                            G[s][met][f][i, j] = sm[f]
                        G[s][met]["rel_a"][i, j] = ra[a] / (LA - 1)
        res = {}
        for s in SUBS:
            res[s] = {}
            for met in A.METRICS:
                g = G[s][met]
                res[s][met] = {
                    "arh_r_before": A.arh_r(g["s_obs"], A.COVARIATES["arh_rank"], "x", False)["r"],
                    "arh_r_after": A.arh_r(g["s_cal"], A.COVARIATES["arh_rank"], "x", False)["r"],
                    "arh_r_unc": A.arh_r(g["uncensored"], A.COVARIATES["arh_rank"], "x", False)["r"],
                    "Tr_raw": A.S12.lm_trend(g["s_obs"])["Tr"],
                    "Tr_scal": A.S12.lm_trend(g["s_cal"])["Tr"],
                    "Tr_unc": A.S12.lm_trend(g["uncensored"])["Tr"],
                    "Tr_unc_p_one": A.S12.lm_trend(g["uncensored"])["p_one"],
                    "Tr_calpha": A.S12.lm_trend(g["c_alpha"])["Tr"],
                    "mean_c_alpha": float(g["c_alpha"].mean()),
                    "mean_sel_rel_a": float(g["rel_a"].mean()),
                    "calpha_by_lm": {lm: float(g["c_alpha"][i].mean()) for i, lm in enumerate(L.LM_STEMS)},
                }
                r = res[s][met]
                print(f"{pre:12s} {s:24s} {met:6s} ARH r {r['arh_r_before']:.2f}->{r['arh_r_after']:.2f} (unc {r['arh_r_unc']:.2f}) | "
                      f"Tr raw {r['Tr_raw']:.2f} scal {r['Tr_scal']:.2f} unc {r['Tr_unc']:.2f} (p {r['Tr_unc_p_one']:.3f}) "
                      f"| Tr c_a {r['Tr_calpha']:.2f} mean c_a {r['mean_c_alpha']:.4f}", flush=True)
        out[pre] = res
    json.dump(L.jsonable(out), open(os.path.join(L.OUT, "diag_no_embedding.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
