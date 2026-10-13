"""EXPLORATORY, POST HOC (not in the frozen plan; written after the main
results were seen). Where does the layer-max null threshold come from?

From the stored per-pair arrays (observed matrices f64, null tensors CKA
float32 / mKNN int16 counts, entry-wise null means f64):

 1. Null-level profile over LM depth: for each LM layer a, the highest
    entry-wise null mean over ViT layers, max_b E0[S_ab]; per LM, averaged over
    the 17 ViTs; the LM layer that attains max_ab E0[S_ab].
 2. Level vs multiplicity: E0[max_ab S_ab] (null mean of the layer-max) vs
    max_ab E0[S_ab] (the highest single-entry null level). If they are close,
    c_alpha(max) is set by the null LEVEL of one layer pair (heterogeneous
    nulls), not by taking a max over many comparisons.
 3. Leave-layers-out layer-max (recomputed from the stored nulls): all LM
    layers except the embedding output; except embedding and last; LM layers
    with relative depth >= 0.5; the embedding layer alone. Same trend
    statistics as the main analysis. 'full' is recomputed too, as a check of
    the float32 null storage against the float64 main-run values.

Output: results/S1_2_layermax/posthoc_null_structure.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402
import importlib.util  # noqa: E402
_sp = importlib.util.spec_from_file_location("lm_aggregate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "aggregate.py"))
AG = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(AG)
from hrhv2 import calibrate as cal  # noqa: E402

METS = ("cka_biased", "mknn", "cka_debiased")


def null_tensor(z, met):
    if met == "mknn":
        return z["mknn.null_counts_int16"].astype(np.float64) / (LC.M_ITEMS * LC.KNN_K)
    return z[f"{met}.null_f32"].astype(np.float64)


def lm_sets(LA):
    return {"full": list(range(LA)), "noLMemb": list(range(1, LA)), "noLMemb_noLMlast": list(range(1, LA - 1)),
            "LMdeep(rel>=0.5)": [a for a in range(LA) if a / (LA - 1) >= 0.5], "LMemb_only": [0]}


def main():
    out = {"note": "EXPLORATORY, post hoc (not preregistered); written after the main results were seen", "per_pre": {}}
    for pre in LC.PRES:
        P = {"profile": {}, "level_vs_multiplicity": {}, "leave_out": {}, "storage_check": {}}
        grids = {met: {} for met in METS}
        prof = {met: {lm: [] for lm, _ in LC.LMS} for met in METS}
        lvm = {met: [] for met in METS}
        argmax_layer = {met: {lm: [] for lm, _ in LC.LMS} for met in METS}
        store_err = {met: 0.0 for met in METS}
        for i, (lm, _) in enumerate(LC.LMS):
            for j, (vit, _) in enumerate(LC.VITS):
                z = np.load(os.path.join(LC.RES, "pairs", pre, f"{lm}__{vit}.npz"))
                js = json.load(open(os.path.join(LC.RES, "pairs", pre, f"{lm}__{vit}.json")))
                for met in METS:
                    obs = z[f"{met}.entry_obs"]
                    nm = z[f"{met}.entry_null_mean"]
                    null = null_tensor(z, met)
                    LA = obs.shape[0]
                    prof[met][lm].append(nm.max(axis=1))                       # (LA,)
                    a0, b0 = np.unravel_index(np.argmax(nm), nm.shape)
                    argmax_layer[met][lm].append(int(a0) / (LA - 1))
                    lvm[met].append({"E_max": float(null.reshape(null.shape[0], -1).max(axis=1).mean()),
                                     "max_E": float(nm.max()),
                                     "inflation": js["summaries"][met]["full"]["c_alpha"]
                                     - js["summaries"][met]["full"]["entry_c_alpha_at_argmax"]})
                    for name, ia in lm_sets(LA).items():
                        T = float(obs[ia].max())
                        tk = null[:, ia].reshape(null.shape[0], -1).max(axis=1)
                        s = cal.summarize(T, tk, LC.ALPHA)
                        for col in ("s_obs", "s_cal", "uncensored", "c_alpha", "null_mean"):
                            grids[met].setdefault(f"{name}|{col}", np.full((len(LC.LMS), len(LC.VITS)), np.nan))[i, j] = s[col]
                        if name == "full":
                            ref = js["summaries"][met]["full"]
                            store_err[met] = max(store_err[met], abs(s["c_alpha"] - ref["c_alpha"]),
                                                 abs(s["uncensored"] - ref["uncensored"]))
        for met in METS:
            # profile: mean over ViTs of max_b E0[S_ab], at relative depths
            P["profile"][met] = {}
            for lm, _ in LC.LMS:
                M = np.mean(np.stack(prof[met][lm]), axis=0)                    # (LA,)
                LA = M.size
                P["profile"][met][lm] = {
                    "null_level_by_layer": [float(v) for v in M],
                    "emb": float(M[0]), "mid": float(M[LC.fixed_layer(LA, 0.5)]), "last": float(M[-1]),
                    "max": float(M.max()), "argmax_layer": int(np.argmax(M)),
                    "argmax_rel_depth_mean_over_vits": float(np.mean(argmax_layer[met][lm])),
                    "frac_vits_argmax_at_emb": float(np.mean(np.array(argmax_layer[met][lm]) == 0))}
            L = lvm[met]
            P["level_vs_multiplicity"][met] = {
                "mean_E0_max": float(np.mean([d["E_max"] for d in L])),
                "mean_max_E0": float(np.mean([d["max_E"] for d in L])),
                "mean_multiplicity_excess": float(np.mean([d["E_max"] - d["max_E"] for d in L])),
                "mean_inflation_calpha_minus_entry": float(np.mean([d["inflation"] for d in L]))}
            P["storage_check"][met] = {"max_absdiff_full_recomputed_vs_main_f64": store_err[met]}
            P["leave_out"][met] = {}
            for key, G in grids[met].items():
                name, col = key.split("|")
                blk = {"mean": float(np.nanmean(G))}
                if col in ("s_obs", "s_cal", "uncensored", "c_alpha"):
                    blk.update({"arh_r": AG.arh_r(G), "tr_lm": AG.tr_lm(G, f"posthoc|{pre}|{met}|{key}"),
                                "tr_vis": AG.tr_vis(G)})
                P["leave_out"][met][key] = blk
        out["per_pre"][pre] = P
    with open(os.path.join(LC.RES, "posthoc_null_structure.json"), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    for pre in LC.PRES:
        P = out["per_pre"][pre]
        print("==", pre, "storage check", P["storage_check"])
        for met in METS:
            print(met, "level vs multiplicity", P["level_vs_multiplicity"][met])
            for lm, _ in LC.LMS:
                p = P["profile"][met][lm]
                print(f"  {lm:35s} emb {p['emb']:.4f} mid {p['mid']:.4f} last {p['last']:.4f} max {p['max']:.4f} "
                      f"@L{p['argmax_layer']} relmean {p['argmax_rel_depth_mean_over_vits']:.2f} fracEmb {p['frac_vits_argmax_at_emb']:.2f}")
            for key, b in P["leave_out"][met].items():
                if "tr_lm" in b:
                    print(f"  {key:32s} mean {b['mean']:.4f} arh_r {b['arh_r']:.2f} TrLM {b['tr_lm']['Tr']:.2f} "
                          f"(p {b['tr_lm']['p_one']:.3f}) TrVis {b['tr_vis']['Tr']:.2f} (p {b['tr_vis']['p_one']:.3f})")


if __name__ == "__main__":
    main()
