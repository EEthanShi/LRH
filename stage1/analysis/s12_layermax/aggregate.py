"""Aggregation for the ARH layer-max replication (plan sections 5-7).

Reads results/S1_2_layermax/pairs/<pre>/*.json (+ .npz for null-argmax
locations) and the S1.2 fixed-depth panel (results/S1_2/panel_main.parquet,
read-only). Trend statistics reuse S1.2's own functions (analysis/s12/
aggregate.py: lm_trend, vision_trend; read-only import) so that "Tr" means
exactly what it meant in S1.2.

Outputs: results/S1_2_layermax/panel_layermax.parquet
         results/S1_2_layermax/layermax_summary.json
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, pearsonr  # noqa: E402

import lm_core as LC  # noqa: E402

sys.path.insert(0, os.path.join(LC.PROJ, "analysis", "s12"))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("s12_aggregate", os.path.join(LC.PROJ, "analysis", "s12", "aggregate.py"))
A12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A12)
from hrhv2 import util  # noqa: E402

METRICS = ("mknn", "cka_biased", "cka_debiased")
MET_LABEL = {"mknn": "mKNN(10)", "cka_biased": "CKA biased", "cka_debiased": "CKA debiased (expl.)"}
SUBSETS = ["fixed_r0.5", "fixed_r1.0", "grid2", "grid3", "grid5", "grid9", "grid17", "LMall_ViTlast",
           "LMlast_ViTall", "full"]
LM_IDX = {s: i for i, (s, _) in enumerate(LC.LMS)}
VIT_IDX = {s: j for j, (s, _) in enumerate(LC.VITS)}
NB = 2000
EPS = 1e-12


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_panel() -> pd.DataFrame:
    rows = []
    for pre in LC.PRES:
        for f in sorted(glob.glob(os.path.join(LC.RES, "pairs", pre, "*.json"))):
            rec = json.load(open(f))
            for met, subs in rec["summaries"].items():
                for name, s in subs.items():
                    row = {"lm": rec["lm"], "vit": rec["vit"], "pre": pre, "metric": met, "subset": name,
                           "K": rec["K"], "L_A": rec["L_A"], "L_B": rec["L_B"]}
                    for k, v in s.items():
                        if not isinstance(v, list):
                            row[k] = v
                    rows.append(row)
    df = pd.DataFrame(rows)
    df["rel_A"] = df["argmax_A"] / (df["L_A"] - 1)
    df["rel_B"] = df["argmax_B"] / (df["L_B"] - 1)
    df["shift"] = df["s_obs"] - df["c_alpha"]
    df["sel_entry_thr"] = df["s_obs"] - df["entry_c_alpha_at_argmax"]
    df["sel_entry_mean"] = df["s_obs"] - df["entry_null_mean_at_argmax"]
    df["inflation"] = df["c_alpha"] - df["entry_c_alpha_at_argmax"]
    df["arh_rank"] = df["lm"].map(LC.ARH_RANK)
    df["dense_rank"] = df["lm"].map(LC.DENSE_RANK)
    df["lm_params"] = df["lm"].map(LC.C12.n_params)
    df["vit_params"] = df["vit"].map(LC.C12.n_params)
    return df


def grid_of(df, pre, metric, subset, col) -> np.ndarray:
    G = np.full((len(LC.LMS), len(LC.VITS)), np.nan)
    sub = df[(df.pre == pre) & (df.metric == metric) & (df.subset == subset)]
    for lm, vit, v in zip(sub.lm, sub.vit, sub[col].astype(float)):
        G[LM_IDX[lm], VIT_IDX[vit]] = v
    return G


def s12_grid(pre, r, col) -> np.ndarray:
    df = pd.read_parquet(os.path.join(LC.PROJ, "results", "S1_2", "panel_main.parquet"))
    return A12.grid(df, col, r, pre)


# ----------------------------------------------------------------------------
# ARH-style Pearson r with LLM rank, exact LM-relabelling p, ViT bootstrap
# ----------------------------------------------------------------------------

_LM_PERMS = np.array(list(itertools.permutations(range(len(LC.LMS)))))     # 40320 x 8


def rank_vec(kind="arh"):
    m = LC.ARH_RANK if kind == "arh" else LC.DENSE_RANK
    return np.array([m[s] for s, _ in LC.LMS], dtype=float)


def arh_r(G: np.ndarray, kind="arh") -> float:
    x = np.repeat(rank_vec(kind)[:, None], G.shape[1], axis=1).ravel()
    y = G.ravel()
    if np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def arh_r_test(G: np.ndarray, kind="arh", key="") -> dict:
    x = rank_vec(kind)
    nv = G.shape[1]
    y = G - G.mean()
    rows = y.sum(axis=1)                                   # (8,)
    xc_all = x[_LM_PERMS] - x.mean()                       # (P, 8): rank given to LM i under relabelling
    num = xc_all @ rows
    den = np.sqrt(nv * np.sum((x - x.mean()) ** 2)) * np.sqrt(np.sum(y ** 2))
    R = num / den
    r_obs = arh_r(G, kind)
    rng = util.rng_from_key(f"S1.2-layermax|boot|arh_r|{key}")
    bs = np.empty(NB)
    for b in range(NB):
        j = rng.integers(0, nv, nv)
        bs[b] = arh_r(G[:, j], kind)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"r": r_obs, "p_one": float(np.mean(R >= r_obs - EPS)), "ci": [float(lo), float(hi)],
            "n_relabel": int(R.size)}


def arh_delta_boot(G1b, G1a, G2b, G2a, key) -> dict:
    """(r(G1b)-r(G1a)) - (r(G2b)-r(G2a)) with a ViT bootstrap CI."""
    f = lambda a, b, c, d: (arh_r(a) - arh_r(b)) - (arh_r(c) - arh_r(d))  # noqa: E731
    est = f(G1b, G1a, G2b, G2a)
    rng = util.rng_from_key(f"S1.2-layermax|boot|arh_delta|{key}")
    nv = G1b.shape[1]
    bs = np.empty(NB)
    for b in range(NB):
        j = rng.integers(0, nv, nv)
        bs[b] = f(G1b[:, j], G1a[:, j], G2b[:, j], G2a[:, j])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"est": float(est), "ci": [float(lo), float(hi)]}


def tr_lm(G: np.ndarray, key="") -> dict:
    t = A12.lm_trend(G)
    est, lo, hi = A12.boot_vits(np.array([np.nan if v is None else v for v in t["rho_per_vit"]]), f"layermax|{key}")
    return {"Tr": t["Tr"], "p_one": t["p_one"], "p_two": t["p_two"], "ci": [lo, hi],
            "n_vit_constant": t["n_vit_constant"], "openllama_frac": t["openllama_frac_larger_higher"]}


def tr_vis(G: np.ndarray) -> dict:
    t = A12.vision_trend(G)
    return {"Tr": t["Tr"], "p_one": t["p_one"], "p_two": t["p_two"], "n_cells": t["n_cells"]}


def trend_block(G: np.ndarray, key: str) -> dict:
    return {"arh_r": arh_r_test(G, "arh", key), "arh_r_dense": arh_r(G, "dense"),
            "tr_lm": tr_lm(G, key), "tr_vis": tr_vis(G),
            "mean": float(np.nanmean(G)), "median": float(np.nanmedian(G))}


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    df = load_panel()
    df.to_parquet(os.path.join(LC.RES, "panel_layermax.parquet"))
    out = {"n_rows": int(len(df)), "checks": {}, "per_pre": {}}
    for pre in LC.PRES:
        n_pairs = df[(df.pre == pre) & (df.metric == "mknn") & (df.subset == "full")].shape[0]
        out["checks"][pre] = {"n_pairs": n_pairs, "K_values": sorted(df[df.pre == pre].K.unique().tolist())}
    for pre in LC.PRES:
        P = {"trends": {}, "fixed_s12": {}, "decomp": {}, "selection": {}, "null_vs_size": {}, "per_lm": {},
             "subset_curve": {}, "reading": {}}
        # ---- 5(a-c): layer-max trends on raw / s_cal / uncensored for each metric; fixed-depth anchors
        for met in METRICS:
            P["trends"][met] = {}
            for col in ("s_obs", "s_cal", "uncensored"):
                for subset in SUBSETS:
                    G = grid_of(df, pre, met, subset, col)
                    P["trends"][met][f"{subset}|{col}"] = trend_block(G, f"{pre}|{met}|{subset}|{col}")
        s12map = {"mknn": "cal.s_N_k10", "cka_biased": "cal.cka_biased", "cka_debiased": "cal.cka_debiased"}
        for met in METRICS:
            P["fixed_s12"][met] = {}
            for r in (0.5, 1.0):
                for col in ("s_obs", "s_cal", "uncensored"):
                    G = s12_grid(pre, r, f"{s12map[met]}.{col}")
                    P["fixed_s12"][met][f"r{r}|{col}"] = trend_block(G, f"s12|{pre}|{met}|{r}|{col}")
        # ---- 6: replication reading (ARH-style, layer-max full)
        def rr(met, col):
            return P["trends"][met][f"full|{col}"]["arh_r"]["r"]
        for cka in ("cka_biased", "cka_debiased"):
            d_c = rr(cka, "s_obs") - rr(cka, "s_cal")
            d_m = rr("mknn", "s_obs") - rr("mknn", "s_cal")
            gap_after = rr("mknn", "s_cal") - rr(cka, "s_cal")
            if (d_c - d_m) >= 0.20 and gap_after >= 0.20:
                verdict = "ARH pattern reappears"
            elif (d_c - d_m) <= 0.10 or gap_after <= 0.10:
                verdict = "ARH pattern absent"
            else:
                verdict = "partial"
            boot = arh_delta_boot(grid_of(df, pre, cka, "full", "s_obs"), grid_of(df, pre, cka, "full", "s_cal"),
                                  grid_of(df, pre, "mknn", "full", "s_obs"), grid_of(df, pre, "mknn", "full", "s_cal"),
                                  f"{pre}|{cka}")
            tr_c = P["trends"][cka]["full|uncensored"]["tr_lm"]["Tr"]
            tr_m = P["trends"]["mknn"]["full|uncensored"]["tr_lm"]["Tr"]
            if tr_c <= 0.40 and tr_m >= 0.60:
                v2 = "layer-max loses the CKA trend on the BLOOMZ axis"
            elif tr_c >= 0.60:
                v2 = "no loss"
            else:
                v2 = "intermediate"
            P["reading"][cka] = {"delta_cka": d_c, "delta_mknn": d_m, "delta_diff": d_c - d_m,
                                 "delta_diff_ci": boot["ci"], "r_after_gap_mknn_minus_cka": gap_after,
                                 "verdict_arh_style": verdict, "Tr_unc_cka": tr_c, "Tr_unc_mknn": tr_m,
                                 "verdict_s12_axis": v2}
        # ---- 7 A3: decomposition (full subset)
        for met in METRICS:
            P["decomp"][met] = {}
            for col in ("s_obs", "shift", "s_cal", "sel_entry_thr", "sel_entry_mean", "uncensored", "c_alpha",
                        "inflation", "null_mean", "entry_c_alpha_at_argmax"):
                G = grid_of(df, pre, met, "full", col)
                P["decomp"][met][col] = trend_block(G, f"decomp|{pre}|{met}|{col}")
        # ---- 7 A1: selection
        for met in METRICS:
            sub = df[(df.pre == pre) & (df.metric == met) & (df.subset == "full")]
            GA = grid_of(df, pre, met, "full", "rel_A")
            GB = grid_of(df, pre, met, "full", "rel_B")
            sel = {"rel_A_mean": float(sub.rel_A.mean()), "rel_B_mean": float(sub.rel_B.mean()),
                   "rel_A_by_lm": {lm: float(sub[sub.lm == lm].rel_A.mean()) for lm, _ in LC.LMS},
                   "rel_A_median_by_lm": {lm: float(sub[sub.lm == lm].rel_A.median()) for lm, _ in LC.LMS},
                   "argmax_A_mode_by_lm": {lm: int(sub[sub.lm == lm].argmax_A.mode().iloc[0]) for lm, _ in LC.LMS},
                   "rel_B_by_vit": {v: float(sub[sub.vit == v].rel_B.mean()) for v, _ in LC.VITS},
                   "argmax_B_mode_by_vit": {v: int(sub[sub.vit == v].argmax_B.mode().iloc[0]) for v, _ in LC.VITS},
                   "frac_lm_embedding_selected": float(np.mean(sub.argmax_A == 0)),
                   "frac_lm_last_selected": float(np.mean(sub.argmax_A == sub.L_A - 1)),
                   "frac_vit_last_selected": float(np.mean(sub.argmax_B == sub.L_B - 1)),
                   "n_tied_max_gt1": int(np.sum(sub.n_tied_max > 1)),
                   "spearman_relA_vs_arh_rank": spearmanr(sub.rel_A, sub.arh_rank).statistic,
                   "spearman_relB_vs_vit_params": spearmanr(sub.rel_B, sub.vit_params).statistic,
                   "tr_lm_relA": tr_lm(GA, f"sel|{pre}|{met}|A"), "tr_vis_relB": tr_vis(GB)}
            # null maxima locations
            nA, nB = [], []
            for lm, _ in LC.LMS:
                for vit, _ in LC.VITS:
                    z = np.load(os.path.join(LC.RES, "pairs", pre, f"{lm}__{vit}.npz"))
                    LA, LB = LC.n_stored(lm), LC.n_stored(vit)
                    nA.append(z[f"{met}.full_null_argmax_A"] / (LA - 1))
                    nB.append(z[f"{met}.full_null_argmax_B"] / (LB - 1))
            nA, nB = np.concatenate(nA), np.concatenate(nB)
            sel["null_rel_A_mean"] = float(nA.mean())
            sel["null_rel_B_mean"] = float(nB.mean())
            sel["null_frac_lm_embedding"] = float(np.mean(nA == 0))
            sel["null_rel_A_hist"] = np.histogram(nA, bins=10, range=(0, 1))[0].tolist()
            sel["null_rel_B_hist"] = np.histogram(nB, bins=10, range=(0, 1))[0].tolist()
            sel["obs_rel_A_hist"] = np.histogram(sub.rel_A, bins=10, range=(0, 1))[0].tolist()
            sel["obs_rel_B_hist"] = np.histogram(sub.rel_B, bins=10, range=(0, 1))[0].tolist()
            P["selection"][met] = sel
        a = df[(df.pre == pre) & (df.metric == "mknn") & (df.subset == "full")].set_index(["lm", "vit"])
        b = df[(df.pre == pre) & (df.metric == "cka_biased") & (df.subset == "full")].set_index(["lm", "vit"])
        b = b.loc[a.index]
        P["selection"]["agreement_cka_vs_mknn"] = {
            "same_pair": float(np.mean((a.argmax_A == b.argmax_A) & (a.argmax_B == b.argmax_B))),
            "mean_abs_drel_A": float(np.mean(np.abs(a.rel_A - b.rel_A))),
            "mean_abs_drel_B": float(np.mean(np.abs(a.rel_B - b.rel_B)))}
        # ---- 7 A2: null vs number of layer pairs
        for met in METRICS:
            sub = df[(df.pre == pre) & (df.metric == met) & (df.subset == "full")]
            npairs = sub.L_A * sub.L_B
            P["null_vs_size"][met] = {
                "spearman_calpha_vs_nlayerpairs": spearmanr(sub.c_alpha, npairs).statistic,
                "spearman_nullmean_vs_nlayerpairs": spearmanr(sub.null_mean, npairs).statistic,
                "spearman_inflation_vs_nlayerpairs": spearmanr(sub.inflation, npairs).statistic,
                "spearman_calpha_vs_LA": spearmanr(sub.c_alpha, sub.L_A).statistic,
                "spearman_calpha_vs_LB": spearmanr(sub.c_alpha, sub.L_B).statistic,
                "pearson_calpha_vs_arh_rank": pearsonr(sub.c_alpha, sub.arh_rank).statistic,
                "inflation_mean": float(sub.inflation.mean()), "inflation_median": float(sub.inflation.median()),
                "calpha_mean": float(sub.c_alpha.mean()), "entry_calpha_at_argmax_mean": float(sub.entry_c_alpha_at_argmax.mean()),
                "frac_s_cal_zero": float(np.mean(sub.s_cal <= 0)), "frac_p_le_0.05": float(np.mean(sub.p_value <= 0.05)),
            }
            curve = {}
            for subset in SUBSETS:
                s2 = df[(df.pre == pre) & (df.metric == met) & (df.subset == subset)]
                curve[subset] = {
                    "mean_n_layer_pairs": float(s2.n_layer_pairs.mean()),
                    "mean_s_obs": float(s2.s_obs.mean()), "mean_c_alpha": float(s2.c_alpha.mean()),
                    "mean_null_mean": float(s2.null_mean.mean()), "mean_s_cal": float(s2.s_cal.mean()),
                    "mean_uncensored": float(s2.uncensored.mean()),
                    "frac_s_cal_zero": float(np.mean(s2.s_cal <= 0)),
                    "arh_r_raw": P["trends"][met][f"{subset}|s_obs"]["arh_r"]["r"],
                    "arh_r_scal": P["trends"][met][f"{subset}|s_cal"]["arh_r"]["r"],
                    "arh_r_unc": P["trends"][met][f"{subset}|uncensored"]["arh_r"]["r"],
                    "tr_unc": P["trends"][met][f"{subset}|uncensored"]["tr_lm"]["Tr"],
                    "tr_unc_p": P["trends"][met][f"{subset}|uncensored"]["tr_lm"]["p_one"],
                    "tr_scal": P["trends"][met][f"{subset}|s_cal"]["tr_lm"]["Tr"],
                    "tr_raw": P["trends"][met][f"{subset}|s_obs"]["tr_lm"]["Tr"],
                    "trvis_unc": P["trends"][met][f"{subset}|uncensored"]["tr_vis"]["Tr"],
                    "trvis_unc_p": P["trends"][met][f"{subset}|uncensored"]["tr_vis"]["p_one"],
                    "trvis_scal": P["trends"][met][f"{subset}|s_cal"]["tr_vis"]["Tr"],
                }
            P["subset_curve"][met] = curve
        # ---- per-LM table (full)
        for met in METRICS:
            sub = df[(df.pre == pre) & (df.metric == met) & (df.subset == "full")]
            P["per_lm"][met] = {lm: {c: float(sub[sub.lm == lm][c].mean()) for c in
                                     ("s_obs", "c_alpha", "s_cal", "uncensored", "null_mean", "entry_c_alpha_at_argmax",
                                      "inflation", "rel_A", "rel_B", "sel_entry_thr")}
                                for lm, _ in LC.LMS}
            P["per_lm"][met]["_family_means"] = {
                fam: {c: float(sub[sub.lm.map(dict(LC.LMS)) == fam][c].mean()) for c in ("s_obs", "s_cal", "c_alpha")}
                for fam in ("BLOOMZ", "OpenLLaMA", "LLaMA")}
        out["per_pre"][pre] = P
    out["arh_table2_reference"] = {"cka_linear": [0.86, 0.45], "cka_unbiased": [0.65, -0.01], "mknn_k10": [0.85, 0.84]}
    out["s12_reference"] = {"Tr_raw_cka_uncensored": {"0.5": 0.70, "1.0": 0.92}, "Tr_mknn_uncensored": {"0.5": 0.84, "1.0": 0.98}}
    with open(os.path.join(LC.RES, "layermax_summary.json"), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    print("written", os.path.join(LC.RES, "layermax_summary.json"))
    for pre in LC.PRES:
        print(pre, json.dumps(LC.jsonable(out["per_pre"][pre]["reading"]), indent=1))


if __name__ == "__main__":
    main()
