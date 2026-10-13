#!/usr/bin/env python
"""Aggregation of the independent ARH layer-max run (role arh-layermax).

Reads results/S1_2_layermax/indep/<pre>/pairs/*.npz (full observed and null
L_A x L_B matrices) and writes results/S1_2_layermax/indep/layermax_indep_summary.json.

Statistics (ANALYSIS_PLAN_LAYERMAX.md v1 sections 5-7, adopted by PLAN_INDEP.md):
  * per pair / metric / layer subset: T = max, T_k = max under pi_k, SPEC-5 summary
  * ARH-style trend: Pearson r across the 136 pairs of score vs LLM rank (ARH order),
    exact one-sided p over all 8! LM relabellings, ViT-bootstrap CI
  * S1.2-style trend: Tr = mean over ViTs of within-BLOOMZ Spearman (S1.2 code,
    analysis/s12/aggregate.py::lm_trend, read-only import), exact p over 120 relabellings
  * vision-axis trend (S1.2 vision_trend)
  * selection (argmax depths), null inflation (subset maxima), counterfactuals
"""
from __future__ import annotations

import os

_T = os.environ.get("LMX_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

import lmx_common as L  # noqa: E402
from lmx_common import C, cal  # noqa: E402
from hrhv2 import util  # noqa: E402

sys.path.insert(0, os.path.join(L.PROJ, "analysis", "s12"))
import aggregate as S12  # noqa: E402  (S1.2 trend code, read-only import)

METRICS = ("mknn", "cka_b", "cka_d")
NB = 2000
GRID_NS = (2, 3, 5, 9, 17)
NL, NV = len(L.LM_STEMS), len(L.VIT_STEMS)


def rhu(x):
    return int(math.floor(x + 0.5))


def subsets(lm, vit, LA, LB):
    S = {"full": (np.arange(LA), np.arange(LB))}
    for n in GRID_NS:
        ra = np.unique([rhu(j / (n - 1) * (LA - 1)) for j in range(n)])
        rb = np.unique([rhu(j / (n - 1) * (LB - 1)) for j in range(n)])
        S[f"grid{n}"] = (ra, rb)
    S["lmall_x_vitlast"] = (np.arange(LA), np.array([LB - 1]))
    S["lmlast_x_vitall"] = (np.array([LA - 1]), np.arange(LB))
    for r in (0.5, 1.0):
        S[f"fixed_r{r}"] = (np.array([C.layer_index(lm, r)]), np.array([C.layer_index(vit, r)]))
    return S


def pair_stats(z, lm, vit):
    m, k = L.M, L.KNN
    obs = {"mknn": z["obs_cnt"] / (m * k), "cka_b": z["obs_cka_b"], "cka_d": z["obs_cka_d"]}
    null = {"mknn": z["null_cnt"].astype(np.float64) / (m * k), "cka_b": z["null_cka_b"], "cka_d": z["null_cka_d"]}
    LA, LB = obs["mknn"].shape
    subs = subsets(lm, vit, LA, LB)
    out = {"LA": LA, "LB": LB, "K": int(z["K"])}
    for met in METRICS:
        S, N = obs[met], null[met]
        res = {}
        for name, (ra, rb) in subs.items():
            Ssub = S[np.ix_(ra, rb)]
            Nsub = N[:, ra][:, :, rb]
            T = float(Ssub.max())
            Tk = Nsub.reshape(Nsub.shape[0], -1).max(axis=1)
            sm = L.summarize(T, Tk)
            ia, ib = np.unravel_index(int(np.argmax(Ssub)), Ssub.shape)
            d = {f: sm[f] for f in ("s_obs", "c_alpha", "s_cal", "p_value", "null_mean", "null_sd", "uncensored")}
            d.update({"a": int(ra[ia]), "b": int(rb[ib]), "n_entries": int(Ssub.size),
                      "n_tied_max": int(np.sum(Ssub == T))})
            res[name] = d
        # selected-entry calibration (selection kept, max inflation removed)
        a, b = res["full"]["a"], res["full"]["b"]
        se = L.summarize(float(S[a, b]), N[:, a, b])
        res["full"].update({"c_entry": se["c_alpha"], "null_mean_entry": se["null_mean"], "s_cal_entry": se["s_cal"],
                            "rel_a": a / (LA - 1), "rel_b": b / (LB - 1)})
        # null maxima locations (full grid)
        flat = N.reshape(N.shape[0], -1).argmax(axis=1)
        na, nb = np.unravel_index(flat, (LA, LB))
        res["full"].update({"null_argmax_rel_a_mean": float(np.mean(na / (LA - 1))),
                            "null_argmax_rel_b_mean": float(np.mean(nb / (LB - 1))),
                            "null_argmax_frac_same_as_obs": float(np.mean((na == a) & (nb == b)))})
        # mean over all entries of the per-entry null mean (baseline of a single entry)
        res["full"]["entry_null_mean_avg"] = float(N.mean())
        out[met] = res
    # do CKA and mKNN select the same pair?
    out["sel_agree_cka_b_mknn"] = bool(out["cka_b"]["full"]["a"] == out["mknn"]["full"]["a"]
                                       and out["cka_b"]["full"]["b"] == out["mknn"]["full"]["b"])
    return out


# ----------------------------------------------------------------------------
# trend statistics
# ----------------------------------------------------------------------------

LM_PERMS = np.array(list(itertools.permutations(range(NL))))       # 40320 x 8


def arh_r(G: np.ndarray, x: np.ndarray, key: str, boot: bool = True) -> dict:
    """Pearson r across all 136 pairs between score G[i, j] and LM covariate x[i];
    exact one-sided p over all 8! relabellings of the LMs; ViT-bootstrap CI."""
    G = np.asarray(G, dtype=np.float64)
    if not np.all(np.isfinite(G)):
        return {"r": None}
    xc = x - x.mean()
    Gc = G - G.mean()
    den = math.sqrt(NV * float(np.sum(xc ** 2))) * math.sqrt(float(np.sum(Gc ** 2)))
    if den == 0:
        return {"r": None, "note": "constant score"}
    R = Gc.sum(axis=1)
    r_obs = float(np.dot(xc, R) / den)
    r_perm = (xc[LM_PERMS] @ R) / den
    out = {"r": r_obs, "p_one": float(np.mean(r_perm >= r_obs - 1e-12)),
           "p_two": float(np.mean(np.abs(r_perm) >= abs(r_obs) - 1e-12))}
    if boot:
        rng = util.rng_from_key(f"S1.2-layermax|indep|boot|{key}")
        vals = np.empty(NB)
        for t in range(NB):
            jj = rng.integers(0, NV, NV)
            g = G[:, jj]
            gc = g - g.mean()
            dd = math.sqrt(NV * float(np.sum(xc ** 2))) * math.sqrt(float(np.sum(gc ** 2)))
            vals[t] = np.dot(xc, gc.sum(axis=1)) / dd if dd > 0 else np.nan
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        out["ci"] = [float(lo), float(hi)]
    return out


def tr_s12(G: np.ndarray, key: str = "lmx-indep") -> dict:
    t = S12.lm_trend(np.asarray(G, dtype=np.float64))
    rho = np.array([np.nan if v is None else v for v in t["rho_per_vit"]])
    est, lo, hi = S12.boot_vits(rho, "lmx-indep|" + key)
    return {"Tr": t["Tr"], "p_one": t["p_one"], "p_two": t["p_two"], "n_vit_used": t["n_vit_used"],
            "ci": [lo, hi], "openllama_frac_larger_higher": t["openllama_frac_larger_higher"]}


def vis_s12(G):
    v = S12.vision_trend(np.asarray(G, dtype=np.float64))
    return {k: v[k] for k in ("Tr", "p_one", "p_two", "n_cells")}


COVARIATES = {
    "arh_rank": np.array([L.ARH_RANK[s] for s in L.LM_STEMS], dtype=np.float64),
    "dense_rank": rankdata([L.ARH_RANK[s] for s in L.LM_STEMS]).astype(np.float64),
    "log_params": np.log(np.array([C.n_params(s) for s in L.LM_STEMS], dtype=np.float64)),
}


def all_trends(G, key, boot=True, covs=("arh_rank", "dense_rank", "log_params")):
    out = {"arh": {c: arh_r(G, COVARIATES[c], f"{key}|{c}", boot=(boot and c == "arh_rank")) for c in covs},
           "s12": tr_s12(G, key), "vision": vis_s12(G), "mean": float(np.mean(G))}
    return out


def grid_of(P, met, sub, field):
    G = np.empty((NL, NV))
    for i, lm in enumerate(L.LM_STEMS):
        for j, vit in enumerate(L.VIT_STEMS):
            G[i, j] = P[(lm, vit)][met][sub][field]
    return G


def reading_arh(rb_cka, ra_cka, rb_k, ra_k):
    d = (rb_cka - ra_cka) - (rb_k - ra_k)
    if d >= 0.20 and ra_cka <= ra_k - 0.20:
        v = "ARH pattern reappears"
    elif d <= 0.10 or ra_cka >= ra_k - 0.10:
        v = "ARH pattern absent"
    else:
        v = "partial"
    return {"dDelta": d, "r_after_gap": ra_k - ra_cka, "verdict": v}


def reading_s12(tr_cka, tr_k):
    if tr_cka <= 0.40 and tr_k >= 0.60:
        return "layer-max loses the CKA trend on the BLOOMZ axis"
    if tr_cka >= 0.60:
        return "no loss"
    return "intermediate"


def main():
    t0 = time.time()
    summary = {"plan": "analysis/s12_layermax/indep/PLAN_INDEP.md", "K": L.K_PERM, "alpha": L.ALPHA,
               "lm_order": L.LM_STEMS, "vit_order": L.VIT_STEMS, "arh_rank": L.ARH_RANK, "per_pre": {}}
    for pre in L.PRES:
        P = {}
        for lm in L.LM_STEMS:
            for vit in L.VIT_STEMS:
                z = np.load(os.path.join(L.OUT, pre, "pairs", f"{lm}__{vit}.npz"))
                assert int(z["K"]) == L.K_PERM and str(z["key"]) == L.perm_key(lm, vit)
                P[(lm, vit)] = pair_stats(z, lm, vit)
        print(f"[{time.time() - t0:.0f}s] {pre}: pair stats done", flush=True)
        R = {"pairs": {f"{lm}|{vit}": P[(lm, vit)] for (lm, vit) in P}}
        nlp = np.array([[P[(lm, vit)]["LA"] * P[(lm, vit)]["LB"] for vit in L.VIT_STEMS] for lm in L.LM_STEMS], float)
        R["n_layer_pairs_grid"] = nlp.tolist()
        T = {}
        for met in METRICS:
            Tm = {}
            subs = list(P[(L.LM_STEMS[0], L.VIT_STEMS[0])][met].keys())
            for sub in subs:
                g = {f: grid_of(P, met, sub, f) for f in ("s_obs", "s_cal", "uncensored", "c_alpha", "null_mean", "p_value")}
                boot = sub in ("full", "fixed_r0.5", "fixed_r1.0")
                Tm[sub] = {
                    "before_raw": all_trends(g["s_obs"], f"{pre}|{met}|{sub}|raw", boot),
                    "after_scal": all_trends(g["s_cal"], f"{pre}|{met}|{sub}|scal", boot),
                    "uncensored": all_trends(g["uncensored"], f"{pre}|{met}|{sub}|unc", boot),
                    "c_alpha": all_trends(g["c_alpha"], f"{pre}|{met}|{sub}|calpha", False),
                    "null_mean": all_trends(g["null_mean"], f"{pre}|{met}|{sub}|nullmean", False),
                    "frac_s_cal_pos": float(np.mean(g["s_cal"] > 0)),
                    "frac_p_le_0.05": float(np.mean(g["p_value"] <= 0.05)),
                    "mean_c_alpha": float(np.mean(g["c_alpha"])), "mean_null_mean": float(np.mean(g["null_mean"])),
                    "mean_s_obs": float(np.mean(g["s_obs"])), "mean_s_cal": float(np.mean(g["s_cal"])),
                    "mean_entries": float(np.mean(grid_of(P, met, sub, "n_entries"))),
                }
                if sub == "full":
                    gT, gc = g["s_obs"], g["c_alpha"]
                    ge, gn = grid_of(P, met, "full", "c_entry"), grid_of(P, met, "full", "null_mean_entry")
                    cf = {
                        "1_T_raw": gT,
                        "2_T_minus_calpha": gT - gc,
                        "3_s_cal_ARH": g["s_cal"],
                        "3b_uncapped_ratio": (gT - gc) / (1.0 - gc),
                        "4_T_minus_c_entry": gT - ge,
                        "5_T_minus_entry_null_mean": gT - gn,
                        "6_uncensored_T_minus_mean_Tk": g["uncensored"],
                        "c_alpha": gc,
                        "inflation_calpha_minus_c_entry": gc - ge,
                        "mean_Tk": g["null_mean"],
                    }
                    Tm["counterfactual"] = {k: {"arh_r": arh_r(v, COVARIATES["arh_rank"], f"{pre}|{met}|cf|{k}", False),
                                                "s12": {kk: vv for kk, vv in tr_s12(v).items() if kk != "ci"},
                                                "mean": float(np.mean(v))} for k, v in cf.items()}
                    ra, rb = grid_of(P, met, "full", "rel_a"), grid_of(P, met, "full", "rel_b")
                    aa, bb = grid_of(P, met, "full", "a"), grid_of(P, met, "full", "b")
                    Tm["selection"] = {
                        "rel_a_grid": ra.tolist(), "rel_b_grid": rb.tolist(), "a_grid": aa.tolist(), "b_grid": bb.tolist(),
                        "rel_a_by_lm": {lm: float(ra[i].mean()) for i, lm in enumerate(L.LM_STEMS)},
                        "rel_b_by_lm": {lm: float(rb[i].mean()) for i, lm in enumerate(L.LM_STEMS)},
                        "a_by_lm_mode": {lm: int(np.bincount(aa[i].astype(int)).argmax()) for i, lm in enumerate(L.LM_STEMS)},
                        "rel_b_by_vit": {v: float(rb[:, j].mean()) for j, v in enumerate(L.VIT_STEMS)},
                        "rel_a_by_vit": {v: float(ra[:, j].mean()) for j, v in enumerate(L.VIT_STEMS)},
                        "rel_a_trend": {"arh_r": arh_r(ra, COVARIATES["arh_rank"], "sel", False),
                                        "s12": {kk: vv for kk, vv in tr_s12(ra).items() if kk != "ci"}},
                        "rel_b_trend": {"arh_r": arh_r(rb, COVARIATES["arh_rank"], "selb", False),
                                        "s12": {kk: vv for kk, vv in tr_s12(rb).items() if kk != "ci"}},
                        "n_tied_max_total": int(grid_of(P, met, "full", "n_tied_max").sum() - NL * NV),
                        "null_argmax_rel_a_mean": float(np.mean(grid_of(P, met, "full", "null_argmax_rel_a_mean"))),
                        "null_argmax_rel_b_mean": float(np.mean(grid_of(P, met, "full", "null_argmax_rel_b_mean"))),
                        "null_argmax_same_as_obs_mean": float(np.mean(grid_of(P, met, "full", "null_argmax_frac_same_as_obs"))),
                    }
                    Tm["inflation"] = {
                        "spearman_calpha_vs_nlayerpairs": float(spearmanr(gc.ravel(), nlp.ravel()).correlation),
                        "spearman_meanTk_vs_nlayerpairs": float(spearmanr(g["null_mean"].ravel(), nlp.ravel()).correlation),
                        "spearman_inflation_vs_nlayerpairs": float(spearmanr((gc - ge).ravel(), nlp.ravel()).correlation),
                        "mean_inflation_calpha_minus_c_entry": float(np.mean(gc - ge)),
                        "mean_c_entry": float(np.mean(ge)),
                        "inflation_by_lm": {lm: float((gc - ge)[i].mean()) for i, lm in enumerate(L.LM_STEMS)},
                        "calpha_by_lm": {lm: float(gc[i].mean()) for i, lm in enumerate(L.LM_STEMS)},
                        "T_by_lm": {lm: float(gT[i].mean()) for i, lm in enumerate(L.LM_STEMS)},
                        "s_cal_by_lm": {lm: float(g["s_cal"][i].mean()) for i, lm in enumerate(L.LM_STEMS)},
                        "unc_by_lm": {lm: float(g["uncensored"][i].mean()) for i, lm in enumerate(L.LM_STEMS)},
                    }
                    Tm["grids"] = {f: g[f].tolist() for f in ("s_obs", "s_cal", "uncensored", "c_alpha", "null_mean", "p_value")}
            T[met] = Tm
            print(f"[{time.time() - t0:.0f}s] {pre} {met} trends done", flush=True)
        R["trends"] = T
        # S1.2 stored fixed-depth values (their own permutations), same statistics
        import pandas as pd
        panel = pd.read_parquet(os.path.join(C.RES, "panel_main.parquet"))
        fx = {}
        for r in (0.5, 1.0):
            for met, col in (("mknn", "cal.s_N_k10"), ("cka_b", "cal.cka_biased"), ("cka_d", "cal.cka_debiased")):
                if f"{col}.s_obs" not in panel.columns:
                    continue
                fx[f"{met}|r{r}"] = {
                    "before_raw": all_trends(S12.grid(panel, f"{col}.s_obs", r, pre), f"{pre}|s12|{met}|{r}|raw", False),
                    "after_scal": all_trends(S12.grid(panel, f"{col}.s_cal", r, pre), f"{pre}|s12|{met}|{r}|scal", False),
                    "uncensored": all_trends(S12.grid(panel, f"{col}.uncensored", r, pre), f"{pre}|s12|{met}|{r}|unc", False)}
        R["s12_stored_fixed_depth"] = fx
        # readings (PLAN section 6), layer-max full grid
        rd = {}
        for cka in ("cka_b", "cka_d"):
            tb = T[cka]["full"]
            tk = T["mknn"]["full"]
            rd[cka] = {
                "arh": reading_arh(tb["before_raw"]["arh"]["arh_rank"]["r"], tb["after_scal"]["arh"]["arh_rank"]["r"] or 0.0,
                                   tk["before_raw"]["arh"]["arh_rank"]["r"], tk["after_scal"]["arh"]["arh_rank"]["r"] or 0.0),
                "s12_axis": reading_s12(tb["uncensored"]["s12"]["Tr"], tk["uncensored"]["s12"]["Tr"]),
            }
            for fd in ("fixed_r0.5", "fixed_r1.0"):
                fb, fk = T[cka][fd], T["mknn"][fd]
                rd[cka][f"arh_{fd}"] = reading_arh(fb["before_raw"]["arh"]["arh_rank"]["r"], fb["after_scal"]["arh"]["arh_rank"]["r"] or 0.0,
                                                   fk["before_raw"]["arh"]["arh_rank"]["r"], fk["after_scal"]["arh"]["arh_rank"]["r"] or 0.0)
                rd[cka][f"s12_axis_{fd}"] = reading_s12(fb["uncensored"]["s12"]["Tr"], fk["uncensored"]["s12"]["Tr"])
        R["readings"] = rd
        summary["per_pre"][pre] = R

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        return L.jsonable(o)
    out = os.path.join(L.OUT, "layermax_indep_summary.json")
    json.dump(clean(summary), open(out, "w"), indent=1)
    print(f"wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
