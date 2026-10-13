"""E0 evaluation: slopes, elasticities, levels, the confirmatory labels P1, P2b, P2c, and every pre-specified
secondary and descriptive analysis (ANALYSIS_PLAN_E0.md v1.0 §6-§9, §14).

    venv/bin/python -I analysis/e0_sentence/e0_evaluate.py            # step 6, after the new-pair obs step

Uses the byte-identical S1.1 functions draw_slopes, mean_slopes, drawmean_cells (s11copy/s11v11_evaluate.py).
Refuses to run unless the guard conditions of e0_obs.guard_ok() hold. Writes results/E0_sentence/eval/
E0_{cells,drawslopes,slopes,elasticity,levels}_<arm>.parquet, VERDICTS_E0.json, eval/E0_tables.md,
eval/E0_eval_summary.json. Never overwrites.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e0_common as E  # noqa: E402
import e0_rules as R  # noqa: E402
import s11v11_evaluate as SE  # noqa: E402  (byte-identical copy)

KS = (1, 10, 100)
M_GRID = (256, 512, 1024, 2048, 4096, 8192, 16384)
S11_INPUTS = {  # frozen S1.1 outputs used here (sha256 checked against freeze_constants.json inputs)
    "clip": ("results/S1_1/v11/S1_1_v11_cells_clip.parquet", "results/S1_1/v11/S1_1_v11_slopes_clip.parquet"),
    "l2": ("results/S1_1/v11/S1_1_v11_cells_l2.parquet", "results/S1_1/v11/S1_1_v11_slopes_l2.parquet"),
}
TABLE_COLS = ("pair", "cls", "primary", "m", "k", "obs", "obs_allH2", "obs_clean", "n_anchors", "n_draws", "eps",
              "chance", "obs_draw_sd")


# ---------------------------------------------------------------------------------------------- per-arm tables
def arm_tables(obs: pd.DataFrame):
    """(draw-level df, draw slopes, mean slopes, draw-mean cells) exactly as S1.1's evaluate_config for the
    observed columns (eps_d and obs_cc as in s11v11_evaluate.build_cells)."""
    df = obs.copy()
    df["eps_d"] = 0.5 / (df["n_anchors"] * df["k"])
    df["obs_cc"] = (df["obs"] - df["chance"]) / (1.0 - df["chance"])
    ds = SE.draw_slopes(df, [])
    S = SE.mean_slopes(ds)
    c = SE.drawmean_cells(df, [])
    return df, ds, S, c


def elasticities(c: pd.DataFrame, ks, col: str = "obs") -> pd.DataFrame:
    """OLS of ln max(s_bar, eps_bar) = c + e_k ln k + e_m ln m over cells with k in ks (plan §6)."""
    out = []
    for pair, g in c[c["k"].isin(ks)].groupby("pair", sort=True):
        X = np.column_stack([np.ones(len(g)), np.log(g["k"].to_numpy(float)), np.log(g["m"].to_numpy(float))])
        y = np.log(np.maximum(g[col].to_numpy(float), g["eps"].to_numpy(float)))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        out.append({"pair": pair, "cls": g["cls"].iloc[0], "e_k": float(beta[1]), "e_m": float(beta[2]),
                    "sum": float(beta[1] + beta[2]), "rmse": float(np.sqrt(np.mean(r ** 2))), "n_cells": int(len(g))})
    return pd.DataFrame(out)


def levels(c: pd.DataFrame, col: str = "obs") -> pd.DataFrame:
    """L(k, m) for every grid m, floored; Schnaus-matched L(k, 10000) and fixed-m ratios (plan §6)."""
    rows = []
    for (pair, k), g in c.groupby(["pair", "k"], sort=True):
        g = g.set_index("m")
        r = {"pair": pair, "k": int(k), "cls": g["cls"].iloc[0]}
        for m in M_GRID:
            if m in g.index:
                r[f"L_m{m}"] = float(max(g.loc[m, col], g.loc[m, "eps"]))
        if 8192 in g.index and 16384 in g.index:
            l8, l16 = math.log(r["L_m8192"]), math.log(r["L_m16384"])
            r["L_m10000"] = math.exp(l8 + (math.log(10000) - math.log(8192)) / math.log(2) * (l16 - l8))
        rows.append(r)
    L = pd.DataFrame(rows)
    w = L.pivot(index="pair", columns="k")
    ratio = {}
    for m in (8192, 16384):
        if (f"L_m{m}", 100) in w.columns and (f"L_m{m}", 10) in w.columns:
            ratio[m] = (w[(f"L_m{m}", 100)] / w[(f"L_m{m}", 10)]).to_dict()
    for m, d in ratio.items():
        L.loc[L["k"] == 10, f"ratio_k100_over_k10_m{m}"] = L.loc[L["k"] == 10, "pair"].map(d)
    return L


def x4_slopes(c: pd.DataFrame) -> pd.DataFrame:
    """EXPLORATORY X4: slope of ln max(1 - s_bar, eps_bar) on ln m at fixed k."""
    out = []
    for (pair, k), g in c.groupby(["pair", "k"], sort=True):
        g = g.sort_values("m")
        y = np.log(np.maximum(1.0 - g["obs"].to_numpy(float), g["eps"].to_numpy(float)))
        out.append({"pair": pair, "k": int(k), "slope_log_one_minus_s": R._ols_slope(np.log(g["m"].to_numpy(float)), y)})
    return pd.DataFrame(out)


def pair_meta(pairs):
    return {p["pair"]: p for p in pairs}


# ---------------------------------------------------------------------------------------------- rule inputs
def level_slope(c, S, pairs, k=10, mref=1024, col="obs"):
    """(L floored, b) arrays in the order of ``pairs``."""
    cc = c[(c["k"] == k) & (c["m"] == mref)].set_index("pair")
    scol = {"obs": "slope_obs", "obs_allH2": "slope_obs_allH2", "obs_clean": "slope_obs_clean"}[col]
    if col == "obs_allH2":
        eps = 0.5 / (cc["n_draws"] * cc["n_allH2"] * k)
    else:
        eps = cc["eps"]
    Lv = np.maximum(cc[col], eps)
    Ss = S[S["k"] == k].set_index("pair")[scol]
    names = [p["pair"] for p in pairs]
    return Lv.reindex(names).to_numpy(float), Ss.reindex(names).to_numpy(float)


def run_p2b(c, S, pairs, line, comp, k=10, mref=1024, col="obs"):
    L, b = level_slope(c, S, pairs, k, mref, col)
    vits = [p["vit"] for p in pairs]
    encs = [p["enc"] for p in pairs]
    names = [p["pair"] for p in pairs]
    r = R.p2b_rule(L, b, vits, encs, line, comp, pairs=names)
    r["leave_one_out"] = R.p2b_leave_one_out(L, b, vits, encs, line, comp)
    r["k"], r["m_ref"], r["column"] = k, mref, col
    return r


def run_p2c(arm_tabs, pairs, k=10, mref=1024, col="obs"):
    Ln = np.empty((len(pairs), 5))
    bn = np.empty((len(pairs), 5))
    for n, arm in enumerate(E.P2C_ARMS):
        c, S = arm_tabs[arm]["cells"], arm_tabs[arm]["slopes"]
        Ln[:, n], bn[:, n] = level_slope(c, S, pairs, k, mref, col)
    gam, dln = R.gamma_pairs(Ln, bn)
    encs = [p["enc"] for p in pairs]
    names = [p["pair"] for p in pairs]
    r = R.p2c_rule(gam, dln, encs, pairs=names)
    r["leave_encoder_out_median_gamma"] = R.p2c_leave_one_out(gam, encs)
    r["ladder"] = {names[j]: {"L": Ln[j].tolist(), "b": bn[j].tolist()} for j in range(len(names))}
    r["k"], r["m_ref"], r["column"] = k, mref, col
    return r


def run_p1(el_k10_100, pairs):
    e = el_k10_100.set_index("pair").reindex([p["pair"] for p in pairs])
    r = R.p1_rule(e["sum"].to_numpy(float), e["e_m"].to_numpy(float), pairs=list(e.index))
    r["per_pair"] = e.reset_index()[["pair", "e_k", "e_m", "sum", "rmse"]].to_dict("records")
    r["per_encoder_median_sum"] = {enc: float(np.median([x["sum"] for x, p in zip(r["per_pair"], pairs)
                                                         if p["enc"] == enc])) for enc in sorted({p["enc"] for p in pairs})}
    return r


# ---------------------------------------------------------------------------------------------- old pairs (S1.1 frozen)
def load_s11(pre: str, expected: dict):
    cpath, spath = (os.path.join(E.PROJECT, p) for p in S11_INPUTS[pre])
    for p in (cpath, spath):
        rel = os.path.relpath(p, E.PROJECT)
        if E.sha256_file(p) != expected[rel]:
            raise SystemExit(f"REFUSED: {rel} differs from the frozen S1.1 output used by the freeze role")
    c = pd.read_parquet(cpath)
    S = pd.read_parquet(spath)
    return c[c["primary"]].copy(), S[S["primary"]].copy()


def old_vl_pairs():
    out = []
    for p in E.old_pairs():
        if p["cls"] == "cross-modal":
            q = dict(p)
            q["vit"], q["enc"] = p["a"], p["b"]
            out.append(q)
    return out


# ---------------------------------------------------------------------------------------------- P2 descriptive
def p2_descriptive(cP, SP, c11, S11, new_pairs):
    vits = sorted(E.VITS)
    Lm = np.empty((5, 6))
    Sm = np.empty((5, 6))
    oldL, oldb = level_slope(c11, S11, old_vl_pairs())
    old = {p["pair"]: (l, b) for p, l, b in zip(old_vl_pairs(), oldL, oldb)}
    newL, newb = level_slope(cP, SP, new_pairs)
    new = {p["pair"]: (l, b) for p, l, b in zip(new_pairs, newL, newb)}
    both = {**old, **new}
    for i, v in enumerate(vits):
        for j, t in enumerate(E.TEXT_ORDER_P2):
            Lm[i, j], Sm[i, j] = both[f"{v}__{t}"]
    return {"within_vit_permutation": R.p2_within_vit_permutation(Lm, Sm),
            "text_model_level": R.p2_text_model_level(Lm, Sm),
            "vit_order": vits, "text_order": list(E.TEXT_ORDER_P2), "levels": Lm.tolist(), "slopes": Sm.tolist(),
            "label": "DESCRIPTIVE (plan §7.2; no threshold)"}


# ---------------------------------------------------------------------------------------------- main evaluation
def evaluate(obs_by_arm: dict, s11: dict, ctr_cal: dict | None, write_dir: str | None = None) -> dict:
    """obs_by_arm: arm -> obs DataFrame (new arms and optionally 'ctrold'); s11: pre -> (cells, slopes)."""
    T = {}
    for arm, obs in obs_by_arm.items():
        df, ds, S, c = arm_tables(obs)
        T[arm] = {"df": df, "drawslopes": ds, "slopes": S, "cells": c,
                  "el": {name: elasticities(c, ks) for name, ks in
                         (("k10_100", (10, 100)), ("k1_10_100", (1, 10, 100)), ("k1_10", (1, 10)))},
                  "levels": levels(c), "x4": x4_slopes(c)}
    newP = E.new_pairs("P")
    V = {"plan": "ANALYSIS_PLAN_E0.md v1.0", "created_utc": E.utcnow(), "order": ["P1", "P2b", "P2c"]}
    # ------------------------------------------------ confirmatory
    if "P" in T:
        V["P1"] = run_p1(T["P"]["el"]["k10_100"], newP)
        V["P1"]["k_set"] = list(R.P1_KS)
        V["P1"]["disclosure"] = "k >= 10 restriction chosen after seeing S1.1, 2026-10-10, before any E0 score"
        c, S = T["P"]["cells"], T["P"]["slopes"]
        V["P2b"] = run_p2b(c, S, newP, R.LINES[("clip", 10, 1024)], R.COMPETITOR[("clip", 10)])
        Lp, bp = level_slope(c, S, newP)
        V["P2b"]["wilcoxon_descriptive"] = R.wilcoxon_descriptive(Lp, bp, [p["vit"] for p in newP])
        V["P2b"]["leave_mpnet_out_lambda"] = V["P2b"]["leave_one_out"]["leave_encoder_out"].get("mpnet")
    if all(a in T for a in E.P2C_ARMS):
        V["P2c"] = run_p2c(T, newP)
        V["P2c"]["leave_mpnet_out_median_gamma"] = V["P2c"]["leave_encoder_out_median_gamma"].get("mpnet")
    V["labels"] = {k: V[k]["label"] for k in ("P1", "P2b", "P2c") if k in V}
    # ------------------------------------------------ descriptive P2
    c11, S11 = s11["clip"]
    if "P" in T:
        V["P2_descriptive"] = p2_descriptive(T["P"]["cells"], T["P"]["slopes"], c11, S11, newP)
    # ------------------------------------------------ secondary (plan §7.5), same rules, no confirmatory weight
    sec = {"P1": {}, "P2b": {}, "P2c_other_k": {}}
    for arm in T:
        if arm == E.OLD_ARM:
            pairs = E.old_pairs()
            for p in pairs:
                p.setdefault("enc", p["b"])
            sec["P1"]["ctrold"] = run_p1(T[arm]["el"]["k10_100"], pairs)
        elif arm != "P" and arm in E.ARMS:
            sec["P1"][arm] = run_p1(T[arm]["el"]["k10_100"], E.new_pairs(arm))
    if "P" in T:
        c, S = T["P"]["cells"], T["P"]["slopes"]
        for k in (1, 100):
            sec["P2b"][f"P|k{k}"] = run_p2b(c, S, newP, R.LINES[("clip", k, 1024)], R.COMPETITOR[("clip", k)], k=k)
        sec["P2b"]["P|mref2048"] = run_p2b(c, S, newP, R.LINES[("clip", 10, 2048)], R.COMPETITOR[("clip", 10)], mref=2048)
        for col in ("obs_allH2", "obs_clean"):
            sec["P2b"][f"P|{col}"] = run_p2b(c, S, newP, R.LINES[("clip", 10, 1024)], R.COMPETITOR[("clip", 10)], col=col)
    for arm in ("A", "Aall", "Pdd"):
        if arm in T:
            sec["P2b"][arm] = run_p2b(T[arm]["cells"], T[arm]["slopes"], E.new_pairs(arm), R.LINES[("clip", 10, 1024)],
                                      R.COMPETITOR[("clip", 10)])
    if "L" in T:
        sec["P2b"]["L"] = run_p2b(T["L"]["cells"], T["L"]["slopes"], E.new_pairs("L"), R.LINES[("l2", 10, 1024)],
                                  R.COMPETITOR[("l2", 10)])
    if ctr_cal is not None:
        for arm in ("C", "CA", "CAall"):
            if arm in T:
                line = tuple(ctr_cal["lines"]["k10|m1024"][x] for x in ("a", "b", "resid_sd"))
                sec["P2b"][arm] = run_p2b(T[arm]["cells"], T[arm]["slopes"], E.new_pairs(arm), line,
                                          ctr_cal["competitor"]["k10"])
    if all(a in T for a in E.P2C_ARMS):
        for k in (1, 100):
            r = run_p2c(T, newP, k=k)
            sec["P2c_other_k"][f"k{k}"] = {"median_gamma": r["median_gamma"], "gamma_e": r["gamma_e"],
                                           "label_if_rule_applied": r["label"]}
        for col in ("obs_allH2", "obs_clean"):
            r = run_p2c(T, newP, col=col)
            sec["P2c_other_k"][col] = {"median_gamma": r["median_gamma"], "gamma_e": r["gamma_e"],
                                       "label_if_rule_applied": r["label"]}
    V["secondary"] = sec
    V["descriptive"] = descriptive(T, s11, ctr_cal)
    if write_dir is not None:
        write_outputs(T, V, write_dir)
    return V


def class_summary(c, S, el, Lv, label):
    out = {}
    for cl, g in c.groupby("cls"):
        pairs = sorted(g["pair"].unique())
        L1024 = Lv[(Lv["k"] == 10) & Lv["pair"].isin(pairs)]["L_m1024"]
        L8192 = Lv[(Lv["k"] == 100) & Lv["pair"].isin(pairs)]["L_m8192"]
        s10 = S[(S["k"] == 10) & S["pair"].isin(pairs)]["slope_obs"]
        e = el[el["pair"].isin(pairs)]
        out[cl] = {"n": len(pairs), "L_k10_m1024": [float(L1024.median()), float(L1024.min()), float(L1024.max())],
                   "L_k100_m8192": [float(L8192.median()), float(L8192.min()), float(L8192.max())],
                   "median_b10": float(s10.median()), "median_sum_k10_100": float(e["sum"].median()),
                   "slope_classes": {int(k): S[(S["k"] == k) & S["pair"].isin(pairs)]["slope_class"].value_counts().to_dict()
                                     for k in KS}}
    return {label: out}


def descriptive(T, s11, ctr_cal):
    D = {"P3a_level": {}, "P3b_caption": {}, "P3c_preprocessing": {}, "P3e_encoders": {}, "P3f_ordering": {},
         "P3g_shared_k_over_m": {}, "P3h_schnaus": {}, "P3j_reference": {}, "robustness": {}, "x4_exploratory": {}}
    for arm, t in T.items():
        D["P3a_level"].update(class_summary(t["cells"], t["slopes"], t["el"]["k10_100"], t["levels"], arm))
    for pre, (c, S) in s11.items():
        if pre not in ("clip", "l2"):
            continue
        Lv = levels(c)
        D["P3a_level"].update(class_summary(c, S, elasticities(c, (10, 100)), Lv, f"S1.1-{pre}"))

    def lev(arm, k=10, m=1024):
        Lv = T[arm]["levels"]
        return Lv[Lv["k"] == k].set_index("pair")[f"L_m{m}"]

    def slo(arm, k=10):
        S = T[arm]["slopes"]
        return S[S["k"] == k].set_index("pair")["slope_obs"]
    for a, b, name in (("A", "P", "AVG5-C1|clip"), ("A", "D1", "AVG5-N1|clip"), ("Aall", "A", "ALL-AVG5|clip"),
                       ("CA", "C", "AVG5-C1|ctr"), ("CAall", "CA", "ALL-AVG5|ctr")):
        if a in T and b in T:
            dl = (np.log(lev(a)) - np.log(lev(b)))
            db = slo(a) - slo(b)
            D["P3b_caption"][name] = {"per_pair_dlnL": dl.to_dict(), "per_pair_db10": db.to_dict(),
                                      "median_dlnL": float(dl.median()), "median_db10": float(db.median())}
    for a, b, name in (("C", "P", "ctr-clip|new"), ("L", "P", "l2-clip|new")):
        if a in T and b in T:
            dl = np.log(lev(a)) - np.log(lev(b))
            db = slo(a) - slo(b)
            D["P3c_preprocessing"][name] = {"median_dlnL": float(dl.median()), "median_db10": float(db.median()),
                                            "per_pair_dlnL": dl.to_dict(), "per_pair_db10": db.to_dict()}
    if E.OLD_ARM in T:
        c11, S11 = s11["clip"]
        Lo = levels(c11)
        Lc = T[E.OLD_ARM]["levels"]
        for cl in ("cross-modal", "cross-vision", "seed-seed"):
            pc = [p["pair"] for p in E.old_pairs() if p["cls"] == cl]
            a = Lc[(Lc["k"] == 10) & Lc["pair"].isin(pc)].set_index("pair")["L_m1024"]
            b = Lo[(Lo["k"] == 10) & Lo["pair"].isin(pc)].set_index("pair")["L_m1024"]
            sa = T[E.OLD_ARM]["slopes"]
            sa = sa[(sa["k"] == 10) & sa["pair"].isin(pc)].set_index("pair")["slope_obs"]
            sb = S11[(S11["k"] == 10) & S11["pair"].isin(pc)].set_index("pair")["slope_obs"]
            D["P3c_preprocessing"][f"ctr-clip|old {cl}"] = {"median_dlnL": float((np.log(a) - np.log(b)).median()),
                                                            "median_db10": float((sa - sb).median())}
    if "P" in T:
        newP = E.new_pairs("P")
        for enc in E.ENCODERS:
            pe = [p["pair"] for p in newP if p["enc"] == enc]
            el = T["P"]["el"]["k10_100"]
            D["P3e_encoders"][enc] = {"median_L_k10_m1024": float(lev("P")[pe].median()),
                                      "median_b10": float(slo("P")[pe].median()),
                                      "median_sum_k10_100": float(el[el["pair"].isin(pe)]["sum"].median())}
        c11, S11 = s11["clip"]
        for k in KS:
            med = {cl: float(S11[(S11["k"] == k) & (S11["cls"] == cl)]["slope_obs"].median())
                   for cl in ("seed-seed", "cross-vision", "cross-modal")}
            med[E.NEW_CLASS] = float(slo("P", k).median())
            Lo = levels(c11)
            lv = {cl: float(Lo[(Lo["k"] == k) & (Lo["cls"] == cl)]["L_m1024"].median())
                  for cl in ("seed-seed", "cross-vision", "cross-modal")}
            lv[E.NEW_CLASS] = float(lev("P", k).median())
            D["P3f_ordering"][int(k)] = {"median_slope": med, "median_level_m1024": lv,
                                         "slope_order": sorted(med, key=lambda x: -med[x]),
                                         "level_order": sorted(lv, key=lambda x: -lv[x])}
        D["P3g_shared_k_over_m"] = shared_kom_profile(T["P"]["cells"])
        a, b, _ = R.LINES[("clip", 10, 1024)]
        Lp = lev("P")
        D["P3j_reference"] = {p: {"L": float(l), "b": float(slo("P")[p]), "line": float(min(a + b * math.log(l), 0)),
                                  "N1_own_line": float(R.N1_OWN_LINE[0] + R.N1_OWN_LINE[1] * math.log(l)),
                                  "anchored_power_law": float(math.log(l) / math.log(1024 / 10))}
                              for p, l in Lp.items()}
        D["P3j_reference_within_class_coefficients"] = R.WITHIN_CLASS_K10
    for arm in ("CA", "P", "C", E.OLD_ARM):
        if arm in T:
            Lv = T[arm]["levels"]
            w = Lv[Lv["k"].isin((10, 100))]
            cols = [c for c in w.columns if c in ("L_m10000", "ratio_k100_over_k10_m8192", "ratio_k100_over_k10_m16384")]
            D["P3h_schnaus"][arm] = {f"{r.pair}|k{r.k}": {c: getattr(r, c) for c in cols if not pd.isna(getattr(r, c))}
                                     for r in w.itertuples(index=False)}
    for arm, t in T.items():
        S = t["slopes"]
        D["robustness"][arm] = {"draw_sd_b10": S[S["k"] == 10].set_index("pair")["slope_obs_draw_sd"].to_dict(),
                                "elasticity_k1_10_100": t["el"]["k1_10_100"][["pair", "sum"]].set_index("pair")["sum"].to_dict(),
                                "elasticity_k1_10": t["el"]["k1_10"][["pair", "sum"]].set_index("pair")["sum"].to_dict()}
        D["x4_exploratory"][arm] = t["x4"][t["x4"]["k"] == 10].set_index("pair")["slope_log_one_minus_s"].to_dict()
    if any(a in T for a in E.Q1_ARMS):
        D["Q1_lm_caption_average"] = q1_descriptive(T, s11)
    if "P" in T:
        df = T["P"]["df"]
        per_draw = {}
        for d, g in df.groupby("draw"):
            gg = g.rename(columns={"eps_d": "eps"})
            el = elasticities(gg, (10, 100))
            per_draw[int(d)] = R.p1_rule(el["sum"], el["e_m"])["label"]
        D["robustness"]["P1_per_draw_labels"] = per_draw
        el30 = pd.concat([T["P"]["el"]["k10_100"], elasticities(s11["clip"][0], (10, 100))])
        el30 = el30[el30["cls"].isin(("cross-modal", E.NEW_CLASS))]
        D["corr_ek_minus_em_30VL"] = float(np.corrcoef(el30["e_k"], -el30["e_m"])[0, 1])
    return D


def shared_kom_profile(c: pd.DataFrame) -> dict:
    """P3g: per pair, s_bar(k, m*k/10) / s_bar(10, m) for k in {1, 100} at every grid m of k = 10 whose shared
    m*k/10 lies in [256, 16384], with ln s_bar(k, .) interpolated linearly in ln m between grid points; median over
    pairs (descriptive)."""
    out = {}
    for k in (1, 100):
        for m10 in M_GRID:
            mk = m10 * k / 10
            if not (256 <= mk <= 16384):
                continue
            ratios = []
            for pair, g in c.groupby("pair"):
                gk = g[g["k"] == k].sort_values("m")
                lm = np.log(gk["m"].to_numpy(float))
                ls = np.log(np.maximum(gk["obs"].to_numpy(float), gk["eps"].to_numpy(float)))
                sk = float(np.exp(np.interp(np.log(mk), lm, ls)))
                s10 = g[(g["k"] == 10) & (g["m"] == m10)]
                if len(s10):
                    ratios.append(sk / float(max(s10["obs"].iloc[0], s10["eps"].iloc[0])))
            if ratios:
                out[f"k{k}_m{mk:g}_over_k10_m{m10}"] = float(np.median(ratios))
    return out


def q1_descriptive(T, s11):
    """Conditional arm Q1 (plan §15): AVG5 - C1 for the 15 LM-token VL pairs, against the sentence encoders."""
    out = {"note": "decision to run recorded in STATUS ('E0 Q1 APPROVED'); descriptive only"}
    vl = [p["pair"] for p in old_vl_pairs()]
    comps = (("LA", "cliplayer"), ("LCA", E.OLD_ARM))
    for arm, base in comps:
        if arm not in T:
            continue
        if base == E.OLD_ARM:
            if base not in T:
                continue
            cb, Sb = T[base]["cells"], T[base]["slopes"]
        else:
            if "cliplayer" not in s11:
                continue
            cb, Sb = s11["cliplayer"]
        La = levels(T[arm]["cells"])
        Lb = levels(cb)
        a = La[La["k"] == 10].set_index("pair")["L_m1024"]
        b = Lb[(Lb["k"] == 10) & Lb["pair"].isin(vl)].set_index("pair")["L_m1024"]
        sa = T[arm]["slopes"]
        sa = sa[sa["k"] == 10].set_index("pair")["slope_obs"]
        sb = Sb[(Sb["k"] == 10) & Sb["pair"].isin(vl)].set_index("pair")["slope_obs"]
        dl = (np.log(a) - np.log(b.reindex(a.index))).dropna()
        db = (sa - sb.reindex(sa.index)).dropna()
        out[arm] = {"vs": base, "median_dlnL_AVG5_minus_C1": float(dl.median()), "median_db10": float(db.median()),
                    "per_lm_median_dlnL": {lm: float(dl[[i for i in dl.index if i.endswith("__" + lm)]].median())
                                           for lm in E.Q1_LMS}}
    return out


def tables_md(V, T) -> str:
    L = ["# E0 tables (produced by code; ANALYSIS_PLAN_E0.md v1.0 §14)", "",
         "## Confirmatory labels (order fixed: P1, P2b, P2c)", "", "| prediction | label | key statistics |", "|---|---|---|"]
    if "P1" in V:
        L.append(f"| P1 | {V['P1']['label']} | informative {V['P1']['n_informative']}/15; median sum {V['P1']['median_sum']}; "
                 f"frac |sum|<=0.05 {V['P1']['frac_abs_sum_le']} |")
    if "P2b" in V:
        le = {e: v["lam"] for e, v in V["P2b"]["lam_e"].items()}
        L.append(f"| P2b | {V['P2b']['label']} | lambda {V['P2b']['lam']}; lambda_e {le}; median |gap| "
                 f"{V['P2b']['median_abs_gap']:.4f} (guard {V['P2b']['guard_threshold']:.5f}) |")
    if "P2c" in V:
        ge = {e: v["median_gamma"] for e, v in V["P2c"]["gamma_e"].items()}
        L.append(f"| P2c | {V['P2c']['label']} | median gamma {V['P2c']['median_gamma']:.4f}; gamma_e {ge}; "
                 f"median dln {V['P2c']['median_dln']:.4f} |")
    L += ["", "## Secondary labels (no confirmatory weight)", "", "| analysis | cell | label |", "|---|---|---|"]
    for name, d in V["secondary"].items():
        for cell, r in d.items():
            L.append(f"| {name} | {cell} | {r.get('label', r.get('label_if_rule_applied'))} |")
    L += ["", "## Per arm, per class", "", "| arm | class | n | L(10,1024) median [min,max] | L(100,8192) median | median b(10) | "
          "median sum k10,100 |", "|---|---|---|---|---|---|---|"]
    for arm, byc in V["descriptive"]["P3a_level"].items():
        for cl, v in byc.items():
            L.append(f"| {arm} | {cl} | {v['n']} | {v['L_k10_m1024'][0]:.4f} [{v['L_k10_m1024'][1]:.4f}, "
                     f"{v['L_k10_m1024'][2]:.4f}] | {v['L_k100_m8192'][0]:.4f} | {v['median_b10']:.4f} | "
                     f"{v['median_sum_k10_100']:.4f} |")
    if "P" in T:
        L += ["", "## Arm P, per new pair", "", "| pair | L(10,1024) | b(1) | b(10) | b(100) | e_k | e_m | sum (k10,100) |",
              "|---|---|---|---|---|---|---|---|"]
        S = T["P"]["slopes"].set_index(["pair", "k"])
        Lv = T["P"]["levels"]
        el = T["P"]["el"]["k10_100"].set_index("pair")
        for p in E.new_pairs("P"):
            n = p["pair"]
            l = Lv[(Lv["pair"] == n) & (Lv["k"] == 10)]["L_m1024"].iloc[0]
            L.append(f"| {n} | {l:.4f} | {S.loc[(n, 1), 'slope_obs']:.4f} | {S.loc[(n, 10), 'slope_obs']:.4f} | "
                     f"{S.loc[(n, 100), 'slope_obs']:.4f} | {el.loc[n, 'e_k']:.4f} | {el.loc[n, 'e_m']:.4f} | {el.loc[n, 'sum']:.4f} |")
    L += ["", "## Descriptive P2", "", "```", json.dumps(E.jsonable({k: v for k, v in V.get("P2_descriptive", {}).items()
                                                                      if k in ("within_vit_permutation", "text_model_level")}),
                                                        indent=1), "```", ""]
    return "\n".join(L) + "\n"


def write_outputs(T, V, d):
    os.makedirs(d, exist_ok=True)
    for arm, t in T.items():
        E.save_parquet_new(os.path.join(d, f"E0_cells_{arm}.parquet"), t["cells"])
        E.save_parquet_new(os.path.join(d, f"E0_drawslopes_{arm}.parquet"), t["drawslopes"])
        E.save_parquet_new(os.path.join(d, f"E0_slopes_{arm}.parquet"), t["slopes"])
        el = pd.concat([x.assign(k_set=name) for name, x in t["el"].items()])
        E.save_parquet_new(os.path.join(d, f"E0_elasticity_{arm}.parquet"), el)
        E.save_parquet_new(os.path.join(d, f"E0_levels_{arm}.parquet"), t["levels"])
    E.write_new(os.path.join(d, "E0_tables.md"), tables_md(V, T))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    import e0_obs
    g = e0_obs.guard_ok()
    with open(E.FREEZE_CONSTANTS) as f:
        fc = json.load(f)
    s11 = {pre: load_s11(pre, fc["inputs_sha256"]) for pre in ("clip", "l2")}
    with open(E.CTR_CALIB) as f:
        ctr_cal = json.load(f)
    if any(os.path.exists(os.path.join(E.OBS, f"obs_{a}.parquet")) for a in E.Q1_ARMS):
        cp = os.path.join(E.S11_OUT, "S1_1_v11_cells_cliplayer.parquet")
        sp = os.path.join(E.S11_OUT, "S1_1_v11_slopes_cliplayer.parquet")
        c, S = pd.read_parquet(cp), pd.read_parquet(sp)
        s11["cliplayer"] = (c[c["primary"]].copy(), S[S["primary"]].copy())
        s11_extra = {os.path.relpath(p, E.PROJECT): E.sha256_file(p) for p in (cp, sp)}
    else:
        s11_extra = {}
    obs = {}
    for arm in (E.OLD_ARM,) + tuple(E.ARMS) + tuple(E.Q1_ARMS):
        p = os.path.join(E.OBS, f"obs_{arm}.parquet")
        if os.path.exists(p):
            obs[arm] = pd.read_parquet(p)
    missing = [a for a in E.ARMS if a not in obs]
    if missing:
        raise SystemExit(f"obs missing for arms {missing}")
    V = evaluate(obs, s11, ctr_cal, write_dir=E.EVAL)
    V["guard"] = g
    V["s11_extra_inputs_sha256"] = s11_extra
    V["obs_sha256"] = {a: E.sha256_file(os.path.join(E.OBS, f"obs_{a}.parquet")) for a in obs}
    V["code_sha256"] = {f: E.sha256_file(os.path.join(HERE, f)) for f in ("e0_evaluate.py", "e0_rules.py", "e0_common.py")}
    h = E.write_json_new(E.VERDICTS, V)
    print("VERDICTS_E0.json", h, V["labels"])


if __name__ == "__main__":
    main()
