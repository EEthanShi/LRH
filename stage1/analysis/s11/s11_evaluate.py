"""S1.1 evaluation: PREREG confirm criteria (1)-(3), the outcome rule, the
secondary analyses, and exploratory sensitivity tables.

    python s11_evaluate.py --root results/S1_1 --pre prh_clip+l2 \
        [--n1-undefined chance|nan] [--direction mean|AB|BA] [--tag NAME]

Every operationalization used here is listed in ANALYSIS_PLAN_S11.md
(O1-O12); the defaults below are the plan's defaults.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import s11_common as C  # noqa: E402

LOG115 = math.log(1.15)
STEEP = 0.1
SIGN_FRAC = 0.8
PREDICTORS = ("P1", "P2", "N0", "N1", "N2")
CLASS_ORDER = ("seed-seed", "cross-vision", "cross-modal")


# ---------------------------------------------------------------------------
# Loading and cell aggregation
# ---------------------------------------------------------------------------

def load_raw(root, pre):
    files = sorted(glob.glob(os.path.join(root, "raw", pre, "*.parquet")))
    done = [f for f in files if os.path.exists(f[:-len(".parquet")] + ".done")]
    if not done:
        raise SystemExit(f"no completed jobs under {root}/raw/{pre}")
    df = pd.concat([pd.read_parquet(f) for f in done], ignore_index=True)
    return df, [os.path.basename(f)[:-len(".parquet")] for f in done]


def combine_directions(df, direction, n1_undefined):
    """Predictor columns P1, P2, N1, N2 from the per-direction columns (O6, O8)."""
    out = df.copy()
    n1col = "N1c" if n1_undefined == "chance" else "N1"
    for X, src in (("P1", "P1"), ("P2", "P2"), ("N1", n1col), ("N2", "N2")):
        cols = [f"{src}_{d}" for d in ("AB", "BA") if f"{src}_{d}" in df]
        if direction == "mean":
            if len(cols) != 2:
                raise SystemExit(f"direction=mean needs both {src}_AB and {src}_BA")
            out[X] = 0.5 * (df[cols[0]] + df[cols[1]])     # NaN if either is NaN
        else:
            out[X] = df[f"{src}_{direction}"]
    return out


def cells(df, obs_col="obs"):
    """Draw-averaged cells (pair, m, k) (O3)."""
    g = df.groupby(["pair", "cls", "primary", "m", "k"], sort=True)
    agg = {"obs": (obs_col, "mean"), "obs_draw_sd": (obs_col, "std"), "n_draws": ("draw", "nunique"),
           "n_anchors": ("n_anchors", "mean"), "chance": ("chance", "first"),
           "is_fit_m": ("is_fit_m", "first"), "n_tieA": ("n_tieA", "mean"), "n_tieB": ("n_tieB", "mean")}
    for X in PREDICTORS:
        agg[X] = (X, "mean")          # NaN propagates (mean of NaN draws -> NaN via skipna=False below)
    c = g.agg(**agg).reset_index()
    # pandas skips NaN in mean; enforce NaN propagation for predictors
    for X in PREDICTORS:
        anynan = df.groupby(["pair", "m", "k"])[X].apply(lambda s: s.isna().any())
        c = c.merge(anynan.rename(f"_{X}_nan").reset_index(), on=["pair", "m", "k"])
        c.loc[c[f"_{X}_nan"], X] = np.nan
        c = c.drop(columns=[f"_{X}_nan"])
    c["eps"] = 0.5 / (c["n_draws"] * c["n_anchors"] * c["k"])     # O5
    return c


def flog(x, eps):
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), eps))


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def slopes_table(c, predictors=PREDICTORS):
    """Fixed-k log-log m-slope per (pair, k) over M_EVAL for obs and each predictor (O4)."""
    rows = []
    ev = c[c["m"].isin(C.M_EVAL)]
    for (pair, k), g in ev.groupby(["pair", "k"]):
        g = g.sort_values("m")
        x = np.log(g["m"].to_numpy(float))
        r = {"pair": pair, "k": int(k), "cls": g["cls"].iloc[0], "primary": bool(g["primary"].iloc[0]),
             "n_m": int(len(g)), "slope_obs": C.ols_slope(x, flog(g["obs"], g["eps"]))}
        for X in predictors:
            y = g[X].to_numpy(float)
            r[f"slope_{X}"] = C.ols_slope(x, flog(y, g["eps"])) if np.all(np.isfinite(y)) else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def msle(pred, obs, eps):
    pred = np.asarray(pred, dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        return float("nan")
    d = flog(pred, eps) - flog(obs, eps)
    return float(np.mean(d * d))


def evaluate(c, S, pair_mask, predictor="P1", comparators=("N0", "N1")):
    """PREREG confirm (1)-(3) for ``predictor`` on the pairs in pair_mask."""
    pairs = set(c.loc[pair_mask(c), "pair"])
    ev = c[c["pair"].isin(pairs) & c["m"].isin(C.M_EVAL)].copy()
    # (1) median |log(pred/obs)| over cells with obs > 2 x chance (O7: pred <= 0 -> +inf)
    el = ev[ev["obs"] > 2.0 * ev["chance"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.abs(np.log(el[predictor].to_numpy(float) / el["obs"].to_numpy(float)))
    lr = np.where(el[predictor].to_numpy(float) > 0, lr, np.inf)
    lr = np.where(np.isfinite(el[predictor].to_numpy(float)), lr, np.nan)
    c1_val = float(np.nanmedian(lr)) if np.isfinite(lr).any() or np.isinf(lr).any() else float("nan")
    c1 = {"n_cells_eligible": int(len(el)), "n_cells_eval": int(len(ev)),
          "median_abs_log_ratio": c1_val, "threshold": LOG115,
          "frac_cells_within_15pct": float(np.mean(lr <= LOG115)) if len(lr) else float("nan"),
          "pass": bool(c1_val <= LOG115) if np.isfinite(c1_val) or np.isinf(c1_val) else None}
    # steep units (O4): (pair, k) with |observed fixed-k slope| >= 0.1
    Ss = S[S["pair"].isin(pairs)]
    steep = Ss[np.abs(Ss["slope_obs"]) >= STEEP]
    if len(steep) == 0:
        return {"c1": c1, "c2": {"evaluable": False}, "c3": {"evaluable": False},
                "n_units": int(len(Ss)), "n_steep": 0, "only_smooth": True}
    key = set(zip(steep["pair"], steep["k"]))
    sc = ev[[(p, k) in key for p, k in zip(ev["pair"], ev["k"])]]
    ms = {X: msle(sc[X], sc["obs"], sc["eps"]) for X in (predictor,) + tuple(comparators)}
    comp_vals = [ms[X] for X in comparators if np.isfinite(ms[X])]
    comp = min(comp_vals) if comp_vals else float("nan")
    c2 = {"evaluable": True, "n_cells": int(len(sc)), "msle": ms, "comparator_min": comp,
          "comparators_undefined": [X for X in comparators if not np.isfinite(ms[X])],
          "ratio": ms[predictor] / comp if np.isfinite(comp) and comp > 0 else float("nan"),
          "pass": bool(ms[predictor] <= 0.5 * comp) if np.isfinite(comp) and np.isfinite(ms[predictor]) else False}
    ps = steep[f"slope_{predictor}"].to_numpy(float)
    os_ = steep["slope_obs"].to_numpy(float)
    agree = np.sign(ps) == np.sign(os_)
    agree &= np.isfinite(ps)
    c3 = {"evaluable": True, "n_units": int(len(steep)), "n_sign_correct": int(agree.sum()),
          "frac": float(agree.mean()), "threshold": SIGN_FRAC, "pass": bool(agree.mean() >= SIGN_FRAC)}
    return {"c1": c1, "c2": c2, "c3": c3, "n_units": int(len(Ss)), "n_steep": int(len(steep)),
            "only_smooth": False,
            "steep_units": [f"{p}@k={k}" for p, k in sorted(key)]}


def outcome(res):
    """PREREG S1.1 outcome rule, verbatim wording (O9)."""
    c1 = res["c1"]["pass"]
    if res["only_smooth"]:
        txt = ("No pair has |slope| >= 0.1: record \"only smooth class observed\". "
               + ("Criterion (1) passes; (2) and (3) are vacuous." if c1 else
                  "Criterion (1) fails -> the crossing law is reported as descriptive only, and the "
                  "paper's lead result shifts to the lattice/link theorems."))
        return {"confirm": bool(c1), "descriptive_only": not c1, "only_smooth": True, "text": txt}
    c2, c3 = res["c2"]["pass"], res["c3"]["pass"]
    if not (c1 and c2):
        txt = ("Criterion " + " and ".join(n for n, ok in (("(1)", c1), ("(2)", c2)) if not ok)
               + " failed: the crossing law is reported as descriptive only, and the paper's lead "
               "result shifts to the lattice/link theorems.")
        return {"confirm": False, "descriptive_only": True, "only_smooth": False, "text": txt}
    if not c3:
        return {"confirm": False, "descriptive_only": False, "only_smooth": False,
                "text": "Criteria (1) and (2) pass but (3) fails: not confirmed (all three are required); "
                        "the PREREG's descriptive-only trigger ((1) or (2) fails) is not met. Lead decision."}
    return {"confirm": True, "descriptive_only": False, "only_smooth": False,
            "text": "Criteria (1)-(3) all pass: G1 passed; the crossing law stays the lead mechanism."}


def secondary(c, S, pair_mask):
    pairs = set(c.loc[pair_mask(c), "pair"])
    Ss = S[S["pair"].isin(pairs)]
    order = {}
    for k, g in Ss.groupby("k"):
        med = {cl: float(np.median(g.loc[g["cls"] == cl, "slope_obs"])) for cl in CLASS_ORDER
               if (g["cls"] == cl).any()}
        ok = None
        if len(med) == 3:
            ok = bool(med["seed-seed"] >= med["cross-vision"] >= med["cross-modal"])
        order[int(k)] = {"median_slope_by_class": med, "ordering_holds": ok,
                         "cross_modal_negative (plan, not PREREG)": (med.get("cross-modal", float("nan")) < 0)}
    # elasticities (O10): log obs ~ a + e_k log k + e_m log m over the full grid
    el = []
    for pair, g in c[c["pair"].isin(pairs)].groupby("pair"):
        X = np.column_stack([np.ones(len(g)), np.log(g["k"].to_numpy(float)), np.log(g["m"].to_numpy(float))])
        y = flog(g["obs"], g["eps"])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        el.append({"pair": pair, "cls": g["cls"].iloc[0], "e_k": float(beta[1]), "e_m": float(beta[2]),
                   "sum": float(beta[1] + beta[2])})
    el = pd.DataFrame(el)
    summ = {}
    for cl, g in el.groupby("cls"):
        summ[cl] = {"median_e_k": float(g["e_k"].median()), "median_e_m": float(g["e_m"].median()),
                    "median_sum": float(g["sum"].median()), "n": int(len(g))}
    corr = float(np.corrcoef(el["e_k"], -el["e_m"])[0, 1]) if len(el) > 2 else float("nan")
    return {"slope_ordering": order, "elasticity_by_class": summ, "corr_ek_minus_em": corr,
            "elasticity_per_pair": el.to_dict("records")}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "pass" if x else "FAIL"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "NaN"
    if math.isinf(xf):
        return "inf"
    return f"{xf:.{nd}f}"


def per_pair_table(c, S, pairs):
    rows = []
    for pair in pairs:
        g = c[(c["pair"] == pair) & c["m"].isin(C.M_EVAL)]
        s = S[S["pair"] == pair].set_index("k")
        el = g[g["obs"] > 2 * g["chance"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.abs(np.log(el["P1"].to_numpy(float) / el["obs"].to_numpy(float)))
        lr = np.where(el["P1"].to_numpy(float) > 0, lr, np.inf)
        r = {"pair": pair, "cls": g["cls"].iloc[0]}
        for k in C.KS:
            o = g[(g["k"] == k)].sort_values("m")
            r[f"obs_k{k}_m256"] = float(o["obs"].iloc[0]) if len(o) else float("nan")
            r[f"obs_k{k}_m16384"] = float(o["obs"].iloc[-1]) if len(o) else float("nan")
            r[f"P1_k{k}_m16384"] = float(o["P1"].iloc[-1]) if len(o) else float("nan")
            r[f"slope_obs_k{k}"] = float(s.loc[k, "slope_obs"]) if k in s.index else float("nan")
            r[f"slope_P1_k{k}"] = float(s.loc[k, "slope_P1"]) if k in s.index else float("nan")
        r["median_abslogratio_P1"] = float(np.median(lr)) if len(lr) else float("nan")
        r["msle_P1"] = msle(g["P1"], g["obs"], g["eps"])
        r["msle_N0"] = msle(g["N0"], g["obs"], g["eps"])
        r["msle_N1"] = msle(g["N1"], g["obs"], g["eps"])
        rows.append(r)
    return pd.DataFrame(rows)


def write_md(path, R, ptab, header_lines):
    L = list(header_lines)
    P = R["primary"]
    o = R["outcome"]
    L += ["", "## G1 outcome (primary: P1, primary pair set, draw-mean cells)", "", f"**{o['text']}**", ""]
    L += ["| criterion | value | threshold | result |", "|---|---|---|---|"]
    L.append(f"| (1) median abs log(P1/obs), cells obs > 2x chance (n={P['c1']['n_cells_eligible']}) | "
             f"{fmt(P['c1']['median_abs_log_ratio'])} | {fmt(LOG115)} | {fmt(P['c1']['pass'])} |")
    if P["c2"].get("evaluable"):
        L.append(f"| (2) MSLE(P1) vs 0.5 min(MSLE N0, N1), {P['n_steep']} steep (pair,k) units, {P['c2']['n_cells']} cells | "
                 f"{fmt(P['c2']['msle']['P1'])} vs {fmt(P['c2']['comparator_min'])} (N0 {fmt(P['c2']['msle']['N0'])}, N1 {fmt(P['c2']['msle']['N1'])}) | ratio <= 0.5 | {fmt(P['c2']['pass'])} |")
        L.append(f"| (3) slope sign correct | {P['c3']['n_sign_correct']}/{P['c3']['n_units']} = {fmt(P['c3']['frac'])} | 0.80 | {fmt(P['c3']['pass'])} |")
    else:
        L.append("| (2) | no steep unit | - | vacuous |")
        L.append("| (3) | no steep unit | - | vacuous |")
    L += ["", "## Secondary (PREREG)", ""]
    for k, v in R["secondary"]["slope_ordering"].items():
        med = v["median_slope_by_class"]
        L.append(f"- k={k}: median fixed-k slope seed-seed {fmt(med.get('seed-seed'))}, cross-vision "
                 f"{fmt(med.get('cross-vision'))}, cross-modal {fmt(med.get('cross-modal'))}; ordering "
                 f"seed >= vision >= modal: {v['ordering_holds']}")
    L.append("")
    L.append("| class | median e_k | median e_m | median e_k + e_m | n |")
    L.append("|---|---|---|---|---|")
    for cl, v in R["secondary"]["elasticity_by_class"].items():
        L.append(f"| {cl} | {fmt(v['median_e_k'])} | {fmt(v['median_e_m'])} | {fmt(v['median_sum'])} | {v['n']} |")
    L.append(f"\ncorr(e_k, -e_m) across pairs = {fmt(R['secondary']['corr_ek_minus_em'])}")
    L += ["", "## Exploratory / sensitivity (not preregistered as confirmatory)", "",
          "| variant | (1) median abs log ratio | (2) MSLE pred / min null | (3) sign frac | outcome |",
          "|---|---|---|---|---|"]
    for name, v in R["sensitivity"].items():
        c2 = v["res"]["c2"]
        L.append(f"| {name} | {fmt(v['res']['c1']['median_abs_log_ratio'])} | "
                 f"{fmt(c2.get('ratio')) if c2.get('evaluable') else 'vacuous'} | "
                 f"{fmt(v['res']['c3'].get('frac')) if v['res']['c3'].get('evaluable') else 'vacuous'} | "
                 f"{'confirm' if v['outcome']['confirm'] else ('descriptive only' if v['outcome']['descriptive_only'] else 'not confirmed')} |")
    for cl in CLASS_ORDER:
        t = ptab[ptab["cls"] == cl]
        if not len(t):
            continue
        L += ["", f"## Per pair: {cl}", "",
              "| pair | s_N k=10 m=256 | s_N k=10 m=16384 | P1 k=10 m=16384 | slope obs k=1/10/100 | slope P1 k=1/10/100 | med abs log(P1/obs) | MSLE P1 / N0 / N1 |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            L.append(f"| {r['pair']} | {fmt(r['obs_k10_m256'])} | {fmt(r['obs_k10_m16384'])} | {fmt(r['P1_k10_m16384'])} | "
                     f"{fmt(r['slope_obs_k1'])} / {fmt(r['slope_obs_k10'])} / {fmt(r['slope_obs_k100'])} | "
                     f"{fmt(r['slope_P1_k1'])} / {fmt(r['slope_P1_k10'])} / {fmt(r['slope_P1_k100'])} | "
                     f"{fmt(r['median_abslogratio_P1'])} | {fmt(r['msle_P1'])} / {fmt(r['msle_N0'])} / {fmt(r['msle_N1'])} |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.RESULTS_DIR)
    ap.add_argument("--pre", default="prh_clip+l2")
    ap.add_argument("--n1-undefined", default="chance", choices=("chance", "nan"))
    ap.add_argument("--direction", default="mean", choices=("mean", "AB", "BA"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--title", default="S1.1 out-of-sample (m, k) test of the crossing law (gate G1)")
    args = ap.parse_args(argv)

    raw, jobs = load_raw(args.root, args.pre)
    df = combine_directions(raw, args.direction, args.n1_undefined)
    c = cells(df)
    S = slopes_table(c)
    prim = lambda t: t["primary"]          # noqa: E731  O1: 4 disjoint seed + 10 vision + 15 modal
    allp = lambda t: np.ones(len(t), bool)  # noqa: E731
    R = {"pre": args.pre, "direction": args.direction, "n1_undefined": args.n1_undefined,
         "jobs": jobs, "n_pairs": int(c["pair"].nunique()),
         "n_primary_pairs": int(c.loc[c["primary"], "pair"].nunique()),
         "draws_per_cell": sorted(set(int(x) for x in c["n_draws"])),
         "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    R["primary"] = evaluate(c, S, prim)
    R["outcome"] = outcome(R["primary"])
    R["secondary"] = secondary(c, S, prim)
    R["secondary_all_seed_pairs"] = secondary(c, S, allp)
    sens = {}
    sens["all 61 pairs (36 seed-seed)"] = evaluate(c, S, allp)
    for X in ("P2", "N2", "N0", "N1"):
        sens[f"{X} as the predictor"] = evaluate(c, S, prim, predictor=X,
                                                 comparators=tuple(z for z in ("N0", "N1") if z != X) or ("N0",))
    for d in ("AB", "BA"):
        if d != args.direction and f"P1_{d}" in raw:
            dfd = combine_directions(raw, d, args.n1_undefined)
            cd = cells(dfd)
            sens[f"P1 direction {d} only"] = evaluate(cd, slopes_table(cd), prim)
    alt = "nan" if args.n1_undefined == "chance" else "chance"
    ca = cells(combine_directions(raw, args.direction, alt))
    sens[f"N1 undefined -> {alt}"] = evaluate(ca, slopes_table(ca), prim)
    cc = cells(df, obs_col="obs_clean")
    sens["obs on anchors without exact k-boundary ties"] = evaluate(cc, slopes_table(cc), prim)
    R["sensitivity"] = {k: {"res": v, "outcome": outcome(v)} for k, v in sens.items()}
    for cl in CLASS_ORDER:
        sub = c[c["cls"] == cl]
        if len(sub):
            R.setdefault("by_class", {})[cl] = evaluate(c, S, lambda t, cl=cl: t["primary"] & (t["cls"] == cl))
    ptab = per_pair_table(c, S, sorted(c["pair"].unique(), key=lambda p: (CLASS_ORDER.index(C.PAIRS[p]["cls"]) if p in C.PAIRS else 0, p)))
    tag = args.tag or args.pre.replace("+", "_")
    out_json = os.path.join(args.root, f"S1_1_summary_{tag}.json")
    out_md = os.path.join(args.root, f"S1_1_summary_{tag}.md")
    c.to_parquet(os.path.join(args.root, f"S1_1_cells_{tag}.parquet"), index=False)
    S.to_parquet(os.path.join(args.root, f"S1_1_slopes_{tag}.parquet"), index=False)
    ptab.to_parquet(os.path.join(args.root, f"S1_1_perpair_{tag}.parquet"), index=False)
    with open(out_json, "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    hdr = [f"# {args.title}", "",
           f"pre = `{args.pre}`; P1 direction = {args.direction}; N1 with slope <= 0 -> {args.n1_undefined}; "
           f"{R['n_pairs']} pairs ({R['n_primary_pairs']} primary); draws per cell {R['draws_per_cell']}; "
           f"evaluation m = {list(C.M_EVAL)} (fit m = {C.M_FIT} excluded), k = {list(C.KS)}.",
           "Operationalizations: analysis/s11/ANALYSIS_PLAN_S11.md (O1-O12)."]
    write_md(out_md, R, ptab, hdr)
    print(json.dumps({"outcome": R["outcome"], "c1": R["primary"]["c1"], "c2": R["primary"]["c2"],
                      "c3": R["primary"]["c3"]}, indent=1, default=str))
    print("wrote", out_md, out_json)


if __name__ == "__main__":
    main()
