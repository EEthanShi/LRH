"""Gate G0: the known-answer plant suite of SPEC section 7.

    OMP_NUM_THREADS=1 python -m hrhv2.run_g0 --jobs 10 \\
        --out results/G0_report        # writes G0_report.json and G0_report.md

Plants (point clouds: flat torus [0,1)^p, Clifford-embedded, random isometry
into R^64; nested pools m in {256, 1024, 4096} are prefixes of one cloud):

  P-Bern    K_B = exp(beta (K_A - 1)), beta in {1,3,10}, p in {2,4,8}.
            Gate: every ordinal rung identical (Delta_N all k, components
            k in {1,3,10}, Delta_O, Delta_E, s_T = 1) at every (p, m, beta);
            biased and debiased CKA strictly decreasing in beta at every (p, m);
            global-isotonic explained fraction >= 0.999 (mean of both
            directions; in-sample and cross-fitted).
  P-Aniso   B = A M (first intrinsic circle scaled by 2).
            Gate: |OLS slope of mean s_N(k) on log2 m| < 0.03 for k in {1,3,10}, all p.
  P-Conf    sphere inversion (conformal). Gate: mean s_N(k) strictly increasing
            over m = 256 < 1024 < 4096 for k in {1,3,10}, all p.
  P-Rough   shared + private noise. Gate: mean s_N(k) strictly decreasing, k in {1,3,10}, all p.
  P-Gauss   D_B = D_A + E, E iid N(0, sigma^2), sigma = c * median D_(10) (m=1024 reference
            cloud), c in {0.02, 0.05, 0.15, 0.4}. P1 fitted on H1 x H1 at m = 1024 and
            evaluated on H2 anchors at m in {256, 1024, 4096}, k in {1,3,10,30,41,100}
            (41 = k_q at m=4096; k_q is 3 / 10 at m = 256 / 1024).
            Gate: |mean P1 - mean observed s_N| <= 0.02 on every cell with observed
            s_N in [0.05, 0.99] (means over replicate draws).
  Known-p   Levina-Bickel bias (mean p-hat / p) at each (m, p, k). Reported, no gate.
  WarpShare (supplementary, final_plan S0.1) D_B = g(D_A) + E: global-isotonic
            explained fraction vs the planted share, |diff| <= 0.05.

All seeds are sha256 of string keys 'G0f|<plant>|p=<p>|...|rep=<r>' (formal run;
the prefix 'G0' was used by smoke runs and 'G0dev' by development runs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback

import numpy as np

from . import __version__, SPEC_VERSION
from . import plants, predict, util, structural as st, geometric as geo
from .ladder import ladder
from .prep import Geom, center_gram

M_LIST = (256, 1024, 4096)
M_MAX = 4096
M_FIT = 1024
P_LIST = (2, 4, 8)
BETAS = (1, 3, 10)
KS_SLOPE = (1, 3, 10, 30, 100)
KS_GATE_SLOPE = (1, 3, 10)
KS_GAUSS = (1, 3, 10, 30, 41, 100)
GAUSS_C = (0.02, 0.05, 0.15, 0.4)
ROUGH_KAPPA = 0.3
LB_KS = (3, 5, 10, 20, 30, 50, 100)
EXACT_KS = (1, 3)  # diagnostic P1x / oracle_x on P-Gauss (exact E|N_A \ N_B| under the same law)
SALT = "hrhv2-halves"
KEY = "G0f"  # formal G0 seeds (smoke runs used "G0", dev runs "G0dev")


# ----------------------------------------------------------------------------
# task workers (run in forked processes, single-threaded BLAS)
# ----------------------------------------------------------------------------

def _ladder_slim(r: dict) -> dict:
    keep = {}
    for k, v in r.items():
        if isinstance(v, (bool, int, float, str, np.floating, np.integer, np.bool_)):
            keep[k] = v
    return keep


def task_bern(p: int, beta: float, m: int):
    KA, KBs, info = plants.bern_grams(M_MAX, p, BETAS, f"{KEY}|P-Bern|p={p}")
    KA, KB = KA[:m, :m], KBs[beta][:m, :m]
    A, B = Geom.from_gram(KA), Geom.from_gram(KB)
    r = ladder(A, B, with_gl=False, n_jobs=1)
    pr = geo.participation_ratio(center_gram(KA))
    out = _ladder_slim(r)
    out.update({"p": p, "beta": beta, "m": m, "cka_limit_sqrt_PR_over_m1": math.sqrt(pr / (m - 1)),
                "float_collisions_K": info[beta]["float_collisions_K"],
                "float_collisions_D": info[beta]["float_collisions_D"],
                "float_collisions_DA": info["float_collisions_DA"], "max_D_A": info["max_D_A"]})
    return out


def _pair_dists(plant: str, p: int, rep: int):
    key = f"{KEY}|{plant}|p={p}|rep={rep}"
    if plant == "P-Aniso":
        XA, XB = plants.aniso_features(M_MAX, p, key)
    elif plant == "P-Conf":
        XA, XB = plants.conf_features(M_MAX, p, key)
    elif plant == "P-Rough":
        ref = plants.feature_dists(plants.base_cloud(256, p, f"{KEY}|P-Rough|p={p}|ref"))
        s2 = ROUGH_KAPPA * plants.knn_d_median(ref, 10)
        XA, XB = plants.rough_features(M_MAX, p, key, s2=s2)
    else:
        raise ValueError(plant)
    return plants.feature_dists(XA), plants.feature_dists(XB)


def gauss_sigma(p: int, c: float) -> float:
    ref = plants.feature_dists(plants.base_cloud(M_FIT, p, f"{KEY}|P-Gauss|p={p}|sigma_ref"))
    return c * plants.knn_d_median(ref, 10)


def task_pair(plant: str, p: int, rep: int, variants, c: float | None = None):
    """Observed s_N on all anchors (slope plants) + cross-fitted predictors."""
    t0 = time.perf_counter()
    h1 = util.hash_half(plants.nested_ids(M_MAX), SALT)
    sigma = None
    if plant == "P-Gauss":
        sigma = gauss_sigma(p, c)
        DA, DB = plants.gauss_plant(M_MAX, p, sigma, f"{KEY}|P-Gauss|p={p}|c={c}|rep={rep}")
        ks = KS_GAUSS
    else:
        DA, DB = _pair_dists(plant, p, rep)
        ks = KS_SLOPE
    allanch = {}
    for m in M_LIST:
        nA, _ = st.knn_table(DA[:m, :m], max(ks) + 1)
        nB, _ = st.knn_table(DB[:m, :m], max(ks) + 1)
        allanch[m] = {k: float(st.knn_overlap(nA, nB, k)) for k in ks if k <= m - 2}
    cf = predict.crossfit_predict(DA, DB, h1, m_fit=M_FIT, m_list=M_LIST, ks=ks, variants=variants,
                                  with_nulls=True, oracle_sigma=sigma,
                                  exact_ks=EXACT_KS if plant == "P-Gauss" else (),
                                  exact_variants=variants)
    fit = cf.pop("_fit")
    return {"plant": plant, "p": p, "rep": rep, "c": c, "sigma": sigma, "sN_all": allanch,
            "cf": cf, "fit": fit, "time_s": time.perf_counter() - t0}


def task_knownp(p: int, rep: int):
    X = plants.base_cloud(M_MAX, p, f"{KEY}|Known-p|p={p}|rep={rep}")
    D = plants.feature_dists(X)
    out = {}
    for m in M_LIST:
        _, v = st.knn_table(D[:m, :m], max(LB_KS))
        out[m] = {}
        for k in LB_KS:
            lb = predict.lb_dimension(v, k)
            out[m][k] = {"lb": lb["lb"], "mg": lb["mg"]}
    return {"p": p, "rep": rep, "lb": out}


def task_warpshare(p: int, noise_rel: float):
    m = 1024
    DA, DB, planted = plants.warpshare_plant(m, p, noise_rel, f"{KEY}|WarpShare|p={p}|nr={noise_rel}")
    from . import quotients as qu
    h1 = util.hash_half(plants.nested_ids(m), SALT)
    q = qu.quotient_ladder(DA, DB, h1, with_row=False)
    return {"p": p, "noise_rel": noise_rel, "planted": planted,
            "iso_expl_AB": q["quot_AB_iso_expl"], "iso_expl_AB_cf": q["quot_AB_iso_expl_cf"],
            "affine_expl_AB": q["quot_AB_affine_expl"]}


def _run(task):
    name, args = task
    fn = globals()[name]
    try:
        return name, args, fn(*args), None
    except Exception:
        return name, args, None, traceback.format_exc()


# ----------------------------------------------------------------------------
# aggregation and gates
# ----------------------------------------------------------------------------

def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    return float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))


def eval_bern(rows):
    cells = sorted(rows, key=lambda r: (r["p"], r["m"], r["beta"]))
    ordinal_fail = []
    for r in cells:
        ks = st.default_k_grid(r["m"])
        bad = [f"Delta_N_k{k}" for k in ks if r[f"Delta_N_k{k}"] != 0.0]
        bad += [f"comp_rand_k{k}" for k in (1, 3, 10) if r[f"comp_rand_k{k}"] != 0.0]
        bad += [n for n in ("Delta_O", "Delta_E") if r[n] != 0.0]
        if r["s_T"] != 1.0:
            bad.append("s_T")
        if bad:
            ordinal_fail.append({"p": r["p"], "m": r["m"], "beta": r["beta"], "failed": bad})
    cka_fail, iso_fail = [], []
    for p in P_LIST:
        for m in M_LIST:
            cc = [r for r in cells if r["p"] == p and r["m"] == m]
            cc.sort(key=lambda r: r["beta"])
            for name in ("cka_biased", "cka_debiased"):
                v = [r[name] for r in cc]
                if not all(a > b for a, b in zip(v, v[1:])):
                    cka_fail.append({"p": p, "m": m, "which": name, "values": v})
            for r in cc:
                vals = [r["quot_mean_iso_expl"], r["quot_mean_iso_expl_cf"]]
                if not all(v >= 0.999 for v in vals):
                    iso_fail.append({"p": p, "m": m, "beta": r["beta"], "values": vals})
    keep = ["p", "m", "beta", "Delta_N_k1", "Delta_N_k10", "Delta_O", "Delta_E", "s_T",
            "comp_rand_k1", "comp_rand_k10", "cka_biased", "cka_debiased",
            "cka_limit_sqrt_PR_over_m1", "quot_mean_iso_expl", "quot_mean_iso_expl_cf",
            "quot_mean_rowiso_expl", "quot_mean_affine_expl", "Gs_rms", "rsa_pearson",
            "float_collisions_K", "float_collisions_D", "neartie_A_k10", "neartie_B_k10",
            "time_total_s"]
    return {"pass": not (ordinal_fail or cka_fail or iso_fail),
            "ordinal_failures": ordinal_fail, "cka_monotonicity_failures": cka_fail,
            "iso_failures": iso_fail,
            "cells": [{k: r.get(k) for k in keep} for r in cells],
            "full_ladders": cells}


def _mean_sN(rows, p):
    """Mean over reps of all-anchor s_N: {m: {k: (mean, se)}}."""
    rr = [r for r in rows if r["p"] == p]
    out = {}
    for m in M_LIST:
        out[m] = {}
        for k in rr[0]["sN_all"][m]:
            v = np.array([r["sN_all"][m][k] for r in rr])
            out[m][k] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan"))
    return out


def eval_slope_plant(plant, rows):
    table, fails, slopes = {}, [], {}
    for p in P_LIST:
        ms = _mean_sN(rows, p)
        table[p] = {m: {k: {"mean": ms[m][k][0], "se": ms[m][k][1]} for k in ms[m]} for m in M_LIST}
        slopes[p] = {}
        for k in KS_SLOPE:
            if not all(k in ms[m] for m in M_LIST):
                continue
            y = [ms[m][k][0] for m in M_LIST]
            sl = ols_slope(np.log2(M_LIST), y)
            per_rep = [ols_slope(np.log2(M_LIST), [r["sN_all"][m][k] for m in M_LIST])
                       for r in rows if r["p"] == p]
            se = float(np.std(per_rep, ddof=1) / np.sqrt(len(per_rep))) if len(per_rep) > 1 else float("nan")
            slopes[p][k] = {"slope_per_doubling": sl, "slope_se": se, "values": y}
            if k in KS_GATE_SLOPE:
                ok = {"P-Aniso": abs(sl) < 0.03,
                      "P-Conf": y[0] < y[1] < y[2],
                      "P-Rough": y[0] > y[1] > y[2]}[plant]
                if not ok:
                    fails.append({"p": p, "k": k, "values": y, "slope": sl})
    return {"pass": not fails, "failures": fails, "slopes": slopes, "table": table}


PRED_NAMES = ("oracle", "N0", "N1", "N2", "P2")


def predictor_cells(rows, variants):
    """Aggregate cross-fitted predictions over reps: one record per
    (plant, p, c, m, k) with mean obs / preds and the obs standard error."""
    groups = {}
    for r in rows:
        for m, cell in r["cf"].items():
            for k, c in cell.items():
                g = groups.setdefault((r["plant"], r["p"], r["c"], int(m), int(k)), [])
                g.append(c)
    out = []
    for (plant, p, cc, m, k), cs in sorted(groups.items(), key=lambda t: (t[0][0], t[0][1], t[0][2] or 0, t[0][3], t[0][4])):
        obs = np.array([c["obs"] for c in cs])
        n_anchor = np.array([c["n_anchors"] for c in cs])
        sd = np.array([c["obs_sd"] for c in cs])
        # SE of the mean over reps from the pooled per-anchor variance
        se = float(np.sqrt(np.sum(sd ** 2 * n_anchor) / np.sum(n_anchor) / np.sum(n_anchor)))
        rec = {"plant": plant, "p": p, "c": cc, "m": m, "k": k, "obs": float(obs.mean()),
               "obs_se": se, "chance": k / (m - 1), "n_reps": len(cs)}
        for v in cs[0]:
            if v in ("obs", "obs_sd", "n_anchors", "chance", "P2_p_hat"):
                continue
            vals = [c[v] for c in cs]
            rec[v] = float(np.mean(vals)) if all(np.isfinite(vals)) else float("nan")
        if "P2_p_hat" in cs[0]:
            rec["P2_p_hat"] = float(np.mean([c["P2_p_hat"] for c in cs]))
        out.append(rec)
    return out


def eligible(rec):
    return 0.05 <= rec["obs"] <= 0.99


def eval_gauss(cells, variant):
    cc = [c for c in cells if c["plant"] == "P-Gauss" and eligible(c)]
    errs = [c[variant] - c["obs"] for c in cc]
    fails = [{"p": c["p"], "c": c["c"], "m": c["m"], "k": c["k"], "obs": c["obs"], "pred": c[variant],
              "err": c[variant] - c["obs"], "oracle_err": c.get("oracle", float("nan")) - c["obs"],
              "obs_se": c["obs_se"]} for c in cc if abs(c[variant] - c["obs"]) > 0.02]
    by_k = {}
    for k in KS_GAUSS:
        ck = [c for c in cc if c["k"] == k]
        if not ck:
            continue
        e = [abs(c[variant] - c["obs"]) for c in ck]
        d = {"n_cells": len(e), "max_abs_err": float(max(e)), "mean_abs_err": float(np.mean(e)),
             "signed_mean_err": float(np.mean([c[variant] - c["obs"] for c in ck])),
             "oracle_max_abs_err": float(max(abs(c["oracle"] - c["obs"]) for c in ck)),
             "pass": bool(max(e) <= 0.02)}
        ex = [c for c in ck if np.isfinite(c.get(f"P1x:{variant}", np.nan))]
        if ex:
            d["P1x_max_abs_err"] = float(max(abs(c[f"P1x:{variant}"] - c["obs"]) for c in ex))
            d["P1x_n_cells"] = len(ex)
        exo = [c for c in ck if np.isfinite(c.get("oracle_x", np.nan))]
        if exo:
            d["oracle_x_max_abs_err"] = float(max(abs(c["oracle_x"] - c["obs"]) for c in exo))
            d["oracle_x_n_cells"] = len(exo)
        by_k[k] = d
    fails3 = [f for f in fails if f["k"] >= 3]
    return {"variant": variant, "pass": not fails, "pass_k_ge_3": not fails3, "n_eligible_cells": len(cc),
            "max_abs_err": float(np.max(np.abs(errs))) if errs else float("nan"),
            "mean_abs_err": float(np.mean(np.abs(errs))) if errs else float("nan"),
            "oracle_max_abs_err": float(max(abs(c["oracle"] - c["obs"]) for c in cc)) if cc else float("nan"),
            "by_k": by_k, "failures": fails}


def variant_scores(cells, variants):
    out = {}
    for v in list(variants) + ["oracle", "N0", "N1", "N2", "P2"]:
        per_class = {}
        for plant in ("P-Gauss", "P-Aniso", "P-Conf", "P-Rough"):
            cc = [c for c in cells if c["plant"] == plant and eligible(c) and v in c]
            if not cc:
                continue
            e = np.abs([c[v] - c["obs"] for c in cc])
            per_class[plant] = {"mean_abs_err": float(e.mean()), "max_abs_err": float(e.max()),
                                "n_cells": int(e.size),
                                "mean_abs_err_k_ge_3": float(np.mean([abs(c[v] - c["obs"]) for c in cc if c["k"] >= 3]))}
        if per_class:
            out[v] = {"per_class": per_class,
                      "class_mean_abs_err": float(np.mean([d["mean_abs_err"] for d in per_class.values()]))}
    return out


def estimator_fidelity(cells, variant):
    """max |P1 - oracle| over eligible P-Gauss cells (oracle = the same mu-law
    with the true coupling): the error attributable to the coupling estimator."""
    cc = [c for c in cells if c["plant"] == "P-Gauss" and eligible(c)]
    e = [abs(c[variant] - c["oracle"]) for c in cc]
    return {"max_abs_dev_from_oracle": float(max(e)), "mean_abs_dev_from_oracle": float(np.mean(e)),
            "pass": bool(max(e) <= 0.02)}


def choose_variant(scores, fidelity, variants, preference):
    """Pre-registered rule (PREDICTOR_CHOICE.md): (1) keep the variants whose
    P1 stays within 0.02 of the oracle (the same mu-law computed with the true
    P-Gauss coupling) on every eligible P-Gauss cell, if any does; this
    isolates the estimator from the mu-law's own small-k error; (2) among them
    minimise the class-mean absolute error vs observed s_N over all eligible
    cells (observed s_N in [0.05, 0.99]) of P-Gauss, P-Aniso, P-Conf,
    P-Rough; (3) variants within 0.002 of the minimum are tied and resolved
    by the a-priori preference order."""
    cand = [v for v in variants if fidelity[v]["pass"]]
    filt = bool(cand)
    if not cand:
        cand = list(variants)
    best = min(scores[v]["class_mean_abs_err"] for v in cand)
    tied = [v for v in cand if scores[v]["class_mean_abs_err"] <= best + 0.002]
    chosen = sorted(tied, key=lambda v: preference.index(v))[0]
    return {"chosen": chosen, "fidelity_filter_applied": filt, "candidates_after_filter": cand,
            "best_class_mean_abs_err": best, "tied_within_0.002": tied}


def eval_knownp(rows):
    out = {}
    for p in P_LIST:
        out[p] = {}
        for m in M_LIST:
            out[p][m] = {}
            for k in LB_KS:
                lb = [r["lb"][m][k]["lb"] for r in rows if r["p"] == p]
                mg = [r["lb"][m][k]["mg"] for r in rows if r["p"] == p]
                out[p][m][k] = {"lb_mean": float(np.mean(lb)), "mg_mean": float(np.mean(mg)),
                                "bias_ratio_lb": float(np.mean(lb) / p), "bias_ratio_mg": float(np.mean(mg) / p)}
    return out


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------

def _f(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    return f"{x:.{nd}f}"


def write_md(rep: dict, path: str):
    L = []
    P = rep["plants"]
    L.append("# G0 report: known-answer plants (SPEC v1.0 section 7)\n")
    L.append(f"Generated {rep['meta']['created_utc']} on {rep['meta']['host']} by hrhv2 {rep['meta']['hrhv2_version']} "
             f"(run_g0.py sha256 {rep['meta']['run_g0_sha256'][:12]}, predict.py sha256 {rep['meta']['predict_sha256'][:12]}). "
             f"Wall time {rep['meta']['wall_s']/60:.1f} min with {rep['meta']['jobs']} worker processes.\n")
    L.append("## Verdict\n")
    L.append("| Plant | SPEC 7 requirement | Result |\n|---|---|---|")
    for name, req in rep["gate_definitions"].items():
        res = P[name]["pass"] if name in P and "pass" in P[name] else None
        L.append(f"| {name} | {req} | {'PASS' if res else ('FAIL' if res is not None else 'reported')} |")
    L.append(f"\n**G0 (SPEC 7, harness-main): {'PASS' if rep['overall']['G0_SPEC7_pass'] else 'FAIL'}**"
             + ("" if rep["overall"]["G0_SPEC7_pass"] else f" — failing: {', '.join(rep['overall']['failed'])}") + "\n")
    L.append("SPEC 8 (agreement with hrhv2_ref) is not assessed here: harness-main has no access to hrhv2_ref by design. "
             "Every number below is in G0_report.json for that comparison.\n")

    b = P["P-Bern"]
    L.append("## P-Bern (Bernstein warp)\n")
    L.append(f"Ordinal failures: {len(b['ordinal_failures'])}; CKA monotonicity failures: {len(b['cka_monotonicity_failures'])}; "
             f"isotonic failures: {len(b['iso_failures'])}.\n")
    L.append("| p | m | beta | Delta_N(1) | Delta_N(10) | Delta_O | Delta_E | s_T | CKA | dCKA | sqrt(PR/(m-1)) | iso expl | iso expl cf | float collisions |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in b["cells"]:
        L.append(f"| {c['p']} | {c['m']} | {c['beta']} | {c['Delta_N_k1']:.0f} | {c['Delta_N_k10']:.0f} | {c['Delta_O']:.0f} | "
                 f"{c['Delta_E']:.0f} | {c['s_T']:.0f} | {_f(c['cka_biased'],4)} | {_f(c['cka_debiased'],4)} | "
                 f"{_f(c['cka_limit_sqrt_PR_over_m1'],4)} | {_f(c['quot_mean_iso_expl'],6)} | {_f(c['quot_mean_iso_expl_cf'],6)} | "
                 f"{c['float_collisions_K']}/{c['float_collisions_D']} |")
    for plant in ("P-Aniso", "P-Conf", "P-Rough"):
        d = P[plant]
        L.append(f"\n## {plant}\n")
        L.append(f"Mean all-anchor s_N over {rep['meta']['reps_slope']} replicate clouds; slope = OLS slope on log2 m (per doubling), "
                 "SE from replicate-to-replicate spread. Gate k in {1, 3, 10}.\n")
        L.append("| p | k | s_N(256) | s_N(1024) | s_N(4096) | slope/doubling | slope SE | gate |")
        L.append("|---|---|---|---|---|---|---|---|")
        for p in P_LIST:
            for k, s in d["slopes"][str(p)].items():
                fail = any(f["p"] == p and f["k"] == int(k) for f in d["failures"])
                gate = ("FAIL" if fail else "pass") if int(k) in KS_GATE_SLOPE else "-"
                L.append(f"| {p} | {k} | {_f(s['values'][0])} | {_f(s['values'][1])} | {_f(s['values'][2])} | "
                         f"{_f(s['slope_per_doubling'],4)} | {_f(s['slope_se'],4)} | {gate} |")
    g = P["P-Gauss"]
    L.append("\n## P-Gauss (crossing-law predictor P1)\n")
    L.append(f"Frozen P1 variant: `{g['variant']}`. Eligible cells (observed s_N in [0.05, 0.99]): {g['n_eligible_cells']}. "
             f"Max |P1 - obs| = {_f(g['max_abs_err'],4)}; mean = {_f(g['mean_abs_err'],4)}. "
             f"Oracle (true sigma, same mu-law) max |err| = {_f(g['oracle_max_abs_err'],4)}.\n")
    L.append("| k | cells | max abs err P1 | mean abs err P1 | signed mean err P1 | max abs err oracle (mu-law, true law) | max abs err P1x (exact, fitted law) | max abs err oracle_x (exact, true law) | pass |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, d in g["by_k"].items():
        L.append(f"| {k} | {d['n_cells']} | {_f(d['max_abs_err'],4)} | {_f(d['mean_abs_err'],4)} | {_f(d['signed_mean_err'],4)} | "
                 f"{_f(d['oracle_max_abs_err'],4)} | {_f(d.get('P1x_max_abs_err'),4)} | {_f(d.get('oracle_x_max_abs_err'),4)} | {'pass' if d['pass'] else 'FAIL'} |")
    L.append("\nPer-cell table (means over replicates; SE of the observed mean):\n")
    v = g["variant"]
    L.append(f"| p | c | m | k | obs | SE | {v} | P1x | oracle | oracle_x | N0 | N1 | N2 | P2 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in rep["predictor_cells"]:
        if c["plant"] != "P-Gauss":
            continue
        L.append(f"| {c['p']} | {c['c']} | {c['m']} | {c['k']} | {_f(c['obs'])} | {_f(c['obs_se'])} | {_f(c.get(v))} | "
                 f"{_f(c.get('P1x:' + v))} | {_f(c.get('oracle'))} | {_f(c.get('oracle_x'))} | {_f(c.get('N0'))} | "
                 f"{_f(c.get('N1'))} | {_f(c.get('N2'))} | {_f(c.get('P2'))} |")
    L.append("\n## P1 on the other plant classes (supplementary; used for the variant choice)\n")
    L.append("| p | plant | m | k | obs | " + " | ".join(rep["predictor_choice"]["variants"]) + " | N1 |")
    L.append("|---|---|---|---|---|" + "---|" * (len(rep["predictor_choice"]["variants"]) + 1))
    for c in rep["predictor_cells"]:
        if c["plant"] == "P-Gauss" or c["k"] not in (1, 10, 100):
            continue
        L.append(f"| {c['p']} | {c['plant']} | {c['m']} | {c['k']} | {_f(c['obs'])} | "
                 + " | ".join(_f(c.get(x)) for x in rep["predictor_choice"]["variants"]) + f" | {_f(c.get('N1'))} |")
    pc = rep["predictor_choice"]
    L.append("\n## Predictor choice (P1)\n")
    L.append(f"Rule: {pc['rule']}\n")
    L.append("| variant | max abs dev from oracle (P-Gauss) | P-Gauss gate (all k / k>=3) | class-mean abs err | P-Gauss | P-Aniso | P-Conf | P-Rough |")
    L.append("|---|---|---|---|---|---|---|---|")
    for v in pc["variants"] + ["oracle", "N0", "N1", "N2", "P2"]:
        s = pc["scores"].get(v)
        if not s:
            continue
        pcl = s["per_class"]
        gg = pc["gauss_gate"].get(v)
        gtxt = "-" if gg is None else f"{'pass' if gg['pass'] else 'FAIL'} / {'pass' if gg['pass_k_ge_3'] else 'FAIL'}"
        fd = pc["estimator_fidelity"].get(v)
        ftxt = "-" if fd is None else f"{_f(fd['max_abs_dev_from_oracle'],4)} ({'pass' if fd['pass'] else 'FAIL'})"
        L.append(f"| {v} | {ftxt} | {gtxt} | {_f(s['class_mean_abs_err'],4)} | "
                 + " | ".join(_f(pcl.get(pl, {}).get("mean_abs_err"), 4) for pl in ("P-Gauss", "P-Aniso", "P-Conf", "P-Rough")) + " |")
    L.append(f"\nChosen: **{pc['decision']['chosen']}** (tied within 0.002: {pc['decision']['tied_within_0.002']}).\n")
    kp = P["Known-p"]
    L.append("## Known-p: Levina-Bickel bias (mean p-hat / p; 'lb' = mean of per-anchor MLEs, 'mg' = inverse mean inverse)\n")
    L.append("| p | m | " + " | ".join(f"k={k} lb/mg" for k in LB_KS) + " |")
    L.append("|---|---|" + "---|" * len(LB_KS))
    for p in P_LIST:
        for m in M_LIST:
            d = kp["table"][str(p)][str(m)]
            L.append(f"| {p} | {m} | " + " | ".join(f"{d[str(k)]['bias_ratio_lb']:.2f}/{d[str(k)]['bias_ratio_mg']:.2f}" for k in LB_KS) + " |")
    ws = P["WarpShare"]
    L.append("\n## WarpShare (supplementary)\n")
    L.append("| p | noise | planted share | iso explained | iso explained cf | |diff| <= 0.05 |")
    L.append("|---|---|---|---|---|---|")
    for r in ws["rows"]:
        L.append(f"| {r['p']} | {r['noise_rel']} | {_f(r['planted'],4)} | {_f(r['iso_expl_AB'],4)} | {_f(r['iso_expl_AB_cf'],4)} | "
                 f"{'yes' if abs(r['iso_expl_AB_cf'] - r['planted']) <= 0.05 else 'NO'} |")
    L.append("\n## Notes\n")
    for n in rep["notes"]:
        L.append(f"- {n}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--reps-gauss", type=int, default=24)
    ap.add_argument("--reps-slope", type=int, default=4)
    ap.add_argument("--reps-knownp", type=int, default=2)
    ap.add_argument("--variants", default=",".join(predict.P1_CANDIDATES))
    ap.add_argument("--preference", default=None, help="a-priori tie-break order (default: --variants order)")
    ap.add_argument("--frozen", default=None, help="variant whose P-Gauss gate is the SPEC gate "
                    "(default: the variant chosen by the rule in this run)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (not a G0 result)")
    ap.add_argument("--out", default="results/G0_report")
    args = ap.parse_args(argv)
    variants = args.variants.split(",")
    preference = args.preference.split(",") if args.preference else variants
    t0 = time.perf_counter()
    global P_LIST, GAUSS_C
    if args.quick:
        P_LIST, GAUSS_C = (2,), (0.15,)
        args.reps_gauss, args.reps_slope, args.reps_knownp = 2, 2, 1
    tasks = []
    for p in P_LIST:
        for beta in BETAS:
            for m in M_LIST:
                tasks.append(("task_bern", (p, beta, m)))
        for plant in ("P-Aniso", "P-Conf", "P-Rough"):
            for rep in range(args.reps_slope):
                tasks.append(("task_pair", (plant, p, rep, variants)))
        for c in GAUSS_C:
            for rep in range(args.reps_gauss):
                tasks.append(("task_pair", ("P-Gauss", p, rep, variants, c)))
        for rep in range(args.reps_knownp):
            tasks.append(("task_knownp", (p, rep)))
        for nr in (0.1, 0.3, 0.6):
            tasks.append(("task_warpshare", (p, nr)))
    # heavy tasks first
    tasks.sort(key=lambda t: {"task_pair": 0, "task_bern": 1}.get(t[0], 2))
    print(f"[run_g0] {len(tasks)} tasks, jobs={args.jobs}", flush=True)
    results, errors = [], []
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.jobs, maxtasksperchild=4) as pool:
        for i, (name, targs, res, err) in enumerate(pool.imap_unordered(_run, tasks)):
            if err:
                errors.append({"task": name, "args": [str(a) for a in targs], "error": err})
                print(f"[run_g0] ERROR {name}{targs}: {err}", flush=True)
            else:
                results.append((name, res))
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"[run_g0] {i+1}/{len(tasks)} done, {time.perf_counter()-t0:.0f}s", flush=True)
    if errors:
        with open(args.out + ".errors.json", "w") as f:
            json.dump(errors, f, indent=1)
        raise SystemExit(f"{len(errors)} task(s) failed; see {args.out}.errors.json")
    bern = [r for n, r in results if n == "task_bern"]
    pairs = [r for n, r in results if n == "task_pair"]
    kp = [r for n, r in results if n == "task_knownp"]
    ws = [r for n, r in results if n == "task_warpshare"]

    plants_out = {"P-Bern": eval_bern(bern)}
    for plant in ("P-Aniso", "P-Conf", "P-Rough"):
        plants_out[plant] = eval_slope_plant(plant, [r for r in pairs if r["plant"] == plant])
    cells = predictor_cells(pairs, variants)
    gauss_evals = {v: eval_gauss(cells, v) for v in variants}
    scores = variant_scores(cells, variants)
    fidelity = {v: estimator_fidelity(cells, v) for v in variants}
    decision = choose_variant(scores, fidelity, variants, preference)
    frozen = args.frozen or decision["chosen"]
    plants_out["P-Gauss"] = gauss_evals[frozen]
    plants_out["Known-p"] = {"table": eval_knownp(kp)}
    plants_out["WarpShare"] = {"rows": sorted(ws, key=lambda r: (r["p"], r["noise_rel"])),
                               "pass": all(abs(r["iso_expl_AB_cf"] - r["planted"]) <= 0.05 for r in ws)}
    gate_names = ["P-Bern", "P-Aniso", "P-Conf", "P-Rough", "P-Gauss"]
    failed = [n for n in gate_names if not plants_out[n]["pass"]]
    here = os.path.dirname(os.path.abspath(__file__))
    rep = {
        "meta": {"hrhv2_version": __version__, "spec_version": SPEC_VERSION,
                 "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "host": platform.node(), "python": sys.version.split()[0], "numpy": np.__version__,
                 "jobs": args.jobs, "reps_gauss": args.reps_gauss, "reps_slope": args.reps_slope,
                 "reps_knownp": args.reps_knownp, "quick": args.quick, "wall_s": time.perf_counter() - t0,
                 "run_g0_sha256": file_sha(os.path.join(here, "run_g0.py")),
                 "predict_sha256": file_sha(os.path.join(here, "predict.py")),
                 "plants_sha256": file_sha(os.path.join(here, "plants.py")),
                 "m_list": M_LIST, "m_fit": M_FIT, "p_list": P_LIST, "gauss_c": GAUSS_C,
                 "rough_kappa": ROUGH_KAPPA, "seed_key_prefix": KEY, "halves_salt": SALT,
                 "task_time_s": {"pairs_total": float(sum(r["time_s"] for r in pairs))}},
        "gate_definitions": {
            "P-Bern": "all ordinal rungs identical (Delta = 0 exactly); CKA (biased and debiased) strictly decreasing in beta; global-isotonic explained >= 0.999 (in-sample and cross-fitted)",
            "P-Aniso": "|OLS slope of fixed-k s_N on log2 m| < 0.03 per doubling, k in {1,3,10}, every p",
            "P-Conf": "fixed-k s_N strictly increasing over m = 256, 1024, 4096, k in {1,3,10}, every p",
            "P-Rough": "fixed-k s_N strictly decreasing over m = 256, 1024, 4096, k in {1,3,10}, every p",
            "P-Gauss": "|P1 - observed s_N| <= 0.02 on every cell with observed s_N in [0.05, 0.99] (frozen P1 variant)",
            "Known-p": "report Levina-Bickel bias at each (m, p)",
            "WarpShare": "supplementary: |global-isotonic explained (cross-fitted) - planted share| <= 0.05",
        },
        "overall": {"G0_SPEC7_pass": not failed, "failed": failed},
        "plants": plants_out,
        "predictor_cells": cells,
        "predictor_choice": {
            "variants": variants, "preference": preference,
            "rule": choose_variant.__doc__.split("rule (PREDICTOR_CHOICE.md):")[1].strip().replace("\n   ", ""),
            "gauss_gate": {v: {k: gauss_evals[v][k] for k in ("pass", "pass_k_ge_3", "max_abs_err", "mean_abs_err", "by_k")} for v in variants},
            "scores": scores, "estimator_fidelity": fidelity, "decision": decision, "frozen_for_gate": frozen,
            "fit_summaries_example": next((r["fit"] for r in pairs if r["plant"] == "P-Gauss"), None),
        },
        "notes": [
            "Observed s_N for the slope plants uses all anchors of each nested pool; P1/nulls are cross-fitted: "
            "coupling fitted on H1 x H1 pairs of the m=1024 prefix, evaluated on H2 anchors of every m.",
            "'oracle' = the same crossing-law (mu) computation with the TRUE P-Gauss coupling (b = u + sigma N(0,1)). "
            "Errors shared by P1 and the oracle are properties of the mu-law approximation, not of the coupling estimator.",
            "N0 = s_N of H1 anchors at m=1024 (full rows); N1 = Gaussian additive noise with sigma from the global affine "
            "residual on H1 x H1; N2 = per-anchor Gaussian copula; P2 = (p/4) mean delta_D law with MG Levina-Bickel p-hat "
            "at scale max(k,3) (no bias correction).",
            "Bern plant uses a spherical cap (max D_A ~ 0.16) so that D_A -> D_B stays strictly increasing in float64 "
            "(float collisions reported per cell, must be 0).",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    from .run_pair import jsonable
    with open(args.out + ".json.tmp", "w") as f:
        json.dump(jsonable(rep), f, indent=1, allow_nan=False)
    os.replace(args.out + ".json.tmp", args.out + ".json")
    rep_json = json.load(open(args.out + ".json"))
    write_md(rep_json, args.out + ".md")
    print(f"[run_g0] wrote {args.out}.json/.md; G0 SPEC7 pass={not failed} failed={failed}; "
          f"chosen P1={decision['chosen']}; wall {time.perf_counter()-t0:.0f}s", flush=True)
    return rep


if __name__ == "__main__":
    main()
