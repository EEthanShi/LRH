"""E0 freeze role (e0-freeze), 2026-10-10: recompute from the FROZEN S1.1 outputs every constant the
frozen plan ANALYSIS_PLAN_E0.md adopts, and the operating characteristics of its decision rules.

Reads only frozen S1.1 outputs (cells/slopes parquets under clip and l2, summary JSON) and the designer's
design_inputs.json (for the 1e-12 cross-check). No feature array, no kNN cache, no sentence-encoder file.
Writes analysis/e0_sentence/freeze/freeze_constants.json.

    OMP_NUM_THREADS=1 nice -n 10 venv/bin/python -I analysis/e0_sentence/freeze/e0_freeze_constants.py
"""
import hashlib
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

P = os.path.expanduser("~/hrh_v2_stage1_20261008")
V11 = os.path.join(P, "results/S1_1/v11")
OUT = os.path.join(P, "analysis/e0_sentence/freeze/freeze_constants.json")
DESIGN = os.path.join(P, "analysis/e0_sentence/design/design_inputs.json")
VITS = ("augreg_b", "clip_b", "dinov2_b", "mae_b", "siglip_b")
CLS = ("seed-seed", "cross-vision", "cross-modal")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def ols(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(beta[0]), float(beta[1]), float(np.sqrt(r @ r / (len(x) - 2))), r


def load(pre):
    c = pd.read_parquet(os.path.join(V11, f"S1_1_v11_cells_{pre}.parquet"))
    S = pd.read_parquet(os.path.join(V11, f"S1_1_v11_slopes_{pre}.parquet"))
    return c[c.primary].copy(), S[S.primary].copy()


def table(c, S, k, mref):
    t = c[(c.m == mref) & (c.k == k)].set_index("pair")[["cls", "obs"]]
    t = t.join(S[S.k == k].set_index("pair")[["slope_obs", "slope_obs_draw_sd"]])
    t["lnL"] = np.log(t["obs"])
    t["vit"] = [p.split("__")[0] if c_ == "cross-modal" else None for p, c_ in zip(t.index, t.cls)]
    return t


def elasticities(c, ks):
    out = []
    for pair, g in c[c.k.isin(ks)].groupby("pair"):
        X = np.column_stack([np.ones(len(g)), np.log(g.k.to_numpy(float)), np.log(g.m.to_numpy(float))])
        y = np.log(np.maximum(g.obs.to_numpy(float), g.eps.to_numpy(float)))
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.append({"pair": pair, "cls": g.cls.iloc[0], "e_k": float(b[1]), "e_m": float(b[2]),
                    "sum": float(b[1] + b[2]), "n_cells": int(len(g))})
    return pd.DataFrame(out)


R = {"role": "e0-freeze", "created_utc": None, "inputs_sha256": {}, "note":
     "all numbers recomputed by the freeze role from frozen S1.1 outputs; constants adopted by ANALYSIS_PLAN_E0.md"}
for pre in ("clip", "l2"):
    for kind in ("cells", "slopes"):
        p = os.path.join(V11, f"S1_1_v11_{kind}_{pre}.parquet")
        R["inputs_sha256"][os.path.relpath(p, P)] = sha(p)
R["inputs_sha256"]["results/S1_1/v11/S1_1_v11_summary.json"] = sha(os.path.join(V11, "S1_1_v11_summary.json"))
R["inputs_sha256"]["analysis/e0_sentence/design/design_inputs.json"] = sha(DESIGN)
design = json.load(open(DESIGN))

# ---------------------------------------------------------------- A, B, C: calibration lines, competitors, within-class
R["calibration"] = {}
R["competitor_vit"] = {}
R["within_class"] = {}
R["design_crosscheck_maxabs"] = 0.0
for pre in ("clip", "l2"):
    c, S = load(pre)
    for k in (1, 10, 100):
        for mref in (1024, 2048):
            t = table(c, S, k, mref)
            assert len(t) == 29
            a, b, sd, r = ols(t.lnL.to_numpy(), t.slope_obs.to_numpy())
            key = f"{pre}|k{k}|m{mref}"
            R["calibration"][key] = {"a": a, "b": b, "resid_sd": sd, "n": 29,
                                     "resid_max_abs": float(np.abs(r).max()),
                                     "resid_mean_by_class": {cl: float(r[(t.cls == cl).to_numpy()].mean()) for cl in CLS},
                                     "anchored_powerlaw_coef_1_over_ln_mref_over_k": float(1 / np.log(mref / k))}
            if mref == 1024:
                d = design["s11_level_slope"][pre][str(k)]
                R["design_crosscheck_maxabs"] = max(R["design_crosscheck_maxabs"], abs(a - d["a"]), abs(b - d["b"]),
                                                    abs(sd - d["resid_sd_ddof2"]))
                vl = t[t.cls == "cross-modal"]
                R["competitor_vit"][f"{pre}|k{k}"] = {
                    "beta_M_v": {v: float(vl[vl.vit == v].slope_obs.mean()) for v in VITS},
                    "level_M_v_m1024": {v: float(vl[vl.vit == v].obs.mean()) for v in VITS},
                    "vl_median_slope": float(vl.slope_obs.median()),
                    "vl_level_range_m1024": [float(vl.obs.min()), float(vl.obs.max())],
                    "vl_median_level_m1024": float(vl.obs.median()),
                    "vl_slope_draw_sd_median": float(vl.slope_obs_draw_sd.median())}
                wc = {}
                num = den = 0.0
                for cl in CLS:
                    g = t[t.cls == cl]
                    x = g.lnL.to_numpy() - g.lnL.mean()
                    y = g.slope_obs.to_numpy() - g.slope_obs.mean()
                    wc[cl] = float(x @ y / (x @ x))
                    num += x @ y
                    den += x @ x
                wc["pooled_within_class"] = float(num / den)
                cent = t.groupby("cls")[["lnL", "slope_obs"]].mean()
                wc["centroid_line"] = list(ols(cent.lnL.to_numpy(), cent.slope_obs.to_numpy())[:2])
                R["within_class"][f"{pre}|k{k}"] = wc
R["design_crosscheck_pass_1e-12"] = bool(R["design_crosscheck_maxabs"] <= 1e-12)

# ---------------------------------------------------------------- D: elasticity sums by k subset
R["elasticity"] = {}
for pre in ("clip", "l2"):
    c, _ = load(pre)
    for name, ks in (("k1_10_100", (1, 10, 100)), ("k10_100", (10, 100)), ("k1_10", (1, 10))):
        el = elasticities(c, ks)
        R["elasticity"][f"{pre}|{name}"] = {
            cl: {"median_sum": float(g["sum"].median()), "max_abs_sum": float(g["sum"].abs().max()),
                 "median_e_m": float(g.e_m.median()), "min_abs_e_m": float(g.e_m.abs().min()),
                 "n": int(len(g))} for cl, g in el.groupby("cls")}
        if name == "k10_100":
            R["elasticity"][f"{pre}|{name}"]["per_pair"] = el.round(6).to_dict("records")
summ = json.load(open(os.path.join(V11, "S1_1_v11_summary.json")))
el_s = summ["configs"]["clip"]["secondary"]["primary"]["elasticity_by_class"]
R["elasticity_crosscheck_vs_summary_maxabs"] = max(
    abs(el_s[cl]["median_sum"] - R["elasticity"]["clip|k1_10_100"][cl]["median_sum"]) for cl in CLS)


# ---------------------------------------------------------------- frozen rules (copied verbatim into e0_rules.py)
def bhat_star(lnL, a, b):
    return np.minimum(a + b * lnL, 0.0)


def p1_rule(sums, ems):
    sums, ems = np.asarray(sums, float), np.asarray(ems, float)
    inf = np.abs(ems) >= 0.10
    if inf.sum() < 8:
        return "uninformative"
    s = sums[inf]
    med = abs(float(np.median(s)))
    frac = float(np.mean(np.abs(s) <= 0.05))
    if med <= 0.03 and frac >= 0.80:
        return "holds"
    if med > 0.06 or frac <= 0.50:
        return "fails"
    return "partly holds"


def p2b_rule(g, y, enc, sres):
    """g = bhat*(L_j) - beta_Mv(j), y = b_j - beta_Mv(j), enc = encoder label per pair."""
    g, y, enc = np.asarray(g, float), np.asarray(y, float), np.asarray(enc)
    if np.median(np.abs(g)) < 2 * sres:
        return "uninformative", np.nan
    lam = float(g @ y / (g @ g))
    lam_e = []
    for e in sorted(set(enc)):
        s = enc == e
        if np.median(np.abs(g[s])) >= 2 * sres:
            lam_e.append(float(g[s] @ y[s] / (g[s] @ g[s])))
    if len(lam_e) < 2:
        return "inconclusive", lam
    lam_e = np.array(lam_e)
    if 0.75 <= lam <= 1.25 and np.all((lam_e >= 0.5) & (lam_e <= 1.5)):
        return "holds", lam
    if lam <= 0.25 and np.all(lam_e <= 0.5):
        return "fails", lam
    return "inconclusive", lam


def p2c_rule(gam, dln, enc):
    """gam = per-pair OLS slope of b_{j,n} on ln L_{j,n}; dln = ln(L_{j,5}/L_{j,1})."""
    gam, dln, enc = np.asarray(gam, float), np.asarray(dln, float), np.asarray(enc)
    if np.median(dln) < 0.3:
        return "uninformative"
    ge = [float(np.median(gam[enc == e])) for e in sorted(set(enc)) if np.median(dln[enc == e]) >= 0.3]
    if len(ge) < 2:
        return "inconclusive"
    ge = np.array(ge)
    med = float(np.median(gam))
    if 0.17 <= med <= 0.29 and np.all((ge >= 0.154) & (ge <= 0.342)):
        return "holds"
    if med <= 0.12 and np.all(ge <= 0.154):
        return "fails"
    return "inconclusive"


# ---------------------------------------------------------------- P1 on S1.1 (sanity: what the frozen rule says on old classes)
c, _ = load("clip")
el = elasticities(c, (10, 100))
R["P1_rule_on_S11_clip"] = {cl: p1_rule(g["sum"], g.e_m) for cl, g in el.groupby("cls")}
R["P1_rule_on_S11_clip_informative_n"] = {cl: int((g.e_m.abs() >= 0.10).sum()) for cl, g in el.groupby("cls")}

# ---------------------------------------------------------------- guard: level at which the predicted gap reaches 2 sd
cal = R["calibration"]["clip|k10|m1024"]
a10, b10, sres = cal["a"], cal["b"], cal["resid_sd"]
bMv = R["competitor_vit"]["clip|k10"]["beta_M_v"]
grid = np.exp(np.linspace(np.log(0.005), np.log(0.9), 200001))
R["guard_level_thresholds"] = {}
for v in VITS:
    gap = np.abs(bhat_star(np.log(grid), a10, b10) - bMv[v])
    up = grid[(gap >= 2 * sres) & (grid > np.exp((bMv[v] - a10) / b10))]
    lo = grid[(gap >= 2 * sres) & (grid < np.exp((bMv[v] - a10) / b10))]
    R["guard_level_thresholds"][v] = {"crossing_level": float(np.exp((bMv[v] - a10) / b10)),
                                      "informative_if_L_at_least": float(up.min()) if up.size else None,
                                      "informative_if_L_at_most": float(lo.max()) if lo.size else None}
R["bhat_star_at"] = {str(L): float(bhat_star(np.log(L), a10, b10)) for L in (0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)}

# ---------------------------------------------------------------- G: operating characteristics of the frozen P2b rule
t = table(*load("clip"), 10, 1024)
vl = t[t.cls == "cross-modal"]
vitL = np.log(vl.groupby("vit").obs.mean().reindex(VITS).to_numpy())
vitL -= vitL.mean()
Mv = np.repeat(np.array([bMv[v] for v in VITS])[:, None], 3, 1)
enc = np.tile(np.arange(3), (5, 1)).ravel()
rng = np.random.default_rng(int(hashlib.sha256(b"e0-freeze-oc-p2b").hexdigest()[:16], 16))
SIM = {}
NREP = 2000
for medL in (0.06, 0.09, 0.12, 0.2, 0.3, 0.45):
    for lam_true in (1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.0, -0.25):
        for sd_enc in (0.0, 0.05, 0.10):
            labs = []
            for _ in range(NREP):
                L = np.exp(np.log(medL) + 0.5 * vitL[:, None] + rng.normal(0, 0.25, 3)[None, :]).clip(0.005, 0.9)
                bh = bhat_star(np.log(L), a10, b10)
                bo = lam_true * bh + (1 - lam_true) * Mv + rng.normal(0, sd_enc, 3)[None, :] + rng.normal(0, sres, (5, 3))
                labs.append(p2b_rule((bh - Mv).ravel(), (bo - Mv).ravel(), enc, sres)[0])
            SIM[f"L{medL}|lam{lam_true}|encsd{sd_enc}"] = {l: labs.count(l) / NREP for l in
                                                         ("holds", "inconclusive", "fails", "uninformative")}
R["oc_P2b"] = {"assumptions": "levels: median medL, ViT spread 0.5 x S1.1 VL ViT ln-level spread, encoder ln-level sd 0.25; "
                              "observed slope = lam*bhat* + (1-lam)*beta_Mv + encoder shift N(0, encsd) + pair noise "
                              "N(0, S1.1 residual sd); 2000 runs per cell", "sim": SIM}

# ---------------------------------------------------------------- H: operating characteristics of the frozen P2c rule
rng = np.random.default_rng(int(hashlib.sha256(b"e0-freeze-oc-p2c").hexdigest()[:16], 16))
f = np.array([0.0, 0.5, 0.75, 0.9, 1.0])     # assumed concave ladder of ln L_{j,n} - ln L_{j,1} in units of the n=5 gain
SIMC = {}
for dmed in (0.2, 0.3, 0.5, 0.8):
    for g_true in (0.30, 0.228, 0.17, 0.154, 0.12, 0.08, 0.0):
        for tau in (0.005, 0.01, 0.02, 0.04):
            for gsd_enc in (0.0, 0.05):
                dln = np.exp(np.log(dmed) + rng.normal(0, 0.25, (NREP, 15)))
                ge = g_true + rng.normal(0, gsd_enc, (NREP, 3))
                x = dln[:, :, None] * f[None, None, :]
                yb = ge[:, enc][:, :, None] * x + rng.normal(0, tau, (NREP, 15, 5))
                xc = x - x.mean(axis=2, keepdims=True)
                gam = (xc * (yb - yb.mean(axis=2, keepdims=True))).sum(2) / (xc * xc).sum(2)
                labs = [p2c_rule(gam[r], dln[r], enc) for r in range(NREP)]
                SIMC[f"dln{dmed}|gamma{g_true}|tau{tau}|encsd{gsd_enc}"] = {
                    l: labs.count(l) / NREP for l in ("holds", "inconclusive", "fails", "uninformative")}
R["oc_P2c"] = {"assumptions": "per pair: ln L_{j,n} = ln L_{j,1} + dln_j*f_n, f = (0, .5, .75, .9, 1), dln_j lognormal "
                              "(median dmed, sd 0.25 on the log); b_{j,n} = const + gamma_e*(ln L_{j,n}-ln L_{j,1}) + "
                              "N(0, tau); gamma_e = gamma + N(0, encsd); 2000 runs per cell", "sim": SIMC}

R["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(R, fh, indent=1)
print(OUT, sha(OUT))
