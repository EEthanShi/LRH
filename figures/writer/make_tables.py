#!/usr/bin/env python
"""Writer role (Section 7 draft): generate LaTeX table bodies for the appendix draft.

Reads frozen artifacts only (read-only); writes under ~/hrh_v2_figs/writer/out/.
Every number in the generated tables comes from the input files listed in
out/MANIFEST.json, whose sha256 are recorded at run time.
Run: taskset -c 0-2 nice ~/hrh_v2_figs_venv/bin/python make_tables.py
"""
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd

H = os.path.expanduser("~")
S1 = H + "/hrh_v2_stage1_20261008"
S2 = H + "/hrh_v2_stage2_20261008"
OUT = H + "/hrh_v2_figs/writer/out"
os.makedirs(OUT, exist_ok=True)

INPUTS = {
    "s22_recompute": S2 + "/results/review_final/s22-recompute/recompute_summary.json",
    "s22_pairs": S2 + "/results/review_final/s22-recompute/recompute_pairs.json",
    "s22_summary": S2 + "/results/S2_2/s22_summary.json",
    "eps_s22": S2 + "/prereg/eps_S2_2.json",
    "eps_sens": S2 + "/results/review_final/s22-recompute/eps_sensitivity.json",
    "s22_skeptic_a2b": S2 + "/results/review_final/s22-skeptic/a2b_summary.json",
    "s13a": S1 + "/results/S1_3a_linae_summary.json",
    "s11_summary": S1 + "/results/S1_1/v11/S1_1_v11_summary.json",
    "s11_cells": S1 + "/results/S1_1/v11/S1_1_v11_cells_clip.parquet",
    "s11_slopes": S1 + "/results/S1_1/v11/S1_1_v11_slopes_clip.parquet",
    "s11_curves": S1 + "/results/S1_1/v11/S1_1_v11_curves_clip.parquet",
    "s12_summary": S1 + "/results/S1_2/s12_summary.json",
    "layermax_summary": S1 + "/results/S1_2_layermax/layermax_summary.json",
}


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


manifest = {"inputs": {k: {"path": v, "sha256": sha(v)} for k, v in INPUTS.items()},
            "outputs": {}, "checks": {}}


def write(name, text):
    p = OUT + "/" + name
    with open(p, "w") as f:
        f.write(text)
    manifest["outputs"][name] = sha(p)


def jload(k):
    with open(INPUTS[k]) as f:
        return json.load(f)


def f2(x, nd=2):
    s = f"{x:.{nd}f}"
    if s.startswith("-0." + "0" * nd) or s == "-" + "0." + "0" * nd:
        s = s[1:]
    return s.replace("-", "$-$") if s.startswith("-") else s


def fsig(x, nd=3):
    """Format with nd decimals; minus sign in math."""
    s = f"{x:.{nd}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return s.replace("-", "$-$")


# ---------------------------------------------------------------- S2.2 full table
r = jload("s22_recompute")
T = r["tables"]
assert r["mismatches"] in ([], 0, {}, None) or len(r["mismatches"]) == 0, r["mismatches"]
manifest["checks"]["s22_mismatches"] = r["mismatches"]

RUNGS = [("dN_k10", r"$T_{N,10}$"), ("dO", r"$T_O$"), ("dE", r"$T_E$"),
         ("dMST", r"$T_{\mathrm{MST}}$"), ("dSOG_t0p1", r"SOG$_{0.1}$"),
         ("dAff", r"$T_{G/\mathrm{Aff}}$"), ("Gs", r"$G_s$"), ("dCKA", r"$1-\mathrm{CKA}$"),
         ("dPi", r"$\Pi$")]
SETS = [("noWD_25000", "noWD", "25k"), ("WD_25000", "WD", "25k"),
        ("smallinit_25000", "small init", "25k"),
        ("noWD-shared_25000", "noWD, shared", "25k"),
        ("noWD_250000", "noWD", "250k"), ("WD_250000", "WD", "250k"),
        ("smallinit_250000", "small init", "250k")]
REPS = [("y", r"$y$"), ("u", r"$u$"), ("h", r"$h$"), ("h_l2", r"$h$, $\ell_2$")]
DEC = {"rel": r"\textsc{r}", "inc": r"\textsc{i}", "unr": r"\textsc{u}"}

lines = []
for key, arm, N in SETS:
    for i, (rep, rlab) in enumerate(REPS):
        row = T[f"{key}|{rep}"]
        cells = []
        for rk, _ in RUNGS:
            c = row[rk]
            assert c["match"] is True, (key, rep, rk)
            assert c["n"] == 96
            cells.append(f"{c['x']}\\,{DEC[c['dec']]}")
        lead = f"{arm} & {N}" if i == 0 else " & "
        lines.append(f"{lead} & {rlab} & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace[2pt]")
lines = lines[:-1]
head = "Arm & $N$ & Rep. & " + " & ".join(lab for _, lab in RUNGS) + r" \\"
write("tab_s22_full.tex", head + "\n\\midrule\n" + "\n".join(lines) + "\n")

# tolerances
E = jload("eps_s22")
eps_lines = []
for rep, rlab in [("y", r"$y$ and $u$"), ("h", r"$h$"), ("h_l2", r"$h$, $\ell_2$")]:
    vals = [E["eps"][rep][rk] for rk, _ in RUNGS]
    eps_lines.append(rlab + " & " + " & ".join(f"{v:.4f}" for v in vals) + r" \\")
    if rep == "y":
        for rk, _ in RUNGS:
            assert abs(E["eps"]["u"][rk] - E["eps"]["y"][rk]) < 1e-15
write("tab_s22_eps.tex", "Rep. & " + " & ".join(lab for _, lab in RUNGS) + r" \\" + "\n\\midrule\n"
      + "\n".join(eps_lines) + "\n")
manifest["checks"]["eps_n_pairs"] = E["n_pairs"]
manifest["checks"]["eps_cohort"] = E["cohort"]

# p-hat at multiples of eps, 250k and 25k, u and y, three rungs
C = r["curves"]
mults = ["0.9", "1.0", "1.1"]
curve_lines = []
for rk, rlab in [("Gs", r"$G_s$"), ("dCKA", r"$1-\mathrm{CKA}$"), ("dN_k10", r"$T_{N,10}$")]:
    first = True
    for key, arm, N in [s for s in SETS if s[0] != "noWD-shared_25000"]:
        if N != "250k":
            continue
        for rep in ["u", "y"]:
            c = C[f"{key}|{rep}|{rk}"]
            pm = [c["p_at_mult"][m] for m in mults]
            lead = rlab if first else ""
            first = False
            curve_lines.append(f"{lead} & {arm} & ${rep}$ & " + " & ".join(f"{v:.2f}" for v in pm)
                               + f" & {c['eps_needed_rel_ratio']:.2f}" + r" \\")
    curve_lines.append(r"\addlinespace[2pt]")
curve_lines = curve_lines[:-1]
write("tab_s22_curves.tex",
      r"Rung & Arm & Rep. & $\hat p(0.9\varepsilon)$ & $\hat p(\varepsilon)$ & $\hat p(1.1\varepsilon)$ & $\varepsilon_{\rm rel}/\varepsilon$ \\"
      + "\n\\midrule\n" + "\n".join(curve_lines) + "\n")

# secondary eps (pilot 0.9 quantile) at 250k, u, Gs and CKA
ES = jload("eps_sens")["q90"]
sec_lines = []
for key, arm, N in SETS:
    if N != "250k":
        continue
    for rk, rlab in [("Gs", r"$G_s$"), ("dCKA", r"$1-\mathrm{CKA}$")]:
        e = ES[f"{key}|u|{rk}"]
        sec_lines.append(f"{arm} & {rlab} & {e['eps_q90']:.4f} & {e['x']}\\,{DEC[e['dec']]} & "
                         f"{T[key + '|u'][rk]['x']}\\,{DEC[T[key + '|u'][rk]['dec']]}" + r" \\")
write("tab_s22_eps_secondary.tex",
      r"Arm & Rung & $\varepsilon_{0.9}$ & $x$ at $\varepsilon_{0.9}$ & $x$ at $\varepsilon$ \\" + "\n\\midrule\n"
      + "\n".join(sec_lines) + "\n")

# paired excess Delta(u) - Delta(y), medians, from per-pair ladder values
P = jload("s22_pairs")
ex_rungs = [("dN_k10", r"$T_{N,10}$"), ("dO", r"$T_O$"), ("dE", r"$T_E$"), ("dMST", r"$T_{\mathrm{MST}}$"),
            ("dAff", r"$T_{G/\mathrm{Aff}}$"), ("Gs", r"$G_s$"), ("dCKA", r"$1-\mathrm{CKA}$")]
ex_lines = []
ex_store = {}
for key, arm, N in SETS:
    pairs = P["sets"][key]
    vals = []
    for rk, _ in ex_rungs:
        d = []
        for a, b in pairs:
            rec = P["pairs"][f"{a}__{b}"]["ladder"]
            d.append(rec["u"][rk] - rec["y"][rk])
        d = np.asarray(d)
        assert len(d) == 96, (key, len(d))
        med = float(np.median(d))
        frac = float(np.mean(d > 0))
        ex_store[f"{key}|{rk}"] = {"median": med, "frac_u_gt_y": frac}
        vals.append(f"{fsig(med, 4)} ({int(round(frac * 96))})")
    ex_lines.append(f"{arm} & {N} & " + " & ".join(vals) + r" \\")
manifest["checks"]["paired_excess"] = ex_store
# cross-check against the recompute's own exploratory medians
for key, _, _ in SETS:
    for rk in ["dN_k10", "Gs", "dCKA"]:
        ref = r["exploratory"][f"{key}|{rk}"]["median_u_minus_y"]
        mine = ex_store[f"{key}|{rk}"]["median"]
        assert abs(ref - mine) < 1e-6, (key, rk, ref, mine)
write("tab_s22_excess.tex",
      "Arm & $N$ & " + " & ".join(lab for _, lab in ex_rungs) + r" \\" + "\n\\midrule\n"
      + "\n".join(ex_lines) + "\n")

# imbalance and cond(W)
S22 = jload("s22_summary")
cw = r["condW_by_cell"]
cov_lines = []
for key, arm, N in SETS:
    q = cw[key]
    cov_lines.append(f"{arm} & {N} & {q['0.5']:.2f} & {q['0']:.2f}--{q['1']:.2f}" + r" \\")
write("tab_s22_condW.tex", r"Arm & $N$ & median cond$(W)$ & range \\" + "\n\\midrule\n" + "\n".join(cov_lines) + "\n")

# decomposition of the Gs and CKA excess (review-role exploratory analysis)
A2 = jload("s22_skeptic_a2b")
dec_lines = []
for key, arm, N in SETS:
    v = A2[key]
    for rk, rlab in [("Gs", r"$G_s$"), ("dCKA", r"$1-\mathrm{CKA}$")]:
        g = v[rk]
        dec_lines.append(f"{arm} & {N} & {rlab} & {fsig(g['excess_total'], 4)} & {fsig(g['excess_common_pairgauge'], 4)} & "
                         f"{fsig(g['excess_mismatch'], 4)}" + r" \\")
write("tab_s22_decomp.tex",
      r"Arm & $N$ & Rung & total excess & common gauge & mismatch \\" + "\n\\midrule\n" + "\n".join(dec_lines) + "\n")

# ---------------------------------------------------------------- S1.3a decision table
L = jload("s13a")
cells = L["cells"]
ARMS = [("P-GD", "GD, step 0.05 (prereg.)"), ("P-GD-flow", "GD, step 0.002 (secondary)"),
        ("WD", "weight decay"), ("WD-flow", "weight decay, step 0.002 (sec.)"),
        ("TIED", "tied weights"), ("small-init", "small init"),
        ("P-SGD", "SGD, batch 16"), ("P-Adam", "Adam, batch 16")]
NS = [("256", "$2^{8}$"), ("4096", "$2^{12}$"), ("65536", "$2^{16}$"), ("inf", r"$\infty$")]
CRIT = ["kNN", "Gs", "CKA", "proj"]
DEC3 = {"reliable": r"\textsc{r}", "inconclusive": r"\textsc{i}", "unreliable": r"\textsc{u}"}
l_lines = []
for arm, lab in ARMS:
    for j, (N, Nlab) in enumerate(NS):
        c = cells[f"{arm}|{N}"]["primary"]
        vals = [f"{c[q]['passes']}/{c[q]['n_pairs']}\\,{DEC3[c[q]['decision']]}" for q in CRIT]
        lead = lab if j == 0 else ""
        l_lines.append(f"{lead} & {Nlab} & " + " & ".join(vals) + r" \\")
    l_lines.append(r"\addlinespace[2pt]")
l_lines = l_lines[:-1]
write("tab_s13a.tex",
      r"Training rule & $N$ & $1-s_N(10)\le0.05$ & $G_s\le0.02$ & $1-\mathrm{CKA}\le0.01$ & projector $\le0.01$ \\"
      + "\n\\midrule\n" + "\n".join(l_lines) + "\n")
manifest["checks"]["s13a_verdicts"] = {k: (v.get("verdict") if isinstance(v, dict) else v) for k, v in L["verdicts"].items()}

# imbalance and anisotropy table at N = inf
imb_lines = []
for arm, lab in ARMS:
    rr = cells[f"{arm}|inf"]["runs"]
    imb_lines.append(lab)
manifest["checks"]["s13a_runs_keys"] = list(cells["P-GD|inf"]["runs"].keys())

# ---------------------------------------------------------------- S1.1 tables
S = jload("s11_summary")
cfg = S["configs"]["clip"]
manifest["checks"]["s11_summary_created_utc"] = S["created_utc"]
manifest["checks"]["s11_mode"] = S["mode"]
CLS = [("seed-seed", "seed--seed"), ("cross-vision", "cross-vision"), ("cross-modal", "cross-modal")]
sc = cfg["slope_classes"]
cls_lines = []
setnames = list(sc.keys())
manifest["checks"]["s11_slope_class_sets"] = setnames
for setkey, setlab in [("primary", "primary"), (setnames[1], "all seed pairs")]:
    for cl, clab in CLS:
        if setkey != "primary" and cl != "seed-seed":
            continue
        vals = []
        for k in ["1", "10", "100"]:
            e = sc[setkey][cl][k]
            vals.append(f"{e['rising']}/{e['flat']}/{e['falling']} & {fsig(e['median_slope'], 3)}")
        n = sc[setkey][cl]["10"]["n"]
        lab = clab if setkey == "primary" else clab + " (all)"
        cls_lines.append(f"{lab} & {n} & " + " & ".join(vals) + r" \\")
write("tab_s11_classes.tex",
      r"Pair class & pairs & \multicolumn{2}{c}{$k=1$} & \multicolumn{2}{c}{$k=10$} & \multicolumn{2}{c}{$k=100$} \\"
      + "\n" + r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}" + "\n"
      + r" & & r/f/d & median & r/f/d & median & r/f/d & median \\" + "\n\\midrule\n"
      + "\n".join(cls_lines) + "\n")

# class medians of s_N per m and k, primary pairs, with [min, max]; chance row
cellsdf = pd.read_parquet(INPUTS["s11_cells"])
prim = cellsdf[cellsdf["primary"]]
MS = [256, 512, 1024, 2048, 4096, 8192, 16384]
curve_lines = []
store = {}
for k in [1, 10, 100]:
    sub = prim[prim["k"] == k]
    for cl, clab in CLS:
        vals = []
        for m in MS:
            x = sub[(sub["cls"] == cl) & (sub["m"] == m)]["obs"].to_numpy()
            assert len(x) == {"seed-seed": 4, "cross-vision": 10, "cross-modal": 15}[cl], (k, cl, m, len(x))
            med = float(np.median(x))
            store[f"{cl}|{k}|{m}"] = [med, float(x.min()), float(x.max())]
            vals.append(f"{med:.3f}" if med >= 0.01 else f"{med:.4f}")
        lead = f"$k={k}$" if cl == "seed-seed" else ""
        curve_lines.append(f"{lead} & {clab} & " + " & ".join(vals) + r" \\")
    ch = [float(sub[sub["m"] == m]["chance"].iloc[0]) for m in MS]
    for m, c in zip(MS, ch):
        assert abs(c - k / (m - 1)) < 1e-12
    curve_lines.append(f" & chance $k/(m-1)$ & " + " & ".join(f"{c:.4f}" for c in ch) + r" \\")
    curve_lines.append(r"\addlinespace[2pt]")
curve_lines = curve_lines[:-1]
manifest["checks"]["s11_class_medians_primary"] = store
write("tab_s11_curves.tex",
      r" & Pair class & " + " & ".join(f"$m={m}$" for m in MS) + r" \\" + "\n\\midrule\n"
      + "\n".join(curve_lines) + "\n")

# predictor accuracy
acc = cfg["accuracy"]["primary"]
acc_lines = []
for pred, plab in [("N0", "N0 (no change)"), ("N1", "N1 (affine fit with noise)"),
                   ("iii-LSlog-EB (unvalidated)", "crossing law (unvalidated)")]:
    vals = []
    for cl, _ in CLS:
        e = acc[pred][cl]
        vals.append(f"{e['median_abs_log_ratio']:.3f} ($\\times${math.exp(e['median_abs_log_ratio']):.2f}) & "
                    f"{e['n_sign_correct']}/{e['n_nonflat_units']}")
    acc_lines.append(f"{plab} & " + " & ".join(vals) + r" \\")
write("tab_s11_accuracy.tex",
      r"Predictor & \multicolumn{2}{c}{seed--seed} & \multicolumn{2}{c}{cross-vision} & \multicolumn{2}{c}{cross-modal} \\"
      + "\n" + r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" + "\n"
      + r" & level error & sign & level error & sign & level error & sign \\" + "\n\\midrule\n"
      + "\n".join(acc_lines) + "\n")

# elasticities
sec = cfg["secondary"]["primary"]
el = sec["elasticity_by_class"]
el_lines = []
for cl, clab in CLS:
    e = el[cl]
    el_lines.append(f"{clab} & {e['n']} & {fsig(e['median_e_k'])} & {fsig(e['median_e_m'])} & {fsig(e['median_sum'])} & {e['median_rmse']:.3f}" + r" \\")
manifest["checks"]["s11_corr_ek_minus_em"] = sec["corr_ek_minus_em"]
write("tab_s11_elasticity.tex",
      r"Pair class & pairs & median $e_k$ & median $e_m$ & median $e_k+e_m$ & median fit RMSE \\" + "\n\\midrule\n"
      + "\n".join(el_lines) + "\n")

# per pair, primary
sl = pd.read_parquet(INPUTS["s11_slopes"])
slp = sl[sl["primary"]]
pp_lines = []
order = {"seed-seed": 0, "cross-vision": 1, "cross-modal": 2}
pairs = sorted(slp["pair"].unique(), key=lambda p: (order[slp[slp["pair"] == p]["cls"].iloc[0]], p))
SYM = {"rising": "+", "flat": "0", "falling": r"$-$"}
for p in pairs:
    rows = slp[slp["pair"] == p].set_index("k")
    cl = rows["cls"].iloc[0]
    c10 = prim[(prim["pair"] == p) & (prim["k"] == 10)].set_index("m")["obs"]
    sN = " / ".join(f"{c10[m]:.3f}" for m in [256, 1024, 16384])
    slopes = " / ".join(fsig(rows.loc[k, "slope_obs"]) for k in [1, 10, 100])
    clsym = "/".join(SYM[rows.loc[k, "slope_class"]] for k in [1, 10, 100])
    pname = p.replace("_", r"\_").replace(r"\_\_", r" $\times$ ")
    pp_lines.append(f"{pname} & {dict(CLS)[cl]} & {sN} & {slopes} & {clsym}" + r" \\")
write("tab_s11_pairs.tex",
      r"Pair & class & $s_N(10)$ at $m=256$ / $1024$ / $16384$ & slope at $k=1$ / $10$ / $100$ & class \\"
      + "\n\\midrule\n" + "\n".join(pp_lines) + "\n")

# configuration sensitivity
cd = S["config_deltas_vs_clip"]
cd_lines = []
for cf, lab in [("l2", r"$\ell_2$ preprocessing"), ("cliplayer", "clip threshold per layer"),
                ("dedup", "duplicates removed")]:
    e = cd[cf]
    changes = "; ".join(x.replace("_", r"\_").replace("->", r"$\to$") for x in e["class_changes"]) or "none"
    cd_lines.append(f"{lab} & {e['n_units']} & {e['median_abs_dslope']:.3f} & {e['max_abs_dslope']:.3f} & {e['n_class_changes']} ({changes})" + r" \\")
write("tab_s11_sens.tex",
      r"Configuration & units & median $|\Delta\text{slope}|$ & max $|\Delta\text{slope}|$ & class changes \\" + "\n\\midrule\n"
      + "\n".join(cd_lines) + "\n")

# ---------------------------------------------------------------- S1.2 PE2 / PE3
D = jload("s12_summary")
pe2_lines = []
for pool in ["m1024", "m2047"]:
    for rr in ["0.5", "1.0"]:
        key = f"{pool}|r={rr}|prh_clip+l2|A (SPEC-literal, F_A = LM forest)"
        e = D["pe2"][key]
        v = D["verdicts"]["PE2"][key]
        tb = e["Tr_bridge"]
        vtxt = v["verdict"].replace("\u2248", "$\\approx$")
        pe2_lines.append(
            f"{pool[1:]} & {rr} & {e['frac_forest_ge_0.70'][0]:.2f} & {e['median_forest_share'][0]:.3f} & "
            f"{e['frac_abs_sT_minus_sN1_le_0.03'][0]:.2f} & {fsig(tb['Tr'], 2)} ({tb['p_one']:.3f}) & "
            f"{e['frac_bridge_exceeds_null'][0]:.2f} & {e['mean_shared_bridges_A']:.2f} & {vtxt}"
            + r" \\")
write("tab_s12_pe2.tex",
      r"$m$ & $r$ & (a) frac.\ $\ge0.70$ & median share & (b) frac. & (c,d) $\mathrm{Tr}_{\rm bridge}$ ($p$) & (e) frac. & shared bridges & verdict \\"
      + "\n\\midrule\n" + "\n".join(pe2_lines) + "\n")

pe3_lines = []
for rr in ["0.5", "1.0"]:
    for pre in ["prh_clip+l2", "l2"]:
        e = D["pe3"][f"r={rr}|{pre}|biased"]
        v = D["verdicts"]["PE3"].get(f"r={rr}|{pre}|biased", {})
        verdict = v.get("verdict", "") if isinstance(v, dict) else str(v)
        pre_tex = pre.replace("_", "\\_")
        pe3_lines.append(
            f"{rr} & {pre_tex} & {fsig(e['Tr_raw']['Tr'], 2)} ({e['Tr_raw']['p_one']:.3f}) & "
            f"{fsig(e['Tr_eq']['Tr'], 2)} ({e['Tr_eq']['p_one']:.3f}) & {fsig(e['Tr_mKNN']['Tr'], 2)} ({e['Tr_mKNN']['p_one']:.3f}) & "
            f"{fsig(e['Tr_eq_minus_raw']['diff'], 2)} & {verdict}" + r" \\")
write("tab_s12_pe3.tex",
      r"$r$ & pre & $\mathrm{Tr}_{\rm raw}$ ($p$) & $\mathrm{Tr}_{\rm eq}$ ($p$) & $\mathrm{Tr}_{\rm mKNN}$ ($p$) & $\mathrm{Tr}_{\rm eq}-\mathrm{Tr}_{\rm raw}$ & verdict \\"
      + "\n\\midrule\n" + "\n".join(pe3_lines) + "\n")

# ---------------------------------------------------------------- layer-max trends
LM = jload("layermax_summary")
tr = LM["per_pre"]["prh_clip+l2"]["trends"]
lm_lines = []
for met, mlab in [("cka_biased", "CKA"), ("mknn", "mKNN"), ("cka_debiased", "debiased CKA (expl.)")]:
    for sub, slab in [("full", "layer-max"), ("fixed_r0.5", "fixed, $r=0.5$"), ("fixed_r1.0", "fixed, $r=1$")]:
        vals = []
        for sc_ in ["s_obs", "s_cal", "uncensored"]:
            t = tr[met][f"{sub}|{sc_}"]["tr_lm"]
            vals.append(f"{fsig(t['Tr'], 2)} ({t['p_one']:.3f})")
        lead = mlab if sub == "full" else ""
        lm_lines.append(f"{lead} & {slab} & " + " & ".join(vals) + r" \\")
    lm_lines.append(r"\addlinespace[2pt]")
lm_lines = lm_lines[:-1]
write("tab_layermax_tr.tex",
      r"Metric & layers & raw $s_{\rm obs}$ & $s_{\rm cal}$ & $s_{\rm obs}-\overline{s}_{\pi}$ \\" + "\n\\midrule\n"
      + "\n".join(lm_lines) + "\n")
rd = LM["per_pre"]["prh_clip+l2"]["reading"]
manifest["checks"]["layermax_reading"] = rd
manifest["checks"]["arh_table2_reference"] = LM["arh_table2_reference"]

# ---------------------------------------------------------------- S2.1 decision table (inserted 2026-10-09, writer v1)
import glob as _glob

INPUTS["s21_summary"] = S2 + "/results/S2_1/s21_summary.json"
INPUTS["eps_s21"] = S2 + "/prereg/eps_S2_1.json"
for _k in ["s21_summary", "eps_s21"]:
    manifest["inputs"][_k] = {"path": INPUTS[_k], "sha256": sha(INPUTS[_k])}

S21 = jload("s21_summary")
assert S21["interim"] is False and S21["R"] == 64, (S21["interim"], S21["R"])
manifest["checks"]["s21_verdict"] = S21["verdict"]
T21 = S21["tables"]
R21 = [("dN_k10", r"$T_{N,10}$"), ("dO", r"$T_O$"), ("dE", r"$T_E$"), ("dMST", r"$T_{\mathrm{MST}}$"),
       ("dSOG_t0p1", r"SOG$_{0.1}$"), ("dAff", r"$T_{G/\mathrm{Aff}}$"), ("Gs", r"$G_s$"),
       ("dCKA", r"$1-\mathrm{CKA}$"), ("dIso", r"$\Delta_{\rm iso}$"), ("dPi", r"$\Pi$")]
SETS21 = [("within_A", r"InfoNCE $\tau{=}0.2$, within"), ("within_Atau", r"InfoNCE $\tau{=}0.5$, within"),
          ("within_B", "spectral, within"), ("cross_A_B", r"InfoNCE $\tau{=}0.2$ vs spectral"),
          ("cross_A_Atau", r"InfoNCE $\tau{=}0.2$ vs $\tau{=}0.5$"), ("cross_Atau_B", r"InfoNCE $\tau{=}0.5$ vs spectral"),
          ("bern_self", r"link check, self"), ("bern_peer", r"link check, peer")]
DEC21 = {"reliable": r"\textsc{r}", "inconclusive": r"\textsc{i}", "unreliable": r"\textsc{u}"}
l21 = []
for Nk, Nlab in [("50000", "50k"), ("500000", "500k")]:
    for j, (sk, slab) in enumerate(SETS21):
        row = T21[f"{sk}_{Nk}|z"]
        cells = []
        for rk, _ in R21:
            c = row.get(rk)
            if c is None:
                cells.append("--")
                continue
            cells.append(f"{c['x']}/{c['n']}\\,{DEC21[c['decision']]}")
        lead = Nlab if j == 0 else ""
        l21.append(f"{lead} & {slab} & " + " & ".join(cells) + r" \\")
    l21.append(r"\addlinespace[2pt]")
l21 = l21[:-1]
write("tab_s21_full.tex", r"$N$ & Pair set & " + " & ".join(lab for _, lab in R21) + r" \\" + "\n\\midrule\n"
      + "\n".join(l21) + "\n")

E21 = jload("eps_s21")
manifest["checks"]["s21_eps_z"] = {rk: E21["eps"]["z"][rk] for rk, _ in R21}
manifest["checks"]["s21_eps_n_pairs"] = E21["n_pairs"]
ep = S21["endpoints"]
manifest["checks"]["s21_endpoints"] = {
    "i_p_cross": {"50k": ep["i"]["p_cross_Nmin"], "500k": ep["i"]["p_cross_Nmax"]},
    "i_ceiling": ep["i"]["ceiling_Nmax"], "i_gap": ep["i"]["gap_to_ceiling_Nmax"],
    "i_improvement": ep["i"]["improvement_500k_minus_50k"], "i_holm": ep["i"]["holm"], "i_pass": ep["i"]["pass"],
    "ii_pass": ep["ii"]["pass"], "iii": ep["iii"], "iv": ep["iv"], "v_pass": ep["v"]["pass"]}
us = S21["ustat"]
manifest["checks"]["s21_ustat_500k"] = {k: us[k] for k in us if k.endswith("_500000")}
manifest["checks"]["s21_ustat_50k"] = {k: us[k] for k in us if k.endswith("_50000")}
manifest["checks"]["s21_cka_cross_minus_within"] = S21["cka_cross_minus_within"]
manifest["checks"]["s21_beta"] = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "values"})
                                  for k, v in S21["beta"].items()}
manifest["checks"]["s21_covariates"] = S21["covariates"]

# ---------------------------------------------------------------- compute accounting (all on ds2213)
comp = {}
secs = {}
nunits = {}
for f in _glob.glob(S1 + "/results/S1_1/v11/units/*/*/*.json"):
    with open(f) as fh:
        d = json.load(fh)
    key = f"{d['config']}|{d['kind']}"
    secs[key] = secs.get(key, 0.0) + float(d["seconds"])
    nunits[key] = nunits.get(key, 0) + 1
comp["s11_v11_cpu_hours_by_config_kind"] = {k: v / 3600 for k, v in secs.items()}
comp["s11_v11_units"] = nunits
comp["s11_v11_cpu_hours_total"] = sum(secs.values()) / 3600


def runs_cost(d):
    metas = _glob.glob(d + "/*/meta.json")
    tt = []
    mt = []
    for f in metas:
        with open(f) as fh:
            m = json.load(fh)
        tt.append(float(m["t_total_s"]))
        mt.append(os.path.getmtime(f))
    return {"n_runs": len(metas), "sum_run_hours": sum(tt) / 3600,
            "median_run_s": float(np.median(tt)) if tt else None,
            "meta_mtime_span_hours": (max(mt) - min(mt)) / 3600 if mt else None}


for nm in ["S2_2_pilot", "S2_2_main", "S2_1_pilot", "S2_1_main"]:
    comp[nm] = runs_cost(S2 + "/runs/" + nm)
manifest["checks"]["compute"] = comp
print(json.dumps(comp, indent=1))

with open(OUT + "/MANIFEST.json", "w") as f:
    json.dump(manifest, f, indent=1, default=str)
print("wrote", len(manifest["outputs"]), "tables")
