"""S2.2 v1.1 final analysis (PREREG_S2_2_v1.1.md sections 7-13). Runs ONCE on the complete main cohort.

    python src_v11/analyze_v11.py [--cohort S2_2v11_main] [--jobs 6]

Refuses to run unless: the code is frozen (guard_v11), the eps file exists and its sha256 is quoted by the
[s22v11-gates] 'eps_S2_2_v11 FROZEN' STATUS line, and every planned main run is .done or .failed (no INTERIM).
Blocks: C (conservation, Remark rem:optimizers), I (balanced map vs imbalance channel), A (decisions, Theorem
thm:gauge analogs, controls), K (Corollary cor:gauge-neighbors). Outputs results/S2_2_v11/{s22v11_summary.json,
S2_2_v11_report.md}; ladders cached in results/S2_2_v11/ladders/; K-block pair rows in results/S2_2_v11/kblock_pairs.json.
A cohort named UT_* is the EXCLUDED pipeline dry run (outputs under results/S2_2_v11/dryrun/, never evidence).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_v11 as CM  # noqa: E402
import cor3_v11 as K3  # noqa: E402
import decomp_v11 as DC  # noqa: E402
import gaugemath_v11 as GM  # noqa: E402
import make_plan_v11 as MP  # noqa: E402

STATUS = os.path.expanduser("~/hrh_v2_stage1_20261008/STATUS.md")
LADDER_ARMS = MP.CORE + MP.SECONDARY
PRIMARY_Q = ("dN_k10", "Gs")
NB = 10000


# ============================================================================ per-run quantities
def load_head(cohort, rid):
    d = CM.run_dir(cohort, rid)
    if not os.path.exists(os.path.join(d, ".done")):
        return None
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "feats.npz"))
    f = {k: np.asarray(z[k], np.float64) for k in ("P0", "W0", "P", "W")}
    return {"meta": meta, "feats": f}


def run_quantities(cohort, cells):
    out = {}
    for (arm, N), reps in cells.items():
        for rep, rid in sorted(reps.items()):
            r = load_head(cohort, rid)
            q = {"arm": arm, "N": N, "rep": rep, "state": CM.run_state(cohort, rid)}
            if r is None:
                q["ok"] = False
                out[rid] = q
                continue
            m, f = r["meta"], r["feats"]
            ok = (not m.get("diverged")) and m.get("finite") and all(np.all(np.isfinite(f[k])) for k in f)
            q.update({"ok": bool(ok), "diverged": m.get("diverged"), "acc_bank": m.get("acc_bank"), "acc_S": m.get("acc_S"),
                      "ce_train": m.get("ce_train"), "ce_hold": m.get("ce_hold"), "gen_gap": m.get("gen_gap"),
                      "disp_P_median": m.get("disp_P_median"), "disp_W_median": m.get("disp_W_median"),
                      "head_substeps_total": m.get("head_substeps_total"), "fp_fail": m.get("fp_fail"),
                      "euler_identity_resid": m.get("euler_identity_resid"), "u_recompute_relerr": m.get("u_recompute_relerr"),
                      "t_total_s": m.get("t_total_s")})
            if ok:
                try:
                    q.update(GM.run_gauge(f["P"], f["W"], f["P0"], f["W0"]))
                    if GM.spd_ok(GM.G_of(f["W"])) and GM.spd_ok(GM.G_of(f["W0"])):   # R-NUM
                        q["Mf"] = GM.norm_metric(f["W"]).tolist()
                        q["Mi"] = GM.norm_metric(f["W0"]).tolist()
                except Exception as e:  # singular head
                    q["gauge_error"] = repr(e)
            out[rid] = q
    return out


def vals_of(RQ, cells, arm, N, key, only_ok=True):
    reps = cells.get((arm, N), {})
    return [RQ[rid][key] for rep, rid in sorted(reps.items())
            if (RQ[rid].get("ok") or not only_ok) and key in RQ[rid] and RQ[rid][key] is not None
            and np.isfinite(RQ[rid][key])]


def frac_own_closer(RQ, cells, arm, N):
    """v1.0 a4_mechanism_eps.py definition with the exact saved W0 (rng default_rng(0) per cell)."""
    rids = [rid for rep, rid in sorted(cells[(arm, N)].items()) if RQ[rid].get("ok") and "Mf" in RQ[rid]]
    n = len(rids)
    if n < 3:
        return {"n": n}
    Mf = [np.array(RQ[r]["Mf"]) for r in rids]
    Mi = [np.array(RQ[r]["Mi"]) for r in rids]
    own = np.array([GM.d_ai(Mf[i], Mi[i]) for i in range(n)])
    rng = np.random.default_rng(0)
    other = np.array([GM.d_ai(Mf[i], Mi[(i + 1 + rng.integers(n - 1)) % n]) for i in range(n)])
    x = int(np.sum(own < other))
    return {"x": x, "n": n, "p": x / n, "cp95": CM.cp_two_sided(x, n), "d_own_mean": float(own.mean()),
            "d_other_mean": float(other.mean())}


# ============================================================================ C-block
def c_block(RQ, cells, choice):
    C = {}
    flow = {f"{a}_{N}": {"n": len(cells[(a, N)]), "n_ok": len(vals_of(RQ, cells, a, N, "delta_C")),
                         "delta_C_max": max(vals_of(RQ, cells, a, N, "delta_C") or [float("nan")]),
                         "n_viol": int(sum(v > 2e-4 for v in vals_of(RQ, cells, a, N, "delta_C")))}
            for a in MP.FLOW_ARMS for N in CM.NS if (a, N) in cells}
    C["C1"] = {"tier": "T; check", "cells": flow, "pass": all(v["n_viol"] == 0 and v["n_ok"] == v["n"] for v in flow.values()),
               "note": "a diverged / non-finite flow run has no delta_C and counts against pass"}
    eul = {}
    for a in ("SGD-K", "KS-4", "KS-16"):
        for N in CM.NS:
            if (a, N) in cells:
                v = vals_of(RQ, cells, a, N, "euler_identity_resid")
                eul[f"{a}_{N}"] = {"n": len(cells[(a, N)]), "n_ok": len(v), "max": max(v or [float("nan")]),
                                   "n_viol": int(sum(x > 1e-8 for x in v))}
    C["C2"] = {"tier": "T; check", "cells": eul, "pass": all(v["n_viol"] == 0 for v in eul.values())}
    c3a = {}
    for N in CM.NS:
        if ("BAL-MOM", N) in cells:
            rho = vals_of(RQ, cells, "BAL-MOM", N, "rho_end")
            b = CM.boot_median_ci(rho, f"C3a|{N}", level=0.95, n=NB)
            c3a[N] = {"median": b["median"], "lower_975": b["ci"][0], "frac_ge_0p02": float(np.mean(np.array(rho) >= 0.02)),
                      "n": len(rho), "pass": b["ci"][0] > 0.05 and float(np.mean(np.array(rho) >= 0.02)) >= 0.9}
    c3b = {}
    for N in CM.NS:
        g = vals_of(RQ, cells, "MOM", N, "growth_rel")
        b = CM.boot_median_ci(g, f"C3b|{N}", level=0.95, n=NB)
        c3b[N] = {"median": b["median"], "lower_975": b["ci"][0], "n": len(g), "pass": b["ci"][0] > 1}
    C["C3"] = {"tier": "R (Remark)", "a_BALMOM_rho": c3a, "b_MOM_growth": c3b,
               "a_pass": bool(c3a) and all(v["pass"] for v in c3a.values()), "b_pass": all(v["pass"] for v in c3b.values()),
               "source": "v1.0 smallinit ||C||_F 0.0098 -> 6.92, rho_end 0.24/0.28; v1.0 noWD growth 8.8-9.2 (KA6)"}
    c4a = {}
    for N in CM.NS:
        mu = vals_of(RQ, cells, "MOM", N, "mu_C")
        b = CM.boot_median_ci(mu, f"C4a|{N}", level=0.95, n=NB)
        c4a[N] = {"median": b["median"], "upper_975": b["ci"][1], "n": len(mu), "pass": b["ci"][1] < 0.5}
    flowmu = {N: float(np.median(vals_of(RQ, cells, "FLOW", N, "mu_C") or [float("nan")])) for N in CM.NS}
    C["C4a"] = {"tier": "R (source: KA6, v1.0 noWD median mu_C -0.005 / -0.025, upper 97.5% 0.018 / -0.003)",
                "MOM": c4a, "pass": all(v["pass"] for v in c4a.values()),
                "check_FLOW_median_mu_C": flowmu, "check_pass": all(v >= 0.99 for v in flowmu.values())}
    c4b = {f"{a}_{N}": frac_own_closer(RQ, cells, a, N) for a in ("FLOW", "MOM", "WD", "BAL-FLOW", "SMALL-FLOW")
           for N in CM.NS if (a, N) in cells}
    flow_has_half = [c4b[f"FLOW_{N}"]["cp95"][0] <= 0.5 <= c4b[f"FLOW_{N}"]["cp95"][1] for N in CM.NS if "cp95" in c4b.get(f"FLOW_{N}", {})]
    flow_above = [c4b[f"FLOW_{N}"]["cp95"][0] > 0.5 for N in CM.NS if "cp95" in c4b.get(f"FLOW_{N}", {})]
    mom_half = [c4b[f"MOM_{N}"]["cp95"][0] <= 0.5 <= c4b[f"MOM_{N}"]["cp95"][1] for N in CM.NS if "cp95" in c4b.get(f"MOM_{N}", {})]
    if flow_has_half and any(flow_has_half):
        read = "statistic insensitive: FLOW's CP interval contains 0.5 -> the v3 gauge-memory sentence loses its evidence"
    elif flow_above and all(flow_above) and mom_half and all(mom_half):
        read = "v1.0 reading stands: FLOW above 0.5 and MOM's interval contains 0.5"
    else:
        read = "neither pre-agreed reading applies: report only"
    C["C4b"] = {"tier": "E (calibration of the v1.0 statistic)", "cells": c4b, "reading": read}
    # C5: drift = delta_C
    c5 = {"medians": {}}
    for a in ("MOM", "SGD-K", "FLOW", "KS-4", "KS-16"):
        for N in CM.NS:
            if (a, N) in cells:
                c5["medians"][f"{a}_{N}"] = CM.boot_median_ci(vals_of(RQ, cells, a, N, "delta_C"), f"C5|{a}|{N}", n=NB)

    def ratio_ci(a1, a2, N, key):
        x = np.array(vals_of(RQ, cells, a1, N, "delta_C"))
        y = np.array(vals_of(RQ, cells, a2, N, "delta_C"))
        if x.size == 0 or y.size == 0:
            return None
        rng = CM.key_rng(key)
        r = np.median(x[rng.integers(0, x.size, (NB, x.size))], 1) / np.median(y[rng.integers(0, y.size, (NB, y.size))], 1)
        return {"ratio": float(np.median(x) / np.median(y)), "ci95": [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))]}
    c5["ratios"] = {f"{p}_{N}": ratio_ci(*p.split("/"), N, f"C5r|{p}|{N}") for p in ("MOM/SGD-K", "MOM/FLOW", "SGD-K/FLOW")
                    for N in CM.NS}
    K_min = choice.get("K_min")
    arms = ("SGD-K", "KS-4", "KS-16")
    have = bool(K_min) and all((a, 250000) in cells for a in arms)
    if have:
        Ks = np.array([K_min, 4 * K_min, 16 * K_min], float)
        data = [np.array(vals_of(RQ, cells, a, 250000, "delta_C")) for a in arms]
        mom = np.array(vals_of(RQ, cells, "MOM", 250000, "delta_C"))
        have = min(d.size for d in data + [mom]) > 0 and all(np.all(d > 0) for d in data)
    if have:

        def fit(ds):
            yv = np.log([np.median(d) for d in ds])
            sl, ic = np.polyfit(np.log(Ks), yv, 1)
            return sl, float(np.exp(ic))   # predicted drift at K = 1
        sl, d1 = fit(data)
        rng = CM.key_rng("C5|extrap")
        rs, sls = [], []
        for _ in range(NB):
            bs = [d[rng.integers(0, d.size, d.size)] for d in data]
            s_, p_ = fit(bs)
            mb = mom[rng.integers(0, mom.size, mom.size)]
            rs.append(np.median(mb) / p_)
            sls.append(s_)
        ratio = float(np.median(mom) / d1)
        ci = [float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975))]
        c5["extrapolation_250k"] = {"K": Ks.tolist(), "slope": float(sl), "slope_ci95": [float(np.quantile(sls, 0.025)), float(np.quantile(sls, 0.975))],
                                    "plain_step_drift_pred": d1, "MOM_over_plain": ratio, "MOM_over_plain_ci95": ci,
                                    "momentum_named_main_cause": bool(ratio > 3 and ci[0] > 3)}
    else:
        c5["extrapolation_250k"] = {"skipped": "K_min None, KS cells absent, or an Euler cell without ok runs"}
    C["C5"] = {"tier": "E", **c5}
    c6 = {}
    if ("MOM-BF16", 250000) in cells:
        for key in ("rho_end", "delta_C"):
            a = np.array(vals_of(RQ, cells, "MOM-BF16", 250000, key))
            b = np.array(vals_of(RQ, cells, "MOM", 250000, key))
            if a.size == 0 or b.size == 0:
                c6[key] = {"skipped": "empty cell", "sentence_triggered": False}
                continue
            rng = CM.key_rng(f"C6|{key}")
            d = np.median(a[rng.integers(0, a.size, (NB, a.size))], 1) - np.median(b[rng.integers(0, b.size, (NB, b.size))], 1)
            rel = float((np.median(a) - np.median(b)) / np.median(b))
            ci = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
            c6[key] = {"median_bf16": float(np.median(a)), "median_fp64": float(np.median(b)), "rel_diff": rel,
                       "diff_ci95": ci, "sentence_triggered": bool(abs(rel) > 0.2 and (ci[0] > 0 or ci[1] < 0))}
    C["C6"] = {"tier": "E", **c6}
    return C


# ============================================================================ ladders, decisions, decompositions
def build_ladder_jobs(lroot, cohort, cells, G):
    jobs, pairs = [], {}
    for arm in LADDER_ARMS:
        for N in CM.NS:
            if (arm, N) not in cells:
                continue
            pr = CM.cell_pairs(cells, arm, N)
            pairs[(arm, N)] = pr
            g = G[f"{arm}_{N}"]
            tags = ["y", "u", "h"] + (["h_l2"] if arm in MP.CORE else [])
            for a, b in pr:
                for t in tags:
                    jobs.append((lroot, cohort, t, a, b, None, None))
            jobs += DC.pair_jobs(lroot, cohort, pr, g["Gbar"], g["Gbar_bal"], tags=("uc", "ucpA", "ucpB", "ubal", "ubalc"))
    return jobs, pairs


def dv(vals, eps):
    """CP decision (section 9) plus the p_hat(eps) curve on eps/eps_anchor in [0.80, 1.25] (section 13)."""
    d = CM.decide_vals(vals, eps)
    if d.get("eps") is not None and np.isfinite(d["eps"]):
        d["p_hat_curve"] = CM.phat_curve(vals, d["eps"])
    return d


def decision_tables(res, pairs, E, pool):
    T = {}
    for (arm, N), pr in pairs.items():
        cell = f"{arm}_{N}"
        e = E[cell]
        P = pool[str(N)]
        dl = lambda tag, Q: [res[(tag, a, b)].get(Q, np.nan) for a, b in pr]
        parts = DC.cell_parts(res, pr)
        tab = {}
        for Q in CM.TOL_KEYS:
            row = {}
            yY = dv(dl("y", Q), e["Y"].get(Q))
            row["y@Y"] = yY
            row["y@Yq90"] = dv(dl("y", Q), e["Y_q90"].get(Q))
            row["y@Pool"] = dv(dl("y", Q), P["Pool"].get(Q))
            row["u@Y"] = dv(dl("u", Q), e["Y"].get(Q))
            row["u@Yq90"] = dv(dl("u", Q), e["Y_q90"].get(Q))
            row["u@Pool"] = dv(dl("u", Q), P["Pool"].get(Q))
            row["u@O"] = dv(dl("u", Q), e["O"].get(Q))
            row["h@Y"] = dv(dl("h", Q), e["Y"].get(Q))
            row["h@h"] = dv(dl("h", Q), e["h"].get(Q))
            if "h_l2" in e and ("h_l2", pr[0][0], pr[0][1]) in res:
                row["h_l2@h_l2"] = dv(dl("h_l2", Q), e["h_l2"].get(Q))
            if Q in CM.R_CF:
                row["uc@Y"] = dv(dl("uc", Q), e["Y"].get(Q))
                row["uc@Pool"] = dv(dl("uc", Q), P["Pool"].get(Q))
                row["ubal@Y"] = dv(dl("ubal", Q), e["Y"].get(Q))
                row["uc@C"] = dv(dl("uc", Q), e["C"].get(Q))
                row["u@C"] = dv(dl("u", Q), e["C"].get(Q))
                row["yM@Y"] = dv([p[Q]["y_plus_M"] for p in parts], e["Y"].get(Q))
                row["yI@Y"] = dv([p[Q]["y_plus_I"] for p in parts], e["Y"].get(Q))
            # reference guard
            for k, v in list(row.items()):
                guard = None
                if k in ("u@Y", "u@Yq90", "uc@Y", "ubal@Y", "h@Y", "yM@Y", "yI@Y"):
                    guard = row["y@Yq90" if k == "u@Yq90" else "y@Y"]
                elif k in ("u@Pool", "uc@Pool"):
                    guard = row["y@Pool"]
                elif k == "u@C":
                    guard = row["uc@C"]
                v["final"] = v["decision"] if (guard is None or guard["decision"] == "reliable") else "inconclusive (RG)"
                v["rg_ok"] = guard is None or guard["decision"] == "reliable"
            # anchor-sensitivity flag (u and u^c at Y_a against Pool)
            for base_ in ("u", "uc"):
                if f"{base_}@Y" in row:
                    s = {row[f"{base_}@Y"]["final"], row[f"{base_}@Pool"]["final"]}
                    row[f"{base_}_anchor_sensitive"] = s == {"reliable", "unreliable"}
            tab[Q] = row
        T[cell] = tab
    return T


def sigma_cell(res, pr):
    return {Q: DC.sigma_y(res, pr, Q) for Q in CM.TOL_KEYS}


def a_block(T, res, pairs, parts, sig):
    A = {}
    fin = lambda cell, Q, k: T[cell][Q][k]["final"]
    # A1
    a1 = {}
    for N in CM.NS:
        for Q in PRIMARY_Q:
            r = T[f"WD_{N}"][Q]
            a1[f"{Q}_{N}"] = {"u@Y": r["u@Y"]["final"], "u@Pool": r["u@Pool"]["final"], "rg_ok": r["u@Y"]["rg_ok"],
                              "anchor_sensitive": r["u_anchor_sensitive"], "x": r["u@Y"]["x"], "n": r["u@Y"]["n"]}
    if not all(v["rg_ok"] for v in a1.values()):
        st = "inconclusive (RG)"
    elif any(v["anchor_sensitive"] for v in a1.values()):
        st = "anchor-sensitive"
    elif all(v["u@Y"] == "reliable" for v in a1.values()):
        st = "holds"
    else:
        st = "fails"
    A["A1"] = {"tier": "R; positive control", "cells": a1, "status": st,
               "label_all_u_decisions": "positive control failed" if st == "fails" else None}
    # A2
    a2 = {}
    for arm in ("MOM", "WD", "FLOW", "BAL-FLOW", "SMALL-FLOW"):
        for N in CM.NS:
            cell = f"{arm}_{N}"
            r = T[cell]["dN_k10"]
            X = [p["dN_k10"]["X"] for p in parts[cell]]
            b = CM.boot_mean_ci(X, f"A2|{cell}", n=NB)
            applies = r["y@Y"]["decision"] == "reliable"
            ok = (not applies) or (r["u@Y"]["decision"] != "unreliable" and -0.002 <= b["ci"][0] and b["ci"][1] <= 0.002)
            a2[cell] = {"tier": "R" if arm in ("MOM", "WD") else "H", "y_reliable": applies, "u@Y": r["u@Y"]["decision"],
                        "X_mean": b["mean"], "X_ci95": b["ci"], "pass": ok}
    A["A2"] = {"cells": a2, "pass_R": all(v["pass"] for k, v in a2.items() if v["tier"] == "R"),
               "pass_H": all(v["pass"] for k, v in a2.items() if v["tier"] == "H")}
    # A3
    a3 = {}
    for N in CM.NS:
        r = T[f"MOM_{N}"]["Gs"]
        a3[N] = {"u@Y": r["u@Y"]["final"], "uc@Y": r["uc@Y"]["final"], "u@Pool": r["u@Pool"]["final"],
                 "uc@Pool": r["uc@Pool"]["final"], "anchor_sensitive": r["u_anchor_sensitive"] or r["uc_anchor_sensitive"],
                 "rg_ok": r["u@Y"]["rg_ok"]}
    if not all(v["rg_ok"] for v in a3.values()):
        st3 = "inconclusive (RG)"
    elif any(v["anchor_sensitive"] for v in a3.values()):
        st3 = "anchor-sensitive"
    elif all(v["u@Y"] == "unreliable" and v["uc@Y"] == "unreliable" for v in a3.values()):
        st3 = "holds"
    else:
        st3 = "fails"
    A["A3"] = {"tier": "R; negative control", "cells": a3, "status": st3}
    # A4
    A["A4"] = {"tier": "E", "cells": {cell: DC.summarize_parts(parts[cell], sig[cell], f"A4|{cell}") for cell in parts}}
    m = A["A4"]["cells"].get("MOM_250000", {}).get("Gs")
    if m:
        A["A4"]["replication_MOM_250k"] = {
            "S_over_X": m["S_over_X"], "S_over_X_in_0.5_0.9": 0.5 <= m["S_over_X"] <= 0.9,
            "M_ci_excl_0": m["M"]["ci"][0] > 0 or m["M"]["ci"][1] < 0, "I_ci_excl_0": m["I"]["ci"][0] > 0 or m["I"]["ci"][1] < 0,
            "B_ci_excl_0": m["B"]["ci"][0] > 0 or m["B"]["ci"][1] < 0}
    # A5, A6
    A["A5"] = {"tier": "E", "cells": {cell: {"u@C": T[cell]["Gs"]["u@C"]["final"], "yM@Y": T[cell]["Gs"]["yM@Y"]["final"],
                                             "yM_x": T[cell]["Gs"]["yM@Y"]["x"], "n": T[cell]["Gs"]["yM@Y"]["n"],
                                             "M_sigma": A["A4"]["cells"][cell]["Gs"]["M"]["sigma_units"]["mean"]}
                                      for cell in parts}}
    A["A6"] = {"tier": "E", "cells": {cell: {Q: {"u@O": T[cell][Q]["u@O"]["decision"],
                                                 "epsO_over_epsY": (T[cell][Q]["u@O"]["eps"] / T[cell][Q]["y@Y"]["eps"])
                                                 if T[cell][Q]["y@Y"]["eps"] else float("nan")} for Q in PRIMARY_Q + ("dCKA",)}
                                      for cell in parts}}
    # A7
    a7 = {}
    for arm in ("BAL-FLOW", "SMALL-FLOW"):
        cells_ = {}
        for N in CM.NS:
            cell = f"{arm}_{N}"
            cells_[N] = {Q: {"u@Y": T[cell][Q]["u@Y"]["final"], "u@Pool": T[cell][Q]["u@Pool"]["final"],
                             "anchor_sensitive": T[cell][Q]["u_anchor_sensitive"], "rg_ok": T[cell][Q]["u@Y"]["rg_ok"]}
                         for Q in ("Gs", "dCKA")}
            cells_[N]["B_Gs_sigma"] = A["A4"]["cells"][cell]["Gs"]["B"]["sigma_units"]
        ra = all(cells_[N]["Gs"]["u@Y"] == "reliable" and not cells_[N]["Gs"]["anchor_sensitive"] for N in CM.NS)
        rb = all(cells_[N]["B_Gs_sigma"]["ci"][0] > 0.5 for N in CM.NS)
        a7[arm] = {"cells": cells_, "reading_a": ra, "reading_b": rb}
    wd_b = all(-0.5 <= A["A4"]["cells"][f"WD_{N}"]["Gs"]["B"]["sigma_units"]["ci"][0] and
               A["A4"]["cells"][f"WD_{N}"]["Gs"]["B"]["sigma_units"]["ci"][1] <= 0.5 for N in CM.NS)
    A["A7"] = {"tier": "H-open (decision without a direction)", "arms": a7, "WD_B_Gs_within_0.5sigma": wd_b}
    # A8, A9, A10
    x25 = [p["Gs"]["X"] for p in parts["MOM_25000"] if np.isfinite(p["Gs"]["X"])]
    x250 = [p["Gs"]["X"] for p in parts["MOM_250000"] if np.isfinite(p["Gs"]["X"])]
    rng = CM.key_rng("A8")
    a, b = np.array(x250), np.array(x25)
    if a.size and b.size:
        rr = a[rng.integers(0, a.size, (NB, a.size))].mean(1) / b[rng.integers(0, b.size, (NB, b.size))].mean(1)
        ratio = float(a.mean() / b.mean())
        A["A8"] = {"tier": "R", "ratio": ratio, "ci95": [float(np.quantile(rr, 0.025)), float(np.quantile(rr, 0.975))],
                   "pass": 0.5 <= ratio <= 1.5}
    else:
        A["A8"] = {"tier": "R", "skipped": "no finite MOM pairs", "pass": False}
    a9 = {}
    for N in CM.NS:
        pr = pairs[("MOM", N)]
        f = [res[("u", x, y)]["dE"] < res[("y", x, y)]["dE"] for x, y in pr]
        a9[N] = {"frac": float(np.mean(f)), "n": len(f), "pass": float(np.mean(f)) >= 0.9}
    A["A9"] = {"tier": "R", "cells": a9, "pass": all(v["pass"] for v in a9.values())}
    a10 = {}
    for N in CM.NS:
        mm = {}
        for arm in LADDER_ARMS:
            if (arm, N) in pairs:
                v = [1 - res[("h", x, y)]["dN_k10"] for x, y in pairs[(arm, N)] if np.isfinite(res[("h", x, y)]["dN_k10"])]
                mm[arm] = float(np.mean(v)) if v else float("nan")
        a10[N] = {"mean_sN10_h": mm, "max_abs_diff": float(max(abs(p - q) for p in mm.values() for q in mm.values()))}
    A["A10"] = {"tier": "E", "cells": a10}
    return A


# ============================================================================ I-block
def i_block(RQ, cells, parts, sig):
    I = {}
    i1 = {"n_runs": 0, "n_vacuous": 0, "n_viol": 0}
    for rid, q in RQ.items():
        if q.get("ok") and "N1_ok_end" in q:
            i1["n_runs"] += 1
            i1["n_vacuous"] += int(q["N1_vacuous_end"])
            i1["n_viol"] += int(not q["N1_ok_end"])
    I["I1"] = {"tier": "T; check", **i1, "pass": i1["n_viol"] == 0}
    i2 = {"n_runs": 0, "n_vacuous": 0, "n_viol": 0, "median_D": {}}
    for rid, q in RQ.items():
        if q["arm"] in MP.FLOW_ARMS and q.get("ok") and "N1_ok_flow" in q:
            i2["n_runs"] += 1
            i2["n_vacuous"] += int(q["N1_vacuous_flow"])
            i2["n_viol"] += int(not q["N1_ok_flow"])
    for arm in ("BAL-FLOW", "SMALL-FLOW", "FLOW", "MOM"):
        for N in CM.NS:
            i2["median_D"][f"{arm}_{N}"] = float(np.median(vals_of(RQ, cells, arm, N, "D") or [float("nan")]))
    I["I2"] = {"tier": "T given C1; check", **i2, "pass": i2["n_viol"] == 0}
    pv, det = {}, {}
    for init in ("BAL", "SMALL"):
        for Q in ("Gs", "dCKA"):
            for N in CM.NS:
                k = f"{init}|{Q}|{N}"
                a = [p[Q]["I"] for p in parts[f"{init}-MOM_{N}"]]
                b = [p[Q]["I"] for p in parts[f"{init}-FLOW_{N}"]]
                r = CM.boot_diff_means(a, b, f"I3|{k}", n=NB)
                det[k] = {**r, "tier": "R" if (init == "SMALL" and Q == "Gs" and N == 250000) else "H"}
                pv[k] = r["p_one_sided"]
    hm = CM.holm(pv)
    for k in det:
        det[k]["holm"] = hm[k]
    I["I3"] = {"tier": "primary, family F1 (Holm alpha 0.05, 8 tests, 10,000 resamples)", "tests": det,
               "n_reject": int(sum(v["reject"] for v in hm.values()))}
    I["I4"] = {"tier": "primary; H; two-sided decision map",
               **DC.i4_map({N: parts[f"FLOW_{N}"] for N in CM.NS}, {N: sig[f"FLOW_{N}"] for N in CM.NS}, "FLOW", nboot=NB)}
    i5 = {}
    for Q in ("Gs", "dCKA"):
        for N in CM.NS:
            g = lambda arm: np.array([p[Q]["I"] for p in parts[f"{arm}_{N}"] if np.isfinite(p[Q]["I"])], float)
            arr = {a: g(a) for a in ("FLOW", "BAL-FLOW", "MOM", "BAL-MOM")}
            if min(v.size for v in arr.values()) == 0:
                i5[f"I_{Q}_{N}"] = {"skipped": "an arm without finite pairs"}
                continue
            rng = CM.key_rng(f"I5|{Q}|{N}")
            bm = {a: np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(NB)]) for a, v in arr.items()}
            did = (arr["FLOW"].mean() - arr["BAL-FLOW"].mean()) - (arr["MOM"].mean() - arr["BAL-MOM"].mean())
            bd = (bm["FLOW"] - bm["BAL-FLOW"]) - (bm["MOM"] - bm["BAL-MOM"])
            i5[f"I_{Q}_{N}"] = {"did": float(did), "ci95": [float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975))]}
    for N in CM.NS:
        v = {a: np.mean(vals_of(RQ, cells, a, N, "rho_end")) for a in ("FLOW", "BAL-FLOW", "MOM", "BAL-MOM")}
        i5[f"rho_end_{N}"] = {"did": float((v["FLOW"] - v["BAL-FLOW"]) - (v["MOM"] - v["BAL-MOM"]))}
    I["I5"] = {"tier": "E", **i5}
    from scipy.stats import spearmanr
    i6 = {}
    for arm, cname in (("FLOW", "C0_op"), ("MOM", "C_end_op")):
        for N in CM.NS:
            rids = [rid for rep, rid in sorted(cells[(arm, N)].items()) if RQ[rid].get("ok") and "D" in RQ[rid]]
            x = np.array([RQ[r][cname] / RQ[r]["s"] for r in rids])
            y = np.log([RQ[r]["D"] for r in rids])
            rho, p = spearmanr(y, x)
            p1 = p / 2 if rho > 0 else 1 - p / 2
            i6[f"{arm}_{N}"] = {"spearman": float(rho), "p_one_sided": float(p1), "n": len(rids), "x": f"{cname}/s"}
    I["I6"] = {"tier": "E (unadjusted)", **i6}
    return I


# ============================================================================ K-block
def k_block(cohort, pairs, n_jobs, out_dir):
    jobs = [(cohort, f"{arm}_{N}", j, a, b) for (arm, N), pr in pairs.items() for j, (a, b) in enumerate(pr)]
    rows = K3.run_pairs(cohort, jobs, n_jobs, "main")
    CM.jdump(rows, os.path.join(out_dir, "kblock_pairs.json"))
    cells = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(r)
    S = {c: K3.summarize_cell(R) for c, R in cells.items()}
    K = {"cells": S}
    K["K1"] = {"tier": "T; check", "violations_pure": sum(S[c][f"pure_k{k}"]["violations_total"] for c in S for k in K3.KS),
               "violations_imb": sum(S[c][f"imb_k{k}"]["violations_total"] for c in S for k in K3.KS)}
    K["K1"]["pass"] = K["K1"]["violations_pure"] == 0 and K["K1"]["violations_imb"] == 0
    K["K2"] = {"tier": "T; check", "anchor_violations": sum(S[c][f"{e}_k{k}"]["anchor_bound_violations_total"]
                                                          for c in S for k in K3.KS for e in ("pure", "imb")),
               "pairs_bound_lt_obs": sum(S[c][f"{e}_k{k}"]["n_dir"] - S[c][f"{e}_k{k}"]["pairs_bound_ge_obs"]
                                         for c in S for k in K3.KS for e in ("pure", "imb"))}
    K["K2"]["pass"] = K["K2"]["anchor_violations"] == 0 and K["K2"]["pairs_bound_lt_obs"] == 0
    K["K3"] = {"tier": "T; check", "violations": sum(S[c].get("K3_violations", 0) for c in S),
               "median_kappa_over_kappa_bal": {c: S[c].get("kappa_over_kappa_bal_median") for c in S}}
    K["K3"]["pass"] = K["K3"]["violations"] == 0
    K["K4a"] = {"tier": "E (post hoc heuristic, not in the paper)",
                "cells": {c: {"band_n": S[c]["pure_k10"]["k4a_band_n"], "band_kept": S[c]["pure_k10"]["k4a_band_kept"],
                              "kept_frac": S[c]["pure_k10"]["k4a_band_kept"] / max(1, S[c]["pure_k10"]["k4a_band_n"])} for c in S},
                "v10_reference": "changed 11/14,925, 5/3,348, 2/424 (cor3_verify)"}
    K["K4b"] = {"tier": "R", "cells": {c: S[c]["pure_k10"]["bound_over_obs_median"] for c in S}}
    K["K4b"]["pass"] = all(v is not None and 5 <= v <= 25 for v in K["K4b"]["cells"].values())
    K["K4c"] = {"tier": "R", "cells": {c: S[c]["pure_k10"]["cert_frac_mean"] for c in S}}
    K["K4c"]["pass"] = all(v <= 0.10 for v in K["K4c"]["cells"].values())
    K["K5"] = {"tier": "H (MOM); BAL-FLOW, SMALL-FLOW reported", "cells": {c: {"obs_dN_k10_mean": S[c]["imb_k10"]["obs_dN_mean"],
                                                                            "ci95": S[c]["imb_k10"]["obs_dN_ci95"]} for c in S}}
    K["K5"]["pass_MOM"] = all(S[f"MOM_{N}"]["imb_k10"]["obs_dN_mean"] >= 0.01 for N in CM.NS)
    K["K6"] = {"tier": "R (MOM, WD, SMALL-MOM); H (others)", "cells": {c: S[c]["cross_k10_excess_mean"] for c in S}}
    K["K6"]["pass_R"] = all((S[c]["cross_k10_excess_mean"] is not None and abs(S[c]["cross_k10_excess_mean"]) <= 0.0005)
                             for c in S if c.split("_")[0] in ("MOM", "WD", "SMALL-MOM"))
    K["K6"]["pass_H"] = all((S[c]["cross_k10_excess_mean"] is not None and abs(S[c]["cross_k10_excess_mean"]) <= 0.0005)
                             for c in S if c.split("_")[0] not in ("MOM", "WD", "SMALL-MOM"))
    return K


# ============================================================================ consequences (section 13)
def consequences(S):
    L = []
    C, I, A, K = S["C"], S["I"], S["A"], S["K"]
    if C["C1"]["pass"] and C["C3"]["a_pass"] and C["C3"]["b_pass"] and C["C4a"]["pass"]:
        L.append("C1, C3, C4a hold: Remark rem:optimizers gains the prospective network test (minibatch sampling alone "
                 "does not change C; discretization and momentum do; from C0 = 0 momentum raised rho to "
                 f"{C['C3']['a_BALMOM_rho']}; the final imbalance no longer reflects its initial value).")
    if not C["C4a"]["pass"]:
        L.append("C4a fails: the sentence 'under momentum, the imbalance at the end of training no longer reflects its "
                 "initial value' is not made.")
    if not C["C3"]["a_pass"]:
        L.append("C3(a) fails: the Section 7 momentum sentence is restricted to default and small inits.")
    L.append(f"C4b: {C['C4b']['reading']}" + (" (replaced by the C4a wording)" if "loses" in C["C4b"]["reading"] and C["C4a"]["pass"]
                                              else (" (removed)" if "loses" in C["C4b"]["reading"] else "")))
    ex = C["C5"].get("extrapolation_250k", {})
    if ex.get("momentum_named_main_cause"):
        L.append("C5: momentum may be named the main cause of the imbalance in this network.")
    else:
        L.append("C5: both causes (discretization and momentum) stay in the Remark.")
    for key, v in C.get("C6", {}).items():
        if isinstance(v, dict) and v.get("sentence_triggered"):
            L.append(f"C6 ({key}): the Section 7 'ninefold' sentence states that v1.0's head ran in bf16 and gives the fp64 value.")
    if I["I3"]["n_reject"] > 0:
        L.append(f"I3: {I['I3']['n_reject']}/8 Holm rejections; for the rejected cells Section 6.3/7 may say 'from a balanced "
                 "or small start, momentum creates the imbalance channel; under head flow it stays absent by construction'.")
    o = I["I4"]["outcome"]
    if o.startswith("(i)"):
        L.append("I4 (i): 'with C held at its random initial value, the head's imbalance still lowers geometric agreement "
                 "between runs ... the network realizes the mechanism of part 2, at a reduced size'"
                 + ("; the size lies within the 0.5 sigma_y margin, stated as such." if "within" in o else "."))
    elif o.startswith("(ii)"):
        L.append("I4 (ii): 'even with C conserved, the initial imbalance did not measurably lower agreement ...; the failure "
                 "of part 2 in this network is not only a consequence of momentum'; the v3 section 6.3 attribution changes.")
    else:
        L.append("I4 (iii): estimates only.")
    a1 = A["A1"]["status"]
    L.append({"holds": "A1 holds: WD's u agrees at both sizes under same-N tolerances (removes the '25k undecided' caveat).",
              "fails": "A1 fails: no categorical claim about u; every u decision is labelled 'positive control failed'."}.get(
        a1, f"A1 {a1}: no categorical claim from it."))
    if A["A3"]["status"] == "holds":
        L.append("A3 holds: the knife-edge text in Section 7 is replaced by the same-N decision.")
    else:
        L.append(f"A3 {A['A3']['status']}: the knife-edge text is replaced by the same-N result as reported (no categorical claim if anchor-sensitive).")
    m = A["A5"]["cells"].get("MOM_250000", {})
    L.append(f"A5: the v3 run-dependent sentence is replaced by the v1.1 same-N decision for MOM 250k G_s: Delta(y)+M at "
             f"eps^Y = {m.get('yM@Y')} ({m.get('yM_x')}/{m.get('n')}); M = {m.get('M_sigma')} sigma_y"
             + ("; its categorical status depends on the tolerance." if m.get("yM@Y", "").startswith("inconclusive") else "."))
    for arm, v in A["A7"]["arms"].items():
        if v["reading_a"]:
            L.append(f"A7 (a) {arm}: 'under head flow, a balanced head, or a small init, makes u reproducible on scale-normalized "
                     "geometry in this network'" + (" (the Table 2 row itself)" if arm == "SMALL-FLOW" else ""))
        if v["reading_b"]:
            L.append(f"A7 (b) {arm}: 'with C ~ 0 an excess remains, carried by the balanced map (WP is not a projector), outside "
                     "Theorem 4'" + ("; weight decay's agreement also involves its effect on WP" if A["A7"]["WD_B_Gs_within_0.5sigma"] else ""))
    if K["K4b"]["pass"] and K["K4c"]["pass"]:
        L.append("K4(b, c) hold: may be called prospective; K1-K3 remain code checks.")
    L.append("Lemma N1: entering v3 is the author's decision.")
    return L


def write_report(S, path):
    L = [f"# S2.2 v1.1 report{' (DRY RUN on EXCLUDED unit-test runs: NOT EVIDENCE)' if S['dry'] else ''}", "",
         f"cohort {S['cohort']}, eps sha256 {S['eps_sha256']}, code manifest {S.get('manifest_sha256')}", "",
         "## Checks", ""]
    for k in ("C1", "C2"):
        L.append(f"- {k}: {'PASS' if S['C'][k]['pass'] else 'FAIL'}")
    L.append(f"- C4 flow check (median mu_C >= 0.99): {'PASS' if S['C']['C4a']['check_pass'] else 'FAIL'}")
    for k in ("I1", "I2"):
        L.append(f"- {k}: {'PASS' if S['I'][k]['pass'] else 'FAIL'} (violations {S['I'][k]['n_viol']}, vacuous {S['I'][k]['n_vacuous']})")
    for k in ("K1", "K2", "K3"):
        L.append(f"- {k}: {'PASS' if S['K'][k]['pass'] else 'FAIL'}")
    L.append(f"- numerical failures (R-NUM; counted as failed rungs): {S.get('numerical_failures')}")
    L += ["", "## Primary", "",
          f"- C3(a) BAL-MOM: {S['C']['C3']['a_BALMOM_rho']}", f"- C3(b) MOM growth: {S['C']['C3']['b_MOM_growth']}",
          f"- C4a MOM mu_C: {S['C']['C4a']['MOM']}",
          f"- I3: {S['I']['I3']['n_reject']}/8 rejected (Holm)", f"- I4: {S['I']['I4']['outcome']}",
          f"- A1: {S['A']['A1']['status']}", f"- A3: {S['A']['A3']['status']}",
          f"- A7: " + "; ".join(f"{a}: (a) {v['reading_a']}, (b) {v['reading_b']}" for a, v in S["A"]["A7"]["arms"].items()),
          "", "## Pre-agreed consequences", ""] + [f"- {c}" for c in S["consequences"]]
    L += ["", "## Decision tables (primary rungs; final = after reference guard)", "",
          "| cell | Q | y@Y | u@Y | u@Pool | u@C | uc@Y | ubal@Y | yM@Y | yI@Y |", "|---|---|---|---|---|---|---|---|---|---|"]
    for cell, tab in S["tables"].items():
        for Q in ("dN_k10", "Gs", "dCKA"):
            r = tab[Q]
            f = lambda k: f"{r[k]['final']} {r[k]['x']}/{r[k]['n']}" if k in r else "-"
            L.append(f"| {cell} | {Q} | {f('y@Y')} | {f('u@Y')} | {f('u@Pool')} | {f('u@C')} | {f('uc@Y')} | {f('ubal@Y')} | {f('yM@Y')} | {f('yI@Y')} |")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="S2_2v11_main")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    dry = a.cohort.startswith("UT_")
    base = os.path.join(CM.RES, "dryrun") if dry else CM.RES
    eps_path = os.path.join(base, "eps", "eps_S2_2_v11.json")
    manifest_sha = None
    if not dry:
        import guard_v11
        ok, msg = guard_v11.verify_frozen()
        if not ok:
            raise SystemExit(f"REFUSED: {msg}")
        manifest_sha = CM.sha(guard_v11.MANIFEST)
        es = CM.sha(eps_path)
        if not any("[s22v11-gates] eps_S2_2_v11 FROZEN" in ln and es in ln for ln in open(STATUS).read().splitlines()):
            raise SystemExit("REFUSED: the eps file sha256 is not quoted by an [s22v11-gates] FROZEN STATUS line")
    out_json = os.path.join(base, "s22v11_summary.json")
    if os.path.exists(out_json) and not dry:
        raise SystemExit("summary exists: the final analysis runs once")
    complete, counts = CM.cohort_complete(a.cohort)
    if not complete:
        raise SystemExit(f"main cohort incomplete {counts}: no INTERIM analysis")
    t0 = time.time()
    epsf = json.load(open(eps_path))
    choice = json.load(open(os.path.join(CM.PLANS, "eng_choice.json"))) if not dry else MP.UT_CHOICE
    cells = CM.cohort_cells(a.cohort)
    RQ = run_quantities(a.cohort, cells)
    print(f"per-run quantities: {len(RQ)} runs ({time.time() - t0:.0f} s)", flush=True)
    lroot = os.path.join(base, "ladders")
    jobs, pairs = build_ladder_jobs(lroot, a.cohort, cells, epsf["gbars"])
    res = CM.run_ladders(jobs, a.jobs, "main")
    parts = {f"{arm}_{N}": DC.cell_parts(res, pr) for (arm, N), pr in pairs.items()}
    sig = {f"{arm}_{N}": sigma_cell(res, pr) for (arm, N), pr in pairs.items()}
    T = decision_tables(res, pairs, epsf["eps"], epsf["pool"])
    S = {"dry": dry, "cohort": a.cohort, "counts": counts, "eps_sha256": CM.sha(eps_path), "manifest_sha256": manifest_sha,
         "eng_choice": choice, "sigma_y": sig}
    # resolution R-NUM: numerically degenerate heads (counted, never dropped)
    nf = {}
    for (tag, ra, rb), d in res.items():
        if d.get("_numfail"):
            k = next(f"{arm}_{N}" for (arm, N), pr in pairs.items() if (ra, rb) in pr)
            nf.setdefault(k, {}).setdefault(tag, 0)
            nf[k][tag] += 1
    S["numerical_failures"] = {"ladder_pairs_by_cell_tag": nf,
                               "runs_gauge_not_spd": sorted(r for r, q in RQ.items() if q.get("gauge_numerical_failure"))}
    S["C"] = c_block(RQ, cells, choice)
    S["I"] = i_block(RQ, cells, parts, sig)
    S["A"] = a_block(T, res, pairs, parts, sig)
    if S["A"]["A1"]["status"] == "fails":
        for cell in T:
            for Q in T[cell]:
                for k, v in T[cell][Q].items():
                    if k.startswith("u@") and isinstance(v, dict):
                        v["label"] = "positive control failed"
    S["K"] = k_block(a.cohort, pairs, a.jobs, base)
    S["tables"] = T
    S["consequences"] = consequences(S)
    S["per_run"] = {rid: {k: v for k, v in q.items() if k not in ("Mf", "Mi")} for rid, q in RQ.items()}
    S["values"] = {f"{tag}|{a_}|{b_}": d for (tag, a_, b_), d in res.items()}
    S["wall_s"] = time.time() - t0
    CM.jdump(S, out_json)
    write_report(S, os.path.join(base, "S2_2_v11_report.md"))
    print(f"wrote {out_json} sha256 {CM.sha(out_json)} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
