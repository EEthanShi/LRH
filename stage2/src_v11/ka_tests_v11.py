"""S2.2 v1.1 known-answer tests KA0-KA5 on the v1.0 S2.2 main cohort (READ-ONLY, seen data; PREREG_S2_2_v1.1.md
section 14, step 3). Nothing in v1.0 is written; outputs go to results/S2_2_v11/ka/.

    OMP_NUM_THREADS=1 python src_v11/ka_tests_v11.py [all|ka0|ka1|ka2|ka3|ka4|ka5] [--jobs 6]

KA0 the reduced ladder (R_cf rungs only) equals the full hrhv2 ladder bit for bit (4 v1.0 pairs x {y, u}).
KA1 common_v11 / decomp_v11 reproduce the v1.0 review a2b numbers (noWD 250k, 96 pairs): mean Delta(y), Delta(u),
    pair-own common part S_pair and mismatch M_pair (G_s 0.0148 / 0.0051), and the cell-mean common gauge (Gbar from
    the cell's 192 runs, as a2 did) on G_s, 1-CKA, 1-s_N(10); |difference| <= 1e-6. Delta(y), Delta(u) per pair
    also equal the frozen v1.0 ladder values (results/S2_2/s22_summary.json) to 1e-12.
KA2 cor3_v11 reproduces v1.0 cor3 on all 672 pairs: kappa per pair (relative 1e-10); certified-anchor totals
    {k=1: 1,074,449; 10: 29,208; 30: 1,220} with 0 violations (pure exchange, both directions); graded bound and
    observed slot change per cell (direction A) equal v1.0 graded.json (1e-12). Identities that must also hold on
    v1.0: K1 for the imbalance-only exchange (0 violations), K2 (0 anchor violations), K3 (0 violations).
KA3 Lemma N1 (gaugemath_v11.run_gauge) holds on all 1,344 v1.0 runs (vacuous cases counted).
KA4 the decision / curve code reproduces v1.0: every (cell, rep, rung) entry of the v1.0 S2.2 decision tables
    (n, x, p_hat, CP bounds, decision) and p_hat at q90, plus the review's p_hat(mult x eps) and eps for x >= 92.
KA5 (a) plants (plants_v11) on v1.0 WD 250k (48 pairs): planted order of S and M, I = 0 for the balanced plant;
    (b) the I4 map on synthetic per-pair parts with known answers ((i), (ii), (iii), (i) within the margin);
    (c) the I4 map on planted heads at both N (balanced+run0.30 -> realized);
    (d) the I4 map on the real v1.0 WD and noWD cells (96 pairs, both N; Gbar from the cell's own runs): the draft's
        stated expectations are (ii) for WD and (i) for noWD; reported with a match flag.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_v11 as CM  # noqa: E402
import cor3_v11 as K3  # noqa: E402
import crit as C  # noqa: E402
import decomp_v11 as DC  # noqa: E402
import gaugemath_v11 as GM  # noqa: E402
import plants_v11 as PL  # noqa: E402

ROOT = CM.ROOT
V10 = "V10:S2_2_main"
OUT = os.path.join(CM.RES, "ka")
LROOT = os.path.join(OUT, "ladders")
RF = os.path.join(ROOT, "results", "review_final")
RESULTS = {}
JOBS = 6


def rec(name, ok, **kw):
    RESULTS[name] = {"pass": bool(ok), **kw}
    CM.jdump({"label": "KA test on v1.0 S2.2 main runs (READ-ONLY, seen data)", "test": name, **RESULTS[name]},
             os.path.join(OUT, f"{name}.json"))
    print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(kw, default=float)[:600]}", flush=True)


def v10_cells():
    """{cell: [(pid, ra, rb)]} with the v1.0 pairing ((2j, 2j+1) within cell; within corpus for noWD-shared)."""
    plan = json.load(open(os.path.join(ROOT, "plans", "S2_2_main.json")))["runs"]
    d = {}
    for r in plan:
        d.setdefault(f"{r['arm']}_{r['N']}", {})[(r.get("corpus"), r["rep"])] = r["run_id"]
    out = {}
    for name, dd in d.items():
        rows = []
        for c in sorted({c for c, _ in dd}, key=lambda x: -1 if x is None else x):
            reps = sorted(rr for cc, rr in dd if cc == c)
            for j in range(len(reps) // 2):
                rows.append((j if c is None else f"c{c}_{j}", dd[(c, 2 * j)], dd[(c, 2 * j + 1)]))
        out[name] = rows
    return out


def cell_gbars(rids):
    runs = [CM.load_run(V10, r) for r in rids]
    Gbar, Gbal, n = DC.gbars(runs)
    return Gbar, Gbal, n


# ============================================================================ KA0
def _ka0_job(args):
    ra, rb, tag = args
    A, B = CM.load_run(V10, ra), CM.load_run(V10, rb)
    XA, XB = CM.pair_features(A, B, tag)
    GA, GB = C.geom(XA, "none"), C.geom(XB, "none")
    full = C.deltas(C.pair_ladder(GA, GB, n_jobs=1))
    red = C.deltas(CM.reduced_ladder(GA, GB))
    return ra, rb, tag, {k: [full[k], red[k]] for k in CM.R_CF}


def ka0():
    pairs = [("S2.2_noWD_N250000_r000", "S2.2_noWD_N250000_r001"), ("S2.2_WD_N25000_r000", "S2.2_WD_N25000_r001"),
             ("S2.2_smallinit_N250000_r002", "S2.2_smallinit_N250000_r003"),
             ("S2.2_noWD_N25000_r004", "S2.2_noWD_N25000_r005")]
    jobs = [(a, b, t) for a, b in pairs for t in ("y", "u")]
    with get_context("fork").Pool(min(JOBS, len(jobs))) as pool:
        res = pool.map(_ka0_job, jobs)
    bad = [(a, b, t, k, v) for a, b, t, d in res for k, v in d.items() if not (v[0] == v[1])]
    rec("KA0_reduced_equals_full", not bad, n_comparisons=sum(len(d) for *_, d in res), mismatches=bad[:10])


# ============================================================================ KA1
def ka1():
    cells = v10_cells()
    pr = [(a, b) for _, a, b in cells["noWD_250000"]]
    rids = [r for p in pr for r in p]
    Gbar, Gbal, n = cell_gbars(rids)
    jobs = [(LROOT, V10, t, a, b, Gbar.tolist() if t == "uc" else None, None)
            for a, b in pr for t in ("y", "u", "ucpA", "ucpB", "uc")]
    res = CM.run_ladders(jobs, JOBS, "KA1 noWD 250k")
    a2b = json.load(open(os.path.join(RF, "s22-skeptic", "a2b_summary.json")))["noWD_250000"]
    S10 = json.load(open(os.path.join(ROOT, "results", "S2_2", "s22_summary.json")))["values"]
    out, worst = {}, 0.0
    for Q, Qa in (("Gs", "Gs"), ("dCKA", "dCKA"), ("dN_k10", "dN10")):
        v = lambda t: np.array([res[(t, a, b)][Q] for a, b in pr])
        y, u, cA, cB, uc = v("y"), v("u"), v("ucpA"), v("ucpB"), v("uc")
        mine = {"yy": y.mean(), "uu": u.mean(), "common_A": cA.mean(), "common_B": cB.mean(), "common_mean": uc.mean(),
                "excess_total": (u - y).mean(), "excess_common_pairgauge": (0.5 * (cA + cB) - y).mean(),
                "excess_mismatch": (u - 0.5 * (cA + cB)).mean()}
        ref = a2b[Qa]
        dev = {k: abs(float(mine[k]) - ref[k]) for k in mine}
        worst = max(worst, max(dev.values()))
        out[Q] = {"mine": {k: float(x) for k, x in mine.items()}, "ref_a2b": {k: ref[k] for k in mine}, "absdev": dev,
                  "S_cellmean": float(mine["common_mean"] - mine["yy"]), "M_cellmean": float(mine["uu"] - mine["common_mean"])}
    # per-pair Delta(y), Delta(u) against the frozen v1.0 ladder values (pairs in the same order as v1.0 j)
    lad = 0.0
    for tag in ("y", "u"):
        for Q in ("Gs", "dCKA", "dN_k10", "dE", "dO"):
            mine = np.array([res[(tag, a, b)][Q] for a, b in pr])
            ref = np.array(S10[f"noWD_250000|{tag}"][Q])
            lad = max(lad, float(np.max(np.abs(mine - ref))))
    # the cached ladder rows were written by the same _job code path; recompute 6 of them now and compare bit for bit
    spot_root = os.path.join(OUT, "ladders_spotcheck")
    import shutil
    shutil.rmtree(spot_root, ignore_errors=True)   # always recomputed
    spot = [(spot_root, V10, t, a, b, Gbar.tolist() if t == "uc" else None, None) for a, b in pr[:3] for t in ("u", "uc")]
    with get_context("fork").Pool(min(JOBS, len(spot))) as pool:
        sres = pool.map(CM._job, spot)
    cache_dev = max(abs(r["deltas"][Q] - res[(r["tag"], r["A"], r["B"])][Q]) for r in sres for Q in CM.R_CF)
    rec("KA1_decomp_reproduces_a2b", worst <= 1e-6 and lad <= 1e-12 and cache_dev == 0, cache_spotcheck_max_absdev=cache_dev, n_pairs=len(pr), n_runs_Gbar=n,
        max_absdev_vs_a2b=worst, max_absdev_vs_v10_ladder=lad,
        Gs_S_pair=out["Gs"]["mine"]["excess_common_pairgauge"], Gs_M_pair=out["Gs"]["mine"]["excess_mismatch"],
        Gs_S_cellmean=out["Gs"]["S_cellmean"], Gs_M_cellmean=out["Gs"]["M_cellmean"], detail=out)


# ============================================================================ KA2
def ka2():
    cells = v10_cells()
    jobs = [(V10, cell, pid, a, b) for cell, rows in cells.items() for pid, a, b in rows]
    t = time.time()
    rows = K3.run_pairs(V10, jobs, JOBS, "KA2 v1.0")
    CM.jdump(rows, os.path.join(OUT, "ka2_pairs.json"))
    Aref = {(r["cell"], str(r["pair"])): r for r in json.load(open(os.path.join(RF, "cor3_posthoc", "compute", "A_pairs.json")))}
    gref = json.load(open(os.path.join(RF, "cor3_posthoc", "compute", "graded.json")))["A_pure_gauge"]
    kap_dev = max(abs(r["kappa"] - Aref[(r["cell"], str(r["pair"]))]["kappa"]) / Aref[(r["cell"], str(r["pair"]))]["kappa"]
                  for r in rows)
    cert = {k: int(sum(r[f"pureA_k{k}"]["n_cert"] + r[f"pureB_k{k}"]["n_cert"] for r in rows)) for k in K3.KS}
    viol = {k: int(sum(r[f"pureA_k{k}"]["violations"] + r[f"pureB_k{k}"]["violations"] for r in rows)) for k in K3.KS}
    target = {1: 1074449, 10: 29208, 30: 1220}
    gdev = 0.0
    gtab = {}
    for cell in cells:
        R = [r for r in rows if r["cell"] == cell]
        for k in K3.KS:
            b = float(np.mean([r[f"pureA_k{k}"]["bound_dN"] for r in R]))
            o = float(np.mean([r[f"pureA_k{k}"]["obs_dN"] for r in R]))
            av = int(sum(r[f"pureA_k{k}"]["anchor_bound_violations"] for r in R))
            ref = gref[cell][f"k{k}"]
            gdev = max(gdev, abs(b - ref["bound_dN_mean"]), abs(o - ref["obs_dN_mean"]))
            gdev = max(gdev, float(av != ref["anchor_violations_total"]))
            gtab[f"{cell}|k{k}"] = {"bound": b, "obs": o, "ref_bound": ref["bound_dN_mean"], "ref_obs": ref["obs_dN_mean"]}
    S = {c: K3.summarize_cell([r for r in rows if r["cell"] == c]) for c in cells}
    imb_viol = int(sum(S[c][f"imb_k{k}"]["violations_total"] for c in S for k in K3.KS))
    k2_viol = int(sum(S[c][f"{e}_k{k}"]["anchor_bound_violations_total"] for c in S for k in K3.KS for e in ("pure", "imb")))
    k3_viol = int(sum(S[c]["K3_violations"] for c in S))
    ok = (kap_dev <= 1e-10 and cert == target and all(v == 0 for v in viol.values()) and gdev <= 1e-12
          and imb_viol == 0 and k2_viol == 0 and k3_viol == 0)
    rec("KA2_cor3_reproduces_v10", ok, n_pairs=len(rows), kappa_max_reldev=kap_dev, certified_by_k=cert,
        target=target, violations_pure_by_k=viol, graded_max_absdev=gdev, imbalance_only_violations=imb_viol,
        K2_anchor_violations=k2_viol, K3_violations=k3_viol, wall_s=time.time() - t,
        imb_k10_obs_dN_mean={c: S[c]["imb_k10"]["obs_dN_mean"] for c in S},
        D_median={c: S[c]["D_median"] for c in S}, kappa_median={c: S[c]["kappa_median"] for c in S},
        kappa_bal_median={c: S[c]["kappa_bal_median"] for c in S}, graded=gtab)


# ============================================================================ KA3
def ka3():
    plan = json.load(open(os.path.join(ROOT, "plans", "S2_2_main.json")))["runs"]
    n = viol = vac = 0
    cs = {}
    for r in plan:
        z = np.load(os.path.join(ROOT, "runs", "S2_2_main", r["run_id"], "feats.npz"))
        g = GM.run_gauge(z["P"].astype(np.float64), z["W"].astype(np.float64))
        n += 1
        if "N1_ok_end" not in g:   # not numerically SPD: counted as a violation of the test
            viol += 1
            continue
        vac += int(g["N1_vacuous_end"])
        viol += int(not g["N1_ok_end"])
        cs.setdefault(f"{r['arm']}_{r['N']}", []).append(g["N1_arg_end"])
    rec("KA3_lemmaN1_v10", viol == 0 and n == 1344, n_runs=n, violations=viol, vacuous=vac,
        median_c_over_s={c: float(np.median(v)) for c, v in cs.items()})


# ============================================================================ KA4
def ka4():
    S = json.load(open(os.path.join(ROOT, "results", "S2_2", "s22_summary.json")))
    n_cmp, bad = 0, []
    for key, tab in S["tables"].items():
        cell, tag = key.split("|")
        for Q, ent in tab.items():
            if not isinstance(ent, dict) or "x" not in ent:
                continue
            vals = S["values"][key][Q]
            d = CM.decide_vals(vals, S["eps"][tag][Q])
            n_cmp += 1
            for f in ("n", "x", "decision"):
                if d[f] != ent[f]:
                    bad.append((key, Q, f, d[f], ent[f]))
            for f in ("p_hat", "cp_lo", "cp_hi"):
                if not (abs(d[f] - ent[f]) <= 1e-12 or (np.isnan(d[f]) and np.isnan(ent[f]))):
                    bad.append((key, Q, f, d[f], ent[f]))
            q = S["p_hat_q90"][key].get(Q)
            if q is not None:
                dq = CM.decide_vals(vals, S["eps_q90"][tag][Q])
                if abs(dq["p_hat"] - q) > 1e-12:
                    bad.append((key, Q, "p_hat_q90", dq["p_hat"], q))
    R = json.load(open(os.path.join(RF, "s22-recompute", "recompute_summary.json")))["curves"]
    P = json.load(open(os.path.join(RF, "s22-recompute", "recompute_pairs.json")))
    n_curve = 0
    for key, c in R.items():
        if key.startswith("pilot") or "p_at_mult" not in c:
            continue
        s, tag, Q = key.split("|")
        v = [P["pairs"][f"{a}__{b}"]["indep"][tag][Q] for a, b in P["sets"][s]]
        for m, ref in c["p_at_mult"].items():
            mine = CM.phat_curve(v, c["eps"], lo=float(m), hi=float(m))[0][1]
            n_curve += 1
            if abs(mine - ref) > 1e-12:
                bad.append((key, "p_at_mult", m, mine, ref))
        d = CM.decide_vals(v, c["eps"])
        if abs(d["eps_star_r"] - c["eps_needed_rel_92"]) > 0:
            bad.append((key, "eps_star_r", d["eps_star_r"], c["eps_needed_rel_92"]))
    rec("KA4_decisions_curves_reproduce_v10", not bad, n_table_entries=n_cmp, n_curve_points=n_curve, mismatches=bad[:20])


# ============================================================================ KA5
def i4_synthetic():
    """The I4 map on synthetic per-pair parts with known answers (sigma_y = 1)."""
    rng = CM.key_rng("S2.2v11|KA5|I4synthetic")
    cases = {"(ii)": (0.0, 0.1, 0.0, 0.1), "(i)": (1.0, 0.2, 0.8, 0.2), "(iii)": (0.4, 0.5, 0.0, 0.5),
             "(i) within": (0.2, 0.05, 0.2, 0.05)}
    out = {}
    for name, (mI, sI, mM, sM) in cases.items():
        pbn, sig = {}, {}
        for N in (25000, 250000):
            pl = []
            for _ in range(96):
                d = {Q: {"I": float(rng.normal(mI, sI)), "I_M": float(rng.normal(mM, sM))} for Q in ("Gs", "dCKA")}
                pl.append(d)
            pbn[N] = pl
            sig[N] = {"Gs": 1.0, "dCKA": 1.0}
        o = DC.i4_map(pbn, sig, f"KA5syn|{name}")["outcome"]
        exp = {"(ii)": "(ii)", "(i)": "(i) realized", "(iii)": "(iii)", "(i) within": "(i) realized, size within"}[name]
        out[name] = {"outcome": o, "expected_prefix": exp, "ok": o.startswith(exp) and (name != "(i)" or "within" not in o)}
    return out


def ka5():
    cells = v10_cells()
    t0 = time.time()
    # reference common gauge: v1.0 noWD cell-mean metric at each N
    Gref = {N: cell_gbars([r for _, a, b in cells[f"noWD_{N}"] for r in (a, b)])[0] for N in CM.NS}
    jobs, meta = [], {}
    for N in CM.NS:
        pr = [(a, b) for _, a, b in cells[f"WD_{N}"][:48]]
        rr = [r for p in pr for r in p]
        runs = [(CM.load_run(V10, r), r) for r in rr]
        # Gbar_bal of the whole v1.0 cell (192 runs), the same value part (d) uses, so the shared ladder cache is consistent
        Gbal0 = cell_gbars([r for _, a, b in cells[f"WD_{N}"] for r in (a, b)])[1]
        jobs += [(LROOT, V10, t, a, b, None, Gbal0.tolist() if t == "ubalc" else None) for a, b in pr
                 for t in ("y", "ubal", "ubalc")]
        plants = ("common", "common+run0.15", "common+run0.30", "balanced", "balanced+run0.30") if N == 250000 \
            else ("balanced", "balanced+run0.30")
        for p in plants:
            Gb, Gbb = PL.gbar_of_plant(runs, p, Gref[N])
            meta[(p, N)] = pr
            for a, b in pr:
                for t in ("u", "uc"):
                    jobs.append(("PLANT", os.path.join(OUT, "plants"), V10, p, t, a, b, Gref[N].tolist(),
                                 Gb.tolist() if t == "uc" else None, None))
    base_jobs = [j for j in jobs if j[0] != "PLANT"]
    pjobs = [j[1:] for j in jobs if j[0] == "PLANT"]
    base = CM.run_ladders(base_jobs, JOBS, "KA5 base")
    print(f"  plant ladders: {len(pjobs)} ({time.time() - t0:.0f} s)", flush=True)
    with get_context("fork").Pool(JOBS) as pool:
        pres = list(pool.imap_unordered(PL.plant_job, pjobs, chunksize=2))
    P = {(r["plant"], r["tag"], r["A"], r["B"]): r["deltas"] for r in pres}

    def parts_for(plant, N):
        out = []
        for a, b in meta[(plant, N)]:
            d = {"y": base[("y", a, b)], "ubal": base[("ubal", a, b)], "ubalc": base[("ubalc", a, b)],
                 "u": P[(plant, "u", a, b)], "uc": P[(plant, "uc", a, b)]}
            out.append(DC.decompose(d))
        return out
    sig = {N: {Q: DC.sigma_y(base, meta[("balanced", N)], Q) for Q in CM.R_CF} for N in CM.NS}
    summ = {}
    for (p, N) in meta:
        pl = parts_for(p, N)
        s = {}
        for Q in ("Gs", "dCKA", "dN_k10"):
            s[Q] = {k: CM.boot_mean_ci([x[Q][k] for x in pl], f"KA5|{p}|{N}|{Q}|{k}") for k in ("X", "S", "M", "B", "I", "I_M")}
            s[Q]["max_abs_I"] = float(np.max(np.abs([x[Q]["I"] for x in pl])))
            s[Q]["max_abs_M"] = float(np.max(np.abs([x[Q]["M"] for x in pl])))
            s[Q]["max_abs_X_minus_B"] = float(np.max(np.abs([x[Q]["X"] - x[Q]["B"] for x in pl])))
        summ[f"{p}|{N}"] = s
    g = lambda p, k, f="mean", N=250000: summ[f"{p}|{N}"]["Gs"][k][f] if f == "mean" else summ[f"{p}|{N}"]["Gs"][k]["ci"]
    checks = {
        "common: S>0 (CI)": g("common", "S", "ci")[0] > 0,
        "common: M=0 (max|M| G_s <= 1e-9)": summ["common|250000"]["Gs"]["max_abs_M"] <= 1e-9,
        "common+run0.15: S>0, M>0 (CI)": g("common+run0.15", "S", "ci")[0] > 0 and g("common+run0.15", "M", "ci")[0] > 0,
        "common+run0.30: S>0, M>0 (CI)": g("common+run0.30", "S", "ci")[0] > 0 and g("common+run0.30", "M", "ci")[0] > 0,
        "planted order M: 0 < M(0.15) < M(0.30)": 0 < g("common+run0.15", "M") < g("common+run0.30", "M"),
        "balanced: I=0, X=B (G_s, 1-CKA, max <= 1e-9; both N)": all(
            summ[f"balanced|{N}"][Q]["max_abs_I"] <= 1e-9 and summ[f"balanced|{N}"][Q]["max_abs_X_minus_B"] <= 1e-9
            for N in CM.NS for Q in ("Gs", "dCKA")),
        "balanced+run0.30: I>0 and I_M>0 on G_s (CI; both N)": all(
            summ[f"balanced+run0.30|{N}"]["Gs"]["I"]["ci"][0] > 0 and summ[f"balanced+run0.30|{N}"]["Gs"]["I_M"]["ci"][0] > 0
            for N in CM.NS),
    }
    syn = i4_synthetic()
    i4_plant = DC.i4_map({N: parts_for("balanced+run0.30", N) for N in CM.NS}, sig, "KA5|I4|plant")
    i4_bal = DC.i4_map({N: parts_for("balanced", N) for N in CM.NS}, sig, "KA5|I4|balanced")
    checks["I4 synthetic: all four known answers"] = all(v["ok"] for v in syn.values())
    checks["I4 planted balanced+run0.30: realized"] = bool(i4_plant["realized"])
    checks["I4 planted balanced: within margin (I = 0)"] = bool(i4_bal["within_margin"])
    # (d) real v1.0 cells (draft expectations), 96 pairs, Gbar / Gbar_bal from the cell's own 192 runs
    real = {}
    rjobs, rpairs = [], {}
    for arm in ("WD", "noWD"):
        for N in CM.NS:
            pr = [(a, b) for _, a, b in cells[f"{arm}_{N}"]]
            Gbar, Gbal, _ = cell_gbars([r for p in pr for r in p])
            rpairs[(arm, N)] = pr
            rjobs += DC.pair_jobs(LROOT, V10, pr, Gbar, Gbal, tags=("y", "u", "uc", "ubal", "ubalc"))
    rres = CM.run_ladders(rjobs, JOBS, "KA5 real v1.0")
    for arm, expect in (("WD", "(ii)"), ("noWD", "(i)")):
        pbn = {N: DC.cell_parts(rres, rpairs[(arm, N)]) for N in CM.NS}
        sg = {N: {Q: DC.sigma_y(rres, rpairs[(arm, N)], Q) for Q in ("Gs", "dCKA")} for N in CM.NS}
        m = DC.i4_map(pbn, sg, f"KA5|I4|real|{arm}")
        real[arm] = {"outcome": m["outcome"], "draft_expectation": expect, "matches_draft": m["outcome"].startswith(expect),
                     "intervals_sigma": {k: v["ci_sigma"] for k, v in m["intervals"].items()},
                     "means_raw": {k: v["mean_raw"] for k, v in m["intervals"].items()},
                     "B_Gs_mean": {N: float(np.mean([p["Gs"]["B"] for p in pbn[N]])) for N in CM.NS},
                     "X_Gs_mean": {N: float(np.mean([p["Gs"]["X"] for p in pbn[N]])) for N in CM.NS}}
    ok = all(checks.values())
    rec("KA5_plants_and_I4", ok, checks=checks, i4_synthetic=syn, i4_planted=i4_plant["outcome"],
        i4_planted_balanced=i4_bal["outcome"], real_v10=real,
        plant_summary={k: {Q: {p: v[Q][p]["mean"] for p in ("X", "S", "M", "B", "I", "I_M")} for Q in v} for k, v in summ.items()},
        sigma_y_WD48=sig, wall_s=time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    JOBS = a.jobs
    os.makedirs(OUT, exist_ok=True)
    tests = {"ka0": ka0, "ka1": ka1, "ka2": ka2, "ka3": ka3, "ka4": ka4, "ka5": ka5}
    t0 = time.time()
    for k, f in tests.items():
        if a.which in ("all", k):
            f()
    print("ALL PASS" if all(v["pass"] for v in RESULTS.values()) else "SOME FAILED", f"({time.time() - t0:.0f} s)")
