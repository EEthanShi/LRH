"""Independent re-verification of the FINAL S2.2 analysis (review role s22-recompute).

Read-only on all existing files. Writes only under results/review_final/s22-recompute/.
Does NOT import src/analyze.py or src/crit.py decision code. Primary rungs (s_N(10), Gs, CKA, Pi)
are recomputed from the saved features with independent numpy code; the other rungs are
re-derived from the cached hrhv2 ladder rows with an independent row->Delta mapping.

    OMP_NUM_THREADS=1 venv/bin/python s22_recompute.py compute --jobs 3   # -> recompute_pairs.json
    venv/bin/python s22_recompute.py analyze                               # -> recompute_summary.json
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import glob
import hashlib
import json
import re
import sys
import time
from multiprocessing import get_context

import numpy as np
from scipy.stats import beta as BETA

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
OUT = os.path.join(ROOT, "results", "review_final", "s22-recompute")
LAD = os.path.join(ROOT, "results", "ladders", "S2_2_main")
ARMS = ("noWD", "WD", "smallinit")
NS = (25000, 250000)
TAGS = (("y", "y", "none"), ("u", "u", "none"), ("h", "h", "none"), ("h_l2", "h", "l2"))
RUNGS = ["dN_k10", "Gs", "dCKA", "dO", "dE", "dMST", "dAff", "dSOG_t0p1", "dPi"]
INDEP = ["dN_k10", "Gs", "dCKA", "dPi"]  # recomputed from features


# ----------------------------------------------------------------- independent row -> Delta map
def row_to_delta(row):
    return {"dN_k10": 1.0 - row["s_N_k10"], "Gs": row["Gs_rms"], "dCKA": 1.0 - row["cka_biased"],
            "dO": row["Delta_O"], "dE": row["Delta_E"], "dMST": 1.0 - row["s_T"],
            "dAff": 1.0 - row["rsa_pearson"], "dSOG_t0p1": 1.0 - row["sog_knn10_tau0p1"],
            "dPi": row["gl_projector_dist"]}


# ----------------------------------------------------------------- independent metrics from features
def prep(X, pre):
    X = np.asarray(X, dtype=np.float64)
    if pre == "l2":
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def sqdist(X):
    K = X @ X.T
    d = np.diag(K).copy()
    D = d[:, None] + d[None, :] - 2.0 * K
    np.maximum(D, 0.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def knn(D, k):
    Dc = D.copy()
    np.fill_diagonal(Dc, np.inf)
    return np.argsort(Dc, axis=1, kind="stable")[:, :k]  # stable -> ties broken by index j


def s_overlap(nA, nB):
    k = nA.shape[1]
    return float(np.mean([len(set(a).intersection(b)) for a, b in zip(nA.tolist(), nB.tolist())]) / k)


def gs_rms(DA, DB):
    m = DA.shape[0]
    nuA = DA.sum() / (m * (m - 1))
    nuB = DB.sum() / (m * (m - 1))
    E = DA / nuA - DB / nuB
    return float(np.sqrt((E * E).sum() / (m * (m - 1))))


def cka_lin(XA, XB):
    A = XA - XA.mean(0)
    B = XB - XB.mean(0)
    ab = np.linalg.norm(A.T @ B) ** 2
    return float(ab / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def proj_dist(XA, XB):
    def basis(X):
        Xc = X - X.mean(0)
        U, s, _ = np.linalg.svd(Xc, full_matrices=False)
        r = int(np.sum(s > max(Xc.shape) * np.finfo(float).eps * s[0]))
        return U[:, :r], r
    UA, rA = basis(XA)
    UB, rB = basis(XB)
    s2 = float(np.sum((UA.T @ UB) ** 2))
    return (rA + rB - 2 * s2) / (rA + rB)


def indep_metrics(XA, XB, pre):
    XA, XB = prep(XA, pre), prep(XB, pre)
    DA, DB = sqdist(XA), sqdist(XB)
    sN = s_overlap(knn(DA, 10), knn(DB, 10))
    return {"dN_k10": 1.0 - sN, "Gs": gs_rms(DA, DB), "dCKA": 1.0 - cka_lin(XA, XB), "dPi": proj_dist(XA, XB)}


def load_feats(cohort, rid):
    z = np.load(os.path.join(ROOT, "runs", cohort, rid, "feats.npz"))
    return {k: z[k] for k in ("y", "u", "h", "W")}


def pair_job(args):
    cohort, ra, rb, with_ladder = args
    FA, FB = load_feats(cohort, ra), load_feats(cohort, rb)
    out = {"A": ra, "B": rb, "indep": {}, "ladder": {}, "ladder_check": {}}
    for tag, key, pre in TAGS:
        ok = np.all(np.isfinite(FA[key])) and np.all(np.isfinite(FB[key]))
        out["indep"][tag] = indep_metrics(FA[key], FB[key], pre) if ok else {k: float("inf") for k in INDEP}
        if with_ladder:
            p = os.path.join(LAD, tag, f"{ra}__{rb}.json")
            if not os.path.exists(p):
                out["ladder_check"][tag] = "MISSING"
                continue
            L = json.load(open(p))
            row = L["row"]
            chk = {"A_ok": L["A"] == ra, "B_ok": L["B"] == rb, "tag_ok": L["tag"] == tag,
                   "failed_run": L["failed_run"], "m": row.get("m"),
                   "pre_ok": row.get("pre_A") == pre and row.get("pre_B") == pre}
            mine = row_to_delta(row)
            stored = L["deltas"]
            chk["max_absdiff_row_vs_stored"] = max(abs(mine[k] - stored[k]) for k in RUNGS)
            out["ladder"][tag] = mine
            out["ladder_check"][tag] = chk
    # Pi_u == Pi_y within-run sanity (own code)
    out["piuy"] = [proj_dist(FA["u"].astype(float), FA["y"].astype(float)),
                   proj_dist(FB["u"].astype(float), FB["y"].astype(float))]
    out["condW"] = [float(np.linalg.cond(FA["W"])), float(np.linalg.cond(FB["W"]))]
    return out


# ----------------------------------------------------------------- inventory
def inventory():
    plan = json.load(open(os.path.join(ROOT, "plans", "S2_2_main.json")))["runs"]
    inv = {"n_plan": len(plan), "problems": [], "cells": {}}
    subsets = {}
    for r in plan:
        d = os.path.join(ROOT, "runs", "S2_2_main", r["run_id"])
        done = os.path.exists(os.path.join(d, ".done"))
        failed = os.path.exists(os.path.join(d, ".failed"))
        meta = json.load(open(os.path.join(d, "meta.json"))) if done else {}
        cellk = f"{r['arm']}_{r['N']}" + (f"_c{r['corpus']}" if r.get("corpus") is not None else "")
        c = inv["cells"].setdefault(cellk, {"n": 0, "done": 0, "failed": 0, "diverged": 0, "nonfinite_meta": 0,
                                            "meta_mismatch": 0, "steps": set(), "reps": []})
        c["n"] += 1
        c["done"] += done
        c["failed"] += failed
        c["reps"].append(r["rep"])
        if done:
            c["diverged"] += bool(meta.get("diverged"))
            c["nonfinite_meta"] += (not meta.get("finite", False))
            mm = (meta.get("arm") != r["arm"] or meta.get("N") != r["N"] or meta.get("rep") != r["rep"]
                  or meta.get("run_id") != r["run_id"])
            c["meta_mismatch"] += mm
            c["steps"].add(meta.get("steps"))
            subsets.setdefault(meta.get("subset_sha256"), []).append(r["run_id"])
    for k, c in inv["cells"].items():
        c["steps"] = sorted(c["steps"])
        c["reps_complete"] = sorted(c["reps"]) == list(range(len(c["reps"])))
        del c["reps"]
    # subset structure: independent runs should all have distinct subsets; shared corpus: one per corpus
    shared_groups = {}
    for sha_, rids in subsets.items():
        if len(rids) > 1:
            shared_groups[sha_] = rids
    sg = []
    for sha_, rids in shared_groups.items():
        corp = sorted(set(re.sub(r".*_c(\d)$", r"\1", x) if "shared" in x else "INDEP" for x in rids))
        sg.append({"n": len(rids), "corpora": corp})
    inv["n_distinct_subsets"] = len(subsets)
    inv["shared_subset_groups"] = sg
    return inv, plan


def expected_pairs(plan):
    by = {}
    for r in plan:
        key = (r["arm"], r["N"], r.get("corpus"))
        by.setdefault(key, {})[r["rep"]] = r["run_id"]
    sets = {}
    for a in ARMS:
        for N in NS:
            reps = by[(a, N, None)]
            sets[f"{a}_{N}"] = [(reps[2 * j], reps[2 * j + 1]) for j in range(len(reps) // 2)]
    sets["noWD-shared_25000"] = [(by[("noWD-shared", 25000, c)][2 * j], by[("noWD-shared", 25000, c)][2 * j + 1])
                                 for c in range(4) for j in range(len(by[("noWD-shared", 25000, c)]) // 2)]
    return sets


def pilot_pairs():
    plan = json.load(open(os.path.join(ROOT, "plans", "S2_2_pilot.json")))["runs"]
    out = []
    for a in ARMS:
        reps = {r["rep"]: r["run_id"] for r in plan if r["arm"] == a}
        Ns = {r["N"] for r in plan if r["arm"] == a}
        out += [(a, reps[2 * j], reps[2 * j + 1]) for j in range(len(reps) // 2)]
    return out, sorted({r["N"] for r in plan}), len(plan)


def compute(n_jobs):
    os.makedirs(OUT, exist_ok=True)
    inv, plan = inventory()
    sets = expected_pairs(plan)
    # ladder dir contents vs expected
    exp_files = {f"{a}__{b}.json" for s in sets.values() for a, b in s}
    lad_inv = {}
    for tag, _, _ in TAGS:
        have = {os.path.basename(p) for p in glob.glob(os.path.join(LAD, tag, "*.json"))}
        lad_inv[tag] = {"n_files": len(have), "missing": sorted(exp_files - have)[:5], "n_missing": len(exp_files - have),
                        "unexpected": sorted(have - exp_files)[:5], "n_unexpected": len(have - exp_files)}
    jobs = [("S2_2_main", a, b, True) for s in sets.values() for a, b in s]
    pp, pilotN, npilot = pilot_pairs()
    jobs_p = [("S2_2_pilot", a, b, False) for _, a, b in pp]
    t = time.time()
    res = {}
    with get_context("fork").Pool(n_jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(pair_job, jobs + jobs_p, chunksize=2)):
            res[(r["A"], r["B"])] = r
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(jobs) + len(jobs_p)} {time.time() - t:.0f}s", flush=True)
    out = {"inventory": inv, "ladder_inventory": lad_inv, "sets": sets,
           "pilot": {"pairs": pp, "N": pilotN, "n_runs": npilot},
           "pairs": {f"{a}__{b}": v for (a, b), v in res.items()}}
    with open(os.path.join(OUT, "recompute_pairs.json"), "w") as f:
        json.dump(out, f, default=lambda v: sorted(v) if isinstance(v, set) else float(v))
    print("done", time.time() - t)


# ----------------------------------------------------------------- decisions (own CP)
def cp(x, n, a=0.05):
    lo = 0.0 if x == 0 else float(BETA.ppf(a, x, n - x + 1))
    hi = 1.0 if x == n else float(BETA.ppf(1 - a, x + 1, n - x))
    return lo, hi


def decide(x, n):
    lo, hi = cp(x, n)
    return ("rel" if lo > 0.9 else "unr" if hi < 0.8 else "inc"), lo, hi


def boot_diff(a, b, nb=2000, seed=12345):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = a[rng.integers(0, a.size, (nb, a.size))].mean(1) - b[rng.integers(0, b.size, (nb, b.size))].mean(1)
    return {"diff": float(a.mean() - b.mean()), "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
            "p_diff_ge_0": float((np.sum(d >= 0) + 1) / (nb + 1))}


def analyze():
    P = json.load(open(os.path.join(OUT, "recompute_pairs.json")))
    epsf = json.load(open(os.path.join(ROOT, "prereg", "eps_S2_2.json")))
    S = json.load(open(os.path.join(ROOT, "results", "S2_2", "s22_summary.json")))
    rep_md = open(os.path.join(ROOT, "results", "S2_2", "S2_2_report.md")).read()
    R = {"inventory": P["inventory"], "ladder_inventory": P["ladder_inventory"]}
    sets = P["sets"]
    pairs = P["pairs"]

    # ---- pairing check
    pc = {}
    for s, lst in sets.items():
        bad = 0
        for a, b in lst:
            ra, rb = int(re.search(r"_r(\d+)", a).group(1)), int(re.search(r"_r(\d+)", b).group(1))
            ca, cb = a.rsplit("_r", 1)[0], b.rsplit("_r", 1)[0]
            ka = re.search(r"_c(\d)$", a)
            kb = re.search(r"_c(\d)$", b)
            if not (ra % 2 == 0 and rb == ra + 1 and ca == cb and (ka.group(1) if ka else None) == (kb.group(1) if kb else None)):
                bad += 1
        allruns = [x for p in lst for x in p]
        pc[s] = {"n_pairs": len(lst), "bad_pairing": bad, "runs_used": len(allruns), "runs_distinct": len(set(allruns))}
    # compare with analyze's pair lists (summary 'values' order follows its pair list; use ladder files)
    R["pairing"] = pc

    # ---- ladder checks
    lc = {"n": 0, "A/B/tag mismatch": 0, "failed_run": 0, "m!=2048": 0, "pre mismatch": 0, "max_row_vs_stored": 0.0}
    for k, v in pairs.items():
        for tag, c in v.get("ladder_check", {}).items():
            if c == "MISSING":
                lc.setdefault("missing", 0)
                lc["missing"] += 1
                continue
            lc["n"] += 1
            lc["A/B/tag mismatch"] += not (c["A_ok"] and c["B_ok"] and c["tag_ok"])
            lc["failed_run"] += bool(c["failed_run"])
            lc["m!=2048"] += c["m"] != 2048
            lc["pre mismatch"] += not c["pre_ok"]
            lc["max_row_vs_stored"] = max(lc["max_row_vs_stored"], c["max_absdiff_row_vs_stored"])
    R["ladder_checks"] = lc
    piuy = [x for v in pairs.values() if "S2.2_" in v["A"] and "pilot" not in v["A"] for x in v["piuy"]]
    condW = [x for v in pairs.values() if "pilot" not in v["A"] for x in v["condW"]]
    R["pi_u_eq_pi_y"] = {"n_runs": len(piuy), "max": float(np.max(piuy)), "max_condW": float(np.max(condW))}
    cw = {}
    for s_, lst in sets.items():
        c_ = [x for a, b in lst for x in pairs[f"{a}__{b}"]["condW"]]
        cw[s_] = {q: float(np.quantile(c_, q)) for q in (0, .1, .5, .9, 1)}
    R["condW_by_cell"] = cw

    # ---- agreement of independent feature metrics vs ladder rows
    agree = {}
    for tag, _, _ in TAGS:
        for k in INDEP:
            d = [abs(v["indep"][tag][k] - v["ladder"][tag][k]) for v in pairs.values() if v["ladder"]]
            agree[f"{tag}|{k}"] = float(np.max(d))
    R["indep_vs_ladder_maxabs"] = agree

    # ---- eps check from pilot
    eps = epsf["eps"]
    pil = P["pilot"]
    prow = epsf["pilot_rows"]
    epcheck = {"pilot_N": pil["N"], "pilot_n_runs": pil["n_runs"], "n_pairs": len(pil["pairs"]),
               "file_cohort": epsf["cohort"], "file_anchor": epsf["anchor"], "u_equals_y": eps["u"] == eps["y"],
               "pilot_pairs_match_file": sorted((a, b) for _, a, b in pil["pairs"]) ==
               sorted({(r["A"], r["B"]) for r in prow}), "rows": {}}
    for tag in ("y", "u", "h", "h_l2"):
        src = "y" if tag == "u" else tag
        for k in RUNGS:
            vals_file = [r[k] for r in prow if r["rep"] == src]
            mx_file = max(vals_file)
            ent = {"eps_file": eps[tag][k], "max_pilot_rows_file": mx_file, "match": abs(mx_file - eps[tag][k]) < 1e-12}
            if k in INDEP:
                mine = [pairs[f"{a}__{b}"]["indep"][src][k] for _, a, b in pil["pairs"]]
                ent["max_pilot_recomputed_from_feats"] = float(max(mine))
                ent["absdiff_recomputed"] = abs(float(max(mine)) - eps[tag][k])
            epcheck["rows"][f"{tag}|{k}"] = ent
    # pilot u pairs measured against eps(y): how much of the pilot u already exceeds eps(y)?
    pu = {}
    for a in ARMS:
        for k in INDEP:
            uu = [pairs[f"{x}__{y}"]["indep"]["u"][k] for arm, x, y in pil["pairs"] if arm == a]
            yy = [pairs[f"{x}__{y}"]["indep"]["y"][k] for arm, x, y in pil["pairs"] if arm == a]
            pu[f"{a}|{k}"] = {"u_le_eps_y": int(np.sum(np.array(uu) <= eps["y"][k])), "n": len(uu),
                              "u_median": float(np.median(uu)), "y_median": float(np.median(yy)),
                              "u_max": float(np.max(uu)), "y_max": float(np.max(yy))}
    epcheck["pilot_u_vs_eps_y"] = pu
    R["eps_check"] = epcheck

    # ---- decision tables (ladder-row deltas; independent for INDEP rungs)
    tabs, mism = {}, []
    md_rows = {}
    for line in rep_md.splitlines():
        m_ = re.match(r"\| (\S+) \| (\S+) \| (.+) \|$", line)
        if m_ and m_.group(1) not in ("set", "---"):
            cells = [c.strip() for c in m_.group(3).split("|")]
            md_rows[f"{m_.group(1)}|{m_.group(2)}"] = cells
    for s, lst in sets.items():
        for tag, _, _ in TAGS:
            name = f"{s}|{tag}"
            t = {}
            for j, k in enumerate(RUNGS):
                e = eps[tag][k]
                v_l = np.array([pairs[f"{a}__{b}"]["ladder"][tag][k] for a, b in lst])
                x = int(np.sum(v_l <= e))
                d, lo, hi = decide(x, len(v_l))
                ent = {"x": x, "n": len(v_l), "dec": d, "lo": lo, "hi": hi, "eps": e,
                       "mean": float(v_l.mean()), "median": float(np.median(v_l))}
                if k in INDEP:
                    v_i = np.array([pairs[f"{a}__{b}"]["indep"][tag][k] for a, b in lst])
                    ent["x_indep"] = int(np.sum(v_i <= e))
                    ent["dec_indep"] = decide(ent["x_indep"], len(v_i))[0]
                    ent["min_margin_to_eps"] = float(np.min(np.abs(v_i - e)))
                # compare with summary json
                sj = S["tables"][name][k]
                ent["summary_x"] = sj["x"]
                ent["summary_dec"] = sj["decision"][:3]
                ent["summary_eps_match"] = abs(sj["eps"] - e) < 1e-12
                sv = np.array(S["values"][name][k], float)
                ent["summary_values_maxabs"] = float(np.max(np.abs(np.sort(sv) - np.sort(v_l))))
                ent["md"] = md_rows.get(name, ["?"] * 9)[j]
                ok = (ent["md"] == f"{d} {x}/{len(v_l)}" and sj["x"] == x and sj["decision"][:3] == d
                      and ent.get("x_indep", x) == x)
                ent["match"] = ok
                if not ok:
                    mism.append((name, k, ent))
                t[k] = ent
            tabs[name] = t
    R["tables"] = tabs
    R["mismatches"] = [(a, b, {kk: c[kk] for kk in ("x", "dec", "md", "summary_x", "summary_dec", "x_indep") if kk in c})
                       for a, b, c in mism]
    # CP thresholds at n = 96
    xs = list(range(97))
    R["cp_thresholds_n96"] = {"reliable_min_x": min(x for x in xs if decide(x, 96)[0] == "rel"),
                              "unreliable_max_x": max(x for x in xs if decide(x, 96)[0] == "unr")}

    # ---- predictions
    dec = lambda s, tag, k: tabs[f"{s}|{tag}"][k]["dec"]
    ph = lambda s, tag, k: tabs[f"{s}|{tag}"][k]["x"] / tabs[f"{s}|{tag}"][k]["n"]
    prim = ("dN_k10", "Gs")
    pr = {}
    pr["P1"] = {f"{k}_{N}": dec(f"noWD_{N}", "u", k) for k in prim for N in NS}
    pr["P1"]["pass"] = all(v == "unr" for v in list(pr["P1"].values()))
    pr["P2"] = {f"y_{k}_{N}": dec(f"noWD_{N}", "y", k) for k in prim for N in NS}
    pr["P2"].update({f"Pi_y_{N}": dec(f"noWD_{N}", "y", "dPi") for N in NS})
    pr["P2"].update({f"Pi_u_{N}": dec(f"noWD_{N}", "u", "dPi") for N in NS})
    pr["P2"]["pass_250k"] = all(dec("noWD_250000", "y", k) == "rel" for k in prim + ("dPi",))
    pr["P3"] = {f"{a}_{k}_{N}": dec(f"{a}_{N}", "u", k) for a in ("WD", "smallinit") for k in prim for N in NS}
    pr["P3"]["pass_WD"] = all(dec("WD_250000", "u", k) == "rel" for k in prim)
    pr["P3"]["pass_smallinit"] = all(dec("smallinit_250000", "u", k) == "rel" for k in prim)
    p4 = {}
    for N in NS:
        m = {a: float(np.mean([1 - pairs[f"{x}__{y}"]["indep"]["h"]["dN_k10"] for x, y in sets[f"{a}_{N}"]])) for a in ARMS}
        ml = {a: float(np.mean([1 - pairs[f"{x}__{y}"]["indep"]["h_l2"]["dN_k10"] for x, y in sets[f"{a}_{N}"]])) for a in ARMS}
        p4[str(N)] = {"mean_sN10_h_none": m, "max_abs_diff": max(abs(m[i] - m[j]) for i in ARMS for j in ARMS),
                      "mean_sN10_h_l2": ml, "max_abs_diff_l2": max(abs(ml[i] - ml[j]) for i in ARMS for j in ARMS)}
    p4["pass"] = all(p4[str(N)]["max_abs_diff"] < 0.05 for N in NS)
    pr["P4"] = p4
    pr["P5"] = {f"{k}_{N}": abs(ph(f"smallinit_{N}", "u", k) - ph(f"WD_{N}", "u", k)) for k in prim for N in NS}
    pr["P5"]["pass"] = all(v <= 0.1 for v in pr["P5"].values())
    pr["falsifier"] = {f"{k}_{N}": dec(f"noWD_{N}", "u", k) == "rel" for k in prim for N in NS}
    pr["falsifier"]["triggered"] = any(pr["falsifier"].values())
    sh = {}
    for tag in ("y", "u", "h"):
        for k in ("dN_k10", "Gs", "dPi", "dCKA"):
            a = [pairs[f"{x}__{y}"]["ladder"][tag][k] for x, y in sets["noWD-shared_25000"]]
            b = [pairs[f"{x}__{y}"]["ladder"][tag][k] for x, y in sets["noWD_25000"]]
            sj = S["predictions"]["shared_corpus"][f"{tag}_{k}"]
            sh[f"{tag}_{k}"] = {"p_shared": ph("noWD-shared_25000", tag, k), "p_indep": ph("noWD_25000", tag, k),
                                "dec_shared": dec("noWD-shared_25000", tag, k), "dec_indep": dec("noWD_25000", tag, k),
                                "mean_shared": float(np.mean(a)), "mean_indep": float(np.mean(b)),
                                "boot": boot_diff(a, b), "report_diff": sj["mean_delta_diff_shared_minus_indep"]["diff"],
                                "report_ci": sj["mean_delta_diff_shared_minus_indep"]["ci95"]}
    pr["shared"] = sh
    R["predictions"] = pr

    # ---- p_hat(eps) curves and eps sensitivity (exploratory)
    curves = {}
    mults = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    for s in list(sets.keys()):
        for tag in ("y", "u", "h"):
            for k in ("dN_k10", "Gs", "dCKA"):
                v = np.sort([pairs[f"{a}__{b}"]["indep"][tag][k] for a, b in sets[s]])
                e = eps[tag][k]
                n = v.size
                curves[f"{s}|{tag}|{k}"] = {
                    "eps": e, "q": {q: float(np.quantile(v, q)) for q in (0, .1, .5, .9, 1)},
                    "p_at_mult": {m_: float(np.mean(v <= m_ * e)) for m_ in mults},
                    "eps_needed_rel_92": float(v[91]), "eps_needed_rel_ratio": float(v[91] / e),
                    "eps_unrel_70th": float(v[69]),  # x<=69 iff eps < v[69]
                }
    # pilot distributions for reference
    for a in ARMS + ("pooled",):
        for tag in ("y", "u", "h"):
            for k in ("dN_k10", "Gs", "dCKA"):
                v = np.sort([pairs[f"{x}__{y}"]["indep"][tag][k] for arm, x, y in pil["pairs"] if a in ("pooled", arm)])
                curves[f"pilot100k_{a}|{tag}|{k}"] = {"q": {q: float(np.quantile(v, q)) for q in (0, .1, .5, .9, 1)}}
    R["curves"] = curves

    # ---- exploratory: N-matched function anchor and paired u-vs-y
    expl = {}
    for N in NS:
        pooled_y = {k: max(pairs[f"{x}__{y}"]["indep"]["y"][k] for a in ARMS for x, y in sets[f"{a}_{N}"]) for k in INDEP}
        for a in ARMS + (("noWD-shared",) if N == 25000 else ()):
            s = f"{a}_{N}"
            for k in ("dN_k10", "Gs", "dCKA"):
                uu = np.array([pairs[f"{x}__{y}"]["indep"]["u"][k] for x, y in sets[s]])
                yy = np.array([pairs[f"{x}__{y}"]["indep"]["y"][k] for x, y in sets[s]])
                # cross-fit: eps from y of first 48 pairs, test u on last 48 and vice versa
                h1, h2 = slice(0, 48), slice(48, 96)
                xcf = int(np.sum(uu[h2] <= yy[h1].max()) + np.sum(uu[h1] <= yy[h2].max()))
                expl[f"{s}|{k}"] = {
                    "u_le_maxy_samecell": int(np.sum(uu <= yy.max())),
                    "u_le_maxy_crossfit": xcf,
                    "dec_crossfit": decide(xcf, 96)[0],
                    "u_le_pooledmaxy_N": int(np.sum(uu <= pooled_y[k])),
                    "median_u_minus_y": float(np.median(uu - yy)), "frac_u_gt_y": float(np.mean(uu > yy)),
                    "median_ratio_u_over_y": float(np.median(uu / yy)),
                    "mean_u": float(uu.mean()), "mean_y": float(yy.mean())}
    R["exploratory"] = expl
    with open(os.path.join(OUT, "recompute_summary.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("analysis written; mismatches:", len(mism))


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        nj = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 3
        compute(min(nj, 4))
    else:
        analyze()
