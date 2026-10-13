"""S2.2 v1.1 pilot gates GA-GC and the tolerance freeze (PREREG_S2_2_v1.1.md sections 6 and 8).

    python src_v11/gates_v11.py [--cohort S2_2v11_pilot] [--jobs 6]

Requires: frozen code (guard_v11), the complete pilot cohort (every planned run .done or .failed after the one
retry). Pilots are EXCLUDED from every count; only pilot-pair ladders are computed here (no main ladder).
1. GA conservation / identity, GB health, GC recipe sanity (per-run metadata only) -> runs_v11/GATE_S2_2v11_main.go|nogo
   (written first, so main runs can start), results/S2_2_v11/gates/GATES.json, one STATUS line.
2. Gbar, Gbar_bal per (arm, N) from the arm's pilot runs; pilot ladders (y, u, h full; h_l2 core arms; u^c reduced);
   anchors Y_a, Pool, O, C, h, h_l2 and their 0.9-quantile versions; homogeneity diagnostic ->
   results/S2_2_v11/eps/eps_S2_2_v11.json (chmod 444), sha256 in one STATUS line.
Resolutions applied (prereg section 17): a pilot pair containing a failed run is excluded from the anchors (count
reported); a diverged or non-finite pilot run in a core arm makes GB fail (NOGO, the author decides).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_v11 as CM  # noqa: E402
import decomp_v11 as DC  # noqa: E402
import gaugemath_v11 as GM  # noqa: E402
import make_plan_v11 as MP  # noqa: E402

STATUS = os.path.expanduser("~/hrh_v2_stage1_20261008/STATUS.md")


def status(msg, tag="[s22v11-gates]"):
    with open(STATUS, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {tag} {msg}\n")


def gates(cohort, cells, dry=False):
    runs = {}
    for (arm, N), reps in cells.items():
        for rep, rid in reps.items():
            runs[(arm, N, rep)] = CM.load_run(cohort, rid)
    meta = lambda r: r["meta"] if r is not None else None
    GA = {"flow_delta_C_max": {}, "flow_viol": [], "euler_resid_max": {}, "euler_viol": []}
    for (arm, N, rep), r in runs.items():
        if r is None or not CM.run_ok(r):
            continue
        m = meta(r)
        if arm in MP.FLOW_ARMS:
            dC = GM.delta_C(r["feats"]["P0"], r["feats"]["W0"], r["feats"]["P"], r["feats"]["W"])
            GA["flow_delta_C_max"][f"{arm}_{N}"] = max(GA["flow_delta_C_max"].get(f"{arm}_{N}", 0.0), dC)
            if dC > 2e-4:
                GA["flow_viol"].append([arm, N, rep, dC])
        if m["cfg"]["head_opt"] == "euler":
            e = m["euler_identity_resid"]
            GA["euler_resid_max"][f"{arm}_{N}"] = max(GA["euler_resid_max"].get(f"{arm}_{N}", 0.0), e)
            if not (e <= 1e-8):
                GA["euler_viol"].append([arm, N, rep, e])
    GA["pass"] = not GA["flow_viol"] and not GA["euler_viol"]
    GB = {"diverged_or_nonfinite": [], "acc_bank_mean": {}, "acc_flag": [], "failed_runs": []}
    for (arm, N), reps in cells.items():
        accs = []
        for rep in reps:
            r = runs[(arm, N, rep)]
            if r is None:
                GB["failed_runs"].append([arm, N, rep])
                if arm in MP.CORE:
                    GB["diverged_or_nonfinite"].append([arm, N, rep, "failed"])
                continue
            if not CM.run_ok(r):
                if arm in MP.CORE:
                    GB["diverged_or_nonfinite"].append([arm, N, rep, "diverged/non-finite"])
                continue
            accs.append(r["meta"]["acc_bank"])
        GB["acc_bank_mean"][f"{arm}_{N}"] = float(np.mean(accs)) if accs else float("nan")
    for (arm, N) in cells:
        a, mo = GB["acc_bank_mean"][f"{arm}_{N}"], GB["acc_bank_mean"].get(f"MOM_{N}", float("nan"))
        if not (abs(a - mo) <= 0.015):
            GB["acc_flag"].append(f"{arm}_{N}")
    GB["pass"] = not GB["diverged_or_nonfinite"]
    GC = {"rho_end_median": {}}
    for arm in ("MOM", "WD"):
        for N in CM.NS:
            v = [GM.rho_of(runs[(arm, N, rep)]["feats"]["P"], runs[(arm, N, rep)]["feats"]["W"])
                 for rep in cells.get((arm, N), {}) if CM.run_ok(runs[(arm, N, rep)])]
            GC["rho_end_median"][f"{arm}_{N}"] = float(np.median(v)) if v else float("nan")
    GC["pass"] = all(GC["rho_end_median"][f"MOM_{N}"] >= 0.2 and GC["rho_end_median"][f"WD_{N}"] <= 0.05 for N in CM.NS)
    ok = GA["pass"] and GB["pass"] and GC["pass"]
    return {"GA": GA, "GB": GB, "GC": GC, "go": ok}, runs


def freeze(cohort, cells, runs, gb_flags, n_jobs, lroot, out_path):
    G = {}
    for (arm, N), reps in cells.items():
        Gbar, Gbal, n = DC.gbars([runs[(arm, N, r)] for r in reps])
        G[f"{arm}_{N}"] = {"Gbar": None if Gbar is None else Gbar.tolist(),
                           "Gbar_bal": None if Gbal is None else Gbal.tolist(), "n_runs": n,
                           "n_skipped_not_ok_or_not_spd": len(reps) - n}
    jobs = []
    pairs = {}
    for (arm, N) in cells:
        pr = CM.cell_pairs(cells, arm, N)
        pairs[(arm, N)] = pr
        tags = ["y", "u", "h"] + (["h_l2"] if arm in MP.CORE else [])
        for a, b in pr:
            for t in tags:
                jobs.append((lroot, cohort, t, a, b, None, None))
            jobs.append((lroot, cohort, "uc", a, b, G[f"{arm}_{N}"]["Gbar"], None))
    res = CM.run_ladders(jobs, n_jobs, "pilot")
    eps = {}
    excluded = {}

    def anchor(vals):
        v = np.array([x for x in vals if np.isfinite(x)], float)
        return (float(v.max()), float(np.quantile(v, 0.9)), int(v.size)) if v.size else (float("nan"), float("nan"), 0)

    for (arm, N), pr in pairs.items():
        cell = f"{arm}_{N}"
        e = {}
        byrid = {rid: runs[(arm, N, rep)] for rep, rid in cells[(arm, N)].items()}
        failed_pairs = [(a, b) for a, b in pr if not (CM.run_ok(byrid[a]) and CM.run_ok(byrid[b]))]
        excluded[cell] = len(failed_pairs)
        for name, tag, keys in (("Y", "y", CM.TOL_KEYS), ("O", "u", CM.TOL_KEYS), ("C", "uc", CM.R_CF),
                                ("h", "h", CM.TOL_KEYS), ("h_l2", "h_l2", CM.TOL_KEYS)):
            if (tag, pr[0][0], pr[0][1]) not in res:
                continue
            e[name], e[name + "_q90"], e[name + "_n"] = {}, {}, {}
            for k in keys:
                mx, q9, n = anchor([res[(tag, a, b)].get(k, np.nan) for a, b in pr])
                e[name][k], e[name + "_q90"][k], e[name + "_n"][k] = mx, q9, n
        eps[cell] = e
    # pooled function anchor per N over the arms that pass GB's accuracy test
    pool = {}
    homog = {}
    for N in CM.NS:
        arms = [a for (a, n) in pairs if n == N and f"{a}_{N}" not in gb_flags]
        pool[str(N)] = {"arms": arms, "Pool": {}, "Pool_q90": {}}
        for k in CM.TOL_KEYS:
            v = [res[("y", a, b)].get(k, np.nan) for arm in arms for a, b in pairs[(arm, N)]]
            mx, q9, n = anchor(v)
            pool[str(N)]["Pool"][k], pool[str(N)]["Pool_q90"][k] = mx, q9
        hN = {}
        for k in ("dN_k10", "Gs", "dCKA", "dE", "dO", "dMST", "dAff", "dSOG_t0p1"):
            allv = np.array([res[("y", a, b)].get(k, np.nan) for arm in arms for a, b in pairs[(arm, N)]], float)
            allv = allv[np.isfinite(allv)]
            sd = allv.std(ddof=1) if allv.size > 1 else float("nan")
            for (arm, n2), pr in pairs.items():
                if n2 != N:
                    continue
                v = np.array([res[("y", a, b)].get(k, np.nan) for a, b in pr], float)
                v = v[np.isfinite(v)]
                hN.setdefault(arm, {})[k] = float((v.mean() - allv.mean()) / sd) if v.size and sd > 0 else float("nan")
        homog[str(N)] = hN
    out = {"rule": "PREREG_S2_2_v1.1 section 8: anchors from pilot pairs at the same N (max; 0.9-quantile secondary); "
                   "Y_a primary, Pool secondary (arms passing GB accuracy), O, C (u^c with the frozen Gbar), h, h_l2",
           "cohort": cohort, "eps": eps, "pool": pool, "gbars": G, "homogeneity_sigma_units": homog,
           "pilot_pairs_excluded_failed_run": excluded, "gb_acc_flags": sorted(gb_flags),
           "pilot_rows": {f"{t}|{a}|{b}": d for (t, a, b), d in res.items()}}
    CM.jdump(out, out_path)
    os.chmod(out_path, 0o444)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="S2_2v11_pilot")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    dry = a.cohort.startswith("UT_")
    if not dry:
        import guard_v11
        ok, msg = guard_v11.verify_frozen()
        if not ok:
            raise SystemExit(f"REFUSED: {msg}")
    complete, counts = CM.cohort_complete(a.cohort)
    if not complete:
        raise SystemExit(f"pilot cohort incomplete: {counts}")
    cells = CM.cohort_cells(a.cohort)
    base = os.path.join(CM.RES, "dryrun") if dry else CM.RES
    gd = os.path.join(base, "gates")
    os.makedirs(gd, exist_ok=True)
    res, runs = gates(a.cohort, cells, dry)
    res["counts"] = counts
    CM.jdump(res, os.path.join(gd, "GATES.json"))
    gate_name = ("UT_" if dry else "") + "GATE_S2_2v11_main"
    for ext in ("go", "nogo"):
        p = os.path.join(CM.RUNS, gate_name + "." + ext)
        if os.path.exists(p):
            os.remove(p)
    open(os.path.join(CM.RUNS, gate_name + (".go" if res["go"] else ".nogo")), "w").write(json.dumps(
        {"GA": res["GA"]["pass"], "GB": res["GB"]["pass"], "GC": res["GC"]["pass"]}))
    msg = (f"gates GA {'PASS' if res['GA']['pass'] else 'FAIL'} (flow delta_C max {res['GA']['flow_delta_C_max']}, "
           f"Euler resid max {res['GA']['euler_resid_max']}), GB {'PASS' if res['GB']['pass'] else 'FAIL'} "
           f"(diverged/non-finite core pilots {len(res['GB']['diverged_or_nonfinite'])}, accuracy flags {res['GB']['acc_flag']}), "
           f"GC {'PASS' if res['GC']['pass'] else 'FAIL'} (median rho_end {res['GC']['rho_end_median']}); "
           f"pilot EXCLUDED, counts {counts} -> runs_v11/{gate_name}.{'go' if res['go'] else 'nogo'}")
    print(msg, flush=True)
    if not dry:
        status(msg)
    out_path = os.path.join(base, "eps", "eps_S2_2_v11.json")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists (frozen once)")
    t = time.time()
    lroot = os.path.join(base, "ladders")
    freeze(a.cohort, cells, runs, set(res["GB"]["acc_flag"]), a.jobs, lroot, out_path)
    msg2 = (f"eps_S2_2_v11 FROZEN from pilot pairs only (anchors Y_a/Pool/O/C/h/h_l2 + q90, Gbar/Gbar_bal per arm x N): "
            f"{os.path.relpath(out_path, CM.ROOT)} sha256 {CM.sha(out_path)} ({time.time() - t:.0f} s, {a.jobs} procs)")
    print(msg2, flush=True)
    if not dry:
        status(msg2)


if __name__ == "__main__":
    main()
