"""S2.2 v1.1 report rendering (role s22v11-analyze, 2026-10-10). READ-ONLY on the frozen analysis output.

The frozen analysis (src_v11/analyze_v11.py, manifest f2d5d28d...) ran ONCE on the complete main cohort
(STATUS 01:43:50Z -> 03:02:39Z, rc 0) and wrote results/S2_2_v11/{s22v11_summary.json, S2_2_v11_report.md,
kblock_pairs.json, ladders/}. analyze_v11.py refuses to run a second time ("summary exists: the final analysis
runs once", prereg section 14 step 9), so this script does NOT re-run any test. It only renders what the frozen
run computed, in the full form required by prereg section 13 (n, x, p_hat, both CP bounds, decision, RG flag and
final call, anchor flag, every anchor eps, eps*_r, p_hat(eps) curves; estimation endpoints with mean, median, CI,
share > 0, sigma_y units), plus per-cell medians of per-run covariates (descriptive aggregation only).

Refuses to run unless: the frozen-code guard verifies; the summary and eps sha256 equal the values quoted in STATUS;
no output file exists yet (write-once). Outputs:
  results/S2_2_v11/summary.json                 compact extract (no per_run / values / p_hat curves) + provenance
  results/S2_2_v11/S2_2_v11_report_full.md      full report
  results/S2_2_v11/render/{decision_entries.csv, phat_curves.csv, a4_parts.csv, anchors.csv, covariates_by_cell.csv}
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
RES = os.path.join(ROOT, "results", "S2_2_v11")
SUMMARY = os.path.join(RES, "s22v11_summary.json")
EPSF = os.path.join(RES, "eps", "eps_S2_2_v11.json")
STATUS = os.path.expanduser("~/hrh_v2_stage1_20261008/STATUS.md")
SUM_SHA = "c05262cba20219458eb132be822c7114a881fbf2de2b8ea1adc24d525fd2437c"
EPS_SHA = "dd649d74fb3a87fd99f049a8676c2fe1751c476093522db7d9f3aba7145935f7"
REPORT_SHA = "c53d1bca417ee75daf6d708c0ea4b6b9e30872f11b1502db58325f455440aa2b"
KB_SHA = "85f14643d6bd124b0bea2b0317aef59abed3da6befd41293222d8ac541159c7c"
OUT_JSON = os.path.join(RES, "summary.json")
OUT_MD = os.path.join(RES, "S2_2_v11_report_full.md")
RD = os.path.join(RES, "render")

NS = ("25000", "250000")
CORE = ("MOM", "WD", "FLOW", "BAL-FLOW", "SMALL-FLOW")
SEC = ("BAL-MOM", "SMALL-MOM", "SGD-K")
LADDER_CELLS = [f"{a}_{n}" for a in CORE + SEC for n in NS]
PRIMQ = ("dN_k10", "Gs", "dCKA")
QNAME = {"dN_k10": "T_N(10)", "Gs": "G_s", "dCKA": "1-CKA", "dE": "T_E", "dO": "T_O", "dMST": "MST", "dAff": "1-RSA",
         "dSOG_t0p1": "SOG_0.1"}
ENTRY_ORDER = ("y@Y", "u@Y", "uc@Y", "ubal@Y", "yM@Y", "yI@Y", "h@Y", "y@Pool", "u@Pool", "uc@Pool", "u@O", "uc@C", "u@C",
               "y@Yq90", "u@Yq90", "h@h", "h_l2@h_l2")
PARTS = ("X", "S", "M", "B", "I", "S_bal", "M_bal", "I_S", "I_M", "S_pair", "M_pair")
COVS = ("acc_bank", "acc_S", "ce_train", "ce_hold", "gen_gap", "cond_W", "cond_W0", "cond_M", "s", "D", "rho_0", "rho_end",
        "delta_C", "growth_rel", "mu_C", "C0_op", "C_end_op", "disp_P_median", "disp_W_median", "head_substeps_total",
        "t_total_s")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def f(x, d=4):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return str(x)
    if x != 0 and (abs(x) < 10 ** (-d) or abs(x) >= 1e5):
        return f"{x:.3g}"
    return f"{x:.{d}f}"


def ci(c, d=4):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


def main():
    sys.path.insert(0, os.path.join(ROOT, "src_v11"))
    import guard_v11
    ok, msg = guard_v11.verify_frozen()
    if not ok:
        raise SystemExit(f"REFUSED: {msg}")
    st = open(STATUS).read()
    for p, h in ((SUMMARY, SUM_SHA), (EPSF, EPS_SHA), (os.path.join(RES, "S2_2_v11_report.md"), REPORT_SHA),
                 (os.path.join(RES, "kblock_pairs.json"), KB_SHA)):
        got = sha(p)
        if got != h or h not in st:
            raise SystemExit(f"REFUSED: {p} sha256 {got} != {h} or not quoted in STATUS")
    outs = [OUT_JSON, OUT_MD] + [os.path.join(RD, n) for n in ("decision_entries.csv", "phat_curves.csv", "a4_parts.csv",
                                                                "anchors.csv", "covariates_by_cell.csv")]
    if any(os.path.exists(o) for o in outs):
        raise SystemExit("REFUSED: an output exists already (write-once)")
    os.makedirs(RD, exist_ok=True)
    S = json.load(open(SUMMARY))
    E = json.load(open(EPSF))
    C, I, A, K, T = S["C"], S["I"], S["A"], S["K"], S["tables"]

    # ------------------------------------------------------------------ CSV: decision entries + curves
    with open(os.path.join(RD, "decision_entries.csv"), "w", newline="") as fh, \
            open(os.path.join(RD, "phat_curves.csv"), "w", newline="") as fc:
        w = csv.writer(fh)
        wc = csv.writer(fc)
        w.writerow(["cell", "rung", "entry", "n", "x", "p_hat", "cp_lo_one_sided95", "cp_hi_one_sided95", "decision",
                    "rg_ok", "final", "eps", "x_needed_reliable", "eps_star_r", "u_anchor_sensitive", "uc_anchor_sensitive"])
        wc.writerow(["cell", "rung", "entry", "eps_ratio", "p_hat"])
        for cell, tab in T.items():
            for Q, row in tab.items():
                for k, v in row.items():
                    if not isinstance(v, dict):
                        continue
                    w.writerow([cell, Q, k, v.get("n"), v.get("x"), v.get("p_hat"), v.get("cp_lo"), v.get("cp_hi"),
                                v.get("decision"), v.get("rg_ok"), v.get("final"), v.get("eps"), v.get("x_needed_reliable"),
                                v.get("eps_star_r"), row.get("u_anchor_sensitive"), row.get("uc_anchor_sensitive")])
                    for r, p in v.get("p_hat_curve", []) or []:
                        wc.writerow([cell, Q, k, r, p])
    # ------------------------------------------------------------------ CSV: A4 parts
    with open(os.path.join(RD, "a4_parts.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "rung", "part", "n", "mean", "median", "ci95_lo", "ci95_hi", "share_gt0", "mean_sigma_y",
                    "ci95_lo_sigma_y", "ci95_hi_sigma_y", "sigma_y", "S_over_X", "I_over_X"])
        for cell, qs in A["A4"]["cells"].items():
            for Q, d in qs.items():
                for p in PARTS:
                    v = d.get(p)
                    if not v:
                        continue
                    su = v.get("sigma_units", {})
                    w.writerow([cell, Q, p, v.get("n"), v.get("mean"), v.get("median"), v["ci"][0], v["ci"][1],
                                v.get("share_gt0"), su.get("mean"), (su.get("ci") or [None, None])[0],
                                (su.get("ci") or [None, None])[1], d.get("sigma_y"), d.get("S_over_X"), d.get("I_over_X")])
    # ------------------------------------------------------------------ CSV: anchors
    with open(os.path.join(RD, "anchors.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "anchor", "rung", "eps", "n_pilot_pairs"])
        for cell, e in E["eps"].items():
            for an in ("Y", "Y_q90", "O", "O_q90", "C", "C_q90", "h", "h_q90", "h_l2", "h_l2_q90"):
                if an not in e:
                    continue
                nkey = an.replace("_q90", "") + "_n"
                for Q, v in e[an].items():
                    w.writerow([cell, an, Q, v, (e.get(nkey) or {}).get(Q)])
        for N, P in E["pool"].items():
            for an in ("Pool", "Pool_q90"):
                for Q, v in P[an].items():
                    w.writerow([f"POOL_{N} ({'+'.join(P['arms'])})", an, Q, v, None])
    # ------------------------------------------------------------------ covariates (descriptive)
    cov = {}
    for rid, q in S["per_run"].items():
        cell = f"{q['arm']}_{q['N']}"
        c = cov.setdefault(cell, {"n": 0, "n_ok": 0, "vals": {k: [] for k in COVS}})
        c["n"] += 1
        c["n_ok"] += int(bool(q.get("ok")))
        if q.get("ok"):
            for k in COVS:
                v = q.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                    c["vals"][k].append(float(v))
    covsum = {cell: {"n": c["n"], "n_ok": c["n_ok"],
                     **{k: (float(np.median(v)) if v else None) for k, v in c["vals"].items()}} for cell, c in cov.items()}
    with open(os.path.join(RD, "covariates_by_cell.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "n", "n_ok"] + [f"median_{k}" for k in COVS])
        for cell, d in covsum.items():
            w.writerow([cell, d["n"], d["n_ok"]] + [d[k] for k in COVS])

    # ------------------------------------------------------------------ compact summary.json
    tables_nocurve = {cell: {Q: {k: ({kk: vv for kk, vv in v.items() if kk != "p_hat_curve"} if isinstance(v, dict) else v)
                                 for k, v in row.items()} for Q, row in tab.items()} for cell, tab in T.items()}
    compact = {
        "what": "S2.2 v1.1 compact summary: an extract of the frozen analysis output (no number recomputed). "
                "Full per-run rows, per-pair ladder values and p_hat(eps) curves are in s22v11_summary.json and render/.",
        "provenance": {"source_summary": "results/S2_2_v11/s22v11_summary.json", "source_summary_sha256": SUM_SHA,
                       "eps_sha256": EPS_SHA, "frozen_report_sha256": REPORT_SHA, "kblock_pairs_sha256": KB_SHA,
                       "code_manifest_sha256": S.get("manifest_sha256"), "guard_now": msg,
                       "analysis_run": "analyze_v11.py once, 2026-10-10T01:43:50Z..03:02:39Z, rc 0 (STATUS)",
                       "rendered_by": "results/S2_2_v11/render/render_v11.py (role s22v11-analyze)"},
        "cohort": S["cohort"], "counts": S["counts"], "eng_choice": {k: S["eng_choice"].get(k) for k in ("status", "K_min", "flow", "projection_total_gpu_h")},
        "numerical_failures": S["numerical_failures"], "sigma_y": S["sigma_y"],
        "C": C, "I": I, "A": A, "K": {k: v for k, v in K.items()}, "consequences": S["consequences"],
        "tables_without_curves": tables_nocurve, "covariates_by_cell_median": covsum,
        "eps_pool_arms": {N: E["pool"][N]["arms"] for N in E["pool"]}, "gb_acc_flags": E.get("gb_acc_flags"),
        "pilot_pairs_excluded_failed_run": E.get("pilot_pairs_excluded_failed_run"),
        "homogeneity_sigma_units": E.get("homogeneity_sigma_units"),
        "gbar_runs": {c: {"n_runs": g.get("n_runs"), "n_skipped": g.get("n_skipped_not_ok_or_not_spd")} for c, g in E["gbars"].items()},
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(compact, fh, indent=1, default=str)

    # ------------------------------------------------------------------ markdown
    L = []
    a = L.append
    a("# S2.2 v1.1 — full analysis report")
    a("")
    a("Rendered by role `s22v11-analyze` from the output of the single frozen analysis run. **No test was re-run and no "
      "number was recomputed here**; every value below is read from `s22v11_summary.json` (sha256 "
      f"`{SUM_SHA}`) and the frozen eps file (`{EPS_SHA}`). The frozen code's own short report "
      f"`S2_2_v11_report.md` (`{REPORT_SHA}`) is unchanged. Per prereg §14 step 9, two independent recompute reviews "
      "must follow before any claim.")
    a("")
    a(f"- Cohort `{S['cohort']}`: {S['counts']}. Code manifest `{S.get('manifest_sha256')}`; guard at render time: {msg}.")
    a(f"- Engineering choice: K_min = {S['eng_choice'].get('K_min')}, head flow = {S['eng_choice'].get('flow')}.")
    a(f"- Pooled anchor arms: 25k {E['pool']['25000']['arms']}; 250k {E['pool']['250000']['arms']} (GB accuracy flag: {E.get('gb_acc_flags')}).")
    a(f"- R-NUM numerical failures in the main cohort: {S['numerical_failures']}.")
    a("- Decision rule (§9): one-sided 95% Clopper–Pearson; reliable if lower > 0.9, unreliable if upper < 0.8. "
      "n = 96: x ≥ 92 reliable, x ≤ 69 unreliable; n = 32: 32 reliable, ≤ 21 unreliable.")
    a("")

    i4 = I["I4"]["intervals"]
    # ---- 0. observations
    a("## 0. Observations from rendering, for the two recompute reviews (not claims)")
    a("")
    a2 = A["A2"]["cells"]
    fails = [k for k, v in a2.items() if not v["pass"]]
    a("1. **A2 (secondary replication) fails** by its frozen rule in " + ", ".join(
        f"{k} (tier {a2[k]['tier']}: mean X_TN(10) {f(a2[k]['X_mean'],5)}, 95% CI {ci(a2[k]['X_ci95'],5)})" for k in fails)
      + (". In every failing cell the CI extends below −0.002, on the side where u agrees *better* than y (X < 0)" if all(a2[k]["X_ci95"][0] < -0.002 and a2[k]["X_ci95"][1] <= 0.002 for k in fails) else ". Direction mixed; see §5")
      + ("; u stays reliable on T_N(10) in each" if all(a2[k]["u@Y"] == "reliable" for k in fails) else "")
      + ". The rule's ±0.002 band is two-sided, so this is a failure of the replication expectation as written.")
    r250 = T["MOM_250000"]["Gs"]
    a(f"2. **A3 is 'anchor-sensitive' only through u^c at 250k**: uc@Y {r250['uc@Y']['final']} {r250['uc@Y']['x']}/96 at ε^Y_MOM = {f(r250['uc@Y']['eps'],5)} "
      f"vs uc@Pool {r250['uc@Pool']['final']} {r250['uc@Pool']['x']}/96 at ε^Pool = {f(r250['uc@Pool']['eps'],5)} (ratio {f(r250['uc@Pool']['eps']/r250['uc@Y']['eps'],3)}). "
      f"u itself is unreliable at ε^Y at both N ({T['MOM_25000']['Gs']['u@Y']['x']}/96, {r250['u@Y']['x']}/96) and not reliable at ε^Pool "
      f"({T['MOM_25000']['Gs']['u@Pool']['final']} {T['MOM_25000']['Gs']['u@Pool']['x']}/96, {r250['u@Pool']['final']} {r250['u@Pool']['x']}/96). By R-A1/A3 the status order puts 'anchor-sensitive' before 'holds', so A3 does not hold as a categorical claim.")
    bf = T["BAL-FLOW_250000"]["Gs"]
    a(f"3. **A7 reading (a) fails in both arms for different reasons**: SMALL-FLOW G_s is unreliable at 25k ({T['SMALL-FLOW_25000']['Gs']['u@Y']['x']}/96) and reliable at 250k ({T['SMALL-FLOW_250000']['Gs']['u@Y']['x']}/96); "
      f"BAL-FLOW G_s is unreliable at ε^Y at both N ({T['BAL-FLOW_25000']['Gs']['u@Y']['x']}/96, {bf['u@Y']['x']}/96) and anchor-sensitive at 250k (reliable {bf['u@Pool']['x']}/96 at ε^Pool, ratio {f(bf['u@Pool']['eps']/bf['u@Y']['eps'],3)}). Reading (b) holds in both (B_Gs " + f"{f(min(A['A7']['arms'][x]['cells'][N]['B_Gs_sigma']['mean'] for x in A['A7']['arms'] for N in NS),2)}–{f(max(A['A7']['arms'][x]['cells'][N]['B_Gs_sigma']['mean'] for x in A['A7']['arms'] for N in NS),2)} σ_y).")
    e_sg = E["eps"]["SGD-K_250000"]
    a(f"4. **SGD-K 250k anchors are vacuous.** One pilot pair (S2.2v11-pilot SGD-K N250000 r022/r023; the cell also carries the GB accuracy flag) has Δ_Gs(y) = {f(e_sg['Y']['Gs'],3)}, "
      f"so ε^Y_Gs = {f(e_sg['Y']['Gs'],3)} against ε^Y_q90 = {f(e_sg['Y_q90']['Gs'],4)} (1−CKA: {f(e_sg['Y']['dCKA'],3)} vs {f(e_sg['Y_q90']['dCKA'],4)}; ε^C likewise); "
      f"pilot homogeneity {f(E['homogeneity_sigma_units']['250000']['SGD-K']['Gs'],2)} σ on G_s. Every 'reliable' at ε^Y or ε^C in SGD-K 250k (e.g. u@Y G_s 32/32) is therefore uninformative. "
      "No primary endpoint uses SGD-K ladders (C5 uses δ_C only); SGD-K 250k was already excluded from the pooled anchor.")
    ex = C["C5"]["extrapolation_250k"]
    a(f"5. **C5 direction**: at the momentum-equivalent step, a plain Euler head drifts *more* than the momentum head (MOM/SGD-K median δ_C ratio "
      f"{f(C['C5']['ratios']['MOM/SGD-K_25000']['ratio'],3)} at 25k, {f(C['C5']['ratios']['MOM/SGD-K_250000']['ratio'],3)} at 250k; MOM / extrapolated plain step = {f(ex.get('MOM_over_plain'),3)} "
      f"{ci(ex.get('MOM_over_plain_ci95',[None,None]),3)}; Euler slope {f(ex.get('slope'),3)}, near the −1 expected from identity 2.2). The pre-agreed outcome is 'both causes stay'.")
    a(f"6. **I4 size**: FLOW I_Gs = {f(i4['25000|Gs|I']['mean_sigma'],2)} / {f(i4['250000|Gs|I']['mean_sigma'],2)} σ_y (25k / 250k) vs MOM I_Gs = "
      f"{f(A['A4']['cells']['MOM_25000']['Gs']['I']['sigma_units']['mean'],2)} / {f(A['A4']['cells']['MOM_250000']['Gs']['I']['sigma_units']['mean'],2)} σ_y. "
      "Outcome (i) rests on lower bounds > 0; (ii) fails because the upper 98.75% bounds of I_Gs and I_CKA (" + ", ".join(f(i4[N + '|' + q + '|I']['ci_sigma'][1], 2) for N in NS for q in ('Gs', 'dCKA'))
      + " σ_y) exceed the 0.5 σ_y margin. The FLOW imbalance effect is about one fifth of momentum's on G_s.")
    a("7. **Frozen short report formatting**: its C1/C3/C4a consequence line prints the whole C3(a) dict where the prereg sentence has 'raised ρ to X'; "
      f"X = BAL-MOM median ρ_end {f(C['C3']['a_BALMOM_rho']['25000']['median'],3)} (25k) and {f(C['C3']['a_BALMOM_rho']['250000']['median'],3)} (250k). No decision is affected.")
    a("8. 'h@Y' entries judge h at y's tolerance (listed by prereg §10.3); h's own decision is 'h@h' at ε^h.")
    a("")

    # ---- 1. primary
    a("## 1. Primary endpoints")
    a("")
    a("| Item | Tier | Rule | Numbers | Frozen-code verdict |")
    a("|---|---|---|---|---|")
    c3a, c3b = C["C3"]["a_BALMOM_rho"], C["C3"]["b_MOM_growth"]
    a("| C3(a) | R | BAL-MOM: lower 97.5% of median ρ_end > 0.05 and ≥ 90% runs ρ_end ≥ 0.02, each N | "
      + "; ".join(f"{int(N)//1000}k: median {f(c3a[N]['median'])}, lower {f(c3a[N]['lower_975'])}, frac≥0.02 {f(c3a[N]['frac_ge_0p02'],2)}, n {c3a[N]['n']}" for N in NS)
      + f" | {'PASS' if C['C3']['a_pass'] else 'FAIL'} |")
    a("| C3(b) | R | MOM: lower 97.5% of median growth ‖C_end−C₀‖/‖C₀‖ > 1, each N | "
      + "; ".join(f"{int(N)//1000}k: median {f(c3b[N]['median'],2)}, lower {f(c3b[N]['lower_975'],2)}, n {c3b[N]['n']}" for N in NS)
      + f" | {'PASS' if C['C3']['b_pass'] else 'FAIL'} |")
    c4 = C["C4a"]["MOM"]
    a("| C4a | R (KA6) | MOM: upper 97.5% of median μ_C < 0.5, each N | "
      + "; ".join(f"{int(N)//1000}k: median {f(c4[N]['median'])}, upper {f(c4[N]['upper_975'])}, n {c4[N]['n']}" for N in NS)
      + f" | {'PASS' if C['C4a']['pass'] else 'FAIL'} |")
    i3 = I["I3"]
    a(f"| I3 (F1) | 1 R + 7 H | mean I(init-MOM) − mean I(init-FLOW) > 0; 8 one-sided bootstrap tests, Holm 0.05 | "
      f"{i3['n_reject']}/8 rejected; smallest diff {f(min(v['diff'] for v in i3['tests'].values()))}, largest p {f(max(v['p_one_sided'] for v in i3['tests'].values()),5)} (§4) | "
      f"{i3['n_reject']}/8 Holm rejections |")
    i4 = I["I4"]["intervals"]
    a("| I4 | H, two-sided map | FLOW: (i) lower 98.75% bounds of I_Gs and I_M,Gs > 0 at both N; (ii) I_Gs, I_CKA CIs within ±0.5 σ_y at both N | "
      + "; ".join(f"{int(N)//1000}k: I_Gs {f(i4[N+'|Gs|I']['mean_sigma'],2)}σ {ci(i4[N+'|Gs|I']['ci_sigma'],2)}, I_M,Gs {f(i4[N+'|Gs|I_M']['mean_sigma'],2)}σ {ci(i4[N+'|Gs|I_M']['ci_sigma'],2)}" for N in NS)
      + f" | {I['I4']['outcome']} (within margin: {f(I['I4']['within_margin'])}) |")
    a1 = A["A1"]["cells"]
    a("| A1 | R, positive control | WD u reliable at ε^Y_WD on T_N(10), G_s, both N; RG; anchor flag | "
      + "; ".join(f"{k}: {v['u@Y']} {v['x']}/{v['n']} (Pool {v['u@Pool']})" for k, v in a1.items())
      + f" | {A['A1']['status']} |")
    a3 = A["A3"]["cells"]
    a("| A3 | R, negative control | MOM u and u^c unreliable on G_s at ε^Y_MOM, both N; RG; anchor flag | "
      + "; ".join(f"{int(N)//1000}k: u@Y {a3[N]['u@Y']} {T['MOM_'+N]['Gs']['u@Y']['x']}/96, uc@Y {a3[N]['uc@Y']} {T['MOM_'+N]['Gs']['uc@Y']['x']}/96, u@Pool {a3[N]['u@Pool']} {T['MOM_'+N]['Gs']['u@Pool']['x']}/96, uc@Pool {a3[N]['uc@Pool']} {T['MOM_'+N]['Gs']['uc@Pool']['x']}/96" for N in NS)
      + f" | {A['A3']['status']} |")
    a5 = A["A5"]["cells"]["MOM_250000"]
    a(f"| A5 | E | MOM 250k G_s: Δ(y)+M at ε^Y_a replaces the v3 '96 of 96' sentence | Δ(y)+M {a5['yM@Y']} {a5['yM_x']}/{a5['n']}; u@C {a5['u@C']}; M = {f(a5['M_sigma'],2)} σ_y | {a5['yM@Y']} |")
    for arm, v in A["A7"]["arms"].items():
        a(f"| A7 {arm} | H-open | (a) u reliable on G_s at both N and not anchor-sensitive; (b) lower 95% of B_Gs > 0.5 σ_y at both N | "
          + "; ".join(f"{int(N)//1000}k: G_s u@Y {v['cells'][N]['Gs']['u@Y']} {T[arm+'_'+N]['Gs']['u@Y']['x']}/96 (Pool {v['cells'][N]['Gs']['u@Pool']}, anchor-sens {f(v['cells'][N]['Gs']['anchor_sensitive'])}), 1−CKA u@Y {v['cells'][N]['dCKA']['u@Y']} {T[arm+'_'+N]['dCKA']['u@Y']['x']}/96, B_Gs {f(v['cells'][N]['B_Gs_sigma']['mean'],2)}σ {ci(v['cells'][N]['B_Gs_sigma']['ci'],2)}" for N in NS)
          + f" | (a) {f(v['reading_a'])}, (b) {f(v['reading_b'])} |")
    a(f"| A7 WD clause | – | WD B_Gs 95% CI within ±0.5 σ_y at both N | "
      + "; ".join(f"{int(N)//1000}k: B_Gs {f(A['A4']['cells']['WD_'+N]['Gs']['B']['sigma_units']['mean'],2)}σ {ci(A['A4']['cells']['WD_'+N]['Gs']['B']['sigma_units']['ci'],2)}" for N in NS)
      + f" | {f(A['A7']['WD_B_Gs_within_0.5sigma'])} |")
    a(f"| K (Corollary 3, prospective) | T checks; R K4b,c; H K5 | K1–K3 zero violations; K4b median bound/obs at k=10 in [5, 25]; K4c cert. fraction ≤ 0.10; K5 MOM imbalance-only exchange moves ≥ 0.01 of k=10 slots | "
      f"K1 {K['K1']['violations_pure']}+{K['K1']['violations_imb']} viol.; K2 {K['K2']['anchor_violations']} anchor viol., {K['K2']['pairs_bound_lt_obs']} pairs bound<obs; K3 {K['K3']['violations']} viol.; "
      f"K4b range {f(min(K['K4b']['cells'].values()),1)}–{f(max(K['K4b']['cells'].values()),1)}; K4c max {f(max(K['K4c']['cells'].values()))} ({max(K['K4c']['cells'], key=K['K4c']['cells'].get)}); "
      f"K5 MOM {f(K['K5']['cells']['MOM_25000']['obs_dN_k10_mean'])} / {f(K['K5']['cells']['MOM_250000']['obs_dN_k10_mean'])} | "
      f"K1 {'PASS' if K['K1']['pass'] else 'FAIL'}, K2 {'PASS' if K['K2']['pass'] else 'FAIL'}, K3 {'PASS' if K['K3']['pass'] else 'FAIL'}, K4b {'PASS' if K['K4b']['pass'] else 'FAIL'}, K4c {'PASS' if K['K4c']['pass'] else 'FAIL'}, K5 {'PASS' if K['K5'].get('pass_MOM') else 'FAIL'} |")
    a("")

    # ---- 2. checks
    a("## 2. Checks (a failure would mean an error)")
    a("")
    a("| Check | Result | Detail |")
    a("|---|---|---|")
    a(f"| C1 flow δ_C ≤ 2e-4 | {'PASS' if C['C1']['pass'] else 'FAIL'} | " + "; ".join(f"{k}: max {f(v['delta_C_max'])}, ok {v['n_ok']}/{v['n']}, viol {v['n_viol']}" for k, v in C["C1"]["cells"].items()) + " |")
    a(f"| C2 Euler identity ≤ 1e-8 | {'PASS' if C['C2']['pass'] else 'FAIL'} | " + "; ".join(f"{k}: max {f(v['max'])}, ok {v['n_ok']}/{v['n']}" for k, v in C["C2"]["cells"].items()) + " |")
    a(f"| C4 flow check median μ_C ≥ 0.99 | {'PASS' if C['C4a']['check_pass'] else 'FAIL'} | " + "; ".join(f"FLOW {int(N)//1000}k {f(v,7)}" for N, v in C["C4a"]["check_FLOW_median_mu_C"].items()) + " |")
    a(f"| I1 Lemma N1 (end) | {'PASS' if I['I1']['pass'] else 'FAIL'} | runs {I['I1']['n_runs']}, violations {I['I1']['n_viol']}, vacuous {I['I1']['n_vacuous']} |")
    a(f"| I2 Lemma N1 (flow arms) | {'PASS' if I['I2']['pass'] else 'FAIL'} | runs {I['I2']['n_runs']}, violations {I['I2']['n_viol']}, vacuous {I['I2']['n_vacuous']} |")
    a(f"| K1, K2, K3 | {'PASS' if K['K1']['pass'] and K['K2']['pass'] and K['K3']['pass'] else 'FAIL'} | see §7 |")
    a("")

    # ---- 3. C-block
    a("## 3. C-block (Remark rem:optimizers)")
    a("")
    a("**C4b [E]: `frac_own_closer`, two-sided 95% CP over main runs.** Reading by the frozen code: " + C["C4b"]["reading"] + ".")
    a("")
    a("| Cell | x/n | p̂ | 95% CP | mean d(own) | mean d(other) |")
    a("|---|---|---|---|---|---|")
    for k, v in C["C4b"]["cells"].items():
        a(f"| {k} | {v['x']}/{v['n']} | {f(v['p'],3)} | {ci(v['cp95'],3)} | {f(v['d_own_mean'],3)} | {f(v['d_other_mean'],3)} |")
    a("")
    a("**C5 [E]: drift δ_C, medians with 95% bootstrap CI.**")
    a("")
    a("| Cell | median δ_C | 95% CI | n |")
    a("|---|---|---|---|")
    for k, v in C["C5"]["medians"].items():
        a(f"| {k} | {f(v['median'])} | {ci(v['ci'])} | {v['n']} |")
    a("")
    a("| Ratio | value | 95% CI |")
    a("|---|---|---|")
    for k, v in C["C5"]["ratios"].items():
        if v:
            a(f"| {k} | {f(v['ratio'],3)} | {ci(v['ci95'],3)} |")
    ex = C["C5"]["extrapolation_250k"]
    a("")
    a(f"Extrapolation at 250k from SGD-K, KS-4, KS-16 (K = {ex.get('K')}): log-log slope {f(ex.get('slope'),3)} {ci(ex.get('slope_ci95',[None,None]),3)}; "
      f"predicted plain-step (K = 1) drift {f(ex.get('plain_step_drift_pred'))}; MOM / plain = {f(ex.get('MOM_over_plain'),3)} {ci(ex.get('MOM_over_plain_ci95',[None,None]),3)}; "
      f"momentum named main cause (ratio > 3 with CI excluding 3): {f(ex.get('momentum_named_main_cause'))}.")
    a("")
    a("**C6 [E]: MOM-BF16 vs MOM (fp64) at 250k.**")
    a("")
    a("| Quantity | median bf16 | median fp64 | relative diff | 95% CI of diff | sentence triggered |")
    a("|---|---|---|---|---|---|")
    for k, v in C["C6"].items():
        if isinstance(v, dict) and "median_bf16" in v:
            a(f"| {k} | {f(v['median_bf16'])} | {f(v['median_fp64'])} | {f(v['rel_diff'])} | {ci(v['diff_ci95'])} | {f(v['sentence_triggered'])} |")
    a("")

    # ---- 4. I-block
    a("## 4. I-block (balanced map vs imbalance channel)")
    a("")
    a("**I2: median D (estimate).** " + "; ".join(f"{k} {f(v,4)}" for k, v in I["I2"]["median_D"].items()) + ".")
    a("")
    a("**I3 (family F1), mean I(init-MOM) − mean I(init-FLOW), 10,000 resamples, Holm α = 0.05.**")
    a("")
    a("| Test | tier | diff (raw) | 95% CI | p (one-sided) | Holm threshold | reject | n_MOM / n_FLOW pairs |")
    a("|---|---|---|---|---|---|---|---|")
    for k, v in I["I3"]["tests"].items():
        a(f"| {k.replace('|', ' · ')} | {v['tier']} | {f(v['diff'],5)} | {ci(v['ci95'],5)} | {f(v['p_one_sided'],5)} | {f(v['holm']['holm_threshold'],5)} | {f(v['holm']['reject'])} | {v['n_a']} / {v['n_b']} |")
    a("")
    a(f"**I4 (FLOW), simultaneous {I['I4']['level']} percentile bootstrap CIs, margin δ_eq = {I['I4']['delta_eq_sigma']} σ_y.** Outcome: **{I['I4']['outcome']}**.")
    a("")
    a("| N, Q, part | mean raw | CI raw | mean σ_y | CI σ_y | σ_y | n |")
    a("|---|---|---|---|---|---|---|")
    for k, v in I["I4"]["intervals"].items():
        a(f"| {k.replace('|', ' · ')} | {f(v['mean_raw'],5)} | {ci(v['ci_raw'],5)} | {f(v['mean_sigma'],3)} | {ci(v['ci_sigma'],3)} | {f(v['sigma_y'],5)} | {v['n']} |")
    a("")
    a("For comparison (A4 means, σ_y units of each cell): " + "; ".join(
        f"{arm} {int(N)//1000}k I_Gs {f(A['A4']['cells'][arm+'_'+N]['Gs']['I']['sigma_units']['mean'],2)}σ, I_M,Gs {f(A['A4']['cells'][arm+'_'+N]['Gs']['I_M']['sigma_units']['mean'],2)}σ"
        for arm in ("MOM", "FLOW") for N in NS) + ".")
    a("")
    a("**I5 [E]: DiD [I(FLOW) − I(BAL-FLOW)] − [I(MOM) − I(BAL-MOM)].** " + "; ".join(
        f"{k} {f(v.get('did'),5)}" + (f" {ci(v['ci95'],5)}" if "ci95" in v else " (point)") for k, v in I["I5"].items() if isinstance(v, dict)) + ".")
    a("")
    a("**I6 [E, unadjusted]: Spearman(log D, ‖C‖_op/s).** " + "; ".join(
        f"{k} ρ = {f(v['spearman'],3)}, one-sided p = {f(v['p_one_sided'],3)}, n = {v['n']} (x = {v['x']})" for k, v in I["I6"].items() if isinstance(v, dict)) + ".")
    a("")

    # ---- 5. A-block secondary
    a("## 5. A-block: secondary replications and estimates")
    a("")
    a(f"**A2 [R for MOM, WD; H for flow arms]: wherever y is reliable, u not unreliable on T_N(10) and the 95% CI of mean X_TN(10) within ±0.002.** pass_R = {f(A['A2']['pass_R'])}, pass_H = {f(A['A2']['pass_H'])}.")
    a("")
    a("| Cell | tier | y reliable | u@Y | mean X_TN(10) | 95% CI | pass |")
    a("|---|---|---|---|---|---|---|")
    for k, v in A["A2"]["cells"].items():
        a(f"| {k} | {v['tier']} | {f(v['y_reliable'])} | {v['u@Y']} | {f(v['X_mean'],5)} | {ci(v['X_ci95'],5)} | {'PASS' if v['pass'] else 'FAIL'} |")
    a("")
    a8 = A["A8"]
    a(f"**A8 [R]:** MOM X_Gs(250k)/X_Gs(25k) = {f(a8.get('ratio'),3)} {ci(a8.get('ci95',[None,None]),3)}; rule [0.5, 1.5]: {'PASS' if a8.get('pass') else 'FAIL'}.")
    a("")
    a(f"**A9 [R]:** MOM Δ_E(u) < Δ_E(y) in " + "; ".join(f"{int(N)//1000}k {f(v['frac'],3)} of {v['n']}" for N, v in A["A9"]["cells"].items()) + f" pairs; rule ≥ 0.9: {'PASS' if A['A9']['pass'] else 'FAIL'}.")
    a("")
    a("**A10 [E]: mean s_N(10)(h) by arm.** " + " ".join(
        f"{int(N)//1000}k: " + ", ".join(f"{arm} {f(x,4)}" for arm, x in v["mean_sN10_h"].items()) + f" (max |diff| {f(v['max_abs_diff'],4)})." for N, v in A["A10"]["cells"].items()))
    a("")
    a("**A5 [E] (all ladder cells): u at ε^C and Δ(y)+M at ε^Y_a on G_s.**")
    a("")
    a("| Cell | u@C | Δ(y)+M @Y | x/n | M (σ_y) |")
    a("|---|---|---|---|---|")
    for k, v in A["A5"]["cells"].items():
        a(f"| {k} | {v['u@C']} | {v['yM@Y']} | {v['yM_x']}/{v['n']} | {f(v['M_sigma'],2)} |")
    a("")
    a("**A6 [E]: u at its own anchor ε^O; ratio ε^O/ε^Y.**")
    a("")
    a("| Cell | " + " | ".join(f"{QNAME[q]} u@O (ε^O/ε^Y)" for q in ("dN_k10", "Gs", "dCKA")) + " |")
    a("|---|---|---|---|")
    for k, v in A["A6"]["cells"].items():
        a(f"| {k} | " + " | ".join(f"{v[q]['u@O']} ({f(v[q]['epsO_over_epsY'],3)})" for q in ("dN_k10", "Gs", "dCKA")) + " |")
    a("")

    # ---- 6. A4
    a("## 6. A4 decompositions (preregistered endpoints)")
    a("")
    a("Means in σ_Q(y, cell) units with 95% bootstrap CIs; raw values, medians and share > 0 for all 8 R_cf rungs are in `render/a4_parts.csv`. "
      "X = Δ(u) − Δ(y) = S + M = B + I; S_bal + M_bal = B; I_S + I_M = I.")
    a("")
    rep = A["A4"].get("replication_MOM_250k", {})
    a(f"Replication expectations, MOM 250k G_s: S/X = {f(rep.get('S_over_X'),3)} (rule [0.5, 0.9]: {f(rep.get('S_over_X_in_0.5_0.9'))}); "
      f"M CI excludes 0: {f(rep.get('M_ci_excl_0'))}; I CI excludes 0: {f(rep.get('I_ci_excl_0'))}; B CI excludes 0: {f(rep.get('B_ci_excl_0'))}.")
    a("")
    for Q in ("Gs", "dCKA"):
        a(f"**{QNAME[Q]}**")
        a("")
        a("| Cell | σ_y (raw) | X | S | M | B | I | S_bal | M_bal | I_S | I_M | S/X | I/X |")
        a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for cell in LADDER_CELLS:
            d = A["A4"]["cells"][cell][Q]
            cellv = [f"{f(d[p]['sigma_units']['mean'],2)} {ci(d[p]['sigma_units']['ci'],2)}" for p in ("X", "S", "M", "B", "I", "S_bal", "M_bal", "I_S", "I_M")]
            a(f"| {cell} | {f(d['sigma_y'],5)} | " + " | ".join(cellv) + f" | {f(d.get('S_over_X'),2)} | {f(d.get('I_over_X'),2)} |")
        a("")
    a("**X on T_N(10), T_E, T_O (σ_y units, 95% CI)**")
    a("")
    a("| Cell | X T_N(10) | X T_E | X T_O |")
    a("|---|---|---|---|")
    for cell in LADDER_CELLS:
        d = A["A4"]["cells"][cell]
        a(f"| {cell} | " + " | ".join(f"{f(d[q]['X']['sigma_units']['mean'],2)} {ci(d[q]['X']['sigma_units']['ci'],2)}" for q in ("dN_k10", "dE", "dO")) + " |")
    a("")

    # ---- 7. K-block
    a("## 7. K-block (Corollary cor:gauge-neighbors, prospective)")
    a("")
    a("| Cell | pairs ok | κ_AB med | κ^bal med | D med | κ/κ^bal med | cert. frac k=10 | bound/obs k=10 | obs pure k=10 | obs imb k=10 [95% CI] | K6 excess k=10 | K4a band kept |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in LADDER_CELLS:
        s = K["cells"][cell]
        a(f"| {cell} | {s['n_ok']}/{s['n_pairs']} | {f(s['kappa_median'],3)} | {f(s['kappa_bal_median'],3)} | {f(s['D_median'],3)} | {f(s['kappa_over_kappa_bal_median'],3)} | "
          f"{f(s['pure_k10']['cert_frac_mean'])} | {f(s['pure_k10']['bound_over_obs_median'],2)} | {f(s['pure_k10']['obs_dN_mean'])} | "
          f"{f(s['imb_k10']['obs_dN_mean'])} {ci(s['imb_k10']['obs_dN_ci95'])} | {f(s['cross_k10_excess_mean'],5)} | "
          f"{s['pure_k10']['k4a_band_kept']}/{s['pure_k10']['k4a_band_n']} |")
    a("")
    a(f"Certified anchors with zero violations, summed over cells (pure exchange): k=1 {sum(K['cells'][c]['pure_k1']['certified_anchors_total'] for c in LADDER_CELLS)}, "
      f"k=10 {sum(K['cells'][c]['pure_k10']['certified_anchors_total'] for c in LADDER_CELLS)}, k=30 {sum(K['cells'][c]['pure_k30']['certified_anchors_total'] for c in LADDER_CELLS)}; "
      f"imbalance-only exchange: k=1 {sum(K['cells'][c]['imb_k1']['certified_anchors_total'] for c in LADDER_CELLS)}, k=10 {sum(K['cells'][c]['imb_k10']['certified_anchors_total'] for c in LADDER_CELLS)}, "
      f"k=30 {sum(K['cells'][c]['imb_k30']['certified_anchors_total'] for c in LADDER_CELLS)}. K6 pass_R {f(K['K6']['pass_R'])}, pass_H {f(K['K6']['pass_H'])}. "
      f"K4a is descriptive only and not in the paper (author decision 7).")
    a("")

    # ---- 8. pooled-anchor sensitivity
    a("## 8. Sensitivity: per-arm anchor ε^Y_a vs pooled anchor ε^Pool")
    a("")
    a("Final calls after RG. 'flag' = reliable at one anchor and unreliable at the other (§9).")
    a("")
    a("| Cell | Q | ε^Pool/ε^Y | y@Y | y@Pool | u@Y | u@Pool | u flag | uc@Y | uc@Pool | uc flag |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in LADDER_CELLS:
        for Q in PRIMQ:
            r = T[cell][Q]
            g = lambda k: f"{r[k]['final']} {r[k]['x']}/{r[k]['n']}" if k in r else "-"
            ratio = (r["u@Pool"]["eps"] / r["u@Y"]["eps"]) if r["u@Y"].get("eps") else float("nan")
            a(f"| {cell} | {QNAME[Q]} | {f(ratio,3)} | {g('y@Y')} | {g('y@Pool')} | {g('u@Y')} | {g('u@Pool')} | {f(r.get('u_anchor_sensitive'))} | "
              f"{g('uc@Y')} | {g('uc@Pool')} | {f(r.get('uc_anchor_sensitive'))} |")
    a("")

    # ---- 9. decision tables
    a("## 9. Decision tables, primary rungs (every entry)")
    a("")
    a("x/n, p̂, one-sided 95% CP [lower, upper], ε used, ε*_r (the x_reliable-th order statistic), decision → final call after RG. "
      "All 24 rungs and every entry are in `render/decision_entries.csv`.")
    a("")
    a("| Cell | Q | entry | x/n | p̂ | CP [lo, hi] | ε | ε*_r | decision | final |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for cell in LADDER_CELLS:
        for Q in PRIMQ:
            r = T[cell][Q]
            for k in ENTRY_ORDER:
                if k not in r:
                    continue
                v = r[k]
                a(f"| {cell} | {QNAME[Q]} | {k} | {v['x']}/{v['n']} | {f(v['p_hat'],3)} | [{f(v['cp_lo'],3)}, {f(v['cp_hi'],3)}] | {f(v['eps'],5)} | "
                  f"{f(v.get('eps_star_r'),5)} | {v['decision']} | {v['final']} |")
    a("")

    # ---- 10. p_hat curves
    a("## 10. p̂(ε) curves (selected points; full curves on ε/ε_anchor ∈ [0.80, 1.25] step 0.01 in `render/phat_curves.csv`)")
    a("")
    pts = (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25)
    a("| Cell | Q | entry | " + " | ".join(f"×{p}" for p in pts) + " |")
    a("|---|---|---|" + "---|" * len(pts))
    for cell in LADDER_CELLS:
        for Q in ("Gs", "dCKA"):
            for k in ("y@Y", "u@Y", "uc@Y", "ubal@Y", "u@Pool", "uc@Pool"):
                v = T[cell][Q].get(k)
                if not v or not v.get("p_hat_curve"):
                    continue
                cm = {round(r, 2): p for r, p in v["p_hat_curve"]}
                a(f"| {cell} | {QNAME[Q]} | {k} | " + " | ".join(f(cm.get(p), 3) for p in pts) + " |")
    a("")

    # ---- 11. anchors
    a("## 11. Frozen anchors (pilot pairs at the same N), primary rungs")
    a("")
    a("| Cell | Q | ε^Y | ε^Y_q90 | ε^O | ε^C | ε^h | ε^Pool | pilot pairs |")
    a("|---|---|---|---|---|---|---|---|---|")
    for cell in LADDER_CELLS:
        e = E["eps"][cell]
        N = cell.split("_")[1]
        for Q in PRIMQ:
            a(f"| {cell} | {QNAME[Q]} | {f(e['Y'].get(Q),5)} | {f(e['Y_q90'].get(Q),5)} | {f(e['O'].get(Q),5)} | {f(e['C'].get(Q),5)} | "
              f"{f(e['h'].get(Q),5)} | {f(E['pool'][N]['Pool'].get(Q),5)} | {e['Y_n'].get(Q)} |")
    a("")
    a("Pilot homogeneity diagnostic (mean Δ_Q(y) per arm in pooled-SD units; reported, not a gate): " + "; ".join(
        f"{int(N)//1000}k " + ", ".join(f"{arm} G_s {f(v.get('Gs'),2)} / CKA {f(v.get('dCKA'),2)}" for arm, v in d.items())
        for N, d in E["homogeneity_sigma_units"].items()) + ".")
    a("")

    # ---- 12. covariates
    a("## 12. Covariates per cell (medians over ok main runs; descriptive aggregation of per-run fields)")
    a("")
    cols = ("acc_bank", "ce_train", "ce_hold", "gen_gap", "cond_W0", "cond_W", "cond_M", "s", "rho_end", "delta_C", "mu_C",
            "disp_P_median", "disp_W_median", "head_substeps_total", "t_total_s")
    a("| Cell | ok/n | " + " | ".join(cols) + " |")
    a("|---|---|" + "---|" * len(cols))
    for cell, d in covsum.items():
        a(f"| {cell} | {d['n_ok']}/{d['n']} | " + " | ".join(f(d[c], 4) for c in cols) + " |")
    a("")

    # ---- 13. consequences
    a("## 13. Pre-agreed consequences as evaluated by the frozen code (verbatim)")
    a("")
    for c in S["consequences"]:
        a(f"- {c}")
    a("")
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    for o in outs:
        os.chmod(o, 0o444)
    print({os.path.relpath(o, ROOT): sha(o) for o in outs})


if __name__ == "__main__":
    main()
