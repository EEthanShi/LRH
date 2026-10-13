#!/usr/bin/env python
"""Pythia axis: the section 0 known-offset table and band fit and the section 12 feasibility tables of the plan,  # PYT-AXIS
recomputed on ds2213 from KNOWN files only (step S0b). EXPLORATORY design calculations, not citable as results.
PYT-AXIS copy of analysis/s12_layermax_openllama/feasibility_ol.py (sha256 3faad6b1...), rewritten at S0b by role  # PYT-AXIS
pyt-code-infra (2026-10-10) for analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md (sha256 6de89603...); changed  # PYT-AXIS
lines carry PYT-AXIS; diff: feasibility_pyt.py.diff. The OpenLLaMA tables (J, scenario, conditional size, ARH  # PYT-AXIS
pooled reading) do not exist on this axis and are dropped.  # PYT-AXIS

Inputs (frozen, read-only; no Pythia file exists at S0b and none is read):  # PYT-AXIS
  results/S1_2_layermax_llama/partB/partB_features.json      PR* of 10 known LMs and the 17 ViTs  # PYT-AXIS
  results/S1_2_layermax_openllama/partB/partB_features_ol.json  PR* of open_llama_13b  # PYT-AXIS
  the 187 known pair files (11 LMs x 17 ViTs, cka_biased, full) of results/S1_2_layermax/pairs/,  # PYT-AXIS
  results/S1_2_layermax_llama/early/pairs/ and results/S1_2_layermax_openllama/early/pairs/ (both pre)  # PYT-AXIS
UNGATED: no cross-model score involving a Pythia model.  # PYT-AXIS

Tables:  # PYT-AXIS
  (a) plan section 0: per known LM PR*, L_stored, median offset c_alpha - Bbar and its range; the 11-LM OLS fit of  # PYT-AXIS
      the per-LM median on 1/sqrt(PR*) (a, b, residual sd, correlation, max |pair - fit|, PR* range) for both pre,  # PYT-AXIS
      against the frozen constants of plan section 9 B6 ("a difference above 1e-6 is logged; the frozen constants  # PYT-AXIS
      apply regardless"); the critic's 10-LM fit (without open_llama_13b); null mean - Bbar and null sd ranges;  # PYT-AXIS
      the half-widths h (max |pair - fit| + about 2 Monte Carlo sd of the 476th order statistic);  # PYT-AXIS
  (b) plan section 12: attainable p; the exact p_perm of the stylised calibrated-rank patterns (every ViT with the  # PYT-AXIS
      same pattern, raw ranks 1..7), e.g. top pair swapped 2520/5040, top-5 reversal 444/5040;  # PYT-AXIS
  (c) resources: layer pairs, feature bytes, disk, RAM, the axis GPU ledger.  # PYT-AXIS
Not ported: the designer's scenario model (design/scen.py, random replicates) and the near-chance rows  # PYT-AXIS
(freeze/scen_nearchance.py); they remain exploratory design files of the plan.  # PYT-AXIS

Outputs: results/S1_2_layermax_pythia/feasibility/{feasibility_pyt.json, FEASIBILITY_PYT.md}  # PYT-AXIS
         (S0b test mode PYT_S0B_TEST=1: <guard.SCRATCH>/PB/feasibility/); never overwritten (no-delete rule)  # PYT-AXIS

RUN (S0b, once; CPU, 2 threads, about a minute):  # PYT-AXIS
  cd ~/hrh_v2_stage1_20261008
  analysis/s12_layermax_pythia/launch_pyt.sh feasibility_pyt 2 -- venv/bin/python analysis/s12_layermax_pythia/feasibility_pyt.py  # PYT-AXIS
"""
from __future__ import annotations

import os

_TH = os.environ.get("PYT_THREADS", "1")  # PYT-AXIS
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _TH

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_pythia")  # PYT-AXIS
sys.path.insert(0, PROJ)
sys.path.insert(0, AXIS)

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import guard  # noqa: E402  (test mode only)
import partb_pyt as PB  # noqa: E402  # PYT-AXIS: constants and known_offsets() only (main() not run)

RES = os.path.join(PROJ, "results")
OUT_REAL = os.path.join(RES, "S1_2_layermax_pythia", "feasibility")  # PYT-AXIS
PRES = ("prh_clip+l2", "l2")
TOL = 1e-9
PLAN0 = {  # PYT-AXIS: plan section 0 table (prh_clip+l2): PR*, L_stored, median offset, [min, max]
    "bigscience__bloomz-560m": (133.92, 25, 0.002190, 0.001723, 0.002802),  # PYT-AXIS
    "bigscience__bloomz-1b1": (124.50, 25, 0.002228, 0.002033, 0.002809),  # PYT-AXIS
    "bigscience__bloomz-1b7": (242.37, 25, 0.001945, 0.001734, 0.002133),  # PYT-AXIS
    "bigscience__bloomz-3b": (248.17, 31, 0.001898, 0.001710, 0.002263),  # PYT-AXIS
    "bigscience__bloomz-7b1": (249.33, 31, 0.001939, 0.001699, 0.002263),  # PYT-AXIS
    "openlm-research__open_llama_3b": (201.89, 27, 0.001900, 0.001691, 0.002114),  # PYT-AXIS
    "openlm-research__open_llama_7b": (228.57, 33, 0.001926, 0.001617, 0.002134),  # PYT-AXIS
    "openlm-research__open_llama_13b": (261.01, 41, 0.001884, 0.001644, 0.002257),  # PYT-AXIS
    "huggyllama__llama-7b": (257.33, 33, 0.001841, 0.001612, 0.002135),  # PYT-AXIS
    "huggyllama__llama-13b": (308.57, 41, 0.001818, 0.001554, 0.002124),  # PYT-AXIS
    "huggyllama__llama-30b": (359.22, 61, 0.001712, 0.001548, 0.001901),  # PYT-AXIS
}  # PYT-AXIS
PLAN_FIT = {  # PYT-AXIS: plan section 0 fits (11 LMs) and the critic's 10-LM fit; ranges over the 187 pairs
    "prh_clip+l2": {"a": 0.001069, "b": 0.012920, "resid_sd": 0.000040, "corr": 0.969, "max_pair_minus_fit": 0.000617,  # PYT-AXIS
                    "PR_min": 124.5, "PR_max": 359.2, "a10": 0.001062, "b10": 0.012993,  # PYT-AXIS
                    "nm_lo": -0.00010, "nm_hi": 0.00046, "sd_lo": 0.00096, "sd_hi": 0.00129},  # PYT-AXIS
    "l2": {"a": 0.001551, "b": 0.006843, "resid_sd": 0.000049, "corr": 0.872, "max_pair_minus_fit": 0.001530,  # PYT-AXIS
           "PR_min": 79.6, "PR_max": 180.0, "a10": 0.001536, "b10": 0.006991,  # PYT-AXIS
           "nm_lo": -0.00011, "nm_hi": 0.00022, "sd_lo": 0.00102, "sd_hi": 0.00185},  # PYT-AXIS
}  # PYT-AXIS
PATTERNS = {  # PYT-AXIS: plan section 12 (critic/pattern_p.py): calibrated rank pattern of sizes 1..7 -> (dTr, count/5040)
    "top pair swapped": ([1, 2, 3, 4, 5, 7, 6], -0.036, 2520),  # PYT-AXIS
    "top 3 reversed": ([1, 2, 3, 4, 7, 6, 5], -0.143, 1800),  # PYT-AXIS
    "top 4 reversed": ([1, 2, 3, 7, 6, 5, 4], -0.357, 966),  # PYT-AXIS
    "top 5 reversed": ([1, 2, 7, 6, 5, 4, 3], -0.714, 444),  # PYT-AXIS
    "6.9B drops to rank 4": ([1, 2, 3, 5, 6, 7, 4], -0.214, 1440),  # PYT-AXIS
    "6.9B drops to rank 1": ([2, 3, 4, 5, 6, 7, 1], -0.750, 720),  # PYT-AXIS
    "three disjoint adjacent swaps (1-2, 3-4, 5-6)": ([2, 1, 4, 3, 6, 5, 7], -0.107, 1692),  # PYT-AXIS
    "full reversal": ([7, 6, 5, 4, 3, 2, 1], -2.000, 1),  # PYT-AXIS
}  # PYT-AXIS


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def section0(LLPB: dict, OLPB: dict, inputs: dict) -> dict:  # PYT-AXIS
    out = {}  # PYT-AXIS
    for pre in PRES:  # PYT-AXIS
        ro = PB.known_offsets(pre, LLPB, OLPB, inputs)  # PYT-AXIS
        per = ro["per_lm"]  # PYT-AXIS
        x = np.array([1 / math.sqrt(per[k]["PR_star"]) for k in per])  # PYT-AXIS
        y = np.array([per[k]["median_offset"] for k in per])  # PYT-AXIS
        A = np.column_stack([np.ones_like(x), x])  # PYT-AXIS
        c, *_ = np.linalg.lstsq(A, y, rcond=None)  # PYT-AXIS
        res = y - A @ c  # PYT-AXIS
        keep10 = np.array([k != "openlm-research__open_llama_13b" for k in per])  # PYT-AXIS
        c10, *_ = np.linalg.lstsq(A[keep10], y[keep10], rcond=None)  # PYT-AXIS
        out[pre] = {"per_lm": per, "fit11": {"a": float(c[0]), "b": float(c[1]),  # PYT-AXIS
                                             "resid_sd_ddof0": float(np.std(res)), "resid_sd_ddof1": float(np.std(res, ddof=1)),  # PYT-AXIS
                                             "resid_sd_ddof2": float(np.std(res, ddof=2)), "corr": float(np.corrcoef(x, y)[0, 1]),  # PYT-AXIS
                                             "max_pair_minus_fit": ro["refit"]["max_abs_pair_minus_fit"],  # PYT-AXIS
                                             "PR_min": ro["refit"]["PR_range"][0], "PR_max": ro["refit"]["PR_range"][1]},  # PYT-AXIS
                    "fit10_critic": {"a": float(c10[0]), "b": float(c10[1])},  # PYT-AXIS
                    "frozen": {"a": PB.BAND[pre]["a"], "b": PB.BAND[pre]["b"], "h": PB.BAND[pre]["h"]},  # PYT-AXIS
                    "abs_diff_a": abs(float(c[0]) - PB.BAND[pre]["a"]), "abs_diff_b": abs(float(c[1]) - PB.BAND[pre]["b"]),  # PYT-AXIS
                    "n_in_band_187": ro["n_in_band"], "null_mean_minus_Bbar_range": ro["null_mean_minus_Bbar_range"],  # PYT-AXIS
                    "null_sd_range": ro["null_sd_range"],  # PYT-AXIS
                    "h_note": "h = max|pair - fit| + about 2 MC sd of the 476th order statistic, rounded up (plan B6); "  # PYT-AXIS
                              "the MC sd term is reported as the plan's 0.0001 and not re-derived here",  # PYT-AXIS
                    "h_check_max_plus_0.0001": ro["refit"]["max_abs_pair_minus_fit"] + 0.0001}  # PYT-AXIS
    return out  # PYT-AXIS


def rho(rank_size, ranks) -> float:  # PYT-AXIS: Spearman of two rank vectors (no ties)
    n = len(ranks)  # PYT-AXIS
    d2 = sum((a - b) ** 2 for a, b in zip(rank_size, ranks))  # PYT-AXIS
    return 1 - 6 * d2 / (n * (n * n - 1))  # PYT-AXIS


def pattern_table() -> dict:  # PYT-AXIS: plan section 12, exact over the 5040 relabellings
    out = {}  # PYT-AXIS
    ident = list(range(1, 8))  # PYT-AXIS
    perms = list(itertools.permutations(ident))  # PYT-AXIS
    for name, (cal, dtr_plan, cnt_plan) in PATTERNS.items():  # PYT-AXIS
        def stat(sig):  # PYT-AXIS: Tr_cal(sigma) - Tr_raw(sigma); every ViT identical, so the mean is one value
            return rho(sig, cal) - rho(sig, ident)  # PYT-AXIS
        d0 = stat(ident)  # PYT-AXIS
        cnt = sum(1 for sig in perms if stat(sig) <= d0 + TOL)  # PYT-AXIS
        out[name] = {"calibrated_ranks": cal, "dTr": d0, "count": cnt, "p_perm": cnt / 5040,  # PYT-AXIS
                     "plan_dTr": dtr_plan, "plan_count": cnt_plan,  # PYT-AXIS
                     "equal_plan": bool(cnt == cnt_plan and round(d0, 3) == round(dtr_plan, 3))}  # PYT-AXIS
    return out  # PYT-AXIS


def attainable() -> dict:  # PYT-AXIS: plan section 12
    return {"p_perm_min_7": 1 / 5040, "p_boot_min": 1 / 2001, "p_H1_min": max(1 / 5040, 1 / 2001),  # PYT-AXIS
            "holm_adjusted_min": 3 / 2001, "loo_min": 1 / 720, "five_size_min": 1 / 120}  # PYT-AXIS


def resources() -> dict:  # PYT-AXIS
    du = shutil.disk_usage(PROJ)
    mem = {}
    try:
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    led = os.path.join(RES, "S1_2_layermax_pythia", "logs", "gpu_ledger.tsv")  # PYT-AXIS
    tot = 0.0
    if os.path.isfile(led):
        for i, ln in enumerate(open(led)):
            if i and ln.strip():
                tot += float(ln.split("\t")[4])
    LB = 352  # PYT-AXIS: ViT layers summed over the 17 ViTs (plan section 1)
    Ls = dict(zip(PB.SIZES, PB.L_STORED))  # PYT-AXIS
    d = {"70m": 512, "160m": 768, "410m": 1024, "1b": 2048, "1.4b": 2048, "2.8b": 2560, "6.9b": 4096}  # PYT-AXIS
    return {"layer_pairs": {s: Ls[s] * LB for s in PB.SIZES}, "layer_pairs_total": sum(Ls[s] * LB for s in PB.SIZES),  # PYT-AXIS
            "features_bytes": sum(1024 * Ls[s] * d[s] * 4 for s in PB.SIZES),  # PYT-AXIS
            "disk_free_bytes": du.free, "mem_total_bytes": mem.get("MemTotal"), "mem_available_bytes": mem.get("MemAvailable"),
            "gpu_ledger_minutes": tot, "gpu_budget_minutes": 60}


def compare(res) -> list:  # PYT-AXIS
    rows = []

    def add(name, plan, server, nd):
        ok = (round(server, nd) == round(plan, nd)) if isinstance(plan, float) else (server == plan)
        rows.append({"item": name, "plan": plan, "server": server, "equal_at_plan_precision": bool(ok)})
    for pre in PRES:  # PYT-AXIS
        s0 = res["section0"][pre]  # PYT-AXIS
        pf = PLAN_FIT[pre]  # PYT-AXIS
        f = s0["fit11"]  # PYT-AXIS
        add(f"{pre} fit a (11 LMs)", pf["a"], f["a"], 6)  # PYT-AXIS
        add(f"{pre} fit b (11 LMs)", pf["b"], f["b"], 6)  # PYT-AXIS
        add(f"{pre} fit a within 1e-6 of frozen", True, bool(s0["abs_diff_a"] <= 1e-6), 0)  # PYT-AXIS
        add(f"{pre} fit b within 1e-6 of frozen", True, bool(s0["abs_diff_b"] <= 1e-6), 0)  # PYT-AXIS
        add(f"{pre} residual sd (ddof 2)", pf["resid_sd"], f["resid_sd_ddof2"], 6)  # PYT-AXIS
        add(f"{pre} correlation", pf["corr"], f["corr"], 3)  # PYT-AXIS
        add(f"{pre} max |pair - fit|", pf["max_pair_minus_fit"], f["max_pair_minus_fit"], 6)  # PYT-AXIS
        add(f"{pre} PR* min", pf["PR_min"], f["PR_min"], 1)  # PYT-AXIS
        add(f"{pre} PR* max", pf["PR_max"], f["PR_max"], 1)  # PYT-AXIS
        add(f"{pre} critic 10-LM a", pf["a10"], s0["fit10_critic"]["a"], 6)  # PYT-AXIS
        add(f"{pre} critic 10-LM b", pf["b10"], s0["fit10_critic"]["b"], 6)  # PYT-AXIS
        add(f"{pre} null mean - Bbar min", pf["nm_lo"], s0["null_mean_minus_Bbar_range"][0], 5)  # PYT-AXIS
        add(f"{pre} null mean - Bbar max", pf["nm_hi"], s0["null_mean_minus_Bbar_range"][1], 5)  # PYT-AXIS
        add(f"{pre} null sd min", pf["sd_lo"], s0["null_sd_range"][0], 5)  # PYT-AXIS
        add(f"{pre} null sd max", pf["sd_hi"], s0["null_sd_range"][1], 5)  # PYT-AXIS
        add(f"{pre} known pairs inside the PR*-indexed band", 187, s0["n_in_band_187"], 0)  # PYT-AXIS
    s0 = res["section0"]["prh_clip+l2"]  # PYT-AXIS
    for lm, (pr, L, med, lo, hi) in PLAN0.items():  # PYT-AXIS
        d = s0["per_lm"][lm]  # PYT-AXIS
        add(f"s0 {lm} PR*", pr, d["PR_star"], 2)  # PYT-AXIS
        add(f"s0 {lm} L_stored", L, d["L_stored"], 0)  # PYT-AXIS
        add(f"s0 {lm} median offset", med, d["median_offset"], 6)  # PYT-AXIS
        add(f"s0 {lm} offset min", lo, d["min"], 6)  # PYT-AXIS
        add(f"s0 {lm} offset max", hi, d["max"], 6)  # PYT-AXIS
    for name, d in res["patterns"].items():  # PYT-AXIS
        add(f"s12 pattern '{name}' count/5040", d["plan_count"], d["count"], 0)  # PYT-AXIS
        add(f"s12 pattern '{name}' dTr", d["plan_dTr"], d["dTr"], 3)  # PYT-AXIS
    add("s1 layer pairs total", 53856, res["resources"]["layer_pairs_total"], 0)  # PYT-AXIS
    add("s2 primary feature bytes", 1412431872, res["resources"]["features_bytes"], 0)  # PYT-AXIS
    return rows


def fmt_md(res, cmp_rows) -> list[str]:  # PYT-AXIS
    L = ["# Pythia axis: feasibility tables recomputed on ds2213 (S0b; EXPLORATORY design calculations)", "",  # PYT-AXIS
         f"Written by `analysis/s12_layermax_pythia/feasibility_pyt.py` at {res['meta']['finished_utc']} from known "  # PYT-AXIS
         f"frozen files only ({res['meta']['n_inputs']} files; sha256 in `feasibility_pyt.json` → `meta.inputs`). "  # PYT-AXIS
         "Rules, bands and thresholds are numeric constants of the plan and do not move (plan §9 B6: the frozen constants "  # PYT-AXIS
         "apply regardless). Where a server number differs from the plan's design number, the difference is listed.", "",  # PYT-AXIS
         "## Plan vs server", "", "| item | plan | server | equal at plan precision |", "|---|---|---|---|"]
    for r in cmp_rows:
        sv = r["server"]
        sv = f"{sv:.6g}" if isinstance(sv, float) else str(sv)  # PYT-AXIS
        L.append(f"| {r['item']} | {r['plan']} | {sv} | {'yes' if r['equal_at_plan_precision'] else '**NO**'} |")
    n_bad = sum(not r["equal_at_plan_precision"] for r in cmp_rows)
    L += ["", f"{len(cmp_rows)} items compared, {n_bad} differ at the plan's printed precision.", ""]
    for pre in PRES:  # PYT-AXIS
        s0 = res["section0"][pre]  # PYT-AXIS
        f = s0["fit11"]  # PYT-AXIS
        L += [f"## Band fit, {pre}", "",  # PYT-AXIS
              f"- 11 LMs: {f['a']:.6f} + {f['b']:.6f}/√PR* (frozen {s0['frozen']['a']} + {s0['frozen']['b']}/√PR*; |Δa| "  # PYT-AXIS
              f"{s0['abs_diff_a']:.2e}, |Δb| {s0['abs_diff_b']:.2e}{'; DIFFERENCE ABOVE 1e-6 LOGGED' if max(s0['abs_diff_a'], s0['abs_diff_b']) > 1e-6 else ''}); "  # PYT-AXIS
              f"residual sd {f['resid_sd_ddof2']:.6f} (ddof 2), corr {f['corr']:.4f}, max |pair − fit| {f['max_pair_minus_fit']:.6f}, "  # PYT-AXIS
              f"PR* {f['PR_min']:.1f}–{f['PR_max']:.1f}; critic 10-LM {s0['fit10_critic']['a']:.6f} + {s0['fit10_critic']['b']:.6f}/√PR*.",  # PYT-AXIS
              f"- Known pairs inside f(PR*) ± {s0['frozen']['h']}: {s0['n_in_band_187']}/187; null mean − B̄ in "  # PYT-AXIS
              f"[{s0['null_mean_minus_Bbar_range'][0]:+.5f}, {s0['null_mean_minus_Bbar_range'][1]:+.5f}]; null sd in "  # PYT-AXIS
              f"[{s0['null_sd_range'][0]:.5f}, {s0['null_sd_range'][1]:.5f}].", ""]  # PYT-AXIS
    L += ["## Exact p_perm of the stylised patterns (plan §12)", ""]  # PYT-AXIS
    for name, d in res["patterns"].items():  # PYT-AXIS
        L.append(f"- {name}: ΔTr {d['dTr']:+.3f}, p_perm {d['count']}/5040 (plan {d['plan_count']}/5040).")  # PYT-AXIS
    a = res["attainable_p"]  # PYT-AXIS
    r = res["resources"]
    L += ["", "## Attainable p and resources", "",  # PYT-AXIS
          f"- p_perm ≥ {a['p_perm_min_7']:.2e}; p_mag and TOST ≥ {a['p_boot_min']:.2e}; Holm-adjusted ≥ {a['holm_adjusted_min']:.2e}; "  # PYT-AXIS
          f"leave-one-size-out ≥ {a['loo_min']:.2e}; five sizes ≥ {a['five_size_min']:.2e}.",  # PYT-AXIS
          f"- Layer pairs {r['layer_pairs_total']}; primary features {r['features_bytes'] / 1e9:.3f} GB; disk free "  # PYT-AXIS
          f"{r['disk_free_bytes'] / 1e9:.1f} GB; RAM available {(r['mem_available_bytes'] or 0) / 1e9:.1f} GB; GPU ledger "  # PYT-AXIS
          f"{r['gpu_ledger_minutes']:.3f} / 60 min."]  # PYT-AXIS
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    test = guard.test_mode()
    out_dir = os.path.join(guard.SCRATCH, "PB", "feasibility") if test else OUT_REAL
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    outs = [os.path.join(out_dir, "feasibility_pyt.json"), os.path.join(out_dir, "FEASIBILITY_PYT.md")]  # PYT-AXIS
    for p in outs:  # PYT-AXIS: no-delete rule
        if os.path.exists(p):  # PYT-AXIS
            if not test:  # PYT-AXIS
                raise SystemExit(f"{p} exists (no-delete rule; feasibility runs once)")  # PYT-AXIS
            os.replace(p, p + ".prev-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))  # PYT-AXIS
    for p_, want in ((PB.LLAMA_PARTB, PB.LLAMA_PARTB_SHA), (PB.OL_PARTB, PB.OL_PARTB_SHA)):  # PYT-AXIS
        if sha256_file(p_) != want:  # PYT-AXIS
            raise SystemExit(f"frozen input changed: {p_}")  # PYT-AXIS
    inputs: dict = {os.path.relpath(PB.LLAMA_PARTB, PROJ): None, os.path.relpath(PB.OL_PARTB, PROJ): None}  # PYT-AXIS
    LLPB, OLPB = json.load(open(PB.LLAMA_PARTB)), json.load(open(PB.OL_PARTB))  # PYT-AXIS
    res = {"label": "EXPLORATORY design calculations from known files only (plan sections 0 and 12); not citable results"}  # PYT-AXIS
    res["section0"] = section0(LLPB, OLPB, inputs)  # PYT-AXIS
    print(f"[{time.time() - t0:5.1f}s] section 0 fits", flush=True)  # PYT-AXIS
    res["patterns"] = pattern_table()  # PYT-AXIS
    res["attainable_p"] = attainable()  # PYT-AXIS
    res["resources"] = resources()  # PYT-AXIS
    cmp_rows = compare(res)
    res["plan_vs_server"] = cmp_rows
    res["meta"] = {"script_sha256": sha256_file(os.path.abspath(__file__)), "started_utc": started,
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "threads": _TH,
                   "test_mode": test, "n_inputs": len(inputs),
                   "inputs": {p: sha256_file(os.path.join(PROJ, p)) for p in sorted(inputs)}}
    os.makedirs(out_dir, exist_ok=True)
    with open(outs[0], "w") as f:  # PYT-AXIS
        json.dump(res, f, indent=1, default=float)
    with open(outs[1], "w") as f:  # PYT-AXIS
        f.write("\n".join(fmt_md(res, cmp_rows)) + "\n")
    n_bad = sum(not r["equal_at_plan_precision"] for r in cmp_rows)
    for r in cmp_rows:
        if not r["equal_at_plan_precision"]:
            print(f"DIFF {r['item']}: plan {r['plan']} server {r['server']}", flush=True)
    print(f"[{time.time() - t0:5.1f}s] wrote {outs[0]}, {outs[1]}; {len(cmp_rows)} plan items compared, "  # PYT-AXIS
          f"{n_bad} differ at plan precision", flush=True)  # PYT-AXIS


if __name__ == "__main__":
    main()
