"""PYT-AXIS (new file, implementation E, S0b sub-role pyt-code-E): plan V-P8, the generalisation check of E's
aggregator (analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md section 11, step S0b, CPU). GATED (lm_core.gate():
in the S0b test mode it needs no Part B; otherwise the Part B gate).

Reads only frozen, non-Pythia files; computes nothing about any Pythia model.

(a) Legacy mode, n = 3, on the frozen OpenLLaMA pair files of implementation E
    (results/S1_2_layermax_openllama/early/pairs/<pre>/, the 3B/7B reruns and 13B) with the frozen OpenLLaMA Part B
    (partB_features_ol.json) and the 11-LM records of E's sources (for PO4|arh): aggregate_pyt.ol_legacy_compute()
    runs the generalised engine (rho_table / tr_series / diff_series, 3! relabellings, per-score exclusion) under the
    OpenLLaMA section A-C rules. Every key of sections A, B, C of the OpenLLaMA verdict contract
    (analysis/s12_layermax_openllama/verdict_keys.py, read-only import) is compared with the E verdicts embedded in
    results/S1_2_layermax_openllama/layermax_openllama_summary.json (and, for information, with the embedded I
    verdicts): strings, booleans and ints exact; floats |a - b| <= 1e-12 max(|a|, |b|) + 1e-15; null = null.
    Pass = 0 mismatches and 0 missing keys.
(b) Common-ViT-set mode, n = 3, same inputs: the keys whose value changes against (a) are listed, each with the reason:
    the ViTs valid under one of the two scores only (a ViT constant or non-finite under one score). Pass = every
    changed key is explained this way.
(c) Legacy mode, n = 5, the frozen BLOOMZ pair files (results/S1_2_layermax/pairs/<pre>/): for every pre, metric,
    subset of the frozen summary and score (s_obs = raw, s_cal, uncensored = unc), Tr, the exact one-sided and
    two-sided p over 5! = 120 relabellings and the count of excluded ViTs are compared with
    results/S1_2_layermax/layermax_summary.json (sha256 0dddd17a...; per_pre.<pre>.trends.<metric>.<subset|score>.tr_lm):
    |a - b| <= 1e-12 (absolute, plus 1e-12 relative). Pass = 0 mismatches.

Run (S0b, CPU, <= 4 threads):
  cd ~/hrh_v2_stage1_20261008 && PYT_S0B_TEST=1 analysis/s12_layermax_pythia/launch_pyt.sh pyt_E_vp8 2 -- \\
      venv/bin/python analysis/s12_layermax_pythia/early/vl8_generalise.py
Output: <RES>/validation/VP8_generalisation.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402

OL_AXIS = os.path.join(LC.PROJ, "analysis", "s12_layermax_openllama")
LAYERMAX_SUMMARY = os.path.join(LC.FROZEN_RES, "layermax_summary.json")
LAYERMAX_SUMMARY_SHA = "0dddd17a588dda2be47b19ec30b182f61f01dcfb5f791f237b39bf421d2c5613"
OL_MERGED_SHA = "2777dadf287dfc02433eb07e84cefad0ec67381258a199b9d306b6c7d32aebaa"
OL_PARTB_SHA = "e1104bd71e5eff9ced65ed402a61c187590bce5dee306c25ee2f64b18fdd7e51"
RTOL, ATOL = 1e-12, 1e-15


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return math.isnan(fa) and math.isnan(fb)
        return abs(fa - fb) <= RTOL * max(abs(fa), abs(fb)) + ATOL
    return type(a) is type(b) and a == b


def _ol_verdict_keys():
    spec = importlib.util.spec_from_file_location("ol_verdict_keys", os.path.join(OL_AXIS, "verdict_keys.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [k for s in "ABC" for k in mod.required_keys(s)]


def part_a_b(AG):
    t0 = time.time()
    OL = AG.OPENLLAMA
    REC = {(pre, lm, vit): AG.load_record(os.path.join(LC.OL_E_RES, "pairs", pre, f"{lm}__{vit}"))
           for pre in AG.PRES for lm in OL for vit in AG.VITS}
    POOL = {(pre, lm, vit): AG.load_record(AG.known_base(pre, lm, vit), full_only=True)
            for pre in AG.PRES for lm in AG.KNOWN11 if lm not in OL for vit in AG.VITS}
    PB = json.load(open(LC.OL_PARTB_JSON))
    vL, _ = AG.ol_legacy_compute(REC, POOL, PB, common=False)
    S = json.load(open(LC.OL_MERGED))
    keys = _ol_verdict_keys()
    res = {"inputs": {"ol_merged_summary_sha256": AG.sha(LC.OL_MERGED), "expected": OL_MERGED_SHA,
                      "ol_partB_sha256": AG.sha(LC.OL_PARTB_JSON), "expected_partB": OL_PARTB_SHA},
           "n_required_keys_A_B_C": len(keys)}
    for impl in ("E", "I"):
        ref = S["embedded"]["verdicts"][impl]
        miss_ref = [k for k in keys if k not in ref]
        miss_new = [k for k in keys if k not in vL]
        mism = [k for k in keys if k in ref and k in vL and not same(AG.fnum(vL[k]) if not isinstance(vL[k], str) else vL[k],
                                                                     ref[k])]
        res[f"vs_{impl}"] = {"n_compared": len(keys) - len(set(miss_ref) | set(miss_new)), "n_mismatch": len(mism),
                             "missing_in_reference": miss_ref[:50], "missing_in_generalised": miss_new[:50],
                             "mismatches": [{"key": k, "generalised": AG.fnum(vL[k]) if not isinstance(vL[k], str)
                                             else vL[k], "reference": ref[k]} for k in mism[:100]],
                             "pass": not mism and not miss_ref and not miss_new}
    res["pass_a"] = res["vs_E"]["pass"]
    # (b) common-ViT-set mode
    vC, _ = AG.ol_legacy_compute(REC, POOL, PB, common=True)
    F = AG.Family(REC, OL, AG.OL_SIZES)
    changed = []
    for k in keys:
        if same(AG.fnum(vL[k]) if not isinstance(vL[k], str) else vL[k],
                AG.fnum(vC[k]) if not isinstance(vC[k], str) else vC[k]):
            continue
        reason = None
        parts = k.split("|")
        if parts[0].startswith("dtr") and len(parts) == 5:
            pre, met, q, X = parts[1:]
            rx = F.rt(pre, met, q, X)
            rb = F.rt(pre, met, "full" if q == "ctm" else q, "raw")
            one = [AG.VITS[j] for j in range(AG.NV) if rx.valid[j] != rb.valid[j]]
            reason = {"vits_valid_under_one_score_only": one, "explained": bool(one)}
        changed.append({"key": k, "legacy": vL[k], "common": vC[k], "reason": reason})
    res["part_b"] = {"n_changed": len(changed), "changed": changed[:200],
                     "all_explained": all(c["reason"] is not None and c["reason"]["explained"] for c in changed)}
    res["pass_b"] = res["part_b"]["all_explained"]
    res["elapsed_s"] = time.time() - t0
    return res


def part_c(AG):
    t0 = time.time()
    S = json.load(open(LAYERMAX_SUMMARY))
    BL = AG.BLOOMZ
    REC = {(pre, lm, vit): AG.load_record(os.path.join(LC.FROZEN_RES, "pairs", pre, f"{lm}__{vit}"))
           for pre in AG.PRES for lm in BL for vit in AG.VITS}
    np_ = [AG.num_params_known(s) for s in BL]
    assert np_ == sorted(np_) and len(set(np_)) == 5, np_
    F = AG.Family(REC, BL, [str(i) for i in range(5)])
    smap = {"s_obs": "raw", "s_cal": "s_cal", "uncensored": "unc"}
    n = 0
    mism = []
    worst = 0.0
    for pre in AG.PRES:
        tr = S["per_pre"][pre]["trends"]
        for met in AG.METRICS:
            for key, d in tr[met].items():
                q, sc = key.split("|")
                rt = F.rt(pre, met, q, smap[sc])
                T = AG.tr_series(rt)
                mine = {"Tr": float(T[0]), "p_one": float(np.mean(T >= T[0] - AG.TOL)),
                        "p_two": float(np.mean(np.abs(T) >= abs(T[0]) - AG.TOL)), "n_vit_constant": int(rt.nexcl)}
                ref = d["tr_lm"]
                for f_ in ("Tr", "p_one", "p_two", "n_vit_constant"):
                    n += 1
                    a, b = mine[f_], ref[f_]
                    dd = abs(float(a) - float(b))
                    worst = max(worst, dd)
                    if dd > 1e-12 + 1e-12 * max(abs(float(a)), abs(float(b))):
                        mism.append({"where": f"{pre}|{met}|{key}|{f_}", "generalised": a, "frozen": b})
    return {"inputs": {"layermax_summary_sha256": AG.sha(LAYERMAX_SUMMARY), "expected": LAYERMAX_SUMMARY_SHA},
            "n_compared": n, "n_mismatch": len(mism), "max_absdiff": worst, "mismatches": mism[:100],
            "pass": not mism and n > 0, "elapsed_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("all", "ab", "c"), default="all")
    args = ap.parse_args()
    LC.gate("early/vl8_generalise.py")
    import aggregate_pyt as AG
    AG.ensure_a12()
    out = {"plan_item": "ANALYSIS_PLAN_PYTHIA.md section 11 V-P8 (E side)", "started_utc": LC.utc()}
    if args.part in ("all", "ab"):
        out["a_b"] = part_a_b(AG)
        print("V-P8(a) vs E:", out["a_b"]["vs_E"]["n_mismatch"], "mismatches of", out["a_b"]["vs_E"]["n_compared"],
              "; vs I:", out["a_b"]["vs_I"]["n_mismatch"], "; (b) changed keys:", out["a_b"]["part_b"]["n_changed"],
              "explained:", out["a_b"]["part_b"]["all_explained"], flush=True)
    if args.part in ("all", "c"):
        out["c"] = part_c(AG)
        print("V-P8(c):", out["c"]["n_mismatch"], "mismatches of", out["c"]["n_compared"], "max abs diff",
              out["c"]["max_absdiff"], flush=True)
    out["pass"] = bool((args.part == "c" or (out["a_b"]["pass_a"] and out["a_b"]["pass_b"]))
                       and (args.part == "ab" or out["c"]["pass"]))
    out["finished_utc"] = LC.utc()
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    name = "VP8_generalisation.json" if args.part == "all" else f"VP8_generalisation.{args.part}.json"
    with open(os.path.join(LC.RES, "validation", name), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    print("V-P8", "PASS" if out["pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
