#!/usr/bin/env python
"""Role pyt-run-I: diagnostic of the V-P5 float mismatches (NOT frozen code; decides nothing; run after I's final
summary, after the l2 go line and after the frozen V-P5 record). For every float field that the frozen
indep/cross_validate.py compares at the pair level (CKA metrics, 13 subsets, both pre; ctm and full entry fields) and
for the mismatching verdict keys named in indep/cross_EI.json, it writes the SIZE of the E - I differences (absolute,
relative, in units of the last place) and the order of magnitude of the quantity, never the values themselves.
It also measures the distance (in ulps) between E's reported mKNN floats and float(Fraction) of the integer counts.
Output: results/S1_2_layermax_pythia/indep/vp5_diag_I.json (refuses to overwrite).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
PROJ = os.path.realpath(os.path.expanduser("~/hrh_v2_stage1_20261008"))
sys.path.insert(0, os.path.join(PROJ, "analysis", "s12_layermax_pythia", "indep"))
import pyt_gate  # noqa: E402
OUT_ROOT = pyt_gate.gate(__file__)

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from fractions import Fraction  # noqa: E402

import numpy as np  # noqa: E402

import lmx_common as L  # noqa: E402  (frozen I helpers, read-only import)
import lmx_pyt_aggregate as AG  # noqa: E402  (pair_record, load_z; main() not run)

E_DIR = os.path.join(PROJ, "results", "S1_2_layermax_pythia", "early")
FLOATS = (("raw", "s_obs"), ("c", "c_alpha"), ("s_cal", "s_cal"), ("unc", "uncensored"), ("mu", "null_mean"),
          ("null_sd", "null_sd"))
RTOL, ATOL = 1e-12, 1e-15


def ulps(a, b):
    a, b = float(a), float(b)
    if a == b:
        return 0
    return int(abs(np.int64(np.float64(a).view(np.int64)) - np.int64(np.float64(b).view(np.int64))))


class Acc:
    def __init__(self):
        self.d = {}

    def add(self, key, a, b, where):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            r = self.d.setdefault(key, {"n": 0, "n_noncomparable": 0})
            r["n"] += 1
            r["n_noncomparable"] = r.get("n_noncomparable", 0) + 1
            return
        r = self.d.setdefault(key, {"n": 0, "n_fail_tol": 0, "max_abs_diff": 0.0, "max_rel_diff": 0.0,
                                    "max_ulps": 0, "fail_abs_diff_max": 0.0, "fail_mag_log10": [], "where_max_rel": None})
        r["n"] += 1
        d = abs(fa - fb)
        m = max(abs(fa), abs(fb))
        rel = d / m if m > 0 else 0.0
        if d > r["max_abs_diff"]:
            r["max_abs_diff"] = d
        if rel > r["max_rel_diff"]:
            r["max_rel_diff"], r["where_max_rel"] = rel, where
        r["max_ulps"] = max(r["max_ulps"], ulps(fa, fb))
        if d > RTOL * m + ATOL:
            r["n_fail_tol"] += 1
            r["fail_abs_diff_max"] = max(r["fail_abs_diff_max"], d)
            if m > 0:
                r["fail_mag_log10"].append(round(math.log10(m), 1))

    def out(self):
        res = {}
        for k, r in self.d.items():
            r = dict(r)
            mags = r.pop("fail_mag_log10", [])
            if mags:
                r["fail_magnitude_log10_min_median_max"] = [min(mags), float(np.median(mags)), max(mags)]
            res[k] = r
        return res


def main():
    dst = os.path.join(OUT_ROOT, "vp5_diag_I.json")
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists")
    t0 = time.time()
    acc, enc = Acc(), Acc()
    for pre in L.PRES:
        for lm in L.LM_STEMS:
            for vit in L.VIT_STEMS:
                where = f"{pre}|{lm}|{vit}"
                Ej = json.load(open(os.path.join(E_DIR, "pairs", pre, f"{lm}__{vit}.json")))
                Ir = AG.pair_record(AG.load_z(L.OUT, pre, lm, vit), detail=True)
                for met in ("mknn", "cka_biased", "cka_debiased"):
                    for q in AG.PV.SUBSETS:
                        I, E = Ir[met][q], ((Ej.get("summaries") or {}).get(met) or {}).get(q) or {}
                        for fi, fe in FLOATS:
                            if met == "mknn" and isinstance(I[fi], Fraction):
                                enc.add(f"mknn.{fe}.E_vs_float(Fraction)", float(I[fi]), E.get(fe), f"{where}|{q}")
                            acc.add(f"{met}.{fe}", float(I[fi]), E.get(fe), f"{where}|{met}|{q}")
                    I, E = Ir[met]["full"], ((Ej.get("summaries") or {}).get(met) or {}).get("full") or {}
                    acc.add(f"{met}.full.entry_c_alpha_at_argmax", float(I["c_entry"]), E.get("entry_c_alpha_at_argmax"), where)
                    acc.add(f"{met}.full.entry_null_mean_at_argmax", float(I["mu_entry"]), E.get("entry_null_mean_at_argmax"), where)
                    I, E = Ir[met]["ctm"], (Ej.get("calib_then_max") or {}).get(met) or {}
                    for fi, fe in (("M", "M"), ("M_unc", "M_unc"), ("cM", "c_alpha_at_argmax"), ("TM", "s_obs_at_argmax")):
                        acc.add(f"{met}.ctm.{fe}", float(I[fi]), E.get(fe), where)
                        if met == "mknn" and isinstance(I[fi], Fraction):
                            enc.add(f"mknn.ctm.{fe}.E_vs_float(Fraction)", float(I[fi]), E.get(fe), where)
    names = json.load(open(os.path.join(OUT_ROOT, "cross_EI.json"))).get("mismatching_keys", [])
    vi = json.load(open(os.path.join(OUT_ROOT, "verdicts_pyt.json")))["v"]
    ve = json.load(open(os.path.join(E_DIR, "verdicts_pyt.json")))["v"]
    vacc = Acc()
    for k in names:
        vacc.add(k, vi.get(k), ve.get(k), k)
    res = {"what": "size of the E - I float differences behind the V-P5 mismatches (magnitudes only, no values); "
                   "diagnostic of role pyt-run-I, decides nothing", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tolerance": "|a-b| <= 1e-12 max(|a|,|b|) + 1e-15 (plan V-P5)",
           "pair_level": acc.out(), "mknn_float_encoding": enc.out(), "verdict_level_mismatching_keys": vacc.out(),
           "secs": time.time() - t0}
    with open(dst + ".tmp", "w") as f:
        json.dump(res, f, indent=1, default=str)
    os.replace(dst + ".tmp", dst)
    nf = {k: v["n_fail_tol"] for k, v in res["pair_level"].items() if v.get("n_fail_tol")}
    print(f"[vp5_diag_I] written {dst}; pair-level fields failing tolerance: {nf}", flush=True)


if __name__ == "__main__":
    main()
