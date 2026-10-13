#!/usr/bin/env python
"""PYT-AXIS COPY of analysis/s12_layermax_openllama/indep/cross_validate.py (sha256 ab3a4b19...), sub-role pyt-code-I
(S0b, 2026-10-10): plan V-P5 from the I side (E vs I).

PYT-AXIS rules (plan 11 V-P5, R.2 B6.4): this script PRINTS AND WRITES MISMATCH COUNTS AND THE NAMES OF MISMATCHING KEYS
ONLY, NEVER THEIR VALUES. It runs per pre (--pres; the primary first, at S7, so that the l2 go/no-go can be logged before
any primary verdict is opened), and at S8 on the verdict files. Order rule: before any E file is opened, the sha256,
mtime and birth time of the I files it compares and the UTC time are written to indep/cross_order_<parts>_<pres>.json;
E files are opened afterwards.

Part 1, pair level (plan V-P5): every one of the 119 pairs x the selected pre x 3 metrics x the 13 layer subsets and
  calibrate-then-max: E's pair summaries (results/S1_2_layermax_pythia/early/pairs/<pre>/<lm>__<vit>.json; field names
  of the OpenLLaMA E copy) vs I's records recomputed from I's pair files with the aggregate's exact code (mKNN as
  Fractions): floats within |a - b| <= 1e-12 max(|a|, |b|) + 1e-15; p exactly equal; argmax, tied maxima and entry
  counts identical; mKNN observed and null counts identical.
Part 2, verdict level (VERDICTS_SPEC_PYT.md section 0): the shared compare_verdicts.py on I's verdicts_pyt.json vs
  E's (--e-verdicts; default results/S1_2_layermax_pythia/early/verdicts_pyt.json) with --sections EI, and on
  forecast/I vs forecast/E with --sections FK when those files exist; the comparator writes counts and key names only.
Part 3: clip thresholds, E (results/S1_2_layermax_pythia/cache/clip_thresholds.json or early/cache/clip.json) vs I
  (indep/cache/clip_thresholds_I.json), bitwise (V-P1 "E and I").
Output: results/S1_2_layermax_pythia/indep/validation_cross[_<pres>].json (counts and key names only).
usage: launch_pyt.sh pyt_I_cross_prh 2 -- venv/bin/python analysis/s12_layermax_pythia/indep/cross_validate.py
       --pres prh_clip+l2 --parts 1          (S7, before the l2 go/no-go)
       --pres prh_clip+l2,l2 --parts 1,2,3   (S8, after I's summary.json and verdicts_pyt.json are final)
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True   # PYT-AXIS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyt_gate  # noqa: E402  PYT-AXIS
OUT_ROOT = pyt_gate.gate(__file__)   # PYT-AXIS F1(a): exits 3 unless the plan's gates hold (S0b test mode: test root)

_T = os.environ.get("PYT_THREADS", "1")   # PYT-AXIS
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np  # noqa: E402

import lmx_common as L  # noqa: E402
import lmx_pyt_aggregate as AG  # noqa: E402  (pair_record, load_z; main() not run)

RES_PYT = pyt_gate.RES_PYT if not pyt_gate.test_mode() else os.path.dirname(L.OUT)   # PYT-AXIS (test: the test root)
E_DIR = os.environ.get("PYT_I_TEST_EDIR") if pyt_gate.test_mode() and os.environ.get("PYT_I_TEST_EDIR") else \
    os.path.join(RES_PYT, "early")
I_SUM = os.path.join(L.OUT, "summary.json")
I_VER = os.path.join(L.OUT, "verdicts_pyt.json")
PYT = list(L.LM_STEMS)
VITS = list(L.VIT_STEMS)
METRICS = ("mknn", "cka_biased", "cka_debiased")
FLOATS = (("raw", "s_obs"), ("c", "c_alpha"), ("s_cal", "s_cal"), ("unc", "uncensored"), ("mu", "null_mean"),
          ("null_sd", "null_sd"))
ORDER_IDX = AG.ORDER_IDX
RTOL, ATOL = 1e-12, 1e-15


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def birth(p):
    out = subprocess.run(["stat", "-c", "%W", p], capture_output=True, text=True).stdout.strip()
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(out)))
    except ValueError:
        return None


class Cnt:
    """PYT-AXIS: counts and the names (keys) of failing checks only; values are never stored."""

    def __init__(self):
        self.d = {}

    def add(self, key, ok, where):
        r = self.d.setdefault(key, {"n": 0, "n_fail": 0, "fail_where": []})
        r["n"] += 1
        if not ok:
            r["n_fail"] += 1
            if len(r["fail_where"]) < 20:
                r["fail_where"].append(where)


def fclose(a, b) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    return abs(fa - fb) <= RTOL * max(abs(fa), abs(fb)) + ATOL


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# ----------------------------------------------------------------------------------------- part 1
def per_pair(pres):
    cnt = Cnt()
    n_pairs = 0
    missing = []
    for pre in pres:
        for lm in PYT:
            for vit in VITS:
                where = f"{pre}|{lm}|{vit}"
                fj = os.path.join(E_DIR, "pairs", pre, f"{lm}__{vit}.json")
                fz = os.path.join(E_DIR, "pairs", pre, f"{lm}__{vit}.npz")
                if not (os.path.exists(fj) and os.path.exists(fz)):
                    missing.append(where)
                    continue
                Ej = json.load(open(fj))
                zi = AG.load_z(L.OUT, pre, lm, vit)
                Ir = AG.pair_record(zi, detail=True)
                LA, LB = zi["obs_cnt"].shape
                cnt.add("meta_K_key_shape", Ej.get("K") == L.K_PERM and Ej.get("perm_key") == L.perm_key(lm, vit)
                        and Ej.get("L_A") == LA and Ej.get("L_B") == LB, where)
                for met in METRICS:
                    for q in AG.PV.SUBSETS:
                        I, E = Ir[met][q], _g(Ej, "summaries", met, q)
                        w = f"{where}|{met}|{q}"
                        if E is None:
                            cnt.add(f"{met}.summary_present", False, w)
                            continue
                        for fi, fe in FLOATS:
                            cnt.add(f"{met}.{fe}", fclose(I[fi], E.get(fe)), w)
                        cnt.add(f"{met}.p_value_exact", float(I["p"]) == E.get("p_value"), w)
                        cnt.add(f"{met}.argmax_identical", (I["a"], I["b"]) == (E.get("argmax_A"), E.get("argmax_B")), w)
                        cnt.add(f"{met}.n_tied_max_identical", I["n_tied_max"] == E.get("n_tied_max"), w)
                        cnt.add(f"{met}.n_entries_identical", I["n_entries"] == E.get("n_layer_pairs"), w)
                    I, E = Ir[met]["full"], _g(Ej, "summaries", met, "full") or {}
                    cnt.add(f"{met}.full.entry_c_alpha_at_argmax", fclose(I["c_entry"], E.get("entry_c_alpha_at_argmax")),
                            where)
                    cnt.add(f"{met}.full.entry_null_mean_at_argmax",
                            fclose(I["mu_entry"], E.get("entry_null_mean_at_argmax")), where)
                    I, E = Ir[met]["ctm"], _g(Ej, "calib_then_max", met) or {}
                    cnt.add(f"{met}.ctm.M", fclose(I["M"], E.get("M")), where)
                    cnt.add(f"{met}.ctm.M_unc", fclose(I["M_unc"], E.get("M_unc")), where)
                    cnt.add(f"{met}.ctm.c_alpha_at_argmax", fclose(I["cM"], E.get("c_alpha_at_argmax")), where)
                    cnt.add(f"{met}.ctm.s_obs_at_argmax", fclose(I["TM"], E.get("s_obs_at_argmax")), where)
                    cnt.add(f"{met}.ctm.argmax_identical", (I["a"], I["b"]) == (E.get("argmax_A"), E.get("argmax_B")), where)
                    cnt.add(f"{met}.ctm.unc_argmax_identical",
                            (I["unc_a"], I["unc_b"]) == (E.get("M_unc_argmax_A"), E.get("M_unc_argmax_B")), where)
                    cnt.add(f"{met}.ctm.n_entries_scal_pos_identical", I["n_entries_scal_pos"] == E.get("n_entries_scal_pos"),
                            where)
                    cnt.add(f"{met}.ctm.n_tied_max_identical", I["n_tied_max"] == E.get("n_tied_max"), where)
                ze = np.load(fz)
                mk = L.M * L.KNN
                ecnt = ze["mknn.null_counts_int16"].astype(np.int64) if "mknn.null_counts_int16" in ze.files else None
                cnt.add("mknn.null_counts_identical", ecnt is not None and np.array_equal(
                    zi["null_cnt"].astype(np.int64), ecnt), where)
                eobs = ze["mknn.entry_obs"] if "mknn.entry_obs" in ze.files else None
                cnt.add("mknn.obs_counts_identical", eobs is not None and np.array_equal(
                    zi["obs_cnt"].astype(np.int64), np.rint(eobs * mk).astype(np.int64)), where)
                n_pairs += 1
        log(f"{pre}: per-pair comparison done")
    ok = not missing and all(v["n_fail"] == 0 for v in cnt.d.values())
    return {"n_pair_records": n_pairs, "n_missing_E_pair_files": len(missing), "missing_E_pair_files": missing[:50],
            "checks": cnt.d, "n_checks": sum(v["n"] for v in cnt.d.values()),
            "n_fail": sum(v["n_fail"] for v in cnt.d.values()), "pass": bool(ok)}


# ----------------------------------------------------------------------------------------- part 2
COMPARE = os.path.join(pyt_gate.AXIS, "compare_verdicts.py")   # PYT-AXIS: the shared comparator (names only)


def verdict_level(e_path, sections="EI", i_path=None):
    """PYT-AXIS: VERDICTS_SPEC_PYT.md section 0: V-P5 = compare_verdicts.py on E vs I (--sections EI) and on
    forecast/E vs forecast/I (--sections FK); the comparator prints and writes counts and key names only."""
    i_path = i_path or I_VER
    if not os.path.exists(i_path) or e_path is None or not os.path.exists(e_path) or not os.path.exists(COMPARE):
        return {"status": "missing", "I": i_path, "E": e_path, "comparator": os.path.exists(COMPARE), "pass": False}
    rep = os.path.join(L.OUT, f"cross_{sections}.json")
    L.keep_aside(rep)
    pr = subprocess.run([sys.executable, COMPARE, i_path, e_path, "--sections", sections, "--out", rep],
                        capture_output=True, text=True)
    r = json.load(open(rep)) if os.path.exists(rep) else {}
    return {"status": "compared", "sections": sections, "I": i_path, "E": e_path, "I_sha256": sha(i_path),
            "E_sha256": sha(e_path), "exit": pr.returncode, "n_required": r.get("n_required"),
            "n_compared": r.get("n_compared"), "n_mismatch": r.get("n_mismatch"),
            "n_missing_in_I": r.get("n_missing_in_a"), "n_missing_in_E": r.get("n_missing_in_b"),
            "report_names_only": rep, "pass": bool(r.get("pass")) and pr.returncode == 0}


# ----------------------------------------------------------------------------------------- part 3
def clip_cmp():
    ci_path = os.path.join(L.OUT, "cache", "clip_thresholds_I.json")
    cands = [os.path.join(RES_PYT, "cache", "clip_thresholds.json"), os.path.join(E_DIR, "cache", "clip.json")]
    ce_path = next((p for p in cands if os.path.exists(p)), None)
    if not os.path.exists(ci_path) or ce_path is None:
        return {"status": "missing", "I": ci_path, "E_candidates": cands, "pass": False}
    ci = json.load(open(ci_path))["clip"]
    ce = json.load(open(ce_path))
    ce = ce.get("clip", ce) if isinstance(ce, dict) else {}

    def cval(v):
        return float(v["c"]) if isinstance(v, dict) and "c" in v else (float(v) if isinstance(v, (int, float)) else None)

    def find(stem):
        for k, v in ce.items():
            if k == stem or k.startswith(stem + "|"):
                return cval(v)
        return None
    bad = [s for s, c in ci.items() if find(s) != c]
    return {"status": "compared", "E_file": ce_path, "n_models": len(ci), "n_not_bitwise_equal": len(bad),
            "models_not_bitwise_equal": bad, "pass": bool(ci and not bad)}


def main():
    ap = argparse.ArgumentParser(description="V-P5 from the I side: counts and key names only")
    ap.add_argument("--parts", default="1,2,3")
    ap.add_argument("--pres", default="prh_clip+l2", help="comma list (primary first; l2 only after the go line)")
    ap.add_argument("--e-verdicts", default=None, help="E's verdict file (default: early/verdicts_pyt.json, "
                    "early/verdicts.json)")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    pres = tuple(args.pres.split(","))
    assert set(pres) <= set(L.PRES)
    t0 = time.time()
    i_files = [I_SUM, I_VER] if "2" in parts else []
    for p in i_files:
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing: run lmx_pyt_aggregate.py first (I outputs must be final before E is opened)")
    tagp = f"{'-'.join(sorted(parts))}_{'+'.join(pres)}"
    order = {"I_final": {os.path.relpath(p, L.PROJ) if p.startswith(L.PROJ) else p:
                         {"sha256": sha(p), "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(p))),
                          "birth_utc": birth(p)} for p in i_files},
             "E_first_opened_after_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "pres": list(pres), "parts": sorted(parts),
             "rule": "plan V-P5: E outputs are opened only after I's outputs are final; counts and key names only"}
    L.write_json(os.path.join(L.OUT, f"cross_order_{tagp}.json"), order)
    log(f"order recorded (parts {sorted(parts)}, pres {list(pres)})")
    res = {"what": "implementation I vs implementation E (V-P5), I side; counts and key names only", "order": order}
    if "1" in parts:
        res["part1_pair_level"] = per_pair(pres)
    if "2" in parts:
        e_path = args.e_verdicts or os.path.join(E_DIR, "verdicts_pyt.json")
        res["part2_verdict_level"] = verdict_level(e_path, "EI")
        fI = os.path.join(RES_PYT, "forecast", "I", "verdicts_forecast.json")
        fE = os.path.join(RES_PYT, "forecast", "E", "verdicts_forecast.json")
        if os.path.exists(fI) or os.path.exists(fE):
            res["part2_forecast_FK"] = verdict_level(fE, "FK", i_path=fI)
    if "3" in parts:
        res["part3_clip_E_vs_I"] = clip_cmp()
    flags = [res[k]["pass"] for k in ("part1_pair_level", "part2_verdict_level", "part2_forecast_FK", "part3_clip_E_vs_I")
             if k in res]
    res["pass_V-P5_selected_parts"] = bool(flags and all(flags))
    res["secs"] = time.time() - t0
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    L.write_json(os.path.join(L.OUT, f"validation_cross_{tagp}.json"), L.jsonable(res), default=str)
    p1 = res.get("part1_pair_level", {})
    p2 = res.get("part2_verdict_level", {})
    log(f"V-P5 parts {sorted(parts)} pres {list(pres)}: pair level {p1.get('n_checks')} checks, {p1.get('n_fail')} fail, "
        f"{p1.get('n_missing_E_pair_files')} E files missing; verdict level (EI) {p2.get('n_compared')} keys compared, "
        f"{p2.get('n_mismatch')} mismatch, missing in I {p2.get('n_missing_in_I')}, in E {p2.get('n_missing_in_E')}; "
        f"pass={res['pass_V-P5_selected_parts']} (key names of any mismatch: {p2.get('report_names_only')})")


if __name__ == "__main__":
    main()
