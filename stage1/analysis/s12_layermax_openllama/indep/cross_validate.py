#!/usr/bin/env python
"""OPENLLAMA-AXIS COPY of analysis/s12_layermax_llama/indep/cross_validate.py (sub-role I of oll-axis-extract,
S0b): plan V-O5 from the I side (E vs I), run at S8 only after I's own summary.json and verdicts_ol.json are
final. The order is enforced and recorded: before any E file is opened, the sha256, mtime and birth time of
I's two files and the UTC time are written to indep/cross_order.json; E files are opened afterwards.

Part 1, pair level (plan V-O5): every one of the 51 pairs x 2 pre x 3 metrics x the 13 layer subsets and
  calibrate-then-max: E's pair summaries (results/S1_2_layermax_openllama/early/pairs/<pre>/<lm>__<vit>.json)
  vs I's records recomputed from I's pair files with the aggregate's exact code (mKNN as Fractions):
  floats (s_obs, c_alpha, s_cal, uncensored, null mean, null sd, selected-entry c_alpha and null mean, M,
  M_unc, c and s_obs at the M-argmax) within |a - b| <= 1e-12 max(|a|, |b|) + 1e-15; p exactly equal;
  argmax, tied maxima and entry counts identical; mKNN null counts (E npz mknn.null_counts_int16) and
  observed counts identical. Informational: E's entry-level arrays.
Part 2, verdict level: compare_verdicts.py (the parent's comparator, read-only) on
  indep/verdicts_ol.json vs early/verdicts_ol.json (--sections EI),
  forecast/I/verdicts_forecast.json vs forecast/E/verdicts_forecast.json (--sections F),
  joint/I/verdicts_joint.json vs joint/E/verdicts_joint.json (--sections J).
  0 mismatches required. A missing file is reported as "missing" (pass = false).
Part 3: clip thresholds, E (results/S1_2_layermax_openllama/cache/) vs I (indep/cache/clip_thresholds_I.json),
  bitwise (V-O1 'E and I').
Output: results/S1_2_layermax_openllama/indep/validation_cross.json (+ cross_order.json, cross_{EI,F,J}.json).
usage: launch_ol.sh ol_I_cross 2 -- venv/bin/python analysis/s12_layermax_openllama/indep/cross_validate.py
       [--parts 1,2,3]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ol_gate  # noqa: E402
OUT_ROOT = ol_gate.gate(__file__)   # F1(a): exits 3 unless Part B is frozen (S0b test mode: scratch root)

_T = os.environ.get("OL_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np  # noqa: E402

import lmx_common as L  # noqa: E402
import lmx_ol_aggregate as AG  # noqa: E402  (pair_record, subsets; main() not run)

RES_OL = ol_gate.RES_OL if not ol_gate.test_mode() else os.path.dirname(L.OUT)
E_DIR = os.environ.get("OL_I_TEST_EDIR") if ol_gate.test_mode() and os.environ.get("OL_I_TEST_EDIR") else \
    os.path.join(RES_OL, "early")
I_SUM = os.path.join(L.OUT, "summary.json")
I_VER = os.path.join(L.OUT, "verdicts_ol.json")
OUTF = os.path.join(L.OUT, "validation_cross.json")
ORDERF = os.path.join(L.OUT, "cross_order.json")
COMPARE = os.path.join(ol_gate.AXIS, "compare_verdicts.py")
OL = list(L.LM_STEMS)
VITS = list(L.VIT_STEMS)
METRICS = ("mknn", "cka_biased", "cka_debiased")
FLOATS = (("raw", "s_obs"), ("c", "c_alpha"), ("s_cal", "s_cal"), ("unc", "uncensored"), ("mu", "null_mean"),
          ("null_sd", "null_sd"))
ORDER_IDX = AG.ORDER_IDX


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


class Acc:
    """accumulates float comparisons under the V-O5 mixed tolerance"""

    def __init__(self):
        self.d = {}

    def add(self, key, a, b, where):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        r = self.d.setdefault(key, {"n": 0, "n_fail": 0, "max_abs": 0.0, "max_ratio_to_tol": 0.0, "worst": None,
                                    "fail_examples": []})
        if a.shape != b.shape:
            r["n"] += 1
            r["n_fail"] += 1
            r["fail_examples"].append({"where": where, "shape": [list(a.shape), list(b.shape)]})
            return
        diff = np.abs(a - b)
        tol = 1e-12 * np.maximum(np.abs(a), np.abs(b)) + 1e-15
        r["n"] += int(diff.size)
        nf = int(np.sum(~(diff <= tol)))
        r["n_fail"] += nf
        if diff.size:
            i = int(np.argmax(diff / tol))
            ratio = float((diff / tol).ravel()[i])
            r["max_abs"] = max(r["max_abs"], float(np.nanmax(diff)))
            if ratio > r["max_ratio_to_tol"]:
                r["max_ratio_to_tol"] = ratio
                r["worst"] = {"where": where, "a": float(a.ravel()[i]), "b": float(b.ravel()[i]),
                              "absdiff": float(diff.ravel()[i])}
        if nf and len(r["fail_examples"]) < 5:
            r["fail_examples"].append({"where": where, "n_fail_here": nf})


class Cnt:
    def __init__(self):
        self.d = {}

    def add(self, key, ok, where, detail=None):
        r = self.d.setdefault(key, {"n": 0, "n_fail": 0, "fail_examples": []})
        r["n"] += 1
        if not ok:
            r["n_fail"] += 1
            if len(r["fail_examples"]) < 10:
                r["fail_examples"].append({"where": where, "detail": detail})


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# ----------------------------------------------------------------------------------------- part 1
def per_pair():
    acc, cnt, arr, arrc = Acc(), Cnt(), Acc(), Cnt()
    n_pairs = 0
    missing = []
    for pre in L.PRES:
        for lm in OL:
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
                    for q in AG.OV.SUBSETS:
                        I, E = Ir[met][q], _g(Ej, "summaries", met, q)
                        w = f"{where}|{met}|{q}"
                        if E is None:
                            cnt.add(f"{met}.summary_present", False, w)
                            continue
                        for fi, fe in FLOATS:
                            acc.add(f"{met}.{fe}", float(I[fi]), float(E.get(fe, np.nan)), w)
                        cnt.add(f"{met}.p_value_exact", float(I["p"]) == E.get("p_value"), w, [I["p"], E.get("p_value")])
                        cnt.add(f"{met}.argmax_identical", (I["a"], I["b"]) == (E.get("argmax_A"), E.get("argmax_B")), w,
                                [I["a"], I["b"], E.get("argmax_A"), E.get("argmax_B")])
                        cnt.add(f"{met}.n_tied_max_identical", I["n_tied_max"] == E.get("n_tied_max"), w)
                        cnt.add(f"{met}.n_entries_identical", I["n_entries"] == E.get("n_layer_pairs"), w)
                    I, E = Ir[met]["full"], _g(Ej, "summaries", met, "full") or {}
                    acc.add(f"{met}.full.entry_c_alpha_at_argmax", float(I["c_entry"]),
                            float(E.get("entry_c_alpha_at_argmax", np.nan)), where)
                    acc.add(f"{met}.full.entry_null_mean_at_argmax", float(I["mu_entry"]),
                            float(E.get("entry_null_mean_at_argmax", np.nan)), where)
                    I, E = Ir[met]["ctm"], _g(Ej, "calib_then_max", met) or {}
                    acc.add(f"{met}.ctm.M", float(I["M"]), float(E.get("M", np.nan)), where)
                    acc.add(f"{met}.ctm.M_unc", float(I["M_unc"]), float(E.get("M_unc", np.nan)), where)
                    acc.add(f"{met}.ctm.c_alpha_at_argmax", float(I["cM"]), float(E.get("c_alpha_at_argmax", np.nan)), where)
                    acc.add(f"{met}.ctm.s_obs_at_argmax", float(I["TM"]), float(E.get("s_obs_at_argmax", np.nan)), where)
                    cnt.add(f"{met}.ctm.argmax_identical", (I["a"], I["b"]) == (E.get("argmax_A"), E.get("argmax_B")), where)
                    cnt.add(f"{met}.ctm.unc_argmax_identical",
                            (I["unc_a"], I["unc_b"]) == (E.get("M_unc_argmax_A"), E.get("M_unc_argmax_B")), where)
                    cnt.add(f"{met}.ctm.n_entries_scal_pos_identical", I["n_entries_scal_pos"] == E.get("n_entries_scal_pos"), where)
                    cnt.add(f"{met}.ctm.n_tied_max_identical", I["n_tied_max"] == E.get("n_tied_max"), where)
                # ---------------- raw arrays
                ze = np.load(fz)
                mk = L.M * L.KNN
                ecnt = ze["mknn.null_counts_int16"].astype(np.int64) if "mknn.null_counts_int16" in ze.files else None
                arrc.add("mknn.null_counts_identical", ecnt is not None and np.array_equal(
                    zi["null_cnt"].astype(np.int64), ecnt), where)
                eobs = ze["mknn.entry_obs"] if "mknn.entry_obs" in ze.files else None
                arrc.add("mknn.obs_counts_identical", eobs is not None and np.array_equal(
                    zi["obs_cnt"].astype(np.int64), np.rint(eobs * mk).astype(np.int64)), where)
                obs = {"mknn": zi["obs_cnt"].astype(np.float64) / mk, "cka_biased": zi["obs_cka_b"],
                       "cka_debiased": zi["obs_cka_d"]}
                null = {"mknn": zi["null_cnt"].astype(np.float64) / mk, "cka_biased": zi["null_cka_b"],
                        "cka_debiased": zi["null_cka_d"]}
                for met in METRICS:
                    S, N = obs[met], null[met]
                    w = f"{where}|{met}"
                    for nm, val in (("entry_obs", S), ("entry_null_mean", N.mean(axis=0))):
                        if f"{met}.{nm}" in ze.files:
                            arr.add(f"{met}.{nm}", val, ze[f"{met}.{nm}"], w)
                    if f"{met}.entry_c_alpha" in ze.files:
                        c_e = np.sort(np.concatenate([S[None], N], axis=0), axis=0)[ORDER_IDX]
                        arr.add(f"{met}.entry_c_alpha", c_e, ze[f"{met}.entry_c_alpha"], w)
                    if f"{met}.full_null_max" in ze.files:
                        arr.add(f"{met}.full_null_max", N.reshape(N.shape[0], -1).max(axis=1), ze[f"{met}.full_null_max"], w)
                n_pairs += 1
        log(f"{pre}: per-pair comparison done")
    ok = (not missing and all(v["n_fail"] == 0 for v in acc.d.values()) and all(v["n_fail"] == 0 for v in cnt.d.values())
          and all(arrc.d[k]["n_fail"] == 0 for k in ("mknn.null_counts_identical", "mknn.obs_counts_identical")))
    return {"n_pair_records": n_pairs, "missing_E_pair_files": missing, "floats_mixed_tol": acc.d, "exact": cnt.d,
            "arrays_exact": arrc.d, "arrays_informational": arr.d, "pass": bool(ok)}


# ----------------------------------------------------------------------------------------- part 2
def verdict_level():
    out = {}
    jobs = {"EI": (I_VER, os.path.join(E_DIR, "verdicts_ol.json")),
            "F": (os.path.join(RES_OL, "forecast", "I", "verdicts_forecast.json"),
                  os.path.join(RES_OL, "forecast", "E", "verdicts_forecast.json")),
            "J": (os.path.join(RES_OL, "joint", "I", "verdicts_joint.json"),
                  os.path.join(RES_OL, "joint", "E", "verdicts_joint.json"))}
    for sec, (a, b) in jobs.items():
        if not (os.path.exists(a) and os.path.exists(b)):
            out[sec] = {"status": "missing", "a": a, "b": b, "a_exists": os.path.exists(a), "b_exists": os.path.exists(b),
                        "pass": False}
            continue
        rep = os.path.join(L.OUT, f"cross_{sec}.json")
        pr = subprocess.run([sys.executable, COMPARE, a, b, "--sections", sec, "--out", rep], capture_output=True, text=True)
        r = json.load(open(rep))
        out[sec] = {"status": "compared", "a": a, "b": b, "a_sha256": sha(a), "b_sha256": sha(b), "exit": pr.returncode,
                    "n_required": r["n_required"], "n_compared": r["n_compared"], "n_mismatch": r["n_mismatch"],
                    "n_missing": len(r["missing_in_a"]) + len(r["missing_in_b"]), "mismatches": r["mismatches"][:50],
                    "pass": bool(r["pass"]), "report": rep}
    return out


# ----------------------------------------------------------------------------------------- part 3
def clip_cmp():
    ci_path = os.path.join(L.OUT, "cache", "clip_thresholds_I.json")
    cands = [os.path.join(RES_OL, "cache", "clip_thresholds.json"), os.path.join(E_DIR, "cache", "clip.json")]
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
    per = {s: {"I": c, "E": find(s), "bitwise_equal": find(s) == c} for s, c in ci.items()}
    return {"status": "compared", "E_file": ce_path, "per_model": per,
            "pass": bool(per and all(v["bitwise_equal"] for v in per.values()))}


def main():
    ap = argparse.ArgumentParser(description="V-O5 from the I side (run after I's summary.json and verdicts_ol.json)")
    ap.add_argument("--parts", default="1,2,3")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    t0 = time.time()
    # ---- order: I's outputs final before any E file is opened
    for p in (I_SUM, I_VER):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing: run lmx_ol_aggregate.py first (I outputs must be final before E is opened)")
    order = {"I_final": {os.path.relpath(p, L.PROJ) if p.startswith(L.PROJ) else p:
                         {"sha256": sha(p), "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(p))),
                          "birth_utc": birth(p)} for p in (I_SUM, I_VER)},
             "E_first_opened_after_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "rule": "plan V-O5 / sub-role I: E outputs are opened only after I's summary.json and verdicts_ol.json are final"}
    json.dump(order, open(ORDERF, "w"), indent=1)
    log(f"order recorded in {ORDERF}")
    res = {"what": "implementation I vs implementation E (V-O5), I side", "order": order}
    if "1" in parts:
        res["part1_pair_level"] = per_pair()
    if "2" in parts:
        res["part2_verdict_level"] = verdict_level()
    if "3" in parts:
        res["part3_clip_E_vs_I"] = clip_cmp()
    p1 = res.get("part1_pair_level", {}).get("pass")
    p2 = all(v["pass"] for v in res.get("part2_verdict_level", {}).values()) if "2" in parts else None
    p3 = res.get("part3_clip_E_vs_I", {}).get("pass")
    res["pass_V-O5"] = bool(p1 and p2 and p3)
    lines = []
    if "1" in parts:
        P1 = res["part1_pair_level"]
        fl = P1["floats_mixed_tol"]
        lines.append(f"- **V-O5 pair level** ({P1['n_pair_records']} pair records; missing E files: "
                     f"{len(P1['missing_E_pair_files'])}): floats {sum(v['n'] for v in fl.values())} values, "
                     f"{sum(v['n_fail'] for v in fl.values())} outside 1e-12·max + 1e-15; exact items "
                     f"{sum(v['n'] for v in P1['exact'].values())} checks, {sum(v['n_fail'] for v in P1['exact'].values())} differ; "
                     f"mKNN null counts identical in {P1['arrays_exact'].get('mknn.null_counts_identical', {}).get('n', 0) - P1['arrays_exact'].get('mknn.null_counts_identical', {}).get('n_fail', 0)}"
                     f"/{P1['arrays_exact'].get('mknn.null_counts_identical', {}).get('n', 0)} pair records. Pass: {P1['pass']}.")
    if "2" in parts:
        for sec, v in res["part2_verdict_level"].items():
            if v["status"] == "missing":
                lines.append(f"- **compare_verdicts --sections {sec}:** file missing (I {v['a_exists']}, E {v['b_exists']}).")
            else:
                lines.append(f"- **compare_verdicts --sections {sec}:** {v['n_compared']} of {v['n_required']} required keys compared, "
                             f"{v['n_mismatch']} mismatches, {v['n_missing']} missing. Pass: {v['pass']}.")
    if "3" in parts:
        v = res["part3_clip_E_vs_I"]
        lines.append(f"- **Clip thresholds E vs I (V-O1):** {v['status']}; pass {v['pass']}.")
    res["report_lines"] = lines
    res["secs"] = time.time() - t0
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(L.jsonable(res), open(OUTF + ".tmp", "w"), indent=1, default=str)
    os.replace(OUTF + ".tmp", OUTF)
    log(f"wrote {OUTF}: pass_V-O5={res['pass_V-O5']} (pair {p1}, verdicts {p2}, clip {p3}) ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
