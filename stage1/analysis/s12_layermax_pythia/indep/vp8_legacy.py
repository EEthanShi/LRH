#!/usr/bin/env python
"""Pythia axis, implementation I: plan V-P8 (generalisation check, S0b, CPU) (NEW file, PYT-AXIS, sub-role pyt-code-I,
2026-10-10). Reads only frozen non-Pythia files; runs only in S0b test mode (PYT_S0B_TEST=1 and no CODE FROZEN line;
otherwise exit 3 with no output); writes only analysis/s12_layermax_pythia/s0b_scratch/I/vp8/. Part (a) also runs the
shared comparator as VERDICTS_SPEC_PYT.md section 0 prescribes: compare_verdicts.py <legacy file>
results/S1_2_layermax_openllama/layermax_openllama_summary.json --b-sub embedded.verdicts.{I,E} --legacy --sections A,B,C.

  (a) legacy mode, n = 3, on the frozen OpenLLaMA I pair files (results/S1_2_layermax_openllama/indep/<pre>/pairs/,
      plus the frozen 11-LM I files for the P-O4 pooled r): the generalised aggregator (pyt_verdicts.ol_verdicts with the
      n-size trend, mode "legacy") must reproduce every section A-C verdict key (OpenLLaMA verdict_keys.required_keys
      "A", "B", "C") of the I verdicts embedded in results/S1_2_layermax_openllama/layermax_openllama_summary.json
      (2777dadf...): 0 mismatches (comparator: the OpenLLaMA compare_verdicts tolerance 1e-12 max + 1e-15). The E block
      is compared too (reported).
  (b) common-ViT-set mode, n = 3, same inputs: the keys that change against (a) are listed; each changed dtr key must be
      explained by a ViT that is valid (finite, not constant) under exactly one of its two scores; every other changed key
      must derive from a changed dtr key of the same pre (P-O1/P-O2/P-O4/P-O5 categories and curves).
  (c) legacy mode, n = 5, on the frozen BLOOMZ I pair files (results/S1_2_layermax/indep/<pre>/pairs/): the BLOOMZ
      layer-max trends and exact p of results/S1_2_layermax/layermax_summary.json (0dddd17a...; per_pre/<pre>/trends/
      <metric>/<subset>|<score>/tr_lm: Tr, p_one, p_two, n_vit_constant) are reproduced to 1e-12 absolute.
usage: PYT_S0B_TEST=1 launch_pyt.sh pyt_I_vp8 2 -- venv/bin/python analysis/s12_layermax_pythia/indep/vp8_legacy.py
       [--parts a,b,c]
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyt_gate  # noqa: E402
if not pyt_gate.help_requested() and not pyt_gate.test_mode():
    pyt_gate._fail("vp8_legacy.py: V-P8 is an S0b check; it runs only with PYT_S0B_TEST=1 before CODE FROZEN")

_T = os.environ.get("PYT_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import lmx_common as L  # noqa: E402
import lmx_pyt_aggregate as AG  # noqa: E402  (pair_record, load_z; main() not run)
import pyt_verdicts as PV  # noqa: E402

OUTD = os.path.join(pyt_gate.TEST_ROOT, "vp8")
OLAX = os.path.join(L.PROJ, "analysis", "s12_layermax_openllama")
OL_SUMMARY = os.path.join(L.PROJ, "results", "S1_2_layermax_openllama", "layermax_openllama_summary.json")
OL_PARTB = os.path.join(L.PROJ, "results", "S1_2_layermax_openllama", "partB", "partB_features_ol.json")
BL_SUMMARY = os.path.join(L.PROJ, "results", "S1_2_layermax", "layermax_summary.json")
SHA = {OL_SUMMARY: "2777dadf287dfc02433eb07e84cefad0ec67381258a199b9d306b6c7d32aebaa",
       BL_SUMMARY: "0dddd17a588dda2be47b19ec30b182f61f01dcfb5f791f237b39bf421d2c5613",
       OL_PARTB: "e1104bd71e5eff9ced65ed402a61c187590bce5dee306c25ee2f64b18fdd7e51"}
OL = ["openlm-research__open_llama_3b", "openlm-research__open_llama_7b", "openlm-research__open_llama_13b"]
BLOOMZ = ["bigscience__bloomz-560m", "bigscience__bloomz-1b1", "bigscience__bloomz-1b7", "bigscience__bloomz-3b",
          "bigscience__bloomz-7b1"]
VITS = list(L.VIT_STEMS)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ol_root(lm):
    """the OpenLLaMA I aggregator's source rule: OpenLLaMA models from its own run, llama-13b/30b from the LLaMA-axis
    I run, BLOOMZ and llama-7b from the frozen 8-LM I run."""
    if lm in OL:
        return L.OL_INDEP
    if lm in ("huggyllama__llama-13b", "huggyllama__llama-30b"):
        return L.LLAMA_INDEP
    return L.OLD_INDEP


def same(a, b) -> bool:   # the OpenLLaMA compare_verdicts semantics
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return math.isnan(fa) and math.isnan(fb)
        return abs(fa - fb) <= 1e-12 * max(abs(fa), abs(fb)) + 1e-15
    return type(a) is type(b) and a == b


def jv(V):
    return {k: (v if isinstance(v, str) else PV.f(v)) for k, v in V.items()}


def part_ab(parts):
    VK = _load_module("ol_verdict_keys", os.path.join(OLAX, "verdict_keys.py"))
    req = VK.required_keys("A") + VK.required_keys("B") + VK.required_keys("C")
    summ = json.load(open(OL_SUMMARY))
    emb = summ["embedded"]["verdicts"]
    t0 = time.time()
    Vleg, rec = AG.legacy_verdicts("legacy")   # the aggregator's legacy mode (spec 0), n = 3
    log(f"legacy verdicts built ({time.time() - t0:.0f}s)")
    out = {}
    for impl in ("I", "E"):
        ref = emb[impl]
        miss = [k for k in req if k not in Vleg or k not in ref]
        mism = [k for k in req if k in Vleg and k in ref and not same(Vleg[k], ref[k])]
        out[f"a_vs_{impl}"] = {"n_required": len(req), "n_missing": len(miss), "missing": miss[:50],
                               "n_mismatch": len(mism), "mismatching_keys": mism[:50],
                               "pass": not miss and not mism}
        log(f"V-P8(a) vs {impl}: {len(req)} keys, {len(miss)} missing, {len(mism)} mismatches")
    lf = os.path.join(OUTD, "verdicts_ol_legacy.json")
    L.write_json(lf, {"schema": "OL-VERDICTS v1", "impl": "I", "mode": "legacy (n = 3)", "v": Vleg})
    cmp = os.path.join(pyt_gate.AXIS, "compare_verdicts.py")
    if os.path.exists(cmp):
        for impl in ("I", "E"):
            rep = os.path.join(OUTD, f"compare_verdicts_legacy_vs_{impl}.json")
            L.keep_aside(rep)
            pr = subprocess.run([sys.executable, cmp, lf, OL_SUMMARY, "--b-sub", f"embedded.verdicts.{impl}", "--legacy",
                                 "--sections", "A,B,C", "--out", rep], capture_output=True, text=True)
            r = json.load(open(rep)) if os.path.exists(rep) else {}
            out[f"a_shared_comparator_vs_{impl}"] = {"exit": pr.returncode, "n_compared": r.get("n_compared"),
                                                     "n_mismatch": r.get("n_mismatch"), "pass": bool(r.get("pass"))}
            log(f"V-P8(a) shared comparator vs {impl}: exit {pr.returncode}, n_compared {r.get('n_compared')}, "
                f"n_mismatch {r.get('n_mismatch')}")
    else:
        out["a_shared_comparator"] = "compare_verdicts.py absent"
    out["a_pass"] = bool(out["a_vs_I"]["pass"] and out.get("a_shared_comparator_vs_I", {"pass": True})["pass"])
    if "b" in parts:
        Vcom, _ = AG.legacy_verdicts("common")
        changed = [k for k in req if not same(Vleg.get(k), Vcom.get(k))]
        expl = {}
        for k in changed:
            f_ = k.split("|")
            if f_[0] in ("dtr", "dtr_pge", "dtr_ple", "dtr_ptwo"):
                p, m, q, x = f_[1], f_[2], f_[3], f_[4]
                bq = "full" if q == "ctm" else q
                cx = PV.cols_of(rec[p], OL, VITS, m, q, x)
                cb = PV.cols_of(rec[p], OL, VITS, m, bq, "raw")
                one = [VITS[j] for j in range(len(VITS)) if PV.is_valid(cx[j]) != PV.is_valid(cb[j])]
                expl[k] = {"vits_valid_under_one_score_only": one, "explained": bool(one)}
        derived = {}
        src = {"PO1": ("mknn", "full"), "PO2": ("cka_biased", "full"), "PO4": ("cka_debiased", "full"),
               "PO5": ("cka_biased", "ctm")}
        for k in changed:
            if k.startswith("dtr"):
                continue
            f_ = k.split("|")
            p = f_[1]
            fam = f_[2] if f_[0] == "curve" else f_[0]
            if fam not in src:
                derived[k] = {"explained": False, "why": "not a dtr-derived key"}
                continue
            m, q = src[fam]
            dep = [kk for kk in changed if kk.startswith(f"dtr|{p}|{m}|{q}|")]
            derived[k] = {"family": fam, "depends_on_changed": dep,
                          "explained": bool(dep and all(expl.get(d, {}).get("explained") for d in dep))}
        out["b"] = {"n_changed": len(changed), "changed_keys": changed, "dtr_explanations": expl,
                    "derived_explanations": derived,
                    "pass": all(v["explained"] for v in expl.values()) and all(v["explained"] for v in derived.values())}
        log(f"V-P8(b): {len(changed)} keys change in the common-ViT-set mode; all explained {out['b']['pass']}")
    return out


def part_c():
    S = json.load(open(BL_SUMMARY))
    t0 = time.time()
    xmap = {"s_obs": "raw", "s_cal": "s_cal", "uncensored": "unc"}
    rows, worst, n, bad = [], 0.0, 0, 0
    for p in L.PRES:
        rec = {lm: {v: AG.pair_record(AG.load_z(L.OLD_INDEP, p, lm, v), detail=True) for v in VITS} for lm in BLOOMZ}
        log(f"BLOOMZ {p} records built ({time.time() - t0:.0f}s)")
        for m in ("mknn", "cka_biased", "cka_debiased"):
            for key, ent in S["per_pre"][p]["trends"][m].items():
                sub, xs = key.split("|")
                cols = [[rec[lm][v][m][sub][xmap[xs]] for lm in BLOOMZ] for v in VITS]
                t = PV.trend(cols)
                pge, ple, ptwo = PV.pvals(t["T"], t["obs"])
                ref = ent["tr_lm"]
                for fld, ours in (("Tr", t["obs"]), ("p_one", pge), ("p_two", ptwo), ("n_vit_constant", t["nconst"])):
                    n += 1
                    d = abs(float(ours) - float(ref[fld])) if ours is not None and ref[fld] is not None else (
                        0.0 if ours is None and ref[fld] is None else float("inf"))
                    worst = max(worst, d)
                    if not d <= 1e-12:
                        bad += 1
                        rows.append({"pre": p, "metric": m, "key": key, "field": fld, "ours": ours, "frozen": ref[fld]})
    return {"n_values": n, "n_differ": bad, "max_absdiff": worst, "differ": rows[:50], "pass": bad == 0}


def main():
    ap = argparse.ArgumentParser(description="plan V-P8 (a), (b), (c), implementation I (S0b test mode only)")
    ap.add_argument("--parts", default="a,b,c")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    os.makedirs(OUTD, exist_ok=True)
    hashes = {os.path.relpath(p, L.PROJ): {"sha256": sha(p), "expected": e, "equal": sha(p) == e} for p, e in SHA.items()}
    if not all(v["equal"] for v in hashes.values()):
        raise SystemExit(f"frozen input hash differs: {hashes}")
    res = {"what": "plan V-P8, implementation I", "inputs": hashes, "test_mode": True}
    t0 = time.time()
    if parts & {"a", "b"}:
        res.update(part_ab(parts))
    if "c" in parts:
        res["c"] = part_c()
        log(f"V-P8(c): {res['c']['n_values']} values, {res['c']['n_differ']} differ (max {res['c']['max_absdiff']:.2e})")
    res["pass"] = bool(all(res[k]["pass"] for k in ("b", "c") if k in res) and res.get("a_pass", True))
    res["secs"] = time.time() - t0
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    L.write_json(os.path.join(OUTD, f"vp8_{'_'.join(sorted(parts))}.json"), L.jsonable(res), default=str)
    log(f"V-P8 pass={res['pass']}")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
