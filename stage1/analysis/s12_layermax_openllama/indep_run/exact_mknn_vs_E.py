#!/usr/bin/env python
"""OpenLLaMA axis, role oll-run-indep: compare the integer-only mKNN values of exact_mknn_check_I.py (I's pair
files, ints and cross-multiplication only) with E's verdicts_ol.json, key by key, with no float tolerance on
anything that is a count and |d| <= 1e-12 on trends (one order change moves a trend by >= 1/34). Run only after
I's summary.json / verdicts_ol.json were final and cross_validate.py had opened E (plan V-O5 order).
NEW helper, not in the S0b manifest; reads two JSON files, writes indep/exact_mknn_vs_E.json.

usage: analysis/s12_layermax_openllama/launch_ol.sh ol_I_exact_vs_E 1 -- venv/bin/python -I \
         analysis/s12_layermax_openllama/indep_run/exact_mknn_vs_E.py
"""
import os
import sys

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
sys.path.insert(0, os.path.join(PROJ, "analysis", "s12_layermax_openllama"))
import guard  # noqa: E402

RES = os.path.join(PROJ, "results", "S1_2_layermax_openllama")
OUT = os.path.join(RES, "indep")
guard.require_partb_frozen(OUT, script="indep_run/exact_mknn_vs_E.py")   # F1(a): first action

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    fi = os.path.join(OUT, "exact_mknn_check_I.json")
    fe = os.path.join(RES, "early", "verdicts_ol.json")
    X = json.load(open(fi))
    VE = json.load(open(fe))["v"]
    n = bad = 0
    fails = []
    for k, a in X["values"].items():
        n += 1
        b = VE.get(k, "MISSING")
        if isinstance(a, float) or isinstance(b, float):
            ok = (a is None and b is None) or (a is not None and b not in (None, "MISSING")
                                                and abs(float(a) - float(b)) <= 1e-12)
        else:
            ok = a == b
        if not ok:
            bad += 1
            fails.append([k, a, b])
    tied = {k: v for k, v in X["exact_ties_per_pre_Q_X_of_51"].items() if v > 0}
    res = {"what": "integer-only mKNN values (from I's pair files) vs E verdicts_ol.json",
           "inputs": {os.path.relpath(fi, PROJ): sha(fi), os.path.relpath(fe, PROJ): sha(fe)},
           "n_compared": n, "n_mismatch": bad, "mismatches": fails[:50], "pass": bad == 0,
           "keys_with_exact_ties": tied, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    fn = os.path.join(OUT, "exact_mknn_vs_E.json")
    with open(fn + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(fn + ".tmp", fn)
    print(json.dumps({k: res[k] for k in ("n_compared", "n_mismatch", "pass")}), flush=True)
    print(json.dumps(tied), flush=True)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
