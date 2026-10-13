#!/usr/bin/env python
"""Pythia axis, implementation I: S0b cross-check of I's own P-P2b and plan 7 category (x|I| extras of
verdicts_pyt.json) against forecast_pyt.py --impl I run on the same I output (sections F and K of VERDICTS_SPEC_PYT.md)
(NEW, PYT-AXIS, sub-role pyt-code-I, 2026-10-10). forecast_pyt.py is used as a black box (its code is not read by I).
S0b test mode only. Counts and key names only.
usage: PYT_S0B_TEST=1 venv/bin/python analysis/s12_layermax_pythia/indep/tests/test_forecast_xcheck.py
       <I verdicts_pyt.json> <forecast verdicts_forecast.json>
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pyt_gate  # noqa: E402
if not pyt_gate.test_mode():
    pyt_gate._fail("test_forecast_xcheck.py runs only in S0b test mode (PYT_S0B_TEST=1, before CODE FROZEN)")

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-12 * max(abs(float(a)), abs(float(b))) + 1e-15
    return a == b


def main():
    vi = json.load(open(sys.argv[1]))["v"]
    vf = json.load(open(sys.argv[2]))["v"]
    pairs = []
    for k in vf:
        f = k.split("|")
        if f[0] == "PP2b":
            mk = "x|I|" + k
        elif f[0] == "K" and f[3] in ("category", "premise", "F_reason", "M1", "M2", "M3", "M4", "M4_sizes",
                                       "M5_extrap_sizes", "M5_B", "M5_A", "M5_A_sizes", "H2_reading", "H3_reading",
                                       "H4_reading", "curve"):
            mk = "x|I|" + k
        else:
            continue
        pairs.append((k, mk))
    present = [(k, mk) for k, mk in pairs if mk in vi]
    absent = [k for k, mk in pairs if mk not in vi]
    mism = [k for k, mk in present if not same(vf[k], vi[mk])]
    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "I": os.path.abspath(sys.argv[1]),
           "forecast": os.path.abspath(sys.argv[2]), "n_forecast_keys_mapped": len(pairs), "n_compared": len(present),
           "n_without_I_counterpart": len(absent), "without_I_counterpart": absent[:100],
           "n_mismatch": len(mism), "mismatching_keys": mism[:300], "pass": not mism}
    out = os.path.join(pyt_gate.TEST_ROOT, "forecast_xcheck", rep["utc"].replace(":", "").replace("-", ""))
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, "forecast_xcheck.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("n_forecast_keys_mapped", "n_compared", "n_without_I_counterpart",
                                          "n_mismatch", "pass")}))
    if mism:
        print("mismatching keys (names only):", ", ".join(mism[:60]))
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
