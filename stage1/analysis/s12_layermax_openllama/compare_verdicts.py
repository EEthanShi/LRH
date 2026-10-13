"""Compare two OpenLLaMA-axis verdict files (VERDICTS_SPEC.md section 0). Comparator only, no computation.

Written at S0b by role oll-axis-extract (2026-10-09); hashed in the CODE FROZEN line.
usage: python compare_verdicts.py A.json B.json [--sections EI|F|J|ALL|A,B,...] [--out report.json]
Exit 0 if 0 mismatches and no required key is missing on either side, else 1.
With --sections, exactly the required keys of those sections are compared (other keys are ignored);
without it, every key of either file. Keys starting with "x|" are never compared. Floats: |a-b| <= 1e-12*max(|a|,|b|) + 1e-15; NaN == NaN;
null == null; str/bool/int exact (an int and a float compare as floats only if both are numbers and
neither is a bool).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verdict_keys as VK  # noqa: E402

RTOL, ATOL = 1e-12, 1e-15


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return math.isnan(fa) and math.isnan(fb)
        if math.isinf(fa) or math.isinf(fb):
            return fa == fb
        return abs(fa - fb) <= RTOL * max(abs(fa), abs(fb)) + ATOL
    return type(a) is type(b) and a == b


def compare(va: dict, vb: dict, required: list[str]) -> dict:
    miss_a = [k for k in required if k not in va]
    miss_b = [k for k in required if k not in vb]
    if required:   # compare exactly the required keys of the selected sections
        keys = sorted(set(required))
    else:          # no section given: every non-extra key of either file
        keys = sorted(k for k in set(va) | set(vb) if not k.startswith("x|"))
    only_a = [k for k in keys if k in va and k not in vb]
    only_b = [k for k in keys if k in vb and k not in va]
    mism = [{"key": k, "a": va[k], "b": vb[k]} for k in keys if k in va and k in vb and not same(va[k], vb[k])]
    return {"n_required": len(required), "n_compared": len([k for k in keys if k in va and k in vb]),
            "missing_in_a": miss_a, "missing_in_b": miss_b, "only_in_a": only_a, "only_in_b": only_b,
            "n_mismatch": len(mism), "mismatches": mism[:500],
            "pass": not miss_a and not miss_b and not mism and not only_a and not only_b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--sections", default=None, help="EI, F, J, ALL or a comma list of A,B,C,D,G,F,J")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A = json.load(open(args.a))
    B = json.load(open(args.b))
    secs = args.sections
    if secs is None:
        required = []
    elif secs in ("EI", "ALL", "F", "J"):
        required = VK.required_keys(secs)
    else:
        required = [k for s in secs.split(",") for k in VK.required_keys(s)]
    rep = compare(A["v"], B["v"], required)
    rep.update({"a": os.path.abspath(args.a), "b": os.path.abspath(args.b), "sections": secs})
    txt = json.dumps(rep, indent=1, default=str)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)
    print(json.dumps({k: rep[k] for k in ("n_required", "n_compared", "n_mismatch", "pass")}))
    if not rep["pass"]:
        print(txt[:4000], file=sys.stderr)
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
