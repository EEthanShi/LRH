"""Compare two Pythia-axis verdict files (VERDICTS_SPEC_PYT.md section 0). Comparator only, no computation.  # PYT-AXIS

PYT-AXIS copy of analysis/s12_layermax_openllama/compare_verdicts.py (sha256 233351a4...), written at S0b by role  # PYT-AXIS
pyt-code-infra (2026-10-10); hashed in the CODE FROZEN line. Changed lines carry PYT-AXIS; diff: compare_verdicts.py.diff.  # PYT-AXIS
usage: python compare_verdicts.py A.json B.json [--sections EI|FK|ALL|A,B,C,H,R,D,G,F,K] [--out report.json]  # PYT-AXIS
                                  [--a-sub DOTTED] [--b-sub DOTTED] [--legacy]  # PYT-AXIS
Exit 0 if 0 mismatches and no required key is missing on either side, else 1.
With --sections, exactly the required keys of those sections are compared (other keys are ignored);
without it, every key of either file. Keys starting with "x|" are never compared. Floats: |a-b| <= 1e-12*max(|a|,|b|) + 1e-15; NaN == NaN;
null == null; str/bool/int exact (an int and a float compare as floats only if both are numbers and
neither is a bool).
PYT-AXIS: NAMES ONLY. The printed output and the --out report carry counts and key names, never a value of
either file (plan section 11 V-P5: "prints mismatch counts and the names of mismatching keys only"; critique B6.4).  # PYT-AXIS
PYT-AXIS: --a-sub/--b-sub select the verdict dict inside a file by a dotted path (default "v"), e.g.
"embedded.verdicts.E" of results/S1_2_layermax_openllama/layermax_openllama_summary.json for V-P8(a).  # PYT-AXIS
PYT-AXIS: --legacy takes the required keys from the frozen OpenLLaMA verdict_keys.py (V-P8, n = 3, sections A,B,C).
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
    """PYT-AXIS: returns counts and key names only (no value of va or vb)."""
    miss_a = [k for k in required if k not in va]
    miss_b = [k for k in required if k not in vb]
    if required:   # compare exactly the required keys of the selected sections
        keys = sorted(set(required))
    else:          # no section given: every non-extra key of either file
        keys = sorted(k for k in set(va) | set(vb) if not k.startswith("x|"))
    only_a = [k for k in keys if k in va and k not in vb]
    only_b = [k for k in keys if k in vb and k not in va]
    mism = [k for k in keys if k in va and k in vb and not same(va[k], vb[k])]  # PYT-AXIS: names only
    return {"n_required": len(required), "n_compared": len([k for k in keys if k in va and k in vb]),
            "n_missing_in_a": len(miss_a), "n_missing_in_b": len(miss_b),  # PYT-AXIS
            "missing_in_a": miss_a[:500], "missing_in_b": miss_b[:500], "only_in_a": only_a[:500],  # PYT-AXIS
            "only_in_b": only_b[:500],  # PYT-AXIS
            "n_mismatch": len(mism), "mismatching_keys": mism[:2000],  # PYT-AXIS: names only, never values
            "mismatches": [{"key": k} for k in mism[:500]],  # PYT-AXIS: the OpenLLaMA field name, key names only
            "pass": not miss_a and not miss_b and not mism and not only_a and not only_b}


def subdict(doc: dict, dotted: str) -> dict:  # PYT-AXIS
    cur = doc  # PYT-AXIS
    for part in dotted.split("."):  # PYT-AXIS
        cur = cur[part]  # PYT-AXIS
    if not isinstance(cur, dict):  # PYT-AXIS
        raise SystemExit(f"--sub {dotted} does not select a dict")  # PYT-AXIS
    return cur  # PYT-AXIS


def required_for(secs: str | None, legacy: bool) -> list[str]:  # PYT-AXIS
    if secs is None:  # PYT-AXIS
        return []  # PYT-AXIS
    if legacy:  # PYT-AXIS: V-P8 (frozen OpenLLaMA key names)
        return [k for s in secs.split(",") for k in VK.legacy_required_keys(s)]  # PYT-AXIS
    if secs in ("EI", "FK", "ALL"):  # PYT-AXIS
        return VK.required_keys(secs)  # PYT-AXIS
    return [k for s in secs.split(",") for k in VK.required_keys(s)]  # PYT-AXIS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--sections", default=None, help="EI, FK, ALL or a comma list of A,B,C,H,R,D,G,F,K")  # PYT-AXIS
    ap.add_argument("--out", default=None)
    ap.add_argument("--a-sub", default="v", help="dotted path of the verdict dict in A (default v)")  # PYT-AXIS
    ap.add_argument("--b-sub", default="v", help="dotted path of the verdict dict in B (default v)")  # PYT-AXIS
    ap.add_argument("--legacy", action="store_true", help="V-P8: frozen OpenLLaMA key lists")  # PYT-AXIS
    args = ap.parse_args()
    A = json.load(open(args.a))
    B = json.load(open(args.b))
    required = required_for(args.sections, args.legacy)  # PYT-AXIS
    rep = compare(subdict(A, args.a_sub), subdict(B, args.b_sub), required)  # PYT-AXIS
    rep.update({"a": os.path.abspath(args.a), "b": os.path.abspath(args.b), "sections": args.sections,  # PYT-AXIS
                "a_sub": args.a_sub, "b_sub": args.b_sub, "legacy": args.legacy,  # PYT-AXIS
                "note": "names only; no value of either file is printed or written (plan section 11 V-P5)"})  # PYT-AXIS
    txt = json.dumps(rep, indent=1, default=str)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)
    print(json.dumps({k: rep[k] for k in ("n_required", "n_compared", "n_missing_in_a", "n_missing_in_b",  # PYT-AXIS
                                          "n_mismatch", "pass")}))  # PYT-AXIS
    if not rep["pass"]:
        print(txt[:4000], file=sys.stderr)  # PYT-AXIS: names only
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
