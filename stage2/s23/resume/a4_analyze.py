#!/usr/bin/env python3
"""S2.3 AMENDMENT A4 analysis entry point (prereg/AMENDMENT_A4_p5b_zero.md, rules A4.1-A4.2).

Lives in s23/resume/, NEVER in src/. It runs the FROZEN src/analyze_A.py with exactly two differences, both in
this process only (the file on disk and its hash are untouched; the frozen main() re-verifies the 37 code hashes):

1. (A3.1, unchanged) Store.K is replaced by the corrected memo of the frozen resume/a3_analyze.py (K_memo).
2. (A4.1) One expression of phase_main (line 705, secondary item P5b 'mc_rho_vs_obs') divides by the frozen MC
   prediction mcr[r_]["rho"]. frozen_predictions.json has pi_mc["40"]["dMST"]["rho"] = 0.0 (a pre-data constant),
   so the frozen line raises ZeroDivisionError whatever the data. The module is built from the frozen source text
   (sha256 checked) with that ONE expression replaced by a call to _a4_within_15(obs, mc), which evaluates the
   frozen expression verbatim when mc != 0 and, when mc == 0, returns obs == 0 (the band [mc/1.5, 1.5*mc] is {0}).
   Line count and every other line are unchanged.

  python a4_analyze.py --phase nmatch|main     real run (A4 launcher only; cwd src/)
  python a4_analyze.py --selftest              excluded unit test (R29): source diff and the helper; no data
"""
from __future__ import annotations

import difflib
import hashlib
import math
import os
import sys
import types

HOME = os.path.expanduser("~")
S23 = os.path.join(HOME, "hrh_v2_stage2_20261008", "s23")
SRC = os.path.join(S23, "src")
RES = os.path.join(S23, "resume")
DEV = os.path.join(os.path.realpath(S23), "dev") + os.sep
if "S23_ROOT" in os.environ and not os.path.realpath(os.environ["S23_ROOT"]).startswith(DEV):
    raise SystemExit("A4 analyze: a test root must lie under s23/dev/")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import s23common as C  # noqa: E402  (frozen)

C.cap_threads(1)  # as the frozen analyze_A does under __main__, before numpy is imported

import numpy as np  # noqa: E402,F401

FROZEN_PATH = os.path.join(SRC, "analyze_A.py")
FROZEN_SHA = "e43e4a8e5be1ed8f6e3092d43646bf2df3ceb1c1f18eaabb069296a5fe3e522a"
OLD = 'and 1 / 1.5 <= pis[r_]["rho"] / mcr[r_]["rho"] <= 1.5)} for r_ in ORDER}'
NEW = 'and _a4_within_15(pis[r_]["rho"], mcr[r_]["rho"]))} for r_ in ORDER}'
LINE = 705


def _a4_within_15(obs, mc):
    """Rule A4.1: the frozen test 1/1.5 <= obs/mc <= 1.5, defined also for mc == 0."""
    if mc != 0:
        return 1 / 1.5 <= obs / mc <= 1.5  # the frozen expression, verbatim
    return obs == 0  # mc == 0: the factor-1.5 band [mc/1.5, 1.5*mc] is the single point 0


def _load_patched():
    raw = open(FROZEN_PATH, "rb").read()
    if hashlib.sha256(raw).hexdigest() != FROZEN_SHA:
        raise SystemExit("A4 analyze: src/analyze_A.py is not the frozen file")
    text = raw.decode("utf-8")
    if text.count(OLD) != 1:
        raise SystemExit("A4 analyze: the frozen P5b expression was not found exactly once")
    new = text.replace(OLD, NEW)
    mod = types.ModuleType("analyze_A")
    mod.__file__ = FROZEN_PATH
    mod.__dict__["_a4_within_15"] = _a4_within_15
    sys.modules["analyze_A"] = mod
    exec(compile(new, FROZEN_PATH, "exec"), mod.__dict__)
    return mod, text, new


A, _FROZEN_TEXT, _PATCHED_TEXT = _load_patched()
if RES not in sys.path:
    sys.path.insert(0, RES)
import a3_analyze as A3  # noqa: E402  (frozen A3 file; imports the module registered above)

if A3.A is not A or A3.C is not C:
    raise SystemExit("A4 analyze: a3_analyze did not bind the A4 module")


def selftest() -> int:
    fails = 0

    def check(ok, msg):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + msg, flush=True)
        fails += 0 if ok else 1

    fl, pl = _FROZEN_TEXT.splitlines(), _PATCHED_TEXT.splitlines()
    changed = [i + 1 for i, (x, y) in enumerate(zip(fl, pl)) if x != y]
    check(len(fl) == len(pl) and changed == [LINE], f"patched source differs from the frozen file only on line(s) {changed} "
          f"(expected [{LINE}]); {len(fl)} lines each")
    d = [l for l in difflib.unified_diff(fl, pl, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    print("\n".join("   " + l.strip() for l in d), flush=True)
    vals = [-3.0, -1.5, -1.0, -0.7, -0.5, 0.0, 0.3, 0.6666, 0.67, 1.0, 1.2, 1.5, 1.51, 2.0, 1e-12, -1e-12, math.inf,
            -math.inf, math.nan]
    same = True
    for mc in [v for v in vals if v != 0 and v == v]:
        for obs in vals:
            same &= _a4_within_15(obs, mc) == (1 / 1.5 <= obs / mc <= 1.5)
    check(same, "for every mc != 0 the helper equals the frozen expression (grid incl. negatives, inf, nan obs)")
    z = {obs: _a4_within_15(obs, 0.0) for obs in vals}
    check(all(z[o] == (o == 0) for o in vals), "mc == 0: within iff obs == 0; no exception")
    try:
        1 / 1.5 <= 0.5 / 0.0 <= 1.5
        raised = False
    except ZeroDivisionError:
        raised = True
    check(raised, "the frozen expression raises ZeroDivisionError for mc == 0.0 (Python float)")
    check(A.Store.K is A3.FROZEN_K and A3.K_memo is not A.Store.K, "Store.K of the built module is the frozen memo until A3.1 is applied")
    print(f"A4 selftest: {fails} failures", flush=True)
    return 1 if fails else 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) != 3 or sys.argv[1] != "--phase" or sys.argv[2] not in ("nmatch", "main"):
        raise SystemExit("usage: a4_analyze.py --phase nmatch|main | --selftest")
    if os.path.realpath(os.getcwd()) != os.path.realpath(SRC):
        raise SystemExit("A4 analyze: run from s23/src (the frozen command lines' cwd)")
    A.Store.K = A3.K_memo  # rule A3.1, in this process only
    print(f"{C.utcnow()} A4 analyze --phase {sys.argv[2]}: frozen src/analyze_A.py text with line {LINE} (P5b mc_rho_vs_obs) "
          f"defined for mc == 0 (rule A4.1) and the A3.1 memo (bound {A3.CAP}); test mode {C.TEST_MODE}", flush=True)
    sys.argv = [FROZEN_PATH, "--phase", sys.argv[2]]
    A.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
