#!/usr/bin/env python
"""Pythia axis, implementation I: F1(a) gate tests and S0b dry runs (NEW, PYT-AXIS, sub-role pyt-code-I, 2026-10-10).
Runs only in S0b test mode (PYT_S0B_TEST=1, no CODE FROZEN line), else exit 3 with no output.

Part D (dry runs, plan 11 S0b "F1(a) dry runs exit 3"): every entry script of implementation I is started WITHOUT the
  test switch (as a real run would be, before CODE FROZEN / Part B FROZEN exist) and must exit 3 and write nothing:
  the trees analysis/s12_layermax_pythia/indep/, analysis/s12_layermax_pythia/s0b_scratch/I/ (minus this test's own
  logs/, gate_unit/, dryrun/) and results/S1_2_layermax_pythia/{indep,cache} are snapshotted (path, size, mtime)
  before and after each run. Also: test mode with an output root outside s0b_scratch/I/ must exit 3.
Part U (unit tests of pyt_gate.require with a synthetic STATUS file and synthetic Part B files under
  s0b_scratch/I/gate_unit/; the real STATUS.md is only read): missing CODE FROZEN, missing / earlier / quoted Part B line,
  missing hashes, wrong hash, missing file, unguarded output root, shared guard refusing / passing, the all-good case,
  and PYT_S0B_TEST=1 after CODE FROZEN.
Report: analysis/s12_layermax_pythia/s0b_scratch/I/dryrun/gate_report_<UTC>.json.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
INDEP = os.path.dirname(HERE)
sys.path.insert(0, INDEP)
import pyt_gate  # noqa: E402
if not pyt_gate.test_mode():
    pyt_gate._fail("test_gate_dryrun.py runs only in S0b test mode (PYT_S0B_TEST=1, before CODE FROZEN)")

import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import contextlib  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

PROJ = pyt_gate.PROJ
PY = os.path.join(PROJ, "venv", "bin", "python")
UTC = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
OUTD = os.path.join(pyt_gate.TEST_ROOT, "dryrun")
UNIT = os.path.join(pyt_gate.TEST_ROOT, "gate_unit", UTC)
EXCL = (os.path.join(pyt_gate.TEST_ROOT, "logs"), os.path.join(pyt_gate.TEST_ROOT, "gate_unit"), OUTD)
TREES = (INDEP, pyt_gate.TEST_ROOT, os.path.join(pyt_gate.RES_PYT, "indep"), os.path.join(pyt_gate.RES_PYT, "cache"))
SCRIPTS = [
    ("run_layermax.py", ["--pre", "prh_clip+l2"]),
    ("run_layermax.py", ["--pre", "l2", "--known"]),
    ("validate.py", []),
    ("vl_checks.py", []),
    ("vl6_arh.py", []),
    ("lmx_pyt_aggregate.py", []),
    ("lmx_pyt_report.py", []),
    ("cross_validate.py", ["--parts", "1,2,3", "--pres", "prh_clip+l2,l2"]),
    ("lmx_aggregate.py", []),
    ("vp8_legacy.py", []),
    ("pyt_gate.py", ["check"]),
    ("tests/test_pyt_verdicts.py", []),
    ("tests/test_gate_dryrun.py", []),
    ("tests/test_cross_report.py", ["/nonexistent"]),
    ("tests/test_forecast_xcheck.py", ["/nonexistent", "/nonexistent"]),
]


def snapshot():
    snap = {}
    for root in TREES:
        if not os.path.exists(root):
            snap[root] = "absent"
            continue
        for dp, dn, fn in os.walk(root):
            if any(dp == e or dp.startswith(e + os.sep) for e in EXCL):
                dn[:] = []
                continue
            for n in fn + dn:
                p = os.path.join(dp, n)
                try:
                    st = os.lstat(p)
                    snap[p] = (st.st_size, st.st_mtime_ns)
                except FileNotFoundError:
                    pass
    return snap


def dry_runs():
    env = {k: v for k, v in os.environ.items() if k not in ("PYT_S0B_TEST", "PYT_I_TEST_OUT")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    rows = []
    for script, args in SCRIPTS:
        before = snapshot()
        pr = subprocess.run([PY, os.path.join(INDEP, script)] + args, cwd=PROJ, env=env, capture_output=True,
                            text=True, timeout=600)
        after = snapshot()
        changed = sorted(set(before.items()) ^ set(after.items()))
        rows.append({"script": script, "args": args, "rc": pr.returncode, "n_changed_paths": len(changed),
                     "changed": [c[0] for c in changed][:20], "stderr_tail": pr.stderr.strip().splitlines()[-1:],
                     "pass": pr.returncode == 3 and not changed})
    env2 = dict(env)
    env2.update({"PYT_S0B_TEST": "1", "PYT_I_TEST_OUT": os.path.join(pyt_gate.RES_PYT, "indep")})
    for script, args in (("lmx_pyt_aggregate.py", []), ("run_layermax.py", ["--pre", "prh_clip+l2"])):
        before = snapshot()
        pr = subprocess.run([PY, os.path.join(INDEP, script)] + args, cwd=PROJ, env=env2, capture_output=True,
                            text=True, timeout=600)
        after = snapshot()
        changed = sorted(set(before.items()) ^ set(after.items()))
        rows.append({"script": script, "args": args, "mode": "test mode, output root outside the test root",
                     "rc": pr.returncode, "n_changed_paths": len(changed), "changed": [c[0] for c in changed][:20],
                     "stderr_tail": pr.stderr.strip().splitlines()[-1:], "pass": pr.returncode == 3 and not changed})
    return rows


def unit():
    os.makedirs(UNIT)
    res_dir = os.path.join(UNIT, "res")
    guarded = os.path.join(res_dir, "indep")
    os.makedirs(guarded)
    axis = os.path.join(UNIT, "axis")
    os.makedirs(axis)
    md, js = os.path.join(UNIT, "PARTB_PREDICTIONS_PYT.md"), os.path.join(UNIT, "partB_features_pyt.json")
    with open(md, "w") as f:
        f.write("stand-in\n")
    with open(js, "w") as f:
        f.write("{}\n")
    hmd = hashlib.sha256(open(md, "rb").read()).hexdigest()
    hjs = hashlib.sha256(open(js, "rb").read()).hexdigest()
    status = os.path.join(UNIT, "STATUS.md")
    saved = {k: getattr(pyt_gate, k) for k in ("STATUS", "PARTB_MD", "PARTB_JSON", "GUARDED_ROOTS", "AXIS",
                                               "TEST_ROOT", "_TEST")}
    cf = "2026-10-10T05:00:00Z [pyt-axis-freeze] CODE FROZEN manifest sha256 " + "0" * 64
    pbl = (f"2026-10-10T06:00:00Z [pyt-axis-predict] Part B FROZEN: PARTB_PREDICTIONS_PYT.md sha256 {hmd}; "
           f"partB_features_pyt.json sha256 {hjs}; partb_pyt.py sha256 " + "1" * 64)
    cases = [
        ("no CODE FROZEN line", [pbl], guarded, 3),
        ("no Part B line", [cf], guarded, 3),
        ("Part B line before CODE FROZEN", [pbl, cf], guarded, 3),
        ("quoted Part B string inside another line does not count",
         [cf, "2026-10-10T06:00:00Z [pyt-axis-review] notes: '[pyt-axis-predict] Part B FROZEN' not yet"], guarded, 3),
        ("Part B line without hashes", [cf, "2026-10-10T06:00:00Z [pyt-axis-predict] Part B FROZEN (no hashes)"],
         guarded, 3),
        ("wrong hash", [cf, pbl.replace(hmd, "f" * 64)], guarded, 3),
        ("output root not guarded", [cf, pbl], os.path.join(res_dir, "elsewhere"), 3),
        ("all good", [cf, pbl], guarded, 0),
        ("all good, colon-separated hash form", [cf, pbl.replace("PARTB_PREDICTIONS_PYT.md sha256 ",
                                                                  "PARTB_PREDICTIONS_PYT.md sha256: ")], guarded, 0),
        ("missing Part B file", [cf, pbl], guarded, "missing"),
        ("shared guard refuses", [cf, pbl], guarded, "guard_refuse"),
        ("shared guard passes", [cf, pbl], guarded, "guard_pass"),
    ]
    rows = []
    try:
        for name, lines, root, want in cases:
            with open(status, "w") as f:
                f.write("\n".join(lines) + "\n")
            pyt_gate.STATUS, pyt_gate.PARTB_MD, pyt_gate.PARTB_JSON = status, md, js
            pyt_gate.GUARDED_ROOTS = (guarded,)
            pyt_gate.AXIS = axis
            pyt_gate.TEST_ROOT = os.path.join(UNIT, "fake_test_root")
            pyt_gate._TEST = False
            gpath = os.path.join(axis, "guard.py")
            exp = want
            if want == "missing":
                pyt_gate.PARTB_JSON = js + ".absent"
                exp = 3
            if want in ("guard_refuse", "guard_pass"):
                with open(gpath, "w") as f:
                    f.write("import sys\n\ndef require_partb_frozen(root, script=None):\n" +
                            ("    sys.exit(3)\n" if want == "guard_refuse" else "    return {'ok': True}\n"))
                sys.modules.pop("guard", None)
                exp = 3 if want == "guard_refuse" else 0
            code, info = 0, None
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                try:
                    info = pyt_gate.require(root, "unit")
                except SystemExit as e:
                    code = e.code
            if os.path.exists(gpath):
                os.rename(gpath, gpath + f".used-{len(rows)}")   # kept (no-delete rule), out of the way
                sys.modules.pop("guard", None)
                if axis in sys.path:
                    sys.path.remove(axis)
            ok = code == exp and (exp != 0 or (info is not None and info.get("mode") == "partB-frozen"
                                               and (want != "guard_pass" or info.get("shared_guard_called"))))
            rows.append({"case": name, "exit": code, "expected": exp, "info": info, "stderr": err.getvalue().strip()[-300:],
                         "pass": bool(ok)})
        # PYT_S0B_TEST=1 after CODE FROZEN -> exit 3
        with open(status, "w") as f:
            f.write(cf + "\n")
        pyt_gate.STATUS = status
        pyt_gate._TEST = None
        code = 0
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                pyt_gate.test_mode()
            except SystemExit as e:
                code = e.code
        rows.append({"case": "PYT_S0B_TEST=1 after CODE FROZEN", "exit": code, "expected": 3, "pass": code == 3})
    finally:
        for k, v in saved.items():
            setattr(pyt_gate, k, v)
    return rows


def main():
    os.makedirs(OUTD, exist_ok=True)
    t0 = time.time()
    rep = {"what": "F1(a) gate unit tests and S0b dry runs, implementation I", "utc": UTC}
    rep["dry_runs"] = dry_runs()
    rep["unit"] = unit()
    rep["n_dry"] = len(rep["dry_runs"])
    rep["n_dry_pass"] = sum(r["pass"] for r in rep["dry_runs"])
    rep["n_unit"] = len(rep["unit"])
    rep["n_unit_pass"] = sum(r["pass"] for r in rep["unit"])
    rep["pass"] = rep["n_dry_pass"] == rep["n_dry"] and rep["n_unit_pass"] == rep["n_unit"]
    rep["secs"] = time.time() - t0
    fn = os.path.join(OUTD, f"gate_report_{UTC}.json")
    with open(fn, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    for r in rep["dry_runs"]:
        print(f"dry run {r['script']} {' '.join(r['args'])}{' [' + r['mode'] + ']' if 'mode' in r else ''}: rc {r['rc']}, "
              f"{r['n_changed_paths']} paths changed -> {'PASS' if r['pass'] else 'FAIL'}")
    for r in rep["unit"]:
        print(f"unit {r['case']}: exit {r['exit']} (expected {r['expected']}) -> {'PASS' if r['pass'] else 'FAIL'}")
    print(f"gate tests: dry runs {rep['n_dry_pass']}/{rep['n_dry']}, unit {rep['n_unit_pass']}/{rep['n_unit']}; "
          f"pass={rep['pass']}; report {fn}")
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
