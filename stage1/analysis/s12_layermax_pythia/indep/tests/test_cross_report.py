#!/usr/bin/env python
"""Pythia axis, implementation I: S0b test of cross_validate.py (V-P5, counts and key names only) and lmx_pyt_report.py
on the 7-size stand-in fixture output (NEW, PYT-AXIS, sub-role pyt-code-I, 2026-10-10). S0b test mode only.
  c1 cross_validate --parts 2 with E := an identical copy of I's verdicts      -> 0 mismatches, pass (shared comparator,
     --sections EI)
  c2 cross_validate --parts 2 with E := a perturbed copy (one float changed by 1e-9 relative, one string changed,
     one required key removed, one extra key added)                          -> 2 mismatches, 1 key missing in E, pass
     false; the written files and stdout contain no value of any compared key
  c3 cross_validate --parts 1 (no E pair files exist)                         -> reported missing, pass false, rc 0
  c4 lmx_pyt_report.py writes REPORT.md (a second run keeps the first aside)
usage: PYT_S0B_TEST=1 venv/bin/python analysis/s12_layermax_pythia/indep/tests/test_cross_report.py <fixture_out dir>
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
    pyt_gate._fail("test_cross_report.py runs only in S0b test mode (PYT_S0B_TEST=1, before CODE FROZEN)")

import glob  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

PY = os.path.join(pyt_gate.PROJ, "venv", "bin", "python")


def run(args, out_root):
    env = dict(os.environ)
    env["PYT_I_TEST_OUT"] = out_root
    return subprocess.run([PY] + args, cwd=pyt_gate.PROJ, env=env, capture_output=True, text=True, timeout=900)


def main():
    out_root = os.path.realpath(sys.argv[1])
    utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    td = os.path.join(pyt_gate.TEST_ROOT, "cross_test", utc)
    os.makedirs(td)
    ver = json.load(open(os.path.join(out_root, "verdicts_pyt.json")))
    same = os.path.join(td, "E_identical.json")
    json.dump(ver, open(same, "w"))
    pert = json.loads(json.dumps(ver))
    v = pert["v"]
    kf = next(k for k, x in v.items() if k.startswith("dtr|prh_clip+l2|") and isinstance(x, float) and x != 0.0)
    ks = "H|prh_clip+l2|s_cal|gate"
    kr = next(k for k in v if k.startswith("tr0|prh_clip+l2|"))
    assert len({kf, ks, kr}) == 3 and isinstance(v[ks], str)
    secret_f, secret_s = v[kf], v[ks]
    v[kf] = secret_f * (1 + 1e-9)
    v[ks] = secret_s + " PERTURBED-SENTINEL"
    del v[kr]
    v["zz|prh_clip+l2|added"] = 1.0
    pfile = os.path.join(td, "E_perturbed.json")
    json.dump(pert, open(pfile, "w"))
    rep = {"tests": {}}
    cv = os.path.join(INDEP, "cross_validate.py")
    r1 = run([cv, "--parts", "2", "--pres", "prh_clip+l2,l2", "--e-verdicts", same], out_root)
    d1 = json.load(open(os.path.join(out_root, "validation_cross_2_prh_clip+l2+l2.json")))["part2_verdict_level"]
    rep["tests"]["c1"] = {"rc": r1.returncode, "n_compared": d1.get("n_compared"), "n_mismatch": d1.get("n_mismatch"),
                          "pass": r1.returncode == 0 and d1["pass"] and d1.get("n_mismatch") == 0}
    r2 = run([cv, "--parts", "2", "--pres", "prh_clip+l2,l2", "--e-verdicts", pfile], out_root)
    f2 = os.path.join(out_root, "validation_cross_2_prh_clip+l2+l2.json")
    d2 = json.load(open(f2))["part2_verdict_level"]
    txt = open(f2).read() + open(d2["report_names_only"]).read()
    leak = (repr(secret_f) in txt or repr(secret_f) in r2.stdout + r2.stderr or "PERTURBED-SENTINEL" in txt
            or "PERTURBED-SENTINEL" in r2.stdout + r2.stderr)
    names = json.load(open(d2["report_names_only"]))
    rep["tests"]["c2"] = {"rc": r2.returncode, "n_mismatch": d2.get("n_mismatch"), "n_missing_in_E": d2.get("n_missing_in_E"),
                          "mismatch_names": names.get("mismatches", names.get("mismatching_keys")),
                          "value_leak": leak,
                          "pass": (r2.returncode == 0 and d2.get("n_mismatch") == 2 and d2.get("n_missing_in_E") == 1
                                   and not d2["pass"] and not leak)}
    r3 = run([cv, "--parts", "1", "--pres", "prh_clip+l2"], out_root)
    f3 = os.path.join(out_root, "validation_cross_1_prh_clip+l2.json")
    d3 = json.load(open(f3))["part1_pair_level"]
    rep["tests"]["c3"] = {"rc": r3.returncode, "n_missing": d3["n_missing_E_pair_files"],
                          "pass": r3.returncode == 0 and d3["n_missing_E_pair_files"] == 119 and not d3["pass"]}
    rp = os.path.join(INDEP, "lmx_pyt_report.py")
    r4a = run([rp], out_root)
    r4b = run([rp], out_root)
    prev = glob.glob(os.path.join(out_root, "REPORT.md.prev-*"))
    rep["tests"]["c4"] = {"rc": [r4a.returncode, r4b.returncode], "n_prev": len(prev),
                          "pass": r4a.returncode == 0 and r4b.returncode == 0 and len(prev) >= 1
                          and os.path.exists(os.path.join(out_root, "REPORT.md"))}
    rep["pass"] = all(t["pass"] for t in rep["tests"].values())
    rep["stderr_tails"] = {k: r.stderr.strip().splitlines()[-3:] for k, r in (("c1", r1), ("c2", r2), ("c3", r3),
                                                                            ("c4", r4a))}
    fn = os.path.join(td, "cross_report_test.json")
    json.dump(rep, open(fn, "w"), indent=1, default=str)
    for k, t in rep["tests"].items():
        print(k, "PASS" if t["pass"] else "FAIL", {kk: vv for kk, vv in t.items() if kk != "pass"})
    print(f"cross/report tests pass={rep['pass']}; report {fn}")
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
