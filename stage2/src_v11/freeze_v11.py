"""S2.2 v1.1 freeze (PREREG_S2_2_v1.1.md section 14, step 5). Run ONCE, before any engineering, pilot or main run.

    python src_v11/freeze_v11.py

Preconditions (refuses otherwise): no STATUS '[s22v11-freeze]' line yet; runs_v11/ holds only EXCLUDED unit-test
directories (UT_*, _archive_*); plans_v11/ holds no cohort or engineering file; the unit tests, KA0-KA5 and the
dry-run summary exist and pass. Writes
  results/S2_2_v11/freeze/CODE_AT_PREREG_v11.sha256   (guard manifest: prereg, src_v11/*.py, v1.0 src copies'
                                                       originals, hrhv2 harness) chmod 444
  results/S2_2_v11/freeze/TEST_RESULTS_AT_FREEZE.sha256 (record: KA6, unit tests, KA0-KA5, dry run) chmod 444
chmods the prereg 444, appends ONE STATUS line '[s22v11-freeze] PREREG v1.1 + CODE FROZEN' quoting both files'
sha256 and every listed file's sha256, then checks guard_v11.verify_frozen().
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
S1 = os.path.expanduser("~/hrh_v2_stage1_20261008")
STATUS = os.path.join(S1, "STATUS.md")
FRZ = os.path.join(ROOT, "results", "S2_2_v11", "freeze")
PREREG = os.path.join(ROOT, "prereg", "PREREG_S2_2_v1.1.md")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rel(p):
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p


def main():
    txt = open(STATUS).read()
    if "[s22v11-freeze]" in txt:
        raise SystemExit("a [s22v11-freeze] STATUS line exists already: the freeze runs once")
    extra = [d for d in os.listdir(os.path.join(ROOT, "runs_v11")) if not (d.startswith("UT_") or d.startswith("_archive"))]
    if extra:
        raise SystemExit(f"runs_v11 holds non-unit-test entries {extra}: refusing to freeze after runs")
    pl = [f for f in os.listdir(os.path.join(ROOT, "plans_v11")) if not f.startswith("UT_")] \
        if os.path.isdir(os.path.join(ROOT, "plans_v11")) else []
    if pl:
        raise SystemExit(f"plans_v11 holds {pl}: refusing to freeze after engineering / cohort plans")
    res = os.path.join(ROOT, "results", "S2_2_v11")
    ut = json.load(open(os.path.join(res, "unittest", "unit_tests_all.json")))["results"]
    ka = {os.path.basename(p): json.load(open(p))["pass"] for p in sorted(glob.glob(os.path.join(res, "ka", "KA*.json")))}
    need = ["KA0_reduced_equals_full.json", "KA1_decomp_reproduces_a2b.json", "KA2_cor3_reproduces_v10.json",
            "KA3_lemmaN1_v10.json", "KA4_decisions_curves_reproduce_v10.json", "KA5_plants_and_I4.json"]
    dry = os.path.join(res, "dryrun", "s22v11_summary.json")
    if not all(v["pass"] for v in ut.values()) or len(ut) != 8:
        raise SystemExit(f"unit tests not all passing: {ut.keys()}")
    if not all(ka.get(n) for n in need):
        raise SystemExit(f"KA tests not all passing: {ka}")
    if not os.path.exists(dry):
        raise SystemExit("dry-run summary absent")
    est = os.path.join(res, "eng_selftest", "eng_selftest.json")
    if not json.load(open(est))["pass"]:
        raise SystemExit("engineering decision self-test not passing")
    code = [PREREG] + sorted(glob.glob(os.path.join(HERE, "*.py"))) + \
        [os.path.join(ROOT, "src", "crit.py"), os.path.join(ROOT, "src", "s2lib.py")] + \
        sorted(glob.glob(os.path.join(S1, "hrhv2", "*.py")))
    tests = [os.path.join(res, "ka6", "ka6_v10.json"), os.path.join(res, "unittest", "unit_tests_all.json")] + \
        [os.path.join(res, "ka", n) for n in need] + \
        [est] + \
        [dry, os.path.join(res, "dryrun", "S2_2_v11_report.md"), os.path.join(res, "dryrun", "eps", "eps_S2_2_v11.json")] + \
        sorted(glob.glob(os.path.join(ROOT, "plans_v11", "UT_*.json")))
    os.makedirs(FRZ, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    man = os.path.join(FRZ, "CODE_AT_PREREG_v11.sha256")
    rec = os.path.join(FRZ, "TEST_RESULTS_AT_FREEZE.sha256")
    for path, files, head in ((man, code, "guard manifest (verified by src_v11/guard_v11.py before every v1.1 step)"),
                              (rec, tests, "record of the test outputs at freeze (not guard-checked)")):
        if os.path.exists(path):
            raise SystemExit(f"{path} exists")
        with open(path, "w") as f:
            f.write(f"# S2.2 v1.1 {head}; written {stamp} by src_v11/freeze_v11.py\n")
            for p in files:
                f.write(f"{sha(p)}  {rel(p)}\n")
        os.chmod(path, 0o444)
    os.chmod(PREREG, 0o444)
    hm, hr = sha(man), sha(rec)
    k6 = json.load(open(os.path.join(res, "ka6", "ka6_v10.json")))["summary"]
    ka6s = "; ".join(f"{c} median mu_C {v['mu_C_median']:+.3f} [{v['mu_C_median_ci95'][0]:+.3f},{v['mu_C_median_ci95'][1]:+.3f}] "
                     f"own-closer {v['frac_own_closer_v10def']['x']}/{v['frac_own_closer_v10def']['n']}" for c, v in k6.items())
    files = "; ".join(f"{rel(p)} {sha(p)}" for p in code + tests)
    line = (f"{stamp} [s22v11-freeze] PREREG v1.1 + CODE FROZEN (before any engineering, pilot or main run; runs_v11 holds only "
            f"EXCLUDED unit-test runs): manifest {rel(man)} sha256 {hm}; test record {rel(rec)} sha256 {hr}; "
            f"unit tests U1-U8 PASS; KA0-KA5 PASS; engineering-rule self-test PASS; dry run (EXCLUDED UT cohorts) complete; KA6 (v1.0 READ-ONLY, before freeze): "
            f"{ka6s}; files: {files}; next: src_v11/eng_v11.py run (EXCLUDED engineering under the guard), then "
            f"src_v11/pipeline_v11.py run")
    with open(STATUS, "a") as f:
        f.write(line + "\n")
    import guard_v11
    ok, msg = guard_v11.verify_frozen()
    print(line)
    print("guard:", ok, msg)
    if not ok:
        raise SystemExit("guard does not verify after the freeze")


if __name__ == "__main__":
    main()
