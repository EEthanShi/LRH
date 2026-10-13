"""S2.2 v1.1 unattended driver: pilots -> gates + eps freeze -> main runs -> final analysis (PREREG_S2_2_v1.1.md
sections 6-8 and 14, steps 6-8).

    setsid nohup python src_v11/pipeline_v11.py run [--workers 4] [--jobs 6] > results/S2_2_v11/logs/pipeline.log 2>&1 &
    python src_v11/pipeline_v11.py dryrun [--workers 2] [--jobs 6] [--max-steps 20]   # EXCLUDED unit-test cohorts

run: refuses unless the code is frozen (guard_v11) and plans_v11/ENG_GO exists; writes the cohort plans with
make_plan_v11 (deterministic function of plans_v11/eng_choice.json) if absent; starts W training workers (1 thread each)
walking the pilot plan and then the main plan (main runs wait for runs_v11/GATE_S2_2v11_main.go); when the pilot cohort
has no pending run, re-queues failed pilot runs once, runs gates_v11 (gate file first, then the eps freeze from pilot
pairs only); when the main cohort has no pending run, re-queues failed runs once, then runs analyze_v11 once.
Between those points it reads only .done/.failed markers and the diverged / finite flags of meta.json (counts only:
no INTERIM analysis, no rho or any C statistic is printed).
dryrun: the same chain on the EXCLUDED unit-test plans (make_plan_v11 unittest; exp keys S2.2v11-unittest*; runs
with --max-steps); outputs under runs_v11/UT_*/ and results/S2_2_v11/dryrun/ (never evidence).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
PY = os.path.expanduser("~/hrh_v2_stage1_20261008/venv/bin/python")
PLANS = os.path.join(ROOT, "plans_v11")
RUNS = os.path.join(ROOT, "runs_v11")
LOGS = os.path.join(ROOT, "results", "S2_2_v11", "logs")
STATUS = os.path.expanduser("~/hrh_v2_stage1_20261008/STATUS.md")
ENV1 = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def status(msg):
    with open(STATUS, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [s22v11-run] {msg}\n")


def state(cohort):
    runs = json.load(open(os.path.join(PLANS, f"{cohort}.json")))["runs"]
    c = {"n": len(runs), "done": 0, "failed": 0, "pending": 0, "diverged_or_nonfinite": 0}
    for r in runs:
        d = os.path.join(RUNS, cohort, r["run_id"])
        if os.path.exists(os.path.join(d, ".done")):
            c["done"] += 1
            try:
                m = json.load(open(os.path.join(d, "meta.json")))
                c["diverged_or_nonfinite"] += int(bool(m.get("diverged")) or not m.get("finite", False))
            except Exception:
                pass
        elif os.path.exists(os.path.join(d, ".failed")) and not os.path.exists(os.path.join(d, ".lock")):
            c["failed"] += 1
        else:
            c["pending"] += 1
    return c


def start_workers(plans, nw, extra=(), tag="w"):
    os.makedirs(LOGS, exist_ok=True)
    procs = []
    for w in range(nw):
        lf = open(os.path.join(LOGS, f"{tag}{w}.log"), "a")
        procs.append(subprocess.Popen(["nice", "-n", "5", PY, os.path.join(HERE, "train_v11.py"), "worker", "--wid", str(w),
                                       "--plan", *plans, *extra], cwd=ROOT, env=ENV1, stdout=lf,
                                      stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(2)
    return procs


def wait_cohort(cohort, every=120, max_h=30):
    t0 = time.time()
    while True:
        c = state(cohort)
        if c["pending"] == 0:
            return c
        if time.time() - t0 > max_h * 3600:
            return c
        time.sleep(every)


def retry_failed(cohort, plan_path, extra=()):
    c = state(cohort)
    if c["failed"]:
        log(f"{cohort}: {c['failed']} failed runs -> one retry pass")
        with open(os.path.join(LOGS, f"retry_{cohort}.log"), "a") as f:
            subprocess.run([PY, os.path.join(HERE, "train_v11.py"), "worker", "--wid", "90", "--plan", plan_path, *extra],
                           cwd=ROOT, env=ENV1, stdout=f, stderr=subprocess.STDOUT)
    return state(cohort)


def py(script, *args, logname):
    with open(os.path.join(LOGS, logname), "a") as f:
        return subprocess.run([PY, os.path.join(HERE, script), *args], cwd=ROOT, env=ENV1, stdout=f,
                              stderr=subprocess.STDOUT).returncode


def gpu_free_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return float(out.split()[0]) / 1024
    except Exception:
        return float("nan")


def chain(pilot, main, nw, jobs, extra=(), dry=False):
    pp, mp = os.path.join(PLANS, f"{pilot}.json"), os.path.join(PLANS, f"{main}.json")
    free = gpu_free_gb()
    log(f"GPU free {free:.1f} GB; starting {nw} workers on {pilot} then {main}")
    if free == free and free < 5.0 * nw:
        raise SystemExit(f"not enough free GPU memory for {nw} workers ({free:.1f} GB); nothing started")
    procs = start_workers([pp] if dry else [pp, mp], nw, extra, tag=("ut_w" if dry else "w"))
    c = wait_cohort(pilot, every=20 if dry else 120)
    c = retry_failed(pilot, pp, extra)
    log(f"{pilot} complete: {c}")
    if not dry:
        status(f"pilot cohort {pilot} complete (EXCLUDED): {c['done']}/{c['n']} done, {c['failed']} failed after one retry, "
               f"{c['diverged_or_nonfinite']} diverged or non-finite -> gates_v11")
    rc = py("gates_v11.py", "--cohort", pilot, "--jobs", str(jobs), logname=("ut_" if dry else "") + "gates.log")
    log(f"gates_v11 rc={rc}")
    if rc != 0:
        if not dry:
            status(f"gates_v11 FAILED rc={rc} (results/S2_2_v11/logs/gates.log); main runs blocked unless a gate file exists")
        raise SystemExit(rc)
    gate = os.path.join(RUNS, ("UT_" if dry else "") + "GATE_S2_2v11_main.nogo")
    if os.path.exists(gate) and dry:
        log("dry run: gate NOGO on tiny unit-test runs (expected: GC needs trained heads); the chain continues to exercise the code")
    elif os.path.exists(gate):
        log("gate NOGO: main runs are skipped by the workers; no main analysis (the author decides)")
        for p in procs:
            p.wait()
        return
    if dry:
        for p in procs:
            p.wait()
        procs = start_workers([mp], nw, extra, tag="ut_w")
    c = wait_cohort(main, every=20 if dry else 300)
    for p in procs:
        p.wait()
    c = retry_failed(main, mp, extra)
    log(f"{main} complete: {c}")
    if not dry:
        status(f"main cohort {main} complete: {c['done']}/{c['n']} done, {c['failed']} failed after one retry, "
               f"{c['diverged_or_nonfinite']} diverged or non-finite -> analyze_v11 (runs once)")
    rc = py("analyze_v11.py", "--cohort", main, "--jobs", str(jobs), logname=("ut_" if dry else "") + "analyze.log")
    log(f"analyze_v11 rc={rc}")
    if not dry:
        status(f"analyze_v11 rc={rc} (results/S2_2_v11/S2_2_v11_report.md, logs/analyze.log)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "dryrun"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=20)
    a = ap.parse_args()
    if a.workers > 6 or a.jobs > 6:
        raise SystemExit("at most 6 training workers and 6 analysis processes (shared machine)")
    os.makedirs(LOGS, exist_ok=True)
    if a.mode == "run":
        import guard_v11
        ok, msg = guard_v11.verify_frozen()
        if not ok:
            raise SystemExit(f"REFUSED: {msg}")
        if not os.path.exists(os.path.join(PLANS, "ENG_GO")):
            raise SystemExit("REFUSED: plans_v11/ENG_GO absent (engineering decision not GO)")
        if not os.path.exists(os.path.join(PLANS, "S2_2v11_pilot.json")):
            rc = subprocess.run([PY, os.path.join(HERE, "make_plan_v11.py"), "cohort"], cwd=ROOT, env=ENV1).returncode
            if rc != 0:
                raise SystemExit("make_plan_v11 cohort failed")
        chain("S2_2v11_pilot", "S2_2v11_main", a.workers, a.jobs)
    else:
        if not os.path.exists(os.path.join(PLANS, "UT_S2_2v11_pilot.json")):
            subprocess.run([PY, os.path.join(HERE, "make_plan_v11.py"), "unittest"], cwd=ROOT, env=ENV1, check=True)
        chain("UT_S2_2v11_pilot", "UT_S2_2v11_main", a.workers, a.jobs, extra=("--max-steps", str(a.max_steps)), dry=True)


if __name__ == "__main__":
    main()
