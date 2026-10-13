"""Unattended finalisation: when a main cohort has no pending runs, retry its failed runs once
(prereg §4: failed runs are re-queued), then run the preregistered analysis and log to STATUS.

    setsid nohup python src/auto_finalize.py > logs/auto_finalize.log 2>&1 &
Order: S2_2_main -> analyze s22 ; S2_1_main -> analyze s21.  Analysis uses 2 CPU processes.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
PY = os.path.expanduser("~/hrh_v2_stage1_20261008/venv/bin/python")
STATUS = os.path.expanduser("~/hrh_v2_stage1_20261008/STATUS.md")


def status(msg):
    with open(STATUS, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [stage2] {msg}\n")


def state(cohort):
    runs = json.load(open(os.path.join(ROOT, "plans", f"{cohort}.json")))["runs"]
    done = failed = pending = 0
    for r in runs:
        d = os.path.join(ROOT, "runs", cohort, r["run_id"])
        if os.path.exists(os.path.join(d, ".done")):
            done += 1
        elif os.path.exists(os.path.join(d, ".failed")) and not os.path.exists(os.path.join(d, ".lock")):
            failed += 1
        else:
            pending += 1
    return len(runs), done, failed, pending


def finalize(cohort, what, max_wait_h=40):
    t0 = time.time()
    while True:
        n, done, failed, pending = state(cohort)
        if pending == 0:
            break
        if time.time() - t0 > max_wait_h * 3600:
            status(f"auto_finalize: {cohort} still has {pending} pending runs after {max_wait_h} h; running INTERIM analysis")
            break
        time.sleep(300)
    n, done, failed, pending = state(cohort)
    if failed:
        status(f"auto_finalize: {cohort} {failed} failed runs -> one retry pass (logs/retry_{cohort}.log)")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with open(os.path.join(ROOT, "logs", f"retry_{cohort}.log"), "w") as f:
            subprocess.run([PY, "src/train.py", "worker", "--wid", "9", "--plan", f"plans/{cohort}.json"],
                           cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        n, done, failed, pending = state(cohort)
    status(f"auto_finalize: {cohort} {done}/{n} done, {failed} failed, {pending} pending -> analyze.py {what}")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with open(os.path.join(ROOT, "logs", f"analyze_{what}.log"), "w") as f:
        rc = subprocess.run([PY, "src/analyze.py", what, "--jobs", "2"], cwd=ROOT, env=env, stdout=f,
                            stderr=subprocess.STDOUT).returncode
    out = {"s22": "results/S2_2/S2_2_report.md", "s21": "results/S2_1/S2_1_report.md"}[what]
    status(f"auto_finalize: analyze.py {what} rc={rc} ({out}, logs/analyze_{what}.log)")
    open(os.path.join(ROOT, "logs", f"analyze_{what}." + ("done" if rc == 0 else "failed")), "w").write(str(rc))


if __name__ == "__main__":
    os.chdir(ROOT)
    finalize("S2_2_main", "s22")
    finalize("S2_1_main", "s21")
