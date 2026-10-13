"""Run S1.1 jobs with a bounded number of parallel workers (resumable: jobs
with a .done marker are skipped).

COCO (only after LEAD_AUTHORIZATION.json exists; every job re-checks it):
    setsid nohup venv/bin/python analysis/s11/s11_queue.py --coco --clip-mode pool-layer \
        > logs/s11_queue.log 2>&1 &
Synthetic dry run:
    ... s11_queue.py --synthetic --out-root results/S1_1/dryrun_synthetic

Defaults: 3 workers x 4 BLAS threads (12 cores; role limit), ~11 GB RSS per
worker at m = 16384. Writes <out-root>/QUEUE_<tag>.done | .failed at the end.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s11_common as C  # noqa: E402

PY = os.path.join(C.PROJECT, "venv", "bin", "python")


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--coco", action="store_true")
    g.add_argument("--synthetic", action="store_true")
    ap.add_argument("--pres", default=",".join(C.PRES))
    ap.add_argument("--clip-mode", default=None, choices=C.CLIP_MODES)
    ap.add_argument("--pairs", default=None, help="comma list (default: all 61, primary first)")
    ap.add_argument("--syn-plants", default="P-Gauss,P-Rough,P-Conf,P-Aniso")
    ap.add_argument("--syn-reps", default="0")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-root", default=C.RESULTS_DIR)
    ap.add_argument("--extra", default="", help="extra args passed to s11_run.py (e.g. '--draws 0')")
    args = ap.parse_args(argv)
    if args.workers * args.threads > 12:
        raise SystemExit("role limit: <= 12 cores")

    jobs = []
    if args.coco:
        pairs = args.pairs.split(",") if args.pairs else sorted(C.PAIRS, key=lambda p: (not C.PAIRS[p]["primary"], p))
        for pre in args.pres.split(","):
            for p in pairs:
                a = ["--pair", p, "--pre", pre, "--out-root", args.out_root]
                if pre == "prh_clip+l2":
                    a += ["--clip-mode", args.clip_mode]
                jobs.append((p, pre, a))
    else:
        for plant in args.syn_plants.split(","):
            for rep in args.syn_reps.split(","):
                pid = f"syn_{plant}_p4_r{rep}"
                jobs.append((pid, "none", ["--synthetic", plant, "--syn-rep", rep, "--pre", "none",
                                           "--out-root", args.out_root]))
    tag = "coco" if args.coco else "synthetic"
    logdir = os.path.join(C.PROJECT, "logs", "s11")
    os.makedirs(logdir, exist_ok=True)
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = str(args.threads)
    todo = [j for j in jobs if not os.path.exists(os.path.join(args.out_root, "raw", j[1], j[0] + ".done"))]
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {len(jobs)} jobs, {len(todo)} to run", flush=True)
    running, failed = [], []
    t0 = time.time()
    while todo or running:
        while todo and len(running) < args.workers:
            pid, pre, a = todo.pop(0)
            lf = open(os.path.join(logdir, f"{tag}__{pre.replace('+', '_')}__{pid}.log"), "w")
            cmd = [PY, os.path.join(HERE, "s11_run.py")] + a + (args.extra.split() if args.extra else [])
            running.append((pid, pre, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env), lf))
            print(f"  start {pid} {pre}", flush=True)
        time.sleep(5)
        still = []
        for pid, pre, p, lf in running:
            if p.poll() is None:
                still.append((pid, pre, p, lf))
            else:
                lf.close()
                print(f"  end {pid} {pre} rc={p.returncode} t={time.time() - t0:.0f}s", flush=True)
                if p.returncode != 0:
                    failed.append((pid, pre))
        running = still
    marker = os.path.join(args.out_root, f"QUEUE_{tag}." + ("failed" if failed else "done"))
    with open(marker, "w") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} failed={failed}\n")
    print("finished", marker, flush=True)


if __name__ == "__main__":
    main()
