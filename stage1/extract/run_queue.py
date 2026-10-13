"""Sequential, resumable GPU extraction queue for HRH v2 Stage 1.

Usage:
  python run_queue.py                       # full queue -> features/<ds>/
  python run_queue.py --smoke 16 --models vit_tiny_patch16_224.augreg_in21k EleutherAI/pythia-410m-seed1
                                            # first 16 items of every dataset -> features/_smoke/<ds>/
Resumable: a (model, dataset) whose .npy + .json + .done exist and validate (shape, dtype, sha256) is skipped,
and a model is not even loaded if all of its datasets are valid. Failures write <slug>.failed and the queue continues.
At the end writes features/QUEUE.done (all valid) or features/QUEUE.failed (list of failures).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from common import FEATURES, append_status, disk_free_gb, utcnow  # noqa: E402
from registry import build_queue, slug  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None, help="limit every dataset to its first N items; output to features/_smoke")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--min_free_gb", type=float, default=40.0)
    args = ap.parse_args()

    import extract_core as ec
    try:
        from transformers.utils import logging as tlog
        tlog.disable_progress_bar()
    except Exception:
        pass

    out_root = FEATURES / "_smoke" if args.smoke else FEATURES
    out_root.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "queue"
    q = build_queue()
    if args.models:
        q = [j for j in q if j["model"] in args.models]
        missing = set(args.models) - {j["model"] for j in q}
        if missing:
            raise SystemExit(f"unknown models {missing}")
    if args.datasets:
        q = [dict(j, datasets=[d for d in j["datasets"] if d in args.datasets]) for j in q]
        q = [j for j in q if j["datasets"]]

    def log(msg):
        print(f"{utcnow()} {msg}", flush=True)

    for marker in ["QUEUE.done", "QUEUE.failed"]:
        if (out_root / marker).exists():
            (out_root / marker).unlink()
    n_jobs = sum(len(j["datasets"]) for j in q)
    log(f"[{tag}] {len(q)} model loads, {n_jobs} (model,dataset) jobs, out_root={out_root}, pid={os.getpid()}")
    if not args.smoke:
        append_status(f"queue started pid={os.getpid()}: {len(q)} model loads / {n_jobs} jobs", "logs/extract_queue.log")
    failures = []
    t_start = time.time()
    for k, job in enumerate(q):
        model = job["model"]
        todo = []
        for ds in job["datasets"]:
            m = len(ec.load_items(ds, args.smoke))
            ok, why = ec.is_valid(out_root, ds, model, m)
            if not ok:
                todo.append(ds)
        if not todo:
            log(f"[{k+1}/{len(q)}] {model}: all datasets valid, skip")
            continue
        free = disk_free_gb()
        if free < args.min_free_gb:
            msg = f"disk free {free:.0f} GB < {args.min_free_gb} GB; stopping before {model}"
            log(msg)
            failures.append((model, "ALL", msg))
            if not args.smoke:
                append_status("STOP: " + msg)
            break
        log(f"[{k+1}/{len(q)}] {model} -> {todo} (disk free {free:.0f} GB)")
        t0 = time.time()
        try:
            if job["kind"] == "vit":
                ec.run_vit(model, todo, out_root, args.smoke, log)
            else:
                ec.run_lm(model, todo, out_root, args.smoke, log)
            dt = time.time() - t0
            if not args.smoke:
                append_status(f"done {model} on {todo} in {dt/60:.1f} min",
                              ", ".join(f"features/{ds}/{slug(model)}.npy" for ds in todo))
        except Exception as e:
            tb = traceback.format_exc()
            log(f"FAILED {model}: {e!r}\n{tb}")
            failures.append((model, todo, repr(e)))
            for ds in todo:
                p = ec.out_paths(out_root, ds, slug(model))
                if not p["done"].exists():
                    p["dir"].mkdir(parents=True, exist_ok=True)
                    p["failed"].write_text(f"{utcnow()}\n{tb}")
            if not args.smoke:
                append_status(f"FAILED {model} on {todo}: {e!r}"[:400], "logs/extract_queue.log")
            try:
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
    # final verification pass over every job
    bad = []
    for job in q:
        for ds in job["datasets"]:
            m = len(ec.load_items(ds, args.smoke))
            ok, why = ec.is_valid(out_root, ds, job["model"], m, check_hash=True)
            if not ok:
                bad.append(f"{job['model']}/{ds}: {why}")
    total = (time.time() - t_start) / 60
    if bad or failures:
        (out_root / "QUEUE.failed").write_text(utcnow() + "\n" + "\n".join(bad) + "\n" +
                                               "\n".join(map(str, failures)) + "\n")
        log(f"[{tag}] finished with {len(bad)} invalid jobs in {total:.1f} min")
        if not args.smoke:
            append_status(f"queue finished with {len(bad)} invalid/failed jobs in {total:.1f} min", "features/QUEUE.failed")
    else:
        (out_root / "QUEUE.done").write_text(utcnow() + f"\n{n_jobs} jobs valid\n")
        log(f"[{tag}] all {n_jobs} jobs valid in {total:.1f} min")
        if not args.smoke:
            append_status(f"queue finished: all {n_jobs} jobs valid in {total:.1f} min", "features/QUEUE.done")


if __name__ == "__main__":
    main()
