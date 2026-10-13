#!/usr/bin/env python3
"""S2.3 AMENDMENT A2 GPU worker (prereg/AMENDMENT_A2_complete.md, rules A2.1(c) and A2.3(d)).

Lives in s23/resume/, NEVER in src/. It runs the FROZEN src/worker.py main loop, imported unchanged (the
file on disk and its hash are untouched), with ONE difference inside this process only: the GPU budget
constants of the imported frozen s23common module are set to +inf, because the author removed the GPU
budget limits of Stage A (A2.1). Everything else -- frozen-code verification, plan, g4_decision check,
reduction skip set, frozen split (i % nw), .done restart rule, GpuCtx, do_item (keys, solver settings,
certificates, re-solve rule), output files, worker log and GPU ledger -- is the frozen code.

  python a2_worker.py --phase main --pool gpu --wid k --nw 2            real run (launcher only)
  python a2_worker.py --dry-run --dry-dir D --phase main --pool gpu|cpu --wid k --nw W [--frozen-budget]
        excluded dry run (R29): the same frozen main loop, with worker.GpuCtx and worker.do_item replaced
        by recorders and C.LOGS redirected to D. CUDA must be hidden (CUDA_VISIBLE_DEVICES empty): no GPU
        work, no output under runs/, no ledger line, no statistic. --frozen-budget keeps the frozen
        constants (shows that the frozen pool refuses). Writes D/items_<pool><wid>[_frozen].json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HOME = os.path.expanduser("~")
S23 = os.path.join(HOME, "hrh_v2_stage2_20261008", "s23")
SRC = os.path.join(S23, "src")
if "S23_ROOT" in os.environ:
    _r = os.path.realpath(os.environ["S23_ROOT"])
    if not _r.startswith(os.path.join(os.path.realpath(S23), "dev") + os.sep):
        raise SystemExit("A2 worker: a test root must lie under s23/dev/")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import s23common as C  # noqa: E402  (frozen)

C.cap_threads(1)  # as the frozen worker does under __main__, before numpy / torch are imported

FROZEN_SOFT, FROZEN_HARD = 11.5, 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["main"])
    ap.add_argument("--pool", required=True, choices=["gpu", "cpu"])
    ap.add_argument("--wid", type=int, required=True)
    ap.add_argument("--nw", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-dir")
    ap.add_argument("--frozen-budget", action="store_true")
    a = ap.parse_args()
    if (C.GPU_SOFT_STOP_H, C.GPU_HARD_STOP_H) != (FROZEN_SOFT, FROZEN_HARD):
        raise SystemExit(f"A2 worker: unexpected frozen budget constants {C.GPU_SOFT_STOP_H}, {C.GPU_HARD_STOP_H}")
    if a.pool == "cpu" and not a.dry_run:
        raise SystemExit("A2 worker: the CPU pool runs the frozen src/worker.py directly (rule A2.1(d))")
    if a.frozen_budget and not a.dry_run:
        raise SystemExit("A2 worker: --frozen-budget is a dry-run option")
    if a.dry_run:
        if not a.dry_dir:
            raise SystemExit("A2 worker: --dry-run needs --dry-dir")
        dd = os.path.realpath(a.dry_dir)
        if not dd.startswith(os.path.join(os.path.realpath(S23), "dev") + os.sep):
            raise SystemExit("A2 worker: --dry-dir must lie under s23/dev/")
        if os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "":
            raise SystemExit("A2 worker: a dry run needs CUDA_VISIBLE_DEVICES set and empty")
    ledger = C.gpu_hours_used()
    frozen_ok = ledger < FROZEN_SOFT
    if not a.frozen_budget:
        C.GPU_SOFT_STOP_H = math.inf      # rule A2.1(c): in this process only; the frozen file is unchanged
        C.GPU_HARD_STOP_H = math.inf
    import worker as W  # noqa: E402  (frozen src/worker.py, imported unchanged)
    if os.path.realpath(W.__file__) != os.path.realpath(os.path.join(SRC, "worker.py")) or W.C is not C:
        raise SystemExit("A2 worker: did not import the frozen src/worker.py / s23common")
    tag = f"{a.pool}{a.wid}_{a.phase}"
    print(f"{C.utcnow()} A2 worker {tag}{' DRY RUN' if a.dry_run else ''}: GPU budget constants soft "
          f"{FROZEN_SOFT} -> {C.GPU_SOFT_STOP_H}, hard {FROZEN_HARD} -> {C.GPU_HARD_STOP_H}; ledger {ledger:.4f} h; "
          f"frozen predicate gpu_budget_ok(soft=True) = {frozen_ok}", flush=True)
    sys.argv = [os.path.join(SRC, "worker.py"), "--phase", a.phase, "--pool", a.pool,
                "--wid", str(a.wid), "--nw", str(a.nw)]
    if not a.dry_run:
        W.main()                         # frozen: exits via sys.exit(4/5) only on budget, now unreachable
        return 0
    # ------------------------------------------------------------------ dry run: recorders, no GPU work
    os.makedirs(a.dry_dir, exist_ok=True)
    rec, ctx = [], []

    class _NoGpuCtx:  # replaces worker.GpuCtx: no CUDA, no Cholesky, no ledger line
        def __init__(self, *args, **kw):
            ctx.append(kw.get("worker", args[-1] if args else None))

    def _record(it, inst, fr, arr, gpu, worker):  # replaces worker.do_item: no counts, no solve, no output
        rec.append(it["id"])

    W.GpuCtx = _NoGpuCtx
    W.do_item = _record
    C.LOGS = a.dry_dir                   # the frozen loop's worker log goes to the dry-run folder
    rc = 0
    try:
        W.main()
    except SystemExit as e:              # frozen budget stop: sys.exit(4) / sys.exit(5)
        rc = e.code if isinstance(e.code, int) else 1
    out = {"utc": C.utcnow(), "worker": tag, "frozen_budget": a.frozen_budget, "rc": rc,
           "ledger_h": ledger, "frozen_predicate_soft_ok": frozen_ok, "gpu_ctx_built": len(ctx),
           "n_items_would_run": len(rec), "items_would_run": rec}
    p = os.path.join(a.dry_dir, f"items_{a.pool}{a.wid}{'_frozen' if a.frozen_budget else ''}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"A2 worker dry run {tag}{' (frozen budget)' if a.frozen_budget else ' (A2.1(c))'}: rc={rc}, "
          f"GpuCtx stand-in built {len(ctx)}x, items that would run {len(rec)}"
          + (f" (first {rec[0]}, last {rec[-1]})" if rec else "") + f"; -> {p}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
