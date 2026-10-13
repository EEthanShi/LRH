"""GPU gates G0 and G1, the free-GPU-memory wait, and the read-only S2.2 rate snapshot (plan section 11).

NEW FILE (role pyt-axis-extract, S0b, 2026-10-10). Plan: analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md v1.0
(sha256 6de89603...), section 11 "GPU budget" and section 2 "Order and memory". Hashed in the S0b CODE FROZEN line.
CPU only; it never holds the GPU (nvidia-smi queries only), so it is not a GPU step of the ledger.

Nominal GPU minutes (plan section 11): S0b smoke 3; E2 0.5; S4 4; S6 3; S7 primary 7.4; S7 l2 7.4; V-P6 2 (scalar 0.5).
Primary path = 3 + 0.5 + 4 + 3 + 7.4 + 0.5 = 18.4.

  g0      Gate G0 (S0b, before any download). f0 = (sum of the ledger minutes of the named smoke steps) / (their
          nominal). If ledger + f0 x (primary path remaining after the smoke = 0.5 + 4 + 3 + 7.4 + 0.5 = 15.4) > 50, the
          decision is DEFERRED: STATUS "[pyt-axis-<role>] DEFERRED (GPU contention) ...", nothing is downloaded and
          nothing Pythia-specific is computed. At most 3 G0 smokes in total (history file); a 4th is refused.
          python gpu_gate_pyt.py g0 --smoke-steps a,b,c --smoke-nominal 3.0
  g1      Gate G1 (before S4, S6, S7). f = max(1, measured / nominal of the last GPU step); if ledger + f x (nominal of
          the remaining primary steps) > 60 the step is DEFERRED (not started). Remaining primary nominal: before S4
          4 + 3 + 7.4 + 0.5 = 14.9; before S6 3 + 7.4 + 0.5 = 10.9; before S7 7.4 + 0.5 = 7.9.
          python gpu_gate_pyt.py g1 --before S4 --last-step pyt_S2_E2 --last-nominal 0.5
  memwait Before each extraction load: free GPU memory >= 4*num_params + 2 GB + 4 GiB, else wait (poll 60 s, up to
          30 min), then DEFERRED (exit 77).
          python gpu_gate_pyt.py memwait --size 6.9b
  s22rate Read-only: completed S2.2 v1.1 runs per hour (count and mtimes of *.done under its main cohort; no file is
          opened) and nvidia-smi per-process memory and utilisation; reported with each Pythia GPU step.
Exit codes: 0 GO, 77 DEFERRED, 3 refused (bad input / more than 3 G0 smokes / no ledger row).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "extract" / "pythia_axis"))
from pyt_axis_common import (LOGS_DIR, R, EXIT_DEFER, EXIT_GATE, append_status, check_plan_unchanged,  # noqa: E402
                             gpu_ledger_total_min, ledger_rows, log, read_json, require, utcnow, write_json,
                             code_frozen_line, TEST_MODE)

NOM = R.NOMINAL_GPU_MIN
PRIMARY_PATH = NOM["S0b_smoke"] + NOM["E2"] + NOM["S4"] + NOM["S6"] + NOM["S7_primary"] + NOM["VP6_scalar"]
assert abs(PRIMARY_PATH - 18.4) < 1e-9
REMAINING_AFTER_SMOKE = PRIMARY_PATH - NOM["S0b_smoke"]  # 15.4
REMAINING_BEFORE = {"S4": NOM["S4"] + NOM["S6"] + NOM["S7_primary"] + NOM["VP6_scalar"],  # 14.9
                    "S6": NOM["S6"] + NOM["S7_primary"] + NOM["VP6_scalar"],  # 10.9
                    "S7": NOM["S7_primary"] + NOM["VP6_scalar"]}  # 7.9
G0_LIMIT, G1_LIMIT, G0_MAX_SMOKES = 50.0, 60.0, 3
S22_COHORT = Path.home() / "hrh_v2_stage2_20261008" / "runs_v11" / "S2_2v11_main"


def _rel(p) -> str:
    return str(p).split("hrh_v2_stage1_20261008/")[-1]


def g0_decision(ledger_total: float, measured: float, nominal: float) -> dict:
    f0 = measured / nominal
    proj = ledger_total + f0 * REMAINING_AFTER_SMOKE
    return dict(f0=f0, ledger_total_min=ledger_total, remaining_primary_nominal_min=REMAINING_AFTER_SMOKE,
                projection_min=proj, limit_min=G0_LIMIT, decision=("GO" if proj <= G0_LIMIT else "DEFERRED"))


def g1_decision(before: str, ledger_total: float, measured: float, nominal: float) -> dict:
    f = max(1.0, measured / nominal)
    rem = REMAINING_BEFORE[before]
    proj = ledger_total + f * rem
    return dict(before=before, f=f, ledger_total_min=ledger_total, remaining_primary_nominal_min=rem,
                projection_min=proj, limit_min=G1_LIMIT, decision=("GO" if proj <= G1_LIMIT else "DEFERRED"))


def _ledger_minutes(names: list[str]) -> dict:
    rows = {r[0]: r for r in ledger_rows()}
    out = {}
    for n in names:
        require(n in rows, f"no ledger row named {n}")
        out[n] = float(rows[n][4])
    return out


def cmd_g0(args):
    hist_path = LOGS_DIR / "gpu_gate_G0_history.jsonl"
    hist = [json.loads(ln) for ln in hist_path.read_text().splitlines()] if hist_path.exists() else []
    require(len(hist) < G0_MAX_SMOKES, f"{len(hist)} G0 smokes already recorded (at most {G0_MAX_SMOKES})")
    steps = [s for s in args.smoke_steps.split(",") if s]
    mins = _ledger_minutes(steps)
    res = g0_decision(gpu_ledger_total_min(), sum(mins.values()), args.smoke_nominal)
    res.update(utc=utcnow(), smoke_steps=mins, smoke_nominal_min=args.smoke_nominal, n_smoke=len(hist) + 1,
               s22=s22_snapshot(), rule=("G0: ledger + f0 x 15.4 <= 50 -> GO; else DEFERRED (nothing downloaded, "
                                         "nothing Pythia-specific computed; resume when no S2.2 v1.1 process holds the "
                                         "GPU or a new smoke projects <= 50; at most 3 smokes)"))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "a") as f:
        f.write(json.dumps(res, default=str) + "\n")
    h = write_json(LOGS_DIR / "gpu_gate_G0.json", res)
    core = (f"G0 (S0b, before any download): smoke {mins} = {sum(mins.values()):.3f} GPU-min vs nominal "
            f"{args.smoke_nominal} -> f0 {res['f0']:.2f}; projection ledger {res['ledger_total_min']:.3f} + f0 x 15.4 = "
            f"{res['projection_min']:.2f} GPU-min vs 50; smoke {res['n_smoke']} of at most 3; S2.2 snapshot "
            f"{res['s22'].get('runs_done_last_60min')} runs/h, GPU util {res['s22'].get('gpu_util_pct')}%; json sha256 {h}")
    if res["decision"] == "GO":
        msg = "G0 GO: " + core
    else:
        msg = "DEFERRED (GPU contention): " + core + "; nothing is downloaded, the frozen plan stays frozen"
    log(msg)
    append_status(msg, _rel(LOGS_DIR / "gpu_gate_G0.json"))
    if res["decision"] != "GO":
        raise SystemExit(EXIT_DEFER)


def cmd_g1(args):
    mins = _ledger_minutes([args.last_step])
    res = g1_decision(args.before, gpu_ledger_total_min(), mins[args.last_step], args.last_nominal)
    res.update(utc=utcnow(), last_step=args.last_step, last_measured_min=mins[args.last_step],
               last_nominal_min=args.last_nominal, s22=s22_snapshot(),
               rule="G1: f = max(1, measured/nominal of the last GPU step); ledger + f x remaining primary <= 60 -> GO")
    h = write_json(LOGS_DIR / f"gpu_gate_G1_{args.before}.json", res)
    core = (f"G1 before {args.before}: last GPU step {args.last_step} {mins[args.last_step]:.3f} min vs nominal "
            f"{args.last_nominal} -> f {res['f']:.2f}; projection {res['ledger_total_min']:.3f} + f x "
            f"{res['remaining_primary_nominal_min']} = {res['projection_min']:.2f} GPU-min vs 60; json sha256 {h}")
    msg = ("G1 GO: " + core) if res["decision"] == "GO" else ("DEFERRED (GPU contention): " + core +
                                                             f"; {args.before} not started")
    log(msg)
    append_status(msg, _rel(LOGS_DIR / f"gpu_gate_G1_{args.before}.json"))
    if res["decision"] != "GO":
        raise SystemExit(EXIT_DEFER)


def nvsmi_free_bytes() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    return int(float(out[0])) * 2**20


def cmd_memwait(args):
    model = R.MODEL_OF[args.size]
    need = R.gpu_need_bytes(model) + R.GPU_HEADROOM_BYTES
    t_end = time.time() + args.timeout_min * 60
    while True:
        free = nvsmi_free_bytes()
        if free >= need:
            log(f"memwait {args.size}: free {free / 2**30:.1f} GiB >= need {need / 2**30:.1f} GiB -> GO")
            return
        if time.time() >= t_end:
            break
        log(f"memwait {args.size}: free {free / 2**30:.1f} GiB < need {need / 2**30:.1f} GiB; waiting {args.poll_s} s")
        time.sleep(args.poll_s)
    msg = (f"DEFERRED (GPU memory): S4 {model} needs {need / 2**30:.1f} GiB free (4*num_params + 2 GB + 4 GiB), "
           f"only {free / 2**30:.1f} GiB after {args.timeout_min} min of polling (plan section 2 'Order and memory')")
    log(msg)
    append_status(msg)
    raise SystemExit(EXIT_DEFER)


def s22_snapshot(window_min: float = 60.0) -> dict:  # read-only; never opens an S2.2 file
    out = dict(cohort=str(S22_COHORT), utc=utcnow())
    try:
        now = time.time()
        n_done, recent = 0, 0
        if S22_COHORT.is_dir():
            with os.scandir(S22_COHORT) as it:
                for e in it:
                    if not e.is_dir():
                        continue
                    with os.scandir(e.path) as it2:
                        for f in it2:
                            if f.name.endswith(".done"):
                                n_done += 1
                                if now - f.stat().st_mtime <= window_min * 60:
                                    recent += 1
        out.update(runs_done_total=n_done, runs_done_last_60min=recent)
    except Exception as e:  # noqa: BLE001
        out["scan_error"] = repr(e)
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        u, mu, mt = [float(x) for x in q.strip().splitlines()[0].split(",")]
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        out.update(gpu_util_pct=u, gpu_mem_used_mib=mu, gpu_mem_total_mib=mt, compute_apps=apps)
    except Exception as e:  # noqa: BLE001
        out["nvsmi_error"] = repr(e)
    return out


def cmd_s22rate(args):
    snap = s22_snapshot(args.window_min)
    h = write_json(LOGS_DIR / f"s22rate_{utcnow().replace(':', '')}.json", snap)
    log(f"S2.2 v1.1 snapshot (read-only): {json.dumps(snap)}; json sha256 {h}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["g0", "g1", "memwait", "s22rate"])
    ap.add_argument("--smoke-steps", default="")
    ap.add_argument("--smoke-nominal", type=float, default=NOM["S0b_smoke"])
    ap.add_argument("--before", choices=["S4", "S6", "S7"], default=None)
    ap.add_argument("--last-step", default=None)
    ap.add_argument("--last-nominal", type=float, default=None)
    ap.add_argument("--size", choices=R.SIZES, default=None)
    ap.add_argument("--timeout-min", type=float, default=30.0)
    ap.add_argument("--poll-s", type=float, default=60.0)
    ap.add_argument("--window-min", type=float, default=60.0)
    args = ap.parse_args()
    check_plan_unchanged()
    if args.cmd == "g0":
        require(bool(args.smoke_steps) and args.smoke_nominal > 0, "g0 needs --smoke-steps and --smoke-nominal")
    if args.cmd == "g1":
        require(args.before is not None and args.last_step and args.last_nominal and args.last_nominal > 0,
                "g1 needs --before, --last-step and --last-nominal")
    if args.cmd == "memwait":
        require(args.size is not None, "memwait needs --size")
    if args.cmd in ("g1", "memwait"):  # after S0b only (G0 belongs to S0b, plan section 11)
        require(code_frozen_line() is not None or TEST_MODE, "the S0b CODE FROZEN line is missing in STATUS.md")
    {"g0": cmd_g0, "g1": cmd_g1, "memwait": cmd_memwait, "s22rate": cmd_s22rate}[args.cmd](args)


if __name__ == "__main__":
    main()
