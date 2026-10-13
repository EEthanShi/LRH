"""G0 smoke on known data for the Pythia axis: post-freeze amendment A-G0 (role pyt-axis-extract-run, 2026-10-10).
NOT citable. Measures the GPU slowdown f0 of plan section 11 gate G0; computes nothing about any Pythia model.

Why this file exists (a deviation, logged in STATUS with this file's sha256 before its first run):
  ANALYSIS_PLAN_PYTHIA.md (sha256 6de89603...) section 11: "Gate G0 (S0b, before any download). The smoke on known
  data gives f0 = measured/nominal." The code was frozen before G0 (STATUS 2026-10-10T04:05:05Z
  [pyt-axis-code-freeze] CODE FROZEN, manifest S0B_CODE_MANIFEST.json sha256 ca600c80..., "G0 NOT YET RUN"), and no
  frozen entry point can run a GPU smoke after that line: extract/pythia_axis/tests/test_gpu_smoke.py asserts
  PYT_S0B_TEST=1, which pyt_axis_common.py refuses after the CODE FROZEN line (exit 3), and every E, I and R entry
  point exits 3 until Part B is frozen. The plan's G0 rule also foresees new smokes ("at most 3 smokes in total").
What it does: nothing new algorithmically. It calls the frozen extract_core_pyt.run_lm exactly as the frozen
  `run_pyt.py e2` does (bigscience/bloomz-560m at its Stage-1 pin, read-only from the standard Hugging Face cache,
  local_files_only, fp32 load, all 1024 WIT items, token-0 diagnostics on, inside the frozen ResourceWatch), but
  writes only under analysis/s12_layermax_pythia/g0_smoke/run_<tag>_<UTC>/ (a new directory; refused if it exists).
  It then compares the stack with the stored features/wit1024/bigscience__bloomz-560m.npy (descriptive, bitwise
  expected) and records the per-item forward time against the Stage-1 sidecar (descriptive slowdown).
Workload and nominal: this is the plan's E2 workload, whose nominal is 0.5 GPU-min (plan section 11 "Nominal"), so
  the gate call is   gpu_gate_pyt.py g0 --smoke-steps <launcher NAME of this run> --smoke-nominal 0.5
Never writes STATUS.md, features/, results/ (except the launcher's own log/marker/ledger rows), or any hub cache;
never opens a Pythia snapshot; makes no network call (HF_HUB_OFFLINE=1).

usage (GPU, only when the lead's GPU-ORDER condition holds):
  analysis/s12_layermax_pythia/launch_pyt.sh pyt_G0_smoke_1 4 --gpu 1.0 -- \
      venv/bin/python analysis/s12_layermax_pythia/g0_smoke/g0_smoke_pyt.py
CPU self-test of this driver (never touches the GPU; CUDA hidden before torch is imported; 4 items):
  PYT_LOGDIR=analysis/s12_layermax_pythia/g0_smoke/logs analysis/s12_layermax_pythia/launch_pyt.sh \
      pyt_G0_cpuselftest_1 4 -- venv/bin/python analysis/s12_layermax_pythia/g0_smoke/g0_smoke_pyt.py --cpu-selftest
"""
from __future__ import annotations

import os
import sys

CPU_SELFTEST = "--cpu-selftest" in sys.argv[1:]
if CPU_SELFTEST:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported anywhere in this process
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_TH = str(max(1, min(4, int(os.environ.get("PYT_THREADS", "4")))))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _TH

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[2]
assert PROJ.name == "hrh_v2_stage1_20261008", PROJ
sys.path.insert(0, str(PROJ / "extract" / "pythia_axis"))
if os.environ.get("PYT_S0B_TEST"):
    raise SystemExit("PYT_S0B_TEST must be unset (the S0b test mode is closed after CODE FROZEN)")

import pyt_axis_common as C  # noqa: E402  (frozen)
import registry_pythia_axis as R  # noqa: E402  (frozen)

assert not C.TEST_MODE
C.check_plan_unchanged()
if C.code_frozen_line() is None:
    raise SystemExit("no CODE FROZEN line in STATUS.md; this post-freeze amendment does not apply")

import torch  # noqa: E402

torch.set_num_threads(int(_TH))
import extract_core_pyt as ec  # noqa: E402  (frozen)


def nvsmi() -> dict:
    out = {}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        out["gpu"] = q.strip()
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        out["compute_apps"] = a
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-selftest", action="store_true")
    ap.add_argument("--m", type=int, default=None, help="items (default: all 1024; CPU self-test: 4)")
    args = ap.parse_args()
    m = args.m if args.m is not None else (4 if args.cpu_selftest else None)
    if args.cpu_selftest:
        ec.DEV = "cpu"  # the self-test never touches the GPU (CUDA is hidden above)
        assert not torch.cuda.is_available()
        tag = "cpuselftest"
    else:
        assert m is None, "the G0 smoke runs the full E2 workload (1024 items)"
        assert torch.cuda.is_available() and str(ec.DEV).startswith("cuda")
        tag = "gpu"
    utc = C.utcnow()
    out = HERE / f"run_{tag}_{utc.replace('-', '').replace(':', '')}"
    out.mkdir(parents=True, exist_ok=False)
    model = R.E2_MODEL
    ref = C.STAGE1_E2
    ref_sha = C.sha256_file(str(ref) + ".npy")
    assert ref_sha == R.E2_NPY_SHA256, ref_sha
    smi_before = nvsmi()
    gm = None
    if not args.cpu_selftest:
        free, total = torch.cuda.mem_get_info()
        gm = dict(free_bytes=int(free), total_bytes=int(total))
    t0 = time.time()
    with C.ResourceWatch(label="G0 smoke bloomz-560m") as rw:
        ec.run_lm(model, [R.ITEMS_DS], out, m, C.log, mode="fp32", tok1_dir=out / "_tok1",
                  sink_path=out / "sink_g0_bloomz-560m.json",
                  extra_meta=dict(role="pyt-axis-extract-run", check="G0 smoke on known data (amendment A-G0)",
                                  plan_sha256=C.PLAN_SHA256))
    wall = time.time() - t0
    stem = out / R.slug(model)
    ok, why = ec.is_valid(out, R.ITEMS_DS, model, m or 1024)
    cmp = C.compare_features(stem, ref, first_n=64, ref_rows=m)
    meta = json.load(open(str(stem) + ".json"))
    ref_meta = json.load(open(str(ref) + ".json"))
    aux = cmp.get("lm_aux") or {}
    aux_bitwise = bool(aux) and "error" not in aux and all(v.get("bitwise_equal", v.get("equal")) for v in aux.values())
    rr = meta["rerun_check"]
    tm = meta["timings"]
    res = dict(check="G0 smoke on known data (amendment A-G0; not citable)", mode=tag, model=model,
               source_revision=meta["source"]["revision"], out=str(out.relative_to(PROJ)), m=m or 1024,
               valid=ok, valid_reason=why, shape=meta["shape"], npy_sha256=meta["files_sha256"]["primary"]["sha256"],
               bitwise_equal_stored=bool(cmp["bitwise_equal"]), lm_aux_bitwise_equal_stored=aux_bitwise,
               max_abs_diff_stored=cmp.get("max_abs_diff"), ref=str(ref.relative_to(PROJ)) + ".npy",
               ref_sha256=ref_sha, semantics=meta["hidden_state_semantics"], rerun=rr,
               timings=tm, run_lm_wall_s=wall, stage1_per_item_ms=ref_meta["timings"]["per_item_ms"],
               per_item_slowdown_vs_stage1=tm["per_item_ms"] / ref_meta["timings"]["per_item_ms"],
               gpu_peak_allocated_gb=meta["pythia_axis"]["gpu_peak_allocated_gb"], gpu_mem_before=gm,
               nvidia_smi_before=smi_before, nvidia_smi_after=nvsmi(), resources=rw.summary(),
               threads=int(_TH), utc_start=utc, utc_end=C.utcnow(),
               nominal_gpu_min_of_this_workload=R.NOMINAL_GPU_MIN["E2"],
               gate_call="gpu_gate_pyt.py g0 --smoke-steps <launcher NAME> --smoke-nominal 0.5 (plan section 11)")
    res["pass"] = bool(ok and rr["max_abs_diff"] == 0.0)
    h = C.write_json(out / "g0_smoke_report.json", res)
    print(f"G0 SMOKE {tag}: valid={ok} shape={meta['shape']} bitwise_vs_stored={cmp['bitwise_equal']} "
          f"lm_aux_bitwise={aux_bitwise} per_item_ms={tm['per_item_ms']:.2f} (Stage-1 "
          f"{ref_meta['timings']['per_item_ms']:.2f}, x{res['per_item_slowdown_vs_stage1']:.2f}) load_s={tm['load_s']:.1f} "
          f"fwd_s={tm['forward_s']:.1f} run_lm_wall_s={wall:.1f} report sha256 {h}", flush=True)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
