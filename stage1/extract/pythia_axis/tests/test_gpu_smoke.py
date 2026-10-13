"""S0b GPU smoke of the Pythia-axis extraction code on known, non-Pythia data (role pyt-axis-extract, 2026-10-10).
Not citable. One GPU step of the axis ledger:
  PYT_S0B_TEST=1 analysis/s12_layermax_pythia/launch_pyt.sh pyt_s0b_smoke_extract 2 --gpu 1.0 -- \
      venv/bin/python extract/pythia_axis/tests/test_gpu_smoke.py
(a) bigscience/bloomz-560m, read-only from the standard cache (local_files_only), WIT items 0..7, the copied
    pipeline with the token-0 diagnostics switched on: every layer and lm_aux must equal the first 8 rows of the
    stored features/wit1024/bigscience__bloomz-560m.npy bitwise (E2 in miniature: the diagnostics leave the primary
    stack unchanged); E6 rerun bitwise; per-item forward time vs the Stage-1 sidecar (descriptive slowdown only).
(b) the synthetic GPT-NeoX snapshot of the CPU tests on the GPU, 16 items: loading check, rotary check, E4
    semantics exactly 0.0, E6, _tok1, sink statistics; GPU features vs the CPU-test features (reported only).
(c) the S4 free-memory rule evaluated for every size (report only; no load).
Outputs only under extract/pythia_axis/s0b_test_out/ (report test_report_gpu.json).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

assert os.environ.get("PYT_S0B_TEST") == "1", "run with PYT_S0B_TEST=1"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = Path(__file__).resolve().parent
AXIS = HERE.parent
PROJ = AXIS.parents[1]
sys.path[:0] = [str(HERE), str(AXIS), str(AXIS.parent)]

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("PYT_THREADS", "2")))
import pyt_axis_common as C  # noqa: E402
import registry_pythia_axis as R  # noqa: E402
import extract_core_pyt as ec  # noqa: E402

assert C.TEST_MODE and ec.DEV.startswith("cuda")
REPORT = {"started_utc": C.utcnow(), "tests": {}}
RUNID = C.utcnow().replace("-", "").replace(":", "")  # outputs are never overwritten: one directory per run
RUNROOT = C.TEST_ROOT / "runs" / ("gpu_" + RUNID)
CPU_RUN = C.TEST_ROOT / "runs" / (C.TEST_ROOT / "latest_cpu_run.txt").read_text().strip()


def rec(name, fn):
    t = time.time()
    try:
        det = fn()
        REPORT["tests"][name] = dict(pass_=True, details=det, s=round(time.time() - t, 2))
        print(f"PASS {name} ({time.time() - t:.1f} s) {json.dumps(det, default=str)[:800]}", flush=True)
    except Exception as e:  # noqa: BLE001
        REPORT["tests"][name] = dict(pass_=False, error=repr(e), tb=traceback.format_exc()[-3000:])
        print(f"FAIL {name}: {e!r}\n{traceback.format_exc()}", flush=True)


def bloomz():
    free0, tot = torch.cuda.mem_get_info()
    out_root = RUNROOT / "features" / "e2_smoke"
    sink = RUNROOT / "results" / "sink_e2_smoke.json"
    model = R.E2_MODEL
    t0 = time.time()
    ec.run_lm(model, ["wit1024"], out_root, 8, print, tok1_dir=out_root / "_tok1", sink_path=sink)
    wall = time.time() - t0
    stem = out_root / R.slug(model)
    cmp = C.compare_features(stem, C.STAGE1_E2, first_n=8, ref_rows=8)
    meta = json.load(open(str(stem) + ".json"))
    ref_meta = json.load(open(str(C.STAGE1_E2) + ".json"))
    aux_ok = all(v.get("bitwise_equal", v.get("equal")) for v in cmp["lm_aux"].values())
    assert cmp["ok_shape"] and cmp["bitwise_equal"] and aux_ok, (cmp.get("max_abs_diff"), cmp.get("lm_aux"))
    rr = meta["rerun_check"]
    assert rr["max_abs_diff"] == 0.0 and all(rr["primary_bitwise"]) and all(rr["tok1_bitwise"]), rr
    sem = meta["hidden_state_semantics"]
    return dict(shape=meta["shape"], bitwise_first8=cmp["bitwise_equal"], lm_aux_bitwise=aux_ok, semantics=sem,
                attn=meta["attn_implementation"], per_item_ms=meta["timings"]["per_item_ms"],
                stage1_per_item_ms=ref_meta["timings"]["per_item_ms"],
                slowdown_per_item=meta["timings"]["per_item_ms"] / ref_meta["timings"]["per_item_ms"],
                load_s=meta["timings"]["load_s"], wall_s=wall, gpu_free_gib_before=free0 / 2**30,
                gpu_peak_gb=meta["pythia_axis"]["gpu_peak_allocated_gb"], torch_load_calls=len(
                    meta["pythia_axis"]["load"]["torch_load_calls"]))


def synthetic():
    rev = "1" * 40
    snap = C.TEST_ROOT / "synthetic_snaps" / "main" / rev
    assert (snap / "pytorch_model.bin").exists(), "run the CPU tests first (they build the synthetic snapshot)"
    out_root = RUNROOT / "features" / "synthetic_gpu"
    model_id = "synthetic/neox-tiny-gpu"
    checks = ec.run_lm(model_id, ["wit1024"], out_root, 16, print, snap=snap, tok1_dir=out_root / "_tok1",
                       sink_path=RUNROOT / "results" / "sink_synthetic_gpu.json",
                       gptneox_checks=True)
    meta = json.load(open(out_root / f"{R.slug(model_id)}.json"))
    sem = meta["hidden_state_semantics"]
    assert sem["hs0_vs_layer0_input"] == 0.0 and sem["hs_i_vs_layer_i-1_output_max"] == 0.0 \
        and sem["hsL_vs_finalnorm(lastlayer)"] == 0.0, sem
    assert checks["loading_check"]["pass"] and checks["rotary_check"]["ok"], checks
    rr = meta["rerun_check"]
    assert rr["max_abs_diff"] == 0.0 and all(rr["primary_bitwise"]) and all(rr["tok1_bitwise"]), rr
    g = np.load(out_root / f"{R.slug(model_id)}.npy")
    cpu = np.load(CPU_RUN / "features" / "synthetic_cpu" / "synthetic__neox-tiny-cpu.npy")
    return dict(semantics=sem, loading_pass=True, rotary=checks["rotary_check"], shape=list(g.shape),
                gpu_vs_cpu_max_abs=float(np.abs(g.astype(np.float64) - cpu).max()),
                attn=meta["attn_implementation"])


def memrule():
    free, total = torch.cuda.mem_get_info()
    return {s: dict(need_gib=(R.gpu_need_bytes(R.MODEL_OF[s]) + R.GPU_HEADROOM_BYTES) / 2**30,
                    free_gib_now=free / 2**30,
                    ok_now=bool(free >= R.gpu_need_bytes(R.MODEL_OF[s]) + R.GPU_HEADROOM_BYTES)) for s in R.SIZES}


rec("g1_bloomz_first8_bitwise", bloomz)
rec("g2_synthetic_gptneox_gpu", synthetic)
rec("g3_s4_memory_rule", memrule)
REPORT["run_id"] = RUNID
REPORT["cpu_run_compared"] = CPU_RUN.name
REPORT["code_sha256"] = {str(p.relative_to(PROJ)): C.sha256_file(p) for p in sorted(list(AXIS.glob("*.py")) + list(HERE.glob("*.py")))}
REPORT["finished_utc"] = C.utcnow()
REPORT["n_pass"] = sum(1 for v in REPORT["tests"].values() if v["pass_"])
REPORT["n_fail"] = sum(1 for v in REPORT["tests"].values() if not v["pass_"])
h = C.write_json(RUNROOT / "test_report_gpu.json", REPORT)
print(f"GPU SMOKE: {REPORT['n_pass']} pass, {REPORT['n_fail']} fail; report sha256 {h}")
sys.exit(0 if REPORT["n_fail"] == 0 else 1)
