"""E1-PROV for pythia-2.8b (plan section 2): main .bin vs step143000 .bin, tensor by tensor, on the CPU.

NEW FILE (role pyt-axis-extract, S0b, 2026-10-10). Plan: analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md v1.0
(sha256 6de89603...), sections 1 ("Provenance rule") and 2 ("E1-PROV"). Hashed in the S0b CODE FROZEN line.

Rule (frozen):
  * both .bin files load with torch.load(weights_only=True, mmap=True);
  * names in main but not in step143000 may only match ^gpt_neox\\.layers\\.\\d+\\.attention\\.(bias|masked_bias)$;
    no name may be in step143000 only;
  * every tensor present in both is compared for dtype, shape and bytes;
  * PASS iff the name rule holds and every tensor not matching attention\\.(bias|masked_bias|rotary_emb\\.inv_freq)$
    is bitwise equal (buffer differences are reported only).
  PASS -> 2.8b is loaded from the main snapshot; FAIL -> from the step143000 snapshot (pin dbe7ae30...), and every
  report says "2.8B from step143000 because the main file differs" with the per-tensor differences. An E1-PROV failure
  is not a STOP. The decision is written to results/S1_2_layermax_pythia/extraction/e1prov_2.8b.json and logged in
  STATUS before any 2.8b extraction (extract_core_pyt.source_revision reads it).

Run by download_pyt.py right after the 2.8b step143000 E1, or alone (CPU only, 2 threads):
  launch_pyt.sh pyt_E1PROV 2 -- venv/bin/python analysis/s12_layermax_pythia/e1prov_pyt.py
S0b test mode: `e1prov_pyt.py --main A.bin --step B.bin --out X.json` compares two synthetic files (no decision file
of the axis is written; PYT_S0B_TEST=1 required).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "extract" / "pythia_axis"))
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("PYT_THREADS", "2"))

from pyt_axis_common import (R, RES_DIR, TEST_MODE, append_status, check_plan_unchanged, log, read_json,  # noqa: E402
                             require, sha256_file, write_json, code_frozen_line)


def _bytes_equal(a, b) -> bool:
    import torch
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    a, b = a.contiguous().reshape(-1), b.contiguous().reshape(-1)
    if a.numel() == 0:
        return True
    if a.dtype == torch.bool:
        return bool(torch.equal(a, b))
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def compare_bins(main_path: Path, step_path: Path) -> dict:
    import torch
    t0 = time.time()
    A = torch.load(str(main_path), map_location="cpu", weights_only=True, mmap=True)
    B = torch.load(str(step_path), map_location="cpu", weights_only=True, mmap=True)
    na, nb = set(A), set(B)
    main_only = sorted(na - nb)
    step_only = sorted(nb - na)
    bad_main_only = [k for k in main_only if not R.PROV_MAIN_ONLY_OK_RE.match(k)]
    per = []
    diff_nonbuf, diff_buf = [], []
    for k in sorted(na & nb):
        a, b = A[k], B[k]
        eq = _bytes_equal(a, b)
        is_buf = bool(R.BUFFER_RE.search(k))
        row = dict(name=k, dtype_main=str(a.dtype).replace("torch.", ""), dtype_step=str(b.dtype).replace("torch.", ""),
                   shape_main=list(a.shape), shape_step=list(b.shape), bitwise_equal=eq, buffer=is_buf)
        if not eq:
            if a.dtype == b.dtype and a.shape == b.shape and a.is_floating_point():
                d = (a.double() - b.double()).abs()
                row.update(max_abs_diff=float(d.max()), n_diff=int((d > 0).sum()))
            (diff_buf if is_buf else diff_nonbuf).append(row)
        per.append(row)
    res = dict(main=str(main_path), step143000=str(step_path), n_main=len(na), n_step=len(nb), n_common=len(na & nb),
               main_only=main_only, step_only=step_only, main_only_not_buffer=bad_main_only,
               n_nonbuffer_compared=sum(1 for r in per if not r["buffer"]), n_buffer_compared=sum(1 for r in per if r["buffer"]),
               nonbuffer_differences=diff_nonbuf, buffer_differences=diff_buf[:64], n_buffer_differences=len(diff_buf),
               name_rule_ok=bool(not bad_main_only and not step_only), compare_s=time.time() - t0)
    res["pass"] = bool(res["name_rule_ok"] and not diff_nonbuf)
    del A, B
    return res


def run_e1prov() -> dict:
    """Axis run: both 2.8b snapshots in the dedicated cache, both E1 passed; writes the decision and logs it."""
    m = R.PROV_MODEL
    e1m = read_json(RES_DIR / f"E1_{R.slug(m)}.json")
    e1s = read_json(RES_DIR / f"E1_{R.slug(m)}.step143000.json")
    require(bool(e1m and e1m.get("E1_pass")), "E1 has not passed for the 2.8b main snapshot")
    require(bool(e1s and e1s.get("E1_pass")), "E1 has not passed for the 2.8b step143000 snapshot")
    main_bin = Path(e1m["snapshot_dir"]) / "pytorch_model.bin"
    step_bin = Path(e1s["snapshot_dir"]) / "pytorch_model.bin"
    require(Path(e1m["snapshot_dir"]).name == R.LM_REVISIONS[m] and Path(e1s["snapshot_dir"]).name == R.STEP143000[m],
            "E1 snapshot directories are not the 2.8b pins")
    res = compare_bins(main_bin, step_bin)
    res.update(model=m, main_revision=R.LM_REVISIONS[m], step143000_revision=R.STEP143000[m],
               main_sha256=e1m["shards"][0]["sha256"], step143000_sha256=e1s["shards"][0]["sha256"],
               source=("main" if res["pass"] else "step143000"),
               report_sentence=(None if res["pass"] else "2.8B from step143000 because the main file differs"),
               rule=("PASS iff names in main only match ^gpt_neox.layers.N.attention.(bias|masked_bias)$, none in "
                     "step143000 only, and every tensor not matching attention.(bias|masked_bias|rotary_emb.inv_freq)$ "
                     "is bitwise equal (dtype, shape, bytes); PASS -> main, FAIL -> step143000 (plan section 1)"))
    h = write_json(RES_DIR / "e1prov_2.8b.json", res)
    msg = (f"E1-PROV pythia-2.8b main {R.LM_REVISIONS[m][:12]} ({res['main_sha256'][:16]}...) vs step143000 "
           f"{R.STEP143000[m][:12]} ({res['step143000_sha256'][:16]}...): {res['n_common']} common tensors "
           f"({res['n_nonbuffer_compared']} non-buffer, {res['n_buffer_compared']} buffer), main only "
           f"{len(res['main_only'])} (non-buffer {len(res['main_only_not_buffer'])}), step143000 only "
           f"{len(res['step_only'])}; non-buffer differences {len(res['nonbuffer_differences'])}, buffer differences "
           f"{res['n_buffer_differences']} (reported only); {'PASS' if res['pass'] else 'FAIL'} -> 2.8b SOURCE = "
           f"{res['source'].upper()}" + ("" if res["pass"] else " (2.8B from step143000 because the main file differs)")
           + f"; {res['compare_s']:.0f} s; json sha256 {h}")
    log(msg)
    append_status(msg, str(RES_DIR / "e1prov_2.8b.json").split("hrh_v2_stage1_20261008/")[-1])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default=None)
    ap.add_argument("--step", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    check_plan_unchanged()
    import torch
    torch.set_num_threads(int(os.environ.get("PYT_THREADS", "2")))
    if args.main or args.step:
        require(TEST_MODE, "explicit --main/--step are allowed only in S0b test mode")
        res = compare_bins(Path(args.main), Path(args.step))
        res["source"] = "main" if res["pass"] else "step143000"
        h = write_json(Path(args.out), res)
        log(f"[S0b test] E1-PROV synthetic: pass={res['pass']} source={res['source']} name_rule_ok={res['name_rule_ok']} "
            f"nonbuffer_diffs={len(res['nonbuffer_differences'])} buffer_diffs={res['n_buffer_differences']} "
            f"main_only={res['main_only']} step_only={res['step_only']} json sha256 {h}")
        return
    require(code_frozen_line() is not None, "the S0b CODE FROZEN line is missing in STATUS.md")
    run_e1prov()


if __name__ == "__main__":
    main()
