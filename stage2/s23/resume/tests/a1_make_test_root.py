#!/usr/bin/env python3
"""Excluded A1 test fixture (R29; TEST keys, CPU only, never the real STATUS.md or the real runs/).

Builds a tiny Stage A root under s23/dev/A1_dryrun*/ that mimics the stopped state of 2026-10-09:
pre-step and plan written (plan chmod 444), every CPU pre item done, the first 3 arm-0 A pilots done
(ledger lines under gpu0_pre / gpu1_pre), partial files of the next two A pilots (a .npy and a .json.tmp,
no .done: the r015 / r016 analogue), a ledger correction file, the finished-items manifest, the A1 manifest
and a TEST 'S2.3 AMENDMENT A1 (resume) FROZEN' line in <root>/STATUS_test.md.

  S23_ROOT=<root> S23_KEY=hrhv2-s23-TEST CUDA_VISIBLE_DEVICES= python a1_make_test_root.py --root <root>
"""
import argparse
import os
import subprocess
import sys

HOME = os.path.expanduser("~")
S23 = os.path.join(HOME, "hrh_v2_stage2_20261008", "s23")
SRC = os.path.join(S23, "src")
RES = os.path.join(S23, "resume")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = os.path.realpath(a.root)
    if not root.startswith(os.path.join(os.path.realpath(S23), "dev", "A1_dryrun")):
        raise SystemExit("test root must lie under s23/dev/A1_dryrun*")
    if os.environ.get("S23_ROOT") != a.root or os.environ.get("S23_KEY") != "hrhv2-s23-TEST" \
            or os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "":
        raise SystemExit("set S23_ROOT=<root> S23_KEY=hrhv2-s23-TEST CUDA_VISIBLE_DEVICES= (empty)")
    sys.path.insert(0, SRC)
    import s23common as C
    C.cap_threads(1)
    import numpy as np
    import torch
    import cfg as CF
    import make_plan as MP
    import prestep as PS
    import worker as WK

    assert C.TEST_MODE and C.ROOT == a.root
    c = CF.tiny()                                   # exactly the frozen unit-test configuration (conftest.py)
    c["g0"].update({"lam_min": 0.0, "omega": 0.0, "twoview_ptp": 0.0, "ex8_dgap": 0.0})
    c["grid"].update({"j_lo": -2, "j_hi": 2, "br_offset": 8, "br_min_log2": 8, "dep_offsets": [-6, 0, 6],
                      "pilot_offsets": [-6, 6]})
    c["runs"].update({"g1b_offsets": [-6, 0, 6], "rm_points": ["br", -6]})
    fr, _arr = PS.run(cfg=c, dev=torch.device("cpu"), write=True, status=True)
    assert fr["gates_pass"], "tiny pre-step gates failed"
    plan = MP.build(C.load_json(C.FROZEN_PRED), fr["cfg"])
    pp = os.path.join(C.PLANS, "stageA_plan.json")
    C.dump_json(plan, pp)
    os.chmod(pp, 0o444)
    inst = WK.instance_from(C.load_json(C.FROZEN_PRED))
    fr2, arr2 = WK.load_frozen()
    pre = [it for it in plan["items"] if it["phase"] == "pre"]
    for it in pre:
        if it["kind"] in ("B", "G3B"):
            WK.do_item(it, inst, fr2, arr2, None, "cpu0_pre")
    gpu = WK.GpuCtx(inst, fr2, arr2, "gpu0_pre")
    A = [it for it in pre if it["kind"] == "A"]
    for i, it in enumerate(A[:3]):
        WK.do_item(it, inst, fr2, arr2, gpu, "gpu0_pre" if i % 2 == 0 else "gpu1_pre")
    for it in A[3:5]:                               # killed in flight: partial files, no .done
        d = os.path.join(C.RUNS, it["cell"])
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, it["id"] + ".npy"), np.zeros(3))
        with open(os.path.join(d, it["id"] + ".json.tmp"), "w") as f:
            f.write("{partial")
    corr = os.path.join(C.GPU_LEDGER_DIR, "A1_ledger_correction.tsv")
    with open(corr, "w") as f:
        f.write(f"2026-10-09T23:06:27Z\tA1_correction_killed_{A[3]['id']}\t1.500\n"
                f"2026-10-09T23:06:27Z\tA1_correction_killed_{A[4]['id']}\t1.000\n")
    os.chmod(corr, 0o444)
    os.makedirs(os.path.join(a.root, "resume"), exist_ok=True)
    py = sys.executable
    man = os.path.join(a.root, "resume", "A1_finished_items.sha256")
    subprocess.run([py, os.path.join(RES, "a1_resume.py"), "manifest", "--out", man], check=True, cwd=RES)
    os.chmod(man, 0o444)
    a1m = os.path.join(a.root, "resume", "A1_MANIFEST.sha256")
    subprocess.run([py, os.path.join(RES, "a1_resume.py"), "a1manifest", "--out", a1m], check=True, cwd=RES)
    os.chmod(a1m, 0o444)
    C.append_status("amend", f"S2.3 AMENDMENT A1 (resume) FROZEN (TEST root, excluded): A1_MANIFEST.sha256 sha256 "
                             f"{C.sha256_file(a1m)}")
    print("test root ready:", a.root, "| A pilots done:", [it["id"] for it in A[:3]], "| partial:", [it["id"] for it in A[3:5]],
          "| ledger h:", round(C.gpu_hours_used(), 6))


if __name__ == "__main__":
    main()
