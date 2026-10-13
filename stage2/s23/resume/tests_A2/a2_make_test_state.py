#!/usr/bin/env python3
"""Excluded A2 test fixture (R29; TEST keys, CPU only, never the real STATUS.md or the real runs/).

Turns a copy of the finished A1 TEST root (s23/dev/A1_dryrun_20261009T235208Z/T1, copied by run_a2_tests.sh)
into the analogue of the real state before A2: the main phase stopped part-way through R- (the last 5 of the 8
A' reps at the larger R- point are not done), a partial file of one of them is left (an .npy and a .json.tmp,
no .done), the analysis outputs N_match.json / stageA_results.json / REPORT_stageA.md and the analysis cache are
absent, and the GPU ledger is above BOTH frozen limits (an added 12 h ledger line), so the frozen GPU pool would
refuse. It then writes the TEST A2 finished-items manifest, the TEST A2 manifest and (unless --no-frozen-line)
a TEST 'S2.3 AMENDMENT A2 (...) FROZEN' line into <root>/STATUS_test.md. Everything moved away goes to
<root>_removed/ (never deleted); original hashes go to <root>_expect.json.

  S23_ROOT=<root> S23_KEY=hrhv2-s23-TEST CUDA_VISIBLE_DEVICES= python a2_make_test_state.py --root <root>
        [--no-frozen-line] [--corrupt]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

HOME = os.path.expanduser("~")
S23 = os.path.join(HOME, "hrh_v2_stage2_20261008", "s23")
RES = os.path.join(S23, "resume")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--no-frozen-line", action="store_true")
    ap.add_argument("--corrupt", action="store_true")
    a = ap.parse_args()
    root = os.path.realpath(a.root)
    if not root.startswith(os.path.join(os.path.realpath(S23), "dev", "A2_tests_")):
        raise SystemExit("test root must lie under s23/dev/A2_tests_*")
    if os.environ.get("S23_ROOT") != a.root or os.environ.get("S23_KEY") != "hrhv2-s23-TEST" \
            or os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "":
        raise SystemExit("set S23_ROOT=<root> S23_KEY=hrhv2-s23-TEST CUDA_VISIBLE_DEVICES= (empty)")
    sys.path.insert(0, RES)
    import a2_complete as A2
    C = A2.C
    assert C.TEST_MODE and os.path.realpath(C.ROOT) == root
    removed = root + "_removed"
    os.makedirs(removed, exist_ok=False)
    plan = C.load_json(os.path.join(C.PLANS, "stageA_plan.json"))
    l2 = max(plan["rm_points"])
    rm = sorted([it for it in plan["items"] if it["phase"] == "main" and it.get("arm") == "Rm" and it["log2N"] == l2],
                key=lambda it: it["rep"])
    gone = [it for it in rm if it["rep"] >= 3]
    exp = {"removed_items": {}, "results": {}, "planted": [], "rm_log2N": l2}
    for it in gone:
        d = os.path.join(C.RUNS, it["cell"])
        assert os.path.exists(os.path.join(d, it["id"] + ".done")), it["id"]
        fs = [os.path.join(d, it["id"] + s) for s in (".npy", ".json", ".done")] + \
            sorted(glob.glob(os.path.join(d, it["id"] + "__*.npy")))
        exp["removed_items"][it["id"]] = {os.path.relpath(f, C.RUNS): sha(f) for f in fs}
        for f in fs:
            dst = os.path.join(removed, "runs", os.path.relpath(f, C.RUNS))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(f, dst)
    # partial files of the first removed item (an .npy without .done and a .json.tmp)
    it0 = gone[0]
    d0 = os.path.join(C.RUNS, it0["cell"])
    shutil.copy2(os.path.join(removed, "runs", it0["cell"], it0["id"] + ".npy"), os.path.join(d0, it0["id"] + ".npy"))
    with open(os.path.join(d0, it0["id"] + ".json.tmp"), "w") as f:
        f.write('{"partial": tru')
    exp["planted"] = [os.path.join(it0["cell"], it0["id"] + ".npy"), os.path.join(it0["cell"], it0["id"] + ".json.tmp")]
    # analysis outputs and cache moved away (the analysis must recompute everything)
    for name in ("N_match.json", "stageA_results.json", "REPORT_stageA.md"):
        p = os.path.join(C.STAGEA_DIR, name)
        exp["results"][name] = sha(p)
        os.makedirs(os.path.join(removed, "results"), exist_ok=True)
        shutil.move(p, os.path.join(removed, "results", name))
    shutil.move(os.path.join(C.STAGEA_DIR, "cache"), os.path.join(removed, "results", "cache"))
    # ledger above both frozen limits
    with open(os.path.join(C.GPU_LEDGER_DIR, "zz_A2test_budget.tsv"), "w") as f:
        f.write("2026-10-10T00:00:00Z\tA2TEST_ledger_above_both_frozen_limits\t43200.000\n")
    exp["ledger_h_before"] = C.gpu_hours_used()
    exp["frozen_soft_ok_before"] = C.gpu_budget_ok(soft=True)
    led = {}
    for p in glob.glob(os.path.join(C.GPU_LEDGER_DIR, "*.tsv")):
        with open(p) as f:
            led[os.path.basename(p)] = sum(1 for _ in f)
    exp["ledger_lines_before"] = led
    # TEST A2 finished-items manifest and TEST A2 manifest (+ TEST FROZEN line)
    rc = A2.cmd_manifest(A2.A2_FIN)
    assert rc == 0, rc
    os.chmod(A2.A2_FIN, 0o444)
    if a.corrupt:   # a finished item changes after the manifest: prep must refuse
        it = next(x for x in plan["items"] if x["phase"] == "main" and x.get("arm") == "0" and x["kind"] == "A")
        pj = os.path.join(C.RUNS, it["cell"], it["id"] + ".json")
        os.chmod(pj, 0o644)
        with open(pj, "a") as f:
            f.write("\n")
        exp["corrupted"] = os.path.relpath(pj, C.RUNS)
    rc = A2.cmd_a2manifest(A2.A2_MAN)
    assert rc == 0, rc
    os.chmod(A2.A2_MAN, 0o444)
    h = sha(A2.A2_MAN)
    if not a.no_frozen_line:
        C.append_status("amend-A2", f"{A2.A2_MARK} (TEST root) A2_MANIFEST.sha256 sha256 {h}")
    exp["a2_manifest_sha256"] = h
    with open(root + "_expect.json", "w") as f:
        json.dump(exp, f, indent=1)
    print(f"A2 test state: removed {len(gone)} R- items at log2N={l2} ({gone[0]['id']} .. {gone[-1]['id']}); planted "
          f"{len(exp['planted'])} partial files; ledger {exp['ledger_h_before']:.3f} h (frozen soft ok: "
          f"{exp['frozen_soft_ok_before']}); corrupt={a.corrupt}; frozen line={not a.no_frozen_line}")


if __name__ == "__main__":
    main()
