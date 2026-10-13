#!/usr/bin/env python3
"""Excluded A2 test checks (R29; TEST roots only). Reads <root>, <root>_removed/ and <root>_expect.json written by
a2_make_test_state.py, and checks one scenario of run_a2_tests.sh:

  full      first launch: rc 0; the 5 removed R- items redone from their frozen keys with .npy hash-equal to
            the originals; partial files moved aside (not deleted); GPU pool ran through a2_worker.py while the
            ledger was above both frozen limits; no budget STOP; ledger lines added; nmatch and main analysis
            re-run with results equal to the originals (stageA_results.json up to its ledger field);
            markers A2.started + A2.done; STATUS lines in the frozen order.
  stale     relaunch with the old markers in place: rc 9, no new attempt, nothing run.
  rerun     relaunch after the markers were moved aside: rc 0, nothing redone, no ledger line, both analysis
            phases skipped because their outputs exist.
  corrupt   a finished item changed after the A2 finished-items manifest: rc 2, prep REFUSED, no worker.
  nofrozen  no A2 FROZEN line: rc 2, verification refused, no attempt, no STATUS START line.
"""
import argparse
import glob
import hashlib
import json
import os
import sys


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def ledger_lines(root):
    out = {}
    for p in glob.glob(os.path.join(root, "logs", "gpu_ledger", "*.tsv")):
        with open(p) as f:
            out[os.path.basename(p)] = f.read().splitlines()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenario", required=True, choices=["full", "stale", "rerun", "corrupt", "nofrozen"])
    ap.add_argument("--rc", type=int, required=True)
    a = ap.parse_args()
    root = os.path.realpath(a.root)
    exp = json.load(open(root + "_expect.json"))
    runs = os.path.join(root, "runs")
    st_path = os.path.join(root, "resume")
    stA = os.path.join(root, "results", "stageA")
    status = open(os.path.join(root, "STATUS_test.md")).read().splitlines()
    a2_lines = [ln for ln in status if "[s23-run-A2]" in ln or "[s23-run]" in ln]
    atts = sorted(glob.glob(os.path.join(st_path, "A2_attempts", "*")))
    state_p = root + "_state.json"
    res = []

    def ok(name, cond, detail=""):
        res.append(bool(cond))
        print(f"{'PASS' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail and not cond else ''}")

    alive = []
    envs = {b"S23_ROOT=" + a.root.encode(), b"S23_ROOT=" + root.encode()}
    for d in glob.glob("/proc/[0-9]*"):
        try:
            argv = open(os.path.join(d, "cmdline"), "rb").read().split(b"\0")
            if any(x.endswith(b"a2_worker.py") or x == b"worker.py" for x in argv) and \
                    envs & set(open(os.path.join(d, "environ"), "rb").read().split(b"\0")):
                alive.append(d)
        except Exception:  # noqa: BLE001
            pass
    ok("no A2 / Stage A worker of this TEST root left running", not alive, str(alive))
    led = ledger_lines(root)

    if a.scenario == "full":
        ok("launcher rc 0 (Stage A done)", a.rc == 0, f"rc={a.rc}")
        ok("A2.started and A2.done exist, no A2.failed",
           os.path.exists(os.path.join(st_path, "A2.started")) and os.path.exists(os.path.join(st_path, "A2.done"))
           and not os.path.exists(os.path.join(st_path, "A2.failed")))
        ok("exactly one attempt folder with rc 0", len(atts) == 1 and open(os.path.join(atts[0], "rc")).read().strip() == "0")
        att = atts[0] if atts else ""
        att_id = os.path.basename(att)
        pids = open(os.path.join(att, "pids_main")).read() if att else ""
        ok("GPU pool ran through a2_worker.py, CPU pool through the frozen worker.py",
           pids.count("a2_worker.py") == 2 and "cpu0" in pids and "worker.py" in pids.split("cpu0", 1)[1].splitlines()[0]
           and "gpu0 rc=0 gpu1 rc=0 cpu0 rc=0" in pids, pids)
        budget = open(os.path.join(att, "budget.out")).read() if att else ""
        ok("frozen predicate refused before the run (ledger above both limits)",
           "gpu_budget_ok(soft=True)=False" in budget and "gpu_budget_ok(soft=False)=False" in budget
           and exp["frozen_soft_ok_before"] is False, budget)
        nsame, ndone = 0, 0
        for iid, files in exp["removed_items"].items():
            for rel, h in files.items():
                p = os.path.join(runs, rel)
                if rel.endswith(".done"):
                    ndone += os.path.exists(p)
                elif rel.endswith(".npy"):
                    nsame += os.path.exists(p) and sha(p) == h
        nnpy = sum(1 for files in exp["removed_items"].values() for r in files if r.endswith(".npy"))
        ok(f"all {len(exp['removed_items'])} removed R- items redone (.done)", ndone == len(exp["removed_items"]), f"{ndone}")
        ok(f"redone K-hat arrays hash-equal to the originals (frozen keys -> same solution), {nsame}/{nnpy}",
           nsame == nnpy, f"{nsame}/{nnpy}")
        js_ok = True
        for iid, files in exp["removed_items"].items():
            rel = [r for r in files if r.endswith(".json")][0]
            j = json.load(open(os.path.join(runs, rel)))
            orig = json.load(open(os.path.join(root + "_removed", "runs", rel)))
            js_ok &= j["id"] == orig["id"] and j["item"] == orig["item"] and j.get("cert_pass") == orig.get("cert_pass") \
                and j.get("path") == orig.get("path")
        ok("redone .json: same id, plan item, certificate and path as the originals", js_ok)
        pd = os.path.join(runs, "_partial_20261010", "A2_" + att_id)
        ok("planted partial files moved aside into runs/_partial_20261010/A2_<attempt>/ (not deleted)",
           all(os.path.exists(os.path.join(pd, r)) for r in exp["planted"]), pd)
        ok("no .json.tmp left in runs/ (outside _partial_)",
           not [p for p in glob.glob(os.path.join(runs, "**", "*.tmp"), recursive=True) if "_partial_" not in p])
        logs = "".join(open(p).read() for p in glob.glob(os.path.join(root, "logs", "worker_gpu*_main.log")))
        ok("no budget STOP / HARD STOP in the GPU worker logs", "STOP" not in logs)
        ok("every redone item has a 'done' line in a GPU worker log", all(f"done {iid} " in logs for iid in exp["removed_items"]))
        new = {k: len(v) - exp["ledger_lines_before"].get(k, 0) for k, v in led.items()}
        ids_in_ledger = {ln.split("\t")[1] for k in ("gpu0_main.tsv", "gpu1_main.tsv") for ln in led.get(k, [])}
        ok("GPU ledger kept recording: one line per redone item + GpuCtx setup lines",
           all(iid in ids_in_ledger for iid in exp["removed_items"]) and sum(new.values()) >= len(exp["removed_items"]),
           str(new))
        ok("added 12 h ledger line untouched", led.get("zz_A2test_budget.tsv", [""])[0].endswith("\t43200.000"))
        nm = os.path.join(stA, "N_match.json")
        ok("N_match.json rewritten by the frozen nmatch phase, hash-equal to the original, chmod 444",
           os.path.exists(nm) and sha(nm) == exp["results"]["N_match.json"] and not (os.stat(nm).st_mode & 0o222))
        r_new = json.load(open(os.path.join(stA, "stageA_results.json")))
        r_old = json.load(open(os.path.join(root + "_removed", "results", "stageA_results.json")))
        g_new, g_old = r_new.pop("gpu_hours_ledger", None), r_old.pop("gpu_hours_ledger", None)
        ok("stageA_results.json equal to the original except its ledger field", r_new == r_old)
        ok("results ledger field includes the 12 h line", g_new is not None and g_new > 12.0 and g_old < 1.0, f"{g_new} {g_old}")
        rep_new = [ln for ln in open(os.path.join(stA, "REPORT_stageA.md")).read().splitlines() if not ln.startswith("GPU-hours")]
        rep_old = [ln for ln in open(os.path.join(root + "_removed", "results", "REPORT_stageA.md")).read().splitlines()
                   if not ln.startswith("GPU-hours")]
        ok("REPORT_stageA.md equal to the original except the GPU-hours line", rep_new == rep_old)
        order = ["A2 attempt " + att_id + " START", "A2 prep OK", "frozen predicate gpu_budget_ok(soft=True)=False",
                 "main phase complete", "N_match written BEFORE any cross statistic", "Stage A analysis done",
                 "A2 attempt " + att_id + " END rc=0"]
        pos = []
        start = [i for i, ln in enumerate(status) if order[0] in ln]
        i0 = start[0] if start else len(status)    # the copied STATUS_test.md holds older A1-test lines: search after START
        for key in order:
            hit = [i for i, ln in enumerate(status) if key in ln and i >= i0]
            pos.append(hit[0] if hit else -1)
        ok("STATUS_test lines present and in the frozen order (START, prep, budget, main complete, N_match, analysis, END)",
           -1 not in pos and pos == sorted(pos), str(pos))
        with open(state_p, "w") as f:
            json.dump({"ledger": {k: len(v) for k, v in led.items()}, "n_status": len(status), "n_atts": len(atts)}, f)
    elif a.scenario == "stale":
        prev = json.load(open(state_p))
        ok("launcher rc 9 (stale markers)", a.rc == 9, f"rc={a.rc}")
        ok("no new attempt folder", len(atts) == prev["n_atts"])
        ok("no new ledger line", {k: len(v) for k, v in led.items()} == prev["ledger"])
        ok("no new STATUS line", len(status) == prev["n_status"])
    elif a.scenario == "rerun":
        prev = json.load(open(state_p))
        ok("launcher rc 0", a.rc == 0, f"rc={a.rc}")
        ok("one new attempt folder", len(atts) == prev["n_atts"] + 1)
        ok("no new ledger line (nothing redone, no GpuCtx built)", {k: len(v) for k, v in led.items()} == prev["ledger"])
        tail = status[prev["n_status"]:]
        ok("nmatch and main analysis skipped because their outputs exist",
           any("N_match.json exists, nmatch phase not re-run" in ln for ln in tail)
           and any("stageA_results.json exists, main analysis not re-run" in ln for ln in tail))
        ok("no new analysis line", not any("Stage A analysis done" in ln or "N_match written" in ln for ln in tail))
        ok("A2.done written again", os.path.exists(os.path.join(st_path, "A2.done")))
    elif a.scenario == "corrupt":
        ok("launcher rc 2 (prep refused)", a.rc == 2, f"rc={a.rc}")
        ok("A2.failed written, no A2.done", os.path.exists(os.path.join(st_path, "A2.failed"))
           and not os.path.exists(os.path.join(st_path, "A2.done")))
        ok("STATUS: A2 prep REFUSED names the changed item",
           any("A2 prep REFUSED" in ln and "MISMATCH" in ln for ln in status))
        ok("no worker started", not any(os.path.exists(os.path.join(x, "pids_main")) for x in atts))
        ok("no new ledger line", {k: len(v) for k, v in led.items()} == exp["ledger_lines_before"])
        ok("removed items still not done", not any(os.path.exists(os.path.join(runs, r)) for files in exp["removed_items"].values()
                                                    for r in files if r.endswith(".done")))
        ok("planted partial files NOT moved (prep refused before moving)",
           all(os.path.exists(os.path.join(runs, r)) for r in exp["planted"]))
    elif a.scenario == "nofrozen":
        ok("launcher rc 2 (verification refused)", a.rc == 2, f"rc={a.rc}")
        ok("A2.failed says verification failed", os.path.exists(os.path.join(st_path, "A2.failed"))
           and "verification failed" in open(os.path.join(st_path, "A2.failed")).read())
        ok("no attempt folder, no START line", not atts and not any("A2 attempt" in ln for ln in status))
        ok("no new ledger line", {k: len(v) for k, v in led.items()} == exp["ledger_lines_before"])
    print(f"{a.scenario}: {sum(res)}/{len(res)} checks passed")
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
