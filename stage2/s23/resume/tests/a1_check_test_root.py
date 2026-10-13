#!/usr/bin/env python3
"""Excluded A1 test checks (R29) on a TEST root built by a1_make_test_root.py and run by
launch_stageA_resume.sh --test.  Scenarios:
  T1   fake compute-apps list = own workers only: full sequence, rc 0
  T4   T1 root launched again: every step skipped or no-op, no chmod-444 file rewritten, rc 0
  T2a  guard threshold 0 s (test override): speed-guard pause, rc 6
  T2b  T2 root launched again with the frozen threshold: completes, rc 0
  T3   fake compute-apps list with a foreign process: rule A1.2(d) pause before the gates phase, rc 7
  T5a  as T2a with 0.1 s snapshots, so the guard fires while the frozen GPU workers are alive (kill path)
  T5b  T5 root launched again: completes, rc 0
  S23_ROOT=<root> S23_KEY=hrhv2-s23-TEST python a1_check_test_root.py --root R --scenario T1 --rc 0
"""
import argparse
import glob
import json
import os
import statistics
import sys

HOME = os.path.expanduser("~")
S23 = os.path.join(HOME, "hrh_v2_stage2_20261008", "s23")
SRC = os.path.join(S23, "src")
RES = os.path.join(S23, "resume")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenario", required=True, choices=["T1", "T4", "T2a", "T2b", "T3", "T5a", "T5b"])
    ap.add_argument("--rc", type=int, required=True)
    a = ap.parse_args()
    sys.path.insert(0, RES)
    import a1_resume as R
    C = R.C
    assert C.TEST_MODE and C.ROOT == a.root
    res = []

    def ok(name, cond, detail=""):
        res.append((name, bool(cond), detail))

    scen = {"T5a": "T2a", "T5b": "T2b"}.get(a.scenario, a.scenario)
    st = open(C.STATUS).read()
    atts = sorted(glob.glob(os.path.join(a.root, "resume", "attempts", "*")))
    last = atts[-1] if atts else None
    stagea = C.STAGEA_DIR
    plan = R.load_plan()
    fr = C.load_json(C.FROZEN_PRED)
    alive = R.stage_a_workers_alive()
    ok("no Stage A worker process left running", not alive, alive)
    ok("attempt folder exists", last is not None)
    ok("rc file matches", last and open(os.path.join(last, "rc")).read().strip() == str(a.rc), a.rc)
    if scen in ("T1", "T2b", "T4"):
        ok("launcher rc 0 (Stage A done)", a.rc == 0, a.rc)
        for f in ("eps_arm0.json", "A1_g4_inputs.json", "N_match.json"):
            p = os.path.join(stagea, f)
            ok(f"{f} exists and is write-protected", os.path.exists(p) and not R.writable(p))
        ok("stageA_results.json and REPORT_stageA.md exist", os.path.exists(os.path.join(stagea, "stageA_results.json"))
           and os.path.exists(os.path.join(stagea, "REPORT_stageA.md")))
        g4in = C.load_json(os.path.join(stagea, "A1_g4_inputs.json"))
        g4 = C.load_json(os.path.join(stagea, "g4_decision.json"))
        first3 = [it["id"] for it in plan["items"] if it["phase"] == "pre" and it["kind"] == "A"][:3]
        for key, grp in g4in["groups"].items():
            if not key.startswith("A_pilot_l"):
                continue
            l2 = key.split("_l")[1]
            free = [r["gpu_section_s"] for r in grp["rows"] if r["free"]]
            allv = [r["gpu_section_s"] for r in grp["rows"] if r["done"]]
            want = statistics.median(free)
            got = float(g4["median_gpu_s_per_A"][l2])
            ok(f"G4 median at log2N={l2} = median of the free timings only ({len(free)}/{len(allv)})",
               abs(got - want) < 1e-12, f"g4 {got} free-median {want} all-median {statistics.median(allv)}")
        masked = sorted(r["id"] for g in g4in["groups"].values() for r in g["rows"] if not r["free"])
        if scen in ("T1", "T4"):
            ok("exactly the 3 pre-A1 pilots are masked (no record)", masked == sorted(first3), masked)
        else:
            ok("the 3 pre-A1 pilots are among the masked ones", set(first3) <= set(masked), masked)
        ok("STATUS: G4 inputs written before the sampled-gates line",
           0 <= st.find("A1 G4 inputs written BEFORE") < st.find("S2.3 Stage A sampled gates"))
        ok("STATUS: analysis done line", "S2.3 Stage A analysis done" in st)
    if scen in ("T1", "T2a"):
        mv = glob.glob(os.path.join(C.RUNS, R.PARTIAL, os.path.basename(atts[0]), "**", "*"), recursive=True)
        mv = sorted(os.path.basename(p) for p in mv if os.path.isfile(p))
        ok("first attempt moved the 4 partial files of the 2 killed pilots aside (not deleted)", len(mv) == 4, mv)
        ok("no partial file left in runs/", R.scan_partials(plan) == [] or scen == "T2a", R.scan_partials(plan))
        prep = json.load(open(os.path.join(atts[0], "prep.json")))
        ok("prep verified the manifest items", prep["n_manifest_files"] > 0 and not prep["problems"], prep["problems"])
        ok("STATUS: prep OK line", "A1 prep OK" in st)
    if scen == "T1":
        g = json.load(open(os.path.join(last, "guard.json")))
        ok("speed guard evaluated and passed", g.get("evaluated") and g.get("decision") == "continue", g)
        ok("redone pilots have .done", all(os.path.exists(R.item_paths(it)[3]) for it in plan["items"]
                                           if it["phase"] == "pre"))
        ok("snapshot log written", len(glob.glob(os.path.join(R.SNAP_DIR, "*.tsv"))) >= 1)
        ok("ledger includes the correction file", any("A1_correction" in ln[2] for ln in R.ledger_lines()))
    if scen == "T4":
        ok("STATUS: gates not re-run", "gates phase already done" in st)
        ok("STATUS: nmatch not re-run", "N_match.json exists, nmatch phase not re-run" in st)
        ok("STATUS: main analysis not re-run", "main analysis not re-run" in st)
        ok("only one sampled-gates line", st.count("S2.3 Stage A sampled gates") == 1)
    if scen == "T2a":
        ok("launcher rc 6 (speed-guard pause)", a.rc == 6, a.rc)
        p = os.path.join(last, "PAUSED")
        ok("PAUSED record written", os.path.exists(p))
        g = json.load(open(os.path.join(last, "guard.json")))
        ok("guard decision PAUSE", g.get("decision") == "PAUSE", g.get("median_s"))
        ok("no eps_arm0.json / g4_decision.json (no statistic opened)",
           not os.path.exists(os.path.join(stagea, "eps_arm0.json")) and not os.path.exists(os.path.join(stagea, "g4_decision.json")))
        sw = g.get("stopped_workers", [])
        lf = os.path.join(C.GPU_LEDGER_DIR, f"A1_guard_{os.path.basename(last)}.tsv")
        n_sig = sum(1 for s in sw if s["signalled"])
        ok("every signalled worker is charged in the ledger", n_sig == 0 or
           (os.path.exists(lf) and len(open(lf).read().splitlines()) == n_sig), (sw, lf))
        ok("STATUS: guard PAUSE line", "A1 speed guard (rule A1.3) PAUSE" in st)
        if a.scenario == "T5a":
            ok("kill path exercised: at least one live frozen GPU worker was signalled and charged", n_sig >= 1, sw)
    if scen == "T2b":
        prep = json.load(open(os.path.join(last, "prep.json")))
        ok("second attempt checked the items finished in the first attempt", prep["n_finished_after_A1"] > 0 and not prep["problems"],
           prep)
    if scen == "T3":
        ok("launcher rc 7 (too few free timings)", a.rc == 7, a.rc)
        ok("PAUSED_G4 written", os.path.exists(os.path.join(last, "PAUSED_G4")))
        ok("no eps_arm0.json / g4_decision.json / A1_g4_inputs.json (analyze_A not run)",
           not any(os.path.exists(os.path.join(stagea, f)) for f in ("eps_arm0.json", "g4_decision.json", "A1_g4_inputs.json")))
        ins = json.load(open(os.path.join(last, "A1_g4_inputs_insufficient.json")))
        ok("foreign process -> no new solve classified free",
           all(not r["free"] for g in ins["groups"].values() for r in g["rows"]),
           {k: (v["free"], v["planned"]) for k, v in ins["groups"].items()})
        ok("STATUS: A1 G4 NOT EVALUATED line", "A1 G4 NOT EVALUATED" in st)
    npass = sum(r[1] for r in res)
    for name, p, d in res:
        print(("PASS " if p else "FAIL ") + name + ("" if p else f"  [{d}]"))
    print(f"{a.scenario}: {npass}/{len(res)} checks passed")
    return 0 if npass == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
