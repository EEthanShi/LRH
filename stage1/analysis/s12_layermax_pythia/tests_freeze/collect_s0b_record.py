#!/usr/bin/env python
"""Pythia axis, step S0b: the consolidated S0b test record of role pyt-code-freeze (2026-10-10). NEW FILE (PYT-AXIS).

Reads the outputs of one run of tests_freeze/run_s0b_integrated.sh (tag <TAG>) and writes
analysis/s12_layermax_pythia/s0b_scratch/FZ/run_<TAG>/S0B_TEST_RECORD_PYT.json with
  * every step of the battery with its exit code and launcher retries (exit 75 = thread tokens busy, nothing started);
  * per battery the pass flag read from the battery's own record (extraction CPU report, infrastructure report, E's
    test_log_E.json, I's step exit codes and records, R's records, fixture X report), with the record's sha256;
  * "pass": true only if every battery passes (VERDICTS_SPEC_PYT.md section 14 "--require-pass");
  * the inventory of EVERY failed pre-freeze test run of the axis (the review asked for it in the CODE FROZEN line):
    (a) every launcher marker NAME.failed found under analysis/s12_layermax_pythia/ (scratch included),
        results/S1_2_layermax_pythia/logs/ and extract/pythia_axis/s0b_test_out/, with its end line and the last line
        of its log; markers of runs whose non-zero exit was the test's expected outcome are flagged "expected";
    (b) failures that leave no marker (launcher refusals with exit 75, failing checks inside a run that exited 0, test
        rounds superseded before the final round), as reported by the writer roles (source named for each).
Reads only; writes the one record. No verdict value is read or written (pass flags and counts only), except the
section-K branch values that fixture X records under "coverage" for its stand-in and synthetic TEST INPUTS (no Pythia
byte; lead ruling R3: which categories and modifiers the fixture exercised); they are copied, never part of "pass".
S0b test mode only (PYT_S0B_TEST=1 before CODE FROZEN; else exit 3).
usage: PYT_S0B_TEST=1 venv/bin/python analysis/s12_layermax_pythia/tests_freeze/collect_s0b_record.py --tag <TAG>
"""
from __future__ import annotations

import argparse
import calendar
import glob
import hashlib
import json
import os
import re
import sys
import time

PROJ = os.path.realpath(os.path.expanduser("~/hrh_v2_stage1_20261008"))
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_pythia")
EXTR = os.path.join(PROJ, "extract", "pythia_axis")
RESLOG = os.path.join(PROJ, "results", "S1_2_layermax_pythia", "logs")
sys.path.insert(0, AXIS)
import guard  # noqa: E402

PLAN_SHA = "6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b"
# runs whose non-zero exit is the designed outcome of the test (documented by the writer role)
EXPECTED_NONZERO = {  # launcher NAME -> (designed exit code, reason); any other exit code is an unexpected failure
    "pytE_cpu_main_noresume": (2, "E: run_layermax.py must refuse a non-empty output directory with exit 2 (plan "
                                  "section 11, interrupted S7 restarts); E's own driver counts rc 2 as pass"),
    "pytE_t16_main_noresume": (2, "E (pre-final round): the same designed exit-2 refusal"),
    "pyt_R_s0b_e2e_2": (1, "R: rc 1 by design, the R-only fixture has no E/I/forecast verdict files to compare "
                           "(MISSING); compute phase complete"),
    "pyt_R_s0b_e2e_final_1": (1, "R: rc 1 by design (as e2e_2)"),
    "pyt_R_s0b_e2e_final_2": (1, "R: rc 1 by design (as e2e_2)"),
}
# failures without a launcher marker, as reported by the writer roles (STATUS lines and role reports of 2026-10-10)
NO_MARKER = [
    dict(role="extract", run="pyt_s0b_smoke_extract_2 (GPU, 0.085 GPU-min)", what="rc 0 but a no-op: it re-read the "
         "first smoke's output instead of extracting; counted in the GPU ledger; fixed (one output dir per run)",
         source="notes .../pythia_axis/extract.md section 4.3"),
    dict(role="infra", run="infra Part B stand-in rerun at 4 threads", what="refused by launch_pyt.sh (exit 75, tokens "
         "taken by other roles), nothing ran; rerun at 2 threads rc 0", source="role report pyt-code-infra section 3.1"),
    dict(role="infra", run="manual forecast_pyt.py rerun before the t11 fix", what="KeyError mu_band (same error as "
         "infra_tests_run1 t11); fixed", source="role report pyt-code-infra section 3.2"),
    dict(role="E", run="E battery final_20261010T015514Z (19/25)", what="pytE_cpu_precompute and pytE_cpu_main refused "
         "by launch_pyt.sh (exit 75, tokens held by another role), so the dependent CPU-smoke steps failed (markers "
         "listed under (a)); superseded by final_20261010T015728Z 25/25", source="STATUS 02:03:29Z, role report E"),
    dict(role="I", run="pyt_I_c02_vp9", what="refused by launch_pyt.sh (exit 75), nothing ran",
         source="role report pyt-code-I section 3"),
    dict(role="I", run="I CPU smoke (both rounds)", what="V-P4 strict form FALSE: debiased-CKA null entries near 0 "
         "differ by 1.34e-9 relative between the CPU rerun and the frozen GPU files (counts identical; V-P5 mixed form "
         "true); the run exits 0; expected to pass at S6 on the GPU", source="STATUS 02:07:24Z, role report I"),
    dict(role="I", run="36 first-version runs (pre-contract code)", what="rc 0, superseded by the contract rewrite; "
         "moved with outputs to s0b_scratch/I/pre_contract_s0b_test_20261010T015516Z/", source="STATUS 02:07:24Z"),
    dict(role="R", run="pyt_R_s0b_e2e_1", what="refused by launch_pyt.sh (exit 75: 4 tokens requested, 2 free), "
         "nothing ran, no marker", source="R_test_runs.json"),
]


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def jl(p):
    return json.load(open(p))


def newest(pattern: str, after: float):
    c = [p for p in glob.glob(pattern) if os.path.getmtime(p) >= after]
    return max(c, key=os.path.getmtime) if c else None


def rec(p, pass_=None, **kw):
    if p is None or not os.path.exists(p):
        return dict(path=None, sha256=None, pass_=False, note="record missing", **kw)
    return dict(path=os.path.relpath(p, PROJ), sha256=sha(p), pass_=pass_, **kw)


def failed_markers() -> list[dict]:
    out = []
    pats = [os.path.join(AXIS, "**", "*.failed"), os.path.join(RESLOG, "*.failed"),
            os.path.join(EXTR, "s0b_test_out", "**", "*.failed")]
    seen = set()
    for pat in pats:
        for p in sorted(glob.glob(pat, recursive=True)):
            if p in seen:
                continue
            seen.add(p)
            name = os.path.basename(p)[:-len(".failed")]
            log = p[:-len(".failed")] + ".log"
            last = ""
            if os.path.exists(log):
                lines = [ln for ln in open(log, errors="replace").read().splitlines()
                         if ln.strip() and not ln.startswith("[launch_pyt]")]
                last = (lines[-1] if lines else "")[:200]
            base = name.rsplit("_try", 1)[0]
            end = open(p).read().strip()
            m = re.search(r"rc=(-?\d+)", end)
            got = int(m.group(1)) if m else None
            exp = EXPECTED_NONZERO.get(base)
            out.append(dict(marker=os.path.relpath(p, PROJ), end=end, log_last_line=last,
                            expected=(exp[1] if exp and exp[0] == got else None)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    if not guard.test_mode():
        print("[collect_s0b_record] REFUSED (exit 3): S0b test mode only", file=sys.stderr)
        sys.exit(3)
    tag = args.tag
    if not tag.replace("T", "").replace("Z", "").isdigit():
        sys.exit("bad --tag")
    sc = guard.SCRATCH
    fz = os.path.join(sc, "FZ", f"run_{tag}")
    t0 = calendar.timegm(time.strptime(tag, "%Y%m%dT%H%M%SZ")) - 1
    steps = []
    for ln in open(os.path.join(fz, "steps.tsv")).read().splitlines()[1:]:
        name, rc, tries, a, b = ln.split("\t")
        steps.append(dict(step=name, rc=int(rc), tries=int(tries), start_utc=a, end_utc=b))
    rc = {s["step"]: s["rc"] for s in steps}
    B = {}
    # 1 extraction-side CPU tests
    rid = open(os.path.join(EXTR, "s0b_test_out", "latest_cpu_run.txt")).read().strip()
    p = os.path.join(EXTR, "s0b_test_out", "runs", rid, "test_report_cpu.json")
    d = jl(p) if os.path.exists(p) and os.path.getmtime(p) >= t0 else None
    B["extraction_cpu"] = rec(p if d else None, pass_=bool(d and d["n_fail"] == 0 and rc.get("X_extract_cpu") == 0),
                              n_pass=(d or {}).get("n_pass"), n_fail=(d or {}).get("n_fail"))
    # 2 infrastructure tests and Part B checks
    p = newest(os.path.join(sc, "PB", "tests", "*", "report.json"), t0)
    d = jl(p) if p else {}
    B["infra_tests"] = rec(p, pass_=bool(d.get("pass") and rc.get("infra_tests") == 0), n_pass=d.get("n_pass"),
                           n=d.get("n"))
    p = os.path.join(sc, "PB", "partB_vp9", "partB_retro_checks_pyt.json")
    B["partb_checks_vp9"] = rec(p, pass_=rc.get("partb_checks_vp9") == 0)
    # 3 E battery
    p = newest(os.path.join(sc, "E", "final_*", "test_log_E.json"), t0)
    d = jl(p) if p else {}
    B["E_battery"] = rec(p, pass_=bool(d.get("pass") and rc.get("E_battery") == 0), n_pass=d.get("n_pass"),
                         n_runs=d.get("n_runs"),
                         records={k: rec(os.path.join(PROJ, v), pass_=jl(os.path.join(PROJ, v)).get("pass"))
                                  for k, v in (d.get("records_for_s0b_freeze") or {}).items()})
    # 4 I battery (step exit codes printed by run_all_s0b.sh, and I's own records)
    out = os.path.join(fz, "logs", "I_battery.out")
    irc = {}
    for ln in open(out).read().splitlines() if os.path.exists(out) else []:
        if ln.startswith("step ") and " rc=" in ln:
            k, v = ln[5:].rsplit(" rc=", 1)
            irc[k.strip()] = int(v)
    irec = {"vp9": newest(os.path.join(sc, "I", "vp9", "test_report_*.json"), t0),
            "vp8": os.path.join(sc, "I", "vp8", "vp8_a_b_c.json"),
            "gate": newest(os.path.join(sc, "I", "dryrun", "gate_report_*.json"), t0)}
    irec = {k: rec(v if v and os.path.exists(v) and os.path.getmtime(v) >= t0 else None,
                   pass_=(jl(v).get("pass") if v and os.path.exists(v) and os.path.getmtime(v) >= t0 else False))
            for k, v in irec.items()}
    B["I_battery"] = dict(steps=irc, n_steps=len(irc), records=irec,
                          pass_=bool(len(irc) >= 12 and all(v == 0 for v in irc.values()) and rc.get("I_battery") == 0
                                     and all(r["pass_"] for r in irec.values())))
    # 5 R battery
    rr = {k: rc.get(k) for k in ("R_dryrun", "R_vp9", "R_vp8", "R_fx", "R_fx_flip", "R_fx_nol2", "R_review_fx",
                                 "R_review_flip", "R_review_nol2")}
    rv9 = os.path.join(sc, "R", "vp9", f"vp9_known_answers_FZ_{tag}.json")
    rv8 = os.path.join(sc, "R", "vp8", f"vp8_r_FZ_{tag}.json")
    rrec = {"vp9": rec(rv9, pass_=jl(rv9).get("pass") if os.path.exists(rv9) else False),
            "vp8": rec(rv8, pass_=jl(rv8).get("pass") if os.path.exists(rv8) else False)}
    B["R_battery"] = dict(steps=rr, records=rrec, pass_=bool(all(v == 0 for v in rr.values())
                                                             and all(r["pass_"] for r in rrec.values())))
    # 6 cross-implementation fixture X
    # fixture X: one report per Part B variant (xfixture_pyt.py); exit codes required to be 0 for 9 steps per variant
    # (the synthetic R3 variants run stage synth in place of E's builder, still 9); R's review exit (its own
    # comparisons), E's vl5 and I's cross_validate exits are recorded only (their comparisons include the pair-level
    # fixture artefact; xfixture_pyt.py compare re-checks every verdict file)
    XV = {}
    synth_variants = ("h1A", "h1Aprime", "h1B")   # lead ruling R3 (xfixture_pyt.py docstring)
    for V in ("base", "flipH1", "bandfail", "bandwide") + synth_variants:
        xs = {k: v for k, v in rc.items() if k.startswith("X_") and k.endswith("_" + V)}
        x_req = {k: v for k, v in xs.items() if not k.startswith(("X_R_review_", "X_E_vl5_", "X_I_cross_"))}
        p = os.path.join(fz, f"xfixture_report_{V}.json")
        d = jl(p) if os.path.exists(p) else {}
        XV[V] = rec(p if d else None, pass_=bool(d.get("pass") and len(x_req) == 9 and all(v == 0 for v in x_req.values())),
                    strict_pass=d.get("strict_pass"), steps=xs, steps_required_rc0=sorted(x_req),
                    comparisons={k: {f: c.get(f) for f in ("sections", "n_compared", "n_mismatch", "n_artefact",
                                                            "n_artefact_class2_post_hoc", "n_other_mismatch",
                                                            "other_mismatching_keys", "n_missing_in_a",
                                                            "n_missing_in_b", "pass_")}
                                 for k, c in (d.get("comparisons") or {}).items()},
                    R_own_fixtures=d.get("R_own_fixtures"), coverage=d.get("coverage"),
                    intended=d.get("intended"), coverage_as_intended=d.get("coverage_as_intended"),
                    coverage_misses=d.get("coverage_misses"))
    # branches reached over all variants (values of the test inputs; reported, not part of pass)
    reach = {}
    for V, xv in XV.items():
        for slot, cv in (xv.get("coverage") or {}).items():
            for w, kv in (cv.get("K") or {}).items():
                for f, val in kv.items():
                    if val is not None:
                        reach.setdefault(f"{f}={val}", set()).add(f"{V}:{slot}:{w}")
    B["fixture_X"] = dict(variants=XV, pass_=all(v["pass_"] for v in XV.values()),
                          strict_pass=all(bool(v.get("strict_pass")) for v in XV.values()),
                          coverage_as_intended=all(XV[V].get("coverage_as_intended") is True for V in synth_variants),
                          branches_reached={k: sorted(v) for k, v in sorted(reach.items())
                                            if k.split("=")[0] in ("category", "premise", "M2", "M5_A", "M5_B")})
    fm = failed_markers()
    this_run = [m for m in fm if f"run_{tag}" in m["marker"] or f"FZ_{tag}" in m["marker"]
                or os.path.getmtime(os.path.join(PROJ, m["marker"])) >= t0]
    doc = dict(what="Pythia axis S0b consolidated test record (role pyt-code-freeze); CPU only, synthetic and frozen "
                    "non-Pythia data only; not citable", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), tag=tag,
               plan_sha256=PLAN_SHA, battery_script_sha256={f: sha(os.path.join(AXIS, "tests_freeze", f))
                                                            for f in sorted(os.listdir(os.path.join(AXIS, "tests_freeze")))
                                                            if not f.startswith(".") and not f.endswith(".pyc")
                                                            and os.path.isfile(os.path.join(AXIS, "tests_freeze", f))},
               steps=steps, batteries=B,
               failed_runs_before_freeze=dict(
                   markers=fm, n_markers=len(fm), n_markers_expected=sum(1 for m in fm if m["expected"]),
                   n_markers_unexpected=sum(1 for m in fm if not m["expected"]),
                   markers_in_this_battery=[m["marker"] for m in this_run], no_marker=NO_MARKER),
               not_run_here=["G0 GPU smoke and gate G0 (GPU order of the lead; G0 runs after CODE FROZEN and before S3 GO)",
                             "F1(a) dry runs, F1(d) and feasibility_pyt.py (run by s0b_freeze.py and its caller)"])
    doc["pass"] = all(b["pass_"] for b in B.values())
    path = os.path.join(fz, "S0B_TEST_RECORD_PYT.json")
    if os.path.lexists(path):
        sys.exit(f"{path} exists (no-delete rule)")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    print(json.dumps({k: v["pass_"] for k, v in B.items()}))
    print(f"record {path} sha256 {sha(path)} pass {doc['pass']}; failed-run markers {len(fm)} "
          f"({doc['failed_runs_before_freeze']['n_markers_expected']} expected by design), in this battery "
          f"{len(this_run)}")
    sys.exit(0 if doc["pass"] else 1)


if __name__ == "__main__":
    main()
