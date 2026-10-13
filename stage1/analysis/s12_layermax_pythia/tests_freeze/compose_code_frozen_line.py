#!/usr/bin/env python
"""Pythia axis, step S0b: append the ONE 'CODE FROZEN' STATUS line (role pyt-code-freeze, 2026-10-10). NEW FILE (PYT-AXIS).

s0b_freeze.py (infrastructure role) writes S0B_CODE_MANIFEST.json, chmods the code and builds its STATUS line from the
manifest. The S8-review line of 2026-10-10T00:57:58Z asked that the CODE FROZEN line also list every failed pre-freeze
test run, which the line of s0b_freeze.py does not do. So s0b_freeze.py runs with --no-status and this script, run once
right after it, appends the one milestone line:
  1. refuses (exit 5, nothing written) if STATUS.md already has a '<UTC>Z [pyt-axis-<role>] CODE FROZEN' milestone line
     (the regex of guard.py), if the manifest is missing, writable, not passing, or its sha256 differs from --manifest-sha;
  2. refuses unless the consolidated test record (--record) has "pass": true and is one of the manifest's pass records
     with the same sha256;
  3. rebuilds the line text of s0b_freeze.py main() from the manifest fields (same wording and order), and inserts before
     its file list: the integration note, the S0b battery result, the fixture X counts, the GPU and G0 status, and the full
     list of failed pre-freeze test runs from the record;
  4. appends it as ONE line under flock, re-reads STATUS.md and checks that exactly one CODE FROZEN milestone line exists.
Reads the manifest and the record; writes only the STATUS line (--dry: prints it, writes nothing; --manifest/--status
are for the dry test of this script on a scratch copy).
usage: venv/bin/python analysis/s12_layermax_pythia/tests_freeze/compose_code_frozen_line.py --manifest-sha <64 hex>
           --record <path of S0B_TEST_RECORD_PYT.json> [--role pyt-axis-code-freeze] [--dry]
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import time

PROJ = os.path.realpath(os.path.expanduser("~/hrh_v2_stage1_20261008"))
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_pythia")
MANIFEST = os.path.join(AXIS, "S0B_CODE_MANIFEST.json")
STATUS = os.path.join(PROJ, "STATUS.md")
CODE_FROZEN_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \[pyt-axis-[A-Za-z0-9-]+\] CODE FROZEN")  # = guard.py


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def fail(msg: str):
    print(f"[compose_code_frozen_line] REFUSED (exit 5): {msg}", file=sys.stderr)
    sys.exit(5)


def frozen_lines(path: str) -> list[str]:
    return [ln for ln in open(path, encoding="utf-8").read().splitlines() if CODE_FROZEN_RE.match(ln)]


def freezer_text(man: dict, msha: str, role: str, utc: str) -> tuple[str, str]:
    """The line of s0b_freeze.py main(), split before 'FILES:' (head, files part)."""
    inv, orig, dr = man["files"], man["originals"], man["dry_runs"]
    n_orig = len(orig["originals"])
    files_txt = "; ".join(f"{r['path']} {r['sha256']}" for r in inv)
    dr_txt = ", ".join(f"{r['label']}={r['exit']}" for r in dr)
    rec_txt = "; ".join(f"{r['path']} {r['sha256']} pass={r['pass_']}" for r in man["records"]) or "none"
    tl_txt = "; ".join(f"{r['path']} {r['sha256']}" for r in man["test_logs"]) or "none"
    g0 = man["g0"]
    g0_txt = (f"G0 {g0['decision']} ({g0['path']} {g0['sha256']})" if g0.get("sha256")
              else "G0 NOT YET RUN (frozen before G0 under the order plan -> code -> G0; G0 precedes S3 GO)")
    feas = man["feasibility"]
    head = (f"{utc} [{role}] CODE FROZEN (S0b, plan 6de89603 section 11) BEFORE S1 and before any "
            f"Pythia byte exists on ds2213: {len(inv)} files chmod 444, manifest "
            f"analysis/s12_layermax_pythia/S0B_CODE_MANIFEST.json sha256 {msha} (freezer "
            f"{man['freezer']['path']} {man['freezer']['sha256']}); originals of plan section 3: "
            f"{n_orig - orig['n_bad']}/{n_orig} equal with pristine copies; F1(a) dry runs (PYT_S0B_TEST unset) "
            f"all exit 3 and write nothing: {dr_txt}; F1(d) f1d_outroots.py PASS (json sha256 "
            f"{man['f1d_json_sha256']}); feasibility_pyt.py outputs {', '.join(f'{p} {h}' for p, h in feas)}; "
            f"{g0_txt}; pass records: {rec_txt}; test logs (incl. failed runs): {tl_txt}; AppleDouble files excluded: "
            f"{len(man['appledouble_excluded'])}; results tree top level {man['scan']['top']}, negative-scan hits "
            f"{len(man['scan']['hits'])}; ")
    return head, f"FILES: {files_txt}"


def extra_text(rec: dict, rpath: str, rsha: str, n_dry: int) -> str:
    B = rec["batteries"]
    X = B["fixture_X"]
    ctxt = " || ".join(
        f"variant {V} ({XVr['path']} {XVr['sha256']}, pass={XVr['pass_']}): " + "; ".join(
            f"{k} [{c['sections']}] {c['n_compared']} compared, {c['n_mismatch']} mismatches ({c['n_artefact']} fixture "
            f"artefact, of which {c.get('n_artefact_class2_post_hoc') or 0} class 2 post hoc; {c['n_other_mismatch']} "
            f"other), missing {c['n_missing_in_a']}/{c['n_missing_in_b']}" for k, c in (XVr.get("comparisons") or {}).items())
        for V, XVr in X["variants"].items())
    E = B["E_battery"]
    erec = ", ".join(f"{k} {v['path']} {v['sha256']} pass={v['pass_']}" for k, v in (E.get("records") or {}).items())
    I = B["I_battery"]
    irec = ", ".join(f"{k} {v['path']} {v['sha256']} pass={v['pass_']}" for k, v in I["records"].items())
    R = B["R_battery"]
    rrec = ", ".join(f"{k} {v['path']} {v['sha256']} pass={v['pass_']}" for k, v in R["records"].items())
    fr = rec["failed_runs_before_freeze"]
    mk = "; ".join(f"{m['marker']} [{m['end']}]" + (f" EXPECTED BY DESIGN ({m['expected'].split(':')[0]})"
                                                     if m["expected"] else f" (log: {m['log_last_line'][:140]})")
                   for m in fr["markers"])
    nm = "; ".join(f"{d['role']}: {d['run']}: {d['what']} [{d['source']}]" for d in fr["no_marker"])
    this = ", ".join(fr["markers_in_this_battery"]) or "none"
    return (f"INTEGRATION by role pyt-code-freeze (no E, I or R logic changed or merged; E early/, I indep/, R review/, the "
            f"infrastructure code, the extraction side and the freeze role's tests_freeze/ are all in this manifest; merged "
            f"dry-run list analysis/s12_layermax_pythia/s0b_dryruns_all.json, {n_dry} gated commands of every role); "
            f"S0b BATTERY tag {rec['tag']} (CPU only, CUDA hidden, <= 4 thread tokens, synthetic and frozen non-Pythia data, "
            f"outputs only in s0b_scratch/ and extract/pythia_axis/s0b_test_out/; not citable): consolidated record "
            f"{rpath} sha256 {rsha} pass={rec['pass']}; extraction CPU {B['extraction_cpu'].get('n_pass')} pass "
            f"{B['extraction_cpu'].get('n_fail')} fail; infrastructure {B['infra_tests'].get('n_pass')}/"
            f"{B['infra_tests'].get('n')} pass, partb_checks_pyt on the shared stand-in Part B pass="
            f"{B['partb_checks_vp9']['pass_']}; E battery {E.get('n_pass')}/{E.get('n_runs')} pass ({erec}); I battery "
            f"{I['n_steps']} steps all rc 0={I['pass_']} ({irec}); R battery dry runs, V-P9, V-P8, three R fixtures "
            f"pass={R['pass_']} ({rrec}); CROSS-IMPLEMENTATION FIXTURE X (plan V-P9 last item: E, I, forecast_pyt.py E "
            f"and I, and R on ONE 7-size stand-in, 5 BLOOMZ + llama-13b + llama-30b, shared stand-in Part B "
            f"s0b_scratch/PB/partB_vp9, variant base, three Part B test variants flipH1 = category E path, bandfail = "
            f"category B with M5_B, bandwide = premise path, and the three H1-POSITIVE SYNTHETIC variants of lead ruling "
            f"R3, added after run 20261010T024645Z: the stand-in's biased CKA transformed per pair by an increasing affine "
            f"map so that c_alpha = Bbar + f(PR*) and the calibrated ranks drop with size, h1A = prh_clip+l2 A with M5_A, "
            f"l2 A-att; h1Aprime = prh_clip+l2 A-prime, l2 A with premise unverified; h1B = B with M5_B true and false; "
            f"I run without --fixture on synthetic I-format files) pass={X['pass_']} strict={X.get('strict_pass')}, "
            f"synthetic coverage as intended={X.get('coverage_as_intended')}, branches reached (section K, any writer): "
            f"{', '.join(sorted(X.get('branches_reached') or {}))}: {ctxt}; "
            f"fixture artefact class 1 "
            f"= keys of the subsets noLMemb, LMdeep, LMemb_only, which the frozen BLOOMZ E-format pair files do not store "
            f"(rule fixed in tests_freeze/xfixture_pyt.py before its first run); class 2 (POST HOC, added after run "
            f"20261010T022655Z) = G|clip_equal_partB where I is null because I's fixture mode does not evaluate V-P1 "
            f"by design (also in the synthetic variants: no I single-model prep); LEAD RULINGS 2026-10-10 ~03:05Z under "
            f"the author's delegation: (R1) a key that the contract defines only for some variants is null in every other "
            f"variant in every implementation, so under x=unc all three M4 keys (M4, M4_n_zero, M4_sizes) are null; (R2) "
            f"G|clip_equal_partB null in I's fixture mode is accepted as a pre-registered fixture exclusion of type 2, "
            f"disclosed as POST HOC (added after fixture round 1), and V-P1 must be evaluated in the real run; (R3) a "
            f"synthetic H1-positive fixture exercises A, A-att, A', B, M2, M5_A, M5_B and the premise, R is compared with "
            f"forecast (E and I sides) on every variant, disagreements R1 resolves are fixed by the owning role and any "
            f"other is reported, not improvised; GPU: none used by the freeze role, "
            f"ledger unchanged; the GPU paths of E and I ran on CPU only at S0b (kernels unchanged from the OpenLLaMA "
            f"copies), so the G0 smoke is their first GPU run; FAILED PRE-FREEZE TEST RUNS (review request, all before "
            f"this line, none on Pythia data): {fr['n_markers']} launcher .failed markers, "
            f"{fr['n_markers_expected']} of them the designed outcome of the test: {mk}; failures without a marker: "
            f"{nm}; markers created by the final battery {rec['tag']}: {this}; ")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-sha", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--role", default="pyt-axis-code-freeze")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--manifest", default=MANIFEST, help="dry test only")
    ap.add_argument("--status", default=STATUS, help="dry test only")
    args = ap.parse_args()
    if not re.fullmatch(r"pyt-axis-[a-z0-9-]+", args.role):
        fail(f"bad --role {args.role}")
    if not args.dry and (args.manifest != MANIFEST or args.status != STATUS):
        fail("--manifest/--status are for --dry tests only")
    if frozen_lines(args.status):
        fail("a CODE FROZEN milestone line already exists in STATUS.md")
    if not os.path.isfile(args.manifest):
        fail(f"{args.manifest} missing (run s0b_freeze.py --no-status first)")
    msha = sha(args.manifest)
    if msha != args.manifest_sha:
        fail(f"manifest sha256 {msha} != --manifest-sha")
    if not args.dry and os.stat(args.manifest).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        fail("manifest is writable (s0b_freeze.py makes it 444)")
    man = json.load(open(args.manifest))
    if man.get("pass_") is not True or not all(v for k, v in man["checks"].items() if k not in ("scan_top", "scan_hits")):
        fail("manifest does not record a passing freeze")
    rpath = os.path.relpath(os.path.abspath(args.record), PROJ)
    rsha = sha(args.record)
    rec = json.load(open(args.record))
    if rec.get("pass") is not True:
        fail("the consolidated test record does not pass")
    if not any(r["path"] == rpath and r["sha256"] == rsha and r["pass_"] for r in man["records"]):
        fail("the consolidated test record is not a pass record of the manifest (same path and sha256)")
    dr_file = os.path.join(AXIS, "s0b_dryruns_all.json")
    n_dry = len(json.load(open(dr_file)))
    if n_dry != len(man["dry_runs"]):
        fail("dry-run count differs from the manifest")
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    head, files = freezer_text(man, msha, f"{args.role}", utc)
    line = (head + extra_text(rec, rpath, rsha, n_dry) + files).replace("\n", " ")
    assert CODE_FROZEN_RE.match(line)
    if args.dry:
        print(line)
        print(f"[compose_code_frozen_line] DRY: {len(line)} characters, nothing written", file=sys.stderr)
        return
    with open(args.status, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    n = frozen_lines(args.status)
    print(line[:1500])
    print(f"[compose_code_frozen_line] appended ONE line ({len(line)} characters); CODE FROZEN milestone lines now: "
          f"{len(n)}")
    sys.exit(0 if len(n) == 1 else 5)


if __name__ == "__main__":
    main()
