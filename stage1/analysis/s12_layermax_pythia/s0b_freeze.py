"""S0b code freeze of the Pythia axis (ANALYSIS_PLAN_PYTHIA.md sections 3, 4 F1(a)/F1(d), 11 step S0b).  # PYT-AXIS

PYT-AXIS copy of analysis/s12_layermax_openllama/s0b_freeze.py (sha256 490d714e...), written at S0b by role  # PYT-AXIS
pyt-code-infra (2026-10-10); changed lines carry PYT-AXIS; diff: s0b_freeze.py.diff. It is the last S0b step:  # PYT-AXIS
  1. inventory: every file of analysis/s12_layermax_pythia/ (except the frozen plan, run/, __pycache__, s0b_scratch/)  # PYT-AXIS
     and of extract/pythia_axis/ (except hf_home/, s0b_test_out/, __pycache__), with sha256 and size; AppleDouble  # PYT-AXIS
     files named ._* are excluded by pattern and listed separately (plan section 4);  # PYT-AXIS
  2. originals: the 40 files of the plan section 3 table are re-hashed and must equal the plan's values; a pristine  # PYT-AXIS
     copy of each must exist in some orig_copy/ directory of the axis (matched by sha256);  # PYT-AXIS
  3. F1(a) dry runs: every gated command of --dryruns (JSON list of [label, argv...]) is run with PYT_S0B_TEST unset  # PYT-AXIS
     and must exit with code 3 AND write nothing: the file listings (path, size, mtime) of results/S1_2_layermax_pythia/,  # PYT-AXIS
     features/wit1024_pythia_axis/ and analysis/s12_layermax_pythia/s0b_scratch/ are equal before and after;  # PYT-AXIS
  4. F1(d): f1d_outroots.py must pass;
  5. feasibility: results/S1_2_layermax_pythia/feasibility/feasibility_pyt.json must exist, and its meta.script_sha256  # PYT-AXIS
     must equal the current feasibility_pyt.py (it ran for real with the code being frozen);  # PYT-AXIS
  6. F1(b)-style check: guard.scan() must be clean (results tree holds only allowed entries);
  7. V-P8/V-P9 and test records: every --require-pass FILE must be JSON with top-level "pass": true (hashed);  # PYT-AXIS
     --test-log FILE (JSON or text, e.g. the list of every S0b test run including failed ones) is hashed only;  # PYT-AXIS
  8. gate G0: results/S1_2_layermax_pythia/logs/gpu_gate_G0.json must exist (its decision is recorded), unless  # PYT-AXIS
     --g0-later is given (then the line says that G0 is still to run, as the lead's order plan -> code -> G0 allows);  # PYT-AXIS
  9. writes S0B_CODE_MANIFEST.json (chmod 444), chmods every inventoried code file 444 (555 if executable), and  # PYT-AXIS
     appends ONE STATUS line "<UTC>Z [pyt-axis-<role>] CODE FROZEN (S0b) ..." with every hash (unless --no-status).  # PYT-AXIS
usage: venv/bin/python analysis/s12_layermax_pythia/s0b_freeze.py --dryruns dryruns.json [--require-pass F ...]  # PYT-AXIS
       [--test-log F ...] [--g0-later] [--role pyt-axis-<role>] [--check-only] [--no-status]  # PYT-AXIS
  --check-only runs 1-8 and prints, writes nothing and appends nothing.  # PYT-AXIS
Exit 0 on success; 5 if any check fails (then nothing is frozen and no STATUS line is written).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re  # PYT-AXIS
import stat
import subprocess
import sys
import time

PROJ = os.path.realpath(os.path.expanduser("~/hrh_v2_stage1_20261008"))
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_pythia")  # PYT-AXIS
EXTR = os.path.join(PROJ, "extract", "pythia_axis")  # PYT-AXIS
RES = os.path.join(PROJ, "results", "S1_2_layermax_pythia")  # PYT-AXIS
FEATS = os.path.join(PROJ, "features", "wit1024_pythia_axis")  # PYT-AXIS
STATUS = os.path.join(PROJ, "STATUS.md")
PLAN = os.path.join(AXIS, "ANALYSIS_PLAN_PYTHIA.md")  # PYT-AXIS
PLAN_SHA = "6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b"  # PYT-AXIS
MANIFEST = os.path.join(AXIS, "S0B_CODE_MANIFEST.json")
PY = os.path.join(PROJ, "venv", "bin", "python")
SKIP = {"run", "__pycache__", "s0b_scratch", "s0b_test_out", "hf_home"}  # PYT-AXIS
G0_FILE = os.path.join(RES, "logs", "gpu_gate_G0.json")  # PYT-AXIS: written by gpu_gate_pyt.py g0
# plan section 3, "Originals to copy" (paths without a prefix are under analysis/s12_layermax_openllama/)  # PYT-AXIS
ORIGINALS = {  # PYT-AXIS
    "early/aggregate_ol.py": "f7043a9ed7ee16a8afde8531503bb24ba8ef7d74900cf0554e49e73514dedba5",  # PYT-AXIS
    "early/lm_core.py": "c1572fd9cbd7f8332e35ff5620bf21a79909e552fe462b17ea8d076f910a28b5",  # PYT-AXIS
    "early/precompute.py": "bd0f5e2accfc535a9e4fd926294c74cd2f9428165ae8b4322b030847d2124fb9",  # PYT-AXIS
    "early/report_ol.py": "67580f2e5121ba5225557b1fa1c37a3aa4ef9b11da1606652874eb770669e606",  # PYT-AXIS
    "early/run_layermax.py": "4f690fa2032b17ecc14300cb1a26815dc059464ab7359cab4dd3f805e47ebac5",  # PYT-AXIS
    "early/validate.py": "958d1c940d6a0a082dc20db1e13e796b5a90950f7fc54e2a3cb14904c7ffb880",  # PYT-AXIS
    "early/vl5_e_vs_i.py": "08e6955ec08a90d7c7792477e5c0d058bb2fc837bc0a6cb6e8f66c92091740b1",  # PYT-AXIS
    "early/vl7_prop5.py": "210d97f52cac589feb755c02d2e1a14fe1d7c035bc3f787f192396a22b6f849c",  # PYT-AXIS
    "early/write_hashes.py": "a49a206ea14147349523c2d11746556f7f22b74fbeeae8dbfe5b73a6f1d12b08",  # PYT-AXIS
    "indep/common_ol.py": "2b253cd95f42e3ad34ae4ee5c20523d81232d24e52ebb9c2b729f798dc98ac26",  # PYT-AXIS
    "indep/cross_validate.py": "ab3a4b19a8904452407089034304abbd3faed3dd0873fa6bacdbf9226dd25636",  # PYT-AXIS
    "indep/lmx_aggregate.py": "60170b7290b0bb6fdcad340cbf2c022d7fc522a53e85706668267629ce113de1",  # PYT-AXIS
    "indep/lmx_common.py": "d0cc9b81ec8b1a0807a583116f77e0ef7d6d0db47f620b584f9378c94b647d53",  # PYT-AXIS
    "indep/lmx_ol_aggregate.py": "90796c045f7130256ad887ae66c257dcb33f2efc4a5acada7cb85f75315bfb82",  # PYT-AXIS
    "indep/lmx_ol_report.py": "b60a5ea4a852eb8803ba762f7938edd256ead52f981fde5c8bb114cfa3be13e1",  # PYT-AXIS
    "indep/ol_gate.py": "2a55dce403d8eb03ab23a4f9dab7dd61a72003d2b4bcad4991fbc95b5794ee9c",  # PYT-AXIS
    "indep/ol_verdicts.py": "21515722ac15b6551ee7641bbf11f5ace92d1c976ec9e1dbacc40419026d0e01",  # PYT-AXIS
    "indep/run_layermax.py": "dee3a805cd650399250246e8b6625ffe272385c9d9d3b8bd2434c17edecd5855",  # PYT-AXIS
    "indep/validate.py": "08f5d0414fd814df503ccf0f019e9010e080714c35c25c54f0f90c27225c49b4",  # PYT-AXIS
    "indep/vl6_arh.py": "e75f5fb3e91c1721b13832df476d801c66fec787cb7606c483d8fb0de1e958f0",  # PYT-AXIS
    "indep/vl_checks.py": "ee34896ec5f77db374c3b27f457b1ea7665c258aafa7cb04d58b1fe6de284b7a",  # PYT-AXIS
    "review/ol_axis_review.py": "00c98f2a6fa95078c416605f27596d7b65473729c8814082ac7507803b59de07",  # PYT-AXIS
    "review/ol_review_lib.py": "b224a24ed5249bf6b2796c9cadc9537f38e9363381d822f413711b34fc2942aa",  # PYT-AXIS
    "review/ol_review_sections.py": "e2e41713d601aee4d0b6a540dc4e027b4dd80283dda18045237010b2369a7c4b",  # PYT-AXIS
    "review/make_test_fixture.py": "e565004a3b14e13b30504694dca23473407cf7957215f2b781e479ddcc145400",  # PYT-AXIS
    "partb_ol.py": "4aab8ee1942cb0e94307306b0ccd9d450102d6bcd41236370f614b3192b8aef6",  # PYT-AXIS
    "partb_checks_ol.py": "48d91f9eac0915ce8a5712af438435f100f85f9c767be503514da5a2ae71774f",  # PYT-AXIS
    "guard.py": "de4c8ef942f04d8fc50e6775cf7517ce7f9407d7686413b9fe066aedb028cd13",  # PYT-AXIS
    "launch_ol.sh": "3675d6f20658e3526e834367c9b375639ca72c1cabd1a86db1f8ed611fddc769",  # PYT-AXIS
    "feasibility_ol.py": "3faad6b1f1fc57b8e36a37590ef5d86fcaed7a99c32ffedf6cb91216feeef4aa",  # PYT-AXIS
    "forecast_ol.py": "aff9a505562631ecddad92b3e8b1cc40f02303cd1f0921c3f0887307b0b3d886",  # PYT-AXIS
    "f1d_outroots.py": "0589ab49e05463b5f404055350ab14f33879bcd843f435587d59e490bdae5799",  # PYT-AXIS
    "s0b_freeze.py": "490d714e3f7099023606334353168f8599bd000747e07089cb3d6a84ef7d4004",  # PYT-AXIS
    "verdict_keys.py": "95a152c8fcecc51ac03b79a0905d7aaa2baf754e2a12a40295d5d36749b22f8f",  # PYT-AXIS
    "compare_verdicts.py": "233351a41039ca2d136e56b8609f9ab077214c5e7900863b1d309bab449bf45f",  # PYT-AXIS
    "../../extract/openllama_axis/download_ol.py": "93c11d6194bb27d8ed26157d8e6cd32a9da6063b8392a17bf0a34fecaf35f73c",  # PYT-AXIS
    "../../extract/openllama_axis/extract_core_ol.py": "425f02ab0537db51bd51824dccd85e53776a5a11f5cd1981b7235c206439e21f",  # PYT-AXIS
    "../../extract/openllama_axis/ol_axis_common.py": "58fb4c4e36bda52c8d58d49c62b78cf168bb1d2d19626dd781c8fc03e61513cb",  # PYT-AXIS
    "../../extract/openllama_axis/registry_openllama_axis.py": "885caec2f8182a4e64a5c41b1f1060fdd52dfb3a2a971962af4a3494b5141680",  # PYT-AXIS
    "../../extract/openllama_axis/run_ol.py": "4231c834a81069f3f0469d8d584c555447a79e92dc1d47b698499be0c610dcf0",  # PYT-AXIS
}
NO_OUTPUT_ROOTS = (RES, FEATS, os.path.join(AXIS, "s0b_scratch"))  # PYT-AXIS: "exit 3, no output" (dry runs)


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def inventory() -> tuple[list[dict], list[dict]]:  # PYT-AXIS: (+ AppleDouble files listed separately)
    rows, apple = [], []  # PYT-AXIS
    for base in (AXIS, EXTR):
        for d, subdirs, fs in os.walk(base):
            subdirs[:] = sorted(s for s in subdirs if s not in SKIP)
            for f in sorted(fs):
                p = os.path.join(d, f)
                if p == PLAN or p == MANIFEST or f.endswith(".pyc"):
                    continue
                if f.startswith("._"):  # PYT-AXIS: plan section 4, excluded by pattern, listed separately
                    apple.append(dict(path=os.path.relpath(p, PROJ), bytes=os.path.getsize(p)))  # PYT-AXIS
                    continue  # PYT-AXIS
                rows.append(dict(path=os.path.relpath(p, PROJ), sha256=sha(p), bytes=os.path.getsize(p)))
    return rows, apple  # PYT-AXIS


def check_originals(inv: list[dict]) -> dict:
    base = os.path.join(PROJ, "analysis", "s12_layermax_openllama")  # PYT-AXIS
    out, bad = {}, []
    by_sha = {}
    for r in inv:
        if "orig_copy/" in r["path"]:
            by_sha.setdefault(r["sha256"], []).append(r["path"])
    for rel, want in ORIGINALS.items():
        p = os.path.normpath(os.path.join(base, rel))
        got = sha(p) if os.path.exists(p) else None
        out[os.path.relpath(p, PROJ)] = dict(sha256=got, plan=want, equal=got == want,
                                             pristine_copies=by_sha.get(want, []))
        if got != want or not by_sha.get(want):  # PYT-AXIS: a pristine copy must exist too
            bad.append(rel)
    return dict(originals=out, n_bad=len(bad), bad=bad)


def listing(roots=NO_OUTPUT_ROOTS) -> dict:  # PYT-AXIS: (path -> (size, mtime_ns)) of the output roots
    out = {}  # PYT-AXIS
    for root in roots:  # PYT-AXIS
        for d, _, fs in os.walk(root):  # PYT-AXIS
            for f in fs:  # PYT-AXIS
                p = os.path.join(d, f)  # PYT-AXIS
                try:  # PYT-AXIS
                    st = os.lstat(p)  # PYT-AXIS
                    out[os.path.relpath(p, PROJ)] = (st.st_size, st.st_mtime_ns)  # PYT-AXIS
                except FileNotFoundError:  # PYT-AXIS
                    pass  # PYT-AXIS
    return out  # PYT-AXIS


def dry_runs(spec: list) -> list[dict]:
    env = {k: v for k, v in os.environ.items() if k != "PYT_S0B_TEST"}  # PYT-AXIS
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYT_THREADS="1")  # PYT-AXIS
    res = []
    for label, *argv in spec:
        before = listing()  # PYT-AXIS
        t0 = time.time()
        p = subprocess.run([PY] + argv, cwd=PROJ, env=env, capture_output=True, text=True, timeout=600)
        after = listing()  # PYT-AXIS
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))  # PYT-AXIS
        res.append(dict(label=label, argv=argv, exit=p.returncode, s=round(time.time() - t0, 1),
                        n_files_changed=len(changed), files_changed=changed[:20],  # PYT-AXIS
                        stdout_bytes=len(p.stdout),  # PYT-AXIS
                        stderr_tail=p.stderr.strip().splitlines()[-2:] if p.stderr else []))
    return res


def pass_records(paths: list[str]) -> list[dict]:  # PYT-AXIS: V-P8 / V-P9 / test records
    out = []  # PYT-AXIS
    for p in paths:  # PYT-AXIS
        ap = os.path.abspath(p)  # PYT-AXIS
        try:  # PYT-AXIS
            ok = json.load(open(ap)).get("pass") is True  # PYT-AXIS
        except Exception as e:  # noqa: BLE001  # PYT-AXIS
            ok = False  # PYT-AXIS
            print(f"[s0b_freeze] cannot read {ap}: {e}", file=sys.stderr)  # PYT-AXIS
        out.append(dict(path=os.path.relpath(ap, PROJ), sha256=sha(ap) if os.path.exists(ap) else None, pass_=ok))  # PYT-AXIS
    return out  # PYT-AXIS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dryruns", required=True, help="JSON file: list of [label, script, args...] (gated commands)")
    ap.add_argument("--require-pass", nargs="*", default=[], help="JSON records with top-level pass: true")  # PYT-AXIS
    ap.add_argument("--test-log", nargs="*", default=[], help="test-run records (hashed only)")  # PYT-AXIS
    ap.add_argument("--g0-later", action="store_true", help="freeze before G0 (recorded in the line)")  # PYT-AXIS
    ap.add_argument("--role", default=os.environ.get("PYT_ROLE", "pyt-axis-freeze"))  # PYT-AXIS
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--no-status", action="store_true")
    args = ap.parse_args()
    if not re.fullmatch(r"pyt-axis-[a-z0-9-]+", args.role):  # PYT-AXIS
        sys.exit(f"bad --role {args.role}")  # PYT-AXIS
    assert sha(PLAN) == PLAN_SHA, "frozen plan changed"
    sys.path.insert(0, AXIS)
    import guard  # noqa: E402
    if guard.code_frozen_line() is not None:
        sys.exit("a CODE FROZEN line already exists; S0b cannot be frozen twice (version instead)")
    inv, apple = inventory()  # PYT-AXIS
    orig = check_originals(inv)
    spec = json.load(open(args.dryruns))
    dr = dry_runs(spec)
    f1d = subprocess.run([PY, os.path.join(AXIS, "f1d_outroots.py")], cwd=PROJ, capture_output=True, text=True)
    scan = guard.scan()
    feas_json = os.path.join(RES, "feasibility", "feasibility_pyt.json")  # PYT-AXIS
    feas_dir = os.path.dirname(feas_json)  # PYT-AXIS
    feas = sorted((os.path.relpath(os.path.join(d, f), PROJ), sha(os.path.join(d, f)))
                  for d, _, fs in os.walk(feas_dir) for f in fs) if os.path.isdir(feas_dir) else []
    feas_code_ok = False  # PYT-AXIS
    if os.path.isfile(feas_json):  # PYT-AXIS
        feas_code_ok = json.load(open(feas_json))["meta"]["script_sha256"] == sha(os.path.join(AXIS, "feasibility_pyt.py"))  # PYT-AXIS
    recs = pass_records(args.require_pass)  # PYT-AXIS
    tlogs = [dict(path=os.path.relpath(os.path.abspath(p), PROJ), sha256=sha(p)) for p in args.test_log]  # PYT-AXIS
    g0 = json.load(open(G0_FILE)) if os.path.isfile(G0_FILE) else None  # PYT-AXIS
    checks = dict(originals_equal=orig["n_bad"] == 0, dry_runs_all_exit_3=all(r["exit"] == 3 for r in dr),
                  dry_runs_no_output=all(r["n_files_changed"] == 0 for r in dr),  # PYT-AXIS
                  dry_runs_nonempty=len(dr) > 0,  # PYT-AXIS
                  f1d_pass=f1d.returncode == 0, feasibility_present=bool(feas), feasibility_is_current_code=feas_code_ok,  # PYT-AXIS
                  scan_clean=scan["clean"], records_pass=all(r["pass_"] for r in recs),  # PYT-AXIS
                  g0_present_or_later=(g0 is not None) or args.g0_later,  # PYT-AXIS
                  scan_top=scan["results_top_level"], scan_hits=scan["find_hits"])
    ok = all(v for k, v in checks.items() if k not in ("scan_top", "scan_hits"))
    man = dict(schema="PYT-S0B-MANIFEST v1", utc=utc(), plan_sha256=PLAN_SHA, n_files=len(inv), files=inv,  # PYT-AXIS
               appledouble_excluded=apple, originals=orig, dry_runs=dr,  # PYT-AXIS
               f1d=dict(exit=f1d.returncode, stdout=f1d.stdout[-4000:]),  # PYT-AXIS
               f1d_json_sha256=(sha(os.path.join(RES, "logs", "f1d_outroots.json"))
                                if os.path.exists(os.path.join(RES, "logs", "f1d_outroots.json")) else None),
               feasibility=feas, scan=dict(clean=scan["clean"], top=scan["results_top_level"],
                                           hits=scan["find_hits"], procs=scan["forbidden_processes"]),
               records=recs, test_logs=tlogs, g0=dict(path=os.path.relpath(G0_FILE, PROJ),  # PYT-AXIS
                                                      sha256=sha(G0_FILE) if g0 is not None else None,  # PYT-AXIS
                                                      decision=(g0 or {}).get("decision"), later=args.g0_later),  # PYT-AXIS
               checks=checks, pass_=ok, freezer=dict(path=os.path.relpath(__file__, PROJ), sha256=sha(__file__)))
    print(json.dumps(dict(checks=checks, n_files=len(inv), n_appledouble=len(apple),  # PYT-AXIS
                          dry_runs=[(r["label"], r["exit"], r["n_files_changed"]) for r in dr],  # PYT-AXIS
                          bad_originals=orig["bad"]), indent=1))
    if args.check_only:
        sys.exit(0 if ok else 5)
    if not ok:
        print("S0b NOT frozen: a check failed", file=sys.stderr)
        sys.exit(5)
    if os.path.exists(MANIFEST):  # PYT-AXIS: no-delete rule
        sys.exit(f"{MANIFEST} exists; S0b cannot be frozen twice")  # PYT-AXIS
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=1)
    os.chmod(MANIFEST, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    for r in inv:  # code is read-only from now on; a change is a new versioned file and a logged deviation
        p = os.path.join(PROJ, r["path"])
        x = 0o111 if os.stat(p).st_mode & stat.S_IXUSR else 0  # keep launch_pyt.sh executable (555)  # PYT-AXIS
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | x)
    msha = sha(MANIFEST)
    files_txt = "; ".join(f"{r['path']} {r['sha256']}" for r in inv)
    dr_txt = ", ".join(f"{r['label']}={r['exit']}" for r in dr)
    rec_txt = "; ".join(f"{r['path']} {r['sha256']} pass={r['pass_']}" for r in recs) or "none"  # PYT-AXIS
    tl_txt = "; ".join(f"{r['path']} {r['sha256']}" for r in tlogs) or "none"  # PYT-AXIS
    g0_txt = (f"G0 {man['g0']['decision']} ({man['g0']['path']} {man['g0']['sha256']})" if g0 is not None  # PYT-AXIS
              else "G0 NOT YET RUN (frozen before G0 under the order plan -> code -> G0; G0 precedes S3 GO)")  # PYT-AXIS
    line = (f"{utc()} [{args.role}] CODE FROZEN (S0b, plan 6de89603 section 11) BEFORE S1 and before any "  # PYT-AXIS
            f"Pythia byte exists on ds2213: {len(inv)} files chmod 444, manifest "  # PYT-AXIS
            f"analysis/s12_layermax_pythia/S0B_CODE_MANIFEST.json sha256 {msha} (freezer "  # PYT-AXIS
            f"{man['freezer']['path']} {man['freezer']['sha256']}); originals of plan section 3: "  # PYT-AXIS
            f"{len(ORIGINALS) - orig['n_bad']}/{len(ORIGINALS)} equal with pristine copies; F1(a) dry runs (PYT_S0B_TEST unset) "  # PYT-AXIS
            f"all exit 3 and write nothing: {dr_txt}; F1(d) f1d_outroots.py PASS (json sha256 "  # PYT-AXIS
            f"{man['f1d_json_sha256']}); feasibility_pyt.py outputs {', '.join(f'{p} {h}' for p, h in feas)}; "  # PYT-AXIS
            f"{g0_txt}; pass records: {rec_txt}; test logs (incl. failed runs): {tl_txt}; AppleDouble files excluded: "  # PYT-AXIS
            f"{len(apple)}; results tree top level {scan['results_top_level']}, negative-scan hits "  # PYT-AXIS
            f"{len(scan['find_hits'])}; FILES: {files_txt}")  # PYT-AXIS
    if not args.no_status:
        with open(STATUS, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line.replace("\n", " ") + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)
    print(line[:3000])
    print(f"[s0b_freeze] manifest sha256 {msha}; STATUS line {'written' if not args.no_status else 'NOT written'}")


if __name__ == "__main__":
    main()
