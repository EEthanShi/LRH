"""S0b code freeze of the OpenLLaMA axis (ANALYSIS_PLAN_OPENLLAMA.md sections 3, 4 F1(a)/F1(d), 11 step S0b).

Written at S0b by role oll-axis-extract (2026-10-09). It is the last S0b step:
  1. inventory: every file of analysis/s12_layermax_openllama/ (except the frozen plan, p0/, run/, __pycache__) and of
     extract/openllama_axis/ (except superseded_draft_0949Z/ and __pycache__), with sha256 and size;
  2. originals: the 23 files of the plan section 3 table are re-hashed and must equal the plan's values; each pristine
     copy under */orig_copy/ or pb_orig_copy/ must equal its original;
  3. F1(a) dry runs: every gated command of --dryruns (JSON list of [label, argv...]) is run with OL_S0B_TEST unset
     and must exit with code 3 (no Part B FROZEN line exists);
  4. F1(d): f1d_outroots.py must pass;
  5. feasibility: results/S1_2_layermax_openllama/feasibility/ must exist (feasibility_ol.py ran for real); hashed;
  6. F1(b)-style check: guard.scan() must be clean (results tree holds only allowed entries);
  7. writes S0B_CODE_MANIFEST.json (chmod 444), chmods every inventoried code file 444, and appends ONE STATUS line
     "<UTC>Z [oll-axis-extract] CODE FROZEN (S0b) ..." with every hash (unless --no-status).
usage: venv/bin/python analysis/s12_layermax_openllama/s0b_freeze.py --dryruns dryruns.json [--check-only]
  --check-only runs 1-6 and prints, writes nothing and appends nothing.
Exit 0 on success; 5 if any check fails (then nothing is frozen and no STATUS line is written).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import time

PROJ = os.path.realpath(os.path.expanduser("~/hrh_v2_stage1_20261008"))
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_openllama")
EXTR = os.path.join(PROJ, "extract", "openllama_axis")
RES = os.path.join(PROJ, "results", "S1_2_layermax_openllama")
STATUS = os.path.join(PROJ, "STATUS.md")
PLAN = os.path.join(AXIS, "ANALYSIS_PLAN_OPENLLAMA.md")
PLAN_SHA = "4f18d710beda45492b1e191f74580bfc4ef53f6dafe791ad245ff1cf335f3240"
MANIFEST = os.path.join(AXIS, "S0B_CODE_MANIFEST.json")
PY = os.path.join(PROJ, "venv", "bin", "python")
SKIP = {"p0", "run", "__pycache__", "superseded_draft_0949Z"}
# plan section 3, "Originals to copy" (paths without a prefix are under analysis/s12_layermax_llama/)
ORIGINALS = {
    "early/run_layermax.py": "5d03448510c75c2a6f77ca3f63cca2203cdbcd9f0ff987958fc94712825df7be",
    "early/lm_core.py": "26306eac6a5caed19c911e997ea2b036b76215a867d1d3fd61d09c83c27edf08",
    "early/precompute.py": "7106c4b5a98d69a9519419fdab06cf9a8d1ac6f5e60b1d6af7d683050ef66ed9",
    "early/validate.py": "0ad3fb865b4a54be6bfba64995547b470305dbeac65da06346ac6965fc002318",
    "early/aggregate_llama.py": "d4de015749eb6981635c0851ee2990b4e9910a31334b9e6647621b8a17b0bb95",
    "early/report_llama.py": "00c995bd20f2afaeb6accecf265710a68526ee45ecb1b70472aaea9e8aeefc9d",
    "early/vl7_prop5.py": "1cca911d4a72ee4a343b9758299447167760ecd4dcd2553066dd1233570b2a18",
    "early/vl5_e_vs_i.py": "e31773fdb766346580f897ee1da5a1bae16e5b29b3efe591de88914b016c9438",
    "indep/run_layermax.py": "c97444aeb5474567660d6ff84a2a772074609a017de05a95fcb00335e5f13bea",
    "indep/lmx_common.py": "e58b2dd58616342fd63a058f56dabcd7704e934b5b56f2b7d3e228652de78f13",
    "indep/common_llama.py": "ab40a7a143b2c715e140e1d40cbb4de4e68049788933722ec4ee759a3abe1144",
    "indep/lmx_llama_aggregate.py": "bd82d6451f8c7c7688b19da99693db9955ea00e5abd78224ec24e5fecdfe2572",
    "indep/validate.py": "c269a320f68abd9dac89419cb72568a6f4971287823d59f584da3108b457bf3f",
    "indep/vl6_arh.py": "a156c7a0b2425539bc07ce270f1042f29c165e12428bea128acbdbc2901d9496",
    "indep/cross_validate.py": "966ca52e854a561389f32817de13ef81645d2e9026b3312bc2711aa45b723bce",
    "partb_pr.py": "6103a8be8bddcda65b12c7ddca812dc601189550fb5b3c50009899ed8dd2b9ba",
    "partb_checks.py": "b2e55a169532e82330378af30b16bf4931e321837405d57064653346c09397dd",
    "review/llama_axis_review.py": "a96576195e69d2cb42ab28db3da509da348e5b37f9832f1f5843374c307dc4ef",
    "../../extract/llama_axis/extract_core_llama.py": "5b26c082314f7ab6b7cbdebf39b332535eaf6f15f9bbc7df21edb323e2f92d50",
    "../../extract/llama_axis/download_llama_axis.py": "7e5e2b81fda71726185a6625d937913101386c75356b4534ff26f6a9df0cbc83",
    "../../extract/llama_axis/run_llama_axis.py": "12bcaa29d2091b7466920a4b5f03ec3eacc6b5bc6d65f3ca8ac8ba97d27622f9",
    "../../extract/registry_llama_axis.py": "1af88e931897a07a1a64c909733b64ffe15d0e84dfd2efe84fbc5b0a9daa7f95",
    "../../extract/extract_core.py": "12f11531923e0dd48f2a1b9c8ad27480f6e157a3bd6a2ff4fc2db6e4721cb4d0",
}


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def inventory() -> list[dict]:
    rows = []
    for base in (AXIS, EXTR):
        for d, subdirs, fs in os.walk(base):
            subdirs[:] = sorted(s for s in subdirs if s not in SKIP)
            for f in sorted(fs):
                p = os.path.join(d, f)
                if p == PLAN or p == MANIFEST or f.endswith(".pyc"):
                    continue
                rows.append(dict(path=os.path.relpath(p, PROJ), sha256=sha(p), bytes=os.path.getsize(p)))
    return rows


def check_originals(inv: list[dict]) -> dict:
    base = os.path.join(PROJ, "analysis", "s12_layermax_llama")
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
        if got != want:
            bad.append(rel)
    return dict(originals=out, n_bad=len(bad), bad=bad)


def dry_runs(spec: list) -> list[dict]:
    env = {k: v for k, v in os.environ.items() if k != "OL_S0B_TEST"}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OL_THREADS="1")
    res = []
    for label, *argv in spec:
        t0 = time.time()
        p = subprocess.run([PY] + argv, cwd=PROJ, env=env, capture_output=True, text=True, timeout=600)
        res.append(dict(label=label, argv=argv, exit=p.returncode, s=round(time.time() - t0, 1),
                        stderr_tail=p.stderr.strip().splitlines()[-2:] if p.stderr else []))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dryruns", required=True, help="JSON file: list of [label, script, args...] (gated commands)")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--no-status", action="store_true")
    args = ap.parse_args()
    assert sha(PLAN) == PLAN_SHA, "frozen plan changed"
    sys.path.insert(0, AXIS)
    import guard  # noqa: E402
    if guard.code_frozen_line() is not None:
        sys.exit("a CODE FROZEN line already exists; S0b cannot be frozen twice (version instead)")
    inv = inventory()
    orig = check_originals(inv)
    spec = json.load(open(args.dryruns))
    dr = dry_runs(spec)
    f1d = subprocess.run([PY, os.path.join(AXIS, "f1d_outroots.py")], cwd=PROJ, capture_output=True, text=True)
    scan = guard.scan()
    feas_dir = os.path.join(RES, "feasibility")
    feas = sorted((os.path.relpath(os.path.join(d, f), PROJ), sha(os.path.join(d, f)))
                  for d, _, fs in os.walk(feas_dir) for f in fs) if os.path.isdir(feas_dir) else []
    checks = dict(originals_equal=orig["n_bad"] == 0, dry_runs_all_exit_3=all(r["exit"] == 3 for r in dr),
                  f1d_pass=f1d.returncode == 0, feasibility_present=bool(feas), scan_clean=scan["clean"],
                  scan_top=scan["results_top_level"], scan_hits=scan["find_hits"])
    ok = all(v for k, v in checks.items() if k not in ("scan_top", "scan_hits"))
    man = dict(schema="OL-S0B-MANIFEST v1", utc=utc(), plan_sha256=PLAN_SHA, n_files=len(inv), files=inv,
               originals=orig, dry_runs=dr, f1d=dict(exit=f1d.returncode, stdout=f1d.stdout[-4000:]),
               f1d_json_sha256=(sha(os.path.join(RES, "logs", "f1d_outroots.json"))
                                if os.path.exists(os.path.join(RES, "logs", "f1d_outroots.json")) else None),
               feasibility=feas, scan=dict(clean=scan["clean"], top=scan["results_top_level"],
                                           hits=scan["find_hits"], procs=scan["forbidden_processes"]),
               checks=checks, pass_=ok, freezer=dict(path=os.path.relpath(__file__, PROJ), sha256=sha(__file__)))
    print(json.dumps(dict(checks=checks, n_files=len(inv), dry_runs=[(r["label"], r["exit"]) for r in dr],
                          bad_originals=orig["bad"]), indent=1))
    if args.check_only:
        sys.exit(0 if ok else 5)
    if not ok:
        print("S0b NOT frozen: a check failed", file=sys.stderr)
        sys.exit(5)
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=1)
    os.chmod(MANIFEST, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    for r in inv:  # code is read-only from now on; a change is a new versioned file and a logged deviation
        p = os.path.join(PROJ, r["path"])
        x = 0o111 if os.stat(p).st_mode & stat.S_IXUSR else 0  # S0b fix: keep launch_ol.sh executable (555)
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | x)
    msha = sha(MANIFEST)
    files_txt = "; ".join(f"{r['path']} {r['sha256']}" for r in inv)
    dr_txt = ", ".join(f"{r['label']}={r['exit']}" for r in dr)
    line = (f"{utc()} [oll-axis-extract] CODE FROZEN (S0b, plan 4f18d710 section 11) BEFORE S1 and before any "
            f"OpenLLaMA-13B byte exists on ds2213: {len(inv)} files chmod 444, manifest "
            f"analysis/s12_layermax_openllama/S0B_CODE_MANIFEST.json sha256 {msha} (freezer "
            f"{man['freezer']['path']} {man['freezer']['sha256']}); originals of plan section 3: 23/23 equal; F1(a) "
            f"dry runs (OL_S0B_TEST unset) all exit 3: {dr_txt}; F1(d) f1d_outroots.py PASS (json sha256 "
            f"{man['f1d_json_sha256']}); feasibility_ol.py outputs {', '.join(f'{p} {h}' for p, h in feas)}; "
            f"results tree top level {scan['results_top_level']}, find '*open_llama_13b__*' hits "
            f"{len(scan['find_hits'])}; FILES: {files_txt}")
    if not args.no_status:
        with open(STATUS, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line.replace("\n", " ") + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)
    print(line[:3000])
    print(f"[s0b_freeze] manifest sha256 {msha}; STATUS line {'written' if not args.no_status else 'NOT written'}")


if __name__ == "__main__":
    main()
