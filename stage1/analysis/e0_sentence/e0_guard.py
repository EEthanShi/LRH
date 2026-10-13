"""E0 guard before the first cross-model score with a sentence encoder, and the post-run birth-time audit
(ANALYSIS_PLAN_E0.md v1.0 §11 steps 6-7; critique R5).

    venv/bin/python -I analysis/e0_sentence/e0_guard.py check     # writes results/E0_sentence/guard/GUARD_CHECK_<utc>.json
    venv/bin/python -I analysis/e0_sentence/e0_guard.py audit --out analysis/e0_sentence/review/audit_<utc>.json

check: (i) STATUS lines PLAN FROZEN, E0 CODE FROZEN, EXTRACTION DONE, CTR CALIBRATION FROZEN in this order with
the files' sha256; (ii) every file of CODE_MANIFEST_E0.json has its frozen sha256; (iii) negative scan: no
new-pair obs, eval or verdict file exists. Exit 0 = CLEAN, 4 = VIOLATION.
audit: every new-pair obs/eval/verdict file has a birth time (coreutils `stat -c %W`) after the GUARD CLEAN
STATUS line; every E0 feature file is born after PLAN FROZEN; every E0 cache after E0 CODE FROZEN.
Exit 0 = pass, 4 = violation, 3 = no GUARD CLEAN line.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e0_common as E  # noqa: E402

OBS_OLD_OK = {f"obs_{E.OLD_ARM}.parquet", f"obs_{E.OLD_ARM}.json"}
EVAL_OLD_OK = {"ctr_calibration.json"} | {f"E0_{n}_{E.OLD_ARM}.parquet" for n in ("cells", "drawslopes", "slopes", "elasticity")}


def forbidden_files(out_root: str = E.OUT) -> list:
    """New-pair obs / eval / verdict files that must not exist before the guard (plan §11 step 6 (iii))."""
    bad = []
    od = os.path.join(out_root, "obs")
    if os.path.isdir(od):
        bad += [os.path.join(od, f) for f in sorted(os.listdir(od)) if f not in OBS_OLD_OK]
    ed = os.path.join(out_root, "eval")
    if os.path.isdir(ed):
        bad += [os.path.join(ed, f) for f in sorted(os.listdir(ed)) if f not in EVAL_OLD_OK]
    for f in ("VERDICTS_E0.json", "E0_summary.json", "E0_summary.md", "DONE"):
        if os.path.exists(os.path.join(out_root, f)):
            bad.append(os.path.join(out_root, f))
    return bad


def manifest_check(manifest: str = E.MANIFEST) -> list:
    with open(manifest) as f:
        man = json.load(f)
    bad = []
    for rel, h in man["files"].items():
        p = os.path.join(E.PROJECT, rel)
        if not os.path.exists(p):
            bad.append({"file": rel, "problem": "missing"})
        elif E.sha256_file(p) != h:
            bad.append({"file": rel, "problem": "sha256 differs from manifest"})
    return bad


def check(out_root: str = E.OUT, status: str = E.STATUS) -> dict:
    res = {"utc": E.utcnow(), "violations": []}
    try:
        res["milestones"] = E.require_milestones("plan", "code", "extract", "ctrcal", path=status)
    except SystemExit as e:
        res["violations"].append({"milestones": str(e)})
    try:
        res["manifest_problems"] = manifest_check()
        if res["manifest_problems"]:
            res["violations"].append({"manifest": res["manifest_problems"]})
    except FileNotFoundError as e:
        res["violations"].append({"manifest": str(e)})
    res["forbidden_files_present"] = forbidden_files(out_root)
    if res["forbidden_files_present"]:
        res["violations"].append({"negative_scan": res["forbidden_files_present"]})
    if E.find_status(r"\[e0-run\] GUARD CLEAN", None, status) is not None:
        res["note"] = "a GUARD CLEAN line already exists in STATUS"
    res["result"] = "CLEAN" if not res["violations"] else "VIOLATION"
    return res


def births(paths: list) -> dict:
    """path -> birth epoch (coreutils stat %W; 0 = unknown -> None)."""
    out = {}
    for i in range(0, len(paths), 400):
        chunk = paths[i:i + 400]
        r = subprocess.run(["stat", "--printf", "%W|%n\n"] + chunk, capture_output=True, text=True, check=True)
        for ln in r.stdout.splitlines():
            w, n = ln.split("|", 1)
            out[n] = int(w) if int(w) > 0 else None
    return out


def status_epoch(utc: str) -> int:
    return calendar.timegm(time.strptime(utc, "%Y-%m-%dT%H:%M:%SZ"))


def walk(root: str) -> list:
    out = []
    for d, _, fs in os.walk(root):
        out += [os.path.join(d, f) for f in fs]
    return sorted(out)


def audit(out_root: str = E.OUT, status: str = E.STATUS) -> tuple[dict, int]:
    g = E.find_status(r"\[e0-run\] GUARD CLEAN", None, status)
    if g is None:
        return {"result": "NO GUARD LINE"}, 3
    t_guard = status_epoch(g[1])
    ms = {n: E.milestone(n, status) for n in ("plan", "code", "extract", "ctrcal")}
    res = {"utc": E.utcnow(), "guard_line_utc": g[1], "milestones": ms, "violations": []}
    order = [ms[n]["line"] if ms[n] else None for n in ("plan", "code", "extract", "ctrcal")] + [g[0]]
    if None in order or order != sorted(order):
        res["violations"].append({"status_order": order})
    new_files = [p for p in walk(os.path.join(out_root, "obs")) if os.path.basename(p) not in OBS_OLD_OK]
    new_files += [p for p in walk(os.path.join(out_root, "eval")) if os.path.basename(p) not in EVAL_OLD_OK]
    new_files += [os.path.join(out_root, "VERDICTS_E0.json")] if os.path.exists(os.path.join(out_root, "VERDICTS_E0.json")) else []
    b = births(new_files) if new_files else {}
    early = [p for p in new_files if b.get(p) is None or b[p] < t_guard]
    if early:
        res["violations"].append({"new_pair_files_born_before_guard_or_unknown": early})
    if ms["plan"]:
        t_plan = status_epoch(ms["plan"]["utc"])
        feats = walk(E.FEAT)
        bf = births(feats) if feats else {}
        early = [p for p in feats if bf.get(p) is None or bf[p] < t_plan]
        if early:
            res["violations"].append({"feature_files_born_before_plan_freeze": early})
    if ms["code"]:
        t_code = status_epoch(ms["code"]["utc"])
        caches = walk(os.path.join(out_root, "cache"))
        bc = births(caches) if caches else {}
        early = [p for p in caches if bc.get(p) is None or bc[p] < t_code]
        if early:
            res["violations"].append({"cache_files_born_before_code_freeze": early})
    res["n_new_pair_files"] = len(new_files)
    res["result"] = "PASS" if not res["violations"] else "VIOLATION"
    return res, (0 if not res["violations"] else 4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check", "audit"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.mode == "check":
        res = check()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        path = os.path.join(E.GUARD, f"GUARD_CHECK_{stamp}.json")
        h = E.write_json_new(path, res)
        print(res["result"], path, h)
        if res["result"] == "CLEAN":
            print(f"suggested STATUS: <UTC> [e0-run] GUARD CLEAN: results/E0_sentence/guard/{os.path.basename(path)} sha256 {h}")
        sys.exit(0 if res["result"] == "CLEAN" else 4)
    res, code = audit()
    if a.out:
        E.write_json_new(a.out, res)
    print(json.dumps(E.jsonable({k: v for k, v in res.items() if k != "milestones"}), indent=1))
    sys.exit(code)


if __name__ == "__main__":
    main()
