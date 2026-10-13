#!/usr/bin/env python3
"""S2.3 AMENDMENT A2 (budget removed by the author; complete and analyze) helper.
Lives in s23/resume/, NEVER in src/. Rules: prereg/AMENDMENT_A2_complete.md (A2.1 - A2.7), on top of
PREREG_S2_3_v1.0.md and AMENDMENT_A1_resume.md. Reuses the frozen A1 helper (resume/a1_resume.py, imported
unchanged) for the frozen-code / A1-manifest verification, the finished-item check and the partial scan.

Subcommands
  verify     --mode run|dry|test   A1 verification (37 frozen code files, frozen data, A1 manifest vs its FROZEN
                                   line) + the A2 manifest vs 'S2.3 AMENDMENT A2 (...) FROZEN' (all chmod 444).
  manifest   --out F               sha256 of every output file of every finished plan item (A2.3(b)3).
  a2manifest --out F               sha256 of every A2 file (A2 section 11).
  prep       --attempt D [--dry-run]   A2.3(b)-(c): workers alive?, A1 + A2 finished-item manifests, every
                                   finished item checked, partial files moved aside (not in a dry run).
  remaining  [--json F]            plan items not yet done, per phase / pool / kind / log2N and per frozen worker.
  budget                           ledger and the frozen predicate; rc 0 = frozen GPU pool would refuse
                                   (ledger >= 11.5 h), rc 1 = it would accept.
  g4                               eps_arm0.json exists (chmod 444) and g4_decision.json allows the main runs.
  complete                         rc 0 iff every main item of the plan (frozen reductions applied) has .done.
  ledger                           print the ledger in hours.
  status     ROLE TEXT             append one STATUS line through the frozen s23common.append_status.

Nothing here opens a sample statistic. Test mode (excluded, R29): S23_ROOT=<dir under s23/dev/> and
S23_KEY=hrhv2-s23-TEST, exactly as the frozen unit tests and A1.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys

HOME = os.path.expanduser("~")
S23 = os.path.join(HOME, "hrh_v2_stage2_20261008", "s23")
RES = os.path.join(S23, "resume")
if RES not in sys.path:
    sys.path.insert(0, RES)
import a1_resume as R  # noqa: E402  (frozen A1 helper: imports the frozen s23common, caps threads)

C = R.C
TEST = R.TEST
STATE = R.STATE                                       # <root>/resume
A2_MARK = "S2.3 AMENDMENT A2 (budget removed by author; complete and analyze) FROZEN"
A2_MAN = os.path.join(STATE, "A2_MANIFEST.sha256")
A2_FIN = os.path.join(STATE, "A2_finished_items.sha256")
A1_FIN = os.path.join(STATE, "A1_finished_items.sha256")
PARTIAL = R.PARTIAL                                   # runs/_partial_20261010 (excluded from the partial scan)
A2_REQUIRED = [
    "S23/prereg/AMENDMENT_A2_complete.md",
    "S23/resume/a2_complete.py",
    "S23/resume/a2_worker.py",
    "S23/resume/launch_stageA_A2.sh",
    "S23/resume/tests_A2/run_a2_tests.sh",
    "S23/resume/tests_A2/a2_make_test_state.py",
    "S23/resume/tests_A2/a2_check_test_state.py",
    "ROOT/resume/A2_finished_items.sha256",
]


def status(role: str, text: str) -> str:
    return R.status(role, text)


def a2_frozen_line():
    if not os.path.exists(C.STATUS):
        return None
    with open(C.STATUS) as f:
        hits = [ln.rstrip("\n") for ln in f if A2_MARK in ln]
    return hits


# ============================================================================ verify (A2.7)
def a2_problems(mode: str):
    probs, notes = [], []
    hits = a2_frozen_line() or []
    if not hits:
        if mode == "dry":
            notes.append(f"no '{A2_MARK}' line yet: candidate A2 manifest checked only")
        else:
            probs.append(f"no '{A2_MARK}' line in {C.STATUS}")
    elif len(hits) > 1:
        probs.append(f"{len(hits)} A2 FROZEN lines in {C.STATUS} (expected exactly one)")
    else:
        m = re.search(r"A2_MANIFEST\.sha256 sha256 ([0-9a-f]{64})", hits[0])
        if not m:
            probs.append("A2 FROZEN line carries no A2_MANIFEST.sha256 hash")
        elif not os.path.isfile(A2_MAN) or R.sha(A2_MAN) != m.group(1):
            probs.append("A2_MANIFEST.sha256 differs from the hash in the A2 FROZEN line")
        else:
            notes.append(f"A2 manifest {m.group(1)[:12]} matches the A2 FROZEN line")
    if not os.path.isfile(A2_MAN):
        if mode == "dry":
            notes.append(f"A2 manifest not written yet ({A2_MAN}); A2 files listed only")
            for t in A2_REQUIRED:
                f = R.resolve(t)
                notes.append(f"  {t}: {'present ' + R.sha(f)[:12] if os.path.isfile(f) else 'MISSING'}")
            return probs, notes
        probs.append(f"A2 manifest missing: {A2_MAN}")
        return probs, notes
    ent = R.parse_sha_file(A2_MAN)
    for r in A2_REQUIRED:
        if r not in ent:
            probs.append(f"A2 manifest lacks {r}")
    for t, h in ent.items():
        try:
            f = R.resolve(t)
        except ValueError as e:
            probs.append(str(e))
            continue
        if not os.path.isfile(f):
            probs.append(f"A2 file missing: {t}")
        elif R.sha(f) != h:
            probs.append(f"A2 file hash MISMATCH: {t}")
        elif mode == "run" and R.writable(f):
            probs.append(f"A2 file is writable (not frozen): {t}")
    if mode == "run" and R.writable(A2_MAN):
        probs.append("A2 manifest is writable (not frozen)")
    return probs, notes


def a1_problems_real():
    """The A1 manifest and its FROZEN line always live in the real tree (A1 code files are real files)."""
    if TEST:
        # test mode: a1_resume resolves ROOT/ against the TEST root (copied from the A1 test root T1) and
        # reads the TEST STATUS; the real A1 manifest is checked by the launcher's bash side.
        return R.a1_problems("test")
    return R.a1_problems("run")


def cmd_verify(mode: str) -> int:
    probs = R.frozen_code_problems()
    p1, n1 = a1_problems_real()
    p2, n2 = a2_problems(mode)
    probs += p1 + p2
    for n in n1 + n2:
        print("A2 verify note:", n)
    for p in probs:
        print("A2 verify PROBLEM:", p)
    print(f"A2 verify ({mode}{', TEST root ' + C.ROOT if TEST else ''}): "
          f"{'OK' if not probs else 'FAILED (' + str(len(probs)) + ' problems)'}; {R.N_FROZEN_CODE} frozen code files, "
          f"{len(R.FROZEN_DATA)} frozen data files, A1 manifest, A2 manifest; GPU ledger {C.gpu_hours_used():.4f} h")
    return 0 if not probs else 2


# ============================================================================ finished-items manifest (A2.3(b)3)
def finished_hashes(plan, n):
    lines, bad, nitems = {}, [], 0
    for it in plan["items"]:
        if os.path.exists(R.item_paths(it)[3]):
            errs, hs = R.check_finished(it, n)
            if errs:
                bad.append((it["id"], errs))
            lines.update(hs)
            nitems += 1
    return lines, bad, nitems


def cmd_manifest(out: str) -> int:
    plan = R.load_plan()
    n = R.n_from_arrays()
    hs, bad, nitems = finished_hashes(plan, n)
    partials = R.scan_partials(plan)
    for b in bad:
        print("A2 manifest PROBLEM:", b)
    for p in partials:
        print("A2 manifest: file not belonging to a finished item:", p)
    if bad:
        return 2
    if os.path.exists(out) and not R.writable(out):
        print("A2 manifest: refusing to overwrite a write-protected file", out)
        return 8
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# S2.3 AMENDMENT A2 (rule A2.3(b)3): sha256 of every output file of the {nitems} plan items finished "
                f"before A2 (paths relative to runs/); written {R.utc_iso()}{' (TEST root)' if TEST else ''}\n")
        f.write("".join(f"{hs[k]}  {k}\n" for k in sorted(hs)))
    print(f"A2 manifest: {nitems} finished items, {len(hs)} files, partial files at writing: {len(partials)}; "
          f"{out} sha256 {R.sha(out)}")
    return 0


def cmd_a2manifest(out: str) -> int:
    if os.path.exists(out) and not R.writable(out):
        print("A2 a2manifest: refusing to overwrite a write-protected file", out)
        return 8
    lines = [f"# S2.3 AMENDMENT A2 manifest (sha256  tagged path; S23/ = ~/hrh_v2_stage2_20261008/s23, ROOT/ = the "
             f"output root{' = TEST root ' + C.ROOT if TEST else ' = S23'}); written {R.utc_iso()}"]
    for t in A2_REQUIRED:
        lines.append(f"{R.sha(R.resolve(t))}  {t}")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"A2 a2manifest: {len(A2_REQUIRED)} files; {out} sha256 {R.sha(out)}")
    return 0


# ============================================================================ prep (A2.3(b)-(c))
def a2_workers_alive() -> list:
    out = []
    for d in glob.glob("/proc/[0-9]*"):
        try:
            argv = [x.decode(errors="replace") for x in open(os.path.join(d, "cmdline"), "rb").read().split(b"\0") if x]
            if any(os.path.basename(x) == "a2_worker.py" for x in argv) and "--dry-run" not in argv:
                out.append(int(os.path.basename(d)))
        except Exception:  # noqa: BLE001
            continue
    return out


def cmd_prep(att: str, dry: bool) -> int:
    att_id = os.path.basename(att.rstrip("/"))
    plan = R.load_plan()
    n = R.n_from_arrays()
    probs = []
    alive = R.stage_a_workers_alive() + a2_workers_alive()
    if alive:
        probs.append(f"Stage A worker processes alive: {alive}")
    mans = {}
    for name, p in (("A1", A1_FIN), ("A2", A2_FIN)):
        m = R.parse_sha_file(p) if os.path.exists(p) else {}
        if not m:
            probs.append(f"{name} finished-items manifest missing or empty: {p}")
        for rel, h in m.items():
            f = os.path.join(C.RUNS, rel)
            if not os.path.exists(f):
                probs.append(f"{name} manifest file missing: {rel}")
            elif R.sha(f) != h:
                probs.append(f"{name} manifest file hash MISMATCH: {rel}")
        mans[name] = m
    by_id = {}
    for rel in mans["A2"]:
        iid = os.path.basename(rel).split(".", 1)[0].split("__", 1)[0]
        by_id.setdefault(iid, set()).add(rel)
    finished = [it for it in plan["items"] if os.path.exists(R.item_paths(it)[3])]
    fin_ids = {it["id"] for it in finished}
    later = {}
    for it in finished:
        errs, hs = R.check_finished(it, n)
        if errs:
            probs.append(f"{it['id']}: {errs}")
        if it["id"] in by_id:
            if set(hs) != by_id[it["id"]]:
                probs.append(f"{it['id']}: output file set differs from the A2 manifest")
        else:
            later.update(hs)
    missing = [iid for iid in by_id if iid not in fin_ids]
    if missing:
        probs.append(f"A2 manifest items without .done now: {missing[:5]}")
    partials = R.scan_partials(plan)
    moved, dest_root = [], os.path.join(C.RUNS, PARTIAL, "A2_" + att_id)
    if not dry and not probs:
        for rel in partials:
            src, dst = os.path.join(C.RUNS, rel), os.path.join(dest_root, rel)
            if os.path.exists(dst):
                probs.append(f"partial destination exists: {dst}")
                break
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved.append(rel)
    n_later = len({os.path.basename(k).split(".", 1)[0].split("__", 1)[0] for k in later})
    n_a2_items = len(by_id)
    rec = {"utc": R.utc_iso(), "attempt": att_id, "dry_run": dry, "n_finished": len(finished),
           "n_A1_manifest_files": len(mans["A1"]), "n_A2_manifest_items": n_a2_items,
           "n_A2_manifest_files": len(mans["A2"]), "n_finished_after_A2_freeze": n_later, "partials": partials,
           "moved": moved, "moved_to": dest_root if moved else None, "problems": probs,
           "ledger_h": C.gpu_hours_used()}
    os.makedirs(att, exist_ok=True)
    with open(os.path.join(att, "prep.json"), "w") as f:
        json.dump(rec, f, indent=1)
    if later:
        with open(os.path.join(att, "finished_after_A2_freeze.sha256"), "w") as f:
            f.write("".join(f"{later[k]}  {k}\n" for k in sorted(later)))
    print(json.dumps({k: v for k, v in rec.items() if k != "partials"}, indent=1))
    print("partials:", partials)
    if not dry:
        if probs:
            status("run-A2", f"A2 prep REFUSED (attempt {att_id}; rule A2.3(b)): {len(probs)} problems, first: {probs[0]}; "
                             f"no worker started")
        else:
            status("run-A2", f"A2 prep OK (attempt {att_id}; rule A2.3(b)-(c)): {len(finished)} finished items checked; "
                             f"A1 manifest {len(mans['A1'])} files and A2 manifest {len(mans['A2'])} files "
                             f"({n_a2_items} items) hash-equal; {n_later} items finished after the A2 freeze checked; "
                             f"partial files moved aside: {len(moved)}"
                             + (f" -> runs/{PARTIAL}/A2_{att_id}/ ({', '.join(moved[:6])}{' ...' if len(moved) > 6 else ''})"
                                if moved else "") + f"; GPU ledger {C.gpu_hours_used():.4f} h")
    return 0 if not probs else 2


# ============================================================================ remaining / budget / g4 / complete
def reductions():
    p = os.path.join(C.STAGEA_DIR, "g4_decision.json")
    if not os.path.exists(p):
        return None
    return set(C.load_json(p).get("reductions", []))


def main_items(plan, red):
    return [it for it in plan["items"] if it["phase"] == "main"
            and not any(x in red and it.get(f"red_{x}") for x in ("a", "b", "c"))]


def cmd_remaining(out=None) -> int:
    from collections import Counter
    plan = R.load_plan()
    red = reductions() or set()
    todo, done, per_worker, todo_ids = Counter(), Counter(), Counter(), {}
    for ph in ("pre", "main"):
        for pool, nw in (("gpu", 2), ("cpu", 1)):
            its = [it for it in plan["items"] if it["phase"] == ph and R.pool_of(it) == pool]
            if ph == "main":
                its = [it for it in its if not any(x in red and it.get(f"red_{x}") for x in ("a", "b", "c"))]
            for i, it in enumerate(its):
                key = (ph, pool, it["kind"], it.get("arm", "-"), it["log2N"])
                if os.path.exists(R.item_paths(it)[3]):
                    done[key] += 1
                else:
                    todo[key] += 1
                    w = f"{pool}{i % nw}"
                    per_worker[(ph, w)] += 1
                    todo_ids.setdefault(f"{ph}:{w}", []).append(it["id"])
    print("phase pool kind arm log2N done todo")
    for k in sorted(set(todo) | set(done), key=str):
        print(*k, done[k], todo[k])
    for ph in ("pre", "main"):
        print(ph, "todo:", sum(v for k, v in todo.items() if k[0] == ph), "done:",
              sum(v for k, v in done.items() if k[0] == ph), "per worker (frozen split i % nw):",
              {w: v for (p, w), v in sorted(per_worker.items()) if p == ph})
    print("reductions applied:", sorted(red))
    if out:
        with open(out, "w") as f:
            json.dump({"reductions": sorted(red), "todo_ids": todo_ids}, f, indent=1)
    return 0


def cmd_budget() -> int:
    used = C.gpu_hours_used()
    soft_ok = C.gpu_budget_ok(soft=True)          # frozen predicate, frozen constants (this process is unpatched)
    hard_ok = C.gpu_budget_ok(soft=False)
    print(f"GPU ledger {used:.4f} h; frozen constants soft {C.GPU_SOFT_STOP_H} h, hard {C.GPU_HARD_STOP_H} h; "
          f"frozen gpu_budget_ok(soft=True)={soft_ok}, gpu_budget_ok(soft=False)={hard_ok} -> the frozen GPU pool "
          f"would {'ACCEPT new items' if soft_ok else 'REFUSE every remaining item (worker.py exits rc 4)'}")
    return 1 if soft_ok else 0


def cmd_g4() -> int:
    e = os.path.join(C.STAGEA_DIR, "eps_arm0.json")
    g = os.path.join(C.STAGEA_DIR, "g4_decision.json")
    if not os.path.exists(e):
        print("A2 g4: eps_arm0.json missing (gates phase never ran): A2 does not run the gates phase")
        return 8
    if R.writable(e):
        print("A2 g4: eps_arm0.json is not write-protected")
        return 8
    if not os.path.exists(g):
        print("A2 g4: eps_arm0.json exists without g4_decision.json")
        return 8
    d = C.load_json(g)
    print(f"A2 g4: eps_arm0.json sha256 {R.sha(e)} (chmod 444); g4_decision.json sha256 {R.sha(g)}: "
          f"main_allowed={d.get('main_allowed')}, reductions={d.get('reductions')}")
    return 0 if d.get("main_allowed", False) else 3


def cmd_complete() -> int:
    plan = R.load_plan()
    red = reductions()
    if red is None:
        print("A2 complete: g4_decision.json missing")
        return 1
    its = main_items(plan, red)
    miss = [it["id"] for it in its if not os.path.exists(R.item_paths(it)[3])]
    print(f"A2 complete: {len(its) - len(miss)}/{len(its)} main items have .done (reductions {sorted(red)})"
          + (f"; missing e.g. {miss[:5]}" if miss else ""))
    return 0 if not miss else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("verify")
    p.add_argument("--mode", choices=["run", "dry", "test"], required=True)
    p = sp.add_parser("manifest")
    p.add_argument("--out", required=True)
    p = sp.add_parser("a2manifest")
    p.add_argument("--out", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--attempt", required=True)
    p.add_argument("--dry-run", action="store_true")
    p = sp.add_parser("remaining")
    p.add_argument("--json")
    sp.add_parser("budget")
    sp.add_parser("g4")
    sp.add_parser("complete")
    sp.add_parser("ledger")
    p = sp.add_parser("status")
    p.add_argument("role")
    p.add_argument("text")
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        return cmd_verify(a.mode)
    if a.cmd == "manifest":
        return cmd_manifest(a.out)
    if a.cmd == "a2manifest":
        return cmd_a2manifest(a.out)
    if a.cmd == "prep":
        return cmd_prep(a.attempt, a.dry_run)
    if a.cmd == "remaining":
        return cmd_remaining(a.json)
    if a.cmd == "budget":
        return cmd_budget()
    if a.cmd == "g4":
        return cmd_g4()
    if a.cmd == "complete":
        return cmd_complete()
    if a.cmd == "ledger":
        print(f"{C.gpu_hours_used():.4f}")
        return 0
    if a.cmd == "status":
        status(a.role, a.text)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
