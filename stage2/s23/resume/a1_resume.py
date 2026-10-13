#!/usr/bin/env python3
"""S2.3 AMENDMENT A1 (resume) helper.  Lives in s23/resume/, NEVER in src/ (src/ must keep exactly its
frozen file set).  Rules: prereg/AMENDMENT_A1_resume.md (A1.1 - A1.6) on top of PREREG_S2_3_v1.0.md.

Subcommands
  verify     --mode run|dry|test   frozen code (independent re-hash of the 37 files + the frozen
                                   s23common.verify_frozen_code), frozen data hashes, write protection,
                                   A1 manifest against the STATUS line 'S2.3 AMENDMENT A1 (resume) FROZEN'.
  manifest   --out F               A1.1(b): sha256 of every output file of every finished plan item
                                   (written once, at the A1 freeze).
  a1manifest --out F               sha256 of every A1 file (A1 section 8).
  prep       --attempt D [--dry-run]   A1.1(b)-(c): re-verify finished items, move partial files aside.
  sidecar    --attempt D [...]     A1.2(a): nvidia-smi compute-apps snapshot every 2 s; A1.3 speed guard.
  guard      --attempt D --final   A1.3 one synchronous guard evaluation (end of run_pool pre).
  gates      --attempt D           A1.2(b)-(e): free classification, results/stageA/A1_g4_inputs.json, then
                                   the FROZEN analyze_A gates phase with non-free timings masked (NaN).
  classify   [--snap-file F]       print the free classification of every A-pilot / G1b ledger line.
  remaining                        plan items not yet done, per phase / pool / kind (no computation).
  smi        [--fake F]            one compute-apps query: prints 'rc n_compute_processes'.
  selftest   --tmp D               synthetic tests of the classifier, guard, mask, partial scan, smi parsing.
  status     ROLE TEXT             append one STATUS line through the frozen s23common.append_status.

Nothing here opens a sample statistic, except 'gates', which runs the frozen gates phase (A1.2).
Test mode (excluded, R29): S23_ROOT=<dir under s23/dev/> and S23_KEY=hrhv2-s23-TEST, exactly as the frozen
unit tests; only then are A1_GUARD_S, A1_SNAP_INTERVAL and A1_FAKE_SMI honoured.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import glob
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
S23 = os.path.join(HOME, "hrh_v2_stage2_20261008", "s23")
SRC = os.path.join(S23, "src")
TEST = "S23_ROOT" in os.environ
if TEST:
    _r = os.path.realpath(os.environ["S23_ROOT"])
    if not _r.startswith(os.path.join(os.path.realpath(S23), "dev") + os.sep):
        raise SystemExit("A1: a test root must lie under s23/dev/")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import s23common as C  # noqa: E402  (frozen; in test mode every output path follows S23_ROOT)

C.cap_threads(1)  # before any numpy import (only sets environment variables)

STATE = os.path.join(C.ROOT, "resume")                 # markers, attempts, manifests
A1_MARK = "S2.3 AMENDMENT A1 (resume) FROZEN"
PARTIAL = "_partial_20261010"
SNAP_DIR = os.path.join(C.LOGS, "A1_gpu_snapshots")
SNAP_INTERVAL = 2.0                                     # s between compute-apps snapshots (A1.2(a))
MAX_GAP = 10.0                                          # s: before / after / gap limit (A1.2(a))
GUARD_N = 10                                            # A1.3: first 10 A-pilot solves of an attempt
GUARD_S = 30.0                                          # A1.3: median threshold, GPU seconds
if TEST and os.environ.get("A1_GUARD_S"):
    GUARD_S = float(os.environ["A1_GUARD_S"])          # excluded tests only
if TEST and os.environ.get("A1_SNAP_INTERVAL"):
    SNAP_INTERVAL = float(os.environ["A1_SNAP_INTERVAL"])  # excluded tests only
A_PILOT_RE = re.compile(r"^0_A_l\d+_r\d{3}_pilot$")
SMI_CMD = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]

FROZEN_DATA = {   # real tree, always (read-only)
    "prereg/PREREG_S2_3_v1.0.md": "7a93b7a9519b505ee887467baab7aa724d2a0ee05635a2afdc363da280e6b5d6",
    "prereg/CODE_AT_PREREG.sha256": "f9560f0a9d3d2bc44c8407d1a9d3d440c0e8a0960765c67cc1709c7c039cf546",
    "results/prestep/frozen_predictions.json": "44d45c418da1f505bfec0c7500e1a5c2d3b5504b7d776760a0498a8b85bfb4cb",
    "results/prestep/prestep_arrays.npz": "7c560f00299c5532f687af0efd9b230f0f1f1782182e6528c76a281c73110429",
    "plans/stageA_plan.json": "dfeb2c96b80ea5cddeec5e1331206ce40e88b260b716111c3bb661ae9a17980b",
}
N_FROZEN_CODE = 37
A1_REQUIRED = [
    "S23/prereg/AMENDMENT_A1_resume.md",
    "S23/resume/a1_resume.py",
    "S23/resume/launch_stageA_resume.sh",
    "S23/resume/gate_then_launch.sh",
    "S23/resume/tests/a1_make_test_root.py",
    "S23/resume/tests/a1_check_test_root.py",
    "S23/resume/tests/run_a1_tests.sh",
    "ROOT/resume/A1_finished_items.sha256",
    "ROOT/logs/gpu_ledger/A1_ledger_correction.tsv",
]
GPU_KINDS = ("A", "G1b", "G3A", "D2")


# ============================================================================ small helpers
def sha(p: str) -> str:
    return C.sha256_file(p)


def utc_iso(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_epoch(s: str) -> float:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def resolve(tagged: str) -> str:
    if tagged.startswith("S23/"):
        return os.path.join(S23, tagged[4:])
    if tagged.startswith("ROOT/"):
        return os.path.join(C.ROOT, tagged[5:])
    raise ValueError(f"untagged A1 manifest path {tagged}")


def writable(p: str) -> bool:
    return bool(os.stat(p).st_mode & 0o222)


def load_plan():
    return C.load_json(os.path.join(C.PLANS, "stageA_plan.json"))


def item_paths(it, runs=None):
    d = os.path.join(runs or C.RUNS, it["cell"])
    return d, os.path.join(d, it["id"] + ".npy"), os.path.join(d, it["id"] + ".json"), os.path.join(d, it["id"] + ".done")


def pool_of(it) -> str:
    return "gpu" if it["kind"] in GPU_KINDS else "cpu"


def n_from_arrays() -> int:
    import numpy as np
    with np.load(C.PRESTEP_ARR) as z:
        return int(z["0_A"].shape[0])


def status(role: str, text: str) -> str:
    line = C.append_status(role, text)
    print(line, end="")
    return line


def frozen_line():
    if not os.path.exists(C.STATUS):
        return None
    with open(C.STATUS) as f:
        for line in f:
            if A1_MARK in line:
                return line.rstrip("\n")
    return None


def parse_sha_file(p: str) -> dict:
    out = {}
    with open(p) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            h, q = line.split(None, 1)
            out[q.strip()] = h
    return out


# ============================================================================ verify (A1.6)
def frozen_code_problems() -> list:
    probs = []
    ref = os.path.join(S23, "prereg", "CODE_AT_PREREG.sha256")
    want = parse_sha_file(ref)
    if len(want) != N_FROZEN_CODE:
        probs.append(f"CODE_AT_PREREG.sha256 lists {len(want)} files, expected {N_FROZEN_CODE}")
    for p, h in want.items():
        f = os.path.join(S23, p) if p.startswith("src/") else os.path.join(HOME, p)
        if not os.path.isfile(f):
            probs.append(f"frozen file missing: {p}")
        elif sha(f) != h:
            probs.append(f"frozen code hash MISMATCH: {p}")
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), S23)
            if rel not in want:
                probs.append(f"file in src/ that is not frozen: {rel}")
            elif writable(os.path.join(dirpath, fn)):
                probs.append(f"frozen file is writable: {rel}")
    if not TEST:
        try:
            C.verify_frozen_code(strict=True)
        except Exception as e:  # noqa: BLE001
            probs.append(f"s23common.verify_frozen_code: {e}")
    for p, h in FROZEN_DATA.items():
        f = os.path.join(S23, p)
        if not os.path.isfile(f):
            probs.append(f"frozen data file missing: {p}")
            continue
        if sha(f) != h:
            probs.append(f"frozen data hash MISMATCH: {p}")
        if writable(f):
            probs.append(f"frozen data file is writable: {p}")
    return probs


def a1_problems(mode: str):
    probs, notes = [], []
    man = os.path.join(STATE, "A1_MANIFEST.sha256")
    line = frozen_line()
    if line is None:
        if mode == "dry":
            notes.append("no 'S2.3 AMENDMENT A1 (resume) FROZEN' line yet: candidate manifest checked only")
        else:
            probs.append(f"no '{A1_MARK}' line in {C.STATUS}")
    else:
        m = re.search(r"A1_MANIFEST\.sha256 sha256 ([0-9a-f]{64})", line)
        if not m:
            probs.append("FROZEN line carries no A1_MANIFEST.sha256 hash")
        elif not os.path.isfile(man) or sha(man) != m.group(1):
            probs.append("A1_MANIFEST.sha256 differs from the hash in the FROZEN line")
        else:
            notes.append(f"A1 manifest {m.group(1)[:12]} matches the FROZEN line")
    if not os.path.isfile(man):
        probs.append(f"A1 manifest missing: {man}")
        return probs, notes
    ent = parse_sha_file(man)
    for r in A1_REQUIRED:
        if r not in ent:
            probs.append(f"A1 manifest lacks {r}")
    for t, h in ent.items():
        try:
            f = resolve(t)
        except ValueError as e:
            probs.append(str(e))
            continue
        if not os.path.isfile(f):
            probs.append(f"A1 file missing: {t}")
        elif sha(f) != h:
            probs.append(f"A1 file hash MISMATCH: {t}")
        elif mode == "run" and writable(f):
            probs.append(f"A1 file is writable (not frozen): {t}")
    if mode == "run" and writable(man):
        probs.append("A1 manifest is writable (not frozen)")
    return probs, notes


def cmd_verify(mode: str) -> int:
    probs = frozen_code_problems()
    p2, notes = a1_problems(mode)
    probs += p2
    for n in notes:
        print("A1 verify note:", n)
    for p in probs:
        print("A1 verify PROBLEM:", p)
    print(f"A1 verify ({mode}{', TEST root ' + C.ROOT if TEST else ''}): "
          f"{'OK' if not probs else 'FAILED (' + str(len(probs)) + ' problems)'}; "
          f"{N_FROZEN_CODE} frozen code files, {len(FROZEN_DATA)} frozen data files, GPU ledger {C.gpu_hours_used():.4f} h")
    return 0 if not probs else 2


# ============================================================================ finished items (A1.1(b))
def check_finished(it, n: int, runs=None):
    import numpy as np
    runs = runs or C.RUNS
    d, pk, pj, pd = item_paths(it, runs)
    errs, hashes = [], {}
    if os.path.getsize(pd) != 0:
        errs.append("done marker not empty")
    js = None
    try:
        js = C.load_json(pj)
    except Exception as e:  # noqa: BLE001
        errs.append(f"json unreadable ({e.__class__.__name__})")
    if js is not None:
        if js.get("id") != it["id"]:
            errs.append("json id differs")
        if js.get("item") != it:
            errs.append("json item record differs from the plan item")
    arrays = [pk] + sorted(glob.glob(os.path.join(glob.escape(d), glob.escape(it["id"]) + "__*.npy")))
    for f in arrays:
        if not os.path.exists(f):
            errs.append(f"missing {os.path.basename(f)}")
            continue
        try:
            a = np.load(f, allow_pickle=False, mmap_mode="r")
            if a.dtype != np.float64 or a.shape != (n, n):
                errs.append(f"{os.path.basename(f)}: dtype {a.dtype} shape {a.shape}")
        except Exception as e:  # noqa: BLE001
            errs.append(f"{os.path.basename(f)} unreadable ({e.__class__.__name__})")
    for f in arrays + [pj, pd]:
        if os.path.exists(f):
            hashes[os.path.relpath(f, runs)] = sha(f)
    return errs, hashes


def scan_partials(plan, runs=None) -> list:
    """Every file under runs/ that is not an output of a finished plan item (A1.1(c))."""
    runs = runs or C.RUNS
    items = {it["id"]: it for it in plan["items"]}
    out = []
    for dirpath, dirnames, filenames in os.walk(runs):
        rel_dir = os.path.relpath(dirpath, runs)
        if rel_dir == ".":
            dirnames[:] = [x for x in dirnames if x != PARTIAL]
        for fn in filenames:
            iid = fn.split(".", 1)[0].split("__", 1)[0]
            it = items.get(iid)
            ok = False
            if it is not None and it["cell"] == rel_dir.replace(os.sep, "/") \
                    and os.path.exists(os.path.join(dirpath, iid + ".done")):
                ok = fn in (iid + ".npy", iid + ".json", iid + ".done") or \
                    (fn.startswith(iid + "__") and fn.endswith(".npy") and "." not in fn[:-4])
            if not ok:
                out.append(os.path.normpath(os.path.join(rel_dir, fn)))
    return sorted(out)


def cmd_manifest(out: str) -> int:
    plan = load_plan()
    n = n_from_arrays()
    lines, bad, nitems = [], [], 0
    for it in plan["items"]:
        if os.path.exists(item_paths(it)[3]):
            errs, hs = check_finished(it, n)
            if errs:
                bad.append((it["id"], errs))
            lines += [f"{hs[k]}  {k}" for k in sorted(hs)]
            nitems += 1
    partials = scan_partials(plan)
    for b in bad:
        print("A1 manifest PROBLEM:", b)
    for p in partials:
        print("A1 manifest: file not belonging to a finished item:", p)
    if bad:
        return 2
    if os.path.exists(out) and not writable(out):
        print("A1 manifest: refusing to overwrite a write-protected file", out)
        return 8
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# S2.3 AMENDMENT A1 (rule A1.1(b)): sha256 of every output file of the {nitems} plan items finished "
                f"before A1 (paths relative to runs/); written {utc_iso()}{' (TEST root)' if TEST else ''}\n")
        f.write("\n".join(lines) + "\n")
    print(f"A1 manifest: {nitems} finished items, {len(lines)} files, partial files at writing: {len(partials)}; "
          f"{out} sha256 {sha(out)}")
    return 0


def cmd_a1manifest(out: str) -> int:
    if os.path.exists(out) and not writable(out):
        print("A1 a1manifest: refusing to overwrite a write-protected file", out)
        return 8
    lines = [f"# S2.3 AMENDMENT A1 manifest (sha256  tagged path; S23/ = ~/hrh_v2_stage2_20261008/s23, ROOT/ = the "
             f"output root{' = TEST root ' + C.ROOT if TEST else ' = S23'}); written {utc_iso()}"]
    for t in A1_REQUIRED:
        lines.append(f"{sha(resolve(t))}  {t}")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"A1 a1manifest: {len(A1_REQUIRED)} files; {out} sha256 {sha(out)}")
    return 0


# ============================================================================ prep (A1.1(b)-(c))
def stage_a_workers_alive() -> list:
    out = []
    src_real = os.path.realpath(SRC)
    for d in glob.glob("/proc/[0-9]*"):
        try:
            argv = [a.decode(errors="replace") for a in open(os.path.join(d, "cmdline"), "rb").read().split(b"\0") if a]
            if "worker.py" in argv and "--phase" in argv and os.path.realpath(os.path.join(d, "cwd")) == src_real:
                out.append(int(os.path.basename(d)))
        except Exception:  # noqa: BLE001
            continue
    return out


def cmd_prep(att: str, dry: bool) -> int:
    att_id = os.path.basename(att.rstrip("/"))
    plan = load_plan()
    n = n_from_arrays()
    probs = []
    alive = stage_a_workers_alive()
    if alive:
        probs.append(f"Stage A worker processes alive: {alive}")
    manp = os.path.join(STATE, "A1_finished_items.sha256")
    man = parse_sha_file(manp) if os.path.exists(manp) else {}
    if not man:
        probs.append("finished-items manifest missing or empty")
    for rel, h in man.items():
        f = os.path.join(C.RUNS, rel)
        if not os.path.exists(f):
            probs.append(f"manifest file missing: {rel}")
        elif sha(f) != h:
            probs.append(f"manifest file hash MISMATCH: {rel}")
    man_by_id = {}
    for rel in man:
        iid = os.path.basename(rel).split(".", 1)[0].split("__", 1)[0]
        man_by_id.setdefault(iid, set()).add(rel)
    finished = [it for it in plan["items"] if os.path.exists(item_paths(it)[3])]
    later = {}
    for it in finished:
        errs, hs = check_finished(it, n)
        if errs:
            probs.append(f"{it['id']}: {errs}")
        if it["id"] in man_by_id:
            if set(hs) != man_by_id[it["id"]]:
                probs.append(f"{it['id']}: output file set differs from the manifest")
        else:
            later.update(hs)
    missing_done = [iid for iid in man_by_id if not any(it["id"] == iid for it in finished)]
    if missing_done:
        probs.append(f"manifest items without .done now: {missing_done[:5]}")
    partials = scan_partials(plan)
    moved, dest_root = [], os.path.join(C.RUNS, PARTIAL, att_id)
    if not dry and not probs:
        for rel in partials:
            src, dst = os.path.join(C.RUNS, rel), os.path.join(dest_root, rel)
            if os.path.exists(dst):
                probs.append(f"partial destination exists: {dst}")
                break
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved.append(rel)
    n_later_items = len({os.path.basename(k).split(".", 1)[0].split("__", 1)[0] for k in later})
    rec = {"utc": utc_iso(), "attempt": att_id, "dry_run": dry, "n_finished": len(finished),
           "n_manifest_files": len(man), "n_finished_after_A1": n_later_items, "partials": partials,
           "moved": moved, "moved_to": dest_root if moved else None, "problems": probs,
           "ledger_h": C.gpu_hours_used()}
    os.makedirs(att, exist_ok=True)
    with open(os.path.join(att, "prep.json"), "w") as f:
        json.dump(rec, f, indent=1)
    if later:
        with open(os.path.join(att, "finished_after_A1.sha256"), "w") as f:
            f.write("".join(f"{later[k]}  {k}\n" for k in sorted(later)))
    print(json.dumps({k: v for k, v in rec.items() if k != "partials"}, indent=1))
    print("partials:", partials)
    if not dry:
        if probs:
            status("resume", f"A1 prep REFUSED (attempt {att_id}; rule A1.1(b)): {len(probs)} problems, first: {probs[0]}; "
                             f"no worker started")
        else:
            status("resume", f"A1 prep OK (attempt {att_id}; rule A1.1): {len(finished)} finished items verified "
                             f"({len(man)} manifest files hash-equal; {n_later_items} items finished in earlier A1 "
                             f"attempts checked); partial files moved aside: {len(moved)}"
                             + (f" -> runs/{PARTIAL}/{att_id}/ ({', '.join(moved[:6])}{' ...' if len(moved) > 6 else ''})"
                                if moved else "") + f"; GPU ledger {C.gpu_hours_used():.4f} h")
    return 0 if not probs else 2


# ============================================================================ GPU snapshots (A1.2(a))
def parse_smi_text(txt: str):
    """-> (rc, [(pid, name, mem)]); an unparseable line counts as a failed query."""
    apps = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no running processes"):
            continue
        parts = [x.strip() for x in line.split(",")]
        try:
            pid = int(parts[0])
        except ValueError:
            return 2, []
        apps.append((pid, parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else ""))
    return 0, apps


def query_smi(own=None, fake=None):
    fake = fake or (os.environ.get("A1_FAKE_SMI") if TEST else None)
    if fake:
        try:
            txt = open(fake).read()
        except OSError:
            return 1, []
        if txt.strip() == "ERR":
            return 1, []
        lines = []
        for line in txt.splitlines():
            if line.strip() == "OWN":
                lines += [f"{p}, python, 100" for p in sorted(own or {})]
            else:
                lines.append(line)
        return parse_smi_text("\n".join(lines))
    try:
        r = subprocess.run(SMI_CMD, capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return 1, []
    if r.returncode != 0:
        return r.returncode, []
    return parse_smi_text(r.stdout)


def own_pids(att: str) -> dict:
    pids = {}
    for p in glob.glob(os.path.join(att, "pids_*")):
        phase = os.path.basename(p)[5:]
        with open(p) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    pids[int(parts[1])] = f"{parts[0]}_{phase}"
    return pids


def load_snapshots(files=None):
    files = sorted(glob.glob(os.path.join(SNAP_DIR, "*.tsv"))) if files is None else files
    snaps = []
    for p in files:
        with open(p) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                try:
                    t0, t1, rc, nfor = float(parts[0]), float(parts[1]), int(parts[3]), int(parts[5])
                except ValueError:
                    continue
                snaps.append((t0, t1, rc == 0 and nfor == 0))
    snaps.sort()
    return snaps


def classify_window(snaps, ledger_iso: str, seconds: float):
    """A1.2(a): free iff a snapshot ended <= 10 s before the start, one began <= 10 s after the end, no gap
    > 10 s in between, and every snapshot from the first to the last succeeded with no foreign process."""
    if not snaps:
        return False, "no snapshot record"
    T0 = [s[0] for s in snaps]
    end = iso_epoch(ledger_iso) + 1.0
    start = iso_epoch(ledger_iso) - float(seconds)
    j = bisect.bisect_right(T0, start) - 1
    while j >= 0 and snaps[j][1] > start:
        j -= 1
    if j < 0 or start - snaps[j][1] > MAX_GAP:
        return False, "no snapshot within 10 s before the start"
    k = bisect.bisect_left(T0, end)
    if k >= len(snaps) or snaps[k][0] - end > MAX_GAP:
        return False, "no snapshot within 10 s after the end"
    seg = snaps[j:k + 1]
    for a, b in zip(seg, seg[1:]):
        if b[0] - a[1] > MAX_GAP:
            return False, "gap > 10 s between snapshots"
    if not all(s[2] for s in seg):
        return False, "foreign compute process or failed query"
    return True, "free"


def ledger_lines(names=None, since_iso=None, ledger_dir=None):
    out = []
    for p in sorted(glob.glob(os.path.join(ledger_dir or C.GPU_LEDGER_DIR, "*.tsv"))):
        w = os.path.basename(p)[:-4]
        if names is not None and w not in names:
            continue
        with open(p) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                try:
                    sec = float(parts[2])
                except ValueError:
                    continue
                if since_iso and parts[0] < since_iso:
                    continue
                out.append((parts[0], w, parts[1], sec))
    out.sort()
    return out


def ledger_by_id():
    d = {}
    for ln in ledger_lines():
        d.setdefault(ln[2], []).append(ln)
    return d


def match_ledger(led, iid, gsec):
    if gsec is None:
        return None, "no gpu_section_s"
    m = [c for c in led.get(iid, []) if abs(c[3] - float(gsec)) <= 0.0015]
    if len(m) != 1:
        return None, f"ledger line not matched ({len(m)} candidates)"
    return m[0], ""


# ============================================================================ speed guard (A1.3)
def is_our_worker(pid: int, phase: str, pool: str):
    try:
        argv = [a.decode(errors="replace") for a in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if a]
    except OSError:
        return False, "not running"
    try:
        cwd = os.path.realpath(f"/proc/{pid}/cwd")
    except OSError:
        return False, "cwd unreadable"
    if not all(x in argv for x in ("worker.py", "--phase", phase, "--pool", pool)):
        return False, "command line differs"
    i = argv.index("--phase")
    if i + 1 >= len(argv) or argv[i + 1] != phase:
        return False, "phase differs"
    if cwd != os.path.realpath(SRC):
        return False, "working directory differs"
    return True, ""


def pause_workers(att: str, start_iso: str, phase: str = "pre", ledger_dir=None):
    """SIGTERM to this attempt's GPU workers only (pid, command line and cwd re-checked); their in-flight GPU
    time (stop time - last ledger time of that worker in the attempt, or spawn time) goes into the ledger."""
    ledger_dir = ledger_dir or C.GPU_LEDGER_DIR
    out = []
    pf = os.path.join(att, f"pids_{phase}")
    if not os.path.exists(pf):
        return out
    with open(pf) as f:
        rows = [ln.split() for ln in f if ln.strip()]
    for parts in rows:
        name, pid, spawn = parts[0], int(parts[1]), float(parts[2]) if len(parts) > 2 else time.time()
        if not name.startswith("gpu"):
            continue
        ok, why = is_our_worker(pid, phase, "gpu")
        if not ok:
            out.append({"worker": name, "pid": pid, "signalled": False, "why": why})
            continue
        wname = f"{name}_{phase}"
        last = spawn
        for ts, _w, _what, _sec in ledger_lines(names={wname}, since_iso=start_iso, ledger_dir=ledger_dir):
            last = max(last, iso_epoch(ts))
        os.kill(pid, signal.SIGTERM)
        tk = time.time()
        charge = max(0.0, tk - last)
        os.makedirs(ledger_dir, exist_ok=True)
        with open(os.path.join(ledger_dir, f"A1_guard_{os.path.basename(att.rstrip('/'))}.tsv"), "a") as lf:
            lf.write(f"{utc_iso(tk)}\tA1_guard_inflight_{wname}\t{charge:.3f}\n")
        out.append({"worker": name, "pid": pid, "signalled": True, "charged_s": round(charge, 3),
                    "since_utc": utc_iso(last)})
    return out


def guard_check(att: str, final: bool, ledger_dir=None, snap_files=None, do_status=True):
    import fcntl
    gj = os.path.join(att, "guard.json")
    if os.path.exists(gj):
        return json.load(open(gj))
    start_iso = open(os.path.join(att, "start_utc")).read().strip()
    with open(os.path.join(att, "guard.lock"), "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(gj):
            return json.load(open(gj))
        lines = [ln for ln in ledger_lines(names={"gpu0_pre", "gpu1_pre"}, since_iso=start_iso, ledger_dir=ledger_dir)
                 if A_PILOT_RE.match(ln[2])]
        if len(lines) < GUARD_N:
            if not final:
                return None
            rec = {"evaluated": False, "n_A_pilot_solves": len(lines), "threshold_s": GUARD_S,
                   "reason": f"fewer than {GUARD_N} A-pilot solves in this attempt", "utc": utc_iso()}
            with open(gj, "w") as f:
                json.dump(rec, f, indent=1)
            if do_status:
                status("resume", f"A1 speed guard (rule A1.3) not evaluated in attempt {os.path.basename(att)}: "
                                 f"{len(lines)} A-pilot solves (< {GUARD_N})")
            return rec
        first = lines[:GUARD_N]
        med = statistics.median([ln[3] for ln in first])
        snaps = load_snapshots(snap_files)
        nfree = sum(classify_window(snaps, ln[0], ln[3])[0] for ln in first)
        pause = med > GUARD_S
        rec = {"evaluated": True, "ids": [ln[2] for ln in first], "gpu_s": [ln[3] for ln in first],
               "median_s": med, "threshold_s": GUARD_S, "n_free_of_10": nfree, "decision": "PAUSE" if pause else "continue",
               "reason": "GPU contention (median above threshold)" if pause else "", "utc": utc_iso()}
        if pause:
            with open(os.path.join(att, "PAUSED"), "w") as f:
                json.dump(rec, f, indent=1)
            rec["stopped_workers"] = pause_workers(att, start_iso, "pre", ledger_dir)
            with open(os.path.join(att, "PAUSED"), "w") as f:
                json.dump(rec, f, indent=1)
        with open(gj, "w") as f:
            json.dump(rec, f, indent=1)
        if do_status:
            if pause:
                st = rec["stopped_workers"]
                status("resume", f"A1 speed guard (rule A1.3) PAUSE in attempt {os.path.basename(att)}: median GPU s of the "
                                 f"first {GUARD_N} resumed A-pilot solves {med:.2f} > {GUARD_S:g} (free under A1.2(a): "
                                 f"{nfree}/{GUARD_N}); reason: GPU contention; SIGTERM to own GPU workers "
                                 f"{[s['pid'] for s in st if s['signalled']]}, in-flight time charged "
                                 f"{sum(s.get('charged_s', 0) for s in st):.1f} s to logs/gpu_ledger/A1_guard_"
                                 f"{os.path.basename(att)}.tsv; no analysis run, no statistic opened")
            else:
                status("resume", f"A1 speed guard (rule A1.3) passed in attempt {os.path.basename(att)}: median GPU s of the "
                                 f"first {GUARD_N} resumed A-pilot solves {med:.2f} <= {GUARD_S:g} (free under A1.2(a): "
                                 f"{nfree}/{GUARD_N}); timings only")
        return rec


def cmd_sidecar(att: str, snap_file=None, max_seconds=None, guard=True) -> int:
    att_id = os.path.basename(att.rstrip("/"))
    snap_file = snap_file or os.path.join(SNAP_DIR, att_id + ".tsv")
    os.makedirs(os.path.dirname(snap_file), exist_ok=True)
    stop_file = os.path.join(att, "sidecar.stop")
    flag = {"stop": False}
    signal.signal(signal.SIGTERM, lambda *_: flag.__setitem__("stop", True))
    t_start = time.time()
    with open(snap_file, "a") as f:
        while not flag["stop"] and not os.path.exists(stop_file):
            t0 = time.time()
            try:
                own = own_pids(att)
                rc, apps = query_smi(own)
                t1 = time.time()
                ents, nfor = [], 0
                for pid, name, mem in apps:
                    o = pid in own
                    nfor += 0 if o else 1
                    ents.append(f"{pid}:{'own' if o else 'foreign'}:{re.sub(r'[^A-Za-z0-9_./-]', '_', name)}:{mem}")
                f.write(f"{t0:.3f}\t{t1:.3f}\t{utc_iso(t0)}\t{rc}\t{len(apps)}\t{nfor}\t{';'.join(ents)}\n")
                f.flush()
                if guard:
                    guard_check(att, final=False)
            except Exception as e:  # noqa: BLE001  (the sidecar must not die; a missing snapshot = not free)
                print(f"{utc_iso()} sidecar error: {e!r}", flush=True)
            if max_seconds and time.time() - t_start >= max_seconds:
                break
            while not flag["stop"] and time.time() < t0 + SNAP_INTERVAL:
                time.sleep(0.1)
    return 0


def cmd_guard_final(att: str) -> int:
    guard_check(att, final=True)
    return 6 if os.path.exists(os.path.join(att, "PAUSED")) else 0


# ============================================================================ G4 inputs (A1.2(b)-(e))
G4_SOURCE_SNIPPETS = (
    'st.info(rid("0", "A", l2, r_, True)).get("gpu_section_s", np.nan)',
    "times[l2] = float(np.nanmedian(ts)) if ts else float(\"nan\")",
    'float(st.info(r)["t_half"].get("time_s", np.nan))',
    "rm_t = float(np.nanmedian(rm_ts)) if rm_ts else lo_t",
)


def check_analyze_source(AN) -> list:
    import inspect
    src = inspect.getsource(AN.phase_gates)
    return [s for s in G4_SOURCE_SNIPPETS if s not in src]


def install_mask(AN, mask_a: set, mask_g: set) -> None:
    orig = AN.Store.info

    def info(self, r, _orig=orig):
        d = _orig(self, r)
        if r in mask_a:
            d = dict(d)
            d["gpu_section_s"] = float("nan")
        elif r in mask_g:
            d = dict(d)
            th = dict(d.get("t_half") or {})
            th["time_s"] = float("nan")
            d["t_half"] = th
        return d

    AN.Store.info = info


def g4_groups(plan, fr):
    g = fr["grid"]
    l0 = g["log2N0"]
    groups = {f"A_pilot_l{l2}": [] for l2 in g["log2N_pilot"]}
    groups[f"G1b_l{l0 - 12}"] = []
    for it in plan["items"]:
        if it["phase"] != "pre":
            continue
        if it["kind"] == "A" and it["arm"] == "0" and it["pilot"] and it["log2N"] in g["log2N_pilot"]:
            groups[f"A_pilot_l{it['log2N']}"].append(it)
        elif it["kind"] == "G1b" and it["log2N"] == l0 - 12:
            groups[f"G1b_l{l0 - 12}"].append(it)
    return groups


def g4_classification(plan, fr, snap_files=None):
    snaps = load_snapshots(snap_files)
    led = ledger_by_id()
    rec, mask_a, mask_g, short = {}, set(), set(), []
    for key, its in g4_groups(plan, fr).items():
        rows = []
        for it in its:
            _d, _pk, pj, pd = item_paths(it)
            if not os.path.exists(pd):
                rows.append({"id": it["id"], "done": False, "free": False, "reason": "not done"})
                continue
            info = C.load_json(pj)
            gsec = info.get("gpu_section_s")
            ln, why = match_ledger(led, it["id"], gsec)
            free, reason = (False, why) if ln is None else classify_window(snaps, ln[0], ln[3])
            t = (info.get("t_half") or {}).get("time_s") if key.startswith("G1b") else gsec
            rows.append({"id": it["id"], "done": True, "free": bool(free), "reason": reason, "gpu_section_s": gsec,
                         "g4_timing_s": t, "ledger_utc": ln[0] if ln else None})
            if not free:
                (mask_g if key.startswith("G1b") else mask_a).add(it["id"])
        planned = len(its)
        need = int(math.ceil(planned / 2.0))
        nfree = sum(r["free"] for r in rows)
        ft = [r["g4_timing_s"] for r in rows if r["free"] and r.get("g4_timing_s") is not None]
        rec[key] = {"planned": planned, "done": sum(r["done"] for r in rows), "free": nfree, "required_free": need,
                    "ok": nfree >= need, "median_free_timing_s": statistics.median(ft) if ft else None, "rows": rows}
        if nfree < need:
            short.append(key)
    return rec, mask_a, mask_g, short


def snapshot_prefix_hashes(files=None):
    import hashlib
    out = {}
    for p in (sorted(glob.glob(os.path.join(SNAP_DIR, "*.tsv"))) if files is None else files):
        data = open(p, "rb").read()
        out[os.path.basename(p)] = {"lines": data.count(b"\n"), "sha256_of_these_lines": hashlib.sha256(data).hexdigest()}
    return out


def wait_after_snapshot(max_wait: float = 60.0) -> float:
    """Wait (<= max_wait s) until the sidecar has a snapshot that began after the end of the latest GPU-ledger
    line, so that the last solves of the pre phase have their 'after' snapshot (A1.2(a)(2)). Returns the wait."""
    t_in = time.time()
    lines = ledger_lines()
    if not lines:
        return 0.0
    e_max = max(iso_epoch(ln[0]) for ln in lines) + 1.0
    while True:
        snaps = load_snapshots()
        if snaps and max(s[0] for s in snaps) >= e_max:
            return time.time() - t_in
        if time.time() - t_in >= max_wait:
            return -1.0
        time.sleep(1.0)


def cmd_gates(att: str) -> int:
    att_id = os.path.basename(att.rstrip("/"))
    if os.path.exists(os.path.join(C.STAGEA_DIR, "eps_arm0.json")):
        print("A1 gates: eps_arm0.json exists, the gates phase was already run (launcher should skip it)")
        return 8
    w = wait_after_snapshot()
    print(f"A1 gates: waited {w:.1f} s for the snapshot after the last GPU section" if w >= 0 else
          "A1 gates: no snapshot after the last GPU section within 60 s (sidecar down?); classifying anyway")
    plan = load_plan()
    fr = C.load_json(C.FROZEN_PRED)
    groups, mask_a, mask_g, short = g4_classification(plan, fr)
    rec = {"version": "S2.3 A1 G4 inputs v1", "rule": "prereg/AMENDMENT_A1_resume.md A1.2", "attempt": att_id,
           "utc": utc_iso(), "groups": groups, "n_masked_A_pilot": len(mask_a), "n_masked_G1b": len(mask_g),
           "snapshot_files": snapshot_prefix_hashes(),
           "classification": {"interval_s": SNAP_INTERVAL, "max_gap_s": MAX_GAP, "window": "ledger time - GPU s .. ledger time + 1 s"}}
    summ = ", ".join(f"{k} {v['free']}/{v['planned']} (need {v['required_free']})" for k, v in groups.items())
    if short:
        p = os.path.join(att, "A1_g4_inputs_insufficient.json")
        with open(p, "w") as f:
            json.dump(rec, f, indent=1)
        with open(os.path.join(att, "PAUSED_G4"), "w") as f:
            f.write(f"{utc_iso()} too few free timings (rule A1.2(d)): {short}\n")
        status("resume", f"A1 G4 NOT EVALUATED (rule A1.2(d)) in attempt {att_id}: too few free timings in {short}; free/planned: "
                         f"{summ}; paused before the gates phase: analyze_A not run, no eps_arm0.json, no statistic "
                         f"opened; record {os.path.relpath(p, C.ROOT)} sha256 {sha(p)}; author decision needed")
        return 7
    out = os.path.join(C.STAGEA_DIR, "A1_g4_inputs.json")
    if os.path.exists(out):
        old = C.load_json(out)
        if json.loads(json.dumps(C._jsonable(rec["groups"]))) != old.get("groups"):
            print("A1 gates: existing A1_g4_inputs.json differs from the recomputed classification; refusing")
            return 8
        h = sha(out)
        print(f"A1 gates: reusing identical results/stageA/A1_g4_inputs.json sha256 {h}")
    else:
        h = C.dump_json(rec, out)
        os.chmod(out, 0o444)
        status("resume", f"A1 G4 inputs written BEFORE the gates phase (rule A1.2): results/stageA/A1_g4_inputs.json sha256 {h}; "
                         f"free/planned: {summ}; masked as not free: {len(mask_a)} A-pilot and {len(mask_g)} G1b "
                         f"timings (the 15 pilots from before A1 have no record); timings only, no statistic")
    import analyze_A as AN  # frozen
    miss = check_analyze_source(AN)
    if miss:
        print("A1 gates: the frozen G4 code does not contain the expected timing lines; refusing:", miss)
        return 8
    install_mask(AN, mask_a, mask_g)
    sys.argv = [os.path.join(SRC, "analyze_A.py"), "--phase", "gates"]
    AN.main()       # frozen: verify_frozen_code, phase_gates, sys.exit(0 main allowed | 3 not allowed)
    return 0


def cmd_classify(snap_files=None) -> int:
    snaps = load_snapshots(snap_files)
    print(f"snapshots: {len(snaps)}")
    rows = [ln for ln in ledger_lines() if A_PILOT_RE.match(ln[2]) or ln[2].startswith("G1b_")]
    for ts, w, what, sec in rows:
        free, why = classify_window(snaps, ts, sec)
        print(f"{ts}\t{w}\t{what}\t{sec:.3f}\t{'free' if free else 'NOT free'}\t{why}")
    print(f"A-pilot/G1b ledger lines: {len(rows)}; free: {sum(classify_window(snaps, r[0], r[3])[0] for r in rows)}")
    return 0


def cmd_remaining() -> int:
    from collections import Counter
    plan = load_plan()
    todo, done = Counter(), Counter()
    per_worker = Counter()
    for ph in ("pre", "main"):
        for pool, nw in (("gpu", 2), ("cpu", 1)):
            its = [it for it in plan["items"] if it["phase"] == ph and pool_of(it) == pool]
            for i, it in enumerate(its):
                key = (ph, pool, it["kind"], it["log2N"])
                if os.path.exists(item_paths(it)[3]):
                    done[key] += 1
                else:
                    todo[key] += 1
                    per_worker[(ph, f"{pool}{i % nw}")] += 1
    keys = sorted(set(todo) | set(done))
    print("phase pool kind log2N done todo")
    for k in keys:
        print(*k, done[k], todo[k])
    for ph in ("pre", "main"):
        print(ph, "todo:", sum(v for k, v in todo.items() if k[0] == ph), "done:", sum(v for k, v in done.items() if k[0] == ph),
              "per worker (frozen split i % nw):", {w: v for (p, w), v in sorted(per_worker.items()) if p == ph})
    return 0


def cmd_smi(fake=None) -> int:
    if fake:
        if not os.path.realpath(fake).startswith(os.path.join(os.path.realpath(S23), "dev") + os.sep):
            print("1 0")
            return 2
    rc, apps = query_smi({}, fake=fake)
    print(rc, len(apps))
    return 0


# ============================================================================ self-tests (dry run)
def cmd_selftest(tmp: str) -> int:
    import numpy as np
    os.makedirs(tmp, exist_ok=True)
    res = []

    def ok(name, cond, detail=""):
        res.append({"test": name, "pass": bool(cond), "detail": str(detail)})

    # 1. classifier
    base = 1_800_000_000.0
    snaps = [(base + 2 * i, base + 2 * i + 0.2, True) for i in range(100)]
    iso = utc_iso(base + 50)
    ok("free window", classify_window(snaps, iso, 10.0)[0])
    s2 = list(snaps)
    s2[22] = (s2[22][0], s2[22][1], False)
    ok("foreign process inside window -> not free", not classify_window(s2, iso, 10.0)[0], classify_window(s2, iso, 10.0)[1])
    ok("no snapshot before start -> not free", not classify_window(snaps[30:], utc_iso(base + 62), 5.0)[0])
    ok("no snapshot after end -> not free", not classify_window(snaps[:20], utc_iso(base + 38), 2.0)[0])
    s3 = snaps[:21] + snaps[27:]
    ok("gap > 10 s -> not free", not classify_window(s3, iso, 10.0)[0], classify_window(s3, iso, 10.0)[1])
    s4 = list(snaps)
    s4[24] = (s4[24][0], s4[24][1], False)
    ok("failed query adjacent -> not free", not classify_window(s4, iso, 10.0)[0])
    ok("foreign process outside window -> free", classify_window(s2, utc_iso(base + 120), 10.0)[0])
    # 2. smi parsing
    ok("smi empty -> 0 procs", parse_smi_text("") == (0, []))
    ok("smi 'No running processes found' -> 0 procs", parse_smi_text("No running processes found\n") == (0, []))
    ok("smi two procs", parse_smi_text("123, /x/python, 2666\n456, y, 10\n")[1] == [(123, "/x/python", "2666"), (456, "y", "10")])
    ok("smi garbage -> failed query", parse_smi_text("NVIDIA-SMI has failed\n")[0] != 0)
    # 3. guard decision + pause on dummy processes (own dummies only)
    att = os.path.join(tmp, "att_guard")
    led = os.path.join(tmp, "ledger")
    os.makedirs(att, exist_ok=True)
    os.makedirs(led, exist_ok=True)
    start = utc_iso(time.time() - 100)
    open(os.path.join(att, "start_utc"), "w").write(start + "\n")
    with open(os.path.join(led, "gpu0_pre.tsv"), "w") as f:
        for i in range(6):
            f.write(f"{utc_iso(time.time() - 90 + i)}\t0_A_l22_r{20 + 2 * i:03d}_pilot\t{45.0 + i:.3f}\n")
    with open(os.path.join(led, "gpu1_pre.tsv"), "w") as f:
        for i in range(5):
            f.write(f"{utc_iso(time.time() - 89 + i)}\t0_A_l22_r{21 + 2 * i:03d}_pilot\t{40.0 + i:.3f}\n")
    dummy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "worker.py", "--phase", "pre",
                              "--pool", "gpu", "--wid", "0", "--nw", "2"], cwd=SRC)
    other = subprocess.Popen(["sleep", "120"])
    with open(os.path.join(att, "pids_pre"), "w") as f:
        f.write(f"gpu0 {dummy.pid} {time.time():.3f}\ngpu1 {other.pid} {time.time():.3f}\n")
    try:
        rec = guard_check(att, final=False, ledger_dir=led, snap_files=[], do_status=False)
        ok("guard evaluated on the first 10 A-pilot solves", rec and rec["evaluated"] and len(rec["ids"]) == 10, rec and rec.get("ids"))
        ok("guard median > 30 s -> PAUSE", rec and rec["decision"] == "PAUSE" and os.path.exists(os.path.join(att, "PAUSED")),
           rec and rec.get("median_s"))
        try:
            dummy.wait(timeout=10)
            killed = True
        except subprocess.TimeoutExpired:
            killed = False
        ok("own dummy GPU worker (worker.py --phase pre --pool gpu, cwd src) received SIGTERM", killed)
        ok("process with another command line NOT signalled", other.poll() is None,
           [s for s in rec.get("stopped_workers", []) if s["worker"] == "gpu1"])
        gl = os.path.join(led, f"A1_guard_{os.path.basename(att)}.tsv")
        ok("in-flight time charged to the (temporary) ledger", os.path.exists(gl) and len(open(gl).read().splitlines()) == 1,
           open(gl).read() if os.path.exists(gl) else "")
        rec2 = guard_check(att, final=True, ledger_dir=led, snap_files=[], do_status=False)
        ok("guard is evaluated once per attempt", rec2 == json.load(open(os.path.join(att, "guard.json"))))
    finally:
        for p in (dummy, other):
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=10)
    att2 = os.path.join(tmp, "att_guard_ok")
    os.makedirs(att2, exist_ok=True)
    open(os.path.join(att2, "start_utc"), "w").write(start + "\n")
    led2 = os.path.join(tmp, "ledger2")
    os.makedirs(led2, exist_ok=True)
    with open(os.path.join(led2, "gpu0_pre.tsv"), "w") as f:
        for i in range(12):
            f.write(f"{utc_iso(time.time() - 90 + i)}\t0_A_l22_r{20 + i:03d}_pilot\t{9.0 + i:.3f}\n")
    rec3 = guard_check(att2, final=False, ledger_dir=led2, snap_files=[], do_status=False)
    ok("guard median <= 30 s -> continue, no PAUSED", rec3 and rec3["decision"] == "continue"
       and not os.path.exists(os.path.join(att2, "PAUSED")), rec3 and rec3.get("median_s"))
    att3 = os.path.join(tmp, "att_guard_few")
    os.makedirs(att3, exist_ok=True)
    open(os.path.join(att3, "start_utc"), "w").write(start + "\n")
    rec4 = guard_check(att3, final=True, ledger_dir=os.path.join(tmp, "empty_ledger"), snap_files=[], do_status=False)
    ok("fewer than 10 A-pilot solves -> guard not evaluated", rec4 and rec4["evaluated"] is False)
    # 4. the G4 mask, against the frozen analyze_A source (import only; no phase is run)
    import analyze_A as AN
    miss = check_analyze_source(AN)
    ok("frozen analyze_A.phase_gates contains the 4 expected G4 timing lines", not miss, miss)

    class FakeStore:
        data = {}

        def info(self, r):
            return FakeStore.data[r]

    FakeStore.data = {f"p{i}": {"gpu_section_s": float(10 + i), "cert_pass": True} for i in range(10)}
    FakeStore.data.update({f"g{i}": {"t_half": {"time_s": float(20 + i), "path": "bm"}} for i in range(4)})
    fake_mod = type("M", (), {"Store": FakeStore})
    install_mask(fake_mod, {"p0", "p1", "p2"}, {"g3"})
    st = FakeStore()
    ts = [st.info(f"p{i}").get("gpu_section_s", np.nan) for i in range(10)]
    ok("masked A-pilot median = median of the free ones", float(np.nanmedian(ts)) == statistics.median(range(13, 20)),
       float(np.nanmedian(ts)))
    rm = [float(st.info(f"g{i}")["t_half"].get("time_s", np.nan)) for i in range(4)]
    ok("masked G1b t/2 median = median of the free ones", float(np.nanmedian(rm)) == 21.0, float(np.nanmedian(rm)))
    ok("mask leaves other fields unchanged", st.info("p0")["cert_pass"] is True and st.info("g3")["t_half"]["path"] == "bm"
       and FakeStore.data["p0"]["gpu_section_s"] == 10.0)
    # 5. partial scan
    runs = os.path.join(tmp, "runs")
    plan = {"items": [{"id": "0_A_l22_r000_pilot", "cell": "0/l22/A_pilot"}, {"id": "0_A_l22_r001_pilot", "cell": "0/l22/A_pilot"},
                      {"id": "G1b_AAp_l22_r000", "cell": "G1b/l22/AAp"}]}
    d = os.path.join(runs, "0/l22/A_pilot")
    os.makedirs(d, exist_ok=True)
    for fn in ("0_A_l22_r000_pilot.npy", "0_A_l22_r000_pilot.json", "0_A_l22_r000_pilot.done",
               "0_A_l22_r001_pilot.npy", "0_A_l22_r001_pilot.json.tmp", "stray.txt"):
        open(os.path.join(d, fn), "w").close()
    d2 = os.path.join(runs, "G1b/l22/AAp")
    os.makedirs(d2, exist_ok=True)
    for fn in ("G1b_AAp_l22_r000.npy", "G1b_AAp_l22_r000.json", "G1b_AAp_l22_r000.done", "G1b_AAp_l22_r000__t_half.npy"):
        open(os.path.join(d2, fn), "w").close()
    os.makedirs(os.path.join(runs, PARTIAL, "old", "0/l22/A_pilot"), exist_ok=True)
    open(os.path.join(runs, PARTIAL, "old", "0/l22/A_pilot", "x.npy"), "w").close()
    got = scan_partials(plan, runs)
    want = sorted(["0/l22/A_pilot/0_A_l22_r001_pilot.npy", "0/l22/A_pilot/0_A_l22_r001_pilot.json.tmp", "0/l22/A_pilot/stray.txt"])
    ok("partial scan: files of unfinished items and strays only (finished items and _partial_ kept)", got == want, got)
    # 6. check_finished on a synthetic finished item
    it = {"id": "0_A_l22_r000_pilot", "cell": "0/l22/A_pilot", "x": 1}
    np.save(os.path.join(d, "0_A_l22_r000_pilot.npy"), np.eye(4))
    C.dump_json({"id": it["id"], "item": it}, os.path.join(d, "0_A_l22_r000_pilot.json"))
    errs, hs = check_finished(it, 4, runs)
    ok("check_finished passes a well-formed item", not errs and len(hs) == 3, errs)
    errs2, _ = check_finished(dict(it, x=2), 4, runs)
    ok("check_finished flags an item record that differs from the plan", bool(errs2), errs2)
    npass = sum(r["pass"] for r in res)
    with open(os.path.join(tmp, "selftest.json"), "w") as f:
        json.dump(res, f, indent=1)
    for r in res:
        print(("PASS " if r["pass"] else "FAIL ") + r["test"] + ("" if r["pass"] else f"  [{r['detail']}]"))
    print(f"A1 selftest: {npass}/{len(res)} passed")
    return 0 if npass == len(res) else 1


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("verify")
    p.add_argument("--mode", choices=["run", "dry", "test"], required=True)
    p = sp.add_parser("manifest")
    p.add_argument("--out", required=True)
    p = sp.add_parser("a1manifest")
    p.add_argument("--out", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--attempt", required=True)
    p.add_argument("--dry-run", action="store_true")
    p = sp.add_parser("sidecar")
    p.add_argument("--attempt", required=True)
    p.add_argument("--snap-file")
    p.add_argument("--max-seconds", type=float)
    p.add_argument("--no-guard", action="store_true")
    p = sp.add_parser("guard")
    p.add_argument("--attempt", required=True)
    p.add_argument("--final", action="store_true", required=True)
    p = sp.add_parser("gates")
    p.add_argument("--attempt", required=True)
    p = sp.add_parser("classify")
    p.add_argument("--snap-file", action="append")
    sp.add_parser("remaining")
    p = sp.add_parser("smi")
    p.add_argument("--fake")
    p = sp.add_parser("selftest")
    p.add_argument("--tmp", required=True)
    p = sp.add_parser("status")
    p.add_argument("role")
    p.add_argument("text")
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        return cmd_verify(a.mode)
    if a.cmd == "manifest":
        return cmd_manifest(a.out)
    if a.cmd == "a1manifest":
        return cmd_a1manifest(a.out)
    if a.cmd == "prep":
        return cmd_prep(a.attempt, a.dry_run)
    if a.cmd == "sidecar":
        return cmd_sidecar(a.attempt, a.snap_file, a.max_seconds, guard=not a.no_guard)
    if a.cmd == "guard":
        return cmd_guard_final(a.attempt)
    if a.cmd == "gates":
        return cmd_gates(a.attempt)
    if a.cmd == "classify":
        return cmd_classify(a.snap_file if a.snap_file else None)
    if a.cmd == "remaining":
        return cmd_remaining()
    if a.cmd == "smi":
        return cmd_smi(a.fake)
    if a.cmd == "selftest":
        return cmd_selftest(a.tmp)
    if a.cmd == "status":
        status(a.role, a.text)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
