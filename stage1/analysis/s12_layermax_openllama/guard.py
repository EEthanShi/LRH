"""Part B guard of the OpenLLaMA axis (F1 of analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md,
sha256 4f18d710beda45492b1e191f74580bfc4ef53f6dafe791ad245ff1cf335f3240, section 4).

Written at step S0b by role oll-axis-extract (2026-10-09). New file; hashed in the S0b CODE FROZEN line.

Layers implemented here
-----------------------
(a) ``require_partb_frozen(out_root)``: the positive gate. Every script that can compute or read an observed
    or permuted cross-model score involving open_llama_13b calls it FIRST, before it opens any score, pair
    file or feature of a cross-model computation. It exits with code 3 unless
      (i)   STATUS.md has a milestone line starting ``<UTC>Z [oll-axis-predict] Part B FROZEN`` (the last such
            line is used; the string quoted inside another line does not count),
            and that line names the sha256 of both Part B files in the form
            ``PARTB_PREDICTIONS_OL.md sha256 <64 hex>`` and ``partB_features_ol.json sha256 <64 hex>``;
      (ii)  both files currently hash to exactly those values;
      (iii) ``out_root`` lies inside one of the guarded output directories of the plan's F1 table
            (results/S1_2_layermax_openllama/{early,indep,review,forecast,joint,cache} or below).
(b) ``scan()``: the negative scan, run and logged at Part B start and at Part B freeze (CLI ``scan``).
(c) ``audit()``: the birth-time audit of step S8 (CLI ``audit``).
(d) The static output-root check is a separate script (f1d_outroots.py).

S0b unit tests
--------------
``test_mode()`` returns True only if the environment variable OL_S0B_TEST=1 is set AND no milestone line
``<UTC>Z [oll-axis-<role>] CODE FROZEN`` exists yet in STATUS.md. If OL_S0B_TEST=1 is set after the CODE FROZEN
line exists, it exits with code 3. In test mode ``require_partb_frozen`` accepts only an ``out_root`` under
~/oll_s0b_scratch/ (outside the project tree and /tmp) and skips the Part B check. Test mode therefore can
only run before any OpenLLaMA-13B byte exists on the server (the download is step S3, after CODE FROZEN),
and it can never write into the results tree of this axis.

CLI
---
  python guard.py check <out_root>     run the gate (exit 0 or 3), print the verdict as JSON
  python guard.py scan                 F1(b) negative scan; exit 0 = clean, 4 = violation
  python guard.py audit                F1(c) birth-time audit; exit 0 = pass, 4 = violation, 3 = no Part B line
"""
from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import subprocess
import sys
import time

PROJ = os.path.realpath(os.path.expanduser("~/hrh_v2_stage1_20261008"))
STATUS = os.path.join(PROJ, "STATUS.md")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_openllama")
RES = os.path.join(PROJ, "results", "S1_2_layermax_openllama")
PARTB_MD = os.path.join(AXIS, "PARTB_PREDICTIONS_OL.md")
PARTB_JSON = os.path.join(RES, "partB", "partB_features_ol.json")
SCRATCH = os.path.realpath(os.path.expanduser("~/oll_s0b_scratch"))

PARTB_TAG = "[oll-axis-predict] Part B FROZEN"
# output roots of the gated scripts (plan section 4, F1 table)
GUARDED_ROOTS = tuple(os.path.join(RES, d) for d in ("early", "indep", "review", "forecast", "joint", "cache"))
# the only entries allowed under RES when Part B is frozen (plan section 4 F1(b), R.3.4)
ALLOWED_AT_PARTB = ("extraction", "feasibility", "logs", "partB")
# command-line fragments of processes that must not run at Part B start/freeze (F1(b))
FORBIDDEN_PROCS = ("s12_layermax_openllama/early", "s12_layermax_openllama/indep", "s12_layermax_openllama/review",
                   "forecast_ol", "joint_summary", "cross_validate")
NEW_SLUG = "openlm-research__open_llama_13b"
EXIT_GATE = 3
EXIT_VIOLATION = 4


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _status_lines() -> list[str]:
    with open(STATUS, encoding="utf-8") as f:
        return f.read().splitlines()


# a milestone line starts with its UTC time and the role tag; mentions of these strings inside other
# lines (e.g. the S0a freeze line, which quotes them) do not count
_TS = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ "
CODE_FROZEN_RE = re.compile(_TS + r"\[oll-axis-[A-Za-z0-9-]+\] CODE FROZEN")
PARTB_RE = re.compile(_TS + re.escape(PARTB_TAG))


def code_frozen_line() -> str | None:
    for line in _status_lines():
        if CODE_FROZEN_RE.match(line):
            return line
    return None


def partb_line() -> str | None:
    hits = [ln for ln in _status_lines() if PARTB_RE.match(ln)]
    return hits[-1] if hits else None


def _fail(msg: str, code: int = EXIT_GATE):
    print(f"[guard] REFUSED (exit {code}): {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def _inside(path: str, root: str) -> bool:
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def test_mode() -> bool:
    """S0b unit-test mode (see module docstring). Exits 3 if requested after the CODE FROZEN line."""
    if os.environ.get("OL_S0B_TEST") != "1":
        return False
    if code_frozen_line() is not None:
        _fail("OL_S0B_TEST=1 is refused: the S0b CODE FROZEN line exists in STATUS.md")
    print("[guard] S0B TEST MODE (no CODE FROZEN line yet; outputs only under ~/oll_s0b_scratch/)",
          file=sys.stderr, flush=True)
    return True


def require_partb_frozen(out_root: str, script: str | None = None) -> dict:
    """F1(a) positive gate. Returns a dict describing the pass; exits 3 otherwise."""
    who = script or (sys.argv[0] if sys.argv else "?")
    if test_mode():
        if not _inside(out_root, SCRATCH):
            _fail(f"{who}: test mode requires out_root under {SCRATCH}, got {out_root}")
        return {"mode": "s0b-test", "out_root": os.path.realpath(out_root)}
    if _inside(out_root, SCRATCH):
        _fail(f"{who}: out_root {out_root} is the S0b scratch root but OL_S0B_TEST is not set")
    line = partb_line()
    if line is None:
        _fail(f"{who}: no '{PARTB_TAG}' line in STATUS.md")
    m_md = re.search(r"PARTB_PREDICTIONS_OL\.md sha256 ([0-9a-f]{64})", line)
    m_js = re.search(r"partB_features_ol\.json sha256 ([0-9a-f]{64})", line)
    if not (m_md and m_js):
        _fail(f"{who}: the Part B FROZEN line does not name both sha256 values in the required form")
    for path, want in ((PARTB_MD, m_md.group(1)), (PARTB_JSON, m_js.group(1))):
        if not os.path.isfile(path):
            _fail(f"{who}: {path} missing")
        got = _sha(path)
        if got != want:
            _fail(f"{who}: {os.path.basename(path)} sha256 {got} != STATUS {want}")
    if not any(_inside(out_root, r) for r in GUARDED_ROOTS):
        _fail(f"{who}: out_root {out_root} is not a guarded output directory {GUARDED_ROOTS}")
    info = {"mode": "partB-frozen", "out_root": os.path.realpath(out_root),
            "partB_md_sha256": m_md.group(1), "partB_json_sha256": m_js.group(1),
            "status_line_utc": line.split(" ", 1)[0]}
    print(f"[guard] PASS {who}: Part B frozen ({info['status_line_utc']}), out_root {info['out_root']}",
          file=sys.stderr, flush=True)
    return info


# ----------------------------------------------------------------------------------------------
# F1(b) negative scan and F1(c) birth-time audit
# ----------------------------------------------------------------------------------------------

def _stat_rows(root: str) -> list[dict]:
    """Every entry under root with type (f/d/l/o), size and statx birth time. GNU find 4.9 on ds2213 has no
    birth-time directive (its %W is unrecognised), so the birth time comes from coreutils `stat -c %W`
    (seconds since the epoch; 0 = unknown, recorded as birth_epoch None)."""
    rows = []
    if not os.path.isdir(root):
        return rows
    paths = [p for p in subprocess.run(["find", root, "-print0"], capture_output=True).stdout.decode().split("\0") if p]
    for i in range(0, len(paths), 400):
        out = subprocess.run(["stat", "--printf", "%F|%s|%W|%n\n"] + paths[i:i + 400], capture_output=True,
                             text=True).stdout
        for ln in out.splitlines():
            t, s, w, p = ln.split("|", 3)
            typ = "f" if "regular" in t else ("d" if t == "directory" else ("l" if "link" in t else "o"))
            we = float(w) if float(w) > 0 else None
            rows.append({"type": typ, "size": int(s), "birth_epoch": we,
                         "birth_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(we)) if we else None,
                         "path": os.path.relpath(p, PROJ)})
    return rows


def scan() -> dict:
    res = subprocess.run(["find", PROJ, "/tmp", "-name", "*open_llama_13b__*"], capture_output=True, text=True)
    hits = [h for h in res.stdout.splitlines() if h.strip()]
    top = sorted(os.listdir(RES)) if os.path.isdir(RES) else []
    bad_top = [d for d in top if d not in ALLOWED_AT_PARTB]
    ps = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()[1:]
    me = str(os.getpid())
    procs = [ln.strip() for ln in ps if any(f in ln for f in FORBIDDEN_PROCS) and not ln.strip().startswith(me)
             and "guard.py" not in ln]
    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "find_cmd": f"find {PROJ} /tmp -name '*open_llama_13b__*'", "find_hits": hits,
           "find_stderr_lines": len(res.stderr.splitlines()),
           "results_top_level": top, "results_top_level_not_allowed": bad_top,
           "forbidden_processes": procs, "results_tree": _stat_rows(RES)}
    rep["clean"] = not hits and not bad_top and not procs
    return rep


def audit() -> dict:
    line = partb_line()
    if line is None:
        _fail("no Part B FROZEN line in STATUS.md")
    t_utc = line.split(" ", 1)[0]
    t_epoch = calendar.timegm(time.strptime(t_utc, "%Y-%m-%dT%H:%M:%SZ"))
    rows = []
    for d in ("cache", "early", "indep", "review", "forecast", "joint"):
        rows += [r for r in _stat_rows(os.path.join(RES, d)) if r["type"] == "f"]
    unknown = [r for r in rows if r["birth_epoch"] is None]  # a missing birth time cannot be verified: fails
    early = [r for r in rows if r["birth_epoch"] is not None and r["birth_epoch"] <= t_epoch]
    known = [r for r in rows if r["birth_epoch"] is not None]
    mn = min(known, key=lambda r: r["birth_epoch"]) if known else None
    return {"partB_status_utc": t_utc, "n_files": len(rows), "min_birth": mn,
            "n_born_before_partB": len(early), "born_before_partB": early[:50],
            "n_birth_unknown": len(unknown), "birth_unknown": unknown[:50], "pass": not early and not unknown}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == "check":
        print(json.dumps(require_partb_frozen(sys.argv[2], script="guard.py check"), indent=1))
    elif cmd == "scan":
        rep = scan()
        print(json.dumps(rep, indent=1))
        sys.exit(0 if rep["clean"] else EXIT_VIOLATION)
    elif cmd == "audit":
        rep = audit()
        print(json.dumps(rep, indent=1))
        sys.exit(0 if rep["pass"] else EXIT_VIOLATION)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
