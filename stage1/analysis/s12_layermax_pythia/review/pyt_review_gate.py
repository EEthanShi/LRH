"""Pythia axis, sub-role R: the order gate of the review recomputation (NEW FILE, PYT-AXIS).

Written at S0b by role pyt-code-R (2026-10-10), before any Pythia byte existed on ds2213; to be hashed in the one S0b
CODE FROZEN line. Plan: analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md (sha256 6de89603...), section 4 F1(a),
section 11 (order plan -> code -> G0 -> S3 GO -> download/extract -> Part B -> runs -> review); contract
VERDICTS_SPEC_PYT.md section 13 ("every gated script calls guard.require_partb_frozen(out_root) first").

require(out_root, script) is the first action of every R entry point; it writes nothing and exits 3 unless
  (1) the shared guard analysis/s12_layermax_pythia/guard.py exists and its require_partb_frozen(out_root) passes
      (Part B FROZEN milestone line naming both Part B sha256 values, both files hashing to them, out_root inside a
      guarded root; in S0b test mode: PYT_S0B_TEST=1 before CODE FROZEN and out_root under guard.SCRATCH);
  and, in production (not test mode), R's additional order checks:
  (2) the frozen plan still hashes to 6de89603...;
  (3) a "[pyt-axis-<role>] CODE FROZEN" milestone line exists and precedes the Part B FROZEN line;
  (4) out_root lies in results/S1_2_layermax_pythia/review/ (R's own guarded root).
  In test mode R additionally requires out_root under guard.SCRATCH/R/ (R's sub-root, contract section 13).
audit(): F1(c) birth-time audit; the shared guard's audit() when present.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys

PROJ = os.path.realpath(os.path.expanduser("~/hrh_v2_stage1_20261008"))
STATUS = os.path.join(PROJ, "STATUS.md")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_pythia")
RES = os.path.join(PROJ, "results", "S1_2_layermax_pythia")
PLAN = os.path.join(AXIS, "ANALYSIS_PLAN_PYTHIA.md")
PLAN_SHA256 = "6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b"
REVIEW_OUT = os.path.join(RES, "review")
SHARED_GUARD = os.path.join(AXIS, "guard.py")
EXIT_GATE = 3
_TS = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ "
CODE_FROZEN_RE = re.compile(_TS + r"\[pyt-axis-[A-Za-z0-9-]+\] CODE FROZEN")
PARTB_RE = re.compile(_TS + re.escape("[pyt-axis-predict] Part B FROZEN"))


def _fail(msg: str):
    print(f"[pyt-review-gate] REFUSED (exit {EXIT_GATE}): {msg}", file=sys.stderr, flush=True)
    sys.exit(EXIT_GATE)


def _guard():
    if not os.path.isfile(SHARED_GUARD):
        _fail(f"the shared guard {SHARED_GUARD} is missing (contract section 13)")
    spec = importlib.util.spec_from_file_location("pyt_shared_guard", SHARED_GUARD)
    g = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(g)
    return g


GUARD = None


def guard():
    global GUARD
    if GUARD is None:
        GUARD = _guard()
    return GUARD


def test_root() -> str:
    return os.path.join(guard().SCRATCH, "R")


def _inside(path: str, root: str) -> bool:
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def test_mode() -> bool:
    """True iff PYT_S0B_TEST=1 and no CODE FROZEN line (the shared guard exits 3 if set after CODE FROZEN)."""
    return bool(guard().test_mode())


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(out_root: str, script: str = "?") -> dict:
    g = guard()
    if g.test_mode():
        if not _inside(out_root, test_root()):
            _fail(f"{script}: S0b test mode requires out_root under {test_root()}, got {out_root}")
        info = g.require_partb_frozen(out_root, script=script)
        return dict(info, r_gate="s0b-test")
    if not _inside(out_root, REVIEW_OUT):
        _fail(f"{script}: out_root {out_root} is not R's guarded output root {REVIEW_OUT}")
    if not os.path.isfile(PLAN) or _sha(PLAN) != PLAN_SHA256:
        _fail(f"{script}: the frozen plan {PLAN} is missing or changed")
    lines = open(STATUS, encoding="utf-8").read().splitlines()
    ic = next((i for i, ln in enumerate(lines) if CODE_FROZEN_RE.match(ln)), None)
    ip = max((i for i, ln in enumerate(lines) if PARTB_RE.match(ln)), default=None)
    if ic is None:
        _fail(f"{script}: no '[pyt-axis-<role>] CODE FROZEN' milestone line in STATUS.md (plan section 11 S0b)")
    if ip is not None and ip < ic:
        _fail(f"{script}: the Part B FROZEN line precedes the CODE FROZEN line (freeze order)")
    info = g.require_partb_frozen(out_root, script=script)       # exits 3 itself on refusal
    return dict(info, r_gate="partB-frozen", code_frozen_utc=lines[ic].split(" ", 1)[0], plan_sha256=PLAN_SHA256)


def audit() -> dict:
    g = guard()
    rep = g.audit()
    rep["source"] = "guard.audit (shared)"
    return rep
