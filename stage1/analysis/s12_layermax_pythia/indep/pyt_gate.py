"""Pythia axis, implementation I: output root and order gate (PYT-AXIS COPY of
analysis/s12_layermax_openllama/indep/ol_gate.py, sha256 2a55dce4..., sub-role pyt-code-I, S0b, 2026-10-10).

Every I script that can compute or read an observed or permuted cross-model score involving a Pythia model calls
``gate()`` as its FIRST action (plan ANALYSIS_PLAN_PYTHIA.md section 4 F1(a)), before it imports the data modules or
opens any pair file, score or cross-model input. ``gate()`` returns the output root of implementation I:

  * normal run:   ~/hrh_v2_stage1_20261008/results/S1_2_layermax_pythia/indep                       # PYT-AXIS
                  The gate exits 3 (no output written) unless ALL of the following hold:              # PYT-AXIS
                  (i)   STATUS.md has a milestone line "<UTC>Z [pyt-axis-<role>] CODE FROZEN" (plan S0b);
                  (ii)  STATUS.md has a later milestone line "<UTC>Z [pyt-axis-predict] Part B FROZEN" (plan S5,
                        section 9 B7; the last such line is used; the string quoted inside another line does not count);
                  (iii) that line names the sha256 of PARTB_PREDICTIONS_PYT.md and of partB_features_pyt.json (form
                        "<name> sha256 <64 hex>", punctuation between name, 'sha256' and the hex tolerated), and both
                        files currently hash to exactly those values;
                  (iv)  the output root lies inside one of the guarded directories of plan F1
                        (results/S1_2_layermax_pythia/{cache,early,indep,review,forecast} or below);
                  (v)   if the shared guard analysis/s12_layermax_pythia/guard.py exists (written by another role), its
                        require_partb_frozen(out_root) passes too (it exits 3 itself otherwise). If it is absent, this
                        file's own check (i)-(iv) is the whole gate. (Implementation I never reads the shared guard's
                        source; it only calls it.)
  * S0b test mode (PYT_S0B_TEST=1, the axis-wide test switch of pyt_axis_common.py and guard.py, AND no CODE FROZEN
                  line yet): $PYT_I_TEST_OUT (default analysis/s12_layermax_pythia/s0b_scratch/I/out), which must lie
                  under analysis/s12_layermax_pythia/s0b_scratch/I/ (VERDICTS_SPEC_PYT.md section 13: the shared scratch
                  root, sub-root I; inside the plan's four trees, never under results/). The Part B check is skipped; the
                  shared guard, if present, is called too (it accepts only roots under s0b_scratch/).
                  PYT_S0B_TEST=1 after the CODE FROZEN line exists -> exit 3.

``-h`` / ``--help`` on the command line skips the gate: argparse then prints the usage and exits before any data is read.
"""
from __future__ import annotations

import hashlib  # PYT-AXIS
import os
import re  # PYT-AXIS
import sys

sys.dont_write_bytecode = True   # PYT-AXIS: a refused dry run leaves no file behind (not even __pycache__)

PROJ = os.path.realpath(os.path.expanduser("~/hrh_v2_stage1_20261008"))   # PYT-AXIS
STATUS = os.path.join(PROJ, "STATUS.md")   # PYT-AXIS
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_pythia")   # PYT-AXIS
HERE = os.path.join(AXIS, "indep")   # PYT-AXIS
RES_PYT = os.path.join(PROJ, "results", "S1_2_layermax_pythia")   # PYT-AXIS
OUT_MAIN = os.path.join(RES_PYT, "indep")   # PYT-AXIS
TEST_ROOT = os.path.join(AXIS, "s0b_scratch", "I")   # PYT-AXIS: S0b test outputs (VERDICTS_SPEC_PYT 13: sub-root I)
PARTB_MD = os.path.join(AXIS, "PARTB_PREDICTIONS_PYT.md")   # PYT-AXIS (plan section 9 B7)
PARTB_JSON = os.path.join(RES_PYT, "partB", "partB_features_pyt.json")   # PYT-AXIS
PARTB_TAG = "[pyt-axis-predict] Part B FROZEN"   # PYT-AXIS (plan section 4 F1)
GUARDED_ROOTS = tuple(os.path.join(RES_PYT, d) for d in ("cache", "early", "indep", "review", "forecast"))  # PYT-AXIS
EXIT_GATE = 3

_TS = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ "   # PYT-AXIS: a milestone line starts with its UTC time and the role tag
CODE_FROZEN_RE = re.compile(_TS + r"\[pyt-axis-[A-Za-z0-9-]+\] CODE FROZEN")   # PYT-AXIS
PARTB_RE = re.compile(_TS + re.escape(PARTB_TAG))   # PYT-AXIS
_HEX = r"[^0-9a-f\n]{0,6}sha256[^0-9a-f\n]{0,6}([0-9a-f]{64})"   # PYT-AXIS
MD_SHA_RE = re.compile(r"PARTB_PREDICTIONS_PYT\.md" + _HEX)   # PYT-AXIS
JS_SHA_RE = re.compile(r"partB_features_pyt\.json" + _HEX)   # PYT-AXIS

_TEST = None


def _fail(msg: str):   # PYT-AXIS
    print(f"[pyt_gate I] REFUSED (exit {EXIT_GATE}): {msg}", file=sys.stderr, flush=True)
    sys.exit(EXIT_GATE)


def _status_lines() -> list[str]:   # PYT-AXIS
    try:
        with open(STATUS, encoding="utf-8") as f:
            return f.read().splitlines()
    except OSError as e:
        _fail(f"cannot read {STATUS}: {e}")


def _sha(path: str) -> str:   # PYT-AXIS
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inside(path: str, root: str) -> bool:   # PYT-AXIS
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def code_frozen_index(lines=None):   # PYT-AXIS
    lines = _status_lines() if lines is None else lines
    for i, ln in enumerate(lines):
        if CODE_FROZEN_RE.match(ln):
            return i
    return None


def partb_index(lines=None):   # PYT-AXIS
    lines = _status_lines() if lines is None else lines
    hits = [i for i, ln in enumerate(lines) if PARTB_RE.match(ln)]
    return hits[-1] if hits else None


def test_mode() -> bool:
    """PYT_S0B_TEST=1 and no CODE FROZEN line yet (evaluated once per process). Exits 3 if PYT_S0B_TEST=1 is set
    after the CODE FROZEN line exists."""   # PYT-AXIS
    global _TEST
    if _TEST is None:
        if os.environ.get("PYT_S0B_TEST") != "1":
            _TEST = False
        elif code_frozen_index() is not None:
            _fail("PYT_S0B_TEST=1 is refused: the S0b CODE FROZEN line exists in STATUS.md")
        else:
            print("[pyt_gate I] S0B TEST MODE (no CODE FROZEN line yet; outputs only under "
                  f"{TEST_ROOT})", file=sys.stderr, flush=True)
            _TEST = True
    return _TEST


def out_root() -> str:
    if test_mode():   # PYT-AXIS
        return os.path.realpath(os.environ.get("PYT_I_TEST_OUT", os.path.join(TEST_ROOT, "out")))
    return OUT_MAIN


def help_requested() -> bool:
    return any(a in ("-h", "--help") for a in sys.argv[1:])


def _shared(root: str, script: str) -> bool:   # PYT-AXIS: (v) the shared guard (VERDICTS_SPEC_PYT 13), if present
    shared = os.path.join(AXIS, "guard.py")
    if not os.path.isfile(shared):
        return False
    if AXIS not in sys.path:
        sys.path.insert(0, AXIS)
    try:
        import guard as _shared_guard   # noqa: E402
        fn = getattr(_shared_guard, "require_partb_frozen", None)
        if fn is None:
            _fail(f"{script}: shared guard {shared} has no require_partb_frozen()")
        try:
            fn(root, script=script)
        except TypeError:
            fn(root)
    except SystemExit as e:
        if e.code not in (0, None):
            _fail(f"{script}: shared guard refused (exit {e.code})")
    except Exception as e:   # noqa: BLE001
        _fail(f"{script}: shared guard failed: {type(e).__name__}: {e}")
    return True


def require(root: str, script: str) -> dict:   # PYT-AXIS: checks (i)-(v) of the module docstring
    if test_mode():
        if not _inside(root, TEST_ROOT):
            _fail(f"{script}: test mode requires an output root under {TEST_ROOT}, got {root}")
        return {"mode": "s0b-test", "out_root": os.path.realpath(root), "shared_guard_called": _shared(root, script)}
    if _inside(root, TEST_ROOT):
        _fail(f"{script}: output root {root} is the S0b test root but test mode is off")
    lines = _status_lines()
    icf = code_frozen_index(lines)
    if icf is None:
        _fail(f"{script}: no '[pyt-axis-<role>] CODE FROZEN' milestone line in STATUS.md (plan S0b)")
    ipb = partb_index(lines)
    if ipb is None:
        _fail(f"{script}: no '{PARTB_TAG}' milestone line in STATUS.md (plan S5)")
    if ipb < icf:
        _fail(f"{script}: the Part B FROZEN line precedes the CODE FROZEN line (plan order S0b -> S5)")
    line = lines[ipb]
    m_md, m_js = MD_SHA_RE.search(line), JS_SHA_RE.search(line)
    if not (m_md and m_js):
        _fail(f"{script}: the Part B FROZEN line does not name both sha256 values (PARTB_PREDICTIONS_PYT.md, "
              "partB_features_pyt.json)")
    for path, want in ((PARTB_MD, m_md.group(1)), (PARTB_JSON, m_js.group(1))):
        if not os.path.isfile(path):
            _fail(f"{script}: {path} missing")
        got = _sha(path)
        if got != want:
            _fail(f"{script}: {os.path.basename(path)} sha256 {got} != STATUS {want}")
    if not any(_inside(root, r) for r in GUARDED_ROOTS):
        _fail(f"{script}: output root {root} is not a guarded output directory {GUARDED_ROOTS}")
    shared_used = _shared(root, script)   # (v)
    info = {"mode": "partB-frozen", "out_root": os.path.realpath(root),
            "code_frozen_utc": lines[icf].split(" ", 1)[0], "partB_status_utc": line.split(" ", 1)[0],
            "partB_md_sha256": m_md.group(1), "partB_json_sha256": m_js.group(1), "shared_guard_called": shared_used}
    print(f"[pyt_gate I] PASS {script}: code frozen {info['code_frozen_utc']}, Part B frozen "
          f"{info['partB_status_utc']}, out_root {info['out_root']}", file=sys.stderr, flush=True)
    return info


def gate(script: str) -> str:
    """F1(a) positive gate; returns the output root. Exits 3 unless the plan's gates hold."""
    root = out_root()
    if not help_requested():
        require(root, os.path.basename(script))   # PYT-AXIS
    return root


if __name__ == "__main__":   # PYT-AXIS: python pyt_gate.py check  -> runs the gate for the main output root
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.argv = sys.argv[:1]
        print(gate("pyt_gate.py check"))
    else:
        print(__doc__)
        sys.exit(2)
