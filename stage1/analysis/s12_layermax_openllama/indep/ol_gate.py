"""OpenLLaMA axis, implementation I: output root and Part B gate (NEW file, S0b, sub-role I).

Every I script that can compute or read an observed or permuted cross-model score involving
open_llama_13b calls ``gate()`` as its FIRST action (plan section 4 F1(a)), before it imports
the data modules or opens any pair file, score or cross-model input. ``gate()`` returns the
output root of implementation I:

  * normal run:   ~/hrh_v2_stage1_20261008/results/S1_2_layermax_openllama/indep
                  (guard.require_partb_frozen exits 3 unless the '[oll-axis-predict] Part B FROZEN'
                  STATUS line exists and both Part B files hash to its values)
  * S0b test mode (guard.test_mode(): OL_S0B_TEST=1 and no CODE FROZEN line yet):
                  $OL_I_TEST_OUT (default ~/oll_s0b_scratch/I/out), which the guard requires to lie
                  under ~/oll_s0b_scratch/

``-h`` / ``--help`` on the command line skips the gate: argparse then prints the usage and exits
before any data is read.
"""
from __future__ import annotations

import os
import sys

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_openllama")
RES_OL = os.path.join(PROJ, "results", "S1_2_layermax_openllama")
OUT_MAIN = os.path.join(RES_OL, "indep")
SCRATCH_I = os.path.join(os.path.expanduser("~/oll_s0b_scratch"), "I")
if AXIS not in sys.path:
    sys.path.insert(0, AXIS)
import guard  # noqa: E402  (analysis/s12_layermax_openllama/guard.py, read-only import)

_TEST = None


def test_mode() -> bool:
    """guard.test_mode(), evaluated once per process (exits 3 if OL_S0B_TEST=1 after CODE FROZEN)."""
    global _TEST
    if _TEST is None:
        _TEST = bool(guard.test_mode())
    return _TEST


def out_root() -> str:
    if test_mode():
        return os.path.realpath(os.environ.get("OL_I_TEST_OUT", os.path.join(SCRATCH_I, "out")))
    return OUT_MAIN


def help_requested() -> bool:
    return any(a in ("-h", "--help") for a in sys.argv[1:])


def gate(script: str) -> str:
    """F1(a) positive gate; returns the output root. Exits 3 if Part B is not frozen."""
    root = out_root()
    if not help_requested():
        guard.require_partb_frozen(root, script=os.path.basename(script))
    return root
