#!/usr/bin/env python3
"""S2.3 AMENDMENT A3 analysis entry point (prereg/AMENDMENT_A3_analysis_memo.md, rules A3.1-A3.2).

Lives in s23/resume/, NEVER in src/. It runs the FROZEN src/analyze_A.py, imported unchanged (file and hash
untouched; its main() re-verifies the 37 frozen code hashes), with ONE difference inside this process only:
the in-memory memo Store.K is replaced by a corrected copy.

The frozen Store.K inserts the array it loads into self._K; when the memo then holds more than 1500 entries it
clears the whole memo -- including the array it has just inserted -- and then returns self._K[key], which raises
KeyError. So any frozen analysis process crashes on its 1501st distinct K load. The corrected memo returns the
array it loaded (np.load of the same path, or None when the file does not exist), keeps the same 1500-entry
bound and the same clear-all eviction, and re-inserts the current entry after a clear. No computed value can
change: every caller receives exactly what the frozen code means to return; only the KeyError is removed.

  python a3_analyze.py --phase nmatch|main     real run (A3 launcher only; cwd src/, frozen phase semantics)
  python a3_analyze.py --selftest DIR           excluded unit test (R29): DIR under s23/dev/A3_tests_*/
                                                (synthetic arrays only; no Stage A data, no STATUS line)
Test roots only (S23_ROOT set, under s23/dev/): A3_TEST_CAP=<n> lowers the memo bound to force evictions.
"""
from __future__ import annotations

import os
import sys

HOME = os.path.expanduser("~")
S23 = os.path.join(HOME, "hrh_v2_stage2_20261008", "s23")
SRC = os.path.join(S23, "src")
DEV = os.path.join(os.path.realpath(S23), "dev") + os.sep
if "S23_ROOT" in os.environ and not os.path.realpath(os.environ["S23_ROOT"]).startswith(DEV):
    raise SystemExit("A3 analyze: a test root must lie under s23/dev/")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import s23common as C  # noqa: E402  (frozen)

C.cap_threads(1)  # as the frozen analyze_A does under __main__, before numpy is imported

import numpy as np  # noqa: E402

import analyze_A as A  # noqa: E402  (frozen src/analyze_A.py, imported unchanged)

if os.path.realpath(A.__file__) != os.path.realpath(os.path.join(SRC, "analyze_A.py")) or A.C is not C:
    raise SystemExit("A3 analyze: did not import the frozen src/analyze_A.py / s23common")

FROZEN_CAP = 1500
CAP = FROZEN_CAP
if "A3_TEST_CAP" in os.environ:
    if not C.TEST_MODE:
        raise SystemExit("A3 analyze: A3_TEST_CAP is allowed on TEST roots only")
    CAP = int(os.environ["A3_TEST_CAP"])
FROZEN_K = A.Store.K
_MISS = object()


def K_memo(self, r, extra=None):
    """Corrected Store.K (rule A3.1): same path, same np.load, same None for a missing file, same bound."""
    key = (r, extra)
    v = self._K.get(key, _MISS)
    if v is _MISS:
        pk, _, _, d = self.paths(r)
        p = pk if extra is None else os.path.join(d, f"{r}__{extra}.npy")
        v = np.load(p) if os.path.exists(p) else None
        self._K[key] = v
        if len(self._K) > CAP:
            self._K.clear()
            self._K[key] = v
    return v


class _FakeStore:
    """Synthetic store for the unit test: paths() like the frozen Store, files under a dev directory."""

    def __init__(self, d):
        self.d = d
        self._K = {}

    def paths(self, r):
        return (os.path.join(self.d, r + ".npy"), os.path.join(self.d, r + ".json"),
                os.path.join(self.d, r + ".done"), self.d)


def selftest(d: str) -> int:
    d = os.path.realpath(d)
    if not d.startswith(os.path.join(os.path.realpath(S23), "dev", "A3_tests_")):
        raise SystemExit("A3 selftest: directory must lie under s23/dev/A3_tests_*")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(20261010)
    n = 1700
    ids = [f"x_r{i:04d}" for i in range(n)]
    for r in ids:
        np.save(os.path.join(d, r + ".npy"), rng.standard_normal((4, 4)))
    np.save(os.path.join(d, ids[0] + "__t_half.npy"), rng.standard_normal((4, 4)))
    fails = 0

    def check(ok, msg):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + msg, flush=True)
        fails += 0 if ok else 1

    # 1. the frozen memo raises KeyError on the 1501st distinct load in a process
    st = _FakeStore(d)
    crash_at = None
    for i, r in enumerate(ids):
        try:
            FROZEN_K(st, r)
        except KeyError:
            crash_at = i + 1
            break
    check(crash_at == FROZEN_CAP + 1, f"frozen Store.K raises KeyError at distinct load #{crash_at} (expected {FROZEN_CAP + 1})")
    # 2. the corrected memo returns np.load of the same path for every id, twice (hits and post-eviction reloads)
    st = _FakeStore(d)
    ok, maxlen = True, 0
    for rnd in range(2):
        for r in ids:
            v = K_memo(st, r)
            ok &= v is not None and np.array_equal(v, np.load(os.path.join(d, r + ".npy")))
            maxlen = max(maxlen, len(st._K))
    check(ok, f"corrected memo returns the loaded array for {n} ids x 2 rounds")
    check(maxlen <= CAP, f"memo size stays <= {CAP} (max seen {maxlen})")
    # 3. the first 1500 loads agree with what the frozen memo returns before it crashes
    st_f, st_c = _FakeStore(d), _FakeStore(d)
    same = all(np.array_equal(FROZEN_K(st_f, r), K_memo(st_c, r)) for r in ids[:FROZEN_CAP])
    check(same, "first 1500 loads: frozen and corrected memo return equal arrays")
    # 4. extra suffix and missing file
    st = _FakeStore(d)
    v = K_memo(st, ids[0], "t_half")
    check(v is not None and np.array_equal(v, np.load(os.path.join(d, ids[0] + "__t_half.npy"))), "extra='t_half' path")
    check(K_memo(st, "x_missing") is None and K_memo(st, ids[1], "nope") is None, "missing file -> None (as frozen)")
    print(f"A3 selftest: {fails} failures", flush=True)
    return 1 if fails else 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        return selftest(sys.argv[2])
    if len(sys.argv) != 3 or sys.argv[1] != "--phase" or sys.argv[2] not in ("nmatch", "main"):
        raise SystemExit("usage: a3_analyze.py --phase nmatch|main | --selftest DIR")
    if os.path.realpath(os.getcwd()) != os.path.realpath(SRC):
        raise SystemExit("A3 analyze: run from s23/src (the frozen command lines' cwd)")
    A.Store.K = K_memo  # rule A3.1: in this process only; the frozen file is unchanged
    print(f"{C.utcnow()} A3 analyze --phase {sys.argv[2]}: frozen src/analyze_A.py, Store.K replaced by the corrected "
          f"memo (bound {CAP}{'' if CAP == FROZEN_CAP else ', TEST ROOT'}); test mode {C.TEST_MODE}", flush=True)
    sys.argv = [os.path.join(SRC, "analyze_A.py"), "--phase", sys.argv[2]]
    A.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
