"""S2.3 shared helpers: paths, frozen constants, keyed seeds, hashing, STATUS, JSON, GPU ledger.

PREREG_S2_3_v1.0.md sections 8.1, 12, 13. Entry scripts call ``cap_threads()`` BEFORE importing
numpy / torch (it only sets environment variables).

Test isolation: unit tests set S23_ROOT (a temporary directory) and S23_KEY (a key prefix starting
with 'hrhv2-s23-TEST'). Then every output, the STATUS file and every random key are redirected; the
real STATUS.md and the real keys 'hrhv2-s23|...' are never touched by a test.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys

HOME = os.path.expanduser("~")
SRC = os.path.dirname(os.path.abspath(__file__))
S23_DIR = os.path.dirname(SRC)                                    # .../hrh_v2_stage2_20261008/s23
ROOT = os.environ.get("S23_ROOT", S23_DIR)                         # outputs (tests: a temp dir)
TEST_MODE = "S23_ROOT" in os.environ
STAGE1 = os.path.join(HOME, "hrh_v2_stage1_20261008")
STAGE2_SRC = os.path.join(HOME, "hrh_v2_stage2_20261008", "src")
STATUS = os.path.join(ROOT, "STATUS_test.md") if TEST_MODE else os.path.join(STAGE1, "STATUS.md")
PREREG_DIR = os.path.join(S23_DIR, "prereg")
RESULTS = os.path.join(ROOT, "results")
RUNS = os.path.join(ROOT, "runs")
PLANS = os.path.join(ROOT, "plans")
LOGS = os.path.join(ROOT, "logs")
PRESTEP_DIR = os.path.join(RESULTS, "prestep")
FROZEN_PRED = os.path.join(PRESTEP_DIR, "frozen_predictions.json")
PRESTEP_ARR = os.path.join(PRESTEP_DIR, "prestep_arrays.npz")
STAGEA_DIR = os.path.join(RESULTS, "stageA")
GPU_LEDGER_DIR = os.path.join(LOGS, "gpu_ledger")

KEY = os.environ.get("S23_KEY", "hrhv2-s23")                       # every random key starts with this
if TEST_MODE and not KEY.startswith("hrhv2-s23-TEST"):
    raise RuntimeError("test mode requires S23_KEY starting with 'hrhv2-s23-TEST'")
if (not TEST_MODE) and KEY != "hrhv2-s23":
    raise RuntimeError("a non-default key prefix is only allowed in test mode")
GEOM_KEY = KEY + "|geometry"                                       # alt keys: GEOM_KEY + "|alt=k"
T_DATA = 0.3                                                       # data temperature t
GPU_HARD_STOP_H = 12.0                                             # author decision 2026-10-09 (Stage A)
GPU_SOFT_STOP_H = 11.5                                             # no new GPU task is started beyond this

# hrhv2 (read-only) is imported from STAGE1; Stage 2's src is NOT put on sys.path (its gates.py,
# make_plan.py, analyze.py would shadow ours): crit.py is loaded by file path (load_crit).
if STAGE1 not in sys.path:
    sys.path.insert(0, STAGE1)
if SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)


def load_crit():
    """Stage 2 crit.py (read-only, hashed) as module 'crit', loaded by path."""
    import importlib.util
    if "crit" in sys.modules and getattr(sys.modules["crit"], "__file__", "") == os.path.join(STAGE2_SRC, "crit.py"):
        return sys.modules["crit"]
    spec = importlib.util.spec_from_file_location("crit", os.path.join(STAGE2_SRC, "crit.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crit"] = mod
    spec.loader.exec_module(mod)
    return mod


def cap_threads(n: int = 1) -> None:
    """Pin BLAS / OpenMP thread pools (call before importing numpy)."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(n)
    os.environ.setdefault("HRHV2_JOBS", "1")


def seed_from_key(key: str) -> int:
    """First 8 bytes (little endian) of sha256(key) -- the hrhv2.util rule."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def rng(key: str):
    import numpy as np
    return np.random.default_rng(seed_from_key(key))


def torch_gen(key: str):
    import torch
    g = torch.Generator(device="cpu")
    g.manual_seed(seed_from_key(key) % (2 ** 63))
    return g


def run_key(arm: str, L: str, log2N: int, rep: int, pilot: bool = False) -> str:
    """hrhv2-s23|arm=<arm>|L=<L>|log2N=<n>|rep=<r>[|pilot]  (suffix |data or |init added by callers)."""
    k = f"{KEY}|arm={arm}|L={L}|log2N={log2N}|rep={rep}"
    if pilot:
        k += "|pilot"
    return k


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_status(role: str, text: str) -> str:
    """Append ONE line '<UTC>Z [s23-<role>] text' to the shared STATUS.md (test mode: a test file)."""
    import fcntl
    line = f"{utcnow()} [s23-{role}] {' '.join(text.split())}\n"
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    with open(STATUS, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return line


def status_contains(s: str) -> bool:
    if not os.path.exists(STATUS):
        return False
    with open(STATUS) as f:
        return s in f.read()


def _jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return str(o)  # 'nan' / 'inf' / '-inf' as strings (strict JSON)
    return o


def dump_json(obj, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_jsonable(obj), f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return sha256_file(path)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def fnum(x):
    """Inverse of the 'nan'/'inf' string encoding used by dump_json."""
    if isinstance(x, str) and x in ("nan", "inf", "-inf"):
        return float(x)
    return x


def torch_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- frozen code
def code_files() -> list:
    import glob
    real_src = os.path.join(S23_DIR, "src")
    files = sorted(glob.glob(os.path.join(real_src, "*.py")) + glob.glob(os.path.join(real_src, "*.sh"))
                   + glob.glob(os.path.join(real_src, "tests", "*.py")))
    return files


def code_hashes() -> dict:
    """sha256 of every frozen file: s23/src/*.py, *.sh, tests/*.py, hrhv2/*.py and Stage 2 crit.py."""
    import glob
    out = {}
    for p in code_files():
        out[os.path.relpath(p, S23_DIR)] = sha256_file(p)
    for p in sorted(glob.glob(os.path.join(STAGE1, "hrhv2", "*.py"))):
        out[os.path.relpath(p, HOME)] = sha256_file(p)
    crit = os.path.join(STAGE2_SRC, "crit.py")
    out[os.path.relpath(crit, HOME)] = sha256_file(crit)
    return out


def verify_frozen_code(strict: bool = True) -> bool:
    """Compare current hashes with prereg/CODE_AT_PREREG.sha256. Every Stage A entry point calls this
    (test mode: skipped). A mismatch aborts: a code change after the freeze is an amendment (section 13)."""
    if TEST_MODE:
        return True
    ref = os.path.join(PREREG_DIR, "CODE_AT_PREREG.sha256")
    if not os.path.exists(ref):
        if strict:
            raise RuntimeError("CODE_AT_PREREG.sha256 missing: code not frozen")
        return False
    want = {}
    with open(ref) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                h, p = line.split(None, 1)
                want[p] = h
    have = code_hashes()
    bad = [p for p, h in want.items() if have.get(p) != h]
    extra = [p for p in have if p.startswith("src/") and p not in want]
    if (bad or extra) and strict:
        raise RuntimeError(f"frozen-code hash mismatch: {bad + extra}")
    return not (bad or extra)


# ----------------------------------------------------------------------------- GPU ledger
def gpu_hours_used() -> float:
    """Sum of the wall seconds of every recorded GPU section (logs/gpu_ledger/*.tsv), in hours.
    Sections of concurrent processes are summed (conservative: >= device time)."""
    import glob
    tot = 0.0
    for p in glob.glob(os.path.join(GPU_LEDGER_DIR, "*.tsv")):
        with open(p) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    try:
                        tot += float(parts[2])
                    except ValueError:
                        pass
    return tot / 3600.0


def gpu_ledger_add(worker: str, what: str, seconds: float) -> None:
    os.makedirs(GPU_LEDGER_DIR, exist_ok=True)
    with open(os.path.join(GPU_LEDGER_DIR, f"{worker}.tsv"), "a") as f:
        f.write(f"{utcnow()}\t{what}\t{seconds:.3f}\n")


def gpu_budget_ok(soft: bool = True) -> bool:
    return gpu_hours_used() < (GPU_SOFT_STOP_H if soft else GPU_HARD_STOP_H)


class GpuSection:
    """with GpuSection(worker, what): ...  -> records the wall time of the block in the ledger."""

    def __init__(self, worker: str, what: str):
        self.worker, self.what = worker, what

    def __enter__(self):
        import time
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        import time
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        self.seconds = time.time() - self.t0
        gpu_ledger_add(self.worker, self.what, self.seconds)
        return False
