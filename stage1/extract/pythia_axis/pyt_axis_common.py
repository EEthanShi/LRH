"""Shared helpers for the Pythia-axis extraction role (role pyt-axis-extract, S0b, 2026-10-10).

PYT-AXIS COPY of extract/openllama_axis/ol_axis_common.py (sha256 58fb4c4e...). Changes, all marked "PYT-AXIS": paths
(features/wit1024_pythia_axis/, its _tok1/ and _e2_check/, results/S1_2_layermax_pythia/{extraction,logs}/, the
dedicated Hugging Face home extract/pythia_axis/hf_home/), the plan and its sha256, the role tag [pyt-axis-extract],
the resource STOP limits of plan sections 2 and 11 (free disk >= 20 GB measured with statvfs on the filesystem of the
dedicated HF_HOME; available RAM >= 16 GB during every load), the order-gate readers of STATUS.md (CODE FROZEN, S3 GO,
DEFERRED), the S0b test mode, and the no-delete rule (an output that would be replaced is first copied aside).
Paths follow analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md v1.0 (sha256 6de89603...). Nothing under
features/wit1024/ and nothing in the standard Hugging Face cache is written.

S0b test mode (PYT-AXIS): PYT_S0B_TEST=1 is accepted only while STATUS.md has no "[pyt-axis-<role>] CODE FROZEN"
milestone line (else exit 3). In test mode every output root, the STATUS file and the hub root move under
extract/pythia_axis/s0b_test_out/, nothing is ever downloaded (download_pyt.py refuses the network in test mode),
and no output name of the axis (EleutherAI__pythia-<size>...) is written.
"""
from __future__ import annotations

import fcntl
import json
import os
import re  # PYT-AXIS
import shutil  # PYT-AXIS
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTRACT_DIR = HERE.parent
for p in (str(EXTRACT_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from common import FEATURES, ROOT, sha256_file  # noqa: E402,F401
from common import STATUS as REAL_STATUS  # noqa: E402  PYT-AXIS
import registry_pythia_axis as R  # noqa: E402  PYT-AXIS

ROLE = os.environ.get("PYT_ROLE", "pyt-axis-extract")  # PYT-AXIS: status tag [pyt-axis-<role>]
if not re.fullmatch(r"pyt-axis-[a-z0-9-]+", ROLE):  # PYT-AXIS
    raise SystemExit(f"bad PYT_ROLE {ROLE!r}")
PLAN = ROOT / "analysis" / "s12_layermax_pythia" / "ANALYSIS_PLAN_PYTHIA.md"  # PYT-AXIS
PLAN_SHA256 = "6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b"  # PYT-AXIS
EXIT_GATE = 3  # PYT-AXIS: an order gate refused (dry run without the plan's gates)
EXIT_DEFER = 77  # PYT-AXIS: GPU shortage -> deferral (plan section 11), not a failure

# ------------------------------------------------------------------------------------------- PYT-AXIS: STATUS readers
_TS = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ "
CODE_FROZEN_RE = re.compile(_TS + r"\[pyt-axis-[A-Za-z0-9-]+\] CODE FROZEN")
S3_GO_RE = re.compile(_TS + r"\[pyt-axis-[A-Za-z0-9-]+\] S3 GO")
DEFERRED_RE = re.compile(_TS + r"\[pyt-axis-[A-Za-z0-9-]+\] DEFERRED")


def _real_status_lines() -> list[str]:  # PYT-AXIS
    with open(REAL_STATUS, encoding="utf-8") as f:
        return f.read().splitlines()


def code_frozen_line() -> str | None:  # PYT-AXIS: the S0b milestone line (always read from the real STATUS.md)
    for ln in _real_status_lines():
        if CODE_FROZEN_RE.match(ln):
            return ln
    return None


def _test_mode() -> bool:  # PYT-AXIS
    if os.environ.get("PYT_S0B_TEST") != "1":
        return False
    if code_frozen_line() is not None:
        print("[pyt] REFUSED (exit 3): PYT_S0B_TEST=1 after the CODE FROZEN line", file=sys.stderr, flush=True)
        raise SystemExit(EXIT_GATE)
    return True


TEST_MODE = _test_mode()  # PYT-AXIS
TEST_ROOT = HERE / "s0b_test_out"  # PYT-AXIS: every test output (never the axis output names)
if TEST_MODE:  # PYT-AXIS
    STATUS = TEST_ROOT / "STATUS_test.md"
    OUT_DIR = TEST_ROOT / "features"
    TOK1_DIR = TEST_ROOT / "features" / "_tok1"
    CHECK_DIR = TEST_ROOT / "features" / "_e2_check"
    RES_DIR = TEST_ROOT / "results" / "extraction"
    LOGS_DIR = TEST_ROOT / "results" / "logs"
    HF_HOME = TEST_ROOT / "hf_home"
else:
    STATUS = REAL_STATUS
    OUT_DIR = FEATURES / R.OUT_DIR_NAME
    TOK1_DIR = FEATURES / R.TOK1_DIR_NAME
    CHECK_DIR = FEATURES / R.CHECK_DIR_NAME
    RES_DIR = ROOT / "results" / "S1_2_layermax_pythia" / "extraction"
    LOGS_DIR = ROOT / "results" / "S1_2_layermax_pythia" / "logs"
    HF_HOME = HERE / "hf_home"  # PYT-AXIS: dedicated HF_HOME (plan section 2), receives every Pythia download
PYT_HUB = HF_HOME / "hub"  # PYT-AXIS
STD_HUB = Path.home() / ".cache" / "huggingface" / "hub"  # PYT-AXIS: read-only, E2 (bloomz-560m) only
LEDGER = ROOT / "results" / "S1_2_layermax_pythia" / "logs" / "gpu_ledger.tsv"  # PYT-AXIS (written by launch_pyt.sh)
STAGE1_E2 = FEATURES / "wit1024" / R.slug(R.E2_MODEL)  # PYT-AXIS: stored bloomz-560m features (E2 reference)
REL_TOL = 1e-6
MIN_FREE_DISK_GB = 20.0  # PYT-AXIS: R.1.4 / plan section 11 disk guard (GB = 1e9 B)
MIN_AVAIL_RAM_GB = 16.0  # PYT-AXIS: plan section 2 STOP condition (during the load)
RESERVE_FEATURES_GB = 3.0  # PYT-AXIS: plan section 11 reserve (primary + _tok1 + E2)
RESERVE_RESULTS_GB = 3.0  # PYT-AXIS: plan section 11 reserve (results)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_status(msg: str, paths: str = "") -> None:
    line = f"{utcnow()} [{ROLE}] {msg}" + (f" ({paths})" if paths else "")
    line = line.replace("\n", " ") + "\n"
    Path(STATUS).parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def log(msg: str) -> None:
    print(f"{utcnow()} {msg}", flush=True)


def keep_aside(path: Path) -> Path | None:  # PYT-AXIS: no-delete rule; an existing file is copied aside, never removed
    path = Path(path)
    if not path.exists():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dst = path.with_name(path.name + f".prev-{ts}")
    shutil.copy2(path, dst)
    return dst


def move_marker_aside(path: Path) -> Path | None:  # PYT-AXIS: replaces unlink() of .done/.failed markers
    path = Path(path)
    if not path.exists():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dst = path.with_name(path.name + f".prev-{ts}")
    os.replace(path, dst)
    return dst


def write_json(path: Path, obj) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep_aside(path)  # PYT-AXIS
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)
    return sha256_file(path)


def read_json(path: Path):  # PYT-AXIS
    return json.load(open(path)) if Path(path).exists() else None


def check_plan_unchanged() -> None:
    h = sha256_file(PLAN)
    if h != PLAN_SHA256:
        raise SystemExit(f"frozen plan changed: {h} != {PLAN_SHA256}")


def require(cond: bool, why: str) -> None:  # PYT-AXIS: order gate (exit 3, nothing run)
    if not cond:
        print(f"[pyt] REFUSED (exit {EXIT_GATE}): order gate: {why}", file=sys.stderr, flush=True)
        raise SystemExit(EXIT_GATE)


def s3_go_line() -> str | None:  # PYT-AXIS: plan R.3 Q0 / section 11 S3
    """The last '[pyt-axis-<role>] S3 GO' milestone line that cites G0 and the authorisation and is not followed by a
    '[pyt-axis-<role>] DEFERRED' line; None otherwise."""
    lines = _real_status_lines()
    last_go, last_def = None, None
    for i, ln in enumerate(lines):
        if S3_GO_RE.match(ln) and "G0" in ln and "author" in ln.lower():
            last_go = i
        if DEFERRED_RE.match(ln):
            last_def = i
    if last_go is None or (last_def is not None and last_def > last_go):
        return None
    return lines[last_go]


def compare_features(cand_stem: Path, ref_stem: Path, first_n: int = 64, ref_rows: int | None = None) -> dict:  # PYT-AXIS
    """Compare <cand>.npy/.lm_aux.npz with <ref>.npy/.lm_aux.npz. Relative difference per layer =
    max|cand - ref| / max|ref| over that layer (the fx_check convention of extract_core).
    PYT-AXIS: ref_rows=k compares the candidate with the first k rows of the reference (S0b smoke on k items)."""
    A = np.load(str(cand_stem) + ".npy", mmap_mode="r")
    B = np.load(str(ref_stem) + ".npy", mmap_mode="r")
    if ref_rows is not None:  # PYT-AXIS
        B = B[:ref_rows]
    out = dict(cand=str(cand_stem) + ".npy", ref=str(ref_stem) + ".npy", cand_shape=list(A.shape),
               ref_shape=list(B.shape), cand_dtype=str(A.dtype), ref_dtype=str(B.dtype), ref_rows=ref_rows)
    if A.shape != B.shape or A.dtype != B.dtype:
        out.update(ok_shape=False, bitwise_equal=False, max_rel_diff=float("inf"))
        return out
    out["ok_shape"] = True
    out["cand_sha256"] = sha256_file(str(cand_stem) + ".npy")
    out["ref_sha256"] = sha256_file(str(ref_stem) + ".npy")
    per_layer = []
    for l in range(A.shape[1]):
        a = np.asarray(A[:, l], dtype=np.float64)
        b = np.asarray(B[:, l], dtype=np.float64)
        d = np.abs(a - b)
        refmax = float(np.abs(b).max())
        dn = d[:first_n]
        refn = float(np.abs(b[:first_n]).max())
        per_layer.append(dict(layer=l, max_abs=float(d.max()), max_rel=float(d.max() / refmax) if refmax > 0 else None,
                              n_diff=int((d > 0).sum()),
                              first_n_max_abs=float(dn.max()), first_n_max_rel=float(dn.max() / refn) if refn > 0 else None))
    out["per_layer"] = per_layer
    out["bitwise_equal"] = bool((ref_rows is None and out["cand_sha256"] == out["ref_sha256"])
                                or np.array_equal(np.asarray(A), np.asarray(B)))  # PYT-AXIS: ref_rows
    out["max_abs_diff"] = max(r["max_abs"] for r in per_layer)
    out["max_rel_diff"] = max((r["max_rel"] or 0.0) for r in per_layer)
    out["first_n"] = first_n
    out["first_n_max_abs_diff"] = max(r["first_n_max_abs"] for r in per_layer)
    out["first_n_max_rel_diff"] = max((r["first_n_max_rel"] or 0.0) for r in per_layer)
    out["n_elements_differing"] = int(sum(r["n_diff"] for r in per_layer))
    # lm_aux
    try:
        xa = np.load(str(cand_stem) + ".lm_aux.npz")
        xb = np.load(str(ref_stem) + ".lm_aux.npz")
        aux = {}
        for k in xb.files:
            a, b = xa[k], xb[k]
            if ref_rows is not None:  # PYT-AXIS
                b = b[:ref_rows]
            if a.shape != b.shape:
                aux[k] = dict(shape_mismatch=[list(a.shape), list(b.shape)])
                continue
            if k == "n_tokens":
                aux[k] = dict(equal=bool(np.array_equal(a, b)))
            else:
                a64, b64 = a.astype(np.float64), b.astype(np.float64)
                fin = np.isfinite(b64)
                d = np.abs(a64[fin] - b64[fin])
                aux[k] = dict(bitwise_equal=bool(np.array_equal(a, b, equal_nan=True)), max_abs=float(d.max()) if d.size else 0.0,
                              max_rel=float(d.max() / np.abs(b64[fin]).max()) if d.size and np.abs(b64[fin]).max() > 0 else 0.0,
                              nan_pattern_equal=bool(np.array_equal(np.isnan(a64), np.isnan(b64))))
        out["lm_aux"] = aux
    except Exception as e:  # pragma: no cover
        out["lm_aux"] = dict(error=repr(e))
    return out


# ------------------------------------------------------------------------------------------- resource checks
def mem_available_gb() -> float:  # PYT-AXIS
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) * 1024 / 1e9  # PYT-AXIS: kB (KiB) -> GB (1e9 B)
    return float("nan")


def _existing(path: Path) -> Path:  # PYT-AXIS: nearest existing ancestor (the guard measures that filesystem)
    p = Path(path)
    while not p.exists():
        p = p.parent
    return p


def disk_free_bytes(path: Path | None = None) -> int:  # PYT-AXIS: statvfs on the filesystem of the dedicated HF_HOME
    st = os.statvfs(str(_existing(path or HF_HOME)))
    return int(st.f_bavail) * int(st.f_frsize)


def disk_free_gb(path: Path | None = None) -> float:  # PYT-AXIS
    return disk_free_bytes(path) / 1e9


class ResourceWatch:  # plan sections 2 and 11 STOP conditions, polled in a daemon thread PYT-AXIS
    """Polls available RAM and free disk every `period` s. On a breach it records the reading and calls
    os._exit(code): the process ends at once (no output file is written), and the launcher marks it failed.
    PYT-AXIS: limits 16 GB RAM (during loads) and 20 GB free disk on the filesystem of the dedicated HF_HOME."""

    def __init__(self, min_ram_gb: float = MIN_AVAIL_RAM_GB, min_disk_gb: float = MIN_FREE_DISK_GB,
                 period: float = 1.0, code: int = 7, label: str = ""):
        import threading
        self.min_ram, self.min_disk, self.period, self.code, self.label = min_ram_gb, min_disk_gb, period, code, label
        self.min_seen_ram = float("inf")
        self.min_seen_disk = float("inf")
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            ram, disk = mem_available_gb(), disk_free_gb()  # PYT-AXIS
            self.min_seen_ram, self.min_seen_disk = min(self.min_seen_ram, ram), min(self.min_seen_disk, disk)
            if ram < self.min_ram or disk < self.min_disk:
                msg = (f"STOP {self.label}: available RAM {ram:.1f} GB (min {self.min_ram}) / free disk {disk:.1f} GB "
                       f"(min {self.min_disk}) (plan sections 2 and 11 STOP condition)")
                log(msg)
                try:
                    append_status(msg)
                finally:
                    os._exit(self.code)
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)
        return False

    def summary(self) -> dict:
        return dict(min_available_ram_gb=self.min_seen_ram, min_free_disk_gb=self.min_seen_disk,
                    limits=dict(ram_gb=self.min_ram, disk_gb=self.min_disk))


def ledger_rows() -> list[list[str]]:  # PYT-AXIS
    if not LEDGER.exists():
        return []
    return [ln.split("\t") for i, ln in enumerate(LEDGER.read_text().splitlines()) if i > 0 and ln.strip()]


def gpu_ledger_total_min() -> float:  # running total of the axis GPU ledger (launch_pyt.sh) PYT-AXIS
    return float(sum(float(r[4]) for r in ledger_rows()))
