"""Shared helpers for the OpenLLaMA-axis extraction role (role oll-axis-extract, S0b, 2026-10-09).

OL-AXIS COPY of extract/llama_axis/llama_axis_common.py (sha256 1e4aef6d...). Changes, all marked "OL-AXIS": paths
(features/wit1024_openllama_axis/, its _e2_check/ and results/S1_2_layermax_openllama/extraction/), the plan and its
sha256, the role tag [oll-axis-extract], the stored Stage-1 references of 3b/7b, the resource STOP checks of plan
section 2 (disk >= 40 GB free, available RAM >= 16 GB during the load) and the GPU ledger reader.
Paths follow analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md v1.0 (sha256 4f18d710...). Nothing under
features/wit1024/ is written.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTRACT_DIR = HERE.parent
for p in (str(EXTRACT_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from common import FEATURES, LOGS, ROOT, STATUS, sha256_file  # noqa: E402,F401
import registry_openllama_axis as R  # noqa: E402  OL-AXIS

ROLE = "oll-axis-extract"  # OL-AXIS: status tag [oll-axis-<role>] (plan R.3.1)
PLAN = ROOT / "analysis" / "s12_layermax_openllama" / "ANALYSIS_PLAN_OPENLLAMA.md"  # OL-AXIS
PLAN_SHA256 = "4f18d710beda45492b1e191f74580bfc4ef53f6dafe791ad245ff1cf335f3240"  # OL-AXIS
OUT_DIR = FEATURES / R.OUT_DIR_NAME
CHECK_DIR = FEATURES / R.CHECK_DIR_NAME
E3_DIR = FEATURES / R.E3_DIR_NAME  # OL-AXIS
E5_DIR = FEATURES / R.E5_DIR_NAME  # OL-AXIS
RES_DIR = ROOT / "results" / "S1_2_layermax_openllama" / "extraction"  # OL-AXIS
LEDGER = ROOT / "results" / "S1_2_layermax_openllama" / "logs" / "gpu_ledger.tsv"  # OL-AXIS (written by launch_ol.sh)
STAGE1 = {m: FEATURES / "wit1024" / R.slug(m) for m in R.LM_E2}  # OL-AXIS: stored 3b/7b features (E2 references)
STAGE1_NPY_SHA256 = R.STAGE1_NPY_SHA256  # OL-AXIS
REL_TOL = 1e-6  # plan section 2, E2/E3
MIN_FREE_DISK_GB = 40.0  # OL-AXIS: plan section 2 STOP condition
MIN_AVAIL_RAM_GB = 16.0  # OL-AXIS: plan section 2 STOP condition (during the load)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_status(msg: str, paths: str = "") -> None:
    line = f"{utcnow()} [{ROLE}] {msg}" + (f" ({paths})" if paths else "")
    line = line.replace("\n", " ") + "\n"
    with open(STATUS, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def log(msg: str) -> None:
    print(f"{utcnow()} {msg}", flush=True)


def write_json(path: Path, obj) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)
    return sha256_file(path)


def check_plan_unchanged() -> None:
    h = sha256_file(PLAN)
    if h != PLAN_SHA256:
        raise SystemExit(f"frozen plan changed: {h} != {PLAN_SHA256}")


def compare_features(cand_stem: Path, ref_stem: Path, first_n: int = 64) -> dict:
    """Compare <cand>.npy/.lm_aux.npz with <ref>.npy/.lm_aux.npz. Relative difference per layer =
    max|cand - ref| / max|ref| over that layer (the fx_check convention of extract_core)."""
    A = np.load(str(cand_stem) + ".npy", mmap_mode="r")
    B = np.load(str(ref_stem) + ".npy", mmap_mode="r")
    out = dict(cand=str(cand_stem) + ".npy", ref=str(ref_stem) + ".npy", cand_shape=list(A.shape),
               ref_shape=list(B.shape), cand_dtype=str(A.dtype), ref_dtype=str(B.dtype))
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
    out["bitwise_equal"] = bool(out["cand_sha256"] == out["ref_sha256"] or np.array_equal(A, B))
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


# ------------------------------------------------------------------------------------------- OL-AXIS: resource checks
def mem_available_gb() -> float:  # OL-AXIS
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6  # kB -> GB
    return float("nan")


def disk_free_gb_home() -> float:  # OL-AXIS
    import shutil
    return shutil.disk_usage(Path.home()).free / 1e9


class ResourceWatch:  # OL-AXIS: plan section 2 STOP conditions, polled in a daemon thread
    """Polls available RAM and free disk every `period` s. On a breach it records the reading and calls
    os._exit(code): the process ends at once (no output file is written), and the launcher marks it failed."""

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
            ram, disk = mem_available_gb(), disk_free_gb_home()
            self.min_seen_ram, self.min_seen_disk = min(self.min_seen_ram, ram), min(self.min_seen_disk, disk)
            if ram < self.min_ram or disk < self.min_disk:
                msg = (f"STOP {self.label}: available RAM {ram:.1f} GB (min {self.min_ram}) / free disk {disk:.1f} GB "
                       f"(min {self.min_disk}) (plan section 2 STOP condition)")
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


def gpu_ledger_total_min() -> float:  # OL-AXIS: running total of the axis GPU ledger (launch_ol.sh)
    if not LEDGER.exists():
        return 0.0
    tot = 0.0
    for i, ln in enumerate(LEDGER.read_text().splitlines()):
        if i == 0 or not ln.strip():
            continue
        tot += float(ln.split("\t")[4])
    return tot
