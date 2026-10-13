"""Shared helpers for the llama-axis extraction role (new file; role llama-axis-extract, 2026-10-08).

Paths follow analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md v1 (sha256 990428e3...): features go to
features/wit1024_llama_axis/, check outputs to features/_llama_axis_check/, validation JSONs to
results/S1_2_layermax_llama/extraction/. Nothing under features/wit1024/ is written.
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
import registry_llama_axis as R  # noqa: E402

ROLE = "llama-axis-extract"
PLAN = ROOT / "analysis" / "s12_layermax_llama" / "ANALYSIS_PLAN_LLAMA.md"
PLAN_SHA256 = "990428e3f29c7ecfba9cf314a45c73c79f0154f86a07520429910d282c0a2fc8"
OUT_DIR = FEATURES / R.OUT_DIR_NAME
CHECK_DIR = FEATURES / R.CHECK_DIR_NAME
RES_DIR = ROOT / "results" / "S1_2_layermax_llama" / "extraction"
STAGE1_7B = FEATURES / "wit1024" / "huggyllama__llama-7b"
STAGE1_7B_NPY_SHA256 = "1a727b782b144573ba4218d655b50d60d585a031422d0bb8cec6e67e7021e0a1"
REL_TOL = 1e-6  # plan section 2, E2/E3


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
