"""Shared paths and helpers for HRH v2 Stage-1 feature extraction (role: extraction)."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path.home() / "hrh_v2_stage1_20261008"
EXTRACT = ROOT / "extract"
FEATURES = ROOT / "features"
LOGS = ROOT / "logs"
STATUS = ROOT / "STATUS.md"
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
ROLE = "extraction"

# Pinned dataset snapshots (verified 2026-10-08 against HfApi.list_repo_refs / dataset_info).
PRH_REPO = "minhuh/prh"
WIT1024_REV = "0220f2b7b28bacc1f7034f358e46e39bd9078a03"  # branch wit_1024
WIT4096_REV = "c41c877fc431ea20589dc0d8a57fb5aee1167baa"  # branch wit_4096
COCO_REPO = "sayakpaul/coco-30-val-2014"
COCO_REV = "abdde3200f533ddae5bed2438057f1f7ea2d5131"  # main
COCO_SALT = "hrh-v2-coco-pool-v1"
COCO_POOL_N = 16384
SEL_ROWS = (1024, 2048)  # 0-based half-open: dataset rows 1025..2048 (1-based) of wit_4096


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


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrs) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def disk_free_gb(path=None) -> float:
    return shutil.disk_usage(path or Path.home()).free / 1e9


def code_hashes() -> dict:
    out = {}
    for p in sorted(EXTRACT.glob("*.py")):
        out[p.name] = sha256_file(p)
    return out


def env_info() -> dict:
    import torch
    info = dict(python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__,
                torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
    try:
        import transformers, timm, PIL, datasets, huggingface_hub, torchvision
        info.update(transformers=transformers.__version__, timm=timm.__version__, pillow=PIL.__version__,
                    datasets=datasets.__version__, huggingface_hub=huggingface_hub.__version__,
                    torchvision=torchvision.__version__)
    except Exception as e:  # pragma: no cover
        info["import_error"] = repr(e)
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def set_strict_fp32() -> dict:
    """Disable TF32 everywhere (true IEEE fp32 matmul/conv). Returns the resulting flags."""
    import torch
    m = torch.backends.cuda.matmul
    if hasattr(m, "fp32_precision"):
        m.fp32_precision = "ieee"
        try:
            torch.backends.cudnn.conv.fp32_precision = "ieee"
        except Exception:
            pass
        flags = dict(api="fp32_precision", matmul=m.fp32_precision,
                     cudnn_conv=getattr(getattr(torch.backends.cudnn, "conv", None), "fp32_precision", None))
    else:  # legacy API
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        flags = dict(api="legacy", matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                     cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                     matmul_precision=torch.get_float32_matmul_precision())
    torch.backends.cudnn.benchmark = False
    flags["cudnn_benchmark"] = False
    return flags


def blob_info_for_snapshot_file(path: Path) -> dict:
    """For a file inside an HF cache snapshot, return size and the blob name (= sha256 for LFS files)."""
    p = Path(path)
    real = p.resolve()
    return dict(file=p.name, size=real.stat().st_size, blob=real.name)
