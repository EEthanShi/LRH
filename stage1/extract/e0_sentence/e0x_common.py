"""E0 extraction: shared paths, pins and helpers (ANALYSIS_PLAN_E0.md v1.0 §2.1, §3.2, §4).

Pins are read from analysis/e0_sentence/design/design_inputs.json, whose sha256 must equal the designer's
value recorded in STATUS (a97543f6...). HF_HOME is set to extract/e0_sentence/hf_home before any
huggingface import by the scripts that download.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import time

import numpy as np

ROOT = os.path.expanduser("~/hrh_v2_stage1_20261008")
HERE = os.path.dirname(os.path.abspath(__file__))
STATUS = os.path.join(ROOT, "STATUS.md")
HF_HOME = os.path.join(HERE, "hf_home")
HUB_DIR = os.path.join(HF_HOME, "hub")
CAPDIR = os.path.join(HERE, "coco_captions")
FEAT = os.path.join(ROOT, "features", "coco_e0")
ITEMS = os.path.join(ROOT, "features", "coco_pool", "items.parquet")
OUT = os.path.join(ROOT, "results", "E0_sentence")
EXTRACTION = os.path.join(OUT, "extraction")
DESIGN_INPUTS = os.path.join(ROOT, "analysis", "e0_sentence", "design", "design_inputs.json")
DESIGN_INPUTS_SHA256 = "a97543f6045403488a112f38513a979970c65d376064d619deca2689542894e4"
SHARED_HUB = os.path.expanduser("~/.cache/huggingface/hub")      # read only (Q1 LMs)

ENCODERS = {
    "mpnet": {"repo": "sentence-transformers/all-mpnet-base-v2", "pooling": "mean", "normalize": True,
              "max_tokens": 384, "weights": "model.safetensors"},
    "contriever": {"repo": "facebook/contriever", "pooling": "mean", "normalize": False,
                   "max_tokens": 512, "weights": "pytorch_model.bin"},
    "qwen3emb06": {"repo": "Qwen/Qwen3-Embedding-0.6B", "pooling": "last", "normalize": True,
                   "max_tokens": 8192, "weights": "model.safetensors", "last_token_id": 151643},
}
COCO_ZIP_URLS = ("https://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                 "http://images.cocodataset.org/annotations/annotations_trainval2014.zip")
COCO_ZIP_SIZE = 252872794
COCO_ZIP_MD5 = "0a379cfc70b0e71301e0f377548639bd"
COCO_MEMBER = "annotations/captions_val2014.json"
Q1_LMS = {  # S1.1 pins (extract/registry.py LM_REVISIONS), local snapshots only
    "bloomz_1b7": ("bigscience/bloomz-1b7", "4ea0141c75b2b96e17bc41f0d52c5368c879a281", "bigscience__bloomz-1b7"),
    "openllama_3b": ("openlm-research/open_llama_3b", "141067009124b9c0aea62c76b3eb952174864057",
                     "openlm-research__open_llama_3b"),
    "pythia410m_s1": ("EleutherAI/pythia-410m-seed1", "33803c4f5a1e9a4bece59e52f17bb8755add33e1",
                      "EleutherAI__pythia-410m-seed1"),
}
B7_REPO = ("sentence-transformers/coco-captions", "bd2601822b9af9a41656d678ffbd5c80d81e276a")


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(path) -> str:
    """git's blob id: sha1(b'blob <size>\\0' + content)."""
    size = os.path.getsize(path)
    h = hashlib.sha1(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def jsonable(v):
    import math
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    return v


def write_new(path, data) -> None:
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        raise FileExistsError(f"stale temporary {tmp}; inspect it")
    with open(tmp, "wb" if isinstance(data, bytes) else "w") as f:
        f.write(data)
    os.replace(tmp, path)


def write_json_new(path, obj) -> str:
    write_new(path, json.dumps(jsonable(obj), indent=1))
    return sha256_file(path)


def save_npy_new(path, arr) -> str:
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)
    return sha256_file(path)


def load_pins() -> dict:
    if sha256_file(DESIGN_INPUTS) != DESIGN_INPUTS_SHA256:
        raise SystemExit("REFUSED: design_inputs.json differs from the designer's file (pins unverifiable)")
    with open(DESIGN_INPUTS) as f:
        return json.load(f)["hub_pins"]


def verify_snapshot(snap: str, pin: dict) -> dict:
    """Every pinned file present; LFS files by sha256, others by git blob sha1 (plan §2.1). Raises on mismatch."""
    res = {}
    for name, info in pin["files"].items():
        p = os.path.join(snap, name)
        if not os.path.exists(p):
            raise RuntimeError(f"{name} missing in snapshot {snap}")
        size = os.path.getsize(p)
        if size != info["size"]:
            raise RuntimeError(f"{name}: size {size} != pinned {info['size']}")
        if info.get("lfs_sha256"):
            h = sha256_file(p)
            ok = h == info["lfs_sha256"]
            res[name] = {"sha256": h, "check": "lfs_sha256", "ok": ok}
        else:
            h = git_blob_sha1(p)
            ok = h == info["git_blob"]
            res[name] = {"git_blob_sha1": h, "check": "git_blob", "ok": ok}
        if not ok:
            raise RuntimeError(f"{name}: hash mismatch with the pin ({res[name]})")
    return res


def set_strict_fp32() -> dict:
    """Copy of extract/common.py set_strict_fp32 (TF32 off; PROTOCOL §6)."""
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
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        flags = dict(api="legacy", matmul_allow_tf32=False, cudnn_allow_tf32=False)
    torch.backends.cudnn.benchmark = False
    flags["cudnn_benchmark"] = False
    return flags


def env_info() -> dict:
    import torch
    info = dict(python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__,
                torch=torch.__version__, cuda=torch.version.cuda)
    try:
        import transformers
        import huggingface_hub
        import tokenizers
        info.update(transformers=transformers.__version__, huggingface_hub=huggingface_hub.__version__,
                    tokenizers=tokenizers.__version__)
    except Exception as e:  # pragma: no cover
        info["import_error"] = repr(e)
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def status_has(pattern: str) -> bool:
    with open(STATUS, encoding="utf-8") as f:
        return any(re.search(pattern, ln) for ln in f)


def require_plan_and_code_frozen() -> dict:
    """STATUS has PLAN FROZEN and E0 CODE FROZEN (in this order) and both files still have the logged sha256."""
    checks = ((r"\[e0-freeze\] PLAN FROZEN", r"ANALYSIS_PLAN_E0\.md sha256 ([0-9a-f]{64})",
               os.path.join(ROOT, "analysis", "e0_sentence", "ANALYSIS_PLAN_E0.md")),
              (r"\[e0-freeze\] E0 CODE FROZEN", r"CODE_MANIFEST_E0\.json sha256 ([0-9a-f]{64})",
               os.path.join(ROOT, "analysis", "e0_sentence", "CODE_MANIFEST_E0.json")))
    with open(STATUS, encoding="utf-8") as f:
        lines = f.read().splitlines()
    out, last = {}, -1
    for tag, hre, path in checks:
        hit = [(i, ln) for i, ln in enumerate(lines) if re.search(tag, ln)]
        if not hit:
            raise SystemExit(f"REFUSED: STATUS lacks {tag}")
        i, ln = hit[0]
        m = re.search(hre, ln)
        if not m or not os.path.exists(path) or sha256_file(path) != m.group(1) or i <= last:
            raise SystemExit(f"REFUSED: {path} does not match its STATUS line (or out of order)")
        last = i
        out[os.path.basename(path)] = m.group(1)
    return out
