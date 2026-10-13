"""Download every model in the queue to the HF cache ahead of the GPU driver (network only, no GPU).

Uses huggingface_hub downloads only (no transformers loading, so no safetensors auto-conversion side effects).
Stops if free disk < --min_free_gb. Writes features/PREFETCH.done or PREFETCH.failed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from common import FEATURES, append_status, disk_free_gb, utcnow  # noqa: E402
from registry import build_queue  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_free_gb", type=float, default=60.0)
    args = ap.parse_args()
    import extract_core as ec
    fails = []
    t0 = time.time()
    for marker in ["PREFETCH.done", "PREFETCH.failed"]:
        if (FEATURES / marker).exists():
            (FEATURES / marker).unlink()
    for job in build_queue():
        m = job["model"]
        free = disk_free_gb()
        if free < args.min_free_gb:
            fails.append((m, f"disk free {free:.0f} GB < {args.min_free_gb}"))
            print(f"{utcnow()} STOP disk", flush=True)
            break
        t = time.time()
        try:
            if job["kind"] == "vit":
                info = ec.ensure_timm_weights(m)
                print(f"{utcnow()} ok {m} {info['snapshot']} {info['weights']} {time.time()-t:.0f}s", flush=True)
            else:
                p = ec.ensure_lm_snapshot(m)
                print(f"{utcnow()} ok {m} {p} {time.time()-t:.0f}s", flush=True)
        except Exception as e:
            fails.append((m, repr(e)))
            print(f"{utcnow()} FAIL {m} {e!r}\n{traceback.format_exc()}", flush=True)
    dt = (time.time() - t0) / 60
    if fails:
        (FEATURES / "PREFETCH.failed").write_text("\n".join(map(str, fails)) + "\n")
        append_status(f"prefetch finished with {len(fails)} failures in {dt:.0f} min", "features/PREFETCH.failed")
    else:
        (FEATURES / "PREFETCH.done").write_text(utcnow() + "\n")
        append_status(f"prefetch finished OK in {dt:.0f} min (disk free {disk_free_gb():.0f} GB)", "logs/extract_prefetch.log")


if __name__ == "__main__":
    main()
