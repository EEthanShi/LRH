"""Download huggyllama/llama-13b and llama-30b at the pinned commits into the HF cache, then verify them (E1).

New file (role llama-axis-extract). Plan section 2 (E1, STOP conditions) and section 11 (S2).
- token=False on every Hub call; no token is configured or created.
- STOP (exit 2, STATUS line, no retry) on any 401/403, gated repo, licence/terms prompt or missing repo.
- allow_patterns = registry_llama_axis.ALLOW_PATTERNS (safetensors + small files; no *.bin, no *.py).
- E1: every shard's sha256 recomputed from the file content equals the Hub LFS sha256; small files checked by size and
  git blob sha1 (tokenizer.model by sha256); safetensors dtype recorded from the headers; snapshot dir = pinned commit.
Usage: python -I download_llama_axis.py [--models huggyllama/llama-13b ...]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

from llama_axis_common import (RES_DIR, LOGS, R, append_status, check_plan_unchanged, log, sha256_file,  # noqa: E402
                               write_json)
from common import HUB, disk_free_gb  # noqa: E402


class StopRun(Exception):
    pass


def is_auth_error(e: Exception) -> bool:
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError
    if isinstance(e, (GatedRepoError, RepositoryNotFoundError)):
        return True
    if isinstance(e, HfHubHTTPError):
        code = getattr(getattr(e, "response", None), "status_code", None)
        return code in (401, 403)
    s = repr(e)
    return any(x in s for x in ("401", "403", "gated", "Gated", "license", "terms", "authenticat"))


def git_blob_sha1(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


LEGACY_BUFFER_SUFFIX = "rotary_emb.inv_freq"  # legacy per-layer F32 buffer in old HF LLaMA checkpoints (not a weight)


def safetensors_header_dtypes(path: Path) -> tuple[dict, dict]:
    """(weight dtype counts, legacy-buffer dtype counts). Keys ending in rotary_emb.inv_freq are legacy fp32 rotary
    buffers saved by old transformers; transformers 5.15 does not load them (inv_freq is recomputed)."""
    import json
    import struct
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    c, leg = {}, {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        tgt = leg if k.endswith(LEGACY_BUFFER_SUFFIX) else c
        tgt[v["dtype"]] = tgt.get(v["dtype"], 0) + 1
    return c, leg


def verify(model: str, snap: Path) -> dict:
    rev = R.LM_REVISIONS[model]
    out = dict(model=model, revision=rev, snapshot_dir=str(snap), snapshot_name_ok=(snap.name == rev))
    files = sorted(p.name for p in snap.iterdir())
    out["snapshot_files"] = files
    out["forbidden_present"] = [f for f in files if f.endswith(".bin") or f.endswith(".py")]
    shards, dtypes, legacy = [], {}, {}
    t0 = time.time()
    for fname, size, lfs in R.WEIGHT_SHARDS[model]:
        p = snap / fname
        r = dict(file=fname, expected_size=size, expected_sha256=lfs, exists=p.exists())
        if p.exists():
            real = p.resolve()
            r["size"] = real.stat().st_size
            r["blob_name"] = real.name
            th = time.time()
            r["sha256"] = sha256_file(p)
            r["hash_s"] = time.time() - th
            r["ok"] = bool(r["size"] == size and r["sha256"] == lfs and real.name == lfs)
            d, lg = safetensors_header_dtypes(p)
            r["dtypes"] = d
            r["legacy_inv_freq_dtypes"] = lg
            for k, v in d.items():
                dtypes[k] = dtypes.get(k, 0) + v
            for k, v in lg.items():
                legacy[k] = legacy.get(k, 0) + v
        else:
            r["ok"] = False
        shards.append(r)
        log(f"  E1 {model} {fname}: ok={r['ok']} size={r.get('size')} sha256={r.get('sha256', '')[:16]}...")
    smalls = []
    for fname, size, bid in R.SMALL_FILES[model]:
        p = snap / fname
        r = dict(file=fname, expected_size=size, expected_id=bid, exists=p.exists())
        if p.exists():
            r["size"] = p.resolve().stat().st_size
            r["id"] = sha256_file(p) if fname == "tokenizer.model" else git_blob_sha1(p)
            r["ok"] = bool(r["size"] == size and r["id"] == bid)
        else:
            r["ok"] = False
        smalls.append(r)
    out.update(shards=shards, small_files=smalls, safetensors_dtypes=dtypes, legacy_inv_freq_dtypes=legacy,
               verify_s=time.time() - t0, total_weight_bytes=int(sum(s.get("size", 0) for s in shards)))
    out["shards_sha256_all_match"] = bool(all(s["ok"] for s in shards))
    out["small_files_all_match"] = bool(all(s["ok"] for s in smalls))
    out["weights_single_16bit_dtype"] = bool(len(dtypes) == 1 and next(iter(dtypes)) in ("F16", "BF16"))
    out["E1_rule"] = ("snapshot dir = pinned commit; no *.bin/*.py; every shard size + recomputed sha256 = Hub LFS; small "
                      "files size + git blob id (tokenizer.model sha256); all weight tensors one 16-bit dtype (legacy "
                      "per-layer F32 rotary_emb.inv_freq buffers recorded separately, not weights)")
    out["E1_pass"] = bool(out["snapshot_name_ok"] and not out["forbidden_present"] and out["shards_sha256_all_match"]
                          and out["small_files_all_match"] and out["weights_single_16bit_dtype"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=R.LM_TO_EXTRACT)
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--min_free_gb", type=float, default=40.0)
    ap.add_argument("--attempts", type=int, default=3)
    args = ap.parse_args()
    check_plan_unchanged()
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    log(f"download start pid={os.getpid()} models={args.models} allow={R.ALLOW_PATTERNS} max_workers={args.max_workers} "
        f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET')} affinity={sorted(os.sched_getaffinity(0))}")
    for model in args.models:
        rev = R.LM_REVISIONS[model]
        s = R.slug(model)
        marker_ok = LOGS / f"llama_axis_download.{s}.done"
        marker_bad = LOGS / f"llama_axis_download.{s}.failed"
        for mk in (marker_ok, marker_bad):
            if mk.exists():
                mk.unlink()
        try:
            # 1) metadata re-check without a token
            info = api.model_info(model, token=False)
            info_rev = api.model_info(model, revision=rev, token=False)
            meta = dict(main_sha=info.sha, pinned=rev, pinned_sha=info_rev.sha, gated=info.gated, private=info.private,
                        disabled=getattr(info, "disabled", None))
            log(f"{model}: {meta}")
            if info.gated not in (False, None) or info.private or info_rev.sha != rev:
                raise StopRun(f"metadata check failed: {meta}")
            need_gb = sum(x[1] for x in R.WEIGHT_SHARDS[model]) / 1e9
            free = disk_free_gb()
            if free < need_gb + args.min_free_gb:
                raise StopRun(f"disk free {free:.1f} GB < need {need_gb:.1f} + margin {args.min_free_gb} GB")
            # 2) download
            t0 = time.time()
            p = None
            last = None
            for att in range(1, args.attempts + 1):
                try:
                    p = snapshot_download(model, revision=rev, allow_patterns=R.ALLOW_PATTERNS, token=False,
                                          max_workers=args.max_workers)
                    break
                except Exception as e:  # noqa: BLE001
                    if is_auth_error(e):
                        raise StopRun(f"auth/gating/licence error, not retried: {e!r}")
                    last = e
                    log(f"{model}: attempt {att} failed (non-auth): {e!r}")
                    time.sleep(30)
            if p is None:
                raise RuntimeError(f"download failed after {args.attempts} attempts: {last!r}")
            dl_s = time.time() - t0
            snap = Path(p)
            log(f"{model}: downloaded to {snap} in {dl_s:.0f} s; verifying (E1)")
            res = verify(model, snap)
            res.update(hub_meta=meta, download_s=dl_s,
                       download_rate_MBps=(res["total_weight_bytes"] / 1e6 / dl_s) if dl_s > 0 else None,
                       disk_free_gb_after=disk_free_gb())
            h = write_json(RES_DIR / f"E1_{s}.json", res)
            dt = res["safetensors_dtypes"]
            msg = (f"download+E1 {model}@{rev[:12]}: {len(res['shards'])} safetensors shards "
                   f"{res['total_weight_bytes']/1e9:.2f} GB in {dl_s:.0f} s ({res['download_rate_MBps']:.0f} MB/s), "
                   f"shard sha256 recomputed vs Hub LFS: {'ALL MATCH' if res['shards_sha256_all_match'] else 'MISMATCH'}"
                   f", small files {'match' if res['small_files_all_match'] else 'MISMATCH'} ({res['verify_s']:.0f} s); "
                   f"weight dtype {dt}, legacy rotary_emb.inv_freq buffers {res['legacy_inv_freq_dtypes']}; "
                   f"no *.bin/*.py: {not res['forbidden_present']}; E1 {'PASS' if res['E1_pass'] else 'FAIL'}; "
                   f"E1 json sha256 {h}")
            log(msg)
            append_status(msg, f"results/S1_2_layermax_llama/extraction/E1_{s}.json, logs/llama_axis_download.log")
            if not res["E1_pass"]:
                marker_bad.write_text(log_time() + " E1 failed\n")
                raise SystemExit(3)
            marker_ok.write_text(log_time() + "\n")
        except StopRun as e:
            msg = f"STOP download {model}: {e}"
            log(msg)
            append_status(msg, "logs/llama_axis_download.log")
            marker_bad.write_text(log_time() + " " + str(e) + "\n")
            raise SystemExit(2)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            log(f"FAILED {model}: {e!r}\n{tb}")
            append_status(f"FAILED download {model}: {e!r}"[:400], "logs/llama_axis_download.log")
            marker_bad.write_text(log_time() + "\n" + tb)
            raise SystemExit(1)
    log("download finished")


def log_time():
    from llama_axis_common import utcnow
    return utcnow()


if __name__ == "__main__":
    main()
