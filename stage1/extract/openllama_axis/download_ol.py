"""S1 (registry re-resolution) and S3 (download + E1) for openlm-research/open_llama_13b.

OL-AXIS COPY (role oll-axis-extract, S0b, 2026-10-09) of extract/llama_axis/download_llama_axis.py (sha256 7e5e2b81...).
Plan: analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md v1.0 (sha256 4f18d710...), sections 1, 2 and 11.
Changes, all marked "OL-AXIS":
  * a new sub-command `resolve` (plan step S1): HfApi().model_info(..., token=False, files_metadata=True) for the three
    OpenLLaMA repos and HfApi().list_repo_refs(..., token=False) for 13b; every Hub fact of plan section 1 and every
    pin of registry_openllama_axis.py is compared; any difference, gating, privacy, licence change or auth error STOPS
    (exit 2). Writes results/S1_2_layermax_openllama/extraction/S1_registry.json.
  * `download` (plan step S3) requires: the S0b CODE FROZEN line in STATUS.md, S1 passed, E2 passed for 3b and 7b.
    allow_patterns = exactly the 3 .bin shards, the index and the 6 small files (no *.py, no *.safetensors, no
    .gitattributes). The free-disk STOP (>= 40 GB) is checked before and polled during the download.
  * E1 for .bin pickles: every shard's sha256 recomputed = Hub LFS sha256 (one re-download of a mismatching file with
    force_download=True, then STOP); small files size + git blob id (tokenizer.model: sha256); every shard loaded on CPU
    with torch.load(weights_only=True, mmap=True), dtypes recorded per shard, legacy rotary inv_freq buffers listed
    separately (the LLaMA-axis E1 checker had a bug on exactly these: here they are counted in their own field and
    never mixed with the weight dtypes).
  * token=False on every Hub call; HF_TOKEN / HUGGING_FACE_HUB_TOKEN are removed from the environment; no token is
    configured or created. STOP (exit 2, STATUS line, no retry) on any 401/403, gated repo, licence/terms prompt.

Usage (always through analysis/s12_layermax_openllama/launch_ol.sh; CPU + network only, no GPU):
  launch_ol.sh oll_S1_resolve 1 -- venv/bin/python extract/openllama_axis/download_ol.py resolve
  launch_ol.sh oll_S3_download 2 -- venv/bin/python extract/openllama_axis/download_ol.py download
  launch_ol.sh oll_E1_verify 2 -- venv/bin/python extract/openllama_axis/download_ol.py verify   (re-run E1 only)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

from ol_axis_common import (RES_DIR, LOGS, R, ROOT, STATUS, append_status, check_plan_unchanged, log,  # noqa: E402
                            sha256_file, write_json, ResourceWatch, disk_free_gb_home, MIN_FREE_DISK_GB)  # OL-AXIS
from common import HUB  # noqa: E402

MODEL13 = "openlm-research/open_llama_13b"  # OL-AXIS
CODE_FROZEN_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \[oll-axis-[A-Za-z0-9-]+\] CODE FROZEN")  # OL-AXIS


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


LEGACY_BUFFER_SUFFIX = "rotary_emb.inv_freq"  # legacy per-layer buffer in old HF LLaMA checkpoints (not a weight)
_TORCH2ST = None  # OL-AXIS


def bin_shard_dtypes(path: Path) -> tuple[dict, dict, int]:  # OL-AXIS: replaces safetensors_header_dtypes
    """(weight dtype counts, legacy-buffer dtype counts, n_tensors) of one .bin shard, loaded on the CPU with
    torch.load(weights_only=True, mmap=True). Keys ending in rotary_emb.inv_freq are legacy buffers: transformers 5.15
    does not load them (inv_freq is recomputed at init), so they are recorded apart from the weights."""
    import torch
    m = {torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32", torch.float64: "F64"}
    sd = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    c, leg = {}, {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            raise RuntimeError(f"non-tensor entry {k!r} in {path.name}")
        tgt = leg if k.endswith(LEGACY_BUFFER_SUFFIX) else c
        d = m.get(v.dtype, str(v.dtype))
        tgt[d] = tgt.get(d, 0) + 1
    n = len(sd)
    del sd
    return c, leg, n


def code_frozen() -> bool:  # OL-AXIS
    return any(CODE_FROZEN_RE.match(ln) for ln in open(STATUS, encoding="utf-8").read().splitlines())


def _read(path: Path):
    import json
    return json.load(open(path)) if Path(path).exists() else None


# ------------------------------------------------------------------------------------------------ OL-AXIS: S1 resolve
def cmd_resolve(args):  # OL-AXIS
    from huggingface_hub import HfApi
    api = HfApi()
    res = dict(step="S1", rule="every Hub fact of plan section 1 and every registry pin re-observed with token=False; "
                               "any difference, gating, privacy, licence change or auth error stops the role",
               models={})
    problems = []
    try:
        for model in R.LM_OL_AXIS:
            rev = R.LM_REVISIONS[model]
            info = api.model_info(model, token=False, files_metadata=True)
            info_rev = api.model_info(model, revision=rev, token=False, files_metadata=True)
            card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
            lic = None
            if card is not None:
                cd = card.to_dict() if hasattr(card, "to_dict") else (dict(card) if isinstance(card, dict) else {})
                lic = cd.get("license", getattr(card, "license", None))
            sib = {s.rfilename: dict(size=s.size, blob_id=s.blob_id,
                                     lfs_sha256=(s.lfs.sha256 if getattr(s, "lfs", None) else None))
                   for s in (info_rev.siblings or [])}
            mrec = dict(main_sha=info.sha, pinned=rev, pinned_sha=info_rev.sha, gated=info.gated, private=info.private,
                        disabled=getattr(info, "disabled", None), last_modified=str(getattr(info, "last_modified", None)),
                        license=lic, siblings=sib,
                        safetensors_siblings=[f for f in sib if f.endswith(".safetensors")],
                        py_siblings=[f for f in sib if f.endswith(".py")])
            if info.gated not in (False, None) or info.private or getattr(info, "disabled", False):
                problems.append(f"{model}: gated/private/disabled {info.gated}/{info.private}/{getattr(info, 'disabled', None)}")
            if info_rev.sha != rev:
                problems.append(f"{model}: pinned revision resolves to {info_rev.sha} != {rev}")
            if model == MODEL13 and lic != R.HUB_FACTS_13B["license"]:  # 3b/7b: recorded only (already extracted)
                problems.append(f"{model}: card licence {lic!r} != 'apache-2.0'")
            # weights at the pin = registry
            for fname, size, lfs in R.WEIGHT_SHARDS[model]:
                s = sib.get(fname)
                if s is None or s["size"] != size or s["lfs_sha256"] != lfs:
                    problems.append(f"{model}: shard {fname} Hub {s} != registry ({size}, {lfs})")
            if model == MODEL13:
                f = R.HUB_FACTS_13B
                if info.sha != f["sha"]:
                    problems.append(f"{model}: main {info.sha} != plan {f['sha']}")
                if not mrec["last_modified"].startswith(f["lastModified"].replace("T", " ")) and \
                        not mrec["last_modified"].startswith(f["lastModified"]):
                    problems.append(f"{model}: lastModified {mrec['last_modified']} != plan {f['lastModified']}")
                for fname, size, bid in R.SMALL_FILES[model]:
                    s = sib.get(fname)
                    want_id = s["lfs_sha256"] if (s and fname == "tokenizer.model") else (s["blob_id"] if s else None)
                    if s is None or s["size"] != size or want_id != bid:
                        problems.append(f"{model}: small file {fname} Hub {s} != registry ({size}, {bid})")
                extra = sorted(set(sib) - set(R.ALLOW_PATTERNS) - {".gitattributes"})
                if extra:
                    problems.append(f"{model}: files at the pin not in the registry: {extra}")
                if mrec["safetensors_siblings"] or mrec["py_siblings"]:
                    problems.append(f"{model}: safetensors/py files present {mrec['safetensors_siblings']} "
                                    f"{mrec['py_siblings']}")
                refs = api.list_repo_refs(model, token=False)
                br = sorted((b.name, b.target_commit) for b in refs.branches)
                conv = sorted(c.name for c in (getattr(refs, "converts", None) or []))
                tags = sorted(t.name for t in (refs.tags or []))
                mrec.update(branches=br, converts=conv, tags=tags)
                if [b for b, _ in br] != R.HUB_FACTS_13B["refs"] or dict(br).get("main") != rev:
                    problems.append(f"{model}: refs {br} != plan (main only, at {rev})")
                if conv:
                    problems.append(f"{model}: safetensors conversion refs exist {conv}")
                mrec["total_weight_bytes_hub"] = int(sum(sib[f]["size"] for f, _, _ in R.WEIGHT_SHARDS[model]))
                mrec["total_repo_bytes_hub"] = int(sum(v["size"] for v in sib.values()))  # OL-AXIS
                if mrec["total_weight_bytes_hub"] != R.WEIGHTS_TOTAL_BYTES_13B:
                    problems.append(f"{model}: shard bytes {mrec['total_weight_bytes_hub']} != registry "
                                    f"{R.WEIGHTS_TOTAL_BYTES_13B}")
                if mrec["total_repo_bytes_hub"] != R.REPO_TOTAL_BYTES_13B:  # OL-AXIS: plan section 1 total
                    problems.append(f"{model}: repo bytes {mrec['total_repo_bytes_hub']} != plan {R.REPO_TOTAL_BYTES_13B}")
            res["models"][model] = mrec
            log(f"S1 {model}: main={info.sha[:12]} pin={info_rev.sha[:12]} gated={info.gated} private={info.private} "
                f"licence={lic} files={len(sib)}")
    except Exception as e:  # noqa: BLE001
        if is_auth_error(e):
            problems.append(f"auth/gating/licence error: {e!r}")
            res["auth_error"] = repr(e)
        else:
            raise
    # ranks and family (registry; plan section 1)
    res.update(arh_rank={m: R.ARH_RANK[m] for m in R.POOLED_LMS}, dense_rank=R.DENSE_RANK, family=R.FAMILY,
               registry_sha256=sha256_file(Path(R.__file__)), problems=problems, pass_=not problems)
    res["pass"] = res.pop("pass_")
    h = write_json(RES_DIR / "S1_registry.json", res)
    m13 = res["models"].get(MODEL13, {})
    msg = (f"S1 registry (extract/openllama_axis/registry_openllama_axis.py sha256 {res['registry_sha256']}): Hub "
           f"re-resolved with token=False on ds2213: open_llama_13b main={str(m13.get('main_sha'))[:12]} (pin "
           f"{R.LM_REVISIONS[MODEL13][:12]}), gated={m13.get('gated')}, private={m13.get('private')}, "
           f"disabled={m13.get('disabled')}, licence={m13.get('license')}, refs={m13.get('branches')}, converts="
           f"{m13.get('converts')}, 3 .bin shards {m13.get('total_weight_bytes_hub')} B (all files at the pin "
           f"{m13.get('total_repo_bytes_hub')} B = plan section 1 total) with LFS sha256 = registry, 7 small "
           f"files = registry; 3b/7b pins resolve; ARH rank open_llama_13b = {R.ARH_RANK[MODEL13]}, family OpenLLaMA; "
           f"{'PASS' if res['pass'] else 'STOP: ' + '; '.join(problems)}; S1 json sha256 {h}")
    log(msg)
    append_status(msg, "results/S1_2_layermax_openllama/extraction/S1_registry.json")
    if not res["pass"]:
        raise SystemExit(2)


# --------------------------------------------------------------------------------------------------------- E1 verify
def verify(model: str, snap: Path) -> dict:
    rev = R.LM_REVISIONS[model]
    out = dict(model=model, revision=rev, snapshot_dir=str(snap), snapshot_name_ok=(snap.name == rev))
    files = sorted(p.name for p in snap.iterdir())
    out["snapshot_files"] = files
    # OL-AXIS: anything outside the allow list is forbidden (no *.py, no *.safetensors, no .gitattributes)
    out["forbidden_present"] = [f for f in files if f not in R.ALLOW_PATTERNS]  # OL-AXIS
    out["missing"] = [f for f in R.ALLOW_PATTERNS if f not in files]  # OL-AXIS
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
            tl = time.time()  # OL-AXIS: weights_only + mmap load on CPU (plan section 2, E1)
            try:
                d, lg, n = bin_shard_dtypes(p)
                r.update(weights_only_mmap_load="ok", dtypes=d, legacy_inv_freq_dtypes=lg, n_tensors=n,
                         load_s=time.time() - tl)
            except Exception as e:  # noqa: BLE001
                r.update(weights_only_mmap_load=f"FAILED: {e!r}", dtypes={}, legacy_inv_freq_dtypes={}, ok=False)
                d, lg = {}, {}
            for k, v in d.items():
                dtypes[k] = dtypes.get(k, 0) + v
            for k, v in lg.items():
                legacy[k] = legacy.get(k, 0) + v
        else:
            r["ok"] = False
        shards.append(r)
        log(f"  E1 {model} {fname}: ok={r['ok']} size={r.get('size')} sha256={r.get('sha256', '')[:16]}... "
            f"load={r.get('weights_only_mmap_load')} dtypes={r.get('dtypes')} legacy={r.get('legacy_inv_freq_dtypes')}")
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
    out.update(shards=shards, small_files=smalls, weight_dtypes=dtypes, legacy_inv_freq_dtypes=legacy,
               verify_s=time.time() - t0, total_weight_bytes=int(sum(s.get("size", 0) for s in shards)))
    out["shards_sha256_all_match"] = bool(all(s["ok"] for s in shards))
    out["small_files_all_match"] = bool(all(s["ok"] for s in smalls))
    out["all_shards_weights_only_load_ok"] = bool(all(s.get("weights_only_mmap_load") == "ok" for s in shards))
    out["weights_single_16bit_dtype"] = bool(len(dtypes) == 1 and next(iter(dtypes)) in ("F16", "BF16"))
    out["E1_rule"] = ("snapshot dir = pinned commit; exactly the allow-listed files (no *.py/*.safetensors); every shard "
                      "size + recomputed sha256 = Hub LFS; every shard loads with torch.load(weights_only=True, "
                      "mmap=True) on CPU; small files size + git blob id (tokenizer.model sha256); all weight tensors "
                      "one 16-bit dtype (legacy per-layer rotary_emb.inv_freq buffers recorded separately, not weights)")
    out["E1_pass"] = bool(out["snapshot_name_ok"] and not out["forbidden_present"] and not out["missing"]
                          and out["shards_sha256_all_match"] and out["small_files_all_match"]
                          and out["all_shards_weights_only_load_ok"] and out["weights_single_16bit_dtype"])
    return out


def _snap_dir(model: str) -> Path:
    return HUB / ("models--" + model.replace("/", "--")) / "snapshots" / R.LM_REVISIONS[model]


def _e1_status(model: str, res: dict, dl_s: float | None, h: str, retried: list) -> str:
    return (f"S3 download+E1 {model}@{R.LM_REVISIONS[model][:12]}: {len(res['shards'])} .bin shards "
            f"{res['total_weight_bytes']/1e9:.2f} GB" + (f" in {dl_s:.0f} s ({res['total_weight_bytes']/1e6/dl_s:.0f} MB/s)"
                                                       if dl_s else "") +
            f"; shard sha256 recomputed vs Hub LFS: {'ALL MATCH' if res['shards_sha256_all_match'] else 'MISMATCH'}"
            f" (re-downloaded once: {retried or 'none'}); every shard torch.load(weights_only=True, mmap=True) on CPU: "
            f"{'ok' if res['all_shards_weights_only_load_ok'] else 'FAILED'}; weight dtypes {res['weight_dtypes']}, "
            f"legacy rotary_emb.inv_freq buffers {res['legacy_inv_freq_dtypes']} (separate); small files "
            f"{'match' if res['small_files_all_match'] else 'MISMATCH'}; files outside the allow list "
            f"{res['forbidden_present'] or 'none'}; free disk {disk_free_gb_home():.0f} GB; E1 "
            f"{'PASS' if res['E1_pass'] else 'FAIL -> STOP'}; E1 json sha256 {h}")


def cmd_download(args):
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    model = MODEL13
    # OL-AXIS: order gates (plan section 11): S0b CODE FROZEN, S1 passed, E2 passed for 3b and 7b
    if not code_frozen():
        raise SystemExit("S0b CODE FROZEN line missing in STATUS.md; S3 not started")
    s1 = _read(RES_DIR / "S1_registry.json")
    if not (s1 and s1.get("pass")):
        raise SystemExit("S1 has not passed; S3 not started")
    for sz in ("3b", "7b"):
        e2 = _read(RES_DIR / f"E2_open_llama_{sz}.json")
        if not (e2 and e2.get("pass")):
            raise SystemExit(f"E2 has not passed for open_llama_{sz}; S3 not started (plan section 2: E2 before download)")
    api = HfApi()
    rev = R.LM_REVISIONS[model]
    s = R.slug(model)
    marker_ok = LOGS / f"oll_download.{s}.done"
    marker_bad = LOGS / f"oll_download.{s}.failed"
    for mk in (marker_ok, marker_bad):
        if mk.exists():
            mk.unlink()
    try:
        info = api.model_info(model, token=False)
        info_rev = api.model_info(model, revision=rev, token=False)
        meta = dict(main_sha=info.sha, pinned=rev, pinned_sha=info_rev.sha, gated=info.gated, private=info.private,
                    disabled=getattr(info, "disabled", None))
        log(f"{model}: {meta}")
        if info.gated not in (False, None) or info.private or info_rev.sha != rev:
            raise StopRun(f"metadata check failed: {meta}")
        need_gb = R.WEIGHTS_TOTAL_BYTES_13B / 1e9
        free = disk_free_gb_home()
        if free < need_gb + MIN_FREE_DISK_GB:
            raise StopRun(f"disk free {free:.1f} GB < need {need_gb:.1f} + {MIN_FREE_DISK_GB} GB (plan STOP)")
        t0 = time.time()
        p = None
        last = None
        with ResourceWatch(min_ram_gb=0.0, label="S3 download (free disk)"):  # OL-AXIS: disk STOP during download
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
        retried = []
        bad = [sh["file"] for sh in res["shards"] if not sh["ok"] and sh.get("exists")]
        if bad:  # OL-AXIS: plan section 2, "an LFS sha256 mismatch after one retry" stops the role
            for fname in bad:
                log(f"{model}: {fname} failed E1; one re-download with force_download=True")
                hf_hub_download(model, fname, revision=rev, token=False, force_download=True)
                retried.append(fname)
            res = verify(model, snap)
        res.update(hub_meta=meta, download_s=dl_s, retried=retried, disk_free_gb_after=disk_free_gb_home())
        h = write_json(RES_DIR / f"E1_{s}.json", res)
        msg = _e1_status(model, res, dl_s, h, retried)
        log(msg)
        append_status(msg, f"results/S1_2_layermax_openllama/extraction/E1_{s}.json, HF cache {snap}")
        if not res["E1_pass"]:
            marker_bad.write_text(_t() + " E1 failed\n")
            raise SystemExit(3)
        marker_ok.write_text(_t() + "\n")
    except StopRun as e:
        msg = f"STOP download {model}: {e}"
        log(msg)
        append_status(msg)
        marker_bad.write_text(_t() + " " + str(e) + "\n")
        raise SystemExit(2)


def cmd_verify(args):  # OL-AXIS: re-run E1 on the existing snapshot (no network)
    model = MODEL13
    snap = _snap_dir(model)
    res = verify(model, snap)
    res.update(retried=[], note="E1 re-run on the existing snapshot (verify sub-command)")
    h = write_json(RES_DIR / f"E1_{R.slug(model)}.verify.json", res)
    msg = _e1_status(model, res, None, h, [])
    log(msg)
    append_status("E1 re-run: " + msg)
    if not res["E1_pass"]:
        raise SystemExit(3)


def _t():
    from ol_axis_common import utcnow
    return utcnow()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["resolve", "download", "verify"])
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=3)
    args = ap.parse_args()
    check_plan_unchanged()
    log(f"download_ol {args.cmd} pid={os.getpid()} allow={R.ALLOW_PATTERNS} "
        f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET')}")
    try:
        {"resolve": cmd_resolve, "download": cmd_download, "verify": cmd_verify}[args.cmd](args)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"FAILED {args.cmd}: {e!r}\n{tb}")
        append_status(f"FAILED download_ol {args.cmd}: {e!r}"[:400])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
