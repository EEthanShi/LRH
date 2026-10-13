"""S1 (registry re-resolution), S3 (download + E1, F-cache) for the 7 Pythia sizes and the 2.8b step143000 file.

PYT-AXIS COPY (role pyt-axis-extract, S0b, 2026-10-10) of extract/openllama_axis/download_ol.py (sha256 93c11d61...).
Plan: analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md v1.0 (sha256 6de89603...), sections 1, 2 and 11.
Changes, all marked "PYT-AXIS":
  * every Hub call and every download runs with the dedicated HF_HOME=extract/pythia_axis/hf_home (hub cache
    hf_home/hub, Xet cache hf_home/xet), HF_XET_CHUNK_CACHE_SIZE_BYTES=0, HF_HUB_DISABLE_TELEMETRY=1, token=False
    passed explicitly; HF_TOKEN / HUGGING_FACE_HUB_TOKEN are removed from the environment; the standard cache is never
    written; no token is read, configured or created. STOP (exit 2, STATUS line, no retry) on any 401/403, gated
    repo, licence/terms prompt or token request.
  * `resolve` (S1): for the 7 repos, model_info at main, at the pin and at the step143000 commit (files_metadata),
    list_repo_refs, and config.json at the pin read into memory (never written): every Hub fact and every registry
    value of plan section 1 is compared (gated/private/disabled, licence, card dataset, main = pin, 155 branches,
    step143000 commit, converts = [], weight sizes and LFS sha256, small-file sizes and blob ids, main .bin = step143000
    .bin for the 6 sizes, the 2.8b step143000 files, the F16 parameter count minus the per-layer rotary buffers = the
    expected num_params, the config architecture); any difference STOPS (exit 2).
  * `download` (S3) requires the S0b CODE FROZEN line, S1 passed, E2 run (descriptive), the "[pyt-axis-<role>] S3 GO"
    line citing G0 and the authorisation (and not followed by a DEFERRED line) and the G0 record = GO. Order: smallest
    first, 2.8b main then step143000, then 6.9b. Before each download: free - bytes(this) - reserve >= 20 GB, reserve =
    bytes of every later download + 3.0 GB features + 3 GB results (statvfs on the dedicated HF_HOME); else STOP. The
    disk is also polled every second during the download (STOP below 20 GB). allow_patterns = exactly the section 1
    file names of that repo and revision. The F-cache listing is written before the first download.
  * E1 per snapshot: exactly the allow-list; weight size + recomputed sha256 = Hub LFS (one re-download of a
    mismatching file, then STOP); small files size + git blob id; every .bin loads with torch.load(weights_only=True,
    mmap=True) on CPU; tensor inventory (name, dtype, shape); non-buffer tensors all F16; parameter count without
    buffer names vs plan section 1 (difference reported); config.json architecture = plan section 1 (STOP keys).
  * After the 2.8b step143000 E1, E1-PROV (analysis/s12_layermax_pythia/e1prov_pyt.py) runs in this process and logs
    the 2.8b source decision before anything else.
  * no-delete rule: markers are moved aside, never unlinked.
  * S0b test mode (PYT_S0B_TEST=1, before CODE FROZEN only): the network is refused; `resolve` needs --mock-hub (the
    designer's metadata JSON) and `download` needs --fake-repos (a local directory of synthetic repos).

Usage (always through analysis/s12_layermax_pythia/launch_pyt.sh; CPU + network only, no GPU):
  launch_pyt.sh pyt_S1_resolve 1 -- venv/bin/python extract/pythia_axis/download_pyt.py resolve
  launch_pyt.sh pyt_S3_download 2 -- venv/bin/python extract/pythia_axis/download_pyt.py download
  launch_pyt.sh pyt_E1_verify 2 -- venv/bin/python extract/pythia_axis/download_pyt.py verify --model EleutherAI/pythia-70m
  launch_pyt.sh pyt_fcache 1 -- venv/bin/python extract/pythia_axis/download_pyt.py fcache --phase before_S3
"""
from __future__ import annotations

import argparse
import hashlib
import json  # PYT-AXIS
import os
import shutil  # PYT-AXIS
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyt_axis_common import (HF_HOME, PYT_HUB, STD_HUB, RES_DIR, LOGS_DIR, R, TEST_MODE, TEST_ROOT,  # noqa: E402
                             append_status, check_plan_unchanged, log, sha256_file, write_json, read_json, require,
                             ResourceWatch, disk_free_bytes, disk_free_gb, MIN_FREE_DISK_GB, RESERVE_FEATURES_GB,
                             RESERVE_RESULTS_GB, code_frozen_line, s3_go_line, move_marker_aside, utcnow)

# PYT-AXIS: dedicated Hugging Face home, set before huggingface_hub is imported anywhere in this process
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(PYT_HUB)
os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
os.environ.pop("HF_HUB_OFFLINE", None)  # S1/S3 need the Hub; extraction (run_pyt.py) sets HF_HUB_OFFLINE=1
if TEST_MODE:  # PYT-AXIS: no network in S0b test mode
    os.environ["HF_HUB_OFFLINE"] = "1"

EXIT_STOP = 2
EXIT_E1 = 3


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
    return any(x in s for x in ("401", "403", "gated", "Gated", "license", "terms", "authenticat", "token"))


def git_blob_sha1(data_or_path) -> str:  # PYT-AXIS
    data = data_or_path if isinstance(data_or_path, (bytes, bytearray)) else Path(data_or_path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _rel(p) -> str:  # PYT-AXIS
    return str(p).split("hrh_v2_stage1_20261008/")[-1]


# ------------------------------------------------------------------------------------------- PYT-AXIS: config check
def check_config(model: str, cfg: dict) -> dict:  # PYT-AXIS: plan section 1 architecture (S1 and E1)
    exp = R.EXPECTED[model]
    stop = {k: dict(config=cfg.get(k), expected=exp[v], ok=cfg.get(k) == exp[v]) for k, v in R.CONFIG_STOP_KEYS.items()}
    rec = {k: cfg.get(k) for k in R.CONFIG_RECORD_KEYS}
    try:
        n_cfg = R.expected_num_params_from_config(cfg)
    except Exception as e:  # noqa: BLE001
        n_cfg = repr(e)
    return dict(stop_keys=stop, recorded=rec, num_params_from_config=n_cfg,
                num_params_expected=exp["num_params"], num_params_from_config_ok=(n_cfg == exp["num_params"]),
                pass_=bool(all(v["ok"] for v in stop.values()) and n_cfg == exp["num_params"]))


def fetch_config_bytes(model: str, rev: str) -> bytes:  # PYT-AXIS: S1 reads config.json into memory, never to disk
    if TEST_MODE:
        raise StopRun("network refused in S0b test mode")
    from huggingface_hub import hf_hub_url  # PYT-AXIS
    from huggingface_hub.utils import build_hf_headers, get_session  # PYT-AXIS
    url = hf_hub_url(model, "config.json", revision=rev)
    r = get_session().get(url, headers=build_hf_headers(token=False), follow_redirects=True, timeout=60)
    if r.status_code in (401, 403):
        raise StopRun(f"{model} config.json: HTTP {r.status_code} (auth/gating)")
    r.raise_for_status()
    return r.content


# ------------------------------------------------------------------------------------------------ PYT-AXIS: S1 resolve
def _license(info):  # PYT-AXIS
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    if card is None:
        return None, None
    cd = card.to_dict() if hasattr(card, "to_dict") else (dict(card) if isinstance(card, dict) else {})
    return cd.get("license", getattr(card, "license", None)), cd.get("datasets")


def _sib(info) -> dict:  # PYT-AXIS
    return {s.rfilename: dict(size=s.size, blob_id=s.blob_id,
                              lfs_sha256=((s.lfs.sha256 if hasattr(s.lfs, "sha256") else s.lfs.get("sha256"))
                                          if getattr(s, "lfs", None) else None))
            for s in (info.siblings or [])}


def resolve_all(api, fetch_config=fetch_config_bytes) -> dict:  # PYT-AXIS
    res = dict(step="S1", rule=("every Hub fact of plan section 1 and every registry value re-observed with token=False; "
                                "any difference, gating, privacy, licence change or auth error stops the role"),
               models={})
    problems = []
    try:
        for model in R.LM_PYT_AXIS:
            pin, step = R.LM_REVISIONS[model], R.STEP143000[model]
            info = api.model_info(model, token=False, files_metadata=True)
            info_pin = api.model_info(model, revision=pin, token=False, files_metadata=True)
            info_step = api.model_info(model, revision=step, token=False, files_metadata=True)
            lic, dsets = _license(info)
            sib, sib_step = _sib(info_pin), _sib(info_step)
            st = getattr(info, "safetensors", None)
            st_params = dict(getattr(st, "parameters", None) or {}) if st is not None else {}
            mrec = dict(main_sha=info.sha, pinned=pin, pinned_sha=info_pin.sha, step143000=step,
                        step143000_sha=info_step.sha, gated=info.gated, private=info.private,
                        disabled=getattr(info, "disabled", None), last_modified=str(getattr(info, "last_modified", None)),
                        license=lic, datasets=dsets, safetensors_parameters=st_params, siblings=sib,
                        siblings_step143000=sib_step, py_siblings=[f for f in sib if f.endswith(".py")])
            f = R.HUB_FACTS
            if info.gated not in (False, None) or info.private or getattr(info, "disabled", False):
                problems.append(f"{model}: gated/private/disabled {info.gated}/{info.private}/{getattr(info, 'disabled', None)}")
            if info.sha != pin:
                problems.append(f"{model}: main {info.sha} != pin {pin}")
            if info_pin.sha != pin or info_step.sha != step:
                problems.append(f"{model}: revisions resolve to {info_pin.sha}/{info_step.sha} != {pin}/{step}")
            if lic != f["license"]:
                problems.append(f"{model}: card licence {lic!r} != {f['license']!r}")
            if dsets != R.CARD_DATASETS[model]:  # PYT-AXIS: observed per-repo values (registry erratum note)
                problems.append(f"{model}: card datasets {dsets!r} != registry {R.CARD_DATASETS[model]!r}")
            for fname, size, lfs in R.WEIGHT_FILES[model]:
                s = sib.get(fname)
                if s is None or s["size"] != size or s["lfs_sha256"] != lfs:
                    problems.append(f"{model}: weight {fname} Hub {s} != registry ({size}, {lfs})")
            for fname, size, bid in R.SMALL_FILES[model]:
                s = sib.get(fname)
                if s is None or s["size"] != size or s["blob_id"] != bid:
                    problems.append(f"{model}: small file {fname} Hub {s} != registry ({size}, {bid})")
            # provenance rule (plan section 1): main .bin = step143000 .bin for 6 sizes; 2.8b step143000 files
            if model in R.MAIN_EQUALS_STEP:
                for fname, size, lfs in R.WEIGHT_FILES[model]:
                    s = sib_step.get(fname)
                    if s is None or s["size"] != size or s["lfs_sha256"] != lfs:
                        problems.append(f"{model}: main .bin {fname} != step143000 file {s} (provenance rule)")
            else:
                for fname, size, lfs in R.PROV_WEIGHT_FILES:
                    s = sib_step.get(fname)
                    if s is None or s["size"] != size or s["lfs_sha256"] != lfs:
                        problems.append(f"{model}: step143000 {fname} Hub {s} != registry ({size}, {lfs})")
                for fname, size, bid in R.PROV_SMALL_FILES:
                    s = sib_step.get(fname)
                    if s is None or s["size"] != size or s["blob_id"] != bid:
                        problems.append(f"{model}: step143000 small file {fname} Hub {s} != registry ({size}, {bid})")
            if mrec["py_siblings"]:
                problems.append(f"{model}: *.py files at the pin {mrec['py_siblings']} (never downloaded)")
            refs = api.list_repo_refs(model, token=False)
            br = {b.name: b.target_commit for b in refs.branches}
            conv = sorted(getattr(c, "ref", getattr(c, "name", str(c))) for c in (getattr(refs, "converts", None) or []))
            mrec.update(n_branches=len(br), main_branch=br.get("main"), step143000_branch=br.get("step143000"),
                        converts=conv)
            if len(br) != f["n_branches"] or br.get("main") != pin or br.get("step143000") != step:
                problems.append(f"{model}: refs n={len(br)} main={br.get('main')} step143000={br.get('step143000')} "
                                f"!= plan (155, {pin}, {step})")
            if conv != f["converts"]:
                problems.append(f"{model}: converts {conv} != []")
            # expected num_params = Hub F16 parameter count - per-layer rotary inv_freq entries (plan section 1)
            exp = R.EXPECTED[model]
            head_dim = exp["hidden_size"] // exp["n_heads"]
            rot = exp["n_layers"] * int(head_dim * 0.25) // 2  # rotary_pct 0.25, checked on config.json below
            mrec["f16_minus_rotary"] = (st_params.get("F16") - rot) if st_params.get("F16") is not None else None
            if mrec["f16_minus_rotary"] != exp["num_params"]:
                problems.append(f"{model}: Hub F16 count {st_params.get('F16')} - {rot} != expected num_params "
                                f"{exp['num_params']}")
            # config.json at the pin, read into memory
            cfg_bytes = fetch_config(model, pin)
            if cfg_bytes is None:  # S0b mock only
                mrec["config"] = "not fetched (S0b mock)"
                if not TEST_MODE:
                    problems.append(f"{model}: config.json not fetched")
            else:
                cid = git_blob_sha1(cfg_bytes)
                cfg = json.loads(cfg_bytes)
                cc = check_config(model, cfg)
                mrec["config"] = dict(blob_id=cid, blob_ok=cid == dict((a, c) for a, _, c in R.SMALL_FILES[model])["config.json"],
                                      check=cc, raw=cfg)
                if not mrec["config"]["blob_ok"] or not cc["pass_"] or cfg.get("rotary_pct") != 0.25:
                    problems.append(f"{model}: config.json blob {cid} / check {cc} / rotary_pct {cfg.get('rotary_pct')}")
            res["models"][model] = mrec
            log(f"S1 {model}: main={info.sha[:12]} step143000={info_step.sha[:12]} gated={info.gated} "
                f"private={info.private} licence={lic} files={len(sib)}")
    except StopRun:
        raise
    except Exception as e:  # noqa: BLE001
        if is_auth_error(e):
            problems.append(f"auth/gating/licence error: {e!r}")
            res["auth_error"] = repr(e)
        else:
            raise
    res["download_plan"] = [dict(model=d["model"], revision=d["revision"], kind=d["kind"], bytes=d["bytes"],
                                 allow=d["allow"]) for d in R.DOWNLOADS]
    res["download_total_bytes"] = sum(d["bytes"] for d in R.DOWNLOADS)
    if res["download_total_bytes"] != R.DOWNLOAD_TOTAL_BYTES:
        problems.append(f"download total {res['download_total_bytes']} != plan {R.DOWNLOAD_TOTAL_BYTES}")
    res.update(registry_sha256=sha256_file(Path(R.__file__)), family=R.FAMILY, size_rank=R.SIZE_RANK,
               problems=problems)
    res["pass"] = not problems
    return res


def cmd_resolve(args):  # PYT-AXIS
    require(code_frozen_line() is not None or TEST_MODE, "the S0b CODE FROZEN line is missing in STATUS.md (S1 after S0b)")
    if TEST_MODE:
        require(bool(args.mock_hub), "S0b test mode: resolve needs --mock-hub (no network)")
        from tests_mock_hub import MockApi  # noqa: E402  (tests/ helper, S0b only) PYT-AXIS
        api = MockApi(args.mock_hub, faults=args.mock_fault)
        fetch = api.fetch_config
    else:
        from huggingface_hub import HfApi  # PYT-AXIS
        api = HfApi()
        fetch = fetch_config_bytes
    try:
        res = resolve_all(api, fetch)
    except StopRun as e:
        res = dict(step="S1", problems=[str(e)], models={})
        res["pass"] = False
    h = write_json(RES_DIR / "S1_registry.json", res)
    msg = (f"S1 registry (extract/pythia_axis/registry_pythia_axis.py sha256 {res.get('registry_sha256')}): Hub "
           f"re-resolved with token=False for the 7 Pythia repos: " +
           "; ".join(f"{R.SIZE_OF[m]} main={str(v.get('main_sha'))[:12]} step143000={str(v.get('step143000_sha'))[:12]} "
                     f"gated={v.get('gated')} licence={v.get('license')} branches={v.get('n_branches')} "
                     f"converts={v.get('converts')}" for m, v in res.get("models", {}).items()) +
           f"; download total {res.get('download_total_bytes')} B; "
           f"{'PASS' if res['pass'] else 'STOP: ' + '; '.join(res['problems'])[:1500]}; S1 json sha256 {h}")
    log(msg)
    append_status(msg, _rel(RES_DIR / "S1_registry.json"))
    if not res["pass"]:
        raise SystemExit(EXIT_STOP)


# --------------------------------------------------------------------------------------------------------- E1 verify
def bin_inventory(path: Path) -> tuple[dict, dict]:  # PYT-AXIS: (inventory name -> [dtype, shape], dtype counts)
    import torch  # PYT-AXIS
    m = {torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32", torch.float64: "F64", torch.bool: "BOOL",
         torch.uint8: "U8"}
    sd = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    inv, counts = {}, {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            raise RuntimeError(f"non-tensor entry {k!r} in {path.name}")
        d = m.get(v.dtype, str(v.dtype))
        inv[k] = [d, list(v.shape), int(v.numel())]
        counts[d] = counts.get(d, 0) + 1
    del sd
    return inv, counts


def verify(model: str, rev: str, snap: Path, weights: list, small: list, allow: list) -> dict:  # PYT-AXIS
    out = dict(model=model, revision=rev, snapshot_dir=str(snap), snapshot_name_ok=(snap.name == rev))
    files = sorted(p.name for p in snap.iterdir()) if snap.is_dir() else []
    out["snapshot_files"] = files
    out["forbidden_present"] = [f for f in files if f not in allow]
    out["missing"] = [f for f in allow if f not in files]
    shards, inventory, dtypes = [], {}, {}
    t0 = time.time()
    for fname, size, lfs in weights:
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
            tl = time.time()
            try:
                inv, cnt = bin_inventory(p)
                r.update(weights_only_mmap_load="ok", dtype_counts=cnt, n_tensors=len(inv), load_s=time.time() - tl)
                for k, v in inv.items():
                    inventory[k] = v + [fname]
                for k, v in cnt.items():
                    dtypes[k] = dtypes.get(k, 0) + v
            except Exception as e:  # noqa: BLE001
                r.update(weights_only_mmap_load=f"FAILED: {e!r}", ok=False)
        else:
            r["ok"] = False
        shards.append(r)
        log(f"  E1 {model}@{rev[:12]} {fname}: ok={r['ok']} size={r.get('size')} sha256={r.get('sha256', '')[:16]}... "
            f"load={r.get('weights_only_mmap_load')} dtypes={r.get('dtype_counts')}")
    smalls = []
    for fname, size, bid in small:
        p = snap / fname
        r = dict(file=fname, expected_size=size, expected_id=bid, exists=p.exists())
        if p.exists():
            r["size"] = p.resolve().stat().st_size
            r["id"] = git_blob_sha1(p)
            r["sha256"] = sha256_file(p)
            r["ok"] = bool(r["size"] == size and r["id"] == bid)
        else:
            r["ok"] = False
        smalls.append(r)
    non_buf = {k: v for k, v in inventory.items() if not R.BUFFER_RE.search(k)}
    buf = {k: v for k, v in inventory.items() if R.BUFFER_RE.search(k)}
    nb_dtypes = sorted({v[0] for v in non_buf.values()})
    n_params = int(sum(v[2] for v in non_buf.values()))
    cfg = json.load(open(snap / "config.json")) if (snap / "config.json").exists() else {}
    cc = check_config(model, cfg) if cfg else dict(pass_=False, error="config.json missing")
    idx_ok = None
    if (snap / "pytorch_model.bin.index.json").exists():  # 6.9b: every indexed tensor is in its named shard
        wm = json.load(open(snap / "pytorch_model.bin.index.json")).get("weight_map", {})
        idx_ok = bool(all(k in inventory and inventory[k][3] == f for k, f in wm.items())
                      and set(wm) == set(inventory))
    out.update(shards=shards, small_files=smalls, dtype_counts=dtypes, non_buffer_dtypes=nb_dtypes,
               n_buffer_tensors=len(buf), buffer_dtypes=sorted({v[0] for v in buf.values()}),
               inventory={k: v[:2] for k, v in sorted(inventory.items())},
               n_params_without_buffers=n_params, num_params_expected=R.EXPECTED[model]["num_params"],
               n_params_diff=n_params - R.EXPECTED[model]["num_params"], config_check=cc, index_consistent=idx_ok,
               verify_s=time.time() - t0, total_weight_bytes=int(sum(s.get("size", 0) for s in shards)))
    out["weights_sha256_all_match"] = bool(all(s["ok"] for s in shards))
    out["small_files_all_match"] = bool(all(s["ok"] for s in smalls))
    out["all_weights_only_load_ok"] = bool(all(s.get("weights_only_mmap_load") == "ok" for s in shards))
    out["non_buffer_all_f16"] = bool(nb_dtypes == ["F16"])  # recorded (plan: "F16 weights expected"), not a STOP
    out["E1_rule"] = ("snapshot dir = pinned commit; exactly the allow-listed files; every .bin size + recomputed sha256 "
                      "= Hub LFS; every .bin loads with torch.load(weights_only=True, mmap=True) on CPU; small files "
                      "size + git blob id; tensor inventory recorded; non-buffer dtypes recorded (F16 expected); config architecture = "
                      "plan section 1; parameter count without buffers vs plan section 1 reported (a missing parameter "
                      "stops at the loading check)")
    out["E1_pass"] = bool(out["snapshot_name_ok"] and not out["forbidden_present"] and not out["missing"]
                          and out["weights_sha256_all_match"] and out["small_files_all_match"]
                          and out["all_weights_only_load_ok"] and cc.get("pass_") and idx_ok is not False)
    return out


def snap_dir(model: str, rev: str) -> Path:  # PYT-AXIS
    return PYT_HUB / ("models--" + model.replace("/", "--")) / "snapshots" / rev


def e1_name(item: dict) -> str:  # PYT-AXIS
    return f"E1_{R.slug(item['model'])}" + ("" if item["kind"] == "main" else f".{item['kind']}") + ".json"


def _e1_status(item: dict, res: dict, dl_s, h: str, retried: list) -> str:  # PYT-AXIS
    return (f"S3 download+E1 {item['model']}@{item['revision'][:12]} ({item['kind']}): {len(res['shards'])} .bin "
            f"{res['total_weight_bytes'] / 1e9:.3f} GB" + (f" in {dl_s:.0f} s ({res['total_weight_bytes'] / 1e6 / max(dl_s, 1e-9):.0f} MB/s)"
                                                          if dl_s else "") +
            f"; sha256 recomputed vs Hub LFS: {'ALL MATCH' if res['weights_sha256_all_match'] else 'MISMATCH'} "
            f"({', '.join(s.get('sha256', '?')[:16] + '...' for s in res['shards'])}; re-downloaded once: "
            f"{retried or 'none'}); torch.load(weights_only=True, mmap=True): "
            f"{'ok' if res['all_weights_only_load_ok'] else 'FAILED'}; dtypes {res['dtype_counts']} (non-buffer "
            f"{res['non_buffer_dtypes']}, {res['n_buffer_tensors']} buffer tensors); params without buffers "
            f"{res['n_params_without_buffers']} (plan {res['num_params_expected']}, diff {res['n_params_diff']}); small "
            f"files {'match' if res['small_files_all_match'] else 'MISMATCH'}; config {'ok' if res['config_check'].get('pass_') else 'MISMATCH'}; "
            f"outside allow-list {res['forbidden_present'] or 'none'}; free disk {disk_free_gb():.1f} GB; "
            f"E1 {'PASS' if res['E1_pass'] else 'FAIL -> STOP'}; E1 json sha256 {h}")


def disk_check(i: int, free_bytes: int | None = None) -> dict:  # PYT-AXIS: plan section 11 disk guard
    item = R.DOWNLOADS[i]
    later = sum(d["bytes"] for d in R.DOWNLOADS[i + 1:])
    reserve = later + int(RESERVE_FEATURES_GB * 1e9) + int(RESERVE_RESULTS_GB * 1e9)
    free = disk_free_bytes(HF_HOME) if free_bytes is None else free_bytes
    left = free - item["bytes"] - reserve
    return dict(item=f"{item['model']}@{item['kind']}", free_bytes=free, this_bytes=item["bytes"], later_bytes=later,
                reserve_bytes=reserve, left_bytes=left, min_bytes=int(MIN_FREE_DISK_GB * 1e9),
                ok=bool(left >= int(MIN_FREE_DISK_GB * 1e9)))


# ---------------------------------------------------------------------------------------------- PYT-AXIS: F-cache
def fcache_listing() -> dict:  # plan section 2 F-cache: stat-listing by name only, no content read PYT-AXIS
    top = sorted(os.listdir(STD_HUB)) if STD_HUB.is_dir() else []
    seeds = {}
    for d in R.POLYPYTHIA_DIRS:
        p = STD_HUB / d
        if not p.is_dir():
            seeds[d] = None
            continue
        ents = []
        with os.scandir(p) as it:
            for e in it:
                st = e.stat(follow_symlinks=False)
                ents.append([e.name, int(st.st_size), int(st.st_mtime_ns)])
        seeds[d] = sorted(ents)
    return dict(std_hub=str(STD_HUB), top_level=top, polypythia_top_entries=seeds, utc=utcnow())


def cmd_fcache(args):  # PYT-AXIS
    require(code_frozen_line() is not None or TEST_MODE, "the S0b CODE FROZEN line is missing in STATUS.md")
    phase = args.phase
    lst = fcache_listing()
    h = write_json(RES_DIR / f"fcache_{phase}.json", lst)
    cmp = None
    if phase != "before_S3":
        before = read_json(RES_DIR / "fcache_before_S3.json")
        if before is None:
            cmp = "no before_S3 listing"
        else:
            cmp = dict(top_level_equal=before["top_level"] == lst["top_level"],
                       polypythia_equal=before["polypythia_top_entries"] == lst["polypythia_top_entries"])
    msg = (f"F-cache listing {phase}: standard hub top-level {len(lst['top_level'])} names, 9 PolyPythia seed dirs "
           f"stat-listed by name only ({sum(len(v or []) for v in lst['polypythia_top_entries'].values())} top-level "
           f"entries; no content read)" + (f"; vs before_S3: {cmp}" if cmp is not None else "") + f"; json sha256 {h}")
    log(msg)
    append_status(msg, _rel(RES_DIR / f"fcache_{phase}.json"))
    if isinstance(cmp, dict) and not all(cmp.values()):
        raise SystemExit(EXIT_STOP)


# ---------------------------------------------------------------------------------------------- S3 download
def _real_snapshot_download(item: dict, max_workers: int) -> Path:  # PYT-AXIS
    if TEST_MODE:
        raise StopRun("network refused in S0b test mode")
    from huggingface_hub import snapshot_download, constants  # PYT-AXIS
    assert Path(constants.HF_HUB_CACHE).resolve() == PYT_HUB.resolve(), constants.HF_HUB_CACHE  # dedicated cache
    return Path(snapshot_download(item["model"], revision=item["revision"], allow_patterns=item["allow"], token=False,
                                  cache_dir=str(PYT_HUB), max_workers=max_workers))


def _real_redownload(item: dict, fname: str) -> None:  # PYT-AXIS
    if TEST_MODE:
        raise StopRun("network refused in S0b test mode")
    from huggingface_hub import hf_hub_download  # PYT-AXIS
    hf_hub_download(item["model"], fname, revision=item["revision"], token=False, cache_dir=str(PYT_HUB),
                    force_download=True)


def download_all(downloader, redownloader, attempts: int = 3, max_workers: int = 4, only: str | None = None) -> None:  # PYT-AXIS
    PYT_HUB.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(R.DOWNLOADS):
        if only and item["model"] != only:
            continue
        tag = f"{R.slug(item['model'])}.{item['kind']}"
        e1 = read_json(RES_DIR / e1_name(item))
        if e1 and e1.get("E1_pass") and (LOGS_DIR / f"pyt_download.{tag}.done").exists():
            log(f"{item['model']}@{item['kind']}: already downloaded and E1 passed; skipped")
            continue
        dc = disk_check(i)
        log(f"disk check before {tag}: {dc}")
        if not dc["ok"]:
            raise StopRun(f"disk guard before {tag}: free {dc['free_bytes'] / 1e9:.2f} GB - this "
                          f"{dc['this_bytes'] / 1e9:.2f} GB - reserve {dc['reserve_bytes'] / 1e9:.2f} GB = "
                          f"{dc['left_bytes'] / 1e9:.2f} GB < 20 GB (plan section 11)")
        for mk in (LOGS_DIR / f"pyt_download.{tag}.done", LOGS_DIR / f"pyt_download.{tag}.failed"):
            move_marker_aside(mk)
        t0 = time.time()
        p, last = None, None
        with ResourceWatch(min_ram_gb=0.0, label=f"S3 download {tag} (free disk)"):
            for att in range(1, attempts + 1):
                try:
                    p = downloader(item, max_workers)
                    break
                except StopRun:
                    raise
                except Exception as e:  # noqa: BLE001
                    if is_auth_error(e):
                        raise StopRun(f"auth/gating/licence error, not retried: {e!r}")
                    last = e
                    log(f"{tag}: attempt {att} failed (non-auth): {e!r}")
                    time.sleep(30)
        if p is None:
            raise RuntimeError(f"download {tag} failed after {attempts} attempts: {last!r}")
        dl_s = time.time() - t0
        snap = Path(p)
        require(snap.resolve() == snap_dir(item["model"], item["revision"]).resolve(),
                f"snapshot path {snap} is not the dedicated cache path")
        res = verify(item["model"], item["revision"], snap, item["weights"], item["small"], item["allow"])
        retried = []
        bad = [s["file"] for s in res["shards"] if not s["ok"] and s.get("exists")] + \
              [s["file"] for s in res["small_files"] if not s["ok"] and s.get("exists")]
        if bad:  # plan section 2: a mismatch after one retry stops the role
            for fname in bad:
                log(f"{tag}: {fname} failed E1; one re-download with force_download=True")
                redownloader(item, fname)
                retried.append(fname)
            res = verify(item["model"], item["revision"], snap, item["weights"], item["small"], item["allow"])
        res.update(download_s=dl_s, retried=retried, disk_check=dc, disk_free_gb_after=disk_free_gb())
        h = write_json(RES_DIR / e1_name(item), res)
        msg = _e1_status(item, res, dl_s, h, retried)
        log(msg)
        append_status(msg, f"{_rel(RES_DIR / e1_name(item))}, {_rel(snap)}")
        if not res["E1_pass"]:
            (LOGS_DIR / f"pyt_download.{tag}.failed").write_text(utcnow() + " E1 failed\n")
            raise SystemExit(EXIT_E1)
        (LOGS_DIR / f"pyt_download.{tag}.done").write_text(utcnow() + "\n")
        if item["model"] == R.PROV_MODEL and item["kind"] == "step143000":  # E1-PROV before anything else
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "s12_layermax_pythia"))
            import e1prov_pyt  # PYT-AXIS
            e1prov_pyt.run_e1prov()


def cmd_download(args):  # PYT-AXIS
    if TEST_MODE:
        require(bool(args.fake_repos), "S0b test mode: download needs --fake-repos (no network)")
        from tests_mock_hub import fake_downloader  # noqa: E402  (tests/ helper, S0b only) PYT-AXIS
        dl, redl = fake_downloader(args.fake_repos, PYT_HUB)
    else:
        # PYT-AXIS: order gates (plan section 11): CODE FROZEN, S1 passed, E2 run, S3 GO citing G0, G0 = GO
        require(code_frozen_line() is not None, "the S0b CODE FROZEN line is missing in STATUS.md; S3 not started")
        s1 = read_json(RES_DIR / "S1_registry.json")
        require(bool(s1 and s1.get("pass")), "S1 has not passed; S3 not started")
        require(read_json(RES_DIR / "E2_bloomz-560m.json") is not None, "E2 has not run; S3 not started")
        g0 = read_json(LOGS_DIR / "gpu_gate_G0.json")
        require(bool(g0 and g0.get("decision") == "GO"), "gate G0 has no GO record; S3 not started")
        require(s3_go_line() is not None, "no '[pyt-axis-<role>] S3 GO' line citing G0 and the authorisation (or it "
                                          "is followed by a DEFERRED line); S3 not started (plan R.3 Q0)")
        dl, redl = _real_snapshot_download, _real_redownload
    if read_json(RES_DIR / "fcache_before_S3.json") is None:
        cmd_fcache(argparse.Namespace(phase="before_S3"))
    try:
        download_all(dl, redl, attempts=args.attempts, max_workers=args.max_workers, only=args.only)
    except StopRun as e:
        msg = f"STOP download: {e}"
        log(msg)
        append_status(msg)
        raise SystemExit(EXIT_STOP)
    msg = (f"S3 downloads complete ({len(R.DOWNLOADS)} snapshots, plan total {R.DOWNLOAD_TOTAL_BYTES} B) into "
           f"{_rel(PYT_HUB)}; free disk {disk_free_gb():.1f} GB")
    log(msg)
    append_status(msg)


def cmd_verify(args):  # PYT-AXIS: re-run E1 on an existing snapshot (no network)
    require(code_frozen_line() is not None or TEST_MODE, "the S0b CODE FROZEN line is missing in STATUS.md")
    items = [d for d in R.DOWNLOADS if d["model"] == args.model and (args.kind is None or d["kind"] == args.kind)]
    require(bool(items), f"unknown model/kind {args.model}/{args.kind}")
    for item in items:
        snap = snap_dir(item["model"], item["revision"])
        res = verify(item["model"], item["revision"], snap, item["weights"], item["small"], item["allow"])
        res.update(retried=[], note="E1 re-run on the existing snapshot (verify sub-command)")
        h = write_json(RES_DIR / (e1_name(item)[:-5] + ".verify.json"), res)
        msg = _e1_status(item, res, None, h, [])
        log(msg)
        append_status("E1 re-run: " + msg)
        if not res["E1_pass"]:
            raise SystemExit(EXIT_E1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["resolve", "download", "verify", "fcache"])
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--kind", default=None, choices=[None, "main", "step143000"])
    ap.add_argument("--only", default=None, help="download only this model (resume aid)")
    ap.add_argument("--phase", default="before_S3", choices=["before_S3", "after_S4", "check"])
    ap.add_argument("--mock-hub", default=None, help="S0b test mode only: designer metadata JSON directory")
    ap.add_argument("--mock-fault", default=None, help="S0b test mode only: fault to inject into the mock")
    ap.add_argument("--fake-repos", default=None, help="S0b test mode only: directory of synthetic repos")
    args = ap.parse_args()
    check_plan_unchanged()
    if TEST_MODE:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
    log(f"download_pyt {args.cmd} pid={os.getpid()} test_mode={TEST_MODE} HF_HOME={os.environ['HF_HOME']}")
    try:
        {"resolve": cmd_resolve, "download": cmd_download, "verify": cmd_verify, "fcache": cmd_fcache}[args.cmd](args)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"FAILED {args.cmd}: {e!r}\n{tb}")
        append_status(f"FAILED download_pyt {args.cmd}: {e!r}"[:400])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
