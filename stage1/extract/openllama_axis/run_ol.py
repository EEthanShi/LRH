"""Driver for the OpenLLaMA-axis extraction and its checks E2-E7 (role oll-axis-extract, S0b, 2026-10-09).

OL-AXIS COPY of extract/llama_axis/run_llama_axis.py (sha256 12bcaa29...). Plan:
analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md v1.0 (sha256 4f18d710...), section 2 and section 11
(steps S2 and S4). Uses extract_core_ol.py (a copy of extract/llama_axis/extract_core_llama.py; only OL-AXIS changes).
Changes, all marked "OL-AXIS": model list and paths; E2 re-extracts BOTH open_llama_3b and open_llama_7b from their
cached Stage-1 snapshots and compares them bitwise (all layers, and lm_aux) with the stored features/wit1024/ files,
BEFORE the 13b download; order gates (CODE FROZEN line, S1, E1, E2); the RAM/disk STOP watch during loads; the
torch.load(weights_only=True) record; E5 for the OpenLLaMA tokenizer (13b vs 7b vs 3b ids, fast vs slow for 13b);
13b is extracted with the standard fp32 load (streamed fp32 only if that fails, and only after E3 on 7b); a final
`summary` sub-command. Threads: OL_THREADS (set by launch_ol.sh), at most 4 used by torch.

Sub-commands (always through analysis/s12_layermax_openllama/launch_ol.sh; GPU steps with --gpu):
  e2       S2. open_llama_3b and open_llama_7b, fp32 path, all 1024 WIT items -> features/wit1024_openllama_axis/_e2_check/;
           each must reproduce features/wit1024/<slug>.npy bitwise (expected); max relative difference <= 1e-6 is
           accepted, reported and triggers sensitivity S-E2; above 1e-6 the role STOPS before downloading (exit 4).
           Requires the S0b CODE FROZEN line and S1 passed. Also saves the fp32-load rotary inv_freq of 7b (for E3).
           launch_ol.sh oll_S2_E2 4 --gpu 4 -- venv/bin/python extract/openllama_axis/run_ol.py e2
  e5       E5 (CPU, after the download): token ids of 13b vs 7b vs 3b on all 1024 captions; 13b fast vs slow; sp_check.
           launch_ol.sh oll_E5_tok 2 -- venv/bin/python extract/openllama_axis/run_ol.py e5
  primary  S4. open_llama_13b, standard fp32 load -> features/wit1024_openllama_axis/; E4 semantics, E6 rerun/array
           checks, E7 num_params/NLL/bpb. Requires CODE FROZEN, S1, E1, E2 passed. If the standard load fails, exit 8
           (then: e3, and primary --mode streamed_fp32). --tokenizer-dir DIR (E5 sensitivity only) writes to
           features/wit1024_openllama_axis/_e5_tok7b/ instead.
           launch_ol.sh oll_S4_primary 4 --gpu 6 -- venv/bin/python extract/openllama_axis/run_ol.py primary
  e3       Only if needed: streamed-fp32 path of open_llama_7b -> features/wit1024_openllama_axis/_e3_check/, compared
           with the stored fp32 file (bitwise expected; <= 1e-6 relative required).
           launch_ol.sh oll_E3 4 --gpu 3 -- venv/bin/python extract/openllama_axis/run_ol.py e3
  ledger   CPU: appends one STATUS line with the last GPU step's minutes and the axis running total (run after
           every GPU step; plan section 11).
           launch_ol.sh oll_ledger 1 -- venv/bin/python extract/openllama_axis/run_ol.py ledger
  summary  CPU: collects S1/E1/E2/E3/E5/primary JSONs into results/S1_2_layermax_openllama/extraction/
           EXTRACTION_SUMMARY.json and logs one "EXTRACTION DONE" STATUS line with paths, shapes, hashes and the GPU
           ledger total.
           launch_ol.sh oll_S4_summary 1 -- venv/bin/python extract/openllama_axis/run_ol.py summary
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

_TH = str(min(4, int(os.environ.get("OL_THREADS", "4"))))  # OL-AXIS: launcher thread count, at most 4
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _TH  # OL-AXIS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ["HF_HUB_OFFLINE"] = "1"  # extraction never contacts the Hub
os.environ.pop("HF_TOKEN", None)  # OL-AXIS
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)  # OL-AXIS
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ol_axis_common import (CHECK_DIR, E3_DIR, E5_DIR, OUT_DIR, REL_TOL, RES_DIR, STAGE1, STAGE1_NPY_SHA256, R,  # noqa
                            STATUS, ResourceWatch, append_status, check_plan_unchanged, compare_features,
                            gpu_ledger_total_min, log, sha256_file, utcnow, write_json, mem_available_gb,
                            disk_free_gb_home)

ITEMS = R.ITEMS_DS
M13 = "openlm-research/open_llama_13b"  # OL-AXIS
M7 = "openlm-research/open_llama_7b"  # OL-AXIS
M3 = "openlm-research/open_llama_3b"  # OL-AXIS
ROPE_REF = CHECK_DIR / "rope_inv_freq_open_llama_7b_fp32load.npy"  # OL-AXIS
CODE_FROZEN_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \[oll-axis-[A-Za-z0-9-]+\] CODE FROZEN")  # OL-AXIS


def _torch_setup():
    import torch
    torch.set_num_threads(int(_TH))  # OL-AXIS
    return torch


def _rel(p: Path) -> str:
    return str(p).split("hrh_v2_stage1_20261008/")[-1]


def _read(path: Path):
    return json.load(open(path)) if Path(path).exists() else None


def _gate(name: str) -> bool:
    j = _read(RES_DIR / name)
    return bool(j and j.get("pass"))


def _short(m: str) -> str:  # OL-AXIS
    return m.split("/")[-1]


def _require(cond: bool, why: str):  # OL-AXIS
    if not cond:
        raise SystemExit(f"order gate: {why}")


def code_frozen() -> bool:  # OL-AXIS
    return any(CODE_FROZEN_RE.match(ln) for ln in open(STATUS, encoding="utf-8").read().splitlines())


def env_extra() -> dict:  # OL-AXIS: driver and kernel (plan section 0, environment change)
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        drv = repr(e)
    return dict(nvidia_driver=drv, kernel=platform.release())


def _sidecar_summary(stem: Path) -> dict:
    meta = json.load(open(str(stem) + ".json"))
    return dict(shape=meta["shape"], npy_sha256=meta["files_sha256"]["primary"]["sha256"],
                lm_aux_sha256=meta["files_sha256"]["lm_aux"]["sha256"], json_sha256=sha256_file(str(stem) + ".json"),
                num_params=meta["num_params"], n_layers=meta["n_layers"], hidden_size=meta["hidden_size"],
                weights_dtype=meta["weights_dtype"], semantics=meta["hidden_state_semantics"],
                rerun=meta["rerun_check"], array_checks=meta["array_checks"], sp_check=meta["tokenizer"]["sp_check"],
                tokenizer=meta["tokenizer"], nll=meta["nll_mean_over_items"], bpb=meta["bpb_mean_over_items"],
                n_tokens=meta["n_tokens"], timings=meta["timings"], source=meta.get("source"),
                openllama_axis=meta.get("openllama_axis"))  # OL-AXIS


def _semantics_ok(sem: dict) -> bool:
    return bool(sem and sem.get("hs0_vs_layer0_input") == 0.0 and sem.get("hs_i_vs_layer_i-1_output_max") == 0.0
                and sem.get("hsL_vs_finalnorm(lastlayer)") == 0.0)


def _torch_load_check(side: dict, model: str) -> dict:  # OL-AXIS: every .bin shard went through weights_only=True
    calls = ((side.get("openllama_axis") or {}).get("load") or {}).get("torch_load_calls") or []
    seen = {Path(c["file"]).name for c in calls}
    shards = {f: lfs for f, _, lfs in R.WEIGHT_SHARDS[model]}  # a call may name the snapshot file or its blob
    ok = {f: bool(f in seen or lfs in seen) for f, lfs in shards.items()}
    return dict(n_calls=len(calls), calls=calls[:20], shards=sorted(shards), shards_seen=sorted(f for f in ok if ok[f]),
                all_shards_via_weights_only=bool(all(ok.values())))


def run_extraction(model: str, mode: str, out_dir: Path, rope_ref=None, rope_out=None, extra=None, tokenizer_dir=None,
                   m_limit=None):
    torch = _torch_setup()
    import extract_core_ol as ec  # OL-AXIS
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ResourceWatch(label=f"{_short(model)} {mode} load/forward") as rw:  # OL-AXIS: RAM/disk STOP
        ec.run_lm(model, [ITEMS], out_dir, m_limit, log, mode=mode, rope_ref=rope_ref, rope_out=rope_out,
                  tokenizer_dir=tokenizer_dir,
                  extra_meta=dict(role="oll-axis-extract",  # OL-AXIS
                                  plan_sha256="4f18d710beda45492b1e191f74580bfc4ef53f6dafe791ad245ff1cf335f3240",
                                  env_extra=env_extra(), **(extra or {})))
    dt = time.time() - t0
    stem = out_dir / R.slug(model)
    ok, why = ec.is_valid(out_dir, ITEMS, model, m_limit or 1024)
    if not ok:
        raise RuntimeError(f"output invalid for {model} in {out_dir}: {why}")
    del torch
    return stem, dt, rw.summary()


# --------------------------------------------------------------------------------------------------------- S2: E2
def cmd_e2(args):
    _require(code_frozen(), "the S0b CODE FROZEN line is missing in STATUS.md")  # OL-AXIS
    _require(_gate("S1_registry.json"), "S1 has not passed")  # OL-AXIS
    results, all_pass, s_e2 = {}, True, False
    for model in R.LM_E2:  # OL-AXIS: 3b then 7b
        ref = STAGE1[model]
        ref_sha = sha256_file(str(ref) + ".npy")
        _require(ref_sha == STAGE1_NPY_SHA256[model], f"stored {ref}.npy sha256 {ref_sha} != frozen value")
        # OL-AXIS: the cached Stage-1 weights are the registry ones (blob name = LFS sha256, size); not re-hashed here
        import extract_core_ol as ec
        snap = ec.ensure_lm_snapshot(model)
        wchk = []
        for fname, size, lfs in R.WEIGHT_SHARDS[model]:
            real = (snap / fname).resolve()
            wchk.append(dict(file=fname, size=real.stat().st_size, blob=real.name,
                             ok=bool(real.stat().st_size == size and real.name == lfs)))
        _require(all(w["ok"] for w in wchk), f"cached {model} weights differ from the registry: {wchk}")
        rope_out = str(ROPE_REF) if model == M7 else None
        stem, dt, rw = run_extraction(model, "fp32", CHECK_DIR, rope_out=rope_out, extra=dict(check="E2"))
        cmp = compare_features(stem, ref, first_n=64)
        side = _sidecar_summary(stem)
        ref_meta = json.load(open(str(ref) + ".json"))
        tl = _torch_load_check(side, model)
        bitwise = bool(cmp["bitwise_equal"])
        aux_bitwise = all(v.get("bitwise_equal", v.get("equal")) for v in cmp["lm_aux"].values()) \
            if isinstance(cmp.get("lm_aux"), dict) and "error" not in cmp["lm_aux"] else False
        passed = bool(cmp["ok_shape"] and cmp["max_rel_diff"] <= REL_TOL and tl["all_shards_via_weights_only"])
        res = dict(check="E2", model=model, path="fp32 (Stage-1 load, copied pipeline)", out=_rel(stem) + ".npy",
                   ref=_rel(ref) + ".npy", ref_sha256=ref_sha, bitwise_equal=bitwise, lm_aux_bitwise_equal=aux_bitwise,
                   pass_rule=("bitwise expected; max rel (per layer max|d|/max|ref|) <= 1e-6 accepted and triggers "
                              "S-E2; else STOP before the download"),
                   compare=cmp, sidecar=side, wall_s=dt, resources=rw, cached_weights=wchk, torch_load=tl,
                   env_now=env_extra(), ref_env=ref_meta.get("env"), ref_nll=ref_meta["nll_mean_over_items"],
                   ref_bpb=ref_meta["bpb_mean_over_items"], ref_num_params=ref_meta["num_params"],
                   s_e2_triggered=bool(passed and not bitwise))
        if rope_out:
            import numpy as np
            res["rope_ref"] = dict(path=_rel(ROPE_REF), sha256=sha256_file(ROPE_REF),
                                   dtype=str(np.load(ROPE_REF).dtype), n=int(np.load(ROPE_REF).size))
        res["pass"] = passed
        h = write_json(RES_DIR / f"E2_{_short(model)}.json", res)
        results[model] = dict(res=res, h=h)
        all_pass &= passed
        s_e2 |= res["s_e2_triggered"]
        diff_where = ([dict(layer=r["layer"], max_abs=r["max_abs"], max_rel=r["max_rel"], n_diff=r["n_diff"])
                       for r in cmp.get("per_layer", []) if r["n_diff"] > 0][:8])
        msg = (f"E2 {_short(model)} re-extraction under driver {res['env_now']['nvidia_driver']} / kernel "
               f"{res['env_now']['kernel']} (copied pipeline, fp32 path, 1024 items x {cmp['ref_shape'][1]} layers) vs "
               f"stored features/wit1024/{R.slug(model)}.npy (sha256 {ref_sha[:16]}...): bitwise_equal={bitwise}, "
               f"max_abs={cmp['max_abs_diff']:.3g}, max_rel={cmp['max_rel_diff']:.3g}, elements differing "
               f"{cmp['n_elements_differing']}" + (f" (first layers with differences: {diff_where})" if diff_where else "")
               + f"; lm_aux bitwise {aux_bitwise}; .bin shards via torch.load(weights_only=True): "
               f"{tl['all_shards_via_weights_only']} ({tl['n_calls']} calls); new npy sha256 {side['npy_sha256']}; "
               f"{'PASS' if passed else 'FAIL -> STOP before download'}"
               f"{' (S-E2 sensitivity triggered)' if res['s_e2_triggered'] else ''}; {dt:.0f} s; E2 json sha256 {h}")
        log(msg)
        append_status(msg, f"results/S1_2_layermax_openllama/extraction/E2_{_short(model)}.json, {_rel(stem)}.npy")
        if not passed:
            break  # OL-AXIS: stop at the first failure (nothing further runs)
    if not all_pass:
        raise SystemExit(4)


# --------------------------------------------------------------------------------------------------------- E3
def cmd_e3(args):
    import numpy as np
    model = M7  # OL-AXIS: plan section 2, E3 on open_llama_7b
    _require(_gate("E2_open_llama_7b.json"), "E2 has not passed; E3 not run")
    ref = STAGE1[model]
    rope_ref = np.load(ROPE_REF)
    stem, dt, rw = run_extraction(model, "streamed_fp32", E3_DIR, rope_ref=rope_ref, extra=dict(check="E3"))
    cmp = compare_features(stem, ref, first_n=64)
    side = _sidecar_summary(stem)
    passed = bool(cmp["ok_shape"] and cmp["max_rel_diff"] <= REL_TOL)
    res = dict(check="E3", model=model, path="streamed_fp32 vs stored fp32", out=_rel(stem) + ".npy",
               ref=_rel(ref) + ".npy", bitwise_equal=cmp["bitwise_equal"],
               pass_rule="bitwise expected; max rel <= 1e-6 required", compare=cmp, sidecar=side, wall_s=dt,
               resources=rw)
    res["pass"] = passed
    h = write_json(RES_DIR / "E3_open_llama_7b.json", res)
    msg = (f"E3 open_llama_7b streamed-fp32 vs stored fp32 path (1024 items x {cmp['ref_shape'][1]} layers): "
           f"bitwise_equal={cmp['bitwise_equal']}, max_abs={cmp['max_abs_diff']:.3g}, max_rel={cmp['max_rel_diff']:.3g}; "
           f"lm_aux bitwise { {k: v.get('bitwise_equal', v.get('equal')) for k, v in cmp['lm_aux'].items()} }; "
           f"rotary inv_freq = fp32-load reference bitwise "
           f"{side['openllama_axis']['load'].get('rope_check', {}).get('equals_fp32_load_bitwise')}; "
           f"GPU peak {side['openllama_axis']['gpu_peak_allocated_gb']:.1f} GB; {'PASS' if passed else 'FAIL -> STOP'}; "
           f"{dt:.0f} s; E3 json sha256 {h}")
    log(msg)
    append_status(msg, f"results/S1_2_layermax_openllama/extraction/E3_open_llama_7b.json, {_rel(stem)}.npy")
    if not passed:
        raise SystemExit(5)


# --------------------------------------------------------------------------------------------------------- S4
def cmd_primary(args):
    model = M13  # OL-AXIS
    _require(code_frozen(), "the S0b CODE FROZEN line is missing in STATUS.md")
    _require(_gate("S1_registry.json"), "S1 has not passed")
    for sz in ("3b", "7b"):
        _require(_gate(f"E2_open_llama_{sz}.json"), f"E2 has not passed for open_llama_{sz}")
    e1 = _read(RES_DIR / f"E1_{R.slug(model)}.json")
    _require(bool(e1 and e1.get("E1_pass")), f"E1 has not passed for {model}")
    mode = args.mode
    rope_ref = None
    if mode == "streamed_fp32":  # plan section 2: only if the standard load failed, and only after E3 passed
        _require(_gate("E3_open_llama_7b.json"), "E3 has not passed; no streamed extraction")
        fail = _read(RES_DIR / "primary_open_llama_13b.fp32_failed.json")
        _require(bool(fail), "the standard fp32 load has not failed; streamed path not allowed")
        import numpy as np
        rope_ref = np.load(ROPE_REF)
    out_dir = OUT_DIR
    tag = ""
    if args.tokenizer_dir:  # OL-AXIS: E5 sensitivity only
        e5 = _read(RES_DIR / "E5_tokenizer.json")
        _require(bool(e5) and not e5.get("identical_13b_7b_3b", True), "E5 found identical ids; no sensitivity run")
        out_dir, tag = E5_DIR, ".e5_tok7b"
    t_start = utcnow()
    try:
        stem, dt, rw = run_extraction(model, mode, out_dir, rope_ref=rope_ref, tokenizer_dir=args.tokenizer_dir,
                                      extra=dict(check="primary" + tag))
    except Exception as e:  # noqa: BLE001
        if mode == "fp32" and not args.tokenizer_dir:
            fail = dict(model=model, mode=mode, started_utc=t_start, error=repr(e), traceback=traceback.format_exc(),
                        mem_available_gb=mem_available_gb(), disk_free_gb=disk_free_gb_home(),
                        next_steps="run e3; if E3 passes, run primary --mode streamed_fp32 (plan section 2)")
            h = write_json(RES_DIR / "primary_open_llama_13b.fp32_failed.json", fail)
            append_status(f"S4 standard fp32 extraction of {model} FAILED ({e!r})[:300]; streamed fp32 allowed only "
                          f"after E3 passes (plan section 2); failure json sha256 {h}")
            raise SystemExit(8)
        raise
    side = _sidecar_summary(stem)
    exp = R.EXPECTED[model]
    shape_ok = side["shape"] == [1024, exp["stored_layers"], exp["hidden_size"]] and side["n_layers"] == exp["n_layers"]
    tl = _torch_load_check(side, model)
    e6_ok = bool(side["rerun"]["max_abs_diff"] == 0.0 and side["array_checks"]["finite"]
                 and side["array_checks"]["zero_rows"] == 0)
    res = dict(model=model, mode=mode, out=_rel(stem), tokenizer_dir=args.tokenizer_dir, shape_ok=shape_ok,
               E4_semantics_ok=_semantics_ok(side["semantics"]), E6_ok=e6_ok,
               E7=dict(num_params=side["num_params"], num_params_expected=exp["num_params"],
                       num_params_match=side["num_params"] == exp["num_params"], nll_mean=side["nll"],
                       bpb_mean=side["bpb"]),
               torch_load=tl, sidecar=side, wall_s=dt, resources=rw, e1_weight_dtypes=e1.get("weight_dtypes"),
               env_now=env_extra())
    res["pass"] = bool(shape_ok and res["E4_semantics_ok"] and e6_ok and tl["all_shards_via_weights_only"])
    h = write_json(RES_DIR / f"primary_open_llama_13b{tag}.json", res)
    sh = side["shape"]
    msg = (f"S4 extraction {model} ({mode}{', tokenizer ' + args.tokenizer_dir if args.tokenizer_dir else ''}) -> "
           f"{_rel(stem)}.npy shape {sh} (layers x items x dim = {sh[1]} x {sh[0]} x {sh[2]}) sha256 {side['npy_sha256']}; "
           f"lm_aux sha256 {side['lm_aux_sha256']}; json sha256 {side['json_sha256']}; E4 semantics {side['semantics']}; "
           f"E6 rerun max|d|={side['rerun']['max_abs_diff']}, finite={side['array_checks']['finite']}, zero_rows="
           f"{side['array_checks']['zero_rows']}; E5 sp_check {side['sp_check']['mismatches']}/"
           f"{side['sp_check']['checked']} mismatches; E7 num_params {side['num_params']} (expected "
           f"{exp['num_params']}), nll {side['nll']:.4f}, bpb {side['bpb']:.4f}; .bin shards via "
           f"torch.load(weights_only=True): {tl['all_shards_via_weights_only']} ({tl['n_calls']} calls); fwd "
           f"{side['timings']['forward_s']:.0f} s ({side['timings']['per_item_ms']:.1f} ms/item), load "
           f"{side['timings']['load_s']:.0f} s, GPU peak {side['openllama_axis']['gpu_peak_allocated_gb']:.1f} GB, min "
           f"available RAM {rw['min_available_ram_gb']:.0f} GB, min free disk {rw['min_free_disk_gb']:.0f} GB; "
           f"{'PASS' if res['pass'] else 'FAIL'}; summary json sha256 {h}")
    log(msg)
    append_status(msg, f"{_rel(stem)}.{{npy,json,lm_aux.npz,done}}, "
                       f"results/S1_2_layermax_openllama/extraction/primary_open_llama_13b{tag}.json")
    if not res["pass"]:
        raise SystemExit(6)


# --------------------------------------------------------------------------------------------------------- E5
def cmd_e5(args):
    import pandas as pd
    import sentencepiece as spm
    from transformers import AutoTokenizer
    import extract_core_ol as ec
    from common import FEATURES
    e1 = _read(RES_DIR / f"E1_{R.slug(M13)}.json")
    _require(bool(e1 and e1.get("E1_pass")), "E1 has not passed for open_llama_13b")
    items = pd.read_parquet(FEATURES / ITEMS / "items.parquet")
    man = json.load(open(FEATURES / ITEMS / "items_manifest.json"))
    items_sha = sha256_file(FEATURES / ITEMS / "items.parquet")
    texts = items["text"].astype(str).tolist()
    ids, info = {}, {}
    for model in R.LM_OL_AXIS:
        snap = ec.ensure_lm_snapshot(model)
        tok, fast = ec._tok_load(str(snap))
        sp = spm.SentencePieceProcessor(model_file=str(snap / "tokenizer.model"))
        ids[model] = [tok(t)["input_ids"] for t in texts]
        sp_mis = 0
        for t, x in zip(texts, ids[model]):
            y = x[1:] if x and x[0] == tok.bos_token_id else x
            sp_mis += int(y != sp.encode(t))
        files = {f: (sha256_file(snap / f) if (snap / f).exists() else None) for f in R.TOKENIZER_FILES_SHARED}
        info[model] = dict(cls=type(tok).__name__, fast=fast, bos=tok.bos_token_id, eos=tok.eos_token_id,
                           files_sha256=files, sp_check=dict(checked=len(texts), mismatches=sp_mis))
    # 13b fast vs slow (the README warns about the auto-converted fast tokenizer)
    snap13 = ec.ensure_lm_snapshot(M13)
    try:
        slow = AutoTokenizer.from_pretrained(str(snap13), use_fast=False)
        slow_ids = [slow(t)["input_ids"] for t in texts]
        n_fast_slow = sum(a != b for a, b in zip(ids[M13], slow_ids))
        slow_info = dict(cls=type(slow).__name__, n_captions_fast_ne_slow=n_fast_slow)
    except Exception as e:  # noqa: BLE001
        slow_info = dict(error=repr(e))
    d7 = [i for i in range(len(texts)) if ids[M13][i] != ids[M7][i]]
    d3 = [i for i in range(len(texts)) if ids[M13][i] != ids[M3][i]]
    d73 = [i for i in range(len(texts)) if ids[M7][i] != ids[M3][i]]
    same_files = all(len({info[m]["files_sha256"][f] for m in R.LM_OL_AXIS}) == 1 for f in R.TOKENIZER_FILES_SHARED)
    identical = not d7 and not d3
    res = dict(check="E5", items_sha256=items_sha, items_manifest_sha256=man["items_sha256"], tokenizers=info,
               n_captions=len(texts), n_13b_ne_7b=len(d7), n_13b_ne_3b=len(d3), n_7b_ne_3b=len(d73),
               differing_items=dict(b13_vs_7b=d7[:50], b13_vs_3b=d3[:50]), fast_vs_slow_13b=slow_info,
               tokenizer_files_identical_across_repos=same_files, identical_13b_7b_3b=identical,
               sp_check_expected="15/1024 mismatches, as in both stored sidecars",
               consequence=("ids identical: no tokenizer sensitivity extraction is needed (plan section 2 E5)" if identical
                            else "ids differ: primary keeps 13b's own tokenizer; sensitivity extraction with the 7b "
                                 "tokenizer files required (run_ol.py primary --tokenizer-dir <7b snapshot>)"))
    res["pass"] = identical
    h = write_json(RES_DIR / "E5_tokenizer.json", res)
    msg = (f"E5 tokenizer: token ids of open_llama_13b vs 7b / vs 3b on all {len(texts)} WIT captions differ in "
           f"{len(d7)} / {len(d3)} (7b vs 3b {len(d73)}); 13b fast vs slow: {slow_info}; sp_check mismatches "
           f"{ {_short(m): info[m]['sp_check']['mismatches'] for m in info} } (expected 15 each); tokenizer files "
           f"byte-identical across the 3 repos: {same_files}; {res['consequence']}; E5 json sha256 {h}")
    log(msg)
    append_status(msg, "results/S1_2_layermax_openllama/extraction/E5_tokenizer.json")


# --------------------------------------------------------------------------------------------------------- summary
def cmd_summary(args):
    names = ["S1_registry.json", "E2_open_llama_3b.json", "E2_open_llama_7b.json", f"E1_{R.slug(M13)}.json",
             "E5_tokenizer.json", "primary_open_llama_13b.json", "E3_open_llama_7b.json",
             "primary_open_llama_13b.fp32_failed.json", "primary_open_llama_13b.e5_tok7b.json"]
    files = {n: dict(sha256=sha256_file(RES_DIR / n), pass_=(_read(RES_DIR / n) or {}).get("pass",
                     (_read(RES_DIR / n) or {}).get("E1_pass"))) for n in names if (RES_DIR / n).exists()}
    stem = OUT_DIR / R.slug(M13)
    side = _sidecar_summary(stem) if Path(str(stem) + ".done").exists() else None
    ledger = gpu_ledger_total_min()
    res = dict(utc=utcnow(), files=files, features_13b=(dict(npy=_rel(stem) + ".npy", shape=side["shape"],
                                                           npy_sha256=side["npy_sha256"],
                                                           lm_aux_sha256=side["lm_aux_sha256"],
                                                           json_sha256=side["json_sha256"]) if side else None),
               gpu_ledger_total_min=ledger)
    h = write_json(RES_DIR / "EXTRACTION_SUMMARY.json", res)
    ok = all(v["pass_"] for k, v in files.items() if not k.endswith("fp32_failed.json"))
    msg = (f"EXTRACTION {'DONE' if (ok and side) else 'INCOMPLETE'} (S1-S4 of the plan): "
           + "; ".join(f"{k} pass={v['pass_']} sha256 {v['sha256'][:16]}..." for k, v in files.items())
           + (f"; open_llama_13b features {_rel(stem)}.npy layers x items x dim = {side['shape'][1]} x "
              f"{side['shape'][0]} x {side['shape'][2]} sha256 {side['npy_sha256']}" if side else "")
           + f"; GPU ledger total {ledger:.2f} min (budget 60); summary json sha256 {h}")
    log(msg)
    append_status(msg, "results/S1_2_layermax_openllama/extraction/EXTRACTION_SUMMARY.json")


def cmd_ledger(args):  # OL-AXIS: plan section 11, every GPU step logs its GPU minutes and the running total
    from ol_axis_common import LEDGER
    rows = [ln.split("\t") for ln in LEDGER.read_text().splitlines()[1:] if ln.strip()] if LEDGER.exists() else []
    last = rows[-1] if rows else None
    tot = gpu_ledger_total_min()
    msg = (f"GPU ledger (results/S1_2_layermax_openllama/logs/gpu_ledger.tsv): last step "
           f"{(last[0] + ' ' + last[4] + ' min (' + last[1] + ' -> ' + last[2] + ', exit ' + last[6] + ')') if last else 'none'}"
           f"; running total {tot:.2f} GPU-min of 60 over {len(rows)} GPU steps "
           f"({', '.join(r[0] + ' ' + r[4] for r in rows)})")
    log(msg)
    append_status(msg, "results/S1_2_layermax_openllama/logs/gpu_ledger.tsv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["e2", "e3", "primary", "e5", "summary", "ledger"])
    ap.add_argument("--mode", default="fp32", choices=["fp32", "streamed_fp32"])
    ap.add_argument("--tokenizer-dir", default=None)
    args = ap.parse_args()
    check_plan_unchanged()
    log(f"run_ol {args.cmd} pid={os.getpid()} threads={_TH}")
    try:
        {"e2": cmd_e2, "e3": cmd_e3, "primary": cmd_primary, "e5": cmd_e5, "summary": cmd_summary,
         "ledger": cmd_ledger}[args.cmd](args)  # OL-AXIS: ledger
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"FAILED {args.cmd}: {e!r}\n{tb}")
        append_status(f"FAILED run_ol {args.cmd}: {e!r}"[:400])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
