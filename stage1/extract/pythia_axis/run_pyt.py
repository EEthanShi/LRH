"""Driver for the Pythia-axis extraction and its checks E2, E4-E7 (role pyt-axis-extract, S0b, 2026-10-10).

PYT-AXIS COPY of extract/openllama_axis/run_ol.py (sha256 4231c834...). Plan:
analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md v1.0 (sha256 6de89603...), section 2 and section 11 (steps S2, S4).
Uses extract_core_pyt.py (a copy of extract_core_ol.py; only PYT-AXIS changes). Changes, all marked "PYT-AXIS":
model list and paths; E2 re-extracts bigscience/bloomz-560m (descriptive: a difference flags the cross-family tables
and never stops the axis) from the standard cache, read-only, with the token-0 diagnostics switched on so that E2 also
guards the primary stack against them; S4 extracts one Pythia size per process, smallest first, after its E1 (2.8b
after E1-PROV, from the snapshot E1-PROV chose), only after the G1 gate for S4 recorded GO, with the free-GPU-memory
check (need = 4*num_params + 2 GB, plus 4 GiB), the RAM/disk STOP watch (16 GB / 20 GB), the loading check, E4
(semantics exactly 0.0, else STOP), the rotary and head checks, the token-0 diagnostics (_tok1 stack and sink
statistics), E6 (items 0..2 twice, bitwise, primary and _tok1) and E7 (num_params, NLL, bpb); E5 compares the
transformers ids with the raw `tokenizers` encoder of tokenizer.json for every caption and the ids across the 7
sizes; `summary` logs the E7 bpb-monotonicity flag before Part B, the F-cache listing after S4 and one EXTRACTION DONE
line; `diskwatch` polls free disk every 10 s during S3-S8 and stops the axis's own processes below 20 GB. The
streamed-fp32 path, E3 and the tokenizer-sensitivity re-extraction are not ported (plan section 2). Threads:
PYT_THREADS (set by launch_pyt.sh), at most 6.

Sub-commands (always through analysis/s12_layermax_pythia/launch_pyt.sh; GPU steps with --gpu):
  e2       S2. bloomz-560m, fp32 path, all 1024 WIT items -> features/wit1024_pythia_axis/_e2_check/ (+ _tok1/);
           compared bitwise with features/wit1024/bigscience__bloomz-560m.npy (descriptive). Requires CODE FROZEN, S1.
           launch_pyt.sh pyt_S2_E2 4 --gpu 0.5 -- venv/bin/python extract/pythia_axis/run_pyt.py e2
  primary  S4, one size: --size 70m|160m|410m|1b|1.4b|2.8b|6.9b. Requires CODE FROZEN, S1, E2 run, E1 (2.8b: both
           E1 and E1-PROV), G1(S4) = GO, every smaller size done. Exit 6 = STOP (loading check / E4), 77 = GPU memory
           short (deferral; run gpu_gate_pyt.py memwait first).
           launch_pyt.sh pyt_S4_70m 4 --gpu 0.3 -- venv/bin/python extract/pythia_axis/run_pyt.py primary --size 70m
  e5       E5 (CPU, after all E1): ids transformers vs tokenizers, BOS/EOS, n_tokens, ids across the 7 sizes.
  summary  CPU: E7 bpb flag, F-cache after S4, EXTRACTION_SUMMARY.json and one EXTRACTION DONE line.
  ledger   CPU: one STATUS line with the GPU ledger.
  diskwatch  CPU, background: disk guard watcher (plan section 11), until logs/diskwatch.stop exists.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal  # PYT-AXIS
import subprocess
import sys
import time
import traceback
from pathlib import Path

_TH = str(max(1, min(6, int(os.environ.get("PYT_THREADS", "4")))))  # PYT-AXIS: launcher thread count, at most 6
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _TH
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ["HF_HUB_OFFLINE"] = "1"  # extraction never contacts the Hub
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyt_axis_common import (CHECK_DIR, OUT_DIR, TOK1_DIR, RES_DIR, LOGS_DIR, STAGE1_E2, R, TEST_MODE,  # noqa: E402
                             EXIT_DEFER, ResourceWatch, append_status, check_plan_unchanged, compare_features,
                             gpu_ledger_total_min, ledger_rows, log, sha256_file, utcnow, write_json, read_json,
                             require, code_frozen_line, mem_available_gb, disk_free_gb, MIN_FREE_DISK_GB, HF_HOME)

ITEMS = R.ITEMS_DS
EXIT_STOP = 6


def _torch_setup():
    import torch
    torch.set_num_threads(int(_TH))
    return torch


def _rel(p) -> str:  # PYT-AXIS
    return str(p).split("hrh_v2_stage1_20261008/")[-1]


def _short(m: str) -> str:  # PYT-AXIS
    return m.split("/")[-1]


def env_extra() -> dict:  # driver and kernel (plan section 0, environment) PYT-AXIS
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
                rerun=meta["rerun_check"], array_checks=meta["array_checks"], tokenizer=meta["tokenizer"],
                nll=meta["nll_mean_over_items"], bpb=meta["bpb_mean_over_items"], n_tokens=meta["n_tokens"],
                timings=meta["timings"], source=meta.get("source"), attn=meta.get("attn_implementation"),
                pythia_axis=meta.get("pythia_axis"))  # PYT-AXIS


def _semantics_ok(sem: dict) -> bool:
    return bool(sem and sem.get("hs0_vs_layer0_input") == 0.0 and sem.get("hs_i_vs_layer_i-1_output_max") == 0.0
                and sem.get("hsL_vs_finalnorm(lastlayer)") == 0.0)


def _torch_load_check(side: dict, weight_files: list) -> dict:  # every .bin went through weights_only=True PYT-AXIS
    calls = ((side.get("pythia_axis") or {}).get("load") or {}).get("torch_load_calls") or []
    seen = {Path(c["file"]).name for c in calls}
    ok = {f: bool(f in seen or lfs in seen) for f, _, lfs in weight_files}
    return dict(n_calls=len(calls), calls=calls[:20], files=sorted(ok), files_seen=sorted(f for f in ok if ok[f]),
                all_via_weights_only=bool(all(ok.values())))


def _gpu_free_check(model: str) -> dict:  # PYT-AXIS: plan section 2 "Order and memory" (no waiting inside a GPU step)
    import torch  # PYT-AXIS
    free, total = torch.cuda.mem_get_info()
    need = R.gpu_need_bytes(model) + R.GPU_HEADROOM_BYTES
    return dict(free_bytes=int(free), total_bytes=int(total), need_bytes=int(need), ok=bool(free >= need))


def run_extraction(model: str, out_dir: Path, tok1_dir=None, sink_path=None, extra=None, m_limit=None,
                   gptneox_checks=False, inventory_ref=None, snap=None):
    torch = _torch_setup()
    import extract_core_pyt as ec  # PYT-AXIS
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ResourceWatch(label=f"{_short(model)} load/forward") as rw:  # RAM 16 GB / disk 20 GB STOP
        checks = ec.run_lm(model, [ITEMS], out_dir, m_limit, log, mode="fp32", snap=snap, tok1_dir=tok1_dir,
                           sink_path=sink_path, gptneox_checks=gptneox_checks, inventory_ref=inventory_ref,
                           extra_meta=dict(role="pyt-axis-extract", plan_sha256="6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b",
                                           env_extra=env_extra(), **(extra or {})))
    dt = time.time() - t0
    stem = out_dir / R.slug(model)
    ok, why = ec.is_valid(out_dir, ITEMS, model, m_limit or 1024)
    if not ok:
        raise RuntimeError(f"output invalid for {model} in {out_dir}: {why}")
    del torch
    return stem, dt, rw.summary(), checks


# --------------------------------------------------------------------------------------------------------- S2: E2
def cmd_e2(args):
    require(code_frozen_line() is not None, "the S0b CODE FROZEN line is missing in STATUS.md")
    require(bool((read_json(RES_DIR / "S1_registry.json") or {}).get("pass")), "S1 has not passed")
    model = R.E2_MODEL
    ref = STAGE1_E2
    ref_sha = sha256_file(str(ref) + ".npy")
    require(ref_sha == R.E2_NPY_SHA256, f"stored {ref}.npy sha256 {ref_sha} != frozen value")
    stem, dt, rw, _ = run_extraction(model, CHECK_DIR, tok1_dir=CHECK_DIR / "_tok1",
                                     sink_path=RES_DIR / "sink_e2_bloomz-560m.json", extra=dict(check="E2"))
    cmp = compare_features(stem, ref, first_n=64)
    side = _sidecar_summary(stem)
    ref_meta = json.load(open(str(ref) + ".json"))
    bitwise = bool(cmp["bitwise_equal"])
    aux_bitwise = all(v.get("bitwise_equal", v.get("equal")) for v in cmp["lm_aux"].values()) \
        if isinstance(cmp.get("lm_aux"), dict) and "error" not in cmp["lm_aux"] else False
    res = dict(check="E2", model=model, path="fp32 (Stage-1 load, copied pipeline, token-0 diagnostics on)",
               out=_rel(stem) + ".npy", ref=_rel(ref) + ".npy", ref_sha256=ref_sha, bitwise_equal=bitwise,
               lm_aux_bitwise_equal=aux_bitwise, compare=cmp, sidecar=side, wall_s=dt, resources=rw,
               env_now=env_extra(), ref_env=ref_meta.get("env"), ref_nll=ref_meta["nll_mean_over_items"],
               ref_bpb=ref_meta["bpb_mean_over_items"], ref_num_params=ref_meta["num_params"],
               rule=("descriptive (plan section 2 E2): bitwise expected; a difference flags the cross-family "
                     "descriptive tables and does not stop the axis"),
               flag_cross_family_tables=bool(not (bitwise and aux_bitwise)))
    res["pass"] = bool(bitwise and aux_bitwise)
    h = write_json(RES_DIR / "E2_bloomz-560m.json", res)
    msg = (f"E2 bloomz-560m re-extraction under driver {res['env_now']['nvidia_driver']} / kernel "
           f"{res['env_now']['kernel']} (copied pipeline, fp32, token-0 diagnostics on, 1024 items x "
           f"{cmp['ref_shape'][1]} layers) vs stored features/wit1024/{R.slug(model)}.npy (sha256 {ref_sha[:16]}...): "
           f"bitwise_equal={bitwise}, max_abs={cmp['max_abs_diff']:.3g}, max_rel={cmp['max_rel_diff']:.3g}, elements "
           f"differing {cmp['n_elements_differing']}; lm_aux bitwise {aux_bitwise}; new npy sha256 {side['npy_sha256']}; "
           f"{'equal' if res['pass'] else 'DIFFERS -> cross-family descriptive tables flagged (not a stop)'}; "
           f"{dt:.0f} s; E2 json sha256 {h}")
    log(msg)
    append_status(msg, f"{_rel(RES_DIR / 'E2_bloomz-560m.json')}, {_rel(stem)}.npy")


# --------------------------------------------------------------------------------------------------------- S4
def _source_files(model: str):  # PYT-AXIS: weight files of the snapshot that is loaded
    import extract_core_pyt as ec  # PYT-AXIS
    rev = ec.source_revision(model)
    if model == R.PROV_MODEL and rev == R.STEP143000[model]:
        return rev, R.PROV_WEIGHT_FILES, f"E1_{R.slug(model)}.step143000.json"
    return rev, R.WEIGHT_FILES[model], f"E1_{R.slug(model)}.json"


def cmd_primary(args):
    model = R.MODEL_OF[args.size]
    require(code_frozen_line() is not None, "the S0b CODE FROZEN line is missing in STATUS.md")
    require(bool((read_json(RES_DIR / "S1_registry.json") or {}).get("pass")), "S1 has not passed")
    require(read_json(RES_DIR / "E2_bloomz-560m.json") is not None, "E2 has not run")
    g1 = read_json(LOGS_DIR / "gpu_gate_G1_S4.json")
    require(bool(g1 and g1.get("decision") == "GO"), "gate G1 before S4 has no GO record")
    e1 = read_json(RES_DIR / f"E1_{R.slug(model)}.json")
    require(bool(e1 and e1.get("E1_pass")), f"E1 has not passed for {model}")
    if model == R.PROV_MODEL:
        require(bool((read_json(RES_DIR / "e1prov_2.8b.json") or {}).get("source") in ("main", "step143000")),
                "E1-PROV decision missing for 2.8b")
    for smaller in R.LM_PYT_AXIS[:R.LM_PYT_AXIS.index(model)]:  # smallest first
        require(bool((read_json(RES_DIR / f"primary_{_short(smaller)}.json") or {}).get("pass")),
                f"{smaller} has not been extracted yet (order: smallest first)")
    rev, wfiles, e1name = _source_files(model)
    e1src = read_json(RES_DIR / e1name)
    require(bool(e1src and e1src.get("E1_pass")), f"E1 has not passed for the loaded snapshot {e1name}")
    _torch_setup()
    gm = _gpu_free_check(model)
    if not gm["ok"]:
        msg = (f"S4 {model}: free GPU memory {gm['free_bytes'] / 2**30:.1f} GiB < need "
               f"{gm['need_bytes'] / 2**30:.1f} GiB; not started (run gpu_gate_pyt.py memwait; deferral rule)")
        log(msg)
        append_status(msg)
        raise SystemExit(EXIT_DEFER)
    t_start = utcnow()
    import extract_core_pyt as ec  # PYT-AXIS
    try:
        stem, dt, rw, checks = run_extraction(model, OUT_DIR, tok1_dir=TOK1_DIR,
                                              sink_path=RES_DIR / f"sink_{args.size}.json",
                                              extra=dict(check="primary", source_revision=rev, gpu_mem_check=gm),
                                              gptneox_checks=True, inventory_ref=e1src.get("inventory"))
    except ec.LoadingCheckError as e:
        h = write_json(RES_DIR / f"primary_{_short(model)}.loading_check_failed.json",
                       dict(model=model, started_utc=t_start, error=str(e)))
        append_status(f"STOP S4 {model}: loading check failed (plan section 2) {str(e)[:300]}; json sha256 {h}")
        raise SystemExit(EXIT_STOP)
    side = _sidecar_summary(stem)
    exp = R.EXPECTED[model]
    shape_ok = side["shape"] == [1024, exp["stored_layers"], exp["hidden_size"]] and side["n_layers"] == exp["n_layers"]
    tl = _torch_load_check(side, wfiles)
    tok1_side = json.load(open(TOK1_DIR / f"{R.slug(model)}.json"))
    sink = json.load(open(RES_DIR / f"sink_{args.size}.json"))
    rr = side["rerun"]
    e6_ok = bool(rr["max_abs_diff"] == 0.0 and all(rr.get("primary_bitwise") or [False])
                 and all(rr.get("tok1_bitwise") or [False]) and side["array_checks"]["finite"]
                 and side["array_checks"]["zero_rows"] == 0 and tok1_side["array_checks"]["finite"]
                 and tok1_side["array_checks"]["zero_rows"] == 0)
    lc = (side["pythia_axis"] or {}).get("checks", {}).get("loading_check", {})
    rot = (side["pythia_axis"] or {}).get("checks", {}).get("rotary_check", {})
    res = dict(model=model, size=args.size, source_revision=rev, out=_rel(stem), shape_ok=shape_ok,
               E4_semantics_ok=_semantics_ok(side["semantics"]), E4_loading_check_pass=bool(lc.get("pass")),
               E4_rotary=rot, E4_head_ok=lc.get("head_ok"), E6_ok=e6_ok,
               E7=dict(num_params=side["num_params"], num_params_expected=exp["num_params"],
                       num_params_match=side["num_params"] == exp["num_params"], nll_mean=side["nll"],
                       bpb_mean=side["bpb"]),
               tok1=dict(path=_rel(TOK1_DIR / f"{R.slug(model)}.npy"), sha256=tok1_side["sha256"],
                         shape=tok1_side["shape"], n_one_token_captions=tok1_side["n_one_token_captions"]),
               sink=dict(path=_rel(RES_DIR / f"sink_{args.size}.json"),
                         sha256=sha256_file(RES_DIR / f"sink_{args.size}.json"),
                         ratio_median_layer0=sink["per_layer"][0]["ratio_median"],
                         ratio_median_last=sink["per_layer"][-1]["ratio_median"]),
               torch_load=tl, sidecar=side, wall_s=dt, resources=rw, gpu_mem_check=gm, env_now=env_extra())
    res["pass"] = bool(shape_ok and res["E4_semantics_ok"] and res["E4_loading_check_pass"] and e6_ok
                       and tl["all_via_weights_only"])
    h = write_json(RES_DIR / f"primary_{_short(model)}.json", res)
    sh = side["shape"]
    msg = (f"S4 extraction {model} (fp32, source {rev[:12]}) -> {_rel(stem)}.npy shape {sh} (items x layers x dim = "
           f"{sh[0]} x {sh[1]} x {sh[2]}) sha256 {side['npy_sha256']}; lm_aux sha256 {side['lm_aux_sha256']}; json "
           f"sha256 {side['json_sha256']}; _tok1 sha256 {tok1_side['sha256']} ({tok1_side['n_one_token_captions']} "
           f"one-token captions); sink json {res['sink']['sha256'][:16]}...; E4 semantics {side['semantics']}; loading "
           f"check {'PASS' if lc.get('pass') else 'FAIL'} (missing {lc.get('missing_keys')}, unexpected "
           f"{len(lc.get('unexpected_keys') or [])}, head shared={lc.get('lm_head_shares_embed_in_storage')}); rotary "
           f"{rot.get('inv_freq_dtype')} max|d| fp32-formula {rot.get('max_abs_diff_vs_fp32_formula')}; E6 "
           f"{'ok' if e6_ok else 'FAIL'}; E7 num_params {side['num_params']} (plan {exp['num_params']}), nll "
           f"{side['nll']:.4f}, bpb {side['bpb']:.4f}; .bin via torch.load(weights_only=True): {tl['all_via_weights_only']}; "
           f"fwd {side['timings']['forward_s']:.0f} s ({side['timings']['per_item_ms']:.1f} ms/item), load "
           f"{side['timings']['load_s']:.0f} s, GPU peak {side['pythia_axis']['gpu_peak_allocated_gb']:.1f} GB, min RAM "
           f"{rw['min_available_ram_gb']:.0f} GB, min free disk {rw['min_free_disk_gb']:.0f} GB; "
           f"{'PASS' if res['pass'] else 'FAIL -> STOP'}; json sha256 {h}")
    log(msg)
    append_status(msg, f"{_rel(stem)}.{{npy,json,lm_aux.npz,done}}, {_rel(RES_DIR / f'primary_{_short(model)}.json')}")
    if not res["pass"]:
        raise SystemExit(EXIT_STOP)


# --------------------------------------------------------------------------------------------------------- E5
def e5_compare(snaps: dict, texts: list) -> dict:  # PYT-AXIS: plan section 2 E5 (replaced)
    import numpy as np  # PYT-AXIS
    from tokenizers import Tokenizer  # PYT-AXIS
    import extract_core_pyt as ec  # PYT-AXIS
    ids, info = {}, {}
    for model, snap in snaps.items():
        tok, fast = ec._tok_load(str(snap))
        raw = Tokenizer.from_file(str(Path(snap) / "tokenizer.json"))
        tids = [tok(t)["input_ids"] for t in texts]
        rids = [raw.encode(t).ids for t in texts]
        diff = [i for i, (a, b) in enumerate(zip(tids, rids)) if list(a) != list(b)]
        bos, eos = tok.bos_token_id, tok.eos_token_id
        n_tok = np.array([len(x) for x in tids])
        info[model] = dict(cls=type(tok).__name__, fast=fast, bos=bos, eos=eos,
                           n_transformers_ne_tokenizers=len(diff), differing_items=diff[:50],
                           n_first_is_bos=int(sum(1 for x in tids if x and x[0] == bos)),
                           n_last_is_eos=int(sum(1 for x in tids if x and x[-1] == eos)),
                           n_tokens=dict(min=int(n_tok.min()), median=float(np.median(n_tok)), max=int(n_tok.max()),
                                         q05=float(np.quantile(n_tok, 0.05)), q95=float(np.quantile(n_tok, 0.95)),
                                         n_one_token=int((n_tok == 1).sum())),
                           post_processor=(json.load(open(Path(snap) / "tokenizer.json")).get("post_processor") or {}).get("type"),
                           files_sha256={f: sha256_file(Path(snap) / f) for f in R.TOKENIZER_FILES_SHARED})
        ids[model] = tids
    models = list(snaps)
    across = {m: sum(1 for a, b in zip(ids[models[0]], ids[m]) if list(a) != list(b)) for m in models[1:]}
    same_files = all(len({info[m]["files_sha256"][f] for m in models}) == 1 for f in R.TOKENIZER_FILES_SHARED)
    res = dict(check="E5", n_captions=len(texts), tokenizers=info, ids_vs_first_model=across,
               tokenizer_files_identical=same_files,
               any_bos_added=any(v["n_first_is_bos"] for v in info.values()),
               any_eos_added=any(v["n_last_is_eos"] for v in info.values()),
               rule=("for every caption the transformers ids equal tokenizers.Tokenizer.from_file(tokenizer.json)"
                     ".encode(text).ids (the file's own post-processor); BOS/EOS addition recorded (none expected); "
                     "n_tokens recorded; ids across sizes compared; a difference is reported, the primary features "
                     "keep the PROTOCOL path"))
    res["pass"] = bool(all(v["n_transformers_ne_tokenizers"] == 0 for v in info.values())
                       and all(v == 0 for v in across.values()) and same_files)
    return res


def cmd_e5(args):
    import pandas as pd
    from common import FEATURES
    import extract_core_pyt as ec  # PYT-AXIS
    for m in R.LM_PYT_AXIS:
        require(bool((read_json(RES_DIR / f"E1_{R.slug(m)}.json") or {}).get("E1_pass")), f"E1 has not passed for {m}")
    items = pd.read_parquet(FEATURES / ITEMS / "items.parquet")
    texts = items["text"].astype(str).tolist()
    snaps = {m: ec.ensure_lm_snapshot(m, revision=R.LM_REVISIONS[m]) for m in R.LM_PYT_AXIS}
    res = e5_compare(snaps, texts)
    res["items_sha256"] = sha256_file(FEATURES / ITEMS / "items.parquet")
    h = write_json(RES_DIR / "E5_tokenizer.json", res)
    msg = (f"E5 tokenizer: transformers vs raw tokenizers ids differ in "
           f"{ {_short(m): v['n_transformers_ne_tokenizers'] for m, v in res['tokenizers'].items()} } of "
           f"{res['n_captions']} captions; ids vs 70m differ in { {_short(m): v for m, v in res['ids_vs_first_model'].items()} }; "
           f"tokenizer files identical {res['tokenizer_files_identical']}; BOS added {res['any_bos_added']}, EOS added "
           f"{res['any_eos_added']}; n_tokens {res['tokenizers'][R.LM_PYT_AXIS[0]]['n_tokens']}; "
           f"{'PASS' if res['pass'] else 'DIFFERENCE REPORTED (primary keeps the PROTOCOL path)'}; json sha256 {h}")
    log(msg)
    append_status(msg, _rel(RES_DIR / "E5_tokenizer.json"))


# --------------------------------------------------------------------------------------------------------- summary
def cmd_summary(args):
    rows = {}
    for m in R.LM_PYT_AXIS:
        j = read_json(RES_DIR / f"primary_{_short(m)}.json")
        rows[m] = j
    done = all(j and j.get("pass") for j in rows.values())
    bpb = {R.SIZE_OF[m]: (j["E7"]["bpb_mean"] if j else None) for m, j in rows.items()}
    vals = [bpb[s] for s in R.SIZES]
    not_mono = [f"{R.SIZES[i]}->{R.SIZES[i + 1]}" for i in range(len(vals) - 1)
                if vals[i] is not None and vals[i + 1] is not None and not vals[i + 1] < vals[i]]
    if done:
        append_status(f"E7 bpb per size {bpb}; " + (f"FLAG: bpb not monotone at {not_mono}" if not_mono else
                                                    "bpb strictly decreasing in size (no flag)")
                      + " (plan section 2 E7; logged before Part B)")
    from download_pyt import cmd_fcache  # PYT-AXIS
    cmd_fcache(argparse.Namespace(phase="after_S4"))
    names = (["S1_registry.json", "E2_bloomz-560m.json", "E5_tokenizer.json", "e1prov_2.8b.json",
              "fcache_before_S3.json", "fcache_after_S4.json"]
             + [f"E1_{R.slug(m)}.json" for m in R.LM_PYT_AXIS] + [f"E1_{R.slug(R.PROV_MODEL)}.step143000.json"]
             + [f"primary_{_short(m)}.json" for m in R.LM_PYT_AXIS])
    files = {}
    for n in names:
        if (RES_DIR / n).exists():
            j = read_json(RES_DIR / n)
            files[n] = dict(sha256=sha256_file(RES_DIR / n),
                            pass_=j.get("pass", j.get("E1_pass")) if isinstance(j, dict) else None)
    feats = {R.SIZE_OF[m]: (dict(npy=j["out"] + ".npy", shape=j["sidecar"]["shape"], npy_sha256=j["sidecar"]["npy_sha256"],
                                 lm_aux_sha256=j["sidecar"]["lm_aux_sha256"], json_sha256=j["sidecar"]["json_sha256"],
                                 tok1_sha256=j["tok1"]["sha256"], source_revision=j["source_revision"])
                            if j else None) for m, j in rows.items()}
    ledger = gpu_ledger_total_min()
    res = dict(utc=utcnow(), files=files, features=feats, bpb=bpb, bpb_not_monotone=not_mono,
               gpu_ledger_total_min=ledger, free_disk_gb=disk_free_gb())
    h = write_json(RES_DIR / "EXTRACTION_SUMMARY.json", res)
    flist = ", ".join(k + " pass=" + str(v["pass_"]) for k, v in files.items())
    msg = (f"EXTRACTION {'DONE' if done else 'INCOMPLETE'} (S1-S4 of the plan): "
           + "; ".join(f"{s} {v['shape']} {v['npy_sha256'][:16]}..." for s, v in feats.items() if v)
           + f"; files {flist}"
           + f"; GPU ledger total {ledger:.2f} min (budget 60); free disk {res['free_disk_gb']:.1f} GB; summary json "
             f"sha256 {h}")
    log(msg)
    append_status(msg, _rel(RES_DIR / "EXTRACTION_SUMMARY.json"))


def cmd_ledger(args):  # every GPU step logs its GPU minutes and the running total PYT-AXIS
    rows = ledger_rows()
    last = rows[-1] if rows else None
    tot = gpu_ledger_total_min()
    msg = (f"GPU ledger (results/S1_2_layermax_pythia/logs/gpu_ledger.tsv): last step "
           f"{(last[0] + ' ' + last[4] + ' min (' + last[1] + ' -> ' + last[2] + ', exit ' + last[6] + ')') if last else 'none'}"
           f"; running total {tot:.3f} GPU-min of 60 over {len(rows)} GPU steps "
           f"({', '.join(r[0] + ' ' + r[4] for r in rows)})")
    log(msg)
    append_status(msg, "results/S1_2_layermax_pythia/logs/gpu_ledger.tsv")


def cmd_diskwatch(args):  # PYT-AXIS: plan section 11 watcher (S3-S8): poll every 10 s, stop the axis below 20 GB
    stop_file = LOGS_DIR / "diskwatch.stop"
    t_end = time.time() + args.max_hours * 3600
    me = os.getpid()
    min_seen = float("inf")
    log(f"diskwatch start: limit {MIN_FREE_DISK_GB} GB on the filesystem of {HF_HOME}; stop file {stop_file}")
    while time.time() < t_end and not stop_file.exists():
        free = disk_free_gb()
        min_seen = min(min_seen, free)
        if free < MIN_FREE_DISK_GB:
            ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout.splitlines()
            victims = []
            for ln in ps:
                pid, _, cmd = ln.strip().partition(" ")
                if int(pid) != me and "diskwatch" not in cmd and ("extract/pythia_axis/" in cmd
                                                                    or "analysis/s12_layermax_pythia/" in cmd):
                    victims.append((int(pid), cmd[:120]))
            for pid, _ in victims:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            msg = (f"STOP disk guard: free disk {free:.1f} GB < {MIN_FREE_DISK_GB} GB; SIGTERM sent to the axis's own "
                   f"processes {victims} (plan section 11)")
            log(msg)
            append_status(msg)
            raise SystemExit(7)
        time.sleep(args.period)
    log(f"diskwatch end: min free disk seen {min_seen:.1f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["e2", "primary", "e5", "summary", "ledger", "diskwatch"])
    ap.add_argument("--size", default=None, choices=R.SIZES)
    ap.add_argument("--period", type=float, default=10.0)
    ap.add_argument("--max-hours", type=float, default=12.0)
    args = ap.parse_args()
    check_plan_unchanged()
    require(not TEST_MODE or args.cmd in ("ledger",), "run_pyt.py sub-commands other than ledger do not run in S0b "
                                                      "test mode (the tests call the functions directly)")
    require(TEST_MODE or code_frozen_line() is not None, "the S0b CODE FROZEN line is missing in STATUS.md")  # PYT-AXIS
    if args.cmd == "primary":
        require(args.size is not None, "--size is required")
    log(f"run_pyt {args.cmd} pid={os.getpid()} threads={_TH}")
    try:
        {"e2": cmd_e2, "primary": cmd_primary, "e5": cmd_e5, "summary": cmd_summary, "ledger": cmd_ledger,
         "diskwatch": cmd_diskwatch}[args.cmd](args)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"FAILED {args.cmd}: {e!r}\n{tb}")
        append_status(f"FAILED run_pyt {args.cmd}: {e!r}"[:400])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
