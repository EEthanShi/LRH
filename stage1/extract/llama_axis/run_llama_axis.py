"""Driver for the LLaMA-axis extraction and its checks E2-E7 (new file; role llama-axis-extract, 2026-10-08).

Plan: analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md v1 section 2. Uses extract_core_llama.py (a copy of
extract/extract_core.py; only LLAMA-AXIS-marked changes). Threads: OMP/MKL/OPENBLAS and torch <= 4.

  e2       llama-7b, Stage-1 fp32 path, all 1024 WIT items -> features/_llama_axis_check/fp32/; must reproduce
           features/wit1024/huggyllama__llama-7b.npy (sha256 1a727b78...): bitwise expected, <= 1e-6 rel accepted,
           else STOP. Also saves the fp32-load rotary inv_freq (reference for the streamed path).
  e3 M     streamed-fp32 path of M (llama-7b or llama-13b) -> features/_llama_axis_check/streamed_fp32/; compared with
           the fp32 path (7b: stored Stage-1 file; 13b: the primary file). Bitwise expected; <= 1e-6 rel required.
  primary M  extraction in registry mode (13b fp32, 30b streamed_fp32) -> features/wit1024_llama_axis/.
           30b is gated on E2 and E3 (7b, 13b) having passed.
  e5       tokenizer ids across the three repos on all 1024 captions (CPU).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ["HF_HUB_OFFLINE"] = "1"  # extraction never contacts the Hub
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llama_axis_common import (CHECK_DIR, OUT_DIR, REL_TOL, RES_DIR, STAGE1_7B, STAGE1_7B_NPY_SHA256, R,  # noqa: E402
                               append_status, check_plan_unchanged, compare_features, log, sha256_file, utcnow,
                               write_json)

ITEMS = R.ITEMS_DS
ROPE_REF = CHECK_DIR / "fp32" / "rope_inv_freq_llama-7b_fp32load.npy"


def _torch_setup():
    import torch
    torch.set_num_threads(4)
    return torch


def _rel(p: Path) -> str:
    return str(p).split("hrh_v2_stage1_20261008/")[-1]


def _read(path: Path):
    return json.load(open(path)) if Path(path).exists() else None


def _gate(name: str) -> bool:
    j = _read(RES_DIR / name)
    return bool(j and j.get("pass"))


def _sidecar_summary(stem: Path) -> dict:
    meta = json.load(open(str(stem) + ".json"))
    return dict(shape=meta["shape"], npy_sha256=meta["files_sha256"]["primary"]["sha256"],
                lm_aux_sha256=meta["files_sha256"]["lm_aux"]["sha256"], json_sha256=sha256_file(str(stem) + ".json"),
                num_params=meta["num_params"], n_layers=meta["n_layers"], hidden_size=meta["hidden_size"],
                weights_dtype=meta["weights_dtype"], semantics=meta["hidden_state_semantics"],
                rerun=meta["rerun_check"], array_checks=meta["array_checks"], sp_check=meta["tokenizer"]["sp_check"],
                nll=meta["nll_mean_over_items"], bpb=meta["bpb_mean_over_items"], n_tokens=meta["n_tokens"],
                timings=meta["timings"], llama_axis=meta.get("llama_axis"))


def _semantics_ok(sem: dict) -> bool:
    return bool(sem and sem.get("hs0_vs_layer0_input") == 0.0 and sem.get("hs_i_vs_layer_i-1_output_max") == 0.0
                and sem.get("hsL_vs_finalnorm(lastlayer)") == 0.0)


def run_extraction(model: str, mode: str, out_dir: Path, rope_ref=None, rope_out=None, extra=None, tokenizer_dir=None):
    torch = _torch_setup()
    import extract_core_llama as ec
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ec.run_lm(model, [ITEMS], out_dir, None, log, mode=mode, rope_ref=rope_ref, rope_out=rope_out,
              tokenizer_dir=tokenizer_dir, extra_meta=dict(role="llama-axis-extract", plan_sha256=
                                                            "990428e3f29c7ecfba9cf314a45c73c79f0154f86a07520429910d282c0a2fc8",
                                                            **(extra or {})))
    dt = time.time() - t0
    stem = out_dir / R.slug(model)
    ok, why = ec.is_valid(out_dir, ITEMS, model, 1024)
    if not ok:
        raise RuntimeError(f"output invalid for {model} in {out_dir}: {why}")
    del torch
    return stem, dt


def cmd_e2(args):
    import numpy as np
    model = "huggyllama/llama-7b"
    out = CHECK_DIR / "fp32"
    assert sha256_file(str(STAGE1_7B) + ".npy") == STAGE1_7B_NPY_SHA256
    stem, dt = run_extraction(model, "fp32", out, rope_out=str(ROPE_REF), extra=dict(check="E2"))
    cmp = compare_features(stem, STAGE1_7B, first_n=64)
    side = _sidecar_summary(stem)
    ref_meta = json.load(open(str(STAGE1_7B) + ".json"))
    passed = bool(cmp["ok_shape"] and cmp["max_rel_diff"] <= REL_TOL)
    res = dict(check="E2", model=model, path="fp32 (Stage-1 load, copied pipeline)", out=_rel(stem) + ".npy",
               ref=_rel(STAGE1_7B) + ".npy", bitwise_equal=cmp["bitwise_equal"], pass_rule=
               "bitwise expected; max rel (per layer max|d|/max|ref|) <= 1e-6 accepted; else STOP",
               compare=cmp, sidecar=side, wall_s=dt,
               ref_nll=ref_meta["nll_mean_over_items"], ref_bpb=ref_meta["bpb_mean_over_items"],
               rope_ref=dict(path=_rel(ROPE_REF), sha256=sha256_file(ROPE_REF),
                             dtype=str(np.load(ROPE_REF).dtype), n=int(np.load(ROPE_REF).size)))
    res["pass"] = passed
    h = write_json(RES_DIR / "E2_llama-7b.json", res)
    msg = (f"E2 llama-7b re-extraction (copied pipeline, fp32 path, 1024 items x 33 layers) vs stored "
           f"huggyllama__llama-7b.npy: bitwise_equal={cmp['bitwise_equal']}, max_abs={cmp['max_abs_diff']:.3g}, "
           f"max_rel={cmp['max_rel_diff']:.3g} (first 64 items: {cmp['first_n_max_abs_diff']:.3g}/"
           f"{cmp['first_n_max_rel_diff']:.3g}); lm_aux bitwise "
           f"{ {k: v.get('bitwise_equal', v.get('equal')) for k, v in cmp['lm_aux'].items()} }; "
           f"{'PASS' if passed else 'FAIL -> STOP'}; {dt:.0f} s; E2 json sha256 {h}")
    log(msg)
    append_status(msg, f"results/S1_2_layermax_llama/extraction/E2_llama-7b.json, {_rel(stem)}.npy")
    if not passed:
        raise SystemExit(4)


def cmd_e3(args):
    import numpy as np
    model = args.model
    if not _gate("E2_llama-7b.json"):
        raise SystemExit("E2 has not passed; E3 not run")
    if model == "huggyllama/llama-7b":
        ref = STAGE1_7B
    else:
        ref = OUT_DIR / R.slug(model)
        if not Path(str(ref) + ".done").exists():
            raise SystemExit(f"primary fp32 extraction of {model} missing; run 'primary' first")
    rope_ref = np.load(ROPE_REF)
    out = CHECK_DIR / "streamed_fp32"
    stem, dt = run_extraction(model, "streamed_fp32", out, rope_ref=rope_ref, extra=dict(check="E3"))
    cmp = compare_features(stem, ref, first_n=64)
    side = _sidecar_summary(stem)
    passed = bool(cmp["ok_shape"] and cmp["max_rel_diff"] <= REL_TOL)
    res = dict(check="E3", model=model, path="streamed_fp32 vs fp32", out=_rel(stem) + ".npy", ref=_rel(ref) + ".npy",
               bitwise_equal=cmp["bitwise_equal"], pass_rule="bitwise expected; max rel <= 1e-6 required",
               compare=cmp, sidecar=side, wall_s=dt)
    res["pass"] = passed
    s = model.split("/")[-1]
    h = write_json(RES_DIR / f"E3_{s}.json", res)
    msg = (f"E3 {s} streamed-fp32 vs fp32 path (1024 items x {cmp['ref_shape'][1]} layers): "
           f"bitwise_equal={cmp['bitwise_equal']}, max_abs={cmp['max_abs_diff']:.3g}, max_rel={cmp['max_rel_diff']:.3g}; "
           f"lm_aux bitwise { {k: v.get('bitwise_equal', v.get('equal')) for k, v in cmp['lm_aux'].items()} }; "
           f"rotary inv_freq = fp32-load reference bitwise "
           f"{side['llama_axis']['load'].get('rope_check', {}).get('equals_fp32_load_bitwise')}; "
           f"GPU peak {side['llama_axis']['gpu_peak_allocated_gb']:.1f} GB; {'PASS' if passed else 'FAIL -> STOP'}; "
           f"{dt:.0f} s; E3 json sha256 {h}")
    log(msg)
    append_status(msg, f"results/S1_2_layermax_llama/extraction/E3_{s}.json, {_rel(stem)}.npy")
    if not passed:
        raise SystemExit(5)


def cmd_primary(args):
    import numpy as np
    model = args.model
    if model not in R.LM_TO_EXTRACT:
        raise SystemExit(f"{model} is not a llama-axis extraction target")
    if not _gate("E2_llama-7b.json"):
        raise SystemExit("E2 has not passed")
    s = model.split("/")[-1]
    e1 = _read(RES_DIR / f"E1_{R.slug(model)}.json")
    if not (e1 and e1.get("E1_pass")):
        raise SystemExit(f"E1 has not passed for {model}")
    mode = R.EXTRACT_MODE[model]
    rope_ref = None
    if mode == "streamed_fp32":
        for g in ("E3_llama-7b.json", "E3_llama-13b.json"):
            if not _gate(g):
                raise SystemExit(f"{g} has not passed; {model} is not extracted (no fp16 fallback)")
        rope_ref = np.load(ROPE_REF)
    stem, dt = run_extraction(model, mode, OUT_DIR, rope_ref=rope_ref, extra=dict(check="primary"))
    side = _sidecar_summary(stem)
    exp = R.EXPECTED[model]
    shape_ok = side["shape"] == [1024, exp["stored_layers"], exp["hidden_size"]] and side["n_layers"] == exp["n_layers"]
    res = dict(model=model, mode=mode, out=_rel(stem), shape_ok=shape_ok, semantics_ok=_semantics_ok(side["semantics"]),
               rerun_ok=(side["rerun"]["max_abs_diff"] == 0.0), sidecar=side, wall_s=dt,
               e1_dtype=e1["safetensors_dtypes"])
    h = write_json(RES_DIR / f"primary_{s}.json", res)
    msg = (f"extraction {model} ({mode}) -> {_rel(stem)}.npy shape {side['shape']} (layers x items x dim = "
           f"{side['shape'][1]} x {side['shape'][0]} x {side['shape'][2]}) sha256 {side['npy_sha256']}; lm_aux sha256 "
           f"{side['lm_aux_sha256']}; json sha256 {side['json_sha256']}; E4 semantics {side['semantics']}; E6 rerun "
           f"max|d|={side['rerun']['max_abs_diff']}, finite={side['array_checks']['finite']}, zero_rows="
           f"{side['array_checks']['zero_rows']}; E5 sp_check {side['sp_check']['mismatches']}/{side['sp_check']['checked']}"
           f" mismatches; E7 num_params {side['num_params']}, nll {side['nll']:.4f}, bpb {side['bpb']:.4f}; "
           f"fwd {side['timings']['forward_s']:.0f} s ({side['timings']['per_item_ms']:.1f} ms/item), load "
           f"{side['timings']['load_s']:.0f} s, GPU peak {side['llama_axis']['gpu_peak_allocated_gb']:.1f} GB; "
           f"summary json sha256 {h}")
    log(msg)
    append_status(msg, f"{_rel(stem)}.{{npy,json,lm_aux.npz,done}}, results/S1_2_layermax_llama/extraction/primary_{s}.json")
    if not (shape_ok and res["semantics_ok"] and res["rerun_ok"]):
        raise SystemExit(6)


def cmd_e5(args):
    import pandas as pd
    import sentencepiece as spm
    import extract_core_llama as ec
    from common import FEATURES, HUB
    items = pd.read_parquet(FEATURES / ITEMS / "items.parquet")
    man = json.load(open(FEATURES / ITEMS / "items_manifest.json"))
    items_sha = sha256_file(FEATURES / ITEMS / "items.parquet")
    texts = items["text"].astype(str).tolist()
    ids, info = {}, {}
    for model in R.LM_LLAMA_AXIS:
        snap = ec.ensure_lm_snapshot(model)
        tok, fast = ec._tok_load(str(snap))
        sp = spm.SentencePieceProcessor(model_file=str(snap / "tokenizer.model"))
        ids[model] = [tok(t)["input_ids"] for t in texts]
        sp_mis = 0
        for t, x in zip(texts, ids[model]):
            y = x[1:] if x and x[0] == tok.bos_token_id else x
            sp_mis += int(y != sp.encode(t))
        info[model] = dict(cls=type(tok).__name__, fast=fast, bos=tok.bos_token_id, eos=tok.eos_token_id,
                           tokenizer_config_sha256=sha256_file(snap / "tokenizer_config.json"),
                           tokenizer_json_sha256=sha256_file(snap / "tokenizer.json"),
                           tokenizer_model_sha256=sha256_file(snap / "tokenizer.model"),
                           special_tokens_map_sha256=sha256_file(snap / "special_tokens_map.json"),
                           has_chat_template=bool(getattr(tok, "chat_template", None)),
                           sp_check=dict(checked=len(texts), mismatches=sp_mis))
    ref = ids["huggyllama/llama-7b"]
    diffs = {m: [i for i in range(len(texts)) if ids[m][i] != ref[i]] for m in R.LM_TO_EXTRACT}
    n_diff = {m: len(v) for m, v in diffs.items()}
    passed = all(v == 0 for v in n_diff.values())
    res = dict(check="E5", items_sha256=items_sha, items_manifest_sha256=man["items_sha256"], tokenizers=info,
               n_captions=len(texts), n_captions_ids_differ_from_llama7b=n_diff,
               differing_items={m: v[:50] for m, v in diffs.items()}, identical_all=passed,
               consequence=("ids identical: no tokenizer sensitivity extraction is needed (plan section 2 E5)" if passed
                            else "ids differ: primary keeps own tokenizer; sensitivity extraction with llama-7b "
                                 "tokenizer files required"))
    res["pass"] = passed
    h = write_json(RES_DIR / "E5_tokenizer.json", res)
    msg = (f"E5 tokenizer: token ids of llama-13b/llama-30b vs llama-7b on all {len(texts)} WIT captions differ in "
           f"{n_diff}; sp_check mismatches { {m.split('/')[-1]: info[m]['sp_check']['mismatches'] for m in info} }; "
           f"tokenizer.json/tokenizer.model identical across repos: "
           f"{len({i['tokenizer_json_sha256'] for i in info.values()}) == 1 and len({i['tokenizer_model_sha256'] for i in info.values()}) == 1}; "
           f"tokenizer_config differs only (7b has chat_template={info['huggyllama/llama-7b']['has_chat_template']}); "
           f"{res['consequence']}; E5 json sha256 {h}")
    log(msg)
    append_status(msg, "results/S1_2_layermax_llama/extraction/E5_tokenizer.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["e2", "e3", "primary", "e5"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    check_plan_unchanged()
    log(f"run_llama_axis {args.cmd} {args.model or ''} pid={os.getpid()}")
    try:
        {"e2": cmd_e2, "e3": cmd_e3, "primary": cmd_primary, "e5": cmd_e5}[args.cmd](args)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"FAILED {args.cmd} {args.model}: {e!r}\n{tb}")
        append_status(f"FAILED {args.cmd} {args.model or ''}: {e!r}"[:400], "logs/llama_axis_extract.log")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
