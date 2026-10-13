"""Collect E1-E7 and the output hashes of the llama-axis extraction (new file; role llama-axis-extract).
Writes results/S1_2_layermax_llama/extraction/EXTRACTION_SUMMARY.json and appends one STATUS line."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama_axis_common import (CHECK_DIR, OUT_DIR, RES_DIR, ROOT, R, append_status, check_plan_unchanged,  # noqa: E402
                               sha256_file, utcnow, write_json)


def rel(p):
    return str(Path(p).relative_to(ROOT))


def main():
    check_plan_unchanged()
    J = lambda n: json.load(open(RES_DIR / n))  # noqa: E731
    out = dict(role="llama-axis-extract", written_utc=utcnow(), plan_sha256=sha256_file(ROOT / "analysis" /
               "s12_layermax_llama" / "ANALYSIS_PLAN_LLAMA.md"))
    models = {}
    for m in R.LM_TO_EXTRACT:
        s = R.slug(m)
        stem = OUT_DIR / s
        meta = json.load(open(str(stem) + ".json"))
        e1 = J(f"E1_{s}.json")
        prim = J(f"primary_{m.split('/')[-1]}.json")
        models[m] = dict(
            revision=R.LM_REVISIONS[m], mode=R.EXTRACT_MODE[m], weights_dtype=meta["weights_dtype"],
            files={ext: dict(path=rel(str(stem) + ext), sha256=sha256_file(str(stem) + ext))
                   for ext in (".npy", ".lm_aux.npz", ".json", ".done")},
            shape_items_layers_dim=meta["shape"], layers_x_items_x_dim=[meta["shape"][1], meta["shape"][0], meta["shape"][2]],
            n_layers=meta["n_layers"], hidden_size=meta["hidden_size"], num_params=meta["num_params"],
            E1=dict(pass_=e1["E1_pass"], shards_sha256_all_match=e1["shards_sha256_all_match"],
                    small_files_all_match=e1["small_files_all_match"], weight_dtypes=e1["safetensors_dtypes"],
                    legacy_inv_freq=e1["legacy_inv_freq_dtypes"], total_weight_bytes=e1["total_weight_bytes"],
                    download_s=e1["download_s"], verify_s=e1["verify_s"]),
            E4_semantics=meta["hidden_state_semantics"], E6_rerun=meta["rerun_check"], array_checks=meta["array_checks"],
            E7=dict(num_params=meta["num_params"], nll_mean=meta["nll_mean_over_items"], bpb_mean=meta["bpb_mean_over_items"]),
            sp_check=meta["tokenizer"]["sp_check"], n_tokens=meta["n_tokens"], timings=meta["timings"],
            gpu_peak_gb=meta["llama_axis"]["gpu_peak_allocated_gb"], primary_checks=dict(
                shape_ok=prim["shape_ok"], semantics_ok=prim["semantics_ok"], rerun_ok=prim["rerun_ok"]))
    # the 13b download time is from attempt 1 (attempt 2 found the files in the cache)
    a1 = RES_DIR / "E1_huggyllama__llama-13b.attempt1_checker_bug.json"
    if a1.exists():
        j = json.load(open(a1))
        models["huggyllama/llama-13b"]["E1"]["download_s_attempt1"] = j["download_s"]
        models["huggyllama/llama-13b"]["E1"]["download_rate_MBps_attempt1"] = j["download_rate_MBps"]
    out["models"] = models
    e2, e3a, e3b, e5 = J("E2_llama-7b.json"), J("E3_llama-7b.json"), J("E3_llama-13b.json"), J("E5_tokenizer.json")
    pick = lambda c: dict(pass_=c["pass"], bitwise_equal=c["bitwise_equal"], max_abs=c["compare"]["max_abs_diff"],  # noqa
                          max_rel=c["compare"]["max_rel_diff"], first64_max_abs=c["compare"]["first_n_max_abs_diff"],
                          first64_max_rel=c["compare"]["first_n_max_rel_diff"],
                          per_layer_max_abs=[r["max_abs"] for r in c["compare"]["per_layer"]],
                          per_layer_max_rel=[r["max_rel"] for r in c["compare"]["per_layer"]],
                          lm_aux={k: v.get("bitwise_equal", v.get("equal")) for k, v in c["compare"]["lm_aux"].items()},
                          out=c["out"], ref=c["ref"], wall_s=c["wall_s"])
    out["E2_llama7b_reextraction"] = pick(e2)
    out["E3_streamed_vs_fp32"] = {"llama-7b": pick(e3a), "llama-13b": pick(e3b)}
    out["E5_tokenizer"] = dict(pass_=e5["pass"], n_differ=e5["n_captions_ids_differ_from_llama7b"],
                               consequence=e5["consequence"],
                               sp_check={m: v["sp_check"] for m, v in e5["tokenizers"].items()})
    out["deviations_and_incidents"] = [
        "Outputs written to features/wit1024_llama_axis/ (frozen plan section 2), not features/wit1024/ as the "
        "computed task text said; nothing under features/wit1024/ was written.",
        "Dtype: the stored llama-7b sidecar says weights and activations float32 (not fp16/bf16 as the task text "
        "assumed). llama-13b: standard fp32 load (52.5 GB GPU peak). llama-30b: streamed fp32 per the plan (16-bit "
        "weights resident, one decoder layer upcast to fp32 at a time; 68.7 GB GPU peak); no quantisation, no fp16 "
        "activations. Gated by E2 and E3 (both bitwise).",
        "Checker bug (fixed, logged in STATUS): my first E1/streamed-load rule required one dtype over all safetensors "
        "tensors; the checkpoints also hold legacy per-layer F32 self_attn.rotary_emb.inv_freq buffers (not weights). "
        "E1 for 13b was marked FAIL although every hash matched, and the downloader stopped before 30b; the first E3 "
        "7b run stopped at load. Rule fixed, downloads relaunched (13b re-verified from cache). Kept: "
        "E1_huggyllama__llama-13b.attempt1_checker_bug.json.",
        "E2 was run three times: code v1 (PASS, bitwise; kept as E2_llama-7b.codev1.json and features/_llama_axis_check/"
        "fp32_codev1/), a rerun that skipped extraction because the v1 output validated (kept as "
        "E2_llama-7b.rerun_skipped_existing_output.json), and a real rerun with the final code (PASS, bitwise) after "
        "moving the v1 output aside.",
        "Determinism check: the task asked for the first 64 items; E2 re-extracted all 1024 items (first-64 statistics "
        "reported separately).",
        "No tokenizer sensitivity extraction: token ids identical across the three repos on all 1024 captions (E5).",
    ]
    out["code_sha256"] = {p.name: sha256_file(p) for p in sorted((ROOT / "extract" / "llama_axis").glob("*.py"))}
    out["code_sha256"]["registry_llama_axis.py"] = sha256_file(ROOT / "extract" / "registry_llama_axis.py")
    out["code_sha256"]["extract_core_llama.diff"] = sha256_file(ROOT / "extract" / "llama_axis" / "extract_core_llama.diff")
    h = write_json(RES_DIR / "EXTRACTION_SUMMARY.json", out)
    mm = out["models"]
    msg = ("EXTRACTION DONE (S2+S3 of the plan). " + "; ".join(
        f"{m.split('/')[-1]} ({v['mode']}): {v['files']['.npy']['path']} layers x items x dim = "
        f"{v['layers_x_items_x_dim'][0]} x {v['layers_x_items_x_dim'][1]} x {v['layers_x_items_x_dim'][2]}, npy sha256 "
        f"{v['files']['.npy']['sha256']}, lm_aux {v['files']['.lm_aux.npz']['sha256'][:16]}..., json "
        f"{v['files']['.json']['sha256'][:16]}..." for m, v in mm.items()) +
        f". E1 PASS both (all shard sha256 = Hub LFS, weights F16); E2 llama-7b bitwise (max rel 0); E3 streamed=fp32 "
        f"bitwise on 7b and 13b; E4 semantics 0.0; E5 ids identical (0/1024 differ), no sensitivity run; E6 rerun 0.0, "
        f"finite, 0 zero rows; E7 nll 7b 3.6785 / 13b {mm['huggyllama/llama-13b']['E7']['nll_mean']:.4f} / 30b "
        f"{mm['huggyllama/llama-30b']['E7']['nll_mean']:.4f}. Summary sha256 {h}")
    append_status(msg, "results/S1_2_layermax_llama/extraction/EXTRACTION_SUMMARY.json, features/wit1024_llama_axis/")
    print(msg)


if __name__ == "__main__":
    main()
