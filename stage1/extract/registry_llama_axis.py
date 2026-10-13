"""Model registry for the S1.2 secondary LLaMA size axis (role llama-axis-extract, 2026-10-08).

New file. It extends extract/registry.py (frozen at Stage 1, sha256 5e95fe65...) and changes nothing in it.
Plan: analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md v1 (sha256 990428e3...), section 1 and section 2.

Pins were resolved with HfApi().model_info(repo, token=False, files_metadata=True) on 2026-10-08 (no token used or
configured). All three repos: gated=False, private=False, disabled=False. The resolved main commits equal the pins
frozen in the plan. Shard sha256 values are the Hub LFS sha256 (= HF cache blob name), and E1 checks each downloaded
blob against them by recomputing sha256 over the file content.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import LM_REVISIONS as LM_REVISIONS_STAGE1  # noqa: E402  (frozen Stage-1 registry, read-only)
from registry import LM_WIT, REL_DEPTHS, VIT_ARH_VAL, depth_index, kind, slug  # noqa: E402,F401

# Pinned commits (main as of 2026-10-08). llama-7b is the Stage-1 pin (already extracted in features/wit1024/).
LM_REVISIONS = {
    "huggyllama/llama-7b": "4782ad278652c7c71b72204d462d6d01eaaf7549",
    "huggyllama/llama-13b": "bf57045473f207bb1de1ed035ace226f4d9f9bba",
    "huggyllama/llama-30b": "2b1edcdb3c7ced7bce6c1aa75c94545777c3118b",
}
assert LM_REVISIONS["huggyllama/llama-7b"] == LM_REVISIONS_STAGE1["huggyllama/llama-7b"]

LM_LLAMA_AXIS = ["huggyllama/llama-7b", "huggyllama/llama-13b", "huggyllama/llama-30b"]
LM_TO_EXTRACT = ["huggyllama/llama-13b", "huggyllama/llama-30b"]  # llama-65b skipped (does not fit; no quantisation)

# Safetensors shards: (file, size in bytes, LFS sha256). pytorch_model*.bin duplicates are never downloaded.
WEIGHT_SHARDS = {
    "huggyllama/llama-7b": [
        ("model-00001-of-00002.safetensors", 9976579217, "d43476fdd2fca0c44d55ee930039dd5dafb6331764dc0b5e5f89c60b551fcc12"),
        ("model-00002-of-00002.safetensors", 3500297444, "94712c4b457815e0f8aaa4f78170aebba2b9ee3339ff2b40fceb983338fbafee"),
    ],
    "huggyllama/llama-13b": [
        ("model-00001-of-00003.safetensors", 9948693455, "48118f785c89f786a1243d0bdec28f5a24c054b77d77bece58d082400bc9d065"),
        ("model-00002-of-00003.safetensors", 9904129563, "6ad4cf0a196552b13a0b7b74f023e2d5ef3a3750f738b395dbd8a2a9cc7b0af6"),
        ("model-00003-of-00003.safetensors", 6178962382, "534f1a58a5894355e8aedd9d4f312cd3ab6e5708ca446c154b0913936cdacb3d"),
    ],
    "huggyllama/llama-30b": [
        ("model-00001-of-00007.safetensors", 9818304536, "154e2b45db8542b21b16db3d1a60216936bb7d61de7795074ca46e02b8a4f832"),
        ("model-00002-of-00007.safetensors", 9958081294, "f141f0fae159b72e8019ab23cec1275f46497f23330b1083a6ba773fedf28b7d"),
        ("model-00003-of-00007.safetensors", 9896713232, "8dbc39c9178aca271399122be9f7438cd60384b18c1307e5e8894c30e01b9f0d"),
        ("model-00004-of-00007.safetensors", 9869449664, "5bb79933d1f957745c3099fba478e0a55c653c98a03e35b1159bf80b65509045"),
        ("model-00005-of-00007.safetensors", 9869449664, "acb4f57bd91eea6f7dba298e2a5263fef83dd41d5cbca2b6fe4ebbb6ae8b53d0"),
        ("model-00006-of-00007.safetensors", 9958081304, "ffc901be61646fcae0228df7651408092150983210369c657aced2f2f9c872ab"),
        ("model-00007-of-00007.safetensors", 5687891898, "28d1f0c0a686719860ec794f66484d41db8d03c6608e5619b312cedd145d9350"),
    ],
}

# Non-weight files: (file, size, git blob id). tokenizer.model is an LFS file (value = LFS sha256).
SMALL_FILES = {
    "huggyllama/llama-13b": [
        ("LICENSE", 10646, "b1c9239ba64af3ac330790fbdc8a2282612bf437"),
        ("README.md", 473, "2fc43f713ed09b0421277a7c1bda2fc97337f3aa"),
        ("config.json", 595, "cfa72b4709759990ab5771cf66a3f01fabc2b6a1"),
        ("generation_config.json", 137, "55d7b5b6db760f8c1963be3d56a3bc363bacdfb1"),
        ("model.safetensors.index.json", 33444, "d7cf668857ba82e2876c1fd532d8b7f4d5474658"),
        ("pytorch_model.bin.index.json", 33444, "dd9a5597b14a925bbe5a278e71f5cffd1ad35096"),
        ("special_tokens_map.json", 411, "d85ba6cb6820b01226ef8bd40b46bb489041c6a8"),
        ("tokenizer.json", 1842665, "a0201ea3b8e89373271760acaf53782ea760be62"),
        ("tokenizer.model", 499723, "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"),
        ("tokenizer_config.json", 700, "48d369b8fcca5effcdbbf8c1a8659a754da40dec"),
    ],
    "huggyllama/llama-30b": [
        ("LICENSE", 10646, "b1c9239ba64af3ac330790fbdc8a2282612bf437"),
        ("README.md", 473, "08eecc7b793b480aafdac5c90f0c627d6df32368"),
        ("config.json", 503, "79308f84631a1b369307d64f98258b02c2c1ef64"),
        ("generation_config.json", 137, "55d7b5b6db760f8c1963be3d56a3bc363bacdfb1"),
        ("model.safetensors.index.json", 50084, "5bbf1c98d01e025e2c25f633e988c5aff237ffb7"),
        ("pytorch_model.bin.index.json", 50084, "c76e80de8e9a4f0c983114aa75dc463c773b63e6"),
        ("special_tokens_map.json", 411, "d85ba6cb6820b01226ef8bd40b46bb489041c6a8"),
        ("tokenizer.json", 1842665, "a0201ea3b8e89373271760acaf53782ea760be62"),
        ("tokenizer.model", 499723, "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"),
        ("tokenizer_config.json", 700, "48d369b8fcca5effcdbbf8c1a8659a754da40dec"),
    ],
}

# Download filter: the Stage-1 LM pattern (extract_core.lm_allow_patterns for safetensors repos) plus LICENSE and
# README.md (plan section 1). Never *.py (no remote code), never pytorch_model*.bin.
ALLOW_PATTERNS = ["*.json", "*.model", "tokenizer*", "*.txt", "*.safetensors", "LICENSE", "README.md"]

# Architecture expected from the plan (checked against config.json and the loaded model).
EXPECTED = {
    "huggyllama/llama-7b": dict(n_layers=32, hidden_size=4096, stored_layers=33, num_params=6738415616),
    "huggyllama/llama-13b": dict(n_layers=40, hidden_size=5120, stored_layers=41, num_params=None),
    "huggyllama/llama-30b": dict(n_layers=60, hidden_size=6656, stored_layers=61, num_params=None),
}

# Extraction mode (plan section 2). fp32 = Stage-1 path (AutoModelForCausalLM fp32, whole model on the GPU).
# streamed_fp32 = 16-bit weights resident on the GPU, each decoder layer upcast to fp32 just before its forward and
# restored after; embedding, final norm, lm_head and rotary buffers fp32. Same fp32 arithmetic on the same fp32 values.
EXTRACT_MODE = {
    "huggyllama/llama-7b": "fp32",
    "huggyllama/llama-13b": "fp32",
    "huggyllama/llama-30b": "streamed_fp32",
}

FAMILY = {m: "LLaMA" for m in LM_LLAMA_AXIS}

# ARH LLM rank = prh_models.get_models("val") order (= analysis/s12_layermax/lm_core.ARH_LLM_ORDER).
ARH_LLM_ORDER = [
    "bigscience/bloomz-560m", "bigscience/bloomz-1b1", "bigscience/bloomz-1b7",
    "bigscience/bloomz-3b", "bigscience/bloomz-7b1", "openlm-research/open_llama_3b",
    "openlm-research/open_llama_7b", "openlm-research/open_llama_13b", "huggyllama/llama-7b",
    "huggyllama/llama-13b", "huggyllama/llama-30b", "huggyllama/llama-65b",
]
POOLED_LMS = LM_WIT + LM_TO_EXTRACT  # 10-LM pooled ARH panel
ARH_RANK = {m: ARH_LLM_ORDER.index(m) + 1 for m in POOLED_LMS}
DENSE_RANK = {m: i + 1 for i, m in enumerate(sorted(ARH_RANK, key=ARH_RANK.get))}

OUT_DIR_NAME = "wit1024_llama_axis"  # features/wit1024_llama_axis/<slug>.{npy,json,lm_aux.npz,done}
CHECK_DIR_NAME = "_llama_axis_check"  # E2/E3/E5 check outputs (not for analysis)
ITEMS_DS = "wit1024"  # items come from features/wit1024/items.parquet (sha256 7716bf4d...)
