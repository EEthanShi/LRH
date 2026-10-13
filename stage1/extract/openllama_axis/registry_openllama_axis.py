"""Model registry for the S1.2 third size axis, OpenLLaMA 3B/7B/13B (role oll-axis-extract, S0b, 2026-10-09).

OL-AXIS COPY of extract/registry_llama_axis.py (sha256 1af88e93...). Changes, all marked "OL-AXIS": the model list, the
pins, the weight shards (.bin pickles: OpenLLaMA has no safetensors), the small files, the paths and the family label.
It extends extract/registry.py (frozen at Stage 1, sha256 5e95fe65...) and changes nothing in it.
Plan: analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md v1.0 (FROZEN, sha256 4f18d710...), sections 1 and 2.

OL-AXIS: the open_llama_13b values are the plan's section 1 values (Hub metadata queried 2026-10-09 ~07:35Z by the
designer with the public API and no token: gated False, private False, disabled False, main = b6d7fde8..., no
safetensors conversion; full git blob ids from that query's `siblings`, whose 8-hex prefixes are the plan's). They
are written here at S0b, before S1. Step S1 (`download_ol.py resolve`) re-resolves everything on ds2213 with
HfApi().model_info(..., token=False, files_metadata=True) and STOPS on any difference (plan section 1, section 2 STOP).
open_llama_3b / open_llama_7b: the Stage-1 pins (already extracted; their cached .bin blobs are used by E2 only).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # OL-AXIS: extract/ (one level up)
from registry import LM_REVISIONS as LM_REVISIONS_STAGE1  # noqa: E402  (frozen Stage-1 registry, read-only)
from registry import LM_WIT, REL_DEPTHS, VIT_ARH_VAL, depth_index, kind, slug  # noqa: E402,F401

# OL-AXIS: pinned commits. 3b/7b = Stage-1 pins (already extracted in features/wit1024/); 13b = plan section 1.
LM_REVISIONS = {
    "openlm-research/open_llama_3b": "141067009124b9c0aea62c76b3eb952174864057",
    "openlm-research/open_llama_7b": "6fb184ff23774c25bf84b3628e49c8b78372c7be",
    "openlm-research/open_llama_13b": "b6d7fde8392250730d24cc2fcfa3b7e5f9a03ce8",
}
assert LM_REVISIONS["openlm-research/open_llama_3b"] == LM_REVISIONS_STAGE1["openlm-research/open_llama_3b"]  # OL-AXIS
assert LM_REVISIONS["openlm-research/open_llama_7b"] == LM_REVISIONS_STAGE1["openlm-research/open_llama_7b"]  # OL-AXIS

LM_OL_AXIS = ["openlm-research/open_llama_3b", "openlm-research/open_llama_7b",  # OL-AXIS
              "openlm-research/open_llama_13b"]
LM_TO_EXTRACT = ["openlm-research/open_llama_13b"]  # OL-AXIS
LM_E2 = ["openlm-research/open_llama_3b", "openlm-research/open_llama_7b"]  # OL-AXIS: E2 drift check (plan section 2)

# OL-AXIS: weight shards are PyTorch pickles (.bin); (file, size in bytes, LFS sha256 = HF cache blob name).
# Loaded only through torch.load(..., weights_only=True) (transformers 5.15 .bin path); never weights_only=False.
WEIGHT_SHARDS = {
    "openlm-research/open_llama_3b": [  # OL-AXIS: Stage-1 snapshot (sidecar 'source'; blob name = LFS sha256)
        ("pytorch_model.bin", 6853038093, "9ffd42dc58c4f49154e98bc7796306fde40febef278e99636a240a731d626a4a"),
    ],
    "openlm-research/open_llama_7b": [  # OL-AXIS: Stage-1 snapshot (sidecar 'source')
        ("pytorch_model-00001-of-00002.bin", 9976634558, "66347239b4c055cb28879975813f20fd57807a133e8c5ec10f2287af4ffe242c"),
        ("pytorch_model-00002-of-00002.bin", 3500315539, "d8764ea9d807e0bdadd901218911aa0e7ca421f66134e08c3326846865c1c786"),
    ],
    "openlm-research/open_llama_13b": [  # OL-AXIS: plan section 1 (26,032,458,406 B in total)
        ("pytorch_model-00001-of-00003.bin", 9948728430, "ee6e332e96c88a17730c55d617b9c86e2c4b93db4f49ba3473164b312a54dae4"),
        ("pytorch_model-00002-of-00003.bin", 9904165024, "2338f70c6ea6dea1f2537e5d7fb2bfe75a18b9513bd7f9e0b3b77bf94c120c29"),
        ("pytorch_model-00003-of-00003.bin", 6178983625, "fd70b3dcf65ec979f2a68ea80722c39ba8d5894c2bfe29eb411aba54a4ca42df"),
    ],
}
# OL-AXIS: the 3 shards sum to 26,031,877,079 B. The plan's "3 .bin shards, 26,032,458,406 B" (section 1) is the total of
# ALL files at the pin (shards + the 7 small files + .gitattributes, i.e. + 581,327 B), as the Hub siblings show; the
# per-shard sizes of the plan's table are the authoritative values. Both totals are checked by S1. (Plan erratum found
# by the S0b mocked S1 test; no rule of the plan depends on the total.)
WEIGHTS_TOTAL_BYTES_13B = 26031877079  # OL-AXIS: sum of the plan's per-shard sizes
REPO_TOTAL_BYTES_13B = 26032458406  # OL-AXIS: plan section 1 total = every file at the pin

# OL-AXIS: non-weight files of open_llama_13b: (file, size, git blob id); tokenizer.model is LFS (value = LFS sha256).
# Plan section 1 prefixes: config 58dfec18, generation_config 737a3637, index dd9a5597, special_tokens_map 1c141cfe,
# tokenizer_config 75ff0f5b, tokenizer.model ab1b681e..., README 0f8514bc. .gitattributes is excluded.
SMALL_FILES = {
    "openlm-research/open_llama_13b": [
        ("README.md", 10645, "0f8514bc8d0316f298f7d7cb314c52504eff6e4d"),
        ("config.json", 507, "58dfec1899f9cee1ae011b400a008641facbaeac"),
        ("generation_config.json", 137, "737a363761e41a11a1575fe06b90729fdb7ed2b5"),
        ("pytorch_model.bin.index.json", 33444, "dd9a5597b14a925bbe5a278e71f5cffd1ad35096"),
        ("special_tokens_map.json", 330, "1c141cfedc26fa17771e88d0a1372d4a7387a3ec"),
        ("tokenizer.model", 534194, "ab1b681ec7fc02fed5edd3026687d7a692a918c4dd8e150ca2e3994a6229843b"),
        ("tokenizer_config.json", 593, "75ff0f5b6d38cd1300f7277c250615aa6ae6d241"),
    ],
}
# OL-AXIS: the tokenizer and generation files are byte-identical to the 3B and 7B repos (plan section 1).
TOKENIZER_FILES_SHARED = ("generation_config.json", "special_tokens_map.json", "tokenizer_config.json",
                          "tokenizer.model")

# OL-AXIS: download filter = exactly the 3 shards, the index and the 6 small files (plan section 11 step S3).
# Never *.py (no remote code), never *.safetensors (none exist), never .gitattributes.
ALLOW_PATTERNS = [f for f, _, _ in WEIGHT_SHARDS["openlm-research/open_llama_13b"]] + \
                 [f for f, _, _ in SMALL_FILES["openlm-research/open_llama_13b"]]

# OL-AXIS: Hub facts that S1 must re-observe (plan section 1); any difference stops the role.
HUB_FACTS_13B = dict(sha="b6d7fde8392250730d24cc2fcfa3b7e5f9a03ce8", gated=False, private=False, disabled=False,
                     lastModified="2023-06-16T05:47:29", license="apache-2.0", refs=["main"])

# Architecture expected from the plan (checked against config.json and the loaded model).
EXPECTED = {  # OL-AXIS
    "openlm-research/open_llama_3b": dict(n_layers=26, hidden_size=3200, stored_layers=27, num_params=3426473600),
    "openlm-research/open_llama_7b": dict(n_layers=32, hidden_size=4096, stored_layers=33, num_params=6738415616),
    "openlm-research/open_llama_13b": dict(n_layers=40, hidden_size=5120, stored_layers=41, num_params=13015864320,
                                           n_heads=40, intermediate_size=13824, vocab_size=32000, rms_norm_eps=1e-6,
                                           tie_word_embeddings=False, torch_dtype="float16"),
}

# OL-AXIS: extraction mode (plan section 2): standard fp32 load for every model. streamed_fp32 (16-bit weights on the
# GPU, each decoder layer upcast to fp32 before its forward) is used for 13b ONLY if the standard load fails, and then
# only after E3 (streamed = standard on open_llama_7b) passes.
EXTRACT_MODE = {m: "fp32" for m in LM_OL_AXIS}

FAMILY = {m: "OpenLLaMA" for m in LM_OL_AXIS}  # OL-AXIS

# ARH LLM rank = prh_models.get_models("val") order (= analysis/s12_layermax/lm_core.ARH_LLM_ORDER).
ARH_LLM_ORDER = [
    "bigscience/bloomz-560m", "bigscience/bloomz-1b1", "bigscience/bloomz-1b7",
    "bigscience/bloomz-3b", "bigscience/bloomz-7b1", "openlm-research/open_llama_3b",
    "openlm-research/open_llama_7b", "openlm-research/open_llama_13b", "huggyllama/llama-7b",
    "huggyllama/llama-13b", "huggyllama/llama-30b", "huggyllama/llama-65b",
]
# OL-AXIS: 11-LM pooled ARH panel (plan section 1): the 8 S1.2 LMs + llama-13b/30b (LLaMA axis) + open_llama_13b
POOLED_LMS = [m for m in ARH_LLM_ORDER if m in set(LM_WIT) | {"huggyllama/llama-13b", "huggyllama/llama-30b",
                                                              "openlm-research/open_llama_13b"}]
ARH_RANK = {m: ARH_LLM_ORDER.index(m) + 1 for m in POOLED_LMS}
DENSE_RANK = {m: i + 1 for i, m in enumerate(sorted(ARH_RANK, key=ARH_RANK.get))}
assert ARH_RANK["openlm-research/open_llama_13b"] == 8 and len(POOLED_LMS) == 11  # OL-AXIS: plan section 1

OUT_DIR_NAME = "wit1024_openllama_axis"  # OL-AXIS: features/wit1024_openllama_axis/<slug>.{npy,json,lm_aux.npz,done}
CHECK_DIR_NAME = "wit1024_openllama_axis/_e2_check"  # OL-AXIS: E2 re-extractions of 3b/7b (plan section 2, R.3.2)
E3_DIR_NAME = "wit1024_openllama_axis/_e3_check"  # OL-AXIS: E3 streamed-vs-standard check (only if needed)
E5_DIR_NAME = "wit1024_openllama_axis/_e5_tok7b"  # OL-AXIS: E5 tokenizer sensitivity (only if ids differ)
ITEMS_DS = "wit1024"  # items come from features/wit1024/items.parquet (sha256 7716bf4d...)

# OL-AXIS: stored Stage-1 features compared bitwise by E2 (plan section 2; hashes re-verified at the S0a freeze)
STAGE1_NPY_SHA256 = {
    "openlm-research/open_llama_3b": "f6f966ca8bb50301a8e1f9e317cb1bc36bdfa67dbfcc68430de228274b6e9899",
    "openlm-research/open_llama_7b": "3c7da1c40b319efb1f06034a79756e3c51fe0313cabd34fa0848750fbb60ddd0",
}
