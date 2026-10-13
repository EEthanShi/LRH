"""Model registry for the S1.2 fourth size axis, Pythia 70M..6.9B (role pyt-axis-extract, S0b, 2026-10-10).

PYT-AXIS COPY of extract/openllama_axis/registry_openllama_axis.py (sha256 885caec2...). Changes, all marked "PYT-AXIS":
the model list (7 Pythia sizes), the main pins and the step143000 commits, the weight files (.bin at the main pin; the
2.8b step143000 .bin of E1-PROV), the small files of every repo and revision, the exact download allow-lists, the Hub
facts S1 must re-observe, the expected architecture, the family label `Pythia`, the size ranks, the paths, the
loading-check patterns and the E2 reference (bloomz-560m). The OpenLLaMA pooled ARH panel, ARH ranks and the
streamed-fp32 switch are dropped: Pythia does not enter the pooled panel (plan section 1) and streamed fp32 is not
ported (plan section 2).
It extends extract/registry.py (frozen at Stage 1, sha256 5e95fe65...) and changes nothing in it.
Plan: analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md v1.0 (FROZEN, sha256 6de89603...), sections 1, 2, 3, 11.

PYT-AXIS: every value below is the plan's section 1 value (Hub metadata queried 2026-10-09 ~23:15Z by the designer
with token=False; full blob ids and sizes from design/hub_pythia_main.json and design/hub_pythia_step143000.json,
whose prefixes are the plan's). Step S1 (`download_pyt.py resolve`) re-resolves everything on ds2213 with token=False
and STOPS on any difference (plan section 1, section 2 STOP).
"""
from __future__ import annotations

import os
import re  # PYT-AXIS
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # extract/ (one level up)
from registry import LM_REVISIONS as LM_REVISIONS_STAGE1  # noqa: E402  (frozen Stage-1 registry, read-only)
from registry import REL_DEPTHS, VIT_ARH_VAL, depth_index, kind, slug  # noqa: E402,F401

# PYT-AXIS: sizes in num_params order (size rank 1..7, plan section 1 "Ranks and labels")
SIZES = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]
LM_PYT_AXIS = [f"EleutherAI/pythia-{s}" for s in SIZES]  # PYT-AXIS
SIZE_OF = {m: s for m, s in zip(LM_PYT_AXIS, SIZES)}  # PYT-AXIS
MODEL_OF = {s: m for m, s in zip(LM_PYT_AXIS, SIZES)}  # PYT-AXIS
SIZE_RANK = {m: i + 1 for i, m in enumerate(LM_PYT_AXIS)}  # PYT-AXIS
FAMILY = {m: "Pythia" for m in LM_PYT_AXIS}  # PYT-AXIS
LM_TO_EXTRACT = list(LM_PYT_AXIS)  # PYT-AXIS: all 7 are new (no Pythia model has WIT features)

# PYT-AXIS: main pins (plan section 1 "main commit (pin)") and step143000 branch commits
LM_REVISIONS = {
    "EleutherAI/pythia-70m": "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
    "EleutherAI/pythia-160m": "50f5173d932e8e61f858120bcb800b97af589f46",
    "EleutherAI/pythia-410m": "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
    "EleutherAI/pythia-1b": "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2",
    "EleutherAI/pythia-1.4b": "fedc38a16eea3bd36a96b906d78d11d2ce18ed79",
    "EleutherAI/pythia-2.8b": "2a259cdd96a4beb1cdf467512e3904197345f6a9",
    "EleutherAI/pythia-6.9b": "c0e3eee36dc47af0c49f361c74cfe459c09f7f23",
}
STEP143000 = {
    "EleutherAI/pythia-70m": "de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852",
    "EleutherAI/pythia-160m": "b56d9bee36300031aeea723b73c4d62ac7fa71a2",
    "EleutherAI/pythia-410m": "bba6a464f54bbf08fc174cfb351d9794d58af21d",
    "EleutherAI/pythia-1b": "c0fea4b917615c60370c0aeb76877f18fef30ab3",
    "EleutherAI/pythia-1.4b": "9cc5c8c8148a4e0115d9e29c6b4f21124cfe748a",
    "EleutherAI/pythia-2.8b": "dbe7ae300a54abcdc475a33907b3dff81d25709f",
    "EleutherAI/pythia-6.9b": "21bfa02e806e253fe453702c29c81d9f83617255",
}
PROV_MODEL = "EleutherAI/pythia-2.8b"  # PYT-AXIS: the one size whose main .bin differs from step143000 (E1-PROV)

# PYT-AXIS: weight files at the main pin: (file, bytes, LFS sha256 = HF cache blob name). Original uploads only;
# loaded only through torch.load(..., weights_only=True). No *.safetensors file is ever downloaded or used.
WEIGHT_FILES = {
    "EleutherAI/pythia-70m": [
        ("pytorch_model.bin", 166049099, "f629437d6fe950ffda3ef94f6a956a1bb63a2e79e03c296c92fe208999aeb092")],
    "EleutherAI/pythia-160m": [
        ("pytorch_model.bin", 375036845, "8d856725c4a8266f10568cc1269948fc851e2aed1fb230036af2eac68e9df072")],
    "EleutherAI/pythia-410m": [
        ("pytorch_model.bin", 911449213, "786bd9bcde112a970e1614807cbb23999482de65fc05c3e0795198719b5e1543")],
    "EleutherAI/pythia-1b": [
        ("pytorch_model.bin", 2090751965, "d2523eec75f5c1236ed7787bde500935c1f87874119c528cedc9f445476d38c2")],
    "EleutherAI/pythia-1.4b": [
        ("pytorch_model.bin", 2930076797, "056766b124a4ac70afcc722084d7c2b6cab80581e0d066709d72ae5c1a8640d8")],
    "EleutherAI/pythia-2.8b": [
        ("pytorch_model.bin", 5684793945, "36f0f93d1f5d44ca9c48a602c0788001a36ec10db9d1b84a40b81799d724c4f6")],
    "EleutherAI/pythia-6.9b": [
        ("pytorch_model-00001-of-00002.bin", 9911682774, "753cc06e909d686fa6b1c096a3f8474794e5d72fa7163d20fa645fff94743b58"),
        ("pytorch_model-00002-of-00002.bin", 3937306520, "1fde0bc4136f8f4cfe4ed79ffc09cce3f887b56d6f1998fc52b6f92e3942837d")],
}
WEIGHTS_TOTAL_BYTES = 26007147158  # PYT-AXIS: plan section 1 "total weights"
# PYT-AXIS: plan section 1 provenance rule: for these 6 sizes the main .bin has the LFS sha256 of the step143000 .bin
MAIN_EQUALS_STEP = [m for m in LM_PYT_AXIS if m != PROV_MODEL]

# PYT-AXIS: E1-PROV file (2.8b only) at the step143000 commit
PROV_WEIGHT_FILES = [
    ("pytorch_model.bin", 5550545787, "a294bfcf6ff2c091091d04489a3f1c233b4106d74201e152a01f492d24e6fbaf")]

# PYT-AXIS: small files (file, bytes, git blob id). Shared by all 7 repos (plan section 1): tokenizer.json,
# tokenizer_config.json, special_tokens_map.json. Per repo: config.json, README.md; 6.9b also the .bin index.
_SHARED_SMALL = [
    ("special_tokens_map.json", 99, "0204ed10c186a4c7c68f55dff8f26087a45898d6"),
    ("tokenizer.json", 2113710, "f74dfbfab8f97770a87769c739fb080c21c8bacc"),
    ("tokenizer_config.json", 396, "f1860edb10f80bcaf7b023fce47c68a23b724c23"),
]
_CONFIG = {  # (bytes, blob)
    "EleutherAI/pythia-70m": (567, "d7a9196e329eaf06d6e2802fed376e7459834236"),
    "EleutherAI/pythia-160m": (569, "b8368ff94f3bcf3088de5e9912251fc0208ae524"),
    "EleutherAI/pythia-410m": (570, "0425fa136ba3f95d9428d832cfd7bfd82c78bf1f"),
    "EleutherAI/pythia-1b": (569, "631d0e859247ad69414b72e657e7471a07403090"),
    "EleutherAI/pythia-1.4b": (570, "76f45dd29f6ae40ae5d4960a8df6386aa9cd8d5a"),
    "EleutherAI/pythia-2.8b": (571, "490234a04b8fc9587db08c7dbc7d73f99152f697"),
    "EleutherAI/pythia-6.9b": (571, "16300e33aceb80d021ab84256b555a23e4a0309a"),
}
_README = {  # (bytes, blob)
    "EleutherAI/pythia-70m": (14256, "df0253c0ab197de15c12fa7fbb7edcca9b6848a3"),
    "EleutherAI/pythia-160m": (13593, "2e8b2b93bf534833600f862c90c0b53fc6f76a46"),
    "EleutherAI/pythia-410m": (13593, "d4751d90f46c8c5fff4b7eaa8e6308189da48153"),
    "EleutherAI/pythia-1b": (13568, "9e9f3a77336c4887f46ab5d0642016e70ae0c416"),
    "EleutherAI/pythia-1.4b": (13597, "57b72646ed3f7feb4b037e07c67b3c63fbe72742"),
    "EleutherAI/pythia-2.8b": (13593, "3c0c6f5a7cad87f8d947f8470885501a4f40374b"),
    "EleutherAI/pythia-6.9b": (13593, "89f57d3f5a6cd5c81450da42316a9d0d63fa3808"),
}
SMALL_FILES = {m: sorted([("config.json",) + _CONFIG[m], ("README.md",) + _README[m]] + list(_SHARED_SMALL)
                         + ([("pytorch_model.bin.index.json", 42028, "94ed2a228c079e755b17a87f07c83f6bcaa24181")]
                            if m == "EleutherAI/pythia-6.9b" else []))
               for m in LM_PYT_AXIS}
# PYT-AXIS: 2.8b at step143000: the same 5 small files, blob ids equal to main's (plan section 1)
PROV_SMALL_FILES = sorted([("config.json",) + _CONFIG[PROV_MODEL], ("README.md",) + _README[PROV_MODEL]]
                          + list(_SHARED_SMALL))
TOKENIZER_FILES_SHARED = ("special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")  # PYT-AXIS

# PYT-AXIS: download allow-lists = exactly the section 1 file names of that repo and revision
ALLOW = {m: [f for f, _, _ in WEIGHT_FILES[m]] + [f for f, _, _ in SMALL_FILES[m]] for m in LM_PYT_AXIS}
PROV_ALLOW = [f for f, _, _ in PROV_WEIGHT_FILES] + [f for f, _, _ in PROV_SMALL_FILES]

MAIN_TOTAL_BYTES = 26022088401  # PYT-AXIS: plan section 1 "Download total: main"
PROV_TOTAL_BYTES = 5552674156  # PYT-AXIS: plan section 1 "2.8b step143000"
DOWNLOAD_TOTAL_BYTES = 31574762557  # PYT-AXIS: plan section 1

# PYT-AXIS: download order (plan section 11 S3: smallest first; for 2.8b main then step143000)
DOWNLOADS = []
for _m in LM_PYT_AXIS:
    DOWNLOADS.append(dict(model=_m, revision=LM_REVISIONS[_m], kind="main", allow=ALLOW[_m],
                          weights=WEIGHT_FILES[_m], small=SMALL_FILES[_m],
                          bytes=sum(b for _, b, _ in WEIGHT_FILES[_m]) + sum(b for _, b, _ in SMALL_FILES[_m])))
    if _m == PROV_MODEL:
        DOWNLOADS.append(dict(model=_m, revision=STEP143000[_m], kind="step143000", allow=PROV_ALLOW,
                              weights=PROV_WEIGHT_FILES, small=PROV_SMALL_FILES,
                              bytes=sum(b for _, b, _ in PROV_WEIGHT_FILES) + sum(b for _, b, _ in PROV_SMALL_FILES)))
assert sum(sum(b for _, b, _ in WEIGHT_FILES[m]) for m in LM_PYT_AXIS) == WEIGHTS_TOTAL_BYTES  # PYT-AXIS
assert sum(d["bytes"] for d in DOWNLOADS if d["kind"] == "main") == MAIN_TOTAL_BYTES  # PYT-AXIS
assert sum(d["bytes"] for d in DOWNLOADS if d["kind"] == "step143000") == PROV_TOTAL_BYTES  # PYT-AXIS
assert sum(d["bytes"] for d in DOWNLOADS) == DOWNLOAD_TOTAL_BYTES  # PYT-AXIS

# PYT-AXIS: Hub facts S1 must re-observe for every repo (plan section 1 "Hub facts (all 7)")
HUB_FACTS = dict(gated=False, private=False, disabled=False, license="apache-2.0", n_branches=155, converts=[])
# PYT-AXIS: card `datasets` per repo, as the designer's metadata (the plan's section 1 source,
# design/hub_pythia_main.json sha256 592eca9a...) shows them. Plan erratum found by the S0b mocked S1 test: plan
# section 1 summarises them as "card dataset EleutherAI/pile" for all 7, but 1b says ['the_pile'] and 1.4b
# ['EleutherAI/the_pile'] (the same corpus). S1 checks these observed values exactly; no rule depends on them.
CARD_DATASETS = {m: ["EleutherAI/pile"] for m in LM_PYT_AXIS}
CARD_DATASETS["EleutherAI/pythia-1b"] = ["the_pile"]
CARD_DATASETS["EleutherAI/pythia-1.4b"] = ["EleutherAI/the_pile"]

# PYT-AXIS: architecture (plan section 1 table). n_layers = blocks, stored = blocks + 1 hidden states.
EXPECTED = {
    "EleutherAI/pythia-70m": dict(n_layers=6, stored_layers=7, hidden_size=512, n_heads=8, num_params=70426624, lr=1.0e-3),
    "EleutherAI/pythia-160m": dict(n_layers=12, stored_layers=13, hidden_size=768, n_heads=12, num_params=162322944, lr=6.0e-4),
    "EleutherAI/pythia-410m": dict(n_layers=24, stored_layers=25, hidden_size=1024, n_heads=16, num_params=405334016, lr=3.0e-4),
    "EleutherAI/pythia-1b": dict(n_layers=16, stored_layers=17, hidden_size=2048, n_heads=8, num_params=1011781632, lr=3.0e-4),
    "EleutherAI/pythia-1.4b": dict(n_layers=24, stored_layers=25, hidden_size=2048, n_heads=16, num_params=1414647808, lr=2.0e-4),
    "EleutherAI/pythia-2.8b": dict(n_layers=32, stored_layers=33, hidden_size=2560, n_heads=32, num_params=2775208960, lr=1.6e-4),
    "EleutherAI/pythia-6.9b": dict(n_layers=32, stored_layers=33, hidden_size=4096, n_heads=32, num_params=6857302016, lr=1.2e-4),
}
# PYT-AXIS: config.json keys checked at S1/E1 (plan section 1): STOP keys must equal EXPECTED; RECORD keys are recorded
CONFIG_STOP_KEYS = dict(num_hidden_layers="n_layers", hidden_size="hidden_size", num_attention_heads="n_heads")
CONFIG_RECORD_KEYS = ("rotary_pct", "rotary_emb_base", "tie_word_embeddings", "use_parallel_residual", "vocab_size",
                      "intermediate_size", "max_position_embeddings", "layer_norm_eps", "torch_dtype", "architectures",
                      "model_type", "hidden_act")

# PYT-AXIS: fixed depths of plan section 3 (stored index floor(r*(L_stored-1)+0.5)), asserted against the formula
FIXED_DEPTHS = {0.5: [3, 6, 12, 8, 12, 16, 16], 1.0: [6, 12, 24, 16, 24, 32, 32]}
for _r, _v in FIXED_DEPTHS.items():
    assert [depth_index(_r, EXPECTED[m]["n_layers"]) for m in LM_PYT_AXIS] == _v  # PYT-AXIS

# PYT-AXIS: loading check patterns (plan section 2 "Loading check")
MISSING_OK_RE = re.compile(r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias)$")
UNEXPECTED_OK_RE = re.compile(r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$")
BUFFER_RE = re.compile(r"attention\.(bias|masked_bias|rotary_emb\.inv_freq)$")  # E1 parameter count, E1-PROV pass rule
PROV_MAIN_ONLY_OK_RE = re.compile(r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias)$")  # E1-PROV name rule
CKPT_RENAME = [(re.compile(r"^embed_out\."), "lm_head.")]  # transformers 5.15 conversion_mapping (plan R.5)

# PYT-AXIS: GPU need per load (plan section 2 "Order and memory"): 4*num_params bytes + 2 GB, plus 4 GiB headroom
def gpu_need_bytes(model: str) -> int:  # PYT-AXIS
    return 4 * EXPECTED[model]["num_params"] + 2 * 10**9


GPU_HEADROOM_BYTES = 4 * 2**30

# PYT-AXIS: E2 (environment, descriptive): bloomz-560m from the standard cache, read-only
E2_MODEL = "bigscience/bloomz-560m"
E2_REVISION = "a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05"
assert LM_REVISIONS_STAGE1[E2_MODEL] == E2_REVISION  # PYT-AXIS
E2_NPY_SHA256 = "2683baa99f94c5cd2fc8eb129117f8f325ed29b407ad3a07fc1a966682995760"  # plan section 2 E2

# PYT-AXIS: F-cache (plan section 2): directories of another use, stat-listed by name only, never opened
POLYPYTHIA_DIRS = [f"models--EleutherAI--pythia-410m-seed{k}" for k in range(1, 10)]

OUT_DIR_NAME = "wit1024_pythia_axis"  # PYT-AXIS: features/wit1024_pythia_axis/<slug>.{npy,json,lm_aux.npz,done}
TOK1_DIR_NAME = "wit1024_pythia_axis/_tok1"  # PYT-AXIS: token-0-excluded stacks (descriptive, D7)
CHECK_DIR_NAME = "wit1024_pythia_axis/_e2_check"  # PYT-AXIS: E2 re-extraction of bloomz-560m
ITEMS_DS = "wit1024"  # items come from features/wit1024/items.parquet

# PYT-AXIS: nominal GPU minutes (plan section 11 "Nominal"), used by gpu_gate_pyt.py and as launcher estimates
NOMINAL_GPU_MIN = dict(S0b_smoke=3.0, E2=0.5, S4=4.0, S6=3.0, S7_primary=7.4, S7_l2=7.4, VP6=2.0, VP6_scalar=0.5)
S4_EST_MIN = {"EleutherAI/pythia-70m": 0.30, "EleutherAI/pythia-160m": 0.35, "EleutherAI/pythia-410m": 0.45,
              "EleutherAI/pythia-1b": 0.45, "EleutherAI/pythia-1.4b": 0.55, "EleutherAI/pythia-2.8b": 0.75,
              "EleutherAI/pythia-6.9b": 1.15}  # PYT-AXIS: per-process launcher estimates, sum = nominal S4 4.0
assert abs(sum(S4_EST_MIN.values()) - NOMINAL_GPU_MIN["S4"]) < 1e-9  # PYT-AXIS


def expected_num_params_from_config(cfg: dict) -> int:  # PYT-AXIS: GPT-NeoX parameter count from config.json
    d, L, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    inter = cfg.get("intermediate_size", 4 * d)
    per_layer = 4 * d + (3 * d * d + 3 * d) + (d * d + d) + (d * inter + inter) + (inter * d + d)
    emb = V * d * (1 if cfg.get("tie_word_embeddings", False) else 2)
    return emb + L * per_layer + 2 * d
