"""S0b test fixture (role pyt-axis-extract, 2026-10-10): tiny synthetic GPT-NeoX snapshots in the Pythia checkpoint
format, with no Pythia byte. Hashed in the CODE FROZEN line; used only in S0b test mode.

* tokenizer: a byte-level BPE (vocab 1024) trained on the 1024 WIT captions, with the structure of the GPT-NeoX
  tokenizer.json (NFC normaliser, ByteLevel pre-tokenizer/post-processor/decoder, <|endoftext|> = 0, <|padding|> = 1)
  and a Pythia-style tokenizer_config.json (tokenizer_class GPTNeoXTokenizer);
* config.json in the raw Pythia style (rotary_pct, rotary_emb_base, use_parallel_residual, tie_word_embeddings false);
* pytorch_model.bin (torch.save, zip format): float16 weights, the head saved as `embed_out.weight`, and (variant
  "main") the per-layer buffers attention.bias (bool, [1,1,2048,2048]), attention.masked_bias and
  attention.rotary_emb.inv_freq of the original GPT-NeoX checkpoints; variant "step" drops bias/masked_bias, as the
  step143000 files of 2.8b do.
Variants for negative tests: perturbed (one weight element changed by one ulp), extra_nonbuf (a non-buffer name only
in main), step_only (a name only in step), missing_param (final_layer_norm.bias absent), unexpected_param (an extra
non-buffer tensor), buffer_diff (inv_freq values changed only).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

SPECIAL = ["<|endoftext|>", "<|padding|>"]
TOK_CONFIG = {"add_prefix_space": False, "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>",
              "tokenizer_class": "GPTNeoXTokenizer", "unk_token": "<|endoftext|>"}
SPECIAL_MAP = {"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>"}


def raw_config(L=3, d=64, heads=4, vocab=1024) -> dict:
    return {"architectures": ["GPTNeoXForCausalLM"], "bos_token_id": 0, "eos_token_id": 0, "hidden_act": "gelu",
            "hidden_size": d, "initializer_range": 0.02, "intermediate_size": 4 * d, "layer_norm_eps": 1e-05,
            "max_position_embeddings": 2048, "model_type": "gpt_neox", "num_attention_heads": heads,
            "num_hidden_layers": L, "rotary_emb_base": 10000, "rotary_pct": 0.25, "tie_word_embeddings": False,
            "torch_dtype": "float16", "transformers_version": "4.24.0", "use_cache": True,
            "use_parallel_residual": True, "vocab_size": vocab}


def build_tokenizer(texts: list[str], vocab: int = 1024) -> str:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL, show_progress=False,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=tr)
    return tok.to_str()


def build_state_dict(cfg: dict, variant: str = "main", seed: int = 0) -> dict:
    import torch
    from transformers import AutoConfig, GPTNeoXForCausalLM
    import tempfile
    torch.manual_seed(seed)
    with tempfile.TemporaryDirectory() as td:
        Path(td, "config.json").write_text(json.dumps(cfg))
        conf = AutoConfig.from_pretrained(td)
    model = GPTNeoXForCausalLM(conf)
    with torch.no_grad():  # larger, non-degenerate weights (the default init is tiny)
        for n, p in model.named_parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.08)
            elif n.endswith("weight"):
                p.normal_(1.0, 0.1)
            else:
                p.normal_(0.0, 0.05)
    sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("gpt_neox.rotary_emb."):
            continue
        k2 = "embed_out.weight" if k == "lm_head.weight" else k
        sd[k2] = v.detach().clone().to(torch.float16)
    L = cfg["num_hidden_layers"]
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    dim = int(head_dim * cfg["rotary_pct"])
    inv = 1.0 / (cfg["rotary_emb_base"] ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    mp = cfg["max_position_embeddings"]
    for i in range(L):
        if variant not in ("step", "perturbed", "step_only", "buffer_diff"):
            sd[f"gpt_neox.layers.{i}.attention.bias"] = torch.tril(torch.ones(mp, mp, dtype=torch.bool)).view(1, 1, mp, mp)
            sd[f"gpt_neox.layers.{i}.attention.masked_bias"] = torch.tensor(-1e9)
        sd[f"gpt_neox.layers.{i}.attention.rotary_emb.inv_freq"] = inv.clone() * (1.5 if variant == "buffer_diff" else 1.0)
    if variant == "perturbed":
        flat = sd["gpt_neox.layers.1.mlp.dense_h_to_4h.weight"].view(-1)
        flat[7] = (flat[7:8].view(torch.int16) + 1).view(torch.float16)[0]  # one ulp
    if variant == "extra_nonbuf":
        sd["gpt_neox.layers.0.attention.extra_weight"] = torch.zeros(4, dtype=torch.float16)
    if variant == "step_only":
        sd["gpt_neox.layers.0.attention.step_only_weight"] = torch.zeros(4, dtype=torch.float16)
    if variant == "missing_param":
        del sd["gpt_neox.final_layer_norm.bias"]
    if variant == "unexpected_param":
        sd["gpt_neox.extra_module.weight"] = torch.zeros(4, 4, dtype=torch.float16)
    return sd


def make_snapshot(root: Path, rev: str, texts: list[str], variant: str = "main", seed: int = 0,
                  tokenizer_json: str | None = None, cfg: dict | None = None) -> Path:
    import torch
    snap = Path(root) / rev
    snap.mkdir(parents=True, exist_ok=True)
    cfg = cfg or raw_config()
    (snap / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    tj = tokenizer_json or build_tokenizer(texts)
    (snap / "tokenizer.json").write_text(tj)
    (snap / "tokenizer_config.json").write_text(json.dumps(TOK_CONFIG, indent=2) + "\n")
    (snap / "special_tokens_map.json").write_text(json.dumps(SPECIAL_MAP) + "\n")
    (snap / "README.md").write_text(f"synthetic GPT-NeoX test fixture ({variant}); not a Pythia model\n")
    sd = build_state_dict(cfg, variant, seed)
    tmp = snap / "pytorch_model.bin.tmp"
    torch.save(sd, str(tmp))
    os.replace(tmp, snap / "pytorch_model.bin")
    return snap


def file_ids(snap: Path) -> dict:
    out = {}
    for p in sorted(Path(snap).iterdir()):
        data = p.read_bytes()
        out[p.name] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                           blob=hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest())
    return out
