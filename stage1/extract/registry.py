"""Model registry and job queue for Stage-1 extraction."""
from __future__ import annotations

import math

# ARH 'val' vision set (aristotelian/prh/prh_models.py @785054b; identical in PRH tasks.py @dcd76ba and SOG @be654a37).
VIT_ARH_VAL = [
    "vit_tiny_patch16_224.augreg_in21k",
    "vit_small_patch16_224.augreg_in21k",
    "vit_base_patch16_224.augreg_in21k",
    "vit_large_patch16_224.augreg_in21k",
    "vit_base_patch16_224.mae",
    "vit_large_patch16_224.mae",
    "vit_huge_patch14_224.mae",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_giant_patch14_dinov2.lvd142m",
    "vit_base_patch16_clip_224.laion2b",
    "vit_large_patch14_clip_224.laion2b",
    "vit_huge_patch14_clip_224.laion2b",
    "vit_base_patch16_clip_224.laion2b_ft_in12k",
    "vit_large_patch14_clip_224.laion2b_ft_in12k",
    "vit_huge_patch14_clip_224.laion2b_ft_in12k",
]

# S1.2 LMs (prereg: BLOOMZ 560m..7b1, OpenLLaMA 3B/7B, LLaMA-7B). Revisions = HF main commit resolved 2026-10-08.
LM_REVISIONS = {
    "bigscience/bloomz-560m": "a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05",
    "bigscience/bloomz-1b1": "3c4e8ecc1fbeb3a127a9869dd5a2d0f34560577d",
    "bigscience/bloomz-1b7": "4ea0141c75b2b96e17bc41f0d52c5368c879a281",
    "bigscience/bloomz-3b": "31eefcb2bcd69632925adf07e090debafe95436d",
    "bigscience/bloomz-7b1": "fa72440b228d7f42a42ec3681cf09735ad14822f",
    "openlm-research/open_llama_3b": "141067009124b9c0aea62c76b3eb952174864057",
    "openlm-research/open_llama_7b": "6fb184ff23774c25bf84b3628e49c8b78372c7be",
    "huggyllama/llama-7b": "4782ad278652c7c71b72204d462d6d01eaaf7549",
    # PolyPythias 410m, final checkpoint. main == step143000 weights (identical LFS sha256 of pytorch_model.bin).
    "EleutherAI/pythia-410m-seed1": "33803c4f5a1e9a4bece59e52f17bb8755add33e1",
    "EleutherAI/pythia-410m-seed2": "da2549c8d3e03e6e927594b9060073abdb94e6ba",
    "EleutherAI/pythia-410m-seed3": "1dd84c3d638fd5b6e3c77815ca2a26f1cdf1a307",
    "EleutherAI/pythia-410m-seed4": "57c8ecb67375772335b779545246e2f3a4aed69b",
    "EleutherAI/pythia-410m-seed5": "8f2c179eee40284065226c82b9fa0d8106658a6f",
    "EleutherAI/pythia-410m-seed6": "6a2463662069b7a4f5e77cf3c222dc60ff6cd3e4",
    "EleutherAI/pythia-410m-seed7": "84f51f7862c384d3d3d4095fb3b4d0438022b39e",
    "EleutherAI/pythia-410m-seed8": "368c5c53fda3ea210d154fcf6673328fa05da6ca",
    "EleutherAI/pythia-410m-seed9": "509afc6f4768f37ebab1705b93a098ecb5d78855",
}

LM_WIT = [
    "bigscience/bloomz-560m",
    "bigscience/bloomz-1b1",
    "bigscience/bloomz-1b7",
    "bigscience/bloomz-3b",
    "bigscience/bloomz-7b1",
    "openlm-research/open_llama_3b",
    "openlm-research/open_llama_7b",
    "huggyllama/llama-7b",
]

# S1.1 COCO pool models. SigLIP-B/16 = original SigLIP (webli); timm's untagged default now resolves to SigLIP-2.
COCO_VISION = [
    "vit_base_patch14_dinov2.lvd142m",
    "vit_base_patch16_clip_224.laion2b",
    "vit_base_patch16_224.mae",
    "vit_base_patch16_224.augreg_in21k",
    "vit_base_patch16_siglip_224.webli",
]
COCO_TEXT = [f"EleutherAI/pythia-410m-seed{i}" for i in range(1, 10)] + [
    "bigscience/bloomz-1b7",
    "openlm-research/open_llama_3b",
]

REL_DEPTHS = (0.25, 0.5, 0.75, 1.0)
WIT_DATASETS = ["wit1024", "wit4096_sel"]


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def kind(model_id: str) -> str:
    return "lm" if "/" in model_id else "vit"


def depth_index(r: float, L: int) -> int:
    """Residual-stream index ell(r) = floor(r*L + 1/2) in 0..L (0 = embedding output, ell = output of block ell)."""
    return int(math.floor(r * L + 0.5))


def build_queue():
    """Phase A: WIT models (each load also does COCO if that model is in the COCO set). Phase B: remaining COCO models."""
    q = []
    for m in VIT_ARH_VAL:
        q.append(dict(model=m, kind="vit", datasets=WIT_DATASETS + (["coco_pool"] if m in COCO_VISION else [])))
    for m in LM_WIT:
        q.append(dict(model=m, kind="lm", datasets=WIT_DATASETS + (["coco_pool"] if m in COCO_TEXT else [])))
    seen = {j["model"] for j in q}
    for m in COCO_VISION + COCO_TEXT:
        if m not in seen:
            q.append(dict(model=m, kind=kind(m), datasets=["coco_pool"]))
    return q
