# Stage-1 feature extraction protocol (role: extraction, v1.0, 2026-10-08)

Code: `extract/{common,registry,build_items,extract_core,run_queue,prefetch}.py`. Each sidecar JSON records the sha256 of
every code file used.

## 1. Outputs

```
features/<ds>/items.parquet          frozen item table (row i of every array = item idx i)
features/<ds>/items_manifest.json    provenance, sha256 of the table, caption/text/image hash digests
features/<ds>/<slug>.npy             float32 (m, n_layers, d): RAW pooled features (no clip, no L2, no centering)
features/<ds>/<slug>.tokmean.npy     ViTs only: float32 (m, n_layers, d), mean over patch tokens
features/<ds>/<slug>.lm_aux.npz      LMs only: n_tokens (m,), nll_mean (m,) nats/token, bpb (m,) bits/byte,
                                     last_prenorm (m, d) = token mean of the last decoder layer output BEFORE final norm
features/<ds>/<slug>.json            sidecar: model id, hub id, revision/snapshot hash, weight-file blob sha256,
                                     layer names/indices, pooling, transform, tokenizer, dtype, fp32 flags, checks, timings, env
features/<ds>/<slug>.done            written last; <slug>.failed on error
features/QUEUE.done | QUEUE.failed   end-of-queue marker
```
`<slug>` = model id with `/` replaced by `__` (timm names are kept, e.g. `vit_base_patch16_224.mae`).
`<ds>` ∈ {`wit1024`, `wit4096_sel`, `coco_pool`}. Smoke-test outputs live under `features/_smoke/` (m = 16; not for analysis).

Preprocessing for metrics (the PRH/ARH outlier clamp `prh_clip+l2`, or plain `l2`) is part of the criterion Q and is
applied at analysis time. Nothing here is clipped or normalised.

## 2. Datasets

| ds | source (pinned) | m | order | LM text |
|---|---|---|---|---|
| `wit1024` (S) | `minhuh/prh` branch `wit_1024` = snapshot `0220f2b7b28bacc1f7034f358e46e39bd9078a03` (verified with `HfApi.list_repo_refs`) | 1024 | dataset order = sorted shard names, then row order (checked equal to `load_dataset(..., streaming=True)` on all 1024 rows) | `text[0]` unchanged (caption_idx 0, no prompt) |
| `wit4096_sel` (S_sel) | `minhuh/prh` branch `wit_4096` = snapshot `c41c877fc431ea20589dc0d8a57fb5aee1167baa` | 1024 | dataset rows 1025–2048 (1-based) = 0-based 1024..2047 in dataset order (checked against `load_dataset` on 2048 rows) | `text[0]` unchanged |
| `coco_pool` | `sayakpaul/coco-30-val-2014` snapshot `abdde3200f533ddae5bed2438057f1f7ea2d5131` | 16384 | ascending `sha256("hrh-v2-coco-pool-v1:" + str(image_id))`; image_id parsed from `COCO_val2014_<12 digits>.jpg`; first 16384 = pool | `caption.strip()` |

**S_sel caveat (important).** `wit_4096` is not `wit_1024` followed by 3072 new items. About 1024 of its 4096 items carry
an S caption, scattered over the rows (most with differently encoded image bytes). Within rows 1024..2047 exactly one
item overlaps S: S_sel `idx` 953 (dataset row 1977, `image_2778.jpg`; same caption as an S item, different image bytes).
It is flagged in `items.parquet` (`overlap_S_caption0`, `overlap_S_anycaption`, `overlap_S_image`) and in the manifest
(`overlap_rows`). Features are extracted for all 1024 rows; **analysis must drop flagged rows** (m_sel = 1023 usable).
One caption pair is duplicated inside S_sel (`dup_caption0_within_sel`).

**COCO pool.** The dataset has 30000 rows with 30000 distinct image ids and one caption per image, so deduplication
drops nothing. The stored caption was sampled at random by the dataset author; it is not necessarily COCO's first
annotation. The full 30000-item permutation is saved in `coco_pool/coco_all_order.parquet`; pool rows 16384..29999 can
be extracted later as a separate dataset without touching the pool. Nested m-sweeps use the prefixes `idx < m`. Other
nested draws (other salts) must permute within the 16384 pool, or extend the extraction.

**Images.** Decoded exactly as `datasets.Image().decode_example` does (PIL; EXIF transpose only when an Orientation
tag exists, and no item in any of the three tables has one). WIT images are all RGB. 30 COCO pool images are grayscale
(`L`) and are converted with `convert("RGB")`. The count is recorded per sidecar (`n_converted_to_rgb`).

## 3. Models

Vision (timm 1.0.29, weights from the `timm/<name>` HF repos, model.safetensors; snapshot hash and blob sha256 per sidecar):
- WIT (S1.2): the 17 ARH `val` ViTs (`aristotelian/prh/prh_models.py` @785054b; identical in PRH `tasks.py` @dcd76ba
  and SOG @be654a37).
- COCO (S1.1): `vit_base_patch14_dinov2.lvd142m`, `vit_base_patch16_clip_224.laion2b`, `vit_base_patch16_224.mae`,
  `vit_base_patch16_224.augreg_in21k`, `vit_base_patch16_siglip_224.webli`.
  SigLIP-B/16 is pinned to the `.webli` tag (original SigLIP). In timm 1.0.29 the untagged name resolves to `.v2_webli`
  (SigLIP 2).

Language (transformers 5.15, pinned commits, loaded from the local snapshot dir with `local_files_only=True`):

| model | revision |
|---|---|
| bigscience/bloomz-560m | a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05 |
| bigscience/bloomz-1b1 | 3c4e8ecc1fbeb3a127a9869dd5a2d0f34560577d |
| bigscience/bloomz-1b7 | 4ea0141c75b2b96e17bc41f0d52c5368c879a281 |
| bigscience/bloomz-3b | 31eefcb2bcd69632925adf07e090debafe95436d |
| bigscience/bloomz-7b1 | fa72440b228d7f42a42ec3681cf09735ad14822f |
| openlm-research/open_llama_3b | 141067009124b9c0aea62c76b3eb952174864057 (pytorch_model.bin) |
| openlm-research/open_llama_7b | 6fb184ff23774c25bf84b3628e49c8b78372c7be (pytorch_model*.bin) |
| huggyllama/llama-7b | 4782ad278652c7c71b72204d462d6d01eaaf7549 (not gated) |
| EleutherAI/pythia-410m-seed{1..9} | main commits (see `registry.py`); main == `step143000` (identical LFS sha256 of pytorch_model.bin, checked for all 9) |

No `trust_remote_code`. Loading from local snapshot directories also avoids transformers' automatic safetensors
conversion for `.bin`-only repos, which would otherwise open a conversion PR on the Hub.

## 4. Extraction conventions (replicated from PRH `extract_features.py` @dcd76ba / ARH `prh_experiment.py` @785054b)

- **ViTs.** `timm.create_model(name, pretrained=True, num_classes=0)` (ARH form); eval transform
  `create_transform(**resolve_data_config(model.pretrained_cfg, model=model))` (DINOv2 therefore runs at 518×518,
  crop_pct 1.0). Feature = CLS token (index 0) of the residual stream at `blocks.{i}.add_1` = output of block i, for every
  block, with no final norm. Captured by forward hooks on `model.blocks[i]` during `forward_features`. On the first WIT
  batch of every ViT, the ARH/PRH torchvision FX extractor (`create_feature_extractor(..., blocks.{i}.add_1)`) is run and
  compared (`fx_check` in the sidecar). Smoke runs gave max relative difference ≈ 2e-7 to 3e-6, which is fp32
  batch-size rounding.
- **SigLIP (COCO only)** has no class token (`global_pool='map'`). Its primary feature is therefore the mean over all
  patch tokens of each block output. This is a protocol choice: the MAP head is not used.
- **`tokmean`** (all ViTs) = mean over tokens `[num_prefix_tokens:]`, i.e. patch tokens only, accumulated in float64.
- **LMs.** `AutoTokenizer.from_pretrained(path, use_fast=True)` (ValueError → slow), as in ARH/PRH.
  `AutoModelForCausalLM.from_pretrained(path, dtype=float32)`, then `model(input_ids, attention_mask,
  output_hidden_states=True, use_cache=False)`. Each feature is the mean over all tokens of every hidden state,
  including the embedding output (L+1 states). The last state is post-final-norm, which is the HF convention used by
  PRH/ARH. The semantics are verified per model on item 0 (`hidden_state_semantics`):
  - hs[0] = input to layer 0;
  - hs[i] = output of layer i−1 (difference 0.0);
  - hs[L] = final_norm(output of the last layer) (difference 0.0).
  The pre-norm last layer is saved in `lm_aux.last_prenorm`. Per-item NLL and bits-per-byte follow the PRH
  `utils.cross_entropy_loss` / `cross_entropy_to_bits_per_unit` definitions and are computed in float64.

## 5. Layer axes and the relative-depth rule

- WIT arrays hold all layers:
  - ViT: axis = blocks 0..L−1;
  - LM: axis = hidden_states 0..L.
- COCO arrays hold relative depths r ∈ {0.25, 0.5, 0.75, 1.0} with ℓ(r) = floor(r·L + 0.5) on the residual-stream
  index:
  - ViT: stored block index ℓ−1. For L=12, ℓ = 3,6,9,12, so the blocks are 2,5,8,11.
  - LM: hidden_states[ℓ]. Pythia-410m and BLOOMZ-1b7 have L=24, giving 6,12,18,24. OpenLLaMA-3B has L=26, giving
    7,13,20,26.
  - The indices are written in `layer_axis_*_indices` and `rel_depths_to_residual_index` in each sidecar.
- For S1.2 "relative depth 0.5 / 1.0" on WIT, use the same rule:
  - ViT array index = ℓ−1;
  - LM array index = ℓ, where index L is post-final-norm; the pre-norm last layer is in `lm_aux`.

## 6. Deviations from ARH/PRH (all deliberate)

| item | PRH @dcd76ba | ARH @785054b | here |
|---|---|---|---|
| LM dtype | bf16 (`auto_determine_dtype`), `device_map="auto"` | bf16 if supported | **fp32 weights and activations** for every model (all ≤ 7B) |
| LM batching | all texts padded to the global longest, left padding, batch 4 | batch 4, left padding per batch | **batch size 1, no padding** |
| TF32 | default | enabled (`matmul_precision("high")`, allow_tf32) | **disabled** (`fp32_precision="ieee"` for matmul and cuDNN conv) |
| pooling arithmetic | model dtype on GPU | model dtype | float64 accumulation, stored float32 |
| ViT dtype / batch | fp32, batch 4 | fp32, batch 4 | fp32 (no TF32), batch 4–64 by size (batch-size effect ≈ 1e-7 relative) |
| saved features | `torch.save` of pooled features | same; clamp applied at analysis | `.npy` float32 RAW; clamp/L2 at analysis |
| COCO text | n/a | n/a | `caption.strip()` (2393 of 16384 pool captions had leading/trailing whitespace, which would add a whitespace token in every LM); WIT text unchanged (it has none) |
| SigLIP pooling | n/a | n/a | patch-token mean (no CLS exists) |
| libraries | unpinned | unpinned | torch 2.13.0+cu132, transformers 5.15.0, timm 1.0.29, datasets 5.0.1 (SOG pinned timm 0.9.16 / transformers 5.12.1; the timm pretrained cfgs of the 17 ViTs are recorded per sidecar) |

## 7. Known caveats

- **OpenLLaMA tokenizer.** The HF tokenizer (fast and slow give identical ids in transformers 5.15) keeps runs of
  whitespace that sentencepiece's own normaliser collapses. Token ids differ from `sentencepiece.encode` in about 15/1024
  WIT captions (double spaces). This replicates PRH/ARH, which used the HF fast tokenizer. Counts are recorded per
  sidecar (`tokenizer.sp_check`). Whether PRH's transformers-4.x fast tokenizer produced the same ids cannot be verified
  here.
- **Batch-size dependence.** fp32 batch-size dependence of ViT features is ~1e-6 relative (`rerun_check`). LM features
  are bitwise reproducible at batch 1.
- Possible train/test leakage (WIT/COCO in LAION, the Pile or WebLI) is not addressed here.

## 8. Running

```
cd ~/hrh_v2_stage1_20261008/extract
../venv/bin/python -I build_items.py                      # item tables (done once; idempotent)
../venv/bin/python -I run_queue.py --smoke 16 --models ...  # smoke -> features/_smoke
setsid nohup ../venv/bin/python -I prefetch.py  > ../logs/extract_prefetch.log 2>&1 &
setsid nohup ../venv/bin/python -I run_queue.py > ../logs/extract_queue.log 2>&1 &
```
Re-running `run_queue.py` resumes. A finished (model, dataset) pair is skipped if its `.done`, sidecar, `items_sha256`,
shape, dtype and sha256 all validate.

Loading example:
```python
import numpy as np, json
X = np.load("features/wit1024/bigscience__bloomz-7b1.npy", mmap_mode="r")   # (1024, 31, 4096)
meta = json.load(open("features/wit1024/bigscience__bloomz-7b1.json"))
```
