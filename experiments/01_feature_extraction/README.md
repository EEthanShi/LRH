# 01 · Feature extraction

**Question.** Extract the features on which every pretrained-model experiment is scored. "We compare the 17 vision transformers of ARH with the BLOOMZ, OpenLLaMA, and LLaMA-1 language models on the 1,024 image–caption pairs of WIT used by PRH." The same queue extracts a disjoint WIT selection set and a COCO pool of 16,384 images for the evaluation-size experiment.

**Paper.** Section 5.1 and the protocol appendix.

**Status.** Protocol frozen: `stage1/extract/PROTOCOL.md` (sha256 e013f98b5dd963f4). Every sidecar JSON records the sha256 of every code file used.

**Datasets** (pinned snapshots, read from the Hugging Face cache):

| dataset folder | source | revision |
|---|---|---|
| `wit1024` | `minhuh/prh`, branch `wit_1024` | `0220f2b7b28bacc1f7034f358e46e39bd9078a03` |
| `wit4096_sel` (rows 1,025 to 2,048) | `minhuh/prh`, branch `wit_4096` | `c41c877fc431ea20589dc0d8a57fb5aee1167baa` |
| `coco_pool` (first 16,384 of a sha256 permutation) | `sayakpaul/coco-30-val-2014` | `abdde3200f533ddae5bed2438057f1f7ea2d5131` |

**Models.** Pinned in `stage1/extract/registry.py`: the 17 vision transformers of the ARH validation set, BLOOMZ, OpenLLaMA 3B and 7B, LLaMA-1 7B, and the models of the COCO panel. The later axes (LLaMA-1 13B and 30B, OpenLLaMA 13B, Pythia) have their own extraction code, described in [05](../05_layer_max/README.md); the sentence encoders and DINOv2-B are in [03](../03_sentence_encoders/README.md).

**Steps.**
```bash
# 1. the pinned dataset snapshots into the Hugging Face cache (HF_HUB_CACHE, default ~/.cache/huggingface/hub)
hf download minhuh/prh --repo-type dataset --revision 0220f2b7b28bacc1f7034f358e46e39bd9078a03
hf download minhuh/prh --repo-type dataset --revision c41c877fc431ea20589dc0d8a57fb5aee1167baa
hf download sayakpaul/coco-30-val-2014 --repo-type dataset --revision abdde3200f533ddae5bed2438057f1f7ea2d5131
# 2. frozen item tables, model downloads, then the resumable GPU queue
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && venv/bin/python extract/build_items.py'
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && venv/bin/python extract/prefetch.py'
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && venv/bin/python extract/run_queue.py'
```
`run_queue.py --smoke 16 --models <model ids>` runs a 16-item smoke test into `features/_smoke/`. The queue skips every (model, dataset) whose outputs already validate.

**Outputs.** `stage1/features/<dataset>/` (not shipped; gitignored): item tables, raw float32 features per layer, language-model auxiliaries, and one sidecar JSON per model. `PROTOCOL.md` lists every file.

**Compute in the paper.** 38.7 min on the GPU for 66 model and dataset jobs.

**Notes.** Features are stored raw (no clip, no ℓ<sub>2</sub>, no centering); the preprocessing is part of each criterion and is applied at analysis time.
