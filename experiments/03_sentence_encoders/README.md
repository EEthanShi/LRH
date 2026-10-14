# 03 · Sentence encoders and the depth follow-ups

**Question.** "In a second preregistered experiment, we pair the vision models with three sentence encoders instead of the language models." The test asks whether vision–language agreement with sentence encoders stays close to stable at a shared k/m (Theorem 1, Conjecture D.1). Two exploratory follow-ups (X3, X3b) replace the mid-depth vision features by last-layer features.

**Paper.** Section 5.2 (Insight 1), Appendix G (last-layer features), Figure F.2, and Appendix F.3.

**Status.** Preregistered, with one hashed addendum. "We fixed both values and the tolerance after the results above and before any sentence-encoder score." X3 and X3b are exploratory; their readings (`X3_READINGS.md`, `X3B_READINGS.md`) were fixed before their scores. We wrote the rules of X3b after X3 had produced the scores of the three last-layer pairs of DINOv2-B with the sentence encoders, and these rules include the three pairs.

| frozen file | sha256 (first 16) |
|---|---|
| `stage1/analysis/e0_sentence/ANALYSIS_PLAN_E0.md` (withheld for now, see [`prereg/README.md`](../../prereg/README.md#frozen-files-withheld-from-this-release)) | e859c6ea52f941b1 |
| `stage1/analysis/e0_sentence/ANALYSIS_PLAN_E0_ADDENDUM_1.md` | dcf0ecfc9b3e7906 |

The freeze records ship: `analysis/e0_sentence/CODE_MANIFEST_E0.json` (code hashes) and `analysis/e0_sentence/freeze/freeze_constants.json` (adopted constants), together with `analysis/e0_sentence/design/design_inputs.json`, the pinned revisions and file hashes of the three encoders, which `e0x_common.py` checks against a recorded sha256 before any download. Since the plan itself is withheld, every step that checks the plan's hash refuses to run in this checkout.

**Inputs.**
- The COCO pool and vision features of [01](../01_feature_extraction/README.md).
- COCO captions: `e0_captions.py` downloads `annotations_trainval2014.zip` from cocodataset.org and accepts it only with its recorded size and md5.
- Sentence encoders at pinned revisions: `sentence-transformers/all-mpnet-base-v2`, `facebook/contriever`, `Qwen/Qwen3-Embedding-0.6B`; X3 also uses DINOv2-B (`vit_base_patch14_dinov2.lvd142m`). Every pinned file is verified before loading, and the models are cached in `stage1/extract/e0_sentence/hf_home/`.

**Steps.**
```bash
mkdir -p stage1/extract/e0_sentence/logs stage1/results/E0_sentence/logs
S1='cd ~/hrh_v2_stage1_20261008'
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python -I extract/e0_sentence/e0_captions.py"
for enc in mpnet contriever qwen3emb06; do      # download, verify and time each encoder
  scripts/lrh_env.sh bash -c "$S1 && venv/bin/python -I extract/e0_sentence/e0_extract.py smoke $enc"
done
scripts/lrh_env.sh bash -c "$S1 && bash extract/e0_sentence/run/run_full.sh"        # full extraction, then finalize
scripts/lrh_env.sh bash -c "$S1 && bash analysis/e0_sentence/run/run_step5.sh"      # caches, diagnostics, calibration
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python -I analysis/e0_sentence/e0_guard.py check"
scripts/lrh_env.sh bash -c "$S1 && bash analysis/e0_sentence/run/run_step6.sh"      # scores and the addendum-1 evaluator
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python analysis/e0_sentence/run/e0_write_summary.py summary --write"
# exploratory depth follow-ups
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python -I extract/e0_sentence/x3/x3_extract.py"
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python -I analysis/e0_sentence/x3/x3_scores.py --workers 5"
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python -I analysis/e0_sentence/x3/x3_report.py"
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python -I analysis/e0_sentence/x3b/x3b_scores.py --workers 6"
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python -I analysis/e0_sentence/x3b/x3b_report.py"
```
The guard and the summary step check the freeze lines of the plan, the code and the extraction in the project's run log before they run; section 11 of the plan gives the order. The independent reimplementation is in `analysis/e0_sentence/indep/`, the review in `review/`.

**Outputs (shipped).**
- `stage1/results/E0_sentence/E0_summary.md` (its last section, internal notes to the authors, is removed; see [`results/README.md`](../../results/README.md)), `E0_summary.json`, `REPORT.md`, `VERDICTS_E0.json`, `eval/E0_tables.md`, `eval/E0_*.parquet`
- `stage1/results/E0_sentence/x3/X3_REPORT.md`, `X3_results.json`; `x3b/X3B_REPORT.md`, `X3B_results.json`
- plotted values: `figures/v8/F_e0_app/numbers.json`

**Compute in the experimental record (Table R.2).** Extraction of the three encoders: 22.3 min run and 0.7 min smoke test (0.38 GPU-h), peak 2.4 GB; caches, scores, and checks 1.5 to 1.9 CPU-h on at most 6 cores. Depth follow-ups: DINOv2-B extraction 0.04 GPU-h; caches, scores, and checks about 1.5 CPU-h.

**Results and failures.**
- These pairs pass the preregistered test of stability at a shared k/m for k = 10 and k = 100; mKNN also falls with m on all 15 pairs.
- Two preregistered predictions linking the slope to the level of mKNN remained undecided.
- Exploratory: with last-layer vision features, agreement at k/m = 0.01 rises by a median factor of 2.8 with language models and 4.3 with sentence encoders (on COCO with ViT-B models); DINOv2-B and MPNet reach 0.43 with centered features and five averaged captions.
- The training data of MPNet include COCO caption pairs, probably from the 2014 training images outside our validation pool (unverified). The ten pairs of Contriever and Qwen3-Embedding-0.6B meet the rule of P-S1 without MPNet (post hoc).
- Deviation: the addendum fixed a programming error in the file-writing step of the frozen evaluation code. The plan describes the pool caption as drawn at random by the dataset author, whereas the pool holds the last caption of every image, and every computation used this stored caption.
- Deviation: the review recomputed the labels ahead of the independent reimplementation, in the reverse of the order listed in the plan, and the summary waited for the agreement of both.
