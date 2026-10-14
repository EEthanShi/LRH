# 05 · Layer maximum: BLOOMZ, LLaMA-1, OpenLLaMA, Pythia

**Question.** ARH calibrates each score after taking its maximum over layer pairs and finds that calibration largely removes the size trend of CKA and keeps that of mKNN. Corollary 1 assumes that the calibration threshold of the maximum lies near the largest random-pairing baseline of Proposition 4. "We test this assumption and both parts of the corollary on the layer maximum along four families, BLOOMZ (0.56B to 7.1B), LLaMA-1 (7B to 30B), OpenLLaMA (3B to 13B), and Pythia (70M to 6.9B)."

**Paper.** Section 5.4 (Insight 3), Figure 4, Figure F.4, Appendices F.5 and F.6, and Table F.2.

**Status.** "The preregistered BLOOMZ experiment applied the layer-maximum protocol of ARH, and we obtained its result before deriving Corollary 1", so the threshold analysis on BLOOMZ is post hoc. "For the last three families, we fixed and recorded predictions from single-model features before computing any score involving their new models. On LLaMA-1 and OpenLLaMA, we compared each outcome with its prediction without a confirmatory test, whereas on Pythia we preregistered a confirmatory test of the loss of the CKA trend."

| family | frozen files (sha256, first 16) |
|---|---|
| BLOOMZ | `stage1/analysis/s12_layermax/ANALYSIS_PLAN_LAYERMAX.md` (dd1897214d449aaa) |
| LLaMA-1 | `stage1/analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md` (990428e3f29c7ecf), `PARTB_PREDICTIONS.md` (f9afa919957d01ee) |
| OpenLLaMA | `stage1/analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md` (4f18d710beda4549, withheld), `PARTB_PREDICTIONS_OL.md` (f3527d09fedf5927), `VERDICTS_SPEC.md` (61da992ed1a171e3) |
| Pythia | `stage1/analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md` (6de896036146993a, withheld), `PARTB_PREDICTIONS_PYT.md` (64a5b83a5ae67413, withheld), `VERDICTS_SPEC_PYT.md` (96dce74e27aa69b0) |

The `PARTB_PREDICTIONS*.md` files hold the thresholds and orders forecast from single-model features before any cross-model score of the new models. The OpenLLaMA and Pythia plans and the Pythia predictions are withheld for now, see [`prereg/README.md`](../../prereg/README.md#frozen-files-withheld-from-this-release); their sha256 are recorded, and the OpenLLaMA and Pythia guards refuse to run in this checkout without them. The `S0B_CODE_MANIFEST.json` freeze records of these two axes, which only the review and entry-check scripts read, are not shipped either.

**Inputs.** The WIT-1024 features of [01](../01_feature_extraction/README.md) for the 17 vision transformers, BLOOMZ, OpenLLaMA 3B and 7B, and LLaMA-1 7B. Each later family has its own extraction code, which downloads its models at pinned commits with `token=False` and verifies every shard: `stage1/extract/llama_axis/` (`huggyllama/llama-13b`, `llama-30b`), `stage1/extract/openllama_axis/` (OpenLLaMA 13B), and `stage1/extract/pythia_axis/` (seven Pythia sizes). The vendored ARH checks need `bash scripts/fetch_vendor.sh`.

**Steps.** Every family has two independent implementations: E (`early/`, or the top folder for BLOOMZ) and I (`indep/`). The order of steps and their gates are in each plan (section 11 for OpenLLaMA and Pythia).
```bash
S1='cd ~/hrh_v2_stage1_20261008'
# BLOOMZ (layer-maximum protocol of ARH): each step through the launcher, in the order of the plan
#   launch_step.sh NAME THREADS script.py [args]   with precompute.py, validate.py, v4_arh.py, run_layermax.py, aggregate.py, report.py
scripts/lrh_env.sh bash -c "$S1 && bash analysis/s12_layermax/launch_step.sh precompute 5 precompute.py"
scripts/lrh_env.sh bash -c "$S1 && bash analysis/s12_layermax/indep/launch_main.sh"           # implementation I
# LLaMA-1: download and extract 13B and 30B, then E (early/launch_step.sh) and I (indep/launch_*.sh)
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python -I extract/llama_axis/download_llama_axis.py"
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python extract/llama_axis/run_llama_axis.py primary --model huggyllama/llama-13b"
# OpenLLaMA and Pythia: every process starts through the axis launcher, for example
scripts/lrh_env.sh bash -c "$S1 && bash analysis/s12_layermax_openllama/launch_ol.sh oll_S2_E2 4 --gpu 4 -- venv/bin/python extract/openllama_axis/run_ol.py e2"
scripts/lrh_env.sh bash -c "$S1 && bash analysis/s12_layermax_pythia/launch_pyt.sh pyt_S4_70m 4 --gpu 0.3 -- venv/bin/python extract/pythia_axis/run_pyt.py primary --size 70m"
```
The axis launchers hold thread tokens (`flock`), keep a GPU ledger, and the guards (`guard.py`) refuse a step until the freeze lines of the plan, the code and the Part B predictions are in the run log.

**Outputs (shipped).**
- BLOOMZ: `stage1/results/S1_2_layermax/S1_2_layermax_report.md`, `layermax_summary.json`, `panel_layermax.parquet`, `indep/LAYERMAX_INDEP_REPORT.md`
- LLaMA-1: `stage1/results/S1_2_layermax_llama/early/REPORT.md`, `early/summary.json`, `indep/REPORT.md`, `review/recomputed_verdicts.json`, `extraction/EXTRACTION_SUMMARY.json`
- OpenLLaMA: `stage1/results/S1_2_layermax_openllama/layermax_openllama_summary.json`, `early/REPORT.md`, `review/REVIEW_REPORT.md`
- Pythia: `stage1/results/S1_2_layermax_pythia/layermax_pythia_summary.json`, `early/REPORT.md`, `indep/REPORT.md`, `review/REVIEW_REPORT.md`
- plotted values: `figures/v8/F_aggregation/numbers.json`, `figures/v8/F_aggregation_app/numbers.json`

**Compute in the experimental record (Table R.2).** BLOOMZ preregistered layer maximum 1,284 s on the GPU (peak 6.9 GB), independent implementation 384 + 425 s. LLaMA-1 13B and 30B extraction: forward 53 + 263 s, peak 52.5 and 68.7 GB; two implementations 321 + 195 s and 170 + 169 s; vendored ARH check 4,598 s. OpenLLaMA 13B extraction: forward 61 s, peak 52.6 GB; two implementations 244 s and 155 + 181 s. Pythia: 15.0 GPU-min for the whole axis with checks.

**Results and failures.**
- The CKA trend is lost or reversed along BLOOMZ (0.89 to −0.02 under s<sub>cal</sub>), LLaMA-1 (0.68 to −0.85), and OpenLLaMA (0.88 to 0.18), and kept along Pythia (0.93 to 0.89), where the raw score grows by 0.024 per doubling of parameters against 0.0063 for the threshold.
- In every family, the trends of mKNN and debiased CKA move by 0.06 or less under calibration; with the preprocessing of PRH, the calibrated CKA trend keeps its sign at mid depth and at the last layer in all four families.
- The forecast thresholds yield a predicted order after calibration in 30 of the 34 comparisons involving OpenLLaMA 13B and in 352 of the 357 Pythia comparisons, and every predicted order is correct.
- Along Pythia, the preregistered prediction of a drop of at least 0.10 was not confirmed.
- On LLaMA-1, calibration of each layer pair before the maximum also reversed the trend, to −0.88, although we had predicted in advance that the trend would keep its sign under this protocol.
- With three sizes each, every exact p-value of a size trend along LLaMA-1 or OpenLLaMA is at least 1/6, and size, depth, and width grow together along these families.
- The loss of the CKA trend under calibration rests on BLOOMZ, LLaMA-1, and OpenLLaMA, with exact one-sided p-values of at least 1/120 along BLOOMZ and 1/6 along the other two families. On Pythia, the one family with a confirmatory test, the raw score outgrew the threshold and calibration kept the trend. This loss therefore still awaits a confirmatory test on a family that meets the condition of part 2 of Corollary 1.

**Deviations** (details in the shipped `CHANGES*.md` files and in the post hoc readings `interpretation.txt`):
- The BLOOMZ main run started before the vendored ARH check finished; its outputs were opened after the check passed.
- LLaMA-1: the automated order check inspected a directory left unused by the two implementations, so the order rests on the creation times of the output files, all later than the freeze of the predictions; for about 77 minutes concurrent validation jobs exceeded the plan's cap of six CPU threads.
- OpenLLaMA: we saw the exploratory joint statistic of the three families in a log of one implementation before the agreement check of the two implementations, with its inputs, code, and rule frozen beforehand; read-only check scripts were added after the code freeze; a fixed-depth baseline statement of the plan fails at the last layer under plain ℓ<sub>2</sub>.
- Pythia: the two implementations differ in 163 pair-level and 3 descriptive floating-point values by at most 9.1 × 10<sup>−14</sup>, so the preregistered agreement check failed by its letter; it was accepted with an added absolute tolerance of 10<sup>−12</sup> in a post hoc ruling, and a third implementation reproduces both with 0 mismatches.
