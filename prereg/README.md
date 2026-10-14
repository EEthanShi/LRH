# Preregistration

We call an experiment preregistered if we fixed its predictions, evaluation settings, and decision rules before running it and recorded the hash and time of this plan. A prediction is frozen before scoring if we computed it from single-model features before any cross-model score of the new models, and an analysis is post hoc if we formulated the tested statement after seeing the result. An analysis is exploratory if it lacks a decision rule fixed in advance or if its plan labels it exploratory.

The frozen files are not copied into this folder. They ship byte-identical next to the code that reads them, because several pipelines read them or check their hashes at run time. This folder holds the index below, [`SHA256SUMS`](SHA256SUMS) for the 32 frozen files in this release, and [`SHA256SUMS.withheld`](SHA256SUMS.withheld) for 10 frozen files that are withheld for now (see [below](#frozen-files-withheld-from-this-release)), all with repository-relative paths.

```bash
bash scripts/verify_prereg.sh          # sha256sum -c prereg/SHA256SUMS; each withheld file is checked once it is present
```

## Frozen files by experiment

| Experiment | Status | Frozen files (sha256, first 16 hex digits; *withheld* marks a file not in this release) |
|---|---|---|
| [00](../experiments/00_criteria_library/README.md) Criterion library and known-answer checks | preregistered | `stage1/SPEC.md` (e781e44f6b809e90), `stage1/SPEC_CHANGELOG.md` (02b8ba5871eae7ab), `stage1/prereg/PREREG_stage1.md` (bee0583c2bfd3ea9) with its record `PREREG_stage1.sha256` (c6f2f717996abb5f), `stage1/prereg/LEAD_DECISIONS_v1.1.md` (eaebd68787a7111f) |
| [01](../experiments/01_feature_extraction/README.md) Feature extraction | protocol frozen | `stage1/extract/PROTOCOL.md` (e013f98b5dd963f4) |
| [02](../experiments/02_evaluation_size/README.md) Evaluation size | preregistered; the out-of-sample test of the crossing law was not run | `stage1/analysis/s11/ANALYSIS_PLAN_S11.md` (62fa39c637f9e604), `stage1/analysis/s11/ANALYSIS_PLAN_S11_v1.1.md` and its identical copy in `s11_select/` (736ec1d4c9d20a84), `stage1/hrhv2/PREDICTOR_CHOICE.md` (75e1ba3dec56b5db), `stage1/hrhv2/PREDICTOR_CHOICE_v1.1.md` (f83b13a9e2a78f42) |
| [03](../experiments/03_sentence_encoders/README.md) Sentence encoders | preregistered, with one hashed addendum; the depth follow-ups X3 and X3b are exploratory | `stage1/analysis/e0_sentence/ANALYSIS_PLAN_E0.md` (e859c6ea52f941b1, *withheld*), `ANALYSIS_PLAN_E0_ADDENDUM_1.md` (dcf0ecfc9b3e7906) |
| [04](../experiments/04_spanning_trees_cka/README.md) Spanning trees and CKA | preregistered | `stage1/analysis/s12/ANALYSIS_PLAN_S12.md` (889a24db388777cf), under the Stage-1 preregistration |
| [05](../experiments/05_layer_max/README.md) Layer maximum, BLOOMZ | preregistered; applies the layer-maximum protocol of ARH; the threshold analysis is post hoc | `stage1/analysis/s12_layermax/ANALYSIS_PLAN_LAYERMAX.md` (dd1897214d449aaa) |
| [05](../experiments/05_layer_max/README.md) Layer maximum, LLaMA-1 | forecast frozen before scoring; outcomes compared with predictions without a confirmatory test | `stage1/analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md` (990428e3f29c7ecf), `PARTB_PREDICTIONS.md` (f9afa919957d01ee) |
| [05](../experiments/05_layer_max/README.md) Layer maximum, OpenLLaMA | forecast frozen before scoring; outcomes compared with predictions without a confirmatory test | `stage1/analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md` (4f18d710beda4549, *withheld*), `PARTB_PREDICTIONS_OL.md` (f3527d09fedf5927), `VERDICTS_SPEC.md` (61da992ed1a171e3) |
| [05](../experiments/05_layer_max/README.md) Layer maximum, Pythia | forecast frozen before scoring; confirmatory test preregistered | `stage1/analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md` (6de896036146993a, *withheld*), `PARTB_PREDICTIONS_PYT.md` (64a5b83a5ae67413, *withheld*), `VERDICTS_SPEC_PYT.md` (96dce74e27aa69b0) |
| [06](../experiments/06_linear_autoencoder/README.md) Linear autoencoder | preregistered in the Stage-1 preregistration; implementation plans v1.0 and v1.1 | `stage1/linae/ANALYSIS_PLAN.md` (bccacb10d1d25494) with `ANALYSIS_PLAN.sha256` (faba0ab4c820e11b), `stage1/linae/ANALYSIS_PLAN_v1.1.md` (7329052ac8bedb2c) with `ANALYSIS_PLAN_v1.1.sha256` (0fe5ad477659451a) |
| [07](../experiments/07_gauge_networks/README.md) Gauge networks, first experiment | preregistered | `stage2/prereg/PREREG_stage2.md` (6ce39abe7094034d) with `PREREG_stage2.sha256` (46c4a13cba75e092), `PREREG_stage2_amendment1.md` (dfce9385fd76a206), `eps_S2_2.json` (b46e3f1dd62e3579), `CODE_AT_PREREG.sha256` (fc1f54d271c781f8) |
| [07](../experiments/07_gauge_networks/README.md) Gauge networks, second experiment | preregistered | `stage2/prereg/PREREG_S2_2_v1.1.md` (3c9d32171d303566, *withheld*), `stage2/results/S2_2_v11/eps/eps_S2_2_v11.json` (dd649d74fb3a87fd), `stage2/results/S2_2_v11/freeze/CODE_AT_PREREG_v11.sha256` (f2d5d28dfda1b393) |
| [08](../experiments/08_finite_gibbs/README.md) Finite Gibbs model | preregistered, with four hashed amendments | `stage2/s23/prereg/PREREG_S2_3_v1.0.md` (7a93b7a9519b505e, *withheld*), `AMENDMENT_A1_resume.md` (8a858f5394c4be17, *withheld*), `AMENDMENT_A2_complete.md` (435260f60abeba90, *withheld*), `AMENDMENT_A3_analysis_memo.md` (d32110645e1fc414, *withheld*), `AMENDMENT_A4_p5b_zero.md` (c92194b69112d0ce, *withheld*), `CODE_AT_PREREG.sha256` (f9560f0a9d3d2bc4); the N* prediction made before the runs is in `stage2/s23/results/prestep/frozen_predictions.json` |
| [09](../experiments/09_contrastive_networks/README.md) Contrastive networks | preregistered; the prediction failed | `stage2/prereg/PREREG_stage2.md`, `PREREG_stage2_amendment1.md`, `eps_S2_1.json` (0b28027d7366fb75), `CODE_AT_PREREG.sha256` (shared with 07) |

## Code-hash records

Where a plan froze the hashes of its code, the record ships as written. A static check of this checkout against the shipped records gives:

| record | result |
|---|---|
| `stage2/prereg/CODE_AT_PREREG.sha256` | all 8 files match |
| `stage2/results/S2_2_v11/freeze/CODE_AT_PREREG_v11.sha256` | all 37 shipped files match; the 38th entry is the withheld plan `PREREG_S2_2_v1.1.md`, with the same sha256 as in `SHA256SUMS.withheld`; the record names the library files by their absolute paths on the experiment machine |
| `stage2/s23/prereg/CODE_AT_PREREG.sha256` | all 37 files match |
| `stage2/s23/resume/A1_MANIFEST.sha256` to `A4_MANIFEST.sha256` | every shipped file matches; each record also lists its amendment, which is withheld (same sha256 as in `SHA256SUMS.withheld`), and the lists of finished run items are not shipped |
| `stage1/linae/ANALYSIS_PLAN_v1.1.sha256` | all files match, including the recorded reporting-only revision of `analyze_s13a.py` |
| `stage1/linae/ANALYSIS_PLAN.sha256` (v1.0) | all files match the v1.0 code, which ships in `stage1/linae/code_v1.0/` |
| `stage1/prereg/PREREG_stage1.sha256`, `stage2/prereg/PREREG_stage2.sha256` | all files match |

The code freezes of the OpenLLaMA, Pythia and sentence-encoder pipelines were recorded as hash lines in the internal run log of the project, which is not part of this repository. The sentence-encoder freeze also wrote `stage1/analysis/e0_sentence/CODE_MANIFEST_E0.json`, which ships.

## Amendments and deviations

Stage-2 amendment 1 and addendum 1 of the sentence-encoder plan ship as written. Amendments A1 to A4 of the finite Gibbs model, two of which fixed programming errors in the frozen analysis code, are withheld for now; their sha256 are listed below, and the amendment runners and their hash records (`stage2/s23/resume/`) ship. The paper's appendix lists every deviation from the frozen plans; each experiment README summarizes the ones that concern it.

## Frozen files that are not shipped

| file | reason |
|---|---|
| `stage1/analysis/s12_layermax_pythia/orig_copy/VERDICTS_SPEC.md` | identical copy of the OpenLLaMA `VERDICTS_SPEC.md` |
| `stage1/results/S1_2_layermax_openllama/partB/PARTB_PREDICTIONS_OL.generated_draft.md` | generated draft; the final `PARTB_PREDICTIONS_OL.md` ships |
| `stage1/results/S1_2_layermax_pythia/partB/PARTB_PREDICTIONS_PYT.generated_draft.md` | generated draft; the final `PARTB_PREDICTIONS_PYT.md` ships |
| `stage2/results/S2_2_v11/dryrun/eps/eps_S2_2_v11.json` | tolerances of the excluded dry run, not the frozen main ones |

## Frozen files withheld from this release

Besides their plans, these frozen files hold internal working notes of the project (local file paths and notes addressed to the authors). A frozen file cannot be edited without changing its hash, so these files are withheld for now. Each one is replaced by a short `*.WITHHELD.md` note at its path, and its sha256 is listed in [`SHA256SUMS.withheld`](SHA256SUMS.withheld), so that the file can be checked against the recorded hash if it is released later. Code that reads or hashes one of these files refuses to run without it.

| file | experiment | size (bytes) | sha256 |
|---|---|---|---|
| `stage1/analysis/e0_sentence/ANALYSIS_PLAN_E0.md` | [03](../experiments/03_sentence_encoders/README.md) | 60,375 | `e859c6ea52f941b18ca27b83150f489dded0085a4d99218d909b36c19603f0ee` |
| `stage1/analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md` | [05](../experiments/05_layer_max/README.md) | 74,977 | `4f18d710beda45492b1e191f74580bfc4ef53f6dafe791ad245ff1cf335f3240` |
| `stage1/analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md` | [05](../experiments/05_layer_max/README.md) | 99,200 | `6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b` |
| `stage1/analysis/s12_layermax_pythia/PARTB_PREDICTIONS_PYT.md` | [05](../experiments/05_layer_max/README.md) | 22,251 | `64a5b83a5ae67413168d612bfe447bcd43e20fd6c5afd2d7e662b8489699d600` |
| `stage2/prereg/PREREG_S2_2_v1.1.md` | [07](../experiments/07_gauge_networks/README.md) | 61,124 | `3c9d32171d303566d947d31ea7be6a236547174c1cd2f2509aa724a792194e4d` |
| `stage2/s23/prereg/PREREG_S2_3_v1.0.md` | [08](../experiments/08_finite_gibbs/README.md) | 48,950 | `7a93b7a9519b505ee887467baab7aa724d2a0ee05635a2afdc363da280e6b5d6` |
| `stage2/s23/prereg/AMENDMENT_A1_resume.md` | [08](../experiments/08_finite_gibbs/README.md) | 15,442 | `8a858f5394c4be17a5162374992585f405a9cfc75fff8c38074860da9761948d` |
| `stage2/s23/prereg/AMENDMENT_A2_complete.md` | [08](../experiments/08_finite_gibbs/README.md) | 16,392 | `435260f60abeba90c5874d5633b32ce76c9d60c74ce372bce2fd5173c625bed6` |
| `stage2/s23/prereg/AMENDMENT_A3_analysis_memo.md` | [08](../experiments/08_finite_gibbs/README.md) | 12,084 | `d32110645e1fc414fc963ee15a4d8fd7dbab364a2398963de28b8d8ecad4d338` |
| `stage2/s23/prereg/AMENDMENT_A4_p5b_zero.md` | [08](../experiments/08_finite_gibbs/README.md) | 11,072 | `c92194b69112d0ceac3aca9db8a26ba93cdcbd2a6eb1e3ce0f1cc9ca188a82e2` |
