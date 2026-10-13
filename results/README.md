# Results index

The result files stay where the pipelines wrote them (`stage1/results/`, `stage2/results/`, `stage2/s23/results/`), because analysis and figure code read them from there. This page maps each number of the top-level README to the files it comes from. The paper rounds some of these numbers and reports medians or trends computed from the files.

## Numbers of the README

| README number | Experiment | Source files |
|---|---|---|
| BLOOMZ CKA trend 0.89 to −0.02 under s<sub>cal</sub> | [05](../experiments/05_layer_max/README.md) | `stage1/results/S1_2_layermax/S1_2_layermax_report.md`, `layermax_summary.json`, `panel_layermax.parquet` |
| LLaMA-1 0.68 to −0.85; −0.88 under per-pair calibration | [05](../experiments/05_layer_max/README.md) | `stage1/results/S1_2_layermax_llama/early/REPORT.md`, `early/summary.json`; second implementation `indep/REPORT.md`, `indep/summary.json`; review `review/recomputed_verdicts.json` |
| OpenLLaMA 0.88 to 0.18; predicted order in 30 of 34 comparisons involving OpenLLaMA 13B | [05](../experiments/05_layer_max/README.md) | `stage1/results/S1_2_layermax_openllama/layermax_openllama_summary.json`, `early/REPORT.md`, `review/REVIEW_REPORT.md` |
| Pythia 0.93 to 0.89; 0.024 against 0.0063 per doubling; 352 of 357 comparisons; drop of at least 0.10 not confirmed | [05](../experiments/05_layer_max/README.md) | `stage1/results/S1_2_layermax_pythia/layermax_pythia_summary.json`, `early/REPORT.md`, `indep/REPORT.md`, `review/REVIEW_REPORT.md` |
| Thresholds forecast from single-model features (LLaMA-1, OpenLLaMA, Pythia) | [05](../experiments/05_layer_max/README.md) | `stage1/analysis/s12_layermax_llama/PARTB_PREDICTIONS.md`, `stage1/analysis/s12_layermax_openllama/PARTB_PREDICTIONS_OL.md`, `stage1/analysis/s12_layermax_pythia/PARTB_PREDICTIONS_PYT.md` (all frozen; the Pythia file is withheld for now, with its sha256 in [`prereg/SHA256SUMS.withheld`](../prereg/SHA256SUMS.withheld)) |
| Median mKNN 0.091 to 0.0085 at k = 10; slopes −0.57, −0.21, −0.01; k = 100 within 4% of k = 10 | [02](../experiments/02_evaluation_size/README.md) | `stage1/results/S1_1/v11/S1_1_v11_summary.json`, `S1_1_v11_tables.md`, `S1_1_v11_{cells,curves,slopes}_*.parquet`; plotted values in `figures/v8/F_neighbors/numbers.json` |
| Crossing-law test not run (no estimator passed the preliminary check) | [02](../experiments/02_evaluation_size/README.md) | `stage1/results/S1_1_select/selection_report.md`, `selection_summary.json`; `stage1/hrhv2/PREDICTOR_CHOICE_v1.1.md` (frozen) |
| Sentence encoders: stability at a shared k/m for k = 10 and k = 100 | [03](../experiments/03_sentence_encoders/README.md) | `stage1/results/E0_sentence/E0_summary.md`, `E0_summary.json`, `VERDICTS_E0.json`, `REPORT.md`, `eval/E0_tables.md` |
| Shares 0.875 and 0.889 of shared MST edges; bridge overlap 93% and 100%, trend 0.57 (Holm-adjusted p = 0.23) | [04](../experiments/04_spanning_trees_cka/README.md) | `stage1/results/S1_2/S1_2_report.md`, `s12_summary.json`, `panel_pool2047.parquet`; plotted values in `figures/v8/F_mst_app/numbers.json` |
| Monotone fit adds 0.0004 and 0.0012, below 0.10 in all 136 pairs; BLOOMZ trends 0.70 and 0.84 reversing to −1.00 and −0.92 | [04](../experiments/04_spanning_trees_cka/README.md) | `stage1/results/S1_2/S1_2_report.md`, `s12_summary.json`, `panel_main.parquet`; plotted values in `figures/v8/F_criteria/numbers.json` |
| Linear autoencoder: 0 of 48 and 0 of 500 pairs; reliable for N ≥ 2<sup>12</sup>; condition numbers 4.66 against 4.67 and 1.0330 against 1.0329; plateau prediction failed | [06](../experiments/06_linear_autoencoder/README.md) | `stage1/results/S1_3a_linae_summary.md`, `S1_3a_linae_summary.json`; plotted values in `figures/v8/F_gauge/numbers.json` |
| Gauge networks: 96 of 96 pairs on neighbor sets without weight decay (against prediction P1) | [07](../experiments/07_gauge_networks/README.md) | `stage2/results/S2_2/S2_2_report.md`, `s22_summary.json`; independent recompute `stage2/results/review_final/s22-recompute/REPORT.md`, `recompute_summary.json`, `recompute_pairs.json` |
| Gauge networks, second experiment: head flow and momentum imbalance parts | [07](../experiments/07_gauge_networks/README.md) | `stage2/results/S2_2_v11/S2_2_v11_report.md`, `S2_2_v11_report_full.md`, `summary.json`, `render/a4_parts.csv` |
| Finite Gibbs model: 0 of 96 up to 2<sup>28</sup>, 94 of 96 at 2<sup>34</sup>; median 0.348 at 2<sup>40</sup>; nine of nine confirmatory predictions | [08](../experiments/08_finite_gibbs/README.md) | `stage2/s23/results/stageA/REPORT_stageA.md`, `stageA_results.json`, `N_match.json`; independent recompute `stageA/review_A2/recompute_A2.json` |
| N* = 2<sup>33.75</sup> predicted before the runs | [08](../experiments/08_finite_gibbs/README.md) | `stage2/s23/results/prestep/frozen_predictions.json` |
| CIFAR-5m contrastive networks: 64 of 64 pairs disagree on neighbor sets; monotone link removes a median 15% against 100% for the exact link | [09](../experiments/09_contrastive_networks/README.md) | `stage2/results/S2_1/S2_1_report.md`, `s21_summary.json` |
| Two implementations reproduce PRH, ARH and SOG to within 10<sup>−6</sup> | [00](../experiments/00_criteria_library/README.md) | `stage1/results/verification_report.md`, `stage1/hrhv2_ref/REPORT.md`, `stage1/results/S1_2/g1prime.json` |
| Known-answer checks on synthetic couplings | [00](../experiments/00_criteria_library/README.md) | `stage1/results/G0_report.md`, `G0_report.json` |

## Other shipped summaries

- `stage1/results/E0_sentence/x3/X3_REPORT.md`, `X3_results.json` and `x3b/X3B_REPORT.md`, `X3B_results.json`: the exploratory depth follow-ups (last-layer vision features).
- `stage1/results/S1_2_layermax/indep/LAYERMAX_INDEP_REPORT.md`: the independent implementation of the BLOOMZ layer maximum.
- `stage1/results/S1_2_layermax_llama/extraction/EXTRACTION_SUMMARY.json`: checks of the LLaMA-1 13B and 30B extraction.
- `stage2/results/review_final/s21-recompute/*.py`, `s22-recompute/*.py`: the independent recompute scripts.

## Files whose bytes differ from the experiment machine

Every result file is a byte-identical copy, except the 19 files below, which no shipped record hashes. In the `numbers.json` files, `selection_summary.json` and `layermax_pythia_summary.json`, the absolute home-directory prefix of the experiment machine was replaced by `~/`. In `stage1/hrhv2_ref/REPORT.md`, `G0_report.md` and `G0_report.json`, the host name was replaced by "the experiment server". In `layermax_pythia_summary.json`, one internal clause about a temporary file on the experiment machine was also removed, and in `E0_summary.md` its last section, internal notes to the authors in Chinese, was removed. Nothing else changed.

| file | sha256 on the experiment machine | sha256 here |
|---|---|---|
| `stage1/results/S1_1_select/selection_summary.json` | `f1fac874f3f368b34ef9214c10ea00485a6951eedbec30b5585c2c5859ca8f35` | `f1401486b1e25eab84714eeb0420b4f19362aae130dc0de11c34bb278498b6c8` |
| `stage1/results/S1_2_layermax_pythia/layermax_pythia_summary.json` | `71f090b7436a473fbd13754f772f89ada2f50e1c602c766e5f56057e18290d70` | `b3a9c802313d4d2b684285bc5187125a1492b61ea880aac7d900d509046e6be9` |
| `stage1/results/E0_sentence/E0_summary.md` | `fa54172b3a40f9f57942be4db6bb8c6ce46bc37207c8c7b95e0017dc2397563a` | `2872bd8f7b733b7d9a40bf19e5b71a1a2f3b8ad937982b6bc87f4e9fbcf16989` |
| `stage1/results/G0_report.md` | `9342c2b3a4ad93ccf9313d2b579b18f21a8c03a3d866d1092a06ac305d9109ce` | `c19c233e3844b237ff702c7cfb3b4f2a93fd18ed30479105c3ba4d1a260a8133` |
| `stage1/results/G0_report.json` | `90e43bf75d42a724a166e9423b29d849935e33b9b34dfae098c21e0f919dc39a` | `fe05263ff4d1423236324735877848ed95d7b42a32cbc2352d31cf4aad1ad213` |
| `stage1/hrhv2_ref/REPORT.md` | `a7b0b8e294e3d16bbede4db7f6e9b5e8f9f0618376b339a9e77f848870219578` | `9c5bf2e43c9fdd7078182b28c700ae378195bbda3c477426d193b3cd8f661ff9` |
| `figures/v8/F_aggregation_app/numbers.json` | `d55d81728e7d580778d5fb955a591858af2fdadbb3fbe0daf9989476a0676829` | `a26da04c410ba48c7323f6e779d7235d2bdc48fb157bc0b9d2464f41d87811bc` |
| `figures/v8/F_criteria/numbers.json` | `9a812f30b8cbeeff0b9186a450575f3bfdef3b2a0e50e2fedf7e47a4d7cdd311` | `27170ad8cd9d90de684f59e9f1f73dfacf0744560195b5a21f04cbd1afe72d03` |
| `figures/v8/F_e0_app/numbers.json` | `f8ea5b0c0581bf7ca602b249787ba6afd36e45a090290afefd036873151e5848` | `fca1a851f83fd26c99af2bcffe3aa2b2bccecdf78eed63a4950258998cc4cc40` |
| `figures/v8/F_gauge/numbers.json` | `ce3fb02e94275c95e195323bc6e7e040242d3038196193d4f6c211ac3c03cf49` | `86b0c1de47d4bad71b4ea60ced21337109be26c2ba83abc3c7922aae9259f797` |
| `figures/v8/F_gauge_app/numbers.json` | `f0e9a078299365fe75f60768ec008bcc98badaee8fa1b27e49c474a886ce6568` | `1aee63bdfc680802e28cf721b6b0bfdc64b28a829d5695e964f5d180edb0173c` |
| `figures/v8/F_gauge_v11_app/numbers.json` | `e54172d48e4bb0df8e2f9e04ea882bd5fd627787187e5f26a255dffb47875186` | `61bc4045e9fda9b7369c7b9b179f636d63a9273ea60bd72153e3b8d6638529b4` |
| `figures/v8/F_linae_app/numbers.json` | `99fc022554a23265cd220418a72af49da6d9e82c3171d37b96e308440490830e` | `7a2fd256e2a64f0f43ed792a8acc90cbb989db774fc64d74a0f9a48367aaef64` |
| `figures/v8/F_link_app/numbers.json` | `e4e984051f66f4c267fa8998982a0ff54ea16119b0c62741a85c460d5bae45da` | `df03d4fed4e1020b8b1cdf558f49c8e584d327105d5d1de0b7912433a364b7a2` |
| `figures/v8/F_link_net/numbers.json` | `a66e78c6d0d393bcf46e1663243fdfe9d7cc480bd65636d68ff55494694325be` | `6e1021072716f33a0d2bb507aa62680d3bfc824ceb367992b137fd3936bb89e8` |
| `figures/v8/F_link_top/numbers.json` | `b2c1769c2ebc0dcef2ffa031475749b323e3a8a75fe2a3227ae04b97710b46be` | `a3e6591dd18ab102ed0ac289bc0d484fcdff8720ebedf4deb5e4d3e5ee8ffea2` |
| `figures/v8/F_mst_app/numbers.json` | `ef6fb72bc334f1196d94837892e21105606b16a471da4b8fbb19ef9b357a2d48` | `027babe278e71e73ca242fd591ec8fb0d9cc5bdcc4694926618f819cf2adbd20` |
| `figures/v8/F_neighbors/numbers.json` | `b4ba2c0493542a69b0a13d9d3e494ab1ee6e656032976c93b657c374e4330c41` | `fe7fbde739db27e4960135c75d090dcfdaaf41e9954b612db52eb22772dc8d99` |
| `figures/v8/F_scale_app/numbers.json` | `bc64bf717dc63af581489e7ff3b16aadcfa0e355a89996ff187d0748952ef85c` | `92b040363f5b7bd1ef0df9ebd187bc5fc019286a27c8996f574ece5e7aab064f` |

Files that other shipped files hash keep their original bytes, including any absolute paths in them (for example `VERDICTS_E0.json`, `layermax_openllama_summary.json`, `S1_2_layermax_pythia/early/REPORT.md`, `s21_summary.json`, and every frozen file).
