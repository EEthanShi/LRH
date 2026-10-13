# OpenLLaMA axis: review recomputation (R)

Written 2026-10-09T20:09:19Z by review/ol_axis_review.py (sha256 00c98f2a6fa95078c416605f27596d7b65473729c8814082ac7507803b59de07); mode: partB-frozen.

R recomputed every verdict of VERDICTS_SPEC.md sections A, B, C, D, F, G, J from the pair files of E and, separately, of I (plan section 3 and section 11 step S8). All OpenLLaMA-axis results are descriptive (no confirmatory endpoint, plan R.1.2); J is EXPLORATORY.

## 0-mismatch status

| comparison | sections | result | mismatches |
|---|---|---|---|
| results/S1_2_layermax_openllama/review/E/verdicts_review.json vs results/S1_2_layermax_openllama/early/verdicts_ol.json | EI | PASS | 0 |
| results/S1_2_layermax_openllama/review/E/verdicts_review.json vs results/S1_2_layermax_openllama/forecast/E/verdicts_forecast.json | F | PASS | 0 |
| results/S1_2_layermax_openllama/review/E/verdicts_review.json vs results/S1_2_layermax_openllama/joint/E/verdicts_joint.json | J | PASS | 0 |
| results/S1_2_layermax_openllama/review/I/verdicts_review.json vs results/S1_2_layermax_openllama/indep/verdicts_ol.json | EI | PASS | 0 |
| results/S1_2_layermax_openllama/review/I/verdicts_review.json vs results/S1_2_layermax_openllama/forecast/I/verdicts_forecast.json | F | PASS | 0 |
| results/S1_2_layermax_openllama/review/I/verdicts_review.json vs results/S1_2_layermax_openllama/joint/I/verdicts_joint.json | J | PASS | 0 |
| results/S1_2_layermax_openllama/review/E/verdicts_review.json vs results/S1_2_layermax_openllama/review/I/verdicts_review.json | ALL | PASS | 0 |

Required comparisons (R vs E, I, forecast, joint): **all pass, 0 mismatches**.

## Categories reached (as recomputed by R)

| key | R-E | R-I |
|---|---|---|
| PO1|prh_clip+l2|holds | True | True |
| PO2|prh_clip+l2|s_cal|cat | loss reproduced | loss reproduced |
| PO2|prh_clip+l2|unc|cat | loss reproduced | loss reproduced |
| PO2|prh_clip+l2|s_cal|kedge | False | False |
| PO2ai|prh_clip+l2|median | 0.0018844044983174768 | 0.0018844044983175323 |
| PO2ai|prh_clip+l2|in_band | True | True |
| PO2aii|prh_clip+l2|verdict | holds | holds |
| PO2b|prh_clip+l2|s_cal|verdict | holds | holds |
| PO2b|prh_clip+l2|s_cal|n_decided | 30 | 30 |
| PO2b|prh_clip+l2|s_cal|x_correct | 30 | 30 |
| PO2b|prh_clip+l2|unc|verdict | holds | holds |
| PO3|prh_clip+l2|0.5|s_cal|desc | keeps sign | keeps sign |
| PO3|prh_clip+l2|1.0|s_cal|desc | keeps sign | keeps sign |
| PO4|prh_clip+l2|s_cal|holds | True | True |
| PO4|prh_clip+l2|unc|holds | True | True |
| PO4|prh_clip+l2|arh|holds | True | True |
| PO5a|prh_clip+l2|verdict | holds | holds |
| PO5|prh_clip+l2|M|cat | loss reproduced | loss reproduced |
| readO|prh_clip+l2|cka_biased | loses | loses |
| readO|prh_clip+l2|cka_debiased | not applied | not applied |
| dd|11|prh_clip+l2|cka_biased | 0.10315641689935967 | 0.10315641689936017 |
| dd|11|prh_clip+l2|cka_biased|ci_lo | -0.016730398525735657 | -0.01673039852573506 |
| dd|11|prh_clip+l2|cka_biased|ci_hi | 0.2388829954985636 | 0.23888299549856365 |
| dd|11|prh_clip+l2|cka_biased|pge | 0.12214874438832772 | 0.12214874438832772 |
| rule7a|prh_clip+l2 | direction not established | direction not established |
| size_not_reproduced|prh_clip+l2 | True | True |
| fp6|prh_clip+l2|cat | not decidable at this panel size (knife edge) | not decidable at this panel size (knife edge) |
| plugin|prh_clip+l2|within | True | True |
| D3|prh_clip+l2|nviol | 0 | 0 |
| PO1|l2|holds | True | True |
| PO2|l2|s_cal|cat | no category (not material) | no category (not material) |
| PO2|l2|unc|cat | no category (not material) | no category (not material) |
| PO2|l2|s_cal|kedge | False | False |
| PO2ai|l2|median | 0.002079246882021449 | 0.002079246882021449 |
| PO2ai|l2|in_band | True | True |
| PO2aii|l2|verdict | holds | holds |
| PO2b|l2|s_cal|verdict | holds | holds |
| PO2b|l2|s_cal|n_decided | 32 | 32 |
| PO2b|l2|s_cal|x_correct | 32 | 32 |
| PO2b|l2|unc|verdict | holds | holds |
| PO3|l2|0.5|s_cal|desc | keeps sign | keeps sign |
| PO3|l2|1.0|s_cal|desc | keeps sign | keeps sign |
| PO4|l2|s_cal|holds | True | True |
| PO4|l2|unc|holds | True | True |
| PO4|l2|arh|holds | True | True |
| PO5a|l2|verdict | holds | holds |
| PO5|l2|M|cat | no category (not material) | no category (not material) |
| readO|l2|cka_biased | not applied | not applied |
| readO|l2|cka_debiased | not applied | not applied |
| dd|11|l2|cka_biased | 0.08133953392853771 | 0.08133953392853827 |
| dd|11|l2|cka_biased|ci_lo | 0.006084084863734587 | 0.006084084863734484 |
| dd|11|l2|cka_biased|ci_hi | 0.1648455827343763 | 0.16484558273437663 |
| dd|11|l2|cka_biased|pge | 0.11479657687991021 | 0.11479657687991021 |
| rule7a|l2 | replicated the direction | replicated the direction |
| size_not_reproduced|l2 | True | True |
| fp6|l2|cat | not decidable at this panel size (knife edge) | not decidable at this panel size (knife edge) |
| plugin|l2|within | True | True |
| D3|l2|nviol | 0 | 0 |
| J|s_cal|Jmin (EXPLORATORY) | -0.7058823529411764 | -0.7058823529411764 |
| J|s_cal|pmin (EXPLORATORY) | 0.011111111111111112 | 0.011111111111111112 |
| J|s_cal|Jmean (EXPLORATORY) | -1.049019607843137 | -1.049019607843137 |
| J|s_cal|pmean (EXPLORATORY) | 0.004166666666666667 | 0.004166666666666667 |
| J|s_cal|nO (EXPLORATORY) | 2 | 2 |
| J|s_cal|cond_size (EXPLORATORY) | 0.3333333333333333 | 0.3333333333333333 |
| J|unc|Jmin (EXPLORATORY) | -1.1823529411764706 | -1.1823529411764706 |
| J|unc|pmin (EXPLORATORY) | 0.000462962962962963 | 0.000462962962962963 |
| J|unc|nO (EXPLORATORY) | 1 | 1 |
| J|unc|cond_size (EXPLORATORY) | 0.3333333333333333 | 0.3333333333333333 |

J statistics carry the label "EXPLORATORY", OpenLLaMA's own n_O/6 and the conditional size above, and the provenance labels "BLOOMZ hypothesis-generating; BLOOMZ fails B4; BLOOMZ and LLaMA known before this plan".

## F1(c) birth-time audit

```
{
 "exit_code": 0,
 "pass": true,
 "partB_status_utc": "2026-10-09T19:41:32Z",
 "n_files": 517,
 "n_born_before_partB": 0,
 "min_birth": {
  "type": "f",
  "size": 5000,
  "birth_epoch": 1791575233.0,
  "birth_utc": "2026-10-09T19:47:13Z",
  "path": "results/S1_2_layermax_openllama/indep/role_hashcheck_start.json"
 },
 "note": null
}
```

## Checks and deviations

- R-E: G|llama_pairs_equal_merged_summary = True, G|rerun_equals_frozen_3b7b = True, G|n_pairs_read = 374; legacy trend() cross-check 0/168 differ; Part B key forms slug; F notes {"prh_clip+l2|kappa_recomputed_vs_partB_maxabs": 0.0, "prh_clip+l2|h=0.0003|source": "partB s_cal_h['0.0003']", "prh_clip+l2|h=0.0005|source": "partB s_cal_h['0.0005']", "prh_clip+l2|h=0.001|source": "partB s_cal_h['0.001']", "prh_clip+l2|h=0.0015|source": "partB s_cal_h['0.0015']", "prh_clip+l2|h=0.002|source": "partB s_cal_h['0.002']", "l2|kappa_recomputed_vs_partB_maxabs": 9.093457273406935e-05, "l2|h=0.0003|source": "partB s_cal_h['0.0003']", "l2|h=0.0005|source": "partB s_cal_h['0.0005']", "l2|h=0.001|source": "partB s_cal_h['0.001']", "l2|h=0.0015|source": "partB s_cal_h['0.0015']", "l2|h=0.002|source": "partB s_cal_h['0.002']"}; E-format consistency findings {"ol|prh_clip+l2|openlm-research__open_llama_7b|vit_base_patch16_clip_224.laion2b_ft_in12k": {"max_rel_mu_f32": 1.7509663939533883e-06}, "ol|prh_clip+l2|openlm-research__open_llama_13b|vit_base_patch16_224.augreg_in21k": {"max_rel_mu_f32": 1.245085798678195e-06}, "frz|prh_clip+l2|openlm-research__open_llama_7b|vit_base_patch16_clip_224.laion2b_ft_in12k": {"max_rel_mu_f32": 1.7509663939533883e-06}, "ol|l2|openlm-research__open_llama_7b|vit_large_patch14_clip_224.laion2b": {"max_rel_mu_f32": 3.9523249847640706e-06}, "frz|l2|openlm-research__open_llama_7b|vit_large_patch14_clip_224.laion2b": {"ma; identity issues [].
- R-I: G|llama_pairs_equal_merged_summary = True, G|rerun_equals_frozen_3b7b = True, G|n_pairs_read = 374; legacy trend() cross-check 0/168 differ; Part B key forms slug; F notes {"prh_clip+l2|kappa_recomputed_vs_partB_maxabs": 0.0, "prh_clip+l2|h=0.0003|source": "partB s_cal_h['0.0003']", "prh_clip+l2|h=0.0005|source": "partB s_cal_h['0.0005']", "prh_clip+l2|h=0.001|source": "partB s_cal_h['0.001']", "prh_clip+l2|h=0.0015|source": "partB s_cal_h['0.0015']", "prh_clip+l2|h=0.002|source": "partB s_cal_h['0.002']", "l2|kappa_recomputed_vs_partB_maxabs": 9.093457273406935e-05, "l2|h=0.0003|source": "partB s_cal_h['0.0003']", "l2|h=0.0005|source": "partB s_cal_h['0.0005']", "l2|h=0.001|source": "partB s_cal_h['0.001']", "l2|h=0.0015|source": "partB s_cal_h['0.0015']", "l2|h=0.002|source": "partB s_cal_h['0.002']"}; E-format consistency findings {}; identity issues [].
- Operational choices of R not fixed by the plan or the spec are listed in analysis/s12_layermax_openllama/review/CHANGES.md ([R-op]).
- Deviations: none recorded by this script. Any mismatch above is listed in full in recomputed_verdicts.json.

