# Pythia axis: review recomputation (R)

Written 2026-10-10T10:08:20Z by review/pyt_axis_review.py (sha256 2964952d31f5517f47dfb97cfb410347f455e2bac13b30aaaaf14098f25e922a); mode: partB-frozen.

R recomputed every verdict of VERDICTS_SPEC_PYT.md (sections A-D, G, H, R, F, K) from the pair files of E and, separately, of I, and Part B (plan section 3 and section 11 step S8). Confirmatory: H1 (gatekeeper) and the Holm family {H2, H3, H4} of the primary instance (prh_clip+l2, s_cal) only; everything else descriptive.

## 0-mismatch status

| R file | target | sections | status | compared | mismatches | pass |
|---|---|---|---|---|---|---|
| results/S1_2_layermax_pythia/review/E/verdicts_review.json | results/S1_2_layermax_pythia/early/verdicts_pyt.json | EI | compared | 8445 | 0 | True |
| results/S1_2_layermax_pythia/review/E/verdicts_review.json | results/S1_2_layermax_pythia/forecast/E/verdicts_forecast.json | FK | compared | 4510 | 0 | True |
| results/S1_2_layermax_pythia/review/I/verdicts_review.json | results/S1_2_layermax_pythia/indep/verdicts_pyt.json | EI | compared | 8445 | 0 | True |
| results/S1_2_layermax_pythia/review/I/verdicts_review.json | results/S1_2_layermax_pythia/forecast/I/verdicts_forecast.json | FK | compared | 4510 | 0 | True |
| results/S1_2_layermax_pythia/review/E/verdicts_review.json | results/S1_2_layermax_pythia/review/I/verdicts_review.json | ALL | compared | 12955 | 3 | False |

Required comparisons (R-E vs E and forecast/E, R-I vs I and forecast/I): **all pass, 0 mismatches**.

## Categories and decisions (as recomputed by R)

| key | R-E | R-I |
|---|---|---|
| K|prh_clip+l2|s_cal|category | C | C |
| K|prh_clip+l2|s_cal|F_reason | None | None |
| K|prh_clip+l2|s_cal|premise | None | None |
| H|prh_clip+l2|s_cal|H1|member | True | True |
| H|prh_clip+l2|s_cal|H1|D | -0.03361344537815125 | -0.03361344537815125 |
| H|prh_clip+l2|s_cal|H1|p_perm | 0.3198412698412698 | 0.3198412698412698 |
| H|prh_clip+l2|s_cal|H1|p_mag | 1.0 | 1.0 |
| H|prh_clip+l2|s_cal|H1|p | 1.0 | 1.0 |
| H|prh_clip+l2|s_cal|H1|confirmed | False | False |
| H|prh_clip+l2|s_cal|gate | closed | closed |
| H|prh_clip+l2|s_cal|holm_order | None | None |
| H|prh_clip+l2|s_cal|H2|confirmed | None | None |
| H|prh_clip+l2|s_cal|H3|confirmed | None | None |
| H|prh_clip+l2|s_cal|H4|confirmed | None | None |
| K|prh_clip+l2|s_cal|H2_reading | kept (not tested, gate closed) | kept (not tested, gate closed) |
| K|prh_clip+l2|s_cal|H3_reading | no comparison changed order | no comparison changed order |
| K|prh_clip+l2|s_cal|H4_reading | no comparison changed order | no comparison changed order |
| K|prh_clip+l2|s_cal|M1 | False | False |
| K|prh_clip+l2|s_cal|M2 | None | None |
| K|prh_clip+l2|s_cal|M3 | False | False |
| K|prh_clip+l2|s_cal|M4 | False | False |
| K|prh_clip+l2|s_cal|M5_extrap_sizes | 70m,160m | 70m,160m |
| K|prh_clip+l2|s_cal|kedge | False | False |
| K|prh_clip+l2|unc|category | C | C |
| K|prh_clip+l2|unc|F_reason | None | None |
| K|prh_clip+l2|unc|premise | None | None |
| H|prh_clip+l2|unc|H1|member | True | True |
| H|prh_clip+l2|unc|H1|D | -0.037815126050420145 | -0.037815126050420145 |
| H|prh_clip+l2|unc|H1|p_perm | 0.30158730158730157 | 0.30158730158730157 |
| H|prh_clip+l2|unc|H1|p_mag | 1.0 | 1.0 |
| H|prh_clip+l2|unc|H1|p | 1.0 | 1.0 |
| H|prh_clip+l2|unc|H1|confirmed | False | False |
| H|prh_clip+l2|unc|gate | closed | closed |
| H|prh_clip+l2|unc|holm_order | None | None |
| H|prh_clip+l2|unc|H2|confirmed | None | None |
| H|prh_clip+l2|unc|H3|confirmed | None | None |
| H|prh_clip+l2|unc|H4|confirmed | None | None |
| K|prh_clip+l2|unc|H2_reading | kept (not tested, gate closed) | kept (not tested, gate closed) |
| K|prh_clip+l2|unc|H3_reading | no comparison changed order | no comparison changed order |
| K|prh_clip+l2|unc|H4_reading | no comparison changed order | no comparison changed order |
| K|prh_clip+l2|unc|M1 | False | False |
| K|prh_clip+l2|unc|M2 | None | None |
| K|prh_clip+l2|unc|M3 | False | False |
| K|prh_clip+l2|unc|M4 | None | None |
| K|prh_clip+l2|unc|M5_extrap_sizes | 70m,160m | 70m,160m |
| K|prh_clip+l2|unc|kedge | False | False |
| PP1|prh_clip+l2|holds | True | True |
| PP2ai|prh_clip+l2|s_cal|n_models_in_band | 7 | 7 |
| PP2ai|prh_clip+l2|s_cal|holds | True | True |
| PP2aii|prh_clip+l2|s_cal|verdict | holds | holds |
| PP2a|prh_clip+l2|s_cal|verdict | holds | holds |
| PP2b|prh_clip+l2|s_cal|verdict | holds | holds |
| PP2b|prh_clip+l2|s_cal|n_decided | 352 | 352 |
| PP2b|prh_clip+l2|s_cal|x_correct | 352 | 352 |
| PP5a|prh_clip+l2|s_cal|verdict | holds | holds |
| D3|prh_clip+l2|nviol | 0 | 0 |
| K|l2|s_cal|category | A | A |
| K|l2|s_cal|F_reason | None | None |
| K|l2|s_cal|premise | verified | verified |
| H|l2|s_cal|H1|member | True | True |
| H|l2|s_cal|H1|D | -0.7394957983193278 | -0.7394957983193278 |
| H|l2|s_cal|H1|p_perm | 0.009325396825396825 | 0.009325396825396825 |
| H|l2|s_cal|H1|p_mag | 0.0004997501249375312 | 0.0004997501249375312 |
| H|l2|s_cal|H1|p | 0.009325396825396825 | 0.009325396825396825 |
| H|l2|s_cal|H1|confirmed | True | True |
| H|l2|s_cal|gate | open (H1 confirmed) | open (H1 confirmed) |
| H|l2|s_cal|holm_order | H3,H4,H2 | H3,H4,H2 |
| H|l2|s_cal|H2|confirmed | True | True |
| H|l2|s_cal|H3|confirmed | True | True |
| H|l2|s_cal|H4|confirmed | True | True |
| K|l2|s_cal|H2_reading | kept as predicted | kept as predicted |
| K|l2|s_cal|H3_reading | equivalence confirmed | equivalence confirmed |
| K|l2|s_cal|H4_reading | no comparison changed order | no comparison changed order |
| K|l2|s_cal|M1 | True | True |
| K|l2|s_cal|M2 | False | False |
| K|l2|s_cal|M3 | False | False |
| K|l2|s_cal|M4 | False | False |
| K|l2|s_cal|M5_extrap_sizes |  |  |
| K|l2|s_cal|kedge | False | False |
| K|l2|unc|category | A | A |
| K|l2|unc|F_reason | None | None |
| K|l2|unc|premise | verified | verified |
| H|l2|unc|H1|member | True | True |
| H|l2|unc|H1|D | -0.8046218487394958 | -0.8046218487394958 |
| H|l2|unc|H1|p_perm | 0.011706349206349206 | 0.011706349206349206 |
| H|l2|unc|H1|p_mag | 0.0004997501249375312 | 0.0004997501249375312 |
| H|l2|unc|H1|p | 0.011706349206349206 | 0.011706349206349206 |
| H|l2|unc|H1|confirmed | True | True |
| H|l2|unc|gate | open (H1 confirmed) | open (H1 confirmed) |
| H|l2|unc|holm_order | H3,H4,H2 | H3,H4,H2 |
| H|l2|unc|H2|confirmed | True | True |
| H|l2|unc|H3|confirmed | True | True |
| H|l2|unc|H4|confirmed | True | True |
| K|l2|unc|H2_reading | kept as predicted | kept as predicted |
| K|l2|unc|H3_reading | no comparison changed order | no comparison changed order |
| K|l2|unc|H4_reading | equivalence confirmed | equivalence confirmed |
| K|l2|unc|M1 | True | True |
| K|l2|unc|M2 | False | False |
| K|l2|unc|M3 | False | False |
| K|l2|unc|M4 | None | None |
| K|l2|unc|M5_extrap_sizes |  |  |
| K|l2|unc|kedge | False | False |
| PP1|l2|holds | True | True |
| PP2ai|l2|s_cal|n_models_in_band | 7 | 7 |
| PP2ai|l2|s_cal|holds | True | True |
| PP2aii|l2|s_cal|verdict | holds | holds |
| PP2a|l2|s_cal|verdict | holds | holds |
| PP2b|l2|s_cal|verdict | holds | holds |
| PP2b|l2|s_cal|n_decided | 300 | 300 |
| PP2b|l2|s_cal|x_correct | 300 | 300 |
| PP5a|l2|s_cal|verdict | holds | holds |
| D3|l2|nviol | 0 | 0 |

Primary instance: prh_clip+l2, biased CKA, layer-max (`full`), s_cal. l2 and uncensored rows are secondary.

## F1(c) birth-time audit

```
{
 "pass": true,
 "partB_status_utc": "2026-10-10T08:22:10Z",
 "n_files": 997,
 "n_born_before_partB": 0,
 "n_birth_unknown": 0,
 "min_birth": {
  "type": "f",
  "size": 7689,
  "birth_epoch": 1791621174.0,
  "birth_utc": "2026-10-10T08:32:54Z",
  "path": "results/S1_2_layermax_pythia/early/validation/VP9_known_answers.json"
 },
 "note": null,
 "source": "guard.audit (shared)"
}
```

## Checks and deviations

- R-E: G|n_pairs_read = 238, G|l2_run = True, G|clip_equal_partB = True, Part B recheck ok = True (); layers vs plan True; legacy trend() cross-check 0/24 differ; required keys missing 0; E-format findings {"EleutherAI__pythia-1b|vit_large_patch16_224.augreg_in21k": {"max_rel_mu_f32": 1.5008236666106817e-06}, "EleutherAI__pythia-1.4b|vit_small_patch14_dinov2.lvd142m": {"max_rel_mu_f32": 1.050349394690316e-06}, "EleutherAI__pythia-6.9b|vit_huge_patch14_clip_224.laion2b_ft_in12k": {"max_rel_mu_f32": 4.19534008268307e-06}, "EleutherAI__pythia-70m|vit_giant_patch14_dinov2.lvd142m": {"max_rel_mu_f32": 3.7540896543124346e-06}, "EleutherAI__pythia-160m|vit_base_patch16_224.augreg_in21k": {"max_rel_mu_f32": 1.3801246090334088e-06}, "EleutherAI__pythia-160m|vit_large_patch14_dinov2.lvd142m": {"max_rel_mu; identity issues .
- R-I: G|n_pairs_read = 238, G|l2_run = True, G|clip_equal_partB = True, Part B recheck ok = True (); layers vs plan True; legacy trend() cross-check 0/24 differ; required keys missing 0; E-format findings {}; identity issues .
- Operational choices of R not fixed by the plan or the contract are listed in analysis/s12_layermax_pythia/review/CHANGES.md ([R-op]).
- Every mismatch is listed by name in cmp/*_EI.json and cmp/*_FK.json and with values in cmp/R*_detail_*.json.

