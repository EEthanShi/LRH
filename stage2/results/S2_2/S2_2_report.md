# S2.2 gauge net — FINAL analysis (PREREG_stage2 v1.0)

- pairs per cell: R/2 with R = 192 of 192 runs; shared-corpus R = 48 of 48 per corpus
- prereg sha256 6ce39abe7094034d…; eps sha256 b46e3f1dd62e3579…; Π_u = Π_y check: max 0.00 over 1344 runs

## Predictions

- **P1_noWD_u_unreliable**: dN_k10_25000: unreliable; dN_k10_250000: reliable; Gs_25000: unreliable; Gs_250000: unreliable; pass: False
- **P2_noWD_y_Pi_reliable_250k**: y_dN_k10_25000: unreliable; y_dN_k10_250000: reliable; y_Gs_25000: unreliable; y_Gs_250000: reliable; Pi_25000: unreliable; Pi_250000: reliable; pass: True
- **P3_WD_smallinit_u_reliable_250k**: WD_dN_k10_25000: unreliable; WD_dN_k10_250000: reliable; WD_Gs_25000: unreliable; WD_Gs_250000: reliable; smallinit_dN_k10_25000: unreliable; smallinit_dN_k10_250000: reliable; smallinit_Gs_25000: unreliable; smallinit_Gs_250000: inconclusive; pass_WD: True; pass_smallinit: False
- **P4_h_localization**: pass: True
  - 25000: mean_sN10_h: {'noWD': 0.15744883219401043, 'WD': 0.15896504720052085, 'smallinit': 0.1526372273763021}, max_abs_diff: 0.006
  - 250000: mean_sN10_h: {'noWD': 0.2962849934895833, 'WD': 0.2999333699544271, 'smallinit': 0.2888524373372396}, max_abs_diff: 0.011
- **P5_smallinit_matches_WD**: dN_k10_25000: 0.000; dN_k10_250000: 0.000; Gs_25000: 0.000; Gs_250000: 0.146; pass: False
- **falsifier_noWD_u_reliable**: True
- **shared_corpus**: 
  - y_dN_k10: p_shared: 0.000, p_indep: 0.000, dec_shared: unreliable, dec_indep: unreliable, mean_delta_diff_shared_minus_indep: {'diff': -0.027695210774739465, 'ci95': [-0.028569552103678314, -0.026869341532389228], 'p_one_sided_diff_ge_0': 0.0004997501249375312}
  - y_Gs: p_shared: 0.000, p_indep: 0.000, dec_shared: unreliable, dec_indep: unreliable, mean_delta_diff_shared_minus_indep: {'diff': -0.033322634931086614, 'ci95': [-0.034374425885827856, -0.032274860857481456], 'p_one_sided_diff_ge_0': 0.0004997501249375312}
  - y_dPi: p_shared: 0.000, p_indep: 0.000, dec_shared: unreliable, dec_indep: unreliable, mean_delta_diff_shared_minus_indep: {'diff': -0.049502516112961484, 'ci95': [-0.05053970910696366, -0.048486851714324306], 'p_one_sided_diff_ge_0': 0.0004997501249375312}
  - y_dCKA: p_shared: 0.000, p_indep: 0.000, dec_shared: unreliable, dec_indep: unreliable, mean_delta_diff_shared_minus_indep: {'diff': -0.042063498357530996, 'ci95': [-0.04309449056449096, -0.041162165454596056], 'p_one_sided_diff_ge_0': 0.0004997501249375312}
  - u_dN_k10: p_shared: 0.000, p_indep: 0.000, dec_shared: unreliable, dec_indep: unreliable, mean_delta_diff_shared_minus_indep: {'diff': -0.02793731689453116, 'ci95': [-0.028794873555501387, -0.027106602986653763], 'p_one_sided_diff_ge_0': 0.0004997501249375312}
  - u_Gs: p_shared: 0.000, p_indep: 0.000, dec_shared: unreliable, dec_indep: unreliable, mean_delta_diff_shared_minus_indep: {'diff': -0.03899487386478662, 'ci95': [-0.040423524235183354, -0.037622605247564486], 'p_one_sided_diff_ge_0': 0.0004997501249375312}
  - u_dPi: p_shared: 0.000, p_indep: 0.000, dec_shared: unreliable, dec_indep: unreliable, mean_delta_diff_shared_minus_indep: {'diff': -0.04950221334009107, 'ci95': [-0.05053896530210414, -0.04848644184023013], 'p_one_sided_diff_ge_0': 0.0004997501249375312}
  - u_dCKA: p_shared: 0.000, p_indep: 0.000, dec_shared: unreliable, dec_indep: unreliable, mean_delta_diff_shared_minus_indep: {'diff': -0.04609929883608241, 'ci95': [-0.04730254914419126, -0.04488213705339332], 'p_one_sided_diff_ge_0': 0.0004997501249375312}
  - h_dN_k10: p_shared: 0.000, p_indep: 0.000, dec_shared: unreliable, dec_indep: unreliable, mean_delta_diff_shared_minus_indep: {'diff': -0.049724324544271115, 'ci95': [-0.051165301005045426, -0.048264198303222754], 'p_one_sided_diff_ge_0': 0.0004997501249375312}
  - h_Gs: p_shared: 0.438, p_indep: 0.000, dec_shared: unreliable, dec_indep: unreliable, mean_delta_diff_shared_minus_indep: {'diff': -0.055692258883683565, 'ci95': [-0.059561560858280876, -0.051687609001852285], 'p_one_sided_diff_ge_0': 0.0004997501249375312}
  - h_dPi: p_shared: 0.000, p_indep: 0.000, dec_shared: unreliable, dec_indep: unreliable, mean_delta_diff_shared_minus_indep: {'diff': -0.020533672236051403, 'ci95': [-0.02106375640796427, -0.019976712170779252], 'p_one_sided_diff_ge_0': 0.0004997501249375312}
  - h_dCKA: p_shared: 0.552, p_indep: 0.000, dec_shared: unreliable, dec_indep: unreliable, mean_delta_diff_shared_minus_indep: {'diff': -0.024555532684124734, 'ci95': [-0.026149465174365363, -0.023032237332053514], 'p_one_sided_diff_ge_0': 0.0004997501249375312}

## Decision tables (ε function-anchored: u uses ε(y))

| set | rep | dN_k10 | Gs | dCKA | dO | dE | dMST | dAff | dSOG_t0p1 | dPi |
|---|---|---|---|---|---|---|---|---|---|---|
| noWD_25000 | y | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| noWD_25000 | u | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| noWD_25000 | h | unr 0/96 | unr 0/96 | unr 0/96 | unr 31/96 | unr 1/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| noWD_25000 | h_l2 | unr 0/96 | unr 0/96 | unr 0/96 | unr 2/96 | unr 4/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| WD_25000 | y | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| WD_25000 | u | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| WD_25000 | h | unr 0/96 | rel 96/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| WD_25000 | h_l2 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| smallinit_25000 | y | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| smallinit_25000 | u | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| smallinit_25000 | h | unr 0/96 | unr 0/96 | unr 0/96 | unr 43/96 | unr 37/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| smallinit_25000 | h_l2 | unr 0/96 | unr 0/96 | unr 0/96 | unr 1/96 | unr 1/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| noWD_250000 | y | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 |
| noWD_250000 | u | rel 96/96 | unr 65/96 | inc 73/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 95/96 | rel 96/96 | rel 96/96 |
| noWD_250000 | h | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 |
| noWD_250000 | h_l2 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 |
| WD_250000 | y | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 |
| WD_250000 | u | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 |
| WD_250000 | h | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 |
| WD_250000 | h_l2 | rel 96/96 | rel 96/96 | rel 96/96 | rel 95/96 | rel 94/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 |
| smallinit_250000 | y | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 |
| smallinit_250000 | u | rel 96/96 | inc 82/96 | inc 86/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 |
| smallinit_250000 | h | rel 96/96 | rel 93/96 | inc 91/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 95/96 | rel 96/96 | rel 96/96 |
| smallinit_250000 | h_l2 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 | rel 96/96 |
| noWD-shared_25000 | y | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| noWD-shared_25000 | u | unr 0/96 | unr 0/96 | unr 0/96 | unr 15/96 | unr 31/96 | unr 0/96 | unr 0/96 | unr 0/96 | unr 0/96 |
| noWD-shared_25000 | h | unr 0/96 | unr 42/96 | unr 53/96 | rel 96/96 | rel 96/96 | unr 0/96 | inc 70/96 | unr 0/96 | unr 0/96 |
| noWD-shared_25000 | h_l2 | unr 0/96 | rel 96/96 | inc 86/96 | inc 86/96 | inc 89/96 | unr 0/96 | inc 85/96 | unr 0/96 | unr 0/96 |

## Covariates

- noWD_25000: n 192.000, acc_bank 0.884, acc_S 0.877, gen_gap 0.295, ce_train 0.229, imb_rel_median 0.273, imb_rel_q10 0.266, imb_rel_q90 0.279, n_diverged 0.000, t_total_s 12.384
- noWD_250000: n 192.000, acc_bank 0.901, acc_S 0.894, gen_gap 0.026, ce_train 0.445, imb_rel_median 0.313, imb_rel_q10 0.306, imb_rel_q90 0.319, n_diverged 0.000, t_total_s 12.517
- WD_25000: n 192.000, acc_bank 0.884, acc_S 0.877, gen_gap 0.298, ce_train 0.237, imb_rel_median 0.030, imb_rel_q10 0.029, imb_rel_q90 0.031, n_diverged 0.000, t_total_s 11.883
- WD_250000: n 192.000, acc_bank 0.902, acc_S 0.895, gen_gap 0.026, ce_train 0.452, imb_rel_median 0.040, imb_rel_q10 0.038, imb_rel_q90 0.041, n_diverged 0.000, t_total_s 13.323
- smallinit_25000: n 192.000, acc_bank 0.883, acc_S 0.876, gen_gap 0.293, ce_train 0.235, imb_rel_median 0.240, imb_rel_q10 0.230, imb_rel_q90 0.246, n_diverged 0.000, t_total_s 11.818
- smallinit_250000: n 192.000, acc_bank 0.900, acc_S 0.893, gen_gap 0.027, ce_train 0.448, imb_rel_median 0.278, imb_rel_q10 0.269, imb_rel_q90 0.287, n_diverged 0.000, t_total_s 13.541
- noWD-shared_25000: n 192.000, acc_bank 0.884, acc_S 0.876, gen_gap 0.295, ce_train 0.230, imb_rel_median 0.273, imb_rel_q10 0.267, imb_rel_q90 0.279, n_diverged 0.000, t_total_s 13.488
