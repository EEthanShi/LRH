# S2.1 objective link — FINAL analysis (PREREG_stage2 v1.0)

- runs per (arm, N) used: R = 64 of 64 (balanced prefix); prereg sha256 6ce39abe7094034d…; eps file sha256 0b28027d7366fb75…
- **Verdict: falsified**

## Endpoints (z, cosine kernel)

(i) ordinal near ceiling at 500k and improving with N:
| rung | p̂ cross 50k | p̂ cross 500k | ceiling 500k | gap | mean Δ diff (500k−50k) [95% CI] | Holm reject |
|---|---|---|---|---|---|---|
| dN_k10 | 0.000 | 0.000 | 1.000 | 1.000 | -0.0125 [-0.0145, -0.0105] | True |
| dO | 0.000 | 0.000 | 1.000 | 1.000 | -0.0005 [-0.0011, 0.0001] | False |
| dE | 0.000 | 0.000 | 1.000 | 1.000 | -0.0002 [-0.0008, 0.0004] | False |
| dMST | 0.000 | 0.000 | 1.000 | 1.000 | -0.0119 [-0.0145, -0.0094] | True |

(i) pass: False

(ii) geometric obstruction (cross A–B unreliable at both N; K3 guard within p̂ ≥ 0.8 at 500k):
- Gs: cross decision {50000: 'unreliable', 500000: 'unreliable'}; within p̂ {A: 1.000, B: 1.000} → **pass**
- dAff: cross decision {50000: 'unreliable', 500000: 'unreliable'}; within p̂ {A: 1.000, B: 1.000} → **pass**
- dSOG_t0p1: cross decision {50000: 'unreliable', 500000: 'unreliable'}; within p̂ {A: 1.000, B: 1.000} → **pass**

(iii) Iso: mean Δ_iso cross 0.1767 vs within 0.0553 (ratio 3.198; pass if ≤ 1.5): False

(iv) warp share ω (median):
- N=50000: omega_cross_AB 0.142, omega_within_A 0.020, omega_within_B 0.004, omega_within_Atau 0.008, omega_cross_A_Atau 0.107, omega_cross_Atau_B 0.139
- N=500000: omega_cross_AB 0.147, omega_within_A 0.012, omega_within_B 0.002, omega_within_Atau 0.004, omega_cross_A_Atau 0.109, omega_cross_Atau_B 0.144
- pass (500k: cross ≥ 0.5, within A and B ≤ 0.2): False

(v) A vs A_τ:
- Gs: p_cross 0.000, ceiling 1.000, pass False
- dCKA: p_cross 0.000, ceiling 1.000, pass False
- dSOG_t0p1: p_cross 0.000, ceiling 1.000, pass False
- rawG: cross_decision unreliable, within_p {'A': 1.0, 'Atau': 1.0}, pass True
- pass: False

CKA (1 − CKA): cross A–B minus within (A,B):
- N=50000: 0.5370 [0.5253, 0.5481]
- N=500000: 0.5575 [0.5496, 0.5650]

## Decision tables (z; ε = max pooled pilot within-arm Δ)

| pair set | dN_k10 | dO | dE | dMST | Gs | dAff | dSOG_t0p1 | dCKA | rawG | dIso | dPi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| within_A_50000|z | rel 32/32 | unr 0/32 | unr 0/32 | rel 32/32 | rel 32/32 | unr 0/32 | rel 32/32 | unr 0/32 | rel 32/32 | rel 32/32 | rel 32/32 |
| within_B_50000|z | unr 1/32 | rel 32/32 | rel 32/32 | unr 7/32 | unr 19/32 | rel 32/32 | unr 2/32 | rel 32/32 | unr 19/32 | unr 19/32 | unr 0/32 |
| within_Atau_50000|z | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 |
| cross_A_B_50000|z | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 |
| cross_A_Atau_50000|z | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | rel 64/64 |
| cross_Atau_B_50000|z | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 |
| bern_self_50000|z | rel 64/64 | rel 64/64 | rel 64/64 | rel 64/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | – | rel 64/64 | – |
| bern_peer_50000|z | rel 32/32 | unr 0/32 | unr 0/32 | rel 32/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | – | rel 32/32 | – |
| init_within_A_50000|z_init | unr 0/32 | unr 0/32 | unr 13/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 |
| init_within_B_50000|z_init | unr 0/32 | unr 0/32 | unr 11/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 |
| init_within_Atau_50000|z_init | unr 0/32 | unr 0/32 | unr 14/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 |
| within_A_500000|z | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 |
| within_B_500000|z | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 |
| within_Atau_500000|z | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 |
| cross_A_B_500000|z | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 |
| cross_A_Atau_500000|z | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | rel 64/64 |
| cross_Atau_B_500000|z | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 |
| bern_self_500000|z | rel 64/64 | rel 64/64 | rel 64/64 | rel 64/64 | unr 0/64 | unr 0/64 | unr 0/64 | unr 0/64 | – | rel 64/64 | – |
| bern_peer_500000|z | rel 32/32 | rel 32/32 | rel 32/32 | rel 32/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | – | rel 32/32 | – |
| init_within_A_500000|z_init | unr 0/32 | unr 0/32 | unr 15/32 | unr 0/32 | unr 0/32 | unr 1/32 | unr 0/32 | unr 1/32 | unr 0/32 | unr 0/32 | unr 0/32 |
| init_within_B_500000|z_init | unr 0/32 | unr 0/32 | unr 12/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 |
| init_within_Atau_500000|z_init | unr 0/32 | unr 0/32 | unr 11/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 |
| fixedepoch_within_A|z | unr 7/16 | unr 0/16 | unr 0/16 | inc 16/16 | unr 0/16 | unr 0/16 | inc 16/16 | unr 0/16 | unr 0/16 | unr 0/16 | inc 16/16 |
| fixedepoch_within_B|z | unr 0/16 | inc 16/16 | inc 16/16 | unr 0/16 | unr 0/16 | inc 16/16 | unr 0/16 | inc 16/16 | unr 0/16 | unr 0/16 | unr 0/16 |
| fixedepoch_cross_A_B|z | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 | unr 0/32 |

(rel = reliable, unr = unreliable, inc = inconclusive)

## Secondary

Sharp link β̂ (fit K_B ≈ exp(β(K_A − 1))): A_B_50000: median 2.042; Atau_B_50000: median 2.031; A_B_500000: median 2.039; Atau_B_500000: median 2.026; ratios ratio_50000 1.005, ratio_500000 1.006; prediction ok pred_ok_50000 False, pred_ok_500000 False

Lemma 6 (near vs far log-distance error, improvement 50k→500k):
- A: R_near 1.194, R_far 1.199, log ratio -0.004 [-0.010, 0.001]
- B: R_near 1.158, R_far 1.194, log ratio -0.031 [-0.075, 0.013]
- Atau: R_near 1.182, R_far 1.216, log ratio -0.028 [-0.039, -0.016]
- A − A_τ: 0.023 [0.010, 0.035]; prediction ok: False

All-pairs U-statistics (z-l2; mean [95% run-bootstrap CI]):
- hubness_skew_50000: A 0.269, B 0.096, Atau 0.204
- A_A_50000: sN10 0.662 [0.662, 0.663]; Gs 0.060 [0.060, 0.060]; CKA 0.875 [0.875, 0.876]; RSA 0.871 [0.870, 0.871]
- B_B_50000: sN10 0.585 [0.582, 0.588]; Gs 0.075 [0.074, 0.076]; CKA 0.969 [0.968, 0.970]; RSA 0.969 [0.968, 0.970]
- Atau_Atau_50000: sN10 0.679 [0.677, 0.680]; Gs 0.061 [0.061, 0.062]; CKA 0.948 [0.947, 0.948]; RSA 0.946 [0.946, 0.947]
- A_B_50000: sN10 0.300 [0.299, 0.302]; Gs 0.277 [0.276, 0.279]; CKA 0.386 [0.384, 0.387]; RSA 0.371 [0.369, 0.373]
- A_Atau_50000: sN10 0.487 [0.486, 0.488]; Gs 0.145 [0.145, 0.145]; CKA 0.642 [0.641, 0.642]; RSA 0.635 [0.634, 0.636]
- Atau_B_50000: sN10 0.405 [0.403, 0.407]; Gs 0.210 [0.208, 0.211]; CKA 0.734 [0.733, 0.736]; RSA 0.718 [0.716, 0.719]
- hubness_skew_500000: A 0.287, B 0.106, Atau 0.211
- A_A_500000: sN10 0.715 [0.714, 0.715]; Gs 0.050 [0.050, 0.050]; CKA 0.916 [0.915, 0.916]; RSA 0.912 [0.912, 0.913]
- B_B_500000: sN10 0.640 [0.638, 0.643]; Gs 0.062 [0.061, 0.063]; CKA 0.979 [0.979, 0.980]; RSA 0.979 [0.978, 0.979]
- Atau_Atau_500000: sN10 0.724 [0.723, 0.725]; Gs 0.051 [0.051, 0.051]; CKA 0.964 [0.964, 0.964]; RSA 0.963 [0.963, 0.964]
- A_B_500000: sN10 0.313 [0.312, 0.314]; Gs 0.278 [0.277, 0.278]; CKA 0.390 [0.389, 0.391]; RSA 0.374 [0.373, 0.376]
- A_Atau_500000: sN10 0.505 [0.504, 0.505]; Gs 0.144 [0.144, 0.145]; CKA 0.651 [0.650, 0.651]; RSA 0.644 [0.643, 0.645]
- Atau_B_500000: sN10 0.425 [0.423, 0.427]; Gs 0.208 [0.208, 0.209]; CKA 0.741 [0.740, 0.743]; RSA 0.723 [0.722, 0.725]

Covariates (mean ± sd):
- A_50000: knn_acc_h 0.796±0.004, knn_acc_z 0.796±0.004, final_train_loss 2.820±0.004, evalloss_train 2.816±0.007, evalloss_hold 2.867±0.003, gen_gap 0.051±0.007, t_total_s 279.916±3.650, z_norm_med 3.305±0.041, diverged 0
- A_500000: knn_acc_h 0.798±0.004, knn_acc_z 0.799±0.004, final_train_loss 2.845±0.004, evalloss_train 2.841±0.008, evalloss_hold 2.851±0.003, gen_gap 0.010±0.008, t_total_s 280.833±3.899, z_norm_med 3.640±0.043, diverged 0
- B_50000: knn_acc_h 0.794±0.004, knn_acc_z 0.771±0.006, final_train_loss -1.823±0.001, evalloss_train -1.826±0.003, evalloss_hold -1.817±0.001, gen_gap 0.009±0.002, t_total_s 278.226±3.063, z_norm_med 6.311±0.112, diverged 0
- B_500000: knn_acc_h 0.794±0.004, knn_acc_z 0.772±0.004, final_train_loss -1.818±0.001, evalloss_train -1.821±0.003, evalloss_hold -1.820±0.001, gen_gap 0.002±0.002, t_total_s 281.222±3.635, z_norm_med 6.417±0.106, diverged 0
- Atau_50000: knn_acc_h 0.809±0.004, knn_acc_z 0.807±0.004, final_train_loss 5.175±0.001, evalloss_train 5.175±0.003, evalloss_hold 5.190±0.001, gen_gap 0.015±0.004, t_total_s 277.838±3.497, z_norm_med 4.197±0.052, diverged 0
- Atau_500000: knn_acc_h 0.808±0.004, knn_acc_z 0.807±0.004, final_train_loss 5.183±0.001, evalloss_train 5.183±0.003, evalloss_hold 5.186±0.001, gen_gap 0.003±0.003, t_total_s 282.298±4.104, z_norm_med 4.342±0.049, diverged 0

Permutation nulls (first 4 pairs, K=200): see s21_summary.json `perm_nulls`.
