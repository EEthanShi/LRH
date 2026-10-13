# X3b report (role `x3b-run`): last-block vision features, CPU only. EXPLORATORY.

Readings `X3B_READINGS.md` sha256 `fef078b8058a0bbb19f25697014130f60b5f8d83a8e93d6b8b61900ecb2dd34e` (STATUS 2026-10-10T21:18:49Z); validation `VALIDATION_X3B.json` 7b309c6caa896ad8... pass = True; scores `obs_X3B.json` d2abd08227e9e14d....

## Binding readings

| question | set | statistic | value | reading |
|---|---|---|---|---|
| Q-a level | A1 | median R = L_last/L_mid at L(10,1024) | 2.821 | the level is mostly a depth effect |
| Q-a fall with m | A1 | median b(10) | -0.419 | still falls with m |
| Q-a stability | A1 | abs(e_k+e_m) <= 0.05 | 15/15 | stays close to stable at fixed k/m |
| Q-b direction | B | L_last > L_mid at L(100,10000) | 15/15 | the rise with depth is not specific to DINOv2-B |
| Q-b stability | B | abs(e_k+e_m) <= 0.05 | 15/15 | stays close to stable at fixed k/m |

Descriptive companions: Q-b size (12 non-DINOv2 pairs) median R at L(100,10000) 3.385 (a large rise, as for DINOv2-B); Q-b fall with m: median b(10) -0.297 (still falls with m).

## Per set (medians over 15 pairs)

| set | what | L_last(10,1024) | L_mid(10,1024) | R(10,1024) | R(100,8192) | R(100,10000) | n last>mid (10,1024 / 100,10000) | b(10) last / mid | stable last / mid | depth label | fall label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | Q-a binding: clip, ViT block 11 x LM tokens (LM rel 0.5) | 0.1443 | 0.0430 | 2.821 | 2.756 | 2.846 | 15 / 15 | -0.419 / -0.571 | 15/15 / 15/15 | the level is mostly a depth effect | still falls with m |
| A2_r025 | Q-a, LM rel 0.25 (clip), descriptive | 0.1710 | 0.0454 | 3.160 | 3.107 | 3.194 | 15 / 15 | -0.413 / -0.559 | 15/15 / 15/15 | the level is mostly a depth effect | still falls with m |
| A2_r075 | Q-a, LM rel 0.75 (clip), descriptive | 0.1943 | 0.0517 | 3.243 | 3.103 | 3.192 | 15 / 15 | -0.428 / -0.576 | 15/15 / 15/15 | the level is mostly a depth effect | still falls with m |
| A2_r10 | Q-a, LM rel 1.0 (clip), descriptive | 0.1960 | 0.0505 | 3.244 | 3.175 | 3.263 | 15 / 15 | -0.436 / -0.580 | 15/15 / 15/15 | the level is mostly a depth effect | still falls with m |
| A3 | Q-a sensitivity: l2 | 0.1236 | 0.0405 | 2.592 | 2.591 | 2.691 | 15 / 15 | -0.401 / -0.566 | 15/15 / 15/15 | the level is mostly a depth effect | still falls with m |
| A4 | Q-a sensitivity: ctr | 0.1427 | 0.0462 | 2.616 | 2.581 | 2.652 | 15 / 15 | -0.414 / -0.564 | 15/15 / 15/15 | the level is mostly a depth effect | still falls with m |
| B | Q-b binding: ctr, ViT block 11 x sentence encoders (N5) | 0.4195 | 0.0740 | 4.289 | 4.210 | 4.335 | 15 / 15 | -0.297 / -0.433 | 15/15 / 15/15 | the level is mostly a depth effect | still falls with m |

## Per ViT, median R over the 3 text models

| set | cell | dinov2_b | clip_b | mae_b | augreg_b | siglip_b |
|---|---|---|---|---|---|---|
| A1 | L10_1024 | 3.36 | 4.40 | 1.97 | 2.82 | 1.78 |
| A1 | L100_10000 | 3.37 | 4.36 | 2.03 | 2.85 | 1.76 |
| A3 | L10_1024 | 3.06 | 4.17 | 1.93 | 2.59 | 1.75 |
| A3 | L100_10000 | 3.11 | 4.09 | 1.90 | 2.69 | 1.76 |
| A4 | L10_1024 | 3.07 | 4.06 | 1.95 | 2.62 | 1.68 |
| A4 | L100_10000 | 3.09 | 4.09 | 1.99 | 2.65 | 1.72 |
| B | L10_1024 | 5.77 | 7.26 | 2.44 | 4.29 | 2.37 |
| B | L100_10000 | 5.56 | 7.41 | 2.44 | 4.34 | 2.34 |
| A2_r10 | L10_1024 | 3.95 | 5.31 | 2.10 | 3.24 | 1.91 |
| A2_r10 | L100_10000 | 3.96 | 5.18 | 2.14 | 3.26 | 1.94 |

## Set A1, per pair

| vit | text | L(10,1024) last / mid | R | L(100,10000) last / mid | R | b(1) | b(10) | b(100) | b(10) mid | e_k | e_m | sum | sum mid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| augreg_b | bloomz_1b7 | 0.1915 / 0.0602 | 3.18 | 0.1956 / 0.0616 | 3.18 | -0.561 | -0.415 | -0.220 | -0.571 | 0.325 | -0.318 | 0.0072 | -0.0059 |
| augreg_b | openllama_3b | 0.1454 / 0.0515 | 2.82 | 0.1500 / 0.0527 | 2.85 | -0.491 | -0.420 | -0.276 | -0.582 | 0.360 | -0.349 | 0.0116 | -0.0119 |
| augreg_b | pythia410m_s1 | 0.1334 / 0.0478 | 2.79 | 0.1357 / 0.0495 | 2.74 | -0.602 | -0.419 | -0.304 | -0.573 | 0.365 | -0.362 | 0.0036 | -0.0012 |
| clip_b | bloomz_1b7 | 0.2011 / 0.0391 | 5.14 | 0.2022 / 0.0412 | 4.90 | -0.444 | -0.405 | -0.252 | -0.554 | 0.333 | -0.329 | 0.0040 | 0.0116 |
| clip_b | openllama_3b | 0.1553 / 0.0353 | 4.40 | 0.1594 / 0.0365 | 4.36 | -0.444 | -0.411 | -0.297 | -0.546 | 0.366 | -0.354 | 0.0118 | 0.0131 |
| clip_b | pythia410m_s1 | 0.1402 / 0.0327 | 4.28 | 0.1424 / 0.0348 | 4.09 | -0.401 | -0.405 | -0.326 | -0.550 | 0.371 | -0.366 | 0.0052 | 0.0124 |
| dinov2_b | bloomz_1b7 | 0.1877 / 0.0503 | 3.73 | 0.1910 / 0.0503 | 3.80 | -0.497 | -0.374 | -0.209 | -0.569 | 0.297 | -0.291 | 0.0060 | 0.0046 |
| dinov2_b | openllama_3b | 0.1443 / 0.0430 | 3.36 | 0.1477 / 0.0438 | 3.37 | -0.461 | -0.380 | -0.265 | -0.606 | 0.337 | -0.323 | 0.0145 | 0.0050 |
| dinov2_b | pythia410m_s1 | 0.1307 / 0.0414 | 3.15 | 0.1319 / 0.0417 | 3.16 | -0.424 | -0.368 | -0.288 | -0.592 | 0.338 | -0.328 | 0.0097 | -0.0023 |
| mae_b | bloomz_1b7 | 0.0814 / 0.0394 | 2.07 | 0.0860 / 0.0412 | 2.09 | -0.538 | -0.496 | -0.454 | -0.630 | 0.492 | -0.476 | 0.0161 | 0.0044 |
| mae_b | openllama_3b | 0.0693 / 0.0357 | 1.94 | 0.0723 / 0.0356 | 2.03 | -0.560 | -0.512 | -0.483 | -0.663 | 0.502 | -0.498 | 0.0038 | -0.0031 |
| mae_b | pythia410m_s1 | 0.0647 / 0.0328 | 1.97 | 0.0674 / 0.0341 | 1.98 | -0.509 | -0.500 | -0.506 | -0.640 | 0.515 | -0.504 | 0.0106 | 0.0123 |
| siglip_b | bloomz_1b7 | 0.1870 / 0.0981 | 1.91 | 0.1901 / 0.1007 | 1.89 | -0.464 | -0.424 | -0.274 | -0.513 | 0.356 | -0.349 | 0.0063 | 0.0019 |
| siglip_b | openllama_3b | 0.1449 / 0.0825 | 1.76 | 0.1476 / 0.0841 | 1.76 | -0.457 | -0.421 | -0.320 | -0.512 | 0.381 | -0.371 | 0.0099 | -0.0052 |
| siglip_b | pythia410m_s1 | 0.1320 / 0.0743 | 1.78 | 0.1332 / 0.0776 | 1.72 | -0.434 | -0.430 | -0.345 | -0.509 | 0.394 | -0.388 | 0.0063 | 0.0048 |

## Set B, per pair

| vit | text | L(10,1024) last / mid | R | L(100,10000) last / mid | R | b(1) | b(10) | b(100) | b(10) mid | e_k | e_m | sum | sum mid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| augreg_b | contriever | 0.4186 / 0.0976 | 4.29 | 0.4248 / 0.0978 | 4.34 | -0.372 | -0.317 | -0.063 | -0.433 | 0.197 | -0.190 | 0.0066 | -0.0010 |
| augreg_b | mpnet | 0.4208 / 0.0984 | 4.28 | 0.4239 / 0.0978 | 4.34 | -0.364 | -0.307 | -0.065 | -0.443 | 0.193 | -0.186 | 0.0075 | -0.0048 |
| augreg_b | qwen3emb06 | 0.4195 / 0.0962 | 4.36 | 0.4243 / 0.0980 | 4.33 | -0.346 | -0.307 | -0.068 | -0.424 | 0.193 | -0.188 | 0.0055 | 0.0003 |
| clip_b | contriever | 0.4420 / 0.0609 | 7.26 | 0.4456 / 0.0602 | 7.40 | -0.286 | -0.291 | -0.095 | -0.452 | 0.199 | -0.193 | 0.0061 | 0.0198 |
| clip_b | mpnet | 0.4404 / 0.0611 | 7.21 | 0.4470 / 0.0603 | 7.41 | -0.292 | -0.279 | -0.097 | -0.454 | 0.194 | -0.188 | 0.0061 | 0.0160 |
| clip_b | qwen3emb06 | 0.4437 / 0.0602 | 7.37 | 0.4503 / 0.0606 | 7.43 | -0.287 | -0.279 | -0.099 | -0.441 | 0.196 | -0.189 | 0.0069 | 0.0216 |
| dinov2_b | contriever | 0.4324 / 0.0747 | 5.79 | 0.4323 / 0.0763 | 5.66 | -0.378 | -0.292 | 0.000 | -0.428 | 0.148 | -0.146 | 0.0018 | 0.0096 |
| dinov2_b | mpnet | 0.4266 / 0.0740 | 5.77 | 0.4289 / 0.0772 | 5.56 | -0.372 | -0.283 | -0.001 | -0.421 | 0.144 | -0.142 | 0.0017 | 0.0155 |
| dinov2_b | qwen3emb06 | 0.4254 / 0.0740 | 5.75 | 0.4285 / 0.0772 | 5.55 | -0.387 | -0.285 | -0.002 | -0.413 | 0.145 | -0.144 | 0.0015 | 0.0151 |
| mae_b | contriever | 0.1423 / 0.0595 | 2.39 | 0.1446 / 0.0608 | 2.38 | -0.327 | -0.371 | -0.333 | -0.483 | 0.371 | -0.352 | 0.0191 | 0.0209 |
| mae_b | mpnet | 0.1459 / 0.0586 | 2.49 | 0.1474 / 0.0605 | 2.44 | -0.333 | -0.365 | -0.332 | -0.468 | 0.366 | -0.348 | 0.0176 | 0.0246 |
| mae_b | qwen3emb06 | 0.1446 / 0.0594 | 2.44 | 0.1469 / 0.0602 | 2.44 | -0.293 | -0.361 | -0.333 | -0.460 | 0.363 | -0.347 | 0.0163 | 0.0231 |
| siglip_b | contriever | 0.3921 / 0.1639 | 2.39 | 0.3951 / 0.1674 | 2.36 | -0.331 | -0.308 | -0.113 | -0.374 | 0.218 | -0.210 | 0.0072 | 0.0104 |
| siglip_b | mpnet | 0.3943 / 0.1687 | 2.34 | 0.3956 / 0.1697 | 2.33 | -0.315 | -0.296 | -0.113 | -0.363 | 0.210 | -0.204 | 0.0055 | 0.0079 |
| siglip_b | qwen3emb06 | 0.3931 / 0.1655 | 2.37 | 0.3980 / 0.1700 | 2.34 | -0.304 | -0.297 | -0.114 | -0.359 | 0.213 | -0.205 | 0.0074 | 0.0119 |

## Cross-reading (descriptive): same ViT features under ctr, LM tokens (A4) vs sentence encoders (B)

| vit | A4 R(10,1024) | B R(10,1024) | A4 R(100,10000) | B R(100,10000) |
|---|---|---|---|---|
| dinov2_b | 3.07 | 5.77 | 3.09 | 5.56 |
| clip_b | 4.06 | 7.26 | 4.09 | 7.41 |
| mae_b | 1.95 | 2.44 | 1.99 | 2.44 |
| augreg_b | 2.62 | 4.29 | 2.65 | 4.34 |
| siglip_b | 1.68 | 2.37 | 1.72 | 2.34 |

## Checks

- Validation (STOP rule): {"cache_clip_dinov2_b_r05_vs_S11": true, "cache_clip_pythia410m_s1_r05_vs_S11": true, "obs_dinov2_b__pythia410m_s1_rebuilt_vs_obs_clip": true, "cache_ctr_dinov2_b_r10_vs_X3_x3_cls_r10": true, "obs_x3_cls_r10__mpnet_rebuilt_vs_obs_X3": true, "obs_ref_clip_15VL": true, "obs_ref_l2_15VL": true, "obs_ref_ctrold_15VL": true, "obs_ref_CA_15": true}
- Reference consistency with frozen tables: {"Bref": {"n_rows": 45, "max_abs_diff": {"L_m1024": 0.0, "L_m8192": 0.0, "L_m10000": 0.0}}, "A4ref": {"n_rows": 45, "max_abs_diff": {"L_m1024": 0.0, "L_m8192": 0.0, "L_m10000": 0.0}}, "A1ref_vs_S1_1_cells_clip": {"n_rows": 315, "max_abs_diff_obs": 0.0}}
- Compute (scores): CPU 0.275 core-h, wall 239 s, 6 workers x 1 BLAS thread, affinity [0, 1, 2, 3, 4, 5]; GPU 0 h.

All values: column `obs` (S1.1 anchors), E0/X3 interpolation for m = 10,000. EXPLORATORY; no S1.1, E0 or X3 label changes.
