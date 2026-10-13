# X3 protocol bridge: DINOv2-B output features × sentence encoders (EXPLORATORY)

Role `x3-run`, written by `analysis/e0_sentence/x3/x3_report.py` from the hashed tables. Readings fixed beforehand: `X3_READINGS.md` sha256 `0151bf9ac26dd718652d69c72912445fb9fd15d770ae829db8a052e085d8663e`. Everything here is exploratory (plan §7.6). No E0 label changes.

## 1. What ran

- 12 X3 pairs: 3 frozen DINOv2-B features plus 1 added feature, each × MPNet, Contriever and Qwen3-Embedding-0.6B.
- Text side: N5 (five-caption average). Preprocessing: `ctr`. Design: E0's 5 nested draws, m = 256 to 16,384, k ∈ {1, 10, 100}.
- The E0 reference pairs (rel-0.5 CLS) are listed next to the X3 pairs.
- The task text said 30 pairs (10 vision variants). The frozen X3 is 9 pairs, made of 3 features × 3 encoders. No variant was invented.

| feature | what it is |
|---|---|
| `dinov2_b` | E0 reference: CLS, block 5 (rel 0.5), no final norm, 518 px |
| `x3_cls_r10` | frozen: CLS, block 11 (rel 1.0), no final norm, 518 px |
| `x3_tok_r10` | frozen: patch-token mean, block 11 (rel 1.0), no final norm, 518 px |
| `x3_tok_r05` | frozen: patch-token mean, block 5 (rel 0.5), no final norm, 518 px |
| `x3add_schnaus224` | ADDED: mean of all 257 output tokens after the final norm, 224 px |

## 2. Checks

- **Extraction at the S1.1 settings:** reproduces the S1.1 stored arrays bitwise on all 16,384 rows (CLS and patch mean, blocks 5 and 11). STOP rule passed: True.
- **Determinism:** 64 images run twice give bitwise-equal outputs in both passes.
- **Side note on position embeddings:** timm's antialiased resampling, instead of the HF-style one used here, changes the 224 px feature (64 images: min cosine 0.9844, mean 0.99861). This is one of the open implementation details.
- **Wrapper validation (STOP rule):** the rebuilt reference cache equals E0's bitwise: True. The obs rows from the rebuilt cache equal E0's `obs_CA` rows: True; from E0's cache: True. Overall pass: True.

## 3. The gap to Schnaus et al. (binding reading: DINOv2 × MPNet, L(100, 10000), column `obs`)

- share = ln(L / 0.07715) / ln(0.38 / 0.07715). Labels: most ≥ 0.5, part 0.2–0.5, little < 0.2, widens < 0.
- Schnaus' 0.38 is a figure reading (Fig. 23, p. 48), so the shares against 0.37 and 0.39 are shown as a band.

| feature | L(100, 10000) | ratio to 0.38 | share (0.38) | band (0.39 to 0.37) | share, all-H2 queries | reading |
|---|---|---|---|---|---|---|
| `x3_cls_r10` | 0.4289 | 1.129 | 1.076 | 1.059 to 1.094 | 1.069 | closes the whole gap (reaches or passes Schnaus' reading) |
| `x3_tok_r10` | 0.4064 | 1.069 | 1.042 | 1.025 to 1.060 | 1.038 | closes the whole gap (reaches or passes Schnaus' reading) |
| `x3_tok_r05` | 0.1315 | 0.346 | 0.335 | 0.329 to 0.340 | 0.337 | closes part of the gap |
| `x3add_schnaus224` | 0.4098 | 1.078 | 1.047 | 1.031 to 1.065 | 1.042 | closes the whole gap (reaches or passes Schnaus' reading) |
| E0 reference | 0.0772 | 0.203 | 0 | | | |

**The other two encoders (descriptive).** Schnaus et al. used no Contriever, and their Qwen embedder is the 8B model.

| pair | L(100, 10000) | share (0.38) | reading (descriptive) |
|---|---|---|---|
| `x3_cls_r10__contriever` | 0.4323 | 1.080 | closes the whole gap (reaches or passes Schnaus' reading) |
| `x3_cls_r10__qwen3emb06` | 0.4285 | 1.075 | closes the whole gap (reaches or passes Schnaus' reading) |
| `x3_tok_r10__contriever` | 0.3981 | 1.029 | closes the whole gap (reaches or passes Schnaus' reading) |
| `x3_tok_r10__qwen3emb06` | 0.4062 | 1.042 | closes the whole gap (reaches or passes Schnaus' reading) |
| `x3_tok_r05__contriever` | 0.1297 | 0.330 | closes part of the gap |
| `x3_tok_r05__qwen3emb06` | 0.1310 | 0.332 | closes part of the gap |
| `x3add_schnaus224__contriever` | 0.4024 | 1.036 | closes the whole gap (reaches or passes Schnaus' reading) |
| `x3add_schnaus224__qwen3emb06` | 0.4100 | 1.048 | closes the whole gap (reaches or passes Schnaus' reading) |

## 4. Levels, all pairs (column `obs`; the all-H2 value is shown where it applies)

| pair | L(10, 1024) | L(100, 8192) | L(10, 10000) | L(100, 10000) | L(100, 10000), all-H2 | s(100)/s(10) at m = 8192 |
|---|---|---|---|---|---|---|
| `dinov2_b__mpnet` | 0.0740 | 0.0846 | 0.0285 | 0.0772 | 0.0757 | 2.741 |
| `dinov2_b__contriever` | 0.0747 | 0.0839 | 0.0284 | 0.0763 | 0.0750 | 2.713 |
| `dinov2_b__qwen3emb06` | 0.0740 | 0.0847 | 0.0296 | 0.0772 | 0.0757 | 2.658 |
| `x3_cls_r10__mpnet` | 0.4266 | 0.4572 | 0.1830 | 0.4289 | 0.4249 | 2.321 |
| `x3_cls_r10__contriever` | 0.4324 | 0.4611 | 0.1790 | 0.4323 | 0.4290 | 2.382 |
| `x3_cls_r10__qwen3emb06` | 0.4254 | 0.4574 | 0.1818 | 0.4285 | 0.4253 | 2.341 |
| `x3_tok_r10__mpnet` | 0.4028 | 0.4339 | 0.1832 | 0.4064 | 0.4042 | 2.209 |
| `x3_tok_r10__contriever` | 0.3922 | 0.4259 | 0.1732 | 0.3981 | 0.3954 | 2.286 |
| `x3_tok_r10__qwen3emb06` | 0.4020 | 0.4341 | 0.1818 | 0.4062 | 0.4033 | 2.242 |
| `x3_tok_r05__mpnet` | 0.1325 | 0.1436 | 0.0492 | 0.1315 | 0.1305 | 2.737 |
| `x3_tok_r05__contriever` | 0.1287 | 0.1418 | 0.0484 | 0.1297 | 0.1284 | 2.700 |
| `x3_tok_r05__qwen3emb06` | 0.1315 | 0.1432 | 0.0500 | 0.1310 | 0.1297 | 2.658 |
| `x3add_schnaus224__mpnet` | 0.4043 | 0.4386 | 0.1810 | 0.4098 | 0.4068 | 2.264 |
| `x3add_schnaus224__contriever` | 0.3943 | 0.4314 | 0.1720 | 0.4024 | 0.3991 | 2.334 |
| `x3add_schnaus224__qwen3emb06` | 0.4040 | 0.4390 | 0.1801 | 0.4100 | 0.4063 | 2.292 |

## 5. Contrasts: ln L(100, 10000) minus the E0 reference (descriptive)

| contrast | mpnet | contriever | qwen3emb06 |
|---|---|---|---|
| depth only: x3_cls_r10 vs ref (ln L(100,10000)) | +1.715 | +1.734 | +1.714 |
| token set only: x3_tok_r05 vs ref | +0.534 | +0.530 | +0.529 |
| depth and token set: x3_tok_r10 vs ref | +1.661 | +1.652 | +1.661 |
| added (norm + all tokens + 224 px): x3add_schnaus224 vs ref | +1.670 | +1.662 | +1.670 |
| added vs x3_tok_r10 (norm + CLS in the mean + 224 px) | +0.008 | +0.011 | +0.009 |

## 6. Scale reading of P1 type (descriptive; e_k + e_m on k ∈ {10, 100})

| pair | e_k | e_m | sum | abs(sum) ≤ 0.05 | sum on {1, 10, 100} | sum on {1, 10} |
|---|---|---|---|---|---|---|
| `dinov2_b__mpnet` | 0.462 | -0.447 | +0.0155 | yes | +0.0504 | +0.0581 |
| `dinov2_b__contriever` | 0.459 | -0.450 | +0.0096 | yes | +0.0648 | +0.0832 |
| `dinov2_b__qwen3emb06` | 0.457 | -0.442 | +0.0151 | yes | +0.0737 | +0.1012 |
| `x3_cls_r10__mpnet` | 0.144 | -0.142 | +0.0017 | yes | +0.0386 | +0.0429 |
| `x3_cls_r10__contriever` | 0.148 | -0.146 | +0.0018 | yes | +0.0382 | +0.0400 |
| `x3_cls_r10__qwen3emb06` | 0.145 | -0.144 | +0.0015 | yes | +0.0327 | +0.0334 |
| `x3_tok_r10__mpnet` | 0.218 | -0.213 | +0.0049 | yes | +0.0331 | +0.0376 |
| `x3_tok_r10__contriever` | 0.224 | -0.217 | +0.0073 | yes | +0.0361 | +0.0375 |
| `x3_tok_r10__qwen3emb06` | 0.221 | -0.213 | +0.0077 | yes | +0.0323 | +0.0338 |
| `x3_tok_r05__mpnet` | 0.398 | -0.388 | +0.0094 | yes | +0.0358 | +0.0422 |
| `x3_tok_r05__contriever` | 0.403 | -0.389 | +0.0137 | yes | +0.0644 | +0.0816 |
| `x3_tok_r05__qwen3emb06` | 0.399 | -0.388 | +0.0112 | yes | +0.0435 | +0.0532 |
| `x3add_schnaus224__mpnet` | 0.218 | -0.210 | +0.0078 | yes | +0.0356 | +0.0356 |
| `x3add_schnaus224__contriever` | 0.224 | -0.214 | +0.0094 | yes | +0.0377 | +0.0338 |
| `x3add_schnaus224__qwen3emb06` | 0.221 | -0.211 | +0.0097 | yes | +0.0290 | +0.0253 |

Over the 12 X3 pairs:
- median sum +0.0078; 12 of 12 within 0.05; largest |sum| 0.0137; 12 informative (|e_m| ≥ 0.10).
- The plan's P1 thresholds, applied descriptively: |median| ≤ 0.03 is True; at least 80% within 0.05 is True.

## 7. Slopes and the level–slope orientation (descriptive)

The `ctr` line at k = 10, m = 1024 is b = 0.1593 + 0.2362 ln L. It is shown for orientation only.

| pair | b(1) | b(10) | b(100) | line b(10) at L(10, 1024) | residual |
|---|---|---|---|---|---|
| `dinov2_b__mpnet` | -0.202 | -0.421 | -0.473 | -0.456 | +0.034 |
| `dinov2_b__contriever` | -0.141 | -0.428 | -0.473 | -0.453 | +0.026 |
| `dinov2_b__qwen3emb06` | -0.154 | -0.413 | -0.473 | -0.456 | +0.043 |
| `x3_cls_r10__mpnet` | -0.372 | -0.283 | -0.001 | -0.042 | -0.241 |
| `x3_cls_r10__contriever` | -0.378 | -0.292 | 0.000 | -0.039 | -0.254 |
| `x3_cls_r10__qwen3emb06` | -0.387 | -0.285 | -0.002 | -0.043 | -0.242 |
| `x3_tok_r10__mpnet` | -0.326 | -0.298 | -0.128 | -0.055 | -0.242 |
| `x3_tok_r10__contriever` | -0.335 | -0.311 | -0.123 | -0.062 | -0.249 |
| `x3_tok_r10__qwen3emb06` | -0.331 | -0.298 | -0.129 | -0.056 | -0.242 |
| `x3_tok_r05__mpnet` | -0.298 | -0.398 | -0.379 | -0.318 | -0.080 |
| `x3_tok_r05__contriever` | -0.236 | -0.406 | -0.374 | -0.325 | -0.081 |
| `x3_tok_r05__qwen3emb06` | -0.276 | -0.396 | -0.379 | -0.320 | -0.076 |
| `x3add_schnaus224__mpnet` | -0.314 | -0.300 | -0.120 | -0.055 | -0.246 |
| `x3add_schnaus224__contriever` | -0.316 | -0.312 | -0.116 | -0.060 | -0.252 |
| `x3add_schnaus224__qwen3emb06` | -0.343 | -0.301 | -0.121 | -0.055 | -0.246 |

## 8. Full curves: draw-mean s(m, k), column `obs`

**k = 1**

| pair | m=256 | m=512 | m=1024 | m=2048 | m=4096 | m=8192 | m=16384 |
|---|---|---|---|---|---|---|---|
| `dinov2_b__mpnet` | 0.0491 | 0.0260 | 0.0265 | 0.0186 | 0.0193 | 0.0189 | 0.0179 |
| `dinov2_b__contriever` | 0.0346 | 0.0286 | 0.0272 | 0.0197 | 0.0212 | 0.0182 | 0.0181 |
| `dinov2_b__qwen3emb06` | 0.0361 | 0.0274 | 0.0252 | 0.0194 | 0.0202 | 0.0214 | 0.0158 |
| `x3_cls_r10__mpnet` | 0.3013 | 0.2052 | 0.1697 | 0.1224 | 0.0976 | 0.0737 | 0.0654 |
| `x3_cls_r10__contriever` | 0.3109 | 0.2069 | 0.1659 | 0.1142 | 0.0941 | 0.0741 | 0.0650 |
| `x3_cls_r10__qwen3emb06` | 0.3143 | 0.2200 | 0.1655 | 0.1180 | 0.0945 | 0.0750 | 0.0642 |
| `x3_tok_r10__mpnet` | 0.2892 | 0.2121 | 0.1647 | 0.1219 | 0.1071 | 0.0888 | 0.0725 |
| `x3_tok_r10__contriever` | 0.2814 | 0.1965 | 0.1534 | 0.1214 | 0.0980 | 0.0786 | 0.0693 |
| `x3_tok_r10__qwen3emb06` | 0.2833 | 0.2114 | 0.1626 | 0.1310 | 0.1097 | 0.0876 | 0.0690 |
| `x3_tok_r05__mpnet` | 0.0847 | 0.0632 | 0.0413 | 0.0335 | 0.0272 | 0.0269 | 0.0254 |
| `x3_tok_r05__contriever` | 0.0661 | 0.0541 | 0.0391 | 0.0324 | 0.0295 | 0.0269 | 0.0252 |
| `x3_tok_r05__qwen3emb06` | 0.0789 | 0.0586 | 0.0407 | 0.0354 | 0.0304 | 0.0273 | 0.0242 |
| `x3add_schnaus224__mpnet` | 0.3011 | 0.2039 | 0.1647 | 0.1233 | 0.1009 | 0.0890 | 0.0809 |
| `x3add_schnaus224__contriever` | 0.2894 | 0.1920 | 0.1546 | 0.1222 | 0.0997 | 0.0838 | 0.0752 |
| `x3add_schnaus224__qwen3emb06` | 0.3035 | 0.2073 | 0.1633 | 0.1311 | 0.1056 | 0.0844 | 0.0694 |
| chance k/(m−1) | 0.0039 | 0.0020 | 0.0010 | 0.0005 | 0.0002 | 0.0001 | 0.0001 |

**k = 10**

| pair | m=256 | m=512 | m=1024 | m=2048 | m=4096 | m=8192 | m=16384 |
|---|---|---|---|---|---|---|---|
| `dinov2_b__mpnet` | 0.1306 | 0.1029 | 0.0740 | 0.0553 | 0.0411 | 0.0309 | 0.0233 |
| `dinov2_b__contriever` | 0.1336 | 0.1024 | 0.0747 | 0.0541 | 0.0406 | 0.0309 | 0.0230 |
| `dinov2_b__qwen3emb06` | 0.1346 | 0.1016 | 0.0740 | 0.0541 | 0.0409 | 0.0318 | 0.0247 |
| `x3_cls_r10__mpnet` | 0.4429 | 0.4807 | 0.4266 | 0.3365 | 0.2547 | 0.1970 | 0.1524 |
| `x3_cls_r10__contriever` | 0.4460 | 0.4816 | 0.4324 | 0.3388 | 0.2571 | 0.1936 | 0.1473 |
| `x3_cls_r10__qwen3emb06` | 0.4447 | 0.4818 | 0.4254 | 0.3360 | 0.2564 | 0.1954 | 0.1520 |
| `x3_tok_r10__mpnet` | 0.5000 | 0.4784 | 0.4028 | 0.3217 | 0.2523 | 0.1965 | 0.1540 |
| `x3_tok_r10__contriever` | 0.4989 | 0.4644 | 0.3922 | 0.3100 | 0.2402 | 0.1863 | 0.1444 |
| `x3_tok_r10__qwen3emb06` | 0.5012 | 0.4771 | 0.4020 | 0.3181 | 0.2493 | 0.1936 | 0.1556 |
| `x3_tok_r05__mpnet` | 0.2095 | 0.1667 | 0.1325 | 0.0965 | 0.0738 | 0.0525 | 0.0420 |
| `x3_tok_r05__contriever` | 0.2075 | 0.1630 | 0.1287 | 0.0942 | 0.0695 | 0.0525 | 0.0394 |
| `x3_tok_r05__qwen3emb06` | 0.2101 | 0.1649 | 0.1315 | 0.0960 | 0.0728 | 0.0539 | 0.0416 |
| `x3add_schnaus224__mpnet` | 0.4955 | 0.4797 | 0.4043 | 0.3204 | 0.2485 | 0.1938 | 0.1530 |
| `x3add_schnaus224__contriever` | 0.4959 | 0.4663 | 0.3943 | 0.3103 | 0.2399 | 0.1849 | 0.1438 |
| `x3add_schnaus224__qwen3emb06` | 0.4997 | 0.4785 | 0.4040 | 0.3172 | 0.2477 | 0.1916 | 0.1546 |
| chance k/(m−1) | 0.0392 | 0.0196 | 0.0098 | 0.0049 | 0.0024 | 0.0012 | 0.0006 |

**k = 100**

| pair | m=256 | m=512 | m=1024 | m=2048 | m=4096 | m=8192 | m=16384 |
|---|---|---|---|---|---|---|---|
| `dinov2_b__mpnet` | 0.4573 | 0.3002 | 0.2152 | 0.1570 | 0.1157 | 0.0846 | 0.0615 |
| `dinov2_b__contriever` | 0.4525 | 0.2940 | 0.2152 | 0.1576 | 0.1149 | 0.0839 | 0.0604 |
| `dinov2_b__qwen3emb06` | 0.4564 | 0.2994 | 0.2163 | 0.1581 | 0.1155 | 0.0847 | 0.0614 |
| `x3_cls_r10__mpnet` | 0.4679 | 0.3703 | 0.3735 | 0.4367 | 0.5019 | 0.4572 | 0.3662 |
| `x3_cls_r10__contriever` | 0.4703 | 0.3704 | 0.3766 | 0.4413 | 0.5073 | 0.4611 | 0.3685 |
| `x3_cls_r10__qwen3emb06` | 0.4672 | 0.3713 | 0.3788 | 0.4384 | 0.5037 | 0.4574 | 0.3645 |
| `x3_tok_r10__mpnet` | 0.6430 | 0.5704 | 0.5410 | 0.5251 | 0.5062 | 0.4339 | 0.3455 |
| `x3_tok_r10__contriever` | 0.6170 | 0.5462 | 0.5350 | 0.5224 | 0.4994 | 0.4259 | 0.3369 |
| `x3_tok_r10__qwen3emb06` | 0.6419 | 0.5715 | 0.5494 | 0.5307 | 0.5076 | 0.4341 | 0.3445 |
| `x3_tok_r05__mpnet` | 0.5354 | 0.3901 | 0.3064 | 0.2428 | 0.1868 | 0.1436 | 0.1060 |
| `x3_tok_r05__contriever` | 0.5148 | 0.3771 | 0.3050 | 0.2419 | 0.1850 | 0.1418 | 0.1041 |
| `x3_tok_r05__qwen3emb06` | 0.5301 | 0.3911 | 0.3090 | 0.2449 | 0.1871 | 0.1432 | 0.1053 |
| `x3add_schnaus224__mpnet` | 0.6313 | 0.5585 | 0.5346 | 0.5219 | 0.5090 | 0.4386 | 0.3465 |
| `x3add_schnaus224__contriever` | 0.6084 | 0.5378 | 0.5297 | 0.5195 | 0.5033 | 0.4314 | 0.3388 |
| `x3add_schnaus224__qwen3emb06` | 0.6296 | 0.5611 | 0.5431 | 0.5282 | 0.5112 | 0.4390 | 0.3460 |
| chance k/(m−1) | 0.3922 | 0.1957 | 0.0978 | 0.0489 | 0.0244 | 0.0122 | 0.0061 |

## 9. Protocol differences that remain (not replicated)

- Registers: we use the no-register DINOv2-B. Their p. 26 pools "class, register and patch tokens", and their p. 27 names a with-registers model for the RAE only.
- Library and position-embedding interpolation: HF-style here; see §2 for the effect of the timm alternative.
- bf16 storage (p. 21).
- Centering: their aligner centers on a disjoint training half (p. 25), and their metric centers each side on its 10,000 samples (p. 26). We center on the 16,384 pool and renormalize.
- Captions: they average all of the up to 7 captions (p. 25); we use 5. E0's arm CAall shows that ALL against AVG5 changes the DINOv2 × MPNet cell by less than 0.001.
- Queries: they use all 10,000 points of a random subsample. We use the nested prefixes with S1.1's anchors (`obs`), and all H2 items as a sensitivity.
- Qwen3-Embedding: 0.6B here, 8B there.

## 10. Observations (descriptive; wording written after seeing the numbers)

- **The gap closes with depth.** Only the block changes between the reference and `x3_cls_r10`: both are the CLS token, with no norm, at 518 px. That change alone moves DINOv2 × MPNet from 0.0772 to 0.4289, against Schnaus' about 0.38. The patch mean at the same mid depth gets to 0.1315. The Schnaus-style feature sits within 0.01 in ln L of the block-11 patch mean, at 0.4098 against 0.4064. So the final norm, putting the CLS token in the mean, and 224 px change almost nothing. The three encoders agree to about 0.01 in every row.
- **k = 10 at m = 10,000.** The block-11 features give 0.1810–0.1832 with MPNet. That is inside the 0.07–0.19 read from Schnaus' Fig. 14b (descriptive, no rule).
- **The scale reading still holds at these levels.** On k ∈ {10, 100}, e_k + e_m stays between +0.0015 and +0.0137 for all 12 X3 pairs, with |e_m| ≥ 0.14. Agreement still falls with m at fixed k: b(10) is -0.312 to -0.283 for the block-11 features.
- **The level–slope line does not extend to these levels.** The E0 `ctr` line predicts b(10) of about −0.04 to −0.06 at L(10, 1024) ≈ 0.40–0.43. The block-11 pairs give b(10) residuals from -0.254 to -0.241, which are much steeper. The frozen plan (§0, B6) said in advance that if the levels came out near Schnaus', P1 and the level–slope regularity could not both hold, and that the tension would fall on the latter. That is what these exploratory numbers show. No E0 label changes.
- **Curve shapes at high levels.** These pairs have a curve that is not non-increasing in m (pair, k, the m of the maximum): `x3_cls_r10__mpnet` k = 10, peak at m = 512 (k/m = 0.020); `x3_cls_r10__mpnet` k = 100, peak at m = 4096 (k/m = 0.024); `x3_cls_r10__contriever` k = 10, peak at m = 512 (k/m = 0.020); `x3_cls_r10__contriever` k = 100, peak at m = 4096 (k/m = 0.024); `x3_cls_r10__qwen3emb06` k = 10, peak at m = 512 (k/m = 0.020); `x3_cls_r10__qwen3emb06` k = 100, peak at m = 4096 (k/m = 0.024).
  - The block-11 patch-mean features (`x3_tok_r10`, `x3add_schnaus224`) fall at every step at k = 100, but flatten between m = 1024 and 4096. Their b(100) is -0.129 to -0.116.
  - For `x3_cls_r10`, b(100) is -0.002 to 0.000. That number summarizes a curve that rises and then falls; the curve is not flat.
  - The sums in §6 are close to 0, so these shapes line up along k/m. The k = 10 and k = 100 peaks of `x3_cls_r10` sit at k/m of about 0.02.

## 11. Compute (no cap set)

- **GPU:** 0.0385 h, all of it the extraction process (wall time).
- **CPU:** extraction 0.091 core-h; scores 0.049 core-h; report 0.0001 core-h. All runs were pinned to at most 6 cores.

## 12. Input hashes

- `obs_X3.json` `f50c30f475aa4a3738b65676b0d2b6ed80cc4574634b6b9e48fa5dd0020b8f00`
- `x3/obs/obs_X3.parquet` `69155062c6136a4eb03539e9e30d491ea77b93aa0c20c2092e6255c6e787a78c`
- `x3/eval/X3_cells.parquet` `179c89fe96dc5bb0464ebf2ee6a46c6ec887a948c98280536f3bf1cb93070729`
- `x3/eval/X3_drawslopes.parquet` `308c13f873774c08ea3e76870f5f9772f0b8d1935ff3c27f8a59f9b23427b513`
- `x3/eval/X3_slopes.parquet` `a864756febb494d822d6fb056ee81c9c679dac8d0601cc0c2216a062660b1405`
- `x3/eval/X3_elasticity.parquet` `4e776b53fc0339f078374ce8b99865acf549b478dc47d3a1f28bf4dd8fc8d522`
- `x3/eval/X3_levels.parquet` `cb5312f9870f169e0e24d5c356e8c0529060cf5f83c5383c39f6726e5f487eff`
- `x3/eval/X3_levels_allH2.parquet` `12154d532ea866ae9a926b9852f45b86f7cf92782719c21cd646f530d38405a8`
- `results/E0_sentence/eval/E0_cells_CA.parquet` `a04bc4e7efd86cf214dc995223fdcab3f668dad392901bd29712d619f73ecc5e`
- `results/E0_sentence/eval/E0_slopes_CA.parquet` `3f777f8400288d2d72c57c99dd6ac72e2a3797c3a36c743ba36853ec78cb5308`
- `results/E0_sentence/eval/E0_elasticity_CA.parquet` `46fa3fa07561f285b5df3ddd41a9c58826739a900aaa3e8422b5afddc50acd64`
- `results/E0_sentence/eval/E0_levels_CA.parquet` `e86da8255097f140b1c28a7cd5ee2c8300126d1955eb1ae46729f81e2158cc5c`
- `x3/validation/VALIDATION_X3.json` `a8ba5774f47c3b07a28759c7b9b4594b3a7d5ef6d2c186561e71b91a52f51b69`
- `x3/extraction/EXTRACTION_X3.json` `317e108ed511281267b02708dbe723af8c1345abe360c39cea0d60c864d925a6`

