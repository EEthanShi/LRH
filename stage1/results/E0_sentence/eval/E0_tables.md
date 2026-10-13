# E0 tables (produced by code; ANALYSIS_PLAN_E0.md v1.0 §14)

## Confirmatory labels (order fixed: P1, P2b, P2c)

| prediction | label | key statistics |
|---|---|---|
| P1 | holds | informative 15/15; median sum 0.005985754050756853; frac |sum|<=0.05 1.0 |
| P2b | uninformative | lambda None; lambda_e {}; median |gap| 0.0921 (guard 0.11295) |
| P2c | uninformative | median gamma 0.4353; gamma_e {'contriever': 0.316027110567337, 'mpnet': 0.3456849726227776, 'qwen3emb06': 0.5160717663662399}; median dln 0.1777 |

## Secondary labels (no confirmatory weight)

| analysis | cell | label |
|---|---|---|
| P1 | ctrold | holds |
| P1 | D1 | holds |
| P1 | D2 | holds |
| P1 | D3 | holds |
| P1 | D4 | holds |
| P1 | A | holds |
| P1 | Aall | holds |
| P1 | C | holds |
| P1 | CA | holds |
| P1 | CAall | holds |
| P1 | L | holds |
| P1 | Pdd | holds |
| P2b | P|k1 | uninformative |
| P2b | P|k100 | uninformative |
| P2b | P|mref2048 | uninformative |
| P2b | P|obs_allH2 | uninformative |
| P2b | P|obs_clean | uninformative |
| P2b | A | holds |
| P2b | Aall | holds |
| P2b | Pdd | uninformative |
| P2b | L | inconclusive |
| P2b | C | uninformative |
| P2b | CA | uninformative |
| P2b | CAall | uninformative |
| P2c_other_k | k1 | holds |
| P2c_other_k | k100 | uninformative |
| P2c_other_k | obs_allH2 | uninformative |
| P2c_other_k | obs_clean | uninformative |

## Per arm, per class

| arm | class | n | L(10,1024) median [min,max] | L(100,8192) median | median b(10) | median sum k10,100 |
|---|---|---|---|---|---|---|
| ctrold | cross-modal | 15 | 0.0462 [0.0361, 0.0866] | 0.0529 | -0.5645 | -0.0007 |
| ctrold | cross-vision | 10 | 0.1963 [0.1290, 0.2823] | 0.2119 | -0.2082 | 0.0138 |
| ctrold | seed-seed | 4 | 0.4100 [0.3816, 0.7157] | 0.4226 | -0.0044 | 0.0058 |
| P | cross-modal-sentence | 15 | 0.0594 [0.0460, 0.1375] | 0.0671 | -0.5248 | 0.0060 |
| D1 | cross-modal-sentence | 15 | 0.0595 [0.0458, 0.1320] | 0.0675 | -0.5092 | 0.0122 |
| D2 | cross-modal-sentence | 15 | 0.0645 [0.0517, 0.1492] | 0.0724 | -0.4816 | 0.0091 |
| D3 | cross-modal-sentence | 15 | 0.0656 [0.0534, 0.1554] | 0.0739 | -0.4568 | 0.0110 |
| D4 | cross-modal-sentence | 15 | 0.0667 [0.0536, 0.1578] | 0.0747 | -0.4459 | 0.0116 |
| A | cross-modal-sentence | 15 | 0.0668 [0.0536, 0.1607] | 0.0752 | -0.4321 | 0.0119 |
| Aall | cross-modal-sentence | 15 | 0.0667 [0.0536, 0.1608] | 0.0752 | -0.4323 | 0.0120 |
| C | cross-modal-sentence | 15 | 0.0662 [0.0498, 0.1454] | 0.0762 | -0.5210 | 0.0068 |
| CA | cross-modal-sentence | 15 | 0.0740 [0.0586, 0.1687] | 0.0846 | -0.4329 | 0.0151 |
| CAall | cross-modal-sentence | 15 | 0.0739 [0.0586, 0.1686] | 0.0846 | -0.4322 | 0.0151 |
| L | cross-modal-sentence | 15 | 0.0591 [0.0459, 0.1296] | 0.0675 | -0.5227 | 0.0061 |
| Pdd | cross-modal-sentence | 15 | 0.0593 [0.0459, 0.1371] | 0.0672 | -0.5236 | 0.0045 |
| S1.1-clip | cross-modal | 15 | 0.0430 [0.0327, 0.0981] | 0.0491 | -0.5706 | 0.0044 |
| S1.1-clip | cross-vision | 10 | 0.1885 [0.1288, 0.2656] | 0.2049 | -0.2136 | 0.0080 |
| S1.1-clip | seed-seed | 4 | 0.5253 [0.3601, 0.6918] | 0.5384 | -0.0138 | 0.0049 |
| S1.1-l2 | cross-modal | 15 | 0.0405 [0.0295, 0.0763] | 0.0453 | -0.5663 | -0.0020 |
| S1.1-l2 | cross-vision | 10 | 0.1852 [0.1260, 0.2670] | 0.2005 | -0.2212 | 0.0096 |
| S1.1-l2 | seed-seed | 4 | 0.5262 [0.3833, 0.6966] | 0.5435 | -0.0118 | 0.0056 |

## Arm P, per new pair

| pair | L(10,1024) | b(1) | b(10) | b(100) | e_k | e_m | sum (k10,100) |
|---|---|---|---|---|---|---|---|
| dinov2_b__mpnet | 0.0613 | -0.4110 | -0.5122 | -0.5222 | 0.5284 | -0.5168 | 0.0115 |
| dinov2_b__contriever | 0.0561 | -0.2049 | -0.5493 | -0.5391 | 0.5400 | -0.5444 | -0.0043 |
| dinov2_b__qwen3emb06 | 0.0594 | -0.3934 | -0.5099 | -0.5211 | 0.5269 | -0.5147 | 0.0122 |
| clip_b__mpnet | 0.0502 | -0.3301 | -0.5284 | -0.5738 | 0.5665 | -0.5509 | 0.0155 |
| clip_b__contriever | 0.0460 | -0.3398 | -0.5451 | -0.5957 | 0.5766 | -0.5706 | 0.0060 |
| clip_b__qwen3emb06 | 0.0498 | -0.3359 | -0.5248 | -0.5711 | 0.5602 | -0.5480 | 0.0122 |
| mae_b__mpnet | 0.0491 | -0.4304 | -0.5790 | -0.5770 | 0.5926 | -0.5774 | 0.0152 |
| mae_b__contriever | 0.0474 | -0.3187 | -0.6103 | -0.5969 | 0.6084 | -0.6022 | 0.0063 |
| mae_b__qwen3emb06 | 0.0489 | -0.3476 | -0.5866 | -0.5778 | 0.5864 | -0.5806 | 0.0058 |
| augreg_b__mpnet | 0.0775 | -0.4190 | -0.5079 | -0.4765 | 0.4871 | -0.4926 | -0.0055 |
| augreg_b__contriever | 0.0715 | -0.4436 | -0.5459 | -0.4963 | 0.5148 | -0.5212 | -0.0064 |
| augreg_b__qwen3emb06 | 0.0761 | -0.4226 | -0.5220 | -0.4764 | 0.4909 | -0.4995 | -0.0086 |
| siglip_b__mpnet | 0.1372 | -0.3814 | -0.4425 | -0.3399 | 0.3942 | -0.3913 | 0.0028 |
| siglip_b__contriever | 0.1163 | -0.4562 | -0.4736 | -0.3780 | 0.4341 | -0.4257 | 0.0084 |
| siglip_b__qwen3emb06 | 0.1375 | -0.4012 | -0.4392 | -0.3401 | 0.3953 | -0.3897 | 0.0055 |

## Descriptive P2

```
{
 "within_vit_permutation": {
  "rho": 0.7899888765294771,
  "p_one_sided": 9.99990000099999e-06,
  "R": 100000,
  "seed": 2266047818474480236
 },
 "text_model_level": {
  "rho": 0.942857142857143,
  "p_exact_one_sided": 0.008333333333333333,
  "n_perms": 720,
  "level_by_text": [
   0.05034788986377402,
   0.04295406802450704,
   0.04144104111215817,
   0.05614749142622155,
   0.06128927972631028,
   0.059366025151831224
  ],
  "slope_by_text": [
   -0.5691430413122778,
   -0.5817362179815275,
   -0.5734862824105721,
   -0.5458609397700313,
   -0.5122296526065334,
   -0.5220403042077384
  ]
 }
}
```

