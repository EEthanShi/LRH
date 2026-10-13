# S1.1 on COCO under PREREG v1.1: descriptive results

Gate G1 is NOT run as a confirmatory test: no eligible estimator (PREREG v1.1 D4.7; analysis plan V1). The crossing law is reported as an unvalidated predictor (iii-LSlog-EB).

## Configuration `clip` (primary): pre = prh_clip+l2, clip = pool-stored

61 pairs (29 primary); draws per cell [5].

### Fixed-k slope classes, primary pair set (D4.8 slopes; rising = conformal-like, flat = smooth, falling = rough)

| class | k | rising | flat | falling | n | median slope |
|---|---|---|---|---|---|---|
| seed-seed | 1 | 0 | 4 | 0 | 4 | 0.009 |
| seed-seed | 10 | 0 | 4 | 0 | 4 | -0.014 |
| seed-seed | 100 | 0 | 3 | 1 | 4 | -0.055 |
| cross-vision | 1 | 0 | 7 | 3 | 10 | -0.076 |
| cross-vision | 10 | 0 | 0 | 10 | 10 | -0.214 |
| cross-vision | 100 | 0 | 0 | 10 | 10 | -0.278 |
| cross-modal | 1 | 0 | 0 | 15 | 15 | -0.522 |
| cross-modal | 10 | 0 | 0 | 15 | 15 | -0.571 |
| cross-modal | 100 | 0 | 0 | 15 | 15 | -0.598 |
| all | 1 | 0 | 11 | 18 | 29 | -0.405 |
| all | 10 | 0 | 4 | 25 | 29 | -0.509 |
| all | 100 | 0 | 3 | 26 | 29 | -0.428 |

Only smooth class observed: False.

### Fixed-k slope classes, all (36 seed pairs) pair set (D4.8 slopes; rising = conformal-like, flat = smooth, falling = rough)

| class | k | rising | flat | falling | n | median slope |
|---|---|---|---|---|---|---|
| seed-seed | 1 | 0 | 36 | 0 | 36 | 0.012 |
| seed-seed | 10 | 0 | 36 | 0 | 36 | -0.018 |
| seed-seed | 100 | 0 | 28 | 8 | 36 | -0.061 |
| cross-vision | 1 | 0 | 7 | 3 | 10 | -0.076 |
| cross-vision | 10 | 0 | 0 | 10 | 10 | -0.214 |
| cross-vision | 100 | 0 | 0 | 10 | 10 | -0.278 |
| cross-modal | 1 | 0 | 0 | 15 | 15 | -0.522 |
| cross-modal | 10 | 0 | 0 | 15 | 15 | -0.571 |
| cross-modal | 100 | 0 | 0 | 15 | 15 | -0.598 |
| all | 1 | 0 | 43 | 18 | 61 | -0.004 |
| all | 10 | 0 | 36 | 25 | 61 | -0.023 |
| all | 100 | 0 | 28 | 33 | 61 | -0.112 |

Only smooth class observed: False.

Slope classes, primary pairs, all H2 items of each prefix (draft quantity obs_allH2): seed-seed k=1: 0/4/0 (median 0.011); seed-seed k=10: 0/4/0 (median -0.013); seed-seed k=100: 0/3/1 (median -0.053); cross-vision k=1: 0/6/4 (median -0.088); cross-vision k=10: 0/0/10 (median -0.217); cross-vision k=100: 0/0/10 (median -0.277); cross-modal k=1: 0/0/15 (median -0.553); cross-modal k=10: 0/0/15 (median -0.592); cross-modal k=100: 0/0/15 (median -0.605)

Slope classes, primary pairs, anchors without an exact k-boundary tie (O11): seed-seed k=1: 0/4/0 (median 0.009); seed-seed k=10: 0/4/0 (median -0.014); seed-seed k=100: 0/3/1 (median -0.055); cross-vision k=1: 0/7/3 (median -0.076); cross-vision k=10: 0/0/10 (median -0.214); cross-vision k=100: 0/0/10 (median -0.278); cross-modal k=1: 0/0/15 (median -0.521); cross-modal k=10: 0/0/15 (median -0.571); cross-modal k=100: 0/0/15 (median -0.598)

Slope classes, primary pairs, EXPLORATORY, not in the plan: chance-corrected (s - c)/(1 - c): seed-seed k=1: 0/4/0 (median 0.010); seed-seed k=10: 0/4/0 (median -0.009); seed-seed k=100: 0/4/0 (median -0.001); cross-vision k=1: 0/7/3 (median -0.068); cross-vision k=10: 0/0/10 (median -0.195); cross-vision k=100: 0/3/7 (median -0.153); cross-modal k=1: 0/0/15 (median -0.492); cross-modal k=10: 0/0/15 (median -0.489); cross-modal k=100: 0/0/15 (median -0.225)

### Descriptive accuracy, primary pair set (no thresholds, no pass/fail; plan V5)

| predictor | class | pairs | median abs log(X/obs) (n cells) | MSLE non-flat (units/cells) | slope sign agreement |
|---|---|---|---|---|---|
| N0 | pooled | 29/29 | 0.396 (468) | 0.667 (69/414) | 7/69 = 0.101 |
| N0 | seed-seed | 4/4 | 0.036 (69) | 0.043 (1/6) | 0/1 = 0.000 |
| N0 | cross-vision | 10/10 | 0.213 (167) | 0.156 (23/138) | 0/23 = 0.000 |
| N0 | cross-modal | 15/15 | 0.816 (232) | 0.943 (45/270) | 7/45 = 0.156 |
| N1 | pooled | 29/29 | 1.599 (468) | 2.894 (69/414) | 69/69 = 1.000 |
| N1 | seed-seed | 4/4 | 2.156 (69) | 2.886 (1/6) | 1/1 = 1.000 |
| N1 | cross-vision | 10/10 | 2.094 (167) | 4.361 (23/138) | 23/23 = 1.000 |
| N1 | cross-modal | 15/15 | 1.388 (232) | 2.144 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB (unvalidated) | pooled | 29/29 | 0.312 (468) | 1.122 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB (unvalidated) | seed-seed | 4/4 | 0.020 (69) | 0.008 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB (unvalidated) | cross-vision | 10/10 | 0.150 (167) | 0.030 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB (unvalidated) | cross-modal | 15/15 | 1.050 (232) | 1.704 (45/270) | 45/45 = 1.000 |
| N1|mu | pooled | 29/29 | 1.588 (468) | 2.863 (69/414) | 69/69 = 1.000 |
| N1|mu | seed-seed | 4/4 | 2.156 (69) | 2.886 (1/6) | 1/1 = 1.000 |
| N1|mu | cross-vision | 10/10 | 2.065 (167) | 4.317 (23/138) | 23/23 = 1.000 |
| N1|mu | cross-modal | 15/15 | 1.370 (232) | 2.120 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB|mu | pooled | 29/29 | 0.315 (468) | 1.162 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB|mu | seed-seed | 4/4 | 0.029 (69) | 0.008 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB|mu | cross-vision | 10/10 | 0.153 (167) | 0.031 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB|mu | cross-modal | 15/15 | 1.064 (232) | 1.766 (45/270) | 45/45 = 1.000 |
| P2 (v1.0, expl.) | pooled | 29/29 | inf (468) | 66.901 (69/414) | 69/69 = 1.000 |
| P2 (v1.0, expl.) | seed-seed | 4/4 | 0.326 (69) | 123.751 (1/6) | 1/1 = 1.000 |
| P2 (v1.0, expl.) | cross-vision | 10/10 | inf (167) | 82.327 (23/138) | 23/23 = 1.000 |
| P2 (v1.0, expl.) | cross-modal | 15/15 | inf (232) | 57.753 (45/270) | 45/45 = 1.000 |
| N2 (v1.0, expl.) | pooled | 29/29 | 1.031 (468) | 1.244 (69/414) | 69/69 = 1.000 |
| N2 (v1.0, expl.) | seed-seed | 4/4 | 0.972 (69) | 0.371 (1/6) | 1/1 = 1.000 |
| N2 (v1.0, expl.) | cross-vision | 10/10 | 0.871 (167) | 0.989 (23/138) | 23/23 = 1.000 |
| N2 (v1.0, expl.) | cross-modal | 15/15 | 1.121 (232) | 1.393 (45/270) | 45/45 = 1.000 |
| N0H1 (v1.0 N0) | pooled | 29/29 | 0.386 (468) | 0.674 (69/414) | 4/69 = 0.058 |
| N0H1 (v1.0 N0) | seed-seed | 4/4 | 0.040 (69) | 0.043 (1/6) | 0/1 = 0.000 |
| N0H1 (v1.0 N0) | cross-vision | 10/10 | 0.238 (167) | 0.156 (23/138) | 0/23 = 0.000 |
| N0H1 (v1.0 N0) | cross-modal | 15/15 | 0.813 (232) | 0.952 (45/270) | 4/45 = 0.089 |
| iii-LSlog-EB A->B only | pooled | 29/29 | 0.468 (468) | 1.026 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB A->B only | seed-seed | 4/4 | 0.091 (69) | 0.001 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB A->B only | cross-vision | 10/10 | 0.263 (167) | 0.081 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB A->B only | cross-modal | 15/15 | 1.133 (232) | 1.532 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB B->A only | pooled | 29/29 | 0.242 (468) | 1.213 (69/414) | 68/69 = 0.986 |
| iii-LSlog-EB B->A only | seed-seed | 4/4 | 0.085 (69) | 0.043 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB B->A only | cross-vision | 10/10 | 0.098 (167) | 0.015 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB B->A only | cross-modal | 15/15 | 0.970 (232) | 1.851 (45/270) | 44/45 = 0.978 |
| N1 A->B only | pooled | 29/29 | 1.777 (468) | 3.416 (69/414) | 69/69 = 1.000 |
| N1 A->B only | seed-seed | 4/4 | 1.908 (69) | 2.141 (1/6) | 1/1 = 1.000 |
| N1 A->B only | cross-vision | 10/10 | 2.410 (167) | 5.292 (23/138) | 23/23 = 1.000 |
| N1 A->B only | cross-modal | 15/15 | 1.499 (232) | 2.486 (45/270) | 45/45 = 1.000 |
| N1 B->A only | pooled | 29/29 | 1.513 (468) | 2.565 (69/414) | 69/69 = 1.000 |
| N1 B->A only | seed-seed | 4/4 | 2.323 (69) | 4.139 (1/6) | 1/1 = 1.000 |
| N1 B->A only | cross-vision | 10/10 | 1.971 (167) | 3.827 (23/138) | 23/23 = 1.000 |
| N1 B->A only | cross-modal | 15/15 | 1.242 (232) | 1.884 (45/270) | 45/45 = 1.000 |

### Descriptive accuracy, all (36 seed pairs) pair set (no thresholds, no pass/fail; plan V5)

| predictor | class | pairs | median abs log(X/obs) (n cells) | MSLE non-flat (units/cells) | slope sign agreement |
|---|---|---|---|---|---|
| N0 | pooled | 61/61 | 0.074 (1023) | 0.609 (76/456) | 7/76 = 0.092 |
| N0 | seed-seed | 36/36 | 0.035 (624) | 0.036 (8/48) | 0/8 = 0.000 |
| N0 | cross-vision | 10/10 | 0.213 (167) | 0.156 (23/138) | 0/23 = 0.000 |
| N0 | cross-modal | 15/15 | 0.816 (232) | 0.943 (45/270) | 7/45 = 0.156 |
| N1 | pooled | 61/61 | 1.598 (1023) | 2.835 (76/456) | 76/76 = 1.000 |
| N1 | seed-seed | 36/36 | 1.640 (624) | 2.334 (8/48) | 8/8 = 1.000 |
| N1 | cross-vision | 10/10 | 2.094 (167) | 4.361 (23/138) | 23/23 = 1.000 |
| N1 | cross-modal | 15/15 | 1.388 (232) | 2.144 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB (unvalidated) | pooled | 61/61 | 0.052 (1023) | 1.019 (76/456) | 76/76 = 1.000 |
| iii-LSlog-EB (unvalidated) | seed-seed | 36/36 | 0.019 (624) | 0.003 (8/48) | 8/8 = 1.000 |
| iii-LSlog-EB (unvalidated) | cross-vision | 10/10 | 0.150 (167) | 0.030 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB (unvalidated) | cross-modal | 15/15 | 1.050 (232) | 1.704 (45/270) | 45/45 = 1.000 |
| N1|mu | pooled | 61/61 | 1.589 (1023) | 2.807 (76/456) | 76/76 = 1.000 |
| N1|mu | seed-seed | 36/36 | 1.637 (624) | 2.334 (8/48) | 8/8 = 1.000 |
| N1|mu | cross-vision | 10/10 | 2.065 (167) | 4.317 (23/138) | 23/23 = 1.000 |
| N1|mu | cross-modal | 15/15 | 1.370 (232) | 2.120 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB|mu | pooled | 61/61 | 0.056 (1023) | 1.056 (76/456) | 76/76 = 1.000 |
| iii-LSlog-EB|mu | seed-seed | 36/36 | 0.026 (624) | 0.003 (8/48) | 8/8 = 1.000 |
| iii-LSlog-EB|mu | cross-vision | 10/10 | 0.153 (167) | 0.031 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB|mu | cross-modal | 15/15 | 1.064 (232) | 1.766 (45/270) | 45/45 = 1.000 |
| P2 (v1.0, expl.) | pooled | 61/61 | 1.690 (1023) | 70.900 (76/456) | 73/76 = 0.961 |
| P2 (v1.0, expl.) | seed-seed | 36/36 | 0.287 (624) | 111.999 (8/48) | 5/8 = 0.625 |
| P2 (v1.0, expl.) | cross-vision | 10/10 | inf (167) | 82.327 (23/138) | 23/23 = 1.000 |
| P2 (v1.0, expl.) | cross-modal | 15/15 | inf (232) | 57.753 (45/270) | 45/45 = 1.000 |
| N2 (v1.0, expl.) | pooled | 61/61 | 0.788 (1023) | 1.140 (76/456) | 76/76 = 1.000 |
| N2 (v1.0, expl.) | seed-seed | 36/36 | 0.454 (624) | 0.146 (8/48) | 8/8 = 1.000 |
| N2 (v1.0, expl.) | cross-vision | 10/10 | 0.871 (167) | 0.989 (23/138) | 23/23 = 1.000 |
| N2 (v1.0, expl.) | cross-modal | 15/15 | 1.121 (232) | 1.393 (45/270) | 45/45 = 1.000 |
| N0H1 (v1.0 N0) | pooled | 61/61 | 0.074 (1023) | 0.615 (76/456) | 4/76 = 0.053 |
| N0H1 (v1.0 N0) | seed-seed | 36/36 | 0.037 (624) | 0.036 (8/48) | 0/8 = 0.000 |
| N0H1 (v1.0 N0) | cross-vision | 10/10 | 0.238 (167) | 0.156 (23/138) | 0/23 = 0.000 |
| N0H1 (v1.0 N0) | cross-modal | 15/15 | 0.813 (232) | 0.952 (45/270) | 4/45 = 0.089 |
| iii-LSlog-EB A->B only | pooled | 61/61 | 0.146 (1023) | 0.933 (76/456) | 76/76 = 1.000 |
| iii-LSlog-EB A->B only | seed-seed | 36/36 | 0.061 (624) | 0.013 (8/48) | 8/8 = 1.000 |
| iii-LSlog-EB A->B only | cross-vision | 10/10 | 0.263 (167) | 0.081 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB A->B only | cross-modal | 15/15 | 1.133 (232) | 1.532 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB B->A only | pooled | 61/61 | 0.125 (1023) | 1.102 (76/456) | 75/76 = 0.987 |
| iii-LSlog-EB B->A only | seed-seed | 36/36 | 0.051 (624) | 0.016 (8/48) | 8/8 = 1.000 |
| iii-LSlog-EB B->A only | cross-vision | 10/10 | 0.098 (167) | 0.015 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB B->A only | cross-modal | 15/15 | 0.970 (232) | 1.851 (45/270) | 44/45 = 0.978 |
| N1 A->B only | pooled | 61/61 | 1.684 (1023) | 3.377 (76/456) | 76/76 = 1.000 |
| N1 A->B only | seed-seed | 36/36 | 1.638 (624) | 2.881 (8/48) | 8/8 = 1.000 |
| N1 A->B only | cross-vision | 10/10 | 2.410 (167) | 5.292 (23/138) | 23/23 = 1.000 |
| N1 A->B only | cross-modal | 15/15 | 1.499 (232) | 2.486 (45/270) | 45/45 = 1.000 |
| N1 B->A only | pooled | 61/61 | 1.568 (1023) | 2.532 (76/456) | 76/76 = 1.000 |
| N1 B->A only | seed-seed | 36/36 | 1.700 (624) | 2.455 (8/48) | 8/8 = 1.000 |
| N1 B->A only | cross-vision | 10/10 | 1.971 (167) | 3.827 (23/138) | 23/23 = 1.000 |
| N1 B->A only | cross-modal | 15/15 | 1.242 (232) | 1.884 (45/270) | 45/45 = 1.000 |

### Secondary (O10, descriptive), primary pair set

- k = 1: median fixed-k slope seed-seed 0.009, cross-vision -0.076, cross-modal -0.522; seed >= vision >= modal: True
- k = 10: median fixed-k slope seed-seed -0.014, cross-vision -0.214, cross-modal -0.571; seed >= vision >= modal: True
- k = 100: median fixed-k slope seed-seed -0.055, cross-vision -0.278, cross-modal -0.598; seed >= vision >= modal: True

| class | median e_k | median e_m | median e_k + e_m | median fit rmse | n |
|---|---|---|---|---|---|
| cross-modal | 0.583 | -0.565 | 0.030 | 0.131 | 15 |
| cross-vision | 0.250 | -0.194 | 0.061 | 0.122 | 10 |
| seed-seed | 0.047 | -0.019 | 0.027 | 0.050 | 4 |

corr(e_k, -e_m) across pairs = 0.990

### Secondary (O10, descriptive), all (36 seed pairs) pair set

- k = 1: median fixed-k slope seed-seed 0.012, cross-vision -0.076, cross-modal -0.522; seed >= vision >= modal: True
- k = 10: median fixed-k slope seed-seed -0.018, cross-vision -0.214, cross-modal -0.571; seed >= vision >= modal: True
- k = 100: median fixed-k slope seed-seed -0.061, cross-vision -0.278, cross-modal -0.598; seed >= vision >= modal: True

| class | median e_k | median e_m | median e_k + e_m | median fit rmse | n |
|---|---|---|---|---|---|
| cross-modal | 0.583 | -0.565 | 0.030 | 0.131 | 15 |
| cross-vision | 0.250 | -0.194 | 0.061 | 0.122 | 10 |
| seed-seed | 0.052 | -0.028 | 0.029 | 0.047 | 36 |

corr(e_k, -e_m) across pairs = 0.995

### Observed s_N(m, k) curves per class (median over pairs of the draw-mean; [min, max])

k = 1:

| class | m=256 | m=512 | m=1024 | m=2048 | m=4096 | m=8192 | m=16384 |
|---|---|---|---|---|---|---|---|
| seed-seed | 0.461 [0.33, 0.64] | 0.463 [0.30, 0.63] | 0.468 [0.31, 0.62] | 0.474 [0.30, 0.62] | 0.483 [0.32, 0.63] | 0.482 [0.31, 0.61] | 0.493 [0.32, 0.62] |
| cross-vision | 0.127 [0.07, 0.21] | 0.112 [0.06, 0.16] | 0.098 [0.06, 0.14] | 0.083 [0.05, 0.12] | 0.082 [0.05, 0.12] | 0.085 [0.06, 0.12] | 0.088 [0.06, 0.12] |
| cross-modal | 0.026 [0.01, 0.05] | 0.017 [0.01, 0.03] | 0.010 [0.01, 0.02] | 0.007 [0.00, 0.02] | 0.005 [0.00, 0.01] | 0.004 [0.00, 0.01] | 0.004 [0.00, 0.01] |
| chance k/(m-1) | 0.0039 | 0.0020 | 0.0010 | 0.0005 | 0.0002 | 0.0001 | 0.0001 |

k = 10:

| class | m=256 | m=512 | m=1024 | m=2048 | m=4096 | m=8192 | m=16384 |
|---|---|---|---|---|---|---|---|
| seed-seed | 0.556 [0.40, 0.73] | 0.542 [0.38, 0.71] | 0.530 [0.36, 0.70] | 0.531 [0.35, 0.68] | 0.533 [0.35, 0.68] | 0.537 [0.35, 0.67] | 0.541 [0.35, 0.67] |
| cross-vision | 0.256 [0.19, 0.37] | 0.217 [0.15, 0.32] | 0.189 [0.13, 0.27] | 0.163 [0.11, 0.23] | 0.141 [0.09, 0.20] | 0.123 [0.08, 0.18] | 0.109 [0.07, 0.16] |
| cross-modal | 0.091 [0.08, 0.18] | 0.066 [0.05, 0.13] | 0.043 [0.03, 0.10] | 0.028 [0.02, 0.07] | 0.018 [0.01, 0.05] | 0.013 [0.01, 0.03] | 0.008 [0.01, 0.02] |
| chance k/(m-1) | 0.0392 | 0.0196 | 0.0098 | 0.0049 | 0.0024 | 0.0012 | 0.0006 |

k = 100:

| class | m=256 | m=512 | m=1024 | m=2048 | m=4096 | m=8192 | m=16384 |
|---|---|---|---|---|---|---|---|
| seed-seed | 0.693 [0.63, 0.89] | 0.606 [0.53, 0.84] | 0.572 [0.45, 0.79] | 0.558 [0.41, 0.75] | 0.550 [0.39, 0.73] | 0.543 [0.37, 0.71] | 0.545 [0.36, 0.70] |
| cross-vision | 0.566 [0.49, 0.68] | 0.418 [0.35, 0.55] | 0.332 [0.26, 0.46] | 0.273 [0.20, 0.39] | 0.236 [0.17, 0.34] | 0.205 [0.14, 0.29] | 0.175 [0.12, 0.25] |
| cross-modal | 0.432 [0.42, 0.50] | 0.258 [0.24, 0.34] | 0.161 [0.15, 0.26] | 0.107 [0.09, 0.20] | 0.073 [0.06, 0.15] | 0.049 [0.04, 0.11] | 0.033 [0.03, 0.08] |
| chance k/(m-1) | 0.3922 | 0.1957 | 0.0978 | 0.0489 | 0.0244 | 0.0122 | 0.0061 |

### Per pair (draw-mean obs on the predicted anchors; D4.8 slopes)

| pair | class | s_N k=1 m=256 / 1024 / 16384 | s_N k=10 m=256 / 1024 / 16384 | s_N k=100 m=256 / 1024 / 16384 | slope k=1 / 10 / 100 | class k=1/10/100 (+ rising, 0 flat, - falling) |
|---|---|---|---|---|---|---|
| pythia410m_s1__pythia410m_s2 | seed-seed | 0.454 / 0.480 / 0.516 | 0.561 / 0.537 / 0.544 | 0.684 / 0.575 / 0.550 | 0.028 / -0.006 / -0.046 | 0/0/0 |
| pythia410m_s1__pythia410m_s3 (sec.) | seed-seed | 0.438 / 0.436 / 0.424 | 0.559 / 0.526 / 0.497 | 0.749 / 0.613 / 0.533 | -0.006 / -0.026 / -0.079 | 0/0/0 |
| pythia410m_s1__pythia410m_s4 (sec.) | seed-seed | 0.336 / 0.332 / 0.354 | 0.410 / 0.382 / 0.383 | 0.642 / 0.465 / 0.386 | 0.021 / -0.018 / -0.118 | 0/0/- |
| pythia410m_s1__pythia410m_s5 (sec.) | seed-seed | 0.583 / 0.580 / 0.572 | 0.688 / 0.659 / 0.636 | 0.828 / 0.729 / 0.668 | -0.004 / -0.017 / -0.050 | 0/0/0 |
| pythia410m_s1__pythia410m_s6 (sec.) | seed-seed | 0.432 / 0.446 / 0.481 | 0.535 / 0.514 / 0.521 | 0.666 / 0.553 / 0.525 | 0.028 / -0.005 / -0.052 | 0/0/0 |
| pythia410m_s1__pythia410m_s7 (sec.) | seed-seed | 0.469 / 0.470 / 0.509 | 0.550 / 0.531 / 0.541 | 0.678 / 0.568 / 0.546 | 0.021 / -0.003 / -0.045 | 0/0/0 |
| pythia410m_s1__pythia410m_s8 (sec.) | seed-seed | 0.474 / 0.465 / 0.494 | 0.549 / 0.526 / 0.532 | 0.675 / 0.566 / 0.539 | 0.012 / -0.005 / -0.048 | 0/0/0 |
| pythia410m_s1__pythia410m_s9 (sec.) | seed-seed | 0.458 / 0.481 / 0.512 | 0.560 / 0.538 / 0.553 | 0.686 / 0.576 / 0.554 | 0.026 / -0.002 / -0.044 | 0/0/0 |
| pythia410m_s2__pythia410m_s3 (sec.) | seed-seed | 0.376 / 0.365 / 0.380 | 0.481 / 0.443 / 0.434 | 0.647 / 0.506 / 0.451 | 0.013 / -0.022 / -0.080 | 0/0/0 |
| pythia410m_s2__pythia410m_s4 (sec.) | seed-seed | 0.364 / 0.369 / 0.395 | 0.459 / 0.433 / 0.426 | 0.720 / 0.512 / 0.436 | 0.028 / -0.017 / -0.112 | 0/0/- |
| pythia410m_s2__pythia410m_s5 (sec.) | seed-seed | 0.471 / 0.467 / 0.496 | 0.560 / 0.536 / 0.544 | 0.677 / 0.575 / 0.551 | 0.016 / -0.005 / -0.044 | 0/0/0 |
| pythia410m_s2__pythia410m_s6 (sec.) | seed-seed | 0.569 / 0.589 / 0.595 | 0.713 / 0.671 / 0.643 | 0.872 / 0.766 / 0.679 | 0.010 / -0.021 / -0.061 | 0/0/0 |
| pythia410m_s2__pythia410m_s7 (sec.) | seed-seed | 0.622 / 0.616 / 0.618 | 0.727 / 0.692 / 0.670 | 0.888 / 0.783 / 0.697 | -0.003 / -0.018 / -0.058 | 0/0/0 |
| pythia410m_s2__pythia410m_s8 (sec.) | seed-seed | 0.609 / 0.607 / 0.597 | 0.727 / 0.685 / 0.656 | 0.884 / 0.779 / 0.691 | -0.002 / -0.023 / -0.060 | 0/0/0 |
| pythia410m_s2__pythia410m_s9 (sec.) | seed-seed | 0.601 / 0.614 / 0.602 | 0.732 / 0.696 / 0.664 | 0.889 / 0.787 / 0.701 | 0.002 / -0.021 / -0.058 | 0/0/0 |
| pythia410m_s3__pythia410m_s4 | seed-seed | 0.344 / 0.307 / 0.317 | 0.401 / 0.360 / 0.348 | 0.633 / 0.453 / 0.363 | -0.008 / -0.033 / -0.130 | 0/0/- |
| pythia410m_s3__pythia410m_s5 (sec.) | seed-seed | 0.482 / 0.482 / 0.467 | 0.592 / 0.566 / 0.541 | 0.788 / 0.653 / 0.573 | -0.004 / -0.022 / -0.074 | 0/0/0 |
| pythia410m_s3__pythia410m_s6 (sec.) | seed-seed | 0.377 / 0.381 / 0.398 | 0.488 / 0.453 / 0.449 | 0.655 / 0.517 / 0.459 | 0.014 / -0.020 / -0.080 | 0/0/0 |
| pythia410m_s3__pythia410m_s7 (sec.) | seed-seed | 0.381 / 0.363 / 0.384 | 0.473 / 0.441 / 0.433 | 0.647 / 0.504 / 0.448 | 0.010 / -0.021 / -0.081 | 0/0/0 |
| pythia410m_s3__pythia410m_s8 (sec.) | seed-seed | 0.374 / 0.388 / 0.392 | 0.497 / 0.458 / 0.450 | 0.657 / 0.522 / 0.466 | 0.013 / -0.023 / -0.077 | 0/0/0 |
| pythia410m_s3__pythia410m_s9 (sec.) | seed-seed | 0.390 / 0.377 / 0.403 | 0.492 / 0.461 / 0.454 | 0.657 / 0.517 / 0.467 | 0.014 / -0.018 / -0.075 | 0/0/0 |
| pythia410m_s4__pythia410m_s5 (sec.) | seed-seed | 0.331 / 0.314 / 0.341 | 0.409 / 0.373 / 0.374 | 0.643 / 0.459 / 0.380 | 0.025 / -0.022 / -0.123 | 0/0/- |
| pythia410m_s4__pythia410m_s6 (sec.) | seed-seed | 0.345 / 0.356 / 0.384 | 0.443 / 0.412 / 0.411 | 0.700 / 0.492 / 0.419 | 0.034 / -0.017 / -0.115 | 0/0/- |
| pythia410m_s4__pythia410m_s7 (sec.) | seed-seed | 0.376 / 0.369 / 0.393 | 0.461 / 0.431 / 0.428 | 0.724 / 0.514 / 0.437 | 0.022 / -0.016 / -0.112 | 0/0/- |
| pythia410m_s4__pythia410m_s8 (sec.) | seed-seed | 0.375 / 0.360 / 0.379 | 0.444 / 0.415 / 0.414 | 0.702 / 0.495 / 0.420 | 0.016 / -0.014 / -0.115 | 0/0/- |
| pythia410m_s4__pythia410m_s9 (sec.) | seed-seed | 0.358 / 0.364 / 0.392 | 0.461 / 0.432 / 0.427 | 0.723 / 0.512 / 0.437 | 0.027 / -0.018 / -0.112 | 0/0/- |
| pythia410m_s5__pythia410m_s6 | seed-seed | 0.435 / 0.445 / 0.484 | 0.534 / 0.514 / 0.527 | 0.664 / 0.557 / 0.528 | 0.025 / -0.003 / -0.050 | 0/0/0 |
| pythia410m_s5__pythia410m_s7 (sec.) | seed-seed | 0.495 / 0.482 / 0.505 | 0.568 / 0.542 / 0.548 | 0.682 / 0.578 / 0.554 | 0.011 / -0.008 / -0.045 | 0/0/0 |
| pythia410m_s5__pythia410m_s8 (sec.) | seed-seed | 0.465 / 0.473 / 0.496 | 0.553 / 0.529 / 0.541 | 0.671 / 0.569 / 0.544 | 0.015 / -0.004 / -0.046 | 0/0/0 |
| pythia410m_s5__pythia410m_s9 (sec.) | seed-seed | 0.467 / 0.476 / 0.493 | 0.563 / 0.542 / 0.551 | 0.681 / 0.578 / 0.555 | 0.011 / -0.005 / -0.044 | 0/0/0 |
| pythia410m_s6__pythia410m_s7 (sec.) | seed-seed | 0.593 / 0.602 / 0.595 | 0.706 / 0.681 / 0.655 | 0.877 / 0.770 / 0.684 | -0.000 / -0.019 / -0.060 | 0/0/0 |
| pythia410m_s6__pythia410m_s8 (sec.) | seed-seed | 0.641 / 0.599 / 0.580 | 0.722 / 0.681 / 0.656 | 0.884 / 0.778 / 0.688 | -0.021 / -0.023 / -0.061 | 0/0/0 |
| pythia410m_s6__pythia410m_s9 (sec.) | seed-seed | 0.575 / 0.595 / 0.579 | 0.704 / 0.672 / 0.645 | 0.874 / 0.768 / 0.677 | -0.002 / -0.021 / -0.062 | 0/0/0 |
| pythia410m_s7__pythia410m_s8 | seed-seed | 0.622 / 0.601 / 0.586 | 0.720 / 0.692 / 0.663 | 0.886 / 0.783 / 0.695 | -0.006 / -0.021 / -0.059 | 0/0/0 |
| pythia410m_s7__pythia410m_s9 (sec.) | seed-seed | 0.600 / 0.597 / 0.594 | 0.721 / 0.690 / 0.658 | 0.892 / 0.785 / 0.692 | -0.000 / -0.022 / -0.062 | 0/0/0 |
| pythia410m_s8__pythia410m_s9 (sec.) | seed-seed | 0.612 / 0.618 / 0.585 | 0.727 / 0.693 / 0.662 | 0.889 / 0.787 / 0.698 | -0.012 / -0.023 / -0.059 | 0/0/0 |
| augreg_b__siglip_b | cross-vision | 0.206 / 0.136 / 0.121 | 0.368 / 0.263 / 0.156 | 0.657 / 0.458 / 0.245 | -0.123 / -0.205 / -0.238 | -/-/- |
| clip_b__augreg_b | cross-vision | 0.161 / 0.129 / 0.120 | 0.368 / 0.266 / 0.156 | 0.678 / 0.453 / 0.245 | -0.067 / -0.213 / -0.238 | 0/-/- |
| clip_b__mae_b | cross-vision | 0.103 / 0.072 / 0.074 | 0.252 / 0.160 / 0.085 | 0.571 / 0.324 / 0.145 | -0.075 / -0.256 / -0.325 | 0/-/- |
| clip_b__siglip_b | cross-vision | 0.068 / 0.065 / 0.075 | 0.229 / 0.146 / 0.079 | 0.561 / 0.319 / 0.134 | 0.023 / -0.255 / -0.342 | 0/-/- |
| dinov2_b__augreg_b | cross-vision | 0.163 / 0.117 / 0.104 | 0.259 / 0.210 / 0.133 | 0.523 / 0.341 / 0.200 | -0.091 / -0.164 / -0.224 | 0/-/- |
| dinov2_b__clip_b | cross-vision | 0.090 / 0.077 / 0.081 | 0.218 / 0.160 / 0.092 | 0.489 / 0.288 / 0.148 | -0.012 / -0.210 / -0.278 | 0/-/- |
| dinov2_b__mae_b | cross-vision | 0.080 / 0.061 / 0.063 | 0.191 / 0.129 / 0.069 | 0.493 / 0.264 / 0.117 | -0.039 / -0.238 / -0.341 | 0/-/- |
| dinov2_b__siglip_b | cross-vision | 0.098 / 0.085 / 0.076 | 0.234 / 0.167 / 0.092 | 0.513 / 0.312 / 0.152 | -0.078 / -0.226 / -0.287 | 0/-/- |
| mae_b__augreg_b | cross-vision | 0.150 / 0.112 / 0.096 | 0.303 / 0.214 / 0.125 | 0.608 / 0.386 / 0.198 | -0.106 / -0.210 / -0.267 | -/-/- |
| mae_b__siglip_b | cross-vision | 0.173 / 0.111 / 0.098 | 0.324 / 0.228 / 0.132 | 0.670 / 0.429 / 0.210 | -0.120 / -0.214 / -0.278 | -/-/- |
| augreg_b__bloomz_1b7 | cross-modal | 0.031 / 0.009 / 0.005 | 0.131 / 0.060 / 0.012 | 0.460 / 0.201 / 0.047 | -0.468 / -0.571 / -0.539 | -/-/- |
| augreg_b__openllama_3b | cross-modal | 0.039 / 0.012 / 0.004 | 0.116 / 0.052 / 0.011 | 0.441 / 0.181 / 0.040 | -0.539 / -0.582 / -0.568 | -/-/- |
| augreg_b__pythia410m_s1 | cross-modal | 0.026 / 0.013 / 0.005 | 0.106 / 0.048 / 0.010 | 0.446 / 0.180 / 0.037 | -0.478 / -0.573 / -0.589 | -/-/- |
| clip_b__bloomz_1b7 | cross-modal | 0.023 / 0.010 / 0.004 | 0.087 / 0.039 / 0.008 | 0.433 / 0.160 / 0.031 | -0.494 / -0.554 / -0.627 | -/-/- |
| clip_b__openllama_3b | cross-modal | 0.021 / 0.009 / 0.002 | 0.078 / 0.035 / 0.008 | 0.424 / 0.147 / 0.027 | -0.538 / -0.546 / -0.652 | -/-/- |
| clip_b__pythia410m_s1 | cross-modal | 0.018 / 0.007 / 0.003 | 0.076 / 0.033 / 0.007 | 0.428 / 0.146 / 0.026 | -0.496 / -0.550 / -0.668 | -/-/- |
| dinov2_b__bloomz_1b7 | cross-modal | 0.030 / 0.010 / 0.002 | 0.100 / 0.050 / 0.009 | 0.428 / 0.171 / 0.039 | -0.592 / -0.569 / -0.567 | -/-/- |
| dinov2_b__openllama_3b | cross-modal | 0.029 / 0.010 / 0.000 | 0.091 / 0.043 / 0.007 | 0.417 / 0.159 / 0.033 | -0.723 / -0.606 / -0.598 | -/-/- |
| dinov2_b__pythia410m_s1 | cross-modal | 0.022 / 0.010 / 0.001 | 0.089 / 0.041 / 0.008 | 0.420 / 0.158 / 0.031 | -0.579 / -0.592 / -0.614 | -/-/- |
| mae_b__bloomz_1b7 | cross-modal | 0.022 / 0.011 / 0.004 | 0.089 / 0.039 / 0.006 | 0.432 / 0.161 / 0.030 | -0.405 / -0.630 / -0.631 | -/-/- |
| mae_b__openllama_3b | cross-modal | 0.013 / 0.010 / 0.001 | 0.081 / 0.036 / 0.005 | 0.421 / 0.147 / 0.026 | -0.516 / -0.663 / -0.661 | -/-/- |
| mae_b__pythia410m_s1 | cross-modal | 0.007 / 0.008 / 0.002 | 0.077 / 0.033 / 0.005 | 0.421 / 0.147 / 0.025 | -0.458 / -0.640 / -0.673 | -/-/- |
| siglip_b__bloomz_1b7 | cross-modal | 0.053 / 0.023 / 0.006 | 0.179 / 0.098 / 0.021 | 0.496 / 0.255 / 0.079 | -0.534 / -0.513 / -0.428 | -/-/- |
| siglip_b__openllama_3b | cross-modal | 0.049 / 0.020 / 0.004 | 0.158 / 0.083 / 0.019 | 0.468 / 0.226 / 0.065 | -0.577 / -0.512 / -0.459 | -/-/- |
| siglip_b__pythia410m_s1 | cross-modal | 0.035 / 0.016 / 0.004 | 0.144 / 0.074 / 0.018 | 0.471 / 0.221 / 0.060 | -0.522 / -0.509 / -0.484 | -/-/- |

## Configuration `l2` (secondary preprocessing): pre = l2

61 pairs (29 primary); draws per cell [5].

### Fixed-k slope classes, primary pair set (D4.8 slopes; rising = conformal-like, flat = smooth, falling = rough)

| class | k | rising | flat | falling | n | median slope |
|---|---|---|---|---|---|---|
| seed-seed | 1 | 0 | 4 | 0 | 4 | 0.021 |
| seed-seed | 10 | 0 | 4 | 0 | 4 | -0.012 |
| seed-seed | 100 | 0 | 3 | 1 | 4 | -0.054 |
| cross-vision | 1 | 0 | 7 | 3 | 10 | -0.083 |
| cross-vision | 10 | 0 | 0 | 10 | 10 | -0.221 |
| cross-vision | 100 | 0 | 0 | 10 | 10 | -0.283 |
| cross-modal | 1 | 0 | 0 | 15 | 15 | -0.461 |
| cross-modal | 10 | 0 | 0 | 15 | 15 | -0.566 |
| cross-modal | 100 | 0 | 0 | 15 | 15 | -0.619 |
| all | 1 | 0 | 11 | 18 | 29 | -0.327 |
| all | 10 | 0 | 4 | 25 | 29 | -0.504 |
| all | 100 | 0 | 3 | 26 | 29 | -0.462 |

Only smooth class observed: False.

### Fixed-k slope classes, all (36 seed pairs) pair set (D4.8 slopes; rising = conformal-like, flat = smooth, falling = rough)

| class | k | rising | flat | falling | n | median slope |
|---|---|---|---|---|---|---|
| seed-seed | 1 | 0 | 36 | 0 | 36 | 0.012 |
| seed-seed | 10 | 0 | 36 | 0 | 36 | -0.018 |
| seed-seed | 100 | 0 | 28 | 8 | 36 | -0.068 |
| cross-vision | 1 | 0 | 7 | 3 | 10 | -0.083 |
| cross-vision | 10 | 0 | 0 | 10 | 10 | -0.221 |
| cross-vision | 100 | 0 | 0 | 10 | 10 | -0.283 |
| cross-modal | 1 | 0 | 0 | 15 | 15 | -0.461 |
| cross-modal | 10 | 0 | 0 | 15 | 15 | -0.566 |
| cross-modal | 100 | 0 | 0 | 15 | 15 | -0.619 |
| all | 1 | 0 | 43 | 18 | 61 | -0.013 |
| all | 10 | 0 | 36 | 25 | 61 | -0.025 |
| all | 100 | 0 | 28 | 33 | 61 | -0.123 |

Only smooth class observed: False.

Slope classes, primary pairs, all H2 items of each prefix (draft quantity obs_allH2): seed-seed k=1: 0/4/0 (median 0.024); seed-seed k=10: 0/4/0 (median -0.010); seed-seed k=100: 0/3/1 (median -0.052); cross-vision k=1: 0/7/3 (median -0.080); cross-vision k=10: 0/0/10 (median -0.224); cross-vision k=100: 0/0/10 (median -0.282); cross-modal k=1: 0/0/15 (median -0.519); cross-modal k=10: 0/0/15 (median -0.590); cross-modal k=100: 0/0/15 (median -0.624)

Slope classes, primary pairs, anchors without an exact k-boundary tie (O11): seed-seed k=1: 0/4/0 (median 0.020); seed-seed k=10: 0/4/0 (median -0.012); seed-seed k=100: 0/3/1 (median -0.054); cross-vision k=1: 0/7/3 (median -0.083); cross-vision k=10: 0/0/10 (median -0.221); cross-vision k=100: 0/0/10 (median -0.283); cross-modal k=1: 0/0/15 (median -0.459); cross-modal k=10: 0/0/15 (median -0.569); cross-modal k=100: 0/0/15 (median -0.619)

Slope classes, primary pairs, EXPLORATORY, not in the plan: chance-corrected (s - c)/(1 - c): seed-seed k=1: 0/4/0 (median 0.022); seed-seed k=10: 0/4/0 (median -0.002); seed-seed k=100: 0/4/0 (median 0.013); cross-vision k=1: 0/8/2 (median -0.078); cross-vision k=10: 0/0/10 (median -0.200); cross-vision k=100: 0/3/7 (median -0.157); cross-modal k=1: 0/0/15 (median -0.366); cross-modal k=10: 0/0/15 (median -0.474); cross-modal k=100: 0/1/14 (median -0.185)

### Descriptive accuracy, primary pair set (no thresholds, no pass/fail; plan V5)

| predictor | class | pairs | median abs log(X/obs) (n cells) | MSLE non-flat (units/cells) | slope sign agreement |
|---|---|---|---|---|---|
| N0 | pooled | 29/29 | 0.399 (463) | 0.661 (69/414) | 11/69 = 0.159 |
| N0 | seed-seed | 4/4 | 0.036 (69) | 0.035 (1/6) | 0/1 = 0.000 |
| N0 | cross-vision | 10/10 | 0.235 (167) | 0.161 (23/138) | 0/23 = 0.000 |
| N0 | cross-modal | 15/15 | 0.821 (227) | 0.931 (45/270) | 11/45 = 0.244 |
| N1 | pooled | 29/29 | 1.636 (463) | 3.058 (69/414) | 69/69 = 1.000 |
| N1 | seed-seed | 4/4 | 3.104 (69) | 4.042 (1/6) | 1/1 = 1.000 |
| N1 | cross-vision | 10/10 | 2.341 (167) | 5.685 (23/138) | 23/23 = 1.000 |
| N1 | cross-modal | 15/15 | 1.014 (227) | 1.694 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB (unvalidated) | pooled | 29/29 | 0.363 (463) | 1.368 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB (unvalidated) | seed-seed | 4/4 | 0.025 (69) | 0.011 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB (unvalidated) | cross-vision | 10/10 | 0.160 (167) | 0.033 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB (unvalidated) | cross-modal | 15/15 | 1.274 (227) | 2.081 (45/270) | 45/45 = 1.000 |
| N1|mu | pooled | 29/29 | 1.636 (463) | 3.043 (69/414) | 69/69 = 1.000 |
| N1|mu | seed-seed | 4/4 | 3.104 (69) | 4.042 (1/6) | 1/1 = 1.000 |
| N1|mu | cross-vision | 10/10 | 2.339 (167) | 5.645 (23/138) | 23/23 = 1.000 |
| N1|mu | cross-modal | 15/15 | 1.014 (227) | 1.690 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB|mu | pooled | 29/29 | 0.373 (463) | 1.411 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB|mu | seed-seed | 4/4 | 0.046 (69) | 0.011 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB|mu | cross-vision | 10/10 | 0.171 (167) | 0.034 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB|mu | cross-modal | 15/15 | 1.283 (227) | 2.146 (45/270) | 45/45 = 1.000 |
| P2 (v1.0, expl.) | pooled | 29/29 | inf (463) | 66.960 (69/414) | 68/69 = 0.986 |
| P2 (v1.0, expl.) | seed-seed | 4/4 | inf (69) | 144.256 (1/6) | 1/1 = 1.000 |
| P2 (v1.0, expl.) | cross-vision | 10/10 | inf (167) | 84.824 (23/138) | 22/23 = 0.957 |
| P2 (v1.0, expl.) | cross-modal | 15/15 | inf (227) | 56.112 (45/270) | 45/45 = 1.000 |
| N2 (v1.0, expl.) | pooled | 29/29 | 1.206 (463) | 1.710 (69/414) | 69/69 = 1.000 |
| N2 (v1.0, expl.) | seed-seed | 4/4 | 1.178 (69) | 0.450 (1/6) | 1/1 = 1.000 |
| N2 (v1.0, expl.) | cross-vision | 10/10 | 0.873 (167) | 1.304 (23/138) | 23/23 = 1.000 |
| N2 (v1.0, expl.) | cross-modal | 15/15 | 1.368 (227) | 1.946 (45/270) | 45/45 = 1.000 |
| N0H1 (v1.0 N0) | pooled | 29/29 | 0.385 (463) | 0.701 (69/414) | 7/69 = 0.101 |
| N0H1 (v1.0 N0) | seed-seed | 4/4 | 0.037 (69) | 0.035 (1/6) | 0/1 = 0.000 |
| N0H1 (v1.0 N0) | cross-vision | 10/10 | 0.249 (167) | 0.161 (23/138) | 0/23 = 0.000 |
| N0H1 (v1.0 N0) | cross-modal | 15/15 | 0.821 (227) | 0.992 (45/270) | 7/45 = 0.156 |
| iii-LSlog-EB A->B only | pooled | 29/29 | 0.501 (463) | 1.201 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB A->B only | seed-seed | 4/4 | 0.155 (69) | 0.007 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB A->B only | cross-vision | 10/10 | 0.267 (167) | 0.089 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB A->B only | cross-modal | 15/15 | 1.317 (227) | 1.795 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB B->A only | pooled | 29/29 | 0.296 (463) | 1.519 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB B->A only | seed-seed | 4/4 | 0.149 (69) | 0.040 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB B->A only | cross-vision | 10/10 | 0.112 (167) | 0.017 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB B->A only | cross-modal | 15/15 | 1.235 (227) | 2.320 (45/270) | 45/45 = 1.000 |
| N1 A->B only | pooled | 29/29 | 1.648 (463) | 3.208 (69/414) | 69/69 = 1.000 |
| N1 A->B only | seed-seed | 4/4 | 3.005 (69) | 3.746 (1/6) | 1/1 = 1.000 |
| N1 A->B only | cross-vision | 10/10 | 2.515 (167) | 6.243 (23/138) | 23/23 = 1.000 |
| N1 A->B only | cross-modal | 15/15 | 1.005 (227) | 1.645 (45/270) | 45/45 = 1.000 |
| N1 B->A only | pooled | 29/29 | 1.596 (463) | 2.971 (69/414) | 69/69 = 1.000 |
| N1 B->A only | seed-seed | 4/4 | 3.214 (69) | 4.382 (1/6) | 1/1 = 1.000 |
| N1 B->A only | cross-vision | 10/10 | 2.244 (167) | 5.309 (23/138) | 23/23 = 1.000 |
| N1 B->A only | cross-modal | 15/15 | 1.025 (227) | 1.745 (45/270) | 45/45 = 1.000 |

### Descriptive accuracy, all (36 seed pairs) pair set (no thresholds, no pass/fail; plan V5)

| predictor | class | pairs | median abs log(X/obs) (n cells) | MSLE non-flat (units/cells) | slope sign agreement |
|---|---|---|---|---|---|
| N0 | pooled | 61/61 | 0.079 (1019) | 0.604 (76/456) | 11/76 = 0.145 |
| N0 | seed-seed | 36/36 | 0.038 (625) | 0.040 (8/48) | 0/8 = 0.000 |
| N0 | cross-vision | 10/10 | 0.235 (167) | 0.161 (23/138) | 0/23 = 0.000 |
| N0 | cross-modal | 15/15 | 0.821 (227) | 0.931 (45/270) | 11/45 = 0.244 |
| N1 | pooled | 61/61 | 2.058 (1019) | 3.058 (76/456) | 76/76 = 1.000 |
| N1 | seed-seed | 36/36 | 2.437 (625) | 3.174 (8/48) | 8/8 = 1.000 |
| N1 | cross-vision | 10/10 | 2.341 (167) | 5.685 (23/138) | 23/23 = 1.000 |
| N1 | cross-modal | 15/15 | 1.014 (227) | 1.694 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB (unvalidated) | pooled | 29/61 | 0.363 (463) | 1.368 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB (unvalidated) | seed-seed | 4/36 | 0.025 (69) | 0.011 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB (unvalidated) | cross-vision | 10/10 | 0.160 (167) | 0.033 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB (unvalidated) | cross-modal | 15/15 | 1.274 (227) | 2.081 (45/270) | 45/45 = 1.000 |
| N1|mu | pooled | 61/61 | 2.056 (1019) | 3.043 (76/456) | 76/76 = 1.000 |
| N1|mu | seed-seed | 36/36 | 2.423 (625) | 3.174 (8/48) | 8/8 = 1.000 |
| N1|mu | cross-vision | 10/10 | 2.339 (167) | 5.645 (23/138) | 23/23 = 1.000 |
| N1|mu | cross-modal | 15/15 | 1.014 (227) | 1.690 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB|mu | pooled | 29/61 | 0.373 (463) | 1.411 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB|mu | seed-seed | 4/36 | 0.046 (69) | 0.011 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB|mu | cross-vision | 10/10 | 0.171 (167) | 0.034 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB|mu | cross-modal | 15/15 | 1.283 (227) | 2.146 (45/270) | 45/45 = 1.000 |
| P2 (v1.0, expl.) | pooled | 61/61 | inf (1019) | 73.408 (76/456) | 75/76 = 0.987 |
| P2 (v1.0, expl.) | seed-seed | 36/36 | inf (625) | 137.878 (8/48) | 8/8 = 1.000 |
| P2 (v1.0, expl.) | cross-vision | 10/10 | inf (167) | 84.824 (23/138) | 22/23 = 0.957 |
| P2 (v1.0, expl.) | cross-modal | 15/15 | inf (227) | 56.112 (45/270) | 45/45 = 1.000 |
| N2 (v1.0, expl.) | pooled | 61/61 | 0.967 (1019) | 1.565 (76/456) | 76/76 = 1.000 |
| N2 (v1.0, expl.) | seed-seed | 36/36 | 0.399 (625) | 0.174 (8/48) | 8/8 = 1.000 |
| N2 (v1.0, expl.) | cross-vision | 10/10 | 0.873 (167) | 1.304 (23/138) | 23/23 = 1.000 |
| N2 (v1.0, expl.) | cross-modal | 15/15 | 1.368 (227) | 1.946 (45/270) | 45/45 = 1.000 |
| N0H1 (v1.0 N0) | pooled | 61/61 | 0.076 (1019) | 0.640 (76/456) | 7/76 = 0.092 |
| N0H1 (v1.0 N0) | seed-seed | 36/36 | 0.036 (625) | 0.041 (8/48) | 0/8 = 0.000 |
| N0H1 (v1.0 N0) | cross-vision | 10/10 | 0.249 (167) | 0.161 (23/138) | 0/23 = 0.000 |
| N0H1 (v1.0 N0) | cross-modal | 15/15 | 0.821 (227) | 0.992 (45/270) | 7/45 = 0.156 |
| iii-LSlog-EB A->B only | pooled | 29/61 | 0.501 (463) | 1.201 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB A->B only | seed-seed | 4/36 | 0.155 (69) | 0.007 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB A->B only | cross-vision | 10/10 | 0.267 (167) | 0.089 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB A->B only | cross-modal | 15/15 | 1.317 (227) | 1.795 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB B->A only | pooled | 29/61 | 0.296 (463) | 1.519 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB B->A only | seed-seed | 4/36 | 0.149 (69) | 0.040 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB B->A only | cross-vision | 10/10 | 0.112 (167) | 0.017 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB B->A only | cross-modal | 15/15 | 1.235 (227) | 2.320 (45/270) | 45/45 = 1.000 |
| N1 A->B only | pooled | 61/61 | 2.025 (1019) | 3.179 (76/456) | 76/76 = 1.000 |
| N1 A->B only | seed-seed | 36/36 | 2.341 (625) | 3.001 (8/48) | 8/8 = 1.000 |
| N1 A->B only | cross-vision | 10/10 | 2.515 (167) | 6.243 (23/138) | 23/23 = 1.000 |
| N1 A->B only | cross-modal | 15/15 | 1.005 (227) | 1.645 (45/270) | 45/45 = 1.000 |
| N1 B->A only | pooled | 61/61 | 2.072 (1019) | 2.997 (76/456) | 76/76 = 1.000 |
| N1 B->A only | seed-seed | 36/36 | 2.543 (625) | 3.391 (8/48) | 8/8 = 1.000 |
| N1 B->A only | cross-vision | 10/10 | 2.244 (167) | 5.309 (23/138) | 23/23 = 1.000 |
| N1 B->A only | cross-modal | 15/15 | 1.025 (227) | 1.745 (45/270) | 45/45 = 1.000 |

### Secondary (O10, descriptive), primary pair set

- k = 1: median fixed-k slope seed-seed 0.021, cross-vision -0.083, cross-modal -0.461; seed >= vision >= modal: True
- k = 10: median fixed-k slope seed-seed -0.012, cross-vision -0.221, cross-modal -0.566; seed >= vision >= modal: True
- k = 100: median fixed-k slope seed-seed -0.054, cross-vision -0.283, cross-modal -0.619; seed >= vision >= modal: True

| class | median e_k | median e_m | median e_k + e_m | median fit rmse | n |
|---|---|---|---|---|---|
| cross-modal | 0.610 | -0.545 | 0.040 | 0.172 | 15 |
| cross-vision | 0.253 | -0.203 | 0.061 | 0.122 | 10 |
| seed-seed | 0.049 | -0.023 | 0.033 | 0.042 | 4 |

corr(e_k, -e_m) across pairs = 0.990

### Secondary (O10, descriptive), all (36 seed pairs) pair set

- k = 1: median fixed-k slope seed-seed 0.012, cross-vision -0.083, cross-modal -0.461; seed >= vision >= modal: True
- k = 10: median fixed-k slope seed-seed -0.018, cross-vision -0.221, cross-modal -0.566; seed >= vision >= modal: True
- k = 100: median fixed-k slope seed-seed -0.068, cross-vision -0.283, cross-modal -0.619; seed >= vision >= modal: True

| class | median e_k | median e_m | median e_k + e_m | median fit rmse | n |
|---|---|---|---|---|---|
| cross-modal | 0.610 | -0.545 | 0.040 | 0.172 | 15 |
| cross-vision | 0.253 | -0.203 | 0.061 | 0.122 | 10 |
| seed-seed | 0.055 | -0.032 | 0.028 | 0.040 | 36 |

corr(e_k, -e_m) across pairs = 0.995

## Configuration `cliplayer` (sensitivity: clip scope (V6)): pre = prh_clip+l2, clip = pool-layer

29 pairs (29 primary); draws per cell [5].

### Fixed-k slope classes, primary pair set (D4.8 slopes; rising = conformal-like, flat = smooth, falling = rough)

| class | k | rising | flat | falling | n | median slope |
|---|---|---|---|---|---|---|
| seed-seed | 1 | 0 | 4 | 0 | 4 | 0.005 |
| seed-seed | 10 | 0 | 4 | 0 | 4 | -0.015 |
| seed-seed | 100 | 0 | 3 | 1 | 4 | -0.055 |
| cross-vision | 1 | 0 | 9 | 1 | 10 | -0.067 |
| cross-vision | 10 | 0 | 0 | 10 | 10 | -0.218 |
| cross-vision | 100 | 0 | 0 | 10 | 10 | -0.284 |
| cross-modal | 1 | 0 | 0 | 15 | 15 | -0.523 |
| cross-modal | 10 | 0 | 0 | 15 | 15 | -0.574 |
| cross-modal | 100 | 0 | 0 | 15 | 15 | -0.594 |
| all | 1 | 0 | 13 | 16 | 29 | -0.387 |
| all | 10 | 0 | 4 | 25 | 29 | -0.499 |
| all | 100 | 0 | 3 | 26 | 29 | -0.419 |

Only smooth class observed: False.

Slope classes, primary pairs, all H2 items of each prefix (draft quantity obs_allH2): seed-seed k=1: 0/4/0 (median 0.010); seed-seed k=10: 0/4/0 (median -0.015); seed-seed k=100: 0/3/1 (median -0.053); cross-vision k=1: 0/8/2 (median -0.076); cross-vision k=10: 0/0/10 (median -0.217); cross-vision k=100: 0/0/10 (median -0.286); cross-modal k=1: 0/0/15 (median -0.524); cross-modal k=10: 0/0/15 (median -0.590); cross-modal k=100: 0/0/15 (median -0.601)

Slope classes, primary pairs, anchors without an exact k-boundary tie (O11): seed-seed k=1: 0/4/0 (median 0.004); seed-seed k=10: 0/4/0 (median -0.015); seed-seed k=100: 0/3/1 (median -0.055); cross-vision k=1: 0/9/1 (median -0.067); cross-vision k=10: 0/0/10 (median -0.218); cross-vision k=100: 0/0/10 (median -0.284); cross-modal k=1: 0/0/15 (median -0.522); cross-modal k=10: 0/0/15 (median -0.574); cross-modal k=100: 0/0/15 (median -0.594)

Slope classes, primary pairs, EXPLORATORY, not in the plan: chance-corrected (s - c)/(1 - c): seed-seed k=1: 0/4/0 (median 0.006); seed-seed k=10: 0/4/0 (median -0.009); seed-seed k=100: 0/4/0 (median -0.008); cross-vision k=1: 0/9/1 (median -0.058); cross-vision k=10: 0/0/10 (median -0.192); cross-vision k=100: 0/3/7 (median -0.142); cross-modal k=1: 0/0/15 (median -0.464); cross-modal k=10: 0/0/15 (median -0.489); cross-modal k=100: 0/0/15 (median -0.224)

### Descriptive accuracy, primary pair set (no thresholds, no pass/fail; plan V5)

| predictor | class | pairs | median abs log(X/obs) (n cells) | MSLE non-flat (units/cells) | slope sign agreement |
|---|---|---|---|---|---|
| N0 | pooled | 29/29 | 0.400 (470) | 0.692 (67/402) | 6/67 = 0.090 |
| N0 | seed-seed | 4/4 | 0.033 (69) | 0.040 (1/6) | 0/1 = 0.000 |
| N0 | cross-vision | 10/10 | 0.232 (166) | 0.171 (21/126) | 0/21 = 0.000 |
| N0 | cross-modal | 15/15 | 0.800 (235) | 0.950 (45/270) | 6/45 = 0.133 |
| N1 | pooled | 29/29 | 1.598 (470) | 2.567 (67/402) | 67/67 = 1.000 |
| N1 | seed-seed | 4/4 | 1.621 (69) | 2.269 (1/6) | 1/1 = 1.000 |
| N1 | cross-vision | 10/10 | 1.997 (166) | 3.223 (21/126) | 21/21 = 1.000 |
| N1 | cross-modal | 15/15 | 1.410 (235) | 2.267 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB (unvalidated) | pooled | 29/29 | 0.330 (470) | 1.195 (67/402) | 67/67 = 1.000 |
| iii-LSlog-EB (unvalidated) | seed-seed | 4/4 | 0.018 (69) | 0.006 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB (unvalidated) | cross-vision | 10/10 | 0.156 (166) | 0.035 (21/126) | 21/21 = 1.000 |
| iii-LSlog-EB (unvalidated) | cross-modal | 15/15 | 1.088 (235) | 1.763 (45/270) | 45/45 = 1.000 |
| N1|mu | pooled | 29/29 | 1.594 (470) | 2.545 (67/402) | 67/67 = 1.000 |
| N1|mu | seed-seed | 4/4 | 1.621 (69) | 2.269 (1/6) | 1/1 = 1.000 |
| N1|mu | cross-vision | 10/10 | 1.969 (166) | 3.211 (21/126) | 21/21 = 1.000 |
| N1|mu | cross-modal | 15/15 | 1.389 (235) | 2.240 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB|mu | pooled | 29/29 | 0.344 (470) | 1.237 (67/402) | 67/67 = 1.000 |
| iii-LSlog-EB|mu | seed-seed | 4/4 | 0.023 (69) | 0.006 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB|mu | cross-vision | 10/10 | 0.163 (166) | 0.035 (21/126) | 21/21 = 1.000 |
| iii-LSlog-EB|mu | cross-modal | 15/15 | 1.098 (235) | 1.826 (45/270) | 45/45 = 1.000 |
| P2 (v1.0, expl.) | pooled | 29/29 | inf (470) | 69.956 (67/402) | 67/67 = 1.000 |
| P2 (v1.0, expl.) | seed-seed | 4/4 | 0.305 (69) | 123.425 (1/6) | 1/1 = 1.000 |
| P2 (v1.0, expl.) | cross-vision | 10/10 | inf (166) | 93.896 (21/126) | 21/21 = 1.000 |
| P2 (v1.0, expl.) | cross-modal | 15/15 | inf (235) | 57.596 (45/270) | 45/45 = 1.000 |
| N2 (v1.0, expl.) | pooled | 29/29 | 1.031 (470) | 1.232 (67/402) | 67/67 = 1.000 |
| N2 (v1.0, expl.) | seed-seed | 4/4 | 0.832 (69) | 0.366 (1/6) | 1/1 = 1.000 |
| N2 (v1.0, expl.) | cross-vision | 10/10 | 0.913 (166) | 0.923 (21/126) | 21/21 = 1.000 |
| N2 (v1.0, expl.) | cross-modal | 15/15 | 1.145 (235) | 1.396 (45/270) | 45/45 = 1.000 |
| N0H1 (v1.0 N0) | pooled | 29/29 | 0.407 (470) | 0.684 (67/402) | 5/67 = 0.075 |
| N0H1 (v1.0 N0) | seed-seed | 4/4 | 0.040 (69) | 0.041 (1/6) | 0/1 = 0.000 |
| N0H1 (v1.0 N0) | cross-vision | 10/10 | 0.252 (166) | 0.172 (21/126) | 0/21 = 0.000 |
| N0H1 (v1.0 N0) | cross-modal | 15/15 | 0.795 (235) | 0.938 (45/270) | 5/45 = 0.111 |
| iii-LSlog-EB A->B only | pooled | 29/29 | 0.470 (470) | 1.124 (67/402) | 67/67 = 1.000 |
| iii-LSlog-EB A->B only | seed-seed | 4/4 | 0.119 (69) | 0.006 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB A->B only | cross-vision | 10/10 | 0.251 (166) | 0.083 (21/126) | 21/21 = 1.000 |
| iii-LSlog-EB A->B only | cross-modal | 15/15 | 1.175 (235) | 1.634 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB B->A only | pooled | 29/29 | 0.245 (470) | 1.264 (67/402) | 66/67 = 0.985 |
| iii-LSlog-EB B->A only | seed-seed | 4/4 | 0.101 (69) | 0.048 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB B->A only | cross-vision | 10/10 | 0.104 (166) | 0.017 (21/126) | 21/21 = 1.000 |
| iii-LSlog-EB B->A only | cross-modal | 15/15 | 1.003 (235) | 1.873 (45/270) | 44/45 = 0.978 |
| N1 A->B only | pooled | 29/29 | 1.731 (470) | 3.043 (67/402) | 67/67 = 1.000 |
| N1 A->B only | seed-seed | 4/4 | 1.541 (69) | 1.492 (1/6) | 1/1 = 1.000 |
| N1 A->B only | cross-vision | 10/10 | 2.284 (166) | 4.029 (21/126) | 21/21 = 1.000 |
| N1 A->B only | cross-modal | 15/15 | 1.565 (235) | 2.617 (45/270) | 45/45 = 1.000 |
| N1 B->A only | pooled | 29/29 | 1.501 (470) | 2.284 (67/402) | 67/67 = 1.000 |
| N1 B->A only | seed-seed | 4/4 | 1.816 (69) | 3.820 (1/6) | 1/1 = 1.000 |
| N1 B->A only | cross-vision | 10/10 | 1.809 (166) | 2.813 (21/126) | 21/21 = 1.000 |
| N1 B->A only | cross-modal | 15/15 | 1.263 (235) | 2.003 (45/270) | 45/45 = 1.000 |

### Secondary (O10, descriptive), primary pair set

- k = 1: median fixed-k slope seed-seed 0.005, cross-vision -0.067, cross-modal -0.523; seed >= vision >= modal: True
- k = 10: median fixed-k slope seed-seed -0.015, cross-vision -0.218, cross-modal -0.574; seed >= vision >= modal: True
- k = 100: median fixed-k slope seed-seed -0.055, cross-vision -0.284, cross-modal -0.594; seed >= vision >= modal: True

| class | median e_k | median e_m | median e_k + e_m | median fit rmse | n |
|---|---|---|---|---|---|
| cross-modal | 0.603 | -0.567 | 0.040 | 0.142 | 15 |
| cross-vision | 0.250 | -0.190 | 0.060 | 0.136 | 10 |
| seed-seed | 0.046 | -0.022 | 0.025 | 0.048 | 4 |

corr(e_k, -e_m) across pairs = 0.994

## Configuration `dedup` (sensitivity: surplus duplicates dropped (D5)): pre = prh_clip+l2, clip = pool-stored, surplus duplicates dropped

29 pairs (29 primary); draws per cell [5].

### Fixed-k slope classes, primary pair set (D4.8 slopes; rising = conformal-like, flat = smooth, falling = rough)

| class | k | rising | flat | falling | n | median slope |
|---|---|---|---|---|---|---|
| seed-seed | 1 | 0 | 4 | 0 | 4 | 0.010 |
| seed-seed | 10 | 0 | 4 | 0 | 4 | -0.014 |
| seed-seed | 100 | 0 | 3 | 1 | 4 | -0.055 |
| cross-vision | 1 | 0 | 7 | 3 | 10 | -0.080 |
| cross-vision | 10 | 0 | 0 | 10 | 10 | -0.214 |
| cross-vision | 100 | 0 | 0 | 10 | 10 | -0.278 |
| cross-modal | 1 | 0 | 0 | 15 | 15 | -0.524 |
| cross-modal | 10 | 0 | 0 | 15 | 15 | -0.567 |
| cross-modal | 100 | 0 | 0 | 15 | 15 | -0.598 |
| all | 1 | 0 | 11 | 18 | 29 | -0.414 |
| all | 10 | 0 | 4 | 25 | 29 | -0.503 |
| all | 100 | 0 | 3 | 26 | 29 | -0.428 |

Only smooth class observed: False.

Slope classes, primary pairs, all H2 items of each prefix (draft quantity obs_allH2): seed-seed k=1: 0/4/0 (median 0.011); seed-seed k=10: 0/4/0 (median -0.014); seed-seed k=100: 0/3/1 (median -0.053); cross-vision k=1: 0/6/4 (median -0.089); cross-vision k=10: 0/0/10 (median -0.217); cross-vision k=100: 0/0/10 (median -0.277); cross-modal k=1: 0/0/15 (median -0.559); cross-modal k=10: 0/0/15 (median -0.589); cross-modal k=100: 0/0/15 (median -0.605)

Slope classes, primary pairs, anchors without an exact k-boundary tie (O11): seed-seed k=1: 0/4/0 (median 0.010); seed-seed k=10: 0/4/0 (median -0.014); seed-seed k=100: 0/3/1 (median -0.055); cross-vision k=1: 0/7/3 (median -0.080); cross-vision k=10: 0/0/10 (median -0.214); cross-vision k=100: 0/0/10 (median -0.278); cross-modal k=1: 0/0/15 (median -0.524); cross-modal k=10: 0/0/15 (median -0.567); cross-modal k=100: 0/0/15 (median -0.598)

Slope classes, primary pairs, EXPLORATORY, not in the plan: chance-corrected (s - c)/(1 - c): seed-seed k=1: 0/4/0 (median 0.011); seed-seed k=10: 0/4/0 (median -0.009); seed-seed k=100: 0/4/0 (median -0.002); cross-vision k=1: 0/8/2 (median -0.072); cross-vision k=10: 0/0/10 (median -0.195); cross-vision k=100: 0/3/7 (median -0.153); cross-modal k=1: 0/0/15 (median -0.490); cross-modal k=10: 0/0/15 (median -0.483); cross-modal k=100: 0/0/15 (median -0.227)

### Descriptive accuracy, primary pair set (no thresholds, no pass/fail; plan V5)

| predictor | class | pairs | median abs log(X/obs) (n cells) | MSLE non-flat (units/cells) | slope sign agreement |
|---|---|---|---|---|---|
| N0 | pooled | 29/29 | 0.396 (469) | 0.656 (69/414) | 7/69 = 0.101 |
| N0 | seed-seed | 4/4 | 0.039 (69) | 0.043 (1/6) | 0/1 = 0.000 |
| N0 | cross-vision | 10/10 | 0.219 (167) | 0.156 (23/138) | 0/23 = 0.000 |
| N0 | cross-modal | 15/15 | 0.809 (233) | 0.925 (45/270) | 7/45 = 0.156 |
| N1 | pooled | 29/29 | 1.602 (469) | 2.918 (69/414) | 69/69 = 1.000 |
| N1 | seed-seed | 4/4 | 2.156 (69) | 2.877 (1/6) | 1/1 = 1.000 |
| N1 | cross-vision | 10/10 | 2.086 (167) | 4.348 (23/138) | 23/23 = 1.000 |
| N1 | cross-modal | 15/15 | 1.387 (233) | 2.188 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB (unvalidated) | pooled | 29/29 | 0.315 (469) | 1.088 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB (unvalidated) | seed-seed | 4/4 | 0.016 (69) | 0.008 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB (unvalidated) | cross-vision | 10/10 | 0.148 (167) | 0.030 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB (unvalidated) | cross-modal | 15/15 | 1.040 (233) | 1.653 (45/270) | 45/45 = 1.000 |
| N1|mu | pooled | 29/29 | 1.586 (469) | 2.887 (69/414) | 69/69 = 1.000 |
| N1|mu | seed-seed | 4/4 | 2.156 (69) | 2.877 (1/6) | 1/1 = 1.000 |
| N1|mu | cross-vision | 10/10 | 2.071 (167) | 4.304 (23/138) | 23/23 = 1.000 |
| N1|mu | cross-modal | 15/15 | 1.365 (233) | 2.163 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB|mu | pooled | 29/29 | 0.320 (469) | 1.129 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB|mu | seed-seed | 4/4 | 0.030 (69) | 0.008 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB|mu | cross-vision | 10/10 | 0.155 (167) | 0.031 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB|mu | cross-modal | 15/15 | 1.053 (233) | 1.715 (45/270) | 45/45 = 1.000 |
| P2 (v1.0, expl.) | pooled | 29/29 | inf (469) | 66.954 (69/414) | 69/69 = 1.000 |
| P2 (v1.0, expl.) | seed-seed | 4/4 | 0.324 (69) | 123.701 (1/6) | 1/1 = 1.000 |
| P2 (v1.0, expl.) | cross-vision | 10/10 | inf (167) | 82.360 (23/138) | 23/23 = 1.000 |
| P2 (v1.0, expl.) | cross-modal | 15/15 | inf (233) | 57.818 (45/270) | 45/45 = 1.000 |
| N2 (v1.0, expl.) | pooled | 29/29 | 1.029 (469) | 1.262 (69/414) | 69/69 = 1.000 |
| N2 (v1.0, expl.) | seed-seed | 4/4 | 0.972 (69) | 0.370 (1/6) | 1/1 = 1.000 |
| N2 (v1.0, expl.) | cross-vision | 10/10 | 0.871 (167) | 0.979 (23/138) | 23/23 = 1.000 |
| N2 (v1.0, expl.) | cross-modal | 15/15 | 1.123 (233) | 1.426 (45/270) | 45/45 = 1.000 |
| N0H1 (v1.0 N0) | pooled | 29/29 | 0.387 (469) | 0.664 (69/414) | 4/69 = 0.058 |
| N0H1 (v1.0 N0) | seed-seed | 4/4 | 0.045 (69) | 0.043 (1/6) | 0/1 = 0.000 |
| N0H1 (v1.0 N0) | cross-vision | 10/10 | 0.243 (167) | 0.156 (23/138) | 0/23 = 0.000 |
| N0H1 (v1.0 N0) | cross-modal | 15/15 | 0.791 (233) | 0.937 (45/270) | 4/45 = 0.089 |
| iii-LSlog-EB A->B only | pooled | 29/29 | 0.470 (469) | 0.998 (69/414) | 69/69 = 1.000 |
| iii-LSlog-EB A->B only | seed-seed | 4/4 | 0.089 (69) | 0.001 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB A->B only | cross-vision | 10/10 | 0.264 (167) | 0.081 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB A->B only | cross-modal | 15/15 | 1.126 (233) | 1.489 (45/270) | 45/45 = 1.000 |
| iii-LSlog-EB B->A only | pooled | 29/29 | 0.246 (469) | 1.175 (69/414) | 68/69 = 0.986 |
| iii-LSlog-EB B->A only | seed-seed | 4/4 | 0.095 (69) | 0.043 (1/6) | 1/1 = 1.000 |
| iii-LSlog-EB B->A only | cross-vision | 10/10 | 0.096 (167) | 0.015 (23/138) | 23/23 = 1.000 |
| iii-LSlog-EB B->A only | cross-modal | 15/15 | 0.951 (233) | 1.793 (45/270) | 44/45 = 0.978 |
| N1 A->B only | pooled | 29/29 | 1.778 (469) | 3.442 (69/414) | 69/69 = 1.000 |
| N1 A->B only | seed-seed | 4/4 | 1.906 (69) | 2.134 (1/6) | 1/1 = 1.000 |
| N1 A->B only | cross-vision | 10/10 | 2.434 (167) | 5.279 (23/138) | 23/23 = 1.000 |
| N1 A->B only | cross-modal | 15/15 | 1.501 (233) | 2.533 (45/270) | 45/45 = 1.000 |
| N1 B->A only | pooled | 29/29 | 1.510 (469) | 2.586 (69/414) | 69/69 = 1.000 |
| N1 B->A only | seed-seed | 4/4 | 2.319 (69) | 4.127 (1/6) | 1/1 = 1.000 |
| N1 B->A only | cross-vision | 10/10 | 1.967 (167) | 3.814 (23/138) | 23/23 = 1.000 |
| N1 B->A only | cross-modal | 15/15 | 1.249 (233) | 1.925 (45/270) | 45/45 = 1.000 |

### Secondary (O10, descriptive), primary pair set

- k = 1: median fixed-k slope seed-seed 0.010, cross-vision -0.080, cross-modal -0.524; seed >= vision >= modal: True
- k = 10: median fixed-k slope seed-seed -0.014, cross-vision -0.214, cross-modal -0.567; seed >= vision >= modal: True
- k = 100: median fixed-k slope seed-seed -0.055, cross-vision -0.278, cross-modal -0.598; seed >= vision >= modal: True

| class | median e_k | median e_m | median e_k + e_m | median fit rmse | n |
|---|---|---|---|---|---|
| cross-modal | 0.580 | -0.565 | 0.020 | 0.142 | 15 |
| cross-vision | 0.251 | -0.194 | 0.058 | 0.121 | 10 |
| seed-seed | 0.048 | -0.019 | 0.027 | 0.051 | 4 |

corr(e_k, -e_m) across pairs = 0.989

## Sensitivity of the observed fixed-k slopes to the configuration (primary pairs, vs `clip`)

| configuration | (pair,k) units | median abs dslope | max abs dslope | mean dslope | slope-class changes |
|---|---|---|---|---|---|
| l2 | 87 | 0.010 | 0.169 | 0.004 | 2 (dinov2_b__augreg_b@k=1: flat->falling; mae_b__augreg_b@k=1: falling->flat) |
| cliplayer | 87 | 0.008 | 0.080 | 0.002 | 2 (augreg_b__siglip_b@k=1: falling->flat; mae_b__augreg_b@k=1: falling->flat) |
| dedup | 87 | 0.001 | 0.017 | 0.001 | 0 |

## Consistency checks

```
{
 "clip/nulls_unit_files": 345,
 "clip/iii_unit_files": 610,
 "clip/obs_nulls_vs_iii_maxdiff": 0.0,
 "clip/N0_nulls_vs_iii_maxdiff": 0.0,
 "clip/N0H1_nulls_vs_iii_maxdiff": 0.0,
 "clip/N1_nulls_vs_iii_maxdiff": 0.0,
 "clip/N1|mu_nulls_vs_iii_maxdiff": 0.0,
 "clip/P2_nulls_vs_iii_maxdiff": 0.0,
 "clip/N2_nulls_vs_iii_maxdiff": 0.0,
 "clip/unit_obs_equal_obsstep_AB": {
  "n": 6405,
  "max_abs_diff": 0.0
 },
 "clip/unit_obs_equal_obsstep_BA": {
  "n": 6405,
  "max_abs_diff": 0.0
 },
 "clip/N0_AB_eq_BA_maxdiff": 0.0,
 "clip/N0H1_AB_eq_BA_maxdiff": 0.0,
 "clip/N0_units_eq_obs_at_mf_maxdiff": 0.0,
 "l2/nulls_unit_files": 610,
 "l2/iii_unit_files": 290,
 "l2/obs_nulls_vs_iii_maxdiff": 0.0,
 "l2/N0_nulls_vs_iii_maxdiff": 0.0,
 "l2/N0H1_nulls_vs_iii_maxdiff": 0.0,
 "l2/N1_nulls_vs_iii_maxdiff": 0.0,
 "l2/N1|mu_nulls_vs_iii_maxdiff": 0.0,
 "l2/P2_nulls_vs_iii_maxdiff": 0.0,
 "l2/N2_nulls_vs_iii_maxdiff": 0.0,
 "l2/unit_obs_equal_obsstep_AB": {
  "n": 6405,
  "max_abs_diff": 0.0
 },
 "l2/unit_obs_equal_obsstep_BA": {
  "n": 6405,
  "max_abs_diff": 0.0
 },
 "l2/N0_AB_eq_BA_maxdiff": 0.0,
 "l2/N0H1_AB_eq_BA_maxdiff": 0.0,
 "l2/N0_units_eq_obs_at_mf_maxdiff": 0.0,
 "cliplayer/nulls_unit_files": 290,
 "cliplayer/iii_unit_files": 290,
 "cliplayer/obs_nulls_vs_iii_maxdiff": 0.0,
 "cliplayer/N0_nulls_vs_iii_maxdiff": 0.0,
 "cliplayer/N0H1_nulls_vs_iii_maxdiff": 0.0,
 "cliplayer/N1_nulls_vs_iii_maxdiff": 0.0,
 "cliplayer/N1|mu_nulls_vs_iii_maxdiff": 0.0,
 "cliplayer/P2_nulls_vs_iii_maxdiff": 0.0,
 "cliplayer/N2_nulls_vs_iii_maxdiff": 0.0,
 "cliplayer/unit_obs_equal_obsstep_AB": {
  "n": 3045,
  "max_abs_diff": 0.0
 },
 "cliplayer/unit_obs_equal_obsstep_BA": {
  "n": 3045,
  "max_abs_diff": 0.0
 },
 "cliplayer/N0_AB_eq_BA_maxdiff": 0.0,
 "cliplayer/N0H1_AB_eq_BA_maxdiff": 0.0,
 "cliplayer/N0_units_eq_obs_at_mf_maxdiff": 0.0,
 "dedup/nulls_unit_files": 290,
 "dedup/iii_unit_files": 290,
 "dedup/obs_nulls_vs_iii_maxdiff": 0.0,
 "dedup/N0_nulls_vs_iii_maxdiff": 0.0,
 "dedup/N0H1_nulls_vs_iii_maxdiff": 0.0,
 "dedup/N1_nulls_vs_iii_maxdiff": 0.0,
 "dedup/N1|mu_nulls_vs_iii_maxdiff": 0.0,
 "dedup/P2_nulls_vs_iii_maxdiff": 0.0,
 "dedup/N2_nulls_vs_iii_maxdiff": 0.0,
 "dedup/unit_obs_equal_obsstep_AB": {
  "n": 3045,
  "max_abs_diff": 0.0
 },
 "dedup/unit_obs_equal_obsstep_BA": {
  "n": 3045,
  "max_abs_diff": 0.0
 },
 "dedup/N0_AB_eq_BA_maxdiff": 0.0,
 "dedup/N0H1_AB_eq_BA_maxdiff": 0.0,
 "dedup/N0_units_eq_obs_at_mf_maxdiff": 0.0
}
```
