# S1.3a — certified linear-AE GL instance: results summary

Generated 2026-10-08T00:38:30Z by `linae/analyze_s13a.py` from `linae/criteria/*.csv`. PREREG sha256 bee0583c2bfd…, SPEC sha256 e781e44f6b80…. Verdict rules: `linae/ANALYSIS_PLAN.md` (frozen before the confirmatory runs) and the v1.1 addendum `linae/ANALYSIS_PLAN_v1.1.md` (crash fix; no verdict rule changed).

## Verdicts

- **P1** — P-GD: kNN, Gs and CKA unreliable at every N; the largest-N mean is inside the 95% CI of the N=inf plateau. → **FALSIFIED**
- **P1_secondary_flow** — (secondary, deviation arm) same as P1 for full-batch GD at lr=0.002 (gradient-flow proxy). → **FALSIFIED**
- **P2** — WD, TIED and small-init: reliable at the largest N (65536), all four criteria. → **CONFIRMED**
- **P2_gauge_criteria_only** — (reading check) P2 restricted to kNN, Gs, CKA → **CONFIRMED**
- **P3** — Projector criterion reliable in every arm at the largest N (65536). → **CONFIRMED**
- **P4** — P-SGD and P-Adam: no prediction (Ziyin & Chuang self-balancing question); outcome reported. → **NO PREDICTION (outcome reported)**

## Decision table (one-sided 95% Clopper–Pearson; reliable if lower > 0.9, unreliable if upper < 0.8)

Cells: passes/pairs (one-sided 95% Clopper–Pearson lower–upper), decision, mean Δ over finite pairs [95% t-CI]. Finite N: 48 disjoint pairs (96 runs). N=inf: all 500 disjoint pairs (1000 inits) are primary; the first 48 pairs are shown on the next row. A pair with a diverged run fails every criterion.

| arm | N | kNN (1−s_N(10) ≤ 0.05) | Gs_rms ≤ 0.02 | 1−CKA ≤ 0.01 | proj ≤ 0.01 | diverged | cert fail |
|---|---|---|---|---|---|---|---|
| P-GD | 256 | 0/48 (CP 0.000–0.061) **unreliable**; 0.274 [0.257, 0.292] | 0/48 (CP 0.000–0.061) **unreliable**; 0.405 [0.368, 0.441] | 0/48 (CP 0.000–0.061) **unreliable**; 0.135 [0.113, 0.157] | 46/48 (CP 0.875–0.993) **inconclusive**; 0.00113 [0.00104, 0.00123] | 2 | 0 |
| P-GD | 4096 | 0/48 (CP 0.000–0.061) **unreliable**; 0.28 [0.262, 0.298] | 0/48 (CP 0.000–0.061) **unreliable**; 0.43 [0.389, 0.47] | 0/48 (CP 0.000–0.061) **unreliable**; 0.15 [0.124, 0.176] | 47/48 (CP 0.905–0.999) **reliable**; 6.75e-05 [6.28e-05, 7.22e-05] | 1 | 0 |
| P-GD | 65536 | 0/48 (CP 0.000–0.061) **unreliable**; 0.279 [0.26, 0.297] | 0/48 (CP 0.000–0.061) **unreliable**; 0.429 [0.389, 0.469] | 0/48 (CP 0.000–0.061) **unreliable**; 0.15 [0.124, 0.175] | 47/48 (CP 0.905–0.999) **reliable**; 4.16e-06 [3.87e-06, 4.45e-06] | 1 | 0 |
| P-GD | inf | 0/500 (CP 0.000–0.006) **unreliable**; 0.275 [0.27, 0.281] | 0/500 (CP 0.000–0.006) **unreliable**; 0.418 [0.407, 0.429] | 0/500 (CP 0.000–0.006) **unreliable**; 0.138 [0.132, 0.145] | 486/500 (CP 0.957–0.983) **reliable**; 2.49e-09 [2.38e-09, 2.6e-09] | 14 | 0 |
| P-GD | inf, first 48 pairs | 0/48 unreliable; 0.278 | 0/48 unreliable; 0.429 | 0/48 unreliable; 0.15 | 47/48 reliable; 2.65e-09 |  |  |
| P-Adam | 256 | 0/48 (CP 0.000–0.061) **unreliable**; 0.118 [0.112, 0.123] | 0/48 (CP 0.000–0.061) **unreliable**; 0.136 [0.127, 0.146] | 7/48 (CP 0.071–0.257) **unreliable**; 0.0184 [0.0158, 0.021] | 48/48 (CP 0.939–1.000) **reliable**; 0.00114 [0.00104, 0.00123] | 0 | 0 |
| P-Adam | 4096 | 0/48 (CP 0.000–0.061) **unreliable**; 0.0914 [0.0859, 0.0969] | 0/48 (CP 0.000–0.061) **unreliable**; 0.116 [0.108, 0.124] | 16/48 (CP 0.222–0.461) **unreliable**; 0.0134 [0.0115, 0.0152] | 48/48 (CP 0.939–1.000) **reliable**; 6.73e-05 [6.27e-05, 7.2e-05] | 0 | 0 |
| P-Adam | 65536 | 1/48 (CP 0.001–0.095) **unreliable**; 0.0888 [0.0829, 0.0947] | 0/48 (CP 0.000–0.061) **unreliable**; 0.114 [0.106, 0.123] | 20/48 (CP 0.296–0.545) **unreliable**; 0.013 [0.0112, 0.0148] | 48/48 (CP 0.939–1.000) **reliable**; 4.13e-06 [3.84e-06, 4.41e-06] | 0 | 0 |
| P-Adam | inf | 13/500 (CP 0.015–0.041) **unreliable**; 0.0921 [0.09, 0.0942] | 0/500 (CP 0.000–0.006) **unreliable**; 0.119 [0.116, 0.122] | 172/500 (CP 0.309–0.381) **unreliable**; 0.0144 [0.0137, 0.0151] | 500/500 (CP 0.994–1.000) **reliable**; 1.23e-09 [1.2e-09, 1.25e-09] | 0 | 0 |
| P-Adam | inf, first 48 pairs | 1/48 unreliable; 0.0894 | 0/48 unreliable; 0.116 | 17/48 unreliable; 0.0132 | 48/48 reliable; 1.2e-09 |  |  |
| P-SGD | 256 | 0/48 (CP 0.000–0.061) **unreliable**; 0.103 [0.0942, 0.112] | 0/48 (CP 0.000–0.061) **unreliable**; 0.117 [0.0986, 0.136] | 26/48 (CP 0.414–0.666) **unreliable**; 0.015 [0.0104, 0.0196] | 48/48 (CP 0.939–1.000) **reliable**; 0.00114 [0.00104, 0.00123] | 0 | 0 |
| P-SGD | 4096 | 14/48 (CP 0.186–0.418) **unreliable**; 0.0744 [0.0639, 0.0849] | 1/48 (CP 0.001–0.095) **unreliable**; 0.0987 [0.0792, 0.118] | 30/48 (CP 0.496–0.742) **unreliable**; 0.012 [0.00779, 0.0162] | 48/48 (CP 0.939–1.000) **reliable**; 6.73e-05 [6.27e-05, 7.2e-05] | 0 | 0 |
| P-SGD | 65536 | 16/48 (CP 0.222–0.461) **unreliable**; 0.0822 [0.0672, 0.0973] | 0/48 (CP 0.000–0.061) **unreliable**; 0.118 [0.0922, 0.144] | 28/48 (CP 0.455–0.704) **unreliable**; 0.0187 [0.0108, 0.0265] | 48/48 (CP 0.939–1.000) **reliable**; 4.13e-06 [3.85e-06, 4.42e-06] | 0 | 0 |
| P-SGD | inf | 164/500 (CP 0.293–0.364) **unreliable**; 0.0734 [0.0696, 0.0772] | 12/500 (CP 0.014–0.039) **unreliable**; 0.102 [0.0954, 0.109] | 322/500 (CP 0.607–0.680) **unreliable**; 0.0133 [0.0114, 0.0151] | 498/500 (CP 0.987–0.999) **reliable**; 1.2e-09 [1.17e-09, 1.23e-09] | 2 | 0 |
| P-SGD | inf, first 48 pairs | 21/48 unreliable; 0.0644 | 2/48 unreliable; 0.0871 | 32/48 unreliable; 0.00899 | 48/48 reliable; 1.15e-09 |  |  |
| WD | 256 | 0/48 (CP 0.000–0.061) **unreliable**; 0.0647 [0.0625, 0.067] | 0/48 (CP 0.000–0.061) **unreliable**; 0.0281 [0.0273, 0.029] | 46/48 (CP 0.875–0.993) **inconclusive**; 0.00073 [0.000687, 0.000774] | 46/48 (CP 0.875–0.993) **inconclusive**; 0.00113 [0.00104, 0.00123] | 2 | 0 |
| WD | 4096 | 47/48 (CP 0.905–0.999) **reliable**; 0.0209 [0.0201, 0.0217] | 47/48 (CP 0.905–0.999) **reliable**; 0.00702 [0.00683, 0.0072] | 47/48 (CP 0.905–0.999) **reliable**; 4.54e-05 [4.3e-05, 4.79e-05] | 47/48 (CP 0.905–0.999) **reliable**; 6.75e-05 [6.28e-05, 7.22e-05] | 1 | 0 |
| WD | 65536 | 47/48 (CP 0.905–0.999) **reliable**; 0.00514 [0.00473, 0.00556] | 47/48 (CP 0.905–0.999) **reliable**; 0.00177 [0.00171, 0.00183] | 47/48 (CP 0.905–0.999) **reliable**; 2.88e-06 [2.71e-06, 3.05e-06] | 47/48 (CP 0.905–0.999) **reliable**; 4.16e-06 [3.87e-06, 4.45e-06] | 1 | 0 |
| WD | inf | 488/500 (CP 0.961–0.986) **reliable**; 0.000191 [0.000173, 0.000208] | 488/500 (CP 0.961–0.986) **reliable**; 0.000127 [0.000124, 0.000131] | 488/500 (CP 0.961–0.986) **reliable**; 1.58e-08 [1.5e-08, 1.66e-08] | 488/500 (CP 0.961–0.986) **reliable**; 7.33e-13 [5.57e-13, 9.09e-13] | 12 | 0 |
| WD | inf, first 48 pairs | 47/48 reliable; 0.000183 | 47/48 reliable; 0.000126 | 47/48 reliable; 1.53e-08 | 47/48 reliable; 6.46e-13 |  |  |
| TIED | 256 | 0/48 (CP 0.000–0.061) **unreliable**; 0.0649 [0.0627, 0.0671] | 0/48 (CP 0.000–0.061) **unreliable**; 0.0279 [0.027, 0.0288] | 48/48 (CP 0.939–1.000) **reliable**; 0.000725 [0.00068, 0.000771] | 48/48 (CP 0.939–1.000) **reliable**; 0.00114 [0.00104, 0.00123] | 0 | 0 |
| TIED | 4096 | 48/48 (CP 0.939–1.000) **reliable**; 0.0198 [0.019, 0.0205] | 48/48 (CP 0.939–1.000) **reliable**; 0.00698 [0.00679, 0.00716] | 48/48 (CP 0.939–1.000) **reliable**; 4.52e-05 [4.27e-05, 4.77e-05] | 48/48 (CP 0.939–1.000) **reliable**; 6.73e-05 [6.27e-05, 7.2e-05] | 0 | 0 |
| TIED | 65536 | 48/48 (CP 0.939–1.000) **reliable**; 0.00455 [0.00425, 0.00485] | 48/48 (CP 0.939–1.000) **reliable**; 0.00176 [0.0017, 0.00181] | 48/48 (CP 0.939–1.000) **reliable**; 2.83e-06 [2.67e-06, 3e-06] | 48/48 (CP 0.939–1.000) **reliable**; 4.14e-06 [3.85e-06, 4.42e-06] | 0 | 0 |
| TIED | inf | 500/500 (CP 0.994–1.000) **reliable**; 0 [0, 0] | 500/500 (CP 0.994–1.000) **reliable**; 2.44e-06 [2.19e-06, 2.69e-06] | 500/500 (CP 0.994–1.000) **reliable**; 1.29e-11 [9.64e-12, 1.62e-11] | 500/500 (CP 0.994–1.000) **reliable**; 3.19e-11 [2.38e-11, 4.01e-11] | 0 | 0 |
| TIED | inf, first 48 pairs | 48/48 reliable; 0 | 48/48 reliable; 2.97e-06 | 48/48 reliable; 1.58e-11 | 48/48 reliable; 3.92e-11 |  |  |
| small-init | 256 | 0/48 (CP 0.000–0.061) **unreliable**; 0.065 [0.0628, 0.0672] | 0/48 (CP 0.000–0.061) **unreliable**; 0.028 [0.0271, 0.0289] | 48/48 (CP 0.939–1.000) **reliable**; 0.000727 [0.000681, 0.000772] | 48/48 (CP 0.939–1.000) **reliable**; 0.00114 [0.00104, 0.00123] | 0 | 0 |
| small-init | 4096 | 48/48 (CP 0.939–1.000) **reliable**; 0.0198 [0.019, 0.0205] | 48/48 (CP 0.939–1.000) **reliable**; 0.00704 [0.00684, 0.00723] | 48/48 (CP 0.939–1.000) **reliable**; 4.61e-05 [4.35e-05, 4.86e-05] | 48/48 (CP 0.939–1.000) **reliable**; 6.73e-05 [6.27e-05, 7.2e-05] | 0 | 0 |
| small-init | 65536 | 48/48 (CP 0.939–1.000) **reliable**; 0.00462 [0.00431, 0.00494] | 48/48 (CP 0.939–1.000) **reliable**; 0.00196 [0.0019, 0.00202] | 48/48 (CP 0.939–1.000) **reliable**; 3.55e-06 [3.33e-06, 3.78e-06] | 48/48 (CP 0.939–1.000) **reliable**; 4.16e-06 [3.86e-06, 4.45e-06] | 0 | 0 |
| small-init | inf | 500/500 (CP 0.994–1.000) **reliable**; 0.000641 [0.000599, 0.000683] | 500/500 (CP 0.994–1.000) **reliable**; 0.00086 [0.000836, 0.000883] | 500/500 (CP 0.994–1.000) **reliable**; 7.56e-07 [7.12e-07, 8e-07] | 500/500 (CP 0.994–1.000) **reliable**; 4.26e-09 [4.12e-09, 4.41e-09] | 0 | 0 |
| small-init | inf, first 48 pairs | 48/48 reliable; 0.000586 | 48/48 reliable; 0.000822 | 48/48 reliable; 6.93e-07 | 48/48 reliable; 4.29e-09 |  |  |
| P-GD-flow (secondary) | 256 | 0/48 (CP 0.000–0.061) **unreliable**; 0.36 [0.343, 0.378] | 0/48 (CP 0.000–0.061) **unreliable**; 0.54 [0.505, 0.575] | 0/48 (CP 0.000–0.061) **unreliable**; 0.204 [0.182, 0.226] | 48/48 (CP 0.939–1.000) **reliable**; 0.00114 [0.00104, 0.00123] | 0 | 0 |
| P-GD-flow (secondary) | 4096 | 0/48 (CP 0.000–0.061) **unreliable**; 0.361 [0.343, 0.378] | 0/48 (CP 0.000–0.061) **unreliable**; 0.549 [0.514, 0.584] | 0/48 (CP 0.000–0.061) **unreliable**; 0.211 [0.188, 0.234] | 48/48 (CP 0.939–1.000) **reliable**; 6.73e-05 [6.27e-05, 7.2e-05] | 0 | 0 |
| P-GD-flow (secondary) | 65536 | 0/48 (CP 0.000–0.061) **unreliable**; 0.36 [0.342, 0.378] | 0/48 (CP 0.000–0.061) **unreliable**; 0.549 [0.514, 0.585] | 0/48 (CP 0.000–0.061) **unreliable**; 0.212 [0.189, 0.235] | 48/48 (CP 0.939–1.000) **reliable**; 4.14e-06 [3.85e-06, 4.43e-06] | 0 | 0 |
| P-GD-flow (secondary) | inf | 0/500 (CP 0.000–0.006) **unreliable**; 0.351 [0.345, 0.358] | 0/500 (CP 0.000–0.006) **unreliable**; 0.535 [0.523, 0.548] | 0/500 (CP 0.000–0.006) **unreliable**; 0.2 [0.192, 0.208] | 500/500 (CP 0.994–1.000) **reliable**; 3.92e-09 [3.79e-09, 4.05e-09] | 0 | 0 |
| P-GD-flow (secondary) | inf, first 48 pairs | 0/48 unreliable; 0.361 | 0/48 unreliable; 0.55 | 0/48 unreliable; 0.212 | 48/48 reliable; 4.09e-09 |  |  |
| WD-flow (secondary) | 256 | 0/48 (CP 0.000–0.061) **unreliable**; 0.0649 [0.0627, 0.0671] | 0/48 (CP 0.000–0.061) **unreliable**; 0.0281 [0.0273, 0.029] | 48/48 (CP 0.939–1.000) **reliable**; 0.000732 [0.000687, 0.000777] | 48/48 (CP 0.939–1.000) **reliable**; 0.00114 [0.00104, 0.00123] | 0 | 0 |
| WD-flow (secondary) | 4096 | 48/48 (CP 0.939–1.000) **reliable**; 0.0208 [0.02, 0.0216] | 48/48 (CP 0.939–1.000) **reliable**; 0.00701 [0.00683, 0.0072] | 48/48 (CP 0.939–1.000) **reliable**; 4.54e-05 [4.3e-05, 4.78e-05] | 48/48 (CP 0.939–1.000) **reliable**; 6.73e-05 [6.27e-05, 7.2e-05] | 0 | 0 |
| WD-flow (secondary) | 65536 | 48/48 (CP 0.939–1.000) **reliable**; 0.00509 [0.0047, 0.00549] | 48/48 (CP 0.939–1.000) **reliable**; 0.00177 [0.00172, 0.00183] | 48/48 (CP 0.939–1.000) **reliable**; 2.88e-06 [2.71e-06, 3.04e-06] | 48/48 (CP 0.939–1.000) **reliable**; 4.14e-06 [3.85e-06, 4.42e-06] | 0 | 0 |
| WD-flow (secondary) | inf | 500/500 (CP 0.994–1.000) **reliable**; 0.000204 [0.000187, 0.000221] | 500/500 (CP 0.994–1.000) **reliable**; 0.000168 [0.000164, 0.000171] | 500/500 (CP 0.994–1.000) **reliable**; 2.67e-08 [2.57e-08, 2.78e-08] | 500/500 (CP 0.994–1.000) **reliable**; 4.69e-13 [3.35e-13, 6.04e-13] | 0 | 0 |
| WD-flow (secondary) | inf, first 48 pairs | 48/48 reliable; 0.000203 | 48/48 reliable; 0.000174 | 48/48 reliable; 2.84e-08 | 48/48 reliable; 3.35e-13 |  |  |

## Run outcomes and divergences (diverged runs are kept; every pair containing one fails every criterion)

| arm | N | runs | certified | uncertified (finite) | diverged (stochastic / GD-polish) | pairs hit | diverged runs (run@step) | code |
|---|---|---|---|---|---|---|---|---|
| P-GD | 256 | 96 | 94 | 0 | 2 (0 / 2) | 2 | 64@50(gd), 88@50(gd) | v1.1 |
| P-GD | 4096 | 96 | 95 | 0 | 1 (0 / 1) | 1 | 88@50(gd) | v1.1 |
| P-GD | 65536 | 96 | 95 | 0 | 1 (0 / 1) | 1 | 88@50(gd) | v1.1 |
| P-GD | inf | 1000 | 986 | 0 | 14 (0 / 14) | 14 | 88@50(gd), 96@50(gd), 116@50(gd), 152@50(gd), 164@50(gd), 272@50(gd), 331@50(gd), 501@50(gd), 518@50(gd), 586@50(gd), 687@50(gd), 694@50(gd), … (+2) | v1.1 |
| P-Adam | 256 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| P-Adam | 4096 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| P-Adam | 65536 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| P-Adam | inf | 1000 | 1000 | 0 | 0 (0 / 0) | 0 |  | v1.0+v1.1 |
| P-SGD | 256 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| P-SGD | 4096 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| P-SGD | 65536 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| P-SGD | inf | 1000 | 998 | 0 | 2 (2 / 0) | 2 | 385@288000, 557@310000 | v1.0+v1.1 |
| WD | 256 | 96 | 94 | 0 | 2 (0 / 2) | 2 | 64@50(gd), 88@50(gd) | v1.1 |
| WD | 4096 | 96 | 95 | 0 | 1 (0 / 1) | 1 | 88@50(gd) | v1.1 |
| WD | 65536 | 96 | 95 | 0 | 1 (0 / 1) | 1 | 88@50(gd) | v1.1 |
| WD | inf | 1000 | 988 | 0 | 12 (0 / 12) | 12 | 88@50(gd), 96@50(gd), 116@50(gd), 152@50(gd), 164@50(gd), 272@50(gd), 331@50(gd), 501@50(gd), 586@50(gd), 687@50(gd), 824@50(gd), 985@50(gd) | v1.1 |
| TIED | 256 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| TIED | 4096 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| TIED | 65536 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| TIED | inf | 1000 | 1000 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| small-init | 256 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| small-init | 4096 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| small-init | 65536 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| small-init | inf | 1000 | 1000 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| P-GD-flow | 256 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| P-GD-flow | 4096 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| P-GD-flow | 65536 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| P-GD-flow | inf | 1000 | 1000 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| WD-flow | 256 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| WD-flow | 4096 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| WD-flow | 65536 | 96 | 96 | 0 | 0 (0 / 0) | 0 |  | v1.1 |
| WD-flow | inf | 1000 | 1000 | 0 | 0 (0 / 0) | 0 |  | v1.1 |

## Imbalance C = EEᵀ − WᵀW and latent anisotropy (spec EEᵀ = spec S)

| arm | N | median ‖C₀‖_F | median ‖C_end‖_F | ‖C_end−C₀‖/‖C₀‖ median (q90) | κ(EEᵀ) q10/q50/q90 | median normalised spec(EEᵀ) | spec vs f(spec C₀) rel.err median |
|---|---|---|---|---|---|---|---|
| P-GD | 256 | 7.17 | 3.88 | 0.595 (0.892) | 2.1/2.91/4.16 | 0.55, 0.752, 1.11, 1.58 | 0.234 |
| P-GD | 4096 | 7.17 | 4.3 | 0.521 (0.9) | 2.13/2.96/4.06 | 0.546, 0.747, 1.08, 1.61 | 0.193 |
| P-GD | 65536 | 7.17 | 4.27 | 0.498 (0.904) | 2.15/2.96/4.21 | 0.553, 0.739, 1.07, 1.62 | 0.192 |
| P-GD | inf | 7.25 | 4.18 | 0.533 (0.905) | 2.12/3.06/4.37 | 0.54, 0.748, 1.06, 1.62 | 0.217 |
| P-Adam | 256 | 7.17 | 19.4 | 2 (2.55) | 1.39/1.58/1.76 | 0.787, 0.924, 1.06, 1.23 | 0.838 |
| P-Adam | 4096 | 7.17 | 19.5 | 1.97 (2.53) | 1.37/1.54/1.7 | 0.797, 0.916, 1.07, 1.22 | 0.838 |
| P-Adam | 65536 | 7.17 | 19.6 | 1.96 (2.51) | 1.37/1.53/1.69 | 0.802, 0.919, 1.07, 1.21 | 0.839 |
| P-Adam | inf | 7.25 | 19.5 | 1.93 (2.67) | 1.38/1.54/1.73 | 0.793, 0.923, 1.06, 1.22 | 0.834 |
| P-SGD | 256 | 7.17 | 21.7 | 2.31 (3.08) | 1.07/1.23/1.68 | 0.921, 0.952, 0.997, 1.13 | 0.863 |
| P-SGD | 4096 | 7.17 | 22.5 | 2.39 (3.39) | 1.08/1.19/1.61 | 0.925, 0.961, 1, 1.1 | 0.868 |
| P-SGD | 65536 | 7.17 | 22.7 | 2.44 (3.23) | 1.07/1.19/1.75 | 0.928, 0.962, 1, 1.11 | 0.87 |
| P-SGD | inf | 7.25 | 22.6 | 2.36 (3.49) | 1.07/1.19/1.61 | 0.93, 0.961, 1, 1.11 | 0.871 |
| WD | 256 | 7.17 | 0.00083 | 1 (1) | 1.03/1.03/1.04 | 0.981, 0.995, 1.01, 1.02 | 0.98 |
| WD | 4096 | 7.17 | 0.000865 | 1 (1) | 1.03/1.03/1.03 | 0.982, 0.995, 1.01, 1.01 | 0.98 |
| WD | 65536 | 7.17 | 0.000919 | 1 (1) | 1.03/1.03/1.03 | 0.982, 0.995, 1.01, 1.01 | 0.978 |
| WD | inf | 7.25 | 0.000905 | 1 (1) | 1.03/1.03/1.03 | 0.982, 0.995, 1.01, 1.01 | 0.984 |
| TIED | 256 | 0 | 0 | nan (nan) | 1/1/1 | 1, 1, 1, 1 | 3.31e-12 |
| TIED | 4096 | 0 | 0 | nan (nan) | 1/1/1 | 1, 1, 1, 1 | 2.34e-12 |
| TIED | 65536 | 0 | 0 | nan (nan) | 1/1/1 | 1, 1, 1, 1 | 2.27e-12 |
| TIED | inf | 0 | 0 | nan (nan) | 1/1/1 | 1, 1, 1, 1 | 1.45e-12 |
| small-init | 256 | 0.00232 | 0.00356 | 1.12 (2.08) | 1/1/1 | 0.999, 1, 1, 1 | 0.000628 |
| small-init | 4096 | 0.00232 | 0.00354 | 1.15 (2.18) | 1/1/1 | 0.999, 1, 1, 1 | 0.000661 |
| small-init | 65536 | 0.00232 | 0.00353 | 1.18 (2.22) | 1/1/1 | 0.999, 1, 1, 1 | 0.00064 |
| small-init | inf | 0.00241 | 0.00357 | 1.19 (2.16) | 1/1/1 | 0.999, 1, 1, 1 | 0.000651 |
| P-GD-flow | 256 | 7.17 | 7.15 | 0.00184 (0.00328) | 3.08/4.6/6.6 | 0.407, 0.661, 1.1, 1.86 | 0.00123 |
| P-GD-flow | 4096 | 7.17 | 7.15 | 0.00174 (0.0031) | 3.08/4.6/6.6 | 0.407, 0.66, 1.1, 1.86 | 0.00119 |
| P-GD-flow | 65536 | 7.17 | 7.15 | 0.00174 (0.00318) | 3.08/4.6/6.6 | 0.407, 0.661, 1.1, 1.86 | 0.00113 |
| P-GD-flow | inf | 7.25 | 7.24 | 0.00182 (0.00334) | 2.98/4.66/7.43 | 0.401, 0.652, 1.07, 1.84 | 0.00101 |
| WD-flow | 256 | 7.17 | 0.00121 | 1 (1) | 1.03/1.03/1.04 | 0.981, 0.995, 1.01, 1.02 | 0.976 |
| WD-flow | 4096 | 7.17 | 0.00122 | 1 (1) | 1.03/1.03/1.03 | 0.982, 0.995, 1.01, 1.01 | 0.977 |
| WD-flow | 65536 | 7.17 | 0.00122 | 1 (1) | 1.03/1.03/1.03 | 0.982, 0.995, 1.01, 1.01 | 0.975 |
| WD-flow | inf | 7.25 | 0.00122 | 1 (1) | 1.03/1.03/1.03 | 0.982, 0.995, 1.01, 1.01 | 0.983 |

## Prediction 1 part (b) detail and diagnostics

- P1 kNN: mean(65536) = 0.2787 (n=47); N=inf mean 0.2752 CI95 [0.2699, 0.2806] (n=486) → inside=True (CONFIRMED). Diagnostics: Welch diff 0.003459 CI95 [-0.01567, 0.02259] p=0.718; paired-CRN diff 0.0008644 CI95 [-0.0005987, 0.002327]; P(literal test passes | equal means) ≈ 0.458
- P1 Gs: mean(65536) = 0.4287 (n=47); N=inf mean 0.4179 CI95 [0.4072, 0.4286] (n=486) → inside=False (FALSIFIED). Diagnostics: Welch diff 0.01075 CI95 [-0.03044, 0.05195] p=0.603; paired-CRN diff 0.0001315 CI95 [-0.002108, 0.002371]; P(literal test passes | equal means) ≈ 0.458
- P1 CKA: mean(65536) = 0.1496 (n=47); N=inf mean 0.1383 CI95 [0.1319, 0.1447] (n=486) → inside=False (FALSIFIED). Diagnostics: Welch diff 0.01125 CI95 [-0.01478, 0.03728] p=0.39; paired-CRN diff -0.0001013 CI95 [-0.001369, 0.001166]; P(literal test passes | equal means) ≈ 0.458
- P1_secondary_flow kNN: mean(65536) = 0.3604 (n=48); N=inf mean 0.3515 CI95 [0.3451, 0.3578] (n=500) → inside=False (FALSIFIED). Diagnostics: Welch diff 0.008955 CI95 [-0.01001, 0.02792] p=0.349; paired-CRN diff -0.0003662 CI95 [-0.0008431, 0.0001107]; P(literal test passes | equal means) ≈ 0.456
- P1_secondary_flow Gs: mean(65536) = 0.5495 (n=48); N=inf mean 0.5354 CI95 [0.5229, 0.5478] (n=500) → inside=False (FALSIFIED). Diagnostics: Welch diff 0.01414 CI95 [-0.0235, 0.05179] p=0.455; paired-CRN diff -0.0001162 CI95 [-0.0005168, 0.0002845]; P(literal test passes | equal means) ≈ 0.456
- P1_secondary_flow CKA: mean(65536) = 0.2119 (n=48); N=inf mean 0.2 CI95 [0.1919, 0.2082] (n=500) → inside=False (FALSIFIED). Diagnostics: Welch diff 0.01193 CI95 [-0.01265, 0.03651] p=0.335; paired-CRN diff -3.899e-05 CI95 [-0.0003941, 0.0003161]; P(literal test passes | equal means) ≈ 0.456

## Stochastic arms: trajectory of median κ(EEᵀ) and ‖C‖_F during the minibatch phase

- P-SGD N=256: κ t=25k:1.68, t=100k:1.3, t=200k:1.18, t=300k:1.21, t=400k:1.26, t=500k:1.23; ‖C‖ 25k:12.8, 100k:20.4, 200k:21.1, 300k:21.3, 400k:21, 500k:21.7; pre-polish κ q50 1.23, pre-polish rel gap median 5.81e-06
- P-SGD N=4096: κ t=25k:1.62, t=100k:1.29, t=200k:1.22, t=300k:1.23, t=400k:1.2, t=500k:1.19; ‖C‖ 25k:12.8, 100k:20.7, 200k:21.7, 300k:21.8, 400k:21.7, 500k:22.5; pre-polish κ q50 1.19, pre-polish rel gap median 5.91e-06
- P-SGD N=65536: κ t=25k:1.63, t=100k:1.3, t=200k:1.18, t=300k:1.18, t=400k:1.2, t=500k:1.19; ‖C‖ 25k:12.8, 100k:20.8, 200k:22.2, 300k:22.1, 400k:22, 500k:22.7; pre-polish κ q50 1.19, pre-polish rel gap median 5.98e-06
- P-SGD N=inf: κ t=25k:1.64, t=100k:1.3, t=200k:1.19, t=300k:1.19, t=400k:1.2, t=500k:1.19; ‖C‖ 25k:12.9, 100k:20.9, 200k:22, 300k:21.8, 400k:21.8, 500k:22.6; pre-polish κ q50 1.19, pre-polish rel gap median 5.8e-06
- P-Adam N=256: κ t=25k:3.78, t=100k:2.14, t=200k:1.74, t=300k:1.64, t=400k:1.58, t=500k:1.58; ‖C‖ 25k:6.85, 100k:9.13, 200k:12.5, 300k:15.9, 400k:19.2, 500k:19.4; pre-polish κ q50 1.58, pre-polish rel gap median 5.08e-06
- P-Adam N=4096: κ t=25k:3.76, t=100k:2.16, t=200k:1.72, t=300k:1.6, t=400k:1.54, t=500k:1.54; ‖C‖ 25k:6.9, 100k:9.19, 200k:12.5, 300k:15.9, 400k:19.3, 500k:19.5; pre-polish κ q50 1.54, pre-polish rel gap median 5.26e-06
- P-Adam N=65536: κ t=25k:3.73, t=100k:2.18, t=200k:1.71, t=300k:1.59, t=400k:1.53, t=500k:1.53; ‖C‖ 25k:6.91, 100k:9.21, 200k:12.5, 300k:16, 400k:19.3, 500k:19.6; pre-polish κ q50 1.53, pre-polish rel gap median 5.6e-06
- P-Adam N=inf: κ t=25k:3.75, t=100k:2.16, t=200k:1.73, t=300k:1.6, t=400k:1.55, t=500k:1.54; ‖C‖ 25k:6.89, 100k:9.13, 200k:12.5, 300k:15.8, 400k:19.2, 500k:19.5; pre-polish κ q50 1.54, pre-polish rel gap median 5.25e-06

## P4 (no prediction): descriptive outcome at N=65536

- P-SGD: kNN/Gs/CKA = unreliable/unreliable/unreliable → gauge criteria unreliable at N=65536 (no self-balancing to the criterion tolerance); median ‖C₀‖ 7.17 → ‖C_end‖ 22.7; κ(EEᵀ) q10/50/90 1.07/1.19/1.75; diverged runs 0
- P-Adam: kNN/Gs/CKA = unreliable/unreliable/unreliable → gauge criteria unreliable at N=65536 (no self-balancing to the criterion tolerance); median ‖C₀‖ 7.17 → ‖C_end‖ 19.6; κ(EEᵀ) q10/50/90 1.37/1.53/1.69; diverged runs 0

## Run integrity (v1.1)

- v1.0 crash reproduction P-Adam|inf|chunk0: 0 run(s) diverged in the stochastic phase ; v1.0 polish → no exception (first 100 polish steps)
- v1.0 crash reproduction P-Adam|inf|chunk6: 0 run(s) diverged in the stochastic phase ; v1.0 polish → no exception (first 100 polish steps)
- v1.0 crash reproduction P-Adam|inf|chunk7: 0 run(s) diverged in the stochastic phase ; v1.0 polish → no exception (first 100 polish steps)
- v1.0 crash reproduction P-SGD|inf|chunk1: 0 run(s) diverged in the stochastic phase ; v1.0 polish → no exception (first 100 polish steps)
- v1.0 crash reproduction P-SGD|inf|chunk3: 1 run(s) diverged in the stochastic phase (run 385 by step 288000) ; v1.0 polish → LinAlgError: SVD did not converge
- Bitwise reproduction of a v1.0 chunk by v1.1 (P-SGD N=inf chunk 0): all_equal=True over 28 stored arrays
- Totals: 10304 runs; diverged 36 ({"P-GD|256": 2, "P-GD|4096": 1, "P-GD|65536": 1, "P-GD|inf": 14, "P-SGD|inf": 2, "WD|256": 2, "WD|4096": 1, "WD|65536": 1, "WD|inf": 12}); finite but uncertified 0; pair metric errors 0; runs by code version {"v1.1": 9429, "v1.0": 875}

## Sensitivity

- Decisions excluding pairs with a diverged run: {"P1_part_a": {"kNN|256": "unreliable", "kNN|4096": "unreliable", "kNN|65536": "unreliable", "kNN|inf": "unreliable", "Gs|256": "unreliable", "Gs|4096": "unreliable", "Gs|65536": "unreliable", "Gs|inf": "unreliable", "CKA|256": "unreliable", "CKA|4096": "unreliable", "CKA|65536": "unreliable", "CKA|inf": "unreliable"}, "P2": {"WD|kNN": "reliable", "WD|Gs": "reliable", "WD|CKA": "reliable", "WD|proj": "reliable", "TIED|kNN": "reliable", "TIED|Gs": "reliable", "TIED|CKA": "reliable", "TIED|proj": "reliable", "small-init|kNN": "reliable", "small-init|Gs": "reliable", "small-init|CKA": "reliable", "small-init|proj": "reliable"}, "P3": {"P-GD": "reliable", "P-Adam": "reliable", "P-SGD": "reliable", "WD": "reliable", "TIED": "reliable", "small-init": "reliable"}}
- P-SGD pre-polish decisions at N=65536: {"kNN": "unreliable", "Gs": "unreliable", "CKA": "unreliable", "proj": "reliable"}
- P-Adam pre-polish decisions at N=65536: {"kNN": "unreliable", "Gs": "unreliable", "CKA": "unreliable", "proj": "reliable"}

## Recovery vs the isotropic population-PCA target (median Δ per run)

- P-GD: N=256: kNN 0.202, Gs 0.277, CKA 0.0631, proj 0.000515; N=4096: kNN 0.199, Gs 0.283, CKA 0.0653, proj 3.2e-05; N=65536: kNN 0.202, Gs 0.276, CKA 0.0656, proj 2.08e-06; N=inf: kNN 0.203, Gs 0.284, CKA 0.0658, proj 1.04e-09
- P-Adam: N=256: kNN 0.098, Gs 0.121, CKA 0.0138, proj 0.000518; N=4096: kNN 0.0854, Gs 0.114, CKA 0.0118, proj 3.2e-05; N=65536: kNN 0.0834, Gs 0.114, CKA 0.0117, proj 2.08e-06; N=inf: kNN 0.0857, Gs 0.116, CKA 0.012, proj 6.07e-10
- P-SGD: N=256: kNN 0.0646, Gs 0.0658, CKA 0.00374, proj 0.000518; N=4096: kNN 0.0445, Gs 0.0464, CKA 0.00198, proj 3.2e-05; N=65536: kNN 0.0418, Gs 0.0505, CKA 0.0023, proj 2.08e-06; N=inf: kNN 0.0406, Gs 0.0484, CKA 0.00205, proj 5.95e-10
- WD: N=256: kNN 0.0477, Gs 0.0218, CKA 0.000421, proj 0.000515; N=4096: kNN 0.018, Gs 0.00961, CKA 7.65e-05, proj 3.2e-05; N=65536: kNN 0.0105, Gs 0.00829, CKA 5.56e-05, proj 2.08e-06; N=inf: kNN 0.00977, Gs 0.00821, CKA 5.42e-05, proj 1.14e-13
- TIED: N=256: kNN 0.0469, Gs 0.0196, CKA 0.000353, proj 0.000518; N=4096: kNN 0.0137, Gs 0.00486, CKA 2.17e-05, proj 3.21e-05; N=65536: kNN 0.00313, Gs 0.00123, CKA 1.39e-06, proj 2.08e-06; N=inf: kNN 0, Gs 6.62e-07, CKA 3.91e-13, proj 9.64e-13
- small-init: N=256: kNN 0.0469, Gs 0.0196, CKA 0.000354, proj 0.000518; N=4096: kNN 0.0137, Gs 0.00492, CKA 2.23e-05, proj 3.2e-05; N=65536: kNN 0.00313, Gs 0.00137, CKA 1.72e-06, proj 2.08e-06; N=inf: kNN 0.000391, Gs 0.000585, CKA 3.16e-07, proj 2.11e-09
- P-GD-flow: N=256: kNN 0.265, Gs 0.372, CKA 0.0998, proj 0.000519; N=4096: kNN 0.264, Gs 0.377, CKA 0.101, proj 3.21e-05; N=65536: kNN 0.263, Gs 0.377, CKA 0.101, proj 2.09e-06; N=inf: kNN 0.265, Gs 0.38, CKA 0.104, proj 1.94e-09
- WD-flow: N=256: kNN 0.048, Gs 0.0218, CKA 0.000421, proj 0.000518; N=4096: kNN 0.018, Gs 0.00963, CKA 7.69e-05, proj 3.21e-05; N=65536: kNN 0.0105, Gs 0.00831, CKA 5.59e-05, proj 2.08e-06; N=inf: kNN 0.00977, Gs 0.00823, CKA 5.44e-05, proj 6.59e-14
