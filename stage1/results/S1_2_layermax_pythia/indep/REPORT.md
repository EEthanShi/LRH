# Pythia axis (S1.2, fourth size axis): layer-max calibration, implementation I

Panel: 7 Pythia models (70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b) x the 17 ARH `val` ViTs on WIT-1024 (119 pairs); pre run: prh_clip+l2, l2 (l2 run: True). K = 500, keys `S1.2-layermax|<lm>|<vit>`. Every number is read from `summary.json` (its `verdicts` block = `verdicts_pyt.json`). Test mode: False; stand-in fixture: False.

**Labels (plan 5.4, 7).** H1 is the confirmatory gatekeeper at alpha = 0.05; H2a/H2b, H3, H4 are tested with Holm at 0.05 only if the gate is open. Everything else is descriptive. l2 and uncensored are secondary.

## prh_clip+l2 (primary)

- Part B: H1 member yes, s = 1, H2 branch H2a, s_late = 1.
- Deductive checks before any category (plan 7): 0 violations (keep identity, sign rules, D3, P-P1).
- **Implementation I's own category (cross-check; the axis category is `forecast_pyt.py` section K): C** (premise None; F reason None); R = +0.929, C = +0.895 on 17 ViTs; knife edge no.
- Modifiers (I): M1 no (), M2 n/a (grid5 D1 +0.011), M3 no, M4 no (), M5 extrapolated: 70m,160m; readings: H2 kept (not tested, gate closed); H3 no comparison changed order; H4 no comparison changed order.

| test | statistic | p | Holm | status |
|---|---|---|---|---|
| H1 (gatekeeper) | D1 = -0.034 on 17 ViTs | 1 (perm 0.3198, mag 1) | n/a | confirmed no |
| H2 | D2 = -0.034 (H2a) | 1 | n/a | tested no, confirmed n/a |
| H3 | Delta = +0.000 (degenerate yes) | 0.0004998 | n/a | tested no, confirmed n/a |
| H4 | Delta = +0.000 (degenerate yes) | 0.0004998 | n/a | tested no, confirmed n/a |

- Gate: closed; Holm order None.
- Premise rules: P-P1 holds yes; P-P2a(i) holds yes (7 of 7 in band; failing ''); P-P2a(ii) holds (307/307, need 277); P-P2a holds; I's P-P2b holds (352/352 decided correct); P-P5a holds.

| per model | band | median c - B-bar | in band | pairs in band |
|---|---|---|---|---|
| 70m | [+0.00162, +0.00322] | +0.00207 | yes | 17/17 |
| 160m | [+0.00146, +0.00306] | +0.00204 | yes | 17/17 |
| 410m | [+0.00141, +0.00301] | +0.00200 | yes | 17/17 |
| 1b | [+0.00134, +0.00294] | +0.00202 | yes | 17/17 |
| 1.4b | [+0.00130, +0.00290] | +0.00204 | yes | 17/17 |
| 2.8b | [+0.00135, +0.00295] | +0.00211 | yes | 17/17 |
| 6.9b | [+0.00127, +0.00287] | +0.00200 | yes | 17/17 |

Trends (layer-max `full`, exact one-sided p over 5040 relabellings; descriptive except through H1-H4):

| metric | Tr raw | Tr s_cal | dTr s_cal (common set) | p (<=) | Tr unc | dTr unc |
|---|---|---|---|---|---|---|
| mKNN | +1.000 | +1.000 | +0.000 | 1 | +1.000 | +0.000 |
| CKA (biased) | +0.929 | +0.895 | -0.034 | 0.3198 | +0.891 | -0.038 |
| CKA (debiased) | +0.901 | +0.901 | +0.000 | 1 | +0.901 | +0.000 |

Robustness (descriptive): five sizes D1 -0.065; grid5 D1 +0.011; rho0 D1 -0.034; leave-one-size-out D1: 70m -0.037, 160m -0.037, 410m -0.054, 1b -0.044, 1.4b -0.050, 2.8b -0.024, 6.9b -0.017.
Uncensored (secondary): D1 -0.038, p 1; I's category C.

## l2 (secondary)

- Part B: H1 member yes, s = 1, H2 branch H2b, s_late = -1.
- Deductive checks before any category (plan 7): 0 violations (keep identity, sign rules, D3, P-P1).
- **Implementation I's own category (cross-check; the axis category is `forecast_pyt.py` section K): A** (premise verified; F reason None); R = +0.681, C = -0.059 on 17 ViTs; knife edge no.
- Modifiers (I): M1 yes (6.9b), M2 no (grid5 D1 -0.855), M3 no, M4 no (), M5 extrapolated: ; readings: H2 kept as predicted; H3 equivalence confirmed; H4 no comparison changed order.

| test | statistic | p | Holm | status |
|---|---|---|---|---|
| H1 (gatekeeper) | D1 = -0.739 on 17 ViTs | 0.009325 (perm 0.009325, mag 0.0004998) | n/a | confirmed yes |
| H2 | D2 = -0.011 (H2b) | 0.01049 | 0.01049 | tested yes, confirmed yes |
| H3 | Delta = -0.002 (degenerate no) | 0.0004998 | 0.001499 | tested yes, confirmed yes |
| H4 | Delta = +0.000 (degenerate yes) | 0.0004998 | 0.001499 | tested yes, confirmed yes |

- Gate: open (H1 confirmed); Holm order H3,H4,H2.
- Premise rules: P-P1 holds yes; P-P2a(i) holds yes (7 of 7 in band; failing ''); P-P2a(ii) holds (277/277, need 250); P-P2a holds; I's P-P2b holds (300/300 decided correct); P-P5a holds.

| per model | band | median c - B-bar | in band | pairs in band |
|---|---|---|---|---|
| 70m | [+0.00047, +0.00407] | +0.00212 | yes | 17/17 |
| 160m | [+0.00038, +0.00398] | +0.00215 | yes | 17/17 |
| 410m | [+0.00035, +0.00395] | +0.00211 | yes | 17/17 |
| 1b | [+0.00032, +0.00392] | +0.00206 | yes | 17/17 |
| 1.4b | [+0.00035, +0.00395] | +0.00216 | yes | 17/17 |
| 2.8b | [+0.00033, +0.00393] | +0.00219 | yes | 17/17 |
| 6.9b | [+0.00030, +0.00390] | +0.00202 | yes | 17/17 |

Trends (layer-max `full`, exact one-sided p over 5040 relabellings; descriptive except through H1-H4):

| metric | Tr raw | Tr s_cal | dTr s_cal (common set) | p (<=) | Tr unc | dTr unc |
|---|---|---|---|---|---|---|
| mKNN | +0.941 | +0.939 | -0.002 | 0.5 | +0.941 | +0.000 |
| CKA (biased) | +0.681 | -0.059 | -0.739 | 0.009325 | -0.124 | -0.805 |
| CKA (debiased) | +0.670 | +0.670 | +0.000 | 1 | +0.679 | +0.008 |

Robustness (descriptive): five sizes D1 -0.400; grid5 D1 -0.855; rho0 D1 -0.739; leave-one-size-out D1: 70m -0.484, 160m -0.713, 410m -0.729, 1b -0.854, 1.4b -0.837, 2.8b -1.022, 6.9b -0.464.
Uncensored (secondary): D1 -0.805, p 0.01171; I's category A.

## Validation files
- validation.json: I2 pass yes
- validation_vl.json: V-P4 pass yes, V-P7c pass yes, V-P1 pass yes, V-P2 pass yes, V-P7a pass yes
- validation_vp6_arh.json: V-P6 pass yes
- validation_cross_1-2-3_prh_clip+l2+l2.json: pass no (counts and key names only)
- validation_cross_1_prh_clip+l2.json: pass no (counts and key names only)

Written 2026-10-10T10:00:15Z by lmx_pyt_report.py from summary.json (finished 2026-10-10T09:57:14Z).
