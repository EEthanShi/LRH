# P1 crossing-law predictor: choice of estimator (SPEC v1.0 section 6, G0)

Owner: harness-main. P1 is the functional SPEC defines: the balance threshold t* with E X(t*) = E Y(t*), μ̂_i = Σ_{j∈N_A(i)} P̂(b_j > t*), and 1 − s_N = mean_i μ̂_i / k. This file does not change P1. It freezes only the **estimator of the conditional coupling** P̂(b | u), which SPEC leaves to be "chosen on plants (G0) … then frozen".

## Part A: written before the formal G0 run

### Fixed design (common to every candidate)

These choices are not compared on plants.

- **Fit data.** Only pairs with both endpoints in H1 at the fit size m_fit = 1024. Predictions are made for H2 anchors only, from D_A alone. Tests show that changing any D_B entry outside H1×H1, or any D_A entry outside H1×H1, leaves the fit bit-identical (tests/test_predict.py).
- **Warp removal.** Per fit row, a multi-scale local affine link on dyadic rank bands with empirical-Bayes-shrunk slopes. The residual in A-units is r = (b − a − c u)/c, studentized by its leverage.
- **Pooling.** 'med': covariate log(u/ν_i), residuals in units of ν_i, where ν_i is the median of anchor i's A-row. ν_i is m-invariant in expectation and removes per-anchor scale, a source of heterogeneity on real data.
- **Shape.** 'B': the law of the MAD-standardized residual is estimated per bin of the covariate and interpolated between bins.
- **Extrapolation.** log-MAD is extended by power laws fitted on the lowest and highest quarter of the bins. The location is modelled in local-MAD units.
- **Balance.** μ is computed exactly for the fitted law: safeguarded Illinois root finding to |g| ≤ 1e-12 k, with column truncation that is exact within the law's support (tested against brentq to 1e-9).

### Candidates (three, compared on the formal G0 seeds)

1. `P1-med-B-r8`: residuals at ranks ≥ 8 only. At lower ranks the band window cannot be centred, so link-intercept error dominates the residual.
2. `P1-med-B-r8f`: as 1, with each column's law truncated to b̃ ≥ 0 in A-units (distances are non-negative), then renormalized.
3. `P1-med-B-difff`: scale from adjacent-rank differences of A-unit values, which are free of link intercept at every rank, with the same b̃ ≥ 0 truncation.

### Selection rule

Implemented as `run_g0.choose_variant`.

1. **Fidelity filter.** Keep the variants whose P1 stays within 0.02 of the oracle on every eligible P-Gauss cell, if any variant does. The oracle is the same μ-law computed with the true P-Gauss coupling b = u + σN(0,1). This filter isolates the estimator from the μ-law's own small-k error.
2. **Accuracy.** Among the kept variants, minimise the class-mean absolute error against observed s_N, taken over all eligible cells (observed s_N ∈ [0.05, 0.99]) of P-Gauss, P-Aniso, P-Conf and P-Rough. Each class gets equal weight.
3. **Ties.** Variants within 0.002 of the minimum are tied and resolved by the a-priori order `P1-med-B-r8` > `P1-med-B-r8f` > `P1-med-B-difff` (simplest first).

The protocol is S1.1's: fit on H1×H1 at m = 1024, then predict on H2 anchors at m ∈ {256, 1024, 4096}. Cells use k ∈ {1, 3, 10, 30, 41, 100} for P-Gauss and {1, 3, 10, 30, 100} for the other classes. Results are means over replicate draws: 24 per P-Gauss setting and 4 per other plant, for p ∈ {2, 4, 8}. The formal run uses the fresh seed prefix `G0f`.

### Disclosure of the development history

The development history was not pre-registered.

- **Development seeds.** Development used seed prefix `G0dev` with one draw per plant. It compared, in this order: pooling u / med / q with a pooled empirical shape (E); link handling all / r8 / diff; per-bin shape (B); and the b̃ ≥ 0 floor. Three fixes came out of those runs: per-bin shapes, r8/diff link handling, and location measured in MAD units. Pooling 'q' (literal rank position) was dropped because it was consistently worse on P-Aniso. That is the expected Jensen bias of absolute residual scales pooled by rank position at small ranks. On the homogeneous plants 'u' and 'med' were indistinguishable, and 'med' was kept for its robustness to anchor scale on real data.
- **Smoke run.** A smoke run with seed prefix `G0` (p = 2, 2 replicates) was inspected before this rule was fixed. In that run the original filter, "pass the P-Gauss gate on k ≥ 3", excluded every candidate, because the oracle μ-law itself misses some k = 3 cells by about 0.025. The filter was therefore restated as fidelity to the oracle. The formal run uses fresh seeds.
- **Known limitation.** Every candidate fails badly on P-Conf (spurious predicted collapse) and is biased low at small k on P-Aniso. The diagnosis: under the pooled location-scale coupling, far columns carry heavy lower tails, and heterogeneous anchor couplings (a 9× variation of the conformal factor) are pooled. The floor variants reduce, but do not remove, the P-Conf error, at the cost of bias on P-Gauss.

## Part B: result (filled after the formal G0 run)

Formal run: 2026-10-08T03:53:42Z, seeds `G0f`, run_g0.py sha256 `d380cb4d3b0d99daea51ff3045e24fd0bd0bdc34bef9af0000129736b3836c86`, predict.py sha256 at run time `aedbd0ebfb45e0b4fa281eb9cf34a9e72ab397349db2a421b75405fce949fc26` (the only later edit of predict.py is the `FROZEN_P1_VARIANT` constant). Replicates: 24 per P-Gauss setting, 4 per other plant.

| variant | max abs(P1 - oracle) on P-Gauss (filter <= 0.02) | class-mean abs err | P-Gauss | P-Aniso | P-Conf | P-Rough |
|---|---|---|---|---|---|---|
| `P1-med-B-r8` | 0.0048 (pass) | 0.1414 | 0.0077 | 0.0497 | 0.4804 | 0.0278 |
| `P1-med-B-r8f` | 0.0774 (fail) | 0.0931 | 0.0141 | 0.0498 | 0.2811 | 0.0275 |
| `P1-med-B-difff` | 0.0791 (fail) | 0.0895 | 0.0143 | 0.0625 | 0.2525 | 0.0286 |
| oracle (reference) | - | 0.0076 | 0.0076 | - | - | - |
| N0 (reference) | - | 0.0477 | 0.0550 | 0.0121 | 0.0300 | 0.0939 |
| N1 (reference) | - | 0.2698 | 0.0076 | 0.3921 | 0.6314 | 0.0483 |
| N2 (reference) | - | 0.2435 | 0.0840 | 0.2468 | 0.4825 | 0.1609 |
| P2 (reference) | - | 0.0694 | 0.0723 | 0.0822 | 0.0406 | 0.0824 |

Fidelity filter applied: True; candidates after filter: ['P1-med-B-r8']; tied within 0.002: ['P1-med-B-r8'].

**FROZEN: `P1-med-B-r8`** (`hrhv2.predict.FROZEN_P1_VARIANT`). It is used unchanged for S1.1: fit on H1 x H1 at m = 1024 and predict on H2 anchors at the other m. It will not be changed without a new version of this file and a STATUS entry.

SPEC 7 P-Gauss gate with the frozen variant: **FAIL**. Max abs(P1 - obs) = 0.0565 over 190 eligible cells. By k:

| k | max abs(P1-obs) | signed mean | max abs(oracle-obs) (mu-law, true law) | max abs(P1x-obs) (exact, fitted law) | max abs(oracle_x-obs) (exact, true law) |
|---|---|---|---|---|---|
| 1 | 0.0565 | -0.0241 | 0.0569 | 0.0139 | 0.0142 |
| 3 | 0.0161 | -0.0085 | 0.0165 | 0.0061 | 0.0056 |
| 10 | 0.0057 | -0.0026 | 0.0060 | - | - |
| 30 | 0.0028 | -0.0006 | 0.0019 | - | - |
| 41 | 0.0031 | -0.0004 | 0.0013 | - | - |
| 100 | 0.0009 | -0.0000 | 0.0005 | - | - |

Reading: where P1 and the oracle miss observed s_N together at small k, the cause is the mu-law approximation in SPEC's P1 definition, not the coupling estimator. In that case the exact expected discrepancy under the true law (oracle_x) recovers observed s_N. See G0_report.md.
