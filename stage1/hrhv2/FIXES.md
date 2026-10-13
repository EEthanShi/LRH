# hrhv2: fixes made by the verifier

Each entry gives the symptom, how it was resolved, the code change, and the test that guards it. The files as they were before each fix are kept in `results/verification/backup_hrhv2/`.

## F1 (2026-10-08): debiased CKA lost up to 1.2e-9 relative accuracy to cancellation

**Symptom.** The SPEC §8 agreement check failed on one real-feature case: WIT-1024, `bigscience/bloomz-1b7` (hidden state 12) vs DINOv2-L (block 11), `pre="l2"`, m = 1024.

| | `cka_debiased` |
|---|---|
| hrhv2 | 0.007979246145953 |
| hrhv2_ref | 0.007979246155486 |

The relative difference is 1.19e-9, against a SPEC tolerance of 1e-10. The same 1.19e-9 shows up in `cka_debiased_arheps`. Of the 684 float comparisons over the 12 real cases (1,640 comparisons in all), these were the only two failures.

**Resolution by brute force.** `results/verification/exact_cka.py` recomputes the exact value. Every float64 Gram entry is a dyadic rational, so Song et al. (2012) eq. 5 can be evaluated in Python integers and `Fraction`s, with CKA taken in 60-digit `Decimal`. Errors against that exact value (`results/verification/exact_cka.json`):

| | relative error on the failing case | across all 12 real cases |
|---|---|---|
| hrhv2 | 1.2e-9 | up to 1.2e-9 |
| hrhv2_ref | 1.4e-11 | at most 1.4e-11 |

hrhv2_ref was correct; hrhv2 was the inaccurate one. hrhv2's error also changed with the BLAS thread count: 1.5e-10 with 2 threads, 1.2e-9 with 1.

**Cause.** `cka_pair` evaluated Song's formula on the raw Gram K. On L2-normalised rows its three terms are large and nearly cancel. On the failing pair, |t1| / |m(m−3)·HSIC_AB| = 2.7e5, which amplifies ordinary float64 summation rounding.

**Fix** (`hrhv2/geometric.py::cka_pair`). The unbiased HSIC is now evaluated on the double-centred Grams HKH, which the biased CKA already computes. This is the same estimator in exact arithmetic, for two reasons:
- the Song U-statistic is invariant to K_ij → K_ij + a_i + a_j;
- for symmetric K, HKH = K + a1ᵀ + 1aᵀ with a_i = −r_i/m + S/(2m²), where r_i is the row sum of K and S its total sum. Song's estimator ignores the diagonal.

hrhv2 already used −½HDH for distance-only inputs, so the two input paths are now treated the same way. Errors against the exact value after the fix, for 12 real cases (`results/verification/hsic_variants.json`):

| formulation | relative error |
|---|---|
| raw Gram (old) | up to 1.5e-10 (2 BLAS threads) |
| HKH (new) | at most 3.1e-15 |
| explicit U-centring (alternative) | at most 2.0e-15 |

The SPEC §8 check was then re-run on all 12 real cases: 0 of 1,640 comparisons fail, and the largest float relative difference is 1.4e-11 (`results/verification/agree_postF1/`).

**Test.** `hrhv2/tests/test_geometric.py::test_cka_debiased_accurate_on_ill_conditioned_gram` uses 40 rows that share a large common mean, with exact `Fraction` arithmetic as the reference. It checks that the new code has relative error ≤ 1e-12 (measured: 7e-15) and that the old raw-Gram evaluation is at least 100× worse (measured: 1.2e-8). `pytest hrhv2/tests` passes 100/100.

**Impact.** Only `cka_debiased`, `cka_debiased_arheps` and `hsicu_*` change, by at most about 1e-9 relative. All other keys are bitwise unchanged. The G0 re-run with this fix is in `results/verification/G0_rerun/`.
