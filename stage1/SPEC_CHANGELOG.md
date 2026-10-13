# SPEC changelog

SPEC.md v1.0 stays frozen and is never edited. This file records three things:
- checks that SPEC asks to have recorded;
- deviations from SPEC, with their reasons;
- implementation changes that touch a SPEC quantity.

Entries are append-only.

## 2026-10-08 · verifier

No SPEC definition was changed. The evidence for every entry is in `results/verification/` and summarised in `results/verification_report.md`.

### Checks SPEC asks to have recorded

#### C1. §0 `prh_clip+l2` against the vendored ARH (785054b) and PRH (dcd76ba) code

What the vendored code does:
1. Computes a threshold once per model, on the stacked (m, L, d) feature tensor: for each sample, the linearly interpolated 0.95-quantile of |x| over all L·d coordinates; then the mean over samples.
2. Clamps every coordinate to that threshold.
3. L2-normalises each layer separately.

SOG (be654a37) does the same thing, in float32, through `prepare_features` followed by `_normalize_layers`.

How hrhv2 compares:
- hrhv2 implements this as `run_pair --clip-scope all-layers`, which is the default.
- On the full WIT-1024 stacks it matches PRH and ARH `remove_outliers` to within 2.2e-16 in float64.
- The float32 threshold computed by the vendored code differs from hrhv2's by at most 1.5e-7 (relative).

The per-layer scope is a different criterion, not a rounding variant. For LLaMA-7B at layer 16 it gives a threshold of 0.52 instead of 1.24.

#### C2. §2.10 SOG quantities, checked against be654a37 (`layerwise_engine`, `topology_alignment`)

- **Log-ratio.** SOG uses the plain chord distance d, not squared D:
  - kNN version: `sqrt(clamp(2 − 2⟨x,y⟩))`;
  - MST version: `torch.cdist`.

  SPEC's default is therefore correct. Fed float64 rows, hrhv2 reproduces SOG to ≤ 2e-14 on a pair with no duplicate items. A squared-D variant would be 11–99% lower.
- **MST.** SOG builds the MST on d (quantile-normalised, with SOG's own Prim tie rule). Without exact ties this gives the same tree as D under SPEC's tie key. The edge sets were identical on all 6 real cases, both in float64 and in SOG's native float32.
- **Preprocessing.** SOG's preprocessing equals C1 (`prh_clip+l2`, all-layer threshold).
- **Q_0.9.**
  - The MST variant takes `np.quantile` over the positive entries of both triangles.
  - The kNN variant takes `torch.quantile` over all off-diagonal entries, zeros included.
  - hrhv2 and hrhv2_ref use the positive entries for both variants. The two definitions differ only when items are exact duplicates.

#### C3. §2.8 CKA denominators

| Implementation | Denominator term added |
|---|---|
| PRH `cka` | 1e-6 |
| ARH `CKAUnbiased` | 1e-8 (its source is not vendored here; reconstructed from `utils.hsic_unbiased`, as hrhv2_ref also does) |
| ARH `estimators.cka_debiased` | 1e-8, on unnormalised sums, so negligible |

On WIT-1024, the +1e-8 changes debiased CKA by up to 2.2e-5, and the +1e-6 by up to 2.2e-3. hrhv2 reports two keys:
- `cka_debiased`: the exact Song et al. value, with no epsilon;
- `cka_debiased_arheps`: the ARH +1e-8 convention.

The "≤ 1e-8 vs ARH" match holds only when each key is compared with the vendored routine that uses the same convention: ≤ 2.1e-11 in float64.

#### C4. §5 calibration

SPEC's threshold is the ⌈(1−α)(K+1)⌉-th order statistic, which is what ARH `calibrated_similarity.calibrate()` uses. ARH/SOG's experiment code instead uses `compute_null_summary`, i.e. `np.quantile(null ∪ {obs}, 1−α)`. For K = 500 and α = 0.05 this is the same order statistic (the 476th).

Verified on 2 real pairs with ARH `calibrate()` and hrhv2's permutations injected: s_cal, p and c_α agree to ≤ 3.5e-8 (ARH stores values in float32), both for mKNN at k = 10 and for biased CKA.

### Implementation change touching a SPEC quantity

#### F1 (`hrhv2/FIXES.md`). §2.8 debiased CKA

The Song estimator is now evaluated on HKH. This is the same estimator, and the value is unchanged in exact arithmetic. The change only fixes a numerical cancellation that put hrhv2 outside the §8 tolerance on one real pair.

### Convention difference between the two implementations (no change made)

#### D1. §0 / §7 distance matrices supplied directly (P-Gauss and WarpShare plants)

- hrhv2 uses D as given, including negative entries.
- hrhv2_ref clips negative entries at 0.

SPEC states the clip only for a D built from a Gram. The P-Gauss oracle and the P1 model assume an unclipped D_B (b = u + σZ), so G0 rightly uses hrhv2's convention. When both implementations are given the same pre-clipped D, they agree exactly (26/26 cases).

### Points for the lead (not applied; see the verification report §5)

- **P1 at k = 1 (G0 P-Gauss).** This failure was confirmed independently: it is a property of the §6 balance-threshold μ approximation.
- **P-Bern isotonic gate.** Whether the gate should be read in-sample or cross-fitted.
- **§8 "≤ 1e-6 vs PRH/ARH/SOG".** This tolerance can only be met with float64 inputs. The vendored float32 pipelines differ by:
  - up to 1/(mk) per neighbour flip in mKNN;
  - up to 4.5e-4 for Song CKA in float32;
  - up to 4e-6 absolute for the SOG overlaps.
- **Exact-duplicate items.**
  - WIT-1024 items 493 and 531 have the same caption, so their LM features are identical.
  - The COCO pools contain 3 duplicate captions at m = 2048 and 49 at m = 16384.
  - Ties are broken by SPEC's lower-index rule, whereas torch `topk`/`argsort` breaks them arbitrarily.
  - The SOG weight of a duplicate pair is 1 in exact arithmetic, but ≈0 in SOG's float pipeline.
- **S1.1 with `prh_clip+l2` on nested pools.**
  - The threshold depends on which rows are selected.
  - The COCO files store only 4 layers, so the "all-layers" threshold there is not the ARH one.
  - Using `l2`, or a single threshold fixed in advance, avoids both problems.
