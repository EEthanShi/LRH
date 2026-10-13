"""hrhv2 -- main implementation of the HRH v2 Stage 0/1 criteria ladder.

Implements SPEC.md v1.0 sections 0-7 (frozen 2026-10-08):

    prep        section 0  preprocessing, Gram/distance construction, tie rule
    structural  section 1  kNN sets, kNN-graph components, per-row Kendall,
                           global edge order, MST with forest/bridge split
    geometric   section 2  Gs, raw G, CKA (biased / debiased), Pearson RSA,
                           SOG distance-aware overlaps
    quotients   section 3  scale / affine / global isotonic / row isotonic,
                           cross-fitted, both directions
    gl          section 4  projector distance and CCA (linear models only)
    calibrate   section 5  permutation calibration with sha256 seeds
    predict     section 6  crossing-law predictor P1, dispersion law P2,
                           Levina-Bickel p-hat, nulls N0/N1/N2
    plants      section 7  known-answer plant generators
    ladder      one call computing the full criteria ladder for a pair
    run_pair    CLI: full ladder (+ optional kNN calibration) for two
                feature arrays -> JSON
    run_g0      gate G0: the SPEC 7 plant suite -> results/G0_report.*
    bench_ladder runtime benchmark

Interpretations of open SPEC points: SPEC_INTERPRETATIONS.md. The frozen P1
estimator: PREDICTOR_CHOICE.md (predict.FROZEN_P1_VARIANT).

All metric arithmetic is float64. No global RNG state is used anywhere:
every random draw comes from ``util.rng_from_key(<string key>)``.
"""

__version__ = "1.0.0"
SPEC_VERSION = "1.0"
