# ARH layer-max replication: independent second implementation (role `arh-layermax`)

Written and hashed before any layer-max number is computed by this role. No layer-max output of the parallel role `arh-layermax-early` has been read; its code is not read until my own numbers are final.

## Scope and definitions

- **Definitions.** Adopted verbatim from `analysis/s12_layermax/ANALYSIS_PLAN_LAYERMAX.md` v1 (sha256 dd1897214d…), §1–§3 and §5–§7: data, layers, preprocessing, scores, calibration, trend statistics, reading rule and attribution diagnostics. Its permutation keys `S1.2-layermax|<lm>|<vit>` (k = 0..499) are reused, so the two implementations must agree number for number.
- **Primary** = `prh_clip+l2` with the all-layer clip threshold (D5). **Secondary** = `l2`. **Exploratory** = debiased CKA.
- **Outputs.** Everything this role writes goes under `analysis/s12_layermax/indep/` and `results/S1_2_layermax/indep/`.

## Implementation (independent of the early role's code)

- **kNN tables.** Computed on the CPU by `hrhv2.structural.knn_table`, using hrhv2 float64 arithmetic: `common.preprocess_rows`, then K = X Xᵀ, then `gram_to_dist`. This gives SPEC tie-rule identity by construction for every stored layer.
- **mKNN on the GPU.** count_π[a,b] = Σ_ij M_A^a[i,j] · M_B^b[π(i),π(j)], with M the 0/1 kNN adjacency.
  - Computed as a float32 GEMM over the m² flattened entries. Every partial sum is an integer ≤ 10240, so the result is exact.
  - The relabelling equals SPEC recomputation on the permuted D, which hrhv2 slices rather than recomputes, whenever no ViT row has an exact tie D_(10) = D_(11). This is checked for every ViT layer and pre.
- **CKA on the GPU.** Centred Grams Kc = `hrhv2.prep.center_gram(K)` are computed on the CPU and moved to the GPU in float64.
  - HSIC_π = 2·Σ_{i<j} Kc_A[i,j]·Kc_B[π(i),π(j)] + Σ_i Kc_A[i,i]·Kc_B[π(i),π(i)]. This is one float64 GEMM per batch of permutations over the upper triangle.
  - Norms ‖Kc‖_F² and the unbiased self-HSICs are computed with hrhv2 on the CPU.
  - Debiased: Song's estimator on Kc, with t1 = 2·(upper-triangle GEMM), and t3 from the permuted row sums of K̃.
- **Storage.** The full observed and null L_A × L_B matrices are stored for every pair, with all K = 500 nulls. Every sub-grid maximum (A2) is therefore exact and needs no rerun.

## Validation (reported whatever the outcome)

- **I1.** Clip thresholds recomputed from the float64 stacks must equal the S1.2 cache bitwise.
- **I2.** On ≥ 12 layer pairs, chosen before the run (below), check the identity and permutations k ∈ {0, 1, 499}:
  - GPU CKA (biased and debiased) vs `hrhv2.geometric.cka_pair` on `B.permuted(π)`: ≤ 1e-6 absolute, expected ~1e-13.
  - mKNN vs `hrhv2.structural.knn_overlap` from `knn_table(B.permuted(π).D)`: exactly equal.
- **I3.** The fixed-depth entries (S1.2 layer rule, r = 0.5 and 1.0) must reproduce the S1.2 panel s_obs for all 136 × 2 × 2 configurations: mKNN exactly, CKA ≤ 1e-10.
- **I4.** Vendored ARH `calibrate()` (one layer pair) and `calibrate_layers(agg="max")` (bloomz-560m × ViT-Ti/16-in21k, full 25 × 12 grid), with our permutations injected. Tolerance 1e-6 (ARH stores float32). Tie-rule differences are reported, not hidden.
- **I5.** Cross-implementation comparison with `arh-layermax-early`, if its outputs exist when mine are final.

I2 layer pairs, chosen now: (A, B) ∈ {(bloomz-560m, ViT-Ti in21k), (llama-7b, DINOv2-g), (open_llama_7b, CLIP-H ft), (bloomz-7b1, MAE-H)}, each at (first, first), (middle, middle) and (last, last) stored layers. That is 12 pairs, 2 pre and 4 permutation indices (identity, 0, 1, 499).

## Reading

The question is answered with the plan's §6 rule (ARH-style Δ_CKA − Δ_mKNN and r_after; S1.2-axis Tr_unc thresholds) and the §7 attribution rule. Anything beyond those is labelled exploratory.
