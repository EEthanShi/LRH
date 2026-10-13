# hrhv2_ref: reference implementation of SPEC v1.0 §0–§5 (+ §4 projector), report

Role: harness-ref. Package: `~/hrh_v2_stage1_20261008/hrhv2_ref/hrhv2_ref/` (v0.2.0). Written from SPEC.md v1.0 and the vendored PRH/ARH/SOG code only. `hrhv2/` was never read.
Date: 2026-10-08 (UTC). All numbers below were computed on the experiment server, CPU only, float64, ≤ 8 threads.

## 0. Status in one paragraph

The previous run of this role wrote v0.1 (pytest 45/45; 50/50 by the time it was interrupted). I re-ran the suite (50/50 pass), checked every SPEC metric name, and filled the gaps (§2 below) to make v0.2.0. **`pytest -q tests` now passes 56/56 with no skips** (≈ 4 s). `ladder_ref(X_A, X_B, pre, kgrid)` returns every SPEC v1.0 name. `scripts/key_inventory.py` at m = 1024 finds no missing names for `pre="l2"` or `pre="prh_clip+l2"` (169 and 171 keys, about 3.8 s each). `calibrate_ref` implements §5 and matches the vendored ARH `calibrate()`. The v0.1 files I changed are kept in `hrhv2_ref/_bak_v0.1/`, and the results they superseded are in `hrhv2_ref/results/v0.1_superseded/`.

Three findings need a decision by the lead, because they touch SPEC/G0 and so are not mine to change:
1. **The P-Bern requirement "Δ_O = Δ_E = 0 exactly" cannot be met in float64 at β = 10** (and for Δ_E at β = 3, m = 4096) with the SPEC §0 distance formula. This holds for any correct implementation; see §6. Every other ordinal rung (kNN at all k, components, MST) is exactly identical in all 27 cells.
2. **"Match the ARH debiased CKA to ≤ 1e-8"** depends on which ARH routine is meant. ARH's `CKAUnbiased` adds 1e-8 to the denominator, which moves CKA by about 6e-8. The exact Song estimator (and ARH `estimators.cka_debiased`) does not.
3. **"Match the SOG code to ≤ 1e-6"** holds to ~1e-12 when SOG's functions get float64 input. SOG's own pipeline runs in float32 and differs by up to 1.2e-5 absolute (17 % relative for `sog_mst@1e-4`).

## 1. Public API

```python
from hrhv2_ref import ladder_ref, calibrate_ref, ladder_score
out = ladder_ref(X_A, X_B, pre="l2"|"prh_clip+l2"|"none", kgrid=None,
                 gram=False, dist=False, parts=None, H1=None, item_ids=None,
                 clip_threshold_A=None, clip_threshold_B=None,
                 sog_q_mode="ordered", sog_eps=1e-12, ...)       # flat dict
cal = calibrate_ref(X_A, X_B, ladder_score("s_N@10", pre="l2"), K=500, alpha=0.05,
                    key="...", perms=None, gram=False)           # layered: pass lists
from hrhv2_ref.compare import compare_ladders                     # SPEC §8 tolerance check
```

Inputs can be features (default), Gram matrices (`gram=True`), or squared-distance matrices (`dist=True`; CKA then uses K = −D/2, which is exact because both HSICs are invariant to K → K + a1ᵀ + 1aᵀ). `parts` restricts the computation to a subset of {knn, comp, rank, edge, mst, geom, cka, rsa, sog, quot, proj}.

### Returned SPEC names (all present; checked by `tests/test_ladder.py::test_all_spec_names_present_and_finite` and `results/key_inventory_m1024.json`)

| SPEC item | keys |
|---|---|
| §0 near-tie | `near_tie_A@k`, `near_tie_B@k` |
| §1.1 kNN | `s_N@k`, `delta_N@k`, `chance@k` for k ∈ {1,3,10,30,100} ∪ {k_q}; `k_q` |
| §1.2 components | `comp_rand@k`, `comp_n_A@k`, `comp_n_B@k`, `comp_degenerate@k`, k ∈ {1,3,10} |
| §1.3 per-anchor | `delta_O` (+ `delta_O_strict_disc`, `delta_O_tieA_only`, `delta_O_tieB_only`, `rows_with_ties_frac`, `delta_O_method`) |
| §1.4 global | `kendall_global` (τ_b), `spearman_global`, `delta_E` (+ exact pair counts `E_n_disc_strict`, `E_ties_A_only`, `E_ties_B_only`, `E_ties_both`) |
| §1.5 MST | `s_T`, `s_T_forest`, `s_T_bridge`, `forest_share_A`, `s_N@1` (always), `F_A_subset_T_A` (+ `forest_share_B`, `forest_share_shared_{A,B,union}`, `n_bridges_A`, `n_shared_bridges`, `F_B_subset_T_B`, `prim_eq_kruskal_{A,B}`) |
| §2.6–7 geometry | `Gs_rms`, `Gs_max`, `G_rms`, `G_max` |
| §2.8 CKA | `cka_biased`, `cka_debiased` (+ `cka_debiased_eps8` = ARH, `cka_debiased_eps6` = PRH, `hsic_u_{AB,AA,BB}`) |
| §2.9 RSA | `rsa_pearson` |
| §2.10 SOG | `sog_knn@τ`, `sog_mst@τ`, τ labels `1, 0.1, 0.01, 0.001, 0.0001` (+ `sog_Q_A`, `sog_Q_B`, `sog_Q_mode`) |
| §3 quotients | `quot_{scale,affine,iso,rowiso}_{expl,rms,max}` (A→B), each also with `_BA` and `_mean`, and the cross-fitted set `..._xfit`, `..._xfit_BA`, `..._xfit_mean`; `H1_size` |
| §4 projector | `proj_dist`, `cca_mean_sq`, `proj_rank_A`, `proj_rank_B` |
| §0 prh_clip | `clip_c_A`, `clip_c_B` |
| §5 | `calibrate_ref` → `s_obs`, `c_alpha`, `rank_r`, `s_cal`, `p_value`, `uncensored`, `K`, `alpha`, `null` |

## 2. What changed in this session (v0.1 → v0.2.0)

| Change | Why |
|---|---|
| SOG scale Q_0.9 is now taken over the **positive off-diagonal matrix entries (ordered pairs)** instead of unordered pairs. Implemented as an analytic quantile of the doubled multiset (`geometric.quantile_linear_doubled`). | This is the literal SPEC wording, and it is exactly what the pinned SOG code does (`_normalize_distance_array`: `np.quantile(D[D>0], q)` on the full matrix). The two conventions differ by about half a sorted-index step. At m = 1024 that changes `sog_*@1e-4` by 6e-4 to 2e-3 relative. At m = 4096 the two coincide by arithmetic accident: the fractional position is 0.1 in both. Option `sog_q_mode="unordered"` restores v0.1. |
| The SOG weight now uses log max(d̃, 1e-12). | This is the pinned SOG convention in both variants. Both distances zero → w = 1. Exactly one zero → w is tiny but positive (v0.1 gave 0). `sog_eps=0` restores v0.1. |
| New keys `delta_N@k`; `s_N@1` is always reported by the MST block. | SPEC §1.1 defines Δ_N, and SPEC §1.5 lists s_N(1) with the forest/bridge split even when 1 is not in `kgrid`. |
| New keys `forest_share_shared_B` and `forest_share_shared_union`. | PREREG PE2 says "forest share of shared MST edges" without saying whose forest. |
| New tests: `tests/test_vendor_sog.py` and `tests/sog_vendor_loader.py` (SOG be654a37 functions extracted with `ast`; the vendored files are untouched), a full SPEC-name test, a projector-vs-CCA identity test, and doubled-quantile tests. | Records the SOG check that SPEC §2.10 asks for. |
| The PRH vendor test now reads this project's `vendor/platonic-rep` (dcd76ba, byte-identical to the earlier project's copy). | Self-containment. |
| New scripts: `sog_crosscheck.py`, `pbern_tie_anatomy.py`, `key_inventory.py`. `pbern_scale_check.py` now keeps the strict/tie decomposition. | Evidence for §5 and §6 of this report. |

## 3. Algorithms (chosen to differ from the obvious implementation)

| Quantity | Reference algorithm | Independent check |
|---|---|---|
| kNN sets | Full lexicographic `np.lexsort` on the explicit (D_ij, j) key for every row; self pushed to +inf. Overlap is counted from duplicates in the sorted concatenation of the two neighbour lists. | PRH `mutual_knn` (exact integer counts); PRH-style argsort at m = 1024 and 4096; hand cases. |
| kNN components | Union-find (path halving, union by size). Rand distance from the pair-count identity on the contingency table. | Brute-force Rand over all pairs. |
| Δ_O (per anchor) | Fagin K^(1/2) via a **vectorised bottom-up merge-sort inversion count** applied to all rows of a chunk at once, plus exact tied-pair counts. Brute-force O(n²) sign products for m ≤ 200. | Brute force equals merge sort to 1e-15, including rows with ties. Scipy per-row τ_b gives (1−τ_b)/2 to 1e-16 on tie-free rows. |
| τ_b, Δ_E (global) | The same merge-sort counter on the m(m−1)/2 pairs, with exact integer counts of strict discordances and A-only, B-only and joint ties. | scipy `kendalltau`, difference 1e-16 at m = 1024 and 4096. |
| Spearman | Pearson correlation of mid-ranks (own mid-rank routine). | scipy `spearmanr`, difference ≤ 1.2e-13. |
| MST | **Kruskal** with union-find on the strict total order (D, min(i,j), max(i,j)). Dense Prim using the full SPEC key is run as a cross-check (`prim_eq_kruskal_*`). | scipy csgraph MST (tie-free); Prim with the tie key equals Kruskal under heavy ties. |
| 1-NN forest, bridges | Edge codes i·m + j and set algebra. | `F_A ⊆ T_A` holds in every run (this follows from the tie keys; see item 12 of §7). |
| Biased CKA | HSIC with an **explicit centring matrix** H (Gretton 2005). Double-centring identity used for m > 6000. | ARH `estimators.cka_biased`; PRH `cka` (with its +1e-6); feature-space ‖YᵀX‖² formula at m = 1024 and 4096. |
| Debiased CKA | **Song et al. 2012, eq. 5** written out explicitly (K̃ with zero diagonal). | ARH `utils.hsic_unbiased` (rel 1e-11); ARH U-centering `estimators.cka_debiased` (1e-9); the U-statistic definition by brute force (small m). |
| RSA | Pearson correlation of the upper-triangle entries of D. | Hand case. |
| SOG | Own neighbour lists and Kruskal tree; analytic doubled-multiset quantile; vectorised weights. | Pinned SOG functions on float64 input: kNN ≤ 1.3e-12 rel, MST exact (§5). |
| Quotients | Closed-form LS for scale and affine. **Own weighted pool-adjacent-violators** with tie pooling for the global and per-row isotonic fits, then linear interpolation between knots with constant extrapolation. | scipy `isotonic_regression`, ≤ 6e-15 on tie-free data; min–max formula for PAV; exact-warp plants. |
| Projector | Literal formula Xc (XcᵀXc)⁺ Xcᵀ via `pinv`. CCA from the singular values of Q_AᵀQ_B. | Identity `proj_dist = 1 − cca_mean_sq` when r_A = r_B (1e-12); hand case. |
| Calibration | Critical rank computed in exact rational arithmetic (`Fraction`). Permutations from sha256-seeded PCG64. Layer max over all layer pairs with one shared π. | Vendored ARH `calibrate()` with injected permutations (agrees to its float32 precision); hand cases. |
| prh_clip | Per-row linear-interpolated quantile of \|x\| (own sort and interpolation), averaged over rows. | PRH `remove_outliers` and ARH `preprocess.remove_outliers`, max diff ≤ 1e-13, for an (m, d) array and for a stacked (m, L, d) array. |

## 4. Verification evidence (files under `hrhv2_ref/results/`)

- **pytest 56/56** (`tests/`): hand cases, brute-force equivalences, and the vendor matches (PRH dcd76ba, ARH 785054b, SOG be654a37).
- `scipy_crosscheck.json` (v0.1, still valid because nothing it covers changed). On non-monotone pairs at m = 1024 and 4096: τ_b and ρ match scipy to about 1e-16; row Kendall to 1e-16; isotonic fits to ≤ 6e-15; MST equals csgraph; kNN@100 equals PRH-style; the two biased-CKA routes agree to 1e-16. `cka_debiased − cka_debiased_eps8` = 5.9e-8 / 6.4e-8 (finding 2 in §0).
- `sog_crosscheck.json` (new): see §5.
- `pbern_ref_small.json` (m = 256, 1024) and `pbern_ref_4096.json`, regenerated with v0.2. In all 27 (m, p, β) cells: s_N = 1 at every k (k_q = 3, 10, 41), comp_rand = 0, s_T = 1, F_A ⊆ T_A, Prim = Kruskal, global-isotonic explained fraction = 1.000000000 (≥ 0.999 ✓), and CKA decreasing in β in every (m, p) ✓. Δ_E is not exactly 0 in 9 cells, and Δ_O in 4 of those (§6). The full ladder takes about 2.5 s at m = 1024 and 63–86 s at m = 4096; the row-isotonic quotients dominate (about 30 s).
- `calib_bench.json` (v0.1, unchanged code path), m = 1024, K = 500: `s_N@10` gives s_cal = 0.118 (p = 1/501) in 64 s; `cka_debiased` gives 0.506 in 67 s; `s_T` gives 0.059 in 106 s. The kNN relabelling shortcut agrees with full recomputation in 50/50 permutations. **Observation for the lead:** `quot_iso_expl` gets s_cal = 0 and p = 1, because its null values (c_α = 0.37) exceed the observed 0.08. The explained fraction is measured against SS0 = Σ(y − x)², which grows under permutation while the isotonic fit absorbs the level shift. So `quot_*_expl` is not an alignment score that can be calibrated with §5; use it only as a descriptive share.
- `key_inventory_m1024.json`: all keys and values for a realistic pair, `pre ∈ {l2, prh_clip+l2}`, with no missing SPEC names.

## 5. SOG check (SPEC §2.10 asks for it to be recorded)

Pinned code: `vendor/sog_what_converges_prh`, HEAD = be654a3732c8ed92329899d68df759d61ae040a4.

- **Quantity.** SOG uses the **plain (chord) distance d on L2-normalised rows**, not D. The kNN path computes `sqrt(clamp(2 − 2·x·xᵀ))` and the MST path uses `torch.cdist`. The SPEC default is therefore correct.
- **Scale.** The MST path takes `np.quantile(D[D>0], 0.9)` over the full matrix (ordered, positive entries). The kNN path takes `torch.quantile(D[offdiag], 0.9)`: ordered entries with **zeros kept**, which is identical unless two items coincide. Both use linear interpolation. hrhv2_ref v0.2 follows the MST-path rule, which is also the SPEC wording ("positive").
- **Log clamp.** Both paths use log(max(d/Q, 1e-12)).
- **Support.** SOG's kNN uses `torch.topk` on cosine similarity (arbitrary ties), and its MST uses dense Prim with strict-improvement updates and first-index argmin. That is not the SPEC tie key, and it can produce a different tree under exact ties (`test_sog_prim_can_differ_under_ties`). On tie-free data the supports are identical.
- **Results** (`sog_crosscheck.json`, the scipy_crosscheck pair, m = 1024 / 4096, all τ):
  - SOG functions on float64 input vs ref: kNN ≤ 4e-15 rel at m = 1024 and ≤ 1.3e-12 at m = 4096; MST exactly 0. kNN sets, MST edges and Q all identical.
  - SOG native float32 pipeline vs ref: the kNN sets (all anchors) and MST edges (all m − 1) are identical, but the values are not. Max absolute difference is 1.5e-7 (kNN) and 1.2e-5 (MST). Relative difference at τ = 1e-4 is 1.8e-3 (kNN) and **0.175** (MST, m = 1024). The cause is float32 cancellation in `cdist` on short MST edges, amplified by 1/τ.
  - Ordered vs unordered Q, m = 1024: relative difference 8e-8 at τ = 1 rising to 6.5e-4 (kNN) and 2.1e-3 (MST) at τ = 1e-4. At m = 4096 the difference is 0, by coincidence.
- **Implication for SPEC §8 ("match … the SOG reference code to ≤ 1e-6").** This can only be met against SOG's functions run in float64 (the procedure in `tests/test_vendor_sog.py`). Against SOG's float32 pipeline the absolute gap reaches 1.2e-5 for `sog_mst`. The tolerance should name the dtype, and say whether it is absolute or relative.

## 6. `results/pbern_tie_diagnostic.txt`: content and explanation

Content (verbatim; produced by `scripts/pbern_tie_diagnostic.py` on the reference's own P-Bern plant, a Gaussian N(0, I_p) cloud isometrically embedded in ℝ⁶⁴ and L2-normalised, with K_B = exp(β(K_A − 1))):

```
m=1024 p=2 beta=10.0 | general: E_strict_disc=9720 E_Bonly_ties=529 rows_strict_disc_mean=4.67e-08 rows_Bonly_tie_mean=1.87e-09 | unit+expm1: E_strict_disc=0 E_Bonly_ties=585 rows_strict_disc_mean=0 rows_Bonly_tie_mean=0
m=1024 p=8 beta=10.0 | general: E_strict_disc=15 E_Bonly_ties=0 rows_strict_disc_mean=0 rows_Bonly_tie_mean=0 | unit+expm1: E_strict_disc=0 E_Bonly_ties=1 rows_strict_disc_mean=0 rows_Bonly_tie_mean=0
m=4096 p=2 beta=10.0 | general: E_strict_disc=2448979 E_Bonly_ties=146084 rows_strict_disc_mean=4.67e-08 rows_Bonly_tie_mean=5.04e-09 | unit+expm1: E_strict_disc=0 E_Bonly_ties=179666 rows_strict_disc_mean=0 rows_Bonly_tie_mean=5.33e-09
m=4096 p=4 beta=10.0 | general: E_strict_disc=25796 E_Bonly_ties=1413 rows_strict_disc_mean=3.79e-10 rows_Bonly_tie_mean=8.74e-11 | unit+expm1: E_strict_disc=0 E_Bonly_ties=1417 rows_strict_disc_mean=0 rows_Bonly_tie_mean=2.91e-11
m=4096 p=8 beta=10.0 | general: E_strict_disc=2973 E_Bonly_ties=164 rows_strict_disc_mean=5.83e-11 rows_Bonly_tie_mean=0 | unit+expm1: E_strict_disc=0 E_Bonly_ties=157 rows_strict_disc_mean=0 rows_Bonly_tie_mean=0
m=4096 p=2 beta=3.0 | general: E_strict_disc=3 E_Bonly_ties=1 rows_strict_disc_mean=0 rows_Bonly_tie_mean=0 | unit+expm1: E_strict_disc=0 E_Bonly_ties=0 rows_strict_disc_mean=0 rows_Bonly_tie_mean=0
m=4096 p=8 beta=3.0 | general: E_strict_disc=0 E_Bonly_ties=0 rows_strict_disc_mean=0 rows_Bonly_tie_mean=0 | unit+expm1: E_strict_disc=0 E_Bonly_ties=0 rows_strict_disc_mean=0 rows_Bonly_tie_mean=0
```

**Columns.** "general" builds D_B with the SPEC §0 formula K_B,ii + K_B,jj − 2K_B,ij from the Gram plant; this is what `ladder_ref(..., gram=True)` does. "unit+expm1" builds D_A = 2 − 2K_A and D_B = −2·expm1(β(K_A − 1)): a chain of maps (subtract, scale, expm1, scale) that are each monotone in floating point, so it can merge values but never reorder them. `E_strict_disc` counts strictly discordant pairs of pairs in the global edge order (out of about 1.4e11 at m = 1024 and 3.5e13 at m = 4096). `E_Bonly_ties` counts pairs of pairs that are tied in D_B but not in D_A. The `rows_*` columns are the per-anchor averages of the same two fractions, which make up Δ_O (Δ_O = strict + ½·ties).

**What it shows.**
1. **The order flips come from the diagonal, not from the implementation.** After L2 normalisation, K_A,ii = ‖x̂_i‖² differs from 1 by up to 8.9e-16 (a few ulp) in 72–75 % of rows (`pbern_tie_anatomy.json`). The plant multiplies this by β: K_B,ii − 1 reaches 8.9e-15 at β = 10. In the general formula, the row offsets ε_i + ε_j (up to about 1.8e-14) are then compared against differences between far pairs. For far pairs K_B ≈ e^{−2β} ≈ 2e-9 (at β = 10), so D_B ≈ 2 − 4e-9, and neighbouring far pairs differ in D_B by only 2β·K_B·ΔK_A. That is comparable to or smaller than the offsets, so the offsets reorder them, giving strict discordances. With the diagonal fixed at exactly 1 ("unit") or with expm1, there are **zero** strict discordances in every cell (diagnostic and anatomy).
2. **The ties come from float64 resolution.** Just below 2 the spacing of doubles is 2.2e-16. Far pairs whose true D_B values differ by less than that collapse to the same double. These B-only ties remain even under the exactly monotone construction (179,666 global ties at m = 4096, p = 2). Δ_O and Δ_E count each such tie as ½, so they become positive (about 1e-8 to 1e-13) without any reordering.
3. **Both effects sit in the far tail.** In each anchor's distance ranking (`pbern_tie_anatomy.json`), the first flip appears no earlier than rank 897 of 1023 (m = 1024) or 3067 of 4095 (m = 4096). The median position is at the 0.97–0.998 rank quantile. B-only ties start at the 0.83 rank quantile or later. kNN (k ≤ 100 or k_q), the kNN-graph components, and the MST (built from short edges) never reach that region. This is why all 27 P-Bern cells give **s_N = 1 at every k, comp_rand = 0 and s_T = 1 exactly**, while Δ_O/Δ_E are about 5e-8 at β = 10, p = 2 and about 1e-13 at β = 3, m = 4096.
4. **Scaling.** The effect is largest for p = 2 (most far pairs, the most saturated K_B) and β = 10. It vanishes for β = 1 everywhere, and for β = 3 except m = 4096, p = 2 (Δ_E = 9.2e-14: 3 strict discordances and 1 tie) and m = 4096, p = 4 (Δ_E = 2.8e-14: 1 strict discordance).

**Consequence for G0 (lead's decision, SPEC changelog needed).** As written, P-Bern requires Δ_O = Δ_E = 0 exactly. In `pbern_ref_{small,4096}.json`, Δ_E ≠ 0 in 9 of 27 cells: 7 of 9 at β = 10 (all except m = 256, p ∈ {4, 8}) and 2 at β = 3 (m = 4096, p ∈ {2, 4}). Δ_O ≠ 0 in 4 cells, all at β = 10: (m, p) = (1024, 2), (4096, 2), (4096, 4) and (4096, 8). This is a float64 artefact of the plant. Two options are coherent:
- **Option (a):** make the P-Bern criterion for Δ_O/Δ_E "zero strict discordances" (`E_n_disc_strict = 0`, `delta_O_strict_disc = 0`), with ties reported separately, and build D_B with the exactly monotone expm1 construction.
- **Option (b):** keep the general formula and use a tolerance of 1e-6 on Δ_O/Δ_E.

hrhv2_ref already reports the strict and tie parts separately so that either option can be checked. Under the "general" construction no float tolerance can make the strict-discordance count exactly 0.

## 7. SPEC v1.0 ambiguities encountered, and the choice made

(Also listed in the docstrings: ladder.py A1–A13, core.py, quotient.py, calibrate.py, geometric.py.)

**§0 Inputs**
1. *D formula.* SPEC writes D_ij = K_ii + K_jj − 2K_ij "(= 2 − 2K_ij after L2)", but these are not equal in float64. **Choice:** the general formula on every path, with the Gram made bitwise symmetric (upper triangle mirrored), D clipped at 0 and a zero diagonal. This choice changes P-Bern exactness (§6). Implementations must use the same formula, or set and tree keys can differ at near-ties.
2. *prh_clip threshold: interpolation and scope.* **Choice:** linear interpolation (torch/numpy default, as in PRH/ARH), computed on the (m, d) array passed in. If ARH/PRH receive a stacked (n, L, d) tensor they compute one threshold per model, and one per layer if they receive a list. `clip_threshold_A/B` overrides reproduce the stacked convention. **Check recorded:** matches PRH and ARH `remove_outliers(q=0.95)` to ≤ 1e-13 in both layouts. ARH's `prepare_features` clamps only; L2 comes later in the metric, which is the same order as SPEC.
3. *Near-tie fraction: which median, which k, which model.* **Choice:** median over the unordered off-diagonal D, strict "<", reported for every k in the grid and for A and B separately.
4. *"Diagonal is excluded everywhere" vs CKA.* Biased linear CKA uses the full Gram by definition, while Song's estimator removes the diagonal itself. **Choice:** the standard definitions.
5. *float32 features.* Everything is cast to float64 before preprocessing, so the clip threshold is computed in float64.

**§1 Structural**
6. *k_q rounding and k ≥ m.* **Choice:** round((m−1)/100) with Python's half-to-even rounding, then max(1, ·). No half case occurs for m ∈ {256, 1024, 4096, 16384} → 3, 10, 41, 164. Any k ≥ m gives NaN. An explicit `kgrid` is used as given; k_q is added only to the default grid.
7. *Per-anchor Kendall with ties.* "Ties counted as ½" does not say what happens to pairs tied in both rankings. **Choice:** Fagin K^(1/2): discordant = 1, tied in exactly one ranking = ½, tied in both = 0. This equals (1−τ_b)/2 only on tie-free rows. The scipy τ_b route that SPEC also allows gives *different* values on rows with ties, which matters on P-Bern. Both implementations must use the same rule.
8. *Spearman ties.* **Choice:** mid-ranks.
9. *MST algorithm.* SPEC says "dense Prim from node 0 with the tie key". The tie key is a strict total order, so the MST is unique and any correct algorithm returns the same tree. **Choice:** Kruskal as the primary, with Prim (full key) as a cross-check (`prim_eq_kruskal_*`). Note that SOG's Prim is not tie-key-consistent.
10. *Forest share: SPEC vs PREREG.* SPEC's `forest_share_A` is |F_A|/(m−1). PREREG PE2's "forest share of shared MST edges" is a different quantity, |T_A∩T_B∩F_A| / |T_A∩T_B|, and PE2 does not say whose forest. **Choice:** report all of them (`forest_share_A`, `forest_share_B`, `forest_share_shared_{A,B,union}`).
11. *s_T_bridge when T_A ∖ F_A = ∅.* **Choice:** NaN. Likewise `forest_share_shared_*` when T_A ∩ T_B = ∅.
12. *F_A ⊆ T_A "must hold exactly".* It does hold for the two SPEC keys: among equal D, both the kNN key and the MST key prefer the smaller neighbour index. Reported as a bool and true in all runs.
13. *Component degenerate flag.* Reported per k as `comp_degenerate@k`.

**§2 Geometric**
14. *Gs / G pair set.* **Choice:** unordered off-diagonal pairs. This gives the same rms and max as ordered pairs. ν is the mean of off-diagonal D (squared distances).
15. *"Raw geometry only when preprocessing is identical."* `ladder_ref` takes a single `pre`, so the condition always holds and G_* is always reported. The caller must not use G_* across different preprocessings.
16. *Which ARH "unbiased CKA".* ARH `CKAUnbiased` = hab / (sqrt(haa·hbb) + 1e-8). This differs from the exact Song CKA by about 6e-8 at m = 1024 and 4096, which is more than SPEC's 1e-8. ARH `estimators.cka_debiased` (U-centering) equals the exact value to 1e-9. **Choice:** `cka_debiased` = exact Song with no epsilon; `cka_debiased_eps8` (ARH class) and `cka_debiased_eps6` (PRH) are reported for the vendor matches.
17. *RSA on D or d.* SPEC says D (squared distances). **Choice:** D.
18. *SOG details* (all checked against the code; §5): use d, not D; Q over ordered positive entries; log clamp at 1e-12; kNN support uses the SPEC tie rule (SOG uses topk); MST support uses the SPEC key (SOG's Prim differs under ties); float64 (SOG uses float32); τ labels are `f"{τ:g}"`.

**§3 Quotients**
19. *Ties in x for the isotonic fit.* `scipy.optimize.isotonic_regression` takes a sequence and does not pool equal x, so its result depends on the order within ties. **Choice:** pool tied x into their weighted mean before PAV, so that g is a function of x. This gives the same result as scipy on tie-free data (≤ 6e-15).
20. *Row isotonic: SS0 and pair set.* **Choice:** for each anchor i, fit g_i over j ≠ i, giving m(m−1) ordered residuals. SS0 is taken over the same ordered pairs (= 2·SS0 over unordered pairs), so the explained fraction is consistent.
21. *Cross-fit hash.* "Fixed hash" is not specified. **Choice:** rank by sha256(f"{salt}|{item_id}") with salt `hrhv2_ref/crossfit/v1`; the first ⌊m/2⌋ form H1. Override with `H1=`. **§8 agreement requires both implementations to use the same H1, so pass it explicitly.**
22. *Knots for out-of-sample prediction.* **Choice:** all distinct training x values with their fitted values; linear interpolation between neighbouring knots and constant outside the training range. Block-level knots would give different predictions.
23. *Cross-fitting the row-isotonic quotient is ill-posed*, because g_i exists only for the anchor it was fitted on. **Choice:** for every anchor i, fit g_i on candidates j ∈ H1∖{i} and evaluate on j ∈ H2∖{i}.
24. *Cross-fit explained fraction.* **Choice:** relative to SS0 on the evaluation pairs. The scale and affine quotients are also fitted on H1 pairs and evaluated on H2 pairs.
25. *Key naming for directions.* `quot_{q}_{stat}` is A→B (x = D_A, y = D_B); `_BA` and `_mean` are added.

**§4 Projector**
26. *Normalisation when d_A ≠ d_B or the rank is deficient.* SPEC writes /(2d). **Choice:** /(r_A + r_B) with numerical ranks, which equals /(2d) when both have full rank d. NaN when d ≥ m or when the input is a Gram/distance matrix. The projector uses the preprocessed X (SPEC intends `pre="none"` for linear models). `cca_mean_sq` averages over min(r_A, r_B).

**§5 Calibration**
27. *What is permuted.* **Choice:** the rows of B (ARH `sim(X, Y[π])`); for Gram input, both axes.
28. *Order statistic.* **Choice:** ascending, 1-based, r = ⌈(1−α)(K+1)⌉ computed in exact rationals (K = 500 → 476; avoids float rounding at exact integer products).
29. *c_α ≥ s_max.* **Choice:** s_cal = 0. ARH clamps the denominator to 1e-12, which gives the same result unless s_obs > s_max.
30. *p-value.* **Choice:** (1 + #{s_π ≥ s_obs}) / (K + 1).
31. *Layer max.* **Choice:** maximum over all (A-layer, B-layer) pairs, with the same π for every B layer.
32. *Seed derivation* is not specified. **Choice:** PCG64 seeded with the first 8 bytes of sha256(f"{key}|perm|{k}"). `perms=` can be passed for §8 agreement.
33. *Discrepancy-type scores* (Δ, G, rms) do not fit s_cal with s_max = 1. SPEC is silent; the caller must wrap them (e.g. 1 − Δ). `quot_*_expl` is not a calibratable similarity (§4).

**§8 Agreement**
34. *Tolerance type.* **Choice in `compare.py`:** exact equality for set/graph/tree keys (including `delta_N@k`); |Δ| ≤ 1e-10·max(|a|, |b|) + 1e-13 for floats (an absolute floor is needed when values are exactly 0); |Δ| ≤ 1e-6 absolute for isotonic fits. Exact-class mismatches should be read together with the `near_tie_*` fractions.

## 8. Files

- Code: `hrhv2_ref/hrhv2_ref/{core,structural,geometric,quotient,ladder,calibrate,compare}.py`
- Tests: `hrhv2_ref/tests/` (56 tests); vendored ARH metric files in `tests/vendor_arh_785054b/`; SOG loader in `tests/sog_vendor_loader.py`
- Scripts: `hrhv2_ref/scripts/{pbern_scale_check,pbern_tie_diagnostic,pbern_tie_anatomy,scipy_crosscheck,sog_crosscheck,calib_bench,key_inventory}.py`, `run_chain{A_pbern,B_sog}.sh`
- Results: `hrhv2_ref/results/{pbern_ref_small,pbern_ref_4096,pbern_tie_anatomy,sog_crosscheck,key_inventory_m1024,scipy_crosscheck,calib_bench}.json`, `pbern_tie_diagnostic.txt`; superseded v0.1 P-Bern JSONs in `results/v0.1_superseded/`
- Logs: `~/hrh_v2_stage1_20261008/logs/hrhv2_ref_*.log`
- Reproduce: `cd ~/hrh_v2_stage1_20261008/hrhv2_ref && source ../venv/bin/activate && OMP_NUM_THREADS=8 python -m pytest -q tests`
