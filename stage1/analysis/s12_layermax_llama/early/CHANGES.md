# Implementation E for the LLaMA axis: changes against the early layer-max code

Role `llama-axis-run-early`, 2026-10-08. Plan: `analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md` §3 "Implementations" (copy of the early role's code, changed only in model list, paths and family label), plus the protocol additions that plan §3 (iii)–(iv) and §4 require.

- **Originals** are in `analysis/s12_layermax/`. They are unchanged.
- **Pristine copies** are in `early/orig_copy/`, chmod 444, so that `diff -u orig_copy/X X` shows every change.
- Every changed line carries the marker `LLAMA-AXIS`.
- The sha256 values of the originals and of the final copies are in `STATUS.md` and `results/S1_2_layermax_llama/early/hashes.json`.

## Copied and changed

### `lm_core.py` (orig a6b8851b…)

- **Output paths:**
  - `RES` → `results/S1_2_layermax_llama/early`, `CACHE` → `RES/cache`;
  - new read-only constants `FROZEN_RES` (`results/S1_2_layermax`) and `PARTB`.
- **Model list:** `LMS` = llama-7b, llama-13b, llama-30b, with family `LLaMA`. `VITS` is unchanged.
- **Feature loader.** New functions `_feat_dir`, `sidecar`, `load_stack`, `n_params` and `n_stored`.
  - llama-13b and llama-30b are read from `features/wit1024_llama_axis/`; every other model from `features/wit1024/`.
  - These replace the three `C12.load_stack(stem, "wit1024")` calls (in `clip_c`, `layer_rows`, `ModelGPU`) and `C12.n_stored`.
- **`s12_clip_c` reference values:**
  - llama-13b and llama-30b: the frozen Part B value (`clip_thresholds_partB.json`), because S1.2 has no cache entry for them;
  - every other model: the S1.2 cache, as before.
- **`subsets`:** adds the plan §3 (iii) descriptive subsets `noLMemb`, `LMdeep` (relative depth ≥ 0.5) and `LMemb_only`, each over all ViT layers. They use the same LM-layer sets as `posthoc_null_structure.lm_sets`. The existing subsets are unchanged.
- **`setup_torch`:** optional cap `LLAMA_AXIS_THREADS`, so that concurrent llama-axis processes stay at or below 6 threads in total.
- **Unchanged:** the kernels `gpu_layer`, `ModelGPU`, `hsic_unbiased_gpu`, `pair_scores` and `run_pair`; the permutation key `S1.2-layermax|<lm>|<vit>`; K = 500; the ARH rank list.

### `precompute.py` (orig e8f2cdc3…): S4, V-L1 and V-L2

- Uses the new loader.
- **V-L1** writes the shared cache `results/S1_2_layermax_llama/cache/clip_thresholds.json` if it is absent; otherwise it compares against it bitwise.
- **V-L2** also compares the final tables of llama-7b and the 17 ViTs with the frozen `results/S1_2_layermax/cache/knn_final/`.
- Tasks are sorted longest-first.

### `run_layermax.py` (orig b6490719…): S7

- The pair list comes from `lm_core` (51 pairs).
- **Protocol (iv), calibrate-then-max**, is added as `calib_then_max(E)`, built from the existing entry-wise statistics, and stored in `rec["calib_then_max"][metric]`.
  - M = max_uv max((S_uv − c_uv)/(1 − c_uv), 0), with its argmax, ties, and the number of positive entries.
  - M_unc = max_uv (S_uv − E0[S_uv]).
- The log line shows completion status only (plan §4), so the two summary lookups that fed it were removed.
- No existing field changes; V-L4(b) checks this bitwise.

### `validate.py` (orig 83524c39…) → V-L3 and V-L4(a)

- **V-L3:**
  - the 12 layer pairs fixed in plan §4;
  - permutations {identity, π_0, π_1, π_499};
  - both pre, all 12 cases. The original used π_0 and π_1, and only a third of the cases under `l2`.
- **V-L4(a):** the llama-7b fixed-depth entries vs the stored S1.2 values.
- The S1.2-key single-entry calibration check keeps the original llama-7b case.

### `v4_arh.py` (orig 74fdbd33…) → V-L6

The plan's three cases:
- scalar `calibrate()` on (llama-30b last, ViT-Ti last), for mKNN and biased CKA;
- `calibrate_layers(agg="max")` with biased CKA on the full llama-13b × ViT-Ti grid;
- `calibrate_layers(agg="max")` with mKNN on llama-30b `grid_layers(61, 9)` × all ViT-Ti layers. This is compared against E's exact sub-grid maximum (`run_layermax.subset_summary`), because I does not exist yet.

The comparison logic is the original's.

### `launch_step.sh` (orig dcb82ee2…)

- **Paths:** `logs/llama_early_<NAME>.log` and the markers `results/S1_2_layermax_llama/early/<NAME>.{done,failed}`.
- **Threads:** capped at 6, and also exported as `LLAMA_AXIS_THREADS` and `VERIF_TORCH_THREADS`.

## Copied but not used

The copies `aggregate.py`, `report.py` and `posthoc_null_structure.py` were removed from `early/` and remain in `orig_copy/`. Their trend and report code is tied to the 8-LM/BLOOMZ layout, so the following new files replace them.

## New files

- **`vl4b_llama7b.py`:** V-L4(b). Compares the llama-7b layer-max outputs of E field by field and array by array against the frozen `results/S1_2_layermax/pairs/`.
- **`vl7_prop5.py`:** V-L7. Compares the entry-wise null means with Proposition 5: biased CKA against the Part B PRs, debiased CKA against 0, and mKNN against 10/1023.
- **`aggregate_llama.py`:**
  - computes Tr_L (the S1.2 `lm_trend` arithmetic with family LLaMA, self-checked against `A12.lm_trend` on the BLOOMZ rows), ΔTr_L, Tr_BL, the pooled ARH-style r on 10 LMs (exact 10! p, ViT bootstrap), the vision axis and the diagnostics of plan §6;
  - evaluates every Part B prediction rule and runs the deductive checks;
  - writes `results/S1_2_layermax_llama/early/summary.json`.
- **`report_llama.py`:** writes `REPORT.md` from `summary.json` and the validation JSONs.

## Added after the resume (2026-10-09, same role, new agent)

- **`vl5_e_vs_i.py`:** V-L5, E vs I, run once I had finished its main run. It reads I's pair files read-only and summarises them with I's own per-pair code (`lmx_aggregate.pair_stats`, `lmx_llama_aggregate.pair_stats_ext`, imported read-only; nothing of I is written). Output: `results/S1_2_layermax_llama/early/validation/VL5_E_vs_I.json`. The first run compared entry-wise p as floats and is kept as `VL5_E_vs_I.attempt1_entry_p_float.json`.
- **`report_llama.py`:** now reads the V-L5 JSON; the scope paragraph is updated to match.
- **`interpretation.txt`:** section 0 of REPORT.md, written after the numbers were seen and labelled as such.
- `aggregate_llama.py` is unchanged from the version written before the cut-off (sha256 d4de0157…). It was first run after the resume.
