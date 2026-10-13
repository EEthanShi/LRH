# LLaMA axis, implementation I (role `llama-axis-run-indep`): run notes

Written before any LLaMA-axis cross-model score is computed by this role. This file adds no definition. All definitions, validations, decision rules and predictions come from the frozen files: `analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md` (990428e3…), `PARTB_PREDICTIONS.md` (f9afa919…), `analysis/s12_layermax/ANALYSIS_PLAN_LAYERMAX.md` (dd189721…) and `analysis/s12_layermax/indep/PLAN_INDEP.md` (085746a8…). Where this file and those disagree, those win.

## Code

**Copies** (plan §3, "Implementations"). Each copy changes only model list, paths and family label. Diffs against the originals are stored next to the copies as `*.diff`.

| copy | original | change |
|---|---|---|
| `common_llama.py` | `analysis/s12/common.py` | LMS = 3 huggyllama models (family `LLaMA`); LMS_POOLED = 8 S1.2 LMs + llama-13b/30b; feature path `features/wit1024_llama_axis/` for llama-13b/30b; their clip threshold read from `partB/clip_thresholds_partB.json` |
| `lmx_common.py` | `analysis/s12_layermax/indep/lmx_common.py` | imports `common_llama`; OUT = `results/S1_2_layermax_llama/indep`; ARH ranks of llama-13b (10) and llama-30b (11); pooled LM list; path of the frozen 8-LM indep outputs |
| `run_layermax.py` | same name | docstring only |
| `validate.py` | same name | I2 pairs = plan V-L3 pairs; I3 rows restricted to llama-7b (= V-L4a) |
| `lmx_aggregate.py` | same name | none (byte-identical). Only `pair_stats` / `subsets` / `reading_arh` are imported from it, and its `main()` is not run. |

**New files.** They implement plan items that the 8-LM code does not cover. No definition is changed.
- `vl_checks.py`: V-L1 summary, V-L2, V-L4b, V-L7.
- `vl6_arh.py`: V-L6, adapted from `validate.i4`.
- `lmx_llama_aggregate.py`: the plan's §3 (iii) added subsets and (iv) calibrate-then-max; §5 trend statistics; §6 diagnostics; §7 readings; the Part B verdicts P-L1–P-L5 and P-L2a with their deductive checks; §10.
- `lmx_llama_report.py`: writes `REPORT.md` from `summary.json` only.
- `cross_validate.py`: V-L5 against the early implementation, written only after this role's own `summary.json` and `REPORT.md` exist.

## Run

- **Output.** Panel = 3 LLaMA × 17 ViTs × {prh_clip+l2, l2}. Output is `results/S1_2_layermax_llama/indep/<pre>/pairs/<lm>__<vit>.npz`, holding the full observed and K = 500 null L_A × L_B matrices for mKNN counts, biased CKA and debiased CKA.
- **Permutations.** Same keys as the plan: `S1.2-layermax|<lm>|<vit>`.
- **Resources.**
  - At most 3 BLAS/torch threads per process and at most 2 concurrent processes for this role (plan §11 caps all llama-axis processes at 6 threads, and E may run at the same time).
  - GPU memory fraction 0.30 per process. No other job is touched.
- **Order** (plan §4).
  - The main run starts first.
  - V-L1–V-L4 and V-L7 run on the stored matrices, and V-L6 runs alongside.
  - No main-run output other than completion status is read until V-L1–V-L7 pass.
  - The early implementation's new outputs (`results/S1_2_layermax_llama/early/`) are not opened until this role's `summary.json` and `REPORT.md` are written.
- **Pooled 10-LM panel.**
  - The 7 non-LLaMA LMs are read read-only from the frozen `results/S1_2_layermax/indep/<pre>/pairs/` (same keys and definitions).
  - llama-7b comes from this run, after V-L4b has shown it reproduces the frozen file.
