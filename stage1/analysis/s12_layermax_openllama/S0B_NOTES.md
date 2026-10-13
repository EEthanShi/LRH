# OpenLLaMA axis, step S0b: completion notes (role `oll-axis-extract`, resumed instance, 2026-10-09 ~18:50-19:25Z)

Plan: `ANALYSIS_PLAN_OPENLLAMA.md` (sha256 `4f18d710…`), section 3 "Code is hashed before any new data exists", section 4 F1, section 11 step S0b. Hashed in the S0b `CODE FROZEN` STATUS line together with all other code.

## State found at resumption

- All code had been written between 09:50Z and 10:27Z by the previous instance of this role and its sub-roles E, I, R, PB; no `CODE FROZEN` line existed; nothing of S1-S4 had run.
- Three GPU smoke steps of that instance are in `results/S1_2_layermax_openllama/logs/gpu_ledger.tsv` (1.161 GPU-min in total).
- Two files of sub-role I (`indep/cross_validate.py`, `indep/lmx_ol_report.py`), the last edit of `feasibility_ol.py` and the pristine originals of I and R existed only in the local working copy; they were uploaded unchanged.
- The R tests of the previous instance compared R with mock copies of R's own output; E, I, R, forecast and joint had not been compared on one common input. That test was added (below).

## Changes made at completion (all before the CODE FROZEN line)

1. `guard.py`: `_stat_rows` took the birth time from `find -printf %W`, which GNU find 4.9.0 on ds2213 does not support, so `scan()` and `audit()` failed. It now uses coreutils `stat -c %W`. A file whose birth time is unknown fails the F1(c) audit.
2. `f1d_outroots.py`: the three configuration modules are imported each in its own subprocess. In one process, `analysis/s12/common.py` (imported by E's `lm_core`) shadowed `extract/common.py`, so the extraction module failed to import and the failure was not counted. An import failure is now a violation.
3. `s0b_freeze.py`: files with an execute bit become 555 instead of 444, so `launch_ol.sh` stays executable.
4. `indep/lmx_aggregate.py` (library copy of the 8-LM I aggregator): when run directly it now calls the Part B gate before anything else, and its thread default is `OL_THREADS` (was 8).
5. Added: `early/CHANGES.md`, `indep/CHANGES.md`, `review/CHANGES.md`; diffs `X.diff` against `orig_copy/` for every E, I and R copy; `pb_orig_copy/` with `partb_ol.py.diff` and `partb_checks_ol.py.diff`; this file.
6. `feasibility_ol.py` ran on known files only: `results/S1_2_layermax_openllama/feasibility/`, 87 plan numbers compared, 0 differ at plan precision.

## Tests (S0b test mode, outputs only under `~/oll_s0b_scratch/`, not citable)

- **Part B** (`partb_ol.py`, both stand-ins) with the fixed guard: scans clean at start and end; recomputation of the 27 known models passes (PR ≤ 3.3e-15 relative, clips bitwise); `partb_checks_ol.py` passes.
- **Integration fixture X** (`~/oll_s0b_scratch/X`): 3b/7b/13b ← llama-7b/13b/30b pair files of the LLaMA axis, one Part B made by `partb_ol.py`; E aggregate, I aggregate, `forecast_ol.py` E and I, `joint_summary.py` E and I, E `vl5_e_vs_i.py`, I `cross_validate.py`, both reports and R all run on it.
  - E vs I: 5233 EI keys, 452 F keys, 24 J keys, 0 mismatches; pair level (counts, p, argmax, floats) pass.
  - R-E vs E and R-I vs I: 0 mismatches except `G|rerun_equals_frozen_3b7b`, a fixture artifact (R reads the fixture's stand-in frozen files, E and I the real frozen 3b/7b files). R-E vs R-I: 0 mismatches.
  - The OpenLLaMA-axis trends of this fixture equal the LLaMA-axis values (e.g. ΔTr(s_cal) −1.529).
- **GPU smoke G** (stand-in 13B = llama-13b features × ViT-Ti, both pre): E `precompute.py` cpu phase (V-O1/V-O2 on the subset) pass; E `validate.py` V-O3 entries ≤ 5e-16, mKNN exact; E `run_layermax.py` observed matrices bitwise equal to the LLaMA-axis E file of llama-13b × ViT-Ti; E vs I on the same key: mKNN observed and null counts identical, CKA differences ≤ 1.1e-15.
- **GPU ledger after all S0b tests:** 2.238 GPU-min over 7 steps (budget 60 for the whole axis).

## Operational notes for the later roles

- **V-O4(b)** is read "E vs `results/S1_2_layermax/pairs/`, I vs `results/S1_2_layermax/indep/`" (each against its own frozen outputs, 1e-12 relative). I also compares itself with the frozen E files in the V-O5 form; its log line prints the all-strict form (`pass=False`, from debiased-CKA null means near 1e-6 with absolute differences near 1e-18). The decision key is `pass_V-O4b_I`.
- **E2:** the lead's task text asks this role to STOP before downloading if the re-extraction is not bitwise equal. That is stricter than plan section 2 (≤ 1e-6 accepted, with S-E2); this role follows the stricter rule.
- **Byte total:** the 3 shards sum to 26,031,877,079 B; the plan's 26,032,458,406 B is the total of all files at the pin. S1 checks both.
- **Weights:** the pin has only `.bin` pickles (no safetensors exist, plan section 1); they are loaded only through `torch.load(weights_only=True)` (plan section 2), which `extract_core_ol.WeightsOnlyGuard` enforces.
- **Order inside S6/S7:** V-O4(b) and V-O7 read main-run pair files, so they run after the main runs of their implementation and before any main-run output is opened for a verdict.
