# E0 analysis plan, ADDENDUM 1 (implementation fix; no rule changes)

**Role:** `e0-run`, 2026-10-10 (UTC). **Extends:** `ANALYSIS_PLAN_E0.md` v1.0 (sha256 `e859c6ea52f941b18ca27b83150f489dded0085a4d99218d909b36c19603f0ee`) and `CODE_MANIFEST_E0.json` (sha256 `6b0111c85aa5416b1d7c97b7f75fe2fe413ce8e2741bdb4e852e43e707664b13`). **Affects:** plan §11 step 6, the evaluation write step only. **Logged in STATUS** before the guard (step 6), before any cross-model score with a sentence encoder exists. When this addendum was written, no sentence-encoder kNN cache, obs, eval or verdict file existed. The extraction of single-model embeddings was still running.

## 1. Defect
- In plan §11 step 5, the frozen `e0_calib.py` `main()` writes `results/E0_sentence/eval/E0_{cells,drawslopes,slopes,elasticity}_ctrold.parquet`. The guard whitelist (`e0_guard.EVAL_OLD_OK`) expects these files.
- In step 6, the frozen `e0_evaluate.py` `main()` also loads `obs_ctrold.parquet`, because the old pairs under `ctr` enter the secondary analyses. Its `write_outputs` then writes `E0_<table>_<arm>.parquet` for every arm, `ctrold` included, through `E.save_parquet_new`.
- `E.save_parquet_new` refuses to overwrite. The run therefore stops with `FileExistsError` on `E0_cells_ctrold.parquet` before `VERDICTS_E0.json` is written.
- The frozen tests did not catch this. The integration test calls `evaluate(..., write_dir=None)` with arm P only.
- **How it was found.** A pre-guard dry run, `addendum1/e0_dryrun_synthetic.py`, ran on synthetic new-pair obs. These were the 15 S1.1 vision-language obs rows with the language models relabelled as encoders; no sentence-encoder file was read. The dry run reproduced the calib-then-evaluate order in one directory.
  - Frozen path: `addendum1/dryrun_frozen_20261010T133608Z/dryrun_result.json` shows the `FileExistsError` traceback.
  - Patched path: `addendum1/dryrun_patched_20261010T133648Z/` completes and writes all 66 files.
  - The labels in these directories come from synthetic data and mean nothing.

## 2. Fix
- `addendum1/e0_evaluate_add1.py` imports the frozen `e0_evaluate` and replaces only `write_outputs`.
- When a target file already exists, the replacement reads it and requires it to be content-identical (`pandas.DataFrame.equals`, directly or after a parquet round trip) to the table the frozen code computed. It records the file and its sha256 under `VERDICTS_E0.json["addendum_1"]["reused_existing_files"]` and does not write the file again. Any difference raises an error and stops the run. Every other file is written exactly as by the frozen code. That includes `E0_levels_ctrold.parquet`, which calib does not write.
- Its `main()` is the frozen `e0_evaluate.main()` line for line. The only addition is the `addendum_1` record in `VERDICTS_E0.json`.

## 3. What does not change
- Every statistic, rule, threshold, label, table and output name stays the same. All of them are computed by the frozen `e0_evaluate.evaluate`, `e0_rules` and `s11copy` code. The guard, `e0_obs` and `e0_calib` are used unchanged.
- The confirmatory labels P1, P2b and P2c cannot be affected, because the defect sits only in the file-writing step after all computation.
- Per the plan header, every output this addendum touches carries its number. That means `VERDICTS_E0.json` (field `addendum_1`) and the eval tables written by the wrapper.

## 4. Files (chmod 444; sha256 in the STATUS line that logs this addendum)
- `analysis/e0_sentence/ANALYSIS_PLAN_E0_ADDENDUM_1.md` (this file)
- `analysis/e0_sentence/addendum1/e0_evaluate_add1.py`
- `analysis/e0_sentence/addendum1/e0_dryrun_synthetic.py`
