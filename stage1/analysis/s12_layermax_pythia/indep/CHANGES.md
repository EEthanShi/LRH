# Implementation I of the Pythia axis: changes against the OpenLLaMA-axis I code (S0b, written and tested, NOT FROZEN)

Sub-role `pyt-code-I`, step S0b of `analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md` (sha256 `6de89603…879e7b`, sections 3, 4, 5, 7, 8, 9, 11). Written 2026-10-10 between about 01:05Z and 02:20Z, before any Pythia byte exists on ds2213. Not frozen: the single `CODE FROZEN` line of plan section 11 covers all code of the axis and is not written by this sub-role.

**Independence (plan 3).** This sub-role wrote implementation I only. It never opened the E copy (`early/`) or the R copy (`review/`); it saw E and R process command lines in `ps` while waiting for thread tokens. It read: the frozen plan and `CRITIQUE.md`; the OpenLLaMA originals (`analysis/s12_layermax_openllama/indep/*`, and as references `guard.py`, `VERDICTS_SPEC.md`, `verdict_keys.py`, `compare_verdicts.py`, `S0B_NOTES.md`); the launcher `launch_pyt.sh` and the header of `extract/pythia_axis/pyt_axis_common.py`; and, after the infrastructure role logged them at 01:41:15Z, the shared contract and tools `VERDICTS_SPEC_PYT.md` (`2d5332eb…`), `verdict_keys.py` (`6d20e04d…`), `guard.py` (`a66a5711…`, interface and test mode), `compare_verdicts.py` (`86c762ca…`, CLI), `CHANGES_INFRA.md`, `s0b_dryruns_infra.json`, and the field names of the stand-in Part B data `s0b_scratch/PB/partB_vp9/partB_features_pyt.json`. It did not open `partb_pyt.py`, `partb_checks_pyt.py`, `forecast_pyt.py` or `feasibility_pyt.py`, so its own application of the P-P2b rule and of the plan 7 category (written as `x|I|` extras) stays independent of `forecast_pyt.py`.

**Two versions.** A first version (01:05Z–01:50Z) used I's own key names, because no contract existed. When the contract appeared, I was rewritten to it (01:50Z–02:20Z): key names and definitions of sections A, B, C, H, R, D, G; Part B field names (spec 11); test root `s0b_scratch/I/` (spec 13); the shared comparator for V-P5 and V-P8. The first version's test outputs were moved, not deleted, to `s0b_scratch/I/pre_contract_s0b_test_20261010T015516Z/` (with its upload backups), and its fixture builder to `s0b_scratch/I/superseded_code_20261010T015516Z/`.

## 1. Conventions

- `orig_copy/` holds pristine copies of the 12 OpenLLaMA-axis originals (`analysis/s12_layermax_openllama/indep/`, sha256 equal to plan section 3 and to `S0B_CODE_MANIFEST.json`), chmod 444, plus their `CHANGES.md` as `CHANGES_ol.md`.
- `X.diff` next to each copy is `diff -u --label orig_copy/<original> --label X orig_copy/<original> X`. Changed lines carry `PYT-AXIS`; files rewritten in large parts (`pyt_verdicts.py`, `lmx_pyt_aggregate.py`, `lmx_pyt_report.py`, `cross_validate.py`) mark blocks and say so in their headers.
- **Order gate** (`pyt_gate.py`, plan 4 F1(a)): every script that reads a pair file or a cross-model score calls `pyt_gate.gate()` first. Outside test mode it exits 3, before importing any data module and before writing anything, unless (i) a `[pyt-axis-<role>] CODE FROZEN` milestone line exists, (ii) a later `[pyt-axis-predict] Part B FROZEN` milestone line exists, (iii) that line names the sha256 of `PARTB_PREDICTIONS_PYT.md` and `partB_features_pyt.json` and both files hash to them, (iv) the output root is a guarded directory, and (v) the shared `guard.require_partb_frozen(out_root)` passes (called whenever `guard.py` exists, in test mode too).
- **S0b test mode**: `PYT_S0B_TEST=1` and no `CODE FROZEN` line; outputs only under `analysis/s12_layermax_pythia/s0b_scratch/I/` (spec 13; never under `results/`). After `CODE FROZEN`, `PYT_S0B_TEST=1` exits 3. In test mode the 7 Pythia slugs become `synth__pyt-<size>` stand-ins (the 5 BLOOMZ, llama-13b, llama-30b in size order, plan V-P9 / spec 13) and the Part B inputs default to the shared stand-in `s0b_scratch/PB/partB_vp9/`.
- **No-delete rule**: an output that would be replaced is renamed `<name>.prev-<UTC>` (`lmx_common.keep_aside`); `run_layermax.py` never resumes an interrupted run: `--restart` keeps the old `<pre>/` aside and starts again (plan 11, B6.5). Code uploads that replaced an earlier version moved it to `s0b_scratch/I/prev_code/<UTC>/`.
- Entry scripts set `sys.dont_write_bytecode`: no `__pycache__` in the code tree. Output root `results/S1_2_layermax_pythia/indep/`. Threads `PYT_THREADS` (launcher). GPU memory fraction 0.30.
- `s0b_dryruns_I.json`: I's 18 gated commands in the `[[label, script, args...]]` format of `s0b_freeze.py --dryruns`.

## 2. Files

| copy | original (sha256 prefix) | what changed |
|---|---|---|
| `pyt_gate.py` | `ol_gate.py` (`2a55dce4`) | The order gate above (own checks plus the shared guard); test mode and test root. |
| `common_pyt.py` | `common_ol.py` (`2b253cd9`) | Run panel = 7 Pythia (size order), `features/wit1024_pythia_axis/`, clip source = frozen Part B `clip_thresholds_partB_pyt.json`; the 4 V-P4 known models and the 11 pooled LMs with their feature dirs and frozen clip files; ViT families of spec 1 (in21k, MAE, DINOv2, CLIP); S-E2 removed; test stand-ins. |
| `lmx_common.py` | `lmx_common.py` (`d0cc9b81`) | Registry `common_pyt`, output root from `pyt_gate`, frozen I roots (8-LM, LLaMA, OpenLLaMA) for V-P4 and section D, size ranks, `PYT_THREADS`, `keep_aside` / `write_json`. |
| `run_layermax.py` | `run_layermax.py` (`dee3a805`) | Gate first; 119 pairs; `--known` (V-P4: 4 known pairs → `indep/known/<pre>/pairs/`); `--restart`; `--device cpu` and a non-default `--K` only in test mode; S-E2 removed. Kernel arithmetic unchanged. |
| `validate.py` | `validate.py` (`08f5d041`) | Gate first; I2 = V-P3 pairs (6.9b × DINOv2-g, 6.9b × CLIP-H ft-in12k, 70m × ViT-Ti, 70m × MAE-H; first/middle/last × {id, π0, π1, π499} × both pre); V-P7(b); I3 (V-O4a) dropped; test options. |
| `vl_checks.py` | `vl_checks.py` (`ee34896e`) | Gate first; V-P1 (24 models), V-P2 (+V-P2b on 119 pairs), V-P4 (replaces V-O4b: the 4 known reruns vs their frozen I files; strict 1e-12 relative decides, the V-P5 mixed form is reported too), V-P7(a) on 119 pairs (PR from Part B `per_pre[pre]["PR"]`), V-P7(c), new V-P7(d) (dCKA second moment, descriptive, entry- and pair-level MC errors). |
| `vl6_arh.py` | `vl6_arh.py` (`e75f5fb3`) | V-P6 on pythia-6.9b × ViT-Ti: scalar (mKNN, biased CKA) and `calibrate_layers` on grid_layers(33,5) × grid_layers(12,5); `--scalar-only` (plan 11 budget rule). |
| `lmx_aggregate.py` | `lmx_aggregate.py` (`60170b72`) | Library copy only: gate, then refusal (exit 2) when run directly; covariates guarded (no ARH rank for Pythia). Not used for any Pythia verdict. |
| `pyt_verdicts.py` | `ol_verdicts.py` (`21515722`) | Generic n-size trend with exact n! enumeration; common-ViT-set rule and legacy mode; ρ:=0 sensitivity; ViT bootstrap (one `rng.integers(0, n, n)` per draw); OpenLLaMA sections A–C kept as legacy functions (V-P8); spec sections A, B, C, H, R; I's own category, modifiers, readings and P-P2b as `x|I|` extras. |
| `lmx_pyt_aggregate.py` | `lmx_ol_aggregate.py` (`90796c04`) | Pair records unchanged (exact mKNN), `muM` (entry null mean at the M_unc-argmax) and `n_lm_layers` added; reads Part B by the spec 11 names; writes `verdicts_pyt.json` (schema `PYT-VERDICTS v1`, every key of `verdict_keys.required_keys("EI")`, exit 1 if one is missing) and `summary.json`; spec sections D and G; every l2 key null if l2 was not run; `--legacy` (V-P8 file, schema `OL-VERDICTS v1`); test fixture. |
| `lmx_pyt_report.py` | `lmx_ol_report.py` (`b60a5ea4`) | REPORT.md with the Pythia labels (H1 gatekeeper, Holm secondaries, everything else descriptive; I's own category marked as a cross-check of `forecast_pyt.py`). |
| `cross_validate.py` | `cross_validate.py` (`ab3a4b19`) | V-P5 from the I side, per pre: pair level (counts and the names of failing checks only), verdict level through the shared `compare_verdicts.py --sections EI` (and `FK` on the forecast files), clip thresholds; order file before E is opened. |
| `vp8_legacy.py` | new | V-P8 (a) the aggregator's legacy mode (n = 3) vs the frozen OpenLLaMA summary, by I's comparator and by the shared `compare_verdicts.py --legacy`; (b) common-set changes with explanations; (c) BLOOMZ n = 5 trends and exact p to 1e-12. S0b test mode only. |
| `tests/test_pyt_verdicts.py` | new | V-P9 known answers (pattern table, Holm incl. order, IUT, TOST, P-P2b, category table, synthetic end-to-end incl. "every contract key present", trend vs OpenLLaMA and scipy, band constants, common-set rule, censoring counterexample). |
| `tests/test_gate_dryrun.py` | new | Dry runs of every entry script without the gates (exit 3, no file changed) and unit tests of the gate on a synthetic STATUS file. |
| `tests/test_cross_report.py` | new | V-P5 comparator on identical / perturbed E files (no value written or printed); report generation and keep-aside. |
| `tests/test_forecast_xcheck.py` | new | I's own P-P2b and category (`x|I|` extras) against `forecast_pyt.py --impl I` run as a black box on the same I output (names only). |
| `tests/cpu_smoke.sh`, `tests/run_all_s0b.sh` | new | CPU smoke of the run path on the 4 known pairs (K = 8); one driver for all S0b tests (waits for free thread tokens). |
| `s0b_dryruns_I.json` | new | I's gated commands for `s0b_freeze.py --dryruns`. |

## 3. Verdict keys

`verdicts_pyt.json` carries exactly the keys of `verdict_keys.required_keys("EI")` (sections A, B, C, H, R, D, G of `VERDICTS_SPEC_PYT.md`), with the spec's definitions, plus extras that are never compared:
- `x|tr_ci_lo|…`, `x|tr_ci_hi|…` (ViT bootstrap of Tr, key `S1.2|bootvit|layermax-pythia|<p>|<m>|<q>|<x>`); the per-ViT ρ vectors go to `summary.json`;
- `x|I|K|<p>|<x>|…`: I's own category, premise, F reason, τ and δ_b curves, M1–M5, H2/H3/H4 readings (the plan 7 rules applied by I; the contract's section K is written by `forecast_pyt.py` and can be compared with these);
- `x|I|PP2b|<p>|<x>|…`: I's own application of the P-P2b rule (decisions per comparison, totals, verdict, h curve);
- `x|I|vis|…` (vision axis, S1.2 `vision_trend`), `x|I|partb_xcheck|…` (Part B's stored decisions re-derived from its own tables), `x|I|D7|…` (Part B D7 passed through).

## 4. Operationalisations, plan gaps, open points

1. **Contract choices followed** (marked `[op]` in the spec): thresholds compared without tolerance; relabelling counts with 1e-9; bootstrap counts with 1e-9 on the conservative side; `|unc` key suffix; one `rng.integers(0, n, n)` per draw; Holm ties broken H2, H3, H4; "exactly 0" Δρ decided in rationals; per-ViT sign rule and D3 applied only to ViTs without censoring ties.
2. **Per-ViT sign rule (plan 8) is not a theorem under censoring** (found independently by I and by the infrastructure role): a raw-discordant pair tied at s_cal = 0 raises ρ_v although no comparison turned concordant (I's test t10: ρ_raw 0.964 → ρ_scal 0.982). The contract excludes such ViTs (`nexcl_tie`); I follows it.
3. **B4 boundary.** |Tr| = 0.75 (Σd² = 14) is attainable exactly and evaluates to 0.7499999999999999 in float64 (seen on the first-version stand-in fixture's B_late). The contract puts a 1e-9 tolerance on it in Part B; I's `x|I|partb_xcheck` uses the same rule.
4. **Ambiguities left open by the contract, I's reading**: (a) `signvit`: a ViT whose thresholds are exactly constant over the 7 sizes is counted in both `nmono_up` and `nmono_down`; (b) `nV` of H1/H2 is 0 when the common set is empty, while every TOST field (H3, H4 included) is `null` (spec 6 "null for all if V is empty"); (c) `dtr_nv` is 0 when V is empty. These only matter in degenerate data.
5. **V-P4 tolerance.** The plan's "CKA ≤ 1e-12 relative" is decided strictly; the CPU smoke (CPU rerun vs frozen GPU files) gives 1.3e-9 relative on debiased-CKA null entries near 0 and passes only in the V-P5 mixed form. On the GPU at S6 the strict form is expected to hold, as on the OpenLLaMA axis.
6. **V-P7(d)** compares with 2/(m(m−3)); on the 4 known pairs (K = 8, CPU smoke, exploratory) the mean ratio was 0.91 (prh_clip+l2) and 0.80 (l2), within about 0.5 and 2.5 pair-level standard errors of 1. Descriptive, as the plan says.
7. **Not implemented in I** (as on the OpenLLaMA I side): plan 6 "as OP§6" A1–A3, null profiles and leave-layers-out; D7 is passed through from Part B.
8. **E pair-file format.** Part 1 of `cross_validate.py` reads E's per-pair JSON/NPZ by the field names of the OpenLLaMA E copy (`summaries`, `calib_then_max`, `mknn.null_counts_int16`, `mknn.entry_obs`); the contract does not fix them. If E's Pythia copy renamed them, Part 1 reports "missing", not a mismatch.

## 5. Order of use (plan 11; every command through `launch_pyt.sh`, after the gates)

1. **S6** (after G1): `run_layermax.py --pre prh_clip+l2 --known` and `--pre l2 --known` (GPU, V-P4 reruns); `vl_checks.py --items vp4,vl7c`.
2. **S7**: `run_layermax.py --pre prh_clip+l2` (GPU); `vl_checks.py --pres prh_clip+l2 --items vl1,vl2,vl7,vl7d`; `validate.py --pres prh_clip+l2`; `cross_validate.py --pres prh_clip+l2 --parts 1` (counts and names only); the l2 go/no-go line in STATUS; if go, the same for `l2`.
3. **S8**: `vl6_arh.py [--scalar-only]` (alone, ≤ 4 threads, GPU); `lmx_pyt_aggregate.py` (or `--pres prh_clip+l2` when l2 was not run); `forecast_pyt.py --impl I` (infrastructure); `cross_validate.py --pres <pres run> --parts 1,2,3`; `lmx_pyt_report.py`.

## 6. S0b tests (test mode, CPU only, known or synthetic data, not citable)

Every launcher run is listed with its log in the role report; logs and end markers are in `s0b_scratch/I/logs/` (contract version) and `s0b_scratch/I/pre_contract_s0b_test_20261010T015516Z/logs/` (first version). No GPU process was started.
