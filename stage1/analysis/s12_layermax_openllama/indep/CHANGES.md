# Implementation I of the OpenLLaMA axis: changes against the LLaMA-axis I code (S0b)

Role `oll-axis-extract`, step S0b of `analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md` (sha256 `4f18d710…`, sections 3, 4, 11). Written before any OpenLLaMA-13B byte existed on ds2213; hashed in the S0b `CODE FROZEN` STATUS line.

**Conventions.**
- `orig_copy/` holds pristine copies of the LLaMA-axis originals (`analysis/s12_layermax_llama/indep/`, chmod 444).
- `X.diff` next to each copy is `diff -u orig_copy/<original> X`. Copies are marked `OPENLLAMA-AXIS` in their headers.
- Output root: `results/S1_2_layermax_openllama/indep/` (`lmx_common.OUT`, from `ol_gate.out_root()`). Every script that reads a pair file calls `ol_gate.gate()` (= `guard.require_partb_frozen`) first (plan F1(a)).

| copy | original (sha256 prefix) | what changed |
|---|---|---|
| `common_ol.py` | `common_llama.py` (`ab40a7a1`) | Model list (OpenLLaMA 3b/7b/13b), 11-LM pooled list, 13B feature path, 13B clip source = frozen Part B, S-E2 source, S0b test hooks. |
| `lmx_common.py` | `lmx_common.py` (`e58b2dd5`) | Registry module, output path, ARH rank of 13B (8), LLaMA-axis indep path (R.3.7), `OL_THREADS`. |
| `run_layermax.py` | `run_layermax.py` (`c97444ae`) | Gate first; `OL_THREADS`; `--sens-e2`; smoke runs under `indep/smoke/`. |
| `validate.py` | `validate.py` (`c269a320`) | Gate first; I2 = V-O3 pairs, I3 = V-O4(a) rows of 3b/7b, V-O7(b) closed form; `--pres/--items/--pairs-root`. |
| `vl_checks.py` | `vl_checks.py` (`5b857a19`) | V-O1, V-O2, V-O4(b), V-O7(a), new V-O7(c). V-O4(b): I reruns vs the frozen I outputs at 1e-12 relative (counts, p, argmax identical); vs the frozen E outputs, p, argmax and null counts identical and floats in the V-O5 form (`pass_V-O4b_I`; the all-strict form is also reported). |
| `vl6_arh.py` | `vl6_arh.py` (`a156c7a0`) | V-O6 reduced (scalar calibrate on 13B last x ViT-Ti last; `calibrate_layers` on grid 41x5 by 12x5); gate first. |
| `lmx_aggregate.py` | `lmx_aggregate.py` (`288702b8`) | Unchanged import base (record builder of the 8-LM I run). |
| `lmx_ol_aggregate.py` | `lmx_llama_aggregate.py` (`bd82d645`) | Aggregation for `VERDICTS_SPEC.md` A, B, C, D, G; exact 11! enumeration; ΔΔ CI with key `S1.2-layermax-openllama|boot|arh_delta|<pre>|<metric>` (computed by I too, R.3.6). |
| `lmx_ol_report.py` | `lmx_llama_report.py` (`cca33054`) | Report of I from `summary.json` and the validation files; labels R.1.2/R.1.3. |
| `cross_validate.py` | `cross_validate.py` (`966ca52e`) | V-O5 from the I side (pair level, verdict level EI/F/J, clip E vs I); order recorded before any E file is opened. |
| `ol_gate.py` (new) | — | Output root and Part B gate of I. |
| `ol_verdicts.py` (new) | — | Pure verdict arithmetic of `VERDICTS_SPEC.md` (mKNN as `Fraction`). |

**S0b tests (scratch `~/oll_s0b_scratch/`, test mode only, not citable).**
- T1 (LLaMA-axis I pair files relabelled): every trend, ΔTr and p of the LLaMA-axis I summary reproduced; `G|llama_pairs_equal_merged_summary` true.
- T2 (frozen 3b/7b + llama-30b): pooled ΔΔ CIs equal the LLaMA E values (same key); `rerun = frozen` true.
- GPU smoke (`run_layermax.py --only`, stand-in 13B x 4 ViTs), V-O1/V-O3/V-O7(b) on the smoke outputs: pass.
- Integration fixture X: E vs I 0 mismatches over 5233 + 452 + 24 keys; `cross_validate.py` pair and verdict level pass (clip part missing in the fixture).
