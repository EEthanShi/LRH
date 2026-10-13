# Implementation E of the OpenLLaMA axis: changes against the LLaMA-axis E code (S0b)

Role `oll-axis-extract`, step S0b of `analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md` (sha256 `4f18d710…`, sections 3, 4, 11). Written before any OpenLLaMA-13B byte existed on ds2213; hashed in the S0b `CODE FROZEN` STATUS line.

**Conventions.**
- `orig_copy/` holds pristine copies of the LLaMA-axis originals (`analysis/s12_layermax_llama/early/`, chmod 444).
- `X.diff` next to each copy is `diff -u orig_copy/<original> X`.
- Changed or added lines carry the marker `OL-AXIS` (`aggregate_ol.py`, `report_ol.py`, `vl5_e_vs_i.py` were rewritten; their diffs are long).
- Output root: `results/S1_2_layermax_openllama/early/` (`lm_core.RES`); shared single-model cache `results/S1_2_layermax_openllama/cache/`. Every gated script calls `guard.require_partb_frozen(lm_core.RES)` first (plan F1(a)).

| copy | original (sha256 prefix) | what changed |
|---|---|---|
| `lm_core.py` | `lm_core.py` (`26306eac`) | Model list (OpenLLaMA 3B/7B/13B, family OpenLLaMA), feature paths (13B in `features/wit1024_openllama_axis/`), output and Part B paths, `OL_THREADS`, the S-E2 switch (`OL_SENS_E2=1`), the S0b test hooks (`OL_S0B_TEST=1`, refused after CODE FROZEN). Kernels unchanged. |
| `precompute.py` | `precompute.py` (`7106c4b5`) | V-O1 reference = Part B clip for 13B, S1.2 cache for the rest; shared clip cache; V-O2 vs the frozen knn_final; worker count from `OL_THREADS`; `--phase gpu|cpu`, `--models`. Not gated (single-model). |
| `run_layermax.py` | `run_layermax.py` (`5d034485`) | Gate first; 51-pair list; `--lms/--vits` debug filters; inputs through `lm_core.cache_dir`. Pair-file format unchanged. |
| `validate.py` | `validate.py` (`0ad3fb86`) | Gate first; V-O3 = the 12 layer pairs of plan section 11 (13B), V-O4(a) = 3B/7B fixed-depth entries; output `validation/VO3_VO4a.json`. |
| `vl4b_ol.py` | `vl4b_llama7b.py` (`361e4c04`) | V-O4(b) for the 34 3B/7B reruns vs the frozen 8-LM E files; gate first; `compare()` reused by the aggregate for `G|rerun_equals_frozen_3b7b`. |
| `vl7_prop5.py` | `vl7_prop5.py` (`1cca911d`) | V-O7 on the new pairs (51-pair values also reported); PRs from `partB_features_ol.json`; new debiased checks (b) closed form and (c) m = 8 plant. |
| `vl5_e_vs_i.py` | `vl5_e_vs_i.py` (`e31773fd`) | V-O5 from the E side on all 51 pairs; I's raw arrays summarised with E's own definitions (mKNN in exact rationals); verdict level through `compare_verdicts.py` (EI, F, J); order rule (E final before I opened). |
| `aggregate_ol.py` | `aggregate_llama.py` (`d4de0157`) | Rewritten for plan sections 5-10 and `VERDICTS_SPEC.md` sections A, B, C, D, G; mKNN exact (F2); `delta_boot` kept with key `S1.2-layermax-openllama|boot|arh_delta|<pre>|<metric>` and the 11-LM panel (R.3.6). |
| `report_ol.py` | `report_llama.py` (`00c995bd`) | Rewritten report; every number from `summary.json` / validation files; labels of R.1.2, R.1.3. |
| `write_hashes.py` | `write_hashes.py` (`2c107c4e`) | Paths; gate first; E pair digest and the 204-file LLaMA-axis E digest (`0d918377…`, R.3.7). |

`v4_arh.py` of the LLaMA E code is not copied: V-O6 runs once, in I only (plan section 11).

**S0b tests (scratch `~/oll_s0b_scratch/`, test mode only, not citable).**
- T1 (LLaMA-axis E pair files relabelled 3b/7b/13b <- llama-7b/13b/30b): 1566 trend values equal the LLaMA-axis E summary; the only 3 differences are the descriptive mKNN trends where this copy is exact and equals the exact-rational values of the merged LLaMA summary (`validation_cross` part 4).
- T2 (frozen OpenLLaMA 3b/7b + llama-30b): pooled p and bootstrap CIs equal the LLaMA E values; `G` keys as expected.
- GPU precompute smoke (3b, 7b, stand-in 13b, ViT-Ti): clip thresholds bitwise equal to their references.
- Integration fixture X (one real `partb_ol.py` Part B, E, I, forecast, joint, R on the same data): E vs I 5233 EI keys, 452 F keys, 24 J keys, 0 mismatches; pair-level V-O5 pass; R-E vs E and R-I vs I 0 mismatches except `G|rerun_equals_frozen_3b7b` (fixture artifact: R reads the fixture's stand-in frozen files, E and I the real frozen 3b/7b files).
- GPU smoke of `validate.py` and `run_layermax.py` on the stand-in 13B x ViT-Ti (see the S0b CODE FROZEN line and `S0B_NOTES.md`).
