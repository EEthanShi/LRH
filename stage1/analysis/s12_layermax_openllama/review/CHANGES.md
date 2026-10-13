# Review recomputation R of the OpenLLaMA axis (S0b)

Role `oll-axis-extract`, step S0b of `analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md` (sha256 `4f18d710…`, section 3 "R", section 11 step S8). Written before any OpenLLaMA-13B byte existed on ds2213; hashed in the S0b `CODE FROZEN` STATUS line.

| file | origin | role |
|---|---|---|
| `ol_axis_review.py` | copy of `analysis/s12_layermax_llama/review/llama_axis_review.py` (`a9657619`, pristine in `orig_copy/`; diff `ol_axis_review.py.diff`) | Entry point: gate first; recomputes every verdict of `VERDICTS_SPEC.md` (A-D, F, G, J) from the pair files of E and, separately, of I; compares with E, I, `forecast_ol.py` and `joint_summary.py` through `compare_verdicts.py` (0 mismatches required); F1(c) birth-time audit through `guard.py audit`. Output root `results/S1_2_layermax_openllama/review/`. |
| `ol_review_lib.py` (new) | written from the plan, the spec and the stored pair formats | Pair readers for both formats, exact mKNN arithmetic. No E/I/forecast/joint code was read. |
| `ol_review_sections.py` (new) | as above | The verdict sections; choices left open are marked `[R-op]` and listed in `REVIEW_REPORT.md`. |
| `make_test_fixture.py` (new, test only) | — | S0b fixtures (test mode only; refused after CODE FROZEN). |
| `test_reproduce_llama.py` (new, test only) | — | S0b reproduction check of R against the LLaMA-axis results (test mode only). |

Usage: `analysis/s12_layermax_openllama/launch_ol.sh R_review 4 -- venv/bin/python analysis/s12_layermax_openllama/review/ol_axis_review.py [--impl both|E|I] [--phase all|compute|compare]`. Exit 0 = every comparison passed, 1 = a mismatch or a missing target, 3 = refused by the Part B gate.

**S0b tests (scratch, test mode only, not citable).**
- `llama3` fixture: 3458 trend values of R equal the LLaMA-axis E/I values (mKNN vs the exact-rational rows); `ol2` fixture: 184 pooled values equal; bootstrap CIs and 10! p equal the LLaMA E values (22/22, 36/36); planted mismatches are detected and listed.
- Integration fixture X (real `partb_ol.py` Part B; E, I, forecast and joint run on the same data): R-E vs E and R-I vs I 0 mismatches on 5232 of 5233 EI keys and on all F and J keys; R-E vs R-I 0 mismatches. The one difference, `G|rerun_equals_frozen_3b7b`, is a fixture artifact (R reads the fixture's stand-in frozen files, E and I the real frozen 3b/7b files).
