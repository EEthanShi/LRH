# 02 · Evaluation size: estimator selection and the (m, k) test

**Question.** Does neighbor overlap at a fixed k change with the evaluation size m at a rate set by the model pair, and is it stable at a fixed k/m? From the paper: "The heuristic first-order law behind Conjecture D.1 predicts that s<sub>N</sub> at a fixed k rises, stays flat, or falls with m according to the class of the model pair. The same law also predicts a stable s<sub>N</sub> at a fixed k/m for large m." This tests Theorem 1 (Crossing Law) against the collapse of neighbor overlap reported by BPC.

**Paper.** Section 5.2, Figure 2a and 2b, Figure F.1, and Appendix F.3.

**Status.** Preregistered. The out-of-sample test of the crossing law needed an estimator of the coupling; the estimator was re-selected under a frozen rule, no candidate was eligible, and the test was not run. The (m, k) measurements are therefore reported as measurements, except for two preregistered secondary predictions (the order of the slopes and the cancellation of the elasticities).

| frozen file | sha256 (first 16) |
|---|---|
| `stage1/analysis/s11/ANALYSIS_PLAN_S11.md` (v1.0) | 62fa39c637f9e604 |
| `stage1/analysis/s11/ANALYSIS_PLAN_S11_v1.1.md` (identical copy in `s11_select/`) | 736ec1d4c9d20a84 |
| `stage1/hrhv2/PREDICTOR_CHOICE.md` | 75e1ba3dec56b5db |
| `stage1/hrhv2/PREDICTOR_CHOICE_v1.1.md` (selection rule) | f83b13a9e2a78f42 |
| `stage1/prereg/LEAD_DECISIONS_v1.1.md` (decisions D1 to D4) | eaebd68787a7111f |

**Inputs.** The COCO pool features of [01](../01_feature_extraction/README.md) (m = 256 to 16,384, each set contained in every larger one) for 4 pairs of Pythia-410M models trained from different seeds, 10 pairs of vision models, and 15 vision–language pairs; the WIT selection set for the real-data eligibility check.

**Steps.**
```bash
mkdir -p stage1/results/S1_1/v11/logs
# 1. estimator re-selection: synthetic plants, the real-data eligibility check, then the frozen rule
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && venv/bin/python analysis/s11_select/run_plants.py --prefix S11sel --reps 4 --jobs 12 --out results/S1_1_select/plants'
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && venv/bin/python analysis/s11_select/run_ssel.py --jobs 12 --out results/S1_1_select/ssel'
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && venv/bin/python analysis/s11_select/select_v11.py --plants results/S1_1_select/plants --ssel results/S1_1_select/ssel --out results/S1_1_select'
# 2. distance caches, then the resumable (m, k) pipeline on COCO
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && venv/bin/python analysis/s11/s11v11_prep.py --workers 3 --threads 4'
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && bash analysis/s11/s11v11_pipeline.sh'
```
On the experiment machine the pipeline was reordered during the run by `s11v11_pipeline_v3.sh` and resumed with `s11v11_pipeline_v3_resume.sh`; their headers give the order. `s11v11_indep_check.py` is the independent script that reproduces the numbers of the analysis plan. The v1.1 scripts refuse to start without the authorization record `analysis/s11/LEAD_AUTHORIZATION.json`, which ships.

**Outputs (shipped).**
- `stage1/results/S1_1_select/selection_report.md`, `selection_summary.json` (decision: no eligible estimator)
- `stage1/results/S1_1/v11/S1_1_v11_summary.json`, `S1_1_v11_tables.md`, `S1_1_v11_{cells,curves,slopes}_{clip,l2,cliplayer,dedup}.parquet`
- plotted values: `figures/v8/F_neighbors/numbers.json`, `figures/v8/F_scale_app/numbers.json`

**Compute in the experimental record (Table R.2).** Estimator selection 31.1 + 2.7 + about 1.5 CPU-h; evaluation-size test 158.7 CPU-h for 1,879 units.

**Deviations and failures.**
- "The preregistered out-of-sample test of the crossing law needed an estimator of the coupling, and every candidate estimator failed the preliminary check against two baseline predictors on synthetic couplings."
- The first coupling estimator also failed a real-data sanity check on held-out WIT items, which led to the re-selection.
- "The rising class was absent, but our slope rule also classifies a synthetic coupling of this class as flat, which leaves the prediction for this class untested."
- A script of the evaluation-size test was edited after its outputs existed, without an entry in the event log; an independent script reproduces every number of the analysis plan.
- Descriptively, the crossing law with an estimated coupling predicts the right sign for every non-flat slope of the 29 pairs without sentence encoders, as does an affine null; for the vision–language pairs it predicts too slow a fall of mKNN with m (Figure F.1) and errs more in the level of mKNN than a no-change null.
