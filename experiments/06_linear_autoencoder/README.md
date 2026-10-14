# 06 · Linear autoencoder

**Question.** "For the training procedure, Theorem 2 predicts that two gradient-flow runs of a linear autoencoder keep the gauges set by their initializations and disagree in geometry at every sample size. Part 2 certifies the resulting obstruction of CKA criteria only below a tolerance that it leaves unspecified, so our tolerance of 0.01 is a choice of the preregistration rather than a prediction of the theorem. Weight decay and tied weights instead fix one gauge, and by part 1 a small initialization keeps Ω close to the identity in both runs. The global minimizers of parts 1 and 3 on one sample all share the projectors of part 4. On independent samples, the projectors of two global minimizers converge in probability to one limit as the samples grow (part 5 of the full version of Theorem 2)." The experiment tests these predictions at sample sizes from N = 2<sup>8</sup> to the population limit, and each run draws its own training sample at a finite N.

**Paper.** Section 5.5 (Insight 4), Figure 5a to 5c, Figure F.5, and Appendix F.7.

**Status.** Preregistered in the Stage-1 preregistration. The implementation plan v1.0 was frozen before the confirmatory runs; v1.1 records the handling of diverged runs after a crash of v1.0, and a reporting-only revision of `analyze_s13a.py` that changed no decision or verdict rule. Both hash records ship.

| frozen file | sha256 (first 16) |
|---|---|
| `stage1/prereg/PREREG_stage1.md` | bee0583c2bfd3ea9 |
| `stage1/linae/ANALYSIS_PLAN.md` (+ `ANALYSIS_PLAN.sha256`) | bccacb10d1d25494 |
| `stage1/linae/ANALYSIS_PLAN_v1.1.md` (+ `ANALYSIS_PLAN_v1.1.sha256`) | 7329052ac8bedb2c |

The shipped code matches the v1.1 record, including its line for the revised `analyze_s13a.py` (see [`prereg/README.md`](../../prereg/README.md)).

**Inputs.** None. The data are synthetic Gaussian inputs in 16 dimensions with a 4-dimensional linear autoencoder (`linae_core.py`), in float64.

**Steps.**
```bash
# unit tests, training (resumable, at most 12 processes with 1 BLAS thread each), pair criteria, analysis
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && bash linae/run_all.sh'
```
`run_all.sh` runs `pytest test_linae.py`, `run_s13a.py train`, `run_s13a.py criteria` and `analyze_s13a.py` in this order and appends progress lines to the run log. `LINAE_NPROC` sets the number of worker processes. `pilot/` holds the excluded pilots that fixed the learning rates, and `diag/` the check that chunks written by v1.0 are reproduced bitwise by v1.1. The v1.0 code that `diag/` imports ships in `stage1/linae/code_v1.0/` and matches the v1.0 record `ANALYSIS_PLAN.sha256`.

**Outputs.** `stage1/linae/runs/` and `stage1/linae/criteria/` (not shipped). Shipped summaries: `stage1/results/S1_3a_linae_summary.md`, `S1_3a_linae_summary.json`; plotted values in `figures/v8/F_gauge/numbers.json` (panels a to c) and `figures/v8/F_linae_app/numbers.json`.

**Compute in the experimental record (Table R.2).** About 1.5 CPU-h.

**Results and failures.**
- Under gradient descent without a regularizer, the criteria on neighbor sets, mean-normalized squared distances (G<sub>s</sub>), and CKA each accepted 0 of 48 pairs of runs at every finite N and 0 of 500 at N = ∞.
- With weight decay, tied weights, or small initialization, agreement under these criteria was reliable for N ≥ 2<sup>12</sup>, and agreement under the projector criterion of part 4 was reliable at these sizes for every training procedure, as preregistered for N = 2<sup>16</sup>.
- Outside the preregistered tests, the medians of the condition number of the learned EE<sup>⊤</sup> are 4.66 against 4.67 predicted under a small step close to gradient flow and 1.0330 against 1.0329 under weight decay.
- Failed: the preregistered prediction for gradient descent also required the mean discrepancy over pairs of runs at N = 2<sup>16</sup> to lie inside the 95% interval of this mean at N = ∞, and this part of the prediction failed on G<sub>s</sub> and CKA.

**Deviations.** The preregistered step 0.05 changes the imbalance C by a median 53%, so arms with step 0.002 were added as secondary arms before the confirmatory runs. The learning rates of stochastic gradient descent and Adam, which the preregistration left open, were fixed on excluded pilots.
