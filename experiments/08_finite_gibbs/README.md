# 08 · Finite Gibbs model

**Question.** "Theorem 3 states that a learner minimizing InfoNCE and a learner minimizing the spectral contrastive loss learn one relation through different increasing links in the finite Gibbs model. As both samples grow, exact agreement of the two learners on the order of distances becomes reliable, whereas a criterion on scale-normalized geometry with a small tolerance is asymptotically obstructed." This experiment tests the theorem and the analysis pipeline inside the premises of the theorem.

**Paper.** Section 5.6 (Insight 5), Figure 6, and the finite-Gibbs appendix.

**Status.** Preregistered, with four hashed amendments; two of them fixed programming errors in the frozen analysis code. The sample size N* at which agreement on the global order reaches 90% of pairs was predicted before the runs (`stage2/s23/results/prestep/frozen_predictions.json`).

| frozen file | sha256 (first 16) |
|---|---|
| `stage2/s23/prereg/PREREG_S2_3_v1.0.md` (withheld) | 7a93b7a9519b505e |
| `stage2/s23/prereg/CODE_AT_PREREG.sha256` (37 code files; all match this checkout) | f9560f0a9d3d2bc4 |
| `stage2/s23/prereg/AMENDMENT_A1_resume.md` (withheld) | 8a858f5394c4be17 |
| `stage2/s23/prereg/AMENDMENT_A2_complete.md` (withheld) | 435260f60abeba90 |
| `stage2/s23/prereg/AMENDMENT_A3_analysis_memo.md` (withheld) | d32110645e1fc414 |
| `stage2/s23/prereg/AMENDMENT_A4_p5b_zero.md` (withheld) | c92194b69112d0ce |

The preregistration and the four amendments are withheld for now, see [`prereg/README.md`](../../prereg/README.md#frozen-files-withheld-from-this-release). Their sha256 are recorded there and in `stage2/s23/resume/A*_MANIFEST.sha256`, and the amendment runners refuse to run without them. The frozen code record `CODE_AT_PREREG.sha256` and the prediction of N* made before the runs ship.

**Inputs.** None; the instance, the probes and every sample are generated from the frozen settings in `stage2/s23/src/cfg.py`. The run list is `stage2/s23/plans/stageA_plan.json`.

**Steps.**
```bash
S23='cd ~/hrh_v2_stage2_20261008/s23'
# unit tests on a tiny configuration (they set S23_ROOT to a temporary directory)
scripts/lrh_env.sh bash -c "$S23/src && ~/hrh_v2_stage1_20261008/venv/bin/python -m pytest -q tests"
# Stage A: frozen pre-step, plan, pilots and gates, main runs (2 GPU workers + 1 CPU worker), then analysis
scripts/lrh_env.sh bash -c "$S23 && bash src/launch_stageA.sh"
```
On the experiment machine Stage A stopped under contention and was completed under the amendments, with runners kept outside `src/` so that `src/` keeps its frozen file set: A1 resumed the runs (`resume/gate_then_launch.sh`, `resume/launch_stageA_resume.sh`), A2 completed them without the GPU budget limit (`resume/launch_stageA_A2.sh`), A3 replaced a memo of the frozen analysis that failed on its 1,501st distinct load (`resume/launch_A3_analyze.sh`), and A4 defined one line of the frozen analysis for a Monte Carlo prediction equal to 0 (`resume/launch_A4_analyze.sh`). `resume/A*_MANIFEST.sha256` record their code. `launch_stageA.sh` starts only after the freeze line of the preregistration is in the run log.

**Outputs (shipped).** `stage2/s23/results/stageA/REPORT_stageA.md`, `stageA_results.json`, `N_match.json`; independent recompute of the completion `stageA/review_A2/recompute_A2.json` and its scripts; `stage2/s23/results/prestep/frozen_predictions.json`; plotted values in `figures/v8/F_link_top/numbers.json`.

**Compute in the paper.** 12.71 GPU-h (frozen pre-step 2.71; pilots and gates 1.37 and main runs 8.63 for the InfoNCE learner); the spectral learner ran on the CPU.

**Results and failures.**
- Both learners return certified minimizers of their exact empirical risks, in 192 runs per learner at each of eight sample sizes from 2<sup>20</sup> to 2<sup>40</sup> examples.
- Exact agreement across objectives on the order of all distances among eight probe inputs held in 0 of 96 pairs up to 2<sup>28</sup> examples and in 94 of 96 at 2<sup>34</sup>, near the value N* = 2<sup>33.75</sup> predicted before the runs.
- At 2<sup>40</sup> examples, the largest difference of mean-normalized distances across objectives settled at a median 0.348, the limit given by the theorem. Under the scale-normalized discrepancy all 96 pairs stayed outside the tolerance, whereas under the isotonic discrepancy every pair fell within it.
- After Holm correction, all nine preregistered confirmatory predictions held, but two secondary Monte Carlo predictions failed: those of the ratio of the discrepancy across objectives to the mean discrepancy within an objective on neighbor sets and on spanning trees.
- This experiment tests Theorem 3 and our analysis pipeline inside the premises of the theorem, not trained networks, and the sample size it needs for exact agreement on the global order (about 2<sup>34</sup> examples here) reflects how fine that criterion is rather than a practical training budget.

**Deviations.** We wrote and froze amendments A3 and A4 during the analysis run without first consulting the lead author. The preregistration allows such code fixes when they are recorded before the affected data are opened. The file metadata leave open whether the 418 pair caches written before the A4 freeze were opened in between, and every decision is independent of the answer, since A4 changes one secondary boolean. Every solve was to be certified, with failures kept and counted against the predictions: 44 of 3,456 solves narrowly failed their certificates (39 at 2<sup>25</sup>, 5 in the secondary arm at 2<sup>22</sup>).
