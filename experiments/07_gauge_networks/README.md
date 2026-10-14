# 07 · Gauge networks on CIFAR-5m

**Question.** How does the training procedure select the gauge outside the premises of Theorem 2? "We train CIFAR-5m networks with a linear head. The head maps its input features h to coordinates u = Mh ∈ ℝ<sup>9</sup> and outputs y = Wu, and the task fixes u up to a gauge." Two preregistered experiments train this head by momentum without weight decay, with weight decay, or from a small initialization, on N = 25,000 and 250,000 images; the second adds gradient flow of the head (head flow) to separate the gauge kept by gradient flow from the change of C under momentum (Remark 1).

**Paper.** Section 5.5 (Insight 4), Figure 5d and 5e, Figures F.6 and F.7, and Appendices F.8 and F.8.1.

**Status.** Both experiments are preregistered.

| frozen file | sha256 (first 16) |
|---|---|
| first experiment: `stage2/prereg/PREREG_stage2.md` (+ `PREREG_stage2.sha256`) | 6ce39abe7094034d |
| `stage2/prereg/PREREG_stage2_amendment1.md` | dfce9385fd76a206 |
| `stage2/prereg/eps_S2_2.json` (tolerances, from pilot pairs only) | b46e3f1dd62e3579 |
| `stage2/prereg/CODE_AT_PREREG.sha256` | fc1f54d271c781f8 |
| second experiment: `stage2/prereg/PREREG_S2_2_v1.1.md` (withheld for now, see [`prereg/README.md`](../../prereg/README.md#frozen-files-withheld-from-this-release)) | 3c9d32171d303566 |
| `stage2/results/S2_2_v11/eps/eps_S2_2_v11.json` (tolerances) | dd649d74fb3a87fd |
| `stage2/results/S2_2_v11/freeze/CODE_AT_PREREG_v11.sha256` | f2d5d28dfda1b393 |

**Inputs.** CIFAR-5m, downloaded and md5-checked by `stage2/src/download_cifar5m.sh` (about 18 GB), then indexed by `prep_data.py`: parts 0 to 4 form the training pool, and part 5 gives the evaluation set of 2,048 images, a labelled bank, and a held-out set. Run lists (seeds and configurations) ship in `stage2/plans/` and `stage2/plans_v11/`.

**Steps.**
```bash
mkdir -p stage2/data/cifar5m stage2/logs stage2/results/S2_2_v11/logs
S2='cd ~/hrh_v2_stage2_20261008'
PY='~/hrh_v2_stage1_20261008/venv/bin/python'
scripts/lrh_env.sh bash -c "$S2 && bash src/download_cifar5m.sh && $PY src/prep_data.py"
# first experiment (shared queue with the contrastive networks of 09): workers walk the plans
# S2.2 pilot, S2.1 pilot, S2.2 main, S2.1 main; gates.py freezes the tolerances from pilot pairs and opens the main runs
scripts/lrh_env.sh bash -c "$S2 && bash src/launch_workers.sh 3"
scripts/lrh_env.sh bash -c "$S2 && $PY src/gates.py all"
scripts/lrh_env.sh bash -c "$S2 && OMP_NUM_THREADS=1 $PY src/analyze.py s22 --jobs 2"
# second experiment: engineering phase (excluded runs; writes plans_v11/eng_choice.json and the ENG_GO marker),
# then pilots, gates and tolerance freeze, main runs, final analysis
scripts/lrh_env.sh bash -c "$S2 && $PY src_v11/eng_v11.py run --workers 4"
scripts/lrh_env.sh bash -c "$S2 && $PY src_v11/pipeline_v11.py run --workers 4 --jobs 6"
```
`src/auto_finalize.py` re-queues failed runs once and runs the analysis when a cohort is complete. The second experiment's guard (`src_v11/guard_v11.py`) checks the code and the plan against `CODE_AT_PREREG_v11.sha256`, which names the shared library files by their absolute paths on the experiment machine, and looks for the freeze line in the run log; elsewhere, and while the plan is withheld, it refuses to start. The `ENG_GO` marker written by `eng_v11.py` ships as `stage2/plans_v11/ENG_GO`.

**Outputs (shipped).**
- first experiment: `stage2/results/S2_2/S2_2_report.md`, `s22_summary.json`; independent recompute `stage2/results/review_final/s22-recompute/REPORT.md`, `recompute_summary.json`, `recompute_pairs.json`, `eps_sensitivity.json` and its scripts
- second experiment: `stage2/results/S2_2_v11/S2_2_v11_report.md`, `S2_2_v11_report_full.md`, `summary.json`, `render/a4_parts.csv`, `render/render_v11.py`
- plotted values: `figures/v8/F_gauge/numbers.json` (panels d and e), `figures/v8/F_gauge_app/numbers.json`, `figures/v8/F_gauge_v11_app/numbers.json`

**Compute in the experimental record (Table R.2).** First experiment: 48 pilot + 1,344 runs, 0.17 + 4.74 h summed run time, 2.34 h wall with 4 workers. Second experiment: 1,984 pilot + 2,368 runs, 9.91 + 11.01 h summed run time and 0.43 h engineering, 6 h 38 min wall for the whole pipeline.

**Results and failures.**
- Under head flow, the imbalance part added 0.54 and 0.59 SD to the G<sub>s</sub> discrepancy at the two sizes, about one fifth of the imbalance part under momentum; momentum created a larger imbalance part than head flow on G<sub>s</sub> and 1 − CKA in 8 of 8 Holm-corrected tests.
- With weight decay, u agreed in 96 of 96 pairs on G<sub>s</sub> and on neighbor sets, at 250,000 images in the first experiment and at both sizes in the second, without an excess on G<sub>s</sub> (−0.04 SD at 250,000).
- Failed: without weight decay, u agreed on neighbor sets in 96 of 96 pairs at 250,000 images, contrary to the preregistered prediction P1 that this agreement would be unreliable at both sizes.
- Failed: in the second experiment at 250,000 images, the mean neighbor-set discrepancy of u lay about 0.002 below that of y under four training procedures, against the prediction that its 95% interval would lie within ±0.002; under momentum, small initialization fell short of weight decay; under head flow, the prediction that agreement of u on G<sub>s</sub> would be reliable at both sizes failed under both initializations.

**Deviations.** In the first experiment the analysis code was hashed after the tolerances were frozen, and an interim analysis on two runs per cell preceded the freeze. An aborted interim analysis left cached per-pair results, which the final analysis reused after an independent recompute matched all 252 decision cells. The code checks the projector criterion on u, where the preregistration names y, and the two agree to 10<sup>−7</sup>. The paper also records seven errors in the Stage-2 design, each of which leaves every number and decision unchanged.
