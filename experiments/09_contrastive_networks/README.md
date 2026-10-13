# 09 · Contrastive networks on CIFAR-5m

**Question.** Does the prediction of Theorem 3 carry over to trained networks outside its premises? The theorem predicts that InfoNCE and the spectral contrastive loss learn one relation through different increasing links, so that their representations agree on every order-based criterion and disagree on scale-normalized geometry. The networks share an airbench94 trunk with a projector to a 128-dimensional embedding; InfoNCE uses a cosine critic at temperature 0.2 (arm A) and 0.5 (arm A′), and the spectral contrastive loss (arm B) uses the ball projection of HaoChen et al. Each arm trains 64 runs on N = 50,000 and 500,000 images at a budget of five million images seen.

**Paper.** Section 5.6, Figures F.8 and F.9, and the contrastive-network appendix.

**Status.** Preregistered; the prediction failed.

| frozen file | sha256 (first 16) |
|---|---|
| `stage2/prereg/PREREG_stage2.md` (+ `PREREG_stage2.sha256`) | 6ce39abe7094034d |
| `stage2/prereg/PREREG_stage2_amendment1.md` | dfce9385fd76a206 |
| `stage2/prereg/eps_S2_1.json` (tolerances, from pilot pairs only) | 0b28027d7366fb75 |
| `stage2/prereg/CODE_AT_PREREG.sha256` (all 8 files match this checkout) | fc1f54d271c781f8 |

**Inputs.** CIFAR-5m, prepared as in [07](../07_gauge_networks/README.md). The run lists `stage2/plans/S2_1_pilot.json` and `S2_1_main.json` ship.

**Steps.** The contrastive runs share the training queue of the first gauge-network experiment.
```bash
S2='cd ~/hrh_v2_stage2_20261008'
PY='~/hrh_v2_stage1_20261008/venv/bin/python'
scripts/lrh_env.sh bash -c "$S2 && bash src/launch_workers.sh 3"      # plans: S2.2 pilot, S2.1 pilot, S2.2 main, S2.1 main
scripts/lrh_env.sh bash -c "$S2 && $PY src/gates.py all"             # gate and tolerance freeze from pilot pairs
scripts/lrh_env.sh bash -c "$S2 && OMP_NUM_THREADS=1 $PY src/analyze.py s21 --jobs 2"
```

**Outputs (shipped).** `stage2/results/S2_1/S2_1_report.md`, `s21_summary.json`; recompute scripts in `stage2/results/review_final/s21-recompute/`; plotted values in `figures/v8/F_link_net/numbers.json` and `figures/v8/F_link_app/numbers.json`.

**Compute in the paper.** 48 pilot + 448 runs, 5.05 + 30.38 h summed run time, 1.57 + 7.52 h wall.

**Results.** "The preregistered prediction failed in CIFAR-5m networks outside these premises, where at 500,000 images all 64 pairs of InfoNCE and spectral runs disagree even on neighbor sets and a fitted monotone link removes a median 15% of the residual left by an affine map, against 100% for the exact link of the theorem applied to an InfoNCE Gram matrix as a positive control." Both secondary predictions failed as well: the prediction derived from the co-occurrence lemma, since more data did not shrink the log-distance error of near pairs of images faster than that of far pairs, and the prediction of a fitted link exponent in [2.5, 10].

**Deviations.** The unconstrained spectral loss diverged at every learning rate tried, so the spectral arm uses the ball projection, which was active on every evaluated input, and the arm trains on the unit sphere. The paper records the design errors of the Stage-2 preregistration that concern this experiment (the temperature item assumed a pure rescaling, and the falsification threshold of 0.2 on the warp share rejects the sharp link of the theorem at the InfoNCE temperature 0.2); each leaves every number and decision unchanged.
