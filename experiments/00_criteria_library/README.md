# 00 · Criterion library and known-answer checks

**Question.** Every criterion of the lattice (neighbor sets at each k, connected components, rankings, the global order, the MST with its nearest-neighbor forest and bridges, biased and debiased CKA, RSA, the SOG overlap, and permutation calibration) is implemented twice from one frozen specification and checked on synthetic couplings with known answers before any real-data analysis. In the words of the paper: "Our code agreed with the reference implementations of PRH and ARH to 1.67 × 10<sup>−8</sup>, within a tolerance of 10<sup>−6</sup>. On three pairs of models at both fixed depths, it also agreed to 10<sup>−6</sup> with the published code of PRH and ARH, and with that of SOG under the preprocessing of PRH. The exceptions come from two WIT inputs with identical captions."

**Paper.** Sections 2 and 3 (definitions), Section 5.1, and Appendix F.2 (known-answer checks).

**Status.** Preregistered in the Stage-1 preregistration, with its amendment v1.1 (`stage1/prereg/LEAD_DECISIONS_v1.1.md`).

| frozen file | sha256 (first 16) |
|---|---|
| `stage1/SPEC.md` (definition of every criterion) | e781e44f6b809e90 |
| `stage1/SPEC_CHANGELOG.md` | 02b8ba5871eae7ab |
| `stage1/prereg/PREREG_stage1.md` (+ `PREREG_stage1.sha256`) | bee0583c2bfd3ea9 |
| `stage1/prereg/LEAD_DECISIONS_v1.1.md` | eaebd68787a7111f |

**Code.**
- `stage1/hrhv2/`: the criterion library ("ladder": `ladder.py`, `structural.py`, `geometric.py`, `quotients.py`, `calibrate.py`, `prep.py`, `gl.py`, `run_pair.py`), the known-answer plants (`plants.py`, `run_g0.py`), the coupling estimators (`predict.py`, `predict_v11.py`) and 10 test files. `FIXES.md` and `SPEC_INTERPRETATIONS.md` record every fix and reading of the specification.
- `stage1/hrhv2_ref/`: an independent implementation written from `SPEC.md` and the published code only, with its own tests and scripts; `REPORT.md` describes it.
- `stage1/results/verification/*.py`: the cross-checks of the two implementations against each other and against the pinned PRH, ARH and SOG code.

**Inputs.** Synthetic couplings only for the tests and the known-answer checks. The cross-checks against the published code need `bash scripts/fetch_vendor.sh`, and the real-feature cases of `vendor_identity.py` need the features of [01](../01_feature_extraction/README.md).

**Steps.**
```bash
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && venv/bin/python -m pytest -q hrhv2/tests'
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008/hrhv2_ref && ../venv/bin/python -m pytest -q tests'
# known-answer checks (gate G0): writes results/G0_report.json and results/G0_report.md
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && OMP_NUM_THREADS=1 venv/bin/python -m hrhv2.run_g0 --jobs 10 --out results/G0_report'
# identity against the pinned PRH, ARH and SOG code
scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && venv/bin/python results/verification/vendor_identity.py OUT.json'
```

**Outputs (shipped).** `stage1/results/G0_report.md`, `G0_report.json`, `stage1/results/verification_report.md`, `stage1/hrhv2_ref/REPORT.md`. The check of the fixed-depth driver against the pinned code is `stage1/results/S1_2/g1prime.json` (see [04](../04_spanning_trees_cka/README.md)).

**Compute in the experimental record (Table R.2).** Known-answer checks: 54.0 min wall, 10 processes (CPU).

**Deviations and failures.** Two of the known-answer checks failed. On a coupling built by an exponential link, one cross-fitted isotonic fit explained 0.9978 of the variance against a required 0.999; every neighbor, component, and spanning-tree criterion agreed exactly in all 27 cells. On Gaussian couplings, the first coupling estimator missed the observed s<sub>N</sub> at k = 1 by up to 0.057 against a tolerance of 0.02. The amendment (`LEAD_DECISIONS_v1.1.md`) based the isotonic check on the in-sample fit and, for k ≤ 3, replaced the approximation by the exact expected discrepancy; the rebuild of the exponential coupling it specified was not executed. Agreement with the reference implementations of PRH and ARH passed, with a largest difference of 1.67 × 10<sup>−8</sup> against a tolerance of 10<sup>−6</sup>.
