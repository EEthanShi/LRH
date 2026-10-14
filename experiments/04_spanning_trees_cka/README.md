# 04 · Spanning trees and CKA at fixed depth

**Question.** "Since the MST contains the nearest-neighbor forest for pairwise distinct distances (Proposition 2), we ask how many shared MST edges lie in this forest and whether the overlap of the bridges between its trees has a size trend." The same panel asks whether pretrained vision and language representations differ by a monotone link or by a residual (Proposition 3), and how the preprocessing changes size trends at a fixed depth.

**Paper.** Section 5.3 (Insight 2), Figure 2c, Figure 3, Figure F.3, and Appendix F.4.

**Status.** Preregistered in the Stage-1 preregistration; every operational choice it left open was fixed in the analysis plan before the panel was computed. The monotone-fit analysis is a preregistered check rather than a confirmatory test.

| frozen file | sha256 (first 16) |
|---|---|
| `stage1/analysis/s12/ANALYSIS_PLAN_S12.md` | 889a24db388777cf |
| `stage1/prereg/PREREG_stage1.md` | bee0583c2bfd3ea9 |

**Inputs.** The WIT-1024 features of [01](../01_feature_extraction/README.md): the 136 pairs of a vision model and a language model at mid depth (r = 0.5) and the last layer (r = 1.0), and the WIT selection set for the m = 2047 pool and the principal components of the width-equalized CKA.

**Steps.**
```bash
S1='cd ~/hrh_v2_stage1_20261008'
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python analysis/s12/prep.py"
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python analysis/s12/run_pairs.py --run main --jobs 12"
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python analysis/s12/run_pairs.py --run pool2047"       # power-gate pool, m = 2047
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python analysis/s12/run_pairs.py --run sidecar_rule"   # exploratory layer rule
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python analysis/s12/aggregate.py"
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python analysis/s12/g1prime.py"                        # check against the pinned PRH and ARH code
scripts/lrh_env.sh bash -c "$S1 && venv/bin/python analysis/s12/report.py"
```
`g1prime.py` needs the reference code of `bash scripts/fetch_vendor.sh`.

**Outputs (shipped).** `stage1/results/S1_2/S1_2_report.md`, `s12_summary.json`, `panel_main.parquet`, `panel_pool2047.parquet`, `panel_sidecar_rule.parquet`, `g1prime.json`; plotted values in `figures/v8/F_neighbors/numbers.json` (panel c), `figures/v8/F_criteria/numbers.json` and `figures/v8/F_mst_app/numbers.json`.

**Compute in the experimental record (Table R.2).** 3.93 CPU-h (m = 1024), 0.38 CPU-h (m = 2047), 1.84 CPU-h (layer-rule sensitivity).

**Results and open points.**
- On a larger evaluation set of m = 2047 inputs, "the nearest-neighbor forest makes up a median 0.875 and 0.889 of the shared MST edges at mid depth and the last layer, near its 84 to 85% share of each tree."
- Beyond an affine fit, a global monotone fit explains a further median 0.0004 and 0.0012 of the squared-distance discrepancy, below the preregistered level of 0.10 in all 136 pairs.
- At mid depth and after subtraction of the permutation mean, the BLOOMZ trends of CKA and mKNN are 0.70 and 0.84 with the preprocessing of PRH and reverse to −1.00 and −0.92 under plain ℓ<sub>2</sub> normalization, the preregistered secondary preprocessing.
- Open: bridge overlap exceeds its permutation null in 93% and 100% of pairs, but its BLOOMZ size trend at the last layer stays unresolved (trend 0.57, Holm-adjusted p = 0.23).
- The preregistered rule classifies MST overlap as mostly one-nearest-neighbor overlap at both depths, although its conditions on the forest share and on s<sub>T</sub> − s<sub>N,1</sub> hold nearly automatically.
- Width test (PE3): the preregistered rule returns "genuine global loss" (change 0.00 and −0.07), a name false as worded, since CKA minus its permutation mean keeps its fixed-depth trend (0.70, 0.92); the test shows only that width does not drive this trend.

**Deviations.**
- The power gate of the spanning-tree test asked for a rerun at m = 3072, a pool that could not be built from the extracted inputs; the largest available disjoint pool, m = 2047, replaced it. The verdict is the same on m = 1024 under the same correction.
- The fixed-depth panel ran before the amendment responding to the two failed known-answer checks was recorded ([00](../00_criteria_library/README.md)). The panel uses neither failed check, and the cross-fitted isotonic value is flagged wherever it appears.
