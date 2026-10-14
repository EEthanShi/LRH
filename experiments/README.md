# Experiments

One folder per experiment of the paper. Each README states the question, the frozen files, the inputs, the commands in order, the shipped outputs, the compute reported in the paper's experimental record (Table R.2), and the deviations and failures the paper reports. The code itself stays in `stage1/`, `stage2/` and `figures/`, in the layout it ran in.

| | Experiment | Paper | Status |
|---|---|---|---|
| [00](00_criteria_library/README.md) | Criterion library and known-answer checks | Sections 2 and 3 | preregistered |
| [01](01_feature_extraction/README.md) | Feature extraction (WIT, COCO pool) | Section 5.1 | protocol frozen |
| [02](02_evaluation_size/README.md) | Evaluation size: estimator selection and the (m, k) test | Section 5.2, Fig. 2a, 2b, F.1 | preregistered; crossing-law test not run |
| [03](03_sentence_encoders/README.md) | Sentence encoders; depth follow-ups | Section 5.2, Appendix G, Fig. F.2 | preregistered (+ addendum 1); depth follow-ups exploratory |
| [04](04_spanning_trees_cka/README.md) | Spanning trees and CKA at fixed depth | Section 5.3, Fig. 2c, 3, F.3 | preregistered |
| [05](05_layer_max/README.md) | Layer maximum: BLOOMZ, LLaMA-1, OpenLLaMA, Pythia | Section 5.4, Fig. 4, F.4 | BLOOMZ threshold analysis post hoc; other families forecast before scoring |
| [06](06_linear_autoencoder/README.md) | Linear autoencoder | Section 5.5, Fig. 5a to 5c, F.5 | preregistered |
| [07](07_gauge_networks/README.md) | Gauge networks on CIFAR-5m | Section 5.5, Fig. 5d, 5e, F.6, F.7 | preregistered |
| [08](08_finite_gibbs/README.md) | Finite Gibbs model | Section 5.6, Fig. 6 | preregistered (+ amendments A1 to A4) |
| [09](09_contrastive_networks/README.md) | Contrastive networks on CIFAR-5m | Section 5.6, Fig. F.8, R.9 (record) | preregistered; prediction failed |
| [10](10_figures_tables/README.md) | Figures and tables | all | plotted values shipped |

Conventions used in every README:
- Commands run through the path shim, for example `scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && <command>'`, from the repository root. See the top-level README, section Installation.
- Create the log folders first: `mkdir -p stage1/logs stage2/logs`.
- Several launchers check freeze lines in the project's internal run log (`STATUS.md`, not shipped) and append to it. The frozen plan of each experiment gives the order of its steps and the freeze lines its guards expect.
- The code has not been re-run from this checkout.
- Ten frozen files are withheld for now (see [`prereg/README.md`](../prereg/README.md#frozen-files-withheld-from-this-release)); the experiments whose guards check them are 03, 05 (OpenLLaMA and Pythia), 07 (second experiment) and 08.
