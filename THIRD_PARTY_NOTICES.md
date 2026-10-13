# Third-party notices

The code of this repository is under the MIT licence in [LICENSE](LICENSE), except for the files listed below.

## Shipped third-party files

**ARH (Aristotelian Representation Hypothesis)**, [github.com/mlbio-epfl/Aristotelian](https://github.com/mlbio-epfl/Aristotelian), commit `785054b19c952dca0f605768552ddd50fa9286dc`, MIT licence, "Copyright (c) 2026 Fabian Gröger, Shuo Wen, Maria Brbić".

Three files are copied verbatim into `stage1/hrhv2_ref/tests/vendor_arh_785054b/` as a cross-check oracle for the tests of the reference implementation:

| shipped file | upstream file |
|---|---|
| `utils.py` | `aristotelian/metrics/utils.py` |
| `estimators.py` | `aristotelian/metrics/estimators.py` |
| `preprocess.py` | `aristotelian/prh/preprocess.py` |

The upstream licence is in `stage1/hrhv2_ref/tests/vendor_arh_785054b/LICENSE` (byte-identical copy). The folder's `__init__.py` is ours.

## Fetched, not shipped

`scripts/fetch_vendor.sh` clones the following code into `stage1/vendor/` (ignored by git) for the cross-checks against the published implementations:

| folder | upstream | commit | licence |
|---|---|---|---|
| `stage1/vendor/platonic-rep` | PRH, [github.com/minyoungg/platonic-rep](https://github.com/minyoungg/platonic-rep) | `dcd76ba3c950c1b197a2ae8b1c6713535c94ecf9` | no licence file in the repository, so it is not redistributed here |
| `stage1/vendor/sog_what_converges_prh` | SOG, [github.com/junwon0/what-converges-prh](https://github.com/junwon0/what-converges-prh) | `be654a3732c8ed92329899d68df759d61ae040a4` | MIT |
| `stage1/vendor/aristotelian` | ARH, [github.com/mlbio-epfl/Aristotelian](https://github.com/mlbio-epfl/Aristotelian): only `calibrated_similarity/calibration.py` and `LICENSE`, copied from a clone in `stage1/vendor/_src/Aristotelian` and checked by sha256 | `785054b19c952dca0f605768552ddd50fa9286dc` | MIT |

## Data and model weights

No dataset and no model weight is included. WIT (through the PRH subsets on the Hugging Face Hub), COCO, CIFAR-5m, and every pretrained model keep their own licences and terms of use. LLaMA-1 weights are subject to Meta's licence.
