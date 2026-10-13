#!/usr/bin/env bash
# Fetch the published code of PRH, SOG and ARH at the commits used by the paper's cross-checks into
# stage1/vendor/. None of it is redistributed in this repository (see THIRD_PARTY_NOTICES.md).
#
#   stage1/vendor/platonic-rep             github.com/minyoungg/platonic-rep     @ dcd76ba3c950c1b197a2ae8b1c6713535c94ecf9
#   stage1/vendor/sog_what_converges_prh   github.com/junwon0/what-converges-prh @ be654a3732c8ed92329899d68df759d61ae040a4
#   stage1/vendor/aristotelian             github.com/mlbio-epfl/Aristotelian    @ 785054b19c952dca0f605768552ddd50fa9286dc,
#                                          only LICENSE and calibrated_similarity/calibration.py
#                                          (the full clone goes to stage1/vendor/_src/Aristotelian)
#
# The ARH folder holds only those two files and no __init__.py, as on the experiment machine. A full ARH checkout
# there would put ARH's own top-level package `aristotelian` on sys.path ahead of SOG's package of the same name,
# and results/verification/vendor_identity.py and every check that imports it would then fail or load ARH modules.
#
# Users of these files: the cross-check tests (hrhv2/tests/conftest.py, hrhv2_ref/tests/test_vendor*.py,
# hrhv2_ref/tests/test_calibrate.py), results/verification/vendor_identity.py, and the vendored ARH checks of the
# layer-maximum experiments (v4_arh.py, vl6_arh.py, validate.py). Those tests skip when the files are absent.
# scripts/lrh_env.sh makes ~/hrh_finite_20260922/vendor point at stage1/vendor, the path the ARH checks expect.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V="$REPO/stage1/vendor"
mkdir -p "$V"
fetch() {  # name url commit: clone into $V/name and check out the commit
  local dir="$V/$1" url="$2" commit="$3"
  if [ -e "$dir" ] && [ ! -d "$dir/.git" ]; then
    echo "skip $1: $dir exists and is not a git clone" >&2
    return 0
  fi
  [ -d "$dir/.git" ] || git clone --quiet "$url" "$dir"
  git -C "$dir" checkout --quiet "$commit"
  echo "$1 at $(git -C "$dir" rev-parse HEAD)"
}
fetch platonic-rep           https://github.com/minyoungg/platonic-rep.git     dcd76ba3c950c1b197a2ae8b1c6713535c94ecf9
fetch sog_what_converges_prh https://github.com/junwon0/what-converges-prh.git be654a3732c8ed92329899d68df759d61ae040a4

# ARH: copy two files of the pinned clone into the layout the ARH checks expect.
A="$V/aristotelian"
if [ -e "$A/aristotelian" ] || [ -d "$A/.git" ]; then
  echo "error: $A is a full ARH checkout; its package 'aristotelian' would shadow SOG's. Move it aside and rerun." >&2
  exit 1
fi
mkdir -p "$V/_src"
fetch _src/Aristotelian https://github.com/mlbio-epfl/Aristotelian.git 785054b19c952dca0f605768552ddd50fa9286dc
mkdir -p "$A/calibrated_similarity"
cp "$V/_src/Aristotelian/LICENSE" "$A/LICENSE"
cp "$V/_src/Aristotelian/calibrated_similarity/calibration.py" "$A/calibrated_similarity/calibration.py"
if command -v sha256sum >/dev/null 2>&1; then SHA=(sha256sum); else SHA=(shasum -a 256); fi
( cd "$A" && printf '%s\n' \
    "1cdb16cb776b6c96860bc82d68306c5fb8ae5f0d437be5933eabeb9737edf519  LICENSE" \
    "3ea86bc2bc35bb0da62410588f1de5b8b6bd9195d186b89d013244ed3b19ce59  calibrated_similarity/calibration.py" \
  | "${SHA[@]}" -c - )
echo "aristotelian: LICENSE and calibrated_similarity/calibration.py as used on the experiment machine"
