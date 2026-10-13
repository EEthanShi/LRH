#!/bin/bash
# Pythia axis, sub-role R: S0b dry runs of every R entry point (TEST ONLY; NEW FILE, PYT-AXIS).
# Plan section 4 F1(a) and the role task: without the plan's gates every entry point must exit 3 and write nothing.
# Run from ~/hrh_v2_stage1_20261008 through launch_pyt.sh (1 thread). One JSON line per case and a summary line.
# "Nothing written" is checked on R's own trees (review/, s0b_scratch/R/ except this script's own scratch files and
# the launcher logs), on results/S1_2_layermax_pythia/review/ (must not exist) and on an out-of-tree probe path.
set -u
cd "$HOME/hrh_v2_stage1_20261008" || exit 70
PY=venv/bin/python
R=analysis/s12_layermax_pythia/review
SR=analysis/s12_layermax_pythia/s0b_scratch/R
OUTSIDE=/tmp/pyt_R_dryrun_probe_doesnotexist
TEST_IN=$SR/dryrun_probe
mkdir -p "$SR/dryrun"
MARK=$(mktemp -p "$SR/dryrun" mark.XXXXXX)
ERR="$SR/dryrun/stderr_$(basename "$MARK")"
export PYTHONDONTWRITEBYTECODE=1
fail=0; n=0
run() {  # name, expected rc, env assignments ..., -- cmd ...
  local name="$1" want="$2"; shift 2
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env -u PYT_S0B_TEST "${envs[@]}" "$@" >/dev/null 2>"$ERR"
  local rc=$?
  n=$((n+1))
  local msg; msg=$(grep -v "S0B TEST MODE" "$ERR" | tail -1 | tr '"' "'" | cut -c1-220)
  if [ "$rc" -ne "$want" ]; then fail=$((fail+1)); fi
  printf '{"case": "%s", "rc": %d, "want": %d, "ok": %s, "stderr_last": "%s"}\n' "$name" "$rc" "$want" \
    "$([ "$rc" -eq "$want" ] && echo true || echo false)" "$msg"
}
# 1. production invocations (no PYT_S0B_TEST): no CODE FROZEN / Part B FROZEN line exists -> 3
run review_prod_all 3 -- $PY $R/pyt_axis_review.py
run review_prod_compute_E 3 -- $PY $R/pyt_axis_review.py --impl E --phase compute
run review_prod_compare 3 -- $PY $R/pyt_axis_review.py --phase compare
run review_prod_with_test_root 3 -- $PY $R/pyt_axis_review.py --test-root $TEST_IN
run fixture_prod 3 -- $PY $R/make_test_fixture_pyt.py --root $TEST_IN
run vp8_prod 3 -- $PY $R/tests/test_vp8_r.py --out $TEST_IN/x.json
run vp9_prod 3 -- $PY $R/tests/test_vp9_r.py --out $TEST_IN/x.json
# 2. test mode with roots outside R's scratch sub-root -> 3
run review_test_outside_root 3 PYT_S0B_TEST=1 -- $PY $R/pyt_axis_review.py --test-root $OUTSIDE
run review_test_out_outside 3 PYT_S0B_TEST=1 -- $PY $R/pyt_axis_review.py --test-root $TEST_IN --out $OUTSIDE/review
run review_test_out_in_results 3 PYT_S0B_TEST=1 -- $PY $R/pyt_axis_review.py --test-root $TEST_IN --out results/S1_2_layermax_pythia/review
run review_test_out_other_scratch 3 PYT_S0B_TEST=1 -- $PY $R/pyt_axis_review.py --test-root $TEST_IN --out analysis/s12_layermax_pythia/s0b_scratch/E/x
run review_test_no_root 3 PYT_S0B_TEST=1 -- $PY $R/pyt_axis_review.py
run fixture_test_outside 3 PYT_S0B_TEST=1 -- $PY $R/make_test_fixture_pyt.py --root $OUTSIDE
run vp8_test_outside 3 PYT_S0B_TEST=1 -- $PY $R/tests/test_vp8_r.py --out $OUTSIDE/x.json
run vp9_test_outside 3 PYT_S0B_TEST=1 -- $PY $R/tests/test_vp9_r.py --out $OUTSIDE/x.json
# 3. nothing written
NEW=$(find $R $SR results/S1_2_layermax_pythia -newer "$MARK" -type f -not -path "$SR/dryrun/*" -not -path "$SR/logs/*" 2>/dev/null | wc -l)
PROBE=$( [ -e "$OUTSIDE" ] && echo exists || echo absent )
REVOUT=$( [ -e results/S1_2_layermax_pythia/review ] && echo exists || echo absent )
PROBE_IN=$( [ -e "$TEST_IN" ] && echo exists || echo absent )
printf '{"summary": true, "n_cases": %d, "n_fail": %d, "new_files": %d, "outside_probe": "%s", "results_review_dir": "%s", "test_in_probe": "%s"}\n' \
  "$n" "$fail" "$NEW" "$PROBE" "$REVOUT" "$PROBE_IN"
[ "$fail" -eq 0 ] && [ "$NEW" -eq 0 ] && [ "$PROBE" = absent ] && [ "$REVOUT" = absent ] && [ "$PROBE_IN" = absent ]
