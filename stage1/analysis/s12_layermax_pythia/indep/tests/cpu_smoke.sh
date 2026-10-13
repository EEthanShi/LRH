#!/bin/bash
# Pythia axis, implementation I: S0b CPU smoke of the run path on KNOWN data only (NEW, PYT-AXIS, sub-role pyt-code-I,
# 2026-10-10). No GPU, no Pythia byte. Each step goes through launch_pyt.sh (thread tokens; logs in
# analysis/s12_layermax_pythia/s0b_scratch/I/logs/). Outputs only under analysis/s12_layermax_pythia/s0b_scratch/I/
# cpu_smoke_<UTC>/ (S0b test mode, PYT_S0B_TEST=1; refused after CODE FROZEN).
#   1. run_layermax.py --known --device cpu --K 8, both pre: the 4 known pairs of plan V-P4 (identity + pi_0..pi_7)
#   2. validate.py --known-pairs --K-test 8: V-P3 arithmetic (I2) and V-P7(b) on those pairs, I1 clip/tie summary
#   3. vl_checks.py vp4 (vs the frozen I files, first 8 permutations), vl7/vl7d (PR of the frozen OpenLLaMA Part B),
#      vl7c (m = 8 plant)
#   4. vl6_arh.py --device cpu --K-test 8 --lm bigscience__bloomz-560m: V-P6 scalar and grid5 x grid5 (vendored ARH)
# usage: bash analysis/s12_layermax_pythia/indep/tests/cpu_smoke.sh [THREADS]
set -u
PROJ="$HOME/hrh_v2_stage1_20261008"
cd "$PROJ" || exit 70
TH="${1:-4}"
UTC=$(date -u +%Y%m%dT%H%M%SZ)
export PYT_S0B_TEST=1
export PYT_LOGDIR="$PROJ/analysis/s12_layermax_pythia/s0b_scratch/I/logs"
export PYT_I_TEST_OUT="$PROJ/analysis/s12_layermax_pythia/s0b_scratch/I/cpu_smoke_$UTC"
L="$PROJ/analysis/s12_layermax_pythia/launch_pyt.sh"
I="$PROJ/analysis/s12_layermax_pythia/indep"
PY="$PROJ/venv/bin/python"
OLPB="$PROJ/results/S1_2_layermax_openllama/partB/partB_features_ol.json"
RC=0
run() {   # launch_pyt.sh with a wait for free thread tokens (exit 75 = tokens busy; retry every 10 s, up to 15 min)
  local rc=75 i
  for i in $(seq 90); do "$L" "$@"; rc=$?; [ "$rc" -ne 75 ] && break; sleep 10; done
  return $rc
}
for pre in prh_clip+l2 l2; do
  tag=$( [ "$pre" = "l2" ] && echo l2 || echo prh )
  run "pyt_I_smoke_run_${tag}_$UTC" "$TH" -- "$PY" "$I/run_layermax.py" --pre "$pre" --known --device cpu --K 8 \
       --mem-gb 1.5 || RC=1
done
run "pyt_I_smoke_validate_$UTC" "$TH" -- "$PY" "$I/validate.py" --known-pairs --K-test 8 \
     --pairs-root "$PYT_I_TEST_OUT/known" || RC=1
run "pyt_I_smoke_vl_$UTC" "$TH" -- "$PY" "$I/vl_checks.py" --items vp4,vl7,vl7d,vl7c --K-test 8 \
     --pairs-root "$PYT_I_TEST_OUT" --partb "$OLPB" --vp7-pairs known || RC=1
run "pyt_I_smoke_vp6_$UTC" "$TH" -- "$PY" "$I/vl6_arh.py" --device cpu --K-test 8 --lm bigscience__bloomz-560m \
     --pairs-root "$PYT_I_TEST_OUT/known" || RC=1
echo "cpu smoke $UTC rc=$RC outputs $PYT_I_TEST_OUT"
exit $RC
