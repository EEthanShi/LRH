#!/bin/bash
# Pythia axis, implementation I: every S0b test of implementation I in one go (NEW, PYT-AXIS, sub-role pyt-code-I,
# 2026-10-10). CPU only, known or synthetic data only, through launch_pyt.sh (a step waits up to 15 min for free thread
# tokens, retrying every 10 s); S0b test mode (PYT_S0B_TEST=1, refused after CODE FROZEN). Logs and outputs under
# analysis/s12_layermax_pythia/s0b_scratch/I/ (VERDICTS_SPEC_PYT.md section 13). The V-P9 fixture uses the shared
# stand-in Part B s0b_scratch/PB/partB_vp9/ (partb_pyt.py --standin vp9, written by the infrastructure role).
# usage: bash analysis/s12_layermax_pythia/indep/tests/run_all_s0b.sh [THREADS<=4]
set -u
PROJ="$HOME/hrh_v2_stage1_20261008"
cd "$PROJ" || exit 70
TH="${1:-4}"
UTC=$(date -u +%Y%m%dT%H%M%SZ)
export PYT_S0B_TEST=1
T="$PROJ/analysis/s12_layermax_pythia/s0b_scratch/I"
export PYT_LOGDIR="$T/logs"
L="$PROJ/analysis/s12_layermax_pythia/launch_pyt.sh"
I="$PROJ/analysis/s12_layermax_pythia/indep"
PB="$PROJ/analysis/s12_layermax_pythia/s0b_scratch/PB/partB_vp9/partB_features_pyt.json"
PY="$PROJ/venv/bin/python"
declare -A RC
step() {
  local name="$1"; shift
  local rc=75 i
  for i in $(seq 90); do "$L" "pyt_I_${name}_$UTC" "$TH" -- "$@"; rc=$?; [ "$rc" -ne 75 ] && break; sleep 10; done
  RC[$name]=$rc; echo "step $name rc=$rc"
}
step vp9 "$PY" "$I/tests/test_pyt_verdicts.py"
FX1="$T/fixture_${UTC}_a"; FX2="$T/fixture_${UTC}_b"
PYT_I_TEST_OUT="$FX1" step fixture_agg_a "$PY" "$I/lmx_pyt_aggregate.py" --fixture --partb "$PB"
PYT_I_TEST_OUT="$FX2" step fixture_agg_b "$PY" "$I/lmx_pyt_aggregate.py" --fixture --partb "$PB"
step determinism "$PY" -c "
import json,sys
a=json.load(open('$FX1/verdicts_pyt.json'))['v']; b=json.load(open('$FX2/verdicts_pyt.json'))['v']
bad=[k for k in set(a)|set(b) if a.get(k)!=b.get(k) and not (a.get(k)!=a.get(k) and b.get(k)!=b.get(k))]
print('keys', len(a), 'differ', len(bad), bad[:5]); sys.exit(1 if bad or len(a)!=len(b) else 0)"
PYT_I_TEST_OUT="$FX1" step cross_report "$PY" "$I/tests/test_cross_report.py" "$FX1"
PYT_I_TEST_OUT="$T/fixture_${UTC}_primary_only" step fixture_agg_primary_only "$PY" "$I/lmx_pyt_aggregate.py" --fixture \
     --partb "$PB" --pres prh_clip+l2
SC="$PROJ/analysis/s12_layermax_pythia/s0b_scratch"
step forecast_on_I "$PY" "$PROJ/analysis/s12_layermax_pythia/forecast_pyt.py" --impl I --test-root "$SC/PB/fixture_vp9" \
     --partb "$PB" --verdicts "$FX1/verdicts_pyt.json" --out "$T/forecast_on_I_$UTC"
step forecast_xcheck "$PY" "$I/tests/test_forecast_xcheck.py" "$FX1/verdicts_pyt.json" "$T/forecast_on_I_$UTC/verdicts_forecast.json"
PYT_I_TEST_OUT="$T/legacy_$UTC" step legacy_file "$PY" "$I/lmx_pyt_aggregate.py" --legacy
step vp8 "$PY" "$I/vp8_legacy.py"
step gate_dryrun "$PY" "$I/tests/test_gate_dryrun.py"
bash "$I/tests/cpu_smoke.sh" "$TH"; RC[cpu_smoke]=$?; echo "step cpu_smoke rc=${RC[cpu_smoke]}"
echo "run_all_s0b $UTC:"; for k in "${!RC[@]}"; do echo "  $k rc=${RC[$k]}"; done
