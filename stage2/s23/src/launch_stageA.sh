#!/usr/bin/env bash
# S2.3 Stage A launcher (PREREG section 13). Runs only after the 'S2.3 PREREG v1.0 + CODE FROZEN' line.
# Concurrency: 2 GPU workers + 1 CPU worker (1 BLAS thread each) <= 4 cores; never kills other jobs.
set -u
S23=~/hrh_v2_stage2_20261008/s23
PY=~/hrh_v2_stage1_20261008/venv/bin/python
cd "$S23/src"
grep -q "S2.3 PREREG v1.0 + CODE FROZEN" ~/hrh_v2_stage1_20261008/STATUS.md || { echo "not frozen"; exit 2; }
mkdir -p "$S23/logs"
run_pool () {  # $1 = phase
  "$PY" worker.py --phase "$1" --pool gpu --wid 0 --nw 2 >> "$S23/logs/w_gpu0_$1.out" 2>&1 & p1=$!
  "$PY" worker.py --phase "$1" --pool gpu --wid 1 --nw 2 >> "$S23/logs/w_gpu1_$1.out" 2>&1 & p2=$!
  "$PY" worker.py --phase "$1" --pool cpu --wid 0 --nw 1 >> "$S23/logs/w_cpu0_$1.out" 2>&1 & p3=$!
  rc=0
  wait $p1 || rc=$?
  wait $p2 || rc=$?
  wait $p3 || rc=$?
  return $rc
}
"$PY" prestep.py >> "$S23/logs/prestep.out" 2>&1 || { echo "pre-step gates failed (stop)"; exit 3; }
"$PY" make_plan.py >> "$S23/logs/make_plan.out" 2>&1 || exit 3
run_pool pre || { echo "pre phase stopped (budget or error)"; exit 4; }
"$PY" analyze_A.py --phase gates >> "$S23/logs/analyze_gates.out" 2>&1 || { echo "sampled gates did not allow main runs (stop)"; exit 3; }
run_pool main || { echo "main phase stopped (budget or error)"; exit 4; }
"$PY" analyze_A.py --phase nmatch >> "$S23/logs/analyze_nmatch.out" 2>&1 || exit 5
"$PY" analyze_A.py --phase main >> "$S23/logs/analyze_main.out" 2>&1 || exit 5
echo "Stage A done"
