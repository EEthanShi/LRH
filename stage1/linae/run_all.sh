#!/bin/bash
# S1.3a full pipeline (v1.1): unit tests -> train (resumable; skips complete chunks) -> criteria -> analyze.
# Writes linae/S1_3a.done or linae/S1_3a.failed. Safe to relaunch: completed chunks are reused.
set -o pipefail
cd ~/hrh_v2_stage1_20261008/linae
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=../venv/bin/python
ST=../STATUS.md
rm -f S1_3a.done S1_3a.failed
T0=$(date +%s)
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [linae] S1.3a v1.1 pipeline (re)launched, NPROC=${LINAE_NPROC:-12}, resuming from existing chunks (logs/linae_s13a_v11.log, linae/runs/)" >> $ST
if $PY -m pytest -q -p no:cacheprovider test_linae.py && \
   $PY run_s13a.py train && \
   echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [linae] S1.3a training finished (all chunks complete); computing criteria (linae/criteria/)" >> $ST && \
   $PY run_s13a.py criteria && $PY analyze_s13a.py; then
  echo "elapsed_s=$(( $(date +%s) - T0 ))" > S1_3a.done
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [linae] S1.3a v1.1 pipeline done in $(( $(date +%s) - T0 ))s (results/S1_3a_linae_summary.md/.json)" >> $ST
else
  echo "elapsed_s=$(( $(date +%s) - T0 ))" > S1_3a.failed
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [linae] S1.3a v1.1 pipeline FAILED (logs/linae_s13a_v11.log; per-chunk tracebacks in linae/runs/*/*/chunk*.failed)" >> $ST
fi
