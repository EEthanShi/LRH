#!/bin/bash
# S1.3a full pipeline: train -> criteria -> analyze. Writes linae/S1_3a.done or linae/S1_3a.failed.
set -o pipefail
cd ~/hrh_v2_stage1_20261008/linae
PY=../venv/bin/python
ST=../STATUS.md
rm -f S1_3a.done S1_3a.failed
T0=$(date +%s)
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [linae] S1.3a training launched (logs/linae_s13a.log, linae/runs/)" >> $ST
if $PY run_s13a.py train && \
   echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [linae] S1.3a training finished; computing criteria (linae/criteria/)" >> $ST && \
   $PY run_s13a.py criteria && $PY analyze_s13a.py; then
  echo "elapsed_s=$(( $(date +%s) - T0 ))" > S1_3a.done
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [linae] S1.3a pipeline done in $(( $(date +%s) - T0 ))s (results/S1_3a_linae_summary.md/.json)" >> $ST
else
  echo "elapsed_s=$(( $(date +%s) - T0 ))" > S1_3a.failed
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [linae] S1.3a pipeline FAILED (logs/linae_s13a.log)" >> $ST
fi
