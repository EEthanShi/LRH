#!/bin/bash
# Launch NW queue workers (1 CPU thread each). Plans in order: S2.2 pilot, S2.1 pilot, S2.2 main, S2.1 main.
# Main plans wait for runs/GATE_*.go written by src/gates.py. Resumable: re-running this continues the queue.
cd ~/hrh_v2_stage2_20261008
NW=${1:-3}
PY=~/hrh_v2_stage1_20261008/venv/bin/python
for w in $(seq 0 $((NW-1))); do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 setsid nohup $PY src/train.py worker --wid $w \
    --plan plans/S2_2_pilot.json plans/S2_1_pilot.json plans/S2_2_main.json plans/S2_1_main.json \
    > logs/worker_$w.log 2>&1 < /dev/null &
  echo "worker $w pid $!"
  sleep 2
done
