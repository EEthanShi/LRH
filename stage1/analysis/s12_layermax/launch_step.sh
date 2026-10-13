#!/bin/bash
# usage: launch_step.sh NAME THREADS script.py [args...]
# runs one step with logs/s12lm_NAME.log and markers results/S1_2_layermax/NAME.{done,failed}
cd ~/hrh_v2_stage1_20261008
NAME=$1; TH=$2; shift 2
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH NUMEXPR_NUM_THREADS=$TH PYTHONUNBUFFERED=1
rm -f results/S1_2_layermax/$NAME.done results/S1_2_layermax/$NAME.failed
if venv/bin/python analysis/s12_layermax/"$@" > logs/s12lm_$NAME.log 2>&1; then
  date -u +%Y-%m-%dT%H:%M:%SZ > results/S1_2_layermax/$NAME.done
else
  date -u +%Y-%m-%dT%H:%M:%SZ > results/S1_2_layermax/$NAME.failed
fi
