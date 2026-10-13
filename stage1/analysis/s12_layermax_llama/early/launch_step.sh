#!/bin/bash
# LLAMA-AXIS COPY of analysis/s12_layermax/launch_step.sh (sha256 dcb82ee2...): paths changed only
# (script dir analysis/s12_layermax_llama/early, logs/llama_early_NAME.log,
#  markers results/S1_2_layermax_llama/early/NAME.{done,failed}); thread cap <= 6.
# usage: launch_step.sh NAME THREADS script.py [args...]
cd ~/hrh_v2_stage1_20261008
NAME=$1; TH=$2; shift 2
if [ "$TH" -gt 6 ]; then TH=6; fi
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH NUMEXPR_NUM_THREADS=$TH LLAMA_AXIS_THREADS=$TH VERIF_TORCH_THREADS=$TH PYTHONUNBUFFERED=1
mkdir -p results/S1_2_layermax_llama/early
rm -f results/S1_2_layermax_llama/early/$NAME.done results/S1_2_layermax_llama/early/$NAME.failed
if venv/bin/python analysis/s12_layermax_llama/early/"$@" > logs/llama_early_$NAME.log 2>&1; then
  date -u +%Y-%m-%dT%H:%M:%SZ > results/S1_2_layermax_llama/early/$NAME.done
else
  date -u +%Y-%m-%dT%H:%M:%SZ > results/S1_2_layermax_llama/early/$NAME.failed
fi
