#!/bin/bash
# LLaMA-axis implementation I: validate.py (I1 = clip/ties, I3 = V-L4a, I2 = V-L3) then vl_checks.py (V-L1, V-L4b, V-L7, V-L2); 3 threads
cd ~/hrh_v2_stage1_20261008
source venv/bin/activate
R=results/S1_2_layermax_llama/indep
rm -f $R/VAL.done $R/VAL.failed
export LMX_THREADS=3
( python analysis/s12_layermax_llama/indep/validate.py && python analysis/s12_layermax_llama/indep/vl_checks.py --items vl1,vl4b,vl7,vl2 ) > logs/llama_axis_indep_validate.log 2>&1 && touch $R/VAL.done || touch $R/VAL.failed
