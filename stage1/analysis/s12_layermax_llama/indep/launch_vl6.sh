#!/bin/bash
# LLaMA-axis implementation I, V-L6 (vendored ARH, our permutations injected); 3 threads
cd ~/hrh_v2_stage1_20261008
source venv/bin/activate
R=results/S1_2_layermax_llama/indep
rm -f $R/VL6.done $R/VL6.failed
export LMX_THREADS=3
python analysis/s12_layermax_llama/indep/vl6_arh.py > logs/llama_axis_indep_vl6.log 2>&1 && touch $R/VL6.done || touch $R/VL6.failed
