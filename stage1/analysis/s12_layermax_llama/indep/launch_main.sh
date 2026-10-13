#!/bin/bash
# LLaMA-axis implementation I, main run (51 pairs x {prh_clip+l2, l2}); 3 threads, GPU fraction 0.30
cd ~/hrh_v2_stage1_20261008
source venv/bin/activate
R=results/S1_2_layermax_llama/indep
mkdir -p $R
rm -f $R/RUN_main.done $R/RUN_main.failed
export LMX_THREADS=3
( python analysis/s12_layermax_llama/indep/run_layermax.py --pre prh_clip+l2 && python analysis/s12_layermax_llama/indep/run_layermax.py --pre l2 ) > logs/llama_axis_indep_main.log 2>&1 && touch $R/RUN_main.done || touch $R/RUN_main.failed
