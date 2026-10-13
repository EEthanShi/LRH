#!/bin/bash
# LLaMA-axis implementation I, S8: aggregation (summary.json) then REPORT.md; 3 threads, CPU only
cd ~/hrh_v2_stage1_20261008
source venv/bin/activate
R=results/S1_2_layermax_llama/indep
rm -f $R/AGG.done $R/AGG.failed
export LMX_THREADS=3
( python analysis/s12_layermax_llama/indep/lmx_llama_aggregate.py && python analysis/s12_layermax_llama/indep/lmx_llama_report.py ) > logs/llama_axis_indep_aggregate.log 2>&1 && touch $R/AGG.done || touch $R/AGG.failed
