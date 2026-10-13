#!/bin/bash
# LLaMA-axis implementation I: cross-implementation comparison with E (V-L5 + summary level), then REPORT.md regenerated with section 7b; 3 threads, CPU only
cd ~/hrh_v2_stage1_20261008
source venv/bin/activate
R=results/S1_2_layermax_llama/indep
rm -f $R/CROSS.done $R/CROSS.failed
export LMX_THREADS=3
( python analysis/s12_layermax_llama/indep/cross_validate.py && python analysis/s12_layermax_llama/indep/lmx_llama_report.py ) > logs/llama_axis_indep_cross.log 2>&1 && touch $R/CROSS.done || touch $R/CROSS.failed
