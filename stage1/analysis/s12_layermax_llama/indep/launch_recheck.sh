#!/bin/bash
# LLaMA-axis implementation I: provenance recheck of validation_vl.json with the current vl_checks.py; 2 threads
cd ~/hrh_v2_stage1_20261008
source venv/bin/activate
R=results/S1_2_layermax_llama/indep
rm -f $R/RECHECK.done $R/RECHECK.failed
export LMX_THREADS=2
python analysis/s12_layermax_llama/indep/vl_recheck.py > logs/llama_axis_indep_recheck.log 2>&1 && touch $R/RECHECK.done || touch $R/RECHECK.failed
