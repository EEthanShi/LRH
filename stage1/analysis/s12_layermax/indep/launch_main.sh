#!/bin/bash
cd ~/hrh_v2_stage1_20261008
source venv/bin/activate
rm -f results/S1_2_layermax/indep/RUN_main.done results/S1_2_layermax/indep/RUN_main.failed
export LMX_THREADS=8
( python analysis/s12_layermax/indep/run_layermax.py --pre prh_clip+l2 && python analysis/s12_layermax/indep/run_layermax.py --pre l2 ) > logs/lmx_indep_main.log 2>&1 && touch results/S1_2_layermax/indep/RUN_main.done || touch results/S1_2_layermax/indep/RUN_main.failed
