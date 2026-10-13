#!/bin/bash
cd ~/hrh_v2_stage1_20261008
source venv/bin/activate
export LMX_THREADS=4
rm -f results/S1_2_layermax/indep/I4.done results/S1_2_layermax/indep/I4.failed
python analysis/s12_layermax/indep/validate.py --arh > logs/lmx_indep_i4.log 2>&1 && touch results/S1_2_layermax/indep/I4.done || touch results/S1_2_layermax/indep/I4.failed
