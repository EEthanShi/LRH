#!/bin/bash
cd ~/hrh_v2_stage1_20261008
source venv/bin/activate
export LMX_THREADS=4
rm -f results/S1_2_layermax/indep/I4cka.done results/S1_2_layermax/indep/I4cka.failed
python analysis/s12_layermax/indep/validate_i4b.py --arh --arh-metric cka_b > logs/lmx_indep_i4_cka.log 2>&1 && touch results/S1_2_layermax/indep/I4cka.done || touch results/S1_2_layermax/indep/I4cka.failed
