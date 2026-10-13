cd ~/hrh_v2_stage1_20261008/hrhv2_ref && source ../venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python scripts/sog_crosscheck.py results/sog_crosscheck.json 1024 4096 && touch results/sog_crosscheck.done || touch results/sog_crosscheck.failed
