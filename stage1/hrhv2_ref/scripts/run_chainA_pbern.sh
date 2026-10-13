cd ~/hrh_v2_stage1_20261008/hrhv2_ref && source ../venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python scripts/pbern_scale_check.py results/pbern_ref_small.json 256 1024 && touch results/pbern_ref_small.done || touch results/pbern_ref_small.failed
python scripts/pbern_scale_check.py results/pbern_ref_4096.json 4096 && touch results/pbern_ref_4096.done || touch results/pbern_ref_4096.failed
