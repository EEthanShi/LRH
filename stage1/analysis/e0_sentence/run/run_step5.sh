#!/bin/bash
# e0-run runner, plan section 11 step 5 (not frozen code; calls only frozen scripts). CPU only, pinned to 6 cores.
cd ~/hrh_v2_stage1_20261008 || exit 9
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
L=results/E0_sentence/logs
M=$L/markers_run.log
step () { echo "$(date -u +%FT%TZ) START $1" >> $M; shift_name=$1; shift; taskset -c 0-5 "$@" > $L/step5_$shift_name.log 2>&1; rc=$?; echo "$(date -u +%FT%TZ) END $shift_name rc=$rc" >> $M; return $rc; }
step prep venv/bin/python -I analysis/e0_sentence/e0_prep.py --workers 6 --threads 1 || exit 1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
step diag venv/bin/python -I analysis/e0_sentence/e0_diag.py || exit 2
step obs_ctrold venv/bin/python -I analysis/e0_sentence/e0_obs.py --arms ctrold || exit 3
step calib venv/bin/python -I analysis/e0_sentence/e0_calib.py || exit 4
echo "$(date -u +%FT%TZ) ALLDONE step5" >> $M
