#!/bin/bash
# e0-run runner, plan section 11 step 6 after GUARD CLEAN (not frozen code; calls frozen e0_obs.py and the ADDENDUM 1 evaluator).
cd ~/hrh_v2_stage1_20261008 || exit 9
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
L=results/E0_sentence/logs
M=$L/markers_run.log
echo "$(date -u +%FT%TZ) START obs_new" >> $M
taskset -c 0-5 venv/bin/python -I analysis/e0_sentence/e0_obs.py --arms P,D1,D2,D3,D4,A,Aall,C,CA,CAall,L,Pdd > $L/step6_obs.log 2>&1
rc=$?; echo "$(date -u +%FT%TZ) END obs_new rc=$rc" >> $M; [ $rc -eq 0 ] || exit 1
echo "$(date -u +%FT%TZ) START evaluate_add1" >> $M
taskset -c 0-5 venv/bin/python -I analysis/e0_sentence/addendum1/e0_evaluate_add1.py > $L/step6_evaluate.log 2>&1
rc=$?; echo "$(date -u +%FT%TZ) END evaluate_add1 rc=$rc" >> $M; [ $rc -eq 0 ] || exit 2
echo "$(date -u +%FT%TZ) ALLDONE step6" >> $M
