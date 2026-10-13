#!/usr/bin/env bash
# V-f (role e0-indep) runner: CPU only, pinned to cores 0-5, 6 single-thread workers.
set -euo pipefail
ROOT=~/hrh_v2_stage1_20261008
HERE=$ROOT/analysis/e0_sentence/indep
PY=$ROOT/venv/bin/python
RUN=${1:?run id}
OUT=$HERE/run_$RUN
KNN=$OUT/knn
mkdir -p "$OUT" "$KNN"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "start $(date -u +%FT%TZ)" >> "$OUT/run.log"
taskset -c 0-5 nice -n 10 $PY "$HERE/vf_design_check.py" "$OUT" >> "$OUT/run.log" 2>&1
models=$($PY -c "import sys; sys.path.insert(0,'$HERE'); import vf_common as C; print(' '.join(C.model_names()))")
printf '%s\n' $models | taskset -c 0-5 nice -n 10 xargs -P 6 -I{} $PY "$HERE/vf_knn.py" "$KNN" {} >> "$OUT/run.log" 2>&1
taskset -c 0-5 nice -n 10 $PY "$HERE/vf_pairs.py" "$KNN" "$OUT" >> "$OUT/run.log" 2>&1
echo "end $(date -u +%FT%TZ)" >> "$OUT/run.log"
touch "$OUT/VF_OBS.done"
