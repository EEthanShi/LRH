#!/bin/bash
# x3-indep launcher: 8 neighbour-table builds (6 at a time, 1 BLAS thread each),
# then overlaps/cells, then statistics. Everything pinned to cores 0-5.
set -euo pipefail
ROOT="$HOME/hrh_v2_stage1_20261008"
HERE="$ROOT/analysis/e0_sentence/x3/indep"
RUN="$1"
mkdir -p "$RUN/knn" "$RUN/logs"
export PYTHONPATH="$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$ROOT/venv/bin/python"
cd "$ROOT"
MODELS="x3_cls_r10 x3_tok_r10 x3_tok_r05 x3add_schnaus224 dinov2_b mpnet.n5 contriever.n5 qwen3emb06.n5"
export PY HERE RUN
printf '%s\n' $MODELS | xargs -P 6 -I{} sh -c 'taskset -c 0-5 nice -n 10 "$PY" "$HERE/x3i_knn.py" "$RUN" "{}" > "$RUN/logs/knn_{}.log" 2>&1'
taskset -c 0-5 nice -n 10 "$PY" "$HERE/x3i_obs.py" "$RUN" > "$RUN/logs/obs.log" 2>&1
taskset -c 0-5 nice -n 10 "$PY" "$HERE/x3i_stats.py" "$RUN" > "$RUN/logs/stats.log" 2>&1
echo DONE
