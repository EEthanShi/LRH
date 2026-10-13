#!/bin/bash
# X3b independent check (role x3b-indep), as run on 2026-10-10. EXPLORATORY. CPU only.
# Limits from the task: taskset -c 0-5, nice 10, 1 BLAS thread per worker.
set -euo pipefail
cd ~/hrh_v2_stage1_20261008/analysis/e0_sentence/x3b/indep
source ~/hrh_v2_stage1_20261008/venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=${1:-run_$(date -u +%Y%m%dT%H%M%SZ)}; mkdir -p "$R"
taskset -c 0-5 nice -n 10 python x3bi_knn.py "$R" 6 > "$R/knn.log" 2>&1     # 51 models x 5 draws x 7 prefixes
taskset -c 0-5 nice -n 10 python x3bi_obs.py "$R" > "$R/obs.log" 2>&1        # tables vs run caches, per-anchor counts, cells
taskset -c 0-5 nice -n 10 python x3bi_stats.py "$R"                           # statistics, readings, comparison
taskset -c 0-5 nice -n 10 python x3bi_manifest.py "$R" "${CPU_CORE_HOURS:-0}"
