#!/usr/bin/env bash
# S1.1 v1.1 (descriptive) pipeline after phase A (s11v11_prep.py):
#   obs -> nulls units (all configurations) -> interim evaluation ->
#   unvalidated predictor (iii) in priority order -> final evaluation.
# Each stage is resumable (existing outputs are skipped). Markers:
#   results/S1_1/v11/PIPELINE_<stage>.done|failed, PIPELINE.done|failed at the end.
# Launch:  cd ~/hrh_v2_stage1_20261008; setsid nohup bash analysis/s11/s11v11_pipeline.sh \
#              > results/S1_1/v11/logs/pipeline.log 2>&1 < /dev/null &
set -u
P=~/hrh_v2_stage1_20261008
cd "$P" || exit 1
PY="$P/venv/bin/python"
OUT="$P/results/S1_1/v11"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
rm -f "$OUT"/PIPELINE.done "$OUT"/PIPELINE.failed

stage () {   # name, command...
  local name="$1"; shift
  if [ -f "$OUT/PIPELINE_${name}.done" ]; then echo "[$(date -u +%H:%M:%S)] skip $name (done)"; return 0; fi
  echo "[$(date -u +%H:%M:%S)] start $name: $*"
  if "$@" > "$OUT/logs/${name}.log" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/PIPELINE_${name}.done"; echo "[$(date -u +%H:%M:%S)] done $name"
  else
    date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/PIPELINE_${name}.failed"; echo "[$(date -u +%H:%M:%S)] FAILED $name"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/PIPELINE.failed"; exit 1
  fi
}

soft () {    # evaluation stages: a failure is logged, the pipeline continues
  local name="$1"; shift
  echo "[$(date -u +%H:%M:%S)] start $name (soft): $*"
  if "$@" > "$OUT/logs/${name}.log" 2>&1; then echo "[$(date -u +%H:%M:%S)] done $name";
  else echo "[$(date -u +%H:%M:%S)] FAILED (soft) $name"; fi
}

[ -f "$OUT/PREP.done" ] || { echo "PREP not done"; date -u > "$OUT/PIPELINE.failed"; exit 1; }
# v2 order (10:58Z): 'iii' units include obs/N0/N0H1/N1/P2/N2, so primary pairs get 'iii' directly;
# 'nulls' units only where no 'iii' unit is planned before them.
stage obs        "$PY" analysis/s11/s11v11_obs.py --configs clip,l2,cliplayer,dedup
stage iii_clip_p "$PY" analysis/s11/s11v11_units.py --kind iii --configs clip --which primary --workers 11 --tag iii_clip_primary
soft eval1       "$PY" analysis/s11/s11v11_evaluate.py
stage iii_l2_p   "$PY" analysis/s11/s11v11_units.py --kind iii --configs l2 --which primary --workers 11 --tag iii_l2_primary
stage iii_dedup  "$PY" analysis/s11/s11v11_units.py --kind iii --configs dedup --workers 11 --tag iii_dedup_primary
soft eval2       "$PY" analysis/s11/s11v11_evaluate.py
stage nul_cl     "$PY" analysis/s11/s11v11_units.py --kind nulls --configs cliplayer --workers 11 --tag nulls_cliplayer_primary
stage nul_sec    "$PY" analysis/s11/s11v11_units.py --kind nulls --configs clip,l2 --which secondary --workers 11 --tag nulls_clip-l2_secondary
soft eval3       "$PY" analysis/s11/s11v11_evaluate.py
stage iii_clip_s "$PY" analysis/s11/s11v11_units.py --kind iii --configs clip --which secondary --workers 11 --tag iii_clip_secondary
stage iii_l2_s   "$PY" analysis/s11/s11v11_units.py --kind iii --configs l2 --which secondary --workers 11 --tag iii_l2_secondary
stage iii_cl     "$PY" analysis/s11/s11v11_units.py --kind iii --configs cliplayer --workers 11 --tag iii_cliplayer_primary
soft eval4       "$PY" analysis/s11/s11v11_evaluate.py
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/PIPELINE.done"
echo "[$(date -u +%H:%M:%S)] pipeline finished"
