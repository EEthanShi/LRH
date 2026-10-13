#!/usr/bin/env bash
# S1.1 v1.1 pipeline, order v3 (2026-10-08 ~11:35Z): measured real-data cost ~900 s per 'iii' unit
# and ~250 s per 'nulls' unit (exact k=1 terms, saturated couplings), so after the running
# 'iii' clip-primary job the remaining configurations get nulls (N0/N1/P2/N2) first and the
# unvalidated predictor only afterwards, while time remains (stop near ~20 h wall).
# Waits for the units job started by s11v11_pipeline.sh (pid given as $1) before starting.
set -u
P=~/hrh_v2_stage1_20261008
cd "$P" || exit 1
PY="$P/venv/bin/python"
OUT="$P/results/S1_1/v11"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
rm -f "$OUT"/PIPELINE.done "$OUT"/PIPELINE.failed
W=11
stage () {
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
soft () {
  local name="$1"; shift
  echo "[$(date -u +%H:%M:%S)] start $name (soft): $*"
  if "$@" > "$OUT/logs/${name}.log" 2>&1; then echo "[$(date -u +%H:%M:%S)] done $name";
  else echo "[$(date -u +%H:%M:%S)] FAILED (soft) $name"; fi
}
if [ -n "${1:-}" ]; then
  echo "[$(date -u +%H:%M:%S)] waiting for running units job pid $1"
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
  echo "[$(date -u +%H:%M:%S)] pid $1 exited"
fi
U=analysis/s11/s11v11_units.py
stage iii_clip_p "$PY" $U --kind iii --configs clip --which primary --workers $W --tag iii_clip_primary
soft eval1       "$PY" analysis/s11/s11v11_evaluate.py
stage nul_l2_p   "$PY" $U --kind nulls --configs l2 --which primary --workers $W --tag nulls_l2_primary
stage nul_dedup  "$PY" $U --kind nulls --configs dedup --workers $W --tag nulls_dedup_primary
stage nul_cl     "$PY" $U --kind nulls --configs cliplayer --workers $W --tag nulls_cliplayer_primary
soft eval2       "$PY" analysis/s11/s11v11_evaluate.py
stage nul_sec    "$PY" $U --kind nulls --configs clip,l2 --which secondary --workers $W --tag nulls_clip-l2_secondary
soft eval3       "$PY" analysis/s11/s11v11_evaluate.py
stage iii_l2_p   "$PY" $U --kind iii --configs l2 --which primary --workers $W --tag iii_l2_primary
soft eval4       "$PY" analysis/s11/s11v11_evaluate.py
stage iii_dedup  "$PY" $U --kind iii --configs dedup --workers $W --tag iii_dedup_primary
stage iii_clip_s "$PY" $U --kind iii --configs clip --which secondary --workers $W --tag iii_clip_secondary
stage iii_cl     "$PY" $U --kind iii --configs cliplayer --workers $W --tag iii_cliplayer_primary
soft eval5       "$PY" analysis/s11/s11v11_evaluate.py
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/PIPELINE.done"
echo "[$(date -u +%H:%M:%S)] pipeline finished"
