#!/bin/bash
# launch_ol.sh -- the only way processes of the OpenLLaMA axis start (ANALYSIS_PLAN_OPENLLAMA.md section 4 F5,
# section 11 Compute). Written at S0b by role oll-axis-extract (2026-10-09); hashed in the CODE FROZEN line.
#
# usage: launch_ol.sh NAME THREADS [--gpu MIN_EST] -- CMD [ARGS...]
#
#  * Thread cap: takes THREADS of the 8 thread tokens (flock on analysis/s12_layermax_openllama/run/token.0..7,
#    non-blocking). If fewer than THREADS tokens are free it starts nothing and exits 75. The tokens are held
#    by this shell and inherited by CMD, so they stay taken until CMD and its children have exited.
#    OMP/MKL/OPENBLAS/NUMEXPR thread counts and OL_THREADS (read by the python code for torch.set_num_threads)
#    are set to THREADS.
#  * GPU budget (plan section 11, author decision R.1.1, <= 60 GPU-minutes for the whole axis): a GPU step must
#    pass --gpu MIN_EST. It is refused (exit 76) if the ledger total plus MIN_EST exceeds 60. After CMD ends,
#    its wall-clock minutes (process start to end) are appended to
#    results/S1_2_layermax_openllama/logs/gpu_ledger.tsv together with the running total.
#  * Log: $OL_LOGDIR/NAME.log (default results/S1_2_layermax_openllama/logs/); markers NAME.done or
#    NAME.failed (UTC end time, exit code, seconds) next to it. NAME.started is written at start.
#  * CMD runs from the project root ~/hrh_v2_stage1_20261008.
set -u
PROJ="$HOME/hrh_v2_stage1_20261008"
cd "$PROJ" || exit 70
NAME="$1"; TH="$2"; shift 2
GPU_EST=""
if [ "${1:-}" = "--gpu" ]; then GPU_EST="$2"; shift 2; fi
if [ "${1:-}" != "--" ]; then echo "usage: launch_ol.sh NAME THREADS [--gpu MIN_EST] -- CMD [ARGS...]" >&2; exit 64; fi
shift
case "$NAME" in *[!A-Za-z0-9_.-]*|"") echo "bad NAME" >&2; exit 64;; esac
case "$TH" in ''|*[!0-9]*) echo "bad THREADS" >&2; exit 64;; esac
if [ "$TH" -lt 1 ] || [ "$TH" -gt 8 ]; then echo "THREADS must be 1..8" >&2; exit 64; fi

RUN="$PROJ/analysis/s12_layermax_openllama/run"
LEDGER_DIR="$PROJ/results/S1_2_layermax_openllama/logs"
LEDGER="$LEDGER_DIR/gpu_ledger.tsv"
LOGDIR="${OL_LOGDIR:-$LEDGER_DIR}"
mkdir -p "$RUN" "$LOGDIR"

# ---- thread tokens -------------------------------------------------------------------------------
GOT=0
TOKS=""
for i in 0 1 2 3 4 5 6 7; do
  [ "$GOT" -ge "$TH" ] && break
  exec {fd}>>"$RUN/token.$i"
  if flock -n "$fd"; then GOT=$((GOT + 1)); TOKS="$TOKS $i"; else exec {fd}>&-; fi
done
if [ "$GOT" -lt "$TH" ]; then
  echo "[launch_ol] REFUSED $NAME: needs $TH thread tokens, only $GOT of 8 free (plan F5 cap)" >&2
  exit 75
fi

# ---- GPU budget ----------------------------------------------------------------------------------
if [ -n "$GPU_EST" ]; then
  mkdir -p "$LEDGER_DIR"
  exec {gfd}>>"$RUN/gpu.lock"
  flock "$gfd"
  TOTAL=$(awk -F'\t' 'NR>1 {s+=$5} END {printf "%.3f", s+0}' "$LEDGER" 2>/dev/null || echo 0)
  [ -z "$TOTAL" ] && TOTAL=0
  if awk -v t="$TOTAL" -v e="$GPU_EST" 'BEGIN {exit !(t + e > 60.0)}'; then
    echo "[launch_ol] REFUSED $NAME: GPU ledger $TOTAL min + estimate $GPU_EST min > 60 (plan section 11)" >&2
    exit 76
  fi
  flock -u "$gfd"
  [ -f "$LEDGER" ] || printf 'name\tstart_utc\tend_utc\tseconds\tminutes\trunning_total_min\texit\n' > "$LEDGER"
fi

export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH NUMEXPR_NUM_THREADS=$TH \
       OL_THREADS=$TH PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
LOG="$LOGDIR/$NAME.log"
rm -f "$LOGDIR/$NAME.done" "$LOGDIR/$NAME.failed"
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
T0=$(date +%s.%N)
echo "$START_UTC threads=$TH tokens=[$TOKS ] gpu_est=${GPU_EST:-none} cmd: $*" > "$LOGDIR/$NAME.started"
{ echo "[launch_ol] $START_UTC start $NAME threads=$TH tokens=[$TOKS ] gpu_est=${GPU_EST:-none}"; echo "[launch_ol] cmd: $*"; } > "$LOG"
"$@" >> "$LOG" 2>&1
RC=$?
T1=$(date +%s.%N)
END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SECS=$(awk -v a="$T0" -v b="$T1" 'BEGIN {printf "%.1f", b - a}')
if [ -n "$GPU_EST" ]; then
  flock "$gfd"
  MIN=$(awk -v s="$SECS" 'BEGIN {printf "%.3f", s / 60.0}')
  TOTAL=$(awk -F'\t' -v m="$MIN" 'NR>1 {s+=$5} END {printf "%.3f", s + m}' "$LEDGER")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$NAME" "$START_UTC" "$END_UTC" "$SECS" "$MIN" "$TOTAL" "$RC" >> "$LEDGER"
  flock -u "$gfd"
  echo "[launch_ol] GPU step $NAME: $MIN min; axis running total $TOTAL min (budget 60)" >> "$LOG"
fi
echo "[launch_ol] $END_UTC end $NAME rc=$RC ${SECS}s" >> "$LOG"
if [ "$RC" -eq 0 ]; then
  echo "$END_UTC rc=0 ${SECS}s" > "$LOGDIR/$NAME.done"
else
  echo "$END_UTC rc=$RC ${SECS}s" > "$LOGDIR/$NAME.failed"
fi
exit "$RC"
