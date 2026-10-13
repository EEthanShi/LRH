#!/bin/bash
# launch_pyt.sh -- the only way processes of the Pythia axis start (ANALYSIS_PLAN_PYTHIA.md section 4 F5,
# section 11 Compute). PYT-AXIS COPY of analysis/s12_layermax_openllama/launch_ol.sh (sha256 3675d6f2...), written at
# S0b by role pyt-axis-extract (2026-10-10); hashed in the CODE FROZEN line. Changes, all marked PYT-AXIS: 6 thread
# tokens (lead budget, plan R.4 "thread cap 6"), PYT_THREADS, the Pythia-axis ledger and run directory, and the
# no-delete rule (an existing NAME.done / NAME.failed marker is never removed: the launch is refused, use a new NAME).
#
# usage: launch_pyt.sh NAME THREADS [--gpu MIN_EST] -- CMD [ARGS...]
#
#  * Thread cap: takes THREADS of the 6 thread tokens (flock on analysis/s12_layermax_pythia/run/token.0..5,
#    non-blocking). If fewer than THREADS tokens are free it starts nothing and exits 75. The tokens are held
#    by this shell and inherited by CMD, so they stay taken until CMD and its children have exited.
#    OMP/MKL/OPENBLAS/NUMEXPR thread counts and PYT_THREADS (read by the python code for torch.set_num_threads)
#    are set to THREADS.
#  * GPU budget (plan section 11, R.1.4, <= 60 GPU-minutes for the whole axis): a GPU step must pass --gpu MIN_EST.
#    It is refused (exit 76) if the ledger total plus MIN_EST exceeds 60. After CMD ends, its wall-clock minutes
#    (process start to end) are appended to results/S1_2_layermax_pythia/logs/gpu_ledger.tsv with the running total.
#    The G0/G1 deferral gates are separate (gpu_gate_pyt.py) and run before the step.
#  * Log: $PYT_LOGDIR/NAME.log (default results/S1_2_layermax_pythia/logs/); markers NAME.done or NAME.failed (UTC end
#    time, exit code, seconds) next to it. NAME.started is written at start. A NAME whose .done/.failed/.started
#    marker exists is refused (exit 65; PYT-AXIS no-delete rule).
#  * CMD runs from the project root ~/hrh_v2_stage1_20261008.
set -u
PROJ="$HOME/hrh_v2_stage1_20261008"
cd "$PROJ" || exit 70
NAME="${1:-}"; TH="${2:-}"; shift 2 || { echo "usage: launch_pyt.sh NAME THREADS [--gpu MIN_EST] -- CMD [ARGS...]" >&2; exit 64; }
GPU_EST=""
if [ "${1:-}" = "--gpu" ]; then GPU_EST="$2"; shift 2; fi
if [ "${1:-}" != "--" ]; then echo "usage: launch_pyt.sh NAME THREADS [--gpu MIN_EST] -- CMD [ARGS...]" >&2; exit 64; fi
shift
case "$NAME" in *[!A-Za-z0-9_.-]*|"") echo "bad NAME" >&2; exit 64;; esac
case "$TH" in ''|*[!0-9]*) echo "bad THREADS" >&2; exit 64;; esac
if [ "$TH" -lt 1 ] || [ "$TH" -gt 6 ]; then echo "THREADS must be 1..6" >&2; exit 64; fi  # PYT-AXIS
if [ -n "$GPU_EST" ]; then case "$GPU_EST" in ''|*[!0-9.]*) echo "bad MIN_EST" >&2; exit 64;; esac; fi  # PYT-AXIS

RUN="$PROJ/analysis/s12_layermax_pythia/run"  # PYT-AXIS
LEDGER_DIR="$PROJ/results/S1_2_layermax_pythia/logs"  # PYT-AXIS
LEDGER="$LEDGER_DIR/gpu_ledger.tsv"
LOGDIR="${PYT_LOGDIR:-$LEDGER_DIR}"  # PYT-AXIS
mkdir -p "$RUN" "$LOGDIR"

# PYT-AXIS: no-delete rule -- never remove an earlier marker; a re-run needs a new NAME
for mk in "$LOGDIR/$NAME.done" "$LOGDIR/$NAME.failed" "$LOGDIR/$NAME.started"; do
  if [ -e "$mk" ]; then
    echo "[launch_pyt] REFUSED $NAME: marker $mk exists (no-delete rule; use a new NAME)" >&2
    exit 65
  fi
done

# ---- thread tokens -------------------------------------------------------------------------------
GOT=0
TOKS=""
for i in 0 1 2 3 4 5; do  # PYT-AXIS: 6 tokens
  [ "$GOT" -ge "$TH" ] && break
  exec {fd}>>"$RUN/token.$i"
  if flock -n "$fd"; then GOT=$((GOT + 1)); TOKS="$TOKS $i"; else exec {fd}>&-; fi
done
if [ "$GOT" -lt "$TH" ]; then
  echo "[launch_pyt] REFUSED $NAME: needs $TH thread tokens, only $GOT of 6 free (plan F5 cap)" >&2
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
    echo "[launch_pyt] REFUSED $NAME: GPU ledger $TOTAL min + estimate $GPU_EST min > 60 (plan section 11)" >&2
    exit 76
  fi
  flock -u "$gfd"
  [ -f "$LEDGER" ] || printf 'name\tstart_utc\tend_utc\tseconds\tminutes\trunning_total_min\texit\n' > "$LEDGER"
fi

export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH NUMEXPR_NUM_THREADS=$TH \
       PYT_THREADS=$TH PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false  # PYT-AXIS
LOG="$LOGDIR/$NAME.log"
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
T0=$(date +%s.%N)
echo "$START_UTC threads=$TH tokens=[$TOKS ] gpu_est=${GPU_EST:-none} cmd: $*" > "$LOGDIR/$NAME.started"
{ echo "[launch_pyt] $START_UTC start $NAME threads=$TH tokens=[$TOKS ] gpu_est=${GPU_EST:-none}"; echo "[launch_pyt] cmd: $*"; } > "$LOG"
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
  echo "[launch_pyt] GPU step $NAME: $MIN min; axis running total $TOTAL min (budget 60)" >> "$LOG"
fi
echo "[launch_pyt] $END_UTC end $NAME rc=$RC ${SECS}s" >> "$LOG"
if [ "$RC" -eq 0 ]; then
  echo "$END_UTC rc=0 ${SECS}s" > "$LOGDIR/$NAME.done"
else
  echo "$END_UTC rc=$RC ${SECS}s" > "$LOGDIR/$NAME.failed"
fi
exit "$RC"
