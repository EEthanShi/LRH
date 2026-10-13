#!/usr/bin/env bash
# S2.3 AMENDMENT A1, rule A1.5: wait until S2.2 v1.1 has finished (pipeline.done or pipeline.failed) AND
# nvidia-smi has listed no compute process at every check (every 60 s) for at least 10 consecutive minutes,
# then run resume/launch_stageA_resume.sh. Never signals or changes another job.
#
#   setsid nohup bash gate_then_launch.sh > resume/gate.out 2>&1 < /dev/null &      (real)
#   bash gate_then_launch.sh --dry-run <dir under s23/dev/A1_dryrun*>               (test)
#       dry run: fake S2.2 marker <dir>/fake_s22/pipeline.{done,failed}, fake compute-apps list <dir>/fake_smi
#       (a line per process, or ERR), markers in <dir>, interval/window from A1_GATE_INTERVAL/A1_GATE_WINDOW
#       (default 1 s / 3 s), and the launcher is called with --dry-run; no STATUS line.
# Markers: gate.started, gate.heartbeat, gate.launched, gate.failed; a file CANCEL stops the gate before launch.
set -u
MODE=run
S23="$HOME/hrh_v2_stage2_20261008/s23"
PY="$HOME/hrh_v2_stage1_20261008/venv/bin/python"
RES="$S23/resume"
L="$RES/launch_stageA_resume.sh"
S22M="$HOME/hrh_v2_stage2_20261008/results/S2_2_v11/logs/pipeline"
INTERVAL=60
WINDOW=600
STATE="$RES"
LARGS=""
unset S23_ROOT S23_KEY A1_GUARD_S A1_FAKE_SMI A1_TEST_ROOT A1_SNAP_INTERVAL
if [ "${1:-}" = "--dry-run" ]; then
  MODE=dry
  STATE="${2:?dry run needs a directory}"
  case "$STATE" in "$S23"/dev/A1_dryrun*/*) ;; *) echo "dry-run dir must lie under $S23/dev/A1_dryrun*/"; exit 2 ;; esac
  mkdir -p "$STATE/fake_s22"
  S22M="$STATE/fake_s22/pipeline"
  INTERVAL="${A1_GATE_INTERVAL:-1}"
  WINDOW="${A1_GATE_WINDOW:-3}"
  LARGS="--dry-run"
elif [ -n "${1:-}" ]; then
  echo "usage: gate_then_launch.sh [--dry-run DIR]"; exit 2
fi
now () { date -u +%Y-%m-%dT%H:%M:%SZ; }
st () {
  if [ "$MODE" = run ]; then (cd "$RES" && "$PY" "$RES/a1_resume.py" status resume "$1") >/dev/null
  else echo "$(now) [dry, not in STATUS] $1" >> "$STATE/gate_status_dry.log"; fi
}
fail () { echo "$(now) $1" > "$STATE/gate.failed"; st "A1 gate FAILED: $1"; echo "$(now) gate failed: $1"; exit 1; }
smi () {  # prints 'rc n_compute_processes'
  if [ "$MODE" = dry ]; then (cd "$RES" && "$PY" "$RES/a1_resume.py" smi --fake "$STATE/fake_smi")
  else (cd "$RES" && "$PY" "$RES/a1_resume.py" smi); fi
}

for m in gate.started gate.launched gate.failed; do
  [ -e "$STATE/$m" ] && { echo "$(now) marker $STATE/$m exists: move old gate markers aside (never delete) first"; exit 9; }
done
exec 8>>"$STATE/gate.lock"
flock -n 8 || { echo "$(now) another gate holds $STATE/gate.lock"; exit 9; }
echo "$(now) pid $$ mode $MODE interval ${INTERVAL}s window ${WINDOW}s" > "$STATE/gate.started"

bash "$L" --verify-only $LARGS > "$STATE/gate_verify_start.log" 2>&1 || fail "hash verification failed before waiting (see $STATE/gate_verify_start.log)"
st "A1 gate STARTED (rule A1.5), pid $$: waiting for S2.2 v1.1 pipeline.done|pipeline.failed and no GPU compute process at every check for >= ${WINDOW} s (checks every ${INTERVAL} s); hashes verified; markers resume/gate.{started,heartbeat,launched,failed}"

first=""; nfree=0; nchk=0; s22=none
while :; do
  [ -e "$STATE/CANCEL" ] && fail "cancelled (CANCEL file present) before launch"
  t=$(date +%s); nchk=$((nchk+1))
  s22=none
  [ -e "$S22M.done" ] && s22=done
  [ -e "$S22M.failed" ] && s22=failed
  read -r rc napps < <(smi) || { rc=1; napps=-1; }
  if [ "$s22" != none ] && [ "${rc:-1}" = 0 ] && [ "${napps:-1}" = 0 ]; then
    [ -z "$first" ] && { first=$t; nfree=0; }
    nfree=$((nfree+1))
    msg="S2.2 $s22; GPU free since $(date -u -d @"$first" +%FT%TZ) ($((t-first)) s, $nfree checks)"
    if [ $((t-first)) -ge "$WINDOW" ]; then
      echo "$(now) check $nchk: $msg -> condition met" > "$STATE/gate.heartbeat.tmp"; mv -f "$STATE/gate.heartbeat.tmp" "$STATE/gate.heartbeat"
      break
    fi
  else
    first=""; nfree=0
    msg="waiting: S2.2 marker=$s22, compute-apps query rc=${rc:-?}, compute processes=${napps:-?}"
  fi
  echo "$(now) check $nchk: $msg" > "$STATE/gate.heartbeat.tmp"; mv -f "$STATE/gate.heartbeat.tmp" "$STATE/gate.heartbeat"
  sleep "$INTERVAL"
done

bash "$L" --verify-only $LARGS > "$STATE/gate_verify_launch.log" 2>&1 || fail "hash verification failed just before launch (see $STATE/gate_verify_launch.log)"
echo "$(now) S2.2 marker pipeline.$s22; no compute process at $nfree consecutive checks from $(date -u -d @"$first" +%FT%TZ) to $(now)" > "$STATE/gate.launched"
st "A1 gate condition met (rule A1.5): S2.2 v1.1 marker pipeline.$s22; no GPU compute process at $nfree consecutive checks from $(date -u -d @"$first" +%FT%TZ) to $(now); launching resume/launch_stageA_resume.sh"
bash "$L" $LARGS >> "$STATE/launch_resume.out" 2>&1
rc=$?
echo "$(now) rc=$rc" > "$STATE/launch_resume.rc"
st "A1 gate: resume launcher exited rc=$rc (0 Stage A done; 2 refused; 3 gates did not allow main runs; 4 pool stopped; 5 analysis error; 6 speed-guard pause; 7 too few free G4 timings; 8 inconsistent state; 9 lock)"
exit 0
