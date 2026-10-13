#!/usr/bin/env bash
# S2.3 Stage A RESUME launcher -- AMENDMENT A1 (prereg/AMENDMENT_A1_resume.md, rules A1.1-A1.6).
# Lives in s23/resume/, never in src/ (src/ must keep exactly its frozen file set). It follows
# src/launch_stageA.sh from 'run_pool pre' on, with the frozen entry points src/worker.py and
# src/analyze_A.py and the frozen command lines; it never runs prestep.py or make_plan.py and never
# overwrites a chmod-444 file. Normally started by resume/gate_then_launch.sh (rule A1.5).
#
#   bash launch_stageA_resume.sh                 real run
#   bash launch_stageA_resume.sh --verify-only   re-verify every hash, then exit (0 OK, 2 failed)
#   bash launch_stageA_resume.sh --dry-run       real tree, read-only: verification, prep check, remaining
#                                                items, 8 s of GPU snapshots, self-tests; no worker, no GPU
#                                                work, no STATUS line, no statistic (outputs in s23/dev/)
#   A1_TEST_ROOT=<s23/dev/...> bash launch_stageA_resume.sh --test
#                                                excluded TEST-key tiny instance (CPU only), full sequence
# Exit codes: 0 Stage A done; 2 verification or prep refused (nothing run); 3 sampled gates did not allow
# the main runs; 4 a worker pool stopped (budget or error); 5 analysis error; 6 paused by the A1.3 speed
# guard; 7 paused, too few free timings for G4 (A1.2(d)); 8 inconsistent state; 9 lock held.
set -u
MODE=run
VERIFY_ONLY=0
for a in "$@"; do
  case "$a" in
    --dry-run) MODE=dry ;;
    --test) MODE=test ;;
    --verify-only) VERIFY_ONLY=1 ;;
    *) echo "usage: launch_stageA_resume.sh [--verify-only] [--dry-run|--test]"; exit 2 ;;
  esac
done
S23="$HOME/hrh_v2_stage2_20261008/s23"
PY="$HOME/hrh_v2_stage1_20261008/venv/bin/python"
RES="$S23/resume"
H="$RES/a1_resume.py"
T_ROOT="${A1_TEST_ROOT:-}"; T_GUARD="${A1_GUARD_S:-}"; T_SMI="${A1_FAKE_SMI:-}"; T_SNAP="${A1_SNAP_INTERVAL:-}"
unset S23_ROOT S23_KEY A1_GUARD_S A1_FAKE_SMI A1_TEST_ROOT A1_SNAP_INTERVAL
if [ "$MODE" = test ]; then
  case "$T_ROOT" in "$S23"/dev/A1_dryrun*/*) ;; *) echo "A1 test root must lie under $S23/dev/A1_dryrun*/"; exit 2 ;; esac
  export S23_ROOT="$T_ROOT" S23_KEY="hrhv2-s23-TEST" CUDA_VISIBLE_DEVICES=""
  [ -n "$T_GUARD" ] && export A1_GUARD_S="$T_GUARD"
  [ -n "$T_SMI" ] && export A1_FAKE_SMI="$T_SMI"
  [ -n "$T_SNAP" ] && export A1_SNAP_INTERVAL="$T_SNAP"
  ROOT="$T_ROOT"; STATUS_FILE="$ROOT/STATUS_test.md"
else
  ROOT="$S23"; STATUS_FILE="$HOME/hrh_v2_stage1_20261008/STATUS.md"
fi
LOGS="$ROOT/logs"; STATE="$ROOT/resume"; STAGEA="$ROOT/results/stageA"
now () { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ---------------------------------------------------------------- verification (rule A1.6), bash side
verify_bash () {
  local bad=0 n=0 h p f a line mh
  while read -r h p; do
    case "$h" in ""|\#*) continue ;; esac
    case "$p" in src/*) f="$S23/$p" ;; *) f="$HOME/$p" ;; esac
    a=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
    [ "$a" = "$h" ] || { echo "A1 verify(bash): frozen code hash MISMATCH $p"; bad=1; }
    n=$((n+1))
  done < "$S23/prereg/CODE_AT_PREREG.sha256"
  [ "$n" -eq 37 ] || { echo "A1 verify(bash): $n frozen code entries, expected 37"; bad=1; }
  for f in "$S23"/src/*.py "$S23"/src/*.sh "$S23"/src/tests/*.py; do
    p="src/${f#"$S23"/src/}"
    awk -v p="$p" '$2==p {f=1} END {exit !f}' "$S23/prereg/CODE_AT_PREREG.sha256" || { echo "A1 verify(bash): extra file in src/: $p"; bad=1; }
  done
  for kv in "prereg/CODE_AT_PREREG.sha256 f9560f0a9d3d2bc44c8407d1a9d3d440c0e8a0960765c67cc1709c7c039cf546" \
            "prereg/PREREG_S2_3_v1.0.md 7a93b7a9519b505ee887467baab7aa724d2a0ee05635a2afdc363da280e6b5d6" \
            "results/prestep/frozen_predictions.json 44d45c418da1f505bfec0c7500e1a5c2d3b5504b7d776760a0498a8b85bfb4cb" \
            "results/prestep/prestep_arrays.npz 7c560f00299c5532f687af0efd9b230f0f1f1782182e6528c76a281c73110429" \
            "plans/stageA_plan.json dfeb2c96b80ea5cddeec5e1331206ce40e88b260b716111c3bb661ae9a17980b"; do
    set -- $kv
    a=$(sha256sum "$S23/$1" 2>/dev/null | cut -d' ' -f1)
    [ "$a" = "$2" ] || { echo "A1 verify(bash): frozen data hash MISMATCH $1"; bad=1; }
  done
  grep -q "S2.3 PREREG v1.0 + CODE FROZEN" "$HOME/hrh_v2_stage1_20261008/STATUS.md" || { echo "A1 verify(bash): PREREG freeze line missing"; bad=1; }
  line=$(grep -m1 -F "S2.3 AMENDMENT A1 (resume) FROZEN" "$STATUS_FILE" 2>/dev/null || true)
  if [ -z "$line" ]; then
    if [ "$MODE" = dry ]; then echo "A1 verify(bash): no FROZEN line yet (dry run: candidate manifest only)"
    else echo "A1 verify(bash): no 'S2.3 AMENDMENT A1 (resume) FROZEN' line in $STATUS_FILE"; bad=1; fi
  else
    mh=$(printf '%s\n' "$line" | sed -n 's/.*A1_MANIFEST\.sha256 sha256 \([0-9a-f]\{64\}\).*/\1/p')
    a=$(sha256sum "$STATE/A1_MANIFEST.sha256" 2>/dev/null | cut -d' ' -f1)
    { [ -n "$mh" ] && [ "$a" = "$mh" ]; } || { echo "A1 verify(bash): A1_MANIFEST.sha256 differs from the FROZEN line"; bad=1; }
  fi
  if [ -f "$STATE/A1_MANIFEST.sha256" ]; then
    while read -r h p; do
      case "$h" in ""|\#*) continue ;; esac
      case "$p" in S23/*) f="$S23/${p#S23/}" ;; ROOT/*) f="$ROOT/${p#ROOT/}" ;; *) echo "A1 verify(bash): bad entry $p"; bad=1; continue ;; esac
      a=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
      [ "$a" = "$h" ] || { echo "A1 verify(bash): A1 file hash MISMATCH $p"; bad=1; }
    done < "$STATE/A1_MANIFEST.sha256"
    grep -q "  S23/resume/launch_stageA_resume.sh\$" "$STATE/A1_MANIFEST.sha256" || { echo "A1 verify(bash): launcher not in the A1 manifest"; bad=1; }
  else
    echo "A1 verify(bash): no A1 manifest at $STATE/A1_MANIFEST.sha256"; bad=1
  fi
  return $bad
}

verify_all () {
  verify_bash || return 2
  (cd "$RES" && "$PY" "$H" verify --mode "$MODE") || return 2
  return 0
}

if ! verify_all; then
  echo "$(now) A1 verification FAILED: nothing run"
  exit 2
fi
[ "$VERIFY_ONLY" = 1 ] && { echo "$(now) A1 verification OK ($MODE)"; exit 0; }

# ---------------------------------------------------------------- dry run (read-only on the real tree)
if [ "$MODE" = dry ]; then
  DRY="$S23/dev/A1_dryrun_launcher/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$DRY/att"
  now > "$DRY/att/start_utc"
  echo "== A1 dry run (no worker, no GPU work, no STATUS line, no statistic); outputs in $DRY"
  echo "== prep check (rule A1.1(b)-(c)): finished items vs manifest, partial files (listed, not moved)"
  (cd "$RES" && "$PY" "$H" prep --attempt "$DRY/att" --dry-run) || { echo "dry run: prep would REFUSE"; exit 2; }
  echo "== remaining plan items (frozen worker split)"
  (cd "$RES" && "$PY" "$H" remaining)
  echo "== sidecar for 8 s (real nvidia-smi; guard off; snapshots to $DRY)"
  (cd "$RES" && "$PY" "$H" sidecar --attempt "$DRY/att" --snap-file "$DRY/snapshots.tsv" --max-seconds 8 --no-guard)
  cut -f3- "$DRY/snapshots.tsv"
  echo "== free classification of the existing A-pilot ledger lines (rule A1.2(a),(c))"
  (cd "$RES" && "$PY" "$H" classify --snap-file "$DRY/snapshots.tsv")
  echo "== self-tests (classifier, smi parsing, guard on own dummy processes, G4 mask vs frozen analyze_A source, partial scan)"
  (cd "$RES" && "$PY" "$H" selftest --tmp "$DRY/selftest") || { echo "dry run: self-tests FAILED"; exit 2; }
  echo "== run mode would execute (cwd $S23/src):"
  echo "   prep --attempt <att> | sidecar (supervised) | run_pool pre: $PY worker.py --phase pre --pool gpu --wid 0|1 --nw 2, --pool cpu --wid 0 --nw 1"
  echo "   guard --final | gates --attempt <att> (A1_g4_inputs.json, then frozen analyze_A --phase gates with the A1 mask)"
  echo "   run_pool main | $PY analyze_A.py --phase nmatch | $PY analyze_A.py --phase main"
  echo "$(now) A1 dry run OK"
  exit 0
fi

# ---------------------------------------------------------------- real or test run
mkdir -p "$STATE/attempts" "$LOGS"
exec 9>>"$STATE/launch.lock"
flock -n 9 || { echo "$(now) another resume launcher holds $STATE/launch.lock"; exit 9; }
ATT_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ATT="$STATE/attempts/$ATT_ID"
mkdir -p "$ATT"
now > "$ATT/start_utc"
st () { (cd "$RES" && "$PY" "$H" status resume "$1") >/dev/null; }
finish () {  # $1 rc, $2 text
  echo "$1" > "$ATT/rc"
  touch "$ATT/sidecar.stop"
  [ -n "${SUP:-}" ] && wait "$SUP" 2>/dev/null
  st "A1 resume attempt $ATT_ID END rc=$1: $2; GPU ledger $(cd "$RES" && "$PY" -c 'import a1_resume as R; print(f"{R.C.gpu_hours_used():.4f}")') h"
  echo "$(now) $2"
  exit "$1"
}
st "A1 resume attempt $ATT_ID START ($MODE) via resume/launch_stageA_resume.sh sha256 $(sha256sum "$0" | cut -c1-12): verification OK (37 frozen code hashes, frozen data hashes, A1 manifest $(sha256sum "$STATE/A1_MANIFEST.sha256" | cut -c1-12) vs the FROZEN line)"

(cd "$RES" && "$PY" "$H" prep --attempt "$ATT") || finish 2 "prep refused (rule A1.1(b)); nothing run"

# sidecar (rule A1.2(a) snapshots, rule A1.3 guard), restarted if it dies, stopped by $ATT/sidecar.stop
( cd "$RES"; while [ ! -e "$ATT/sidecar.stop" ]; do "$PY" "$H" sidecar --attempt "$ATT" >> "$ATT/sidecar.log" 2>&1; sleep 1; done ) &
SUP=$!
sleep 3

cd "$S23/src" || finish 8 "cannot cd to src"
run_pool () {  # $1 = phase; identical worker command lines to src/launch_stageA.sh; pids recorded for A1.2(a)/A1.3
  "$PY" worker.py --phase "$1" --pool gpu --wid 0 --nw 2 >> "$LOGS/w_gpu0_$1.out" 2>&1 & p1=$!
  echo "gpu0 $p1 $(date +%s.%N)" >> "$ATT/pids_$1"
  "$PY" worker.py --phase "$1" --pool gpu --wid 1 --nw 2 >> "$LOGS/w_gpu1_$1.out" 2>&1 & p2=$!
  echo "gpu1 $p2 $(date +%s.%N)" >> "$ATT/pids_$1"
  "$PY" worker.py --phase "$1" --pool cpu --wid 0 --nw 1 >> "$LOGS/w_cpu0_$1.out" 2>&1 & p3=$!
  echo "cpu0 $p3 $(date +%s.%N)" >> "$ATT/pids_$1"
  rc=0
  wait $p1 || rc=$?
  wait $p2 || rc=$?
  wait $p3 || rc=$?
  return $rc
}

# ---- pre phase (rule A1.1(a): starts here; the frozen worker skips every .done item)
run_pool pre; rc_pre=$?
(cd "$RES" && "$PY" "$H" guard --attempt "$ATT" --final) >> "$ATT/guard_final.log" 2>&1
[ -e "$ATT/PAUSED" ] && finish 6 "PAUSED by the A1.3 speed guard (GPU contention); no analysis run, no statistic opened; resume only on a new author go"
[ "$rc_pre" -eq 0 ] || finish 4 "pre phase stopped (budget or error), worker rc=$rc_pre"

# ---- gates (rule A1.2; skipped when its chmod-444 output exists: rule A1.1(d))
if [ -e "$STAGEA/eps_arm0.json" ]; then
  [ -e "$STAGEA/g4_decision.json" ] || finish 8 "eps_arm0.json exists without g4_decision.json"
  allowed=$("$PY" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("main_allowed", False)))' "$STAGEA/g4_decision.json")
  [ "$allowed" = 1 ] || finish 3 "sampled gates did not allow main runs (stop) [existing g4_decision.json]"
  st "A1 resume attempt $ATT_ID: gates phase already done (eps_arm0.json exists), not re-run; existing g4_decision.json allows the main runs"
else
  (cd "$RES" && "$PY" "$H" gates --attempt "$ATT") >> "$LOGS/analyze_gates.out" 2>&1; rc_g=$?
  case "$rc_g" in
    0) ;;
    7) finish 7 "PAUSED before the gates phase: too few free timings for G4 (rule A1.2(d)); no statistic opened" ;;
    8) finish 8 "gates refused (inconsistent state)" ;;
    *) finish 3 "sampled gates did not allow main runs (stop), rc=$rc_g" ;;
  esac
fi

# ---- main phase
run_pool main || finish 4 "main phase stopped (budget or error)"

# ---- analysis (frozen order; each phase skipped when its output exists)
if [ -e "$STAGEA/N_match.json" ]; then
  st "A1 resume attempt $ATT_ID: N_match.json exists, nmatch phase not re-run"
else
  "$PY" analyze_A.py --phase nmatch >> "$LOGS/analyze_nmatch.out" 2>&1 || finish 5 "analysis error (nmatch)"
fi
if [ -e "$STAGEA/stageA_results.json" ]; then
  st "A1 resume attempt $ATT_ID: stageA_results.json exists, main analysis not re-run"
else
  "$PY" analyze_A.py --phase main >> "$LOGS/analyze_main.out" 2>&1 || finish 5 "analysis error (main)"
fi
finish 0 "Stage A done"
