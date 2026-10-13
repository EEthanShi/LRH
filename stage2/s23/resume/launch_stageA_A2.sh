#!/usr/bin/env bash
# S2.3 Stage A COMPLETION launcher -- AMENDMENT A2 (prereg/AMENDMENT_A2_complete.md, rules A2.1-A2.7):
# the author removed the GPU budget limits of Stage A ("你现在就跑", 2026-10-10); every scientific rule stays
# frozen. Lives in s23/resume/, never in src/. It runs the frozen 'run_pool main' (GPU pool through
# resume/a2_worker.py = frozen worker.main() with the two budget constants set to +inf in that process only;
# CPU pool = frozen src/worker.py), then the frozen 'analyze_A.py --phase nmatch' and '--phase main' with the
# frozen command lines of src/launch_stageA.sh. It never runs prestep.py, make_plan.py or the gates phase and
# never overwrites a chmod-444 file.
#
#   bash launch_stageA_A2.sh                 real run (started with setsid nohup; markers resume/A2.{started,done,failed})
#   bash launch_stageA_A2.sh --verify-only   re-verify every hash, then exit (0 OK, 2 failed)
#   bash launch_stageA_A2.sh --dry-run       real tree, read-only: verification, prep check, remaining items, budget
#                                            predicate, frozen worker loop with recorders (CUDA hidden), no GPU work,
#                                            no STATUS line, no statistic (outputs in s23/dev/A2_dryrun/)
#   A2_TEST_ROOT=<s23/dev/A2_tests_*/X> bash launch_stageA_A2.sh --test
#                                            excluded TEST-key tiny root (CPU only), full sequence
# Exit codes: 0 Stage A done; 2 verification or prep refused (nothing run); 3 gates do not allow the main runs;
# 4 main pool stopped or main phase incomplete (no analysis); 5 analysis error; 8 inconsistent state;
# 9 stale A2 markers or lock held.
set -u
MODE=run
VERIFY_ONLY=0
for a in "$@"; do
  case "$a" in
    --dry-run) MODE=dry ;;
    --test) MODE=test ;;
    --verify-only) VERIFY_ONLY=1 ;;
    *) echo "usage: launch_stageA_A2.sh [--verify-only] [--dry-run|--test]"; exit 2 ;;
  esac
done
S23="$HOME/hrh_v2_stage2_20261008/s23"
PY="$HOME/hrh_v2_stage1_20261008/venv/bin/python"
RES="$S23/resume"
H="$RES/a2_complete.py"
WK="$RES/a2_worker.py"
A2_MARK="S2.3 AMENDMENT A2 (budget removed by author; complete and analyze) FROZEN"
T_ROOT="${A2_TEST_ROOT:-}"
unset S23_ROOT S23_KEY A2_TEST_ROOT
if [ "$MODE" = test ]; then
  case "$T_ROOT" in "$S23"/dev/A2_tests_*/*) ;; *) echo "A2 test root must lie under $S23/dev/A2_tests_*/"; exit 2 ;; esac
  export S23_ROOT="$T_ROOT" S23_KEY="hrhv2-s23-TEST" CUDA_VISIBLE_DEVICES=""
  ROOT="$T_ROOT"; STATUS_FILE="$ROOT/STATUS_test.md"
else
  ROOT="$S23"; STATUS_FILE="$HOME/hrh_v2_stage1_20261008/STATUS.md"
fi
LOGS="$ROOT/logs"; STATE="$ROOT/resume"; STAGEA="$ROOT/results/stageA"
now () { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ---------------------------------------------------------------- verification (rule A2.7), bash side
verify_bash () {
  local bad=0 n=0 h p f a line mh nl
  while read -r h p; do
    case "$h" in ""|\#*) continue ;; esac
    case "$p" in src/*) f="$S23/$p" ;; *) f="$HOME/$p" ;; esac
    a=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
    [ "$a" = "$h" ] || { echo "A2 verify(bash): frozen code hash MISMATCH $p"; bad=1; }
    n=$((n+1))
  done < "$S23/prereg/CODE_AT_PREREG.sha256"
  [ "$n" -eq 37 ] || { echo "A2 verify(bash): $n frozen code entries, expected 37"; bad=1; }
  for f in "$S23"/src/*.py "$S23"/src/*.sh "$S23"/src/tests/*.py; do
    p="src/${f#"$S23"/src/}"
    awk -v p="$p" '$2==p {f=1} END {exit !f}' "$S23/prereg/CODE_AT_PREREG.sha256" || { echo "A2 verify(bash): extra file in src/: $p"; bad=1; }
  done
  for kv in "prereg/CODE_AT_PREREG.sha256 f9560f0a9d3d2bc44c8407d1a9d3d440c0e8a0960765c67cc1709c7c039cf546" \
            "prereg/PREREG_S2_3_v1.0.md 7a93b7a9519b505ee887467baab7aa724d2a0ee05635a2afdc363da280e6b5d6" \
            "prereg/AMENDMENT_A1_resume.md 8a858f5394c4be17a5162374992585f405a9cfc75fff8c38074860da9761948d" \
            "results/prestep/frozen_predictions.json 44d45c418da1f505bfec0c7500e1a5c2d3b5504b7d776760a0498a8b85bfb4cb" \
            "results/prestep/prestep_arrays.npz 7c560f00299c5532f687af0efd9b230f0f1f1782182e6528c76a281c73110429" \
            "plans/stageA_plan.json dfeb2c96b80ea5cddeec5e1331206ce40e88b260b716111c3bb661ae9a17980b"; do
    set -- $kv
    a=$(sha256sum "$S23/$1" 2>/dev/null | cut -d' ' -f1)
    [ "$a" = "$2" ] || { echo "A2 verify(bash): frozen data hash MISMATCH $1"; bad=1; }
  done
  grep -q "S2.3 PREREG v1.0 + CODE FROZEN" "$HOME/hrh_v2_stage1_20261008/STATUS.md" || { echo "A2 verify(bash): PREREG freeze line missing"; bad=1; }
  # A1 manifest of the real tree vs the real STATUS line (always the real tree: the A1 files are real files)
  line=$(grep -m1 -F "S2.3 AMENDMENT A1 (resume) FROZEN" "$HOME/hrh_v2_stage1_20261008/STATUS.md" 2>/dev/null || true)
  mh=$(printf '%s\n' "$line" | sed -n 's/.*A1_MANIFEST\.sha256 sha256 \([0-9a-f]\{64\}\).*/\1/p')
  a=$(sha256sum "$S23/resume/A1_MANIFEST.sha256" 2>/dev/null | cut -d' ' -f1)
  { [ -n "$mh" ] && [ "$a" = "$mh" ]; } || { echo "A2 verify(bash): real A1_MANIFEST.sha256 differs from the A1 FROZEN line"; bad=1; }
  while read -r h p; do
    case "$h" in ""|\#*) continue ;; esac
    case "$p" in S23/*|ROOT/*) f="$S23/${p#*/}" ;; *) echo "A2 verify(bash): bad A1 entry $p"; bad=1; continue ;; esac
    a=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
    [ "$a" = "$h" ] || { echo "A2 verify(bash): A1 file hash MISMATCH $p"; bad=1; }
  done < "$S23/resume/A1_MANIFEST.sha256"
  # A2 manifest vs the A2 FROZEN line (real STATUS; TEST root: its STATUS_test.md; dry run before the freeze: candidate)
  nl=$(grep -c -F "$A2_MARK" "$STATUS_FILE" 2>/dev/null || true)
  if [ "${nl:-0}" = 0 ]; then
    if [ "$MODE" = dry ]; then echo "A2 verify(bash): no A2 FROZEN line yet (dry run: candidate files only)"
    else echo "A2 verify(bash): no '$A2_MARK' line in $STATUS_FILE"; bad=1; fi
  elif [ "$nl" != 1 ]; then
    echo "A2 verify(bash): $nl A2 FROZEN lines in $STATUS_FILE (expected 1)"; bad=1
  else
    line=$(grep -m1 -F "$A2_MARK" "$STATUS_FILE")
    mh=$(printf '%s\n' "$line" | sed -n 's/.*A2_MANIFEST\.sha256 sha256 \([0-9a-f]\{64\}\).*/\1/p')
    a=$(sha256sum "$STATE/A2_MANIFEST.sha256" 2>/dev/null | cut -d' ' -f1)
    { [ -n "$mh" ] && [ "$a" = "$mh" ]; } || { echo "A2 verify(bash): A2_MANIFEST.sha256 differs from the A2 FROZEN line"; bad=1; }
    while read -r h p; do
      case "$h" in ""|\#*) continue ;; esac
      case "$p" in S23/*) f="$S23/${p#S23/}" ;; ROOT/*) f="$ROOT/${p#ROOT/}" ;; *) echo "A2 verify(bash): bad A2 entry $p"; bad=1; continue ;; esac
      a=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
      [ "$a" = "$h" ] || { echo "A2 verify(bash): A2 file hash MISMATCH $p"; bad=1; }
      if [ "$MODE" = run ] && [ -w "$f" ]; then echo "A2 verify(bash): A2 file writable (not frozen): $p"; bad=1; fi
    done < "$STATE/A2_MANIFEST.sha256"
    grep -q "  S23/resume/launch_stageA_A2.sh\$" "$STATE/A2_MANIFEST.sha256" || { echo "A2 verify(bash): launcher not in the A2 manifest"; bad=1; }
  fi
  return $bad
}

verify_all () {
  verify_bash || return 2
  (cd "$RES" && "$PY" "$H" verify --mode "$MODE") || return 2
  return 0
}

# ---------------------------------------------------------------- dry run (read-only on the real tree)
if [ "$MODE" = dry ]; then
  if ! verify_all; then echo "$(now) A2 verification FAILED (dry run)"; exit 2; fi
  [ "$VERIFY_ONLY" = 1 ] && { echo "$(now) A2 verification OK (dry)"; exit 0; }
  DRY="$S23/dev/A2_dryrun/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$DRY/att"
  echo "== A2 dry run (no worker process on the GPU, no GPU work, no STATUS line, no statistic); outputs in $DRY"
  echo "== prep check (rule A2.3(b)-(c)): finished items vs the A1 and A2 manifests, partial files (listed, not moved)"
  (cd "$RES" && "$PY" "$H" prep --attempt "$DRY/att" --dry-run) || echo "dry run: prep would REFUSE (expected only before the A2 finished-items manifest exists)"
  echo "== remaining plan items (frozen worker split)"
  (cd "$RES" && "$PY" "$H" remaining --json "$DRY/remaining.json") || { echo "dry run: remaining FAILED"; exit 2; }
  echo "== gates already decided (not re-run)"
  (cd "$RES" && "$PY" "$H" g4); rc_g=$?
  [ "$rc_g" = 0 ] || { echo "dry run: g4 check rc=$rc_g"; exit 2; }
  echo "== frozen budget predicate (rule A2.1(d))"
  (cd "$RES" && "$PY" "$H" budget); rc_b=$?
  echo "== frozen worker main loop with recorders (CUDA hidden), FROZEN constants: expect refusal (rc 4)"
  (cd "$S23/src" && CUDA_VISIBLE_DEVICES= "$PY" "$WK" --dry-run --dry-dir "$DRY/frozen" --frozen-budget --phase main --pool gpu --wid 0 --nw 2); rc_f=$?
  echo "   -> rc $rc_f"
  echo "== frozen worker main loop with recorders (CUDA hidden), rule A2.1(c): each worker lists its remaining items"
  for w in "gpu 0 2" "gpu 1 2" "cpu 0 1"; do
    set -- $w
    (cd "$S23/src" && CUDA_VISIBLE_DEVICES= "$PY" "$WK" --dry-run --dry-dir "$DRY/a2" --phase main --pool "$1" --wid "$2" --nw "$3") \
      || { echo "dry run: A2 worker dry run $1$2 FAILED"; exit 2; }
  done
  "$PY" - "$DRY" <<'EOF' || { echo "dry run: item lists do not match the remaining plan items"; exit 2; }
import json, sys
d = sys.argv[1]
rem = json.load(open(f"{d}/remaining.json"))["todo_ids"]
ok = True
for w in ("gpu0", "gpu1", "cpu0"):
    got = json.load(open(f"{d}/a2/items_{w}.json"))["items_would_run"]
    want = rem.get(f"main:{w}", [])
    print(f"   {w}: dry-run loop would run {len(got)} items; remaining plan items of {w}: {len(want)}; equal: {got == want}")
    ok &= got == want
fz = json.load(open(f"{d}/frozen/items_gpu0_frozen.json"))
print(f"   frozen constants: rc {fz['rc']}, items run {fz['n_items_would_run']} (expected rc 4, 0 items)")
ok &= fz["rc"] == 4 and fz["n_items_would_run"] == 0
sys.exit(0 if ok else 1)
EOF
  echo "== analysis outputs (would run nmatch then main, frozen command lines, cwd src)"
  for f in N_match.json stageA_results.json REPORT_stageA.md; do
    if [ -e "$STAGEA/$f" ]; then echo "   $f exists (phase would be skipped)"; else echo "   $f absent (phase would run)"; fi
  done
  echo "== one nvidia-smi compute-apps query (read-only)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 | sed 's/^/   /'
  echo "== run mode would execute: prep | snapshots every 30 s | (cwd src) $PY $WK --phase main --pool gpu --wid 0|1 --nw 2,"
  echo "   $PY worker.py --phase main --pool cpu --wid 0 --nw 1 | complete | $PY analyze_A.py --phase nmatch | $PY analyze_A.py --phase main"
  [ "$rc_b" = 0 ] && echo "   (frozen predicate: the frozen GPU pool would refuse, so the GPU pool runs through a2_worker.py)"
  echo "$(now) A2 dry run OK"
  exit 0
fi

# ---------------------------------------------------------------- real or test run
if ! verify_all; then
  echo "$(now) A2 verification FAILED: nothing run"
  [ "$VERIFY_ONLY" = 1 ] || { [ -e "$STATE/A2.started" ] || [ -e "$STATE/A2.failed" ] || echo "$(now) rc=2 verification failed; nothing run" > "$STATE/A2.failed"; }
  exit 2
fi
[ "$VERIFY_ONLY" = 1 ] && { echo "$(now) A2 verification OK ($MODE)"; exit 0; }

mkdir -p "$STATE/A2_attempts" "$LOGS"
for m in A2.started A2.done A2.failed; do
  [ -e "$STATE/$m" ] && { echo "$(now) stale marker $STATE/$m: move it aside (never delete) before a new start"; exit 9; }
done
exec 9>>"$STATE/A2.lock"
flock -n 9 || { echo "$(now) another A2 launcher holds $STATE/A2.lock"; exit 9; }
ATT_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ATT="$STATE/A2_attempts/$ATT_ID"
mkdir -p "$ATT"
echo "$(now) attempt $ATT_ID pid $$ mode $MODE" > "$STATE/A2.started"
st () { (cd "$RES" && "$PY" "$H" status run-A2 "$1") >/dev/null; }
ledger () { (cd "$RES" && "$PY" "$H" ledger); }
SNAP_STOP="$ATT/snapshots.stop"
finish () {  # $1 rc, $2 text
  touch "$SNAP_STOP"
  [ -n "${SNP:-}" ] && wait "$SNP" 2>/dev/null
  echo "$1" > "$ATT/rc"
  st "A2 attempt $ATT_ID END rc=$1: $2; GPU ledger $(ledger) h"
  if [ "$1" = 0 ]; then echo "$(now) rc=0 $2" > "$STATE/A2.done"; else echo "$(now) rc=$1 $2" > "$STATE/A2.failed"; fi
  echo "$(now) $2"
  exit "$1"
}
st "A2 attempt $ATT_ID START ($MODE) via resume/launch_stageA_A2.sh sha256 $(sha256sum "$0" | cut -c1-12): verification OK (37 frozen code hashes, frozen data hashes, A1 manifest vs its FROZEN line, A2 manifest $(sha256sum "$STATE/A2_MANIFEST.sha256" | cut -c1-12) vs the A2 FROZEN line); S2.3 Stage A uses the GPU until resume/A2.done or A2.failed; GPU ledger $(ledger) h"

(cd "$RES" && "$PY" "$H" prep --attempt "$ATT") > "$ATT/prep.out" 2>&1 || finish 2 "prep refused (rule A2.3(b)); nothing run"
(cd "$RES" && "$PY" "$H" g4) > "$ATT/g4.out" 2>&1; rc_g=$?
case "$rc_g" in
  0) ;;
  3) finish 3 "existing g4_decision.json does not allow the main runs (stop)" ;;
  *) finish 8 "gates state inconsistent ($(tail -1 "$ATT/g4.out"))" ;;
esac
(cd "$RES" && "$PY" "$H" remaining --json "$ATT/remaining.json") > "$ATT/remaining.out" 2>&1
(cd "$RES" && "$PY" "$H" budget) > "$ATT/budget.out" 2>&1; rc_b=$?
if [ "$rc_b" = 0 ]; then
  GPUWORKER="$WK"; how="frozen predicate gpu_budget_ok(soft=True)=False, so the frozen GPU pool would refuse every remaining item; GPU pool runs through resume/a2_worker.py (frozen worker.main, budget constants +inf in-process, rule A2.1(c))"
else
  GPUWORKER="worker.py"; how="frozen predicate allows new items; GPU pool runs the frozen src/worker.py"
fi
st "A2 attempt $ATT_ID: prep OK; gates not re-run ($(tail -1 "$ATT/g4.out" | cut -c1-200)); $(grep -m1 '^main todo' "$ATT/remaining.out"); $how"

# disclosure only (rule A2.5): one compute-apps query every 30 s
mkdir -p "$LOGS/A2_gpu_snapshots"
( while [ ! -e "$SNAP_STOP" ]; do
    printf '%s\t%s\n' "$(now)" "$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 | tr '\n' ';')" >> "$LOGS/A2_gpu_snapshots/$ATT_ID.tsv"
    for _ in $(seq 30); do [ -e "$SNAP_STOP" ] && break; sleep 1; done
  done ) &
SNP=$!

cd "$S23/src" || finish 8 "cannot cd to src"
run_pool_main () {  # the frozen run_pool main: 2 GPU workers + 1 CPU worker, same arguments and log files
  "$PY" "$GPUWORKER" --phase main --pool gpu --wid 0 --nw 2 >> "$LOGS/w_gpu0_main.out" 2>&1 & p1=$!
  echo "gpu0 $p1 $(date +%s.%N) $GPUWORKER" >> "$ATT/pids_main"
  "$PY" "$GPUWORKER" --phase main --pool gpu --wid 1 --nw 2 >> "$LOGS/w_gpu1_main.out" 2>&1 & p2=$!
  echo "gpu1 $p2 $(date +%s.%N) $GPUWORKER" >> "$ATT/pids_main"
  "$PY" worker.py --phase main --pool cpu --wid 0 --nw 1 >> "$LOGS/w_cpu0_main.out" 2>&1 & p3=$!
  echo "cpu0 $p3 $(date +%s.%N) worker.py" >> "$ATT/pids_main"
  rc=0; r1=0; r2=0; r3=0
  wait $p1 || r1=$?
  wait $p2 || r2=$?
  wait $p3 || r3=$?
  echo "gpu0 rc=$r1 gpu1 rc=$r2 cpu0 rc=$r3" >> "$ATT/pids_main"
  [ "$r1" = 0 ] && [ "$r2" = 0 ] && [ "$r3" = 0 ] || rc=1
  return $rc
}
run_pool_main || finish 4 "main pool stopped (error), $(tail -1 "$ATT/pids_main"); no analysis run"
(cd "$RES" && "$PY" "$H" complete) > "$ATT/complete.out" 2>&1 || finish 4 "main phase incomplete after the pool exited: $(tail -1 "$ATT/complete.out"); no analysis run"
st "A2 attempt $ATT_ID: main phase complete ($(tail -1 "$ATT/complete.out" | cut -c14-120)); GPU ledger $(ledger) h; starting the frozen analysis (nmatch, then main)"

# ---- analysis: frozen command lines of src/launch_stageA.sh; a phase is skipped when its chmod-444 output exists
if [ -e "$STAGEA/N_match.json" ]; then
  st "A2 attempt $ATT_ID: N_match.json exists, nmatch phase not re-run"
else
  "$PY" analyze_A.py --phase nmatch >> "$LOGS/analyze_nmatch.out" 2>&1 || finish 5 "analysis error (nmatch)"
fi
if [ -e "$STAGEA/stageA_results.json" ]; then
  st "A2 attempt $ATT_ID: stageA_results.json exists, main analysis not re-run"
else
  "$PY" analyze_A.py --phase main >> "$LOGS/analyze_main.out" 2>&1 || finish 5 "analysis error (main)"
fi
finish 0 "Stage A done"
