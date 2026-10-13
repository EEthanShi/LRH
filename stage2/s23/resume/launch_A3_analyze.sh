#!/usr/bin/env bash
# S2.3 Stage A ANALYSIS launcher -- AMENDMENT A3 (prereg/AMENDMENT_A3_analysis_memo.md, rules A3.1-A3.5).
# Lives in s23/resume/, never in src/. CPU only (one process, one BLAS thread); never touches the GPU.
# After the A2 attempt 20261010T083303Z (all 3,456 main items done, then rc=5 because the frozen Store.K memo raises
# KeyError on its 1501st distinct load), it runs the frozen 'analyze_A.py --phase nmatch' and '--phase main' through
# resume/a3_analyze.py (frozen analyze_A imported unchanged; Store.K replaced by the corrected memo in that process).
#
#   bash launch_A3_analyze.sh                 real run (setsid nohup; markers resume/A3.{started,done,failed})
#   bash launch_A3_analyze.sh --verify-only   re-verify every hash (A2 verification + A3 manifest), then exit
#   A3_TEST_ROOT=<s23/dev/A3_tests_*/X> bash launch_A3_analyze.sh --test     excluded TEST root (R29)
# Exit codes: 0 done; 2 verification or precondition refused (nothing run); 5 analysis error;
#             6 regenerated within-A cache differs from the crashed attempt's files (main not run); 8 cannot cd to src;
#             9 stale markers/lock.
set -u
MODE=run
VERIFY_ONLY=0
for a in "$@"; do
  case "$a" in
    --test) MODE=test ;;
    --verify-only) VERIFY_ONLY=1 ;;
    *) echo "usage: launch_A3_analyze.sh [--verify-only] [--test]"; exit 2 ;;
  esac
done
S23="$HOME/hrh_v2_stage2_20261008/s23"
PY="$HOME/hrh_v2_stage1_20261008/venv/bin/python"
RES="$S23/resume"
WRAP="$RES/a3_analyze.py"
A3_MARK="S2.3 AMENDMENT A3 (analysis memo fix) FROZEN"
ANALYZE_SHA="e43e4a8e5be1ed8f6e3092d43646bf2df3ceb1c1f18eaabb069296a5fe3e522a"
T_ROOT="${A3_TEST_ROOT:-}"
T_CAP="${A3_TEST_CAP:-}"
unset S23_ROOT S23_KEY A3_TEST_ROOT A3_TEST_CAP
if [ "$MODE" = test ]; then
  case "$T_ROOT" in "$S23"/dev/A3_tests_*/*) ;; *) echo "A3 test root must lie under $S23/dev/A3_tests_*/"; exit 2 ;; esac
  export S23_ROOT="$T_ROOT" S23_KEY="hrhv2-s23-TEST" CUDA_VISIBLE_DEVICES=""
  [ -n "$T_CAP" ] && export A3_TEST_CAP="$T_CAP"
  ROOT="$T_ROOT"; STATUS_FILE="$ROOT/STATUS_test.md"
else
  export CUDA_VISIBLE_DEVICES=""
  ROOT="$S23"; STATUS_FILE="$HOME/hrh_v2_stage1_20261008/STATUS.md"
fi
LOGS="$ROOT/logs"; STATE="$ROOT/resume"; STAGEA="$ROOT/results/stageA"
CRASHED_LIST="$STATE/A3_crashed_cache.sha256"
ASIDE="$STAGEA/_crashed_cache_A2_20261010T083303Z"
now () { date -u +%Y-%m-%dT%H:%M:%SZ; }
st () { "$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); import s23common as C; C.append_status(sys.argv[2], sys.argv[3])' \
          "$S23/src" run-A3 "$1" >/dev/null; }

# ---------------------------------------------------------------- verification (rule A3.4)
verify_a3 () {
  local bad=0 nl line mh a h p f
  bash "$RES/launch_stageA_A2.sh" --verify-only > /dev/null 2>&1 || { echo "A3 verify: A2 verification (--verify-only) FAILED"; bad=1; }
  a=$(sha256sum "$S23/src/analyze_A.py" | cut -d' ' -f1)
  [ "$a" = "$ANALYZE_SHA" ] || { echo "A3 verify: src/analyze_A.py hash differs from the frozen one"; bad=1; }
  nl=$(grep -c -F "$A3_MARK" "$HOME/hrh_v2_stage1_20261008/STATUS.md" 2>/dev/null || true)
  if [ "${nl:-0}" = 0 ]; then
    if [ "$MODE" = test ]; then echo "A3 verify: no A3 FROZEN line yet (TEST root: candidate files only)"
    else echo "A3 verify: no '$A3_MARK' line in STATUS.md"; bad=1; fi
  elif [ "$nl" != 1 ]; then
    echo "A3 verify: $nl A3 FROZEN lines (expected 1)"; bad=1
  else
    line=$(grep -m1 -F "$A3_MARK" "$HOME/hrh_v2_stage1_20261008/STATUS.md")
    mh=$(printf '%s\n' "$line" | sed -n 's/.*A3_MANIFEST\.sha256 sha256 \([0-9a-f]\{64\}\).*/\1/p')
    a=$(sha256sum "$RES/A3_MANIFEST.sha256" 2>/dev/null | cut -d' ' -f1)
    { [ -n "$mh" ] && [ "$a" = "$mh" ]; } || { echo "A3 verify: A3_MANIFEST.sha256 differs from the A3 FROZEN line"; bad=1; }
    while read -r h p; do
      case "$h" in ""|\#*) continue ;; esac
      f="$S23/${p#S23/}"
      a=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
      [ "$a" = "$h" ] || { echo "A3 verify: A3 file hash MISMATCH $p"; bad=1; }
      [ -w "$f" ] && { echo "A3 verify: A3 file writable (not frozen): $p"; bad=1; }
    done < "$RES/A3_MANIFEST.sha256"
    grep -q "  S23/resume/launch_A3_analyze.sh\$" "$RES/A3_MANIFEST.sha256" || { echo "A3 verify: launcher not in the A3 manifest"; bad=1; }
  fi
  return $bad
}
if ! verify_a3; then echo "$(now) A3 verification FAILED: nothing run"; exit 2; fi
[ "$VERIFY_ONLY" = 1 ] && { echo "$(now) A3 verification OK ($MODE)"; exit 0; }

# ---------------------------------------------------------------- preconditions (rule A3.3)
pre_bad=0
if [ "$MODE" = run ]; then
  grep -q "rc=5 analysis error (nmatch)" "$RES/A2.failed" 2>/dev/null || { echo "A3 pre: resume/A2.failed does not record rc=5 analysis error (nmatch)"; pre_bad=1; }
  [ -e "$RES/A2.done" ] && { echo "A3 pre: resume/A2.done exists"; pre_bad=1; }
fi
for f in N_match.json stageA_results.json REPORT_stageA.md; do
  [ -e "$STAGEA/$f" ] && { echo "A3 pre: $STAGEA/$f already exists"; pre_bad=1; }
done
if pgrep -u "$(id -u)" -f "(analyze_A|a3_analyze)\.py --phase|worker\.py --phase (pre|main) --pool" > /dev/null; then
  echo "A3 pre: a Stage A worker or analysis process is alive"; pre_bad=1
fi
COMPLETE_MSG=$(cd "$RES" && "$PY" a2_complete.py complete 2>&1) || { echo "A3 pre: main phase not complete: $COMPLETE_MSG"; pre_bad=1; }
COMPLETE_MSG=$(printf '%s\n' "$COMPLETE_MSG" | tail -1 | sed 's/^A2 complete: //')
[ -s "$CRASHED_LIST" ] || { echo "A3 pre: $CRASHED_LIST missing"; pre_bad=1; }
while read -r h p; do
  case "$h" in ""|\#*) continue ;; esac
  a=$(sha256sum "$ROOT/$p" 2>/dev/null | cut -d' ' -f1)
  [ "$a" = "$h" ] || { echo "A3 pre: crashed-attempt cache file missing or changed: $p"; pre_bad=1; }
done < "$CRASHED_LIST"
[ -e "$ASIDE" ] && { echo "A3 pre: $ASIDE already exists"; pre_bad=1; }
[ "$pre_bad" = 0 ] || { echo "$(now) A3 preconditions FAILED: nothing run"; exit 2; }

# ---------------------------------------------------------------- markers
for m in A3.started A3.done A3.failed; do
  [ -e "$STATE/$m" ] && { echo "$(now) stale marker $STATE/$m: move it aside (never delete) before a new start"; exit 9; }
done
exec 9>>"$STATE/A3.lock"
flock -n 9 || { echo "$(now) another A3 launcher holds $STATE/A3.lock"; exit 9; }
ATT_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ATT="$STATE/A3_attempts/$ATT_ID"
mkdir -p "$ATT" "$LOGS"
echo "$(now) attempt $ATT_ID pid $$ mode $MODE" > "$STATE/A3.started"
finish () {  # $1 rc, $2 text
  echo "$1" > "$ATT/rc"
  st "A3 attempt $ATT_ID END rc=$1: $2"
  if [ "$1" = 0 ]; then echo "$(now) rc=0 $2" > "$STATE/A3.done"; else echo "$(now) rc=$1 $2" > "$STATE/A3.failed"; fi
  echo "$(now) rc=$1 $2"
  exit "$1"
}
st "A3 attempt $ATT_ID START ($MODE) via resume/launch_A3_analyze.sh sha256 $(sha256sum "$0" | cut -c1-12): verification OK (A2 --verify-only: 37 frozen code, 5 frozen data, A1 and A2 manifests; A3 manifest vs the A3 FROZEN line); preconditions OK ($COMPLETE_MSG; no analysis output yet; no Stage A process alive); CPU only, one BLAS thread, no GPU use"

# ---------------------------------------------------------------- crashed-attempt cache files: move aside (rule A3.3(c))
mkdir -p "$ASIDE"
n_moved=0
while read -r h p; do
  case "$h" in ""|\#*) continue ;; esac
  mv -n "$ROOT/$p" "$ASIDE/$(basename "$p")" || finish 2 "could not move $p aside"
  echo "$(now) moved $p -> ${ASIDE#"$ROOT"/}/$(basename "$p") sha256 $h" >> "$ATT/moves.log"
  n_moved=$((n_moved+1))
done < "$CRASHED_LIST"
st "A3 attempt $ATT_ID: $n_moved cache files written by the crashed A2 nmatch process moved aside (not opened) to ${ASIDE#"$ROOT"/}/; the frozen nmatch phase recomputes them"

# ---------------------------------------------------------------- analysis (rule A3.2): frozen order, cwd src, same logs
cd "$S23/src" || finish 8 "cannot cd to src"
"$PY" "$WRAP" --phase nmatch >> "$LOGS/analyze_nmatch.out" 2>&1 || finish 5 "analysis error (nmatch); see logs/analyze_nmatch.out"
[ -e "$STAGEA/N_match.json" ] || finish 5 "nmatch exited 0 but N_match.json is missing"
n_eq=0; n_all=0; diffs=""
while read -r h p; do
  case "$h" in ""|\#*) continue ;; esac
  n_all=$((n_all+1))
  a=$(sha256sum "$ROOT/$p" 2>/dev/null | cut -d' ' -f1)
  if [ "$a" = "$h" ]; then n_eq=$((n_eq+1)); else diffs="$diffs $(basename "$p")"; fi
done < "$CRASHED_LIST"
st "A3 attempt $ATT_ID: nmatch done, N_match.json sha256 $(sha256sum "$STAGEA/N_match.json" | cut -d' ' -f1); regenerated within-A cache files byte-equal to the crashed attempt's: $n_eq/$n_all (sha256 compared, contents not opened)${diffs:+; differ:$diffs}"
[ "$n_eq" = "$n_all" ] || finish 6 "regenerated within-A cache differs from the crashed attempt's files ($diffs); main phase not run"
"$PY" "$WRAP" --phase main >> "$LOGS/analyze_main.out" 2>&1 || finish 5 "analysis error (main); see logs/analyze_main.out"
for f in stageA_results.json REPORT_stageA.md; do
  [ -e "$STAGEA/$f" ] || finish 5 "main exited 0 but $f is missing"
  chmod 444 "$STAGEA/$f"
done
finish 0 "Stage A analysis done (A3): N_match.json $(sha256sum "$STAGEA/N_match.json" | cut -d' ' -f1), stageA_results.json $(sha256sum "$STAGEA/stageA_results.json" | cut -d' ' -f1), REPORT_stageA.md $(sha256sum "$STAGEA/REPORT_stageA.md" | cut -d' ' -f1) (all chmod 444)"
