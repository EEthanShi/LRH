#!/usr/bin/env bash
# S2.3 Stage A MAIN-ANALYSIS launcher -- AMENDMENT A4 (prereg/AMENDMENT_A4_p5b_zero.md, rules A4.1-A4.4).
# Lives in s23/resume/, never in src/. CPU only (one process, one BLAS thread); never touches the GPU.
# After the A3 attempt 20261010T092047Z (nmatch done, N_match.json written; main stopped rc=5 by ZeroDivisionError at
# analyze_A.py line 705 because the frozen MC prediction pi_mc["40"]["dMST"]["rho"] is 0.0), it runs the frozen
# 'analyze_A.py --phase main' through resume/a4_analyze.py (frozen source text with line 705 defined for mc == 0, plus
# the A3.1 memo). The nmatch phase is not re-run.
#
#   bash launch_A4_analyze.sh                 real run (setsid nohup; markers resume/A4.{started,done,failed})
#   bash launch_A4_analyze.sh --verify-only   re-verify every hash (A2, A3 and A4), then exit
#   A4_TEST_ROOT=<s23/dev/A4_tests_*/X> bash launch_A4_analyze.sh --test     excluded TEST root (R29)
# Exit codes: 0 done; 2 verification or precondition refused (nothing run); 5 analysis error; 8 cannot cd to src;
#             9 stale markers/lock.
set -u
MODE=run
VERIFY_ONLY=0
for a in "$@"; do
  case "$a" in
    --test) MODE=test ;;
    --verify-only) VERIFY_ONLY=1 ;;
    *) echo "usage: launch_A4_analyze.sh [--verify-only] [--test]"; exit 2 ;;
  esac
done
S23="$HOME/hrh_v2_stage2_20261008/s23"
PY="$HOME/hrh_v2_stage1_20261008/venv/bin/python"
RES="$S23/resume"
WRAP="$RES/a4_analyze.py"
A4_MARK="S2.3 AMENDMENT A4 (P5b zero MC denominator) FROZEN"
NMATCH_SHA="72e57b317dcb603157e494e32a9068b021073abab2d8b0723962affec5ff78f9"
T_ROOT="${A4_TEST_ROOT:-}"
unset S23_ROOT S23_KEY A3_TEST_ROOT A3_TEST_CAP A4_TEST_ROOT
if [ "$MODE" = test ]; then
  case "$T_ROOT" in "$S23"/dev/A4_tests_*/*) ;; *) echo "A4 test root must lie under $S23/dev/A4_tests_*/"; exit 2 ;; esac
  export S23_ROOT="$T_ROOT" S23_KEY="hrhv2-s23-TEST" CUDA_VISIBLE_DEVICES=""
  ROOT="$T_ROOT"
else
  export CUDA_VISIBLE_DEVICES=""
  ROOT="$S23"
fi
STATE="$ROOT/resume"; STAGEA="$ROOT/results/stageA"; LOGS="$ROOT/logs"
now () { date -u +%Y-%m-%dT%H:%M:%SZ; }
st () { "$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); import s23common as C; C.append_status(sys.argv[2], sys.argv[3])' \
          "$S23/src" run-A4 "$1" >/dev/null; }

# ---------------------------------------------------------------- verification (rule A4.4)
verify_a4 () {
  local bad=0 nl line mh a h p f
  bash "$RES/launch_A3_analyze.sh" --verify-only > /dev/null 2>&1 || { echo "A4 verify: A3 verification (A2 --verify-only, analyze_A hash, A3 manifest) FAILED"; bad=1; }
  nl=$(grep -c -F "$A4_MARK" "$HOME/hrh_v2_stage1_20261008/STATUS.md" 2>/dev/null || true)
  if [ "${nl:-0}" = 0 ]; then
    if [ "$MODE" = test ]; then echo "A4 verify: no A4 FROZEN line yet (TEST root: candidate files only)"
    else echo "A4 verify: no '$A4_MARK' line in STATUS.md"; bad=1; fi
  elif [ "$nl" != 1 ]; then
    echo "A4 verify: $nl A4 FROZEN lines (expected 1)"; bad=1
  else
    line=$(grep -m1 -F "$A4_MARK" "$HOME/hrh_v2_stage1_20261008/STATUS.md")
    mh=$(printf '%s\n' "$line" | sed -n 's/.*A4_MANIFEST\.sha256 sha256 \([0-9a-f]\{64\}\).*/\1/p')
    a=$(sha256sum "$RES/A4_MANIFEST.sha256" 2>/dev/null | cut -d' ' -f1)
    { [ -n "$mh" ] && [ "$a" = "$mh" ]; } || { echo "A4 verify: A4_MANIFEST.sha256 differs from the A4 FROZEN line"; bad=1; }
    while read -r h p; do
      case "$h" in ""|\#*) continue ;; esac
      f="$S23/${p#S23/}"
      a=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
      [ "$a" = "$h" ] || { echo "A4 verify: A4 file hash MISMATCH $p"; bad=1; }
      [ -w "$f" ] && { echo "A4 verify: A4 file writable (not frozen): $p"; bad=1; }
    done < "$RES/A4_MANIFEST.sha256"
    grep -q "  S23/resume/launch_A4_analyze.sh\$" "$RES/A4_MANIFEST.sha256" || { echo "A4 verify: launcher not in the A4 manifest"; bad=1; }
  fi
  return $bad
}
if ! verify_a4; then echo "$(now) A4 verification FAILED: nothing run"; exit 2; fi
[ "$VERIFY_ONLY" = 1 ] && { echo "$(now) A4 verification OK ($MODE)"; exit 0; }

# ---------------------------------------------------------------- preconditions (rule A4.3)
pre_bad=0
if [ "$MODE" = run ]; then
  grep -q "rc=5 analysis error (main)" "$RES/A3.failed" 2>/dev/null || { echo "A4 pre: resume/A3.failed does not record rc=5 analysis error (main)"; pre_bad=1; }
  [ -e "$RES/A3.done" ] && { echo "A4 pre: resume/A3.done exists"; pre_bad=1; }
  a=$(sha256sum "$STAGEA/N_match.json" 2>/dev/null | cut -d' ' -f1)
  [ "$a" = "$NMATCH_SHA" ] || { echo "A4 pre: N_match.json missing or not the one hashed into STATUS by the A3 nmatch phase"; pre_bad=1; }
fi
[ -e "$STAGEA/N_match.json" ] && [ ! -w "$STAGEA/N_match.json" ] || { echo "A4 pre: N_match.json missing or writable"; pre_bad=1; }
for f in stageA_results.json REPORT_stageA.md; do
  [ -e "$STAGEA/$f" ] && { echo "A4 pre: $STAGEA/$f already exists"; pre_bad=1; }
done
if pgrep -u "$(id -u)" -f "(analyze_A|a3_analyze|a4_analyze)\.py --phase|worker\.py --phase (pre|main) --pool" > /dev/null; then
  echo "A4 pre: a Stage A worker or analysis process is alive"; pre_bad=1
fi
COMPLETE_MSG=$(cd "$RES" && "$PY" a2_complete.py complete 2>&1) || { echo "A4 pre: main phase not complete: $COMPLETE_MSG"; pre_bad=1; }
COMPLETE_MSG=$(printf '%s\n' "$COMPLETE_MSG" | tail -1 | sed 's/^A2 complete: //')
[ -z "$(find "$STAGEA/cache" -name '*.tmp' 2>/dev/null)" ] || { echo "A4 pre: unfinished .tmp file in results/stageA/cache"; pre_bad=1; }
[ "$pre_bad" = 0 ] || { echo "$(now) A4 preconditions FAILED: nothing run"; exit 2; }

# ---------------------------------------------------------------- markers
for m in A4.started A4.done A4.failed; do
  [ -e "$STATE/$m" ] && { echo "$(now) stale marker $STATE/$m: move it aside (never delete) before a new start"; exit 9; }
done
exec 9>>"$STATE/A4.lock"
flock -n 9 || { echo "$(now) another A4 launcher holds $STATE/A4.lock"; exit 9; }
ATT_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ATT="$STATE/A4_attempts/$ATT_ID"
mkdir -p "$ATT" "$LOGS"
echo "$(now) attempt $ATT_ID pid $$ mode $MODE" > "$STATE/A4.started"
finish () {  # $1 rc, $2 text
  echo "$1" > "$ATT/rc"
  st "A4 attempt $ATT_ID END rc=$1: $2"
  if [ "$1" = 0 ]; then echo "$(now) rc=0 $2" > "$STATE/A4.done"; else echo "$(now) rc=$1 $2" > "$STATE/A4.failed"; fi
  echo "$(now) rc=$1 $2"
  exit "$1"
}
# record the disk cache the frozen pair_rows will reuse (rule A4.3(c)); names and hashes only, contents not opened
(cd "$STAGEA" && find cache -type f | LC_ALL=C sort | xargs -r sha256sum) > "$ATT/cache_before.sha256"
CH=$(sha256sum "$ATT/cache_before.sha256" | cut -d' ' -f1); CN=$(wc -l < "$ATT/cache_before.sha256")
st "A4 attempt $ATT_ID START ($MODE) via resume/launch_A4_analyze.sh sha256 $(sha256sum "$0" | cut -c1-12): verification OK (A3 --verify-only incl. A2: 37 frozen code, 5 frozen data, A1/A2/A3 manifests; A4 manifest vs the A4 FROZEN line); preconditions OK ($COMPLETE_MSG; N_match.json present and read-only; no main output yet; no Stage A process alive); disk cache reused as the frozen pair_rows rule allows: $CN files, listing sha256 $CH (A4_attempts/$ATT_ID/cache_before.sha256); main phase only; CPU only, one BLAS thread, no GPU use"

cd "$S23/src" || finish 8 "cannot cd to src"
"$PY" "$WRAP" --phase main >> "$LOGS/analyze_main.out" 2>&1 || finish 5 "analysis error (main); see logs/analyze_main.out"
for f in stageA_results.json REPORT_stageA.md; do
  [ -e "$STAGEA/$f" ] || finish 5 "main exited 0 but $f is missing"
  chmod 444 "$STAGEA/$f"
done
finish 0 "Stage A analysis done (A4): N_match.json $(sha256sum "$STAGEA/N_match.json" | cut -d' ' -f1), stageA_results.json $(sha256sum "$STAGEA/stageA_results.json" | cut -d' ' -f1), REPORT_stageA.md $(sha256sum "$STAGEA/REPORT_stageA.md" | cut -d' ' -f1) (all chmod 444)"
