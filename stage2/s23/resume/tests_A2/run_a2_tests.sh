#!/usr/bin/env bash
# Excluded A2 tests (R29). No GPU work, no Stage A statistic of the real tree, never the real STATUS.md or runs/.
#  1. launch_stageA_A2.sh --dry-run on the real tree (read-only; frozen worker loop with recorders, CUDA hidden).
#  2. launch_stageA_A2.sh --test on TEST-key tiny roots under s23/dev/A2_tests_<utc>/ (copies of the finished A1
#     TEST root T1; CPU only, CUDA_VISIBLE_DEVICES empty):
#       T6 full sequence with the ledger above both frozen limits (5 R- items redone, partials moved, nmatch + main
#          re-run and equal to the originals), then a relaunch over stale markers (rc 9), then a relaunch after the
#          markers are moved aside (rc 0, nothing redone, analysis skipped);
#       T7 a finished item changed after the A2 finished-items manifest -> prep refuses (rc 2);
#       T8 no A2 FROZEN line -> verification refuses (rc 2).
# At most 4 CPU threads: one root at a time (3 single-thread workers + the launcher).
set -u
S23="$HOME/hrh_v2_stage2_20261008/s23"
PY="$HOME/hrh_v2_stage1_20261008/venv/bin/python"
RES="$S23/resume"
T="$RES/tests_A2"
L="$RES/launch_stageA_A2.sh"
SRC_T1="$S23/dev/A1_dryrun_20261009T235208Z/T1"
OUT="${1:-$S23/dev/A2_tests_$(date -u +%Y%m%dT%H%M%SZ)}"
case "$OUT" in "$S23"/dev/A2_tests_*) ;; *) echo "output must lie under $S23/dev/A2_tests_*"; exit 2 ;; esac
mkdir -p "$OUT"
SUM="$OUT/summary.log"
fails=0
log () { echo "$(date -u +%FT%TZ) $*" | tee -a "$SUM"; }
res () { if [ "$1" = 0 ]; then log "PASS $2"; else log "FAIL $2"; fails=$((fails+1)); fi; }
unset S23_ROOT S23_KEY A2_TEST_ROOT

log "== 1. real-tree dry run"
bash "$L" --dry-run > "$OUT/real_dry_run.log" 2>&1; rc=$?
res $([ $rc = 0 ] && grep -q "A2 dry run OK" "$OUT/real_dry_run.log"; echo $?) "real-tree dry run rc=$rc"
grep -E "A2 verify|^main todo|^pre todo|frozen GPU pool|dry-run loop|frozen constants:|-> rc|absent|exists|A2 g4" "$OUT/real_dry_run.log" >> "$SUM"

mk () {  # $1 name, then options for a2_make_test_state.py
  local n="$1"; shift
  cp -a "$SRC_T1" "$OUT/$n" && \
  (cd "$T" && S23_ROOT="$OUT/$n" S23_KEY=hrhv2-s23-TEST CUDA_VISIBLE_DEVICES= "$PY" a2_make_test_state.py --root "$OUT/$n" "$@") \
    > "$OUT/$n.make.log" 2>&1
}
chk () {  # $1 root name, $2 scenario, $3 launcher rc
  (cd "$T" && S23_ROOT="$OUT/$1" S23_KEY=hrhv2-s23-TEST CUDA_VISIBLE_DEVICES= "$PY" a2_check_test_state.py --root "$OUT/$1" \
    --scenario "$2" --rc "$3") > "$OUT/$1.check_$2.log" 2>&1
  r=$?; cat "$OUT/$1.check_$2.log" >> "$SUM"; res $r "scenario $1/$2 checks (launcher rc=$3)"
}

log "== 2. TEST roots"
mk T6; res $? "make T6 ($(tail -1 "$OUT/T6.make.log"))"
A2_TEST_ROOT="$OUT/T6" bash "$L" --test > "$OUT/T6.launch1.log" 2>&1; rc=$?
chk T6 full $rc
A2_TEST_ROOT="$OUT/T6" bash "$L" --test > "$OUT/T6.launch2.log" 2>&1; rc=$?
chk T6 stale $rc
mkdir -p "$OUT/T6/resume/_moved_markers_1" && mv "$OUT/T6/resume/A2.started" "$OUT/T6/resume/A2.done" "$OUT/T6/resume/_moved_markers_1/"
sleep 1
A2_TEST_ROOT="$OUT/T6" bash "$L" --test > "$OUT/T6.launch3.log" 2>&1; rc=$?
chk T6 rerun $rc

mk T7 --corrupt; res $? "make T7 ($(tail -1 "$OUT/T7.make.log"))"
A2_TEST_ROOT="$OUT/T7" bash "$L" --test > "$OUT/T7.launch1.log" 2>&1; rc=$?
chk T7 corrupt $rc

mk T8 --no-frozen-line; res $? "make T8 ($(tail -1 "$OUT/T8.make.log"))"
A2_TEST_ROOT="$OUT/T8" bash "$L" --test > "$OUT/T8.launch1.log" 2>&1; rc=$?
chk T8 nofrozen $rc

log "== done: $fails failures; outputs in $OUT"
exit $fails
