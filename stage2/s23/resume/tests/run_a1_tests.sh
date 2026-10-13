#!/usr/bin/env bash
# Excluded A1 dry-run tests (R29). No GPU work, no Stage A statistic, never the real STATUS.md or runs/.
#  1. launch_stageA_resume.sh --dry-run on the real tree (read-only).
#  2. launch_stageA_resume.sh --test on TEST-key tiny roots under s23/dev/A1_dryrun_<utc>/ (CPU only,
#     CUDA_VISIBLE_DEVICES empty): T1 full sequence, T4 relaunch, T2 speed-guard pause then relaunch,
#     T5 speed-guard pause with live workers (0.1 s snapshots) then relaunch, T3 foreign GPU process ->
#     rule A1.2(d) pause.
#  3. gate_then_launch.sh --dry-run with fake S2.2 markers and fake compute-apps lists: waiting, reset,
#     launch (launcher --dry-run), cancel, refusal on stale markers.
# At most 4 CPU threads: one tiny root at a time (3 single-thread workers + sidecar).
set -u
S23="$HOME/hrh_v2_stage2_20261008/s23"
PY="$HOME/hrh_v2_stage1_20261008/venv/bin/python"
RES="$S23/resume"
T="$RES/tests"
L="$RES/launch_stageA_resume.sh"
G="$RES/gate_then_launch.sh"
OUT="${1:-$S23/dev/A1_dryrun_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
SUM="$OUT/summary.log"
fails=0
log () { echo "$(date -u +%FT%TZ) $*" | tee -a "$SUM"; }
res () { if [ "$1" = 0 ]; then log "PASS $2"; else log "FAIL $2"; fails=$((fails+1)); fi; }
unset S23_ROOT S23_KEY A1_GUARD_S A1_FAKE_SMI A1_TEST_ROOT A1_SNAP_INTERVAL

log "== 1. real-tree dry run"
bash "$L" --dry-run > "$OUT/real_dry_run.log" 2>&1; rc=$?
res $([ $rc = 0 ] && grep -q "A1 dry run OK" "$OUT/real_dry_run.log"; echo $?) "real-tree dry run rc=$rc"
grep -E "^(PASS|FAIL) |A1 selftest|A1 verify|A-pilot/G1b ledger lines|^pre todo|^main todo" "$OUT/real_dry_run.log" >> "$SUM"

make_root () {
  mkdir -p "$OUT/$1"
  (cd "$T" && S23_ROOT="$OUT/$1" S23_KEY=hrhv2-s23-TEST CUDA_VISIBLE_DEVICES= "$PY" a1_make_test_root.py --root "$OUT/$1") \
    > "$OUT/$1.make.log" 2>&1
}
check () {  # $1 root, $2 scenario, $3 rc
  (cd "$T" && S23_ROOT="$OUT/$1" S23_KEY=hrhv2-s23-TEST CUDA_VISIBLE_DEVICES= "$PY" a1_check_test_root.py --root "$OUT/$1" \
    --scenario "$2" --rc "$3") > "$OUT/$1.check_$2.log" 2>&1
  r=$?; cat "$OUT/$1.check_$2.log" >> "$SUM"; res $r "scenario $2 checks (launcher rc=$3)"
}

log "== 2. TEST-root scenarios"
make_root T1; res $? "make T1 root"
printf 'OWN\n' > "$OUT/T1/fake_smi"
A1_TEST_ROOT="$OUT/T1" A1_FAKE_SMI="$OUT/T1/fake_smi" bash "$L" --test > "$OUT/T1.launch1.log" 2>&1; rc=$?
check T1 T1 $rc
sleep 2
A1_TEST_ROOT="$OUT/T1" A1_FAKE_SMI="$OUT/T1/fake_smi" bash "$L" --test > "$OUT/T1.launch2.log" 2>&1; rc=$?
check T1 T4 $rc

make_root T2; res $? "make T2 root"
printf 'OWN\n' > "$OUT/T2/fake_smi"
A1_TEST_ROOT="$OUT/T2" A1_FAKE_SMI="$OUT/T2/fake_smi" A1_GUARD_S=0 bash "$L" --test > "$OUT/T2.launch1.log" 2>&1; rc=$?
check T2 T2a $rc
sleep 2
A1_TEST_ROOT="$OUT/T2" A1_FAKE_SMI="$OUT/T2/fake_smi" bash "$L" --test > "$OUT/T2.launch2.log" 2>&1; rc=$?
check T2 T2b $rc

make_root T5; res $? "make T5 root"
printf 'OWN\n' > "$OUT/T5/fake_smi"
A1_TEST_ROOT="$OUT/T5" A1_FAKE_SMI="$OUT/T5/fake_smi" A1_GUARD_S=0 A1_SNAP_INTERVAL=0.1 bash "$L" --test > "$OUT/T5.launch1.log" 2>&1; rc=$?
check T5 T5a $rc
sleep 2
A1_TEST_ROOT="$OUT/T5" A1_FAKE_SMI="$OUT/T5/fake_smi" bash "$L" --test > "$OUT/T5.launch2.log" 2>&1; rc=$?
check T5 T5b $rc

make_root T3; res $? "make T3 root"
printf 'OWN\n424242, /usr/bin/foreign_job, 100\n' > "$OUT/T3/fake_smi"
A1_TEST_ROOT="$OUT/T3" A1_FAKE_SMI="$OUT/T3/fake_smi" bash "$L" --test > "$OUT/T3.launch1.log" 2>&1; rc=$?
check T3 T3 $rc

log "== 3. gate dry runs"
GD="$OUT/gate1"; mkdir -p "$GD"
printf '123, /x/python, 2600\n' > "$GD/fake_smi"
A1_GATE_INTERVAL=1 A1_GATE_WINDOW=3 bash "$G" --dry-run "$GD" > "$GD/gate.out" 2>&1 &
GP=$!
hb () { sleep "$1"; log "gate1 heartbeat: $(cat "$GD/gate.heartbeat" 2>/dev/null)"; }
hb 4
grep -q "S2.2 marker=none" "$GD/gate.heartbeat"; res $? "gate waits while S2.2 has no marker"
touch "$GD/fake_s22/pipeline.done"; hb 3
grep -q "compute processes=1" "$GD/gate.heartbeat"; res $? "gate waits while a compute process is listed"
: > "$GD/fake_smi"; hb 2
grep -q "GPU free since" "$GD/gate.heartbeat"; res $? "gate starts the free window"
printf 'ERR\n' > "$GD/fake_smi"; hb 2
grep -q "query rc=1" "$GD/gate.heartbeat"; res $? "failed query resets the window"
: > "$GD/fake_smi"
wait $GP; rc=$?
log "gate1 heartbeat: $(cat "$GD/gate.heartbeat")"
log "gate1 launched: $(cat "$GD/gate.launched" 2>/dev/null)"
[ $rc = 0 ] && [ -e "$GD/gate.launched" ] && [ ! -e "$GD/gate.failed" ] && grep -q "rc=0" "$GD/launch_resume.rc" \
  && grep -q "A1 dry run OK" "$GD/launch_resume.out"
res $? "gate launches the launcher (--dry-run) after the window; launcher rc 0"
[ "$(grep -o "[0-9]* consecutive checks" "$GD/gate.launched" | cut -d" " -f1)" -ge 3 ]; res $? "window covered >= 3 s with >= 3 checks"

GD2="$OUT/gate2"; mkdir -p "$GD2"
printf '123, /x/python, 2600\n' > "$GD2/fake_smi"
A1_GATE_INTERVAL=1 A1_GATE_WINDOW=3 bash "$G" --dry-run "$GD2" > "$GD2/gate.out" 2>&1 &
GP=$!
sleep 3; touch "$GD2/CANCEL"; wait $GP
grep -q cancelled "$GD2/gate.failed" && [ ! -e "$GD2/gate.launched" ]; res $? "CANCEL stops the gate before launch"
bash "$G" --dry-run "$GD2" > "$GD2/gate_again.out" 2>&1; rc=$?
[ $rc = 9 ]; res $? "gate refuses to start over stale markers (rc=$rc)"

log "== done: $fails failures; outputs in $OUT"
exit $fails
