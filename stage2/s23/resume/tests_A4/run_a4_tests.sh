#!/usr/bin/env bash
# Excluded A4 tests (R29). CPU only, CUDA hidden; never the real STATUS.md, runs/ or results/.
#  1. unit test (a4_analyze.py --selftest): the built module differs from the frozen source only on line 705; the
#     helper equals the frozen expression for every mc != 0 and gives obs == 0 for mc == 0 without an exception.
#  2. T11: copy of the finished A1 TEST root T1 (outputs written by the FROZEN analyze_A); stageA_results.json and
#     REPORT_stageA.md moved to T11_orig/; launch_A4_analyze.sh --test must give rc 0 and outputs equal to the
#     originals; a relaunch must refuse (rc 2).
#  3. T12: like T11, but its frozen_predictions.json sets pi_mc[*][R]["rho"] = 0.0 for the first ORDER rung R whose
#     observed P5b rho in T1 is finite (the real-tree situation: a zero MC rho with a finite observed rho; in T1 the
#     dMST rho is NaN, which the frozen line short-circuits). The FROZEN analyze_A --phase main must stop with
#     ZeroDivisionError; the A4 launcher must give rc 0 and outputs equal to the originals except
#     secondary.P5b.mc_rho_vs_obs.R (mc 0.0, within_1.5 == (obs == 0)).
set -u
S23="$HOME/hrh_v2_stage2_20261008/s23"
PY="$HOME/hrh_v2_stage1_20261008/venv/bin/python"
RES="$S23/resume"
L="$RES/launch_A4_analyze.sh"
SRC_T1="$S23/dev/A1_dryrun_20261009T235208Z/T1"
OUT="${1:-$S23/dev/A4_tests_$(date -u +%Y%m%dT%H%M%SZ)}"
case "$OUT" in "$S23"/dev/A4_tests_*) ;; *) echo "output must lie under $S23/dev/A4_tests_*"; exit 2 ;; esac
mkdir -p "$OUT"
SUM="$OUT/summary.log"
fails=0
log () { echo "$(date -u +%FT%TZ) $*" | tee -a "$SUM"; }
res () { if [ "$1" = 0 ]; then log "PASS $2"; else log "FAIL $2"; fails=$((fails+1)); fi; }
unset S23_ROOT S23_KEY A3_TEST_ROOT A3_TEST_CAP A4_TEST_ROOT
export CUDA_VISIBLE_DEVICES=""

log "== 1. unit test"
(cd "$RES" && "$PY" a4_analyze.py --selftest) > "$OUT/unit.log" 2>&1; rc=$?
cat "$OUT/unit.log" >> "$SUM"
res $rc "unit test rc=$rc"

mk () {  # $1 root name, $2 zero|keep
  local T="$OUT/$1" O="$OUT/$1_orig"
  cp -a "$SRC_T1" "$T" && mkdir -p "$O" || return 1
  for f in stageA_results.json REPORT_stageA.md; do mv "$T/results/stageA/$f" "$O/" || return 1; done
  [ -e "$T/results/stageA/N_match.json" ] || return 1
  if [ "$2" = zero ]; then
    local P="$T/results/prestep/frozen_predictions.json"
    chmod u+w "$P" && "$PY" - "$P" "$O/stageA_results.json" "$OUT/$1.rung" <<'EOF' || return 1
import json, math, sys
p, orig, out = sys.argv[1:4]; fr = json.load(open(p))
m = json.load(open(orig))["secondary"]["P5b"]["mc_rho_vs_obs"]
rung = next(r for r in ("dN_k10", "dO", "dEk", "dMST") if isinstance(m[r]["obs"], (int, float)) and math.isfinite(m[r]["obs"]))
for k in fr["pi_mc"]:
    fr["pi_mc"][k][rung]["rho"] = 0.0
json.dump(fr, open(p, "w"), indent=1, sort_keys=True)
open(out, "w").write(rung)
print(f"pi_mc[*][{rung}] rho set to 0.0 (T1 observed rho {m[rung]['obs']}); keys {sorted(fr['pi_mc'], key=float)}")
EOF
    chmod 444 "$P"
  fi
  echo "made $1 ($2)"
}
chk () {  # $1 root name, $2 zero|keep
  "$PY" - "$OUT/$1" "$OUT/$1_orig" "$2" <<'EOF'
import hashlib, json, os, sys
T, O, mode = sys.argv[1], sys.argv[2], sys.argv[3]
S = os.path.join(T, "results", "stageA")
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
bad = 0
def ck(ok, msg):
    global bad
    print(("ok   " if ok else "BAD  ") + msg); bad += 0 if ok else 1
ck(os.path.exists(os.path.join(T, "resume", "A4.done")), "A4.done written")
a, b = json.load(open(os.path.join(S, "stageA_results.json"))), json.load(open(os.path.join(O, "stageA_results.json")))
if mode == "zero":
    rung = open(T + ".rung").read().strip()
    m = a["secondary"]["P5b"]["mc_rho_vs_obs"][rung]
    ck(m["mc"] == 0.0 and m["within_1.5"] == (m["obs"] == 0) and m["obs"] == b["secondary"]["P5b"]["mc_rho_vs_obs"][rung]["obs"],
       f"{rung} entry: mc {m['mc']}, obs unchanged {m['obs']}, within_1.5 {m['within_1.5']} == (obs == 0)")
    b["secondary"]["P5b"]["mc_rho_vs_obs"][rung] = a["secondary"]["P5b"]["mc_rho_vs_obs"][rung]
dk = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
note = " apart from that rung's P5b entry" if mode == "zero" else ""
ck(not dk, f"stageA_results.json equal to the frozen-code original{note} (differing top-level keys: {dk})")
ck(h(os.path.join(S, "REPORT_stageA.md")) == h(os.path.join(O, "REPORT_stageA.md")), "REPORT_stageA.md byte-equal")
for f in ("stageA_results.json", "REPORT_stageA.md"):
    ck(not os.access(os.path.join(S, f), os.W_OK), f"{f} read-only")
allst = open(os.path.join(T, "STATUS_test.md")).read()
allst = allst[allst.find("[s23-run-A4]"):] if "[s23-run-A4]" in allst else ""
pos = [allst.find(w) for w in ("START", "analysis done", "END rc=0")]
ck(all(p >= 0 for p in pos) and pos == sorted(pos), "STATUS_test.md lines in order: START, analysis done, END rc=0")
print(f"check {os.path.basename(T)}: {bad} problems")
sys.exit(1 if bad else 0)
EOF
}

log "== 2. T11 (frozen predictions as in T1)"
mk T11 keep > "$OUT/T11.make.log" 2>&1; res $? "make T11 ($(tail -1 "$OUT/T11.make.log"))"
A4_TEST_ROOT="$OUT/T11" bash "$L" --test > "$OUT/T11.launch1.log" 2>&1; rc=$?
res $([ "$rc" = 0 ]; echo $?) "T11 launch rc=$rc (expected 0)"
chk T11 keep > "$OUT/T11.check.log" 2>&1; r=$?; cat "$OUT/T11.check.log" >> "$SUM"; res $r "T11 outputs equal to the frozen-code originals"
A4_TEST_ROOT="$OUT/T11" bash "$L" --test > "$OUT/T11.launch2.log" 2>&1; rc=$?
res $([ "$rc" = 2 ]; echo $?) "T11 relaunch refused rc=$rc (expected 2: outputs exist)"

log "== 3. T12 (pi_mc rho = 0.0 for a rung with finite observed rho)"
mk T12 zero > "$OUT/T12.make.log" 2>&1; res $? "make T12 ($(tail -1 "$OUT/T12.make.log"))"
(cd "$S23/src" && S23_ROOT="$OUT/T12" S23_KEY=hrhv2-s23-TEST "$PY" analyze_A.py --phase main) > "$OUT/T12.frozen_main.log" 2>&1; rc=$?
res $([ "$rc" != 0 ] && grep -q "ZeroDivisionError" "$OUT/T12.frozen_main.log" && [ ! -e "$OUT/T12/results/stageA/stageA_results.json" ]; echo $?) \
  "T12 frozen analyze_A --phase main stops with ZeroDivisionError (rc=$rc), no output written"
tail -3 "$OUT/T12.frozen_main.log" | sed 's/^/   /' >> "$SUM"
A4_TEST_ROOT="$OUT/T12" bash "$L" --test > "$OUT/T12.launch1.log" 2>&1; rc=$?
res $([ "$rc" = 0 ]; echo $?) "T12 launch rc=$rc (expected 0)"
chk T12 zero > "$OUT/T12.check.log" 2>&1; r=$?; cat "$OUT/T12.check.log" >> "$SUM"; res $r "T12 outputs equal to the frozen-code originals apart from that rung's P5b entry"
grep -h "A4 analyze --phase" "$OUT"/T11/logs/analyze_main.out "$OUT"/T12/logs/analyze_main.out >> "$SUM" 2>/dev/null
log "== done: $fails failures; outputs in $OUT"
exit $fails
