#!/usr/bin/env bash
# Excluded A3 tests (R29). CPU only, CUDA hidden; never the real STATUS.md, runs/ or results/.
#  1. unit test (a3_analyze.py --selftest): the frozen Store.K raises KeyError on distinct load #1501; the corrected
#     memo returns np.load of the same path for 1,700 ids x 2 rounds, stays within the bound, agrees with the frozen
#     memo on the first 1,500 loads, handles extra= and missing files as frozen.
#  2. TEST roots T9 (memo bound lowered to 3: evictions on almost every load) and T10 (frozen bound 1500), copies of
#     the finished A1 TEST root T1, whose N_match.json / stageA_results.json / REPORT_stageA.md / cache were written
#     by the FROZEN analyze_A. The analysis outputs and cache are moved to <T>_orig/; the within-A cache files of all
#     but the last grid N are put back as the "crashed attempt" files; launch_A3_analyze.sh --test must give rc 0,
#     byte-equal regenerated cache, and outputs equal to the frozen-code originals. A relaunch must refuse (rc 2).
set -u
S23="$HOME/hrh_v2_stage2_20261008/s23"
PY="$HOME/hrh_v2_stage1_20261008/venv/bin/python"
RES="$S23/resume"
L="$RES/launch_A3_analyze.sh"
SRC_T1="$S23/dev/A1_dryrun_20261009T235208Z/T1"
OUT="${1:-$S23/dev/A3_tests_$(date -u +%Y%m%dT%H%M%SZ)}"
case "$OUT" in "$S23"/dev/A3_tests_*) ;; *) echo "output must lie under $S23/dev/A3_tests_*"; exit 2 ;; esac
mkdir -p "$OUT"
SUM="$OUT/summary.log"
fails=0
log () { echo "$(date -u +%FT%TZ) $*" | tee -a "$SUM"; }
res () { if [ "$1" = 0 ]; then log "PASS $2"; else log "FAIL $2"; fails=$((fails+1)); fi; }
unset S23_ROOT S23_KEY A3_TEST_ROOT A3_TEST_CAP
export CUDA_VISIBLE_DEVICES=""

log "== 1. unit test"
(cd "$RES" && "$PY" a3_analyze.py --selftest "$OUT/unit") > "$OUT/unit.log" 2>&1; rc=$?
cat "$OUT/unit.log" >> "$SUM"
res $rc "unit test rc=$rc"

mk () {  # $1 root name
  local T="$OUT/$1" O="$OUT/$1_orig"
  cp -a "$SRC_T1" "$T" || return 1
  mkdir -p "$O/cache" || return 1
  for f in N_match.json stageA_results.json REPORT_stageA.md; do mv "$T/results/stageA/$f" "$O/" || return 1; done
  mv "$T"/results/stageA/cache/arm0_*.json "$T"/results/stageA/cache/rm_*.json "$O/cache/" || return 1
  [ -d "$T/results/stageA/cache/F" ] && { mv "$T/results/stageA/cache/F" "$O/cache/F" || return 1; }
  local last; last=$(ls "$O"/cache/arm0_l*_wA_n*.json | sed -n 's/.*arm0_l\([0-9]*\)_wA_.*/\1/p' | sort -n | tail -1)
  mkdir -p "$T/resume" || return 1
  : > "$T/resume/A3_crashed_cache.sha256"
  for f in "$O"/cache/arm0_l*_wA_n*.json; do
    case "$f" in *"arm0_l${last}_wA_"*) continue ;; esac
    cp -p "$f" "$T/results/stageA/cache/" || return 1
    echo "$(sha256sum "$f" | cut -d' ' -f1)  results/stageA/cache/$(basename "$f")" >> "$T/resume/A3_crashed_cache.sha256"
  done
  echo "made $1: crashed-attempt files $(wc -l < "$T/resume/A3_crashed_cache.sha256") (last grid N l$last left out)"
}
chk () {  # $1 root name
  "$PY" - "$OUT/$1" "$OUT/$1_orig" <<'EOF'
import hashlib, json, os, sys
T, O = sys.argv[1], sys.argv[2]
S = os.path.join(T, "results", "stageA")
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
bad = 0
def ck(ok, msg):
    global bad
    print(("ok   " if ok else "BAD  ") + msg); bad += 0 if ok else 1
ck(os.path.exists(os.path.join(T, "resume", "A3.done")), "A3.done written")
ck(h(os.path.join(S, "N_match.json")) == h(os.path.join(O, "N_match.json")), "N_match.json byte-equal to the frozen-code original")
a, b = json.load(open(os.path.join(S, "stageA_results.json"))), json.load(open(os.path.join(O, "stageA_results.json")))
dk = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
ck(not dk, f"stageA_results.json equal to the frozen-code original (differing top-level keys: {dk})")
ck(h(os.path.join(S, "REPORT_stageA.md")) == h(os.path.join(O, "REPORT_stageA.md")), "REPORT_stageA.md byte-equal")
n = m = 0
for root, _, fs in os.walk(os.path.join(O, "cache")):
    for f in fs:
        p = os.path.join(root, f); q = os.path.join(S, "cache", os.path.relpath(p, os.path.join(O, "cache")))
        n += 1; m += os.path.exists(q) and h(p) == h(q)
ck(n > 0 and m == n, f"regenerated cache files byte-equal to the frozen-code originals: {m}/{n}")
aside = os.path.join(S, "_crashed_cache_A2_20261010T083303Z")
lst = [l.split() for l in open(os.path.join(T, "resume", "A3_crashed_cache.sha256")) if l.strip()]
ck(all(h(os.path.join(aside, os.path.basename(p))) == hh for hh, p in lst), f"{len(lst)} crashed-attempt files moved aside intact")
for f in ("stageA_results.json", "REPORT_stageA.md", "N_match.json"):
    ck(not os.access(os.path.join(S, f), os.W_OK), f"{f} read-only")
st = [l for l in open(os.path.join(T, "STATUS_test.md")) if "[s23-run-A3]" in l]
want = ["START", "moved aside", "N_match written", "nmatch done", "END rc=0"]
allst = open(os.path.join(T, "STATUS_test.md")).read()
allst = allst[allst.find("[s23-run-A3]"):] if "[s23-run-A3]" in allst else ""
pos = [allst.find(w) for w in want]
ck(all(p >= 0 for p in pos) and pos == sorted(pos), f"STATUS_test.md lines in order: {want} ({len(st)} run-A3 lines)")
ck("byte-equal to the crashed attempt's: " in allst and "differ:" not in allst, "launcher compared the crashed-attempt files: equal")
print(f"check {os.path.basename(T)}: {bad} problems")
sys.exit(1 if bad else 0)
EOF
}

log "== 2. TEST roots"
for spec in "T9 3" "T10 -"; do
  set -- $spec
  mk "$1" > "$OUT/$1.make.log" 2>&1; res $? "make $1 ($(tail -1 "$OUT/$1.make.log"))"
  if [ "$2" = - ]; then
    A3_TEST_ROOT="$OUT/$1" bash "$L" --test > "$OUT/$1.launch1.log" 2>&1; rc=$?
  else
    A3_TEST_ROOT="$OUT/$1" A3_TEST_CAP="$2" bash "$L" --test > "$OUT/$1.launch1.log" 2>&1; rc=$?
  fi
  res $([ "$rc" = 0 ]; echo $?) "$1 launch (memo bound $2) rc=$rc (expected 0)"
  chk "$1" > "$OUT/$1.check.log" 2>&1; r=$?; cat "$OUT/$1.check.log" >> "$SUM"; res $r "$1 outputs equal to the frozen-code originals"
  A3_TEST_ROOT="$OUT/$1" bash "$L" --test > "$OUT/$1.launch2.log" 2>&1; rc=$?
  res $([ "$rc" = 2 ]; echo $?) "$1 relaunch refused rc=$rc (expected 2: outputs exist)"
done
grep -h "Store.K replaced" "$OUT"/T9/logs/analyze_*.out "$OUT"/T10/logs/analyze_*.out >> "$SUM" 2>/dev/null
log "== done: $fails failures; outputs in $OUT"
exit $fails
