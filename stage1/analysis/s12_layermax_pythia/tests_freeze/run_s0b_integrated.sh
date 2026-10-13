#!/bin/bash
# Pythia axis, step S0b: the integrated pre-freeze test battery of role pyt-code-freeze (2026-10-10). NEW FILE (PYT-AXIS).
# Plan analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md (sha256 6de89603...879e7b) section 11, step S0b and V-P8/V-P9.
# Runs once, on the code that is about to be frozen, every S0b test that needs no GPU, each writer role's battery
# unchanged (integration only; no E, I or R logic is touched or merged):
#   1 extraction side: extract/pythia_axis/tests/test_extract_pyt.py (13 CPU tests incl. 16 gated dry runs)
#   2 infrastructure: tests_infra/test_infra_pyt.py (15 tests) and partb_checks_pyt.py on the shared stand-in Part B
#   3 implementation E: early/tests/run_e_tests.py (E dry runs, V-P8, V-P9, stand-in end to end, V-P5 tests, CPU smoke)
#   4 implementation I: indep/tests/run_all_s0b.sh (V-P9, stand-in, determinism, V-P8, gate dry runs, CPU smoke)
#   5 implementation R: review/tests (dry runs, V-P9, V-P8) and R's three stand-in fixtures (compute phase)
#   6 cross-implementation fixture X (plan V-P9 last item; tests_freeze/xfixture_pyt.py): E aggregate, I aggregate,
#     forecast_pyt.py --impl E and --impl I, R review (compute + compare), E's vl5_e_vs_i.py and I's cross_validate.py
#     on ONE 7-size stand-in (5 BLOOMZ + llama-13b + llama-30b) with the shared stand-in Part B s0b_scratch/PB/partB_vp9
#     (variant base) and three Part B test variants flipH1, bandfail, bandwide (other branches of the section 7 table),
#     then the three H1-positive SYNTHETIC variants h1A, h1Aprime, h1B of lead ruling R3 (categories A, A-att, A', B,
#     modifiers M2, M5_A, M5_B, premise; xfixture_pyt.py stage synth builds the synthetic pair files, I runs without
#     --fixture on them; they need variant base of the same tag first, for E's stand-in map and clip cache)
#   7 tests_freeze/collect_s0b_record.py: the consolidated record S0B_TEST_RECORD_PYT.json (pass flag, every step, every
#     failed pre-freeze test run of the axis)
# Not here: the F1(a) dry runs of every gated command, F1(d) and the manifest (s0b_freeze.py), feasibility_pyt.py (its one
# real run), the GPU smoke and gate G0 (GPU order of the lead: no Pythia GPU process before S2.2 v1.1 is done, the S2.3
# resume launcher has exited and the GPU has been idle 10 minutes).
# GPU: none (CUDA_VISIBLE_DEVICES is empty for every child). Threads: at most 4 launch_pyt.sh tokens at a time.
# S0b test mode only (PYT_S0B_TEST=1, refused after CODE FROZEN); outputs only under analysis/s12_layermax_pythia/s0b_scratch/
# and the extraction side's own extract/pythia_axis/s0b_test_out/. No file is deleted; every name is new.
# usage: bash analysis/s12_layermax_pythia/tests_freeze/run_s0b_integrated.sh [TAG=UTC yyyymmddThhmmssZ] [all|xonly]
#        xonly = section 6 only (diagnostic runs of fixture X; no consolidated record, so never used for the freeze)
set -u
PROJ="$HOME/hrh_v2_stage1_20261008"
cd "$PROJ" || exit 70
TAG="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
MODE="${2:-all}"
case "$MODE" in all|xonly) ;; *) echo "MODE must be all or xonly" >&2; exit 64;; esac
AX=analysis/s12_layermax_pythia
SC="$PROJ/$AX/s0b_scratch"
FZ="$SC/FZ/run_$TAG"
if [ -e "$FZ" ]; then echo "$FZ exists (no-delete rule; use a new TAG)" >&2; exit 65; fi
mkdir -p "$FZ/logs"
export PYT_S0B_TEST=1 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYT_LOGDIR="$FZ/logs"
L="$PROJ/$AX/launch_pyt.sh"
PY="$PROJ/venv/bin/python"
ST="$FZ/steps.tsv"
printf 'step\trc\ttries\tstart_utc\tend_utc\n' > "$ST"
step() {  # step NAME THREADS CMD... : through launch_pyt.sh; exit 75 (tokens busy, nothing started) is retried
  local name="$1" th="$2"; shift 2
  local t0 rc=75 i
  t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  for i in $(seq 120); do "$L" "pyt_FZ_${TAG}_${name}" "$th" -- "$@"; rc=$?; [ "$rc" -ne 75 ] && break; sleep 10; done
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$rc" "$i" "$t0" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ST"
  echo "[run_s0b_integrated] $(date -u +%H:%M:%SZ) $name rc=$rc tries=$i"
}
direct() {  # direct NAME CMD... : drivers that start their own children through launch_pyt.sh
  local name="$1"; shift
  local t0 rc
  t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  "$@" > "$FZ/logs/$name.out" 2>&1; rc=$?
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$rc" 1 "$t0" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ST"
  echo "[run_s0b_integrated] $(date -u +%H:%M:%SZ) $name rc=$rc"
}
echo "[run_s0b_integrated] tag $TAG mode $MODE start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$MODE" = all ]; then
# ---- 1 extraction side
step X_extract_cpu 4 "$PY" extract/pythia_axis/tests/test_extract_pyt.py
# ---- 2 infrastructure
step infra_tests 2 "$PY" "$AX/tests_infra/test_infra_pyt.py"
step partb_checks_vp9 1 "$PY" "$AX/partb_checks_pyt.py" --partb-dir "$SC/PB/partB_vp9"
# ---- 3 implementation E (its driver records every child run in s0b_scratch/E/final_<UTC>/test_log_E.json)
direct E_battery "$PY" "$AX/early/tests/run_e_tests.py"
# ---- 4 implementation I (its driver prints "step NAME rc=N" for every child)
direct I_battery bash "$AX/indep/tests/run_all_s0b.sh" 4
# ---- 5 implementation R (the commands of R_test_runs.json, new names and outputs)
step R_dryrun 1 bash "$AX/review/tests/dryrun_gate_r.sh"
step R_vp9 1 "$PY" "$AX/review/tests/test_vp9_r.py" --out "$SC/R/vp9/vp9_known_answers_FZ_$TAG.json"
step R_vp8 2 "$PY" "$AX/review/tests/test_vp8_r.py" --out "$SC/R/vp8/vp8_r_FZ_$TAG.json"
step R_fx 1 "$PY" "$AX/review/make_test_fixture_pyt.py" --root "$SC/R/fz_$TAG"
step R_fx_flip 1 "$PY" "$AX/review/make_test_fixture_pyt.py" --root "$SC/R/fz_${TAG}_flip" --flip-partb-material
step R_fx_nol2 1 "$PY" "$AX/review/make_test_fixture_pyt.py" --root "$SC/R/fz_${TAG}_nol2" --no-l2
step R_review_fx 2 "$PY" "$AX/review/pyt_axis_review.py" --test-root "$SC/R/fz_$TAG" --phase compute
step R_review_flip 2 "$PY" "$AX/review/pyt_axis_review.py" --test-root "$SC/R/fz_${TAG}_flip" --impl E --phase compute
step R_review_nol2 2 "$PY" "$AX/review/pyt_axis_review.py" --test-root "$SC/R/fz_${TAG}_nol2" --impl I --phase compute
fi
# ---- 6 cross-implementation fixture X, with the shared stand-in Part B (base) and three Part B variants (test inputs
#      written by xfixture_pyt.py partb; see its docstring; added after run 20261010T022655Z), and the three synthetic
#      H1-positive variants of lead ruling R3 (added after run 20261010T024645Z, before any run with them)
run_x() {
  local V="$1"
  local XE="$SC/E/X_${TAG}_$V" XI="$SC/I/X_${TAG}_$V" XF="$SC/PB/X_${TAG}_$V" XR="$SC/R/X_${TAG}_$V"
  local PB="$SC/PB/partB_vp9"
  [ "$V" != base ] && PB="$FZ/partB_$V"
  step "X_partb_$V" 1 "$PY" "$AX/tests_freeze/xfixture_pyt.py" partb --tag "$TAG" --variant "$V"
  case "$V" in
  h1A|h1Aprime|h1B)  # R3 synthetic variants: synthetic E- and I-format pair files (no E builder; I without --fixture)
  step "X_synth_$V" 1 "$PY" "$AX/tests_freeze/xfixture_pyt.py" synth --tag "$TAG" --variant "$V"
  step "X_E_aggregate_$V" 2 env PYT_E_TEST_ROOT="$XE" PYT_E_TEST_STANDINS="$XE/standins.json" PYT_E_TEST_PARTB="$PB" \
       "$PY" "$AX/early/aggregate_pyt.py"
  step "X_I_aggregate_$V" 2 env PYT_I_TEST_OUT="$XI/indep" PYT_I_TEST_PARTB="$PB" \
       "$PY" "$AX/indep/lmx_pyt_aggregate.py" --partb "$PB/partB_features_pyt.json"
  ;;
  *)
  step "X_E_build_$V" 1 env PYT_E_TEST_ROOT="$XE" PYT_E_TEST_STANDINS="$XE/standins.json" PYT_E_TEST_PARTB="$PB" \
       "$PY" "$AX/early/tests/build_standin_fixture.py"
  step "X_E_aggregate_$V" 2 env PYT_E_TEST_ROOT="$XE" PYT_E_TEST_STANDINS="$XE/standins.json" PYT_E_TEST_PARTB="$PB" \
       "$PY" "$AX/early/aggregate_pyt.py"
  step "X_I_aggregate_$V" 2 env PYT_I_TEST_OUT="$XI/indep" PYT_I_TEST_PARTB="$PB" \
       "$PY" "$AX/indep/lmx_pyt_aggregate.py" --fixture --partb "$PB/partB_features_pyt.json"
  ;;
  esac
  step "X_inputs_$V" 1 "$PY" "$AX/tests_freeze/xfixture_pyt.py" inputs --tag "$TAG" --variant "$V"
  step "X_forecast_E_$V" 1 "$PY" "$AX/forecast_pyt.py" --impl E --test-root "$XE" --partb "$PB/partB_features_pyt.json" \
       --verdicts "$XE/early/verdicts_pyt.json" --out "$XF/forecast/E"
  step "X_forecast_I_$V" 1 "$PY" "$AX/forecast_pyt.py" --impl I --test-root "$XR/results/S1_2_layermax_pythia" \
       --partb "$PB/partB_features_pyt.json" --verdicts "$XI/indep/verdicts_pyt.json" --out "$XF/forecast/I"
  step "X_verdicts_$V" 1 "$PY" "$AX/tests_freeze/xfixture_pyt.py" verdicts --tag "$TAG" --variant "$V"
  step "X_R_review_$V" 2 "$PY" "$AX/review/pyt_axis_review.py" --test-root "$XR"
  step "X_E_vl5_$V" 2 env PYT_E_TEST_ROOT="$XE" PYT_E_TEST_STANDINS="$XE/standins.json" PYT_E_TEST_PARTB="$PB" \
       "$PY" "$AX/early/vl5_e_vs_i.py"
  step "X_I_cross_$V" 2 env PYT_I_TEST_OUT="$XI/indep" PYT_I_TEST_PARTB="$PB" PYT_I_TEST_EDIR="$XE/early" \
       "$PY" "$AX/indep/cross_validate.py" --parts 1,2,3 --pres prh_clip+l2,l2 --e-verdicts "$XE/early/verdicts_pyt.json"
  step "X_compare_$V" 1 "$PY" "$AX/tests_freeze/xfixture_pyt.py" compare --tag "$TAG" --variant "$V"
}
for V in base flipH1 bandfail bandwide h1A h1Aprime h1B; do run_x "$V"; done
# ---- 7 consolidated record
[ "$MODE" = all ] && step collect 1 "$PY" "$AX/tests_freeze/collect_s0b_record.py" --tag "$TAG"
echo "[run_s0b_integrated] tag $TAG end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat "$ST"
