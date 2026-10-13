# S1.3a analysis plan — addendum v1.1 (2026-10-08): crash fix, no change to any rule

Written by [linae] after the v1.0 pipeline crashed and before the relaunch. `ANALYSIS_PLAN.md` (v1.0, sha256 bccacb10…)
stays in force unchanged; PREREG v1.0 (bee0583c…) and SPEC v1.0 (e781e44f…) are unchanged; predictions 1–4, criteria,
tolerances, decision rule, verdict rules, learning rates, schedules, seeds and run counts are all unchanged.
The v1.0 code is kept verbatim in `linae/code_v1.0/` (its hashes are the ones in `ANALYSIS_PLAN.sha256`); the v1.1 code
hashes are in `ANALYSIS_PLAN_v1.1.sha256`.

## What happened (diagnosis, reproduced)
- The v1.0 run (`logs/linae_s13a.log`) died after 275 s with `LinAlgError: SVD did not converge` in
  `linae_core.gd_to_certificate` (`np.linalg.norm(E, ord=2, axis=(1, 2))`, the Hessian-bound lr cap of the polish).
- Reproduction with the frozen v1.0 code (`diag/diag_crash.py`, results `diag/crash_*.json`, logs
  `logs/linae_diag_crash_*.log`) on the five tasks that could have been running at that time:
  - **P-SGD, N = inf, chunk 3: run 385 diverged** (non-finite or |param| > 1e6) between step 287 000 and 288 000 of the
    constant-lr phase (lr 0.02, batch 16). Its imbalance ‖C‖_F had grown from 12.27 at init to 22.44 at step 287 000.
    `stochastic_phase` correctly set the run to NaN, but the polish then called LAPACK on the NaN row, which raised
    and killed the whole worker pool. Running the v1.0 polish on this chunk reproduces the exact exception.
  - P-SGD N = inf chunk 1 and P-Adam N = inf chunks 0, 6, 7: 0 divergences, no exception.
- So the crash was a bookkeeping bug triggered by one genuinely diverged run, not a numerical problem of the
  certified runs.

## Changes in v1.1 (code only)
1. `linae_core.safe_spec_norm` / `safe_eigvalsh`: LAPACK calls on finite rows only; non-finite rows give +inf / NaN,
   never an exception. Finite rows give bitwise the same values as v1.0.
2. Divergence is a recorded run outcome: `outcome ∈ {certified, uncertified (finite, missed the certificate),
   diverged}`, with `div_phase` (1 = stochastic phase, 2 = full-batch GD or polish) and `div_step`. Definition
   (unchanged from v1.0): a parameter non-finite or > 1e6 in absolute value at a stochastic-phase check (every 1000
   steps), or a non-finite loss/parameter or loss > 1e8 at a GD/polish check (every 50 steps).
3. Plan item 6 is now explicit in `analyze_s13a.pass_mask`: a pair passes criterion Q only if neither run diverged
   and Δ_Q is finite and ≤ ε_Q. A diverged run is kept, flagged, never dropped, and every pair containing it counts as a
   FAILURE for every criterion (kNN, Gs, CKA, projector) and in every variant (final and pre-polish). The sensitivity
   analysis excluding those pairs (plan item 6) is still reported, never used for verdicts. A metric that raises on
   finite inputs (not expected) is also a failure and is counted.
4. Fault isolation: an exception in one training task is written to `runs/<arm>/N<N>/chunk<c>.failed` and the other
   tasks continue; `train` exits non-zero at the end and the pipeline writes `S1_3a.failed`.
5. Resumable: chunk files (fixed run-id sets, trained together) are written atomically (tmp + rename) and a complete,
   valid chunk is skipped on relaunch (run ids and required keys checked). Results do not depend on execution order
   (each chunk has its own sha256-keyed RNG streams).
6. `run_all.sh` runs the unit tests first (5 new tests cover the failure mode); worker count via `LINAE_NPROC` (≤ 12).

## Learning rates: not changed
The relaunch instructions allow lowering an lr only where the prereg and this analysis plan leave it unspecified.
P-SGD (lr 0.02) and P-Adam (lr 1e-3) are silent in the prereg but were fixed in `ANALYSIS_PLAN.md` items 3–4 before any
confirmatory run, and the full-batch arms are fixed by the prereg or plan items 1–2. So no lr is changed, and diverged
runs stay in the analysis as failures (conservative). Pilot evidence (excluded, `pilot/`): P-SGD lr 0.02 gave 0/500
divergences at N = inf over 50k steps and 0/48 at N = 256 and N = 4096 over 500k steps; lr 0.05 diverged in about 20%
of runs. A per-run divergence rate of order 1e-3 to 1e-2 over 500k steps was therefore not resolvable by the pilot. The
pilot also shows the mechanism: minibatch SGD at this lr *grows* ‖C‖ (median relative ‖C‖ 2.8× after 100k steps at
N = 4096), which raises the sharpness and drives rare runs to instability. The divergence rate per (arm, N) is reported.

## Reuse of chunks completed by v1.0
Seven chunks finished under v1.0 before the crash (P-SGD N = inf chunks 0 and 2; P-Adam N = inf chunks 1–5) and are
reused. v1.0 could only finish a chunk in which no run was non-finite (otherwise the same exception would have been
raised), and v1.1 numerics for finite runs are unchanged. `diag/repro_check.py` retrains P-SGD N = inf chunk 0 with
v1.1 and compares every stored array bitwise; its result (`diag/repro_check.json`) is reported in the summary.
