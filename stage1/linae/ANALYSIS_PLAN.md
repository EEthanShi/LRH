# S1.3a analysis plan and deviations (frozen before the confirmatory runs)

Written by [linae] before any confirmatory run (pilots under `linae/pilot/` are excluded and labelled; they used
run indices >= 100000 and pilot-only data/batch keys, disjoint from the confirmatory streams).
PREREG v1.0 (sha256 bee0583c...) and SPEC v1.0 (sha256 e781e44f...) are unchanged. Predictions 1-4 are not changed.

## Implementation (as preregistered)
- x in R^16 ~ N(0, U diag(4,3,2,1.5,0.2x12) U^T); U Haar, fixed by sha256 key `S1.3a/U/v1`. d = 4. No biases.
- Probes: m = 256 draws from key `S1.3a/probes/v1`, disjoint from all training streams. Latent features H = X_S E^T, pre = "none".
- Training data of run r at size N: key `S1.3a/data/N={N}/run={r}`; Sigma_hat = X^T X / N (uncentred; true mean 0).
  N = inf uses the exact Sigma (and, for minibatch arms, fresh population samples each step).
- Init of run r: base draws from key `S1.3a/init/run={r}`; default = N(0, 1/fan_in) (E: var 1/16, W: var 1/4);
  small-init = every entry N(0, 0.01^2). Common random numbers: run r has the same base init draws in every arm and
  every N, and the same training sample in every arm.
- Loss tr((I-WE) Sigma_hat (I-WE)^T) (+ lambda(|E|^2+|W|^2) for WD, lambda = 0.05*1.5 = 0.075). float64 numpy throughout.
- Certificate: rel_gap = (F - F*)/F* with F* the closed-form optimum (sum of the 12 smallest eigenvalues of Sigma_hat; for
  WD, + sum_{i<=4}(2 lambda - lambda^2/lambda_hat_i)); rel_grad = |grad F|_F / (|Sigma_hat|_F |theta|_F). Training stops when
  rel_gap <= 1e-8 or rel_grad <= 1e-8 (checked every 50 steps). Both are recorded per run. Runs that miss the certificate
  are kept and flagged.
- Runs: 96 per (arm, N in {256, 4096, 65536}); 1000 per (arm, N = inf). Pairs: (2j, 2j+1), disjoint.
- Criteria (local implementation `linae_metrics.py`, unit-tested; the shared harness `hrhv2/` was empty at run time):
  kNN k=10: 1-s_N <= 0.05; Gs_rms <= 0.02; 1-CKA (biased) <= 0.01; |P_A-P_B|_F^2/(2d) <= 0.01 (d = 4).
- Decision: one-sided 95% Clopper-Pearson; reliable if lower > 0.9; unreliable if upper < 0.8; else inconclusive.
  (48 pairs: reliable <=> >= 47 passes; unreliable <=> <= 33 passes.)

## Deviations / choices where the prereg is silent or infeasible
1. P-GD lr = 0.05 is in the large-step regime. Pilot (excluded): from default init, ~1.5-3% of runs diverge and the
   "conserved" C drifts by a median ~50% (edge-of-stability balancing), so it is not a gradient-flow proxy.
   The preregistered arm is run exactly as specified and is the confirmatory arm for P1. A secondary arm `P-GD-flow`
   (lr = 0.002), where C is conserved to ~0.2%, is added and P1 is also evaluated on it (labelled secondary).
2. WD, TIED, small-init use full-batch GD at lr = 0.05 (the prereg defines them as modifications of P-GD). WD from default
   init diverges in ~1% of pilot runs, so a secondary arm `WD-flow` (lr = 0.002) is added.
3. P-SGD lr is not given in the prereg. lr = 0.02, the largest of {0.05, 0.02, 0.01} with zero divergences in the
   pilot (lr 0.05 diverged in ~20% of pilot runs). Minibatches: 16 i.i.d. draws with replacement from the run's own
   training set (N = inf: fresh population draws).
4. P-Adam: minibatch 16 ("common practice", Design 1), lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, no WD.
5. Stochastic-arm schedule (not in prereg): 400k steps at constant lr, then 100k steps of geometric decay to 1e-3*lr,
   then a full-batch GD polish (lr <= 0.002, capped by a Hessian bound) until the certificate. A minibatch method cannot
   reach a 1e-8 certificate by itself; the polish approximates gradient flow, which conserves C, so it does not
   rebalance. Criteria are also computed on the pre-polish endpoint (sensitivity).
6. Diverged runs are kept and flagged; any pair containing one counts as a failure of every criterion (primary).
   Sensitivity: decisions excluding those pairs. Means of Delta are over finite pairs only.
7. N = inf has 1000 inits (prereg) -> 500 disjoint pairs; decisions at N = inf use all 500 pairs (primary) and the
   first 48 pairs (reported).
8. "Largest N" = 65536 (largest finite N), consistent with P1's contrast between the largest N and the N = inf plateau.

## Verdict rules
- Per (arm, N, criterion) decision as above.
- P1 (arm P-GD): (a) for Q in {kNN, Gs, CKA} and every N in {256, 4096, 65536, inf}: part CONFIRMED if decision =
  unreliable, FALSIFIED if reliable, INCONCLUSIVE if inconclusive. (b) for each of the three Q: mean Delta_Q at N = 65536
  (48 pairs) inside the two-sided 95% t-CI of mean Delta_Q at N = inf (500 pairs) -> CONFIRMED, else FALSIFIED.
  P1 = FALSIFIED if any part is falsified; CONFIRMED if all parts confirmed; otherwise INCONCLUSIVE.
  Known design caveat (stated before running): if the two means were exactly equal, the literal test (b) passes with
  probability only ~ 2*Phi(1.96*sqrt(48/500)) - 1 ~ 0.46 per criterion, because the 48-pair mean has a much wider
  sampling error than the 500-pair CI. Non-confirmatory diagnostics are reported next to it: Welch CI of the difference
  and the paired common-random-numbers difference (same inits at N = 65536 and N = inf).
- P2 (WD, TIED, small-init) at N = 65536 for all four criteria: CONFIRMED if all 12 decisions are reliable, FALSIFIED
  if any is unreliable, otherwise INCONCLUSIVE. Also reported restricted to kNN/Gs/CKA.
- P3 projector at N = 65536 in all six primary arms: same rule with "reliable".
- P4 no prediction; report decisions for P-SGD and P-Adam at every N, and the imbalance |C| and kappa(EE^T) trajectories.
- Secondary arms (P-GD-flow, WD-flow) never replace the primary verdicts; they are reported alongside.
