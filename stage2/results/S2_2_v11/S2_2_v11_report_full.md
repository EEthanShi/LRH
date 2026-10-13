# S2.2 v1.1 — full analysis report

Rendered by role `s22v11-analyze` from the output of the single frozen analysis run. **No test was re-run and no number was recomputed here**; every value below is read from `s22v11_summary.json` (sha256 `c05262cba20219458eb132be822c7114a881fbf2de2b8ea1adc24d525fd2437c`) and the frozen eps file (`dd649d74fb3a87fd99f049a8676c2fe1751c476093522db7d9f3aba7145935f7`). The frozen code's own short report `S2_2_v11_report.md` (`c53d1bca417ee75daf6d708c0ea4b6b9e30872f11b1502db58325f455440aa2b`) is unchanged. Per prereg §14 step 9, two independent recompute reviews must follow before any claim.

- Cohort `S2_2v11_main`: {'done': 2368, 'failed': 0, 'missing': 0}. Code manifest `f2d5d28dfda1b393ac30625e6ab3e29c663416a176e88cfeed2044f7ee155c68`; guard at render time: frozen code verified (manifest f2d5d28dfda1b393ac30625e6ab3e29c663416a176e88cfeed2044f7ee155c68).
- Engineering choice: K_min = 1, head flow = {'head_opt': 'heun_adapt', 'tol': 0.0001, 'K': 16}.
- Pooled anchor arms: 25k ['MOM', 'WD', 'FLOW', 'BAL-FLOW', 'SMALL-FLOW', 'BAL-MOM', 'SMALL-MOM', 'SGD-K']; 250k ['MOM', 'WD', 'FLOW', 'BAL-FLOW', 'SMALL-FLOW', 'BAL-MOM', 'SMALL-MOM'] (GB accuracy flag: ['SGD-K_250000']).
- R-NUM numerical failures in the main cohort: {'ladder_pairs_by_cell_tag': {}, 'runs_gauge_not_spd': []}.
- Decision rule (§9): one-sided 95% Clopper–Pearson; reliable if lower > 0.9, unreliable if upper < 0.8. n = 96: x ≥ 92 reliable, x ≤ 69 unreliable; n = 32: 32 reliable, ≤ 21 unreliable.

## 0. Observations from rendering, for the two recompute reviews (not claims)

1. **A2 (secondary replication) fails** by its frozen rule in MOM_250000 (tier R: mean X_TN(10) -0.00168, 95% CI [-0.00218, -0.00117]), FLOW_250000 (tier H: mean X_TN(10) -0.00226, 95% CI [-0.00260, -0.00191]), BAL-FLOW_250000 (tier H: mean X_TN(10) -0.00212, 95% CI [-0.00244, -0.00180]), SMALL-FLOW_250000 (tier H: mean X_TN(10) -0.00195, 95% CI [-0.00237, -0.00154]). In every failing cell the CI extends below −0.002, on the side where u agrees *better* than y (X < 0); u stays reliable on T_N(10) in each. The rule's ±0.002 band is two-sided, so this is a failure of the replication expectation as written.
2. **A3 is 'anchor-sensitive' only through u^c at 250k**: uc@Y unreliable 21/96 at ε^Y_MOM = 0.19460 vs uc@Pool reliable 95/96 at ε^Pool = 0.20985 (ratio 1.078). u itself is unreliable at ε^Y at both N (0/96, 5/96) and not reliable at ε^Pool (unreliable 19/96, inconclusive 87/96). By R-A1/A3 the status order puts 'anchor-sensitive' before 'holds', so A3 does not hold as a categorical claim.
3. **A7 reading (a) fails in both arms for different reasons**: SMALL-FLOW G_s is unreliable at 25k (8/96) and reliable at 250k (93/96); BAL-FLOW G_s is unreliable at ε^Y at both N (20/96, 35/96) and anchor-sensitive at 250k (reliable 96/96 at ε^Pool, ratio 1.068). Reading (b) holds in both (B_Gs 2.74–3.89 σ_y).
4. **SGD-K 250k anchors are vacuous.** One pilot pair (S2.2v11-pilot SGD-K N250000 r022/r023; the cell also carries the GB accuracy flag) has Δ_Gs(y) = 1.172, so ε^Y_Gs = 1.172 against ε^Y_q90 = 0.2016 (1−CKA: 0.956 vs 0.0770; ε^C likewise); pilot homogeneity 14.96 σ on G_s. Every 'reliable' at ε^Y or ε^C in SGD-K 250k (e.g. u@Y G_s 32/32) is therefore uninformative. No primary endpoint uses SGD-K ladders (C5 uses δ_C only); SGD-K 250k was already excluded from the pooled anchor.
5. **C5 direction**: at the momentum-equivalent step, a plain Euler head drifts *more* than the momentum head (MOM/SGD-K median δ_C ratio 0.802 at 25k, 0.837 at 250k; MOM / extrapolated plain step = 0.761 [0.756, 0.765]; Euler slope -0.935, near the −1 expected from identity 2.2). The pre-agreed outcome is 'both causes stay'.
6. **I4 size**: FLOW I_Gs = 0.54 / 0.59 σ_y (25k / 250k) vs MOM I_Gs = 2.81 / 3.21 σ_y. Outcome (i) rests on lower bounds > 0; (ii) fails because the upper 98.75% bounds of I_Gs and I_CKA (0.59, 0.60, 0.67, 0.74 σ_y) exceed the 0.5 σ_y margin. The FLOW imbalance effect is about one fifth of momentum's on G_s.
7. **Frozen short report formatting**: its C1/C3/C4a consequence line prints the whole C3(a) dict where the prereg sentence has 'raised ρ to X'; X = BAL-MOM median ρ_end 0.284 (25k) and 0.326 (250k). No decision is affected.
8. 'h@Y' entries judge h at y's tolerance (listed by prereg §10.3); h's own decision is 'h@h' at ε^h.

## 1. Primary endpoints

| Item | Tier | Rule | Numbers | Frozen-code verdict |
|---|---|---|---|---|
| C3(a) | R | BAL-MOM: lower 97.5% of median ρ_end > 0.05 and ≥ 90% runs ρ_end ≥ 0.02, each N | 25k: median 0.2844, lower 0.2833, frac≥0.02 1.00, n 64; 250k: median 0.3264, lower 0.3248, frac≥0.02 1.00, n 64 | PASS |
| C3(b) | R | MOM: lower 97.5% of median growth ‖C_end−C₀‖/‖C₀‖ > 1, each N | 25k: median 9.34, lower 9.09, n 192; 250k: median 8.77, lower 8.46, n 192 | PASS |
| C4a | R (KA6) | MOM: upper 97.5% of median μ_C < 0.5, each N | 25k: median -0.0166, upper 0.0110, n 192; 250k: median -0.0051, upper 0.0117, n 192 | PASS |
| I3 (F1) | 1 R + 7 H | mean I(init-MOM) − mean I(init-FLOW) > 0; 8 one-sided bootstrap tests, Holm 0.05 | 8/8 rejected; smallest diff 0.0012, largest p 0.00030 (§4) | 8/8 Holm rejections |
| I4 | H, two-sided map | FLOW: (i) lower 98.75% bounds of I_Gs and I_M,Gs > 0 at both N; (ii) I_Gs, I_CKA CIs within ±0.5 σ_y at both N | 25k: I_Gs 0.54σ [0.48, 0.59], I_M,Gs 0.29σ [0.23, 0.35]; 250k: I_Gs 0.59σ [0.52, 0.67], I_M,Gs 0.46σ [0.39, 0.54] | (i) realized (within margin: no) |
| A1 | R, positive control | WD u reliable at ε^Y_WD on T_N(10), G_s, both N; RG; anchor flag | dN_k10_25000: reliable 96/96 (Pool reliable); Gs_25000: reliable 96/96 (Pool reliable); dN_k10_250000: reliable 96/96 (Pool reliable); Gs_250000: reliable 96/96 (Pool reliable) | holds |
| A3 | R, negative control | MOM u and u^c unreliable on G_s at ε^Y_MOM, both N; RG; anchor flag | 25k: u@Y unreliable 0/96, uc@Y unreliable 1/96, u@Pool unreliable 19/96, uc@Pool unreliable 47/96; 250k: u@Y unreliable 5/96, uc@Y unreliable 21/96, u@Pool inconclusive 87/96, uc@Pool reliable 95/96 | anchor-sensitive |
| A5 | E | MOM 250k G_s: Δ(y)+M at ε^Y_a replaces the v3 '96 of 96' sentence | Δ(y)+M inconclusive 78/96; u@C inconclusive; M = 1.63 σ_y | inconclusive |
| A7 BAL-FLOW | H-open | (a) u reliable on G_s at both N and not anchor-sensitive; (b) lower 95% of B_Gs > 0.5 σ_y at both N | 25k: G_s u@Y unreliable 20/96 (Pool unreliable, anchor-sens no), 1−CKA u@Y unreliable 58/96, B_Gs 3.66σ [3.52, 3.80]; 250k: G_s u@Y unreliable 35/96 (Pool reliable, anchor-sens yes), 1−CKA u@Y unreliable 67/96, B_Gs 3.14σ [2.95, 3.33] | (a) no, (b) yes |
| A7 SMALL-FLOW | H-open | (a) u reliable on G_s at both N and not anchor-sensitive; (b) lower 95% of B_Gs > 0.5 σ_y at both N | 25k: G_s u@Y unreliable 8/96 (Pool unreliable, anchor-sens no), 1−CKA u@Y unreliable 65/96, B_Gs 3.89σ [3.74, 4.04]; 250k: G_s u@Y reliable 93/96 (Pool reliable, anchor-sens no), 1−CKA u@Y reliable 95/96, B_Gs 2.74σ [2.57, 2.92] | (a) no, (b) yes |
| A7 WD clause | – | WD B_Gs 95% CI within ±0.5 σ_y at both N | 25k: B_Gs -1.23σ [-1.29, -1.18]; 250k: B_Gs -0.21σ [-0.28, -0.15] | no |
| K (Corollary 3, prospective) | T checks; R K4b,c; H K5 | K1–K3 zero violations; K4b median bound/obs at k=10 in [5, 25]; K4c cert. fraction ≤ 0.10; K5 MOM imbalance-only exchange moves ≥ 0.01 of k=10 slots | K1 0+0 viol.; K2 0 anchor viol., 0 pairs bound<obs; K3 0 viol.; K4b range 10.9–12.7; K4c max 0.0471 (WD_250000); K5 MOM 0.0521 / 0.0701 | K1 PASS, K2 PASS, K3 PASS, K4b PASS, K4c PASS, K5 PASS |

## 2. Checks (a failure would mean an error)

| Check | Result | Detail |
|---|---|---|
| C1 flow δ_C ≤ 2e-4 | PASS | FLOW_25000: max 2.6e-05, ok 192/192, viol 0; FLOW_250000: max 2.74e-05, ok 192/192, viol 0; BAL-FLOW_25000: max 2.55e-05, ok 192/192, viol 0; BAL-FLOW_250000: max 2.75e-05, ok 192/192, viol 0; SMALL-FLOW_25000: max 2.51e-05, ok 192/192, viol 0; SMALL-FLOW_250000: max 2.66e-05, ok 192/192, viol 0 |
| C2 Euler identity ≤ 1e-8 | PASS | SGD-K_25000: max 9.03e-16, ok 64/64; SGD-K_250000: max 9.41e-16, ok 64/64; KS-4_250000: max 1.78e-15, ok 16/16; KS-16_250000: max 3.37e-15, ok 16/16 |
| C4 flow check median μ_C ≥ 0.99 | PASS | FLOW 25k 0.9999995; FLOW 250k 0.9999995 |
| I1 Lemma N1 (end) | PASS | runs 2368, violations 0, vacuous 31 |
| I2 Lemma N1 (flow arms) | PASS | runs 1152, violations 0, vacuous 0 |
| K1, K2, K3 | PASS | see §7 |

## 3. C-block (Remark rem:optimizers)

**C4b [E]: `frac_own_closer`, two-sided 95% CP over main runs.** Reading by the frozen code: statistic insensitive: FLOW's CP interval contains 0.5 -> the v3 gauge-memory sentence loses its evidence.

| Cell | x/n | p̂ | 95% CP | mean d(own) | mean d(other) |
|---|---|---|---|---|---|
| FLOW_25000 | 95/192 | 0.495 | [0.422, 0.568] | 6.361 | 6.480 |
| FLOW_250000 | 102/192 | 0.531 | [0.458, 0.603] | 6.202 | 6.189 |
| MOM_25000 | 101/192 | 0.526 | [0.453, 0.598] | 6.497 | 6.522 |
| MOM_250000 | 95/192 | 0.495 | [0.422, 0.568] | 6.670 | 6.482 |
| WD_25000 | 108/192 | 0.562 | [0.489, 0.634] | 6.545 | 6.748 |
| WD_250000 | 92/192 | 0.479 | [0.407, 0.552] | 6.434 | 6.369 |
| BAL-FLOW_25000 | 95/192 | 0.495 | [0.422, 0.568] | 0.896 | 0.899 |
| BAL-FLOW_250000 | 101/192 | 0.526 | [0.453, 0.598] | 1.162 | 1.170 |
| SMALL-FLOW_25000 | 95/192 | 0.495 | [0.422, 0.568] | 6.558 | 6.446 |
| SMALL-FLOW_250000 | 100/192 | 0.521 | [0.448, 0.593] | 6.532 | 6.596 |

**C5 [E]: drift δ_C, medians with 95% bootstrap CI.**

| Cell | median δ_C | 95% CI | n |
|---|---|---|---|
| MOM_25000 | 0.2756 | [0.2737, 0.2766] | 192 |
| MOM_250000 | 0.3143 | [0.3136, 0.3152] | 192 |
| SGD-K_25000 | 0.3435 | [0.3418, 0.3461] | 64 |
| SGD-K_250000 | 0.3755 | [0.3736, 0.3776] | 64 |
| FLOW_25000 | 2.36e-05 | [2.34e-05, 2.37e-05] | 192 |
| FLOW_250000 | 2.54e-05 | [2.52e-05, 2.56e-05] | 192 |
| KS-4_250000 | 0.1370 | [0.1364, 0.1385] | 16 |
| KS-16_250000 | 0.0281 | [0.0278, 0.0285] | 16 |

| Ratio | value | 95% CI |
|---|---|---|
| MOM/SGD-K_25000 | 0.802 | [0.794, 0.808] |
| MOM/SGD-K_250000 | 0.837 | [0.833, 0.842] |
| MOM/FLOW_25000 | 11697.859 | [11584.549, 11783.857] |
| MOM/FLOW_250000 | 12353.663 | [12259.282, 12466.564] |
| SGD-K/FLOW_25000 | 14578.197 | [14459.474, 14735.396] |
| SGD-K/FLOW_250000 | 14759.184 | [14625.236, 14902.906] |

Extrapolation at 250k from SGD-K, KS-4, KS-16 (K = [1.0, 4.0, 16.0]): log-log slope -0.935 [-0.939, -0.929]; predicted plain-step (K = 1) drift 0.4133; MOM / plain = 0.761 [0.756, 0.765]; momentum named main cause (ratio > 3 with CI excluding 3): no.

**C6 [E]: MOM-BF16 vs MOM (fp64) at 250k.**

| Quantity | median bf16 | median fp64 | relative diff | 95% CI of diff | sentence triggered |
|---|---|---|---|---|---|
| rho_end | 0.3121 | 0.3119 | 0.0006 | [-0.0013, 0.0024] | no |
| delta_C | 0.3150 | 0.3143 | 0.0022 | [-0.0029, 0.0027] | no |

## 4. I-block (balanced map vs imbalance channel)

**I2: median D (estimate).** BAL-FLOW_25000 1.0000; BAL-FLOW_250000 1.0000; SMALL-FLOW_25000 1.0008; SMALL-FLOW_250000 1.0010; FLOW_25000 1.0773; FLOW_250000 1.1001; MOM_25000 1.3843; MOM_250000 1.6337.

**I3 (family F1), mean I(init-MOM) − mean I(init-FLOW), 10,000 resamples, Holm α = 0.05.**

| Test | tier | diff (raw) | 95% CI | p (one-sided) | Holm threshold | reject | n_MOM / n_FLOW pairs |
|---|---|---|---|---|---|---|---|
| BAL · Gs · 25000 | H | 0.00970 | [0.00922, 0.01020] | 0.00010 | 0.00625 | yes | 32 / 96 |
| BAL · Gs · 250000 | H | 0.01107 | [0.01036, 0.01182] | 0.00010 | 0.00714 | yes | 32 / 96 |
| BAL · dCKA · 25000 | H | 0.00460 | [0.00417, 0.00504] | 0.00010 | 0.00833 | yes | 32 / 96 |
| BAL · dCKA · 250000 | H | 0.00433 | [0.00390, 0.00477] | 0.00010 | 0.01000 | yes | 32 / 96 |
| SMALL · Gs · 25000 | H | 0.00632 | [0.00556, 0.00710] | 0.00010 | 0.01250 | yes | 32 / 96 |
| SMALL · Gs · 250000 | R | 0.00921 | [0.00821, 0.01026] | 0.00010 | 0.01667 | yes | 32 / 96 |
| SMALL · dCKA · 25000 | H | 0.00119 | [0.00054, 0.00184] | 0.00030 | 0.05000 | yes | 32 / 96 |
| SMALL · dCKA · 250000 | H | 0.00362 | [0.00299, 0.00424] | 0.00010 | 0.02500 | yes | 32 / 96 |

**I4 (FLOW), simultaneous 0.9875 percentile bootstrap CIs, margin δ_eq = 0.5 σ_y.** Outcome: **(i) realized**.

| N, Q, part | mean raw | CI raw | mean σ_y | CI σ_y | σ_y | n |
|---|---|---|---|---|---|---|
| 25000 · Gs · I | 0.00203 | [0.00182, 0.00224] | 0.538 | [0.484, 0.594] | 0.00377 | 96 |
| 25000 · Gs · I_M | 0.00109 | [0.00088, 0.00130] | 0.289 | [0.234, 0.346] | 0.00377 | 96 |
| 25000 · dCKA · I | 0.00206 | [0.00183, 0.00229] | 0.542 | [0.482, 0.603] | 0.00379 | 96 |
| 25000 · dCKA · I_M | 0.00147 | [0.00123, 0.00170] | 0.387 | [0.326, 0.450] | 0.00379 | 96 |
| 250000 · Gs · I | 0.00240 | [0.00212, 0.00271] | 0.592 | [0.521, 0.667] | 0.00406 | 96 |
| 250000 · Gs · I_M | 0.00188 | [0.00160, 0.00219] | 0.463 | [0.394, 0.538] | 0.00406 | 96 |
| 250000 · dCKA · I | 0.00206 | [0.00184, 0.00229] | 0.668 | [0.597, 0.743] | 0.00308 | 96 |
| 250000 · dCKA · I_M | 0.00187 | [0.00165, 0.00211] | 0.606 | [0.534, 0.685] | 0.00308 | 96 |

For comparison (A4 means, σ_y units of each cell): MOM 25k I_Gs 2.81σ, I_M,Gs 0.44σ; MOM 250k I_Gs 3.21σ, I_M,Gs 0.76σ; FLOW 25k I_Gs 0.54σ, I_M,Gs 0.29σ; FLOW 250k I_Gs 0.59σ, I_M,Gs 0.46σ.

**I5 [E]: DiD [I(FLOW) − I(BAL-FLOW)] − [I(MOM) − I(BAL-MOM)].** I_Gs_25000 0.00137 [0.00075, 0.00200]; I_Gs_250000 0.00247 [0.00160, 0.00336]; I_dCKA_25000 0.00183 [0.00127, 0.00238]; I_dCKA_250000 0.00232 [0.00179, 0.00286]; rho_end_25000 0.03373 (point); rho_end_250000 0.04001 (point).

**I6 [E, unadjusted]: Spearman(log D, ‖C‖_op/s).** FLOW_25000 ρ = 0.894, one-sided p = 1.76e-68, n = 192 (x = C0_op/s); FLOW_250000 ρ = 0.730, one-sided p = 1.62e-33, n = 192 (x = C0_op/s); MOM_25000 ρ = 0.330, one-sided p = 1.44e-06, n = 192 (x = C_end_op/s); MOM_250000 ρ = 0.316, one-sided p = 3.9e-06, n = 192 (x = C_end_op/s).

## 5. A-block: secondary replications and estimates

**A2 [R for MOM, WD; H for flow arms]: wherever y is reliable, u not unreliable on T_N(10) and the 95% CI of mean X_TN(10) within ±0.002.** pass_R = no, pass_H = no.

| Cell | tier | y reliable | u@Y | mean X_TN(10) | 95% CI | pass |
|---|---|---|---|---|---|---|
| MOM_25000 | R | yes | reliable | -6.61e-06 | [-0.00026, 0.00025] | PASS |
| MOM_250000 | R | yes | reliable | -0.00168 | [-0.00218, -0.00117] | FAIL |
| WD_25000 | R | yes | reliable | -0.00030 | [-0.00047, -0.00012] | PASS |
| WD_250000 | R | yes | reliable | -0.00085 | [-0.00116, -0.00055] | PASS |
| FLOW_25000 | H | yes | reliable | -0.00021 | [-0.00045, 0.00003] | PASS |
| FLOW_250000 | H | yes | reliable | -0.00226 | [-0.00260, -0.00191] | FAIL |
| BAL-FLOW_25000 | H | yes | reliable | -0.00028 | [-0.00052, -0.00004] | PASS |
| BAL-FLOW_250000 | H | yes | reliable | -0.00212 | [-0.00244, -0.00180] | FAIL |
| SMALL-FLOW_25000 | H | yes | reliable | -0.00061 | [-0.00090, -0.00032] | PASS |
| SMALL-FLOW_250000 | H | yes | reliable | -0.00195 | [-0.00237, -0.00154] | FAIL |

**A8 [R]:** MOM X_Gs(250k)/X_Gs(25k) = 0.826 [0.786, 0.868]; rule [0.5, 1.5]: PASS.

**A9 [R]:** MOM Δ_E(u) < Δ_E(y) in 25k 1.000 of 96; 250k 1.000 of 96 pairs; rule ≥ 0.9: PASS.

**A10 [E]: mean s_N(10)(h) by arm.** 25k: MOM 0.1575, WD 0.1596, FLOW 0.1617, BAL-FLOW 0.1624, SMALL-FLOW 0.1586, BAL-MOM 0.1570, SMALL-MOM 0.1520, SGD-K 0.1611 (max |diff| 0.0104). 250k: MOM 0.2960, WD 0.3007, FLOW 0.2965, BAL-FLOW 0.2957, SMALL-FLOW 0.2906, BAL-MOM 0.2971, SMALL-MOM 0.2887, SGD-K 0.2999 (max |diff| 0.0120).

**A5 [E] (all ladder cells): u at ε^C and Δ(y)+M at ε^Y_a on G_s.**

| Cell | u@C | Δ(y)+M @Y | x/n | M (σ_y) |
|---|---|---|---|---|
| MOM_25000 | inconclusive | inconclusive | 82/96 | 1.23 |
| MOM_250000 | inconclusive | inconclusive | 78/96 | 1.63 |
| WD_25000 | reliable | reliable | 92/96 | 0.28 |
| WD_250000 | reliable | reliable | 96/96 | 0.07 |
| FLOW_25000 | inconclusive | inconclusive | 74/96 | 1.50 |
| FLOW_250000 | inconclusive | inconclusive | 83/96 | 1.71 |
| BAL-FLOW_25000 | inconclusive | inconclusive | 84/96 | 0.82 |
| BAL-FLOW_250000 | inconclusive | inconclusive | 80/96 | 1.19 |
| SMALL-FLOW_25000 | inconclusive | inconclusive | 78/96 | 1.10 |
| SMALL-FLOW_250000 | reliable | reliable | 96/96 | 1.37 |
| BAL-MOM_25000 | inconclusive (RG) | inconclusive (RG) | 25/32 | 0.79 |
| BAL-MOM_250000 | inconclusive | inconclusive | 22/32 | 1.78 |
| SMALL-MOM_25000 | inconclusive | inconclusive (RG) | 24/32 | 1.60 |
| SMALL-MOM_250000 | inconclusive (RG) | inconclusive (RG) | 19/32 | 1.46 |
| SGD-K_25000 | inconclusive (RG) | inconclusive (RG) | 26/32 | 0.80 |
| SGD-K_250000 | reliable | reliable | 32/32 | 0.52 |

**A6 [E]: u at its own anchor ε^O; ratio ε^O/ε^Y.**

| Cell | T_N(10) u@O (ε^O/ε^Y) | G_s u@O (ε^O/ε^Y) | 1-CKA u@O (ε^O/ε^Y) |
|---|---|---|---|
| MOM_25000 | reliable (1.000) | reliable (1.102) | reliable (1.095) |
| MOM_250000 | reliable (0.997) | reliable (1.127) | reliable (1.150) |
| WD_25000 | reliable (1.000) | reliable (0.982) | reliable (0.951) |
| WD_250000 | reliable (1.003) | reliable (0.996) | reliable (1.025) |
| FLOW_25000 | reliable (0.999) | reliable (1.086) | reliable (1.089) |
| FLOW_250000 | reliable (0.997) | reliable (1.071) | reliable (1.116) |
| BAL-FLOW_25000 | reliable (0.998) | reliable (1.066) | reliable (1.064) |
| BAL-FLOW_250000 | reliable (0.997) | reliable (1.058) | reliable (1.081) |
| SMALL-FLOW_25000 | reliable (1.000) | reliable (1.085) | reliable (1.068) |
| SMALL-FLOW_250000 | reliable (0.994) | reliable (1.049) | reliable (1.095) |
| BAL-MOM_25000 | inconclusive (0.999) | inconclusive (1.093) | reliable (1.119) |
| BAL-MOM_250000 | reliable (1.000) | inconclusive (1.100) | reliable (1.154) |
| SMALL-MOM_25000 | inconclusive (0.999) | inconclusive (1.061) | inconclusive (1.032) |
| SMALL-MOM_250000 | inconclusive (1.001) | inconclusive (1.060) | inconclusive (1.147) |
| SGD-K_25000 | reliable (1.001) | inconclusive (1.117) | inconclusive (1.138) |
| SGD-K_250000 | inconclusive (0.823) | reliable (0.191) | reliable (0.095) |

## 6. A4 decompositions (preregistered endpoints)

Means in σ_Q(y, cell) units with 95% bootstrap CIs; raw values, medians and share > 0 for all 8 R_cf rungs are in `render/a4_parts.csv`. X = Δ(u) − Δ(y) = S + M = B + I; S_bal + M_bal = B; I_S + I_M = I.

Replication expectations, MOM 250k G_s: S/X = 0.710 (rule [0.5, 0.9]: yes); M CI excludes 0: yes; I CI excludes 0: yes; B CI excludes 0: yes.

**G_s**

| Cell | σ_y (raw) | X | S | M | B | I | S_bal | M_bal | I_S | I_M | S/X | I/X |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOM_25000 | 0.00369 | 6.32 [6.13, 6.51] | 5.09 [4.99, 5.19] | 1.23 [1.08, 1.37] | 3.51 [3.38, 3.64] | 2.81 [2.71, 2.90] | 2.73 [2.66, 2.80] | 0.78 [0.68, 0.89] | 2.36 [2.32, 2.40] | 0.44 [0.36, 0.53] | 0.81 | 0.44 |
| MOM_250000 | 0.00343 | 5.61 [5.38, 5.84] | 3.98 [3.79, 4.18] | 1.63 [1.46, 1.80] | 2.41 [2.26, 2.55] | 3.21 [3.08, 3.33] | 1.53 [1.41, 1.65] | 0.87 [0.78, 0.97] | 2.45 [2.37, 2.53] | 0.76 [0.65, 0.87] | 0.71 | 0.57 |
| WD_25000 | 0.00477 | -0.91 [-0.96, -0.85] | -1.19 [-1.22, -1.16] | 0.28 [0.23, 0.33] | -1.23 [-1.29, -1.18] | 0.33 [0.32, 0.34] | -1.52 [-1.55, -1.49] | 0.29 [0.24, 0.33] | 0.33 [0.32, 0.33] | -0.00188 [-0.00765, 0.00394] | 1.31 | -0.36 |
| WD_250000 | 0.00394 | -0.04 [-0.12, 0.03] | -0.11 [-0.17, -0.06] | 0.07 [0.02, 0.12] | -0.21 [-0.28, -0.15] | 0.17 [0.15, 0.18] | -0.28 [-0.32, -0.23] | 0.07 [0.02, 0.11] | 0.17 [0.15, 0.18] | 0.00122 [-0.00969, 0.01] | 2.51 | -3.71 |
| FLOW_25000 | 0.00377 | 5.48 [5.27, 5.71] | 3.99 [3.89, 4.08] | 1.50 [1.32, 1.69] | 4.95 [4.75, 5.15] | 0.54 [0.50, 0.58] | 3.74 [3.65, 3.83] | 1.21 [1.05, 1.38] | 0.25 [0.24, 0.25] | 0.29 [0.25, 0.33] | 0.73 | 0.10 |
| FLOW_250000 | 0.00406 | 3.45 [3.28, 3.63] | 1.74 [1.62, 1.86] | 1.71 [1.57, 1.86] | 2.86 [2.71, 3.02] | 0.59 [0.53, 0.65] | 1.61 [1.49, 1.73] | 1.25 [1.13, 1.37] | 0.13 [0.12, 0.13] | 0.46 [0.41, 0.52] | 0.50 | 0.17 |
| BAL-FLOW_25000 | 0.00443 | 3.66 [3.52, 3.80] | 2.85 [2.78, 2.92] | 0.82 [0.71, 0.93] | 3.66 [3.52, 3.80] | -0.000163 [-0.000171, -0.000155] | 2.85 [2.78, 2.91] | 0.82 [0.71, 0.93] | -0.000177 [-0.00018, -0.000174] | 1.42e-05 [6.9e-06, 2.19e-05] | 0.78 | -4.45e-05 |
| BAL-FLOW_250000 | 0.00341 | 3.14 [2.94, 3.33] | 1.95 [1.83, 2.08] | 1.19 [1.04, 1.34] | 3.14 [2.95, 3.33] | -8.47e-05 [-9.9e-05, -7e-05] | 1.95 [1.83, 2.07] | 1.19 [1.04, 1.34] | -9.23e-05 [-9.71e-05, -8.74e-05] | 7.58e-06 [-4.73e-06, 2.04e-05] | 0.62 | -2.7e-05 |
| SMALL-FLOW_25000 | 0.00441 | 3.89 [3.74, 4.05] | 2.80 [2.72, 2.87] | 1.10 [0.97, 1.23] | 3.89 [3.74, 4.04] | 0.00374 [0.00337, 0.00414] | 2.79 [2.72, 2.87] | 1.10 [0.96, 1.23] | 0.00261 [0.00257, 0.00265] | 0.00113 [0.000758, 0.00153] | 0.72 | 0.000961 |
| SMALL-FLOW_250000 | 0.00373 | 2.75 [2.58, 2.92] | 1.38 [1.27, 1.49] | 1.37 [1.22, 1.52] | 2.74 [2.57, 2.92] | 0.00386 [0.00318, 0.00455] | 1.37 [1.26, 1.48] | 1.37 [1.22, 1.52] | 0.00169 [0.00163, 0.00175] | 0.00217 [0.00149, 0.00289] | 0.50 | 0.0014 |
| BAL-MOM_25000 | 0.00419 | 5.07 [4.78, 5.36] | 4.28 [4.11, 4.45] | 0.79 [0.57, 1.01] | 2.76 [2.55, 2.96] | 2.31 [2.20, 2.43] | 2.26 [2.15, 2.38] | 0.49 [0.33, 0.65] | 2.01 [1.95, 2.08] | 0.30 [0.20, 0.40] | 0.84 | 0.46 |
| BAL-MOM_250000 | 0.00287 | 6.56 [6.06, 7.09] | 4.79 [4.45, 5.12] | 1.78 [1.37, 2.20] | 2.70 [2.40, 3.01] | 3.86 [3.62, 4.12] | 1.88 [1.68, 2.08] | 0.82 [0.59, 1.06] | 2.91 [2.76, 3.06] | 0.96 [0.74, 1.18] | 0.73 | 0.59 |
| SMALL-MOM_25000 | 0.00297 | 5.61 [5.10, 6.14] | 4.01 [3.81, 4.22] | 1.60 [1.17, 2.04] | 3.48 [3.07, 3.90] | 2.13 [1.88, 2.40] | 2.47 [2.30, 2.63] | 1.01 [0.68, 1.38] | 1.55 [1.47, 1.62] | 0.59 [0.34, 0.85] | 0.71 | 0.38 |
| SMALL-MOM_250000 | 0.00532 | 2.99 [2.68, 3.28] | 1.53 [1.36, 1.69] | 1.46 [1.19, 1.74] | 1.25 [1.05, 1.47] | 1.73 [1.55, 1.93] | 0.58 [0.45, 0.71] | 0.67 [0.50, 0.85] | 0.95 [0.90, 1.00] | 0.78 [0.60, 0.99] | 0.51 | 0.58 |
| SGD-K_25000 | 0.00424 | 7.31 [6.97, 7.66] | 6.51 [6.26, 6.76] | 0.80 [0.57, 1.02] | 2.16 [1.96, 2.38] | 5.15 [4.94, 5.37] | 1.69 [1.58, 1.80] | 0.48 [0.31, 0.65] | 4.82 [4.68, 4.97] | 0.33 [0.17, 0.48] | 0.89 | 0.70 |
| SGD-K_250000 | 0.00381 | 4.57 [4.20, 4.98] | 4.05 [3.74, 4.38] | 0.52 [0.30, 0.76] | 0.89 [0.68, 1.15] | 3.68 [3.46, 3.93] | 0.47 [0.32, 0.60] | 0.43 [0.27, 0.63] | 3.59 [3.39, 3.80] | 0.09 [-0.06, 0.25] | 0.89 | 0.80 |

**1-CKA**

| Cell | σ_y (raw) | X | S | M | B | I | S_bal | M_bal | I_S | I_M | S/X | I/X |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOM_25000 | 0.00351 | 3.54 [3.37, 3.72] | 2.28 [2.18, 2.39] | 1.26 [1.12, 1.39] | 2.17 [2.04, 2.29] | 1.38 [1.29, 1.46] | 1.33 [1.26, 1.40] | 0.83 [0.74, 0.94] | 0.95 [0.92, 0.99] | 0.43 [0.35, 0.50] | 0.64 | 0.39 |
| MOM_250000 | 0.00291 | 3.89 [3.74, 4.05] | 2.45 [2.34, 2.56] | 1.45 [1.31, 1.58] | 2.50 [2.40, 2.60] | 1.39 [1.31, 1.48] | 1.66 [1.59, 1.74] | 0.84 [0.76, 0.92] | 0.79 [0.74, 0.83] | 0.61 [0.53, 0.69] | 0.63 | 0.36 |
| WD_25000 | 0.00386 | -1.65 [-1.71, -1.58] | -1.88 [-1.91, -1.85] | 0.24 [0.19, 0.28] | -2.11 [-2.18, -2.05] | 0.47 [0.46, 0.48] | -2.35 [-2.38, -2.31] | 0.23 [0.19, 0.28] | 0.47 [0.46, 0.47] | 0.00111 [-0.00493, 0.00711] | 1.14 | -0.28 |
| WD_250000 | 0.00248 | 1.02 [0.96, 1.08] | 1.04 [0.99, 1.09] | -0.02 [-0.07, 0.03] | 0.71 [0.65, 0.77] | 0.31 [0.30, 0.32] | 0.72 [0.68, 0.76] | -0.01 [-0.06, 0.03] | 0.32 [0.31, 0.33] | -0.00642 [-0.02, 0.00394] | 1.02 | 0.30 |
| FLOW_25000 | 0.00379 | 3.32 [3.13, 3.51] | 1.64 [1.56, 1.72] | 1.68 [1.51, 1.85] | 2.77 [2.61, 2.95] | 0.54 [0.49, 0.59] | 1.48 [1.40, 1.56] | 1.29 [1.15, 1.44] | 0.16 [0.15, 0.16] | 0.39 [0.34, 0.43] | 0.49 | 0.16 |
| FLOW_250000 | 0.00308 | 3.34 [3.17, 3.53] | 1.34 [1.25, 1.43] | 2.00 [1.85, 2.16] | 2.68 [2.53, 2.83] | 0.67 [0.61, 0.73] | 1.28 [1.19, 1.36] | 1.40 [1.28, 1.52] | 0.06 [0.06, 0.06] | 0.61 [0.55, 0.66] | 0.40 | 0.20 |
| BAL-FLOW_25000 | 0.00426 | 1.96 [1.83, 2.08] | 1.18 [1.12, 1.24] | 0.78 [0.68, 0.88] | 1.96 [1.84, 2.08] | -0.000268 [-0.000276, -0.000261] | 1.18 [1.12, 1.24] | 0.78 [0.68, 0.88] | -0.000277 [-0.000281, -0.000272] | 8.23e-06 [1.87e-06, 1.48e-05] | 0.60 | -0.000137 |
| BAL-FLOW_250000 | 0.00272 | 2.76 [2.61, 2.92] | 1.59 [1.49, 1.69] | 1.18 [1.06, 1.29] | 2.76 [2.62, 2.91] | -0.000138 [-0.000149, -0.000128] | 1.59 [1.49, 1.69] | 1.18 [1.06, 1.29] | -0.000142 [-0.000147, -0.000137] | 3.26e-06 [-4.89e-06, 1.15e-05] | 0.57 | -5e-05 |
| SMALL-FLOW_25000 | 0.00415 | 1.65 [1.51, 1.80] | 0.54 [0.48, 0.61] | 1.11 [0.98, 1.24] | 1.65 [1.50, 1.80] | 0.00272 [0.00231, 0.00314] | 0.54 [0.47, 0.60] | 1.11 [0.98, 1.24] | 0.00109 [0.00105, 0.00112] | 0.00164 [0.00124, 0.00206] | 0.33 | 0.00165 |
| SMALL-FLOW_250000 | 0.00270 | 2.28 [2.13, 2.44] | 0.80 [0.71, 0.88] | 1.49 [1.35, 1.63] | 2.28 [2.13, 2.44] | 0.0032 [0.00261, 0.00382] | 0.80 [0.71, 0.88] | 1.48 [1.34, 1.63] | 0.000343 [0.0003, 0.000386] | 0.00286 [0.00227, 0.00348] | 0.35 | 0.0014 |
| BAL-MOM_25000 | 0.00399 | 2.75 [2.49, 3.00] | 1.98 [1.82, 2.14] | 0.77 [0.54, 0.99] | 1.59 [1.42, 1.76] | 1.15 [1.04, 1.26] | 1.14 [1.03, 1.24] | 0.46 [0.30, 0.62] | 0.84 [0.79, 0.90] | 0.31 [0.23, 0.40] | 0.72 | 0.42 |
| BAL-MOM_250000 | 0.00252 | 4.50 [4.14, 4.85] | 2.96 [2.72, 3.21] | 1.54 [1.24, 1.83] | 2.79 [2.55, 3.02] | 1.72 [1.54, 1.89] | 2.00 [1.85, 2.16] | 0.79 [0.60, 0.97] | 0.96 [0.86, 1.06] | 0.75 [0.60, 0.92] | 0.66 | 0.38 |
| SMALL-MOM_25000 | 0.00388 | 1.18 [0.88, 1.52] | 0.14 [0.000893, 0.27] | 1.05 [0.79, 1.30] | 0.87 [0.64, 1.11] | 0.31 [0.14, 0.48] | 0.12 [0.02, 0.23] | 0.75 [0.57, 0.95] | 0.02 [-0.04, 0.07] | 0.29 [0.14, 0.45] | 0.11 | 0.26 |
| SMALL-MOM_250000 | 0.00385 | 2.44 [2.18, 2.69] | 0.95 [0.78, 1.13] | 1.48 [1.27, 1.69] | 1.49 [1.32, 1.67] | 0.94 [0.78, 1.10] | 0.74 [0.61, 0.88] | 0.75 [0.62, 0.89] | 0.21 [0.16, 0.27] | 0.73 [0.57, 0.89] | 0.39 | 0.39 |
| SGD-K_25000 | 0.00411 | 4.19 [3.97, 4.43] | 3.48 [3.32, 3.64] | 0.71 [0.51, 0.90] | 0.99 [0.83, 1.16] | 3.20 [3.05, 3.35] | 0.55 [0.47, 0.62] | 0.45 [0.30, 0.59] | 2.93 [2.84, 3.03] | 0.26 [0.12, 0.40] | 0.83 | 0.76 |
| SGD-K_250000 | 0.00273 | 3.78 [3.54, 4.03] | 3.02 [2.84, 3.23] | 0.76 [0.58, 0.95] | 1.65 [1.48, 1.83] | 2.14 [1.99, 2.29] | 1.19 [1.09, 1.30] | 0.45 [0.33, 0.60] | 1.83 [1.71, 1.96] | 0.30 [0.19, 0.42] | 0.80 | 0.56 |

**X on T_N(10), T_E, T_O (σ_y units, 95% CI)**

| Cell | X T_N(10) | X T_E | X T_O |
|---|---|---|---|
| MOM_25000 | -0.00241 [-0.09, 0.09] | -5.44 [-5.62, -5.27] | -6.84 [-7.05, -6.64] |
| MOM_250000 | -0.31 [-0.40, -0.22] | -5.73 [-5.94, -5.52] | -4.92 [-5.14, -4.71] |
| WD_25000 | -0.10 [-0.15, -0.04] | -3.23 [-3.36, -3.11] | -3.84 [-3.99, -3.69] |
| WD_250000 | -0.15 [-0.20, -0.10] | -0.38 [-0.49, -0.27] | -0.07 [-0.19, 0.05] |
| FLOW_25000 | -0.08 [-0.17, 0.01] | -5.50 [-5.69, -5.31] | -6.77 [-7.01, -6.53] |
| FLOW_250000 | -0.43 [-0.50, -0.37] | -4.75 [-4.96, -4.54] | -4.17 [-4.40, -3.93] |
| BAL-FLOW_25000 | -0.11 [-0.20, -0.02] | -4.73 [-4.87, -4.60] | -5.86 [-6.03, -5.70] |
| BAL-FLOW_250000 | -0.40 [-0.46, -0.34] | -5.06 [-5.28, -4.84] | -4.56 [-4.78, -4.34] |
| SMALL-FLOW_25000 | -0.22 [-0.33, -0.11] | -6.77 [-6.92, -6.61] | -8.51 [-8.69, -8.31] |
| SMALL-FLOW_250000 | -0.34 [-0.41, -0.27] | -5.80 [-5.98, -5.61] | -5.50 [-5.69, -5.30] |
| BAL-MOM_25000 | -0.05 [-0.23, 0.13] | -5.69 [-6.03, -5.37] | -7.16 [-7.56, -6.77] |
| BAL-MOM_250000 | -0.26 [-0.43, -0.08] | -6.68 [-7.03, -6.33] | -5.51 [-5.84, -5.18] |
| SMALL-MOM_25000 | -0.02 [-0.13, 0.09] | -10.17 [-10.81, -9.56] | -11.34 [-12.04, -10.66] |
| SMALL-MOM_250000 | -0.07 [-0.17, 0.02] | -4.24 [-4.61, -3.88] | -4.00 [-4.39, -3.61] |
| SGD-K_25000 | -0.04 [-0.20, 0.12] | -5.67 [-5.99, -5.32] | -6.42 [-6.72, -6.10] |
| SGD-K_250000 | -0.06 [-0.24, 0.12] | -6.21 [-6.56, -5.86] | -5.82 [-6.16, -5.45] |

## 7. K-block (Corollary cor:gauge-neighbors, prospective)

| Cell | pairs ok | κ_AB med | κ^bal med | D med | κ/κ^bal med | cert. frac k=10 | bound/obs k=10 | obs pure k=10 | obs imb k=10 [95% CI] | K6 excess k=10 | K4a band kept |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOM_25000 | 96/96 | 1.407 | 1.345 | 1.384 | 1.053 | 0.0004 | 11.09 | 0.0585 | 0.0521 [0.0516, 0.0526] | -0.00007 | 4398/4404 |
| MOM_250000 | 96/96 | 1.381 | 1.285 | 1.634 | 1.078 | 0.0011 | 11.37 | 0.0513 | 0.0701 [0.0696, 0.0707] | 0.00007 | 8390/8401 |
| WD_25000 | 96/96 | 1.179 | 1.174 | 1.074 | 1.000 | 0.0221 | 12.73 | 0.0266 | 0.0108 [0.0106, 0.0109] | -0.00002 | 41345/41377 |
| WD_250000 | 96/96 | 1.145 | 1.140 | 1.122 | 1.004 | 0.0471 | 12.14 | 0.0219 | 0.0163 [0.0161, 0.0165] | -0.00011 | 57802/57836 |
| FLOW_25000 | 96/96 | 1.520 | 1.443 | 1.077 | 1.049 | 0.0002 | 10.89 | 0.0674 | 0.0141 [0.0138, 0.0143] | -0.00005 | 2202/2203 |
| FLOW_250000 | 96/96 | 1.458 | 1.374 | 1.100 | 1.059 | 0.0006 | 11.60 | 0.0556 | 0.0158 [0.0155, 0.0160] | -0.00006 | 5040/5047 |
| BAL-FLOW_25000 | 96/96 | 1.367 | 1.367 | 1.000 | 1.000 | 0.0007 | 11.61 | 0.0533 | 8.9e-06 [5.85e-06, 1.22e-05] | -0.00011 | 5964/5971 |
| BAL-FLOW_250000 | 96/96 | 1.319 | 1.319 | 1.000 | 1.000 | 0.0025 | 12.12 | 0.0433 | 7.88e-06 [5.09e-06, 1.12e-05] | -0.00002 | 13104/13120 |
| SMALL-FLOW_25000 | 96/96 | 1.446 | 1.446 | 1.001 | 1.000 | 0.0003 | 11.30 | 0.0599 | 0.0002 [0.0001, 0.0002] | -0.00001 | 3924/3929 |
| SMALL-FLOW_250000 | 96/96 | 1.357 | 1.357 | 1.001 | 1.000 | 0.0022 | 12.11 | 0.0455 | 0.0002 [0.0002, 0.0002] | -0.00004 | 11259/11272 |
| BAL-MOM_25000 | 32/32 | 1.415 | 1.327 | 1.356 | 1.064 | 0.0004 | 11.21 | 0.0582 | 0.0500 [0.0494, 0.0506] | 0.00003 | 1441/1444 |
| BAL-MOM_250000 | 32/32 | 1.387 | 1.278 | 1.621 | 1.083 | 0.0011 | 11.47 | 0.0509 | 0.0702 [0.0693, 0.0711] | -0.00004 | 2653/2657 |
| SMALL-MOM_25000 | 32/32 | 1.423 | 1.374 | 1.505 | 1.032 | 0.0004 | 11.17 | 0.0580 | 0.0606 [0.0595, 0.0617] | 0.00009 | 1511/1515 |
| SMALL-MOM_250000 | 32/32 | 1.443 | 1.315 | 1.710 | 1.103 | 0.0007 | 11.44 | 0.0541 | 0.0728 [0.0717, 0.0740] | 0.00020 | 2153/2155 |
| SGD-K_25000 | 32/32 | 1.356 | 1.289 | 1.788 | 1.049 | 0.0014 | 12.04 | 0.0483 | 0.0876 [0.0869, 0.0883] | 0.00004 | 2908/2911 |
| SGD-K_250000 | 32/32 | 1.325 | 1.228 | 2.198 | 1.072 | 0.0033 | 11.57 | 0.0433 | 0.1083 [0.1071, 0.1094] | 0.00004 | 5098/5108 |

Certified anchors with zero violations, summed over cells (pure exchange): k=1 1689814, k=10 31327, k=30 1202; imbalance-only exchange: k=1 3071007, k=10 1720013, k=30 1536398. K6 pass_R yes, pass_H yes. K4a is descriptive only and not in the paper (author decision 7).

## 8. Sensitivity: per-arm anchor ε^Y_a vs pooled anchor ε^Pool

Final calls after RG. 'flag' = reliable at one anchor and unreliable at the other (§9).

| Cell | Q | ε^Pool/ε^Y | y@Y | y@Pool | u@Y | u@Pool | u flag | uc@Y | uc@Pool | uc flag |
|---|---|---|---|---|---|---|---|---|---|---|
| MOM_25000 | T_N(10) | 1.004 | reliable 92/96 | reliable 95/96 | reliable 93/96 | reliable 95/96 | no | reliable 93/96 | reliable 95/96 | no |
| MOM_25000 | G_s | 1.035 | reliable 96/96 | reliable 96/96 | unreliable 0/96 | unreliable 19/96 | no | unreliable 1/96 | unreliable 47/96 | no |
| MOM_25000 | 1-CKA | 1.030 | reliable 96/96 | reliable 96/96 | unreliable 36/96 | inconclusive 74/96 | no | inconclusive 74/96 | inconclusive 90/96 | no |
| MOM_250000 | T_N(10) | 1.002 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | no | reliable 96/96 | reliable 96/96 | no |
| MOM_250000 | G_s | 1.078 | reliable 96/96 | reliable 96/96 | unreliable 5/96 | inconclusive 87/96 | no | unreliable 21/96 | reliable 95/96 | yes |
| MOM_250000 | 1-CKA | 1.113 | reliable 96/96 | reliable 96/96 | unreliable 23/96 | reliable 93/96 | yes | unreliable 66/96 | reliable 96/96 | yes |
| WD_25000 | T_N(10) | 1.012 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | no | reliable 96/96 | reliable 96/96 | no |
| WD_25000 | G_s | 1.013 | reliable 94/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | no | reliable 96/96 | reliable 96/96 | no |
| WD_25000 | 1-CKA | 1.180 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | no | reliable 96/96 | reliable 96/96 | no |
| WD_250000 | T_N(10) | 1.026 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | no | reliable 96/96 | reliable 96/96 | no |
| WD_250000 | G_s | 1.040 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | no | reliable 96/96 | reliable 96/96 | no |
| WD_250000 | 1-CKA | 1.147 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | no | reliable 96/96 | reliable 96/96 | no |
| FLOW_25000 | T_N(10) | 1.002 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | no | reliable 95/96 | reliable 96/96 | no |
| FLOW_25000 | G_s | 1.039 | reliable 95/96 | reliable 96/96 | unreliable 1/96 | unreliable 46/96 | no | unreliable 5/96 | inconclusive 82/96 | no |
| FLOW_25000 | 1-CKA | 1.002 | reliable 96/96 | reliable 96/96 | unreliable 25/96 | unreliable 25/96 | no | inconclusive 71/96 | inconclusive 73/96 | no |
| FLOW_250000 | T_N(10) | 1.001 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | no | reliable 96/96 | reliable 96/96 | no |
| FLOW_250000 | G_s | 1.051 | reliable 93/96 | reliable 96/96 | unreliable 32/96 | inconclusive 88/96 | no | inconclusive 84/96 | reliable 95/96 | no |
| FLOW_250000 | 1-CKA | 1.029 | reliable 96/96 | reliable 96/96 | unreliable 55/96 | inconclusive 75/96 | no | reliable 93/96 | reliable 94/96 | no |
| BAL-FLOW_25000 | T_N(10) | 1.000 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | no | reliable 96/96 | reliable 96/96 | no |
| BAL-FLOW_25000 | G_s | 1.032 | reliable 96/96 | reliable 96/96 | unreliable 20/96 | unreliable 68/96 | no | unreliable 37/96 | inconclusive 90/96 | no |
| BAL-FLOW_25000 | 1-CKA | 1.000 | reliable 94/96 | reliable 94/96 | unreliable 58/96 | unreliable 58/96 | no | inconclusive 81/96 | inconclusive 81/96 | no |
| BAL-FLOW_250000 | T_N(10) | 1.005 | reliable 95/96 | reliable 96/96 | reliable 95/96 | reliable 96/96 | no | reliable 95/96 | reliable 96/96 | no |
| BAL-FLOW_250000 | G_s | 1.068 | reliable 96/96 | reliable 96/96 | unreliable 35/96 | reliable 96/96 | yes | unreliable 65/96 | reliable 96/96 | yes |
| BAL-FLOW_250000 | 1-CKA | 1.052 | reliable 96/96 | reliable 96/96 | unreliable 67/96 | reliable 92/96 | yes | inconclusive 89/96 | reliable 96/96 | no |
| SMALL-FLOW_25000 | T_N(10) | 1.004 | reliable 96/96 | reliable 96/96 | reliable 95/96 | reliable 96/96 | no | reliable 95/96 | reliable 96/96 | no |
| SMALL-FLOW_25000 | G_s | 1.026 | reliable 94/96 | reliable 96/96 | unreliable 8/96 | unreliable 43/96 | no | unreliable 24/96 | inconclusive 77/96 | no |
| SMALL-FLOW_25000 | 1-CKA | 1.025 | reliable 96/96 | reliable 96/96 | unreliable 65/96 | inconclusive 88/96 | no | inconclusive 91/96 | reliable 96/96 | no |
| SMALL-FLOW_250000 | T_N(10) | 1.000 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | no | reliable 96/96 | reliable 96/96 | no |
| SMALL-FLOW_250000 | G_s | 1.000 | reliable 96/96 | reliable 96/96 | reliable 93/96 | reliable 93/96 | no | reliable 95/96 | reliable 95/96 | no |
| SMALL-FLOW_250000 | 1-CKA | 1.000 | reliable 96/96 | reliable 96/96 | reliable 95/96 | reliable 95/96 | no | reliable 96/96 | reliable 96/96 | no |
| BAL-MOM_25000 | T_N(10) | 1.003 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | no | reliable 32/32 | reliable 32/32 | no |
| BAL-MOM_25000 | G_s | 1.045 | inconclusive 31/32 | reliable 32/32 | inconclusive (RG) 0/32 | unreliable 11/32 | no | inconclusive (RG) 0/32 | unreliable 20/32 | no |
| BAL-MOM_25000 | 1-CKA | 1.059 | inconclusive 30/32 | reliable 32/32 | inconclusive (RG) 9/32 | inconclusive 25/32 | no | inconclusive (RG) 11/32 | inconclusive 30/32 | no |
| BAL-MOM_250000 | T_N(10) | 1.012 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | no | reliable 32/32 | reliable 32/32 | no |
| BAL-MOM_250000 | G_s | 1.092 | reliable 32/32 | reliable 32/32 | unreliable 0/32 | inconclusive 29/32 | no | unreliable 2/32 | reliable 32/32 | yes |
| BAL-MOM_250000 | 1-CKA | 1.129 | reliable 32/32 | reliable 32/32 | unreliable 6/32 | inconclusive 31/32 | no | unreliable 19/32 | reliable 32/32 | yes |
| SMALL-MOM_25000 | T_N(10) | 1.007 | inconclusive 31/32 | reliable 32/32 | inconclusive (RG) 31/32 | reliable 32/32 | no | inconclusive (RG) 31/32 | reliable 32/32 | no |
| SMALL-MOM_25000 | G_s | 1.025 | inconclusive 31/32 | reliable 32/32 | inconclusive (RG) 1/32 | unreliable 13/32 | no | inconclusive (RG) 3/32 | inconclusive 24/32 | no |
| SMALL-MOM_25000 | 1-CKA | 1.065 | reliable 32/32 | reliable 32/32 | inconclusive 22/32 | inconclusive 31/32 | no | inconclusive 29/32 | reliable 32/32 | no |
| SMALL-MOM_250000 | T_N(10) | 1.019 | inconclusive 26/32 | reliable 32/32 | inconclusive (RG) 26/32 | reliable 32/32 | no | inconclusive (RG) 26/32 | reliable 32/32 | no |
| SMALL-MOM_250000 | G_s | 1.072 | inconclusive 31/32 | reliable 32/32 | inconclusive (RG) 5/32 | inconclusive 28/32 | no | inconclusive (RG) 21/32 | inconclusive 31/32 | no |
| SMALL-MOM_250000 | 1-CKA | 1.167 | inconclusive 29/32 | reliable 32/32 | inconclusive (RG) 7/32 | inconclusive 31/32 | no | inconclusive (RG) 24/32 | reliable 32/32 | no |
| SGD-K_25000 | T_N(10) | 1.005 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | no | reliable 32/32 | reliable 32/32 | no |
| SGD-K_25000 | G_s | 1.000 | inconclusive 30/32 | inconclusive 30/32 | inconclusive (RG) 0/32 | inconclusive (RG) 0/32 | no | inconclusive (RG) 0/32 | inconclusive (RG) 0/32 | no |
| SGD-K_25000 | 1-CKA | 1.070 | inconclusive 28/32 | reliable 32/32 | inconclusive (RG) 0/32 | unreliable 17/32 | no | inconclusive (RG) 1/32 | inconclusive 25/32 | no |
| SGD-K_250000 | T_N(10) | 0.837 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | no | reliable 32/32 | reliable 32/32 | no |
| SGD-K_250000 | G_s | 0.179 | reliable 32/32 | reliable 32/32 | reliable 32/32 | unreliable 18/32 | yes | reliable 32/32 | unreliable 20/32 | yes |
| SGD-K_250000 | 1-CKA | 0.091 | reliable 32/32 | reliable 32/32 | reliable 32/32 | inconclusive 31/32 | no | reliable 32/32 | reliable 32/32 | no |

## 9. Decision tables, primary rungs (every entry)

x/n, p̂, one-sided 95% CP [lower, upper], ε used, ε*_r (the x_reliable-th order statistic), decision → final call after RG. All 24 rungs and every entry are in `render/decision_entries.csv`.

| Cell | Q | entry | x/n | p̂ | CP [lo, hi] | ε | ε*_r | decision | final |
|---|---|---|---|---|---|---|---|---|---|
| MOM_25000 | T_N(10) | y@Y | 92/96 | 0.958 | [0.907, 0.986] | 0.91821 | 0.91821 | reliable | reliable |
| MOM_25000 | T_N(10) | u@Y | 93/96 | 0.969 | [0.921, 0.991] | 0.91821 | 0.91812 | reliable | reliable |
| MOM_25000 | T_N(10) | uc@Y | 93/96 | 0.969 | [0.921, 0.991] | 0.91821 | 0.91816 | reliable | reliable |
| MOM_25000 | T_N(10) | ubal@Y | 92/96 | 0.958 | [0.907, 0.986] | 0.91821 | 0.91797 | reliable | reliable |
| MOM_25000 | T_N(10) | yM@Y | 91/96 | 0.948 | [0.894, 0.979] | 0.91821 | 0.91851 | inconclusive | inconclusive |
| MOM_25000 | T_N(10) | yI@Y | 93/96 | 0.969 | [0.921, 0.991] | 0.91821 | 0.91802 | reliable | reliable |
| MOM_25000 | T_N(10) | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91821 | 0.85068 | reliable | reliable |
| MOM_25000 | T_N(10) | y@Pool | 95/96 | 0.990 | [0.952, 0.999] | 0.92153 | 0.91821 | reliable | reliable |
| MOM_25000 | T_N(10) | u@Pool | 95/96 | 0.990 | [0.952, 0.999] | 0.92153 | 0.91812 | reliable | reliable |
| MOM_25000 | T_N(10) | uc@Pool | 95/96 | 0.990 | [0.952, 0.999] | 0.92153 | 0.91816 | reliable | reliable |
| MOM_25000 | T_N(10) | u@O | 93/96 | 0.969 | [0.921, 0.991] | 0.91855 | 0.91812 | reliable | reliable |
| MOM_25000 | T_N(10) | uc@C | 94/96 | 0.979 | [0.936, 0.996] | 0.91948 | 0.91816 | reliable | reliable |
| MOM_25000 | T_N(10) | u@C | 94/96 | 0.979 | [0.936, 0.996] | 0.91948 | 0.91812 | reliable | reliable |
| MOM_25000 | T_N(10) | y@Yq90 | 82/96 | 0.854 | [0.781, 0.910] | 0.91648 | 0.91821 | inconclusive | inconclusive |
| MOM_25000 | T_N(10) | u@Yq90 | 85/96 | 0.885 | [0.817, 0.934] | 0.91648 | 0.91812 | inconclusive | inconclusive (RG) |
| MOM_25000 | T_N(10) | h@h | 93/96 | 0.969 | [0.921, 0.991] | 0.85156 | 0.85068 | reliable | reliable |
| MOM_25000 | T_N(10) | h_l2@h_l2 | 95/96 | 0.990 | [0.952, 0.999] | 0.81313 | 0.81069 | reliable | reliable |
| MOM_25000 | G_s | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.28477 | 0.28124 | reliable | reliable |
| MOM_25000 | G_s | u@Y | 0/96 | 0.000 | [0.000, 0.031] | 0.28477 | 0.30848 | unreliable | unreliable |
| MOM_25000 | G_s | uc@Y | 1/96 | 0.010 | [0.000534, 0.048] | 0.28477 | 0.30118 | unreliable | unreliable |
| MOM_25000 | G_s | ubal@Y | 18/96 | 0.188 | [0.125, 0.265] | 0.28477 | 0.29659 | unreliable | unreliable |
| MOM_25000 | G_s | yM@Y | 82/96 | 0.854 | [0.781, 0.910] | 0.28477 | 0.28700 | inconclusive | inconclusive |
| MOM_25000 | G_s | yI@Y | 33/96 | 0.344 | [0.263, 0.432] | 0.28477 | 0.29269 | unreliable | unreliable |
| MOM_25000 | G_s | h@Y | 0/96 | 0.000 | [0.000, 0.031] | 0.28477 | 0.38627 | unreliable | unreliable |
| MOM_25000 | G_s | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.29467 | 0.28124 | reliable | reliable |
| MOM_25000 | G_s | u@Pool | 19/96 | 0.198 | [0.134, 0.277] | 0.29467 | 0.30848 | unreliable | unreliable |
| MOM_25000 | G_s | uc@Pool | 47/96 | 0.490 | [0.402, 0.578] | 0.29467 | 0.30118 | unreliable | unreliable |
| MOM_25000 | G_s | u@O | 95/96 | 0.990 | [0.952, 0.999] | 0.31370 | 0.30848 | reliable | reliable |
| MOM_25000 | G_s | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.30468 | 0.30118 | reliable | reliable |
| MOM_25000 | G_s | u@C | 84/96 | 0.875 | [0.805, 0.926] | 0.30468 | 0.30848 | inconclusive | inconclusive |
| MOM_25000 | G_s | y@Yq90 | 82/96 | 0.854 | [0.781, 0.910] | 0.27968 | 0.28124 | inconclusive | inconclusive |
| MOM_25000 | G_s | u@Yq90 | 0/96 | 0.000 | [0.000, 0.031] | 0.27968 | 0.30848 | unreliable | inconclusive (RG) |
| MOM_25000 | G_s | h@h | 94/96 | 0.979 | [0.936, 0.996] | 0.39731 | 0.38627 | reliable | reliable |
| MOM_25000 | G_s | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.13318 | 0.12780 | reliable | reliable |
| MOM_25000 | 1-CKA | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.16397 | 0.15975 | reliable | reliable |
| MOM_25000 | 1-CKA | u@Y | 36/96 | 0.375 | [0.292, 0.464] | 0.16397 | 0.17638 | unreliable | unreliable |
| MOM_25000 | 1-CKA | uc@Y | 74/96 | 0.771 | [0.689, 0.839] | 0.16397 | 0.16935 | inconclusive | inconclusive |
| MOM_25000 | 1-CKA | ubal@Y | 73/96 | 0.760 | [0.678, 0.830] | 0.16397 | 0.16946 | inconclusive | inconclusive |
| MOM_25000 | 1-CKA | yM@Y | 88/96 | 0.917 | [0.855, 0.958] | 0.16397 | 0.16740 | inconclusive | inconclusive |
| MOM_25000 | 1-CKA | yI@Y | 88/96 | 0.917 | [0.855, 0.958] | 0.16397 | 0.16541 | inconclusive | inconclusive |
| MOM_25000 | 1-CKA | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.16397 | 0.10124 | reliable | reliable |
| MOM_25000 | 1-CKA | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.16885 | 0.15975 | reliable | reliable |
| MOM_25000 | 1-CKA | u@Pool | 74/96 | 0.771 | [0.689, 0.839] | 0.16885 | 0.17638 | inconclusive | inconclusive |
| MOM_25000 | 1-CKA | uc@Pool | 90/96 | 0.938 | [0.880, 0.972] | 0.16885 | 0.16935 | inconclusive | inconclusive |
| MOM_25000 | 1-CKA | u@O | 95/96 | 0.990 | [0.952, 0.999] | 0.17953 | 0.17638 | reliable | reliable |
| MOM_25000 | 1-CKA | uc@C | 95/96 | 0.990 | [0.952, 0.999] | 0.17086 | 0.16935 | reliable | reliable |
| MOM_25000 | 1-CKA | u@C | 83/96 | 0.865 | [0.793, 0.918] | 0.17086 | 0.17638 | inconclusive | inconclusive |
| MOM_25000 | 1-CKA | y@Yq90 | 85/96 | 0.885 | [0.817, 0.934] | 0.15809 | 0.15975 | inconclusive | inconclusive |
| MOM_25000 | 1-CKA | u@Yq90 | 6/96 | 0.062 | [0.028, 0.120] | 0.15809 | 0.17638 | unreliable | inconclusive (RG) |
| MOM_25000 | 1-CKA | h@h | 95/96 | 0.990 | [0.952, 0.999] | 0.11370 | 0.10124 | reliable | reliable |
| MOM_25000 | 1-CKA | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.08181 | 0.07747 | reliable | reliable |
| MOM_250000 | T_N(10) | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83096 | 0.82031 | reliable | reliable |
| MOM_250000 | T_N(10) | u@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83096 | 0.82109 | reliable | reliable |
| MOM_250000 | T_N(10) | uc@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83096 | 0.82026 | reliable | reliable |
| MOM_250000 | T_N(10) | ubal@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83096 | 0.81890 | reliable | reliable |
| MOM_250000 | T_N(10) | yM@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83096 | 0.81997 | reliable | reliable |
| MOM_250000 | T_N(10) | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83096 | 0.82017 | reliable | reliable |
| MOM_250000 | T_N(10) | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83096 | 0.71572 | reliable | reliable |
| MOM_250000 | T_N(10) | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82031 | reliable | reliable |
| MOM_250000 | T_N(10) | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82109 | reliable | reliable |
| MOM_250000 | T_N(10) | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82026 | reliable | reliable |
| MOM_250000 | T_N(10) | u@O | 96/96 | 1.000 | [0.969, 1.000] | 0.82817 | 0.82109 | reliable | reliable |
| MOM_250000 | T_N(10) | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.82959 | 0.82026 | reliable | reliable |
| MOM_250000 | T_N(10) | u@C | 96/96 | 1.000 | [0.969, 1.000] | 0.82959 | 0.82109 | reliable | reliable |
| MOM_250000 | T_N(10) | y@Yq90 | 83/96 | 0.865 | [0.793, 0.918] | 0.81736 | 0.82031 | inconclusive | inconclusive |
| MOM_250000 | T_N(10) | u@Yq90 | 86/96 | 0.896 | [0.830, 0.942] | 0.81736 | 0.82109 | inconclusive | inconclusive (RG) |
| MOM_250000 | T_N(10) | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.72021 | 0.71572 | reliable | reliable |
| MOM_250000 | T_N(10) | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.67944 | 0.67100 | reliable | reliable |
| MOM_250000 | G_s | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.19460 | 0.19055 | reliable | reliable |
| MOM_250000 | G_s | u@Y | 5/96 | 0.052 | [0.021, 0.106] | 0.19460 | 0.21200 | unreliable | unreliable |
| MOM_250000 | G_s | uc@Y | 21/96 | 0.219 | [0.152, 0.300] | 0.19460 | 0.20718 | unreliable | unreliable |
| MOM_250000 | G_s | ubal@Y | 68/96 | 0.708 | [0.623, 0.784] | 0.19460 | 0.20001 | unreliable | unreliable |
| MOM_250000 | G_s | yM@Y | 78/96 | 0.812 | [0.735, 0.875] | 0.19460 | 0.19804 | inconclusive | inconclusive |
| MOM_250000 | G_s | yI@Y | 38/96 | 0.396 | [0.312, 0.485] | 0.19460 | 0.20226 | unreliable | unreliable |
| MOM_250000 | G_s | h@Y | 0/96 | 0.000 | [0.000, 0.031] | 0.19460 | 0.26856 | unreliable | unreliable |
| MOM_250000 | G_s | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.20985 | 0.19055 | reliable | reliable |
| MOM_250000 | G_s | u@Pool | 87/96 | 0.906 | [0.842, 0.950] | 0.20985 | 0.21200 | inconclusive | inconclusive |
| MOM_250000 | G_s | uc@Pool | 95/96 | 0.990 | [0.952, 0.999] | 0.20985 | 0.20718 | reliable | reliable |
| MOM_250000 | G_s | u@O | 95/96 | 0.990 | [0.952, 0.999] | 0.21923 | 0.21200 | reliable | reliable |
| MOM_250000 | G_s | uc@C | 95/96 | 0.990 | [0.952, 0.999] | 0.21093 | 0.20718 | reliable | reliable |
| MOM_250000 | G_s | u@C | 90/96 | 0.938 | [0.880, 0.972] | 0.21093 | 0.21200 | inconclusive | inconclusive |
| MOM_250000 | G_s | y@Yq90 | 86/96 | 0.896 | [0.830, 0.942] | 0.18869 | 0.19055 | inconclusive | inconclusive |
| MOM_250000 | G_s | u@Yq90 | 0/96 | 0.000 | [0.000, 0.031] | 0.18869 | 0.21200 | unreliable | inconclusive (RG) |
| MOM_250000 | G_s | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.28235 | 0.26856 | reliable | reliable |
| MOM_250000 | G_s | h_l2@h_l2 | 89/96 | 0.927 | [0.867, 0.965] | 0.10229 | 0.10354 | inconclusive | inconclusive |
| MOM_250000 | 1-CKA | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.07790 | 0.07453 | reliable | reliable |
| MOM_250000 | 1-CKA | u@Y | 23/96 | 0.240 | [0.170, 0.322] | 0.07790 | 0.08599 | unreliable | unreliable |
| MOM_250000 | 1-CKA | uc@Y | 66/96 | 0.688 | [0.601, 0.765] | 0.07790 | 0.08080 | unreliable | unreliable |
| MOM_250000 | 1-CKA | ubal@Y | 67/96 | 0.698 | [0.612, 0.775] | 0.07790 | 0.08166 | unreliable | unreliable |
| MOM_250000 | 1-CKA | yM@Y | 86/96 | 0.896 | [0.830, 0.942] | 0.07790 | 0.07919 | inconclusive | inconclusive |
| MOM_250000 | 1-CKA | yI@Y | 90/96 | 0.938 | [0.880, 0.972] | 0.07790 | 0.07919 | inconclusive | inconclusive |
| MOM_250000 | 1-CKA | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.07790 | 0.05629 | reliable | reliable |
| MOM_250000 | 1-CKA | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.07453 | reliable | reliable |
| MOM_250000 | 1-CKA | u@Pool | 93/96 | 0.969 | [0.921, 0.991] | 0.08670 | 0.08599 | reliable | reliable |
| MOM_250000 | 1-CKA | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.08080 | reliable | reliable |
| MOM_250000 | 1-CKA | u@O | 95/96 | 0.990 | [0.952, 0.999] | 0.08955 | 0.08599 | reliable | reliable |
| MOM_250000 | 1-CKA | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.08596 | 0.08080 | reliable | reliable |
| MOM_250000 | 1-CKA | u@C | 91/96 | 0.948 | [0.894, 0.979] | 0.08596 | 0.08599 | inconclusive | inconclusive |
| MOM_250000 | 1-CKA | y@Yq90 | 83/96 | 0.865 | [0.793, 0.918] | 0.07258 | 0.07453 | inconclusive | inconclusive |
| MOM_250000 | 1-CKA | u@Yq90 | 1/96 | 0.010 | [0.000534, 0.048] | 0.07258 | 0.08599 | unreliable | inconclusive (RG) |
| MOM_250000 | 1-CKA | h@h | 95/96 | 0.990 | [0.952, 0.999] | 0.06458 | 0.05629 | reliable | reliable |
| MOM_250000 | 1-CKA | h_l2@h_l2 | 91/96 | 0.948 | [0.894, 0.979] | 0.04463 | 0.04469 | inconclusive | inconclusive |
| WD_25000 | T_N(10) | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91079 | 0.90742 | reliable | reliable |
| WD_25000 | T_N(10) | u@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91079 | 0.90776 | reliable | reliable |
| WD_25000 | T_N(10) | uc@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91079 | 0.90732 | reliable | reliable |
| WD_25000 | T_N(10) | ubal@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91079 | 0.90757 | reliable | reliable |
| WD_25000 | T_N(10) | yM@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91079 | 0.90723 | reliable | reliable |
| WD_25000 | T_N(10) | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91079 | 0.90737 | reliable | reliable |
| WD_25000 | T_N(10) | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91079 | 0.84683 | reliable | reliable |
| WD_25000 | T_N(10) | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.90742 | reliable | reliable |
| WD_25000 | T_N(10) | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.90776 | reliable | reliable |
| WD_25000 | T_N(10) | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.90732 | reliable | reliable |
| WD_25000 | T_N(10) | u@O | 96/96 | 1.000 | [0.969, 1.000] | 0.91069 | 0.90776 | reliable | reliable |
| WD_25000 | T_N(10) | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.91079 | 0.90732 | reliable | reliable |
| WD_25000 | T_N(10) | u@C | 96/96 | 1.000 | [0.969, 1.000] | 0.91079 | 0.90776 | reliable | reliable |
| WD_25000 | T_N(10) | y@Yq90 | 88/96 | 0.917 | [0.855, 0.958] | 0.90676 | 0.90742 | inconclusive | inconclusive |
| WD_25000 | T_N(10) | u@Yq90 | 90/96 | 0.938 | [0.880, 0.972] | 0.90676 | 0.90776 | inconclusive | inconclusive (RG) |
| WD_25000 | T_N(10) | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.85073 | 0.84683 | reliable | reliable |
| WD_25000 | T_N(10) | h_l2@h_l2 | 95/96 | 0.990 | [0.952, 0.999] | 0.83506 | 0.83223 | reliable | reliable |
| WD_25000 | G_s | y@Y | 94/96 | 0.979 | [0.936, 0.996] | 0.29079 | 0.28954 | reliable | reliable |
| WD_25000 | G_s | u@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.29079 | 0.28487 | reliable | reliable |
| WD_25000 | G_s | uc@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.29079 | 0.28331 | reliable | reliable |
| WD_25000 | G_s | ubal@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.29079 | 0.28330 | reliable | reliable |
| WD_25000 | G_s | yM@Y | 92/96 | 0.958 | [0.907, 0.986] | 0.29079 | 0.29059 | reliable | reliable |
| WD_25000 | G_s | yI@Y | 91/96 | 0.948 | [0.894, 0.979] | 0.29079 | 0.29110 | inconclusive | inconclusive |
| WD_25000 | G_s | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.29079 | 0.27960 | reliable | reliable |
| WD_25000 | G_s | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.29467 | 0.28954 | reliable | reliable |
| WD_25000 | G_s | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.29467 | 0.28487 | reliable | reliable |
| WD_25000 | G_s | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.29467 | 0.28331 | reliable | reliable |
| WD_25000 | G_s | u@O | 93/96 | 0.969 | [0.921, 0.991] | 0.28546 | 0.28487 | reliable | reliable |
| WD_25000 | G_s | uc@C | 94/96 | 0.979 | [0.936, 0.996] | 0.28533 | 0.28331 | reliable | reliable |
| WD_25000 | G_s | u@C | 93/96 | 0.969 | [0.921, 0.991] | 0.28533 | 0.28487 | reliable | reliable |
| WD_25000 | G_s | y@Yq90 | 83/96 | 0.865 | [0.793, 0.918] | 0.28647 | 0.28954 | inconclusive | inconclusive |
| WD_25000 | G_s | u@Yq90 | 94/96 | 0.979 | [0.936, 0.996] | 0.28647 | 0.28487 | reliable | inconclusive (RG) |
| WD_25000 | G_s | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.28357 | 0.27960 | reliable | reliable |
| WD_25000 | G_s | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.14722 | 0.14447 | reliable | reliable |
| WD_25000 | 1-CKA | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.14314 | 0.14059 | reliable | reliable |
| WD_25000 | 1-CKA | u@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.14314 | 0.13330 | reliable | reliable |
| WD_25000 | 1-CKA | uc@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.14314 | 0.13263 | reliable | reliable |
| WD_25000 | 1-CKA | ubal@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.14314 | 0.13123 | reliable | reliable |
| WD_25000 | 1-CKA | yM@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.14314 | 0.14103 | reliable | reliable |
| WD_25000 | 1-CKA | yI@Y | 93/96 | 0.969 | [0.921, 0.991] | 0.14314 | 0.14244 | reliable | reliable |
| WD_25000 | 1-CKA | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.14314 | 0.09802 | reliable | reliable |
| WD_25000 | 1-CKA | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.16885 | 0.14059 | reliable | reliable |
| WD_25000 | 1-CKA | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.16885 | 0.13330 | reliable | reliable |
| WD_25000 | 1-CKA | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.16885 | 0.13263 | reliable | reliable |
| WD_25000 | 1-CKA | u@O | 96/96 | 1.000 | [0.969, 1.000] | 0.13606 | 0.13330 | reliable | reliable |
| WD_25000 | 1-CKA | uc@C | 95/96 | 0.990 | [0.952, 0.999] | 0.13485 | 0.13263 | reliable | reliable |
| WD_25000 | 1-CKA | u@C | 95/96 | 0.990 | [0.952, 0.999] | 0.13485 | 0.13330 | reliable | reliable |
| WD_25000 | 1-CKA | y@Yq90 | 88/96 | 0.917 | [0.855, 0.958] | 0.13894 | 0.14059 | inconclusive | inconclusive |
| WD_25000 | 1-CKA | u@Yq90 | 96/96 | 1.000 | [0.969, 1.000] | 0.13894 | 0.13330 | reliable | inconclusive (RG) |
| WD_25000 | 1-CKA | h@h | 94/96 | 0.979 | [0.936, 0.996] | 0.09955 | 0.09802 | reliable | reliable |
| WD_25000 | 1-CKA | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.09455 | 0.09200 | reliable | reliable |
| WD_250000 | T_N(10) | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.81162 | 0.80396 | reliable | reliable |
| WD_250000 | T_N(10) | u@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.81162 | 0.80288 | reliable | reliable |
| WD_250000 | T_N(10) | uc@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.81162 | 0.80254 | reliable | reliable |
| WD_250000 | T_N(10) | ubal@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.81162 | 0.80298 | reliable | reliable |
| WD_250000 | T_N(10) | yM@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.81162 | 0.80420 | reliable | reliable |
| WD_250000 | T_N(10) | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.81162 | 0.80396 | reliable | reliable |
| WD_250000 | T_N(10) | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.81162 | 0.71172 | reliable | reliable |
| WD_250000 | T_N(10) | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.80396 | reliable | reliable |
| WD_250000 | T_N(10) | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.80288 | reliable | reliable |
| WD_250000 | T_N(10) | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.80254 | reliable | reliable |
| WD_250000 | T_N(10) | u@O | 96/96 | 1.000 | [0.969, 1.000] | 0.81392 | 0.80288 | reliable | reliable |
| WD_250000 | T_N(10) | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.81431 | 0.80254 | reliable | reliable |
| WD_250000 | T_N(10) | u@C | 96/96 | 1.000 | [0.969, 1.000] | 0.81431 | 0.80288 | reliable | reliable |
| WD_250000 | T_N(10) | y@Yq90 | 81/96 | 0.844 | [0.770, 0.901] | 0.80007 | 0.80396 | inconclusive | inconclusive |
| WD_250000 | T_N(10) | u@Yq90 | 84/96 | 0.875 | [0.805, 0.926] | 0.80007 | 0.80288 | inconclusive | inconclusive (RG) |
| WD_250000 | T_N(10) | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.71670 | 0.71172 | reliable | reliable |
| WD_250000 | T_N(10) | h_l2@h_l2 | 95/96 | 0.990 | [0.952, 0.999] | 0.69941 | 0.69238 | reliable | reliable |
| WD_250000 | G_s | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.20175 | 0.19189 | reliable | reliable |
| WD_250000 | G_s | u@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.20175 | 0.19094 | reliable | reliable |
| WD_250000 | G_s | uc@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.20175 | 0.19111 | reliable | reliable |
| WD_250000 | G_s | ubal@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.20175 | 0.19039 | reliable | reliable |
| WD_250000 | G_s | yM@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.20175 | 0.19169 | reliable | reliable |
| WD_250000 | G_s | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.20175 | 0.19245 | reliable | reliable |
| WD_250000 | G_s | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.20175 | 0.18912 | reliable | reliable |
| WD_250000 | G_s | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.20985 | 0.19189 | reliable | reliable |
| WD_250000 | G_s | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.20985 | 0.19094 | reliable | reliable |
| WD_250000 | G_s | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.20985 | 0.19111 | reliable | reliable |
| WD_250000 | G_s | u@O | 96/96 | 1.000 | [0.969, 1.000] | 0.20091 | 0.19094 | reliable | reliable |
| WD_250000 | G_s | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.19984 | 0.19111 | reliable | reliable |
| WD_250000 | G_s | u@C | 96/96 | 1.000 | [0.969, 1.000] | 0.19984 | 0.19094 | reliable | reliable |
| WD_250000 | G_s | y@Yq90 | 83/96 | 0.865 | [0.793, 0.918] | 0.18995 | 0.19189 | inconclusive | inconclusive |
| WD_250000 | G_s | u@Yq90 | 85/96 | 0.885 | [0.817, 0.934] | 0.18995 | 0.19094 | inconclusive | inconclusive (RG) |
| WD_250000 | G_s | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.19872 | 0.18912 | reliable | reliable |
| WD_250000 | G_s | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.11609 | 0.10724 | reliable | reliable |
| WD_250000 | 1-CKA | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.07557 | 0.06380 | reliable | reliable |
| WD_250000 | 1-CKA | u@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.07557 | 0.06597 | reliable | reliable |
| WD_250000 | 1-CKA | uc@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.07557 | 0.06636 | reliable | reliable |
| WD_250000 | 1-CKA | ubal@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.07557 | 0.06524 | reliable | reliable |
| WD_250000 | 1-CKA | yM@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.07557 | 0.06336 | reliable | reliable |
| WD_250000 | 1-CKA | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.07557 | 0.06460 | reliable | reliable |
| WD_250000 | 1-CKA | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.07557 | 0.05300 | reliable | reliable |
| WD_250000 | 1-CKA | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.06380 | reliable | reliable |
| WD_250000 | 1-CKA | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.06597 | reliable | reliable |
| WD_250000 | 1-CKA | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.06636 | reliable | reliable |
| WD_250000 | 1-CKA | u@O | 96/96 | 1.000 | [0.969, 1.000] | 0.07744 | 0.06597 | reliable | reliable |
| WD_250000 | 1-CKA | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.07678 | 0.06636 | reliable | reliable |
| WD_250000 | 1-CKA | u@C | 96/96 | 1.000 | [0.969, 1.000] | 0.07678 | 0.06597 | reliable | reliable |
| WD_250000 | 1-CKA | y@Yq90 | 85/96 | 0.885 | [0.817, 0.934] | 0.06278 | 0.06380 | inconclusive | inconclusive |
| WD_250000 | 1-CKA | u@Yq90 | 65/96 | 0.677 | [0.590, 0.756] | 0.06278 | 0.06597 | unreliable | inconclusive (RG) |
| WD_250000 | 1-CKA | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.06438 | 0.05300 | reliable | reliable |
| WD_250000 | 1-CKA | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.05919 | 0.04944 | reliable | reliable |
| FLOW_25000 | T_N(10) | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91992 | 0.91719 | reliable | reliable |
| FLOW_25000 | T_N(10) | u@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91992 | 0.91704 | reliable | reliable |
| FLOW_25000 | T_N(10) | uc@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.91992 | 0.91694 | reliable | reliable |
| FLOW_25000 | T_N(10) | ubal@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91992 | 0.91724 | reliable | reliable |
| FLOW_25000 | T_N(10) | yM@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91992 | 0.91733 | reliable | reliable |
| FLOW_25000 | T_N(10) | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91992 | 0.91724 | reliable | reliable |
| FLOW_25000 | T_N(10) | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91992 | 0.84546 | reliable | reliable |
| FLOW_25000 | T_N(10) | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91719 | reliable | reliable |
| FLOW_25000 | T_N(10) | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91704 | reliable | reliable |
| FLOW_25000 | T_N(10) | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91694 | reliable | reliable |
| FLOW_25000 | T_N(10) | u@O | 95/96 | 0.990 | [0.952, 0.999] | 0.91875 | 0.91704 | reliable | reliable |
| FLOW_25000 | T_N(10) | uc@C | 95/96 | 0.990 | [0.952, 0.999] | 0.91919 | 0.91694 | reliable | reliable |
| FLOW_25000 | T_N(10) | u@C | 95/96 | 0.990 | [0.952, 0.999] | 0.91919 | 0.91704 | reliable | reliable |
| FLOW_25000 | T_N(10) | y@Yq90 | 83/96 | 0.865 | [0.793, 0.918] | 0.91570 | 0.91719 | inconclusive | inconclusive |
| FLOW_25000 | T_N(10) | u@Yq90 | 82/96 | 0.854 | [0.781, 0.910] | 0.91570 | 0.91704 | inconclusive | inconclusive (RG) |
| FLOW_25000 | T_N(10) | h@h | 93/96 | 0.969 | [0.921, 0.991] | 0.84736 | 0.84546 | reliable | reliable |
| FLOW_25000 | T_N(10) | h_l2@h_l2 | 95/96 | 0.990 | [0.952, 0.999] | 0.79907 | 0.79702 | reliable | reliable |
| FLOW_25000 | G_s | y@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.28375 | 0.28120 | reliable | reliable |
| FLOW_25000 | G_s | u@Y | 1/96 | 0.010 | [0.000534, 0.048] | 0.28375 | 0.30789 | unreliable | unreliable |
| FLOW_25000 | G_s | uc@Y | 5/96 | 0.052 | [0.021, 0.106] | 0.28375 | 0.29707 | unreliable | unreliable |
| FLOW_25000 | G_s | ubal@Y | 3/96 | 0.031 | [0.009, 0.079] | 0.28375 | 0.30447 | unreliable | unreliable |
| FLOW_25000 | G_s | yM@Y | 74/96 | 0.771 | [0.689, 0.839] | 0.28375 | 0.29211 | inconclusive | inconclusive |
| FLOW_25000 | G_s | yI@Y | 93/96 | 0.969 | [0.921, 0.991] | 0.28375 | 0.28328 | reliable | reliable |
| FLOW_25000 | G_s | h@Y | 0/96 | 0.000 | [0.000, 0.031] | 0.28375 | 0.41275 | unreliable | unreliable |
| FLOW_25000 | G_s | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.29467 | 0.28120 | reliable | reliable |
| FLOW_25000 | G_s | u@Pool | 46/96 | 0.479 | [0.392, 0.568] | 0.29467 | 0.30789 | unreliable | unreliable |
| FLOW_25000 | G_s | uc@Pool | 82/96 | 0.854 | [0.781, 0.910] | 0.29467 | 0.29707 | inconclusive | inconclusive |
| FLOW_25000 | G_s | u@O | 92/96 | 0.958 | [0.907, 0.986] | 0.30816 | 0.30789 | reliable | reliable |
| FLOW_25000 | G_s | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.30289 | 0.29707 | reliable | reliable |
| FLOW_25000 | G_s | u@C | 85/96 | 0.885 | [0.817, 0.934] | 0.30289 | 0.30789 | inconclusive | inconclusive |
| FLOW_25000 | G_s | y@Yq90 | 90/96 | 0.938 | [0.880, 0.972] | 0.28105 | 0.28120 | inconclusive | inconclusive |
| FLOW_25000 | G_s | u@Yq90 | 0/96 | 0.000 | [0.000, 0.031] | 0.28105 | 0.30789 | unreliable | inconclusive (RG) |
| FLOW_25000 | G_s | h@h | 94/96 | 0.979 | [0.936, 0.996] | 0.41665 | 0.41275 | reliable | reliable |
| FLOW_25000 | G_s | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.12854 | 0.12557 | reliable | reliable |
| FLOW_25000 | 1-CKA | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.16847 | 0.16553 | reliable | reliable |
| FLOW_25000 | 1-CKA | u@Y | 25/96 | 0.260 | [0.188, 0.344] | 0.16847 | 0.18286 | unreliable | unreliable |
| FLOW_25000 | 1-CKA | uc@Y | 71/96 | 0.740 | [0.656, 0.812] | 0.16847 | 0.17271 | inconclusive | inconclusive |
| FLOW_25000 | 1-CKA | ubal@Y | 39/96 | 0.406 | [0.322, 0.495] | 0.16847 | 0.17961 | unreliable | unreliable |
| FLOW_25000 | 1-CKA | yM@Y | 70/96 | 0.729 | [0.645, 0.803] | 0.16847 | 0.17570 | inconclusive | inconclusive |
| FLOW_25000 | 1-CKA | yI@Y | 93/96 | 0.969 | [0.921, 0.991] | 0.16847 | 0.16730 | reliable | reliable |
| FLOW_25000 | 1-CKA | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.16847 | 0.09488 | reliable | reliable |
| FLOW_25000 | 1-CKA | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.16885 | 0.16553 | reliable | reliable |
| FLOW_25000 | 1-CKA | u@Pool | 25/96 | 0.260 | [0.188, 0.344] | 0.16885 | 0.18286 | unreliable | unreliable |
| FLOW_25000 | 1-CKA | uc@Pool | 73/96 | 0.760 | [0.678, 0.830] | 0.16885 | 0.17271 | inconclusive | inconclusive |
| FLOW_25000 | 1-CKA | u@O | 93/96 | 0.969 | [0.921, 0.991] | 0.18340 | 0.18286 | reliable | reliable |
| FLOW_25000 | 1-CKA | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.17584 | 0.17271 | reliable | reliable |
| FLOW_25000 | 1-CKA | u@C | 77/96 | 0.802 | [0.723, 0.866] | 0.17584 | 0.18286 | inconclusive | inconclusive |
| FLOW_25000 | 1-CKA | y@Yq90 | 87/96 | 0.906 | [0.842, 0.950] | 0.16434 | 0.16553 | inconclusive | inconclusive |
| FLOW_25000 | 1-CKA | u@Yq90 | 5/96 | 0.052 | [0.021, 0.106] | 0.16434 | 0.18286 | unreliable | inconclusive (RG) |
| FLOW_25000 | 1-CKA | h@h | 93/96 | 0.969 | [0.921, 0.991] | 0.09821 | 0.09488 | reliable | reliable |
| FLOW_25000 | 1-CKA | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.08663 | 0.08410 | reliable | reliable |
| FLOW_250000 | T_N(10) | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83223 | 0.82539 | reliable | reliable |
| FLOW_250000 | T_N(10) | u@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83223 | 0.82402 | reliable | reliable |
| FLOW_250000 | T_N(10) | uc@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83223 | 0.82383 | reliable | reliable |
| FLOW_250000 | T_N(10) | ubal@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83223 | 0.82378 | reliable | reliable |
| FLOW_250000 | T_N(10) | yM@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83223 | 0.82578 | reliable | reliable |
| FLOW_250000 | T_N(10) | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83223 | 0.82544 | reliable | reliable |
| FLOW_250000 | T_N(10) | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83223 | 0.71597 | reliable | reliable |
| FLOW_250000 | T_N(10) | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82539 | reliable | reliable |
| FLOW_250000 | T_N(10) | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82402 | reliable | reliable |
| FLOW_250000 | T_N(10) | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82383 | reliable | reliable |
| FLOW_250000 | T_N(10) | u@O | 96/96 | 1.000 | [0.969, 1.000] | 0.83008 | 0.82402 | reliable | reliable |
| FLOW_250000 | T_N(10) | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.83037 | 0.82383 | reliable | reliable |
| FLOW_250000 | T_N(10) | u@C | 96/96 | 1.000 | [0.969, 1.000] | 0.83037 | 0.82402 | reliable | reliable |
| FLOW_250000 | T_N(10) | y@Yq90 | 94/96 | 0.979 | [0.936, 0.996] | 0.82580 | 0.82539 | reliable | reliable |
| FLOW_250000 | T_N(10) | u@Yq90 | 93/96 | 0.969 | [0.921, 0.991] | 0.82580 | 0.82402 | reliable | reliable |
| FLOW_250000 | T_N(10) | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.72534 | 0.71597 | reliable | reliable |
| FLOW_250000 | T_N(10) | h_l2@h_l2 | 94/96 | 0.979 | [0.936, 0.996] | 0.66865 | 0.66431 | reliable | reliable |
| FLOW_250000 | G_s | y@Y | 93/96 | 0.969 | [0.921, 0.991] | 0.19959 | 0.19419 | reliable | reliable |
| FLOW_250000 | G_s | u@Y | 32/96 | 0.333 | [0.254, 0.421] | 0.19959 | 0.21251 | unreliable | unreliable |
| FLOW_250000 | G_s | uc@Y | 84/96 | 0.875 | [0.805, 0.926] | 0.19959 | 0.20449 | inconclusive | inconclusive |
| FLOW_250000 | G_s | ubal@Y | 57/96 | 0.594 | [0.505, 0.678] | 0.19959 | 0.20929 | unreliable | unreliable |
| FLOW_250000 | G_s | yM@Y | 83/96 | 0.865 | [0.793, 0.918] | 0.19959 | 0.20455 | inconclusive | inconclusive |
| FLOW_250000 | G_s | yI@Y | 92/96 | 0.958 | [0.907, 0.986] | 0.19959 | 0.19770 | reliable | reliable |
| FLOW_250000 | G_s | h@Y | 0/96 | 0.000 | [0.000, 0.031] | 0.19959 | 0.30294 | unreliable | unreliable |
| FLOW_250000 | G_s | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.20985 | 0.19419 | reliable | reliable |
| FLOW_250000 | G_s | u@Pool | 88/96 | 0.917 | [0.855, 0.958] | 0.20985 | 0.21251 | inconclusive | inconclusive |
| FLOW_250000 | G_s | uc@Pool | 95/96 | 0.990 | [0.952, 0.999] | 0.20985 | 0.20449 | reliable | reliable |
| FLOW_250000 | G_s | u@O | 92/96 | 0.958 | [0.907, 0.986] | 0.21381 | 0.21251 | reliable | reliable |
| FLOW_250000 | G_s | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.21215 | 0.20449 | reliable | reliable |
| FLOW_250000 | G_s | u@C | 90/96 | 0.938 | [0.880, 0.972] | 0.21215 | 0.21251 | inconclusive | inconclusive |
| FLOW_250000 | G_s | y@Yq90 | 87/96 | 0.906 | [0.842, 0.950] | 0.19285 | 0.19419 | inconclusive | inconclusive |
| FLOW_250000 | G_s | u@Yq90 | 3/96 | 0.031 | [0.009, 0.079] | 0.19285 | 0.21251 | unreliable | inconclusive (RG) |
| FLOW_250000 | G_s | h@h | 94/96 | 0.979 | [0.936, 0.996] | 0.31764 | 0.30294 | reliable | reliable |
| FLOW_250000 | G_s | h_l2@h_l2 | 95/96 | 0.990 | [0.952, 0.999] | 0.10253 | 0.09962 | reliable | reliable |
| FLOW_250000 | 1-CKA | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.08421 | 0.07927 | reliable | reliable |
| FLOW_250000 | 1-CKA | u@Y | 55/96 | 0.573 | [0.484, 0.659] | 0.08421 | 0.09150 | unreliable | unreliable |
| FLOW_250000 | 1-CKA | uc@Y | 93/96 | 0.969 | [0.921, 0.991] | 0.08421 | 0.08348 | reliable | reliable |
| FLOW_250000 | 1-CKA | ubal@Y | 73/96 | 0.760 | [0.678, 0.830] | 0.08421 | 0.08963 | inconclusive | inconclusive |
| FLOW_250000 | 1-CKA | yM@Y | 83/96 | 0.865 | [0.793, 0.918] | 0.08421 | 0.08762 | inconclusive | inconclusive |
| FLOW_250000 | 1-CKA | yI@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.08421 | 0.08151 | reliable | reliable |
| FLOW_250000 | 1-CKA | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.08421 | 0.06019 | reliable | reliable |
| FLOW_250000 | 1-CKA | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.07927 | reliable | reliable |
| FLOW_250000 | 1-CKA | u@Pool | 75/96 | 0.781 | [0.700, 0.848] | 0.08670 | 0.09150 | inconclusive | inconclusive |
| FLOW_250000 | 1-CKA | uc@Pool | 94/96 | 0.979 | [0.936, 0.996] | 0.08670 | 0.08348 | reliable | reliable |
| FLOW_250000 | 1-CKA | u@O | 94/96 | 0.979 | [0.936, 0.996] | 0.09398 | 0.09150 | reliable | reliable |
| FLOW_250000 | 1-CKA | uc@C | 95/96 | 0.990 | [0.952, 0.999] | 0.08815 | 0.08348 | reliable | reliable |
| FLOW_250000 | 1-CKA | u@C | 83/96 | 0.865 | [0.793, 0.918] | 0.08815 | 0.09150 | inconclusive | inconclusive |
| FLOW_250000 | 1-CKA | y@Yq90 | 88/96 | 0.917 | [0.855, 0.958] | 0.07804 | 0.07927 | inconclusive | inconclusive |
| FLOW_250000 | 1-CKA | u@Yq90 | 5/96 | 0.052 | [0.021, 0.106] | 0.07804 | 0.09150 | unreliable | inconclusive (RG) |
| FLOW_250000 | 1-CKA | h@h | 94/96 | 0.979 | [0.936, 0.996] | 0.06202 | 0.06019 | reliable | reliable |
| FLOW_250000 | 1-CKA | h_l2@h_l2 | 95/96 | 0.990 | [0.952, 0.999] | 0.05079 | 0.04622 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91694 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | u@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91665 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | uc@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91685 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | ubal@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91665 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | yM@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91689 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91694 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.84624 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91694 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91665 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91685 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | u@O | 95/96 | 0.990 | [0.952, 0.999] | 0.91934 | 0.91665 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.91943 | 0.91685 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | u@C | 95/96 | 0.990 | [0.952, 0.999] | 0.91943 | 0.91665 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | y@Yq90 | 93/96 | 0.969 | [0.921, 0.991] | 0.91714 | 0.91694 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | u@Yq90 | 92/96 | 0.958 | [0.907, 0.986] | 0.91714 | 0.91665 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.85161 | 0.84624 | reliable | reliable |
| BAL-FLOW_25000 | T_N(10) | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.80410 | 0.79624 | reliable | reliable |
| BAL-FLOW_25000 | G_s | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.28545 | 0.28286 | reliable | reliable |
| BAL-FLOW_25000 | G_s | u@Y | 20/96 | 0.208 | [0.143, 0.288] | 0.28545 | 0.30131 | unreliable | unreliable |
| BAL-FLOW_25000 | G_s | uc@Y | 37/96 | 0.385 | [0.302, 0.474] | 0.28545 | 0.29551 | unreliable | unreliable |
| BAL-FLOW_25000 | G_s | ubal@Y | 20/96 | 0.208 | [0.143, 0.288] | 0.28545 | 0.30131 | unreliable | unreliable |
| BAL-FLOW_25000 | G_s | yM@Y | 84/96 | 0.875 | [0.805, 0.926] | 0.28545 | 0.28788 | inconclusive | inconclusive |
| BAL-FLOW_25000 | G_s | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.28545 | 0.28286 | reliable | reliable |
| BAL-FLOW_25000 | G_s | h@Y | 0/96 | 0.000 | [0.000, 0.031] | 0.28545 | 0.41819 | unreliable | unreliable |
| BAL-FLOW_25000 | G_s | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.29467 | 0.28286 | reliable | reliable |
| BAL-FLOW_25000 | G_s | u@Pool | 68/96 | 0.708 | [0.623, 0.784] | 0.29467 | 0.30131 | unreliable | unreliable |
| BAL-FLOW_25000 | G_s | uc@Pool | 90/96 | 0.938 | [0.880, 0.972] | 0.29467 | 0.29551 | inconclusive | inconclusive |
| BAL-FLOW_25000 | G_s | u@O | 96/96 | 1.000 | [0.969, 1.000] | 0.30420 | 0.30131 | reliable | reliable |
| BAL-FLOW_25000 | G_s | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.29870 | 0.29551 | reliable | reliable |
| BAL-FLOW_25000 | G_s | u@C | 86/96 | 0.896 | [0.830, 0.942] | 0.29870 | 0.30131 | inconclusive | inconclusive |
| BAL-FLOW_25000 | G_s | y@Yq90 | 88/96 | 0.917 | [0.855, 0.958] | 0.28078 | 0.28286 | inconclusive | inconclusive |
| BAL-FLOW_25000 | G_s | u@Yq90 | 4/96 | 0.042 | [0.014, 0.093] | 0.28078 | 0.30131 | unreliable | inconclusive (RG) |
| BAL-FLOW_25000 | G_s | h@h | 90/96 | 0.938 | [0.880, 0.972] | 0.41550 | 0.41819 | inconclusive | inconclusive |
| BAL-FLOW_25000 | G_s | h_l2@h_l2 | 93/96 | 0.969 | [0.921, 0.991] | 0.12731 | 0.12612 | reliable | reliable |
| BAL-FLOW_25000 | 1-CKA | y@Y | 94/96 | 0.979 | [0.936, 0.996] | 0.16885 | 0.16651 | reliable | reliable |
| BAL-FLOW_25000 | 1-CKA | u@Y | 58/96 | 0.604 | [0.515, 0.688] | 0.16885 | 0.17676 | unreliable | unreliable |
| BAL-FLOW_25000 | 1-CKA | uc@Y | 81/96 | 0.844 | [0.770, 0.901] | 0.16885 | 0.17225 | inconclusive | inconclusive |
| BAL-FLOW_25000 | 1-CKA | ubal@Y | 58/96 | 0.604 | [0.515, 0.688] | 0.16885 | 0.17676 | unreliable | unreliable |
| BAL-FLOW_25000 | 1-CKA | yM@Y | 88/96 | 0.917 | [0.855, 0.958] | 0.16885 | 0.17047 | inconclusive | inconclusive |
| BAL-FLOW_25000 | 1-CKA | yI@Y | 94/96 | 0.979 | [0.936, 0.996] | 0.16885 | 0.16651 | reliable | reliable |
| BAL-FLOW_25000 | 1-CKA | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.16885 | 0.09887 | reliable | reliable |
| BAL-FLOW_25000 | 1-CKA | y@Pool | 94/96 | 0.979 | [0.936, 0.996] | 0.16885 | 0.16651 | reliable | reliable |
| BAL-FLOW_25000 | 1-CKA | u@Pool | 58/96 | 0.604 | [0.515, 0.688] | 0.16885 | 0.17676 | unreliable | unreliable |
| BAL-FLOW_25000 | 1-CKA | uc@Pool | 81/96 | 0.844 | [0.770, 0.901] | 0.16885 | 0.17225 | inconclusive | inconclusive |
| BAL-FLOW_25000 | 1-CKA | u@O | 95/96 | 0.990 | [0.952, 0.999] | 0.17969 | 0.17676 | reliable | reliable |
| BAL-FLOW_25000 | 1-CKA | uc@C | 93/96 | 0.969 | [0.921, 0.991] | 0.17329 | 0.17225 | reliable | reliable |
| BAL-FLOW_25000 | 1-CKA | u@C | 83/96 | 0.865 | [0.793, 0.918] | 0.17329 | 0.17676 | inconclusive | inconclusive |
| BAL-FLOW_25000 | 1-CKA | y@Yq90 | 85/96 | 0.885 | [0.817, 0.934] | 0.16367 | 0.16651 | inconclusive | inconclusive |
| BAL-FLOW_25000 | 1-CKA | u@Yq90 | 24/96 | 0.250 | [0.179, 0.333] | 0.16367 | 0.17676 | unreliable | inconclusive (RG) |
| BAL-FLOW_25000 | 1-CKA | h@h | 94/96 | 0.979 | [0.936, 0.996] | 0.10372 | 0.09887 | reliable | reliable |
| BAL-FLOW_25000 | 1-CKA | h_l2@h_l2 | 92/96 | 0.958 | [0.907, 0.986] | 0.08498 | 0.08463 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | y@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.82881 | 0.82739 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | u@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.82881 | 0.82339 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | uc@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.82881 | 0.82563 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | ubal@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.82881 | 0.82339 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | yM@Y | 94/96 | 0.979 | [0.936, 0.996] | 0.82881 | 0.82676 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | yI@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.82881 | 0.82739 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.82881 | 0.71616 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82739 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82339 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82563 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | u@O | 92/96 | 0.958 | [0.907, 0.986] | 0.82622 | 0.82339 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | uc@C | 92/96 | 0.958 | [0.907, 0.986] | 0.82588 | 0.82563 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | u@C | 92/96 | 0.958 | [0.907, 0.986] | 0.82588 | 0.82339 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | y@Yq90 | 81/96 | 0.844 | [0.770, 0.901] | 0.82209 | 0.82739 | inconclusive | inconclusive |
| BAL-FLOW_250000 | T_N(10) | u@Yq90 | 87/96 | 0.906 | [0.842, 0.950] | 0.82209 | 0.82339 | inconclusive | inconclusive (RG) |
| BAL-FLOW_250000 | T_N(10) | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.72197 | 0.71616 | reliable | reliable |
| BAL-FLOW_250000 | T_N(10) | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.67446 | 0.66646 | reliable | reliable |
| BAL-FLOW_250000 | G_s | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.19656 | 0.19319 | reliable | reliable |
| BAL-FLOW_250000 | G_s | u@Y | 35/96 | 0.365 | [0.283, 0.453] | 0.19656 | 0.20539 | unreliable | unreliable |
| BAL-FLOW_250000 | G_s | uc@Y | 65/96 | 0.677 | [0.590, 0.756] | 0.19656 | 0.20015 | unreliable | unreliable |
| BAL-FLOW_250000 | G_s | ubal@Y | 35/96 | 0.365 | [0.283, 0.453] | 0.19656 | 0.20539 | unreliable | unreliable |
| BAL-FLOW_250000 | G_s | yM@Y | 80/96 | 0.833 | [0.758, 0.893] | 0.19656 | 0.19919 | inconclusive | inconclusive |
| BAL-FLOW_250000 | G_s | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.19656 | 0.19319 | reliable | reliable |
| BAL-FLOW_250000 | G_s | h@Y | 0/96 | 0.000 | [0.000, 0.031] | 0.19656 | 0.29368 | unreliable | unreliable |
| BAL-FLOW_250000 | G_s | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.20985 | 0.19319 | reliable | reliable |
| BAL-FLOW_250000 | G_s | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.20985 | 0.20539 | reliable | reliable |
| BAL-FLOW_250000 | G_s | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.20985 | 0.20015 | reliable | reliable |
| BAL-FLOW_250000 | G_s | u@O | 95/96 | 0.990 | [0.952, 0.999] | 0.20801 | 0.20539 | reliable | reliable |
| BAL-FLOW_250000 | G_s | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.20366 | 0.20015 | reliable | reliable |
| BAL-FLOW_250000 | G_s | u@C | 83/96 | 0.865 | [0.793, 0.918] | 0.20366 | 0.20539 | inconclusive | inconclusive |
| BAL-FLOW_250000 | G_s | y@Yq90 | 90/96 | 0.938 | [0.880, 0.972] | 0.19261 | 0.19319 | inconclusive | inconclusive |
| BAL-FLOW_250000 | G_s | u@Yq90 | 12/96 | 0.125 | [0.074, 0.195] | 0.19261 | 0.20539 | unreliable | inconclusive (RG) |
| BAL-FLOW_250000 | G_s | h@h | 95/96 | 0.990 | [0.952, 0.999] | 0.33215 | 0.29368 | reliable | reliable |
| BAL-FLOW_250000 | G_s | h_l2@h_l2 | 95/96 | 0.990 | [0.952, 0.999] | 0.10239 | 0.09882 | reliable | reliable |
| BAL-FLOW_250000 | 1-CKA | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.08242 | 0.07771 | reliable | reliable |
| BAL-FLOW_250000 | 1-CKA | u@Y | 67/96 | 0.698 | [0.612, 0.775] | 0.08242 | 0.08662 | unreliable | unreliable |
| BAL-FLOW_250000 | 1-CKA | uc@Y | 89/96 | 0.927 | [0.867, 0.965] | 0.08242 | 0.08289 | inconclusive | inconclusive |
| BAL-FLOW_250000 | 1-CKA | ubal@Y | 67/96 | 0.698 | [0.612, 0.775] | 0.08242 | 0.08662 | unreliable | unreliable |
| BAL-FLOW_250000 | 1-CKA | yM@Y | 93/96 | 0.969 | [0.921, 0.991] | 0.08242 | 0.08165 | reliable | reliable |
| BAL-FLOW_250000 | 1-CKA | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.08242 | 0.07771 | reliable | reliable |
| BAL-FLOW_250000 | 1-CKA | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.08242 | 0.05644 | reliable | reliable |
| BAL-FLOW_250000 | 1-CKA | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.07771 | reliable | reliable |
| BAL-FLOW_250000 | 1-CKA | u@Pool | 92/96 | 0.958 | [0.907, 0.986] | 0.08670 | 0.08662 | reliable | reliable |
| BAL-FLOW_250000 | 1-CKA | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.08289 | reliable | reliable |
| BAL-FLOW_250000 | 1-CKA | u@O | 94/96 | 0.979 | [0.936, 0.996] | 0.08907 | 0.08662 | reliable | reliable |
| BAL-FLOW_250000 | 1-CKA | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.08497 | 0.08289 | reliable | reliable |
| BAL-FLOW_250000 | 1-CKA | u@C | 84/96 | 0.875 | [0.805, 0.926] | 0.08497 | 0.08662 | inconclusive | inconclusive |
| BAL-FLOW_250000 | 1-CKA | y@Yq90 | 93/96 | 0.969 | [0.921, 0.991] | 0.07778 | 0.07771 | reliable | reliable |
| BAL-FLOW_250000 | 1-CKA | u@Yq90 | 17/96 | 0.177 | [0.116, 0.254] | 0.07778 | 0.08662 | unreliable | unreliable |
| BAL-FLOW_250000 | 1-CKA | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.06775 | 0.05644 | reliable | reliable |
| BAL-FLOW_250000 | 1-CKA | h_l2@h_l2 | 95/96 | 0.990 | [0.952, 0.999] | 0.05045 | 0.04605 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91753 | 0.91543 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | u@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.91753 | 0.91519 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | uc@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.91753 | 0.91577 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | ubal@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.91753 | 0.91523 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | yM@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91753 | 0.91519 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91753 | 0.91543 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.91753 | 0.85093 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91543 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91519 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.92153 | 0.91577 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | u@O | 95/96 | 0.990 | [0.952, 0.999] | 0.91797 | 0.91519 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | uc@C | 95/96 | 0.990 | [0.952, 0.999] | 0.91763 | 0.91577 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | u@C | 95/96 | 0.990 | [0.952, 0.999] | 0.91763 | 0.91519 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | y@Yq90 | 88/96 | 0.917 | [0.855, 0.958] | 0.91479 | 0.91543 | inconclusive | inconclusive |
| SMALL-FLOW_25000 | T_N(10) | u@Yq90 | 90/96 | 0.938 | [0.880, 0.972] | 0.91479 | 0.91519 | inconclusive | inconclusive (RG) |
| SMALL-FLOW_25000 | T_N(10) | h@h | 93/96 | 0.969 | [0.921, 0.991] | 0.85171 | 0.85093 | reliable | reliable |
| SMALL-FLOW_25000 | T_N(10) | h_l2@h_l2 | 95/96 | 0.990 | [0.952, 0.999] | 0.80718 | 0.80542 | reliable | reliable |
| SMALL-FLOW_25000 | G_s | y@Y | 94/96 | 0.979 | [0.936, 0.996] | 0.28731 | 0.28451 | reliable | reliable |
| SMALL-FLOW_25000 | G_s | u@Y | 8/96 | 0.083 | [0.042, 0.145] | 0.28731 | 0.30395 | unreliable | unreliable |
| SMALL-FLOW_25000 | G_s | uc@Y | 24/96 | 0.250 | [0.179, 0.333] | 0.28731 | 0.29698 | unreliable | unreliable |
| SMALL-FLOW_25000 | G_s | ubal@Y | 8/96 | 0.083 | [0.042, 0.145] | 0.28731 | 0.30394 | unreliable | unreliable |
| SMALL-FLOW_25000 | G_s | yM@Y | 78/96 | 0.812 | [0.735, 0.875] | 0.28731 | 0.29193 | inconclusive | inconclusive |
| SMALL-FLOW_25000 | G_s | yI@Y | 94/96 | 0.979 | [0.936, 0.996] | 0.28731 | 0.28453 | reliable | reliable |
| SMALL-FLOW_25000 | G_s | h@Y | 0/96 | 0.000 | [0.000, 0.031] | 0.28731 | 0.41988 | unreliable | unreliable |
| SMALL-FLOW_25000 | G_s | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.29467 | 0.28451 | reliable | reliable |
| SMALL-FLOW_25000 | G_s | u@Pool | 43/96 | 0.448 | [0.361, 0.537] | 0.29467 | 0.30395 | unreliable | unreliable |
| SMALL-FLOW_25000 | G_s | uc@Pool | 77/96 | 0.802 | [0.723, 0.866] | 0.29467 | 0.29698 | inconclusive | inconclusive |
| SMALL-FLOW_25000 | G_s | u@O | 96/96 | 1.000 | [0.969, 1.000] | 0.31165 | 0.30395 | reliable | reliable |
| SMALL-FLOW_25000 | G_s | uc@C | 95/96 | 0.990 | [0.952, 0.999] | 0.30236 | 0.29698 | reliable | reliable |
| SMALL-FLOW_25000 | G_s | u@C | 88/96 | 0.917 | [0.855, 0.958] | 0.30236 | 0.30395 | inconclusive | inconclusive |
| SMALL-FLOW_25000 | G_s | y@Yq90 | 90/96 | 0.938 | [0.880, 0.972] | 0.28380 | 0.28451 | inconclusive | inconclusive |
| SMALL-FLOW_25000 | G_s | u@Yq90 | 1/96 | 0.010 | [0.000534, 0.048] | 0.28380 | 0.30395 | unreliable | inconclusive (RG) |
| SMALL-FLOW_25000 | G_s | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.56210 | 0.41988 | reliable | reliable |
| SMALL-FLOW_25000 | G_s | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.13243 | 0.12988 | reliable | reliable |
| SMALL-FLOW_25000 | 1-CKA | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.16472 | 0.16201 | reliable | reliable |
| SMALL-FLOW_25000 | 1-CKA | u@Y | 65/96 | 0.677 | [0.590, 0.756] | 0.16472 | 0.17113 | unreliable | unreliable |
| SMALL-FLOW_25000 | 1-CKA | uc@Y | 91/96 | 0.948 | [0.894, 0.979] | 0.16472 | 0.16530 | inconclusive | inconclusive |
| SMALL-FLOW_25000 | 1-CKA | ubal@Y | 65/96 | 0.677 | [0.590, 0.756] | 0.16472 | 0.17111 | unreliable | unreliable |
| SMALL-FLOW_25000 | 1-CKA | yM@Y | 79/96 | 0.823 | [0.746, 0.884] | 0.16472 | 0.16803 | inconclusive | inconclusive |
| SMALL-FLOW_25000 | 1-CKA | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.16472 | 0.16203 | reliable | reliable |
| SMALL-FLOW_25000 | 1-CKA | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.16472 | 0.09047 | reliable | reliable |
| SMALL-FLOW_25000 | 1-CKA | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.16885 | 0.16201 | reliable | reliable |
| SMALL-FLOW_25000 | 1-CKA | u@Pool | 88/96 | 0.917 | [0.855, 0.958] | 0.16885 | 0.17113 | inconclusive | inconclusive |
| SMALL-FLOW_25000 | 1-CKA | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.16885 | 0.16530 | reliable | reliable |
| SMALL-FLOW_25000 | 1-CKA | u@O | 96/96 | 1.000 | [0.969, 1.000] | 0.17585 | 0.17113 | reliable | reliable |
| SMALL-FLOW_25000 | 1-CKA | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.16864 | 0.16530 | reliable | reliable |
| SMALL-FLOW_25000 | 1-CKA | u@C | 87/96 | 0.906 | [0.842, 0.950] | 0.16864 | 0.17113 | inconclusive | inconclusive |
| SMALL-FLOW_25000 | 1-CKA | y@Yq90 | 91/96 | 0.948 | [0.894, 0.979] | 0.16154 | 0.16201 | inconclusive | inconclusive |
| SMALL-FLOW_25000 | 1-CKA | u@Yq90 | 44/96 | 0.458 | [0.371, 0.547] | 0.16154 | 0.17113 | unreliable | inconclusive (RG) |
| SMALL-FLOW_25000 | 1-CKA | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.13684 | 0.09047 | reliable | reliable |
| SMALL-FLOW_25000 | 1-CKA | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.09122 | 0.08822 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82510 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | u@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82397 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | uc@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82368 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | ubal@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82397 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | yM@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82642 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82515 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.72271 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82510 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | u@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82397 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.83271 | 0.82368 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | u@O | 96/96 | 1.000 | [0.969, 1.000] | 0.82749 | 0.82397 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.82734 | 0.82368 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | u@C | 96/96 | 1.000 | [0.969, 1.000] | 0.82734 | 0.82397 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | y@Yq90 | 78/96 | 0.812 | [0.735, 0.875] | 0.81953 | 0.82510 | inconclusive | inconclusive |
| SMALL-FLOW_250000 | T_N(10) | u@Yq90 | 86/96 | 0.896 | [0.830, 0.942] | 0.81953 | 0.82397 | inconclusive | inconclusive (RG) |
| SMALL-FLOW_250000 | T_N(10) | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.72612 | 0.72271 | reliable | reliable |
| SMALL-FLOW_250000 | T_N(10) | h_l2@h_l2 | 95/96 | 0.990 | [0.952, 0.999] | 0.67754 | 0.67183 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.20985 | 0.19644 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | u@Y | 93/96 | 0.969 | [0.921, 0.991] | 0.20985 | 0.20907 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | uc@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.20985 | 0.20074 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | ubal@Y | 93/96 | 0.969 | [0.921, 0.991] | 0.20985 | 0.20906 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | yM@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.20985 | 0.20428 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.20985 | 0.19647 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | h@Y | 0/96 | 0.000 | [0.000, 0.031] | 0.20985 | 0.31913 | unreliable | unreliable |
| SMALL-FLOW_250000 | G_s | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.20985 | 0.19644 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | u@Pool | 93/96 | 0.969 | [0.921, 0.991] | 0.20985 | 0.20907 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | uc@Pool | 95/96 | 0.990 | [0.952, 0.999] | 0.20985 | 0.20074 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | u@O | 96/96 | 1.000 | [0.969, 1.000] | 0.22009 | 0.20907 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.21277 | 0.20074 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | u@C | 94/96 | 0.979 | [0.936, 0.996] | 0.21277 | 0.20907 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | y@Yq90 | 88/96 | 0.917 | [0.855, 0.958] | 0.19463 | 0.19644 | inconclusive | inconclusive |
| SMALL-FLOW_250000 | G_s | u@Yq90 | 22/96 | 0.229 | [0.161, 0.311] | 0.19463 | 0.20907 | unreliable | inconclusive (RG) |
| SMALL-FLOW_250000 | G_s | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.35939 | 0.31913 | reliable | reliable |
| SMALL-FLOW_250000 | G_s | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.11184 | 0.10088 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | y@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.07690 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | u@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.08670 | 0.08456 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | uc@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.08092 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | ubal@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.08670 | 0.08455 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | yM@Y | 95/96 | 0.990 | [0.952, 0.999] | 0.08670 | 0.08228 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | yI@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.07690 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | h@Y | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.05485 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | y@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.07690 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | u@Pool | 95/96 | 0.990 | [0.952, 0.999] | 0.08670 | 0.08456 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | uc@Pool | 96/96 | 1.000 | [0.969, 1.000] | 0.08670 | 0.08092 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | u@O | 96/96 | 1.000 | [0.969, 1.000] | 0.09489 | 0.08456 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | uc@C | 96/96 | 1.000 | [0.969, 1.000] | 0.09141 | 0.08092 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | u@C | 96/96 | 1.000 | [0.969, 1.000] | 0.09141 | 0.08456 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | y@Yq90 | 89/96 | 0.927 | [0.867, 0.965] | 0.07607 | 0.07690 | inconclusive | inconclusive |
| SMALL-FLOW_250000 | 1-CKA | u@Yq90 | 32/96 | 0.333 | [0.254, 0.421] | 0.07607 | 0.08456 | unreliable | inconclusive (RG) |
| SMALL-FLOW_250000 | 1-CKA | h@h | 96/96 | 1.000 | [0.969, 1.000] | 0.06960 | 0.05485 | reliable | reliable |
| SMALL-FLOW_250000 | 1-CKA | h_l2@h_l2 | 96/96 | 1.000 | [0.969, 1.000] | 0.05626 | 0.04906 | reliable | reliable |
| BAL-MOM_25000 | T_N(10) | y@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91855 | 0.91792 | reliable | reliable |
| BAL-MOM_25000 | T_N(10) | u@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91855 | 0.91812 | reliable | reliable |
| BAL-MOM_25000 | T_N(10) | uc@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91855 | 0.91821 | reliable | reliable |
| BAL-MOM_25000 | T_N(10) | ubal@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91855 | 0.91831 | reliable | reliable |
| BAL-MOM_25000 | T_N(10) | yM@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91855 | 0.91831 | reliable | reliable |
| BAL-MOM_25000 | T_N(10) | yI@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91855 | 0.91797 | reliable | reliable |
| BAL-MOM_25000 | T_N(10) | h@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91855 | 0.85454 | reliable | reliable |
| BAL-MOM_25000 | T_N(10) | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.92153 | 0.91792 | reliable | reliable |
| BAL-MOM_25000 | T_N(10) | u@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.92153 | 0.91812 | reliable | reliable |
| BAL-MOM_25000 | T_N(10) | uc@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.92153 | 0.91821 | reliable | reliable |
| BAL-MOM_25000 | T_N(10) | u@O | 29/32 | 0.906 | [0.775, 0.974] | 0.91743 | 0.91812 | inconclusive | inconclusive |
| BAL-MOM_25000 | T_N(10) | uc@C | 32/32 | 1.000 | [0.911, 1.000] | 0.91870 | 0.91821 | reliable | reliable |
| BAL-MOM_25000 | T_N(10) | u@C | 32/32 | 1.000 | [0.911, 1.000] | 0.91870 | 0.91812 | reliable | reliable |
| BAL-MOM_25000 | T_N(10) | y@Yq90 | 25/32 | 0.781 | [0.628, 0.893] | 0.91685 | 0.91792 | inconclusive | inconclusive |
| BAL-MOM_25000 | T_N(10) | u@Yq90 | 27/32 | 0.844 | [0.699, 0.936] | 0.91685 | 0.91812 | inconclusive | inconclusive (RG) |
| BAL-MOM_25000 | T_N(10) | h@h | 31/32 | 0.969 | [0.860, 0.998] | 0.85059 | 0.85454 | inconclusive | inconclusive |
| BAL-MOM_25000 | G_s | y@Y | 31/32 | 0.969 | [0.860, 0.998] | 0.28198 | 0.28326 | inconclusive | inconclusive |
| BAL-MOM_25000 | G_s | u@Y | 0/32 | 0.000 | [0.000, 0.089] | 0.28198 | 0.30887 | unreliable | inconclusive (RG) |
| BAL-MOM_25000 | G_s | uc@Y | 0/32 | 0.000 | [0.000, 0.089] | 0.28198 | 0.30438 | unreliable | inconclusive (RG) |
| BAL-MOM_25000 | G_s | ubal@Y | 4/32 | 0.125 | [0.044, 0.264] | 0.28198 | 0.29705 | unreliable | inconclusive (RG) |
| BAL-MOM_25000 | G_s | yM@Y | 25/32 | 0.781 | [0.628, 0.893] | 0.28198 | 0.28779 | inconclusive | inconclusive (RG) |
| BAL-MOM_25000 | G_s | yI@Y | 8/32 | 0.250 | [0.131, 0.406] | 0.28198 | 0.29509 | unreliable | inconclusive (RG) |
| BAL-MOM_25000 | G_s | h@Y | 0/32 | 0.000 | [0.000, 0.089] | 0.28198 | 0.38548 | unreliable | inconclusive (RG) |
| BAL-MOM_25000 | G_s | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.29467 | 0.28326 | reliable | reliable |
| BAL-MOM_25000 | G_s | u@Pool | 11/32 | 0.344 | [0.206, 0.504] | 0.29467 | 0.30887 | unreliable | unreliable |
| BAL-MOM_25000 | G_s | uc@Pool | 20/32 | 0.625 | [0.464, 0.767] | 0.29467 | 0.30438 | unreliable | unreliable |
| BAL-MOM_25000 | G_s | u@O | 31/32 | 0.969 | [0.860, 0.998] | 0.30817 | 0.30887 | inconclusive | inconclusive |
| BAL-MOM_25000 | G_s | uc@C | 31/32 | 0.969 | [0.860, 0.998] | 0.30053 | 0.30438 | inconclusive | inconclusive |
| BAL-MOM_25000 | G_s | u@C | 24/32 | 0.750 | [0.594, 0.869] | 0.30053 | 0.30887 | inconclusive | inconclusive (RG) |
| BAL-MOM_25000 | G_s | y@Yq90 | 28/32 | 0.875 | [0.736, 0.956] | 0.28005 | 0.28326 | inconclusive | inconclusive |
| BAL-MOM_25000 | G_s | u@Yq90 | 0/32 | 0.000 | [0.000, 0.089] | 0.28005 | 0.30887 | unreliable | inconclusive (RG) |
| BAL-MOM_25000 | G_s | h@h | 32/32 | 1.000 | [0.911, 1.000] | 0.40918 | 0.38548 | reliable | reliable |
| BAL-MOM_25000 | 1-CKA | y@Y | 30/32 | 0.938 | [0.816, 0.989] | 0.15947 | 0.16252 | inconclusive | inconclusive |
| BAL-MOM_25000 | 1-CKA | u@Y | 9/32 | 0.281 | [0.155, 0.439] | 0.15947 | 0.17771 | unreliable | inconclusive (RG) |
| BAL-MOM_25000 | 1-CKA | uc@Y | 11/32 | 0.344 | [0.206, 0.504] | 0.15947 | 0.17216 | unreliable | inconclusive (RG) |
| BAL-MOM_25000 | 1-CKA | ubal@Y | 17/32 | 0.531 | [0.373, 0.685] | 0.15947 | 0.17027 | unreliable | inconclusive (RG) |
| BAL-MOM_25000 | 1-CKA | yM@Y | 24/32 | 0.750 | [0.594, 0.869] | 0.15947 | 0.16888 | inconclusive | inconclusive (RG) |
| BAL-MOM_25000 | 1-CKA | yI@Y | 22/32 | 0.688 | [0.528, 0.820] | 0.15947 | 0.16996 | inconclusive | inconclusive (RG) |
| BAL-MOM_25000 | 1-CKA | h@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.15947 | 0.10201 | reliable | inconclusive (RG) |
| BAL-MOM_25000 | 1-CKA | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.16885 | 0.16252 | reliable | reliable |
| BAL-MOM_25000 | 1-CKA | u@Pool | 25/32 | 0.781 | [0.628, 0.893] | 0.16885 | 0.17771 | inconclusive | inconclusive |
| BAL-MOM_25000 | 1-CKA | uc@Pool | 30/32 | 0.938 | [0.816, 0.989] | 0.16885 | 0.17216 | inconclusive | inconclusive |
| BAL-MOM_25000 | 1-CKA | u@O | 32/32 | 1.000 | [0.911, 1.000] | 0.17852 | 0.17771 | reliable | reliable |
| BAL-MOM_25000 | 1-CKA | uc@C | 31/32 | 0.969 | [0.860, 0.998] | 0.17180 | 0.17216 | inconclusive | inconclusive |
| BAL-MOM_25000 | 1-CKA | u@C | 28/32 | 0.875 | [0.736, 0.956] | 0.17180 | 0.17771 | inconclusive | inconclusive (RG) |
| BAL-MOM_25000 | 1-CKA | y@Yq90 | 29/32 | 0.906 | [0.775, 0.974] | 0.15754 | 0.16252 | inconclusive | inconclusive |
| BAL-MOM_25000 | 1-CKA | u@Yq90 | 4/32 | 0.125 | [0.044, 0.264] | 0.15754 | 0.17771 | unreliable | inconclusive (RG) |
| BAL-MOM_25000 | 1-CKA | h@h | 32/32 | 1.000 | [0.911, 1.000] | 0.10427 | 0.10201 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | y@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.82261 | 0.81885 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | u@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.82261 | 0.82061 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | uc@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.82261 | 0.81919 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | ubal@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.82261 | 0.82041 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | yM@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.82261 | 0.82012 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | yI@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.82261 | 0.81958 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | h@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.82261 | 0.71606 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.83271 | 0.81885 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | u@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.83271 | 0.82061 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | uc@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.83271 | 0.81919 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | u@O | 32/32 | 1.000 | [0.911, 1.000] | 0.82285 | 0.82061 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | uc@C | 32/32 | 1.000 | [0.911, 1.000] | 0.82354 | 0.81919 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | u@C | 32/32 | 1.000 | [0.911, 1.000] | 0.82354 | 0.82061 | reliable | reliable |
| BAL-MOM_250000 | T_N(10) | y@Yq90 | 31/32 | 0.969 | [0.860, 0.998] | 0.81875 | 0.81885 | inconclusive | inconclusive |
| BAL-MOM_250000 | T_N(10) | u@Yq90 | 31/32 | 0.969 | [0.860, 0.998] | 0.81875 | 0.82061 | inconclusive | inconclusive (RG) |
| BAL-MOM_250000 | T_N(10) | h@h | 30/32 | 0.938 | [0.816, 0.989] | 0.71406 | 0.71606 | inconclusive | inconclusive |
| BAL-MOM_250000 | G_s | y@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.19220 | 0.18795 | reliable | reliable |
| BAL-MOM_250000 | G_s | u@Y | 0/32 | 0.000 | [0.000, 0.089] | 0.19220 | 0.21196 | unreliable | unreliable |
| BAL-MOM_250000 | G_s | uc@Y | 2/32 | 0.062 | [0.011, 0.184] | 0.19220 | 0.20531 | unreliable | unreliable |
| BAL-MOM_250000 | G_s | ubal@Y | 16/32 | 0.500 | [0.344, 0.656] | 0.19220 | 0.19907 | unreliable | unreliable |
| BAL-MOM_250000 | G_s | yM@Y | 22/32 | 0.688 | [0.528, 0.820] | 0.19220 | 0.19699 | inconclusive | inconclusive |
| BAL-MOM_250000 | G_s | yI@Y | 8/32 | 0.250 | [0.131, 0.406] | 0.19220 | 0.20152 | unreliable | unreliable |
| BAL-MOM_250000 | G_s | h@Y | 0/32 | 0.000 | [0.000, 0.089] | 0.19220 | 0.27049 | unreliable | unreliable |
| BAL-MOM_250000 | G_s | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.20985 | 0.18795 | reliable | reliable |
| BAL-MOM_250000 | G_s | u@Pool | 29/32 | 0.906 | [0.775, 0.974] | 0.20985 | 0.21196 | inconclusive | inconclusive |
| BAL-MOM_250000 | G_s | uc@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.20985 | 0.20531 | reliable | reliable |
| BAL-MOM_250000 | G_s | u@O | 31/32 | 0.969 | [0.860, 0.998] | 0.21136 | 0.21196 | inconclusive | inconclusive |
| BAL-MOM_250000 | G_s | uc@C | 32/32 | 1.000 | [0.911, 1.000] | 0.20796 | 0.20531 | reliable | reliable |
| BAL-MOM_250000 | G_s | u@C | 28/32 | 0.875 | [0.736, 0.956] | 0.20796 | 0.21196 | inconclusive | inconclusive |
| BAL-MOM_250000 | G_s | y@Yq90 | 32/32 | 1.000 | [0.911, 1.000] | 0.18819 | 0.18795 | reliable | reliable |
| BAL-MOM_250000 | G_s | u@Yq90 | 0/32 | 0.000 | [0.000, 0.089] | 0.18819 | 0.21196 | unreliable | unreliable |
| BAL-MOM_250000 | G_s | h@h | 31/32 | 0.969 | [0.860, 0.998] | 0.26774 | 0.27049 | inconclusive | inconclusive |
| BAL-MOM_250000 | 1-CKA | y@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.07678 | 0.07385 | reliable | reliable |
| BAL-MOM_250000 | 1-CKA | u@Y | 6/32 | 0.188 | [0.085, 0.337] | 0.07678 | 0.08756 | unreliable | unreliable |
| BAL-MOM_250000 | 1-CKA | uc@Y | 19/32 | 0.594 | [0.433, 0.740] | 0.07678 | 0.08332 | unreliable | unreliable |
| BAL-MOM_250000 | 1-CKA | ubal@Y | 21/32 | 0.656 | [0.496, 0.794] | 0.07678 | 0.08139 | unreliable | unreliable |
| BAL-MOM_250000 | 1-CKA | yM@Y | 29/32 | 0.906 | [0.775, 0.974] | 0.07678 | 0.07890 | inconclusive | inconclusive |
| BAL-MOM_250000 | 1-CKA | yI@Y | 27/32 | 0.844 | [0.699, 0.936] | 0.07678 | 0.07829 | inconclusive | inconclusive |
| BAL-MOM_250000 | 1-CKA | h@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.07678 | 0.05761 | reliable | reliable |
| BAL-MOM_250000 | 1-CKA | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.08670 | 0.07385 | reliable | reliable |
| BAL-MOM_250000 | 1-CKA | u@Pool | 31/32 | 0.969 | [0.860, 0.998] | 0.08670 | 0.08756 | inconclusive | inconclusive |
| BAL-MOM_250000 | 1-CKA | uc@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.08670 | 0.08332 | reliable | reliable |
| BAL-MOM_250000 | 1-CKA | u@O | 32/32 | 1.000 | [0.911, 1.000] | 0.08863 | 0.08756 | reliable | reliable |
| BAL-MOM_250000 | 1-CKA | uc@C | 32/32 | 1.000 | [0.911, 1.000] | 0.08608 | 0.08332 | reliable | reliable |
| BAL-MOM_250000 | 1-CKA | u@C | 30/32 | 0.938 | [0.816, 0.989] | 0.08608 | 0.08756 | inconclusive | inconclusive |
| BAL-MOM_250000 | 1-CKA | y@Yq90 | 28/32 | 0.875 | [0.736, 0.956] | 0.07188 | 0.07385 | inconclusive | inconclusive |
| BAL-MOM_250000 | 1-CKA | u@Yq90 | 1/32 | 0.031 | [0.002, 0.140] | 0.07188 | 0.08756 | unreliable | inconclusive (RG) |
| BAL-MOM_250000 | 1-CKA | h@h | 32/32 | 1.000 | [0.911, 1.000] | 0.05976 | 0.05761 | reliable | reliable |
| SMALL-MOM_25000 | T_N(10) | y@Y | 31/32 | 0.969 | [0.860, 0.998] | 0.91514 | 0.91694 | inconclusive | inconclusive |
| SMALL-MOM_25000 | T_N(10) | u@Y | 31/32 | 0.969 | [0.860, 0.998] | 0.91514 | 0.91763 | inconclusive | inconclusive (RG) |
| SMALL-MOM_25000 | T_N(10) | uc@Y | 31/32 | 0.969 | [0.860, 0.998] | 0.91514 | 0.91719 | inconclusive | inconclusive (RG) |
| SMALL-MOM_25000 | T_N(10) | ubal@Y | 30/32 | 0.938 | [0.816, 0.989] | 0.91514 | 0.91782 | inconclusive | inconclusive (RG) |
| SMALL-MOM_25000 | T_N(10) | yM@Y | 30/32 | 0.938 | [0.816, 0.989] | 0.91514 | 0.91738 | inconclusive | inconclusive (RG) |
| SMALL-MOM_25000 | T_N(10) | yI@Y | 31/32 | 0.969 | [0.860, 0.998] | 0.91514 | 0.91675 | inconclusive | inconclusive (RG) |
| SMALL-MOM_25000 | T_N(10) | h@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91514 | 0.85708 | reliable | inconclusive (RG) |
| SMALL-MOM_25000 | T_N(10) | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.92153 | 0.91694 | reliable | reliable |
| SMALL-MOM_25000 | T_N(10) | u@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.92153 | 0.91763 | reliable | reliable |
| SMALL-MOM_25000 | T_N(10) | uc@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.92153 | 0.91719 | reliable | reliable |
| SMALL-MOM_25000 | T_N(10) | u@O | 29/32 | 0.906 | [0.775, 0.974] | 0.91377 | 0.91763 | inconclusive | inconclusive |
| SMALL-MOM_25000 | T_N(10) | uc@C | 31/32 | 0.969 | [0.860, 0.998] | 0.91572 | 0.91719 | inconclusive | inconclusive |
| SMALL-MOM_25000 | T_N(10) | u@C | 31/32 | 0.969 | [0.860, 0.998] | 0.91572 | 0.91763 | inconclusive | inconclusive (RG) |
| SMALL-MOM_25000 | T_N(10) | y@Yq90 | 29/32 | 0.906 | [0.775, 0.974] | 0.91409 | 0.91694 | inconclusive | inconclusive |
| SMALL-MOM_25000 | T_N(10) | u@Yq90 | 30/32 | 0.938 | [0.816, 0.989] | 0.91409 | 0.91763 | inconclusive | inconclusive (RG) |
| SMALL-MOM_25000 | T_N(10) | h@h | 28/32 | 0.875 | [0.736, 0.956] | 0.85444 | 0.85708 | inconclusive | inconclusive |
| SMALL-MOM_25000 | G_s | y@Y | 31/32 | 0.969 | [0.860, 0.998] | 0.28744 | 0.28750 | inconclusive | inconclusive |
| SMALL-MOM_25000 | G_s | u@Y | 1/32 | 0.031 | [0.002, 0.140] | 0.28744 | 0.30927 | unreliable | inconclusive (RG) |
| SMALL-MOM_25000 | G_s | uc@Y | 3/32 | 0.094 | [0.026, 0.225] | 0.28744 | 0.30153 | unreliable | inconclusive (RG) |
| SMALL-MOM_25000 | G_s | ubal@Y | 10/32 | 0.312 | [0.180, 0.472] | 0.28744 | 0.30275 | unreliable | inconclusive (RG) |
| SMALL-MOM_25000 | G_s | yM@Y | 24/32 | 0.750 | [0.594, 0.869] | 0.28744 | 0.29621 | inconclusive | inconclusive (RG) |
| SMALL-MOM_25000 | G_s | yI@Y | 23/32 | 0.719 | [0.561, 0.845] | 0.28744 | 0.29484 | inconclusive | inconclusive (RG) |
| SMALL-MOM_25000 | G_s | h@Y | 0/32 | 0.000 | [0.000, 0.089] | 0.28744 | 0.42177 | unreliable | inconclusive (RG) |
| SMALL-MOM_25000 | G_s | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.29467 | 0.28750 | reliable | reliable |
| SMALL-MOM_25000 | G_s | u@Pool | 13/32 | 0.406 | [0.260, 0.567] | 0.29467 | 0.30927 | unreliable | unreliable |
| SMALL-MOM_25000 | G_s | uc@Pool | 24/32 | 0.750 | [0.594, 0.869] | 0.29467 | 0.30153 | inconclusive | inconclusive |
| SMALL-MOM_25000 | G_s | u@O | 30/32 | 0.938 | [0.816, 0.989] | 0.30492 | 0.30927 | inconclusive | inconclusive |
| SMALL-MOM_25000 | G_s | uc@C | 32/32 | 1.000 | [0.911, 1.000] | 0.30165 | 0.30153 | reliable | reliable |
| SMALL-MOM_25000 | G_s | u@C | 24/32 | 0.750 | [0.594, 0.869] | 0.30165 | 0.30927 | inconclusive | inconclusive |
| SMALL-MOM_25000 | G_s | y@Yq90 | 29/32 | 0.906 | [0.775, 0.974] | 0.28366 | 0.28750 | inconclusive | inconclusive |
| SMALL-MOM_25000 | G_s | u@Yq90 | 0/32 | 0.000 | [0.000, 0.089] | 0.28366 | 0.30927 | unreliable | inconclusive (RG) |
| SMALL-MOM_25000 | G_s | h@h | 29/32 | 0.906 | [0.775, 0.974] | 0.39915 | 0.42177 | inconclusive | inconclusive |
| SMALL-MOM_25000 | 1-CKA | y@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.15856 | 0.15848 | reliable | reliable |
| SMALL-MOM_25000 | 1-CKA | u@Y | 22/32 | 0.688 | [0.528, 0.820] | 0.15856 | 0.16979 | inconclusive | inconclusive |
| SMALL-MOM_25000 | 1-CKA | uc@Y | 29/32 | 0.906 | [0.775, 0.974] | 0.15856 | 0.16053 | inconclusive | inconclusive |
| SMALL-MOM_25000 | 1-CKA | ubal@Y | 24/32 | 0.750 | [0.594, 0.869] | 0.15856 | 0.16532 | inconclusive | inconclusive |
| SMALL-MOM_25000 | 1-CKA | yM@Y | 24/32 | 0.750 | [0.594, 0.869] | 0.15856 | 0.16618 | inconclusive | inconclusive |
| SMALL-MOM_25000 | 1-CKA | yI@Y | 28/32 | 0.875 | [0.736, 0.956] | 0.15856 | 0.16140 | inconclusive | inconclusive |
| SMALL-MOM_25000 | 1-CKA | h@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.15856 | 0.10293 | reliable | reliable |
| SMALL-MOM_25000 | 1-CKA | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.16885 | 0.15848 | reliable | reliable |
| SMALL-MOM_25000 | 1-CKA | u@Pool | 31/32 | 0.969 | [0.860, 0.998] | 0.16885 | 0.16979 | inconclusive | inconclusive |
| SMALL-MOM_25000 | 1-CKA | uc@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.16885 | 0.16053 | reliable | reliable |
| SMALL-MOM_25000 | 1-CKA | u@O | 29/32 | 0.906 | [0.775, 0.974] | 0.16371 | 0.16979 | inconclusive | inconclusive |
| SMALL-MOM_25000 | 1-CKA | uc@C | 31/32 | 0.969 | [0.860, 0.998] | 0.15976 | 0.16053 | inconclusive | inconclusive |
| SMALL-MOM_25000 | 1-CKA | u@C | 25/32 | 0.781 | [0.628, 0.893] | 0.15976 | 0.16979 | inconclusive | inconclusive (RG) |
| SMALL-MOM_25000 | 1-CKA | y@Yq90 | 29/32 | 0.906 | [0.775, 0.974] | 0.15544 | 0.15848 | inconclusive | inconclusive |
| SMALL-MOM_25000 | 1-CKA | u@Yq90 | 14/32 | 0.438 | [0.287, 0.597] | 0.15544 | 0.16979 | unreliable | inconclusive (RG) |
| SMALL-MOM_25000 | 1-CKA | h@h | 32/32 | 1.000 | [0.911, 1.000] | 0.10955 | 0.10293 | reliable | reliable |
| SMALL-MOM_250000 | T_N(10) | y@Y | 26/32 | 0.812 | [0.663, 0.915] | 0.81689 | 0.82339 | inconclusive | inconclusive |
| SMALL-MOM_250000 | T_N(10) | u@Y | 26/32 | 0.812 | [0.663, 0.915] | 0.81689 | 0.82275 | inconclusive | inconclusive (RG) |
| SMALL-MOM_250000 | T_N(10) | uc@Y | 26/32 | 0.812 | [0.663, 0.915] | 0.81689 | 0.82363 | inconclusive | inconclusive (RG) |
| SMALL-MOM_250000 | T_N(10) | ubal@Y | 27/32 | 0.844 | [0.699, 0.936] | 0.81689 | 0.82319 | inconclusive | inconclusive (RG) |
| SMALL-MOM_250000 | T_N(10) | yM@Y | 25/32 | 0.781 | [0.628, 0.893] | 0.81689 | 0.82378 | inconclusive | inconclusive (RG) |
| SMALL-MOM_250000 | T_N(10) | yI@Y | 24/32 | 0.750 | [0.594, 0.869] | 0.81689 | 0.82563 | inconclusive | inconclusive (RG) |
| SMALL-MOM_250000 | T_N(10) | h@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.81689 | 0.72910 | reliable | inconclusive (RG) |
| SMALL-MOM_250000 | T_N(10) | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.83271 | 0.82339 | reliable | reliable |
| SMALL-MOM_250000 | T_N(10) | u@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.83271 | 0.82275 | reliable | reliable |
| SMALL-MOM_250000 | T_N(10) | uc@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.83271 | 0.82363 | reliable | reliable |
| SMALL-MOM_250000 | T_N(10) | u@O | 27/32 | 0.844 | [0.699, 0.936] | 0.81807 | 0.82275 | inconclusive | inconclusive |
| SMALL-MOM_250000 | T_N(10) | uc@C | 27/32 | 0.844 | [0.699, 0.936] | 0.81782 | 0.82363 | inconclusive | inconclusive |
| SMALL-MOM_250000 | T_N(10) | u@C | 26/32 | 0.812 | [0.663, 0.915] | 0.81782 | 0.82275 | inconclusive | inconclusive (RG) |
| SMALL-MOM_250000 | T_N(10) | y@Yq90 | 23/32 | 0.719 | [0.561, 0.845] | 0.81492 | 0.82339 | inconclusive | inconclusive |
| SMALL-MOM_250000 | T_N(10) | u@Yq90 | 24/32 | 0.750 | [0.594, 0.869] | 0.81492 | 0.82275 | inconclusive | inconclusive (RG) |
| SMALL-MOM_250000 | T_N(10) | h@h | 23/32 | 0.719 | [0.561, 0.845] | 0.71626 | 0.72910 | inconclusive | inconclusive |
| SMALL-MOM_250000 | G_s | y@Y | 31/32 | 0.969 | [0.860, 0.998] | 0.19566 | 0.19812 | inconclusive | inconclusive |
| SMALL-MOM_250000 | G_s | u@Y | 5/32 | 0.156 | [0.064, 0.301] | 0.19566 | 0.21357 | unreliable | inconclusive (RG) |
| SMALL-MOM_250000 | G_s | uc@Y | 21/32 | 0.656 | [0.496, 0.794] | 0.19566 | 0.21134 | unreliable | inconclusive (RG) |
| SMALL-MOM_250000 | G_s | ubal@Y | 22/32 | 0.688 | [0.528, 0.820] | 0.19566 | 0.20796 | inconclusive | inconclusive (RG) |
| SMALL-MOM_250000 | G_s | yM@Y | 19/32 | 0.594 | [0.433, 0.740] | 0.19566 | 0.20369 | unreliable | inconclusive (RG) |
| SMALL-MOM_250000 | G_s | yI@Y | 19/32 | 0.594 | [0.433, 0.740] | 0.19566 | 0.20457 | unreliable | inconclusive (RG) |
| SMALL-MOM_250000 | G_s | h@Y | 0/32 | 0.000 | [0.000, 0.089] | 0.19566 | 0.29250 | unreliable | inconclusive (RG) |
| SMALL-MOM_250000 | G_s | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.20985 | 0.19812 | reliable | reliable |
| SMALL-MOM_250000 | G_s | u@Pool | 28/32 | 0.875 | [0.736, 0.956] | 0.20985 | 0.21357 | inconclusive | inconclusive |
| SMALL-MOM_250000 | G_s | uc@Pool | 31/32 | 0.969 | [0.860, 0.998] | 0.20985 | 0.21134 | inconclusive | inconclusive |
| SMALL-MOM_250000 | G_s | u@O | 25/32 | 0.781 | [0.628, 0.893] | 0.20741 | 0.21357 | inconclusive | inconclusive |
| SMALL-MOM_250000 | G_s | uc@C | 31/32 | 0.969 | [0.860, 0.998] | 0.20450 | 0.21134 | inconclusive | inconclusive |
| SMALL-MOM_250000 | G_s | u@C | 20/32 | 0.625 | [0.464, 0.767] | 0.20450 | 0.21357 | unreliable | inconclusive (RG) |
| SMALL-MOM_250000 | G_s | y@Yq90 | 28/32 | 0.875 | [0.736, 0.956] | 0.19261 | 0.19812 | inconclusive | inconclusive |
| SMALL-MOM_250000 | G_s | u@Yq90 | 3/32 | 0.094 | [0.026, 0.225] | 0.19261 | 0.21357 | unreliable | inconclusive (RG) |
| SMALL-MOM_250000 | G_s | h@h | 32/32 | 1.000 | [0.911, 1.000] | 0.29891 | 0.29250 | reliable | reliable |
| SMALL-MOM_250000 | 1-CKA | y@Y | 29/32 | 0.906 | [0.775, 0.974] | 0.07431 | 0.07605 | inconclusive | inconclusive |
| SMALL-MOM_250000 | 1-CKA | u@Y | 7/32 | 0.219 | [0.107, 0.372] | 0.07431 | 0.09168 | unreliable | inconclusive (RG) |
| SMALL-MOM_250000 | 1-CKA | uc@Y | 24/32 | 0.750 | [0.594, 0.869] | 0.07431 | 0.08582 | inconclusive | inconclusive (RG) |
| SMALL-MOM_250000 | 1-CKA | ubal@Y | 18/32 | 0.562 | [0.403, 0.713] | 0.07431 | 0.08426 | unreliable | inconclusive (RG) |
| SMALL-MOM_250000 | 1-CKA | yM@Y | 19/32 | 0.594 | [0.433, 0.740] | 0.07431 | 0.08665 | unreliable | inconclusive (RG) |
| SMALL-MOM_250000 | 1-CKA | yI@Y | 23/32 | 0.719 | [0.561, 0.845] | 0.07431 | 0.08316 | inconclusive | inconclusive (RG) |
| SMALL-MOM_250000 | 1-CKA | h@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.07431 | 0.06405 | reliable | inconclusive (RG) |
| SMALL-MOM_250000 | 1-CKA | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.08670 | 0.07605 | reliable | reliable |
| SMALL-MOM_250000 | 1-CKA | u@Pool | 31/32 | 0.969 | [0.860, 0.998] | 0.08670 | 0.09168 | inconclusive | inconclusive |
| SMALL-MOM_250000 | 1-CKA | uc@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.08670 | 0.08582 | reliable | reliable |
| SMALL-MOM_250000 | 1-CKA | u@O | 29/32 | 0.906 | [0.775, 0.974] | 0.08521 | 0.09168 | inconclusive | inconclusive |
| SMALL-MOM_250000 | 1-CKA | uc@C | 31/32 | 0.969 | [0.860, 0.998] | 0.08094 | 0.08582 | inconclusive | inconclusive |
| SMALL-MOM_250000 | 1-CKA | u@C | 25/32 | 0.781 | [0.628, 0.893] | 0.08094 | 0.09168 | inconclusive | inconclusive (RG) |
| SMALL-MOM_250000 | 1-CKA | y@Yq90 | 26/32 | 0.812 | [0.663, 0.915] | 0.07221 | 0.07605 | inconclusive | inconclusive |
| SMALL-MOM_250000 | 1-CKA | u@Yq90 | 3/32 | 0.094 | [0.026, 0.225] | 0.07221 | 0.09168 | unreliable | inconclusive (RG) |
| SMALL-MOM_250000 | 1-CKA | h@h | 32/32 | 1.000 | [0.911, 1.000] | 0.06623 | 0.06405 | reliable | reliable |
| SGD-K_25000 | T_N(10) | y@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91699 | 0.91592 | reliable | reliable |
| SGD-K_25000 | T_N(10) | u@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91699 | 0.91514 | reliable | reliable |
| SGD-K_25000 | T_N(10) | uc@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91699 | 0.91616 | reliable | reliable |
| SGD-K_25000 | T_N(10) | ubal@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91699 | 0.91563 | reliable | reliable |
| SGD-K_25000 | T_N(10) | yM@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91699 | 0.91562 | reliable | reliable |
| SGD-K_25000 | T_N(10) | yI@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91699 | 0.91538 | reliable | reliable |
| SGD-K_25000 | T_N(10) | h@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.91699 | 0.84971 | reliable | reliable |
| SGD-K_25000 | T_N(10) | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.92153 | 0.91592 | reliable | reliable |
| SGD-K_25000 | T_N(10) | u@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.92153 | 0.91514 | reliable | reliable |
| SGD-K_25000 | T_N(10) | uc@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.92153 | 0.91616 | reliable | reliable |
| SGD-K_25000 | T_N(10) | u@O | 32/32 | 1.000 | [0.911, 1.000] | 0.91782 | 0.91514 | reliable | reliable |
| SGD-K_25000 | T_N(10) | uc@C | 32/32 | 1.000 | [0.911, 1.000] | 0.91772 | 0.91616 | reliable | reliable |
| SGD-K_25000 | T_N(10) | u@C | 32/32 | 1.000 | [0.911, 1.000] | 0.91772 | 0.91514 | reliable | reliable |
| SGD-K_25000 | T_N(10) | y@Yq90 | 28/32 | 0.875 | [0.736, 0.956] | 0.91272 | 0.91592 | inconclusive | inconclusive |
| SGD-K_25000 | T_N(10) | u@Yq90 | 29/32 | 0.906 | [0.775, 0.974] | 0.91272 | 0.91514 | inconclusive | inconclusive (RG) |
| SGD-K_25000 | T_N(10) | h@h | 32/32 | 1.000 | [0.911, 1.000] | 0.85869 | 0.84971 | reliable | reliable |
| SGD-K_25000 | G_s | y@Y | 30/32 | 0.938 | [0.816, 0.989] | 0.29467 | 0.29713 | inconclusive | inconclusive |
| SGD-K_25000 | G_s | u@Y | 0/32 | 0.000 | [0.000, 0.089] | 0.29467 | 0.33842 | unreliable | inconclusive (RG) |
| SGD-K_25000 | G_s | uc@Y | 0/32 | 0.000 | [0.000, 0.089] | 0.29467 | 0.33348 | unreliable | inconclusive (RG) |
| SGD-K_25000 | G_s | ubal@Y | 13/32 | 0.406 | [0.260, 0.567] | 0.29467 | 0.31033 | unreliable | inconclusive (RG) |
| SGD-K_25000 | G_s | yM@Y | 26/32 | 0.812 | [0.663, 0.915] | 0.29467 | 0.30453 | inconclusive | inconclusive (RG) |
| SGD-K_25000 | G_s | yI@Y | 0/32 | 0.000 | [0.000, 0.089] | 0.29467 | 0.32522 | unreliable | inconclusive (RG) |
| SGD-K_25000 | G_s | h@Y | 4/32 | 0.125 | [0.044, 0.264] | 0.29467 | 0.41301 | unreliable | inconclusive (RG) |
| SGD-K_25000 | G_s | y@Pool | 30/32 | 0.938 | [0.816, 0.989] | 0.29467 | 0.29713 | inconclusive | inconclusive |
| SGD-K_25000 | G_s | u@Pool | 0/32 | 0.000 | [0.000, 0.089] | 0.29467 | 0.33842 | unreliable | inconclusive (RG) |
| SGD-K_25000 | G_s | uc@Pool | 0/32 | 0.000 | [0.000, 0.089] | 0.29467 | 0.33348 | unreliable | inconclusive (RG) |
| SGD-K_25000 | G_s | u@O | 30/32 | 0.938 | [0.816, 0.989] | 0.32911 | 0.33842 | inconclusive | inconclusive |
| SGD-K_25000 | G_s | uc@C | 29/32 | 0.906 | [0.775, 0.974] | 0.32316 | 0.33348 | inconclusive | inconclusive |
| SGD-K_25000 | G_s | u@C | 28/32 | 0.875 | [0.736, 0.956] | 0.32316 | 0.33842 | inconclusive | inconclusive (RG) |
| SGD-K_25000 | G_s | y@Yq90 | 29/32 | 0.906 | [0.775, 0.974] | 0.29383 | 0.29713 | inconclusive | inconclusive |
| SGD-K_25000 | G_s | u@Yq90 | 0/32 | 0.000 | [0.000, 0.089] | 0.29383 | 0.33842 | unreliable | inconclusive (RG) |
| SGD-K_25000 | G_s | h@h | 32/32 | 1.000 | [0.911, 1.000] | 0.43086 | 0.41301 | reliable | reliable |
| SGD-K_25000 | 1-CKA | y@Y | 28/32 | 0.875 | [0.736, 0.956] | 0.15775 | 0.16176 | inconclusive | inconclusive |
| SGD-K_25000 | 1-CKA | u@Y | 0/32 | 0.000 | [0.000, 0.089] | 0.15775 | 0.18449 | unreliable | inconclusive (RG) |
| SGD-K_25000 | 1-CKA | uc@Y | 1/32 | 0.031 | [0.002, 0.140] | 0.15775 | 0.17726 | unreliable | inconclusive (RG) |
| SGD-K_25000 | 1-CKA | ubal@Y | 23/32 | 0.719 | [0.561, 0.845] | 0.15775 | 0.16628 | inconclusive | inconclusive (RG) |
| SGD-K_25000 | 1-CKA | yM@Y | 26/32 | 0.812 | [0.663, 0.915] | 0.15775 | 0.16673 | inconclusive | inconclusive (RG) |
| SGD-K_25000 | 1-CKA | yI@Y | 0/32 | 0.000 | [0.000, 0.089] | 0.15775 | 0.17771 | unreliable | inconclusive (RG) |
| SGD-K_25000 | 1-CKA | h@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.15775 | 0.13867 | reliable | inconclusive (RG) |
| SGD-K_25000 | 1-CKA | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.16885 | 0.16176 | reliable | reliable |
| SGD-K_25000 | 1-CKA | u@Pool | 17/32 | 0.531 | [0.373, 0.685] | 0.16885 | 0.18449 | unreliable | unreliable |
| SGD-K_25000 | 1-CKA | uc@Pool | 25/32 | 0.781 | [0.628, 0.893] | 0.16885 | 0.17726 | inconclusive | inconclusive |
| SGD-K_25000 | 1-CKA | u@O | 30/32 | 0.938 | [0.816, 0.989] | 0.17948 | 0.18449 | inconclusive | inconclusive |
| SGD-K_25000 | 1-CKA | uc@C | 28/32 | 0.875 | [0.736, 0.956] | 0.17150 | 0.17726 | inconclusive | inconclusive |
| SGD-K_25000 | 1-CKA | u@C | 27/32 | 0.844 | [0.699, 0.936] | 0.17150 | 0.18449 | inconclusive | inconclusive (RG) |
| SGD-K_25000 | 1-CKA | y@Yq90 | 27/32 | 0.844 | [0.699, 0.936] | 0.15487 | 0.16176 | inconclusive | inconclusive |
| SGD-K_25000 | 1-CKA | u@Yq90 | 0/32 | 0.000 | [0.000, 0.089] | 0.15487 | 0.18449 | unreliable | inconclusive (RG) |
| SGD-K_25000 | 1-CKA | h@h | 32/32 | 1.000 | [0.911, 1.000] | 0.14267 | 0.13867 | reliable | reliable |
| SGD-K_250000 | T_N(10) | y@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.99434 | 0.81948 | reliable | reliable |
| SGD-K_250000 | T_N(10) | u@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.99434 | 0.81978 | reliable | reliable |
| SGD-K_250000 | T_N(10) | uc@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.99434 | 0.82031 | reliable | reliable |
| SGD-K_250000 | T_N(10) | ubal@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.99434 | 0.81885 | reliable | reliable |
| SGD-K_250000 | T_N(10) | yM@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.99434 | 0.81899 | reliable | reliable |
| SGD-K_250000 | T_N(10) | yI@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.99434 | 0.82051 | reliable | reliable |
| SGD-K_250000 | T_N(10) | h@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.99434 | 0.71245 | reliable | reliable |
| SGD-K_250000 | T_N(10) | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.83271 | 0.81948 | reliable | reliable |
| SGD-K_250000 | T_N(10) | u@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.83271 | 0.81978 | reliable | reliable |
| SGD-K_250000 | T_N(10) | uc@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.83271 | 0.82031 | reliable | reliable |
| SGD-K_250000 | T_N(10) | u@O | 31/32 | 0.969 | [0.860, 0.998] | 0.81826 | 0.81978 | inconclusive | inconclusive |
| SGD-K_250000 | T_N(10) | uc@C | 32/32 | 1.000 | [0.911, 1.000] | 0.99360 | 0.82031 | reliable | reliable |
| SGD-K_250000 | T_N(10) | u@C | 32/32 | 1.000 | [0.911, 1.000] | 0.99360 | 0.81978 | reliable | reliable |
| SGD-K_250000 | T_N(10) | y@Yq90 | 32/32 | 1.000 | [0.911, 1.000] | 0.81965 | 0.81948 | reliable | reliable |
| SGD-K_250000 | T_N(10) | u@Yq90 | 31/32 | 0.969 | [0.860, 0.998] | 0.81965 | 0.81978 | inconclusive | inconclusive |
| SGD-K_250000 | T_N(10) | h@h | 27/32 | 0.844 | [0.699, 0.936] | 0.70542 | 0.71245 | inconclusive | inconclusive |
| SGD-K_250000 | G_s | y@Y | 32/32 | 1.000 | [0.911, 1.000] | 1.17234 | 0.19960 | reliable | reliable |
| SGD-K_250000 | G_s | u@Y | 32/32 | 1.000 | [0.911, 1.000] | 1.17234 | 0.22347 | reliable | reliable |
| SGD-K_250000 | G_s | uc@Y | 32/32 | 1.000 | [0.911, 1.000] | 1.17234 | 0.21759 | reliable | reliable |
| SGD-K_250000 | G_s | ubal@Y | 32/32 | 1.000 | [0.911, 1.000] | 1.17234 | 0.20647 | reliable | reliable |
| SGD-K_250000 | G_s | yM@Y | 32/32 | 1.000 | [0.911, 1.000] | 1.17234 | 0.20366 | reliable | reliable |
| SGD-K_250000 | G_s | yI@Y | 32/32 | 1.000 | [0.911, 1.000] | 1.17234 | 0.21587 | reliable | reliable |
| SGD-K_250000 | G_s | h@Y | 32/32 | 1.000 | [0.911, 1.000] | 1.17234 | 0.27398 | reliable | reliable |
| SGD-K_250000 | G_s | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.20985 | 0.19960 | reliable | reliable |
| SGD-K_250000 | G_s | u@Pool | 18/32 | 0.562 | [0.403, 0.713] | 0.20985 | 0.22347 | unreliable | unreliable |
| SGD-K_250000 | G_s | uc@Pool | 20/32 | 0.625 | [0.464, 0.767] | 0.20985 | 0.21759 | unreliable | unreliable |
| SGD-K_250000 | G_s | u@O | 32/32 | 1.000 | [0.911, 1.000] | 0.22374 | 0.22347 | reliable | reliable |
| SGD-K_250000 | G_s | uc@C | 32/32 | 1.000 | [0.911, 1.000] | 1.18756 | 0.21759 | reliable | reliable |
| SGD-K_250000 | G_s | u@C | 32/32 | 1.000 | [0.911, 1.000] | 1.18756 | 0.22347 | reliable | reliable |
| SGD-K_250000 | G_s | y@Yq90 | 32/32 | 1.000 | [0.911, 1.000] | 0.20163 | 0.19960 | reliable | reliable |
| SGD-K_250000 | G_s | u@Yq90 | 0/32 | 0.000 | [0.000, 0.089] | 0.20163 | 0.22347 | unreliable | unreliable |
| SGD-K_250000 | G_s | h@h | 31/32 | 0.969 | [0.860, 0.998] | 0.24618 | 0.27398 | inconclusive | inconclusive |
| SGD-K_250000 | 1-CKA | y@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.95565 | 0.07474 | reliable | reliable |
| SGD-K_250000 | 1-CKA | u@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.95565 | 0.08850 | reliable | reliable |
| SGD-K_250000 | 1-CKA | uc@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.95565 | 0.08625 | reliable | reliable |
| SGD-K_250000 | 1-CKA | ubal@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.95565 | 0.08269 | reliable | reliable |
| SGD-K_250000 | 1-CKA | yM@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.95565 | 0.07699 | reliable | reliable |
| SGD-K_250000 | 1-CKA | yI@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.95565 | 0.08055 | reliable | reliable |
| SGD-K_250000 | 1-CKA | h@Y | 32/32 | 1.000 | [0.911, 1.000] | 0.95565 | 0.07781 | reliable | reliable |
| SGD-K_250000 | 1-CKA | y@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.08670 | 0.07474 | reliable | reliable |
| SGD-K_250000 | 1-CKA | u@Pool | 31/32 | 0.969 | [0.860, 0.998] | 0.08670 | 0.08850 | inconclusive | inconclusive |
| SGD-K_250000 | 1-CKA | uc@Pool | 32/32 | 1.000 | [0.911, 1.000] | 0.08670 | 0.08625 | reliable | reliable |
| SGD-K_250000 | 1-CKA | u@O | 32/32 | 1.000 | [0.911, 1.000] | 0.09062 | 0.08850 | reliable | reliable |
| SGD-K_250000 | 1-CKA | uc@C | 32/32 | 1.000 | [0.911, 1.000] | 0.96986 | 0.08625 | reliable | reliable |
| SGD-K_250000 | 1-CKA | u@C | 32/32 | 1.000 | [0.911, 1.000] | 0.96986 | 0.08850 | reliable | reliable |
| SGD-K_250000 | 1-CKA | y@Yq90 | 32/32 | 1.000 | [0.911, 1.000] | 0.07702 | 0.07474 | reliable | reliable |
| SGD-K_250000 | 1-CKA | u@Yq90 | 11/32 | 0.344 | [0.206, 0.504] | 0.07702 | 0.08850 | unreliable | unreliable |
| SGD-K_250000 | 1-CKA | h@h | 31/32 | 0.969 | [0.860, 0.998] | 0.07135 | 0.07781 | inconclusive | inconclusive |

## 10. p̂(ε) curves (selected points; full curves on ε/ε_anchor ∈ [0.80, 1.25] step 0.01 in `render/phat_curves.csv`)

| Cell | Q | entry | ×0.8 | ×0.9 | ×0.95 | ×1.0 | ×1.05 | ×1.1 | ×1.25 |
|---|---|---|---|---|---|---|---|---|---|
| MOM_25000 | G_s | y@Y | 0.000 | 0.000 | 0.083 | 1.000 | 1.000 | 1.000 | 1.000 |
| MOM_25000 | G_s | u@Y | 0.000 | 0.000 | 0.000 | 0.000 | 0.417 | 0.990 | 1.000 |
| MOM_25000 | G_s | uc@Y | 0.000 | 0.000 | 0.000 | 0.010 | 0.833 | 1.000 | 1.000 |
| MOM_25000 | G_s | ubal@Y | 0.000 | 0.000 | 0.000 | 0.188 | 0.979 | 1.000 | 1.000 |
| MOM_25000 | G_s | u@Pool | 0.000 | 0.000 | 0.000 | 0.198 | 0.958 | 1.000 | 1.000 |
| MOM_25000 | G_s | uc@Pool | 0.000 | 0.000 | 0.000 | 0.490 | 1.000 | 1.000 | 1.000 |
| MOM_25000 | 1-CKA | y@Y | 0.000 | 0.021 | 0.771 | 1.000 | 1.000 | 1.000 | 1.000 |
| MOM_25000 | 1-CKA | u@Y | 0.000 | 0.000 | 0.010 | 0.375 | 0.896 | 0.990 | 1.000 |
| MOM_25000 | 1-CKA | uc@Y | 0.000 | 0.000 | 0.062 | 0.771 | 0.990 | 1.000 | 1.000 |
| MOM_25000 | 1-CKA | ubal@Y | 0.000 | 0.000 | 0.115 | 0.760 | 0.979 | 1.000 | 1.000 |
| MOM_25000 | 1-CKA | u@Pool | 0.000 | 0.000 | 0.104 | 0.771 | 0.969 | 1.000 | 1.000 |
| MOM_25000 | 1-CKA | uc@Pool | 0.000 | 0.000 | 0.406 | 0.938 | 1.000 | 1.000 | 1.000 |
| MOM_250000 | G_s | y@Y | 0.000 | 0.000 | 0.573 | 1.000 | 1.000 | 1.000 | 1.000 |
| MOM_250000 | G_s | u@Y | 0.000 | 0.000 | 0.000 | 0.052 | 0.552 | 0.979 | 1.000 |
| MOM_250000 | G_s | uc@Y | 0.000 | 0.000 | 0.000 | 0.219 | 0.896 | 1.000 | 1.000 |
| MOM_250000 | G_s | ubal@Y | 0.000 | 0.000 | 0.031 | 0.708 | 0.990 | 1.000 | 1.000 |
| MOM_250000 | G_s | u@Pool | 0.000 | 0.000 | 0.156 | 0.906 | 1.000 | 1.000 | 1.000 |
| MOM_250000 | G_s | uc@Pool | 0.000 | 0.010 | 0.615 | 0.990 | 1.000 | 1.000 | 1.000 |
| MOM_250000 | 1-CKA | y@Y | 0.000 | 0.625 | 0.938 | 1.000 | 1.000 | 1.000 | 1.000 |
| MOM_250000 | 1-CKA | u@Y | 0.000 | 0.000 | 0.021 | 0.240 | 0.615 | 0.938 | 1.000 |
| MOM_250000 | 1-CKA | uc@Y | 0.000 | 0.000 | 0.240 | 0.688 | 0.958 | 1.000 | 1.000 |
| MOM_250000 | 1-CKA | ubal@Y | 0.000 | 0.021 | 0.198 | 0.698 | 0.969 | 1.000 | 1.000 |
| MOM_250000 | 1-CKA | u@Pool | 0.000 | 0.250 | 0.719 | 0.969 | 1.000 | 1.000 | 1.000 |
| MOM_250000 | 1-CKA | uc@Pool | 0.000 | 0.688 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_25000 | G_s | y@Y | 0.000 | 0.000 | 0.156 | 0.979 | 1.000 | 1.000 | 1.000 |
| WD_25000 | G_s | u@Y | 0.000 | 0.000 | 0.438 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_25000 | G_s | uc@Y | 0.000 | 0.000 | 0.521 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_25000 | G_s | ubal@Y | 0.000 | 0.000 | 0.552 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_25000 | G_s | u@Pool | 0.000 | 0.000 | 0.740 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_25000 | G_s | uc@Pool | 0.000 | 0.000 | 0.812 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_25000 | 1-CKA | y@Y | 0.000 | 0.094 | 0.719 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_25000 | 1-CKA | u@Y | 0.000 | 0.635 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_25000 | 1-CKA | uc@Y | 0.000 | 0.719 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_25000 | 1-CKA | ubal@Y | 0.000 | 0.771 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_25000 | 1-CKA | u@Pool | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_25000 | 1-CKA | uc@Pool | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_250000 | G_s | y@Y | 0.000 | 0.292 | 0.948 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_250000 | G_s | u@Y | 0.000 | 0.385 | 0.958 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_250000 | G_s | uc@Y | 0.000 | 0.323 | 0.958 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_250000 | G_s | ubal@Y | 0.000 | 0.406 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_250000 | G_s | u@Pool | 0.000 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_250000 | G_s | uc@Pool | 0.000 | 0.854 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_250000 | 1-CKA | y@Y | 0.688 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_250000 | 1-CKA | u@Y | 0.323 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_250000 | 1-CKA | uc@Y | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_250000 | 1-CKA | ubal@Y | 0.417 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_250000 | 1-CKA | u@Pool | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WD_250000 | 1-CKA | uc@Pool | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| FLOW_25000 | G_s | y@Y | 0.000 | 0.000 | 0.062 | 0.990 | 1.000 | 1.000 | 1.000 |
| FLOW_25000 | G_s | u@Y | 0.000 | 0.000 | 0.000 | 0.010 | 0.677 | 0.990 | 1.000 |
| FLOW_25000 | G_s | uc@Y | 0.000 | 0.000 | 0.000 | 0.052 | 0.958 | 1.000 | 1.000 |
| FLOW_25000 | G_s | ubal@Y | 0.000 | 0.000 | 0.000 | 0.031 | 0.812 | 1.000 | 1.000 |
| FLOW_25000 | G_s | u@Pool | 0.000 | 0.000 | 0.000 | 0.479 | 0.979 | 1.000 | 1.000 |
| FLOW_25000 | G_s | uc@Pool | 0.000 | 0.000 | 0.010 | 0.854 | 1.000 | 1.000 | 1.000 |
| FLOW_25000 | 1-CKA | y@Y | 0.000 | 0.031 | 0.604 | 1.000 | 1.000 | 1.000 | 1.000 |
| FLOW_25000 | 1-CKA | u@Y | 0.000 | 0.000 | 0.010 | 0.260 | 0.875 | 0.979 | 1.000 |
| FLOW_25000 | 1-CKA | uc@Y | 0.000 | 0.000 | 0.115 | 0.740 | 1.000 | 1.000 | 1.000 |
| FLOW_25000 | 1-CKA | ubal@Y | 0.000 | 0.000 | 0.021 | 0.406 | 0.927 | 0.990 | 1.000 |
| FLOW_25000 | 1-CKA | u@Pool | 0.000 | 0.000 | 0.010 | 0.260 | 0.885 | 0.990 | 1.000 |
| FLOW_25000 | 1-CKA | uc@Pool | 0.000 | 0.000 | 0.115 | 0.760 | 1.000 | 1.000 | 1.000 |
| FLOW_250000 | G_s | y@Y | 0.000 | 0.000 | 0.719 | 0.969 | 1.000 | 1.000 | 1.000 |
| FLOW_250000 | G_s | u@Y | 0.000 | 0.000 | 0.000 | 0.333 | 0.917 | 0.990 | 1.000 |
| FLOW_250000 | G_s | uc@Y | 0.000 | 0.000 | 0.094 | 0.875 | 0.990 | 1.000 | 1.000 |
| FLOW_250000 | G_s | ubal@Y | 0.000 | 0.000 | 0.031 | 0.594 | 0.958 | 1.000 | 1.000 |
| FLOW_250000 | G_s | u@Pool | 0.000 | 0.000 | 0.302 | 0.917 | 0.990 | 1.000 | 1.000 |
| FLOW_250000 | G_s | uc@Pool | 0.000 | 0.052 | 0.875 | 0.990 | 1.000 | 1.000 | 1.000 |
| FLOW_250000 | 1-CKA | y@Y | 0.010 | 0.802 | 0.958 | 1.000 | 1.000 | 1.000 | 1.000 |
| FLOW_250000 | 1-CKA | u@Y | 0.000 | 0.031 | 0.135 | 0.573 | 0.885 | 0.969 | 1.000 |
| FLOW_250000 | 1-CKA | uc@Y | 0.000 | 0.260 | 0.802 | 0.969 | 1.000 | 1.000 | 1.000 |
| FLOW_250000 | 1-CKA | ubal@Y | 0.000 | 0.052 | 0.292 | 0.760 | 0.927 | 0.990 | 1.000 |
| FLOW_250000 | 1-CKA | u@Pool | 0.000 | 0.052 | 0.406 | 0.781 | 0.917 | 0.990 | 1.000 |
| FLOW_250000 | 1-CKA | uc@Pool | 0.000 | 0.604 | 0.927 | 0.979 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_25000 | G_s | y@Y | 0.000 | 0.000 | 0.208 | 1.000 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_25000 | G_s | u@Y | 0.000 | 0.000 | 0.000 | 0.208 | 0.927 | 1.000 | 1.000 |
| BAL-FLOW_25000 | G_s | uc@Y | 0.000 | 0.000 | 0.000 | 0.385 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_25000 | G_s | ubal@Y | 0.000 | 0.000 | 0.000 | 0.208 | 0.927 | 1.000 | 1.000 |
| BAL-FLOW_25000 | G_s | u@Pool | 0.000 | 0.000 | 0.031 | 0.708 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_25000 | G_s | uc@Pool | 0.000 | 0.000 | 0.073 | 0.938 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_25000 | 1-CKA | y@Y | 0.000 | 0.042 | 0.656 | 0.979 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_25000 | 1-CKA | u@Y | 0.000 | 0.000 | 0.115 | 0.604 | 0.990 | 1.000 | 1.000 |
| BAL-FLOW_25000 | 1-CKA | uc@Y | 0.000 | 0.000 | 0.219 | 0.844 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_25000 | 1-CKA | ubal@Y | 0.000 | 0.000 | 0.115 | 0.604 | 0.979 | 1.000 | 1.000 |
| BAL-FLOW_25000 | 1-CKA | u@Pool | 0.000 | 0.000 | 0.115 | 0.604 | 0.990 | 1.000 | 1.000 |
| BAL-FLOW_25000 | 1-CKA | uc@Pool | 0.000 | 0.000 | 0.219 | 0.844 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_250000 | G_s | y@Y | 0.000 | 0.000 | 0.375 | 1.000 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_250000 | G_s | u@Y | 0.000 | 0.000 | 0.000 | 0.365 | 0.979 | 1.000 | 1.000 |
| BAL-FLOW_250000 | G_s | uc@Y | 0.000 | 0.000 | 0.010 | 0.677 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_250000 | G_s | ubal@Y | 0.000 | 0.000 | 0.000 | 0.365 | 0.979 | 1.000 | 1.000 |
| BAL-FLOW_250000 | G_s | u@Pool | 0.000 | 0.010 | 0.552 | 1.000 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_250000 | G_s | uc@Pool | 0.000 | 0.062 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_250000 | 1-CKA | y@Y | 0.000 | 0.635 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_250000 | 1-CKA | u@Y | 0.000 | 0.031 | 0.229 | 0.698 | 0.948 | 1.000 | 1.000 |
| BAL-FLOW_250000 | 1-CKA | uc@Y | 0.000 | 0.135 | 0.583 | 0.927 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_250000 | 1-CKA | ubal@Y | 0.000 | 0.031 | 0.229 | 0.698 | 0.948 | 1.000 | 1.000 |
| BAL-FLOW_250000 | 1-CKA | u@Pool | 0.000 | 0.208 | 0.698 | 0.958 | 1.000 | 1.000 | 1.000 |
| BAL-FLOW_250000 | 1-CKA | uc@Pool | 0.000 | 0.583 | 0.927 | 1.000 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_25000 | G_s | y@Y | 0.000 | 0.000 | 0.125 | 0.979 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_25000 | G_s | u@Y | 0.000 | 0.000 | 0.000 | 0.083 | 0.875 | 1.000 | 1.000 |
| SMALL-FLOW_25000 | G_s | uc@Y | 0.000 | 0.000 | 0.000 | 0.250 | 0.990 | 1.000 | 1.000 |
| SMALL-FLOW_25000 | G_s | ubal@Y | 0.000 | 0.000 | 0.000 | 0.083 | 0.875 | 1.000 | 1.000 |
| SMALL-FLOW_25000 | G_s | u@Pool | 0.000 | 0.000 | 0.000 | 0.448 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_25000 | G_s | uc@Pool | 0.000 | 0.000 | 0.021 | 0.802 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_25000 | 1-CKA | y@Y | 0.000 | 0.073 | 0.604 | 1.000 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_25000 | 1-CKA | u@Y | 0.000 | 0.000 | 0.135 | 0.677 | 0.990 | 1.000 | 1.000 |
| SMALL-FLOW_25000 | 1-CKA | uc@Y | 0.000 | 0.042 | 0.344 | 0.948 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_25000 | 1-CKA | ubal@Y | 0.000 | 0.000 | 0.146 | 0.677 | 0.990 | 1.000 | 1.000 |
| SMALL-FLOW_25000 | 1-CKA | u@Pool | 0.000 | 0.042 | 0.354 | 0.917 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_25000 | 1-CKA | uc@Pool | 0.000 | 0.115 | 0.688 | 1.000 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_250000 | G_s | y@Y | 0.000 | 0.594 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_250000 | G_s | u@Y | 0.000 | 0.000 | 0.542 | 0.969 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_250000 | G_s | uc@Y | 0.000 | 0.104 | 0.938 | 0.990 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_250000 | G_s | ubal@Y | 0.000 | 0.000 | 0.542 | 0.969 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_250000 | G_s | u@Pool | 0.000 | 0.000 | 0.542 | 0.969 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_250000 | G_s | uc@Pool | 0.000 | 0.104 | 0.938 | 0.990 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_250000 | 1-CKA | y@Y | 0.146 | 0.979 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_250000 | 1-CKA | u@Y | 0.000 | 0.552 | 0.865 | 0.990 | 0.990 | 1.000 | 1.000 |
| SMALL-FLOW_250000 | 1-CKA | uc@Y | 0.062 | 0.927 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SMALL-FLOW_250000 | 1-CKA | ubal@Y | 0.000 | 0.552 | 0.865 | 0.990 | 0.990 | 1.000 | 1.000 |
| SMALL-FLOW_250000 | 1-CKA | u@Pool | 0.000 | 0.552 | 0.865 | 0.990 | 0.990 | 1.000 | 1.000 |
| SMALL-FLOW_250000 | 1-CKA | uc@Pool | 0.062 | 0.927 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| BAL-MOM_25000 | G_s | y@Y | 0.000 | 0.000 | 0.062 | 0.969 | 1.000 | 1.000 | 1.000 |
| BAL-MOM_25000 | G_s | u@Y | 0.000 | 0.000 | 0.000 | 0.000 | 0.438 | 1.000 | 1.000 |
| BAL-MOM_25000 | G_s | uc@Y | 0.000 | 0.000 | 0.000 | 0.000 | 0.719 | 1.000 | 1.000 |
| BAL-MOM_25000 | G_s | ubal@Y | 0.000 | 0.000 | 0.000 | 0.125 | 0.969 | 1.000 | 1.000 |
| BAL-MOM_25000 | G_s | u@Pool | 0.000 | 0.000 | 0.000 | 0.344 | 1.000 | 1.000 | 1.000 |
| BAL-MOM_25000 | G_s | uc@Pool | 0.000 | 0.000 | 0.000 | 0.625 | 1.000 | 1.000 | 1.000 |
| BAL-MOM_25000 | 1-CKA | y@Y | 0.000 | 0.000 | 0.375 | 0.938 | 1.000 | 1.000 | 1.000 |
| BAL-MOM_25000 | 1-CKA | u@Y | 0.000 | 0.000 | 0.000 | 0.281 | 0.750 | 0.938 | 1.000 |
| BAL-MOM_25000 | 1-CKA | uc@Y | 0.000 | 0.000 | 0.000 | 0.344 | 0.938 | 1.000 | 1.000 |
| BAL-MOM_25000 | 1-CKA | ubal@Y | 0.000 | 0.000 | 0.094 | 0.531 | 0.938 | 1.000 | 1.000 |
| BAL-MOM_25000 | 1-CKA | u@Pool | 0.000 | 0.000 | 0.281 | 0.781 | 0.969 | 1.000 | 1.000 |
| BAL-MOM_25000 | 1-CKA | uc@Pool | 0.000 | 0.000 | 0.500 | 0.938 | 1.000 | 1.000 | 1.000 |
| BAL-MOM_250000 | G_s | y@Y | 0.000 | 0.000 | 0.406 | 1.000 | 1.000 | 1.000 | 1.000 |
| BAL-MOM_250000 | G_s | u@Y | 0.000 | 0.000 | 0.000 | 0.000 | 0.375 | 0.969 | 1.000 |
| BAL-MOM_250000 | G_s | uc@Y | 0.000 | 0.000 | 0.000 | 0.062 | 0.875 | 1.000 | 1.000 |
| BAL-MOM_250000 | G_s | ubal@Y | 0.000 | 0.000 | 0.000 | 0.500 | 1.000 | 1.000 | 1.000 |
| BAL-MOM_250000 | G_s | u@Pool | 0.000 | 0.000 | 0.250 | 0.906 | 1.000 | 1.000 | 1.000 |
| BAL-MOM_250000 | G_s | uc@Pool | 0.000 | 0.000 | 0.688 | 1.000 | 1.000 | 1.000 | 1.000 |
| BAL-MOM_250000 | 1-CKA | y@Y | 0.000 | 0.531 | 0.938 | 1.000 | 1.000 | 1.000 | 1.000 |
| BAL-MOM_250000 | 1-CKA | u@Y | 0.000 | 0.000 | 0.031 | 0.188 | 0.500 | 0.844 | 1.000 |
| BAL-MOM_250000 | 1-CKA | uc@Y | 0.000 | 0.000 | 0.188 | 0.594 | 0.906 | 1.000 | 1.000 |
| BAL-MOM_250000 | 1-CKA | ubal@Y | 0.000 | 0.000 | 0.156 | 0.656 | 0.938 | 1.000 | 1.000 |
| BAL-MOM_250000 | 1-CKA | u@Pool | 0.000 | 0.281 | 0.750 | 0.969 | 1.000 | 1.000 | 1.000 |
| BAL-MOM_250000 | 1-CKA | uc@Pool | 0.000 | 0.750 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_25000 | G_s | y@Y | 0.000 | 0.000 | 0.000 | 0.969 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_25000 | G_s | u@Y | 0.000 | 0.000 | 0.000 | 0.031 | 0.781 | 1.000 | 1.000 |
| SMALL-MOM_25000 | G_s | uc@Y | 0.000 | 0.000 | 0.000 | 0.094 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_25000 | G_s | ubal@Y | 0.000 | 0.000 | 0.000 | 0.312 | 0.969 | 1.000 | 1.000 |
| SMALL-MOM_25000 | G_s | u@Pool | 0.000 | 0.000 | 0.000 | 0.406 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_25000 | G_s | uc@Pool | 0.000 | 0.000 | 0.000 | 0.750 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_25000 | 1-CKA | y@Y | 0.000 | 0.031 | 0.344 | 1.000 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_25000 | 1-CKA | u@Y | 0.000 | 0.000 | 0.156 | 0.688 | 0.906 | 1.000 | 1.000 |
| SMALL-MOM_25000 | 1-CKA | uc@Y | 0.000 | 0.031 | 0.281 | 0.906 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_25000 | 1-CKA | ubal@Y | 0.000 | 0.031 | 0.188 | 0.750 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_25000 | 1-CKA | u@Pool | 0.000 | 0.219 | 0.812 | 0.969 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_25000 | 1-CKA | uc@Pool | 0.000 | 0.375 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_250000 | G_s | y@Y | 0.000 | 0.000 | 0.562 | 0.969 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_250000 | G_s | u@Y | 0.000 | 0.000 | 0.000 | 0.156 | 0.750 | 1.000 | 1.000 |
| SMALL-MOM_250000 | G_s | uc@Y | 0.000 | 0.000 | 0.062 | 0.656 | 0.969 | 1.000 | 1.000 |
| SMALL-MOM_250000 | G_s | ubal@Y | 0.000 | 0.000 | 0.094 | 0.688 | 0.969 | 1.000 | 1.000 |
| SMALL-MOM_250000 | G_s | u@Pool | 0.000 | 0.031 | 0.406 | 0.875 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_250000 | G_s | uc@Pool | 0.000 | 0.188 | 0.906 | 0.969 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_250000 | 1-CKA | y@Y | 0.000 | 0.406 | 0.750 | 0.906 | 1.000 | 1.000 | 1.000 |
| SMALL-MOM_250000 | 1-CKA | u@Y | 0.000 | 0.031 | 0.094 | 0.219 | 0.594 | 0.812 | 1.000 |
| SMALL-MOM_250000 | 1-CKA | uc@Y | 0.000 | 0.062 | 0.438 | 0.750 | 0.938 | 0.969 | 1.000 |
| SMALL-MOM_250000 | 1-CKA | ubal@Y | 0.000 | 0.031 | 0.188 | 0.562 | 0.844 | 0.938 | 1.000 |
| SMALL-MOM_250000 | 1-CKA | u@Pool | 0.031 | 0.594 | 0.844 | 0.969 | 0.969 | 1.000 | 1.000 |
| SMALL-MOM_250000 | 1-CKA | uc@Pool | 0.250 | 0.938 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |
| SGD-K_25000 | G_s | y@Y | 0.000 | 0.000 | 0.031 | 0.938 | 1.000 | 1.000 | 1.000 |
| SGD-K_25000 | G_s | u@Y | 0.000 | 0.000 | 0.000 | 0.000 | 0.094 | 0.875 | 1.000 |
| SGD-K_25000 | G_s | uc@Y | 0.000 | 0.000 | 0.000 | 0.000 | 0.125 | 0.938 | 1.000 |
| SGD-K_25000 | G_s | ubal@Y | 0.000 | 0.000 | 0.000 | 0.406 | 0.969 | 1.000 | 1.000 |
| SGD-K_25000 | G_s | u@Pool | 0.000 | 0.000 | 0.000 | 0.000 | 0.094 | 0.875 | 1.000 |
| SGD-K_25000 | G_s | uc@Pool | 0.000 | 0.000 | 0.000 | 0.000 | 0.125 | 0.938 | 1.000 |
| SGD-K_25000 | 1-CKA | y@Y | 0.000 | 0.000 | 0.281 | 0.875 | 1.000 | 1.000 | 1.000 |
| SGD-K_25000 | 1-CKA | u@Y | 0.000 | 0.000 | 0.000 | 0.000 | 0.312 | 0.844 | 1.000 |
| SGD-K_25000 | 1-CKA | uc@Y | 0.000 | 0.000 | 0.000 | 0.031 | 0.562 | 0.906 | 1.000 |
| SGD-K_25000 | 1-CKA | ubal@Y | 0.000 | 0.000 | 0.125 | 0.719 | 0.969 | 1.000 | 1.000 |
| SGD-K_25000 | 1-CKA | u@Pool | 0.000 | 0.000 | 0.000 | 0.531 | 0.906 | 1.000 | 1.000 |
| SGD-K_25000 | 1-CKA | uc@Pool | 0.000 | 0.000 | 0.094 | 0.781 | 1.000 | 1.000 | 1.000 |
| SGD-K_250000 | G_s | y@Y | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SGD-K_250000 | G_s | u@Y | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SGD-K_250000 | G_s | uc@Y | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SGD-K_250000 | G_s | ubal@Y | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SGD-K_250000 | G_s | u@Pool | 0.000 | 0.000 | 0.000 | 0.562 | 0.969 | 1.000 | 1.000 |
| SGD-K_250000 | G_s | uc@Pool | 0.000 | 0.000 | 0.000 | 0.625 | 1.000 | 1.000 | 1.000 |
| SGD-K_250000 | 1-CKA | y@Y | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SGD-K_250000 | 1-CKA | u@Y | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SGD-K_250000 | 1-CKA | uc@Y | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SGD-K_250000 | 1-CKA | ubal@Y | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SGD-K_250000 | 1-CKA | u@Pool | 0.000 | 0.469 | 0.875 | 0.969 | 1.000 | 1.000 | 1.000 |
| SGD-K_250000 | 1-CKA | uc@Pool | 0.000 | 0.656 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |

## 11. Frozen anchors (pilot pairs at the same N), primary rungs

| Cell | Q | ε^Y | ε^Y_q90 | ε^O | ε^C | ε^h | ε^Pool | pilot pairs |
|---|---|---|---|---|---|---|---|---|
| MOM_25000 | T_N(10) | 0.91821 | 0.91648 | 0.91855 | 0.91948 | 0.85156 | 0.92153 | 96 |
| MOM_25000 | G_s | 0.28477 | 0.27968 | 0.31370 | 0.30468 | 0.39731 | 0.29467 | 96 |
| MOM_25000 | 1-CKA | 0.16397 | 0.15809 | 0.17953 | 0.17086 | 0.11370 | 0.16885 | 96 |
| MOM_250000 | T_N(10) | 0.83096 | 0.81736 | 0.82817 | 0.82959 | 0.72021 | 0.83271 | 96 |
| MOM_250000 | G_s | 0.19460 | 0.18869 | 0.21923 | 0.21093 | 0.28235 | 0.20985 | 96 |
| MOM_250000 | 1-CKA | 0.07790 | 0.07258 | 0.08955 | 0.08596 | 0.06458 | 0.08670 | 96 |
| WD_25000 | T_N(10) | 0.91079 | 0.90676 | 0.91069 | 0.91079 | 0.85073 | 0.92153 | 96 |
| WD_25000 | G_s | 0.29079 | 0.28647 | 0.28546 | 0.28533 | 0.28357 | 0.29467 | 96 |
| WD_25000 | 1-CKA | 0.14314 | 0.13894 | 0.13606 | 0.13485 | 0.09955 | 0.16885 | 96 |
| WD_250000 | T_N(10) | 0.81162 | 0.80007 | 0.81392 | 0.81431 | 0.71670 | 0.83271 | 96 |
| WD_250000 | G_s | 0.20175 | 0.18995 | 0.20091 | 0.19984 | 0.19872 | 0.20985 | 96 |
| WD_250000 | 1-CKA | 0.07557 | 0.06278 | 0.07744 | 0.07678 | 0.06438 | 0.08670 | 96 |
| FLOW_25000 | T_N(10) | 0.91992 | 0.91570 | 0.91875 | 0.91919 | 0.84736 | 0.92153 | 64 |
| FLOW_25000 | G_s | 0.28375 | 0.28105 | 0.30816 | 0.30289 | 0.41665 | 0.29467 | 64 |
| FLOW_25000 | 1-CKA | 0.16847 | 0.16434 | 0.18340 | 0.17584 | 0.09821 | 0.16885 | 64 |
| FLOW_250000 | T_N(10) | 0.83223 | 0.82580 | 0.83008 | 0.83037 | 0.72534 | 0.83271 | 64 |
| FLOW_250000 | G_s | 0.19959 | 0.19285 | 0.21381 | 0.21215 | 0.31764 | 0.20985 | 64 |
| FLOW_250000 | 1-CKA | 0.08421 | 0.07804 | 0.09398 | 0.08815 | 0.06202 | 0.08670 | 64 |
| BAL-FLOW_25000 | T_N(10) | 0.92153 | 0.91714 | 0.91934 | 0.91943 | 0.85161 | 0.92153 | 96 |
| BAL-FLOW_25000 | G_s | 0.28545 | 0.28078 | 0.30420 | 0.29870 | 0.41550 | 0.29467 | 96 |
| BAL-FLOW_25000 | 1-CKA | 0.16885 | 0.16367 | 0.17969 | 0.17329 | 0.10372 | 0.16885 | 96 |
| BAL-FLOW_250000 | T_N(10) | 0.82881 | 0.82209 | 0.82622 | 0.82588 | 0.72197 | 0.83271 | 96 |
| BAL-FLOW_250000 | G_s | 0.19656 | 0.19261 | 0.20801 | 0.20366 | 0.33215 | 0.20985 | 96 |
| BAL-FLOW_250000 | 1-CKA | 0.08242 | 0.07778 | 0.08907 | 0.08497 | 0.06775 | 0.08670 | 96 |
| SMALL-FLOW_25000 | T_N(10) | 0.91753 | 0.91479 | 0.91797 | 0.91763 | 0.85171 | 0.92153 | 96 |
| SMALL-FLOW_25000 | G_s | 0.28731 | 0.28380 | 0.31165 | 0.30236 | 0.56210 | 0.29467 | 96 |
| SMALL-FLOW_25000 | 1-CKA | 0.16472 | 0.16154 | 0.17585 | 0.16864 | 0.13684 | 0.16885 | 96 |
| SMALL-FLOW_250000 | T_N(10) | 0.83271 | 0.81953 | 0.82749 | 0.82734 | 0.72612 | 0.83271 | 96 |
| SMALL-FLOW_250000 | G_s | 0.20985 | 0.19463 | 0.22009 | 0.21277 | 0.35939 | 0.20985 | 96 |
| SMALL-FLOW_250000 | 1-CKA | 0.08670 | 0.07607 | 0.09489 | 0.09141 | 0.06960 | 0.08670 | 96 |
| BAL-MOM_25000 | T_N(10) | 0.91855 | 0.91685 | 0.91743 | 0.91870 | 0.85059 | 0.92153 | 16 |
| BAL-MOM_25000 | G_s | 0.28198 | 0.28005 | 0.30817 | 0.30053 | 0.40918 | 0.29467 | 16 |
| BAL-MOM_25000 | 1-CKA | 0.15947 | 0.15754 | 0.17852 | 0.17180 | 0.10427 | 0.16885 | 16 |
| BAL-MOM_250000 | T_N(10) | 0.82261 | 0.81875 | 0.82285 | 0.82354 | 0.71406 | 0.83271 | 16 |
| BAL-MOM_250000 | G_s | 0.19220 | 0.18819 | 0.21136 | 0.20796 | 0.26774 | 0.20985 | 16 |
| BAL-MOM_250000 | 1-CKA | 0.07678 | 0.07188 | 0.08863 | 0.08608 | 0.05976 | 0.08670 | 16 |
| SMALL-MOM_25000 | T_N(10) | 0.91514 | 0.91409 | 0.91377 | 0.91572 | 0.85444 | 0.92153 | 16 |
| SMALL-MOM_25000 | G_s | 0.28744 | 0.28366 | 0.30492 | 0.30165 | 0.39915 | 0.29467 | 16 |
| SMALL-MOM_25000 | 1-CKA | 0.15856 | 0.15544 | 0.16371 | 0.15976 | 0.10955 | 0.16885 | 16 |
| SMALL-MOM_250000 | T_N(10) | 0.81689 | 0.81492 | 0.81807 | 0.81782 | 0.71626 | 0.83271 | 16 |
| SMALL-MOM_250000 | G_s | 0.19566 | 0.19261 | 0.20741 | 0.20450 | 0.29891 | 0.20985 | 16 |
| SMALL-MOM_250000 | 1-CKA | 0.07431 | 0.07221 | 0.08521 | 0.08094 | 0.06623 | 0.08670 | 16 |
| SGD-K_25000 | T_N(10) | 0.91699 | 0.91272 | 0.91782 | 0.91772 | 0.85869 | 0.92153 | 16 |
| SGD-K_25000 | G_s | 0.29467 | 0.29383 | 0.32911 | 0.32316 | 0.43086 | 0.29467 | 16 |
| SGD-K_25000 | 1-CKA | 0.15775 | 0.15487 | 0.17948 | 0.17150 | 0.14267 | 0.16885 | 16 |
| SGD-K_250000 | T_N(10) | 0.99434 | 0.81965 | 0.81826 | 0.99360 | 0.70542 | 0.83271 | 16 |
| SGD-K_250000 | G_s | 1.17234 | 0.20163 | 0.22374 | 1.18756 | 0.24618 | 0.20985 | 16 |
| SGD-K_250000 | 1-CKA | 0.95565 | 0.07702 | 0.09062 | 0.96986 | 0.07135 | 0.08670 | 16 |

Pilot homogeneity diagnostic (mean Δ_Q(y) per arm in pooled-SD units; reported, not a gate): 25k MOM G_s -0.41 / CKA 0.14, WD G_s 0.69 / CKA -1.80, FLOW G_s -0.41 / CKA 0.79, BAL-FLOW G_s -0.48 / CKA 0.75, SMALL-FLOW G_s 0.18 / CKA 0.40, BAL-MOM G_s -0.46 / CKA 0.10, SMALL-MOM G_s 0.13 / CKA -0.14, SGD-K G_s 2.10 / CKA -0.05; 250k MOM G_s -0.51 / CKA 0.00203, WD G_s -0.48 / CKA -1.62, FLOW G_s 0.38 / CKA 0.79, BAL-FLOW G_s 0.31 / CKA 0.69, SMALL-FLOW G_s 0.53 / CKA 0.43, BAL-MOM G_s -0.44 / CKA 0.12, SMALL-MOM G_s -0.23 / CKA -0.25, SGD-K G_s 14.96 / CKA 9.22.

## 12. Covariates per cell (medians over ok main runs; descriptive aggregation of per-run fields)

| Cell | ok/n | acc_bank | ce_train | ce_hold | gen_gap | cond_W0 | cond_W | cond_M | s | rho_end | delta_C | mu_C | disp_P_median | disp_W_median | head_substeps_total | t_total_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOM_25000 | 192/192 | 0.8841 | 0.2282 | 0.5253 | 0.2963 | 25.0031 | 1.5384 | 1.8113 | 3.9548 | 0.2733 | 0.2756 | -0.0166 | 0.0971 | 0.0631 | 0.0000 | 5.4804 |
| WD_25000 | 192/192 | 0.8842 | 0.2370 | 0.5346 | 0.2978 | 28.4538 | 1.1603 | 1.3360 | 1.8373 | 0.0301 | 0.0778 | 0.1683 | 0.1013 | 0.0708 | 0.0000 | 5.6205 |
| FLOW_25000 | 192/192 | 0.8841 | 0.2298 | 0.5250 | 0.2949 | 27.2398 | 1.5410 | 2.3108 | 4.8781 | 0.0221 | 2.36e-05 | 1.0000 | 0.2911 | 0.1389 | 9260.0000 | 25.4244 |
| BAL-FLOW_25000 | 192/192 | 0.8843 | 0.2297 | 0.5247 | 0.2947 | 1.3483 | 1.4955 | 2.2365 | 5.0294 | 2.36e-05 | 2.36e-05 | -0.0171 | 0.2918 | 0.1382 | 9210.0000 | 25.1799 |
| SMALL-FLOW_25000 | 192/192 | 0.8837 | 0.2341 | 0.5262 | 0.2924 | 28.0936 | 1.5989 | 2.5557 | 4.0622 | 0.0002 | 2.32e-05 | 0.9953 | 0.2623 | 0.1386 | 9708.0000 | 25.9979 |
| BAL-MOM_25000 | 64/64 | 0.8844 | 0.2277 | 0.5248 | 0.2969 | 1.3346 | 1.5197 | 1.7767 | 4.0192 | 0.2844 | 0.2844 | -0.0138 | 0.0968 | 0.0630 | 0.0000 | 5.4538 |
| SMALL-MOM_25000 | 64/64 | 0.8839 | 0.2353 | 0.5274 | 0.2930 | 27.0078 | 1.6143 | 2.0446 | 3.3076 | 0.2393 | 0.2393 | -0.0827 | 0.0875 | 0.0624 | 0.0000 | 5.5100 |
| SGD-K_25000 | 64/64 | 0.8837 | 0.2323 | 0.5270 | 0.2957 | 27.7326 | 1.7207 | 1.8237 | 4.1969 | 0.3456 | 0.3435 | 0.0767 | 0.3832 | 0.3321 | 488.0000 | 6.2629 |
| MOM_250000 | 192/192 | 0.9018 | 0.4454 | 0.4702 | 0.0248 | 24.7905 | 2.0948 | 2.8032 | 2.4401 | 0.3119 | 0.3143 | -0.0051 | 0.0956 | 0.0672 | 0.0000 | 5.9094 |
| WD_250000 | 192/192 | 0.9021 | 0.4524 | 0.4785 | 0.0252 | 25.3392 | 1.4089 | 1.7804 | 1.3154 | 0.0396 | 0.0912 | 0.1649 | 0.1044 | 0.0760 | 0.0000 | 6.2112 |
| FLOW_250000 | 192/192 | 0.9014 | 0.4446 | 0.4707 | 0.0266 | 23.2025 | 1.8573 | 3.3258 | 3.3791 | 0.0255 | 2.54e-05 | 1.0000 | 0.2853 | 0.1429 | 9353.5000 | 26.0626 |
| BAL-FLOW_250000 | 192/192 | 0.9012 | 0.4445 | 0.4705 | 0.0264 | 1.3450 | 1.8043 | 3.2554 | 3.4732 | 2.54e-05 | 2.54e-05 | -0.0288 | 0.2860 | 0.1424 | 9300.0000 | 25.9211 |
| SMALL-FLOW_250000 | 192/192 | 0.9005 | 0.4471 | 0.4735 | 0.0267 | 25.6258 | 1.8173 | 3.3010 | 3.1160 | 0.0003 | 2.49e-05 | 0.9960 | 0.2587 | 0.1416 | 9802.0000 | 26.5312 |
| BAL-MOM_250000 | 64/64 | 0.9022 | 0.4434 | 0.4711 | 0.0282 | 1.3298 | 2.0772 | 2.7555 | 2.4794 | 0.3264 | 0.3264 | -0.0390 | 0.0960 | 0.0676 | 0.0000 | 5.7496 |
| SMALL-MOM_250000 | 64/64 | 0.9005 | 0.4492 | 0.4754 | 0.0258 | 24.3107 | 1.9843 | 2.7359 | 2.3661 | 0.2780 | 0.2780 | -0.1290 | 0.0880 | 0.0655 | 0.0000 | 5.8209 |
| SGD-K_250000 | 64/64 | 0.9012 | 0.4460 | 0.4709 | 0.0257 | 26.0601 | 2.3321 | 2.5680 | 2.7979 | 0.3754 | 0.3755 | 0.0666 | 0.3855 | 0.3474 | 488.0000 | 6.5948 |
| MOM-BF16_250000 | 32/32 | 0.9013 | 0.4438 | 0.4715 | 0.0301 | 27.9329 | 2.1041 | 2.8082 | 2.4398 | 0.3121 | 0.3150 | 0.0461 | 0.0955 | 0.0676 | 0.0000 | 6.2902 |
| KS-4_250000 | 16/16 | 0.9007 | 0.4455 | 0.4708 | 0.0279 | 16.5578 | 1.9007 | 3.2177 | 3.2006 | 0.1355 | 0.1370 | 0.0353 | 0.2892 | 0.1695 | 1952.0000 | 11.8985 |
| KS-16_250000 | 16/16 | 0.9015 | 0.4469 | 0.4698 | 0.0229 | 22.8538 | 1.8704 | 3.3175 | 3.3235 | 0.0352 | 0.0281 | 0.6166 | 0.2863 | 0.1506 | 7808.0000 | 14.0986 |

## 13. Pre-agreed consequences as evaluated by the frozen code (verbatim)

- C1, C3, C4a hold: Remark rem:optimizers gains the prospective network test (minibatch sampling alone does not change C; discretization and momentum do; from C0 = 0 momentum raised rho to {25000: {'median': 0.28435432632652485, 'lower_975': 0.2833269160522509, 'frac_ge_0p02': 1.0, 'n': 64, 'pass': True}, 250000: {'median': 0.32644341610795935, 'lower_975': 0.3248465259082973, 'frac_ge_0p02': 1.0, 'n': 64, 'pass': True}}; the final imbalance no longer reflects its initial value).
- C4b: statistic insensitive: FLOW's CP interval contains 0.5 -> the v3 gauge-memory sentence loses its evidence (replaced by the C4a wording)
- C5: both causes (discretization and momentum) stay in the Remark.
- I3: 8/8 Holm rejections; for the rejected cells Section 6.3/7 may say 'from a balanced or small start, momentum creates the imbalance channel; under head flow it stays absent by construction'.
- I4 (i): 'with C held at its random initial value, the head's imbalance still lowers geometric agreement between runs ... the network realizes the mechanism of part 2, at a reduced size'.
- A1 holds: WD's u agrees at both sizes under same-N tolerances (removes the '25k undecided' caveat).
- A3 anchor-sensitive: the knife-edge text is replaced by the same-N result as reported (no categorical claim if anchor-sensitive).
- A5: the v3 run-dependent sentence is replaced by the v1.1 same-N decision for MOM 250k G_s: Delta(y)+M at eps^Y = inconclusive (78/96); M = 1.6296505363353346 sigma_y; its categorical status depends on the tolerance.
- A7 (b) BAL-FLOW: 'with C ~ 0 an excess remains, carried by the balanced map (WP is not a projector), outside Theorem 4'
- A7 (b) SMALL-FLOW: 'with C ~ 0 an excess remains, carried by the balanced map (WP is not a projector), outside Theorem 4'
- K4(b, c) hold: may be called prospective; K1-K3 remain code checks.
- Lemma N1: entering v3 is the author's decision.

