# S2.2 gauge net, FINAL analysis: independent recompute (review role `s22-recompute`, 2026-10-08)

Scope: `results/S2_2/S2_2_report.md` and `results/S2_2/s22_summary.json` (FINAL, written 14:56Z by `auto_finalize` → `src/analyze.py s22`), checked against the runs, the cached per-pair ladders and the frozen `prereg/eps_S2_2.json` (sha256 b46e3f1d…, the hash in STATUS line 104). I did not reuse `analyze.py`/`crit.py` decision code. What I did instead:

- read the run plan, every `meta.json` and every `feats.npz` (1344 main + 48 pilot runs);
- recomputed s_N(10), Gs, 1−CKA (biased, linear) and Δ_Π from the saved features with my own numpy code, for all 672 main pairs × {y, u, h, h_l2} and all 24 pilot pairs;
- mapped every cached hrhv2 ladder row (2688 files) to Δ myself for the other five rungs (Δ_O, Δ_E, 1−s_T, 1−RSA, 1−SOG-τ0.1);
- applied the one-sided 95% Clopper–Pearson rule myself (`scipy.stats.beta`) and compared every cell with both the md and the json.

Compute: 3 CPU cores, about 17 min, no GPU. Everything new is in `results/review_final/s22-recompute/`.

## 0. Bottom line

1. **The numbers in the report are correct.** All 252 decision cells (7 pair sets × 4 representations × 9 rungs) match: report md, s22_summary.json and my recomputation give the same x and the same decision, with **0 mismatches**. For the four rungs I recomputed from features, my values agree with the hrhv2 ladder rows to ≤ 1.7e-14. P1–P5, the falsifier, the P4 means and the shared-corpus differences also agree. My bootstrap, run with a different seed, reproduces the report's CIs to within 2e-4.
2. **The data are complete.** 1344/1344 main runs are done, with 0 failed, 0 diverged and 0 non-finite runs, and all ran 488 steps. Pairs are exactly (2j, 2j+1) within each cell (and within each corpus for the shared cell), each run is used once, and each cell has 96 pairs. All 4 × 672 ladder files exist, with no extras, no `.tmp` files, no `failed_run` entries and m = 2048. The subsets are as designed: 1156 distinct subset hashes, which is 1152 independent runs plus 4 shared corpora of 48 runs each.
3. **ε was applied as preregistered.** It comes from the 24 pilot pairs only (cohort S2_2_pilot, N = 100,000, 8 pairs (2j, 2j+1) per arm), using the max rule. I re-derived it from the pilot features and it matches the file to ≤ 3.7e-15. ε(u) = ε(y) holds for every rung (function-anchored, `anchor {'u': 'y'}`). The file was frozen at 10:39Z, before the first main ladder at 10:48Z.
4. **Preregistered verdicts (reproduced):**
   - **P1** fails: noWD u T_N(10) at 250k is reliable (96/96).
   - **The falsifier is TRIGGERED** (noWD u T_N(10) reliable at 250k).
   - **P2** passes.
   - **P3** passes for WD and fails for smallinit: Gs is inconclusive at 82/96 and CKA is inconclusive at 86/96.
   - **P4** passes (max arm difference in mean s_N(10)(h): 0.006 at 25k, 0.011 at 250k).
   - **P5** fails (|p̂_smallinit − p̂_WD| for Gs at 250k = 14/96 = 0.146).
5. **Two of the headline cells sit on a knife edge.**
   - noWD 250k u Gs is 65/96 "unreliable". Mean Δ is 0.2050 against ε = 0.2073, and the pair closest to ε is 1.0e-4 away. p̂ is 0.00 at 0.9ε, 0.68 at ε and 1.00 at 1.1ε. A 5% larger ε (0.2171) would make this cell "reliable".
   - noWD 250k u CKA is 73/96 "inconclusive". p̂ is 0.04 at 0.9ε, 0.76 at ε and 1.00 at 1.1ε.
   - smallinit 250k u Gs is 82/96 (p̂ 0.85 at ε). Under the preregistered *secondary* ε (pilot 0.9-quantile, 0.2030) it would be 58/96 "unreliable", and noWD would be 39/96.
6. **The 25k cells say nothing about the gauge under the frozen ε, and the 100k anchor is the reason.** At 25k every representation fails both primary rungs (0/96) in all three independent cells, including the function y. The one exception is WD h Gs (see §7.7). The *smallest* y-pair discrepancy at 25k, over all three arms, is already above the 100k maximum:
   - Gs: 0.2662 vs ε = 0.2073; reliability would need 1.36–1.39× ε.
   - T_N(10): Δ 0.8967 vs ε = 0.8399, i.e. s_N(10) 0.103 vs 0.160; reliability would need 1.08–1.09× ε.

   These decisions only record that seed-to-seed agreement at 25k is worse than at 100k, which is a sample-size effect. They also make P5's two 25k terms pass trivially (0/96 vs 0/96).
7. **Exploratory: the gauge is visible at both N when read without the absolute ε, and it shows only in the geometric rungs.** I compared each pair's u-discrepancy with its own y-discrepancy (median Δu − Δy):
   - Gs: noWD +0.023 (25k), about +0.020 at the pilot's 100k (difference of medians) and +0.020 (250k), with u > y in 100% of pairs. WD −0.004 / −0.0001, with u > y in 0% and 46% of pairs. smallinit +0.017 / +0.016, with u > y in 100% of pairs.
   - The excess is roughly constant in N, which is what Theorem 4 part 2 says: the initialisation-set gauge spread does not shrink with data. WD removes it. Small initialisation removes only about 20–30% of it, which matches its unbalanced ρ_end ≈ 0.24–0.28.
   - T_N(10) is blind to this gauge: median Δu − Δy is between −0.0011 and +0.0001 in every cell, and cond(W) ≤ 2.25 in every run.

   So the falsifier fires on a rung the gauge does not move. Theorem 4 and Table 2 claim an obstruction for *geometry*, not for ordinal kNN rungs, so the T_N(10) part of P1 went beyond the theorem. That is an interpretation. The preregistered outcome still stands as written.
8. **Errors and gaps found** (details in §7). None changes a number or a decision.
   - The report omits the p̂(ε) curves, CP bounds and ε that prereg §4 requires in every table.
   - P2 reads Δ_Π on u instead of y. The two are identical to 1e-7, so this is harmless.
   - P5 counts vacuous 25k terms as passes.
   - The analysis and gate code were not hashed at prereg time.
   - An unlogged, aborted INTERIM run (R = 62) wrote ladders that the final analysis reused. I verified those ladders.
   - The Π-check is printed as "0.00".
   - ε(h, Gs) is set by one smallinit pilot pair, which makes WD 25k h Gs "reliable" while y and u fail.

## 1. Run and pair inventory

| cell | runs | done | failed | diverged | non-finite | steps | reps 0..R−1 complete | pairs | bad pairing |
|---|---|---|---|---|---|---|---|---|---|
| noWD / WD / smallinit × 25k, 250k (6 cells) | 192 each | 192 | 0 | 0 | 0 | 488 | yes | 96 | 0 |
| noWD-shared 25k, corpora c0..c3 | 48 each | 48 | 0 | 0 | 0 | 488 | yes | 24 × 4 = 96 | 0 |

- Ladder cache `results/ladders/S2_2_main/{y,u,h,h_l2}`: 672 files each. None are missing, none are unexpected, there are no `.tmp` files, and A/B/tag/pre/m are consistent in all 2688. Mapping the stored row to Δ reproduces the stored `deltas` exactly (max |diff| = 0).
- Π_u = Π_y: max projector distance 4.9e-8 over 1344 runs (prereg threshold 1e-6). Max cond(W) is 2.25.
- CP thresholds at n = 96, recomputed: reliable iff x ≥ 92, unreliable iff x ≤ 69. These agree with prereg §2.5.

## 2. Decision table: every cell, report vs recomputed

### T1. Decision table, report vs recomputed (all 252 cells)

Each cell: recomputed `decision x/96`; `=` means the report (md), s22_summary.json (x and decision) and, for the four feature-recomputed rungs (dN_k10, Gs, dCKA, dPi), the independent numpy recomputation all give the same x. Any disagreement is written out as `MISMATCH`.

| set | rep | dN_k10 | Gs | dCKA | dO | dE | dMST | dAff | dSOG_t0p1 | dPi |
|---|---|---|---|---|---|---|---|---|---|---|
| noWD_25000 | y | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| noWD_25000 | u | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| noWD_25000 | h | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 31/96 = | unr 1/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| noWD_25000 | h_l2 | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 2/96 = | unr 4/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| WD_25000 | y | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| WD_25000 | u | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| WD_25000 | h | unr 0/96 = | rel 96/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| WD_25000 | h_l2 | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| smallinit_25000 | y | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| smallinit_25000 | u | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| smallinit_25000 | h | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 43/96 = | unr 37/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| smallinit_25000 | h_l2 | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 1/96 = | unr 1/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| noWD-shared_25000 | y | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| noWD-shared_25000 | u | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 15/96 = | unr 31/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = | unr 0/96 = |
| noWD-shared_25000 | h | unr 0/96 = | unr 42/96 = | unr 53/96 = | rel 96/96 = | rel 96/96 = | unr 0/96 = | inc 70/96 = | unr 0/96 = | unr 0/96 = |
| noWD-shared_25000 | h_l2 | unr 0/96 = | rel 96/96 = | inc 86/96 = | inc 86/96 = | inc 89/96 = | unr 0/96 = | inc 85/96 = | unr 0/96 = | unr 0/96 = |
| noWD_250000 | y | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = |
| noWD_250000 | u | rel 96/96 = | unr 65/96 = | inc 73/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 95/96 = | rel 96/96 = | rel 96/96 = |
| noWD_250000 | h | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = |
| noWD_250000 | h_l2 | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = |
| WD_250000 | y | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = |
| WD_250000 | u | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = |
| WD_250000 | h | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = |
| WD_250000 | h_l2 | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 95/96 = | rel 94/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = |
| smallinit_250000 | y | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = |
| smallinit_250000 | u | rel 96/96 = | inc 82/96 = | inc 86/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = |
| smallinit_250000 | h | rel 96/96 = | rel 93/96 = | inc 91/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 95/96 = | rel 96/96 = | rel 96/96 = |
| smallinit_250000 | h_l2 | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = | rel 96/96 = |


## 3. Tolerances

### T2. Frozen tolerances (prereg/eps_S2_2.json) and their re-derivation from the 24 pilot pairs (N = 100k)

| rep | dN_k10 | Gs | dCKA | dO | dE | dMST | dAff | dSOG_t0p1 | dPi |
|---|---|---|---|---|---|---|---|---|---|
| y | 0.8399 | 0.2073 | 0.0833 | 0.2463 | 0.2433 | 0.9541 | 0.1463 | 0.9659 | 0.0923 |
| u | 0.8399 | 0.2073 | 0.0833 | 0.2463 | 0.2433 | 0.9541 | 0.1463 | 0.9659 | 0.0923 |
| h | 0.7404 | 0.3032 | 0.0643 | 0.1787 | 0.1775 | 0.8793 | 0.1577 | 0.9054 | 0.7040 |
| h_l2 | 0.7145 | 0.1137 | 0.0552 | 0.1736 | 0.1599 | 0.8578 | 0.0610 | 0.9014 | 0.7042 |

- eps = max of the file's pilot rows for every (rep, rung): yes; max |eps − max of my feature recomputation over the 24 pilot pairs| (dN_k10, Gs, dCKA, dPi) = 3.66e-15
- pilot cohort S2_2_pilot, N = [100000], 48 runs, 24 pairs (pairs in file = (2j, 2j+1) per arm: True); anchor {'u': 'y'}; eps(u) == eps(y) for every rung: True

Pilot (N = 100k) u-pairs measured against eps(y) (how far the gauge already moved u at the pilot N):

| arm | rung | u pairs ≤ eps(y) | median Δ(u) | median Δ(y) | max Δ(u) | max Δ(y) |
|---|---|---|---|---|---|---|
| noWD | dN_k10 | 7/8 | 0.8322 | 0.8330 | 0.8417 | 0.8399 |
| noWD | Gs | 0/8 | 0.2167 | 0.1970 | 0.2232 | 0.2053 |
| noWD | dCKA | 0/8 | 0.0903 | 0.0793 | 0.0969 | 0.0833 |
| noWD | dPi | 7/8 | 0.0886 | 0.0886 | 0.0923 | 0.0923 |
| WD | dN_k10 | 8/8 | 0.8169 | 0.8175 | 0.8219 | 0.8235 |
| WD | Gs | 8/8 | 0.1973 | 0.1955 | 0.2055 | 0.2038 |
| WD | dCKA | 8/8 | 0.0703 | 0.0679 | 0.0747 | 0.0724 |
| WD | dPi | 8/8 | 0.0766 | 0.0766 | 0.0797 | 0.0797 |
| smallinit | dN_k10 | 8/8 | 0.8283 | 0.8288 | 0.8359 | 0.8342 |
| smallinit | Gs | 1/8 | 0.2161 | 0.1977 | 0.2284 | 0.2073 |
| smallinit | dCKA | 0/8 | 0.0874 | 0.0776 | 0.0932 | 0.0828 |
| smallinit | dPi | 8/8 | 0.0860 | 0.0860 | 0.0916 | 0.0916 |


Timing: eps_S2_2.json was frozen at 10:39:06Z (STATUS l.104, sha256 b46e3f1d…, the same as the file now). The first main S2.2 ladder file has mtime 10:48:36Z. G1 passed at 10:34Z, so some main runs were training while the eps file was written, but no main ladder existed yet, as prereg §3.3 requires.

## 4. p̂(ε) curves (feature recomputation)

### T3. Δ distributions and p̂(ε) (feature recomputation; exploratory except the 1.0×ε column)

p̂ at multiples of the frozen ε; `ε for rel` = the 92nd smallest Δ (the smallest ε that gives x ≥ 92), shown also as a ratio to the frozen ε; `ε for not-unr` = 70th smallest Δ (x ≥ 70 needs ε ≥ this).

**u / Gs** (frozen ε = 0.2073)

| cell | min | q10 | median | q90 | max | p̂(0.5ε) | p̂(0.75ε) | p̂(0.9ε) | p̂(1.0ε) | p̂(1.1ε) | p̂(1.25ε) | p̂(1.5ε) | p̂(2.0ε) | ε for rel (×ε) | ε for not-unr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot100k_noWD (8 pairs) | 0.2095 | 0.2120 | 0.2167 | 0.2220 | 0.2232 |  |  |  |  |  |  |  |  | | |
| pilot100k_WD (8 pairs) | 0.1924 | 0.1936 | 0.1973 | 0.2017 | 0.2055 |  |  |  |  |  |  |  |  | | |
| pilot100k_smallinit (8 pairs) | 0.2060 | 0.2084 | 0.2161 | 0.2242 | 0.2284 |  |  |  |  |  |  |  |  | | |
| noWD_25000 | 0.2884 | 0.2929 | 0.2984 | 0.3045 | 0.3132 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.98 | 1.00 | 0.3066 (1.48×) | 0.3011 |
| WD_25000 | 0.2685 | 0.2717 | 0.2761 | 0.2811 | 0.2875 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.2832 (1.37×) | 0.2794 |
| smallinit_25000 | 0.2814 | 0.2900 | 0.2960 | 0.3041 | 0.3097 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.3070 (1.48×) | 0.2987 |
| noWD-shared_25000 | 0.2494 | 0.2528 | 0.2594 | 0.2673 | 0.2710 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 1.00 | 1.00 | 0.2689 (1.30×) | 0.2628 |
| noWD_250000 | 0.1919 | 0.1982 | 0.2039 | 0.2134 | 0.2208 | 0.00 | 0.00 | 0.00 | 0.68 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2171 (1.05×) | 0.2082 |
| WD_250000 | 0.1754 | 0.1792 | 0.1842 | 0.1901 | 0.1969 | 0.00 | 0.00 | 0.72 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1921 (0.93×) | 0.1869 |
| smallinit_250000 | 0.1904 | 0.1945 | 0.2012 | 0.2092 | 0.2165 | 0.00 | 0.00 | 0.00 | 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2145 (1.03×) | 0.2048 |

**u / dN_k10** (frozen ε = 0.8399)

| cell | min | q10 | median | q90 | max | p̂(0.5ε) | p̂(0.75ε) | p̂(0.9ε) | p̂(1.0ε) | p̂(1.1ε) | p̂(1.25ε) | p̂(1.5ε) | p̂(2.0ε) | ε for rel (×ε) | ε for not-unr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot100k_noWD (8 pairs) | 0.8256 | 0.8293 | 0.8322 | 0.8371 | 0.8417 |  |  |  |  |  |  |  |  | | |
| pilot100k_WD (8 pairs) | 0.8121 | 0.8125 | 0.8169 | 0.8213 | 0.8219 |  |  |  |  |  |  |  |  | | |
| pilot100k_smallinit (8 pairs) | 0.8207 | 0.8229 | 0.8283 | 0.8344 | 0.8359 |  |  |  |  |  |  |  |  | | |
| noWD_25000 | 0.9063 | 0.9101 | 0.9126 | 0.9167 | 0.9195 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9176 (1.09×) | 0.9142 |
| WD_25000 | 0.8959 | 0.8986 | 0.9026 | 0.9063 | 0.9115 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9085 (1.08×) | 0.9040 |
| smallinit_25000 | 0.9030 | 0.9079 | 0.9112 | 0.9143 | 0.9184 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9152 (1.09×) | 0.9122 |
| noWD-shared_25000 | 0.8772 | 0.8810 | 0.8846 | 0.8899 | 0.8920 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.8904 (1.06×) | 0.8871 |
| noWD_250000 | 0.7961 | 0.8035 | 0.8106 | 0.8169 | 0.8256 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.8199 (0.98×) | 0.8140 |
| WD_250000 | 0.7789 | 0.7846 | 0.7929 | 0.8001 | 0.8082 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.8030 (0.96×) | 0.7962 |
| smallinit_250000 | 0.7962 | 0.8021 | 0.8079 | 0.8147 | 0.8207 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.8183 (0.97×) | 0.8109 |

**y / Gs** (frozen ε = 0.2073)

| cell | min | q10 | median | q90 | max | p̂(0.5ε) | p̂(0.75ε) | p̂(0.9ε) | p̂(1.0ε) | p̂(1.1ε) | p̂(1.25ε) | p̂(1.5ε) | p̂(2.0ε) | ε for rel (×ε) | ε for not-unr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot100k_noWD (8 pairs) | 0.1940 | 0.1945 | 0.1970 | 0.2007 | 0.2053 |  |  |  |  |  |  |  |  | | |
| pilot100k_WD (8 pairs) | 0.1939 | 0.1945 | 0.1955 | 0.2007 | 0.2038 |  |  |  |  |  |  |  |  | | |
| pilot100k_smallinit (8 pairs) | 0.1934 | 0.1947 | 0.1977 | 0.2031 | 0.2073 |  |  |  |  |  |  |  |  | | |
| noWD_25000 | 0.2662 | 0.2706 | 0.2752 | 0.2806 | 0.2862 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.2823 (1.36×) | 0.2772 |
| WD_25000 | 0.2731 | 0.2761 | 0.2802 | 0.2857 | 0.2915 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.2881 (1.39×) | 0.2829 |
| smallinit_25000 | 0.2705 | 0.2749 | 0.2793 | 0.2842 | 0.2914 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.2851 (1.38×) | 0.2819 |
| noWD-shared_25000 | 0.2328 | 0.2370 | 0.2422 | 0.2460 | 0.2509 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.2486 (1.20×) | 0.2444 |
| noWD_250000 | 0.1780 | 0.1813 | 0.1847 | 0.1897 | 0.1930 | 0.00 | 0.00 | 0.69 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1920 (0.93×) | 0.1866 |
| WD_250000 | 0.1763 | 0.1792 | 0.1840 | 0.1901 | 0.1981 | 0.00 | 0.00 | 0.72 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1924 (0.93×) | 0.1868 |
| smallinit_250000 | 0.1797 | 0.1820 | 0.1863 | 0.1908 | 0.1971 | 0.00 | 0.00 | 0.52 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1925 (0.93×) | 0.1882 |

**y / dN_k10** (frozen ε = 0.8399)

| cell | min | q10 | median | q90 | max | p̂(0.5ε) | p̂(0.75ε) | p̂(0.9ε) | p̂(1.0ε) | p̂(1.1ε) | p̂(1.25ε) | p̂(1.5ε) | p̂(2.0ε) | ε for rel (×ε) | ε for not-unr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot100k_noWD (8 pairs) | 0.8221 | 0.8274 | 0.8330 | 0.8375 | 0.8399 |  |  |  |  |  |  |  |  | | |
| pilot100k_WD (8 pairs) | 0.8112 | 0.8124 | 0.8175 | 0.8205 | 0.8235 |  |  |  |  |  |  |  |  | | |
| pilot100k_smallinit (8 pairs) | 0.8209 | 0.8255 | 0.8288 | 0.8341 | 0.8342 |  |  |  |  |  |  |  |  | | |
| noWD_25000 | 0.9074 | 0.9093 | 0.9128 | 0.9160 | 0.9179 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9171 (1.09×) | 0.9145 |
| WD_25000 | 0.8967 | 0.8987 | 0.9027 | 0.9066 | 0.9120 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9086 (1.08×) | 0.9042 |
| smallinit_25000 | 0.9031 | 0.9072 | 0.9111 | 0.9145 | 0.9163 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9155 (1.09×) | 0.9124 |
| noWD-shared_25000 | 0.8779 | 0.8807 | 0.8848 | 0.8899 | 0.8933 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.8915 (1.06×) | 0.8873 |
| noWD_250000 | 0.7991 | 0.8047 | 0.8112 | 0.8185 | 0.8262 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.8211 (0.98×) | 0.8151 |
| WD_250000 | 0.7779 | 0.7860 | 0.7936 | 0.8006 | 0.8093 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.8041 (0.96×) | 0.7977 |
| smallinit_250000 | 0.7988 | 0.8035 | 0.8083 | 0.8164 | 0.8240 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.8187 (0.97×) | 0.8118 |

**u / dCKA** (frozen ε = 0.0833)

| cell | min | q10 | median | q90 | max | p̂(0.5ε) | p̂(0.75ε) | p̂(0.9ε) | p̂(1.0ε) | p̂(1.1ε) | p̂(1.25ε) | p̂(1.5ε) | p̂(2.0ε) | ε for rel (×ε) | ε for not-unr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot100k_noWD (8 pairs) | 0.0874 | 0.0882 | 0.0903 | 0.0967 | 0.0969 |  |  |  |  |  |  |  |  | | |
| pilot100k_WD (8 pairs) | 0.0669 | 0.0676 | 0.0703 | 0.0744 | 0.0747 |  |  |  |  |  |  |  |  | | |
| pilot100k_smallinit (8 pairs) | 0.0835 | 0.0857 | 0.0874 | 0.0926 | 0.0932 |  |  |  |  |  |  |  |  | | |
| noWD_25000 | 0.1526 | 0.1587 | 0.1651 | 0.1707 | 0.1749 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.59 | 0.1718 (2.06×) | 0.1682 |
| WD_25000 | 0.1219 | 0.1236 | 0.1274 | 0.1325 | 0.1383 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 1.00 | 0.1352 (1.62×) | 0.1300 |
| smallinit_25000 | 0.1460 | 0.1483 | 0.1559 | 0.1632 | 0.1685 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.97 | 0.1658 (1.99×) | 0.1584 |
| noWD-shared_25000 | 0.1111 | 0.1143 | 0.1185 | 0.1238 | 0.1262 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.95 | 1.00 | 0.1253 (1.50×) | 0.1213 |
| noWD_250000 | 0.0735 | 0.0767 | 0.0813 | 0.0855 | 0.0889 | 0.00 | 0.00 | 0.04 | 0.76 | 1.00 | 1.00 | 1.00 | 1.00 | 0.0870 (1.04×) | 0.0828 |
| WD_250000 | 0.0575 | 0.0590 | 0.0620 | 0.0666 | 0.0698 | 0.00 | 0.60 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.0672 (0.81×) | 0.0640 |
| smallinit_250000 | 0.0702 | 0.0743 | 0.0774 | 0.0831 | 0.0867 | 0.00 | 0.00 | 0.19 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 | 0.0840 (1.01×) | 0.0799 |

**h / dN_k10** (frozen ε = 0.7404)

| cell | min | q10 | median | q90 | max | p̂(0.5ε) | p̂(0.75ε) | p̂(0.9ε) | p̂(1.0ε) | p̂(1.1ε) | p̂(1.25ε) | p̂(1.5ε) | p̂(2.0ε) | ε for rel (×ε) | ε for not-unr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot100k_noWD (8 pairs) | 0.7220 | 0.7221 | 0.7270 | 0.7361 | 0.7363 |  |  |  |  |  |  |  |  | | |
| pilot100k_WD (8 pairs) | 0.7205 | 0.7218 | 0.7277 | 0.7351 | 0.7401 |  |  |  |  |  |  |  |  | | |
| pilot100k_smallinit (8 pairs) | 0.7313 | 0.7316 | 0.7366 | 0.7394 | 0.7404 |  |  |  |  |  |  |  |  | | |
| noWD_25000 | 0.8292 | 0.8359 | 0.8427 | 0.8485 | 0.8543 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.8492 (1.15×) | 0.8459 |
| WD_25000 | 0.8312 | 0.8356 | 0.8409 | 0.8467 | 0.8488 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.8474 (1.14×) | 0.8442 |
| smallinit_25000 | 0.8374 | 0.8411 | 0.8478 | 0.8537 | 0.8576 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.8552 (1.16×) | 0.8501 |
| noWD-shared_25000 | 0.7813 | 0.7859 | 0.7926 | 0.8001 | 0.8065 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.8016 (1.08×) | 0.7956 |
| noWD_250000 | 0.6877 | 0.6931 | 0.7041 | 0.7126 | 0.7216 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.7148 (0.97×) | 0.7076 |
| WD_250000 | 0.6779 | 0.6914 | 0.6999 | 0.7094 | 0.7175 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.7123 (0.96×) | 0.7040 |
| smallinit_250000 | 0.6972 | 0.7030 | 0.7109 | 0.7190 | 0.7297 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.7218 (0.97×) | 0.7149 |

**h / Gs** (frozen ε = 0.3032)

| cell | min | q10 | median | q90 | max | p̂(0.5ε) | p̂(0.75ε) | p̂(0.9ε) | p̂(1.0ε) | p̂(1.1ε) | p̂(1.25ε) | p̂(1.5ε) | p̂(2.0ε) | ε for rel (×ε) | ε for not-unr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot100k_noWD (8 pairs) | 0.2368 | 0.2417 | 0.2454 | 0.2519 | 0.2610 |  |  |  |  |  |  |  |  | | |
| pilot100k_WD (8 pairs) | 0.1863 | 0.1896 | 0.1952 | 0.1990 | 0.2016 |  |  |  |  |  |  |  |  | | |
| pilot100k_smallinit (8 pairs) | 0.2405 | 0.2548 | 0.2663 | 0.2831 | 0.3032 |  |  |  |  |  |  |  |  | | |
| noWD_25000 | 0.3228 | 0.3458 | 0.3638 | 0.3805 | 0.3961 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.86 | 1.00 | 1.00 | 0.3908 (1.29×) | 0.3704 |
| WD_25000 | 0.2512 | 0.2563 | 0.2666 | 0.2766 | 0.2868 | 0.00 | 0.00 | 0.77 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2819 (0.93×) | 0.2714 |
| smallinit_25000 | 0.3168 | 0.3423 | 0.3603 | 0.3873 | 0.4260 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.77 | 1.00 | 1.00 | 0.4003 (1.32×) | 0.3758 |
| noWD-shared_25000 | 0.2768 | 0.2918 | 0.3056 | 0.3237 | 0.3520 | 0.00 | 0.00 | 0.00 | 0.44 | 0.97 | 1.00 | 1.00 | 1.00 | 0.3305 (1.09×) | 0.3162 |
| noWD_250000 | 0.2126 | 0.2223 | 0.2379 | 0.2584 | 0.2807 | 0.00 | 0.18 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2675 (0.88×) | 0.2470 |
| WD_250000 | 0.1682 | 0.1728 | 0.1795 | 0.1883 | 0.1983 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1905 (0.63×) | 0.1830 |
| smallinit_250000 | 0.2071 | 0.2281 | 0.2485 | 0.2787 | 0.3236 | 0.00 | 0.08 | 0.85 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2941 (0.97×) | 0.2598 |


## 5. Predictions: report vs recomputed (preregistered verdicts)

| item | report | recomputed | match |
|---|---|---|---|
| P1 noWD u unreliable (T_N10, Gs; both N) | 25k: unr, unr; 250k: T_N10 **rel**, Gs unr; pass False | same (Gs 250k 65/96, CP hi 0.756) | yes |
| P2 noWD y + Δ_Π reliable at 250k | y T_N10 rel, y Gs rel, Π rel (25k all unr); pass True | same. Π on y and on u both 96/96 | yes |
| P3 WD / smallinit u reliable at 250k | WD rel/rel, pass True; smallinit T_N10 rel, Gs **inc**, pass False | same (smallinit Gs 82/96, CP lo 0.781) | yes |
| P4 h localisation (mean s_N(10)(h), pre none) | 25k 0.1574 / 0.1590 / 0.1526 (max diff 0.006); 250k 0.2963 / 0.2999 / 0.2889 (0.011); pass | same. Secondary h_l2: max diff 0.023 (25k), 0.022 (250k), also < 0.05 | yes |
| P5 \|p̂_smallinit − p̂_WD\| ≤ 0.1 | T_N10 0, 0; Gs 0 (25k), **0.146** (250k); pass False | same (14/96) | yes |
| falsifier: noWD u reliable (T_N10 or Gs, either N) | True | True, fired by T_N10 at 250k only | yes |

Shared corpus (noWD 25k, 4 corpora × 24 pairs vs independent noWD 25k). p̂ and decisions match. Mean Δ differences (shared − independent) match to 1e-5 and the bootstrap CIs to within 2e-4:

| rep / rung | p̂ shared | p̂ indep | mean Δ shared | mean Δ indep | diff (my 95% CI) | report diff (CI) |
|---|---|---|---|---|---|---|
| y T_N10 | 0/96 | 0/96 | 0.8850 | 0.9127 | −0.0277 (−0.0286, −0.0269) | −0.0277 (−0.0286, −0.0269) |
| y Gs | 0 | 0 | 0.2419 | 0.2753 | −0.0333 (−0.0345, −0.0323) | −0.0333 (−0.0344, −0.0323) |
| y Δ_Π | 0 | 0 | 0.1226 | 0.1721 | −0.0495 (−0.0505, −0.0485) | −0.0495 (−0.0505, −0.0485) |
| y CKA | 0 | 0 | 0.1106 | 0.1527 | −0.0421 (−0.0430, −0.0412) | −0.0421 (−0.0431, −0.0412) |
| u T_N10 | 0 | 0 | 0.8850 | 0.9129 | −0.0279 (−0.0288, −0.0271) | −0.0279 (−0.0288, −0.0271) |
| u Gs | 0 | 0 | 0.2595 | 0.2985 | −0.0390 (−0.0405, −0.0376) | −0.0390 (−0.0404, −0.0376) |
| u Δ_Π | 0 | 0 | 0.1226 | 0.1721 | −0.0495 (−0.0505, −0.0485) | −0.0495 (−0.0505, −0.0485) |
| u CKA | 0 | 0 | 0.1187 | 0.1648 | −0.0461 (−0.0474, −0.0449) | −0.0461 (−0.0473, −0.0449) |
| h T_N10 | 0 | 0 | 0.7928 | 0.8426 | −0.0497 (−0.0510, −0.0483) | −0.0497 (−0.0512, −0.0483) |
| h Gs | 42/96 | 0 | 0.3075 | 0.3632 | −0.0557 (−0.0595, −0.0518) | −0.0557 (−0.0596, −0.0517) |
| h Δ_Π | 0 | 0 | 0.7416 | 0.7621 | −0.0205 (−0.0211, −0.0200) | −0.0205 (−0.0211, −0.0200) |
| h CKA | 53/96 | 0 | 0.0642 | 0.0888 | −0.0246 (−0.0262, −0.0230) | −0.0246 (−0.0261, −0.0230) |

How to read it: a shared corpus lowers every discrepancy, by 0.02–0.06. Under the 100k ε, all u and y decisions are still unreliable at 25k, so the preregistered expectation "u stays unreliable under a shared corpus" holds only trivially. Read at the same N (T4), the u-minus-y Gs excess is still there with a shared corpus: +0.017, with u > y in 100% of pairs. So conditioning on the corpus does not remove the gauge. That reading is exploratory.

## 6. ε sensitivity (exploratory)

The preregistered *secondary* ε is the pooled pilot 0.9-quantile, which the report omits; the json has it under `p_hat_q90`. I recomputed it and it matches the json in every cell. At 250k:

| cell | ε_q90 | x (decision) at ε_q90 | x at primary ε |
|---|---|---|---|
| noWD u Gs | 0.2030 | 39/96 (unr) | 65 (unr) |
| noWD u CKA | 0.0810 | 45/96 (unr) | 73 (inc) |
| smallinit u Gs | 0.2030 | 58/96 (unr) | 82 (inc) |
| smallinit u CKA | 0.0810 | 79/96 (inc) | 86 (inc) |
| WD u Gs / CKA | 0.2030 / 0.0810 | 96 / 96 (rel) | 96 / 96 |
| all arms u, y T_N10; y Gs; y CKA | 0.8341 / 0.2030 / 0.0810 | 96/96 (rel) | 96 |

I also bootstrapped the 24 pilot y-pairs (ε* = max of a resample, 2000 draws, plus leave-one-out). The resampled max can only be ≤ ε, so this probes downward only. Under it, noWD u Gs stays unreliable in 100% of draws, smallinit u Gs stays inconclusive in 88% (unreliable in 12%), and every WD cell stays reliable. Upward, see the 1.1ε column of T3: a 10% larger ε turns noWD and smallinit u Gs and CKA at 250k into 96/96. **What survives any reasonable ε is the ordering WD < smallinit ≲ noWD in u-Gs and u-CKA at fixed N, and the paired u-minus-y excess.** The noWD/smallinit "unreliable vs inconclusive" labels do not survive.


### 6b. Paired u-vs-y read at fixed N (exploratory)

`u ≤ max y (cell)` = #u-pairs with Δ(u) ≤ max Δ(y) over the 96 y-pairs of the same cell; `cross-fit` = same with the max taken on the other half of the pairs (48/48 split), with the CP decision; `pooled N` = threshold = max Δ(y) over all 288 independent-corpus pairs at that N; `u>y` = fraction of pairs whose u-discrepancy exceeds their own y-discrepancy.

| cell | rung | u ≤ max y (cell) | cross-fit x (dec) | u ≤ pooled-N max y | median Δu−Δy | median Δu/Δy | u>y | mean Δu | mean Δy |
|---|---|---|---|---|---|---|---|---|---|
| noWD_25000 | dN_k10 | 93/96 | 93/96 (rel) | 93/96 | +0.0000 | 1.000 | 0.52 | 0.9129 | 0.9127 |
| noWD_25000 | Gs | 0/96 | 0/96 (unr) | 8/96 | +0.0233 | 1.085 | 1.00 | 0.2985 | 0.2753 |
| noWD_25000 | dCKA | 21/96 | 17/96 (unr) | 21/96 | +0.0121 | 1.078 | 1.00 | 0.1648 | 0.1527 |
| WD_25000 | dN_k10 | 96/96 | 95/96 (rel) | 96/96 | -0.0002 | 1.000 | 0.42 | 0.9027 | 0.9028 |
| WD_25000 | Gs | 96/96 | 96/96 (rel) | 96/96 | -0.0042 | 0.985 | 0.00 | 0.2764 | 0.2806 |
| WD_25000 | dCKA | 96/96 | 96/96 (rel) | 96/96 | -0.0062 | 0.953 | 0.00 | 0.1278 | 0.1341 |
| smallinit_25000 | dN_k10 | 93/96 | 93/96 (rel) | 95/96 | +0.0001 | 1.000 | 0.51 | 0.9110 | 0.9108 |
| smallinit_25000 | Gs | 18/96 | 14/96 (unr) | 18/96 | +0.0165 | 1.059 | 1.00 | 0.2961 | 0.2795 |
| smallinit_25000 | dCKA | 79/96 | 75/96 (inc) | 82/96 | +0.0043 | 1.028 | 0.93 | 0.1558 | 0.1512 |
| noWD-shared_25000 | dN_k10 | 96/96 | 96/96 (rel) | 96/96 | -0.0001 | 1.000 | 0.46 | 0.8850 | 0.8850 |
| noWD-shared_25000 | Gs | 3/96 | 3/96 (unr) | 96/96 | +0.0173 | 1.071 | 1.00 | 0.2595 | 0.2419 |
| noWD-shared_25000 | dCKA | 48/96 | 41/96 (unr) | 96/96 | +0.0081 | 1.073 | 1.00 | 0.1187 | 0.1106 |
| noWD_250000 | dN_k10 | 96/96 | 93/96 (rel) | 96/96 | -0.0011 | 0.999 | 0.25 | 0.8104 | 0.8115 |
| noWD_250000 | Gs | 2/96 | 2/96 (unr) | 9/96 | +0.0198 | 1.108 | 1.00 | 0.2050 | 0.1850 |
| noWD_250000 | dCKA | 11/96 | 10/96 (unr) | 11/96 | +0.0111 | 1.160 | 1.00 | 0.0812 | 0.0699 |
| WD_250000 | dN_k10 | 96/96 | 96/96 (rel) | 96/96 | -0.0010 | 0.999 | 0.25 | 0.7928 | 0.7937 |
| WD_250000 | Gs | 96/96 | 95/96 (rel) | 96/96 | -0.0001 | 0.999 | 0.46 | 0.1843 | 0.1845 |
| WD_250000 | dCKA | 95/96 | 90/96 (inc) | 96/96 | +0.0026 | 1.044 | 1.00 | 0.0624 | 0.0597 |
| smallinit_250000 | dN_k10 | 96/96 | 96/96 (rel) | 96/96 | -0.0010 | 0.999 | 0.32 | 0.8083 | 0.8093 |
| smallinit_250000 | Gs | 18/96 | 17/96 (unr) | 19/96 | +0.0158 | 1.084 | 1.00 | 0.2020 | 0.1864 |
| smallinit_250000 | dCKA | 36/96 | 24/96 (unr) | 46/96 | +0.0091 | 1.132 | 1.00 | 0.0779 | 0.0686 |

cond(W) per cell (quantiles over the 192 runs):

| cell | min | q10 | median | q90 | max |
|---|---|---|---|---|---|
| noWD_25000 | 1.45 | 1.49 | 1.55 | 1.59 | 1.65 |
| WD_25000 | 1.12 | 1.14 | 1.16 | 1.18 | 1.22 |
| smallinit_25000 | 1.48 | 1.57 | 1.63 | 1.69 | 1.76 |
| noWD-shared_25000 | 1.44 | 1.48 | 1.53 | 1.59 | 1.67 |
| noWD_250000 | 1.96 | 2.02 | 2.09 | 2.16 | 2.24 |
| WD_250000 | 1.37 | 1.39 | 1.41 | 1.43 | 1.47 |
| smallinit_250000 | 1.82 | 1.90 | 1.99 | 2.09 | 2.25 |

## 7. Errors and gaps found

None of these changes a reported number or a preregistered decision.

1. **Reporting is incomplete against prereg §4.** §4 says "Every table reports n, x, p̂, CP bounds, decision, ε, and the p̂(ε) curve". The md table prints only `decision x/n`. The json has p̂, CP bounds, ε, mean and median, but **no p̂(ε) curve**: only p̂ at ε_q90, which is not shown in the md. Given the knife edges in §0.5, the curves (T3 here) belong in the report.
2. **P2 reads Δ_Π on u, not y** (`analyze.py` l.624–625: `dec(f"noWD_{N}", "u", "dPi")`). The prereg defines ε_Π from Δ_Π(y, y′) and P2 says "y reliable … and Δ_Π reliable". Because Π_u = Π_y (4.9e-8), the y and u Δ_Π counts are identical: 0/96 at 25k and 96/96 at 250k. Harmless, but the code should say what it tests.
3. **P5 counts vacuous terms.** The 25k terms of P5 (and of P1) are |0 − 0| = 0 and unr/unr only because no representation, not even y, passes the 100k ε at 25k. They carry no information about the gauge, and the write-up should not describe P5 or P1 at 25k as tested. P5 still fails on its 250k Gs term.
4. **Provenance gap: analysis and gate code were not hashed at prereg time.** `prereg/CODE_AT_PREREG.sha256` covers only s2lib.py, train.py, make_plan.py, prep_data.py and the plans. `analyze.py` (e35ca395…) and `crit.py` (07e454c1…) were first hashed in STATUS at 11:01Z. That is after the ε freeze (10:39Z) and after an end-to-end test on S2.2 main INTERIM R = 2, which is one pair per main cell. The current files match those hashes. `gates.py` was amended at 10:42Z, after it froze eps_S2_2, so the version that produced eps_S2_2 is not hashed anywhere. STATUS says the amendment touched only `gate_s21`. My from-features re-derivation of every primary ε (≤ 3.7e-15) makes this immaterial for the values, but it should be stated.
5. **An unlogged, aborted INTERIM analysis.** `src/run_interim_s22.sh` (created 12:24Z) ran `analyze.py s22` at balanced R = 62. Its log stops at 650/1088 ladders, and `logs/analyze_s22_interim.failed` was touched at 13:04Z, the minute `auto_finalize` started the final run. It produced no summary. STATUS has no entry for it. The final analysis reused its cached ladders, which is why the log shows "1973 to compute" of 2688. All 2688 cached ladders pass my checks, so there was no contamination. Prereg §4 allows labelled interim analyses, but this one should be logged.
6. **Cosmetic: the Π check prints "max 0.00".** The true value is 4.9e-8 against the prereg threshold 1e-6, and the report should print it in scientific notation.
7. **ε(h, Gs) is set by a single smallinit pilot pair** (0.3032; the WD pilot max is 0.2016 and the noWD pilot max 0.2610). Because pooling makes ε(h, Gs) loose for WD, **WD 25k h Gs is "reliable 96/96" while every other independent-corpus 25k primary cell (y, u, h × T_N10, Gs) is 0/96.** This follows the prereg, but it is an artefact of pooling arms whose h-Gs levels differ, not evidence that WD 25k representations agree.
8. **Not an error: the bootstrap seed depends only on the rung.** `boot_mean_diff` seeds its RNG with `boot|{key}`, so y, u and h use the same resample indices for a given rung. The marginal CIs are valid, and my independent bootstrap reproduces them, but CIs across representations are not independent.

## 8. Files (all under `~/hrh_v2_stage2_20261008/results/review_final/s22-recompute/`)

- `s22_recompute.py` (sha256 dc6b5f72…): `compute` reads runs, features and ladders; `analyze` does the decisions, predictions, curves and exploratory reads.
- `s22_tables.py` (c2061537…) renders `tables.md`. `s22_epssens.py` (45e05192…) writes `eps_sensitivity.json`.
- `recompute_pairs.json`: per-pair independent values and ladder checks. `recompute_summary.json`: all tables, curves and checks.
- `REPORT.md`: this file.

