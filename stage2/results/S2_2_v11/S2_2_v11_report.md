# S2.2 v1.1 report

cohort S2_2v11_main, eps sha256 dd649d74fb3a87fd99f049a8676c2fe1751c476093522db7d9f3aba7145935f7, code manifest f2d5d28dfda1b393ac30625e6ab3e29c663416a176e88cfeed2044f7ee155c68

## Checks

- C1: PASS
- C2: PASS
- C4 flow check (median mu_C >= 0.99): PASS
- I1: PASS (violations 0, vacuous 31)
- I2: PASS (violations 0, vacuous 0)
- K1: PASS
- K2: PASS
- K3: PASS
- numerical failures (R-NUM; counted as failed rungs): {'ladder_pairs_by_cell_tag': {}, 'runs_gauge_not_spd': []}

## Primary

- C3(a) BAL-MOM: {25000: {'median': 0.28435432632652485, 'lower_975': 0.2833269160522509, 'frac_ge_0p02': 1.0, 'n': 64, 'pass': True}, 250000: {'median': 0.32644341610795935, 'lower_975': 0.3248465259082973, 'frac_ge_0p02': 1.0, 'n': 64, 'pass': True}}
- C3(b) MOM growth: {25000: {'median': 9.33973690496617, 'lower_975': 9.093589753174273, 'n': 192, 'pass': True}, 250000: {'median': 8.774062077738769, 'lower_975': 8.460899200355284, 'n': 192, 'pass': True}}
- C4a MOM mu_C: {25000: {'median': -0.01657127902358052, 'upper_975': 0.010987318581113854, 'n': 192, 'pass': True}, 250000: {'median': -0.005136144289200869, 'upper_975': 0.011659613424682223, 'n': 192, 'pass': True}}
- I3: 8/8 rejected (Holm)
- I4: (i) realized
- A1: holds
- A3: anchor-sensitive
- A7: BAL-FLOW: (a) False, (b) True; SMALL-FLOW: (a) False, (b) True

## Pre-agreed consequences

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

## Decision tables (primary rungs; final = after reference guard)

| cell | Q | y@Y | u@Y | u@Pool | u@C | uc@Y | ubal@Y | yM@Y | yI@Y |
|---|---|---|---|---|---|---|---|---|---|
| MOM_25000 | dN_k10 | reliable 92/96 | reliable 93/96 | reliable 95/96 | reliable 94/96 | reliable 93/96 | reliable 92/96 | inconclusive 91/96 | reliable 93/96 |
| MOM_25000 | Gs | reliable 96/96 | unreliable 0/96 | unreliable 19/96 | inconclusive 84/96 | unreliable 1/96 | unreliable 18/96 | inconclusive 82/96 | unreliable 33/96 |
| MOM_25000 | dCKA | reliable 96/96 | unreliable 36/96 | inconclusive 74/96 | inconclusive 83/96 | inconclusive 74/96 | inconclusive 73/96 | inconclusive 88/96 | inconclusive 88/96 |
| MOM_250000 | dN_k10 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 |
| MOM_250000 | Gs | reliable 96/96 | unreliable 5/96 | inconclusive 87/96 | inconclusive 90/96 | unreliable 21/96 | unreliable 68/96 | inconclusive 78/96 | unreliable 38/96 |
| MOM_250000 | dCKA | reliable 96/96 | unreliable 23/96 | reliable 93/96 | inconclusive 91/96 | unreliable 66/96 | unreliable 67/96 | inconclusive 86/96 | inconclusive 90/96 |
| WD_25000 | dN_k10 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 |
| WD_25000 | Gs | reliable 94/96 | reliable 96/96 | reliable 96/96 | reliable 93/96 | reliable 96/96 | reliable 96/96 | reliable 92/96 | inconclusive 91/96 |
| WD_25000 | dCKA | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 95/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 93/96 |
| WD_250000 | dN_k10 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 |
| WD_250000 | Gs | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 |
| WD_250000 | dCKA | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 |
| FLOW_25000 | dN_k10 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 95/96 | reliable 95/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 |
| FLOW_25000 | Gs | reliable 95/96 | unreliable 1/96 | unreliable 46/96 | inconclusive 85/96 | unreliable 5/96 | unreliable 3/96 | inconclusive 74/96 | reliable 93/96 |
| FLOW_25000 | dCKA | reliable 96/96 | unreliable 25/96 | unreliable 25/96 | inconclusive 77/96 | inconclusive 71/96 | unreliable 39/96 | inconclusive 70/96 | reliable 93/96 |
| FLOW_250000 | dN_k10 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 |
| FLOW_250000 | Gs | reliable 93/96 | unreliable 32/96 | inconclusive 88/96 | inconclusive 90/96 | inconclusive 84/96 | unreliable 57/96 | inconclusive 83/96 | reliable 92/96 |
| FLOW_250000 | dCKA | reliable 96/96 | unreliable 55/96 | inconclusive 75/96 | inconclusive 83/96 | reliable 93/96 | inconclusive 73/96 | inconclusive 83/96 | reliable 95/96 |
| BAL-FLOW_25000 | dN_k10 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 95/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 |
| BAL-FLOW_25000 | Gs | reliable 96/96 | unreliable 20/96 | unreliable 68/96 | inconclusive 86/96 | unreliable 37/96 | unreliable 20/96 | inconclusive 84/96 | reliable 96/96 |
| BAL-FLOW_25000 | dCKA | reliable 94/96 | unreliable 58/96 | unreliable 58/96 | inconclusive 83/96 | inconclusive 81/96 | unreliable 58/96 | inconclusive 88/96 | reliable 94/96 |
| BAL-FLOW_250000 | dN_k10 | reliable 95/96 | reliable 95/96 | reliable 96/96 | reliable 92/96 | reliable 95/96 | reliable 95/96 | reliable 94/96 | reliable 95/96 |
| BAL-FLOW_250000 | Gs | reliable 96/96 | unreliable 35/96 | reliable 96/96 | inconclusive 83/96 | unreliable 65/96 | unreliable 35/96 | inconclusive 80/96 | reliable 96/96 |
| BAL-FLOW_250000 | dCKA | reliable 96/96 | unreliable 67/96 | reliable 92/96 | inconclusive 84/96 | inconclusive 89/96 | unreliable 67/96 | reliable 93/96 | reliable 96/96 |
| SMALL-FLOW_25000 | dN_k10 | reliable 96/96 | reliable 95/96 | reliable 96/96 | reliable 95/96 | reliable 95/96 | reliable 95/96 | reliable 96/96 | reliable 96/96 |
| SMALL-FLOW_25000 | Gs | reliable 94/96 | unreliable 8/96 | unreliable 43/96 | inconclusive 88/96 | unreliable 24/96 | unreliable 8/96 | inconclusive 78/96 | reliable 94/96 |
| SMALL-FLOW_25000 | dCKA | reliable 96/96 | unreliable 65/96 | inconclusive 88/96 | inconclusive 87/96 | inconclusive 91/96 | unreliable 65/96 | inconclusive 79/96 | reliable 96/96 |
| SMALL-FLOW_250000 | dN_k10 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 | reliable 96/96 |
| SMALL-FLOW_250000 | Gs | reliable 96/96 | reliable 93/96 | reliable 93/96 | reliable 94/96 | reliable 95/96 | reliable 93/96 | reliable 96/96 | reliable 96/96 |
| SMALL-FLOW_250000 | dCKA | reliable 96/96 | reliable 95/96 | reliable 95/96 | reliable 96/96 | reliable 96/96 | reliable 95/96 | reliable 95/96 | reliable 96/96 |
| BAL-MOM_25000 | dN_k10 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 |
| BAL-MOM_25000 | Gs | inconclusive 31/32 | inconclusive (RG) 0/32 | unreliable 11/32 | inconclusive (RG) 24/32 | inconclusive (RG) 0/32 | inconclusive (RG) 4/32 | inconclusive (RG) 25/32 | inconclusive (RG) 8/32 |
| BAL-MOM_25000 | dCKA | inconclusive 30/32 | inconclusive (RG) 9/32 | inconclusive 25/32 | inconclusive (RG) 28/32 | inconclusive (RG) 11/32 | inconclusive (RG) 17/32 | inconclusive (RG) 24/32 | inconclusive (RG) 22/32 |
| BAL-MOM_250000 | dN_k10 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 |
| BAL-MOM_250000 | Gs | reliable 32/32 | unreliable 0/32 | inconclusive 29/32 | inconclusive 28/32 | unreliable 2/32 | unreliable 16/32 | inconclusive 22/32 | unreliable 8/32 |
| BAL-MOM_250000 | dCKA | reliable 32/32 | unreliable 6/32 | inconclusive 31/32 | inconclusive 30/32 | unreliable 19/32 | unreliable 21/32 | inconclusive 29/32 | inconclusive 27/32 |
| SMALL-MOM_25000 | dN_k10 | inconclusive 31/32 | inconclusive (RG) 31/32 | reliable 32/32 | inconclusive (RG) 31/32 | inconclusive (RG) 31/32 | inconclusive (RG) 30/32 | inconclusive (RG) 30/32 | inconclusive (RG) 31/32 |
| SMALL-MOM_25000 | Gs | inconclusive 31/32 | inconclusive (RG) 1/32 | unreliable 13/32 | inconclusive 24/32 | inconclusive (RG) 3/32 | inconclusive (RG) 10/32 | inconclusive (RG) 24/32 | inconclusive (RG) 23/32 |
| SMALL-MOM_25000 | dCKA | reliable 32/32 | inconclusive 22/32 | inconclusive 31/32 | inconclusive (RG) 25/32 | inconclusive 29/32 | inconclusive 24/32 | inconclusive 24/32 | inconclusive 28/32 |
| SMALL-MOM_250000 | dN_k10 | inconclusive 26/32 | inconclusive (RG) 26/32 | reliable 32/32 | inconclusive (RG) 26/32 | inconclusive (RG) 26/32 | inconclusive (RG) 27/32 | inconclusive (RG) 25/32 | inconclusive (RG) 24/32 |
| SMALL-MOM_250000 | Gs | inconclusive 31/32 | inconclusive (RG) 5/32 | inconclusive 28/32 | inconclusive (RG) 20/32 | inconclusive (RG) 21/32 | inconclusive (RG) 22/32 | inconclusive (RG) 19/32 | inconclusive (RG) 19/32 |
| SMALL-MOM_250000 | dCKA | inconclusive 29/32 | inconclusive (RG) 7/32 | inconclusive 31/32 | inconclusive (RG) 25/32 | inconclusive (RG) 24/32 | inconclusive (RG) 18/32 | inconclusive (RG) 19/32 | inconclusive (RG) 23/32 |
| SGD-K_25000 | dN_k10 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 |
| SGD-K_25000 | Gs | inconclusive 30/32 | inconclusive (RG) 0/32 | inconclusive (RG) 0/32 | inconclusive (RG) 28/32 | inconclusive (RG) 0/32 | inconclusive (RG) 13/32 | inconclusive (RG) 26/32 | inconclusive (RG) 0/32 |
| SGD-K_25000 | dCKA | inconclusive 28/32 | inconclusive (RG) 0/32 | unreliable 17/32 | inconclusive (RG) 27/32 | inconclusive (RG) 1/32 | inconclusive (RG) 23/32 | inconclusive (RG) 26/32 | inconclusive (RG) 0/32 |
| SGD-K_250000 | dN_k10 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 |
| SGD-K_250000 | Gs | reliable 32/32 | reliable 32/32 | unreliable 18/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 |
| SGD-K_250000 | dCKA | reliable 32/32 | reliable 32/32 | inconclusive 31/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 | reliable 32/32 |
