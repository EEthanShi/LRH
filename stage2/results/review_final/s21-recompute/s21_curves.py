"""p_hat(eps) curves (prereg 2.4 'full p_hat_Q(eps) curves always reported'; not in the FINAL report).
Reads results/review_final/s21-recompute/recompute_pair_values_z.json (own Delta mapping) and prereg/eps_S2_1.json.
"""
import json
import os

import numpy as np

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
OUT = os.path.join(ROOT, "results", "review_final", "s21-recompute")
V = json.load(open(os.path.join(OUT, "recompute_pair_values_z.json")))
E = json.load(open(os.path.join(ROOT, "prereg", "eps_S2_1.json")))["eps"]["z"]
MULT = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
RUNGS = ["dN_k10", "dO", "dE", "dMST", "Gs", "dAff", "dSOG_t0p1", "dCKA", "rawG", "dIso"]
SETS = [f"{s}_{N}|z" for N in (50000, 500000) for s in
        ("within_A", "within_B", "within_Atau", "cross_A_B", "cross_A_Atau", "cross_Atau_B", "bern_self", "bern_peer")]
out = {"multipliers": MULT, "eps": {k: E[k] for k in RUNGS}, "curves": {}}
for s in SETS:
    out["curves"][s] = {}
    for k in RUNGS:
        v = np.array(V[s][k], dtype=float)
        if np.all(np.isnan(v)):
            continue
        out["curves"][s][k] = [float(np.mean(v <= m * E[k])) for m in MULT]
json.dump(out, open(os.path.join(OUT, "p_hat_eps_curves.json"), "w"), indent=1)
for s in ("cross_A_B_500000|z", "cross_A_Atau_500000|z", "bern_peer_500000|z"):
    for k in ("dN_k10", "dO", "dE", "dMST", "Gs", "dIso", "dCKA"):
        c = out["curves"][s].get(k)
        if c:
            print(s, k, " ".join(f"{m}x:{p:.2f}" for m, p in zip(MULT, c)))
