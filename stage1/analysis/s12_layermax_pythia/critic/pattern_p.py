# Exploratory (not citable): exact one-sided p of H1 over 7! relabellings when every ViT shows the same calibrated rank pattern
import itertools, numpy as np
P = np.array(list(itertools.permutations(range(1, 8))), float)   # sigma(i) for positions 1..7
def p_of(cal):   # cal: calibrated ranks of sizes 1..7 (raw ranks = 1..7)
    raw = np.arange(1, 8, dtype=float); cal = np.array(cal, float)
    def rho(S, r):
        Sc = S - S.mean(1, keepdims=True); rc = r - r.mean()
        return (Sc @ rc) / (np.sqrt((Sc**2).sum(1)) * np.sqrt((rc**2).sum()))
    D = rho(P, cal) - rho(P, raw); d0 = D[0]
    return d0, np.mean(D <= d0 + 1e-9)
pats = {"top pair swapped (2.8b<->6.9b)": [1,2,3,4,5,7,6],
        "top 3 reversed": [1,2,3,4,7,6,5],
        "top 4 reversed": [1,2,3,7,6,5,4],
        "top 5 reversed": [1,2,7,6,5,4,3],
        "bottom pair swapped (70m<->160m)": [2,1,3,4,5,6,7],
        "top 3 tied (censor-like)": [1,2,3,4,6,6,6],
        "6.9b drops to rank 4": [1,2,3,5,6,7,4],
        "6.9b drops to rank 1": [2,3,4,5,6,7,1],
        "all 3 adjacent disjoint swaps (1-2,3-4,5-6)": [2,1,4,3,6,5,7],
        "flat after 1b (ranks 1,2,3 then 4 ties)": [1,2,3,5.5,5.5,5.5,5.5],
        "full reversal": [7,6,5,4,3,2,1]}
for k, c in pats.items():
    d, p = p_of(c); print(f"{k:45s} dTr={d:+.3f}  p1={p:.4f}  ({p*5040:.0f}/5040)")
