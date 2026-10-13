"""Probe sets (PREREG section 4.2), deterministic from K* and the Fisher information only (no data).

S_lad = X (all n items).
S_ex8: one probe in each of 2*n_super distinct leaf clusters, two per supercluster (8 for the frozen
  instance). n_cand random candidate sets (key KEY|probes|ex8); keep the n_keep with the largest
  minimum D-gap (D = 2 - 2K* on S); for each, Fisher-normalized per-example margins of every T_E
  hyperplane between entries at most max_rank_gap ranks apart, for A (I_1^{-1}, thm:peer) and B
  (delta method); choose the set with the largest min(rho_A, rho_B).
S_ex16: one probe per leaf cluster (16 for the frozen instance), same procedure (key KEY|probes|ex16);
  the hard set, used only by the secondary item P2c.
"""
from __future__ import annotations

import numpy as np

import s23common as C
import fisher as F


def candidates(inst, which: str, n_cand: int):
    r = C.rng(f"{C.KEY}|probes|{which}")
    nS, nC, nP = inst.meta["n_super"], inst.meta["n_clu"], inst.meta["n_per"]
    size = 2 * nS if which == "ex8" else nS * nC
    out = np.empty((n_cand, size), dtype=np.int64)
    for c in range(n_cand):
        S = []
        for s in range(nS):
            clus = sorted(r.choice(nC, 2, replace=False)) if which == "ex8" else list(range(nC))
            for cl in clus:
                S.append((s * nC + cl) * nP + int(r.integers(nP)))
        out[c] = S
    return out


def min_dgap(K, S):
    S = np.asarray(S)
    D = np.sort((2.0 - 2.0 * K[np.ix_(S, S)])[np.triu_indices(len(S), 1)])
    return float(np.min(np.diff(D)))


def margins(inst, SigA, SigB, S, max_rank_gap=3):
    """Per-example Fisher-normalized margins of T_E hyperplanes (entries <= max_rank_gap ranks apart).
    SigA / SigB: per-example covariance of the S-entries (triu order) for A (theta) and B (K_B).
    Returns (rho_A_min, rho_B_min, hyperplanes, pairs); a hyperplane is
    (e, f, rho_A, rho_B, sd_A, sd_B) with e, f indices into the triu pair list."""
    S = list(S)
    m = len(S)
    pairs = [(S[i], S[j]) for i in range(m) for j in range(i + 1, m)]
    kA = np.array([inst.K[a, b] for a, b in pairs])
    kB = np.array([inst.KBt[a, b] for a, b in pairs])
    order = np.argsort(-kA, kind="stable")           # ascending D
    hyp = []
    for r1 in range(len(order)):
        for r2 in range(r1 + 1, min(len(order), r1 + max_rank_gap + 1)):
            e, f = order[r1], order[r2]
            va = SigA[e, e] + SigA[f, f] - 2 * SigA[e, f]
            vb = SigB[e, e] + SigB[f, f] - 2 * SigB[e, f]
            hyp.append((int(e), int(f), float(abs(kA[e] - kA[f]) / np.sqrt(va)),
                        float(abs(kB[e] - kB[f]) / np.sqrt(vb)), float(np.sqrt(va)), float(np.sqrt(vb))))
    rA = min(h[2] for h in hyp)
    rB = min(h[3] for h in hyp)
    return rA, rB, hyp, pairs


def select(inst, L, which: str, n_cand: int, n_keep: int, max_rank_gap: int = 3):
    cands = candidates(inst, which, n_cand)
    gaps = np.array([min_dgap(inst.K, S) for S in cands])
    top = np.argsort(-gaps, kind="stable")[:n_keep]
    best = None
    rows = []
    for ci in top:
        S = [int(v) for v in cands[ci]]
        ids, pairs = F.entry_ids(S, inst.n)
        SigA = F.cov_A_entries(L, ids)
        _, SigB = F.cov_B_entries(inst, pairs)
        rA, rB, hyp, _ = margins(inst, SigA, SigB, S, max_rank_gap)
        rows.append({"cand": int(ci), "S": S, "min_dgap": float(gaps[ci]), "rho_A": rA, "rho_B": rB})
        if best is None or min(rA, rB) > best[0]:
            best = (min(rA, rB), S, hyp, SigA, SigB, rA, rB, float(gaps[ci]), ids, pairs)
    _, S, hyp, SigA, SigB, rA, rB, g, ids, pairs = best
    return {"which": which, "S": S, "min_dgap": g, "rho_A": rA, "rho_B": rB, "hyperplanes": hyp,
            "SigA": SigA, "SigB": SigB, "theta_ids": [int(v) for v in ids],
            "pairs": [[int(a), int(b)] for a, b in pairs], "top": rows}
