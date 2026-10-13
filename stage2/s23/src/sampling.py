"""Exact multinomial count tensors for J = 1 triplets (PREREG section 4.1).

A run's sufficient statistic is C ~ Multinomial(N, pi) over n^3 cells with
pi(x, a, b) = P(x) P*(a|x) P(b). It is drawn hierarchically and exactly:
  N_x ~ Mult(N, P);  N_xa ~ Mult(N_x, P*(.|x));  C[x, a, :] ~ Mult(N_xa, P).
(Two-view data: P*(.|x) -> P_aug(.|v1); the anchor marginal is P by balance.)
Learner B uses C1 = sum_b C (anchor, positive) and C2 = sum_a C (anchor, negative).
"""
from __future__ import annotations

import numpy as np


def _rows(M):
    M = np.asarray(M, dtype=np.float64)
    return M / M.sum(axis=1, keepdims=True)


def sample_counts(N: int, P_anchor: np.ndarray, P_pos_given: np.ndarray, P_neg: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Exact hierarchical draw of the (n, n, n) int64 count tensor."""
    N = int(N)
    P_anchor = np.asarray(P_anchor, dtype=np.float64)
    P_anchor = P_anchor / P_anchor.sum()
    P_neg = np.asarray(P_neg, dtype=np.float64)
    P_neg = P_neg / P_neg.sum()
    Pc = _rows(P_pos_given)
    Nx = rng.multinomial(N, P_anchor)                       # (n,)
    Nxa = rng.multinomial(Nx, Pc)                           # (n, n), row x ~ Mult(N_x, P*(.|x))
    C = rng.multinomial(Nxa, P_neg)                         # (n, n, n)
    return C.astype(np.int64, copy=False)


def sample_gibbs(inst, N: int, rng) -> np.ndarray:
    return sample_counts(N, inst.P, inst.Pstar, inst.P, rng)


def sample_twoview(inst, N: int, rng) -> np.ndarray:
    Pa = inst.Paug
    return sample_counts(N, Pa.sum(axis=1), Pa, inst.P, rng)


def marginals(C: np.ndarray):
    """C1(x, a) = sum_b C;  C2(x, b) = sum_a C."""
    return C.sum(axis=2), C.sum(axis=1)


def sample_B_marginals(inst, N: int, rng) -> tuple:
    """Same law as marginals(sample_gibbs(...)) (a and b are independent given x), drawn
    without the n^3 tensor. Used ONLY by the [Model] Monte Carlo in nstar.py, never for runs."""
    Nx = rng.multinomial(int(N), inst.P / inst.P.sum())
    C1 = rng.multinomial(Nx, _rows(inst.Pstar))
    C2 = rng.multinomial(Nx, np.broadcast_to(inst.P / inst.P.sum(), (inst.n, inst.n)))
    return C1.astype(np.int64), C2.astype(np.int64)
