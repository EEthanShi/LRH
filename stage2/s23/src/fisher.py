"""Per-example Fisher information I_1 of J = 1 InfoNCE (= Hessian of the population risk at the
population minimizer, thm:peer), dense in theta coordinates (strict upper triangle, row-major),
and the exact delta-method covariance of learner B's normalized ratio estimator on probe entries.

I_1 = sum_x Pi_x L_x Pi_x^T, L_x the Laplacian of row x (grounded at the fixed diagonal entry) with
weights w~_x = w_x + w_x^T, w_x(a,b) = pi(x,a,b) s(1-s)/tau^2, s = sigmoid((k(x,b)-k(x,a))/tau).
Covariance of sqrt(N)(theta_hat - theta*) for learner A: I_1^{-1} (thm:peer part 1) [Thm+CLT].
Learner B (no CLT in the paper): delta method on log K_B = log C1s - log Wm - (1/2)(log C1_xx -
log C2_xx + log C1_yy - log C2_yy), multinomial cell covariance [Model].
"""
from __future__ import annotations

import numpy as np
import torch

DT = torch.float64


def theta_index(n: int, device=None):
    iu = torch.triu_indices(n, n, 1, device=device)
    pid = torch.full((n, n), -1, dtype=torch.long, device=device)
    ar = torch.arange(iu.shape[1], device=device)
    pid[iu[0], iu[1]] = ar
    pid[iu[1], iu[0]] = ar
    return iu, pid


def assemble_I1(W, K, tau: float):
    """Dense I_1 (n_theta x n_theta) on W's device. W: (n,n,n) per-example weights pi."""
    n = int(W.shape[0])
    dev = W.device
    iu, pid = theta_index(n, dev)
    nt = iu.shape[1]
    I1 = torch.zeros((nt, nt), dtype=DT, device=dev)
    ar = torch.arange(n, device=dev)
    for x in range(n):
        Mx = (K[x][None, :] - K[x][:, None]) / tau          # [a, b]
        s = torch.sigmoid(Mx)
        w = W[x] * s * (1.0 - s) / tau ** 2
        wt = w + w.T
        L = torch.diag(wt.sum(dim=1)) - wt
        keep = ar[ar != x]
        idx = pid[x, keep]
        Lk = L[keep][:, keep]
        I1.index_put_((idx[:, None].expand(-1, n - 1), idx[None, :].expand(n - 1, -1)), Lk, accumulate=True)
    return I1


def cholesky_inplace_safe(I1):
    L, info = torch.linalg.cholesky_ex(I1)
    if int(info) != 0:
        raise RuntimeError(f"I_1 Cholesky failed (info={int(info)})")
    return L


def entry_ids(S, n):
    """theta indices (and (i,j) pairs) of the strict upper triangle of K restricted to probes S."""
    S = list(S)
    pairs = [(S[i], S[j]) for i in range(len(S)) for j in range(i + 1, len(S))]
    _, pid = theta_index(n)
    ids = np.array([int(pid[a, b]) for a, b in pairs])
    return ids, pairs


def entry_ids_pairs(pairs, n):
    """theta indices of arbitrary unordered pairs (x, y), x != y."""
    _, pid = theta_index(n)
    ids = np.array([int(pid[a, b]) for a, b in pairs])
    return ids, [tuple(p) for p in pairs]


def cov_A_entries(L, ids):
    """(I_1^{-1})_{ids, ids} via cholesky_solve with unit right-hand sides."""
    nt = L.shape[0]
    E = torch.zeros((nt, len(ids)), dtype=DT, device=L.device)
    E[torch.as_tensor(ids, device=L.device), torch.arange(len(ids), device=L.device)] = 1.0
    X = torch.cholesky_solve(E, L)
    return X[torch.as_tensor(ids, device=L.device)].cpu().numpy()


def sample_theta_A(L, theta_star, N: float, n_draws: int, gen: torch.Generator):
    """theta* + L^{-T} z / sqrt(N), z ~ N(0, I) (Gaussian [Thm+CLT] approximation of A's ERM)."""
    nt = L.shape[0]
    Z = torch.randn((nt, n_draws), generator=gen, dtype=DT, device="cpu").to(L.device)
    X = torch.linalg.solve_triangular(L.T, Z, upper=True)
    return theta_star[:, None] + X / np.sqrt(N)


# ----------------------------------------------------------------------------- learner B (delta method)
def cov_B_entries(inst, pairs):
    """Per-example covariance of log K_hat_B over the given (x, y) pairs, x != y, and of K_hat_B
    itself (delta method), for Gibbs data with uniform or general P. Returns (Cov_logK, Cov_K)."""
    P = inst.P
    Q = inst.Q                      # Q[x, a] = P(x) P*(a|x)  (C1 cell law)
    n = inst.n
    KB = inst.KBt

    # a functional is a list of (coef, type, x, j): type 'a' slice {(x, j, b): all b} (C1 cell),
    # type 'b' slice {(x, a, j): all a} (C2 cell). g = sum coef * indicator(slice) / E[slice].
    def terms(x, y):
        t = []
        q = Q[x, y]
        pp = P[x] * P[y]
        t += [(0.5 / q, "a", x, y), (0.5 / q, "a", y, x)]          # log C1s_xy, E = N q (Q symmetric)
        t += [(-0.5 / pp, "b", x, y), (-0.5 / pp, "b", y, x)]      # -log Wm_xy, E = N P_x P_y
        for z in (x, y):
            t += [(-0.5 / Q[z, z], "a", z, z), (0.5 / (P[z] * P[z]), "b", z, z)]
        return t

    def mass(t1, t2):
        _, ty1, x1, j1 = t1
        _, ty2, x2, j2 = t2
        if x1 != x2:
            return 0.0
        if ty1 == ty2:
            if j1 != j2:
                return 0.0
            return Q[x1, j1] if ty1 == "a" else P[x1] * P[j1]
        # one 'a' slice (x, a=ja, all b) and one 'b' slice (x, all a, b=jb): cell (x, ja, jb)
        ja = j1 if ty1 == "a" else j2
        jb = j2 if ty1 == "a" else j1
        return Q[x1, ja] * P[jb]

    T = [terms(x, y) for (x, y) in pairs]
    m = len(pairs)
    C = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            s = 0.0
            for t1 in T[i]:
                for t2 in T[j]:
                    mm = mass(t1, t2)
                    if mm:
                        s += t1[0] * t2[0] * mm
            C[i, j] = C[j, i] = s
    kv = np.array([KB[x, y] for (x, y) in pairs])
    return C, C * np.outer(kv, kv)
