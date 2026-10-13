"""Learner B: unconstrained spectral contrastive ERM (HaoChen et al.), features of dimension >= |X|,
Gram normalized to unit diagonal (PREREG section 5.2), plus the convex population maps used by
the pre-step: ball projections (departure (i), (v)), Higham's nearest correlation matrix (gate G1),
Eckart-Young truncation (departure (ii)), and the D2 gradient trainer.

Empirical risk on a run's counts (C1 = sum_b C, C2 = sum_a C), K = H H^T symmetric PSD:
  L_B(K) = (1/N) [ -2 sum_{x,a} C1[x,a] K[x,a] + sum_{x,b} C2[x,b] K[x,b]^2 ]
         = (1/N) sum_{x,y} Wm[x,y] (K[x,y] - R[x,y])^2 + const,
  Wm = (C2 + C2^T)/2 off-diagonal, C2[x,x] on the diagonal; C1s likewise; R = C1s / Wm.
With features of dimension >= |X| the feasible set is the PSD cone, so the ERM is R when R is PSD,
otherwise the Wm-weighted projection of R onto the PSD cone (FISTA, certified by a duality-gap bound
that converts to ||K - K_hat||_F <= sqrt(gap / min Wm) by strong convexity).
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import minimize


def spectral_parts(C1, C2):
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    Wm = 0.5 * (C2 + C2.T)
    np.fill_diagonal(Wm, np.diag(C2))
    C1s = 0.5 * (C1 + C1.T)
    np.fill_diagonal(C1s, np.diag(C1))
    zero = Wm <= 0
    R = np.where(zero, 0.0, C1s / np.where(zero, 1.0, Wm))
    return Wm, C1s, R, zero


def proj_psd(A):
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    return (V * np.maximum(w, 0.0)) @ V.T, w


def normalize_unit_diag(K):
    d = np.diag(K).copy()
    if np.any(d <= 0):
        raise ValueError("normalize: non-positive diagonal")
    s = 1.0 / np.sqrt(d)
    Kn = K * s[:, None] * s[None, :]
    np.fill_diagonal(Kn, 1.0)
    return Kn


def _certificate(K, R, Wm, zero):
    """Error bound from the projected-gradient mapping: F is mu-strongly convex and L-smooth on
    symmetric matrices (mu = 2 min Wm, L = 2 max Wm over identified cells); with
    K+ = Pi_+(K - grad F(K)/L) and G = L (K - K+), ||K - K_hat||_F <= 2 ||G||_F / mu."""
    Lc = 2.0 * float(Wm.max())
    mu = 2.0 * float(Wm[~zero].min())
    Kp, _ = proj_psd(K - (2.0 * Wm * (K - R)) / Lc)
    Gn = Lc * float(np.linalg.norm(K - Kp))
    S = 2.0 * Wm * (K - R)
    lam = float(np.linalg.eigvalsh(S)[0])
    return {"F": float((Wm * (K - R) ** 2).sum()), "lam_min_S": lam, "SK": float((S * K).sum()),
            "G_F": Gn, "kappa": Lc / mu, "err_F_bound": 2.0 * Gn / mu}


def solve_B(C1, C2, tol_abs=1e-9, max_iter=20000, check_every=10, init="proj"):
    """Certified spectral ERM. Returns (K_unnormalized, K_normalized or None, info).
    Certificate: ||K - K_hat||_F <= tol_abs (bound of _certificate), no zero-weight cell.
    init: FISTA start when the projection is active, 'proj' = Pi_+(R) (runs) or 'zero' (gate G3)."""
    t0 = time.time()
    Wm, C1s, R, zero = spectral_parts(C1, C2)
    nzero = int(zero.sum())
    lamR = float(np.linalg.eigvalsh(R)[0])
    info = {"n_zero_weight_cells": nzero, "lam_min_R": lamR}
    if nzero == 0 and lamR >= 0.0:
        K = R.copy()
        info.update(path="closed_form", proj_active=False, cert_pass=True, err_F_bound=0.0, iters=0)
    else:
        L = 2.0 * float(Wm.max())
        K = proj_psd(R)[0] if init == "proj" else np.zeros_like(R)
        Y = K.copy()
        tk = 1.0
        Fprev = float((Wm * (K - R) ** 2).sum())
        cert = None
        it = 0
        for it in range(1, max_iter + 1):
            G = 2.0 * Wm * (Y - R)
            Kn, _ = proj_psd(Y - G / L)
            Fn = float((Wm * (Kn - R) ** 2).sum())
            if Fn > Fprev:                       # function-value restart
                tk = 1.0
                Y = K.copy()
                G = 2.0 * Wm * (Y - R)
                Kn, _ = proj_psd(Y - G / L)
                Fn = float((Wm * (Kn - R) ** 2).sum())
            tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
            Y = Kn + ((tk - 1.0) / tn) * (Kn - K)
            K, tk, Fprev = Kn, tn, Fn
            if it % check_every == 0:
                cert = _certificate(K, R, Wm, zero)
                if cert["err_F_bound"] <= tol_abs:
                    break
        cert = _certificate(K, R, Wm, zero)
        info.update(path="fista", proj_active=True, iters=it, cert=cert,
                    err_F_bound=cert["err_F_bound"], cert_pass=bool(cert["err_F_bound"] <= tol_abs and nzero == 0))
    try:
        Kn = normalize_unit_diag(K)
    except ValueError:
        Kn = None
        info["cert_pass"] = False
        info["normalize_failed"] = True
    info["time_s"] = time.time() - t0
    return K, Kn, info


# ----------------------------------------------------------------------------- population maps
def ball_projection(Wt, mu, maxiter=20000):
    """argmin_{K >= 0, diag K <= mu} 1/2 ||K - Wt||_F^2 via the dual (L-BFGS-B on y >= 0):
    K(y) = Pi_+(Wt - Diag(y)), d(y) = 1/2||K(y)-Wt||^2 + sum y (K(y)_ii - mu).
    Primal feasible point: Kf = D K(y) D, D_ii = min(1, sqrt(mu/K(y)_ii)).
    Certificate: gap = p(Kf) - d(y) (cancellation-free form); ||Kf - K*||_F <= sqrt(2 gap)."""
    Wt = np.asarray(Wt, dtype=np.float64)
    n = Wt.shape[0]

    def Ky(y):
        return proj_psd(Wt - np.diag(y))[0]

    def negdual(y):
        K = Ky(y)
        d = 0.5 * float(((K - Wt) ** 2).sum()) + float(np.dot(y, np.diag(K) - mu))
        return -d, -(np.diag(K) - mu)

    y0 = np.maximum(np.diag(Wt) - mu, 0.0)
    res = minimize(negdual, y0, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * n,
                   options=dict(maxiter=maxiter, maxfun=2 * maxiter, ftol=1e-16, gtol=1e-14, maxcor=30))
    y = res.x
    K = Ky(y)
    dg = np.diag(K)
    s = np.where(dg > mu, np.sqrt(mu / np.where(dg > 0, dg, 1.0)), 1.0)
    Kf = K * s[:, None] * s[None, :]
    # gap = 1/2||Kf-Wt||^2 - 1/2||K-Wt||^2 - sum y (K_ii - mu)
    gap = 0.5 * float(((Kf - K) * (Kf + K - 2.0 * Wt)).sum()) - float(np.dot(y, dg - mu))
    gap = max(gap, 0.0)
    ev = np.linalg.eigvalsh(Kf)
    return Kf, {"dual_iters": int(res.nit), "dual_msg": str(res.message)[:80], "gap": gap,
                "err_F_bound": math.sqrt(2.0 * gap), "max_diag_violation": float(max(0.0, (dg - mu).max())),
                "diag_active_frac": float(np.mean(np.diag(Kf) >= mu * (1 - 1e-9))),
                "rank_1e-8": int((ev > 1e-8 * ev.max()).sum()), "lam_min": float(ev[0])}


def ncm_higham(A, tol=1e-14, maxit=200000):
    """Higham (2002) alternating projections with Dykstra's correction: nearest correlation matrix
    in the Frobenius norm. Independent algorithm used only by gate G1."""
    Y = np.array(A, dtype=np.float64, copy=True)
    dS = np.zeros_like(Y)
    it = 0
    for it in range(1, maxit + 1):
        Rm = Y - dS
        X, _ = proj_psd(Rm)
        dS = X - Rm
        Yn = X.copy()
        np.fill_diagonal(Yn, 1.0)
        diff = float(np.max(np.abs(Yn - Y)))
        Y = Yn
        if diff < tol:
            break
    return Y, {"iters": it, "last_change": diff, "lam_min": float(np.linalg.eigvalsh(Y)[0])}


def eckart_young(Wt, p):
    w, V = np.linalg.eigh(0.5 * (Wt + Wt.T))
    idx = np.argsort(w)[::-1][:p]
    Wr = (V[:, idx] * w[idx]) @ V[:, idx].T
    return Wr, {"lam_p": float(w[idx[-1]]), "lam_p1": float(np.sort(w)[::-1][p]) if p < len(w) else float("nan")}


# ----------------------------------------------------------------------------- D2 gradient trainer
def train_H(C1, C2, N, seed, lrs=(1e-2, 3e-3, 1e-3), adam_steps=5000, lbfgs_evals=5000, init_scale=0.1,
            device="cpu"):
    """Descriptive D2: gradient training of H in R^{n x n} on the spectral empirical risk with no
    projection (full batch Adam, then L-BFGS). First lr in lrs that does not diverge is used.
    Returns (K = H H^T unnormalized, info) or (None, info) if every lr diverged."""
    import torch
    Wm, C1s, R, zero = spectral_parts(C1, C2)
    Wt = torch.as_tensor(Wm / float(N), dtype=torch.float64, device=device)
    Ct = torch.as_tensor(C1s / float(N), dtype=torch.float64, device=device)
    n = Wm.shape[0]
    attempts = []
    for lr in lrs:
        g = torch.Generator().manual_seed(int(seed) % (2 ** 63))
        H = (init_scale * torch.randn((n, n), generator=g, dtype=torch.float64)).to(device).requires_grad_(True)

        def loss_fn():
            K = H @ H.T
            return -2.0 * (Ct * K).sum() + (Wt * K * K).sum()

        opt = torch.optim.Adam([H], lr=lr)
        l0 = float(loss_fn().detach())
        diverged = False
        for _ in range(adam_steps):
            opt.zero_grad()
            l = loss_fn()
            if not torch.isfinite(l) or abs(float(l.detach())) > 1e6 * (abs(l0) + 1.0):
                diverged = True
                break
            l.backward()
            opt.step()
        if diverged:
            attempts.append({"lr": lr, "diverged": True})
            continue
        lb = torch.optim.LBFGS([H], lr=1.0, max_iter=lbfgs_evals, max_eval=lbfgs_evals,
                               tolerance_grad=0.0, tolerance_change=0.0, history_size=50,
                               line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            l = loss_fn()
            l.backward()
            return l

        lb.step(closure)
        lf = float(loss_fn().detach())
        if not math.isfinite(lf):
            attempts.append({"lr": lr, "diverged": True})
            continue
        attempts.append({"lr": lr, "diverged": False, "final_loss": lf})
        K = (H @ H.T).detach().cpu().numpy()
        return K, {"attempts": attempts, "lr_used": lr, "n_divergences": sum(a["diverged"] for a in attempts)}
    return None, {"attempts": attempts, "lr_used": None, "n_divergences": len(attempts)}
