"""Learner A: Gram-parameterized unit-diagonal InfoNCE ERM (J = 1), PREREG section 5.1.

Risk (per example):  f(K) = sum_{x,a,b} w[x,a,b] * softplus((K[x,b] - K[x,a]) / tau),
w = C / N (a run's counts) or w = pi (population), minimized over the elliptope
{K symmetric, K >= 0, diag K = 1}. The parameter is theta = strict upper triangle of K
(K[x,y] = K[y,x] = theta_xy), so the problem is convex in theta.

Solver paths (one solve):
  1. 'newton': damped Newton on the diag-1 affine space, Newton systems by PCG with the
     dense population Fisher I_1 (fisher.py) as a fixed preconditioner (any positive
     multiple works); every trial point must be positive definite (Cholesky). Success =
     an interior point with max |Newton step| <= step_tol (1e-11) and PCG residual <= 1e-9:
     the PSD constraint is inactive and the KKT conditions reduce to a zero gradient on the
     affine space ('interior certificate').
  2. 'bm': if Newton cannot finish in the interior (PSD constraint active, or the iterate
     is blocked at the boundary), Burer-Monteiro K = Y Y^T with Y in R^{n x p}, p = n,
     unit rows, minimized by L-BFGS (strong Wolfe) on U with Y = U / ||U||_row. If the BM
     solution is interior (lambda_min > 1e-6) it is polished by path 1 ('bm+newton').
Certificate of a BM solution (KKT of the convex problem): Gs = sym gradient,
y_i = (Gs K)_ii, S = Gs - Diag(y); pass iff lambda_min(S) >= -rel*||Gs||_F and
||S Y||_max <= rel*||Gs||_F, rel = cfg solver.bm_rel = 1e-6 (PREREG resolution R8; the draft's
1e-9 is not reached by first-order BM in the PSD-active regime). Duality-gap bound
f(K) - f* <= |<S,K>| + n*max(0,-lambda_min(S)) is reported.
Rank-p BM (p < n) for the non-convex population limits is in this module too (bm_rank).

Lemma T identity check (binding correction (a) of the lemma-check verdict, 2026-10-09):
  theta_hess_bounds   extreme eigenvalues of M = L^{-1} H_theta L^{-T} (L: Cholesky factor of the
                      population Fisher I_1) by Lanczos; lambda_min(H_theta) >= lambda_min(M) lambda_min(I_1).
                      'Unique' = no null space of the Hessian over the off-diagonal coordinates, decided
                      with the margin lambda_min(M) - residual >= UNIQ_MARGIN (1e-3).
  err_bound_K         certified distance to the exact minimizer of an interior solution,
                      ||theta_hat - theta_exact||_2 <= 2 ||g_theta||_2 / lambda_lb(H_theta) (factor 2 for the
                      variation of H over the 1e-11 neighbourhood); it bounds the max-norm error of K.
  lemmaT_decision     PD with margin (lambda_min(K_t) >= max(1e-6, 10 e_t)), uniqueness, curvature-derived
                      residual tolerance tol = e_{t2} + r e_{t1} + 1e-13, violation flag.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

DT = torch.float64


def _softplus(M):
    return torch.clamp(M, min=0.0) + torch.log1p(torch.exp(-M.abs()))


def sym_hollow(G):
    H = G + G.T
    H.fill_diagonal_(0.0)
    return H


def inner(U, V) -> float:
    """theta inner product for symmetric hollow matrices: sum_{x<y} U V."""
    return 0.5 * float((U * V).sum())


def is_pd(K) -> bool:
    _, info = torch.linalg.cholesky_ex(K)
    return int(info) == 0


def lam_min(K) -> float:
    return float(torch.linalg.eigvalsh(K)[0])


class InfoNCE:
    """J = 1 InfoNCE risk on a (n, n, n) weight tensor (torch float64)."""

    def __init__(self, W, tau: float):
        self.W = W
        self.tau = float(tau)
        self.n = int(W.shape[0])
        self._D2 = None

    def _M(self, K):
        return (K[:, None, :] - K[:, :, None]) / self.tau   # M[x,a,b] = (K[x,b]-K[x,a])/tau

    def value(self, K) -> float:
        return float((self.W * _softplus(self._M(K))).sum())

    def value_grad(self, K):
        """Returns (f, Grow): Grow[x,j] = d f / d K[x,j] treating row entries as free."""
        M = self._M(K)
        f = (self.W * _softplus(M)).sum()
        S = self.W * torch.sigmoid(M) / self.tau
        Grow = S.sum(dim=1) - S.sum(dim=2)
        return float(f), Grow

    def grad_theta(self, K):
        f, Grow = self.value_grad(K)
        return f, sym_hollow(Grow), Grow

    # accurate risk differences f(K) - f(K_ref) (no cancellation of the n^3-term sums):
    # sp(M) - sp(M_ref) = log1p(sigmoid(M_ref) * expm1(M - M_ref)), with M - M_ref from K - K_ref.
    def set_ref(self, K):
        self._Kref = K.clone()
        self._sref = torch.sigmoid(self._M(K))

    def dvalue(self, K) -> float:
        dK = K - self._Kref
        dM = (dK[:, None, :] - dK[:, :, None]) / self.tau
        return float((self.W * torch.log1p(self._sref * torch.expm1(dM))).sum())

    def dvalue_grad(self, K):
        dK = K - self._Kref
        dM = (dK[:, None, :] - dK[:, :, None]) / self.tau
        df = (self.W * torch.log1p(self._sref * torch.expm1(dM))).sum()
        M = self._M(K)
        S = self.W * torch.sigmoid(M) / self.tau
        Grow = S.sum(dim=1) - S.sum(dim=2)
        return float(df), Grow

    def prep_hess(self, K):
        s = torch.sigmoid(self._M(K))
        self._D2 = self.W * s * (1.0 - s) / (self.tau ** 2)

    def hvp(self, V):
        """theta-space Hessian-vector product at the K given to prep_hess (V symmetric hollow)."""
        dM = V[:, None, :] - V[:, :, None]
        T = self._D2 * dM
        return sym_hollow(T.sum(dim=1) - T.sum(dim=2))

    def dgs(self, V):
        """Derivative of Gs = sym(Grow) in a symmetric hollow direction V (for BM Hessians)."""
        dM = V[:, None, :] - V[:, :, None]
        T = self._D2 * dM
        H = T.sum(dim=1) - T.sum(dim=2)
        return 0.5 * (H + H.T)


# ----------------------------------------------------------------------------- PCG / Newton
class FisherPrecond:
    """z = I_1^{-1} r in theta coordinates via a dense Cholesky factor L (I_1 = L L^T)."""

    def __init__(self, L, n: int):
        self.L = L
        self.iu = torch.triu_indices(n, n, 1, device=L.device)
        self.n = n

    def __call__(self, R):
        v = R[self.iu[0], self.iu[1]]
        z = torch.cholesky_solve(v[:, None], self.L)[:, 0]
        Z = torch.zeros((self.n, self.n), dtype=R.dtype, device=R.device)
        Z[self.iu[0], self.iu[1]] = z
        return Z + Z.T


def pcg(Aop, b, Minv=None, rtol=1e-10, maxit=300):
    x = torch.zeros_like(b)
    r = b.clone()
    bn = math.sqrt(max(inner(b, b), 0.0))
    if bn == 0.0:
        return x, 0, 0.0
    z = Minv(r) if Minv is not None else r.clone()
    p = z.clone()
    rz = inner(r, z)
    k = 0
    rel = 1.0
    for k in range(1, maxit + 1):
        Ap = Aop(p)
        pAp = inner(p, Ap)
        if pAp <= 0:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rel = math.sqrt(max(inner(r, r), 0.0)) / bn
        if rel <= rtol:
            break
        z = Minv(r) if Minv is not None else r.clone()
        rz_new = inner(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, k, rel


def newton_solve(obj: InfoNCE, K0, precond=None, max_newton=60, step_tol=1e-11, pcg_rtol=1e-10,
                 pcg_maxit=400, max_backtracks=40, tiny_rel=1e-13, max_creep=2):
    """Damped Newton in the PD interior. Returns dict(ok, K, reason, ...).
    Line search: every trial point must be PD (Cholesky). Armijo on the accurate difference
    f(K + s d) - f(K) (InfoNCE.dvalue), except when the Newton decrement is below tiny_rel*|f|:
    then a step is accepted if the theta-gradient norm decreases. max_creep consecutive steps
    with s <= 1/64 -> 'blocked' (boundary)."""
    K = K0.clone()
    K.fill_diagonal_(1.0)
    hist = []
    if not is_pd(K):
        return {"ok": False, "K": K, "reason": "init_not_pd", "iters": 0, "hist": hist}
    f, g, _ = obj.grad_theta(K)
    creep = 0
    for it in range(1, max_newton + 1):
        obj.prep_hess(K)
        obj.set_ref(K)
        d, kpcg, relres = pcg(obj.hvp, -g, precond, pcg_rtol, pcg_maxit)
        gd = inner(g, d)
        if not (gd < 0):
            d = -g
            gd = inner(g, d)
        smax = float(d.abs().max())
        tiny = (-gd) <= tiny_rel * max(abs(f), 1e-300)
        gn = math.sqrt(inner(g, g))
        s = 1.0
        accepted = False
        fn = gnew = None
        for _ in range(max_backtracks):
            Kn = K + s * d
            if is_pd(Kn):
                if tiny:
                    fn, gnew, _ = obj.grad_theta(Kn)
                    if math.sqrt(inner(gnew, gnew)) < gn or gn == 0.0:
                        accepted = True
                        break
                else:
                    dfn = obj.dvalue(Kn)
                    if dfn <= 1e-4 * s * gd:
                        accepted = True
                        break
            s *= 0.5
        hist.append((it, f, -gd, smax, s, kpcg, relres))
        if not accepted:
            return {"ok": False, "K": K, "reason": "blocked", "iters": it, "hist": hist}
        creep = creep + 1 if s <= 1.0 / 64 else 0
        if creep >= max_creep:
            return {"ok": False, "K": Kn, "reason": "blocked_creep", "iters": it, "hist": hist}
        K = Kn
        if gnew is not None:
            f, g = fn, gnew
        else:
            f, g, _ = obj.grad_theta(K)
        if s == 1.0 and smax <= step_tol and relres <= 10 * pcg_rtol:
            return {"ok": True, "K": K, "reason": "converged", "iters": it, "hist": hist,
                    "last_step_max": smax, "pcg_relres": relres, "f": f,
                    "grad_max": float(g.abs().max()), "grad_norm2": math.sqrt(max(inner(g, g), 0.0))}
    return {"ok": False, "K": K, "reason": "max_newton", "iters": max_newton, "hist": hist}


# ----------------------------------------------------------------------------- Burer-Monteiro
def factor_from(K, p=None):
    """Y (n x p) with unit rows such that Y Y^T approximates K (top-p eigenpairs, clipped)."""
    w, V = torch.linalg.eigh(0.5 * (K + K.T))
    n = K.shape[0]
    p = n if p is None else p
    idx = torch.argsort(w, descending=True)[:p]
    Y = V[:, idx] * torch.sqrt(torch.clamp(w[idx], min=0.0))[None, :]
    nr = Y.norm(dim=1, keepdim=True)
    bad = (nr[:, 0] <= 1e-12)
    if bool(bad.any()):
        Y[bad] = 0.0
        Y[bad, 0] = 1.0
        nr = Y.norm(dim=1, keepdim=True)
    return Y / nr


def random_factor(n, p, seed, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) % (2 ** 63))
    U = torch.randn((n, p), generator=g, dtype=DT).to(device)
    return U / U.norm(dim=1, keepdim=True)


def _bm_fg(obj: InfoNCE, U, scale, rel=True):
    r = U.norm(dim=1, keepdim=True)
    Y = U / r
    K = Y @ Y.T
    K.fill_diagonal_(1.0)
    f, Grow = obj.dvalue_grad(K) if rel else obj.value_grad(K)
    G = (Grow + Grow.T) @ Y
    gU = (G - Y * (G * Y).sum(dim=1, keepdim=True)) / r
    return f * scale, gU * scale


def _set_ref_Y(obj: InfoNCE, U):
    Y = U / U.norm(dim=1, keepdim=True)
    K = Y @ Y.T
    K.fill_diagonal_(1.0)
    obj.set_ref(K)


def bm_certificate(obj: InfoNCE, Y, rel=1e-6):
    K = Y @ Y.T
    K.fill_diagonal_(1.0)
    f, Grow = obj.value_grad(K)
    Gs = 0.5 * (Grow + Grow.T)
    y = (Gs @ K).diagonal()
    S = Gs - torch.diag(y)
    lam = float(torch.linalg.eigvalsh(S)[0])
    SY = float((S @ Y).abs().max())
    sc = float(torch.linalg.norm(Gs))
    SK = float((S * K).sum())
    n = K.shape[0]
    return {"f": f, "lam_min_S": lam, "SY_max": SY, "Gs_F": sc, "SK": SK,
            "gap_bound": abs(SK) + n * max(0.0, -lam),
            "rel_lam": (-lam) / sc if sc > 0 else 0.0, "rel_SY": SY / sc if sc > 0 else 0.0,
            "pass": bool((-lam) <= rel * sc and SY <= rel * sc)}


def bm_solve(obj: InfoNCE, Y0, max_evals=20000, history=50, chunk=100, rel=1e-6, scale=None, stall_chunks=30):
    n, p = Y0.shape
    scale = float(n * n) if scale is None else float(scale)
    U = torch.nn.Parameter(Y0.clone())
    opt = torch.optim.LBFGS([U], lr=1.0, max_iter=chunk, max_eval=int(chunk * 1.25) + 1,
                            history_size=history, tolerance_grad=0.0, tolerance_change=0.0,
                            line_search_fn="strong_wolfe")
    nev = [0]

    def closure():
        f, gU = _bm_fg(obj, U.data, scale)
        U.grad = gU
        nev[0] += 1
        return torch.tensor(f, dtype=DT)

    cert = None
    best = None
    stall = 0
    while nev[0] < max_evals:
        _set_ref_Y(obj, U.data)          # risk differences are taken from the chunk's start point
        opt.step(closure)
        Y = U.data / U.data.norm(dim=1, keepdim=True)
        cert = bm_certificate(obj, Y, rel)
        if cert["pass"]:
            break
        score = max(cert["rel_lam"], cert["rel_SY"])
        if best is not None and score >= 0.5 * best:
            stall += 1
            if stall >= stall_chunks:
                break
        else:
            stall = 0
            best = score if best is None else min(best, score)
    Y = U.data / U.data.norm(dim=1, keepdim=True)
    K = Y @ Y.T
    K.fill_diagonal_(1.0)
    return {"K": K, "Y": Y, "cert": cert, "evals": nev[0]}


# ----------------------------------------------------------------------------- one A solve
def solve_A(W, tau, K_init, precond=None, rel=1e-6, max_evals=20000, step_tol=1e-11,
            Y_init=None, newton_first=True, stall_chunks=30, pcg_rtol=1e-10):
    """Certified ERM over the elliptope. W: torch (n,n,n) weights (counts/N or pi).
    Returns (K, info). info['cert_pass'] is the run certificate (section 5.4)."""
    t0 = time.time()
    obj = InfoNCE(W, tau)
    info = {"tau": float(tau)}
    if newton_first and K_init is not None:
        r = newton_solve(obj, K_init, precond, step_tol=step_tol, pcg_rtol=pcg_rtol)
        info["newton"] = {k: r[k] for k in ("ok", "reason", "iters")}
        if r["ok"]:
            K = r["K"]
            info.update(path="newton", cert_pass=True, lam_min_K=lam_min(K), psd_active=False,
                        f=r["f"], newton_last_step=r["last_step_max"], pcg_relres=r["pcg_relres"],
                        grad_theta_max=r["grad_max"], grad_theta_norm2=r["grad_norm2"], time_s=time.time() - t0)
            return K, info
        Kb = r["K"]
    else:
        Kb = K_init
    Y0 = Y_init if Y_init is not None else factor_from(Kb)
    b = bm_solve(obj, Y0, max_evals=max_evals, rel=rel, stall_chunks=stall_chunks)
    K = b["K"]
    lm = lam_min(K)
    info.update(path="bm", bm_evals=b["evals"], cert=b["cert"], cert_pass=b["cert"]["pass"],
                lam_min_K=lm, f=b["cert"]["f"])
    if lm > 1e-6:
        r2 = newton_solve(obj, K, precond, step_tol=step_tol, pcg_rtol=pcg_rtol)
        info["polish"] = {k: r2[k] for k in ("ok", "reason", "iters")}
        if r2["ok"]:
            K = r2["K"]
            info.update(path="bm+newton", cert_pass=True, lam_min_K=lam_min(K), f=r2["f"],
                        newton_last_step=r2["last_step_max"], grad_theta_max=r2["grad_max"],
                        grad_theta_norm2=r2["grad_norm2"])
    info["psd_active"] = bool(info["lam_min_K"] <= 1e-6)
    info["time_s"] = time.time() - t0
    return K, info


# ----------------------------------------------------------------------------- rank-p BM (non-convex)
def riemannian_parts(obj: InfoNCE, Y):
    K = Y @ Y.T
    K.fill_diagonal_(1.0)
    f, Grow = obj.value_grad(K)
    Gs = 0.5 * (Grow + Grow.T)
    EG = 2.0 * Gs @ Y                                   # Euclidean gradient wrt Y
    c = (EG * Y).sum(dim=1, keepdim=True)
    RG = EG - Y * c                                      # Riemannian gradient (oblique manifold)
    return f, K, Gs, EG, RG, c


def riemannian_hess_op(obj: InfoNCE, Y):
    """Returns (hess(V), proj(V)) on the oblique manifold at Y (rows on unit spheres)."""
    f, K, Gs, EG, RG, c = riemannian_parts(obj, Y)
    obj.prep_hess(K)

    def proj(V):
        return V - Y * (V * Y).sum(dim=1, keepdim=True)

    def hess(V):
        V = proj(V)
        dK = V @ Y.T + Y @ V.T
        dK.fill_diagonal_(0.0)
        e2 = 2.0 * Gs @ V + 2.0 * obj.dgs(dK) @ Y
        return proj(e2) - V * c

    return hess, proj, f, RG, EG


def bm_rank(obj: InfoNCE, Y0, max_evals=60000, history=50, chunk=200, rel_grad=1e-9,
            newton_polish=True, polish_iters=30, stall_chunks=8):
    """Non-convex rank-p fit: L-BFGS on U, then Riemannian Newton-CG polish.
    Returns K, Y and the relative Riemannian gradient ||RG||_F / ||EG||_F."""
    n, p = Y0.shape
    scale = float(n * n)
    U = torch.nn.Parameter(Y0.clone())
    opt = torch.optim.LBFGS([U], lr=1.0, max_iter=chunk, max_eval=int(chunk * 1.25) + 1,
                            history_size=history, tolerance_grad=0.0, tolerance_change=0.0,
                            line_search_fn="strong_wolfe")
    nev = [0]

    def closure():
        f, gU = _bm_fg(obj, U.data, scale)
        U.grad = gU
        nev[0] += 1
        return torch.tensor(f, dtype=DT)

    best, stall = None, 0
    relg = float("inf")
    while nev[0] < max_evals:
        _set_ref_Y(obj, U.data)
        opt.step(closure)
        Y = U.data / U.data.norm(dim=1, keepdim=True)
        f, K, Gs, EG, RG, c = riemannian_parts(obj, Y)
        relg = float(torch.linalg.norm(RG) / torch.linalg.norm(EG))
        if relg <= rel_grad:
            break
        if best is not None and relg >= 0.5 * best:
            stall += 1
            if stall >= stall_chunks:
                break
        else:
            stall = 0
            best = relg if best is None else min(best, relg)
    Y = U.data / U.data.norm(dim=1, keepdim=True)
    polish = []
    if newton_polish:
        for _ in range(polish_iters):
            hess, proj, f, RG, EG = riemannian_hess_op(obj, Y)
            relg = float(torch.linalg.norm(RG) / torch.linalg.norm(EG))
            if relg <= rel_grad:
                break
            # truncated CG on Hess d = -RG (stop at negative curvature)
            b = -RG
            x = torch.zeros_like(b)
            r = b.clone()
            pdir = r.clone()
            rr = float((r * r).sum())
            b0 = math.sqrt(rr)
            for _k in range(200):
                Hp = hess(pdir)
                pHp = float((pdir * Hp).sum())
                if pHp <= 0:
                    if _k == 0:
                        x = b
                    break
                a = rr / pHp
                x = x + a * pdir
                r = r - a * Hp
                rr_new = float((r * r).sum())
                if math.sqrt(rr_new) <= 1e-3 * b0 * min(1.0, b0):
                    break
                pdir = r + (rr_new / rr) * pdir
                rr = rr_new
            # retraction with backtracking on the gradient norm
            g0 = float(torch.linalg.norm(RG))
            s = 1.0
            ok = False
            Kc = Y @ Y.T
            Kc.fill_diagonal_(1.0)
            obj.set_ref(Kc)
            for _b in range(30):
                Yn = Y + s * x
                Yn = Yn / Yn.norm(dim=1, keepdim=True)
                Kn = Yn @ Yn.T
                Kn.fill_diagonal_(1.0)
                dfn = obj.dvalue(Kn)
                _, _, _, EGn, RGn, _ = riemannian_parts(obj, Yn)
                if dfn <= 0.0 and float(torch.linalg.norm(RGn)) < g0:
                    ok = True
                    break
                s *= 0.5
            polish.append((relg, s, ok))
            if not ok:
                break
            Y = Yn
    f, K, Gs, EG, RG, c = riemannian_parts(obj, Y)
    relg = float(torch.linalg.norm(RG) / torch.linalg.norm(EG))
    return {"K": K, "Y": Y, "f": f, "rel_grad": relg, "evals": nev[0], "polish": polish}


def hess_extreme_eigs(obj: InfoNCE, Y, k=1, tol=1e-6, maxiter=3000):
    """Smallest and largest eigenvalue of the Riemannian Hessian on the tangent space (Lanczos).
    Normal directions are shifted to a large positive value so they cannot be the minimum."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    hess, proj, f, RG, EG = riemannian_hess_op(obj, Y)
    n, p = Y.shape
    dev = Y.device

    def mv_big(v):
        V = torch.as_tensor(v.reshape(n, p), dtype=DT, device=dev)
        return hess(V).reshape(-1).cpu().numpy()

    op_big = LinearOperator((n * p, n * p), matvec=mv_big, dtype=np.float64)
    lmax = float(eigsh(op_big, k=1, which="LA", tol=tol, maxiter=maxiter, return_eigenvectors=False)[0])
    shift = 10.0 * abs(lmax) + 1e-300

    def mv_small(v):
        V = torch.as_tensor(v.reshape(n, p), dtype=DT, device=dev)
        H = hess(V) + shift * (V - proj(V))
        return H.reshape(-1).cpu().numpy()

    op_small = LinearOperator((n * p, n * p), matvec=mv_small, dtype=np.float64)
    lmin = float(eigsh(op_small, k=1, which="SA", tol=tol, maxiter=maxiter, return_eigenvectors=False)[0])
    return lmin, lmax


# ----------------------------------------------------------------------------- Lemma T identity check
UNIQ_MARGIN = 1e-3          # lambda_min(M) - Ritz residual >= 1e-3  <=>  no null space of H_theta
PD_ABS = 1e-6               # PD margin, absolute part
PD_FACTOR = 10.0            # PD margin, multiple of the certified error bound e_t
ERR_SAFETY = 2.0            # e = ERR_SAFETY * ||g_theta||_2 / lambda_lb(H_theta)
FLOAT_FLOOR = 1e-13         # rounding of the affine map and of the stored K entries


def lanczos_extremes(op, dim, dev, key: str, k: int = 120):
    """k-step Lanczos with full reorthogonalization for a symmetric operator on R^dim (torch float64).
    Returns smallest / largest Ritz values and the residual norms of the two Ritz pairs."""
    import s23common as C
    g = C.torch_gen(key)
    v = torch.randn(dim, generator=g, dtype=DT).to(dev)
    v = v / torch.linalg.norm(v)
    Q = [v]
    al, be = [], []
    w = op(v)
    a = float(w @ v)
    al.append(a)
    w = w - a * v
    blast = None                              # beta_{m}: norm of the next Lanczos vector (residual factor)
    for _ in range(1, min(k, dim)):
        Qm = torch.stack(Q, dim=1)
        w = w - Qm @ (Qm.T @ w)
        w = w - Qm @ (Qm.T @ w)
        b = float(torch.linalg.norm(w))
        if b <= 1e-13 * max(1.0, abs(a)):
            blast = 0.0                       # invariant subspace: Ritz values exact
            break
        be.append(b)
        v = w / b
        Q.append(v)
        w = op(v) - b * Q[-2]
        a = float(w @ v)
        al.append(a)
        w = w - a * v
    if blast is None:
        if len(al) >= dim:
            blast = 0.0                       # full space spanned
        else:
            Qm = torch.stack(Q, dim=1)
            w = w - Qm @ (Qm.T @ w)
            w = w - Qm @ (Qm.T @ w)
            blast = float(torch.linalg.norm(w))
    m = len(al)
    T = np.diag(al) + np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1)
    ev, S = np.linalg.eigh(T)
    res_min = abs(blast * S[-1, 0])
    res_max = abs(blast * S[-1, -1])
    return float(ev[0]), float(ev[-1]), float(res_min), float(res_max), m


def theta_hess_bounds(obj: InfoNCE, K, Lfac, key: str, k: int = 120):
    """Extreme eigenvalues of M = L^{-1} H_theta(K) L^{-T}, H_theta the Hessian of obj over the
    off-diagonal coordinates at K, L the Cholesky factor of I_1 (I_1 = L L^T, theta coordinates)."""
    n = obj.n
    dev = K.device
    iu = torch.triu_indices(n, n, 1, device=dev)
    obj.prep_hess(K)

    def op(v):
        w = torch.linalg.solve_triangular(Lfac.T, v[:, None], upper=True)[:, 0]
        Wm = torch.zeros((n, n), dtype=DT, device=dev)
        Wm[iu[0], iu[1]] = w
        Wm = Wm + Wm.T
        Hw = obj.hvp(Wm)[iu[0], iu[1]]
        return torch.linalg.solve_triangular(Lfac, Hw[:, None], upper=False)[:, 0]

    lmin, lmax, rmin, rmax, m = lanczos_extremes(op, iu.shape[1], dev, key, k)
    return {"lam_min_M": lmin, "lam_max_M": lmax, "res_min": rmin, "res_max": rmax, "lanczos_steps": m}


def lam_min_I1(Lfac, key: str, k: int = 200):
    """lambda_min(I_1) >= 1 / (theta_max + residual), theta_max the largest Ritz value of I_1^{-1}."""
    dev = Lfac.device

    def op(v):
        return torch.cholesky_solve(v[:, None], Lfac)[:, 0]

    lmin, lmax, rmin, rmax, m = lanczos_extremes(op, Lfac.shape[0], dev, key, k)
    return {"lam_min_I1_lb": 1.0 / (lmax + rmax), "ritz_max_inv": lmax, "res": rmax,
            "lam_max_I1_est": 1.0 / lmin if lmin > 0 else float("nan"), "lanczos_steps": m}


def err_bound_K(info: dict, hb: dict, lam_I1: float):
    """Certified-distance bound e (theta 2-norm, hence max-norm of K) of an interior Newton solution;
    inf if the solution is not interior or the Hessian bound is not positive."""
    if info.get("path") not in ("newton", "bm+newton") or "grad_theta_norm2" not in info:
        return float("inf"), float("nan")
    lam_lb = (hb["lam_min_M"] - hb["res_min"]) * lam_I1
    if not lam_lb > 0:
        return float("inf"), lam_lb
    return ERR_SAFETY * float(info["grad_theta_norm2"]) / lam_lb, lam_lb


def lemmaT_decision(K_t1, info_t1, hb_t1, K_t2, info_t2, hb_t2, r: float, lam_I1: float):
    """Lemma T identity check on one sample (an identity check, never a claim).
    applicable  = ERM_{t1} interior, PD with margin, unique (no Hessian null space), both certified;
    tol         = e_{t2} + r e_{t1} + FLOAT_FLOOR (curvature-derived);
    violation   = applicable and residual > tol (=> solver / code error: the identity is exact)."""
    K1 = K_t1.detach().cpu().numpy() if torch.is_tensor(K_t1) else np.asarray(K_t1)
    K2 = K_t2.detach().cpu().numpy() if torch.is_tensor(K_t2) else np.asarray(K_t2)
    M = r * K1 + (1.0 - r)
    np.fill_diagonal(M, 1.0)
    resid = float(np.max(np.abs(K2 - M)))
    e1, lb1 = err_bound_K(info_t1, hb_t1, lam_I1) if hb_t1 is not None else (float("inf"), float("nan"))
    e2, lb2 = err_bound_K(info_t2, hb_t2, lam_I1) if hb_t2 is not None else (float("inf"), float("nan"))
    lamK = float(np.linalg.eigvalsh(K1)[0])
    pd_margin = max(PD_ABS, PD_FACTOR * e1) if np.isfinite(e1) else float("inf")
    interior = info_t1.get("path") in ("newton", "bm+newton")
    pd = bool(interior and lamK >= pd_margin)
    unique = bool(hb_t1 is not None and (hb_t1["lam_min_M"] - hb_t1["res_min"]) >= UNIQ_MARGIN)
    certs = bool(info_t1.get("cert_pass") and info_t2.get("cert_pass"))
    applicable = bool(pd and unique and certs)
    tol = e2 + r * e1 + FLOAT_FLOOR
    return {"resid": resid, "lam_min_K_t1": lamK, "pd_margin": pd_margin, "interior_t1": bool(interior),
            "pd": pd, "unique": unique, "certs": certs, "applicable": applicable,
            "e_t1": e1, "e_t2": e2, "lam_lb_H_t1": lb1, "lam_lb_H_t2": lb2, "tol": tol,
            "violation": bool(applicable and not (resid <= tol)),
            "path_t2": info_t2.get("path")}
