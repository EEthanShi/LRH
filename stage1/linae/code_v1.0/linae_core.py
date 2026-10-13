"""S1.3a linear-AE GL instance: data, arms, vectorised training with certificates.

Model: x_hat = W E x, E in R^{d x n} (encoder, d=4), W in R^{n x d} (decoder), n=16, no biases.
Loss (training): F(E,W) = tr((I-WE) S (I-WE)^T) + wd * (||E||_F^2 + ||W||_F^2),
S = X^T X / N (uncentred second moment; the data have mean zero) or the exact population covariance (N = inf).
All arithmetic float64 (numpy). Runs of one cell are vectorised along a leading axis R.
"""
import hashlib
import numpy as np

n, d = 16, 4
LAM = np.array([4.0, 3.0, 2.0, 1.5] + [0.2] * 12, dtype=np.float64)
M_PROBE = 256
WD_LAMBDA = 0.05 * 1.5  # 0.05 * lambda_4 (population lambda_4 = 1.5)
SMALL_SIGMA0 = 0.01
CERT_TOL = 1e-8

# Arms. "primary" = the six preregistered arms. Secondary arms are deviations/sensitivity (labelled).
ARMS = {
    'P-GD':        dict(opt='gd',   lr=0.05,  init='default', wd=0.0, tied=False, primary=True),
    'P-Adam':      dict(opt='adam', lr=1e-3,  init='default', wd=0.0, tied=False, batch=16, primary=True),
    # P-SGD lr not fixed by the prereg; 0.02 = largest of {0.05, 0.02, 0.01} with zero divergences in the
    # (excluded) pilot; lr 0.05 diverged in ~20% of pilot runs.
    'P-SGD':       dict(opt='sgd',  lr=0.02,  init='default', wd=0.0, tied=False, batch=16, primary=True),
    'WD':          dict(opt='gd',   lr=0.05,  init='default', wd=WD_LAMBDA, tied=False, primary=True),
    'TIED':        dict(opt='gd',   lr=0.05,  init='default', wd=0.0, tied=True, primary=True),
    'small-init':  dict(opt='gd',   lr=0.05,  init='small',   wd=0.0, tied=False, primary=True),
    # secondary (deviation) arm: small-step full-batch GD = faithful gradient-flow proxy (Theorem 5(a))
    'P-GD-flow':   dict(opt='gd',   lr=0.002, init='default', wd=0.0, tied=False, primary=False),
    # secondary (deviation) arm: WD with the same small step (lr=0.05 diverges for ~1% of default inits)
    'WD-flow':     dict(opt='gd',   lr=0.002, init='default', wd=WD_LAMBDA, tied=False, primary=False),
}

# stochastic-arm schedule (prereg leaves it open): constant lr for T_CONST steps, geometric decay to
# lr*DECAY_FACTOR over T_DECAY steps, then full-batch GD polish (lr<=0.002, capped by a Hessian bound)
# to the certificate. The polish approximates gradient flow, which conserves C, so it does not rebalance.
T_CONST, T_DECAY, DECAY_FACTOR = 400_000, 100_000, 1e-3
POLISH_LR, POLISH_MAX = 0.002, 400_000
GD_MAX_STEPS = 400_000
TRACK_EVERY = 25_000


def seed_of(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'little')


def rng_of(key: str) -> np.random.Generator:
    return np.random.default_rng(seed_of(key))


def population():
    """U (Haar, fixed by hash) and Sigma = U diag(LAM) U^T."""
    rng = rng_of('S1.3a/U/v1')
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    U = Q * np.sign(np.diag(R))
    Sigma = (U * LAM) @ U.T
    return U, 0.5 * (Sigma + Sigma.T)


def sample(rng, num, U):
    z = rng.standard_normal((num, n))
    return (z * np.sqrt(LAM)) @ U.T


def probes(U):
    return sample(rng_of('S1.3a/probes/v1'), M_PROBE, U)


def train_data(N, r, U):
    """Training sample of run r at size N. Shared across arms (common random numbers)."""
    return sample(rng_of(f'S1.3a/data/N={N}/run={r}'), N, U)


def init_params(r, kind):
    """Base draws depend only on the run index r (common random numbers across arms and N)."""
    rng = rng_of(f'S1.3a/init/run={r}')
    ZE = rng.standard_normal((d, n))
    ZW = rng.standard_normal((n, d))
    if kind == 'default':        # N(0, 1/fan_in): fan_in(E) = n, fan_in(W) = d
        return ZE / np.sqrt(n), ZW / np.sqrt(d)
    if kind == 'small':          # every entry N(0, sigma0^2)
        return SMALL_SIGMA0 * ZE, SMALL_SIGMA0 * ZW
    raise ValueError(kind)


# ---------------------------------------------------------------- objective, gradient, certificate
I_n = np.eye(n)


def T(A):
    return np.swapaxes(A, -1, -2)


def objective(E, W, S, wd, tied):
    Wm = T(E) if tied else W
    Rm = I_n - Wm @ E
    loss = np.einsum('rij,rjk,rik->r', Rm, S, Rm)
    reg = wd * ((E ** 2).sum((1, 2)) + (Wm ** 2).sum((1, 2))) if wd > 0 else 0.0
    return loss + reg


def grads(E, W, S, wd, tied):
    """Returns (gE, gW); gW is None for the tied model."""
    if tied:
        G = -2.0 * (I_n - T(E) @ E) @ S
        gE = E @ (G + T(G))
        if wd > 0:
            gE = gE + 4.0 * wd * E
        return gE, None
    G = -2.0 * (I_n - W @ E) @ S
    gE = T(W) @ G
    gW = G @ T(E)
    if wd > 0:
        gE = gE + 2.0 * wd * E
        gW = gW + 2.0 * wd * W
    return gE, gW


def optimum(S, wd):
    """Closed-form global optimum value of F over rank<=d (PCA; with wd: nuclear-norm shrinkage)."""
    ev = np.sort(np.linalg.eigvalsh(S), axis=-1)[..., ::-1]  # descending
    tail = ev[..., d:].sum(-1)
    if wd == 0:
        return tail
    top = ev[..., :d]
    assert np.all(wd < top), 'wd must be below lambda_d-hat'
    return tail + (2.0 * wd - wd ** 2 / top).sum(-1)


def certificate(E, W, S, wd, tied, Fstar):
    F = objective(E, W, S, wd, tied)
    gE, gW = grads(E, W, S, wd, tied)
    gn2 = (gE ** 2).sum((1, 2)) + (0.0 if gW is None else (gW ** 2).sum((1, 2)))
    pn2 = (E ** 2).sum((1, 2)) + (0.0 if tied else (W ** 2).sum((1, 2)))
    Snorm = np.linalg.norm(S, axis=(1, 2))
    rel_grad = np.sqrt(gn2) / (Snorm * np.sqrt(pn2))
    rel_gap = (F - Fstar) / Fstar
    return F, rel_gap, rel_grad


def imbalance(E, W, tied):
    if tied:
        return np.zeros((E.shape[0], d, d))
    return E @ T(E) - T(W) @ W


# ---------------------------------------------------------------- training loops
def _finite(E, W):
    ok = np.isfinite(E).all((1, 2))
    if W is not None:
        ok &= np.isfinite(W).all((1, 2))
    return ok


def gd_to_certificate(E, W, S, wd, tied, lr, Fstar, max_steps, check_every=50, lr_cap_by_bound=False):
    """Full-batch GD with per-run freezing once certified (rel_gap<=tol or rel_grad<=tol).
    Returns E, W, steps, diverged flags, certificate fields at the stopping point."""
    R = E.shape[0]
    E = E.copy(); W = None if tied else W.copy()
    steps = np.zeros(R, dtype=np.int64)
    active = np.ones(R, dtype=bool)
    diverged = np.zeros(R, dtype=bool)
    lr_vec = np.full(R, lr)
    Snorm2 = np.linalg.norm(S, ord=2, axis=(1, 2))
    t = 0
    while True:
        if t % check_every == 0:
            with np.errstate(all='ignore'):
                F, gap, rg = certificate(E, W if not tied else E, S, wd, tied, Fstar)
            bad = ~np.isfinite(F) | (F > 1e8) | ~_finite(E, W)
            newly_div = bad & active
            diverged |= newly_div
            active &= ~bad
            cert = (gap <= CERT_TOL) | (rg <= CERT_TOL)
            active &= ~cert
            if lr_cap_by_bound:
                # safe step: lr <= 1/L, L = 2(|W|^2+|E|^2)|S| + 2|S| (upper bound on the Hessian norm)
                nE = np.linalg.norm(E, ord=2, axis=(1, 2)) ** 2
                nW = nE if tied else np.linalg.norm(W, ord=2, axis=(1, 2)) ** 2
                L = 2.0 * (nE + nW) * Snorm2 * (2.0 if tied else 1.0) + 2.0 * Snorm2 + 4.0 * wd
                lr_vec = np.minimum(lr, 1.0 / L)
            if not active.any() or t >= max_steps:
                break
        a = active
        with np.errstate(all='ignore'):
            gE, gW = grads(E[a], None if tied else W[a], S[a], wd, tied)
            E[a] = E[a] - lr_vec[a, None, None] * gE
            if not tied:
                W[a] = W[a] - lr_vec[a, None, None] * gW
        steps[a] += 1
        t += 1
    with np.errstate(all='ignore'):
        F, gap, rg = certificate(E, W if not tied else E, S, wd, tied, Fstar)
    return dict(E=E, W=W, steps=steps, diverged=diverged, F=F, rel_gap=gap, rel_grad=rg,
                cert_ok=((gap <= CERT_TOL) | (rg <= CERT_TOL)) & ~diverged)


def stochastic_phase(E, W, opt, lr, batch, T_const, T_decay, decay_factor, X=None, U=None, rng=None,
                     track_every=0, wd=0.0):
    """Minibatch SGD / Adam (i.i.d. minibatches with replacement from the run's own training set X[r];
    X=None means N=inf: fresh population samples each step). lr constant for T_const steps, then
    geometric decay to lr*decay_factor over T_decay steps. Diverged runs are frozen (NaN-guarded)."""
    R = E.shape[0]
    E = E.copy(); W = W.copy()
    if opt == 'adam':
        mE = np.zeros_like(E); vE = np.zeros_like(E); mW = np.zeros_like(W); vW = np.zeros_like(W)
        b1, b2, eps = 0.9, 0.999, 1e-8
    alive = np.ones(R, dtype=bool)
    track = []
    ar = np.arange(R)[:, None]
    sq = np.sqrt(LAM)
    T_total = T_const + T_decay
    for t in range(T_total):
        eta = lr if t < T_const else lr * decay_factor ** ((t - T_const + 1) / T_decay)
        if X is None:
            Xb = (rng.standard_normal((R, batch, n)) * sq) @ U.T
        else:
            idx = rng.integers(0, X.shape[1], size=(R, batch))
            Xb = X[ar, idx]
        Sb = T(Xb) @ Xb / batch
        with np.errstate(all='ignore'):
            gE, gW = grads(E, W, Sb, wd, False)
            if opt == 'sgd':
                E -= eta * gE
                W -= eta * gW
            else:
                k = t + 1
                mE = b1 * mE + (1 - b1) * gE; vE = b2 * vE + (1 - b2) * gE * gE
                mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW * gW
                c1 = 1 - b1 ** k; c2 = 1 - b2 ** k
                E -= eta * (mE / c1) / (np.sqrt(vE / c2) + eps)
                W -= eta * (mW / c1) / (np.sqrt(vW / c2) + eps)
        if (t + 1) % 1000 == 0 or t == T_total - 1:
            ok = _finite(E, W) & (np.abs(E).max((1, 2)) < 1e6) & (np.abs(W).max((1, 2)) < 1e6)
            newly = alive & ~ok
            if newly.any():
                E[newly] = np.nan; W[newly] = np.nan
                if opt == 'adam':
                    for A in (mE, vE, mW, vW):
                        A[newly] = 0.0
            alive &= ok
        if track_every and ((t + 1) % track_every == 0):
            C = imbalance(E, W, False)
            with np.errstate(all='ignore'):
                ev = np.full((R, d), np.nan)
                ev[alive] = np.linalg.eigvalsh(E[alive] @ T(E[alive]))
            track.append((t + 1, eta, np.linalg.norm(C, axis=(1, 2)), ev[:, -1] / ev[:, 0]))
    return E, W, alive, track
