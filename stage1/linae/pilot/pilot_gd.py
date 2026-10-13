# PILOT (excluded from confirmatory results): stability / C-conservation of full-batch GD from default init,
# population covariance (N = inf), as a function of lr.
import hashlib, sys, numpy as np
n, d = 16, 4
LAM = np.array([4, 3, 2, 1.5] + [0.2] * 12)

def rng_of(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))

rng = rng_of("PILOT/U")
Q, R = np.linalg.qr(rng.standard_normal((n, n))); Q = Q * np.sign(np.diag(R))
Sig = (Q * LAM) @ Q.T
R_ = int(sys.argv[2]) if len(sys.argv) > 2 else 200
rng = rng_of("PILOT/init")
E = rng.standard_normal((R_, d, n)) / 4.0
W = rng.standard_normal((R_, n, d)) / 2.0
lr = float(sys.argv[1])
maxsteps = int(sys.argv[3]) if len(sys.argv) > 3 else 400000
I = np.eye(n)
C0 = E @ E.transpose(0, 2, 1) - W.transpose(0, 2, 1) @ W
lstar = LAM[d:].sum()
done = np.full(R_, -1)
for t in range(maxsteps + 1):
    P = W @ E
    G = -2 * (I - P) @ Sig
    if t % 100 == 0:
        Rm = (I - P)
        with np.errstate(all='ignore'):
            loss = np.einsum('rij,jk,rik->r', Rm, Sig, Rm)
            gap = (loss - lstar) / lstar
        newly = (gap <= 1e-8) & (done < 0)
        done[newly] = t
        if (done >= 0).sum() + np.sum(~np.isfinite(gap)) == R_:
            break
    gE = W.transpose(0, 2, 1) @ G
    gW = G @ E.transpose(0, 2, 1)
    act = (done < 0)[:, None, None]
    E = np.where(act, E - lr * gE, E)
    W = np.where(act, W - lr * gW, W)
ok = np.isfinite(E).all(axis=(1, 2)) & np.isfinite(W).all(axis=(1, 2))
print(f'lr={lr} R={R_} diverged={np.sum(~ok)} certified={np.sum(done>=0)} steps-to-cert q50/q90/max',
      np.quantile(done[done >= 0], [0.5, 0.9, 1.0]) if (done >= 0).any() else None)
E, W, C0 = E[ok], W[ok], C0[ok]
C1 = E @ E.transpose(0, 2, 1) - W.transpose(0, 2, 1) @ W
drift = np.linalg.norm(C1 - C0, axis=(1, 2)) / np.linalg.norm(C0, axis=(1, 2))
print(' C rel drift q50/q90/max', np.quantile(drift, [0.5, 0.9, 1.0]))
f = lambda c: (c + np.sqrt(c * c + 4)) / 2
pred = np.sort(f(np.linalg.eigvalsh(C0)), axis=1)
got = np.sort(np.linalg.eigvalsh(E @ E.transpose(0, 2, 1)), axis=1)
rel = np.abs(got - pred).max(1) / pred.max(1)
print(' spec(EE^T) vs f(spec C0) rel err q50/q90/max', np.quantile(rel, [0.5, 0.9, 1.0]))
print(' kappa(EE^T) q10/q50/q90', np.quantile(got[:, -1] / got[:, 0], [0.1, 0.5, 0.9]),
      ' kappa(f(C0)) q10/q50/q90', np.quantile(pred[:, -1] / pred[:, 0], [0.1, 0.5, 0.9]))
