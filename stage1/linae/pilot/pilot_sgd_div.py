# PILOT (excluded): divergence rate of minibatch SGD (batch 16) from default init, and kappa trajectory.
import sys, numpy as np
sys.path.insert(0, "..")
import linae_core as lc
opt, lr, N, R, T = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
U, Sigma = lc.population()
runs = range(200000, 200000 + R)
E = np.stack([lc.init_params(r, "default")[0] for r in runs]); W = np.stack([lc.init_params(r, "default")[1] for r in runs])
X = None if N == "inf" else np.stack([lc.sample(lc.rng_of(f"PILOT2/data/{N}/{r}"), int(N), U) for r in runs])
rng = lc.rng_of(f"PILOT2/batches/{sys.argv[1:]}")
for seg in range(T // 50000):
    E, W, alive, _ = lc.stochastic_phase(E, W, opt, lr, 16, 50000, 0, 1.0, X=X, U=U, rng=rng)
    ev = np.sort(np.linalg.eigvalsh(E[alive] @ lc.T(E[alive])), axis=1)
    print(sys.argv[1:], "t", (seg + 1) * 50000, "alive", alive.sum(), "/", R, "kappa q10/50/90", np.round(np.quantile(ev[:, -1] / ev[:, 0], [.1, .5, .9]), 3), "spec med", np.round(np.median(ev, 0), 4), flush=True)
