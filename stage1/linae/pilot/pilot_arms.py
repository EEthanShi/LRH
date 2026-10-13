# PILOT (excluded, labelled): divergence / imbalance behaviour of each arm. Uses run indices 100000+ (disjoint
# from the confirmatory run indices 0..999) and pilot-only data keys.
import sys, time, numpy as np
sys.path.insert(0, '..')
import linae_core as lc

what = sys.argv[1]
U, Sigma = lc.population()
R = int(sys.argv[2]) if len(sys.argv) > 2 else 48
runs = range(100000, 100000 + R)


def inits(kind):
    E = np.stack([lc.init_params(r, kind)[0] for r in runs]); W = np.stack([lc.init_params(r, kind)[1] for r in runs])
    return E, W


t0 = time.time()
if what.startswith('gd:'):
    _, arm = what.split(':')
    cfg = lc.ARMS[arm]
    E, W = inits(cfg['init'])
    S = np.broadcast_to(Sigma, (R, lc.n, lc.n)).copy()
    Fs = lc.optimum(S, cfg['wd'])
    C0 = lc.imbalance(E, W, cfg['tied'])
    out = lc.gd_to_certificate(E, W, S, cfg['wd'], cfg['tied'], cfg['lr'], Fs, 400000)
    ok = ~out['diverged']
    print(arm, 'R', R, 'diverged', out['diverged'].sum(), 'cert', out['cert_ok'].sum(), 'steps q50/max',
          np.quantile(out['steps'], [0.5, 1.0]), 'gap min/max', np.nanmin(out['rel_gap']), np.nanmax(out['rel_gap'][ok]))
    C1 = lc.imbalance(out['E'], out['W'] if not cfg['tied'] else None, cfg['tied'])
    nC0 = np.linalg.norm(C0, axis=(1, 2)); nC1 = np.linalg.norm(C1, axis=(1, 2))
    print('  |C0| q50', np.median(nC0), ' |C_end| q50', np.nanmedian(nC1[ok]))
    ev = np.sort(np.linalg.eigvalsh(out['E'][ok] @ lc.T(out['E'][ok])), axis=1)
    print('  spec(EE^T) median', np.median(ev, 0), ' kappa q10/50/90', np.quantile(ev[:, -1] / ev[:, 0], [.1, .5, .9]))
else:
    opt, lr, N = what.split(':')
    lr = float(lr)
    E, W = inits('default')
    rng = lc.rng_of(f'PILOT/batches/{what}')
    if N == 'inf':
        X = None
    else:
        X = np.stack([lc.sample(lc.rng_of(f'PILOT/data/{N}/{r}'), int(N), U) for r in runs])
    C0 = lc.imbalance(E, W, False)
    nC0 = np.linalg.norm(C0, axis=(1, 2))
    E2, W2, alive, track = lc.stochastic_phase(E, W, opt, lr, 16, 100000, 0, 1.0, X=X, U=U, rng=rng, track_every=10000)
    print(what, 'R', R, 'alive', alive.sum())
    for (t, eta, nC) in track:
        rel = nC / nC0
        print('  t', t, 'rel|C| q10/50/90', np.round(np.nanquantile(rel[alive], [.1, .5, .9]), 4))
    S = (np.broadcast_to(Sigma, (R, lc.n, lc.n)) if X is None else lc.T(X) @ X / X.shape[1]).copy()
    Fs = lc.optimum(S, 0.0)
    F, gap, rg = lc.certificate(E2[alive], W2[alive], S[alive], 0.0, False, Fs[alive])
    print('  end-of-phase rel_gap q50/max', np.quantile(gap, [.5, 1.]))
    ev = np.sort(np.linalg.eigvalsh(E2[alive] @ lc.T(E2[alive])), axis=1)
    print('  spec(EE^T) median', np.median(ev, 0), ' kappa q10/50/90', np.quantile(ev[:, -1] / ev[:, 0], [.1, .5, .9]))
print('  time', round(time.time() - t0, 1), 's')
