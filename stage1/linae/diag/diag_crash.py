"""DIAGNOSTIC (not a confirmatory run): reproduce the 2026-10-07 S1.3a crash with the frozen v1.0 code.
For one (arm, N=inf, chunk) task, re-run the stochastic phase exactly as run_s13a.py v1.0 does (same RNG keys,
same inits), with tracking every 1000 steps to locate the step at which a run became non-finite, then call the
v1.0 polish (gd_to_certificate with lr_cap_by_bound=True) to check whether it raises LinAlgError.
Usage: python diag_crash.py <arm> <chunk>   (writes diag/crash_<arm>_c<chunk>.json)"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, json, time, traceback
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'code_v1.0'))   # frozen v1.0 code
import linae_core as lc
assert os.path.dirname(lc.__file__).endswith('code_v1.0'), lc.__file__

arm, c = sys.argv[1], int(sys.argv[2])
N = 'inf'
cfg = lc.ARMS[arm]
INF_CHUNK = 125
runs = list(range(c * INF_CHUNK, (c + 1) * INF_CHUNK))
U, Sigma = lc.population()
R = len(runs)
E0 = np.stack([lc.init_params(r, cfg['init'])[0] for r in runs])
W0 = np.stack([lc.init_params(r, cfg['init'])[1] for r in runs])
S = np.broadcast_to(Sigma, (R, lc.n, lc.n)).copy()
Fstar = lc.optimum(S, cfg['wd'])
rng = lc.rng_of(f'S1.3a/batches/arm={arm}/N={N}/chunk={c}')
t0 = time.time()
Ep, Wp, alive, track = lc.stochastic_phase(E0, W0, cfg['opt'], cfg['lr'], cfg['batch'], lc.T_CONST, lc.T_DECAY,
                                           lc.DECAY_FACTOR, X=None, U=U, rng=rng, track_every=1000, wd=cfg['wd'])
tt = np.array([x[0] for x in track]); Cn = np.stack([x[2] for x in track], 1)
dead = np.where(~alive)[0]
info = dict(arm=arm, N=N, chunk=c, n_runs=R, n_dead_stochastic=int(len(dead)), stochastic_s=round(time.time() - t0, 1),
            dead=[])
for i in dead:
    nanpos = np.where(~np.isfinite(Cn[i]))[0]
    t_first_nan = int(tt[nanpos[0]]) if len(nanpos) else None
    fin = np.isfinite(Cn[i]); last = np.where(fin)[0]
    info['dead'].append(dict(run=int(runs[i]), died_by_step=t_first_nan,
                             last_finite_step=int(tt[last[-1]]) if len(last) else None,
                             Cnorm_last_finite=float(Cn[i][last[-1]]) if len(last) else None,
                             Cnorm_init=float(np.linalg.norm(lc.imbalance(E0[i:i+1], W0[i:i+1], False)))))
try:
    lc.gd_to_certificate(Ep, Wp, S, cfg['wd'], False, lc.POLISH_LR, Fstar, 100, lr_cap_by_bound=True)
    info['v1_polish'] = 'no exception (first 100 polish steps)'
except Exception as e:
    info['v1_polish'] = f'{type(e).__name__}: {e}'
    info['v1_polish_tb_last'] = traceback.format_exc().strip().splitlines()[-3:]
json.dump(info, open(os.path.join(HERE, f'crash_{arm}_c{c}.json'), 'w'), indent=1)
print(json.dumps(info), flush=True)
