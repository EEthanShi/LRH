"""S1.3a driver: train every (arm, N) cell, then compute pair criteria. Usage:
    python run_s13a.py train      # writes runs/<arm>/N<N>/chunk<c>.npz   (skips existing chunks)
    python run_s13a.py criteria   # writes criteria/<arm>_N<N>_pairs.csv and _runs.csv
Uses at most 12 worker processes, 1 BLAS thread each.

v1.1 (2026-10-08, after the v1.0 crash; see ANALYSIS_PLAN_v1.1.md):
  * a diverged run (non-finite parameters or loss, or |param| > 1e6 in a stochastic phase, or loss > 1e8 in GD)
    is a recorded run outcome: outcome = 2 ('diverged'), with div_phase (1 = stochastic, 2 = full-batch GD/polish)
    and div_step. Its parameters are NaN; it is kept, never dropped, and every pair containing it fails every
    criterion (analyze_s13a.py). It can no longer crash the pool.
  * any exception inside one task is caught, written to runs/<arm>/N<N>/chunk<c>.failed, and the other tasks go on;
    `train` exits 1 at the end if any task failed.
  * resumable: a chunk (a fixed set of run ids, vectorised together) is written atomically (tmp + rename) and a
    complete, valid chunk file is skipped on relaunch. Chunks written by v1.0 before the crash are reused (they are
    bitwise reproducible by v1.1, checked in diag/); their missing outcome fields are filled in load_chunk().
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, json, traceback, multiprocessing as mp
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linae_core as lc
import linae_metrics as lm

NS = [256, 4096, 65536, 'inf']
FIN_RUNS, INF_RUNS = 96, 1000
FIN_CHUNK, INF_CHUNK = 48, 125
RUNS_DIR = os.path.join(HERE, 'runs')
CRIT_DIR = os.path.join(HERE, 'criteria')
NPROC = max(1, min(12, int(os.environ.get('LINAE_NPROC', '12'))))   # role cap: <= 12 cores


def chunks_of(N):
    if N == 'inf':
        return [list(range(c * INF_CHUNK, (c + 1) * INF_CHUNK)) for c in range(INF_RUNS // INF_CHUNK)]
    return [list(range(c * FIN_CHUNK, (c + 1) * FIN_CHUNK)) for c in range(FIN_RUNS // FIN_CHUNK)]


def chunk_path(arm, N, c):
    return os.path.join(RUNS_DIR, arm, f'N{N}', f'chunk{c}.npz')


def log(msg):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), msg, flush=True)


# ------------------------------------------------------------------------------------------- training
REQUIRED_KEYS = ('run_ids', 'E', 'W', 'E0', 'W0', 'C0', 'C_end', 'diverged', 'cert_ok', 'rel_gap', 'rel_grad',
                 'steps', 'spec_EEt', 'spec_pred_flow', 'Fstar', 'S_eigs', 'polish')
OUTCOME_NAMES = {0: 'certified', 1: 'uncertified', 2: 'diverged'}


def chunk_complete(path, runs):
    """True iff path is a readable chunk with the expected run ids and all required keys."""
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as z:
            ok = all(k in z.files for k in REQUIRED_KEYS) and np.array_equal(z['run_ids'], np.asarray(runs))
    except Exception:
        ok = False
    if not ok:
        os.replace(path, path + f'.corrupt{int(time.time())}')
    return ok


def train_task(task):
    arm, N, c, runs = task
    path = chunk_path(arm, N, c)
    if chunk_complete(path, runs):
        return path, 0.0, 'cached'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg = lc.ARMS[arm]
    U, Sigma = lc.population()
    R = len(runs)
    E0 = np.stack([lc.init_params(r, cfg['init'])[0] for r in runs])
    W0 = np.stack([lc.init_params(r, cfg['init'])[1] for r in runs])
    if cfg['tied']:
        W0 = lc.T(E0).copy()
    if N == 'inf':
        X = None
        S = np.broadcast_to(Sigma, (R, lc.n, lc.n)).copy()
    else:
        X = np.stack([lc.train_data(N, r, U) for r in runs])
        S = lc.T(X) @ X / N
        S = 0.5 * (S + lc.T(S))
    Fstar = lc.optimum(S, cfg['wd'])
    C0 = lc.imbalance(E0, W0, cfg['tied'])
    rec = dict(run_ids=np.array(runs), E0=E0, W0=W0, C0=C0, Fstar=Fstar,
               S_eigs=np.sort(np.linalg.eigvalsh(S), axis=1)[:, ::-1])
    t0 = time.time()
    if cfg['opt'] == 'gd':
        out = lc.gd_to_certificate(E0, W0, S, cfg['wd'], cfg['tied'], cfg['lr'], Fstar, lc.GD_MAX_STEPS)
        rec['polish'] = np.zeros(R, dtype=bool)
        div_phase = np.where(out['diverged'], 2, 0)
        div_step = out['div_step']
    else:
        rng = lc.rng_of(f'S1.3a/batches/arm={arm}/N={N}/chunk={c}')
        Ep, Wp, alive, track, death_t = lc.stochastic_phase(E0, W0, cfg['opt'], cfg['lr'], cfg['batch'], lc.T_CONST,
                                                            lc.T_DECAY, lc.DECAY_FACTOR, X=X, U=U, rng=rng,
                                                            track_every=lc.TRACK_EVERY, wd=cfg['wd'],
                                                            return_death=True)
        with np.errstate(all='ignore'):
            Fp, gp, rgp = lc.certificate(Ep, Wp, S, cfg['wd'], False, Fstar)
        rec.update(E_pre=Ep, W_pre=Wp, C_pre=lc.imbalance(Ep, Wp, False), rel_gap_pre=gp, rel_grad_pre=rgp,
                   alive_pre=alive, stoch_death_t=death_t,
                   track_t=np.array([x[0] for x in track]), track_lr=np.array([x[1] for x in track]),
                   track_Cnorm=np.stack([x[2] for x in track], 1), track_kappa=np.stack([x[3] for x in track], 1))
        out = lc.gd_to_certificate(Ep, Wp, S, cfg['wd'], False, lc.POLISH_LR, Fstar, lc.POLISH_MAX,
                                   lr_cap_by_bound=True)
        div_phase = np.where(~alive, 1, np.where(out['diverged'], 2, 0))
        div_step = np.where(~alive, death_t, out['div_step'])
        out['diverged'] = out['diverged'] | ~alive
        rec['polish'] = np.ones(R, dtype=bool)
    E = out['E']
    W = lc.T(E).copy() if cfg['tied'] else out['W']
    # belt and braces: any non-finite end state is a divergence even if no check caught it
    div = out['diverged'] | ~lc._finite(E, W)
    div_phase = np.where(div & (div_phase == 0), 2, div_phase)
    E = np.where(div[:, None, None], np.nan, E)
    W = np.where(div[:, None, None], np.nan, W)
    cert_ok = out['cert_ok'] & ~div
    with np.errstate(all='ignore'):
        Cend = lc.imbalance(E, W, cfg['tied'])
        spec = np.full((R, lc.d), np.nan)
        okr = ~div
        spec[okr] = lc.safe_eigvalsh(E[okr] @ lc.T(E[okr]))
        f = lambda c_: (c_ + np.sqrt(c_ * c_ + 4.0)) / 2.0
        spec_pred = np.sort(f(lc.safe_eigvalsh(C0)), axis=1)
    outcome = np.where(div, 2, np.where(cert_ok, 0, 1)).astype(np.int64)
    rec.update(E=E, W=W, steps=out['steps'], diverged=div, F=out['F'], rel_gap=out['rel_gap'],
               rel_grad=out['rel_grad'], cert_ok=cert_ok, C_end=Cend, spec_EEt=spec,
               spec_pred_flow=spec_pred, wall_s=np.array(time.time() - t0),
               outcome=outcome, div_phase=div_phase.astype(np.int64), div_step=div_step.astype(np.int64),
               code_version=np.array(lc.CODE_VERSION))
    tmp = path[:-4] + f'.tmp{os.getpid()}.npz'
    np.savez(tmp, **rec)
    os.replace(tmp, path)                      # atomic: a chunk file exists only when complete
    stale = path[:-4] + '.failed'
    if os.path.exists(stale):
        os.replace(stale, stale + f'.resolved{int(time.time())}')
    nd = int(div.sum())
    return path, time.time() - t0, 'ok' + (f' [diverged {nd}/{R}: runs {list(np.asarray(runs)[div])}]' if nd else '')


def train_task_safe(task):
    """Never lets one task kill the pool: the traceback goes to chunk<c>.failed and the status is FAILED."""
    try:
        return train_task(task)
    except Exception:
        arm, N, c, runs = task
        path = chunk_path(arm, N, c)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path[:-4] + '.failed', 'w') as fh:
                fh.write(traceback.format_exc())
        except Exception:
            pass
        return path, 0.0, 'FAILED ' + traceback.format_exc().strip().splitlines()[-1]


def cost(task):
    arm, N, c, runs = task
    opt = lc.ARMS[arm]['opt']
    base = 100.0 if opt != 'gd' else 1.0
    return base * len(runs) * (3.0 if N == 'inf' else 1.0)


def do_train():
    tasks = [(arm, N, c, runs) for arm in lc.ARMS for N in NS for c, runs in enumerate(chunks_of(N))]
    tasks.sort(key=lambda t: -cost(t))
    todo = [t for t in tasks if not chunk_complete(chunk_path(t[0], t[1], t[2]), t[3])]
    log(f'train: {len(tasks)} tasks, {len(tasks) - len(todo)} already complete (skipped), {len(todo)} to run, '
        f'{NPROC} procs, code {lc.CODE_VERSION}')
    nfail = 0
    with mp.Pool(NPROC) as pool:
        for path, sec, st in pool.imap_unordered(train_task_safe, todo):
            log(f'  {st} {os.path.relpath(path, HERE)} {sec:.1f}s')
            nfail += st.startswith('FAILED')
    log(f'train: finished, {nfail} failed task(s)')
    return nfail


# ------------------------------------------------------------------------------------------- criteria
def load_chunk(path):
    p = dict(np.load(path))
    R = len(p['run_ids'])
    if 'outcome' not in p:
        # written by v1.0 before the crash. v1.0 could only finish a chunk in which no run was non-finite
        # (a NaN row made the polish raise LinAlgError), so there is nothing to fill but defaults.
        assert not p['diverged'].any(), path
        p['outcome'] = np.where(p['cert_ok'], 0, 1).astype(np.int64)
        p['div_phase'] = np.zeros(R, np.int64)
        p['div_step'] = np.full(R, -1, np.int64)
        if 'alive_pre' in p:
            assert p['alive_pre'].all(), path
            p['stoch_death_t'] = np.full(R, -1, np.int64)
        p['code_version'] = np.array('v1.0')
    return p


def load_cell(arm, N):
    parts = [load_chunk(chunk_path(arm, N, c)) for c in range(len(chunks_of(N)))]
    keys = set(parts[0])
    assert all(set(p_) == keys for p_ in parts), f'inconsistent chunk keys in {arm} N={N}'
    cat = {}
    for k in parts[0]:
        if k in ('track_t', 'track_lr'):
            cat[k] = parts[0][k]
        elif parts[0][k].ndim == 0:
            cat[k] = np.array([p[k] for p in parts])
        else:
            cat[k] = np.concatenate([p[k] for p in parts], axis=0)
    assert np.all(cat['run_ids'] == np.arange(len(cat['run_ids'])))
    cat['run_code_version'] = np.concatenate([np.full(len(p_['run_ids']), str(p_['code_version'])) for p_ in parts])
    return cat


def target_features(U, Xs):
    return Xs @ U[:, :lc.d]   # population PCA (isotropic, tied/WD/small-init limit up to O(d) and scale)


def criteria_task(task):
    arm, N = task
    U, Sigma = lc.population()
    Xs = lc.probes(U)
    cell = load_cell(arm, N)
    E = cell['E']
    R = E.shape[0]
    Ht = target_features(U, Xs)
    variants = [('final', E)]
    if 'E_pre' in cell:
        variants.append(('prepolish', np.where(cell['alive_pre'][:, None, None], cell['E_pre'], np.nan)))
    pair_rows = []
    for vname, EE in variants:
        for j in range(R // 2):
            a, b = 2 * j, 2 * j + 1
            row = dict(arm=arm, N=str(N), variant=vname, pair=j, runA=a, runB=b,
                       divA=bool(cell['diverged'][a]), divB=bool(cell['diverged'][b]),
                       certA=bool(cell['cert_ok'][a]), certB=bool(cell['cert_ok'][b]),
                       outcomeA=OUTCOME_NAMES[int(cell['outcome'][a])], outcomeB=OUTCOME_NAMES[int(cell['outcome'][b])],
                       metric_error='')
            # a pair with a diverged run gets no metrics (NaN) and fails every criterion in analyze_s13a.py
            if np.isfinite(EE[a]).all() and np.isfinite(EE[b]).all():
                try:
                    with np.errstate(all='ignore'):
                        row.update(lm.pair_criteria(Xs @ EE[a].T, Xs @ EE[b].T, lc.d))
                except Exception as e:      # counted as a failure, reported
                    row['metric_error'] = f'{type(e).__name__}: {e}'
            pair_rows.append(row)
    run_rows = []
    C0n = np.linalg.norm(cell['C0'], axis=(1, 2))
    Cen = np.linalg.norm(cell['C_end'], axis=(1, 2))
    drift = np.linalg.norm(cell['C_end'] - cell['C0'], axis=(1, 2))
    for r in range(R):
        row = dict(arm=arm, N=str(N), run=r, outcome=OUTCOME_NAMES[int(cell['outcome'][r])],
                   div_phase=int(cell['div_phase'][r]), div_step=int(cell['div_step'][r]),
                   code_version=str(cell['run_code_version'][r]),
                   diverged=bool(cell['diverged'][r]), cert_ok=bool(cell['cert_ok'][r]),
                   rel_gap=float(cell['rel_gap'][r]), rel_grad=float(cell['rel_grad'][r]), steps=int(cell['steps'][r]),
                   C0_norm=float(C0n[r]), Cend_norm=float(Cen[r]), C_drift_abs=float(drift[r]),
                   C_drift_rel=float(drift[r] / C0n[r]) if C0n[r] > 0 else np.nan)
        sp = cell['spec_EEt'][r]
        for i in range(lc.d):
            row[f'eigEEt_{i}'] = float(sp[i])
            row[f'eig_fC0_{i}'] = float(cell['spec_pred_flow'][r][i])
        row['kappa'] = float(sp[-1] / sp[0]) if np.isfinite(sp).all() else np.nan
        if 'rel_gap_pre' in cell:
            row['rel_gap_pre'] = float(cell['rel_gap_pre'][r])
            row['Cpre_norm'] = float(np.linalg.norm(cell['C_pre'][r]))
            row['alive_pre'] = bool(cell['alive_pre'][r])
            row['stoch_death_t'] = int(cell['stoch_death_t'][r])
            spp = lc.safe_eigvalsh((cell['E_pre'][r] @ cell['E_pre'][r].T)[None])[0]
            row['kappa_pre'] = float(spp[-1] / spp[0])
            for ti, tt in enumerate(cell['track_t']):
                row[f'kappa_t{int(tt)}'] = float(cell['track_kappa'][r, ti])
                row[f'Cnorm_t{int(tt)}'] = float(cell['track_Cnorm'][r, ti])
        if np.isfinite(E[r]).all():
            try:
                with np.errstate(all='ignore'):
                    tc = lm.pair_criteria(Xs @ E[r].T, Ht, lc.d, ks=(10,))
                for k_ in ('Delta_kNN', 'Delta_Gs', 'Delta_CKA', 'Delta_proj'):
                    row['target_' + k_] = tc[k_]
            except Exception as e:
                row['target_metric_error'] = f'{type(e).__name__}: {e}'
        run_rows.append(row)
    os.makedirs(CRIT_DIR, exist_ok=True)
    pd.DataFrame(pair_rows).to_csv(os.path.join(CRIT_DIR, f'{arm}_N{N}_pairs.csv'), index=False)
    pd.DataFrame(run_rows).to_csv(os.path.join(CRIT_DIR, f'{arm}_N{N}_runs.csv'), index=False)
    return arm, N, len(pair_rows)


def do_criteria():
    tasks = [(arm, N) for arm in lc.ARMS for N in NS]
    tasks.sort(key=lambda t: t[1] != 'inf')
    log(f'criteria: {len(tasks)} cells')
    with mp.Pool(NPROC) as pool:
        for arm, N, npair in pool.imap_unordered(criteria_task, tasks):
            log(f'  criteria {arm} N={N}: {npair} pair rows')


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'train':
        sys.exit(1 if do_train() else 0)
    elif what == 'criteria':
        do_criteria()
    else:
        raise SystemExit(what)
