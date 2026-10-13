"""DIAGNOSTIC: v1.1 must reproduce a chunk written by v1.0 bitwise (same RNG keys, same numerics for finite runs).
Re-trains P-SGD N=inf chunk 0 with the v1.1 code into diag/repro_runs/ and compares every array stored by v1.0.
Usage: python repro_check.py   (writes diag/repro_check.json)"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LIN = os.path.dirname(HERE)
sys.path.insert(0, LIN)
import run_s13a as rs
import linae_core as lc
assert lc.CODE_VERSION == 'v1.1'
arm, N, c = sys.argv[1] if len(sys.argv) > 1 else 'P-SGD', 'inf', 0
old_path = rs.chunk_path(arm, N, c)
rs.RUNS_DIR = os.path.join(HERE, 'repro_runs')
runs = rs.chunks_of(N)[c]
path, sec, st = rs.train_task((arm, N, c, runs))
old = dict(np.load(old_path)); new = dict(np.load(path))
res = dict(arm=arm, N=N, chunk=c, status=st, seconds=round(sec, 1), keys_v10=len(old), compared={}, all_equal=True)
for k, v in old.items():
    if k == 'wall_s':
        continue
    eq = bool(np.array_equal(v, new[k], equal_nan=True)) if v.dtype.kind in 'fc' else bool(np.array_equal(v, new[k]))
    res['compared'][k] = eq
    res['all_equal'] &= eq
res['new_keys'] = sorted(set(new) - set(old))
json.dump(res, open(os.path.join(HERE, 'repro_check.json'), 'w'), indent=1)
print(json.dumps(res), flush=True)
