"""V-f step 1 (role e0-indep): neighbour tables of one model by FULL argsort.

For one model (a ViT tag or `<enc>.<input>`), under `clip`:
  Xp = hrhv2.prep.preprocess(..., "prh_clip+l2"); K = Xp Xp^T (full pool, stored
  order); D = hrhv2.prep.gram_to_dist(K).
For every draw d and m in the grid, the prefix is order_d[:m] (columns in draw
order); for EVERY H2 item of the prefix (a superset of the predicted anchors),
the full row D[i, prefix] is argsorted with a stable sort, which orders the
candidates by the key (D, position in the draw's order); the anchor itself is
removed; the first 101 candidates (pool indices) and their D values are kept.
Also the median of the strict upper triangle of D on the prefix (draw order).

Usage: python vf_knn.py <outdir> <model>
"""
from __future__ import annotations

import io
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf_common as C  # noqa: E402

BLOCK = 256


def main(outdir: str, name: str):
    t0 = time.time()
    tgt = os.path.join(outdir, f"{name}.npz")
    tgtj = os.path.join(outdir, f"{name}.json")
    if os.path.exists(tgt) or os.path.exists(tgtj):
        raise FileExistsError(tgt)
    ids, orders, h1 = C.design()
    Xp, prov = C.load_preprocessed(name)
    t_load = time.time() - t0
    K = Xp @ Xp.T
    D = C.HP.gram_to_dist(K)
    del K
    t_dist = time.time() - t0 - t_load
    arrays = {}
    meds = {}
    n_ties_self_block = 0
    for d in range(C.N_DRAWS):
        order = orders[d]
        for m in C.M_GRID:
            pre = order[:m]
            h2pos = np.flatnonzero(~h1[pre])
            nn = np.empty((len(h2pos), C.KMAX), dtype=np.int32)
            dv = np.empty((len(h2pos), C.KMAX), dtype=np.float64)
            for lo in range(0, len(h2pos), BLOCK):
                rp = h2pos[lo:lo + BLOCK]
                blk = D[np.ix_(pre[rp], pre)]  # rows: anchors; columns: prefix in draw order
                srt = np.argsort(blk, axis=1, kind="stable")  # full argsort, key (D, position)
                head = srt[:, :C.KMAX + 1]
                is_self = head == rp[:, None]
                if not np.all(is_self.sum(axis=1) == 1):
                    raise RuntimeError("anchor not within its first KMAX+1 sorted entries")
                # rows where self is not first: duplicates at D = 0 with a lower position
                n_ties_self_block += int(np.sum(~is_self[:, 0]))
                pos = head[~is_self].reshape(len(rp), C.KMAX)
                nn[lo:lo + len(rp)] = pre[pos]
                dv[lo:lo + len(rp)] = np.take_along_axis(blk, pos, axis=1)
            sub = D[np.ix_(pre, pre)]
            meds[(d, m)] = float(np.median(C.HU.upper_values(sub)))
            del sub
            arrays[f"d{d}_m{m}_anchor"] = pre[h2pos].astype(np.int32)
            arrays[f"d{d}_m{m}_nn"] = nn
            arrays[f"d{d}_m{m}_dv"] = dv
    arrays["medians"] = np.array([[d, m, meds[(d, m)]] for d in range(C.N_DRAWS) for m in C.M_GRID])
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    sha = C.write_new_bytes(tgt, buf.getvalue())
    ru = resource.getrusage(resource.RUSAGE_SELF)
    meta = dict(prov, npz=os.path.basename(tgt), npz_sha256=sha, wall_s=time.time() - t0,
                load_s=t_load, dist_s=t_dist, cpu_user_s=ru.ru_utime, cpu_sys_s=ru.ru_stime,
                maxrss_kb=ru.ru_maxrss, rows_with_zero_dist_duplicate_before_self=n_ties_self_block,
                blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
                affinity=sorted(os.sched_getaffinity(0)))
    C.write_new_json(tgtj, meta)
    print(name, "done", round(time.time() - t0, 1), "s", sha[:16], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
