"""V-f step 2 (role e0-indep): s_N of the 15 new pairs of arms P, D1..D5 at
every (draw, m, k) from the own neighbour tables (vf_knn.py).

obs       = mean over the predicted anchors (S1.1 V2) of |N_k^A & N_k^B| / k
obs_allH2 = the same over all H2 items of the prefix (descriptive extra)
Also per cell: integer overlap sums, exact k-boundary ties and near ties
(gap < 1e-12 * prefix median D) per model on the predicted anchors, and the
per-anchor overlap counts (npz) for an anchor-level comparison.

Usage: python vf_pairs.py <knndir> <outdir>
"""
from __future__ import annotations

import io
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf_common as C  # noqa: E402

NEAR = 1e-12


def load_tables(knndir, name):
    p = os.path.join(knndir, f"{name}.npz")
    meta = json.load(open(os.path.join(knndir, f"{name}.json")))
    assert C.sha256_file(p) == meta["npz_sha256"], p
    z = np.load(p)
    t = {k: z[k] for k in z.files}
    med = {(int(r[0]), int(r[1])): float(r[2]) for r in t["medians"]}
    return t, med


def main(knndir, outdir):
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    ids, orders, h1 = C.design()
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = load_tables(knndir, name)
        return cache[name]

    rows = []
    per_anchor = {}
    for arm, inp in C.ARMS.items():
        for vit in C.VITS:
            tA, medA = get(vit)
            for enc in C.ENCS:
                bname = f"{enc}.{inp}"
                tB, medB = get(bname)
                pair = f"{vit}__{enc}"
                for d in range(C.N_DRAWS):
                    for m in C.M_GRID:
                        key = f"d{d}_m{m}"
                        anc = tA[key + "_anchor"]
                        assert np.array_equal(anc, tB[key + "_anchor"])
                        pa = C.predicted_anchors(orders[d], h1, m)
                        # positions of the predicted anchors among the H2 rows (same draw order)
                        pos = {int(a): i for i, a in enumerate(anc)}
                        sel = np.array([pos[int(a)] for a in pa], dtype=np.int64)
                        assert np.array_equal(anc[sel], pa)
                        for k in C.K_GRID:
                            nA = tA[key + "_nn"][:, :k]
                            nB = tB[key + "_nn"][:, :k]
                            ov = C.overlap_counts(nA, nB)
                            dA = tA[key + "_dv"]
                            dB = tB[key + "_dv"]
                            gA = dA[:, k] - dA[:, k - 1]
                            gB = dB[:, k] - dB[:, k - 1]
                            tieA = gA == 0
                            tieB = gB == 0
                            nearA = gA < NEAR * medA[(d, m)]
                            nearB = gB < NEAR * medB[(d, m)]
                            ovs = ov[sel]
                            clean = ~(tieA[sel] | tieB[sel])
                            rows.append(dict(
                                arm=arm, pair=pair, vit=vit, enc=enc, text_input=inp, draw=d, m=m, k=k,
                                n_anchors=int(len(sel)), ov_sum=int(ovs.sum()),
                                obs=float(ovs.sum()) / (len(sel) * k),
                                n_allH2=int(len(anc)), ov_sum_allH2=int(ov.sum()),
                                obs_allH2=float(ov.sum()) / (len(anc) * k),
                                n_clean=int(clean.sum()),
                                obs_clean=(float(ovs[clean].sum()) / (clean.sum() * k)) if clean.any() else np.nan,
                                exact_tie_A=int(tieA[sel].sum()), exact_tie_B=int(tieB[sel].sum()),
                                near_tie_A=int(nearA[sel].sum()), near_tie_B=int(nearB[sel].sum()),
                                near_frac_A=float(nearA[sel].mean()), near_frac_B=float(nearB[sel].mean()),
                                exact_tie_A_allH2=int(tieA.sum()), exact_tie_B_allH2=int(tieB.sum()),
                                med_A=medA[(d, m)], med_B=medB[(d, m)],
                            ))
                            per_anchor[f"{arm}|{pair}|d{d}|m{m}|k{k}"] = ov.astype(np.int16)
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    sha_p = C.write_new_bytes(os.path.join(outdir, "vf_obs.parquet"), buf.getvalue())
    buf = io.BytesIO()
    np.savez_compressed(buf, **per_anchor)
    sha_a = C.write_new_bytes(os.path.join(outdir, "vf_per_anchor_overlaps.npz"), buf.getvalue())
    ru = resource.getrusage(resource.RUSAGE_SELF)
    meta = dict(n_rows=len(df), vf_obs_sha256=sha_p, per_anchor_sha256=sha_a,
                knn_jsons={n: C.sha256_file(os.path.join(knndir, n + ".json")) for n in sorted(cache)},
                wall_s=time.time() - t0, cpu_user_s=ru.ru_utime, cpu_sys_s=ru.ru_stime)
    C.write_new_json(os.path.join(outdir, "vf_obs_meta.json"), meta)
    print("rows", len(df), "sha", sha_p, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
