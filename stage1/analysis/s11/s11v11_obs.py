"""S1.1 v1.1: observed s_N(m, k) per (configuration, pair, draw) from the phase-A
kNN caches (no predictor involved; plan V2/V5, O11, D3).

Per cell (pair, draw, m, k):
  obs, obs_sd        mean / sd over the PREDICTED anchors (plan V2) of |N_k^A & N_k^B| / k
  obs_allH2          the same over all H2 items of the m prefix (draft quantity; descriptive)
  n_tieA, n_tieB     predicted anchors with an exact tie between the k-th and (k+1)-th candidate
  obs_clean          obs on the predicted anchors with no exact boundary tie in either model (O11)
  neartie_A/B        SPEC near-tie fraction (gap < 1e-12 median(D) of the prefix), predicted anchors
  neartie_allH2_A/B  the same over all H2 items
Writes results/S1_1/v11/obs/obs_<config>.parquet (+ .json provenance).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s11_common as C  # noqa: E402
import s11v11_common as V  # noqa: E402
from hrhv2 import structural as st  # noqa: E402


def _col(k):
    return V.VAL_COLS.index(k)


def pair_rows(config, P, des, cacheA, cacheB, medA, medB):
    out = []
    for dd in des["draws"]:
        d = dd["draw"]
        anc = dd["anchor_pos"]
        A, B = cacheA[d], cacheB[d]
        for m in dd["m_grid"]:
            posA, posB = A[f"pos_m{m}"], B[f"pos_m{m}"]
            if not np.array_equal(posA, posB):
                raise RuntimeError("H2 positions differ between models")
            pred = posA < V.M_FIT if m >= V.M_FIT else np.ones(posA.size, bool)
            if not np.array_equal(posA[pred].astype(np.int64), anc[anc < m]):
                raise RuntimeError("predicted anchors do not match the design")
            iA, iB = A[f"idx_m{m}"], B[f"idx_m{m}"]
            vA, vB = A[f"vals_m{m}"], B[f"vals_m{m}"]
            for k in V.KS:
                if k > m - 2:
                    continue
                ov = st.overlap_counts(iA, iB, k) / k
                tA = vA[:, _col(k)] == vA[:, _col(k - 1)]
                tB = vB[:, _col(k)] == vB[:, _col(k - 1)]
                ntA = (vA[:, _col(k)] - vA[:, _col(k - 1)]) < 1e-12 * medA[d][m]
                ntB = (vB[:, _col(k)] - vB[:, _col(k - 1)]) < 1e-12 * medB[d][m]
                clean = ~(tA | tB) & pred
                op = ov[pred]
                out.append({"config": config, "pair": P["pair"], "cls": P["cls"], "primary": P["primary"],
                            "draw": d, "m": int(m), "k": int(k), "is_fit_m": bool(m == V.M_FIT),
                            "chance": k / (m - 1), "n_anchors": int(pred.sum()),
                            "obs": float(op.mean()), "obs_sd": float(op.std(ddof=1)) if op.size > 1 else 0.0,
                            "n_allH2": int(ov.size), "obs_allH2": float(ov.mean()),
                            "obs_allH2_sd": float(ov.std(ddof=1)),
                            "n_tieA": int(tA[pred].sum()), "n_tieB": int(tB[pred].sum()),
                            "n_clean": int(clean.sum()),
                            "obs_clean": float(ov[clean].mean()) if clean.any() else float("nan"),
                            "neartie_A": float(ntA[pred].mean()), "neartie_B": float(ntB[pred].mean()),
                            "neartie_allH2_A": float(ntA.mean()), "neartie_allH2_B": float(ntB.mean())})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="clip,l2,cliplayer,dedup")
    args = ap.parse_args(argv)
    os.makedirs(V.OBS, exist_ok=True)
    for config in args.configs.split(","):
        V.check_authorization(config)
        t0 = time.perf_counter()
        des = V.design(config)
        caches, meds = {}, {}
        for mdl in V.models_for(config):
            cd = V.cache_dir(config, mdl)
            if not os.path.exists(os.path.join(cd, "DONE")):
                raise SystemExit(f"cache missing: {cd}")
            with open(os.path.join(cd, "meta.json")) as f:
                meta = json.load(f)
            caches[mdl] = {dd["draw"]: dict(np.load(os.path.join(cd, f"d{dd['draw']}_knn.npz")))
                           for dd in des["draws"]}
            meds[mdl] = {int(d): {int(m): float(v) for m, v in dm["median_D"].items()}
                         for d, dm in meta["draws"].items()}
        rows = []
        for P in V.pairs_for(config):
            rows += pair_rows(config, P, des, caches[P["a"]], caches[P["b"]], meds[P["a"]], meds[P["b"]])
        df = pd.DataFrame(rows)
        path = os.path.join(V.OBS, f"obs_{config}.parquet")
        df.to_parquet(path, index=False)
        prov = {"config": config, "n_rows": len(df), "n_pairs": int(df["pair"].nunique()),
                "seconds": time.perf_counter() - t0, "parquet_sha256": V.file_sha256(path),
                "code_sha256": {f: V.file_sha256(os.path.join(HERE, f)) for f in
                                ("s11v11_obs.py", "s11v11_common.py", "s11v11_prep.py")},
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(path[:-8] + ".json", "w") as f:
            json.dump(prov, f, indent=1)
        print(config, len(df), "rows", f"{prov['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
