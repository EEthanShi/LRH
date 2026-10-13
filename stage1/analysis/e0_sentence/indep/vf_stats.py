"""V-f step 4 (role e0-indep): every P1, P2b, P2c statistic recomputed with the
own code (vf_rules.py) from E0's obs files (arms P, D1..D4, A = D5), and, as a
cross-check, from the own V-f obs (vf_obs.parquet). Old S1.1 VL pairs for the
P1 companion corr(e_k, -e_m) come from the frozen S1.1 obs_clip.parquet.

Usage: python vf_stats.py <outdir> <vf_obs.parquet>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf_common as C  # noqa: E402
import vf_rules as R  # noqa: E402

E0_OBS = os.path.join(C.ROOT, "results", "E0_sentence", "obs")
S11_OBS = os.path.join(C.ROOT, "results", "S1_1", "v11", "obs", "obs_clip.parquet")
NEW_PAIRS = [(v, e) for v in sorted(C.VITS) for e in C.ENCS]
NEW_NAMES = [f"{v}__{e}" for v, e in NEW_PAIRS]


def rows_from_df(df, pairs):
    df = df[df["pair"].isin(pairs)]
    out = {}
    for r in df.itertuples(index=False):
        out[(r.pair, int(r.draw), int(r.m), int(r.k))] = (float(r.obs), int(r.n_anchors))
    exp = len(pairs) * C.N_DRAWS * len(C.M_GRID) * len(C.K_GRID)
    assert len(out) == exp == len(df), (len(out), exp, len(df))
    return out


def per_draw_cells(rows, d):
    return {(p, m, k): (s, 0.5 / (n * k)) for (p, dd, m, k), (s, n) in rows.items() if dd == d}


def all_stats(arm_rows, old_vl_rows):
    """arm_rows: dict arm -> rows (new pairs). Returns the statistics dict."""
    out = {}
    P = arm_rows["P"]
    cellsP = R.cells_from_obs(P)
    # ---------------- P1 ----------------
    ek, em, sm, sm_all, sm_110 = {}, {}, {}, {}, {}
    for nm in NEW_NAMES:
        ek[nm], em[nm], sm[nm] = R.elasticities(cellsP, nm, [10, 100])
        sm_all[nm] = R.elasticities(cellsP, nm, [1, 10, 100])[2]
        sm_110[nm] = R.elasticities(cellsP, nm, [1, 10])[2]
    p1 = R.p1_label([em[n] for n in NEW_NAMES], [sm[n] for n in NEW_NAMES])
    p1.update(e_k=ek, e_m=em, sum=sm, sum_allk=sm_all, sum_k1_10=sm_110,
              max_abs_sum=float(max(abs(v) for v in sm.values())))
    per_enc = {}
    for e in C.ENCS:
        sub = [f"{v}__{e}" for v in sorted(C.VITS)]
        inf = [n for n in sub if abs(em[n]) >= 0.10]
        per_enc[e] = dict(median_sum_all5=float(np.median([sm[n] for n in sub])),
                          median_sum_informative=(float(np.median([sm[n] for n in inf])) if inf else None),
                          n_informative=len(inf))
    p1["per_encoder"] = per_enc
    # corr(e_k, -e_m) over the 30 VL pairs of arm P (15 old S1.1 cross-modal + 15 new)
    cells_old = R.cells_from_obs(old_vl_rows)
    old_names = sorted({p for (p, d, m, k) in old_vl_rows})
    ek30 = [R.elasticities(cells_old, n, [10, 100])[0] for n in old_names] + [ek[n] for n in NEW_NAMES]
    em30 = [R.elasticities(cells_old, n, [10, 100])[1] for n in old_names] + [em[n] for n in NEW_NAMES]
    p1["corr_ek_minus_em_30VL"] = float(np.corrcoef(ek30, -np.asarray(em30))[0, 1])
    p1["corr_ek_minus_em_15new"] = float(np.corrcoef([ek[n] for n in NEW_NAMES], [-em[n] for n in NEW_NAMES])[0, 1])
    # per-draw P1 (robustness, plan 9)
    pdraw = {}
    for d in range(C.N_DRAWS):
        cd = per_draw_cells(P, d)
        e_ = {n: R.elasticities(cd, n, [10, 100]) for n in NEW_NAMES}
        lab = R.p1_label([e_[n][1] for n in NEW_NAMES], [e_[n][2] for n in NEW_NAMES])
        lab["sum"] = {n: e_[n][2] for n in NEW_NAMES}
        pdraw[d] = lab
    p1["per_draw"] = pdraw
    out["P1"] = p1
    # ---------------- P2b (arm P, k = 10, m_ref = 1024) ----------------
    L, lnL, b, bdraws = {}, {}, {}, {}
    for nm in NEW_NAMES:
        L[nm], lnL[nm] = R.level(cellsP, nm, 10, 1024)
        b[nm], bdraws[nm] = R.mean_slope(P, nm, 10)
    p2b = R.p2b(NEW_PAIRS, lnL, b)
    p2b.update(L=L, lnL=lnL, b=b, b_draws=bdraws, median_L=float(np.median(list(L.values()))),
               median_b=float(np.median(list(b.values()))))
    out["P2b"] = p2b
    # secondary (plan 7.5): arm A (= D5) with the arm-P line
    cellsA = R.cells_from_obs(arm_rows["D5"])
    lnLA = {nm: R.level(cellsA, nm, 10, 1024)[1] for nm in NEW_NAMES}
    bA = {nm: R.mean_slope(arm_rows["D5"], nm, 10)[0] for nm in NEW_NAMES}
    sA = R.p2b(NEW_PAIRS, lnLA, bA)
    out["P2b_secondary_armA"] = {k: sA[k] for k in ("label", "median_abs_g", "lambda_hat", "lambda_e",
                                                    "n_informative_encoders")}
    # ---------------- P2c (arms D1..D5) ----------------
    p2c_by_k = {}
    for k in (10, 1, 100):
        lnLn = {nm: [] for nm in NEW_NAMES}
        Ln = {nm: [] for nm in NEW_NAMES}
        bn = {nm: [] for nm in NEW_NAMES}
        for n in range(1, 6):
            rows = arm_rows[f"D{n}"]
            cells = R.cells_from_obs(rows)
            for nm in NEW_NAMES:
                s, l = R.level(cells, nm, k, 1024)
                Ln[nm].append(s)
                lnLn[nm].append(l)
                bn[nm].append(R.mean_slope(rows, nm, k)[0])
        res = R.p2c(NEW_PAIRS, lnLn, bn)
        res.update(L_ladder=Ln, b_ladder=bn)
        p2c_by_k[k] = res
    out["P2c"] = p2c_by_k[10]
    out["P2c_gamma_k1"] = {"gamma": p2c_by_k[1]["gamma"], "median_gamma": p2c_by_k[1]["median_gamma"]}
    out["P2c_gamma_k100"] = {"gamma": p2c_by_k[100]["gamma"], "median_gamma": p2c_by_k[100]["median_gamma"]}
    return out


def load_e0_arm_rows():
    rows, files = {}, {}
    for arm, f in C.E0_OBS_FILE.items():
        p = os.path.join(E0_OBS, f + ".parquet")
        files[arm] = dict(path=p, sha256=C.sha256_file(p))
        rows[arm] = rows_from_df(pd.read_parquet(p), NEW_NAMES)
    return rows, files


def load_vf_arm_rows(vf_path):
    df = pd.read_parquet(vf_path)
    return {arm: rows_from_df(df[df["arm"] == arm], NEW_NAMES) for arm in C.ARMS}


def load_old_vl():
    o = pd.read_parquet(S11_OBS)
    vl = sorted(o[(o["cls"] == "cross-modal") & (o["primary"])]["pair"].unique())
    assert len(vl) == 15
    return rows_from_df(o, vl), C.sha256_file(S11_OBS)


def main(outdir, vf_path):
    e0_rows, e0_files = load_e0_arm_rows()
    old, old_sha = load_old_vl()
    st_e0 = all_stats(e0_rows, old)
    st_vf = all_stats(load_vf_arm_rows(vf_path), old)
    res = dict(inputs=dict(e0_obs=e0_files, s11_obs_clip=dict(path=S11_OBS, sha256=old_sha),
                           vf_obs=dict(path=vf_path, sha256=C.sha256_file(vf_path))),
               from_e0_obs=st_e0, from_vf_obs=st_vf)
    sha = C.write_new_json(os.path.join(outdir, "vf_stats.json"), res)
    for key in ("P1", "P2b", "P2c"):
        print(key, st_e0[key]["label"], "| from own obs:", st_vf[key]["label"])
    print("vf_stats.json", sha)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
