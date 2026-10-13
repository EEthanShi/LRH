"""Independent recompute of the S2.1 secondary items from run features (role s21-recompute):
sharp-link beta-hat (PREREG 2.5 Secondary) and Lemma 6 co-occurrence ratios.
Reads runs/S2_1_main/*/feats.npz (read-only); compares with results/S2_1/s21_summary.json.
No import of analyze.py / crit.py. Writes only under results/review_final/s21-recompute/.
"""
from __future__ import annotations

import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
OUT = os.path.join(ROOT, "results", "review_final", "s21-recompute")
NS = (50000, 500000)
M = 2048
IU = None


def rid(arm, N, r):
    return f"S2.1_{arm}_N{N}_r{r:03d}"


def load_z(run_id):
    d = np.load(os.path.join(ROOT, "runs", "S2_1_main", run_id, "feats.npz"))
    X = np.asarray(d["z"], dtype=np.float64)
    assert X.shape == (M, 128) and np.all(np.isfinite(X))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def upper(K):
    global IU
    if IU is None:
        IU = np.triu_indices(M, 1)
    return K[IU]


# ------------------------------------------------------------------ beta-hat
LO, HI = math.log(0.01), math.log(100.0)


def beta_fit(task):
    a, b = task
    XA, XB = load_z(a), load_z(b)
    ka = upper(XA @ XA.T) - 1.0
    kb = upper(XB @ XB.T)

    def f(lb):
        r = kb - np.exp(math.exp(lb) * ka)
        return float(np.dot(r, r))

    grid = np.linspace(LO, HI, 81)
    fv = np.array([f(x) for x in grid])
    i = int(np.argmin(fv))
    n_locmin = int(sum(1 for j in range(1, len(fv) - 1) if fv[j] < fv[j - 1] and fv[j] < fv[j + 1]))
    lo_b, hi_b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(f, bounds=(lo_b, hi_b), method="bounded", options={"xatol": 1e-10})
    lb = float(res.x) if res.fun <= fv[i] else float(grid[i])
    return {"A": a, "B": b, "beta": math.exp(lb), "loss": min(res.fun, fv[i]), "grid_argmin_idx": i,
            "n_local_minima_on_grid": n_locmin, "at_bound": i in (0, len(grid) - 1)}


# ------------------------------------------------------------------ Lemma 6
def lemma6_pair(task):
    a, b = task
    XA, XB = load_z(a), load_z(b)
    out = {"A": a, "B": b}
    kA, kB = upper(XA @ XA.T), upper(XB @ XB.T)
    for variant in ("dist", "sqdist"):
        dA = np.maximum(2.0 - 2.0 * kA, 0.0)
        dB = np.maximum(2.0 - 2.0 * kB, 0.0)
        if variant == "dist":
            dA, dB = np.sqrt(dA), np.sqrt(dB)
        tA = dA / np.quantile(dA, 0.9)
        tB = dB / np.quantile(dB, 0.9)
        mbar = 0.5 * (tA + tB)
        n = mbar.size
        k_near = int(math.ceil(0.01 * n))
        near = np.argpartition(mbar, k_near - 1)[:k_near]
        k_far = n // 2
        far = np.argpartition(mbar, n - k_far)[n - k_far:]
        with np.errstate(divide="ignore", invalid="ignore"):
            la = np.log(tA[near] / tB[near])
            lf = np.log(tA[far] / tB[far])
        out[f"{variant}_n_nonfinite_near"] = int(np.sum(~np.isfinite(la)))
        la = la[np.isfinite(la)]
        out[f"{variant}_e_near"] = float(np.sqrt(np.mean(la ** 2)))
        out[f"{variant}_e_far"] = float(np.sqrt(np.mean(lf ** 2)))
        out[f"{variant}_n_zero_dist"] = int(np.sum(dA == 0) + np.sum(dB == 0))
    return out


def main():
    t0 = time.time()
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    SUM = json.load(open(os.path.join(ROOT, "results", "S2_1", "s21_summary.json")))
    beta_tasks, lem_tasks = [], []
    for N in NS:
        for i in range(64):
            beta_tasks.append((rid("A", N, i), rid("B", N, i)))
            beta_tasks.append((rid("Atau", N, i), rid("B", N, i)))
        for arm in ("A", "B", "Atau"):
            for j in range(32):
                lem_tasks.append((rid(arm, N, 2 * j), rid(arm, N, 2 * j + 1)))
    with Pool(nproc) as pool:
        lem = pool.map(lemma6_pair, lem_tasks, chunksize=4)
        print(f"lemma6 done {time.time() - t0:.0f}s", flush=True)
        bet = pool.map(beta_fit, beta_tasks, chunksize=4)
        print(f"beta done {time.time() - t0:.0f}s", flush=True)

    res = {"beta": {}, "lemma6": {}, "compare": {}}
    # beta summaries
    bmap = {(r["A"], r["B"]): r for r in bet}
    for N in NS:
        for src in ("A", "Atau"):
            v = np.array([bmap[(rid(src, N, i), rid("B", N, i))]["beta"] for i in range(64)])
            key = f"{src}_B_{N}"
            res["beta"][key] = {"median": float(np.median(v)), "q25": float(np.quantile(v, .25)),
                                "q75": float(np.quantile(v, .75)), "min": float(v.min()), "max": float(v.max()),
                                "values": v.tolist(),
                                "any_at_bound": any(bmap[(rid(src, N, i), rid("B", N, i))]["at_bound"] for i in range(64)),
                                "max_local_minima_on_grid": max(bmap[(rid(src, N, i), rid("B", N, i))]["n_local_minima_on_grid"] for i in range(64))}
            th = np.array(SUM["beta"][key]["values"], dtype=float)
            res["compare"][f"beta/{key}"] = {"max_abs_diff_values": float(np.max(np.abs(v - th))),
                                             "median_mine": float(np.median(v)), "median_theirs": SUM["beta"][key]["median"]}
        ratio = res["beta"][f"A_B_{N}"]["median"] / res["beta"][f"Atau_B_{N}"]["median"]
        res["beta"][f"ratio_{N}"] = ratio
        mAB = res["beta"][f"A_B_{N}"]["median"]
        res["beta"][f"pred_ok_{N}"] = bool(2.5 <= mAB <= 10 and 1.5 <= ratio <= 3.5)
        res["compare"][f"beta/ratio_{N}"] = {"mine": ratio, "theirs": SUM["beta"][f"ratio_{N}"]}
        res["compare"][f"beta/pred_ok_{N}"] = {"mine": res["beta"][f"pred_ok_{N}"], "theirs": SUM["beta"][f"pred_ok_{N}"]}

    # Lemma 6
    lmap = {(r["A"], r["B"]): r for r in lem}
    rng = np.random.default_rng(20261009 + 6)
    Bt = 2000
    for variant in ("dist", "sqdist"):
        L = {}
        boots = {}
        for arm in ("A", "B", "Atau"):
            E = {}
            for N in NS:
                rows = [lmap[(rid(arm, N, 2 * j), rid(arm, N, 2 * j + 1))] for j in range(32)]
                E[N] = (np.array([r[f"{variant}_e_near"] for r in rows]), np.array([r[f"{variant}_e_far"] for r in rows]))
            Rn = E[50000][0].mean() / E[500000][0].mean()
            Rf = E[50000][1].mean() / E[500000][1].mean()
            i0 = rng.integers(0, 32, (Bt, 32)); i5 = rng.integers(0, 32, (Bt, 32))
            bn = E[50000][0][i0].mean(1) / E[500000][0][i5].mean(1)
            bf = E[50000][1][i0].mean(1) / E[500000][1][i5].mean(1)
            bl = np.log(bn / bf)
            boots[arm] = bl
            L[arm] = {"e_near_50k": float(E[50000][0].mean()), "e_near_500k": float(E[500000][0].mean()),
                      "e_far_50k": float(E[50000][1].mean()), "e_far_500k": float(E[500000][1].mean()),
                      "R_near": float(Rn), "R_far": float(Rf), "log_Rnear_over_Rfar": float(math.log(Rn / Rf)),
                      "ci95": [float(np.quantile(bl, .025)), float(np.quantile(bl, .975))]}
        dd = L["A"]["log_Rnear_over_Rfar"] - L["Atau"]["log_Rnear_over_Rfar"]
        bd = boots["A"] - boots["Atau"]
        L["A_minus_Atau"] = {"diff": dd, "ci95": [float(np.quantile(bd, .025)), float(np.quantile(bd, .975))]}
        L["prediction_ok"] = bool(L["A"]["R_near"] > L["A"]["R_far"] and L["Atau"]["R_near"] > L["Atau"]["R_far"] and dd > 0)
        res["lemma6"][variant] = L
    SL = SUM["lemma6"]
    for variant in ("dist", "sqdist"):
        c = {}
        for arm in ("A", "B", "Atau"):
            for f in ("e_near_50k", "e_near_500k", "e_far_50k", "e_far_500k", "R_near", "R_far", "log_Rnear_over_Rfar"):
                c[f"{arm}/{f}"] = {"mine": res["lemma6"][variant][arm][f], "theirs": SL[arm][f],
                                   "abs_diff": abs(res["lemma6"][variant][arm][f] - SL[arm][f])}
        c["A_minus_Atau/diff"] = {"mine": res["lemma6"][variant]["A_minus_Atau"]["diff"], "theirs": SL["A_minus_Atau"]["diff"]}
        c["prediction_ok"] = {"mine": res["lemma6"][variant]["prediction_ok"], "theirs": SL["prediction_ok"]}
        c["max_abs_diff"] = max(v["abs_diff"] for k, v in c.items() if isinstance(v, dict) and "abs_diff" in v)
        res["compare"][f"lemma6/{variant}"] = c
    res["lemma6_pairs"] = lem
    res["beta_pairs"] = bet
    res["elapsed_s"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, "secondary_recompute.json"), "w"), indent=1)
    print(json.dumps(res["compare"], indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "values"} if isinstance(v, dict) else v
                      for k, v in res["beta"].items()}, indent=1))
    print(json.dumps(res["lemma6"], indent=1))


if __name__ == "__main__":
    main()
