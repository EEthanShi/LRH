"""Spot checks for the S2.1 recompute (role s21-recompute).

(a) eps_S2_1 (z, pre=l2): recompute the 24 pooled within-arm pilot ladders from the pilot features
    with the frozen hrhv2 harness, derive Delta_Q per prereg 2.4 (own mapping), take the max, compare
    with prereg/eps_S2_1.json.
(b) ladder cache integrity: recompute 8 main-cohort ladders from features with the harness and compare
    every numeric field of the cached row.
(c) from-scratch (no hrhv2): s_N(10), Gs_rms, CKA(biased), Pearson RSA, Delta_iso and omega (own hash split,
    own cross-fitted isotonic) on the same 8 pairs.
No import of analyze.py / crit.py. Writes only under results/review_final/s21-recompute/.
"""
from __future__ import annotations

import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import hashlib
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import isotonic_regression

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
OUT = os.path.join(ROOT, "results", "review_final", "s21-recompute")
sys.path.insert(0, os.path.expanduser("~/hrh_v2_stage1_20261008"))
sys.path.insert(0, OUT)
from hrhv2.prep import Geom  # noqa: E402  (frozen metric harness, read-only)
from hrhv2.ladder import ladder  # noqa: E402
from s21_recompute import my_deltas, RUNG_KEYS  # noqa: E402  (own Delta mapping)

S_IDX = np.load(os.path.join(ROOT, "data", "S_idx.npy"))
ITEM_IDS = [f"part5:{i}" for i in S_IDX]
M = len(ITEM_IDS)
NJ = 2


def feats(cohort, run_id, key="z"):
    return np.asarray(np.load(os.path.join(ROOT, "runs", cohort, run_id, "feats.npz"))[key], dtype=np.float64)


def l2(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def bern_gram(X, beta=5.0):
    Xn = l2(X)
    K = np.exp(beta * (Xn @ Xn.T - 1.0))
    np.fill_diagonal(K, 1.0)
    return K


# ------------------------------------------------------------------ from-scratch metrics
def my_hash_half(ids, salt="hrhv2-halves"):
    return np.array([(int.from_bytes(hashlib.sha256(f"{salt}|{i}".encode()).digest()[:8], "little") & 1) == 0
                     for i in ids], dtype=bool)


def sqdist_from_gram(K):
    d = np.diag(K)
    D = d[:, None] + d[None, :] - 2.0 * K
    np.maximum(D, 0.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def knn_sets(D, k):
    Dc = D.copy()
    np.fill_diagonal(Dc, np.inf)
    return np.argsort(Dc, axis=1, kind="stable")[:, :k]


def s_N(DA, DB, k=10):
    a, b = knn_sets(DA, k), knn_sets(DB, k)
    return float(np.mean([len(set(a[i]) & set(b[i])) for i in range(DA.shape[0])]) / k)


def gs_rms(DA, DB):
    m = DA.shape[0]
    nA, nB = DA.sum() / (m * (m - 1)), DB.sum() / (m * (m - 1))
    R = DA / nA - DB / nB
    return float(np.sqrt((R ** 2).sum() / (m * (m - 1)))), nA, nB


def cka_b(KA, KB):
    H = lambda K: K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()
    A, B = H(KA), H(KB)
    return float((A * B).sum() / math.sqrt((A * A).sum() * (B * B).sum()))


def iso_cf(x1, y1, x2, y2):
    o = np.argsort(x1, kind="stable")
    xs, ys = x1[o], y1[o]
    g = isotonic_regression(ys, increasing=True).x
    # collapse exact ties in x by averaging fitted values (rare for float distances)
    pred = np.interp(x2, xs, g)
    r = y2 - pred
    return float(np.sqrt(np.mean(r ** 2))), float(np.dot(r, r))


def aff_cf(x1, y1, x2, y2):
    a, b = np.polyfit(x1, y1, 1)
    r = y2 - (a * x2 + b)
    return float(np.sqrt(np.mean(r ** 2))), float(np.dot(r, r))


def scratch(KA, KB, h1):
    DA, DB = sqdist_from_gram(KA), sqdist_from_gram(KB)
    out = {"s_N_k10": s_N(DA, DB, 10)}
    out["Gs_rms"], nA, nB = gs_rms(DA, DB)
    out["cka_biased"] = cka_b(KA, KB)
    iu = np.triu_indices(DA.shape[0], 1)
    out["rsa_pearson"] = float(np.corrcoef(DA[iu], DB[iu])[0, 1])
    i1, i2 = np.flatnonzero(h1), np.flatnonzero(~h1)
    u1 = np.triu_indices(len(i1), 1)
    u2 = np.triu_indices(len(i2), 1)
    A1, B1 = DA[np.ix_(i1, i1)][u1], DB[np.ix_(i1, i1)][u1]
    A2, B2 = DA[np.ix_(i2, i2)][u2], DB[np.ix_(i2, i2)][u2]
    isoAB, ssIAB = iso_cf(A1, B1, A2, B2)
    isoBA, ssIBA = iso_cf(B1, A1, B2, A2)
    affAB, ssAAB = aff_cf(A1, B1, A2, B2)
    affBA, ssABA = aff_cf(B1, A1, B2, A2)
    out["dIso"] = 0.5 * (isoAB / nB + isoBA / nA)
    out["omega"] = 0.5 * ((1 - ssIAB / ssAAB) + (1 - ssIBA / ssABA))
    out["n_H1"] = int(h1.sum())
    return out


def main():
    t0 = time.time()
    res = {"pilot_eps": {}, "ladder_cache": [], "scratch": []}
    eps_file = json.load(open(os.path.join(ROOT, "prereg", "eps_S2_1.json")))
    h1 = my_hash_half(ITEM_IDS)

    # (a) pilot eps for z (pre=l2)
    plan = json.load(open(os.path.join(ROOT, "plans", "S2_1_pilot.json")))
    pilot_rows = {(r["A"], r["B"]): r for r in eps_file["pilot_rows"] if r["rep"] == "z"}
    deltas = []
    for arm in ("A", "B", "Atau"):
        reps = sorted(r["rep"] for r in plan["runs"] if r["arm"] == arm)
        assert len(reps) == 16, (arm, len(reps))
        for j in range(8):
            a = f"S2.1-pilot_{arm}_N150000_r{reps[2 * j]:03d}"
            b = f"S2.1-pilot_{arm}_N150000_r{reps[2 * j + 1]:03d}"
            ma = json.load(open(os.path.join(ROOT, "runs", "S2_1_pilot", a, "meta.json")))
            mb = json.load(open(os.path.join(ROOT, "runs", "S2_1_pilot", b, "meta.json")))
            assert not ma["diverged"] and not mb["diverged"]
            GA = Geom.from_features(feats("S2_1_pilot", a), "l2")
            GB = Geom.from_features(feats("S2_1_pilot", b), "l2")
            row = ladder(GA, GB, item_ids=ITEM_IDS, n_jobs=NJ)
            d = my_deltas(row)
            d["pair"] = [a, b]
            pr = pilot_rows.get((a, b))
            d["max_abs_diff_vs_eps_pilot_row"] = max(abs(d[k] - pr[k]) for k in RUNG_KEYS
                                                     if not (math.isnan(d[k]) and math.isnan(pr[k]))) if pr else None
            deltas.append(d)
            print(f"pilot {a} {b} {time.time() - t0:.0f}s", flush=True)
    cmpd = {}
    for k in RUNG_KEYS:
        v = np.array([d[k] for d in deltas])
        mx = float(np.nanmax(v))
        cmpd[k] = {"mine_max": mx, "eps_file": eps_file["eps"]["z"][k], "abs_diff": abs(mx - eps_file["eps"]["z"][k]),
                   "mine_q90": float(np.nanquantile(v, 0.9)), "eps_q90_file": eps_file["eps_q90"]["z"][k]}
    res["pilot_eps"] = {"n_pairs": len(deltas), "by_rung": cmpd,
                        "max_abs_diff_eps": max(c["abs_diff"] for c in cmpd.values() if not math.isnan(c["abs_diff"])),
                        "max_abs_diff_q90": max(abs(c["mine_q90"] - c["eps_q90_file"]) for c in cmpd.values()
                                                if not math.isnan(c["mine_q90"])),
                        "max_abs_diff_pilot_rows": max(d["max_abs_diff_vs_eps_pilot_row"] for d in deltas)}

    # (b)+(c) main ladder cache + from-scratch
    picks = [
        ("within_A_500000", "S2.1_A_N500000_r010", "S2.1_A_N500000_r011", False),
        ("within_B_500000", "S2.1_B_N500000_r034", "S2.1_B_N500000_r035", False),
        ("cross_A_B_500000", "S2.1_A_N500000_r040", "S2.1_B_N500000_r040", False),
        ("cross_A_B_50000", "S2.1_A_N50000_r003", "S2.1_B_N50000_r003", False),
        ("cross_A_Atau_500000", "S2.1_A_N500000_r011", "S2.1_Atau_N500000_r011", False),
        ("bern_self_500000", "S2.1_A_N500000_r007", "S2.1_A_N500000_r007", True),
        ("bern_peer_500000", "S2.1_A_N500000_r018", "S2.1_A_N500000_r019", True),
        ("within_B_50000", "S2.1_B_N50000_r000", "S2.1_B_N50000_r001", False),
    ]
    for sname, a, b, bern in picks:
        XA, XB = feats("S2_1_main", a), feats("S2_1_main", b)
        GA = Geom.from_features(XA, "l2")
        GB = Geom.from_gram(bern_gram(XB), pre="bern") if bern else Geom.from_features(XB, "l2")
        row = ladder(GA, GB, item_ids=ITEM_IDS, n_jobs=NJ)
        cached = json.load(open(os.path.join(ROOT, "results", "ladders", "S2_1_main", "z",
                                             f"{a}__{b}{'__bern' if bern else ''}.json")))["row"]
        worst, wk, nnum = 0.0, None, 0
        for k, v in cached.items():
            if k.startswith("time_") or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            w = row.get(k)
            if w is None:
                continue
            if isinstance(v, float) and math.isnan(v) and math.isnan(float(w)):
                continue
            nnum += 1
            rel = abs(float(w) - float(v)) / max(1.0, abs(float(v)))
            if rel > worst:
                worst, wk = rel, (k, float(w), float(v))
        res["ladder_cache"].append({"set": sname, "A": a, "B": b, "n_fields": nnum, "max_rel_diff": worst, "worst": wk})
        KA = l2(XA) @ l2(XA).T
        KB = bern_gram(XB) if bern else l2(XB) @ l2(XB).T
        sc = scratch(KA, KB, h1)
        cd = my_deltas(cached)
        comp = {"s_N_k10": (sc["s_N_k10"], cached["s_N_k10"]), "Gs_rms": (sc["Gs_rms"], cached["Gs_rms"]),
                "cka_biased": (sc["cka_biased"], cached["cka_biased"]), "rsa_pearson": (sc["rsa_pearson"], cached["rsa_pearson"]),
                "dIso": (sc["dIso"], cd["dIso"]), "omega": (sc["omega"], cd["omega"]), "n_H1": (sc["n_H1"], cached["n_H1"])}
        res["scratch"].append({"set": sname, "A": a, "B": b,
                               "values": {k: {"scratch": x, "cached": y, "abs_diff": abs(x - y)} for k, (x, y) in comp.items()}})
        print(f"main {sname} {time.time() - t0:.0f}s", flush=True)
    res["elapsed_s"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, "spotcheck.json"), "w"), indent=1)
    print(json.dumps({"pilot_eps": {k: v for k, v in res["pilot_eps"].items() if k != "by_rung"},
                      "pilot_eps_ord": {k: res["pilot_eps"]["by_rung"][k] for k in ("dN_k10", "dO", "dE", "dMST", "Gs", "dAff", "dSOG_t0p1", "dCKA", "rawG", "dIso")},
                      "ladder_cache": res["ladder_cache"], "scratch": res["scratch"]}, indent=1))


if __name__ == "__main__":
    main()
