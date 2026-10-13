#!/usr/bin/env python
"""Verifier: independent check of the G0 P-Gauss k=1 failure attribution.

Regenerates G0 P-Gauss cells with the formal G0 seeds (G0f), and computes, with
code independent of hrhv2.predict:
  obs     observed 1 - s_N(k) on H2 anchors (own lexsort (D, j) kNN)
  mulaw   SPEC 6 balance-threshold mu with the TRUE coupling b = u + sigma Z
          (own scipy brentq per anchor)
  exact   E|N_A \\ N_B|/k under the same independent-column law, by Monte Carlo
          (R draws per anchor)
Claim under test (harness-main): at k=1 the mu-law itself misses obs by ~0.045-0.057
while the exact expectation is within ~0.014.
Usage: python pgauss_check.py OUT.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/hrh_v2_stage1_20261008"))
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from hrhv2 import plants, util  # noqa: E402

R_MC = 400


def knn_sets(D, anchors, k):
    m = D.shape[0]
    out = []
    for i in anchors:
        row = D[i].copy()
        row[i] = np.inf
        o = np.lexsort((np.arange(m), row))
        out.append(set(o[:k].tolist()))
    return out


def mulaw_row(u, s, k):
    uA, uN = u[:k], u[k:]
    g = lambda t: (1 - ndtr((t - uA) / s)).sum() - ndtr((t - uN) / s).sum()
    t = brentq(g, u.min() - 40 * s, u.max() + 40 * s, xtol=1e-15, rtol=1e-14, maxiter=500)
    return (1 - ndtr((t - uA) / s)).sum()


def cell(p, c, m, rep, ks=(1, 3), rng_key="VERIF1|pgauss_mc"):
    ref = plants.feature_dists(plants.base_cloud(1024, p, f"G0f|P-Gauss|p={p}|sigma_ref"))
    sigma = c * plants.knn_d_median(ref, 10)
    DA, DB = plants.gauss_plant(4096, p, sigma, f"G0f|P-Gauss|p={p}|c={c}|rep={rep}")
    DA, DB = DA[:m, :m], DB[:m, :m]
    h1 = util.hash_half(plants.nested_ids(4096), "hrhv2-halves")[:m]
    anchors = np.flatnonzero(~h1)
    rng = util.rng_from_key(f"{rng_key}|{p}|{c}|{m}|{rep}")
    res = {}
    for k in ks:
        NA = knn_sets(DA, anchors, k)
        NB = knn_sets(DB, anchors, k)
        obs = np.mean([len(a - b) for a, b in zip(NA, NB)]) / k
        mus, ex = [], []
        for i in anchors:
            row = np.delete(DA[i], i)
            u = np.sort(row)
            mus.append(mulaw_row(u, sigma, k))
            # Monte Carlo of E|N_A \ N_B| with independent Gaussian columns
            Z = rng.standard_normal((R_MC, u.size))
            b = u[None, :] + sigma * Z
            kth = np.partition(b, k - 1, axis=1)[:, k - 1]
            # true A-neighbours are columns 0..k-1 of the sorted row (no exact ties here a.s.)
            lost = np.sum(b[:, :k] > kth[:, None], axis=1)
            ex.append(lost.mean())
        res[k] = {"obs_1ms": float(obs), "mulaw_1ms": float(np.mean(mus) / k), "exact_mc_1ms": float(np.mean(ex) / k),
                  "mulaw_minus_obs": float(np.mean(mus) / k - obs), "exact_minus_obs": float(np.mean(ex) / k - obs),
                  "mulaw_minus_exact": float(np.mean(mus) / k - np.mean(ex) / k),
                  "n_anchors": int(anchors.size), "sigma": float(sigma)}
    return res


def main():
    out = []
    for p, c, m in ((2, 0.05, 1024), (2, 0.4, 256), (4, 0.15, 256), (4, 0.05, 1024)):
        for rep in range(4):
            r = cell(p, c, m, rep)
            out.append({"p": p, "c": c, "m": m, "rep": rep, "res": r})
            print(json.dumps(out[-1]), flush=True)
    # aggregate over reps (G0 averages obs and predictions over reps)
    agg = {}
    for o in out:
        key = f"p={o['p']}|c={o['c']}|m={o['m']}"
        for k, v in o["res"].items():
            a = agg.setdefault(f"{key}|k={k}", {"obs": [], "mulaw": [], "exact": []})
            a["obs"].append(v["obs_1ms"])
            a["mulaw"].append(v["mulaw_1ms"])
            a["exact"].append(v["exact_mc_1ms"])
    summ = {}
    for key, a in agg.items():
        summ[key] = {"s_obs": 1 - float(np.mean(a["obs"])), "s_mulaw": 1 - float(np.mean(a["mulaw"])),
                     "s_exact": 1 - float(np.mean(a["exact"])),
                     "mulaw_err": float(np.mean(a["obs"]) - np.mean(a["mulaw"])),
                     "exact_err": float(np.mean(a["obs"]) - np.mean(a["exact"])),
                     "mulaw_vs_exact": float(np.mean(a["exact"]) - np.mean(a["mulaw"]))}
        print(key, summ[key], flush=True)
    json.dump({"cells": out, "summary": summ, "R_MC": R_MC}, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
