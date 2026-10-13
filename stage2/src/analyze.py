"""Stage-2 analysis implementing PREREG_stage2.md v1.0 decisions.

    python analyze.py s22 [--jobs 3]           # S2.2 gauge net
    python analyze.py s21 [--jobs 3]           # S2.1 objective link
    python analyze.py status                   # progress of every cohort

Uses the frozen tolerances prereg/eps_S2_{1,2}.json (written by gates.py from pilot pairs only).
If main runs are incomplete, the analysis uses the balanced rep prefix R (largest even R with every
cell complete) and labels the output INTERIM. Pair ladders are cached in results/ladders/ (resumable).
Outputs: results/S2_1/{s21_summary.json, S2_1_report.md}, results/S2_2/{s22_summary.json, S2_2_report.md}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import get_context

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crit as C  # noqa: E402

ROOT = C.ROOT
NBOOT = 2000


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda v: float(v) if isinstance(v, (np.floating, np.integer)) else str(v))
    os.replace(tmp, path)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ============================================================================ run bookkeeping
def plan_runs(cohort):
    return json.load(open(os.path.join(ROOT, "plans", f"{cohort}.json")))["runs"]


def is_done(cohort, rid):
    return os.path.exists(os.path.join(C.run_dir(cohort, rid), ".done"))


def cell_reps(cohort, exp, arm, N, corpus=None):
    """{rep: run_id} of done runs in a cell."""
    out = {}
    for r in plan_runs(cohort):
        if r["exp"] == exp and r["arm"] == arm and r["N"] == N and r.get("corpus") == corpus and is_done(cohort, r["run_id"]):
            out[r["rep"]] = r["run_id"]
    return out


def balanced_R(cells: list, target: int):
    """Largest even R <= target such that every cell has reps 0..R-1 done."""
    R = target
    for reps in cells:
        r = 0
        while r in reps:
            r += 1
        R = min(R, r)
    return R - (R % 2)


# ============================================================================ ladder jobs (cached)
def ladder_path(cohort, tag, ra, rb, transform):
    name = f"{ra}__{rb}" + (f"__{transform}" if transform else "")
    return os.path.join(ROOT, "results", "ladders", cohort, tag, name + ".json")


def _job(args):
    cohort, tag, key, pre, ra, rb, transform, cohort_b = args
    path = ladder_path(cohort, tag, ra, rb, transform)
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    A = C.load_run(cohort, ra)
    B = C.load_run(cohort_b or cohort, rb)
    if not (C.run_ok(A, key) and C.run_ok(B, key)):
        res = {"deltas": {k: C.FAIL for k in C.TOL_KEYS}, "row": {}, "failed_run": True}
    else:
        GA = C.geom(A["feats"][key], pre)
        GB = GA if (rb == ra and cohort_b is None) else C.geom(B["feats"][key], pre)
        if transform == "bern":
            GB = C.bern_geom(GB)
        row = C.pair_ladder(GA, GB, n_jobs=1)
        res = {"deltas": C.deltas(row), "row": row, "failed_run": False}
    res.update(A=ra, B=rb, tag=tag, transform=transform)
    jdump(res, path)
    return res


def run_jobs(jobs, n_jobs):
    todo = [j for j in jobs if not os.path.exists(ladder_path(j[0], j[1], j[4], j[5], j[6]))]
    print(f"  ladders: {len(jobs)} total, {len(todo)} to compute (jobs={n_jobs})", flush=True)
    t = time.time()
    if todo:
        with get_context("fork").Pool(n_jobs) as pool:
            for i, _ in enumerate(pool.imap_unordered(_job, todo, chunksize=1)):
                if (i + 1) % 50 == 0:
                    print(f"    {i + 1}/{len(todo)} ({time.time() - t:.0f} s)", flush=True)
    return {(j[1], j[4], j[5], j[6]): _job(j) for j in jobs}


# ============================================================================ statistics
def pset_decisions(D: list, eps: dict, keys):
    """D: list of delta dicts for one pair set. Returns per key: n, x, p_hat, CP, decision, values."""
    out = {}
    for k in keys:
        e = eps.get(k, float("nan"))
        vals = np.array([d.get(k, float("nan")) for d in D], dtype=float)
        if np.all(np.isnan(vals)) or not np.isfinite(e):
            continue
        vals = np.where(np.isnan(vals), np.inf, vals)
        n = len(vals)
        x = int(np.sum(vals <= e))
        dec, lo, hi = C.decide(x, n)
        fin = vals[np.isfinite(vals)]
        out[k] = {"n": n, "x": x, "p_hat": x / n if n else float("nan"), "cp_lo": lo, "cp_hi": hi, "decision": dec,
                  "eps": e, "mean": float(fin.mean()) if fin.size else float("nan"),
                  "median": float(np.median(fin)) if fin.size else float("nan"), "values": vals.tolist()}
    return out


def p_hat_at(D, eps, k):
    vals = np.array([d.get(k, float("nan")) for d in D], dtype=float)
    vals = np.where(np.isnan(vals), np.inf, vals)
    return float(np.mean(vals <= eps)) if len(vals) else float("nan")


def boot_mean_diff(a, b, key, n=NBOOT):
    """Bootstrap (pairs resampled within each set) of mean(a) - mean(b); a, b lists of delta dicts."""
    va = np.array([d[key] for d in a], float)
    vb = np.array([d[key] for d in b], float)
    va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"boot|{key}".encode()).digest()[:8], "little"))
    ia = rng.integers(0, va.size, (n, va.size))
    ib = rng.integers(0, vb.size, (n, vb.size))
    d = va[ia].mean(1) - vb[ib].mean(1)
    return {"diff": float(va.mean() - vb.mean()), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "p_one_sided_diff_ge_0": float((np.sum(d >= 0) + 1) / (n + 1))}


def holm(pvals: dict, alpha=0.05):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, stop = {}, False
    for i, (k, p) in enumerate(items):
        thr = alpha / (m - i)
        rej = (not stop) and p <= thr
        if not rej:
            stop = True
        out[k] = {"p": p, "holm_threshold": thr, "reject": rej}
    return out


# ============================================================================ S2.1
def s21(n_jobs, dry_pilot=False, eps_file=None):
    cohort, exp, Ns, target, outdir = "S2_1_main", "S2.1", (50000, 500000), 64, "S2_1"
    if dry_pilot:  # pipeline test on EXCLUDED pilot runs (one N); not evidence
        cohort, exp, Ns, target, outdir = "S2_1_pilot", "S2.1-pilot", (150000,), 16, "S2_1_DRYRUN_on_pilot"
    eps_path = eps_file or os.path.join(ROOT, "prereg", "eps_S2_1.json")
    eps_all = json.load(open(eps_path))
    eps = eps_all["eps"]
    eps90 = eps_all["eps_q90"]
    arms = ("A", "B", "Atau")
    cells = {(a, N): cell_reps(cohort, exp, a, N) for a in arms for N in Ns}
    R = balanced_R(list(cells.values()), target)
    interim = R < target
    print(f"S2.1: balanced R = {R} (target 64) {'INTERIM' if interim else 'FINAL'}")
    if R < 2:
        print("not enough runs")
        return
    rid = lambda a, N, r: cells[(a, N)][r]
    jobs, sets = [], {}

    def add(set_name, tag, key, pre, pairs, transform=None):
        sets.setdefault(set_name, {})[tag] = [(a, b, transform) for a, b in pairs]
        for a, b in pairs:
            jobs.append((cohort, tag, key, pre, a, b, transform, None))

    reps_spec = [("z", "z", "l2"), ("h", "h", "l2"), ("z_raw", "z", "none")]
    for N in Ns:
        for tag, key, pre in reps_spec:
            for a in arms:
                add(f"within_{a}_{N}", tag, key, pre, [(rid(a, N, 2 * j), rid(a, N, 2 * j + 1)) for j in range(R // 2)])
            add(f"cross_A_B_{N}", tag, key, pre, [(rid("A", N, i), rid("B", N, i)) for i in range(R)])
            add(f"cross_A_Atau_{N}", tag, key, pre, [(rid("A", N, i), rid("Atau", N, i)) for i in range(R)])
            add(f"cross_Atau_B_{N}", tag, key, pre, [(rid("Atau", N, i), rid("B", N, i)) for i in range(R)])
        # calibration arm (Bernstein warp of A's cosine Gram), z-l2 only
        add(f"bern_self_{N}", "z", "z", "l2", [(rid("A", N, i), rid("A", N, i)) for i in range(R)], "bern")
        add(f"bern_peer_{N}", "z", "z", "l2", [(rid("A", N, 2 * j), rid("A", N, 2 * j + 1)) for j in range(R // 2)], "bern")
        # random-init null (within pairs at initialisation), z-l2
        for a in arms:
            add(f"init_within_{a}_{N}", "z_init", "z_init", "l2",
                [(rid(a, N, 2 * j), rid(a, N, 2 * j + 1)) for j in range(R // 2)])
    # fixed-epoch sensitivity cell (descriptive)
    fe = {a: cell_reps(cohort, "S2.1-fixedepoch", a, 50000) for a in ("A", "B")}
    Rfe = balanced_R(list(fe.values()), 32)
    if Rfe >= 2:
        for a in ("A", "B"):
            add(f"fixedepoch_within_{a}", "z", "z", "l2", [(fe[a][2 * j], fe[a][2 * j + 1]) for j in range(Rfe // 2)])
        add("fixedepoch_cross_A_B", "z", "z", "l2", [(fe["A"][i], fe["B"][i]) for i in range(Rfe)])
    res = run_jobs(jobs, n_jobs)

    def D(set_name, tag):
        return [res[(tag, a, b, t)]["deltas"] for a, b, t in sets[set_name][tag]]

    eps_for = lambda tag: eps["z"] if tag == "z_init" else eps[tag]
    tables = {}
    for s_name, by_tag in sets.items():
        for tag in by_tag:
            tables[f"{s_name}|{tag}"] = pset_decisions(D(s_name, tag), eps_for(tag), C.TOL_KEYS + ["omega"])
    # --------------------------------------------------------------- endpoints (z-l2)
    Nmax, Nmin = Ns[-1], Ns[0]
    z = lambda s: D(s, "z")
    ph = lambda s, k, tag="z": p_hat_at(D(s, tag), eps_for(tag)[k], k)
    ceil = lambda a, b, N, k, tag="z": math.sqrt(max(ph(f"within_{a}_{N}", k, tag), 0) * max(ph(f"within_{b}_{N}", k, tag), 0))
    E = {}
    ordk = ["dN_k10", "dO", "dE", "dMST"]
    gaps = {k: ceil("A", "B", Nmax, k) - ph(f"cross_A_B_{Nmax}", k) for k in ordk}
    imp = {k: boot_mean_diff(z(f"cross_A_B_{Nmax}"), z(f"cross_A_B_{Nmin}"), k) for k in ordk}
    hm = holm({k: v["p_one_sided_diff_ge_0"] for k, v in imp.items()})
    E["i"] = {"gap_to_ceiling_Nmax": gaps, "improvement_500k_minus_50k": imp, "holm": hm,
              "ceiling_Nmax": {k: ceil("A", "B", Nmax, k) for k in ordk},
              "p_cross_Nmax": {k: ph(f"cross_A_B_{Nmax}", k) for k in ordk},
              "p_cross_Nmin": {k: ph(f"cross_A_B_{Nmin}", k) for k in ordk},
              "pass": all(gaps[k] <= 0.1 for k in ordk) and all(hm[k]["reject"] for k in ordk)}
    geok = ["Gs", "dAff", "dSOG_t0p1"]
    ii = {}
    for k in geok:
        dec = {N: tables[f"cross_A_B_{N}|z"][k]["decision"] for N in Ns}
        k3 = {a: ph(f"within_{a}_{Nmax}", k) for a in ("A", "B")}
        guard = all(v >= 0.8 for v in k3.values())
        ii[k] = {"cross_decision": dec, "within_p_Nmax": k3, "K3_guard_ok": guard,
                 "status": ("inconclusive (K3)" if not guard else
                            ("pass" if all(d == "unreliable" for d in dec.values()) else "fail"))}
    E["ii"] = {"items": ii, "pass": all(v["status"] == "pass" for v in ii.values()),
               "any_K3": any(v["status"].startswith("inconclusive") for v in ii.values())}
    iso_cross = float(np.mean([d["dIso"] for d in z(f"cross_A_B_{Nmax}") if np.isfinite(d["dIso"])]))
    iso_within = float(np.mean([d["dIso"] for a in ("A", "B") for d in z(f"within_{a}_{Nmax}") if np.isfinite(d["dIso"])]))
    E["iii"] = {"mean_dIso_cross": iso_cross, "mean_dIso_within_AB": iso_within, "ratio": iso_cross / iso_within,
                "pass": iso_cross <= 1.5 * iso_within}
    med = lambda s, N: float(np.nanmedian([d["omega"] for d in z(f"{s}_{N}")]))
    E["iv"] = {N: {"omega_cross_AB": med("cross_A_B", N), "omega_within_A": med("within_A", N),
                   "omega_within_B": med("within_B", N), "omega_within_Atau": med("within_Atau", N),
                   "omega_cross_A_Atau": med("cross_A_Atau", N), "omega_cross_Atau_B": med("cross_Atau_B", N)}
               for N in Ns}
    E["iv"]["pass"] = (E["iv"][Nmax]["omega_cross_AB"] >= 0.5 and E["iv"][Nmax]["omega_within_A"] <= 0.2
                       and E["iv"][Nmax]["omega_within_B"] <= 0.2)
    v = {}
    for k in ("Gs", "dCKA", "dSOG_t0p1"):
        c = ceil("A", "Atau", Nmax, k)
        p = ph(f"cross_A_Atau_{Nmax}", k)
        v[k] = {"p_cross": p, "ceiling": c, "pass": p >= c - 0.1}
    rawdec = tables[f"cross_A_Atau_{Nmax}|z"].get("rawG", {}).get("decision")
    rw = {a: ph(f"within_{a}_{Nmax}", "rawG") for a in ("A", "Atau")}
    v["rawG"] = {"cross_decision": rawdec, "within_p": rw,
                 "pass": rawdec == "unreliable" and all(x >= 0.8 for x in rw.values())}
    E["v"] = {"items": v, "pass": all(x["pass"] for x in v.values())}
    falsified = E["iv"][Nmax]["omega_cross_AB"] < 0.2 or any(g > 0.1 for g in gaps.values())
    if all(E[k]["pass"] for k in ("i", "ii", "iii", "iv", "v")) and not E["ii"]["any_K3"]:
        verdict = "link confirmed"
    elif falsified:
        verdict = "falsified"
    else:
        verdict = "partial: " + ", ".join(f"({k}) {'pass' if E[k]['pass'] else 'fail'}" for k in ("i", "ii", "iii", "iv", "v"))
        if E["ii"]["any_K3"]:
            verdict += "; (ii) has K3-inconclusive items"
    # CKA cross minus within
    cka = {}
    for N in Ns:
        cka[N] = boot_mean_diff(z(f"cross_A_B_{N}"), z(f"within_A_{N}") + z(f"within_B_{N}"), "dCKA")
    # --------------------------------------------------------------- secondary
    beta = s21_beta(cells, R, Ns, cohort)
    lem6 = s21_lemma6(cells, R, Ns, cohort)
    ustat = s21_ustats(cells, R, Ns, cohort)
    nulls = s21_perm_nulls(cells, R, Ns, cohort)
    cov = s21_covariates(cells, R, Ns, cohort)
    summary = {"interim": interim, "R": R, "target": target, "cohort": cohort, "dry_run_on_pilot": dry_pilot, "eps_file_sha256": sha(eps_path), "eps_file": eps_path,
               "prereg_sha256": sha(os.path.join(ROOT, "prereg", "PREREG_stage2.md")),
               "verdict": verdict, "endpoints": E, "cka_cross_minus_within": cka, "beta": beta, "lemma6": lem6,
               "ustat": ustat, "perm_nulls": nulls, "covariates": cov, "fixed_epoch_R": Rfe,
               "tables": {k: {kk: {x: y for x, y in vv.items() if x != "values"} for kk, vv in t.items()}
                          for k, t in tables.items()},
               "values": {k: {kk: vv["values"] for kk, vv in t.items()} for k, t in tables.items()},
               "eps": eps, "eps_q90": eps90,
               "p_hat_q90": {f"{s}|{tag}": {k: p_hat_at(D(s, tag), (eps90["z"] if tag == "z_init" else eps90[tag])[k], k)
                                             for k in C.TOL_KEYS} for s, bt in sets.items() for tag in bt}}
    out = os.path.join(ROOT, "results", outdir, "s21_summary.json")
    jdump(summary, out)
    write_s21_report(summary, out)


def _cos_gram(X):
    X = np.asarray(X, np.float64)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T


def s21_beta(cells, R, Ns, cohort):
    from scipy.optimize import minimize_scalar
    iu = None
    out = {}
    for N in Ns:
        for a in ("A", "Atau"):
            bs = []
            for i in range(R):
                A = C.load_run(cohort, cells[(a, N)][i])
                B = C.load_run(cohort, cells[("B", N)][i])
                if not (C.run_ok(A, "z") and C.run_ok(B, "z")):
                    bs.append(float("nan"))
                    continue
                KA, KB = _cos_gram(A["feats"]["z"]), _cos_gram(B["feats"]["z"])
                if iu is None:
                    iu = np.triu_indices(KA.shape[0], 1)
                x, y = KA[iu] - 1.0, KB[iu]
                f = lambda lb: float(np.sum((y - np.exp(np.exp(lb) * x)) ** 2))
                r = minimize_scalar(f, bounds=(math.log(0.01), math.log(100)), method="bounded",
                                    options={"xatol": 1e-4})
                bs.append(float(np.exp(r.x)))
            out[f"{a}_B_{N}"] = {"median": float(np.nanmedian(bs)), "q25": float(np.nanquantile(bs, 0.25)),
                                 "q75": float(np.nanquantile(bs, 0.75)), "values": bs}
        ra = out[f"A_B_{N}"]["median"] / out[f"Atau_B_{N}"]["median"]
        out[f"ratio_{N}"] = ra
        out[f"pred_ok_{N}"] = bool(2.5 <= out[f"A_B_{N}"]["median"] <= 10 and 1.5 <= ra <= 3.5)
    return out


def s21_lemma6(cells, R, Ns, cohort):
    iu = None
    per = {}
    for a in ("A", "B", "Atau"):
        for N in Ns:
            en, ef = [], []
            for j in range(R // 2):
                A = C.load_run(cohort, cells[(a, N)][2 * j])
                B = C.load_run(cohort, cells[(a, N)][2 * j + 1])
                if not (C.run_ok(A, "z") and C.run_ok(B, "z")):
                    continue
                dA = np.sqrt(np.clip(2 - 2 * _cos_gram(A["feats"]["z"]), 0, None))
                dB = np.sqrt(np.clip(2 - 2 * _cos_gram(B["feats"]["z"]), 0, None))
                if iu is None:
                    iu = np.triu_indices(dA.shape[0], 1)
                xa, xb = dA[iu], dB[iu]
                xa = xa / np.quantile(xa[xa > 0], 0.9)
                xb = xb / np.quantile(xb[xb > 0], 0.9)
                ok = (xa > 0) & (xb > 0)
                xa, xb = xa[ok], xb[ok]
                avg = 0.5 * (xa + xb)
                near = avg <= np.quantile(avg, 0.01)
                far = avg >= np.quantile(avg, 0.5)
                lr = np.log(xa / xb)
                en.append(float(np.sqrt(np.mean(lr[near] ** 2))))
                ef.append(float(np.sqrt(np.mean(lr[far] ** 2))))
            per[(a, N)] = (np.array(en), np.array(ef))
    out = {}
    rng = np.random.default_rng(12345)
    for a in ("A", "B", "Atau"):
        n50, f50 = per[(a, Ns[0])]
        n5, f5 = per[(a, Ns[-1])]
        if min(len(n50), len(n5)) == 0:
            continue
        Rn, Rf = n50.mean() / n5.mean(), f50.mean() / f5.mean()
        bs = []
        for _ in range(NBOOT):
            i1 = rng.integers(0, len(n50), len(n50))
            i2 = rng.integers(0, len(n5), len(n5))
            bs.append(math.log((n50[i1].mean() / n5[i2].mean()) / (f50[i1].mean() / f5[i2].mean())))
        out[a] = {"e_near_50k": float(n50.mean()), "e_near_500k": float(n5.mean()), "e_far_50k": float(f50.mean()),
                  "e_far_500k": float(f5.mean()), "R_near": float(Rn), "R_far": float(Rf),
                  "log_Rnear_over_Rfar": float(math.log(Rn / Rf)),
                  "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "_boot": bs}
    if "A" in out and "Atau" in out:
        d = np.array(out["A"]["_boot"]) - np.array(out["Atau"]["_boot"])
        out["A_minus_Atau"] = {"diff": out["A"]["log_Rnear_over_Rfar"] - out["Atau"]["log_Rnear_over_Rfar"],
                               "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}
        out["prediction_ok"] = bool(out["A"]["R_near"] > out["A"]["R_far"] and out["Atau"]["R_near"] > out["Atau"]["R_far"]
                                    and out["A_minus_Atau"]["diff"] > 0)
    for a in ("A", "B", "Atau"):
        if a in out:
            out[a].pop("_boot")
    return out


def _run_cheap(cohort, rid, key="z"):
    from hrhv2 import structural as st
    run = C.load_run(cohort, rid)
    if not C.run_ok(run, key):
        return None
    X = np.asarray(run["feats"][key], np.float64)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    K = X @ X.T
    Dm = np.clip(2 - 2 * K, 0, None)
    np.fill_diagonal(Dm, 0)
    nb, _ = st.knn_table(Dm, 11)
    iu = np.triu_indices(X.shape[0], 1)
    d = Dm[iu]
    Xc = X - X.mean(0)
    occ = np.bincount(nb[:, :10].ravel(), minlength=X.shape[0])
    from scipy.stats import skew
    dn = (d / d.mean()).astype(np.float32)
    return {"nb": nb[:, :10], "d": dn, "dnorm": float(np.linalg.norm(dn.astype(np.float64) - 1.0)),
            "Xc": Xc, "XtX": np.linalg.norm(Xc.T @ Xc), "hub_skew": float(skew(occ))}


def _pair_cheap(a, b):
    if a is None or b is None:
        return (0.0, np.inf, 0.0, -1.0)
    hit = (a["nb"][:, :, None] == b["nb"][:, None, :]).any(2).sum() / a["nb"].size
    gs = float(np.sqrt(np.mean((a["d"].astype(np.float64) - b["d"]) ** 2)))
    cka = float(np.linalg.norm(a["Xc"].T @ b["Xc"]) ** 2 / (a["XtX"] * b["XtX"]))
    rsa = float(np.dot(a["d"].astype(np.float64) - 1.0, b["d"].astype(np.float64) - 1.0) / (a["dnorm"] * b["dnorm"]))
    return (float(hit), gs, cka, rsa)


def s21_ustats(cells, R, Ns, cohort):
    """All-pairs U-statistics (s_N(10), Gs_rms, CKA, RSA) on z-l2 with run-level bootstrap."""
    out = {}
    names = ("sN10", "Gs", "CKA", "RSA")
    rng = np.random.default_rng(777)
    for N in Ns:
        feats = {a: [_run_cheap(cohort, cells[(a, N)][i]) for i in range(R)] for a in ("A", "B", "Atau")}
        hub = {a: float(np.nanmean([f["hub_skew"] for f in feats[a] if f is not None])) for a in feats}
        out[f"hubness_skew_{N}"] = hub
        for (a, b) in (("A", "A"), ("B", "B"), ("Atau", "Atau"), ("A", "B"), ("A", "Atau"), ("Atau", "B")):
            M = np.full((len(names), R, R), np.nan)
            for i in range(R):
                for j in range(R):
                    if a == b and j <= i:
                        continue
                    M[:, i, j] = _pair_cheap(feats[a][i], feats[b][j])
                    if a == b:
                        M[:, j, i] = M[:, i, j]
            est = np.nanmean(M.reshape(len(names), -1), axis=1)
            bs = []
            for _ in range(500):
                ia = rng.integers(0, R, R)
                ib = ia if a == b else rng.integers(0, R, R)
                sub = M[:, ia][:, :, ib]
                if a == b:
                    mask = ia[:, None] != ia[None, :]
                    sub = sub[:, mask]
                bs.append(np.nanmean(sub.reshape(len(names), -1), axis=1))
            bs = np.array(bs)
            out[f"{a}_{b}_{N}"] = {nm: {"mean": float(est[k]), "ci95": [float(np.nanquantile(bs[:, k], 0.025)),
                                                                         float(np.nanquantile(bs[:, k], 0.975))]}
                                   for k, nm in enumerate(names)}
    return out


def s21_perm_nulls(cells, R, Ns, cohort, K=200, npairs=4):
    from hrhv2 import calibrate as cal, geometric as geo
    out = {}
    for N in Ns:
        for s_name, pairs in (("within_A", [(cells[("A", N)][2 * j], cells[("A", N)][2 * j + 1]) for j in range(R // 2)]),
                              ("cross_A_B", [(cells[("A", N)][i], cells[("B", N)][i]) for i in range(R)])):
            rows = []
            for ra, rb in pairs[:npairs]:
                A, B = C.load_run(cohort, ra), C.load_run(cohort, rb)
                if not (C.run_ok(A, "z") and C.run_ok(B, "z")):
                    continue
                GA, GB = C.geom(A["feats"]["z"], "l2"), C.geom(B["feats"]["z"], "l2")
                from hrhv2 import structural as st
                nA, _ = st.knn_table(GA.D, 11)
                nB, _ = st.knn_table(GB.D, 11)
                kn = cal.calibrate_knn(nA, nB, 10, key=f"stage2|perm|{ra}|{rb}|knn10", K=K)
                ck = cal.calibrate(lambda x, y: geo.cka_pair(x, y)["cka_biased"], GA, GB, key=f"stage2|perm|{ra}|{rb}|cka", K=K)
                rows.append({"A": ra, "B": rb, "knn10": {k: kn[k] for k in ("s_obs", "null_mean", "p_value", "s_cal")
                                                         if k in kn},
                             "cka": {k: ck[k] for k in ("s_obs", "null_mean", "p_value", "s_cal") if k in ck}})
            out[f"{s_name}_{N}"] = rows
    return out


def s21_covariates(cells, R, Ns, cohort):
    out = {}
    for (a, N), reps in cells.items():
        ms = [C.load_run(cohort, reps[r])["meta"] for r in range(R)]
        f = lambda k: [m.get(k, float("nan")) for m in ms]
        out[f"{a}_{N}"] = {k: {"mean": float(np.nanmean(f(k))), "sd": float(np.nanstd(f(k)))}
                           for k in ("knn_acc_h", "knn_acc_z", "final_train_loss", "evalloss_train", "evalloss_hold",
                                     "gen_gap", "t_total_s", "z_norm_med")}
        out[f"{a}_{N}"]["n_diverged"] = int(sum(bool(m.get("diverged")) for m in ms))
    return out


def fmt(x, d=3):
    try:
        return f"{x:.{d}f}"
    except Exception:
        return str(x)


def write_s21_report(S, out_json):
    L = []
    L.append(f"# S2.1 objective link — {'INTERIM' if S['interim'] else 'FINAL'} analysis (PREREG_stage2 v1.0)\n")
    L.append(f"- runs per (arm, N) used: R = {S['R']} of {S['target']} (balanced prefix); prereg sha256 {S['prereg_sha256'][:16]}…; eps file sha256 {S['eps_file_sha256'][:16]}…")
    L.append(f"- **Verdict: {S['verdict']}**\n")
    E = S["endpoints"]
    L.append("## Endpoints (z, cosine kernel)\n")
    L.append("(i) ordinal near ceiling at 500k and improving with N:")
    L.append("| rung | p̂ cross 50k | p̂ cross 500k | ceiling 500k | gap | mean Δ diff (500k−50k) [95% CI] | Holm reject |")
    L.append("|---|---|---|---|---|---|---|")
    for k in E["i"]["gap_to_ceiling_Nmax"]:
        im = E["i"]["improvement_500k_minus_50k"][k]
        L.append(f"| {k} | {fmt(E['i']['p_cross_Nmin'][k])} | {fmt(E['i']['p_cross_Nmax'][k])} | {fmt(E['i']['ceiling_Nmax'][k])} | "
                 f"{fmt(E['i']['gap_to_ceiling_Nmax'][k])} | {fmt(im['diff'], 4)} [{fmt(im['ci95'][0], 4)}, {fmt(im['ci95'][1], 4)}] | {E['i']['holm'][k]['reject']} |")
    L.append(f"\n(i) pass: {E['i']['pass']}\n")
    L.append("(ii) geometric obstruction (cross A–B unreliable at both N; K3 guard within p̂ ≥ 0.8 at 500k):")
    for k, v in E["ii"]["items"].items():
        L.append(f"- {k}: cross decision {v['cross_decision']}; within p̂ {{{', '.join(f'{a}: {fmt(p)}' for a, p in v['within_p_Nmax'].items())}}} → **{v['status']}**")
    L.append(f"\n(iii) Iso: mean Δ_iso cross {fmt(E['iii']['mean_dIso_cross'], 4)} vs within {fmt(E['iii']['mean_dIso_within_AB'], 4)} (ratio {fmt(E['iii']['ratio'])}; pass if ≤ 1.5): {E['iii']['pass']}\n")
    L.append("(iv) warp share ω (median):")
    for N in [k for k in E["iv"] if k != "pass"]:
        v = E["iv"][N]
        L.append(f"- N={N}: " + ", ".join(f"{k} {fmt(x)}" for k, x in v.items()))
    L.append(f"- pass (500k: cross ≥ 0.5, within A and B ≤ 0.2): {E['iv']['pass']}\n")
    L.append("(v) A vs A_τ:")
    for k, v in E["v"]["items"].items():
        L.append(f"- {k}: " + ", ".join(f"{kk} {fmt(vv) if not isinstance(vv, (dict, str, bool, type(None))) else vv}" for kk, vv in v.items()))
    L.append(f"- pass: {E['v']['pass']}\n")
    L.append("CKA (1 − CKA): cross A–B minus within (A,B):")
    for N, v in S["cka_cross_minus_within"].items():
        L.append(f"- N={N}: {fmt(v['diff'], 4)} [{fmt(v['ci95'][0], 4)}, {fmt(v['ci95'][1], 4)}]")
    L.append("\n## Decision tables (z; ε = max pooled pilot within-arm Δ)\n")
    keys = ["dN_k10", "dO", "dE", "dMST", "Gs", "dAff", "dSOG_t0p1", "dCKA", "rawG", "dIso", "dPi"]
    L.append("| pair set | " + " | ".join(keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for name, t in S["tables"].items():
        if not name.endswith("|z") and not name.endswith("|z_init"):
            continue
        cells = []
        for k in keys:
            if k in t:
                d = t[k]["decision"]
                cells.append(f"{d[:3]} {t[k]['x']}/{t[k]['n']}")
            else:
                cells.append("–")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("\n(rel = reliable, unr = unreliable, inc = inconclusive)\n")
    L.append("## Secondary\n")
    b = S["beta"]
    L.append("Sharp link β̂ (fit K_B ≈ exp(β(K_A − 1))): " + "; ".join(
        f"{k}: median {fmt(v['median'])}" for k, v in b.items() if isinstance(v, dict)) +
        "; ratios " + ", ".join(f"{k} {fmt(v)}" for k, v in b.items() if k.startswith("ratio")) +
        "; prediction ok " + ", ".join(f"{k} {v}" for k, v in b.items() if k.startswith("pred_ok")))
    L.append("\nLemma 6 (near vs far log-distance error, improvement 50k→500k):")
    for a, v in S["lemma6"].items():
        if isinstance(v, dict) and "R_near" in v:
            L.append(f"- {a}: R_near {fmt(v['R_near'])}, R_far {fmt(v['R_far'])}, log ratio {fmt(v['log_Rnear_over_Rfar'])} [{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}]")
    if "A_minus_Atau" in S["lemma6"]:
        v = S["lemma6"]["A_minus_Atau"]
        L.append(f"- A − A_τ: {fmt(v['diff'])} [{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}]; prediction ok: {S['lemma6'].get('prediction_ok')}")
    L.append("\nAll-pairs U-statistics (z-l2; mean [95% run-bootstrap CI]):")
    for k, v in S["ustat"].items():
        if k.startswith("hubness"):
            L.append(f"- {k}: " + ", ".join(f"{a} {fmt(x)}" for a, x in v.items()))
        else:
            L.append(f"- {k}: " + "; ".join(f"{nm} {fmt(x['mean'])} [{fmt(x['ci95'][0])}, {fmt(x['ci95'][1])}]" for nm, x in v.items()))
    L.append("\nCovariates (mean ± sd):")
    for k, v in S["covariates"].items():
        L.append(f"- {k}: " + ", ".join(f"{kk} {fmt(vv['mean'])}±{fmt(vv['sd'])}" for kk, vv in v.items() if isinstance(vv, dict))
                 + f", diverged {v['n_diverged']}")
    L.append("\nPermutation nulls (first 4 pairs, K=200): see s21_summary.json `perm_nulls`.")
    open(out_json.replace("s21_summary.json", "S2_1_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


# ============================================================================ S2.2
def s22(n_jobs):
    cohort = "S2_2_main"
    eps_all = json.load(open(os.path.join(ROOT, "prereg", "eps_S2_2.json")))
    eps = eps_all["eps"]
    eps90 = eps_all["eps_q90"]
    arms = ("noWD", "WD", "smallinit")
    Ns = (25000, 250000)
    cells = {(a, N): cell_reps(cohort, "S2.2", a, N) for a in arms for N in Ns}
    R = balanced_R(list(cells.values()), 192)
    shared = {c: cell_reps(cohort, "S2.2", "noWD-shared", 25000, c) for c in range(4)}
    Rs = balanced_R(list(shared.values()), 48)
    interim = R < 192 or Rs < 48
    print(f"S2.2: balanced R = {R} (target 192), shared R = {Rs} (target 48) {'INTERIM' if interim else 'FINAL'}")
    reps_spec = [("y", "y", "none"), ("u", "u", "none"), ("h", "h", "none"), ("h_l2", "h", "l2")]
    jobs, sets = [], {}

    def add(set_name, tag, key, pre, pairs):
        sets.setdefault(set_name, {})[tag] = [(a, b, None) for a, b in pairs]
        for a, b in pairs:
            jobs.append((cohort, tag, key, pre, a, b, None, None))

    for N in Ns:
        for a in arms:
            pairs = [(cells[(a, N)][2 * j], cells[(a, N)][2 * j + 1]) for j in range(R // 2)]
            for tag, key, pre in reps_spec:
                add(f"{a}_{N}", tag, key, pre, pairs)
    if Rs >= 2:
        pairs = [(shared[c][2 * j], shared[c][2 * j + 1]) for c in range(4) for j in range(Rs // 2)]
        for tag, key, pre in reps_spec:
            add("noWD-shared_25000", tag, key, pre, pairs)
    res = run_jobs(jobs, n_jobs)
    D = lambda s, tag: [res[(tag, a, b, t)]["deltas"] for a, b, t in sets[s][tag]]
    tables = {f"{s}|{tag}": pset_decisions(D(s, tag), eps[tag], C.TOL_KEYS) for s, bt in sets.items() for tag in bt}
    dec = lambda s, tag, k: tables[f"{s}|{tag}"][k]["decision"]
    ph = lambda s, tag, k: tables[f"{s}|{tag}"][k]["p_hat"]
    prim = ("dN_k10", "Gs")
    P = {}
    P["P1_noWD_u_unreliable"] = {f"{k}_{N}": dec(f"noWD_{N}", "u", k) for k in prim for N in Ns}
    P["P1_noWD_u_unreliable"]["pass"] = all(v == "unreliable" for v in P["P1_noWD_u_unreliable"].values())
    p2 = {f"y_{k}_{N}": dec(f"noWD_{N}", "y", k) for k in prim for N in Ns}
    p2.update({f"Pi_{N}": dec(f"noWD_{N}", "u", "dPi") for N in Ns})
    p2["pass"] = all(dec("noWD_250000", "y", k) == "reliable" for k in prim) and dec("noWD_250000", "u", "dPi") == "reliable"
    P["P2_noWD_y_Pi_reliable_250k"] = p2
    p3 = {f"{a}_{k}_{N}": dec(f"{a}_{N}", "u", k) for a in ("WD", "smallinit") for k in prim for N in Ns}
    p3["pass_WD"] = all(dec("WD_250000", "u", k) == "reliable" for k in prim)
    p3["pass_smallinit"] = all(dec("smallinit_250000", "u", k) == "reliable" for k in prim)
    P["P3_WD_smallinit_u_reliable_250k"] = p3
    p4 = {}
    for N in Ns:
        m = {a: float(np.mean([1 - d["dN_k10"] for d in D(f"{a}_{N}", "h") if np.isfinite(d["dN_k10"])])) for a in arms}
        p4[N] = {"mean_sN10_h": m, "max_abs_diff": max(abs(m[x] - m[y]) for x in arms for y in arms)}
    p4["pass"] = all(p4[N]["max_abs_diff"] < 0.05 for N in Ns)
    P["P4_h_localization"] = p4
    p5 = {f"{k}_{N}": abs(ph(f"smallinit_{N}", "u", k) - ph(f"WD_{N}", "u", k)) for k in prim for N in Ns}
    p5["pass"] = all(v <= 0.1 for v in p5.values())
    P["P5_smallinit_matches_WD"] = p5
    fals = any(dec(f"noWD_{N}", "u", k) == "reliable" for k in prim for N in Ns)
    P["falsifier_noWD_u_reliable"] = fals
    if Rs >= 2:
        sh = {}
        for tag in ("y", "u", "h"):
            for k in prim + ("dPi", "dCKA"):
                if k in tables[f"noWD-shared_25000|{tag}"]:
                    sh[f"{tag}_{k}"] = {"p_shared": ph("noWD-shared_25000", tag, k), "p_indep": ph("noWD_25000", tag, k),
                                        "dec_shared": dec("noWD-shared_25000", tag, k), "dec_indep": dec("noWD_25000", tag, k),
                                        "mean_delta_diff_shared_minus_indep": boot_mean_diff(
                                            D("noWD-shared_25000", tag), D("noWD_25000", tag), k)}
        P["shared_corpus"] = sh
    # per-run checks and covariates
    cov, pi_check = {}, []
    from hrhv2 import gl
    for (a, N), reps in list(cells.items()) + [(("noWD-shared", 25000), {}) ]:
        if a == "noWD-shared":
            rids = [shared[c][r] for c in range(4) for r in range(Rs)]
        else:
            rids = [reps[r] for r in range(R)]
        if not rids:
            continue
        ms = []
        for r_ in rids:
            run = C.load_run(cohort, r_)
            ms.append(run["meta"])
            W = run["feats"]["W"]
            if np.linalg.cond(W) < 1e6 and C.run_ok(run, "u"):
                pi_check.append(gl.projector_distance(run["feats"]["u"].astype(np.float64),
                                                      run["feats"]["y"].astype(np.float64))["gl_projector_dist"])
        f = lambda k: [m.get(k, float("nan")) for m in ms]
        imb = [m["imb_final"]["imb_rel"] for m in ms]
        cov[f"{a}_{N}"] = {"n": len(ms), "acc_bank": float(np.mean(f("acc_bank"))), "acc_S": float(np.mean(f("acc_S"))),
                           "gen_gap": float(np.mean(f("gen_gap"))), "ce_train": float(np.mean(f("ce_train"))),
                           "imb_rel_median": float(np.median(imb)), "imb_rel_q10": float(np.quantile(imb, 0.1)),
                           "imb_rel_q90": float(np.quantile(imb, 0.9)), "n_diverged": int(sum(bool(m.get("diverged")) for m in ms)),
                           "t_total_s": float(np.mean(f("t_total_s")))}
    summary = {"interim": interim, "R": R, "R_shared": Rs, "target": 192, "target_shared": 48,
               "prereg_sha256": sha(os.path.join(ROOT, "prereg", "PREREG_stage2.md")),
               "eps_file_sha256": sha(os.path.join(ROOT, "prereg", "eps_S2_2.json")),
               "predictions": P, "covariates": cov,
               "pi_u_eq_pi_y_max": float(np.max(pi_check)) if pi_check else float("nan"),
               "pi_check_n": len(pi_check),
               "tables": {k: {kk: {x: y for x, y in vv.items() if x != "values"} for kk, vv in t.items()} for k, t in tables.items()},
               "values": {k: {kk: vv["values"] for kk, vv in t.items()} for k, t in tables.items()},
               "eps": eps, "eps_q90": eps90,
               "p_hat_q90": {f"{s}|{tag}": {k: p_hat_at(D(s, tag), eps90[tag][k], k) for k in C.TOL_KEYS}
                             for s, bt in sets.items() for tag in bt}}
    out = os.path.join(ROOT, "results", "S2_2", "s22_summary.json")
    jdump(summary, out)
    write_s22_report(summary, out)


def write_s22_report(S, out_json):
    L = [f"# S2.2 gauge net — {'INTERIM' if S['interim'] else 'FINAL'} analysis (PREREG_stage2 v1.0)\n",
         f"- pairs per cell: R/2 with R = {S['R']} of {S['target']} runs; shared-corpus R = {S['R_shared']} of {S['target_shared']} per corpus",
         f"- prereg sha256 {S['prereg_sha256'][:16]}…; eps sha256 {S['eps_file_sha256'][:16]}…; Π_u = Π_y check: max {fmt(S['pi_u_eq_pi_y_max'], 2)} over {S['pi_check_n']} runs\n",
         "## Predictions\n"]
    for k, v in S["predictions"].items():
        if isinstance(v, dict):
            L.append(f"- **{k}**: " + "; ".join(f"{kk}: {vv if not isinstance(vv, float) else fmt(vv)}" for kk, vv in v.items()
                                                 if not isinstance(vv, dict)))
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    L.append(f"  - {kk}: " + ", ".join(f"{a}: {fmt(b) if isinstance(b, float) else b}" for a, b in vv.items()))
        else:
            L.append(f"- **{k}**: {v}")
    L.append("\n## Decision tables (ε function-anchored: u uses ε(y))\n")
    keys = ["dN_k10", "Gs", "dCKA", "dO", "dE", "dMST", "dAff", "dSOG_t0p1", "dPi"]
    L.append("| set | rep | " + " | ".join(keys) + " |")
    L.append("|---|---|" + "---|" * len(keys))
    for name, t in S["tables"].items():
        s, tag = name.split("|")
        L.append(f"| {s} | {tag} | " + " | ".join(
            (f"{t[k]['decision'][:3]} {t[k]['x']}/{t[k]['n']}" if k in t else "–") for k in keys) + " |")
    L.append("\n## Covariates\n")
    for k, v in S["covariates"].items():
        L.append(f"- {k}: " + ", ".join(f"{kk} {fmt(vv)}" for kk, vv in v.items()))
    open(out_json.replace("s22_summary.json", "S2_2_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:30]))


def status_cmd():
    for cohort in ("S2_2_pilot", "S2_1_pilot", "S2_2_main", "S2_1_main"):
        p = os.path.join(ROOT, "plans", f"{cohort}.json")
        runs = plan_runs(cohort)
        done = sum(is_done(cohort, r["run_id"]) for r in runs)
        failed = sum(os.path.exists(os.path.join(C.run_dir(cohort, r["run_id"]), ".failed")) for r in runs)
        print(f"{cohort}: {done}/{len(runs)} done, {failed} failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["s21", "s22", "status"])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--dry-pilot", action="store_true", help="pipeline test of s21 on EXCLUDED pilot runs")
    ap.add_argument("--eps-file", default=None, help="only for --dry-pilot code tests")
    a = ap.parse_args()
    if a.what == "status":
        status_cmd()
    elif a.what == "s21":
        if a.eps_file and not a.dry_pilot:
            sys.exit("--eps-file is only allowed with --dry-pilot")
        s21(a.jobs, a.dry_pilot, a.eps_file)
    else:
        s22(a.jobs)
