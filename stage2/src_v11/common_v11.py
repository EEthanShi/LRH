"""S2.2 v1.1 shared analysis helpers (PREREG_S2_2_v1.1.md sections 7-9, 13).

Run loading (runs_v11/<cohort>/<run_id>), cells and disjoint pairs, representations, cached pair ladders
(full hrhv2 ladder via crit.py, unchanged v1.0 copy; reduced ladder = the same hrhv2 ladder() restricted to the
counterfactual rungs R_cf, values identical to the full ladder's, checked in KA0), CP decisions (crit.decide),
bootstrap helpers and Holm (copied from v1.0 analyze.py).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import get_context

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crit as C  # noqa: E402  (verbatim v1.0 copy)
import gaugemath_v11 as GM  # noqa: E402
from hrhv2.ladder import ladder as hr_ladder  # noqa: E402

ROOT = C.ROOT
RUNS = os.path.join(ROOT, "runs_v11")
RES = os.path.join(ROOT, "results", "S2_2_v11")
PLANS = os.path.join(ROOT, "plans_v11")
EPS_PATH = os.path.join(RES, "eps", "eps_S2_2_v11.json")
NS = (25000, 250000)

# counterfactual rungs R_cf = {T_N(10), T_O, T_E, MST, G_s, 1-CKA, 1-RSA, SOG_0.1}
R_CF = ["dN_k10", "dO", "dE", "dMST", "Gs", "dCKA", "dAff", "dSOG_t0p1"]
TOL_KEYS = list(C.TOL_KEYS)
FAIL = C.FAIL
FULL_TAGS = {"y": ("y", "none"), "u": ("u", "none"), "h": ("h", "none"), "h_l2": ("h", "l2")}
CF_TAGS = ("uc", "ucpA", "ucpB", "ubal", "ubalc")


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda v: float(v) if isinstance(v, (np.floating, np.integer)) else str(v))
    os.replace(tmp, path)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def key_rng(key: str):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


# ============================================================================ runs, cells, pairs
def plan_runs(cohort):
    return json.load(open(os.path.join(PLANS, f"{cohort}.json")))["runs"]


def run_dir(cohort, rid):
    """cohort 'V10:<name>' reads the v1.0 cohort runs/<name> READ-ONLY (known-answer tests only)."""
    if cohort.startswith("V10:"):
        return os.path.join(ROOT, "runs", cohort[4:], rid)
    return os.path.join(RUNS, cohort, rid)


def run_state(cohort, rid):
    d = run_dir(cohort, rid)
    if os.path.exists(os.path.join(d, ".done")):
        return "done"
    if os.path.exists(os.path.join(d, ".failed")):
        return "failed"
    return "missing"


def cohort_cells(cohort):
    """{(arm, N): {rep: run_id}} over the PLANNED runs."""
    cells = {}
    for r in plan_runs(cohort):
        cells.setdefault((r["arm"], r["N"]), {})[r["rep"]] = r["run_id"]
    return cells


def cohort_complete(cohort):
    st = [run_state(cohort, r["run_id"]) for r in plan_runs(cohort)]
    return all(s in ("done", "failed") for s in st), {s: st.count(s) for s in ("done", "failed", "missing")}


def cell_pairs(cells, arm, N):
    reps = cells[(arm, N)]
    n = len(reps)
    return [(reps[2 * j], reps[2 * j + 1]) for j in range(n // 2)]


_CACHE = {}


def load_run(cohort, rid, cache=False):
    k = (cohort, rid)
    if cache and k in _CACHE:
        return _CACHE[k]
    d = run_dir(cohort, rid)
    if not os.path.exists(os.path.join(d, ".done")):
        return None
    meta = json.load(open(os.path.join(d, "meta.json")))
    feats = dict(np.load(os.path.join(d, "feats.npz")))
    if cohort.startswith("V10:"):
        meta["_v10"] = True
    run = {"meta": meta, "feats": feats}
    if cache:
        _CACHE[k] = run
    return run


def run_ok(run):
    if run is None:
        return False
    m, f = run["meta"], run["feats"]
    if m.get("diverged", False) or not m.get("finite", False):
        return False
    return all(np.all(np.isfinite(f[k])) for k in ("y", "u", "h", "P", "W"))


def u_recomputed(run):
    """u = W^{-1} y in float64 (section 7); the stored u must agree to 1e-10 (reported, not used).
    v1.0 runs (known-answer tests only) use their stored u, as every v1.0 analysis did."""
    if run["meta"].get("_v10"):
        return np.asarray(run["feats"]["u"], np.float64)
    y = np.asarray(run["feats"]["y"], np.float64)
    W = np.asarray(run["feats"]["W"], np.float64)
    return y @ np.linalg.inv(W).T


# ============================================================================ representations
def rep_features(run, tag, other=None, Gbar=None, Gbar_bal=None):
    """Features for one run of a pair. tag in FULL_TAGS or CF_TAGS. other = the pair partner (for ucpA/ucpB)."""
    f = run["feats"]
    y = np.asarray(f["y"], np.float64)
    if tag == "y":
        return y
    if tag == "u":
        return u_recomputed(run)
    if tag in ("h", "h_l2"):
        return np.asarray(f["h"], np.float64)
    if tag == "uc":
        return GM.rep_metric(y, Gbar)
    if tag == "ubal":
        return GM.rep_bal(y, f["P"], f["W"])
    if tag == "ubalc":
        return GM.rep_metric(y, Gbar_bal)
    raise ValueError(tag)


def pair_features(A, B, tag, Gbar=None, Gbar_bal=None):
    """(XA, XB) for a pair. ucpA = both in A's gauge (u_A, W_A^{-1} y_B); ucpB = both in B's gauge
    (W_B^{-1} y_A, u_B) (v1.0 a2_gauge_decomposition 'common_A' / 'common_B')."""
    if tag == "ucpA":
        WAi = np.linalg.inv(np.asarray(A["feats"]["W"], np.float64))
        return u_recomputed(A), np.asarray(B["feats"]["y"], np.float64) @ WAi.T
    if tag == "ucpB":
        WBi = np.linalg.inv(np.asarray(B["feats"]["W"], np.float64))
        return np.asarray(A["feats"]["y"], np.float64) @ WBi.T, u_recomputed(B)
    return (rep_features(A, tag, B, Gbar, Gbar_bal), rep_features(B, tag, A, Gbar, Gbar_bal))


def pre_of(tag):
    return "l2" if tag == "h_l2" else "none"


def reduced_ladder(GA, GB):
    """hrhv2 ladder restricted to the R_cf rungs (same functions, identical values)."""
    return hr_ladder(GA, GB, item_ids=C.ITEM_IDS, n_jobs=1, with_quotients=False, with_gl=False, comp_ks=(),
                     sog_taus=(0.1,), k_grid=(10,))


# ============================================================================ cached ladder jobs
def ladder_path(lroot, cohort, tag, ra, rb):
    return os.path.join(lroot, cohort, tag, f"{ra}__{rb}.json")


def _job(args):
    lroot, cohort, tag, ra, rb, gb, gbb = args
    path = ladder_path(lroot, cohort, tag, ra, rb)
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    A, B = load_run(cohort, ra), load_run(cohort, rb)
    if not (run_ok(A) and run_ok(B)):
        res = {"deltas": {k: FAIL for k in TOL_KEYS}, "failed_run": True}
    else:
        try:
            if tag in ("uc", "ubalc") and (gb if tag == "uc" else gbb) is None:
                raise ValueError("no frozen Gbar / Gbar_bal for this cell")
            XA, XB = pair_features(A, B, tag, None if gb is None else np.array(gb), None if gbb is None else np.array(gbb))
            if not (np.all(np.isfinite(XA)) and np.all(np.isfinite(XB))):
                raise FloatingPointError("non-finite representation (numerically degenerate head)")
            GA, GB = C.geom(XA, pre_of(tag)), C.geom(XB, pre_of(tag))
            if tag in FULL_TAGS:
                row = C.pair_ladder(GA, GB, n_jobs=1)
                d = C.deltas(row)
            else:
                row = reduced_ladder(GA, GB)
                d = {k: v for k, v in C.deltas(row).items() if k in R_CF}
            res = {"deltas": d, "failed_run": False, "time_s": row.get("time_total_s")}
        except (np.linalg.LinAlgError, FloatingPointError, ValueError, ZeroDivisionError) as e:
            # resolution R-NUM: a numerically degenerate head fails every rung of this representation in its pair
            d = {k: FAIL for k in (TOL_KEYS if tag in FULL_TAGS else R_CF)}
            d["_numfail"] = 1
            res = {"deltas": d, "failed_run": False, "numerical_failure": repr(e)[:300]}
    res.update(A=ra, B=rb, tag=tag, cohort=cohort)
    jdump(res, path)
    return res


def run_ladders(jobs, n_jobs, label=""):
    """jobs: list of (lroot, cohort, tag, ra, rb, Gbar or None, Gbar_bal or None). Returns {(tag, ra, rb): deltas}."""
    todo = [j for j in jobs if not os.path.exists(ladder_path(*j[:5]))]
    print(f"  ladders {label}: {len(jobs)} total, {len(todo)} to compute (jobs={n_jobs})", flush=True)
    t = time.time()
    if todo:
        with get_context("fork").Pool(n_jobs) as pool:
            for i, _ in enumerate(pool.imap_unordered(_job, todo, chunksize=1)):
                if (i + 1) % 100 == 0:
                    print(f"    {i + 1}/{len(todo)} ({time.time() - t:.0f} s)", flush=True)
    out = {}
    for j in jobs:
        r = _job(j)
        out[(j[2], j[3], j[4])] = r["deltas"]
    return out


# ============================================================================ statistics
def decide_vals(vals, eps):
    """n, x (#Delta <= eps; non-finite = failure), p_hat, one-sided 95% CP bounds, decision (crit.decide)."""
    v = np.array([np.inf if (x is None or not np.isfinite(x)) else x for x in vals], dtype=float)
    n = len(v)
    if n == 0 or eps is None or not np.isfinite(eps):
        return {"n": n, "x": None, "p_hat": float("nan"), "cp_lo": float("nan"), "cp_hi": float("nan"),
                "decision": "none", "eps": eps}
    x = int(np.sum(v <= eps))
    dec, lo, hi = C.decide(x, n)
    srt = np.sort(v)
    xr = x_reliable(n)
    return {"n": n, "x": x, "p_hat": x / n, "cp_lo": lo, "cp_hi": hi, "decision": dec, "eps": float(eps),
            "x_needed_reliable": xr, "eps_star_r": float(srt[xr - 1]) if xr is not None else float("nan")}


_XR = {}


def x_reliable(n):
    """Smallest x with decision 'reliable' at n (92 at n = 96; 32 at n = 32; None at n = 16)."""
    if n not in _XR:
        _XR[n] = next((x for x in range(n + 1) if C.decide(x, n)[0] == "reliable"), None)
    return _XR[n]


def phat_curve(vals, eps, lo=0.80, hi=1.25, step=0.01):
    v = np.array([np.inf if not np.isfinite(x) else x for x in vals], dtype=float)
    out = []
    r = lo
    while r <= hi + 1e-9:
        out.append([round(r, 2), float(np.mean(v <= eps * r)) if v.size else float("nan")])
        r += step
    return out


def boot_mean_ci(x, key, level=0.95, n=10000):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if x.size == 0:
        return {"mean": float("nan"), "ci": [float("nan")] * 2, "n": 0}
    rng = key_rng(key)
    m = x[rng.integers(0, x.size, (n, x.size))].mean(1)
    a = (1 - level) / 2
    return {"mean": float(x.mean()), "median": float(np.median(x)), "ci": [float(np.quantile(m, a)), float(np.quantile(m, 1 - a))],
            "share_gt0": float(np.mean(x > 0)), "n": int(x.size), "level": level}


def boot_median_ci(x, key, level=0.95, n=10000):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if x.size == 0:
        return {"median": float("nan"), "ci": [float("nan")] * 2, "n": 0}
    rng = key_rng(key)
    m = np.median(x[rng.integers(0, x.size, (n, x.size))], axis=1)
    a = (1 - level) / 2
    return {"median": float(np.median(x)), "ci": [float(np.quantile(m, a)), float(np.quantile(m, 1 - a))], "n": int(x.size),
            "level": level}


def boot_diff_means(a, b, key, n=10000):
    """Bootstrap of mean(a) - mean(b), the two samples resampled independently. One-sided p for H1: diff > 0 is
    (#{boot diff <= 0} + 1) / (n + 1)."""
    a = np.asarray([v for v in a if np.isfinite(v)], float)
    b = np.asarray([v for v in b if np.isfinite(v)], float)
    if a.size == 0 or b.size == 0:
        return {"diff": float("nan"), "ci95": [float("nan")] * 2, "p_one_sided": 1.0, "n_a": int(a.size), "n_b": int(b.size),
                "nboot": n, "note": "empty sample"}
    rng = key_rng(key)
    d = a[rng.integers(0, a.size, (n, a.size))].mean(1) - b[rng.integers(0, b.size, (n, b.size))].mean(1)
    return {"diff": float(a.mean() - b.mean()), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "p_one_sided": float((np.sum(d <= 0) + 1) / (n + 1)), "n_a": int(a.size), "n_b": int(b.size), "nboot": n}


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


def cp_two_sided(x, n, level=0.95):
    from scipy.stats import beta
    a = (1 - level) / 2
    lo = beta.ppf(a, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(1 - a, x + 1, n - x) if x < n else 1.0
    return [float(lo), float(hi)]
