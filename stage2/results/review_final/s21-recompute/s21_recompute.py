"""Independent recompute of the FINAL S2.1 analysis (role s21-recompute).

Reads only: per-pair ladder JSONs (results/ladders/S2_1_main/<tag>/*.json, the hrhv2 'row' block),
prereg/eps_S2_1.json, run meta.json/.done, and (for comparison only) results/S2_1/s21_summary.json.
Does NOT import analyze.py or crit.py. Delta_Q is re-derived from the raw hrhv2 row per
PREREG_stage2.md section 2.4; decisions per section 2.5 are re-implemented here.
Writes only under results/review_final/s21-recompute/.
"""
from __future__ import annotations

import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import hashlib
import json
import math
import sys

import numpy as np
from scipy.stats import beta as Beta, binom

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
LAD = os.path.join(ROOT, "results", "ladders", "S2_1_main")
OUT = os.path.join(ROOT, "results", "review_final", "s21-recompute")
EPS_PATH = os.path.join(ROOT, "prereg", "eps_S2_1.json")
SUM_PATH = os.path.join(ROOT, "results", "S2_1", "s21_summary.json")
NS = (50000, 500000)
NMAX, NMIN = 500000, 50000
ARMS = ("A", "B", "Atau")
RUNG_KEYS = ["dN_k1", "dN_k3", "dN_k10", "dN_k30", "dN_k100", "dN_kq", "dC_k1", "dC_k3", "dC_k10", "dO", "dE",
             "dMST", "Gs", "Gs_max", "dAff", "dSOG_t1", "dSOG_t0p1", "dSOG_t0p01", "dSOGmst_t0p1", "dCKA", "dCKAd",
             "rawG", "dIso", "dPi"]
ORD = ["dN_k10", "dO", "dE", "dMST"]
SEED = 20261009
BOOT = 2000


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ Delta from raw hrhv2 row (prereg 2.4)
def my_deltas(row: dict) -> dict:
    g = lambda k: float(row[k]) if k in row and row[k] is not None else float("nan")
    d = {}
    for k in (1, 3, 10, 30, 100):
        d[f"dN_k{k}"] = 1.0 - g(f"s_N_k{k}")
    d["dN_kq"] = 1.0 - g("s_N_kq")
    for k in (1, 3, 10):
        d[f"dC_k{k}"] = g(f"comp_rand_k{k}")
    d["dO"] = g("Delta_O")
    d["dE"] = g("Delta_E")
    d["dMST"] = 1.0 - g("s_T")
    d["Gs"] = g("Gs_rms")
    d["Gs_max"] = g("Gs_max")
    d["dAff"] = 1.0 - g("rsa_pearson")
    d["dSOG_t1"] = 1.0 - g("sog_knn10_tau1")
    d["dSOG_t0p1"] = 1.0 - g("sog_knn10_tau0p1")
    d["dSOG_t0p01"] = 1.0 - g("sog_knn10_tau0p01")
    d["dSOGmst_t0p1"] = 1.0 - g("sog_mst_tau0p1")
    d["dCKA"] = 1.0 - g("cka_biased")
    d["dCKAd"] = 1.0 - g("cka_debiased")
    d["rawG"] = g("G_rms") if row.get("raw_G_applicable", False) else float("nan")
    # Iso: 1/2 (iso_rms_cf(A->B)/nu_B + iso_rms_cf(B->A)/nu_A); A->B means x = D_A, y = D_B (target B)
    d["dIso"] = 0.5 * (g("quot_AB_iso_rms_cf") / g("nu_B") + g("quot_BA_iso_rms_cf") / g("nu_A"))
    # warp share via RMS (independent of the *_expl path): SS_iso/SS_aff = (rms_iso/rms_aff)^2, same n_cf
    om = [1.0 - (g(f"quot_{t}_iso_rms_cf") / g(f"quot_{t}_affine_rms_cf")) ** 2 for t in ("AB", "BA")]
    d["omega"] = float(np.mean(om))
    d["omega_AB"], d["omega_BA"] = om
    # pooled-SS variant (sensitivity only)
    d["omega_pooled"] = 1.0 - (g("quot_AB_iso_rms_cf") ** 2 + g("quot_BA_iso_rms_cf") ** 2) / (
        g("quot_AB_affine_rms_cf") ** 2 + g("quot_BA_affine_rms_cf") ** 2)
    d["dPi"] = g("gl_projector_dist")
    return d


# ------------------------------------------------------------------ CP (one-sided 95%)
def cp(x, n, a=0.05):
    lo = float(Beta.ppf(a, x, n - x + 1)) if x > 0 else 0.0
    hi = float(Beta.ppf(1 - a, x + 1, n - x)) if x < n else 1.0
    return lo, hi


def cp_check(x, n, a=0.05):
    """Binomial-tail check of the CP bounds: P(X>=x|lo)=a and P(X<=x|hi)=a."""
    lo, hi = cp(x, n, a)
    e1 = abs(binom.sf(x - 1, n, lo) - a) if x > 0 else 0.0
    e2 = abs(binom.cdf(x, n, hi) - a) if x < n else 0.0
    return max(e1, e2)


def decide(x, n):
    lo, hi = cp(x, n)
    if lo > 0.9:
        return "reliable", lo, hi
    if hi < 0.8:
        return "unreliable", lo, hi
    return "inconclusive", lo, hi


# ------------------------------------------------------------------ run checks
def rid(arm, N, r, exp="S2.1"):
    return f"{exp}_{arm}_N{N}_r{r:03d}"


def run_status(run_id):
    d = os.path.join(ROOT, "runs", "S2_1_main", run_id)
    done = os.path.exists(os.path.join(d, ".done"))
    failed = os.path.exists(os.path.join(d, ".failed"))
    meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
    return {"done": done, "failed": failed, "diverged": bool(meta.get("diverged", True)),
            "finite": bool(meta.get("finite", False)), "N": meta.get("N"), "arm": meta.get("arm"),
            "rep": meta.get("rep"), "steps": meta.get("steps")}


# ------------------------------------------------------------------ pair sets (prereg 2.4)
def pair_sets():
    S = {}
    for N in NS:
        for arm in ARMS:
            S[f"within_{arm}_{N}"] = [(rid(arm, N, 2 * j), rid(arm, N, 2 * j + 1), False) for j in range(32)]
            S[f"init_within_{arm}_{N}"] = S[f"within_{arm}_{N}"]
        S[f"cross_A_B_{N}"] = [(rid("A", N, i), rid("B", N, i), False) for i in range(64)]
        S[f"cross_A_Atau_{N}"] = [(rid("A", N, i), rid("Atau", N, i), False) for i in range(64)]
        S[f"cross_Atau_B_{N}"] = [(rid("Atau", N, i), rid("B", N, i), False) for i in range(64)]
        S[f"bern_self_{N}"] = [(rid("A", N, i), rid("A", N, i), True) for i in range(64)]
        S[f"bern_peer_{N}"] = [(rid("A", N, 2 * j), rid("A", N, 2 * j + 1), True) for j in range(32)]
    fe = "S2.1-fixedepoch"
    S["fixedepoch_within_A"] = [(rid("A", 50000, 2 * j, fe), rid("A", 50000, 2 * j + 1, fe), False) for j in range(16)]
    S["fixedepoch_within_B"] = [(rid("B", 50000, 2 * j, fe), rid("B", 50000, 2 * j + 1, fe), False) for j in range(16)]
    S["fixedepoch_cross_A_B"] = [(rid("A", 50000, i, fe), rid("B", 50000, i, fe), False) for i in range(32)]
    return S


def tags_for(setname):
    if setname.startswith("init_"):
        return ["z_init"]
    if setname.startswith("bern_") or setname.startswith("fixedepoch"):
        return ["z"]
    return ["z", "h", "z_raw"]


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    log = {"inputs": {}, "checks": {}, "mismatches": []}
    eps_all = json.load(open(EPS_PATH))
    SUM = json.load(open(SUM_PATH))
    log["inputs"]["eps_sha256"] = sha256(EPS_PATH)
    log["inputs"]["prereg_sha256"] = sha256(os.path.join(ROOT, "prereg", "PREREG_stage2.md"))
    log["inputs"]["amend1_sha256"] = sha256(os.path.join(ROOT, "prereg", "PREREG_stage2_amendment1.md"))
    log["inputs"]["summary_sha256"] = sha256(SUM_PATH)
    log["inputs"]["report_sha256"] = sha256(os.path.join(ROOT, "results", "S2_1", "S2_1_report.md"))
    log["inputs"]["analyze_py_sha256"] = sha256(os.path.join(ROOT, "src", "analyze.py"))
    log["inputs"]["crit_py_sha256"] = sha256(os.path.join(ROOT, "src", "crit.py"))

    # --- runs
    runs_all = sorted(os.listdir(os.path.join(ROOT, "runs", "S2_1_main")))
    st = {r: run_status(r) for r in runs_all}
    bad = [r for r, s in st.items() if not s["done"] or s["failed"] or s["diverged"] or not s["finite"]]
    log["checks"]["n_runs"] = len(runs_all)
    log["checks"]["runs_not_ok"] = bad
    expected = [rid(a, N, r) for a in ARMS for N in NS for r in range(64)] + \
               [rid(a, 50000, r, "S2.1-fixedepoch") for a in ("A", "B") for r in range(32)]
    log["checks"]["expected_runs_missing"] = [r for r in expected if r not in st]
    log["checks"]["unexpected_runs"] = [r for r in runs_all if r not in set(expected)]
    log["checks"]["steps_values"] = sorted({str(s["steps"]) for s in st.values()})

    # --- per-pair deltas
    S = pair_sets()
    vals, pairs_used, delta_check = {}, {}, {"max_abs_diff_vs_cached_deltas": 0.0, "n": 0, "worst": None}
    missing = []
    for sname, plist in S.items():
        for tag in tags_for(sname):
            key = f"{sname}|{tag}"
            rows = []
            for a, b, bern in plist:
                fn = f"{a}__{b}{'__bern' if bern else ''}.json"
                p = os.path.join(LAD, tag, fn)
                if not os.path.exists(p):
                    missing.append(f"{tag}/{fn}")
                    rows.append({k: float("inf") for k in RUNG_KEYS + ["omega", "omega_AB", "omega_BA", "omega_pooled"]})
                    continue
                J = json.load(open(p))
                assert J["A"] == a and J["B"] == b and J["tag"] == tag, (p, J["A"], J["B"], J["tag"])
                assert (J.get("transform") == "bern") == bern, p
                row = J["row"]
                if bern:
                    assert row["pre_B"] == "bern" and row["pre_A"] == "l2", p
                else:
                    want = "none" if tag == "z_raw" else "l2"
                    assert row["pre_A"] == want and row["pre_B"] == want, p
                assert row["m"] == 2048, p
                d = my_deltas(row)
                fail = bool(J.get("failed_run")) or (a in st and (st[a]["diverged"] or not st[a]["finite"])) or \
                    (b in st and (st[b]["diverged"] or not st[b]["finite"]))
                if fail:
                    d = {k: float("inf") for k in d}
                # cross-check against the cached 'deltas' block written by crit.py
                for k in RUNG_KEYS + ["omega"]:
                    c = J["deltas"].get(k, float("nan"))
                    mine = d[k]
                    if (isinstance(c, float) and math.isnan(c)) and math.isnan(mine):
                        continue
                    diff = abs(mine - c)
                    delta_check["n"] += 1
                    if diff > delta_check["max_abs_diff_vs_cached_deltas"]:
                        delta_check["max_abs_diff_vs_cached_deltas"] = diff
                        delta_check["worst"] = [fn, tag, k, mine, c]
                rows.append(d)
            vals[key] = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
            pairs_used[key] = len(rows)
    log["checks"]["missing_ladder_files"] = missing
    log["checks"]["delta_check"] = delta_check

    # --- CP threshold table (verifies prereg statement)
    thr = {}
    for n in (16, 32, 64, 96):
        rel = [x for x in range(n + 1) if decide(x, n)[0] == "reliable"]
        unr = [x for x in range(n + 1) if decide(x, n)[0] == "unreliable"]
        thr[n] = {"reliable_min_x": min(rel) if rel else None, "unreliable_max_x": max(unr) if unr else None,
                  "max_cp_tail_err": max(cp_check(x, n) for x in range(n + 1))}
    log["cp_thresholds"] = thr

    # --- decision tables
    def eps_for(tag, k, q90=False):
        src = eps_all["eps_q90" if q90 else "eps"]
        t = "z" if tag == "z_init" else tag
        return float(src[t][k])

    tables, tables_q90 = {}, {}
    for key, V in vals.items():
        tag = key.split("|")[1]
        tables[key], tables_q90[key] = {}, {}
        for k in RUNG_KEYS:
            v = V[k]
            for q90, T in ((False, tables), (True, tables_q90)):
                e = eps_for(tag, k, q90)
                n = int(np.sum(~np.isnan(v)))
                if n == 0:
                    T[key][k] = {"n": 0}
                    continue
                x = int(np.sum(v[~np.isnan(v)] <= e))
                dec, lo, hi = decide(x, n)
                T[key][k] = {"n": n, "x": x, "p_hat": x / n, "cp_lo": lo, "cp_hi": hi, "decision": dec, "eps": e,
                             "mean": float(np.mean(v[~np.isnan(v)])), "median": float(np.median(v[~np.isnan(v)])),
                             "min": float(np.min(v[~np.isnan(v)])), "max": float(np.max(v[~np.isnan(v)]))}

    # --- compare tables with the summary
    def cmp(path, mine, theirs, tol=1e-9):
        if theirs is None:
            log["mismatches"].append({"where": path, "mine": mine, "theirs": None, "note": "missing in summary"})
            return
        if isinstance(mine, str) or isinstance(theirs, str) or isinstance(mine, bool) or isinstance(theirs, bool):
            if mine != theirs:
                log["mismatches"].append({"where": path, "mine": mine, "theirs": theirs})
            return
        if mine is None or theirs is None:
            if mine != theirs:
                log["mismatches"].append({"where": path, "mine": mine, "theirs": theirs})
            return
        m_, t_ = float(mine), float(theirs)
        if math.isnan(m_) and math.isnan(t_):
            return
        if not (abs(m_ - t_) <= tol * max(1.0, abs(t_))):
            log["mismatches"].append({"where": path, "mine": m_, "theirs": t_})

    n_cmp = 0
    for key, T in tables.items():
        ST = SUM["tables"].get(key)
        if ST is None:
            log["mismatches"].append({"where": f"tables/{key}", "note": "set missing in summary"})
            continue
        for k, c in T.items():
            s = ST.get(k)
            if s is None:
                log["mismatches"].append({"where": f"tables/{key}/{k}", "note": "rung missing in summary"})
                continue
            for f in ("n", "x", "p_hat", "cp_lo", "cp_hi", "decision", "eps", "mean", "median"):
                if f in c:
                    cmp(f"tables/{key}/{k}/{f}", c[f], s.get(f))
                    n_cmp += 1
    for key in SUM["tables"]:
        if key not in tables:
            log["mismatches"].append({"where": f"tables/{key}", "note": "set in summary but not in recompute"})
    # p_hat_q90
    for key, T in tables_q90.items():
        ST = SUM.get("p_hat_q90", {}).get(key)
        if ST is None:
            continue
        for k, c in T.items():
            if "p_hat" in c and k in ST:
                cmp(f"p_hat_q90/{key}/{k}", c["p_hat"], ST[k]); n_cmp += 1
    # per-pair values
    vmax = 0.0
    for key, V in vals.items():
        SV = SUM.get("values", {}).get(key)
        if SV is None:
            continue
        for k in RUNG_KEYS:
            if k in SV:
                a = V[k]; b = np.array(SV[k], dtype=float)
                if a.shape != b.shape:
                    log["mismatches"].append({"where": f"values/{key}/{k}", "note": f"shape {a.shape} vs {b.shape}"})
                    continue
                both = ~(np.isnan(a) & np.isnan(b))
                if np.any(np.isnan(a[both]) != np.isnan(b[both])):
                    log["mismatches"].append({"where": f"values/{key}/{k}", "note": "nan pattern differs"})
                    continue
                ok = both & ~np.isnan(a)
                if ok.any():
                    vmax = max(vmax, float(np.max(np.abs(a[ok] - b[ok]))))
    log["checks"]["values_max_abs_diff_vs_summary"] = vmax
    log["checks"]["n_table_fields_compared"] = n_cmp

    # ------------------------------------------------------------------ endpoints (z only)
    Z = lambda s: f"{s}|z"
    P = lambda s, k: tables[Z(s)][k]["p_hat"]
    ep = {}

    # (i)
    ceil_i, gap_i, pcross_i, pcross_min = {}, {}, {}, {}
    for k in ORD:
        c = math.sqrt(P(f"within_A_{NMAX}", k) * P(f"within_B_{NMAX}", k))
        ceil_i[k] = c
        pcross_i[k] = P(f"cross_A_B_{NMAX}", k)
        pcross_min[k] = P(f"cross_A_B_{NMIN}", k)
        gap_i[k] = c - pcross_i[k]
    impr = {}
    for B_ in (BOOT, 20000):
        r = np.random.default_rng(SEED + B_)
        res = {}
        for k in ORD:
            v5 = vals[Z(f"cross_A_B_{NMAX}")][k]
            v0 = vals[Z(f"cross_A_B_{NMIN}")][k]
            diff = float(v5.mean() - v0.mean())
            i5 = r.integers(0, v5.size, size=(B_, v5.size))
            i0 = r.integers(0, v0.size, size=(B_, v0.size))
            bd = v5[i5].mean(axis=1) - v0[i0].mean(axis=1)
            cnt = int(np.sum(bd >= 0))
            res[k] = {"diff": diff, "ci95": [float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975))],
                      "p_plus1": (1 + cnt) / (B_ + 1), "p_raw": cnt / B_,
                      "z_normal_approx_p": float(0.5 * math.erfc((-diff / bd.std(ddof=1)) / math.sqrt(2))) if bd.std() > 0 else None}
        # Holm over 4 rungs at 0.05 (step-down)
        order = sorted(ORD, key=lambda k: res[k]["p_plus1"])
        stop = False
        for i, k in enumerate(order):
            th = 0.05 / (len(ORD) - i)
            rej = (not stop) and res[k]["p_plus1"] <= th
            if not rej:
                stop = True
            res[k]["holm_threshold"] = th
            res[k]["holm_reject"] = rej
        impr[str(B_)] = res
    i_pass = all(gap_i[k] <= 0.1 for k in ORD) and all(impr[str(BOOT)][k]["holm_reject"] for k in ORD)
    ep["i"] = {"ceiling_Nmax": ceil_i, "p_cross_Nmax": pcross_i, "p_cross_Nmin": pcross_min, "gap": gap_i,
               "improvement": impr, "pass": i_pass}

    # (ii)
    ii = {}
    for q in ("Gs", "dAff", "dSOG_t0p1"):
        cd = {N: tables[Z(f"cross_A_B_{N}")][q]["decision"] for N in NS}
        wp = {a: P(f"within_{a}_{NMAX}", q) for a in ("A", "B")}
        k3ok = all(v >= 0.8 for v in wp.values())
        unr = all(cd[N] == "unreliable" for N in NS)
        status = ("pass" if unr else "fail") if k3ok else "inconclusive (K3)"
        ii[q] = {"cross_decision": cd, "within_p_Nmax": wp, "K3_guard_ok": k3ok, "status": status}
    ep["ii"] = {"items": ii, "pass": all(v["status"] == "pass" for v in ii.values()),
                "any_K3": any(v["status"].startswith("inconclusive") for v in ii.values())}

    # (iii)
    cross_iso = float(np.mean(vals[Z(f"cross_A_B_{NMAX}")]["dIso"]))
    within_iso = float(np.mean(np.concatenate([vals[Z(f"within_A_{NMAX}")]["dIso"], vals[Z(f"within_B_{NMAX}")]["dIso"]])))
    iii_by_N = {}
    for N in NS:
        c_ = float(np.mean(vals[Z(f"cross_A_B_{N}")]["dIso"]))
        w_ = float(np.mean(np.concatenate([vals[Z(f"within_A_{N}")]["dIso"], vals[Z(f"within_B_{N}")]["dIso"]])))
        iii_by_N[N] = {"cross": c_, "within": w_, "ratio": c_ / w_}
    ep["iii"] = {"mean_dIso_cross": cross_iso, "mean_dIso_within_AB": within_iso, "ratio": cross_iso / within_iso,
                 "pass": cross_iso <= 1.5 * within_iso, "by_N": iii_by_N}

    # (iv)
    iv = {}
    for N in NS:
        iv[N] = {}
        for s in ("cross_A_B", "within_A", "within_B", "within_Atau", "cross_A_Atau", "cross_Atau_B", "bern_self", "bern_peer"):
            v = vals[Z(f"{s}_{N}")]
            iv[N][s] = {"median_omega": float(np.median(v["omega"])),
                        "median_omega_pooled": float(np.median(v["omega_pooled"])),
                        "q25": float(np.quantile(v["omega"], 0.25)), "q75": float(np.quantile(v["omega"], 0.75)),
                        "max": float(np.max(v["omega"])), "min": float(np.min(v["omega"])),
                        "median_omega_AB": float(np.median(v["omega_AB"])), "median_omega_BA": float(np.median(v["omega_BA"]))}
    iv_pass = (iv[NMAX]["cross_A_B"]["median_omega"] >= 0.5 and iv[NMAX]["within_A"]["median_omega"] <= 0.2
               and iv[NMAX]["within_B"]["median_omega"] <= 0.2)
    ep["iv"] = {"by_N": iv, "pass": iv_pass}

    # (v)
    v_items = {}
    for q in ("Gs", "dCKA", "dSOG_t0p1"):
        c = math.sqrt(P(f"within_A_{NMAX}", q) * P(f"within_Atau_{NMAX}", q))
        p_ = P(f"cross_A_Atau_{NMAX}", q)
        v_items[q] = {"p_cross": p_, "ceiling": c, "pass": p_ >= c - 0.1}
    rg_cd = {N: tables[Z(f"cross_A_Atau_{N}")]["rawG"]["decision"] for N in NS}
    rg_w = {a: P(f"within_{a}_{NMAX}", "rawG") for a in ("A", "Atau")}
    v_items["rawG"] = {"cross_decision": rg_cd, "within_p": rg_w,
                       "pass": rg_cd[NMAX] == "unreliable" and all(x >= 0.8 for x in rg_w.values())}
    ep["v"] = {"items": v_items, "pass": all(x["pass"] for x in v_items.values())}

    # verdict
    falsified = (iv[NMAX]["cross_A_B"]["median_omega"] < 0.2) or any(gap_i[k] > 0.1 for k in ORD)
    allpass = ep["i"]["pass"] and ep["ii"]["pass"] and ep["iii"]["pass"] and ep["iv"]["pass"] and ep["v"]["pass"]
    if allpass and not ep["ii"]["any_K3"]:
        verdict = "confirmed"
    elif falsified:
        verdict = "falsified"
    else:
        verdict = "partial"
    ep["falsifier_triggers"] = {"median_omega_AB_Nmax_lt_0.2": iv[NMAX]["cross_A_B"]["median_omega"] < 0.2,
                                "ordinal_gaps_gt_0.1": [k for k in ORD if gap_i[k] > 0.1]}
    ep["verdict"] = verdict

    # CKA cross minus within (bootstrap: cross, within-A, within-B resampled independently)
    cka = {}
    for N in NS:
        c = vals[Z(f"cross_A_B_{N}")]["dCKA"]
        wa = vals[Z(f"within_A_{N}")]["dCKA"]; wb = vals[Z(f"within_B_{N}")]["dCKA"]
        diff = float(c.mean() - np.concatenate([wa, wb]).mean())
        r = np.random.default_rng(SEED + N)
        bc = c[r.integers(0, c.size, (BOOT, c.size))].mean(1)
        bw = np.concatenate([wa[r.integers(0, wa.size, (BOOT, wa.size))], wb[r.integers(0, wb.size, (BOOT, wb.size))]], 1).mean(1)
        bd = bc - bw
        # alternative: pooled within resampled as one cell of 64
        w = np.concatenate([wa, wb])
        bw2 = w[r.integers(0, w.size, (BOOT, w.size))].mean(1)
        bd2 = bc - bw2
        cka[N] = {"diff": diff, "ci95": [float(np.quantile(bd, .025)), float(np.quantile(bd, .975))],
                  "ci95_pooled_within_cell": [float(np.quantile(bd2, .025)), float(np.quantile(bd2, .975))]}
    ep["cka_cross_minus_within"] = cka

    # how far from passing (exploratory): fraction of cross pairs within eps x c
    far = {}
    for k in ORD + ["Gs", "dIso", "dCKA"]:
        e = eps_for("z", k)
        v = vals[Z(f"cross_A_B_{NMAX}")][k]
        far[k] = {"eps": e, "cross_min": float(v.min()), "cross_median": float(np.median(v)),
                  "eps_multiplier_needed_for_62of64": float(np.sort(v)[61] / e) if e > 0 else None,
                  "within_A_max": float(vals[Z(f"within_A_{NMAX}")][k].max()),
                  "within_B_max": float(vals[Z(f"within_B_{NMAX}")][k].max())}
    ep["distance_to_pass_exploratory"] = far

    # --- compare endpoints with summary
    SE = SUM["endpoints"]
    for k in ORD:
        cmp(f"endpoints/i/ceiling_Nmax/{k}", ceil_i[k], SE["i"]["ceiling_Nmax"][k])
        cmp(f"endpoints/i/gap/{k}", gap_i[k], SE["i"]["gap_to_ceiling_Nmax"][k])
        cmp(f"endpoints/i/p_cross_Nmax/{k}", pcross_i[k], SE["i"]["p_cross_Nmax"][k])
        cmp(f"endpoints/i/p_cross_Nmin/{k}", pcross_min[k], SE["i"]["p_cross_Nmin"][k])
        cmp(f"endpoints/i/diff/{k}", impr[str(BOOT)][k]["diff"], SE["i"]["improvement_500k_minus_50k"][k]["diff"])
        cmp(f"endpoints/i/holm_reject/{k}", impr[str(BOOT)][k]["holm_reject"], SE["i"]["holm"][k]["reject"])
        cmp(f"endpoints/i/holm_threshold/{k}", impr[str(BOOT)][k]["holm_threshold"], SE["i"]["holm"][k]["holm_threshold"])
    cmp("endpoints/i/pass", ep["i"]["pass"], SE["i"]["pass"])
    for q in ii:
        for N in NS:
            cmp(f"endpoints/ii/{q}/cross_decision/{N}", ii[q]["cross_decision"][N], SE["ii"]["items"][q]["cross_decision"][str(N)])
        for a in ("A", "B"):
            cmp(f"endpoints/ii/{q}/within_p/{a}", ii[q]["within_p_Nmax"][a], SE["ii"]["items"][q]["within_p_Nmax"][a])
        cmp(f"endpoints/ii/{q}/status", ii[q]["status"], SE["ii"]["items"][q]["status"])
    cmp("endpoints/ii/pass", ep["ii"]["pass"], SE["ii"]["pass"])
    cmp("endpoints/ii/any_K3", ep["ii"]["any_K3"], SE["ii"]["any_K3"])
    cmp("endpoints/iii/mean_dIso_cross", cross_iso, SE["iii"]["mean_dIso_cross"])
    cmp("endpoints/iii/mean_dIso_within_AB", within_iso, SE["iii"]["mean_dIso_within_AB"])
    cmp("endpoints/iii/ratio", ep["iii"]["ratio"], SE["iii"]["ratio"])
    cmp("endpoints/iii/pass", ep["iii"]["pass"], SE["iii"]["pass"])
    names = {"cross_A_B": "omega_cross_AB", "within_A": "omega_within_A", "within_B": "omega_within_B",
             "within_Atau": "omega_within_Atau", "cross_A_Atau": "omega_cross_A_Atau", "cross_Atau_B": "omega_cross_Atau_B"}
    for N in NS:
        for s, nm in names.items():
            cmp(f"endpoints/iv/{N}/{nm}", iv[N][s]["median_omega"], SE["iv"][str(N)][nm], tol=1e-8)
    cmp("endpoints/iv/pass", ep["iv"]["pass"], SE["iv"]["pass"])
    for q in ("Gs", "dCKA", "dSOG_t0p1"):
        for f in ("p_cross", "ceiling", "pass"):
            cmp(f"endpoints/v/{q}/{f}", v_items[q][f], SE["v"]["items"][q][f])
    cmp("endpoints/v/rawG/cross_decision", rg_cd[NMAX], SE["v"]["items"]["rawG"]["cross_decision"])
    for a in ("A", "Atau"):
        cmp(f"endpoints/v/rawG/within_p/{a}", rg_w[a], SE["v"]["items"]["rawG"]["within_p"][a])
    cmp("endpoints/v/rawG/pass", v_items["rawG"]["pass"], SE["v"]["items"]["rawG"]["pass"])
    cmp("endpoints/v/pass", ep["v"]["pass"], SE["v"]["pass"])
    cmp("verdict", verdict, SUM["verdict"])
    for N in NS:
        cmp(f"cka_cross_minus_within/{N}/diff", cka[N]["diff"], SUM["cka_cross_minus_within"][str(N)]["diff"])
    # bootstrap-dependent quantities: compare with tolerance (different RNG)
    boot_cmp = {}
    for k in ORD:
        boot_cmp[k] = {"mine_p": impr[str(BOOT)][k]["p_plus1"], "mine_p_20000": impr["20000"][k]["p_plus1"],
                       "theirs_p": SE["i"]["improvement_500k_minus_50k"][k]["p_one_sided_diff_ge_0"],
                       "mine_ci": impr[str(BOOT)][k]["ci95"], "theirs_ci": SE["i"]["improvement_500k_minus_50k"][k]["ci95"]}
    for N in NS:
        boot_cmp[f"cka_{N}"] = {"mine_ci": cka[N]["ci95"], "mine_ci_pooled": cka[N]["ci95_pooled_within_cell"],
                                "theirs_ci": SUM["cka_cross_minus_within"][str(N)]["ci95"]}
    ep["bootstrap_comparison"] = boot_cmp

    out = {"log": log, "endpoints": ep, "tables": tables, "tables_q90": {k: {r: v.get("p_hat") for r, v in T.items()} for k, T in tables_q90.items()},
           "pairs_used": pairs_used}

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
            return str(o)
        return o
    json.dump(conv(out), open(os.path.join(OUT, "recompute_summary.json"), "w"), indent=1)
    # per-pair z values for downstream use
    json.dump(conv({k: {r: v.tolist() for r, v in V.items()} for k, V in vals.items() if k.endswith("|z")}),
              open(os.path.join(OUT, "recompute_pair_values_z.json"), "w"))
    print(json.dumps(conv({"checks": log["checks"], "n_mismatch": len(log["mismatches"]),
                           "mismatches_head": log["mismatches"][:30], "verdict": verdict,
                           "cp_thresholds": thr}), indent=1))


if __name__ == "__main__":
    main()
