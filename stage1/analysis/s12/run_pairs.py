"""S1.2 per-pair computation: full hrhv2 ladder + extras + K=500 permutation
calibration, for one (LM, ViT, depth) and both preprocessings.

A = LM (language model), B = ViT; the permutation acts on the rows of B
(SPEC 5). Outputs per task (results/S1_2/<run>/pairs/):
  <lm>__<vit>__r<r>.json   ladder + extras + calibration summaries
  <lm>__<vit>__r<r>.npz    the K null values of every calibrated score
  <lm>__<vit>__r<r>.done   marker (tasks with a marker are skipped on re-run)

Usage:
  python run_pairs.py --run main [--jobs 12] [--only-lm STEM] [--limit N]
  python run_pairs.py --run sidecar_rule   (exploratory: ViT layer by the sidecar rule, r=0.5)
  python run_pairs.py --run pool2047        (PE2 power-gate fallback: S + S_sel, m=2047, PE2 keys only)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.expanduser("~/hrh_v2_stage1_20261008"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import common as C  # noqa: E402
from hrhv2 import structural as st, geometric as geo, calibrate as cal, util  # noqa: E402
from hrhv2 import predict as pr  # noqa: E402
from hrhv2.ladder import ladder  # noqa: E402
from hrhv2.prep import Geom, center_gram  # noqa: E402

KS = (1, 3, 10, 30, 100)
KMAX = 101


# ----------------------------------------------------------------------------
# extras computed once per (pair, pre)
# ----------------------------------------------------------------------------

def mst_readings(TA, TB, FA, FB, m):
    """Forest/bridge split under four readings of 'whose 1-NN forest' (D6)."""
    S = np.intersect1d(TA, TB, assume_unique=True)
    inFA = np.isin(S, FA, assume_unique=True)
    inFB = np.isin(S, FB, assume_unique=True)
    nS = S.size
    brA = np.setdiff1d(TA, FA, assume_unique=True)
    brB = np.setdiff1d(TB, FB, assume_unique=True)
    f = lambda x: float(x) / nS if nS else float("nan")  # noqa: E731
    return {
        "n_shared_edges": int(nS),
        "forest_share_shared_A": f(inFA.sum()),
        "forest_share_shared_B": f(inFB.sum()),
        "forest_share_shared_union": f((inFA | inFB).sum()),
        "forest_share_shared_inter": f((inFA & inFB).sum()),
        "n_bridges_A": int(brA.size), "n_bridges_B": int(brB.size),
        "n_shared_bridges_A": int(np.intersect1d(brA, TB, assume_unique=True).size),
        "n_shared_bridges_B": int(np.intersect1d(brB, TA, assume_unique=True).size),
        "n_shared_bridges_pure": int((~inFA & ~inFB).sum()),
        "n_forest_A": int(FA.size), "n_forest_B": int(FB.size),
        "n_shared_forest_edges_both": int(np.intersect1d(FA, FB, assume_unique=True).size),
    }


def _offdiag_rows(D):
    m = D.shape[0]
    mask = ~np.eye(m, dtype=bool)
    return D[mask].reshape(m, m - 1)


def row_linf_iso(RX, RY):
    """e_i = min over increasing f of max_j |RX_ij - f(RY_ij)|
    = 0.5 * max over (a, b) with RY_a <= RY_b of (RX_a - RX_b)_+ (row-wise)."""
    order = np.argsort(RY, axis=1, kind="stable")
    y = np.take_along_axis(RX, order, axis=1)
    runmax = np.maximum.accumulate(y, axis=1)
    return 0.5 * np.max(runmax - y, axis=1)


def eq6_certificate(DA, DB, vA, vB, ks=(10, 1)):
    """v2 eq. (6) neighbour-gap certificate: anchor i is certified at k when
    gamma_i = D_(k+1) - D_(k) of the reference row exceeds 2 e_i, e_i a row-sup
    discrepancy. Variants: raw e_i = max_j |D_A - D_B| with A's gap (v2 as
    written) and with max(gamma_A, gamma_B); scale-normalised D/nu; row-isotonic
    quotient e_i^iso (B warped to A, A's gap; and A warped to B, B's gap)."""
    out = {}
    RA, RB = _offdiag_rows(DA), _offdiag_rows(DB)
    e_raw = np.max(np.abs(RA - RB), axis=1)
    nuA, nuB = geo.mean_offdiag(DA), geo.mean_offdiag(DB)
    e_sc = np.max(np.abs(RA / nuA - RB / nuB), axis=1)
    e_isoAB = row_linf_iso(RA, RB)
    e_isoBA = row_linf_iso(RB, RA)
    for k in ks:
        gA = vA[:, k] - vA[:, k - 1]
        gB = vB[:, k] - vB[:, k - 1]
        t = f"k{k}"
        out[f"eq6_cert_raw_A_{t}"] = float(np.mean(gA > 2 * e_raw))
        out[f"eq6_cert_raw_sym_{t}"] = float(np.mean(np.maximum(gA, gB) > 2 * e_raw))
        out[f"eq6_cert_scaled_sym_{t}"] = float(np.mean(np.maximum(gA / nuA, gB / nuB) > 2 * e_sc))
        out[f"eq6_cert_iso_A_{t}"] = float(np.mean(gA > 2 * e_isoAB))
        out[f"eq6_cert_iso_B_{t}"] = float(np.mean(gB > 2 * e_isoBA))
        out[f"eq6_cert_any_{t}"] = float(np.mean((np.maximum(gA, gB) > 2 * e_raw) | (gA > 2 * e_isoAB)
                                                 | (gB > 2 * e_isoBA)
                                                 | (np.maximum(gA / nuA, gB / nuB) > 2 * e_sc)))
        out[f"eq6_median_gap_over_2e_raw_A_{t}"] = float(np.median(gA / (2 * e_raw)))
        out[f"eq6_median_gap_over_2e_iso_A_{t}"] = float(np.median(gA / (2 * e_isoAB)))
        out[f"eq6_max_gap_over_2e_iso_A_{t}"] = float(np.max(gA / (2 * e_isoAB)))
    out["eq6_median_e_raw"] = float(np.median(e_raw))
    out["eq6_median_e_iso_AB"] = float(np.median(e_isoAB))
    return out


def dispersion_law(DA, DB, vA, k=10):
    """P2 ingredients with A as the reference (descriptive, in-sample, all
    anchors): p-hat = Levina-Bickel MLE (MacKay-Ghahramani average) at scale k
    on A's rows; delta-hat_D = mean_i MAD_iso(D_k,i)/D_k,i of the row-isotonic
    residual of B given A in A-units (hrhv2.predict.fit_iso_dispersion with the
    fit mask = all items); P2 law 1 - s_N ~ (p/4) delta_D."""
    m = DA.shape[0]
    lb = pr.lb_dimension(vA[:, :k], k)
    disp = pr.fit_iso_dispersion(DA, DB, np.ones(m, dtype=bool))
    Dk = vA[:, k - 1]
    ok = Dk > 0
    dD = disp.scale(Dk[ok]) / Dk[ok]
    delta = float(np.mean(dD))
    return {"p_hat_mg": lb["mg"], "p_hat_lb": lb["lb"], "delta_D_mean": delta,
            "P2_one_minus_sN": 0.25 * lb["mg"] * delta, "n_anchor_used": int(ok.sum())}


# ----------------------------------------------------------------------------
# permutation calibration (fast exact path + spot check against recomputation)
# ----------------------------------------------------------------------------

class Calib:
    """Scores for B permuted by pi, from precomputed objects.

    kNN tables and the MST of permuted B are relabellings of B's (exact when
    all off-diagonal D_B values are distinct, checked per config; otherwise
    every permutation is recomputed from scratch). CKA uses the permuted
    centred Gram (H P K P' H = P H K H P')."""

    def __init__(self, A, B, aux, eqA=None, eqB=None, pe2_only=False):
        self.m = m = A.m
        self.pe2_only = pe2_only
        nA, nB = aux["nA"], aux["nB"]
        self.nB = nB
        rows = np.arange(m)[:, None]
        self.rows = rows
        self.MA = {}
        for k in KS:
            if k <= nA.shape[1]:
                M = np.zeros((m, m), dtype=bool)
                M[rows, nA[:, :k]] = True
                self.MA[k] = M
        TA, TB, FA, FB = aux["TA"], aux["TB"], aux["FA"], aux["FB"]
        self.TAmat = np.zeros((m, m), dtype=bool)
        u, v = TA // m, TA % m
        self.TAmat[u, v] = self.TAmat[v, u] = True
        self.FAmat = np.zeros((m, m), dtype=bool)
        u, v = FA // m, FA % m
        self.FAmat[u, v] = self.FAmat[v, u] = True
        self.BRAmat = self.TAmat & ~self.FAmat
        self.nbrA = int(self.BRAmat.sum() // 2)
        self.TBu, self.TBv = TB // m, TB % m
        self.isforestB = np.isin(TB, FB, assume_unique=True)
        self.nbrB = int((~self.isforestB).sum())
        self.exact_relabel = bool(np.unique(util.upper_values(B.D)).size == util.n_pairs(m))
        self.B = B
        self.nA = nA
        if not pe2_only:
            self.KcA = center_gram(A.K)
            self.KcB = center_gram(B.K)
            self.haa = float(np.vdot(self.KcA, self.KcA))
            self.hbb = float(np.vdot(self.KcB, self.KcB))
            self.huaa = geo.hsic_unbiased(self.KcA, self.KcA)
            self.hubb = geo.hsic_unbiased(self.KcB, self.KcB)
            self.eq = eqA is not None
            if self.eq:
                self.ZA = center_gram(eqA.K)
                self.ZB = center_gram(eqB.K)
                self.zaa = float(np.vdot(self.ZA, self.ZA))
                self.zbb = float(np.vdot(self.ZB, self.ZB))
                self.zuaa = geo.hsic_unbiased(self.ZA, self.ZA)
                self.zubb = geo.hsic_unbiased(self.ZB, self.ZB)

    def _tables(self, pi):
        m = self.m
        inv = np.empty(m, dtype=np.int64)
        inv[pi] = np.arange(m)
        if self.exact_relabel:
            bp = inv[self.nB[pi, :]]
            tu, tv = inv[self.TBu], inv[self.TBv]
            isforestB = self.isforestB
        else:  # recompute from the permuted matrix
            Dp = np.ascontiguousarray(self.B.D[np.ix_(pi, pi)])
            bp, _ = st.knn_table(Dp, self.nB.shape[1])
            T = st.mst_prim(Dp)
            F = st.nn1_forest(bp[:, 0])
            tu, tv = T // m, T % m
            isforestB = np.isin(T, F, assume_unique=True)
        return bp, tu, tv, isforestB

    def scores(self, pi):
        m = self.m
        bp, tu, tv, isforestB = self._tables(pi)
        out = {}
        for k, M in self.MA.items():
            out[f"s_N_k{k}"] = float(M[self.rows, bp[:, :k]].sum()) / (m * k)
        shared = self.TAmat[tu, tv]
        inFA = self.FAmat[tu, tv]
        nS = int(shared.sum())
        out["s_T"] = nS / (m - 1)
        out["s_T_bridge"] = float(self.BRAmat[tu, tv].sum()) / self.nbrA if self.nbrA else float("nan")
        out["s_T_bridge_B"] = float((shared & ~isforestB).sum()) / self.nbrB if self.nbrB else float("nan")
        out["s_T_bridge_mean"] = 0.5 * (out["s_T_bridge"] + out["s_T_bridge_B"])
        out["n_shared_bridges_pure"] = float((shared & ~inFA & ~isforestB).sum())
        if not self.pe2_only:
            KcBp = self.KcB[np.ix_(pi, pi)]
            out["cka_biased"] = float(np.vdot(self.KcA, KcBp)) / np.sqrt(self.haa * self.hbb)
            out["cka_debiased"] = geo.hsic_unbiased(self.KcA, KcBp) / np.sqrt(self.huaa * self.hubb)
            if self.eq:
                ZBp = self.ZB[np.ix_(pi, pi)]
                out["cka_eq_biased"] = float(np.vdot(self.ZA, ZBp)) / np.sqrt(self.zaa * self.zbb)
                out["cka_eq_debiased"] = geo.hsic_unbiased(self.ZA, ZBp) / np.sqrt(self.zuaa * self.zubb)
        return out


def recompute_scores(A, B, pi, eqA=None, eqB=None):
    """Brute recomputation on the permuted Geom (spot check of the fast path)."""
    Bp = B.permuted(pi)
    m = A.m
    nA, _ = st.knn_table(A.D, KMAX)
    nB, _ = st.knn_table(Bp.D, KMAX)
    out = {f"s_N_k{k}": st.knn_overlap(nA, nB, k) for k in KS}
    TA, TB = st.mst_prim(A.D), st.mst_prim(Bp.D)
    FA, FB = st.nn1_forest(nA[:, 0]), st.nn1_forest(nB[:, 0])
    c = st.mst_compare(TA, TB, FA, FB)
    out["s_T"], out["s_T_bridge"], out["s_T_bridge_B"] = c["s_T"], c["s_T_bridge"], c["s_T_bridge_B"]
    cc = geo.cka_pair(A, Bp)
    out["cka_biased"], out["cka_debiased"] = cc["cka_biased"], cc["cka_debiased"]
    if eqA is not None:
        ce = geo.cka_pair(eqA, eqB.permuted(pi))
        out["cka_eq_biased"], out["cka_eq_debiased"] = ce["cka_biased"], ce["cka_debiased"]
    return out


# ----------------------------------------------------------------------------
# one task
# ----------------------------------------------------------------------------

def run_task(lm, vit, r, run, K=C.K_PERM, n_check=2):
    t0 = time.time()
    rule = "sidecar" if run == "sidecar_rule" else "primary"
    pool = "S+Ssel" if run == "pool2047" else None
    pe2_only = run == "pool2047"
    pres = (C.PRIMARY_PRE,) if pe2_only else C.PRES
    key = f"S1.2|{run}|{lm}|{vit}|r={r}"
    res = {"lm": lm, "vit": vit, "r": r, "run": run, "perm_key": key, "K": K, "per_pre": {}}
    nulls = {}
    for pre in pres:
        T = {}
        ts = time.time()
        XA, mA = C.features(lm, r, pre, rule="primary", pool=pool)
        XB, mB = C.features(vit, r, pre, rule=rule, pool=pool)
        A, B = C.geom(XA, pre), C.geom(XB, pre)
        m = A.m
        T["features"] = time.time() - ts
        ts = time.time()
        if pe2_only:
            row, aux = ladder(A, B, n_jobs=1, with_gl=False, with_quotients=False, with_row_kendall=False,
                              return_aux=True, k_grid=[1, 3, 10], comp_ks=(1,), sog_k=10)
        else:
            row, aux = ladder(A, B, n_jobs=1, with_gl=False, return_aux=True)
        T["ladder"] = time.time() - ts
        rec = {"meta_A": mA, "meta_B": mB, "ladder": row}
        rec["mst_readings"] = mst_readings(aux["TA"], aux["TB"], aux["FA"], aux["FB"], m)
        ts = time.time()
        if not pe2_only:
            rec["eq6"] = eq6_certificate(A.D, B.D, aux["vA"], aux["vB"])
            try:
                rec["law_A_ref"] = dispersion_law(A.D, B.D, aux["vA"])
                rec["law_B_ref"] = dispersion_law(B.D, A.D, aux["vB"])
            except Exception as e:  # recorded, not fatal
                rec["law_error"] = repr(e)
        T["extras"] = time.time() - ts
        eqA = eqB = None
        if not pe2_only:
            ZA = np.load(os.path.join(C.CACHE, "pc128", f"{lm}__r{r}__{pre}.npy"))
            if rule == "primary":
                ZB = np.load(os.path.join(C.CACHE, "pc128", f"{vit}__r{r}__{pre}.npy"))
            else:  # sidecar-rule ViT layer: same PE3 recipe (top-128 PCs of S_sel at that layer)
                Xs, _ = C.features(vit, r, pre, ds="wit4096_sel", rule=rule)
                mu = Xs.mean(axis=0)
                Vt = np.linalg.svd(Xs - mu, full_matrices=False)[2]
                ZB = (XB - mu) @ Vt[:C.PC_DIM].T
            eqA, eqB = Geom.from_features(ZA, "none"), Geom.from_features(ZB, "none")
            rec["cka_eq"] = {k: v for k, v in geo.cka_pair(eqA, eqB).items()}
        ts = time.time()
        cb = Calib(A, B, aux, eqA, eqB, pe2_only=pe2_only)
        obs = cb.scores(np.arange(m))
        # observed via the fast path must equal the ladder's values
        chk = {"s_T": row["s_T"], "s_T_bridge": row["s_T_bridge"], "s_T_bridge_B": row["s_T_bridge_B"]}
        for k in KS:
            if f"s_N_k{k}" in row:
                chk[f"s_N_k{k}"] = row[f"s_N_k{k}"]
        if not pe2_only:
            chk["cka_biased"] = row["cka_biased"]
            chk["cka_debiased"] = row["cka_debiased"]
            if eqA is not None:
                chk["cka_eq_biased"] = rec["cka_eq"]["cka_biased"]
                chk["cka_eq_debiased"] = rec["cka_eq"]["cka_debiased"]
        rec["fastpath_obs_vs_ladder_maxabs"] = max(abs(obs[k] - v) for k, v in chk.items())
        names = list(obs.keys())
        null = np.empty((K, len(names)))
        perms_check = {}
        for kk in range(K):
            pi = cal.permutation(m, key, kk)
            s = cb.scores(pi)
            null[kk] = [s[n] for n in names]
            if kk < n_check and not pe2_only:
                ref = recompute_scores(A, B, pi, eqA, eqB)
                perms_check[kk] = max(abs(s[n] - ref[n]) for n in ref)
        T["calib"] = time.time() - ts
        rec["calib_exact_relabel"] = cb.exact_relabel
        rec["calib_spotcheck_maxabs"] = perms_check
        rec["calib"] = {}
        for j, n in enumerate(names):
            if n == "n_shared_bridges_pure":
                rec["calib"][n] = {"s_obs": obs[n], "null_mean": float(null[:, j].mean()),
                                   "null_q95": float(np.quantile(null[:, j], 0.95))}
                continue
            sm = cal.summarize(obs[n], null[:, j], C.ALPHA)
            rec["calib"][n] = sm
            nulls[f"{pre}|{n}"] = null[:, j]
        rec["timing_s"] = {k: round(v, 3) for k, v in T.items()}
        res["per_pre"][pre] = rec
    res["time_total_s"] = round(time.time() - t0, 2)
    return res, nulls


def task_name(lm, vit, r):
    return f"{lm}__{vit}__r{r}"


def worker(args):
    lm, vit, r, run, outdir = args
    name = task_name(lm, vit, r)
    done = os.path.join(outdir, name + ".done")
    if os.path.exists(done):
        return name, "skip", 0.0
    try:
        res, nulls = run_task(lm, vit, r, run)
        tmp = os.path.join(outdir, name + ".json.tmp")
        with open(tmp, "w") as f:
            json.dump(C.jsonable(res), f, indent=0, allow_nan=False)
        os.replace(tmp, os.path.join(outdir, name + ".json"))
        np.savez_compressed(os.path.join(outdir, name + ".npz"), **{k.replace("|", "__"): v for k, v in nulls.items()})
        open(done, "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        return name, "ok", res["time_total_s"]
    except Exception:
        with open(os.path.join(outdir, name + ".failed"), "w") as f:
            f.write(traceback.format_exc())
        return name, "failed", 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main", choices=("main", "sidecar_rule", "pool2047"))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--only-lm", default=None)
    ap.add_argument("--only-vit", default=None)
    ap.add_argument("--depths", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    outdir = os.path.join(C.RES, args.run, "pairs")
    os.makedirs(outdir, exist_ok=True)
    depths = [float(x) for x in args.depths.split(",")] if args.depths else list(C.DEPTHS)
    if args.run == "sidecar_rule":
        depths = [0.5]  # the two rules coincide at r = 1.0
    tasks = []
    for r in depths:
        for lm, _ in C.LMS:
            for vit, _ in C.VITS:
                if args.only_lm and lm != args.only_lm:
                    continue
                if args.only_vit and vit != args.only_vit:
                    continue
                tasks.append((lm, vit, r, args.run, outdir))
    if args.limit:
        tasks = tasks[:args.limit]
    t0 = time.time()
    print(f"{len(tasks)} tasks, {args.jobs} workers, run={args.run}", flush=True)
    n_fail = 0
    if args.jobs <= 1:
        it = map(worker, tasks)
        for i, (name, status, dt) in enumerate(it):
            n_fail += status == "failed"
            print(f"[{time.time() - t0:7.0f}s] {i + 1}/{len(tasks)} {status} {name} {dt:.0f}s", flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.jobs, maxtasksperchild=8) as pool:
            for i, (name, status, dt) in enumerate(pool.imap_unordered(worker, tasks)):
                n_fail += status == "failed"
                print(f"[{time.time() - t0:7.0f}s] {i + 1}/{len(tasks)} {status} {name} {dt:.0f}s", flush=True)
    print(f"finished: {len(tasks)} tasks, {n_fail} failed, {time.time() - t0:.0f}s", flush=True)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
