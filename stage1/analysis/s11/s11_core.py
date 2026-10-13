"""S1.1 core: one model pair, one nested draw -> observed s_N and every predictor.

Protocol (PREREG S1.1; identical to hrhv2.predict.crossfit_predict, which is
the protocol G0 froze, see test_s11.py::test_matches_crossfit_predict):
  * nested pools = prefixes of the draw's item order;
  * H1/H2 = per-item hash halves (fixed across draws and prefixes);
  * fit (P1 coupling, N1, N2, P2 dispersion) on pairs with BOTH endpoints in
    H1 inside the m_fit = 1024 prefix -- no H2 row and no other m is read;
  * at every m: anchors = H2 items of the prefix; neighbours are searched in
    the full m-pool (all items of the prefix, H1 and H2, are candidates);
  * observed s_N(k) = mean over H2 anchors of |N_k^A(i) & N_k^B(i)| / k;
  * N0 = s_N of the H1 anchors at m_fit with their full rows (hrhv2 'N0';
    reads H1 x H2 entries of B -- verifier finding A2, conservative for P1).
Additions (all reported separately, none changes a frozen quantity):
  * both directions A->B and B->A of every asymmetric predictor;
  * exact-tie bookkeeping at the k-boundary (SPEC tie rule picks the lower
    index; verifier D3/D5) and s_N on the anchors with no exact tie;
  * N1 with a non-positive global affine slope (sigma_A = inf in hrhv2) is
    returned as NaN, exactly as hrhv2 defines it; the chance-level limit is
    reported next to it as 'N1c' (ANALYSIS_PLAN_S11.md O8).
"""
from __future__ import annotations

import time

import numpy as np

from s11_common import PROJECT  # noqa: F401  (puts hrhv2 on sys.path)
from hrhv2 import predict as pr
from hrhv2 import structural as st


def knn_rows(D: np.ndarray, rows: np.ndarray, kmax: int, *, chunk: int = 512):
    """hrhv2.structural.knn_table restricted to ``rows`` (same tie rule (D_ij, j),
    self excluded). Returns (idx, val) of shape (len(rows), min(kmax, m-1))."""
    m = D.shape[0]
    K = int(min(kmax, m - 1))
    rows = np.asarray(rows, dtype=np.int64)
    idx_out = np.empty((rows.size, K), dtype=np.int64)
    val_out = np.empty((rows.size, K), dtype=np.float64)
    for lo in range(0, rows.size, chunk):
        hi = min(rows.size, lo + chunk)
        n = hi - lo
        r = rows[lo:hi]
        Dc = np.array(D[r], dtype=np.float64, copy=True)
        Dc[np.arange(n), r] = np.inf
        v = np.partition(Dc, K - 1, axis=1)[:, K - 1]
        lt = Dc < v[:, None]
        eq = Dc == v[:, None]
        need = K - lt.sum(axis=1)
        sel = lt | (eq & (np.cumsum(eq, axis=1) <= need[:, None]))
        cols = np.nonzero(sel)[1].reshape(n, K)
        vals = np.take_along_axis(Dc, cols, axis=1)
        order = np.lexsort((cols, vals), axis=-1)
        idx_out[lo:hi] = np.take_along_axis(cols, order, axis=1)
        val_out[lo:hi] = np.take_along_axis(vals, order, axis=1)
    return idx_out, val_out


def _sub(Dpool: np.ndarray, items: np.ndarray, identity_prefix: bool):
    m = items.size
    if identity_prefix:
        return Dpool[:m, :m]
    return Dpool[np.ix_(items, items)]


def _fit_direction(DXf, DYf, fit_mask, variant):
    t0 = time.perf_counter()
    model = pr.fit_coupling(DXf, DYf, fit_mask, variant=variant)
    n1 = pr.fit_n1(DXf, DYf, fit_mask)
    rhos = pr.fit_n2(DXf, DYf, fit_mask)
    disp = pr.fit_iso_dispersion(DXf, DYf, fit_mask)
    diag = {"P1_fit": model.summary(), "N1_fit": n1,
            "N2_rho_median": float(np.median(rhos)), "N2_rho_iqr": [float(np.quantile(rhos, 0.25)), float(np.quantile(rhos, 0.75))],
            "P2_disp": {"n_bins": int(disp.xc.size), "lo_slope": float(disp.lo_slope), "hi_slope": float(disp.hi_slope)},
            "fit_seconds": time.perf_counter() - t0}
    return {"model": model, "n1": n1, "rhos": rhos, "disp": disp}, diag


def _predict_direction(DXm, anchors, ks, fit, m, lb_bias=None):
    out = {}
    t0 = time.perf_counter()
    p1 = pr.predict_p1(DXm, anchors, ks, fit["model"])
    t1 = time.perf_counter()
    sig = fit["n1"]["sigma_A"]
    if np.isfinite(sig):
        n1 = pr.predict_n1(DXm, anchors, ks, fit["n1"])
        n1v = {k: n1[k]["s_pred"] for k in ks}
    else:  # hrhv2: sigma_A = inf when the affine slope <= 0 -> undefined
        n1v = {k: float("nan") for k in ks}
    n2 = pr.predict_n2(fit["rhos"], m, ks)
    p2 = pr.predict_p2(DXm, anchors, ks, fit["disp"], lb_bias=lb_bias)
    t2 = time.perf_counter()
    for k in ks:
        chance = k / (m - 1)
        out[k] = {"P1": p1[k]["s_pred"], "P1_mu": p1[k]["mu_mean"],
                  "N1": n1v[k], "N1c": n1v[k] if np.isfinite(sig) else chance,
                  "N2": n2[k]["s_pred"], "P2": p2[k]["s_pred"], "P2_raw": p2[k]["s_pred_raw"],
                  "P2_phat": p2[k]["p_hat"]}
    return out, {"p1_s": t1 - t0, "nulls_p2_s": t2 - t1}


def run_pair_draw(DA_pool: np.ndarray, DB_pool: np.ndarray, order: np.ndarray, h1_pool: np.ndarray, *,
                  m_grid, m_fit: int, ks, variant: str, directions=("AB", "BA"), lb_bias=None,
                  log=None):
    """One draw of one pair. ``order`` = the draw's item order (pool row indices);
    the pool of size m is order[:m]. Returns (rows, diag)."""
    ks = [int(k) for k in ks]
    kmax = max(ks) + 1
    order = np.asarray(order, dtype=np.int64)
    ident = bool(np.array_equal(order, np.arange(order.size)))
    h1_ord = np.asarray(h1_pool, dtype=bool)[order]
    diag = {"identity_order": ident, "timing": {}}

    # ---- fit on H1 x H1 of the m_fit prefix ---------------------------------
    items_f = order[:m_fit]
    fit_mask = h1_ord[:m_fit]
    DAf = np.ascontiguousarray(_sub(DA_pool, items_f, ident))
    DBf = np.ascontiguousarray(_sub(DB_pool, items_f, ident))
    fits = {}
    for d in directions:
        if d == "AB":
            fits[d], diag[f"fit_{d}"] = _fit_direction(DAf, DBf, fit_mask, variant)
        elif d == "BA":
            fits[d], diag[f"fit_{d}"] = _fit_direction(DBf, DAf, fit_mask, variant)
        else:
            raise ValueError(d)
    h1i = np.flatnonzero(fit_mask)
    nAf, _ = knn_rows(DAf, h1i, kmax)
    nBf, _ = knn_rows(DBf, h1i, kmax)
    n0 = {k: float(st.overlap_counts(nAf, nBf, k).mean() / k) for k in ks if k <= m_fit - 2}
    diag["n_fit_anchors"] = int(h1i.size)
    del DAf, DBf, nAf, nBf

    rows = []
    for m in m_grid:
        t0 = time.perf_counter()
        items = order[:m]
        DAm = _sub(DA_pool, items, ident)
        DBm = _sub(DB_pool, items, ident)
        anchors = np.flatnonzero(~h1_ord[:m])
        kk = [k for k in ks if k <= m - 2]
        nA, vA = knn_rows(DAm, anchors, kmax)
        nB, vB = knn_rows(DBm, anchors, kmax)
        t1 = time.perf_counter()
        preds = {}
        ptime = {}
        for d in directions:
            DX = DAm if d == "AB" else DBm
            preds[d], ptime[d] = _predict_direction(DX, anchors, kk, fits[d], m, lb_bias=lb_bias)
        t2 = time.perf_counter()
        for k in kk:
            ov = st.overlap_counts(nA, nB, k) / k
            tieA = vA[:, k] == vA[:, k - 1]   # exact tie between k-th and (k+1)-th candidate
            tieB = vB[:, k] == vB[:, k - 1]
            clean = ~(tieA | tieB)
            row = {"m": int(m), "k": int(k), "is_fit_m": bool(m == m_fit),
                   "n_anchors": int(anchors.size), "chance": k / (m - 1),
                   "obs": float(ov.mean()), "obs_sd": float(ov.std(ddof=1)),
                   "n_tieA": int(tieA.sum()), "n_tieB": int(tieB.sum()),
                   "n_clean": int(clean.sum()),
                   "obs_clean": float(ov[clean].mean()) if clean.any() else float("nan"),
                   "N0": n0.get(k, float("nan"))}
            for d in directions:
                for key, v in preds[d][k].items():
                    row[f"{key}_{d}"] = v
            rows.append(row)
        diag["timing"][int(m)] = {"knn_s": t1 - t0, "predict_s": t2 - t1, **{f"{d}_{kk_}": v for d in ptime for kk_, v in ptime[d].items()}}
        if log is not None:
            log(f"    m={m} anchors={anchors.size} knn {t1 - t0:.1f}s predict {t2 - t1:.1f}s "
                + " ".join(f"k={r['k']}:obs={r['obs']:.4f}/P1AB={r.get('P1_AB', float('nan')):.4f}" for r in rows[-len(kk):]))
        del DAm, DBm, nA, nB, vA, vB
    return rows, diag
