"""SPEC section 3: quotient (warp) decomposition.

On the upper-triangular pairs of D_A (x) and D_B (y):

    SS0 = sum (y - x)^2
    (a) scale            y ~ c x
    (b) affine           y ~ a x + b
    (c) global isotonic  y ~ g(x), g non-decreasing
    (d) row isotonic     for each anchor i: y_i. ~ g_i(x_i.) over j != i

Explained fraction = 1 - sum r^2 / SS0 (row isotonic: both sums over ordered
pairs, i.e. every pair counted from both of its anchors).

Isotonic fits pool exactly tied x values first (``secondary approach``: g is a
function of x), then run PAVA via ``scipy.optimize.isotonic_regression`` with
the tie counts as weights.

Cross-fitting: items are split by a fixed hash into H1/H2.
  global quotients: fit on pairs with both endpoints in H1, evaluate on pairs
  with both endpoints in H2; isotonic g is extended by linear interpolation
  between fitted knots and constant extrapolation (``np.interp``).
  row isotonic (interpretation, see SPEC_INTERPRETATIONS.md): for every
  anchor i, fit g_i on columns j in H1 \\ {i}, evaluate on columns l in H2 \\ {i}.
Directions: A->B (x=D_A, y=D_B) and B->A (x=D_B, y=D_A); means are reported.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import isotonic_regression

from . import util

QUOTIENTS = ("scale", "affine", "iso", "rowiso")


# ----------------------------------------------------------------------------
# isotonic with tie pooling
# ----------------------------------------------------------------------------

def iso_fit_sorted(xs: np.ndarray, ys: np.ndarray):
    """Isotonic fit for x already sorted ascending (ties allowed).
    Returns (knots_x, knots_g, fitted_sorted)."""
    n = xs.size
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(xs[1:], xs[:-1], out=new[1:])
    if new.all():
        g = isotonic_regression(ys, increasing=True).x
        return xs, g, g
    grp = np.cumsum(new) - 1
    cnt = np.bincount(grp).astype(np.float64)
    ymean = np.bincount(grp, weights=ys) / cnt
    g = isotonic_regression(ymean, weights=cnt, increasing=True).x
    return xs[new], g, g[grp]


def iso_fit(x: np.ndarray, y: np.ndarray):
    """Isotonic fit of y on x. Returns (knots_x, knots_g, fitted in input order)."""
    o = np.argsort(x, kind="stable")
    kx, kg, fs = iso_fit_sorted(x[o], y[o])
    fitted = np.empty_like(fs)
    fitted[o] = fs
    return kx, kg, fitted


def iso_predict(kx: np.ndarray, kg: np.ndarray, xnew: np.ndarray) -> np.ndarray:
    """Linear interpolation between knots, constant extrapolation."""
    return np.interp(xnew, kx, kg)


# ----------------------------------------------------------------------------
# global quotients
# ----------------------------------------------------------------------------

def _fit_scale(x, y):
    return float(np.dot(x, y) / np.dot(x, x))


def _fit_affine(x, y):
    mx, my = x.mean(), y.mean()
    xc = x - mx
    a = float(np.dot(xc, y - my) / np.dot(xc, xc))
    return a, float(my - a * mx)


def _stats(r: np.ndarray, ss0: float):
    if r.size == 0:
        return {"expl": float("nan"), "rms": float("nan"), "max": float("nan")}
    ss = float(np.dot(r, r))
    return {"expl": 1.0 - ss / ss0 if ss0 > 0 else float("nan"),
            "rms": float(np.sqrt(ss / r.size)),
            "max": float(np.max(np.abs(r)))}


def global_quotients_dir(x, y, x1, y1, x2, y2) -> dict:
    """In-sample on (x, y); cross-fitted fit on (x1, y1), evaluate on (x2, y2)."""
    out = {}
    cf_ok = x1.size >= 2 and x2.size >= 1 and np.ptp(x1) > 0
    if not cf_ok:  # too few cross-fit pairs: report NaN for every *_cf value
        x2 = y2 = np.empty(0)
        x1, y1 = x, y
    d0 = y - x
    ss0 = float(np.dot(d0, d0))
    d0c = y2 - x2
    ss0c = float(np.dot(d0c, d0c))
    out["ss0"] = ss0
    out["ss0_cf"] = ss0c
    # scale
    c = _fit_scale(x, y)
    for k, v in _stats(y - c * x, ss0).items():
        out[f"scale_{k}"] = v
    c1 = _fit_scale(x1, y1)
    for k, v in _stats(y2 - c1 * x2, ss0c).items():
        out[f"scale_{k}_cf"] = v
    out["scale_coef"] = c
    # affine
    a, b = _fit_affine(x, y)
    for k, v in _stats(y - (a * x + b), ss0).items():
        out[f"affine_{k}"] = v
    a1, b1 = _fit_affine(x1, y1)
    for k, v in _stats(y2 - (a1 * x2 + b1), ss0c).items():
        out[f"affine_{k}_cf"] = v
    out["affine_slope"], out["affine_intercept"] = a, b
    # global isotonic
    _, _, fitted = iso_fit(x, y)
    for k, v in _stats(y - fitted, ss0).items():
        out[f"iso_{k}"] = v
    del fitted
    kx, kg, _ = iso_fit(x1, y1)
    for k, v in _stats(y2 - iso_predict(kx, kg, x2), ss0c).items():
        out[f"iso_{k}_cf"] = v
    return out


# ----------------------------------------------------------------------------
# row isotonic (parallel over anchors)
# ----------------------------------------------------------------------------

def _row_iso_worker(lo: int, hi: int):
    sh = util.shared()
    DA, DB, h1 = sh["DA"], sh["DB"], sh["h1"]
    m = DA.shape[0]
    # per direction: [ss_res, max_res, ss0, n, ss_res_cf, max_res_cf, ss0_cf, n_cf]
    acc = np.zeros((2, 8), dtype=np.float64)
    allidx = np.arange(m)
    for i in range(lo, hi):
        keep = allidx != i
        f1 = keep & h1
        f2 = keep & ~h1
        for di, (X, Y) in enumerate(((DA, DB), (DB, DA))):
            x = X[i, keep]
            y = Y[i, keep]
            _, _, fit = iso_fit(x, y)
            r = y - fit
            d0 = y - x
            acc[di, 0] += np.dot(r, r)
            acc[di, 1] = max(acc[di, 1], np.max(np.abs(r)))
            acc[di, 2] += np.dot(d0, d0)
            acc[di, 3] += r.size
            x1, y1 = X[i, f1], Y[i, f1]
            x2, y2 = X[i, f2], Y[i, f2]
            if x1.size >= 1 and x2.size >= 1:
                kx, kg, _ = iso_fit(x1, y1)
                r2 = y2 - iso_predict(kx, kg, x2)
                d2 = y2 - x2
                acc[di, 4] += np.dot(r2, r2)
                acc[di, 5] = max(acc[di, 5], np.max(np.abs(r2)))
                acc[di, 6] += np.dot(d2, d2)
                acc[di, 7] += r2.size
    return acc


def row_isotonic(DA: np.ndarray, DB: np.ndarray, h1: np.ndarray, *, n_jobs=None) -> dict:
    res = util.map_rows(_row_iso_worker, DA.shape[0], n_jobs=n_jobs,
                        share={"DA": DA, "DB": DB, "h1": h1})
    acc = np.zeros((2, 8))
    for a in res:
        acc[:, [0, 2, 3, 4, 6, 7]] += a[:, [0, 2, 3, 4, 6, 7]]
        acc[:, 1] = np.maximum(acc[:, 1], a[:, 1])
        acc[:, 5] = np.maximum(acc[:, 5], a[:, 5])
    out = {}
    for di, tag in enumerate(("AB", "BA")):
        ss, mx, ss0, n, ssc, mxc, ss0c, nc = acc[di]
        out[tag] = {
            "rowiso_expl": 1.0 - ss / ss0 if ss0 > 0 else float("nan"),
            "rowiso_rms": float(np.sqrt(ss / n)),
            "rowiso_max": float(mx),
            "rowiso_expl_cf": 1.0 - ssc / ss0c if ss0c > 0 else float("nan"),
            "rowiso_rms_cf": float(np.sqrt(ssc / nc)) if nc else float("nan"),
            "rowiso_max_cf": float(mxc),
        }
    return out


# ----------------------------------------------------------------------------
# full section 3
# ----------------------------------------------------------------------------

def quotient_ladder(DA: np.ndarray, DB: np.ndarray, h1: np.ndarray, *, n_jobs=None,
                    with_row: bool = True) -> dict:
    """All quotients, both directions, in-sample and cross-fitted; flat dict."""
    out = {}
    xA = util.upper_values(DA)
    xB = util.upper_values(DB)
    A1 = util.upper_values_masked(DA, h1)
    B1 = util.upper_values_masked(DB, h1)
    A2 = util.upper_values_masked(DA, ~h1)
    B2 = util.upper_values_masked(DB, ~h1)
    per = {
        "AB": global_quotients_dir(xA, xB, A1, B1, A2, B2),
        "BA": global_quotients_dir(xB, xA, B1, A1, B2, A2),
    }
    del xA, xB, A1, B1, A2, B2
    if with_row:
        rows = row_isotonic(DA, DB, h1, n_jobs=n_jobs)
        for tag in ("AB", "BA"):
            per[tag].update(rows[tag])
    for tag, d in per.items():
        for k, v in d.items():
            out[f"quot_{tag}_{k}"] = v
    for q in QUOTIENTS:
        for suf in ("expl", "expl_cf"):
            ka, kb = f"quot_AB_{q}_{suf}", f"quot_BA_{q}_{suf}"
            if ka in out and kb in out:
                out[f"quot_mean_{q}_{suf}"] = 0.5 * (out[ka] + out[kb])
    # warp shares (mean direction, in-sample and cross-fitted)
    for suf in ("expl", "expl_cf"):
        g = lambda q: out.get(f"quot_mean_{q}_{suf}", float("nan"))
        out[f"warpshare_{suf}_scale"] = g("scale")
        out[f"warpshare_{suf}_affine_extra"] = g("affine") - g("scale")
        out[f"warpshare_{suf}_iso_extra"] = g("iso") - g("affine")
        out[f"warpshare_{suf}_row_extra"] = g("rowiso") - g("iso")
        out[f"warpshare_{suf}_residual"] = 1.0 - g("rowiso")
    return out
