"""SPEC §3: quotient (warp) decomposition with an own pool-adjacent-violators.

Conventions (SPEC is silent; documented choices):
  * Ties in x are pooled first ("tertiary" approach): g is a function of x, so all
    pairs with the same x receive the same fitted value (weighted mean of their y,
    weight = multiplicity), then PAVA runs on the distinct x values.
  * Knots for out-of-sample prediction are *all* distinct training x values with
    their fitted values; prediction = linear interpolation between neighbouring
    knots, constant extrapolation outside [x_min, x_max].
  * Row isotonic in-sample: for every anchor i, fit g_i on {(D_A(i,j), D_B(i,j)) :
    j != i}; residuals over the m(m-1) ordered pairs; explained fraction relative to
    SS0 computed over the same ordered pairs (= 2 x SS0 over unordered pairs).
  * Cross-fit, global quotients: fit on unordered pairs with both endpoints in H1,
    evaluate on unordered pairs with both endpoints in H2; explained fraction uses
    SS0 over the evaluation pairs.
  * Cross-fit, row isotonic (SPEC literally ill-posed: g_i only exists for the
    anchor it was fitted on): for every anchor i, fit g_i on candidates j in
    H1 \\ {i}, evaluate on candidates j in H2 \\ {i}.
"""
from __future__ import annotations

import numpy as np


def pava(y, w):
    """Weighted pool-adjacent-violators for a non-decreasing fit.
    y, w: 1-D sequences (already ordered by x).  Returns fitted values (len(y))."""
    vals = []
    wts = []
    cnts = []
    for yi, wi in zip(np.asarray(y, dtype=np.float64).tolist(), np.asarray(w, dtype=np.float64).tolist()):
        v, ww, c = yi, wi, 1
        while vals and vals[-1] > v:
            pv = vals.pop()
            pw = wts.pop()
            pc = cnts.pop()
            tot = pw + ww
            v = (pv * pw + v * ww) / tot
            ww = tot
            c += pc
        vals.append(v)
        wts.append(ww)
        cnts.append(c)
    return np.repeat(np.array(vals, dtype=np.float64), cnts)


def iso_fit(x, y):
    """Fit non-decreasing g with tie pooling.  Returns (knot_x, knot_g)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    o = np.argsort(x, kind="stable")
    xs = x[o]
    ys = y[o]
    new = np.empty(xs.size, dtype=bool)
    new[0] = True
    new[1:] = xs[1:] != xs[:-1]
    gid = np.cumsum(new) - 1
    kx = xs[new]
    cnt = np.bincount(gid).astype(np.float64)
    ysum = np.bincount(gid, weights=ys)
    ymean = ysum / cnt
    kg = pava(ymean, cnt)
    return kx, kg


def iso_predict(kx, kg, xnew):
    """Linear interpolation between knots, constant extrapolation."""
    xnew = np.asarray(xnew, dtype=np.float64)
    if kx.size == 1:
        return np.full(xnew.shape, kg[0])
    j = np.searchsorted(kx, xnew, side="right")  # kx[j-1] <= x < kx[j]
    out = np.empty(xnew.shape, dtype=np.float64)
    left = j == 0
    right = j >= kx.size
    mid = ~(left | right)
    out[left] = kg[0]
    out[right] = kg[-1]
    jm = j[mid]
    x0 = kx[jm - 1]
    x1 = kx[jm]
    g0 = kg[jm - 1]
    g1 = kg[jm]
    t = (xnew[mid] - x0) / (x1 - x0)
    out[mid] = g0 + t * (g1 - g0)
    return out


def iso_insample(x, y):
    kx, kg = iso_fit(x, y)
    return iso_predict(kx, kg, x)


def _fit_scale(x, y):
    c = np.dot(x, y) / np.dot(x, x)
    return lambda z: c * z


def _fit_affine(x, y):
    xm = x.mean()
    ym = y.mean()
    xc = x - xm
    sxx = np.dot(xc, xc)
    a = np.dot(xc, y - ym) / sxx if sxx > 0 else 0.0
    b = ym - a * xm
    return lambda z: a * z + b


def _fit_iso(x, y):
    kx, kg = iso_fit(x, y)
    return lambda z: iso_predict(kx, kg, z)


FITTERS = {"scale": _fit_scale, "affine": _fit_affine, "iso": _fit_iso}


def _summ(r, ss0):
    rss = float(np.dot(r, r))
    expl = 1.0 - rss / ss0 if ss0 > 0 else float("nan")
    return expl, float(np.sqrt(rss / r.size)), float(np.max(np.abs(r)))


def _offdiag(M, i):
    return np.concatenate([M[i, :i], M[i, i + 1:]])


def quotients_one_direction(DX, DY, H1=None):
    """x = D_X, y = D_Y.  Returns dict {q: (expl, rms, max)} and, if H1 given,
    {q+'_xfit': (...)}."""
    m = DX.shape[0]
    iu, ju = np.triu_indices(m, 1)
    x = DX[iu, ju]
    y = DY[iu, ju]
    ss0 = float(np.dot(y - x, y - x))
    out = {}
    for q, fit in FITTERS.items():
        g = fit(x, y)
        out[q] = _summ(y - g(x), ss0)
    # row isotonic, in-sample
    res = []
    ss0r = 0.0
    for i in range(m):
        xi = _offdiag(DX, i)
        yi = _offdiag(DY, i)
        res.append(yi - iso_insample(xi, yi))
        ss0r += float(np.dot(yi - xi, yi - xi))
    out["rowiso"] = _summ(np.concatenate(res), ss0r)

    if H1 is not None:
        H1 = np.asarray(H1, dtype=bool)
        tr = H1[iu] & H1[ju]
        te = (~H1[iu]) & (~H1[ju])
        xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]
        ss0te = float(np.dot(yte - xte, yte - xte))
        for q, fit in FITTERS.items():
            g = fit(xtr, ytr)
            out[q + "_xfit"] = _summ(yte - g(xte), ss0te)
        res = []
        ss0r = 0.0
        idx = np.arange(m)
        for i in range(m):
            jtr = idx[H1 & (idx != i)]
            jte = idx[(~H1) & (idx != i)]
            kx, kg = iso_fit(DX[i, jtr], DY[i, jtr])
            xi, yi = DX[i, jte], DY[i, jte]
            res.append(yi - iso_predict(kx, kg, xi))
            ss0r += float(np.dot(yi - xi, yi - xi))
        out["rowiso_xfit"] = _summ(np.concatenate(res), ss0r)
    return out
