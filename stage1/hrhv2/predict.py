"""SPEC section 6: mechanism readouts and predictors.

P1  crossing-law predictor mu-hat (Thm 2 of theory_mechanism.md).
    For anchor i with sorted A-row u and A-neighbourhood N_A(i) (first k
    columns), under an estimated conditional law of B's row given u, find t*
    with E X(t*) = E Y(t*),
        X(t) = #{j in N_A: b_j > t},   Y(t) = #{l not in N_A: b_l <= t},
    and set mu_i = sum_{j in N_A} P(b_j > t*). Predicted 1 - s_N = mean_i mu_i / k.

    Coupling estimator (``fit_coupling``), location-scale in A-units:
      1. Fit rows: anchors in the fit mask, columns restricted to the fit mask
         (pairs with BOTH endpoints in H1). No entry of a predicted (H2)
         anchor's B-row is ever read (tested in tests/test_predict.py).
      2. Per fit row, remove B's monotone warp with a multi-scale local affine
         fit (dyadic rank bands, empirical-Bayes-shrunk slopes) and form the
         A-unit residual r = (b - a - c u)/c, studentized by its leverage
         (r / sqrt(1 - h)) so in-sample fitting does not shrink the dispersion.
      3. Pool residuals over fit rows against a covariate x (the ``pool``):
           'u'   : x = log u                (raw A-distance; scale unit 1)
           'med' : x = log(u / nu_i)        (nu_i = median of anchor i's A-row;
                                             residual measured in units nu_i)
           'q'   : x = log(rank / n_items)  (rank-normalized position; unit 1)
         and estimate location and mean-absolute-deviation (MAD) per
         equal-count bin of x; log-MAD is interpolated piecewise-linearly in x
         and extrapolated by power laws fitted on the lowest / highest bins
         (needed to transport the coupling to other m).
      4. Standardized residual law zeta: Gaussian ('G') or the empirical law
         of the MAD-standardized residuals ('E').
    The pooled coupling is transportable across m because, under i.i.d.
    evaluation sampling, P(d_B | d_A, x_i) does not depend on m.
    Variants are named 'P1-<pool>-<zeta>'; the frozen choice is recorded in
    PREDICTOR_CHOICE.md (``FROZEN_P1_VARIANT``).

P2  dispersion law: 1 - s_N ~ (p-hat/2) mean_i delta_i(r_k) with delta in
    plain-distance units; in squared-distance units D this is
    (p-hat/4) mean_i MAD_iso(D_k,i)/D_k,i, with MAD_iso the mean absolute
    deviation of the row-isotonic residual (in A-units via the local band
    slope) at A-distance D_k,i and p-hat the Levina-Bickel MLE at scale k
    (optionally divided by a plant-calibrated bias ratio).

Nulls
    N0: s_N constant in m (fit-m value at the same k);
    N1: independent additive Gaussian noise in A-units with sigma = RMS of the
        global affine residual (fit pairs) / affine slope;
    N2: per-anchor Gaussian copula with constant rho_i fitted on the rank-
        normal scores of the bulk, prediction mean_i Phi2(z_q, z_q; rho_i)/q,
        q = k/(m-1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import multivariate_normal

from . import util

SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))
POOLS = ("u", "med", "q")
ZETAS = {"G": "gauss", "E": "emp", "B": "bin"}
P1_CANDIDATES = ("P1-med-B-r8", "P1-med-B-r8f", "P1-med-B-difff")
# The frozen choice (see PREDICTOR_CHOICE.md). Set once, after the G0 variant
# comparison; never changed afterwards without a new PREDICTOR_CHOICE version.
FROZEN_P1_VARIANT = "P1-med-B-r8"  # frozen after G0 (PREDICTOR_CHOICE.md)

_GAUSS_ZMAX_SD = 9.0  # P(|Z| > 9) ~ 2e-19: Gaussian treated as supported on +-9 sd


LINKS = ("all", "r8", "diff")


def parse_variant(variant: str):
    """'P1-<pool>-<zeta>[-<link>[f]]' -> (pool, zeta, link, floor).

    link = how the dispersion is separated from B's per-anchor warp:
      'all'  : studentized band-affine residuals at every rank (default);
      'r8'   : the same residuals, but ranks < 8 (whose band window cannot be
               centred, so the link extrapolation error dominates) are
               excluded; smaller scales are reached by the power-law tail;
      'diff' : scale from adjacent-rank differences
               d_r = (b_(r+1) - b_(r))/c_r - (u_(r+1) - u_(r)) = xi_(r+1) - xi_(r),
               which cancel the link intercept at every rank (MAD_xi =
               MAD_d / (sqrt(2) kappa), kappa from the zeta law); location and
               zeta shape from the ranks >= 8 residuals.
    A trailing 'f' (e.g. 'r8f') truncates every column's law to b~ >= 0 in
    A-units (distances are non-negative), renormalizing the remaining mass.
    """
    parts = variant.split("-")
    if len(parts) == 3:
        parts = parts + ["all"]
    floor = len(parts) == 4 and parts[3].endswith("f") and parts[3][:-1] in LINKS
    link = parts[3][:-1] if floor else (parts[3] if len(parts) == 4 else None)
    if (len(parts) != 4 or parts[0] != "P1" or parts[1] not in POOLS or parts[2] not in ZETAS
            or link not in LINKS):
        raise ValueError(f"P1 variant must be 'P1-<{'|'.join(POOLS)}>-<G|E|B>[-<{'|'.join(LINKS)}>[f]]', got {variant!r}")
    return parts[1], ZETAS[parts[2]], link, floor


# ----------------------------------------------------------------------------
# Levina-Bickel intrinsic dimension
# ----------------------------------------------------------------------------

def lb_dimension(knn_vals_D: np.ndarray, k: int) -> dict:
    """Levina-Bickel (2004) MLE at scale k from sorted squared NN distances.

    m_k(x) = [ (1/(k-1)) sum_{j<k} log(T_k / T_j) ]^{-1}, T_j = sqrt(D_(j)).
    Returns per-anchor estimates and the two standard averages:
    'lb' = mean of per-anchor estimates (LB 2004), 'mg' = inverse of the mean
    inverse (MacKay & Ghahramani's correction). Anchors with T_1 = 0 or
    T_k = T_1 are dropped.
    """
    if k < 2:
        raise ValueError("Levina-Bickel needs k >= 2")
    T = np.sqrt(np.asarray(knn_vals_D[:, :k], dtype=np.float64))
    ok = (T[:, 0] > 0) & (T[:, k - 1] > T[:, 0])
    T = T[ok]
    logs = np.log(T[:, k - 1:k]) - np.log(T[:, :k - 1])
    inv = logs.sum(axis=1) / (k - 1)
    est = 1.0 / inv
    return {"per_anchor": est, "lb": float(np.mean(est)), "mg": float(1.0 / np.mean(inv)),
            "n_used": int(ok.sum())}


# ----------------------------------------------------------------------------
# Row helpers
# ----------------------------------------------------------------------------

def sorted_rows(DA_sub: np.ndarray, DB_sub: np.ndarray | None = None):
    """Rows of a square block with the diagonal removed, columns ordered by
    the tie rule (D_A, j). Returns (U, B) (B is None if DB_sub is None)."""
    n = DA_sub.shape[0]
    U = np.empty((n, n - 1))
    B = None if DB_sub is None else np.empty((n, n - 1))
    for i in range(n):
        u = np.concatenate((DA_sub[i, :i], DA_sub[i, i + 1:]))
        o = np.argsort(u, kind="stable")  # stable on ascending j == key (u, j)
        U[i] = u[o]
        if B is not None:
            b = np.concatenate((DB_sub[i, :i], DB_sub[i, i + 1:]))
            B[i] = b[o]
    return U, B


def anchor_rows(DA: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Sorted A-rows (j != i) of the given anchors (values only; mu depends on
    the A-row only through its sorted values and N_A, and tied values enter
    the balance symmetrically)."""
    m = DA.shape[0]
    anchors = np.asarray(anchors, dtype=np.int64)
    U = np.empty((anchors.size, m - 1))
    for r, i in enumerate(anchors):
        row = DA[i]
        U[r, :i] = row[:i]
        U[r, i:] = row[i + 1:]
    U.sort(axis=1)
    return U


def band_layout(n: int, min_window: int = 32):
    """Rank bands (0-based, half-open) for the multi-scale local-affine warp.

    Core g covers ranks [2^g - 1, 2^(g+1) - 1) (1-based ranks 2^g..2^(g+1)-1);
    the warp for core g is fitted on the window [2^(g-1) - 1, 2^(g+2) - 1)
    widened to at least ``min_window`` ranks. Returns a list of
    (core_lo, core_hi, win_lo, win_hi)."""
    out = []
    g = 0
    while (1 << g) - 1 < n:
        c_lo, c_hi = (1 << g) - 1, min(n, (1 << (g + 1)) - 1)
        w_lo = max(0, (1 << max(g - 1, 0)) - 1)
        w_hi = min(n, max((1 << (g + 2)) - 1, w_lo + min_window))
        if w_hi - w_lo < min_window:
            w_lo = max(0, w_hi - min_window)
        out.append((c_lo, c_hi, w_lo, w_hi))
        g += 1
    return out


def _eb_affine(U: np.ndarray, B: np.ndarray):
    """Per-row OLS b ~ a + c u with empirical-Bayes shrinkage of c toward the
    precision-weighted pooled slope (method of moments for the prior
    variance). Returns (a, c, w, ubar, sxx): intercepts, slopes, shrinkage
    weights (1 = no shrinkage), row means and centred sums of squares of u."""
    n = U.shape[1]
    um = U.mean(axis=1)
    bm = B.mean(axis=1)
    uc = U - um[:, None]
    sxx = np.einsum("ij,ij->i", uc, uc)
    sxy = np.einsum("ij,ij->i", uc, B - bm[:, None])
    good = sxx > 0
    sxx_safe = np.where(good, sxx, 1.0)
    c_hat = np.where(good, sxy / sxx_safe, np.nan)
    res = (B - bm[:, None]) - np.where(good, c_hat, 0.0)[:, None] * uc
    s2 = np.einsum("ij,ij->i", res, res) / max(n - 2, 1)
    se2 = np.where(good, s2 / sxx_safe, np.inf)
    ok = np.isfinite(c_hat) & np.isfinite(se2)
    if ok.any():
        w0 = 1.0 / np.maximum(se2[ok], 1e-300)
        cbar = float(np.sum(w0 * c_hat[ok]) / np.sum(w0))
    else:
        cbar = 1.0
    tau2 = max(float(np.var(c_hat[ok]) - np.mean(se2[ok])), 0.0) if ok.sum() > 2 else 0.0
    w = np.where(ok, tau2 / (tau2 + np.where(ok, se2, 1.0)), 0.0)
    c = np.where(ok, cbar + w * (np.where(ok, c_hat, cbar) - cbar), cbar)
    floor = 0.05 * abs(cbar) if cbar > 0 else 1e-12
    c = np.maximum(c, floor)
    a = bm - c * um
    return a, c, w, um, sxx


def coupling_residuals(U: np.ndarray, B: np.ndarray, *, min_window: int = 32,
                       studentize: bool = True):
    """Studentized A-unit residuals r = (b - a_i - c_i u)/c_i / sqrt(1 - h)
    for every (row, rank), with (a_i, c_i) the multi-scale local affine warp
    of ``band_layout``. Returns (R, C): residuals and the local slopes."""
    n = U.shape[1]
    R = np.empty_like(U)
    C = np.empty_like(U)
    for c_lo, c_hi, w_lo, w_hi in band_layout(n, min_window):
        a, c, w, um, sxx = _eb_affine(U[:, w_lo:w_hi], B[:, w_lo:w_hi])
        Uc = U[:, c_lo:c_hi]
        r = (B[:, c_lo:c_hi] - a[:, None]) / c[:, None] - Uc
        if studentize:
            nw = w_hi - w_lo
            h = 1.0 / nw + w[:, None] * (Uc - um[:, None]) ** 2 / np.where(sxx > 0, sxx, np.inf)[:, None]
            r = r / np.sqrt(np.maximum(1.0 - h, 1e-3))
        R[:, c_lo:c_hi] = r
        C[:, c_lo:c_hi] = c[:, None]
    return R, C


def pool_coords(U: np.ndarray, pool: str, n_items: int):
    """Covariate x and residual unit for sorted full rows U (n_rows, n_items-1)."""
    with np.errstate(divide="ignore"):
        if pool == "u":
            return np.log(U), np.ones((U.shape[0], 1))
        if pool == "med":
            nu = np.median(U, axis=1)
            nu = np.where(nu > 0, nu, 1.0)
            return np.log(U / nu[:, None]), nu[:, None]
        if pool == "q":
            q = np.arange(1, U.shape[1] + 1, dtype=np.float64) / n_items
            return np.broadcast_to(np.log(q)[None, :], U.shape), np.ones((U.shape[0], 1))
    raise ValueError(pool)


# ----------------------------------------------------------------------------
# Coupling model (P1)
# ----------------------------------------------------------------------------

@dataclass
class CouplingModel:
    variant: str
    pool: str
    zeta: str
    xc: np.ndarray          # bin centres of the covariate (scale / location)
    loc: np.ndarray         # mean residual per bin (residual units)
    mad: np.ndarray         # mean absolute deviation per bin
    cnt: np.ndarray
    lo_slope: float         # d log MAD / dx below the first bin centre
    hi_slope: float         # ... above the last bin centre
    x_floor: float
    zq: np.ndarray | None = None          # 'emp': pooled quantiles of zeta
    shape_xc: np.ndarray | None = None    # 'bin': centres of the shape bins
    cdf_table: np.ndarray | None = None   # 'bin': CDF of zeta per shape bin on zgrid
    zg0: float = 0.0
    dz: float = 1.0
    shape_lo: np.ndarray | None = None    # 'bin': support of zeta per shape bin
    shape_hi: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def scale(self, x: np.ndarray) -> np.ndarray:
        x = np.maximum(x, self.x_floor)
        lm = np.log(self.mad)
        out = np.interp(x, self.xc, lm)
        out = np.where(x < self.xc[0], lm[0] + self.lo_slope * (x - self.xc[0]), out)
        out = np.where(x > self.xc[-1], lm[-1] + self.hi_slope * (x - self.xc[-1]), out)
        return np.exp(out)

    def location(self, x: np.ndarray) -> np.ndarray:
        """Mean residual, modelled in units of the local MAD (interpolated,
        constant outside the fitted range) so that it is extrapolated
        together with the scale; never an absolute offset at tiny u."""
        x = np.maximum(x, self.x_floor)
        return np.interp(x, self.xc, self.loc / self.mad) * self.scale(x)

    def prepare(self, x: np.ndarray):
        """Per-column auxiliary data for F (shape-bin indices and weights;
        the shape is held constant outside the fitted covariate range)."""
        if self.zeta != "bin":
            return None
        sx = self.shape_xc
        if sx.size == 1:
            z0 = np.zeros(np.shape(x), dtype=np.int32)
            return z0, z0, np.zeros(np.shape(x))
        xs = np.clip(x, sx[0], sx[-1])
        j1 = np.clip(np.searchsorted(sx, xs, side="right"), 1, sx.size - 1)
        j0 = j1 - 1
        w = np.clip((xs - sx[j0]) / (sx[j1] - sx[j0]), 0.0, 1.0)
        return j0.astype(np.int32), j1.astype(np.int32), w

    def F(self, z: np.ndarray, aux=None) -> np.ndarray:
        """CDF of the MAD-standardized residual (at the columns described by aux)."""
        if self.zeta == "gauss":
            return ndtr(z * SQRT_2_OVER_PI)
        if self.zeta == "emp":
            n = self.zq.size
            pq = (np.arange(n) + 0.5) / n
            return np.interp(z, self.zq, pq, left=0.0, right=1.0)
        j0, j1, w = aux
        ng = self.cdf_table.shape[1]
        t = np.clip((z - self.zg0) / self.dz, 0.0, ng - 1.000001)
        i = t.astype(np.int64)
        f = t - i
        T = self.cdf_table
        a = T[j0, i] * (1.0 - f) + T[j0, i + 1] * f
        b = T[j1, i] * (1.0 - f) + T[j1, i + 1] * f
        return (1.0 - w) * a + w * b

    def column_support(self, aux):
        """Per-column support [zlo, zhi] of zeta ('bin' only; else None)."""
        if self.zeta != "bin" or aux is None:
            return None
        j0, j1, _ = aux
        return (np.minimum(self.shape_lo[j0], self.shape_lo[j1]),
                np.maximum(self.shape_hi[j0], self.shape_hi[j1]))

    def zsupport(self):
        if self.zeta == "gauss":
            z = _GAUSS_ZMAX_SD / SQRT_2_OVER_PI
            return -z, z
        if self.zeta == "emp":
            return float(self.zq[0]), float(self.zq[-1])
        return self.zg0, self.zg0 + self.dz * (self.cdf_table.shape[1] - 1)

    def summary(self) -> dict:
        return {"variant": self.variant, "pool": self.pool, "zeta": self.zeta,
                "n_bins": int(self.xc.size), "lo_slope": float(self.lo_slope),
                "hi_slope": float(self.hi_slope),
                "x_range": [float(self.xc[0]), float(self.xc[-1])],
                "mad_range": [float(self.mad.min()), float(self.mad.max())],
                "zsupport": [float(v) for v in self.zsupport()],
                **{k: v for k, v in self.meta.items() if isinstance(v, (int, float, str, bool))}}


def _tail_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    if x.size < 2 or np.ptp(x) == 0:
        return 0.0
    return float(np.clip(np.polyfit(x, y, 1, w=np.sqrt(w))[0], -4.0, 4.0))


def _binned(x: np.ndarray, y: np.ndarray, n_bins: int, center: bool = True, return_sel: bool = False):
    """Equal-count bins of x: centres, mean y, mean |y - mean y| (or mean |y|
    if center=False), counts (and the member indices if return_sel)."""
    o = np.argsort(x, kind="stable")
    edges = np.linspace(0, o.size, n_bins + 1).astype(np.int64)
    xc, loc, mad, cnt, sels = [], [], [], [], []
    for b in range(n_bins):
        sel = o[edges[b]:edges[b + 1]]
        if sel.size < 20:
            continue
        yy = y[sel]
        mu = float(yy.mean())
        xc.append(float(x[sel].mean()))
        loc.append(mu)
        mad.append(max(float(np.mean(np.abs(yy - mu))) if center else float(np.mean(np.abs(yy))), 1e-300))
        cnt.append(sel.size)
        sels.append(sel)
    out = tuple(map(np.asarray, (xc, loc, mad, cnt)))
    return out + (sels,) if return_sel else out


def _diff_kappa(zq: np.ndarray | None, zeta: str) -> float:
    """kappa = MAD(Z - Z') / (sqrt 2 MAD(Z)) for Z, Z' iid from the zeta law
    (1 for the Gaussian); MAD = mean absolute deviation."""
    if zeta == "gauss" or zq is None:
        return 1.0
    z = zq[:: max(1, zq.size // 1001)]
    madz = float(np.mean(np.abs(z - z.mean())))
    dd = np.abs(z[:, None] - z[None, :])
    return float(dd.mean() / (np.sqrt(2.0) * madz))


def _fit_shape_tables(model, x, y, sels, xc, n_q: int = 401, n_grid: int = 4001):
    """Per-bin empirical law of the standardized residual, tabulated as CDFs
    on one uniform z grid (shape varies with the covariate; 'B' variants)."""
    qs = []
    pq = (np.arange(n_q) + 0.5) / n_q
    for sel in sels:
        zz = np.sort((y[sel] - model.location(x[sel])) / model.scale(x[sel]))
        qs.append(np.interp(pq * zz.size - 0.5, np.arange(zz.size), zz))
    qs = np.array(qs)
    lo, hi = float(qs.min()), float(qs.max())
    pad = 1e-6 * max(1.0, hi - lo)
    zg = np.linspace(lo - pad, hi + pad, n_grid)
    T = np.empty((len(qs), n_grid))
    for j, q in enumerate(qs):
        T[j] = np.interp(zg, q, pq, left=0.0, right=1.0)
    T[:, 0] = 0.0
    T[:, -1] = 1.0
    model.shape_lo = qs[:, 0].copy()
    model.shape_hi = qs[:, -1].copy()
    model.shape_xc = np.asarray(xc, dtype=np.float64)
    model.cdf_table = T
    model.zg0 = float(zg[0])
    model.dz = float(zg[1] - zg[0])
    model.meta["shape_bins"] = int(len(qs))


def _diff_kappa_bins(model) -> float:
    """kappa for the per-bin law: average over bins of MAD(Z-Z')/(sqrt2 MAD(Z))."""
    pq = np.linspace(0.0005, 0.9995, 1000)
    zg = model.zg0 + model.dz * np.arange(model.cdf_table.shape[1])
    ks = []
    for T in model.cdf_table:
        z = np.interp(pq, T, zg)
        madz = float(np.mean(np.abs(z - z.mean())))
        ks.append(float(np.abs(z[:, None] - z[None, :]).mean() / (np.sqrt(2.0) * madz)))
    return float(np.mean(ks))


def fit_coupling(DA: np.ndarray, DB: np.ndarray, fit_mask: np.ndarray, *,
                 variant: str = "P1-med-E", n_bins: int | None = None,
                 min_window: int = 32, n_quant: int = 2001,
                 tail_frac: float = 0.25, r_min: int = 8) -> CouplingModel:
    """Estimate the pooled A-unit location-scale coupling from pairs with
    BOTH endpoints in ``fit_mask`` (H1 x H1). See the module docstring and
    ``parse_variant``."""
    pool, zeta, link, floor = parse_variant(variant)
    idx = np.flatnonzero(fit_mask)
    n = idx.size
    if n < 2 * min_window:
        raise ValueError("too few fit anchors")
    U, B = sorted_rows(DA[np.ix_(idx, idx)], DB[np.ix_(idx, idx)])
    R, C = coupling_residuals(U, B, min_window=min_window)
    X, unit = pool_coords(U, pool, n)
    unit = np.broadcast_to(unit, U.shape)
    rank0 = np.broadcast_to(np.arange(U.shape[1])[None, :], U.shape)
    keep = np.isfinite(X) & (U > 0)
    if link in ("r8", "diff"):
        keep &= rank0 >= r_min - 1
    x = np.asarray(X[keep], dtype=np.float64)
    y = R[keep] / unit[keep]
    if n_bins is None:
        n_bins = int(np.clip(x.size // 5000, 12, 40))
    xc, loc, mad, cnt, sels = _binned(x, y, n_bins, return_sel=True)
    meta = {"n_fit_anchors": int(n), "n_pairs_used": int(x.size), "link": link, "r_min": int(r_min),
            "floor": bool(floor)}
    model = CouplingModel(variant=variant, pool=pool, zeta=zeta, xc=xc, loc=loc, mad=mad,
                          cnt=cnt, lo_slope=0.0, hi_slope=0.0, x_floor=float(x.min()) - 25.0,
                          meta=meta)
    nt = max(3, int(round(tail_frac * xc.size)))
    model.lo_slope = _tail_slope(xc[:nt], np.log(mad[:nt]), cnt[:nt])
    model.hi_slope = _tail_slope(xc[-nt:], np.log(mad[-nt:]), cnt[-nt:])
    if zeta == "emp":
        zs = (y - model.location(x)) / model.scale(x)
        zs.sort()
        qpos = (np.arange(n_quant) + 0.5) / n_quant * zs.size - 0.5
        model.zq = np.interp(qpos, np.arange(zs.size), zs)
    elif zeta == "bin":
        _fit_shape_tables(model, x, y, sels, xc)
    if link == "diff":
        # link-free scale: adjacent-rank differences of A-unit values
        dU = np.diff(U, axis=1)
        dB = np.diff(B, axis=1) / C[:, :-1]
        Dd = dB - dU
        Um = 0.5 * (U[:, 1:] + U[:, :-1])
        Xd, unit_d = pool_coords(Um, pool, n) if pool != "q" else (0.5 * (X[:, 1:] + X[:, :-1]), unit[:, :1])
        unit_d = np.broadcast_to(unit_d, Dd.shape)
        kd = np.isfinite(Xd) & (U[:, :-1] > 0)
        xd = np.asarray(Xd[kd], dtype=np.float64)
        yd = Dd[kd] / unit_d[kd]
        xc2, _, madd, cnt2 = _binned(xd, yd, int(np.clip(xd.size // 5000, 12, 40)), center=False)
        kappa = _diff_kappa(model.zq, zeta) if zeta != "bin" else _diff_kappa_bins(model)
        model.xc, model.mad, model.cnt = xc2, madd / (np.sqrt(2.0) * kappa), cnt2
        model.loc = np.interp(xc2, xc, loc)
        nt = max(3, int(round(tail_frac * xc2.size)))
        model.lo_slope = _tail_slope(xc2[:nt], np.log(model.mad[:nt]), cnt2[:nt])
        model.hi_slope = _tail_slope(xc2[-nt:], np.log(model.mad[-nt:]), cnt2[-nt:])
        model.x_floor = min(model.x_floor, float(xd.min()) - 25.0)
        model.meta["diff_kappa"] = kappa
    return model


# ----------------------------------------------------------------------------
# Crossing count mu (Thm 2) under independent location-scale laws
# ----------------------------------------------------------------------------

def mu_balance(c: np.ndarray, s: np.ndarray, k: int, F, zsupport, *, aux=None,
               col_support=None, n_iter: int = 60, block: int = 256) -> np.ndarray:
    """mu_i for rows of centres c and scales s (n_anchor, n_cols); the first k
    columns of each row are N_A(i). b = c + s * zeta, zeta ~ F(., aux)
    supported on [zlo, zhi] (F exactly 0 / 1 outside, or negligibly so);
    ``aux`` is an optional tuple of per-column arrays shaped like c that is
    sliced alongside c and passed to F; ``col_support`` optionally gives
    per-column supports (zlo, zhi) arrays shaped like c (tighter truncation
    and bracketing; zsupport is then unused). Solves E X(t) = E Y(t) per row by
    safeguarded Illinois false position on [t_lo, t_hi] (to |g| <= 1e-12 k):
      t_hi = max_{j<k} c_j + zhi s_j   (E X(t_hi) = 0 <= E Y(t_hi)),
      t_lo = min_l c_l + zlo s_l       (E Y(t_lo) = 0 <= E X(t_lo) = k).
    Columns l >= k with c_l + zlo s_l >= t_hi cannot contribute for t <= t_hi
    and are dropped (exact up to F's support), which keeps the cost ~ O(k)."""
    n_a, n_c = c.shape
    if k >= n_c:
        return np.zeros(n_a)
    if col_support is None:
        zlo, zhi = zsupport
    else:
        zlo, zhi = col_support
    zlo = np.broadcast_to(zlo, c.shape)
    zhi = np.broadcast_to(zhi, c.shape)
    s = np.maximum(s, 1e-300)
    t_hi_all = np.max(c[:, :k] + zhi[:, :k] * s[:, :k], axis=1)
    need = (c[:, k:] + zlo[:, k:] * s[:, k:]) < t_hi_all[:, None]
    any_need = need.any(axis=1)
    last = np.where(any_need, n_c - 1 - np.argmax(need[:, ::-1], axis=1), k - 1)
    L_all = last + 1
    order = np.argsort(L_all, kind="stable")
    mu = np.empty(n_a)
    for lo_b in range(0, n_a, block):
        rows = order[lo_b:lo_b + block]
        L = int(L_all[rows].max())
        cc, ss = c[rows, :L], s[rows, :L]
        cA, sA = cc[:, :k], ss[:, :k]
        cN, sN = cc[:, k:], ss[:, k:]
        if aux is None:
            aA = aN = None
        else:
            aA = tuple(np.asarray(a)[rows, :k] for a in aux)
            aN = tuple(np.asarray(a)[rows, k:L] for a in aux)
        t_hi = t_hi_all[rows]
        t_lo = np.min(cc + zlo[rows, :L] * ss, axis=1)

        def g(t):
            return ((1.0 - F((t[:, None] - cA) / sA, aA)).sum(axis=1)
                    - F((t[:, None] - cN) / sN, aN).sum(axis=1))
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            # Illinois false position on the monotone g (g(t_lo) = k > 0 >= g(t_hi)),
            # with a bisection step whenever the bracket fails to halve.
            lo, hi = t_lo.copy(), t_hi.copy()
            glo, ghi = g(lo), g(hi)
            done = ghi >= 0                       # no crossing possible: mu = E X(t_hi) = 0
            side = np.zeros(rows.size)
            t = np.where(done, hi, 0.5 * (lo + hi))
            for it in range(n_iter):
                width0 = hi - lo
                denom = glo - ghi
                tf = np.where(denom > 0, (lo * (-ghi) + hi * glo) / np.where(denom > 0, denom, 1.0), 0.5 * (lo + hi))
                tf = np.clip(tf, lo, hi)
                t = np.where(done, t, tf)
                gt = g(t)
                pos = gt > 0
                glo_new = np.where(pos, gt, np.where(side == -1, 0.5 * glo, glo))
                ghi_new = np.where(~pos, gt, np.where(side == 1, 0.5 * ghi, ghi))
                lo = np.where(done, lo, np.where(pos, t, lo))
                hi = np.where(done, hi, np.where(pos, hi, t))
                glo = np.where(done, glo, glo_new)
                ghi = np.where(done, ghi, ghi_new)
                side = np.where(pos, 1.0, -1.0)
                # bisection safeguard every third iteration if shrinkage is poor
                if it % 3 == 2:
                    slow = (~done) & ((hi - lo) > 0.5 * width0)
                    if slow.any():
                        tb = 0.5 * (lo + hi)
                        gb = g(tb)
                        pb = gb > 0
                        lo = np.where(slow & pb, tb, lo)
                        glo = np.where(slow & pb, gb, glo)
                        hi = np.where(slow & ~pb, tb, hi)
                        ghi = np.where(slow & ~pb, gb, ghi)
                conv = (np.abs(gt) <= 1e-12 * max(k, 1)) | ((hi - lo) <= 1e-15 * (np.abs(hi) + np.abs(lo) + 1e-300))
                done = done | conv
                if done.all():
                    break
            mu[rows] = (1.0 - F((t[:, None] - cA) / sA, aA)).sum(axis=1)
    return mu


def _gauss_F(z, aux=None):
    return ndtr(z)


GAUSS_SD_SUPPORT = (-_GAUSS_ZMAX_SD, _GAUSS_ZMAX_SD)


def _column_laws(model: CouplingModel, U: np.ndarray, m: int):
    """Per-column laws b~_l = c_l + s_l zeta for sorted A-rows U: returns
    (c, s, F, aux, col_support) for mu_balance / exact_loss. With the floor
    option the law is truncated to b~ >= 0 (z >= z0 = -c/s) and renormalized."""
    X, unit = pool_coords(U, model.pool, m)
    c = U + unit * model.location(X)
    s = np.maximum(unit * model.scale(X), 1e-300)
    aux = model.prepare(X)
    csup = model.column_support(aux)
    if not model.meta.get("floor", False):
        return c, s, model.F, aux, csup
    zlo, zhi = csup if csup is not None else model.zsupport()
    zlo = np.broadcast_to(zlo, c.shape)
    zhi = np.broadcast_to(zhi, c.shape)
    z0 = np.minimum(-c / s, zhi)
    F0 = np.clip(model.F(z0, aux), 0.0, 1.0 - 1e-12)
    base = model.F
    aux2 = (aux if aux is not None else ()) + (z0, F0)
    n_base = 0 if aux is None else len(aux)

    def Ff(z, a):
        a0 = None if n_base == 0 else a[:n_base]
        zz0, ff0 = a[n_base], a[n_base + 1]
        return np.where(z < zz0, 0.0, (base(z, a0) - ff0) / (1.0 - ff0))
    return c, s, Ff, aux2, (np.maximum(zlo, z0), zhi)


def predict_p1(DA: np.ndarray, anchors: np.ndarray, ks, model: CouplingModel, *,
               chunk: int = 1024, return_mu: bool = False) -> dict:
    """Predicted s_N(k) = 1 - mean_i mu_i / k on the given anchors. Uses only
    D_A (the anchors' A-rows) and the fitted coupling."""
    m = DA.shape[0]
    anchors = np.asarray(anchors, dtype=np.int64)
    ks = [int(k) for k in ks]
    mus = {k: [] for k in ks}
    for lo in range(0, anchors.size, chunk):
        U = anchor_rows(DA, anchors[lo:lo + chunk])
        X, unit = pool_coords(U, model.pool, m)
        c, s, F, aux, csup = _column_laws(model, U, m)
        for k in ks:
            mus[k].append(mu_balance(c, s, k, F, model.zsupport(), aux=aux, col_support=csup))
    out = {}
    for k in ks:
        mu = np.concatenate(mus[k])
        out[k] = {"s_pred": float(1.0 - mu.mean() / k), "mu_mean": float(mu.mean())}
        if return_mu:
            out[k]["mu"] = mu
    return out


# ----------------------------------------------------------------------------
# Exact expected discrepancy under independence (diagnostic "P1x", not P1)
# ----------------------------------------------------------------------------

def _capped_add(P, p):
    """Count distribution P (..., K+1), last bin = '>= K', after adding one
    Bernoulli(p) (p shaped like P[..., 0])."""
    pe = p[..., None]
    out = P * (1.0 - pe)
    out[..., 1:] += P[..., :-1] * pe
    out[..., -1] += P[..., -1] * p
    return out


def exact_loss(c: np.ndarray, s: np.ndarray, k: int, F, zsupport, *, aux=None, col_support=None,
               n_grid: int = 400, block: int = 32, max_cols: int | None = None) -> np.ndarray:
    """E|N_A(i) \\ N_B(i)| per row, EXACT (up to the t-grid) for independent
    b_l = c_l + s_l zeta_l: sum over j in N_A of
        P(rank_B(j) > k) = int f_j(t) P(#{l != j : b_l < t} >= k) dt,
    with the Poisson-binomial count computed by dynamic programming (capped
    at k). Diagnostic only: SPEC P1 is the balance-threshold mu (mu_balance);
    Thm 2 bounds |E|N_A \\ N_B| - mu| by sqrt(mu/2)."""
    n_a, n_c = c.shape
    if k >= n_c:
        return np.zeros(n_a)
    if col_support is None:
        zlo, zhi = zsupport
    else:
        zlo, zhi = col_support
    zlo = np.broadcast_to(zlo, c.shape)
    zhi = np.broadcast_to(zhi, c.shape)
    s = np.maximum(s, 1e-300)
    t_hi_all = np.max(c[:, :k] + zhi[:, :k] * s[:, :k], axis=1)
    t_lo_all = np.min(c[:, :k] + zlo[:, :k] * s[:, :k], axis=1)
    need = (c[:, k:] + zlo[:, k:] * s[:, k:]) < t_hi_all[:, None]
    any_need = need.any(axis=1)
    L_all = np.where(any_need, n_c - np.argmax(need[:, ::-1], axis=1), k)
    if max_cols is not None and np.median(L_all) > max_cols:
        return np.full(n_a, np.nan)
    order = np.argsort(L_all, kind="stable")
    out = np.empty(n_a)
    g_edges = np.linspace(0.0, 1.0, n_grid + 1)
    g_mid = 0.5 * (g_edges[1:] + g_edges[:-1])
    lo_b = 0
    while lo_b < n_a:
        # bound the (rows, grid, columns) temporaries to ~2e7 elements
        L0 = int(L_all[order[lo_b]])
        nb = int(max(1, min(block, 2e7 // (n_grid * max(L0, 1)))))
        rows = order[lo_b:lo_b + nb]
        lo_b += nb
        L = int(L_all[rows].max())
        cc, ss = c[rows, :L], s[rows, :L]
        W = (t_hi_all[rows] - t_lo_all[rows])[:, None]
        te = t_lo_all[rows][:, None] + W * g_edges[None, :]          # (b, G+1)
        tm = t_lo_all[rows][:, None] + W * g_mid[None, :]            # (b, G)
        sl = lambda a, j0, j1: None if a is None else tuple(np.asarray(x)[rows, j0:j1][:, None, :] for x in a)
        with np.errstate(invalid="ignore", over="ignore"):
            # non-neighbour count M(t) at the midpoints
            P = np.zeros((rows.size, n_grid, k + 1))
            P[..., 0] = 1.0
            if L > k:
                FN = F((tm[:, :, None] - cc[:, None, k:]) / ss[:, None, k:], sl(aux, k, L))
                for l in range(FN.shape[2]):
                    P = _capped_add(P, FN[:, :, l])
            FA_m = F((tm[:, :, None] - cc[:, None, :k]) / ss[:, None, :k], sl(aux, 0, k))   # (b, G, k)
            FA_e = F((te[:, :, None] - cc[:, None, :k]) / ss[:, None, :k], sl(aux, 0, k))   # (b, G+1, k)
        dF = np.diff(FA_e, axis=1)                                    # (b, G, k)
        tot = np.zeros(rows.size)
        for j in range(k):
            Q = P
            for jj in range(k):
                if jj != j:
                    Q = _capped_add(Q, FA_m[:, :, jj])
            ge_k = Q[..., k]                                          # P(N_-j(t) >= k)
            tot += np.sum(dF[:, :, j] * ge_k, axis=1)
        out[rows] = tot
    return out


def predict_p1_exact(DA: np.ndarray, anchors: np.ndarray, ks, model: CouplingModel, *,
                     chunk: int = 1024, max_cols: int | None = 400) -> dict:
    """Diagnostic 'P1x': exact expected s_N under the fitted coupling."""
    m = DA.shape[0]
    anchors = np.asarray(anchors, dtype=np.int64)
    ks = [int(k) for k in ks]
    res = {k: [] for k in ks}
    for lo in range(0, anchors.size, chunk):
        U = anchor_rows(DA, anchors[lo:lo + chunk])
        c, s, F, aux, csup = _column_laws(model, U, m)
        for k in ks:
            res[k].append(exact_loss(c, s, k, F, model.zsupport(), aux=aux, col_support=csup, max_cols=max_cols))
    return {k: {"s_pred": float(1.0 - np.concatenate(res[k]).mean() / k)} for k in ks}


def predict_gauss_const_exact(DA: np.ndarray, anchors: np.ndarray, ks, sigma: float, *,
                              chunk: int = 1024, max_cols: int | None = 400) -> dict:
    anchors = np.asarray(anchors, dtype=np.int64)
    ks = [int(k) for k in ks]
    res = {k: [] for k in ks}
    for lo in range(0, anchors.size, chunk):
        U = anchor_rows(DA, anchors[lo:lo + chunk])
        S = np.full_like(U, sigma)
        for k in ks:
            res[k].append(exact_loss(U, S, k, _gauss_F, GAUSS_SD_SUPPORT, max_cols=max_cols))
    return {k: {"s_pred": float(1.0 - np.concatenate(res[k]).mean() / k)} for k in ks}


def predict_gauss_const(DA: np.ndarray, anchors: np.ndarray, ks, sigma: float, *,
                        chunk: int = 1024) -> dict:
    """Crossing law with b = u + sigma * N(0,1), independent (exact model of
    P-Gauss with the true sigma = 'oracle'; with sigma estimated = null N1)."""
    anchors = np.asarray(anchors, dtype=np.int64)
    ks = [int(k) for k in ks]
    mus = {k: [] for k in ks}
    for lo in range(0, anchors.size, chunk):
        U = anchor_rows(DA, anchors[lo:lo + chunk])
        s = np.full_like(U, sigma)
        for k in ks:
            mus[k].append(mu_balance(U, s, k, _gauss_F, GAUSS_SD_SUPPORT))
    out = {}
    for k in ks:
        mu = np.concatenate(mus[k])
        out[k] = {"s_pred": float(1.0 - mu.mean() / k), "mu_mean": float(mu.mean())}
    return out


# ----------------------------------------------------------------------------
# P2 dispersion law
# ----------------------------------------------------------------------------

@dataclass
class IsoDispersion:
    xc: np.ndarray
    mad: np.ndarray
    lo_slope: float
    hi_slope: float
    x_floor: float

    def scale(self, u):
        with np.errstate(divide="ignore"):
            x = np.maximum(np.log(u), self.x_floor)
        lm = np.log(self.mad)
        out = np.interp(x, self.xc, lm)
        out = np.where(x < self.xc[0], lm[0] + self.lo_slope * (x - self.xc[0]), out)
        out = np.where(x > self.xc[-1], lm[-1] + self.hi_slope * (x - self.xc[-1]), out)
        return np.exp(out)


def fit_iso_dispersion(DA: np.ndarray, DB: np.ndarray, fit_mask: np.ndarray,
                       n_bins: int | None = None, min_window: int = 32) -> IsoDispersion:
    """Row-isotonic residual dispersion in A-units on fit rows (H1 x H1).

    Residual e = b - g_i(u) with g_i the in-sample row isotonic fit; converted
    to A-units by dividing by the local band slope (an isotonic step function
    has no usable derivative). MAD pooled against log u."""
    from .quotients import iso_fit_sorted
    idx = np.flatnonzero(fit_mask)
    U, B = sorted_rows(DA[np.ix_(idx, idx)], DB[np.ix_(idx, idx)])
    _, C = coupling_residuals(U, B, min_window=min_window, studentize=False)
    E = np.empty_like(U)
    for i in range(U.shape[0]):
        _, _, f = iso_fit_sorted(U[i], B[i])
        E[i] = (B[i] - f) / C[i]
    pos = U > 0
    x = np.log(U[pos])
    e = E[pos]
    if n_bins is None:
        n_bins = int(np.clip(x.size // 5000, 12, 40))
    o = np.argsort(x, kind="stable")
    edges = np.linspace(0, o.size, n_bins + 1).astype(np.int64)
    xc, mad, cnt = [], [], []
    for b in range(n_bins):
        sel = o[edges[b]:edges[b + 1]]
        if sel.size < 20:
            continue
        ee = e[sel]
        xc.append(x[sel].mean())
        mad.append(max(np.mean(np.abs(ee - ee.mean())), 1e-300))
        cnt.append(sel.size)
    xc, mad, cnt = map(np.asarray, (xc, mad, cnt))
    nt = max(3, int(round(0.25 * xc.size)))
    return IsoDispersion(xc, mad, _tail_slope(xc[:nt], np.log(mad[:nt]), cnt[:nt]),
                         _tail_slope(xc[-nt:], np.log(mad[-nt:]), cnt[-nt:]),
                         float(x.min()) - 25.0)


def predict_p2(DA: np.ndarray, anchors: np.ndarray, ks, disp: IsoDispersion,
               lb_bias: dict | None = None, lb_min_k: int = 3) -> dict:
    """1 - s_N ~ (p-hat/4) mean_i MAD_iso(D_k,i) / D_k,i (D units).

    p-hat: MacKay-Ghahramani average of the Levina-Bickel MLE at scale
    max(k, lb_min_k) over the predicted anchors' A-rows (A-only information);
    divided by lb_bias[k] if given."""
    U = anchor_rows(DA, anchors)
    out = {}
    for k in ks:
        k = int(k)
        kl = max(k, lb_min_k)
        lb = lb_dimension(U[:, :kl], kl)
        p_hat = lb["mg"]
        if lb_bias is not None and k in lb_bias:
            p_hat = p_hat / lb_bias[k]
        Dk = U[:, k - 1]
        ok = Dk > 0
        delta_D = disp.scale(Dk[ok]) / Dk[ok]
        one_minus = 0.25 * p_hat * float(np.mean(delta_D))
        out[k] = {"s_pred": float(np.clip(1.0 - one_minus, 0.0, 1.0)), "s_pred_raw": float(1.0 - one_minus),
                  "p_hat": float(p_hat),
                  "delta_D_mean": float(np.mean(delta_D))}
    return out


# ----------------------------------------------------------------------------
# Nulls
# ----------------------------------------------------------------------------

def null_n0(sN_fit: dict, ks) -> dict:
    return {int(k): {"s_pred": float(sN_fit[int(k)])} for k in ks if int(k) in sN_fit}


def fit_n1(DA: np.ndarray, DB: np.ndarray, fit_mask: np.ndarray) -> dict:
    """Global affine fit on fit-pairs (H1 x H1, upper triangle);
    sigma_A = RMS residual / slope."""
    x = util.upper_values_masked(DA, fit_mask)
    y = util.upper_values_masked(DB, fit_mask)
    mx, my = x.mean(), y.mean()
    xc = x - mx
    a = float(np.dot(xc, y - my) / np.dot(xc, xc))
    b = float(my - a * mx)
    r = y - (a * x + b)
    rms = float(np.sqrt(np.mean(r * r)))
    return {"slope": a, "intercept": b, "rms_B": rms, "sigma_A": rms / a if a > 0 else float("inf")}


def predict_n1(DA: np.ndarray, anchors: np.ndarray, ks, n1: dict) -> dict:
    return predict_gauss_const(DA, anchors, ks, n1["sigma_A"])


_RHO_GRID = np.linspace(-0.5, 0.995, 300)


def _copula_logZ(q0: float) -> np.ndarray:
    z0 = ndtri(q0)
    return np.array([np.log(1 - 2 * q0 + multivariate_normal([0, 0], [[1, g], [g, 1]]).cdf([z0, z0]))
                     for g in _RHO_GRID])


def fit_n2(DA: np.ndarray, DB: np.ndarray, fit_mask: np.ndarray, q0: float = 0.1) -> np.ndarray:
    """Per-anchor rho_i (fit rows, fit columns) by profile likelihood of a
    bivariate normal truncated to the bulk {z_A > z0, z_B > z0} on a rho grid."""
    idx = np.flatnonzero(fit_mask)
    DAs, DBs = DA[np.ix_(idx, idx)], DB[np.ix_(idx, idx)]
    n = idx.size - 1
    logZ = _copula_logZ(q0)
    g = _RHO_GRID
    rhos = np.empty(idx.size)
    ar = np.arange(idx.size)
    for i in range(idx.size):
        keep = ar != i
        a = DAs[i, keep]
        b = DBs[i, keep]
        ra = np.empty(n)
        rb = np.empty(n)
        ra[np.argsort(a, kind="stable")] = np.arange(1, n + 1)
        rb[np.argsort(b, kind="stable")] = np.arange(1, n + 1)
        msk = (ra > q0 * n) & (rb > q0 * n)
        zA = ndtri((ra[msk] - 0.5) / n)
        zB = ndtri((rb[msk] - 0.5) / n)
        s11, s22, s12, nn = np.dot(zA, zA), np.dot(zB, zB), np.dot(zA, zB), zA.size
        ll = -0.5 * nn * np.log(1 - g ** 2) - (s11 + s22 - 2 * g * s12) / (2 * (1 - g ** 2)) - nn * logZ
        rhos[i] = g[int(np.argmax(ll))]
    return rhos


def predict_n2(rhos: np.ndarray, m: int, ks) -> dict:
    out = {}
    for k in ks:
        q = int(k) / (m - 1)
        z = ndtri(q)
        lam = np.array([multivariate_normal([0, 0], [[1, gg], [gg, 1]]).cdf([z, z]) / q for gg in _RHO_GRID])
        out[int(k)] = {"s_pred": float(np.mean(np.interp(rhos, _RHO_GRID, lam)))}
    return out


# ----------------------------------------------------------------------------
# Cross-fitted protocol on nested pools (S1.1 / G0)
# ----------------------------------------------------------------------------

def crossfit_predict(DA_full: np.ndarray, DB_full: np.ndarray, h1_full: np.ndarray, *,
                     m_fit: int, m_list, ks, variants=P1_CANDIDATES, with_nulls: bool = True,
                     oracle_sigma: float | None = None, lb_bias: dict | None = None,
                     exact_ks=(), exact_variants=()) -> dict:
    """Fit on H1 x H1 of the m_fit prefix; predict s_N on H2 anchors of every
    prefix m in m_list. Pools are prefixes of the item order of D*_full
    (nested). Observed s_N is computed on the same H2 anchors.

    Returns {m: {k: {'obs', 'obs_sd', 'n_anchors', 'chance', <predictor>: s_pred}}}
    plus '_fit' diagnostics."""
    from . import structural as st
    m_list = [int(m) for m in m_list]
    fit_mask = h1_full[:m_fit]
    DAf, DBf = DA_full[:m_fit, :m_fit], DB_full[:m_fit, :m_fit]
    models = {v: fit_coupling(DAf, DBf, fit_mask, variant=v) for v in variants}
    out = {"_fit": {v: models[v].summary() for v in variants}}
    if with_nulls:
        n1 = fit_n1(DAf, DBf, fit_mask)
        rhos = fit_n2(DAf, DBf, fit_mask)
        disp = fit_iso_dispersion(DAf, DBf, fit_mask)
        nAf, _ = st.knn_table(DAf, max(ks) + 1)
        nBf, _ = st.knn_table(DBf, max(ks) + 1)
        h1i = np.flatnonzero(fit_mask)
        sN_fit = {}
        for k in ks:
            if k <= m_fit - 2:
                sN_fit[int(k)] = float(st.overlap_counts(nAf[h1i], nBf[h1i], k).mean() / k)
        out["_fit"]["N1"] = n1
        out["_fit"]["N2_rho_median"] = float(np.median(rhos))
    for m in m_list:
        DA, DB = DA_full[:m, :m], DB_full[:m, :m]
        anchors = np.flatnonzero(~h1_full[:m])
        kk = [int(k) for k in ks if k <= m - 2]
        kmax = max(kk) + 1
        nA, _ = st.knn_table(DA, kmax)
        nB, _ = st.knn_table(DB, kmax)
        cell = {}
        for k in kk:
            ov = st.overlap_counts(nA[anchors], nB[anchors], k) / k
            cell[k] = {"obs": float(ov.mean()), "obs_sd": float(ov.std(ddof=1)),
                       "n_anchors": int(anchors.size), "chance": k / (m - 1)}
        for v in variants:
            pr = predict_p1(DA, anchors, kk, models[v])
            for k in kk:
                cell[k][v] = pr[k]["s_pred"]
        if oracle_sigma is not None:
            pr = predict_gauss_const(DA, anchors, kk, oracle_sigma)
            for k in kk:
                cell[k]["oracle"] = pr[k]["s_pred"]
        ek = [k for k in exact_ks if k in kk]
        if ek:
            for v in exact_variants:
                pr = predict_p1_exact(DA, anchors, ek, models[v])
                for k in ek:
                    cell[k][f"P1x:{v}"] = pr[k]["s_pred"]
            if oracle_sigma is not None:
                pr = predict_gauss_const_exact(DA, anchors, ek, oracle_sigma)
                for k in ek:
                    cell[k]["oracle_x"] = pr[k]["s_pred"]
        if with_nulls:
            p1 = predict_n1(DA, anchors, kk, n1)
            p2 = predict_p2(DA, anchors, kk, disp, lb_bias=lb_bias)
            pn2 = predict_n2(rhos, m, kk)
            for k in kk:
                cell[k]["N0"] = sN_fit.get(k, float("nan"))
                cell[k]["N1"] = p1[k]["s_pred"]
                cell[k]["N2"] = pn2[k]["s_pred"]
                cell[k]["P2"] = p2[k]["s_pred"]
                cell[k]["P2_p_hat"] = p2[k]["p_hat"]
        out[m] = cell
    return out
