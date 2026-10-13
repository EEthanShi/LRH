"""PREREG v1.1 (prereg/LEAD_DECISIONS_v1.1.md, decision D4): anchor-local
estimators of the crossing-law predictor P1, the D2 exact-discrepancy mode,
and the v1.1 nulls. Role s11-select, 2026-10-08.

This module does NOT modify hrhv2/predict.py (frozen v1.0, sha256 a17964b4...).
It imports from it: mu_balance (SPEC balance-threshold mu), exact_loss
(exact E|N_A \\ N_B| by dynamic programming), fit_coupling / _column_laws
(the frozen v1.0 P1 = variant iv), fit_n1 and the Gaussian CDF (null N1).

Protocol (D4.2: "the same anchor at a different m")
---------------------------------------------------
Nested pools are prefixes of one item order (pool positions 0, 1, ...).
m_f is the fit size (1024 on COCO and on the plants; 512 on S_sel, where only
1023 rows exist). Items are split into halves H1/H2 by hrhv2.util.hash_half.
The *predicted anchors* are the H2 items of the m_f prefix; at m < m_f only
those inside the m prefix. For a predicted anchor i the v1.1 estimators read
ONLY
  * the anchor's own A- and B-row at m_f (distances to the other m_f - 1
    items of the m_f prefix), and
  * pooled summaries of those same rows over the predicted anchors (prior of
    variant iii),
and, at the prediction size m, the anchor's A-row (distances to the other
items of the m prefix). No B entry at any m != m_f is read (tested).
Observed s_N(m, k) is the mean over the predicted anchors of
|N_k^A(i) & N_k^B(i)| / k, neighbours searched in the whole m prefix, SPEC tie
rule (D_ij, pool position). Cells at m = m_f are in-sample and never scored.

Predicted rank band and fit window of cell (m, k)
-------------------------------------------------
With n_f = m_f - 1 columns in a fit row, the k-NN boundary at size m sits near
fit-rank rho = k * n_f / (m - 1). Ranks are 1-based positions in the fit row
sorted by the tie rule.
  Band(m, k)   = {r : rho / 2 <= r <= 2 rho}            (may be empty, rho < 1/2)
  Window(m, k) = {r : rho / 8 <= r <= 8 rho} restricted to usable ranks
                 (variants i/iii: ranks outside the band; ii: all ranks),
                 extended upward by whole ranks (then downward) until it
                 holds N_MIN = 64 usable ranks.

Variants (D4.1; at most 4, pre-listed)
--------------------------------------
(i)   'i-LSlog'  per-anchor multiplicative residual (log-distance
      location-scale) model fitted on Window \\ Band:
          log b = x + a_i + beta_i (x - xbar_i) + exp(c_i + gamma_i (x - xbar_i)) zeta,
      x = log u. (a, beta): OLS of log(b/u) on x. (c, gamma): OLS of
      log|residual| on x (Harvey's multiplicative-heteroscedasticity
      estimator; its constant offset E log|zeta| is absorbed by the shape).
      zeta: the anchor's own standardized residuals, variance-corrected
      Gaussian KDE (<= 101 quantile atoms, Silverman bandwidth). Below the
      window, location and log-scale are extrapolated linearly in x (the
      anchor at smaller scale, m > m_f); above the window both deviation terms
      are held at their value at the window's top. Non-positive distances
      (exact duplicates; the additive P-Gauss plant) enter the logs at half the
      row's smallest positive value.
(ii)  'ii-IsoXF' per-anchor row-isotonic link with even/odd candidate-level
      cross-fitting: g_E is the isotonic fit of b on u over the even fit-ranks,
      g_O over the odd fit-ranks. The residual of an odd-rank column is
      b - g_E(u), of an even-rank column b - g_O(u). Dispersion in B units by
      the same Harvey log-linear model on Window (no band exclusion:
      cross-fitting is this variant's guard); b = g(u) + exp(c + gamma (x -
      xbar)) zeta. At prediction a column of the fit row is centred by the link
      of the other parity, a column outside the fit row by the mean of both.
      Links are linear interpolations of the fitted values; below the smallest
      fitted u they are extrapolated linearly with the non-negative slope of
      the first 8 fitted points; above the largest, held constant.
(iii) 'iii-LSlog-EB' variant (i) with the dispersion parameters (c_i, gamma_i)
      shrunk toward their precision-weighted mean over the predicted anchors
      (empirical Bayes, normal-normal, method-of-moments prior variance, each
      parameter separately); zeta is re-standardized with the shrunk scale.
      The base is (i), not (ii): (i)'s dispersion is dimensionless (log
      domain), so pooling it across anchors does not mix local scales, which
      is the v1.0 failure D4 diagnoses; (ii)'s dispersion is in B units.
(iv)  'iv-v10' the frozen v1.0 P1 (hrhv2.predict 'P1-med-B-r8'), coupling
      pooled over H1 x H1 of the m_f prefix. Reference.

Computation modes (D2)
----------------------
'd2' : exact expected discrepancy E|N_A \\ N_B| for k <= EXACT_K_MAX = 3
       (k = 1: closed product form ``exact_loss_k1``; k = 2, 3:
       hrhv2.predict.exact_loss), SPEC balance-threshold mu for k > 3.
'mu' : SPEC mu for every k (diagnostic; the literal v1.0 functional).
The same mode is applied to every predictor and every model-based null
(N1, oracle). Exact for k >= 10 is not used: its cost is O(G (L k + k^3)) per
anchor (L = contributing columns, ~m in the saturated regime), and on P-Gauss
the mu-law error at k >= 10 is <= 0.006 (G0 report).

Nulls (v1.1)
------------
N0   s_N of the predicted anchors at m_f (their own rows; like-for-like with
     the evaluated anchor set). 'N0H1' = the v1.0 definition (H1 anchors at
     m_f), reported.
N1   independent additive Gaussian noise in A units, sigma_A = RMS of the
     global affine residual on H1 x H1 at m_f / slope (hrhv2.predict.fit_n1),
     computed in the same mode as P1. D4.8: if the slope is <= 0 or
     undefined, N1 := N0. 'N1mu' (mu-law, the v1.0 computation) is reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.special import ndtr

from . import predict as p10
from . import structural as st

VARIANTS = ("i-LSlog", "ii-IsoXF", "iii-LSlog-EB", "iv-v10")
# a-priori tie-break order (simplest first) used by the selection rule
PREFERENCE = ("i-LSlog", "iii-LSlog-EB", "ii-IsoXF", "iv-v10")
V10_VARIANT = p10.FROZEN_P1_VARIANT          # 'P1-med-B-r8'
MODES = ("d2", "mu")

BAND_W = 2.0
WIN_W = 8.0
N_MIN = 64
EXACT_K_MAX = 3
EXACT_GRID = 400
KDE_ATOMS = 101
KDE_GRID = 512
KDE_TAIL = 7.0
BETA_CLIP = 5.0
GAMMA_CLIP = 4.0
LOGSCALE_CLIP = 300.0
LINK_LO_POINTS = 8


# ----------------------------------------------------------------------------
# rows, ranks, kNN
# ----------------------------------------------------------------------------

def sort_rows(DA_rows: np.ndarray, self_pos: np.ndarray, m: int, DB_rows: np.ndarray | None = None):
    """Anchor rows over pool positions 0..m-1 with the anchor itself removed,
    columns ordered by the tie rule (D_A, position).

    Returns (U, COL, B): sorted A values (n_a, m-1), the pool positions of the
    columns (int64) and the aligned B values (or None)."""
    n_a = DA_rows.shape[0]
    Dc = np.array(DA_rows[:, :m], dtype=np.float64, copy=True)
    Dc[np.arange(n_a), self_pos] = np.inf
    O = np.argsort(Dc, axis=1, kind="stable")[:, : m - 1]   # stable on ascending position = key (u, j)
    U = np.take_along_axis(Dc, O, axis=1)
    B = None
    if DB_rows is not None:
        B = np.take_along_axis(np.asarray(DB_rows[:, :m], dtype=np.float64), O, axis=1)
    return U, O.astype(np.int64), B


def knn_rows(D_rows: np.ndarray, self_pos: np.ndarray, m: int, kmax: int) -> np.ndarray:
    """First kmax neighbours (pool positions) of each anchor in the m prefix,
    SPEC tie rule (D_ij, j), self excluded."""
    n_a = D_rows.shape[0]
    K = int(min(kmax, m - 1))
    Dc = np.array(D_rows[:, :m], dtype=np.float64, copy=True)
    Dc[np.arange(n_a), self_pos] = np.inf
    v = np.partition(Dc, K - 1, axis=1)[:, K - 1]
    lt = Dc < v[:, None]
    eq = Dc == v[:, None]
    need = K - lt.sum(axis=1)
    sel = lt | (eq & (np.cumsum(eq, axis=1) <= need[:, None]))
    cols = np.nonzero(sel)[1].reshape(n_a, K)
    vals = np.take_along_axis(Dc, cols, axis=1)
    order = np.lexsort((cols, vals), axis=-1)
    return np.take_along_axis(cols, order, axis=1)


def overlap_fracs(DA_rows, DB_rows, self_pos, m, ks) -> dict:
    """Per-anchor |N_k^A & N_k^B| / k for each k <= m - 2."""
    kk = [int(k) for k in ks if k <= m - 2]
    if not kk:
        return {}
    nA = knn_rows(DA_rows, self_pos, m, max(kk))
    nB = knn_rows(DB_rows, self_pos, m, max(kk))
    return {k: st.overlap_counts(nA, nB, k) / k for k in kk}


def cell_rho(k: int, m: int, m_f: int) -> float:
    return k * (m_f - 1) / (m - 1)


def band_mask(rho: float, n_f: int) -> np.ndarray:
    r = np.arange(1, n_f + 1, dtype=np.float64)
    return (r >= rho / BAND_W) & (r <= rho * BAND_W)


def window_mask(rho: float, n_f: int, exclude: np.ndarray | None = None, n_min: int = N_MIN) -> np.ndarray:
    """Usable ranks in [rho/8, 8 rho], extended upward (then downward) to n_min."""
    r = np.arange(1, n_f + 1, dtype=np.float64)
    usable = np.ones(n_f, dtype=bool) if exclude is None else ~np.asarray(exclude, dtype=bool)
    lo, hi = rho / WIN_W, rho * WIN_W
    up = np.flatnonzero(usable & (r >= lo))
    sel = up[r[up] <= hi]
    if sel.size < n_min:
        sel = up[:n_min]
    if sel.size < n_min:
        down = np.flatnonzero(usable & (r < lo))[::-1]
        sel = np.concatenate((sel, down[: n_min - sel.size]))
    out = np.zeros(n_f, dtype=bool)
    out[sel] = True
    return out


def pos_floor_log(V: np.ndarray) -> np.ndarray:
    """log V with non-positive entries replaced by half the row's smallest
    positive value (duplicates; additive plants)."""
    V = np.asarray(V, dtype=np.float64)
    pos = V > 0
    mn = np.where(pos, V, np.inf).min(axis=1)
    fl = np.where(np.isfinite(mn), 0.5 * mn, 1e-300)
    return np.log(np.where(pos, V, fl[:, None]))


# ----------------------------------------------------------------------------
# per-anchor standardized-residual law (variance-corrected Gaussian KDE)
# ----------------------------------------------------------------------------

@dataclass
class Shape:
    T: np.ndarray        # (n_a, G) CDF table on each anchor's own uniform grid
    zg0: np.ndarray      # (n_a,) grid start (= zlo)
    dz: np.ndarray       # (n_a,) grid step
    zlo: np.ndarray      # (n_a,) support: F = 0 below
    zhi: np.ndarray      # (n_a,) support: F = 1 above
    h: np.ndarray        # (n_a,) bandwidth

    def __post_init__(self):
        self._Tf = np.ascontiguousarray(self.T).ravel()
        self._G = self.T.shape[1]

    def F(self, z, aux):
        """CDF at z for the anchors given by aux[0] (row indices broadcastable to z)."""
        r = aux[0]
        t = (np.asarray(z, dtype=np.float64) - self.zg0[r]) / self.dz[r]
        t = np.clip(t, 0.0, self._G - 1.000001)
        i = t.astype(np.int64)
        f = t - i
        base = r * self._G + i
        return self._Tf[base] * (1.0 - f) + self._Tf[base + 1] * f

    def moments(self):
        """Mean and variance of each tabulated law (tests)."""
        zg = self.zg0[:, None] + self.dz[:, None] * np.arange(self._G)[None, :]
        dF = np.diff(self.T, axis=1)
        zm = 0.5 * (zg[:, 1:] + zg[:, :-1])
        mu = (dF * zm).sum(1)
        var = (dF * (zm - mu[:, None]) ** 2).sum(1)
        return mu, var


def build_shape(Z: np.ndarray, n_atoms: int = KDE_ATOMS, n_grid: int = KDE_GRID, tail: float = KDE_TAIL) -> Shape:
    """Per-row variance-corrected Gaussian KDE of the standardized residuals Z
    (n_a, n). Atoms = Z (or n_atoms equal-probability quantiles if n > n_atoms),
    shrunk toward their mean by 1/sqrt(1 + h^2/var) so that the KDE keeps the
    atoms' variance; h = 0.9 min(sd, IQR/1.349) n^(-1/5) (Silverman)."""
    Z = np.asarray(Z, dtype=np.float64)
    n_a, n = Z.shape
    Zs = np.sort(Z, axis=1)
    if n > n_atoms:
        pos = (np.arange(n_atoms) + 0.5) / n_atoms * n - 0.5
        lo = np.floor(pos).astype(np.int64)
        f = pos - lo
        hi = np.minimum(lo + 1, n - 1)
        atoms = Zs[:, lo] * (1.0 - f) + Zs[:, hi] * f
    else:
        atoms = Zs
    mu = atoms.mean(axis=1)
    var = atoms.var(axis=1)
    sd = np.sqrt(var)
    q25 = np.quantile(Zs, 0.25, axis=1)
    q75 = np.quantile(Zs, 0.75, axis=1)
    spread = np.minimum(sd, (q75 - q25) / 1.349)
    spread = np.where(spread > 0, spread, sd)
    spread = np.where(spread > 0, spread, 1e-6 * (1.0 + np.abs(mu)))
    h = 0.9 * spread * float(n) ** (-0.2)
    fac = np.where(var > 0, 1.0 / np.sqrt(1.0 + h ** 2 / np.where(var > 0, var, 1.0)), 1.0)
    at = mu[:, None] + (atoms - mu[:, None]) * fac[:, None]
    zlo = at.min(axis=1) - tail * h
    zhi = at.max(axis=1) + tail * h
    width = np.maximum(zhi - zlo, 1e-12)
    zhi = zlo + width
    g = np.linspace(0.0, 1.0, n_grid)
    T = np.empty((n_a, n_grid))
    step = max(1, int(4e6 // (n_grid * at.shape[1])))
    for a0 in range(0, n_a, step):
        a1 = min(n_a, a0 + step)
        zg = zlo[a0:a1, None] + width[a0:a1, None] * g[None, :]
        T[a0:a1] = ndtr((zg[:, :, None] - at[a0:a1, None, :]) / h[a0:a1, None, None]).mean(axis=2)
    T[:, 0] = 0.0
    T[:, -1] = 1.0
    T = np.maximum.accumulate(T, axis=1)
    return Shape(T=T, zg0=zlo, dz=width / (n_grid - 1), zlo=zlo, zhi=zhi, h=h)


# ----------------------------------------------------------------------------
# small regression helpers (row-wise, vectorized over anchors)
# ----------------------------------------------------------------------------

def _ols_rows(xc: np.ndarray, y: np.ndarray, sxx: np.ndarray):
    """Row-wise OLS on centred xc: returns (intercept = mean y, slope)."""
    ybar = y.mean(axis=1)
    num = (xc * (y - ybar[:, None])).sum(axis=1)
    slope = np.where(sxx > 0, num / np.where(sxx > 0, sxx, 1.0), 0.0)
    return ybar, slope


def _harvey(xc: np.ndarray, E: np.ndarray, sxx: np.ndarray):
    """log|E| = c + gamma xc + noise (row-wise). Returns c, gamma, se_c, se_gamma."""
    A = np.abs(E)
    med = np.median(A, axis=1)
    fl = np.where(med > 0, 1e-12 * med, 1e-300)
    L = np.log(np.maximum(A, fl[:, None]))
    c, gamma = _ols_rows(xc, L, sxx)
    gamma = np.clip(gamma, -GAMMA_CLIP, GAMMA_CLIP)
    res = L - c[:, None] - gamma[:, None] * xc
    n = xc.shape[1]
    s2 = (res ** 2).sum(axis=1) / max(n - 2, 1)
    se_c = np.sqrt(s2 / n)
    se_g = np.sqrt(s2 / np.where(sxx > 0, sxx, np.inf))
    return c, gamma, se_c, se_g


def eb_shrink(theta: np.ndarray, se: np.ndarray):
    """Normal-normal empirical Bayes: shrink theta_i toward the precision-weighted
    mean with weight tau^2 / (tau^2 + se_i^2), tau^2 by the method of moments."""
    theta = np.asarray(theta, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    ok = np.isfinite(theta) & np.isfinite(se) & (se > 0)
    if ok.sum() < 3:
        return theta.copy(), {"pooled": float("nan"), "tau2": float("nan"), "mean_w": 1.0}
    w0 = 1.0 / se[ok] ** 2
    tbar = float(np.sum(w0 * theta[ok]) / np.sum(w0))
    tau2 = max(float(np.var(theta[ok]) - np.mean(se[ok] ** 2)), 0.0)
    w = np.where(ok, tau2 / (tau2 + np.where(ok, se, 1.0) ** 2), 0.0)
    out = tbar + w * (np.where(ok, theta, tbar) - tbar)
    return out, {"pooled": tbar, "tau2": tau2, "mean_w": float(np.mean(w[ok]))}


# ----------------------------------------------------------------------------
# fit rows (the predicted anchors' own rows at m_f)
# ----------------------------------------------------------------------------

@dataclass
class FitRows:
    U: np.ndarray        # (n_a, n_f) sorted A distances at m_f
    B: np.ndarray        # (n_a, n_f) aligned B distances
    COL: np.ndarray      # (n_a, n_f) pool positions of the columns
    logU: np.ndarray
    logB: np.ndarray
    m_f: int

    @property
    def n_f(self):
        return self.U.shape[1]


def make_fit_rows(DA_rows: np.ndarray, DB_rows: np.ndarray, anchor_pos: np.ndarray, m_f: int) -> FitRows:
    U, COL, B = sort_rows(DA_rows, anchor_pos, m_f, DB_rows)
    return FitRows(U=U, B=B, COL=COL, logU=pos_floor_log(U), logB=pos_floor_log(B), m_f=int(m_f))


# ----------------------------------------------------------------------------
# variant (i) / (iii): multiplicative (log-distance) location-scale
# ----------------------------------------------------------------------------

@dataclass
class LocScaleFit:
    kind: str                 # 'log' (i, iii) or 'iso' (ii)
    xbar: np.ndarray
    x_top: np.ndarray         # largest x in the window (upper clamp)
    c: np.ndarray
    gamma: np.ndarray
    shape: Shape
    a: np.ndarray | None = None
    beta: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def fit_lslog(Fr: FitRows, wmask: np.ndarray, *, shrink: bool = False) -> LocScaleFit:
    X = Fr.logU[:, wmask]
    Y = Fr.logB[:, wmask]
    xbar = X.mean(axis=1)
    xc = X - xbar[:, None]
    sxx = (xc ** 2).sum(axis=1)
    d = Y - X
    a, beta = _ols_rows(xc, d, sxx)
    beta = np.clip(beta, -BETA_CLIP, BETA_CLIP)
    E = d - a[:, None] - beta[:, None] * xc
    c, gamma, se_c, se_g = _harvey(xc, E, sxx)
    meta = {"n_window": int(wmask.sum()), "median_beta": float(np.median(beta)),
            "median_gamma_raw": float(np.median(gamma)), "median_c_raw": float(np.median(c))}
    if shrink:
        c, ic = eb_shrink(c, se_c)
        gamma, ig = eb_shrink(gamma, se_g)
        gamma = np.clip(gamma, -GAMMA_CLIP, GAMMA_CLIP)
        meta.update({"eb_c": ic, "eb_gamma": ig})
    meta["median_gamma"] = float(np.median(gamma))
    scale = np.exp(np.clip(c[:, None] + gamma[:, None] * xc, -LOGSCALE_CLIP, LOGSCALE_CLIP))
    Z = E / scale
    return LocScaleFit(kind="log", xbar=xbar, x_top=X.max(axis=1), c=c, gamma=gamma,
                       shape=build_shape(Z), a=a, beta=beta, meta=meta)


def laws_lslog(fit: LocScaleFit, ri: np.ndarray, U: np.ndarray):
    """Column laws on the log-B scale for anchors ri (indices into the fit) and
    their sorted A-rows U at the prediction size."""
    X = pos_floor_log(U)
    xb = fit.xbar[ri][:, None]
    xc = X - xb
    xcl = np.minimum(xc, fit.x_top[ri][:, None] - xb)
    cen = X + fit.a[ri][:, None] + fit.beta[ri][:, None] * xcl
    sc = np.exp(np.clip(fit.c[ri][:, None] + fit.gamma[ri][:, None] * xcl, -LOGSCALE_CLIP, LOGSCALE_CLIP))
    aux = (np.broadcast_to(ri[:, None], U.shape),)
    csup = (np.broadcast_to(fit.shape.zlo[ri][:, None], U.shape),
            np.broadcast_to(fit.shape.zhi[ri][:, None], U.shape))
    return cen, sc, fit.shape.F, aux, csup


# ----------------------------------------------------------------------------
# variant (ii): row-isotonic link, even/odd cross-fitted, + dispersion
# ----------------------------------------------------------------------------

@dataclass
class IsoLinks:
    uE: np.ndarray
    gE: np.ndarray
    sE: np.ndarray            # lower-extrapolation slopes
    uO: np.ndarray
    gO: np.ndarray
    sO: np.ndarray
    Exf: np.ndarray           # (n_a, n_f) cross-fitted residuals, B units
    parity: np.ndarray        # (n_a, m_f) int8: -1 = not in row; 0 -> use g_E; 1 -> use g_O
    m_f: int


def _iso_rows(Y: np.ndarray) -> np.ndarray:
    out = np.empty_like(Y)
    for r in range(Y.shape[0]):
        out[r] = isotonic_regression(Y[r], increasing=True).x
    return out


def _lo_slope(u: np.ndarray, g: np.ndarray) -> np.ndarray:
    j = min(LINK_LO_POINTS - 1, u.shape[1] - 1)
    du = u[:, j] - u[:, 0]
    s = np.where(du > 0, (g[:, j] - g[:, 0]) / np.where(du > 0, du, 1.0), 0.0)
    return np.maximum(s, 0.0)


def link_eval(uK: np.ndarray, gK: np.ndarray, sK: np.ndarray, Q: np.ndarray) -> np.ndarray:
    out = np.empty(Q.shape, dtype=np.float64)
    for r in range(Q.shape[0]):
        out[r] = np.interp(Q[r], uK[r], gK[r])
    below = Q < uK[:, :1]
    return np.where(below, gK[:, :1] + sK[:, None] * (Q - uK[:, :1]), out)


def fit_iso_links(Fr: FitRows) -> IsoLinks:
    n_a, n_f = Fr.U.shape
    tE = np.arange(1, n_f, 2)     # 0-based index odd  = even fit-rank
    tO = np.arange(0, n_f, 2)     # 0-based index even = odd fit-rank
    uE, uO = Fr.U[:, tE], Fr.U[:, tO]
    gE, gO = _iso_rows(Fr.B[:, tE]), _iso_rows(Fr.B[:, tO])
    sE, sO = _lo_slope(uE, gE), _lo_slope(uO, gO)
    Exf = np.empty_like(Fr.B)
    Exf[:, tO] = Fr.B[:, tO] - link_eval(uE, gE, sE, Fr.U[:, tO])
    Exf[:, tE] = Fr.B[:, tE] - link_eval(uO, gO, sO, Fr.U[:, tE])
    parity = np.full((n_a, Fr.m_f), -1, dtype=np.int8)
    rows = np.repeat(np.arange(n_a), n_f)
    parity[rows, Fr.COL.ravel()] = np.tile((np.arange(n_f) % 2).astype(np.int8), n_a)
    return IsoLinks(uE=uE, gE=gE, sE=sE, uO=uO, gO=gO, sO=sO, Exf=Exf, parity=parity, m_f=Fr.m_f)


def fit_isoxf(links: IsoLinks, Fr: FitRows, wmask: np.ndarray) -> LocScaleFit:
    X = Fr.logU[:, wmask]
    E = links.Exf[:, wmask]
    xbar = X.mean(axis=1)
    xc = X - xbar[:, None]
    sxx = (xc ** 2).sum(axis=1)
    c, gamma, _, _ = _harvey(xc, E, sxx)
    scale = np.exp(np.clip(c[:, None] + gamma[:, None] * xc, -LOGSCALE_CLIP, LOGSCALE_CLIP))
    Z = E / scale
    meta = {"n_window": int(wmask.sum()), "median_gamma": float(np.median(gamma)), "median_c": float(np.median(c))}
    return LocScaleFit(kind="iso", xbar=xbar, x_top=X.max(axis=1), c=c, gamma=gamma,
                       shape=build_shape(Z), meta=meta)


def iso_centres(links: IsoLinks, ri: np.ndarray, U: np.ndarray, COL: np.ndarray) -> np.ndarray:
    """Cross-fitted link values: other-parity link for fit-row columns, the
    mean of both links for columns outside the fit row."""
    gEv = link_eval(links.uE[ri], links.gE[ri], links.sE[ri], U)
    gOv = link_eval(links.uO[ri], links.gO[ri], links.sO[ri], U)
    par = np.full(U.shape, -1, dtype=np.int8)
    inf = COL < links.m_f
    rr = np.broadcast_to(ri[:, None], U.shape)
    par[inf] = links.parity[rr[inf], COL[inf]]
    return np.where(par == 0, gEv, np.where(par == 1, gOv, 0.5 * (gEv + gOv)))


def laws_isoxf(fit: LocScaleFit, links: IsoLinks, ri: np.ndarray, U: np.ndarray, COL: np.ndarray):
    cen = iso_centres(links, ri, U, COL)
    X = pos_floor_log(U)
    xb = fit.xbar[ri][:, None]
    xcl = np.minimum(X - xb, fit.x_top[ri][:, None] - xb)
    sc = np.exp(np.clip(fit.c[ri][:, None] + fit.gamma[ri][:, None] * xcl, -LOGSCALE_CLIP, LOGSCALE_CLIP))
    aux = (np.broadcast_to(ri[:, None], U.shape),)
    csup = (np.broadcast_to(fit.shape.zlo[ri][:, None], U.shape),
            np.broadcast_to(fit.shape.zhi[ri][:, None], U.shape))
    return cen, sc, fit.shape.F, aux, csup


# ----------------------------------------------------------------------------
# expected discrepancy: exact (k <= 3) or SPEC mu
# ----------------------------------------------------------------------------

def _slice_aux(aux, rows, j0, j1, expand: bool):
    if aux is None:
        return None
    if expand:
        return tuple(np.asarray(a)[rows, j0:j1][:, None, :] for a in aux)
    return tuple(np.asarray(a)[rows, j0:j1] for a in aux)


def exact_loss_k1(c, s, F, aux, csup, *, n_grid: int = EXACT_GRID, max_elems: float = 2e7) -> np.ndarray:
    """E|N_A \\ N_B| for k = 1, exact for independent columns (up to the t-grid):
    P(rank_B(1) > 1) = int f_1(t) [1 - prod_{l >= 2} (1 - F_l(t))] dt,
    midpoint rule on n_grid cells of the support of b_1 (same grid as
    hrhv2.predict.exact_loss, which this equals at k = 1; tested)."""
    n_a, n_c = c.shape
    if n_c <= 1:
        return np.zeros(n_a)
    zlo = np.broadcast_to(csup[0], c.shape)
    zhi = np.broadcast_to(csup[1], c.shape)
    s = np.maximum(s, 1e-300)
    t_lo = c[:, 0] + zlo[:, 0] * s[:, 0]
    t_hi = c[:, 0] + zhi[:, 0] * s[:, 0]
    need = (c[:, 1:] + zlo[:, 1:] * s[:, 1:]) < t_hi[:, None]
    any_need = need.any(axis=1)
    L_all = np.where(any_need, n_c - np.argmax(need[:, ::-1], axis=1), 1)
    order = np.argsort(L_all, kind="stable")
    edges = np.linspace(0.0, 1.0, n_grid + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    out = np.empty(n_a)
    lo_b = 0
    while lo_b < n_a:
        L0 = int(L_all[order[lo_b]])
        nb = int(max(1, min(64, max_elems // (n_grid * max(L0, 1)))))
        rows = order[lo_b:lo_b + nb]
        lo_b += nb
        L = int(L_all[rows].max())
        W = (t_hi[rows] - t_lo[rows])[:, None]
        te = t_lo[rows][:, None] + W * edges[None, :]
        tm = t_lo[rows][:, None] + W * mid[None, :]
        c0, s0 = c[rows, 0:1], s[rows, 0:1]
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            FA = F((te - c0) / s0, _slice_aux(aux, rows, 0, 1, False))
            dF = np.diff(FA, axis=1)
            if L > 1:
                FN = F((tm[:, :, None] - c[rows, None, 1:L]) / s[rows, None, 1:L],
                       _slice_aux(aux, rows, 1, L, True))
                lsurv = np.log1p(-np.minimum(FN, 1.0)).sum(axis=2)
                p_any = -np.expm1(lsurv)
            else:
                p_any = np.zeros_like(dF)
        out[rows] = np.sum(dF * p_any, axis=1)
    return out


def expected_loss(c, s, k: int, F, aux, csup, mode: str) -> np.ndarray:
    """Per-anchor expected |N_A \\ N_B| (mode 'd2') or SPEC mu (mode 'mu')."""
    if mode == "d2" and k <= EXACT_K_MAX:
        if k == 1:
            return exact_loss_k1(c, s, F, aux, csup)
        return p10.exact_loss(c, s, k, F, None, aux=aux, col_support=csup, n_grid=EXACT_GRID)
    if mode not in MODES:
        raise ValueError(mode)
    return p10.mu_balance(c, s, k, F, None, aux=aux, col_support=csup)


def gauss_laws(U: np.ndarray, sigma: float):
    s = np.full(U.shape, float(sigma))
    lo, hi = p10.GAUSS_SD_SUPPORT
    return U, s, p10._gauss_F, None, (np.broadcast_to(lo, U.shape), np.broadcast_to(hi, U.shape))


# ----------------------------------------------------------------------------
# one unit: one direction of one (pair, draw)
# ----------------------------------------------------------------------------

def run_unit(DA_rows: np.ndarray, DB_rows: np.ndarray, anchor_pos: np.ndarray,
             DA_fit: np.ndarray, DB_fit: np.ndarray, h1_fit: np.ndarray, *,
             m_grid, m_f: int, ks, variants=VARIANTS, modes=MODES,
             oracle_sigma: float | None = None, exact_diag=None, log=None) -> dict:
    """Anchor-local protocol for one direction (A -> B) of one nested draw.

    DA_rows, DB_rows : (n_a, M) distances of the predicted anchors (the H2
        items of the m_f prefix, in ascending pool position ``anchor_pos``) to
        pool positions 0..M-1, M >= max(m_grid). Only DB_rows[:, :m_f] is read
        by any predictor (tested); DB_rows[:, :m] is read for the OBSERVED s_N.
    DA_fit, DB_fit : the m_f x m_f block of the prefix; h1_fit its H1 mask.
        Read only by variant iv (H1 x H1), N1 (H1 x H1) and N0H1 (H1 rows).
    exact_diag : optional {"ks": (10,), "m": (...), "variants": (...)} --
        additionally compute the exact discrepancy at those k (diagnostic for
        the D2 'k <= 3 only' decision).
    Returns {"cells": [...], "diag": {...}}."""
    t_start = time.perf_counter()
    m_grid = [int(m) for m in m_grid]
    ks = [int(k) for k in ks]
    anchor_pos = np.asarray(anchor_pos, dtype=np.int64)
    if np.any(anchor_pos >= m_f) or np.any(np.diff(anchor_pos) <= 0):
        raise ValueError("anchor_pos must be ascending positions inside the m_f prefix")
    if np.any(h1_fit[anchor_pos]):
        raise ValueError("predicted anchors must be H2 items")
    n_all = anchor_pos.size
    diag = {"n_anchors_fit": int(n_all), "m_f": int(m_f), "timing": {}}

    # ---- the predicted anchors' own rows at m_f ------------------------------
    Fr = make_fit_rows(DA_rows[:, :m_f], DB_rows[:, :m_f], anchor_pos, m_f)
    links = fit_iso_links(Fr) if "ii-IsoXF" in variants else None
    model10 = p10.fit_coupling(DA_fit, DB_fit, h1_fit, variant=V10_VARIANT) if "iv-v10" in variants else None
    n1 = p10.fit_n1(DA_fit, DB_fit, h1_fit)
    n1_defined = bool(np.isfinite(n1["sigma_A"]) and n1["slope"] > 0)
    diag["N1_fit"] = {k: float(v) for k, v in n1.items()}
    diag["N1_defined"] = n1_defined
    if model10 is not None:
        diag["iv_fit"] = model10.summary()

    # ---- N0 (own anchors at m_f) and N0H1 (v1.0: H1 anchors at m_f) -----------
    ov_f = overlap_fracs(DA_rows, DB_rows, anchor_pos, m_f, ks)
    N0 = {k: float(v.mean()) for k, v in ov_f.items()}
    h1i = np.flatnonzero(h1_fit)
    kk_f = [k for k in ks if k <= m_f - 2]
    nAf, _ = st.knn_table(DA_fit, max(kk_f) + 1)
    nBf, _ = st.knn_table(DB_fit, max(kk_f) + 1)
    N0H1 = {k: float(st.overlap_counts(nAf[h1i], nBf[h1i], k).mean() / k) for k in kk_f}
    diag["timing"]["fit_s"] = time.perf_counter() - t_start

    cells = []
    for m in m_grid:
        t0 = time.perf_counter()
        sel = anchor_pos < m
        ri = np.flatnonzero(sel)
        pos_m = anchor_pos[ri]
        kk = [k for k in ks if k <= m - 2]
        if not kk or ri.size == 0:
            continue
        ov = overlap_fracs(DA_rows[ri], DB_rows[ri], pos_m, m, kk)
        U, COL, _ = sort_rows(DA_rows[ri], pos_m, m)
        for k in kk:
            tk = time.perf_counter()
            rho = cell_rho(k, m, m_f)
            band = band_mask(rho, Fr.n_f)
            w_ex = window_mask(rho, Fr.n_f, exclude=band)
            w_all = window_mask(rho, Fr.n_f)
            rec = {"m": m, "k": k, "is_fit_m": m == m_f, "rho": rho, "n_band": int(band.sum()),
                   "n_win_i": int(w_ex.sum()), "n_win_ii": int(w_all.sum()),
                   "n_anchors": int(ri.size), "chance": k / (m - 1),
                   "obs": float(ov[k].mean()), "obs_sd": float(ov[k].std(ddof=1)) if ri.size > 1 else 0.0,
                   "N0": N0.get(k, float("nan")), "N0H1": N0H1.get(k, float("nan")),
                   "N1_defined": n1_defined}
            laws = {}
            if "i-LSlog" in variants or "iii-LSlog-EB" in variants:
                if "i-LSlog" in variants:
                    fi = fit_lslog(Fr, w_ex, shrink=False)
                    laws["i-LSlog"] = laws_lslog(fi, ri, U)
                    rec["i_median_gamma"] = fi.meta["median_gamma"]
                    rec["i_median_beta"] = fi.meta["median_beta"]
                if "iii-LSlog-EB" in variants:
                    fiii = fit_lslog(Fr, w_ex, shrink=True)
                    laws["iii-LSlog-EB"] = laws_lslog(fiii, ri, U)
                    rec["iii_median_gamma"] = fiii.meta["median_gamma"]
                    rec["iii_eb_w_c"] = fiii.meta["eb_c"]["mean_w"]
                    rec["iii_eb_w_gamma"] = fiii.meta["eb_gamma"]["mean_w"]
            if "ii-IsoXF" in variants:
                fii = fit_isoxf(links, Fr, w_all)
                laws["ii-IsoXF"] = laws_isoxf(fii, links, ri, U, COL)
                rec["ii_median_gamma"] = fii.meta["median_gamma"]
            if "iv-v10" in variants:
                c10, s10, F10, a10, cs10 = p10._column_laws(model10, U, m)
                laws["iv-v10"] = (c10, s10, F10, a10, cs10)
            if n1_defined:
                laws["N1"] = gauss_laws(U, n1["sigma_A"])
            if oracle_sigma is not None:
                laws["oracle"] = gauss_laws(U, oracle_sigma)
            for name, (c, s, F, aux, csup) in laws.items():
                for mode in modes:
                    if mode == "mu" and k > EXACT_K_MAX and "d2" in modes:
                        continue          # identical to d2 for k > 3
                    loss = expected_loss(c, s, k, F, aux, csup, mode)
                    key = name if mode == "d2" else f"{name}|mu"
                    rec[key] = float(1.0 - loss.mean() / k)
                if "d2" in modes and "mu" in modes and k > EXACT_K_MAX:
                    rec[f"{name}|mu"] = rec[name]
            if exact_diag and k in exact_diag.get("ks", ()) and m in exact_diag.get("m", ()):
                for name in exact_diag.get("variants", ()):
                    if name in laws:
                        c, s, F, aux, csup = laws[name]
                        loss = p10.exact_loss(c, s, k, F, None, aux=aux, col_support=csup, n_grid=EXACT_GRID)
                        rec[f"{name}|exact"] = float(1.0 - loss.mean() / k)
            # D4.8: N1 undefined (slope <= 0) -> N0
            if not n1_defined:
                rec["N1"] = rec["N0"]
                rec["N1|mu"] = rec["N0"]
            del laws
            rec["seconds"] = time.perf_counter() - tk
            cells.append(rec)
        diag["timing"][f"m{m}_s"] = time.perf_counter() - t0
        if log is not None:
            log(f"  m={m} n_anchors={ri.size} {time.perf_counter() - t0:.1f}s")
    diag["timing"]["total_s"] = time.perf_counter() - t_start
    return {"cells": cells, "diag": diag}
