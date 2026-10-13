"""V-f (role e0-indep): own implementation of the plan section 6 quantities and
the section 7.1 (P1), 7.3 (P2b), 7.4 (P2c) rules, written from the plan text.
numpy / scipy only (scipy is used only as a cross-check of the own exact
Wilcoxon enumeration).
"""
from __future__ import annotations

import itertools

import numpy as np

M_GRID = [256, 512, 1024, 2048, 4096, 8192, 16384]
N_DRAWS = 5

# ---- frozen constants, copied from the plan text (7.3, 7.4) ----
LINE_A = 0.1357059076472474
LINE_C = 0.22805384255107658
S_RES = 0.05647474462525915
GUARD_P2B = 2 * S_RES
BETA_MV = {
    "augreg_b": -0.5752592991292489,
    "clip_b": -0.5499745403938117,
    "dinov2_b": -0.5890678384804802,
    "mae_b": -0.6444729401596141,
    "siglip_b": -0.511091295171744,
}
BETA_M_CONST = -0.570555396995647
ENCS = ["contriever", "mpnet", "qwen3emb06"]
VITS = sorted(BETA_MV)


def ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[1])


# ---------------------------------------------------------------------------
# section 6
# ---------------------------------------------------------------------------

def cells_from_obs(obs_rows):
    """obs_rows: dict (pair, draw, m, k) -> (s, n_anchors).
    Returns draw-mean dict (pair, m, k) -> (sbar, epsbar)."""
    out = {}
    keys = sorted({(p, m, k) for (p, d, m, k) in obs_rows})
    for (p, m, k) in keys:
        s = [obs_rows[(p, d, m, k)][0] for d in range(N_DRAWS)]
        n = [obs_rows[(p, d, m, k)][1] for d in range(N_DRAWS)]
        sbar = float(np.mean(s))
        nbar = float(np.mean(n))
        eps = 0.5 / (N_DRAWS * nbar * k)
        out[(p, m, k)] = (sbar, eps)
    return out


def draw_slope(obs_rows, pair, d, k):
    x = np.log(np.array(M_GRID, float))
    y = []
    for m in M_GRID:
        s, n = obs_rows[(pair, d, m, k)]
        y.append(np.log(max(s, 0.5 / (n * k))))
    return ols_slope(x, y)


def mean_slope(obs_rows, pair, k):
    sl = [draw_slope(obs_rows, pair, d, k) for d in range(N_DRAWS)]
    return float(np.mean(sl)), sl


def elasticities(cells, pair, K):
    rows, ys = [], []
    for k in K:
        for m in M_GRID:
            s, e = cells[(pair, m, k)]
            rows.append([1.0, np.log(k), np.log(m)])
            ys.append(np.log(max(s, e)))
    coef, *_ = np.linalg.lstsq(np.array(rows), np.array(ys), rcond=None)
    ek, em = float(coef[1]), float(coef[2])
    return ek, em, ek + em


def level(cells, pair, k=10, mref=1024):
    s, e = cells[(pair, mref, k)]
    return s, float(np.log(max(s, e)))


# ---------------------------------------------------------------------------
# P1 (7.1)
# ---------------------------------------------------------------------------

def p1_label(em, sums):
    em = np.asarray(em)
    sums = np.asarray(sums)
    inf = np.abs(em) >= 0.10
    n_inf = int(inf.sum())
    if n_inf < 8:
        return dict(label="uninformative", n_informative=n_inf, median_sum=None, frac_le_005=None)
    s = sums[inf]
    med = float(np.median(s))
    frac = float(np.mean(np.abs(s) <= 0.05))
    if abs(med) <= 0.03 and frac >= 0.80:
        lab = "holds"
    elif abs(med) > 0.06 or frac <= 0.50:
        lab = "fails"
    else:
        lab = "partly holds"
    return dict(label=lab, n_informative=n_inf, median_sum=med, frac_le_005=frac)


# ---------------------------------------------------------------------------
# P2b (7.3)
# ---------------------------------------------------------------------------

def beta_star(lnL):
    return min(LINE_A + LINE_C * lnL, 0.0)


def lam(g, y):
    g = np.asarray(g)
    y = np.asarray(y)
    return float(np.sum(g * y) / np.sum(g * g))


def wilcoxon_exact_greater(diff):
    """Own exact one-sided signed-rank p = P(W+ >= W+_obs), zeros dropped,
    average ranks for ties, full enumeration of sign patterns."""
    d = np.asarray(diff, float)
    d = d[d != 0]
    n = len(d)
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n)
    sa = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    wplus = float(ranks[d > 0].sum())
    cnt = 0
    tot = 0
    # enumerate all 2^n sign patterns
    signs = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int8)
    w = signs @ ranks
    cnt = int(np.sum(w >= wplus - 1e-9))
    tot = len(w)
    return dict(W_plus=wplus, n=n, p=cnt / tot)


def p2b(pairs, lnL, b):
    """pairs: list of (vit, enc); lnL, b: dict pair -> value."""
    names = [f"{v}__{e}" for v, e in pairs]
    g = {nm: beta_star(lnL[nm]) - BETA_MV[v] for nm, (v, e) in zip(names, pairs)}
    y = {nm: b[nm] - BETA_MV[v] for nm, (v, e) in zip(names, pairs)}
    med_abs_g = float(np.median([abs(g[nm]) for nm in names]))
    enc_inf = {}
    lam_e = {}
    for e in ENCS:
        sub = [nm for nm, (v, ee) in zip(names, pairs) if ee == e]
        mg = float(np.median([abs(g[nm]) for nm in sub]))
        enc_inf[e] = dict(median_abs_g=mg, informative=bool(mg >= GUARD_P2B))
        lam_e[e] = lam([g[nm] for nm in sub], [y[nm] for nm in sub])
    lam_hat = lam([g[nm] for nm in names], [y[nm] for nm in names])
    inf_encs = [e for e in ENCS if enc_inf[e]["informative"]]
    if med_abs_g < GUARD_P2B:
        lab = "uninformative"
    elif len(inf_encs) < 2:
        lab = "inconclusive"
    elif 0.75 <= lam_hat <= 1.25 and all(0.5 <= lam_e[e] <= 1.5 for e in inf_encs):
        lab = "holds"
    elif lam_hat <= 0.25 and all(lam_e[e] <= 0.5 for e in inf_encs):
        lab = "fails"
    else:
        lab = "inconclusive"
    loeo = {e: lam([g[nm] for nm, (v, ee) in zip(names, pairs) if ee != e],
                   [y[nm] for nm, (v, ee) in zip(names, pairs) if ee != e]) for e in ENCS}
    lovo = {v: lam([g[nm] for nm, (vv, ee) in zip(names, pairs) if vv != v],
                   [y[nm] for nm, (vv, ee) in zip(names, pairs) if vv != v]) for v in VITS}
    resid = {nm: b[nm] - beta_star(lnL[nm]) for nm in names}
    diff_v = [abs(b[nm] - BETA_MV[v]) - abs(b[nm] - beta_star(lnL[nm])) for nm, (v, e) in zip(names, pairs)]
    diff_c = [abs(b[nm] - BETA_M_CONST) - abs(b[nm] - beta_star(lnL[nm])) for nm in names]
    return dict(label=lab, median_abs_g=med_abs_g, guard=GUARD_P2B, lambda_hat=lam_hat,
                lambda_e=lam_e, encoder_informative=enc_inf, n_informative_encoders=len(inf_encs),
                leave_one_encoder_out=loeo, leave_one_vit_out=lovo, g=g, y=y,
                beta_star={nm: beta_star(lnL[nm]) for nm in names}, residual=resid,
                wilcoxon_vit=wilcoxon_exact_greater(diff_v), wilcoxon_const=wilcoxon_exact_greater(diff_c),
                diff_vit=dict(zip(names, diff_v)), diff_const=dict(zip(names, diff_c)))


# ---------------------------------------------------------------------------
# P2c (7.4)
# ---------------------------------------------------------------------------

def p2c(pairs, lnL_n, b_n):
    """lnL_n, b_n: dict pair -> list over n = 1..5."""
    names = [f"{v}__{e}" for v, e in pairs]
    gam = {nm: ols_slope(lnL_n[nm], b_n[nm]) for nm in names}
    dlt = {nm: float(lnL_n[nm][4] - lnL_n[nm][0]) for nm in names}
    med_d = float(np.median([dlt[nm] for nm in names]))
    med_g = float(np.median([gam[nm] for nm in names]))
    enc = {}
    for e in ENCS:
        sub = [nm for nm, (v, ee) in zip(names, pairs) if ee == e]
        md = float(np.median([dlt[nm] for nm in sub]))
        mgm = float(np.median([gam[nm] for nm in sub]))
        enc[e] = dict(median_delta=md, informative=bool(md >= 0.3), median_gamma=mgm)
    inf_encs = [e for e in ENCS if enc[e]["informative"]]
    if med_d < 0.3:
        lab = "uninformative"
    elif len(inf_encs) < 2:
        lab = "inconclusive"
    elif 0.17 <= med_g <= 0.29 and all(0.154 <= enc[e]["median_gamma"] <= 0.342 for e in inf_encs):
        lab = "holds"
    elif med_g <= 0.12 and all(enc[e]["median_gamma"] <= 0.154 for e in inf_encs):
        lab = "fails"
    else:
        lab = "inconclusive"
    loeo = {e: dict(median_gamma=float(np.median([gam[nm] for nm, (v, ee) in zip(names, pairs) if ee != e])),
                    median_delta=float(np.median([dlt[nm] for nm, (v, ee) in zip(names, pairs) if ee != e])))
            for e in ENCS}
    return dict(label=lab, median_delta=med_d, median_gamma=med_g, per_encoder=enc,
                n_informative_encoders=len(inf_encs), gamma=gam, delta=dlt, leave_one_encoder_out=loeo)
