"""Metrics (PREREG section 7). Everything is computed on unit-diagonal Gram matrices through the
frozen hrhv2 harness (Geom.from_gram -> D = K_ii + K_jj - 2K_ij clipped at 0, tie rule (D_ij, j)),
imported read-only; the ladder Delta map is Stage 2's crit.deltas (read-only).

  ladder_deltas(KA, KB)     hrhv2.ladder on S_lad = X (m = 256) -> crit.deltas + dEk + iso_inf + raw G_max
  ORDER_RUNGS               the four order rungs of Pi, Lemma L and the verdicts. T_E enters as dEk =
                            hrhv2 globalorder_kendall_dist_halfties (kendall_distance_from_counts: fixed
                            denominator, one-sided ties count 1/2), NOT crit's dE = Delta_E = (1 - tau_b)/2,
                            whose denominator varies with ties (binding correction (d), lemma-check verdict
                            2026-10-09); dE is reported as a secondary rung only.
  exact_obs(KA, KB, S, k)   exact T_N,k / T_N,1 / T_O / T_E / T_MST / T_SL agreement on probes S
  iso_inf(DA, DB)           inf_g || g(D_A) - D_B/psi(D_B) ||_max over nondecreasing g (A->B)
  inv_link_cf(KA, KB, t)    Gs_max(D_A, g_t^{-1}(D_B)), g_t^{-1}(u) = -2t log(1 - u/2)
  spectrum_stats(K)         PR of H K H, max cosine, numerical rank, fraction of entries > 0.999
  direct_* (gate G1)        Lemma-L direct indicator averages, independent of hrhv2
"""
from __future__ import annotations

import numpy as np

import s23common as C  # noqa: F401  (sets sys.path for hrhv2 and crit)
from hrhv2.prep import Geom, gram_to_dist, center_gram
from hrhv2.ladder import ladder
from hrhv2 import structural as st
from hrhv2 import geometric as geo
from hrhv2 import util as hu

crit = C.load_crit()

ORDER_RUNGS = ("dN_k10", "dO", "dEk", "dMST")         # the four order rungs of sections 7.3 / 9 (fixed denominators)
SECONDARY_ORDER = ("dE",)                              # Delta_E = (1 - tau_b)/2: reported only, never in Pi / Lemma L
RESCALE_NONINVARIANT = ("G_max", "rawG")              # raw distances: NOT invariant under K -> aK + (1-a)11^T
GEOM_RUNGS = ("Gs", "dCKA")                            # P5-geom
P5B_UNREL = ("Gs", "dAff", "dCKA", "dSOG_t0p1")       # P5b cross-unreliable rungs


def geom(K):
    return Geom.from_gram(np.asarray(K, dtype=np.float64))


def upper(M):
    return hu.upper_values(np.asarray(M))


def iso_inf_from_D(DA, DB):
    x = upper(DA)
    y = upper(DB)
    y = y / y.mean()
    o = np.lexsort((y, x))
    xs, ys = x[o], y[o]
    new = np.r_[True, xs[1:] != xs[:-1]]
    starts = np.flatnonzero(new)
    gmax = np.maximum.reduceat(ys, starts)
    gmin = np.minimum.reduceat(ys, starts)
    runmax = np.maximum.accumulate(gmax)
    return 0.5 * max(0.0, float(np.max(runmax - gmin)))


def iso_inf(KA, KB):
    return iso_inf_from_D(gram_to_dist(np.asarray(KA)), gram_to_dist(np.asarray(KB)))


def ladder_deltas(KA, KB, n_jobs=1):
    GA, GB = geom(KA), geom(KB)
    row = ladder(GA, GB, n_jobs=n_jobs)
    try:
        d = crit.deltas(row)
        d["omega_insample_undefined"] = False
    except ZeroDivisionError:
        # crit.deltas divides by (1 - affine share); an exactly affine pair (e.g. the rescale-self
        # calibration pair) makes it 0. Only the descriptive omega rungs are affected: set the affine
        # share to NaN for them and keep every other rung exactly as crit defines it.
        row2 = dict(row)
        for dd in ("AB", "BA"):
            for k in (f"quot_{dd}_affine_expl", f"quot_{dd}_affine_expl_cf"):
                if k in row2 and row2[k] == 1.0:
                    row2[k] = float("nan")
        d = crit.deltas(row2)
        d["omega_insample_undefined"] = True
    d["dEk"] = float(row["globalorder_kendall_dist_halfties"])
    d["iso_inf_AB"] = iso_inf_from_D(GA.D, GB.D)
    d["iso_inf_BA"] = iso_inf_from_D(GB.D, GA.D)
    d["G_max"] = float(row.get("G_max", float("nan")))
    d["cka_biased"] = float(row["cka_biased"])
    d["s_N_k10"] = float(row["s_N_k10"])
    d["nu_A"], d["nu_B"] = float(row["nu_A"]), float(row["nu_B"])
    d["globalorder_ties_A"] = int(row["globalorder_ties_A"])
    d["globalorder_ties_B"] = int(row["globalorder_ties_B"])
    return d


def fast_rungs(KA, KB):
    """The four order rungs + Gs + 1-CKA only, via the same hrhv2 primitives as the ladder
    (used by the [Model] Monte Carlo of nstar.py; identical definitions)."""
    GA, GB = geom(KA), geom(KB)
    DA, DB = GA.D, GB.D
    nA, _ = st.knn_table(DA, 11)
    nB, _ = st.knn_table(DB, 11)
    out = {"dN_k10": 1.0 - st.knn_overlap(nA, nB, 10)}
    per, _ = st.row_kendall(DA, DB, n_jobs=1)
    out["dO"] = float(per.mean())
    go = st.global_order(DA, DB, max_pairs=None)
    out["dE"] = go["Delta_E"]
    out["dEk"] = go["globalorder_kendall_dist_halfties"]
    out["dMST"] = 1.0 - st.mst_compare(st.mst_prim(DA), st.mst_prim(DB), st.nn1_forest(nA[:, 0]))["s_T"]
    out["Gs"] = geo.gs(DA, DB)["Gs_rms"]
    out["dCKA"] = 1.0 - geo.cka_biased_from_centered(center_gram(GA.K), center_gram(GB.K))
    return out


# ----------------------------------------------------------------------------- exact observables
def _sl_sequence(D, codes):
    m = D.shape[0]
    a, b = codes // m, codes % m
    o = np.lexsort((codes, D[a, b]))
    return codes[o]


def exact_obs(KA, KB, S, k=2):
    """Exact agreement indicators on probe set S (order of S as given)."""
    S = np.asarray(S)
    DA = gram_to_dist(np.asarray(KA))[np.ix_(S, S)]
    DB = gram_to_dist(np.asarray(KB))[np.ix_(S, S)]
    m = len(S)
    nA, _ = st.knn_table(DA, m - 1)
    nB, _ = st.knn_table(DB, m - 1)
    out = {}
    for kk in sorted({1, k}):
        out[f"T_N{kk}"] = bool(np.all(st.overlap_counts(nA, nB, kk) == kk))
    rows_ok = True
    for i in range(m):
        x = np.concatenate((DA[i, :i], DA[i, i + 1:]))
        y = np.concatenate((DB[i, :i], DB[i, i + 1:]))
        c = st.kendall_counts(x, y)
        if not (c["dis"] == 0 and c["xtie"] == c["ytie"] == c["ntie"]):
            rows_ok = False
            break
    out["T_O"] = rows_ok
    c = st.kendall_counts(upper(DA), upper(DB))
    out["T_E"] = bool(c["dis"] == 0 and c["xtie"] == c["ytie"] == c["ntie"])
    TA, TB = st.mst_prim(DA), st.mst_prim(DB)
    out["T_MST"] = bool(np.array_equal(TA, TB))
    out["T_SL"] = bool(np.array_equal(_sl_sequence(DA, TA), _sl_sequence(DB, TB)))
    return out


def ties_on(K, S, tol=1e-9):
    """Number of pairs of entries of D on S that are tied within tol (computed limits only)."""
    S = np.asarray(S)
    D = np.sort(upper(gram_to_dist(np.asarray(K))[np.ix_(S, S)]))
    return int(np.sum(np.diff(D) <= tol))


# ----------------------------------------------------------------------------- theorem metrics
def gs_max(KA, KB):
    return geo.gs(gram_to_dist(np.asarray(KA)), gram_to_dist(np.asarray(KB)))["Gs_max"]


def omega_self(K, t):
    """omega_psi(K) = Gs_max(K, phi_t(K)) computed directly: || D/psi - g_t(D)/psi(g_t(D)) ||_max."""
    D = upper(gram_to_dist(np.asarray(K)))
    G = 2.0 * (1.0 - np.exp(-D / (2.0 * t)))
    return float(np.max(np.abs(D / D.mean() - G / G.mean())))


def phi_t(K, t):
    """The theorem's link on Grams: exp((K - 1)/t) (unit diagonal preserved)."""
    M = np.exp((np.asarray(K) - 1.0) / t)
    np.fill_diagonal(M, 1.0)
    return M


def rescale_half(K):
    M = 0.5 * np.asarray(K) + 0.5
    np.fill_diagonal(M, 1.0)
    return M


def inv_link_cf(KA, KB, t):
    KB = np.asarray(KB)
    ub = upper(KB)
    frac_nonpos = float(np.mean(ub <= 0))
    if frac_nonpos > 0:
        return float("inf"), frac_nonpos
    DB = -2.0 * t * np.log(np.where(KB > 0, KB, 1.0))
    np.fill_diagonal(DB, 0.0)
    DB = 0.5 * (DB + DB.T)
    DA = gram_to_dist(np.asarray(KA))
    return float(geo.gs(DA, DB)["Gs_max"]), frac_nonpos


def recovery_err(K, Kinf):
    return float(np.max(np.abs(np.asarray(K) - np.asarray(Kinf))))


def spectrum_stats(K):
    K = np.asarray(K, dtype=np.float64)
    Kc = center_gram(K)
    ev = np.linalg.eigvalsh(K)
    off = upper(K)
    return {"PR": geo.participation_ratio(Kc), "max_cos": float(off.max()),
            "num_rank": int((ev > 1e-10 * ev.max()).sum()), "frac_gt_0999": float(np.mean(off > 0.999)),
            "lam_min": float(ev[0])}


# ----------------------------------------------------------------------------- Lemma L direct formulas (G1)
def _sgn_disagree_frac(x, y):
    """Mean over pairs of |sgn(x_i - x_j) - sgn(y_i - y_j)| / 2: discordant 1, one-sided tie 1/2,
    joint tie 0 -- a weighted absolute deviation of ternary features (Lemma L, part 1) with a FIXED
    denominator (all pairs)."""
    sx = np.sign(x[:, None] - x[None, :])
    sy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(len(x), 1)
    return float(np.mean(np.abs(sx[iu] - sy[iu]) / 2.0))


def direct_dEk(DA, DB):
    return _sgn_disagree_frac(upper(DA), upper(DB))


def tau_b_delta(DA, DB):
    """Delta_E = (1 - tau_b)/2 computed directly (secondary; its denominator varies with ties)."""
    from scipy.stats import kendalltau
    return float((1.0 - kendalltau(upper(DA), upper(DB)).statistic) / 2.0)


def direct_dO(DA, DB):
    m = DA.shape[0]
    v = []
    for i in range(m):
        x = np.delete(DA[i], i)
        y = np.delete(DB[i], i)
        v.append(_sgn_disagree_frac(x, y))
    return float(np.mean(v))


def _knn_sets(D, k):
    m = D.shape[0]
    out = []
    for i in range(m):
        js = [j for j in range(m) if j != i]
        js.sort(key=lambda j: (D[i, j], j))
        out.append(set(js[:k]))
    return out


def direct_dNk(DA, DB, k):
    m = DA.shape[0]
    NA, NB = _knn_sets(DA, k), _knn_sets(DB, k)
    dis = sum(len(NA[i] ^ NB[i]) for i in range(m))       # sum_i sum_j 1[f_A(i,j) != f_B(i,j)]
    return dis / (2.0 * k * m)


def _kruskal(D):
    m = D.shape[0]
    edges = sorted(((D[i, j], i, j) for i in range(m) for j in range(i + 1, m)))
    par = list(range(m))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    T = set()
    for _, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            par[ri] = rj
            T.add(i * m + j)
    return T


def direct_dMST(DA, DB):
    m = DA.shape[0]
    TA, TB = _kruskal(DA), _kruskal(DB)
    return len(TA ^ TB) / (2.0 * (m - 1))
