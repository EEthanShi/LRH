"""ladder_ref: the full SPEC v1.0 §0–§5 criterion ladder for one pair (A, B).

Independent reference implementation for cross-checking hrhv2/.  Written from
SPEC.md v1.0 only (plus the vendored ARH/PRH code for the explicitly requested
matches); hrhv2/ was not read.

Choices where SPEC v1.0 is ambiguous or silent (keep in sync with the report):
 A1  D is built with the general formula (K_ii + K_jj) - 2K_ij (not 2 - 2K_ij),
     from a Gram made bitwise symmetric (upper triangle mirrored).  Set/tree
     quantities are exact only up to ulp-level differences in D; compare
     implementations together with near_tie_* fractions.
 A2  prh_clip threshold is computed on the (m, d) array passed in (per layer).
     ARH/PRH compute it on the stacked (n, L, d) tensor = one threshold per model;
     pass clip_threshold_A/B to reproduce that.
 A3  Per-anchor Kendall distance with ties = Fagin et al. 2006 K^(1/2): tied in
     exactly one ranking -> 1/2, tied in both -> 0.  This equals (1 - tau_b)/2 only
     on tie-free rows; scipy's tau_b-based value differs on rows with ties.
 A4  k_q = round((m - 1)/100) with Python round (half-to-even) and max(1, .);
     grid values k >= m are reported as NaN.
 A5  Near-tie fraction uses median(D) over unordered off-diagonal pairs, strict '<'.
 A6  s_T_bridge = NaN when T_A \\ F_A is empty; extra keys forest_share_B and
     forest_share_shared_A = |T_A∩T_B∩F_A| / |T_A∩T_B| (the PREREG PE2 quantity,
     which is NOT SPEC's forest_share_A = |F_A|/(m-1)); PE2 does not say whose
     forest, so forest_share_shared_B and forest_share_shared_union (F_A ∪ F_B)
     are reported too.
 A7  SOG (v0.2, checked against the pinned SOG code be654a37): plain distance d
     (confirmed: SOG uses chord distances on L2 rows, not D); Q_0.9 over the
     positive off-diagonal *matrix entries* (ordered pairs, each unordered pair
     twice), numpy-'linear' quantile = SOG `_normalize_distance_array`; log of
     max(d/Q, 1e-12) as in SOG (both zero -> w=1, one zero -> tiny w>0).
     sog_q_mode="unordered", sog_eps=0 restore the v0.1 convention.
 A13 kgrid=None -> {1,3,10,30,100} ∪ {k_q}; an explicit kgrid is used as given
     (k_q is still reported as a key).  delta_N@k = 1 - s_N@k.  The MST block
     always reports s_N@1 (SPEC §1.5) even if 1 is not in kgrid.
 A8  Quotients: see quotient.py (tie pooling, knots, row-iso SS0 over ordered
     pairs, row-iso cross-fit over candidate halves).
 A9  Cross-fit halves: sha256(f"{salt}|{item_id}") ranking, first floor(m/2) = H1
     (override with H1=).
 A10 Projector distance normalised by r_A + r_B (numerical ranks) — equals 2d when
     both are full rank d; reported only when d_A < m and d_B < m and features
     (not Grams) are given.  CCA mean-squared correlation over min(r_A, r_B).
 A11 Debiased CKA without epsilon; cka_debiased_eps8 / cka_debiased_eps6 add the
     ARH (1e-8) / PRH (1e-6) denominator epsilons for the vendor matches.
 A12 Raw geometry G_* reported when pre_A == pre_B (always true with one `pre`).
"""
from __future__ import annotations

import numpy as np

from . import geometric as geo
from . import quotient as quo
from . import structural as st
from .core import (as_f64_2d, gram_sym, hash_halves, neighbor_order, preprocess,
                   sqdist_from_gram, upper)

DEFAULT_KGRID = (1, 3, 10, 30, 100)
DEFAULT_COMP_K = (1, 3, 10)
DEFAULT_TAUS = (1.0, 0.1, 0.01, 0.001, 0.0001)
ALL_PARTS = ("knn", "comp", "rank", "edge", "mst", "geom", "cka", "rsa", "sog", "quot", "proj")


def tau_label(t):
    return f"{t:g}"


def k_q_of(m):
    return max(1, int(round((m - 1) / 100)))


def ladder_ref(X_A, X_B, pre="l2", kgrid=None, *, gram=False, dist=False, parts=None,
               comp_k=DEFAULT_COMP_K, sog_k=10, taus=DEFAULT_TAUS,
               sog_q_mode="ordered", sog_eps=geo.SOG_EPS,
               item_ids=None, split_salt="hrhv2_ref/crossfit/v1", H1=None,
               clip_threshold_A=None, clip_threshold_B=None,
               kendall_brute_max_m=200, check_prim=True, rank_detail=True):
    """Compute the SPEC ladder.  X_A, X_B: (m, d_A), (m, d_B) features, or (m, m)
    Gram matrices if gram=True, or (m, m) squared-distance matrices if dist=True
    (then `pre` is ignored and the projector is skipped; with dist=True the CKAs use
    K = -D/2, which gives identical biased and Song-unbiased HSIC because both are
    invariant to K -> K + a1' + 1a').  Returns a flat dict of floats/ints."""
    parts = set(ALL_PARTS if parts is None else parts)
    out = {}
    if dist:
        DA = np.array(X_A, dtype=np.float64, copy=True)
        DB = np.array(X_B, dtype=np.float64, copy=True)
        for Dm in (DA, DB):
            if not np.array_equal(Dm, Dm.T):
                raise ValueError("distance matrices must be exactly symmetric")
            np.maximum(Dm, 0.0, out=Dm)
            np.fill_diagonal(Dm, 0.0)
        KA = -0.5 * DA
        KB = -0.5 * DB
        XA = XB = None
        out["pre"] = "dist"
    elif gram:
        KA = np.asarray(X_A, dtype=np.float64)
        KB = np.asarray(X_B, dtype=np.float64)
        XA = XB = None
        out["pre"] = "gram"
    else:
        XA, infoA = preprocess(as_f64_2d(X_A, "X_A"), pre, clip_threshold_A)
        XB, infoB = preprocess(as_f64_2d(X_B, "X_B"), pre, clip_threshold_B)
        if XA.shape[0] != XB.shape[0]:
            raise ValueError("X_A and X_B must have the same number of rows")
        KA = gram_sym(XA)
        KB = gram_sym(XB)
        out["pre"] = pre
        if "clip_c" in infoA:
            out["clip_c_A"] = infoA["clip_c"]
            out["clip_c_B"] = infoB["clip_c"]
    m = KA.shape[0]
    out["m"] = m
    if not dist:
        DA = sqdist_from_gram(KA)
        DB = sqdist_from_gram(KB)

    kq = k_q_of(m)
    out["k_q"] = kq
    ks = sorted(set(DEFAULT_KGRID if kgrid is None else kgrid) | ({kq} if kgrid is None else set()))
    need_order = parts & {"knn", "comp", "mst", "sog"}
    kneed = [k for k in ks if k <= m - 1]
    if "comp" in parts:
        kneed += [k for k in comp_k if k <= m - 1]
    if "sog" in parts and sog_k <= m - 1:
        kneed.append(sog_k)
    kmax = min(m - 1, max(kneed + [1]) + 1)
    if need_order:
        oA = neighbor_order(DA, kmax)
        oB = neighbor_order(DB, kmax)

    # ---- 1. kNN ----------------------------------------------------------
    if "knn" in parts:
        for k in ks:
            if k > m - 1:
                for nm in ("s_N", "delta_N", "chance", "near_tie_A", "near_tie_B"):
                    out[f"{nm}@{k}"] = float("nan")
                continue
            out[f"s_N@{k}"] = st.s_N(oA[:, :k], oB[:, :k])
            out[f"delta_N@{k}"] = 1.0 - out[f"s_N@{k}"]
            out[f"chance@{k}"] = k / (m - 1)
            out[f"near_tie_A@{k}"] = st.near_tie_fraction(DA, oA, k)
            out[f"near_tie_B@{k}"] = st.near_tie_fraction(DB, oB, k)

    # ---- 2. components -----------------------------------------------------
    if "comp" in parts:
        for k in comp_k:
            if k > m - 1:
                continue
            la = st.knn_components(oA[:, :k])
            lb = st.knn_components(oB[:, :k])
            out[f"comp_rand@{k}"] = st.rand_distance(la, lb)
            out[f"comp_n_A@{k}"] = st.n_components(la)
            out[f"comp_n_B@{k}"] = st.n_components(lb)
            out[f"comp_degenerate@{k}"] = bool(out[f"comp_n_A@{k}"] == 1 and out[f"comp_n_B@{k}"] == 1)

    # ---- 3. per-anchor ranking -------------------------------------------------
    if "rank" in parts:
        det = None
        if m <= kendall_brute_max_m:
            rows = st.kendall_rows_brute(DA, DB)
            out["delta_O_method"] = "brute"
        else:
            rows, det = st.kendall_rows_merge(DA, DB, return_detail=True)
            out["delta_O_method"] = "mergesort"
        out["delta_O"] = float(np.mean(rows))
        if rank_detail:
            if det is None:
                _, det = st.kendall_rows_merge(DA, DB, return_detail=True)
            out["delta_O_strict_disc"] = float(np.mean(det[:, 0]))
            out["delta_O_tieA_only"] = float(np.mean(det[:, 1]))
            out["delta_O_tieB_only"] = float(np.mean(det[:, 2]))
            out["rows_with_ties_frac"] = float(np.mean((det[:, 1] + det[:, 2]) > 0))

    # ---- 4. global edge order ----------------------------------------------
    if "edge" in parts:
        a = upper(DA)
        b = upper(DB)
        cnt = st.kendall_pair_counts(a, b)
        tb = st.tau_b_from_counts(cnt)
        out["kendall_global"] = tb
        out["E_n_disc_strict"] = cnt["n_disc_strict"]
        out["E_ties_A_only"] = cnt["ties_A_only"]
        out["E_ties_B_only"] = cnt["ties_B_only"]
        out["E_ties_both"] = cnt["ties_both"]
        out["delta_E"] = (1.0 - tb) / 2.0
        out["spearman_global"] = st.spearman(a, b)

    # ---- 5. MST ---------------------------------------------------------------
    if "mst" in parts or "sog" in parts:
        TA = st.mst_kruskal(DA)
        TB = st.mst_kruskal(DB)
    if "mst" in parts:
        FA = st.forest_1nn(oA, m)
        FB = st.forest_1nn(oB, m)
        shared = np.intersect1d(TA, TB, assume_unique=True)
        out["s_T"] = shared.size / (m - 1)
        out["forest_share_A"] = FA.size / (m - 1)
        out["forest_share_B"] = FB.size / (m - 1)
        sh_forest = np.intersect1d(shared, FA, assume_unique=True)
        out["s_T_forest"] = sh_forest.size / FA.size
        bridgesA = np.setdiff1d(TA, FA, assume_unique=True)
        out["n_bridges_A"] = int(bridgesA.size)
        nsb = np.intersect1d(bridgesA, TB, assume_unique=True).size
        out["n_shared_bridges"] = int(nsb)
        out["s_T_bridge"] = nsb / bridgesA.size if bridgesA.size else float("nan")
        out["forest_share_shared_A"] = sh_forest.size / shared.size if shared.size else float("nan")
        # PREREG PE2 does not say whose 1-NN forest; also report B's and the union
        nB_f = np.intersect1d(shared, FB, assume_unique=True).size
        nU_f = np.intersect1d(shared, np.union1d(FA, FB), assume_unique=True).size
        out["forest_share_shared_B"] = nB_f / shared.size if shared.size else float("nan")
        out["forest_share_shared_union"] = nU_f / shared.size if shared.size else float("nan")
        if "s_N@1" not in out:  # SPEC §1.5 lists s_N(1) with the forest/bridge split
            out["s_N@1"] = st.s_N(oA[:, :1], oB[:, :1])
        out["F_A_subset_T_A"] = bool(np.isin(FA, TA).all())
        out["F_B_subset_T_B"] = bool(np.isin(FB, TB).all())
        if check_prim:
            out["prim_eq_kruskal_A"] = bool(np.array_equal(st.mst_prim_tiekey(DA), TA))
            out["prim_eq_kruskal_B"] = bool(np.array_equal(st.mst_prim_tiekey(DB), TB))

    # ---- 6/7. geometry -----------------------------------------------------
    if "geom" in parts:
        out["Gs_rms"], out["Gs_max"] = geo.scale_normalised_geometry(DA, DB)
        out["G_rms"], out["G_max"] = geo.raw_geometry(DA, DB)

    # ---- 8. CKA ---------------------------------------------------------------
    if "cka" in parts:
        hab_b, haa_b, hbb_b = geo.hsic_biased(KA, KB), geo.hsic_biased(KA, KA), geo.hsic_biased(KB, KB)
        out["cka_biased"] = geo.cka_from_hsic(hab_b, haa_b, hbb_b)
        hab, haa, hbb = geo.hsic_song(KA, KB), geo.hsic_song(KA, KA), geo.hsic_song(KB, KB)
        out["hsic_u_AB"], out["hsic_u_AA"], out["hsic_u_BB"] = hab, haa, hbb
        out["cka_debiased"] = geo.cka_from_hsic(hab, haa, hbb)
        out["cka_debiased_eps8"] = geo.cka_from_hsic(hab, haa, hbb, eps=1e-8)
        out["cka_debiased_eps6"] = geo.cka_from_hsic(hab, haa, hbb, eps=1e-6)

    # ---- 9. RSA ---------------------------------------------------------------
    if "rsa" in parts:
        out["rsa_pearson"] = geo.rsa_pearson(DA, DB)

    # ---- 10. SOG --------------------------------------------------------------
    if "sog" in parts and sog_k <= m - 1:
        QA = geo.sog_scale(DA, mode=sog_q_mode)
        QB = geo.sog_scale(DB, mode=sog_q_mode)
        out["sog_Q_A"], out["sog_Q_B"] = QA, QB
        out["sog_Q_mode"] = sog_q_mode
        dA = np.sqrt(DA)
        dB = np.sqrt(DB)
        NA = oA[:, :sog_k]
        NB = oB[:, :sog_k]
        ii, jj = [], []
        for i in range(m):
            common = np.intersect1d(NA[i], NB[i], assume_unique=True)
            ii.append(np.full(common.size, i, dtype=np.int64))
            jj.append(common)
        ii = np.concatenate(ii)
        jj = np.concatenate(jj)
        shared = np.intersect1d(TA, TB, assume_unique=True)
        ei, ej = shared // m, shared % m
        for t in taus:
            wk = geo.sog_weights(dA[ii, jj], dB[ii, jj], QA, QB, t, eps=sog_eps)
            out[f"sog_knn@{tau_label(t)}"] = float(wk.sum()) / (m * sog_k)
            we = geo.sog_weights(dA[ei, ej], dB[ei, ej], QA, QB, t, eps=sog_eps)
            out[f"sog_mst@{tau_label(t)}"] = float(we.sum()) / (m - 1)

    # ---- §3 quotients --------------------------------------------------------
    if "quot" in parts:
        if H1 is None:
            H1 = hash_halves(m, item_ids, split_salt)
        H1 = np.asarray(H1, dtype=bool)
        out["H1_size"] = int(H1.sum())
        qab = quo.quotients_one_direction(DA, DB, H1)
        qba = quo.quotients_one_direction(DB, DA, H1)
        for q in ("scale", "affine", "iso", "rowiso"):
            for suff in ("", "_xfit"):
                for si, stat in enumerate(("expl", "rms", "max")):
                    v_ab = qab[q + suff][si]
                    v_ba = qba[q + suff][si]
                    base = f"quot_{q}_{stat}" + suff
                    out[base] = v_ab
                    out[base + "_BA"] = v_ba
                    out[base + "_mean"] = 0.5 * (v_ab + v_ba)

    # ---- §4 projector ----------------------------------------------------------
    if "proj" in parts:
        if XA is not None:
            pdist, cca, rA, rB = geo.projector_distance(XA, XB)
            out["proj_dist"] = pdist
            out["cca_mean_sq"] = cca
            out["proj_rank_A"], out["proj_rank_B"] = rA, rB
        else:
            out["proj_dist"] = float("nan")
            out["cca_mean_sq"] = float("nan")
    return out


_PREFIX_PART = (
    ("s_N", "knn"), ("delta_N", "knn"), ("chance", "knn"), ("near_tie", "knn"), ("comp_", "comp"),
    ("delta_O", "rank"), ("rows_with_ties", "rank"), ("kendall_global", "edge"),
    ("spearman_global", "edge"), ("delta_E", "edge"), ("E_", "edge"), ("s_T", "mst"),
    ("forest", "mst"), ("n_bridges", "mst"), ("n_shared", "mst"), ("Gs_", "geom"), ("G_", "geom"),
    ("cka", "cka"), ("hsic", "cka"), ("rsa", "rsa"), ("sog_", "sog"), ("quot_", "quot"),
    ("proj", "proj"), ("cca", "proj"),
)


def part_of(name):
    for pre, part in _PREFIX_PART:
        if name.startswith(pre):
            return part
    raise KeyError(name)


def ladder_score(name, pre="l2", gram=False, **kw):
    """Return score_fn(XA, XB) -> ladder_ref(...)[name], computing only the needed
    part (for calibrate_ref).  For s_N@k pass the k in `name`.  Note: calibrate_ref
    assumes a similarity (higher = more aligned, bounded by s_max); for discrepancy
    keys (delta_*, Gs_*, G_*, comp_rand, quot_*_rms/max) wrap the function yourself."""
    part = part_of(name)
    extra = dict(kw)
    if part == "knn" and "@" in name:
        extra.setdefault("kgrid", [int(name.split("@")[1])])
    extra.setdefault("check_prim", False)
    extra.setdefault("rank_detail", False)

    def fn(XA, XB):
        return ladder_ref(XA, XB, pre, gram=gram, parts={part}, **extra)[name]

    return fn
