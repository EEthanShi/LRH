"""The full criteria ladder for one pair of representations (SPEC 0-4).

    from hrhv2.prep import Geom
    from hrhv2.ladder import ladder
    A = Geom.from_features(XA, "prh_clip+l2")
    B = Geom.from_features(XB, "prh_clip+l2")
    row = ladder(A, B)            # flat dict of floats / ints / bools / str

Permutation calibration (SPEC 5) is separate (``calibrate.py``) because its
cost is K times a score.
"""
from __future__ import annotations

import numpy as np

from . import __version__, SPEC_VERSION
from . import util, structural as st, geometric as geo, quotients as qu, gl as glmod
from .prep import Geom, same_pre


def ladder(A: Geom, B: Geom, *, k_grid=None, comp_ks=(1, 3, 10), sog_k: int = 10,
           sog_taus=(1.0, 0.1, 0.01, 0.001, 0.0001), halves_salt: str = "hrhv2-halves",
           item_ids=None, n_jobs: int | None = None, with_quotients: bool = True,
           with_row_iso: bool = True, with_row_kendall: bool = True, with_gl: bool = True,
           global_order_key: str = "hrhv2|global_order", global_order_max_pairs: int | None = None,
           return_aux: bool = False):
    """Compute every SPEC criterion for the pair (A, B); returns a flat dict.

    item_ids: identifiers used for the H1/H2 hash split (default 0..m-1).
    global_order_max_pairs: None (default) = exact global edge order (SPEC);
        an integer enables the flagged, non-SPEC pair subsampling.
    return_aux: also return neighbour tables and MST edge codes.
    """
    T = util.Timer()
    m = A.m
    if B.m != m:
        raise ValueError("A and B must have the same number of items")
    DA, DB = A.D, B.D
    ks = list(k_grid) if k_grid is not None else st.default_k_grid(m)
    ks = [k for k in ks if 1 <= k <= m - 2]
    out = {"m": m, "pre_A": A.pre, "pre_B": B.pre, "source_A": A.source, "source_B": B.source,
           "hrhv2_version": __version__, "spec_version": SPEC_VERSION, "k_q": st.k_q(m)}
    ids = np.arange(m) if item_ids is None else np.asarray(item_ids)
    h1 = util.hash_half([str(i) for i in ids], halves_salt)
    out["n_H1"] = int(h1.sum())

    # ---- kNN tables + near ties ------------------------------------------------
    with T("knn"):
        kmax = min(max(ks + list(comp_ks) + [sog_k]) + 1, m - 1)
        nA, vA = st.knn_table(DA, kmax)
        nB, vB = st.knn_table(DB, kmax)
        medA = float(np.median(util.upper_values(DA)))
        medB = float(np.median(util.upper_values(DB)))
        for k in ks:
            s = st.knn_overlap(nA, nB, k)
            out[f"s_N_k{k}"] = s
            out[f"Delta_N_k{k}"] = 1.0 - s
            out[f"chance_k{k}"] = k / (m - 1)
            out[f"neartie_A_k{k}"] = st.near_tie_fraction(vA, k, medA)
            out[f"neartie_B_k{k}"] = st.near_tie_fraction(vB, k, medB)
        out["s_N_kq"] = out.get(f"s_N_k{st.k_q(m)}", float("nan"))

    # ---- components ----------------------------------------------------------
    with T("components"):
        for k in comp_ks:
            if k > m - 2:
                continue
            ca, la = st.knn_components(nA, k)
            cb, lb = st.knn_components(nB, k)
            out[f"comp_rand_k{k}"] = st.rand_distance(la, lb)
            out[f"comp_n_A_k{k}"] = ca
            out[f"comp_n_B_k{k}"] = cb
            out[f"comp_degenerate_k{k}"] = bool(ca == 1 and cb == 1)

    # ---- per-anchor ranking --------------------------------------------------
    if with_row_kendall:
        with T("row_kendall"):
            per, rows_with_ties = st.row_kendall(DA, DB, n_jobs=n_jobs)
            out["Delta_O"] = float(per.mean())
            out["Delta_O_rows_with_ties"] = rows_with_ties

    # ---- global edge order ---------------------------------------------------
    with T("global_order"):
        out.update(st.global_order(DA, DB, key=global_order_key, max_pairs=global_order_max_pairs))

    # ---- MST -----------------------------------------------------------------
    with T("mst"):
        TA = st.mst_prim(DA)
        TB = st.mst_prim(DB)
        FA = st.nn1_forest(nA[:, 0])
        FB = st.nn1_forest(nB[:, 0])
        out.update(st.mst_compare(TA, TB, FA, FB))
        out["s_N_k1_for_mst"] = st.knn_overlap(nA, nB, 1)
        if not out["forest_in_mst_A"] or not out["forest_in_mst_B"]:
            raise AssertionError("sanity check failed: 1-NN forest not contained in MST")

    # ---- geometry ------------------------------------------------------------
    with T("gs"):
        out.update(geo.gs(DA, DB))
        out["raw_G_applicable"] = same_pre(A, B)
        if out["raw_G_applicable"]:
            out.update(geo.raw_g(DA, DB))
    with T("cka"):
        out.update(geo.cka_pair(A, B))
    with T("rsa"):
        out["rsa_pearson"] = geo.pearson_rsa(DA, DB)
    with T("sog"):
        if sog_k <= m - 2:
            out.update(geo.sog_overlaps(DA, DB, nA, nB, TA, TB, k=sog_k, taus=sog_taus))

    # ---- quotients -----------------------------------------------------------
    if with_quotients:
        with T("quotients"):
            out.update(qu.quotient_ladder(DA, DB, h1, n_jobs=n_jobs, with_row=with_row_iso))

    # ---- GL ------------------------------------------------------------------
    if with_gl and A.X is not None and B.X is not None:
        with T("gl"):
            out.update(glmod.projector_distance(A.X, B.X))

    for k, v in T.times.items():
        out[f"time_{k}_s"] = round(v, 4)
    out["time_total_s"] = round(sum(T.times.values()), 4)
    if return_aux:
        return out, {"nA": nA, "nB": nB, "vA": vA, "vB": vB, "TA": TA, "TB": TB, "FA": FA,
                     "FB": FB, "h1": h1}
    return out
