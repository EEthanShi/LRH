"""Where do the P-Bern float64 artefacts sit?  (companion to pbern_tie_diagnostic.py)

Plant (as in pbern_scale_check.py): L2 cloud, K_B = exp(beta (K_A - 1)).
Three float64 constructions of D_B, all exactly monotone in K_A in real arithmetic:
  general    : SPEC §0 formula  K_B,ii + K_B,jj - 2 K_B,ij   (what ladder_ref uses on Gram input)
  unit       : 2 - 2 exp(beta (K_A,ij - 1))                    (diagonal forced to exactly 1)
  unit_expm1 : -2 expm1(beta (K_A,ij - 1))
For each anchor row, candidates are sorted by (D_A, j); we record adjacent strict
descents of D_B (order flips) and adjacent D_B equalities with D_A strictly larger
(B-only ties), with the D_A rank-quantile (0 = nearest, 1 = farthest) where they occur.
Also the diagonal deviations |K_ii - 1| that drive the 'general' flips.
Usage: python pbern_tie_anatomy.py OUT_JSON m,p,beta [m,p,beta ...]"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from pbern_scale_check import plant_cloud  # noqa: E402
from hrhv2_ref.core import gram_sym, preprocess, sqdist_from_gram  # noqa: E402


def anatomy(DA, DB):
    m = DA.shape[0]
    flips_q, ties_q = [], []
    n_flip_rows = n_tie_rows = 0
    first_flip_rank = []
    for i in range(m):
        a = np.delete(DA[i], i)
        b = np.delete(DB[i], i)
        o = np.lexsort((np.arange(m - 1), a))
        a_s, b_s = a[o], b[o]
        desc = np.flatnonzero(b_s[1:] < b_s[:-1])
        tie = np.flatnonzero((b_s[1:] == b_s[:-1]) & (a_s[1:] > a_s[:-1]))
        if desc.size:
            n_flip_rows += 1
            first_flip_rank.append(int(desc[0]) + 1)
        if tie.size:
            n_tie_rows += 1
        flips_q.append(desc / (m - 2))
        ties_q.append(tie / (m - 2))
    fq = np.concatenate(flips_q)
    tq = np.concatenate(ties_q)

    def summ(q):
        if q.size == 0:
            return {"count": 0}
        return {"count": int(q.size), "min_rank_quantile": float(q.min()),
                "median_rank_quantile": float(np.median(q))}

    return {"adjacent_flips": summ(fq), "adjacent_Bonly_ties": summ(tq),
            "rows_with_flip": n_flip_rows, "rows_with_Bonly_tie": n_tie_rows,
            "min_rank_of_first_flip": (min(first_flip_rank) if first_flip_rank else None)}


res = []
for cell in sys.argv[2:]:
    m, p, beta = int(cell.split(",")[0]), int(cell.split(",")[1]), float(cell.split(",")[2])
    Xl, _ = preprocess(plant_cloud(m, p), "l2")
    KA = gram_sym(Xl)
    KB = np.exp(beta * (KA - 1.0))
    dgA = np.diag(KA) - 1.0
    dgB = np.diag(KB) - 1.0
    DA = sqdist_from_gram(KA)
    DB_general = sqdist_from_gram(KB)
    DB_unit = 2.0 - 2.0 * KB
    np.fill_diagonal(DB_unit, 0.0)
    DB_expm1 = -2.0 * np.expm1(beta * (KA - 1.0))
    np.fill_diagonal(DB_expm1, 0.0)
    rec = {"m": m, "p": p, "beta": beta,
           "frac_KA_diag_ne_1": float(np.mean(dgA != 0)), "max_abs_KA_diag_minus_1": float(np.abs(dgA).max()),
           "frac_KB_diag_ne_1": float(np.mean(dgB != 0)), "max_abs_KB_diag_minus_1": float(np.abs(dgB).max()),
           "min_offdiag_KB": float(KB[~np.eye(m, dtype=bool)].min())}
    for name, DB in (("general", DB_general), ("unit", DB_unit), ("unit_expm1", DB_expm1)):
        rec[name] = anatomy(DA, DB)
    res.append(rec)
    print(json.dumps(rec), flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
