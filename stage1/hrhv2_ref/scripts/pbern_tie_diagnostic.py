"""P-Bern float64 tie diagnostic.
general = SPEC §0 D formula on the Gram plant K_B = exp(beta (K_A - 1));
unit    = D_A = 2 - 2K_A, D_B = -2 expm1(beta (K_A - 1))  (exactly monotone construction).
Counts strict discordances and B-only ties (global edge order and per-anchor rows)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pbern_scale_check import plant_cloud  # noqa: E402
from hrhv2_ref.core import gram_sym, preprocess, sqdist_from_gram, upper  # noqa: E402
from hrhv2_ref import structural as st  # noqa: E402

cells = [(int(a.split(",")[0]), int(a.split(",")[1]), float(a.split(",")[2])) for a in sys.argv[1:]]
for m, p, beta in cells:
    X = plant_cloud(m, p)
    Xl, _ = preprocess(X, "l2")
    KA = gram_sym(Xl)
    DA, DB = sqdist_from_gram(KA), sqdist_from_gram(np.exp(beta * (KA - 1.0)))
    c1 = st.kendall_pair_counts(upper(DA), upper(DB))
    _, d1 = st.kendall_rows_merge(DA, DB, return_detail=True)
    DA2 = 2.0 - 2.0 * KA
    np.fill_diagonal(DA2, 0)
    DB2 = -2.0 * np.expm1(beta * (KA - 1.0))
    np.fill_diagonal(DB2, 0)
    c2 = st.kendall_pair_counts(upper(DA2), upper(DB2))
    _, d2 = st.kendall_rows_merge(DA2, DB2, return_detail=True)
    print(f"m={m} p={p} beta={beta} | general: E_strict_disc={c1['n_disc_strict']} E_Bonly_ties={c1['ties_B_only']} "
          f"rows_strict_disc_mean={d1[:, 0].mean():.3g} rows_Bonly_tie_mean={d1[:, 2].mean():.3g} | "
          f"unit+expm1: E_strict_disc={c2['n_disc_strict']} E_Bonly_ties={c2['ties_B_only']} "
          f"rows_strict_disc_mean={d2[:, 0].mean():.3g} rows_Bonly_tie_mean={d2[:, 2].mean():.3g}", flush=True)
