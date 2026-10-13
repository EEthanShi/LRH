"""Timing + sanity of calibrate_ref (recompute-from-scratch nulls) at m=1024.
Also checks the kNN relabelling shortcut N_{B,pi}(i) = pi^{-1}(N_B(pi(i))) used by
ARH's fast null against recomputation (identical on tie-free data)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hrhv2_ref import calibrate_ref, ladder_score, perm_from_key  # noqa: E402
from hrhv2_ref.core import gram_sym, neighbor_order, preprocess, sha256_int, sqdist_from_gram  # noqa: E402
from hrhv2_ref import structural as st  # noqa: E402

out_path = sys.argv[1]
m = 1024
rng = np.random.Generator(np.random.PCG64(sha256_int("hrhv2_ref/calib_bench")))
A = rng.normal(size=(m, 32))
B = np.tanh(A @ rng.normal(size=(32, 24)) / 3) + 0.4 * rng.normal(size=(m, 24))
res = {}
for name, K in (("s_N@10", 500), ("cka_debiased", 500), ("s_T", 500), ("quot_iso_expl", 20)):
    t0 = time.time()
    r = calibrate_ref(A, B, ladder_score(name, pre="l2"), K=K, key=f"bench|{name}")
    r.pop("null")
    r["seconds"] = round(time.time() - t0, 1)
    res[name] = r
    print(name, r, flush=True)
# relabelling shortcut check
Ap, _ = preprocess(A, "l2")
Bp, _ = preprocess(B, "l2")
NA = neighbor_order(sqdist_from_gram(gram_sym(Ap)), 10)
NB = neighbor_order(sqdist_from_gram(gram_sym(Bp)), 10)
agree = 0
for k in range(50):
    pi = perm_from_key("bench|relabel", k, m)
    inv = np.empty(m, dtype=np.int64)
    inv[pi] = np.arange(m)
    NBp_short = inv[NB[pi]]
    NBp_full = neighbor_order(sqdist_from_gram(gram_sym(Bp[pi])), 10)
    s1 = st.s_N(NA, NBp_short)
    s2 = st.s_N(NA, NBp_full)
    agree += int(s1 == s2)
res["relabel_shortcut_agree_of_50"] = agree
print("relabel agree", agree, flush=True)
json.dump(res, open(out_path, "w"), indent=1, default=float)
