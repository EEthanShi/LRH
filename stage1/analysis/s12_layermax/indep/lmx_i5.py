#!/usr/bin/env python
"""I5: cross-implementation comparison with role arh-layermax-early
(results/S1_2_layermax/pairs/<pre>/*.{npz,json}, read-only) on every pair present.
Writes results/S1_2_layermax/indep/validation_i5.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import lmx_aggregate as A  # noqa: E402
import lmx_common as L  # noqa: E402

EARLY = os.path.join(L.PROJ, "results", "S1_2_layermax", "pairs")
MAP = {"mknn": "mknn", "cka_b": "cka_biased", "cka_d": "cka_debiased"}
SUBMAP = {"full": "full", "grid2": "grid2", "grid3": "grid3", "grid5": "grid5", "grid9": "grid9", "grid17": "grid17",
          "fixed_r0.5": "fixed_r0.5", "fixed_r1.0": "fixed_r1.0", "lmall_x_vitlast": "LMall_ViTlast",
          "lmlast_x_vitall": "LMlast_ViTall"}
FIELDS = ("s_obs", "c_alpha", "s_cal", "p_value", "uncensored", "null_mean")


def main():
    out = {}
    for pre in L.PRES:
        n, worst, worst_where = 0, {}, {}
        cnt_equal, nullmax_equal_mknn, argmax_equal = True, True, 0
        n_argmax = 0
        for lm in L.LM_STEMS:
            for vit in L.VIT_STEMS:
                fj = os.path.join(EARLY, pre, f"{lm}__{vit}.json")
                fz = os.path.join(EARLY, pre, f"{lm}__{vit}.npz")
                if not (os.path.exists(fj) and os.path.exists(fz)):
                    continue
                e = json.load(open(fj))
                ez = np.load(fz)
                z = np.load(os.path.join(L.OUT, pre, "pairs", f"{lm}__{vit}.npz"))
                assert e["perm_key"] == str(z["key"]) and int(e["K"]) == int(z["K"])
                n += 1
                mine_obs = {"mknn": z["obs_cnt"] / (L.M * L.KNN), "cka_b": z["obs_cka_b"], "cka_d": z["obs_cka_d"]}
                mine_null = {"mknn": z["null_cnt"].astype(np.float64) / (L.M * L.KNN), "cka_b": z["null_cka_b"],
                             "cka_d": z["null_cka_d"]}
                # raw arrays
                if not np.array_equal(ez["mknn.null_counts_int16"].astype(np.int64), z["null_cnt"].astype(np.int64)):
                    cnt_equal = False
                for met, em in MAP.items():
                    d_obs = float(np.max(np.abs(ez[f"{em}.entry_obs"] - mine_obs[met])))
                    my_max = mine_null[met].reshape(mine_null[met].shape[0], -1).max(axis=1)
                    d_nmax = float(np.max(np.abs(ez[f"{em}.full_null_max"] - my_max)))
                    for k, v in ((f"{met}.entry_obs", d_obs), (f"{met}.full_null_max", d_nmax)):
                        if v > worst.get(k, -1):
                            worst[k], worst_where[k] = v, f"{lm}|{vit}"
                    if met == "mknn" and d_nmax != 0.0:
                        nullmax_equal_mknn = False
                    ps = A.pair_stats(z, lm, vit)[met]
                    for sub, esub in SUBMAP.items():
                        es = e["summaries"][em][esub]
                        for f in FIELDS:
                            v = abs(float(es[f]) - float(ps[sub][f]))
                            k = f"{met}.{f}"
                            if v > worst.get(k, -1):
                                worst[k], worst_where[k] = v, f"{lm}|{vit}|{sub}"
                    n_argmax += 1
                    argmax_equal += int(e["summaries"][em]["full"]["argmax_A"] == ps["full"]["a"]
                                        and e["summaries"][em]["full"]["argmax_B"] == ps["full"]["b"])
        out[pre] = {"n_pairs_compared": n, "mknn_null_counts_identical": cnt_equal,
                    "mknn_full_null_max_identical": nullmax_equal_mknn,
                    "argmax_identical": f"{argmax_equal}/{n_argmax}", "max_absdiff": worst, "where": worst_where}
        print(pre, json.dumps(out[pre], indent=1), flush=True)
    json.dump(L.jsonable(out), open(os.path.join(L.OUT, "validation_i5.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
