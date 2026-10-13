#!/usr/bin/env python
"""EXPLORATORY diagnostics (not in PLAN_INDEP.md): where does the layer-max CKA
null threshold come from?

For every pair and every layer entry (a, b):
  entry null mean mu0[a,b] = mean_k S_pi_k[a,b]  and entry q95 (476th order stat)
  predicted null mean   sqrt(PR_A^a PR_B^b) / (m - 1),  PR = tr(Kc)^2 / |Kc|_F^2
  (exact for the permutation expectation of <Kc_A, P Kc_B P'> with centred Grams:
   E = tr(Kc_A) tr(Kc_B) / (m - 1))
Writes results/S1_2_layermax/indep/diag_null_profile.json
"""
from __future__ import annotations

import os

_T = os.environ.get("LMX_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T
import json  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import lmx_common as L  # noqa: E402
from lmx_common import C, cal  # noqa: E402
from hrhv2.prep import center_gram  # noqa: E402


def trace_pr(stem, pre):
    S = L.load_stack64(stem)
    c = C.clip_threshold(stem, "wit1024") if pre == "prh_clip+l2" else None
    tr, pr = [], []
    for l in range(S.shape[1]):
        Xp = C.preprocess_rows(S[:, l, :], pre, c)
        Kc = center_gram(Xp @ Xp.T)
        t = float(np.trace(Kc))
        n2 = float(np.vdot(Kc.ravel(), Kc.ravel()))
        tr.append(t)
        pr.append(t * t / n2)
    return np.array(tr), np.array(pr)


def main():
    m = L.M
    idx = cal.order_index(L.K_PERM, L.ALPHA) - 1
    out = {}
    for pre in L.PRES:
        PR = {s: trace_pr(s, pre)[1] for s in L.LM_STEMS + L.VIT_STEMS}
        rows = []
        pred_err = []
        for lm in L.LM_STEMS:
            for vit in L.VIT_STEMS:
                z = np.load(os.path.join(L.OUT, pre, "pairs", f"{lm}__{vit}.npz"))
                N, S = z["null_cka_b"], z["obs_cka_b"]
                LA, LB = S.shape
                mu0 = N.mean(axis=0)
                comb = np.concatenate([S[None], N], axis=0)
                q = np.sort(comb, axis=0)[idx]                      # per-entry c_alpha
                pred = np.sqrt(PR[lm][:, None] * PR[vit][None, :]) / (m - 1)
                pred_err.append(float(np.max(np.abs(pred - mu0) / mu0)))
                Tk = N.reshape(N.shape[0], -1).max(axis=1)
                c_max = float(np.sort(np.append(Tk, S.max()))[idx])
                a, b = np.unravel_index(int(np.argmax(S)), S.shape)
                qa, qb = np.unravel_index(int(np.argmax(q)), q.shape)
                rows.append({"lm": lm, "vit": vit, "LA": LA, "LB": LB, "c_alpha_max": c_max,
                             "c_entry_sel": float(q[a, b]), "max_entry_c": float(q.max()),
                             "max_entry_c_rel_a": qa / (LA - 1), "max_entry_c_rel_b": qb / (LB - 1),
                             "mu0_sel": float(mu0[a, b]), "mu0_max": float(mu0.max()),
                             "sel_rel_a": a / (LA - 1), "sel_rel_b": b / (LB - 1),
                             "PR_A_sel": float(PR[lm][a]), "PR_A_max": float(PR[lm].max()),
                             "PR_A_argmax_rel": float(np.argmax(PR[lm]) / (LA - 1)),
                             "PR_B_sel": float(PR[vit][b]), "PR_B_max": float(PR[vit].max()),
                             "PR_B_argmax_rel": float(np.argmax(PR[vit]) / (LB - 1))})
        by_lm = {}
        for lm in L.LM_STEMS:
            R = [r for r in rows if r["lm"] == lm]
            by_lm[lm] = {k: float(np.mean([r[k] for r in R])) for k in
                         ("c_alpha_max", "c_entry_sel", "max_entry_c", "max_entry_c_rel_a", "max_entry_c_rel_b",
                          "mu0_sel", "mu0_max", "sel_rel_a", "PR_A_sel", "PR_A_max", "PR_A_argmax_rel")}
            by_lm[lm]["hidden"] = int(C.sidecar(lm)["hidden_size"])
            by_lm[lm]["L"] = int(R[0]["LA"])
        cm = np.array([r["c_alpha_max"] for r in rows])
        me = np.array([r["max_entry_c"] for r in rows])
        out[pre] = {
            "max_rel_err_pred_vs_entry_null_mean": float(np.max(pred_err)),
            "median_rel_err_pred_vs_entry_null_mean": float(np.median(pred_err)),
            "c_alpha_max_minus_max_entry_c": {"mean": float(np.mean(cm - me)), "min": float(np.min(cm - me)),
                                             "max": float(np.max(cm - me))},
            "spearman_c_alpha_max_vs_max_entry_c": float(spearmanr(cm, me).correlation),
            "frac_pairs_PR_A_argmax_ne_selected": float(np.mean([r["PR_A_sel"] < r["PR_A_max"] for r in rows])),
            "by_lm": by_lm,
            "PR_profile": {s: PR[s].tolist() for s in L.LM_STEMS + L.VIT_STEMS},
        }
        print(pre, json.dumps({k: v for k, v in out[pre].items() if k not in ("by_lm", "PR_profile")}), flush=True)
        for lm, d in by_lm.items():
            print(f"  {lm:34s} hid={d['hidden']:5d} L={d['L']} c_max={d['c_alpha_max']:.3f} c_sel={d['c_entry_sel']:.3f} "
                  f"maxentry_c={d['max_entry_c']:.3f}@(a={d['max_entry_c_rel_a']:.2f},b={d['max_entry_c_rel_b']:.2f}) "
                  f"sel_a={d['sel_rel_a']:.2f} PR_A sel={d['PR_A_sel']:.1f} max={d['PR_A_max']:.1f}@{d['PR_A_argmax_rel']:.2f}", flush=True)
    json.dump(L.jsonable(out), open(os.path.join(L.OUT, "diag_null_profile.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
