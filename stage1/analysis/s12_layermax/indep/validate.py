#!/usr/bin/env python
"""Validation of the independent layer-max run (PLAN_INDEP.md I1-I4).

I1  clip thresholds bitwise = S1.2 cache; exact ViT boundary ties (relabelling exactness)
I2  12 layer pairs x 2 pre x {identity, pi_0, pi_1, pi_499}: GPU-run entries vs hrhv2
    from scratch on B.permuted(pi) (mKNN exact; CKA biased/debiased <= 1e-6)
I3  fixed-depth entries (S1.2 rule, r = 0.5/1.0) vs the S1.2 panel s_obs, 136 x 2 x 2
I4  vendored ARH calibrate() (one layer pair) and calibrate_layers(agg="max") on
    bloomz-560m x ViT-Ti/16-in21k (25 x 12), our permutations injected  [--arh]

Output: results/S1_2_layermax/indep/validation.json (I1-I3), validation_arh.json (I4)
"""
from __future__ import annotations

import os

_T = os.environ.get("LMX_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lmx_common as L  # noqa: E402
from lmx_common import C, st, geo  # noqa: E402

I2_PAIRS = [("bigscience__bloomz-560m", "vit_tiny_patch16_224.augreg_in21k"),
            ("huggyllama__llama-7b", "vit_giant_patch14_dinov2.lvd142m"),
            ("openlm-research__open_llama_7b", "vit_huge_patch14_clip_224.laion2b_ft_in12k"),
            ("bigscience__bloomz-7b1", "vit_huge_patch14_224.mae")]
I2_PERMS = (-1, 0, 1, 499)
TOL_CKA = 1e-6


def load_pair(pre, lm, vit):
    return np.load(os.path.join(L.OUT, pre, "pairs", f"{lm}__{vit}.npz"))


def entry(z, name, k, a, b):
    if k < 0:
        return z[f"obs_{name}"][a, b]
    return z[f"null_{name}"][k, a, b]


def i1():
    out = {}
    for pre in L.PRES:
        meta = json.load(open(os.path.join(L.OUT, pre, "prep_meta.json")))
        clip_eq = [v.get("clip_bitwise_equal") for v in meta.values() if "clip_bitwise_equal" in v]
        vit_ties = {s: [x["exact_boundary_ties_k10"] for x in meta[s]["layers"]] for s in L.VIT_STEMS}
        lm_ties = {s: [x["exact_boundary_ties_k10"] for x in meta[s]["layers"]] for s in L.LM_STEMS}
        vit_near = max(max(x["near_tie_frac_k10"] for x in meta[s]["layers"]) for s in L.VIT_STEMS)
        lm_near = max(max(x["near_tie_frac_k10"] for x in meta[s]["layers"]) for s in L.LM_STEMS)
        out[pre] = {"clip_all_bitwise_equal": (all(clip_eq) if clip_eq else None), "n_clip_checked": len(clip_eq),
                    "vit_layers_with_exact_boundary_ties": int(sum(1 for s in vit_ties for t in vit_ties[s] if t > 0)),
                    "vit_total_layers": int(sum(len(v) for v in vit_ties.values())),
                    "lm_layers_with_exact_boundary_ties": int(sum(1 for s in lm_ties for t in lm_ties[s] if t > 0)),
                    "lm_total_exact_boundary_tie_anchors": int(sum(sum(v) for v in lm_ties.values())),
                    "max_near_tie_frac_k10_vit": vit_near, "max_near_tie_frac_k10_lm": lm_near}
    return out


def i2():
    rows = []
    for lm, vit in I2_PAIRS:
        SA, SB = L.load_stack64(lm), L.load_stack64(vit)
        pis = L.perms(lm, vit, L.K_PERM)
        for pre in L.PRES:
            cA = C.clip_threshold(lm, "wit1024") if pre == "prh_clip+l2" else None
            cB = C.clip_threshold(vit, "wit1024") if pre == "prh_clip+l2" else None
            z = load_pair(pre, lm, vit)
            LA, LB = SA.shape[1], SB.shape[1]
            for a, b in [(0, 0), ((LA - 1) // 2, (LB - 1) // 2), (LA - 1, LB - 1)]:
                GA = C.geom(C.preprocess_rows(SA[:, a, :], pre, cA), pre)
                GB = C.geom(C.preprocess_rows(SB[:, b, :], pre, cB), pre)
                nA = st.knn_table(GA.D, L.KNN)[0]
                for k in I2_PERMS:
                    GBp = GB if k < 0 else GB.permuted(pis[k])
                    nB = st.knn_table(GBp.D, L.KNN)[0]
                    cnt_ref = int(st.overlap_counts(nA, nB, L.KNN).sum())
                    s_ref = st.knn_overlap(nA, nB, L.KNN)
                    ck = geo.cka_pair(GA, GBp)
                    cnt = int(entry(z, "cnt", k, a, b))
                    rows.append({"lm": lm, "vit": vit, "pre": pre, "a": a, "b": b, "perm": k,
                                 "cnt_gpu": cnt, "cnt_hrhv2": cnt_ref, "mknn_equal": cnt == cnt_ref,
                                 "s_hrhv2": s_ref, "s_gpu": cnt / (L.M * L.KNN),
                                 "cka_b_absdiff": abs(float(entry(z, "cka_b", k, a, b)) - ck["cka_biased"]),
                                 "cka_d_absdiff": abs(float(entry(z, "cka_d", k, a, b)) - ck["cka_debiased"]),
                                 "cka_b": ck["cka_biased"], "cka_d": ck["cka_debiased"]})
        print(f"I2 {lm} {vit} done", flush=True)
    df = pd.DataFrame(rows)
    return {"n_checks": int(len(df)), "mknn_all_equal": bool(df.mknn_equal.all()),
            "max_cka_b_absdiff": float(df.cka_b_absdiff.max()), "max_cka_d_absdiff": float(df.cka_d_absdiff.max()),
            "pass": bool(df.mknn_equal.all() and df.cka_b_absdiff.max() <= TOL_CKA and df.cka_d_absdiff.max() <= TOL_CKA),
            "rows": rows}


def i3():
    panel = pd.read_parquet(os.path.join(C.RES, "panel_main.parquet"))
    rows = []
    for _, r in panel.iterrows():
        z = load_pair(r["pre"], r["lm"], r["vit"])
        la, lb = int(r["layer_A"]), int(r["layer_B"])
        assert la == C.layer_index(r["lm"], r["r"]) and lb == C.layer_index(r["vit"], r["r"])
        s = int(z["obs_cnt"][la, lb]) / (L.M * L.KNN)
        rows.append({"lm": r["lm"], "vit": r["vit"], "r": r["r"], "pre": r["pre"],
                     "mknn_equal": s == float(r["cal.s_N_k10.s_obs"]),
                     "mknn_absdiff": abs(s - float(r["cal.s_N_k10.s_obs"])),
                     "cka_b_absdiff": abs(float(z["obs_cka_b"][la, lb]) - float(r["cal.cka_biased.s_obs"])),
                     "cka_d_absdiff": abs(float(z["obs_cka_d"][la, lb]) - float(r["L.cka_debiased"]))})
    df = pd.DataFrame(rows)
    return {"n_configs": int(len(df)), "mknn_all_equal": bool(df.mknn_equal.all()),
            "max_mknn_absdiff": float(df.mknn_absdiff.max()),
            "max_cka_b_absdiff": float(df.cka_b_absdiff.max()), "max_cka_d_absdiff": float(df.cka_d_absdiff.max()),
            "pass": bool(df.mknn_equal.all() and df.cka_b_absdiff.max() <= 1e-10 and df.cka_d_absdiff.max() <= 1e-10)}


def i4(only=None):
    import torch
    sys.path.insert(0, os.path.join(L.PROJ, "results", "verification"))
    os.environ.setdefault("VERIF_TORCH_THREADS", "4")
    import vendor_identity as vi  # verifier helpers: pinned PRH mutual_knn, ARH estimators/calibrate
    from calibrated_similarity.calibration import calibrate_layers as arh_calibrate_layers
    dev = torch.device("cuda")
    lm, vit, pre = "bigscience__bloomz-560m", "vit_tiny_patch16_224.augreg_in21k", "prh_clip+l2"
    SA, SB = L.load_stack64(lm), L.load_stack64(vit)
    cA, cB = C.clip_threshold(lm, "wit1024"), C.clip_threshold(vit, "wit1024")
    XA = [torch.from_numpy(C.preprocess_rows(SA[:, a, :], pre, cA)).to(dev) for a in range(SA.shape[1])]
    XB = [torch.from_numpy(C.preprocess_rows(SB[:, b, :], pre, cB)).to(dev) for b in range(SB.shape[1])]
    z = load_pair(pre, lm, vit)
    key = L.perm_key(lm, vit)
    sims = {"mknn": (lambda X, Y: vi.prh_mknn(X, Y, L.KNN)), "cka_b": (lambda X, Y: vi.arh_est.cka_biased(X, Y))}
    if only:
        sims = {only: sims[only]}
    out = {"pair": [lm, vit, pre], "key": key, "K": L.K_PERM}
    # ours
    def ours(name, S_obs, S_null, a=None, b=None):
        if a is None:
            T, null = float(S_obs.max()), S_null.reshape(S_null.shape[0], -1).max(axis=1)
        else:
            T, null = float(S_obs[a, b]), S_null[:, a, b]
        return L.summarize(T, null)
    mats = {"mknn": (z["obs_cnt"] / (L.M * L.KNN), z["null_cnt"].astype(np.float64) / (L.M * L.KNN)),
            "cka_b": (z["obs_cka_b"], z["null_cka_b"])}
    mats = {k: v for k, v in mats.items() if k in sims}
    a1, b1 = 12, 6     # single layer pair (= S1.2 r=0.5 layers of this pair)
    for name, sim in sims.items():
        t0 = time.time()
        feeder = vi.PermFeeder(L.M, key, L.K_PERM)
        sc, p, tau = vi.arh_calibrate(XA[a1], XB[b1], sim, K=L.K_PERM, alpha=L.ALPHA, smax=1.0, perm_fn=feeder)
        o = ours(name, *mats[name], a=a1, b=b1)
        out[f"scalar_{name}"] = {"layers": [a1, b1], "arh": {"s_cal": float(sc), "p": float(p), "tau": float(tau)},
                                 "ours": {k: o[k] for k in ("s_obs", "s_cal", "p_value", "c_alpha")},
                                 "absdiff": {"s_cal": abs(float(sc) - o["s_cal"]), "p": abs(float(p) - o["p_value"]),
                                             "tau": abs(float(tau) - o["c_alpha"])}, "secs": time.time() - t0}
        print("I4 scalar", name, out[f"scalar_{name}"]["absdiff"], flush=True)
        t0 = time.time()
        feeder = vi.PermFeeder(L.M, key, L.K_PERM)
        sc, p, tau = arh_calibrate_layers(XA, XB, sim, agg="max", K=L.K_PERM, alpha=L.ALPHA, smax=1.0, perm_fn=feeder)
        o = ours(name, *mats[name])
        out[f"layers_{name}"] = {"arh": {"s_cal": float(sc), "p": float(p), "tau": float(tau)},
                                 "ours": {k: o[k] for k in ("s_obs", "s_cal", "p_value", "c_alpha")},
                                 "absdiff": {"s_cal": abs(float(sc) - o["s_cal"]), "p": abs(float(p) - o["p_value"]),
                                             "tau": abs(float(tau) - o["c_alpha"])}, "secs": time.time() - t0}
        print("I4 layers", name, out[f"layers_{name}"]["absdiff"], flush=True)
    # observed layer-max matrix vs vendored sims (no permutation), to localise any tie effect
    for name, sim in sims.items():
        S = np.array([[sim(x, y) for y in XB] for x in XA])
        d = np.abs(S - mats[name][0])
        out[f"obsmatrix_{name}_maxabs"] = float(d.max())
        out[f"obsmatrix_{name}_n_diff_gt_1e-6"] = int((d > 1e-6).sum())
    diffs = [v for k, d in out.items() if isinstance(d, dict) and "absdiff" in d for v in d["absdiff"].values()]
    out["max_absdiff"] = float(max(diffs))
    out["pass_1e-6"] = bool(out["max_absdiff"] <= 1e-6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arh", action="store_true")
    ap.add_argument("--arh-metric", default=None, choices=("mknn", "cka_b"))
    args = ap.parse_args()
    if args.arh:
        res = i4(args.arh_metric)
        suffix = f"_{args.arh_metric}" if args.arh_metric else ""
        json.dump(L.jsonable(res), open(os.path.join(L.OUT, f"validation_arh{suffix}.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}), flush=True)
        return
    res = {"I1": i1(), "I3": i3()}
    print("I1", json.dumps(res["I1"]), "\nI3", json.dumps(res["I3"]), flush=True)
    res["I2"] = i2()
    print("I2", {k: v for k, v in res["I2"].items() if k != "rows"}, flush=True)
    json.dump(L.jsonable(res), open(os.path.join(L.OUT, "validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
