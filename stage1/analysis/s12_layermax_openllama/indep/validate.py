#!/usr/bin/env python
"""OPENLLAMA-AXIS COPY of analysis/s12_layermax_llama/indep/validate.py (sub-role I of
oll-axis-extract, S0b). Changes: Part B gate first (plan F1(a)); OL_THREADS; the I2 layer-pair list
= plan V-O3 pairs (13B, DINOv2-g), (13B, CLIP-H ft-in12k), (13B, ViT-Ti in21k), (13B, MAE-H), each at
(first, first), (middle, middle), (last, last) x {id, pi_0, pi_1, pi_499} x both pre; I3 rows = the
OpenLLaMA 3b/7b rows of the S1.2 panel (= plan V-O4a, mKNN exact, CKA <= 1e-10); the plan V-O7(b)
closed-form check of Proposition 5(3) on the same 12 entries (both pre); options --pres, --items,
--pairs-root (S0b tests on smoke outputs). I4 is not used (V-O6 lives in vl6_arh.py).

usage: launch_ol.sh ol_I_validate 2 -- venv/bin/python analysis/s12_layermax_openllama/indep/validate.py
       (CPU only; reads indep/<pre>/pairs/ and prep_meta.json; writes indep/validation.json)
(The LLaMA copy changed only the I2 pair list and the I3 rows.)

Validation of the independent layer-max run (PLAN_INDEP.md I1-I4).

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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ol_gate  # noqa: E402
OUT_ROOT = ol_gate.gate(__file__)   # F1(a): exits 3 unless Part B is frozen (S0b test mode: scratch root)

_T = os.environ.get("OL_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lmx_common as L  # noqa: E402
from lmx_common import C, st, geo  # noqa: E402

I2_PAIRS = [(C.SLUG13, "vit_giant_patch14_dinov2.lvd142m"),
            (C.SLUG13, "vit_huge_patch14_clip_224.laion2b_ft_in12k"),
            (C.SLUG13, "vit_tiny_patch16_224.augreg_in21k"),
            (C.SLUG13, "vit_huge_patch14_224.mae")]   # OPENLLAMA-AXIS: plan V-O3
I2_PERMS = (-1, 0, 1, 499)
TOL_CKA = 1e-6
PAIRS_ROOT = L.OUT   # --pairs-root (S0b tests: the smoke outputs)
PRES = L.PRES        # --pres


def load_pair(pre, lm, vit):
    return np.load(os.path.join(PAIRS_ROOT, pre, "pairs", f"{lm}__{vit}.npz"))


def vo7b(GA, GB):
    """plan V-O7(b), Proposition 5(3) in closed form for one layer pair: with K~ = K - diag K (K = the
    centred Gram the run uses) and S_K = 1'K~1, E_pi tr(K~ L~pi) = S_K S_L/(m(m-1)),
    E_pi (1'K~1)(1'L~pi 1) = S_K S_L and E_pi 1'K~ L~pi 1 = (K~1)'(L~1 averaged) = S_K S_L/m, so the
    expected HSIC_u numerator is t1 + t2/((m-1)(m-2)) - 2 t3/(m-2); it must vanish (<= 1e-10 of the
    largest term)."""
    from hrhv2.prep import center_gram
    m = GA.K.shape[0]
    Ka, Kb = center_gram(GA.K), center_gram(GB.K)
    Ka = Ka - np.diag(np.diag(Ka))
    Kb = Kb - np.diag(np.diag(Kb))
    SK, SL = float(Ka.sum()), float(Kb.sum())
    rK = Ka.sum(axis=1)
    e_tr = SK * SL / (m * (m - 1))
    e_prod = SK * SL
    e_1KL1 = float(np.sum(rK * (SL / m)))
    terms = [e_tr, e_prod / ((m - 1) * (m - 2)), -2.0 * e_1KL1 / (m - 2)]
    big = max(abs(t) for t in terms)
    tot = float(sum(terms))
    return {"S_K": SK, "S_L": SL, "terms": terms, "sum": tot, "rel_to_largest": (abs(tot) / big if big > 0 else 0.0),
            "pass_1e-10": bool(big == 0 or abs(tot) <= 1e-10 * big)}


def entry(z, name, k, a, b):
    if k < 0:
        return z[f"obs_{name}"][a, b]
    return z[f"null_{name}"][k, a, b]


def i1():
    out = {}
    for pre in PRES:
        meta = json.load(open(os.path.join(PAIRS_ROOT, pre, "prep_meta.json")))
        clip_eq = [v.get("clip_bitwise_equal") for v in meta.values() if "clip_bitwise_equal" in v]
        vit_ties = {s: [x["exact_boundary_ties_k10"] for x in meta[s]["layers"]] for s in L.VIT_STEMS if s in meta}
        lm_ties = {s: [x["exact_boundary_ties_k10"] for x in meta[s]["layers"]] for s in L.LM_STEMS if s in meta}
        vit_near = max(max(x["near_tie_frac_k10"] for x in meta[s]["layers"]) for s in vit_ties)
        lm_near = max(max(x["near_tie_frac_k10"] for x in meta[s]["layers"]) for s in lm_ties)
        out[pre] = {"clip_all_bitwise_equal": (all(clip_eq) if clip_eq else None), "n_clip_checked": len(clip_eq),
                    "vit_layers_with_exact_boundary_ties": int(sum(1 for s in vit_ties for t in vit_ties[s] if t > 0)),
                    "vit_total_layers": int(sum(len(v) for v in vit_ties.values())),
                    "lm_layers_with_exact_boundary_ties": int(sum(1 for s in lm_ties for t in lm_ties[s] if t > 0)),
                    "lm_total_exact_boundary_tie_anchors": int(sum(sum(v) for v in lm_ties.values())),
                    "max_near_tie_frac_k10_vit": vit_near, "max_near_tie_frac_k10_lm": lm_near}
    return out


def i2():
    rows = []
    vo7 = []
    for lm, vit in I2_PAIRS:
        SA, SB = L.load_stack64(lm), L.load_stack64(vit)
        pis = L.perms(lm, vit, L.K_PERM)
        for pre in PRES:
            cA = C.clip_threshold(lm, "wit1024") if pre == "prh_clip+l2" else None
            cB = C.clip_threshold(vit, "wit1024") if pre == "prh_clip+l2" else None
            z = load_pair(pre, lm, vit)
            LA, LB = SA.shape[1], SB.shape[1]
            for a, b in [(0, 0), ((LA - 1) // 2, (LB - 1) // 2), (LA - 1, LB - 1)]:
                GA = C.geom(C.preprocess_rows(SA[:, a, :], pre, cA), pre)
                GB = C.geom(C.preprocess_rows(SB[:, b, :], pre, cB), pre)
                vo7.append({"lm": lm, "vit": vit, "pre": pre, "a": a, "b": b, **vo7b(GA, GB)})
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
            "rows": rows,
            "V-O7b": {"what": "Proposition 5(3) closed form on the V-O3 entries (sum of the three expected HSIC_u "
                              "numerator terms relative to the largest term, must be <= 1e-10)",
                      "n": len(vo7), "max_rel_to_largest": float(max(r["rel_to_largest"] for r in vo7)),
                      "pass": bool(all(r["pass_1e-10"] for r in vo7)), "rows": vo7}}


def i3():
    panel = pd.read_parquet(os.path.join(C.RES, "panel_main.parquet"))
    panel = panel[panel.lm.isin(L.LM_STEMS) & panel.pre.isin(PRES)]   # OPENLLAMA-AXIS: OpenLLaMA 3b/7b rows
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
    global PAIRS_ROOT, PRES
    ap = argparse.ArgumentParser(description="V-O3 (I2), V-O4a (I3), I1 clip/tie summary and V-O7(b)")
    ap.add_argument("--arh", action="store_true", help="(LLaMA-era I4; not used on this axis, see vl6_arh.py)")
    ap.add_argument("--arh-metric", default=None, choices=("mknn", "cka_b"))
    ap.add_argument("--pres", default=",".join(L.PRES), help="comma list of pre (default both)")
    ap.add_argument("--items", default="I1,I3,I2", help="comma list of I1, I3, I2 (default all)")
    ap.add_argument("--pairs-root", default=None, help="read pairs/prep_meta from <root>/<pre>/ (S0b test mode only)")
    args = ap.parse_args()
    PRES = tuple(args.pres.split(","))
    if args.pairs_root:
        if not ol_gate.test_mode():
            raise SystemExit("--pairs-root is an S0b test option")
        PAIRS_ROOT = args.pairs_root
    if args.arh:
        res = i4(args.arh_metric)
        suffix = f"_{args.arh_metric}" if args.arh_metric else ""
        json.dump(L.jsonable(res), open(os.path.join(L.OUT, f"validation_arh{suffix}.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}), flush=True)
        return
    items = args.items.split(",")
    res = {"pres": list(PRES), "pairs_root": PAIRS_ROOT}
    if "I1" in items:
        res["I1"] = i1()
        print("I1", json.dumps(res["I1"]), flush=True)
    if "I3" in items:
        res["I3"] = i3()
        print("I3 (V-O4a)", json.dumps(res["I3"]), flush=True)
    if "I2" in items:
        res["I2"] = i2()
        print("I2 (V-O3)", {k: v for k, v in res["I2"].items() if k not in ("rows", "V-O7b")}, flush=True)
        print("V-O7b", {k: v for k, v in res["I2"]["V-O7b"].items() if k != "rows"}, flush=True)
    fn = os.path.join(L.OUT, "validation.json")
    json.dump(L.jsonable(res), open(fn + ".tmp", "w"), indent=1)
    os.replace(fn + ".tmp", fn)


if __name__ == "__main__":
    main()
