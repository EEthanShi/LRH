#!/usr/bin/env python
"""PYT-AXIS COPY of analysis/s12_layermax_openllama/indep/validate.py (sha256 08f5d041...), sub-role pyt-code-I (S0b,
2026-10-10). Changes (marked PYT-AXIS): order gate first (plan F1(a)); PYT_THREADS; the I2 layer-pair list = plan V-P3
pairs (6.9b, DINOv2-g), (6.9b, CLIP-H ft-in12k), (70m, ViT-Ti in21k), (70m, MAE-H), each at (first, first),
(middle, middle), (last, last) x {id, pi_0, pi_1, pi_499} x both pre; the plan V-P7(b) closed-form check of
Proposition 5(3) on the same 12 entries (both pre); I3 (fixed-depth rows of the S1.2 panel, V-O4a) is dropped (no
Pythia model is in the S1.2 panel; V-P4 replaces it, see vl_checks.py); outputs are kept aside, never overwritten.

usage: launch_pyt.sh pyt_I_validate 2 -- venv/bin/python analysis/s12_layermax_pythia/indep/validate.py
       (CPU only; reads indep/<pre>/pairs/ and prep_meta.json; writes indep/validation.json)
       [--pres P,..] [--items I1,I2] [--pairs-root R (S0b test mode only)] [--K-test k (S0b test mode only: the
       permutations checked are {id} and pi_k for k < K-test)]

Validation of the independent layer-max run (PLAN_INDEP.md I1-I2).

I1  clip thresholds bitwise = the reference (frozen Part B for the Pythia models); exact ViT boundary ties
I2  12 layer pairs x 2 pre x {identity, pi_0, pi_1, pi_499}: run entries vs hrhv2 from scratch on B.permuted(pi)
    (mKNN exact; CKA biased/debiased <= 1e-6 absolute) = plan V-P3; plus V-P7(b)

Output: results/S1_2_layermax_pythia/indep/validation.json
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True   # PYT-AXIS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyt_gate  # noqa: E402  PYT-AXIS
OUT_ROOT = pyt_gate.gate(__file__)   # PYT-AXIS F1(a): exits 3 unless the plan's gates hold (S0b test mode: test root)

_T = os.environ.get("PYT_THREADS", "1")   # PYT-AXIS
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lmx_common as L  # noqa: E402
from lmx_common import C, st, geo  # noqa: E402

_S = dict(zip(C.SIZES, C.SLUGS))   # PYT-AXIS
I2_PAIRS = [(_S["6.9b"], "vit_giant_patch14_dinov2.lvd142m"),
            (_S["6.9b"], "vit_huge_patch14_clip_224.laion2b_ft_in12k"),
            (_S["70m"], "vit_tiny_patch16_224.augreg_in21k"),
            (_S["70m"], "vit_huge_patch14_224.mae")]   # PYT-AXIS: plan V-P3
I2_PERMS = (-1, 0, 1, 499)
TOL_CKA = 1e-6
PAIRS_ROOT = L.OUT   # --pairs-root (S0b tests)
PRES = L.PRES        # --pres


def load_pair(pre, lm, vit):
    return np.load(os.path.join(PAIRS_ROOT, pre, "pairs", f"{lm}__{vit}.npz"))


def vo7b(GA, GB):
    """plan V-P7(b), Proposition 5(3) in closed form for one layer pair: with K~ = K - diag K (K = the
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
        lm_ties = {s: [x["exact_boundary_ties_k10"] for x in meta[s]["layers"]] for s in meta if s not in L.VIT_STEMS}
        vit_near = max((max(x["near_tie_frac_k10"] for x in meta[s]["layers"]) for s in vit_ties), default=None)
        lm_near = max((max(x["near_tie_frac_k10"] for x in meta[s]["layers"]) for s in lm_ties), default=None)
        out[pre] = {"clip_all_bitwise_equal": (all(clip_eq) if clip_eq else None), "n_clip_checked": len(clip_eq),
                    "vit_layers_with_exact_boundary_ties": int(sum(1 for s in vit_ties for t in vit_ties[s] if t > 0)),
                    "vit_total_layers": int(sum(len(v) for v in vit_ties.values())),
                    "lm_layers_with_exact_boundary_ties": int(sum(1 for s in lm_ties for t in lm_ties[s] if t > 0)),
                    "lm_total_exact_boundary_tie_anchors": int(sum(sum(v) for v in lm_ties.values())),
                    "max_near_tie_frac_k10_vit": vit_near, "max_near_tie_frac_k10_lm": lm_near}
    return out


def i2(perms_checked=I2_PERMS, pairs=None):
    rows = []
    vo7 = []
    for lm, vit in (pairs or I2_PAIRS):
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
                for k in perms_checked:
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
            "V-P7b": {"what": "Proposition 5(3) closed form on the V-P3 entries (sum of the three expected HSIC_u "
                              "numerator terms relative to the largest term, must be <= 1e-10)",
                      "n": len(vo7), "max_rel_to_largest": float(max(r["rel_to_largest"] for r in vo7)),
                      "pass": bool(all(r["pass_1e-10"] for r in vo7)), "rows": vo7}}


def main():
    global PAIRS_ROOT, PRES
    ap = argparse.ArgumentParser(description="V-P3 (I2), I1 clip/tie summary and V-P7(b)")
    ap.add_argument("--pres", default=",".join(L.PRES), help="comma list of pre (default both)")
    ap.add_argument("--items", default="I1,I2", help="comma list of I1, I2 (default all)")
    ap.add_argument("--pairs-root", default=None, help="read pairs/prep_meta from <root>/<pre>/ (S0b test mode only)")
    ap.add_argument("--K-test", type=int, default=None, help="S0b test mode only: check id and pi_k, k < K-test")
    ap.add_argument("--known-pairs", action="store_true", help="S0b test mode only: run I2 on the V-P4 known pairs")
    args = ap.parse_args()
    PRES = tuple(args.pres.split(","))
    if (args.pairs_root or args.K_test or args.known_pairs) and not pyt_gate.test_mode():
        raise SystemExit("--pairs-root / --K-test / --known-pairs are S0b test options")
    if args.pairs_root:
        PAIRS_ROOT = args.pairs_root
    perms_checked = I2_PERMS if not args.K_test else tuple([-1] + list(range(min(args.K_test, 2))))
    items = args.items.split(",")
    res = {"pres": list(PRES), "pairs_root": PAIRS_ROOT, "test_mode": bool(pyt_gate.test_mode())}
    if "I1" in items:
        res["I1"] = i1()
        print("I1", json.dumps(res["I1"]), flush=True)
    if "I2" in items:
        res["I2"] = i2(perms_checked, list(C.KNOWN_VP4) if args.known_pairs else None)
        print("I2 (V-P3)", {k: v for k, v in res["I2"].items() if k not in ("rows", "V-P7b")}, flush=True)
        print("V-P7b", {k: v for k, v in res["I2"]["V-P7b"].items() if k != "rows"}, flush=True)
    L.write_json(os.path.join(L.OUT, "validation.json"), L.jsonable(res))   # PYT-AXIS: earlier copy kept aside


if __name__ == "__main__":
    main()
