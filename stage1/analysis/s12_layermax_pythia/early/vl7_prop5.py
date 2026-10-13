"""PYT-AXIS copy of vl7_prop5.py (implementation E, S0b sub-role pyt-code-E): plan V-P7. GATED (plan F1(a)).

Copied from analysis/s12_layermax_openllama/early/vl7_prop5.py (sha256 210d97f5...). Changes, marked "PYT-AXIS":
the E gate first; the pairs are the 119 Pythia-axis pairs, all new (plan section 11 "on the new pairs"); PRs come
from results/S1_2_layermax_pythia/partB/partB_features_pyt.json (per_pre.<pre>.PR.<stem>, the field of the
OpenLLaMA Part B; an E-side assumption about the Pythia Part B file, see early/CHANGES.md); the closed form (b)
uses the 12 V-P3 entries of this axis; the plant (c) has its own rng key; new item (d), the debiased-CKA second
moment (descriptive).

Proposition 5 (plan section 8), numerically. Compares the entry-wise permutation null means (500 pi,
float64, stored by the main run as <met>.entry_null_mean with <met>.entry_null_sd) with the exact
random-pairing baselines:
  * biased CKA:   B_uw = sqrt(PR_u PR_w)/(m-1); rule: median relative error <= 3% and |z| > 4 in <= 1%;
  * mKNN(10):     10/1023; rule: |z| > 4 in <= 1% ("within MC error");
  * debiased CKA: 0, checked four ways:
      (a) MC: |z| > 4 in <= 1% of the entries, z = (null mean - 0)/(null sd/sqrt(K));
      (b) closed form on the 12 V-P3 entries (6.9b x DINOv2-g, 6.9b x CLIP-H ft-in12k, 70m x ViT-Ti in21k,
          70m x MAE-H, each at first/middle/last stored layers, both pre): with S_K = 1'K~1 of the centred Gram
          with zeroed diagonal, the three terms of m(m-3) E_pi HSIC_u, S_K S_L/(m(m-1)), S_K S_L/((m-1)(m-2)) and
          -2 S_K S_L/(m(m-2)), sum to 0 to <= 1e-10 relative to the largest term (the bracket is also evaluated in
          exact rationals);
      (c) known-answer plant: 5 random pairs of m = 8 Gram matrices (rng util.rng_from_key(
          "S1.2-layermax-pythia|VP7c|<i>")), debiased CKA averaged over all 8! = 40320 permutations equals
          0 to <= 1e-12 absolute (batched numpy form of the GPU kernel's formula, cross-checked on single
          permutations against hrhv2.geometric.hsic_unbiased and lm_core.hsic_unbiased_gpu on CPU);
      (d) second moment (descriptive): over every entry of the 119 pairs, mean_k dCKA^2 = null_sd^2 (K-1)/K +
          null_mean^2 (from the stored float64 entry arrays), divided by 2/(m(m-3)) = 1.913e-6; the mean ratio is
          reported with its Monte Carlo yardstick sqrt(2/K) per entry ("within MC error" iff |mean ratio - 1| <=
          3 sqrt(2/K)); it is not part of the pass rule.
z = (null mean - baseline)/(null sd/sqrt(K)). Only null statistics are read (no observed score).

Run (step S7, after run_layermax.py; CPU, 2 threads):
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_pythia/launch_pyt.sh pyt_E_vp7 2 -- \\
      venv/bin/python analysis/s12_layermax_pythia/early/vl7_prop5.py [--pre prh_clip+l2]

Output: <RES>/validation/VP7_prop5.json (VP7_prop5.<pre>.json with --pre)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402  (constants and functions only; reads no data at import)

M = LC.M_ITEMS
BASE_MKNN = LC.KNN_K / (M - 1)
VP3_PAIRS = ((LC.SLUG["6.9b"], "vit_giant_patch14_dinov2.lvd142m"),  # PYT-AXIS: the V-P3 pairs of plan section 11
             (LC.SLUG["6.9b"], "vit_huge_patch14_clip_224.laion2b_ft_in12k"),
             (LC.SLUG["70m"], "vit_tiny_patch16_224.augreg_in21k"),
             (LC.SLUG["70m"], "vit_huge_patch14_224.mae"))
PRES_RUN = LC.PRES  # PYT-AXIS: set by --pre (prh_clip+l2 first; l2 only if it was run, plan section 11)


def null_mean_checks(pb, lms, label):
    """Entry null means vs Proposition 5 baselines over the pairs of the given LMs."""
    out = {"lms": list(lms), "per_pre": {}}
    for pre in PRES_RUN:  # PYT-AXIS
        P = pb["per_pre"][pre]["PR"]
        acc = {met: {"rel": [], "z": [], "n_sd0": 0} for met in ("cka_biased", "cka_debiased", "mknn")}
        per_lm = {}
        for lm in lms:
            prA = np.array(P[lm], dtype=np.float64)
            rl, zl = [], []
            for vit, _ in LC.VITS:
                prB = np.array(P[vit], dtype=np.float64)
                z = np.load(os.path.join(LC.pairs_dir(pre, lm), f"{lm}__{vit}.npz"))
                for met in acc:
                    nm = z[f"{met}.entry_null_mean"]
                    sd = z[f"{met}.entry_null_sd"]
                    if met == "cka_biased":
                        base = np.sqrt(prA[:, None] * prB[None, :]) / (M - 1)
                        assert base.shape == nm.shape, (lm, vit, base.shape, nm.shape)
                        r = np.abs(nm - base) / base
                        acc[met]["rel"].append(r.ravel())
                        rl.append(r.ravel())
                    else:
                        base = np.full_like(nm, 0.0 if met == "cka_debiased" else BASE_MKNN)
                    se = sd / np.sqrt(LC.K_PERM)
                    good = se > 0
                    acc[met]["n_sd0"] += int((~good).sum())
                    zz = np.where(good, (nm - base) / np.where(good, se, 1.0), np.nan)
                    acc[met]["z"].append(zz.ravel())
                    if met == "cka_biased":
                        zl.append(zz.ravel())
            rl, zl = np.concatenate(rl), np.concatenate(zl)
            per_lm[lm] = {"cka_biased_median_rel_err": float(np.median(rl)), "cka_biased_max_rel_err": float(rl.max()),
                          "cka_biased_frac_absz_gt4": float(np.nanmean(np.abs(zl) > 4))}
        R = {"per_lm": per_lm}
        for met, a in acc.items():
            zz = np.concatenate(a["z"])
            fin = np.isfinite(zz)
            d = {"n_entries": int(zz.size), "n_entries_null_sd_zero": a["n_sd0"],
                 "frac_absz_gt4": float(np.mean(np.abs(zz[fin]) > 4)), "max_absz": float(np.max(np.abs(zz[fin]))),
                 "mean_z": float(np.mean(zz[fin])), "sd_z": float(np.std(zz[fin]))}
            if a["rel"]:
                rr = np.concatenate(a["rel"])
                d.update({"median_rel_err": float(np.median(rr)), "p95_rel_err": float(np.percentile(rr, 95)),
                          "max_rel_err": float(rr.max())})
            R[met] = d
        R["pass"] = bool(R["cka_biased"]["median_rel_err"] <= 0.03
                         and all(R[m]["frac_absz_gt4"] <= 0.01 for m in ("cka_biased", "cka_debiased", "mknn")))
        out["per_pre"][pre] = R
        print(label, pre, json.dumps(LC.jsonable({m: R[m] for m in acc})), "PASS" if R["pass"] else "FAIL", flush=True)
    out["pass"] = all(v["pass"] for v in out["per_pre"].values())
    return out


def _s_tilde(stem, layer, pre):
    """S_K = 1'K~1 for the centred Gram Kc = HKH of one layer with zeroed diagonal (the main kernel's s_tot)."""
    c = LC.clip_value(stem) if pre == "prh_clip+l2" else None
    X = LC.layer_rows(stem, layer, pre, c)
    K = X @ X.T
    rm = K.mean(axis=1)
    cm = K.mean(axis=0)
    Kc = K - rm[:, None] - cm[None, :] + rm.mean()
    return float((Kc.sum(axis=0) - np.diag(Kc)).sum())


def closed_form_b():
    """(b): the plan section 8 bracket on the 12 V-P3 entries, both pre."""  # PYT-AXIS
    m = M
    exact = Fraction(1, m * (m - 1)) + Fraction(1, (m - 1) * (m - 2)) - Fraction(2, m * (m - 2))
    rows, worst = [], 0.0
    for pre in PRES_RUN:  # PYT-AXIS
        sA = {}
        for lm, vit in VP3_PAIRS:  # PYT-AXIS
            LA = LC.n_stored(lm)  # PYT-AXIS
            LB = LC.n_stored(vit)
            for la, lb in ((0, 0), (LC.fixed_layer(LA, 0.5), LC.fixed_layer(LB, 0.5)), (LA - 1, LB - 1)):
                if (lm, la) not in sA:  # PYT-AXIS
                    sA[(lm, la)] = _s_tilde(lm, la, pre)  # PYT-AXIS
                sB = _s_tilde(vit, lb, pre)
                p = sA[(lm, la)] * sB  # PYT-AXIS
                e1, e2, e3 = p / (m * (m - 1)), p / ((m - 1) * (m - 2)), -2.0 * p / (m * (m - 2))
                big = max(abs(e1), abs(e2), abs(e3))
                relr = abs(e1 + e2 + e3) / big if big > 0 else 0.0
                worst = max(worst, relr)
                rows.append({"pre": pre, "lm": lm, "vit": vit, "layer_A": la, "layer_B": lb, "S_K": sA[(lm, la)],  # PYT-AXIS
                             "S_L": sB, "terms": [e1, e2, e3], "sum": e1 + e2 + e3, "rel_to_largest": relr})
    return {"n_entries": len(rows), "exact_bracket_coefficient": str(exact), "exact_bracket_is_zero": exact == 0,
            "max_rel_to_largest": worst, "pass": bool(exact == 0 and worst <= 1e-10 and len(rows) == 12 * len(PRES_RUN)),  # PYT-AXIS
            "rows": rows}


def _hu_batched(Kc, Lp):
    """Unbiased HSIC (Song) of Kc against each matrix of the batch Lp (n, m, m): the formula of
    lm_core.hsic_unbiased_gpu / hrhv2.geometric.hsic_unbiased, vectorised."""
    m = Kc.shape[0]
    dK = np.diag(Kc)
    dL = np.einsum("nii->ni", Lp)
    t1 = np.einsum("ij,nij->n", Kc, Lp) - dL @ dK
    cK = Kc.sum(axis=0) - dK
    rL = Lp.sum(axis=2) - dL
    sK = cK.sum()
    sL = rL.sum(axis=1)
    t2 = sK * sL / ((m - 1) * (m - 2))
    t3 = 2.0 * (rL @ cK) / (m - 2)
    return (t1 + t2 - t3) / (m * (m - 3))


def plant_c():
    """(c): 5 random pairs of m = 8 Gram matrices, mean debiased CKA over all 8! permutations."""
    import torch
    from hrhv2 import geometric as geo, util
    m = 8
    perms = np.array(list(itertools.permutations(range(m))), dtype=np.int64)
    H = np.eye(m) - np.ones((m, m)) / m
    rows, worst, xchk = [], 0.0, 0.0
    for i in range(5):
        rng = util.rng_from_key(f"S1.2-layermax-pythia|VP7c|{i}")  # PYT-AXIS
        X = rng.standard_normal((m, 5))
        Y = rng.standard_normal((m, 4))
        Kc = H @ (X @ X.T) @ H
        Lc = H @ (Y @ Y.T) @ H
        Lp = Lc[perms[:, :, None], perms[:, None, :]]
        hu = _hu_batched(Kc, Lp)
        den = np.sqrt(_hu_batched(Kc, Kc[None])[0] * _hu_batched(Lc, Lc[None])[0])
        mean_dcka = float(np.mean(hu / den))
        for j in (0, 1, len(perms) - 1):   # single-permutation cross-checks of the batched formula
            ref = geo.hsic_unbiased(Kc, Lp[j])
            gpu_formula = float(LC.hsic_unbiased_gpu(torch.from_numpy(Kc), torch.from_numpy(np.ascontiguousarray(Lp[j]))))
            xchk = max(xchk, abs(ref - hu[j]), abs(gpu_formula - hu[j]))
        worst = max(worst, abs(mean_dcka))
        rows.append({"pair": i, "mean_dcka_over_8fact": mean_dcka, "mean_hsic_u_over_8fact": float(hu.mean()),
                     "dcka_identity": float(hu[0] / den), "n_perms": int(len(perms))})
    return {"n_pairs": 5, "m": m, "max_abs_mean": worst, "max_abs_crosscheck_vs_hrhv2_and_kernel": xchk,
            "pass": bool(worst <= 1e-12 and xchk <= 1e-12), "rows": rows}


def second_moment_d(lms):  # PYT-AXIS: (d), descriptive
    """mean over every entry of (mean_k dCKA^2) / (2/(m(m-3))), from the stored float64 entry null mean and sd."""
    K = LC.K_PERM
    base = 2.0 / (M * (M - 3))
    out = {"second_moment_theory": base, "per_pre": {}}
    for pre in PRES_RUN:
        ratios, per_lm = [], {}
        for lm in lms:
            rl = []
            for vit, _ in LC.VITS:
                z = np.load(os.path.join(LC.pairs_dir(pre, lm), f"{lm}__{vit}.npz"))
                nm = np.asarray(z["cka_debiased.entry_null_mean"], dtype=np.float64)
                sd = np.asarray(z["cka_debiased.entry_null_sd"], dtype=np.float64)
                ms = sd ** 2 * (K - 1) / K + nm ** 2
                rl.append((ms / base).ravel())
            rl = np.concatenate(rl)
            per_lm[lm] = {"mean_ratio": float(np.nanmean(rl)), "median_ratio": float(np.nanmedian(rl))}
            ratios.append(rl)
        rr = np.concatenate(ratios)
        mc = float(np.sqrt(2.0 / K))
        out["per_pre"][pre] = {"n_entries": int(rr.size), "mean_ratio": float(np.nanmean(rr)),
                               "median_ratio": float(np.nanmedian(rr)), "p5_ratio": float(np.nanpercentile(rr, 5)),
                               "p95_ratio": float(np.nanpercentile(rr, 95)), "mc_rel_sd_per_entry": mc,
                               "within_mc_error": bool(abs(float(np.nanmean(rr)) - 1.0) <= 3 * mc), "per_lm": per_lm,
                               "label": "descriptive (plan section 11 V-P7(d))"}
    return out


def main():
    global PRES_RUN  # PYT-AXIS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # PYT-AXIS
    ap.add_argument("--pre", nargs="*", default=list(LC.PRES), choices=list(LC.PRES))  # PYT-AXIS
    args = ap.parse_args()  # PYT-AXIS
    LC.gate("early/vl7_prop5.py")   # PYT-AXIS: plan F1(a), first action
    PRES_RUN = tuple(p for p in LC.PRES if p in args.pre)  # PYT-AXIS
    import torch
    torch.set_num_threads(LC.n_threads(2))
    pb = json.load(open(LC.PARTB_JSON))
    out = {"K": LC.K_PERM, "pres": list(PRES_RUN), "partB_json": LC.PARTB_JSON,  # PYT-AXIS
           "partB_json_sha256": LC.C12.sha256_file(LC.PARTB_JSON)}
    out["new_pairs_119"] = null_mean_checks(pb, LC.NEW_LMS, "new 119")  # PYT-AXIS
    out["debiased_b_closed_form"] = closed_form_b()
    print("V-P7(b)", out["debiased_b_closed_form"]["max_rel_to_largest"],
          "PASS" if out["debiased_b_closed_form"]["pass"] else "FAIL", flush=True)  # PYT-AXIS
    out["debiased_c_plant"] = plant_c()
    print("V-P7(c)", out["debiased_c_plant"]["max_abs_mean"], "PASS" if out["debiased_c_plant"]["pass"] else "FAIL",
          flush=True)  # PYT-AXIS
    out["debiased_d_second_moment"] = second_moment_d(LC.NEW_LMS)  # PYT-AXIS
    print("V-P7(d) (descriptive)", {p: (r["mean_ratio"], r["within_mc_error"])
                                    for p, r in out["debiased_d_second_moment"]["per_pre"].items()}, flush=True)  # PYT-AXIS
    out["pass"] = bool(out["new_pairs_119"]["pass"] and out["debiased_b_closed_form"]["pass"]  # PYT-AXIS
                       and out["debiased_c_plant"]["pass"])
    out["pass_rule"] = ("new_pairs_119 (biased median rel <= 3%, |z|>4 <= 1% for all 3 metrics, each pre run) and (b) "
                        "and (c); (d) descriptive")  # PYT-AXIS
    out["finished_utc"] = LC.utc()
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    name = "VP7_prop5.json" if PRES_RUN == LC.PRES else f"VP7_prop5.{'+'.join(PRES_RUN)}.json"  # PYT-AXIS
    with open(os.path.join(LC.RES, "validation", name), "w") as f:  # PYT-AXIS
        json.dump(LC.jsonable(out), f, indent=1)
    print("V-P7", "PASS" if out["pass"] else "FAIL", flush=True)  # PYT-AXIS


if __name__ == "__main__":
    main()
