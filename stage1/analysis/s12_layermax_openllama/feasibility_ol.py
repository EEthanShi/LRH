#!/usr/bin/env python
"""OpenLLaMA axis: feasibility and power tables of plan section 12 (and the section 0 known-values table),
recomputed on ds2213 from KNOWN files only (step S0b). EXPLORATORY design calculations, not citable as results.
New file, written at S0b by sub-role PB of oll-axis-extract.

It ports the exploratory scripts the designer and the critic ran locally on pulled numbers
(scratchpad/design/{pull,known_summary,joint,curves,scen,band}.py and scratchpad/critic/{condsize,material,
arh11,forecast}.py) and re-derives every input from the frozen server files:
  results/S1_2_layermax/pairs/<pre>/<lm>__<vit>.{json,npz}          8-LM panel (BLOOMZ, OpenLLaMA 3B/7B, llama-7b)
  results/S1_2_layermax_llama/early/pairs/<pre>/<lm>__<vit>.json    llama-13b/30b (and the llama-7b rerun, for the
                                                                    E-key ARH CI reproduction)
  results/S1_2_layermax_llama/partB/partB_features.json             per-layer PRs (frozen LLaMA Part B)
No file of open_llama_13b is read (none exists at S0b). UNGATED: no cross-model score involving 13B.
Every band, threshold and rule is a numeric constant of the plan; nothing here can move one (plan R.3.3).
Differences between these server numbers and the plan's design numbers are listed in the output and logged by
the parent role.

Tables (plan section 12): minimal attainable p; J_min given the known families (n_B(t), n_L(t)); J_mean
evidential content (designer's placement scenarios + 3000 random configurations, rng default_rng(0)); conditional
size of v0's C-J rule (critic's flip count, and the VERDICTS_SPEC f- count for comparison); scenario model
(PR*(13B) x k); materiality threshold PR* >= 241; P-O2b decidability; ARH pooled reading with 13B added
(critic rng default_rng(1) on alphabetically sorted ViTs, plus the E-key CI of the 10-LM panel, LLaMA key
S1.2-layermax-llama|boot|arh_delta|prh_clip+l2|cka_biased, which must give [0.0029, 0.2613]); resources
where computable. Plus the section 0 table of OpenLLaMA 3B/7B known values (the source of the band).

Outputs: results/S1_2_layermax_openllama/feasibility/{feasibility_ol.json, FEASIBILITY_OL.md}
         (S0b test mode OL_S0B_TEST=1: ~/oll_s0b_scratch/PB/feasibility/)

RUN (S0b, once; CPU, 2 threads, about 1-2 min):
  cd ~/hrh_v2_stage1_20261008
  analysis/s12_layermax_openllama/launch_ol.sh feasibility_ol 2 -- venv/bin/python analysis/s12_layermax_openllama/feasibility_ol.py
"""
from __future__ import annotations

import os

_TH = os.environ.get("OL_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _TH

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_openllama")
sys.path.insert(0, PROJ)
sys.path.insert(0, AXIS)

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import guard  # noqa: E402  (test mode only)
from hrhv2 import util  # noqa: E402

RES = os.path.join(PROJ, "results")
FROZEN8 = os.path.join(RES, "S1_2_layermax", "pairs")
LLAMA_E = os.path.join(RES, "S1_2_layermax_llama", "early", "pairs")
LLAMA_PB = os.path.join(RES, "S1_2_layermax_llama", "partB", "partB_features.json")
LLAMA_PB_SHA = "ede0127589a02037427b343e9c3593ee427d430e9746d7f4942b15f76a3031bf"
OUT_REAL = os.path.join(RES, "S1_2_layermax_openllama", "feasibility")
PRES = ("prh_clip+l2", "l2")
METS = ("mknn", "cka_biased", "cka_debiased")
PROTS = ("full", "fixed_r0.5", "fixed_r1.0")
BF = ["bigscience__bloomz-560m", "bigscience__bloomz-1b1", "bigscience__bloomz-1b7", "bigscience__bloomz-3b",
      "bigscience__bloomz-7b1"]
LF = ["huggyllama__llama-7b", "huggyllama__llama-13b", "huggyllama__llama-30b"]
O3, O7 = "openlm-research__open_llama_3b", "openlm-research__open_llama_7b"
LM8 = BF + [O3, O7, "huggyllama__llama-7b"]
VITS_REG = ["vit_tiny_patch16_224.augreg_in21k", "vit_small_patch16_224.augreg_in21k",
            "vit_base_patch16_224.augreg_in21k", "vit_large_patch16_224.augreg_in21k", "vit_base_patch16_224.mae",
            "vit_large_patch16_224.mae", "vit_huge_patch14_224.mae", "vit_small_patch14_dinov2.lvd142m",
            "vit_base_patch14_dinov2.lvd142m", "vit_large_patch14_dinov2.lvd142m", "vit_giant_patch14_dinov2.lvd142m",
            "vit_base_patch16_clip_224.laion2b", "vit_large_patch14_clip_224.laion2b",
            "vit_huge_patch14_clip_224.laion2b", "vit_base_patch16_clip_224.laion2b_ft_in12k",
            "vit_large_patch14_clip_224.laion2b_ft_in12k", "vit_huge_patch14_clip_224.laion2b_ft_in12k"]
VITS = sorted(VITS_REG)          # the design scripts used sorted(set(vit)); order matters only for the rng CIs
RANK = {"bigscience__bloomz-560m": 1, "bigscience__bloomz-1b1": 2, "bigscience__bloomz-1b7": 3,
        "bigscience__bloomz-3b": 4, "bigscience__bloomz-7b1": 5, O3: 6, O7: 7, "huggyllama__llama-7b": 9,
        "huggyllama__llama-13b": 10, "huggyllama__llama-30b": 11}
TOL = 1e-9


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------
# loading (= design/pull.py, from the frozen server files)
# ------------------------------------------------------------------------------------------------

def load(inputs: dict) -> dict:
    D = {"pairs": {}, "pr": {}}
    for pre in PRES:
        for lm in LM8 + LF[1:]:
            base = FROZEN8 if lm in LM8 else LLAMA_E
            for v in VITS:
                p = os.path.join(base, pre, f"{lm}__{v}")
                with open(p + ".json") as f:
                    j = json.load(f)
                assert j["lm"] == lm and j["vit"] == v and j["pre"] == pre, p
                inputs[os.path.relpath(p + ".json", PROJ)] = None
                rec = {"L_A": j["L_A"], "L_B": j["L_B"], "m": {}}
                z = np.load(p + ".npz")
                inputs[os.path.relpath(p + ".npz", PROJ)] = None
                for met in METS:
                    S_ = j["summaries"][met]
                    r = {prot: {k: S_[prot].get(k) for k in ("s_obs", "c_alpha", "s_cal", "uncensored", "null_mean",
                                                             "argmax_A", "argmax_B")} for prot in PROTS}
                    obs, c, mu = z[met + ".entry_obs"], z[met + ".entry_c_alpha"], z[met + ".entry_null_mean"]
                    scal = np.maximum((obs - c) / (1 - c), 0.0)
                    a = np.unravel_index(np.argmax(scal), scal.shape)
                    r["ctm"] = {"M": float(scal.max()), "argmax_A": int(a[0]), "argmax_B": int(a[1]),
                                "c_at_argmax": float(c[a]), "M_unc": float((obs - mu).max())}
                    rec["m"][met] = r
                D["pairs"][(pre, lm, v)] = rec
    pb = json.load(open(LLAMA_PB))
    inputs[os.path.relpath(LLAMA_PB, PROJ)] = None
    for pre in PRES:
        D["pr"][pre] = {"lm": {k: np.array(v["PR"]) for k, v in pb["per_pre"][pre]["lm"].items()},
                        "vit": {k: np.array(v["PR"]) for k, v in pb["per_pre"][pre]["vit"].items()}}
    return D


def vec(D, pre, lm, met, field, prot="full"):
    return np.array([D["pairs"][(pre, lm, v)]["m"][met][prot][field] for v in VITS], dtype=float)


def trallN(X):
    """design trallN: X = vits x sizes; Tr for every relabelling (identity first); all-equal rows excluded."""
    n = X.shape[1]
    res = []
    for perm in itertools.permutations(range(n)):
        lab = np.array(perm, float) + 1
        res.append(np.mean([np.corrcoef(rankdata(r), lab)[0, 1] for r in X if not np.all(r == r[0])]))
    return np.array(res)


def Mx(D, pre, lms, field, met="cka_biased"):
    return np.stack([vec(D, pre, l, met, field) for l in lms], 1)


# ------------------------------------------------------------------------------------------------
# section 0 table (design/known_summary.py, critic/material.py extras)
# ------------------------------------------------------------------------------------------------

def known_table(D):
    out = {}
    for pre in PRES:
        prl, prv = D["pr"][pre]["lm"], D["pr"][pre]["vit"]
        PRv = np.array([prv[v].max() for v in VITS])
        B3 = np.sqrt(prl[O3].max() * PRv) / 1023
        B7 = np.sqrt(prl[O7].max() * PRv) / 1023
        T = {"3b": vec(D, pre, O3, "cka_biased", "s_obs"), "7b": vec(D, pre, O7, "cka_biased", "s_obs")}
        c = {"3b": vec(D, pre, O3, "cka_biased", "c_alpha"), "7b": vec(D, pre, O7, "cka_biased", "c_alpha")}
        nm = {"3b": vec(D, pre, O3, "cka_biased", "null_mean"), "7b": vec(D, pre, O7, "cka_biased", "null_mean")}
        off = np.r_[c["3b"] - B3, c["7b"] - B7]
        moff = np.r_[nm["3b"] - B3, nm["7b"] - B7]
        rows = {}

        def row(name, a, b):
            rows[name] = {"3b": float(np.mean(a)), "7b": float(np.mean(b)), "n_7b_gt_3b": int(np.sum(b > a)),
                          "n_tied": int(np.sum(b == a))}
        for met, f, prot, lab in (("cka_biased", "s_obs", "full", "raw T"), ("cka_biased", "c_alpha", "full", "c_alpha"),
                                  ("cka_biased", "s_cal", "full", "s_cal"), ("cka_biased", "uncensored", "full", "uncensored"),
                                  ("cka_biased", "M", "ctm", "CTM M"), ("cka_biased", "c_at_argmax", "ctm", "c at M-argmax"),
                                  ("mknn", "s_obs", "full", "mKNN T"), ("mknn", "c_alpha", "full", "mKNN c_alpha"),
                                  ("cka_debiased", "s_obs", "full", "debiased T"),
                                  ("cka_biased", "s_obs", "fixed_r0.5", "fixed r=0.5 T"),
                                  ("cka_biased", "c_alpha", "fixed_r0.5", "fixed r=0.5 c_alpha"),
                                  ("cka_biased", "s_obs", "fixed_r1.0", "fixed r=1.0 T"),
                                  ("cka_biased", "c_alpha", "fixed_r1.0", "fixed r=1.0 c_alpha")):
            row(lab, vec(D, pre, O3, met, f, prot), vec(D, pre, O7, met, f, prot))
        row("Bbar", B3, B7)
        prof = {}
        for lm in (O3, O7):
            v = prl[lm]
            L = v.size
            srt = np.sort(v)[::-1]
            argA = vec(D, pre, lm, "cka_biased", "argmax_A")
            ctmA = vec(D, pre, lm, "cka_biased", "argmax_A", "ctm")
            prof[lm] = {"L_stored": L, "PR_star": float(v.max()), "u_star": int(np.argmax(v)),
                        "u_star_rel": float(np.argmax(v) / (L - 1)), "runner_up_ratio": float(srt[0] / srt[1]),
                        "PR_emb": float(v[0]), "PR_last": float(v[-1]),
                        "PR_r0.5": float(v[int(math.floor(0.5 * (L - 1) + 0.5))]),
                        "PR_at_ctm_argmax_median": float(np.median([v[int(a)] for a in ctmA])),
                        "layermax_argmax_rel_mean": float(np.mean(argA / (L - 1))),
                        "ctm_argmax_rel_min": float(np.min(ctmA / (L - 1))), "ctm_argmax_rel_max": float(np.max(ctmA / (L - 1))),
                        "ctm_argmax_rel_mean": float(np.mean(ctmA / (L - 1)))}
        gaps = {"Bbar_7b_minus_3b": {"median": float(np.median(B7 - B3)), "min": float(np.min(B7 - B3)),
                                     "n_ge_0.005": int(np.sum(B7 - B3 >= 0.005))}}
        for r, l3, l7 in (("0.5", 13, 16), ("1.0", 26, 32)):
            g = np.array([math.sqrt(prv[v][int(math.floor(float(r) * (prv[v].size - 1) + 0.5))]) for v in VITS]) / 1023 * \
                (math.sqrt(prl[O7][l7]) - math.sqrt(prl[O3][l3]))
            gaps[f"Bfd_{r}_7b_minus_3b"] = {"median": float(np.median(g)), "n_ge_0.005": int(np.sum(g >= 0.005))}
        offL = {}
        for lm in LF:
            o = vec(D, pre, lm, "cka_biased", "c_alpha") - np.sqrt(prl[lm].max() * PRv) / 1023
            offL[lm] = {"median": float(np.median(o)), "min": float(o.min()), "max": float(o.max())}
        allL = np.concatenate([vec(D, pre, lm, "cka_biased", "c_alpha") - np.sqrt(prl[lm].max() * PRv) / 1023 for lm in LF])
        out[pre] = {"means_over_17_vits": rows, "pr_profiles": prof, "gaps": gaps,
                    "calpha_minus_Bbar_34": {"median": float(np.median(off)), "min": float(off.min()), "max": float(off.max()),
                                             "max_abs_dev_from_median": float(np.max(np.abs(off - np.median(off))))},
                    "null_mean_minus_Bbar_34": {"median": float(np.median(moff)), "min": float(moff.min()),
                                                "max": float(moff.max())},
                    "llama_calpha_minus_Bbar": offL,
                    "llama_all_102": {"min": float(allL.min()), "max": float(allL.max()), "n": int(allL.size)},
                    "raw_gain_3b_7b_median": float(np.median(T["7b"] - T["3b"])),
                    "threshold_gain_3b_7b_median": float(np.median(c["7b"] - c["3b"]))}
    return out


# ------------------------------------------------------------------------------------------------
# J tables (design/curves.py, design/joint.py) and minimal p
# ------------------------------------------------------------------------------------------------

def family_dtr(D, pre, x):
    f = {"s_cal": "s_cal", "unc": "uncensored"}[x]
    dB = trallN(Mx(D, pre, BF, f)) - trallN(Mx(D, pre, BF, "s_obs"))
    dL = trallN(Mx(D, pre, LF, f)) - trallN(Mx(D, pre, LF, "s_obs"))
    return dB, dL


def jtables(D):
    pre = "prh_clip+l2"
    out = {}
    for x in ("s_cal", "unc"):
        dB, dL = family_dtr(D, pre, x)
        rows = []
        for t in (-0.912, -0.60, -0.50, -0.30, -0.15, -0.10, 0.00):
            tt = max(t, dB[0], dL[0])
            nB, nL = int(np.sum(dB <= tt + TOL)), int(np.sum(dL <= tt + TOL))
            rows.append({"sO_dTrO": t, "t": float(tt), "n_B": nB, "n_L": nL,
                         "p_nO_1_2_3": [nB * nL * k / 4320 for k in (1, 2, 3)]})
        BL = (dB[:, None] + dL[None, :]) / 2
        t0 = max(dB[0], dL[0])
        out[x] = {"dTr_B": float(dB[0]), "rank_B": int(np.sum(dB <= dB[0] + TOL)),
                  "dTr_L": float(dL[0]), "rank_L": int(np.sum(dL <= dL[0] + TOL)),
                  "dTr_L_all_sorted": sorted(float(v) for v in dL), "BL_mean_rank_720": int(np.sum(BL <= BL[0, 0] + TOL)),
                  "Jmin_min_count_nO1": int(np.sum(dB <= t0 + TOL)) * int(np.sum(dL <= t0 + TOL)),
                  "Jmean_min_count": int(np.sum(BL <= BL[0, 0] + TOL)), "rows": rows}
    return out


def jmean_scenarios(D):
    """design/joint.py (s_cal, full): 13B placed low/mid/high per ViT; 3000 random placements, default_rng(0)."""
    pre = "prh_clip+l2"
    dB, dL = family_dtr(D, pre, "s_cal")
    XO_raw2 = np.stack([vec(D, pre, l, "cka_biased", "s_obs") for l in (O3, O7)], 1)
    XO_cal2 = np.stack([vec(D, pre, l, "cka_biased", "s_cal") for l in (O3, O7)], 1)

    def tr_all(X):
        n = X.shape[1]
        res = []
        for perm in itertools.permutations(range(n)):
            lab = np.array(perm, float) + 1
            res.append(np.array([np.corrcoef(rankdata(r), lab)[0, 1] for r in X if not np.all(r == r[0])]).mean())
        return np.array(res)

    def place(X, pos):
        lo, hi = X.min(1), X.max(1)
        span = np.maximum(hi - lo, 1e-6)
        xx = {"low": lo - 0.5 * span, "mid": (lo + hi) / 2, "high": hi + 0.5 * span}[pos]
        return np.column_stack([X, xx])
    rows = []
    for rp in ("low", "mid", "high"):
        for cp in ("low", "mid", "high"):
            dO = tr_all(place(XO_cal2, cp)) - tr_all(place(XO_raw2, rp))
            J = (dB[:, None, None] + dL[None, :, None] + dO[None, None, :]) / 3
            Jm = np.maximum(np.maximum(dB[:, None, None], dL[None, :, None]), dO[None, None, :])
            rows.append({"raw_pos": rp, "cal_pos": cp, "dTr_O": float(dO[0]), "O_rank_of_6": int(np.sum(dO <= dO[0] + 1e-12)),
                         "J_mean": float(J[0, 0, 0]), "p_mean": float(np.sum(J <= J[0, 0, 0] + 1e-12) / 4320),
                         "p_min": float(np.sum(Jm <= Jm[0, 0, 0] + 1e-12) / 4320)})
    rng = np.random.default_rng(0)
    ps, pms, d0 = [], [], []
    for _ in range(3000):
        rp = rng.integers(0, 3, 17)
        cp = rng.integers(0, 3, 17)
        lo, hi = XO_raw2.min(1), XO_raw2.max(1)
        sp = np.maximum(hi - lo, 1e-6)
        xr = np.choose(rp, [lo - .5 * sp, (lo + hi) / 2, hi + .5 * sp])
        lo2, hi2 = XO_cal2.min(1), XO_cal2.max(1)
        sp2 = np.maximum(hi2 - lo2, 1e-6)
        xc = np.choose(cp, [lo2 - .5 * sp2, (lo2 + hi2) / 2, hi2 + .5 * sp2])
        dO = tr_all(np.column_stack([XO_cal2, xc])) - tr_all(np.column_stack([XO_raw2, xr]))
        J = (dB[:, None, None] + dL[None, :, None] + dO[None, None, :]) / 3
        ps.append(np.sum(J <= J[0, 0, 0] + 1e-12) / 4320)
        Jm = np.maximum(np.maximum(dB[:, None, None], dL[None, :, None]), dO[None, None, :])
        pms.append(np.sum(Jm <= Jm[0, 0, 0] + 1e-12) / 4320)
        d0.append(dO[0])
    ps, pms, d0 = np.array(ps), np.array(pms), np.array(d0)
    above = d0[ps > 0.05]
    return {"placements": rows,
            "random_3000": {"p_mean_min": float(ps.min()), "p_mean_max": float(ps.max()),
                            "frac_p_mean_le_0.05": float(np.mean(ps <= .05)), "p_min_min": float(pms.min()),
                            "p_min_max": float(pms.max()), "frac_p_min_le_0.05": float(np.mean(pms <= .05)),
                            "min_dTr_O_with_p_mean_gt_0.05": float(above.min()) if above.size else None,
                            "max_dTr_O_with_p_mean_le_0.05": float(d0[ps <= 0.05].max()) if np.any(ps <= .05) else None,
                            "n_p_mean_gt_0.05": int(above.size)}}


# ------------------------------------------------------------------------------------------------
# scenario model (design/scen.py), conditional size (critic/condsize.py), decidability (critic/forecast.py),
# materiality threshold (critic/material.py)
# ------------------------------------------------------------------------------------------------

def base_vectors(D, pre="prh_clip+l2"):
    prl, prv = D["pr"][pre]["lm"], D["pr"][pre]["vit"]
    PRv = np.array([prv[v].max() for v in VITS])
    g = {k: vec(D, pre, lm, "cka_biased", f) for k, lm, f in
         (("T3", O3, "s_obs"), ("T7", O7, "s_obs"), ("c3", O3, "c_alpha"), ("c7", O7, "c_alpha"),
          ("s3", O3, "s_cal"), ("s7", O7, "s_cal"), ("u3", O3, "uncensored"), ("u7", O7, "uncensored"))}
    g["PRv"] = PRv
    g["P3"], g["P7"] = float(prl[O3].max()), float(prl[O7].max())
    g["Bb3"] = np.sqrt(g["P3"] * PRv) / 1023
    g["Bb7"] = np.sqrt(g["P7"] * PRv) / 1023
    g["off"] = float(np.median(np.r_[g["c3"] - g["Bb3"], g["c7"] - g["Bb7"]]))
    return g


def trall3(X):
    res = []
    for perm in itertools.permutations(range(3)):
        lab = np.array(perm, float) + 1
        res.append(np.mean([np.corrcoef(rankdata(r), lab)[0, 1] for r in X]))
    return np.array(res)


def scenario_table(D):
    g = base_vectors(D)
    out = {"offset_used": g["off"], "rows": []}
    for sc in ("s_cal", "uncensored"):
        dB = trallN(Mx(D, "prh_clip+l2", BF, sc)) - trallN(Mx(D, "prh_clip+l2", BF, "s_obs"))
        dL = trallN(Mx(D, "prh_clip+l2", LF, sc)) - trallN(Mx(D, "prh_clip+l2", LF, "s_obs"))
        for PR13 in (215, 240, 260, 290, 320):
            B13 = np.sqrt(PR13 * g["PRv"]) / 1023
            c13 = B13 + g["off"]
            for k in (0.0, 0.5, 1.0, 2.0, 3.0):
                T13 = g["T7"] + k * (g["T7"] - g["T3"])
                if sc == "s_cal":
                    x13 = np.maximum((T13 - c13) / (1 - c13), 0)
                    X = np.column_stack([g["s3"], g["s7"], x13])
                else:
                    x13 = T13 - (B13 + 6e-5)
                    X = np.column_stack([g["u3"], g["u7"], x13])
                R = np.column_stack([g["T3"], g["T7"], T13])
                dO = trall3(X) - trall3(R)
                J = (dB[:, None, None] + dL[None, :, None] + dO[None, None, :]) / 3
                Jm = np.maximum(np.maximum(dB[:, None, None], dL[None, :, None]), dO[None, None, :])
                out["rows"].append({"score": sc, "PR13": PR13, "k": k, "Bbar13_mean": float(B13.mean()),
                                    "median_gap_7_13": float(np.median(B13 - g["Bb7"])),
                                    "Tr_raw": float(trall3(R)[0]), "Tr_score": float(trall3(X)[0]), "dTr_O": float(dO[0]),
                                    "p_Jmean": float((J <= J[0, 0, 0] + 1e-12).mean()),
                                    "p_Jmin": float((Jm <= Jm[0, 0, 0] + 1e-12).mean())})
    return out


def condsize_table(D):
    g = base_vectors(D)
    dB = trallN(Mx(D, "prh_clip+l2", BF, "s_cal")) - trallN(Mx(D, "prh_clip+l2", BF, "s_obs"))
    dL = trallN(Mx(D, "prh_clip+l2", LF, "s_cal")) - trallN(Mx(D, "prh_clip+l2", LF, "s_obs"))
    perms3 = list(itertools.permutations(range(3)))

    def flips(R, X, perm, rule):
        lab = np.array(perm) + 1
        n = 0
        new = int(np.argmax(lab))
        for v in range(R.shape[0]):
            if rule == "critic":           # critic/condsize.py newflips: comparisons with index 2 (true 13B), c->d only
                for a in (0, 1):
                    b = 2
                    lo, hi = (a, b) if lab[a] < lab[b] else (b, a)
                    n += int(R[v, hi] > R[v, lo] and X[v, hi] < X[v, lo])
            else:                           # VERDICTS_SPEC J cond_size: member labelled 13b is new; f- = #{d_X < d_raw}
                for a in range(3):
                    if a == new:
                        continue
                    dr = np.sign(R[v, new] - R[v, a])
                    dx = np.sign(X[v, new] - X[v, a])
                    n += int(dx < dr)
        return n

    rows = []
    for PR13 in (240, 260, 290):
        B13 = np.sqrt(PR13 * g["PRv"]) / 1023
        c13 = B13 + g["off"]
        for k in (0.5, 1.0, 2.0):
            T13 = g["T7"] + k * (g["T7"] - g["T3"])
            x13 = np.maximum((T13 - c13) / (1 - c13), 0)
            X = np.column_stack([g["s3"], g["s7"], x13])
            R = np.column_stack([g["T3"], g["T7"], T13])
            dO = trallN(X) - trallN(R)
            Jm = np.maximum(np.maximum(dB[:, None, None], dL[None, :, None]), dO[None, None, :])
            res = {}
            for rule in ("critic", "spec"):
                rej = []
                for j, perm in enumerate(perms3):
                    p = (Jm <= Jm[0, 0, j] + 1e-9).mean()
                    rej.append(bool(p <= 0.05 and dO[j] <= -0.10 + (1e-9 if rule == "critic" else 0.0)
                                    and flips(R, X, perm, rule) >= 3))
                res[rule] = {"reject_id": rej[0], "n_sigmaO_reject": int(sum(rej)), "cond_size": sum(rej) / 6}
            rows.append({"PR13": PR13, "k": k, "dTr_O_id": float(dO[0]), "dTr_O_all": [float(v) for v in dO],
                         "p_Jmin_id": float((Jm <= Jm[0, 0, 0] + 1e-9).mean()),
                         "material_PR13": None, "critic_rule": res["critic"], "spec_rule": res["spec"]})
    return {"offset_used": g["off"], "rows": rows}


def material_threshold(D):
    g = base_vectors(D)
    sq = np.sqrt(g["PRv"])
    rows = []
    for PR13 in np.arange(236, 252, 1.0):
        g7 = float(np.median((math.sqrt(PR13) - math.sqrt(g["P7"])) * sq / 1023))
        g3 = float(np.median((math.sqrt(PR13) - math.sqrt(g["P3"])) * sq / 1023))
        rows.append({"PR13": float(PR13), "median_gap_7_13": g7, "median_gap_3_13": g3,
                     "material": bool(g7 >= 0.005 and g3 >= 0.005)})
    exact = (math.sqrt(g["P7"]) + 0.005 * 1023 / float(np.median(sq))) ** 2
    first = next((r["PR13"] for r in rows if r["material"]), None)
    return {"rows": rows, "PR13_star_exact": exact, "smallest_integer_PR13_material": first,
            "PR_star_3b": g["P3"], "PR_star_7b": g["P7"],
            "extrapolation": {"openllama_ratio": g["P7"] * g["P7"] / g["P3"]}}


def decidability(D):
    g = base_vectors(D)
    T = {3: g["T3"], 7: g["T7"]}
    c = {3: g["c3"], 7: g["c7"]}
    rows = []
    for PR13 in (245, 260, 290):
        B13 = np.sqrt(PR13 * g["PRv"]) / 1023
        for k in (0.5, 1.0, 2.0):
            T13 = T[7] + k * (T[7] - T[3])
            und = flipF = 0
            for j in (3, 7):
                def kap(off, j=j):
                    return T[j] + (1 - T[j]) * (B13 + off - c[j]) / (1 - c[j])
                klo, khi = kap(0.0009), kap(0.0029)
                und += int(((T13 >= klo) & (T13 <= khi)).sum())
                flipF += int((T13 < klo).sum())
            rows.append({"PR13": PR13, "k": k, "undecided": und, "forecast_flips": flipF,
                         "forecast_keeps": 34 - und - flipF})
    return rows


# ------------------------------------------------------------------------------------------------
# ARH pooled reading with 13B added (critic/arh11.py) + E-key CI reproduction
# ------------------------------------------------------------------------------------------------

def arh_table(D):
    pre = "prh_clip+l2"
    lms = list(RANK)

    def col(met, f):
        return {lm: vec(D, pre, lm, met, f) for lm in lms}

    def r(Dd, extra=None):
        xs, ys = [], []
        for lm, a in Dd.items():
            xs += [RANK[lm]] * len(a)
            ys += list(a)
        if extra is not None:
            xs += [8] * len(extra)
            ys += list(extra)
        return float(np.corrcoef(xs, ys)[0, 1])

    def reading(rb_c, ra_c, rb_m, ra_m):
        dd = (rb_c - ra_c) - (rb_m - ra_m)
        if dd >= 0.20 and ra_c <= ra_m - 0.20:
            return dd, "reappears"
        if dd <= 0.10 or ra_c >= ra_m - 0.10:
            return dd, "absent"
        return dd, "partial"
    C_T, C_S, M_T, M_S = col("cka_biased", "s_obs"), col("cka_biased", "s_cal"), col("mknn", "s_obs"), col("mknn", "s_cal")
    v10 = (r(C_T), r(C_S), r(M_T), r(M_S))
    dd10, rd10 = reading(*v10)
    PRv = np.array([D["pr"][pre]["vit"][x].max() for x in VITS])
    scen = []
    extras = {}
    for PR13 in (245, 260, 290):
        c13 = np.sqrt(PR13 * PRv) / 1023 + 0.0019
        for k in (0.0, 0.5, 1.0, 2.0):
            T13 = C_T[O7] + k * (C_T[O7] - C_T[O3])
            S13 = np.maximum((T13 - c13) / (1 - c13), 0)
            mT = M_T[O7] + k * (M_T[O7] - M_T[O3])
            mS = (mT - 0.0139) / (1 - 0.0139)
            v = (r(C_T, T13), r(C_S, S13), r(M_T, mT), r(M_S, mS))
            dd, rd = reading(*v)
            extras[(PR13, k)] = (T13, S13, mT, mS)
            scen.append({"PR13": PR13, "k": k, "r_cka_before": v[0], "r_cka_after": v[1], "r_mknn_before": v[2],
                         "r_mknn_after": v[3], "dDelta": dd, "gap_after": v[3] - v[1], "category": rd})
    rng = np.random.default_rng(1)

    def boot(extra=None, B=2000):
        out = []
        for _ in range(B):
            ii = rng.integers(0, 17, 17)

            def sub(Dd):
                return {lm: a[ii] for lm, a in Dd.items()}
            if extra is None:
                v = (r(sub(C_T)), r(sub(C_S)), r(sub(M_T)), r(sub(M_S)))
            else:
                v = (r(sub(C_T), extra[0][ii]), r(sub(C_S), extra[1][ii]), r(sub(M_T), extra[2][ii]),
                     r(sub(M_S), extra[3][ii]))
            out.append((v[0] - v[1]) - (v[2] - v[3]))
        return [float(x) for x in np.percentile(out, [2.5, 97.5])]
    ci = {"10LM": boot()}
    for PR13, k in ((245, 1.0), (260, 1.0), (290, 1.0), (260, 0.0)):
        ci[f"PR13={PR13},k={k}"] = boot(extras[(PR13, k)])
    # E-key reproduction (aggregate_llama.py::delta_boot on grid10: 7 frozen rows + LLaMA-axis E rows, VITS order)
    pooled = BF + [O3, O7] + LF
    xr = np.array([RANK[s] for s in pooled], dtype=float)

    def grid(met, f):
        G = np.empty((10, 17))
        for i, lm in enumerate(pooled):
            for j, v in enumerate(VITS_REG):
                if lm in LF:
                    js = json.load(open(os.path.join(LLAMA_E, pre, f"{lm}__{v}.json")))
                    G[i, j] = js["summaries"][met]["full"][f]
                else:
                    G[i, j] = D["pairs"][(pre, lm, v)]["m"][met]["full"][f]
        return G

    def pear(G, x):
        xx = np.repeat(x[:, None], G.shape[1], axis=1).ravel()
        return float(np.corrcoef(xx, G.ravel())[0, 1])
    G1b, G1a, G2b, G2a = grid("cka_biased", "s_obs"), grid("cka_biased", "s_cal"), grid("mknn", "s_obs"), grid("mknn", "s_cal")

    def f4(a, b, c, d):
        return (pear(a, xr) - pear(b, xr)) - (pear(c, xr) - pear(d, xr))
    rk = util.rng_from_key(f"S1.2-layermax-llama|boot|arh_delta|{pre}|cka_biased")
    bs = np.empty(2000)
    for b in range(2000):
        j = rk.integers(0, 17, 17)
        bs[b] = f4(G1b[:, j], G1a[:, j], G2b[:, j], G2a[:, j])
    eci = [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]
    # the frozen OpenLLaMA-axis key applied to the four k = 1 / k = 0 scenarios (descriptive; 11 LMs, 13B at rank 8)
    okey = {}
    for PR13, k in ((245, 1.0), (260, 1.0), (290, 1.0), (260, 0.0)):
        T13, S13, mT, mS = extras[(PR13, k)]
        order = [VITS.index(v) for v in VITS_REG]
        x11 = np.array([RANK[s] for s in pooled[:7]] + [8] + [RANK[s] for s in LF], dtype=float)

        def ins(G, row):
            return np.vstack([G[:7], row[order][None, :], G[7:]])
        H = (ins(G1b, T13), ins(G1a, S13), ins(G2b, mT), ins(G2a, mS))
        rk2 = util.rng_from_key(f"S1.2-layermax-openllama|boot|arh_delta|{pre}|cka_biased")
        bs2 = np.empty(2000)
        for b in range(2000):
            j = rk2.integers(0, 17, 17)
            bs2[b] = (pear(H[0][:, j], x11) - pear(H[1][:, j], x11)) - (pear(H[2][:, j], x11) - pear(H[3][:, j], x11))
        okey[f"PR13={PR13},k={k}"] = {"dDelta": (pear(H[0], x11) - pear(H[1], x11)) - (pear(H[2], x11) - pear(H[3], x11)),
                                      "ci": [float(x) for x in np.nanpercentile(bs2, [2.5, 97.5])]}
    return {"10LM": {"r_cka_before": v10[0], "r_cka_after": v10[1], "r_mknn_before": v10[2], "r_mknn_after": v10[3],
                     "dDelta": dd10, "category": rd10, "ci_critic_rng": ci["10LM"],
                     "ci_E_llama_key": eci, "E_llama_key_dDelta": f4(G1b, G1a, G2b, G2a)},
            "scenarios": scen, "ci_critic_rng": ci, "ci_openllama_key_scenarios": okey,
            "dDelta_range_12": [min(s["dDelta"] for s in scen), max(s["dDelta"] for s in scen)],
            "categories_12": sorted(set(s["category"] for s in scen))}


# ------------------------------------------------------------------------------------------------
# resources, plan comparison, main
# ------------------------------------------------------------------------------------------------

def resources(D):
    LB = sum(D["pairs"][("prh_clip+l2", O3, v)]["L_B"] for v in VITS)
    du = shutil.disk_usage(PROJ)
    mem = {}
    try:
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    led = os.path.join(RES, "S1_2_layermax_openllama", "logs", "gpu_ledger.tsv")
    tot = 0.0
    if os.path.isfile(led):
        for i, ln in enumerate(open(led)):
            if i and ln.strip():
                tot += float(ln.split("\t")[4])
    return {"sum_L_B_17_vits": LB, "layer_pairs": {"3b": 27 * LB, "7b": 33 * LB, "13b": 41 * LB},
            "features_13b_bytes": 1024 * 41 * 5120 * 4, "weights_13b_bytes_plan": 26032458406,
            "disk_free_bytes": du.free, "mem_total_bytes": mem.get("MemTotal"), "mem_available_bytes": mem.get("MemAvailable"),
            "gpu_ledger_minutes": tot, "gpu_budget_minutes": 60}


PLAN12 = {  # plan section 0 / 12 design numbers (for the comparison list only; they move no rule)
    "s0.calpha_minus_Bbar_median": 0.00192, "s0.calpha_minus_Bbar_min": 0.00162, "s0.calpha_minus_Bbar_max": 0.00213,
    "s0.Bbar_3b": 0.1636, "s0.Bbar_7b": 0.1741, "s0.T_3b": 0.4133, "s0.T_7b": 0.4258, "s0.calpha_3b": 0.1655,
    "s0.calpha_7b": 0.1760, "s0.s_cal_3b": 0.2975, "s0.s_cal_7b": 0.3038, "s0.s_cal_n7gt3": 16, "s0.unc_n7gt3": 14,
    "s0.PR_star_3b": 201.89, "s0.PR_star_7b": 228.57, "s0.gap_Bbar_median": 0.0113, "s0.gap_Bbar_min": 0.0039,
    "s0.gap_Bbar_n_ge": 16, "s0.gap_fd05_median": 0.0034, "s0.gap_fd10_median": -0.0009,
    "s12.Jmin_scal_count": 12, "s12.Jmin_unc_count": 2, "s12.Jmean_scal_count": 6,
    "s12.nB_rows": [6, 18, 24, 37, 48, 50, 60], "s12.nL_rows": [2, 2, 3, 3, 3, 3, 3],
    "s12.material_PR13": 241, "s12.gap_at_240": 0.0046, "s12.gap_at_241": 0.0050,
    "s12.arh10_dDelta": 0.116, "s12.arh10_ci_critic": [0.010, 0.255], "s12.arh10_ci_Ekey": [0.0029, 0.2613],
    "s12.jmean_rise_threshold": 0.47,
    "s12.decidability": {"245,0.5": [4, 2, 28], "245,1.0": [1, 1, 32], "245,2.0": [0, 0, 34],
                         "260,0.5": [5, 15, 14], "260,1.0": [2, 2, 30], "260,2.0": [0, 1, 33],
                         "290,0.5": [1, 30, 3], "290,1.0": [4, 16, 14], "290,2.0": [1, 2, 31]},
    "s12.condsize": {"260,0.5": [-0.618, 0.0167, 2], "260,1.0": [-0.147, 0.0333, 1], "290,0.5": [-1.353, 0.0056, 2],
                     "290,1.0": [-0.618, 0.0167, 2], "260,2.0": [-0.088, 0.071, 0], "290,2.0": [-0.118, 0.034, 0]},
    "s12.scen_scal": {"240,0.5": -0.12, "240,1.0": -0.09, "240,2.0": -0.03, "260,0.5": -0.62, "260,1.0": -0.15,
                      "260,2.0": -0.09, "290,0.5": -1.35, "290,1.0": -0.62, "290,2.0": -0.12},
    "s12.scen_unc": {"260,0.5": -1.47, "260,1.0": -0.35, "260,2.0": -0.18, "290,1.0": -1.59, "290,2.0": -0.35},
    "s12.arh_scen": {"245,1.0": 0.099, "260,1.0": 0.102, "290,1.0": 0.107, "260,0.0": 0.106},
    "s12.arh_scen_ci": {"245,1.0": [-0.013, 0.238], "260,1.0": [-0.008, 0.241], "290,1.0": [-0.005, 0.242],
                        "260,0.0": [-0.018, 0.252]},
}


def compare(res) -> list:
    rows = []

    def add(name, plan, server, nd):
        ok = (round(server, nd) == round(plan, nd)) if isinstance(plan, float) else (server == plan)
        rows.append({"item": name, "plan": plan, "server": server, "equal_at_plan_precision": bool(ok)})
    k0 = res["known_section0"]["prh_clip+l2"]
    m = k0["means_over_17_vits"]
    add("s0 c_alpha - Bbar median (34 pilot pairs)", PLAN12["s0.calpha_minus_Bbar_median"], k0["calpha_minus_Bbar_34"]["median"], 5)
    add("s0 c_alpha - Bbar min", PLAN12["s0.calpha_minus_Bbar_min"], k0["calpha_minus_Bbar_34"]["min"], 5)
    add("s0 c_alpha - Bbar max", PLAN12["s0.calpha_minus_Bbar_max"], k0["calpha_minus_Bbar_34"]["max"], 5)
    for lab, key in (("raw T", "T"), ("c_alpha", "calpha"), ("Bbar", "Bbar"), ("s_cal", "s_cal")):
        for s in ("3b", "7b"):
            add(f"s0 mean {lab} {s}", PLAN12[f"s0.{key}_{s}"], m[lab][s], 4)
    add("s0 s_cal 7B>3B", PLAN12["s0.s_cal_n7gt3"], m["s_cal"]["n_7b_gt_3b"], 0)
    add("s0 uncensored 7B>3B", PLAN12["s0.unc_n7gt3"], m["uncensored"]["n_7b_gt_3b"], 0)
    add("s0 PR* 3B", PLAN12["s0.PR_star_3b"], k0["pr_profiles"][O3]["PR_star"], 2)
    add("s0 PR* 7B", PLAN12["s0.PR_star_7b"], k0["pr_profiles"][O7]["PR_star"], 2)
    add("s0 Bbar gap 3B-7B median", PLAN12["s0.gap_Bbar_median"], k0["gaps"]["Bbar_7b_minus_3b"]["median"], 4)
    add("s0 Bbar gap 3B-7B min", PLAN12["s0.gap_Bbar_min"], k0["gaps"]["Bbar_7b_minus_3b"]["min"], 4)
    add("s0 Bbar gap >= 0.005 (ViTs)", PLAN12["s0.gap_Bbar_n_ge"], k0["gaps"]["Bbar_7b_minus_3b"]["n_ge_0.005"], 0)
    add("s0 B_fd(0.5) gap median", PLAN12["s0.gap_fd05_median"], k0["gaps"]["Bfd_0.5_7b_minus_3b"]["median"], 4)
    add("s0 B_fd(1.0) gap median", PLAN12["s0.gap_fd10_median"], k0["gaps"]["Bfd_1.0_7b_minus_3b"]["median"], 4)
    jt = res["J_tables"]
    add("s12 J_min s_cal minimal count (n_O = 1)", PLAN12["s12.Jmin_scal_count"], jt["s_cal"]["Jmin_min_count_nO1"], 0)
    add("s12 J_min unc minimal count (n_O = 1)", PLAN12["s12.Jmin_unc_count"], jt["unc"]["Jmin_min_count_nO1"], 0)
    add("s12 J_mean s_cal minimal count", PLAN12["s12.Jmean_scal_count"], jt["s_cal"]["Jmean_min_count"], 0)
    add("s12 n_B(t) rows (s_cal)", PLAN12["s12.nB_rows"], [r_["n_B"] for r_ in jt["s_cal"]["rows"]], 0)
    add("s12 n_L(t) rows (s_cal)", PLAN12["s12.nL_rows"], [r_["n_L"] for r_ in jt["s_cal"]["rows"]], 0)
    mt = res["materiality_threshold"]
    add("s12 smallest integer PR*(13B) material", PLAN12["s12.material_PR13"], int(mt["smallest_integer_PR13_material"]), 0)
    for p_, key in ((240.0, "s12.gap_at_240"), (241.0, "s12.gap_at_241")):
        add(f"s12 median Bbar gap 7B-13B at PR* {int(p_)}", PLAN12[key],
            next(r_["median_gap_7_13"] for r_ in mt["rows"] if r_["PR13"] == p_), 4)
    for r_ in res["decidability"]:
        key = f"{r_['PR13']},{r_['k']}"
        add(f"s12 P-O2b decidability {key} (undecided/flips/keeps)", PLAN12["s12.decidability"][key],
            [r_["undecided"], r_["forecast_flips"], r_["forecast_keeps"]], 0)
    for r_ in res["condsize"]["rows"]:
        key = f"{r_['PR13']},{r_['k']}"
        if key in PLAN12["s12.condsize"]:
            pv = PLAN12["s12.condsize"][key]
            add(f"s12 condsize {key} dTr_O(id)", pv[0], r_["dTr_O_id"], 3)
            add(f"s12 condsize {key} J_min p(id)", pv[1], r_["p_Jmin_id"], 2 if pv[1] >= 0.03 else 4)
            add(f"s12 condsize {key} #sigma_O rejecting (critic rule)", pv[2], r_["critic_rule"]["n_sigmaO_reject"], 0)
    for r_ in res["scenario"]["rows"]:
        key = f"{r_['PR13']},{r_['k']}"
        tab = PLAN12["s12.scen_scal"] if r_["score"] == "s_cal" else PLAN12["s12.scen_unc"]
        if key in tab:
            add(f"s12 scenario {r_['score']} dTr_O {key}", tab[key], r_["dTr_O"], 2)
    a = res["arh"]
    add("s12 ARH 10-LM dDelta", PLAN12["s12.arh10_dDelta"], a["10LM"]["dDelta"], 3)
    add("s12 ARH 10-LM CI (critic rng) lo", PLAN12["s12.arh10_ci_critic"][0], a["10LM"]["ci_critic_rng"][0], 3)
    add("s12 ARH 10-LM CI (critic rng) hi", PLAN12["s12.arh10_ci_critic"][1], a["10LM"]["ci_critic_rng"][1], 3)
    add("s12 ARH 10-LM CI (E LLaMA key) lo", PLAN12["s12.arh10_ci_Ekey"][0], a["10LM"]["ci_E_llama_key"][0], 4)
    add("s12 ARH 10-LM CI (E LLaMA key) hi", PLAN12["s12.arh10_ci_Ekey"][1], a["10LM"]["ci_E_llama_key"][1], 4)
    for s in a["scenarios"]:
        key = f"{s['PR13']},{s['k']}"
        if key in PLAN12["s12.arh_scen"]:
            add(f"s12 ARH scenario {key} dDelta", PLAN12["s12.arh_scen"][key], s["dDelta"], 3)
    for key, ci in PLAN12["s12.arh_scen_ci"].items():
        PR13, k = key.split(",")
        sv = a["ci_critic_rng"][f"PR13={PR13},k={float(k)}"]
        add(f"s12 ARH scenario {key} CI lo (critic rng)", ci[0], sv[0], 3)
        add(f"s12 ARH scenario {key} CI hi (critic rng)", ci[1], sv[1], 3)
    pl = [x["dTr_O"] for x in res["Jmean_scenarios"]["placements"] if x["p_mean"] > 0.05]
    if pl:
        add("s12 J_mean: smallest dTr_O with p_mean > 0.05 among the 9 uniform placements ('about +0.47')",
            PLAN12["s12.jmean_rise_threshold"], min(pl), 2)
    return rows


def fmt_md(res, cmp_rows) -> list[str]:
    L = ["# OpenLLaMA axis: feasibility tables recomputed on ds2213 (S0b; EXPLORATORY design calculations)", "",
         f"Written by `analysis/s12_layermax_openllama/feasibility_ol.py` at {res['meta']['finished_utc']} from known "
         f"frozen files only ({res['meta']['n_inputs']} files; sha256 in `feasibility_ol.json` → `meta.inputs`). "
         "Rules, bands and thresholds are numeric constants of the plan and do not move (R.3.3). Where a server number "
         "differs from the plan's design number, the server number replaces it in reports and the difference is logged.", "",
         "## Plan vs server", "", "| item | plan | server | equal at plan precision |", "|---|---|---|---|"]
    for r in cmp_rows:
        sv = r["server"]
        if isinstance(sv, float):
            sv = f"{sv:.6g}"
        elif isinstance(sv, list):
            sv = str([round(x, 4) if isinstance(x, float) else x for x in sv])
        else:
            sv = str(sv)
        L.append(f"| {r['item']} | {r['plan']} | {sv} | {'yes' if r['equal_at_plan_precision'] else '**NO**'} |")
    n_bad = sum(not r["equal_at_plan_precision"] for r in cmp_rows)
    L += ["", f"{len(cmp_rows)} items compared, {n_bad} differ at the plan's printed precision.", ""]
    L += ["## J_min given the known families (s_cal / unc)", ""]
    for x in ("s_cal", "unc"):
        j = res["J_tables"][x]
        L.append(f"- {x}: ΔTr_B {j['dTr_B']:+.3f} (rank {j['rank_B']}/120), ΔTr_L {j['dTr_L']:+.3f} (rank {j['rank_L']}/6), "
                 f"BL mean rank {j['BL_mean_rank_720']}/720; minimal J_min count {j['Jmin_min_count_nO1']}/4320, "
                 f"minimal J_mean count {j['Jmean_min_count']}/4320.")
        for r in j["rows"]:
            L.append(f"  - s_O·ΔTr_O {r['sO_dTrO']:+.3f}: t {r['t']:+.3f}, n_B {r['n_B']}, n_L {r['n_L']}, p (n_O = 1/2/3) "
                     + " / ".join(f"{p:.4f}" for p in r["p_nO_1_2_3"]))
    jm = res["Jmean_scenarios"]["random_3000"]
    L += ["", f"- J_mean, 3000 random placements: p_mean in [{jm['p_mean_min']:.4f}, {jm['p_mean_max']:.4f}], "
              f"fraction ≤ 0.05 {jm['frac_p_mean_le_0.05']:.3f}; smallest ΔTr_O with p_mean > 0.05: "
              f"{jm['min_dTr_O_with_p_mean_gt_0.05']}; largest ΔTr_O with p_mean ≤ 0.05: {jm['max_dTr_O_with_p_mean_le_0.05']} "
              "(the 9 uniform placements give the plan's 'about +0.47'; random placements overlap around it).", ""]
    L += ["## Materiality threshold", "",
          f"- Smallest integer PR*(13B) with B̄ material: {res['materiality_threshold']['smallest_integer_PR13_material']}; "
          f"exact boundary {res['materiality_threshold']['PR13_star_exact']:.3f}.", ""]
    L += ["## Conditional size (critic rule / VERDICTS_SPEC f- rule)", ""]
    for r in res["condsize"]["rows"]:
        L.append(f"- PR* {r['PR13']}, k {r['k']}: ΔTr_O(id) {r['dTr_O_id']:+.3f}, J_min p {r['p_Jmin_id']:.4f}; rejects "
                 f"{r['critic_rule']['reject_id']} / {r['spec_rule']['reject_id']}; conditional size "
                 f"{r['critic_rule']['n_sigmaO_reject']}/6 / {r['spec_rule']['n_sigmaO_reject']}/6")
    L += ["", "## Scenario model (s_cal and uncensored ΔTr_O)", ""]
    for r in res["scenario"]["rows"]:
        if r["k"] in (0.5, 1.0, 2.0) and r["PR13"] in (215, 240, 260, 290):
            L.append(f"- {r['score']} PR* {r['PR13']} k {r['k']}: median gap 7→13 {r['median_gap_7_13']:+.4f}, "
                     f"ΔTr_O {r['dTr_O']:+.3f}, p J_mean {r['p_Jmean']:.4f}, p J_min {r['p_Jmin']:.4f}")
    L += ["", "## P-O2b decidability (undecided / flips / keeps of 34)", ""]
    for r in res["decidability"]:
        L.append(f"- PR* {r['PR13']}, k {r['k']}: {r['undecided']} / {r['forecast_flips']} / {r['forecast_keeps']}")
    a = res["arh"]
    L += ["", "## ARH pooled reading", "",
          f"- 10 LMs: ΔΔ {a['10LM']['dDelta']:.4f} ({a['10LM']['category']}); CI critic rng {[round(x, 4) for x in a['10LM']['ci_critic_rng']]}, "
          f"E LLaMA key {[round(x, 4) for x in a['10LM']['ci_E_llama_key']]}.",
          f"- 12 scenarios: ΔΔ in [{a['dDelta_range_12'][0]:.3f}, {a['dDelta_range_12'][1]:.3f}], categories {a['categories_12']}."]
    for k, v in a["ci_critic_rng"].items():
        L.append(f"  - CI (critic rng) {k}: {[round(x, 3) for x in v]}")
    for k, v in a["ci_openllama_key_scenarios"].items():
        L.append(f"  - CI (frozen OpenLLaMA key, descriptive) {k}: ΔΔ {v['dDelta']:.4f}, {[round(x, 4) for x in v['ci']]}")
    r = res["resources"]
    L += ["", "## Resources (computable)", "",
          f"- Layer pairs: 3B {r['layer_pairs']['3b']}, 7B {r['layer_pairs']['7b']}, 13B {r['layer_pairs']['13b']}; 13B features "
          f"{r['features_13b_bytes'] / 1e9:.2f} GB; disk free {r['disk_free_bytes'] / 1e9:.1f} GB; RAM available "
          f"{(r['mem_available_bytes'] or 0) / 1e9:.1f} GB; GPU ledger {r['gpu_ledger_minutes']:.2f} / 60 min."]
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    test = guard.test_mode()
    out_dir = os.path.join(guard.SCRATCH, "PB", "feasibility") if test else OUT_REAL
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if sha256_file(LLAMA_PB) != LLAMA_PB_SHA:
        raise SystemExit("frozen LLaMA partB_features.json hash changed")
    inputs: dict = {}
    D = load(inputs)
    print(f"[{time.time() - t0:5.1f}s] loaded {len(D['pairs'])} known pair records", flush=True)
    res = {"label": "EXPLORATORY design calculations from known files only (plan section 12); not citable results",
           "known_section0": known_table(D)}
    res["J_tables"] = jtables(D)
    print(f"[{time.time() - t0:5.1f}s] J tables", flush=True)
    res["Jmean_scenarios"] = jmean_scenarios(D)
    print(f"[{time.time() - t0:5.1f}s] J_mean scenarios", flush=True)
    res["scenario"] = scenario_table(D)
    res["condsize"] = condsize_table(D)
    res["materiality_threshold"] = material_threshold(D)
    res["decidability"] = decidability(D)
    print(f"[{time.time() - t0:5.1f}s] scenario, condsize, materiality, decidability", flush=True)
    res["arh"] = arh_table(D)
    print(f"[{time.time() - t0:5.1f}s] ARH", flush=True)
    res["resources"] = resources(D)
    cmp_rows = compare(res)
    res["plan_vs_server"] = cmp_rows
    res["meta"] = {"script_sha256": sha256_file(os.path.abspath(__file__)), "started_utc": started,
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "threads": _TH,
                   "test_mode": test, "n_inputs": len(inputs),
                   "inputs": {p: sha256_file(os.path.join(PROJ, p)) for p in sorted(inputs)}}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "feasibility_ol.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(os.path.join(out_dir, "FEASIBILITY_OL.md"), "w") as f:
        f.write("\n".join(fmt_md(res, cmp_rows)) + "\n")
    n_bad = sum(not r["equal_at_plan_precision"] for r in cmp_rows)
    for r in cmp_rows:
        if not r["equal_at_plan_precision"]:
            print(f"DIFF {r['item']}: plan {r['plan']} server {r['server']}", flush=True)
    print(f"[{time.time() - t0:5.1f}s] wrote {out_dir}/feasibility_ol.json, FEASIBILITY_OL.md; "
          f"{len(cmp_rows)} plan items compared, {n_bad} differ at plan precision", flush=True)


if __name__ == "__main__":
    main()
