#!/usr/bin/env python
"""Pythia axis, implementation I: plan V-P9 known-answer tests (NEW, PYT-AXIS, sub-role pyt-code-I, S0b, 2026-10-10).
Synthetic inputs only (no pair file, no feature). Runs only in S0b test mode (PYT_S0B_TEST=1, no CODE FROZEN line),
else exit 3 with no output. Writes analysis/s12_layermax_pythia/s0b_scratch/I/vp9/test_report_<UTC>.json.

  t01 the plan 12 pattern table: exact p_perm over 7! relabellings for 8 stylised calibrated rank patterns (17 ViTs with
      the same pattern, raw ranks 1..7), e.g. top-5 reversal 444/5040, top pair swapped 2520/5040; and their dTr.
  t02 Holm at 0.05 over three members (hand-computed rejections and adjusted p).
  t03 IUT / magnitude bootstrap p (constant per-ViT delta-rho: exact counts) and p_H1 = max(p_perm, p_mag).
  t04 TOST p incl. the degenerate case (all delta-rho = 0 -> 1/2001) and the out-of-range rule (p := 1).
  t05 P-P2b classifications under per-model bands: below / above / tie / undecided (incl. one side censorable) and the
      verdict rule (>= 30 decided, 0 wrong).
  t06 the plan 7 category table: every row reachable (E, D, C, F with both sub-reasons, B via P-P2a and via P-P2b, A,
      A-att, A', premise unverified).
  t07 end-to-end verdicts_pyt on synthetic 7 x 17 records (scenario model of plan 12): gate open/closed logic, Holm only
      behind a confirmed H1, "not tested, gate closed" statuses, category consistency with the components, deductive
      checks zero, determinism (two runs identical).
  t08 generic trend vs the OpenLLaMA 3-size trend (bitwise on random data) and vs scipy Spearman for n = 7.
  t09 band arithmetic (plan 9 B6 constants) and the M5 range limits.
  t10 the common-ViT-set rule vs the per-score exclusion (a ViT constant under s_cal only), and the censoring
      counterexample to the per-ViT sign rule of plan 8 ([op3] in pyt_verdicts.py).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pyt_gate  # noqa: E402
if not pyt_gate.test_mode():
    pyt_gate._fail("test_pyt_verdicts.py runs only in S0b test mode (PYT_S0B_TEST=1, before CODE FROZEN)")

import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from fractions import Fraction  # noqa: E402

import numpy as np  # noqa: E402

import pyt_verdicts as PV  # noqa: E402

OUTD = os.path.join(pyt_gate.TEST_ROOT, "vp9")
RESULTS = []


def check(name, cond, detail=None):
    RESULTS.append({"test": name, "pass": bool(cond), "detail": detail})
    if not cond:
        raise AssertionError(f"{name}: {detail}")


def cols_from_ranks(ranks, nv=17, scale=0.1):
    return [[scale * r for r in ranks] for _ in range(nv)]


def t01():
    raw = list(range(1, 8))
    pats = {"top pair swapped": ((1, 2, 3, 4, 5, 7, 6), -0.036, 2520),
            "top 3 reversed": ((1, 2, 3, 4, 7, 6, 5), -0.143, 1800),
            "top 4 reversed": ((1, 2, 3, 7, 6, 5, 4), -0.357, 966),
            "top 5 reversed": ((1, 2, 7, 6, 5, 4, 3), -0.714, 444),
            "6.9B drops to rank 4": ((1, 2, 3, 5, 6, 7, 4), -0.214, 1440),
            "6.9B drops to rank 1": ((2, 3, 4, 5, 6, 7, 1), -0.750, 720),
            "three disjoint adjacent swaps": ((2, 1, 4, 3, 6, 5, 7), -0.107, 1692),
            "full reversal": ((7, 6, 5, 4, 3, 2, 1), -2.000, 1)}
    out = {}
    for nm, (pat, dtr, k) in pats.items():
        dt = PV.dtrend(cols_from_ranks(pat), cols_from_ranks(raw), "common")
        p = float(np.sum(dt["D"] <= dt["obs"] + PV.TOL))
        out[nm] = {"dTr": dt["obs"], "k": p}
        check(f"t01 {nm} k", int(round(p)) == k and abs(p - round(p)) < 1e-9, (p, k))
        check(f"t01 {nm} dTr", abs(dt["obs"] - dtr) < 6e-4, (dt["obs"], dtr))
    return out


def t02():
    cases = [({"H2a": 0.01, "H3": 0.02, "H4": 0.04}, {"H2a": True, "H3": True, "H4": True},
              {"H2a": 0.03, "H3": 0.04, "H4": 0.04}),
             ({"H2b": 0.02, "H3": 0.0001, "H4": 0.06}, {"H2b": True, "H3": True, "H4": False},
              {"H2b": 0.04, "H3": 0.0003, "H4": 0.06}),
             ({"H2a": 0.03, "H3": 0.02, "H4": 0.018}, {"H2a": False, "H3": False, "H4": False},
              {"H2a": 0.054, "H3": 0.054, "H4": 0.054}),
             ({"H2a": 1 / 2001, "H3": 1 / 2001, "H4": 0.05}, {"H2a": True, "H3": True, "H4": True}, None)]
    orders = ["H2a,H3,H4", "H3,H2b,H4", "H4,H3,H2a", "H2a,H3,H4"]
    for i, (pv, rej, adj) in enumerate(cases):
        mem = list(pv)
        r, a, order = PV.holm(mem, pv)
        check(f"t02 case {i} rejections", r == rej, (r, rej))
        check(f"t02 case {i} order (ties by member order)", order == orders[i], (order, orders[i]))
        if adj:
            check(f"t02 case {i} adjusted", all(abs(a[k] - adj[k]) < 1e-12 for k in mem), (a, adj))


def t03():
    from hrhv2 import util  # noqa: F401
    d = [-0.2] * 17
    draws = PV.boot_means(d, "S1.2-layermax-pythia|boot|test|t03a")
    pm = (1.0 + float(np.sum(draws >= -0.10 - PV.TOL))) / 2001
    check("t03 constant -0.2 -> p_mag = 1/2001", abs(pm - 1 / 2001) < 1e-15, pm)
    d = [-0.05] * 17
    draws = PV.boot_means(d, "S1.2-layermax-pythia|boot|test|t03b")
    pm = (1.0 + float(np.sum(draws >= -0.10 - PV.TOL))) / 2001
    check("t03 constant -0.05 -> p_mag = 1", abs(pm - 1.0) < 1e-15, pm)
    rng = np.random.default_rng(3)
    d = list(rng.normal(-0.12, 0.08, 17))
    key = "S1.2-layermax-pythia|boot|test|t03c"
    draws = PV.boot_means(d, key)
    from hrhv2 import util as U
    r2 = U.rng_from_key(key)
    ref = np.array([np.asarray(d)[r2.integers(0, 17, 17)].mean() for _ in range(2000)])
    check("t03 bootstrap draws = the documented rng convention", np.array_equal(draws, ref), None)
    pm = (1.0 + float(np.sum(draws >= -0.10 - PV.TOL))) / 2001
    pmref = (1.0 + float(np.sum(ref >= -0.10))) / 2001
    check("t03 p_mag (random) = direct count", abs(pm - pmref) < 1e-15, (pm, pmref))
    check("t03 IUT is the max", max(0.01, pm) == max(pm, 0.01), None)
    return {"p_mag_random": pm}


def t04():
    def tost(d, key, margin=0.10):
        draws = PV.boot_means(d, key)
        plo = (1.0 + float(np.sum(draws <= -margin + PV.TOL))) / 2001
        phi = (1.0 + float(np.sum(draws >= margin - PV.TOL))) / 2001
        obs = float(np.mean(d))
        return (max(plo, phi) if -margin < obs < margin else 1.0), plo, phi
    p, plo, phi = tost([0.0] * 17, "S1.2-layermax-pythia|boot|test|t04a")
    check("t04 degenerate -> 1/2001", abs(p - 1 / 2001) < 1e-15 and plo == phi == 1 / 2001, (p, plo, phi))
    p, _, _ = tost([0.05] * 17, "S1.2-layermax-pythia|boot|test|t04b")
    check("t04 constant 0.05 -> 1/2001", abs(p - 1 / 2001) < 1e-15, p)
    p, _, _ = tost([0.15] * 17, "S1.2-layermax-pythia|boot|test|t04c")
    check("t04 observed outside margin -> 1", p == 1.0, p)
    p, _, _ = tost([0.10] * 17, "S1.2-layermax-pythia|boot|test|t04d")
    check("t04 observed on the margin -> 1 (not inside the open interval)", p == 1.0, p)


def t05():
    B = [0.1, 0.1]
    bj, bk = (0.001, 0.003), (0.001, 0.003)
    f1 = PV.forecast_one
    check("t05 below", f1(0.5, 0.45, B[0], B[1], bj, bk) == "below", None)
    check("t05 above", f1(0.5, 0.6, B[0], B[1], bj, bk) == "above", None)
    check("t05 tie (both censored at every offset)", f1(0.09, 0.08, B[0], B[1], bj, bk) == "tie", None)
    check("t05 undecided (overlap)", f1(0.5, 0.5005, B[0], B[1], bj, bk) == "undecided", None)
    check("t05 censorable side, other ordered -> above", f1(0.1015, 0.5, B[0], B[1], bj, bk) == "above", None)
    check("t05 censorable side, not ordered -> undecided", f1(0.1015, 0.1012, B[0], B[1], bj, bk) == "undecided", None)
    check("t05 per-model bands differ -> below",
          f1(0.30, 0.299, 0.10, 0.10, (0.000, 0.0005), (0.003, 0.004)) == "below", None)
    check("t05 unc kind below", f1(0.5, 0.49, 0.1, 0.1, (-0.0003, 0.0008), (-0.0003, 0.0008), "unc") == "below", None)
    check("t05 unc kind undecided", f1(0.5, 0.5005, 0.1, 0.1, (-0.0003, 0.0008), (-0.0003, 0.0008), "unc")
          == "undecided", None)
    # verdict rule on a constructed 7 x 17 table (all comparisons decided and right / one wrong / too few decided)
    n, nv = 7, 17
    T = [[0.2 + 0.05 * i for i in range(n)] for _ in range(nv)]
    Bb = [[0.01 for _ in range(nv)] for _ in range(n)]
    offs = [(0.001, 0.003)] * n
    S = [[PV.scal(T[v][i], 0.012) for i in range(n)] for v in range(nv)]
    cnt, _ = PV.ppb(T, S, Bb, offs, "s_cal", n, nv)
    check("t05 all decided and right -> holds", cnt["verdict"] == "holds" and cnt["n_decided"] == 357, cnt)
    S2 = [row[:] for row in S]
    S2[0][6], S2[0][5] = S2[0][5], S2[0][6]
    cnt, _ = PV.ppb(T, S2, Bb, offs, "s_cal", n, nv)
    check("t05 one observed order wrong -> fails", cnt["verdict"] == "fails" and cnt["n_wrong"] == 1, cnt)
    T3 = [[0.2 + 0.0001 * i for i in range(n)] for _ in range(nv)]
    S3 = [[PV.scal(T3[v][i], 0.012) for i in range(n)] for v in range(nv)]
    cnt, _ = PV.ppb(T3, S3, Bb, offs, "s_cal", n, nv)
    check("t05 < 30 decided -> not evaluable", cnt["verdict"] == "not evaluable" and cnt["n_decided"] < 30, cnt)


def t06():
    cat = PV.category
    cases = {
        "E": (False, -0.5, 0.001, 0.001, "holds", "holds", 0.9, 0.1),
        "D": (True, 0.2, 0.9, 0.9, "holds", "holds", 0.9, 0.9),
        "C": (True, -0.05, 0.3, 0.9, "holds", "holds", 0.9, 0.85),
        "F": (True, -0.3, 0.2, 0.3, "holds", "holds", 0.9, 0.6),
        "B": (True, -0.5, 0.001, 0.001, "fails", "holds", 0.9, 0.4),
        "B2": (True, -0.5, 0.001, 0.001, "holds", "fails", 0.9, 0.4),
        "A": (True, -0.6, 0.001, 0.001, "holds", "holds", 0.95, 0.35),
        "A-att": (True, -0.3, 0.001, 0.001, "holds", "holds", 0.95, 0.65),
        "A'": (True, -0.3, 0.001, 0.001, "holds", "holds", 0.5, 0.2),
        "A-unverified": (True, -0.6, 0.001, 0.001, "holds", "not evaluable", 0.95, 0.35),
    }
    want = {"E": "E", "D": "D", "C": "C", "F": "F", "B": "B", "B2": "B", "A": "A", "A-att": "A-att", "A'": "A-prime",
            "A-unverified": "A"}
    got = {}
    for nm, args in cases.items():
        c = cat(*args)
        got[nm] = c
        check(f"t06 row {nm}", c[0] == want[nm], c)
    check("t06 F sub-reasons both", got["F"][2] == "not consistent across the seven sizes; size of the drop not "
          "established", got["F"])
    c = cat(True, -0.3, 0.001, 0.2, "holds", "holds", 0.9, 0.6)
    check("t06 F magnitude only", c[0] == "F" and c[2] == "size of the drop not established", c)
    check("t06 premise verified", got["A"][1] == "verified", got["A"])
    check("t06 premise unverified", got["A-unverified"][1] == "unverified", got["A-unverified"])
    c = cat(True, -0.10, 0.001, 0.001, "holds", "holds", 0.9, 0.4)
    check("t06 D1 = -0.10 exactly is a drop (inclusive)", c[0] == "A", c)
    c = cat(True, 0.10, 0.9, 0.9, "holds", "holds", 0.9, 0.9)
    check("t06 D1 = +0.10 exactly is D (inclusive)", c[0] == "D", c)
    c = cat(True, -0.6, 0.001, 0.001, "holds", "holds", 0.60, 0.40)
    check("t06 R = 0.60 and C = 0.40 give A (inclusive)", c[0] == "A", c)
    return {k: list(v) for k, v in got.items()}


SUBS = PV.SUBSETS


VIT17 = ("vit_tiny_patch16_224.augreg_in21k", "vit_small_patch16_224.augreg_in21k", "vit_base_patch16_224.augreg_in21k",
         "vit_large_patch16_224.augreg_in21k", "vit_base_patch16_224.mae", "vit_large_patch16_224.mae",
         "vit_huge_patch14_224.mae", "vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m",
         "vit_large_patch14_dinov2.lvd142m", "vit_giant_patch14_dinov2.lvd142m", "vit_base_patch16_clip_224.laion2b",
         "vit_large_patch14_clip_224.laion2b", "vit_huge_patch14_clip_224.laion2b",
         "vit_base_patch16_clip_224.laion2b_ft_in12k", "vit_large_patch14_clip_224.laion2b_ft_in12k",
         "vit_huge_patch14_clip_224.laion2b_ft_in12k")


def synth_records(seed, gain, pr70=80.0, pr69=220.0, anchor=0.18, flat=False, top_only=False):
    """scenario model of plan 12 (design/scen.py, re-implemented): Pythia PR* geometric in log params; c = Bbar + 0.0019
    + noise; T = c(70m) + anchor + gain log2(N/70M) + noise; mKNN and debiased CKA built similarly with flat baselines."""
    rng = np.random.default_rng(seed)
    N = np.array([70.4e6, 162e6, 405e6, 1.01e9, 1.41e9, 2.78e9, 6.86e9])
    nv, n = 17, 7
    prv = rng.uniform(40, 200, nv)
    if flat:
        prl = np.full(n, 150.0)
    else:
        prl = pr70 * (pr69 / pr70) ** ((np.log(N) - np.log(N[0])) / (np.log(N[-1]) - np.log(N[0])))
    Bbar = [[math.sqrt(prl[i] * prv[v]) / 1023 for v in range(nv)] for i in range(n)]
    MK = 10240
    rec = {}
    for pre in PV.PRES:
        R = {}
        for i in range(n):
            lm = f"L{i}"
            R[lm] = {}
            for v in range(nv):
                vit = VIT17[v]
                d = {}
                for m in PV.METRICS:
                    q = {}
                    for s in SUBS:
                        if m == "cka_biased":
                            c = Bbar[i][v] + 0.0019 + rng.normal(0, 0.0002)
                            T = (Bbar[0][v] + 0.0019 + anchor + gain * math.log2(N[i] / N[0]) + rng.normal(0, 0.004))
                            if top_only and i >= 5:
                                c += 0.05 * (i - 4)
                        elif m == "mknn":
                            cc = int(round((10 / 1023 + 0.004 + 0.0005 * i) * MK + rng.integers(0, 5)))
                            tt = int(round((0.05 + 0.01 * i) * MK + rng.integers(0, 20)))
                            c, T = Fraction(cc, MK), Fraction(tt, MK)
                        else:
                            c = 0.004 + 0.0002 * i + rng.normal(0, 0.0001)
                            T = 0.2 + 0.02 * i + rng.normal(0, 0.002)
                        if isinstance(T, Fraction):
                            sc = Fraction(T - c, 1 - c) if T > c else Fraction(0)
                            mu = c - Fraction(3, MK)
                        else:
                            sc = max((T - c) / (1 - c), 0.0)
                            mu = c - 0.0019
                        q[s] = {"raw": T, "c": c, "s_cal": sc, "mu": mu, "unc": T - mu, "p": 0.01, "a": 3, "b": 4,
                                "n_tied_max": 1, "LA": 7 + 4 * i, "LB": 12, "n_entries": 100,
                                "n_lm_layers": (7 + 4 * i) if s == "full" else 5}
                    q["full"].update({"c_entry": q["full"]["c"] * (0.9 if m != "mknn" else 1),
                                      "mu_entry": q["full"]["mu"], "max_entry_mu": q["full"]["mu"]})
                    f = q["full"]
                    M = f["s_cal"] * (0.95 if m != "mknn" else 1)
                    q["ctm"] = {"M": M, "M_unc": f["unc"], "TM": f["raw"], "cM": f["c"], "muM": f["mu"], "a": 3, "b": 4}
                    d[m] = q
                R[lm][vit] = d
        rec[pre] = R
    lms = [f"L{i}" for i in range(n)]
    vits = list(VIT17)
    tags = PV.PYT_SIZES
    pb = {"mu_band": [-0.0003, 0.0008], "per_pre": {}}
    for pre in PV.PRES:
        mat, tr, s_, _ = PV.b4_material(Bbar, n, nv)
        pb["per_pre"][pre] = {
            "PR_star": {lms[i]: float(prl[i]) for i in range(n)},
            "Bbar": {lms[i]: {vits[v]: Bbar[i][v] for v in range(nv)} for i in range(n)},
            "Blate": {lms[i]: {vits[v]: Bbar[i][v] for v in range(nv)} for i in range(n)},
            "material_Bbar": mat, "s": s_, "s_late": s_, "H2_branch": "H2a" if mat else "H2b",
            "band": {tags[i]: dict(zip(("lo", "hi", "f"), PV.band_ends(pre, prl[i]))) for i in range(n)},
            "band_flat": list(PV.FLAT_BAND[pre]),
            "extrapolated": {tags[i]: bool(prl[i] < PV.PR_FIT_LO[pre]) for i in range(n)},
            "H2_same_as_H1": True}
    aux = {"num_params": list(N), "width": [512, 768, 1024, 2048, 2048, 2560, 4096],
           "depth": [6, 12, 24, 16, 24, 32, 32], "L_stored": [7, 13, 25, 17, 25, 33, 33],
           "vit_family": ["in21k"] * 4 + ["MAE"] * 3 + ["DINOv2"] * 4 + ["CLIP"] * 6}
    return rec, lms, vits, pb, aux


def run_verdicts(rec, lms, vits, pb, aux):
    pbv = {p: PV.partb_view(pb, p, lms, vits, PV.PYT_SIZES) for p in PV.PRES}
    ex = {}
    V = PV.verdicts_pyt(rec, lms, vits, pbv, aux, ex, PV.PRES)
    return V, ex


def _cat(V, ex, p, x="s_cal"):
    h = f"H|{p}|{x}|H1"
    return PV.category(V[f"{h}|member"], V[f"{h}|D"], V[f"{h}|p_perm"], V[f"{h}|p_mag"], V[f"PP2a|{p}|{x}|verdict"],
                       ex[f"x|I|PP2b|{p}|{x}|verdict"], V[f"{h}|R"], V[f"{h}|C"])


def t07():
    import importlib.util as iu
    spec = iu.spec_from_file_location("vk_shared", os.path.join(pyt_gate.AXIS, "verdict_keys.py"))
    VK = iu.module_from_spec(spec)
    spec.loader.exec_module(VK)
    req = [k for k in VK.required_keys("EI") if not k.startswith(("famtr|", "famdtr|", "panel18|", "G|"))]
    out = {}
    p = "prh_clip+l2"
    h = f"H|{p}|s_cal"
    # (i) strong drop: thresholds rise with size, small raw gain
    rec, lms, vits, pb, aux = synth_records(11, gain=0.004)
    V, ex = run_verdicts(rec, lms, vits, pb, aux)
    miss = [k for k in req if k not in V]
    check("t07 every contract key of sections A, B, C, H, R present", not miss, miss[:10])
    extra_keys = [k for k in V if k not in set(VK.required_keys("EI"))]
    check("t07 no non-contract key outside x|", not extra_keys, extra_keys[:10])
    out["strong"] = {k: V[k] for k in (f"{h}|H1|member", f"{h}|H1|D", f"{h}|H1|p_perm", f"{h}|H1|p_mag",
                                       f"{h}|H1|confirmed", f"{h}|gate", f"{h}|holm_order", f"PP2a|{p}|s_cal|verdict")}
    out["strong"]["x|I|K category"] = ex[f"x|I|K|{p}|s_cal|category"]
    check("t07 strong: H1 member", V[f"{h}|H1|member"] is True, out["strong"])
    check("t07 strong: D1 <= -0.10", V[f"{h}|H1|D"] <= -0.10, out["strong"])
    check("t07 strong: H1 confirmed", V[f"{h}|H1|confirmed"] is True, out["strong"])
    check("t07 strong: gate open", V[f"{h}|gate"] == "open (H1 confirmed)", out["strong"])
    check("t07 strong: secondaries tested with a Holm order", all(V[f"{h}|{k}|tested"] for k in ("H2", "H3", "H4"))
          and sorted(V[f"{h}|holm_order"].split(",")) == ["H2", "H3", "H4"], out["strong"])
    check("t07 strong: I's category from the components", ex[f"x|I|K|{p}|s_cal|category"] == _cat(V, ex, p)[0],
          out["strong"])
    check("t07 strong: H1 p = max(perm, mag)", V[f"{h}|H1|p"] == max(V[f"{h}|H1|p_perm"], V[f"{h}|H1|p_mag"]), None)
    check("t07 strong: per-comparison sign rule and keep identity hold (cka_biased)",
          all(V[f"signrule|{p}|cka_biased|{q}|{x}|nviol"] == 0 and V[f"keepid|{p}|cka_biased|{q}|{x}|nviol"] == 0
              for q in SUBS for x in ("s_cal", "unc")), None)
    V2, _ = run_verdicts(*synth_records(11, gain=0.004))
    check("t07 determinism (two runs identical)", all(
        (V[k] == V2[k]) or (isinstance(V[k], float) and isinstance(V2[k], float) and math.isnan(V[k]) and
                            math.isnan(V2[k])) for k in V), None)
    # (ii) flat PR*: B-bar not material -> E, Holm without gatekeeper
    V, ex = run_verdicts(*synth_records(12, gain=0.004, flat=True))
    out["flat"] = {f"{h}|H1|member": V[f"{h}|H1|member"], f"{h}|gate": V[f"{h}|gate"],
                   "category": ex[f"x|I|K|{p}|s_cal|category"]}
    check("t07 flat: E", ex[f"x|I|K|{p}|s_cal|category"] == "E", out["flat"])
    check("t07 flat: gate open without gatekeeper", V[f"{h}|gate"] == "open (H1 not a member)", out["flat"])
    check("t07 flat: H1 confirmed is null for a non-member", V[f"{h}|H1|confirmed"] is None, out["flat"])
    check("t07 flat: secondaries tested", all(V[f"{h}|{k}|tested"] for k in ("H2", "H3", "H4")), out["flat"])
    # (iii) large raw gain: thresholds rise but raw rises faster -> C, gate closed
    V, ex = run_verdicts(*synth_records(13, gain=0.03))
    out["fast"] = {f"{h}|H1|D": V[f"{h}|H1|D"], f"{h}|H1|confirmed": V[f"{h}|H1|confirmed"], f"{h}|gate": V[f"{h}|gate"],
                   f"{h}|H3|tested": V[f"{h}|H3|tested"], f"{h}|H3|confirmed": V[f"{h}|H3|confirmed"],
                   "category": ex[f"x|I|K|{p}|s_cal|category"], "H3_reading": ex[f"x|I|K|{p}|s_cal|H3_reading"]}
    check("t07 fast raw gain: C", ex[f"x|I|K|{p}|s_cal|category"] == "C", out["fast"])
    check("t07 fast: gate closed, secondaries not tested, confirmed null",
          V[f"{h}|gate"] == "closed" and V[f"{h}|H3|tested"] is False and V[f"{h}|H3|confirmed"] is None
          and V[f"{h}|holm_order"] is None, out["fast"])
    # (iv) drop confined to the largest models
    V, ex = run_verdicts(*synth_records(14, gain=0.012, top_only=True))
    out["top_only"] = {f"{h}|H1|D": V[f"{h}|H1|D"], f"{h}|H1|p_perm": V[f"{h}|H1|p_perm"],
                       f"{h}|H1|p_mag": V[f"{h}|H1|p_mag"], "category": ex[f"x|I|K|{p}|s_cal|category"]}
    check("t07 top-only: category consistent with components", ex[f"x|I|K|{p}|s_cal|category"] == _cat(V, ex, p)[0],
          out["top_only"])
    return out


def t08():
    spec = importlib.util.spec_from_file_location(
        "ol_verdicts_orig", os.path.join(os.path.dirname(HERE), "orig_copy", "ol_verdicts.py"))
    OV = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OV)
    rng = np.random.default_rng(8)
    nbad = 0
    for t in range(200):
        cols = [list(rng.integers(0, 3, 3).astype(float)) if t % 3 == 0 else list(rng.normal(size=3)) for _ in range(17)]
        a, b = OV.trend(cols), PV.trend(cols)
        if (a["T"] is None) != (b["T"] is None):
            nbad += 1
            continue
        if a["T"] is not None and not np.array_equal(a["T"], b["T"]):
            nbad += 1
        if a["rho"] != b["rho"] or a["nconst"] != b["nconst"]:
            nbad += 1
    check("t08 generic trend == OpenLLaMA trend bitwise (n = 3, 200 random cases incl. ties)", nbad == 0, nbad)
    from scipy.stats import spearmanr
    cols = [list(rng.normal(size=7)) for _ in range(17)]
    t = PV.trend(cols)
    ref = np.mean([spearmanr(np.arange(7), c).correlation for c in cols])
    check("t08 n = 7 identity Tr = mean scipy Spearman", abs(t["obs"] - ref) < 1e-12, (t["obs"], ref))
    check("t08 5040 relabellings", len(t["T"]) == 5040, len(t["T"]))


def t09():
    lo, hi, f = PV.band_ends("prh_clip+l2", 124.5)
    check("t09 prh band at PR* 124.5", abs(f - (0.001069 + 0.012920 / math.sqrt(124.5))) < 1e-15 and
          abs(hi - lo - 0.0016) < 1e-15, (lo, hi, f))
    lo, hi, f = PV.band_ends("l2", 79.6)
    check("t09 l2 band", abs(hi - lo - 0.0036) < 1e-15, (lo, hi))
    check("t09 M5 limits", PV.PR_FIT_LO == {"prh_clip+l2": 124.5, "l2": 79.6}, PV.PR_FIT_LO)


def t10():
    """the common-ViT-set rule (plan 5.1) against the per-score exclusion, and the censoring counterexample to the
    per-ViT sign rule ([op3])."""
    raw = [[0.1 * i + 0.01 * v for i in range(7)] for v in range(17)]
    sc = [[0.05 * (7 - i) + 0.001 * v for i in range(7)] for v in range(17)]
    raw[3] = [0.1 * (7 - i) for i in range(7)]          # ViT 3: raw decreasing in size
    sc[3] = [0.0] * 7                                   # ViT 3: every s_cal censored -> constant under s_cal only
    dl, dc = PV.dtrend(sc, raw, "legacy"), PV.dtrend(sc, raw, "common")
    check("t10 common set excludes the ViT constant under one score", dc["nv"] == 16 and 3 not in dc["V"], dc["nv"])
    check("t10 legacy dTr = -1 - 15/17", abs(dl["obs"] - (-1.0 - 15.0 / 17.0)) < 1e-12, dl["obs"])
    check("t10 common dTr = -2", abs(dc["obs"] + 2.0) < 1e-12, dc["obs"])
    T = [0.30, 0.29, 0.50, 0.60, 0.70, 0.80, 0.90]
    c = [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37]      # non-decreasing in size
    S = [PV.scal(t, cc) for t, cc in zip(T, c)]
    r_raw, r_cal = PV.trend([T])["obs"], PV.trend([S])["obs"]
    check("t10 censoring tie raises rho although c is non-decreasing (plan 8 per-ViT inequality fails, [op3])",
          S[0] == S[1] == 0.0 and r_cal > r_raw, (r_raw, r_cal))
    return {"legacy": dl["obs"], "common": dc["obs"], "rho_raw": r_raw, "rho_scal": r_cal}


def main():
    os.makedirs(OUTD, exist_ok=True)
    t0 = time.time()
    rep = {"what": "plan V-P9 known answers, implementation I (synthetic inputs only)", "tests": {}}
    ok = True
    for fn in (t01, t02, t03, t04, t05, t06, t07, t08, t09, t10):
        n0 = len(RESULTS)
        try:
            det = fn()
            rep["tests"][fn.__name__] = {"pass": True, "n_checks": len(RESULTS) - n0, "detail": det}
        except Exception as e:   # noqa: BLE001
            ok = False
            rep["tests"][fn.__name__] = {"pass": False, "n_checks": len(RESULTS) - n0, "error": f"{type(e).__name__}: {e}",
                                         "traceback": traceback.format_exc()}
        print(f"{fn.__name__}: {'PASS' if rep['tests'][fn.__name__]['pass'] else 'FAIL'}", flush=True)
    rep["checks"] = RESULTS
    rep["n_checks"] = len(RESULTS)
    rep["n_failed_checks"] = sum(1 for r in RESULTS if not r["pass"])
    rep["pass"] = ok
    rep["secs"] = time.time() - t0
    fn = os.path.join(OUTD, f"test_report_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")

    def dflt(o):
        if isinstance(o, Fraction):
            return float(o)
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    with open(fn, "w") as fh:
        json.dump(rep, fh, indent=1, default=dflt)
    print(f"V-P9 I: {rep['n_checks']} checks, {rep['n_failed_checks']} failed, pass={ok}; report {fn}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
