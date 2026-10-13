"""Pythia axis, sub-role R: V-P9 known-answer tests of R's rules (TEST ONLY; NEW FILE, PYT-AXIS).

Plan ANALYSIS_PLAN_PYTHIA.md (6de89603...) section 11 V-P9: "Synthetic inputs with hand-computed gate and Holm
decisions, IUT p, TOST p (including the degenerate case), P-P2b classifications under per-model bands, and the
section 7 category table (every row reachable); the exact p of the section 12 pattern table reproduced exactly".
This file covers R's share (the stand-in end-to-end run is make_test_fixture_pyt.py + pyt_axis_review.py; the
cross-implementation part needs E and I and is not R's to run). Every expected value below is written by hand from
the plan text, not computed by R's code.
Runs only in S0b test mode (PYT_S0B_TEST=1, before CODE FROZEN; else exit 3). Output: --out JSON under
analysis/s12_layermax_pythia/s0b_scratch/R/ (contract section 13).
usage: PYT_S0B_TEST=1 launch_pyt.sh NAME 1 -- env PYTHONDONTWRITEBYTECODE=1 venv/bin/python \
           analysis/s12_layermax_pythia/review/tests/test_vp9_r.py --out F
"""
import argparse
import json
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.dirname(HERE)
sys.path.insert(0, REVIEW)

RESULTS = []


def check(name, got, want, tol=None):
    if tol is None:
        ok = got == want
    else:
        ok = got is not None and abs(got - want) <= tol
    RESULTS.append({"test": name, "got": got, "want": want, "ok": bool(ok)})
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import pyt_review_gate as G
    if not G.test_mode():
        sys.exit(3)
    out = os.path.realpath(args.out)
    if not out.startswith(os.path.realpath(G.test_root()) + os.sep):
        print(f"--out must lie under {G.test_root()}", file=sys.stderr)
        sys.exit(3)
    import pyt_review_conf as CF
    import pyt_review_lib as RL
    import pyt_review_pythia as RP

    # ---------------------------------------------------------------- 1. section 12 pattern table (exact p_perm)
    # every one of 17 ViTs has raw ranks 1..7 and the given calibrated rank pattern; D1 = s * dTr with s = +1
    patterns = {  # name: (calibrated ranks of sizes 1..7, plan dTr (3 decimals), plan count k of k/5040)
        "top pair swapped": ((1, 2, 3, 4, 5, 7, 6), -0.036, 2520),
        "top 3 reversed": ((1, 2, 3, 4, 7, 6, 5), -0.143, 1800),
        "top 4 reversed": ((1, 2, 3, 7, 6, 5, 4), -0.357, 966),
        "top 5 reversed": ((1, 2, 7, 6, 5, 4, 3), -0.714, 444),
        "6.9B drops to rank 4": ((1, 2, 3, 5, 6, 7, 4), -0.214, 1440),
        "6.9B drops to rank 1": ((2, 3, 4, 5, 6, 7, 1), -0.750, 720),
        "three disjoint adjacent swaps (1-2, 3-4, 5-6)": ((2, 1, 4, 3, 6, 5, 7), -0.107, 1692),
        "full reversal": ((7, 6, 5, 4, 3, 2, 1), -2.000, 1),
    }
    raw = [list(range(1, 8))] * 17
    praw = RL.TrendParts(raw, 7)
    for name, (pat, dtr, k) in patterns.items():
        pc = RL.TrendParts([list(pat)] * 17, 7)
        vec, dv, nV = RL.diff_vec(pc, praw, "common", 1)
        check(f"pattern|{name}|dTr", round(float(vec[0]), 3), dtr)
        check(f"pattern|{name}|k", int(round(RL.p_le(vec) * 5040)), k)
        check(f"pattern|{name}|nV", nV, 17)

    # ---------------------------------------------------------------- 2. Spearman with 7 sizes: one adjacent swap
    one = RL.TrendParts([[1, 2, 3, 4, 5, 7, 6]] + [list(range(1, 8))] * 16, 7)
    check("F2|one adjacent swap moves Tr_P by 2/56/17", round(1 - float(one.tr_vec()[0]), 6), round(2 / 56 / 17, 6))

    # ---------------------------------------------------------------- 3. common-ViT-set rule vs legacy (B2.2)
    rows_raw = [list(range(1, 8))] * 16 + [[5] * 7]          # ViT 17 constant under raw
    rows_cal = [[1, 2, 3, 4, 5, 7, 6]] * 16 + [[1, 2, 3, 4, 5, 6, 7]]
    a, b = RL.TrendParts(rows_cal, 7), RL.TrendParts(rows_raw, 7)
    vc, _, nVc = RL.diff_vec(a, b, "common")
    vl, _, _ = RL.diff_vec(a, b, "legacy")
    check("B2.2|common nV", nVc, 16)
    check("B2.2|common dTr = -2/56", float(vc[0]), -2 / 56, 1e-15)
    check("B2.2|legacy dTr = (16*(1-2/56)+1)/17 - 1", float(vl[0]), (16 * (1 - 2 / 56) + 1) / 17 - 1, 1e-15)
    check("5.1|tr0 (rho := 0 for the constant ViT) of raw = 16/17", float(b.tr0_vec()[0]), 16 / 17, 1e-15)

    # ---------------------------------------------------------------- 4. IUT p, magnitude p, TOST p, degenerate
    d = np.array([-0.2] * 17)                                  # every draw has mean -0.2
    dr = RL.boot_draws(d, "test|const", 2000)
    check("p_mag all draws -0.2 < -0.10 -> 1/2001", RL.p_mag(dr, 0.10), 1 / 2001, 0)
    check("p_mag all draws -0.2, margin 0.25 -> 2001/2001", RL.p_mag(dr, 0.25), 1.0, 0)
    dr0 = RL.boot_draws(np.zeros(17), "test|zero", 2000)
    t0 = RL.tost_p(0.0, dr0, 0.10)
    check("TOST degenerate: p_lo = p_hi = p = 1/2001", (t0["p_lo"], t0["p_hi"], t0["p"]), (1 / 2001, 1 / 2001, 1 / 2001))
    check("TOST observed outside (-m, m) -> p := 1", RL.tost_p(0.10, dr0, 0.10)["p"], 1.0)
    draws = np.array([-0.12] * 30 + [0.0] * 1940 + [0.11] * 30)
    t = RL.tost_p(0.0, draws, 0.10)
    check("TOST p_lo = 31/2001", t["p_lo"], 31 / 2001, 0)
    check("TOST p_hi = 31/2001, p = max", (t["p_hi"], t["p"]), (31 / 2001, 31 / 2001))
    pz = RL.TrendParts([[1, 2, 3, 4, 5, 6, 7]] * 17, 7)
    pz2 = RL.TrendParts([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]] * 17, 7)
    check("same_ranks (degenerate test, exact ranks)", RL.same_ranks(pz, pz2), True)
    check("same_ranks false after one swap", RL.same_ranks(one, pz), False)
    # IUT: p_H1 = max(p_perm, p_mag)
    vecp = RL.diff_vec(RL.TrendParts([[7, 6, 5, 4, 3, 2, 1]] * 17, 7), praw, "common", 1)[0]
    check("IUT max(p_perm=1/5040, p_mag=1/2001) = 1/2001", max(RL.p_le(vecp), RL.p_mag(dr, 0.10)), 1 / 2001)
    # contract section 2 tolerances: p_mag counts D* >= -m - 1e-9; TOST counts <= -m + 1e-9 and >= m - 1e-9
    dm = np.full(2000, -0.10)
    check("p_mag with every draw exactly -0.10 -> 2001/2001 (counted)", RL.p_mag(dm, 0.10), 1.0)
    check("TOST p_lo with every draw exactly -0.10 -> 2001/2001", RL.tost_p(0.0, dm, 0.10)["p_lo"], 1.0)
    # exact Delta rho = 0 with different rank vectors (degenerate but not ranks_identical)
    a1 = RL.TrendParts([[2, 1, 3, 4, 5, 6, 7]] * 17, 7)
    b1 = RL.TrendParts([[1, 2, 3, 4, 5, 7, 6]] * 17, 7)
    check("degenerate: equal rho from different ranks (exact)", (RL.degenerate(a1, b1), RL.same_ranks(a1, b1)),
          (True, False))
    check("degenerate false after a real change", RL.degenerate(one, pz), False)
    # contract section 5 [plan-gap]: a censoring tie of a discordant pair raises Spearman's rho
    r_raw = RL.TrendParts([[2, 1, 3, 4, 5, 6, 7]], 7).rho_id()[0]
    r_tie = RL.TrendParts([[0, 0, 3, 4, 5, 6, 7]], 7).rho_id()[0]
    check("plan-gap example: rho 0.964 -> 0.991 under a censoring tie", (round(r_raw, 3), round(r_tie, 3)), (0.964, 0.991))
    check("tie_not_raw detects the censoring tie", RP._tie_not_raw([0, 0, 3, 4, 5, 6, 7], [2, 1, 3, 4, 5, 6, 7]), True)

    # ---------------------------------------------------------------- 5. Holm (hand-computed) and the gate
    h = RL.holm({"H2a": 0.010, "H3": 0.020, "H4": 0.030}, 0.05, order=["H2a", "H3", "H4"])
    check("Holm all reject (0.01<=0.0167, 0.02<=0.025, 0.03<=0.05)", [h[k]["reject"] for k in ("H2a", "H3", "H4")],
          [True, True, True])
    check("Holm adjusted p (0.03, 0.04, 0.04)", [round(h[k]["p_adj"], 12) for k in ("H2a", "H3", "H4")], [0.03, 0.04, 0.04])
    h = RL.holm({"H2b": 0.020, "H3": 0.001, "H4": 0.040}, 0.05, order=["H2b", "H3", "H4"])
    check("Holm (0.001 rej, 0.020 rej at 0.025, 0.040 rej at 0.05)", [h[k]["reject"] for k in ("H3", "H2b", "H4")],
          [True, True, True])
    h = RL.holm({"H2b": 0.030, "H3": 0.001, "H4": 0.026}, 0.05, order=["H2b", "H3", "H4"])
    check("Holm step-down stops (0.026 > 0.025 -> 0.030 not tested)", [h[k]["reject"] for k in ("H3", "H4", "H2b")],
          [True, False, False])
    h = RL.holm({"H2a": 0.02, "H3": 0.02, "H4": 0.02}, 0.05, order=["H2a", "H3", "H4"])
    check("Holm ties: 0.02 > 0.0167 -> none", [h[k]["reject"] for k in ("H2a", "H3", "H4")], [False, False, False])
    check("Holm tie adjusted p = 0.06", [round(h[k]["p_adj"], 12) for k in ("H2a", "H3", "H4")], [0.06, 0.06, 0.06])

    # ---------------------------------------------------------------- 6. section 7 category table (every row)
    cat = RP.category
    cases = [  # (member, D1, p_perm, p_mag, pp2a, pp2b, R, C) -> (code, F_reason, premise)
        ((False, -0.5, 0.001, 0.001, "holds", "holds", 0.9, 0.1), ("E", None, None)),
        ((True, 0.10, 0.9, 0.9, "holds", "holds", 0.9, 0.9), ("D", None, None)),
        ((True, 0.099, 0.5, 0.5, "holds", "holds", 0.9, 0.9), ("C", None, None)),
        ((True, -0.0999, 0.001, 0.001, "holds", "holds", 0.9, 0.8), ("C", None, None)),
        ((True, -0.10, 0.06, 0.001, "holds", "holds", 0.9, 0.8), ("F", RP.F_REASON_PERM, None)),
        ((True, -0.30, 0.001, 0.2, "holds", "holds", 0.9, 0.6), ("F", RP.F_REASON_MAG, None)),
        ((True, -0.30, 0.2, 0.2, "holds", "holds", 0.9, 0.6), ("F", RP.F_REASON_PERM + "; " + RP.F_REASON_MAG, None)),
        ((True, -0.30, 0.001, 0.001, "fails", "holds", 0.9, 0.6), ("B", None, None)),
        ((True, -0.30, 0.001, 0.001, "holds", "fails", 0.9, 0.6), ("B", None, None)),
        ((True, -0.80, 0.001, 0.001, "holds", "holds", 0.9, 0.1), ("A", None, "verified")),
        ((True, -0.80, 0.001, 0.001, "holds", "holds", 0.60, 0.40), ("A", None, "verified")),
        ((True, -0.30, 0.001, 0.001, "holds", "holds", 0.9, 0.6), ("A-att", None, "verified")),
        ((True, -0.30, 0.001, 0.001, "holds", "holds", 0.59, 0.2), ("A-prime", None, "verified")),
        ((True, -0.80, 0.001, 0.001, "not evaluable", "holds", 0.9, 0.1), ("A", None, "unverified")),
        ((True, -0.80, 0.001, 0.001, "holds", "not evaluable", 0.9, 0.1), ("A", None, "unverified")),
        ((True, -0.10, 0.05, 0.05, "holds", "holds", 0.9, 0.85), ("A-att", None, "verified")),
    ]
    reached = set()
    for i, (inp, want) in enumerate(cases):
        got = cat(*inp)
        check(f"category case {i} {inp}", got, want)
        reached.add(got[0])
    check("every category row reached", sorted(reached), sorted(["E", "D", "C", "F", "B", "A", "A-att", "A-prime"]))
    # tau curve: with tau = 0.20, D1 = -0.15 is "C"; with tau = 0.03 and p_H1 unchanged it stays confirmed
    check("category tau=0.20, D1=-0.15 -> C", cat(True, -0.15, 0.001, 0.001, "holds", "holds", 0.9, 0.5, tau=0.20)[0], "C")
    check("category tau=0.03, D1=-0.05, conf given -> A-att",
          cat(True, -0.05, 0.001, 0.001, "holds", "holds", 0.9, 0.85, tau=0.03, conf=True)[0], "A-att")
    check("P-P2a verdict: (i) fails -> fails", RP.pp2a_verdict(False, "holds"), "fails")
    check("P-P2a verdict: (ii) not evaluable -> not evaluable", RP.pp2a_verdict(True, "not evaluable"), "not evaluable")
    check("P-P2a verdict: both hold -> holds", RP.pp2a_verdict(True, "holds"), "holds")

    # ---------------------------------------------------------------- 7. immunity readings
    imm = RP.immunity
    check("immunity moved", imm(0.10, False, True, True), "moved")
    check("immunity degenerate", imm(0.0, True, False, False), "no comparison changed order")
    check("immunity gate closed", imm(0.05, False, False, False), "kept, not tested (gate closed)")
    check("immunity confirmed", imm(0.05, False, True, True), "equivalence confirmed")
    check("immunity not established", imm(0.05, False, True, False), "kept, equivalence not established")

    # ---------------------------------------------------------------- 8. P-P2b classifications under per-model bands
    f1 = RP.forecast_one
    # a: T=0.30, B=0.10, band (0.0010, 0.0030) -> c in [0.101, 0.103], s_cal in [(0.30-0.103)/0.897, (0.30-0.101)/0.899]
    #    = [0.21962, 0.22136]; b: T=0.31, B=0.12, band (0.0005, 0.0021) -> c in [0.1205, 0.1221] -> s_cal in [0.21403, 0.21546]
    check("P-P2b below (max_b < min_a)", f1("s_cal", 0.30, 0.31, 0.10, 0.12, (0.0010, 0.0030), (0.0005, 0.0021)), "below")
    # b: T=0.33 -> s_cal_b in [0.23766, 0.23948] > a max 0.22136 -> above
    check("P-P2b above (min_b > max_a)", f1("s_cal", 0.30, 0.33, 0.10, 0.12, (0.0010, 0.0030), (0.0005, 0.0021)), "above")
    # b: T=0.3195 -> s_cal_b in [(0.3195-0.1221)/0.8779, (0.3195-0.1205)/0.8795] = [0.22486, 0.22627]
    #    a max 0.22136 < b min 0.22486 -> above
    check("P-P2b above (narrow)", f1("s_cal", 0.30, 0.3195, 0.10, 0.12, (0.0010, 0.0030), (0.0005, 0.0021)), "above")
    # overlap -> undecided: b T = 0.3160: [(0.3160-0.1221)/0.8779, (0.3160-0.1205)/0.8795] = [0.22087, 0.22229]
    #    overlaps a [0.21962, 0.22136] -> undecided
    check("P-P2b undecided (intervals overlap)", f1("s_cal", 0.30, 0.3160, 0.10, 0.12, (0.0010, 0.0030), (0.0005, 0.0021)),
          "undecided")
    # censoring: a T = 0.1005 < c_a min 0.101 -> a always 0; b T = 0.1210 in [0.1205, 0.1221] -> b can be 0 or > 0
    check("P-P2b undecided (one side censored, other not orderable)",
          f1("s_cal", 0.1005, 0.1210, 0.10, 0.12, (0.0010, 0.0030), (0.0005, 0.0021)), "undecided")
    check("P-P2b tie (both 0 at every offset)", f1("s_cal", 0.1005, 0.1200, 0.10, 0.12, (0.0010, 0.0030), (0.0005, 0.0021)),
          "tie")
    # b always > 0 (T = 0.13 > 0.1221), a always 0 -> above
    check("P-P2b above (a censored at every offset, b positive)",
          f1("s_cal", 0.1005, 0.13, 0.10, 0.12, (0.0010, 0.0030), (0.0005, 0.0021)), "above")
    # uncensored: unc = T - mu, mu in [B - 0.0003, B + 0.0008]; a: T-B = 0.20 -> [0.1992, 0.2003]; b: T-B = 0.198 -> max 0.1983
    check("P-P2b unc below", f1("unc", 0.30, 0.318, 0.10, 0.12, CF.UNC_BAND, CF.UNC_BAND), "below")
    # ---------------------------------------------------------------- 9. integer needs and the band formula
    check("ceil(0.9*70) = 63", RP.ceil_div(9 * 70, 10), 63)
    check("ceil(0.9*357) = 322", RP.ceil_div(9 * 357, 10), 322)
    check("ceil(2*357/3) = 238", RP.ceil_div(2 * 357, 3), 238)
    check("band f(PR*=124.5) prh = 0.001069 + 0.012920/sqrt(124.5)", round(0.001069 + 0.012920 / np.sqrt(124.5), 9),
          round(CF.BAND["prh_clip+l2"]["a"] + CF.BAND["prh_clip+l2"]["b"] / np.sqrt(124.5), 9))
    check("Prop.5(1) mKNN baseline 10/1023", RP.BASE_MKNN, Fraction(10, 1023))
    # ---------------------------------------------------------------- 10. fixed depths and grid5 against the plan table
    for s in CF.PYT_SIZES:
        L = CF.PYTHIA.L_stored[s]
        check(f"fixed r0.5 {s}", RL.fixed_layer(L, 0.5), CF.FIXED_DEPTH["0.5"][s])
        check(f"fixed r1.0 {s}", RL.fixed_layer(L, 1.0), CF.FIXED_DEPTH["1.0"][s])
        check(f"grid5 {s}", RL.grid_layers(L, 5), CF.GRID5_LM[s])
    check("grid9 saturates at 70m", len(RL.grid_layers(7, 9)), 7)
    check("grid17 saturates at 160m", len(RL.grid_layers(13, 17)), 13)
    check("grid9 does not saturate at 160m", len(RL.grid_layers(13, 9)) < 13, True)

    # ---------------------------------------------------------------- 11. lead ruling R1 (2026-10-10 ~03:05Z)
    # "A key that the contract defines only for some variants is null in every other variant, in every
    # implementation; so under x = unc all three M4 keys (M4, M4_n_zero, M4_sizes) are null." section_K is run on
    # hand-made inputs (stub Family / Part B; every category row, both x) and the restricted keys are checked against
    # hand-written values: M4 trio (s_cal only), premise / M2 / M5_A / M5_A_sizes (A, A-att, A-prime only), F_reason
    # (F only), M5_B (B only); M2_x and M5_extrap_sizes are defined for every variant (contract 10) and stay written.
    # Inputs: 7 sizes; 70m and 160m extrapolated; 160m outside its band (in_band False), the rest inside; the median
    # s_cal of 70m is 0 and 10 of the 119 pairs have s_cal = 0, so M4 = True by the median rule (10 < 119/3);
    # grid5 D1 = -0.20 (so M2 = not(-0.20 <= -0.10) = False); P-P2a(ii) holds; no wrong P-P2b forecast.
    from types import SimpleNamespace as NS
    P1 = "prh_clip+l2"
    sizes = CF.PYT_SIZES
    vits17 = [f"v{i}" for i in range(17)]

    class PBStub:
        def extrapolated(self, P, s):
            return s in ("70m", "160m")

        def same_as_H1(self, P):
            return False

        def material_db(self, P, db):
            return True

    def fam(member, D1, p_perm, p_mag, conf, R, C):
        tst = {"D": 0.0, "degenerate": False}
        return NS(member=member, D1=D1, p_perm=p_perm, p_mag=p_mag, pH1=max(p_perm, p_mag), conf_raw=conf,
                  H1_confirmed=conf if member else None, R=R, C=C, nV=17, kedge=False, open=False, gate="closed",
                  holm=None, p={"H2": None, "H3": None, "H4": None}, confirmed={"H2": None, "H3": None, "H4": None},
                  branch="H2b", D2=0.0, tost={("mknn", "full"): dict(tst), ("cka_debiased", "full"): dict(tst)})

    r1_cases = {  # name: (Family args, P-P2a verdict, P-P2b verdict) -> expected category, premise, F_reason
        "E": ((False, -0.30, 0.001, 0.001, True, 0.9, 0.1), "holds", "holds", "E", None, None),
        "D": ((True, 0.20, 0.5, 0.5, False, 0.9, 0.9), "holds", "holds", "D", None, None),
        "C": ((True, -0.05, 0.5, 0.5, False, 0.9, 0.9), "holds", "holds", "C", None, None),
        "F": ((True, -0.30, 0.2, 0.001, False, 0.9, 0.6), "holds", "holds", "F", RP.F_REASON_PERM, None),
        "B": ((True, -0.30, 0.001, 0.001, True, 0.9, 0.6), "fails", "holds", "B", None, None),
        "A": ((True, -0.30, 0.001, 0.001, True, 0.9, 0.1), "holds", "holds", "A", None, "verified"),
        "A-att": ((True, -0.30, 0.001, 0.001, True, 0.9, 0.6), "holds", "holds", "A-att", None, "verified"),
        "A-prime": ((True, -0.30, 0.001, 0.001, True, 0.5, 0.1), "holds", "not evaluable", "A-prime", None,
                    "unverified"),
    }
    for name, (fa, pp2a_v, pp2b_v, want_code, want_reason, want_prem) in r1_cases.items():
        vk = {f"R|{P1}|n_scal_zero": 10}
        for s in sizes:
            vk[f"R|{P1}|median_scal|{s}"] = 0.0 if s == "70m" else 0.05
        fams_k = {}
        for x in ("s_cal", "unc"):
            r = f"R|{P1}|{x}"
            vk.update({f"PP2a|{P1}|{x}|verdict": pp2a_v, f"PP2b|{P1}|{x}|verdict": pp2b_v,
                       f"PP2aii|{P1}|{x}|verdict": "holds", f"PP2b|{P1}|{x}|wrong_all_extrap": None,
                       f"PP5a|{P1}|{x}|verdict": "holds", f"{r}|fragile": False, f"{r}|fragile_sizes": "",
                       f"{r}|grid5|D1": -0.20, f"{r}|grid5|dtr_mknn": 0.0, f"{r}|grid5|dtr_dcka": 0.0,
                       f"{r}|five|carried": False, f"{r}|five|carried_D": False})
            for s in sizes:
                vk[f"PP2ai|{P1}|{x}|in_band|{s}"] = s != "160m"
            fams_k[(P1, x)] = fam(*fa)
        RP.section_K(vk, fams_k, None, PBStub(), CF.STANDIN, vits17, (P1,))
        Afam = want_code in ("A", "A-att", "A-prime")
        for x in ("s_cal", "unc"):
            k = f"K|{P1}|{x}"
            want = {"category": want_code, "premise": want_prem, "F_reason": want_reason,
                    "M2": False if Afam else None, "M2_x": -0.20,
                    "M4": True if x == "s_cal" else None, "M4_sizes": "70m" if x == "s_cal" else None,
                    "M4_n_zero": 10 if x == "s_cal" else None,
                    "M5_extrap_sizes": "70m,160m", "M5_B": True if want_code == "B" else None,
                    "M5_A": True if Afam else None, "M5_A_sizes": "70m" if Afam else None}
            for f, w in want.items():
                got = vk[f"{k}|{f}"]
                check(f"R1|{name}|{x}|{f}", got, w)
                if w is None:   # null must be JSON null, not "" / 0 / False
                    check(f"R1|{name}|{x}|{f}|is None", got is None, True)

    n_ok = sum(r["ok"] for r in RESULTS)
    rep = {"what": "V-P9 known answers, R's share (S0b test mode)", "n": len(RESULTS), "n_ok": n_ok,
           "pass": n_ok == len(RESULTS), "results": RESULTS}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    for r in RESULTS:
        if not r["ok"]:
            print("FAIL", r)
    print(json.dumps({"n": len(RESULTS), "n_ok": n_ok, "pass": rep["pass"]}))
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
