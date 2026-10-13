"""PYT-AXIS (new file, implementation E, S0b sub-role pyt-code-E): plan V-P9, known answers for E's generalised code
(analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md section 11, step S0b, CPU). GATED (lm_core.gate(); in the S0b
test mode it needs no Part B). Synthetic inputs only; reads no pair file and no feature.

Checks (each with its hand-computed expectation written below):
  1. the plan section 12 pattern table: exact p_perm over 5040 relabellings and Delta Tr, every ViT with raw ranks
     1..7 and the stated calibrated pattern (top pair swapped 2520/5040, top 3 reversed 1800, top 4 reversed 966,
     top 5 reversed 444, 6.9B to rank 4 1440, 6.9B to rank 1 720, three disjoint adjacent swaps 1692, full reversal 1);
  2. bootstrap p (spec 6): p_mag and TOST on constant vectors (degenerate case p = 1/2001; outside the margin p = 1);
     the IUT p = max(p_perm, p_mag);
  3. gate and Holm through sec_H_R on synthetic families: (i) H1 confirmed, H2b -> all three rejected, order H2,H3,H4,
     p_holm 3/2001; (ii) H1 confirmed, H2a with D2 = 0 -> order H3,H4,H2, H3 and H4 rejected, H2 not, p_holm 3/2001,
     3/2001, 1; (iii) H1 a member, not confirmed -> gate closed, nothing tested; (iv) H1 not a member -> gate open,
     Holm run; degenerate TOST readings for H3/H4 (D = 0, p = 1/2001, ranks identical);
  4. P-P2b forecasts (E's own implementation, plan 9 B6) on hand cases: below, above, tie, undecided (s_cal), below,
     above, undecided (unc);
  5. the plan section 7 category table, every row reachable (E's own axis_category; the contract's K is written by
     forecast_pyt.py): E, D, C, F (both sub-reasons, each alone), B (via P-P2a, via P-P2b), A, A-att, A-prime,
     premise unverified, undefined.
Run (S0b, CPU, 1 thread): PYT_S0B_TEST=1 ... launch_pyt.sh pyt_E_vp9 1 -- venv/bin/python .../early/vl9_known_answers.py
Output: <RES>/validation/VP9_known_answers.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402

PRE = "prh_clip+l2"
NB1 = 2001.0


def pattern_table(AG):
    pats = {"top pair swapped": ((1, 2, 3, 4, 5, 7, 6), 2520, -0.036),
            "top 3 reversed": ((1, 2, 3, 4, 7, 6, 5), 1800, -0.143),
            "top 4 reversed": ((1, 2, 3, 7, 6, 5, 4), 966, -0.357),
            "top 5 reversed": ((1, 2, 7, 6, 5, 4, 3), 444, -0.714),
            "6.9B drops to rank 4": ((1, 2, 3, 5, 6, 7, 4), 1440, -0.214),
            "6.9B drops to rank 1": ((2, 3, 4, 5, 6, 7, 1), 720, -0.750),
            "three disjoint adjacent swaps (1-2, 3-4, 5-6)": ((2, 1, 4, 3, 6, 5, 7), 1692, -0.107),
            "full reversal": ((7, 6, 5, 4, 3, 2, 1), 1, -2.000)}
    raw = AG.rho_table([tuple(range(1, 8))] * 17)
    out, ok = {}, True
    for name, (pat, want_k, want_d) in pats.items():
        cal = AG.rho_table([pat] * 17)
        D = AG.tr_series(cal) - AG.tr_series(raw)
        k = int(round(AG.p_le(D) * 5040))
        good = bool(k == want_k and round(float(D[0]), 3) == want_d)
        ok &= good
        out[name] = {"k": k, "expected_k": want_k, "dTr": float(D[0]), "expected_dTr_3dp": want_d, "pass": good}
    return {"rows": out, "pass": ok}


def bootstrap_checks(AG):
    out = {}
    z = AG.boot_draws(np.zeros(17), "S1.2-layermax-pythia|VP9|zero")
    lo, hi, p = AG.p_tost_tol(z, 0.0)
    out["degenerate_tost"] = {"p_lo": lo, "p_hi": hi, "p": p, "pass": lo == hi == p == 1 / NB1}
    m2 = AG.boot_draws(np.full(17, -2.0), "S1.2-layermax-pythia|VP9|m2")
    out["p_mag_all_minus2"] = {"p": AG.p_mag_tol(m2, 1), "pass": AG.p_mag_tol(m2, 1) == 1 / NB1}
    out["p_mag_all_zero"] = {"p": AG.p_mag_tol(z, 1), "pass": AG.p_mag_tol(z, 1) == 1.0}
    out["p_mag_sign_minus"] = {"p": AG.p_mag_tol(-m2, -1), "pass": AG.p_mag_tol(-m2, -1) == 1 / NB1}
    lo2, hi2, p2 = AG.p_tost_tol(z, 0.2)
    out["tost_outside_margin"] = {"p": p2, "pass": p2 == 1.0}
    lo3, hi3, p3 = AG.p_tost_tol(np.full(2000, 0.1), 0.05)
    out["tost_draws_at_margin_tolerance"] = {"p_hi": hi3, "pass": hi3 == 1.0}   # 0.1 >= 0.1 - 1e-9 counted
    out["iut"] = {"p": max(1 / 5040, 1 / NB1), "pass": max(1 / 5040, 1 / NB1) == 1 / NB1}
    out["pass"] = all(v["pass"] for v in out.values())
    return out


def synth_family(AG, cal_pat, mk_same=True):
    """17 ViTs x 7 sizes; cka_biased raw = 0.1 * size, s_cal / unc follow cal_pat (ranks); mKNN and debiased CKA keep
    their raw ranks under calibration (degenerate TOST); ctm M = M_unc = TM = raw."""
    sizes = list(LC.SIZES)
    lms = [f"syn_{s}" for s in sizes]
    REC = {}
    for i, lm in enumerate(lms):
        for vit, _ in LC.VITS:
            R = {"lm": lm, "vit": vit, "LA": 5, "LB": 5}
            for met in AG.METRICS:
                T = (i + 1) / 10.0
                if met == "cka_biased":
                    sc = cal_pat[i] / 100.0
                else:
                    sc = T / 2 if mk_same else (7 - i) / 100.0
                if met == "mknn":
                    T, sc = Fraction(i + 1, 10), Fraction(i + 1, 20)
                d = {"T": T, "scal": sc, "unc": sc, "c": 0.05, "mu": 0.04, "c_entry": 0.05, "mu_entry": 0.04, "aA": 0,
                     "aB": 0, "n_tied": 1, "p": 0.01}
                R[met] = {q: dict(d) for q in AG.SUBSETS}
                R[met]["ctm"] = {"M": T, "M_unc": T, "TM": T, "cM": 0.05, "muMunc": 0.04, "aA": 0, "aB": 0,
                                 "aA_unc": 0, "aB_unc": 0}
                R[met]["D1_max_entry_mu"] = 0.04
                R[met]["null_profile"] = np.zeros(5)
            for pre in AG.PRES:
                REC[(pre, lm, vit)] = R
    return AG.Family(REC, lms, sizes)


def gate_holm_checks(AG):
    out = {}
    rev = (7, 6, 5, 4, 3, 2, 1)
    swap = (1, 2, 3, 4, 5, 7, 6)
    cases = {"i_confirmed_H2b": (rev, True, "H2b"), "ii_confirmed_H2a": (rev, True, "H2a"),
             "iii_member_not_confirmed": (swap, True, "H2b"), "iv_not_member": (swap, False, "H2b")}
    for name, (pat, member, branch) in cases.items():
        F = synth_family(AG, pat)
        v, x = {}, {}
        AG.sec_A(F, PRE, v, x, boot_ci=False)
        PBV = {"s": 1, "s_late": 1, "material": member, "H2_branch": branch}
        AG.sec_H_R(F, PRE, "s_cal", PBV, v, x)
        h = f"H|{PRE}|s_cal"
        got = {k: v[f"{h}|{k}"] for k in ("gate", "holm_order", "H1|D", "H1|p_perm", "H1|p_mag", "H1|p", "H1|confirmed",
                                          "H2|p", "H3|p", "H4|p", "H3|degenerate", "H4|degenerate", "H3|ranks_identical",
                                          "H2|p_holm", "H3|p_holm", "H4|p_holm", "H2|holm_reject", "H3|holm_reject",
                                          "H4|holm_reject", "H2|confirmed", "H2|tested")}
        if name == "i_confirmed_H2b":
            want = {"gate": "open (H1 confirmed)", "holm_order": "H2,H3,H4", "H1|D": -2.0, "H1|p_perm": 1 / 5040,
                    "H1|p_mag": 1 / NB1, "H1|p": 1 / NB1, "H1|confirmed": True, "H2|p": 1 / NB1, "H3|p": 1 / NB1,
                    "H4|p": 1 / NB1, "H3|degenerate": True, "H4|degenerate": True, "H3|ranks_identical": True,
                    "H2|p_holm": 3 / NB1, "H3|p_holm": 3 / NB1, "H4|p_holm": 3 / NB1, "H2|holm_reject": True,
                    "H3|holm_reject": True, "H4|holm_reject": True, "H2|confirmed": True, "H2|tested": True}
        elif name == "ii_confirmed_H2a":
            want = {"gate": "open (H1 confirmed)", "holm_order": "H3,H4,H2", "H1|confirmed": True, "H2|p": 1.0,
                    "H2|p_holm": 1.0, "H3|p_holm": 3 / NB1, "H4|p_holm": 3 / NB1, "H2|holm_reject": False,
                    "H3|holm_reject": True, "H4|holm_reject": True, "H2|confirmed": False}
        elif name == "iii_member_not_confirmed":
            want = {"gate": "closed", "holm_order": None, "H1|confirmed": False, "H2|p_holm": None, "H2|holm_reject": None,
                    "H2|confirmed": None, "H2|tested": False, "H3|degenerate": True}
        else:
            want = {"gate": "open (H1 not a member)", "holm_order": "H2,H3,H4", "H1|confirmed": None, "H2|tested": True,
                    "H2|holm_reject": True}
        bad = {k: {"got": got[k], "want": w} for k, w in want.items()
               if not (got[k] == w or (isinstance(w, float) and got[k] is not None and abs(got[k] - w) <= 1e-15))}
        out[name] = {"got": got, "n_checked": len(want), "mismatch": bad, "pass": not bad}
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def forecast_checks(AG):
    oa = ob = (0.001, 0.003)
    cases = {"below": ((0.30, 0.20, 0.10, 0.10), False, "below"), "above": ((0.20, 0.30, 0.10, 0.10), False, "above"),
             "tie": ((0.05, 0.08, 0.10, 0.10), False, "tie"), "undecided": ((0.2000, 0.2005, 0.10, 0.10), False, "undecided"),
             "unc_below": ((0.30, 0.20, 0.10, 0.10), True, "below"), "unc_above": ((0.20, 0.30, 0.10, 0.10), True, "above"),
             "unc_undecided": ((0.2000, 0.2005, 0.10, 0.10), True, "undecided")}
    out = {}
    for name, ((Ta, Tb, Ba, Bb), unc, want) in cases.items():
        band = (-0.0003, 0.0008) if unc else oa
        got = AG.ppb_forecast(Ta, Tb, Ba, Bb, band, band if unc else ob, unc=unc)
        out[name] = {"got": got, "want": want, "pass": got == want}
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def category_checks(AG):
    C = AG.axis_category
    rows = {
        "E": (C(False, -0.5, None, 0.001, 0.001, "holds", "holds", 0.9, 0.1), ("E", None, None)),
        "undefined": (C(True, None, False, None, None, "holds", "holds", None, None),
                      ("undefined (trend undefined)", None, None)),
        "D": (C(True, 0.2, False, 1.0, 1.0, "holds", "holds", 0.9, 1.0), ("D", None, None)),
        "C": (C(True, -0.05, False, 0.5, 0.9, "holds", "holds", 0.9, 0.85), ("C", None, None)),
        "F_both": (C(True, -0.3, False, 0.2, 0.3, "holds", "holds", 0.9, 0.6),
                   ("F", "not consistent across the seven sizes; size of the drop not established", None)),
        "F_perm": (C(True, -0.3, False, 0.2, 0.01, "holds", "holds", 0.9, 0.6),
                   ("F", "not consistent across the seven sizes", None)),
        "F_mag": (C(True, -0.3, False, 0.01, 0.2, "holds", "holds", 0.9, 0.6),
                  ("F", "size of the drop not established", None)),
        "B_PP2a": (C(True, -0.5, True, 0.001, 0.001, "fails", "holds", 0.9, 0.3), ("B", None, None)),
        "B_PP2b": (C(True, -0.5, True, 0.001, 0.001, "holds", "fails", 0.9, 0.3), ("B", None, None)),
        "A": (C(True, -0.6, True, 0.001, 0.001, "holds", "holds", 0.9, 0.3), ("A", None, "verified")),
        "A-att": (C(True, -0.3, True, 0.001, 0.001, "holds", "holds", 0.9, 0.6), ("A-att", None, "verified")),
        "A-prime": (C(True, -0.8, True, 0.001, 0.001, "holds", "holds", 0.5, -0.3), ("A-prime", None, "verified")),
        "premise_unverified": (C(True, -0.6, True, 0.001, 0.001, "holds", "not evaluable", 0.9, 0.3),
                               ("A", None, "unverified")),
    }
    out = {k: {"got": list(g), "want": list(w), "pass": tuple(g) == tuple(w)} for k, (g, w) in rows.items()}
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    LC.gate("early/vl9_known_answers.py")
    import aggregate_pyt as AG
    AG.ensure_a12()
    out = {"plan_item": "ANALYSIS_PLAN_PYTHIA.md section 11 V-P9 (E side, synthetic known answers)", "started_utc": LC.utc()}
    out["1_pattern_table"] = pattern_table(AG)
    out["2_bootstrap"] = bootstrap_checks(AG)
    out["3_gate_holm"] = gate_holm_checks(AG)
    out["4_forecast"] = forecast_checks(AG)
    out["5_categories"] = category_checks(AG)
    out["pass"] = all(out[k]["pass"] for k in out if k[0].isdigit())
    for k in sorted(out):
        if k[0].isdigit():
            print(k, "PASS" if out[k]["pass"] else "FAIL", flush=True)
    out["finished_utc"] = LC.utc()
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    with open(os.path.join(LC.RES, "validation", "VP9_known_answers.json"), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1, default=str)
    print("V-P9 (E, synthetic)", "PASS" if out["pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
