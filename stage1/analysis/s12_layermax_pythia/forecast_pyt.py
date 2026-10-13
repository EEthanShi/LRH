#!/usr/bin/env python
"""Pythia axis: P-P2b, the features-only forecast of the calibrated order of the 357 size comparisons (plan section 9  # PYT-AXIS
B6 and P-P2b; VERDICTS_SPEC_PYT.md section 10 "Section F"), then the plan section 7 category, modifiers and readings  # PYT-AXIS
(VERDICTS_SPEC_PYT.md section 10 "Section K").  # PYT-AXIS
PYT-AXIS copy of analysis/s12_layermax_openllama/forecast_ol.py (sha256 aff9a505...), written at S0b by role  # PYT-AXIS
pyt-code-infra (2026-10-10); changed lines carry PYT-AXIS; diff: forecast_pyt.py.diff.  # PYT-AXIS

GATED (plan F1(a)): the first action after argument parsing is guard.require_partb_frozen(out_root); before
it, nothing but STATUS.md is read. It exits 3 until the `[pyt-axis-predict] Part B FROZEN` STATUS line exists  # PYT-AXIS
and the two Part B files hash to the values on it.

Inputs (read-only):
  results/S1_2_layermax_pythia/partB/partB_features_pyt.json   Bbar per pair, band_i per model (and its h curve and  # PYT-AXIS
                                                                 the flat band), the uncensored band, the extrapolation  # PYT-AXIS
                                                                 flags, membership, materiality curves  # PYT-AXIS
  pair files of the implementation (layer-max = protocol full, biased CKA, both pre):
    E: results/S1_2_layermax_pythia/early/pairs/<pre>/<lm>__<vit>.json   (E summaries: s_obs, c_alpha, s_cal,  # PYT-AXIS
                                                                          null_mean, uncensored)  # PYT-AXIS
    I: results/S1_2_layermax_pythia/indep/<pre>/pairs/<lm>__<vit>.npz    (obs_cka_b, null_cka_b; layer-max  # PYT-AXIS
                                                                          summarised with hrhv2.calibrate.summarize,  # PYT-AXIS
                                                                          as the I aggregate does)  # PYT-AXIS
  the implementation's aggregate verdict file (sections C, H, R, G; read for section K only):  # PYT-AXIS
    E: results/S1_2_layermax_pythia/early/verdicts_pyt.json; I: results/S1_2_layermax_pythia/indep/verdicts_pyt.json  # PYT-AXIS
  Only the observed layer-max T, s_cal and null mean of the 7 Pythia models are taken from the pair files.  # PYT-AXIS

Rule F (spec section 10): for P in {prh_clip+l2, l2}, x in {s_cal, unc}, every comparison (j < j', v):  # PYT-AXIS
  s_cal: S(T; c) = max((T - c)/(1 - c), 0); "below" iff S(T_j'; Bbar_j' + o_lo,j') < S(T_j; Bbar_j + o_hi,j); "above"  # PYT-AXIS
         iff S(T_j'; Bbar_j' + o_hi,j') > S(T_j; Bbar_j + o_lo,j); "tie" iff both are 0 at every offset in the bands;  # PYT-AXIS
         else "undecided".  unc: the same with T - Bbar - u and the uncensored band (no ties).  # PYT-AXIS
  obs from this implementation's s_cal / uncensored; ok = (dec == obs), null if undecided; verdict "fails" if any  # PYT-AXIS
  decided comparison is wrong, else "not evaluable" if fewer than 30 are decided, else "holds"; h curve and flat band.  # PYT-AXIS
Rule K (spec section 10): the category of plan section 7 from H1 membership, D1, p_perm, p_mag, the premise rules  # PYT-AXIS
  (P-P2a from the aggregate, P-P2b from F), R and C; modifiers M1-M5; the H2/H3/H4 readings; knife edge; curves.  # PYT-AXIS

Outputs (out_root = results/S1_2_layermax_pythia/forecast/<impl>/):  # PYT-AXIS
  verdicts_forecast.json  {"schema": "PYT-VERDICTS v1", "impl": "E"|"I", "inputs": {path: sha256}, "v": {...}}  # PYT-AXIS
                          with every key of verdict_keys.required_keys("FK")  # PYT-AXIS
  forecast_detail.json    per-comparison table, boundaries used, the K inputs  # PYT-AXIS

RUN (step S8, after the aggregates of E and I and V-P5; CPU, 1 thread, seconds):  # PYT-AXIS
  cd ~/hrh_v2_stage1_20261008
  analysis/s12_layermax_pythia/launch_pyt.sh forecast_E 1 -- venv/bin/python analysis/s12_layermax_pythia/forecast_pyt.py --impl E  # PYT-AXIS
  analysis/s12_layermax_pythia/launch_pyt.sh forecast_I 1 -- venv/bin/python analysis/s12_layermax_pythia/forecast_pyt.py --impl I  # PYT-AXIS
  venv/bin/python analysis/s12_layermax_pythia/compare_verdicts.py results/S1_2_layermax_pythia/forecast/E/verdicts_forecast.json results/S1_2_layermax_pythia/forecast/I/verdicts_forecast.json --sections FK  # PYT-AXIS

S0B TEST MODE (PYT_S0B_TEST=1 before CODE FROZEN only): --test-root DIR replaces the Pythia pair roots  # PYT-AXIS
(DIR/early/pairs/<pre>/ and DIR/indep/<pre>/pairs/), --partb PATH the Part B JSON, --verdicts PATH the aggregate  # PYT-AXIS
verdict file, --out DIR the output root (default <guard.SCRATCH>/PB/forecast/<impl>); file identity checks are skipped  # PYT-AXIS
because the stand-in files are relabelled known files. These options are refused outside test mode.  # PYT-AXIS
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_pythia")  # PYT-AXIS
sys.path.insert(0, PROJ)
sys.path.insert(0, AXIS)

import guard  # noqa: E402
import verdict_keys as VK  # noqa: E402  # PYT-AXIS: names only

RES_P = os.path.join(PROJ, "results", "S1_2_layermax_pythia")  # PYT-AXIS
OUT_ROOT = os.path.join(RES_P, "forecast")  # PYT-AXIS
PARTB_JSON = os.path.join(RES_P, "partB", "partB_features_pyt.json")  # PYT-AXIS
PRES = ("prh_clip+l2", "l2")
XS = ("s_cal", "unc")
HS = VK.HS  # PYT-AXIS
TAUS, DBS = VK.TAUS, VK.DBS  # PYT-AXIS
MIN_DECIDED = 30  # PYT-AXIS: plan section 9 P-P2b
ALPHA = 0.05  # PYT-AXIS
TAU = 0.10  # PYT-AXIS: plan section 7 category threshold
SWAP = 0.0021  # PYT-AXIS: plan sections 4 F2 and 5.6 (one adjacent swap at 17 ViTs)
A_FAMILY = ("A", "A-att", "A-prime")  # PYT-AXIS
# pair-file roots of the implementations (plan section 3; VERDICTS_SPEC_PYT section 1 "Sources")  # PYT-AXIS
ROOTS = {"E": os.path.join(RES_P, "early", "pairs", "{pre}"), "I": os.path.join(RES_P, "indep", "{pre}", "pairs")}  # PYT-AXIS
AGG = {"E": os.path.join(RES_P, "early", "verdicts_pyt.json"), "I": os.path.join(RES_P, "indep", "verdicts_pyt.json")}  # PYT-AXIS
EXT = {"E": ".json", "I": ".npz"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fmt(v: float) -> str:
    return repr(float(v))


def cp95(x: int, n: int):
    """VERDICTS_SPEC_PYT section 12 Clopper-Pearson 95%."""  # PYT-AXIS
    if n == 0:
        return None, None
    from scipy.stats import beta
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    return lo, hi


def pair_path(impl: str, pre: str, lm: str, vit: str, test_root: str | None = None) -> str:  # PYT-AXIS
    if test_root is not None:  # PYT-AXIS
        base = os.path.join(test_root, "early", "pairs", pre) if impl == "E" else \
            os.path.join(test_root, "indep", pre, "pairs")
    else:
        base = ROOTS[impl].format(pre=pre)  # PYT-AXIS
    return os.path.join(base, f"{lm}__{vit}{EXT[impl]}")


def layermax(impl: str, path: str, pre: str, lm: str, vit: str, check_identity: bool) -> dict:
    """Layer-max (protocol full) summary for biased CKA: T, c_alpha, s_cal, null_mean, unc."""
    if impl == "E":
        with open(path) as f:
            js = json.load(f)
        if check_identity:
            assert js["lm"] == lm and js["vit"] == vit and js["pre"] == pre, path
        s = js["summaries"]["cka_biased"]["full"]
        return {"T": float(s["s_obs"]), "c_alpha": float(s["c_alpha"]), "s_cal": float(s["s_cal"]),
                "null_mean": float(s["null_mean"]), "unc": float(s["uncensored"])}
    import numpy as np
    from hrhv2.calibrate import summarize
    z = np.load(path, allow_pickle=False)
    if check_identity:
        assert str(z["key"]) == f"S1.2-layermax|{lm}|{vit}", (path, str(z["key"]))
    obs, null = z["obs_cka_b"], z["null_cka_b"]
    K = null.shape[0]
    T = float(obs.max())
    sm = summarize(T, null.reshape(K, -1).max(axis=1))
    return {"T": float(sm["s_obs"]), "c_alpha": float(sm["c_alpha"]), "s_cal": float(sm["s_cal"]),
            "null_mean": float(sm["null_mean"]), "unc": float(sm["uncensored"])}


def resolve(args, what: str) -> tuple[bool, str, str | None, str, str]:  # PYT-AXIS
    """Test mode, out_root, test_root, Part B JSON path, aggregate verdicts. Only STATUS.md is read here."""  # PYT-AXIS
    test = guard.test_mode()
    if not test and (args.test_root or args.partb or args.out or args.verdicts):  # PYT-AXIS
        print("--test-root/--partb/--verdicts/--out are S0b test options (PYT_S0B_TEST=1 before CODE FROZEN)",  # PYT-AXIS
              file=sys.stderr)  # PYT-AXIS
        sys.exit(2)
    if test:
        out_root = args.out or os.path.join(guard.SCRATCH, "PB", what, args.impl)
        return (True, out_root, args.test_root,  # PYT-AXIS
                args.partb or os.path.join(guard.SCRATCH, "PB", "partB_vp9", "partB_features_pyt.json"),  # PYT-AXIS
                args.verdicts or os.path.join(guard.SCRATCH, args.impl, "verdicts_pyt.json"))  # PYT-AXIS
    return False, os.path.join(RES_P, what, args.impl), None, PARTB_JSON, AGG[args.impl]  # PYT-AXIS


def obs_order(xj2: float, xj: float) -> str:  # PYT-AXIS: j' (larger model) against j
    return "below" if xj2 < xj else ("above" if xj2 > xj else "tie")  # PYT-AXIS


def s_cal(T: float, c: float) -> float:  # PYT-AXIS
    return max((T - c) / (1.0 - c), 0.0)  # PYT-AXIS


def forecast_scal(Tj: float, Tj2: float, Bj: float, Bj2: float, bj: tuple, bj2: tuple) -> str:  # PYT-AXIS: plan B6
    """j < j' (j' the larger model); b = (o_lo, o_hi) offsets of each model's band."""  # PYT-AXIS
    hi_j2, lo_j2 = s_cal(Tj2, Bj2 + bj2[0]), s_cal(Tj2, Bj2 + bj2[1])  # PYT-AXIS: largest / smallest possible
    hi_j, lo_j = s_cal(Tj, Bj + bj[0]), s_cal(Tj, Bj + bj[1])  # PYT-AXIS
    if hi_j2 < lo_j:  # PYT-AXIS
        return "below"  # PYT-AXIS
    if lo_j2 > hi_j:  # PYT-AXIS
        return "above"  # PYT-AXIS
    if hi_j2 == 0.0 and hi_j == 0.0:  # PYT-AXIS: both 0 at every offset in the bands
        return "tie"  # PYT-AXIS
    return "undecided"  # PYT-AXIS


def forecast_unc(Tj: float, Tj2: float, Bj: float, Bj2: float, mu: tuple) -> str:  # PYT-AXIS: uncensored (secondary)
    hi_j2, lo_j2 = Tj2 - Bj2 - mu[0], Tj2 - Bj2 - mu[1]  # PYT-AXIS
    hi_j, lo_j = Tj - Bj - mu[0], Tj - Bj - mu[1]  # PYT-AXIS
    if hi_j2 < lo_j:  # PYT-AXIS
        return "below"  # PYT-AXIS
    if lo_j2 > hi_j:  # PYT-AXIS
        return "above"  # PYT-AXIS
    return "undecided"  # PYT-AXIS


def verdict_pp2b(n_dec: int, n_wrong: int) -> str:  # PYT-AXIS: plan section 9
    return "fails" if n_wrong > 0 else ("not evaluable" if n_dec < MIN_DECIDED else "holds")  # PYT-AXIS


def section_F(pre: str, x: str, vals: dict, Q: dict, sizes: list, pyt: list, vits: list) -> tuple[dict, dict]:  # PYT-AXIS
    """Section F keys for one (pre, x); vals[(size, vit)] = layer-max dict; Q = Part B per_pre[pre]."""  # PYT-AXIS
    V, rows = {}, []  # PYT-AXIS
    nd = xc = nu = nt = nflip = 0  # PYT-AXIS
    wrong_sizes, wrong_all_ex = set(), True  # PYT-AXIS
    ex = Q["extrapolated"]  # PYT-AXIS
    for ia, a in enumerate(sizes):  # PYT-AXIS
        for b in sizes[ia + 1:]:  # PYT-AXIS
            la, lb = pyt[sizes.index(a)], pyt[sizes.index(b)]  # PYT-AXIS
            for v in vits:  # PYT-AXIS
                Ta, Tb = vals[(a, v)]["T"], vals[(b, v)]["T"]  # PYT-AXIS
                Ba, Bb = Q["Bbar"][la][v], Q["Bbar"][lb][v]  # PYT-AXIS
                if x == "s_cal":  # PYT-AXIS
                    oa = (Q["band"][a]["lo"], Q["band"][a]["hi"])  # PYT-AXIS
                    ob = (Q["band"][b]["lo"], Q["band"][b]["hi"])  # PYT-AXIS
                    dec = forecast_scal(Ta, Tb, Ba, Bb, oa, ob)  # PYT-AXIS
                else:  # PYT-AXIS
                    dec = forecast_unc(Ta, Tb, Ba, Bb, tuple(Q["mu_band"]))  # PYT-AXIS
                obs = obs_order(vals[(b, v)][x], vals[(a, v)][x])  # PYT-AXIS
                ok = None if dec == "undecided" else (dec == obs)  # PYT-AXIS
                if dec == "undecided":  # PYT-AXIS
                    nu += 1  # PYT-AXIS
                else:  # PYT-AXIS
                    nd += 1  # PYT-AXIS
                    xc += int(ok)  # PYT-AXIS
                    nt += int(dec == "tie")  # PYT-AXIS
                    if not ok:  # PYT-AXIS
                        wrong_sizes.update((a, b))  # PYT-AXIS
                        wrong_all_ex = wrong_all_ex and (ex[a] or ex[b])  # PYT-AXIS
                nflip += int(Tb > Ta and dec == "below")  # PYT-AXIS
                V[f"PP2b|{pre}|{x}|dec|{a}|{b}|{v}"] = dec  # PYT-AXIS
                V[f"PP2b|{pre}|{x}|obs|{a}|{b}|{v}"] = obs  # PYT-AXIS
                V[f"PP2b|{pre}|{x}|ok|{a}|{b}|{v}"] = ok  # PYT-AXIS
                rows.append({"a": a, "b": b, "vit": v, "T_a": Ta, "T_b": Tb, "Bbar_a": Ba, "Bbar_b": Bb, "dec": dec,  # PYT-AXIS
                             "x_a": vals[(a, v)][x], "x_b": vals[(b, v)][x], "obs": obs, "ok": ok})  # PYT-AXIS
    lo_, hi_ = cp95(xc, nd)  # PYT-AXIS
    nw = nd - xc  # PYT-AXIS
    V.update({f"PP2b|{pre}|{x}|n_decided": nd, f"PP2b|{pre}|{x}|x_correct": xc, f"PP2b|{pre}|{x}|n_wrong": nw,  # PYT-AXIS
              f"PP2b|{pre}|{x}|n_undecided": nu, f"PP2b|{pre}|{x}|n_tie_fc": nt,  # PYT-AXIS
              f"PP2b|{pre}|{x}|n_flip_fc_rawconc": nflip, f"PP2b|{pre}|{x}|cp_lo": lo_, f"PP2b|{pre}|{x}|cp_hi": hi_,  # PYT-AXIS
              f"PP2b|{pre}|{x}|verdict": verdict_pp2b(nd, nw),  # PYT-AXIS
              f"PP2b|{pre}|{x}|wrong_all_extrap": (None if nw == 0 else bool(wrong_all_ex)),  # PYT-AXIS
              f"PP2b|{pre}|{x}|wrong_sizes": ",".join(s for s in sizes if s in wrong_sizes)})  # PYT-AXIS
    return V, {"rows": rows}  # PYT-AXIS


def curve_counts(vals: dict, Q: dict, sizes: list, pyt: list, vits: list, band_of) -> tuple[int, int]:  # PYT-AXIS
    nd = xc = 0  # PYT-AXIS
    for ia, a in enumerate(sizes):  # PYT-AXIS
        for b in sizes[ia + 1:]:  # PYT-AXIS
            la, lb = pyt[sizes.index(a)], pyt[sizes.index(b)]  # PYT-AXIS
            for v in vits:  # PYT-AXIS
                dec = forecast_scal(vals[(a, v)]["T"], vals[(b, v)]["T"], Q["Bbar"][la][v], Q["Bbar"][lb][v],  # PYT-AXIS
                                    band_of(a), band_of(b))  # PYT-AXIS
                if dec == "undecided":  # PYT-AXIS
                    continue  # PYT-AXIS
                nd += 1  # PYT-AXIS
                xc += int(dec == obs_order(vals[(b, v)]["s_cal"], vals[(a, v)]["s_cal"]))  # PYT-AXIS
    return nd, xc  # PYT-AXIS


# ------------------------------------------------------------------------------------------------- PYT-AXIS
# Section K (plan section 7)                                                                        PYT-AXIS
# ------------------------------------------------------------------------------------------------- PYT-AXIS

def holm(ps: dict) -> dict:  # PYT-AXIS: VERDICTS_SPEC_PYT section 6 (consistency check of the aggregate)
    order = sorted(VK.HYPS, key=lambda h: (ps[h], VK.HYPS.index(h)))  # PYT-AXIS
    rej, adj, run, stop = {}, {}, 0.0, False  # PYT-AXIS
    for i, h in enumerate(order):  # PYT-AXIS
        run = max(run, (3 - i) * ps[h])  # PYT-AXIS
        adj[h] = min(1.0, run)  # PYT-AXIS
        if not stop and ps[h] <= ALPHA / (3 - i):  # PYT-AXIS
            rej[h] = True  # PYT-AXIS
        else:  # PYT-AXIS
            stop = True  # PYT-AXIS
            rej[h] = False  # PYT-AXIS
    return {"order": ",".join(order), "reject": rej, "adj": adj}  # PYT-AXIS


def category(member, D, p_perm, p_mag, confirmed, pp2a, pp2b, R, C, tau=TAU) -> tuple[str, str | None, str | None]:  # PYT-AXIS
    """Plan section 7 table, applied in order. Returns (category, premise, F_reason)."""  # PYT-AXIS
    if not member:  # PYT-AXIS: step 1
        return "E", None, None  # PYT-AXIS
    if D is None:  # PYT-AXIS: no valid ViT (cannot happen with 17 ViTs and a material baseline; flagged)
        return "undefined (no valid ViT)", None, None  # PYT-AXIS
    if D >= tau:  # PYT-AXIS: step 2
        return "D", None, None  # PYT-AXIS
    if abs(D) < tau:  # PYT-AXIS: step 3
        return "C", None, None  # PYT-AXIS
    if not confirmed:  # PYT-AXIS: step 4 (D <= -tau)
        rs = []  # PYT-AXIS
        if p_perm > ALPHA:  # PYT-AXIS
            rs.append("not consistent across the seven sizes")  # PYT-AXIS
        if p_mag > ALPHA:  # PYT-AXIS
            rs.append("size of the drop not established")  # PYT-AXIS
        return "F", None, ("; ".join(rs) if rs else None)  # PYT-AXIS
    if pp2a == "fails" or pp2b == "fails":  # PYT-AXIS: step 5
        return "B", None, None  # PYT-AXIS
    cat = "A-prime" if R < 0.60 else ("A" if C <= 0.40 else "A-att")  # PYT-AXIS: step 6
    premise = "verified" if (pp2a == "holds" and pp2b == "holds") else "unverified"  # PYT-AXIS
    return cat, premise, None  # PYT-AXIS


def reading_h34(D, degenerate, tested, holm_reject) -> str | None:  # PYT-AXIS: plan section 7 immunity readings
    if D is None:  # PYT-AXIS
        return None  # PYT-AXIS
    if abs(D) >= TAU:  # PYT-AXIS
        return "moved"  # PYT-AXIS
    if degenerate:  # PYT-AXIS
        return "no comparison changed order"  # PYT-AXIS
    if not tested:  # PYT-AXIS
        return "kept, not tested (gate closed)"  # PYT-AXIS
    if holm_reject:  # PYT-AXIS
        return "equivalence confirmed"  # PYT-AXIS
    return "kept, equivalence not established"  # PYT-AXIS


def reading_h2(pp5a, branch, tested, confirmed, D2) -> str | None:  # PYT-AXIS: plan section 7 CTM reading
    if pp5a == "fails":  # PYT-AXIS
        return "premise failed"  # PYT-AXIS
    if tested and confirmed and branch == "H2a":  # PYT-AXIS
        return "follows the late baselines"  # PYT-AXIS
    if tested and confirmed and branch == "H2b":  # PYT-AXIS
        return "kept as predicted"  # PYT-AXIS
    if D2 is None:  # PYT-AXIS
        return None  # PYT-AXIS
    base = ("moved with the late baselines" if D2 >= TAU else "kept" if abs(D2) < TAU else "drop not confirmed")  # PYT-AXIS
    return base + ("" if tested else " (not tested, gate closed)")  # PYT-AXIS


def h_consistent(VA: dict, h: str) -> bool:  # PYT-AXIS: the aggregate's H keys agree with the plan's definitions
    g = VA.get  # PYT-AXIS
    try:  # PYT-AXIS
        ok = g(f"{h}|H1|p") == max(g(f"{h}|H1|p_perm"), g(f"{h}|H1|p_mag"))  # PYT-AXIS
        mem, conf = g(f"{h}|H1|member"), g(f"{h}|H1|confirmed")  # PYT-AXIS
        want = (g(f"{h}|H1|p") <= ALPHA and g(f"{h}|H1|D") <= -TAU) if mem else None  # PYT-AXIS
        ok = ok and conf == want  # PYT-AXIS
        gate = ("open (H1 confirmed)" if mem and conf else "open (H1 not a member)" if not mem else "closed")  # PYT-AXIS
        ok = ok and g(f"{h}|gate") == gate  # PYT-AXIS
        tested = gate != "closed"  # PYT-AXIS
        br = g(f"{h}|H2|branch")  # PYT-AXIS
        if br == "H2a":  # PYT-AXIS
            want_h2p = max(g(f"{h}|H2|p_perm"), g(f"{h}|H2|p_mag"))  # PYT-AXIS
        else:  # PYT-AXIS: H2b = TOST of (cka_biased, ctm)
            d2 = _tost_delta(VA, h)  # PYT-AXIS
            want_h2p = max(g(f"{h}|H2|p_lo"), g(f"{h}|H2|p_hi")) if -TAU < d2 < TAU else 1.0  # PYT-AXIS
        ok = ok and g(f"{h}|H2|p") == want_h2p  # PYT-AXIS
        for hy in VK.HYPS:  # PYT-AXIS
            ok = ok and g(f"{h}|{hy}|tested") == tested  # PYT-AXIS
        if tested:  # PYT-AXIS
            H = holm({hy: g(f"{h}|{hy}|p") for hy in VK.HYPS})  # PYT-AXIS
            ok = ok and g(f"{h}|holm_order") == H["order"]  # PYT-AXIS
            for hy in VK.HYPS:  # PYT-AXIS
                ok = ok and g(f"{h}|{hy}|holm_reject") == H["reject"][hy]  # PYT-AXIS
                ok = ok and abs(g(f"{h}|{hy}|p_holm") - H["adj"][hy]) <= 1e-12  # PYT-AXIS
                cf = (H["reject"][hy] and g(f"{h}|H2|D") <= -TAU) if (hy == "H2" and br == "H2a") else H["reject"][hy]  # PYT-AXIS
                ok = ok and g(f"{h}|{hy}|confirmed") == cf  # PYT-AXIS
        else:  # PYT-AXIS
            for hy in VK.HYPS:  # PYT-AXIS
                ok = ok and g(f"{h}|{hy}|p_holm") is None and g(f"{h}|{hy}|holm_reject") is None  # PYT-AXIS
                ok = ok and g(f"{h}|{hy}|confirmed") is None  # PYT-AXIS
        return bool(ok)  # PYT-AXIS
    except (TypeError, KeyError):  # PYT-AXIS: a missing or null input where a number is required
        return False  # PYT-AXIS


def _tost_delta(VA: dict, h: str) -> float:  # PYT-AXIS: the unsigned CTM difference of the (cka_biased, ctm) TOST
    _, p, x = h.split("|")  # PYT-AXIS
    return VA[f"TOST|{p}|{x}|cka_biased|ctm|D"]  # PYT-AXIS


def section_K(pre: str, x: str, VA: dict, VF: dict, Q: dict, sizes: list) -> dict:  # PYT-AXIS
    h = f"H|{pre}|{x}"  # PYT-AXIS
    c = f"K|{pre}|{x}"  # PYT-AXIS
    g = VA.get  # PYT-AXIS
    member, D = g(f"{h}|H1|member"), g(f"{h}|H1|D")  # PYT-AXIS
    p_perm, p_mag, p, conf = g(f"{h}|H1|p_perm"), g(f"{h}|H1|p_mag"), g(f"{h}|H1|p"), g(f"{h}|H1|confirmed")  # PYT-AXIS
    R, Cc = g(f"{h}|H1|R"), g(f"{h}|H1|C")  # PYT-AXIS
    pp2a, pp2b = g(f"PP2a|{pre}|{x}|verdict"), VF.get(f"PP2b|{pre}|{x}|verdict")  # PYT-AXIS
    cat, premise, freason = category(member, D, p_perm, p_mag, conf, pp2a, pp2b, R, Cc)  # PYT-AXIS
    K = {f"{c}|category": cat, f"{c}|premise": premise, f"{c}|F_reason": freason, f"{c}|R": R, f"{c}|C": Cc,  # PYT-AXIS
         f"{c}|nV": g(f"{h}|H1|nV")}  # PYT-AXIS
    r = f"R|{pre}|{x}"  # PYT-AXIS
    K[f"{c}|M1"] = g(f"{r}|fragile")  # PYT-AXIS
    K[f"{c}|M1_sizes"] = g(f"{r}|fragile_sizes")  # PYT-AXIS
    g5 = g(f"{r}|grid5|D1")  # PYT-AXIS
    K[f"{c}|M2"] = (None if cat not in A_FAMILY or g5 is None else bool(not (g5 <= -TAU)))  # PYT-AXIS
    K[f"{c}|M2_x"] = g5  # PYT-AXIS
    rd = {}  # PYT-AXIS
    for hy in ("H3", "H4"):  # PYT-AXIS
        rd[hy] = reading_h34(g(f"{h}|{hy}|D"), g(f"{h}|{hy}|degenerate"), g(f"{h}|{hy}|tested"), g(f"{h}|{hy}|holm_reject"))  # PYT-AXIS
        K[f"{c}|{hy}_reading"] = rd[hy]  # PYT-AXIS
    kept = ("equivalence confirmed", "no comparison changed order", "kept, equivalence not established",  # PYT-AXIS
            "kept, not tested (gate closed)")  # PYT-AXIS
    for hy, key in (("H3", "dtr_mknn"), ("H4", "dtr_dcka")):  # PYT-AXIS
        gv = g(f"{r}|grid5|{key}")  # PYT-AXIS
        K[f"{c}|M2_{hy}"] = (bool(abs(gv) >= TAU) if (rd[hy] in kept and gv is not None) else None)  # PYT-AXIS
    K[f"{c}|M3"] = g(f"{r}|five|carried_D") if cat == "D" else g(f"{r}|five|carried")  # PYT-AXIS
    if x == "s_cal":  # PYT-AXIS: near chance is a property of the censored score [op]
        nz = g(f"R|{pre}|n_scal_zero")  # PYT-AXIS
        zs = [s for s in sizes if g(f"R|{pre}|median_scal|{s}") == 0.0]  # PYT-AXIS
        K[f"{c}|M4"] = (None if nz is None else bool(nz > 119 / 3 or zs))  # PYT-AXIS
        K[f"{c}|M4_sizes"] = ",".join(zs)  # PYT-AXIS
        K[f"{c}|M4_n_zero"] = nz  # PYT-AXIS
    else:  # PYT-AXIS
        K[f"{c}|M4"], K[f"{c}|M4_sizes"], K[f"{c}|M4_n_zero"] = None, None, None  # PYT-AXIS
    ex = [s for s in sizes if Q["extrapolated"][s]]  # PYT-AXIS
    K[f"{c}|M5_extrap_sizes"] = ",".join(ex)  # PYT-AXIS
    if cat == "B":  # PYT-AXIS [op] VERDICTS_SPEC_PYT section 10
        failing = [s for s in (g(f"PP2ai|{pre}|{x}|failing") or "").split(",") if s]  # PYT-AXIS
        K[f"{c}|M5_B"] = bool(g(f"PP2aii|{pre}|{x}|verdict") != "fails" and all(s in ex for s in failing)  # PYT-AXIS
                              and VF.get(f"PP2b|{pre}|{x}|wrong_all_extrap") is not False)  # PYT-AXIS
    else:  # PYT-AXIS
        K[f"{c}|M5_B"] = None  # PYT-AXIS
    if cat in A_FAMILY:  # PYT-AXIS
        held = [s for s in ex if g(f"PP2ai|{pre}|{x}|in_band|{s}")]  # PYT-AXIS
        K[f"{c}|M5_A"], K[f"{c}|M5_A_sizes"] = bool(held), ",".join(held)  # PYT-AXIS
    else:  # PYT-AXIS
        K[f"{c}|M5_A"], K[f"{c}|M5_A_sizes"] = None, None  # PYT-AXIS
    br, tested2, conf2, D2 = g(f"{h}|H2|branch"), g(f"{h}|H2|tested"), g(f"{h}|H2|confirmed"), g(f"{h}|H2|D")  # PYT-AXIS
    K[f"{c}|H2_reading"] = reading_h2(g(f"PP5a|{pre}|{x}|verdict"), br, tested2, conf2, D2)  # PYT-AXIS
    K[f"{c}|H2_same_as_H1"] = bool(Q["H2_same_as_H1"])  # PYT-AXIS
    K[f"{c}|H2_counted"] = bool(conf2) and not Q["H2_same_as_H1"]  # PYT-AXIS
    K[f"{c}|H3_counted"] = rd["H3"] == "equivalence confirmed"  # PYT-AXIS
    K[f"{c}|H4_counted"] = rd["H4"] == "equivalence confirmed"  # PYT-AXIS
    which = []  # PYT-AXIS
    if g(f"{h}|H1|kedge"):  # PYT-AXIS
        which.append("D1")  # PYT-AXIS
    if cat in A_FAMILY and R is not None and abs(R - 0.60) <= SWAP + 1e-9:  # PYT-AXIS
        which.append("R")  # PYT-AXIS
    if cat in A_FAMILY and Cc is not None and abs(Cc - 0.40) <= SWAP + 1e-9:  # PYT-AXIS
        which.append("C")  # PYT-AXIS
    K[f"{c}|kedge"] = bool(which)  # PYT-AXIS
    K[f"{c}|kedge_which"] = ",".join(which)  # PYT-AXIS
    K[f"{c}|h_consistent"] = h_consistent(VA, h)  # PYT-AXIS
    for t in TAUS:  # PYT-AXIS: tau curve (p_H1 unchanged) [op]
        conf_t = (member and p is not None and D is not None and p <= ALPHA and D <= -t) if member else None  # PYT-AXIS
        K[f"{c}|curve|tau={fmt(t)}"] = category(member, D, p_perm, p_mag, conf_t, pp2a, pp2b, R, Cc, tau=t)[0]  # PYT-AXIS
    for db in DBS:  # PYT-AXIS: delta_b curve (membership from Part B)
        mem_d = bool(Q["material_curve_db"]["Bbar"][fmt(db)])  # PYT-AXIS
        conf_d = (p is not None and D is not None and p <= ALPHA and D <= -TAU) if mem_d else None  # PYT-AXIS
        K[f"{c}|curve|db={fmt(db)}"] = category(mem_d, D, p_perm, p_mag, conf_d, pp2a, pp2b, R, Cc)[0]  # PYT-AXIS
    return K  # PYT-AXIS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=("E", "I"), required=True)
    ap.add_argument("--test-root", default=None, help="S0b test mode only")
    ap.add_argument("--partb", default=None, help="S0b test mode only")
    ap.add_argument("--verdicts", default=None, help="S0b test mode only")  # PYT-AXIS
    ap.add_argument("--out", default=None, help="S0b test mode only")
    args = ap.parse_args()
    test, out_root, test_root, partb, agg = resolve(args, "forecast")  # PYT-AXIS
    gate = guard.require_partb_frozen(out_root, script="forecast_pyt.py")      # F1(a): FIRST data-related action  # PYT-AXIS
    # ---- nothing involving a pair file, score or Part B value is opened before this line ----
    import numpy as np  # noqa: F401  (after the gate)

    F = json.load(open(partb))
    pyt, sizes, vits = list(F["pyt_lms"]), list(F["sizes"]), list(F["vits"])  # PYT-AXIS
    assert sizes == list(VK.SIZES) and vits == list(VK.VITS), "sizes or ViT order differ from verdict_keys"  # PYT-AXIS
    if not test:
        assert pyt == [VK.PYT_SLUGS[s] for s in VK.SIZES], pyt  # PYT-AXIS
    VA_doc = json.load(open(agg))  # PYT-AXIS
    VA = VA_doc["v"]  # PYT-AXIS
    inputs = {os.path.relpath(partb, PROJ): sha256_file(partb), os.path.relpath(agg, PROJ): sha256_file(agg)}  # PYT-AXIS
    l2_run = bool(VA.get("G|l2_run"))  # PYT-AXIS
    pres_run = [p for p in PRES if p == "prh_clip+l2" or l2_run]  # PYT-AXIS
    vals = {}
    for pre in pres_run:  # PYT-AXIS
        for s, lm in zip(sizes, pyt):  # PYT-AXIS
            for v in vits:
                p = pair_path(args.impl, pre, lm, v, test_root)  # PYT-AXIS
                vals[(pre, s, v)] = layermax(args.impl, p, pre, lm, v, check_identity=not test)  # PYT-AXIS
                inputs[os.path.relpath(p, PROJ)] = sha256_file(p)
    V = {}
    detail = {"impl": args.impl, "gate": gate, "partB": os.path.relpath(partb, PROJ), "aggregate": os.path.relpath(agg, PROJ),  # PYT-AXIS
              "pyt_lms": pyt, "l2_run": l2_run, "per_pre": {}}  # PYT-AXIS
    for pre in PRES:
        if pre not in pres_run:  # PYT-AXIS: VERDICTS_SPEC_PYT section 9: every l2 key null when l2 was not run
            for k in VK.required_keys("FK"):  # PYT-AXIS
                if k.split("|")[1] == pre:  # PYT-AXIS
                    V[k] = None  # PYT-AXIS
            continue  # PYT-AXIS
        Q = dict(F["per_pre"][pre], mu_band=list(F["mu_band"]))  # PYT-AXIS: mu_band is top-level (spec section 11)
        vp = {(s, v): vals[(pre, s, v)] for s in sizes for v in vits}  # PYT-AXIS
        dp = {"realised_offsets_calpha_minus_Bbar": {s: {v: vp[(s, v)]["c_alpha"] - Q["Bbar"][lm][v] for v in vits}  # PYT-AXIS
                                                     for s, lm in zip(sizes, pyt)},  # PYT-AXIS
              "band": Q["band"], "mu_band": Q["mu_band"], "x": {}}  # PYT-AXIS
        for x in XS:  # PYT-AXIS
            Vx, dx = section_F(pre, x, vp, Q, sizes, pyt, vits)  # PYT-AXIS
            V.update(Vx)  # PYT-AXIS
            dp["x"][x] = dx  # PYT-AXIS
        hc = {}
        for h in HS:  # PYT-AXIS: h curve (s_cal), band f(PR*_i) +- h
            bh = Q["band_h_curve"][fmt(h)]  # PYT-AXIS
            nd, xc = curve_counts(vp, Q, sizes, pyt, vits, lambda s, bh=bh: (bh[s][0], bh[s][1]))  # PYT-AXIS
            V[f"PP2b|{pre}|s_cal|h={fmt(h)}|n_decided"] = nd  # PYT-AXIS
            V[f"PP2b|{pre}|s_cal|h={fmt(h)}|x_correct"] = xc  # PYT-AXIS
            hc[fmt(h)] = {"n_decided": nd, "x_correct": xc}  # PYT-AXIS
        fl = tuple(Q["band_flat"])  # PYT-AXIS: flat-band sensitivity
        nd, xc = curve_counts(vp, Q, sizes, pyt, vits, lambda s: fl)  # PYT-AXIS
        V[f"PP2b|{pre}|s_cal|flat|n_decided"] = nd  # PYT-AXIS
        V[f"PP2b|{pre}|s_cal|flat|x_correct"] = xc  # PYT-AXIS
        V[f"PP2b|{pre}|s_cal|flat|verdict"] = verdict_pp2b(nd, nd - xc)  # PYT-AXIS
        dp["h_curve_s_cal"] = hc
        dp["flat"] = {"band": list(fl), "n_decided": nd, "x_correct": xc}  # PYT-AXIS
        for x in XS:  # PYT-AXIS: section K after F
            V.update(section_K(pre, x, VA, V, Q, sizes))  # PYT-AXIS
        detail["per_pre"][pre] = dp
    miss = [k for k in VK.required_keys("FK") if k not in V]  # PYT-AXIS
    if miss:
        raise SystemExit(f"internal error: {len(miss)} required FK keys missing, e.g. {miss[:3]}")  # PYT-AXIS
    os.makedirs(out_root, exist_ok=True)
    for fn in ("verdicts_forecast.json", "forecast_detail.json"):  # PYT-AXIS: no-delete rule
        p = os.path.join(out_root, fn)  # PYT-AXIS
        if os.path.exists(p):  # PYT-AXIS
            if not test:  # PYT-AXIS
                raise SystemExit(f"{p} exists (no-delete rule; a rerun needs a logged deviation and a new out_root)")  # PYT-AXIS
            os.replace(p, p + ".prev-" + str(os.getpid()))  # PYT-AXIS
    doc = {"schema": "PYT-VERDICTS v1", "impl": args.impl, "inputs": inputs, "v": V}  # PYT-AXIS
    with open(os.path.join(out_root, "verdicts_forecast.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    with open(os.path.join(out_root, "forecast_detail.json"), "w") as f:
        json.dump(detail, f, indent=1, default=float)
    for pre in pres_run:  # PYT-AXIS: counts only; the category is read from the file after V-P5 (plan section 11)
        for x in XS:  # PYT-AXIS
            print(f"{args.impl} {pre} {x}: P-P2b n_decided {V[f'PP2b|{pre}|{x}|n_decided']} undecided "  # PYT-AXIS
                  f"{V[f'PP2b|{pre}|{x}|n_undecided']}", flush=True)  # PYT-AXIS
    print(f"wrote {out_root}/verdicts_forecast.json, forecast_detail.json (sections F and K; values not printed)", flush=True)  # PYT-AXIS


if __name__ == "__main__":
    main()
