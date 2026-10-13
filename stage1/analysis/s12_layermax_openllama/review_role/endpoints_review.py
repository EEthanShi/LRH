#!/usr/bin/env python
"""OpenLLaMA axis, role oll-review: the reviewer's own recomputation of the plan endpoints from the summary files
(plan ANALYSIS_PLAN_OPENLLAMA.md 4f18d710, sections 5, 7, 9, 10; VERDICTS_SPEC.md). NEW read-only helper of the
review role, written after the run roles finished (not in the S0b manifest). It does not import any E, I, R,
forecast or joint code. Inputs (hashes checked against STATUS):
  - indep/summary.json (per-pair values of all 51 OpenLLaMA pairs; the 11-LM pooled grids),
  - early/pairs/<pre>/*.json (E per-pair summaries; used to cross-check the per-pair values),
  - partB/partB_features_ol.json (frozen Part B: B-bar, B_late, boundaries, s_O, materiality),
  - the E, I, forecast and joint verdict files (only to compare with at the end).
hrhv2.util.rng_from_key is used only to reproduce the pinned bootstrap keys of plan section 5.3.
Writes results/S1_2_layermax_openllama/review/role/endpoints_review.json (chmod 444).

usage: venv/bin/python -I analysis/s12_layermax_openllama/review_role/endpoints_review.py
exit 0 iff every recomputed endpoint equals the E and I values.
"""
import hashlib
import itertools
import json
import math
import os
import sys
import time

import numpy as np
from scipy.stats import beta, rankdata

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
sys.path.insert(0, PROJ)
from hrhv2 import util  # noqa: E402  (rng_from_key only)

RES = os.path.join(PROJ, "results", "S1_2_layermax_openllama")
OUT = os.path.join(RES, "review", "role")
LOGGED = {
    "indep/summary.json": "7b60cf8c8ca646a263da658220f3f3eaf7477dc28b96e18b5df95d6bd7dd0fd9",
    "partB/partB_features_ol.json": "e1104bd71e5eff9ced65ed402a61c187590bce5dee306c25ee2f64b18fdd7e51",
    "early/verdicts_ol.json": "0c7f34b59937a6ed6ed154570126c76f726d559f8962c3e2fc1d9aa8f0bf1f8f",
    "indep/verdicts_ol.json": "58fcc80b6ffca7cecc44cf1951e0bbf6d6e611fe6305b28c4a04ade78ade6967",
    "forecast/E/verdicts_forecast.json": "54055939adfc22dd9bf0e89fe3fa138b4b143522a786e5db50ff7bb38ba82263",
    "forecast/I/verdicts_forecast.json": "87fc0ab8aa80cf6f2620329a8e159524daa836961889f893ec1b294856097b36",
    "joint/E/verdicts_joint.json": "be2e6ee59f5187d7665bae9e52b3248a66ba5a6bada6a8d805831517d888fe5e",
    "joint/I/verdicts_joint.json": "204ed4acab9ed39c143bd668de8d8a827c6ca08a4b1e2b71060694dc777ec00a",
}
PRES = ("prh_clip+l2", "l2")
SH = {"openlm-research__open_llama_3b": "3b", "openlm-research__open_llama_7b": "7b",
      "openlm-research__open_llama_13b": "13b"}
EPS = 1e-9


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load(rel):
    p = os.path.join(RES, rel)
    got = sha(p)
    if got != LOGGED[rel]:
        sys.exit(f"hash mismatch {rel}: {got}")
    return json.load(open(p))


def cp(x, n):
    if n == 0:
        return [None, None]
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    return [lo, hi]


def rho(lab, s):
    s = np.asarray(s, float)
    if not np.all(np.isfinite(s)) or np.all(s == s[0]):
        return None
    a, b = rankdata(lab), rankdata(s)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / math.sqrt((a @ a) * (b @ b)))


def tr(rows, lab=(1, 2, 3)):
    v = [rho(lab, r) for r in rows]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None, len(rows) - len(v)


PERM3 = list(itertools.permutations((1, 2, 3)))


def trend_p(rows):
    obs = tr(rows)[0]
    al = [tr(rows, s)[0] for s in PERM3]
    return obs, sum(a <= obs + EPS for a in al) / 6, sum(a >= obs - EPS for a in al) / 6


def dtrend_p(rows_x, rows_raw, sizes=None):
    f = (lambda s: tr(rows_x, s)[0] - tr(rows_raw, s)[0])
    obs = f((1, 2, 3))
    al = [f(s) for s in PERM3]
    return obs, sum(a <= obs + EPS for a in al) / 6


def pear(G, x):
    xx = np.repeat(np.asarray(x, float)[:, None], G.shape[1], axis=1).ravel()
    return float(np.corrcoef(xx, G.ravel())[0, 1])


def main():
    t0 = time.time()
    I = load("indep/summary.json")
    PBd = load("partB/partB_features_ol.json")
    VE = load("early/verdicts_ol.json")["v"]
    VI = load("indep/verdicts_ol.json")["v"]
    FE = load("forecast/E/verdicts_forecast.json")["v"]
    FI = load("forecast/I/verdicts_forecast.json")["v"]
    JE = load("joint/E/verdicts_joint.json")["v"]
    JI = load("joint/I/verdicts_joint.json")["v"]
    LMS, VITS = I["ol_order"], I["vit_order"]
    assert [SH[x] for x in LMS] == ["3b", "7b", "13b"] and len(VITS) == 17
    out, cmp_rows = {"inputs": {k: v for k, v in LOGGED.items()}}, []

    def mine(key, val, ref_keys=None, src="EI"):
        out.setdefault("values", {})[key] = val
        rk = ref_keys or key
        refs = {"EI": (VE, VI), "F": (FE, FI), "J": (JE, JI)}[src]
        for name, D in zip(("E", "I"), refs):
            r = D.get(rk, "MISSING")
            if isinstance(val, float) and isinstance(r, (int, float)) and not isinstance(r, bool):
                ok = abs(val - r) <= 1e-9 * max(1.0, abs(r))
            else:
                ok = val == r
            cmp_rows.append({"key": rk, "mine": val, name: r, "impl": name, "equal": ok})

    def V(P, lm, vit, M, Q, k):
        return I["pairs_ol"][P][f"{lm}|{vit}"][M][Q][k]

    # ---- 0. per-pair cross-check: I summary vs E pair json --------------------------------------------------
    emap = {"raw": "s_obs", "c": "c_alpha", "s_cal": "s_cal", "unc": "uncensored", "mu": "null_mean"}
    cmap = {"M": "M", "M_unc": "M_unc", "TM": "s_obs_at_argmax", "cM": "c_alpha_at_argmax"}
    worst, nval = {}, 0
    for P in PRES:
        for lm in LMS:
            for vit in VITS:
                e = json.load(open(os.path.join(RES, "early", "pairs", P, f"{lm}__{vit}.json")))
                for M in ("mknn", "cka_biased", "cka_debiased"):
                    for Q in ("full", "fixed_r0.5", "fixed_r1.0"):
                        for k, ek in emap.items():
                            a, b = V(P, lm, vit, M, Q, k), e["summaries"][M][Q][ek]
                            d = abs(a - b) / max(abs(a), abs(b), 1e-300) if a != b else 0.0
                            worst[k] = max(worst.get(k, 0.0), d)
                            nval += 1
                    for k, ek in cmap.items():
                        a, b = V(P, lm, vit, M, "ctm", k), e["calib_then_max"][M][ek]
                        d = abs(a - b) / max(abs(a), abs(b), 1e-300) if a != b else 0.0
                        worst["ctm_" + k] = max(worst.get("ctm_" + k, 0.0), d)
                        nval += 1
    out["pair_values_I_summary_vs_E_json"] = {
        "n": nval, "max_rel_by_field": worst,
        "note": "null mean ('mu') is reported only; R noted float32-rounded null means in some E json files",
        "pass": all(v <= 1e-12 for k, v in worst.items() if k != "mu")}

    for P in PRES:
        pp = PBd["per_pre"][P]
        sO, mat = pp["s_O"], pp["material"]

        def rows(M, Q, k):
            return [[V(P, lm, vit, M, Q, k) for lm in LMS] for vit in VITS]

        # ---- means by size (descriptive) ----
        means = {}
        for k in ("raw", "c", "s_cal", "unc"):
            means[k] = [float(np.mean([V(P, lm, v, "cka_biased", "full", k) for v in VITS])) for lm in LMS]
        means["Bbar"] = [float(np.mean([pp["Bbar"][lm][v] for v in VITS])) for lm in LMS]
        out.setdefault("means", {})[P] = means

        # ---- trends (section A) for the decision metrics ----
        for M in ("mknn", "cka_biased", "cka_debiased"):
            R0 = rows(M, "full", "raw")
            t, ple, pge = trend_p(R0)
            mine(f"tr|{P}|{M}|full|raw", t)
            for X in ("s_cal", "unc"):
                RX = rows(M, "full", X)
                mine(f"tr|{P}|{M}|full|{X}", tr(RX)[0])
                d, dple = dtrend_p(RX, R0)
                mine(f"dtr|{P}|{M}|full|{X}", d)
                mine(f"dtr_ple|{P}|{M}|full|{X}", dple)
            for X in ("M", "M_unc", "TM"):
                RX = rows(M, "ctm", X)
                d, dple = dtrend_p(RX, R0)
                mine(f"dtr|{P}|{M}|ctm|{X}", d)
        # ---- P-O2 (biased CKA, full) ----
        for X in ("s_cal", "unc"):
            d = out["values"][f"dtr|{P}|cka_biased|full|{X}"]
            traw = out["values"][f"tr|{P}|cka_biased|full|raw"]
            if not mat:
                cat = "no category (not material)"
            elif abs(traw + sO) <= EPS:
                cat = "not testable"
            elif sO * d <= -0.10:
                cat = "loss reproduced"
            elif sO * d >= 0.10:
                cat = "moved with the baseline"
            else:
                cat = "trend kept"
            mine(f"PO2|{P}|{X}|cat", cat)
            # F_new and pilot flips
            down = up = 0
            pdown = pup = 0
            for vit in VITS:
                r = [V(P, lm, vit, "cka_biased", "full", "raw") for lm in LMS]
                x = [V(P, lm, vit, "cka_biased", "full", X) for lm in LMS]
                for a in (0, 1):
                    dr, dx = np.sign(r[2] - r[a]), np.sign(x[2] - x[a])
                    down += dx < dr
                    up += dx > dr
                dr, dx = np.sign(r[1] - r[0]), np.sign(x[1] - x[0])
                pdown += dx < dr
                pup += dx > dr
            fm, fp = (down, up) if sO == 1 else (up, down)
            mine(f"chg|{P}|cka_biased|full|{X}|new|down", int(down))
            mine(f"chg|{P}|cka_biased|full|{X}|new|up", int(up))
            mine(f"chg|{P}|cka_biased|full|{X}|pilot|down", int(pdown))
            mine(f"fnew|{P}|cka_biased|full|{X}|F", (fm - fp) / 34)
        # ---- P-O2a(i), (ii) ----
        L13 = LMS[2]
        offs = [V(P, L13, v, "cka_biased", "full", "c") - pp["Bbar"][L13][v] for v in VITS]
        med = float(np.median(offs))
        mine(f"PO2ai|{P}|median", med)
        mine(f"PO2ai|{P}|in_band", bool(0.0009 <= med <= 0.0029))
        mine(f"PO2ai|{P}|x", int(sum(0.0009 <= o <= 0.0029 for o in offs)))
        out.setdefault("PO2ai_range", {})[P] = [min(offs), max(offs)]
        n = x = 0
        for v in VITS:
            for a in LMS[:2]:
                gB = pp["Bbar"][L13][v] - pp["Bbar"][a][v]
                if abs(gB) >= 0.005:
                    n += 1
                    gc = V(P, L13, v, "cka_biased", "full", "c") - V(P, a, v, "cka_biased", "full", "c")
                    x += np.sign(gc) == np.sign(gB)
        need = math.ceil(2 * n / 3)
        mine(f"PO2aii|{P}|n", n)
        mine(f"PO2aii|{P}|x", int(x))
        mine(f"PO2aii|{P}|verdict", "not evaluable" if n < 6 else ("holds" if x >= need else "fails"))
        # ---- P-O2b (section F): boundaries from Part B; kappa recomputed from the frozen known values ----
        bd = pp["boundaries"]
        band = pp["band"]
        kmax = 0.0
        for a in ("3b", "7b"):
            for v in VITS:
                Ta, ca = pp["known"]["T"][a][v], pp["known"]["c_alpha"][a][v]
                B13 = pp["Bbar"][L13][v]
                for o, side in ((band["o_lo"], "lo"), (band["o_hi"], "hi")):
                    k = Ta + (1 - Ta) * (B13 + o - ca) / (1 - ca)
                    kmax = max(kmax, abs(k - bd["s_cal"][side][a][v]))
        out.setdefault("kappa_recomputed_vs_partB_maxabs", {})[P] = kmax
        aidx = {"3b": 0, "7b": 1}
        for X, key in (("s_cal", "s_cal"), ("unc", "unc")):
            nd = xc = nu = 0
            nflip = nkeep = 0
            for a in ("3b", "7b"):
                for v in VITS:
                    T13 = V(P, L13, v, "cka_biased", "full", "raw")
                    lo, hi = bd[key]["lo"][a][v], bd[key]["hi"][a][v]
                    if X == "s_cal" and pp["known"]["s_cal"][a][v] == 0:
                        dec = "undecided"
                    else:
                        dec = "below" if T13 < lo else ("above" if T13 > hi else "undecided")
                    xa = V(P, LMS[aidx[a]], v, "cka_biased", "full", X)
                    x13 = V(P, L13, v, "cka_biased", "full", X)
                    obs = "below" if x13 < xa else ("above" if x13 > xa else "tie")
                    Ta_r = V(P, LMS[aidx[a]], v, "cka_biased", "full", "raw")
                    rawo = "raw_below" if T13 < Ta_r else ("raw_above" if T13 > Ta_r else "raw_tie")
                    tab = out.setdefault("PO2b_table", {}).setdefault(f"{P}|{X}", {})
                    kk = f"forecast_{dec}|{rawo}|obs_{obs}"
                    tab[kk] = tab.get(kk, 0) + 1
                    if dec == "undecided":
                        nu += 1
                        continue
                    nd += 1
                    xc += dec == obs
                    nflip += dec == "below" and T13 >= V(P, LMS[aidx[a]], v, "cka_biased", "full", "raw")
                    nkeep += dec == "above"
            mine(f"PO2b|{P}|{X}|n_decided", nd, src="F")
            mine(f"PO2b|{P}|{X}|x_correct", xc, src="F")
            mine(f"PO2b|{P}|{X}|n_undecided", nu, src="F")
            mine(f"PO2b|{P}|{X}|verdict", "fails" if xc < nd else ("not evaluable" if nd < 10 else "holds"), src="F")
            out.setdefault("PO2b_cp", {})[f"{P}|{X}"] = cp(xc, nd)
            out.setdefault("PO2b_forecast_split", {})[f"{P}|{X}"] = {"raw_concordant_forecast_flip": nflip,
                                                                       "forecast_keep": nkeep}
        hc = {}
        for h, hb in bd["s_cal_h"].items():
            nd = xc = 0
            for a in ("3b", "7b"):
                for v in VITS:
                    if pp["known"]["s_cal"][a][v] == 0:
                        continue
                    T13 = V(P, L13, v, "cka_biased", "full", "raw")
                    lo, hi = hb["lo"][a][v], hb["hi"][a][v]
                    dec = "below" if T13 < lo else ("above" if T13 > hi else None)
                    if dec is None:
                        continue
                    xa = V(P, LMS[aidx[a]], v, "cka_biased", "full", "s_cal")
                    x13 = V(P, L13, v, "cka_biased", "full", "s_cal")
                    obs = "below" if x13 < xa else ("above" if x13 > xa else "tie")
                    nd += 1
                    xc += dec == obs
            hc[h] = [nd, xc]
            mine(f"PO2b|{P}|s_cal|h={h}|n_decided", nd, src="F")
            mine(f"PO2b|{P}|s_cal|h={h}|x_correct", xc, src="F")
        out.setdefault("PO2b_h_curve", {})[P] = hc
        # ---- P-O1, P-O4 ----
        mine(f"PO1|{P}|holds", bool(abs(out["values"][f"dtr|{P}|mknn|full|s_cal"]) < 0.10
                                   and abs(out["values"][f"dtr|{P}|mknn|full|unc"]) < 0.10))
        for X in ("s_cal", "unc"):
            mine(f"PO4|{P}|{X}|holds", bool(out["values"][f"dtr|{P}|cka_debiased|full|{X}"] >= -0.10))
        # ---- P-O5, P-O5a, D3, plug-in, OL reading ----
        n = x = 0
        for v in VITS:
            for a in LMS[:2]:
                gB = pp["Blate"][L13][v] - pp["Blate"][a][v]
                if abs(gB) >= 0.005:
                    n += 1
                    gc = V(P, L13, v, "cka_biased", "ctm", "cM") - V(P, a, v, "cka_biased", "ctm", "cM")
                    x += np.sign(gc) == np.sign(gB)
        mine(f"PO5a|{P}|n", n)
        mine(f"PO5a|{P}|x", int(x))
        v5a = "not evaluable" if n < 6 else ("holds" if x >= math.ceil(2 * n / 3) else "fails")
        mine(f"PO5a|{P}|verdict", v5a)
        sl = pp["s_late"]
        for X in ("M", "M_unc"):
            d = out["values"][f"dtr|{P}|cka_biased|ctm|{X}"]
            traw = out["values"][f"tr|{P}|cka_biased|full|raw"]
            if not pp["material_Blate"]:
                cat = "no category (not material)"
            elif v5a == "fails":
                cat = "premise failed"
            elif abs(traw + sl) <= EPS:
                cat = "not testable"
            elif sl * d <= -0.10:
                cat = "loss reproduced"
            elif sl * d >= 0.10:
                cat = "moved with the baseline"
            else:
                cat = "trend kept"
            mine(f"PO5|{P}|{X}|cat", cat)
        R0 = rows("cka_biased", "full", "raw")
        mine(f"d3cal|{P}|cka_biased", tr(rows("cka_biased", "ctm", "M"))[0] - tr(rows("cka_biased", "ctm", "TM"))[0])
        mine(f"d3sel|{P}|cka_biased", tr(rows("cka_biased", "ctm", "TM"))[0] - tr(R0)[0])
        plug = [[V(P, lm, v, "cka_biased", "full", "raw") - pp["Bbar"][lm][v] for lm in LMS] for v in VITS]
        mine(f"plugin|{P}|diff", tr(plug)[0] - out["values"][f"tr|{P}|cka_biased|full|unc"])
        trr = out["values"][f"tr|{P}|cka_biased|full|raw"]
        tuc = out["values"][f"tr|{P}|cka_biased|full|unc"]
        tum = out["values"][f"tr|{P}|mknn|full|unc"]
        rd = ("not applied" if trr < 0.60 else "loses" if (tuc <= 0.40 and tum >= 0.60)
              else "no loss" if tuc >= 0.60 else "intermediate")
        mine(f"readO|{P}|cka_biased", rd)
        # descriptive per-ViT counts for the text
        g13 = [V(P, L13, v, "cka_biased", "full", "raw") - V(P, LMS[1], v, "cka_biased", "full", "raw") for v in VITS]
        gc13 = [V(P, L13, v, "cka_biased", "full", "c") - V(P, LMS[1], v, "cka_biased", "full", "c") for v in VITS]
        out.setdefault("desc_7b_to_13b", {})[P] = {
            "n_vit_raw_up": int(sum(g > 0 for g in g13)), "median_raw_gain": float(np.median(g13)),
            "median_threshold_gain": float(np.median(gc13)),
            "n_vit_raw_gain_below_threshold_gain": int(sum(a < b for a, b in zip(g13, gc13)))}

        # ---- pooled ARH panel (section D) ----
        G = {k.split("|", 1)[1]: np.asarray(v, float) for k, v in I["pooled"]["grids"].items() if k.startswith(P + "|")}
        order = I["pooled_order"]
        xr = np.array([I["arh_rank"][lm] for lm in order], float)
        assert list(xr) == list(range(1, 12))
        # grid rows of the OpenLLaMA models equal the per-pair values
        for i, lm in enumerate(order):
            if lm in LMS:
                for j, v in enumerate(VITS):
                    assert G["cka_biased|raw"][i, j] == V(P, lm, v, "cka_biased", "full", "raw")
        r = {f"{M}|{X}": pear(G[f"{M}|{X}"], xr) for M in ("mknn", "cka_biased", "cka_debiased")
             for X in ("raw", "s_cal", "unc")}
        for k, val in r.items():
            M, X = k.split("|")
            mine(f"arh|11|{P}|{M}|{X}|arh|r", val)

        def DD(Gs, x, M="cka_biased"):
            return (pear(Gs[f"{M}|raw"], x) - pear(Gs[f"{M}|s_cal"], x)) - (pear(Gs["mknn|raw"], x) - pear(Gs["mknn|s_cal"], x))

        for M in ("cka_biased", "cka_debiased"):
            dd = DD(G, xr, M)
            mine(f"dd|11|{P}|{M}", dd)
            rng = util.rng_from_key(f"S1.2-layermax-openllama|boot|arh_delta|{P}|{M}")
            bs = np.empty(2000)
            for b in range(2000):
                j = rng.integers(0, 17, 17)
                bs[b] = DD({k: g[:, j] for k, g in G.items()}, xr, M)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            mine(f"dd|11|{P}|{M}|ci_lo", float(lo))
            mine(f"dd|11|{P}|{M}|ci_hi", float(hi))
            if M == "cka_biased":
                rule = "replicated the direction" if lo > 0 else ("direction not established" if hi >= 0 else "opposite direction")
                mine(f"rule7a|{P}", rule)
                mine(f"size_not_reproduced|{P}", bool(0.40 > hi))
                out.setdefault("boot_frac_le_0", {})[P] = float(np.mean(bs <= 0))
                # exact one-sided p over all 11! relabellings: DD(sigma) is linear in the permuted rank vector
                w = np.zeros(11)
                for sgn, key in ((1, f"{M}|raw"), (-1, f"{M}|s_cal"), (-1, "mknn|raw"), (1, "mknn|s_cal")):
                    g = G[key]
                    a = g.sum(axis=1) - 17 * g.mean()
                    D = math.sqrt(17 * np.sum((xr - xr.mean()) ** 2)) * math.sqrt(np.sum((g - g.mean()) ** 2))
                    w += sgn * a / D
                assert abs(float(xr @ w) - dd) < 1e-12
                p8 = np.array(list(itertools.permutations(range(8))), dtype=np.int64)
                cnt = 0
                tot = 0
                for pre3 in itertools.permutations(range(11), 3):
                    rest = np.array([i for i in range(11) if i not in pre3])
                    base = sum(xr[pre3[t]] * w[t] for t in range(3))
                    vals = base + (xr[rest][p8] @ w[3:])
                    cnt += int(np.sum(vals >= dd - EPS))
                    tot += len(vals)
                assert tot == math.factorial(11)
                mine(f"dd|11|{P}|{M}|pge", cnt / tot)
                # FP section 6 category
                gap = r["mknn|s_cal"] - r["cka_biased|s_cal"]
                raw_cat = ("ARH pattern reappears" if (dd >= 0.20 and r["cka_biased|s_cal"] <= r["mknn|s_cal"] - 0.20)
                           else "absent" if (dd <= 0.10 or r["cka_biased|s_cal"] >= r["mknn|s_cal"] - 0.10) else "partial")
                mine(f"fp6|{P}|raw_cat", raw_cat)
                rng = util.rng_from_key(f"S1.2-layermax-openllama|boot|arh_gap|{P}")
                gb = np.empty(2000)
                for b in range(2000):
                    j = rng.integers(0, 17, 17)
                    gb[b] = pear(G["mknn|s_cal"][:, j], xr) - pear(G["cka_biased|s_cal"][:, j], xr)
                glo, ghi = np.nanpercentile(gb, [2.5, 97.5])
                kedge = (lo <= 0.10 <= hi) or (glo <= 0.10 <= ghi)
                mine(f"fp6|{P}|cat", "not decidable at this panel size (knife edge)" if kedge else raw_cat)
                # POST HOC sensitivity (not a decision): other bootstrap keys, ViT jackknife, 10-LM panel
                Ls, Us = [], []
                for s in range(50):
                    rr = util.rng_from_key(f"oll-review|posthoc|arh_delta|{P}|{M}|{s}")
                    bb = np.empty(2000)
                    for b in range(2000):
                        j = rr.integers(0, 17, 17)
                        bb[b] = DD({k: g[:, j] for k, g in G.items()}, xr, M)
                    l_, u_ = np.nanpercentile(bb, [2.5, 97.5])
                    Ls.append(float(l_))
                    Us.append(float(u_))
                jk = [DD({k: np.delete(g, j, axis=1) for k, g in G.items()}, xr, M) for j in range(17)]
                idx10 = [i for i in range(11) if order[i] != "openlm-research__open_llama_13b"]
                dd10 = DD({k: g[idx10] for k, g in G.items()}, xr[idx10], M)
                out.setdefault("posthoc_sensitivity", {})[P] = {
                    "label": "POST HOC (reviewer); not part of the section 7(a) decision",
                    "ci_lo_over_50_other_keys": {"min": min(Ls), "median": float(np.median(Ls)), "max": max(Ls),
                                                 "n_lo_gt_0": int(sum(l_ > 0 for l_ in Ls))},
                    "ci_hi_over_50_other_keys": {"min": min(Us), "max": max(Us)},
                    "vit_jackknife_dd": {"min": min(jk), "max": max(jk), "n_le_0": int(sum(x_ <= 0 for x_ in jk))},
                    "dd_10LM_without_13b": dd10}

        # ---- joint summary J (EXPLORATORY), primary pre only ----
        if P == "prh_clip+l2":
            fam = {"B": [order.index(x) for x in order if x.startswith("bigscience__bloomz")],
                   "O": [order.index(x) for x in LMS],
                   "L": [order.index(x) for x in ("huggyllama__llama-7b", "huggyllama__llama-13b", "huggyllama__llama-30b")]}
            assert len(fam["B"]) == 5
            sf = {"B": 1, "L": 1, "O": pp["s_O_J"]}
            for X in ("s_cal", "unc"):
                gx, g0 = G[f"cka_biased|{X}"], G["cka_biased|raw"]
                D = {}
                for f, ix in fam.items():
                    k = len(ix)
                    rx = [list(gx[ix, j]) for j in range(17)]
                    r0 = [list(g0[ix, j]) for j in range(17)]
                    perms = list(itertools.permutations(range(1, k + 1)))
                    D[f] = np.array([sf[f] * (tr(rx, s)[0] - tr(r0, s)[0]) for s in perms])
                    assert perms[0] == tuple(range(1, k + 1))
                    mine(f"J|{X}|dTr|{f}", float(D[f][0]) * sf[f], src="J")
                ids = {f: D[f][0] for f in D}
                jmin = max(ids.values())
                grid = np.array([max(a, b, c) for a in D["B"] for b in D["L"] for c in D["O"]])
                gmean = np.array([(a + b + c) / 3 for a in D["B"] for b in D["L"] for c in D["O"]])
                mine(f"J|{X}|Jmin", float(jmin), src="J")
                mine(f"J|{X}|pmin", float(np.mean(grid <= jmin + EPS)), src="J")
                prod = np.prod([np.sum(D[f] <= jmin + EPS) for f in D]) / 4320
                mine(f"J|{X}|pmin_product", float(prod), src="J")
                jm = float(np.mean(list(ids.values())))
                mine(f"J|{X}|Jmean", jm, src="J")
                mine(f"J|{X}|pmean", float(np.mean(gmean <= jm + EPS)), src="J")
                mine(f"J|{X}|nO", int(np.sum(D["O"] <= D["O"][0] + EPS)), src="J")
                # conditional size of v0's rule
                rx = [[V(P, lm, v, "cka_biased", "full", X) for lm in LMS] for v in VITS]
                r0 = [[V(P, lm, v, "cka_biased", "full", "raw") for lm in LMS] for v in VITS]
                rej = 0
                for si, s in enumerate(PERM3):
                    dO = D["O"][si]
                    jm_s = max(ids["B"], ids["L"], dO)
                    p_s = float(np.mean(grid <= jm_s + EPS))
                    new = [s.index(3)]
                    others = [i for i in range(3) if i != new[0]]
                    fm = fp = 0
                    for vr, vx in zip(r0, rx):
                        for a in others:
                            dr = np.sign(vr[new[0]] - vr[a])
                            dx = np.sign(vx[new[0]] - vx[a])
                            if s[a] < 3:
                                fm += dx < dr if sf["O"] == 1 else dx > dr
                    rej += (p_s <= 0.05) and (dO <= -0.10) and (fm >= 3)
                mine(f"J|{X}|cond_size", rej / 6, src="J")
            # the other families, for the text
            for X in ("s_cal", "unc"):
                pass

    ok = all(r["equal"] for r in cmp_rows)
    res = {"role": "oll-review", "what": "reviewer's own recomputation of the plan endpoints from the summary files",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "secs": time.time() - t0,
           "n_compared": len(cmp_rows), "n_mismatch": sum(not r["equal"] for r in cmp_rows),
           "mismatches": [r for r in cmp_rows if not r["equal"]], "pass": ok and out["pair_values_I_summary_vs_E_json"]["pass"],
           **out}
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, "endpoints_review.json")
    if "--dry" in sys.argv:
        fn = None
    elif os.path.exists(fn):
        sys.exit("refusing to overwrite " + fn)
    if fn:
        with open(fn, "w") as f:
            json.dump(res, f, indent=1, default=float)
        os.chmod(fn, 0o444)
    print(json.dumps({k: res[k] for k in ("n_compared", "n_mismatch", "pass", "secs")}))
    print(json.dumps(res["mismatches"][:20], default=float))
    print(json.dumps({k: res[k] for k in ("pair_values_I_summary_vs_E_json", "means", "PO2ai_range", "PO2b_cp",
                                          "PO2b_forecast_split", "PO2b_table", "PO2b_h_curve", "kappa_recomputed_vs_partB_maxabs",
                                          "desc_7b_to_13b", "boot_frac_le_0", "posthoc_sensitivity")}, default=float, indent=1))
    if fn:
        print("output", os.path.relpath(fn, PROJ), sha(fn))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
