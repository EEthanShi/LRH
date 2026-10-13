"""OpenLLaMA axis, implementation I: the verdict arithmetic of VERDICTS_SPEC.md sections A, B, C, D, G
(NEW file, S0b, sub-role I of oll-axis-extract). Pure functions on in-memory records; no file is read
here (lmx_ol_aggregate.py loads the pair files after the Part B gate and calls ``verdicts``).

Records: rec[lm][vit][met][Q] for Q in SUBSETS, and rec[lm][vit][met]["ctm"], with
  raw, s_cal, unc, c (c_alpha), mu (null mean), p, a, b, n_tied_max            (Q != ctm)
  c_entry, mu_entry, max_entry_mu                                                (Q == full)
  M, M_unc, TM, cM, a, b                                                         (ctm)
For mKNN every value except p, a, b is a fractions.Fraction built from the integer counts out of
m k = 10,240 (plan F2); for CKA they are float64. Orders, ranks and ties use these exact values.
"""
from __future__ import annotations

import itertools
import math
import os
import sys
from fractions import Fraction

import numpy as np
from scipy.stats import beta

_PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)          # hrhv2 (util.rng_from_key)

PRES = ("prh_clip+l2", "l2")
METRICS = ("mknn", "cka_biased", "cka_debiased")
SUBSETS = ("full", "fixed_r0.5", "fixed_r1.0", "grid2", "grid3", "grid5", "grid9", "grid17",
           "LMall_ViTlast", "LMlast_ViTall", "noLMemb", "LMdeep", "LMemb_only")
PROTOCOLS = SUBSETS + ("ctm",)
SIZES = ("3b", "7b", "13b")
KINDS = ("arh", "dense", "logp")
TAUS = (0.03, 0.06, 0.10, 0.15, 0.20)
DBS = (0.0025, 0.005, 0.0075, 0.01)
MAT = 0.10
DELTA_B = 0.005
TOL = 1e-9
NB = 2000
BAND = (0.0009, 0.0029)
PERM6 = [tuple(float(v) for v in p) for p in itertools.permutations((1, 2, 3))]
ID6 = PERM6.index((1.0, 2.0, 3.0))
PAIRS3 = ((0, 1), (0, 2), (1, 2))          # (a, b) with a < b in size order: 3b-7b, 3b-13b, 7b-13b
NEW = ((0, 2), (1, 2))                     # the 34 new comparisons (a in {3b, 7b}, b = 13b)
PILOT = ((0, 1),)                          # the 17 pilot comparisons


def fmt(v: float) -> str:
    return repr(float(v))


def f(v):
    """JSON value: Fraction -> float; None stays None; numpy scalars -> python."""
    if v is None:
        return None
    if isinstance(v, Fraction):
        return float(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    x = float(v)
    return x


def finite(v) -> bool:
    if v is None:
        return False
    if isinstance(v, Fraction):
        return True
    return math.isfinite(float(v))


def sgn(x) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def cp95(x: int, n: int):
    """Clopper-Pearson 95% (spec section 9)."""
    if n == 0:
        return None, None
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    return lo, hi


def med(vals):
    """numpy.median of float64 values (Fractions converted with float(), spec section 9)."""
    return float(np.median(np.array([float(v) for v in vals], dtype=np.float64)))


# ------------------------------------------------------------------------------------------- section A
def avg_ranks(vals):
    """average ranks (1-based) of comparable exact values (= scipy.stats.rankdata 'average' for floats)."""
    return [1 + sum(1 for y in vals if y < x) + 0.5 * (sum(1 for y in vals if y == x) - 1) for x in vals]


def trend(cols):
    """cols: 17 lists [v_3b, v_7b, v_13b]. Spearman rho_v (Pearson on average ranks) for each of the 6
    relabellings sigma of the size ranks (same sigma for every ViT); Tr(sigma) = mean over included ViTs.
    A ViT with three exactly equal scores or any non-finite score is excluded."""
    valid = [j for j, c in enumerate(cols) if all(finite(v) for v in c) and not (c[0] == c[1] == c[2])]
    out = {"nconst": len(cols) - len(valid), "valid": valid}
    if not valid:
        out.update({"T": None, "obs": None, "rho": [None] * len(cols)})
        return out
    R = np.array([avg_ranks(cols[j]) for j in valid], dtype=np.float64)          # (nv, 3)
    Rc = R - R.mean(axis=1, keepdims=True)
    X = np.array(PERM6, dtype=np.float64)                                        # (6, 3)
    Xc = X - X.mean(axis=1, keepdims=True)
    num = Xc @ Rc.T
    den = np.sqrt((Xc ** 2).sum(axis=1))[:, None] * np.sqrt((Rc ** 2).sum(axis=1))[None, :]
    rho = num / den                                                              # (6, nv)
    T = rho.mean(axis=1)
    rv = [None] * len(cols)
    for i, j in enumerate(valid):
        rv[j] = float(rho[ID6, i])
    out.update({"T": T, "obs": float(T[ID6]), "rho": rv})
    return out


def pvals(T, obs):
    if T is None or obs is None:
        return None, None, None
    return (float(np.sum(T >= obs - TOL)) / 6.0, float(np.sum(T <= obs + TOL)) / 6.0,
            float(np.sum(np.abs(T) >= abs(obs) - TOL)) / 6.0)


def neartie(cols) -> int:
    n = 0
    for c in cols:
        for a, b in PAIRS3:
            x, y = c[a], c[b]
            if finite(x) and finite(y) and abs(float(x) - float(y)) <= 1e-12 * max(abs(float(x)), abs(float(y))):
                n += 1
    return n


def cols_of(rec, lms, vits, met, q, x):
    return [[rec[lm][v][met][q][x] for lm in lms] for v in vits]


def xname(q, x):
    """spec score name -> record field"""
    return x


# ------------------------------------------------------------------------------------------- the verdicts
def verdicts(rec, ol_lms, vits, partb, pooled, extra):
    """rec: OL records (rec[lm][vit][met][Q]); ol_lms: [3b, 7b, 13b] slugs; partb: parsed
    partB_features_ol.json; pooled: dict from pooled_section(); extra: dict for x| extras.
    Returns the flat verdict dict of sections A, B, C (D and G are added by the caller)."""
    V = {}
    TR = {}   # (p, m, q, x) -> trend dict
    for p in PRES:
        R = rec[p]
        for m in METRICS:
            for q in PROTOCOLS:
                xs = ("M", "M_unc", "TM") if q == "ctm" else ("raw", "s_cal", "unc")
                for x in xs:
                    cols = cols_of(R, ol_lms, vits, m, q, x)
                    t = trend(cols)
                    TR[(p, m, q, x)] = t
                    k = f"{p}|{m}|{q}|{x}"
                    V[f"tr|{k}"] = t["obs"]
                    V[f"tr_nconst|{k}"] = t["nconst"]
                    pge, ple, ptwo = pvals(t["T"], t["obs"])
                    V[f"tr_pge|{k}"], V[f"tr_ple|{k}"], V[f"tr_ptwo|{k}"] = pge, ple, ptwo
                    if m != "mknn":
                        V[f"neartie|{k}"] = neartie(cols)
                    extra[f"x|rho|{k}"] = t["rho"]
                base = TR[(p, m, "full", "raw")] if q == "ctm" else TR[(p, m, q, "raw")]
                dxs = ("M", "M_unc", "TM") if q == "ctm" else ("s_cal", "unc")
                for x in dxs:
                    t = TR[(p, m, q, x)]
                    k = f"{p}|{m}|{q}|{x}"
                    if t["T"] is None or base["T"] is None:
                        V[f"dtr|{k}"] = None
                        V[f"dtr_pge|{k}"] = V[f"dtr_ple|{k}"] = V[f"dtr_ptwo|{k}"] = None
                        continue
                    D = t["T"] - base["T"]
                    obs = float(D[ID6])
                    V[f"dtr|{k}"] = obs
                    V[f"dtr_pge|{k}"], V[f"dtr_ple|{k}"], V[f"dtr_ptwo|{k}"] = pvals(D, obs)
            trm, trtm, trraw = (V[f"tr|{p}|{m}|ctm|M"], V[f"tr|{p}|{m}|ctm|TM"], V[f"tr|{p}|{m}|full|raw"])
            V[f"d3cal|{p}|{m}"] = None if trm is None or trtm is None else trm - trtm
            V[f"d3sel|{p}|{m}"] = None if trtm is None or trraw is None else trtm - trraw
    extra["_TR"] = TR
    # ---------------------------------------------------------------------------------- section B
    for p in PRES:
        R = rec[p]
        s = int(partb["per_pre"][p]["s_O"])
        for m in METRICS:
            for q in PROTOCOLS:
                for x in (("M", "M_unc") if q == "ctm" else ("s_cal", "unc")):
                    rq = "full" if q == "ctm" else q
                    raw = cols_of(R, ol_lms, vits, m, rq, "raw")
                    sc = cols_of(R, ol_lms, vits, m, q, x)
                    for grp, prs in (("new", NEW), ("pilot", PILOT)):
                        c = {"down": 0, "up": 0, "c2d": 0, "d2c": 0}
                        for v in range(len(vits)):
                            for a, b in prs:
                                dr, dx = sgn(raw[v][b] - raw[v][a]), sgn(sc[v][b] - sc[v][a])
                                c["down"] += int(dx < dr)
                                c["up"] += int(dx > dr)
                                c["c2d"] += int(dr == 1 and dx == -1)
                                c["d2c"] += int(dr == -1 and dx == 1)
                        for kk, vv in c.items():
                            V[f"chg|{p}|{m}|{q}|{x}|{grp}|{kk}"] = vv
                    fm = V[f"chg|{p}|{m}|{q}|{x}|new|down"] if s == 1 else V[f"chg|{p}|{m}|{q}|{x}|new|up"]
                    fp = V[f"chg|{p}|{m}|{q}|{x}|new|up"] if s == 1 else V[f"chg|{p}|{m}|{q}|{x}|new|down"]
                    k = f"fnew|{p}|{m}|{q}|{x}"
                    V[f"{k}|F"] = (fm - fp) / 34.0
                    V[f"{k}|fminus"], V[f"{k}|fplus"] = fm, fp
                    V[f"{k}|cp_lo"], V[f"{k}|cp_hi"] = cp95(fm, 34)
    # ---------------------------------------------------------------------------------- section C
    for p in PRES:
        section_c(V, rec[p], p, ol_lms, vits, partb, pooled, TR, extra)
    return V


def _lm(d, slug, label):
    if slug in d:
        return d[slug]
    if label in d:
        return d[label]
    raise KeyError(f"{slug} / {label} not in Part B table {list(d)[:5]}")


def p2cat(s, d, tr_raw, material, tau=MAT):
    """P-O2 category (spec section 5), decided in the spec's order."""
    if not material:
        return "no category (not material)"
    if tr_raw is not None and abs(tr_raw + s) <= TOL:
        return "not testable"
    if d is None:
        return "undefined (Tr not defined)"
    if s * d <= -tau:
        return "loss reproduced"
    if s * d >= tau:
        return "moved with the baseline"
    if abs(d) < tau:
        return "trend kept"
    return "undefined"


def p5cat(s, d, tr_raw, material, premise, tau=MAT):
    if not material:
        return "no category (not material)"
    if premise == "fails":
        return "premise failed"
    if tr_raw is not None and abs(tr_raw + s) <= TOL:
        return "not testable"
    return p2cat(s, d, tr_raw, True, tau)


def section_c(V, R, p, ol, vits, partb, pooled, TR, extra):
    pp = partb["per_pre"][p]
    lab = dict(zip(ol, SIZES))
    s = int(pp["s_O"])
    material = bool(pp["material"])
    nv = len(vits)
    g = lambda m, q, x: cols_of(R, ol, vits, m, q, x)  # noqa: E731
    # ---- P-O1 (mKNN, full)
    ds, du = V[f"dtr|{p}|mknn|full|s_cal"], V[f"dtr|{p}|mknn|full|unc"]
    V[f"PO1|{p}|holds"] = bool(ds is not None and du is not None and abs(ds) < MAT and abs(du) < MAT)
    T, c, mu = g("mknn", "full", "raw"), g("mknn", "full", "c"), g("mknn", "full", "mu")
    base = Fraction(10, 1023)
    for x, thr in (("s_cal", c), ("unc", mu)):
        X = g("mknn", "full", x)
        nq = nrev = ntie = 0
        for v in range(nv):
            for a, b in PAIRS3:
                za, zb = abs(thr[v][a] - base), abs(thr[v][b] - base)
                gap = T[v][b] - T[v][a]
                if abs(gap) > za + zb:
                    nq += 1
                    dx = sgn(X[v][b] - X[v][a])
                    nrev += int(dx == -sgn(gap) and dx != 0)
                    ntie += int(dx == 0)
        V[f"PO1|{p}|zeta|{x}|nq"], V[f"PO1|{p}|zeta|{x}|nrev"], V[f"PO1|{p}|zeta|{x}|ntie"] = nq, nrev, ntie
    # ---- P-O2 (biased CKA, full)
    tr_raw = V[f"tr|{p}|cka_biased|full|raw"]
    for x in ("s_cal", "unc"):
        d = V[f"dtr|{p}|cka_biased|full|{x}"]
        V[f"PO2|{p}|{x}|cat"] = p2cat(s, d, tr_raw, material)
        if d is None:
            V[f"PO2|{p}|{x}|dist"], V[f"PO2|{p}|{x}|kedge"] = None, None
        else:
            dist = min(abs(s * d + MAT), abs(s * d - MAT))
            V[f"PO2|{p}|{x}|dist"] = dist
            V[f"PO2|{p}|{x}|kedge"] = bool(dist <= 1.0 / 17 + TOL)
        for tau in TAUS:
            V[f"curve|{p}|PO2|{x}|tau={fmt(tau)}"] = p2cat(s, d, tr_raw, material, tau)
    # ---- P-O2a(i)
    Bbar = {lm: [float(_lm(pp["Bbar"], lm, lab[lm])[v]) for v in vits] for lm in ol}
    cb = g("cka_biased", "full", "c")
    offs = [cb[v][2] - Bbar[ol[2]][v] for v in range(nv)]
    md = med(offs)
    V[f"PO2ai|{p}|median"] = md
    V[f"PO2ai|{p}|in_band"] = bool(BAND[0] <= md <= BAND[1])
    xin = sum(1 for o in offs if BAND[0] <= o <= BAND[1])
    V[f"PO2ai|{p}|n"], V[f"PO2ai|{p}|x"] = nv, xin
    V[f"PO2ai|{p}|cp_lo"], V[f"PO2ai|{p}|cp_hi"] = cp95(xin, nv)
    lpm = med([abs(o) for o in offs])
    V[f"PO2ai|{p}|lp_median_abs"], V[f"PO2ai|{p}|lp_le_0.01"] = lpm, bool(lpm <= 0.01)
    for v, o in zip(vits, offs):
        V[f"PO2ai|{p}|offset|{v}"] = o

    # ---- P-O2a(ii) and its delta_b curve
    def p2aii(delta):
        n = x = 0
        for v in range(nv):
            for a, b in NEW:
                gb = Bbar[ol[b]][v] - Bbar[ol[a]][v]
                if abs(gb) >= delta:
                    n += 1
                    x += int(sgn(cb[v][b] - cb[v][a]) == sgn(gb))
        need = math.ceil(2 * n / 3)
        verdict = "not evaluable" if n < 6 else ("holds" if x >= need else "fails")
        return n, x, need, verdict
    n, x, need, verdict = p2aii(DELTA_B)
    V[f"PO2aii|{p}|n"], V[f"PO2aii|{p}|x"], V[f"PO2aii|{p}|need"], V[f"PO2aii|{p}|verdict"] = n, x, need, verdict
    V[f"PO2aii|{p}|cp_lo"], V[f"PO2aii|{p}|cp_hi"] = cp95(x, n)
    tr_pred = float(pp["Tr_pred"])
    for dlt in DBS:
        n_, x_, _, vd_ = p2aii(dlt)
        V[f"curve|{p}|PO2aii|db={fmt(dlt)}|n"], V[f"curve|{p}|PO2aii|db={fmt(dlt)}|x"] = n_, x_
        V[f"curve|{p}|PO2aii|db={fmt(dlt)}|verdict"] = vd_
        gaps_ok = all(med([abs(Bbar[ol[b]][v] - Bbar[ol[a]][v]) for v in range(nv)]) >= dlt for a, b in PAIRS3)
        V[f"curve|{p}|material|db={fmt(dlt)}"] = bool(abs(abs(tr_pred) - 1.0) <= TOL and gaps_ok)
    extra[f"x|check|{p}|curve_material_db0.005_equals_partB_material"] = (
        V[f"curve|{p}|material|db={fmt(0.005)}"] == material)
    # ---- P-O3 (biased CKA, fixed depth)
    for r in ("0.5", "1.0"):
        trr = V[f"tr|{p}|cka_biased|fixed_r{r}|raw"]
        for x in ("s_cal", "unc"):
            trx = V[f"tr|{p}|cka_biased|fixed_r{r}|{x}"]
            if trr is None or abs(trr) < MAT:
                desc = "not applied"
            else:
                sr = 1 if trr > 0 else -1
                tx = 0.0 if trx is None else trx      # [op, I] a Tr without any ranked ViT is read as 0
                if sr * tx >= MAT:
                    desc = "keeps sign"
                elif abs(tx) < MAT:
                    desc = "lost"
                else:
                    desc = "reversed"
            V[f"PO3|{p}|{r}|{x}|desc"] = desc
    # ---- keep identity and sign rule (all M, all Q != ctm, all 51 comparisons)
    for m in METRICS:
        exact = m == "mknn"
        for q in SUBSETS:
            T, c, mu = g(m, q, "raw"), g(m, q, "c"), g(m, q, "mu")
            for x, thr in (("s_cal", c), ("unc", mu)):
                X = g(m, q, x)
                nk = ns = 0
                for v in range(nv):
                    for a, b in PAIRS3:
                        Ta, Tb, ta, tb, Xa, Xb = T[v][a], T[v][b], thr[v][a], thr[v][b], X[v][a], X[v][b]
                        # keep identity
                        if x == "s_cal":
                            if Xa > 0 and Xb > 0:
                                lhs, rhs = Tb - Ta, (1 - Ta) * (tb - ta) / (1 - ta)
                                if (lhs >= rhs) != (Xb - Xa >= 0) and (exact or abs(lhs - rhs) > 1e-12):
                                    nk += 1
                        else:
                            lhs, rhs = Tb - Ta, tb - ta
                            if (lhs >= rhs) != (Xb - Xa >= 0) and (exact or abs(lhs - rhs) > 1e-12):
                                nk += 1
                        # sign rule
                        dr, dx = sgn(Tb - Ta), sgn(Xb - Xa)
                        if (tb > ta and dr <= 0 and dx == 1) or (tb < ta and dr >= 0 and dx == -1) or \
                                (tb == ta and dx == -dr and dx != 0):
                            ns += 1
                V[f"keepid|{p}|{m}|{q}|{x}|nviol"] = nk
                V[f"signrule|{p}|{m}|{q}|{x}|nviol"] = ns
    # ---- P-O4 (debiased CKA)
    for x in ("s_cal", "unc"):
        d = V[f"dtr|{p}|cka_debiased|full|{x}"]
        V[f"PO4|{p}|{x}|holds"] = bool(d is not None and d >= -MAT)
    rb_, ra_ = pooled["r"][("11", p, "cka_debiased", "raw", "arh")], pooled["r"][("11", p, "cka_debiased", "s_cal", "arh")]
    V[f"PO4|{p}|arh|holds"] = bool(rb_ is not None and ra_ is not None and ra_ >= rb_ - 0.05)
    # ---- P-O5a and P-O5 (calibrate-then-max, biased CKA)
    sl = int(pp["s_late"])
    mat_l = bool(pp["material_Blate"])
    Bl = {lm: [float(_lm(pp["Blate"], lm, lab[lm])[v]) for v in vits] for lm in ol}
    cM = g("cka_biased", "ctm", "cM")
    n = x = 0
    for v in range(nv):
        for a, b in NEW:
            gb = Bl[ol[b]][v] - Bl[ol[a]][v]
            if abs(gb) >= DELTA_B:
                n += 1
                x += int(sgn(cM[v][b] - cM[v][a]) == sgn(gb))
    need = math.ceil(2 * n / 3)
    v5a = "not evaluable" if n < 6 else ("holds" if x >= need else "fails")
    V[f"PO5a|{p}|n"], V[f"PO5a|{p}|x"], V[f"PO5a|{p}|need"], V[f"PO5a|{p}|verdict"] = n, x, need, v5a
    for xx in ("M", "M_unc"):
        d = V[f"dtr|{p}|cka_biased|ctm|{xx}"]
        V[f"PO5|{p}|{xx}|pred_holds"] = (None if not mat_l else (None if d is None else bool(sl * d <= 0)))
        V[f"PO5|{p}|{xx}|cat"] = p5cat(sl, d, tr_raw, mat_l, v5a)
        for tau in TAUS:
            V[f"curve|{p}|PO5|{xx}|tau={fmt(tau)}"] = p5cat(sl, d, tr_raw, mat_l, v5a, tau)
    for tau in TAUS:
        V[f"curve|{p}|PO1|tau={fmt(tau)}"] = bool(ds is not None and du is not None and abs(ds) < tau and abs(du) < tau)
    # ---- D3 deductive check (biased CKA): c^M strictly rising and rho_v(M) - rho_v(TM) > 0
    rM, rTM = TR[(p, "cka_biased", "ctm", "M")]["rho"], TR[(p, "cka_biased", "ctm", "TM")]["rho"]
    nviol = 0
    for v in range(nv):
        if cM[v][0] < cM[v][1] < cM[v][2] and rM[v] is not None and rTM[v] is not None and rM[v] - rTM[v] > 0:
            nviol += 1
    V[f"D3|{p}|nviol"] = nviol
    # ---- plug-in check
    Tcols = g("cka_biased", "full", "raw")
    t_pl = trend([[Tcols[v][i] - Bbar[ol[i]][v] for i in range(3)] for v in range(nv)])
    V[f"plugin|{p}|tr"] = t_pl["obs"]
    tu = V[f"tr|{p}|cka_biased|full|unc"]
    V[f"plugin|{p}|diff"] = None if t_pl["obs"] is None or tu is None else t_pl["obs"] - tu
    V[f"plugin|{p}|within"] = bool(V[f"plugin|{p}|diff"] is not None and abs(V[f"plugin|{p}|diff"]) <= MAT)
    # ---- section 7 OpenLLaMA-axis reading
    tku = V[f"tr|{p}|mknn|full|unc"]
    for m in ("cka_biased", "cka_debiased"):
        trr, tru = V[f"tr|{p}|{m}|full|raw"], V[f"tr|{p}|{m}|full|unc"]
        if trr is None or trr < 0.60:
            rd = "not applied"
        elif tru is not None and tku is not None and tru <= 0.40 and tku >= 0.60:
            rd = "loses"
        elif tru is not None and tru >= 0.60:
            rd = "no loss"
        else:
            rd = "intermediate"
        V[f"readO|{p}|{m}"] = rd
    # ---- D1, D2, 7B > 3B
    for m in METRICS:
        for i, lm in enumerate(ol):
            rs = [R[lm][v][m]["full"] for v in vits]
            V[f"D1|{p}|{m}|{SIZES[i]}"] = med([r["mu"] - r["max_entry_mu"] for r in rs])
            V[f"D2|{p}|{m}|{SIZES[i]}|argmax_rel"] = med([r["a"] / (r["LA"] - 1) for r in rs])
            V[f"D2|{p}|{m}|{SIZES[i]}|c_minus_centry"] = med([r["c"] - r["c_entry"] for r in rs])
        T = g(m, "full", "raw")
        V[f"frac7gt3|{p}|{m}"] = sum(1 for v in range(nv) if T[v][1] > T[v][0]) / nv


# ------------------------------------------------------------------------------------------- section D
def pear(G, x):
    """aggregate_llama.py::pear (LLaMA E, plan R.3.6): np.corrcoef of the ravelled (LM x ViT) grid
    against the LM covariate repeated over ViTs; NaN for a constant or non-finite grid."""
    xx = np.repeat(x[:, None], G.shape[1], axis=1).ravel()
    y = G.ravel()
    if np.nanstd(y) == 0 or not np.all(np.isfinite(y)):
        return float("nan")
    return float(np.corrcoef(xx, y)[0, 1])


def r_boot(G, x, key, nv):
    from hrhv2 import util
    rng = util.rng_from_key(key)
    bs = np.empty(NB)
    for b in range(NB):
        j = rng.integers(0, nv, nv)
        bs[b] = pear(G[:, j], x)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def delta_boot(G1b, G1a, G2b, G2a, x, key, nv):
    """aggregate_llama.py::delta_boot (LLaMA E) with the key given in full."""
    from hrhv2 import util
    fn = lambda a, b, c, d: (pear(a, x) - pear(b, x)) - (pear(c, x) - pear(d, x))  # noqa: E731
    est = fn(G1b, G1a, G2b, G2a)
    rng = util.rng_from_key(key)
    bs = np.empty(NB)
    for b in range(NB):
        j = rng.integers(0, nv, nv)
        bs[b] = fn(G1b[:, j], G1a[:, j], G2b[:, j], G2a[:, j])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"est": float(est), "ci": [float(lo), float(hi)]}


def gap_boot(Gk, Gc, x, key, nv):
    from hrhv2 import util
    rng = util.rng_from_key(key)
    bs = np.empty(NB)
    for b in range(NB):
        j = rng.integers(0, nv, nv)
        bs[b] = pear(Gk[:, j], x) - pear(Gc[:, j], x)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def linear_weights(G, x):
    """r(sigma) = sum_i xc[sigma(i)] * w_i with w = (row sums of the centred grid) / den (den is
    invariant under relabelling); returns w or None for a constant / non-finite grid."""
    if np.nanstd(G) == 0 or not np.all(np.isfinite(G)):
        return None
    xc = x - x.mean()
    Gc = G - G.mean()
    den = math.sqrt(G.shape[1] * float(np.sum(xc ** 2))) * math.sqrt(float(np.sum(Gc ** 2)))
    return Gc.sum(axis=1) / den


def exact_ge_counts(xc, W, thr):
    """#{sigma in S_n : sum_i xc[sigma(i)] W[i, j] >= thr[j]} for every column j, over all n! relabellings
    (n = 11: 990 prefixes of length 3 x 8! = 40,320 completions; exact enumeration, chunked)."""
    n = len(xc)
    W = np.asarray(W, dtype=np.float64)
    if n <= 4:
        P = np.array(list(itertools.permutations(range(n))), dtype=np.intp)
        return ((xc[P] @ W) >= thr[None, :]).sum(axis=0).astype(np.int64), int(len(P))
    k = n - 3
    Pk = np.array(list(itertools.permutations(range(k))), dtype=np.intp)
    Wt = W[3:]
    cnt = np.zeros(W.shape[1], dtype=np.int64)
    tot = 0
    for a, b, c in itertools.permutations(range(n), 3):
        rem = np.array([i for i in range(n) if i not in (a, b, c)], dtype=np.intp)
        base = xc[a] * W[0] + xc[b] * W[1] + xc[c] * W[2]
        vals = xc[rem][Pk] @ Wt + base[None, :]
        cnt += (vals >= thr[None, :]).sum(axis=0)
        tot += len(Pk)
    return cnt, tot


def pooled_section(grids, panels, covs, log=print, exact=True):
    """grids[(p, m, x)] = 11 x 17 float grid (ARH order); panels = {"11": idx, "10": idx, "8": idx};
    covs[(panel, kind)] = covariate vector. Returns (V entries, info)."""
    V = {}
    info = {"r": {}, "dd": {}}
    nv = next(iter(grids.values())).shape[1]
    for pan, idx in panels.items():
        for p in PRES:
            for m in METRICS:
                for x in ("raw", "s_cal", "unc"):
                    G = grids[(p, m, x)][idx]
                    for kd in KINDS:
                        r = pear(G, covs[(pan, kd)])
                        r = None if not math.isfinite(r) else r
                        V[f"arh|{pan}|{p}|{m}|{x}|{kd}|r"] = r
                        info["r"][(pan, p, m, x, kd)] = r
                        if pan == "11":
                            lo, hi = r_boot(G, covs[(pan, kd)],
                                            "S1.2-layermax-openllama|boot|arh_r|" + f"{p}|{m}|{x}|{kd}", nv)
                            V[f"arh|11|{p}|{m}|{x}|{kd}|ci_lo"], V[f"arh|11|{p}|{m}|{x}|{kd}|ci_hi"] = lo, hi
            for m in ("cka_biased", "cka_debiased"):
                rr = info["r"]
                vals = [rr[(pan, p, m, "raw", "arh")], rr[(pan, p, m, "s_cal", "arh")],
                        rr[(pan, p, "mknn", "raw", "arh")], rr[(pan, p, "mknn", "s_cal", "arh")]]
                dd = None if any(v is None for v in vals) else (vals[0] - vals[1]) - (vals[2] - vals[3])
                V[f"dd|{pan}|{p}|{m}"] = dd
                info["dd"][(pan, p, m)] = dd
    # ---- panel 11: exact p over all 11! relabellings (r and dd), dd CI, rule 7a, FP section 6
    idx = panels["11"]
    x = covs[("11", "arh")]
    xc = x - x.mean()
    cols, keys = [], []
    for p in PRES:
        for m in METRICS:
            for xx in ("raw", "s_cal", "unc"):
                w = linear_weights(grids[(p, m, xx)][idx], x)
                r = info["r"][("11", p, m, xx, "arh")]
                if w is not None and r is not None:
                    cols.append((w, r - TOL))
                    keys.append(f"arh|11|{p}|{m}|{xx}|arh|pge")
                else:
                    V[f"arh|11|{p}|{m}|{xx}|arh|pge"] = None
        wk = [linear_weights(grids[(p, "mknn", xx)][idx], x) for xx in ("raw", "s_cal")]
        for m in ("cka_biased", "cka_debiased"):
            wm = [linear_weights(grids[(p, m, xx)][idx], x) for xx in ("raw", "s_cal")]
            dd = info["dd"][("11", p, m)]
            if dd is None or any(w is None for w in wm + wk):
                V[f"dd|11|{p}|{m}|pge"] = None
            else:
                cols.append(((wm[0] - wm[1]) - (wk[0] - wk[1]), dd - TOL))
                keys.append(f"dd|11|{p}|{m}|pge")
    if cols and exact:
        import time
        t0 = time.time()
        W = np.stack([c[0] for c in cols], axis=1)
        thr = np.array([c[1] for c in cols])
        cnt, tot = exact_ge_counts(xc, W, thr)
        for k, c in zip(keys, cnt):
            V[k] = float(c) / float(tot)
        info["n_relabel"] = tot
        info["exact_secs"] = time.time() - t0
        log(f"exact relabelling p over {tot} relabellings, {len(keys)} statistics, {time.time() - t0:.0f}s")
    for p in PRES:
        for m in ("cka_biased", "cka_debiased"):
            G = {xx: grids[(p, mm, xx)][idx] for mm in (m,) for xx in ("raw", "s_cal")}
            Gk = {xx: grids[(p, "mknn", xx)][idx] for xx in ("raw", "s_cal")}
            db = delta_boot(G["raw"], G["s_cal"], Gk["raw"], Gk["s_cal"], x,
                            f"S1.2-layermax-openllama|boot|arh_delta|{p}|{m}", nv)
            V[f"dd|11|{p}|{m}|ci_lo"], V[f"dd|11|{p}|{m}|ci_hi"] = db["ci"]
            info["dd_boot"] = info.get("dd_boot", {})
            info["dd_boot"][(p, m)] = db
        lo, hi = V[f"dd|11|{p}|cka_biased|ci_lo"], V[f"dd|11|{p}|cka_biased|ci_hi"]
        V[f"rule7a|{p}"] = ("replicated the direction" if lo > 0 else
                            ("direction not established" if lo <= 0 <= hi else "opposite direction"))
        V[f"size_not_reproduced|{p}"] = bool(0.40 > hi)
        dd = info["dd"][("11", p, "cka_biased")]
        rac, rak = info["r"][("11", p, "cka_biased", "s_cal", "arh")], info["r"][("11", p, "mknn", "s_cal", "arh")]
        if dd is None or rac is None or rak is None:
            cat = "undefined"
        elif dd >= 0.20 and rac <= rak - 0.20:
            cat = "ARH pattern reappears"
        elif dd <= 0.10 or rac >= rak - 0.10:
            cat = "absent"
        else:
            cat = "partial"
        V[f"fp6|{p}|raw_cat"] = cat
        glo, ghi = gap_boot(grids[(p, "mknn", "s_cal")][idx], grids[(p, "cka_biased", "s_cal")][idx], x,
                            f"S1.2-layermax-openllama|boot|arh_gap|{p}", nv)
        V[f"fp6|{p}|gap_ci_lo"], V[f"fp6|{p}|gap_ci_hi"] = glo, ghi
        ke = bool((lo <= 0.10 <= hi) or (glo <= 0.10 <= ghi))
        V[f"fp6|{p}|kedge"] = ke
        V[f"fp6|{p}|cat"] = "not decidable at this panel size (knife edge)" if ke else cat
    return V, info
