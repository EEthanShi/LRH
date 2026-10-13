"""Pythia axis, implementation I: the verdict arithmetic (PYT-AXIS COPY and generalisation of
analysis/s12_layermax_openllama/indep/ol_verdicts.py, sha256 21515722..., sub-role pyt-code-I, S0b, 2026-10-10).
Pure functions on in-memory records; no file is read here (lmx_pyt_aggregate.py and vp8_legacy.py load the pair files
after the order gate and call these functions).

What changed against ol_verdicts.py (plan ANALYSIS_PLAN_PYTHIA.md sections 3, 5, 7, 8, 9; R.4 "new code"):
  * GENERIC PRIMITIVES (PYT-AXIS): trend() is generalised from 3 sizes to n sizes, with the exact enumeration of all
    n! relabellings of the size ranks (the same relabelling for every ViT; identity = row 0). dtrend() implements the
    common-ViT-set rule of plan 5.1 (mode "common": every difference uses V(X, Y), the ViTs valid for both scores)
    and the per-score exclusion of the OpenLLaMA code (mode "legacy", used only by V-P8). trend0() is the rho_v := 0
    sensitivity. boot_means() is the ViT bootstrap of plan 5.4 (2000 draws, rng = hrhv2 rng_from_key(key), one
    rng.integers(0, n, n) per draw over the common set in ViT registry order, value = mean of the fixed per-ViT
    delta-rho; VERDICTS_SPEC_PYT.md section 6).
  * OPENLLAMA LEGACY SECTIONS (V-P8 only): ol_verdicts() and ol_section_c() are the OpenLLaMA sections A, B, C with
    trend() replaced by the generic n-size trend and an optional common-set mode (V-P8(b)).
  * PYTHIA SECTIONS (PYT-AXIS): verdicts_pyt() writes the sections A (trends), B (order changes over the 357
    comparisons), C (premise rules, implementation checks, diagnostics D1-D6), H (H1 gatekeeper with the
    intersection-union p, Holm over {H2, H3, H4}, TOST incl. the exactly-degenerate case) and R (robustness) of the
    contract analysis/s12_layermax_pythia/VERDICTS_SPEC_PYT.md (S0b v1), with its key names (verdict_keys.py) and its
    operationalisations: thresholds compared without tolerance, relabelling counts with 1e-9, bootstrap counts with
    1e-9 on the conservative side, "|unc" key suffix for the uncensored versions, Holm ties broken H2, H3, H4, exact
    rational test of a zero delta-rho, the per-ViT sign rule and D3 applied only to ViTs without censoring ties (the
    plan's per-ViT inequality is not a theorem under censoring at 0; a raw-discordant pair tied at s_cal = 0 raises
    Spearman's rho). The contract puts the plan 7 category and P-P2b in forecast_pyt.py (sections K, F); this module
    also applies them itself and writes them as x|I|... extras, an independent cross-check never compared by V-P5.
Records: rec[p][lm][vit][met][Q] for Q in SUBSETS, and rec[p][lm][vit][met]["ctm"], with
  raw, s_cal, unc, c (c_alpha), mu (null mean), p, a, b, n_tied_max, LA, LB      (Q != ctm)
  c_entry, mu_entry, max_entry_mu                                                (Q == full)
  M, M_unc, TM, cM, muM (entry null mean at the M_unc-argmax), a, b              (ctm)
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
SIZES = ("3b", "7b", "13b")            # OpenLLaMA legacy sections only
KINDS = ("arh", "dense", "logp")
TAUS = (0.03, 0.06, 0.10, 0.15, 0.20)
DBS = (0.0025, 0.005, 0.0075, 0.01)
MAT = 0.10
DELTA_B = 0.005
TOL = 1e-9
NB = 2000
BAND = (0.0009, 0.0029)                # OpenLLaMA legacy band (P-O2a(i))
PERM6 = [tuple(float(v) for v in p) for p in itertools.permutations((1, 2, 3))]
ID6 = PERM6.index((1.0, 2.0, 3.0))
PAIRS3 = ((0, 1), (0, 2), (1, 2))          # (a, b) with a < b in size order: 3b-7b, 3b-13b, 7b-13b
NEW = ((0, 2), (1, 2))                     # the 34 new comparisons (a in {3b, 7b}, b = 13b)
PILOT = ((0, 1),)                          # the 17 pilot comparisons

# ------------------------------------------------------------------------------------------- PYT-AXIS constants
PYT_SIZES = ("70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b")
ALPHA = 0.05
BOOT = "S1.2-layermax-pythia|boot"                                   # plan 5.4 key prefix
BAND_COEF = {"prh_clip+l2": (0.001069, 0.012920, 0.0008),            # plan 9 B6: f(PR*) = a + b / sqrt(PR*), +- h
             "l2": (0.001551, 0.006843, 0.0018)}
FLAT_BAND = {"prh_clip+l2": (0.0010, 0.0040), "l2": (0.0010, 0.0045)}  # plan 9, sensitivity only
MU_BAND = (-0.0003, 0.0008)                                          # plan 9 B6, uncensored (secondary)
PR_FIT_LO = {"prh_clip+l2": 124.5, "l2": 79.6}                       # plan 7 M5
H_CURVE = (0.0004, 0.0006, 0.0008, 0.0010, 0.0015)                   # plan 5.6
MARGINS = (0.05, 0.10, 0.15)                                         # plan 5.6
TR_MAT = 0.75                                                        # plan 9 B4
LOSS_R, LOSS_C = 0.60, 0.40                                          # plan 7
FIVE_FROM = 2                                                        # plan 5.5(b): sizes >= 410M (index 2..6)


def fmt(v: float) -> str:
    return repr(float(v))


def f(v):
    """JSON value: Fraction -> float; None stays None; numpy scalars -> python."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
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


def ceil_frac(n: int, num: int, den: int) -> int:   # PYT-AXIS: ceil(num n / den) in integers
    return -((-num * n) // den)


# ------------------------------------------------------------------------------------------- section A (generic)
def avg_ranks(vals):
    """average ranks (1-based) of comparable exact values (= scipy.stats.rankdata 'average' for floats)."""
    return [1 + sum(1 for y in vals if y < x) + 0.5 * (sum(1 for y in vals if y == x) - 1) for x in vals]


_PERMS = {}


def perms(n: int) -> np.ndarray:   # PYT-AXIS: all n! relabellings of the size ranks 1..n, identity first
    if n not in _PERMS:
        _PERMS[n] = np.array(list(itertools.permutations(range(1, n + 1))), dtype=np.float64)
    return _PERMS[n]


def is_valid(c) -> bool:   # PYT-AXIS: finite and not all equal (plan 5.1)
    return all(finite(v) for v in c) and not all(v == c[0] for v in c)


def valid_set(cols) -> list:
    return [j for j, c in enumerate(cols) if is_valid(c)]


def trend(cols, idx=None):
    """PYT-AXIS generalisation of the OpenLLaMA trend(): cols = one list of n scores per ViT (size order). Spearman
    rho_v (Pearson on average ranks) for each of the n! relabellings sigma of the size ranks (same sigma for every ViT);
    Tr(sigma) = mean over the ViTs in idx (default: the valid ViTs of these scores). Identity = row 0."""
    n = len(cols[0]) if cols else 0
    valid = valid_set(cols) if idx is None else list(idx)
    out = {"n": n, "nconst": len(cols) - len(valid), "valid": valid}
    if not valid:
        out.update({"T": None, "obs": None, "rho": [None] * len(cols)})
        return out
    R = np.array([avg_ranks(cols[j]) for j in valid], dtype=np.float64)          # (nv, n)
    Rc = R - R.mean(axis=1, keepdims=True)
    X = perms(n)                                                                 # (n!, n)
    Xc = X - X.mean(axis=1, keepdims=True)
    num = Xc @ Rc.T
    den = np.sqrt((Xc ** 2).sum(axis=1))[:, None] * np.sqrt((Rc ** 2).sum(axis=1))[None, :]
    rho = num / den                                                              # (n!, nv)
    T = rho.mean(axis=1)
    rv = [None] * len(cols)
    for i, j in enumerate(valid):
        rv[j] = float(rho[0, i])
    out.update({"T": T, "obs": float(T[0]), "rho": rv})
    return out


def trend0(cols):
    """PYT-AXIS plan 5.1 sensitivity: rho_v := 0 for a constant (or non-finite) vector, mean over all ViTs."""
    n = len(cols[0]) if cols else 0
    X = perms(n)
    Xc = X - X.mean(axis=1, keepdims=True)
    tot = np.zeros(X.shape[0])
    for c in cols:
        if is_valid(c):
            r = np.array(avg_ranks(c), dtype=np.float64)
            rc = r - r.mean()
            tot += (Xc @ rc) / (np.sqrt((Xc ** 2).sum(axis=1)) * np.sqrt(float((rc ** 2).sum())))
    T = tot / len(cols)
    return {"T": T, "obs": float(T[0])}


def pvals(T, obs):
    if T is None or obs is None:
        return None, None, None
    n = float(len(T))   # PYT-AXIS: n! relabellings (6 for the OpenLLaMA sections)
    return (float(np.sum(T >= obs - TOL)) / n, float(np.sum(T <= obs + TOL)) / n,
            float(np.sum(np.abs(T) >= abs(obs) - TOL)) / n)


def dtrend(cols_x, cols_b, mode="common"):
    """PYT-AXIS: Tr(X) - Tr(base) under the same relabelling. mode "common": both on V(X, base) (plan 5.1, new rule);
    mode "legacy": each on its own valid set (the OpenLLaMA rule, V-P8 only). Returns None if a trend is undefined."""
    vx, vb = valid_set(cols_x), valid_set(cols_b)
    if mode == "common":
        sb = set(vb)
        V = [j for j in vx if j in sb]
        if not V:
            return None
        tx, tb = trend(cols_x, V), trend(cols_b, V)
    else:
        if not vx or not vb:
            return None
        tx, tb = trend(cols_x, vx), trend(cols_b, vb)
        V = None
    D = tx["T"] - tb["T"]
    out = {"D": D, "obs": float(D[0]), "tx": tx, "tb": tb, "V": V, "nv": (len(V) if V is not None else None)}
    if V is not None:
        out["drho"] = [tx["rho"][j] - tb["rho"][j] for j in V]
    return out


_BOOT_CACHE = {}


def boot_means(vec, key: str, nb: int = NB) -> np.ndarray:
    """PYT-AXIS plan 5.4 ViT bootstrap: 2000 draws; draw b resamples the entries of vec (the fixed per-ViT values of the
    common set, ViT registry order) with replacement, j = rng.integers(0, n, n); value = mean over the draw."""
    v = np.asarray(vec, dtype=np.float64)
    ck = (key, v.tobytes(), nb)
    if ck not in _BOOT_CACHE:
        from hrhv2 import util
        rng = util.rng_from_key(key)
        n = v.size
        out = np.empty(nb)
        for b in range(nb):
            j = rng.integers(0, n, n)
            out[b] = v[j].mean()
        _BOOT_CACHE[ck] = out
    return _BOOT_CACHE[ck]


def neartie(cols, pairs) -> int:
    n = 0
    for c in cols:
        for a, b in pairs:
            x, y = c[a], c[b]
            if finite(x) and finite(y) and abs(float(x) - float(y)) <= 1e-12 * max(abs(float(x)), abs(float(y))):
                n += 1
    return n


def cols_of(rec, lms, vits, met, q, x):
    return [[rec[lm][v][met][q][x] for lm in lms] for v in vits]


def size_pairs(n):   # PYT-AXIS: all (a, b) with a < b in size order (21 for n = 7)
    return [(a, b) for a in range(n) for b in range(a + 1, n)]


# =========================================================================================== OpenLLaMA legacy sections
# (ol_verdicts.py sections A, B, C, verbatim except: trend() is the generic n-size trend; "mode" selects the ViT set of
#  every difference; used only by V-P8 (vp8_legacy.py), never by the Pythia aggregate)
def ol_verdicts(rec, ol_lms, vits, partb, pooled, extra, mode="legacy"):
    """rec: OL records (rec[p][lm][vit][met][Q]); ol_lms: [3b, 7b, 13b] slugs; partb: parsed partB_features_ol.json;
    pooled: {"r": {(panel, p, met, x, kind): r}}; extra: dict for x| extras. Sections A, B, C."""
    V = {}
    TR = {}
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
                        V[f"neartie|{k}"] = neartie(cols, PAIRS3)
                    extra[f"x|rho|{k}"] = t["rho"]
                bq = "full" if q == "ctm" else q
                dxs = ("M", "M_unc", "TM") if q == "ctm" else ("s_cal", "unc")
                for x in dxs:
                    k = f"{p}|{m}|{q}|{x}"
                    dt = dtrend(cols_of(R, ol_lms, vits, m, q, x), cols_of(R, ol_lms, vits, m, bq, "raw"), mode)
                    if dt is None:
                        V[f"dtr|{k}"] = None
                        V[f"dtr_pge|{k}"] = V[f"dtr_ple|{k}"] = V[f"dtr_ptwo|{k}"] = None
                        continue
                    V[f"dtr|{k}"] = dt["obs"]
                    V[f"dtr_pge|{k}"], V[f"dtr_ple|{k}"], V[f"dtr_ptwo|{k}"] = pvals(dt["D"], dt["obs"])
            trm, trtm, trraw = (V[f"tr|{p}|{m}|ctm|M"], V[f"tr|{p}|{m}|ctm|TM"], V[f"tr|{p}|{m}|full|raw"])
            V[f"d3cal|{p}|{m}"] = None if trm is None or trtm is None else trm - trtm
            V[f"d3sel|{p}|{m}"] = None if trtm is None or trraw is None else trtm - trraw
    extra["_TR"] = TR
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
    for p in PRES:
        ol_section_c(V, rec[p], p, ol_lms, vits, partb, pooled, TR, extra)
    return V


def _lm(d, slug, label):
    if slug in d:
        return d[slug]
    if label in d:
        return d[label]
    raise KeyError(f"{slug} / {label} not in Part B table {list(d)[:5]}")


def p2cat(s, d, tr_raw, material, tau=MAT):
    """P-O2 category (OpenLLaMA spec section 5), decided in the spec's order."""
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


def ol_section_c(V, R, p, ol, vits, partb, pooled, TR, extra):
    pp = partb["per_pre"][p]
    lab = dict(zip(ol, SIZES))
    s = int(pp["s_O"])
    material = bool(pp["material"])
    nv = len(vits)
    g = lambda m, q, x: cols_of(R, ol, vits, m, q, x)  # noqa: E731
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
    for r in ("0.5", "1.0"):
        trr = V[f"tr|{p}|cka_biased|fixed_r{r}|raw"]
        for x in ("s_cal", "unc"):
            trx = V[f"tr|{p}|cka_biased|fixed_r{r}|{x}"]
            if trr is None or abs(trr) < MAT:
                desc = "not applied"
            else:
                sr = 1 if trr > 0 else -1
                tx = 0.0 if trx is None else trx
                if sr * tx >= MAT:
                    desc = "keeps sign"
                elif abs(tx) < MAT:
                    desc = "lost"
                else:
                    desc = "reversed"
            V[f"PO3|{p}|{r}|{x}|desc"] = desc
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
                        if x == "s_cal":
                            if Xa > 0 and Xb > 0:
                                lhs, rhs = Tb - Ta, (1 - Ta) * (tb - ta) / (1 - ta)
                                if (lhs >= rhs) != (Xb - Xa >= 0) and (exact or abs(lhs - rhs) > 1e-12):
                                    nk += 1
                        else:
                            lhs, rhs = Tb - Ta, tb - ta
                            if (lhs >= rhs) != (Xb - Xa >= 0) and (exact or abs(lhs - rhs) > 1e-12):
                                nk += 1
                        dr, dx = sgn(Tb - Ta), sgn(Xb - Xa)
                        if (tb > ta and dr <= 0 and dx == 1) or (tb < ta and dr >= 0 and dx == -1) or \
                                (tb == ta and dx == -dr and dx != 0):
                            ns += 1
                V[f"keepid|{p}|{m}|{q}|{x}|nviol"] = nk
                V[f"signrule|{p}|{m}|{q}|{x}|nviol"] = ns
    for x in ("s_cal", "unc"):
        d = V[f"dtr|{p}|cka_debiased|full|{x}"]
        V[f"PO4|{p}|{x}|holds"] = bool(d is not None and d >= -MAT)
    rb_, ra_ = pooled["r"][("11", p, "cka_debiased", "raw", "arh")], pooled["r"][("11", p, "cka_debiased", "s_cal", "arh")]
    V[f"PO4|{p}|arh|holds"] = bool(rb_ is not None and ra_ is not None and ra_ >= rb_ - 0.05)
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
    rM, rTM = TR[(p, "cka_biased", "ctm", "M")]["rho"], TR[(p, "cka_biased", "ctm", "TM")]["rho"]
    nviol = 0
    for v in range(nv):
        if cM[v][0] < cM[v][1] < cM[v][2] and rM[v] is not None and rTM[v] is not None and rM[v] - rTM[v] > 0:
            nviol += 1
    V[f"D3|{p}|nviol"] = nviol
    Tcols = g("cka_biased", "full", "raw")
    t_pl = trend([[Tcols[v][i] - Bbar[ol[i]][v] for i in range(3)] for v in range(nv)])
    V[f"plugin|{p}|tr"] = t_pl["obs"]
    tu = V[f"tr|{p}|cka_biased|full|unc"]
    V[f"plugin|{p}|diff"] = None if t_pl["obs"] is None or tu is None else t_pl["obs"] - tu
    V[f"plugin|{p}|within"] = bool(V[f"plugin|{p}|diff"] is not None and abs(V[f"plugin|{p}|diff"]) <= MAT)
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
    for m in METRICS:
        for i, lm in enumerate(ol):
            rs = [R[lm][v][m]["full"] for v in vits]
            V[f"D1|{p}|{m}|{SIZES[i]}"] = med([r["mu"] - r["max_entry_mu"] for r in rs])
            V[f"D2|{p}|{m}|{SIZES[i]}|argmax_rel"] = med([r["a"] / (r["LA"] - 1) for r in rs])
            V[f"D2|{p}|{m}|{SIZES[i]}|c_minus_centry"] = med([r["c"] - r["c_entry"] for r in rs])
        T = g(m, "full", "raw")
        V[f"frac7gt3|{p}|{m}"] = sum(1 for v in range(nv) if T[v][1] > T[v][0]) / nv


def pear(G, x):
    """aggregate_llama.py::pear (LLaMA E, OpenLLaMA plan R.3.6): np.corrcoef of the ravelled (LM x ViT) grid
    against the LM covariate repeated over ViTs; NaN for a constant or non-finite grid."""
    xx = np.repeat(x[:, None], G.shape[1], axis=1).ravel()
    y = G.ravel()
    if np.nanstd(y) == 0 or not np.all(np.isfinite(y)):
        return float("nan")
    return float(np.corrcoef(xx, y)[0, 1])


# =========================================================================================== PYT-AXIS: Pythia sections
# Key names and definitions: analysis/s12_layermax_pythia/VERDICTS_SPEC_PYT.md (S0b v1, sha256 2d5332eb...) and
# verdict_keys.py (6d20e04d...), sections A, B, C, H, R (here) and D, G (lmx_pyt_aggregate.py). Implementation I's own
# category, modifiers and P-P2b (the contract puts them in forecast_pyt.py, sections F and K) are written as x|I|...
# extras: an independent cross-check of forecast_pyt.py, never compared by V-P5.
XS = ("s_cal", "unc")
FAMILIES4 = ("in21k", "MAE", "DINOv2", "CLIP")
D4_STEPS = ((2, 3), (3, 4), (5, 6))          # 410m|1b, 1b|1.4b, 2.8b|6.9b
KEDGE_SWAP = 0.0021                          # spec 6: one adjacent swap at 17 ViTs (plan 5.6)


def band_ends(pre, pr_star, h=None):
    """plan 9 B6: band_i = [f(PR*_i) - h, f(PR*_i) + h], f = a + b / sqrt(PR*); h default = the frozen h."""
    a, b, h0 = BAND_COEF[pre]
    fv = a + b / math.sqrt(float(pr_star))
    hh = h0 if h is None else h
    return fv - hh, fv + hh, fv


def b4_material(B, lms_n, vits_n, delta_b=DELTA_B):
    """plan 9 B4 from a baseline table B[i][v] (size order): |Tr_P(B)| >= 0.75 - 1e-9 (spec 11) and the median over
    ViTs of s_B [B(largest, v) - B(smallest, v)] >= 3 delta_b. Returns (material, Tr, s_B, median gap). Used only for
    the x|I| cross-check of Part B's stored decisions."""
    t = trend([[B[i][v] for i in range(lms_n)] for v in range(vits_n)])
    tr = t["obs"]
    if tr is None:
        return False, None, 1, None
    s = 1 if tr >= 0 else -1
    gap = med([s * (B[lms_n - 1][v] - B[0][v]) for v in range(vits_n)])
    return bool(abs(tr) >= TR_MAT - 1e-9 and gap >= 3 * delta_b), tr, s, gap


class PartBSchemaError(KeyError):
    pass


def _req(d, name, where):
    if not isinstance(d, dict) or name not in d:
        raise PartBSchemaError(f"Part B {where} lacks '{name}' (keys {list(d)[:25] if isinstance(d, dict) else d})")
    return d[name]


def partb_view(pb, pre, lms, vits, tags):
    """the Part B quantities of one pre, by the field names of VERDICTS_SPEC_PYT.md section 11 (never recomputed)."""
    pp = _req(_req(pb, "per_pre", "top level"), pre, "per_pre")
    w = f"per_pre['{pre}']"
    n, nv = len(lms), len(vits)
    PRs, Bt, Bl = _req(pp, "PR_star", w), _req(pp, "Bbar", w), _req(pp, "Blate", w)
    band, ext = _req(pp, "band", w), _req(pp, "extrapolated", w)
    view = {
        "PR_star": [float(PRs[lm]) for lm in lms],
        "Bbar": [[float(Bt[lm][v]) for v in vits] for lm in lms],
        "Blate": [[float(Bl[lm][v]) for v in vits] for lm in lms],
        "material_Bbar": bool(_req(pp, "material_Bbar", w)), "s": int(_req(pp, "s", w)),
        "s_late": int(_req(pp, "s_late", w)), "H2_branch": str(_req(pp, "H2_branch", w)),
        "band": [(float(band[t]["lo"]), float(band[t]["hi"])) for t in tags],
        "band_flat": tuple(float(x) for x in _req(pp, "band_flat", w)),
        "extrapolated": [bool(ext[t]) for t in tags],
        "H2_same_as_H1": pp.get("H2_same_as_H1"),
        "mu_band": tuple(float(x) for x in _req(pb, "mu_band", "top level")),
        "band_h_curve": pp.get("band_h_curve"),
        "material_curve_db": pp.get("material_curve_db"),
        "tok1": {k: pp[k] for k in ("Tr_Bbar_tok1", "PR_star_tok1") if k in pp},
    }
    # x|I| cross-checks of Part B's stored decisions against the frozen rules applied to Part B's own tables
    mat, tr, s_d, _ = b4_material(view["Bbar"], n, nv)
    matl, trl, sl_d, _ = b4_material(view["Blate"], n, nv)
    view["xcheck"] = {"material_Bbar": mat == view["material_Bbar"], "s": s_d == view["s"],
                      "material_Blate_vs_branch": (("H2a" if matl else "H2b") == view["H2_branch"]),
                      "s_late": sl_d == view["s_late"],
                      "bands_formula": all(abs(view["band"][i][0] - band_ends(pre, view["PR_star"][i])[0]) <= 1e-12
                                           and abs(view["band"][i][1] - band_ends(pre, view["PR_star"][i])[1]) <= 1e-12
                                           for i in range(n)),
                      "extrapolated": all(view["extrapolated"][i] == (view["PR_star"][i] < PR_FIT_LO[pre])
                                          for i in range(n))}
    return view


def scal(T, c):
    """plan 9 B6: s_cal(T; c) = max((T - c)/(1 - c), 0)."""
    T, c = float(T), float(c)
    return max((T - c) / (1.0 - c), 0.0) if c < 1.0 else 0.0


def forecast_one(Tj, Tk, Bj, Bk, bj, bk, kind="s_cal"):
    """plan 9 B6 P-P2b forecast for one comparison (j < k = j'): 'below' (k under j), 'above', 'tie', 'undecided'.
    bj, bk = (lo, hi) offset ends of the two models' bands (for kind 'unc': the mu band)."""
    if kind == "s_cal":
        k_max, k_min = scal(Tk, Bk + bk[0]), scal(Tk, Bk + bk[1])
        j_max, j_min = scal(Tj, Bj + bj[0]), scal(Tj, Bj + bj[1])
    else:
        k_max, k_min = Tk - (Bk + bk[0]), Tk - (Bk + bk[1])
        j_max, j_min = Tj - (Bj + bj[0]), Tj - (Bj + bj[1])
    if k_max < j_min:
        return "below"
    if k_min > j_max:
        return "above"
    if kind == "s_cal" and k_max == 0.0 and j_max == 0.0:
        return "tie"
    return "undecided"


def observed_order(Sj, Sk):
    return "below" if Sk < Sj else ("above" if Sk > Sj else "tie")


def ppb(Tcols, Scols, Bbar, offs, kind, n, nv):
    """I's own application of the P-P2b rule over all comparisons (x|I| extras). Returns (counts, table)."""
    tab = {}
    nd = xc = nu = nt = flips = 0
    for v in range(nv):
        for a, b in size_pairs(n):
            dec = forecast_one(Tcols[v][a], Tcols[v][b], Bbar[a][v], Bbar[b][v], offs[a], offs[b], kind)
            obs = observed_order(Scols[v][a], Scols[v][b])
            ok = None if dec == "undecided" else bool(dec == obs)
            tab[(a, b, v)] = (dec, obs, ok)
            if dec == "undecided":
                nu += 1
            else:
                nd += 1
                xc += int(ok)
                nt += int(dec == "tie")
                flips += int(dec == "below" and float(Tcols[v][b]) > float(Tcols[v][a]))
    nw = nd - xc
    verdict = "fails" if nw > 0 else ("not evaluable" if nd < 30 else "holds")
    return {"n_decided": nd, "x_correct": xc, "n_wrong": nw, "n_undecided": nu, "n_tie_fc": nt,
            "n_flip_fc_rawconc": flips, "verdict": verdict}, tab


def holm(members, pv, alpha=ALPHA):
    """spec 6: sort ascending by p, ties by member order (H2, H3, H4); reject the i-th while p_(i) <= alpha/(3 - i)
    (no tolerance), stop at the first non-rejection; adjusted p_holm of the i-th = min(1, max_{k<=i} (m - k) p_(k))."""
    order = sorted(range(len(members)), key=lambda i: (pv[members[i]], i))
    m = len(members)
    rej, adj = {}, {}
    still, run = True, 0.0
    for r, i in enumerate(order):
        k = members[i]
        if still and pv[k] <= alpha / (m - r):
            rej[k] = True
        else:
            still = False
            rej[k] = False
        run = max(run, (m - r) * pv[k])
        adj[k] = min(1.0, run)
    return rej, adj, ",".join(members[i] for i in order)


def category(member, D1, p_perm, p_mag, ppa, ppb_v, R, C, tau=MAT, alpha=ALPHA):
    """plan 7 axis category (I's own, x|I| extras; thresholds without tolerance as spec 2/10).
    Returns (category, premise, F reason, confirmed)."""
    if not member:
        return "E", None, None, False
    if D1 is None:
        return "undefined (no ViT valid under both scores)", None, None, False
    if D1 >= tau:
        return "D", None, None, False
    if abs(D1) < tau:
        return "C", None, None, False
    confirmed = bool(max(p_perm, p_mag) <= alpha and D1 <= -tau)
    if not confirmed:
        sub = [r for r, c in (("not consistent across the seven sizes", p_perm > alpha),
                              ("size of the drop not established", p_mag > alpha)) if c]
        return "F", None, "; ".join(sub) if sub else None, False
    if ppa == "fails" or ppb_v == "fails":
        return "B", None, None, True
    lev = "A" if (R >= LOSS_R and C <= LOSS_C) else ("A-att" if R >= LOSS_R else "A-prime")
    return lev, ("verified" if (ppa == "holds" and ppb_v == "holds") else "unverified"), None, True


def _rank_corr(x, y):
    """Spearman over the sizes (Pearson on average ranks), None if constant."""
    rx, ry = np.array(avg_ranks(list(x)), float), np.array(avg_ranks(list(y)), float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return None if d == 0 else float((rx * ry).sum() / d)


def _num_S(ranks):
    """spec 2 exact rho: with integer size ranks 1..n and half-integer average ranks, rho = num / sqrt(S_x S_r);
    returns (num, S_r) as Fractions (the mean rank is (n + 1)/2 for both)."""
    n = len(ranks)
    m = Fraction(n + 1, 2)
    r = [Fraction(x) for x in ranks]
    return sum((Fraction(i + 1) - m) * (ri - m) for i, ri in enumerate(r)), sum((ri - m) ** 2 for ri in r)


def rho_equal_exact(cx, cb) -> bool:
    nx, sx = _num_S(avg_ranks(cx))
    nb, sb = _num_S(avg_ranks(cb))
    return sgn(nx) == sgn(nb) and nx * nx * sb == nb * nb * sx


def _sub(cols, sizes_idx=None, vit_idx=None):
    if sizes_idx is not None:
        cols = [[c[i] for i in sizes_idx] for c in cols]
    if vit_idx is not None:
        cols = [cols[j] for j in vit_idx]
    return cols


def _cj(seq):
    return ",".join(seq)


def verdicts_pyt(rec, lms, vits, pbv, aux, extra, pres, tags=PYT_SIZES):
    """sections A, B, C, H, R of VERDICTS_SPEC_PYT.md for the pres run, plus x|I| extras (I's own category,
    modifiers, readings and P-P2b). rec[p][lm][vit][met][Q]; lms in size order; pbv[p] = partb_view(...);
    aux = {"num_params", "width", "depth", "L_stored"} (Part B, size order), "vit_family" (VIT_FAMILY4 labels)."""
    V = {}
    n, nv = len(lms), len(vits)
    SP = size_pairs(n)
    for p in pres:
        R = rec[p]
        g = lambda m, q, x: cols_of(R, lms, vits, m, q, x)  # noqa: E731
        pb = pbv[p]
        s, s_late = int(pb["s"]), int(pb["s_late"])
        Bbar, Blate = pb["Bbar"], pb["Blate"]
        # ------------------------------------------------------------------ A: trends (spec 3)
        TR = {}
        for m in METRICS:
            for q in PROTOCOLS:
                for x in (("M", "M_unc", "TM") if q == "ctm" else ("raw", "s_cal", "unc")):
                    cols = g(m, q, x)
                    t = trend(cols)
                    TR[(m, q, x)] = t
                    k = f"{p}|{m}|{q}|{x}"
                    V[f"tr|{k}"], V[f"tr_nconst|{k}"] = t["obs"], t["nconst"]
                    V[f"tr_pge|{k}"], V[f"tr_ple|{k}"], V[f"tr_ptwo|{k}"] = pvals(t["T"], t["obs"])
                    V[f"tr0|{k}"] = trend0([c for c in cols if all(finite(y) for y in c)])["obs"]
                    if m != "mknn":
                        V[f"neartie|{k}"] = neartie(cols, SP)
                    extra[f"x|rho|{k}"] = t["rho"]
                bq = "full" if q == "ctm" else q
                for x in (("M", "M_unc", "TM") if q == "ctm" else ("s_cal", "unc")):
                    k = f"{p}|{m}|{q}|{x}"
                    cx, cb = g(m, q, x), g(m, bq, "raw")
                    dt = dtrend(cx, cb, "common")
                    fin = [j for j in range(nv) if all(finite(y) for y in cx[j]) and all(finite(y) for y in cb[j])]
                    V[f"dtr0|{k}"] = trend0([cx[j] for j in fin])["obs"] - trend0([cb[j] for j in fin])["obs"]
                    if dt is None:
                        for kk in ("dtr", "dtr_nv", "dtr_pge", "dtr_ple", "dtr_ptwo"):
                            V[f"{kk}|{k}"] = None if kk != "dtr_nv" else 0
                        continue
                    V[f"dtr|{k}"], V[f"dtr_nv|{k}"] = float(np.mean(dt["drho"])), dt["nv"]
                    V[f"dtr_pge|{k}"], V[f"dtr_ple|{k}"], V[f"dtr_ptwo|{k}"] = pvals(dt["D"], dt["obs"])
            d3 = dtrend(g(m, "ctm", "M"), g(m, "ctm", "TM"), "common")
            V[f"d3cal|{p}|{m}"] = None if d3 is None else float(np.mean(d3["drho"]))
            d3 = dtrend(g(m, "ctm", "TM"), g(m, "full", "raw"), "common")
            V[f"d3sel|{p}|{m}"] = None if d3 is None else float(np.mean(d3["drho"]))
        # ------------------------------------------------------------------ B: order changes, all 357 (spec 4)
        for m in METRICS:
            for q in PROTOCOLS:
                rq = "full" if q == "ctm" else q
                for x in (("M", "M_unc") if q == "ctm" else ("s_cal", "unc")):
                    raw, sc = g(m, rq, "raw"), g(m, q, x)
                    c = dict.fromkeys(("down", "up", "c2d", "d2c", "c2t", "t2c", "d2t", "t2d"), 0)
                    for v in range(nv):
                        for a, b in SP:
                            dr, dx = sgn(raw[v][b] - raw[v][a]), sgn(sc[v][b] - sc[v][a])
                            c["down"] += int(dx < dr)
                            c["up"] += int(dx > dr)
                            c["c2d"] += int(dr == 1 and dx == -1)
                            c["d2c"] += int(dr == -1 and dx == 1)
                            c["c2t"] += int(dr == 1 and dx == 0)
                            c["t2c"] += int(dr == 0 and dx == 1)
                            c["d2t"] += int(dr == -1 and dx == 0)
                            c["t2d"] += int(dr == 0 and dx == -1)
                    for kk, vv in c.items():
                        V[f"chg|{p}|{m}|{q}|{x}|all|{kk}"] = vv
                    N = nv * len(SP)
                    kk = f"fcnt|{p}|{m}|{q}|{x}"
                    V[f"{kk}|n"], V[f"{kk}|fminus"], V[f"{kk}|fplus"] = N, c["c2d"], c["d2c"]
                    V[f"{kk}|F"] = (c["c2d"] - c["d2c"]) / N
                    V[f"{kk}|fm_cp_lo"], V[f"{kk}|fm_cp_hi"] = cp95(c["c2d"], N)
                    V[f"{kk}|fp_cp_lo"], V[f"{kk}|fp_cp_hi"] = cp95(c["d2c"], N)
        # ------------------------------------------------------------------ C: Part B rules, checks, attribution
        Tb = g("cka_biased", "full", "raw")
        thr = {"s_cal": g("cka_biased", "full", "c"), "unc": g("cka_biased", "full", "mu")}
        base = Fraction(10, 1023)
        Tk = g("mknn", "full", "raw")
        for x, tk in (("s_cal", g("mknn", "full", "c")), ("unc", g("mknn", "full", "mu"))):
            X = g("mknn", "full", x)
            nq = nrev = ntie = 0
            for v in range(nv):
                for a, b in SP:
                    gap = Tk[v][b] - Tk[v][a]
                    if abs(gap) > abs(tk[v][a] - base) + abs(tk[v][b] - base):
                        nq += 1
                        dx = sgn(X[v][b] - X[v][a])
                        nrev += int(dx == -sgn(gap) and dx != 0)
                        ntie += int(dx == 0 and sgn(gap) != 0)
            V[f"PP1|{p}|{x}|nq"], V[f"PP1|{p}|{x}|nrev"], V[f"PP1|{p}|{x}|ntie"] = nq, nrev, ntie
            V[f"PP1|{p}|{x}|holds"] = nrev == 0
        V[f"PP1|{p}|holds"] = bool(V[f"PP1|{p}|s_cal|holds"] and V[f"PP1|{p}|unc|holds"])
        PPA, FAILM = {}, {}
        for x in XS:
            t_ = thr[x]
            nin, failing = 0, []
            for i in range(n):
                offs = [t_[v][i] - Bbar[i][v] for v in range(nv)]
                lo, hi = pb["band"][i] if x == "s_cal" else pb["mu_band"]
                md = med(offs)
                inb = bool(lo <= md <= hi)
                xin = sum(1 for o in offs if lo <= float(o) <= hi)
                for vv, o in zip(vits, offs):
                    V[f"PP2ai|{p}|{x}|offset|{tags[i]}|{vv}"] = o
                V[f"PP2ai|{p}|{x}|median|{tags[i]}"], V[f"PP2ai|{p}|{x}|lo|{tags[i]}"] = md, lo
                V[f"PP2ai|{p}|{x}|hi|{tags[i]}"], V[f"PP2ai|{p}|{x}|in_band|{tags[i]}"] = hi, inb
                V[f"PP2ai|{p}|{x}|x|{tags[i]}"] = xin
                V[f"PP2ai|{p}|{x}|cp_lo|{tags[i]}"], V[f"PP2ai|{p}|{x}|cp_hi|{tags[i]}"] = cp95(xin, nv)
                nin += int(inb)
                if not inb:
                    failing.append(i)
            V[f"PP2ai|{p}|{x}|n_models_in_band"] = nin
            V[f"PP2ai|{p}|{x}|holds"] = nin >= 6
            V[f"PP2ai|{p}|{x}|failing"] = _cj(tags[i] for i in failing)
            FAILM[x] = failing
            nn = xx = 0
            for v in range(nv):
                for a, b in SP:
                    gb = Bbar[b][v] - Bbar[a][v]
                    if abs(gb) >= DELTA_B:
                        nn += 1
                        xx += int(sgn(t_[v][b] - t_[v][a]) == sgn(gb))
            need, need67 = ceil_frac(nn, 9, 10), ceil_frac(nn, 2, 3)
            k = f"PP2aii|{p}|{x}"
            V[f"{k}|n"], V[f"{k}|x"], V[f"{k}|need"], V[f"{k}|need67"] = nn, xx, need, need67
            V[f"{k}|verdict"] = "not evaluable" if nn < 6 else ("holds" if xx >= need else "fails")
            V[f"{k}|verdict67"] = "not evaluable" if nn < 6 else ("holds" if xx >= need67 else "fails")
            V[f"{k}|cp_lo"], V[f"{k}|cp_hi"] = cp95(xx, nn)
            vi, vii = V[f"PP2ai|{p}|{x}|holds"], V[f"{k}|verdict"]
            PPA[x] = "fails" if (not vi or vii == "fails") else ("holds" if vii == "holds" else "not evaluable")
            V[f"PP2a|{p}|{x}|verdict"] = PPA[x]
            # P-P5a: c at the M-argmax (s_cal) or the entry null mean at the M_unc-argmax (unc)
            cM = g("cka_biased", "ctm", "cM" if x == "s_cal" else "muM")
            nn = xx = 0
            for v in range(nv):
                for a, b in SP:
                    gb = Blate[b][v] - Blate[a][v]
                    if abs(gb) >= DELTA_B:
                        nn += 1
                        xx += int(sgn(cM[v][b] - cM[v][a]) == sgn(gb))
            need = ceil_frac(nn, 2, 3)
            k = f"PP5a|{p}|{x}"
            V[f"{k}|n"], V[f"{k}|x"], V[f"{k}|need"] = nn, xx, need
            V[f"{k}|verdict"] = "not evaluable" if nn < 6 else ("holds" if xx >= need else "fails")
            V[f"{k}|cp_lo"], V[f"{k}|cp_hi"] = cp95(xx, nn)
            for r in ("0.5", "1.0"):
                trr, trx = V[f"tr|{p}|cka_biased|fixed_r{r}|raw"], V[f"tr|{p}|cka_biased|fixed_r{r}|{x}"]
                if trr is None or abs(trr) < MAT:
                    desc = "not applied"
                else:
                    sr = 1 if trr > 0 else -1
                    tx = 0.0 if trx is None else trx
                    desc = "keeps sign" if sr * tx >= MAT else ("lost" if abs(tx) < MAT else "reversed")
                V[f"PP3|{p}|{r}|{x}|desc"] = desc
        fl = pb["band_flat"]
        nin = 0
        for i in range(n):
            inb = bool(fl[0] <= V[f"PP2ai|{p}|s_cal|median|{tags[i]}"] <= fl[1])
            V[f"PP2ai|{p}|s_cal|flat|in_band|{tags[i]}"] = inb
            nin += int(inb)
        V[f"PP2ai|{p}|s_cal|flat|n_models_in_band"], V[f"PP2ai|{p}|s_cal|flat|holds"] = nin, nin >= 6
        # keep identity, sign rule per comparison and per ViT (spec 5; [plan-gap] censoring ties excluded)
        for m in METRICS:
            exact = m == "mknn"
            for q in SUBSETS:
                T_ = g(m, q, "raw")
                for x, th_ in (("s_cal", g(m, q, "c")), ("unc", g(m, q, "mu"))):
                    X = g(m, q, x)
                    nk = ns = 0
                    for v in range(nv):
                        for a, b in SP:
                            Ta, Tb_, ta, tb, Xa, Xb = T_[v][a], T_[v][b], th_[v][a], th_[v][b], X[v][a], X[v][b]
                            if x == "s_cal":
                                if Xa > 0 and Xb > 0:
                                    lhs, rhs = Tb_ - Ta, (1 - Ta) * (tb - ta) / (1 - ta)
                                    if (lhs >= rhs) != (Xb - Xa >= 0) and (exact or abs(lhs - rhs) > 1e-12):
                                        nk += 1
                            else:
                                lhs, rhs = Tb_ - Ta, tb - ta
                                if (lhs >= rhs) != (Xb - Xa >= 0) and (exact or abs(lhs - rhs) > 1e-12):
                                    nk += 1
                            dr, dx = sgn(Tb_ - Ta), sgn(Xb - Xa)
                            if (tb > ta and dr <= 0 and dx == 1) or (tb < ta and dr >= 0 and dx == -1) or \
                                    (tb == ta and dx == -dr and dx != 0):
                                ns += 1
                    V[f"keepid|{p}|{m}|{q}|{x}|nviol"] = nk
                    V[f"signrule|{p}|{m}|{q}|{x}|nviol"] = ns
                    cnt = dict.fromkeys(("nmono_up", "nviol_up", "nmono_down", "nviol_down", "nexcl_tie"), 0)
                    for v in sorted(set(valid_set(T_)) & set(valid_set(X))):
                        tv = [th_[v][i] for i in range(n)]
                        up = all(tv[i] <= tv[i + 1] for i in range(n - 1))
                        down = all(tv[i] >= tv[i + 1] for i in range(n - 1))
                        if not (up or down):
                            continue
                        if any(X[v][a] == X[v][b] and T_[v][a] != T_[v][b] for a, b in SP):
                            cnt["nexcl_tie"] += 1
                            continue
                        rx = trend([X[v]])["obs"]
                        rr = trend([T_[v]])["obs"]
                        if up:
                            cnt["nmono_up"] += 1
                            cnt["nviol_up"] += int(rx > rr + 1e-12)
                        if down:
                            cnt["nmono_down"] += 1
                            cnt["nviol_down"] += int(rx < rr - 1e-12)
                    for kk, vv in cnt.items():
                        V[f"signvit|{p}|{m}|{q}|{x}|{kk}"] = vv
        Mc, TMc, cMc = g("cka_biased", "ctm", "M"), g("cka_biased", "ctm", "TM"), g("cka_biased", "ctm", "cM")
        nvi = nex = 0
        for v in sorted(set(valid_set(Mc)) & set(valid_set(TMc))):
            if not all(cMc[v][i] < cMc[v][i + 1] for i in range(n - 1)):
                continue
            if any(Mc[v][a] == Mc[v][b] and TMc[v][a] != TMc[v][b] for a, b in SP):
                nex += 1
                continue
            nvi += int(trend([Mc[v]])["obs"] - trend([TMc[v]])["obs"] > 1e-12)
        V[f"D3|{p}|nviol"], V[f"D3|{p}|nexcl_tie"] = nvi, nex
        tb_ = [[Tb[v][i] - Bbar[i][v] for i in range(n)] for v in range(nv)]
        V[f"plugin|{p}|tr"] = trend(tb_)["obs"]
        dp = dtrend(tb_, g("cka_biased", "full", "unc"), "common")
        V[f"plugin|{p}|diff"] = None if dp is None else float(np.mean(dp["drho"]))
        V[f"plugin|{p}|nv"] = 0 if dp is None else dp["nv"]
        V[f"plugin|{p}|within"] = bool(V[f"plugin|{p}|diff"] is not None and abs(V[f"plugin|{p}|diff"]) <= MAT)
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
        for m in METRICS:
            for i, lm in enumerate(lms):
                rs = [R[lm][v][m]["full"] for v in vits]
                V[f"D1|{p}|{m}|{tags[i]}"] = med([r["mu"] - r["max_entry_mu"] for r in rs])
                V[f"D2|{p}|{m}|{tags[i]}|argmax_rel"] = med([r["a"] / (r["LA"] - 1) for r in rs])
                V[f"D2|{p}|{m}|{tags[i]}|c_minus_centry"] = med([r["c"] - r["c_entry"] for r in rs])
        cb_ = thr["s_cal"]
        qty = {"PRstar": pb["PR_star"], "Bbar": [float(np.mean(Bbar[i])) for i in range(n)],
               "calpha": [float(np.mean([float(cb_[v][i]) for v in range(nv)])) for i in range(n)],
               "T": [float(np.mean([float(Tb[v][i]) for v in range(nv)])) for i in range(n)]}
        axes = {"logparams": [math.log(float(x)) for x in aux["num_params"]], "width": aux["width"],
                "depth": aux["depth"]}
        for qn, qv in qty.items():
            for an, av in axes.items():
                V[f"D4|{p}|{qn}|{an}"] = _rank_corr(av, qv)
            for a_, b_ in D4_STEPS:
                V[f"D4|{p}|{qn}|step|{tags[a_]}|{tags[b_]}"] = qv[b_] - qv[a_]
        for i in range(n - 1):
            V[f"D5|{p}|{tags[i]}|{tags[i + 1]}|raw_gain"] = med([float(Tb[v][i + 1]) - float(Tb[v][i]) for v in range(nv)])
            V[f"D5|{p}|{tags[i]}|{tags[i + 1]}|thr_gain"] = med([float(cb_[v][i + 1]) - float(cb_[v][i])
                                                                 for v in range(nv)])
        dl = math.log2(float(aux["num_params"][-1]) / float(aux["num_params"][0]))
        V[f"D5|{p}|g"] = med([(float(Tb[v][-1]) - float(Tb[v][0])) / dl for v in range(nv)])
        V[f"D5|{p}|h"] = med([(float(cb_[v][-1]) - float(cb_[v][0])) / dl for v in range(nv)])
        for mm, b0 in (("mknn", Fraction(10, 1023)), ("cka_debiased", 0.0)):
            cc = g(mm, "full", "c")
            cmb = [med([cc[v][i] - b0 for v in range(nv)]) for i in range(n)]
            for i in range(n):
                V[f"D6|{p}|{mm}|{tags[i]}|c_minus_base"] = cmb[i]
            V[f"D6|{p}|{mm}|spearman_L"] = _rank_corr(aux["L_stored"], cmb)
            V[f"D6|{p}|{mm}|step|{tags[2]}|{tags[3]}"] = cmb[3] - cmb[2]
        # ------------------------------------------------------------------ H: the confirmatory family (spec 6)
        Hx = {}
        for x in XS:
            sfx = "" if x == "s_cal" else "|unc"
            h = f"H|{p}|{x}"
            Xm = "M" if x == "s_cal" else "M_unc"

            def drawp(drho, key, sg):
                return sg * boot_means(drho, key)

            def tost(m_, q_):
                cx = g(m_, q_, x if q_ == "full" else Xm)
                cb = g(m_, "full", "raw")
                dt = dtrend(cx, cb, "common")
                if dt is None:
                    return None
                Dd = float(np.mean(dt["drho"]))
                dr = boot_means(dt["drho"], f"{BOOT}|tost|{m_}|{q_}|{p}{sfx}")
                out = {"D": Dd, "nV": dt["nv"],
                       "degenerate": all(rho_equal_exact(cx[j], cb[j]) for j in dt["V"]),
                       "ranks_identical": all(avg_ranks(cx[j]) == avg_ranks(cb[j]) for j in dt["V"])}
                for mg in MARGINS:
                    plo = (1.0 + float(np.sum(dr <= -mg + TOL))) / (NB + 1)
                    phi = (1.0 + float(np.sum(dr >= mg - TOL))) / (NB + 1)
                    pt = max(plo, phi) if (-mg < Dd < mg) else 1.0
                    out[f"p_tost|m={fmt(mg)}"] = pt
                    if mg == MAT:
                        out["p_lo"], out["p_hi"], out["p"] = plo, phi, pt
                return out
            TO = {}
            for m_ in METRICS:
                for q_ in ("full", "ctm"):
                    t_ = TO[(m_, q_)] = tost(m_, q_)
                    for fld in ("D", "nV", "p_lo", "p_hi", "p", "degenerate", "ranks_identical"):
                        V[f"TOST|{p}|{x}|{m_}|{q_}|{fld}"] = None if t_ is None else t_[fld]

            def iut(cx, cb, sg, key, sizes_idx=None, vit_idx=None):
                dt = dtrend(_sub(cx, sizes_idx, vit_idx), _sub(cb, sizes_idx, vit_idx), "common")
                if dt is None:
                    return None
                Dd = sg * float(np.mean(dt["drho"]))
                Dp = sg * dt["D"]
                out = {"D": Dd, "nV": dt["nv"], "p_perm": float(np.sum(Dp <= Dd + TOL)) / float(len(Dp)),
                       "R": dt["tb"]["obs"], "C": dt["tx"]["obs"]}
                if key is not None:
                    dr = drawp(dt["drho"], key, sg)
                    for mg in MARGINS:
                        out[f"p_mag|m={fmt(mg)}"] = (1.0 + float(np.sum(dr >= -mg - TOL))) / (NB + 1)
                    out["p_mag"] = out[f"p_mag|m={fmt(MAT)}"]
                    out["p"] = max(out["p_perm"], out["p_mag"])
                    lo_, hi_ = np.percentile(dr, [2.5, 97.5])
                    out["ci_lo"], out["ci_hi"] = float(lo_), float(hi_)
                return out
            cxH1, cbH1 = g("cka_biased", "full", x), g("cka_biased", "full", "raw")
            k1 = f"{BOOT}|H1mag|cka_biased|full|{p}{sfx}"
            H1 = iut(cxH1, cbH1, s, k1)
            mem = bool(pb["material_Bbar"])
            V[f"{h}|H1|member"], V[f"{h}|H1|s"] = mem, s
            for fld in ("D", "nV", "R", "C", "p_perm", "p_mag", "p", "ci_lo", "ci_hi"):
                V[f"{h}|H1|{fld}"] = None if H1 is None else H1[fld]
            if H1 is None:
                V[f"{h}|H1|nV"] = 0
            for mg in MARGINS:
                V[f"{h}|H1|p_mag|m={fmt(mg)}"] = None if H1 is None else H1[f"p_mag|m={fmt(mg)}"]
            conf1 = None if not mem else bool(H1 is not None and H1["p"] <= ALPHA and H1["D"] <= -MAT)
            V[f"{h}|H1|confirmed"] = conf1
            if H1 is None:
                V[f"{h}|H1|kedge"] = None
            else:
                D1 = H1["D"]
                V[f"{h}|H1|kedge"] = bool(min(abs(D1 + MAT), abs(D1 - MAT)) <= KEDGE_SWAP + 1e-9
                                          or H1["ci_lo"] <= -MAT <= H1["ci_hi"] or H1["ci_lo"] <= MAT <= H1["ci_hi"])
            br = pb["H2_branch"]
            H2 = iut(g("cka_biased", "ctm", Xm), cbH1, s_late, f"{BOOT}|H2mag|cka_biased|ctm|{p}{sfx}")
            V[f"{h}|H2|branch"], V[f"{h}|H2|s_late"] = br, s_late
            for fld in ("D", "nV", "p_perm", "p_mag"):
                V[f"{h}|H2|{fld}"] = None if H2 is None else H2[fld]
            if H2 is None:
                V[f"{h}|H2|nV"] = 0
            for mg in MARGINS:
                V[f"{h}|H2|p_mag|m={fmt(mg)}"] = None if H2 is None else H2[f"p_mag|m={fmt(mg)}"]
            t2 = TO[("cka_biased", "ctm")]
            for fld in ("p_lo", "p_hi", "degenerate", "ranks_identical"):
                V[f"{h}|H2|{fld}"] = None if t2 is None else t2[fld]
            for mg in MARGINS:
                V[f"{h}|H2|p_tost|m={fmt(mg)}"] = None if t2 is None else t2[f"p_tost|m={fmt(mg)}"]
            pH2 = (None if H2 is None else H2["p"]) if br == "H2a" else (None if t2 is None else t2["p"])
            V[f"{h}|H2|p"] = pH2
            for hy, mq in (("H3", ("mknn", "full")), ("H4", ("cka_debiased", "full"))):
                t_ = TO[mq]
                for fld in ("D", "nV", "p_lo", "p_hi", "p", "degenerate", "ranks_identical"):
                    V[f"{h}|{hy}|{fld}"] = None if t_ is None else t_[fld]
                for mg in MARGINS:
                    V[f"{h}|{hy}|p_tost|m={fmt(mg)}"] = None if t_ is None else t_[f"p_tost|m={fmt(mg)}"]
            gate_open = (not mem) or bool(conf1)
            V[f"{h}|gate"] = ("open (H1 confirmed)" if (mem and conf1) else
                              ("open (H1 not a member)" if not mem else "closed"))
            pv = {"H2": pH2 if pH2 is not None else 1.0, "H3": V[f"{h}|H3|p"] if V[f"{h}|H3|p"] is not None else 1.0,
                  "H4": V[f"{h}|H4|p"] if V[f"{h}|H4|p"] is not None else 1.0}
            if gate_open:
                rej, adj, order = holm(["H2", "H3", "H4"], pv)
            else:
                rej, adj, order = {}, {}, None
            V[f"{h}|holm_order"] = order
            for hy in ("H2", "H3", "H4"):
                V[f"{h}|{hy}|tested"] = bool(gate_open)
                V[f"{h}|{hy}|p_holm"] = adj.get(hy) if gate_open else None
                V[f"{h}|{hy}|holm_reject"] = rej.get(hy) if gate_open else None
                if not gate_open:
                    cf = None
                elif hy == "H2" and br == "H2a":
                    cf = bool(rej["H2"] and H2 is not None and H2["D"] <= -MAT)
                else:
                    cf = bool(rej[hy])
                V[f"{h}|{hy}|confirmed"] = cf
            Hx[x] = {"H1": H1, "H2": H2, "TO": TO, "mem": mem, "conf1": conf1, "gate_open": gate_open}
        # ------------------------------------------------------------------ R: robustness (spec 7)
        Scol = g("cka_biased", "full", "s_cal")
        V[f"R|{p}|n_scal_zero"] = sum(1 for v in range(nv) for i in range(n) if Scol[v][i] == 0)
        for i in range(n):
            V[f"R|{p}|median_scal|{tags[i]}"] = med([Scol[v][i] for v in range(nv)])
        fams = aux["vit_family"]
        for x in XS:
            sfx = "" if x == "s_cal" else "|unc"
            r_ = f"R|{p}|{x}"
            H1 = Hx[x]["H1"]
            cx, cb = g("cka_biased", "full", x), g("cka_biased", "full", "raw")
            k1 = f"{BOOT}|H1mag|cka_biased|full|{p}{sfx}"

            def dsub(m_, q_, sizes_idx=None, vit_idx=None):
                dt = dtrend(_sub(g(m_, q_, x), sizes_idx, vit_idx), _sub(g(m_, q_, "raw"), sizes_idx, vit_idx), "common")
                return None if dt is None else float(np.mean(dt["drho"]))

            def iut2(sizes_idx=None, vit_idx=None, key=None):
                dt = dtrend(_sub(cx, sizes_idx, vit_idx), _sub(cb, sizes_idx, vit_idx), "common")
                if dt is None:
                    return None
                Dd = s * float(np.mean(dt["drho"]))
                Dp = s * dt["D"]
                out = {"D1": Dd, "p_perm": float(np.sum(Dp <= Dd + TOL)) / float(len(Dp))}
                if key is not None:
                    dr = s * boot_means(dt["drho"], key)
                    out["p_mag"] = (1.0 + float(np.sum(dr >= -MAT - TOL))) / (NB + 1)
                    out["p"] = max(out["p_perm"], out["p_mag"])
                return out
            frag = []
            for i in range(n):
                keep = [j for j in range(n) if j != i]
                o = iut2(sizes_idx=keep, key=f"{k1}|loo|{tags[i]}")
                for fld in ("D1", "p_perm", "p_mag", "p"):
                    V[f"{r_}|loo|{tags[i]}|{fld}"] = None if o is None else o[fld]
                V[f"{r_}|loo|{tags[i]}|dtr_mknn"] = dsub("mknn", "full", sizes_idx=keep)
                V[f"{r_}|loo|{tags[i]}|dtr_dcka"] = dsub("cka_debiased", "full", sizes_idx=keep)
                if o is not None and H1 is not None and ((o["D1"] <= -MAT) != (H1["D"] <= -MAT)
                                                         or (o["p"] <= ALPHA) != (H1["p"] <= ALPHA)):
                    frag.append(tags[i])
            V[f"{r_}|fragile"], V[f"{r_}|fragile_sizes"] = bool(frag), _cj(frag)
            o5 = iut2(sizes_idx=list(range(FIVE_FROM, n)))
            V[f"{r_}|five|D1"] = None if o5 is None else o5["D1"]
            V[f"{r_}|five|p_perm"] = None if o5 is None else o5["p_perm"]
            if o5 is None or H1 is None:
                V[f"{r_}|five|carried"] = V[f"{r_}|five|carried_D"] = None
            else:
                d5, d7 = o5["D1"], H1["D"]
                V[f"{r_}|five|carried"] = bool(sgn(d5) != sgn(d7) or (d5 <= -MAT) != (d7 <= -MAT))
                V[f"{r_}|five|carried_D"] = bool(sgn(d5) != sgn(d7) or (d5 >= MAT) != (d7 >= MAT))
            gd = V[f"dtr|{p}|cka_biased|grid5|{x}"]
            V[f"{r_}|grid5|D1"] = None if gd is None else s * gd
            V[f"{r_}|grid5|dtr_mknn"] = V[f"dtr|{p}|mknn|grid5|{x}"]
            V[f"{r_}|grid5|dtr_dcka"] = V[f"dtr|{p}|cka_debiased|grid5|{x}"]
            for fam in FAMILIES4:
                keepv = [j for j in range(nv) if fams[j] != fam]
                o = iut2(vit_idx=keepv)
                V[f"{r_}|lofo|{fam}|D1"] = None if o is None else o["D1"]
                V[f"{r_}|lofo|{fam}|dtr_mknn"] = dsub("mknn", "full", vit_idx=keepv)
                V[f"{r_}|lofo|{fam}|dtr_dcka"] = dsub("cka_debiased", "full", vit_idx=keepv)
            V[f"{r_}|rho0|D1"] = s * V[f"dtr0|{p}|cka_biased|full|{x}"]
        # ------------------------------------------------------------------ x|I|: I's own K and F (cross-check only)
        for x in XS:
            hx = Hx[x]
            H1, H2 = hx["H1"], hx["H2"]
            offs = [tuple(b) for b in pb["band"]] if x == "s_cal" else [pb["mu_band"]] * n
            cnt, tab = ppb(Tb, g("cka_biased", "full", x), Bbar, offs, x, n, nv)
            for kk, vv in cnt.items():
                extra[f"x|I|PP2b|{p}|{x}|{kk}"] = vv
            for (a, b, v), (dec, obs, ok) in tab.items():
                extra[f"x|I|PP2b|{p}|{x}|dec|{tags[a]}|{tags[b]}|{vits[v]}"] = dec
                extra[f"x|I|PP2b|{p}|{x}|ok|{tags[a]}|{tags[b]}|{vits[v]}"] = ok
            if x == "s_cal" and pb.get("band_h_curve"):
                for hk, hb in pb["band_h_curve"].items():
                    c2, _ = ppb(Tb, g("cka_biased", "full", x), Bbar, [tuple(hb[t]) for t in tags], x, n, nv)
                    extra[f"x|I|PP2b|{p}|s_cal|h={hk}|n_decided"] = c2["n_decided"]
                    extra[f"x|I|PP2b|{p}|s_cal|h={hk}|x_correct"] = c2["x_correct"]
            if H1 is None:
                cat = ("undefined (no ViT valid under both scores)", None, None, False)
            else:
                cat = category(hx["mem"], H1["D"], H1["p_perm"], H1["p_mag"], V[f"PP2a|{p}|{x}|verdict"],
                               cnt["verdict"], H1["R"], H1["C"])
            c = f"x|I|K|{p}|{x}"
            extra[f"{c}|category"], extra[f"{c}|premise"], extra[f"{c}|F_reason"] = cat[0], cat[1], cat[2]
            for tau in TAUS:
                extra[f"{c}|curve|tau={fmt(tau)}"] = (None if H1 is None else
                                                      category(hx["mem"], H1["D"], H1["p_perm"], H1["p_mag"],
                                                               V[f"PP2a|{p}|{x}|verdict"], cnt["verdict"], H1["R"],
                                                               H1["C"], tau=tau)[0])
            if pb.get("material_curve_db"):
                for dk, dm in pb["material_curve_db"].get("Bbar", {}).items():
                    extra[f"{c}|curve|db={dk}"] = (None if H1 is None else
                                                   category(bool(dm), H1["D"], H1["p_perm"], H1["p_mag"],
                                                            V[f"PP2a|{p}|{x}|verdict"], cnt["verdict"], H1["R"],
                                                            H1["C"])[0])
            extra[f"{c}|M1"] = V[f"R|{p}|{x}|fragile"]
            g5 = V[f"R|{p}|{x}|grid5|D1"]
            extra[f"{c}|M2"] = (None if cat[0] not in ("A", "A-att", "A-prime") else
                                bool(g5 is None or not g5 <= -MAT))
            extra[f"{c}|M3"] = V[f"R|{p}|{x}|five|carried_D"] if cat[0] == "D" else V[f"R|{p}|{x}|five|carried"]
            med0 = [tags[i] for i in range(n) if V[f"R|{p}|median_scal|{tags[i]}"] == 0]
            extra[f"{c}|M4"] = (None if x != "s_cal" else bool(V[f"R|{p}|n_scal_zero"] > n * nv / 3 or med0))
            extra[f"{c}|M4_sizes"] = _cj(med0) if x == "s_cal" else None
            ext = [i for i in range(n) if pb["extrapolated"][i]]
            extra[f"{c}|M5_extrap_sizes"] = _cj(tags[i] for i in ext)
            if cat[0] == "B":
                wrong_ok = all((a in ext or b in ext) for (a, b, v), (_, _, ok) in tab.items() if ok is False)
                extra[f"{c}|M5_B"] = bool(V[f"PP2aii|{p}|{x}|verdict"] != "fails"
                                          and all(i in ext for i in FAILM[x]) and wrong_ok)
            else:
                extra[f"{c}|M5_B"] = None
            inb_ext = [tags[i] for i in ext if V[f"PP2ai|{p}|{x}|in_band|{tags[i]}"]]
            extra[f"{c}|M5_A"] = bool(inb_ext) if cat[0] in ("A", "A-att", "A-prime") else None
            extra[f"{c}|M5_A_sizes"] = _cj(inb_ext) if cat[0] in ("A", "A-att", "A-prime") else None
            h = f"H|{p}|{x}"
            for hy in ("H3", "H4"):
                if V[f"{h}|{hy}|D"] is None:
                    rd = "undefined"
                elif abs(V[f"{h}|{hy}|D"]) >= MAT:
                    rd = "moved"
                elif V[f"{h}|{hy}|degenerate"]:
                    rd = "no comparison changed order"
                elif not V[f"{h}|{hy}|tested"]:
                    rd = "kept, not tested (gate closed)"
                elif V[f"{h}|{hy}|holm_reject"]:
                    rd = "equivalence confirmed"
                else:
                    rd = "kept, equivalence not established"
                extra[f"{c}|{hy}_reading"] = rd
            br = pb["H2_branch"]
            D2 = None if H2 is None else H2["D"]
            sfx_ = "" if V[f"{h}|H2|tested"] else " (not tested, gate closed)"
            if V[f"PP5a|{p}|{x}|verdict"] == "fails":
                r2 = "premise failed"
            elif V[f"{h}|H2|confirmed"]:
                r2 = "follows the late baselines" if br == "H2a" else "kept as predicted"
            elif D2 is None:
                r2 = "undefined"
            elif D2 >= MAT:
                r2 = "moved with the late baselines" + sfx_
            elif abs(D2) < MAT:
                r2 = "kept" + sfx_
            else:
                r2 = "drop not confirmed" + sfx_
            extra[f"{c}|H2_reading"] = r2
        for kk, vv in pb["xcheck"].items():
            extra[f"x|I|partb_xcheck|{p}|{kk}"] = vv
        for kk, vv in pb["tok1"].items():
            extra[f"x|I|D7|{p}|{kk}"] = vv
    return V
