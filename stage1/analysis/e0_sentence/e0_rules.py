"""E0 frozen constants and decision rules (ANALYSIS_PLAN_E0.md v1.0 §7).

Every constant is copied from analysis/e0_sentence/freeze/freeze_constants.json (sha256 144d5aef...), which
the freeze role recomputed from the frozen S1.1 outputs; tests/test_e0_rules.py checks equality to the bit.
Pure functions only (numpy/scipy); no file access.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------------------------- constants
# (pre, k, m_ref) -> (a, b, s_res): OLS of the S1.1 slope b(k) on ln L(k, m_ref) over the 29 primary pairs
LINES = {
    ("clip", 1, 1024): (0.22695755671235662, 0.16082096181112826, 0.11040767125138587),
    ("clip", 1, 2048): (0.201770459969184, 0.14302877788873172, 0.11410318812485032),
    ("clip", 10, 1024): (0.1357059076472474, 0.22805384255107658, 0.05647474462525915),
    ("clip", 10, 2048): (0.12371552579584805, 0.19995975623095777, 0.050145537809852786),
    ("clip", 100, 1024): (0.13600573214151732, 0.407352514400648, 0.04473493676211222),
    ("clip", 100, 2048): (0.11416316513893157, 0.32403871254694294, 0.03561326998512506),
    ("l2", 1, 1024): (0.16019524474322946, 0.1286944348118444, 0.1271717019011142),
    ("l2", 1, 2048): (0.14521825632160074, 0.11518399492040504, 0.1268884056437313),
    ("l2", 10, 1024): (0.1293442135792158, 0.2190581393678084, 0.047656946995276415),
    ("l2", 10, 2048): (0.11387492946459479, 0.1907084442028394, 0.04341619559293701),
    ("l2", 100, 1024): (0.1278709977565826, 0.3924448215688186, 0.04566275164818839),
    ("l2", 100, 2048): (0.10623466545569216, 0.31171473228971375, 0.03516441690669941),
}
# (pre, k) -> ViT -> mean S1.1 vision-language slope b(k) of that ViT (competitor beta_M,v)
COMPETITOR = {
    ("clip", 1): {"augreg_b": -0.49500657522072283, "clip_b": -0.5090410345493782, "dinov2_b": -0.6314249730414413,
                  "mae_b": -0.45968819524021504, "siglip_b": -0.5443540775315643},
    ("clip", 10): {"augreg_b": -0.5752592991292489, "clip_b": -0.5499745403938117, "dinov2_b": -0.5890678384804802,
                   "mae_b": -0.6444729401596141, "siglip_b": -0.511091295171744},
    ("clip", 100): {"augreg_b": -0.565428934032948, "clip_b": -0.6488232012329866, "dinov2_b": -0.5932928956718279,
                    "mae_b": -0.6549401667177925, "siglip_b": -0.4569408960039796},
    ("l2", 1): {"augreg_b": -0.4123673849566633, "clip_b": -0.371315141075734, "dinov2_b": -0.6143294436695782,
                "mae_b": -0.40352394324793445, "siglip_b": -0.5657884808694834},
    ("l2", 10): {"augreg_b": -0.567407713593515, "clip_b": -0.555277521714899, "dinov2_b": -0.590210238450669,
                 "mae_b": -0.6411644106143516, "siglip_b": -0.5163547462375035},
    ("l2", 100): {"augreg_b": -0.5827976853574751, "clip_b": -0.6666895808546708, "dinov2_b": -0.610620902553134,
                  "mae_b": -0.677117601853575, "siglip_b": -0.4849175078773981},
}
BETA_M_CONST = -0.570555396995647          # S1.1 VL median slope at k = 10 (clip); descriptive Wilcoxon only
N1_OWN_LINE = (0.033, 0.219)               # critic's number (B1d), reference curve only (P3j)
WITHIN_CLASS_K10 = {"cross-modal": 0.09341001514804105, "cross-vision": 0.06429111018097422,
                    "seed-seed": 0.021911677310176077, "pooled": 0.08019833966894428}

# P1 (plan §7.1)
P1_KS = (10, 100)
P1_INFORMATIVE_ABS_EM = 0.10
P1_MIN_INFORMATIVE = 8
P1_HOLD_MEDIAN = 0.03
P1_FAIL_MEDIAN = 0.06
P1_ABS_SUM = 0.05
P1_HOLD_FRAC = 0.80
P1_FAIL_FRAC = 0.50
# P2b (plan §7.3)
P2B_GUARD_SD = 2.0
P2B_HOLD = (0.75, 1.25)
P2B_HOLD_ENC = (0.5, 1.5)
P2B_FAIL = 0.25
P2B_FAIL_ENC = 0.5
MIN_INFORMATIVE_ENCODERS = 2
# P2c (plan §7.4)
P2C_GUARD_DLN = 0.3
P2C_HOLD = (0.17, 0.29)
P2C_HOLD_ENC = (0.154, 0.342)
P2C_FAIL = 0.12
P2C_FAIL_ENC = 0.154
# P2 descriptive (plan §7.2)
P2_R = 100_000
P2_SEED = int(hashlib.sha256(b"e0-p2-perm").hexdigest()[:16], 16)

LABELS = {"P1": ("holds", "partly holds", "fails", "uninformative"),
          "P2b": ("holds", "inconclusive", "fails", "uninformative"),
          "P2c": ("holds", "inconclusive", "fails", "uninformative")}


# ---------------------------------------------------------------------------------------------- helpers
def bhat_star(lnL, line):
    a, b, _ = line
    return np.minimum(a + b * np.asarray(lnL, float), 0.0)


def _ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    return float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))


def through_origin(g, y):
    g = np.asarray(g, float)
    y = np.asarray(y, float)
    return float(g @ y / (g @ g))


# ---------------------------------------------------------------------------------------------- P1
def p1_rule(sums, ems, pairs=None) -> dict:
    sums = np.asarray(sums, float)
    ems = np.asarray(ems, float)
    inf = np.abs(ems) >= P1_INFORMATIVE_ABS_EM
    out = {"n_pairs": int(sums.size), "n_informative": int(inf.sum()),
           "informative_pairs": [p for p, i in zip(pairs, inf) if i] if pairs is not None else None,
           "thresholds": {"informative_abs_e_m": P1_INFORMATIVE_ABS_EM, "min_informative": P1_MIN_INFORMATIVE,
                          "hold_abs_median": P1_HOLD_MEDIAN, "fail_abs_median": P1_FAIL_MEDIAN,
                          "abs_sum": P1_ABS_SUM, "hold_frac": P1_HOLD_FRAC, "fail_frac": P1_FAIL_FRAC},
           "median_sum": None, "frac_abs_sum_le": None}
    if inf.sum() < P1_MIN_INFORMATIVE:
        out["label"] = "uninformative"
        return out
    s = sums[inf]
    med = float(np.median(s))
    frac = float(np.mean(np.abs(s) <= P1_ABS_SUM))
    out["median_sum"], out["frac_abs_sum_le"] = med, frac
    if abs(med) <= P1_HOLD_MEDIAN and frac >= P1_HOLD_FRAC:
        out["label"] = "holds"
    elif abs(med) > P1_FAIL_MEDIAN or frac <= P1_FAIL_FRAC:
        out["label"] = "fails"
    else:
        out["label"] = "partly holds"
    return out


# ---------------------------------------------------------------------------------------------- P2b
def p2b_rule(L, b, vits, encs, line=LINES[("clip", 10, 1024)], comp=COMPETITOR[("clip", 10)], pairs=None) -> dict:
    """L: levels (already floored at eps_bar), b: slopes, vits/encs: labels per pair (plan §7.3)."""
    L = np.asarray(L, float)
    b = np.asarray(b, float)
    vits = np.asarray(vits)
    encs = np.asarray(encs)
    sres = line[2]
    bm = np.array([comp[v] for v in vits], float)
    bh = bhat_star(np.log(L), line)
    g = bh - bm
    y = b - bm
    thr = P2B_GUARD_SD * sres
    out = {"line": list(line), "guard_threshold": thr, "median_abs_gap": float(np.median(np.abs(g))),
           "per_pair": [{"pair": (pairs[i] if pairs is not None else i), "vit": str(vits[i]), "enc": str(encs[i]),
                         "L": float(L[i]), "b": float(b[i]), "bhat_star": float(bh[i]), "beta_M_v": float(bm[i]),
                         "g": float(g[i]), "y": float(y[i]), "resid_vs_line": float(b[i] - bh[i])}
                        for i in range(L.size)],
           "lam": None, "lam_e": {}, "informative_encoders": [],
           "thresholds": {"hold": P2B_HOLD, "hold_enc": P2B_HOLD_ENC, "fail": P2B_FAIL, "fail_enc": P2B_FAIL_ENC,
                          "min_informative_encoders": MIN_INFORMATIVE_ENCODERS}}
    if np.median(np.abs(g)) < thr:
        out["label"] = "uninformative"
        return out
    lam = through_origin(g, y)
    out["lam"] = lam
    lam_e = []
    for e in sorted(set(encs.tolist())):
        s = encs == e
        ok = bool(np.median(np.abs(g[s])) >= thr)
        le = through_origin(g[s], y[s]) if np.any(g[s] != 0) else float("nan")
        out["lam_e"][e] = {"lam": le, "informative": ok, "median_abs_gap": float(np.median(np.abs(g[s])))}
        if ok:
            out["informative_encoders"].append(e)
            lam_e.append(le)
    lam_e = np.array(lam_e, float)
    if lam_e.size < MIN_INFORMATIVE_ENCODERS:
        out["label"] = "inconclusive"
    elif P2B_HOLD[0] <= lam <= P2B_HOLD[1] and np.all((lam_e >= P2B_HOLD_ENC[0]) & (lam_e <= P2B_HOLD_ENC[1])):
        out["label"] = "holds"
    elif lam <= P2B_FAIL and np.all(lam_e <= P2B_FAIL_ENC):
        out["label"] = "fails"
    else:
        out["label"] = "inconclusive"
    return out


def p2b_leave_one_out(L, b, vits, encs, line=LINES[("clip", 10, 1024)], comp=COMPETITOR[("clip", 10)]) -> dict:
    """lambda-hat with one encoder or one ViT left out (always reported; no label)."""
    L, b, vits, encs = np.asarray(L, float), np.asarray(b, float), np.asarray(vits), np.asarray(encs)
    bm = np.array([comp[v] for v in vits], float)
    g = bhat_star(np.log(L), line) - bm
    y = b - bm
    out = {"leave_encoder_out": {}, "leave_vit_out": {}}
    for e in sorted(set(encs.tolist())):
        s = encs != e
        out["leave_encoder_out"][e] = through_origin(g[s], y[s]) if np.any(g[s] != 0) else None
    for v in sorted(set(vits.tolist())):
        s = vits != v
        out["leave_vit_out"][v] = through_origin(g[s], y[s]) if np.any(g[s] != 0) else None
    return out


def wilcoxon_descriptive(L, b, vits, line=LINES[("clip", 10, 1024)], comp=COMPETITOR[("clip", 10)],
                         beta_const=BETA_M_CONST) -> dict:
    """Exact one-sided Wilcoxon signed-rank of |b - beta_M| - |b - bhat*| (descriptive; plan §7.3)."""
    L, b = np.asarray(L, float), np.asarray(b, float)
    bh = bhat_star(np.log(L), line)
    out = {}
    for name, bm in (("vit_specific", np.array([comp[v] for v in vits], float)),
                     ("constant", np.full(L.size, beta_const))):
        d = np.abs(b - bm) - np.abs(b - bh)
        try:
            r = stats.wilcoxon(d, alternative="greater", method="exact")
            out[name] = {"statistic": float(r.statistic), "p_one_sided": float(r.pvalue),
                         "median_e_M": float(np.median(np.abs(b - bm))), "median_e_L": float(np.median(np.abs(b - bh)))}
        except ValueError as ex:
            out[name] = {"error": str(ex)}
    return out


# ---------------------------------------------------------------------------------------------- P2c
def gamma_pairs(Ln, bn):
    """Ln, bn: (pairs, 5) levels (floored) and slopes for n = 1..5 -> (gamma_j, dln_j)."""
    Ln = np.asarray(Ln, float)
    bn = np.asarray(bn, float)
    x = np.log(Ln)
    gam = np.array([_ols_slope(x[j], bn[j]) for j in range(x.shape[0])])
    dln = x[:, -1] - x[:, 0]
    return gam, dln


def p2c_rule(gam, dln, encs, pairs=None) -> dict:
    gam, dln, encs = np.asarray(gam, float), np.asarray(dln, float), np.asarray(encs)
    out = {"median_gamma": float(np.median(gam)), "median_dln": float(np.median(dln)), "gamma_e": {},
           "informative_encoders": [],
           "per_pair": [{"pair": (pairs[i] if pairs is not None else i), "enc": str(encs[i]), "gamma": float(gam[i]),
                         "dln": float(dln[i])} for i in range(gam.size)],
           "thresholds": {"guard_dln": P2C_GUARD_DLN, "hold": P2C_HOLD, "hold_enc": P2C_HOLD_ENC, "fail": P2C_FAIL,
                          "fail_enc": P2C_FAIL_ENC, "min_informative_encoders": MIN_INFORMATIVE_ENCODERS}}
    for e in sorted(set(encs.tolist())):
        s = encs == e
        ok = bool(np.median(dln[s]) >= P2C_GUARD_DLN)
        out["gamma_e"][e] = {"median_gamma": float(np.median(gam[s])), "median_dln": float(np.median(dln[s])),
                             "informative": ok}
        if ok:
            out["informative_encoders"].append(e)
    if np.median(dln) < P2C_GUARD_DLN:
        out["label"] = "uninformative"
        return out
    ge = np.array([out["gamma_e"][e]["median_gamma"] for e in out["informative_encoders"]], float)
    med = out["median_gamma"]
    if ge.size < MIN_INFORMATIVE_ENCODERS:
        out["label"] = "inconclusive"
    elif P2C_HOLD[0] <= med <= P2C_HOLD[1] and np.all((ge >= P2C_HOLD_ENC[0]) & (ge <= P2C_HOLD_ENC[1])):
        out["label"] = "holds"
    elif med <= P2C_FAIL and np.all(ge <= P2C_FAIL_ENC):
        out["label"] = "fails"
    else:
        out["label"] = "inconclusive"
    return out


def p2c_leave_one_out(gam, encs) -> dict:
    gam, encs = np.asarray(gam, float), np.asarray(encs)
    return {e: float(np.median(gam[encs != e])) for e in sorted(set(encs.tolist()))}


# ---------------------------------------------------------------------------------------------- P2 (descriptive)
def spearman(x, y) -> float:
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def p2_within_vit_permutation(Lm, Sm, R: int = P2_R, seed: int = P2_SEED) -> dict:
    """Lm, Sm: (5 ViTs alphabetical, 6 text models in plan order) levels and slopes (plan §7.2)."""
    Lm = np.asarray(Lm, float)
    Sm = np.asarray(Sm, float)
    nv, nt = Sm.shape
    rho = spearman(Lm.ravel(), Sm.ravel())
    rng = np.random.Generator(np.random.PCG64(seed))
    perms = np.empty((R, nv, nt), np.int64)
    for r in range(R):
        for i in range(nv):
            perms[r, i] = rng.permutation(nt)
    rL = stats.rankdata(Lm.ravel())
    rS = stats.rankdata(Sm.ravel()).reshape(nv, nt)
    rSp = np.take_along_axis(np.broadcast_to(rS, (R, nv, nt)), perms, axis=2).reshape(R, nv * nt)
    a = rL - rL.mean()
    bb = rSp - rSp.mean(axis=1, keepdims=True)
    rho_p = (bb @ a) / (np.sqrt((bb * bb).sum(axis=1)) * np.sqrt(a @ a))
    p = (1 + int(np.sum(rho_p >= rho - 1e-12))) / (1 + R)
    return {"rho": rho, "p_one_sided": p, "R": R, "seed": seed}


def p2_text_model_level(Lm, Sm) -> dict:
    """Per text model the median over ViTs of L and of b; Spearman over the 6; exact p over 720 permutations."""
    Lt = np.median(np.asarray(Lm, float), axis=0)
    St = np.median(np.asarray(Sm, float), axis=0)
    rho = spearman(Lt, St)
    rhos = [spearman(Lt, St[list(p)]) for p in itertools.permutations(range(St.size))]
    p = float(np.mean(np.asarray(rhos) >= rho - 1e-12))
    return {"rho": rho, "p_exact_one_sided": p, "n_perms": len(rhos), "level_by_text": Lt.tolist(),
            "slope_by_text": St.tolist()}
