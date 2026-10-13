"""Decision statistics (PREREG sections 8.3, 9, 10).

decide(x, n)         one-sided 95% Clopper-Pearson: reliable if lower > 0.9, unreliable if upper < 0.8
p_reliable / p_unreliable   exact binomial P(X >= x | 0.9) / P(X <= x | 0.8)
holm(pvals)          Holm step-down at alpha
pi_test(...)         block bootstrap of Pi = mean_AB - (mean_AA + mean_BB) (20,000 block resamples)
ratio_median_boot    bootstrap CI of median(e_hi)/median(e_lo) (runs resampled within each N)
tost_mean_boot       bootstrap equivalence test of a block mean against a band
betabinom_reliable   P(reliable) for a decision set exchangeable with n_p pilot pairs (shared max)
All bootstrap generators are keyed: rng(KEY|boot|<name>).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta, binom

import s23common as C

NBOOT = 20000
PMIN = 1.0 / NBOOT


def cp_bounds(x: int, n: int, alpha: float = 0.05):
    lo = beta.ppf(alpha, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(1 - alpha, x + 1, n - x) if x < n else 1.0
    return float(lo), float(hi)


def decide(x: int, n: int):
    lo, hi = cp_bounds(x, n)
    if n == 0:
        return "none", lo, hi
    if lo > 0.9:
        return "reliable", lo, hi
    if hi < 0.8:
        return "unreliable", lo, hi
    return "inconclusive", lo, hi


def p_reliable(x: int, n: int) -> float:
    return float(binom.sf(x - 1, n, 0.9))


def p_unreliable(x: int, n: int) -> float:
    return float(binom.cdf(x, n, 0.8))


def count_within(deltas, eps):
    d = np.asarray(deltas, dtype=np.float64)
    return int(np.sum(d <= eps)), int(d.size)


def decision_row(deltas, eps, eps2=None):
    x, n = count_within(deltas, eps)
    dec, lo, hi = decide(x, n)
    row = {"n": n, "x": x, "phat": x / n if n else float("nan"), "cp_lo": lo, "cp_hi": hi,
           "decision": dec, "eps": float(eps), "p_rel": p_reliable(x, n), "p_unrel": p_unreliable(x, n)}
    if eps2 is not None:
        x2, _ = count_within(deltas, eps2)
        row.update(eps2=float(eps2), x_eps2=x2, decision_eps2=decide(x2, n)[0])
    row["phat_curve"] = {str(f): count_within(deltas, f * eps)[0] / n for f in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5)}
    return row


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    stop = False
    for i, (k, p) in enumerate(items):
        thr = alpha / (m - i)
        rej = (not stop) and (p <= thr)
        if not rej:
            stop = True
        out[k] = {"p": p, "holm_threshold": thr, "rejected": rej}
    return out


def pi_test(ab, aa, bb, key: str, nboot: int = NBOOT):
    """ab, aa, bb: per-block values (same block order, length n_b). Returns dict with means, Pi,
    95% percentile CI, plateau / no-plateau flags, rho and the one-sided bootstrap fractions."""
    ab, aa, bb = (np.asarray(v, dtype=np.float64) for v in (ab, aa, bb))
    nb = ab.size
    r = C.rng(f"{C.KEY}|boot|{key}")
    idx = r.integers(0, nb, size=(nboot, nb))
    pis = ab[idx].mean(1) - (aa[idx].mean(1) + bb[idx].mean(1))
    mab, maa, mbb = float(ab.mean()), float(aa.mean()), float(bb.mean())
    Pi = mab - (maa + mbb)
    lo, hi = (float(v) for v in np.quantile(pis, [0.025, 0.975]))
    allzero = (mab == 0.0 and maa == 0.0 and mbb == 0.0)
    den = 0.5 * (maa + mbb)
    return {"n_blocks": nb, "mean_AB": mab, "mean_AA": maa, "mean_BB": mbb, "Pi": Pi, "ci_lo": lo, "ci_hi": hi,
            "plateau": bool(lo > 0), "no_plateau": bool(hi < 0 or allzero), "all_zero": allzero,
            "rho": mab / den if den > 0 else float("nan"),
            "p_noplateau": PMIN if allzero else max(PMIN, float(np.mean(pis >= 0))),
            "p_plateau": max(PMIN, float(np.mean(pis <= 0)))}


def ratio_median_boot(e_hi, e_lo, key: str, nboot: int = NBOOT, thr_conv=0.35, thr_plat=0.7):
    e_hi, e_lo = np.asarray(e_hi, float), np.asarray(e_lo, float)
    r = C.rng(f"{C.KEY}|boot|{key}")
    ih = r.integers(0, e_hi.size, size=(nboot, e_hi.size))
    il = r.integers(0, e_lo.size, size=(nboot, e_lo.size))
    num = np.median(e_hi[ih], axis=1)
    den = np.median(e_lo[il], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.where(den > 0, num / den, np.inf)
    ratio = float(np.median(e_hi) / np.median(e_lo)) if np.median(e_lo) > 0 else float("inf")
    lo, hi = (float(v) for v in np.quantile(rr, [0.025, 0.975]))
    return {"ratio": ratio, "ci_lo": lo, "ci_hi": hi, "converging": bool(hi < thr_conv), "plateau": bool(lo > thr_plat),
            "p_converging": max(PMIN, float(np.mean(rr >= thr_conv)))}


def tost_mean_boot(vals, lo_band, hi_band, key: str, nboot: int = NBOOT):
    v = np.asarray(vals, float)
    r = C.rng(f"{C.KEY}|boot|{key}")
    idx = r.integers(0, v.size, size=(nboot, v.size))
    mb = v[idx].mean(1)
    p1 = max(PMIN, float(np.mean(mb <= lo_band)))
    p2 = max(PMIN, float(np.mean(mb >= hi_band)))
    return {"mean": float(v.mean()), "band": [lo_band, hi_band], "p_low": p1, "p_high": p2, "p": max(p1, p2),
            "ci90": [float(np.quantile(mb, 0.05)), float(np.quantile(mb, 0.95))]}


def upper_mean_boot(vals, thr, key: str, nboot: int = NBOOT):
    v = np.asarray(vals, float)
    r = C.rng(f"{C.KEY}|boot|{key}")
    idx = r.integers(0, v.size, size=(nboot, v.size))
    mb = v[idx].mean(1)
    return {"mean": float(v.mean()), "thr": thr, "p": max(PMIN, float(np.mean(mb >= thr))),
            "ci_hi": float(np.quantile(mb, 0.975))}


def betabinom_reliable(n: int, n_p: int) -> float:
    """P(lower CP > 0.9) when x ~ BetaBinomial(n, n_p, 1): each decision pair is compared with the
    same maximum of n_p exchangeable pilot pairs (the critic's correction of v0.1's binomial)."""
    from scipy.stats import betabinom
    xs = np.arange(n + 1)
    ok = np.array([decide(int(x), n)[0] == "reliable" for x in xs])
    return float(betabinom.pmf(xs, n, n_p, 1)[ok].sum())


def chi2_band(df: int, alpha: float):
    from scipy.stats import chi2
    return float(chi2.ppf(alpha / 2, df) / df), float(chi2.ppf(1 - alpha / 2, df) / df)
