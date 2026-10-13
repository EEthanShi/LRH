#!/usr/bin/env python
"""LLaMA axis, implementation I (role llama-axis-run-indep): aggregation.

Reads the full observed and null L_A x L_B matrices of this run (3 LLaMA x 17 ViTs,
results/S1_2_layermax_llama/indep/<pre>/pairs/) and, for the pooled 10-LM panel, the
frozen 8-LM independent run (results/S1_2_layermax/indep/<pre>/pairs/, read-only; the 7
non-LLaMA LMs; llama-7b is taken from this run after V-L4b). Writes
results/S1_2_layermax_llama/indep/summary.json.

Implements ANALYSIS_PLAN_LLAMA.md (990428e3...) sections 3 (protocols i-iv incl. the
added subsets noLMemb / LMdeep / LMemb_only and calibrate-then-max), 5 (trend
statistics: Tr_L with exact p over 6 relabellings, Delta Tr_L, ViT bootstrap, joint
Tr_BL over 720 relabellings, ARH-style pooled r on 10 LMs with exact p over 10!,
sensitivity covariates, 8-LM comparison, vision axis, OpenLLaMA fraction), 6
(attribution diagnostics A1-A3, level vs multiplicity, per-LM-layer null profile,
plug-in check), 7 (readings) and the Part B verdicts of PARTB_PREDICTIONS.md
(f9afa919...) P-L1..P-L5, P-L2a with their deductive checks, and section 10.
Per-pair statistics reuse lmx_aggregate.pair_stats (byte-identical copy of the frozen
8-LM indep code); the S1.2 trend helpers (_rho_matrix, boot_vits, vision_trend) are
read-only imports of analysis/s12/aggregate.py.
"""
from __future__ import annotations

import os

_T = os.environ.get("LMX_THREADS", "3")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

import lmx_common as L  # noqa: E402
from lmx_common import C, cal  # noqa: E402
from hrhv2 import util  # noqa: E402
import lmx_aggregate as A  # noqa: E402  (pair_stats, reading_arh; main() not run)

S12 = A.S12   # analysis/s12/aggregate.py, read-only
HERE = os.path.dirname(os.path.abspath(__file__))
PARTB = os.path.join(L.PROJ, "results", "S1_2_layermax_llama", "partB", "partB_features.json")
OLD_SUMMARY = os.path.join(L.OLD_INDEP, "layermax_indep_summary.json")
METRICS = A.METRICS
NV = len(L.VIT_STEMS)
LLAMA = list(L.LM_STEMS)
POOLED = list(L.POOLED_LM_STEMS)
BLOOMZ = [s for s, f in C.LMS_POOLED if f == "BLOOMZ"]
IL = [POOLED.index(s) for s in LLAMA]
IB = [POOLED.index(s) for s in BLOOMZ]
I_OL3, I_OL7 = POOLED.index("openlm-research__open_llama_3b"), POOLED.index("openlm-research__open_llama_7b")
EPS = 1e-12
MAT = 0.10           # materiality for changes in Tr_L (plan section 5)
DELTA_B = 0.005      # Part B materiality of baseline gaps
ORDER_IDX = cal.order_index(L.K_PERM, L.ALPHA) - 1
NB = 2000
SUBS = ["full", "grid2", "grid3", "grid5", "grid9", "grid17", "lmall_x_vitlast", "lmlast_x_vitall",
        "fixed_r0.5", "fixed_r1.0", "noLMemb", "LMdeep", "LMemb_only", "ctm"]
MAIN_SUBS = ("full", "fixed_r0.5", "fixed_r1.0", "ctm")
SCORE_FIELDS = ("s_obs", "s_cal", "uncensored")
EXTRA_FIELDS = ("c_alpha", "null_mean")
PRIMARY = "prh_clip+l2"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- per pair
def load_z(pre, lm, vit):
    base = L.OUT if lm in LLAMA else L.OLD_INDEP
    z = np.load(os.path.join(base, pre, "pairs", f"{lm}__{vit}.npz"))
    assert int(z["K"]) == L.K_PERM and str(z["key"]) == L.perm_key(lm, vit), (lm, vit)
    return z


def extra_subsets(LA, LB):
    allB = np.arange(LB)
    deep = np.array([a for a in range(LA) if a / (LA - 1) >= 0.5])
    return {"noLMemb": (np.arange(1, LA), allB), "LMdeep": (deep, allB), "LMemb_only": (np.array([0]), allB)}


def pair_stats_ext(z, lm, vit):
    P = A.pair_stats(z, lm, vit)
    m, k = L.M, L.KNN
    obs = {"mknn": z["obs_cnt"] / (m * k), "cka_b": z["obs_cka_b"], "cka_d": z["obs_cka_d"]}
    null = {"mknn": z["null_cnt"].astype(np.float64) / (m * k), "cka_b": z["null_cka_b"], "cka_d": z["null_cka_d"]}
    LA, LB = P["LA"], P["LB"]
    for met in METRICS:
        S, N = obs[met], null[met]
        K = N.shape[0]
        for name, (ra, rb) in extra_subsets(LA, LB).items():
            Ssub = S[np.ix_(ra, rb)]
            Nsub = N[:, ra][:, :, rb]
            T = float(Ssub.max())
            sm = L.summarize(T, Nsub.reshape(K, -1).max(axis=1))
            ia, ib = np.unravel_index(int(np.argmax(Ssub)), Ssub.shape)
            d = {f: sm[f] for f in ("s_obs", "c_alpha", "s_cal", "p_value", "null_mean", "null_sd", "uncensored")}
            d.update({"a": int(ra[ia]), "b": int(rb[ib]), "n_entries": int(Ssub.size), "n_tied_max": int(np.sum(Ssub == T))})
            P[met][name] = d
        # (iv) calibrate-then-max: each entry against its own c_alpha,uv from the same 500 pi
        comb = np.concatenate([S[None], N], axis=0)
        c_e = np.sort(comb, axis=0)[ORDER_IDX]
        den = 1.0 - c_e
        scal_e = np.where(den > 0, np.maximum((S - c_e) / np.where(den > 0, den, 1.0), 0.0), 0.0)
        mu_e = N.mean(axis=0)
        unc_e = S - mu_e
        ia, ib = np.unravel_index(int(np.argmax(scal_e)), scal_e.shape)
        ua, ub = np.unravel_index(int(np.argmax(unc_e)), unc_e.shape)
        P[met]["ctm"] = {"s_obs": P[met]["full"]["s_obs"], "s_cal": float(scal_e.max()), "uncensored": float(unc_e.max()),
                         "a": int(ia), "b": int(ib), "rel_a": ia / (LA - 1), "c_entry_at_argmax": float(c_e[ia, ib]),
                         "S_at_argmax": float(S[ia, ib]), "n_entries_scal_pos": int(np.sum(scal_e > 0)),
                         "n_tied_max": int(np.sum(scal_e == scal_e.max())), "unc_a": int(ua), "unc_b": int(ub)}
        # diagnostics (plan section 6)
        prof = mu_e.max(axis=1)
        na, nb_ = np.unravel_index(int(np.argmax(mu_e)), mu_e.shape)
        qa, qb = np.unravel_index(int(np.argmax(c_e)), c_e.shape)
        P[met]["diag"] = {"E0max": P[met]["full"]["null_mean"], "max_entry_null_mean": float(mu_e.max()),
                          "level_minus_multiplicity": P[met]["full"]["null_mean"] - float(mu_e.max()),
                          "argmax_entry_null_mean": [int(na), int(nb_)], "argmax_entry_null_mean_rel_a": na / (LA - 1),
                          "lm_layer_null_profile": prof.tolist(), "max_entry_c": float(c_e.max()),
                          "argmax_entry_c": [int(qa), int(qb)], "calpha_minus_max_entry_c": P[met]["full"]["c_alpha"] - float(c_e.max())}
    return P


def grid(Pd, stems, met, sub, field):
    return np.array([[float(Pd[(lm, vit)][met][sub][field]) for vit in L.VIT_STEMS] for lm in stems])


# ----------------------------------------------------------------------------- trends
def size_ranks(stems):
    return rankdata([C.n_params(s) for s in stems])


def fam_trend(G, stems):
    """analysis/s12/aggregate.py::lm_trend with the family given by `stems` (rows of G):
    Tr = mean over ViTs of Spearman(size rank, score); exact p over all relabellings."""
    Y = np.asarray(G, dtype=np.float64)
    ok = np.all(np.isfinite(Y), axis=0)
    Y = Y[:, ok]
    x = size_ranks(stems)
    P = np.array([np.array(p) for p in itertools.permutations(x)])
    rho = S12._rho_matrix(P, Y)
    oi = int(np.flatnonzero(np.all(P == x, axis=1))[0])
    with np.errstate(invalid="ignore"):
        T = np.nanmean(rho, axis=1)
    t = float(T[oi])
    rv = rho[oi]
    return {"Tr": t, "p_one": float(np.mean(T >= t - EPS)), "p_two": float(np.mean(np.abs(T) >= abs(t) - EPS)),
            "n_vit_used": int(np.sum(np.isfinite(rv))), "n_vit_constant": int(np.sum(~np.isfinite(rv)) + np.sum(~ok)),
            "rho_per_vit": [None if not np.isfinite(v) else float(v) for v in rv], "n_perm": int(P.shape[0]),
            "_T": T, "_oi": oi, "_rho": rv}


def pub(t):
    return {k: v for k, v in t.items() if not k.startswith("_")}


def tr_with_ci(G, stems, key):
    t = fam_trend(G, stems)
    est, lo, hi = S12.boot_vits(t["_rho"], f"layermax-llama|{key}")
    t["ci"] = [lo, hi]
    return t


def delta(ts, tr_, key):
    D = ts["_T"] - tr_["_T"]
    obs = float(D[ts["_oi"]])
    est, lo, hi = S12.boot_vits(ts["_rho"], f"layermax-llama|delta|{key}", vec2=tr_["_rho"])
    return {"dTr": obs, "p_one_increase": float(np.mean(D >= obs - EPS)), "p_one_decrease": float(np.mean(D <= obs + EPS)),
            "p_two": float(np.mean(np.abs(D) >= abs(obs) - EPS)), "ci": [lo, hi], "n_perm": int(D.size)}


def joint(tb, tl):
    J = (tb["_T"][:, None] + tl["_T"][None, :]) / 2.0
    obs = (tb["Tr"] + tl["Tr"]) / 2.0
    return {"Tr_BL": float(obs), "Tr_BLOOMZ": tb["Tr"], "Tr_L": tl["Tr"], "p_one": float(np.mean(J >= obs - EPS)),
            "p_two": float(np.mean(np.abs(J) >= abs(obs) - EPS)), "n_perm": int(J.size)}


class Pooled:
    """ARH-style Pearson r across all (LM, ViT) pairs between score and an LM covariate;
    exact one-sided p over all n_LM! relabellings (r is linear in the LM row sums)."""

    def __init__(self, stems, boot_prefix):
        self.stems = stems
        n = len(stems)
        t0 = time.time()
        self.perms = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
        cov = {"arh_rank": np.array([L.ARH_RANK[s] for s in stems], dtype=np.float64)}
        cov["dense_rank"] = rankdata(cov["arh_rank"]).astype(np.float64)
        cov["log_params"] = np.log(np.array([C.n_params(s) for s in stems], dtype=np.float64))
        self.cov = cov
        self.xc = {c: v - v.mean() for c, v in cov.items()}
        self.xp = {c: self.xc[c][self.perms] for c in cov}
        self.boot_prefix = boot_prefix
        log(f"pooled panel {n} LMs: {len(self.perms)} relabellings ({time.time() - t0:.0f}s)")

    def r(self, G, cov, key=None, boot=False):
        G = np.asarray(G, dtype=np.float64)
        xc = self.xc[cov]
        Gc = G - G.mean()
        den = math.sqrt(NV * float(np.sum(xc ** 2))) * math.sqrt(float(np.sum(Gc ** 2)))
        if den == 0:
            return {"r": None, "note": "constant score"}
        R = Gc.sum(axis=1)
        r_obs = float(np.dot(xc, R) / den)
        rp = (self.xp[cov] @ R) / den
        out = {"r": r_obs, "p_one": float(np.mean(rp >= r_obs - EPS)), "p_two": float(np.mean(np.abs(rp) >= abs(r_obs) - EPS)),
               "n_perm": int(rp.size)}
        if boot:
            rng = util.rng_from_key(f"{self.boot_prefix}|{key}")
            vals = np.empty(NB)
            for t in range(NB):
                jj = rng.integers(0, NV, NV)
                g = G[:, jj]
                gc = g - g.mean()
                dd = math.sqrt(NV * float(np.sum(xc ** 2))) * math.sqrt(float(np.sum(gc ** 2)))
                vals[t] = np.dot(xc, gc.sum(axis=1)) / dd if dd > 0 else np.nan
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            out["ci"] = [float(lo), float(hi)]
        return out


# ----------------------------------------------------------------------------- readings / verdicts
def reading_llama(tr_raw_cka, tr_unc_cka, tr_unc_k):
    if tr_raw_cka < 0.60:
        return "no raw trend to lose"
    if tr_unc_cka <= 0.40 and tr_unc_k >= 0.60:
        return "layer-max loses the CKA trend on the LLaMA axis"
    if tr_unc_cka >= 0.60:
        return "no loss"
    return "intermediate"


def keep_sign(s, tr):
    if s * tr >= MAT:
        return "keeps sign"
    if abs(tr) < MAT:
        return "lost"
    return "reversed"


def four_way(s, d, tr_raw):
    if abs(tr_raw - (-s)) <= EPS:
        return "not testable"
    if -s * d >= MAT:
        return "confirmed"
    if abs(d) < MAT:
        return "consistent-null"
    if s * d >= MAT:
        return "contradicted"
    return "undetermined"


def sgn(x, tol=0.0):
    return 0 if abs(x) <= tol else (1 if x > 0 else -1)


def pairwise_flips(Graw, Gcal, qual):
    """(ViT, size-pair) comparisons in `qual` (list of (v, j, j2) with j < j2 = smaller, larger):
    raw concordant (score rises with size) -> calibrated discordant = 'predicted direction'
    (rising thresholds); raw discordant -> calibrated concordant = 'against'."""
    out = {"n": len(qual), "conc_to_disc": 0, "disc_to_conc": 0, "conc_to_tie": 0, "disc_to_tie": 0,
           "tie_to_any": 0, "unchanged": 0, "list_against": []}
    for v, j, j2 in qual:
        r0 = sgn(Graw[j2, v] - Graw[j, v])
        r1 = sgn(Gcal[j2, v] - Gcal[j, v])
        if r0 == 0:
            out["tie_to_any"] += 1
        elif r0 == r1:
            out["unchanged"] += 1
        elif r0 > 0 and r1 < 0:
            out["conc_to_disc"] += 1
        elif r0 < 0 and r1 > 0:
            out["disc_to_conc"] += 1
            out["list_against"].append([L.VIT_STEMS[v], LLAMA[j], LLAMA[j2]])
        elif r0 > 0 and r1 == 0:
            out["conc_to_tie"] += 1
        else:
            out["disc_to_tie"] += 1
    return out


def partb(pre, Pd, blocks, pooled10, pb):
    """Part B verdicts for one pre (PARTB_PREDICTIONS.md sections 5-6)."""
    la = pb["per_pre"][pre]["llama_axis"]
    Bbar = np.array([[pb["per_pre"][pre]["Bbar"][lm][vit] for vit in L.VIT_STEMS] for lm in LLAMA])
    material = bool(la["Bbar"]["material"])
    s_pred = int(np.sign(la["Bbar"]["trend"]["Tr"])) if abs(la["Bbar"]["trend"]["Tr"]) > 0 else 0
    out = {"pre": pre, "Bbar_material": material, "Bbar_Tr_pred": la["Bbar"]["trend"]["Tr"], "s": s_pred}
    gl = lambda met, sub, f: grid(Pd, LLAMA, met, sub, f)  # noqa: E731
    bk = lambda met, sub: blocks[met][sub]  # noqa: E731
    # ---- P-L1 (mKNN, layer-max)
    b = bk("mknn", "full")
    d1s, d1u = b["delta"]["s_cal"]["dTr"], b["delta"]["uncensored"]["dTr"]
    T, c, mu, sc, un = (gl("mknn", "full", f) for f in ("s_obs", "c_alpha", "null_mean", "s_cal", "uncensored"))
    base = L.KNN / (L.M - 1)
    zeta, zeta_u = np.abs(c - base), np.abs(mu - base)
    viol_s, viol_u, n_elig_s, n_elig_u = [], [], 0, 0
    for v in range(NV):
        for j, j2 in ((0, 1), (0, 2), (1, 2)):
            g = T[j2, v] - T[j, v]
            if abs(g) > zeta[j, v] + zeta[j2, v]:
                n_elig_s += 1
                if sgn(sc[j2, v] - sc[j, v]) == -sgn(g):
                    viol_s.append([L.VIT_STEMS[v], LLAMA[j], LLAMA[j2]])
            if abs(g) > zeta_u[j, v] + zeta_u[j2, v]:
                n_elig_u += 1
                if sgn(un[j2, v] - un[j, v]) == -sgn(g):
                    viol_u.append([L.VIT_STEMS[v], LLAMA[j], LLAMA[j2]])
    out["P-L1"] = {"Tr_raw": b["llama_axis"]["s_obs"]["Tr"], "Tr_scal": b["llama_axis"]["s_cal"]["Tr"],
                   "Tr_unc": b["llama_axis"]["uncensored"]["Tr"], "dTr_scal": d1s, "dTr_unc": d1u,
                   "prediction": "|dTr_scal| < 0.10 and |dTr_unc| < 0.10",
                   "outcome": "holds" if (abs(d1s) < MAT and abs(d1u) < MAT) else "fails",
                   "zeta_mean": float(zeta.mean()), "zeta_max": float(zeta.max()),
                   "deductive_check": {"s_cal": {"n_eligible": n_elig_s, "n_violations": len(viol_s), "violations": viol_s},
                                       "uncensored": {"n_eligible": n_elig_u, "n_violations": len(viol_u), "violations": viol_u}}}
    # ---- P-L2 / P-L2a (biased CKA, layer-max)
    b = bk("cka_b", "full")
    tr_raw = b["llama_axis"]["s_obs"]["Tr"]
    d2s, d2u = b["delta"]["s_cal"]["dTr"], b["delta"]["uncensored"]["dTr"]
    T, c, mu, sc, un = (gl("cka_b", "full", f) for f in ("s_obs", "c_alpha", "null_mean", "s_cal", "uncensored"))
    qual = [(v, j, j2) for v in range(NV) for j, j2 in ((0, 1), (0, 2), (1, 2)) if abs(Bbar[j2, v] - Bbar[j, v]) >= DELTA_B]
    n_qual = {f"{LLAMA[j]}|{LLAMA[j2]}": sum(1 for (v, a, bb) in qual if (a, bb) == (j, j2)) for j, j2 in ((0, 1), (0, 2), (1, 2))}
    frozen_nq = {k: la["Bbar"]["gaps"][k]["n_vit_gap_ge_delta_b"] for k in n_qual}
    agree = sum(1 for v, j, j2 in qual if sgn(c[j2, v] - c[j, v]) == sgn(Bbar[j2, v] - Bbar[j, v]))
    need = math.ceil(2 * len(qual) / 3) if qual else 0
    med = float(np.median(np.abs(c - Bbar)))
    p2a_i, p2a_ii = med <= 0.01, (agree >= need) if qual else None
    # deductive check: monotone c_alpha order => rho_v(s_cal) - rho_v(raw) has the opposite sign (or 0)
    rho_raw = np.array(b["llama_axis"]["s_obs"]["rho_per_vit"], dtype=float)
    rho_cal = np.array(b["llama_axis"]["s_cal"]["rho_per_vit"], dtype=float)
    rho_unc = np.array(b["llama_axis"]["uncensored"]["rho_per_vit"], dtype=float)
    ded, ded_u = [], []
    n_inc = n_dec = n_inc_u = n_dec_u = 0
    for v in range(NV):
        inc, dec = c[0, v] < c[1, v] < c[2, v], c[0, v] > c[1, v] > c[2, v]
        n_inc += int(inc)
        n_dec += int(dec)
        dv = rho_cal[v] - rho_raw[v]
        if (inc and dv > 1e-12) or (dec and dv < -1e-12):
            ded.append({"vit": L.VIT_STEMS[v], "c": c[:, v].tolist(), "drho": float(dv), "s_cal": sc[:, v].tolist(),
                        "clipped": bool(np.any(sc[:, v] == 0))})
        inc_u, dec_u = mu[0, v] < mu[1, v] < mu[2, v], mu[0, v] > mu[1, v] > mu[2, v]
        n_inc_u += int(inc_u)
        n_dec_u += int(dec_u)
        du = rho_unc[v] - rho_raw[v]
        if (inc_u and du > 1e-12) or (dec_u and du < -1e-12):
            ded_u.append({"vit": L.VIT_STEMS[v], "E0max": mu[:, v].tolist(), "drho": float(du)})
    p2 = {"Tr_raw": tr_raw, "Tr_scal": b["llama_axis"]["s_cal"]["Tr"], "Tr_unc": b["llama_axis"]["uncensored"]["Tr"],
          "dTr_scal": d2s, "dTr_unc": d2u, "dTr_scal_p_one_decrease": b["delta"]["s_cal"]["p_one_decrease"],
          "dTr_unc_p_one_decrease": b["delta"]["uncensored"]["p_one_decrease"],
          "Tr_calpha": blocks["cka_b"]["full"]["llama_axis"]["c_alpha"]["Tr"],
          "Tr_E0max": blocks["cka_b"]["full"]["llama_axis"]["null_mean"]["Tr"]}
    if material:
        p2["outcome_scal"] = four_way(s_pred, d2s, tr_raw)
        p2["outcome_unc_secondary"] = four_way(s_pred, d2u, tr_raw)
        p2["prediction"] = f"sign(dTr_scal) = {-s_pred}: confirmed if {-s_pred}*... (four-way rule, materiality 0.10)"
    else:
        p2["outcome_scal"] = "no directional prediction (B-bar trend not material)"
        p2["outcome_unc_secondary"] = "no directional prediction (B-bar trend not material)"
    p2["pairwise_flips_qualifying"] = {"s_cal": pairwise_flips(T, sc, qual), "uncensored": pairwise_flips(T, un, qual)}
    out["P-L2"] = p2
    out["P-L2a"] = {"median_abs_calpha_minus_Bbar": med, "criterion_i": "median <= 0.01", "holds_i": bool(p2a_i),
                    "n_qualifying": len(qual), "n_qualifying_by_size_pair": n_qual,
                    "n_qualifying_matches_frozen": n_qual == frozen_nq, "n_agree": agree, "n_needed": need,
                    "holds_ii": p2a_ii, "outcome": "holds" if (p2a_i and p2a_ii) else "fails",
                    "calpha_minus_Bbar": {"mean": float(np.mean(c - Bbar)), "min": float(np.min(c - Bbar)),
                                          "max": float(np.max(c - Bbar))},
                    "E0max_minus_Bbar": {"mean": float(np.mean(mu - Bbar)), "min": float(np.min(mu - Bbar)),
                                         "max": float(np.max(mu - Bbar))},
                    "pearson_calpha_Bbar": float(np.corrcoef(c.ravel(), Bbar.ravel())[0, 1]),
                    "mean_calpha_by_lm": {lm: float(c[i].mean()) for i, lm in enumerate(LLAMA)},
                    "mean_Bbar_by_lm": {lm: float(Bbar[i].mean()) for i, lm in enumerate(LLAMA)},
                    "deductive_check": {"n_vit_calpha_increasing": n_inc, "n_vit_calpha_decreasing": n_dec,
                                        "n_violations": len(ded), "violations": ded,
                                        "uncensored_analogue": {"n_vit_E0max_increasing": n_inc_u,
                                                                "n_vit_E0max_decreasing": n_dec_u,
                                                                "n_violations": len(ded_u), "violations": ded_u}}}
    # ---- P-L3 (biased CKA, fixed depth)
    p3 = {}
    for r in ("0.5", "1.0"):
        bb = bk("cka_b", f"fixed_r{r}")
        trr, trs = bb["llama_axis"]["s_obs"]["Tr"], bb["llama_axis"]["s_cal"]["Tr"]
        e = {"Tr_raw": trr, "Tr_scal": trs, "Tr_unc": bb["llama_axis"]["uncensored"]["Tr"],
             "dTr_scal": bb["delta"]["s_cal"]["dTr"], "Tr_calpha": bb["llama_axis"]["c_alpha"]["Tr"],
             "Bfd_material_partB": bool(la["Bfd"][r]["material"]), "Bfd_Tr_partB": la["Bfd"][r]["trend"]["Tr"]}
        if abs(trr) >= MAT:
            s_r = 1 if trr > 0 else -1
            e["s_r"] = s_r
            e["outcome"] = keep_sign(s_r, trs)
            e["unc_outcome_descriptive"] = keep_sign(s_r, e["Tr_unc"])
        else:
            e["outcome"] = "not applicable (|Tr_L(raw)| < 0.10)"
        p3[f"r{r}"] = e
    out["P-L3"] = p3
    # ---- P-L4 (debiased CKA)
    b = bk("cka_d", "full")
    r_before = b["pooled_r"]["s_obs"]["arh_rank"]["r"]
    r_after = b["pooled_r"]["s_cal"]["arh_rank"]["r"]
    d4s, d4u = b["delta"]["s_cal"]["dTr"], b["delta"]["uncensored"]["dTr"]
    ok_a = d4s >= -MAT and d4u >= -MAT
    ok_b = (r_after if r_after is not None else -9) >= r_before - 0.05
    out["P-L4"] = {"Tr_raw": b["llama_axis"]["s_obs"]["Tr"], "Tr_scal": b["llama_axis"]["s_cal"]["Tr"],
                   "Tr_unc": b["llama_axis"]["uncensored"]["Tr"], "dTr_scal": d4s, "dTr_unc": d4u,
                   "pooled10_r_before": r_before, "pooled10_r_after": r_after, "holds_a": bool(ok_a), "holds_b": bool(ok_b),
                   "outcome": "holds" if (ok_a and ok_b) else "fails"}
    # ---- P-L5 (calibrate-then-max, biased CKA)
    b = bk("cka_b", "ctm")
    trr = blocks["cka_b"]["full"]["llama_axis"]["s_obs"]["Tr"]
    trm = b["llama_axis"]["s_cal"]["Tr"]
    e = {"Tr_raw_layermax": trr, "Tr_M": trm, "Tr_M_unc": b["llama_axis"]["uncensored"]["Tr"]}
    if abs(trr) >= MAT:
        e["s"] = 1 if trr > 0 else -1
        e["outcome"] = keep_sign(e["s"], trm)
        e["unc_outcome_descriptive"] = keep_sign(e["s"], e["Tr_M_unc"])
    else:
        e["outcome"] = "not applicable (|Tr_L(raw)| < 0.10)"
    out["P-L5"] = e
    # ---- plug-in check (section 6): Tr_L(T - Bbar_pred) vs Tr_L(uncensored)
    Tg = gl("cka_b", "full", "s_obs")
    tp = fam_trend(Tg - Bbar, LLAMA)
    out["plug_in_check"] = {"Tr_T_minus_Bbar": tp["Tr"], "Tr_unc": blocks["cka_b"]["full"]["llama_axis"]["uncensored"]["Tr"],
                            "abs_diff": abs(tp["Tr"] - blocks["cka_b"]["full"]["llama_axis"]["uncensored"]["Tr"]),
                            "within_0.10": abs(tp["Tr"] - blocks["cka_b"]["full"]["llama_axis"]["uncensored"]["Tr"]) <= MAT}
    return out


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    pb = json.load(open(PARTB))
    old = json.load(open(OLD_SUMMARY))
    assert list(S12.C.VITS) == list(C.VITS), "ViT order differs between the S1.2 registry and the copy"
    pooled10 = Pooled(POOLED, "S1.2-layermax-llama|boot|arh_r")
    pooled8 = Pooled(POOLED[:8], "S1.2-layermax-llama|boot8|arh_r")
    summary = {"role": "llama-axis-run-indep", "implementation": "I",
               "plan": "analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md", "partB": "analysis/s12_layermax_llama/PARTB_PREDICTIONS.md",
               "K": L.K_PERM, "alpha": L.ALPHA, "llama_order": LLAMA, "pooled_order": POOLED, "vit_order": L.VIT_STEMS,
               "arh_rank": {s: L.ARH_RANK[s] for s in POOLED}, "per_pre": {}, "self_checks": {}}
    for pre in L.PRES:
        Pd = {}
        for lm in POOLED:
            for vit in L.VIT_STEMS:
                Pd[(lm, vit)] = pair_stats_ext(load_z(pre, lm, vit), lm, vit)
        log(f"{pre}: pair stats done ({len(Pd)} pairs)")
        nlp = np.array([[Pd[(lm, v)]["LA"] * Pd[(lm, v)]["LB"] for v in L.VIT_STEMS] for lm in POOLED], float)
        blocks = {}
        for met in METRICS:
            blocks[met] = {}
            for sub in SUBS:
                fields = SCORE_FIELDS + (EXTRA_FIELDS if sub != "ctm" else ())
                G = {f: grid(Pd, POOLED, met, sub, f) for f in fields}
                key = f"{pre}|{met}|{sub}"
                tl = {f: tr_with_ci(G[f][IL], LLAMA, f"{key}|{f}") for f in fields}
                tb = {f: fam_trend(G[f][IB], BLOOMZ) for f in SCORE_FIELDS}
                blk = {
                    "llama_axis": {f: pub(tl[f]) for f in fields},
                    "delta": {f: delta(tl[f], tl["s_obs"], f"{key}|{f}") for f in ("s_cal", "uncensored")},
                    "bloomz_axis": {f: pub(tb[f]) for f in SCORE_FIELDS},
                    "joint_BL": {f: joint(tb[f], tl[f]) for f in SCORE_FIELDS},
                    "pooled_r": {f: {cov: pooled10.r(G[f], cov, f"{key}|{f}|{cov}",
                                                     boot=(cov == "arh_rank" and sub in MAIN_SUBS and f in SCORE_FIELDS))
                                     for cov in ("arh_rank", "dense_rank", "log_params")} for f in fields},
                    "pooled_r_8lm": {f: pooled8.r(G[f][:8], "arh_rank") for f in fields},
                    "vision_axis_llama": {f: {k: v for k, v in S12.vision_trend(G[f][IL]).items()
                                              if k in ("Tr", "p_one", "p_two", "n_cells", "n_perm")} for f in SCORE_FIELDS},
                    "openllama_7b_gt_3b_frac": {f: float(np.mean(G[f][I_OL7] > G[f][I_OL3])) for f in SCORE_FIELDS},
                    "mean_by_lm": {f: {lm: float(G[f][i].mean()) for i, lm in enumerate(POOLED)} for f in fields},
                    "grids_llama": {f: G[f][IL].tolist() for f in fields},
                }
                blocks[met][sub] = blk
            log(f"{pre} {met}: trend blocks done")
        R = {"blocks": blocks}
        # --- A1 selection, A2 inflation, A3 counterfactuals, added diagnostics (LLaMA axis; pooled where stated)
        diag = {}
        for met in METRICS:
            gT = grid(Pd, POOLED, met, "full", "s_obs")
            gc = grid(Pd, POOLED, met, "full", "c_alpha")
            ge = grid(Pd, POOLED, met, "full", "c_entry")
            gn = grid(Pd, POOLED, met, "full", "null_mean_entry")
            gs = grid(Pd, POOLED, met, "full", "s_cal")
            gu = grid(Pd, POOLED, met, "full", "uncensored")
            gm = grid(Pd, POOLED, met, "full", "null_mean")
            cf = {"1_T_raw": gT, "2_T_minus_calpha": gT - gc, "3_s_cal_ARH": gs, "3b_uncapped_ratio": (gT - gc) / (1.0 - gc),
                  "4_T_minus_c_entry": gT - ge, "5_T_minus_entry_null_mean": gT - gn, "6_uncensored_T_minus_mean_Tk": gu,
                  "c_alpha": gc, "inflation_calpha_minus_c_entry": gc - ge, "mean_Tk": gm,
                  "level_minus_multiplicity": np.array([[Pd[(lm, v)][met]["diag"]["level_minus_multiplicity"] for v in L.VIT_STEMS] for lm in POOLED]),
                  "max_entry_null_mean": np.array([[Pd[(lm, v)][met]["diag"]["max_entry_null_mean"] for v in L.VIT_STEMS] for lm in POOLED])}
            cfo = {}
            for k, v in cf.items():
                tl = fam_trend(v[IL], LLAMA)
                cfo[k] = {"Tr_L": tl["Tr"], "Tr_L_p_one": tl["p_one"], "pooled10_arh_r": pooled10.r(v, "arh_rank")["r"],
                          "mean_by_llama": {lm: float(v[i].mean()) for i, lm in zip(IL, LLAMA)}, "mean_llama": float(v[IL].mean())}
            ra = grid(Pd, POOLED, met, "full", "rel_a")[IL]
            rb = grid(Pd, POOLED, met, "full", "rel_b")[IL]
            sel = {"rel_a_by_lm": {lm: float(ra[i].mean()) for i, lm in enumerate(LLAMA)},
                   "rel_b_by_lm": {lm: float(rb[i].mean()) for i, lm in enumerate(LLAMA)},
                   "a_grid": grid(Pd, LLAMA, met, "full", "a").astype(int).tolist(),
                   "b_grid": grid(Pd, LLAMA, met, "full", "b").astype(int).tolist(),
                   "n_tied_max_extra": int(grid(Pd, LLAMA, met, "full", "n_tied_max").sum() - 3 * NV),
                   "null_argmax_rel_a_by_lm": {lm: float(grid(Pd, LLAMA, met, "full", "null_argmax_rel_a_mean")[i].mean())
                                               for i, lm in enumerate(LLAMA)},
                   "null_argmax_same_as_obs_mean": float(grid(Pd, LLAMA, met, "full", "null_argmax_frac_same_as_obs").mean()),
                   "rel_a_Tr_L": fam_trend(ra, LLAMA)["Tr"]}
            if met == "cka_b":
                sel["sel_agree_cka_b_mknn_frac_llama"] = float(np.mean([Pd[(lm, v)]["sel_agree_cka_b_mknn"]
                                                                        for lm in LLAMA for v in L.VIT_STEMS]))
            # per-LM-layer null level (max over ViT layers), averaged over ViTs; LM layer holding the highest null
            prof = {}
            for lm in LLAMA:
                Pm = np.array([Pd[(lm, v)][met]["diag"]["lm_layer_null_profile"] for v in L.VIT_STEMS])
                avg = Pm.mean(axis=0)
                u = int(np.argmax(avg))
                prof[lm] = {"profile_mean_over_vits": avg.tolist(), "argmax_layer": u, "argmax_rel": u / (len(avg) - 1),
                            "max": float(avg.max()),
                            "per_pair_argmax_entry_null_mean_rel_a_mean": float(np.mean(
                                [Pd[(lm, v)][met]["diag"]["argmax_entry_null_mean_rel_a"] for v in L.VIT_STEMS]))}
            nl = nlp[IL]
            infl = {"spearman_calpha_vs_nlayerpairs_llama51": float(spearmanr(gc[IL].ravel(), nl.ravel()).correlation),
                    "spearman_calpha_vs_nlayerpairs_pooled170": float(spearmanr(gc.ravel(), nlp.ravel()).correlation),
                    "spearman_inflation_vs_nlayerpairs_llama51": float(spearmanr((gc - ge)[IL].ravel(), nl.ravel()).correlation),
                    "mean_inflation_by_llama": {lm: float((gc - ge)[i].mean()) for i, lm in zip(IL, LLAMA)},
                    "nested_grid_Tr_L_calpha": {s: blocks[met][s]["llama_axis"]["c_alpha"]["Tr"] for s in SUBS if s != "ctm"},
                    "nested_grid_mean_calpha_by_lm": {s: {lm: blocks[met][s]["mean_by_lm"]["c_alpha"][lm] for lm in LLAMA}
                                                      for s in SUBS if s != "ctm"}}
            diag[met] = {"counterfactual": cfo, "selection": sel, "null_profile": prof, "inflation": infl}
        R["diagnostics"] = diag
        # --- readings (plan section 7)
        rd = {}
        for cka in ("cka_b", "cka_d"):
            tb_, tk_ = blocks[cka]["full"], blocks["mknn"]["full"]
            rd[cka] = {
                "arh_pooled10": A.reading_arh(tb_["pooled_r"]["s_obs"]["arh_rank"]["r"], tb_["pooled_r"]["s_cal"]["arh_rank"]["r"] or 0.0,
                                              tk_["pooled_r"]["s_obs"]["arh_rank"]["r"], tk_["pooled_r"]["s_cal"]["arh_rank"]["r"] or 0.0),
                "arh_pooled8_reference": A.reading_arh(tb_["pooled_r_8lm"]["s_obs"]["r"], tb_["pooled_r_8lm"]["s_cal"]["r"] or 0.0,
                                                       tk_["pooled_r_8lm"]["s_obs"]["r"], tk_["pooled_r_8lm"]["s_cal"]["r"] or 0.0),
                "llama_axis": reading_llama(tb_["llama_axis"]["s_obs"]["Tr"], tb_["llama_axis"]["uncensored"]["Tr"],
                                            tk_["llama_axis"]["uncensored"]["Tr"]),
                "inputs": {"Tr_L_raw_cka": tb_["llama_axis"]["s_obs"]["Tr"], "Tr_L_unc_cka": tb_["llama_axis"]["uncensored"]["Tr"],
                           "Tr_L_unc_mknn": tk_["llama_axis"]["uncensored"]["Tr"],
                           "r_before_cka": tb_["pooled_r"]["s_obs"]["arh_rank"]["r"], "r_after_cka": tb_["pooled_r"]["s_cal"]["arh_rank"]["r"],
                           "r_before_mknn": tk_["pooled_r"]["s_obs"]["arh_rank"]["r"], "r_after_mknn": tk_["pooled_r"]["s_cal"]["arh_rank"]["r"]},
                "label": ("primary" if (pre == PRIMARY and cka == "cka_b") else
                          "exploratory (debiased CKA)" if cka == "cka_d" else "secondary (l2)"),
            }
        R["readings"] = rd
        R["partB"] = partb(pre, Pd, blocks, pooled10, pb)
        # --- self-checks against the frozen 8-LM indep summary (BLOOMZ Tr and 8-LM r must reproduce)
        sc = {"max_abs_diff_Tr_bloomz": 0.0, "max_abs_diff_r8": 0.0, "n": 0}
        for met in METRICS:
            for sub in ("full", "fixed_r0.5", "fixed_r1.0", "grid9", "lmall_x_vitlast"):
                for f, of in (("s_obs", "before_raw"), ("s_cal", "after_scal"), ("uncensored", "uncensored")):
                    o = old["per_pre"][pre]["trends"][met][sub][of]
                    sc["max_abs_diff_Tr_bloomz"] = max(sc["max_abs_diff_Tr_bloomz"],
                                                       abs(o["s12"]["Tr"] - blocks[met][sub]["bloomz_axis"][f]["Tr"]))
                    if o["arh"]["arh_rank"]["r"] is not None:
                        sc["max_abs_diff_r8"] = max(sc["max_abs_diff_r8"],
                                                    abs(o["arh"]["arh_rank"]["r"] - blocks[met][sub]["pooled_r_8lm"][f]["r"]))
                    sc["n"] += 1
        sc["pass"] = bool(sc["max_abs_diff_Tr_bloomz"] <= 1e-12 and sc["max_abs_diff_r8"] <= 1e-12)
        summary["self_checks"][pre] = sc
        log(f"{pre}: self-check vs frozen 8-LM summary {sc}")
        # per-pair record of the 51 LLaMA-axis pairs
        R["pairs_llama"] = {f"{lm}|{vit}": Pd[(lm, vit)] for lm in LLAMA for vit in L.VIT_STEMS}
        summary["per_pre"][pre] = R
    # --- section 10 (primary pre only)
    P = summary["per_pre"][PRIMARY]["partB"]
    items = {
        "item1_premise_failure_P-L2a": P["P-L2a"]["outcome"] == "fails",
        "item2_P-L2_contradicted": P["P-L2"]["outcome_scal"] == "contradicted",
        "item3_P-L1_or_P-L4_fails": P["P-L1"]["outcome"] == "fails" or P["P-L4"]["outcome"] == "fails",
        "item4_P-L3_or_P-L5_reversed": any(v.get("outcome") == "reversed" for v in P["P-L3"].values())
                                       or P["P-L5"]["outcome"] == "reversed",
    }
    impl = {"P-L1_deductive_violations": P["P-L1"]["deductive_check"]["s_cal"]["n_violations"]
            + P["P-L1"]["deductive_check"]["uncensored"]["n_violations"],
            "P-L2a_deductive_violations": P["P-L2a"]["deductive_check"]["n_violations"]}
    summary["section10_primary"] = {"items": items, "implementation_error_flags": impl,
                                    "any_against_corollary_use": items["item1_premise_failure_P-L2a"] or items["item2_P-L2_contradicted"],
                                    "any_against_claims_after_corollary": items["item3_P-L1_or_P-L4_fails"] or items["item4_P-L3_or_P-L5_reversed"]}
    # provenance
    files = {p: sha(os.path.join(HERE, p)) for p in sorted(os.listdir(HERE)) if p.endswith((".py", ".sh", ".md", ".diff"))}
    summary["provenance"] = {"code_sha256": files, "partB_features_sha256": sha(PARTB),
                             "partB_predictions_sha256": sha(os.path.join(L.PROJ, "analysis", "s12_layermax_llama", "PARTB_PREDICTIONS.md")),
                             "plan_sha256": sha(os.path.join(L.PROJ, "analysis", "s12_layermax_llama", "ANALYSIS_PLAN_LLAMA.md")),
                             "old_indep_summary_sha256": sha(OLD_SUMMARY),
                             "validation_files": {f: sha(os.path.join(L.OUT, f)) for f in
                                                  ("validation.json", "validation_vl.json", "validation_vl6_arh.json")
                                                  if os.path.exists(os.path.join(L.OUT, f))},
                             "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "secs": time.time() - t0}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        return L.jsonable(o)
    out = os.path.join(L.OUT, "summary.json")
    json.dump(clean(summary), open(out + ".tmp", "w"), indent=1)
    os.replace(out + ".tmp", out)
    log(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
