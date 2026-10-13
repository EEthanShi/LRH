"""PYT-AXIS aggregation of implementation E (S0b sub-role pyt-code-E). GATED (plan F1(a)).

Copied from analysis/s12_layermax_openllama/early/aggregate_ol.py (sha256 f7043a9e...) and rewritten for
analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md (sha256 6de89603...879e7b; sections 5-10) and the verdict contract
analysis/s12_layermax_pythia/VERDICTS_SPEC_PYT.md (written by role pyt-code-infra; E was conformed to the version with
sha256 73cccb6c..., read 2026-10-10 ~01:35Z; sections A, B, C, H, R, D, G are E's; F and K are written by
forecast_pyt.py). This file is a rewrite: every function whose name is not in the "kept" list below is new (each
carries "PYT-AXIS" on its def line or in its docstring); the .diff against orig_copy/aggregate_ol.py documents every
change. Kept from the OpenLLaMA copy: sha, fnum, isfin, sgn, cp95, median, _mknn_exact and _ctm_numpy and load_record
(one field added: the entry null mean at the M_unc argmax, for P-P5a with x = unc), val, pear, pvals, pb_lm, and the
OpenLLaMA section A-C verdict code, kept as ol_legacy_compute() so that V-P8(a) runs the generalised engine on the
frozen OpenLLaMA pair files. Generalised (plan section 3): the trend arithmetic (Spearman = Pearson on average ranks,
analysis/s12/aggregate.py::_rho_matrix, read-only import) over n sizes with exact enumeration of the n! relabellings
(perm_rows, rho_table, tr_series), the common-ViT-set rule for every difference (d_stat; the per-score exclusion
survives only in legacy mode and as the x|dtr_legacy extra), the order-change counts over every size comparison, the
keep identity, the sign rules, and the IUT, gate, Holm and TOST of plan section 5.4.

Exactness (plan F2, spec 2): every mKNN quantity is a fractions.Fraction recomputed from the integer counts of the pair
files; ranks, ties and order relations use exact values; CKA ties only on exact float64 equality (near-tie audit at
1e-12 relative); relabelling and bootstrap counts with the 1e-9 tolerances of spec 2; category thresholds without
tolerance.

Keys: exactly verdict_keys.required_keys("EI") of the shared analysis/s12_layermax_pythia/verdict_keys.py (sections A,
B, C, H, R, D, G), checked at the end of main(); every key of a pre that was not run is null (spec 9). Keys starting with
x| are extras that are never compared: per-score-exclusion differences, Tr bootstrap CIs, decompositions, vision axis,
null profiles, D7 inputs, the E-side pre-category check flags, Part B decisions recomputed (x|check, never used), and
E's own P-P2b and plan-7 category (x|E_own|..., an independent cross-check of forecast_pyt.py's F and K).

Inputs (read-only except E's own outputs):
  <RES>/pairs/<pre>/EleutherAI__pythia-<size>__<vit>.{json,npz}   E main run (S7)
  results/S1_2_layermax_pythia/partB/partB_features_pyt.json      Part B (spec 11), clip_thresholds_partB_pyt.json
  results/S1_2_layermax/pairs/<pre>/                              BLOOMZ, llama-7b, open_llama_3b/7b (section D)
  results/S1_2_layermax_llama/early/pairs/<pre>/                  llama-13b/30b (section D)
  results/S1_2_layermax_openllama/early/pairs/<pre>/              open_llama_13b (section D)
  <RES>/validation/*.json, <RES>/cache/clip.json                  E's own validation flags and thresholds

Outputs (no verdict value is printed, plan section 11):
  <RES>/verdicts_pyt.json  {"schema": "PYT-VERDICTS v1", "impl": "E", "inputs": {path: sha256}, "v": {...}}
  <RES>/summary.json       the same values plus inputs, timings and counts
  (with --pre prh_clip+l2 only, at S7: the same two files with every l2 key null and G|l2_run false, spec 9; a file
  that already exists is first copied aside to <name>.prev-<UTC>, never deleted)
  --legacy / --legacy-common (S0b test mode only, spec 0): <RES>/legacy/verdicts_ol_legacy[_common].json, schema
  OL-VERDICTS v1, the OpenLLaMA section A-C keys on the frozen OpenLLaMA pair files (V-P8(a)/(b)).

Run (step S7 for the primary pre, so that V-P5 can compare verdicts before the l2 go/no-go; step S8 for both pre):
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_pythia/launch_pyt.sh pyt_E_aggregate_prh 4 -- \\
      venv/bin/python analysis/s12_layermax_pythia/early/aggregate_pyt.py --pre prh_clip+l2
  ... pyt_E_aggregate 4 -- venv/bin/python analysis/s12_layermax_pythia/early/aggregate_pyt.py      (both pre)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402  (constants and functions only; reads no data at import)

PRES = LC.PRES
METRICS = ("mknn", "cka_biased", "cka_debiased")
CKAS = ("cka_biased", "cka_debiased")
SUBSETS = ("full", "fixed_r0.5", "fixed_r1.0", "grid2", "grid3", "grid5", "grid9", "grid17",
           "LMall_ViTlast", "LMlast_ViTall", "noLMemb", "LMdeep", "LMemb_only")
NEW_SUBSETS = ("noLMemb", "LMdeep", "LMemb_only")    # absent from the frozen 8-LM pair files
SIZES = LC.SIZES  # PYT-AXIS
PYT = [LC.SLUG[s] for s in SIZES]  # PYT-AXIS
VITS = [s for s, _ in LC.VITS]
NV = len(VITS)
BLOOMZ = ["bigscience__bloomz-560m", "bigscience__bloomz-1b1", "bigscience__bloomz-1b7", "bigscience__bloomz-3b",
          "bigscience__bloomz-7b1"]
LLAMA1 = ["huggyllama__llama-7b", "huggyllama__llama-13b", "huggyllama__llama-30b"]
OPENLLAMA = ["openlm-research__open_llama_3b", "openlm-research__open_llama_7b",  # PYT-AXIS
             "openlm-research__open_llama_13b"]  # PYT-AXIS
KNOWN11 = BLOOMZ + OPENLLAMA + LLAMA1  # PYT-AXIS: ARH order (descriptive panels only)
FAMILIES = {"BLOOMZ": BLOOMZ, "LLaMA-1": LLAMA1, "OpenLLaMA": OPENLLAMA}  # PYT-AXIS (Pythia added at run time)
CNT = LC.M_ITEMS * LC.KNN_K                               # 10240
K = LC.K_PERM
OIDX = 476                                                # ceil(0.95 * 501), 1-based (hrhv2.calibrate.order_index)
TOL = 1e-9                                                # plan F2
NB = 2000
MAT = 0.10
DELTA_B = 0.005
TAUS = (0.03, 0.06, 0.10, 0.15, 0.20)
DBS = (0.0025, 0.005, 0.0075, 0.01)
MARGINS = (0.05, 0.10, 0.15)  # PYT-AXIS: plan 5.6 TOST and magnitude margins
HS = (0.0004, 0.0006, 0.0008, 0.0010, 0.0015)  # PYT-AXIS: plan 5.6 P-P2b h-curve
BASE_MKNN = Fraction(LC.KNN_K, LC.M_ITEMS - 1)
ALPHA = 0.05  # PYT-AXIS
LOSS_R, LOSS_C = 0.60, 0.40  # PYT-AXIS: plan 7 ("loss" needs R >= 0.60 and C <= 0.40)
MAT_TR, MAT_GAP_MULT = 0.75, 3  # PYT-AXIS: plan B4 (|Tr| >= 0.75, median end-to-end gap >= 3 delta_b)
SWAP = 2.0 / 56.0 / 17.0  # PYT-AXIS: one adjacent swap moves Tr_P by 0.0021 at 17 ViTs (plan F2, 5.6)
# PYT-AXIS: plan section 0 / 9 B6 frozen band constants: f(PR*) = a + b / sqrt(PR*), half-width h, fitted PR* minimum
BAND = {"prh_clip+l2": (0.001069, 0.012920, 0.0008, 124.5), "l2": (0.001551, 0.006843, 0.0018, 79.6)}
FLAT_BAND = {"prh_clip+l2": (0.0010, 0.0040), "l2": (0.0010, 0.0045)}  # PYT-AXIS: sensitivity only
MU_BAND = (-0.0003, 0.0008)  # PYT-AXIS: uncensored band of mu - Bbar (plan 9 B6, secondary)
BOOT = "S1.2-layermax-pythia|boot"  # PYT-AXIS: plan 5.4 rng key prefix
FIVE = ("410m", "1b", "1.4b", "2.8b", "6.9b")  # PYT-AXIS: plan 5.5(b)
FAMILIES4 = ("in21k", "MAE", "DINOv2", "CLIP")  # PYT-AXIS: plan 1 / spec 1 ViT families (section 5.5(d))
VIT_FAMILY4 = {v: ("in21k" if "augreg_in21k" in v else "MAE" if v.endswith(".mae") else  # PYT-AXIS
                   "DINOv2" if "dinov2" in v else "CLIP") for v in VITS}  # PYT-AXIS
PRIMARY = ("prh_clip+l2", "s_cal")  # PYT-AXIS
A12 = None                                                # analysis/s12/aggregate.py, imported after the gate


def _load_a12():
    spec = importlib.util.spec_from_file_location("s12_aggregate", os.path.join(LC.PROJ, "analysis", "s12", "aggregate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ensure_a12():  # PYT-AXIS: for importers (vl8/vl9) that do not run main()
    global A12
    if A12 is None:
        A12 = _load_a12()
    return A12


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fnum(v):
    """JSON value of a number: float(Fraction) for mKNN (spec section 2), floats as they are, None stays."""
    if v is None:
        return None
    if isinstance(v, Fraction):
        return float(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return float(v)


def isfin(v):
    return v is not None and (isinstance(v, Fraction) or math.isfinite(float(v)))


def sgn(x):
    if not isfin(x):
        return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)


def cp95(x, n):
    """Clopper-Pearson 95% (spec section 9)."""
    from scipy.stats import beta
    if n == 0:
        return None, None
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    return lo, hi


def median(vals):
    return float(np.median(np.array([float(v) for v in vals], dtype=np.float64)))


def ceil_frac(n, num, den):  # PYT-AXIS: ceil(n * num/den) in exact integers
    return -((-n * num) // den)


# ----------------------------------------------------------------------------
# loading: one record per (pre, LM, ViT)  (kept from the OpenLLaMA copy)
# ----------------------------------------------------------------------------

READ_FILES = {}     # path -> role, for the inputs digest


def _mknn_exact(z, subs, full_only):
    """Exact mKNN summaries from the integer counts (plan F2)."""
    obs = np.rint(np.asarray(z["mknn.entry_obs"], dtype=np.float64) * CNT).astype(np.int64)
    null = np.asarray(z["mknn.null_counts_int16"]).astype(np.int64)            # (K, LA, LB)
    assert null.shape[0] == K and null.shape[1:] == obs.shape
    out = {}
    ent_c = np.sort(np.concatenate([obs[None], null], axis=0), axis=0)[OIDX - 1]   # entry c_alpha counts
    ent_sum = null.sum(axis=0)                                                       # entry null sums
    for name, (ia, ib) in subs.items():
        if full_only and name != "full":
            continue
        so = obs[np.ix_(ia, ib)]
        j = int(np.argmax(so.ravel()))                                               # first maximum, row-major
        a, b = ia[j // len(ib)], ib[j % len(ib)]
        T = int(so.ravel()[j])
        Tk = null[:, ia][:, :, ib].reshape(K, -1).max(axis=1)
        c = int(np.sort(np.concatenate([[T], Tk]))[OIDX - 1])
        Tf, cf = Fraction(T, CNT), Fraction(c, CNT)
        mu = Fraction(int(Tk.sum()), K * CNT)
        scal = max((Tf - cf) / (1 - cf), Fraction(0)) if cf < 1 else Fraction(0)
        out[name] = {"T": Tf, "c": cf, "mu": mu, "scal": scal, "unc": Tf - mu,
                     "p": Fraction(1 + int(np.sum(Tk >= T)), K + 1), "aA": a, "aB": b,
                     "c_entry": Fraction(int(ent_c[a, b]), CNT), "mu_entry": Fraction(int(ent_sum[a, b]), K * CNT),
                     "n_tied": int(np.sum(so == T))}
    if full_only:          # pooled-panel LMs: layer-max of the full grid only
        return out
    # calibrate-then-max, exact (first maximum in row-major order)
    LA, LB = obs.shape
    best, bj = None, None
    bu, bju = None, None
    for idx in range(LA * LB):
        a, b = divmod(idx, LB)
        o, c = int(obs[a, b]), int(ent_c[a, b])
        s = Fraction(o - c, CNT - c) if (o > c and c < CNT) else Fraction(0)
        if best is None or s > best:
            best, bj = s, idx
        u = 500 * o - int(ent_sum[a, b])         # K * CNT * (obs - entry null mean), exact integer
        if bu is None or u > bu:
            bu, bju = u, idx
    a, b = divmod(bj, LB)
    out["ctm"] = {"M": best, "M_unc": Fraction(bu, K * CNT), "TM": Fraction(int(obs[a, b]), CNT),
                  "cM": Fraction(int(ent_c[a, b]), CNT), "aA": a, "aB": b,
                  "aA_unc": bju // LB, "aB_unc": bju % LB,
                  "muMunc": Fraction(int(ent_sum[bju // LB, bju % LB]), K * CNT)}   # PYT-AXIS: P-P5a for x = unc
    out["D1_max_entry_mu"] = Fraction(int(ent_sum.max()), K * CNT)
    out["null_profile"] = (ent_sum.max(axis=1) / (K * CNT)).astype(np.float64)          # per LM layer
    return out


def _ctm_numpy(z, met):
    """Protocol (iv) from the stored float64 entry arrays (numpy version of run_layermax.calib_then_max)."""
    obs, c, nm = z[f"{met}.entry_obs"], z[f"{met}.entry_c_alpha"], z[f"{met}.entry_null_mean"]
    den = 1.0 - c
    scal = np.where(den > 0, np.maximum((obs - c) / np.where(den > 0, den, 1.0), 0.0), 0.0)
    unc = obs - nm
    j = int(np.argmax(scal.reshape(-1)))
    ju = int(np.argmax(unc.reshape(-1)))
    LB = obs.shape[1]
    return {"M": float(scal.reshape(-1)[j]), "M_unc": float(unc.reshape(-1)[ju]), "TM": float(obs.reshape(-1)[j]),
            "cM": float(c.reshape(-1)[j]), "aA": j // LB, "aB": j % LB, "aA_unc": ju // LB, "aB_unc": ju % LB,
            "muMunc": float(nm.reshape(-1)[ju])}   # PYT-AXIS: entry null mean at the M_unc argmax (P-P5a, x = unc)


def load_record(base, full_only=False):
    """base = path without extension. Returns the per-pair record used by every computation below."""
    with open(base + ".json") as f:
        rec = json.load(f)
    z = np.load(base + ".npz")
    READ_FILES[base + ".json"] = "pair"
    READ_FILES[base + ".npz"] = "pair"
    LA, LB = int(rec["L_A"]), int(rec["L_B"])
    subs = LC.subsets(LA, LB)
    R = {"lm": rec["lm"], "vit": rec["vit"], "LA": LA, "LB": LB, "perm_key": rec.get("perm_key"), "K": rec.get("K"),
         "has_ctm_json": "calib_then_max" in rec}
    R["mknn"] = _mknn_exact(z, subs, full_only)
    # json float values of mKNN (for the float-vs-exact audit)
    R["mknn_json"] = {q: rec["summaries"]["mknn"][q] for q in rec["summaries"]["mknn"]}
    for met in CKAS:
        d = {}
        for q, (ia, ib) in subs.items():
            if full_only and q != "full":
                continue
            s = rec["summaries"][met].get(q)
            if s is None:                     # frozen 8-LM files lack the three LLaMA-plan subsets
                d[q] = None
                continue
            assert list(s["layers_A"]) == list(ia) and list(s["layers_B"]) == list(ib), (base, met, q)
            d[q] = {"T": float(s["s_obs"]), "c": float(s["c_alpha"]), "mu": float(s["null_mean"]),
                    "scal": float(s["s_cal"]), "unc": float(s["uncensored"]), "p": float(s["p_value"]),
                    "aA": int(s["argmax_A"]), "aB": int(s["argmax_B"]), "n_tied": int(s["n_tied_max"]),
                    "c_entry": float(s["entry_c_alpha_at_argmax"]), "mu_entry": float(s["entry_null_mean_at_argmax"])}
        if not full_only:
            npz_ctm = _ctm_numpy(z, met)
            if "calib_then_max" in rec:
                cj = rec["calib_then_max"][met]
                d["ctm"] = {"M": float(cj["M"]), "M_unc": float(cj["M_unc"]), "TM": float(cj["s_obs_at_argmax"]),
                            "cM": float(cj["c_alpha_at_argmax"]), "aA": int(cj["argmax_A"]), "aB": int(cj["argmax_B"]),
                            "aA_unc": int(cj["M_unc_argmax_A"]), "aB_unc": int(cj["M_unc_argmax_B"])}
                d["ctm_json_vs_npz_maxabs"] = max(abs(d["ctm"][k] - npz_ctm[k]) for k in ("M", "M_unc", "TM", "cM"))
                d["ctm"]["muMunc"] = npz_ctm["muMunc"]   # PYT-AXIS: not in the pair json; from the npz entry array
                d["ctm_argmax_unc_json_equals_npz"] = bool(d["ctm"]["aA_unc"] == npz_ctm["aA_unc"]
                                                           and d["ctm"]["aB_unc"] == npz_ctm["aB_unc"])  # PYT-AXIS
            else:
                d["ctm"] = npz_ctm
            nm = np.asarray(z[f"{met}.entry_null_mean"], dtype=np.float64)
            d["D1_max_entry_mu"] = float(nm.max())
            d["null_profile"] = nm.max(axis=1)
        R[met] = d
    return R


def val(R, met, q, x):
    """Score X of protocol Q (spec section 1) for one record; None if unavailable."""
    if q == "ctm":
        d = R[met]["ctm"]
        return d[{"M": "M", "M_unc": "M_unc", "TM": "TM", "cM": "cM", "muMunc": "muMunc"}[x]]  # PYT-AXIS: cM, muMunc
    d = R[met].get(q)
    if d is None:
        return None
    key = {"raw": "T", "s_cal": "scal", "unc": "unc", "c": "c", "mu": "mu", "c_entry": "c_entry",
           "mu_entry": "mu_entry"}.get(x)
    if key is not None:
        return d[key]
    if x == "shift":
        return d["T"] - d["c"]
    if x == "sel_entry_thr":
        return d["T"] - d["c_entry"]
    if x == "sel_entry_mean":
        return d["T"] - d["mu_entry"]
    if x == "inflation":
        return d["c"] - d["c_entry"]
    raise KeyError(x)


# ----------------------------------------------------------------------------
# PYT-AXIS: the generalised trend engine (plan 5.1: n sizes, exact n! relabellings, common-ViT-set rule, legacy)
# ----------------------------------------------------------------------------

_PERMS = {}


def perm_rows(n):  # PYT-AXIS
    """(n!, n) float array of the relabelled size ranks; row 0 = identity (1..n), itertools order."""
    if n not in _PERMS:
        _PERMS[n] = np.array(list(itertools.permutations(range(1, n + 1))), dtype=np.float64)
    return _PERMS[n]


def rank_avg(t):  # PYT-AXIS: average ranks of n values by exact comparisons (Fraction or float); = rank3 for n = 3
    n = len(t)
    return [1.0 + sum(1 for j in range(n) if t[j] < t[i]) + 0.5 * sum(1 for j in range(n) if j != i and t[j] == t[i])
            for i in range(n)]


class RT:  # PYT-AXIS: rho table of one score
    __slots__ = ("RH", "valid", "const", "ranks", "nexcl", "n", "anynone")

    def __init__(self, RH, valid, const, ranks, nexcl, n, anynone):
        self.RH, self.valid, self.const, self.ranks = RH, valid, const, ranks
        self.nexcl, self.n, self.anynone = nexcl, n, anynone


def rho_table(rows):  # PYT-AXIS
    """rows: list over ViTs of n-tuples (values in size order). A ViT with a non-finite value or n equal values is
    invalid (rho NaN); rho(sigma) = Spearman between the relabelled ranks and the values, as Pearson on average ranks
    (analysis/s12/aggregate.py::_rho_matrix, the S1.2 and OpenLLaMA arithmetic)."""
    n = len(rows[0])
    P = perm_rows(n)
    nv = len(rows)
    RH = np.full((P.shape[0], nv), np.nan)
    valid = np.zeros(nv, dtype=bool)
    const = np.zeros(nv, dtype=bool)
    ranks = [None] * nv
    ok = []
    anynone = any(x is None for t in rows for x in t)
    for v, t in enumerate(rows):
        if not all(isfin(x) for x in t):
            continue
        if all(x == t[0] for x in t):
            const[v] = True
            continue
        valid[v] = True
        ok.append(v)
        ranks[v] = rank_avg(t)
    if ok:
        Rk = np.array([[ranks[v][i] for v in ok] for i in range(n)], dtype=np.float64)
        RH[:, ok] = ensure_a12()._rho_matrix(P, Rk)
    return RT(RH, valid, const, ranks, nv - len(ok), n, anynone)


def tr_series(rt, mask=None):  # PYT-AXIS: Tr(sigma) over the ViTs of mask (default: the valid ones); None if empty
    m = rt.valid if mask is None else mask
    if not m.any():
        return None
    return rt.RH[:, m].mean(axis=1)


def tr0_value(rt):  # PYT-AXIS: plan 5.1 sensitivity: rho_v := 0 for a constant vector, all ViTs with finite values
    fin = rt.valid | rt.const
    if not fin.any():
        return None
    v = np.where(rt.valid, rt.RH[0], 0.0)
    return float(v[fin].mean())


def diff_series(rx, rb, legacy=False):  # PYT-AXIS
    """Tr(X)(sigma) - Tr(base)(sigma) under the same sigma. Common mode (plan 5.1): both terms over V(base, X), the
    ViTs valid for both; legacy (V-P8 only): each term over its own valid set (the OpenLLaMA per-score exclusion)."""
    if legacy:
        a, b = tr_series(rx), tr_series(rb)
        if a is None or b is None:
            return None, 0
        return a - b, int((rx.valid & rb.valid).sum())
    V = rx.valid & rb.valid
    if not V.any():
        return None, 0
    return rx.RH[:, V].mean(axis=1) - rb.RH[:, V].mean(axis=1), int(V.sum())


def pvals(T, prefix):
    t = float(T[0])
    return {f"{prefix}_pge": float(np.mean(T >= t - TOL)), f"{prefix}_ple": float(np.mean(T <= t + TOL)),
            f"{prefix}_ptwo": float(np.mean(np.abs(T) >= abs(t) - TOL))}


def p_le(D):  # PYT-AXIS: one-sided exact p of a statistic whose alternative is "small": #{D(s) <= D(id) + tol}/n!
    return float(np.mean(D <= float(D[0]) + TOL))


def neartie(rows):  # PYT-AXIS: generalised to n sizes (every size pair)
    n = 0
    for t in rows:
        for a, b in itertools.combinations(range(len(t)), 2):
            x, y = t[a], t[b]
            if isfin(x) and isfin(y) and abs(x - y) <= 1e-12 * max(abs(x), abs(y)):
                n += 1
    return n


def rho_equal_exact(ra, rb):  # PYT-AXIS: exact test of rho(ranks a) == rho(ranks b) against sizes 1..n (I1)
    n = len(ra)
    xc = [Fraction(2 * i - (n + 1), 2) for i in range(1, n + 1)]

    def nss(r):
        rf = [Fraction(x).limit_denominator(4) for x in r]       # average ranks are multiples of 1/2
        mr = sum(rf) / n
        num = sum(xc[i] * (rf[i] - mr) for i in range(n))
        ss = sum((x - mr) ** 2 for x in rf)
        return num, ss
    na, sa = nss(ra)
    nb, sb = nss(rb)
    return (na >= 0) == (nb >= 0) and na * abs(na) * sb == nb * abs(nb) * sa


def boot_draws(vec, key, n_draw=NB):  # PYT-AXIS: plan 5.4 ViT bootstrap: each draw resamples the ViTs with replacement
    from hrhv2 import util
    vec = np.asarray(vec, dtype=np.float64)
    n = vec.size
    rng = util.rng_from_key(key)
    out = np.empty(n_draw)
    for b in range(n_draw):
        out[b] = vec[rng.integers(0, n, n)].mean()
    return out



# ----------------------------------------------------------------------------
# PYT-AXIS: Part B view (VERDICTS_SPEC_PYT section 11; values are read, never recomputed)
# ----------------------------------------------------------------------------

def pb_lm(d, slug):
    if slug in d:
        return d[slug]
    lab = LC.SIZE_LABEL.get(slug)
    if lab in d:
        return d[lab]
    raise KeyError(slug)


def pb_view(PB, pre, lms, labels):  # PYT-AXIS
    """Every Part B quantity E reads (spec section 11). Membership, directions, H2 branch, bands, extrapolation
    flags, widths and depths come from Part B; E recomputes none of them (spec section 1)."""
    P = PB["per_pre"][pre]
    lab = dict(zip(lms, labels))
    out = {"PR_star": {lm: float(pb_lm(P["PR_star"], lm)) for lm in lms},
           "Bbar": {lm: {v: float(pb_lm(P["Bbar"], lm)[v]) for v in VITS} for lm in lms},
           "Blate": {lm: {v: float(pb_lm(P["Blate"], lm)[v]) for v in VITS} for lm in lms},
           "material": bool(P["material_Bbar"]), "s": int(P["s"]), "s_late": int(P["s_late"]),
           "H2_branch": str(P["H2_branch"]),
           "band": {lm: (float(pb_lm(P["band"], lab[lm])["lo"]), float(pb_lm(P["band"], lab[lm])["hi"])) for lm in lms},
           "band_flat": tuple(float(z) for z in P["band_flat"]),
           "mu_band": tuple(float(z) for z in PB["mu_band"]),
           "extrapolated": {lm: bool(pb_lm(P["extrapolated"], lab[lm])) for lm in lms},
           "u_star_rel": {lm: (P.get("u_star_rel") or {}).get(lm) for lm in lms},
           "width": {lm: int(pb_lm(PB["width"], lm)) for lm in lms},
           "depth": {lm: int(pb_lm(PB["depth_blocks"], lm)) for lm in lms},
           "num_params": {lm: int(pb_lm(PB["num_params"], lm)) for lm in lms},
           "L_stored": {lm: int(pb_lm(PB["L_stored"], lm)) for lm in lms},
           "material_late": bool(P["material_Blate"])}
    return out


def material_from(G, db=DELTA_B):  # PYT-AXIS: plan B4 recomputed (x|check only; decisions come from Part B)
    rows = [tuple(G[:, j]) for j in range(G.shape[1])]
    T = tr_series(rho_table(rows))
    tr = None if T is None else float(T[0])
    if tr is None:
        return {"Tr": None, "material": False, "s": 1, "median_gap": None}
    s = 1 if tr >= 0 else -1
    gap = float(np.median(s * (G[-1] - G[0])))
    return {"Tr": tr, "s": s, "median_gap": gap,
            "material": bool(abs(tr) >= MAT_TR - TOL and gap >= MAT_GAP_MULT * db)}


# ----------------------------------------------------------------------------
# PYT-AXIS: records of one family, rho tables, small helpers
# ----------------------------------------------------------------------------

def _sub(a, b):
    return None if a is None or b is None else a - b


def comparisons(n):  # PYT-AXIS: all size pairs (j < j') of n sizes
    return list(itertools.combinations(range(n), 2))


def scores_of(q):
    return ("M", "M_unc", "TM") if q == "ctm" else ("raw", "s_cal", "unc")


def dscores_of(q):
    return ("M", "M_unc", "TM") if q == "ctm" else ("s_cal", "unc")


def rows_of(REC, lms, pre, met, q, X, vits=None):  # PYT-AXIS
    return [tuple(val(REC[(pre, lm, vit)], met, q, X) for lm in lms) for vit in (vits or VITS)]


class Family:  # PYT-AXIS: the records of one family in size order plus cached rho tables
    def __init__(self, REC, lms, labels):
        self.REC, self.lms, self.labels = REC, list(lms), list(labels)
        self.n = len(lms)
        self._rt = {}

    def rt(self, pre, met, q, X, idx=None):
        key = (pre, met, q, X, tuple(idx) if idx is not None else None)
        if key not in self._rt:
            lms = self.lms if idx is None else [self.lms[i] for i in idx]
            rows = rows_of(self.REC, lms, pre, met, q, X)
            self._rt[key] = rho_table(rows) if not any(x is None for t in rows for x in t) else None
        return self._rt[key]

    def rows(self, pre, met, q, X):
        return rows_of(self.REC, self.lms, pre, met, q, X)


def common_mean(rx, rb, mask=None):  # PYT-AXIS: (mean over V of rho_x - rho_b at the identity, |V|), V = valid for both
    V = rx.valid & rb.valid
    if mask is not None:
        V = V & mask
    if not V.any():
        return None, 0
    return float(rx.RH[0, V].mean() - rb.RH[0, V].mean()), int(V.sum())


def rho0_pair(rx, rb):  # PYT-AXIS: spec 3 dtr0: over the ViTs with finite scores for both, rho := 0 for constants
    fin = (rx.valid | rx.const) & (rb.valid | rb.const)
    if not fin.any():
        return None
    a = np.where(rx.valid, rx.RH[0], 0.0)[fin].mean()
    b = np.where(rb.valid, rb.RH[0], 0.0)[fin].mean()
    return float(a - b)


def ranks_identical(rx, rb, V):  # PYT-AXIS
    return all(rx.ranks[j] == rb.ranks[j] for j in np.flatnonzero(V))


def spearman_avg(a, b):  # PYT-AXIS: spec 2 rho formula on average ranks of both vectors (D4, D6)
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float(np.sum(ra ** 2)) * float(np.sum(rb ** 2)))
    return None if den == 0 else float(np.sum(ra * rb) / den)


def d_stat(F, pre, met, q, X, idx=None, vitmask=None):  # PYT-AXIS
    """Difference of score X against raw (Q; for ctm the base is raw of full), sizes idx (default all), ViTs of
    vitmask (default all), common ViT set. Returns dict: D (series over the relabellings of the sizes used), dr
    (per-ViT delta rho over V, ViT order), nV, degenerate (exact), ranks_identical, V, rx, rb."""
    rb = F.rt(pre, met, "full" if q == "ctm" else q, "raw", idx)
    rx = F.rt(pre, met, q, X, idx)
    out = {"D": None, "dr": None, "nV": 0, "degenerate": None, "ranks_identical": None, "V": None, "rx": rx, "rb": rb}
    if rb is None or rx is None:
        return out
    V = rx.valid & rb.valid
    if vitmask is not None:
        V = V & vitmask
    out["V"] = V
    out["nV"] = int(V.sum())
    if not V.any():
        return out
    out["D"] = rx.RH[:, V].mean(axis=1) - rb.RH[:, V].mean(axis=1)
    out["dr"] = rx.RH[0, V] - rb.RH[0, V]
    js = np.flatnonzero(V)
    out["degenerate"] = bool(all(rho_equal_exact(rx.ranks[j], rb.ranks[j]) for j in js))
    out["ranks_identical"] = bool(ranks_identical(rx, rb, V))
    return out


def bkey(kind, met, q, pre, x, suffix=""):  # PYT-AXIS: spec 6 rng keys (+ "|unc" for x = unc, then the suffix)
    k = f"{BOOT}|{kind}|{met}|{q}|{pre}"
    if x == "unc":
        k += "|unc"
    return k + suffix


def p_mag_tol(draws, s, margin=MAT):  # PYT-AXIS: spec 2/6: count s*D*_b >= -m - 1e-9
    return float((1 + int(np.sum(s * draws >= -margin - TOL))) / (draws.size + 1))


def p_tost_tol(draws, obs, margin=MAT):  # PYT-AXIS: spec 6 TOST with the conservative 1e-9 tolerance
    lo = float((1 + int(np.sum(draws <= -margin + TOL))) / (draws.size + 1))
    hi = float((1 + int(np.sum(draws >= margin - TOL))) / (draws.size + 1))
    pt = max(lo, hi) if (obs is not None and -margin < obs < margin) else 1.0
    return lo, hi, pt


def xname(x):  # PYT-AXIS: the score of x on the ctm protocol
    return "M" if x == "s_cal" else "M_unc"


# ----------------------------------------------------------------------------
# PYT-AXIS: section A (trends) and section B (order changes)
# ----------------------------------------------------------------------------

def sec_A(F, pre, v, x, boot_ci=True):  # PYT-AXIS
    for met in METRICS:
        for q in SUBSETS + ("ctm",):
            for X in scores_of(q):
                key = f"{pre}|{met}|{q}|{X}"
                rt = F.rt(pre, met, q, X)
                T = None if rt is None else tr_series(rt)
                v[f"tr|{key}"] = None if T is None else float(T[0])
                v[f"tr_nconst|{key}"] = None if rt is None else int(rt.nexcl)
                if T is None:
                    v.update({f"tr_pge|{key}": None, f"tr_ple|{key}": None, f"tr_ptwo|{key}": None})
                else:
                    for k_, val_ in pvals(T, "tr").items():
                        v[f"{k_}|{key}"] = val_
                v[f"tr0|{key}"] = None if rt is None else tr0_value(rt)
                if met != "mknn":
                    v[f"neartie|{key}"] = None if rt is None else neartie(F.rows(pre, met, q, X))
                if boot_ci and T is not None:
                    _, lo, hi = A12.boot_vits(np.array([rt.RH[0, j] if rt.valid[j] else np.nan for j in range(NV)]),
                                              f"layermax-pythia|{key}")
                    x[f"x|tr_ci|{key}"] = [lo, hi]
            for X in dscores_of(q):
                key = f"{pre}|{met}|{q}|{X}"
                d = d_stat(F, pre, met, q, X)
                D = d["D"]
                v[f"dtr|{key}"] = None if D is None else float(D[0])
                v[f"dtr_nv|{key}"] = None if d["rx"] is None or d["rb"] is None else d["nV"]
                if D is None:
                    v.update({f"dtr_pge|{key}": None, f"dtr_ple|{key}": None, f"dtr_ptwo|{key}": None})
                else:
                    for k_, val_ in pvals(D, "dtr").items():
                        v[f"{k_}|{key}"] = val_
                v[f"dtr0|{key}"] = None if (d["rx"] is None or d["rb"] is None) else rho0_pair(d["rx"], d["rb"])
                if d["rx"] is not None and d["rb"] is not None:   # per-score exclusion (OpenLLaMA rule), extra
                    a, b = tr_series(d["rx"]), tr_series(d["rb"])
                    x[f"x|dtr_legacy|{key}"] = None if a is None or b is None else float(a[0] - b[0])
        rM, rTM = F.rt(pre, met, "ctm", "M"), F.rt(pre, met, "ctm", "TM")
        rraw = F.rt(pre, met, "full", "raw")
        v[f"d3cal|{pre}|{met}"] = None if rM is None or rTM is None else common_mean(rM, rTM)[0]
        v[f"d3sel|{pre}|{met}"] = None if rTM is None or rraw is None else common_mean(rTM, rraw)[0]


def sec_B(F, pre, v):  # PYT-AXIS: every size comparison (357 on the Pythia axis)
    cmp_ = comparisons(F.n)
    ncmp = len(cmp_) * NV
    for met in METRICS:
        for q in SUBSETS + ("ctm",):
            for X in (("M", "M_unc") if q == "ctm" else ("s_cal", "unc")):
                c = {k: 0 for k in ("down", "up", "c2d", "d2c", "c2t", "t2c", "d2t", "t2d")}
                for vit in VITS:
                    for a, b in cmp_:
                        Ra, Rb = F.REC[(pre, F.lms[a], vit)], F.REC[(pre, F.lms[b], vit)]
                        qr = "full" if q == "ctm" else q
                        ra, rb = val(Ra, met, qr, "raw"), val(Rb, met, qr, "raw")
                        xa, xb = val(Ra, met, q, X), val(Rb, met, q, X)
                        if None in (ra, rb, xa, xb):
                            continue
                        dr, dx = sgn(rb - ra), sgn(xb - xa)
                        c["down"] += int(dx < dr)
                        c["up"] += int(dx > dr)
                        c["c2d"] += int(dr == 1 and dx == -1)
                        c["d2c"] += int(dr == -1 and dx == 1)
                        c["c2t"] += int(dr == 1 and dx == 0)
                        c["t2c"] += int(dr == 0 and dx == 1)
                        c["d2t"] += int(dr == -1 and dx == 0)
                        c["t2d"] += int(dr == 0 and dx == -1)
                key = f"{pre}|{met}|{q}|{X}"
                for k_, n_ in c.items():
                    v[f"chg|{key}|all|{k_}"] = n_
                fm, fp = c["c2d"], c["d2c"]
                lo, hi = cp95(fm, ncmp)
                plo, phi = cp95(fp, ncmp)
                v.update({f"fcnt|{key}|n": ncmp, f"fcnt|{key}|fminus": fm, f"fcnt|{key}|fplus": fp,
                          f"fcnt|{key}|F": (fm - fp) / ncmp, f"fcnt|{key}|fm_cp_lo": lo, f"fcnt|{key}|fm_cp_hi": hi,
                          f"fcnt|{key}|fp_cp_lo": plo, f"fcnt|{key}|fp_cp_hi": phi})


# ----------------------------------------------------------------------------
# PYT-AXIS: section C (Part B rules, implementation checks, attribution)
# ----------------------------------------------------------------------------

def keep_sign_checks(F, pre, v, xs=("s_cal", "unc")):  # PYT-AXIS: keep identity and per-comparison sign rule
    cmp_ = comparisons(F.n)
    for met in METRICS:
        for q in SUBSETS:
            for X in xs:
                nk = ns = 0
                for vit in VITS:
                    for a, b in cmp_:
                        Ra = F.REC[(pre, F.lms[a], vit)][met].get(q)
                        Rb = F.REC[(pre, F.lms[b], vit)][met].get(q)
                        if Ra is None or Rb is None:
                            continue
                        Ta, Tb = Ra["T"], Rb["T"]
                        kx = "scal" if X == "s_cal" else "unc"
                        Xa, Xb = Ra[kx], Rb[kx]
                        if X == "s_cal":
                            ta, tb = Ra["c"], Rb["c"]
                            if Xa > 0 and Xb > 0:
                                lhs, rhs = Tb - Ta, (1 - Ta) * (tb - ta) / (1 - ta)
                                if not (met != "mknn" and abs(lhs - rhs) <= 1e-12):
                                    nk += int((lhs >= rhs) != (Xb - Xa >= 0))
                        else:
                            ta, tb = Ra["mu"], Rb["mu"]
                            lhs, rhs = Tb - Ta, tb - ta
                            if not (met != "mknn" and abs(lhs - rhs) <= 1e-12):
                                nk += int((lhs >= rhs) != (Xb - Xa >= 0))
                        dr, dx = sgn(Tb - Ta), sgn(Xb - Xa)
                        if tb > ta:
                            ns += int(dr <= 0 and dx == 1)
                        elif tb < ta:
                            ns += int(dr >= 0 and dx == -1)
                        else:
                            ns += int(dx == -dr and dx != 0)
                v[f"keepid|{pre}|{met}|{q}|{X}|nviol"] = nk
                v[f"signrule|{pre}|{met}|{q}|{X}|nviol"] = ns


def censor_tie(xv, rv):  # PYT-AXIS: spec 5: two sizes tied under the score but not under raw
    n = len(xv)
    return any(xv[i] == xv[j] and rv[i] != rv[j] for i in range(n) for j in range(i + 1, n))


def sec_C(F, pre, PBV, v, x):  # PYT-AXIS
    cmp_ = comparisons(F.n)
    Bbar, Blate = PBV["Bbar"], PBV["Blate"]
    keep_sign_checks(F, pre, v)
    # ---- per-ViT sign rule (spec 5 signvit; plan 8 with the censoring-tie exclusion)
    for met in METRICS:
        for q in SUBSETS:
            for X in ("s_cal", "unc"):
                thr = "c" if X == "s_cal" else "mu"
                rx, rb = F.rt(pre, met, q, X), F.rt(pre, met, q, "raw")
                c = {"nmono_up": 0, "nviol_up": 0, "nmono_down": 0, "nviol_down": 0, "nexcl_tie": 0}
                if rx is not None and rb is not None:
                    xrows, rrows = F.rows(pre, met, q, X), F.rows(pre, met, q, "raw")
                    for j, vit in enumerate(VITS):
                        if not (rx.valid[j] and rb.valid[j]):
                            continue
                        th = [F.REC[(pre, lm, vit)][met][q][thr] for lm in F.lms]
                        up = all(th[i] <= th[i + 1] for i in range(F.n - 1))
                        down = all(th[i] >= th[i + 1] for i in range(F.n - 1))
                        if not (up or down):
                            continue
                        if censor_tie(xrows[j], rrows[j]):
                            c["nexcl_tie"] += 1
                            continue
                        if up:
                            c["nmono_up"] += 1
                            c["nviol_up"] += int(rx.RH[0, j] > rb.RH[0, j] + 1e-12)
                        if down:
                            c["nmono_down"] += 1
                            c["nviol_down"] += int(rx.RH[0, j] < rb.RH[0, j] - 1e-12)
                for k_, n_ in c.items():
                    v[f"signvit|{pre}|{met}|{q}|{X}|{k_}"] = n_
    # ---- P-P1 (mKNN, exact rationals)
    for X, thr in (("s_cal", "c"), ("unc", "mu")):
        nq = nrev = ntie = 0
        for vit in VITS:
            for a, b in cmp_:
                Ra, Rb = F.REC[(pre, F.lms[a], vit)]["mknn"]["full"], F.REC[(pre, F.lms[b], vit)]["mknn"]["full"]
                za, zb = abs(Ra[thr] - BASE_MKNN), abs(Rb[thr] - BASE_MKNN)
                gap = Rb["T"] - Ra["T"]
                if abs(gap) > za + zb:
                    nq += 1
                    key_ = "scal" if X == "s_cal" else "unc"
                    dx, dr = sgn(Rb[key_] - Ra[key_]), sgn(gap)
                    nrev += int(dx == -dr and dx != 0)
                    ntie += int(dx == 0 and dr != 0)
        v.update({f"PP1|{pre}|{X}|nq": nq, f"PP1|{pre}|{X}|nrev": nrev, f"PP1|{pre}|{X}|ntie": ntie,
                  f"PP1|{pre}|{X}|holds": bool(nrev == 0)})
    v[f"PP1|{pre}|holds"] = bool(v[f"PP1|{pre}|s_cal|holds"] and v[f"PP1|{pre}|unc|holds"])
    # ---- P-P2a(i), P-P2a(ii), P-P2a
    for X, thr in (("s_cal", "c"), ("unc", "mu")):
        nin = 0
        failing = []
        for i, lm in enumerate(F.lms):
            lab = F.labels[i]
            offs = [F.REC[(pre, lm, vit)]["cka_biased"]["full"][thr] - Bbar[lm][vit] for vit in VITS]
            for vit, o in zip(VITS, offs):
                v[f"PP2ai|{pre}|{X}|offset|{lab}|{vit}"] = float(o)
            med = median(offs)
            lo_, hi_ = PBV["band"][lm] if X == "s_cal" else PBV["mu_band"]
            inb = bool(lo_ <= med <= hi_)
            xin = sum(1 for o in offs if lo_ <= o <= hi_)
            clo, chi = cp95(xin, NV)
            v.update({f"PP2ai|{pre}|{X}|median|{lab}": med, f"PP2ai|{pre}|{X}|lo|{lab}": lo_,
                      f"PP2ai|{pre}|{X}|hi|{lab}": hi_, f"PP2ai|{pre}|{X}|in_band|{lab}": inb,
                      f"PP2ai|{pre}|{X}|x|{lab}": xin, f"PP2ai|{pre}|{X}|cp_lo|{lab}": clo,
                      f"PP2ai|{pre}|{X}|cp_hi|{lab}": chi})
            nin += int(inb)
            if not inb:
                failing.append(lab)
        v.update({f"PP2ai|{pre}|{X}|n_models_in_band": nin, f"PP2ai|{pre}|{X}|holds": bool(nin >= F.n - 1),
                  f"PP2ai|{pre}|{X}|failing": ",".join(failing)})
        n = xx = 0
        for vit in VITS:
            for a, b in cmp_:
                la, lb = F.lms[a], F.lms[b]
                gB = Bbar[lb][vit] - Bbar[la][vit]
                if abs(gB) >= DELTA_B:
                    n += 1
                    gc = F.REC[(pre, lb, vit)]["cka_biased"]["full"][thr] - F.REC[(pre, la, vit)]["cka_biased"]["full"][thr]
                    xx += int(sgn(gc) == sgn(gB))
        need9, need67 = ceil_frac(n, 9, 10), ceil_frac(n, 2, 3)
        lo, hi = cp95(xx, n)
        v.update({f"PP2aii|{pre}|{X}|n": n, f"PP2aii|{pre}|{X}|x": xx, f"PP2aii|{pre}|{X}|need": need9,
                  f"PP2aii|{pre}|{X}|need67": need67,
                  f"PP2aii|{pre}|{X}|verdict": "not evaluable" if n < 6 else ("holds" if xx >= need9 else "fails"),
                  f"PP2aii|{pre}|{X}|verdict67": "not evaluable" if n < 6 else ("holds" if xx >= need67 else "fails"),
                  f"PP2aii|{pre}|{X}|cp_lo": lo, f"PP2aii|{pre}|{X}|cp_hi": hi})
        i_ok, ii = v[f"PP2ai|{pre}|{X}|holds"], v[f"PP2aii|{pre}|{X}|verdict"]
        v[f"PP2a|{pre}|{X}|verdict"] = "fails" if (not i_ok or ii == "fails") else ("holds" if ii == "holds"
                                                                                   else "not evaluable")
    # flat-band sensitivity (s_cal only)
    nfl = 0
    for i, lm in enumerate(F.lms):
        med = v[f"PP2ai|{pre}|s_cal|median|{F.labels[i]}"]
        inb = bool(PBV["band_flat"][0] <= med <= PBV["band_flat"][1])
        v[f"PP2ai|{pre}|s_cal|flat|in_band|{F.labels[i]}"] = inb
        nfl += int(inb)
    v[f"PP2ai|{pre}|s_cal|flat|n_models_in_band"] = nfl
    v[f"PP2ai|{pre}|s_cal|flat|holds"] = bool(nfl >= F.n - 1)
    # ---- P-P5a (c^M = entry c_alpha at the M-argmax; for unc the entry null mean at the M_unc-argmax)
    for X, ck in (("s_cal", "cM"), ("unc", "muMunc")):
        n = xx = 0
        for vit in VITS:
            for a, b in cmp_:
                la, lb = F.lms[a], F.lms[b]
                gB = Blate[lb][vit] - Blate[la][vit]
                if abs(gB) >= DELTA_B:
                    n += 1
                    gc = F.REC[(pre, lb, vit)]["cka_biased"]["ctm"][ck] - F.REC[(pre, la, vit)]["cka_biased"]["ctm"][ck]
                    xx += int(sgn(gc) == sgn(gB))
        need = ceil_frac(n, 2, 3)
        lo, hi = cp95(xx, n)
        v.update({f"PP5a|{pre}|{X}|n": n, f"PP5a|{pre}|{X}|x": xx, f"PP5a|{pre}|{X}|need": need,
                  f"PP5a|{pre}|{X}|verdict": "not evaluable" if n < 6 else ("holds" if xx >= need else "fails"),
                  f"PP5a|{pre}|{X}|cp_lo": lo, f"PP5a|{pre}|{X}|cp_hi": hi})
    # ---- P-P3 (descriptive), fixed depth
    for r in ("0.5", "1.0"):
        q = f"fixed_r{r}"
        t_raw = v[f"tr|{pre}|cka_biased|{q}|raw"]
        for X in ("s_cal", "unc"):
            t_x = v[f"tr|{pre}|cka_biased|{q}|{X}"]
            if t_raw is None or abs(t_raw) < MAT:
                dsc = "not applied"
            else:
                s_r = 1 if t_raw > 0 else -1
                if t_x is None:
                    dsc = "undefined (trend undefined)"
                elif s_r * t_x >= MAT:
                    dsc = "keeps sign"
                elif abs(t_x) < MAT:
                    dsc = "lost"
                else:
                    dsc = "reversed"
            v[f"PP3|{pre}|{r}|{X}|desc"] = dsc
    # ---- D3 deductive check (strictly rising c^M; censoring-tie exclusion, spec 5)
    rM, rTM = F.rt(pre, "cka_biased", "ctm", "M"), F.rt(pre, "cka_biased", "ctm", "TM")
    Mrows, TMrows = F.rows(pre, "cka_biased", "ctm", "M"), F.rows(pre, "cka_biased", "ctm", "TM")
    nv3 = nex = 0
    for j, vit in enumerate(VITS):
        cm = [F.REC[(pre, lm, vit)]["cka_biased"]["ctm"]["cM"] for lm in F.lms]
        if not all(cm[i] < cm[i + 1] for i in range(F.n - 1)):
            continue
        if censor_tie(Mrows[j], TMrows[j]):
            nex += 1
            continue
        rm_ = rM.RH[0, j] if rM.valid[j] else 0.0
        rt_ = rTM.RH[0, j] if rTM.valid[j] else 0.0
        nv3 += int(rm_ - rt_ > 1e-12)
    v[f"D3|{pre}|nviol"] = nv3
    v[f"D3|{pre}|nexcl_tie"] = nex
    # ---- plug-in (common ViT set for diff)
    tp = [tuple(F.REC[(pre, lm, vit)]["cka_biased"]["full"]["T"] - Bbar[lm][vit] for lm in F.lms) for vit in VITS]
    rp = rho_table(tp)
    Tp = tr_series(rp)
    v[f"plugin|{pre}|tr"] = None if Tp is None else float(Tp[0])
    ru = F.rt(pre, "cka_biased", "full", "unc")
    dif, nvp = common_mean(rp, ru)
    v[f"plugin|{pre}|diff"] = dif
    v[f"plugin|{pre}|nv"] = nvp
    v[f"plugin|{pre}|within"] = bool(dif is not None and abs(dif) <= MAT)
    # ---- OP-style axis reading (single-score trends)
    for met in CKAS:
        t_raw, t_unc, t_m = v[f"tr|{pre}|{met}|full|raw"], v[f"tr|{pre}|{met}|full|unc"], v[f"tr|{pre}|mknn|full|unc"]
        if t_raw is None or t_raw < 0.60:
            rd = "not applied"
        elif t_unc is not None and t_m is not None and t_unc <= 0.40 and t_m >= 0.60:
            rd = "loses"
        elif t_unc is not None and t_unc >= 0.60:
            rd = "no loss"
        else:
            rd = "intermediate"
        v[f"readO|{pre}|{met}"] = rd
    # ---- D1, D2
    for met in METRICS:
        for i, lm in enumerate(F.lms):
            sz = F.labels[i]
            d1 = [F.REC[(pre, lm, vit)][met]["full"]["mu"] - F.REC[(pre, lm, vit)][met]["D1_max_entry_mu"] for vit in VITS]
            v[f"D1|{pre}|{met}|{sz}"] = median(d1)
            ar = [F.REC[(pre, lm, vit)][met]["full"]["aA"] / (F.REC[(pre, lm, vit)]["LA"] - 1) for vit in VITS]
            v[f"D2|{pre}|{met}|{sz}|argmax_rel"] = median(ar)
            cc = [F.REC[(pre, lm, vit)][met]["full"]["c"] - F.REC[(pre, lm, vit)][met]["full"]["c_entry"] for vit in VITS]
            v[f"D2|{pre}|{met}|{sz}|c_minus_centry"] = median(cc)
            x[f"x|D2|{pre}|{sz}|u_star_rel_partB"] = PBV["u_star_rel"].get(lm)
    # ---- D4 (Spearman on average ranks; widths, depths, num_params from Part B)
    logp = [math.log(PBV["num_params"][lm]) for lm in F.lms]
    width = [PBV["width"][lm] for lm in F.lms]
    depth = [PBV["depth"][lm] for lm in F.lms]
    qty = {"PRstar": [PBV["PR_star"][lm] for lm in F.lms],
           "Bbar": [float(np.mean([Bbar[lm][vt] for vt in VITS])) for lm in F.lms],
           "calpha": [float(np.mean([float(F.REC[(pre, lm, vt)]["cka_biased"]["full"]["c"]) for vt in VITS])) for lm in F.lms],
           "T": [float(np.mean([float(F.REC[(pre, lm, vt)]["cka_biased"]["full"]["T"]) for vt in VITS])) for lm in F.lms]}
    for qn, vals_ in qty.items():
        for an, axis in (("logparams", logp), ("width", width), ("depth", depth)):
            v[f"D4|{pre}|{qn}|{an}"] = spearman_avg(axis, vals_)
        for a, b in (("410m", "1b"), ("1b", "1.4b"), ("2.8b", "6.9b")):
            v[f"D4|{pre}|{qn}|step|{a}|{b}"] = vals_[F.labels.index(b)] - vals_[F.labels.index(a)]
    x[f"x|D4|{pre}|table"] = {"labels": F.labels, "logp": logp, "width": width, "depth": depth, **qty}
    # ---- D5
    for i in range(F.n - 1):
        la, lb = F.lms[i], F.lms[i + 1]
        gT = [F.REC[(pre, lb, vt)]["cka_biased"]["full"]["T"] - F.REC[(pre, la, vt)]["cka_biased"]["full"]["T"] for vt in VITS]
        gc = [F.REC[(pre, lb, vt)]["cka_biased"]["full"]["c"] - F.REC[(pre, la, vt)]["cka_biased"]["full"]["c"] for vt in VITS]
        k_ = f"D5|{pre}|{F.labels[i]}|{F.labels[i + 1]}"
        v[f"{k_}|raw_gain"] = median(gT)
        v[f"{k_}|thr_gain"] = median(gc)
    l0, l1 = F.lms[0], F.lms[-1]
    dbl = math.log2(PBV["num_params"][l1] / PBV["num_params"][l0])
    v[f"D5|{pre}|g"] = median([(F.REC[(pre, l1, vt)]["cka_biased"]["full"]["T"]
                                - F.REC[(pre, l0, vt)]["cka_biased"]["full"]["T"]) / dbl for vt in VITS])
    v[f"D5|{pre}|h"] = median([(F.REC[(pre, l1, vt)]["cka_biased"]["full"]["c"]
                                - F.REC[(pre, l0, vt)]["cka_biased"]["full"]["c"]) / dbl for vt in VITS])
    # ---- D6 (multiplicity, mKNN and debiased CKA)
    for met, base in (("mknn", BASE_MKNN), ("cka_debiased", 0.0)):
        cmb = []
        for i, lm in enumerate(F.lms):
            val_ = median([F.REC[(pre, lm, vt)][met]["full"]["c"] - base for vt in VITS])
            v[f"D6|{pre}|{met}|{F.labels[i]}|c_minus_base"] = val_
            cmb.append(val_)
        v[f"D6|{pre}|{met}|spearman_L"] = spearman_avg([PBV["L_stored"][lm] for lm in F.lms], cmb)
        v[f"D6|{pre}|{met}|step|410m|1b"] = cmb[F.labels.index("1b")] - cmb[F.labels.index("410m")]
    # ---- M4 inputs (section R, x-independent): primary metric, layer-max, s_cal
    n0 = 0
    for i, lm in enumerate(F.lms):
        sc = [F.REC[(pre, lm, vt)]["cka_biased"]["full"]["scal"] for vt in VITS]
        n0 += sum(1 for s_ in sc if s_ == 0)
        v[f"R|{pre}|median_scal|{F.labels[i]}"] = median(sc)
    v[f"R|{pre}|n_scal_zero"] = n0


# ----------------------------------------------------------------------------
# PYT-AXIS: section H (confirmatory family) and section R (robustness)
# ----------------------------------------------------------------------------

def tost_block(F, pre, x, met, q):  # PYT-AXIS: spec 6 TOST for one (metric, protocol)
    X = x if q == "full" else xname(x)
    d = d_stat(F, pre, met, q, X)
    out = {"D": None if d["D"] is None else float(d["D"][0]), "nV": d["nV"] if d["rx"] is not None and d["rb"]
           is not None else None, "p_lo": None, "p_hi": None, "p": None, "degenerate": d["degenerate"],
           "ranks_identical": d["ranks_identical"], "draws": None}
    if d["D"] is not None:
        dd = boot_draws(d["dr"], bkey("tost", met, q, pre, x))
        out["p_lo"], out["p_hi"], out["p"] = p_tost_tol(dd, out["D"])
        out["draws"] = dd
    return out


def sec_H_R(F, pre, x, PBV, v, xx):  # PYT-AXIS
    h = f"H|{pre}|{x}"
    s, s_late = PBV["s"], PBV["s_late"]
    member = PBV["material"]
    branch = PBV["H2_branch"]
    # ---- H1
    d = d_stat(F, pre, "cka_biased", "full", x)
    v[f"{h}|H1|member"] = member
    v[f"{h}|H1|s"] = s
    v[f"{h}|H1|nV"] = d["nV"] if d["rx"] is not None and d["rb"] is not None else None
    if d["D"] is not None:
        V = d["V"]
        R = float(d["rb"].RH[0, V].mean())
        C = float(d["rx"].RH[0, V].mean())
        D1 = s * (C - R)
        pp = p_le(s * d["D"])
        draws = boot_draws(d["dr"], bkey("H1mag", "cka_biased", "full", pre, x))
        sd = s * draws
        pm = p_mag_tol(draws, s)
        ci = [float(np.percentile(sd, 2.5)), float(np.percentile(sd, 97.5))]
        p1 = max(pp, pm)
        kedge = bool(min(abs(D1 + MAT), abs(D1 - MAT)) <= SWAP + TOL or ci[0] <= -MAT <= ci[1] or ci[0] <= MAT <= ci[1])
        v.update({f"{h}|H1|R": R, f"{h}|H1|C": C, f"{h}|H1|D": D1, f"{h}|H1|p_perm": pp, f"{h}|H1|p_mag": pm,
                  f"{h}|H1|p": p1, f"{h}|H1|ci_lo": ci[0], f"{h}|H1|ci_hi": ci[1], f"{h}|H1|kedge": kedge,
                  f"{h}|H1|confirmed": (bool(p1 <= ALPHA and D1 <= -MAT) if member else None)})
        for mg in MARGINS:
            v[f"{h}|H1|p_mag|m={mg!r}"] = p_mag_tol(draws, s, mg)
    else:
        D1 = p1 = None
        for k_ in ("R", "C", "D", "p_perm", "p_mag", "p", "ci_lo", "ci_hi", "kedge"):
            v[f"{h}|H1|{k_}"] = None
        v[f"{h}|H1|confirmed"] = None if not member else False
        for mg in MARGINS:
            v[f"{h}|H1|p_mag|m={mg!r}"] = None
    # ---- TOST (every metric, full and ctm), H3 = (mknn, full), H4 = (cka_debiased, full)
    tost = {}
    for met in METRICS:
        for q in ("full", "ctm"):
            tb = tost_block(F, pre, x, met, q)
            tost[(met, q)] = tb
            for k_ in ("D", "nV", "p_lo", "p_hi", "p", "degenerate", "ranks_identical"):
                v[f"TOST|{pre}|{x}|{met}|{q}|{k_}"] = tb[k_]
    for hy, key_ in (("H3", ("mknn", "full")), ("H4", ("cka_debiased", "full"))):
        tb = tost[key_]
        for k_ in ("D", "nV", "p_lo", "p_hi", "p", "degenerate", "ranks_identical"):
            v[f"{h}|{hy}|{k_}"] = tb[k_]
        for mg in MARGINS:
            v[f"{h}|{hy}|p_tost|m={mg!r}"] = None if tb["draws"] is None else p_tost_tol(tb["draws"], tb["D"], mg)[2]
    # ---- H2 (calibrate-then-max against the raw layer-max; both branches computed)
    d2 = d_stat(F, pre, "cka_biased", "ctm", xname(x))
    t2 = tost[("cka_biased", "ctm")]
    v[f"{h}|H2|branch"] = branch
    v[f"{h}|H2|s_late"] = s_late
    v[f"{h}|H2|nV"] = d2["nV"] if d2["rx"] is not None and d2["rb"] is not None else None
    for k_ in ("p_lo", "p_hi", "degenerate", "ranks_identical"):
        v[f"{h}|H2|{k_}"] = t2[k_]
    for mg in MARGINS:
        v[f"{h}|H2|p_tost|m={mg!r}"] = None if t2["draws"] is None else p_tost_tol(t2["draws"], t2["D"], mg)[2]
    if d2["D"] is not None:
        D2 = s_late * float(d2["D"][0])
        pp2 = p_le(s_late * d2["D"])
        dr2 = boot_draws(d2["dr"], bkey("H2mag", "cka_biased", "ctm", pre, x))
        pm2 = p_mag_tol(dr2, s_late)
        v.update({f"{h}|H2|D": D2, f"{h}|H2|p_perm": pp2, f"{h}|H2|p_mag": pm2})
        for mg in MARGINS:
            v[f"{h}|H2|p_mag|m={mg!r}"] = p_mag_tol(dr2, s_late, mg)
        v[f"{h}|H2|p"] = max(pp2, pm2) if branch == "H2a" else t2["p"]
    else:
        D2 = None
        v.update({f"{h}|H2|D": None, f"{h}|H2|p_perm": None, f"{h}|H2|p_mag": None, f"{h}|H2|p": None})
        for mg in MARGINS:
            v[f"{h}|H2|p_mag|m={mg!r}"] = None
    # ---- gate and Holm
    conf1 = v[f"{h}|H1|confirmed"]
    gate = "open (H1 confirmed)" if (member and conf1) else ("open (H1 not a member)" if not member else "closed")
    v[f"{h}|gate"] = gate
    is_open = gate.startswith("open")
    pmem = {"H2": v[f"{h}|H2|p"], "H3": v[f"{h}|H3|p"], "H4": v[f"{h}|H4|p"]}
    if is_open and all(p_ is not None for p_ in pmem.values()):
        names = ["H2", "H3", "H4"]
        order = sorted(range(3), key=lambda i: (pmem[names[i]], i))
        v[f"{h}|holm_order"] = ",".join(names[i] for i in order)
        run, stop = 0.0, False
        for k, i in enumerate(order):
            nm = names[i]
            p_ = pmem[nm]
            run = max(run, (3 - k) * p_)
            rej = (not stop) and p_ <= ALPHA / (3 - k)
            if not rej:
                stop = True
            v[f"{h}|{nm}|tested"] = True
            v[f"{h}|{nm}|p_holm"] = min(1.0, run)
            v[f"{h}|{nm}|holm_reject"] = bool(rej)
            if nm == "H2" and branch == "H2a":
                v[f"{h}|{nm}|confirmed"] = bool(rej and D2 is not None and D2 <= -MAT)
            else:
                v[f"{h}|{nm}|confirmed"] = bool(rej)
    else:
        v[f"{h}|holm_order"] = None
        for nm in ("H2", "H3", "H4"):
            v[f"{h}|{nm}|tested"] = bool(is_open)
            v[f"{h}|{nm}|p_holm"] = None
            v[f"{h}|{nm}|holm_reject"] = None
            v[f"{h}|{nm}|confirmed"] = None
    # ---- section R (plan 5.5; s = the 7-size Part B sign)
    r = f"R|{pre}|{x}"
    fr = []
    for i_drop, lab in enumerate(F.labels):
        idx = [i for i in range(F.n) if i != i_drop]
        dl = d_stat(F, pre, "cka_biased", "full", x, idx=idx)
        rk = f"{r}|loo|{lab}"
        if dl["D"] is None:
            v.update({f"{rk}|D1": None, f"{rk}|p_perm": None, f"{rk}|p_mag": None, f"{rk}|p": None})
        else:
            d1l = s * float(dl["D"][0])
            ppl = p_le(s * dl["D"])
            pml = p_mag_tol(boot_draws(dl["dr"], bkey("H1mag", "cka_biased", "full", pre, x, f"|loo|{lab}")), s)
            v.update({f"{rk}|D1": d1l, f"{rk}|p_perm": ppl, f"{rk}|p_mag": pml, f"{rk}|p": max(ppl, pml)})
            if D1 is not None and p1 is not None and ((d1l <= -MAT) != (D1 <= -MAT)
                                                      or (max(ppl, pml) <= ALPHA) != (p1 <= ALPHA)):
                fr.append(lab)
        for met, nm in (("mknn", "dtr_mknn"), ("cka_debiased", "dtr_dcka")):
            dm = d_stat(F, pre, met, "full", x, idx=idx)
            v[f"{rk}|{nm}"] = None if dm["D"] is None else float(dm["D"][0])
    v[f"{r}|fragile"] = bool(fr)
    v[f"{r}|fragile_sizes"] = ",".join(fr)
    idx5 = [F.labels.index(z) for z in FIVE]
    d5 = d_stat(F, pre, "cka_biased", "full", x, idx=idx5)
    d15 = None if d5["D"] is None else s * float(d5["D"][0])
    v[f"{r}|five|D1"] = d15
    v[f"{r}|five|p_perm"] = None if d5["D"] is None else p_le(s * d5["D"])
    if d15 is None or D1 is None:
        v[f"{r}|five|carried"] = None
        v[f"{r}|five|carried_D"] = None
    else:
        v[f"{r}|five|carried"] = bool(sgn(d15) != sgn(D1) or (d15 <= -MAT) != (D1 <= -MAT))
        v[f"{r}|five|carried_D"] = bool(sgn(d15) != sgn(D1) or (d15 >= MAT) != (D1 >= MAT))
    g = v[f"dtr|{pre}|cka_biased|grid5|{x}"]
    v[f"{r}|grid5|D1"] = None if g is None else s * g
    v[f"{r}|grid5|dtr_mknn"] = v[f"dtr|{pre}|mknn|grid5|{x}"]
    v[f"{r}|grid5|dtr_dcka"] = v[f"dtr|{pre}|cka_debiased|grid5|{x}"]
    for fam in FAMILIES4:
        mask = np.array([VIT_FAMILY4[vt] != fam for vt in VITS])
        df = d_stat(F, pre, "cka_biased", "full", x, vitmask=mask)
        v[f"{r}|lofo|{fam}|D1"] = None if df["D"] is None else s * float(df["D"][0])
        for met, nm in (("mknn", "dtr_mknn"), ("cka_debiased", "dtr_dcka")):
            dm = d_stat(F, pre, met, "full", x, vitmask=mask)
            v[f"{r}|lofo|{fam}|{nm}"] = None if dm["D"] is None else float(dm["D"][0])
    z0 = v[f"dtr0|{pre}|cka_biased|full|{x}"]
    v[f"{r}|rho0|D1"] = None if z0 is None else s * z0


# ----------------------------------------------------------------------------
# PYT-AXIS: E's own forecast and category (x| extras; the contract's F and K come from forecast_pyt.py)
# ----------------------------------------------------------------------------

def ppb_forecast(Ta, Tb, Ba, Bb, oa, ob, unc=False):  # PYT-AXIS: plan 9 B6 forecast for one comparison (a < b)
    if unc:   # oa = mu band (u_lo, u_hi) of mu - Bbar
        if Tb - Bb - oa[0] < Ta - Ba - oa[1]:
            return "below"
        if Tb - Bb - oa[1] > Ta - Ba - oa[0]:
            return "above"
        return "undecided"

    def S(T, c):
        return max((T - c) / (1 - c), 0.0) if c < 1 else 0.0
    if S(Tb, Bb + ob[0]) < S(Ta, Ba + oa[1]):
        return "below"
    if S(Tb, Bb + ob[1]) > S(Ta, Ba + oa[0]):
        return "above"
    if S(Tb, Bb + ob[0]) == 0 and S(Ta, Ba + oa[0]) == 0:
        return "tie"
    return "undecided"


def axis_category(member, d1, confirmed, p_perm_, p_mag_, p2a, p2b, R, C, tau=MAT):  # PYT-AXIS: plan 7 table
    """The axis category, steps 1-6 of plan section 7 in order. Returns (category, F sub-reason, premise)."""
    if not member:
        return "E", None, None
    if d1 is None:
        return "undefined (trend undefined)", None, None
    if d1 >= tau:
        return "D", None, None
    if abs(d1) < tau:
        return "C", None, None
    if not confirmed:
        sub = []
        if p_perm_ is not None and p_perm_ > ALPHA:
            sub.append("not consistent across the seven sizes")
        if p_mag_ is not None and p_mag_ > ALPHA:
            sub.append("size of the drop not established")
        return "F", "; ".join(sub) or None, None
    if p2a == "fails" or p2b == "fails":
        return "B", None, None
    level = ("A" if (R is not None and C is not None and R >= LOSS_R and C <= LOSS_C) else
             ("A-att" if (R is not None and R >= LOSS_R) else "A-prime"))
    prem = "verified" if (p2a == "holds" and p2b == "holds") else "unverified"
    return level, None, prem


def e_own_extras(F, pre, PBV, v, x):  # PYT-AXIS: E's independent P-P2b and category (x| only; cross-check of F/K)
    cmp_ = comparisons(F.n)
    for X in ("s_cal", "unc"):
        dec = nw = und = 0
        for vit in VITS:
            for a, b in cmp_:
                la, lb = F.lms[a], F.lms[b]
                Ra, Rb = F.REC[(pre, la, vit)]["cka_biased"]["full"], F.REC[(pre, lb, vit)]["cka_biased"]["full"]
                oa = PBV["band"][la] if X == "s_cal" else PBV["mu_band"]
                ob = PBV["band"][lb] if X == "s_cal" else PBV["mu_band"]
                f_ = ppb_forecast(float(Ra["T"]), float(Rb["T"]), PBV["Bbar"][la][vit], PBV["Bbar"][lb][vit], oa, ob,
                                  unc=(X == "unc"))
                kx = "scal" if X == "s_cal" else "unc"
                dd = sgn(Rb[kx] - Ra[kx])
                obs = "below" if dd < 0 else ("above" if dd > 0 else "tie")
                if f_ == "undecided":
                    und += 1
                else:
                    dec += 1
                    nw += int(f_ != obs)
        verd = "fails" if nw > 0 else ("not evaluable" if dec < 30 else "holds")
        x[f"x|E_own|PP2b|{pre}|{X}"] = {"n_decided": dec, "n_wrong": nw, "n_undecided": und, "verdict": verd}
        h = f"H|{pre}|{X}"
        cat = axis_category(v[f"{h}|H1|member"], v[f"{h}|H1|D"], v[f"{h}|H1|confirmed"], v[f"{h}|H1|p_perm"],
                            v[f"{h}|H1|p_mag"], v[f"PP2a|{pre}|{X}|verdict"], verd, v[f"{h}|H1|R"], v[f"{h}|H1|C"])
        x[f"x|E_own|category|{pre}|{X}"] = {"category": cat[0], "F_reason": cat[1], "premise": cat[2],
                                           "label": "E's own reading of plan section 7 (the contract's K is written by "
                                                    "forecast_pyt.py; this extra is a cross-check only)"}


# ----------------------------------------------------------------------------
# PYT-AXIS: section D (cross-family table, 18-LM panel), descriptive diagnostics
# ----------------------------------------------------------------------------

def pear(G, x):
    xx = np.repeat(x[:, None], G.shape[1], axis=1).ravel()
    y = G.ravel()
    if np.nanstd(y) == 0 or not np.all(np.isfinite(y)):
        return float("nan")
    return float(np.corrcoef(xx, y)[0, 1])


def num_params_known(stem):  # PYT-AXIS
    for base in (os.path.join(LC.FEAT_W1024, stem), os.path.join(LC.PROJ, "features", "wit1024_llama_axis", stem),
                 os.path.join(LC.PROJ, "features", "wit1024_openllama_axis", stem)):
        if os.path.exists(base + ".json"):
            return int(json.load(open(base + ".json"))["num_params"])
    return LC.n_params(stem)


def known_base(pre, lm, vit):  # PYT-AXIS: each implementation reads its own counterpart (plan 3); E: the E trees
    if lm in LLAMA1[1:]:
        return os.path.join(LC.LLAMA_E_RES, "pairs", pre, f"{lm}__{vit}")
    if lm == "openlm-research__open_llama_13b":
        return os.path.join(LC.OL_E_RES, "pairs", pre, f"{lm}__{vit}")
    return os.path.join(LC.FROZEN_RES, "pairs", pre, f"{lm}__{vit}")


def sec_D(F, POOL, NPAR, pre, v):  # PYT-AXIS
    from hrhv2 import util
    fams = {"BLOOMZ": BLOOMZ, "LLaMA": LLAMA1, "OpenLLaMA": OPENLLAMA}
    for met in METRICS:
        for fam, lms in fams.items():
            Fk = Family(POOL, lms, [str(i) for i in range(len(lms))])
            for X in ("raw", "s_cal", "unc"):
                rt = Fk.rt(pre, met, "full", X)
                T = None if rt is None else tr_series(rt)
                v[f"famtr|{pre}|{met}|{X}|{fam}"] = None if T is None else float(T[0])
            for X in ("s_cal", "unc"):
                d = d_stat(Fk, pre, met, "full", X)
                v[f"famdtr|{pre}|{met}|{X}|{fam}"] = None if d["D"] is None else float(d["D"][0])
        for X in ("raw", "s_cal", "unc"):
            v[f"famtr|{pre}|{met}|{X}|Pythia"] = v[f"tr|{pre}|{met}|full|{X}"]
        for X in ("s_cal", "unc"):
            v[f"famdtr|{pre}|{met}|{X}|Pythia"] = v[f"dtr|{pre}|{met}|full|{X}"]
        lms18 = BLOOMZ + LLAMA1 + OPENLLAMA + F.lms
        xv = np.log(np.array([NPAR[s_] for s_ in lms18], dtype=float))
        for X in ("raw", "s_cal", "unc"):
            G = np.array([[float(val(POOL[(pre, lm, vt)] if lm not in F.lms else F.REC[(pre, lm, vt)], met, "full", X))
                           for vt in VITS] for lm in lms18])
            rng = util.rng_from_key(f"S1.2|bootvit|layermax-pythia|panel18|{pre}|{met}|{X}")
            bs = np.empty(NB)
            for b in range(NB):
                j = rng.integers(0, NV, NV)
                bs[b] = pear(G[:, j], xv)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            v[f"panel18|{pre}|{met}|{X}|r"] = pear(G, xv)
            v[f"panel18|{pre}|{met}|{X}|ci_lo"] = float(lo)
            v[f"panel18|{pre}|{met}|{X}|ci_hi"] = float(hi)


def extras(F, pre, PBV, x):  # PYT-AXIS: OP section 6 extras and D7 (descriptive, x| only)
    for met in METRICS:
        for col in ("c", "mu", "shift", "sel_entry_thr", "sel_entry_mean", "inflation", "c_entry"):
            T_ = tr_series(rho_table(rows_of(F.REC, F.lms, pre, met, "full", col)))
            x[f"x|decomp|{pre}|{met}|{col}"] = None if T_ is None else float(T_[0])
        for q in ("full", "fixed_r0.5", "fixed_r1.0"):
            for X in ("raw", "s_cal", "unc"):
                G = np.array([[fnum(val(F.REC[(pre, lm, vt)], met, q, X)) for vt in VITS] for lm in F.lms], dtype=float)
                if np.all(np.isfinite(G)):
                    vt_ = A12.vision_trend(G)
                    x[f"x|vision|{pre}|{met}|{q}|{X}"] = {"Tr": vt_["Tr"], "p_one": vt_["p_one"], "n_cells": vt_["n_cells"]}
        for i, lm in enumerate(F.lms):
            prof = np.mean(np.stack([np.asarray(F.REC[(pre, lm, vt)][met]["null_profile"], dtype=float) for vt in VITS]),
                           axis=0)
            x[f"x|nullprof|{pre}|{met}|{F.labels[i]}|argmax_rel"] = float(np.argmax(prof) / (prof.size - 1))
            for col in ("raw", "c", "s_cal", "unc", "mu"):
                x[f"x|mean|{pre}|{met}|full|{col}|{F.labels[i]}"] = float(np.mean(
                    [float(val(F.REC[(pre, lm, vt)], met, "full", col)) for vt in VITS]))
    for i, lm in enumerate(F.lms):
        x[f"x|mean|{pre}|cka_biased|full|Bbar|{F.labels[i]}"] = float(np.mean([PBV["Bbar"][lm][vt] for vt in VITS]))
    for q in SUBSETS:
        if q.startswith("grid"):
            n_ = int(q[4:])
            x[f"x|grid_saturated|{pre}|{q}"] = {F.labels[i]: LC.grid_saturated(F.REC[(pre, lm, VITS[0])]["LA"], n_)
                                                for i, lm in enumerate(F.lms)}
    sink = {}
    for lab in F.labels:   # D7 (descriptive): extraction sink statistics, if present
        p_ = os.path.join(LC.PYT_EXTRACTION, f"sink_{lab}.json")
        if os.path.exists(p_):
            try:
                sink[lab] = json.load(open(p_))
                READ_FILES[p_] = "extraction sink statistics (D7)"
            except Exception as e:  # noqa: BLE001
                sink[lab] = {"error": repr(e)}
    x[f"x|D7|{pre}|sink_files"] = sink if sink else "absent"
    tok1 = PBV.get("PR_star_tok1")
    x[f"x|D7|{pre}|PR_star_tok1"] = tok1 if tok1 else "absent from Part B"


# ----------------------------------------------------------------------------
# PYT-AXIS: the whole Pythia computation
# ----------------------------------------------------------------------------

def compute(REC, POOL, PB, NPAR, lms=None, labels=None, pres=PRES, boot_ci=True, with_panels=True):  # PYT-AXIS
    """REC[(pre, lm, vit)] for the 7 sizes (default: the Pythia slugs of lm_core); POOL for the 11 known LMs (section
    D). Returns (v, x): every required key of sections A, B, C, H, R, D for the pre run, plus x| extras."""
    ensure_a12()
    lms = list(lms or PYT)
    labels = list(labels or SIZES)
    F = Family(REC, lms, labels)
    v, x = {}, {}
    t0 = time.time()
    for pre in pres:
        PBV = pb_view(PB, pre, lms, labels)
        PBV["PR_star_tok1"] = PB["per_pre"][pre].get("PR_star_tok1")
        sec_A(F, pre, v, x, boot_ci=boot_ci)
        sec_B(F, pre, v)
        sec_C(F, pre, PBV, v, x)
        for xs in ("s_cal", "unc"):
            sec_H_R(F, pre, xs, PBV, v, x)
        e_own_extras(F, pre, PBV, v, x)
        extras(F, pre, PBV, x)
        if with_panels:
            sec_D(F, POOL, NPAR, pre, v)
        Gb = np.array([[PBV["Bbar"][lm][vt] for vt in VITS] for lm in lms])
        Gl = np.array([[PBV["Blate"][lm][vt] for vt in VITS] for lm in lms])
        mb, ml = material_from(Gb), material_from(Gl)
        x[f"x|check|{pre}|partB_decisions_recomputed"] = {
            "material_Bbar": [mb["material"], PBV["material"]], "s": [mb["s"], PBV["s"]],
            "material_Blate": [ml["material"], PBV["material_late"]], "s_late": [ml["s"], PBV["s_late"]],
            "H2_branch": ["H2a" if ml["material"] else "H2b", PBV["H2_branch"]],
            "band": {labels[i]: [band_i(pre, PBV["PR_star"][lm]), list(PBV["band"][lm])] for i, lm in enumerate(lms)},
            "all_equal": bool(mb["material"] == PBV["material"] and mb["s"] == PBV["s"]
                              and ml["material"] == PBV["material_late"] and ml["s"] == PBV["s_late"]
                              and ("H2a" if ml["material"] else "H2b") == PBV["H2_branch"]
                              and all(abs(band_i(pre, PBV["PR_star"][lm])[k] - PBV["band"][lm][k]) <= 1e-12
                                      for lm in lms for k in (0, 1)))}
    x["x|timing|compute_s"] = time.time() - t0
    return v, x


def band_i(pre, prstar):  # PYT-AXIS: plan 9 B6 band of one model (x|check only; the decision uses Part B's band)
    a, b, h, _ = BAND[pre]
    f = a + b / math.sqrt(prstar)
    return f - h, f + h


# ----------------------------------------------------------------------------
# PYT-AXIS: legacy mode for V-P8(a)/(b): the OpenLLaMA section A-C verdict code on the generalised engine (n = 3)
# ----------------------------------------------------------------------------

OL_SIZES = ("3b", "7b", "13b")


def ol_legacy_compute(REC, POOL11, PB, common=False):  # PYT-AXIS
    """The OpenLLaMA aggregate_ol.compute() sections A, B, C (keys of the OpenLLaMA VERDICTS_SPEC), with the
    3-size trend() replaced by the generalised engine (rho_table / tr_series / diff_series over 3! relabellings).
    legacy (common=False): per-score exclusion, as the OpenLLaMA code; common=True: the common-ViT-set rule of the
    Pythia plan (V-P8(b)). POOL11 gives the 11-LM records needed by PO4|arh (r only; no 11! enumeration)."""
    ensure_a12()
    OL = OPENLLAMA
    F = Family(REC, OL, OL_SIZES)
    v, x = {}, {}
    TAUS_ = TAUS
    TR = {}
    for pre in PRES:
        for met in METRICS:
            for q in SUBSETS + ("ctm",):
                for X in scores_of(q):
                    rt = F.rt(pre, met, q, X)
                    T6 = None if rt is None else tr_series(rt)
                    TR[(pre, met, q, X)] = rt
                    key = f"{pre}|{met}|{q}|{X}"
                    v[f"tr|{key}"] = None if T6 is None else float(T6[0])
                    v[f"tr_nconst|{key}"] = None if rt is None else rt.nexcl
                    if T6 is None:
                        v.update({f"tr_pge|{key}": None, f"tr_ple|{key}": None, f"tr_ptwo|{key}": None})
                    else:
                        for k_, val_ in pvals(T6, "tr").items():
                            v[f"{k_}|{key}"] = val_
                    if met != "mknn":
                        v[f"neartie|{key}"] = None if rt is None else neartie(rows_of(REC, OL, pre, met, q, X))
                for X in dscores_of(q):
                    key = f"{pre}|{met}|{q}|{X}"
                    rb = TR[(pre, met, "full", "raw")] if q == "ctm" else TR[(pre, met, q, "raw")]
                    rx = TR[(pre, met, q, X)]
                    if rx is None or rb is None:
                        v.update({f"dtr|{key}": None, f"dtr_pge|{key}": None, f"dtr_ple|{key}": None,
                                  f"dtr_ptwo|{key}": None})
                        continue
                    D, _ = diff_series(rx, rb, legacy=not common)
                    if D is None:
                        v.update({f"dtr|{key}": None, f"dtr_pge|{key}": None, f"dtr_ple|{key}": None,
                                  f"dtr_ptwo|{key}": None})
                        continue
                    v[f"dtr|{key}"] = float(D[0])
                    for k_, val_ in pvals(D, "dtr").items():
                        v[f"{k_}|{key}"] = val_
            v[f"d3cal|{pre}|{met}"] = _sub(v[f"tr|{pre}|{met}|ctm|M"], v[f"tr|{pre}|{met}|ctm|TM"])
            v[f"d3sel|{pre}|{met}"] = _sub(v[f"tr|{pre}|{met}|ctm|TM"], v[f"tr|{pre}|{met}|full|raw"])
    # ---------------- section B (OpenLLaMA: new = pairs with 13b, pilot = 3b-7b)
    for pre in PRES:
        P = PB["per_pre"][pre]
        s_O = int(P["s_O"])
        for met in METRICS:
            for q in SUBSETS + ("ctm",):
                for X in (("M", "M_unc") if q == "ctm" else ("s_cal", "unc")):
                    cnt = {g: {"down": 0, "up": 0, "c2d": 0, "d2c": 0} for g in ("new", "pilot")}
                    for vit in VITS:
                        for a, b, g in ((0, 2, "new"), (1, 2, "new"), (0, 1, "pilot")):
                            Ra, Rb = REC[(pre, OL[a], vit)], REC[(pre, OL[b], vit)]
                            qr = "full" if q == "ctm" else q
                            ra, rb = val(Ra, met, qr, "raw"), val(Rb, met, qr, "raw")
                            xa, xb = val(Ra, met, q, X), val(Rb, met, q, X)
                            if None in (ra, rb, xa, xb):
                                continue
                            dr, dx = sgn(rb - ra), sgn(xb - xa)
                            cnt[g]["down"] += int(dx < dr)
                            cnt[g]["up"] += int(dx > dr)
                            cnt[g]["c2d"] += int(dr == 1 and dx == -1)
                            cnt[g]["d2c"] += int(dr == -1 and dx == 1)
                    key = f"{pre}|{met}|{q}|{X}"
                    for g in ("new", "pilot"):
                        for c_ in ("down", "up", "c2d", "d2c"):
                            v[f"chg|{key}|{g}|{c_}"] = cnt[g][c_]
                    fm = cnt["new"]["down"] if s_O == 1 else cnt["new"]["up"]
                    fp = cnt["new"]["up"] if s_O == 1 else cnt["new"]["down"]
                    lo, hi = cp95(fm, 2 * NV)
                    v.update({f"fnew|{key}|F": (fm - fp) / (2 * NV), f"fnew|{key}|fminus": fm, f"fnew|{key}|fplus": fp,
                              f"fnew|{key}|cp_lo": lo, f"fnew|{key}|cp_hi": hi})
    # ---------------- 11-LM r for PO4|arh (section D values used by section C)
    def grid11(pre, met, X):
        return np.array([[float(val(REC[(pre, lm, vt)] if lm in OL else POOL11[(pre, lm, vt)], met, "full", X))
                          for vt in VITS] for lm in KNOWN11])
    x11 = np.arange(1, 12, dtype=float)
    # ---------------- section C (OpenLLaMA rules, verbatim logic of aggregate_ol.compute)
    BAND_OL = (0.0009, 0.0029)
    S13, S3, S7 = OL[2], OL[0], OL[1]
    for pre in PRES:
        P = PB["per_pre"][pre]
        s_O, s_late = int(P["s_O"]), int(P["s_late"])
        material, mat_late = bool(P["material"]), bool(P["material_Blate"])
        Bbar = {lm: {vit: float(pb_lm(P["Bbar"], lm)[vit]) for vit in VITS} for lm in OL}
        Blate = {lm: {vit: float(pb_lm(P["Blate"], lm)[vit]) for vit in VITS} for lm in OL}
        tr = lambda met, q, X: v[f"tr|{pre}|{met}|{q}|{X}"]  # noqa: E731
        dtr = lambda met, q, X: v[f"dtr|{pre}|{met}|{q}|{X}"]  # noqa: E731
        d1s, d1u = dtr("mknn", "full", "s_cal"), dtr("mknn", "full", "unc")
        v[f"PO1|{pre}|holds"] = bool(d1s is not None and d1u is not None and abs(d1s) < MAT and abs(d1u) < MAT)
        for X, thr in (("s_cal", "c"), ("unc", "mu")):
            nq = nrev = ntie = 0
            for vit in VITS:
                for a, b in ((0, 1), (1, 2), (0, 2)):
                    Ra, Rb = REC[(pre, OL[a], vit)]["mknn"]["full"], REC[(pre, OL[b], vit)]["mknn"]["full"]
                    za, zb = abs(Ra[thr] - BASE_MKNN), abs(Rb[thr] - BASE_MKNN)
                    gap = Rb["T"] - Ra["T"]
                    if abs(gap) > za + zb:
                        nq += 1
                        key_ = "scal" if X == "s_cal" else "unc"
                        dx = sgn(Rb[key_] - Ra[key_])
                        nrev += int(dx == -sgn(gap) and dx != 0)
                        ntie += int(dx == 0)
            v.update({f"PO1|{pre}|zeta|{X}|nq": nq, f"PO1|{pre}|zeta|{X}|nrev": nrev, f"PO1|{pre}|zeta|{X}|ntie": ntie})

        def cat_po2(material_, tr_raw, d, s, tau=MAT):
            if not material_:
                return "no category (not material)"
            if tr_raw is not None and abs(tr_raw + s) <= TOL:
                return "not testable"
            if d is None:
                return "undefined (trend undefined)"
            if s * d <= -tau:
                return "loss reproduced"
            if s * d >= tau:
                return "moved with the baseline"
            return "trend kept"

        def cat_po5(material_late, po5a_verdict, tr_raw, d, s, tau=MAT):
            if not material_late:
                return "no category (not material)"
            if po5a_verdict == "fails":
                return "premise failed"
            return cat_po2(True, tr_raw, d, s, tau)

        def verdict_2n3(n, xx):
            need = int(math.ceil(2 * n / 3)) if n else 0
            if n < 6:
                return need, "not evaluable"
            return need, "holds" if xx >= need else "fails"
        trraw = tr("cka_biased", "full", "raw")
        for X in ("s_cal", "unc"):
            d = dtr("cka_biased", "full", X)
            v[f"PO2|{pre}|{X}|cat"] = cat_po2(material, trraw, d, s_O)
            dist = None if d is None else min(abs(s_O * d + MAT), abs(s_O * d - MAT))
            v[f"PO2|{pre}|{X}|dist"] = dist
            v[f"PO2|{pre}|{X}|kedge"] = None if dist is None else bool(dist <= 1.0 / NV + TOL)
            for tau in TAUS_:
                v[f"curve|{pre}|PO2|{X}|tau={tau!r}"] = cat_po2(material, trraw, d, s_O, tau)
        for tau in TAUS_:
            v[f"curve|{pre}|PO1|tau={tau!r}"] = bool(d1s is not None and d1u is not None and abs(d1s) < tau
                                                     and abs(d1u) < tau)
        offs = {vit: REC[(pre, S13, vit)]["cka_biased"]["full"]["c"] - Bbar[S13][vit] for vit in VITS}
        med = median(offs.values())
        xin = sum(1 for o in offs.values() if BAND_OL[0] <= o <= BAND_OL[1])
        lo, hi = cp95(xin, NV)
        lpm = median(abs(o) for o in offs.values())
        v.update({f"PO2ai|{pre}|median": med, f"PO2ai|{pre}|in_band": bool(BAND_OL[0] <= med <= BAND_OL[1]),
                  f"PO2ai|{pre}|n": NV, f"PO2ai|{pre}|x": xin, f"PO2ai|{pre}|cp_lo": lo, f"PO2ai|{pre}|cp_hi": hi,
                  f"PO2ai|{pre}|lp_median_abs": lpm, f"PO2ai|{pre}|lp_le_0.01": bool(lpm <= 0.01)})
        for vit in VITS:
            v[f"PO2ai|{pre}|offset|{vit}"] = offs[vit]
        for db in (DELTA_B,) + DBS:
            n = xx = 0
            for vit in VITS:
                for a in (S3, S7):
                    gB = Bbar[S13][vit] - Bbar[a][vit]
                    if abs(gB) >= db:
                        n += 1
                        gc = REC[(pre, S13, vit)]["cka_biased"]["full"]["c"] - REC[(pre, a, vit)]["cka_biased"]["full"]["c"]
                        xx += int(sgn(gc) == sgn(gB))
            need, verd = verdict_2n3(n, xx)
            if db == DELTA_B and f"PO2aii|{pre}|n" not in v:
                lo, hi = cp95(xx, n)
                v.update({f"PO2aii|{pre}|n": n, f"PO2aii|{pre}|x": xx, f"PO2aii|{pre}|need": need,
                          f"PO2aii|{pre}|verdict": verd, f"PO2aii|{pre}|cp_lo": lo, f"PO2aii|{pre}|cp_hi": hi})
            if db in DBS:
                v.update({f"curve|{pre}|PO2aii|db={db!r}|n": n, f"curve|{pre}|PO2aii|db={db!r}|x": xx,
                          f"curve|{pre}|PO2aii|db={db!r}|verdict": verd})
        trp = float(P["Tr_pred"])
        G3 = np.array([[Bbar[lm][vit] for vit in VITS] for lm in OL])
        inc = bool(np.all(np.diff(G3, axis=0) > 0))
        dec = bool(np.all(np.diff(G3, axis=0) < 0))
        meds = [float(np.median(np.abs(G3[b] - G3[a]))) for a, b in ((0, 1), (1, 2), (0, 2))]
        for db in DBS:
            v[f"curve|{pre}|material|db={db!r}"] = bool(abs(abs(trp) - 1.0) <= TOL and (inc or dec)
                                                       and all(m_ >= db for m_ in meds))
        for r in ("0.5", "1.0"):
            q = f"fixed_r{r}"
            t_raw = tr("cka_biased", q, "raw")
            for X in ("s_cal", "unc"):
                t_x = tr("cka_biased", q, X)
                if t_raw is None or abs(t_raw) < MAT:
                    dsc = "not applied"
                else:
                    s_r = 1 if t_raw > 0 else -1
                    if t_x is None:
                        dsc = "undefined (trend undefined)"
                    elif s_r * t_x >= MAT:
                        dsc = "keeps sign"
                    elif abs(t_x) < MAT:
                        dsc = "lost"
                    else:
                        dsc = "reversed"
                v[f"PO3|{pre}|{r}|{X}|desc"] = dsc
        Fo = Family(REC, OL, OL_SIZES)
        vk = {}
        keep_sign_checks(Fo, pre, vk)   # PYT-AXIS: the unchanged keep identity and per-comparison sign rule
        rhoM, rhoTM = Fo.rt(pre, "cka_biased", "ctm", "M"), Fo.rt(pre, "cka_biased", "ctm", "TM")
        nv3 = 0   # the OpenLLaMA D3 check, verbatim (no censoring-tie exclusion on the OpenLLaMA axis)
        for j, vit in enumerate(VITS):
            cm = [REC[(pre, lm, vit)]["cka_biased"]["ctm"]["cM"] for lm in OL]
            if cm[0] < cm[1] < cm[2]:
                rm_ = rhoM.RH[0, j] if rhoM.valid[j] else 0.0
                rt_ = rhoTM.RH[0, j] if rhoTM.valid[j] else 0.0
                nv3 += int(rm_ - rt_ > 0)
        vk[f"D3|{pre}|nviol"] = nv3
        for met in METRICS:
            for q in SUBSETS:
                for X in ("s_cal", "unc"):
                    v[f"keepid|{pre}|{met}|{q}|{X}|nviol"] = vk[f"keepid|{pre}|{met}|{q}|{X}|nviol"]
                    v[f"signrule|{pre}|{met}|{q}|{X}|nviol"] = vk[f"signrule|{pre}|{met}|{q}|{X}|nviol"]
        d4s, d4u = dtr("cka_debiased", "full", "s_cal"), dtr("cka_debiased", "full", "unc")
        v[f"PO4|{pre}|s_cal|holds"] = bool(d4s is not None and d4s >= -MAT)
        v[f"PO4|{pre}|unc|holds"] = bool(d4u is not None and d4u >= -MAT)
        ra_, rb_ = pear(grid11(pre, "cka_debiased", "s_cal"), x11), pear(grid11(pre, "cka_debiased", "raw"), x11)
        v[f"PO4|{pre}|arh|holds"] = bool(isfin(ra_) and isfin(rb_) and ra_ >= rb_ - 0.05)
        n = xx = 0
        for vit in VITS:
            for a in (S3, S7):
                gB = Blate[S13][vit] - Blate[a][vit]
                if abs(gB) >= DELTA_B:
                    n += 1
                    gc = REC[(pre, S13, vit)]["cka_biased"]["ctm"]["cM"] - REC[(pre, a, vit)]["cka_biased"]["ctm"]["cM"]
                    xx += int(sgn(gc) == sgn(gB))
        need, po5a = verdict_2n3(n, xx)
        v.update({f"PO5a|{pre}|n": n, f"PO5a|{pre}|x": xx, f"PO5a|{pre}|need": need, f"PO5a|{pre}|verdict": po5a})
        for X in ("M", "M_unc"):
            d = dtr("cka_biased", "ctm", X)
            v[f"PO5|{pre}|{X}|pred_holds"] = None if not mat_late else (None if d is None else bool(s_late * d <= 0))
            v[f"PO5|{pre}|{X}|cat"] = cat_po5(mat_late, po5a, trraw, d, s_late)
            for tau in TAUS_:
                v[f"curve|{pre}|PO5|{X}|tau={tau!r}"] = cat_po5(mat_late, po5a, trraw, d, s_late, tau)
        v[f"D3|{pre}|nviol"] = vk[f"D3|{pre}|nviol"]
        tp = [tuple(REC[(pre, lm, vit)]["cka_biased"]["full"]["T"] - Bbar[lm][vit] for lm in OL) for vit in VITS]
        T6 = tr_series(rho_table(tp))
        pt = None if T6 is None else float(T6[0])
        tu = tr("cka_biased", "full", "unc")
        v[f"plugin|{pre}|tr"] = pt
        v[f"plugin|{pre}|diff"] = _sub(pt, tu)
        v[f"plugin|{pre}|within"] = bool(pt is not None and tu is not None and abs(pt - tu) <= MAT)
        for met in CKAS:
            t_raw, t_unc, t_m = tr(met, "full", "raw"), tr(met, "full", "unc"), tr("mknn", "full", "unc")
            if t_raw is None or t_raw < 0.60:
                rd = "not applied"
            elif t_unc is not None and t_m is not None and t_unc <= 0.40 and t_m >= 0.60:
                rd = "loses"
            elif t_unc is not None and t_unc >= 0.60:
                rd = "no loss"
            else:
                rd = "intermediate"
            v[f"readO|{pre}|{met}"] = rd
        for met in METRICS:
            for lm, sz in zip(OL, OL_SIZES):
                d1 = [REC[(pre, lm, vit)][met]["full"]["mu"] - REC[(pre, lm, vit)][met]["D1_max_entry_mu"] for vit in VITS]
                v[f"D1|{pre}|{met}|{sz}"] = median(d1)
                ar = [REC[(pre, lm, vit)][met]["full"]["aA"] / (REC[(pre, lm, vit)]["LA"] - 1) for vit in VITS]
                v[f"D2|{pre}|{met}|{sz}|argmax_rel"] = median(ar)
                cc = [REC[(pre, lm, vit)][met]["full"]["c"] - REC[(pre, lm, vit)][met]["full"]["c_entry"] for vit in VITS]
                v[f"D2|{pre}|{met}|{sz}|c_minus_centry"] = median(cc)
            g = [int(REC[(pre, OL[1], vit)][met]["full"]["T"] > REC[(pre, OL[0], vit)][met]["full"]["T"]) for vit in VITS]
            v[f"frac7gt3|{pre}|{met}"] = sum(g) / NV
    return v, x


# ----------------------------------------------------------------------------
# section G, checks and main
# ----------------------------------------------------------------------------

def selfchecks(REC):  # PYT-AXIS: the family-independent self-checks of the OpenLLaMA copy
    out = {}
    nd = 0
    nall = 0
    for R in REC.values():
        for q, s in R["mknn_json"].items():
            if q not in R["mknn"]:
                continue
            e = R["mknn"][q]
            for jk, ek in (("s_obs", "T"), ("c_alpha", "c"), ("s_cal", "scal"), ("uncensored", "unc"),
                           ("null_mean", "mu"), ("p_value", "p")):
                nall += 1
                nd += int(float(s[jk]) != float(e[ek]))
            nall += 1
            nd += int(s["argmax_A"] != e["aA"] or s["argmax_B"] != e["aB"])
    out["mknn_json_float_vs_exact"] = {"n_fields": nall, "n_not_equal_to_float_of_exact": nd}
    mx = 0.0
    nunc = 0
    for R in REC.values():
        for met in CKAS:
            mx = max(mx, R[met].get("ctm_json_vs_npz_maxabs", 0.0))
            nunc += int(not R[met].get("ctm_argmax_unc_json_equals_npz", True))
    out["ctm_cka_json_vs_npz_numpy_maxabs"] = mx
    out["ctm_M_unc_argmax_json_vs_npz_n_differ"] = nunc
    out["perm_keys_ok"] = all(R["perm_key"] == LC.perm_key(R["lm"], R["vit"]) for R in REC.values())
    out["K_values"] = sorted({R["K"] for R in REC.values()})
    out["lm_fields_ok"] = all(R["lm"] == lm and R["vit"] == vit for (pre, lm, vit), R in REC.items())
    return out


def precat_checks(v, pres):  # PYT-AXIS: plan 7 "checks before any category is read", E side (x| extra)
    vd = os.path.join(LC.RES, "validation")
    out = {}
    for k_, f_ in (("VP1_VP2", "VP1_VP2_knn_tables.json"), ("VP3", "VP3.json"), ("VP4", "VP4_known_pairs.json")):
        p_ = os.path.join(vd, f_)
        if not os.path.exists(p_):
            out[k_] = None
            continue
        d = json.load(open(p_))
        READ_FILES[p_] = "E validation"
        if k_ == "VP1_VP2":
            out["VP1"], out["VP2"] = bool(d["summary"]["VP1_pass"]), bool(d["summary"]["VP2_pass"])
        elif k_ == "VP3":
            out["VP3"] = bool(d["summary"]["VP3_pass"])
        else:
            out["VP4"] = bool(d["pass"])
    p7 = [os.path.join(vd, "VP7_prop5.json")] + [os.path.join(vd, f"VP7_prop5.{p}.json") for p in pres]
    p7 = [p for p in p7 if os.path.exists(p)]
    out["VP7"] = bool(json.load(open(p7[0]))["pass"]) if p7 else None
    for pre in pres:
        out[f"PP1|{pre}"] = v[f"PP1|{pre}|holds"]
        out[f"keepid|{pre}"] = sum(v[k] for k in v if k.startswith(f"keepid|{pre}|")) == 0
        out[f"signrule|{pre}"] = sum(v[k] for k in v if k.startswith(f"signrule|{pre}|")) == 0
        out[f"signvit|{pre}"] = sum(v[k] for k in v if k.startswith(f"signvit|{pre}|")
                                    and (k.endswith("|nviol_up") or k.endswith("|nviol_down"))) == 0
        out[f"D3|{pre}"] = v[f"D3|{pre}|nviol"] == 0
    out["VP5"] = "pending (runs after both aggregates; counts and key names only)"
    return out


def clip_equal_partB(pres_run):  # PYT-AXIS: spec 9 G|clip_equal_partB (V-P1 restated on the thresholds used)
    p_e = os.path.join(LC.CACHE, "clip.json")
    if not (os.path.exists(p_e) and os.path.exists(LC.PARTB_CLIP)):
        return None
    mine = json.load(open(p_e))
    ref = json.load(open(LC.PARTB_CLIP))["clip"]
    READ_FILES[p_e] = "E clip cache"
    READ_FILES[LC.PARTB_CLIP] = "Part B clip thresholds"
    try:
        return bool(all(float(mine[s]["c"]) == float(ref[s]["c"]) for s in PYT))
    except KeyError:
        return False


def digest(paths):
    d = hashlib.sha256()
    for p in sorted(paths):
        d.update(os.path.relpath(p, LC.PROJ).encode())
        d.update(sha(p).encode())
    return d.hexdigest()


def load_family(pres, lms=None, base_fn=None):  # PYT-AXIS
    lms = lms or PYT
    base_fn = base_fn or (lambda pre, lm, vit: os.path.join(LC.pairs_dir(pre, lm), f"{lm}__{vit}"))
    return {(pre, lm, vit): load_record(base_fn(pre, lm, vit)) for pre in pres for lm in lms for vit in VITS}


def load_known(pres, lms=None):  # PYT-AXIS
    return {(pre, lm, vit): load_record(known_base(pre, lm, vit), full_only=True)
            for pre in pres for lm in (lms or KNOWN11) for vit in VITS}


def shared_verdict_keys():  # PYT-AXIS: the shared key list (names only), loaded by path
    path = os.path.join(LC.AXIS, "verdict_keys.py")
    spec = importlib.util.spec_from_file_location("pyt_verdict_keys", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha(path)


def write_json(name, obj):  # PYT-AXIS: an existing file is first copied aside (no-delete rule), then replaced
    os.makedirs(os.path.dirname(name), exist_ok=True)
    if os.path.exists(name):
        import shutil
        shutil.copy2(name, f"{name}.prev-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    with open(name + ".tmp", "w") as f:
        json.dump(LC.jsonable(obj), f, indent=1)
    os.replace(name + ".tmp", name)


def main_legacy(common):  # PYT-AXIS: spec 0 "Legacy mode (V-P8 only)", S0b scratch root only
    if not LC.TEST:
        LC._refuse("--legacy runs only in the S0b test mode (outputs only under the S0b scratch root, spec 0)")
    OL = OPENLLAMA
    REC = {(pre, lm, vit): load_record(os.path.join(LC.OL_E_RES, "pairs", pre, f"{lm}__{vit}"))
           for pre in PRES for lm in OL for vit in VITS}
    POOL = load_known(PRES, [s for s in KNOWN11 if s not in OL])
    PB = json.load(open(LC.OL_PARTB_JSON))
    v, _ = ol_legacy_compute(REC, POOL, PB, common=common)
    vals = {k: fnum(val_) if not isinstance(val_, (str, list, dict)) and val_ is not None else val_ for k, val_ in v.items()}
    name = os.path.join(LC.RES, "legacy", "verdicts_ol_legacy_common.json" if common else "verdicts_ol_legacy.json")
    write_json(name, {"schema": "OL-VERDICTS v1", "impl": "E-legacy" + ("-common" if common else ""),
                      "inputs": {LC.OL_PARTB_JSON: sha(LC.OL_PARTB_JSON)}, "v": vals})
    print(f"written {name}; {len(vals)} keys", flush=True)


def main():
    global PRES_RUN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", nargs="*", default=list(PRES), choices=list(PRES))  # PYT-AXIS
    ap.add_argument("--legacy", action="store_true", help="V-P8 legacy mode (n = 3, OpenLLaMA keys; S0b only)")
    ap.add_argument("--legacy-common", action="store_true", help="V-P8(b): legacy keys with the common-ViT-set rule")
    ap.add_argument("--no-panels", action="store_true", help="tests only: skip section D (its keys are then null)")
    args = ap.parse_args()
    LC.gate("early/aggregate_pyt.py")   # PYT-AXIS: plan F1(a), first action
    ensure_a12()
    if args.legacy or args.legacy_common:
        main_legacy(common=args.legacy_common)
        return
    t_all = time.time()
    PRES_RUN = tuple(p for p in PRES if p in args.pre)
    PB = json.load(open(LC.PARTB_JSON))
    READ_FILES[LC.PARTB_JSON] = "Part B"
    REC = load_family(PRES_RUN)
    POOL = {} if args.no_panels else load_known(PRES_RUN)
    NPAR = {s: num_params_known(s) for s in KNOWN11} if not args.no_panels else {}
    for i, s in enumerate(PYT):
        NPAR[s] = int(pb_lm(PB["num_params"], s))
    t_load = time.time() - t_all
    v, x = compute(REC, POOL, PB, NPAR, pres=PRES_RUN, with_panels=not args.no_panels)
    # ---------------- section G (spec 9)
    v["G|n_pairs_read"] = len(REC)
    v["G|n_known_records"] = len(POOL)
    v["G|l2_run"] = "l2" in PRES_RUN
    v["G|partB_sha256"] = sha(LC.PARTB_JSON)
    v["G|clip_equal_partB"] = clip_equal_partB(PRES_RUN)
    # ---------------- required keys (shared verdict_keys.py, names only): null for a pre not run; nothing missing
    VK, vk_sha = shared_verdict_keys()
    req = VK.required_keys("EI")
    for k in req:
        if k not in v and k.split("|")[1] in PRES and k.split("|")[1] not in PRES_RUN:
            v[k] = None
        elif k not in v and args.no_panels and k.split("|")[0] in ("famtr", "famdtr", "panel18"):
            v[k] = None
    missing = [k for k in req if k not in v]
    nonreq = sorted(set(v) - set(req))
    x["x|check|required_keys_missing"] = missing
    x["x|check|non_required_non_x_keys"] = nonreq
    x["x|check|verdict_keys_sha256"] = vk_sha
    x["x|precat_checks_E_side"] = precat_checks(v, PRES_RUN)
    x["x|selfchecks"] = selfchecks(REC)
    x["x|num_params"] = NPAR
    x["x|mode"] = {"test": LC.TEST, "device": LC.DEV, "pres": list(PRES_RUN), "panels": not args.no_panels,
                   "partB": LC.PARTB_JSON}
    vals = {k: fnum(val_) if not isinstance(val_, (str, list, dict)) and val_ is not None else val_
            for k, val_ in v.items()}
    out_v = dict(vals)
    out_v.update({k: LC.jsonable(val_) for k, val_ in x.items()})
    res_files = [p for p in READ_FILES if os.path.realpath(p).startswith(os.path.realpath(LC.RES))]
    other = [p for p in READ_FILES if p not in res_files and p != LC.PARTB_JSON]
    inputs = {LC.PARTB_JSON: sha(LC.PARTB_JSON),
              f"{LC.RES} ({len(res_files)} files read; sha256 over sorted relpath+sha256)": digest(res_files),
              f"known pair, clip and extraction files ({len(other)} read; sha256 over sorted relpath+sha256)":
                  digest(other)}
    verd = {"schema": "PYT-VERDICTS v1", "impl": "E", "inputs": inputs, "v": out_v}
    summ = {"role": "pyt-axis E (implementation E)", "plan": "analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md",
            "plan_sha256": LC.PLAN_SHA256, "spec": "analysis/s12_layermax_pythia/VERDICTS_SPEC_PYT.md",
            "panel": {"pythia": PYT, "sizes": list(SIZES), "vits": VITS, "known11": KNOWN11},
            "n_records_pythia": len(REC), "n_records_known": len(POOL), "inputs": inputs,
            "timing_s": {"load": t_load, "total": time.time() - t_all}, "n_required_keys": len(req),
            "n_required_missing": len(missing), "v": out_v, "finished_utc": LC.utc()}
    write_json(os.path.join(LC.RES, "summary.json"), summ)
    write_json(os.path.join(LC.RES, "verdicts_pyt.json"), verd)
    print(f"written {LC.RES}/summary.json and verdicts_pyt.json (pre run: {', '.join(PRES_RUN)}); required keys "
          f"{len(req)}, missing "
          f"{len(missing)}, non-required non-x keys {len(nonreq)}; {time.time() - t_all:.0f}s "
          f"(no verdict value printed, plan section 11)", flush=True)
    if missing:
        print("MISSING KEYS (names only):", missing[:50], flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
