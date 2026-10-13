"""OL-AXIS aggregation of implementation E (S0b sub-role E of oll-axis-extract). GATED (plan F1(a)).

Copied from analysis/s12_layermax_llama/early/aggregate_llama.py (sha256 d4de0157...) and rewritten for
analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md (sha256 4f18d710..., sections 5-10) and the verdict
contract analysis/s12_layermax_openllama/VERDICTS_SPEC.md (sections A, B, C, D, G). Every line that is not in
the original carries the marker "OL-AXIS". Kept from the original: the S1.2 trend arithmetic
(analysis/s12/aggregate.py, read-only import: _rho_matrix, boot_vits, vision_trend, lm_trend), `pear`,
`delta_boot` (key and panel changed, plan R.3.6), the field/decomposition logic and the self-checks.

Exactness (plan F2, spec section 2): every mKNN quantity (T, c_alpha, null mean, s_cal, uncensored, entry
c_alpha, M, M_unc, TM, p) is recomputed here as fractions.Fraction from the integer counts stored in the pair
files (mknn.entry_obs * 10240, mknn.null_counts_int16), for the OpenLLaMA LMs and for every LM of the pooled
panels; ranks, ties and order relations use the exact values. CKA values are the float64 values of the pair
files (ties only on exact equality; near-tie audit at 1e-12 relative).

Inputs (read-only except E's own outputs):
  <RES>/pairs/<pre>/openlm-research__open_llama_{3b,7b,13b}__<vit>.{json,npz}   (E reruns + 13B; S7)
  results/S1_2_layermax/pairs/<pre>/*.{json,npz}          frozen 8-LM run: BLOOMZ and llama-7b (pooled panels),
                                                           OpenLLaMA 3B/7B (V-O4b and the x|frozen extras only)
  results/S1_2_layermax/layermax_summary.json             frozen 8-LM summary (self-check only)
  results/S1_2_layermax_llama/early/pairs/<pre>/          llama-13b/30b (pooled panels; plan R.3.7)
  results/S1_2_layermax_llama/layermax_llama_summary.json merged LLaMA summary (section G check, plan R.3.7)
  results/S1_2_layermax_openllama/partB/partB_features_ol.json   every Part B quantity (spec section 8)

Outputs:
  <RES>/verdicts_ol.json  {"schema": "OL-VERDICTS v1", "impl": "E", "inputs": {path: sha256}, "v": {...}}:
                          every key of verdict_keys.required_keys("EI") plus x|... extras (ViT-bootstrap CIs
                          of Tr, FP section 7 decompositions, vision axis, null profiles, checks; and, if
                          V-O4b fails, x|frozen|<key> for every A-D key recomputed with the frozen 3B/7B values)
  <RES>/summary.json      the same values plus descriptive tables (per-LM means, per-ViT grids, self-checks,
                          timings, inputs)

Run (step S8, after V-O1..V-O4 and V-O7 pass; CPU, <= 4 threads; about 1-3 min):
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_openllama/launch_ol.sh ol_E_aggregate 4 -- \\
      venv/bin/python analysis/s12_layermax_openllama/early/aggregate_ol.py
  (S-E2 only: OL_SENS_E2=1 ... ol_E_e2_aggregate ...; outputs under early/sens_e2/)
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
import guard  # noqa: E402

PRES = LC.PRES
METRICS = ("mknn", "cka_biased", "cka_debiased")
CKAS = ("cka_biased", "cka_debiased")
SUBSETS = ("full", "fixed_r0.5", "fixed_r1.0", "grid2", "grid3", "grid5", "grid9", "grid17",
           "LMall_ViTlast", "LMlast_ViTall", "noLMemb", "LMdeep", "LMemb_only")
NEW_SUBSETS = ("noLMemb", "LMdeep", "LMemb_only")    # absent from the frozen 8-LM pair files
SIZES = ("3b", "7b", "13b")
OL = [LC.SLUG_3B, LC.SLUG_7B, LC.SLUG_13B]
VITS = [s for s, _ in LC.VITS]
NV = len(VITS)
BLOOMZ = ["bigscience__bloomz-560m", "bigscience__bloomz-1b1", "bigscience__bloomz-1b7", "bigscience__bloomz-3b",
          "bigscience__bloomz-7b1"]
LLAMA1 = ["huggyllama__llama-7b", "huggyllama__llama-13b", "huggyllama__llama-30b"]
POOLED11 = BLOOMZ + OL + LLAMA1                           # ARH order (spec section 1)
FROZEN8 = [s for s, _ in LC.C12.LMS]                      # frozen 8-LM order (= ARH order)
CNT = LC.M_ITEMS * LC.KNN_K                               # 10240
K = LC.K_PERM
OIDX = 476                                                # ceil(0.95 * 501), 1-based (hrhv2.calibrate.order_index)
TOL = 1e-9                                                # spec section 2
NB = 2000
MAT = 0.10
DELTA_B = 0.005
BAND = (0.0009, 0.0029)
TAUS = (0.03, 0.06, 0.10, 0.15, 0.20)
DBS = (0.0025, 0.005, 0.0075, 0.01)
BASE_MKNN = Fraction(LC.KNN_K, LC.M_ITEMS - 1)
DD_ARH = 0.40                                             # ARH's DeltaDelta (plan section 7(a))
A12 = None                                                # analysis/s12/aggregate.py, imported after the gate


def _load_a12():
    spec = importlib.util.spec_from_file_location("s12_aggregate", os.path.join(LC.PROJ, "analysis", "s12", "aggregate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


# ----------------------------------------------------------------------------
# loading: one record per (pre, LM, ViT)
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
                  "aA_unc": bju // LB, "aB_unc": bju % LB}
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
            "cM": float(c.reshape(-1)[j]), "aA": j // LB, "aB": j % LB, "aA_unc": ju // LB, "aB_unc": ju % LB}


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
            else:
                d["ctm"] = npz_ctm
            nm = np.asarray(z[f"{met}.entry_null_mean"], dtype=np.float64)
            d["D1_max_entry_mu"] = float(nm.max())
            d["null_profile"] = nm.max(axis=1)
        R[met] = d
    return R


def ol_base(src, pre, lm, vit):
    if src == "frozen" and lm in (LC.SLUG_3B, LC.SLUG_7B):
        return os.path.join(LC.FROZEN_RES, "pairs", pre, f"{lm}__{vit}")
    return os.path.join(LC.pairs_dir(pre, lm), f"{lm}__{vit}")


def pooled_base(pre, lm, vit):
    if lm in LLAMA1[1:]:
        return os.path.join(LC.LLAMA_E_RES, "pairs", pre, f"{lm}__{vit}")
    return os.path.join(LC.FROZEN_RES, "pairs", pre, f"{lm}__{vit}")


# ----------------------------------------------------------------------------
# value access
# ----------------------------------------------------------------------------

def val(R, met, q, x):
    """Score X of protocol Q (spec section 1) for one record; None if unavailable."""
    if q == "ctm":
        d = R[met]["ctm"]
        return d[{"M": "M", "M_unc": "M_unc", "TM": "TM"}[x]]
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
# section A: trends over the 3 OpenLLaMA sizes (exact ranks, 6 relabellings)
# ----------------------------------------------------------------------------

X6 = np.array([np.array(p) for p in itertools.permutations([1.0, 2.0, 3.0])])   # row 0 = identity


def rank3(t):
    """Average ranks of 3 values (exact comparisons; works for Fraction and float)."""
    return [1.0 + sum(1 for j in range(3) if t[j] < t[i]) + 0.5 * sum(1 for j in range(3) if j != i and t[j] == t[i])
            for i in range(3)]


def trend(trip):
    """trip: list over ViTs of (v3b, v7b, v13b). Returns T(sigma) (6,), per-ViT rho (identity), n excluded."""
    ok = [v for v in range(NV) if all(isfin(x) for x in trip[v]) and not (trip[v][0] == trip[v][1] == trip[v][2])]
    rho_obs = [None] * NV
    if not ok:
        return None, rho_obs, NV
    Rk = np.array([[rank3(trip[v])[i] for v in ok] for i in range(3)], dtype=np.float64)
    rho = A12._rho_matrix(X6, Rk)                    # (6, n_ok), Spearman = Pearson on average ranks
    for jj, v in enumerate(ok):
        rho_obs[v] = float(rho[0, jj])
    return rho.mean(axis=1), rho_obs, NV - len(ok)


def pvals(T, prefix):
    t = float(T[0])
    return {f"{prefix}_pge": float(np.mean(T >= t - TOL)), f"{prefix}_ple": float(np.mean(T <= t + TOL)),
            f"{prefix}_ptwo": float(np.mean(np.abs(T) >= abs(t) - TOL))}


def neartie(trip):
    n = 0
    for v in range(NV):
        for a, b in ((0, 1), (1, 2), (0, 2)):
            x, y = trip[v][a], trip[v][b]
            if isfin(x) and isfin(y) and abs(x - y) <= 1e-12 * max(abs(x), abs(y)):
                n += 1
    return n


# ----------------------------------------------------------------------------
# section D: pooled panels
# ----------------------------------------------------------------------------

def pear(G, x):
    xx = np.repeat(x[:, None], G.shape[1], axis=1).ravel()
    y = G.ravel()
    if np.nanstd(y) == 0 or not np.all(np.isfinite(y)):
        return float("nan")
    return float(np.corrcoef(xx, y)[0, 1])


def delta_boot(G1b, G1a, G2b, G2a, key, x):
    """aggregate_llama.py::delta_boot with the key and the panel changed (plan R.3.6, spec section 6)."""
    from hrhv2 import util
    f = lambda a, b, c, d: (pear(a, x) - pear(b, x)) - (pear(c, x) - pear(d, x))  # noqa: E731
    est = f(G1b, G1a, G2b, G2a)
    rng = util.rng_from_key(f"S1.2-layermax-openllama|boot|arh_delta|{key}")
    bs = np.empty(NB)
    for b in range(NB):
        j = rng.integers(0, NV, NV)
        bs[b] = f(G1b[:, j], G1a[:, j], G2b[:, j], G2a[:, j])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"est": float(est), "ci": [float(lo), float(hi)]}


def r_boot(G, x, key):
    from hrhv2 import util
    rng = util.rng_from_key(f"S1.2-layermax-openllama|boot|arh_r|{key}")
    bs = np.empty(NB)
    for b in range(NB):
        j = rng.integers(0, NV, NV)
        bs[b] = pear(G[:, j], x)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def gap_boot(Gm_after, Gc_after, x, key):
    from hrhv2 import util
    rng = util.rng_from_key(f"S1.2-layermax-openllama|boot|arh_gap|{key}")
    bs = np.empty(NB)
    for b in range(NB):
        j = rng.integers(0, NV, NV)
        bs[b] = pear(Gm_after[:, j], x) - pear(Gc_after[:, j], x)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def all_perms(n):
    """(n!, n) int8 array; each row a permutation of range(n)."""
    P = np.zeros((1, 0), dtype=np.int8)
    for k in range(n):
        m = P.shape[0]
        out = np.empty((m * (k + 1), k + 1), dtype=np.int8)
        for pos in range(k + 1):
            blk = out[pos * m:(pos + 1) * m]
            blk[:, :pos] = P[:, :pos]
            blk[:, pos] = k
            blk[:, pos + 1:] = P[:, pos:]
        P = out
    return P


def relabel_counts(x, W, obs, leaf=9):
    """Exact enumeration over all n! LM relabellings sigma (LM i receives rank value x[sigma(i)]).
    Statistic s under sigma = sum_i xc[sigma(i)] W[i, s] (r and DeltaDelta are linear in the LM row sums,
    spec section 6). Returns #{stat_s(sigma) >= obs_s - 1e-9} per s and n!. Chunked: the first n - leaf
    positions are enumerated recursively, the last leaf positions as one block of leaf! rows."""
    xc = np.asarray(x, dtype=np.float64) - np.mean(x)
    n, S = W.shape
    k = min(n, leaf)
    Pb = all_perms(k)
    counts = np.zeros(S, dtype=np.int64)
    total = [0]
    obs = np.asarray(obs, dtype=np.float64)

    def rec(avail, rows, offset):
        if len(rows) == k:
            V = xc[np.asarray(avail)][Pb]                   # (k!, k): value of each remaining row
            st = V @ W[rows] + offset
            counts[:] += np.sum(st >= obs[None, :] - TOL, axis=0)
            total[0] += V.shape[0]
            return
        i0 = rows[0]
        for j, v in enumerate(avail):
            rec(avail[:j] + avail[j + 1:], rows[1:], offset + xc[v] * W[i0])

    rec(list(range(n)), list(range(n)), np.zeros(S))
    assert total[0] == math.factorial(n)
    return counts, total[0]


def lin_weights(G, x):
    """W column of the Pearson r of grid G against the repeated rank vector: r = sum_i xc_i rows_i / den."""
    xc = np.asarray(x, dtype=np.float64) - np.mean(x)
    y = G - G.mean()
    rows = y.sum(axis=1)
    den = np.sqrt(G.shape[1] * np.sum(xc ** 2)) * np.sqrt(np.sum(y ** 2))
    return rows / den


# ----------------------------------------------------------------------------
# categories (plan sections 7, 9; spec section 5)
# ----------------------------------------------------------------------------

def cat_po2(material, tr_raw, d, s, tau=MAT):
    if not material:
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


def verdict_2n3(n, x):
    need = int(math.ceil(2 * n / 3)) if n else 0
    if n < 6:
        return need, "not evaluable"
    return need, "holds" if x >= need else "fails"


# ----------------------------------------------------------------------------
# the verdict computation (spec sections A, B, C, D) for one set of OpenLLaMA records
# ----------------------------------------------------------------------------

def pb_lm(d, slug):
    if slug in d:
        return d[slug]
    lab = LC.SIZE_LABEL.get(slug)
    if lab in d:
        return d[lab]
    raise KeyError(slug)


def compute(REC, POOL, PB, NPAR, label=""):
    """REC[(pre, lm, vit)] for the 3 OpenLLaMA LMs; POOL[(pre, lm, vit)] for the other 8 pooled LMs."""
    v, x = {}, {}
    t0 = time.time()

    def trip(pre, met, q, X):
        out = []
        for vit in VITS:
            out.append(tuple(val(REC[(pre, lm, vit)], met, q, X) for lm in OL))
        return out

    TR = {}     # (pre, met, q, X) -> (T6, rho_obs, nexcl)
    for pre in PRES:
        for met in METRICS:
            for q in SUBSETS + ("ctm",):
                xs = ("M", "M_unc", "TM") if q == "ctm" else ("raw", "s_cal", "unc")
                for X in xs:
                    tp = trip(pre, met, q, X)
                    if any(t is None for tr_ in tp for t in tr_):
                        T6, rho, ne = None, [None] * NV, None
                    else:
                        T6, rho, ne = trend(tp)
                    TR[(pre, met, q, X)] = (T6, rho, ne)
                    key = f"{pre}|{met}|{q}|{X}"
                    v[f"tr|{key}"] = None if T6 is None else float(T6[0])
                    v[f"tr_nconst|{key}"] = ne
                    if T6 is None:
                        v.update({f"tr_pge|{key}": None, f"tr_ple|{key}": None, f"tr_ptwo|{key}": None})
                    else:
                        for k_, val_ in pvals(T6, "tr").items():
                            v[f"{k_}|{key}"] = val_
                    if met != "mknn":
                        v[f"neartie|{key}"] = None if any(t is None for tr_ in tp for t in tr_) else neartie(tp)
                    if T6 is not None:
                        _, lo, hi = A12.boot_vits(np.array([np.nan if r is None else r for r in rho]),
                                                  f"layermax-openllama|{key}")
                        x[f"x|tr_ci|{key}"] = [lo, hi]
                # differences
                ds = ("M", "M_unc", "TM") if q == "ctm" else ("s_cal", "unc")
                for X in ds:
                    key = f"{pre}|{met}|{q}|{X}"
                    base = TR[(pre, met, "full", "raw")] if q == "ctm" else TR[(pre, met, q, "raw")]
                    a = TR[(pre, met, q, X)][0]
                    if a is None or base[0] is None:
                        v.update({f"dtr|{key}": None, f"dtr_pge|{key}": None, f"dtr_ple|{key}": None,
                                  f"dtr_ptwo|{key}": None})
                        continue
                    D = a - base[0]
                    v[f"dtr|{key}"] = float(D[0])
                    for k_, val_ in pvals(D, "dtr").items():
                        v[f"{k_}|{key}"] = val_
            v[f"d3cal|{pre}|{met}"] = _sub(v[f"tr|{pre}|{met}|ctm|M"], v[f"tr|{pre}|{met}|ctm|TM"])
            v[f"d3sel|{pre}|{met}"] = _sub(v[f"tr|{pre}|{met}|ctm|TM"], v[f"tr|{pre}|{met}|full|raw"])
    x["x|timing|A_s"] = time.time() - t0

    # ---------------- section B: order changes
    for pre in PRES:
        P = PB["per_pre"][pre]
        s_O = int(P["s_O"])
        for met in METRICS:
            for q in SUBSETS + ("ctm",):
                xs = ("M", "M_unc") if q == "ctm" else ("s_cal", "unc")
                for X in xs:
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

    # ---------------- section D (needed by P-O4): pooled panels
    t1 = time.time()
    D = pooled(REC, POOL, NPAR, v, x)
    v.update(D)
    x["x|timing|D_s"] = time.time() - t1

    # ---------------- section C: Part B decision rules
    for pre in PRES:
        P = PB["per_pre"][pre]
        s_O, s_late = int(P["s_O"]), int(P["s_late"])
        material, mat_late = bool(P["material"]), bool(P["material_Blate"])
        Bbar = {lm: {vit: float(pb_lm(P["Bbar"], lm)[vit]) for vit in VITS} for lm in OL}
        Blate = {lm: {vit: float(pb_lm(P["Blate"], lm)[vit]) for vit in VITS} for lm in OL}
        tr = lambda met, q, X: v[f"tr|{pre}|{met}|{q}|{X}"]  # noqa: E731
        dtr = lambda met, q, X: v[f"dtr|{pre}|{met}|{q}|{X}"]  # noqa: E731
        # P-O1
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
        # P-O2
        trraw = tr("cka_biased", "full", "raw")
        for X in ("s_cal", "unc"):
            d = dtr("cka_biased", "full", X)
            v[f"PO2|{pre}|{X}|cat"] = cat_po2(material, trraw, d, s_O)
            dist = None if d is None else min(abs(s_O * d + MAT), abs(s_O * d - MAT))
            v[f"PO2|{pre}|{X}|dist"] = dist
            v[f"PO2|{pre}|{X}|kedge"] = None if dist is None else bool(dist <= 1.0 / NV + TOL)
            for tau in TAUS:
                v[f"curve|{pre}|PO2|{X}|tau={tau!r}"] = cat_po2(material, trraw, d, s_O, tau)
        for tau in TAUS:
            v[f"curve|{pre}|PO1|tau={tau!r}"] = bool(d1s is not None and d1u is not None and abs(d1s) < tau
                                                     and abs(d1u) < tau)
        # P-O2a(i)
        offs = {vit: REC[(pre, LC.SLUG_13B, vit)]["cka_biased"]["full"]["c"] - Bbar[LC.SLUG_13B][vit] for vit in VITS}
        med = median(offs.values())
        xin = sum(1 for o in offs.values() if BAND[0] <= o <= BAND[1])
        lo, hi = cp95(xin, NV)
        lpm = median(abs(o) for o in offs.values())
        v.update({f"PO2ai|{pre}|median": med, f"PO2ai|{pre}|in_band": bool(BAND[0] <= med <= BAND[1]),
                  f"PO2ai|{pre}|n": NV, f"PO2ai|{pre}|x": xin, f"PO2ai|{pre}|cp_lo": lo, f"PO2ai|{pre}|cp_hi": hi,
                  f"PO2ai|{pre}|lp_median_abs": lpm, f"PO2ai|{pre}|lp_le_0.01": bool(lpm <= 0.01)})
        for vit in VITS:
            v[f"PO2ai|{pre}|offset|{vit}"] = offs[vit]
        # P-O2a(ii) (and its delta_b curve)
        for db in (DELTA_B,) + DBS:
            n = xx = 0
            for vit in VITS:
                for a in (LC.SLUG_3B, LC.SLUG_7B):
                    gB = Bbar[LC.SLUG_13B][vit] - Bbar[a][vit]
                    if abs(gB) >= db:
                        n += 1
                        gc = REC[(pre, LC.SLUG_13B, vit)]["cka_biased"]["full"]["c"] - REC[(pre, a, vit)]["cka_biased"]["full"]["c"]
                        xx += int(sgn(gc) == sgn(gB))
            need, verd = verdict_2n3(n, xx)
            if db == DELTA_B and f"PO2aii|{pre}|n" not in v:
                lo, hi = cp95(xx, n)
                v.update({f"PO2aii|{pre}|n": n, f"PO2aii|{pre}|x": xx, f"PO2aii|{pre}|need": need,
                          f"PO2aii|{pre}|verdict": verd, f"PO2aii|{pre}|cp_lo": lo, f"PO2aii|{pre}|cp_hi": hi})
            if db in DBS:
                v.update({f"curve|{pre}|PO2aii|db={db!r}|n": n, f"curve|{pre}|PO2aii|db={db!r}|x": xx,
                          f"curve|{pre}|PO2aii|db={db!r}|verdict": verd})
        # materiality curve from the per-ViT B-bar of Part B (absolute gaps, LP section 9 / partb_pr.py rule)
        trp = float(P["Tr_pred"])
        G3 = np.array([[Bbar[lm][vit] for vit in VITS] for lm in OL])
        inc = bool(np.all(np.diff(G3, axis=0) > 0))
        dec = bool(np.all(np.diff(G3, axis=0) < 0))
        meds = [float(np.median(np.abs(G3[b] - G3[a]))) for a, b in ((0, 1), (1, 2), (0, 2))]
        for db in DBS:
            v[f"curve|{pre}|material|db={db!r}"] = bool(abs(abs(trp) - 1.0) <= TOL and (inc or dec)
                                                       and all(m_ >= db for m_ in meds))
        x[f"x|check|{pre}|material_curve_db0.005_equals_partB"] = bool(v[f"curve|{pre}|material|db=0.005"] == material)
        x[f"x|Bbar_median_abs_gaps|{pre}"] = {"3b|7b": meds[0], "7b|13b": meds[1], "3b|13b": meds[2]}
        # P-O3
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
        # keep identity and sign rule (all M, Q != ctm, all 51 comparisons)
        for met in METRICS:
            for q in SUBSETS:
                for X in ("s_cal", "unc"):
                    nk = ns = 0
                    for vit in VITS:
                        for a, b in ((0, 1), (1, 2), (0, 2)):
                            Ra = REC[(pre, OL[a], vit)][met].get(q)
                            Rb = REC[(pre, OL[b], vit)][met].get(q)
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
        # P-O4
        d4s, d4u = dtr("cka_debiased", "full", "s_cal"), dtr("cka_debiased", "full", "unc")
        v[f"PO4|{pre}|s_cal|holds"] = bool(d4s is not None and d4s >= -MAT)
        v[f"PO4|{pre}|unc|holds"] = bool(d4u is not None and d4u >= -MAT)
        ra_, rb_ = v[f"arh|11|{pre}|cka_debiased|s_cal|arh|r"], v[f"arh|11|{pre}|cka_debiased|raw|arh|r"]
        v[f"PO4|{pre}|arh|holds"] = bool(isfin(ra_) and isfin(rb_) and ra_ >= rb_ - 0.05)
        # P-O5a, P-O5
        n = xx = 0
        for vit in VITS:
            for a in (LC.SLUG_3B, LC.SLUG_7B):
                gB = Blate[LC.SLUG_13B][vit] - Blate[a][vit]
                if abs(gB) >= DELTA_B:
                    n += 1
                    gc = REC[(pre, LC.SLUG_13B, vit)]["cka_biased"]["ctm"]["cM"] - REC[(pre, a, vit)]["cka_biased"]["ctm"]["cM"]
                    xx += int(sgn(gc) == sgn(gB))
        need, po5a = verdict_2n3(n, xx)
        v.update({f"PO5a|{pre}|n": n, f"PO5a|{pre}|x": xx, f"PO5a|{pre}|need": need, f"PO5a|{pre}|verdict": po5a})
        for X in ("M", "M_unc"):
            d = dtr("cka_biased", "ctm", X)
            v[f"PO5|{pre}|{X}|pred_holds"] = None if not mat_late else (None if d is None else bool(s_late * d <= 0))
            v[f"PO5|{pre}|{X}|cat"] = cat_po5(mat_late, po5a, trraw, d, s_late)
            for tau in TAUS:
                v[f"curve|{pre}|PO5|{X}|tau={tau!r}"] = cat_po5(mat_late, po5a, trraw, d, s_late, tau)
        # D3 deductive check
        rhoM = TR[(pre, "cka_biased", "ctm", "M")][1]
        rhoTM = TR[(pre, "cka_biased", "ctm", "TM")][1]
        nv3 = 0
        for j, vit in enumerate(VITS):
            cm = [REC[(pre, lm, vit)]["cka_biased"]["ctm"]["cM"] for lm in OL]
            if cm[0] < cm[1] < cm[2]:
                rm_ = 0.0 if rhoM[j] is None else rhoM[j]      # an excluded (all-equal) ViT has rho 0
                rt_ = 0.0 if rhoTM[j] is None else rhoTM[j]
                nv3 += int(rm_ - rt_ > 0)
        v[f"D3|{pre}|nviol"] = nv3
        # plug-in check
        tp = [tuple(REC[(pre, lm, vit)]["cka_biased"]["full"]["T"] - Bbar[lm][vit] for lm in OL) for vit in VITS]
        T6, _, _ = trend(tp)
        pt = None if T6 is None else float(T6[0])
        tu = tr("cka_biased", "full", "unc")
        v[f"plugin|{pre}|tr"] = pt
        v[f"plugin|{pre}|diff"] = _sub(pt, tu)
        v[f"plugin|{pre}|within"] = bool(pt is not None and tu is not None and abs(pt - tu) <= MAT)
        # section 7 OpenLLaMA-axis reading
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
        # D1, D2, frac7gt3
        for met in METRICS:
            for lm, sz in zip(OL, SIZES):
                d1 = [REC[(pre, lm, vit)][met]["full"]["mu"] - REC[(pre, lm, vit)][met]["D1_max_entry_mu"] for vit in VITS]
                v[f"D1|{pre}|{met}|{sz}"] = median(d1)
                ar = [REC[(pre, lm, vit)][met]["full"]["aA"] / (REC[(pre, lm, vit)]["LA"] - 1) for vit in VITS]
                v[f"D2|{pre}|{met}|{sz}|argmax_rel"] = median(ar)
                cc = [REC[(pre, lm, vit)][met]["full"]["c"] - REC[(pre, lm, vit)][met]["full"]["c_entry"] for vit in VITS]
                v[f"D2|{pre}|{met}|{sz}|c_minus_centry"] = median(cc)
            g = [int(REC[(pre, OL[1], vit)][met]["full"]["T"] > REC[(pre, OL[0], vit)][met]["full"]["T"]) for vit in VITS]
            v[f"frac7gt3|{pre}|{met}"] = sum(g) / NV
        for lm, sz in zip(OL, SIZES):
            x[f"x|D2|{pre}|{sz}|u_star_rel_partB"] = _get(P, "u_star_rel", lm)
        # extras: FP section 7 decomposition (A3), vision axis, null profiles, per-LM means
        for met in METRICS:
            for col in ("c", "mu", "shift", "sel_entry_thr", "sel_entry_mean", "inflation", "c_entry"):
                T6, _, _ = trend([tuple(val(REC[(pre, lm, vit)], met, "full", col) for lm in OL) for vit in VITS])
                x[f"x|decomp|{pre}|{met}|{col}"] = None if T6 is None else float(T6[0])
            for q in ("full", "fixed_r0.5", "fixed_r1.0"):
                for X in ("raw", "s_cal", "unc"):
                    G3 = np.array([[fnum(val(REC[(pre, lm, vit)], met, q, X)) for vit in VITS] for lm in OL], dtype=float)
                    if np.all(np.isfinite(G3)):
                        vt = A12.vision_trend(G3)
                        x[f"x|vision|{pre}|{met}|{q}|{X}"] = {"Tr": vt["Tr"], "p_one": vt["p_one"], "n_cells": vt["n_cells"]}
            for lm, sz in zip(OL, SIZES):
                prof = np.mean(np.stack([np.asarray(REC[(pre, lm, vit)][met]["null_profile"], dtype=float)
                                         for vit in VITS]), axis=0)
                x[f"x|nullprof|{pre}|{met}|{sz}|argmax_rel"] = float(np.argmax(prof) / (prof.size - 1))
                for col in ("raw", "c", "s_cal", "unc", "mu"):
                    x[f"x|mean|{pre}|{met}|full|{col}|{sz}"] = float(np.mean([float(val(REC[(pre, lm, vit)], met, "full", col))
                                                                            for vit in VITS]))
                for X in ("M", "M_unc", "TM"):
                    x[f"x|mean|{pre}|{met}|ctm|{X}|{sz}"] = float(np.mean([float(val(REC[(pre, lm, vit)], met, "ctm", X))
                                                                         for vit in VITS]))
            for lm, sz in zip(OL, SIZES):
                x[f"x|mean|{pre}|cka_biased|full|Bbar|{sz}"] = float(np.mean([Bbar[lm][vit] for vit in VITS]))
    return v, x


def _sub(a, b):
    return None if a is None or b is None else a - b


def _get(P, field, lm):
    try:
        return pb_lm(P[field], lm)
    except (KeyError, TypeError):
        return None


def pooled(REC, POOL, NPAR, v, x):
    """Section D (spec section 6). Writes r, CIs, exact p, DeltaDelta, section 7(a) rule and FP section 6."""
    D = {}
    panels = {"11": POOLED11, "10": [s for s in POOLED11 if s != LC.SLUG_13B], "8": FROZEN8}
    xs = {}
    for pan, lms in panels.items():
        arh = np.array([LC.arh_rank(s) for s in lms], dtype=float)
        xs[pan] = {"arh": arh, "dense": np.arange(1, len(lms) + 1, dtype=float),
                   "logp": np.log(np.array([NPAR[s] for s in lms], dtype=float))}
    x11 = xs["11"]["arh"]
    assert list(x11) == list(range(1, 12)), x11

    def grid(pre, lms, met, X):
        G = np.empty((len(lms), NV))
        for i, lm in enumerate(lms):
            for j, vit in enumerate(VITS):
                R = REC[(pre, lm, vit)] if lm in OL else POOL[(pre, lm, vit)]
                G[i, j] = float(val(R, met, "full", X))
        return G

    W, OBS, NAMES = [], [], []
    for pre in PRES:
        Gs = {}
        for pan, lms in panels.items():
            for met in METRICS:
                for X in ("raw", "s_cal", "unc"):
                    G = grid(pre, lms, met, X)
                    Gs[(pan, met, X)] = G
                    for kind, xv in xs[pan].items():
                        r = pear(G, xv)
                        D[f"arh|{pan}|{pre}|{met}|{X}|{kind}|r"] = r
                        if pan == "11":
                            lo, hi = r_boot(G, xv, f"{pre}|{met}|{X}|{kind}")
                            D[f"arh|11|{pre}|{met}|{X}|{kind}|ci_lo"] = lo
                            D[f"arh|11|{pre}|{met}|{X}|{kind}|ci_hi"] = hi
                    if pan == "11":
                        W.append(lin_weights(G, x11))
                        OBS.append(D[f"arh|11|{pre}|{met}|{X}|arh|r"])
                        NAMES.append(f"arh|11|{pre}|{met}|{X}|arh|pge")
            for cka in CKAS:
                rr = lambda met, X: D[f"arh|{pan}|{pre}|{met}|{X}|arh|r"]  # noqa: E731
                D[f"dd|{pan}|{pre}|{cka}"] = (rr(cka, "raw") - rr(cka, "s_cal")) - (rr("mknn", "raw") - rr("mknn", "s_cal"))
        x[f"x|panel11_grids|{pre}"] = {f"{met}|{X}": Gs[("11", met, X)].tolist() for met in METRICS
                                       for X in ("raw", "s_cal", "unc")}
        for cka in CKAS:
            b = delta_boot(Gs[("11", cka, "raw")], Gs[("11", cka, "s_cal")], Gs[("11", "mknn", "raw")],
                           Gs[("11", "mknn", "s_cal")], f"{pre}|{cka}", x11)
            D[f"dd|11|{pre}|{cka}|ci_lo"], D[f"dd|11|{pre}|{cka}|ci_hi"] = b["ci"]
            x[f"x|dd_boot_est|{pre}|{cka}"] = b["est"]
            W.append(lin_weights(Gs[("11", cka, "raw")], x11) - lin_weights(Gs[("11", cka, "s_cal")], x11)
                     - lin_weights(Gs[("11", "mknn", "raw")], x11) + lin_weights(Gs[("11", "mknn", "s_cal")], x11))
            OBS.append(D[f"dd|11|{pre}|{cka}"])
            NAMES.append(f"dd|11|{pre}|{cka}|pge")
        lo, hi = D[f"dd|11|{pre}|cka_biased|ci_lo"], D[f"dd|11|{pre}|cka_biased|ci_hi"]
        D[f"rule7a|{pre}"] = ("replicated the direction" if lo > 0 else
                              ("opposite direction" if hi < 0 else "direction not established"))
        D[f"size_not_reproduced|{pre}"] = bool(DD_ARH > hi)
        dd = D[f"dd|11|{pre}|cka_biased"]
        rca, rma = D[f"arh|11|{pre}|cka_biased|s_cal|arh|r"], D[f"arh|11|{pre}|mknn|s_cal|arh|r"]
        if dd >= 0.20 and rca <= rma - 0.20:
            raw_cat = "ARH pattern reappears"
        elif dd <= 0.10 or rca >= rma - 0.10:
            raw_cat = "absent"
        else:
            raw_cat = "partial"
        glo, ghi = gap_boot(Gs[("11", "mknn", "s_cal")], Gs[("11", "cka_biased", "s_cal")], x11, pre)
        kedge = bool((lo <= 0.10 <= hi) or (glo <= 0.10 <= ghi))
        D.update({f"fp6|{pre}|raw_cat": raw_cat, f"fp6|{pre}|gap_ci_lo": glo, f"fp6|{pre}|gap_ci_hi": ghi,
                  f"fp6|{pre}|kedge": kedge,
                  f"fp6|{pre}|cat": "not decidable at this panel size (knife edge)" if kedge else raw_cat})
    # exact p over all 11! LM relabellings, every statistic in one enumeration
    t0 = time.time()
    good = [i for i, o in enumerate(OBS) if math.isfinite(o) and np.all(np.isfinite(W[i]))]
    cnts, tot = relabel_counts(x11, np.stack([W[i] for i in good], axis=1), [OBS[i] for i in good])
    for i in range(len(NAMES)):
        D[NAMES[i]] = None
    for jj, i in enumerate(good):
        D[NAMES[i]] = float(cnts[jj] / tot)
    x["x|timing|enum11_s"] = time.time() - t0
    x["x|enum11_n"] = tot
    return D


# ----------------------------------------------------------------------------
# section G and checks
# ----------------------------------------------------------------------------

def check_llama_merged(POOLJ):
    """G|llama_pairs_equal_merged_summary: llama-13b/30b values read from E's LLaMA-axis pair files vs the
    merged LLaMA summary (embedded I per-pair grids of every subset; embedded E per-LM means), 1e-12 relative."""
    S = json.load(open(LC.LLAMA_MERGED))
    READ_FILES[LC.LLAMA_MERGED] = "merged LLaMA summary"
    im = {"mknn": "mknn", "cka_biased": "cka_b", "cka_debiased": "cka_d"}
    qm = {"LMall_ViTlast": "lmall_x_vitlast", "LMlast_ViTall": "lmlast_x_vitall"}
    cm = {"s_obs": "s_obs", "s_cal": "s_cal", "uncensored": "uncensored", "c_alpha": "c_alpha", "null_mean": "null_mean"}
    n = nf = 0
    worst = 0.0
    fails = []

    def cmp(a, b, where):
        nonlocal n, nf, worst
        n += 1
        d = abs(a - b)
        worst = max(worst, d / max(abs(a), abs(b)) if d > 0 else 0.0)
        if d > 1e-12 * max(abs(a), abs(b)) + 1e-15:
            nf += 1
            if len(fails) < 20:
                fails.append({"where": where, "pairs": a, "merged": b})

    for pre in PRES:
        Ib = S["embedded"]["indep_summary"]["per_pre"][pre]["blocks"]
        Em = S["embedded"]["early_summary"]["per_pre"][pre]["means"]
        for met in METRICS:
            for q in SUBSETS:
                g = Ib[im[met]][qm.get(q, q)]["grids_llama"]
                for col, ic in cm.items():
                    for row, lm in ((1, LLAMA1[1]), (2, LLAMA1[2])):
                        for j, vit in enumerate(VITS):
                            cmp(float(POOLJ[(pre, lm, vit)]["summaries"][met][q][col]), float(g[ic][row][j]),
                                f"I|{pre}|{met}|{q}|{col}|{lm}|{vit}")
            for q in ("full", "fixed_r0.5", "fixed_r1.0"):
                for col in ("s_obs", "c_alpha", "s_cal", "uncensored", "null_mean", "p_value"):
                    for lm in LLAMA1[1:]:
                        mine = float(np.mean([POOLJ[(pre, lm, vit)]["summaries"][met][q][col] for vit in VITS]))
                        cmp(mine, float(Em[met][f"{q}|{col}"][lm]), f"Emeans|{pre}|{met}|{q}|{col}|{lm}")
    return {"n_compared": n, "n_fail": nf, "max_rel": worst, "fails": fails, "pass": nf == 0 and n > 0,
            "merged_sha256": sha(LC.LLAMA_MERGED)}


def num_params(stem):
    for base in (os.path.join(LC.FEAT_W1024, stem), os.path.join(LC.PROJ, "features", "wit1024_llama_axis", stem)):
        if os.path.exists(base + ".json"):
            return int(json.load(open(base + ".json"))["num_params"])
    return LC.n_params(stem)


def selfchecks(REC, POOL, POOLJ):
    out = {}
    # trend code vs S1.2 lm_trend on the frozen BLOOMZ rows (biased CKA, full, uncensored)
    G8 = np.array([[POOLJ[("prh_clip+l2", lm, vit)]["summaries"]["cka_biased"]["full"]["uncensored"] for vit in VITS]
                   for lm in FROZEN8])
    a = A12.lm_trend(G8)
    fz = json.load(open(os.path.join(LC.FROZEN_RES, "layermax_summary.json")))
    READ_FILES[os.path.join(LC.FROZEN_RES, "layermax_summary.json")] = "frozen 8-LM summary"
    f0 = fz["per_pre"]["prh_clip+l2"]["trends"]["cka_biased"]["full|uncensored"]["tr_lm"]["Tr"]
    out["bloomz_Tr_unc_cka_vs_frozen_summary"] = {"recomputed": a["Tr"], "frozen": f0, "equal": bool(a["Tr"] == f0)}
    # mKNN exact vs json floats (all records)
    nd = 0
    nall = 0
    for R in list(REC.values()) + list(POOL.values()):
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
    # calibrate-then-max: E torch (json) vs numpy from npz
    mx = 0.0
    for R in REC.values():
        for met in CKAS:
            mx = max(mx, R[met].get("ctm_json_vs_npz_maxabs", 0.0))
    out["ctm_cka_json_vs_npz_numpy_maxabs"] = mx
    out["perm_keys_ok"] = all(R["perm_key"] == LC.perm_key(R["lm"], R["vit"]) for R in REC.values())
    out["K_values"] = sorted({R["K"] for R in REC.values()})
    out["lm_fields_ok"] = all(R["lm"] == lm and R["vit"] == vit for (pre, lm, vit), R in REC.items())
    return out


def digest(paths):
    d = hashlib.sha256()
    for p in sorted(paths):
        d.update(os.path.relpath(p, LC.PROJ).encode())
        d.update(sha(p).encode())
    return d.hexdigest()


def main():
    global A12
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    guard.require_partb_frozen(LC.RES, script="early/aggregate_ol.py")   # plan F1(a): first action
    t_all = time.time()
    A12 = _load_a12()
    import vl4b_ol
    PB = json.load(open(LC.PARTB_JSON))
    READ_FILES[LC.PARTB_JSON] = "Part B"
    # ---------------- load
    REC = {(pre, lm, vit): load_record(ol_base("rerun", pre, lm, vit)) for pre in PRES for lm in OL for vit in VITS}
    POOLJ, POOL = {}, {}
    for pre in PRES:
        for lm in POOLED11:
            if lm in OL:
                continue
            for vit in VITS:
                b = pooled_base(pre, lm, vit)
                POOL[(pre, lm, vit)] = load_record(b, full_only=True)
                POOLJ[(pre, lm, vit)] = json.load(open(b + ".json"))
    for pre in PRES:            # the frozen open_llama 3B/7B json files (BLOOMZ self-check panel, x|frozen)
        for lm in (LC.SLUG_3B, LC.SLUG_7B):
            for vit in VITS:
                POOLJ[(pre, lm, vit)] = json.load(open(ol_base("frozen", pre, lm, vit) + ".json"))
    NPAR = {s: num_params(s) for s in POOLED11}
    t_load = time.time() - t_all
    # ---------------- main computation
    v, x = compute(REC, POOL, PB, NPAR)
    # ---------------- section G
    g1 = check_llama_merged(POOLJ)
    vo4b = vl4b_ol.compare(verbose=False)
    v["G|llama_pairs_equal_merged_summary"] = bool(g1["pass"])
    v["G|rerun_equals_frozen_3b7b"] = bool(vo4b["pass"])
    v["G|n_pairs_read"] = len([k for k in list(REC) + list(POOL) if k[1] in POOLED11])
    x["x|G|llama_merged_check"] = g1
    x["x|G|vo4b"] = {p: {k: vo4b["per_pre"][p][k] for k in ("n_pairs", "n_fields", "max_rel", "pass",
                                                            "bitwise_all_fields_and_arrays")} for p in PRES}
    if not vo4b["pass"]:       # plan section 3 source rule: every verdict also on the frozen 3B/7B values
        RECF = dict(REC)
        for pre in PRES:
            for lm in (LC.SLUG_3B, LC.SLUG_7B):
                for vit in VITS:
                    RECF[(pre, lm, vit)] = load_record(ol_base("frozen", pre, lm, vit))
        vf, _ = compute(RECF, POOL, PB, NPAR, label="frozen")
        for k, val_ in vf.items():
            x[f"x|frozen|{k}"] = val_
    # ---------------- checks and extras
    import verdict_keys as VK
    req = VK.required_keys("EI")
    missing = [k for k in req if k not in v]
    extra = [k for k in v if k not in set(req)]
    x["x|check|required_keys_missing"] = missing
    x["x|check|non_required_non_x_keys"] = extra
    x["x|selfchecks"] = selfchecks(REC, POOL, POOLJ)
    x["x|num_params"] = NPAR
    pbn = PB.get("num_params", {})
    x["x|check|num_params_equal_partB"] = {s: (pbn.get(s) == NPAR[s]) for s in NPAR if s in pbn}
    x["x|mode"] = {"test": LC.TEST, "sens_e2": LC.SENS_E2}
    vals = {k: fnum(val_) if not isinstance(val_, (str, list, dict)) and val_ is not None else val_
            for k, val_ in v.items()}
    out_v = dict(vals)
    out_v.update({k: LC.jsonable(val_) for k, val_ in x.items()})
    # ---------------- inputs
    ol_files = [p for p in READ_FILES if os.path.realpath(p).startswith(os.path.realpath(LC.RES_PRIMARY))]
    llama_files = [p for p in READ_FILES if p.startswith(LC.LLAMA_E_RES)]
    frozen_files = [p for p in READ_FILES if p.startswith(os.path.join(LC.FROZEN_RES, "pairs"))]
    inputs = {LC.PARTB_JSON: sha(LC.PARTB_JSON), LC.LLAMA_MERGED: sha(LC.LLAMA_MERGED),
              os.path.join(LC.FROZEN_RES, "layermax_summary.json"): sha(os.path.join(LC.FROZEN_RES, "layermax_summary.json")),
              f"{LC.RES_PRIMARY}/pairs ({len(ol_files)} files read; sha256 over sorted relpath+sha256)": digest(ol_files),
              f"{LC.LLAMA_E_RES}/pairs ({len(llama_files)} files read; sha256 over sorted relpath+sha256)": digest(llama_files),
              f"{LC.FROZEN_RES}/pairs ({len(frozen_files)} files read; sha256 over sorted relpath+sha256)": digest(frozen_files)}
    llama_all = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(LC.LLAMA_E_RES, "pairs")) for f in fs)
    d = hashlib.sha256()
    for p in llama_all:   # the digest form of write_hashes.py of the LLaMA axis (plan section 3: 0d918377...)
        rp = os.path.relpath(p, LC.PROJ)
        d.update(rp.encode())
        d.update(sha(p).encode())
    x_llama_digest = {"n_files": len(llama_all), "digest": d.hexdigest(),
                      "expected": "0d918377e7dc22af4cb05f52d4664717a6faaa1476e0aabe41e3d1ba29bcd1b0"}
    x_llama_digest["equal"] = x_llama_digest["digest"] == x_llama_digest["expected"]
    out_v["x|check|llama_E_pairs_digest"] = x_llama_digest
    verd = {"schema": "OL-VERDICTS v1", "impl": "E", "inputs": inputs, "v": out_v}
    summ = {"role": "oll-axis E (implementation E)", "plan": "analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md",
            "plan_sha256": "4f18d710beda45492b1e191f74580bfc4ef53f6dafe791ad245ff1cf335f3240",
            "spec": "analysis/s12_layermax_openllama/VERDICTS_SPEC.md",
            "panel": {"openllama": OL, "vits": VITS, "pooled11": POOLED11, "frozen8": FROZEN8},
            "n_records_ol": len(REC), "n_records_pooled_other": len(POOL), "inputs": inputs,
            "timing_s": {"load": t_load, "total": time.time() - t_all},
            "n_required_keys": len(req), "n_required_missing": len(missing),
            "v": out_v, "finished_utc": LC.utc()}
    os.makedirs(LC.RES, exist_ok=True)
    for name, obj in (("summary.json", summ), ("verdicts_ol.json", verd)):
        with open(os.path.join(LC.RES, name + ".tmp"), "w") as f:
            json.dump(LC.jsonable(obj), f, indent=1)
        os.replace(os.path.join(LC.RES, name + ".tmp"), os.path.join(LC.RES, name))
    print(f"written {LC.RES}/summary.json and verdicts_ol.json; required keys {len(req)}, missing {len(missing)}; "
          f"{time.time() - t_all:.0f}s", flush=True)
    if missing:
        print("MISSING KEYS", missing[:50], flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
