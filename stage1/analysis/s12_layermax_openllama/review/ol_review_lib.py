"""OpenLLaMA axis, sub-role R (review recomputation): data layer and exact arithmetic.

NEW FILE (OL-AXIS), helper of review/ol_axis_review.py. Written at S0b by the R sub-role of role
oll-axis-extract (2026-10-09), before any OpenLLaMA-13B byte existed on ds2213. Hashed in the CODE FROZEN line.

Contract: analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md (sha256 4f18d710...) and VERDICTS_SPEC.md.
Built only from the plan, the spec, the LLaMA-axis review script and the stored pair-file formats (read-only).
No E/I/forecast/joint code of this axis was read.

What is here
------------
* Pair-file readers for both formats, returning per-pair statistics ("PairStats") for every protocol of the
  spec (13 layer subsets + calibrate-then-max) and every metric:
    - E format: <dir>/<lm>__<vit>.json (summaries per metric/subset, written by early/run_layermax.py) and
      <dir>/<lm>__<vit>.npz (float64 entry arrays, float64 full-layer-max nulls, mKNN int16 null counts,
      float32 CKA null tensors).
    - I format: <dir>/<lm>__<vit>.npz with the full observed and null matrices (float64 CKA, integer counts).
* mKNN is exact (plan F2): every mKNN quantity is a fractions.Fraction built from the integer counts out of
  m*k = 10,240 (observed counts, the 500 null counts, c_alpha as a count, the null mean as sum/500, entry-wise
  c_alpha and null mean for calibrate-then-max, M, M_unc, TM).
* Biased / debiased CKA (float64):
    - I format: every statistic recomputed from the stored float64 observed and null matrices with the
      hrhv2 calibrate.summarize arithmetic (c_alpha = 476th ascending order statistic of {T} U {T_k};
      null mean = numpy mean of the 500 maxima; s_cal = max((T - c)/(1 - c), 0); unc = T - null mean).
    - E format: the protocol `full` is recomputed from the stored float64 full-layer-max nulls; T of every
      subset from the float64 entry matrix; c_alpha and the null mean of the other subsets are the stored
      float64 summary values (the float64 subset maxima are not stored), verified against the stored float32
      null tensors (c_alpha: the float32 order statistic must equal float32(c_alpha); null mean within 1e-6
      relative; p recomputed from the float32 maxima). Every check result is returned for the report.
    - Calibrate-then-max from the float64 entry arrays (E) or from the full matrices (I).
* Trend machinery (spec section 3), exact relabelling enumeration (11! = 39,916,800; r linear in the LM row
  sums), the ARH pear / delta_boot arithmetic of analysis/s12_layermax_llama/early/aggregate_llama.py,
  Clopper-Pearson (spec section 9).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import sys
from fractions import Fraction

import numpy as np

_PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)          # hrhv2.util.rng_from_key (shared library, read-only import)

M_ITEMS = 1024
KNN_K = 10
MK = M_ITEMS * KNN_K          # 10,240: mKNN counts are out of m*k
K_PERM = 500
ALPHA = 0.05
ORDER_IDX = int(math.ceil((1 - Fraction(repr(ALPHA))) * (K_PERM + 1)))   # 476 (hrhv2 calibrate.order_index)
METRICS = ("mknn", "cka_biased", "cka_debiased")
SUBSETS = ("full", "fixed_r0.5", "fixed_r1.0", "grid2", "grid3", "grid5", "grid9", "grid17",
           "LMall_ViTlast", "LMlast_ViTall", "noLMemb", "LMdeep", "LMemb_only")
I_KEYS = {"mknn": ("obs_cnt", "null_cnt"), "cka_biased": ("obs_cka_b", "null_cka_b"),
          "cka_debiased": ("obs_cka_d", "null_cka_d")}
TOL_P = 1e-9                  # spec section 2: relabelling counts and |Tr| = 1 tests


# ----------------------------------------------------------------------------------------------------------
# layer subsets (identical to analysis/s12_layermax_llama/early/lm_core.py::subsets; checked against the
# layers_A / layers_B stored in every E summary)
# ----------------------------------------------------------------------------------------------------------

def fixed_layer(L: int, r: float) -> int:
    return int(math.floor(r * (L - 1) + 0.5))


def grid_layers(L: int, n: int) -> list:
    if n <= 1:
        return [L - 1]
    return sorted({fixed_layer(L, j / (n - 1)) for j in range(n)})


def subsets(LA: int, LB: int) -> dict:
    allA, allB = list(range(LA)), list(range(LB))
    out = {"full": (allA, allB)}
    for n in (2, 3, 5, 9, 17):
        out[f"grid{n}"] = (grid_layers(LA, n), grid_layers(LB, n))
    out["fixed_r0.5"] = ([fixed_layer(LA, 0.5)], [fixed_layer(LB, 0.5)])
    out["fixed_r1.0"] = ([LA - 1], [LB - 1])
    out["LMall_ViTlast"] = (allA, [LB - 1])
    out["LMlast_ViTall"] = ([LA - 1], allB)
    out["noLMemb"] = (list(range(1, LA)), allB)
    out["LMdeep"] = ([a for a in range(LA) if a / (LA - 1) >= 0.5], allB)
    out["LMemb_only"] = ([0], allB)
    return out


# ----------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(a: np.ndarray) -> str:
    """sha256 of an integer count array in a canonical dtype (V-O4b: counts must be identical)."""
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest() + str(a.shape)


def sgn(a, b) -> int:
    """sign(b - a) with exact comparison (Fractions or floats); sign(0) = 0."""
    return (b > a) - (b < a)


def fnum(v):
    """JSON value of a score: float(Fraction) for mKNN (spec section 2), float otherwise, None stays None."""
    if v is None:
        return None
    return float(v)


def cp_bounds(x: int, n: int):
    """Clopper-Pearson 95% (spec section 9)."""
    from scipy.stats import beta
    if n == 0:
        return None, None
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    return lo, hi


def median(vals):
    """numpy median of float64 values (mKNN: of the float(Fraction) values), spec section 9."""
    return float(np.median(np.asarray([float(v) for v in vals], dtype=np.float64)))


# ----------------------------------------------------------------------------------------------------------
# per-subset statistics
# ----------------------------------------------------------------------------------------------------------

def _first_argmax(sub: np.ndarray, ia: list, ib: list):
    flat = sub.reshape(-1)
    j = int(np.argmax(flat))                       # first maximum in row-major order (lower LM layer first)
    return ia[j // len(ib)], ib[j % len(ib)]


def mknn_subset(obs_cnt: np.ndarray, null_cnt: np.ndarray, ia: list, ib: list) -> dict:
    """Exact layer-max statistics of one subset from integer counts."""
    sub = obs_cnt[np.ix_(ia, ib)]
    T = int(sub.max())
    aA, aB = _first_argmax(sub, ia, ib)
    nm = null_cnt[:, ia][:, :, ib].reshape(null_cnt.shape[0], -1).max(axis=1).astype(np.int64)
    K = nm.size
    srt = np.sort(np.append(nm, T))
    c = int(srt[ORDER_IDX - 1])
    Tf, cf, muf = Fraction(T, MK), Fraction(c, MK), Fraction(int(nm.sum()), K * MK)
    scal = max((Tf - cf) / (1 - cf), Fraction(0)) if cf < 1 else Fraction(0)
    return {"T": Tf, "c": cf, "mu": muf, "s_cal": scal, "unc": Tf - muf,
            "p": float((1.0 + np.sum(nm >= T)) / (K + 1)), "aA": aA, "aB": aB, "T_count": T, "c_count": c}


def float_from_nullmax(T: float, null_max: np.ndarray) -> dict:
    """hrhv2 calibrate.summarize arithmetic on float64 layer-max nulls."""
    null = np.ascontiguousarray(np.asarray(null_max, dtype=np.float64))
    comb = np.sort(np.concatenate(([float(T)], null)))
    c = float(comb[ORDER_IDX - 1])
    denom = 1.0 - c
    scal = max((T - c) / denom, 0.0) if denom > 0 else 0.0
    mu = float(null.mean())
    return {"T": float(T), "c": c, "mu": mu, "s_cal": float(scal), "unc": float(T - mu),
            "p": float((1.0 + np.sum(null >= T)) / (null.size + 1))}


def float_from_summary(T: float, c: float, mu: float, p: float) -> dict:
    denom = 1.0 - c
    scal = max((T - c) / denom, 0.0) if denom > 0 else 0.0
    return {"T": float(T), "c": float(c), "mu": float(mu), "s_cal": float(scal), "unc": float(T - mu), "p": float(p)}


# ----------------------------------------------------------------------------------------------------------
# calibrate-then-max (plan section 3 (iv)); first maximum in row-major order
# ----------------------------------------------------------------------------------------------------------

def ctm_mknn(obs_cnt: np.ndarray, null_cnt: np.ndarray) -> dict:
    LA, LB = obs_cnt.shape
    K = null_cnt.shape[0]
    allc = np.concatenate([obs_cnt[None].astype(np.int64), null_cnt.astype(np.int64)], axis=0)
    c_ent = np.sort(allc, axis=0)[ORDER_IDX - 1]                       # entry c_alpha as counts
    s_ent = null_cnt.astype(np.int64).sum(axis=0)                     # entry null sums (mean = s/K)
    best, bi = None, (0, 0)
    best_u, best_mu = None, None
    for a in range(LA):
        for b in range(LB):
            T, c = int(obs_cnt[a, b]), int(c_ent[a, b])
            v = Fraction(max(T - c, 0), MK - c) if c < MK else Fraction(0)
            if best is None or v > best:
                best, bi = v, (a, b)
            u = K * T - int(s_ent[a, b])                              # (T - mu) * K * MK, an integer
            if best_u is None or u > best_u:
                best_u = u
            mu_ab = int(s_ent[a, b])
            if best_mu is None or mu_ab > best_mu:
                best_mu = mu_ab
    a, b = bi
    return {"M": best, "M_unc": Fraction(best_u, K * MK), "TM": Fraction(int(obs_cnt[a, b]), MK),
            "cM": Fraction(int(c_ent[a, b]), MK), "aA": a, "aB": b,
            "entry_mu_max": Fraction(best_mu, K * MK), "c_entry": c_ent, "s_entry": s_ent}


def ctm_float(obs: np.ndarray, c: np.ndarray, nm: np.ndarray) -> dict:
    den = 1.0 - c
    scal = np.where(den > 0, np.maximum((obs - c) / np.where(den > 0, den, 1.0), 0.0), 0.0)
    unc = obs - nm
    j = int(np.argmax(scal.reshape(-1)))
    LB = obs.shape[1]
    a, b = j // LB, j % LB
    return {"M": float(scal.reshape(-1)[j]), "M_unc": float(unc.max()), "TM": float(obs[a, b]),
            "cM": float(c[a, b]), "aA": a, "aB": b, "entry_mu_max": float(nm.max())}


# ----------------------------------------------------------------------------------------------------------
# pair readers
# ----------------------------------------------------------------------------------------------------------

def _finish_pair(ps: dict, LA: int, LB: int):
    ps["L_A"], ps["L_B"] = int(LA), int(LB)
    return ps


def read_pair_I(npz_path: str, protocols=SUBSETS, want_ctm=True, metrics=METRICS) -> dict:
    """I format: full float64 observed / null matrices and integer counts."""
    z = np.load(npz_path)
    LA, LB = int(z["L_A"]), int(z["L_B"])
    subs = subsets(LA, LB)
    ps = {"format": "I", "checks": {}}
    for met in metrics:
        ko, kn = I_KEYS[met]
        obs, null = z[ko], z[kn]
        assert obs.shape == (LA, LB) and null.shape[1:] == (LA, LB), (npz_path, met, obs.shape, null.shape)
        d = {}
        if met == "mknn":
            obs_i = obs.astype(np.int64)
            null_i = null.astype(np.int64)
            for q in protocols:
                ia, ib = subs[q]
                d[q] = mknn_subset(obs_i, null_i, ia, ib)
            if want_ctm:
                d["ctm"] = ctm_mknn(obs_i, null_i)
                d["entry_mu_max"] = d["ctm"]["entry_mu_max"]
                d["c_entry_full_argmax"] = Fraction(int(d["ctm"]["c_entry"][d["full"]["aA"], d["full"]["aB"]]), MK) \
                    if "full" in d else None
            if want_ctm:
                d["_counts"] = (_digest(obs_i), _digest(null_i))
        else:
            obs = np.asarray(obs, dtype=np.float64)
            null = np.asarray(null, dtype=np.float64)
            for q in protocols:
                ia, ib = subs[q]
                sub = obs[np.ix_(ia, ib)]
                T = float(sub.max())
                aA, aB = _first_argmax(sub, ia, ib)
                nmx = null[:, ia][:, :, ib].reshape(null.shape[0], -1).max(axis=1)
                st = float_from_nullmax(T, nmx)
                st.update({"aA": aA, "aB": aB})
                d[q] = st
            if want_ctm:
                c_ent = np.sort(np.concatenate([obs[None], null], axis=0), axis=0)[ORDER_IDX - 1]
                nm_ent = null.mean(axis=0)
                d["ctm"] = ctm_float(obs, c_ent, nm_ent)
                d["entry_mu_max"] = float(nm_ent.max())
                d["c_entry_full_argmax"] = float(c_ent[d["full"]["aA"], d["full"]["aB"]]) if "full" in d else None
        ps[met] = d
    return _finish_pair(ps, LA, LB)


def read_pair_E(json_path: str, npz_path: str, protocols=SUBSETS, want_ctm=True, metrics=METRICS,
                allow_missing=False) -> dict:
    """E format: JSON summaries + float64 entry arrays / full-layer-max nulls + mKNN null counts + f32 CKA nulls.
    allow_missing: subsets absent from the JSON (frozen 8-LM files lack noLMemb/LMdeep/LMemb_only) are
    recomputed exactly for mKNN and left None for CKA (only used for frozen or test inputs)."""
    rec = json.load(open(json_path))
    z = np.load(npz_path)
    LA, LB = int(rec["L_A"]), int(rec["L_B"])
    subs = subsets(LA, LB)
    ps = {"format": "E", "lm": rec.get("lm"), "vit": rec.get("vit"), "perm_key": rec.get("perm_key"),
          "K": rec.get("K")}
    chk = {"n_layers_mismatch": 0, "n_T_mismatch": 0, "n_float_recomp_mismatch": 0, "n_c_f32_mismatch": 0,
           "max_rel_mu_f32": 0.0, "n_p_f32_mismatch": 0, "n_mknn_json_mismatch": 0, "n_ctm_json_mismatch": 0,
           "n_argmax_mismatch": 0, "missing_subsets": [], "n_mknn_entry_ulp": 0}
    for met in metrics:
        S = rec["summaries"][met]
        obs = np.asarray(z[f"{met}.entry_obs"], dtype=np.float64)
        assert obs.shape == (LA, LB), (json_path, met, obs.shape)
        d = {}
        if met == "mknn":
            obs_i = np.rint(obs * MK).astype(np.int64)
            # the stored float is count/10240 up to one ulp (GPU scalar division may multiply by the reciprocal);
            # the integer count is recovered exactly by rounding
            assert np.all(np.abs(obs * MK - obs_i) < 1e-6), json_path
            chk["n_mknn_entry_ulp"] = int(np.sum(obs_i.astype(np.float64) / MK != obs))
            # [R-op] robustness: an integer observed-count array, if the E copy stores one, must agree
            for k in ("mknn.obs_counts", "mknn.entry_obs_counts", "mknn.obs_count", "mknn.entry_counts"):
                if k in z.files:
                    assert np.array_equal(np.asarray(z[k]).astype(np.int64), obs_i), (json_path, k)
            nk = "mknn.null_counts_int16" if "mknn.null_counts_int16" in z.files else \
                [k for k in z.files if k.startswith("mknn.null_counts")][0]
            null_i = z[nk].astype(np.int64)
            for q in protocols:
                ia, ib = subs[q]
                st = mknn_subset(obs_i, null_i, ia, ib)
                d[q] = st
                if q not in S:
                    chk["missing_subsets"].append(f"{met}|{q}")
                    if not allow_missing:
                        raise KeyError(f"{json_path}: subset {q} missing")
                    continue
                s = S[q]
                if s.get("layers_A") is not None and (list(s["layers_A"]) != list(ia) or list(s["layers_B"]) != list(ib)):
                    chk["n_layers_mismatch"] += 1
                # stored float summary vs exact recomputation (count equality)
                if (round(s["s_obs"] * MK) != st["T_count"] or round(s["c_alpha"] * MK) != st["c_count"]
                        or s["p_value"] != st["p"] or abs(s["null_mean"] - float(st["mu"])) > 1e-12 * float(st["mu"])):
                    chk["n_mknn_json_mismatch"] += 1
                if (s["argmax_A"], s["argmax_B"]) != (st["aA"], st["aB"]):
                    chk["n_argmax_mismatch"] += 1
            if want_ctm:
                d["ctm"] = ctm_mknn(obs_i, null_i)
                d["entry_mu_max"] = d["ctm"]["entry_mu_max"]
                d["c_entry_full_argmax"] = Fraction(int(d["ctm"]["c_entry"][d["full"]["aA"], d["full"]["aB"]]), MK)
                if "calib_then_max" in rec:
                    e = rec["calib_then_max"][met]
                    if (abs(e["M"] - float(d["ctm"]["M"])) > 1e-15 or abs(e["M_unc"] - float(d["ctm"]["M_unc"])) > 1e-15):
                        chk["n_ctm_json_mismatch"] += 1
            if want_ctm:
                d["_counts"] = (_digest(obs_i), _digest(null_i))
        else:
            f32 = z[f"{met}.null_f32"] if f"{met}.null_f32" in z.files else None
            for q in protocols:
                ia, ib = subs[q]
                sub = obs[np.ix_(ia, ib)]
                T = float(sub.max())
                aA, aB = _first_argmax(sub, ia, ib)
                if q not in S:
                    chk["missing_subsets"].append(f"{met}|{q}")
                    if not allow_missing:
                        raise KeyError(f"{json_path}: subset {q} missing")
                    d[q] = None
                    continue
                s = S[q]
                if s.get("layers_A") is not None and (list(s["layers_A"]) != list(ia) or list(s["layers_B"]) != list(ib)):
                    chk["n_layers_mismatch"] += 1
                if T != s["s_obs"]:
                    chk["n_T_mismatch"] += 1
                if (s["argmax_A"], s["argmax_B"]) != (aA, aB):
                    chk["n_argmax_mismatch"] += 1
                if q == "full":
                    st = float_from_nullmax(T, z[f"{met}.full_null_max"])
                else:
                    st = float_from_summary(T, s["c_alpha"], s["null_mean"], s["p_value"])
                    if f32 is not None:          # verification of the stored float64 summary against f32 nulls
                        nmx = f32[:, ia][:, :, ib].reshape(f32.shape[0], -1).max(axis=1)
                        comb = np.sort(np.concatenate(([np.float32(T)], nmx)))
                        if comb[ORDER_IDX - 1] != np.float32(s["c_alpha"]):
                            chk["n_c_f32_mismatch"] += 1
                        mu32 = float(np.mean(nmx.astype(np.float64)))
                        den = max(abs(s["null_mean"]), 1e-300)
                        chk["max_rel_mu_f32"] = max(chk["max_rel_mu_f32"], abs(mu32 - s["null_mean"]) / den)
                        p32 = float((1.0 + np.sum(nmx >= np.float32(T))) / (nmx.size + 1))
                        if p32 != s["p_value"]:
                            chk["n_p_f32_mismatch"] += 1
                if (st["c"] != s["c_alpha"] or st["s_cal"] != s["s_cal"] or st["unc"] != s["uncensored"]
                        or st["p"] != s["p_value"] or st["mu"] != s["null_mean"]):
                    chk["n_float_recomp_mismatch"] += 1
                st.update({"aA": aA, "aB": aB})
                d[q] = st
            if want_ctm:
                c_ent = np.asarray(z[f"{met}.entry_c_alpha"], dtype=np.float64)
                nm_ent = np.asarray(z[f"{met}.entry_null_mean"], dtype=np.float64)
                d["ctm"] = ctm_float(obs, c_ent, nm_ent)
                d["entry_mu_max"] = float(nm_ent.max())
                d["c_entry_full_argmax"] = float(c_ent[d["full"]["aA"], d["full"]["aB"]])
                if "calib_then_max" in rec:
                    e = rec["calib_then_max"][met]
                    if (e["M"] != d["ctm"]["M"] or e["M_unc"] != d["ctm"]["M_unc"]
                            or e["s_obs_at_argmax"] != d["ctm"]["TM"] or e["c_alpha_at_argmax"] != d["ctm"]["cM"]
                            or (e["argmax_A"], e["argmax_B"]) != (d["ctm"]["aA"], d["ctm"]["aB"])):
                        chk["n_ctm_json_mismatch"] += 1
        ps[met] = d
    ps["checks"] = chk
    return _finish_pair(ps, LA, LB)


# ----------------------------------------------------------------------------------------------------------
# value access
# ----------------------------------------------------------------------------------------------------------

XMAP = {"raw": "T", "s_cal": "s_cal", "unc": "unc"}


def val(ps: dict, met: str, q: str, x: str):
    """Spec section 1 score X of protocol Q (exact Fraction for mKNN, float for CKA); None if unavailable."""
    d = ps[met]
    if q == "ctm":
        c = d.get("ctm")
        return None if c is None else c[x]
    st = d.get(q)
    if st is None:
        return None
    return st[XMAP[x]] if x in XMAP else st[x]


# ----------------------------------------------------------------------------------------------------------
# trends (spec section 3): Spearman between size ranks and scores, average ranks, exact relabellings
# ----------------------------------------------------------------------------------------------------------

def avg_ranks(vals) -> list:
    """Average ranks with ties only on exact equality (Fractions or float64)."""
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson_vec(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / np.sqrt((x @ x) * (y @ y)))


def _usable(row) -> bool:
    for v in row:
        if v is None:
            return False
        if not isinstance(v, Fraction) and not math.isfinite(float(v)):
            return False
    return not all(v == row[0] for v in row)


def trend_table(rows: list, n: int) -> dict:
    """rows: per ViT the n scores in size order. Returns Tr for every relabelling sigma of the size ranks
    (identity first; sigma assigns rank perm[i] to the i-th size), the per-ViT rho at the identity and the
    number of excluded ViTs (three equal scores or a non-finite one)."""
    perms = list(itertools.permutations(range(1, n + 1)))
    inc = [v for v, r in enumerate(rows) if _usable(r)]
    nconst = len(rows) - len(inc)
    yr = [avg_ranks(rows[v]) for v in inc]
    if not inc:
        return {"T": np.full(len(perms), np.nan), "nconst": nconst, "rho_id": [None] * len(rows), "perms": perms}
    R = np.array([[pearson_vec(p, y) for y in yr] for p in perms])      # (n!, n_inc)
    T = R.mean(axis=1)
    rho = [None] * len(rows)
    for k, v in enumerate(inc):
        rho[v] = float(R[0, k])
    return {"T": T, "nconst": nconst, "rho_id": rho, "perms": perms}


def p_counts(T: np.ndarray) -> dict:
    """spec section 3: tail counts over the relabellings, tolerance 1e-9 (identity = index 0)."""
    t = float(T[0])
    if not math.isfinite(t):
        return {"pge": None, "ple": None, "ptwo": None, "obs": None}
    return {"obs": t, "pge": float(np.mean(T >= t - TOL_P)), "ple": float(np.mean(T <= t + TOL_P)),
            "ptwo": float(np.mean(np.abs(T) >= abs(t) - TOL_P))}


# ----------------------------------------------------------------------------------------------------------
# ARH-style pooled r (spec section 6): aggregate_llama.py::pear / arh_test / delta_boot arithmetic
# ----------------------------------------------------------------------------------------------------------

def pear(G: np.ndarray, x: np.ndarray) -> float:
    xx = np.repeat(x[:, None], G.shape[1], axis=1).ravel()
    y = G.ravel()
    if np.nanstd(y) == 0 or not np.all(np.isfinite(y)):
        return float("nan")
    return float(np.corrcoef(xx, y)[0, 1])


def boot_r(G: np.ndarray, x: np.ndarray, key: str, nb: int = 2000):
    from hrhv2 import util
    rng = util.rng_from_key(key)
    bs = np.empty(nb)
    for b in range(nb):
        j = rng.integers(0, G.shape[1], G.shape[1])
        bs[b] = pear(G[:, j], x)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def delta_boot(G1b, G1a, G2b, G2a, x: np.ndarray, key: str, nb: int = 2000):
    """aggregate_llama.py::delta_boot with the key and the panel changed (plan R.3.6, spec section 6)."""
    from hrhv2 import util
    f = lambda a, b, c, d: (pear(a, x) - pear(b, x)) - (pear(c, x) - pear(d, x))  # noqa: E731
    est = f(G1b, G1a, G2b, G2a)
    rng = util.rng_from_key(key)
    nv = G1b.shape[1]
    bs = np.empty(nb)
    for b in range(nb):
        j = rng.integers(0, nv, nv)
        bs[b] = f(G1b[:, j], G1a[:, j], G2b[:, j], G2a[:, j])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def gap_boot(Gm_after, Gc_after, x: np.ndarray, key: str, nb: int = 2000):
    """spec section 6 [op]: CI of r_after(mKNN) - r_after(CKA), same resampling as delta_boot."""
    from hrhv2 import util
    rng = util.rng_from_key(key)
    nv = Gm_after.shape[1]
    bs = np.empty(nb)
    for b in range(nb):
        j = rng.integers(0, nv, nv)
        bs[b] = pear(Gm_after[:, j], x) - pear(Gc_after[:, j], x)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def linear_weights(G: np.ndarray, x: np.ndarray):
    """r(sigma) = sum_i xc[sigma(i)] * w_i with w = row sums of the centred grid / den (arh_test)."""
    y = G - G.mean()
    rows = y.sum(axis=1)
    den = np.sqrt(G.shape[1] * np.sum((x - x.mean()) ** 2)) * np.sqrt(np.sum(y ** 2))
    return rows / den


def enumerate_ge_counts(xc: np.ndarray, W: np.ndarray, thr: np.ndarray, head: int = 3) -> np.ndarray:
    """Exact count over all n! relabellings sigma (value xc[sigma(i)] given to LM i) of
    sum_i xc[sigma(i)] W[i, j] >= thr[j], for every column j. Chunked: the first `head` LMs take every ordered
    choice of values, the other n - head LMs every permutation of the rest (one GEMM per prefix)."""
    n = xc.size
    J = W.shape[1]
    tail = n - head
    perms_t = np.array(list(itertools.permutations(range(tail))), dtype=np.int64)       # (tail!, tail)
    Wt = np.ascontiguousarray(W[head:])                                                  # (tail, J)
    cnt = np.zeros(J, dtype=np.int64)
    total = 0
    for pre in itertools.permutations(range(n), head):
        rest = np.array([i for i in range(n) if i not in pre], dtype=np.int64)
        base = np.zeros(J)
        for k, i in enumerate(pre):
            base = base + xc[i] * W[k]
        V = xc[rest][perms_t]                                                            # (tail!, tail)
        S = V @ Wt + base[None, :]
        cnt += np.sum(S >= thr[None, :], axis=0)
        total += V.shape[0]
    assert total == math.factorial(n)
    return cnt
