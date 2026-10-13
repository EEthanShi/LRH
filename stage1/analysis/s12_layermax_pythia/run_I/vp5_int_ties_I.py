#!/usr/bin/env python
"""Role pyt-run-I: V-P5 supplement, exact ties in integer arithmetic, and the merged record indep/validation_cross.json.

NOT frozen code (written by the run role after CODE FROZEN; it changes no frozen file and decides no verdict). It runs
only AFTER I's summary.json / verdicts_pyt.json are final and after the frozen indep/cross_validate.py (V-P5, I side)
has run. It calls I's order gate first (plan F1(a)), records the order (I outputs final before any E file is opened),
and PRINTS AND WRITES COUNTS AND KEY NAMES ONLY, NEVER VALUES (plan V-P5, critique B6.4).

S1 mKNN, integer arithmetic, both sides recomputed from their own raw integer arrays (I: indep/<pre>/pairs/*.npz
   obs_cnt, null_cnt; E: early/pairs/<pre>/*.npz mknn.entry_obs x 10240 rounded, mknn.null_counts_int16):
   raw arrays identical; full layer-max T (int), number of tied maxima, first row-major argmax, the 500 null maxima,
   c_alpha (476th ascending order statistic of {T} u {T_k}, int), p numerator 1 + #{T_k >= T}, #{T_k == T};
   calibrate-then-max with entry thresholds (ints) and s_cal = max((S - c)/(10240 - c), 0) as Fractions: M, its tie
   count and first row-major argmax. E-derived == I-derived exactly; E's reported full p, tie count and argmax equal
   the integers (p as float(Fraction(num, 501)) exactly); E's reported T and c_alpha equal float(Fraction(., 10240)).
S2 CKA (biased, debiased), full layer-max: from I's raw float64 arrays, tie count of the maximum (exact float64
   equality), first argmax, p numerator 1 + #{T_k >= T} and #{T_k == T} (ints); E's reported p x 501 must be an
   integer (|.| <= 1e-9) equal to I's numerator; E's tie count and argmax equal I's.
S3 verdict level: every p-type key (first or last |-field in: tr_pge, tr_ple, tr_ptwo, dtr_pge, dtr_ple, dtr_ptwo, p_perm, p_mag, p_lo, p_hi,
   p, p_tost, p_holm) present and numeric on both sides: the integer numerator count = p x N for the first N in
   (5040, 720, 120, 2001, 501, 667) at which both sides are integral within 1e-9; the integers must be equal; a key
   integral on neither or only one side is listed by name.
usage (project root, through launch_pyt.sh, <= 2 threads, CPU only):
  venv/bin/python -B analysis/s12_layermax_pythia/run_I/vp5_int_ties_I.py --pres prh_clip+l2,l2 \
      --frozen-cross results/S1_2_layermax_pythia/indep/validation_cross_1-2-3_prh_clip+l2+l2.json
Output: results/S1_2_layermax_pythia/indep/validation_cross.json (refuses to overwrite).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
PROJ = os.path.realpath(os.path.expanduser("~/hrh_v2_stage1_20261008"))
INDEP_CODE = os.path.join(PROJ, "analysis", "s12_layermax_pythia", "indep")
sys.path.insert(0, INDEP_CODE)
import pyt_gate  # noqa: E402  (frozen I gate; read-only import)
OUT_ROOT = pyt_gate.gate(__file__)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from fractions import Fraction  # noqa: E402

import numpy as np  # noqa: E402

RES = os.path.join(PROJ, "results", "S1_2_layermax_pythia")
E_DIR = os.path.join(RES, "early")
SIZES = ("70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b")
LMS = tuple(f"EleutherAI__pythia-{s}" for s in SIZES)
VITS = ("vit_tiny_patch16_224.augreg_in21k", "vit_small_patch16_224.augreg_in21k", "vit_base_patch16_224.augreg_in21k",
        "vit_large_patch16_224.augreg_in21k", "vit_base_patch16_224.mae", "vit_large_patch16_224.mae",
        "vit_huge_patch14_224.mae", "vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m",
        "vit_large_patch14_dinov2.lvd142m", "vit_giant_patch14_dinov2.lvd142m", "vit_base_patch16_clip_224.laion2b",
        "vit_large_patch14_clip_224.laion2b", "vit_huge_patch14_clip_224.laion2b",
        "vit_base_patch16_clip_224.laion2b_ft_in12k", "vit_large_patch14_clip_224.laion2b_ft_in12k",
        "vit_huge_patch14_clip_224.laion2b_ft_in12k")
K, MK, IDX = 500, 1024 * 10, 475
P_FIELDS = {"tr_pge", "tr_ple", "tr_ptwo", "dtr_pge", "dtr_ple", "dtr_ptwo", "p_perm", "p_mag", "p_lo", "p_hi", "p",
            "p_tost", "p_holm"}
P_DENOMS = (5040, 720, 120, 2001, 501, 667)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Cnt:
    def __init__(self):
        self.d = {}

    def add(self, key, ok, where):
        r = self.d.setdefault(key, {"n": 0, "n_fail": 0, "fail_where": []})
        r["n"] += 1
        if not ok:
            r["n_fail"] += 1
            if len(r["fail_where"]) < 20:
                r["fail_where"].append(where)

    def summary(self):
        return {"checks": self.d, "n_checks": sum(v["n"] for v in self.d.values()),
                "n_fail": sum(v["n_fail"] for v in self.d.values())}


def mknn_int(S, N):
    """S (LA, LB) int64 observed counts, N (K, LA, LB) int64 null counts -> integer/Fraction quantities."""
    T = int(S.max())
    nt = int((S == T).sum())
    ia, ib = (int(x) for x in np.unravel_index(int(np.argmax(S)), S.shape))
    Nk = N.reshape(N.shape[0], -1).max(axis=1)
    c = int(np.sort(np.append(Nk, T))[IDX])
    num = 1 + int((Nk >= T).sum())
    nnt = int((Nk == T).sum())
    Ce = np.sort(np.concatenate([S[None], N], axis=0), axis=0)[IDX]
    best, bt, ba = None, 0, None
    for a in range(S.shape[0]):
        for b in range(S.shape[1]):
            ce = int(Ce[a, b])
            v = max(Fraction(int(S[a, b]) - ce, MK - ce), Fraction(0))
            if best is None or v > best:
                best, bt, ba = v, 1, (a, b)
            elif v == best:
                bt += 1
    return {"T": T, "nt": nt, "arg": (ia, ib), "c": c, "num": num, "nnt": nnt, "M": best, "M_ties": bt, "M_arg": ba}


def float_max_int(S, N):
    T = float(S.max())
    nt = int((S == T).sum())
    ia, ib = (int(x) for x in np.unravel_index(int(np.argmax(S)), S.shape))
    Nk = N.reshape(N.shape[0], -1).max(axis=1)
    return {"nt": nt, "arg": (ia, ib), "num": 1 + int((Nk >= T).sum()), "nnt": int((Nk == T).sum())}


def as_int(x, tol=1e-9):
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(float(x)):
        return None
    r = round(float(x))
    return int(r) if abs(float(x) - r) <= tol else None


def pair_level(pres, cnt_m, cnt_c, missing):
    for pre in pres:
        for lm in LMS:
            for vit in VITS:
                where = f"{pre}|{lm}|{vit}"
                fi = os.path.join(OUT_ROOT, pre, "pairs", f"{lm}__{vit}.npz")
                fj = os.path.join(E_DIR, "pairs", pre, f"{lm}__{vit}.json")
                fz = os.path.join(E_DIR, "pairs", pre, f"{lm}__{vit}.npz")
                if not (os.path.exists(fi) and os.path.exists(fj) and os.path.exists(fz)):
                    missing.append(where)
                    continue
                zi, ze, Ej = np.load(fi), np.load(fz), json.load(open(fj))
                # ---- S1 mKNN, both sides from raw integers
                Si, Ni = zi["obs_cnt"].astype(np.int64), zi["null_cnt"].astype(np.int64)
                have_e = "mknn.null_counts_int16" in ze.files and "mknn.entry_obs" in ze.files
                cnt_m.add("E_raw_mknn_arrays_present", have_e, where)
                if have_e:
                    eo = np.asarray(ze["mknn.entry_obs"], dtype=np.float64) * MK
                    Se = np.rint(eo).astype(np.int64)
                    cnt_m.add("E_obs_counts_integral", bool(np.all(np.abs(eo - Se) <= 1e-6)), where)
                    Ne = ze["mknn.null_counts_int16"].astype(np.int64)
                    cnt_m.add("obs_counts_identical", Se.shape == Si.shape and np.array_equal(Se, Si), where)
                    cnt_m.add("null_counts_identical", Ne.shape == Ni.shape and np.array_equal(Ne, Ni), where)
                    qi, qe = mknn_int(Si, Ni), mknn_int(Se, Ne)
                    for k in ("T", "nt", "arg", "c", "num", "nnt", "M", "M_ties", "M_arg"):
                        cnt_m.add(f"full_or_ctm.{k}.E_raw_eq_I_raw", qi[k] == qe[k], where)
                else:
                    qi = mknn_int(Si, Ni)
                Es = (Ej.get("summaries") or {}).get("mknn", {}).get("full") or {}
                cnt_m.add("E_full.p_value_eq_num_over_501_exact", Es.get("p_value") == float(Fraction(qi["num"], 501)), where)
                cnt_m.add("E_full.p_numerator_int_eq", as_int((Es.get("p_value") or float("nan")) * 501) == qi["num"], where)
                cnt_m.add("E_full.n_tied_max_eq", Es.get("n_tied_max") == qi["nt"], where)
                cnt_m.add("E_full.argmax_eq", (Es.get("argmax_A"), Es.get("argmax_B")) == qi["arg"], where)
                cnt_m.add("E_full.s_obs_eq_T_over_mk_exact", Es.get("s_obs") == float(Fraction(qi["T"], MK)), where)
                cnt_m.add("E_full.c_alpha_eq_c_over_mk_exact", Es.get("c_alpha") == float(Fraction(qi["c"], MK)), where)
                Ec = (Ej.get("calib_then_max") or {}).get("mknn") or {}
                cnt_m.add("E_ctm.argmax_eq", (Ec.get("argmax_A"), Ec.get("argmax_B")) == qi["M_arg"], where)
                cnt_m.add("E_ctm.n_tied_max_eq", Ec.get("n_tied_max") == qi["M_ties"], where)
                cnt_m.add("E_ctm.M_eq_float_fraction_exact", Ec.get("M") == float(qi["M"]), where)
                # ---- S2 CKA, integer tie counts and p numerators (full)
                for met, ko, kn in (("cka_biased", "obs_cka_b", "null_cka_b"), ("cka_debiased", "obs_cka_d", "null_cka_d")):
                    q = float_max_int(np.asarray(zi[ko], dtype=np.float64), np.asarray(zi[kn], dtype=np.float64))
                    Es = (Ej.get("summaries") or {}).get(met, {}).get("full") or {}
                    ne = as_int((Es.get("p_value") if isinstance(Es.get("p_value"), (int, float)) else float("nan")) * 501)
                    cnt_c.add(f"{met}.E_p_x501_integral", ne is not None, where)
                    cnt_c.add(f"{met}.p_numerator_int_eq", ne == q["num"], where)
                    cnt_c.add(f"{met}.n_tied_max_eq", Es.get("n_tied_max") == q["nt"], where)
                    cnt_c.add(f"{met}.argmax_eq", (Es.get("argmax_A"), Es.get("argmax_B")) == q["arg"], where)


def last_field(key):
    parts = key.split("|")
    while parts and parts[-1].startswith("m="):
        parts = parts[:-1]
    return parts[-1] if parts else ""


def verdict_int(i_path, e_path):
    out = {"I": i_path, "E": e_path}
    if not (os.path.exists(i_path) and os.path.exists(e_path)):
        out.update(status="missing", n_fail=None, pass_=False)
        return out
    vi, ve = json.load(open(i_path))["v"], json.load(open(e_path))["v"]
    out.update(I_sha256=sha(i_path), E_sha256=sha(e_path))
    n, nfail, fail, nonint, both_null = 0, 0, [], [], 0
    per_denom = {}
    for k in sorted(set(vi) & set(ve)):
        if k.startswith("x|") or (last_field(k) not in P_FIELDS and k.split("|")[0] not in P_FIELDS):
            continue
        a, b = vi[k], ve[k]
        if a is None and b is None:
            both_null += 1
            continue
        n += 1
        hit = None
        for N in P_DENOMS:
            ia, ib = as_int(a * N) if isinstance(a, (int, float)) and not isinstance(a, bool) else None, \
                as_int(b * N) if isinstance(b, (int, float)) and not isinstance(b, bool) else None
            if ia is not None and ib is not None:
                hit = (N, ia, ib)
                break
        if hit is None:
            nonint.append(k)
            nfail += 1
            continue
        per_denom[str(hit[0])] = per_denom.get(str(hit[0]), 0) + 1
        if hit[1] != hit[2]:
            nfail += 1
            fail.append(k)
    out.update(status="compared", n_p_keys_compared=n, n_both_null=both_null, n_by_denominator=per_denom,
               n_fail=nfail, mismatching_keys=fail[:200], keys_not_integral_on_both_sides=nonint[:200],
               pass_=nfail == 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pres", default="prh_clip+l2,l2")
    ap.add_argument("--frozen-cross", required=True, help="output json of the frozen indep/cross_validate.py")
    args = ap.parse_args()
    pres = tuple(args.pres.split(","))
    dst = os.path.join(OUT_ROOT, "validation_cross.json")
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists (no overwrite)")
    i_files = [os.path.join(OUT_ROOT, n) for n in ("summary.json", "verdicts_pyt.json")]
    for p in i_files:
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing: I outputs must be final before any E file is opened")
    order = {"I_final": {os.path.relpath(p, PROJ): {"sha256": sha(p), "mtime_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(p)))} for p in i_files},
        "E_first_opened_by_this_script_after_utc": utc()}
    fc = os.path.join(PROJ, args.frozen_cross) if not os.path.isabs(args.frozen_cross) else args.frozen_cross
    frozen = json.load(open(fc))
    cnt_m, cnt_c, missing = Cnt(), Cnt(), []
    t0 = time.time()
    pair_level(pres, cnt_m, cnt_c, missing)
    s1, s2 = cnt_m.summary(), cnt_c.summary()
    s3 = verdict_int(os.path.join(OUT_ROOT, "verdicts_pyt.json"), os.path.join(E_DIR, "verdicts_pyt.json"))
    fI = os.path.join(RES, "forecast", "I", "verdicts_forecast.json")
    fE = os.path.join(RES, "forecast", "E", "verdicts_forecast.json")
    s3fk = verdict_int(fI, fE) if (os.path.exists(fI) and os.path.exists(fE)) else {"status": "forecast files absent"}
    p_frozen = bool(frozen.get("pass_V-P5_selected_parts"))
    p_sup = (not missing) and s1["n_fail"] == 0 and s2["n_fail"] == 0 and bool(s3.get("pass_"))
    res = {
        "what": "V-P5 (E vs I), I side: the frozen cross_validate.py record plus the run role's integer-arithmetic tie "
                "supplement; counts and key names only, never values",
        "role": "pyt-run-I", "utc": utc(), "pres": list(pres), "order": order,
        "frozen_cross_validate": {"file": os.path.relpath(fc, PROJ), "sha256": sha(fc), "record": frozen,
                                  "pass": p_frozen},
        "supplement_int_ties": {"script": os.path.relpath(os.path.abspath(__file__), PROJ),
                                "script_sha256": sha(os.path.abspath(__file__)),
                                "n_missing_pairs": len(missing), "missing_pairs": missing[:50],
                                "S1_mknn_integer": s1, "S2_cka_integer_ties_and_p_numerators": s2,
                                "S3_verdict_p_numerators_EI": s3, "S3_verdict_p_numerators_FK": s3fk,
                                "pass": bool(p_sup), "secs": time.time() - t0},
        "pass_V-P5": bool(p_frozen and p_sup),
    }
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1, default=str)
    os.replace(tmp, dst)
    print(f"[vp5_int_ties_I] {utc()} written {dst} sha256 {sha(dst)}; frozen cross_validate pass={p_frozen}; "
          f"supplement: missing pairs {len(missing)}, S1 {s1['n_checks']} checks {s1['n_fail']} fail, "
          f"S2 {s2['n_checks']} checks {s2['n_fail']} fail, S3 EI {s3.get('n_p_keys_compared')} p keys "
          f"{s3.get('n_fail')} fail; pass_V-P5={res['pass_V-P5']} (names only in the file)", flush=True)


if __name__ == "__main__":
    main()
