#!/usr/bin/env python
"""OPENLLAMA-AXIS COPY of analysis/s12_layermax_llama/indep/lmx_llama_aggregate.py (sub-role I of
oll-axis-extract, S0b): aggregation of implementation I.

Reads (only after the Part B gate, plan section 4 F1(a)):
  results/S1_2_layermax_openllama/indep/<pre>/pairs/<lm>__<vit>.npz   OpenLLaMA 3b/7b/13b reruns (this run)
  results/S1_2_layermax/indep/<pre>/pairs/                              BLOOMZ x 5 and llama-7b (frozen 8-LM run)
                                                                        and the frozen 3b/7b files (V-O4b, G)
  results/S1_2_layermax_llama/indep/<pre>/pairs/                        llama-13b/30b (plan R.3.7)
  results/S1_2_layermax_llama/layermax_llama_summary.json               (db9242f9..., section G check)
  results/S1_2_layermax_openllama/partB/partB_features_ol.json          (frozen Part B; s_O, materiality,
                                                                         B-bar, B_late; never recomputed)
Writes:
  results/S1_2_layermax_openllama/indep/verdicts_ol.json   VERDICTS_SPEC.md format, every key of
                                                            verdict_keys.required_keys("EI") (sections A-D, G)
  results/S1_2_layermax_openllama/indep/summary.json       full detail (per-pair records, grids, rho
                                                            vectors, descriptive extras, provenance)
Plan items: section 3 protocols (i)-(iv); 5.1-5.3 (Tr_O with exact p over 3! relabellings, F_new,
ARH-style pooled r on 11 LMs with the exact 11! = 39,916,800 relabelling p, the 10- and 8-LM panels,
the dDelta ViT-bootstrap CI = LLaMA E delta_boot with key S1.2-layermax-openllama|boot|arh_delta|<pre>|<metric>);
5.6 curves; 6 diagnostics (D1-D3, plug-in); 7 readings; 9 P-O1, P-O2, P-O2a, P-O3, P-O4, P-O5, P-O5a
with their deductive checks (P-O2b = forecast_ol.py, J = joint_summary.py). F2: mKNN in exact
rationals (fractions.Fraction on the integer counts out of m k = 10,240).
The verdict arithmetic lives in ol_verdicts.py (new file). The LLaMA copy's Part B verdicts P-L1..P-L5
(PARTB_PREDICTIONS.md format), joint Tr_BL and 10! Pooled class are replaced.

usage: launch_ol.sh ol_I_aggregate 4 -- venv/bin/python analysis/s12_layermax_openllama/indep/lmx_ol_aggregate.py
       [--sens-e2]   S-E2 (plan section 6, only if E2 is not bitwise): 3b/7b from indep/sens_e2/<pre>/pairs/,
                     outputs indep/sens_e2/{summary.json, verdicts_ol.json}
       [--no-exact]  skip the 11! enumeration (debug only; the pge keys are then null)
CPU only, <= 4 threads (plan section 12: S8 aggregate). Runtime about 5-10 min.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ol_gate  # noqa: E402
OUT_ROOT = ol_gate.gate(__file__)   # F1(a): exits 3 unless Part B is frozen (S0b test mode: scratch root)

_T = os.environ.get("OL_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from fractions import Fraction  # noqa: E402

import numpy as np  # noqa: E402

import lmx_common as L  # noqa: E402
from lmx_common import C, cal  # noqa: E402
import ol_verdicts as OV  # noqa: E402

sys.path.insert(0, os.path.join(L.PROJ, "analysis", "s12"))
import aggregate as S12  # noqa: E402  (S1.2 trend helpers boot_vits / vision_trend, read-only import)
sys.path.insert(0, ol_gate.AXIS)
import verdict_keys as VK  # noqa: E402  (required key lists, read-only import)

HERE = os.path.dirname(os.path.abspath(__file__))
PARTB = os.path.join(os.path.dirname(C.CLIP_PARTB), "partB_features_ol.json")
PARTB_MD = os.path.join(ol_gate.AXIS, "PARTB_PREDICTIONS_OL.md")
MERGED = os.path.join(L.PROJ, "results", "S1_2_layermax_llama", "layermax_llama_summary.json")
PLAN = os.path.join(ol_gate.AXIS, "ANALYSIS_PLAN_OPENLLAMA.md")
METRICS = OV.METRICS
NPZ = {"mknn": ("obs_cnt", "null_cnt"), "cka_biased": ("obs_cka_b", "null_cka_b"),
       "cka_debiased": ("obs_cka_d", "null_cka_d")}
I_MET = {"mknn": "mknn", "cka_biased": "cka_b", "cka_debiased": "cka_d"}         # LLaMA I record names
I_SUB = {"LMall_ViTlast": "lmall_x_vitlast", "LMlast_ViTall": "lmlast_x_vitall"}
NV = len(L.VIT_STEMS)
OL = list(L.LM_STEMS)
POOLED = list(L.POOLED_LM_STEMS)
BLOOMZ = [s for s, f in C.LMS_POOLED if f == "BLOOMZ"]
LLAMA_AX = ("huggyllama__llama-13b", "huggyllama__llama-30b")
K = L.K_PERM
MK = L.M * L.KNN
ORDER_IDX = cal.order_index(L.K_PERM, L.ALPHA) - 1          # 475: the 476th ascending order statistic
PRIMARY = "prh_clip+l2"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- layer subsets
def fixed_layer(n: int, r: float) -> int:
    return int(math.floor(r * (n - 1) + 0.5))


def grid_layers(n: int, k: int) -> list:
    return sorted({fixed_layer(n, j / (k - 1)) for j in range(k)})


def subsets(LA, LB):
    """exactly analysis/s12_layermax_llama/early/lm_core.py::subsets (spec section 1)."""
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


# ----------------------------------------------------------------------------- per pair records
def _summ(met, Ss, Ns):
    """layer-max summary of one subset. mKNN: exact Fractions from the integer counts."""
    if met == "mknn":
        T = int(Ss.max())
        Tk = Ns.reshape(Ns.shape[0], -1).max(axis=1).astype(np.int64)
        c = int(np.sort(np.append(Tk, T))[ORDER_IDX])
        s_cal = Fraction(T - c, MK - c) if (MK - c > 0 and T > c) else Fraction(0)
        mu = Fraction(int(Tk.sum()), K * MK)
        return {"raw": Fraction(T, MK), "c": Fraction(c, MK), "s_cal": s_cal, "mu": mu, "unc": Fraction(T, MK) - mu,
                "p": (1.0 + float(np.sum(Tk >= T))) / (K + 1), "null_sd": float(np.std(Tk / MK, ddof=1))}
    T = float(Ss.max())
    Tk = Ns.reshape(Ns.shape[0], -1).max(axis=1)
    sm = L.summarize(T, Tk)
    return {"raw": sm["s_obs"], "c": sm["c_alpha"], "s_cal": sm["s_cal"], "mu": sm["null_mean"],
            "unc": sm["uncensored"], "p": sm["p_value"], "null_sd": sm["null_sd"]}


def _entry_c(met, S, N, a, b):
    if met == "mknn":
        ce = int(np.sort(np.append(N[:, a, b], S[a, b]))[ORDER_IDX])
        return Fraction(ce, MK), Fraction(int(N[:, a, b].sum()), K * MK)
    sm = L.summarize(float(S[a, b]), N[:, a, b])
    return sm["c_alpha"], sm["null_mean"]


def _ctm(met, S, N):
    """(iv) calibrate-then-max: every entry against its own c_alpha,uv from the same 500 pi."""
    Ce = np.sort(np.concatenate([S[None], N], axis=0), axis=0)[ORDER_IDX]
    LA, LB = S.shape
    if met == "mknn":
        best, ba, bb = Fraction(0), 0, 0
        vals = {}
        for (u, w), sv in np.ndenumerate(S):
            cu = int(Ce[u, w])
            q = Fraction(int(sv) - cu, MK - cu) if (MK - cu > 0 and int(sv) > cu) else Fraction(0)
            vals[(u, w)] = q
            if q > best:
                best, ba, bb = q, u, w
        U = K * S.astype(np.int64) - N.sum(axis=0)
        ua, ub = np.unravel_index(int(np.argmax(U)), U.shape)
        return {"M": best, "M_unc": Fraction(int(U.max()), K * MK), "TM": Fraction(int(S[ba, bb]), MK),
                "cM": Fraction(int(Ce[ba, bb]), MK), "a": int(ba), "b": int(bb), "unc_a": int(ua), "unc_b": int(ub),
                "rel_a": ba / (LA - 1), "n_entries_scal_pos": int(sum(1 for q in vals.values() if q > 0)),
                "n_tied_max": int(sum(1 for q in vals.values() if q == best))}
    den = 1.0 - Ce
    scal = np.where(den > 0, np.maximum((S - Ce) / np.where(den > 0, den, 1.0), 0.0), 0.0)
    unc = S - N.mean(axis=0)
    ia, ib = np.unravel_index(int(np.argmax(scal)), scal.shape)
    ua, ub = np.unravel_index(int(np.argmax(unc)), unc.shape)
    return {"M": float(scal.max()), "M_unc": float(unc.max()), "TM": float(S[ia, ib]), "cM": float(Ce[ia, ib]),
            "a": int(ia), "b": int(ib), "unc_a": int(ua), "unc_b": int(ub), "rel_a": ia / (LA - 1),
            "n_entries_scal_pos": int(np.sum(scal > 0)), "n_tied_max": int(np.sum(scal == scal.max()))}


def pair_record(z, detail=True):
    rec = {}
    for met in METRICS:
        ko, kn = NPZ[met]
        if met == "mknn":
            S, N = z[ko].astype(np.int64), z[kn].astype(np.int64)
        else:
            S, N = np.asarray(z[ko], dtype=np.float64), np.asarray(z[kn], dtype=np.float64)
        LA, LB = S.shape
        subs = subsets(LA, LB) if detail else {"full": (list(range(LA)), list(range(LB)))}
        r = {}
        for q, (ra, rb) in subs.items():
            ra, rb = np.asarray(ra), np.asarray(rb)
            Ss = S[np.ix_(ra, rb)]
            Ns = N[:, ra][:, :, rb]
            d = _summ(met, Ss, Ns)
            ia, ib = np.unravel_index(int(np.argmax(Ss)), Ss.shape)
            d.update({"a": int(ra[ia]), "b": int(rb[ib]), "n_entries": int(Ss.size),
                      "n_tied_max": int(np.sum(Ss == Ss.max())), "LA": LA, "LB": LB})
            r[q] = d
        a, b = r["full"]["a"], r["full"]["b"]
        r["full"]["c_entry"], r["full"]["mu_entry"] = _entry_c(met, S, N, a, b)
        if met == "mknn":
            r["full"]["max_entry_mu"] = Fraction(int(N.sum(axis=0).max()), K * MK)
        else:
            r["full"]["max_entry_mu"] = float(N.mean(axis=0).max())
        if detail:
            r["ctm"] = _ctm(met, S, N)
        rec[met] = r
    return rec


def load_z(base, pre, lm, vit):
    z = np.load(os.path.join(base, pre, "pairs", f"{lm}__{vit}.npz"))
    assert int(z["K"]) == K and str(z["key"]) == L.perm_key(lm, vit), (base, lm, vit, str(z["key"]))
    return z


def jrec(o):
    """JSON form of a record (Fractions -> float)."""
    if isinstance(o, dict):
        return {str(k): jrec(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jrec(v) for v in o]
    if isinstance(o, Fraction):
        return float(o)
    return L.jsonable(o)


# ----------------------------------------------------------------------------- section G helpers
def rel_ok(a, b, tol=1e-12):
    a, b = float(a), float(b)
    return abs(a - b) <= tol * max(abs(a), abs(b))


def g_merged(recs, merged):
    """G|llama_pairs_equal_merged_summary: llama-13b/30b records from this implementation's LLaMA-axis
    pair files vs the indep summary embedded in layermax_llama_summary.json (every subset, ctm M/M_unc)."""
    emb = merged["embedded"]["indep_summary"]["per_pre"]
    n = bad = 0
    worst = 0.0
    fails = []
    for pre in L.PRES:
        for lm in LLAMA_AX:
            for vit in L.VIT_STEMS:
                Ej = emb[pre]["pairs_llama"][f"{lm}|{vit}"]
                for met in METRICS:
                    for q in OV.SUBSETS:
                        e = Ej[I_MET[met]][I_SUB.get(q, q)]
                        mine = recs[pre][lm][vit][met][q]
                        for fm, fe in (("raw", "s_obs"), ("c", "c_alpha"), ("s_cal", "s_cal"), ("unc", "uncensored"),
                                       ("mu", "null_mean")):
                            n += 1
                            a, b = float(mine[fm]), float(e[fe])
                            if max(abs(a), abs(b)) > 0:
                                worst = max(worst, abs(a - b) / max(abs(a), abs(b)))
                            if not rel_ok(a, b):
                                bad += 1
                                if len(fails) < 20:
                                    fails.append([pre, lm, vit, met, q, fm, a, b])
                        n += 1
                        if float(mine["p"]) != float(e["p_value"]):
                            bad += 1
                    ec = Ej[I_MET[met]]["ctm"]
                    for fm, fe in (("M", "s_cal"), ("M_unc", "uncensored")):
                        n += 1
                        if not rel_ok(recs[pre][lm][vit][met]["ctm"][fm], ec[fe]):
                            bad += 1
    return {"n_values": n, "n_differ": bad, "max_rel": worst, "fails": fails, "pass": bad == 0}


def g_frozen(pre_list, ol_dir_of):
    """V-O4b / G|rerun_equals_frozen_3b7b: rerun 3b/7b pair files vs results/S1_2_layermax/indep (frozen):
    CKA matrices <= 1e-12 relative, observed and null mKNN counts identical, and every subset's p identical
    and floats <= 1e-12 relative."""
    out = {"max_rel_arrays": 0.0, "counts_identical": True, "p_identical": True, "max_rel_summary": 0.0, "n_pairs": 0}
    recs_frozen = {}
    for pre in pre_list:
        recs_frozen[pre] = {}
        for lm in OL[:2]:
            recs_frozen[pre][lm] = {}
            for vit in L.VIT_STEMS:
                z = load_z(ol_dir_of(lm), pre, lm, vit)
                zo = load_z(L.OLD_INDEP, pre, lm, vit)
                out["n_pairs"] += 1
                for k in ("obs_cka_b", "obs_cka_d", "null_cka_b", "null_cka_d"):
                    a, b = np.asarray(z[k], np.float64), np.asarray(zo[k], np.float64)
                    if a.shape != b.shape:
                        out["max_rel_arrays"] = float("inf")
                        continue
                    d = np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
                    out["max_rel_arrays"] = max(out["max_rel_arrays"], float(d.max()))
                out["counts_identical"] &= bool(
                    z["obs_cnt"].shape == zo["obs_cnt"].shape and z["null_cnt"].shape == zo["null_cnt"].shape
                    and np.array_equal(z["obs_cnt"].astype(np.int64), zo["obs_cnt"].astype(np.int64))
                    and np.array_equal(z["null_cnt"].astype(np.int64), zo["null_cnt"].astype(np.int64)))
                rf = pair_record(zo, detail=True)
                recs_frozen[pre][lm][vit] = rf
                mine = REC_CACHE[pre][lm][vit]
                for met in METRICS:
                    for q in OV.SUBSETS:
                        for fld in ("raw", "c", "s_cal", "unc", "mu"):
                            a, b = float(mine[met][q][fld]), float(rf[met][q][fld])
                            if max(abs(a), abs(b)) > 0:
                                out["max_rel_summary"] = max(out["max_rel_summary"], abs(a - b) / max(abs(a), abs(b)))
                        out["p_identical"] &= bool(mine[met][q]["p"] == rf[met][q]["p"])
    out["pass"] = bool(out["counts_identical"] and out["p_identical"] and out["max_rel_arrays"] <= 1e-12
                       and out["max_rel_summary"] <= 1e-12 and out["n_pairs"] == 17 * 2 * len(pre_list))
    return out, recs_frozen


REC_CACHE = {}


# ----------------------------------------------------------------------------- main
def build_grids(recs_pooled):
    grids = {}
    for pre in L.PRES:
        for met in METRICS:
            for x in ("raw", "s_cal", "unc"):
                grids[(pre, met, x)] = np.array([[float(recs_pooled[pre][lm][v][met]["full"][x]) for v in L.VIT_STEMS]
                                                 for lm in POOLED], dtype=np.float64)
    return grids


def panels_and_covs():
    i13 = POOLED.index(C.SLUG13)
    p8 = BLOOMZ + [C.SLUG3, C.SLUG7, "huggyllama__llama-7b"]
    panels = {"11": np.arange(len(POOLED)), "10": np.array([i for i in range(len(POOLED)) if i != i13]),
              "8": np.array([POOLED.index(s) for s in p8])}
    covs = {}
    for pan, idx in panels.items():
        stems = [POOLED[i] for i in idx]
        arh = np.array([L.ARH_RANK[s] for s in stems], dtype=np.float64)
        covs[(pan, "arh")] = arh
        covs[(pan, "dense")] = np.argsort(np.argsort(arh)).astype(np.float64) + 1.0
        covs[(pan, "logp")] = np.log(np.array([C.n_params(s) for s in stems], dtype=np.float64))
    return panels, covs


def all_verdicts(recs_ol, recs_pooled, pb, extra, exact=True):
    grids = build_grids(recs_pooled)
    panels, covs = panels_and_covs()
    VD, info = OV.pooled_section(grids, panels, covs, log=log, exact=exact)
    V = OV.verdicts(recs_ol, OL, list(L.VIT_STEMS), pb, info, extra)
    V.update(VD)
    return V, info, grids, panels, covs


def main():
    ap = argparse.ArgumentParser(description="implementation I aggregate (VERDICTS_SPEC sections A-D, G)")
    ap.add_argument("--sens-e2", action="store_true", help="S-E2 sensitivity (plan section 6)")
    ap.add_argument("--no-exact", action="store_true", help="debug: skip the 11! enumeration")
    args = ap.parse_args()
    assert os.path.realpath(L.OUT) == os.path.realpath(OUT_ROOT)
    t0 = time.time()
    outdir = L.OUT_SENS if args.sens_e2 else L.OUT
    os.makedirs(outdir, exist_ok=True)
    pb = json.load(open(PARTB))
    merged = json.load(open(MERGED))
    assert list(S12.C.VITS) == list(C.VITS), "ViT order differs between the S1.2 registry and the copy"
    assert list(pb["vits"]) == list(L.VIT_STEMS), "Part B ViT order differs"
    assert list(pb["ol_lms"]) == OL, ("Part B OpenLLaMA slugs differ", pb["ol_lms"], OL)

    def src(lm):
        if lm in OL:
            return L.OUT_SENS if (args.sens_e2 and lm in OL[:2]) else L.OUT
        if lm in LLAMA_AX:
            return L.LLAMA_INDEP
        return L.OLD_INDEP
    recs_pooled, n_read = {}, 0
    for pre in L.PRES:
        recs_pooled[pre] = {}
        for lm in POOLED:
            recs_pooled[pre][lm] = {}
            detail = lm in OL or lm in LLAMA_AX
            for vit in L.VIT_STEMS:
                recs_pooled[pre][lm][vit] = pair_record(load_z(src(lm), pre, lm, vit), detail=detail)
                n_read += 1
        log(f"{pre}: {len(POOLED) * NV} pair records built ({time.time() - t0:.0f}s)")
    REC_CACHE.update(recs_pooled)
    recs_ol = {pre: {lm: recs_pooled[pre][lm] for lm in OL} for pre in L.PRES}
    extra = {}
    V, info, grids, panels, covs = all_verdicts(recs_ol, recs_pooled, pb, extra, exact=not args.no_exact)
    TR = extra.pop("_TR")
    rho = {k: v for k, v in list(extra.items()) if k.startswith("x|rho|")}
    for k in rho:
        extra.pop(k)
    log(f"sections A-D done ({time.time() - t0:.0f}s)")
    # ---- section G
    gm = g_merged(recs_pooled, merged)
    gf, recs_frozen = g_frozen(L.PRES, src)
    V["G|llama_pairs_equal_merged_summary"] = bool(gm["pass"])
    V["G|rerun_equals_frozen_3b7b"] = bool(gf["pass"])
    V["G|n_pairs_read"] = int(n_read)
    log(f"G: merged {gm['pass']} (max rel {gm['max_rel']:.1e}), rerun=frozen {gf['pass']} "
        f"(arrays {gf['max_rel_arrays']:.1e}, summaries {gf['max_rel_summary']:.1e})")
    frozen_block = None
    if not gf["pass"]:
        # plan section 3 source rule: every OpenLLaMA-axis verdict also on the frozen 3b/7b values
        rp = {pre: dict(recs_pooled[pre]) for pre in L.PRES}
        for pre in L.PRES:
            for lm in OL[:2]:
                rp[pre][lm] = recs_frozen[pre][lm]
        ro = {pre: {lm: rp[pre][lm] for lm in OL} for pre in L.PRES}
        ex2 = {}
        V2, _, _, _, _ = all_verdicts(ro, rp, pb, ex2, exact=not args.no_exact)
        for k, v in V2.items():
            V[f"x|frozen|{k}"] = v
        frozen_block = {"n_keys": len(V2)}
    # ---- descriptive extras (x|): ViT bootstrap of Tr (key S1.2|bootvit|layermax-openllama|<stat>), vision axis
    for (p, m, q, x), t in TR.items():
        if q in ("full", "fixed_r0.5", "fixed_r1.0", "ctm") and t["obs"] is not None:
            vec = np.array([np.nan if r is None else r for r in t["rho"]], dtype=np.float64)
            _, lo, hi = S12.boot_vits(vec, f"layermax-openllama|{p}|{m}|{q}|{x}")
            extra[f"x|tr_ci_lo|{p}|{m}|{q}|{x}"], extra[f"x|tr_ci_hi|{p}|{m}|{q}|{x}"] = lo, hi
    for p in L.PRES:
        for m in METRICS:
            for x in ("raw", "s_cal", "unc"):
                G = np.array([[float(recs_ol[p][lm][v][m]["full"][x]) for v in L.VIT_STEMS] for lm in OL])
                vt = S12.vision_trend(G)
                extra[f"x|vision_tr|{p}|{m}|{x}"] = vt["Tr"]
                extra[f"x|vision_p_one|{p}|{m}|{x}"] = vt["p_one"]
        for m in ("cka_biased", "cka_debiased"):
            db = info["dd_boot"][(p, m)]
            extra[f"x|dd_boot_est|{p}|{m}"] = db["est"]
    extra["x|n_relabel_11"] = info.get("n_relabel")
    extra["x|exact_secs"] = info.get("exact_secs")
    # ---- required keys present?
    req = VK.required_keys("EI")
    missing = [k for k in req if k not in V]
    vout = {k: (v if isinstance(v, str) else OV.f(v)) for k, v in {**V, **extra}.items()}
    for k, v in vout.items():
        assert v is None or isinstance(v, (int, float, str, bool)), (k, type(v))
    # ---- inputs (sha256)
    def digest(base, lms, pres=L.PRES):
        h = hashlib.sha256()
        for pre in pres:
            for lm in lms:
                for vit in L.VIT_STEMS:
                    fn = os.path.join(base, pre, "pairs", f"{lm}__{vit}.npz")
                    h.update(f"{pre}/pairs/{lm}__{vit}.npz {sha(fn)}\n".encode())
        return h.hexdigest()
    ol_main = [lm for lm in OL if src(lm) == L.OUT]
    inputs = {os.path.relpath(PARTB, L.PROJ): sha(PARTB), os.path.relpath(MERGED, L.PROJ): sha(MERGED),
              os.path.relpath(PLAN, L.PROJ): sha(PLAN),
              os.path.relpath(L.OUT, L.PROJ) + f"/<pre>/pairs/{{{','.join(ol_main)}}} (digest)": digest(L.OUT, ol_main),
              os.path.relpath(L.OLD_INDEP, L.PROJ) + "/<pre>/pairs/{BLOOMZ x5, llama-7b, open_llama_3b/7b} (digest)":
                  digest(L.OLD_INDEP, BLOOMZ + ["huggyllama__llama-7b"] + OL[:2]),
              os.path.relpath(L.LLAMA_INDEP, L.PROJ) + "/<pre>/pairs/{llama-13b, llama-30b} (digest)":
                  digest(L.LLAMA_INDEP, list(LLAMA_AX))}
    if args.sens_e2:
        inputs[os.path.relpath(L.OUT_SENS, L.PROJ) + "/<pre>/pairs/{3b,7b} (digest)"] = digest(L.OUT_SENS, OL[:2])
    if os.path.exists(PARTB_MD):
        inputs[os.path.relpath(PARTB_MD, L.PROJ)] = sha(PARTB_MD)
    for fn in ("ol_verdicts.py", "lmx_ol_aggregate.py", "lmx_common.py", "common_ol.py", "ol_gate.py"):
        inputs[os.path.relpath(os.path.join(HERE, fn), L.PROJ)] = sha(os.path.join(HERE, fn))
    verd = {"schema": "OL-VERDICTS v1", "impl": "I", "inputs": inputs, "v": vout}
    # ---- summary.json (full detail)
    summary = {
        "role": "oll-axis I (implementation I aggregate)", "implementation": "I", "sens_e2": bool(args.sens_e2),
        "test_mode": bool(ol_gate.test_mode()),
        "plan": "analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md", "spec": "VERDICTS_SPEC.md (S0b v1)",
        "K": K, "alpha": L.ALPHA, "ol_order": OL, "pooled_order": POOLED, "vit_order": list(L.VIT_STEMS),
        "arh_rank": {s: L.ARH_RANK[s] for s in POOLED}, "num_params": {s: C.n_params(s) for s in POOLED},
        "missing_required_keys": missing, "n_required_keys": len(req),
        "verdicts": vout,
        "rho_per_vit": {k[len("x|rho|"):]: v for k, v in rho.items()},
        "G": {"merged": gm, "rerun_equals_frozen": gf, "frozen_rule_applied": frozen_block},
        "pooled": {"grids": {f"{p}|{m}|{x}": g.tolist() for (p, m, x), g in grids.items()},
                   "panels": {k: v.tolist() for k, v in panels.items()},
                   "covariates": {f"{a}|{b}": v.tolist() for (a, b), v in covs.items()},
                   "dd_boot": {f"{p}|{m}": v for (p, m), v in info["dd_boot"].items()},
                   "n_relabel_11": info.get("n_relabel")},
        "pairs_ol": {pre: {f"{lm}|{v}": jrec(recs_ol[pre][lm][v]) for lm in OL for v in L.VIT_STEMS} for pre in L.PRES},
        "inputs": inputs,
    }
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["secs"] = time.time() - t0
    for name, obj in (("summary.json", summary), ("verdicts_ol.json", verd)):
        fn = os.path.join(outdir, name)
        with open(fn + ".tmp", "w") as fh:
            json.dump(obj, fh, indent=1, default=lambda o: float(o) if isinstance(o, Fraction) else L.jsonable(o))
        os.replace(fn + ".tmp", fn)
    log(f"wrote {outdir}/summary.json and verdicts_ol.json: {len(vout)} keys, {len(missing)} required keys missing "
        f"({time.time() - t0:.0f}s)")
    if missing:
        log(f"MISSING required keys (first 20): {missing[:20]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
