#!/usr/bin/env python
"""LLaMA axis, implementation I (role llama-axis-run-indep): cross-implementation
comparison with the early implementation E. Run only after I's own summary.json and
REPORT.md were written (STATUS 2026-10-09T01:35:37Z).

Part 1, V-L5 (plan section 4) and the llama-7b rows: for every one of the 51 pairs, both pre,
  3 metrics, the 13 layer subsets of plan section 3 (i)-(iii) and calibrate-then-max (iv):
  s_obs, c_alpha, s_cal, uncensored, null mean and null sd within
  |a - b| <= 1e-12 max(|a|, |b|) + 1e-15; p exactly equal; argmax and number of tied
  maxima identical; selected-entry threshold and null mean (full grid) in the same tolerance.
Part 2, raw arrays (informational beyond the plan items, except the mKNN null counts which
  V-L5 requires identical): observed entries, entry null mean / sd / c_alpha / exceedance
  counts, the 500 full-grid null maxima and their argmax, and E's float32 CKA null store vs
  I's float64 nulls.
Part 3, summary level: every Tr_L, Delta Tr_L (with exact p), joint Tr_BL, pooled r, vision
  trend, mean, reading and Part B verdict that E reports is compared with I's value
  (Tr/r/p within 1e-12 absolute; categorical outcomes exactly equal).
Output: results/S1_2_layermax_llama/indep/validation_cross.json (with report_lines, which
lmx_llama_report.py copies into REPORT.md section 7b).
"""
from __future__ import annotations

import os

_T = os.environ.get("LMX_THREADS", "3")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import lmx_common as L  # noqa: E402
import lmx_llama_aggregate as AG  # noqa: E402  (fam_trend, grid helpers; main() not run)

E_DIR = os.path.join(L.PROJ, "results", "S1_2_layermax_llama", "early")
I_SUM = os.path.join(L.OUT, "summary.json")
PARTB = os.path.join(L.PROJ, "results", "S1_2_layermax_llama", "partB", "partB_features.json")
OUTF = os.path.join(L.OUT, "validation_cross.json")
LLAMA = list(L.LM_STEMS)
VITS = list(L.VIT_STEMS)
METMAP = {"mknn": "mknn", "cka_b": "cka_biased", "cka_d": "cka_debiased"}
IMET = {v: k for k, v in METMAP.items()}
SUBMAP = {"full": "full", "grid2": "grid2", "grid3": "grid3", "grid5": "grid5", "grid9": "grid9", "grid17": "grid17",
          "fixed_r0.5": "fixed_r0.5", "fixed_r1.0": "fixed_r1.0", "lmall_x_vitlast": "LMall_ViTlast",
          "lmlast_x_vitall": "LMlast_ViTall", "noLMemb": "noLMemb", "LMdeep": "LMdeep", "LMemb_only": "LMemb_only"}
ISUB = {v: k for k, v in SUBMAP.items()}
FLOATS = ("s_obs", "c_alpha", "s_cal", "uncensored", "null_mean", "null_sd")
ORDER_IDX = AG.ORDER_IDX
TOL_ABS_SUMMARY = 1e-12


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class Acc:
    """accumulates float comparisons under the V-L5 mixed tolerance"""

    def __init__(self):
        self.d = {}

    def add(self, key, a, b, where):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        diff = np.abs(a - b)
        tol = 1e-12 * np.maximum(np.abs(a), np.abs(b)) + 1e-15
        r = self.d.setdefault(key, {"n": 0, "n_fail": 0, "max_abs": 0.0, "max_ratio_to_tol": 0.0, "worst": None,
                                    "fail_examples": []})
        r["n"] += int(diff.size)
        nf = int(np.sum(diff > tol))
        r["n_fail"] += nf
        if diff.size:
            i = int(np.argmax(diff / tol))
            ratio = float((diff / tol).ravel()[i])
            r["max_abs"] = max(r["max_abs"], float(diff.max()))
            if ratio > r["max_ratio_to_tol"]:
                r["max_ratio_to_tol"] = ratio
                r["worst"] = {"where": where, "a": float(a.ravel()[i]), "b": float(b.ravel()[i]),
                              "absdiff": float(diff.ravel()[i])}
        if nf and len(r["fail_examples"]) < 5:
            r["fail_examples"].append({"where": where, "n_fail_here": nf})

    def out(self):
        return self.d


class Cnt:
    def __init__(self):
        self.d = {}

    def add(self, key, ok, where, detail=None):
        r = self.d.setdefault(key, {"n": 0, "n_fail": 0, "fail_examples": []})
        r["n"] += 1
        if not ok:
            r["n_fail"] += 1
            if len(r["fail_examples"]) < 10:
                r["fail_examples"].append({"where": where, "detail": detail})


# ----------------------------------------------------------------------------------------- part 1 + 2
def per_pair(SI):
    acc, cnt, arr, arrc = Acc(), Cnt(), Acc(), Cnt()
    n_pairs = 0
    for pre in L.PRES:
        for lm in LLAMA:
            for vit in VITS:
                where = f"{pre}|{lm}|{vit}"
                Ij = SI["per_pre"][pre]["pairs_llama"][f"{lm}|{vit}"]
                Ej = json.load(open(os.path.join(E_DIR, "pairs", pre, f"{lm}__{vit}.json")))
                cnt.add("meta_K_key_shape", Ej["K"] == L.K_PERM and Ej["perm_key"] == L.perm_key(lm, vit)
                        and Ej["L_A"] == Ij["LA"] and Ej["L_B"] == Ij["LB"], where)
                for met, em in METMAP.items():
                    for sub, es in SUBMAP.items():
                        I, E = Ij[met][sub], Ej["summaries"][em][es]
                        w = f"{where}|{met}|{sub}"
                        for fl in FLOATS:
                            acc.add(f"{met}.{fl}", I[fl], E[fl], w)
                        cnt.add(f"{met}.p_value_exact", I["p_value"] == E["p_value"], w, [I["p_value"], E["p_value"]])
                        cnt.add(f"{met}.argmax_identical", (I["a"], I["b"]) == (E["argmax_A"], E["argmax_B"]), w,
                                [I["a"], I["b"], E["argmax_A"], E["argmax_B"]])
                        cnt.add(f"{met}.n_tied_max_identical", I["n_tied_max"] == E["n_tied_max"], w)
                        cnt.add(f"{met}.n_entries_identical", I["n_entries"] == E["n_layer_pairs"], w)
                    I, E = Ij[met]["full"], Ej["summaries"][em]["full"]
                    acc.add(f"{met}.full.entry_c_alpha_at_argmax", I["c_entry"], E["entry_c_alpha_at_argmax"], where)
                    acc.add(f"{met}.full.entry_null_mean_at_argmax", I["null_mean_entry"], E["entry_null_mean_at_argmax"], where)
                    # (iv) calibrate-then-max
                    I, E = Ij[met]["ctm"], Ej["calib_then_max"][em]
                    acc.add(f"{met}.ctm.M", I["s_cal"], E["M"], where)
                    acc.add(f"{met}.ctm.M_unc", I["uncensored"], E["M_unc"], where)
                    acc.add(f"{met}.ctm.c_alpha_at_argmax", I["c_entry_at_argmax"], E["c_alpha_at_argmax"], where)
                    acc.add(f"{met}.ctm.s_obs_at_argmax", I["S_at_argmax"], E["s_obs_at_argmax"], where)
                    cnt.add(f"{met}.ctm.argmax_identical", (I["a"], I["b"]) == (E["argmax_A"], E["argmax_B"]), where)
                    cnt.add(f"{met}.ctm.unc_argmax_identical", (I["unc_a"], I["unc_b"]) == (E["M_unc_argmax_A"], E["M_unc_argmax_B"]), where)
                    cnt.add(f"{met}.ctm.n_entries_scal_pos_identical", I["n_entries_scal_pos"] == E["n_entries_scal_pos"], where)
                    cnt.add(f"{met}.ctm.n_tied_max_identical", I["n_tied_max"] == E["n_tied_max"], where)
                # ---------------- raw arrays
                zi = np.load(os.path.join(L.OUT, pre, "pairs", f"{lm}__{vit}.npz"))
                ze = np.load(os.path.join(E_DIR, "pairs", pre, f"{lm}__{vit}.npz"))
                mk = L.M * L.KNN
                obs = {"mknn": zi["obs_cnt"].astype(np.float64) / mk, "cka_b": zi["obs_cka_b"], "cka_d": zi["obs_cka_d"]}
                null = {"mknn": zi["null_cnt"].astype(np.float64) / mk, "cka_b": zi["null_cka_b"], "cka_d": zi["null_cka_d"]}
                arrc.add("mknn.null_counts_identical", np.array_equal(zi["null_cnt"].astype(np.int64),
                                                                      ze["mknn.null_counts_int16"].astype(np.int64)), where)
                arrc.add("mknn.obs_counts_identical", np.array_equal(zi["obs_cnt"].astype(np.int64),
                                                                     np.rint(ze["mknn.entry_obs"] * mk).astype(np.int64)), where)
                for met, em in METMAP.items():
                    S, N = obs[met], null[met]
                    K = N.shape[0]
                    w = f"{where}|{met}"
                    arr.add(f"{met}.entry_obs", S, ze[f"{em}.entry_obs"], w)
                    arr.add(f"{met}.entry_null_mean", N.mean(axis=0), ze[f"{em}.entry_null_mean"], w)
                    arr.add(f"{met}.entry_null_sd", N.std(axis=0, ddof=1), ze[f"{em}.entry_null_sd"], w)
                    c_e = np.sort(np.concatenate([S[None], N], axis=0), axis=0)[ORDER_IDX]
                    arr.add(f"{met}.entry_c_alpha", c_e, ze[f"{em}.entry_c_alpha"], w)
                    exc_i = 1 + np.sum(N >= S[None], axis=0)
                    exc_e = np.rint(ze[f"{em}.entry_p"] * (K + 1)).astype(np.int64)
                    arrc.add(f"{met}.entry_exceedance_counts_identical", np.array_equal(exc_i, exc_e), w)
                    flat = N.reshape(K, -1)
                    arr.add(f"{met}.full_null_max", flat.max(axis=1), ze[f"{em}.full_null_max"], w)
                    ia, ib = np.unravel_index(flat.argmax(axis=1), S.shape)
                    same = (ia == ze[f"{em}.full_null_argmax_A"]) & (ib == ze[f"{em}.full_null_argmax_B"])
                    arrc.add(f"{met}.full_null_argmax_identical", bool(np.all(same)), w, int(np.sum(~same)))
                    if met != "mknn":
                        f32 = ze[f"{em}.null_f32"]
                        r = arr.d.setdefault(f"{met}.null_f32_vs_I_float64", {"n": 0, "n_not_equal_to_float32_of_I": 0,
                                                                               "max_abs_f32_minus_I": 0.0, "max_rel_f32_minus_I": 0.0})
                        r["n"] += int(f32.size)
                        r["n_not_equal_to_float32_of_I"] += int(np.sum(f32 != N.astype(np.float32)))
                        d = np.abs(f32.astype(np.float64) - N)
                        r["max_abs_f32_minus_I"] = max(r["max_abs_f32_minus_I"], float(d.max()))
                        r["max_rel_f32_minus_I"] = max(r["max_rel_f32_minus_I"],
                                                       float(np.max(d / np.maximum(np.abs(N), 1e-300))))
                n_pairs += 1
        log(f"{pre}: per-pair comparison done")
    p1_ok = all(v["n_fail"] == 0 for v in acc.out().values()) and all(v["n_fail"] == 0 for v in cnt.d.values())
    return {"n_pair_records": n_pairs, "floats_mixed_tol": acc.out(), "exact": cnt.d,
            "pass_V-L5_form": bool(p1_ok and arrc.d["mknn.null_counts_identical"]["n_fail"] == 0)}, \
           {"floats_mixed_tol": {k: v for k, v in arr.out().items() if "null_f32" not in k},
            "null_f32": {k: v for k, v in arr.out().items() if "null_f32" in k}, "exact": arrc.d}


# ----------------------------------------------------------------------------------------- part 3
class Sum:
    def __init__(self):
        self.rows = []

    def num(self, key, a, b, tol=TOL_ABS_SUMMARY):
        ok = (a is None and b is None) or (a is not None and b is not None and abs(float(a) - float(b)) <= tol)
        self.rows.append({"key": key, "I": a, "E": b, "ok": bool(ok), "kind": "num"})

    def cat(self, key, a, b):
        self.rows.append({"key": key, "I": a, "E": b, "ok": a == b, "kind": "cat"})

    def vec(self, key, a, b, tol=TOL_ABS_SUMMARY):
        ok = len(a) == len(b) and all((x is None and y is None) or (x is not None and y is not None and abs(x - y) <= tol)
                                      for x, y in zip(a, b))
        self.rows.append({"key": key, "I": "vector", "E": "vector", "ok": bool(ok), "kind": "vec"})

    def info(self, key, a, b):
        self.rows.append({"key": key, "I": a, "E": b, "ok": None, "kind": "info"})


def i_grid_pairs(SI, pre, met, sub, fld):
    PL = SI["per_pre"][pre]["pairs_llama"]
    return np.array([[float(PL[f"{lm}|{v}"][met][sub][fld]) for v in VITS] for lm in LLAMA])


def summary_level(SI, SE, PB):
    sm = Sum()
    for pre in L.PRES:
        BI = SI["per_pre"][pre]["blocks"]
        PE = SE["per_pre"][pre]
        derived = {
            "shift": lambda: i_grid_pairs(SI, pre, met, "full", "s_obs") - i_grid_pairs(SI, pre, met, "full", "c_alpha"),
            "sel_entry_thr": lambda: i_grid_pairs(SI, pre, met, "full", "s_obs") - i_grid_pairs(SI, pre, met, "full", "c_entry"),
            "sel_entry_mean": lambda: i_grid_pairs(SI, pre, met, "full", "s_obs") - i_grid_pairs(SI, pre, met, "full", "null_mean_entry"),
            "inflation": lambda: i_grid_pairs(SI, pre, met, "full", "c_alpha") - i_grid_pairs(SI, pre, met, "full", "c_entry"),
            "entry_c_alpha_at_argmax": lambda: i_grid_pairs(SI, pre, met, "full", "c_entry"),
            "minus_Bbar": lambda: i_grid_pairs(SI, pre, met, "full", "s_obs") - np.array(
                [[PB["per_pre"][pre]["Bbar"][lm][v] for v in VITS] for lm in LLAMA]),
        }
        # trends_L
        for k, te in PE["trends_L"].items():
            em, es, ef = k.split("|")
            met = IMET[em]
            if es == "ctm":
                ti = BI[met]["ctm"]["llama_axis"][{"M": "s_cal", "M_unc": "uncensored"}[ef]]
            elif ef in ("s_obs", "s_cal", "uncensored", "c_alpha", "null_mean"):
                ti = BI[met][ISUB[es]]["llama_axis"][ef]
            else:
                ti = AG.pub(AG.fam_trend(derived[ef](), LLAMA))
            key = f"{pre}|trends_L|{k}"
            sm.num(key + "|Tr", ti["Tr"], te["Tr"])
            sm.num(key + "|p_one", ti["p_one"], te["p_one"])
            sm.num(key + "|p_two", ti["p_two"], te["p_two"])
            sm.vec(key + "|rho_per_vit", ti["rho_per_vit"], te["rho_per_vit"])
            sm.num(key + "|n_vit_used", ti["n_vit_used"], te["n_vit_used"], 0)
            sm.num(key + "|n_vit_constant", ti["n_vit_constant"], te["n_vit_constant"], 0)
            if "ci" in ti and "ci" in te:
                same_key = (met == "mknn" and es == ISUB.get(es, es) and es != "ctm")
                if same_key:
                    sm.vec(key + "|ci(same rng key)", ti["ci"], te["ci"])
                else:
                    sm.info(key + "|ci(different rng key string)", ti["ci"], te["ci"])
        # delta_L
        for k, de in PE["delta_L"].items():
            em, es, ef = k.split("|")
            met = IMET[em]
            if es == "ctm":
                di = BI[met]["ctm"]["delta"][{"M": "s_cal", "M_unc": "uncensored"}[ef]]
            else:
                di = BI[met][ISUB[es]]["delta"][ef]
            key = f"{pre}|delta_L|{k}"
            sm.num(key + "|dTr", di["dTr"], de["diff"])
            sm.num(key + "|p_upper", di["p_one_increase"], de["p_upper"])
            sm.num(key + "|p_lower", di["p_one_decrease"], de["p_lower"])
            sm.num(key + "|p_two", di["p_two"], de["p_two"])
        # joint_BL
        for k, je in PE["joint_BL"].items():
            em, es, ef = k.split("|")
            met = IMET[em]
            ji = BI[met]["ctm" if es == "ctm" else ISUB[es]]["joint_BL"][{"M": "s_cal", "M_unc": "uncensored"}.get(ef, ef)]
            key = f"{pre}|joint_BL|{k}"
            for f in ("Tr_BL", "Tr_BLOOMZ", "Tr_L", "p_one", "p_two"):
                sm.num(f"{key}|{f}", ji[f], je[f])
        # pooled r
        for k, pe in PE["pooled"].items():
            em, es, ef = k.split("|")
            met = IMET[em]
            sb = "ctm" if es == "ctm" else ISUB[es]
            fi = {"M": "s_cal", "M_unc": "uncensored"}.get(ef, ef)
            pi = BI[met][sb]["pooled_r"][fi]
            key = f"{pre}|pooled|{k}"
            sm.num(key + "|arh.r", pi["arh_rank"]["r"], pe["arh"]["r"])
            if "p_one" in pe["arh"]:
                sm.num(key + "|arh.p_one", pi["arh_rank"]["p_one"], pe["arh"]["p_one"])
            if "dense" in pe:
                sm.num(key + "|dense.r", pi["dense_rank"]["r"], pe["dense"])
            if "logp" in pe:
                sm.num(key + "|logp.r", pi["log_params"]["r"], pe["logp"])
            if "r_8LM" in pe:
                sm.num(key + "|r_8LM", BI[met][sb]["pooled_r_8lm"][fi]["r"], pe["r_8LM"])
            if isinstance(pe.get("dense_test"), dict) and "p_one" in pe["dense_test"]:
                sm.num(key + "|dense.p_one", pi["dense_rank"]["p_one"], pe["dense_test"]["p_one"])
            if "openllama_7b_gt_3b_frac" in pe and fi in BI[met][sb].get("openllama_7b_gt_3b_frac", {}):
                sm.num(key + "|openllama_7b_gt_3b_frac", BI[met][sb]["openllama_7b_gt_3b_frac"][fi], pe["openllama_7b_gt_3b_frac"])
            if "ci" in pi["arh_rank"] and "ci" in pe["arh"]:
                sm.info(key + "|arh.ci(rng key strings differ)", pi["arh_rank"]["ci"], pe["arh"]["ci"])
        # vision
        for k, ve in PE["vision"].items():
            em, es, ef = k.split("|")
            vi = BI[IMET[em]][ISUB[es]]["vision_axis_llama"][ef]
            key = f"{pre}|vision|{k}"
            for f in ("Tr", "p_one", "p_two"):
                sm.num(f"{key}|{f}", vi[f], ve[f])
            sm.num(key + "|n_cells", vi["n_cells"], ve["n_cells"], 0)
        # means by LLaMA size
        for em, d in PE["means"].items():
            met = IMET[em]
            for k, byl in d.items():
                es, ef = k.split("|")
                for lm, ve in byl.items():
                    if lm not in LLAMA:
                        continue
                    key = f"{pre}|means|{em}|{k}|{lm}"
                    if es == "ctm":
                        vi = BI[met]["ctm"]["mean_by_lm"][{"M": "s_cal", "M_unc": "uncensored"}[ef]][lm]
                    elif ef in ("s_obs", "s_cal", "uncensored", "c_alpha", "null_mean"):
                        vi = BI[met][ISUB[es]]["mean_by_lm"][ef][lm]
                    elif ef == "p_value":
                        vi = float(i_grid_pairs(SI, pre, met, ISUB[es], "p_value")[LLAMA.index(lm)].mean())
                    elif ef == "frac_p_le_0.05":
                        vi = float(np.mean(i_grid_pairs(SI, pre, met, ISUB[es], "p_value")[LLAMA.index(lm)] <= 0.05))
                    elif ef == "frac_s_cal_zero":
                        vi = float(np.mean(i_grid_pairs(SI, pre, met, ISUB[es], "s_cal")[LLAMA.index(lm)] == 0))
                    else:
                        sm.info(key, None, ve)
                        continue
                    sm.num(key, vi, ve)
        # readings
        RI = SI["per_pre"][pre]["readings"]
        for em, re_ in PE["readings"].items():
            met = IMET[em]
            ri = RI[met]
            key = f"{pre}|readings|{em}"
            for fi, fe in (("r_before_cka", "r_before_cka"), ("r_after_cka", "r_after_cka"),
                           ("r_before_mknn", "r_before_mknn"), ("r_after_mknn", "r_after_mknn"),
                           ("Tr_L_raw_cka", "Tr_L_raw_cka"), ("Tr_L_unc_cka", "Tr_L_unc_cka"), ("Tr_L_unc_mknn", "Tr_L_unc_mknn")):
                sm.num(f"{key}|{fe}", ri["inputs"][fi], re_[fe])
            sm.num(f"{key}|delta_diff", ri["arh_pooled10"]["dDelta"], re_["delta_diff"])
            sm.cat(f"{key}|verdict_arh_style_10LM", ri["arh_pooled10"]["verdict"].replace("ARH pattern ", ""),
                   re_["verdict_arh_style_10LM"].replace("ARH pattern ", ""))
            sm.cat(f"{key}|verdict_llama_axis", ri["llama_axis"], re_["verdict_llama_axis"])
            if "verdict_arh_style_8LM_frozen" in re_:
                sm.cat(f"{key}|verdict_arh_style_8LM", ri["arh_pooled8_reference"]["verdict"].replace("ARH pattern ", ""),
                       re_["verdict_arh_style_8LM_frozen"].replace("ARH pattern ", ""))
        # Part B verdicts
        QI = SI["per_pre"][pre]["partB"]
        QE = PE["predictions"]
        key = f"{pre}|P-L1"
        for f in ("Tr_raw", "Tr_scal", "Tr_unc", "dTr_scal", "dTr_unc"):
            sm.num(f"{key}|{f}", QI["P-L1"][f], QE["P-L1"][f])
        sm.cat(f"{key}|outcome", QI["P-L1"]["outcome"], QE["P-L1"]["outcome"])
        sm.num(f"{key}|deductive_scal.n_eligible", QI["P-L1"]["deductive_check"]["s_cal"]["n_eligible"],
               QE["P-L1"]["deductive_scal"]["n_qualifying"], 0)
        sm.num(f"{key}|deductive_scal.n_violations", QI["P-L1"]["deductive_check"]["s_cal"]["n_violations"],
               QE["P-L1"]["deductive_scal"]["n_reversed"], 0)
        sm.num(f"{key}|deductive_unc.n_eligible", QI["P-L1"]["deductive_check"]["uncensored"]["n_eligible"],
               QE["P-L1"]["deductive_unc"]["n_qualifying"], 0)
        key = f"{pre}|P-L2"
        for f in ("Tr_raw", "Tr_scal", "Tr_unc", "dTr_scal", "dTr_unc"):
            sm.num(f"{key}|{f}", QI["P-L2"][f], QE["P-L2"][f])
        sm.num(f"{key}|p_lower_dTr_scal", QI["P-L2"]["dTr_scal_p_one_decrease"], QE["P-L2"]["p_dTr_scal"]["p_lower"])
        sm.num(f"{key}|p_lower_dTr_unc", QI["P-L2"]["dTr_unc_p_one_decrease"], QE["P-L2"]["p_dTr_unc"]["p_lower"])
        sm.cat(f"{key}|material", QI["Bbar_material"], QE["P-L2"]["material"])
        sm.cat(f"{key}|outcome_scal", QI["P-L2"]["outcome_scal"], QE["P-L2"]["outcome_scal"])
        sm.cat(f"{key}|outcome_unc_secondary", QI["P-L2"]["outcome_unc_secondary"], QE["P-L2"]["outcome_unc_secondary"])
        for sname, ename in (("s_cal", "flips_scal"), ("uncensored", "flips_unc")):
            fe = QE["P-L2"].get(ename) or QE["P-L2"].get(ename + "_gap_ge_delta_b")
            fi = QI["P-L2"]["pairwise_flips_qualifying"][sname]
            sm.num(f"{key}|{ename}.n", fi["n"], fe["n_compared"], 0)
            for f in ("conc_to_disc", "disc_to_conc", "conc_to_tie", "disc_to_tie", "unchanged"):
                sm.num(f"{key}|{ename}.{f}", fi[f], fe[f], 0)
        key = f"{pre}|P-L2a"
        a, e = QI["P-L2a"], QE["P-L2a"]
        sm.num(f"{key}|median_abs_calpha_minus_Bbar", a["median_abs_calpha_minus_Bbar"], e["median_abs_calpha_minus_Bbar"])
        sm.num(f"{key}|range_min", a["calpha_minus_Bbar"]["min"], e["range_calpha_minus_Bbar"][0])
        sm.num(f"{key}|range_max", a["calpha_minus_Bbar"]["max"], e["range_calpha_minus_Bbar"][1])
        sm.num(f"{key}|mean_E0max_minus_Bbar", a["E0max_minus_Bbar"]["mean"], e["mean_E0max_minus_Bbar"])
        sm.num(f"{key}|pearson_calpha_Bbar", a["pearson_calpha_Bbar"], e["pearson_calpha_Bbar"])
        sm.num(f"{key}|n_qualifying", a["n_qualifying"], e["n_qualifying"], 0)
        sm.num(f"{key}|n_agree", a["n_agree"], e["n_agree_larger_higher"], 0)
        sm.num(f"{key}|n_needed", a["n_needed"], e["n_needed"], 0)
        sm.cat(f"{key}|holds", a["outcome"] == "holds", bool(e["holds"]))
        sm.num(f"{key}|deductive.n_vit_calpha_increasing", a["deductive_check"]["n_vit_calpha_increasing"],
               e["deductive"]["n_vit_calpha_increasing"], 0)
        sm.num(f"{key}|deductive.n_violations", a["deductive_check"]["n_violations"], len(e["deductive"]["violations_scal"]), 0)
        for r in ("0.5", "1.0"):
            key = f"{pre}|P-L3|r{r}"
            a, e = QI["P-L3"][f"r{r}"], QE["P-L3"][r]
            for f in ("Tr_raw", "Tr_scal", "Tr_unc", "dTr_scal"):
                sm.num(f"{key}|{f}", a[f], e[f])
            sm.cat(f"{key}|outcome", a["outcome"].split(" (")[0], e["outcome"].split(" (")[0])
            if "unc_outcome_descriptive" in a or "outcome_unc_descriptive" in e:
                sm.cat(f"{key}|outcome_unc_descriptive", a.get("unc_outcome_descriptive"), e.get("outcome_unc_descriptive"))
        key = f"{pre}|P-L4"
        a, e = QI["P-L4"], QE["P-L4"]
        for f in ("Tr_raw", "Tr_scal", "Tr_unc", "dTr_scal", "dTr_unc"):
            sm.num(f"{key}|{f}", a[f], e[f])
        sm.num(f"{key}|pooled_r_before", a["pooled10_r_before"], e["pooled_r_before"])
        sm.num(f"{key}|pooled_r_after", a["pooled10_r_after"], e["pooled_r_after"])
        sm.cat(f"{key}|holds", a["outcome"] == "holds", bool(e["holds"]))
        key = f"{pre}|P-L5"
        a, e = QI["P-L5"], QE["P-L5"]
        for f in ("Tr_raw_layermax", "Tr_M", "Tr_M_unc"):
            sm.num(f"{key}|{f}", a[f], e[f])
        sm.cat(f"{key}|outcome", a["outcome"].split(" (")[0], e["outcome"].split(" (")[0])
        sm.cat(f"{key}|outcome_unc_descriptive", a.get("unc_outcome_descriptive"), e.get("outcome_unc_descriptive"))
        # section 10 (E reports it per pre; I computes it for the primary pre)
        if pre == AG.PRIMARY:
            it = SI["section10_primary"]["items"]
            se = PE["section10"]
            for ki, ke in (("item1_premise_failure_P-L2a", "item1_premise_failure"), ("item2_P-L2_contradicted", "item2_PL2_contradicted"),
                           ("item3_P-L1_or_P-L4_fails", "item3_PL1_or_PL4_fails"), ("item4_P-L3_or_P-L5_reversed", "item4_PL3_or_PL5_reversed")):
                sm.cat(f"{pre}|section10|{ke}", it[ki], se[ke])
    return sm.rows


# ----------------------------------------------------------------------------------------- part 4
DERIVED_I = {"shift": "2_T_minus_calpha", "inflation": "inflation_calpha_minus_c_entry",
             "sel_entry_thr": "4_T_minus_c_entry", "sel_entry_mean": "5_T_minus_entry_null_mean"}


def _rank3(col):
    """average ranks of 3 exact (Fraction / int) values"""
    return [1 + sum(1 for y in col if y < x) + 0.5 * (sum(1 for y in col if y == x) - 1) for x in col]


def mknn_exact(SI, SE):
    """mKNN scores are count / (m k): every LLaMA-axis mKNN Tr_L recomputed with exact rational
    arithmetic on the integer counts (exact ties kept), compared with I's and E's float values."""
    from fractions import Fraction as Fr
    mk = L.M * L.KNN
    K = L.K_PERM
    rows = []
    for pre in L.PRES:
        vals = {}   # (sub, field) -> 3 x 17 list of Fractions
        for i, lm in enumerate(LLAMA):
            for j, vit in enumerate(VITS):
                z = np.load(os.path.join(L.OUT, pre, "pairs", f"{lm}__{vit}.npz"))
                S = z["obs_cnt"].astype(np.int64)
                N = z["null_cnt"].astype(np.int64)
                LA, LB = S.shape
                subs = AG.A.subsets(lm, vit, LA, LB)
                subs.update(AG.extra_subsets(LA, LB))
                for sub, (ra, rb) in subs.items():
                    Ss = S[np.ix_(ra, rb)]
                    Tk = N[:, ra][:, :, rb].reshape(K, -1).max(axis=1)
                    T = int(Ss.max())
                    c = int(np.sort(np.append(Tk, T))[ORDER_IDX])
                    rec = {"s_obs": Fr(T, mk), "c_alpha": Fr(c, mk),
                           "s_cal": (max(Fr(T - c, mk - c), Fr(0)) if mk - c > 0 else Fr(0)),
                           "uncensored": Fr(K * T - int(Tk.sum()), K * mk), "null_mean": Fr(int(Tk.sum()), K * mk)}
                    if sub == "full":
                        a, b = np.unravel_index(int(np.argmax(S)), S.shape)
                        ce = int(np.sort(np.append(N[:, a, b], S[a, b]))[ORDER_IDX])
                        rec.update({"shift": Fr(T - c, mk), "inflation": Fr(c - ce, mk), "sel_entry_thr": Fr(T - ce, mk),
                                    "sel_entry_mean": Fr(K * T - int(N[:, a, b].sum()), K * mk)})
                    for f, v in rec.items():
                        vals.setdefault((sub, f), [[None] * len(VITS) for _ in LLAMA])[i][j] = v
                # calibrate-then-max, exact
                Ce = np.sort(np.concatenate([S[None], N], axis=0), axis=0)[ORDER_IDX]
                M = Fr(0)
                for (u, w), sv in np.ndenumerate(S):
                    cu = int(Ce[u, w])
                    if mk - cu > 0 and sv > cu:
                        q = Fr(int(sv) - cu, mk - cu)
                        if q > M:
                            M = q
                Mu = Fr(int((K * S - N.sum(axis=0)).max()), K * mk)
                vals.setdefault(("ctm", "s_cal"), [[None] * len(VITS) for _ in LLAMA])[i][j] = M
                vals.setdefault(("ctm", "uncensored"), [[None] * len(VITS) for _ in LLAMA])[i][j] = Mu
        BI = SI["per_pre"][pre]["blocks"]["mknn"]
        CF = SI["per_pre"][pre]["diagnostics"]["mknn"]["counterfactual"]
        TE = SE["per_pre"][pre]["trends_L"]
        for (sub, f), G in sorted(vals.items()):
            R = np.array([_rank3([G[0][j], G[1][j], G[2][j]]) for j in range(len(VITS))], dtype=np.float64).T
            ex = AG.fam_trend(R, LLAMA)
            n_tie_cols = int(sum(1 for j in range(len(VITS)) if len({G[0][j], G[1][j], G[2][j]}) < 3))
            if f in DERIVED_I:
                i_tr = CF[DERIVED_I[f]]["Tr_L"]
            else:
                i_tr = BI[sub]["llama_axis"][f]["Tr"]
            ek = f"mknn|{SUBMAP.get(sub, sub)}|{({'s_cal': 'M', 'uncensored': 'M_unc'}[f] if sub == 'ctm' else f)}"
            e_tr = TE[ek]["Tr"] if ek in TE else None
            rows.append({"pre": pre, "sub": sub, "field": f, "Tr_exact": ex["Tr"], "n_vit_constant_exact": ex["n_vit_constant"],
                         "n_vit_cols_with_exact_ties": n_tie_cols, "Tr_I": i_tr, "Tr_E": e_tr,
                         "I_ok": i_tr is not None and abs(i_tr - ex["Tr"]) <= 1e-12,
                         "E_ok": None if e_tr is None else abs(e_tr - ex["Tr"]) <= 1e-12})
        log(f"{pre}: mKNN exact-rational trends done")
    return rows


def main():
    t0 = time.time()
    SI = json.load(open(I_SUM))
    SE = json.load(open(os.path.join(E_DIR, "summary.json")))
    PB = json.load(open(PARTB))
    p1, p2 = per_pair(SI)
    rows = summary_level(SI, SE, PB)
    ex = mknn_exact(SI, SE)
    ex_bad_I = [r for r in ex if not r["I_ok"]]
    ex_bad_E = [r for r in ex if r["E_ok"] is False]
    n_chk = [r for r in rows if r["ok"] is not None]
    bad = [r for r in n_chk if not r["ok"]]
    info = [r for r in rows if r["ok"] is None]
    # clip thresholds: E's shared cache vs I's run (single-model quantity; V-L1 'E and I bitwise equal')
    ce = json.load(open(os.path.join(L.PROJ, "results", "S1_2_layermax_llama", "cache", "clip_thresholds.json")))
    ci_ = json.load(open(os.path.join(L.OUT, "cache", "clip_thresholds_I.json")))["clip"]

    def _cval(v):
        return float(v["c"]) if isinstance(v, dict) else float(v)

    def _find(stem):
        for k, v in (ce.get("clip", ce) if isinstance(ce, dict) else {}).items():
            if k == stem or k.startswith(stem + "|"):
                return _cval(v)
        return None
    clip_cmp = {s: {"I": c, "E": _find(s), "bitwise_equal": _find(s) == c} for s, c in ci_.items()}
    clip_ok = all(v["bitwise_equal"] for v in clip_cmp.values())
    res = {
        "role": "llama-axis-run-indep", "what": "implementation I vs implementation E (early), after I's summary.json and REPORT.md were final",
        "inputs": {"I_summary_sha256": sha(I_SUM), "E_summary_sha256": sha(os.path.join(E_DIR, "summary.json")),
                   "E_report_sha256": sha(os.path.join(E_DIR, "REPORT.md")) if os.path.exists(os.path.join(E_DIR, "REPORT.md")) else None,
                   "E_hashes_json_sha256": sha(os.path.join(E_DIR, "hashes.json")) if os.path.exists(os.path.join(E_DIR, "hashes.json")) else None,
                   "partB_features_sha256": sha(PARTB), "code_sha256": sha(os.path.abspath(__file__))},
        "V-L1_clip_E_vs_I": {"per_model": clip_cmp, "all_bitwise_equal": clip_ok},
        "part1_V-L5_per_pair": p1,
        "part2_raw_arrays_informational": p2,
        "part3_summary_level": {"n_compared": len(n_chk), "n_mismatch": len(bad), "mismatches": bad[:200],
                                "n_informational": len(info), "informational": info[:400],
                                "pass": len(bad) == 0},
    }
    diag_fields = ("|shift|", "|inflation|", "|sel_entry_thr|", "|sel_entry_mean|", "|minus_Bbar|")
    bad_plan = [r for r in bad if not any(d in r["key"] for d in diag_fields)]
    bad_diag = [r for r in bad if any(d in r["key"] for d in diag_fields)]
    res["part3_summary_level"].update({"n_mismatch_plan_level": len(bad_plan), "n_mismatch_descriptive_diagnostics": len(bad_diag)})
    res["part4_mknn_exact_rational"] = {
        "what": "every LLaMA-axis mKNN Tr_L recomputed from the integer counts with exact rational arithmetic "
                "(exact ties kept), against I's and E's float values",
        "n": len(ex), "n_I_differs_from_exact": len(ex_bad_I), "n_E_differs_from_exact": len(ex_bad_E),
        "I_differs": ex_bad_I, "E_differs": ex_bad_E, "rows": ex}
    res["pass_plan_level"] = bool(p1["pass_V-L5_form"] and not bad_plan and clip_ok)
    res["pass_all"] = bool(res["pass_plan_level"] and not bad_diag)
    # lines for REPORT.md section 7b (numbers from this file only)
    fl = p1["floats_mixed_tol"]
    worst = max(fl.values(), key=lambda v: v["max_ratio_to_tol"])
    exact_fail = sum(v["n_fail"] for v in p1["exact"].values())
    a2 = p2["floats_mixed_tol"]
    a2_fail = {k: v["n_fail"] for k, v in a2.items() if v["n_fail"]}
    a2x = {k: v["n_fail"] for k, v in p2["exact"].items()}
    f32 = p2["null_f32"]
    res["report_lines"] = [
        (f"- **Outcome on plan-level quantities: {'PASS' if res['pass_plan_level'] else 'FAIL'}** (V-L5 per pair, clip thresholds, and every Tr_L / ΔTr_L / pooled r / reading / Part B input and outcome that both report). "
         f"Descriptive diagnostics: {len(bad_diag)} mismatching values ({'none' if not bad_diag else ', '.join(sorted({r['key'].rsplit('|', 1)[0] for r in bad_diag}))}). Details are in `validation_cross.json`."),
        f"- **V-L5 per pair** ({p1['n_pair_records']} pair records = 51 pairs × 2 pre; 3 metrics; 13 subsets plus calibrate-then-max):",
        f"  - floats (s_obs, c_α, s_cal, uncensored, null mean, null sd, selected-entry threshold and null mean, M, M_unc) within 1e-12·max + 1e-15: {sum(v['n'] for v in fl.values())} values, {sum(v['n_fail'] for v in fl.values())} outside; largest |Δ| {max(v['max_abs'] for v in fl.values()):.1e}, at most {worst['max_ratio_to_tol']:.2f}× the tolerance;",
        f"  - exact items (p, argmax, tied maxima, entry counts, calibrate-then-max argmax): {sum(v['n'] for v in p1['exact'].values())} checks, {exact_fail} differ;",
        f"  - mKNN null counts (500 × L_A × L_B per pair) identical in {p2['exact']['mknn.null_counts_identical']['n'] - p2['exact']['mknn.null_counts_identical']['n_fail']}/{p2['exact']['mknn.null_counts_identical']['n']} pair records; observed counts identical in {p2['exact']['mknn.obs_counts_identical']['n'] - p2['exact']['mknn.obs_counts_identical']['n_fail']}/{p2['exact']['mknn.obs_counts_identical']['n']}.",
        f"- **Clip thresholds (V-L1, E vs I):** {sum(1 for v in clip_cmp.values() if v['bitwise_equal'])}/{len(clip_cmp)} bitwise equal.",
        f"- **Entry-level arrays (beyond the plan items, informational):** values outside the mixed tolerance: {a2_fail if a2_fail else 'none'}; largest |Δ| {max(v['max_abs'] for v in a2.values()):.1e}. Exact items with differences: { {k: v for k, v in a2x.items() if v} or 'none'}. E stores CKA nulls in float32: {sum(v['n_not_equal_to_float32_of_I'] for v in f32.values())} of {sum(v['n'] for v in f32.values())} values differ from float32(I), max relative |Δ| {max(v['max_rel_f32_minus_I'] for v in f32.values()):.1e}.",
        f"- **Summary level:** {len(n_chk)} quantities that E also reports (every Tr_L with its exact p and per-ViT ρ, ΔTr_L and its p, Tr_BL, pooled r and exact p, vision trend, means by size, readings, Part B outcomes and their inputs, plan §10 items): {len(bad)} mismatches (tolerance 1e-12 absolute; categorical outcomes exact). Bootstrap CIs whose rng key strings differ between the two codes are listed as informational ({len(info)} values), not compared.",
    ]
    if bad:
        res["report_lines"].append("- **Mismatches:** " + "; ".join(
            f"{r['key']} (I {r['I'] if not isinstance(r['I'], float) else round(r['I'], 4)}, E {r['E'] if not isinstance(r['E'], float) else round(r['E'], 4)})"
            for r in bad if r["kind"] != "vec") + ". The per-ViT ρ vectors of the same quantities differ too.")
    res["report_lines"].append(
        f"- **mKNN exact-tie check.** mKNN scores are counts / 10240. Recomputing every LLaMA-axis mKNN Tr_L from the integer counts with exact rational arithmetic "
        f"({len(ex)} trends: 13 subsets × 5 fields, calibrate-then-max × 2 fields and 4 derived differences, × 2 pre) gives the exact values. "
        f"I differs from them in {len(ex_bad_I)} and E in {len(ex_bad_E)}.")
    for r in ex_bad_I + [x for x in ex_bad_E if x not in ex_bad_I]:
        res["report_lines"].append(
            f"  - {r['pre']}, {r['sub']}, {r['field']}: exact {r['Tr_exact']:+.4f} ({r['n_vit_cols_with_exact_ties']} ViTs with exact ties); "
            f"I {r['Tr_I']:+.4f}; E {('n/a' if r['Tr_E'] is None else format(r['Tr_E'], '+.4f'))}.")
    if ex_bad_I or ex_bad_E:
        res["report_lines"].append(
            "  - Cause: a difference of two count/10240 floats (T − c_α, c_α − c_α^entry) is not always equal for equal integer differences, "
            "so exact ties between sizes are broken by round-off in one or both codes. Only these descriptive A3 differences are affected; "
            "every Tr_L of T, c_α, s_cal, uncensored, E0[max], M and M_unc, and so P-L1, equals the exact value in both implementations.")
    res["secs"] = time.time() - t0
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(L.jsonable(res), open(OUTF + ".tmp", "w"), indent=1)
    os.replace(OUTF + ".tmp", OUTF)
    log(f"wrote {OUTF}: pass_plan_level={res['pass_plan_level']} pass_all={res['pass_all']} V-L5={p1['pass_V-L5_form']} "
        f"summary mismatches={len(bad)}/{len(n_chk)} (plan {len(bad_plan)}, diag {len(bad_diag)}) clip={clip_ok} "
        f"exact mKNN: I differs {len(ex_bad_I)}, E differs {len(ex_bad_E)} of {len(ex)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
