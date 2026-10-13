"""LLAMA-AXIS (new file, implementation E, role llama-axis-run-early): V-L5, E vs I.

Plan: analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md section 4, V-L5 (E vs I on the 34 new
pairs, both pre, all metrics and subsets):
  * mKNN null counts identical;
  * argmax identical;
  * s_obs, c_alpha, s_cal, uncensored and null mean within |a - b| <= 1e-12 max(|a|, |b|) + 1e-15;
  * p exactly equal.
The 17 llama-7b pairs are compared too (reported separately; V-L4(b) already tied each
implementation to the frozen S1.2 layer-max outputs).

E side: results/S1_2_layermax_llama/early/pairs/<pre>/<lm>__<vit>.{json,npz} (this role).
I side: results/S1_2_layermax_llama/indep/<pre>/pairs/<lm>__<vit>.npz (read-only), summarised
with I's own per-pair code (lmx_aggregate.pair_stats + lmx_llama_aggregate.pair_stats_ext,
read-only imports; nothing of I is written). Entry-level arrays are compared as well.

Output: results/S1_2_layermax_llama/early/validation/VL5_E_vs_I.json
(attempt 1, which compared the entry-wise p floats instead of their counts, is kept as
VL5_E_vs_I.attempt1_entry_p_float.json; every plan item passed there too.)
"""
from __future__ import annotations

import os

_TH = os.environ.get("LLAMA_AXIS_THREADS", "4")
os.environ["LMX_THREADS"] = _TH
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _TH

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402

INDEP_CODE = os.path.join(LC.PROJ, "analysis", "s12_layermax_llama", "indep")
sys.path.append(INDEP_CODE)          # appended: E's own modules (lm_core) win on name clashes
import lmx_llama_aggregate as IAGG  # noqa: E402  (read-only import; main() not run)

INDEP_RES = os.path.join(LC.PROJ, "results", "S1_2_layermax_llama", "indep")
OUT = os.path.join(LC.RES, "validation", "VL5_E_vs_I.json")
METS = {"mknn": "mknn", "cka_biased": "cka_b", "cka_debiased": "cka_d"}
SUBMAP = {"full": "full", "grid2": "grid2", "grid3": "grid3", "grid5": "grid5", "grid9": "grid9",
          "grid17": "grid17", "fixed_r0.5": "fixed_r0.5", "fixed_r1.0": "fixed_r1.0",
          "LMall_ViTlast": "lmall_x_vitlast", "LMlast_ViTall": "lmlast_x_vitall",
          "noLMemb": "noLMemb", "LMdeep": "LMdeep", "LMemb_only": "LMemb_only"}
TOL_FIELDS = ("s_obs", "c_alpha", "s_cal", "uncensored", "null_mean")
EXTRA_FIELDS = ("null_sd",)


def ok_tol(a, b):
    return abs(a - b) <= 1e-12 * max(abs(a), abs(b)) + 1e-15


def excess(a, b):
    """|a-b| / (1e-12 max(|a|,|b|) + 1e-15): <= 1 passes."""
    return abs(a - b) / (1e-12 * max(abs(a), abs(b)) + 1e-15)


class Acc:
    def __init__(self):
        self.n = 0
        self.n_fail = 0
        self.max_abs = 0.0
        self.max_excess = 0.0
        self.fails = []

    def add(self, a, b, where, tol=True):
        self.n += 1
        d = abs(a - b)
        self.max_abs = max(self.max_abs, d)
        if tol:
            e = excess(a, b)
            self.max_excess = max(self.max_excess, e)
            bad = e > 1.0
        else:
            bad = a != b
        if bad:
            self.n_fail += 1
            if len(self.fails) < 40:
                self.fails.append({"where": where, "E": a, "I": b, "absdiff": d})

    def add_arr(self, a, b, where):
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        d = np.abs(a - b)
        e = d / (1e-12 * np.maximum(np.abs(a), np.abs(b)) + 1e-15)
        self.n += a.size
        self.max_abs = max(self.max_abs, float(d.max()))
        self.max_excess = max(self.max_excess, float(e.max()))
        bad = np.flatnonzero(e > 1.0)
        self.n_fail += int(bad.size)
        for j in bad[: max(0, 40 - len(self.fails))]:
            self.fails.append({"where": f"{where}|flat{int(j)}", "E": float(a[j]), "I": float(b[j]), "absdiff": float(d[j])})

    def out(self):
        return {"n": self.n, "n_fail": self.n_fail, "max_abs": self.max_abs, "max_excess_over_tol": self.max_excess,
                "fails_first40": self.fails}


def compare(groups):
    res = {}
    for gname, lms in groups.items():
        acc = {}

        def A(k):
            if k not in acc:
                acc[k] = Acc()
            return acc[k]

        n_pairs = 0
        for pre in LC.PRES:
            for lm in lms:
                for vit, _ in LC.VITS:
                    n_pairs += 1
                    where = f"{pre}|{lm}|{vit}"
                    rec = json.load(open(os.path.join(LC.RES, "pairs", pre, f"{lm}__{vit}.json")))
                    ze = np.load(os.path.join(LC.RES, "pairs", pre, f"{lm}__{vit}.npz"))
                    zi = np.load(os.path.join(INDEP_RES, pre, "pairs", f"{lm}__{vit}.npz"))
                    A("meta.key_K_shape").add(
                        int(rec["perm_key"] == str(zi["key"]) and rec["K"] == int(zi["K"])
                            and rec["L_A"] == int(zi["L_A"]) and rec["L_B"] == int(zi["L_B"])), 1, where, tol=False)
                    # ---- entry-level arrays
                    cE = np.rint(ze["mknn.entry_obs"] * LC.M_ITEMS * LC.KNN_K).astype(np.int64)
                    A("arr.mknn_obs_counts_identical").add(int(np.array_equal(cE, zi["obs_cnt"].astype(np.int64))), 1,
                                                           where, tol=False)
                    A("arr.mknn_null_counts_identical").add(
                        int(np.array_equal(ze["mknn.null_counts_int16"].astype(np.int64), zi["null_cnt"].astype(np.int64))),
                        1, where, tol=False)
                    for met, im in (("cka_biased", "b"), ("cka_debiased", "d")):
                        oE, oI = ze[f"{met}.entry_obs"], zi[f"obs_cka_{im}"]
                        nI = zi[f"null_cka_{im}"]
                        A(f"arr.{met}.obs").add_arr(oE, oI, where)
                        A(f"arr.{met}.entry_null_mean").add_arr(ze[f"{met}.entry_null_mean"], nI.mean(axis=0), where)
                        comb = np.concatenate([oI[None], nI], axis=0)
                        cI = np.sort(comb, axis=0)[IAGG.ORDER_IDX]
                        pI = (1.0 + (nI >= oI[None]).sum(axis=0)) / (nI.shape[0] + 1)
                        A(f"arr.{met}.entry_c_alpha").add_arr(ze[f"{met}.entry_c_alpha"], cI, where)
                        # entry p: compare the exceedance counts (E divides on the GPU, where x/501 can differ
                        # from numpy's correctly rounded division by 1 ulp; attempt 1 compared the floats)
                        kE = np.rint(ze[f"{met}.entry_p"] * (nI.shape[0] + 1)).astype(np.int64)
                        kI = 1 + (nI >= oI[None]).sum(axis=0).astype(np.int64)
                        A(f"arr.{met}.entry_p_counts_identical").add(int(np.array_equal(kE, kI)), 1, where, tol=False)
                        A(f"arr.{met}.entry_p_float").add_arr(ze[f"{met}.entry_p"], pI, where)
                        # E stores the null tensor as float32; I as float64
                        f32 = ze[f"{met}.null_f32"]
                        acc.setdefault(f"arr.{met}.null_f32_vs_I_cast_f32", {"n": 0, "n_bitwise_equal_tensors": 0,
                                                                            "max_abs": 0.0, "n_entries_differ": 0})
                        blk = acc[f"arr.{met}.null_f32_vs_I_cast_f32"]
                        dd = np.abs(f32.astype(np.float64) - nI.astype(np.float32).astype(np.float64))
                        blk["n"] += 1
                        blk["n_bitwise_equal_tensors"] += int(np.array_equal(f32, nI.astype(np.float32)))
                        blk["max_abs"] = max(blk["max_abs"], float(dd.max()))
                        blk["n_entries_differ"] += int(np.sum(dd > 0))
                        # full-grid null maxima and their location
                        nmax = nI.reshape(nI.shape[0], -1).max(axis=1)
                        narg = nI.reshape(nI.shape[0], -1).argmax(axis=1)
                        A(f"arr.{met}.full_null_max").add_arr(ze[f"{met}.full_null_max"], nmax, where)
                        LB = oE.shape[1]
                        A(f"arr.{met}.full_null_argmax_identical").add(
                            int(np.array_equal(ze[f"{met}.full_null_argmax_A"].astype(np.int64), narg // LB)
                                and np.array_equal(ze[f"{met}.full_null_argmax_B"].astype(np.int64), narg % LB)),
                            1, where, tol=False)
                    # ---- summaries: E json vs I's own per-pair code
                    P = IAGG.pair_stats_ext(zi, lm, vit)
                    isubs = dict(IAGG.A.subsets(lm, vit, rec["L_A"], rec["L_B"]))
                    isubs.update(IAGG.extra_subsets(rec["L_A"], rec["L_B"]))
                    for met, im in METS.items():
                        for se, si in SUBMAP.items():
                            e = rec["summaries"][met][se]
                            i = P[im][si]
                            w2 = f"{where}|{met}|{se}"
                            for f in TOL_FIELDS:
                                A(f"sum.{met}.{f}").add(float(e[f]), float(i[f]), w2)
                            for f in EXTRA_FIELDS:
                                A(f"sum_extra.{met}.{f}").add(float(e[f]), float(i[f]), w2)
                            A(f"sum.{met}.p_value_exact").add(float(e["p_value"]), float(i["p_value"]), w2, tol=False)
                            A(f"sum.{met}.argmax_identical").add(int(e["argmax_A"] == i["a"] and e["argmax_B"] == i["b"]),
                                                                 1, w2, tol=False)
                            A(f"sum.{met}.n_tied_max_identical").add(int(e["n_tied_max"]), int(i["n_tied_max"]), w2,
                                                                     tol=False)
                            ia, ib = isubs[si]
                            A(f"sum.{met}.layer_sets_identical").add(
                                int(list(e["layers_A"]) == [int(x) for x in ia] and list(e["layers_B"]) == [int(x) for x in ib]),
                                1, w2, tol=False)
                        # protocol (iv) calibrate-then-max
                        ce, ci = rec["calib_then_max"][met], P[im]["ctm"]
                        w2 = f"{where}|{met}|ctm"
                        A(f"ctm.{met}.M").add(float(ce["M"]), float(ci["s_cal"]), w2)
                        A(f"ctm.{met}.M_unc").add(float(ce["M_unc"]), float(ci["uncensored"]), w2)
                        A(f"ctm.{met}.c_alpha_at_argmax").add(float(ce["c_alpha_at_argmax"]), float(ci["c_entry_at_argmax"]), w2)
                        A(f"ctm.{met}.argmax_identical").add(int(ce["argmax_A"] == ci["a"] and ce["argmax_B"] == ci["b"]
                                                                 and ce["M_unc_argmax_A"] == ci["unc_a"]
                                                                 and ce["M_unc_argmax_B"] == ci["unc_b"]), 1, w2, tol=False)
                        A(f"ctm.{met}.n_entries_scal_pos_identical").add(int(ce["n_entries_scal_pos"]),
                                                                         int(ci["n_entries_scal_pos"]), w2, tol=False)
        out = {"n_pair_pre": n_pairs}
        for k, v in sorted(acc.items()):
            out[k] = v.out() if isinstance(v, Acc) else v
        # verdict on the plan's V-L5 items
        req = [k for k in out if k.startswith(("sum.", "ctm.", "arr.mknn"))]
        out["plan_items_pass"] = {
            "mknn_null_counts_identical": out["arr.mknn_null_counts_identical"]["n_fail"] == 0,
            "argmax_identical": all(out[k]["n_fail"] == 0 for k in out if k.endswith("argmax_identical")),
            "tolerance_fields": all(out[k]["n_fail"] == 0 for k in out
                                    if any(k == f"sum.{m}.{f}" for m in METS for f in TOL_FIELDS)
                                    or k in [f"ctm.{m}.{f}" for m in METS for f in ("M", "M_unc")]),
            "p_exact": all(out[f"sum.{m}.p_value_exact"]["n_fail"] == 0 for m in METS),
        }
        out["pass"] = all(out["plan_items_pass"].values())
        out["all_checked_keys_fail_counts"] = {k: out[k]["n_fail"] for k in req if isinstance(out[k], dict) and "n_fail" in out[k]}
        res[gname] = out
    return res


def main():
    t0 = time.time()
    groups = {"new_pairs_34": ["huggyllama__llama-13b", "huggyllama__llama-30b"],
              "llama7b_pairs_17": ["huggyllama__llama-7b"]}
    res = compare(groups)
    res["meta"] = {"plan_item": "ANALYSIS_PLAN_LLAMA.md section 4 V-L5",
                   "tolerance": "|a-b| <= 1e-12 max(|a|,|b|) + 1e-15; p, counts, argmax exact",
                   "I_code": {f: IAGG.sha(os.path.join(INDEP_CODE, f)) for f in ("lmx_llama_aggregate.py", "lmx_aggregate.py",
                                                                                "lmx_common.py", "common_llama.py")},
                   "elapsed_s": time.time() - t0, "finished_utc": LC.utc(), "threads": _TH}
    res["pass"] = bool(res["new_pairs_34"]["pass"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(LC.jsonable(res), f, indent=1)
    for g in ("new_pairs_34", "llama7b_pairs_17"):
        print(g, "pass", res[g]["pass"], res[g]["plan_items_pass"], flush=True)
        for k, v in res[g].items():
            if isinstance(v, dict) and "n_fail" in v:
                print(f"  {k}: n={v['n']} fail={v['n_fail']} max_abs={v['max_abs']:.3g} "
                      f"max_excess={v.get('max_excess_over_tol', 0):.3g}", flush=True)
            elif isinstance(v, dict) and "n_bitwise_equal_tensors" in v:
                print(f"  {k}: {v}", flush=True)
    print("V-L5", "PASS" if res["pass"] else "FAIL", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
