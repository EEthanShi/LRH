#!/usr/bin/env python
"""LLaMA axis, implementation I (role llama-axis-run-indep): validation items
V-L1, V-L2, V-L4b and V-L7 of analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md section 4.
(V-L3 and V-L4a = the copied validate.py, I2 and I3; V-L6 = vl6_arh.py.)

  V-L1  clip thresholds used by the run (prep_meta.json): recomputed all-layer
        threshold == reference bitwise (S1.2 cache for llama-7b and the ViTs,
        frozen Part B clip_thresholds_partB.json for llama-13b/30b).
        Writes results/S1_2_layermax_llama/indep/cache/clip_thresholds_I.json.
  V-L2  kNN tables (k+1 = 11) of every stored layer of the 3 LLaMA and 17 ViTs,
        both pre, with hrhv2.structural.knn_table (the tables the run uses by
        construction); ViT and llama-7b tables == the frozen layer-max caches
        (results/S1_2_layermax/cache/knn_hrhv2, knn_final); exact k=10 boundary ties
        and SPEC near-tie fractions per layer; ViT boundary ties must be 0.
        Extra (V-L2b): every observed count matrix of the run (all layer pairs,
        51 pairs, both pre) == the overlap counts of these hrhv2 tables.
  V-L4b llama-7b layer-max outputs of this run vs the frozen 8-LM outputs:
        raw matrices vs results/S1_2_layermax/indep (rel <= 1e-12, counts identical),
        pair summaries (every subset, T, c_alpha, s_cal, p, uncensored, null mean,
        argmax) vs results/S1_2_layermax/indep and results/S1_2_layermax/pairs (early, old).
  V-L7  Proposition 5 on the new pairs: entry-wise permutation null means vs
        sqrt(PR_u PR_v)/(m-1) (biased CKA; PR from partB_features.json), 0
        (debiased CKA) and 10/1023 (mKNN); median relative error and the fraction
        of entries with |z| > 4 against the MC standard error.
Output: results/S1_2_layermax_llama/indep/validation_vl.json
"""
from __future__ import annotations

import os

_T = os.environ.get("LMX_THREADS", "3")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import lmx_common as L  # noqa: E402
from lmx_common import C, st  # noqa: E402
from hrhv2.prep import gram_to_dist, median_offdiag  # noqa: E402
import lmx_aggregate as A  # noqa: E402  (pair_stats only)

KMAX = L.KNN + 1
FROZEN_KNN = os.path.join(L.PROJ, "results", "S1_2_layermax", "cache")
EARLY_OLD = os.path.join(L.PROJ, "results", "S1_2_layermax", "pairs")
PARTB = os.path.join(L.PROJ, "results", "S1_2_layermax_llama", "partB", "partB_features.json")
NEW_LMS = ("huggyllama__llama-13b", "huggyllama__llama-30b")
L7B = "huggyllama__llama-7b"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mixed_ok(a, b):
    """V-L5 tolerance |a - b| <= 1e-12 max(|a|, |b|) + 1e-15 (reported next to the strict relative one)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.all(np.abs(a - b) <= 1e-12 * np.maximum(np.abs(a), np.abs(b)) + 1e-15))


def track(d, k, v_a, v_b, where):
    """record max relative and max absolute difference of field k, with location"""
    r = relmax(v_a, v_b, 1e-15)
    ab = float(np.max(np.abs(np.asarray(v_a, dtype=np.float64) - np.asarray(v_b, dtype=np.float64))))
    if r > d["max_rel"].get(k, -1.0):
        d["max_rel"][k] = r
        d.setdefault("where_max_rel", {})[k] = {"where": where, "a": float(np.max(np.abs(v_a))), "absdiff": ab}
    d.setdefault("max_abs", {})[k] = max(d.setdefault("max_abs", {}).get(k, 0.0), ab)
    d["mixed_tol_ok"] = bool(d.get("mixed_tol_ok", True) and mixed_ok(v_a, v_b))


def relmax(a, b, floor=1e-300):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    s = np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)
    return float(np.max(d / s)) if d.size else 0.0


# --------------------------------------------------------------------------- V-L1
def vl1():
    out = {"per_model": {}}
    meta = json.load(open(os.path.join(L.OUT, "prh_clip+l2", "prep_meta.json")))
    for stem, m in meta.items():
        out["per_model"][stem] = {"c_recomputed": m["clip_recomputed"], "c_reference": m["clip_s12_cache"],
                                  "reference": ("partB/clip_thresholds_partB.json" if stem in NEW_LMS
                                                else "results/S1_2/cache/clip_thresholds.json"),
                                  "bitwise_equal": bool(m["clip_bitwise_equal"])}
    out["n_models"] = len(out["per_model"])
    out["all_bitwise_equal"] = all(v["bitwise_equal"] for v in out["per_model"].values())
    out["pass"] = bool(out["all_bitwise_equal"] and out["n_models"] == 20)
    os.makedirs(os.path.join(L.OUT, "cache"), exist_ok=True)
    json.dump({"note": "role llama-axis-run-indep; hrhv2.prep.prh_clip_threshold on the float64 all-layer stack (D5); "
                       "recomputed inside run_layermax.prep_models",
               "clip": {s: v["c_recomputed"] for s, v in out["per_model"].items()}},
              open(os.path.join(L.OUT, "cache", "clip_thresholds_I.json"), "w"), indent=1)
    return out


# --------------------------------------------------------------------------- V-L2
def tables(stem, pre):
    S = L.load_stack64(stem)
    c = C.clip_threshold(stem, "wit1024") if pre == "prh_clip+l2" else None
    nL = S.shape[1]
    idx_all = np.empty((L.M, nL, KMAX), dtype=np.int64)
    rows = []
    for l in range(nL):
        Xp = C.preprocess_rows(S[:, l, :], pre, c)
        D = gram_to_dist(Xp @ Xp.T)
        idx, vals = st.knn_table(D, KMAX)
        idx_all[:, l, :] = idx
        rows.append({"layer": l, "exact_boundary_tie_rows_k10": int(np.sum(vals[:, L.KNN] == vals[:, L.KNN - 1])),
                     "near_tie_frac_k10": float(st.near_tie_fraction(vals, L.KNN, median_offdiag(D)))})
    del S
    return idx_all, rows


def overlap_matrix(idxA, idxB):
    """count[a, b] = sum_i |N_A^a(i) cap N_B^b(i)| (k = 10) from (m, L, >=10) tables."""
    m = L.M
    LA, LB = idxA.shape[1], idxB.shape[1]
    out = np.empty((LA, LB), dtype=np.int64)
    rr = np.repeat(np.arange(m), L.KNN)
    for a in range(LA):
        MA = np.zeros((m, m), dtype=bool)
        MA[rr, idxA[:, a, :L.KNN].reshape(-1)] = True
        for b in range(LB):
            out[a, b] = int(MA[rr, idxB[:, b, :L.KNN].reshape(-1)].sum())
    return out


def vl2():
    out = {"per_table": {}, "frozen_compare": {}, "obs_count_check": {}}
    tabs = {}
    for pre in L.PRES:
        for stem in L.LM_STEMS + L.VIT_STEMS:
            t0 = time.time()
            idx, rows = tables(stem, pre)
            tabs[(stem, pre)] = idx
            os.makedirs(os.path.join(L.OUT, "cache", "knn_hrhv2"), exist_ok=True)
            np.save(os.path.join(L.OUT, "cache", "knn_hrhv2", f"{stem}__{pre}.npy"), idx)
            out["per_table"][f"{stem}|{pre}"] = rows
            if stem not in NEW_LMS:
                cmp = {}
                for which in ("knn_hrhv2", "knn_final"):
                    fz = np.load(os.path.join(FROZEN_KNN, which, f"{stem}__{pre}.npy"))
                    cmp[f"{which}_identical_11cols"] = bool(np.array_equal(fz, idx))
                    cmp[f"{which}_identical_k10"] = bool(np.array_equal(fz[:, :, :L.KNN], idx[:, :, :L.KNN]))
                out["frozen_compare"][f"{stem}|{pre}"] = cmp
            log(f"V-L2 {stem} {pre} L={idx.shape[1]} ties={sum(r['exact_boundary_tie_rows_k10'] for r in rows)} "
                f"{time.time() - t0:.1f}s")
        # V-L2b: observed counts of every pair == overlap of these tables
        n_bad, n_tot = 0, 0
        for lm in L.LM_STEMS:
            for vit in L.VIT_STEMS:
                z = np.load(os.path.join(L.OUT, pre, "pairs", f"{lm}__{vit}.npz"))
                ref = overlap_matrix(tabs[(lm, pre)], tabs[(vit, pre)])
                obs = z["obs_cnt"].astype(np.int64)
                n_bad += int(np.sum(obs != ref))
                n_tot += int(obs.size)
        out["obs_count_check"][pre] = {"n_layer_pairs": n_tot, "n_differ": n_bad, "identical": n_bad == 0}
        log(f"V-L2b {pre}: {n_tot} layer pairs, {n_bad} differ")
    vit_ties = sum(r["exact_boundary_tie_rows_k10"] for k, rows in out["per_table"].items()
                   if k.split("|")[0] in L.VIT_STEMS for r in rows)
    lm_summary = {}
    for k, rows in out["per_table"].items():
        stem, pre = k.split("|")
        if stem in L.LM_STEMS:
            lm_summary[k] = {"layers_with_exact_boundary_ties": int(sum(r["exact_boundary_tie_rows_k10"] > 0 for r in rows)),
                             "total_exact_boundary_tie_rows": int(sum(r["exact_boundary_tie_rows_k10"] for r in rows)),
                             "max_near_tie_frac_k10": float(max(r["near_tie_frac_k10"] for r in rows)),
                             "n_layers": len(rows)}
    out["summary"] = {
        "vit_total_exact_boundary_tie_rows": int(vit_ties),
        "vit_max_near_tie_frac_k10": float(max(r["near_tie_frac_k10"] for k, rows in out["per_table"].items()
                                               if k.split("|")[0] in L.VIT_STEMS for r in rows)),
        "lm": lm_summary,
        "frozen_tables_all_identical": all(all(v.values()) for v in out["frozen_compare"].values()),
        "n_frozen_compared": len(out["frozen_compare"]),
        "obs_counts_all_identical": all(v["identical"] for v in out["obs_count_check"].values()),
    }
    out["pass"] = bool(vit_ties == 0 and out["summary"]["frozen_tables_all_identical"]
                       and out["summary"]["obs_counts_all_identical"])
    return out


# --------------------------------------------------------------------------- V-L4b
MAP_E = {"mknn": "mknn", "cka_b": "cka_biased", "cka_d": "cka_debiased"}
SUBMAP_E = {"full": "full", "grid2": "grid2", "grid3": "grid3", "grid5": "grid5", "grid9": "grid9", "grid17": "grid17",
            "fixed_r0.5": "fixed_r0.5", "fixed_r1.0": "fixed_r1.0", "lmall_x_vitlast": "LMall_ViTlast",
            "lmlast_x_vitall": "LMlast_ViTall"}
FIELDS = ("s_obs", "c_alpha", "s_cal", "p_value", "uncensored", "null_mean")


def vl4b():
    out = {}
    for pre in L.PRES:
        raw = {"max_rel": {}, "counts_identical": True, "key_K_equal": True}
        summ_old = {"max_rel": {}, "p_identical": True, "argmax_identical": True}
        summ_early = {"max_rel": {}, "p_identical": True, "argmax_identical": True, "null_counts_identical": True,
                      "n_pairs": 0}
        for vit in L.VIT_STEMS:
            fn = f"{L7B}__{vit}.npz"
            z = np.load(os.path.join(L.OUT, pre, "pairs", fn))
            zo = np.load(os.path.join(L.OLD_INDEP, pre, "pairs", fn))
            raw["key_K_equal"] &= bool(str(z["key"]) == str(zo["key"]) and int(z["K"]) == int(zo["K"]))
            for f in ("obs_cka_b", "obs_cka_d", "null_cka_b", "null_cka_d"):
                track(raw, f, z[f], zo[f], vit)
            raw["counts_identical"] &= bool(np.array_equal(z["obs_cnt"], zo["obs_cnt"])
                                            and np.array_equal(z["null_cnt"], zo["null_cnt"]))
            ps, po = A.pair_stats(z, L7B, vit), A.pair_stats(zo, L7B, vit)
            for met in A.METRICS:
                for sub, d in ps[met].items():
                    for f, v in d.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            k = f"{met}.{f}"
                            if f in ("a", "b", "n_entries", "n_tied_max"):
                                summ_old["argmax_identical"] &= (v == po[met][sub][f])
                            elif f == "p_value":
                                summ_old["p_identical"] &= (v == po[met][sub][f])
                            else:
                                track(summ_old, k, v, po[met][sub][f], f"{vit}|{sub}")
            # early implementation, frozen 8-LM outputs (old; plan V-L4b)
            fj = os.path.join(EARLY_OLD, pre, f"{L7B}__{vit}.json")
            fz = os.path.join(EARLY_OLD, pre, f"{L7B}__{vit}.npz")
            if os.path.exists(fj) and os.path.exists(fz):
                e = json.load(open(fj))
                ez = np.load(fz)
                summ_early["n_pairs"] += 1
                summ_early["null_counts_identical"] &= bool(np.array_equal(
                    ez["mknn.null_counts_int16"].astype(np.int64), z["null_cnt"].astype(np.int64)))
                for met, em in MAP_E.items():
                    for sub, esub in SUBMAP_E.items():
                        es = e["summaries"][em][esub]
                        for f in FIELDS:
                            if f == "p_value":
                                summ_early["p_identical"] &= (float(es[f]) == ps[met][sub][f])
                            else:
                                k = f"{met}.{f}"
                                track(summ_early, k, float(es[f]), ps[met][sub][f], f"{vit}|{sub}")
                    summ_early["argmax_identical"] &= bool(e["summaries"][em]["full"]["argmax_A"] == ps[met]["full"]["a"]
                                                           and e["summaries"][em]["full"]["argmax_B"] == ps[met]["full"]["b"])
        tol = 1e-12
        strict_fail = sorted([f"{nm}:{k}" for nm, dd in (("raw", raw), ("indep", summ_old), ("early", summ_early))
                              for k, v in dd["max_rel"].items() if v > 1e-12])
        mixed = bool(raw["counts_identical"] and raw["key_K_equal"] and summ_old["p_identical"]
                     and summ_old["argmax_identical"] and summ_early["n_pairs"] == 17 and summ_early["p_identical"]
                     and summ_early["argmax_identical"] and summ_early["null_counts_identical"]
                     and raw["mixed_tol_ok"] and summ_old["mixed_tol_ok"] and summ_early["mixed_tol_ok"])
        out[pre] = {"strict_relative_1e-12_exceeded_by": strict_fail, "pass_mixed_tolerance_V-L5_form": mixed,
                    "raw_vs_frozen_indep": raw, "summary_vs_frozen_indep": summ_old, "summary_vs_frozen_early": summ_early,
                    "pass": bool(raw["counts_identical"] and raw["key_K_equal"] and max(raw["max_rel"].values()) <= tol
                                 and summ_old["p_identical"] and summ_old["argmax_identical"]
                                 and max(summ_old["max_rel"].values()) <= tol
                                 and summ_early["n_pairs"] == 17 and summ_early["p_identical"]
                                 and summ_early["argmax_identical"] and summ_early["null_counts_identical"]
                                 and max(summ_early["max_rel"].values()) <= tol)}
        log(f"V-L4b {pre}: pass={out[pre]['pass']} raw max_rel={max(raw['max_rel'].values()):.2e} "
            f"summ(indep) {max(summ_old['max_rel'].values()):.2e} summ(early) {max(summ_early['max_rel'].values()):.2e}")
    out["pass"] = all(out[p]["pass"] for p in L.PRES)
    out["pass_mixed_tolerance_V-L5_form"] = all(out[p]["pass_mixed_tolerance_V-L5_form"] for p in L.PRES)
    return out


# --------------------------------------------------------------------------- V-L7
def vl7():
    pb = json.load(open(PARTB))
    m, K = L.M, L.K_PERM
    out = {}
    for pre in L.PRES:
        PR = {**{s: np.array(v["PR"]) for s, v in pb["per_pre"][pre]["lm"].items()},
              **{s: np.array(v["PR"]) for s, v in pb["per_pre"][pre]["vit"].items()}}
        res = {}
        for grp, lms in (("new_34", NEW_LMS), ("llama7b_17", (L7B,))):
            relerr, zb, zd, zk, absd = [], [], [], [], []
            pair_med = []
            for lm in lms:
                for vit in L.VIT_STEMS:
                    z = np.load(os.path.join(L.OUT, pre, "pairs", f"{lm}__{vit}.npz"))
                    Nb = z["null_cka_b"]
                    mu, sd = Nb.mean(axis=0), Nb.std(axis=0, ddof=1)
                    pred = np.sqrt(PR[lm][:, None] * PR[vit][None, :]) / (m - 1)
                    assert pred.shape == mu.shape
                    re = np.abs(mu - pred) / pred
                    relerr.append(re.ravel())
                    pair_med.append(float(np.median(re)))
                    zb.append(((mu - pred) / (sd / np.sqrt(K))).ravel())
                    Nd = z["null_cka_d"]
                    mud, sdd = Nd.mean(axis=0), Nd.std(axis=0, ddof=1)
                    zd.append((mud / (sdd / np.sqrt(K))).ravel())
                    absd.append(np.abs(mud).ravel())
                    Nk = z["null_cnt"].astype(np.float64) / (m * L.KNN)
                    muk, sdk = Nk.mean(axis=0), Nk.std(axis=0, ddof=1)
                    zk.append(((muk - L.KNN / (m - 1)) / (sdk / np.sqrt(K))).ravel())
            relerr, zb, zd, zk, absd = (np.concatenate(x) for x in (relerr, zb, zd, zk, absd))
            res[grp] = {
                "n_entries": int(relerr.size),
                "cka_b": {"median_rel_err": float(np.median(relerr)), "max_rel_err": float(relerr.max()),
                          "p95_rel_err": float(np.percentile(relerr, 95)),
                          "frac_abs_z_gt_4": float(np.mean(np.abs(zb) > 4)), "max_abs_z": float(np.abs(zb).max()),
                          "mean_z": float(zb.mean()), "sd_z": float(zb.std()),
                          "median_over_pairs_of_pair_median_rel_err": float(np.median(pair_med))},
                "cka_d": {"frac_abs_z_gt_4": float(np.mean(np.abs(zd) > 4)), "max_abs_z": float(np.abs(zd).max()),
                          "mean_z": float(zd.mean()), "sd_z": float(zd.std()), "median_abs_null_mean": float(np.median(absd)),
                          "max_abs_null_mean": float(absd.max())},
                "mknn": {"frac_abs_z_gt_4": float(np.mean(np.abs(zk) > 4)), "max_abs_z": float(np.abs(zk).max()),
                         "mean_z": float(zk.mean()), "sd_z": float(zk.std())},
            }
            r = res[grp]
            r["pass"] = bool(r["cka_b"]["median_rel_err"] <= 0.03 and r["cka_b"]["frac_abs_z_gt_4"] <= 0.01
                             and r["cka_d"]["frac_abs_z_gt_4"] <= 0.01 and r["mknn"]["frac_abs_z_gt_4"] <= 0.01)
            log(f"V-L7 {pre} {grp}: cka_b median rel {r['cka_b']['median_rel_err']:.4f} |z|>4 {r['cka_b']['frac_abs_z_gt_4']:.4f}; "
                f"cka_d |z|>4 {r['cka_d']['frac_abs_z_gt_4']:.4f}; mknn |z|>4 {r['mknn']['frac_abs_z_gt_4']:.4f}")
        out[pre] = res
    out["criteria"] = ("pass = biased CKA median relative error <= 3% and fraction |z|>4 <= 1%; debiased CKA and mKNN "
                       "fraction |z|>4 <= 1% (plan V-L7 expectations); z = (null mean - exact expectation)/(sd/sqrt(500))")
    out["pass"] = all(out[p]["new_34"]["pass"] for p in L.PRES)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="vl1,vl4b,vl7,vl2")
    args = ap.parse_args()
    fn = os.path.join(L.OUT, "validation_vl.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for it in args.items.split(","):
        t0 = time.time()
        res[it.upper().replace("VL", "V-L")] = {"vl1": vl1, "vl2": vl2, "vl4b": vl4b, "vl7": vl7}[it]()
        res[it.upper().replace("VL", "V-L")]["secs"] = time.time() - t0
        json.dump(L.jsonable(res), open(fn + ".tmp", "w"), indent=1)
        os.replace(fn + ".tmp", fn)
        log(f"{it} done pass={res[it.upper().replace('VL', 'V-L')].get('pass')}")


if __name__ == "__main__":
    main()
