#!/usr/bin/env python
"""PYT-AXIS COPY of analysis/s12_layermax_openllama/indep/vl_checks.py (sha256 ee34896e...), sub-role pyt-code-I (S0b,
2026-10-10): validation items V-P1, V-P2, V-P4 and V-P7 (a), (c), (d) of ANALYSIS_PLAN_PYTHIA.md section 11, I side.
(V-P3, V-P7(b) = validate.py; V-P6 = vl6_arh.py; V-P5 = cross_validate.py; V-P8 = vp8_legacy.py; V-P9 = tests/.)
Changes (marked PYT-AXIS): order gate first (plan F1(a); the script reads pair files); PYT_THREADS; the 7 Pythia
models are new (reference clip: partB/clip_thresholds_partB_pyt.json); V-O4b (rerun of the OpenLLaMA pilots) is
replaced by V-P4 (rerun of the 4 known pairs of plan 11 against each pair's frozen I counterpart: mKNN counts and p
identical, CKA <= 1e-12 relative); PR from partB_features_pyt.json (field per_pre[pre]["PR"][stem], per stored layer);
V-P7 group "pythia_119" decides; new item vl7d = V-P7(d), the debiased-CKA second moment (descriptive); outputs are
kept aside, never overwritten; options --pres, --pairs-root, --K-test, --partb, --vp7-pairs (S0b tests only).

usage: launch_pyt.sh pyt_I_vl 2 -- venv/bin/python analysis/s12_layermax_pythia/indep/vl_checks.py
       [--items vl1,vl2,vp4,vl7,vl7c,vl7d]   (CPU only; writes indep/validation_vl.json, indep/cache/)
Items:
  V-P1 (vl1)  clip thresholds used by the run (prep_meta.json): recomputed all-layer threshold == reference bitwise
              (frozen Part B clip_thresholds_partB_pyt.json for the 7 Pythia models, the S1.2 cache for the ViTs).
              Writes indep/cache/clip_thresholds_I.json.
  V-P2 (vl2)  kNN tables (k+1 = 11) of every stored layer of the 7 Pythia models and the 17 ViTs, both pre, with
              hrhv2.structural.knn_table; ViT tables == the frozen layer-max caches; exact k = 10 boundary ties and
              SPEC near-tie fractions per layer; ViT boundary ties must be 0. V-P2b: every observed count matrix of the
              run (all layer pairs, 119 pairs, both pre) == the overlap counts of these tables.
  V-P4 (vp4)  the 4 known pairs rerun by run_layermax.py --known vs their frozen I files (results/S1_2_layermax/indep,
              results/S1_2_layermax_llama/indep, results/S1_2_layermax_openllama/indep): raw CKA arrays <= 1e-12
              relative, observed and null mKNN counts identical, and per-subset p identical / floats <= 1e-12 relative.
  V-P7 (vl7)  Proposition 5 on the new pairs: entry-wise permutation null means vs sqrt(PR_u PR_v)/(m-1) (biased CKA),
              0 (debiased CKA) and 10/1023 (mKNN); median relative error <= 3% and |z| > 4 in <= 1% of entries.
       (vl7c) the m = 8 known-answer plant over all 8! permutations with the run's debiased-CKA arithmetic (<= 1e-12).
       (vl7d) the mean over entries of (mean_k dCKA^2) / (2/(m(m-3))) with its MC error (descriptive).
Output: results/S1_2_layermax_pythia/indep/validation_vl.json
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True   # PYT-AXIS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyt_gate  # noqa: E402  PYT-AXIS
OUT_ROOT = pyt_gate.gate(__file__)   # PYT-AXIS F1(a): exits 3 unless the plan's gates hold (S0b test mode: test root)

_T = os.environ.get("PYT_THREADS", "1")   # PYT-AXIS
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import lmx_common as L  # noqa: E402
from lmx_common import C, st  # noqa: E402
from hrhv2.prep import gram_to_dist, median_offdiag  # noqa: E402

KMAX = L.KNN + 1
FROZEN_KNN = os.path.join(L.PROJ, "results", "S1_2_layermax", "cache")
PARTB = os.path.join(os.path.dirname(C.CLIP_PARTB), "partB_features_pyt.json")   # frozen Part B (test: test root)
NEW_LMS = tuple(L.LM_STEMS)   # PYT-AXIS: the 7 Pythia models
PAIRS_ROOT = L.OUT   # --pairs-root (S0b tests)
PRES = L.PRES        # --pres
K_TEST = None        # --K-test (S0b tests)
VP7_PAIRS = None     # --vp7-pairs (S0b tests)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mixed_ok(a, b):
    """V-P5 tolerance |a - b| <= 1e-12 max(|a|, |b|) + 1e-15 (reported next to the strict relative one)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.all(np.abs(a - b) <= 1e-12 * np.maximum(np.abs(a), np.abs(b)) + 1e-15))


def relmax(a, b, floor=1e-300):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    s = np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)
    return float(np.max(d / s)) if d.size else 0.0


# --------------------------------------------------------------------------- V-P1
def vl1():
    out = {"per_model": {}}
    meta = json.load(open(os.path.join(PAIRS_ROOT, "prh_clip+l2", "prep_meta.json")))
    for stem, m in meta.items():
        out["per_model"][stem] = {"c_recomputed": m["clip_recomputed"], "c_reference": m["clip_s12_cache"],
                                  "reference": ("partB/clip_thresholds_partB_pyt.json" if stem in NEW_LMS
                                                else "frozen reference (Part B file or results/S1_2/cache)"),
                                  "bitwise_equal": bool(m["clip_bitwise_equal"])}
    out["n_models"] = len(out["per_model"])
    out["n_expected"] = len(NEW_LMS) + len(L.VIT_STEMS)
    out["all_bitwise_equal"] = all(v["bitwise_equal"] for v in out["per_model"].values())
    out["pass"] = bool(out["all_bitwise_equal"] and out["n_models"] == out["n_expected"])
    os.makedirs(os.path.join(L.OUT, "cache"), exist_ok=True)
    L.write_json(os.path.join(L.OUT, "cache", "clip_thresholds_I.json"),
                 {"note": "Pythia axis, implementation I; hrhv2.prep.prh_clip_threshold on the float64 all-layer stack "
                          "(D5); recomputed inside run_layermax.prep_models",
                  "clip": {s: v["c_recomputed"] for s, v in out["per_model"].items()}})
    return out


# --------------------------------------------------------------------------- V-P2
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
    for pre in PRES:
        for stem in L.LM_STEMS + L.VIT_STEMS:
            t0 = time.time()
            idx, rows = tables(stem, pre)
            tabs[(stem, pre)] = idx
            os.makedirs(os.path.join(L.OUT, "cache", "knn_hrhv2"), exist_ok=True)
            fn = os.path.join(L.OUT, "cache", "knn_hrhv2", f"{stem}__{pre}.npy")
            L.keep_aside(fn)   # PYT-AXIS
            np.save(fn, idx)
            out["per_table"][f"{stem}|{pre}"] = rows
            if stem not in NEW_LMS:
                cmp = {}
                for which in ("knn_hrhv2", "knn_final"):
                    fz = np.load(os.path.join(FROZEN_KNN, which, f"{stem}__{pre}.npy"))
                    cmp[f"{which}_identical_11cols"] = bool(np.array_equal(fz, idx))
                    cmp[f"{which}_identical_k10"] = bool(np.array_equal(fz[:, :, :L.KNN], idx[:, :, :L.KNN]))
                out["frozen_compare"][f"{stem}|{pre}"] = cmp
            log(f"V-P2 {stem} {pre} L={idx.shape[1]} ties={sum(r['exact_boundary_tie_rows_k10'] for r in rows)} "
                f"{time.time() - t0:.1f}s")
        n_bad, n_tot = 0, 0
        for lm in L.LM_STEMS:
            for vit in L.VIT_STEMS:
                z = np.load(os.path.join(PAIRS_ROOT, pre, "pairs", f"{lm}__{vit}.npz"))
                ref = overlap_matrix(tabs[(lm, pre)], tabs[(vit, pre)])
                obs = z["obs_cnt"].astype(np.int64)
                n_bad += int(np.sum(obs != ref))
                n_tot += int(obs.size)
        out["obs_count_check"][pre] = {"n_layer_pairs": n_tot, "n_differ": n_bad, "identical": n_bad == 0}
        log(f"V-P2b {pre}: {n_tot} layer pairs, {n_bad} differ")
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


# --------------------------------------------------------------------------- V-P4 (PYT-AXIS, replaces V-O4b)
def vp4():
    import lmx_pyt_aggregate as AG   # pair_record (its module gate is the same gate, already passed)
    out = {}
    for pre in PRES:
        raw = {"max_rel": {}, "max_abs": {}, "mixed_tol_ok": True, "counts_identical": True, "key_K_equal": True,
               "n_pairs": 0}
        summ = {"max_rel": 0.0, "p_identical": True, "argmax_identical": True}
        for lm, vit in C.KNOWN_VP4:
            fn = f"{lm}__{vit}.npz"
            z = np.load(os.path.join(PAIRS_ROOT, "known", pre, "pairs", fn))
            zo = np.load(os.path.join(L.frozen_indep_root(lm), pre, "pairs", fn))
            Kz = int(z["K"])
            raw["n_pairs"] += 1
            raw["key_K_equal"] &= bool(str(z["key"]) == str(zo["key"]) and (Kz == int(zo["K"]) or K_TEST is not None))
            for f_ in ("obs_cka_b", "obs_cka_d", "null_cka_b", "null_cka_d"):
                a = np.asarray(z[f_], np.float64)
                b = np.asarray(zo[f_], np.float64)[:Kz] if f_.startswith("null") else np.asarray(zo[f_], np.float64)
                raw["max_rel"][f_] = max(raw["max_rel"].get(f_, 0.0), relmax(a, b))
                raw["max_abs"][f_] = max(raw["max_abs"].get(f_, 0.0), float(np.max(np.abs(a - b))))
                raw["mixed_tol_ok"] = bool(raw["mixed_tol_ok"] and mixed_ok(a, b))
            raw["counts_identical"] &= bool(np.array_equal(z["obs_cnt"].astype(np.int64), zo["obs_cnt"].astype(np.int64))
                                            and np.array_equal(z["null_cnt"].astype(np.int64),
                                                               zo["null_cnt"][:Kz].astype(np.int64)))
            if K_TEST is None:   # per-subset summaries need all 500 permutations
                ra, rb = AG.pair_record(z), AG.pair_record(zo)
                for met in AG.METRICS:
                    for q in ra[met]:
                        for fld, va in ra[met][q].items():
                            vb = rb[met][q][fld]
                            if fld in ("a", "b", "unc_a", "unc_b", "n_tied_max", "n_entries", "n_entries_scal_pos"):
                                summ["argmax_identical"] &= bool(va == vb)
                            elif fld == "p":
                                summ["p_identical"] &= bool(va == vb)
                            elif isinstance(va, (int, float)) or hasattr(va, "numerator"):
                                summ["max_rel"] = max(summ["max_rel"], relmax(float(va), float(vb)))
        ok_raw = bool(raw["counts_identical"] and raw["key_K_equal"] and max(raw["max_rel"].values()) <= 1e-12
                      and raw["n_pairs"] == len(C.KNOWN_VP4))
        ok_mixed = bool(raw["counts_identical"] and raw["key_K_equal"] and raw["mixed_tol_ok"]
                        and raw["n_pairs"] == len(C.KNOWN_VP4))
        ok_sum = (None if K_TEST is not None else
                  bool(summ["p_identical"] and summ["argmax_identical"] and summ["max_rel"] <= 1e-12))
        # plan V-P4 decides with the strict relative 1e-12 (pass); the V-P5 mixed form (1e-12 max + 1e-15) is reported
        # next to it, because debiased-CKA null entries near 0 make a pure relative bound fragile across devices
        out[pre] = {"raw_vs_frozen_I": raw, "summary_vs_frozen_I": summ if K_TEST is None else "skipped (K-test)",
                    "K_test": K_TEST, "pass": bool(ok_raw and (ok_sum is not False)),
                    "pass_mixed_tolerance_V-P5_form": bool(ok_mixed and (ok_sum is not False))}
        log(f"V-P4 {pre}: pass={out[pre]['pass']} (mixed form {out[pre]['pass_mixed_tolerance_V-P5_form']}) "
            f"raw max_rel={max(raw['max_rel'].values()):.2e}")
    out["pass"] = all(out[p]["pass"] for p in PRES)
    out["pass_mixed_tolerance_V-P5_form"] = all(out[p]["pass_mixed_tolerance_V-P5_form"] for p in PRES)
    return out


# --------------------------------------------------------------------------- V-P7
def _vp7_pairs():
    return VP7_PAIRS or [(lm, v) for lm in L.LM_STEMS for v in L.VIT_STEMS]


def vl7():
    pb = json.load(open(PARTB))
    m = L.M
    out = {}
    for pre in PRES:
        PR = {s: np.array(v, dtype=np.float64) for s, v in pb["per_pre"][pre]["PR"].items()}
        relerr, zb, zd, zk, absd, pair_med = [], [], [], [], [], []
        npairs = 0
        for lm, vit in _vp7_pairs():
            sub = "known" if VP7_PAIRS else None
            z = np.load(os.path.join(PAIRS_ROOT, sub, pre, "pairs", f"{lm}__{vit}.npz") if sub else
                        os.path.join(PAIRS_ROOT, pre, "pairs", f"{lm}__{vit}.npz"))
            K = int(z["K"])
            npairs += 1
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
            with np.errstate(divide="ignore", invalid="ignore"):
                zk.append(((muk - L.KNN / (m - 1)) / (sdk / np.sqrt(K))).ravel())
        relerr, zb, zd, zk, absd = (np.concatenate(x) for x in (relerr, zb, zd, zk, absd))
        zk = zk[np.isfinite(zk)]
        r = {"group": "known_test_pairs" if VP7_PAIRS else "pythia_119", "n_pairs": npairs,
             "n_entries": int(relerr.size),
             "cka_b": {"median_rel_err": float(np.median(relerr)), "max_rel_err": float(relerr.max()),
                       "p95_rel_err": float(np.percentile(relerr, 95)),
                       "frac_abs_z_gt_4": float(np.mean(np.abs(zb) > 4)), "max_abs_z": float(np.abs(zb).max()),
                       "mean_z": float(zb.mean()), "sd_z": float(zb.std()),
                       "median_over_pairs_of_pair_median_rel_err": float(np.median(pair_med))},
             "cka_d": {"frac_abs_z_gt_4": float(np.mean(np.abs(zd) > 4)), "max_abs_z": float(np.abs(zd).max()),
                       "mean_z": float(zd.mean()), "sd_z": float(zd.std()), "median_abs_null_mean": float(np.median(absd)),
                       "max_abs_null_mean": float(absd.max())},
             "mknn": {"frac_abs_z_gt_4": float(np.mean(np.abs(zk) > 4)) if zk.size else None,
                      "max_abs_z": float(np.abs(zk).max()) if zk.size else None,
                      "mean_z": float(zk.mean()) if zk.size else None}}
        r["pass"] = bool(r["cka_b"]["median_rel_err"] <= 0.03 and r["cka_b"]["frac_abs_z_gt_4"] <= 0.01
                         and r["cka_d"]["frac_abs_z_gt_4"] <= 0.01
                         and (r["mknn"]["frac_abs_z_gt_4"] is None or r["mknn"]["frac_abs_z_gt_4"] <= 0.01))
        log(f"V-P7 {pre}: cka_b median rel {r['cka_b']['median_rel_err']:.4f} |z|>4 {r['cka_b']['frac_abs_z_gt_4']:.4f}; "
            f"cka_d |z|>4 {r['cka_d']['frac_abs_z_gt_4']:.4f}; mknn |z|>4 {r['mknn']['frac_abs_z_gt_4']}")
        out[pre] = r
    out["criteria"] = ("pass = biased CKA median relative error <= 3% and fraction |z|>4 <= 1%; debiased CKA and mKNN "
                       "fraction |z|>4 <= 1% (plan V-P7); z = (null mean - exact expectation)/(sd/sqrt(K))")
    out["pass"] = all(out[p]["pass"] for p in PRES)
    return out


def vl7c(n_pairs=5, m=8):
    """plan V-P7(c): known-answer plant. 5 random pairs of m = 8 Gram matrices; debiased CKA computed with
    the arithmetic of run_layermax.run_pair (centred Gram, H - Dd + t2 - t3 over m(m-3), denominators
    sqrt(HSIC_u(A,A) HSIC_u(B,B)) from hrhv2.geometric.hsic_unbiased) for every one of the 8! = 40320
    permutations of B's rows; the average over all permutations must be 0 within 1e-12 absolute."""
    from hrhv2 import util, geometric as geo
    from hrhv2.prep import center_gram
    rng = util.rng_from_key("S1.2-layermax-pythia|V-P7c|I")   # PYT-AXIS
    P = np.array(list(itertools.permutations(range(m))), dtype=np.int64)
    rows = []
    for t in range(n_pairs):
        XA, XB = rng.standard_normal((m, 6)), rng.standard_normal((m, 4))
        A, B = center_gram(XA @ XA.T), center_gram(XB @ XB.T)
        dA, dB = np.diag(A).copy(), np.diag(B).copy()
        colA, rowB = A.sum(axis=0) - dA, B.sum(axis=1) - dB
        t2 = colA.sum() * rowB.sum() / ((m - 1) * (m - 2))
        Bp = B[P[:, :, None], P[:, None, :]]                       # (8!, m, m): B[pi_i, pi_j]
        H = np.einsum("ij,pij->p", A, Bp)
        Dd = dB[P] @ dA
        t3 = 2.0 * (rowB[P] @ colA) / (m - 2)
        hu = ((H - Dd) + t2 - t3) / (m * (m - 3))
        haa, hbb = float(geo.hsic_unbiased(A, A)), float(geo.hsic_unbiased(B, B))
        den = np.sqrt(haa * hbb) if haa * hbb > 0 else float("nan")
        dcka = hu / den
        ref = [float(geo.hsic_unbiased(A, B[np.ix_(P[k], P[k])])) for k in (0, 1, 12345, len(P) - 1)]
        rows.append({"pair": t, "hsic_aa": haa, "hsic_bb": hbb, "mean_dcka_over_8fact": float(np.mean(dcka)),
                     "mean_hsicu_over_8fact": float(np.mean(hu)), "max_abs_dcka": float(np.max(np.abs(dcka))),
                     "run_formula_vs_hrhv2_hsicu_maxabs": float(max(abs(a - b) for a, b in
                                                                  zip(ref, hu[[0, 1, 12345, len(P) - 1]])))})
    worst = max(abs(r["mean_dcka_over_8fact"]) for r in rows)
    return {"m": m, "n_perm": int(len(P)), "n_pairs": n_pairs, "rng_key": "S1.2-layermax-pythia|V-P7c|I",
            "rows": rows, "max_abs_mean_dcka": worst,
            "max_run_formula_vs_hrhv2": max(r["run_formula_vs_hrhv2_hsicu_maxabs"] for r in rows),
            "pass": bool(worst <= 1e-12 and all(np.isfinite(r["mean_dcka_over_8fact"]) for r in rows))}


def vl7d():
    """PYT-AXIS plan V-P7(d) (descriptive): second moment of debiased CKA under permutation, 2/(m(m-3)) (v4,
    Proposition 5(3)); ratio = mean_k dCKA^2 / (2/(m(m-3))) per entry; mean over entries with its MC error
    (sd over entries of the per-entry MC standard error of mean_k dCKA^2, divided by sqrt(#entries); entries are
    correlated, so this error is optimistic)."""
    m = L.M
    ref = 2.0 / (m * (m - 3))
    out = {"reference_second_moment": ref}
    for pre in PRES:
        ratios, ses, pair_means = [], [], []
        for lm, vit in _vp7_pairs():
            sub = "known" if VP7_PAIRS else None
            z = np.load(os.path.join(PAIRS_ROOT, sub, pre, "pairs", f"{lm}__{vit}.npz") if sub else
                        os.path.join(PAIRS_ROOT, pre, "pairs", f"{lm}__{vit}.npz"))
            d2 = np.asarray(z["null_cka_d"], np.float64) ** 2
            K = d2.shape[0]
            ratios.append((d2.mean(axis=0) / ref).ravel())
            ses.append((d2.std(axis=0, ddof=1) / np.sqrt(K) / ref).ravel())
            pair_means.append(float(ratios[-1].mean()))
        r, s = np.concatenate(ratios), np.concatenate(ses)
        mean = float(r.mean())
        se = float(np.sqrt(np.mean(s ** 2)) / np.sqrt(r.size))
        pm = np.array(pair_means)
        se_p = float(pm.std(ddof=1) / np.sqrt(pm.size)) if pm.size > 1 else None
        out[pre] = {"n_entries": int(r.size), "n_pairs": int(pm.size), "mean_ratio": mean,
                    "median_ratio": float(np.median(r)), "mc_se_of_mean_entries": se,
                    "z_entries": (mean - 1.0) / se if se > 0 else None,
                    "mean_of_pair_means": float(pm.mean()), "se_over_pairs": se_p,
                    "z_pairs": ((float(pm.mean()) - 1.0) / se_p) if se_p else None,
                    "note": "entries within a pair are correlated: z_entries is optimistic; z_pairs treats pairs as units",
                    "label": "descriptive"}
        log(f"V-P7(d) {pre}: mean ratio {mean:.4f} (entry se {se:.4f}; pair se {se_p})")
    return out


def main():
    global PAIRS_ROOT, PRES, K_TEST, PARTB, VP7_PAIRS
    ap = argparse.ArgumentParser(description="V-P1 (vl1), V-P2 (vl2), V-P4 (vp4), V-P7 (a) (vl7), (c) (vl7c), (d) (vl7d)")
    ap.add_argument("--items", default="vl1,vp4,vl7,vl7c,vl7d,vl2")
    ap.add_argument("--pres", default=",".join(L.PRES))
    ap.add_argument("--pairs-root", default=None, help="S0b test mode only")
    ap.add_argument("--K-test", type=int, default=None, help="S0b test mode only: compare the first K permutations")
    ap.add_argument("--partb", default=None, help="S0b test mode only: Part B json with per_pre[pre]['PR']")
    ap.add_argument("--vp7-pairs", default=None, help="S0b test mode only: 'known' = run V-P7 on the V-P4 known pairs")
    args = ap.parse_args()
    PRES = tuple(args.pres.split(","))
    if (args.pairs_root or args.K_test or args.partb or args.vp7_pairs) and not pyt_gate.test_mode():
        raise SystemExit("--pairs-root / --K-test / --partb / --vp7-pairs are S0b test options")
    if args.pairs_root:
        PAIRS_ROOT = args.pairs_root
    K_TEST = args.K_test
    if args.partb:
        PARTB = args.partb
    if args.vp7_pairs == "known":
        VP7_PAIRS = list(C.KNOWN_VP4)
    fn = os.path.join(L.OUT, "validation_vl.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    res["test_mode"] = bool(pyt_gate.test_mode())
    for it in args.items.split(","):
        t0 = time.time()
        name = {"vl1": "V-P1", "vl2": "V-P2", "vp4": "V-P4", "vl7": "V-P7a", "vl7c": "V-P7c", "vl7d": "V-P7d"}[it]
        res[name] = {"vl1": vl1, "vl2": vl2, "vp4": vp4, "vl7": vl7, "vl7c": vl7c, "vl7d": vl7d}[it]()
        res[name]["secs"] = time.time() - t0
        L.write_json(fn, L.jsonable(res))   # PYT-AXIS: earlier copy kept aside
        log(f"{it} done pass={res[name].get('pass')}")


if __name__ == "__main__":
    main()
