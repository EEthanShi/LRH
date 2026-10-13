#!/usr/bin/env python
"""Verifier, task 2: identity of hrhv2 against the pinned PRH / ARH / SOG code.

Pinned code (read-only):
  PRH  vendor/platonic-rep/metrics.py (dcd76ba)             -- functions extracted with ast
       (the module imports torchaudio/sklearn, absent from the venv)
  ARH  hrhv2_ref/tests/vendor_arh_785054b/{utils,estimators,preprocess}.py (785054b)
       ~/hrh_finite_20260922/vendor/aristotelian/calibrated_similarity/calibration.py
  SOG  vendor/sog_what_converges_prh (be654a37), imported as a package
       (aristotelian.experiments.layerwise_engine, aristotelian.prh.topology_alignment,
        aristotelian.metrics.{knn,topology,aggregation}, aristotelian.prh.preprocess)

Two input regimes per real-feature case:
  f64    : the vendored functions receive hrhv2's own float64 preprocessed rows
           (implementation identity; SPEC 8 tolerance 1e-6)
  native : the vendored pipeline exactly as PRH/ARH/SOG run it (float32 torch
           preprocessing + metric), compared with hrhv2 float64 (precision gap)

Usage: python vendor_identity.py OUT.json [--calib]
"""
from __future__ import annotations

import ast
import json
import math
import os
import sys
import time

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "hrhv2_ref", "tests"))
sys.path.insert(0, os.path.join(PROJ, "vendor", "sog_what_converges_prh"))
sys.path.insert(0, os.path.expanduser("~/hrh_finite_20260922/vendor/aristotelian"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(int(os.environ.get("VERIF_TORCH_THREADS", "2")))

from hrhv2.prep import Geom, prh_clip_threshold, center_gram  # noqa: E402
from hrhv2 import structural as st, geometric as geo, calibrate as cal, util  # noqa: E402
from agree import REAL_PAIRS, layer_index, FEAT  # noqa: E402

from vendor_arh_785054b import estimators as arh_est, utils as arh_utils, preprocess as arh_pre  # noqa: E402
from calibrated_similarity.calibration import calibrate as arh_calibrate  # noqa: E402
import aristotelian.experiments.layerwise_engine as lwe  # noqa: E402
import aristotelian.prh.topology_alignment as sta  # noqa: E402
import aristotelian.metrics.knn as sknn  # noqa: E402
import aristotelian.metrics.topology as stopo  # noqa: E402
import aristotelian.metrics.aggregation as sagg  # noqa: E402
import aristotelian.prh.preprocess as spre  # noqa: E402
from aristotelian.metrics.utils import knn_indicator as s_knn_indicator  # noqa: E402

PRH = os.path.join(PROJ, "vendor", "platonic-rep", "metrics.py")
TAUS = (1.0, 0.1, 0.01, 0.001, 0.0001)
KS = (1, 3, 10, 30, 100)


def load_prh():
    src = open(PRH).read()
    tree = ast.parse(src)
    ns = {"torch": torch, "np": np}
    wanted = {"hsic_unbiased", "hsic_biased", "compute_nearest_neighbors", "remove_outliers"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), PRH, "exec"), ns)
        if isinstance(node, ast.ClassDef) and node.name == "AlignmentMetrics":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name in ("mutual_knn", "cka"):
                    fn = ast.FunctionDef(name="prh_" + sub.name, args=sub.args, body=sub.body,
                                         decorator_list=[], returns=None, type_comment=None, type_params=[])
                    ast.fix_missing_locations(fn)
                    exec(compile(ast.Module([fn], []), PRH, "exec"), ns)
    return ns


P = load_prh()


def tag(t):
    return ("%g" % t).replace(".", "p")


def rel(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ----------------------------------------------------------------------------
# hrhv2 side
# ----------------------------------------------------------------------------

def hrhv2_side(XA, XB, stackA, stackB, pre):
    A = Geom.from_features(XA, pre, clip_ref=stackA if pre == "prh_clip+l2" else None)
    B = Geom.from_features(XB, pre, clip_ref=stackB if pre == "prh_clip+l2" else None)
    nA, vA = st.knn_table(A.D, 101)
    nB, vB = st.knn_table(B.D, 101)
    out = {"s_N": {k: st.knn_overlap(nA, nB, k) for k in KS}}
    c = geo.cka_pair(A, B)
    KcA, KcB = center_gram(A.K), center_gram(B.K)
    hb_ab, hb_aa, hb_bb = (float(np.vdot(KcA, KcB)), float(np.vdot(KcA, KcA)), float(np.vdot(KcB, KcB)))
    out["cka"] = {"biased": c["cka_biased"], "debiased": c["cka_debiased"], "debiased_eps8": c["cka_debiased_arheps"],
                  "biased_eps6": hb_ab / (math.sqrt(hb_aa * hb_bb) + 1e-6),
                  "biased_eps8": hb_ab / (math.sqrt(hb_aa * hb_bb) + 1e-8),
                  "debiased_eps6": c["hsicu_AB"] / (math.sqrt(c["hsicu_AA"] * c["hsicu_BB"]) + 1e-6)}
    TA, TB = st.mst_prim(A.D), st.mst_prim(B.D)
    sog = geo.sog_overlaps(A.D, B.D, nA, nB, TA, TB, k=10, taus=TAUS)
    out["sog"] = sog
    out["mst_edges_A"], out["mst_edges_B"] = TA, TB
    out["s_T"] = np.intersect1d(TA, TB).size / (A.m - 1)
    out["A"], out["B"], out["nA"], out["nB"] = A, B, nA, nB
    return out


# ----------------------------------------------------------------------------
# vendored side
# ----------------------------------------------------------------------------

def prh_mknn(tA, tB, k):
    return float(P["prh_mutual_knn"](tA, tB, k))


def sog_mknn(tA, tB, k):
    """ARH/SOG MutualKNN._compute_raw (knn.py) on given rows."""
    kA = sknn._compute_knn_indices(tA, k)
    kB = sknn._compute_knn_indices(tB, k)
    n = tA.shape[0]
    mA, mB = s_knn_indicator(kA, n), s_knn_indicator(kB, n)
    return float(((mA & mB).sum(dim=1).float() / float(k)).mean().item())


def vendor_cka(tA, tB):
    KA, KB = tA @ tA.T, tB @ tB.T
    hab = arh_utils.hsic_unbiased(KA, KB)
    haa = arh_utils.hsic_unbiased(KA, KA)
    hbb = arh_utils.hsic_unbiased(KB, KB)
    return {
        "prh_cka_biased(+1e-6)": float(P["prh_cka"](tA, tB, unbiased=False)),
        "prh_cka_unbiased(+1e-6)": float(P["prh_cka"](tA, tB, unbiased=True)),
        "arh_est_cka_biased(+1e-8)": float(arh_est.cka_biased(tA, tB)),
        "arh_est_cka_debiased(Ucentred,+1e-8)": float(arh_est.cka_debiased(tA, tB)),
        "arh_CKAUnbiased_reconstructed(+1e-8)": float((hab / (torch.sqrt(haa * hbb) + arh_utils.EPS)).item()),
    }


CKA_MAP = {  # vendored key -> hrhv2 key with the same epsilon convention
    "prh_cka_biased(+1e-6)": "biased_eps6",
    "prh_cka_unbiased(+1e-6)": "debiased_eps6",
    "arh_est_cka_biased(+1e-8)": "biased_eps8",
    "arh_est_cka_debiased(Ucentred,+1e-8)": "debiased",
    "arh_CKAUnbiased_reconstructed(+1e-8)": "debiased_eps8",
}


def sog_f64(XpA, XpB, m):
    """SOG functions fed hrhv2's float64 L2 rows."""
    tA, tB = torch.from_numpy(XpA), torch.from_numpy(XpB)
    kA = lwe.compute_nearest_neighbors(tA, topk=10)
    kB = lwe.compute_nearest_neighbors(tB, topk=10)
    lA = lwe._knn_log_distance_values_from_layer(tA, kA, q=0.9, distance_normalization="quantile")
    lB = lwe._knn_log_distance_values_from_layer(tB, kB, q=0.9, distance_normalization="quantile")
    out = {"knn": {}, "mst": {}}
    for t in TAUS:
        out["knn"][t] = float(lwe._knn_distance_overlap_from_log_values(kA, kB, lA, lB, topk=10, tau=t).item())
    # MST: SOG dense Prim on the float64 chord-distance matrix (support), values normalized
    dA = torch.cdist(tA, tA).numpy()      # float64 cdist
    dB = torch.cdist(tB, tB).numpy()
    np.fill_diagonal(dA, 0.0)
    np.fill_diagonal(dB, 0.0)
    eA = sta._mst_edge_codes_numpy(dA)
    eB = sta._mst_edge_codes_numpy(dB)
    vA = sta._normalize_distance_array(dA, mode="quantile", q=0.9)
    vB = sta._normalize_distance_array(dB, mode="quantile", q=0.9)
    for t in TAUS:
        out["mst"][t] = sta._mst_distance_overlap_similarity_from_cached(vA, eA, vB, eB, n=m, distance_tau=t)
    out["edges_A"], out["edges_B"] = np.sort(eA), np.sort(eB)
    out["mst_overlap"] = sta._edge_overlap_fraction(eA, eB, m)
    return out


def sog_native(stackA, stackB, la, lb, m):
    """SOG pipeline as run by SOG (float32): prepare_features on the (n, L, d) stack,
    per-layer L2, kNN support by cosine topk, log-distance cache, MST cache via
    torch.cdist(float32)."""
    pA = spre.prepare_features(torch.from_numpy(np.ascontiguousarray(stackA)), q=0.95)
    pB = spre.prepare_features(torch.from_numpy(np.ascontiguousarray(stackB)), q=0.95)
    layA, kA, _ = lwe.build_knn_cache_with_indices(pA[:, la:la + 1, :], topk=10, normalize=True)
    layB, kB, _ = lwe.build_knn_cache_with_indices(pB[:, lb:lb + 1, :], topk=10, normalize=True)
    lA = lwe.build_knn_log_distance_cache(layA, kA, q=0.9, distance_normalization="quantile")
    lB = lwe.build_knn_log_distance_cache(layB, kB, q=0.9, distance_normalization="quantile")
    out = {"knn": {}, "mst": {}, "mknn10": float(lwe._knn_overlap_from_indices(kA[0], kB[0], topk=10).item())}
    for t in TAUS:
        out["knn"][t] = float(lwe._knn_distance_overlap_from_log_values(kA[0], kB[0], lA[0], lB[0], topk=10, tau=t).item())
    cA = sta.prepare_mst_overlap_cache(layA, dist_metric="euclidean", normalize=True, q=0.9,
                                       distance_normalization="quantile", distance_value_geometry="ambient")
    cB = sta.prepare_mst_overlap_cache(layB, dist_metric="euclidean", normalize=True, q=0.9,
                                       distance_normalization="quantile", distance_value_geometry="ambient")
    for t in TAUS:
        out["mst"][t] = sta._mst_edge_value_overlap_from_cached(cA.mst_edges[0], cA.mst_edge_values[0],
                                                                cB.mst_edges[0], cB.mst_edge_values[0],
                                                                n=m, distance_tau=t)
    out["mst_overlap"] = sta._edge_overlap_fraction(cA.mst_edges[0], cB.mst_edges[0], m)
    out["edges_A"], out["edges_B"] = np.sort(cA.mst_edges[0]), np.sort(cB.mst_edges[0])
    out["layers32"] = (layA[0], layB[0])
    return out


# ----------------------------------------------------------------------------
# one real case
# ----------------------------------------------------------------------------

def knn_set_diff(nA_main, t_rows, k):
    """Anchors whose vendored (cosine topk/argsort) k-set differs from hrhv2's (D, j) k-set."""
    kv = sknn._compute_knn_indices(t_rows, k).numpy()
    diff = [i for i in range(kv.shape[0]) if set(kv[i].tolist()) != set(nA_main[i, :k].tolist())]
    return diff


def run_case(a, b, r, pre):
    arrA = np.load(os.path.join(FEAT, a + ".npy"), mmap_mode="r")
    arrB = np.load(os.path.join(FEAT, b + ".npy"), mmap_mode="r")
    la, lb = layer_index(a, r), layer_index(b, r)
    stackA = np.asarray(arrA, dtype=np.float32)
    stackB = np.asarray(arrB, dtype=np.float32)
    s64A, s64B = stackA.astype(np.float64), stackB.astype(np.float64)
    XA, XB = s64A[:, la, :], s64B[:, lb, :]
    m = XA.shape[0]
    rec = {"pair": [a, b], "r": r, "pre": pre, "layers": [la, lb], "m": m}
    H = hrhv2_side(XA, XB, s64A, s64B, pre)
    XpA, XpB = H["A"].X, H["B"].X
    # ---- preprocessing identity ----------------------------------------------
    if pre == "prh_clip+l2":
        pp = {}
        for tagx, s64, s32 in (("A", s64A, stackA), ("B", s64B, stackB)):
            c_main = prh_clip_threshold(s64)
            prh64 = P["remove_outliers"](torch.from_numpy(s64), q=0.95).numpy()
            arh64 = arh_pre.remove_outliers(torch.from_numpy(s64), q=0.95).numpy()
            sog32 = spre.prepare_features(torch.from_numpy(s32), q=0.95).numpy()
            mine = np.clip(s64, -c_main, c_main)
            c32 = float(torch.quantile(torch.from_numpy(s32).abs().flatten(start_dim=1), 0.95, dim=1).mean())
            pp[tagx] = {"c_hrhv2_f64": c_main, "c_vendor_f32": c32, "c_rel_diff_f32": rel(c_main, c32),
                        "prh_remove_outliers_f64_maxabs": float(np.max(np.abs(prh64 - mine))),
                        "arh_remove_outliers_f64_maxabs": float(np.max(np.abs(arh64 - mine))),
                        "sog_prepare_features_f32_maxabs": float(np.max(np.abs(sog32.astype(np.float64) - mine)))}
            # per-layer clip scope would give a different threshold:
            pp[tagx]["c_per_layer_scope"] = prh_clip_threshold(s64[:, la if tagx == "A" else lb, :])
        rec["preprocess"] = pp
    # ---- mKNN ------------------------------------------------------------------
    tA, tB = torch.from_numpy(XpA), torch.from_numpy(XpB)
    if pre == "prh_clip+l2":
        nA32 = spre.prepare_features(torch.from_numpy(stackA), q=0.95)[:, la, :]
        nB32 = spre.prepare_features(torch.from_numpy(stackB), q=0.95)[:, lb, :]
        prhA32 = P["remove_outliers"](torch.from_numpy(stackA), q=0.95)[:, la, :]
        prhB32 = P["remove_outliers"](torch.from_numpy(stackB), q=0.95)[:, lb, :]
    else:
        nA32 = torch.from_numpy(stackA[:, la, :].copy())
        nB32 = torch.from_numpy(stackB[:, lb, :].copy())
        prhA32, prhB32 = nA32, nB32
    nA32, nB32 = F.normalize(nA32, dim=-1), F.normalize(nB32, dim=-1)
    prhA32, prhB32 = F.normalize(prhA32, dim=-1), F.normalize(prhB32, dim=-1)
    mk = {}
    for k in KS:
        main = H["s_N"][k]
        v = {"hrhv2": main,
             "prh_f64": prh_mknn(tA, tB, k), "sog_arh_f64": sog_mknn(tA, tB, k),
             "prh_native_f32": prh_mknn(prhA32, prhB32, k), "sog_arh_native_f32": sog_mknn(nA32, nB32, k)}
        for kk in list(v):
            if kk != "hrhv2":
                v[f"absdiff_{kk}"] = abs(v[kk] - main)
        if any(v[f"absdiff_{x}"] > 0 for x in ("prh_f64", "sog_arh_f64")):
            v["anchors_set_diff_f64_A"] = knn_set_diff(H["nA"], tA, k)[:20]
            v["anchors_set_diff_f64_B"] = knn_set_diff(H["nB"], tB, k)[:20]
        v["n_anchors_set_diff_native_A"] = len(knn_set_diff(H["nA"], nA32, k))
        v["n_anchors_set_diff_native_B"] = len(knn_set_diff(H["nB"], nB32, k))
        mk[k] = v
    rec["mknn"] = mk
    # ---- CKA -----------------------------------------------------------------
    v64 = vendor_cka(tA, tB)
    v32 = vendor_cka(nA32, nB32)
    ck = {}
    for kv, km in CKA_MAP.items():
        ck[kv] = {"hrhv2_key": km, "hrhv2": H["cka"][km], "vendor_f64": v64[kv], "vendor_native_f32": v32[kv],
                  "absdiff_f64": abs(v64[kv] - H["cka"][km]), "absdiff_native_f32": abs(v32[kv] - H["cka"][km])}
    ck["eps_effects"] = {"debiased_eps8_minus_exact": H["cka"]["debiased_eps8"] - H["cka"]["debiased"],
                         "debiased_eps6_minus_exact": H["cka"]["debiased_eps6"] - H["cka"]["debiased"],
                         "biased_eps6_minus_exact": H["cka"]["biased_eps6"] - H["cka"]["biased"]}
    rec["cka"] = ck
    # ---- SOG -----------------------------------------------------------------
    if pre == "prh_clip+l2":
        S64 = sog_f64(XpA, XpB, m)
        S32 = sog_native(stackA, stackB, la, lb, m)
        sg = {"knn": {}, "mst": {}}
        for t in TAUS:
            mk_ = H["sog"][f"sog_knn10_tau{tag(t)}"]
            ms_ = H["sog"][f"sog_mst_tau{tag(t)}"]
            sg["knn"][t] = {"hrhv2": mk_, "sog_f64": S64["knn"][t], "sog_native_f32": S32["knn"][t],
                            "absdiff_f64": abs(S64["knn"][t] - mk_), "absdiff_native": abs(S32["knn"][t] - mk_)}
            sg["mst"][t] = {"hrhv2": ms_, "sog_f64": S64["mst"][t], "sog_native_f32": S32["mst"][t],
                            "absdiff_f64": abs(S64["mst"][t] - ms_), "absdiff_native": abs(S32["mst"][t] - ms_)}
        sg["mst_overlap"] = {"hrhv2_s_T": H["s_T"], "sog_f64": S64["mst_overlap"], "sog_native_f32": S32["mst_overlap"]}
        sg["mst_edges_identical_f64"] = {"A": bool(np.array_equal(S64["edges_A"], H["mst_edges_A"])),
                                         "B": bool(np.array_equal(S64["edges_B"], H["mst_edges_B"]))}
        sg["mst_edges_ndiff_native"] = {"A": int(np.setdiff1d(H["mst_edges_A"], S32["edges_A"]).size),
                                        "B": int(np.setdiff1d(H["mst_edges_B"], S32["edges_B"]).size)}
        sg["mknn10_native_sog_vs_hrhv2"] = abs(S32["mknn10"] - H["s_N"][10])
        # D vs d check: the same overlap with squared distances D (Q over positive D) in the log-ratio
        A_, B_ = H["A"], H["B"]
        a10, b10 = H["nA"][:, :10], H["nB"][:, :10]
        ii, jj = [], []
        for i in range(m):
            cmn = np.intersect1d(a10[i], b10[i])
            ii.extend([i] * cmn.size)
            jj.extend(cmn.tolist())
        ii, jj = np.array(ii), np.array(jj)
        QA2, QB2 = geo.sog_scale(A_.D), geo.sog_scale(B_.D)
        dd = {}
        for t in TAUS:
            wD = geo.sog_weights(A_.D[ii, jj], B_.D[ii, jj], QA2, QB2, t).sum() / (m * 10)
            dd[t] = {"plain_d(hrhv2)": H["sog"][f"sog_knn10_tau{tag(t)}"], "squared_D_variant": float(wD),
                     "sog_f64": S64["knn"][t]}
        sg["d_vs_D"] = dd
        rec["sog"] = sg
    return rec


# ----------------------------------------------------------------------------
# calibration identity
# ----------------------------------------------------------------------------

class PermFeeder:
    def __init__(self, m, key, K):
        self.perms = [torch.from_numpy(cal.permutation(m, key, k)) for k in range(K)]
        self.i = 0

    def __call__(self, n, device):
        p = self.perms[self.i]
        self.i += 1
        return p


def calib_case(a, b, r, K=500):
    arrA = np.load(os.path.join(FEAT, a + ".npy"), mmap_mode="r")
    arrB = np.load(os.path.join(FEAT, b + ".npy"), mmap_mode="r")
    la, lb = layer_index(a, r), layer_index(b, r)
    s64A, s64B = np.asarray(arrA, dtype=np.float64), np.asarray(arrB, dtype=np.float64)
    A = Geom.from_features(s64A[:, la, :], "prh_clip+l2", clip_ref=s64A)
    B = Geom.from_features(s64B[:, lb, :], "prh_clip+l2", clip_ref=s64B)
    m = A.m
    key = f"VERIF1|calib|{a}|{b}|r={r}"
    out = {"pair": [a, b], "r": r, "K": K, "alpha": 0.05}
    # mKNN k=10: hrhv2 fast path vs ARH calibrate() with PRH mutual_knn recomputed per permutation
    nA, _ = st.knn_table(A.D, 11)
    nB, _ = st.knn_table(B.D, 11)
    t0 = time.time()
    hm = cal.calibrate_knn(nA, nB, 10, key=key, K=K, alpha=0.05)
    tA, tB = torch.from_numpy(A.X), torch.from_numpy(B.X)
    feeder = PermFeeder(m, key, K)
    null_arh = []

    def sim(X, Y):
        v = prh_mknn(X, Y, 10)
        null_arh.append(v)
        return v
    sc, p, tau = arh_calibrate(tA, tB, sim, K=K, alpha=0.05, smax=1.0, perm_fn=feeder)
    null_arh = np.array(null_arh[1:])
    sog_sum = sagg.compute_null_summary(null_arh, T_obs=hm["s_obs"], alpha=0.05)
    out["mknn10"] = {
        "hrhv2": {k: hm[k] for k in ("s_obs", "c_alpha", "s_cal", "p_value", "uncensored")},
        "arh_calibrate": {"s_cal": float(sc), "p_value": float(p), "c_alpha": float(tau)},
        "sog_arh_compute_null_summary": {"tau_alpha": sog_sum["tau_alpha"], "p_value": sog_sum["p_value"],
                                         "g_score": sagg.gated_rescaled(hm["s_obs"], tau_alpha=sog_sum["tau_alpha"], s_max=1.0)},
        "time_s": time.time() - t0,
    }
    # recompute hrhv2 null explicitly to compare elementwise
    hnull = []
    for kk in range(K):
        pi = cal.permutation(m, key, kk)
        inv = np.empty(m, dtype=np.int64)
        inv[pi] = np.arange(m)
        hnull.append(st.knn_overlap(nA[:, :10], inv[nB[pi, :10]], 10))
    hnull = np.array(hnull)
    out["mknn10"]["null_max_absdiff_hrhv2_vs_arh_recompute"] = float(np.max(np.abs(hnull - null_arh)))
    d = out["mknn10"]
    d["absdiff_s_cal"] = abs(d["hrhv2"]["s_cal"] - d["arh_calibrate"]["s_cal"])
    d["absdiff_c_alpha"] = abs(d["hrhv2"]["c_alpha"] - d["arh_calibrate"]["c_alpha"])
    d["absdiff_p"] = abs(d["hrhv2"]["p_value"] - d["arh_calibrate"]["p_value"])
    d["absdiff_c_alpha_vs_sog_summary"] = abs(d["hrhv2"]["c_alpha"] - sog_sum["tau_alpha"])
    # CKA biased: hrhv2.calibrate (Geom permuted, recomputed) vs ARH calibrate with ARH estimators.cka_biased
    t0 = time.time()
    hc = cal.calibrate(lambda X, Y: geo.cka_pair(X, Y)["cka_biased"], A, B, key=key + "|cka", K=K, alpha=0.05)
    feeder = PermFeeder(m, key + "|cka", K)
    sc, p, tau = arh_calibrate(tA, tB, lambda X, Y: arh_est.cka_biased(X, Y), K=K, alpha=0.05, smax=1.0, perm_fn=feeder)
    out["cka_biased"] = {"hrhv2": {k: hc[k] for k in ("s_obs", "c_alpha", "s_cal", "p_value", "uncensored")},
                         "arh_calibrate": {"s_cal": float(sc), "p_value": float(p), "c_alpha": float(tau)},
                         "time_s": time.time() - t0}
    d = out["cka_biased"]
    d["absdiff_s_cal"] = abs(d["hrhv2"]["s_cal"] - d["arh_calibrate"]["s_cal"])
    d["absdiff_c_alpha"] = abs(d["hrhv2"]["c_alpha"] - d["arh_calibrate"]["c_alpha"])
    d["absdiff_p"] = abs(d["hrhv2"]["p_value"] - d["arh_calibrate"]["p_value"])
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if not isinstance(v, (Geom, torch.Tensor)) and not (isinstance(v, np.ndarray) and v.size > 50)}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    outp = sys.argv[1]
    res = {"cases": [], "calibration": []}
    t0 = time.time()
    for a, b in REAL_PAIRS:
        for r in (0.5, 1.0):
            for pre in ("prh_clip+l2", "l2"):
                rec = run_case(a, b, r, pre)
                res["cases"].append(jsonable(rec))
                print(f"[{time.time() - t0:6.0f}s] case {a} {b} r={r} {pre}", flush=True)
                json.dump(res, open(outp + ".tmp", "w"), indent=1)
    if "--calib" in sys.argv:
        for a, b in REAL_PAIRS[:2]:
            res["calibration"].append(jsonable(calib_case(a, b, 1.0)))
            print(f"[{time.time() - t0:6.0f}s] calib {a} {b}", flush=True)
            json.dump(res, open(outp + ".tmp", "w"), indent=1)
    json.dump(res, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
