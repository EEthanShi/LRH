#!/usr/bin/env python
"""Independent re-derivation of the OpenLLaMA-axis Part B numbers that involve open_llama_13b
(role oll-axis-predict, step S5, written AFTER partb_ol.py ran and BEFORE the Part B freeze).

Not part of the S0b code freeze: a read-only cross-check of partb_ol.py's output with separately written
arithmetic. It changes no Part B number; its sha256 is logged in the Part B FROZEN STATUS line.

FEATURES ONLY, plus the frozen 3B/7B layer-max values. Opens:
  features/wit1024_openllama_axis/openlm-research__open_llama_13b.npy        (13B features)
  results/S1_2_layermax_openllama/partB/partB_features_ol.json               (output under check)
  results/S1_2_layermax_openllama/partB/clip_thresholds_partB_ol.json        (output under check)
  results/S1_2_layermax/pairs/<pre>/openlm-research__open_llama_{3b,7b}__<vit>.json  (frozen, known pairs)
No cross-model score involving open_llama_13b exists or is computed here (single-model PRs and products of
single-model numbers only).

Independent paths (different arithmetic from partb_ol.py):
  V1 clip threshold: per-sample 0.95 quantile by np.partition + explicit linear interpolation, math.fsum mean
  V2 per-layer PR from the eigenvalues of the column-centred Gram Xc Xc^T (eigvalsh), not from HKH traces
  V3 Bbar / B_fd / B_late of the 3 OpenLLaMA LMs x 17 ViTs from the PR lists, brute-force max over layer pairs
  V4 Tr_O by explicit concordance counting (3 sizes: C = 3, 2, 1, 0 -> 1, 1/2, -1/2, -1) and B4 materiality
  V5 kappa^lo / kappa^hi (s_cal), uncensored boundaries and the h-curve, from the frozen pair JSON files
  V6 the required fields of VERDICTS_SPEC section 8 are present
Output: results/S1_2_layermax_openllama/partB/partB_verify_ol.json
RUN: analysis/s12_layermax_openllama/launch_ol.sh pb_verify 4 -- venv/bin/python \
         analysis/s12_layermax_openllama/partB_verify/verify_partb_ol.py
"""
from __future__ import annotations

import os

_TH = os.environ.get("OL_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _TH

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
RES = os.path.join(PROJ, "results", "S1_2_layermax_openllama")
PB = os.path.join(RES, "partB")
F13 = os.path.join(PROJ, "features", "wit1024_openllama_axis", "openlm-research__open_llama_13b.npy")
FROZEN8 = os.path.join(PROJ, "results", "S1_2_layermax", "pairs")
OL = ["openlm-research__open_llama_3b", "openlm-research__open_llama_7b", "openlm-research__open_llama_13b"]
PRES = ("prh_clip+l2", "l2")
M = 1024
DELTA_B = 0.005
BAND = (0.0009, 0.0029)
O = 0.0019
MU = (-0.0002, 0.0002)
HS = (0.0003, 0.0005, 0.001, 0.0015, 0.002)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def clip_indep(st):
    """V1: mean over samples of the linear-interpolated 0.95 quantile of |x| over all layers and coordinates."""
    q = 0.95
    vals = []
    for i in range(st.shape[0]):
        a = np.abs(np.asarray(st[i], dtype=np.float64).ravel())
        n = a.size
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        fr = pos - lo
        part = np.partition(a, (lo, min(lo + 1, n - 1)))
        v = part[lo] + fr * (part[min(lo + 1, n - 1)] - part[lo])
        vals.append(float(v))
    return math.fsum(vals) / len(vals)


def pr_indep(X, pre, c):
    """V2: PR of the centred linear Gram from its eigenvalues."""
    X = np.array(X, dtype=np.float64)
    if pre == "prh_clip+l2":
        X = np.minimum(np.maximum(X, -c), c)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xc = X - X.mean(axis=0, keepdims=True)
    lam = np.linalg.eigvalsh(Xc @ Xc.T)
    lam = np.clip(lam, 0.0, None)
    return float(lam.sum() ** 2 / np.sum(lam * lam))


def tr_concord(G):
    """V4: G = 3 x 17 (rows 3B, 7B, 13B). Mean over ViTs of Spearman(size, value) via concordance count."""
    rho = []
    for v in range(G.shape[1]):
        x = G[:, v]
        if len(set(x.tolist())) < 3:
            return None  # ties: not expected for B-type quantities; reported
        C = sum(1 for i in range(3) for j in range(i + 1, 3) if x[j] > x[i])
        rho.append({3: 1.0, 2: 0.5, 1: -0.5, 0: -1.0}[C])
    return float(np.mean(rho))


def main():
    t0 = time.time()
    F = json.load(open(os.path.join(PB, "partB_features_ol.json")))
    CL = json.load(open(os.path.join(PB, "clip_thresholds_partB_ol.json")))
    vits = F["vits"]
    out = {"label": "independent re-derivation of partB_features_ol.json (13B-dependent numbers); read-only check",
           "script_sha256": sha(os.path.abspath(__file__)),
           "inputs": {os.path.relpath(p, PROJ): sha(p) for p in (F13, os.path.join(PB, "partB_features_ol.json"),
                                                                  os.path.join(PB, "clip_thresholds_partB_ol.json"))},
           "threads": _TH, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    st = np.load(F13, mmap_mode="r")
    assert st.shape == (M, 41, 5120), st.shape
    # V1
    c_ind = clip_indep(st)
    c_pb = float(CL["clip"][OL[2]]["c"])
    out["V1_clip_13b"] = {"indep": c_ind, "partB": c_pb, "rel_err": abs(c_ind - c_pb) / c_pb,
                          "json_clip_c_equal": F["per_pre"]["prh_clip+l2"]["clip_c"][OL[2]] == c_pb,
                          "pass": abs(c_ind - c_pb) / c_pb <= 1e-12}
    print(f"[{time.time()-t0:5.0f}s] V1 clip 13B indep {c_ind!r} partB {c_pb!r}", flush=True)
    # V2
    v2 = {}
    for pre in PRES:
        prs = np.array([pr_indep(st[:, l, :], pre, c_pb) for l in range(st.shape[1])])
        ref = np.array(F["per_pre"][pre]["PR"][OL[2]])
        v2[pre] = {"max_rel_err": rel(prs, ref), "PR_star_indep": float(prs.max()), "u_star_indep": int(prs.argmax()),
                   "PR_star_partB": F["per_pre"][pre]["PR_star"][OL[2]], "u_star_partB": F["per_pre"][pre]["u_star"][OL[2]],
                   "runner_up_ratio_indep": float(np.sort(prs)[-1] / np.sort(prs)[-2])}
        v2[pre]["pass"] = bool(v2[pre]["max_rel_err"] <= 1e-9 and v2[pre]["u_star_indep"] == v2[pre]["u_star_partB"])
        print(f"[{time.time()-t0:5.0f}s] V2 {pre}: PR max rel err {v2[pre]['max_rel_err']:.2e}, PR* {prs.max():.4f} "
              f"u* {prs.argmax()}", flush=True)
    out["V2_PR_13b"] = v2
    # V3-V5
    v3, v4, v5 = {}, {}, {}
    for pre in PRES:
        Q = F["per_pre"][pre]
        PRl = {s: np.array(Q["PR"][s]) for s in OL + vits}
        Ls = {s: PRl[s].size for s in PRl}
        Bb = np.empty((3, 17)); Bf = {0.5: np.empty((3, 17)), 1.0: np.empty((3, 17))}; Bl = np.empty((3, 17))
        for i, a in enumerate(OL):
            rel_a = np.arange(Ls[a]) / (Ls[a] - 1)
            for j, b in enumerate(vits):
                Buw = np.sqrt(np.outer(PRl[a], PRl[b])) / (M - 1)
                Bb[i, j] = Buw.max()
                for r in (0.5, 1.0):
                    Bf[r][i, j] = Buw[int(math.floor(r * (Ls[a] - 1) + 0.5)), int(math.floor(r * (Ls[b] - 1) + 0.5))]
                Bl[i, j] = Buw[rel_a >= 0.60 - 1e-12].max()
        e = {"Bbar": rel(Bb, [[Q["Bbar"][a][b] for b in vits] for a in OL]),
             "Bfd_0.5": rel(Bf[0.5], [[Q["Bfd"]["0.5"][a][b] for b in vits] for a in OL]),
             "Bfd_1.0": rel(Bf[1.0], [[Q["Bfd"]["1.0"][a][b] for b in vits] for a in OL]),
             "Blate": rel(Bl, [[Q["Blate"][a][b] for b in vits] for a in OL])}
        v3[pre] = {"max_rel_err": e, "pass": max(e.values()) <= 1e-12,
                   "mean_Bbar_by_size": [float(x) for x in Bb.mean(axis=1)]}

        def blk(G):
            tr = tr_concord(G)
            med = {k: float(np.median(np.abs(G[y] - G[x]))) for x, y, k in ((0, 1, "3b|7b"), (1, 2, "7b|13b"), (0, 2, "3b|13b"))}
            mono = bool(np.all(np.diff(G, axis=0) > 0) or np.all(np.diff(G, axis=0) < 0))
            mat = bool(tr is not None and abs(abs(tr) - 1) <= 1e-9 and mono and min(med.values()) >= DELTA_B)
            return {"Tr": tr, "median_abs_gap": med, "monotone_every_vit": mono, "material": mat}
        b = {"Bbar": blk(Bb), "Bfd_0.5": blk(Bf[0.5]), "Bfd_1.0": blk(Bf[1.0]), "Blate": blk(Bl)}
        def eq(x, y):  # Tr of partb_ol.py comes from float Spearman (e.g. 0.9999999999999998): plan F2 tolerance
            return x is not None and abs(x - y) <= 1e-9
        agree = (eq(b["Bbar"]["Tr"], Q["Tr_pred"]) and b["Bbar"]["material"] == Q["material"]
                 and eq(b["Bfd_0.5"]["Tr"], Q["Tr_Bfd"]["0.5"]) and b["Bfd_0.5"]["material"] == Q["material_Bfd"]["0.5"]
                 and eq(b["Bfd_1.0"]["Tr"], Q["Tr_Bfd"]["1.0"]) and b["Bfd_1.0"]["material"] == Q["material_Bfd"]["1.0"]
                 and eq(b["Blate"]["Tr"], Q["Tr_Blate"]) and b["Blate"]["material"] == Q["material_Blate"])
        sO = 1 if (b["Bbar"]["Tr"] or 0) >= 0 else -1
        v4[pre] = {"blocks": b, "s_O_indep": sO, "s_O_partB": Q["s_O"],
                   "n_new_Bbar_gap_ge_delta_b": int(np.sum(np.abs(Bb[2] - Bb[0]) >= DELTA_B) + np.sum(np.abs(Bb[2] - Bb[1]) >= DELTA_B)),
                   "n_new_partB": Q["qualifying"]["PO2aii_n_Bbar_gap_ge_delta_b"],
                   "pass": bool(agree and sO == Q["s_O"]
                                and int(np.sum(np.abs(Bb[2] - Bb[0]) >= DELTA_B) + np.sum(np.abs(Bb[2] - Bb[1]) >= DELTA_B))
                                == Q["qualifying"]["PO2aii_n_Bbar_gap_ge_delta_b"])}
        # V5: boundaries from the frozen pair files (biased CKA, layer-max, protocol full)
        kn = {}
        for k, s in (("3b", OL[0]), ("7b", OL[1])):
            T, c, mu, sc = [], [], [], []
            for v in vits:
                d = json.load(open(os.path.join(FROZEN8, pre, f"{s}__{v}.json")))
                assert d["lm"] == s and d["vit"] == v and d["pre"] == pre
                f = d["summaries"]["cka_biased"]["full"]
                T.append(f["s_obs"]); c.append(f["c_alpha"]); mu.append(f["null_mean"]); sc.append(f["s_cal"])
            kn[k] = tuple(np.array(x) for x in (T, c, mu, sc))
        band = Q["band"]
        B13 = Bb[2]
        errs = []
        for k in ("3b", "7b"):
            T, c, mu, sc = kn[k]
            for o, side in ((band["o_lo"], "lo"), (band["o_hi"], "hi")):
                kap = (T * (1 - c) + (1 - T) * (B13 + o - c)) / (1 - c)  # algebraically equal form of plan B6
                errs.append(rel(kap, [Q["boundaries"]["s_cal"][side][k][v] for v in vits]))
            for m_, side in ((MU[0], "lo"), (MU[1], "hi")):
                errs.append(rel(T + B13 + m_ - mu, [Q["boundaries"]["unc"][side][k][v] for v in vits]))
            for h in HS:
                for o, side in ((band["o"] - h, "lo"), (band["o"] + h, "hi")):
                    kap = (T * (1 - c) + (1 - T) * (B13 + o - c)) / (1 - c)
                    errs.append(rel(kap, [Q["boundaries"]["s_cal_h"][repr(float(h))][side][k][v] for v in vits]))
            # the known values Part B stored equal the frozen files
            for arr, key in ((T, "T"), (c, "c_alpha"), (mu, "null_mean"), (sc, "s_cal")):
                errs.append(rel(arr, [Q["known"][key][k][v] for v in vits]))
        # independent check of the kappa identity: s_cal(13B) < s_cal(j) <=> T13 < kappa (for c13 = Bbar13 + o)
        T3, c3, _, s3 = kn["3b"]
        o = band["o"]
        kap = (T3 * (1 - c3) + (1 - T3) * (B13 + o - c3)) / (1 - c3)
        c13 = B13 + o
        ident = []
        for eps in (-1e-6, 1e-6):
            T13 = kap + eps
            s13 = (T13 - c13) / (1 - c13)
            ident.append(bool(np.all((s13 < s3) == (eps < 0))))
        width = {k: [float(x) for x in (np.array([Q["boundaries"]["s_cal"]["hi"][k][v] for v in vits])
                                         - np.array([Q["boundaries"]["s_cal"]["lo"][k][v] for v in vits]))] for k in ("3b", "7b")}
        v5[pre] = {"max_rel_err_boundaries_and_known": max(errs), "kappa_identity_holds": all(ident),
                   "band_used": band, "kappa_width_min_max": {k: [min(w), max(w)] for k, w in width.items()},
                   "pass": bool(max(errs) <= 1e-13 and all(ident))}
        print(f"[{time.time()-t0:5.0f}s] {pre}: V3 {v3[pre]['pass']} V4 {v4[pre]['pass']} (Tr {b['Bbar']['Tr']}, mat {b['Bbar']['material']}) "
              f"V5 {v5[pre]['pass']} (max rel {max(errs):.1e})", flush=True)
    out["V3_baselines"] = v3
    out["V4_trends_materiality"] = v4
    out["V5_boundaries"] = v5
    # V6 required fields (VERDICTS_SPEC section 8)
    top = ["schema", "m", "delta_b", "band", "mu_band", "ol_lms", "vits", "pooled11", "num_params", "L_stored", "per_pre"]
    per = ["clip_c", "PR", "PR_star", "u_star", "u_star_rel", "Bbar", "Bbar_argmax", "Bfd", "Blate", "gaps", "Tr_pred",
           "material", "s_O", "s_O_J", "Tr_Bfd", "material_Bfd", "Tr_Blate", "material_Blate", "s_late", "known", "boundaries"]
    miss = [k for k in top if k not in F] + [f"{p}.{k}" for p in PRES for k in per if k not in F["per_pre"][p]]
    out["V6_spec_fields"] = {"missing": miss, "schema": F.get("schema"), "band": F.get("band"), "mu_band": F.get("mu_band"),
                             "pass": not miss and F.get("schema") == "OL-PARTB v1"
                             and F["band"] == {"o": 0.0019, "o_lo": 0.0009, "o_hi": 0.0029, "h": 0.001}
                             and F["mu_band"] == [-0.0002, 0.0002] and F["delta_b"] == 0.005 and F["m"] == 1024}
    out["pass_all"] = bool(out["V1_clip_13b"]["pass"] and all(v2[p]["pass"] for p in PRES) and all(v3[p]["pass"] for p in PRES)
                           and all(v4[p]["pass"] for p in PRES) and all(v5[p]["pass"] for p in PRES) and out["V6_spec_fields"]["pass"])
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(PB, "partB_verify_ol.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[{time.time()-t0:5.0f}s] pass_all {out['pass_all']}; wrote partB/partB_verify_ol.json", flush=True)
    sys.exit(0 if out["pass_all"] else 5)


if __name__ == "__main__":
    main()
