#!/usr/bin/env python
"""Verifier, task 1: SPEC section 8 agreement between hrhv2 (main) and hrhv2_ref.

Every case feeds IDENTICAL inputs (features + pre, Gram, or squared distances)
to hrhv2.ladder and hrhv2_ref.ladder_ref, with the same H1/H2 split (hrhv2's
hash split passed to ref as H1=). Tolerances (SPEC 8): exact for set / graph /
tree quantities, <= 1e-10 relative (+1e-13 floor) for floats, <= 1e-6 absolute
for isotonic fits. For m <= 256 cases, the stress cases, and any case with an
exact-class disagreement, set/tree quantities are recomputed by brute force
(python-sorted (D_ij, j) neighbour lists, BFS components, Kruskal on python-
sorted (D, min, max) edge keys, O(n^2) per-row Kendall) to decide which
implementation is right.

Usage: python agree.py OUTDIR [--only NAME_SUBSTR] [--jobs N]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "hrhv2_ref"))

import numpy as np  # noqa: E402

from hrhv2 import plants, util, structural as st  # noqa: E402
from hrhv2.ladder import ladder  # noqa: E402
from hrhv2.prep import Geom  # noqa: E402
from hrhv2_ref import ladder_ref  # noqa: E402
import hrhv2_ref.core as rcore  # noqa: E402

KEY = "VERIF1"
SALT = "hrhv2-halves"
FEAT = os.path.join(PROJ, "features", "wit1024")
TAUS = (1.0, 0.1, 0.01, 0.001, 0.0001)

REAL_PAIRS = [
    ("huggyllama__llama-7b", "openlm-research__open_llama_7b"),
    ("vit_base_patch16_clip_224.laion2b", "vit_base_patch14_dinov2.lvd142m"),
    ("bigscience__bloomz-1b7", "vit_large_patch14_dinov2.lvd142m"),
]


# ----------------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------------

def case_list():
    cs = []
    for m in (256, 1024):
        for p in (2, 4, 8):
            for beta in (1, 3, 10):
                cs.append({"name": f"P-Bern|m={m}|p={p}|beta={beta}", "gen": "bern", "m": m, "p": p, "beta": beta})
            for plant in ("P-Aniso", "P-Conf", "P-Rough"):
                cs.append({"name": f"{plant}|m={m}|p={p}", "gen": "pair", "plant": plant, "m": m, "p": p})
            for c in (0.02, 0.05, 0.15, 0.4):
                cs.append({"name": f"P-Gauss|m={m}|p={p}|c={c}", "gen": "gauss", "m": m, "p": p, "c": c})
                cs.append({"name": f"P-Gauss-preclip|m={m}|p={p}|c={c}", "gen": "gauss", "m": m, "p": p, "c": c,
                           "preclip": True})
        cs.append({"name": f"WarpShare|m={m}|p=4", "gen": "warp", "m": m, "p": 4})
        cs.append({"name": f"WarpShare-preclip|m={m}|p=4", "gen": "warp", "m": m, "p": 4, "preclip": True})
        cs.append({"name": f"P-Bern-sphere|m={m}|p=2|beta=10", "gen": "bern_sphere", "m": m, "p": 2, "beta": 10})
        for pre in ("l2", "prh_clip+l2", "none"):
            cs.append({"name": f"Gauss-rand|m={m}|pre={pre}", "gen": "rand", "m": m, "pre": pre})
            cs.append({"name": f"Stack3D-rand|m={m}|pre={pre}", "gen": "rand3d", "m": m, "pre": pre})
    # near-tie stress
    for pre in ("l2", "none", "prh_clip+l2"):
        cs.append({"name": f"Stress-dup|m=256|pre={pre}", "gen": "dup", "m": 256, "pre": pre})
    cs.append({"name": "Stress-lattice81|m=256|pre=none", "gen": "lattice", "m": 256, "levels": 3, "d": 4, "pre": "none"})
    cs.append({"name": "Stress-lattice1000|m=1024|pre=none", "gen": "lattice", "m": 1024, "levels": 10, "d": 3, "pre": "none"})
    cs.append({"name": "Stress-lattice-l2|m=512|pre=l2", "gen": "lattice", "m": 512, "levels": 4, "d": 4, "pre": "l2"})
    cs.append({"name": "Stress-quant|m=1024|pre=none", "gen": "quant", "m": 1024, "pre": "none"})
    cs.append({"name": "Stress-smallm|m=60|pre=l2", "gen": "rand", "m": 60, "pre": "l2"})
    cs.append({"name": "Stress-smallm|m=60|pre=none", "gen": "lattice", "m": 60, "levels": 2, "d": 5, "pre": "none"})
    # real features (WIT-1024), 3 pairs x 2 relative depths x 2 pre
    for a, b in REAL_PAIRS:
        for r in (0.5, 1.0):
            for pre in ("l2", "prh_clip+l2"):
                cs.append({"name": f"REAL|{a}|{b}|r={r}|pre={pre}", "gen": "real", "a": a, "b": b, "r": r,
                           "pre": pre, "m": 1024})
    return cs


def layer_index(name: str, r: float) -> int:
    meta = json.load(open(os.path.join(FEAT, name + ".json")))
    arr = np.load(os.path.join(FEAT, name + ".npy"), mmap_mode="r")
    if meta.get("kind") == "lm":
        L = int(meta["n_layers"])            # hidden_states[0..L], 0 = embedding
        assert arr.shape[1] == L + 1
        return int(math.floor(r * L + 0.5))
    L = arr.shape[1]                         # blocks 0..L-1, stored index = ell - 1
    return int(math.floor(r * L + 0.5)) - 1


def make_inputs(c):
    """-> dict(kind, A, B, pre, clipA_full, clipB_full, ids)"""
    g = c["gen"]
    m = c["m"]
    key = f"{KEY}|{c['name']}"
    ids = [str(i) for i in range(m)]
    if g == "bern":
        KA, KBs, info = plants.bern_grams(1024, c["p"], (1, 3, 10), f"{KEY}|P-Bern|p={c['p']}")
        return {"kind": "gram", "A": KA[:m, :m], "B": KBs[c["beta"]][:m, :m], "pre": "none", "ids": ids}
    if g == "bern_sphere":
        X = plants.base_cloud(m, c["p"], key)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        KA = X @ X.T
        KB = np.exp(c["beta"] * (KA - 1.0))
        return {"kind": "gram", "A": KA, "B": KB, "pre": "none", "ids": ids}
    if g == "pair":
        p = c["p"]
        if c["plant"] == "P-Aniso":
            XA, XB = plants.aniso_features(1024, p, key)
        elif c["plant"] == "P-Conf":
            XA, XB = plants.conf_features(1024, p, key)
        else:
            ref = plants.feature_dists(plants.base_cloud(256, p, f"{key}|ref"))
            s2 = 0.3 * plants.knn_d_median(ref, 10)
            XA, XB = plants.rough_features(1024, p, key, s2=s2)
        return {"kind": "feat", "A": XA[:m], "B": XB[:m], "pre": "none", "ids": ids}
    if g == "gauss":
        ref = plants.feature_dists(plants.base_cloud(1024, c["p"], f"{KEY}|P-Gauss|p={c['p']}|sigma_ref"))
        sigma = c["c"] * plants.knn_d_median(ref, 10)
        DA, DB = plants.gauss_plant(1024, c["p"], sigma, f"{KEY}|P-Gauss|m={m}|p={c['p']}|c={c['c']}")
        if c.get("preclip"):
            DB = np.maximum(DB, 0.0)
        return {"kind": "dist", "A": DA[:m, :m], "B": DB[:m, :m], "pre": "none", "ids": ids,
                "n_negative_DB": int(np.sum(util.upper_values(DB[:m, :m]) < 0))}
    if g == "warp":
        DA, DB, _ = plants.warpshare_plant(m, c["p"], 0.3, f"{KEY}|WarpShare|m={m}|p={c['p']}")
        if c.get("preclip"):
            DB = np.maximum(DB, 0.0)
        return {"kind": "dist", "A": DA, "B": DB, "pre": "none", "ids": ids,
                "n_negative_DB": int(np.sum(util.upper_values(DB) < 0))}
    if g == "rand":
        rng = util.rng_from_key(key)
        Z = rng.standard_t(3, size=(m, 64))
        W = rng.standard_normal((64, 96)) / 8.0
        return {"kind": "feat", "A": Z, "B": Z @ W + 0.7 * rng.standard_t(3, size=(m, 96)), "pre": c["pre"], "ids": ids}
    if g == "rand3d":
        rng = util.rng_from_key(key)
        SA = rng.standard_t(3, size=(m, 3, 48))
        SB = SA[:, :, :40] * 1.5 + rng.standard_t(4, size=(m, 3, 40))
        return {"kind": "feat", "A": SA[:, 1, :], "B": SB[:, 2, :], "pre": c["pre"], "ids": ids,
                "clipA_full": SA, "clipB_full": SB}
    if g == "dup":
        rng = util.rng_from_key(key)
        XA = rng.standard_normal((m, 16))
        XB = XA + 0.5 * rng.standard_normal((m, 16))
        XA[200:220] = XA[0:20]
        XB[200:220] = XB[0:20]
        return {"kind": "feat", "A": XA, "B": XB, "pre": c["pre"], "ids": ids}
    if g == "lattice":
        rng = util.rng_from_key(key)
        XA = rng.integers(0, c["levels"], size=(m, c["d"])).astype(np.float64)
        XB = XA.copy()
        flip = rng.random(XB.shape) < 0.15
        XB[flip] = rng.integers(0, c["levels"], size=int(flip.sum()))
        if c["pre"] == "l2":
            XA += 1.0
            XB += 1.0
        return {"kind": "feat", "A": XA, "B": XB, "pre": c["pre"], "ids": ids}
    if g == "quant":
        rng = util.rng_from_key(key)
        Z = rng.standard_normal((m, 6))
        return {"kind": "feat", "A": np.round(Z * 2) / 2, "B": np.round((Z + 0.4 * rng.standard_normal((m, 6))) * 2) / 2,
                "pre": "none", "ids": ids}
    if g == "real":
        arrA = np.load(os.path.join(FEAT, c["a"] + ".npy"), mmap_mode="r")
        arrB = np.load(os.path.join(FEAT, c["b"] + ".npy"), mmap_mode="r")
        la, lb = layer_index(c["a"], c["r"]), layer_index(c["b"], c["r"])
        out = {"kind": "feat", "A": np.asarray(arrA[:, la, :], dtype=np.float64),
               "B": np.asarray(arrB[:, lb, :], dtype=np.float64), "pre": c["pre"], "ids": ids,
               "layer_A": la, "layer_B": lb}
        if c["pre"] == "prh_clip+l2":
            out["clipA_full"] = np.asarray(arrA, dtype=np.float64)
            out["clipB_full"] = np.asarray(arrB, dtype=np.float64)
        return out
    raise ValueError(g)


# ----------------------------------------------------------------------------
# key map main -> ref, tolerance classes
# ----------------------------------------------------------------------------

def tag(t):
    return ("%g" % t).replace(".", "p").replace("-", "m")


def key_map(ks, comp_ks=(1, 3, 10)):
    km = {}
    for k in ks:
        km[f"s_N_k{k}"] = f"s_N@{k}"
        km[f"Delta_N_k{k}"] = f"delta_N@{k}"
        km[f"chance_k{k}"] = f"chance@{k}"
        km[f"neartie_A_k{k}"] = f"near_tie_A@{k}"
        km[f"neartie_B_k{k}"] = f"near_tie_B@{k}"
    for k in comp_ks:
        km[f"comp_rand_k{k}"] = f"comp_rand@{k}"
        km[f"comp_n_A_k{k}"] = f"comp_n_A@{k}"
        km[f"comp_n_B_k{k}"] = f"comp_n_B@{k}"
        km[f"comp_degenerate_k{k}"] = f"comp_degenerate@{k}"
    km.update({
        "k_q": "k_q", "Delta_O": "delta_O",
        "globalorder_tau_b": "kendall_global", "globalorder_spearman": "spearman_global",
        "Delta_E": "delta_E", "globalorder_discordant": "E_n_disc_strict",
        "s_T": "s_T", "s_T_forest": "s_T_forest", "s_T_bridge": "s_T_bridge",
        "forest_share_A": "forest_share_A", "forest_share_B": "forest_share_B",
        "n_bridges_A": "n_bridges_A", "n_shared_bridges": "n_shared_bridges",
        "shared_forest_fraction_of_shared": "forest_share_shared_A", "s_N_k1_for_mst": "s_N@1",
        "forest_in_mst_A": "F_A_subset_T_A", "forest_in_mst_B": "F_B_subset_T_B",
        "Gs_rms": "Gs_rms", "Gs_max": "Gs_max", "G_rms": "G_rms", "G_max": "G_max",
        "cka_biased": "cka_biased", "cka_debiased": "cka_debiased", "cka_debiased_arheps": "cka_debiased_eps8",
        "hsicu_AB": "hsic_u_AB", "hsicu_AA": "hsic_u_AA", "hsicu_BB": "hsic_u_BB",
        "rsa_pearson": "rsa_pearson", "sog_scale_A": "sog_Q_A", "sog_scale_B": "sog_Q_B",
        "gl_projector_dist": "proj_dist", "gl_cca_mean_sq": "cca_mean_sq",
        "gl_rank_A": "proj_rank_A", "gl_rank_B": "proj_rank_B", "n_H1": "H1_size",
    })
    for t in TAUS:
        km[f"sog_knn10_tau{tag(t)}"] = f"sog_knn@{t:g}"
        km[f"sog_mst_tau{tag(t)}"] = f"sog_mst@{t:g}"
    for q in ("scale", "affine", "iso", "rowiso"):
        for stat in ("expl", "rms", "max"):
            km[f"quot_AB_{q}_{stat}"] = f"quot_{q}_{stat}"
            km[f"quot_BA_{q}_{stat}"] = f"quot_{q}_{stat}_BA"
            km[f"quot_AB_{q}_{stat}_cf"] = f"quot_{q}_{stat}_xfit"
            km[f"quot_BA_{q}_{stat}_cf"] = f"quot_{q}_{stat}_xfit_BA"
        km[f"quot_mean_{q}_expl"] = f"quot_{q}_expl_mean"
        km[f"quot_mean_{q}_expl_cf"] = f"quot_{q}_expl_xfit_mean"
    return km


EXACT_MAIN = ("s_N_k", "Delta_N_k", "neartie_", "comp_", "s_T", "forest_share", "n_bridges", "n_shared",
              "shared_forest", "s_N_k1_for_mst", "forest_in_mst", "globalorder_discordant", "k_q", "n_H1",
              "gl_rank", "chance_k")
ISO_MAIN = ("quot_AB_iso", "quot_BA_iso", "quot_mean_iso", "quot_AB_rowiso", "quot_BA_rowiso", "quot_mean_rowiso")


def tol_class(k):
    if k.startswith(EXACT_MAIN):
        return "exact"
    if k.startswith(ISO_MAIN):
        return "iso"
    return "float"


def compare(main, ref, km):
    rows = []
    for km_, kr in km.items():
        if km_ not in main or kr not in ref:
            continue
        a, b = main[km_], ref[kr]
        if isinstance(a, str) or isinstance(b, str):
            continue
        a = float(a)
        b = float(b)
        cls = tol_class(km_)
        if math.isnan(a) or math.isnan(b):
            ok = math.isnan(a) and math.isnan(b)
            d = float("nan")
        else:
            d = abs(a - b)
            if cls == "exact":
                ok = a == b
            elif cls == "iso":
                ok = d <= 1e-6
            else:
                ok = d <= 1e-10 * max(abs(a), abs(b)) + 1e-13
        rows.append({"key": km_, "ref_key": kr, "class": cls, "main": a, "ref": b, "absdiff": d,
                     "reldiff": (d / max(abs(a), abs(b))) if (d == d and max(abs(a), abs(b)) > 0) else 0.0,
                     "ok": bool(ok)})
    return rows


# ----------------------------------------------------------------------------
# brute force
# ----------------------------------------------------------------------------

def brute_knn(D, kmax):
    m = D.shape[0]
    out = np.empty((m, kmax), dtype=np.int64)
    for i in range(m):
        lst = sorted((float(D[i, j]), j) for j in range(m) if j != i)
        out[i] = [j for _, j in lst[:kmax]]
    return out


def brute_components(nbrs, k):
    m = nbrs.shape[0]
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in nbrs[i, :k]:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    lab = -np.ones(m, dtype=np.int64)
    c = 0
    for s in range(m):
        if lab[s] >= 0:
            continue
        stack = [s]
        lab[s] = c
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if lab[w] < 0:
                    lab[w] = c
                    stack.append(w)
        c += 1
    return c, lab


def brute_rand(l1, l2):
    m = l1.size
    s1 = l1[:, None] == l1[None, :]
    s2 = l2[:, None] == l2[None, :]
    iu = np.triu_indices(m, 1)
    return float(np.sum(s1[iu] != s2[iu])) / (m * (m - 1) // 2)


def brute_mst(D):
    m = D.shape[0]
    edges = sorted((float(D[i, j]), i, j) for i in range(m) for j in range(i + 1, m))
    par = list(range(m))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    T = []
    for w, i, j in edges:
        ri, rj = f(i), f(j)
        if ri != rj:
            par[ri] = rj
            T.append(i * m + j)
            if len(T) == m - 1:
                break
    return np.sort(np.array(T, dtype=np.int64))


def brute_row_kendall(DA, DB, rows=None):
    """Fagin K^(1/2) per row by O(n^2) sign products."""
    m = DA.shape[0]
    rows = range(m) if rows is None else rows
    out = {}
    for i in rows:
        x = np.delete(DA[i], i)
        y = np.delete(DB[i], i)
        sx = np.sign(x[:, None] - x[None, :])
        sy = np.sign(y[:, None] - y[None, :])
        iu = np.triu_indices(x.size, 1)
        a, b = sx[iu], sy[iu]
        disc = np.sum(a * b < 0)
        one = np.sum((a == 0) ^ (b == 0))
        out[i] = (disc + 0.5 * one) / a.size
    return out


def brute_checks(DA, DB, ks, comp_ks, main, ref, with_kendall=True):
    """Recompute set/tree quantities by brute force; report which implementation matches."""
    m = DA.shape[0]
    kmax = max(max(ks), max(comp_ks), 10) + 1
    kmax = min(kmax, m - 1)
    bA, bB = brute_knn(DA, kmax), brute_knn(DB, kmax)
    res = {}
    for k in ks:
        s = float(sum(len(set(bA[i, :k].tolist()) & set(bB[i, :k].tolist())) for i in range(m))) / (m * k)
        res[f"s_N_k{k}"] = {"brute": s, "main": main.get(f"s_N_k{k}"), "ref": ref.get(f"s_N@{k}")}
    for k in comp_ks:
        ca, la = brute_components(bA, k)
        cb, lb = brute_components(bB, k)
        res[f"comp_rand_k{k}"] = {"brute": brute_rand(la, lb), "main": main.get(f"comp_rand_k{k}"),
                                  "ref": ref.get(f"comp_rand@{k}")}
        res[f"comp_n_A_k{k}"] = {"brute": ca, "main": main.get(f"comp_n_A_k{k}"), "ref": ref.get(f"comp_n_A@{k}")}
    TA, TB = brute_mst(DA), brute_mst(DB)
    FA = np.unique([min(i, int(bA[i, 0])) * m + max(i, int(bA[i, 0])) for i in range(m)])
    sh = np.intersect1d(TA, TB)
    res["s_T"] = {"brute": sh.size / (m - 1), "main": main.get("s_T"), "ref": ref.get("s_T")}
    res["forest_share_A"] = {"brute": FA.size / (m - 1), "main": main.get("forest_share_A"), "ref": ref.get("forest_share_A")}
    br = np.setdiff1d(TA, FA)
    res["n_shared_bridges"] = {"brute": int(np.intersect1d(br, TB).size), "main": main.get("n_shared_bridges"),
                               "ref": ref.get("n_shared_bridges")}
    res["F_A_subset_T_A"] = {"brute": bool(np.isin(FA, TA).all()), "main": main.get("forest_in_mst_A"),
                             "ref": ref.get("F_A_subset_T_A")}
    if with_kendall:
        bk = brute_row_kendall(DA, DB)
        res["Delta_O"] = {"brute": float(np.mean(list(bk.values()))), "main": main.get("Delta_O"),
                          "ref": ref.get("delta_O")}
    for k, v in res.items():
        b = v["brute"]
        v["main_ok"] = (v["main"] is not None) and (abs(float(v["main"]) - float(b)) <= (1e-12 if k == "Delta_O" else 0))
        v["ref_ok"] = (v["ref"] is not None) and (abs(float(v["ref"]) - float(b)) <= (1e-12 if k == "Delta_O" else 0))
    return res


# ----------------------------------------------------------------------------
# run one case
# ----------------------------------------------------------------------------

def run_case(c, outdir):
    t0 = time.perf_counter()
    inp = make_inputs(c)
    kind, pre = inp["kind"], inp["pre"]
    ids = inp["ids"]
    m = c["m"]
    h1 = util.hash_half(ids, SALT)
    rec = {"case": c, "kind": kind, "pre": pre, "m": m}
    for k in ("n_negative_DB", "layer_A", "layer_B"):
        if k in inp:
            rec[k] = inp[k]
    if kind == "feat":
        cA = inp.get("clipA_full")
        cB = inp.get("clipB_full")
        A = Geom.from_features(inp["A"], pre, clip_ref=cA if pre == "prh_clip+l2" else None)
        B = Geom.from_features(inp["B"], pre, clip_ref=cB if pre == "prh_clip+l2" else None)
        thrA = thrB = None
        if pre == "prh_clip+l2":
            # ref computes its OWN threshold on the same reference array (independent code path)
            thrA = rcore.prh_clip_threshold(cA if cA is not None else inp["A"])
            thrB = rcore.prh_clip_threshold(cB if cB is not None else inp["B"])
            from hrhv2.prep import prh_clip_threshold as main_thr
            mA = main_thr(cA if cA is not None else inp["A"])
            mB = main_thr(cB if cB is not None else inp["B"])
            rec["clip_threshold"] = {"main_A": mA, "ref_A": thrA, "main_B": mB, "ref_B": thrB,
                                     "absdiff_A": abs(mA - thrA), "absdiff_B": abs(mB - thrB)}
        t1 = time.perf_counter()
        ref = ladder_ref(inp["A"], inp["B"], pre, H1=h1, clip_threshold_A=thrA, clip_threshold_B=thrB)
    elif kind == "gram":
        A, B = Geom.from_gram(inp["A"]), Geom.from_gram(inp["B"])
        t1 = time.perf_counter()
        ref = ladder_ref(inp["A"], inp["B"], gram=True, H1=h1)
    else:
        A, B = Geom.from_dist(inp["A"]), Geom.from_dist(inp["B"])
        t1 = time.perf_counter()
        ref = ladder_ref(inp["A"], inp["B"], dist=True, H1=h1)
    t_ref = time.perf_counter() - t1
    t2 = time.perf_counter()
    main = ladder(A, B, item_ids=ids, halves_salt=SALT, n_jobs=1)
    t_main = time.perf_counter() - t2
    # D identical between implementations?
    if kind == "feat":
        XA, _ = rcore.preprocess(inp["A"], pre, thrA)
        XB, _ = rcore.preprocess(inp["B"], pre, thrB)
        DAr, DBr = rcore.sqdist_from_gram(rcore.gram_sym(XA)), rcore.sqdist_from_gram(rcore.gram_sym(XB))
    elif kind == "gram":
        DAr, DBr = rcore.sqdist_from_gram(inp["A"]), rcore.sqdist_from_gram(inp["B"])
    else:
        DAr = np.maximum(np.array(inp["A"], dtype=np.float64), 0.0)
        DBr = np.maximum(np.array(inp["B"], dtype=np.float64), 0.0)
        np.fill_diagonal(DAr, 0.0)
        np.fill_diagonal(DBr, 0.0)
    rec["D_identical"] = {"A": bool(np.array_equal(A.D, DAr)), "B": bool(np.array_equal(B.D, DBr)),
                          "A_maxabs": float(np.max(np.abs(A.D - DAr))), "B_maxabs": float(np.max(np.abs(B.D - DBr))),
                          "A_symmetric_main": bool(np.array_equal(A.D, A.D.T)),
                          "B_symmetric_main": bool(np.array_equal(B.D, B.D.T))}
    ks = st.default_k_grid(m)
    ks = [k for k in ks if k <= m - 2]
    km = key_map(ks)
    rows = compare(main, ref, km)
    bad = [r for r in rows if not r["ok"]]
    rec["n_compared"] = len(rows)
    rec["n_fail"] = len(bad)
    rec["fails"] = bad
    rec["by_class"] = {cl: {"n": sum(r["class"] == cl for r in rows),
                            "n_fail": sum((r["class"] == cl) and not r["ok"] for r in rows),
                            "max_absdiff": max([r["absdiff"] for r in rows if r["class"] == cl and r["absdiff"] == r["absdiff"]] or [0.0]),
                            "max_reldiff": max([r["reldiff"] for r in rows if r["class"] == cl] or [0.0])}
                       for cl in ("exact", "float", "iso")}
    rec["unmapped_main"] = sorted(k for k in main if k not in km and not k.startswith(("time_", "quot_AB_ss0", "quot_BA_ss0", "warpshare_")))
    rec["neartie"] = {k: main[k] for k in main if k.startswith("neartie_")}
    rec["Delta_O_rows_with_ties"] = main.get("Delta_O_rows_with_ties")
    rec["t_main_s"], rec["t_ref_s"] = round(t_main, 3), round(t_ref, 3)
    need_brute = (m <= 256) or c["name"].startswith("Stress") or any(r["class"] == "exact" for r in bad)
    if need_brute:
        tb = time.perf_counter()
        rec["brute"] = brute_checks(A.D, B.D, ks, (1, 3, 10), main, ref, with_kendall=(m <= 300))
        rec["brute_all_main_ok"] = all(v["main_ok"] for v in rec["brute"].values())
        rec["brute_all_ref_ok"] = all(v["ref_ok"] for v in rec["brute"].values())
        rec["t_brute_s"] = round(time.perf_counter() - tb, 2)
    rec["t_total_s"] = round(time.perf_counter() - t0, 2)
    rec["main_values"] = {r["key"]: r["main"] for r in rows}
    rec["ref_values"] = {r["ref_key"]: r["ref"] for r in rows}
    fn = os.path.join(outdir, "cases", c["name"].replace("|", "__").replace("/", "_") + ".json")
    with open(fn + ".tmp", "w") as f:
        json.dump(rec, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(fn + ".tmp", fn)
    return c["name"], rec["n_fail"], rec["t_total_s"]


def _worker(args):
    c, outdir = args
    try:
        return run_case(c, outdir)
    except Exception:
        return c["name"], -1, traceback.format_exc()


def main():
    outdir = sys.argv[1]
    only = None
    jobs = 8
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    if "--jobs" in sys.argv:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1])
    os.makedirs(os.path.join(outdir, "cases"), exist_ok=True)
    cs = [c for c in case_list() if only is None or only in c["name"]]
    # biggest first
    cs.sort(key=lambda c: -c["m"])
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(jobs, maxtasksperchild=4) as pool:
        for name, nf, t in pool.imap_unordered(_worker, [(c, outdir) for c in cs]):
            print(f"[{time.time() - t0:7.1f}s] {name}: n_fail={nf} t={t}", flush=True)


if __name__ == "__main__":
    main()
