#!/usr/bin/env python
"""S1.2 LLaMA axis, Part B, steps B1-B5 (role llama-axis-predict).

Plan: analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md section 9
(sha256 990428e3f29c7ecfba9cf314a45c73c79f0154f86a07520429910d282c0a2fc8).

FEATURES ONLY. This script opens per-model feature stacks (+ their sidecars,
for layer counts and num_params) and the S1.2 clip-threshold cache. It
computes no cross-model quantity of any kind (no CKA, no kNN overlap, no
permutation): every number below is a function of one model's features,
or a product of two such single-model numbers (Proposition 5 baselines).

B1  PR_u = (tr Kc)^2 / ||Kc||_F^2, Kc = H K H, K = X X^T of the 1024
    preprocessed rows (float64), every stored layer of the 10 LMs and 17 ViTs,
    pre in {prh_clip+l2, l2}. Preprocessing = analysis/s12/common.preprocess_rows
    (the function the layer-max code uses), clip threshold = hrhv2.prep.
    prh_clip_threshold on the float64 all-layer stack (D5); for the 25 S1.2
    models it is recomputed and checked bitwise against results/S1_2/cache.
B2  B_uw = sqrt(PR_u PR_w)/(m-1); layer-max baseline Bbar = max_uw B_uw with
    its argmax; fixed-depth baselines at r in {0.5, 1.0} (S1.2 primary rule).
B3  LLaMA-axis trends Tr_L of Bbar and of B_fd(r) (lm_trend definition with
    family LLaMA, exact over 3! relabellings) and gaps |Bbar_j - Bbar_j'|.
B4  materiality: |Tr| = 1 and, for each of the 3 size pairs, median_v gap >= 0.005.
B5  Pearson(ARH rank, Bbar) on the pooled 10-LM x 17-ViT panel (and 8-LM),
    threshold-setting LM layer u*/(L_A-1), BLOOMZ retrodiction Tr_BLOOMZ(Bbar).

Output: results/S1_2_layermax_llama/partB/partB_features.json
        results/S1_2_layermax_llama/partB/clip_thresholds_partB.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "6"

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "analysis", "s12"))

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import common as C  # noqa: E402  (S1.2 registry + preprocess_rows; read-only import)
from hrhv2.prep import prh_clip_threshold, center_gram  # noqa: E402

_spec = importlib.util.spec_from_file_location("s12_aggregate", os.path.join(PROJ, "analysis", "s12", "aggregate.py"))
A12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A12)          # read-only import: _rho_matrix (trend arithmetic of lm_trend)

FEAT = os.path.join(PROJ, "features")
OUTDIR = os.path.join(PROJ, "results", "S1_2_layermax_llama", "partB")
PAIRS_GUARD = os.path.join(PROJ, "results", "S1_2_layermax_llama", "pairs")
PLAN = os.path.join(PROJ, "analysis", "s12_layermax_llama", "ANALYSIS_PLAN_LLAMA.md")
PLAN_SHA = "990428e3f29c7ecfba9cf314a45c73c79f0154f86a07520429910d282c0a2fc8"
S12_CLIP = os.path.join(PROJ, "results", "S1_2", "cache", "clip_thresholds.json")

M = 1024
KNN_K = 10
DELTA_B = 0.005
DEPTHS = (0.5, 1.0)
PRES = ("prh_clip+l2", "l2")

LMS = [(s, f, "wit1024") for s, f in C.LMS] + [
    ("huggyllama__llama-13b", "LLaMA", "wit1024_llama_axis"),
    ("huggyllama__llama-30b", "LLaMA", "wit1024_llama_axis"),
]
VITS = [(s, f, "wit1024") for s, f in C.VITS]
LLAMA = ["huggyllama__llama-7b", "huggyllama__llama-13b", "huggyllama__llama-30b"]
BLOOMZ = [s for s, f, _ in LMS if f == "BLOOMZ"]
ARH_RANK = {  # plan section 1 (prh_models.get_models("val") order)
    "bigscience__bloomz-560m": 1, "bigscience__bloomz-1b1": 2, "bigscience__bloomz-1b7": 3,
    "bigscience__bloomz-3b": 4, "bigscience__bloomz-7b1": 5, "openlm-research__open_llama_3b": 6,
    "openlm-research__open_llama_7b": 7, "huggyllama__llama-7b": 9, "huggyllama__llama-13b": 10,
    "huggyllama__llama-30b": 11,
}
DS = {s: ds for s, _, ds in LMS + VITS}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sidecar(stem: str) -> dict:
    with open(os.path.join(FEAT, DS[stem], stem + ".json")) as f:
        return json.load(f)


def stack(stem: str) -> np.ndarray:
    st = np.load(os.path.join(FEAT, DS[stem], stem + ".npy"), mmap_mode="r")
    assert st.shape[0] == M and st.ndim == 3, (stem, st.shape)
    assert list(st.shape) == list(sidecar(stem)["shape"]), stem
    return st


def fixed_layer(L: int, r: float) -> int:
    return int(math.floor(r * (L - 1) + 0.5))


def layer_pr(X: np.ndarray, pre: str, c: float | None) -> tuple[float, float, float]:
    Xp = C.preprocess_rows(X, pre, c)
    K = Xp @ Xp.T
    Kc = center_gram(K)
    tr = float(np.trace(Kc))
    fr = float(np.vdot(Kc.ravel(), Kc.ravel()))
    return tr * tr / fr, tr, fr


def family_trend(G: np.ndarray, sizes: list[int]) -> dict:
    """lm_trend (analysis/s12/aggregate.py) for an arbitrary family: rows of G
    = family members, columns = ViTs. Tr = mean over ViTs of Spearman(size,
    score); exact p over all relabellings of the sizes (same for every ViT)."""
    Y = np.asarray(G, dtype=float)
    ok = np.all(np.isfinite(Y), axis=0)
    Y = Y[:, ok]
    x = rankdata(sizes)
    P = np.array([np.array(p) for p in itertools.permutations(x)])
    rho = A12._rho_matrix(P, Y)
    obs_idx = int(np.flatnonzero(np.all(P == x, axis=1))[0])
    with np.errstate(invalid="ignore"):
        T = np.nanmean(rho, axis=1)
    t_obs = float(T[obs_idx])
    rho_v = rho[obs_idx]
    eps = 1e-12
    return {"Tr": t_obs, "rho_per_vit": [None if not np.isfinite(v) else float(v) for v in rho_v],
            "n_vit_used": int(np.sum(np.isfinite(rho_v))),
            "n_vit_constant": int(np.sum(~np.isfinite(rho_v)) + np.sum(~ok)),
            "p_one": float(np.mean(T >= t_obs - eps)), "p_two": float(np.mean(np.abs(T) >= abs(t_obs) - eps)),
            "n_relabel": int(P.shape[0])}


def jsonable(v):
    return C.jsonable(v)


def main():
    t0 = time.time()
    started = utc()
    # ---------------- guard -------------------------------------------------
    guard = {"pairs_dir": PAIRS_GUARD, "pairs_dir_exists_at_start": os.path.exists(PAIRS_GUARD),
             "llama_results_listing_at_start": sorted(os.listdir(os.path.dirname(PAIRS_GUARD)))}
    if guard["pairs_dir_exists_at_start"]:
        raise SystemExit("ABORT: results/S1_2_layermax_llama/pairs exists; Part B must precede any cross-model score")
    plan_sha = sha256_file(PLAN)
    if plan_sha != PLAN_SHA:
        raise SystemExit(f"ABORT: plan hash changed: {plan_sha}")
    os.makedirs(OUTDIR, exist_ok=True)

    # ---------------- inputs ------------------------------------------------
    inputs = {}
    for stem, fam, ds in LMS + VITS:
        p = os.path.join(FEAT, ds, stem + ".npy")
        sc = sidecar(stem)
        inputs[stem] = {"path": os.path.relpath(p, PROJ), "sha256": sha256_file(p), "family": fam,
                        "kind": sc["kind"], "shape": sc["shape"], "num_params": int(sc["num_params"])}
    print(f"[{time.time() - t0:6.0f}s] hashed {len(inputs)} feature stacks", flush=True)

    # ---------------- clip thresholds (D5, all-layer float64 stack) --------
    with open(S12_CLIP) as f:
        s12c = json.load(f)
    clip = {}
    for stem, fam, ds in LMS + VITS:
        st64 = np.asarray(stack(stem), dtype=np.float64)
        c = prh_clip_threshold(st64)
        del st64
        cached = s12c.get(f"{stem}|wit1024") if ds == "wit1024" else None
        clip[stem] = {"c": c, "c_s12_cache": cached,
                      "bitwise_equal_s12_cache": (None if cached is None else bool(c == float(cached)))}
        print(f"[{time.time() - t0:6.0f}s] clip {stem}: {c!r} (S1.2 cache equal: {clip[stem]['bitwise_equal_s12_cache']})",
              flush=True)
    bad = [s for s, d in clip.items() if d["bitwise_equal_s12_cache"] is False]
    if bad:
        raise SystemExit(f"ABORT: recomputed clip threshold differs from S1.2 cache for {bad}")
    with open(os.path.join(OUTDIR, "clip_thresholds_partB.json"), "w") as f:
        json.dump(jsonable({"note": "hrhv2.prep.prh_clip_threshold on the float64 all-layer stack (D5), q=0.95; "
                                    "S1.2 models checked bitwise vs results/S1_2/cache/clip_thresholds.json",
                            "clip": clip, "written_utc": utc()}), f, indent=1)

    # ---------------- B1: per-layer PR --------------------------------------
    PR, TR = {pre: {} for pre in PRES}, {pre: {} for pre in PRES}
    for stem, fam, ds in LMS + VITS:
        st = stack(stem)
        L = st.shape[1]
        for pre in PRES:
            c = clip[stem]["c"] if pre == "prh_clip+l2" else None
            prs, trs = np.empty(L), np.empty(L)
            for l in range(L):
                prs[l], trs[l], _ = layer_pr(np.asarray(st[:, l, :]), pre, c)
            PR[pre][stem], TR[pre][stem] = prs, trs
        print(f"[{time.time() - t0:6.0f}s] PR {stem} L={L} max(prh_clip+l2)={PR['prh_clip+l2'][stem].max():.2f} "
              f"max(l2)={PR['l2'][stem].max():.2f}", flush=True)
        del st

    # ---------------- B2-B5 --------------------------------------------------
    lm_stems = [s for s, _, _ in LMS]
    vit_stems = [s for s, _, _ in VITS]
    nparams = {s: inputs[s]["num_params"] for s in inputs}
    out = {"meta": {"role": "llama-axis-predict", "plan": os.path.relpath(PLAN, PROJ), "plan_sha256": plan_sha,
                    "script_sha256": sha256_file(os.path.abspath(__file__)), "m": M, "k": KNN_K, "delta_b": DELTA_B,
                    "started_utc": started, "guard": guard, "inputs": inputs,
                    "clip_thresholds": {s: clip[s]["c"] for s in clip},
                    "mknn_baseline_every_pair": KNN_K / (M - 1), "debiased_cka_baseline_every_pair": 0.0,
                    "note": "features only; no cross-model score computed"},
           "per_pre": {}}
    for pre in PRES:
        P = PR[pre]
        lm_info, vit_info = {}, {}
        for s in lm_stems + vit_stems:
            v = P[s]
            L = v.size
            order = np.argsort(-v, kind="stable")
            info = {"L": L, "PR": v.tolist(), "trKc": TR[pre][s].tolist(),
                    "PR_star": float(v[order[0]]), "argmax": int(order[0]), "argmax_rel": float(order[0] / (L - 1)),
                    "PR_second": float(v[order[1]]), "argmax_second": int(order[1]),
                    "PR_star_over_second": float(v[order[0]] / v[order[1]]),
                    "PR_first_layer": float(v[0]), "PR_last_layer": float(v[-1]), "PR_min": float(v.min()),
                    "fixed": {str(r): {"layer": fixed_layer(L, r), "PR": float(v[fixed_layer(L, r)])} for r in DEPTHS}}
            (lm_info if s in lm_stems else vit_info)[s] = info
        # B2: Bbar (10 x 17), argmax, fixed depth
        Bbar = np.empty((len(lm_stems), len(vit_stems)))
        Bfd = {r: np.empty_like(Bbar) for r in DEPTHS}
        Bbar_bruteforce_maxdiff = 0.0
        for i, a in enumerate(lm_stems):
            for j, b in enumerate(vit_stems):
                Buw = np.sqrt(P[a][:, None] * P[b][None, :]) / (M - 1)
                Bbar[i, j] = math.sqrt(lm_info[a]["PR_star"] * vit_info[b]["PR_star"]) / (M - 1)
                Bbar_bruteforce_maxdiff = max(Bbar_bruteforce_maxdiff, abs(float(Buw.max()) - Bbar[i, j]))
                for r in DEPTHS:
                    Bfd[r][i, j] = Buw[lm_info[a]["fixed"][str(r)]["layer"], vit_info[b]["fixed"][str(r)]["layer"]]
        li = [lm_stems.index(s) for s in LLAMA]
        bi = [lm_stems.index(s) for s in BLOOMZ]
        sizes_L = [nparams[s] for s in LLAMA]
        sizes_B = [nparams[s] for s in BLOOMZ]

        def axis_block(G, rows, stems, sizes):
            tr = family_trend(G[rows], sizes)
            gaps = {}
            for (x, y) in itertools.combinations(range(len(rows)), 2):
                g = np.abs(G[rows[x]] - G[rows[y]])
                d = G[rows[y]] - G[rows[x]]           # larger minus smaller model (rows in size order)
                gaps[f"{stems[x]}|{stems[y]}"] = {"gap_per_vit": g.tolist(), "signed_larger_minus_smaller_per_vit": d.tolist(),
                                                  "median": float(np.median(g)), "min": float(g.min()),
                                                  "max": float(g.max()),
                                                  "n_vit_gap_ge_delta_b": int(np.sum(g >= DELTA_B)),
                                                  "n_vit_larger_higher": int(np.sum(d > 0))}
            # |Tr| = 1 tested with the lm_trend tolerance 1e-12 (Spearman via _rho_matrix is not exact in
            # floating point: a perfectly monotone row can give 0.9999999999999998); monotonicity also
            # checked directly on the values.
            Gm = G[rows]
            inc = bool(np.all(np.diff(Gm, axis=0) > 0))
            dec = bool(np.all(np.diff(Gm, axis=0) < 0))
            tr_is_pm1 = bool(abs(abs(tr["Tr"]) - 1.0) <= 1e-12)
            material = bool(tr_is_pm1 and (inc or dec) and all(v["median"] >= DELTA_B for v in gaps.values()))
            return {"trend": tr, "gaps": gaps, "material": material, "abs_Tr_is_1_tol1e-12": tr_is_pm1,
                    "strictly_increasing_in_size_every_vit": inc, "strictly_decreasing_in_size_every_vit": dec,
                    "min_median_gap": float(min(v["median"] for v in gaps.values())),
                    "mean_over_vits_by_model": {s: float(G[r].mean()) for s, r in zip(stems, rows)},
                    "per_vit": {s: G[r].tolist() for s, r in zip(stems, rows)}}

        assert sizes_L == sorted(sizes_L) and sizes_B == sorted(sizes_B)
        llama_axis = {"Bbar": axis_block(Bbar, li, LLAMA, sizes_L),
                      "Bfd": {str(r): axis_block(Bfd[r], li, LLAMA, sizes_L) for r in DEPTHS},
                      "spearman_size_vs_PRstar_LM": family_trend(
                          np.array([[lm_info[s]["PR_star"]] for s in LLAMA]), sizes_L)["Tr"],
                      "PRstar_LM": {s: lm_info[s]["PR_star"] for s in LLAMA},
                      "sqrt_PRstar_ratio": {f"{LLAMA[x]}|{LLAMA[y]}": math.sqrt(lm_info[LLAMA[y]]["PR_star"] /
                                                                                 lm_info[LLAMA[x]]["PR_star"])
                                            for x, y in itertools.combinations(range(3), 2)},
                      "threshold_setting_LM_layer": {s: {"u_star": lm_info[s]["argmax"], "L_A": lm_info[s]["L"],
                                                         "u_star_rel": lm_info[s]["argmax_rel"],
                                                         "runner_up": lm_info[s]["argmax_second"],
                                                         "PR_star_over_runner_up": lm_info[s]["PR_star_over_second"]}
                                                     for s in LLAMA}}
        # ViT-side argmax w* (common to all LMs)
        vit_star = {s: {"w_star": vit_info[s]["argmax"], "L_B": vit_info[s]["L"], "w_star_rel": vit_info[s]["argmax_rel"]}
                    for s in vit_stems}
        # B5 pooled ARH-style r of Bbar
        def pearson_rank(rows):
            x = np.repeat(np.array([ARH_RANK[lm_stems[i]] for i in rows], float)[:, None], len(vit_stems), axis=1).ravel()
            y = Bbar[rows].ravel()
            return float(np.corrcoef(x, y)[0, 1])
        dense = {s: k + 1 for k, s in enumerate(sorted(ARH_RANK, key=ARH_RANK.get))}
        xr = np.repeat(np.array([dense[s] for s in lm_stems], float)[:, None], len(vit_stems), axis=1).ravel()
        xl = np.repeat(np.log(np.array([nparams[s] for s in lm_stems], float))[:, None], len(vit_stems), axis=1).ravel()
        pooled = {"pearson_arh_rank_Bbar_10LM": pearson_rank(list(range(len(lm_stems)))),
                  "pearson_arh_rank_Bbar_8LM": pearson_rank(list(range(8))),
                  "pearson_dense_rank_Bbar_10LM": float(np.corrcoef(xr, Bbar.ravel())[0, 1]),
                  "pearson_log_params_Bbar_10LM": float(np.corrcoef(xl, Bbar.ravel())[0, 1]),
                  "mean_Bbar_by_lm": {s: float(Bbar[i].mean()) for i, s in enumerate(lm_stems)}}
        bloomz_retro = {"label": "retrodiction, outcome known, not a test",
                        "Bbar": axis_block(Bbar, bi, BLOOMZ, sizes_B),
                        "Bfd": {str(r): axis_block(Bfd[r], bi, BLOOMZ, sizes_B) for r in DEPTHS},
                        "threshold_setting_LM_layer": {s: {"u_star": lm_info[s]["argmax"], "L_A": lm_info[s]["L"],
                                                           "u_star_rel": lm_info[s]["argmax_rel"]} for s in BLOOMZ}}
        out["per_pre"][pre] = {"lm": lm_info, "vit": vit_info, "vit_star": vit_star,
                               "Bbar": {a: {b: float(Bbar[i, j]) for j, b in enumerate(vit_stems)} for i, a in enumerate(lm_stems)},
                               "Bfd": {str(r): {a: {b: float(Bfd[r][i, j]) for j, b in enumerate(vit_stems)}
                                                for i, a in enumerate(lm_stems)} for r in DEPTHS},
                               "Bbar_factorised_vs_bruteforce_maxabs": Bbar_bruteforce_maxdiff,
                               "llama_axis": llama_axis, "pooled": pooled, "bloomz_retro": bloomz_retro}
        la = llama_axis["Bbar"]
        print(f"== {pre}: LLaMA Tr(Bbar) {la['trend']['Tr']:+.2f} material {la['material']} "
              f"means {[round(v, 4) for v in la['mean_over_vits_by_model'].values()]} "
              f"| u* {[(d['u_star'], round(d['u_star_rel'], 2)) for d in llama_axis['threshold_setting_LM_layer'].values()]} "
              f"| Tr(Bfd0.5) {llama_axis['Bfd']['0.5']['trend']['Tr']:+.2f} Tr(Bfd1.0) {llama_axis['Bfd']['1.0']['trend']['Tr']:+.2f} "
              f"| BLOOMZ Tr(Bbar) {bloomz_retro['Bbar']['trend']['Tr']:+.2f} | pooled r {pooled['pearson_arh_rank_Bbar_10LM']:+.2f}",
              flush=True)
    out["meta"]["finished_utc"] = utc()
    out["meta"]["pairs_dir_exists_at_end"] = os.path.exists(PAIRS_GUARD)
    with open(os.path.join(OUTDIR, "partB_features.json.tmp"), "w") as f:
        json.dump(jsonable(out), f, indent=1)
    os.replace(os.path.join(OUTDIR, "partB_features.json.tmp"), os.path.join(OUTDIR, "partB_features.json"))
    print(f"[{time.time() - t0:6.0f}s] wrote {os.path.join(OUTDIR, 'partB_features.json')}", flush=True)


if __name__ == "__main__":
    main()
