"""X3b independent check: neighbour-table comparison, per-anchor overlap
counts, and cells.

1. For each of the 51 models, my neighbour table vs the run-side cache that
   the run used for it (paths read from eval/X3B_pairs.parquet, as data).
2. For each of the 210 pairs (+10 S1.1 vision pairs for the 0.1885 reference):
   per-anchor overlap counts |N_k^A cap N_k^B| from MY tables, and the same
   from the RUN's cached tables; compared anchor by anchor. A differing anchor
   is accepted only under the plan's V-f near-tie clause.
3. Cells per (pair, draw, m, k) from my counts.

Usage: python x3bi_obs.py <outdir>
EXPLORATORY.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/hrh_v2_stage1_20261008"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x3bi_common as C  # noqa: E402

KCOL = {1: (0, 1), 10: (2, 3), 100: (4, 5)}   # positions in VAL_COLS of sorted col k-1, k


def load_tables(d):
    return [dict(np.load(os.path.join(d, f"d{i}_knn.npz"))) for i in range(C.N_DRAWS)]


def overlaps(ia, ib, k):
    c = np.concatenate([ia[:, :k], ib[:, :k]], axis=1)
    c.sort(axis=1)
    return (c[:, 1:] == c[:, :-1]).sum(axis=1).astype(np.int16)


def main():
    outdir = sys.argv[1]
    t0 = time.time()
    run = os.path.join(C.ROOT, "results/E0_sentence/x3b")
    rp = pd.read_parquet(os.path.join(run, "eval/X3B_pairs.parquet"))
    pairs = C.all_pairs()
    # S1.1 clip vision-vision pairs at rel 0.5 (reference 0.1885 only)
    for i, a in enumerate(C.VITS):
        for b in C.VITS[i + 1:]:
            pairs.append({"set": "VVref", "pair": f"{a}.r05__{b}.r05", "pre": "clip", "vit": a,
                          "rel_v": "0.5", "text": b, "rel_t": "0.5",
                          "key_a": f"clip/{a}.r05", "key_b": f"clip/{b}.r05"})
    # map my model keys -> run cache dirs
    runmap = {}
    rpi = rp.set_index(["set", "pair"])
    n_pairs_matched = 0
    for p in pairs:
        if p["set"] == "VVref":
            continue
        row = rpi.loc[(p["set"], p["pair"])]
        n_pairs_matched += 1
        for kk, cc in ((p["key_a"], row.cache_a), (p["key_b"], row.cache_b)):
            if kk in runmap:
                assert runmap[kk] == cc, (kk, runmap[kk], cc)
            runmap[kk] = cc
    assert n_pairs_matched == len(rp) == 210, (n_pairs_matched, len(rp))
    for v in C.VITS:   # VVref uses the S1.1 clip ViT caches
        runmap.setdefault(f"clip/{v}.r05", f"results/S1_1/v11/cache/clip/{v}")
    my_keys = sorted({C.model_key(s) for s in C.all_model_specs()})
    assert sorted(runmap) == my_keys, (set(runmap) ^ set(my_keys))

    # ---- 1. neighbour tables -------------------------------------------------
    mine, theirs, kcmp = {}, {}, {}
    for key in my_keys:
        mdir = os.path.join(outdir, "knn", key.replace("/", "__"))
        mine[key] = load_tables(mdir)
        rdir = os.path.join(C.ROOT, runmap[key])
        theirs[key] = load_tables(rdir)
        mm = json.load(open(os.path.join(mdir, "meta.json")))
        rm = json.load(open(os.path.join(rdir, "meta.json")))
        rinfo = rm.get("info", {})
        if isinstance(rinfo, str):
            rinfo = {}
        rec = {"run_cache": runmap[key], "rows": 0, "rows_all101_equal": 0,
               "rows_first_k_equal": {str(k): 0 for k in C.KS}, "pos_equal": True,
               "vals_max_abs_diff": 0.0, "median_D_max_abs_diff": 0.0,
               "exact_tie_rows_mine": {str(k): 0 for k in C.KS}}
        for d in range(C.N_DRAWS):
            A, B = mine[key][d], theirs[key][d]
            for m in C.M_GRID:
                pa, pb = A[f"pos_m{m}"], B[f"pos_m{m}"]
                if not np.array_equal(pa, pb):
                    rec["pos_equal"] = False
                    continue
                ia, ib = A[f"idx_m{m}"], B[f"idx_m{m}"]
                rec["rows"] += len(pa)
                rec["rows_all101_equal"] += int((ia == ib).all(axis=1).sum())
                for k in C.KS:
                    rec["rows_first_k_equal"][str(k)] += int((ia[:, :k] == ib[:, :k]).all(axis=1).sum())
                    c0, c1 = KCOL[k]
                    rec["exact_tie_rows_mine"][str(k)] += int((A[f"vals_m{m}"][:, c0] == A[f"vals_m{m}"][:, c1]).sum())
                rec["vals_max_abs_diff"] = max(rec["vals_max_abs_diff"],
                                               float(np.max(np.abs(A[f"vals_m{m}"] - B[f"vals_m{m}"]))))
            md = mm["draws"][str(d)]["median_D"]
            rdd = rm["draws"][str(d)]["median_D"] if isinstance(rm.get("draws"), dict) else None
            if rdd:
                rec["median_D_max_abs_diff"] = max(rec["median_D_max_abs_diff"],
                                                   max(abs(md[str(m)] - rdd[str(m)]) for m in C.M_GRID))
        if "clip_threshold" in mm["info"]:
            rec["clip_threshold_mine"] = mm["info"]["clip_threshold"]
            rec["clip_threshold_run"] = rinfo.get("clip_threshold")
        rec["n_zero_offdiag_ordered_mine"] = mm["n_zero_offdiag_ordered"]
        rec["n_zero_offdiag_pool_run"] = rm.get("n_zero_offdiag_pool")
        kcmp[key] = rec
        print(key, rec["rows_all101_equal"], "/", rec["rows"], "vals", rec["vals_max_abs_diff"], flush=True)

    # ---- 2./3. per-anchor counts and cells ----------------------------------
    cells, mism = [], []
    ov_store = {}
    n_anchor_units = 0
    n_anchor_units_equal = 0
    for p in pairs:
        A, B = mine[p["key_a"]], mine[p["key_b"]]
        RA, RB = theirs[p["key_a"]], theirs[p["key_b"]]
        medA = json.load(open(os.path.join(outdir, "knn", p["key_a"].replace("/", "__"), "meta.json")))["draws"]
        medB = json.load(open(os.path.join(outdir, "knn", p["key_b"].replace("/", "__"), "meta.json")))["draws"]
        store = {}
        for d in range(C.N_DRAWS):
            for m in C.M_GRID:
                pos = A[d][f"pos_m{m}"]
                assert np.array_equal(pos, B[d][f"pos_m{m}"])
                obsrows = (pos < C.M_FIT) if m >= C.M_FIT else np.ones(len(pos), bool)
                mA = medA[str(d)]["median_D"][str(m)]
                mB = medB[str(d)]["median_D"][str(m)]
                for k in C.KS:
                    ov = overlaps(A[d][f"idx_m{m}"], B[d][f"idx_m{m}"], k)
                    ovr = overlaps(RA[d][f"idx_m{m}"], RB[d][f"idx_m{m}"], k)
                    n_anchor_units += len(ov)
                    eq = ov == ovr
                    n_anchor_units_equal += int(eq.sum())
                    c0, c1 = KCOL[k]
                    vA, vB = A[d][f"vals_m{m}"], B[d][f"vals_m{m}"]
                    gapA = vA[:, c1] - vA[:, c0]
                    gapB = vB[:, c1] - vB[:, c0]
                    if not eq.all():
                        for r in np.flatnonzero(~eq):
                            rgA = RA[d][f"vals_m{m}"][r, c1] - RA[d][f"vals_m{m}"][r, c0]
                            rgB = RB[d][f"vals_m{m}"][r, c1] - RB[d][f"vals_m{m}"][r, c0]
                            mism.append({"set": p["set"], "pair": p["pair"], "draw": d, "m": m, "k": k,
                                         "row": int(r), "pos": int(pos[r]), "mine": int(ov[r]), "run": int(ovr[r]),
                                         "gapA_mine": float(gapA[r]), "gapB_mine": float(gapB[r]),
                                         "gapA_run": float(rgA), "gapB_run": float(rgB),
                                         "medA": mA, "medB": mB})
                    store[f"d{d}_m{m}_k{k}"] = ov
                    tieA = gapA == 0
                    tieB = gapB == 0
                    frac = ov.astype(np.float64) / k
                    fo = frac[obsrows]
                    clean = obsrows & ~tieA & ~tieB
                    ntA = (gapA < 1e-12 * mA)
                    ntB = (gapB < 1e-12 * mB)
                    cells.append({
                        "set": p["set"], "pair": p["pair"], "draw": d, "m": m, "k": k,
                        "chance": k / (m - 1),
                        "n_anchors": int(obsrows.sum()), "obs": float(fo.mean()),
                        "obs_sd0": float(fo.std(ddof=0)), "obs_sd1": float(fo.std(ddof=1)),
                        "n_allH2": int(len(frac)), "obs_allH2": float(frac.mean()),
                        "obs_allH2_sd0": float(frac.std(ddof=0)), "obs_allH2_sd1": float(frac.std(ddof=1)),
                        "n_tieA": int(tieA[obsrows].sum()), "n_tieB": int(tieB[obsrows].sum()),
                        "n_tieA_allH2": int(tieA.sum()), "n_tieB_allH2": int(tieB.sum()),
                        "n_clean": int(clean.sum()),
                        "obs_clean": float(frac[clean].mean()) if clean.any() else float("nan"),
                        "neartie_A": float(ntA[obsrows].mean()), "neartie_B": float(ntB[obsrows].mean()),
                        "neartie_allH2_A": float(ntA.mean()), "neartie_allH2_B": float(ntB.mean()),
                        "is_fit_m": m == C.M_FIT,
                    })
        ov_store[p["set"] + "|" + p["pair"]] = store
    os.makedirs(os.path.join(outdir, "obs"), exist_ok=True)
    cdf = pd.DataFrame(cells)
    cpath = os.path.join(outdir, "obs", "x3bi_cells.parquet")
    cdf.to_parquet(cpath, index=False)
    np.savez_compressed(os.path.join(outdir, "obs", "x3bi_anchor_counts.npz"),
                        **{f"{k}|{kk}": v for k, st in ov_store.items() for kk, v in st.items()})
    # near-tie clause (ANALYSIS_PLAN_E0 10, V-f): a differing anchor is accepted only if its
    # k-boundary gap < 1e-12 x prefix median D in either implementation (in the model whose
    # list differs) AND the two D rows differ by <= 1e-12 (checked through the stored values)
    acc = []
    for x in mism:
        ok = (min(x["gapA_mine"], x["gapA_run"]) < 1e-12 * x["medA"]) or \
             (min(x["gapB_mine"], x["gapB_run"]) < 1e-12 * x["medB"])
        acc.append(ok)
    out = {"knn_compare": kcmp,
           "anchor_units": n_anchor_units, "anchor_units_equal": n_anchor_units_equal,
           "n_mismatch": len(mism), "n_mismatch_accepted_by_neartie_clause": int(sum(acc)),
           "mismatches": mism[:500], "n_pairs": len(pairs), "seconds": time.time() - t0,
           "cells_sha256": C.sha256_file(cpath)}
    with open(os.path.join(outdir, "obs", "x3bi_obs_compare.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    tm = os.times()
    print("OBS DONE", n_anchor_units_equal, "/", n_anchor_units, "mismatch", len(mism),
          "cpu_s", tm.user + tm.system, "wall", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
