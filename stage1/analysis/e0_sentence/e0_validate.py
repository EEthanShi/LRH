"""E0 code validation on S1.1 data only (ANALYSIS_PLAN_E0.md v1.0 §10: V-a0, V-a, V-b, V-c). Run by the freeze
role before E0 CODE FROZEN; touches no E0 feature (none exists) and computes no sentence-encoder score.

    OMP_NUM_THREADS=6 venv/bin/python -I analysis/e0_sentence/e0_validate.py --threads 6

V-a0  s11copy/*.py have the sha256 of analysis/s11/*.py
V-a   E0 design == S1.1 design('clip') and design('dedup'), bitwise (orders, H1, anchors, m grids)
V-b   e0_prep.build_cache on dinov2_b and bloomz_1b7 under clip (into results/E0_sentence/validation/vb_cache/)
      reproduces S1.1 d<d>_knn.npz arrays, rows_sha256 and median_D bitwise
V-c   e0_obs on the S1.1 clip caches reproduces obs_clip.parquet (29 primary pairs) bitwise; e0_evaluate reproduces
      S1.1 draw-mean obs, slope_obs and the all-k elasticity sums to <= 1e-12
Writes results/E0_sentence/validation/VALIDATION_<utc>.json; exit 0 = all pass.
"""
from __future__ import annotations

import argparse
import os
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--threads", type=int, default=6)
_ap.add_argument("--skip-vb", action="store_true")
_ARGS = _ap.parse_args() if __name__ == "__main__" else None
if _ARGS is not None:
    if not 1 <= _ARGS.threads <= 6:
        raise SystemExit("REFUSED: at most 6 threads (task rule)")
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = str(_ARGS.threads)

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e0_common as E  # noqa: E402
import e0_evaluate as EV  # noqa: E402
import e0_prep as EP  # noqa: E402
import s11v11_obs as SO  # noqa: E402

STAMP = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
COPIES = ("s11_common.py", "s11v11_common.py", "s11v11_prep.py", "s11v11_obs.py", "s11v11_evaluate.py")


def v_a0() -> dict:
    r = {f: {"copy": E.sha256_file(os.path.join(E.S11COPY, f)),
             "orig": E.sha256_file(os.path.join(E.PROJECT, "analysis", "s11", f))} for f in COPIES}
    return {"pass": all(v["copy"] == v["orig"] for v in r.values()), "files": r}


def v_a() -> dict:
    sys.path.insert(0, os.path.join(E.PROJECT, "analysis", "s11"))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("s11v11_common_orig", os.path.join(E.PROJECT, "analysis", "s11", "s11v11_common.py"))
        orig = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(orig)
    finally:
        sys.path.pop(0)
    out = {}
    for kind in ("clip", "dedup"):
        a, b = E.design(kind), orig.design(kind)
        ok = a["n_pool"] == b["n_pool"] and np.array_equal(a["dropped_rows"], b["dropped_rows"])
        for da, db in zip(a["draws"], b["draws"]):
            ok &= all(np.array_equal(da[x], db[x]) for x in ("order", "h1", "anchor_pos"))
            ok &= da["m_grid"] == db["m_grid"] and da["M"] == db["M"]
        out[kind] = {"pass": bool(ok), "n_draws": len(a["draws"]), "n_anchors_d0": int(a["draws"][0]["anchor_pos"].size),
                     "M": [d["M"] for d in a["draws"]]}
    return {"pass": all(v["pass"] for v in out.values()), **out}


def v_b(models=("dinov2_b", "bloomz_1b7")) -> dict:
    out = {}
    for mdl in models:
        t0 = time.perf_counter()
        dst = os.path.join(E.VALID, f"vb_cache_{STAMP}", "clip", mdl)
        cm = EP.build_cache("clip", mdl, dst, save_rows=True)
        ref = E.cache_dir("clip", mdl)
        with open(os.path.join(ref, "meta.json")) as f:
            rmeta = json.load(f)
        ok = True
        details = {}
        for d in range(5):
            A = np.load(os.path.join(dst, f"d{d}_knn.npz"))
            B = np.load(os.path.join(ref, f"d{d}_knn.npz"))
            same_keys = sorted(A.files) == sorted(B.files)
            arr_ok = same_keys and all(A[k].dtype == B[k].dtype and np.array_equal(A[k], B[k]) for k in B.files)
            rows_ok = cm["draws"][d]["rows_sha256"] == rmeta["draws"][str(d)]["rows_sha256"]
            fit_ok = E.sha256_file(os.path.join(dst, f"d{d}_fit.npy")) == E.sha256_file(os.path.join(ref, f"d{d}_fit.npy"))
            med_ok = all(float(cm["draws"][d]["median_D"][int(m)]) == float(v) for m, v in rmeta["draws"][str(d)]["median_D"].items())
            details[d] = {"knn_arrays_equal": bool(arr_ok), "rows_sha256_equal": bool(rows_ok), "fit_sha256_equal": bool(fit_ok),
                          "median_D_equal": bool(med_ok)}
            ok &= arr_ok and rows_ok and fit_ok and med_ok
        ok &= cm["n_zero_offdiag_pool"] == rmeta["n_zero_offdiag_pool"]
        out[mdl] = {"pass": bool(ok), "draws": details, "seconds": time.perf_counter() - t0,
                    "clip_threshold": cm["info"].get("clip_threshold"), "s11_clip_threshold": rmeta["info"].get("clip_threshold")}
    return {"pass": all(v["pass"] for v in out.values()), **out}


def v_c() -> dict:
    des = E.design("clip")
    pairs = E.old_pairs()
    caches, meds = {}, {}
    import e0_obs as EO
    for P in pairs:
        for mdl in (P["a"], P["b"]):
            if mdl not in caches:
                caches[mdl], meds[mdl], _ = EO.load_cache(E.cache_dir("clip", mdl), des)
    rows = []
    for P in pairs:
        rows += SO.pair_rows("clip", P, des, caches[P["a"]], caches[P["b"]], meds[P["a"]], meds[P["b"]])
    new = pd.DataFrame(rows)
    ref = pd.read_parquet(os.path.join(E.S11_OUT, "obs", "obs_clip.parquet"))
    ref = ref[ref["primary"]].reset_index(drop=True)
    key = ["pair", "draw", "m", "k"]
    a = new.sort_values(key).reset_index(drop=True)
    b = ref.sort_values(key).reset_index(drop=True)
    cols_equal = {}
    for c in b.columns:
        x, y = a[c], b[c]
        if x.dtype.kind == "f":
            eq = bool(np.array_equal(x.to_numpy(), y.to_numpy(), equal_nan=True))
        else:
            eq = bool((x.astype(str) == y.astype(str)).all())
        cols_equal[c] = eq
    obs_ok = list(a.columns) == list(b.columns) and len(a) == len(b) and all(cols_equal.values())
    # evaluation
    df, ds, S, c = EV.arm_tables(new)
    c11 = pd.read_parquet(os.path.join(E.S11_OUT, "S1_1_v11_cells_clip.parquet"))
    c11 = c11[c11["primary"]].set_index(["pair", "m", "k"]).sort_index()
    S11 = pd.read_parquet(os.path.join(E.S11_OUT, "S1_1_v11_slopes_clip.parquet"))
    S11 = S11[S11["primary"]].set_index(["pair", "k"]).sort_index()
    cc = c.set_index(["pair", "m", "k"]).sort_index()
    Ss = S.set_index(["pair", "k"]).sort_index()
    d_obs = float(np.max(np.abs(cc["obs"].to_numpy() - c11.loc[cc.index, "obs"].to_numpy())))
    d_slope = float(np.max(np.abs(Ss["slope_obs"].to_numpy() - S11.loc[Ss.index, "slope_obs"].to_numpy())))
    with open(os.path.join(E.S11_OUT, "S1_1_v11_summary.json")) as f:
        summ = json.load(f)
    ref_el = pd.DataFrame(summ["configs"]["clip"]["secondary"]["primary"]["elasticity_per_pair"]).set_index("pair")
    el = EV.elasticities(c, (1, 10, 100)).set_index("pair")
    d_sum = float(np.max(np.abs(el["sum"] - ref_el.loc[el.index, "sum"])))
    ev_ok = d_obs <= 1e-12 and d_slope <= 1e-12 and d_sum <= 1e-12
    return {"pass": bool(obs_ok and ev_ok), "obs_bitwise": bool(obs_ok), "n_rows": int(len(a)),
            "columns_equal": cols_equal, "max_abs_diff_drawmean_obs": d_obs, "max_abs_diff_slope_obs": d_slope,
            "max_abs_diff_elasticity_sum": d_sum, "n_pairs": int(a["pair"].nunique())}


def main():
    args = _ARGS
    E.require_milestones("plan")
    res = {"utc_start": E.utcnow(), "threads": args.threads, "plan": "ANALYSIS_PLAN_E0.md v1.0 §10"}
    t0 = time.perf_counter()
    res["V-a0"] = v_a0()
    res["V-a"] = v_a()
    res["V-b"] = v_b() if not args.skip_vb else {"pass": None, "skipped": True}
    res["V-c"] = v_c()
    res["seconds"] = time.perf_counter() - t0
    res["utc_end"] = E.utcnow()
    res["all_pass"] = all(res[k]["pass"] for k in ("V-a0", "V-a", "V-b", "V-c"))
    h = E.write_json_new(os.path.join(E.VALID, f"VALIDATION_{STAMP}.json"), res)
    print(json.dumps(E.jsonable({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v) for k, v in res.items()}), indent=1))
    print("VALIDATION file sha256", h)
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
