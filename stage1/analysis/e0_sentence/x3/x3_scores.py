"""X3 (EXPLORATORY, ANALYSIS_PLAN_E0.md v1.0 section 7.6) scores; role x3-run. Thin wrapper over frozen E0 code.

Readings: analysis/e0_sentence/x3/X3_READINGS.md (STATUS "[x3-run] READINGS FIXED", sha256 checked here).
Imports, never edits: e0_common (ctr_features, design, cache_dir, sha256), e0_prep.build_cache (S1.1 kNN cache
layout through the byte-identical s11copy functions), e0_obs.load_cache, s11copy/s11v11_obs.pair_rows (cells),
e0_evaluate.arm_tables / elasticities / levels (draw means, slopes, elasticities, levels incl. the plan section 6
Schnaus-matched L(k, 10000)).

  1. ctr caches for the 4 X3 vision features (+ the E0 reference feature rebuilt for validation), 1 BLAS thread
     per worker (E0 built its caches with 1 thread per worker), at most 6 workers;
  2. VALIDATION (readings section 6, STOP rule): rebuilt reference cache == E0 cache bitwise; wrapper obs rows of
     the 3 reference pairs == E0 obs_CA rows bitwise;
  3. X3 obs (12 pairs: 4 features x 3 encoders, text side = E0's ctr N5 caches, read only), evaluation tables.

    taskset -c 0-5 venv/bin/python -I analysis/e0_sentence/x3/x3_scores.py --workers 5
Never overwrites; writes only under results/E0_sentence/x3/.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E0DIR = os.path.dirname(HERE)
sys.path.insert(0, E0DIR)
import e0_common as E  # noqa: E402  (frozen)
import e0_prep as EP  # noqa: E402  (frozen; argparse only under __main__)
import e0_obs as EO  # noqa: E402  (frozen)
import e0_evaluate as EV  # noqa: E402  (frozen)
import s11v11_obs as SO  # noqa: E402  (byte-identical S1.1 copy)

X3OUT = os.path.join(E.OUT, "x3")
X3CACHE = os.path.join(X3OUT, "cache")
VALDIR = os.path.join(X3OUT, "validation")
VALCACHE = os.path.join(VALDIR, "cache")
OBSDIR = os.path.join(X3OUT, "obs")
EVALDIR = os.path.join(X3OUT, "eval")
READINGS = os.path.join(HERE, "X3_READINGS.md")
EXTRACTION_X3 = os.path.join(X3OUT, "extraction", "EXTRACTION_X3.json")
SLUG = E.C.VISION["dinov2_b"]
S11_NPY = os.path.join(E.C.COCO_DIR, SLUG + ".npy")
S11_TOK = os.path.join(E.C.COCO_DIR, SLUG + ".tokmean.npy")
X3_FEATURES = ("x3_cls_r10", "x3_tok_r10", "x3_tok_r05", "x3add_schnaus224")
REF = "dinov2_b_ref"
X3_CLASS = "x3-exploratory"


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def status_line(tag: str):
    for ln in E.status_lines():
        if tag in ln:
            return ln
    return None


def preconditions() -> dict:
    out = {}
    ln = status_line("[x3-run] READINGS FIXED")
    h = E.sha256_file(READINGS)
    if ln is None or h not in ln:
        raise SystemExit("REFUSED: STATUS lacks '[x3-run] READINGS FIXED' with the readings' sha256")
    out["readings_sha256"] = h
    ln = status_line("[x3-run] EXTRACTION DONE")
    hx = E.sha256_file(EXTRACTION_X3)
    if ln is None or hx not in ln:
        raise SystemExit("REFUSED: STATUS lacks '[x3-run] EXTRACTION DONE' with EXTRACTION_X3.json's sha256")
    with open(EXTRACTION_X3) as f:
        man = json.load(f)
    if not man["passA_stop_rule"]["pass"]:
        raise SystemExit("REFUSED: the extraction STOP rule failed")
    out["extraction_x3_sha256"] = hx
    out["extraction_manifest"] = man
    return out


def load_feature(fid: str, man: dict):
    """(X float64 (16384, d), info) for one vision feature; source sha256 verified."""
    meta = E.C.sidecar(SLUG)
    if fid in ("x3_cls_r10", REF):
        p, key = S11_NPY, "primary"
    elif fid in ("x3_tok_r10", "x3_tok_r05"):
        p, key = S11_TOK, "tokmean"
    elif fid == "x3add_schnaus224":
        p = os.path.join(E.PROJECT, "features", "coco_e0", "x3", "x3add_schnaus224.npy")
        want = man["files_sha256"]["features/coco_e0/x3/x3add_schnaus224.npy"]
        h = E.sha256_file(p)
        if h != want:
            raise RuntimeError(f"{p} sha256 {h} != EXTRACTION_X3 {want}")
        X = np.load(p).astype(np.float64)
        return X, {"source": os.path.relpath(p, E.PROJECT), "sha256": h, "axis": None}
    else:
        raise KeyError(fid)
    h = E.sha256_file(p)
    if h != meta["files_sha256"][key]["sha256"]:
        raise RuntimeError(f"{p} sha256 mismatch with the S1.1 sidecar")
    ax = {"x3_cls_r10": 3, "x3_tok_r10": 3, "x3_tok_r05": 1, REF: E.C.layer_axis(meta, E.C.REL_DEPTH)}[fid]
    if fid == REF:
        assert ax == 1
    arr = np.load(p, mmap_mode="r")
    return np.asarray(arr[:, ax, :], dtype=np.float64), {"source": os.path.relpath(p, E.PROJECT), "sha256": h, "axis": ax}


def cache_out(fid: str) -> str:
    return os.path.join(VALCACHE if fid == REF else X3CACHE, "ctr", fid)


def build_one(task):
    fid, man = task
    out = cache_out(fid)
    t0 = time.perf_counter()
    try:
        if os.path.exists(os.path.join(out, "DONE")):
            return fid, "exists", 0.0, None
        X, info = load_feature(fid, man)
        info.update({"pre": "ctr", "d": int(X.shape[1]), "x3": "EXPLORATORY X3 feature (X3_READINGS.md)",
                     "ctr": "e0_common.ctr_features; D = s11_common.features_to_D(., 'none')"})
        D = E.C.features_to_D(E.ctr_features(X), "none")
        del X
        cm = EP.build_cache("ctr", fid, out, save_rows=False, D=D, info=info)
        return fid, "ok", time.perf_counter() - t0, cm["seconds_total"]
    except Exception:
        os.makedirs(os.path.join(X3OUT, "logs"), exist_ok=True)
        with open(os.path.join(X3OUT, "logs", f"cache_{fid}_{E.utcnow()}.failed"), "w") as f:
            f.write(traceback.format_exc())
        return fid, "FAILED", time.perf_counter() - t0, None


def compare_caches(a: str, b: str, des) -> dict:
    res = {"a": os.path.relpath(a, E.PROJECT), "b": os.path.relpath(b, E.PROJECT), "draws": {}, "equal": True}
    with open(os.path.join(a, "meta.json")) as f:
        ma = json.load(f)
    with open(os.path.join(b, "meta.json")) as f:
        mb = json.load(f)
    for dd in des["draws"]:
        d = dd["draw"]
        za = np.load(os.path.join(a, f"d{d}_knn.npz"))
        zb = np.load(os.path.join(b, f"d{d}_knn.npz"))
        keys_eq = sorted(za.files) == sorted(zb.files)
        diff = [k for k in za.files if k in zb.files and not (za[k].dtype == zb[k].dtype and np.array_equal(za[k], zb[k]))]
        med_eq = ma["draws"][str(d)]["median_D"] == mb["draws"][str(d)]["median_D"]
        ok = keys_eq and not diff and med_eq
        res["draws"][d] = {"keys_equal": keys_eq, "arrays_differing": diff, "median_D_equal": med_eq, "n_arrays": len(za.files)}
        res["equal"] &= ok
    res["n_zero_offdiag_pool_equal"] = ma["n_zero_offdiag_pool"] == mb["n_zero_offdiag_pool"]
    res["equal"] &= res["n_zero_offdiag_pool_equal"]
    return res


def pair_dict(fid, enc, cls, vit_label=None):
    return {"pair": f"{vit_label or fid}__{enc}", "a": fid, "b": f"{enc}.n5", "vit": vit_label or fid, "enc": enc,
            "cls": cls, "primary": True}


def obs_for(config, pairs, cache_a_dir_of, des):
    caches, meds, files = {}, {}, {}
    rows = []
    for P in pairs:
        for side, cdir in (("a", cache_a_dir_of(P["a"])), ("b", E.cache_dir("ctr", P["b"]))):
            if cdir not in caches:
                caches[cdir], meds[cdir], f = EO.load_cache(cdir, des)
                files.update(f)
        ca, cb = cache_a_dir_of(P["a"]), E.cache_dir("ctr", P["b"])
        rows += SO.pair_rows(config, P, des, caches[ca], caches[cb], meds[ca], meds[cb])
    df = pd.DataFrame(rows)
    for P in pairs:
        for col in ("vit", "enc"):
            df.loc[df["pair"] == P["pair"], col] = P[col]
    return df, files


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    key = ["pair", "draw", "m", "k"]
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    out = {"n_rows": [len(a), len(b)], "columns_equal": set(a.columns) == set(b.columns),
           "column_order_equal": list(a.columns) == list(b.columns), "differing_columns": []}
    if len(a) != len(b):
        out["equal"] = False
        return out
    for c in b.columns:
        if c not in a.columns:
            out["differing_columns"].append(c)
            continue
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            eq = np.array_equal(x.astype(float), y.astype(float), equal_nan=True)
        else:
            eq = bool((pd.Series(x).astype(str) == pd.Series(y).astype(str)).all())
        if not eq:
            out["differing_columns"].append(c)
    out["equal"] = out["columns_equal"] and not out["differing_columns"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    if args.workers < 1 or args.workers > 6:
        raise SystemExit("REFUSED: at most 6 workers x 1 thread (task rule)")
    aff = sorted(os.sched_getaffinity(0))
    if len(aff) > 6:
        raise SystemExit(f"REFUSED: CPU affinity {aff} has more than 6 cores; run under taskset -c 0-5")
    t0 = time.perf_counter()
    utc0 = E.utcnow()
    pre = preconditions()
    man = pre.pop("extraction_manifest")
    for p in (os.path.join(OBSDIR, "obs_X3.parquet"), os.path.join(VALDIR, "VALIDATION_X3.json")):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists (never overwrite)")
    des = E.design("clip")

    # 1. caches
    tasks = [(f, man) for f in (REF,) + X3_FEATURES]
    built = {}
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=1) as pool:
        for fid, st, sec, inner in pool.imap_unordered(build_one, tasks):
            built[fid] = {"status": st, "seconds": sec, "build_cache_seconds": inner}
            print(f"[{time.strftime('%H:%M:%S')}] cache ctr/{fid} {st} {sec:.0f}s", flush=True)
    if any(v["status"] == "FAILED" for v in built.values()):
        raise SystemExit(f"cache build failed: {built}")

    # 2. validation (STOP rule)
    val = {"role": "x3-run", "readings_sha256": pre["readings_sha256"], "created_utc": E.utcnow(), "caches_built": built}
    val["cache_ref_vs_E0"] = compare_caches(cache_out(REF), E.cache_dir("ctr", "dinov2_b"), des)
    ref_pairs = [pair_dict(REF, enc, E.NEW_CLASS, vit_label="dinov2_b") for enc in E.ENCODERS]
    # the rebuilt cache must give E0's rows; P['a'] names the rebuilt cache, the pair name and vit are E0's
    df_rebuilt, _ = obs_for("CA", ref_pairs, lambda a: cache_out(REF), des)
    df_e0cache, _ = obs_for("CA", [dict(p, a="dinov2_b") for p in ref_pairs], lambda a: E.cache_dir("ctr", "dinov2_b"), des)
    caobs = pd.read_parquet(os.path.join(E.OBS, "obs_CA.parquet"))
    caobs = caobs[caobs["pair"].isin([p["pair"] for p in ref_pairs])]
    val["obs_obs_CA_sha256"] = E.sha256_file(os.path.join(E.OBS, "obs_CA.parquet"))
    val["obs_rebuilt_vs_obs_CA"] = frames_equal(df_rebuilt, caobs)
    val["obs_E0cache_vs_obs_CA"] = frames_equal(df_e0cache, caobs)
    val["pass"] = bool(val["cache_ref_vs_E0"]["equal"] and val["obs_rebuilt_vs_obs_CA"]["equal"]
                       and val["obs_E0cache_vs_obs_CA"]["equal"])
    os.makedirs(VALDIR, exist_ok=True)
    hval = E.write_json_new(os.path.join(VALDIR, "VALIDATION_X3.json"), val)
    print("validation", val["pass"], hval, flush=True)
    if not val["pass"]:
        raise SystemExit("STOP (readings section 6): the wrapper does not reproduce E0; no X3 obs written")

    # 3. X3 obs and evaluation
    pairs = [pair_dict(f, enc, X3_CLASS) for f in X3_FEATURES for enc in E.ENCODERS]
    df, files = obs_for("X3", pairs, lambda a: cache_out(a), des)
    os.makedirs(OBSDIR, exist_ok=True)
    hobs = E.save_parquet_new(os.path.join(OBSDIR, "obs_X3.parquet"), df)
    dfd, ds, S, c = EV.arm_tables(df)
    el = []
    for ks, tag in (((10, 100), "k10_100"), ((1, 10, 100), "k1_10_100"), ((1, 10), "k1_10")):
        e = EV.elasticities(c, ks)
        e["k_set"] = tag
        el.append(e)
    el = pd.concat(el, ignore_index=True)
    Lv = EV.levels(c)
    LvH2 = EV.levels(c, col="obs_allH2")
    os.makedirs(EVALDIR, exist_ok=True)
    hashes = {"obs/obs_X3.parquet": hobs}
    for name, tab in (("X3_cells", c), ("X3_drawslopes", ds), ("X3_slopes", S), ("X3_elasticity", el),
                      ("X3_levels", Lv), ("X3_levels_allH2", LvH2)):
        hashes[f"eval/{name}.parquet"] = E.save_parquet_new(os.path.join(EVALDIR, f"{name}.parquet"), tab)
    prov = {"role": "x3-run", "plan": "ANALYSIS_PLAN_E0.md v1.0 section 7.6 X3 (EXPLORATORY)", **pre,
            "pairs": pairs, "n_rows": len(df), "cache_files_sha256": files, "outputs_sha256": hashes,
            "validation_sha256": hval, "workers": args.workers, "blas_threads": 1, "cpu_affinity": aff,
            "code_sha256": {"x3_scores.py": E.sha256_file(os.path.abspath(__file__)),
                            **{f: E.sha256_file(os.path.join(E0DIR, f)) for f in ("e0_common.py", "e0_prep.py", "e0_obs.py", "e0_evaluate.py")},
                            **{f"s11copy/{f}": E.sha256_file(os.path.join(E.S11COPY, f)) for f in ("s11v11_obs.py", "s11v11_prep.py", "s11v11_evaluate.py", "s11_common.py", "s11v11_common.py")}},
            "started_utc": utc0, "finished_utc": E.utcnow(), "wall_s": time.perf_counter() - t0,
            "cpu_core_hours": cpu_seconds() / 3600, "gpu_hours": 0.0}
    hprov = E.write_json_new(os.path.join(OBSDIR, "obs_X3.json"), prov)
    print(json.dumps({"obs_X3.json": hprov, **hashes, "validation": hval, "cpu_core_h": prov["cpu_core_hours"],
                      "wall_s": prov["wall_s"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
