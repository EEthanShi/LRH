"""X3b (EXPLORATORY follow-up to X3) scores; role x3b-run. Thin wrapper over frozen S1.1 / E0 code.

Readings: analysis/e0_sentence/x3b/X3B_READINGS.md (STATUS "[x3b-run] READINGS FIXED", sha256 checked here).
Imports, never edits: e0_common (ctr_features, design, cache_dir, sha256, writers), s11copy/s11_common
(model_D with rel, layer_axis, features_to_D, pair_list), e0_prep.build_cache (S1.1 kNN cache layout),
e0_obs.load_cache, s11copy/s11v11_obs.pair_rows (cells), e0_evaluate.arm_tables / elasticities / levels.

  phase 1  validation caches (readings section 6): clip dinov2_b and pythia410m_s1 at rel 0.5, ctr dinov2_b at rel 1.0
  phase 2  VALIDATION (STOP rule): caches bitwise; obs rows dinov2_b__pythia410m_s1 (obs_clip) and
           x3_cls_r10__mpnet (obs_X3) bitwise; 60 mid-depth reference pairs (obs_clip, obs_l2, obs_ctrold VL, obs_CA)
  phase 3  23 new single-model caches (1 BLAS thread per worker, at most 6 workers)
  phase 4  X3b obs (sets A1, A2_r025, A2_r075, A2_r10, A3, A4, B) and evaluation tables

    taskset -c 0-5 nice -n 10 venv/bin/python -I analysis/e0_sentence/x3b/x3b_scores.py --workers 6
Never overwrites; writes only under results/E0_sentence/x3b/.
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

C = E.C
OUT = os.path.join(E.OUT, "x3b")
CACHE = os.path.join(OUT, "cache")
VALDIR = os.path.join(OUT, "validation")
VALCACHE = os.path.join(VALDIR, "cache")
OBSDIR = os.path.join(OUT, "obs")
EVALDIR = os.path.join(OUT, "eval")
LOGS = os.path.join(OUT, "logs")
READINGS = os.path.join(HERE, "X3B_READINGS.md")
X3_DINO_R10 = os.path.join(E.OUT, "x3", "cache", "ctr", "x3_cls_r10")
X3_OBS = os.path.join(E.OUT, "x3", "obs", "obs_X3.parquet")
S11_OBS = {"clip": os.path.join(E.S11_OUT, "obs", "obs_clip.parquet"), "l2": os.path.join(E.S11_OUT, "obs", "obs_l2.parquet")}
E0_OBS = {"ctrold": os.path.join(E.OBS, "obs_ctrold.parquet"), "CA": os.path.join(E.OBS, "obs_CA.parquet")}

VITS = tuple(E.VITS)                                   # dinov2_b, clip_b, mae_b, augreg_b, siglip_b
LMS = ("bloomz_1b7", "openllama_3b", "pythia410m_s1")  # S1.1 vision-language text side
ENCS = tuple(E.ENCODERS)                               # mpnet, contriever, qwen3emb06
RTAG = {"0.25": "r025", "0.5": "r05", "0.75": "r075", "1.0": "r10"}
KEY = ["pair", "draw", "m", "k"]


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def preconditions() -> dict:
    h = E.sha256_file(READINGS)
    for ln in E.status_lines():
        if "[x3b-run] READINGS FIXED" in ln and h in ln:
            return {"readings_sha256": h, "readings_status_utc": ln[:20]}
    raise SystemExit("REFUSED: STATUS lacks '[x3b-run] READINGS FIXED' with the readings' sha256")


# ------------------------------------------------------------------------------------------ single-model D at any rel
def model_D_rel(pre: str, model: str, rel: str):
    """D (16384 x 16384 float64) of one S1.1 model at stored relative depth ``rel`` under ``pre``.
    clip / l2: the frozen s11_common.model_D with its rel argument (clip = prh_clip+l2, pool-stored threshold
    over all 4 stored layers). ctr: e0_common.ctr_features on the stored axis of ``rel``, D = features_to_D(., 'none')
    (the X3 / E0 ctr path)."""
    slug = C.MODELS[model]
    meta = C.sidecar(slug)
    npy = os.path.join(C.COCO_DIR, slug + ".npy")
    h = E.sha256_file(npy)
    if h != meta["files_sha256"]["primary"]["sha256"]:
        raise RuntimeError(f"{npy} sha256 mismatch with the S1.1 sidecar")
    if pre == "clip":
        D, info = C.model_D(model, "prh_clip+l2", "pool-stored", rel=rel)
    elif pre == "l2":
        D, info = C.model_D(model, "l2", None, rel=rel)
    elif pre == "ctr":
        ax = C.layer_axis(meta, rel)
        arr = np.load(npy, mmap_mode="r")
        X = np.asarray(arr[:, ax, :], dtype=np.float64)
        info = {"model": model, "slug": slug, "kind": meta["kind"], "axis": ax, "layer_name": meta["layer_names"][ax],
                "d": int(X.shape[1]), "pre": "ctr", "ctr": "e0_common.ctr_features; D = features_to_D(., 'none')"}
        D = C.features_to_D(E.ctr_features(X), "none")
        del X
    else:
        raise KeyError(pre)
    info.update({"npy_sha256_checked": h, "rel": rel, "x3b": "EXPLORATORY X3b (X3B_READINGS.md)"})
    return D, info


def build_one(task):
    pre, model, rel, out = task
    t0 = time.perf_counter()
    try:
        if os.path.exists(os.path.join(out, "DONE")):
            return task, "exists", 0.0
        D, info = model_D_rel(pre, model, rel)
        EP.build_cache(pre, f"{model}.{RTAG[rel]}", out, save_rows=False, D=D, info=info)
        return task, "ok", time.perf_counter() - t0
    except Exception:
        os.makedirs(LOGS, exist_ok=True)
        with open(os.path.join(LOGS, f"cache_{pre}_{model}_{RTAG[rel]}_{E.utcnow()}.failed"), "w") as f:
            f.write(traceback.format_exc())
        return task, "FAILED", time.perf_counter() - t0


def run_pool(tasks, workers):
    built = {}
    with mp.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        for task, st, sec in pool.imap_unordered(build_one, tasks):
            tag = f"{task[0]}/{task[1]}.{RTAG[task[2]]}"
            built[tag] = {"status": st, "seconds": sec, "out": os.path.relpath(task[3], E.PROJECT)}
            print(f"[{time.strftime('%H:%M:%S')}] cache {tag} {st} {sec:.0f}s", flush=True)
    if any(v["status"] == "FAILED" for v in built.values()):
        raise SystemExit(f"cache build failed: {built}")
    return built


# ------------------------------------------------------------------------------------------ cache locations
def cdir(pre: str, model: str, rel: str) -> str:
    """Existing caches are reused read only; everything else lives under results/E0_sentence/x3b/cache."""
    if model in ENCS:                                         # sentence encoders: E0 ctr N5 caches
        assert pre == "ctr" and rel == "n5"
        return E.cache_dir("ctr", f"{model}.n5")
    if rel == "0.5":
        return E.cache_dir(pre, model)                        # S1.1 clip / l2, E0 ctr
    if pre == "ctr" and model == "dinov2_b" and rel == "1.0":
        return X3_DINO_R10                                    # X3 x3_cls_r10 (same stored feature)
    return os.path.join(CACHE, pre, f"{model}.{RTAG[rel]}")


def new_cache_tasks():
    t = []
    for v in VITS:
        t.append(("clip", v, "1.0", cdir("clip", v, "1.0")))
    for lm in LMS:
        for r in ("0.25", "0.75", "1.0"):
            t.append(("clip", lm, r, cdir("clip", lm, r)))
    for v in VITS:
        t.append(("l2", v, "1.0", cdir("l2", v, "1.0")))
    for v in VITS:
        if v != "dinov2_b":
            t.append(("ctr", v, "1.0", cdir("ctr", v, "1.0")))
    assert len(t) == 23
    return t


VAL_TASKS = [("clip", "dinov2_b", "0.5", os.path.join(VALCACHE, "clip", "dinov2_b.r05")),
             ("clip", "pythia410m_s1", "0.5", os.path.join(VALCACHE, "clip", "pythia410m_s1.r05")),
             ("ctr", "dinov2_b", "1.0", os.path.join(VALCACHE, "ctr", "dinov2_b.r10"))]


# ------------------------------------------------------------------------------------------ comparisons
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
        diff = [k for k in zb.files if not (k in za.files and za[k].dtype == zb[k].dtype and np.array_equal(za[k], zb[k]))]
        med_a = {str(k): v for k, v in ma["draws"][str(d)]["median_D"].items()}
        med_b = {str(k): v for k, v in mb["draws"][str(d)]["median_D"].items()}
        med_eq = med_a == med_b
        res["draws"][d] = {"keys_equal": keys_eq, "arrays_differing": diff, "median_D_equal": med_eq,
                           "n_arrays": len(zb.files)}
        res["equal"] &= bool(keys_eq and not diff and med_eq)
    res["n_zero_offdiag_pool_equal"] = ma["n_zero_offdiag_pool"] == mb["n_zero_offdiag_pool"]
    res["equal"] &= res["n_zero_offdiag_pool_equal"]
    res["clip_threshold"] = [ma["info"].get("clip_threshold"), mb["info"].get("clip_threshold")]
    return res


def frames_equal(new: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Bitwise equality on every column of ``ref`` (float columns: exact, NaN == NaN; others: as strings)."""
    a = new.sort_values(KEY).reset_index(drop=True)
    b = ref.sort_values(KEY).reset_index(drop=True)
    out = {"n_rows": [len(a), len(b)], "ref_columns_present": all(c in a.columns for c in b.columns),
           "differing_columns": [], "dtype_differences": []}
    if len(a) != len(b) or not out["ref_columns_present"]:
        out["equal"] = False
        return out
    for c in b.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            eq = x.dtype.kind == y.dtype.kind and np.array_equal(x.astype(np.float64), y.astype(np.float64), equal_nan=True)
        else:
            eq = bool((pd.Series(x).astype(str) == pd.Series(y).astype(str)).all())
            if x.dtype.kind != y.dtype.kind and not (x.dtype.kind in "OUT" and y.dtype.kind in "OUT"):
                out["dtype_differences"].append([c, str(x.dtype), str(y.dtype)])
        if not eq:
            out["differing_columns"].append(c)
    out["equal"] = not out["differing_columns"]
    return out


class Caches:
    def __init__(self, des):
        self.des, self.tabs, self.meds, self.files = des, {}, {}, {}

    def get(self, d):
        if d not in self.tabs:
            self.tabs[d], self.meds[d], f = EO.load_cache(d, self.des)
            self.files.update(f)
        return self.tabs[d], self.meds[d]


def rows_for(config, P, dirA, dirB, cache: Caches, extra=None):
    tA, mA = cache.get(dirA)
    tB, mB = cache.get(dirB)
    rows = SO.pair_rows(config, P, cache.des, tA, tB, mA, mB)
    df = pd.DataFrame(rows)
    for col, v in (extra or {}).items():
        df[col] = v
    return df


# ------------------------------------------------------------------------------------------ validation
def validation(des, cache: Caches, built_val) -> dict:
    val = {"role": "x3b-run", "created_utc": E.utcnow(), "caches_built": built_val, "checks": {}}
    chk = val["checks"]
    # 1. S1.1 cell
    chk["cache_clip_dinov2_b_r05_vs_S11"] = compare_caches(VAL_TASKS[0][3], E.cache_dir("clip", "dinov2_b"), des)
    chk["cache_clip_pythia410m_s1_r05_vs_S11"] = compare_caches(VAL_TASKS[1][3], E.cache_dir("clip", "pythia410m_s1"), des)
    P = dict(C.PAIRS["dinov2_b__pythia410m_s1"])
    new = rows_for("clip", P, VAL_TASKS[0][3], VAL_TASKS[1][3], cache)
    ref = pd.read_parquet(S11_OBS["clip"])
    chk["obs_dinov2_b__pythia410m_s1_rebuilt_vs_obs_clip"] = frames_equal(new, ref[ref["pair"] == P["pair"]])
    # 2. X3 cell
    chk["cache_ctr_dinov2_b_r10_vs_X3_x3_cls_r10"] = compare_caches(VAL_TASKS[2][3], X3_DINO_R10, des)
    P = {"pair": "x3_cls_r10__mpnet", "a": "x3_cls_r10", "b": "mpnet.n5", "vit": "x3_cls_r10", "enc": "mpnet",
         "cls": "x3-exploratory", "primary": True}
    new = rows_for("X3", P, VAL_TASKS[2][3], cdir("ctr", "mpnet", "n5"), cache, {"vit": "x3_cls_r10", "enc": "mpnet"})
    ref = pd.read_parquet(X3_OBS)
    chk["obs_x3_cls_r10__mpnet_rebuilt_vs_obs_X3"] = frames_equal(new, ref[ref["pair"] == P["pair"]])
    # 3. the 60 mid-depth reference pairs, from the existing caches
    refs = {}
    vl = [p for p in C.pair_list() if p["cls"] == "cross-modal"]
    assert len(vl) == 15
    for name, pre, path in (("clip", "clip", S11_OBS["clip"]), ("l2", "l2", S11_OBS["l2"]), ("ctrold", "ctr", E0_OBS["ctrold"])):
        df = pd.concat([rows_for(name, dict(p), cdir(pre, p["a"], "0.5"), cdir(pre, p["b"], "0.5"), cache) for p in vl],
                       ignore_index=True)
        ref = pd.read_parquet(path)
        chk[f"obs_ref_{name}_15VL"] = frames_equal(df, ref[ref["pair"].isin([p["pair"] for p in vl])])
        refs[name] = df
    ca = E.new_pairs("CA")
    df = pd.concat([rows_for("CA", P, cdir("ctr", P["vit"], "0.5"), cdir("ctr", P["enc"], "n5"), cache,
                             {"vit": P["vit"], "enc": P["enc"]}) for P in ca], ignore_index=True)
    chk["obs_ref_CA_15"] = frames_equal(df, pd.read_parquet(E0_OBS["CA"]))
    refs["CA"] = df
    val["input_sha256"] = {os.path.relpath(p, E.PROJECT): E.sha256_file(p)
                           for p in (S11_OBS["clip"], S11_OBS["l2"], E0_OBS["ctrold"], E0_OBS["CA"], X3_OBS)}
    val["pass"] = all(v["equal"] for v in chk.values())
    return val, refs


# ------------------------------------------------------------------------------------------ X3b pairs
def x3b_pairs():
    """(set, pre, vit, rel_v, text, rel_t) for every X3b pair; 165 pairs."""
    out = []
    for v in VITS:
        for lm in LMS:
            out.append(("A1", "clip", v, "1.0", lm, "0.5"))
    for r in ("0.25", "0.75", "1.0"):
        for rv in ("0.5", "1.0"):
            for v in VITS:
                for lm in LMS:
                    out.append((f"A2_{RTAG[r]}", "clip", v, rv, lm, r))
    for v in VITS:
        for lm in LMS:
            out.append(("A3", "l2", v, "1.0", lm, "0.5"))
    for v in VITS:
        for lm in LMS:
            out.append(("A4", "ctr", v, "1.0", lm, "0.5"))
    for v in VITS:
        for enc in ENCS:
            out.append(("B", "ctr", v, "1.0", enc, "n5"))
    assert len(out) == 15 + 90 + 15 + 15 + 15
    return out


def pname(v, rv, t, rt):
    return f"{v}.{RTAG[rv]}__{t}.{'n5' if rt == 'n5' else RTAG[rt]}"


def evaluate(df_all: pd.DataFrame):
    tabs = {n: [] for n in ("cells", "drawslopes", "slopes", "elasticity", "levels", "levels_allH2")}
    for s, df in df_all.groupby("config", sort=True):
        _, ds, S, c = EV.arm_tables(df)
        el = []
        for ks, tag in (((10, 100), "k10_100"), ((1, 10, 100), "k1_10_100"), ((1, 10), "k1_10")):
            e = EV.elasticities(c, ks)
            e["k_set"] = tag
            el.append(e)
        for name, t in (("cells", c), ("drawslopes", ds), ("slopes", S), ("elasticity", pd.concat(el, ignore_index=True)),
                        ("levels", EV.levels(c)), ("levels_allH2", EV.levels(c, col="obs_allH2"))):
            t = t.copy()
            t.insert(0, "set", s)
            tabs[name].append(t)
    return {n: pd.concat(v, ignore_index=True) for n, v in tabs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if not 1 <= args.workers <= 6:
        raise SystemExit("REFUSED: at most 6 workers x 1 BLAS thread (task rule)")
    aff = sorted(os.sched_getaffinity(0))
    if len(aff) > 6:
        raise SystemExit(f"REFUSED: CPU affinity {aff} has more than 6 cores; run under taskset -c 0-5")
    t0, utc0 = time.perf_counter(), E.utcnow()
    pre = preconditions()
    for p in (os.path.join(OBSDIR, "obs_X3B.parquet"), os.path.join(VALDIR, "VALIDATION_X3B.json")):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists (never overwrite)")
    des = E.design("clip")

    # phase 1-2: validation caches, then the STOP rule
    built_val = run_pool(VAL_TASKS, min(args.workers, len(VAL_TASKS)))
    cache = Caches(des)
    val, refs = validation(des, cache, built_val)
    val.update(pre)
    os.makedirs(VALDIR, exist_ok=True)
    hval = E.write_json_new(os.path.join(VALDIR, "VALIDATION_X3B.json"), val)
    print("validation", val["pass"], hval, {k: v["equal"] for k, v in val["checks"].items()}, flush=True)
    if not val["pass"]:
        raise SystemExit("STOP (readings section 6): the wrapper does not reproduce S1.1 / X3 / the references; no X3b score")

    # phase 3: new single-model caches
    built = run_pool(new_cache_tasks(), args.workers)

    # phase 4: X3b obs + reference obs (relabelled), evaluation
    meta_rows, parts = [], []
    for s, prek, v, rv, t, rt in x3b_pairs():
        P = {"pair": pname(v, rv, t, rt), "cls": f"x3b-{s}", "primary": True}
        parts.append(rows_for(s, P, cdir(prek, v, rv), cdir(prek, t, rt), cache))
        meta_rows.append({"set": s, "pair": P["pair"], "pre": prek, "vit": v, "rel_v": rv, "text": t, "rel_t": rt,
                          "cache_a": os.path.relpath(cdir(prek, v, rv), E.PROJECT),
                          "cache_b": os.path.relpath(cdir(prek, t, rt), E.PROJECT)})
    df = pd.concat(parts, ignore_index=True)
    ref_parts = []
    for name, s, prek in (("clip", "A1ref", "clip"), ("l2", "A3ref", "l2"), ("ctrold", "A4ref", "ctr"), ("CA", "Bref", "ctr")):
        r = refs[name].copy()
        if name == "CA":
            r["text"], r["vit_"] = r["enc"], r["vit"]
            r = r.drop(columns=["vit", "enc"]).rename(columns={"vit_": "vit"})
        else:
            r["vit"] = r["pair"].str.split("__").str[0]
            r["text"] = r["pair"].str.split("__").str[1]
        rt = "n5" if name == "CA" else "0.5"
        r["pair"] = [pname(a, "0.5", b, rt) for a, b in zip(r["vit"], r["text"])]
        r["orig_config"] = r["config"]
        r["config"] = s
        r["cls"] = f"x3b-{s}"
        for (a, b) in sorted(set(zip(r["vit"], r["text"]))):
            meta_rows.append({"set": s, "pair": pname(a, "0.5", b, rt), "pre": prek, "vit": a, "rel_v": "0.5", "text": b,
                              "rel_t": rt, "cache_a": os.path.relpath(cdir(prek, a, "0.5"), E.PROJECT),
                              "cache_b": os.path.relpath(cdir(prek, b, rt), E.PROJECT)})
        ref_parts.append(r.drop(columns=["vit", "text"]))
    dref = pd.concat(ref_parts, ignore_index=True)
    os.makedirs(OBSDIR, exist_ok=True)
    hashes = {"obs/obs_X3B.parquet": E.save_parquet_new(os.path.join(OBSDIR, "obs_X3B.parquet"), df),
              "obs/obs_X3B_ref.parquet": E.save_parquet_new(os.path.join(OBSDIR, "obs_X3B_ref.parquet"), dref)}
    pairs_meta = pd.DataFrame(meta_rows)
    os.makedirs(EVALDIR, exist_ok=True)
    hashes["eval/X3B_pairs.parquet"] = E.save_parquet_new(os.path.join(EVALDIR, "X3B_pairs.parquet"), pairs_meta)
    T = evaluate(pd.concat([df, dref.drop(columns=["orig_config"])], ignore_index=True))
    for name, t in T.items():
        hashes[f"eval/X3B_{name}.parquet"] = E.save_parquet_new(os.path.join(EVALDIR, f"X3B_{name}.parquet"), t)

    # consistency of the reference evaluation with the frozen tables (reported, not a STOP rule)
    cons = {}
    Lv = T["levels"]
    for s, path in (("Bref", os.path.join(E.EVAL, "E0_levels_CA.parquet")), ("A4ref", os.path.join(E.EVAL, "E0_levels_ctrold.parquet"))):
        fro = pd.read_parquet(path)
        mine = Lv[Lv["set"] == s].copy()
        mine["orig_pair"] = [p.replace(".r05", "").replace(".n5", "") for p in mine["pair"]]
        m = mine.merge(fro, left_on=["orig_pair", "k"], right_on=["pair", "k"], suffixes=("", "_frozen"))
        cols = [c for c in ("L_m1024", "L_m8192", "L_m10000") if c in fro.columns]
        cons[s] = {"n_rows": int(len(m)), "max_abs_diff": {c: float(np.max(np.abs(m[c] - m[c + "_frozen"]))) for c in cols}}
    c11 = pd.read_parquet(os.path.join(E.S11_OUT, "S1_1_v11_cells_clip.parquet"))
    mine = T["cells"][T["cells"]["set"] == "A1ref"].copy()
    mine["orig_pair"] = [p.replace(".r05", "") for p in mine["pair"]]
    m = mine.merge(c11, left_on=["orig_pair", "m", "k"], right_on=["pair", "m", "k"], suffixes=("", "_s11"))
    cons["A1ref_vs_S1_1_cells_clip"] = {"n_rows": int(len(m)), "max_abs_diff_obs": float(np.max(np.abs(m["obs"] - m["obs_s11"])))}
    print("consistency", json.dumps(cons), flush=True)

    prov = {"role": "x3b-run", "readings": "analysis/e0_sentence/x3b/X3B_READINGS.md", **pre,
            "validation_sha256": hval, "validation_pass": val["pass"], "caches_built": built,
            "n_x3b_pairs": int(df["pair"].nunique()), "n_x3b_rows": int(len(df)), "n_ref_rows": int(len(dref)),
            "cache_files_sha256": cache.files, "outputs_sha256": hashes, "reference_consistency": cons,
            "workers": args.workers, "blas_threads": 1, "cpu_affinity": aff,
            "code_sha256": {"x3b_scores.py": E.sha256_file(os.path.abspath(__file__)),
                            **{f: E.sha256_file(os.path.join(E0DIR, f)) for f in ("e0_common.py", "e0_prep.py", "e0_obs.py", "e0_evaluate.py")},
                            **{f"s11copy/{f}": E.sha256_file(os.path.join(E.S11COPY, f)) for f in
                               ("s11v11_obs.py", "s11v11_prep.py", "s11v11_evaluate.py", "s11_common.py", "s11v11_common.py")}},
            "started_utc": utc0, "finished_utc": E.utcnow(), "wall_s": time.perf_counter() - t0,
            "cpu_core_hours": cpu_seconds() / 3600, "gpu_hours": 0.0}
    hprov = E.write_json_new(os.path.join(OBSDIR, "obs_X3B.json"), prov)
    print(json.dumps({"obs_X3B.json": hprov, **hashes, "validation": hval, "cpu_core_h": prov["cpu_core_hours"],
                      "wall_s": prov["wall_s"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
