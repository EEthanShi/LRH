"""E0 shared constants, paths, inputs and single-model geometry.

Implements analysis/e0_sentence/ANALYSIS_PLAN_E0.md v1.0 (FROZEN, sha256 in STATUS line
"[e0-freeze] PLAN FROZEN"). S1.1 code is used only through the byte-identical copies in
analysis/e0_sentence/s11copy/ (plan V-a0). Nothing here reads a feature array at import time.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S11COPY = os.path.join(HERE, "s11copy")
if S11COPY not in sys.path:
    sys.path.insert(0, S11COPY)
import s11_common as C  # noqa: E402  (byte-identical copy; inserts PROJECT for hrhv2)
import s11v11_common as V  # noqa: E402
from hrhv2 import prep  # noqa: E402

PROJECT = C.PROJECT
STATUS = os.path.join(PROJECT, "STATUS.md")
ANALYSIS = os.path.join(PROJECT, "analysis", "e0_sentence")
PLAN = os.path.join(ANALYSIS, "ANALYSIS_PLAN_E0.md")
MANIFEST = os.path.join(ANALYSIS, "CODE_MANIFEST_E0.json")
FREEZE_CONSTANTS = os.path.join(ANALYSIS, "freeze", "freeze_constants.json")
FEAT = os.path.join(PROJECT, "features", "coco_e0")
OUT = os.path.join(PROJECT, "results", "E0_sentence")
CACHE = os.path.join(OUT, "cache")
OBS = os.path.join(OUT, "obs")
EVAL = os.path.join(OUT, "eval")
GUARD = os.path.join(OUT, "guard")
DIAG = os.path.join(OUT, "diag")
VALID = os.path.join(OUT, "validation")
EXTRACTION = os.path.join(OUT, "extraction")
EXTRACTION_MANIFEST = os.path.join(EXTRACTION, "EXTRACTION_E0.json")
CTR_CALIB = os.path.join(EVAL, "ctr_calibration.json")
VERDICTS = os.path.join(OUT, "VERDICTS_E0.json")
S11_OUT = V.OUT                                   # results/S1_1/v11
S11_CACHE = V.CACHE

# ---------------------------------------------------------------------------------------------- panel (plan §2)
VITS = tuple(C.VISION)                            # dinov2_b, clip_b, mae_b, augreg_b, siglip_b
ENCODERS = ("mpnet", "contriever", "qwen3emb06")
OLD_TEXT = ("bloomz_1b7", "openllama_3b", "pythia410m_s1")
TEXT_ORDER_P2 = OLD_TEXT + ("contriever", "mpnet", "qwen3emb06")   # plan §7.2 fixed text order
Q1_LMS = OLD_TEXT                                  # conditional arm Q1 (plan §15)
INPUTS = ("c1", "n1", "n2", "n3", "n4", "n5", "all")
N_OF_INPUT = {"n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "all": None}
NEW_CLASS = "cross-modal-sentence"
OLD_CTR_MODELS = VITS + ("bloomz_1b7", "openllama_3b") + tuple(f"pythia410m_s{s}" for s in range(1, 9))

# arm -> (cache pre key, text input, design); plan §2.5
ARMS = {
    "P": ("clip", "c1", "clip"),
    "D1": ("clip", "n1", "clip"),
    "D2": ("clip", "n2", "clip"),
    "D3": ("clip", "n3", "clip"),
    "D4": ("clip", "n4", "clip"),
    "A": ("clip", "n5", "clip"),
    "Aall": ("clip", "all", "clip"),
    "C": ("ctr", "c1", "clip"),
    "CA": ("ctr", "n5", "clip"),
    "CAall": ("ctr", "all", "clip"),
    "L": ("l2", "c1", "clip"),
    "Pdd": ("clip-dedup", "c1", "dedup"),
}
P2C_ARMS = ("D1", "D2", "D3", "D4", "A")           # n = 1..5
Q1_ARMS = {"LA": ("cliplayer", "n5", "clip"), "LCA": ("ctr", "n5", "clip")}
OLD_ARM = "ctrold"                                 # 29 old pairs under ctr (pre-guard)
NEW_ARMS = tuple(ARMS) + tuple(Q1_ARMS)

# the 51 new caches of plan §5: (pre key, model key)
def new_cache_list():
    out = []
    for e in ENCODERS:
        for i in INPUTS:
            out.append(("clip", f"{e}.{i}"))
    for e in ENCODERS:
        for i in ("c1", "n5", "all"):
            out.append(("ctr", f"{e}.{i}"))
    for m in OLD_CTR_MODELS:
        out.append(("ctr", m))
    for e in ENCODERS:
        out.append(("l2", f"{e}.c1"))
    for e in ENCODERS:
        out.append(("clip-dedup", f"{e}.c1"))
    assert len(out) == 51
    return out


def q1_cache_list():
    return [("cliplayer", f"{lm}.n5") for lm in Q1_LMS] + [("ctr", f"{lm}.n5") for lm in Q1_LMS]


# ---------------------------------------------------------------------------------------------- small helpers
def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


jsonable = V.jsonable


def write_new(path: str, data: bytes | str) -> None:
    """Atomic write that refuses to overwrite (plan: E0 code never overwrites)."""
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        raise FileExistsError(f"stale temporary file {tmp}; inspect it (E0 code deletes nothing)")
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(tmp, mode) as f:
        f.write(data)
    os.replace(tmp, path)


def write_json_new(path: str, obj) -> str:
    write_new(path, json.dumps(jsonable(obj), indent=1))
    return sha256_file(path)


def save_npy_new(path: str, arr) -> None:
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def save_parquet_new(path: str, df) -> str:
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return sha256_file(path)


def set_threads_env(threads: int) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(threads)


def check_cpu_budget(workers: int, threads: int) -> None:
    """Task rule: at most 6 CPU threads in total (plan §5, §12)."""
    if workers < 1 or threads < 1 or workers * threads > 6:
        raise SystemExit(f"REFUSED: workers x threads = {workers} x {threads} exceeds the 6-thread task rule")


# ---------------------------------------------------------------------------------------------- STATUS
HEX64 = r"([0-9a-f]{64})"
MILESTONES = (
    ("plan", r"\[e0-freeze\] PLAN FROZEN", r"ANALYSIS_PLAN_E0\.md sha256 " + HEX64, PLAN),
    ("code", r"\[e0-freeze\] E0 CODE FROZEN", r"CODE_MANIFEST_E0\.json sha256 " + HEX64, MANIFEST),
    ("extract", r"\[e0-extract\] EXTRACTION DONE", r"EXTRACTION_E0\.json sha256 " + HEX64, EXTRACTION_MANIFEST),
    ("ctrcal", r"\[e0-run\] CTR CALIBRATION FROZEN", r"ctr_calibration\.json sha256 " + HEX64, CTR_CALIB),
)


def status_lines(path: str = STATUS) -> list:
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def find_status(tag_re: str, hash_re: str | None = None, path: str = STATUS):
    """First STATUS line matching tag_re: (line index, utc, hash or None); None if absent."""
    for i, ln in enumerate(status_lines(path)):
        if re.search(tag_re, ln):
            h = None
            if hash_re:
                m = re.search(hash_re, ln)
                h = m.group(1) if m else None
            return i, ln[:20], h
    return None


def milestone(name: str, path: str = STATUS):
    for n, tag, hre, fpath in MILESTONES:
        if n == name:
            r = find_status(tag, hre, path)
            if r is None:
                return None
            i, utc, h = r
            return {"line": i, "utc": utc, "sha256": h, "file": fpath,
                    "file_ok": bool(h) and os.path.exists(fpath) and sha256_file(fpath) == h}
    raise KeyError(name)


def require_milestones(*names, path: str = STATUS) -> dict:
    """SystemExit unless every named milestone is in STATUS, in plan order, with matching file sha256."""
    got = {}
    last = -1
    for n in names:
        m = milestone(n, path)
        if m is None:
            raise SystemExit(f"REFUSED: STATUS lacks milestone {n!r}")
        if not m["file_ok"]:
            raise SystemExit(f"REFUSED: milestone {n!r}: file {m['file']} does not match STATUS sha256 {m['sha256']}")
        if m["line"] <= last:
            raise SystemExit(f"REFUSED: milestone {n!r} is out of plan order in STATUS")
        last = m["line"]
        got[n] = m
    return got


def approved(token: str, path: str = STATUS) -> bool:
    """Conditional items Q1 / B7 (plan §15): a STATUS line containing e.g. 'E0 Q1 APPROVED'."""
    return find_status(re.escape(token), None, path) is not None


# ---------------------------------------------------------------------------------------------- design (plan §3.1)
def design(kind: str = "clip"):
    """Byte-identical S1.1 design: kind 'clip' (all rows) or 'dedup' (surplus duplicate C1 texts dropped)."""
    return V.design(kind)


def old_pairs():
    return [p for p in C.pair_list() if p["primary"]]


def new_pairs(arm: str):
    pre, inp, _ = {**ARMS, **Q1_ARMS}[arm]
    texts = Q1_LMS if arm in Q1_ARMS else ENCODERS
    out = []
    for v in VITS:
        for t in texts:
            out.append({"pair": f"{v}__{t}", "a": v, "b": f"{t}.{inp}", "vit": v, "enc": t,
                        "cls": NEW_CLASS if arm not in Q1_ARMS else "cross-modal",
                        "primary": True})
    return out


# ---------------------------------------------------------------------------------------------- caches
def cache_dir(pre_key: str, model: str, root: str = CACHE) -> str:
    """S1.1 caches for old models under clip / l2 / dedup / cliplayer (read only); E0 caches otherwise."""
    if model in C.MODELS and pre_key in ("clip", "l2", "cliplayer"):
        return os.path.join(S11_CACHE, pre_key, model)
    if model in C.MODELS and pre_key == "clip-dedup":
        return os.path.join(S11_CACHE, "dedup", model)
    return os.path.join(root, pre_key, model)


def design_of_pre(pre_key: str) -> str:
    return "dedup" if pre_key == "clip-dedup" else "clip"


# ---------------------------------------------------------------------------------------------- inputs (plan §2.3)
def l2_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = np.sqrt(np.einsum("ij,ij->i", X, X))
    if np.any(n == 0) or not np.all(np.isfinite(n)):
        raise ValueError("a row has norm 0 or a non-finite norm")
    return X / n[:, None]


def caption_average(caps: np.ndarray, index, n: int | None, n_pool: int = 16384):
    """Plan §2.3: rows of ``caps`` (float32 card embeddings) -> float64; each row / its l2 norm; arithmetic
    mean over the image's captions with rank < n (all if n is None), summed in ascending rank (= annotation id)
    order; / its l2 norm. ``index``: DataFrame with columns cap_row, idx (pool row), rank (0-based by ann id).
    Returns (X (n_pool, d) float64, counts (n_pool,) int)."""
    idx = index["idx"].to_numpy(np.int64)
    rank = index["rank"].to_numpy(np.int64)
    rows = index["cap_row"].to_numpy(np.int64)
    if np.any(idx < 0) or np.any(idx >= n_pool):
        raise ValueError("caption index refers to a row outside the pool")
    Z = l2_rows(np.asarray(caps, dtype=np.float32)[rows].astype(np.float64))
    sel = np.ones(idx.size, bool) if n is None else rank < n
    acc = np.zeros((n_pool, Z.shape[1]), np.float64)
    cnt = np.zeros(n_pool, np.int64)
    for r in range(int(rank[sel].max()) + 1 if sel.any() else 0):
        s = sel & (rank == r)
        ii = idx[s]
        if np.unique(ii).size != ii.size:
            raise ValueError("duplicate (idx, rank) in the caption index")
        acc[ii] += Z[s]
        cnt[ii] += 1
    if np.any(cnt == 0):
        raise ValueError(f"{int((cnt == 0).sum())} pool rows have no caption")
    X = acc / cnt[:, None]
    return l2_rows(X), cnt


def ctr_features(X: np.ndarray) -> np.ndarray:
    """Plan §2.4 `ctr`: l2-normalize rows, subtract the float64 column mean over all rows, l2-normalize."""
    Xn = l2_rows(X)
    mu = Xn.mean(axis=0)
    return l2_rows(Xn - mu[None, :])


def raw_old_features(model: str, check_sha: bool = True):
    """Axis of relative depth 0.5 of an S1.1 model (ViT or LM), float64, with the sidecar sha256 check."""
    slug = C.MODELS[model]
    meta = C.sidecar(slug)
    npy = os.path.join(C.COCO_DIR, slug + ".npy")
    if check_sha and sha256_file(npy) != meta["files_sha256"]["primary"]["sha256"]:
        raise RuntimeError(f"{npy} sha256 mismatch with sidecar")
    ax = C.layer_axis(meta, C.REL_DEPTH)
    arr = np.load(npy, mmap_mode="r")
    return np.asarray(arr[:, ax, :], dtype=np.float64), {"slug": slug, "axis": ax, "npy": npy}


def text_paths(base: str, feat_dir: str = FEAT):
    if base in ENCODERS:
        return {"c1": os.path.join(feat_dir, f"{base}.c1.npy"), "caps": os.path.join(feat_dir, f"{base}.caps.npy"),
                "index": os.path.join(feat_dir, "captions_index.parquet")}
    if base in Q1_LMS:
        slug = C.MODELS[base]
        return {"caps": os.path.join(feat_dir, "lm", f"{slug}.caps.npy"),
                "index": os.path.join(feat_dir, "captions_index.parquet")}
    raise KeyError(base)


def text_matrix(model_key: str, feat_dir: str = FEAT, expected_sha: dict | None = None):
    """Text-side input matrix (16384, d) float64 for '<base>.<input>' (plan §2.3)."""
    import pandas as pd
    base, inp = model_key.split(".")
    P = text_paths(base, feat_dir)
    used = {}
    for k, p in P.items():
        if inp == "c1" and k != "c1":
            continue
        if inp != "c1" and k == "c1":
            continue
        h = sha256_file(p)
        used[os.path.relpath(p, PROJECT) if p.startswith(PROJECT) else p] = h
        if expected_sha is not None:
            want = expected_sha.get(os.path.relpath(p, PROJECT))
            if want != h:
                raise RuntimeError(f"{p}: sha256 {h} differs from the extraction manifest ({want})")
    if inp == "c1":
        X = np.load(P["c1"]).astype(np.float64)
        return X, {"input": "c1", "files": used}
    index = pd.read_parquet(P["index"])
    caps = np.load(P["caps"], mmap_mode="r")
    X, cnt = caption_average(caps, index, N_OF_INPUT[inp])
    return X, {"input": inp, "files": used, "captions_per_row": {int(c): int((cnt == c).sum()) for c in np.unique(cnt)}}


def model_D(pre_key: str, model: str, feat_dir: str = FEAT, expected_sha: dict | None = None):
    """16384 x 16384 float64 D of one model under one preprocessing (plan §2.4); returns (D, info)."""
    if model in C.MODELS:
        slug = C.MODELS[model]
        npy = os.path.join(C.COCO_DIR, slug + ".npy")
        npy_sha = sha256_file(npy)
        if npy_sha != C.sidecar(slug)["files_sha256"]["primary"]["sha256"]:
            raise RuntimeError(f"{npy} sha256 mismatch with sidecar")
        if pre_key in ("clip", "clip-dedup"):
            D, info = C.model_D(model, "prh_clip+l2", "pool-stored")
        elif pre_key == "cliplayer":
            D, info = C.model_D(model, "prh_clip+l2", "pool-layer")
        elif pre_key == "l2":
            D, info = C.model_D(model, "l2", None)
        elif pre_key == "ctr":
            X, info = raw_old_features(model, check_sha=False)
            info.update({"pre": "ctr", "d": int(X.shape[1])})
            D = C.features_to_D(ctr_features(X), "none")
        else:
            raise KeyError(pre_key)
        info["npy_sha256_checked"] = npy_sha
        return D, info
    X, info = text_matrix(model, feat_dir, expected_sha)
    info.update({"model": model, "d": int(X.shape[1]), "pre": pre_key})
    if pre_key in ("clip", "clip-dedup", "cliplayer"):
        thr = prep.prh_clip_threshold(X)
        info["clip_threshold"] = thr
        info["clip_scope"] = "pool rows of this single matrix (pool-stored = pool-layer)"
        return C.features_to_D(X, "prh_clip+l2", clip_ref=X), info
    if pre_key == "l2":
        return C.features_to_D(X, "l2"), info
    if pre_key == "ctr":
        return C.features_to_D(ctr_features(X), "none"), info
    raise KeyError(pre_key)


def extraction_hashes() -> dict:
    """Feature-file sha256 recorded in EXTRACTION_E0.json (read only after the EXTRACTION DONE milestone),
    plus the conditional Q1 manifest LMAVG_E0.json when it exists (plan §15)."""
    with open(EXTRACTION_MANIFEST) as f:
        out = dict(json.load(f)["files_sha256"])
    q1 = os.path.join(EXTRACTION, "LMAVG_E0.json")
    if os.path.exists(q1):
        with open(q1) as f:
            out.update(json.load(f)["files_sha256"])
    return out
