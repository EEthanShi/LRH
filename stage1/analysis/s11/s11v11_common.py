"""S1.1 under PREREG v1.1 (descriptive mode, D4.7): shared constants, the COCO
authorization guard (v1.1 hashes) and the item design.

Role s11-run, 2026-10-08. Authority, highest first:
  prereg/LEAD_DECISIONS_v1.1.md (sha256 eaebd687...),
  hrhv2/PREDICTOR_CHOICE_v1.1.md (f83b13a9...),
  analysis/s11/ANALYSIS_PLAN_S11_v1.1.md (736ec1d4...; verbatim copy of
  analysis/s11_select/ANALYSIS_PLAN_S11_v1.1.md),
  the draft analysis/s11/ANALYSIS_PLAN_S11.md (v0.9, 62fa39c6...).

The v1.0 entry points (s11_run.py, s11_core.py, s11_queue.py, s11_evaluate.py)
are kept unchanged; they implement the v1.0 protocol and refuse to run under the
v1.1 authorization (their plan hash check fails). Nothing here reads a feature
array at import time.

Configurations (one D source per model each):
  clip      prh_clip+l2, one clamp threshold per model pooled over the 4 stored
            layers on all 16384 pool rows ('pool-stored'; D5 definition text,
            plan V6) -- PRIMARY
  l2        l2 -- secondary preprocessing (D5, plan V6)
  cliplayer prh_clip+l2, threshold over the selected layer only ('pool-layer'
            in the draft's naming) -- V6 sensitivity
  dedup     as 'clip', with the 54 surplus duplicate-text items dropped from
            the pool (lowest stored position of each of the 44 groups kept);
            same D entries as 'clip' (D5 sensitivity; primary pairs only)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import s11_common as C  # noqa: E402  (v1.0 design constants: pairs, models, draws, halves, layer rule)

PROJECT = C.PROJECT
OUT = os.path.join(PROJECT, "results", "S1_1", "v11")
CACHE = os.path.join(OUT, "cache")
UNITS = os.path.join(OUT, "units")
OBS = os.path.join(OUT, "obs")
LOGS = os.path.join(OUT, "logs")

AUTH_PATH = os.path.join(C.ANALYSIS_DIR, "LEAD_AUTHORIZATION.json")
PLAN_V11 = os.path.join(C.ANALYSIS_DIR, "ANALYSIS_PLAN_S11_v1.1.md")
PLAN_V11_SEL = os.path.join(PROJECT, "analysis", "s11_select", "ANALYSIS_PLAN_S11_v1.1.md")
LEAD_V11 = os.path.join(PROJECT, "prereg", "LEAD_DECISIONS_v1.1.md")
PREDICT_V11 = os.path.join(PROJECT, "hrhv2", "predict_v11.py")
CHOICE_V11 = os.path.join(PROJECT, "hrhv2", "PREDICTOR_CHOICE_v1.1.md")
PREDICT_V10 = os.path.join(PROJECT, "hrhv2", "predict.py")

# frozen hashes (STATUS.md 06:26:32Z, 10:17:44Z, 10:20:55Z)
SHA = {
    "lead_decisions_v1_1": ("prereg/LEAD_DECISIONS_v1.1.md",
                            "eaebd68787a7111f87bdac87a58ff565c0364bae64416dc9203eff913d2f12c3"),
    "plan_v1_1": ("analysis/s11/ANALYSIS_PLAN_S11_v1.1.md",
                  "736ec1d4c9d20a843fc895ab895babfc13871aadf4e32d1410707e98d8b5c391"),
    "plan_v1_1_select_copy": ("analysis/s11_select/ANALYSIS_PLAN_S11_v1.1.md",
                              "736ec1d4c9d20a843fc895ab895babfc13871aadf4e32d1410707e98d8b5c391"),
    "predict_v11_py": ("hrhv2/predict_v11.py",
                       "a031b2136fb651340ff0589b7ed584e553b5b42c1de35fd64d9deb53ca5cd31e"),
    "predictor_choice_v1_1": ("hrhv2/PREDICTOR_CHOICE_v1.1.md",
                              "f83b13a9e2a78f42ad0aa959f95e1b2a7adcbe763f81f99c9805e6a5ba6aa948"),
    "predict_py_v1_0": ("hrhv2/predict.py",
                        "a17964b4dc4a04673eda1a0798fa5d27afb3ea5c61ea8a9b251a2a2d5777b74c"),
    "prereg_v1_0": ("prereg/PREREG_stage1.md",
                    "bee0583c2bfd3ea941526c90c5eb879753a58f2c5b823e9b2444abb5697d5fcc"),
    "spec": ("SPEC.md", "e781e44f6b809e90af758ac8a08e78c8ef192f1b4d6927d4694a729eca15a7a3"),
}
D4_V11 = "b-v1.1: no eligible estimator (descriptive)"
UNVALIDATED = "iii-LSlog-EB"          # plan V4
PRIMARY_CLIP_MODE = "pool-stored"     # plan V6 / D5 definition text

M_GRID = C.M_GRID                     # 256 ... 16384
M_FIT = C.M_FIT                       # 1024
KS = C.KS                             # 1, 10, 100
KMAX_TAB = max(KS) + 1                # kNN table width (k = 100 needs the 101st value for ties)
VAL_COLS = sorted({k - 1 for k in KS} | {k for k in KS})   # stored D values: k-th and (k+1)-th
N_DRAWS = C.N_DRAWS

CONFIGS = {
    "clip": {"pre": "prh_clip+l2", "clip_mode": "pool-stored", "dedup": False, "dsrc": "clip-stored",
             "role": "primary"},
    "l2": {"pre": "l2", "clip_mode": None, "dedup": False, "dsrc": "l2", "role": "secondary preprocessing"},
    "cliplayer": {"pre": "prh_clip+l2", "clip_mode": "pool-layer", "dedup": False, "dsrc": "clip-layer",
                  "role": "sensitivity: clip scope (V6)"},
    "dedup": {"pre": "prh_clip+l2", "clip_mode": "pool-stored", "dedup": True, "dsrc": "clip-stored",
              "role": "sensitivity: surplus duplicates dropped (D5)"},
}
SENS_CONFIGS_PRIMARY_ONLY = ("cliplayer", "dedup")


def file_sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_hashes() -> dict:
    """sha256 of every frozen input; raises SystemExit on any mismatch."""
    got = {}
    for key, (rel, want) in SHA.items():
        h = file_sha256(os.path.join(PROJECT, rel))
        if h != want:
            raise SystemExit(f"REFUSED: {rel} sha256 {h} != frozen {want}")
        got[key] = h
    return got


def check_authorization(config: str | None = None) -> dict:
    """v1.1 COCO guard. Returns the authorization record or raises SystemExit
    BEFORE any feature file is opened."""
    if not os.path.exists(AUTH_PATH):
        raise SystemExit(f"REFUSED: {AUTH_PATH} does not exist")
    with open(AUTH_PATH) as f:
        auth = json.load(f)
    need = ("authorized_by", "utc", "prereg_version", "D4", "lead_decisions_v1_1_sha256", "plan_sha256",
            "predict_v11_py_sha256", "predictor_choice_sha256", "predict_py_sha256", "clip_mode",
            "unvalidated_predictor")
    miss = [k for k in need if k not in auth]
    if miss:
        raise SystemExit(f"REFUSED: authorization lacks {miss}")
    if auth["prereg_version"] != "v1.1" or auth["D4"] != D4_V11:
        raise SystemExit("REFUSED: authorization is not a PREREG v1.1 / D4.7 record")
    got = check_hashes()
    pairs = (("lead_decisions_v1_1_sha256", "lead_decisions_v1_1"), ("plan_sha256", "plan_v1_1"),
             ("predict_v11_py_sha256", "predict_v11_py"), ("predictor_choice_sha256", "predictor_choice_v1_1"),
             ("predict_py_sha256", "predict_py_v1_0"))
    for ak, hk in pairs:
        if auth[ak] != got[hk]:
            raise SystemExit(f"REFUSED: authorization {ak} {auth[ak]} != file {got[hk]}")
    if auth["clip_mode"] != PRIMARY_CLIP_MODE:
        raise SystemExit(f"REFUSED: authorized primary clip_mode {auth['clip_mode']} != plan V6 {PRIMARY_CLIP_MODE}")
    if auth["unvalidated_predictor"] != UNVALIDATED:
        raise SystemExit("REFUSED: unvalidated predictor differs from plan V4")
    if config is not None and config not in auth.get("configs", {}):
        raise SystemExit(f"REFUSED: configuration {config!r} not authorized")
    return auth


# ---------------------------------------------------------------------------
# Item design (ids/texts only)
# ---------------------------------------------------------------------------

def surplus_duplicates(texts) -> np.ndarray:
    """Pool rows to drop for the D5 sensitivity: in every group of identical
    texts keep the lowest stored position, drop the others (sorted)."""
    g = defaultdict(list)
    for i, t in enumerate(texts):
        g[t].append(i)
    drop = sorted(i for v in g.values() if len(v) > 1 for i in v[1:])
    return np.asarray(drop, dtype=np.int64)


def design(config: str, ids=None, texts=None) -> dict:
    """Per draw: the item order (pool rows), H1 mask in that order, the
    predicted anchors (H2 positions in the m_f prefix), the m grid."""
    if ids is None:
        ids, texts = C.coco_item_ids()
    orders = C.draw_orders(ids)
    h1 = C.h1_mask(ids)
    cfg = CONFIGS[config]
    drop = surplus_duplicates(texts) if cfg["dedup"] else np.zeros(0, dtype=np.int64)
    keep = np.ones(len(ids), dtype=bool)
    keep[drop] = False
    out = {"config": config, "n_pool": int(len(ids)), "dropped_rows": drop, "draws": []}
    for d, o in enumerate(orders):
        o = np.asarray(o, dtype=np.int64)
        o = o[keep[o]]
        M = int(o.size)
        m_grid = tuple(m if m <= M else M for m in M_GRID)
        h1o = h1[o]
        anc = np.flatnonzero(~h1o[:M_FIT])
        out["draws"].append({"draw": d, "order": o, "h1": h1o, "anchor_pos": anc, "m_grid": m_grid, "M": M})
    return out


def pairs_for(config: str, which: str = "all"):
    """Pair dicts for a configuration. Sensitivity configurations: primary pairs only."""
    P = C.pair_list()
    if config in SENS_CONFIGS_PRIMARY_ONLY or which == "primary":
        P = [p for p in P if p["primary"]]
    elif which == "secondary":
        P = [p for p in P if not p["primary"]]
    return P


def models_for(config: str):
    names = set()
    for p in pairs_for(config):
        names.add(p["a"])
        names.add(p["b"])
    return sorted(names)


def cache_dir(config: str, model: str) -> str:
    return os.path.join(CACHE, config, model)


def jsonable(v):
    import math
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else (None if math.isnan(f) else ("inf" if f > 0 else "-inf"))
    return v
