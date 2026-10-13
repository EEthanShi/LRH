"""Full criteria ladder (SPEC 0-4, optional SPEC 5 kNN calibration) for one
pair of feature arrays, written as JSON.

Usage (from the project directory, venv python):

    python -m hrhv2.run_pair --a A.npy --b B.npy --pre prh_clip+l2 \\
        [--pre-b l2] [--layer-a 12] [--layer-b 6] [--clip-scope all-layers] \\
        [--k-grid 1,3,10,30,100] [--item-ids ids.txt|items.parquet --id-col image_id] \\
        [--m 4096 | --rows rows.npy] [--jobs 8] \\
        [--calib-k 10 --calib-K 500 --calib-key mykey] \\
        [--no-quotients] [--no-row-iso] [--no-row-kendall] [--no-gl] \\
        --out pair.json

Inputs are .npy arrays of shape (m, d), or (m, L, d) with --layer-* selecting
one layer (negative indices allowed). Rows of A and B must be the same m
items in the same order. With pre=prh_clip+l2 and a 3-D input, the clamp
threshold is computed over all layers of each sample (PRH/ARH behaviour,
--clip-scope all-layers, default) or over the selected layer only
(--clip-scope layer). --item-ids (a text file with one id per line, or an
items.parquet with --id-col) sets the hash split H1/H2 used by the
cross-fitted quotients (default: the row indices 0..m-1). --m keeps the first
m rows (nested prefix pool); --rows (.npy or text of 0-based row indices)
selects and orders rows explicitly (e.g. another nested-draw salt, or
dropping flagged rows); ids follow the same selection. With prh_clip+l2 the
clamp threshold is computed on the selected rows only.

All metric arithmetic is float64. NaN values are written as null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np

from . import __version__, SPEC_VERSION
from . import calibrate as cal
from . import structural as st
from .ladder import ladder
from .prep import Geom, PRE_CHOICES


def jsonable(v):
    """Recursively convert numpy scalars/arrays to Python; NaN/inf -> None."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    return v


def file_sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_layer(path: str, layer: int | None):
    """Returns (X (m,d) float64, full array or None for the clip reference)."""
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:
        if layer not in (None, 0, -1):
            raise ValueError(f"{path}: 2-D array but --layer {layer} given")
        return np.asarray(arr, dtype=np.float64), None
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected (m, d) or (m, L, d), got {arr.shape}")
    if layer is None:
        raise ValueError(f"{path}: 3-D array needs a layer index")
    return np.asarray(arr[:, layer, :], dtype=np.float64), arr


def build_geom(path, layer, pre, clip_scope, rows=None):
    X, full = load_layer(path, layer)
    if rows is not None:
        X = X[rows]
        full = None if full is None else full[rows]
    clip_ref = None
    if pre == "prh_clip+l2" and full is not None and clip_scope == "all-layers":
        clip_ref = np.asarray(full, dtype=np.float64)
    G = Geom.from_features(X, pre, clip_ref=clip_ref)
    return G, {"path": os.path.abspath(path), "sha256": file_sha256(path),
               "shape": list(np.load(path, mmap_mode="r").shape), "layer": layer,
               "pre": pre, "clip_scope": clip_scope if pre == "prh_clip+l2" else None,
               "d": int(X.shape[1]), "m_used": int(X.shape[0])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--pre", required=True, choices=PRE_CHOICES)
    ap.add_argument("--pre-b", default=None, choices=PRE_CHOICES, help="default: same as --pre")
    ap.add_argument("--layer-a", type=int, default=None)
    ap.add_argument("--layer-b", type=int, default=None)
    ap.add_argument("--clip-scope", default="all-layers", choices=("all-layers", "layer"))
    ap.add_argument("--k-grid", default=None, help="comma-separated; default SPEC grid {1,3,10,30,100,k_q}")
    ap.add_argument("--comp-ks", default="1,3,10")
    ap.add_argument("--sog-k", type=int, default=10)
    ap.add_argument("--item-ids", default=None, help="text file (one id per line) or .parquet")
    ap.add_argument("--id-col", default=None, help="column of --item-ids parquet (default image_id if present, else idx)")
    ap.add_argument("--m", type=int, default=None, help="use the first m rows (nested prefix pool)")
    ap.add_argument("--rows", default=None, help=".npy or text file of 0-based row indices (overrides --m)")
    ap.add_argument("--halves-salt", default="hrhv2-halves")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-quotients", action="store_true")
    ap.add_argument("--no-row-iso", action="store_true")
    ap.add_argument("--no-row-kendall", action="store_true")
    ap.add_argument("--no-gl", action="store_true")
    ap.add_argument("--global-order-max-pairs", type=int, default=None,
                    help="NON-SPEC speed option: Bernoulli-subsample pairs above this count (flagged in output)")
    ap.add_argument("--calib-k", default=None, help="comma-separated k for SPEC-5 kNN calibration (fast path)")
    ap.add_argument("--calib-K", type=int, default=500)
    ap.add_argument("--calib-alpha", type=float, default=0.05)
    ap.add_argument("--calib-key", default=None, help="sha256 seed key (default derived from input hashes)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    pre_b = args.pre_b or args.pre
    rows = None
    if args.rows:
        rows = (np.load(args.rows) if args.rows.endswith(".npy") else np.loadtxt(args.rows, dtype=np.int64)).astype(np.int64)
    elif args.m:
        rows = np.arange(args.m, dtype=np.int64)
    A, meta_a = build_geom(args.a, args.layer_a, args.pre, args.clip_scope, rows)
    B, meta_b = build_geom(args.b, args.layer_b, pre_b, args.clip_scope, rows)
    if A.m != B.m:
        raise SystemExit(f"row mismatch: A has {A.m} items, B has {B.m}")
    item_ids = None
    if args.item_ids:
        if args.item_ids.endswith(".parquet"):
            import pandas as pd
            tab = pd.read_parquet(args.item_ids)
            col = args.id_col or ("image_id" if "image_id" in tab.columns else "idx")
            item_ids = [str(v) for v in tab[col].tolist()]
        else:
            with open(args.item_ids) as f:
                item_ids = [ln.strip() for ln in f if ln.strip()]
        if rows is not None:
            item_ids = [item_ids[i] for i in rows]
        if len(item_ids) != A.m:
            raise SystemExit(f"--item-ids has {len(item_ids)} ids, expected {A.m}")
    k_grid = [int(k) for k in args.k_grid.split(",")] if args.k_grid else None
    comp_ks = tuple(int(k) for k in args.comp_ks.split(","))
    t_prep = time.perf_counter() - t0

    row, aux = ladder(A, B, k_grid=k_grid, comp_ks=comp_ks, sog_k=args.sog_k,
                      halves_salt=args.halves_salt, item_ids=item_ids, n_jobs=args.jobs,
                      with_quotients=not args.no_quotients, with_row_iso=not args.no_row_iso,
                      with_row_kendall=not args.no_row_kendall, with_gl=not args.no_gl,
                      global_order_max_pairs=args.global_order_max_pairs, return_aux=True)
    calib = {}
    if args.calib_k:
        key = args.calib_key or f"hrhv2|calib|{meta_a['sha256'][:16]}|{meta_b['sha256'][:16]}"
        for k in (int(x) for x in args.calib_k.split(",")):
            if aux["nA"].shape[1] >= k:
                nA, nB = aux["nA"], aux["nB"]
            else:
                nA, _ = st.knn_table(A.D, k)
                nB, _ = st.knn_table(B.D, k)
            c = cal.calibrate_knn(nA, nB, k, key=f"{key}|k={k}", K=args.calib_K, alpha=args.calib_alpha)
            c["near_tie_A"] = row.get(f"neartie_A_k{k}")
            c["near_tie_B"] = row.get(f"neartie_B_k{k}")
            calib[f"k{k}"] = c
    out = {
        "hrhv2_version": __version__, "spec_version": SPEC_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "python": sys.version.split()[0], "numpy": np.__version__,
        "inputs": {"A": meta_a, "B": meta_b, "item_ids": os.path.abspath(args.item_ids) if args.item_ids else None,
                   "rows": None if rows is None else {"n": int(rows.size), "sha256": hashlib.sha256(rows.tobytes()).hexdigest()}},
        "args": vars(args),
        "ladder": row,
        "calibration_knn": calib,
        "time_prep_s": round(t_prep, 4),
        "time_wall_s": round(time.perf_counter() - t0, 4),
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(jsonable(out), f, indent=1, sort_keys=False, allow_nan=False)
    os.replace(tmp, args.out)
    return out


if __name__ == "__main__":
    main()
