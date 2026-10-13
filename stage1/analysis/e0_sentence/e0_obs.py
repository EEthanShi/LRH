"""E0 observed s_N(m, k) per (arm, pair, draw) from the single-model caches (ANALYSIS_PLAN_E0.md v1.0 §6).

The cell computation is the byte-identical S1.1 function s11copy/s11v11_obs.pair_rows.

    venv/bin/python -I analysis/e0_sentence/e0_obs.py --arms ctrold          # step 5: 29 old pairs under ctr
    venv/bin/python -I analysis/e0_sentence/e0_obs.py --arms P,D1,D2,...     # step 6: after GUARD CLEAN only

New-pair arms (any arm with a sentence encoder or a Q1 caption average) are refused unless STATUS has the
lines PLAN FROZEN, E0 CODE FROZEN, EXTRACTION DONE, CTR CALIBRATION FROZEN and GUARD CLEAN, in this order,
with the files' sha256 (plan §11 step 6). Writes results/E0_sentence/obs/obs_<arm>.parquet (+ .json).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e0_common as E  # noqa: E402
import s11v11_obs as SO  # noqa: E402  (byte-identical copy)


def load_cache(cdir: str, des):
    if not os.path.exists(os.path.join(cdir, "DONE")):
        raise SystemExit(f"cache missing: {cdir}")
    with open(os.path.join(cdir, "meta.json")) as f:
        meta = json.load(f)
    tabs = {dd["draw"]: dict(np.load(os.path.join(cdir, f"d{dd['draw']}_knn.npz"))) for dd in des["draws"]}
    meds = {int(d): {int(m): float(v) for m, v in dm["median_D"].items()} for d, dm in meta["draws"].items()}
    files = {os.path.relpath(os.path.join(cdir, f"d{dd['draw']}_knn.npz"), E.PROJECT):
             E.sha256_file(os.path.join(cdir, f"d{dd['draw']}_knn.npz")) for dd in des["draws"]}
    files[os.path.relpath(os.path.join(cdir, "meta.json"), E.PROJECT)] = E.sha256_file(os.path.join(cdir, "meta.json"))
    return tabs, meds, files


def arm_spec(arm: str):
    """(pairs, pre key for A, pre key for B, design kind)."""
    if arm == E.OLD_ARM:
        return E.old_pairs(), "ctr", "ctr", "clip"
    if arm == "s11clip":                      # V-c: S1.1 caches, primary pairs (validation only)
        return E.old_pairs(), "clip", "clip", "clip"
    pre, inp, des = {**E.ARMS, **E.Q1_ARMS}[arm]
    return E.new_pairs(arm), pre, pre, des


def obs_rows(arm: str, cache_root: str = E.CACHE, design_fn=None):
    pairs, pre_a, pre_b, des_kind = arm_spec(arm)
    des = (design_fn or E.design)(des_kind)
    caches, meds, files = {}, {}, {}
    for P in pairs:
        for side, mdl, pk in (("a", P["a"], pre_a), ("b", P["b"], pre_b)):
            key = (pk, mdl)
            if key not in caches:
                cdir = E.cache_dir(pk, mdl, cache_root)
                caches[key], meds[key], f = load_cache(cdir, des)
                files.update(f)
    rows = []
    for P in pairs:
        ka, kb = (pre_a, P["a"]), (pre_b, P["b"])
        rows += SO.pair_rows(arm, P, des, caches[ka], caches[kb], meds[ka], meds[kb])
    df = pd.DataFrame(rows)
    for P in pairs:
        for col in ("vit", "enc"):
            if col in P:
                df.loc[df["pair"] == P["pair"], col] = P[col]
    return df, files


def guard_ok() -> dict:
    """Plan §11 step 6: milestones 2-5 in order with matching sha256, then a GUARD CLEAN line whose file is clean."""
    got = E.require_milestones("plan", "code", "extract", "ctrcal")
    r = E.find_status(r"\[e0-run\] GUARD CLEAN", r"(GUARD_CHECK_[0-9TZ:\-]+\.json) sha256 " + E.HEX64)
    if r is None:
        raise SystemExit("REFUSED: STATUS has no '[e0-run] GUARD CLEAN' line")
    i, utc, _ = r
    ln = E.status_lines()[i]
    m = re.search(r"(GUARD_CHECK_[0-9TZ:\-]+\.json) sha256 " + E.HEX64, ln)
    if not m or i <= got["ctrcal"]["line"]:
        raise SystemExit("REFUSED: GUARD CLEAN line malformed or before CTR CALIBRATION FROZEN")
    gpath = os.path.join(E.GUARD, m.group(1))
    if not os.path.exists(gpath) or E.sha256_file(gpath) != m.group(2):
        raise SystemExit("REFUSED: guard file missing or hash differs from STATUS")
    with open(gpath) as f:
        g = json.load(f)
    if g.get("result") != "CLEAN":
        raise SystemExit("REFUSED: guard file does not say CLEAN")
    return {"guard_file": gpath, "guard_sha256": m.group(2), "guard_status_utc": utc}


def run_arm(arm: str, out_dir: str = E.OBS, cache_root: str = E.CACHE) -> dict:
    t0 = time.perf_counter()
    path = os.path.join(out_dir, f"obs_{arm}.parquet")
    if os.path.exists(path):
        raise FileExistsError(path)
    df, files = obs_rows(arm, cache_root)
    os.makedirs(out_dir, exist_ok=True)
    h = E.save_parquet_new(path, df)
    prov = {"arm": arm, "n_rows": len(df), "n_pairs": int(df["pair"].nunique()), "parquet_sha256": h,
            "cache_files_sha256": files, "seconds": time.perf_counter() - t0, "created_utc": E.utcnow(),
            "code_sha256": {f: E.sha256_file(os.path.join(HERE, f)) for f in ("e0_obs.py", "e0_common.py")},
            "s11copy_sha256": {f: E.sha256_file(os.path.join(E.S11COPY, f)) for f in ("s11v11_obs.py", "s11v11_common.py")}}
    E.write_json_new(path[:-8] + ".json", prov)
    return prov


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    args = ap.parse_args(argv)
    arms = args.arms.split(",")
    unknown = [a for a in arms if a != E.OLD_ARM and a not in E.NEW_ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}")
    if E.OLD_ARM in arms:
        E.require_milestones("plan", "code")
    new = [a for a in arms if a != E.OLD_ARM]
    g = None
    if new:
        g = guard_ok()
        if any(a in E.Q1_ARMS for a in new) and not E.approved("E0 Q1 APPROVED"):
            raise SystemExit("REFUSED: Q1 arms need 'E0 Q1 APPROVED' in STATUS (plan §15)")
    for a in arms:
        prov = run_arm(a)
        print(a, prov["n_rows"], "rows", f"{prov['seconds']:.0f}s", prov["parquet_sha256"], flush=True)
    if g:
        print("guard:", g)


if __name__ == "__main__":
    main()
