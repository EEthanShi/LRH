"""E0 `ctr` calibration from the 29 OLD pairs under `ctr` (ANALYSIS_PLAN_E0.md v1.0 §7.5, §11 step 5).

No sentence encoder enters: reads only results/E0_sentence/obs/obs_ctrold.parquet (old pairs) and writes
results/E0_sentence/eval/ctr_calibration.json (lines a + b ln L at k in {1, 10, 100}, m_ref in {1024, 2048},
and the ViT-specific competitors) plus eval/E0_{cells,drawslopes,slopes,elasticity}_ctrold.parquet.
The run role then appends '[e0-run] CTR CALIBRATION FROZEN: ... ctr_calibration.json sha256 <hex>'.

    venv/bin/python -I analysis/e0_sentence/e0_calib.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e0_common as E  # noqa: E402
import e0_evaluate as EV  # noqa: E402
import e0_rules as R  # noqa: E402


def calibrate(obs_old: pd.DataFrame) -> tuple[dict, dict]:
    df, ds, S, c = EV.arm_tables(obs_old)
    pairs = E.old_pairs()
    assert sorted(c["pair"].unique()) == sorted(p["pair"] for p in pairs)
    out = {"plan": "ANALYSIS_PLAN_E0.md v1.0 §7.5", "pre": "ctr", "n_pairs": len(pairs), "lines": {}, "competitor": {},
           "within_class": {}, "created_utc": E.utcnow()}
    for k in (1, 10, 100):
        for mref in (1024, 2048):
            L, b = EV.level_slope(c, S, pairs, k, mref)
            X = np.column_stack([np.ones(L.size), np.log(L)])
            beta, *_ = np.linalg.lstsq(X, b, rcond=None)
            r = b - X @ beta
            out["lines"][f"k{k}|m{mref}"] = {"a": float(beta[0]), "b": float(beta[1]),
                                              "resid_sd": float(np.sqrt(r @ r / (L.size - 2))),
                                              "resid_max_abs": float(np.abs(r).max())}
        vl = [p for p in pairs if p["cls"] == "cross-modal"]
        _, bv = EV.level_slope(c, S, vl, k, 1024)
        out["competitor"][f"k{k}"] = {v: float(np.mean([bb for p, bb in zip(vl, bv) if p["a"] == v])) for v in E.VITS}
    el = EV.elasticities(c, (10, 100))
    for p in pairs:
        p.setdefault("enc", p["b"])
    out["P1_old_pairs_ctr_descriptive"] = {cl: R.p1_rule(g["sum"], g["e_m"])["label"] for cl, g in el.groupby("cls")}
    tabs = {"cells": c, "drawslopes": ds, "slopes": S,
            "elasticity": pd.concat([EV.elasticities(c, ks).assign(k_set=n) for n, ks in
                                     (("k10_100", (10, 100)), ("k1_10_100", (1, 10, 100)), ("k1_10", (1, 10)))])}
    return out, tabs


def main():
    E.require_milestones("plan", "code")
    p = os.path.join(E.OBS, f"obs_{E.OLD_ARM}.parquet")
    obs = pd.read_parquet(p)
    out, tabs = calibrate(obs)
    out["obs_ctrold_sha256"] = E.sha256_file(p)
    out["code_sha256"] = {f: E.sha256_file(os.path.join(HERE, f)) for f in ("e0_calib.py", "e0_evaluate.py", "e0_common.py")}
    for name, t in tabs.items():
        E.save_parquet_new(os.path.join(E.EVAL, f"E0_{name}_{E.OLD_ARM}.parquet"), t)
    h = E.write_json_new(E.CTR_CALIB, out)
    print("ctr_calibration.json sha256", h)
    print(f"suggested STATUS: <UTC> [e0-run] CTR CALIBRATION FROZEN: results/E0_sentence/eval/ctr_calibration.json sha256 {h} ...")


if __name__ == "__main__":
    main()
