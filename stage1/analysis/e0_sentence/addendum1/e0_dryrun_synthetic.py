"""E0 pre-guard DRY RUN of the frozen evaluation path on SYNTHETIC new-pair obs (role e0-run; not frozen code).

No sentence-encoder file is read. The "new pairs" are the 15 S1.1 vision-language pairs of obs_clip.parquet with
their language model relabelled as an encoder (bloomz_1b7 -> mpnet, openllama_3b -> contriever,
pythia410m_s1 -> qwen3emb06) and obs scaled by a fixed factor per arm; 'ctrold' is S1.1's own obs_clip (primary).
Purpose: exercise e0_calib.calibrate + e0_evaluate.evaluate(write_dir=...) with ALL arms in one output directory,
exactly as the frozen main() sequence does (calib writes E0_*_ctrold.parquet into eval/ first), to find
implementation defects before the guard. Outputs go only to analysis/e0_sentence/addendum1/dryrun_<utc>/.

    venv/bin/python -I analysis/e0_sentence/addendum1/e0_dryrun_synthetic.py [--patched]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/hrh_v2_stage1_20261008")
AN = os.path.join(ROOT, "analysis", "e0_sentence")
sys.path.insert(0, AN)
import e0_calib as CAL  # noqa: E402
import e0_common as E  # noqa: E402
import e0_evaluate as EV  # noqa: E402

RELABEL = {"bloomz_1b7": "mpnet", "openllama_3b": "contriever", "pythia410m_s1": "qwen3emb06"}
FACTOR = {"P": 1.0, "D1": 0.9, "D2": 1.1, "D3": 1.2, "D4": 1.3, "A": 1.4, "Aall": 1.45, "C": 1.05, "CA": 1.5,
          "CAall": 1.55, "L": 0.95, "Pdd": 1.0}


def synth(obs_old: pd.DataFrame, arm: str) -> pd.DataFrame:
    vl = obs_old[obs_old["cls"] == "cross-modal"].copy()
    a = vl["pair"].str.split("__").str[0]
    b = vl["pair"].str.split("__").str[1].map(RELABEL)
    vl["pair"] = a + "__" + b
    vl["cls"] = E.NEW_CLASS
    vl["config"] = arm
    vl["vit"] = a.to_numpy()
    vl["enc"] = b.to_numpy()
    f = FACTOR[arm] * (1.0 + 0.02 * vl["m"].map(lambda m: np.log2(m / 256)))   # make slopes vary with the arm
    for c in ("obs", "obs_allH2", "obs_clean"):
        vl[c] = np.minimum(vl[c] * f, 1.0)
    return vl.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patched", action="store_true", help="use the addendum-1 write_outputs")
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = os.path.join(AN, "addendum1", f"dryrun_{'patched' if args.patched else 'frozen'}_{stamp}")
    os.makedirs(out)
    obs = pd.read_parquet(os.path.join(E.S11_OUT, "obs", "obs_clip.parquet"))
    old = obs[obs["primary"]].reset_index(drop=True)
    cal, tabs = CAL.calibrate(old)
    for name, t in tabs.items():                       # as e0_calib.main does, into the same eval dir
        E.save_parquet_new(os.path.join(out, f"E0_{name}_{E.OLD_ARM}.parquet"), t)
    obs_by_arm = {E.OLD_ARM: old}
    for arm in E.ARMS:
        obs_by_arm[arm] = synth(old, arm)
    with open(E.FREEZE_CONSTANTS) as f:
        fc = json.load(f)
    s11 = {pre: EV.load_s11(pre, fc["inputs_sha256"]) for pre in ("clip", "l2")}
    if args.patched:
        sys.path.insert(0, os.path.join(AN, "addendum1"))
        import e0_evaluate_add1 as A1
        A1.install(EV)
    res = {"utc": stamp, "patched": args.patched}
    try:
        V = EV.evaluate(obs_by_arm, s11, cal, write_dir=out)
        res["ok"] = True
        res["labels"] = V["labels"]
        res["secondary_labels"] = {n: {c: r.get("label", r.get("label_if_rule_applied")) for c, r in d.items()}
                                   for n, d in V["secondary"].items()}
    except Exception:
        res["ok"] = False
        res["traceback"] = traceback.format_exc()
    res["files"] = sorted(os.listdir(out))
    with open(os.path.join(out, "dryrun_result.json"), "w") as f:
        json.dump(E.jsonable(res), f, indent=1)
    print(json.dumps(E.jsonable({k: v for k, v in res.items() if k != "files"}), indent=1))
    print("n files", len(res["files"]))


if __name__ == "__main__":
    main()
