"""E0 ADDENDUM 1 (implementation fix only): run the frozen e0_evaluate with a corrected write_outputs.

Defect (found by the pre-guard dry run analysis/e0_sentence/addendum1/e0_dryrun_synthetic.py on synthetic obs):
frozen e0_calib.main writes eval/E0_{cells,drawslopes,slopes,elasticity}_ctrold.parquet before the guard (plan
§11 step 5), and frozen e0_evaluate.main later loads obs_ctrold too and its write_outputs tries to write the same
four files with E.save_parquet_new, which refuses to overwrite -> FileExistsError before VERDICTS_E0.json exists.

Fix: write_outputs_add1 = the frozen write_outputs, except that for an existing target file it requires the
existing file to be content-identical (pandas DataFrame.equals after reading) to the table the frozen code would
write, records that, and does not write it again; any difference raises. Nothing else changes: every statistic,
rule, threshold, label and table is computed by the frozen e0_evaluate / e0_rules code. main() is the frozen
e0_evaluate.main with the patch installed and an "addendum_1" record added to VERDICTS_E0.json.

    venv/bin/python -I analysis/e0_sentence/addendum1/e0_evaluate_add1.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

ROOT = os.path.expanduser("~/hrh_v2_stage1_20261008")
AN = os.path.join(ROOT, "analysis", "e0_sentence")
if AN not in sys.path:
    sys.path.insert(0, AN)
import e0_common as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REUSED = []


def _put(path, df):
    if os.path.exists(path):
        old = pd.read_parquet(path)
        new_rt = df.reset_index(drop=True)
        same = old.reset_index(drop=True).equals(new_rt)
        if not same:
            # compare through a parquet round trip of the new table (dtype normalisation of the writer)
            import io
            buf = io.BytesIO()
            new_rt.to_parquet(buf, index=False)
            buf.seek(0)
            same = old.reset_index(drop=True).equals(pd.read_parquet(buf))
        if not same:
            raise RuntimeError(f"ADDENDUM 1: existing {path} differs from the table the frozen code computes")
        REUSED.append({"file": os.path.relpath(path, E.PROJECT), "sha256": E.sha256_file(path), "content_identical": True})
        return
    E.save_parquet_new(path, df)


def write_outputs_add1(T, V, d):
    """Frozen e0_evaluate.write_outputs, with the reuse rule of ADDENDUM 1 for existing files."""
    import e0_evaluate as EV
    os.makedirs(d, exist_ok=True)
    for arm, t in T.items():
        _put(os.path.join(d, f"E0_cells_{arm}.parquet"), t["cells"])
        _put(os.path.join(d, f"E0_drawslopes_{arm}.parquet"), t["drawslopes"])
        _put(os.path.join(d, f"E0_slopes_{arm}.parquet"), t["slopes"])
        el = pd.concat([x.assign(k_set=name) for name, x in t["el"].items()])
        _put(os.path.join(d, f"E0_elasticity_{arm}.parquet"), el)
        _put(os.path.join(d, f"E0_levels_{arm}.parquet"), t["levels"])
    E.write_new(os.path.join(d, "E0_tables.md"), EV.tables_md(V, T))


def install(EV):
    EV.write_outputs = write_outputs_add1
    return EV


def addendum_record():
    p = os.path.join(AN, "ANALYSIS_PLAN_E0_ADDENDUM_1.md")
    return {"addendum": "ANALYSIS_PLAN_E0_ADDENDUM_1.md", "addendum_sha256": E.sha256_file(p) if os.path.exists(p) else None,
            "wrapper_sha256": E.sha256_file(os.path.abspath(__file__)), "reused_existing_files": REUSED,
            "effect": "implementation fix only; no statistic, rule, threshold or label changed"}


def main():
    import e0_evaluate as EV
    import e0_obs
    install(EV)
    # ---- frozen e0_evaluate.main, verbatim except the addendum record ----
    g = e0_obs.guard_ok()
    with open(E.FREEZE_CONSTANTS) as f:
        fc = json.load(f)
    s11 = {pre: EV.load_s11(pre, fc["inputs_sha256"]) for pre in ("clip", "l2")}
    with open(E.CTR_CALIB) as f:
        ctr_cal = json.load(f)
    if any(os.path.exists(os.path.join(E.OBS, f"obs_{a}.parquet")) for a in E.Q1_ARMS):
        cp = os.path.join(E.S11_OUT, "S1_1_v11_cells_cliplayer.parquet")
        sp = os.path.join(E.S11_OUT, "S1_1_v11_slopes_cliplayer.parquet")
        c, S = pd.read_parquet(cp), pd.read_parquet(sp)
        s11["cliplayer"] = (c[c["primary"]].copy(), S[S["primary"]].copy())
        s11_extra = {os.path.relpath(p, E.PROJECT): E.sha256_file(p) for p in (cp, sp)}
    else:
        s11_extra = {}
    obs = {}
    for arm in (E.OLD_ARM,) + tuple(E.ARMS) + tuple(E.Q1_ARMS):
        p = os.path.join(E.OBS, f"obs_{arm}.parquet")
        if os.path.exists(p):
            obs[arm] = pd.read_parquet(p)
    missing = [a for a in E.ARMS if a not in obs]
    if missing:
        raise SystemExit(f"obs missing for arms {missing}")
    V = EV.evaluate(obs, s11, ctr_cal, write_dir=E.EVAL)
    V["guard"] = g
    V["s11_extra_inputs_sha256"] = s11_extra
    V["obs_sha256"] = {a: E.sha256_file(os.path.join(E.OBS, f"obs_{a}.parquet")) for a in obs}
    V["code_sha256"] = {f: E.sha256_file(os.path.join(AN, f)) for f in ("e0_evaluate.py", "e0_rules.py", "e0_common.py")}
    V["addendum_1"] = addendum_record()
    h = E.write_json_new(E.VERDICTS, V)
    print("VERDICTS_E0.json", h, V["labels"])


if __name__ == "__main__":
    main()
