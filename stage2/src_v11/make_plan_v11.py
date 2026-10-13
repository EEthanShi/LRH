"""S2.2 v1.1 run plans (PREREG_S2_2_v1.1.md section 3, table of arms). Deterministic.

    python src_v11/make_plan_v11.py cohort        # pilot + main plans; needs plans_v11/ENG_GO + eng_choice.json
    python src_v11/make_plan_v11.py unittest      # small EXCLUDED plans for the pipeline dry run (fixed UT choice)

The "flow" treatment, K_min, 4 K_min and 16 K_min are resolved from plans_v11/eng_choice.json (written by
eng_v11.py with the frozen selection rule); the plans are a deterministic function of that file.
Order is rep-major so a partially completed queue leaves every cell with (almost) the same number of runs.
Run ids: <exp>_<arm>_N<N>_r<rep:03d>; keys hrhv2-stage2|<exp>|arm=<arm>|N=<N>|rep=<r>|<purpose>.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
PLANS = os.path.join(ROOT, "plans_v11")
NS = (25000, 250000)

# arm -> (cfg template, role); "FLOW_CFG" / "EULER_K*" are resolved from the engineering choice
ARMS = {
    "MOM": dict(head_opt="mom", init="def", head_wd=False),
    "WD": dict(head_opt="mom", init="def", head_wd=True),
    "FLOW": dict(head_opt="FLOW_CFG", init="def", head_wd=False),
    "BAL-FLOW": dict(head_opt="FLOW_CFG", init="bal", head_wd=False),
    "SMALL-FLOW": dict(head_opt="FLOW_CFG", init="small", head_wd=False),
    "BAL-MOM": dict(head_opt="mom", init="bal", head_wd=False),
    "SMALL-MOM": dict(head_opt="mom", init="small", head_wd=False),
    "SGD-K": dict(head_opt="euler", K="EULER_K1", init="def", head_wd=False),
    "MOM-BF16": dict(head_opt="mom", init="def", head_wd=False, head_bf16=True),
    "KS-4": dict(head_opt="euler", K="EULER_K4", init="def", head_wd=False),
    "KS-16": dict(head_opt="euler", K="EULER_K16", init="def", head_wd=False),
}
CORE = ("MOM", "WD", "FLOW", "BAL-FLOW", "SMALL-FLOW")
SECONDARY = ("BAL-MOM", "SMALL-MOM", "SGD-K")
FLOW_ARMS = ("FLOW", "BAL-FLOW", "SMALL-FLOW")
MOM_ARMS = ("MOM", "WD", "BAL-MOM", "SMALL-MOM", "MOM-BF16")

# runs per (arm, N): main and pilot (author decisions 2026-10-09: core pilots 96 pairs, FLOW pilots 64 pairs,
# SMALL-FLOW core)
MAIN_REPS = {a: {N: 192 for N in NS} for a in CORE}
MAIN_REPS.update({a: {N: 64 for N in NS} for a in SECONDARY})
MAIN_REPS["MOM-BF16"] = {250000: 32}
MAIN_REPS["KS-4"] = {250000: 16}
MAIN_REPS["KS-16"] = {250000: 16}
PILOT_REPS = {a: {N: 192 for N in NS} for a in ("MOM", "WD", "BAL-FLOW", "SMALL-FLOW")}
PILOT_REPS["FLOW"] = {N: 128 for N in NS}
PILOT_REPS.update({a: {N: 32 for N in NS} for a in SECONDARY})

UT_CHOICE = {"status": "GO", "K_min": 1, "flow": {"head_opt": "heun", "K": 8}, "label": "UNIT TEST ONLY (not engineering)"}
UT_MAIN_REPS = {a: {N: 4 for N in v} for a, v in MAIN_REPS.items()}
UT_PILOT_REPS = {a: {N: 4 for N in v} for a, v in PILOT_REPS.items()}


def resolve(arm: str, choice: dict) -> dict:
    cfg = dict(ARMS[arm])
    if cfg.get("head_opt") == "FLOW_CFG":
        cfg.update(choice["flow"])
    k = cfg.get("K")
    if isinstance(k, str) and k.startswith("EULER_K"):
        cfg["K"] = int(choice["K_min"]) * int(k[len("EULER_K"):])
    return cfg


def rid(exp, arm, N, rep):
    return f"{exp}_{arm}_N{N}_r{rep:03d}"


def plan(exp, reps: dict, choice: dict, pilot: bool):
    runs = []
    rmax = max(n for v in reps.values() for n in v.values())
    for r in range(rmax):
        for N in sorted({N for v in reps.values() for N in v}):
            for arm in ARMS:
                if arm in reps and N in reps[arm] and r < reps[arm][N]:
                    runs.append(dict(run_id=rid(exp, arm, N, r), kind="gauge_v11", exp=exp, arm=arm, N=N, rep=r,
                                     cfg=resolve(arm, choice), pilot=pilot))
    ids = [x["run_id"] for x in runs]
    assert len(ids) == len(set(ids))
    return runs


def write(name, cohort, runs):
    os.makedirs(PLANS, exist_ok=True)
    path = os.path.join(PLANS, f"{name}.json")
    with open(path, "w") as f:
        json.dump({"cohort": cohort, "runs": runs}, f, indent=0)
    print(name, len(runs), hashlib.sha256(open(path, "rb").read()).hexdigest())
    return path


def main(mode):
    if mode == "cohort":
        if not os.path.exists(os.path.join(PLANS, "ENG_GO")):
            raise SystemExit("plans_v11/ENG_GO absent: the engineering decision is not GO; no cohort plan is written")
        choice = json.load(open(os.path.join(PLANS, "eng_choice.json")))
        if choice.get("status") != "GO":
            raise SystemExit("eng_choice.json status is not GO")
        for nm in ("S2_2v11_pilot", "S2_2v11_main"):
            if os.path.exists(os.path.join(PLANS, nm + ".json")):
                raise SystemExit(f"{nm}.json exists; plans are written once")
        drop = ("SGD-K", "KS-4", "KS-16") if choice.get("K_min") is None else ()   # no Euler K qualified (logged)
        pr = {a: v for a, v in PILOT_REPS.items() if a not in drop}
        mr = {a: v for a, v in MAIN_REPS.items() if a not in drop}
        write("S2_2v11_pilot", "S2_2v11_pilot", plan("S2.2v11-pilot", pr, choice, True))
        write("S2_2v11_main", "S2_2v11_main", plan("S2.2v11", mr, choice, False))
    elif mode == "unittest":
        write("UT_S2_2v11_pilot", "UT_S2_2v11_pilot", plan("S2.2v11-unittest-pilot", UT_PILOT_REPS, UT_CHOICE, True))
        write("UT_S2_2v11_main", "UT_S2_2v11_main", plan("S2.2v11-unittest", UT_MAIN_REPS, UT_CHOICE, False))
    else:
        raise SystemExit(mode)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
