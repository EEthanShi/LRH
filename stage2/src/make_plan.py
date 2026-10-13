"""Write the Stage-2 run plans (deterministic). Order is rep-major so that a partially
completed queue always leaves every cell with (almost) the same number of runs.

plans/S2_1_pilot.json  S2.1 pilot (EXCLUDED): arms A,B,Atau at N=150k, reps 0-15
plans/S2_2_pilot.json  S2.2 pilot (EXCLUDED): arms noWD,WD,smallinit at N=100k, reps 0-15
plans/S2_1_main.json   S2.1 main: arms A,B,Atau x N in {50k,500k} x reps 0-63, then the fixed-epoch
                       sensitivity cell (arms A,B, N=50k, 10 epochs = 500k images, reps 0-31)
plans/S2_2_main.json   S2.2 main: arms noWD,WD,smallinit x N in {25k,250k} x reps 0-191
                       + shared-corpus cell noWD-shared, N=25k, corpora 0-3 x reps 0-47
"""
import json
import os

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
SSL_ARMS = {
    "A": {"loss": "infonce", "tau": 0.2},
    "B": {"loss": "spectral", "mu": 1.0},
    "Atau": {"loss": "infonce", "tau": 0.5},
}
GAUGE_ARMS = {
    "noWD": {"head_wd": False, "small_init": None},
    "WD": {"head_wd": True, "small_init": None},
    "smallinit": {"head_wd": False, "small_init": 0.1},
}


def rid(exp, arm, N, rep, corpus=None):
    s = f"{exp}_{arm}_N{N}_r{rep:03d}"
    return s if corpus is None else s + f"_c{corpus}"


def plan_ssl(exp, Ns, reps):
    runs = []
    for r in reps:
        for N in Ns:
            for arm, cfg in SSL_ARMS.items():
                runs.append(dict(run_id=rid(exp, arm, N, r), kind="ssl", exp=exp, arm=arm, N=N, rep=r,
                                 corpus=None, cfg=cfg, pilot=exp.endswith("pilot")))
    return runs


def plan_fixedepoch():
    """Fixed-epoch sensitivity cell: arms A and B at N=50k with 10 epochs (500k images seen),
    i.e. the epoch count of the 500k cell instead of the fixed 5M-image budget."""
    runs = []
    for r in range(32):
        for arm in ("A", "B"):
            cfg = dict(SSL_ARMS[arm], budget=500000)
            runs.append(dict(run_id=rid("S2.1-fixedepoch", arm, 50000, r), kind="ssl", exp="S2.1-fixedepoch",
                             arm=arm, N=50000, rep=r, corpus=None, cfg=cfg, pilot=False))
    return runs


def plan_gauge(exp, Ns, reps, shared=None):
    runs = []
    for r in reps:
        for N in Ns:
            for arm, cfg in GAUGE_ARMS.items():
                runs.append(dict(run_id=rid(exp, arm, N, r), kind="gauge", exp=exp, arm=arm, N=N, rep=r,
                                 corpus=None, cfg=cfg, pilot=exp.endswith("pilot")))
        if shared is not None and r < shared["reps"]:
            for c in range(shared["corpora"]):
                runs.append(dict(run_id=rid(exp, "noWD-shared", shared["N"], r, c), kind="gauge", exp=exp,
                                 arm="noWD-shared", N=shared["N"], rep=r, corpus=c, cfg=GAUGE_ARMS["noWD"],
                                 pilot=False))
    return runs


def main():
    os.makedirs(os.path.join(ROOT, "plans"), exist_ok=True)
    plans = {
        "S2_1_pilot": plan_ssl("S2.1-pilot", [150000], range(16)),
        "S2_2_pilot": plan_gauge("S2.2-pilot", [100000], range(16)),
        "S2_1_main": plan_ssl("S2.1", [50000, 500000], range(64)) + plan_fixedepoch(),
        "S2_2_main": plan_gauge("S2.2", [25000, 250000], range(192),
                                shared=dict(N=25000, corpora=4, reps=48)),
    }
    for name, runs in plans.items():
        ids = [r["run_id"] for r in runs]
        assert len(ids) == len(set(ids))
        with open(os.path.join(ROOT, "plans", f"{name}.json"), "w") as f:
            json.dump({"cohort": name, "runs": runs}, f, indent=0)
        print(name, len(runs))


if __name__ == "__main__":
    main()
