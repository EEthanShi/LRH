"""Automatic preregistered gates (PREREG_stage2.md §2.3, §3.2) and tolerance freeze (§2.4, §3.3).

    python gates.py all      # wait for S2.2 pilot -> G1 + eps_S2_2; wait for S2.1 pilot -> G2 + eps_S2_1

Gate decisions are written first (runs/GATE_S2_x_main.go|.nogo, results/gates/G*.json) so main
runs start without waiting for the tolerance ladders; eps files are computed from pilot pairs only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crit as C  # noqa: E402

ROOT = C.ROOT
STATUS = os.path.expanduser("~/hrh_v2_stage1_20261008/STATUS.md")
GD = os.path.join(ROOT, "results", "gates")
os.makedirs(GD, exist_ok=True)


def status(msg):
    with open(STATUS, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [stage2] {msg}\n")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def wait_plan(name, max_h=10):
    plan = json.load(open(os.path.join(ROOT, "plans", f"{name}.json")))
    ids = [r["run_id"] for r in plan["runs"]]
    t0 = time.time()
    while True:
        done = [i for i in ids if os.path.exists(os.path.join(C.run_dir(name, i), ".done"))]
        failed = [i for i in ids if os.path.exists(os.path.join(C.run_dir(name, i), ".failed"))]
        if len(done) + len(failed) == len(ids):
            return plan, done, failed
        if time.time() - t0 > max_h * 3600:
            return plan, done, failed
        time.sleep(60)


def pilot_pairs(plan, arms, cohort):
    runs = {r["run_id"]: r for r in plan["runs"]}
    out = []
    for arm in arms:
        reps = sorted(r["rep"] for r in plan["runs"] if r["arm"] == arm)
        byrep = {r["rep"]: r["run_id"] for r in plan["runs"] if r["arm"] == arm}
        for j in range(len(reps) // 2):
            out.append((arm, byrep[2 * j], byrep[2 * j + 1]))
    return out


def freeze_eps(cohort, pairs, reps_spec, out_path, anchor=None):
    """reps_spec: list of (tag, feature_key, pre). anchor: {tag_target: tag_source} for function-anchored eps."""
    rows = []
    cache = {}
    for arm, ra, rb in pairs:
        A, B = C.load_run(cohort, ra), C.load_run(cohort, rb)
        for tag, key, pre in reps_spec:
            if not (C.run_ok(A, key) and C.run_ok(B, key)):
                d = {k: C.FAIL for k in C.TOL_KEYS}
            else:
                row = C.pair_ladder(C.geom(A["feats"][key], pre), C.geom(B["feats"][key], pre))
                d = C.deltas(row)
            d.update(arm=arm, A=ra, B=rb, rep=tag)
            rows.append(d)
    eps, eps90 = {}, {}
    for tag, _, _ in reps_spec:
        src = anchor.get(tag, tag) if anchor else tag
        R = [r for r in rows if r["rep"] == src]
        eps[tag], eps90[tag] = {}, {}
        for k in C.TOL_KEYS:
            v = np.array([r[k] for r in R], dtype=float)
            v = v[~np.isnan(v)]
            eps[tag][k] = float(v.max()) if v.size else float("nan")
            eps90[tag][k] = float(np.quantile(v, 0.9)) if v.size else float("nan")
    out = {"cohort": cohort, "rule": "eps_Q = max of Delta_Q over pooled within-arm pilot pairs (PREREG_stage2 v1.0)",
           "anchor": anchor or {}, "n_pairs": len(pairs), "eps": eps, "eps_q90": eps90,
           "pilot_rows": rows}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


def gate_s22():
    plan, done, failed = wait_plan("S2_2_pilot")
    imb = {}
    for arm in ("noWD", "WD", "smallinit"):
        v = [C.load_run("S2_2_pilot", r["run_id"])["meta"]["imb_final"]["imb_rel"]
             for r in plan["runs"] if r["arm"] == arm and r["run_id"] in done]
        imb[arm] = {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
    acc = {arm: float(np.mean([C.load_run("S2_2_pilot", r["run_id"])["meta"]["acc_bank"]
                               for r in plan["runs"] if r["arm"] == arm and r["run_id"] in done]))
           for arm in ("noWD", "WD", "smallinit")}
    ok = imb["noWD"]["median"] >= 0.2 and imb["WD"]["median"] <= 0.05
    res = {"gate": "G1", "pass": ok, "imbalance": imb, "acc_bank_mean": acc, "n_done": len(done),
           "n_failed": len(failed), "rule": "median rho_end >= 0.2 noWD and <= 0.05 WD"}
    json.dump(res, open(os.path.join(GD, "G1.json"), "w"), indent=1)
    open(os.path.join(ROOT, "runs", "GATE_S2_2_main." + ("go" if ok else "nogo")), "w").write(json.dumps(res))
    status(f"gate G1 (S2.2 imbalance, pilot EXCLUDED, N=100k, 16/arm) {'PASS' if ok else 'FAIL'}: median rho_end "
           f"noWD {imb['noWD']['median']:.3f}, WD {imb['WD']['median']:.3f}, smallinit {imb['smallinit']['median']:.3f}; "
           f"acc(bank) {acc}; done {len(done)}/48 (results/gates/G1.json, runs/GATE_S2_2_main.{'go' if ok else 'nogo'})")
    pairs = pilot_pairs(plan, ("noWD", "WD", "smallinit"), "S2_2_pilot")
    out_path = os.path.join(ROOT, "prereg", "eps_S2_2.json")
    t = time.time()
    freeze_eps("S2_2_pilot", pairs, [("y", "y", "none"), ("u", "u", "none"), ("h", "h", "none"), ("h_l2", "h", "l2")],
               out_path, anchor={"u": "y"})
    status(f"eps_S2_2 frozen from pilot pairs only (24 within-arm pairs, max rule; u anchored to y): "
           f"prereg/eps_S2_2.json sha256 {sha(out_path)} ({time.time() - t:.0f} s)")
    return ok


def gate_s21():
    plan, done, failed = wait_plan("S2_1_pilot")
    arms = ("A", "B", "Atau")
    knn, tt, sN = {}, {}, {}
    for arm in arms:
        ms = [C.load_run("S2_1_pilot", r["run_id"])["meta"] for r in plan["runs"] if r["arm"] == arm and r["run_id"] in done]
        knn[arm] = float(np.mean([m["knn_acc_h"] if np.isfinite(m["knn_acc_h"]) else 0.0 for m in ms]))
        tt[arm] = [m["t_total_s"] for m in ms]
    from hrhv2 import structural as st
    for arm, ra, rb in pilot_pairs(plan, arms, "S2_1_pilot"):
        A, B = C.load_run("S2_1_pilot", ra), C.load_run("S2_1_pilot", rb)
        if C.run_ok(A, "z") and C.run_ok(B, "z"):
            GA, GB = C.geom(A["feats"]["z"], "l2"), C.geom(B["feats"]["z"], "l2")
            nA, _ = st.knn_table(GA.D, 11)
            nB, _ = st.knn_table(GB.D, 11)
            s = st.knn_overlap(nA, nB, 10)
        else:
            s = 0.0
        sN.setdefault(arm, []).append(float(s))
    means = {a: float(np.mean(v)) for a, v in sN.items()}
    med_t = float(np.median(sum(tt.values(), [])))
    # amendment 1: effective time per run = pilot wall-clock span / number of pilot runs
    import calendar
    ends, starts = [], []
    for r in plan["runs"]:
        if r["run_id"] not in done:
            continue
        dpath = os.path.join(C.run_dir("S2_1_pilot", r["run_id"]), ".done")
        t_end = calendar.timegm(time.strptime(open(dpath).read().strip(), "%Y-%m-%dT%H:%M:%SZ"))
        ends.append(t_end)
        starts.append(t_end - C.load_run("S2_1_pilot", r["run_id"])["meta"]["t_total_s"])
    eff_t = float((max(ends) - min(starts)) / max(1, len(ends)))
    ok = (all(knn[a] >= 0.70 for a in arms) and max(knn.values()) - min(knn.values()) <= 0.05
          and all(means[a] >= 0.3 for a in arms) and eff_t <= 300)
    res = {"gate": "G2", "pass": ok, "knn_acc_h_mean": knn, "within_sN10_z_mean": means, "within_sN10_z": sN,
           "median_wall_s": med_t, "effective_s_per_run": eff_t, "n_done": len(done), "n_failed": len(failed),
           "rule": "knn>=0.70 all arms, spread<=0.05, within s_N(10) z-l2 >= 0.3, effective time per run <= 300 s "
                   "(PREREG amendment 1; literal median wall reported)"}
    json.dump(res, open(os.path.join(GD, "G2.json"), "w"), indent=1)
    open(os.path.join(ROOT, "runs", "GATE_S2_1_main." + ("go" if ok else "nogo")), "w").write(json.dumps(res))
    status(f"gate G2 (S2.1 pilot, EXCLUDED, N=150k, 16/arm) {'PASS' if ok else 'FAIL'}: kNN(h) "
           + ", ".join(f"{a} {knn[a]:.3f}" for a in arms) + "; within s_N(10) z "
           + ", ".join(f"{a} {means[a]:.3f}" for a in arms) + f"; effective {eff_t:.0f} s/run (amendment 1; literal median wall {med_t:.0f} s); done {len(done)}/48 "
           f"(results/gates/G2.json, runs/GATE_S2_1_main.{'go' if ok else 'nogo'})")
    pairs = pilot_pairs(plan, arms, "S2_1_pilot")
    out_path = os.path.join(ROOT, "prereg", "eps_S2_1.json")
    t = time.time()
    freeze_eps("S2_1_pilot", pairs, [("z", "z", "l2"), ("h", "h", "l2"), ("z_raw", "z", "none")], out_path)
    status(f"eps_S2_1 frozen from pilot pairs only (24 within-arm pairs, max rule): prereg/eps_S2_1.json sha256 "
           f"{sha(out_path)} ({time.time() - t:.0f} s)")
    return ok


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "S2_2"):
        gate_s22()
    if which in ("all", "S2_1"):
        gate_s21()
    open(os.path.join(GD, f"GATES_{which}.done"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
