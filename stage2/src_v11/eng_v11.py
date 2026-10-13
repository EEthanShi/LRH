"""S2.2 v1.1 engineering phase (EXCLUDED; PREREG_S2_2_v1.1.md section 3 "Engineering phase", section 12).

    python src_v11/eng_v11.py run [--workers 4]      # runs the stages below, then the gate and the stop rule
    python src_v11/eng_v11.py decide                 # re-evaluate the gate from finished runs only

Runs at N in {12.5k, 100k} (neither size is judged), 488 steps, exp key S2.2v11-eng, no ladders, no pair metric.
Stages (each stage = 4 runs per size of one configuration, launched as 4 queue workers):
  base      MOM (def init): timing of momentum arms; reps 0-1 carry the sharpness probe (lambda_max of the head
            Hessian of the summed batch loss at steps 50, 150, 300, 488; tau_t * lambda_max recorded).
  euler     K in (1, 2, 4, 8, 16, 32, 64) ascending; K_min = first K with 8/8 runs non-diverged and finite.
            None qualifies -> SGD-K and KS arms are dropped (logged).
  flow      three families, each searched in ascending cost order, stopping at its first passing configuration
            or as soon as a failing configuration already costs > 40 s/run:
              heun K in (16, 64, 256, 1024); midpoint K in (4, 16, 64) (fp_tol 1e-12);
              heun_adapt tol in (1e-4, 1e-5, 1e-6, 1e-7).
            pass = 8/8 runs non-diverged, finite, no fixed-point failure, and max delta_C <= 5e-5.
            choice = passing configuration with the smallest t_proj_s (the larger per-size mean t_total_s of the
            runs without the sharpness probe; ties: heun, midpoint, heun_adapt).
  scale6    the chosen flow configuration, 12 runs at 100k with 6 workers (throughput with 6 workers).
Stop rule (draft section 3 + author decision 2026-10-09):
  STOP-a  no flow configuration meets delta_C <= 5e-5 at <= 40 s/run;
  STOP-b  projected total GPU-hours (section 12 convention: sum over planned runs of the measured s/run at
          4 workers, plus engineering itself) > 35.
  On STOP nothing proceeds: plans_v11/ENG_STOP is written with the measured numbers; no ENG_GO, no cohort plan.
Outputs: plans_v11/eng_choice.json (+ ENG_GO or ENG_STOP), results/S2_2_v11/eng/{eng_summary.json, ENG_REPORT.md}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_plan_v11 as MP  # noqa: E402

ROOT = MP.ROOT
PY = os.path.expanduser("~/hrh_v2_stage1_20261008/venv/bin/python")
COHORT = "S2_2v11_eng"
RUNS = os.path.join(ROOT, "runs_v11", COHORT)
OUT = os.path.join(ROOT, "results", "S2_2_v11", "eng")
PLANS = MP.PLANS
SIZES = (12500, 100000)
DELTA_C_TARGET = 5e-5
CAP_S = 40.0
BUDGET_GPU_H = 35.0
ENG_CAP_GPU_H = 2.0
EULER_KS = (1, 2, 4, 8, 16, 32, 64)
FAMILIES = {"heun": [dict(head_opt="heun", K=k) for k in (16, 64, 256, 1024)],
            "midpoint": [dict(head_opt="midpoint", K=k, fp_tol=1e-12, fp_max_iter=30) for k in (4, 16, 64)],
            "heun_adapt": [dict(head_opt="heun_adapt", tol=t, K=16) for t in (1e-4, 1e-5, 1e-6, 1e-7)]}
STATUS = os.path.expanduser("~/hrh_v2_stage1_20261008/STATUS.md")


def tag_of(cfg):
    o = cfg["head_opt"]
    if o == "mom":
        return "MOM"
    if o == "euler":
        return f"EULER-K{cfg['K']}"
    if o == "heun":
        return f"HEUN-K{cfg['K']}"
    if o == "midpoint":
        return f"MID-K{cfg['K']}"
    if o == "heun_adapt":
        return f"ADAPT-T{cfg['tol']:.0e}"
    raise ValueError(o)


def stage_runs(cfg, sizes=SIZES, reps=4, sharp_reps=(), tag_suffix=""):
    tag = tag_of(cfg) + tag_suffix
    runs = []
    for r in range(reps):
        for N in sizes:
            c = dict(cfg, init="def", head_wd=False)
            if r in sharp_reps:
                c["sharp_steps"] = [50, 150, 300, 488]
            runs.append(dict(run_id=f"S2.2v11-eng_{tag}_N{N}_r{r:03d}", kind="gauge_v11", exp="S2.2v11-eng",
                             arm=tag, N=N, rep=r, cfg=c, pilot=True))
    return tag, runs


def launch(runs, name, nw):
    os.makedirs(PLANS, exist_ok=True)
    path = os.path.join(PLANS, f"eng_{name}.json")
    json.dump({"cohort": COHORT, "runs": runs}, open(path, "w"), indent=0)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    procs = []
    for w in range(nw):
        lf = open(os.path.join(OUT, "logs", f"{name}_w{w}.log"), "a")
        procs.append(subprocess.Popen([PY, os.path.join(HERE, "train_v11.py"), "worker", "--plan", path, "--wid", str(w)],
                                      stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT,
                                      env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                               OPENBLAS_NUM_THREADS="1")))
        time.sleep(2)
    for p in procs:
        p.wait()
    # one retry of failed runs (crash re-queued once)
    failed = [r for r in runs if os.path.exists(os.path.join(RUNS, r["run_id"], ".failed"))]
    if failed:
        p = subprocess.Popen([PY, os.path.join(HERE, "train_v11.py"), "worker", "--plan", path, "--wid", "90"],
                             stdout=open(os.path.join(OUT, "logs", f"{name}_retry.log"), "a"), stderr=subprocess.STDOUT,
                             cwd=ROOT)
        p.wait()
    return evaluate(runs)


def evaluate(runs):
    ms = []
    for r in runs:
        d = os.path.join(RUNS, r["run_id"])
        if os.path.exists(os.path.join(d, ".done")):
            ms.append(json.load(open(os.path.join(d, "meta.json"))))
    n = len(runs)
    ok = [m for m in ms if not m["diverged"] and m["finite"]]
    dC = [m["gauge"].get("delta_C", float("nan")) for m in ok]
    # timing excludes the sharpness-probe runs (the probe is engineering-only work); falls back to all runs
    mt = [m for m in ms if not m["cfg"].get("sharp_steps")] or ms
    t = [m["t_total_s"] for m in mt]
    by = {}
    for N in sorted({r["N"] for r in runs}):
        mN = [m for m in ms if m["N"] == N]
        tN = [m["t_total_s"] for m in mt if m["N"] == N]
        by[N] = {"n_done": len(mN), "n_ok": sum(1 for m in mN if not m["diverged"] and m["finite"]),
                 "acc_bank_mean": float(np.mean([m["acc_bank"] for m in mN])) if mN else float("nan"),
                 "t_mean_s": float(np.mean(tN)) if tN else float("nan"),
                 "delta_C_max": float(np.max([m["gauge"].get("delta_C", np.nan) for m in mN if m["finite"]]))
                 if any(m["finite"] for m in mN) else float("nan")}
    sharp = [s for m in ms for s in m.get("sharpness", [])]
    t_by = [v["t_mean_s"] for v in by.values() if np.isfinite(v["t_mean_s"])]
    return {"n_planned": n, "n_done": len(ms), "n_ok": len(ok), "all_ok": len(ok) == n,
            "delta_C_max": float(np.max(dC)) if dC else float("nan"),
            "delta_C_median": float(np.median(dC)) if dC else float("nan"),
            "fp_fail": int(sum(m.get("fp_fail", 0) for m in ms)),
            "t_mean_s": float(np.mean(t)) if t else float("nan"), "t_median_s": float(np.median(t)) if t else float("nan"),
            # projection convention (resolution R-PROJ): the larger of the per-size mean s/run
            "t_proj_s": float(max(t_by)) if t_by else float("nan"),
            "t_head_mean_s": float(np.mean([m.get("t_head_s", 0.0) for m in ms])) if ms else float("nan"),
            "evals_mean": float(np.mean([m.get("head_evals_total", 0) for m in ms])) if ms else float("nan"),
            "substeps_max": int(max([m.get("head_substeps_max", 0) for m in ms] or [0])),
            "disp_P_median": float(np.median([m.get("disp_P_median", np.nan) for m in ms])) if ms else float("nan"),
            "euler_identity_resid_max": float(np.nanmax([m.get("euler_identity_resid", np.nan) for m in ms]))
            if ms and any(np.isfinite(m.get("euler_identity_resid", np.nan)) for m in ms) else float("nan"),
            "sharpness": sharp, "by_N": by,
            "gpu_s": float(np.sum(t))}


def flow_pass(ev):
    return ev["all_ok"] and ev["fp_fail"] == 0 and ev["delta_C_max"] <= DELTA_C_TARGET


def projection(s_mom, s_flow, s_euler_kmin, c_eval, K_min, eng_gpu_s):
    """Section 12 convention: sum over planned (pilot + main) runs of s/run at 4 workers."""
    reps = {}
    for src in (MP.MAIN_REPS, MP.PILOT_REPS):
        for a, v in src.items():
            reps[a] = reps.get(a, 0) + sum(v.values())
    s = {}
    for a in reps:
        if a in MP.FLOW_ARMS:
            s[a] = s_flow
        elif a in MP.MOM_ARMS:
            s[a] = s_mom
        elif K_min is None:
            s[a] = 0.0  # dropped
        else:
            mult = {"SGD-K": 1, "KS-4": 4, "KS-16": 16}[a]
            s[a] = s_euler_kmin + c_eval * 488 * (mult - 1) * K_min
    if K_min is None:
        for a in ("SGD-K", "KS-4", "KS-16"):
            reps[a] = 0
    rows = {a: {"runs": reps[a], "s_per_run": s[a], "gpu_h": reps[a] * s[a] / 3600} for a in reps}
    total = sum(r["gpu_h"] for r in rows.values()) + eng_gpu_s / 3600
    return rows, total


def status(msg):
    with open(STATUS, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [s22v11-eng] {msg}\n")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run_all(nw):
    import guard_v11
    ok, msg = guard_v11.verify_frozen()
    if not ok:
        raise SystemExit(f"REFUSED: {msg}")
    os.makedirs(OUT, exist_ok=True)
    S = {"stages": {}, "order": []}
    spent = 0.0

    def do(name, runs, nw_=nw, cfg=None):
        nonlocal spent
        ev = launch(runs, name, nw_)
        if cfg is not None:
            ev["cfg"] = cfg
        S["stages"][name] = ev
        S["order"].append(name)
        spent += ev["gpu_s"]
        json.dump(S, open(os.path.join(OUT, "eng_summary.json"), "w"), indent=1)
        print(name, json.dumps({k: v for k, v in ev.items() if k not in ("sharpness",)}), flush=True)
        return ev

    _, runs = stage_runs(dict(head_opt="mom"), sharp_reps=(0, 1))
    do("base_MOM", runs)
    K_min = None
    for K in EULER_KS:
        _, runs = stage_runs(dict(head_opt="euler", K=K))
        ev = do(f"euler_K{K}", runs)
        if ev["all_ok"]:
            K_min = K
            break
        if spent / 3600 > ENG_CAP_GPU_H:
            break
    S["K_min"] = K_min
    for fam, cfgs in FAMILIES.items():
        for cfg in cfgs:
            if spent / 3600 > ENG_CAP_GPU_H:
                S.setdefault("eng_cap_hit", True)
                break
            tag, runs = stage_runs(cfg)
            ev = do(f"flow_{tag}", runs, cfg=cfg)
            if flow_pass(ev) or ev["t_proj_s"] > CAP_S:
                break
    decide(S, nw, run_scale=True, do=do)


def decide(S=None, nw=4, run_scale=False, do=None):
    if S is None:
        S = json.load(open(os.path.join(OUT, "eng_summary.json")))
    st = S["stages"]
    cands = []
    order = {"heun": 0, "midpoint": 1, "heun_adapt": 2}
    for name, ev in st.items():
        if name.startswith("flow_") and "cfg" in ev and flow_pass(ev):
            cands.append((ev["t_proj_s"], order[ev["cfg"]["head_opt"]], name))
    cands.sort()
    s_mom = st["base_MOM"]["t_proj_s"]
    K_min = S.get("K_min")
    s_e = st[f"euler_K{K_min}"]["t_proj_s"] if K_min else 0.0
    # per-gradient-evaluation cost of the Euler head: the larger of the host-side estimate and the wall-clock
    # difference to MOM per evaluation (CUDA-graph replays are asynchronous, so host time can understate it)
    c_eval = 0.0
    if K_min:
        ev_e = st[f"euler_K{K_min}"]
        c_eval = max(ev_e["t_head_mean_s"] / max(1.0, ev_e["evals_mean"]),
                     max(0.0, s_e - s_mom) / max(1.0, ev_e["evals_mean"]))
    eng_s = sum(ev["gpu_s"] for ev in st.values())
    choice = {"status": "STOP", "K_min": K_min, "flow": None, "reasons": []}
    if not cands:
        choice["reasons"].append("STOP-a: no flow configuration passed (8/8 ok, no fixed-point failure, max delta_C <= 5e-5)")
    else:
        t_best, _, best = cands[0]
        cfg = dict(st[best]["cfg"])
        choice["flow"] = cfg
        choice["flow_stage"] = best
        if t_best > CAP_S:
            choice["reasons"].append(f"STOP-a: cheapest passing flow configuration costs {t_best:.1f} s/run > {CAP_S} s")
        if run_scale and do is not None and "scale6" not in st and not choice["reasons"]:
            tag, runs = stage_runs(cfg, sizes=(100000,), reps=12, tag_suffix="-x6")
            ev = do("scale6", runs, 6, cfg=cfg)
            st["scale6"] = ev
            eng_s += ev["gpu_s"]
        rows, total = projection(s_mom, t_best, s_e, c_eval, K_min, eng_s)
        choice["projection_rows"] = rows
        choice["projection_total_gpu_h"] = total
        if total > BUDGET_GPU_H:
            choice["reasons"].append(f"STOP-b: projected total {total:.1f} GPU-h > {BUDGET_GPU_H}")
    if not choice["reasons"]:
        choice["status"] = "GO"
    choice.update({"s_mom": s_mom, "s_euler_kmin": s_e, "c_eval_s": c_eval, "eng_gpu_h": eng_s / 3600,
                   "candidates": [{"stage": n, "t_proj_s": t} for t, _, n in cands],
                   "sharpness": st["base_MOM"].get("sharpness", []),
                   "rule": "cheapest flow config with 8/8 ok, no fp failure, max delta_C <= 5e-5; STOP-a / STOP-b "
                           "(PREREG_S2_2_v1.1 section 3, author decision 2026-10-09: STOP if projected > 35 GPU-h)"})
    json.dump(S, open(os.path.join(OUT, "eng_summary.json"), "w"), indent=1)
    cp = os.path.join(PLANS, "eng_choice.json")
    json.dump(choice, open(cp, "w"), indent=1)
    for f in ("ENG_GO", "ENG_STOP"):
        if os.path.exists(os.path.join(PLANS, f)):
            os.remove(os.path.join(PLANS, f))
    marker = os.path.join(PLANS, "ENG_GO" if choice["status"] == "GO" else "ENG_STOP")
    open(marker, "w").write(json.dumps({"eng_choice_sha256": sha(cp), "reasons": choice["reasons"]}))
    write_report(S, choice)
    status(f"ENGINEERING {choice['status']} (EXCLUDED runs, runs_v11/{COHORT}): K_min={K_min}, flow={choice['flow']}, "
           f"s/run MOM {s_mom:.1f}, chosen flow {cands[0][0] if cands else float('nan'):.1f}, projected total "
           f"{choice.get('projection_total_gpu_h', float('nan')):.1f} GPU-h, engineering {eng_s / 3600:.2f} GPU-h; "
           f"reasons {choice['reasons']}; plans_v11/eng_choice.json sha256 {sha(cp)}")
    return choice


def write_report(S, choice):
    L = ["# S2.2 v1.1 engineering report (EXCLUDED runs; not evidence)", "",
         f"Decision: **{choice['status']}** {choice['reasons']}", "",
         "| stage | n ok / planned | max delta_C | median delta_C | mean s/run | proj s/run | head s | evals/run | fp fail | acc (12.5k, 100k) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name in S["order"] + (["scale6"] if "scale6" in S["stages"] and "scale6" not in S["order"] else []):
        ev = S["stages"][name]
        acc = ", ".join(f"{v['acc_bank_mean']:.4f}" for v in ev["by_N"].values())
        L.append(f"| {name} | {ev['n_ok']}/{ev['n_planned']} | {ev['delta_C_max']:.2e} | {ev['delta_C_median']:.2e} | "
                 f"{ev['t_mean_s']:.1f} | {ev['t_proj_s']:.1f} | {ev['t_head_mean_s']:.1f} | {ev['evals_mean']:.0f} | {ev['fp_fail']} | {acc} |")
    L += ["", "Sharpness (step, lambda_max, tau*lambda_max, iterations):", "",
          "\n".join(f"- {s}" for s in choice.get("sharpness", [])), ""]
    if "projection_rows" in choice:
        L += ["| arm | runs | s/run | GPU-h |", "|---|---|---|---|"]
        for a, r in choice["projection_rows"].items():
            L.append(f"| {a} | {r['runs']} | {r['s_per_run']:.1f} | {r['gpu_h']:.2f} |")
        L.append(f"| total (incl. engineering) | | | {choice['projection_total_gpu_h']:.2f} |")
    open(os.path.join(OUT, "ENG_REPORT.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "decide"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "run":
        run_all(a.workers)
    else:
        decide()
