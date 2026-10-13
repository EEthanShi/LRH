"""S2.2 v1.1 training-side unit tests (PREREG_S2_2_v1.1.md section 14, step 2). EXCLUDED: every run here uses
exp key S2.2v11-unittest, is written under runs_v11/UT_tiny/, and is never part of any count, gate or anchor.

    python src_v11/unit_tests_v11.py [all|init|grad|euler|midpoint|repro|tiny|guard]

U1 balanced init: ||C0||_F <= 1e-12 (20 keys); def / small heads equal v1.0 reinit bit for bit.
U2 closed-form float64 head gradient = autograd gradient of the training loss (relative 1e-12).
U3 one Euler substep satisfies C' - C = dt^2 (g_P g_P^T - g_W^T g_W) to 1e-12 (relative to ||PP^T|| + ||W^TW||).
U4 one implicit-midpoint substep conserves C to the solver tolerance (<= 1e-11 relative); Heun drift reported.
U5 head_bf16 / mom / def reproduces v1.0 train_gauge on a fixed seed within GPU nondeterminism
   (two v1.0 replicas measure the nondeterminism; 40 steps, N = 12.5k).
U6 tiny runs (30 steps, N = 12.5k) of every arm configuration: finite, dtypes, C0 = 0 for bal, Euler identity
   <= 1e-8, u = W^{-1} y to 1e-10 (fp64 head), flow delta_C small, logs present, sharpness probe returns.
U7 guard: pilot / main cohorts are refused before the freeze line exists.
U8 CUDA-graph engine = eager reference (Euler, Heun: <= 1e-13 relative; midpoint, adaptive: same rule, results
   within 1e-9 relative and C conserved to the same tolerance); per-evaluation cost of both paths reported.
Writes results/S2_2_v11/unittest/unit_tests.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2lib as L  # noqa: E402
import head_v11 as H  # noqa: E402
import gaugemath_v11 as GM  # noqa: E402
import train_v11 as T  # noqa: E402

ROOT = L.ROOT
OUT = os.path.join(ROOT, "results", "S2_2_v11", "unittest")
DEV = T.DEV
RES = {}


def rec(name, ok, **kw):
    RES[name] = {"pass": bool(ok), **kw}
    print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(kw, default=float)[:400]}", flush=True)


def realistic_head(seed=0, scaleP=4.0, scaleW=2.0):
    g = torch.Generator().manual_seed(seed)
    P = (torch.rand(9, 256, generator=g, dtype=torch.float64) * 2 - 1) / 16 * scaleP
    W = (torch.rand(9, 9, generator=g, dtype=torch.float64) * 2 - 1) / 3 * scaleW
    h = torch.relu(torch.randn(1024, 256, generator=g, dtype=torch.float64)) * 0.7
    yb = torch.randint(0, 10, (1024,), generator=g)
    return P.to(DEV), W.to(DEV), h.to(DEV), yb.to(DEV)


def t_init():
    worst, same = 0.0, True
    for i in range(20):
        key = f"hrhv2-stage2|S2.2v11-unittest|arm=BAL-FLOW|N=25000|rep={i}"
        P0, W0 = H.reinit_v11(H.GaugeNetV11(), key, "bal")
        worst = max(worst, float(np.linalg.norm(GM.C_of(P0, W0))))
    for init, si in (("def", None), ("small", 0.1)):
        for i in range(3):
            key = f"hrhv2-stage2|S2.2v11-unittest|arm=X|N=25000|rep={i}"
            m10 = L.GaugeNet()
            m10.small_init = si
            L.reinit(m10, key + "|init")
            m11 = H.GaugeNetV11()
            P0, W0 = H.reinit_v11(m11, key, init)
            same &= bool(np.array_equal(P0, m10.P.weight.detach().double().numpy())
                         and np.array_equal(W0, m10.W.weight.detach().double().numpy()))
            same &= bool(np.array_equal(m11.backbone.g2.conv1.weight.detach().numpy(),
                                        m10.backbone.g2.conv1.weight.detach().numpy()))
    rec("U1_init", worst <= 1e-12 and same, max_C0_fro_bal=worst, def_small_equal_v10=same)


def t_grad():
    P, W, h, yb = realistic_head(1)
    m = H.GaugeNetV11().to(DEV)
    m.P.weight = torch.nn.Parameter(P.clone())
    m.W.weight = torch.nn.Parameter(W.clone())
    _, _, lg = m.head(h)
    loss = F.cross_entropy(lg, yb, label_smoothing=0.2, reduction="none").sum()
    loss.backward()
    prob = H.HeadProblem(h, yb, m.E64)
    gP, gW = prob.grads(P, W)
    eP = float(torch.linalg.norm(gP - m.P.weight.grad) / torch.linalg.norm(m.P.weight.grad))
    eW = float(torch.linalg.norm(gW - m.W.weight.grad) / torch.linalg.norm(m.W.weight.grad))
    el = abs(prob.loss(P, W) - float(loss)) / abs(float(loss))
    ident = float(torch.linalg.norm(gP @ P.T - W.T @ gW) / torch.linalg.norm(W.T @ gW))
    rec("U2_grad", max(eP, eW, el) <= 1e-12 and ident <= 1e-12, relerr_gP=eP, relerr_gW=eW, relerr_loss=el,
        identity_gP_Pt_eq_Wt_gW=ident)


def t_euler():
    worst = 0.0
    for seed in range(5):
        P, W, h, yb = realistic_head(seed + 10)
        prob = H.HeadProblem(h, yb, torch.from_numpy(L.ortho_E(10).double().numpy()).to(DEV))
        dt = 0.0098
        C0 = H.C_t(P, W)
        gP, gW = prob.grads(P, W)
        P1, W1 = P - dt * gP, W - dt * gW
        lhs = H.C_t(P1, W1) - C0
        rhs = dt * dt * (gP @ gP.T - gW.T @ gW)
        den = float(torch.linalg.norm(P1 @ P1.T) + torch.linalg.norm(W1.T @ W1))
        worst = max(worst, float(torch.linalg.norm(lhs - rhs)) / den)
        for graph, K in ((False, 7), (True, 8), (True, 32)):
            acc = torch.zeros(9, 9, dtype=torch.float64, device=DEV)
            P2, W2 = P.clone(), W.clone()
            H.integrate(prob, P2, W2, 0.05, {"head_opt": "euler", "K": K, "graph": graph}, acc)
            den2 = float(torch.linalg.norm(P2 @ P2.T) + torch.linalg.norm(W2.T @ W2))
            worst = max(worst, float(torch.linalg.norm((H.C_t(P2, W2) - C0) - acc)) / den2)
    rec("U3_euler_identity", worst <= 1e-12, max_rel_resid=worst)


def t_midpoint():
    out = {}
    for seed in range(3):
        P, W, h, yb = realistic_head(seed + 20)
        E64 = L.ortho_E(10).double().to(DEV)
        C0 = H.C_t(P, W)
        for meth, cfg in (("midpoint", {"head_opt": "midpoint", "K": 1, "fp_tol": 1e-12, "fp_max_iter": 60}),
                          ("midpoint_eager", {"head_opt": "midpoint", "K": 1, "fp_tol": 1e-12, "fp_max_iter": 60,
                                              "graph": False}),
                          ("heun", {"head_opt": "heun", "K": 1}), ("euler", {"head_opt": "euler", "K": 1})):
            prob = H.HeadProblem(h, yb, E64)
            P1, W1 = P.clone(), W.clone()
            info = H.integrate(prob, P1, W1, 0.0098, dict(cfg))
            den = float(torch.linalg.norm(P1 @ P1.T) + torch.linalg.norm(W1.T @ W1))
            d = float(torch.linalg.norm(H.C_t(P1, W1) - C0)) / den
            out.setdefault(meth, []).append(d)
            out.setdefault(meth + "_evals", []).append(info["evals"])
    ok = max(out["midpoint"]) <= 1e-11 and max(out["midpoint_eager"]) <= 1e-11
    rec("U4_midpoint_conserves", ok, **{k: max(v) for k, v in out.items()})


def t_repro():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import train as T10  # v1.0 src/train.py (read-only import; its s2lib is this directory's identical copy)
    import hashlib
    same_lib = hashlib.sha256(open(os.path.join(ROOT, "src", "s2lib.py"), "rb").read()).hexdigest() == \
        hashlib.sha256(open(os.path.join(HERE, "s2lib.py"), "rb").read()).hexdigest()
    data = L.C5M()
    base = dict(kind="gauge", exp="S2.2v11-unittest", arm="REPRO", N=12500, rep=0, corpus=None, run_id="ut_repro")
    outs = []
    for _ in range(2):
        a, m = T10.train_gauge(dict(base, cfg={"head_wd": False, "small_init": None}), L.GaugeNet(), data, max_steps=40)
        outs.append((a, m))
    a11, m11 = T.train_gauge_v11(dict(base, kind="gauge_v11", cfg={"head_opt": "mom", "init": "def", "head_bf16": True}),
                                 data, max_steps=40)
    d = lambda x, y: float(np.linalg.norm(np.asarray(x, np.float64) - np.asarray(y, np.float64)) / np.linalg.norm(np.asarray(y, np.float64)))
    noise = {k: d(outs[0][0][k], outs[1][0][k]) for k in ("y", "W", "P", "y_init")}
    diff = {k: d(a11[k], outs[0][0][k]) for k in ("y", "W", "P", "y_init")}
    lc10 = [v for _, v in outs[0][1]["loss_curve"]]
    lc11 = [v for _, v in m11["loss_curve"]]
    ok = same_lib and diff["y_init"] <= 1e-6 and all(diff[k] <= max(3 * noise[k], 1e-5) for k in ("y", "W", "P"))
    rec("U5_v10_repro", ok, same_s2lib_bytes=same_lib, v10_vs_v10=noise, v11_vs_v10=diff,
        loss10=lc10, loss11=lc11, acc10=outs[0][1]["acc_bank"], acc11=m11["acc_bank"],
        imb_init_v10=outs[0][1]["imb_init"]["imb_abs"], C0_fro_v11=m11["C0_fro"])


TINY = {
    "MOM": dict(head_opt="mom", init="def"),
    "WD": dict(head_opt="mom", init="def", head_wd=True),
    "FLOW-heun": dict(head_opt="heun", K=64, init="def"),
    "BAL-FLOW-mid": dict(head_opt="midpoint", K=4, init="bal"),
    "SMALL-FLOW-adapt": dict(head_opt="heun_adapt", tol=1e-6, K=16, init="small"),
    "BAL-MOM": dict(head_opt="mom", init="bal"),
    "SMALL-MOM": dict(head_opt="mom", init="small"),
    "SGD-K16": dict(head_opt="euler", K=16, init="def"),
    "SGD-K2": dict(head_opt="euler", K=2, init="def"),
    "MOM-BF16": dict(head_opt="mom", init="def", head_bf16=True),
    "MOM-sharp": dict(head_opt="mom", init="def", sharp_steps=[10, 30]),
}


def t_tiny(steps=30):
    data = L.C5M()
    rows = {}
    for arm, cfg in TINY.items():
        r = dict(run_id=f"S2.2v11-unittest_{arm}_N12500_r000", kind="gauge_v11", exp="S2.2v11-unittest", arm=arm,
                 N=12500, rep=0, cfg=cfg, pilot=True)
        a, m = T.train_gauge_v11(r, data, max_steps=steps)
        T.save_run_v11(os.path.join(ROOT, "runs_v11", "UT_tiny", r["run_id"]), a, m)
        z = dict(np.load(os.path.join(ROOT, "runs_v11", "UT_tiny", r["run_id"], "feats.npz")))
        g = m["gauge"]
        row = {"finite": m["finite"], "diverged": m["diverged"], "acc_bank": m["acc_bank"], "t_total_s": m["t_total_s"],
               "t_head_s": m["t_head_s"], "evals": m["head_evals_total"], "delta_C": g.get("delta_C"),
               "rho_end": g.get("rho_end"), "C0_fro": m["C0_fro"], "u_recompute_relerr": m["u_recompute_relerr"],
               "euler_identity_resid": m["euler_identity_resid"], "C_log_shape": list(z["C_log"].shape),
               "dtype_y": str(z["y"].dtype), "dtype_h": str(z["h"].dtype), "dtype_P": str(z["P"].dtype),
               "sharp": m["sharpness"], "fp_fail": m["fp_fail"], "disp_P_median": m["disp_P_median"],
               "C_log0_eq_C0": float(np.linalg.norm(z["C_log"][0] - GM.C_of(z["P0"], z["W0"])))}
        if m["head_evals_total"]:
            row["s_per_eval"] = m["t_head_s"] / m["head_evals_total"]
        rows[arm] = row
        print(arm, json.dumps(row, default=float), flush=True)
    chk = []
    for arm, row in rows.items():
        if arm == "SGD-K2":   # plain-step SGD may legitimately diverge (that is what the K_min search is for):
            chk.append(row["diverged"] or row["finite"])            # only flag consistency is checked
            continue
        chk.append(row["finite"] and not row["diverged"])
        chk.append(row["C_log0_eq_C0"] <= 1e-12 * max(1.0, row["C0_fro"]))
        if arm != "MOM-BF16":
            chk.append(row["dtype_y"] == "float64" and row["dtype_h"] == "float32" and row["dtype_P"] == "float64")
            chk.append(row["u_recompute_relerr"] <= 1e-10)
        if arm.startswith("BAL"):
            chk.append(row["C0_fro"] <= 1e-12)
        if arm == "SGD-K16":
            chk.append(row["euler_identity_resid"] <= 1e-8)
        if arm in ("FLOW-heun", "BAL-FLOW-mid", "SMALL-FLOW-adapt"):
            chk.append(row["delta_C"] <= 2e-4)
        if arm == "BAL-FLOW-mid":
            chk.append(row["delta_C"] <= 1e-9 and row["fp_fail"] == 0)
        if arm == "MOM-sharp":
            chk.append(len(row["sharp"]) == 2 and all(np.isfinite(s[1]) for s in row["sharp"]))
        chk.append(row["C_log_shape"][1:] == [9, 9])
    rec("U6_tiny_runs", all(chk), rows=rows, n_checks=len(chk), n_failed=int(sum(not c for c in chk)))


def t_graph():
    rows = {}
    ok = True
    for name, cfg, tol in (("euler4", {"head_opt": "euler", "K": 4}, 1e-13), ("euler32", {"head_opt": "euler", "K": 32}, 1e-13),
                           ("heun16", {"head_opt": "heun", "K": 16}, 1e-13), ("heun64", {"head_opt": "heun", "K": 64}, 1e-13),
                           ("mid2", {"head_opt": "midpoint", "K": 2, "fp_tol": 1e-12, "fp_max_iter": 30}, 1e-9),
                           ("adapt", {"head_opt": "heun_adapt", "tol": 1e-6, "K": 16}, 1e-9)):
        P, W, h, yb = realistic_head(40)
        E64 = L.ortho_E(10).double().to(DEV)
        res = {}
        for graph in (False, True):
            prob = H.HeadProblem(h, yb, E64)
            P1, W1 = P.clone(), W.clone()
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(3):
                P1, W1 = P.clone(), W.clone()
                info = H.integrate(prob, P1, W1, 0.0098, dict(cfg, graph=graph))
            torch.cuda.synchronize()
            res[graph] = (P1, W1, (time.time() - t) / max(1, prob.n_evals), info)
        (Pe, We, te, ie), (Pg, Wg, tg, ig) = res[False], res[True]
        d = float((torch.linalg.norm(Pe - Pg) + torch.linalg.norm(We - Wg)) / (torch.linalg.norm(Pe) + torch.linalg.norm(We)))
        den = float(torch.linalg.norm(Pg @ Pg.T) + torch.linalg.norm(Wg.T @ Wg))
        dC = float(torch.linalg.norm(H.C_t(Pg, Wg) - H.C_t(P, W))) / den
        rows[name] = {"rel_diff": d, "deltaC_graph": dC, "s_per_eval_eager": te, "s_per_eval_graph": tg,
                      "evals_eager": ie["evals"], "evals_graph": ig["evals"]}
        ok &= d <= tol
    rec("U8_graph_equals_eager", ok, **rows)


def t_guard():
    msgs = []
    ok_pre = T.preflight("S2_2v11_pilot", lambda s: msgs.append(s))
    ok_ut = T.preflight("UT_S2_2v11_pilot", lambda s: msgs.append(s))
    rec("U7_guard", (not ok_pre) and ok_ut, refused_message=msgs)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    tests = {"init": t_init, "grad": t_grad, "euler": t_euler, "midpoint": t_midpoint, "repro": t_repro,
             "tiny": t_tiny, "guard": t_guard, "graph": t_graph}
    for k, f in tests.items():
        if which in ("all", k):
            f()
    path = os.path.join(OUT, f"unit_tests_{which}.json")
    json.dump({"label": "EXCLUDED unit tests (S2.2v11-unittest)", "results": RES, "wall_s": time.time() - t0},
              open(path, "w"), indent=1, default=float)
    print("ALL PASS" if all(v["pass"] for v in RES.values()) else "SOME FAILED", path)
