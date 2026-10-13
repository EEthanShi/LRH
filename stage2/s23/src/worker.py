"""Stage A worker (PREREG sections 4.1, 5, 12): executes plan items; restart-safe (skips items whose
.done marker exists); never touches other users' processes.

  python worker.py --phase pre|main --pool gpu|cpu --wid k --nw W

GPU pool (kinds A, G1b, G3A, D2): one CUDA process holds the dense I_1 Cholesky factor (8.5 GB) used
as the PCG preconditioner; each solve's wall time is recorded in the GPU ledger; no new item starts
once the ledger reaches GPU_SOFT_STOP_H (11.5 h), and the process exits at GPU_HARD_STOP_H (12 h).
CPU pool (kinds B, G3B): numpy only.
Each item: exact hierarchical multinomial counts from its data key -> its learner -> runs/<cell>/<id>.npy
(unit-diagonal K-hat, float64) + <id>.json (certificate, path, timings) + <id>.done.
Failure handling (section 5.4): a failed certificate is re-solved once from a keyed random factor; a
run that still fails is kept and flagged (cert_pass false), never dropped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import s23common as C

if __name__ == "__main__":
    C.cap_threads(1)

import numpy as np

import cfg as CF
import instance as I
import learnB as B
import sampling as SM

GPU_KINDS = ("A", "G1b", "G3A", "D2")
CPU_KINDS = ("B", "G3B")


def out_paths(it):
    d = os.path.join(C.RUNS, it["cell"])
    return d, os.path.join(d, it["id"] + ".npy"), os.path.join(d, it["id"] + ".json"), os.path.join(d, it["id"] + ".done")


def is_done(it):
    return os.path.exists(out_paths(it)[3])


def save(it, K, info, extra_arrays=None):
    d, pk, pj, pd = out_paths(it)
    os.makedirs(d, exist_ok=True)
    if K is not None:
        np.save(pk, np.asarray(K, dtype=np.float64))
    if extra_arrays:
        for k, v in extra_arrays.items():
            np.save(os.path.join(d, f"{it['id']}__{k}.npy"), np.asarray(v, dtype=np.float64))
    info = dict(info)
    info.update({"id": it["id"], "utc": C.utcnow(), "item": it})
    C.dump_json(info, pj)
    open(pd, "w").close()


def load_frozen():
    fr = C.load_json(C.FROZEN_PRED)
    arr = dict(np.load(C.PRESTEP_ARR))
    return fr, arr


def instance_from(fr):
    cfg = fr["cfg"]
    return I.build(fr["instance"]["key"], cfg["t"], cfg["geometry"])


def counts_for(it, inst):
    N = 2 ** int(it["log2N"])
    r = C.rng(it["data_key"])
    return SM.sample_gibbs(inst, N, r), N


class GpuCtx:
    def __init__(self, inst, fr, arr, worker):
        import torch
        import fisher as F
        import learnA as A
        self.torch, self.A = torch, A
        self.dev = C.torch_device()
        self.inst, self.fr, self.arr, self.worker = inst, fr, arr, worker
        self.sc = fr["cfg"]["solver"]
        with C.GpuSection(worker, "setup_I1"):
            pi = torch.as_tensor(inst.pi_gibbs(), dtype=torch.float64, device=self.dev)
            Ks = torch.as_tensor(inst.K, dtype=torch.float64, device=self.dev)
            I1 = F.assemble_I1(pi, Ks, inst.t)
            self.L = F.cholesky_inplace_safe(I1)
            del I1, pi
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        self.pre = A.FisherPrecond(self.L, inst.n)

    def T(self, M):
        return self.torch.as_tensor(np.asarray(M), dtype=self.torch.float64, device=self.dev)

    def solveA(self, W, tau, K_init_name, init_key, newton_first=True, random_start=False):
        A, sc = self.A, self.sc
        n = self.inst.n
        if random_start:
            Y0 = A.random_factor(n, n, C.seed_from_key(init_key + "|g3"), self.dev)
            K, info = A.solve_A(W, tau, None, self.pre, rel=sc["bm_rel"], max_evals=sc["bm_max_evals"],
                                step_tol=sc["newton_step_tol"], Y_init=Y0, newton_first=False,
                                stall_chunks=sc["bm_stall_chunks"], pcg_rtol=sc["pcg_rtol"])
        else:
            K, info = A.solve_A(W, tau, self.T(self.arr[K_init_name]), self.pre, rel=sc["bm_rel"],
                                max_evals=sc["bm_max_evals"], step_tol=sc["newton_step_tol"],
                                stall_chunks=sc["bm_stall_chunks"], pcg_rtol=sc["pcg_rtol"])
        if not info["cert_pass"]:
            Y0 = A.random_factor(n, n, C.seed_from_key(init_key + "|resolve"), self.dev)
            K2, info2 = A.solve_A(W, tau, None, self.pre, rel=sc["bm_rel"], max_evals=sc["bm_max_evals"],
                                  step_tol=sc["newton_step_tol"], Y_init=Y0, newton_first=False,
                                  stall_chunks=sc["bm_stall_chunks"], pcg_rtol=sc["pcg_rtol"])
            info["resolve"] = {k: info2.get(k) for k in ("path", "cert_pass", "lam_min_K", "bm_evals")}
            if info2["cert_pass"]:
                K, info = K2, dict(info2, resolved=True, first_attempt=info)
        return K.detach().cpu().numpy(), info


def do_item(it, inst, fr, arr, gpu, worker):
    t0 = time.time()
    kind = it["kind"]
    Cn, N = counts_for(it, inst)
    ts = time.time() - t0
    if kind in ("B", "G3B"):
        C1, C2 = SM.marginals(Cn)
        Kraw, Kn, info = B.solve_B(C1, C2, tol_abs=fr["cfg"]["solver"]["b_tol_abs"],
                                   init="zero" if kind == "G3B" else "proj")
        info["sample_s"] = ts
        save(it, Kn, info)
        return
    torch = gpu.torch
    C1d, C2d = SM.marginals(Cn) if kind == "D2" else (None, None)
    W = torch.as_tensor(Cn, dtype=torch.float64, device=gpu.dev) / float(N)
    del Cn
    with C.GpuSection(worker, it["id"]) as sec:
        if kind == "A":
            K, info = gpu.solveA(W, it["tau"], it["K_init"], it["init_key"])
            info["sample_s"] = ts
            extra = None
        elif kind == "G3A":
            K, info = gpu.solveA(W, it["tau"], None, it["init_key"], random_start=True)
            info["sample_s"] = ts
            extra = None
        elif kind == "G1b":
            # Lemma T identity check (never a claim): same counts at t and t/2; PD margin, uniqueness
            # (no Hessian null space) and the curvature-derived tolerance (learnA.lemmaT_decision).
            K, info = gpu.solveA(W, it["tau"], "0_A", it["init_key"] + "|t")
            K2, info2 = gpu.solveA(W, it["tau"] / 2.0, "Rm_A", it["init_key"] + "|t2")
            A_ = gpu.A
            lamI = float(fr["lam_min_I1"]["lam_min_I1_lb"])
            hb1 = hb2 = None
            if info.get("path") in ("newton", "bm+newton"):
                hb1 = A_.theta_hess_bounds(A_.InfoNCE(W, it["tau"]), gpu.T(K), gpu.L, it["init_key"] + "|lz_t")
            if info2.get("path") in ("newton", "bm+newton"):
                hb2 = A_.theta_hess_bounds(A_.InfoNCE(W, it["tau"] / 2.0), gpu.T(K2), gpu.L, it["init_key"] + "|lz_t2")
            dec = A_.lemmaT_decision(K, info, hb1, K2, info2, hb2, 0.5, lamI)
            info = {"t": info, "t_half": info2, "hess_t": hb1, "hess_t_half": hb2, "lemmaT": dec, "sample_s": ts}
            extra = {"t_half": K2}
        elif kind == "D2":
            Kraw, Kn, ib = B.solve_B(C1d, C2d, tol_abs=fr["cfg"]["solver"]["b_tol_abs"])
            KH, ih = B.train_H(C1d, C2d, N, C.seed_from_key(it["init_key"]), device=gpu.dev)
            dmax = float(np.max(np.abs(KH - Kraw))) if KH is not None else float("inf")
            info = {"erm": ib, "trainH": ih, "max_abs_HHt_minus_ERM": dmax,
                    "rel_max": dmax / float(np.max(np.abs(Kraw))), "sample_s": ts}
            K, extra = Kn, ({"HHt": KH, "erm_raw": Kraw} if KH is not None else {"erm_raw": Kraw})
        else:
            raise ValueError(kind)
    info["gpu_section_s"] = sec.seconds
    save(it, K, info, extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["pre", "main"])
    ap.add_argument("--pool", required=True, choices=["gpu", "cpu"])
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--nw", type=int, default=1)
    a = ap.parse_args()
    C.verify_frozen_code(strict=True)
    plan = C.load_json(os.path.join(C.PLANS, "stageA_plan.json"))
    fr, arr = load_frozen()
    inst = instance_from(fr)
    kinds = GPU_KINDS if a.pool == "gpu" else CPU_KINDS
    skip = set()
    if a.phase == "main":
        g4p = os.path.join(C.STAGEA_DIR, "g4_decision.json")
        if not os.path.exists(g4p):
            raise SystemExit("main phase requires results/stageA/g4_decision.json (gates)")
        g4 = C.load_json(g4p)
        if not g4.get("main_allowed", False):
            raise SystemExit("gates did not allow the main runs (stop rule)")
        red = set(g4.get("reductions", []))
        for it in plan["items"]:
            if any(x in red and it.get(f"red_{x}") for x in ("a", "b", "c")):
                skip.add(it["id"])
    items = [it for it in plan["items"] if it["phase"] == a.phase and it["kind"] in kinds and it["id"] not in skip]
    items = [it for i, it in enumerate(items) if i % a.nw == a.wid]
    worker = f"{a.pool}{a.wid}_{a.phase}"
    gpu = GpuCtx(inst, fr, arr, worker) if a.pool == "gpu" and any(not is_done(it) for it in items) else None
    logp = os.path.join(C.LOGS, f"worker_{worker}.log")
    os.makedirs(C.LOGS, exist_ok=True)
    with open(logp, "a") as lf:
        for it in items:
            if is_done(it):
                continue
            if a.pool == "gpu":
                if not C.gpu_budget_ok(soft=True):
                    lf.write(f"{C.utcnow()} STOP: GPU ledger {C.gpu_hours_used():.3f} h >= soft stop\n")
                    sys.exit(4)
            t0 = time.time()
            do_item(it, inst, fr, arr, gpu, worker)
            lf.write(f"{C.utcnow()} done {it['id']} {time.time() - t0:.2f}s\n")
            lf.flush()
            if a.pool == "gpu" and not C.gpu_budget_ok(soft=False):
                lf.write(f"{C.utcnow()} HARD STOP: GPU ledger {C.gpu_hours_used():.3f} h\n")
                sys.exit(5)


if __name__ == "__main__":
    main()
