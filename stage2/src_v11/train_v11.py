"""S2.2 v1.1: training of one gauge-net run and the queue worker (PREREG_S2_2_v1.1.md sections 3-5).

    python src_v11/train_v11.py worker --plan plans_v11/S2_2v11_pilot.json plans_v11/S2_2v11_main.json --wid 0

Copy of v1.0 src/train.py (train_gauge + worker) with the v1.1 extensions: float64 head, head treatments
{mom, euler, heun, heun_adapt, midpoint}, inits {def, small, bal}, C logged every 10 steps, P0/W0 saved,
per-step head displacement / substep counts, Euler drift accumulator, optional sharpness probe.
Runs are written to runs_v11/<cohort>/<run_id>/{feats.npz, meta.json, .done|.failed} (dtypes preserved).
Worker logs print only accuracy and time (never rho or any C statistic).
Pilot / main cohorts refuse to start unless the frozen code hashes verify (guard_v11) and the engineering
decision plans_v11/ENG_GO exists; the main cohort also waits for runs_v11/GATE_S2_2v11_main.go (gates_v11).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2lib as L  # noqa: E402  (verbatim copy of v1.0)
import head_v11 as H  # noqa: E402
import gaugemath_v11 as GM  # noqa: E402

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True   # v1.0 trunk settings; float64 head matmuls never use TF32
torch.backends.cudnn.allow_tf32 = True
DEV = torch.device("cuda")
ROOT = L.ROOT
RUNS = os.path.join(ROOT, "runs_v11")

GAUGE_DEFAULT_V11 = dict(budget=500_000, bs=1024, lr=11.5, momentum=0.85, wd=0.0153, bias_scaler=64.0,
                         label_smoothing=0.2, whiten_bias_frac=0.3, head_wd=False,
                         head_opt="mom", K=1, tol=None, fp_tol=1e-12, fp_max_iter=30,
                         init="def", head_bf16=False, log_C_every=10, sharp_steps=[])
EXCLUDED_EXPS = ("S2.2v11-eng", "S2.2v11-pilot", "S2.2v11-unittest")


def run_key(r: dict) -> str:
    return f"hrhv2-stage2|{r['exp']}|arm={r['arm']}|N={r['N']}|rep={r['rep']}"


def data_key(r: dict) -> str:
    return run_key(r) + "|data"


def stream_batches(N, bs, steps, gen):
    perm = torch.randperm(N, device=DEV, generator=gen)
    p = 0
    for _ in range(steps):
        if p + bs > N:
            perm = torch.randperm(N, device=DEV, generator=gen)
            p = 0
        yield perm[p:p + bs]
        p += bs


@torch.no_grad()
def forward_eval_v11(model, X_u8, bs=1024):
    """v1.0 forward_eval (eval mode, unaugmented) keeping each output's dtype (u, y float64 for the fp64 head)."""
    was = model.training
    model.eval()
    outs = None
    for i in range(0, X_u8.shape[0], bs):
        xb = L.normalize(L.u8_to_float(X_u8[i:i + bs])).contiguous(memory_format=torch.channels_last)
        o = model(xb)
        if outs is None:
            outs = [[] for _ in o]
        for j, t in enumerate(o):
            outs[j].append(t.detach().cpu())
    model.train(was)
    return [torch.cat(t) for t in outs]


@torch.no_grad()
def forward_init_stats_v11(model, X_u8):
    """v1.0 forward_init_stats (train-mode BN on the probe batch, on a copy) keeping dtypes."""
    import copy
    m2 = copy.deepcopy(model)
    m2.train()
    xb = L.normalize(L.u8_to_float(X_u8)).contiguous(memory_format=torch.channels_last)
    o = m2(xb)
    del m2
    return [t.detach().cpu() for t in o]


def head_np(model):
    return (model.P.weight.detach().double().cpu().numpy().copy(), model.W.weight.detach().double().cpu().numpy().copy())


def train_gauge_v11(r: dict, data: L.C5M, max_steps=None, log=print):
    cfg = dict(GAUGE_DEFAULT_V11)
    cfg.update(r.get("cfg", {}))
    key = run_key(r)
    t0 = time.time()
    ii = data.draw_indices(int(r["N"]), data_key(r))
    Xn, Yn = data.gather(ii)
    Xg = L.to_gpu_u8(Xn, DEV)
    Yg = torch.from_numpy(Yn).to(DEV)
    del Xn
    t_data = time.time() - t0
    model = H.GaugeNetV11(head_bf16=bool(cfg["head_bf16"]))
    P0, W0 = H.reinit_v11(model, key, cfg["init"])
    model.to(DEV).to(memory_format=torch.channels_last)
    model.train()
    g_ord = L.torch_gen(key + "|order", DEV)
    wsel = torch.randperm(Xg.shape[0], device=DEV, generator=g_ord)[:5000]
    model.backbone.init_whitening(L.normalize(L.u8_to_float(Xg[wsel])))
    S_u8 = L.to_gpu_u8(data.S_X, DEV)
    init_out = forward_init_stats_v11(model, S_u8)
    head_opt = cfg["head_opt"]
    bs = int(cfg["bs"])
    mom = cfg["momentum"]
    kilo = 1024 * (1 + 1 / (1 - mom))
    lr = cfg["lr"] / kilo
    wd = cfg["wd"] * bs / kilo
    lr_b = lr * cfg["bias_scaler"]
    head = [model.P.weight, model.W.weight]
    head_ids = {id(p) for p in head}
    norm_biases = [p for n_, p in model.named_parameters() if "norm" in n_ and p.requires_grad]
    nb_ids = {id(p) for p in norm_biases}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in nb_ids and id(p) not in head_ids]
    groups = [dict(params=norm_biases, lr=lr_b, weight_decay=wd / lr_b),
              dict(params=other, lr=lr, weight_decay=wd / lr)]
    if head_opt == "mom":
        groups.append(dict(params=head, lr=lr, weight_decay=(wd / lr) if cfg["head_wd"] else 0.0))
    else:
        if cfg["head_wd"]:
            raise ValueError("head weight decay is defined only for head_opt = mom")
        for p in head:
            p.requires_grad_(False)   # the trunk still receives the gradient through the head
    opt = torch.optim.SGD(groups, momentum=mom, nesterov=True)
    steps = int(cfg["budget"]) // bs
    if max_steps is not None:
        steps = min(steps, max_steps)
    wu = int(steps * 0.23)

    def lr_mult(s):
        if s < wu:
            f = s / wu
            return 0.2 * (1 - f) + 1.0 * f
        f = (s - wu) / max(1, steps - wu)
        return 1.0 * (1 - f) + 0.07 * f
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    lossf = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"], reduction="none")
    g_aug = L.torch_gen(key + "|aug", DEV)
    wb_steps = int(cfg["whiten_bias_frac"] * steps)
    every = int(cfg["log_C_every"])
    sharp_steps = set(int(x) for x in cfg.get("sharp_steps", []))
    losses = []
    C_log, C_steps = [], []
    disp_t = torch.zeros((steps, 2), dtype=torch.float64, device=DEV)   # kept on the GPU (no per-step sync)
    nsub = np.zeros(steps, dtype=np.int64)
    nev = np.zeros(steps, dtype=np.int64)
    taus = np.zeros(steps)
    sharp = []
    acc = torch.zeros((9, 9), dtype=torch.float64, device=DEV) if head_opt == "euler" else None
    hcfg = dict(cfg)
    acc_loss = torch.zeros((), device=DEV, dtype=torch.float64)
    diverged, fp_fail, t_head = False, 0, 0.0
    t1 = time.time()
    for s, idx in enumerate(stream_batches(Xg.shape[0], bs, steps, g_ord)):
        if s == wb_steps:
            model.backbone.whiten.bias.requires_grad = False
        if s % every == 0:
            with torch.no_grad():
                Pd, Wd = model.P.weight.detach().double(), model.W.weight.detach().double()
                C_log.append(torch.stack([Pd @ Pd.T, Wd.T @ Wd]))
                C_steps.append(s)
        xb = L.flip_translate(L.u8_to_float(Xg[idx]), g_aug, pad=2)
        xb = L.normalize(xb).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, _, _, logits = model(xb)
        loss = lossf(logits.float() if cfg["head_bf16"] else logits, Yg[idx]).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        Pb, Wb = model.P.weight.detach().clone(), model.W.weight.detach().clone()
        prob = None
        if (s + 1) in sharp_steps:
            prob = H.HeadProblem(h.detach(), Yg[idx], model.E64)
            lam, it = H.hvp_lambda_max(prob, model.P.weight.detach().double(), model.W.weight.detach().double(),
                                       key=key + f"|sharp|{s + 1}")
            tau_s = opt.param_groups[1]["lr"] / (1 - mom)
            sharp.append([s + 1, lam, tau_s * lam, it])
        opt.step()
        taus[s] = opt.param_groups[1]["lr"] / (1 - mom)
        if head_opt != "mom":
            th = time.time()
            if prob is None:
                prob = H.HeadProblem(h.detach(), Yg[idx], model.E64)
            info = H.integrate(prob, model.P.weight.data, model.W.weight.data, float(taus[s]), hcfg, acc)
            nsub[s], nev[s] = info["substeps"], info["evals"]
            fp_fail += info["fp_fail"]
            if info.get("nonfinite"):
                diverged = True
            t_head += time.time() - th
        with torch.no_grad():
            disp_t[s, 0] = torch.linalg.norm((model.P.weight.detach() - Pb).double())
            disp_t[s, 1] = torch.linalg.norm((model.W.weight.detach() - Wb).double())
        sched.step()
        acc_loss += loss.detach().double() / bs
        if (s + 1) % 50 == 0 or s + 1 == steps or diverged:
            n_ = 50 if (s + 1) % 50 == 0 else ((s + 1) % 50 or 50)
            val = float(acc_loss) / n_
            acc_loss.zero_()
            losses.append((s + 1, val))
            hfin = bool(torch.isfinite(model.P.weight).all() and torch.isfinite(model.W.weight).all())
            if not math.isfinite(val) or not hfin or diverged:
                diverged = True
                log(f"[{r['run_id']}] diverged at step {s + 1}")
                break
    model.backbone.whiten.bias.requires_grad = True
    torch.cuda.synchronize()
    t_train = time.time() - t1
    disp = disp_t.cpu().numpy()
    CL = torch.stack(C_log).cpu().numpy() if C_log else np.zeros((0, 2, 9, 9))
    PPt_log = np.linalg.norm(CL[:, 0], axis=(1, 2))
    WtW_log = np.linalg.norm(CL[:, 1], axis=(1, 2))
    C_log = CL[:, 0] - CL[:, 1]
    h, u, y, lg = forward_eval_v11(model, S_u8)
    bank_u8 = L.to_gpu_u8(data.bank_X, DEV)
    _, _, _, lgb = forward_eval_v11(model, bank_u8)
    sy = torch.from_numpy(data.S_Y)
    by = torch.from_numpy(data.bank_Y)
    acc_S = float((lg.argmax(1) == sy).double().mean())
    acc_bank = float((lgb.argmax(1) == by).double().mean())
    with torch.no_grad():
        g_tr = L.torch_gen(key + "|evalsel", DEV)
        tr_sel = torch.randperm(Xg.shape[0], device=DEV, generator=g_tr)[:4096]
        _, _, _, lgt = forward_eval_v11(model, Xg[tr_sel])
        ce_tr = float(F.cross_entropy(lgt.double(), Yg[tr_sel].cpu()))
        _, _, _, lgh = forward_eval_v11(model, L.to_gpu_u8(data.hold_X, DEV))
        ce_ho = float(F.cross_entropy(lgh.double(), torch.from_numpy(data.hold_Y)))
        acc_tr = float((lgt.argmax(1) == Yg[tr_sel].cpu()).double().mean())
    P1, W1 = head_np(model)
    finite = bool(torch.isfinite(h).all() and torch.isfinite(u).all() and torch.isfinite(y).all()
                  and np.all(np.isfinite(P1)) and np.all(np.isfinite(W1)))
    gauge = {}
    if finite:
        try:
            gauge = GM.run_gauge(P1, W1, P0, W0)
        except Exception as e:  # singular head: recorded, the run still counts (it fails every rung only if non-finite)
            gauge = {"error": repr(e)}
    yy, uu = y.double().numpy(), u.double().numpy()
    u_rel = float(np.linalg.norm(uu - yy @ np.linalg.inv(W1).T) / np.linalg.norm(uu)) if finite else float("nan")
    C0 = GM.C_of(P0, W0)
    C1 = GM.C_of(P1, W1) if finite else np.full((9, 9), np.nan)
    A_end = acc.cpu().numpy() if acc is not None else np.zeros((9, 9))
    euler_resid = float(np.linalg.norm((C1 - C0) - A_end) / (np.linalg.norm(P1 @ P1.T) + np.linalg.norm(W1.T @ W1))) \
        if (head_opt == "euler" and finite) else float("nan")
    hcfg.pop("_dt_hint", None)
    meta = dict(r)
    meta.update(dict(key=key, data_key=data_key(r), cfg=cfg, steps=steps, steps_done=len(losses) and losses[-1][0],
                     diverged=diverged, finite=finite,
                     acc_S=acc_S, acc_bank=acc_bank, acc_train=acc_tr, ce_train=ce_tr, ce_hold=ce_ho,
                     gen_gap=ce_ho - ce_tr, loss_curve=losses, gauge=gauge, u_recompute_relerr=u_rel,
                     euler_identity_resid=euler_resid, C0_fro=float(np.linalg.norm(C0)),
                     head_substeps_total=int(nsub.sum()), head_evals_total=int(nev.sum()), fp_fail=int(fp_fail),
                     head_substeps_max=int(nsub.max()) if steps else 0, sharpness=sharp,
                     disp_P_median=float(np.median(disp[:, 0])), disp_W_median=float(np.median(disp[:, 1])),
                     final_train_loss=losses[-1][1] if losses else float("nan"),
                     t_data_s=t_data, t_train_s=t_train, t_head_s=t_head, t_head_note="host-side time; queued GPU work not included",
                     t_total_s=time.time() - t0,
                     subset_sha256=hashlib.sha256(ii.tobytes()).hexdigest(),
                     host=socket.gethostname(), torch=torch.__version__, code="src_v11/train_v11.py"))
    arrays = dict(h=h.float().numpy(), u=u.numpy(), y=y.numpy(), h_init=init_out[0].float().numpy(),
                  u_init=init_out[1].numpy(), y_init=init_out[2].numpy(), P0=P0, W0=W0, P=P1, W=W1,
                  C_log=C_log, C_log_steps=np.array(C_steps), PPt_log=PPt_log, WtW_log=WtW_log, disp=disp, substeps=nsub, evals=nev, tau=taus, A_end=A_end,
                  sharp=np.array(sharp, dtype=np.float64).reshape(-1, 4))
    del Xg, Yg, S_u8, bank_u8, model, opt
    return arrays, meta


def save_run_v11(out_dir: str, arrays: dict, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "feats.tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
    os.replace(tmp, os.path.join(out_dir, "feats.npz"))
    with open(os.path.join(out_dir, "meta.json.tmp"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    os.replace(os.path.join(out_dir, "meta.json.tmp"), os.path.join(out_dir, "meta.json"))


# ----------------------------------------------------------------------------- worker (v1.0 copy)
def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def claim(run_dir: str) -> bool:
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, ".done")):
        return False
    lock = os.path.join(run_dir, ".lock")
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "t": time.time()}).encode())
            os.close(fd)
            if os.path.exists(os.path.join(run_dir, ".done")):
                os.remove(lock)
                return False
            return True
        except FileExistsError:
            try:
                with open(lock) as f:
                    info = json.load(f)
                if info.get("host") == socket.gethostname() and not pid_alive(int(info["pid"])):
                    os.replace(lock, lock + f".stale.{os.getpid()}")
                    continue
            except Exception:
                pass
            return False
    return False


GATED = {"S2_2v11_main": "GATE_S2_2v11_main"}
COHORT_PLANS = ("S2_2v11_pilot", "S2_2v11_main")


def preflight(cohort: str, log) -> bool:
    if cohort not in COHORT_PLANS:
        return True
    import guard_v11
    ok, msg = guard_v11.verify_frozen()
    if not ok:
        log(f"REFUSED {cohort}: {msg}")
        return False
    if not os.path.exists(os.path.join(ROOT, "plans_v11", "ENG_GO")):
        log(f"REFUSED {cohort}: plans_v11/ENG_GO absent (engineering decision not GO)")
        return False
    return True


def wait_gate(cohort: str, log) -> bool:
    if cohort not in GATED:
        return True
    base = os.path.join(RUNS, GATED[cohort])
    t0 = time.time()
    while True:
        if os.path.exists(base + ".go"):
            return True
        if os.path.exists(base + ".nogo"):
            log(f"gate {GATED[cohort]} = NOGO: skipping cohort {cohort}")
            return False
        if time.time() - t0 > 12 * 3600:
            log(f"gate {GATED[cohort]} undecided after 12 h: skipping cohort {cohort}")
            return False
        time.sleep(30)


def worker(plan_paths: list, wid: int, max_steps=None):
    data = L.C5M()
    log = lambda s: print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [w{wid}] {s}", flush=True)
    for plan_path in plan_paths:
        with open(plan_path) as f:
            plan = json.load(f)
        cohort = plan["cohort"]
        base = os.path.join(RUNS, cohort)
        log(f"plan {plan_path} n_runs={len(plan['runs'])} pid={os.getpid()}")
        if not preflight(cohort, log) or not wait_gate(cohort, log):
            continue
        n_done = 0
        for r in plan["runs"]:
            rd = os.path.join(base, r["run_id"])
            if not claim(rd):
                continue
            if os.path.exists(os.path.join(rd, ".failed")):
                os.replace(os.path.join(rd, ".failed"), os.path.join(rd, f".failed.prev.{int(time.time())}"))
            try:
                arrays, meta = train_gauge_v11(r, data, max_steps=max_steps, log=log)
                meta["worker"] = wid
                meta["plan"] = os.path.basename(plan_path)
                save_run_v11(rd, arrays, meta)
                with open(os.path.join(rd, ".done"), "w") as f:
                    f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                n_done += 1
                log(f"done {r['run_id']} acc={meta.get('acc_bank', float('nan')):.4f} t={meta['t_total_s']:.1f}s")
            except Exception:
                tb = traceback.format_exc()
                with open(os.path.join(rd, ".failed"), "w") as f:
                    f.write(tb)
                log(f"FAILED {r['run_id']}\n{tb}")
            finally:
                try:
                    os.remove(os.path.join(rd, ".lock"))
                except FileNotFoundError:
                    pass
                torch.cuda.empty_cache()
        log(f"plan {cohort} finished by this worker: {n_done} runs done here")
    log("worker exit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["worker"])
    ap.add_argument("--plan", nargs="+", required=True)
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None, help="unit tests only (cohort plans refuse it)")
    a = ap.parse_args()
    if a.max_steps is not None:
        for p in a.plan:
            if json.load(open(p))["cohort"] in COHORT_PLANS:
                raise SystemExit("--max-steps is not allowed for pilot/main plans")
    worker(a.plan, a.wid, a.max_steps)
