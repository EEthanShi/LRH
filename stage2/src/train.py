"""Training of one Stage-2 run (SSL arm of S2.1 or gauge-net arm of S2.2) and the queue worker.

    python train.py worker --plan plans/<cohort>.json --wid 0
    python train.py bench  --kind ssl --N 50000 --steps 200

A worker walks the plan in order, claims a run with an O_EXCL lock file, trains it,
saves features on S to runs/<cohort>/<run_id>/feats.npz (+ meta.json) and writes
.done (or .failed with the traceback). Stale locks (dead pid) are reclaimed, so the
queue is resumable: restarting workers continues where it stopped.
"""
from __future__ import annotations

import argparse
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as L  # noqa: E402

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEV = torch.device("cuda")

SSL_DEFAULT = dict(budget=5_000_000, bs=512, lr=0.5, momentum=0.9, wd=5e-4, warmup=0.02,
                   tau=None, loss="infonce", mu=None)
GAUGE_DEFAULT = dict(budget=500_000, bs=1024, lr=11.5, momentum=0.85, wd=0.0153, bias_scaler=64.0,
                     label_smoothing=0.2, whiten_bias_frac=0.3, head_wd=True, small_init=None)


def run_key(r: dict) -> str:
    k = f"hrhv2-stage2|{r['exp']}|arm={r['arm']}|N={r['N']}|rep={r['rep']}"
    if r.get("corpus") is not None:
        k += f"|corpus={r['corpus']}"
    return k


def data_key(r: dict) -> str:
    if r.get("corpus") is not None:  # shared corpus: one draw per corpus, shared by its runs
        return f"hrhv2-stage2|{r['exp']}|arm={r['arm']}|N={r['N']}|corpus={r['corpus']}|data"
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


# ----------------------------------------------------------------------------- SSL
def ssl_loss_fn(cfg):
    if cfg["loss"] == "infonce":
        tau = float(cfg["tau"])
        return lambda z1, z2: L.infonce_loss(z1, z2, tau)
    if cfg["loss"] == "spectral":
        return lambda z1, z2: L.spectral_loss(z1, z2, cfg.get("mu"))
    raise ValueError(cfg["loss"])


def train_ssl(r: dict, model: L.SSLNet, data: L.C5M, max_steps=None, log=print):
    cfg = dict(SSL_DEFAULT)
    cfg.update(r.get("cfg", {}))
    key = run_key(r)
    t0 = time.time()
    ii = data.draw_indices(int(r["N"]), data_key(r))
    Xn, _ = data.gather(ii)
    Xg = L.to_gpu_u8(Xn, DEV)
    del Xn
    t_data = time.time() - t0
    L.reinit(model, key + "|init")
    model.to(DEV).to(memory_format=torch.channels_last)
    model.train()
    # whitening from 5000 unaugmented images of the run's own subset
    g_ord = L.torch_gen(key + "|order", DEV)
    wsel = torch.randperm(Xg.shape[0], device=DEV, generator=g_ord)[:5000]
    model.backbone.init_whitening(L.normalize(L.u8_to_float(Xg[wsel])))
    S_u8 = L.to_gpu_u8(data.S_X, DEV)
    init_out = L.forward_init_stats(model, S_u8)
    decay, no_decay = [], []
    for n_, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        (decay if p_.ndim >= 2 else no_decay).append(p_)
    opt = torch.optim.SGD([dict(params=decay, weight_decay=cfg["wd"]), dict(params=no_decay, weight_decay=0.0)],
                          lr=cfg["lr"], momentum=cfg["momentum"], nesterov=True)
    bs = int(cfg["bs"])
    steps = int(cfg["budget"]) // bs
    if max_steps is not None:
        steps = min(steps, max_steps)
    wu = max(1, int(cfg["warmup"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / wu if s < wu else 0.5 * (1 + math.cos(math.pi * (s - wu) / max(1, steps - wu))))
    lossf = ssl_loss_fn(cfg)
    g_aug = L.torch_gen(key + "|aug", DEV)
    losses = []
    acc_loss = torch.zeros((), device=DEV)
    diverged = False
    t1 = time.time()
    for s, idx in enumerate(stream_batches(Xg.shape[0], bs, steps, g_ord)):
        xb = L.u8_to_float(Xg[idx])
        v = torch.cat([L.ssl_augment(xb, g_aug), L.ssl_augment(xb, g_aug)])
        v = L.normalize(v).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, z = model(v)
        loss = lossf(z[:bs], z[bs:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        acc_loss += loss.detach()
        if (s + 1) % 100 == 0 or s + 1 == steps:
            n_ = 100 if (s + 1) % 100 == 0 else (s + 1) % 100
            val = float(acc_loss) / n_
            acc_loss.zero_()
            losses.append((s + 1, val))
            if not math.isfinite(val):
                diverged = True
                log(f"[{r['run_id']}] diverged at step {s + 1}")
                break
    torch.cuda.synchronize()
    t_train = time.time() - t1
    # evaluation
    h, z = L.forward_eval(model, S_u8)
    bank_u8 = L.to_gpu_u8(data.bank_X, DEV)
    hb, zb = L.forward_eval(model, bank_u8)
    by = torch.from_numpy(data.bank_Y)
    sy = torch.from_numpy(data.S_Y)
    finite = bool(torch.isfinite(h).all() and torch.isfinite(z).all())
    knn_h = L.knn_acc(hb, by, h, sy) if finite else float("nan")
    knn_z = L.knn_acc(zb, by, z, sy) if finite else float("nan")
    # held-out vs train SSL loss (eval-mode BN, fixed augmentation key)
    ev = {}
    if finite:
        model.eval()
        with torch.no_grad():
            g_tr = L.torch_gen(key + "|evalsel", DEV)
            tr_sel = torch.randperm(Xg.shape[0], device=DEV, generator=g_tr)[:4096]
            for name, X in (("train", Xg[tr_sel]), ("hold", L.to_gpu_u8(data.hold_X, DEV))):
                g_e = L.torch_gen("hrhv2-stage2|evalaug", DEV)
                tot = 0.0
                nb = 0
                for i in range(0, X.shape[0], bs):
                    xb = L.u8_to_float(X[i:i + bs])
                    v = torch.cat([L.ssl_augment(xb, g_e), L.ssl_augment(xb, g_e)])
                    v = L.normalize(v).contiguous(memory_format=torch.channels_last)
                    _, zz = model(v)
                    tot += float(lossf(zz[:xb.shape[0]], zz[xb.shape[0]:]))
                    nb += 1
                ev[f"evalloss_{name}"] = tot / nb
        model.train()
        ev["gen_gap"] = ev["evalloss_hold"] - ev["evalloss_train"]
    zn = z.norm(dim=1).double()
    zstats = dict(z_norm_med=float(zn.median()), z_norm_q10=float(zn.quantile(0.1)), z_norm_q90=float(zn.quantile(0.9)))
    if cfg.get("mu") is not None:
        zstats["z_frac_beyond_ball"] = float((zn >= math.sqrt(cfg["mu"])).double().mean())
    meta = dict(r)
    meta.update(dict(key=key, data_key=data_key(r), cfg=cfg, steps=steps, diverged=diverged, finite=finite,
                     knn_acc_h=knn_h, knn_acc_z=knn_z, loss_curve=losses, **zstats,
                     final_train_loss=losses[-1][1] if losses else float("nan"),
                     t_data_s=t_data, t_train_s=t_train, t_total_s=time.time() - t0,
                     subset_sha256=__import__("hashlib").sha256(ii.tobytes()).hexdigest(),
                     host=socket.gethostname(), torch=torch.__version__, **ev))
    arrays = dict(h=h.numpy(), z=z.numpy(), h_init=init_out[0].numpy(), z_init=init_out[1].numpy())
    del Xg, S_u8, bank_u8
    return arrays, meta


# ----------------------------------------------------------------------------- gauge net
def train_gauge(r: dict, model: L.GaugeNet, data: L.C5M, max_steps=None, log=print):
    cfg = dict(GAUGE_DEFAULT)
    cfg.update(r.get("cfg", {}))
    key = run_key(r)
    t0 = time.time()
    ii = data.draw_indices(int(r["N"]), data_key(r))
    Xn, Yn = data.gather(ii)
    Xg = L.to_gpu_u8(Xn, DEV)
    Yg = torch.from_numpy(Yn).to(DEV)
    del Xn
    t_data = time.time() - t0
    model.small_init = cfg["small_init"]
    L.reinit(model, key + "|init")
    model.to(DEV).to(memory_format=torch.channels_last)
    model.train()
    g_ord = L.torch_gen(key + "|order", DEV)
    wsel = torch.randperm(Xg.shape[0], device=DEV, generator=g_ord)[:5000]
    model.backbone.init_whitening(L.normalize(L.u8_to_float(Xg[wsel])))
    S_u8 = L.to_gpu_u8(data.S_X, DEV)
    init_out = L.forward_init_stats(model, S_u8)
    imb0 = model.imbalance()
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
              dict(params=other, lr=lr, weight_decay=wd / lr),
              dict(params=head, lr=lr, weight_decay=(wd / lr) if cfg["head_wd"] else 0.0)]
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
    losses, imb_curve = [], []
    acc_loss = torch.zeros((), device=DEV)
    diverged = False
    t1 = time.time()
    for s, idx in enumerate(stream_batches(Xg.shape[0], bs, steps, g_ord)):
        if s == wb_steps:
            model.backbone.whiten.bias.requires_grad = False
        xb = L.flip_translate(L.u8_to_float(Xg[idx]), g_aug, pad=2)
        xb = L.normalize(xb).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, _, logits = model(xb)
        loss = lossf(logits.float(), Yg[idx]).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        acc_loss += loss.detach() / bs
        if (s + 1) % 50 == 0 or s + 1 == steps:
            n_ = 50 if (s + 1) % 50 == 0 else (s + 1) % 50
            val = float(acc_loss) / n_
            acc_loss.zero_()
            losses.append((s + 1, val))
            imb_curve.append((s + 1, model.imbalance()["imb_rel"]))
            if not math.isfinite(val):
                diverged = True
                log(f"[{r['run_id']}] diverged at step {s + 1}")
                break
    model.backbone.whiten.bias.requires_grad = True
    torch.cuda.synchronize()
    t_train = time.time() - t1
    h, u, y, lg = L.forward_eval(model, S_u8)
    bank_u8 = L.to_gpu_u8(data.bank_X, DEV)
    _, _, _, lgb = L.forward_eval(model, bank_u8)
    sy = torch.from_numpy(data.S_Y)
    by = torch.from_numpy(data.bank_Y)
    acc_S = float((lg.argmax(1) == sy).double().mean())
    acc_bank = float((lgb.argmax(1) == by).double().mean())
    # train vs held-out CE (eval mode, no aug, no smoothing)
    with torch.no_grad():
        g_tr = L.torch_gen(key + "|evalsel", DEV)
        tr_sel = torch.randperm(Xg.shape[0], device=DEV, generator=g_tr)[:4096]
        _, _, _, lgt = L.forward_eval(model, Xg[tr_sel])
        ce_tr = float(F.cross_entropy(lgt, Yg[tr_sel].cpu()))
        _, _, _, lgh = L.forward_eval(model, L.to_gpu_u8(data.hold_X, DEV))
        ce_ho = float(F.cross_entropy(lgh, torch.from_numpy(data.hold_Y)))
        acc_tr = float((lgt.argmax(1) == Yg[tr_sel].cpu()).double().mean())
    imb = model.imbalance()
    finite = bool(torch.isfinite(h).all() and torch.isfinite(u).all() and torch.isfinite(y).all())
    meta = dict(r)
    meta.update(dict(key=key, data_key=data_key(r), cfg=cfg, steps=steps, diverged=diverged, finite=finite,
                     acc_S=acc_S, acc_bank=acc_bank, acc_train=acc_tr, ce_train=ce_tr, ce_hold=ce_ho,
                     gen_gap=ce_ho - ce_tr, loss_curve=losses, imb_curve=imb_curve, imb_init=imb0, imb_final=imb,
                     final_train_loss=losses[-1][1] if losses else float("nan"),
                     t_data_s=t_data, t_train_s=t_train, t_total_s=time.time() - t0,
                     subset_sha256=__import__("hashlib").sha256(ii.tobytes()).hexdigest(),
                     host=socket.gethostname(), torch=torch.__version__))
    arrays = dict(h=h.numpy(), u=u.numpy(), y=y.numpy(), h_init=init_out[0].numpy(), u_init=init_out[1].numpy(),
                  y_init=init_out[2].numpy(), P=model.P.weight.detach().float().cpu().numpy(),
                  W=model.W.weight.detach().float().cpu().numpy())
    del Xg, Yg, S_u8, bank_u8
    return arrays, meta


# ----------------------------------------------------------------------------- worker
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


GATED = {"S2_1_main": "GATE_S2_1_main", "S2_2_main": "GATE_S2_2_main"}


def wait_gate(cohort: str, log) -> bool:
    """Main cohorts wait for the automatic preregistered gate decision (src/gates.py)."""
    if cohort not in GATED:
        return True
    base = os.path.join(L.ROOT, "runs", GATED[cohort])
    t0 = time.time()
    while True:
        if os.path.exists(base + ".go"):
            return True
        if os.path.exists(base + ".nogo"):
            log(f"gate {GATED[cohort]} = NOGO: skipping cohort {cohort}")
            return False
        if time.time() - t0 > 6 * 3600:
            log(f"gate {GATED[cohort]} undecided after 6 h: skipping cohort {cohort}")
            return False
        time.sleep(30)


def worker(plan_paths: list, wid: int):
    data = L.C5M()
    models = {}
    log = lambda s: print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [w{wid}] {s}", flush=True)
    for plan_path in plan_paths:
        with open(plan_path) as f:
            plan = json.load(f)
        cohort = plan["cohort"]
        base = os.path.join(L.ROOT, "runs", cohort)
        log(f"plan {plan_path} n_runs={len(plan['runs'])} pid={os.getpid()}")
        if not wait_gate(cohort, log):
            continue
        n_done = 0
        for r in plan["runs"]:
            rd = os.path.join(base, r["run_id"])
            if not claim(rd):
                continue
            if os.path.exists(os.path.join(rd, ".failed")):
                os.replace(os.path.join(rd, ".failed"), os.path.join(rd, f".failed.prev.{int(time.time())}"))
            try:
                kind = r["kind"]
                if kind not in models:
                    models[kind] = L.SSLNet() if kind == "ssl" else L.GaugeNet()
                fn = train_ssl if kind == "ssl" else train_gauge
                arrays, meta = fn(r, models[kind], data, log=log)
                meta["worker"] = wid
                meta["plan"] = os.path.basename(plan_path)
                L.save_run(rd, arrays, meta)
                with open(os.path.join(rd, ".done"), "w") as f:
                    f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                n_done += 1
                extra = (f"knn_h={meta.get('knn_acc_h', float('nan')):.4f}" if kind == "ssl"
                         else f"acc={meta.get('acc_bank', float('nan')):.4f} imb={meta['imb_final']['imb_rel']:.3f}")
                log(f"done {r['run_id']} {extra} loss={meta['final_train_loss']:.4f} t={meta['t_total_s']:.1f}s")
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


def bench(kind: str, N: int, steps: int, cfg: dict):
    data = L.C5M()
    r = dict(run_id="bench", kind=kind, exp="BENCH", arm="bench", N=N, rep=0, cfg=cfg)
    model = L.SSLNet() if kind == "ssl" else L.GaugeNet()
    t = time.time()
    arrays, meta = (train_ssl if kind == "ssl" else train_gauge)(r, model, data, max_steps=steps)
    meta.pop("loss_curve", None)
    meta.pop("imb_curve", None)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("cfg",)}, default=str))
    print("peak_mem_GB", torch.cuda.max_memory_allocated() / 1e9, "wall", time.time() - t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["worker", "bench"])
    ap.add_argument("--plan", nargs="+")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--kind", default="ssl")
    ap.add_argument("--N", type=int, default=50000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()
    if a.mode == "worker":
        worker(a.plan, a.wid)
    else:
        bench(a.kind, a.N, a.steps, json.loads(a.cfg))
