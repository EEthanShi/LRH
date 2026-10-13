"""S2.2 v1.1 gauge-net head: model, inits, closed-form float64 head gradients, head integrators,
sharpness probe (PREREG_S2_2_v1.1.md sections 2.1-2.3 and 3).

Model: the v1.0 airbench94 trunk (src_v11/s2lib.py, a verbatim copy of v1.0 src/s2lib.py) -> h in R^256 ->
u = P h in R^9 -> y = W u in R^9 -> logits = (E y) / 9, E the fixed 10 x 9 Helmert basis.
  * head_bf16 = True : the exact v1.0 path (fp32 head parameters, head computed inside bf16 autocast).
  * head_bf16 = False: P, W are float64 parameters; the head is computed in float64 from h cast to float64
    with autocast disabled (TF32 never applies to float64 matmuls); autograd casts the gradient that flows
    back into the trunk to the trunk dtype.
Inits (the head is always first drawn by the v1.0 reinit, so the RNG stream of the trunk is unchanged):
  * def  : PyTorch default (v1.0).
  * small: def x 0.1 (v1.0 small_init = 0.1).
  * bal  : P0 = def; W0 = Q (P0 P0^T)^{1/2}, Q Haar-orthogonal from sha256(key|init-bal), float64; C0 = 0.
Head treatments (head not in the optimizer except for "mom"):
  * mom        : Nesterov 0.85 group, as v1.0.
  * euler      : K Euler substeps of size tau_t/K on batch b's summed loss, h fixed at its forward value.
  * heun       : K Heun (explicit trapezoid) substeps.
  * heun_adapt : adaptive Heun-Euler (embedded pair, local extrapolation) with a per-batch relative tolerance.
  * midpoint   : K implicit-midpoint substeps, fixed-point solve to fp_tol; a substep whose fixed point does not
                 converge in fp_max_iter iterations is split in two halves (recursively, up to depth 12), so the
                 rule stays the implicit midpoint rule and conserves C up to the solver tolerance.
tau_t := lr_t / (1 - momentum) with lr_t the trunk group's current learning rate (= lr_mult(t) * 11.5 / kilo).
Sign of C: C := P P^T - W^T W (paper).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import s2lib as L  # verbatim copy of v1.0 src/s2lib.py (same directory)

LS = 0.2          # label smoothing (v1.0)
SCALE = 1.0 / 9   # logits = (E y) * SCALE (v1.0)
NCLS = 10


class GaugeNetV11(L.GaugeNet):
    """v1.0 GaugeNet (subclass: same modules, same order, same reset_head, so v1.0 s2lib.reinit applies
    unchanged) with an optional float64 head."""

    def __init__(self, head_bf16: bool = False):
        super().__init__(small_init=None, scale=SCALE)
        self.register_buffer("E64", L.ortho_E(10).double())   # float32 Helmert values cast to float64
        self.head_bf16 = head_bf16

    def head(self, h):
        if self.head_bf16:  # exact v1.0 expression (inside the caller's bf16 autocast)
            u = self.P(h)
            y = self.W(u)
            return u, y, (y @ self.E.T) * self.scale
        with torch.autocast("cuda", enabled=False):
            hd = h.double()
            u = hd @ self.P.weight.T
            y = u @ self.W.weight.T
            return u, y, (y @ self.E64.T) * self.scale

    def forward(self, x):
        h = self.backbone(x)
        u, y, lg = self.head(h)
        return h, u, y, lg


def reinit_v11(model: GaugeNetV11, key: str, init: str):
    """v1.0 reinit (identical RNG consumption), then the v1.1 init of the head; returns (P0, W0) float64 numpy.
    Must be called on a freshly constructed CPU model (the v1.0 cached-model GPU path is not reproduced)."""
    if init not in ("def", "small", "bal"):
        raise ValueError(init)
    model.small_init = 0.1 if init == "small" else None
    L.reinit(model, key + "|init")  # v1.0 s2lib.reinit, unchanged
    P0 = model.P.weight.detach().double().cpu().numpy().copy()
    W0 = model.W.weight.detach().double().cpu().numpy().copy()
    if init == "bal":
        rng = L.rng_from_key(key + "|init-bal")
        Z = rng.standard_normal((9, 9))
        Q, R = np.linalg.qr(Z)
        Q = Q * np.sign(np.diag(R))[None, :]
        w, V = np.linalg.eigh(P0 @ P0.T)
        S = (V * np.sqrt(w)) @ V.T
        S = 0.5 * (S + S.T)
        W0 = Q @ S
    if not model.head_bf16:
        model.P.weight = nn.Parameter(torch.from_numpy(P0.copy()))
        model.W.weight = nn.Parameter(torch.from_numpy(W0.copy()))
    elif init == "bal":
        raise ValueError("bal init is defined for the float64 head only")
    return P0, W0


# ----------------------------------------------------------------------------- closed-form head math
def smooth_targets(yb: torch.Tensor, dtype=torch.float64):
    q = torch.full((yb.shape[0], NCLS), LS / NCLS, dtype=dtype, device=yb.device)
    q.scatter_(1, yb.view(-1, 1), 1.0 - LS + LS / NCLS)
    return q


class HeadProblem:
    """Summed label-smoothed CE of batch b as a function of (P, W) with h fixed (float64)."""

    def __init__(self, h: torch.Tensor, yb: torch.Tensor, E64: torch.Tensor):
        self.h = h.detach().double().contiguous()
        self.hT = self.h.T.contiguous()                      # (256, B): layout chosen for float64 GEMM speed
        self.q = smooth_targets(yb)
        self.ES = (E64 * SCALE).contiguous()                 # (10, 9)
        self.n_evals = 0

    def loss(self, P, W):
        z = (self.h @ P.T) @ W.T @ self.ES.T
        return float(-(self.q * torch.log_softmax(z, 1)).sum())

    def grads(self, P, W):
        """g_P = W^T Gy^T h, g_W = Gy^T u with Gy = dL/dy (sum reduction). Same algebra as
        g_P = (dy W)^T h, g_W = dy^T u; operand layouts chosen because float64 (9 x B)(B x 9) and (B x 256)(256 x 9)
        cuBLAS GEMMs are slow on this GPU (unit test U2 checks it against autograd to 1e-12)."""
        self.n_evals += 1
        u = (P @ self.hT).T                                  # (B, 9)
        z = (u @ W.T) @ self.ES.T                            # (B, 10)
        dz = torch.softmax(z, 1) - self.q
        dy = dz @ self.ES                                    # (B, 9)
        gW = (dy[:, :, None] * u[:, None, :]).sum(0)         # dy^T u
        gP = (self.hT @ (dy @ W)).T                          # (dy W)^T h
        return gP, gW


def C_t(P, W):
    return P @ P.T - W.T @ W


class NonFinite(Exception):
    pass


def _fin(x: float):
    if not math.isfinite(x):
        raise NonFinite()
    return x


@torch.no_grad()
def integrate(prob: HeadProblem, P: torch.Tensor, W: torch.Tensor, tau: float, cfg: dict, acc=None):
    """Wrapper of _integrate: a non-finite state stops the integration and is reported (the run diverged)."""
    n0 = prob.n_evals
    try:
        if cfg.get("graph", True) and P.is_cuda:
            return _graph_engine(prob, cfg).integrate(prob, P, W, tau, cfg, acc)
        return _integrate(prob, P, W, tau, cfg, acc)
    except NonFinite:
        return {"substeps": 0, "evals": prob.n_evals - n0, "fp_fail": 0, "rejected": 0, "nonfinite": True}


def _integrate(prob: HeadProblem, P: torch.Tensor, W: torch.Tensor, tau: float, cfg: dict, acc=None):
    """Advance (P, W) in place by the configured head integrator over time tau. Returns a dict with the
    substep / gradient-evaluation counts and flags. acc (9x9 float64) accumulates the exact Euler drift."""
    method = cfg["head_opt"]
    info = {"substeps": 0, "evals0": prob.n_evals, "fp_fail": 0, "rejected": 0}
    if method == "euler":
        K = int(cfg["K"])
        dt = tau / K
        for _ in range(K):
            gP, gW = prob.grads(P, W)
            if acc is not None:
                acc += dt * dt * (gP @ gP.T - gW.T @ gW)
            P -= dt * gP
            W -= dt * gW
        info["substeps"] = K
    elif method == "heun":
        K = int(cfg["K"])
        dt = tau / K
        for _ in range(K):
            g1P, g1W = prob.grads(P, W)
            g2P, g2W = prob.grads(P - dt * g1P, W - dt * g1W)
            P -= 0.5 * dt * (g1P + g2P)
            W -= 0.5 * dt * (g1W + g2W)
        info["substeps"] = K
    elif method == "heun_adapt":
        tol = float(cfg["tol"])
        t = 0.0
        dt = tau / max(1, int(cfg.get("_dt_hint", cfg.get("K", 16))))
        n_acc = 0
        while t < tau * (1 - 1e-12):
            dt = min(dt, tau - t)
            g1P, g1W = prob.grads(P, W)
            PeP, WeW = P - dt * g1P, W - dt * g1W
            g2P, g2W = prob.grads(PeP, WeW)
            dP = 0.5 * dt * (g2P - g1P)
            dW = 0.5 * dt * (g2W - g1W)
            nrm = _fin(math.sqrt(float((P * P).sum() + (W * W).sum())))
            err = math.sqrt(float((dP * dP).sum() + (dW * dW).sum())) / max(nrm, 1e-300)
            if not math.isfinite(err):          # trial step overflowed: reject and shrink
                info["rejected"] += 1
                dt *= 0.2
                if dt < tau * 1e-12:
                    raise NonFinite()
                continue
            if err <= tol or dt <= tau * 1e-7:
                P -= 0.5 * dt * (g1P + g2P)
                W -= 0.5 * dt * (g1W + g2W)
                t += dt
                n_acc += 1
            else:
                info["rejected"] += 1
            fac = 2.0 if err == 0 else min(2.0, max(0.2, 0.9 * math.sqrt(tol / err)))
            dt = dt * fac
        info["substeps"] = n_acc
        cfg["_dt_hint"] = max(1, int(round(tau / max(dt, 1e-300))))
    elif method == "midpoint":
        K = int(cfg["K"])
        fp_tol = float(cfg.get("fp_tol", 1e-12))
        fp_max = int(cfg.get("fp_max_iter", 30))

        def mid_step(P0, W0, dt, depth):
            g1P, g1W = prob.grads(P0, W0)
            P1, W1 = P0 - dt * g1P, W0 - dt * g1W              # explicit Euler predictor
            nrm = _fin(math.sqrt(float((P0 * P0).sum() + (W0 * W0).sum())))
            prev = float("inf")
            for _ in range(fp_max):
                gP, gW = prob.grads(0.5 * (P0 + P1), 0.5 * (W0 + W1))
                Pn, Wn = P0 - dt * gP, W0 - dt * gW
                res = math.sqrt(float(((Pn - P1) ** 2).sum() + ((Wn - W1) ** 2).sum())) / max(nrm, 1e-300)
                if not math.isfinite(res):       # fixed-point iteration overflowed: split the substep
                    break
                P1, W1 = Pn, Wn
                if res <= fp_tol:
                    return P1, W1, 1
                if res > prev * 0.999 and res > 1e3 * fp_tol:   # not contracting
                    break
                prev = res
            if depth >= 12:
                info["fp_fail"] += 1
                if not (math.isfinite(float(P1.abs().max())) and math.isfinite(float(W1.abs().max()))):
                    raise NonFinite()
                return P1, W1, 1
            Pa, Wa, na = mid_step(P0, W0, 0.5 * dt, depth + 1)
            Pb, Wb, nb = mid_step(Pa, Wa, 0.5 * dt, depth + 1)
            return Pb, Wb, na + nb

        n = 0
        dt = tau / K
        for _ in range(K):
            P1, W1, k = mid_step(P.clone(), W.clone(), dt, 0)
            P.copy_(P1)
            W.copy_(W1)
            n += k
        info["substeps"] = n
    else:
        raise ValueError(method)
    info["evals"] = prob.n_evals - info.pop("evals0")
    return info



# ----------------------------------------------------------------------------- CUDA-graph engine
# The same arithmetic as _integrate, recorded once per process as CUDA graphs on static buffers and replayed
# (removes the per-kernel launch cost of the many tiny float64 head kernels). Fixed-K Euler / Heun substeps are
# replayed in blocks of BLOCK substeps; implicit midpoint replays its fixed-point iteration in blocks of FP_BLOCK
# iterations and checks the residual after each block (so it may iterate up to FP_BLOCK - 1 times more than the
# eager path; the rule and its stopping tolerance are unchanged); adaptive Heun-Euler replays one trial substep
# and accepts / rejects on the host. The eager path (cfg graph = False) is the reference in unit test U8.
BLOCK = 16
FP_BLOCK = 4
_ENGINES = {}


class _Static:
    def __init__(self, B, ES):
        dev = ES.device
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
        self.h, self.hT, self.q = z(B, 256), z(256, B), z(B, NCLS)
        self.P, self.W, self.P0, self.W0, self.P1, self.W1 = z(9, 256), z(9, 9), z(9, 256), z(9, 9), z(9, 256), z(9, 9)
        self.Ph, self.Wh = z(9, 256), z(9, 9)
        self.acc, self.dt, self.res, self.err, self.nrm = z(9, 9), z(()), z(()), z(()), z(())
        self.prob = HeadProblem.__new__(HeadProblem)
        self.prob.h, self.prob.hT, self.prob.q = self.h, self.hT, self.q
        self.prob.ES, self.prob.n_evals = ES.clone().contiguous(), 0


class GraphEngine:
    def __init__(self, B, ES):
        self.s = _Static(B, ES)
        self.graphs = {}

    def _capture(self, name, body):
        if name in self.graphs:
            return self.graphs[name]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        self.graphs[name] = g
        return g

    # ---- recorded bodies (static buffers only)
    def _euler_body(self, n):
        s, g = self.s, self.s.prob.grads

        def body():
            for _ in range(n):
                gP, gW = g(s.P, s.W)
                s.acc.add_(s.dt * s.dt * (gP @ gP.T - gW.T @ gW))
                s.P.sub_(s.dt * gP)
                s.W.sub_(s.dt * gW)
        return body

    def _heun_body(self, n):
        s, g = self.s, self.s.prob.grads

        def body():
            for _ in range(n):
                g1P, g1W = g(s.P, s.W)
                g2P, g2W = g(s.P - s.dt * g1P, s.W - s.dt * g1W)
                s.P.sub_(0.5 * s.dt * (g1P + g2P))
                s.W.sub_(0.5 * s.dt * (g1W + g2W))
        return body

    def _mid_pred_body(self):
        s, g = self.s, self.s.prob.grads

        def body():
            gP, gW = g(s.P0, s.W0)
            s.P1.copy_(s.P0 - s.dt * gP)
            s.W1.copy_(s.W0 - s.dt * gW)
            s.nrm.copy_(torch.sqrt((s.P0 * s.P0).sum() + (s.W0 * s.W0).sum()))
        return body

    def _mid_fp_body(self, m):
        s, g = self.s, self.s.prob.grads

        def body():
            for _ in range(m):
                gP, gW = g(0.5 * (s.P0 + s.P1), 0.5 * (s.W0 + s.W1))
                Pn, Wn = s.P0 - s.dt * gP, s.W0 - s.dt * gW
                s.res.copy_(torch.sqrt(((Pn - s.P1) ** 2).sum() + ((Wn - s.W1) ** 2).sum()))
                s.P1.copy_(Pn)
                s.W1.copy_(Wn)
        return body

    def _adapt_body(self):
        s, g = self.s, self.s.prob.grads

        def body():
            g1P, g1W = g(s.P, s.W)
            g2P, g2W = g(s.P - s.dt * g1P, s.W - s.dt * g1W)
            dP, dW = 0.5 * s.dt * (g2P - g1P), 0.5 * s.dt * (g2W - g1W)
            s.nrm.copy_(torch.sqrt((s.P * s.P).sum() + (s.W * s.W).sum()))
            s.err.copy_(torch.sqrt((dP * dP).sum() + (dW * dW).sum()))
            s.Ph.copy_(s.P - 0.5 * s.dt * (g1P + g2P))
            s.Wh.copy_(s.W - 0.5 * s.dt * (g1W + g2W))
        return body

    # ---- integration
    def integrate(self, prob, P, W, tau, cfg, acc):
        s = self.s
        if not torch.equal(s.prob.ES, prob.ES):
            raise ValueError("graph engine built for a different E")
        s.h.copy_(prob.h)
        s.hT.copy_(prob.hT)
        s.q.copy_(prob.q)
        method = cfg["head_opt"]
        info = {"substeps": 0, "fp_fail": 0, "rejected": 0, "evals": 0, "graph": True}
        if method in ("euler", "heun"):
            K = int(cfg["K"])
            n = min(K, BLOCK)
            if K % n:
                raise ValueError("K must be a multiple of the graph block (or <= it)")
            body = self._euler_body(n) if method == "euler" else self._heun_body(n)
            gr = self._capture(f"{method}{n}", body)
            s.P.copy_(P)
            s.W.copy_(W)
            s.dt.fill_(tau / K)
            s.acc.zero_()
            for _ in range(K // n):
                gr.replay()
            P.copy_(s.P)
            W.copy_(s.W)
            if acc is not None and method == "euler":
                acc.add_(s.acc)
            info["substeps"] = K
            info["evals"] = K * (1 if method == "euler" else 2)
        elif method == "midpoint":
            K = int(cfg["K"])
            fp_tol = float(cfg.get("fp_tol", 1e-12))
            fp_max = int(cfg.get("fp_max_iter", 30))
            gpred = self._capture("mid_pred", self._mid_pred_body())
            gfp = self._capture(f"mid_fp{FP_BLOCK}", self._mid_fp_body(FP_BLOCK))

            def mid_step(P0, W0, dt, depth):
                s.P0.copy_(P0)
                s.W0.copy_(W0)
                s.dt.fill_(dt)
                gpred.replay()
                info["evals"] += 1
                nrm = _fin(float(s.nrm))
                prev = float("inf")
                for _ in range(max(1, fp_max // FP_BLOCK)):
                    gfp.replay()
                    info["evals"] += FP_BLOCK
                    res = float(s.res) / max(nrm, 1e-300)
                    if not math.isfinite(res):
                        break
                    if res <= fp_tol:
                        return s.P1.clone(), s.W1.clone(), 1
                    if res > prev * 0.999 and res > 1e3 * fp_tol:
                        break
                    prev = res
                P1, W1 = s.P1.clone(), s.W1.clone()
                if depth >= 12:
                    info["fp_fail"] += 1
                    if not (math.isfinite(float(P1.abs().max())) and math.isfinite(float(W1.abs().max()))):
                        raise NonFinite()
                    return P1, W1, 1
                Pa, Wa, na = mid_step(P0, W0, 0.5 * dt, depth + 1)
                Pb, Wb, nb = mid_step(Pa, Wa, 0.5 * dt, depth + 1)
                return Pb, Wb, na + nb

            n = 0
            dt = tau / K
            for _ in range(K):
                P1, W1, k = mid_step(P.clone(), W.clone(), dt, 0)
                P.copy_(P1)
                W.copy_(W1)
                n += k
            info["substeps"] = n
        elif method == "heun_adapt":
            tol = float(cfg["tol"])
            gtry = self._capture("adapt", self._adapt_body())
            s.P.copy_(P)
            s.W.copy_(W)
            t = 0.0
            dt = tau / max(1, int(cfg.get("_dt_hint", cfg.get("K", 16))))
            n_acc = 0
            while t < tau * (1 - 1e-12):
                dt = min(dt, tau - t)
                s.dt.fill_(dt)
                gtry.replay()
                info["evals"] += 2
                nrm = _fin(float(s.nrm))
                err = float(s.err) / max(nrm, 1e-300)
                if not math.isfinite(err):
                    info["rejected"] += 1
                    dt *= 0.2
                    if dt < tau * 1e-12:
                        raise NonFinite()
                    continue
                if err <= tol or dt <= tau * 1e-7:
                    s.P.copy_(s.Ph)
                    s.W.copy_(s.Wh)
                    t += dt
                    n_acc += 1
                else:
                    info["rejected"] += 1
                fac = 2.0 if err == 0 else min(2.0, max(0.2, 0.9 * math.sqrt(tol / err)))
                dt = dt * fac
            P.copy_(s.P)
            W.copy_(s.W)
            info["substeps"] = n_acc
            cfg["_dt_hint"] = max(1, int(round(tau / max(dt, 1e-300))))
        else:
            raise ValueError(method)
        prob.n_evals += info["evals"]
        return info


def _graph_engine(prob, cfg):
    key = (prob.h.shape[0], str(prob.h.device))
    if key not in _ENGINES:
        _ENGINES[key] = GraphEngine(prob.h.shape[0], prob.ES)
    return _ENGINES[key]


def hvp_lambda_max(prob: HeadProblem, P: torch.Tensor, W: torch.Tensor, iters: int = 50, key: str = "sharp"):
    """Largest-magnitude eigenvalue of the Hessian of the summed batch loss w.r.t. (P, W) (power iteration on
    autograd Hessian-vector products, float64). Returns (lambda, rayleigh_quotient_sign, iterations)."""
    Pv = P.detach().clone().requires_grad_(True)
    Wv = W.detach().clone().requires_grad_(True)
    z = (prob.h @ Pv.T) @ Wv.T @ prob.ES.T
    loss = -(prob.q * torch.log_softmax(z, 1)).sum()
    gP, gW = torch.autograd.grad(loss, (Pv, Wv), create_graph=True)
    gen = torch.Generator(device=P.device)
    gen.manual_seed(L.seed_from_key(key) & ((1 << 63) - 1))
    vP = torch.randn(P.shape, dtype=P.dtype, device=P.device, generator=gen)
    vW = torch.randn(W.shape, dtype=W.dtype, device=W.device, generator=gen)
    lam, it = 0.0, 0
    for it in range(1, iters + 1):
        nv = torch.sqrt((vP * vP).sum() + (vW * vW).sum())
        vP, vW = vP / nv, vW / nv
        hP, hW = torch.autograd.grad((gP * vP).sum() + (gW * vW).sum(), (Pv, Wv), retain_graph=True)
        new = float((hP * vP).sum() + (hW * vW).sum())
        vP, vW = hP.detach(), hW.detach()
        if it > 5 and abs(new - lam) <= 1e-4 * abs(new):
            lam = new
            break
        lam = new
    return lam, it
