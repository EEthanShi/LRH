"""HRH v2 Stage 2 library.

CIFAR-5m access, GPU augmentation, airbench94-style backbone, SSL objectives
(InfoNCE with cosine critic; spectral contrastive loss of HaoChen et al. 2021),
the bias-free gauge head (P: 256->9, W: 9x9, fixed orthonormal E: 10x9 spanning 1-perp),
training loops and feature extraction on the evaluation set S.

Seeds: every random draw uses a generator seeded by sha256(<string key>)
(first 8 bytes, little endian), keys of the form
    hrhv2-stage2|<exp>|arm=<arm>|N=<N>|rep=<r>[|corpus=<c>]|<purpose>
with purpose in {data, init, order, aug}. Runs are seed-deterministic; GPU
kernels (cuDNN autotuning, atomics) are not bitwise reproducible.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
DATA = os.path.join(ROOT, "data")

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


# ----------------------------------------------------------------------------- seeds
def seed_from_key(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def rng_from_key(key: str) -> np.random.Generator:
    return np.random.default_rng(seed_from_key(key))


def torch_gen(key: str, device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed_from_key(key) & ((1 << 63) - 1))
    return g


# ----------------------------------------------------------------------------- data
class C5M:
    """Memory-mapped CIFAR-5m train pool (parts 0-4) + fixed held-out arrays."""

    def __init__(self):
        with open(os.path.join(DATA, "c5m_index.json")) as f:
            self.idx = json.load(f)
        self.X, self.Y = [], []
        for p in self.idx["train_parts"]:
            q = self.idx["parts"][p]
            self.X.append(np.memmap(q["path"], dtype=np.uint8, mode="r", offset=q["X"]["offset"],
                                    shape=tuple(q["X"]["shape"])))
            self.Y.append(np.memmap(q["path"], dtype=np.dtype(q["Y"]["dtype"]), mode="r",
                                    offset=q["Y"]["offset"], shape=tuple(q["Y"]["shape"])))
        self.sizes = np.array(self.idx["train_sizes"], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.total = int(self.idx["train_total"])
        self.S_X = np.load(os.path.join(DATA, "S_X.npy"))
        self.S_Y = np.load(os.path.join(DATA, "S_Y.npy"))
        self.bank_X = np.load(os.path.join(DATA, "bank_X.npy"))
        self.bank_Y = np.load(os.path.join(DATA, "bank_Y.npy"))
        self.hold_X = np.load(os.path.join(DATA, "hold_X.npy"))
        self.hold_Y = np.load(os.path.join(DATA, "hold_Y.npy"))

    def draw_indices(self, N: int, key: str) -> np.ndarray:
        """N distinct global train-pool indices (sorted), from sha256(key)."""
        rng = rng_from_key(key)
        ii = rng.choice(self.total, size=N, replace=False)
        ii.sort()
        return ii

    def gather(self, ii: np.ndarray):
        Xs, Ys = [], []
        for p, (s, n) in enumerate(zip(self.starts, self.sizes)):
            lo, hi = np.searchsorted(ii, [s, s + n])
            loc = ii[lo:hi] - s
            if loc.size:
                Xs.append(np.asarray(self.X[p][loc]))
                Ys.append(np.asarray(self.Y[p][loc]).astype(np.int64))
        return np.concatenate(Xs), np.concatenate(Ys)


def to_gpu_u8(X: np.ndarray, device) -> torch.Tensor:
    """(n,32,32,3) uint8 numpy -> (n,3,32,32) uint8 cuda tensor."""
    t = torch.from_numpy(np.ascontiguousarray(X)).to(device, non_blocking=False)
    return t.permute(0, 3, 1, 2).contiguous()


def normalize(xf: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(MEAN, device=xf.device, dtype=xf.dtype).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=xf.device, dtype=xf.dtype).view(1, 3, 1, 1)
    return (xf - m) / s


def u8_to_float(xb: torch.Tensor) -> torch.Tensor:
    return xb.float().div_(255.0)


# ----------------------------------------------------------------------------- augmentation
def flip_translate(x: torch.Tensor, gen: torch.Generator, pad: int = 2) -> torch.Tensor:
    """airbench augmentation: random horizontal flip + random translate (reflect pad)."""
    B, C, H, W = x.shape
    dev = x.device
    fl = torch.rand(B, device=dev, generator=gen) < 0.5
    x = torch.where(fl.view(B, 1, 1, 1), x.flip(3), x)
    if pad > 0:
        xp = F.pad(x, (pad,) * 4, mode="reflect")
        dy = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=gen)
        dx = torch.randint(0, 2 * pad + 1, (B,), device=dev, generator=gen)
        iy = dy[:, None] + torch.arange(H, device=dev)[None, :]
        ix = dx[:, None] + torch.arange(W, device=dev)[None, :]
        xp = torch.gather(xp, 2, iy[:, None, :, None].expand(B, C, H, W + 2 * pad))
        x = torch.gather(xp, 3, ix[:, None, None, :].expand(B, C, H, W))
    return x


_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
_YIQ2RGB = torch.linalg.inv(_RGB2YIQ)


def _gray(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def ssl_augment(x: torch.Tensor, gen: torch.Generator, scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3),
                jitter=(0.4, 0.4, 0.4, 0.1), p_jitter=0.8, p_gray=0.2) -> torch.Tensor:
    """SimCLR-style CIFAR augmentation on float images in [0,1] (B,3,32,32), all on GPU.

    RandomResizedCrop(32, scale, ratio) + horizontal flip via one bilinear affine
    resampling; ColorJitter(b,c,s,h) with prob p_jitter in fixed order b->c->s->h
    (hue = rotation of the chroma plane in YIQ); grayscale with prob p_gray.
    """
    B = x.shape[0]
    dev = x.device
    u = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi, generator=gen)
    area = u(scale[0], scale[1])
    ar = torch.exp(u(math.log(ratio[0]), math.log(ratio[1])))
    w = torch.sqrt(area * ar).clamp(max=1.0)
    h = torch.sqrt(area / ar).clamp(max=1.0)
    cx = u(-1.0, 1.0) * (1 - w)
    cy = u(-1.0, 1.0) * (1 - h)
    flip = torch.where(torch.rand(B, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = w * flip
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = h
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    # colour jitter
    bj, cj, sj, hj = jitter
    xj = x * u(1 - bj, 1 + bj).view(B, 1, 1, 1)
    xj = xj.clamp(0, 1)
    m = _gray(xj).mean(dim=(1, 2, 3), keepdim=True)
    xj = ((xj - m) * u(1 - cj, 1 + cj).view(B, 1, 1, 1) + m).clamp(0, 1)
    g = _gray(xj)
    xj = ((xj - g) * u(1 - sj, 1 + sj).view(B, 1, 1, 1) + g).clamp(0, 1)
    ang = u(-hj, hj) * 2 * math.pi
    c, s = torch.cos(ang), torch.sin(ang)
    T = _RGB2YIQ.to(dev)
    Ti = _YIQ2RGB.to(dev)
    yiq = torch.einsum("ij,bjhw->bihw", T, xj)
    i2 = c.view(B, 1, 1) * yiq[:, 1] - s.view(B, 1, 1) * yiq[:, 2]
    q2 = s.view(B, 1, 1) * yiq[:, 1] + c.view(B, 1, 1) * yiq[:, 2]
    yiq = torch.stack([yiq[:, 0], i2, q2], dim=1)
    xj = torch.einsum("ij,bjhw->bihw", Ti, yiq).clamp(0, 1)
    pj = (torch.rand(B, device=dev, generator=gen) < p_jitter).view(B, 1, 1, 1)
    x = torch.where(pj, xj, x)
    pg = (torch.rand(B, device=dev, generator=gen) < p_gray).view(B, 1, 1, 1)
    x = torch.where(pg, _gray(x).expand(-1, 3, -1, -1), x)
    return x


# ----------------------------------------------------------------------------- nets (airbench94-style)
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, c, momentum=0.6, eps=1e-12):
        super().__init__(c, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, cin, cout, bn_mom=0.6):
        super().__init__()
        self.conv1 = Conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(cout, bn_mom)
        self.conv2 = Conv(cout, cout)
        self.norm2 = BatchNorm(cout, bn_mom)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        return self.activ(self.norm2(self.conv2(x)))


class Backbone(nn.Module):
    """airbench94 trunk: 2x2 whitening conv (24 ch) -> 3 ConvGroups (64,256,256) -> maxpool -> h (256)."""

    def __init__(self, widths=(64, 256, 256), bn_mom=0.6):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.act = nn.GELU()
        self.g1 = ConvGroup(24, widths[0], bn_mom)
        self.g2 = ConvGroup(widths[0], widths[1], bn_mom)
        self.g3 = ConvGroup(widths[1], widths[2], bn_mom)
        self.pool = nn.MaxPool2d(3)
        self.out_dim = widths[2]

    def forward(self, x):
        x = self.act(self.whiten(x))
        x = self.g3(self.g2(self.g1(x)))
        return self.pool(x).flatten(1)

    @torch.no_grad()
    def init_whitening(self, x_norm: torch.Tensor, eps=5e-4):
        """airbench whitening init from 2x2 patches of (normalized, unaugmented) images."""
        x = x_norm.float()
        c = x.shape[1]
        patches = x.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, c, 2, 2)
        pf = patches.reshape(patches.shape[0], -1)
        cov = (pf.T @ pf) / pf.shape[0]
        evals, evecs = torch.linalg.eigh(cov, UPLO="U")
        evals = evals.flip(0).view(-1, 1, 1, 1)
        evecs = evecs.T.reshape(c * 4, c, 2, 2).flip(0)
        scaled = evecs / torch.sqrt(evals + eps)
        self.whiten.weight.data[:] = torch.cat((scaled, -scaled)).to(self.whiten.weight.dtype)
        self.whiten.bias.data.zero_()


class SSLNet(nn.Module):
    def __init__(self, proj=(256, 512, 128)):
        super().__init__()
        self.backbone = Backbone()
        self.proj = nn.Sequential(nn.Linear(proj[0], proj[1], bias=False), nn.BatchNorm1d(proj[1]),
                                  nn.ReLU(inplace=True), nn.Linear(proj[1], proj[2], bias=True))

    def forward(self, x):
        h = self.backbone(x)
        return h, self.proj(h)


def ortho_E(n_classes=10) -> torch.Tensor:
    """Fixed orthonormal basis (10 x 9) of the sum-zero subspace 1-perp (Helmert)."""
    k = n_classes
    E = np.zeros((k, k - 1))
    for j in range(1, k):
        E[:j, j - 1] = 1.0
        E[j, j - 1] = -j
        E[:, j - 1] /= np.linalg.norm(E[:, j - 1])
    return torch.tensor(E, dtype=torch.float32)


class GaugeNet(nn.Module):
    """airbench94 trunk + bias-free head: u = P h (9), y = W u (9), logits = (E y) * (1/9)."""

    def __init__(self, small_init: float | None = None, scale=1 / 9):
        super().__init__()
        self.backbone = Backbone()
        self.P = nn.Linear(256, 9, bias=False)
        self.W = nn.Linear(9, 9, bias=False)
        self.register_buffer("E", ortho_E(10))
        self.scale = scale
        self.small_init = small_init

    def reset_head(self):
        self.P.reset_parameters()
        self.W.reset_parameters()
        if self.small_init is not None:
            self.P.weight.data.mul_(self.small_init)
            self.W.weight.data.mul_(self.small_init)

    def forward(self, x):
        h = self.backbone(x)
        u = self.P(h)
        y = self.W(u)
        return h, u, y, (y @ self.E.T) * self.scale

    @torch.no_grad()
    def imbalance(self) -> dict:
        P = self.P.weight.double()  # (9,256)
        W = self.W.weight.double()  # (9,9)
        A = W.T @ W
        Bm = P @ P.T
        C = A - Bm
        return {"imb_rel": float(C.norm() / (A.norm() + Bm.norm())),
                "imb_abs": float(C.norm()), "WtW_norm": float(A.norm()), "PPt_norm": float(Bm.norm()),
                "W_cond": float(torch.linalg.cond(W)), "PPt_cond": float(torch.linalg.cond(Bm))}


def reinit(model: nn.Module, seed_key: str):
    """Re-initialise every parameter / buffer deterministically from sha256(seed_key)."""
    torch.manual_seed(seed_from_key(seed_key) & ((1 << 63) - 1))
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.reset_parameters()
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.reset_running_stats()
    if isinstance(model, GaugeNet):
        model.reset_head()
    for mod in model.modules():  # airbench: BN weights frozen at 1
        if isinstance(mod, BatchNorm):
            mod.weight.requires_grad = False
    model.backbone.whiten.weight.requires_grad = False


# ----------------------------------------------------------------------------- SSL losses
def infonce_loss(z1, z2, tau):
    """NT-Xent (SimCLR) with cosine critic; 2B x 2B similarity, positives = other view."""
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    n = z1.shape[0]
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def spectral_loss(z1, z2, mu=None):
    """Spectral contrastive loss (HaoChen et al. 2021):
        -2 E_{pos} <f(x), f(x+)> + E_{x != x'} <f(x), f(x')>^2
    estimated on a batch with both view orders; unconstrained f (mu=None) or HaoChen's
    radial projection onto the ball of radius sqrt(mu)."""
    z1 = z1.float()
    z2 = z2.float()
    if mu is not None:
        r = math.sqrt(mu)
        z1 = z1 * torch.clamp(r / (z1.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
        z2 = z2 * torch.clamp(r / (z2.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
    n = z1.shape[0]
    pos = (z1 * z2).sum(1).mean()
    G = z1 @ z2.T
    off = (G.pow(2).sum() - G.diagonal().pow(2).sum()) / (n * (n - 1))
    return -2.0 * pos + off


# ----------------------------------------------------------------------------- evaluation helpers
@torch.no_grad()
def forward_eval(model, X_u8: torch.Tensor, bs=1024):
    """Eval-mode fp32 forward on unaugmented normalized images; returns list of output tensors (cpu fp32)."""
    was = model.training
    model.eval()
    outs = None
    for i in range(0, X_u8.shape[0], bs):
        xb = normalize(u8_to_float(X_u8[i:i + bs])).contiguous(memory_format=torch.channels_last)
        o = model(xb)
        o = o if isinstance(o, tuple) else (o,)
        if outs is None:
            outs = [[] for _ in o]
        for j, t in enumerate(o):
            outs[j].append(t.float().cpu())
    model.train(was)
    return [torch.cat(t) for t in outs]


@torch.no_grad()
def forward_init_stats(model, X_u8: torch.Tensor):
    """Features at initialisation with BN statistics taken from the batch itself (train-mode BN,
    no running-stat update side effects kept)."""
    import copy
    m2 = copy.deepcopy(model)
    m2.train()
    xb = normalize(u8_to_float(X_u8)).contiguous(memory_format=torch.channels_last)
    o = m2(xb)
    o = o if isinstance(o, tuple) else (o,)
    del m2
    return [t.float().cpu() for t in o]


@torch.no_grad()
def knn_acc(bank_f, bank_y, q_f, q_y, k=20, T=0.07, n_cls=10):
    bf = F.normalize(bank_f.double(), dim=1)
    qf = F.normalize(q_f.double(), dim=1)
    sim = qf @ bf.T
    v, ii = sim.topk(k, dim=1)
    w = torch.exp(v / T)
    lab = bank_y[ii]
    votes = torch.zeros(q_f.shape[0], n_cls, dtype=torch.float64)
    votes.scatter_add_(1, lab, w)
    pred = votes.argmax(1)
    return float((pred == q_y).double().mean())


def save_run(out_dir: str, arrays: dict, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "feats.tmp.npz")
    np.savez(tmp, **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})
    os.replace(tmp, os.path.join(out_dir, "feats.npz"))
    with open(os.path.join(out_dir, "meta.json.tmp"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    os.replace(os.path.join(out_dir, "meta.json.tmp"), os.path.join(out_dir, "meta.json"))
