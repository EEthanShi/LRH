"""N*, the grid and the [Model] Monte Carlo predictions (PREREG section 4.3).

Probe-level MC (S_ex8, S_ex16): Gaussian draws of the probe entries, A ~ N(theta*_S, Sig_A/N)
(thm:peer part 1, [Thm+CLT]) and B ~ N(K*^(t)_S, Sig_B/N) (delta method, [Model]); n_draws draws per
N on the log2 grid log2_lo .. log2_hi (step 1/4 in the frozen cfg); p_AA, p_BB, p_AB for T_E, T_O,
T_N,k (k = 2 on S_ex8, 3 on S_ex16). N* = smallest grid N with p_AB(T_E, S_ex8) >= 0.9;
N0 = 2^round(log2 N*); N*_AA likewise for p_AA; thm:peer single-margin N*_peer = z_delta^2 / rho*^2
(delta = 0.1).
Ladder-level MC (S_lad): per arm-0 grid N, ladder_blocks blocks of (A1, A2, B1, B2): A Gaussian over
all entries via the I_1 Cholesky factor, B simulated exactly from multinomial (C1, C2) and the
certified ERM; predicted Pi_T and rho_T for the four order rungs, Gs and 1-CKA.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm

import s23common as C
import fisher as F
import learnB as B
import metrics as Mt
import sampling as SM

DT = torch.float64
Z_DELTA = float(norm.ppf((1.0 + math.sqrt(0.8)) / 2.0))   # Phi(z)^2 + Phi(-z)^2 = 0.9


def mc_grid(mc: dict):
    n = int(round((mc["log2_hi"] - mc["log2_lo"]) / mc["step"]))
    return [round(mc["log2_lo"] + i * mc["step"], 6) for i in range(n + 1)]


def _psd_chol(S):
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 0.0)
    return V * np.sqrt(w)[None, :]           # factor G with G G^T = S


def _obs_from_entries(X, m, k):
    """X: (draws, E) K-entries in triu order on m probes. Returns keys for T_E, T_O, T_N,k."""
    d = X.shape[0]
    iu = torch.triu_indices(m, m, 1, device=X.device)
    D = torch.zeros((d, m, m), dtype=X.dtype, device=X.device)
    D[:, iu[0], iu[1]] = 2.0 - 2.0 * X
    D[:, iu[1], iu[0]] = 2.0 - 2.0 * X
    TE = torch.argsort(-X, dim=1)
    eye = torch.eye(m, dtype=torch.bool, device=X.device)
    Dm = D.masked_fill(eye[None], float("inf"))
    TO = torch.argsort(Dm, dim=2)[:, :, : m - 1]
    TN = torch.sort(TO[:, :, :k], dim=2).values
    return TE, TO, TN


def _agree(a, b):
    return (a == b).reshape(a.shape[0], -1).all(dim=1)


def probe_mc(kA, kB, SigA, SigB, m, k, log2N_grid, n_draws, key, device, chunk=200000):
    GA = torch.as_tensor(_psd_chol(SigA), dtype=DT, device=device)
    GB = torch.as_tensor(_psd_chol(SigB), dtype=DT, device=device)
    kA = torch.as_tensor(kA, dtype=DT, device=device)
    kB = torch.as_tensor(kB, dtype=DT, device=device)
    E = kA.numel()
    out = {}
    for l2 in log2N_grid:
        N = 2.0 ** l2
        g = C.torch_gen(f"{key}|log2N={l2:.2f}")
        cnt = {f"{o}_{p}": 0 for o in ("TE", "TO", "TN") for p in ("AA", "BB", "AB")}
        done = 0
        while done < n_draws:
            b = min(chunk, n_draws - done)
            Zs = torch.randn((4, b, E), generator=g, dtype=DT).to(device)
            A1 = kA + (Zs[0] @ GA.T) / math.sqrt(N)
            A2 = kA + (Zs[1] @ GA.T) / math.sqrt(N)
            B1 = kB + (Zs[2] @ GB.T) / math.sqrt(N)
            B2 = kB + (Zs[3] @ GB.T) / math.sqrt(N)
            oA1, oA2, oB1, oB2 = (_obs_from_entries(X, m, k) for X in (A1, A2, B1, B2))
            for j, o in enumerate(("TE", "TO", "TN")):
                cnt[f"{o}_AA"] += int(_agree(oA1[j], oA2[j]).sum())
                cnt[f"{o}_BB"] += int(_agree(oB1[j], oB2[j]).sum())
                cnt[f"{o}_AB"] += int(_agree(oA1[j], oB1[j]).sum())
            done += b
        out[f"{l2:.2f}"] = {kk: v / n_draws for kk, v in cnt.items()}
    return out


def first_reaching(curve: dict, field: str, level=0.9):
    for l2 in sorted(curve, key=float):
        if curve[l2][field] >= level:
            return float(l2)
    return float("nan")


def anchor(curve8: dict, level=0.9):
    l2star = first_reaching(curve8, "TE_AB", level)
    l2aa = first_reaching(curve8, "TE_AA", level)
    if not math.isfinite(l2star):
        raise RuntimeError("N* not reached on the MC grid (stop rule R35)")
    return {"log2_Nstar": l2star, "log2_Nstar_AA": l2aa, "log2N0": int(round(l2star))}


def grid_from_N0(log2N0: int, g: dict):
    """Arm-0 grid N0 * 8^j (j = j_lo..j_hi) plus the bracket point N_br = max(N0 / 2^br_offset, 2^br_min)."""
    grid = [log2N0 + g["step_log2"] * j for j in range(g["j_lo"], g["j_hi"] + 1)]
    br = max(log2N0 - g["br_offset"], g["br_min_log2"])
    return {"log2N0": int(log2N0), "grid": sorted(set([br] + grid)), "log2N_br": int(br),
            "log2N_top": int(log2N0 + g["top_offset"]),
            "log2N_dep": [int(log2N0 + o) for o in g["dep_offsets"]],
            "log2N_pilot": [int(log2N0 + o) for o in g["pilot_offsets"]]}


def pi_mc(inst, L, log2N: int, n_blocks: int, key: str):
    """Ladder-level [Model] MC at one N: means of the order rungs, Gs and 1-CKA per pair set."""
    dev = L.device
    n = inst.n
    iu = torch.triu_indices(n, n, 1, device=dev)
    th = torch.as_tensor(inst.K, dtype=DT, device=dev)[iu[0], iu[1]]
    g = C.torch_gen(f"{key}|A|log2N={log2N}")
    N = float(2 ** log2N)
    TH = F.sample_theta_A(L, th, N, 2 * n_blocks, g)
    As = []
    for j in range(2 * n_blocks):
        K = torch.zeros((n, n), dtype=DT, device=dev)
        K[iu[0], iu[1]] = TH[:, j]
        K = K + K.T
        K.fill_diagonal_(1.0)
        As.append(K.cpu().numpy())
    Bs = []
    for j in range(2 * n_blocks):
        r = C.rng(f"{key}|B|log2N={log2N}|rep={j}")
        C1, C2 = SM.sample_B_marginals(inst, 2 ** log2N, r)
        _, Kn, _ = B.solve_B(C1, C2)
        Bs.append(Kn)
    rungs = Mt.ORDER_RUNGS + Mt.GEOM_RUNGS
    res = {s: {r_: [] for r_ in rungs} for s in ("AB", "AA", "BB")}
    for j in range(n_blocks):
        for s, (X, Y) in (("AB", (As[2 * j], Bs[2 * j])), ("AA", (As[2 * j], As[2 * j + 1])),
                          ("BB", (Bs[2 * j], Bs[2 * j + 1]))):
            if X is None or Y is None:
                continue
            d = Mt.fast_rungs(X, Y)
            for r_ in rungs:
                res[s][r_].append(d[r_])
    summ = {}
    for r_ in rungs:
        mab, maa, mbb = (float(np.mean(res[s][r_])) for s in ("AB", "AA", "BB"))
        den = 0.5 * (maa + mbb)
        summ[r_] = {"mean_AB": mab, "mean_AA": maa, "mean_BB": mbb, "Pi": mab - maa - mbb,
                    "rho": mab / den if den > 0 else float("nan"),
                    "frac_nonzero_AB": float(np.mean(np.asarray(res["AB"][r_]) > 0)),
                    "frac_nonzero_AA": float(np.mean(np.asarray(res["AA"][r_]) > 0)),
                    "frac_nonzero_BB": float(np.mean(np.asarray(res["BB"][r_]) > 0))}
    return summ


def peer_nstar(rho_min: float) -> float:
    return Z_DELTA ** 2 / rho_min ** 2
