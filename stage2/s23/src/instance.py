"""The balanced finite Gibbs model (PREREG_S2_3_v1.0 section 3) and the no-probe items of gate G0.

Construction (frozen; geometry parameters in cfg.FROZEN['geometry']):
  z_x = lv0 C1[s] + lv1 C2[s,c] + lv2 eps_x / sqrt(dim),  C1, C2 unit Gaussian directions,
        eps_x ~ N(0, I_dim); draw order: C1 (n_super x dim), then for each supercluster s:
        C2 (n_clu x dim), then for each cluster c: eps (n_per x dim); generator rng(geometry key).
  K0 = (1 - nugget) RBF_sigma + nugget I.
  P uniform. k*(x,y) = k0(x,y) + a_x + a_y (x != y), k*(x,x) = 1, with a solving
  sum_u P(u) exp(k*(x,u)/t) = Z for all x, Z = mean_x sum_u P(u) exp(k0(x,u)/t).
  P*(y|x) = P(y) exp(k*(x,y)/t) / Z.
Derived: W+ = exp(K*/t)/Z (positive-pair lift), K*^(t) = exp((K*-1)/t),
two-view law P_aug(v1,v2) = sum_x P(x) P*(v1|x) P*(v2|x), W2 = P_aug/(P x P).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import s23common as C


def make_coords(key: str, n_super, n_clu, n_per, dim, lv=(1.0, 0.55, 0.30)):
    r = C.rng(key)
    C1 = r.normal(size=(n_super, dim))
    C1 /= np.linalg.norm(C1, axis=1, keepdims=True)
    pts, sup, clu = [], [], []
    for s in range(n_super):
        C2 = r.normal(size=(n_clu, dim))
        C2 /= np.linalg.norm(C2, axis=1, keepdims=True)
        for c in range(n_clu):
            E = r.normal(size=(n_per, dim)) / np.sqrt(dim)
            pts.append(lv[0] * C1[s] + lv[1] * C2[c] + lv[2] * E)
            sup += [s] * n_per
            clu += [s * n_clu + c] * n_per
    return np.vstack(pts), np.array(sup), np.array(clu)


def rbf(z, sigma):
    sq = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    return np.exp(-sq / (2.0 * sigma ** 2))


def balance(K0: np.ndarray, t: float, P: np.ndarray, max_fp: int = 200000):
    """Symmetric shifts a with Z(x) = Z for all x. Fixed point on b = exp(a/t), then Newton
    polish on log b. Returns (K*, a, Z, imbalance = max Z(x)/min Z(x) - 1, iterations)."""
    n = K0.shape[0]
    E0 = np.exp(K0 / t)
    diagv = P * np.exp(np.diag(K0) / t)          # diagonal terms (k*(x,x) = k0(x,x) = 1)
    np.fill_diagonal(E0, 0.0)
    Z = float((np.exp(K0 / t) @ P).mean())
    if np.any(diagv >= Z):
        raise ValueError("balance: diagonal mass exceeds Z")
    target = Z - diagv
    b = np.ones(n)
    it = 0
    for it in range(max_fp):
        off = b * (E0 @ (P * b))
        if np.max(np.abs(off / target - 1.0)) < 1e-10:
            break
        b = b * np.sqrt(target / off)
    for _ in range(50):                          # Newton on u = log b
        s = E0 @ (P * b)
        F = np.log(b * s) - np.log(target)
        if np.max(np.abs(F)) < 1e-16:
            break
        J = np.eye(n) + (E0 * (P * b)[None, :]) / s[:, None]
        du = np.linalg.solve(J, -F)
        b = b * np.exp(du)
    a = t * np.log(b)
    K = K0 + a[:, None] + a[None, :]
    np.fill_diagonal(K, np.diag(K0))
    rs = np.exp(K / t) @ P
    return K, a, Z, float(rs.max() / rs.min() - 1.0), it


@dataclass
class Instance:
    key: str
    t: float
    z: np.ndarray
    sup: np.ndarray
    clu: np.ndarray
    P: np.ndarray
    K: np.ndarray            # K* (unit diagonal, PD)
    Z: float
    imbalance: float
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.K.shape[0]

    @property
    def W(self):             # W+ = P+(x,y)/(P(x)P(y)) = exp(K*/t)/Z under balance
        return np.exp(self.K / self.t) / self.Z

    @property
    def KBt(self):           # K*^(t) = exp((K*-1)/t)
        return np.exp((self.K - 1.0) / self.t)

    @property
    def Pstar(self):         # P*(y|x), rows sum to 1
        M = self.P[None, :] * np.exp(self.K / self.t) / self.Z
        return M / M.sum(axis=1, keepdims=True)

    @property
    def Q(self):             # joint (anchor, positive) law
        return self.P[:, None] * self.Pstar

    @property
    def Paug(self):          # two-view law P_aug(v1, v2)
        Ps = self.Pstar
        return (Ps * self.P[:, None]).T @ Ps

    @property
    def W2(self):            # two-view lift P_aug / (P x P)
        return self.Paug / np.outer(self.P, self.P)

    def pi_gibbs(self):
        """Triplet law pi(x, a, b) = P(x) P*(a|x) P(b), shape (n, n, n)."""
        return self.Q[:, :, None] * self.P[None, None, :]

    def pi_aug(self):
        """Two-view triplet law pi(v1, v2, b) = P_aug(v1, v2) P(b)."""
        return self.Paug[:, :, None] * self.P[None, None, :]


def build(key: str, t: float, geometry: dict) -> Instance:
    g = geometry
    z, sup, clu = make_coords(key, g["n_super"], g["n_clu"], g["n_per"], g["dim"], tuple(g["lv"]))
    n = z.shape[0]
    K0 = (1.0 - g["nugget"]) * rbf(z, g["sigma"]) + g["nugget"] * np.eye(n)
    P = np.full(n, 1.0 / n)
    K, a, Z, imb, it = balance(K0, t, P)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 1.0)
    return Instance(key=key, t=t, z=z, sup=sup, clu=clu, P=P, K=K, Z=Z, imbalance=imb,
                    meta={"sigma": g["sigma"], "nugget": g["nugget"], "balance_fp_iters": it,
                          "max_abs_shift": float(np.abs(a).max()), "n_super": g["n_super"],
                          "n_clu": g["n_clu"], "n_per": g["n_per"]})


def upper(M):
    iu = np.triu_indices(M.shape[0], 1)
    return M[iu]


def omega_psi(K: np.ndarray, t: float) -> float:
    """omega_psi = || D/psi(D) - g_t(D)/psi(g_t(D)) ||_max, psi = mean off-diagonal distance."""
    D = upper(2.0 - 2.0 * K)
    G = 2.0 * (1.0 - np.exp(-D / (2.0 * t)))
    return float(np.max(np.abs(D / D.mean() - G / G.mean())))


def g0_checks(inst: Instance, thr: dict) -> dict:
    """Deterministic gate G0 items that need no probes (section 3.2); the S_ex8 D-gap item is
    added by gates.g0_full after probe selection."""
    K = inst.K
    ev = np.linalg.eigvalsh(K)
    offK = upper(K)
    D = np.sort(upper(2.0 - 2.0 * K))
    gaps = np.diff(D)
    W = inst.W
    offW = upper(W)
    pmi_diag = np.log(np.diag(inst.W2))
    out = {
        "lambda_min_K": float(ev.min()),
        "max_offdiag_k": float(offK.max()),
        "min_offdiag_k": float(offK.min()),
        "imbalance": inst.imbalance,
        "Z": inst.Z,
        "min_gap_offdiag_D": float(gaps.min()),
        "omega_psi_X": omega_psi(K, inst.t),
        "Wplus_diag": float(W[0, 0]),
        "Wplus_max_off": float(offW.max()),
        "Wplus_frac_off_gt1": float((offW > 1.0).mean()),
        "Wplus_lam_min": float(np.linalg.eigvalsh(W)[0]),
        "twoview_pmi_diag_ptp": float(np.ptp(pmi_diag)),
        "twoview_marginal_maxdev": float(np.max(np.abs(inst.Paug.sum(1) - inst.P))),
        "W2_lam_min": float(np.linalg.eigvalsh(inst.W2)[0]),
    }
    out["pass_lambda_min"] = out["lambda_min_K"] >= thr["lam_min"]
    out["pass_max_off"] = out["max_offdiag_k"] < thr["max_off"]
    out["pass_imbalance"] = out["imbalance"] <= thr["imbalance"]
    out["pass_distinct"] = out["min_gap_offdiag_D"] >= thr["min_gap"]
    out["pass_omega"] = out["omega_psi_X"] >= thr["omega"]
    out["pass_twoview"] = out["twoview_pmi_diag_ptp"] > thr["twoview_ptp"]
    out["pass_all_noprobe"] = all(v for k, v in out.items() if k.startswith("pass_"))
    return out


def geometry_keys(n_alt: int):
    return [C.GEOM_KEY] + [f"{C.GEOM_KEY}|alt={j}" for j in range(1, n_alt + 1)]
