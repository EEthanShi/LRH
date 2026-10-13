"""SPEC section 7: known-answer plant generators.

Base cloud (all plants): m points uniform on the flat torus [0,1)^p, embedded
by the Clifford map z -> (cos 2 pi z_j, sin 2 pi z_j)_j / (2 pi) in R^{2p}
(local metric = identity, no boundary, homogeneous density, intrinsic
dimension exactly p), then into R^64 by a random isometry. Nested pools are
prefixes of one cloud of size m_max, so m in {256, 1024, 4096} share points.

Plants
  P-Bern  : unit-diagonal Gram K_A of a small spherical cap (the torus lifted
            by a constant coordinate c, c^2 = 0.6 p, then L2-normalized; max D_A
            ~ 0.16) and K_B = exp(beta (K_A - 1)). The cap keeps the map
            D_A -> D_B strictly increasing IN FLOAT64 for beta <= 10 (checked
            and reported as ``float_collisions``); on the full sphere, beta=10
            maps far pairs to equal doubles.
  P-Aniso : B = A M, M = diag(1, 2, 1, ...) in intrinsic coordinates (the
            first circle's two Clifford coordinates scaled by 2).
  P-Conf  : B = sphere inversion of A's Clifford points about a centre at
            distance 2 rho_0 from the torus centre (rho_0 = torus radius);
            conformal on R^{2p}, hence on the embedded torus.
  P-Rough : z_a = (Clifford(z), eps_a), eps_a in R^d ~ N(0, s^2 I / d),
            independent across models; s^2 = kappa_p * median D_(10)(m=256).
  P-Gauss : D_B = D_A + E, E symmetric, iid N(0, sigma^2) off-diagonal.
  Known-p : the base cloud itself (Levina-Bickel bias).
"""
from __future__ import annotations

import numpy as np

from . import util
from .prep import Geom, preprocess

AMBIENT = 64


def torus_z(m: int, p: int, key: str) -> np.ndarray:
    return util.rng_from_key(f"{key}|torus_z").random((m, p))


def clifford(z: np.ndarray, radii=None) -> np.ndarray:
    m, p = z.shape
    r = np.ones(p) if radii is None else np.asarray(radii, dtype=np.float64)
    ang = 2.0 * np.pi * z
    Y = np.empty((m, 2 * p))
    Y[:, 0::2] = np.cos(ang) * r
    Y[:, 1::2] = np.sin(ang) * r
    return Y / (2.0 * np.pi)


def random_isometry(n_in: int, key: str, n_out: int = AMBIENT) -> np.ndarray:
    """(n_out, n_in) with orthonormal columns (QR of a Gaussian, sign-fixed)."""
    G = util.rng_from_key(f"{key}|isometry").standard_normal((n_out, n_in))
    Q, R = np.linalg.qr(G)
    return Q * np.sign(np.diag(R))[None, :]


def embed(Y: np.ndarray, key: str, n_out: int = AMBIENT) -> np.ndarray:
    return Y @ random_isometry(Y.shape[1], key, n_out).T


def torus_dist(z: np.ndarray) -> np.ndarray:
    """Squared chord distance of the Clifford torus (closed form, for tests)."""
    diff = z[:, None, :] - z[None, :, :]
    return (np.sin(np.pi * diff) ** 2).sum(-1) / np.pi ** 2


# ----------------------------------------------------------------------------
# Plants
# ----------------------------------------------------------------------------

def base_cloud(m: int, p: int, key: str) -> np.ndarray:
    """Features (m, 64) of the base torus cloud."""
    return embed(clifford(torus_z(m, p, key)), f"{key}|A")


def bern_grams(m: int, p: int, betas, key: str, cap_c2: float = 0.6):
    """Returns (K_A, {beta: K_B}, info). K_A has an exactly-unit diagonal."""
    Y = clifford(torus_z(m, p, key))
    c = np.sqrt(cap_c2 * p)
    Y = np.concatenate([np.full((m, 1), c), Y], axis=1)
    X = preprocess(embed(Y, f"{key}|A"), "l2")
    KA = X @ X.T
    KA = 0.5 * (KA + KA.T)
    np.fill_diagonal(KA, 1.0)
    out = {}
    info = {}
    ku = np.unique(util.upper_values(KA))
    da = 2.0 - 2.0 * ku
    info["float_collisions_DA"] = int(np.sum(np.diff(da[::-1]) <= 0))
    for beta in betas:
        KB = np.exp(beta * (KA - 1.0))
        np.fill_diagonal(KB, 1.0)
        kb = np.exp(beta * (ku - 1.0))
        db = 2.0 - 2.0 * kb
        info[beta] = {"float_collisions_K": int(np.sum(np.diff(kb) <= 0)),
                      "float_collisions_D": int(np.sum(np.diff(db[::-1]) <= 0)),
                      "n_unique_KA": int(ku.size)}
        out[beta] = KB
    info["max_D_A"] = float(2.0 - 2.0 * ku.min())
    return KA, out, info


def aniso_features(m: int, p: int, key: str, factor: float = 2.0):
    z = torus_z(m, p, key)
    XA = embed(clifford(z), f"{key}|A")
    radii = np.ones(p)
    radii[0] = factor
    XB = embed(clifford(z, radii), f"{key}|B")
    return XA, XB


def conf_features(m: int, p: int, key: str, centre_dist: float = 2.0):
    z = torus_z(m, p, key)
    Y = clifford(z)
    rho0 = np.sqrt(p) / (2.0 * np.pi)
    v = util.rng_from_key(f"{key}|conf_dir").standard_normal(2 * p)
    c0 = centre_dist * rho0 * v / np.linalg.norm(v)
    W = Y - c0
    YB = c0 + W / np.sum(W * W, axis=1, keepdims=True)
    return embed(Y, f"{key}|A"), embed(YB, f"{key}|B")


def rough_features(m: int, p: int, key: str, s2: float, d_priv: int = 32):
    z = torus_z(m, p, key)
    Y = clifford(z)
    eA = util.rng_from_key(f"{key}|privA").standard_normal((m, d_priv)) * np.sqrt(s2 / d_priv)
    eB = util.rng_from_key(f"{key}|privB").standard_normal((m, d_priv)) * np.sqrt(s2 / d_priv)
    XA = embed(np.concatenate([Y, eA], axis=1), f"{key}|A")
    XB = embed(np.concatenate([Y, eB], axis=1), f"{key}|B")
    return XA, XB


def gauss_noise_matrix(m: int, sigma: float, key: str) -> np.ndarray:
    """Symmetric E with iid N(0, sigma^2) above the diagonal, zero diagonal."""
    rng = util.rng_from_key(f"{key}|gauss_E")
    E = np.zeros((m, m))
    for i in range(m - 1):
        e = rng.standard_normal(m - 1 - i) * sigma
        E[i, i + 1:] = e
        E[i + 1:, i] = e
    return E


def bench_features(m: int, key: str, d: int = 768, p: int = 8, noise: float = 0.05):
    """Two realistic-sized feature matrices for timing: shared torus signal
    plus independent isotropic noise, embedded in R^d."""
    z = torus_z(m, p, key)
    Y = clifford(z)
    outs = []
    for tag in ("A", "B"):
        rng = util.rng_from_key(f"{key}|bench|{tag}")
        S = Y @ util.rng_from_key(f"{key}|bench_map|{tag}").standard_normal((2 * p, d))
        outs.append(S + noise * np.sqrt(d) * rng.standard_normal((m, d)) / np.sqrt(d) * 3.0)
    return outs[0], outs[1]


# ----------------------------------------------------------------------------
# Distance-level plant helpers used by run_g0.py (nested pools = prefixes)
# ----------------------------------------------------------------------------

def nested_ids(m_max: int) -> list[str]:
    """Item ids of a nested plant pool; the pool of size m is the prefix."""
    return [str(i) for i in range(m_max)]


def feature_dists(X: np.ndarray) -> np.ndarray:
    """Squared distances of raw features (pre='none'), SPEC 0 construction."""
    from .prep import gram_to_dist
    X = np.asarray(X, dtype=np.float64)
    return gram_to_dist(X @ X.T)


def knn_d_median(D: np.ndarray, k: int) -> float:
    """Median over anchors of the k-th NN squared distance."""
    from .structural import knn_table
    _, v = knn_table(D, k)
    return float(np.median(v[:, k - 1]))


def gauss_plant(m: int, p: int, sigma: float, key: str):
    """P-Gauss: D_A of the base cloud; D_B = D_A + E, E symmetric iid
    N(0, sigma^2) off-diagonal. Returns (D_A, D_B)."""
    DA = feature_dists(base_cloud(m, p, key))
    return DA, DA + gauss_noise_matrix(m, sigma, key)


def warp_g(x: np.ndarray, scale: float) -> np.ndarray:
    """Planted strictly increasing nonlinear warp g(x) = scale*(x/scale)^1.5 + 0.2 x."""
    return scale * (x / scale) ** 1.5 + 0.2 * x


def warpshare_plant(m: int, p: int, noise_rel: float, key: str):
    """D_B = g(D_A) + E (E symmetric iid Gaussian, sd = noise_rel * sd(g(D_A))).
    Returns (D_A, D_B, planted_iso_share) with planted share
    1 - sum_{i<j} E_ij^2 / sum_{i<j} (D_B - D_A)_ij^2 (the fraction of the
    total discrepancy that the planted global monotone warp accounts for)."""
    DA = feature_dists(base_cloud(m, p, key))
    xs = util.upper_values(DA)
    scale = float(np.mean(xs))
    G = warp_g(DA, scale)
    np.fill_diagonal(G, 0.0)
    sd = float(np.std(util.upper_values(G)))
    E = gauss_noise_matrix(m, noise_rel * sd, f"{key}|ws")
    DB = G + E
    e = util.upper_values(E)
    d0 = util.upper_values(DB) - xs
    return DA, DB, float(1.0 - np.dot(e, e) / np.dot(d0, d0))
