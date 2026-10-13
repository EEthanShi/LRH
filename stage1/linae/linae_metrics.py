"""Local, minimal implementation of the SPEC v1.0 criteria needed by S1.3a (pre="none").

Implemented here (not imported from the shared hrhv2 harness, which was empty when S1.3a ran):
  - kNN sets with the SPEC tie rule (order by (D_ij, j)), mutual overlap s_N(k) (SPEC §1.1)
  - near-tie fraction (SPEC §0)
  - Gs_rms / Gs_max (SPEC §2.6), nu(D) = mean off-diagonal D, D = squared Euclidean distance
  - linear CKA, biased HSIC (SPEC §2.8)
  - projector distance ||P_A - P_B||_F^2 / (2d) on centred features, plus mean squared canonical
    correlation as cross-check (SPEC §4.11)
All arithmetic float64.
"""
import numpy as np


def gram_sqdist(H):
    H = np.asarray(H, dtype=np.float64)
    K = H @ H.T
    dg = np.diag(K)
    D = dg[:, None] + dg[None, :] - 2.0 * K
    np.maximum(D, 0.0, out=D)
    np.fill_diagonal(D, 0.0)
    return K, D


def knn_idx(D, k):
    """First k candidates j != i under the key (D_ij, j): stable argsort keeps index order on exact ties."""
    m = D.shape[0]
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    order = np.argsort(Dm, axis=1, kind='stable')
    return order[:, :k], Dm, order


def knn_overlap(DA, DB, k):
    m = DA.shape[0]
    ia = knn_idx(DA, k)[0]
    ib = knn_idx(DB, k)[0]
    MA = np.zeros((m, m), dtype=bool)
    MB = np.zeros((m, m), dtype=bool)
    rows = np.arange(m)[:, None]
    MA[rows, ia] = True
    MB[rows, ib] = True
    return (MA & MB).sum() / (m * k)


def near_tie_fraction(D, k):
    m = D.shape[0]
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    srt = np.sort(Dm, axis=1)
    off = D[~np.eye(m, dtype=bool)]
    thr = 1e-12 * np.median(off)
    return float(np.mean((srt[:, k] - srt[:, k - 1]) < thr))


def gs(DA, DB):
    m = DA.shape[0]
    off = ~np.eye(m, dtype=bool)
    a = DA[off]; b = DB[off]
    r = a / a.mean() - b / b.mean()
    return float(np.sqrt(np.mean(r * r))), float(np.max(np.abs(r)))


def center_gram(K):
    m = K.shape[0]
    Hc = np.eye(m) - 1.0 / m
    return Hc @ K @ Hc


def cka_biased(KA, KB):
    a = center_gram(KA); b = center_gram(KB)
    hab = np.sum(a * b); haa = np.sum(a * a); hbb = np.sum(b * b)
    return float(hab / np.sqrt(haa * hbb))


def projector(H):
    Hc = H - H.mean(0, keepdims=True)
    return Hc @ np.linalg.pinv(Hc.T @ Hc) @ Hc.T


def projector_distance(HA, HB, d):
    m = HA.shape[0]
    assert d < m
    PA = projector(HA); PB = projector(HB)
    dist = float(np.sum((PA - PB) ** 2) / (2.0 * d))
    # CCA cross-check: canonical correlations = singular values of QA^T QB (orthonormal bases of centred cols)
    QA = np.linalg.qr(HA - HA.mean(0))[0]
    QB = np.linalg.qr(HB - HB.mean(0))[0]
    rho = np.linalg.svd(QA.T @ QB, compute_uv=False)
    rho = np.clip(rho, 0.0, 1.0)
    return dist, float(np.mean(rho ** 2))


def pair_criteria(HA, HB, d, ks=(1, 3, 10, 30)):
    """All S1.3a pair statistics. Discrepancies: knn10 = 1 - s_N(10); gs = Gs_rms; cka = 1 - CKA; proj."""
    KA, DA = gram_sqdist(HA)
    KB, DB = gram_sqdist(HB)
    out = {}
    for k in ks:
        out[f'sN_{k}'] = knn_overlap(DA, DB, k)
    out['near_tie_A_k10'] = near_tie_fraction(DA, 10)
    out['near_tie_B_k10'] = near_tie_fraction(DB, 10)
    out['Gs_rms'], out['Gs_max'] = gs(DA, DB)
    out['CKA'] = cka_biased(KA, KB)
    out['proj'], out['cca_mean_sq'] = projector_distance(HA, HB, d)
    # primary discrepancies
    out['Delta_kNN'] = 1.0 - out['sN_10']
    out['Delta_Gs'] = out['Gs_rms']
    out['Delta_CKA'] = 1.0 - out['CKA']
    out['Delta_proj'] = out['proj']
    return out


# Preregistered tolerances (PREREG S1.3a)
EPS = {'kNN': 0.05, 'Gs': 0.02, 'CKA': 0.01, 'proj': 0.01}
DELTA_KEY = {'kNN': 'Delta_kNN', 'Gs': 'Delta_Gs', 'CKA': 'Delta_CKA', 'proj': 'Delta_proj'}


def clopper_pearson_onesided(x, nn, alpha=0.05):
    """One-sided (1-alpha) exact bounds: lower = Beta_alpha(x, n-x+1), upper = Beta_{1-alpha}(x+1, n-x)."""
    from scipy.stats import beta
    lo = 0.0 if x == 0 else float(beta.ppf(alpha, x, nn - x + 1))
    hi = 1.0 if x == nn else float(beta.ppf(1 - alpha, x + 1, nn - x))
    return lo, hi


def decision(x, nn):
    lo, hi = clopper_pearson_onesided(x, nn)
    if lo > 0.9:
        return 'reliable', lo, hi
    if hi < 0.8:
        return 'unreliable', lo, hi
    return 'inconclusive', lo, hi
