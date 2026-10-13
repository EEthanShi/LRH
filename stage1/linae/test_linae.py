"""pytest -q linae/test_linae.py  -- unit tests for the local S1.3a metrics and the training core."""
import itertools
import numpy as np
import pytest
from scipy.stats import binomtest

import linae_core as lc
import linae_metrics as lm

RNG = np.random.default_rng(12345)


# ------------------------------------------------------------------ metrics
def test_knn_tie_rule_smaller_index_first():
    # anchor 0 equidistant from 1,2,3 ; anchor order must be 1,2,3 then 4
    D = np.array([[0, 1, 1, 1, 2],
                  [1, 0, 3, 3, 3],
                  [1, 3, 0, 3, 3],
                  [1, 3, 3, 0, 3],
                  [2, 3, 3, 3, 0]], dtype=float)
    idx = lm.knn_idx(D, 2)[0]
    assert list(idx[0]) == [1, 2]
    assert list(idx[4]) == [0, 1]  # D(4,0)=2 first, then ties 1,2,3 -> 1
    assert list(idx[1]) == [0, 2]


def test_knn_overlap_bruteforce():
    for _ in range(5):
        A = RNG.standard_normal((60, 4)); B = A + 0.3 * RNG.standard_normal((60, 4))
        _, DA = lm.gram_sqdist(A); _, DB = lm.gram_sqdist(B)
        for k in (1, 3, 10):
            s = 0
            for i in range(60):
                ca = sorted([j for j in range(60) if j != i], key=lambda j: (DA[i, j], j))[:k]
                cb = sorted([j for j in range(60) if j != i], key=lambda j: (DB[i, j], j))[:k]
                s += len(set(ca) & set(cb))
            assert abs(lm.knn_overlap(DA, DB, k) - s / (60 * k)) < 1e-15


def test_gs_scale_invariant_and_value():
    A = RNG.standard_normal((50, 4))
    _, DA = lm.gram_sqdist(A); _, DB = lm.gram_sqdist(3.0 * A)
    assert lm.gs(DA, DB)[0] < 1e-12
    B = A @ np.diag([1, 2, 1, 1])
    _, DB = lm.gram_sqdist(B)
    off = ~np.eye(50, dtype=bool)
    r = DA[off] / DA[off].mean() - DB[off] / DB[off].mean()
    assert abs(lm.gs(DA, DB)[0] - np.sqrt(np.mean(r ** 2))) < 1e-14


def test_cka_feature_space_identity_and_invariance():
    A = RNG.standard_normal((80, 4)); B = A @ RNG.standard_normal((4, 4))
    KA = A @ A.T; KB = B @ B.T
    Ac = A - A.mean(0); Bc = B - B.mean(0)
    ref = np.linalg.norm(Ac.T @ Bc) ** 2 / (np.linalg.norm(Ac.T @ Ac) * np.linalg.norm(Bc.T @ Bc))
    assert abs(lm.cka_biased(KA, KB) - ref) < 1e-12
    Q = np.linalg.qr(RNG.standard_normal((4, 4)))[0]
    C = 2.5 * A @ Q
    assert abs(lm.cka_biased(KA, C @ C.T) - 1.0) < 1e-12


def test_projector_gl_invariant_and_cca_identity():
    A = RNG.standard_normal((100, 4))
    G = RNG.standard_normal((4, 4)) + 3 * np.eye(4)
    dist, cca = lm.projector_distance(A, A @ G, 4)
    assert dist < 1e-12 and abs(cca - 1) < 1e-12
    B = RNG.standard_normal((100, 4))
    dist, cca = lm.projector_distance(A, B, 4)
    assert abs(dist - (1 - cca)) < 1e-12
    assert 0 <= dist <= 1


def test_clopper_pearson_matches_scipy_and_thresholds():
    for nn in (48, 500):
        for x in range(nn + 1):
            lo, hi = lm.clopper_pearson_onesided(x, nn)
            lo_ref = binomtest(x, nn, alternative='greater').proportion_ci(0.95, 'exact').low
            hi_ref = binomtest(x, nn, alternative='less').proportion_ci(0.95, 'exact').high
            assert abs(lo - lo_ref) < 1e-10 and abs(hi - hi_ref) < 1e-10
    # thresholds stated in final_plan (48 disjoint pairs): reliable iff x>=47, unreliable iff x<=33
    dec = [lm.decision(x, 48)[0] for x in range(49)]
    assert [x for x in range(49) if dec[x] == 'reliable'] == [47, 48]
    assert max(x for x in range(49) if dec[x] == 'unreliable') == 33


# ------------------------------------------------------------------ training core
def _num_grad(f, X, h=1e-6):
    g = np.zeros_like(X)
    for idx in itertools.product(*[range(s) for s in X.shape]):
        Xp = X.copy(); Xp[idx] += h
        Xm = X.copy(); Xm[idx] -= h
        g[idx] = (f(Xp) - f(Xm)) / (2 * h)
    return g


@pytest.mark.parametrize('wd,tied', [(0.0, False), (0.075, False), (0.0, True), (0.075, True)])
def test_gradients_finite_difference(wd, tied):
    U, Sig = lc.population()
    S = Sig[None] + 0.1 * np.eye(lc.n)[None]
    E = RNG.standard_normal((1, lc.d, lc.n)) * 0.3
    W = RNG.standard_normal((1, lc.n, lc.d)) * 0.5
    gE, gW = lc.grads(E, W, S, wd, tied)
    fE = lambda Ex: lc.objective(Ex, W, S, wd, tied)[0]
    assert np.max(np.abs(gE - _num_grad(fE, E))) < 1e-6
    if not tied:
        fW = lambda Wx: lc.objective(E, Wx, S, wd, tied)[0]
        assert np.max(np.abs(gW - _num_grad(fW, W))) < 1e-6


def test_closed_form_optima():
    U, Sig = lc.population()
    S = Sig[None]
    ev, V = np.linalg.eigh(Sig); ev = ev[::-1]; V = V[:, ::-1]
    Ud = V[:, :lc.d]
    # PCA optimum
    E = Ud.T[None]; W = Ud[None]
    assert abs(lc.objective(E, W, S, 0.0, False)[0] - lc.optimum(S, 0.0)[0]) < 1e-12
    # weight-decay optimum: E = D^{1/2} Ud^T, W = E^T, D = diag(1 - wd/lambda_i)
    wd = lc.WD_LAMBDA
    Dh = np.sqrt(1 - wd / ev[:lc.d])
    E = (Dh[:, None] * Ud.T)[None]; W = lc.T(E)
    assert abs(lc.objective(E, W, S, wd, False)[0] - lc.optimum(S, wd)[0]) < 1e-12
    gE, gW = lc.grads(E, W, S, wd, False)
    assert np.abs(gE).max() < 1e-12 and np.abs(gW).max() < 1e-12


def test_gd_conserves_imbalance_small_lr_and_reaches_certificate():
    U, Sig = lc.population()
    R = 4
    E = np.stack([lc.init_params(900000 + r, 'default')[0] for r in range(R)])
    W = np.stack([lc.init_params(900000 + r, 'default')[1] for r in range(R)])
    S = np.broadcast_to(Sig, (R, lc.n, lc.n)).copy()
    C0 = lc.imbalance(E, W, False)
    out = lc.gd_to_certificate(E, W, S, 0.0, False, 0.001, lc.optimum(S, 0.0), 200000)
    assert out['cert_ok'].all()
    C1 = lc.imbalance(out['E'], out['W'], False)
    assert np.max(np.linalg.norm(C1 - C0, axis=(1, 2)) / np.linalg.norm(C0, axis=(1, 2))) < 0.01
    # Theorem 5(a): spec(EE^T) = f(spec C0)
    f = lambda c: (c + np.sqrt(c * c + 4)) / 2
    pred = np.sort(f(np.linalg.eigvalsh(C0)), 1)
    got = np.sort(np.linalg.eigvalsh(out['E'] @ lc.T(out['E'])), 1)
    assert np.max(np.abs(got - pred) / pred) < 0.01


def test_seeds_are_hash_derived_and_distinct():
    assert lc.seed_of('a') != lc.seed_of('b')
    U, _ = lc.population()
    Xs = lc.probes(U)
    Xt = lc.train_data(256, 0, U)
    assert not np.any(np.all(np.isclose(Xs[:, None, :], Xt[None, :, :]), axis=2))


# ------------------------------------------------------------------ v1.1: divergence handling (crash of 2026-10-07)
def test_safe_helpers_never_raise_on_nonfinite_rows():
    A = RNG.standard_normal((5, lc.d, lc.n))
    A[1] = np.nan; A[3, 0, 0] = np.inf
    nrm = lc.safe_spec_norm(A)
    assert np.isinf(nrm[1]) and np.isinf(nrm[3])
    ok = [0, 2, 4]
    assert np.array_equal(nrm[ok], np.linalg.norm(A[ok], ord=2, axis=(1, 2)))
    M = A @ lc.T(A)
    ev = lc.safe_eigvalsh(M)
    assert np.isnan(ev[1]).all() and np.isnan(ev[3]).all()
    assert np.array_equal(ev[ok], np.linalg.eigvalsh(M[ok]))
    with pytest.raises(np.linalg.LinAlgError):      # the v1.0 failure mode
        np.linalg.norm(A, ord=2, axis=(1, 2))


def test_polish_with_nan_rows_flags_diverged_and_does_not_raise():
    U, Sig = lc.population()
    R = 4
    E = np.stack([lc.init_params(900100 + r, 'default')[0] for r in range(R)])
    W = np.stack([lc.init_params(900100 + r, 'default')[1] for r in range(R)])
    E[1] = np.nan; W[1] = np.nan                   # a run that died in the stochastic phase
    S = np.broadcast_to(Sig, (R, lc.n, lc.n)).copy()
    out = lc.gd_to_certificate(E, W, S, 0.0, False, lc.POLISH_LR, lc.optimum(S, 0.0), 200000, lr_cap_by_bound=True)
    assert out['diverged'].tolist() == [False, True, False, False]
    assert out['cert_ok'].tolist() == [True, False, True, True]
    assert out['div_step'][1] == 0 and (out['div_step'][[0, 2, 3]] == -1).all()


def test_gd_divergence_is_flagged_with_step():
    U, Sig = lc.population()
    R = 3
    E = np.stack([lc.init_params(900200 + r, 'default')[0] for r in range(R)])
    W = np.stack([lc.init_params(900200 + r, 'default')[1] for r in range(R)])
    W[2] *= 30.0                                   # huge imbalance -> lr 0.05 is unstable for this run
    S = np.broadcast_to(Sig, (R, lc.n, lc.n)).copy()
    out = lc.gd_to_certificate(E, W, S, 0.0, False, 0.05, lc.optimum(S, 0.0), 20000)
    assert out['diverged'][2] and out['div_step'][2] > 0 and not out['cert_ok'][2]


def test_stochastic_divergence_recorded_with_death_step():
    U, Sig = lc.population()
    R = 3
    E = np.stack([lc.init_params(900300 + r, 'default')[0] for r in range(R)])
    W = np.stack([lc.init_params(900300 + r, 'default')[1] for r in range(R)])
    W[0] *= 50.0
    Ep, Wp, alive, track, death = lc.stochastic_phase(E, W, 'sgd', 0.02, 16, 3000, 0, 1.0, X=None, U=U,
                                                      rng=lc.rng_of('TEST/v1.1'), track_every=1000,
                                                      return_death=True)
    assert not alive[0] and np.isnan(Ep[0]).all() and death[0] == 1000
    assert alive[1:].all() and (death[1:] == -1).all()
    # the v1.1 polish accepts these rows
    S = np.broadcast_to(Sig, (R, lc.n, lc.n)).copy()
    out = lc.gd_to_certificate(Ep, Wp, S, 0.0, False, lc.POLISH_LR, lc.optimum(S, 0.0), 100, lr_cap_by_bound=True)
    assert out['diverged'][0] and not out['diverged'][1:].any()


def test_pass_rule_counts_diverged_pairs_as_failures():
    import pandas as pd
    import analyze_s13a as an
    df = pd.DataFrame(dict(divA=[False, True, False, False], divB=[False, False, False, False],
                           Delta_kNN=[0.0, 0.0, np.nan, 0.2], Delta_Gs=[0.0] * 4, Delta_CKA=[0.0] * 4,
                           Delta_proj=[0.0] * 4))
    pm, _, div = an.pass_mask(df, 'kNN')
    assert pm.tolist() == [True, False, False, False]
    pm, _, _ = an.pass_mask(df, 'proj')            # even a finite, tiny Delta fails if a run diverged
    assert pm.tolist() == [True, False, True, True]
