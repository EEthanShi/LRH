"""Unit tests on tiny excluded instances (|X| = 12). Includes the population-limit checks of
Theorem thm:link part 1, the rem:temperature population check and the Lemma T identity check."""
import math

import numpy as np
import pytest
import torch

import s23common as C
import fisher as F
import gates as G
import instance as I
import learnA as A
import learnB as B
import metrics as Mt
import poplim as PL
import sampling as SM
import stats as S

DT = torch.float64
DEV = torch.device("cpu")


def T(M):
    return torch.as_tensor(np.asarray(M), dtype=DT, device=DEV)


def theta(M):
    n = M.shape[0]
    iu = np.triu_indices(n, 1)
    return np.asarray(M)[iu]


def sym_from_theta(v, n):
    M = np.zeros((n, n))
    M[np.triu_indices(n, 1)] = v
    return M + M.T


# ----------------------------------------------------------------------------- instance and sampling
def test_keys_are_test_keys():
    assert C.KEY.startswith("hrhv2-s23-TEST") and C.TEST_MODE
    assert C.STATUS.endswith("STATUS_test.md")


def test_instance_balance(tiny_inst):
    inst = tiny_inst
    assert inst.imbalance <= 1e-12
    assert np.allclose(np.diag(inst.K), 1.0)
    assert np.linalg.eigvalsh(inst.K)[0] > 0
    assert np.allclose(inst.Pstar.sum(1), 1.0, atol=1e-14)
    assert np.allclose(inst.Q, inst.Q.T, atol=1e-15)                       # symmetric positive-pair law
    assert np.allclose(inst.W, inst.Q / np.outer(inst.P, inst.P), rtol=1e-12)
    assert np.allclose(inst.Paug.sum(1), inst.P, atol=1e-15)
    assert abs(inst.pi_gibbs().sum() - 1) < 1e-13 and abs(inst.pi_aug().sum() - 1) < 1e-13


def test_sampling_exact_moments(tiny_inst):
    inst = tiny_inst
    N = 2 ** 12
    pi = inst.pi_gibbs()
    reps = 300
    acc = np.zeros_like(pi)
    for r in range(reps):
        Cn = SM.sample_gibbs(inst, N, C.rng(f"{C.KEY}|t|samp|{r}"))
        assert Cn.sum() == N and Cn.dtype == np.int64 and Cn.min() >= 0
        acc += Cn
    m = acc / reps
    sd = np.sqrt(N * pi * (1 - pi) / reps)
    z = (m - N * pi) / sd
    assert np.abs(z).max() < 6.0
    C1, C2 = SM.marginals(Cn)
    assert C1.sum() == N and C2.sum() == N and np.array_equal(C1.sum(1), C2.sum(1))


# ----------------------------------------------------------------------------- learner A
def test_grad_and_hvp_fd(tiny_inst):
    inst = tiny_inst
    n = inst.n
    obj = A.InfoNCE(T(inst.pi_gibbs()), inst.t)
    r = np.random.default_rng(0)
    K = inst.K + 0.01 * sym_from_theta(r.normal(size=n * (n - 1) // 2), n)
    f, g, _ = obj.grad_theta(T(K))
    V = sym_from_theta(r.normal(size=n * (n - 1) // 2), n)
    h = 1e-6
    fd = (obj.value(T(K + h * V)) - obj.value(T(K - h * V))) / (2 * h)
    assert abs(fd - A.inner(g, T(V))) < 1e-7 * max(1.0, abs(fd))
    obj.prep_hess(T(K))
    Hv = obj.hvp(T(V))
    _, gp, _ = obj.grad_theta(T(K + h * V))
    _, gm, _ = obj.grad_theta(T(K - h * V))
    fdH = (gp - gm) / (2 * h)
    assert float((Hv - fdH).abs().max()) < 1e-6 * float(fdH.abs().max())


def test_I1_equals_population_hessian(tiny_inst):
    inst = tiny_inst
    n = inst.n
    pi = T(inst.pi_gibbs())
    I1 = F.assemble_I1(pi, T(inst.K), inst.t).numpy()
    obj = A.InfoNCE(pi, inst.t)
    obj.prep_hess(T(inst.K))
    V = sym_from_theta(np.random.default_rng(1).normal(size=n * (n - 1) // 2), n)
    Hv = theta(obj.hvp(T(V)).numpy())
    assert np.allclose(I1 @ theta(V), Hv, rtol=1e-10, atol=1e-14)
    assert np.linalg.eigvalsh(I1)[0] > 0                                     # thm:peer: I_J positive definite


def _precond(inst):
    L = F.cholesky_inplace_safe(F.assemble_I1(T(inst.pi_gibbs()), T(inst.K), inst.t))
    return A.FisherPrecond(L, inst.n), L


def test_thm_link_part1_A_population(tiny_inst):
    """Theorem thm:link part 1 at the population: the InfoNCE (tau = t) minimizer over the elliptope is K*."""
    inst = tiny_inst
    pre, _ = _precond(inst)
    K, info = A.solve_A(T(inst.pi_gibbs()), inst.t, torch.eye(inst.n, dtype=DT), pre)
    assert info["cert_pass"] and info["path"] in ("newton", "bm+newton")   # interior certificate
    assert float((K - T(inst.K)).abs().max()) <= 1e-9


def test_thm_link_part1_B_population(tiny_inst):
    """Theorem thm:link part 1 at the population: unconstrained spectral minimizer = W+, normalized = K*^(t)."""
    inst = tiny_inst
    Kraw, Kn, info = B.solve_B(inst.Q, np.outer(inst.P, inst.P))
    assert info["path"] == "closed_form" and info["cert_pass"]
    assert np.max(np.abs(Kraw - inst.W)) <= 1e-12 * np.max(inst.W)
    assert np.max(np.abs(Kn - inst.KBt)) <= 1e-12
    # g_t link on distances
    D, DB = 2 - 2 * inst.K, 2 - 2 * Kn
    assert np.max(np.abs(DB - 2 * (1 - np.exp(-D / (2 * inst.t))))) <= 1e-12


def test_rem_temperature_population(tiny_inst):
    """rem:temperature at the population: tau = t/2 and t/2.5 give (tau/t) K* + (1 - tau/t) 11^T."""
    inst = tiny_inst
    pre, _ = _precond(inst)
    for div in (2.0, 2.5):
        K, info = A.solve_A(T(inst.pi_gibbs()), inst.t / div, torch.eye(inst.n, dtype=DT), pre)
        assert info["cert_pass"]
        assert PL.lemmaT_residual(K.numpy(), inst.K, 1.0 / div) <= 1e-9


def _lamI(L):
    return A.lam_min_I1(L, f"{C.KEY}|t|lamI")["lam_min_I1_lb"]


def test_lanczos_and_fisher_at_Kstar(tiny_inst):
    """lambda_min(I_1) bound against a dense eigensolve; at the population minimizer the preconditioned
    Hessian M = L^{-1} H L^{-T} is the identity (H = I_1, thm:peer)."""
    inst = tiny_inst
    pre, L = _precond(inst)
    I1 = F.assemble_I1(T(inst.pi_gibbs()), T(inst.K), inst.t).numpy()
    ev = np.linalg.eigvalsh(I1)
    lb = A.lam_min_I1(L, f"{C.KEY}|t|lamI")
    assert lb["lam_min_I1_lb"] <= ev[0] * (1 + 1e-9) and lb["lam_min_I1_lb"] >= 0.99 * ev[0]
    hb = A.theta_hess_bounds(A.InfoNCE(T(inst.pi_gibbs()), inst.t), T(inst.K), L, f"{C.KEY}|t|lzK")
    assert abs(hb["lam_min_M"] - 1) < 1e-9 and abs(hb["lam_max_M"] - 1) < 1e-9
    # at t/2 and the rescaled K*, H_{t/2} = 4 H_t: M = 4 I
    hb2 = A.theta_hess_bounds(A.InfoNCE(T(inst.pi_gibbs()), inst.t / 2), T(0.5 * inst.K + 0.5), L, f"{C.KEY}|t|lzK2")
    assert abs(hb2["lam_min_M"] - 4) < 1e-8 and abs(hb2["lam_max_M"] - 4) < 1e-8


def test_lemmaT_identity_same_counts(tiny_inst):
    """Lemma T identity check (an identity check, never a claim; binding correction (a)): on the same
    counts, ERM_{t/2} = 1/2 ERM_t + 1/2 11^T whenever ERM_t is the UNIQUE minimizer and PD. PD is decided
    with the margin max(1e-6, 10 e_t), uniqueness by the absence of a Hessian null space, and the residual
    tolerance comes from the curvature (e = 2 ||g||_2 / lambda_lb(H)), not a fixed 1e-8."""
    inst = tiny_inst
    pre, L = _precond(inst)
    lamI = _lamI(L)
    checked = 0
    for log2N in (14, 18, 22):
        Cn = SM.sample_gibbs(inst, 2 ** log2N, C.rng(f"{C.KEY}|t|lemT|{log2N}"))
        W = T(Cn) / float(2 ** log2N)
        Kt, it = A.solve_A(W, inst.t, T(inst.K), pre)
        Kh, ih = A.solve_A(W, inst.t / 2, T(0.5 * inst.K + 0.5), pre)
        assert it["cert_pass"] and ih["cert_pass"]
        hb1 = A.theta_hess_bounds(A.InfoNCE(W, inst.t), Kt, L, f"{C.KEY}|t|lz1|{log2N}") if it["path"] != "bm" else None
        hb2 = A.theta_hess_bounds(A.InfoNCE(W, inst.t / 2), Kh, L, f"{C.KEY}|t|lz2|{log2N}") if ih["path"] != "bm" else None
        dec = A.lemmaT_decision(Kt, it, hb1, Kh, ih, hb2, 0.5, lamI)
        if dec["applicable"]:
            assert not dec["violation"], dec
            assert dec["tol"] < 1e-9 and dec["resid"] <= dec["tol"], dec
            # a planted violation of 10 tol is flagged
            Kbad = Kh.clone()
            Kbad[0, 1] += 10 * dec["tol"]; Kbad[1, 0] += 10 * dec["tol"]
            assert A.lemmaT_decision(Kt, it, hb1, Kbad, ih, hb2, 0.5, lamI)["violation"]
            checked += 1
    assert checked >= 2


def test_lemmaT_uniqueness_flag_without_self_candidates(tiny_inst):
    """Without self-candidates the risk is flat along 11^T - I (the skeptic's non-uniqueness example):
    the Hessian has a null space over the off-diagonal coordinates and the check must say 'not unique'."""
    inst = tiny_inst
    pre, L = _precond(inst)
    n = inst.n
    Cn = SM.sample_gibbs(inst, 2 ** 22, C.rng(f"{C.KEY}|t|noself")).astype(np.float64)
    for x in range(n):
        Cn[x, x, :] = 0.0
        Cn[x, :, x] = 0.0
    W = T(Cn / Cn.sum())
    Kt, it = A.solve_A(W, inst.t, T(inst.K), pre)
    hb = A.theta_hess_bounds(A.InfoNCE(W, inst.t), Kt, L, f"{C.KEY}|t|lzns")
    assert hb["lam_min_M"] < 1e-8
    Kh, ih = A.solve_A(W, inst.t / 2, T(0.5 * inst.K + 0.5), pre)
    dec = A.lemmaT_decision(Kt, it, hb, Kh, ih, None, 0.5, _lamI(L))
    assert not dec["unique"] and not dec["applicable"] and not dec["violation"]
    # the flat direction: K + c (11^T - I) has the same risk
    obj = A.InfoNCE(W, inst.t)
    E = torch.ones((n, n), dtype=DT) - torch.eye(n, dtype=DT)
    assert abs(obj.value(Kt) - obj.value(Kt + 0.01 * E)) <= 1e-13


def test_A_newton_vs_bm_interior(tiny_inst):
    inst = tiny_inst
    pre, _ = _precond(inst)
    Cn = SM.sample_gibbs(inst, 2 ** 20, C.rng(f"{C.KEY}|t|nbm"))
    W = T(Cn) / float(2 ** 20)
    K1, i1 = A.solve_A(W, inst.t, T(inst.K), pre)
    Y0 = A.random_factor(inst.n, inst.n, 5, DEV)
    K2, i2 = A.solve_A(W, inst.t, None, pre, Y_init=Y0, newton_first=False)
    assert i1["path"] == "newton" and i2["cert_pass"]
    assert float((K1 - K2).abs().max()) <= 1e-8


def test_A_boundary_certificate_two_inits(tiny_inst):
    """Small N: the PSD constraint binds; BM reaches the KKT certificate from two inits, same ERM."""
    inst = tiny_inst
    pre, _ = _precond(inst)
    found = False
    for log2N in (6, 7, 8):
        Cn = SM.sample_gibbs(inst, 2 ** log2N, C.rng(f"{C.KEY}|t|bd|{log2N}"))
        W = T(Cn) / float(2 ** log2N)
        K1, i1 = A.solve_A(W, inst.t, T(inst.K), pre)
        if i1["path"] != "bm":
            continue
        found = True
        Y0 = A.random_factor(inst.n, inst.n, 9, DEV)
        K2, i2 = A.solve_A(W, inst.t, None, pre, Y_init=Y0, newton_first=False)
        assert i1["cert_pass"] and i2["cert_pass"]
        assert float((K1 - K2).abs().max()) <= 1e-4
        assert i1["psd_active"]
    assert found


# ----------------------------------------------------------------------------- learner B and population maps
def test_B_fista_kkt_and_inits(tiny_inst):
    inst = tiny_inst
    hit = False
    for log2N in (6, 7, 8, 9):
        Cn = SM.sample_gibbs(inst, 2 ** log2N, C.rng(f"{C.KEY}|t|bf|{log2N}"))
        C1, C2 = SM.marginals(Cn)
        Wm, C1s, R, zero = B.spectral_parts(C1, C2)
        if zero.any() or np.linalg.eigvalsh(R)[0] >= 0:
            continue
        hit = True
        K1, Kn1, i1 = B.solve_B(C1, C2)
        K2, Kn2, i2 = B.solve_B(C1, C2, init="zero")
        assert i1["cert_pass"] and i2["cert_pass"] and i1["proj_active"]
        Gm = 2.0 * Wm * (K1 - R)
        assert np.linalg.eigvalsh(Gm)[0] >= -1e-6 * np.abs(Gm).max()
        assert np.abs(Gm @ K1).max() <= 1e-6 * np.abs(Gm).max() * np.abs(K1).max()
        assert np.max(np.abs(K1 - K2)) <= 1e-8
    assert hit


def test_ball_projection_vs_higham_and_large_mu(tiny_inst):
    inst = tiny_inst
    Kb, ib = B.ball_projection(inst.W, 1.0)
    Kh, ih = B.ncm_higham(inst.W)
    assert np.max(np.abs(Kb - Kh)) <= 1e-8
    mu = float(np.max(np.diag(inst.W))) * 1.01                    # W+ feasible -> exact recovery
    Kb2, _ = B.ball_projection(inst.W, mu)
    assert np.max(np.abs(Kb2 - inst.W)) <= 1e-8 * np.max(inst.W)


def test_eckart_young(tiny_inst):
    W = tiny_inst.W
    Wr, info = B.eckart_young(W, 4)
    ev = np.linalg.eigvalsh(Wr)
    assert (ev > 1e-10 * ev.max()).sum() == 4
    w = np.sort(np.linalg.eigvalsh(W))[::-1]
    assert abs(np.linalg.norm(W - Wr) ** 2 - (w[4:] ** 2).sum()) <= 1e-9 * (w ** 2).sum()


# ----------------------------------------------------------------------------- metrics
def test_iso_inf_vs_lp():
    out = G.g1_iso(40, f"{C.KEY}|t|iso")
    assert out["pass"], out


def test_lemmaL_rungs_direct():
    out = G.g1_lemmaL(30, 24, f"{C.KEY}|t|lemL")
    assert out["pass"], out


def test_exact_obs_and_links(tiny_inst):
    inst = tiny_inst
    S = list(range(0, 12, 2))
    e = Mt.exact_obs(inst.K, inst.K, S, 2)
    assert all(e.values())
    e2 = Mt.exact_obs(inst.K, Mt.phi_t(inst.K, inst.t), S, 2)          # monotone link keeps every order
    assert all(e2.values())
    om = Mt.omega_self(inst.K, inst.t)
    assert abs(Mt.gs_max(inst.K, Mt.phi_t(inst.K, inst.t)) - om) <= 1e-12
    c, frac = Mt.inv_link_cf(inst.K, Mt.phi_t(inst.K, inst.t), inst.t)
    assert c <= 1e-12 and frac == 0.0
    d = Mt.ladder_deltas(inst.K, Mt.rescale_half(inst.K))
    assert d["Gs_max"] <= 1e-12 and d["dCKA"] <= 1e-12 and d["dE"] == 0.0 and d["dEk"] == 0.0
    r = np.random.default_rng(3)
    X = r.normal(size=(12, 3))
    Kx = X @ X.T
    Kx = Kx / np.sqrt(np.outer(np.diag(Kx), np.diag(Kx)))
    assert not Mt.exact_obs(inst.K, Kx, S, 2)["T_E"]


def test_fast_rungs_equal_ladder(tiny_inst):
    inst = tiny_inst
    r = np.random.default_rng(4)
    E = sym_from_theta(r.normal(size=66) * 0.02, 12)
    KA = inst.K + E
    d = Mt.ladder_deltas(KA, inst.KBt)
    f = Mt.fast_rungs(KA, inst.KBt)
    for k in ("dN_k10", "dO", "dE", "dEk", "dMST", "Gs", "dCKA"):
        assert abs(d[k] - f[k]) <= 1e-12, k


# ----------------------------------------------------------------------------- statistics
def test_decision_thresholds():
    assert S.decide(92, 96)[0] == "reliable" and S.decide(91, 96)[0] != "reliable"
    assert S.decide(69, 96)[0] == "unreliable" and S.decide(70, 96)[0] != "unreliable"
    assert S.decide(47, 48)[0] == "reliable" and S.decide(46, 48)[0] != "reliable"
    assert S.decide(33, 48)[0] == "unreliable" and S.decide(34, 48)[0] != "unreliable"
    assert abs(S.p_reliable(96, 96) - 0.9 ** 96) < 1e-12
    h = S.holm({"a": 0.001, "b": 0.02, "c": 0.04}, 0.05)
    assert h["a"]["rejected"] and h["b"]["rejected"] and h["c"]["rejected"]
    h = S.holm({"a": 0.001, "b": 0.03, "c": 0.04}, 0.05)
    assert h["a"]["rejected"] and not h["b"]["rejected"] and not h["c"]["rejected"]


def test_pi_test_and_betabinom():
    z = np.zeros(20)
    r = S.pi_test(z, z, z, f"{C.KEY}|t|pi0")
    assert r["no_plateau"] and r["all_zero"]
    rng = np.random.default_rng(5)
    aa = rng.uniform(0, 0.01, 48); bb = rng.uniform(0, 0.01, 48)
    ab = 0.5 * (aa + bb)
    assert S.pi_test(ab, aa, bb, f"{C.KEY}|t|pi1")["no_plateau"]
    assert S.pi_test(ab + 0.1, aa, bb, f"{C.KEY}|t|pi2")["plateau"]
    assert abs(S.betabinom_reliable(96, 64) - 0.93) < 0.01


# ----------------------------------------------------------------------------- CLT / delta method (thm:peer part 1)
def test_A_variance_matches_I1_inverse(tiny_inst):
    inst = tiny_inst
    pre, L = _precond(inst)
    ids, pairs = F.entry_ids_pairs([(0, 1), (0, 7), (3, 10)], inst.n)
    Sig = F.cov_A_entries(L, ids)
    N = 2 ** 22
    vals = []
    for r in range(120):
        Cn = SM.sample_gibbs(inst, N, C.rng(f"{C.KEY}|t|clt|{r}"))
        K, info = A.solve_A(T(Cn) / float(N), inst.t, T(inst.K), pre)
        assert info["cert_pass"]
        vals.append([math.sqrt(N) * (float(K[a, b]) - inst.K[a, b]) for a, b in pairs])
    v = np.var(np.array(vals), axis=0, ddof=1)
    ratio = v / np.diag(Sig)
    assert np.all((ratio > 0.6) & (ratio < 1.6)), ratio


def test_B_delta_method(tiny_inst):
    inst = tiny_inst
    pairs = [(0, 1), (0, 7), (3, 10)]
    _, SigB = F.cov_B_entries(inst, pairs)
    N = 2 ** 22
    vals = []
    for r in range(300):
        C1, C2 = SM.sample_B_marginals(inst, N, C.rng(f"{C.KEY}|t|dm|{r}"))
        _, Kn, _ = B.solve_B(C1, C2)
        vals.append([math.sqrt(N) * (Kn[a, b] - inst.KBt[a, b]) for a, b in pairs])
    v = np.var(np.array(vals), axis=0, ddof=1)
    ratio = v / np.diag(SigB)
    assert np.all((ratio > 0.7) & (ratio < 1.4)), ratio


# ----------------------------------------------------------------------------- non-convex population fits
def test_rank_A_and_lanczos(tiny_inst, tiny_cfg):
    inst = tiny_inst
    pc = dict(tiny_cfg["poplim"])
    K, info = PL.rank_A(T(inst.pi_gibbs()), inst.t, 4, inst.K, f"{C.KEY}|t|rank", pc, DEV)
    assert np.allclose(np.diag(K), 1.0)
    ev = np.linalg.eigvalsh(K)
    assert (ev > 1e-8 * ev.max()).sum() <= 4
    for fz in info["fits"]:
        assert math.isfinite(fz["hess_min"]) and math.isfinite(fz["hess_max"]) and fz["hess_max"] > 0
    # full rank (convex, interior): the Lanczos minimum on the horizontal space is non-negative
    obj = A.InfoNCE(T(inst.pi_gibbs()), inst.t)
    Y = A.factor_from(T(inst.K))
    lmin, lmax = PL.hess_lanczos(obj, Y, f"{C.KEY}|t|lz")
    assert lmin >= -1e-8 * lmax


# ----------------------------------------------------------------------------- Lemma L (identity checks only)
def test_lemmaL_binary_identity_exact():
    """Lemma L part 3 (independent runs, binary features): E D_AB - E D_AA - E D_BB equals
    -sum_u w_u [q_A(1-2q_A) + q_B(1-2q_B) + 2 q_A q_B] -- exact enumeration on 3 features."""
    import itertools
    r = np.random.default_rng(11)
    for _ in range(50):
        U = 3
        w = r.uniform(0.1, 1.0, U)
        qA, qB = r.uniform(0, 0.5, U), r.uniform(0, 0.5, U)
        star = r.integers(0, 2, U)

        def law(q):
            out = []
            for flips in itertools.product((0, 1), repeat=U):
                p = np.prod([q[u] if flips[u] else 1 - q[u] for u in range(U)])
                out.append((np.where(np.array(flips) == 1, 1 - star, star), p))
            return out

        def ED(L1, L2):
            return sum(p1 * p2 * float(np.sum(w * np.abs(f1 - f2))) for f1, p1 in L1 for f2, p2 in L2)

        lA, lB = law(qA), law(qB)
        lhs = ED(lA, lB) - ED(lA, lA) - ED(lB, lB)
        rhs = -float(np.sum(w * (qA * (1 - 2 * qA) + qB * (1 - 2 * qB) + 2 * qA * qB)))
        assert abs(lhs - rhs) < 1e-12
        assert lhs <= 1e-15                                        # Pi <= 0 under the median premise
        rho = ED(lA, lB) / (0.5 * (ED(lA, lA) + ED(lB, lB)))
        assert 1 - 1e-12 <= rho <= 2 + 1e-12                       # 1 <= rho <= 2 (independence + premise)


def test_lemmaL_common_median_ternary():
    """Lemma L part 1 for ternary features (ties counted 1/2): with a common per-feature median, Pi <= 0
    for independent runs (random laws; exact)."""
    r = np.random.default_rng(12)
    vals = np.array([-1.0, 0.0, 1.0])
    worst = -np.inf
    for _ in range(2000):
        pA, pB = r.dirichlet(np.ones(3)), r.dirichlet(np.ones(3))

        def med_ok(p, m):
            cdf = np.cumsum(p)
            i = int(np.where(vals == m)[0][0])
            return cdf[i] >= 0.5 and (cdf[i - 1] if i > 0 else 0.0) <= 0.5
        common = [m for m in vals if med_ok(pA, m) and med_ok(pB, m)]
        if not common:
            continue
        D = np.abs(vals[:, None] - vals[None, :]) / 2
        pi = pA @ D @ pB - pA @ D @ pA - pB @ D @ pB
        worst = max(worst, pi)
    assert worst <= 1e-12


def test_tau_b_rung_breaks_lemmaL_with_ties():
    """Why Pi uses dEk: with ties, Delta_E = (1 - tau_b)/2 (varying denominator) can give Pi > 0 although
    every feature has the true sign as median, while the fixed-denominator Kendall distance gives Pi <= 0
    (the skeptic's construction, exact over the 2 x 2 mode laws, through hrhv2's own count functions)."""
    from hrhv2 import structural as st
    n_edges = 15
    truth = 1 + np.arange(n_edges) / n_edges
    found = False
    for g in range(2, n_edges):
        for ptie in (0.2, 0.3, 0.4, 0.45, 0.49):
            A_tie = truth.copy(); A_tie[:g] = truth[g - 1]
            B_tie = truth.copy(); B_tie[-g:] = truth[-g]
            mA, mB, p = [truth, A_tie], [truth, B_tie], [1 - ptie, ptie]

            def E(m1, m2, f):
                return sum(p[i] * p[j] * f(st.kendall_counts(m1[i], m2[j])) for i in range(2) for j in range(2))

            def taub(c):
                return (1.0 - st.tau_b_from_counts(c)) / 2.0

            Pi_t = E(mA, mB, taub) - E(mA, mA, taub) - E(mB, mB, taub)
            Pi_h = (E(mA, mB, st.kendall_distance_from_counts) - E(mA, mA, st.kendall_distance_from_counts)
                    - E(mB, mB, st.kendall_distance_from_counts))
            assert Pi_h <= 1e-15
            if Pi_t > 1e-3:
                found = True
    assert found
