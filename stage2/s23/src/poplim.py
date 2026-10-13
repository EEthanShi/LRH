"""Population limits of every S2.3 arm (PREREG section 4.4, frozen pre-step) and their summaries.

Cases (each gives (K_A^inf, K_B^inf), both unit diagonal):
  '0'      A: InfoNCE tau=t over the elliptope (convex, certified; init I)   B: W+ normalized
  'Rm'     A: tau=t/2 (convex; init I)  -- rem:temperature / Lemma T identity check at population
  'Rp'     A: tau=2t (convex, boundary; init I)                              descriptive
  'i_mu<m>' B: weighted (here: uniform) projection of W+ onto {K >= 0, diag <= mu}, normalized
  'ii_p<p>' B: Eckart-Young rank-p truncation of W+ (exact), normalized; A: rank-p BM, n_inits inits
  'ii_tau' A: rank-128 BM at tau = t/2.5 (n_inits inits) -- Lemma T residual vs 'ii_p128'
  'iv'     A: two-view InfoNCE tau=t (convex)                                B: W2 normalized
  'iv_tau' A: two-view tau = t/2.5 (convex) -- Lemma T identity check vs 'iv' ONLY if the two-view
           limit at t is PD with margin and unique; otherwise residual reported (descriptive)
  'v'      B: projection of W2 onto {K >= 0, diag <= 1}, rank check <= v_rank; A: rank-v_rank BM, two views
Non-convex acceptance rule (section 4.4): every init reaches relative Riemannian gradient <= 1e-9 and
smallest (Lanczos) horizontal Riemannian Hessian eigenvalue >= -1e-8 * largest; all inits within
1e-4 (max norm) of the best. Otherwise the case has NO prediction (descriptive only).
Budget (PREREG R22): <= rank_max_evals L-BFGS evaluations and polish_iters Newton-CG steps per init; all
non-convex fits together get at most nonconvex_cap_h hours of wall time; an init that would start after
the cap is not run, and its case is not accepted (no prediction, reason 'time_cap').
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

import s23common as C
import learnA as A
import learnB as B
import metrics as Mt

DT = torch.float64


def _t(M, dev):
    return torch.as_tensor(np.asarray(M), dtype=DT, device=dev)


def _np(K):
    return K.detach().cpu().numpy() if torch.is_tensor(K) else np.asarray(K)


def ones(n):
    return np.ones((n, n))


# ----------------------------------------------------------------------------- convex A
def convex_A(pi_t, tau, precond, scfg, K_init=None, key=None):
    """Certified population ERM over the elliptope (Newton from K_init (default I), else BM)."""
    n = pi_t.shape[0]
    dev = pi_t.device
    K0 = torch.eye(n, dtype=DT, device=dev) if K_init is None else K_init
    K, info = A.solve_A(pi_t, tau, K0, precond, rel=scfg["bm_rel"], max_evals=scfg["bm_max_evals"],
                        step_tol=scfg["newton_step_tol"], stall_chunks=scfg["bm_stall_chunks"],
                        pcg_rtol=scfg["pcg_rtol"])
    if not info["cert_pass"] and key is not None:      # one re-solve from a keyed random factor
        Y0 = A.random_factor(n, n, C.seed_from_key(key + "|resolve"), dev)
        K2, info2 = A.solve_A(pi_t, tau, None, precond, rel=scfg["bm_rel"], max_evals=scfg["bm_max_evals"],
                              step_tol=scfg["newton_step_tol"], Y_init=Y0, newton_first=False,
                              stall_chunks=scfg["bm_stall_chunks"], pcg_rtol=scfg["pcg_rtol"])
        info["resolve"] = {k: info2.get(k) for k in ("path", "cert_pass", "lam_min_K")}
        if info2["cert_pass"]:
            K, info = K2, dict(info2, resolved=True)
    return _np(K), info


# ----------------------------------------------------------------------------- rank-p A (non-convex)
def _vertical_proj(Y, V):
    """Orthogonal projection of V onto the orbit directions {Y Om : Om skew}."""
    G = Y.T @ Y
    g, U = torch.linalg.eigh(G)
    Bm = Y.T @ V - V.T @ Y
    Bt = U.T @ Bm @ U
    den = g[:, None] + g[None, :]
    Om_t = torch.where(den > 1e-12 * float(g.max()), Bt / torch.where(den > 0, den, torch.ones_like(den)),
                       torch.zeros_like(Bt))
    Om = U @ Om_t @ U.T
    return Y @ Om


def hess_lanczos(obj, Y, key: str, k: int = 300):
    """Smallest / largest Ritz values of the Riemannian Hessian on the horizontal space (tangent to the
    oblique manifold, orthogonal to the O(p) orbit), by k-step Lanczos with full reorthogonalization.
    Normal and vertical directions are shifted to +10*|lambda_max estimate| so they cannot be the minimum."""
    hess, proj, f, RG, EG = A.riemannian_hess_op(obj, Y)
    n, p = Y.shape
    dev = Y.device

    def horiz(V):
        V = proj(V)
        return V - _vertical_proj(Y, V)

    def run(op, k):
        g = C.torch_gen(key)
        v = torch.randn((n, p), generator=g, dtype=DT).to(dev)
        v = v / torch.linalg.norm(v)
        Q = [v]
        al, be = [], []
        w = op(v)
        a = float((w * v).sum())
        al.append(a)
        w = w - a * v
        for _ in range(1, k):
            for q in Q:                                   # full reorthogonalization
                w = w - float((w * q).sum()) * q
            b = float(torch.linalg.norm(w))
            if b < 1e-14:
                break
            be.append(b)
            v = w / b
            Q.append(v)
            w = op(v) - b * Q[-2]
            a = float((w * v).sum())
            al.append(a)
            w = w - a * v
        T = np.diag(al) + np.diag(be[:len(al) - 1], 1) + np.diag(be[:len(al) - 1], -1)
        ev = np.linalg.eigvalsh(T)
        return float(ev[0]), float(ev[-1])

    _, lmax = run(lambda V: horiz(hess(horiz(V))), min(k, 60))
    shift = 10.0 * abs(lmax) + 1e-300

    def op(V):
        Vh = horiz(V)
        return horiz(hess(Vh)) + shift * (V - Vh)

    lmin, _ = run(op, k)
    return lmin, lmax


_NC_DEADLINE = [float("inf")]           # wall-clock deadline of all non-convex fits (set by all_limits)


def rank_A(pi_t, tau, p, init0_K, key, pcfg, dev):
    """n_inits rank-p fits (init 0 = factor of init0_K truncated to rank p; others keyed random)."""
    n = pi_t.shape[0]
    obj = A.InfoNCE(pi_t, tau)
    fits = []
    capped = False
    for j in range(pcfg["n_inits"]):
        if time.time() > _NC_DEADLINE[0]:
            capped = True
            break
        t0 = time.time()
        if j == 0:
            Y0 = A.factor_from(_t(init0_K, dev), p)
        else:
            Y0 = A.random_factor(n, p, C.seed_from_key(f"{key}|init={j}"), dev)
        r = A.bm_rank(obj, Y0, max_evals=pcfg["rank_max_evals"], polish_iters=pcfg["polish_iters"],
                      rel_grad=pcfg["accept_relgrad"])
        lmin, lmax = hess_lanczos(obj, r["Y"], f"{key}|lanczos={j}")
        fits.append({"K": _np(r["K"]), "f": float(r["f"]), "rel_grad": float(r["rel_grad"]),
                     "hess_min": lmin, "hess_max": lmax, "hess_rel": lmin / abs(lmax) if lmax else float("nan"),
                     "evals": int(r["evals"]), "polish_steps": len(r["polish"]), "time_s": time.time() - t0})
    if not fits:
        Y0 = A.factor_from(_t(init0_K, dev), p)
        K0 = Y0 @ Y0.T
        K0.fill_diagonal_(1.0)
        return _np(K0), {"accepted": False, "time_cap": True, "n_inits_run": 0, "fits": [],
                         "note": "no init run before the non-convex time cap; K = truncated init (no prediction)"}
    best = int(np.argmin([fz["f"] for fz in fits]))
    Kb = fits[best]["K"]
    spread = max(float(np.max(np.abs(fz["K"] - Kb))) for fz in fits)
    ok_grad = all(fz["rel_grad"] <= pcfg["accept_relgrad"] for fz in fits)
    ok_hess = all(fz["hess_rel"] >= -pcfg["accept_hess_rel"] for fz in fits)
    accepted = bool(ok_grad and ok_hess and spread <= pcfg["accept_spread"] and not capped)
    summ = [{k: v for k, v in fz.items() if k != "K"} for fz in fits]
    return Kb, {"accepted": accepted, "ok_grad": ok_grad, "ok_hess": ok_hess, "spread_max": spread,
                "time_cap": capped, "n_inits_run": len(fits),
                "best_init": best, "fits": summ, "f_range": [min(fz["f"] for fz in fits), max(fz["f"] for fz in fits)]}


# ----------------------------------------------------------------------------- B limits
def B_unconstrained(Wlift):
    ev = float(np.linalg.eigvalsh(Wlift)[0])
    if ev < 0:
        Kp, _ = B.proj_psd(Wlift)
        return B.normalize_unit_diag(Kp), {"lam_min_lift": ev, "proj_active": True}
    return B.normalize_unit_diag(Wlift), {"lam_min_lift": ev, "proj_active": False}


def B_ball(Wlift, mu):
    Kf, info = B.ball_projection(Wlift, mu)
    return B.normalize_unit_diag(Kf), info


def B_EY(Wlift, p):
    Wr, info = B.eckart_young(Wlift, p)
    return B.normalize_unit_diag(Wr), info


# ----------------------------------------------------------------------------- summaries
def ties(K, S, tol):
    return Mt.ties_on(K, S, tol)


def summarize_pair(KA, KB, probes: dict):
    """Delta_T^inf on every S_lad rung, exact S_ex booleans and ties, Gs_max, 1-CKA, iso_inf, omega,
    spectrum statistics."""
    d = Mt.ladder_deltas(KA, KB)
    out = {"ladder": d}
    for name, (S, k) in probes.items():
        out[f"exact_{name}"] = Mt.exact_obs(KA, KB, S, k)
        for L, K in (("A", KA), ("B", KB)):
            out[f"ties_{name}_{L}_1e-9"] = ties(K, S, 1e-9)
            out[f"ties_{name}_{L}_1e-6"] = ties(K, S, 1e-6)
    out["Gs_max"] = d["Gs_max"]
    out["one_minus_cka"] = d["dCKA"]
    out["iso_inf_AB"], out["iso_inf_BA"] = d["iso_inf_AB"], d["iso_inf_BA"]
    out["omega"] = d["omega"]
    out["spec_A"] = Mt.spectrum_stats(KA)
    out["spec_B"] = Mt.spectrum_stats(KB)
    return out


def hess_bounds(pi_t, tau, K, Lfac, key):
    """Lanczos bounds of the preconditioned theta-Hessian at a convex A solution (learnA.theta_hess_bounds)."""
    return A.theta_hess_bounds(A.InfoNCE(pi_t, tau), _t(K, Lfac.device), Lfac, key)


def lemmaT_residual(K_lo, K_hi, alpha):
    """|| K_lo - (alpha K_hi + (1 - alpha) 11^T) ||_max (Lemma T identity check; not a claim)."""
    M = alpha * K_hi + (1.0 - alpha)
    np.fill_diagonal(M, 1.0)
    return float(np.max(np.abs(K_lo - M)))


def all_limits(inst, precond, cfg, probes: dict, dev, lam_I1: float, log=print):
    """Compute every population limit; returns (summary dict, arrays dict). lam_I1: frozen lower bound
    of lambda_min(I_1) (learnA.lam_min_I1), used by the Lemma T identity checks."""
    t = inst.t
    n = inst.n
    sc, pc = cfg["solver"], cfg["poplim"]
    key = f"{C.KEY}|poplim"
    pi_g = _t(inst.pi_gibbs(), dev)
    pi_a = _t(inst.pi_aug(), dev)
    res, arr = {}, {}

    def put(case, KA, KB, extra):
        arr[f"{case}_A"], arr[f"{case}_B"] = KA, KB
        s = summarize_pair(KA, KB, probes)
        s.update(extra)
        res[case] = s
        log(f"[poplim] {case}: dEk={s['ladder']['dEk']:.4g} dE={s['ladder']['dE']:.4g} sN10={s['ladder']['s_N_k10']:.4f} "
            f"Gs_max={s['Gs_max']:.4f} iso={s['iso_inf_AB']:.3g}")

    # arm 0 / R- / R+
    KA0, iA0 = convex_A(pi_g, t, precond, sc, key=f"{key}|0|A")
    KB0, iB0 = B_unconstrained(inst.W)
    put("0", KA0, KB0, {"info_A": iA0, "info_B": iB0, "errA_vs_Kstar": float(np.max(np.abs(KA0 - inst.K))),
                        "errB_vs_KBt": float(np.max(np.abs(KB0 - inst.KBt)))})
    KAm, iAm = convex_A(pi_g, t / 2.0, precond, sc, key=f"{key}|Rm|A")
    hb0 = hess_bounds(pi_g, t, KA0, precond.L, f"{key}|0|A|lanczos")
    hbm = hess_bounds(pi_g, t / 2.0, KAm, precond.L, f"{key}|Rm|A|lanczos")
    dRm = A.lemmaT_decision(KA0, iA0, hb0, KAm, iAm, hbm, 0.5, lam_I1)
    eRm, _ = A.err_bound_K(iAm, hbm, lam_I1)
    res["0"]["hess_bounds_A"] = hb0
    put("Rm", KAm, KA0, {"info_A": iAm, "note": "pair (A at t/2, A at t)", "hess_bounds_A": hbm,
                         "lemmaT": dRm, "lemmaT_resid_vs_computed": dRm["resid"],
                         "lemmaT_resid_vs_Kstar": lemmaT_residual(KAm, inst.K, 0.5),
                         "lemmaT_tol_vs_Kstar": eRm + A.FLOAT_FLOOR})
    KAp, iAp = convex_A(pi_g, 2.0 * t, precond, sc, key=f"{key}|Rp|A")
    put("Rp", KAp, KA0, {"info_A": iAp, "note": "pair (A at 2t, A at t); misspecified, descriptive",
                         "lam_min_affine_target": float(np.linalg.eigvalsh(2.0 * inst.K - 1.0)[0])})
    # (i) ball dose
    for mu in pc["mus"]:
        KBm, iBm = B_ball(inst.W, float(mu))
        put(f"i_mu{mu}", KA0, KBm, {"info_B": iBm})
    # (ii) rank dose: B exact, A non-convex (all non-convex fits share the wall-clock cap nonconvex_cap_h)
    _NC_DEADLINE[0] = time.time() + 3600.0 * float(pc.get("nonconvex_cap_h", float("inf")))
    for p in pc["ranks"]:
        KBp, iBp = B_EY(inst.W, int(p))
        KAp_, iAp_ = rank_A(pi_g, t, int(p), inst.K, f"{key}|ii_p{p}|A", pc, dev)
        put(f"ii_p{p}", KAp_, KBp, {"info_A": iAp_, "info_B": iBp})
    # (ii-tau): rank v_rank at tau = t / tau_div, vs the tau = t rank fit
    p_tau = int(pc["v_rank"])
    KAt_, iAt_ = rank_A(pi_g, t / pc["tau_div"], p_tau, inst.K, f"{key}|ii_tau|A", pc, dev)
    ref = arr.get(f"ii_p{p_tau}_A")
    if ref is None:
        ref, iref = rank_A(pi_g, t, p_tau, inst.K, f"{key}|ii_p{p_tau}|A", pc, dev)
    put("ii_tau", KAt_, ref, {"info_A": iAt_, "note": f"pair (rank-{p_tau} A at t/{pc['tau_div']}, rank-{p_tau} A at t); "
                              "rank-constrained: Lemma T does not apply, residual reported only",
                              "lemmaT_resid": lemmaT_residual(KAt_, ref, 1.0 / pc["tau_div"])})
    # (iv) two views
    KAiv, iAiv = convex_A(pi_a, t, precond, sc, key=f"{key}|iv|A")
    KBiv, iBiv = B_unconstrained(inst.W2)
    put("iv", KAiv, KBiv, {"info_A": iAiv, "info_B": iBiv})
    KAivt, iAivt = convex_A(pi_a, t / pc["tau_div"], precond, sc, key=f"{key}|iv_tau|A")
    hbiv = hess_bounds(pi_a, t, KAiv, precond.L, f"{key}|iv|A|lanczos")
    hbivt = hess_bounds(pi_a, t / pc["tau_div"], KAivt, precond.L, f"{key}|iv_tau|A|lanczos")
    div = A.lemmaT_decision(KAiv, iAiv, hbiv, KAivt, iAivt, hbivt, 1.0 / pc["tau_div"], lam_I1)
    res["iv"]["hess_bounds_A"] = hbiv
    put("iv_tau", KAivt, KAiv, {"info_A": iAivt, "note": f"pair (two-view A at t/{pc['tau_div']}, two-view A at t); "
                                "Lemma T identity check only if the t limit is PD with margin and unique",
                                "hess_bounds_A": hbivt, "lemmaT": div, "lemmaT_resid": div["resid"]})
    # (v) cumulative
    KBv, iBv = B_ball(inst.W2, 1.0)
    iBv["rank_le_v_rank"] = bool(iBv["rank_1e-8"] <= pc["v_rank"])
    KAv, iAv = rank_A(pi_a, t, int(pc["v_rank"]), KAiv, f"{key}|v|A", pc, dev)
    put("v", KAv, KBv, {"info_A": iAv, "info_B": iBv})
    return res, arr
