"""Gates G0 and G1 (deterministic, before any sample; PREREG sections 3.2 and 5.3) and the
population part of G1b. The sampled gates (G1b sampled part, G3, G4, D2) are evaluated by
analyze_A.py from worker outputs.

G1 items (known answers):
  g1_A_pop        A on the population risk from K_init = I recovers K* to <= 1e-9
  g1_B_pop        B's closed form at the population equals K*^(t) to <= 1e-12
  g1_ncm          (i) at mu = 1 equals Higham's nearest correlation matrix of W+ to <= 1e-8
  g1_ladder       hrhv2 ladder on (K*, K*^(t)): every order rung 0, iso_inf 0 (<= 1e-12),
                  Gs_max = omega_psi (<= 1e-12)
  g1_iso          metrics.iso_inf equals an LP solution on planted monotone / non-monotone pairs
  g1_bernstein    S2.1's Bernstein calibration warp crit.bern_geom(beta = 1/t) applied to K* equals
                  K*^(t) (<= 1e-12), and its self pair has every order rung 0
  g1_lemmaL       every order rung used by Pi / Lemma L (dEk, dO, dN_k, dMST) equals the direct
                  fixed-denominator average of disagreement indicators (one-sided ties 1/2) on random
                  plants (m = 24, 1000 plants; every second plant with tied distances); the tau_b rung
                  dE is shown to differ from dEk under ties (reported; it is secondary only)
Fallback (section 5.3): if g1_A_pop misses 1e-9 for a documented numerical reason, the tolerance is
1e-3 x (median per-entry sd of A at N_top).
G1b population part (Lemma T identity check, never a claim; binding correction (a)): learnA.lemmaT_decision
with PD margin, uniqueness (no Hessian null space) and the curvature-derived tolerance. A violation where
the lemma applies is a code error (pass = False); where it does not apply the residual is reported only.
"""
from __future__ import annotations

import numpy as np

import s23common as C
import instance as I
import learnB as B
import metrics as Mt

ORDER_KEYS_ZERO = ("dN_k1", "dN_k3", "dN_k10", "dN_k30", "dN_k100", "dN_kq", "dC_k1", "dC_k3", "dC_k10",
                   "dO", "dE", "dEk", "dMST", "dMST_forest", "dMST_bridge")


def max_order_delta(d: dict) -> float:
    """Largest |Delta| over the order rungs; NaN counts as failure (inf) except for rungs that can be
    undefined by construction: the forest/bridge split (no bridge or no forest edge) and k-NN rungs with
    k > m - 2 (only on the tiny test instances; every rung is defined for m = 256)."""
    w = 0.0
    for k in ORDER_KEYS_ZERO:
        v = float(d[k])
        if v != v:
            v = 0.0 if k in ("dMST_forest", "dMST_bridge", "dN_k30", "dN_k100") else float("inf")
        w = max(w, abs(v))
    return w


def g0_full(inst, g0: dict, S_ex8, thr: dict) -> dict:
    out = dict(g0)
    out["ex8_min_dgap"] = Mt_min_dgap(inst.K, S_ex8)
    out["pass_ex8_dgap"] = out["ex8_min_dgap"] >= thr["ex8_dgap"]
    out["pass_all"] = bool(out["pass_all_noprobe"] and out["pass_ex8_dgap"])
    return out


def Mt_min_dgap(K, S):
    S = np.asarray(S)
    D = np.sort((2.0 - 2.0 * K[np.ix_(S, S)])[np.triu_indices(len(S), 1)])
    return float(np.min(np.diff(D)))


def iso_lp(x, y):
    """inf_g max_e |g(x_e) - y_e| over nondecreasing g, by linear programming (independent of
    metrics.iso_inf_from_D). x, y: 1-D arrays."""
    from scipy.optimize import linprog
    xs = np.unique(x)
    G = len(xs)
    gid = np.searchsorted(xs, x)
    nv = G + 1                                   # g_1..g_G, s
    A, b = [], []
    for e in range(len(x)):
        r = np.zeros(nv); r[gid[e]] = 1.0; r[-1] = -1.0; A.append(r); b.append(y[e])      # g - y <= s
        r = np.zeros(nv); r[gid[e]] = -1.0; r[-1] = -1.0; A.append(r); b.append(-y[e])    # y - g <= s
    for i in range(G - 1):
        r = np.zeros(nv); r[i] = 1.0; r[i + 1] = -1.0; A.append(r); b.append(0.0)         # g_i <= g_{i+1}
    c = np.zeros(nv); c[-1] = 1.0
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(None, None)] * nv, method="highs")
    return float(res.fun)


def g1_iso(n_plants: int, key: str) -> dict:
    r = C.rng(key)
    worst = 0.0
    for j in range(n_plants):
        m = int(r.integers(5, 10))
        X = r.normal(size=(m, 3)); KA = X @ X.T
        if j % 3 == 0:                                    # planted monotone pair: B = h(A) increasing
            DA = Mt.gram_to_dist(KA)
            DB = np.sqrt(DA) + 0.3 * DA
        else:
            Y = r.normal(size=(m, 3)); DB = Mt.gram_to_dist(Y @ Y.T)
            DA = Mt.gram_to_dist(KA)
        if j % 5 == 1:                                    # planted ties in D_A
            DA = np.round(DA, 0)
        v = Mt.iso_inf_from_D(DA, DB)
        x = Mt.upper(DA); y = Mt.upper(DB); y = y / y.mean()
        lp = iso_lp(x, y)
        worst = max(worst, abs(v - lp))
        if j % 3 == 0 and not (j % 5 == 1) and v > 1e-12:
            worst = max(worst, v)
    return {"max_abs_diff_vs_lp": worst, "pass": bool(worst <= 1e-9)}


def _tied(D, r):
    """Distances rounded to a coarse grid (creates ties); symmetric, zero diagonal."""
    q = float(r.choice([0.25, 0.5, 1.0]))
    Dt = np.round(D / q) * q
    Dt = 0.5 * (Dt + Dt.T)
    np.fill_diagonal(Dt, 0.0)
    return Dt


def g1_lemmaL(n_plants: int, m: int, key: str) -> dict:
    """hrhv2 rungs vs direct fixed-denominator disagreement averages on random plants; odd plants have
    tied distances. dN_k: hrhv2 tie rule (D_ij, j) reproduced by the direct sort; dMST: continuous plants
    against an independent Kruskal tree, tie plants by the definition |T_A xor T_B| / (2(m-1)) on hrhv2's
    own trees."""
    r = C.rng(key)
    worst = {"dEk": 0.0, "dO": 0.0, "dN_k3": 0.0, "dN_k10": 0.0, "dMST": 0.0}
    tau_b_differs = 0
    n_tie_plants = 0
    from hrhv2 import structural as st
    for j in range(n_plants):
        X = r.normal(size=(m, 5)); Y = X + r.normal(size=(m, 5)) * float(r.uniform(0.05, 1.0))
        DA = Mt.gram_to_dist(X @ X.T); DB = Mt.gram_to_dist(Y @ Y.T)
        ties = (j % 2 == 1)
        if ties:
            DA, DB = _tied(DA, r), _tied(DB, r)
            n_tie_plants += 1
        nA, _ = st.knn_table(DA, 11); nB, _ = st.knn_table(DB, 11)
        go = st.global_order(DA, DB)
        TA, TB = st.mst_prim(DA), st.mst_prim(DB)
        h = {"dEk": go["globalorder_kendall_dist_halfties"],
             "dO": float(st.row_kendall(DA, DB, n_jobs=1)[0].mean()),
             "dN_k3": 1.0 - st.knn_overlap(nA, nB, 3), "dN_k10": 1.0 - st.knn_overlap(nA, nB, 10),
             "dMST": 1.0 - st.mst_compare(TA, TB, st.nn1_forest(nA[:, 0]))["s_T"]}
        d = {"dEk": Mt.direct_dEk(DA, DB), "dO": Mt.direct_dO(DA, DB), "dN_k3": Mt.direct_dNk(DA, DB, 3),
             "dN_k10": Mt.direct_dNk(DA, DB, 10),
             "dMST": (len(set(TA.tolist()) ^ set(TB.tolist())) / (2.0 * (m - 1))) if ties else Mt.direct_dMST(DA, DB)}
        for k in worst:
            worst[k] = max(worst[k], abs(h[k] - d[k]))
        if ties and abs(go["Delta_E"] - go["globalorder_kendall_dist_halfties"]) > 1e-12:
            tau_b_differs += 1
    return {"max_abs_diff": worst, "n_tie_plants": n_tie_plants,
            "n_tie_plants_where_tau_b_rung_differs": tau_b_differs,
            "pass": bool(max(worst.values()) <= 1e-12)}


def g1(inst, KA_pop, cfg, sd_entry_top: float, ladder_star: dict) -> dict:
    """G1 known answers. KA_pop: A's certified population solution from K_init = I (poplim case '0')."""
    tc = cfg["tests"]
    out = {}
    eA = float(np.max(np.abs(KA_pop - inst.K)))
    fb = 1e-3 * sd_entry_top
    out["g1_A_pop"] = {"err": eA, "tol": tc["g1_A_tol"], "fallback_tol": fb,
                       "pass": bool(eA <= tc["g1_A_tol"]), "pass_fallback": bool(eA <= fb)}
    Q = inst.Q
    PP = np.outer(inst.P, inst.P)
    _, KBn, ib = B.solve_B(Q, PP)
    eB = float(np.max(np.abs(KBn - inst.KBt)))
    out["g1_B_pop"] = {"err": eB, "path": ib["path"], "pass": bool(eB <= tc["g1_B_tol"])}
    Kball, iball = B.ball_projection(inst.W, 1.0)
    Kncm, incm = B.ncm_higham(inst.W)
    e_ncm = float(np.max(np.abs(Kball - Kncm)))
    out["g1_ncm"] = {"err": e_ncm, "ball_info": iball, "higham_info": incm, "pass": bool(e_ncm <= tc["g1_ncm_tol"])}
    d = ladder_star
    worst_order = max_order_delta(d)
    om = I.omega_psi(inst.K, inst.t)
    out["g1_ladder"] = {"max_order_delta": worst_order, "iso_inf_AB": d["iso_inf_AB"], "iso_inf_BA": d["iso_inf_BA"],
                        "Gs_max": d["Gs_max"], "omega_psi": om,
                        "pass": bool(worst_order == 0.0 and d["iso_inf_AB"] <= tc["g1_exact_tol"]
                                     and d["iso_inf_BA"] <= tc["g1_exact_tol"]
                                     and abs(d["Gs_max"] - om) <= tc["g1_exact_tol"])}
    out["g1_iso"] = g1_iso(tc["iso_plants"], f"{C.KEY}|g1|iso")
    crit = C.load_crit()
    Gb = crit.bern_geom(Mt.geom(inst.K), beta=1.0 / inst.t)
    eb = float(np.max(np.abs(Gb.K - inst.KBt)))
    dself = Mt.ladder_deltas(inst.K, Gb.K)
    wb = max_order_delta(dself)
    out["g1_bernstein"] = {"err": eb, "self_max_order_delta": wb, "pass": bool(eb <= tc["g1_exact_tol"] and wb == 0.0)}
    out["g1_lemmaL"] = g1_lemmaL(tc["lemmaL_plants"], tc["lemmaL_m"], f"{C.KEY}|g1|lemmaL")
    items = ["g1_A_pop", "g1_B_pop", "g1_ncm", "g1_ladder", "g1_iso", "g1_bernstein", "g1_lemmaL"]
    out["pass_all"] = bool(all(out[k]["pass"] for k in items))
    out["pass_all_with_fallback"] = bool(all(out[k]["pass"] for k in items if k != "g1_A_pop")
                                         and (out["g1_A_pop"]["pass"] or out["g1_A_pop"]["pass_fallback"]))
    return out


def g1b_population(poplim_res: dict) -> dict:
    """Population part of G1b: Lemma T identity checks (never a claim). Uses the decisions computed in
    poplim.all_limits (learnA.lemmaT_decision). Rm: (A at t/2) vs (A at t) and vs the exact
    (1/2) K* + (1/2) 11^T; iv_tau: (two-view A at t/2.5) vs (two-view A at t) -- the lemma applies only
    if the two-view limit at t is PD with margin and unique; otherwise the residual is reported and the
    case is descriptive (correction (b): a missing data temperature CAN break the rescaling, since with
    two-view positives a PD minimizer need not exist). ii_tau (rank-constrained) is never covered by the
    lemma: residual reported only."""
    out = {}
    rm = poplim_res["Rm"]["lemmaT"]
    out["Rm"] = rm
    out["Rm_resid_vs_Kstar"] = poplim_res["Rm"]["lemmaT_resid_vs_Kstar"]
    out["Rm_tol_vs_Kstar"] = poplim_res["Rm"]["lemmaT_tol_vs_Kstar"]
    iv = poplim_res["iv_tau"]["lemmaT"]
    out["iv_tau"] = iv
    out["ii_tau_resid_reported"] = poplim_res["ii_tau"]["lemmaT_resid"]
    out["pass_Rm"] = bool(rm["applicable"] and not rm["violation"]
                          and out["Rm_resid_vs_Kstar"] <= out["Rm_tol_vs_Kstar"])
    out["iv_lemmaT_applicable"] = bool(iv["applicable"])
    out["pass_iv"] = bool(not iv["violation"])
    out["pass"] = bool(out["pass_Rm"] and out["pass_iv"])
    return out
