"""Stage A analysis (PREREG sections 6.3, 7, 8.3, 9, 10, 13). Phases, run in this order:

  gates  : pilot tolerances -> results/stageA/eps_arm0.json (hashed into STATUS before any main-run
           ladder); G3 (solver agreement), G1b sampled part (Lemma T identity check, reported),
           G4 (timing, projection, frozen reductions), D2 (descriptive) -> gates_sampled.json,
           g4_decision.json (main_allowed).
  nmatch : within-A s_N(10) on S_lad at every executed grid N (no cross statistic is opened)
           -> N_match.json, hashed into STATUS.
  main   : every remaining pair set; F0 (9 tests, Holm), guards, calibration and planted controls,
           order verdict, secondary items (P1b, P2b, P2c, P4r, P5b, P7, P8), stop rule
           -> stageA_results.json + REPORT_stageA.md, hashed into STATUS.

  python analyze_A.py --phase gates|nmatch|main
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import s23common as C

if __name__ == "__main__":
    C.cap_threads(1)

import numpy as np
from scipy.stats import chi2, norm

import learnB as B
import metrics as Mt
import stats as S

crit = C.load_crit()
TOL_KEYS = list(crit.TOL_KEYS) + ["dEk", "iso_inf_AB"]       # dEk: fixed-denominator T_E rung (correction (d))
ORDER = list(Mt.ORDER_RUNGS)


# ============================================================================ storage
def rid(arm, L, l2, rep, pilot=False):
    return f"{arm}_{L}_l{l2}_r{rep:03d}{'_pilot' if pilot else ''}"


class Store:
    def __init__(self, plan, fr, arr):
        self.items = {it["id"]: it for it in plan["items"]}
        self.fr, self.arr = fr, arr
        self.t = fr["cfg"]["t"]
        self.Z = fr["instance"]["Z"]
        self._K = {}
        self._F = {}

    def paths(self, r):
        it = self.items[r]
        d = os.path.join(C.RUNS, it["cell"])
        return os.path.join(d, r + ".npy"), os.path.join(d, r + ".json"), os.path.join(d, r + ".done"), d

    def done(self, r):
        return r in self.items and os.path.exists(self.paths(r)[2])

    def info(self, r):
        return C.load_json(self.paths(r)[1])

    def failed(self, r):
        if not self.done(r):
            return True
        inf = self.info(r)
        if "t" in inf and "t_half" in inf:
            return not (inf["t"].get("cert_pass") and inf["t_half"].get("cert_pass"))
        return not bool(inf.get("cert_pass", False)) or not os.path.exists(self.paths(r)[0])

    def K(self, r, extra=None):
        key = (r, extra)
        if key not in self._K:
            pk, _, _, d = self.paths(r)
            p = pk if extra is None else os.path.join(d, f"{r}__{extra}.npy")
            self._K[key] = np.load(p) if os.path.exists(p) else None
            if len(self._K) > 1500:
                self._K.clear()
        return self._K[key]

    def transformed(self, r, tr):
        K = self.K(r)
        if K is None or tr is None:
            return K
        if tr == "phi":
            return Mt.phi_t(K, self.t)
        if tr == "half":
            return Mt.rescale_half(K)
        if tr.startswith("F"):
            mu = float(tr[1:])
            key = (r, mu)
            if key not in self._F:
                cdir = os.path.join(C.STAGEA_DIR, "cache", "F")
                os.makedirs(cdir, exist_ok=True)
                p = os.path.join(cdir, f"{r}__F{tr[1:]}.npy")
                if os.path.exists(p):
                    self._F[key] = np.load(p)
                else:
                    Kf, _ = B.ball_projection(np.exp(K / self.t) / self.Z, mu)
                    Kn = B.normalize_unit_diag(Kf)
                    np.save(p, Kn)
                    self._F[key] = Kn
            return self._F[key]
        raise ValueError(tr)


def pair_rows(st: Store, name: str, pairs, S8, S16, ladder=True):
    """pairs: list of ((rid_a, tr_a), (rid_b, tr_b)). Cached per name in results/stageA/cache/."""
    cpath = os.path.join(C.STAGEA_DIR, "cache", f"{name}.json")
    if os.path.exists(cpath):
        rows = C.load_json(cpath)
        if len(rows) == len(pairs) and all(r_["a"] == a[0] and r_["b"] == b[0] for r_, (a, b) in zip(rows, pairs)):
            return rows
    rows = []
    for (a, tra), (b, trb) in pairs:
        fa, fb = st.failed(a), st.failed(b)
        Ka, Kb = st.transformed(a, tra), st.transformed(b, trb)
        row = {"a": a, "b": b, "tra": tra, "trb": trb, "failed": bool(fa or fb)}
        if Ka is None or Kb is None:
            row.update(d={}, ex8={}, ex16={}, maxdiff=float("nan"))
        else:
            row["d"] = Mt.ladder_deltas(Ka, Kb) if ladder else {}
            row["ex8"] = Mt.exact_obs(Ka, Kb, S8, 2)
            row["ex16"] = Mt.exact_obs(Ka, Kb, S16, 3)
            row["maxdiff"] = float(np.max(np.abs(Ka - Kb)))
        rows.append(row)
    C.dump_json(rows, cpath)
    return [C.load_json(cpath)][0]


def val(row, key):
    v = row.get("d", {}).get(key, float("nan"))
    return float(C.fnum(v))


def exv(row, which, obs):
    """0 if the exact observable agrees, 1 otherwise (NaN if missing)."""
    e = row.get(which, {})
    return float("nan") if obs not in e else (0.0 if e[obs] else 1.0)


# ============================================================================ failure-aware statistics
def against_values(vals, failed, predict):
    """Section 5.4: a failed pair counts against the test's prediction.
    predict 'reliable' -> +inf ; 'unreliable' -> -inf (i.e. inside every tolerance)."""
    v = np.array(vals, dtype=float)
    bad = np.array(failed, bool) | ~np.isfinite(v)
    v[bad] = np.inf if predict == "reliable" else -np.inf
    return v


def decide(vals, failed, eps, predict, eps2=None):
    v = against_values(vals, failed, predict)
    row = S.decision_row(v, eps, eps2)
    row["predict"] = predict
    row["n_failed"] = int(np.sum(failed))
    row["p"] = row["p_rel"] if predict == "reliable" else row["p_unrel"]
    row["holds"] = row["decision"] == predict
    return row


def pi_with_fail(ab, aa, bb, fab, faa, fbb, key, predict):
    """Pi = mean_AB - (mean_AA + mean_BB); failed pairs replaced by the worst value of their own pair
    set in the direction against the prediction ('no_plateau' or 'plateau')."""
    def adj(v, f, worst_high):
        v = np.array(v, float)
        bad = np.array(f, bool) | ~np.isfinite(v)
        ok = v[~bad]
        if bad.any():
            fill = (ok.max() if worst_high else ok.min()) if ok.size else (np.inf if worst_high else 0.0)
            v[bad] = fill
        return v
    hi = predict == "no_plateau"
    r = S.pi_test(adj(ab, fab, hi), adj(aa, faa, not hi), adj(bb, fbb, not hi), key)
    r["predict"] = predict
    r["n_failed"] = int(np.sum(fab) + np.sum(faa) + np.sum(fbb))
    r["p"] = r["p_noplateau"] if predict == "no_plateau" else r["p_plateau"]
    r["holds"] = bool(r["no_plateau"] if predict == "no_plateau" else r["plateau"])
    return r


def ratio_with_fail(e_hi, f_hi, e_lo, f_lo, key, tc):
    e_hi = np.array(e_hi, float); e_lo = np.array(e_lo, float)
    e_hi[np.array(f_hi, bool) | ~np.isfinite(e_hi)] = np.inf
    e_lo[np.array(f_lo, bool) | ~np.isfinite(e_lo)] = 0.0
    r = S.ratio_median_boot(e_hi, e_lo, key, thr_conv=tc["conv"], thr_plat=tc["plat"])
    r["p"] = r["p_converging"]
    r["holds"] = r["converging"]
    return r


# ============================================================================ cell helpers
class Arm0:
    def __init__(self, st: Store, fr, l2, nrep=None):
        self.st, self.fr, self.l2 = st, fr, l2
        R = 0
        maxr = fr["cfg"]["runs"]["main"] if nrep is None else nrep
        while R < maxr and st.done(rid("0", "A", l2, R)) and st.done(rid("0", "B", l2, R)):
            R += 1
        self.R = R - (R % 2)
        self.nb = self.R // 2
        self.S8 = fr["probes"]["S_ex8"]
        self.S16 = fr["probes"]["S_ex16"]

    def A(self, i):
        return rid("0", "A", self.l2, i)

    def B(self, i):
        return rid("0", "B", self.l2, i)

    def rows(self, which):
        st, l2, nb = self.st, self.l2, self.nb
        P = {
            "wA": [((self.A(2 * j), None), (self.A(2 * j + 1), None)) for j in range(nb)],
            "wB": [((self.B(2 * j), None), (self.B(2 * j + 1), None)) for j in range(nb)],
            "AB": [((self.A(2 * j), None), (self.B(2 * j), None)) for j in range(nb)],
            "link_self": [((self.A(2 * j), None), (self.A(2 * j), "phi")) for j in range(nb)],
            "link_peer": [((self.A(2 * j), None), (self.A(2 * j + 1), "phi")) for j in range(nb)],
            "w_phi": [((self.A(2 * j), "phi"), (self.A(2 * j + 1), "phi")) for j in range(nb)],
            "rescale_self": [((self.A(2 * j), None), (self.A(2 * j), "half")) for j in range(nb)],
            "plant1": [((self.A(2 * j), None), (self.A(2 * j + 1), "F1")) for j in range(nb)],
            "w_F1": [((self.A(2 * j), "F1"), (self.A(2 * j + 1), "F1")) for j in range(nb)],
            "plant4": [((self.A(2 * j), None), (self.A(2 * j + 1), "F4")) for j in range(nb)],
            "w_F4": [((self.A(2 * j), "F4"), (self.A(2 * j + 1), "F4")) for j in range(nb)],
        }[which]
        return pair_rows(st, f"arm0_l{l2}_{which}_n{nb}", P, self.S8, self.S16)

    def err(self, L):
        Kinf = self.st.arr["Kstar"] if L == "A" else self.st.arr["KBt"]
        e, f = [], []
        for i in range(self.R):
            r = self.A(i) if L == "A" else self.B(i)
            K = self.st.K(r)
            e.append(float(np.max(np.abs(K - Kinf))) if K is not None else float("nan"))
            f.append(self.st.failed(r))
        return np.array(e), np.array(f)


def fails(rows):
    return [r_["failed"] for r_ in rows]


def vals(rows, key):
    return [val(r_, key) for r_ in rows]


def exvals(rows, which, obs):
    return [exv(r_, which, obs) for r_ in rows]


def means(rows, key):
    v = np.array(vals(rows, key), float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


# ============================================================================ phase: gates
def phase_gates(st: Store, plan, fr):
    cfg = fr["cfg"]
    g = fr["grid"]
    tc = cfg["tests"]
    S8, S16 = fr["probes"]["S_ex8"], fr["probes"]["S_ex16"]
    npil = cfg["runs"]["pilot"] // 2
    eps = {"version": "eps_arm0 v1.0", "pilot_pairs_per_learner": npil, "per_N": {}}
    maxdiff_med = {}
    for l2 in g["log2N_pilot"]:
        e = {}
        for L in ("A", "B"):
            P = [((rid("0", L, l2, 2 * j, True), None), (rid("0", L, l2, 2 * j + 1, True), None)) for j in range(npil)]
            rows = pair_rows(st, f"pilot_{L}_l{l2}", P, S8, S16)
            e[L] = {}
            for k in TOL_KEYS + ["Gs_max", "s_N_k10"]:
                v = np.array(vals(rows, k), float)
                v = v[np.isfinite(v)]
                e[L][k] = {"max": float(v.max()) if v.size else float("nan"),
                           "q90": float(np.quantile(v, 0.9)) if v.size else float("nan")}
            for obs in ("T_E", "T_O", "T_N2", "T_MST"):
                e[L][f"ex8_{obs}_agree"] = float(np.nanmean([1 - x for x in exvals(rows, "ex8", obs)]))
            md = np.array([r_["maxdiff"] for r_ in rows], float)
            maxdiff_med[(l2, L)] = float(np.nanmedian(md))
            e[L]["maxdiff_median"] = maxdiff_med[(l2, L)]
            e[L]["n_failed_pairs"] = int(np.sum(fails(rows)))
        e["AB"] = {k: {"max": max(e["A"][k]["max"], e["B"][k]["max"]), "q90": max(e["A"][k]["q90"], e["B"][k]["q90"])}
                   for k in TOL_KEYS + ["Gs_max"]}
        eps["per_N"][str(l2)] = e
    eps_path = os.path.join(C.STAGEA_DIR, "eps_arm0.json")
    h_eps = C.dump_json(eps, eps_path)
    os.chmod(eps_path, 0o444)
    C.append_status("run", f"S2.3 Stage A pilot tolerances written BEFORE any main-run ladder: results/stageA/eps_arm0.json "
                    f"sha256 {h_eps} (64 pilot pairs per learner at log2N {g['log2N_pilot']}; pilots excluded from every count)")
    out = {"eps_sha256": h_eps}
    # ---- G3
    g3 = {"items": [], "pass": True}
    for l2 in g["log2N_pilot"]:
        for L in ("A", "B"):
            thr = tc["g3_frac"] * maxdiff_med[(l2, L)]
            for rep in range(cfg["runs"]["g3"]):
                r1 = rid("0", L, l2, rep, True)
                r2 = rid("G3", L, l2, rep)
                K1, K2 = st.K(r1), st.K(r2)
                d = float(np.max(np.abs(K1 - K2))) if (K1 is not None and K2 is not None) else float("inf")
                ok = bool(d <= thr and not st.failed(r1) and not st.failed(r2))
                g3["items"].append({"log2N": l2, "L": L, "rep": rep, "maxdiff": d, "thr": thr, "pass": ok,
                                    "path_ref": st.info(r1).get("path") if st.done(r1) else None,
                                    "path_g3": st.info(r2).get("path") if st.done(r2) else None})
                g3["pass"] = g3["pass"] and ok
    out["G3"] = g3
    # ---- G1b sampled (Lemma T identity check, never a claim; binding correction (a)): applicable only
    # where ERM_t is interior, PD with margin and unique (no Hessian null space); tolerance derived from the
    # curvature (learnA.lemmaT_decision). A violation where it applies is a code error -> no main runs.
    # Certificate failures at N0 and N_top (interior regime) block the main runs; at N0/4096 (PSD-active
    # regime, BM) they are reported only (gate G3 governs solver accuracy there).
    g1b = {"items": [], "solver_failure": False, "solver_failure_small_N": False, "lemmaT_violation": False,
           "n_applicable": 0}
    for off in cfg["runs"]["g1b_offsets"]:
        l2 = g["log2N0"] + off
        for rep in range(cfg["runs"]["g1b"]):
            r = rid("G1b", "AAp", l2, rep)
            if not st.done(r):
                g1b["solver_failure"] = True
                g1b["items"].append({"log2N": l2, "rep": rep, "missing": True})
                continue
            big = l2 >= g["log2N0"]
            inf = st.info(r)
            dec = inf["lemmaT"]
            Kt, Kh = st.K(r), st.K(r, "t_half")
            M = 0.5 * Kt + 0.5
            np.fill_diagonal(M, 1.0)
            res = float(np.max(np.abs(Kh - M)))                 # recomputed from the stored matrices
            if not dec["certs"]:
                g1b["solver_failure" if big else "solver_failure_small_N"] = True
            if dec["violation"] or (dec["applicable"] and not res <= dec["tol"]):
                g1b["lemmaT_violation"] = True
            g1b["n_applicable"] += int(dec["applicable"])
            g1b["items"].append({"log2N": l2, "rep": rep, "resid": res, "resid_worker": dec["resid"], "tol": dec["tol"],
                                 "applicable": dec["applicable"], "pd": dec["pd"], "unique": dec["unique"],
                                 "lam_min_K_t": dec["lam_min_K_t1"], "pd_margin": dec["pd_margin"],
                                 "e_t": dec["e_t1"], "e_t2": dec["e_t2"], "certs": dec["certs"],
                                 "violation": dec["violation"], "path_t": inf["t"].get("path"),
                                 "path_t2": inf["t_half"].get("path"),
                                 "lam_min_M_t": (inf.get("hess_t") or {}).get("lam_min_M")})
    out["G1b"] = g1b
    # ---- D2 (descriptive)
    d2 = []
    for l2 in cfg["runs"]["d2_log2N"]:
        for rep in range(cfg["runs"]["d2"]):
            r = rid("D2", "B", l2, rep)
            if st.done(r):
                inf = st.info(r)
                d2.append({"log2N": l2, "rep": rep, "max_abs": inf.get("max_abs_HHt_minus_ERM"),
                           "rel_max": inf.get("rel_max"), "n_div": inf.get("trainH", {}).get("n_divergences"),
                           "lr_used": inf.get("trainH", {}).get("lr_used")})
    out["D2"] = d2
    # ---- G4 timing and projection
    times = {}
    for l2 in g["log2N_pilot"]:
        ts = [st.info(rid("0", "A", l2, r_, True)).get("gpu_section_s", np.nan) for r_ in range(cfg["runs"]["pilot"])
              if st.done(rid("0", "A", l2, r_, True))]
        times[l2] = float(np.nanmedian(ts)) if ts else float("nan")
    lo_t, hi_t = times[g["log2N_pilot"][0]], times[g["log2N_pilot"][1]]
    bm_cut = g["log2N0"] - 6                   # N < N0/64: PSD-active (BM) regime timing

    # R- (tau = t/2) in the PSD-active regime: median G1b t/2 solve time at N0/4096 (A at t/2 on small N)
    rm_ts = []
    for rep in range(cfg["runs"]["g1b"]):
        r = rid("G1b", "AAp", g["log2N0"] - 12, rep)
        if st.done(r):
            rm_ts.append(float(st.info(r)["t_half"].get("time_s", np.nan)))
    rm_t = float(np.nanmedian(rm_ts)) if rm_ts else lo_t
    times["Rm_t_half_N0_4096"] = rm_t

    def proj(red):
        """Projected GPU-hours: ledger + main-phase A solves; PSD-active (BM) solves counted at 1.5x their
        pilot / G1b median (BM evaluations grow towards N0/64), interior solves at the N_top median."""
        tot = 0.0
        for it in plan["items"]:
            if it["phase"] != "main" or it["kind"] != "A":
                continue
            if any(x in red and it.get(f"red_{x}") for x in ("a", "b", "c")):
                continue
            if it["arm"] == "Rm":
                tot += 1.5 * rm_t
            else:
                tot += 1.5 * lo_t if it["log2N"] < bm_cut else hi_t
        return C.gpu_hours_used() + tot / 3600.0

    red, budget = [], C.GPU_SOFT_STOP_H - 0.5
    p0 = proj(red)
    pr = {"none": p0}
    for nxt in ("a", "b", "c"):                 # frozen order of reductions (section 5.3)
        if pr[list(pr)[-1]] <= budget:
            break
        red = red + [nxt]
        pr["+".join(red)] = proj(red)
    final = proj(red)
    main_allowed = bool(g3["pass"] and not g1b["solver_failure"] and not g1b["lemmaT_violation"]
                        and final <= budget and fr.get("gates_pass", False))
    g4 = {"median_gpu_s_per_A": {str(k): v for k, v in times.items()}, "ledger_now_h": C.gpu_hours_used(),
          "projected_total_h": pr, "reductions": red, "projected_final_h": final, "budget_h": budget,
          "main_allowed": main_allowed,
          "reasons": {"G3": g3["pass"], "G1b_no_solver_failure": not g1b["solver_failure"],
                      "G1b_no_lemmaT_violation": not g1b["lemmaT_violation"],
                      "projection_ok": final <= budget, "prestep_gates": fr.get("gates_pass", False)}}
    out["G4"] = g4
    hg = C.dump_json(out, os.path.join(C.STAGEA_DIR, "gates_sampled.json"))
    hd = C.dump_json(g4, os.path.join(C.STAGEA_DIR, "g4_decision.json"))
    C.append_status("run", f"S2.3 Stage A sampled gates: G3 pass={g3['pass']}; G1b solver_failure={g1b['solver_failure']} "
                    f"lemmaT_violation={g1b['lemmaT_violation']} (identity check only; applicable in {g1b['n_applicable']} samples); G4 median GPU s/A-solve "
                    f"{ {k: round(v, 2) for k, v in times.items()} }, projected {final:.2f} GPU-h, reductions {red}; "
                    f"main_allowed={main_allowed}; gates_sampled.json sha256 {hg}; g4_decision.json sha256 {hd}")
    return out


# ============================================================================ phase: nmatch
def executed_grid(st, fr):
    out = []
    for l2 in fr["grid"]["grid"]:
        a = Arm0(st, fr, l2)
        if a.nb > 0:
            out.append(l2)
    return out


def phase_nmatch(st: Store, fr):
    tc = fr["cfg"]["tests"]
    tab = {}
    for l2 in executed_grid(st, fr):
        a = Arm0(st, fr, l2)
        rows = a.rows("wA")
        v = np.array(vals(rows, "s_N_k10"), float)
        tab[str(l2)] = {"mean_sN10_withinA": float(np.nanmean(v)), "n_pairs": int(np.isfinite(v).sum())}
    tgt = tc["nmatch_target"]
    best = min(tab, key=lambda k: (abs(tab[k]["mean_sN10_withinA"] - tgt), int(k)))
    out = {"target_sN10": tgt, "table": tab, "log2N_match": int(best),
           "note": "computed from within-A pairs only; no cross statistic opened"}
    p = os.path.join(C.STAGEA_DIR, "N_match.json")
    h = C.dump_json(out, p)
    os.chmod(p, 0o444)
    C.append_status("run", f"S2.3 Stage A N_match written BEFORE any cross statistic at that N: results/stageA/N_match.json "
                    f"sha256 {h}; log2 N_match={best} (within-A mean s_N(10) {tab[best]['mean_sN10_withinA']:.4f} vs 0.715)")
    return out


# ============================================================================ phase: main
def guards_geo(a: Arm0, eps):
    wA, wB = a.rows("wA"), a.rows("wB")
    out = {}
    for L, rows in (("A", wA), ("B", wB)):
        out[f"Gs_max_{L}"] = decide(vals(rows, "Gs_max"), fails(rows), eps["eps_geo"], "reliable")
        out[f"dCKA_{L}"] = decide(vals(rows, "dCKA"), fails(rows), eps["eps_cka"], "reliable")
    out["holds"] = all(v["holds"] for v in out.values())
    return out


def guard_TE(a: Arm0):
    out = {}
    for L, rows in (("A", a.rows("wA")), ("B", a.rows("wB"))):
        out[f"TE_{L}"] = decide(exvals(rows, "ex8", "T_E"), fails(rows), 0.0, "reliable")
    out["holds"] = all(v["holds"] for v in out.values())
    return out


def l_within(a0: Arm0, at: Arm0, tc):
    out = {}
    ok = True
    for L, w in (("A", "wA"), ("B", "wB")):
        r0, rt = a0.rows(w), at.rows(w)
        for r_ in ORDER:
            m0 = float(np.nanmedian(vals(r0, r_))); mt = float(np.nanmedian(vals(rt, r_)))
            good = (m0 == 0.0 and mt == 0.0) or (m0 > 0 and mt / m0 < tc["conv"])
            out[f"{L}_{r_}"] = {"median_N0": m0, "median_Ntop": mt, "ratio": (mt / m0) if m0 > 0 else float("nan"), "ok": good}
            ok = ok and good
    out["holds"] = ok
    return out


def p1b(st, fr, cells):
    """[Thm+CLT] for A, [Model] for B: standardized variances and means of frozen contrasts."""
    cs = fr["p1b"]["contrasts"]
    n_checks = 2 * len(cs) * len(cells) * 2
    lo, hi = None, None
    res = {"checks": [], "n_checks": n_checks}
    allok = True
    Ks, KBt = st.arr["Kstar"], st.arr["KBt"]
    for a in cells:
        N = 2.0 ** a.l2
        for L in ("A", "B"):
            Kinf = Ks if L == "A" else KBt
            runs = [a.A(i) if L == "A" else a.B(i) for i in range(a.R)]
            Ks_ = [st.K(r) for r in runs if not st.failed(r)]
            n = len(Ks_)
            if n < 3:
                allok = False
                continue
            alpha = 0.05 / n_checks
            lo_b, hi_b = chi2.ppf(alpha / 2, n - 1) / (n - 1), chi2.ppf(1 - alpha / 2, n - 1) / (n - 1)
            z = norm.ppf(1 - 0.025 / n_checks)
            for c in cs:
                x, y = c["plus"]
                v = np.array([K[x, y] - Kinf[x, y] for K in Ks_])
                if c["minus"] is not None:
                    u, w = c["minus"]
                    v = v - np.array([K[u, w] - Kinf[u, w] for K in Ks_])
                v = math.sqrt(N) * v
                var_m = c["var_A"] if L == "A" else c["var_B"]
                ratio = float(np.var(v, ddof=1) / var_m)
                mean_ok = bool(abs(v.mean()) <= z * math.sqrt(var_m) / math.sqrt(n))
                var_ok = bool(lo_b <= ratio <= hi_b)
                allok = allok and mean_ok and var_ok
                res["checks"].append({"log2N": a.l2, "L": L, "kind": c["kind"], "var_ratio": ratio, "band": [lo_b, hi_b],
                                      "var_ok": var_ok, "mean_z": float(v.mean() / (math.sqrt(var_m) / math.sqrt(n))),
                                      "mean_ok": mean_ok})
    res["holds"] = allok
    return res


def phase_main(st: Store, fr):
    cfg, tc = fr["cfg"], fr["cfg"]["tests"]
    g = fr["grid"]
    eps = fr["eps"]
    l0, ltop = g["log2N0"], g["log2N_top"]
    nm = C.load_json(os.path.join(C.STAGEA_DIR, "N_match.json"))
    epsp = C.load_json(os.path.join(C.STAGEA_DIR, "eps_arm0.json"))
    out = {"version": "S2.3 Stage A results v1.0", "eps": eps, "grid": g, "N_match": nm}
    grid = executed_grid(st, fr)
    cells = {l2: Arm0(st, fr, l2) for l2 in grid}
    a0, at = cells[l0], cells[ltop]
    # ---------------------------------------------------------------- F0
    F0 = {}
    eA0, fA0 = a0.err("A"); eAt, fAt = at.err("A")
    eB0, fB0 = a0.err("B"); eBt, fBt = at.err("B")
    F0["P1_A"] = ratio_with_fail(eAt, fAt, eA0, fA0, f"{C.KEY}|boot|P1_A", tc)
    F0["P1_B"] = ratio_with_fail(eBt, fBt, eB0, fB0, f"{C.KEY}|boot|P1_B", tc)
    ABt, wAt, wBt = at.rows("AB"), at.rows("wA"), at.rows("wB")
    AB0 = a0.rows("AB")
    gte = guard_TE(at)
    ggeo = {l0: guards_geo(a0, eps), ltop: guards_geo(at, eps)}
    p2 = decide(exvals(ABt, "ex8", "T_E"), fails(ABt), 0.0, "reliable")
    p2["guard"] = gte["holds"]
    F0["P2"] = p2
    p3a = [decide(vals(r_, "Gs_max"), fails(r_), eps["eps_geo"], "unreliable") for r_ in (AB0, ABt)]
    F0["P3a"] = {"N0": p3a[0], "Ntop": p3a[1], "p": max(p3a[0]["p"], p3a[1]["p"]),
                 "holds": bool(p3a[0]["holds"] and p3a[1]["holds"]),
                 "guard": bool(ggeo[l0]["holds"] and ggeo[ltop]["holds"])}
    band = (eps["one_minus_cka_inf"] - tc["cka_band"], eps["one_minus_cka_inf"] + tc["cka_band"])
    cab = against_values(vals(ABt, "dCKA"), fails(ABt), "reliable")
    cab[~np.isfinite(cab)] = band[1] + 1.0
    tost = S.tost_mean_boot(cab, band[0], band[1], f"{C.KEY}|boot|P3b_tost")
    wa = against_values(vals(wAt, "dCKA"), fails(wAt), "reliable"); wa[~np.isfinite(wa)] = 1.0
    wb = against_values(vals(wBt, "dCKA"), fails(wBt), "reliable"); wb[~np.isfinite(wb)] = 1.0
    ua = S.upper_mean_boot(wa, tc["within_cka"], f"{C.KEY}|boot|P3b_wA")
    ub = S.upper_mean_boot(wb, tc["within_cka"], f"{C.KEY}|boot|P3b_wB")
    pP3b = max(tost["p"], ua["p"], ub["p"])
    F0["P3b"] = {"tost": tost, "within_A": ua, "within_B": ub, "band": list(band), "p": pP3b,
                 "holds": bool(pP3b < tc["alpha"])}
    cfs = {}
    for name, a in (("N0", a0), ("Ntop", at)):
        rows = a.rows("AB")
        cv, frac = [], []
        for r_ in rows:
            Ka, Kb = st.K(r_["a"]), st.K(r_["b"])
            if Ka is None or Kb is None:
                cv.append(float("nan")); frac.append(float("nan")); continue
            c_, fr_ = Mt.inv_link_cf(Ka, Kb, st.t)
            cv.append(c_); frac.append(fr_)
        cfs[name] = (np.array(cv), np.array(fails(rows)), float(np.nanmean(frac)))
    F0["P3c"] = ratio_with_fail(cfs["Ntop"][0], cfs["Ntop"][1], cfs["N0"][0], cfs["N0"][1], f"{C.KEY}|boot|P3c", tc)
    F0["P3c"]["frac_KB_nonpos"] = {"N0": cfs["N0"][2], "Ntop": cfs["Ntop"][2]}
    p4a = decide(vals(ABt, "iso_inf_AB"), fails(ABt), eps["eps_geo"], "reliable")
    p4a["guard"] = ggeo[ltop]["holds"]
    F0["P4a"] = p4a
    conf = fr["p5_rungs"]["confirmatory"]
    pis = {r_: pi_with_fail(vals(ABt, r_), vals(wAt, r_), vals(wBt, r_), fails(ABt), fails(wAt), fails(wBt),
                            f"{C.KEY}|boot|P5|{r_}", "no_plateau") for r_ in ORDER}
    falsified = [r_ for r_ in ORDER if pis[r_]["plateau"]]
    F0["P5_order"] = {"rungs": pis, "confirmatory": conf, "secondary_rungs": fr["p5_rungs"]["secondary"],
                      "p": max([pis[r_]["p"] for r_ in conf]) if conf else 1.0,
                      "holds": bool(all(pis[r_]["no_plateau"] for r_ in conf) and not falsified and conf),
                      "falsified_on": falsified}
    pg = {r_: pi_with_fail(vals(ABt, r_), vals(wAt, r_), vals(wBt, r_), fails(ABt), fails(wAt), fails(wBt),
                           f"{C.KEY}|boot|P5g|{r_}", "plateau") for r_ in ("Gs", "dCKA")}
    F0["P5_geom"] = {"rungs": pg, "p": max(p["p"] for p in pg.values()), "holds": all(p["holds"] for p in pg.values())}
    # guards void dependent tests
    guard_of = {"P2": gte["holds"], "P3a": F0["P3a"]["guard"], "P4a": ggeo[ltop]["holds"]}
    pv = {k: (v["p"] if guard_of.get(k, True) else 1.0) for k, v in F0.items()}
    hol = S.holm(pv, tc["alpha"])
    for k in F0:
        F0[k]["guard_ok"] = guard_of.get(k, True)
        F0[k]["holm"] = hol[k]
        F0[k]["confirmed"] = bool(F0[k]["holds"] and hol[k]["rejected"] and guard_of.get(k, True))
    out["F0"] = F0
    out["guards"] = {"geo": {str(k): v for k, v in ggeo.items()}, "TE_Ntop": gte}
    # ---------------------------------------------------------------- calibration and plants
    cal = {}
    for l2, a in cells.items():
        rows = a.rows("link_self")
        x = sum(1 for r_ in rows if exv(r_, "ex8", "T_E") == 0.0)
        om_err = max(abs(val(r_, "Gs_max") - Mt.omega_self(st.K(r_["a"]), st.t)) for r_ in rows) if rows else float("nan")
        cal[f"link_self_l{l2}"] = {"x": x, "n": len(rows), "omega_err_max": om_err,
                                   "ok": bool(x == len(rows) and om_err <= 1e-9)}
    for name, a in (("N0", a0), ("Ntop", at)):
        lp, wphi, wA_ = a.rows("link_peer"), a.rows("w_phi"), a.rows("wA")
        p2l = decide(exvals(lp, "ex8", "T_E"), fails(lp), 0.0, "reliable")
        p3l = decide(vals(lp, "Gs_max"), fails(lp), eps["eps_geo"], "unreliable")
        p4l = decide(vals(lp, "iso_inf_AB"), fails(lp), eps["eps_geo"], "reliable")
        p5l = {r_: pi_with_fail(vals(lp, r_), vals(wA_, r_), vals(wphi, r_), fails(lp), fails(wA_), fails(wphi),
                                f"{C.KEY}|boot|cal|lp|{name}|{r_}", "no_plateau") for r_ in ORDER}
        p5g = {r_: pi_with_fail(vals(lp, r_), vals(wA_, r_), vals(wphi, r_), fails(lp), fails(wA_), fails(wphi),
                                f"{C.KEY}|boot|cal|lpg|{name}|{r_}", "plateau") for r_ in ("Gs", "dCKA")}
        # expectation (draft 6.3: "same P2-P5 verdicts as A-B at N_top"; R32): judged at N_top only, order
        # rungs as in P5-order (no plateau on the confirmatory rungs, no plateau-falsification on any rung);
        # N0 is reported only.
        conf_r = fr["p5_rungs"]["confirmatory"]
        ok = bool(p3l["holds"] and all(p5l[r_]["no_plateau"] for r_ in conf_r) and conf_r
                  and not any(v["plateau"] for v in p5l.values()) and all(v["plateau"] for v in p5g.values())
                  and p2l["holds"] and p4l["holds"])
        cal[f"link_peer_{name}"] = {"P2": p2l, "P3a": p3l, "P4a": p4l, "P5_order": p5l, "P5_geom": p5g,
                                    "ok": ok if name == "Ntop" else True, "ok_reported_only": ok,
                                    "judged": name == "Ntop"}
        rs = a.rows("rescale_self")
        worst = max(max(abs(val(r_, "Gs_max")), abs(val(r_, "Gs")), abs(val(r_, "dCKA")),
                        max(abs(val(r_, k)) for k in ORDER)) for r_ in rs)
        cal[f"rescale_self_{name}"] = {"max_abs": worst, "ok": bool(worst <= 1e-9)}
    plants = {}
    for mu, nm_ in ((1, "plant1"), (4, "plant4")):
        cr, wF, wA_ = at.rows(nm_), at.rows(f"w_F{mu}"), at.rows("wA")
        pp = {r_: pi_with_fail(vals(cr, r_), vals(wA_, r_), vals(wF, r_), fails(cr), fails(wA_), fails(wF),
                               f"{C.KEY}|boot|plant{mu}|{r_}", "plateau") for r_ in ORDER}
        wte = decide(exvals(wF, "ex8", "T_E"), fails(wF), 0.0, "unreliable")
        broken = bool(any(p["plateau"] for p in pp.values()) or wte["decision"] == "unreliable")
        # frozen expectation from the population limit of the plant, (K*, F_mu(K*)) = poplim case i_mu<mu>:
        # BROKEN if some order rung has Delta^inf >= dep_hi (0.01); intact if every rung <= dep_lo (1e-6);
        # otherwise no expectation (section 6.3; FD-d.1 thresholds)
        lad = fr["poplim"][f"i_mu{mu}"]["ladder"]
        dinf = {r_: float(C.fnum(lad[r_])) for r_ in ORDER}
        if any(v >= tc["dep_hi"] for v in dinf.values()):
            expect = "broken"
        elif all(v <= tc["dep_lo"] for v in dinf.values()):
            expect = "intact"
        else:
            expect = "none"
        match = (expect == "none") or (expect == "broken" and broken) or (expect == "intact" and not broken)
        plants[f"plant_break_mu{mu}"] = {"pi": pp, "withinF_TE": wte, "broken": broken, "dE_inf_i_mu": dinf["dEk"],
                                         "delta_inf_i_mu": dinf, "expected": expect, "match": bool(match)}
    cal_ok = all(v["ok"] for v in cal.values()) and all(v["match"] for v in plants.values())
    out["calibration"] = cal
    out["plants"] = plants
    out["calibration_ok"] = bool(cal_ok)
    # ---------------------------------------------------------------- verdict (section 9.5)
    lw = l_within(a0, at, tc)
    # L-cross (R31): no plateau on every confirmatory order rung and no plateau on any order rung
    l_cross = bool(conf and all(pis[r_]["no_plateau"] for r_ in conf) and not falsified)
    if lw["holds"] and gte["holds"] and l_cross:
        verdict = "PRESENT-order"
    elif lw["holds"] and gte["holds"] and falsified:
        verdict = "BROKEN-cross " + ",".join(falsified)
    elif not (lw["holds"] and gte["holds"]):
        verdict = "BROKEN-within" + (" + BROKEN-cross " + ",".join(falsified) if falsified else "")
    else:
        verdict = "INCONCLUSIVE"
    out["verdict"] = {"verdict": verdict, "L_cross": l_cross, "L_within": lw, "TE_guard": gte["holds"],
                      "geometry": {"Pi_Gs": pg["Gs"], "Pi_dCKA": pg["dCKA"]}}
    # ---------------------------------------------------------------- secondary
    sec = {}
    p1b_cells = [cells[l] for l in (l0, l0 + 3, ltop) if l in cells]
    sec["P1b"] = p1b(st, fr, p1b_cells)
    if (l0 + 3) in cells:
        r8 = cells[l0 + 3].rows("AB")
        sec["P2_at_8N0"] = decide(exvals(r8, "ex8", "T_E"), fails(r8), 0.0, "reliable")
    mc8, mc16 = fr["mc_curves"]["ex8"], fr["mc_curves"]["ex16"]
    curve, okb = {}, True
    for l2, a in cells.items():
        k = f"{float(l2):.2f}"
        row = {}
        for which, mc, obs_list in (("ex8", mc8, (("T_E", "TE"),)), ("ex16", mc16, (("T_E", "TE"), ("T_O", "TO"), ("T_N3", "TN")))):
            for obs, mk in obs_list:
                for s_, rows in (("AA", a.rows("wA")), ("BB", a.rows("wB")), ("AB", a.rows("AB"))):
                    ph = float(np.nanmean([1 - x for x in exvals(rows, which, obs)]))
                    pm = mc.get(k, {}).get(f"{mk}_{s_}", float("nan"))
                    inband = 0.05 <= pm <= 0.95 if pm == pm else False
                    within = abs(ph - pm) <= tc["p2b_band"] if inband else None
                    row[f"{which}_{obs}_{s_}"] = {"phat": ph, "mc": pm, "checked": inband, "within": within}
        curve[str(l2)] = row
    first = next((l2 for l2 in sorted(cells) if curve[str(l2)]["ex8_T_E_AA"]["phat"] >= 0.9), None)
    p2b_ok = all(v["within"] for r in curve.values() for kk, v in r.items() if kk.startswith("ex8_T_E_AA") and v["checked"])
    sec["P2b"] = {"curve": curve, "first_N_pAA_ge_0.9": first, "log2_Nstar_AA": fr["nstar"]["log2_Nstar_AA"],
                  "first_within_one_step": (first is not None and abs(first - fr["nstar"]["log2_Nstar_AA"]) <= 3),
                  "band_ok": p2b_ok}
    sec["P2c"] = {"band_ok": all(v["within"] for r in curve.values() for kk, v in r.items()
                                 if kk.startswith("ex16") and v["checked"])}
    sec["P4r"] = pi_with_fail(vals(ABt, "dIso"), vals(wAt, "dIso"), vals(wBt, "dIso"), fails(ABt), fails(wAt), fails(wBt),
                              f"{C.KEY}|boot|P4r", "no_plateau")
    et = epsp["per_N"][str(ltop)]
    p5b = {}
    for r_ in ORDER:
        p5b[f"cross_{r_}"] = decide(vals(ABt, r_), fails(ABt), et["AB"][r_]["max"], "reliable", et["AB"][r_]["q90"])
    for r_ in ("Gs", "dAff", "dCKA", "dSOG_t0p1"):
        p5b[f"cross_{r_}"] = decide(vals(ABt, r_), fails(ABt), et["AB"][r_]["max"], "unreliable", et["AB"][r_]["q90"])
    for L, rows in (("A", wAt), ("B", wBt)):
        for r_ in ORDER + ["Gs", "dAff", "dCKA", "dSOG_t0p1"]:
            p5b[f"within{L}_{r_}"] = decide(vals(rows, r_), fails(rows), et[L][r_]["max"], "reliable", et[L][r_]["q90"])
    mcr = fr["pi_mc"][str(ltop)]
    p5b["mc_rho_vs_obs"] = {r_: {"mc": mcr[r_]["rho"], "obs": pis[r_]["rho"],
                                 "within_1.5": bool(mcr[r_]["rho"] == mcr[r_]["rho"] and pis[r_]["rho"] == pis[r_]["rho"]
                                                    and 1 / 1.5 <= pis[r_]["rho"] / mcr[r_]["rho"] <= 1.5)} for r_ in ORDER}
    sec["P5b"] = p5b
    # P7 (S2.1-matched reading)
    lm = nm["log2N_match"]
    am = cells[lm]
    rAB, rwA, rwB = am.rows("AB"), am.rows("wA"), am.rows("wB")
    p7 = {"log2N_match": lm, "Pi": {}, "rule": None}
    for r_ in ORDER + ["dIso"]:
        p7["Pi"][r_] = pi_with_fail(vals(rAB, r_), vals(rwA, r_), vals(rwB, r_), fails(rAB), fails(rwA), fails(rwB),
                                    f"{C.KEY}|boot|P7|{r_}", "no_plateau")
    if any(p7["Pi"][r_]["plateau"] for r_ in ORDER):
        p7["rule"] = "plateau inside the premises at matched noise: S2.1's order failure may not be attributed to premise departures without a sample-size qualifier"
    elif all(p7["Pi"][r_]["ci_hi"] < 0 for r_ in ORDER):
        p7["rule"] = "inside the model at matched noise, cross-objective order disagreement does not exceed the within-objective level"
    else:
        p7["rule"] = "neither clause of the frozen interpretive rule applies (report Pi with CIs only)"
    lp0 = g["log2N_pilot"][0]
    if lp0 in cells:
        ap = cells[lp0]
        ep = epsp["per_N"][str(lp0)]
        tab = {}
        for k in TOL_KEYS:
            for s_, rows, e_ in (("A", ap.rows("wA"), ep["A"]), ("B", ap.rows("wB"), ep["B"]), ("AB", ap.rows("AB"), ep["AB"])):
                if k in e_ and float(C.fnum(e_[k]["max"])) == float(C.fnum(e_[k]["max"])):
                    tab[f"{s_}_{k}"] = decide(vals(rows, k), fails(rows), float(C.fnum(e_[k]["max"])), "reliable",
                                              float(C.fnum(e_[k]["q90"])))
        p7["s21_format_table_at_pilot_N"] = {"log2N": lp0, "table": tab}
    sec["P7"] = p7
    # P8 (R- small N)
    p8 = {}
    for l2 in sorted({(g["log2N_br"] if p == "br" else l0 + int(p)) for p in cfg["runs"]["rm_points"]}):
        if l2 not in cells:
            continue
        a = cells[l2]
        R = 0
        while R < cfg["runs"]["rm"] and st.done(rid("Rm", "Ap", l2, R)):
            R += 1
        nb = min(R, a.R) // 2
        S8, S16 = fr["probes"]["S_ex8"], fr["probes"]["S_ex16"]
        X = pair_rows(st, f"rm_l{l2}_cross_n{nb}", [((a.A(2 * j), None), (rid("Rm", "Ap", l2, 2 * j), None)) for j in range(nb)], S8, S16)
        WA = pair_rows(st, f"arm0_l{l2}_wA_n{a.nb}", [((a.A(2 * j), None), (a.A(2 * j + 1), None)) for j in range(a.nb)], S8, S16)[:nb]
        WP = pair_rows(st, f"rm_l{l2}_wAp_n{nb}", [((rid("Rm", "Ap", l2, 2 * j), None), (rid("Rm", "Ap", l2, 2 * j + 1), None)) for j in range(nb)], S8, S16)
        ent = {"n_blocks": nb, "Pi": {}}
        for r_ in ORDER + ["Gs", "dCKA"]:
            ent["Pi"][r_] = pi_with_fail(vals(X, r_), vals(WA, r_), vals(WP, r_), fails(X), fails(WA), fails(WP),
                                         f"{C.KEY}|boot|P8|{l2}|{r_}", "no_plateau")
        # raw distances are NOT invariant under the rescaling (correction (c)): descriptive means only
        ent["raw_G_max_means"] = {"cross": means(X, "G_max"), "within_A": means(WA, "G_max"), "within_Ap": means(WP, "G_max"),
                                  "note": "not rescaling-invariant; no identity, no decision"}
        ent["frac_psd_active_A"] = float(np.mean([st.info(a.A(i)).get("psd_active", False) for i in range(2 * nb)]))
        ent["frac_psd_active_Ap"] = float(np.mean([st.info(rid("Rm", "Ap", l2, i)).get("psd_active", False) for i in range(2 * nb)]))
        p8[str(l2)] = ent
    sec["P8"] = p8
    out["secondary"] = sec
    # ---------------------------------------------------------------- descriptive per-N table
    desc = {}
    for l2, a in cells.items():
        r_ab, r_a, r_b = a.rows("AB"), a.rows("wA"), a.rows("wB")
        desc[str(l2)] = {s_: {k: means(rows, k) for k in ORDER + list(Mt.SECONDARY_ORDER)
                              + ["Gs", "Gs_max", "dCKA", "dIso", "iso_inf_AB", "s_N_k10", "G_max"]}
                         for s_, rows in (("AB", r_ab), ("AA", r_a), ("BB", r_b))}
        desc[str(l2)]["n_blocks"] = a.nb
        desc[str(l2)]["frac_psd_active_A"] = float(np.mean([st.info(a.A(i)).get("psd_active", False) for i in range(a.R)]))
        desc[str(l2)]["frac_proj_active_B"] = float(np.mean([st.info(a.B(i)).get("proj_active", False) for i in range(a.R)]))
        desc[str(l2)]["n_failed_A"] = int(sum(st.failed(a.A(i)) for i in range(a.R)))
        desc[str(l2)]["n_failed_B"] = int(sum(st.failed(a.B(i)) for i in range(a.R)))
        desc[str(l2)]["median_eA"] = float(np.nanmedian(a.err("A")[0]))
        desc[str(l2)]["median_eB"] = float(np.nanmedian(a.err("B")[0]))
    out["descriptive"] = desc
    # ---------------------------------------------------------------- stop rule (author decision 2026-10-09)
    reasons = []
    for k, v in F0.items():
        if not v["confirmed"]:
            reasons.append(f"F0 {k} not confirmed")
    if not gte["holds"]:
        reasons.append("S_ex8 within T_E guard failed")
    if verdict != "PRESENT-order":
        reasons.append(f"order verdict {verdict}")
    if not cal_ok:
        reasons.append("calibration / planted-control mismatch (pipeline audit)")
    out["stop_rule"] = {"arm0_pattern_present": not reasons, "reasons": reasons,
                        "action": "report to the author; Stage B needs a separate approval" if not reasons
                        else "STOP: arm 0 failed inside the premises; report and audit; do not proceed"}
    out["gpu_hours_ledger"] = C.gpu_hours_used()
    return out


# ============================================================================ report
def fmt_p(p):
    return f"{p:.2e}" if isinstance(p, float) else str(p)


def write_report(res, fr):
    L = ["# S2.3 Stage A report (generated by analyze_A.py; frozen rules PREREG_S2_3_v1.0)", ""]
    L.append(f"Stop rule: **{res['stop_rule']['action']}**; reasons: {res['stop_rule']['reasons'] or 'none'}")
    L.append(f"Order verdict (arm 0, N_top): **{res['verdict']['verdict']}**")
    L.append("")
    L.append("| F0 test | statistic | p | Holm thr | guard | holds | confirmed |")
    L.append("|---|---|---|---|---|---|---|")
    for k, v in res["F0"].items():
        stat = ""
        if "ratio" in v:
            stat = f"ratio {v['ratio']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]"
        elif "x" in v:
            stat = f"{v['x']}/{v['n']} ({v['decision']})"
        elif k == "P3a":
            stat = f"N0 {v['N0']['x']}/{v['N0']['n']}, Ntop {v['Ntop']['x']}/{v['Ntop']['n']}"
        elif k == "P3b":
            stat = f"cross 1-CKA {v['tost']['mean']:.4f}; within {v['within_A']['mean']:.2e}/{v['within_B']['mean']:.2e}"
        elif k.startswith("P5"):
            stat = "; ".join(f"{r}: Pi {x['Pi']:.2e} [{x['ci_lo']:.2e},{x['ci_hi']:.2e}] rho {x['rho']:.2f}" for r, x in v["rungs"].items())
        L.append(f"| {k} | {stat} | {fmt_p(v['holm']['p'])} | {v['holm']['holm_threshold']:.4f} | {v['guard_ok']} | {v['holds']} | {v['confirmed']} |")
    L.append("")
    L.append(f"Calibration and plants ok: {res['calibration_ok']}")
    for k, v in res["plants"].items():
        L.append(f"- {k}: broken={v['broken']} (expected {v['expected']}; match {v['match']})")
    L.append("")
    L.append(f"P7: log2 N_match = {res['secondary']['P7']['log2N_match']}; rule: {res['secondary']['P7']['rule']}")
    for r, x in res["secondary"]["P7"]["Pi"].items():
        L.append(f"- {r}: AB {x['mean_AB']:.4g}, AA {x['mean_AA']:.4g}, BB {x['mean_BB']:.4g}, Pi {x['Pi']:.3g} "
                 f"[{x['ci_lo']:.3g}, {x['ci_hi']:.3g}], rho {x['rho']:.3g}")
    L.append("")
    for l2, e in res["secondary"]["P8"].items():
        L.append(f"P8 at log2N={l2}: " + "; ".join(f"{r}: Pi {x['Pi']:.3g} [{x['ci_lo']:.3g},{x['ci_hi']:.3g}]" for r, x in e["Pi"].items())
                 + f"; PSD active A {e['frac_psd_active_A']:.2f}, A' {e['frac_psd_active_Ap']:.2f}")
    L.append("")
    L.append("| log2N | blocks | AB dEk | AA dEk | BB dEk | AB sN10 | AA sN10 | AB Gs_max | med e_A | med e_B | PSD-active A |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for l2, d in res["descriptive"].items():
        L.append(f"| {l2} | {d['n_blocks']} | {d['AB']['dEk']:.3g} | {d['AA']['dEk']:.3g} | {d['BB']['dEk']:.3g} | "
                 f"{d['AB']['s_N_k10']:.4f} | {d['AA']['s_N_k10']:.4f} | {d['AB']['Gs_max']:.4f} | {d['median_eA']:.3g} | "
                 f"{d['median_eB']:.3g} | {d['frac_psd_active_A']:.2f} |")
    L.append("")
    L.append(f"GPU-hours (ledger): {res['gpu_hours_ledger']:.3f}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["gates", "nmatch", "main"])
    a = ap.parse_args()
    C.verify_frozen_code(strict=True)
    plan = C.load_json(os.path.join(C.PLANS, "stageA_plan.json"))
    fr = C.load_json(C.FROZEN_PRED)
    arr = dict(np.load(C.PRESTEP_ARR))
    st = Store(plan, fr, arr)
    os.makedirs(C.STAGEA_DIR, exist_ok=True)
    if a.phase == "gates":
        out = phase_gates(st, plan, fr)
        sys.exit(0 if out["G4"]["main_allowed"] else 3)
    if a.phase == "nmatch":
        phase_nmatch(st, fr)
        return
    res = phase_main(st, fr)
    hj = C.dump_json(res, os.path.join(C.STAGEA_DIR, "stageA_results.json"))
    rep = write_report(res, fr)
    rp = os.path.join(C.STAGEA_DIR, "REPORT_stageA.md")
    with open(rp, "w") as f:
        f.write(rep)
    hr = C.sha256_file(rp)
    C.append_status("run", f"S2.3 Stage A analysis done: verdict {res['verdict']['verdict']}; stop rule: "
                    f"{res['stop_rule']['action']}; stageA_results.json sha256 {hj}; REPORT_stageA.md sha256 {hr}; "
                    f"GPU ledger {res['gpu_hours_ledger']:.3f} h")


if __name__ == "__main__":
    main()
