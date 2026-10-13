"""Frozen pre-step of S2.3 Stage A (PREREG sections 3.2, 4.2-4.4, 5.3): no sample of any run is drawn.

Order: G0 (instance; fallback geometry keys) -> I_1, its Cholesky factor and a lower bound of
lambda_min(I_1) (Lanczos on I_1^{-1}; used by every Lemma T identity check) -> probes S_ex8, S_ex16
-> G0 probe item -> MC curves, N*, N0, grid -> P1b contrasts and per-entry sd -> ladder-level MC and
the P5 confirmatory-rung rule -> population limits of every arm -> G1 -> G1b population part ->
results/prestep/frozen_predictions.json (+ prestep_arrays.npz), hashed into STATUS before any sample.

Usage: python prestep.py            (frozen configuration; ds2213 only)
"""
from __future__ import annotations

import os
import sys
import time

import s23common as C

if __name__ == "__main__":
    C.cap_threads(1)

import numpy as np
import torch

import cfg as CF
import fisher as F
import gates as G
import instance as I
import learnA as A
import metrics as Mt
import nstar as NS
import poplim as PL
import probes as PR


def _log(msg, fh=None):
    line = f"{C.utcnow()} {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def build_instance_and_probes(cfg, dev, log):
    tried = []
    for key in I.geometry_keys(cfg["geometry"]["n_alt"]):
        inst = I.build(key, cfg["t"], cfg["geometry"])
        g0 = I.g0_checks(inst, cfg["g0"])
        if not g0["pass_all_noprobe"]:
            tried.append({"key": key, "pass": False, "stage": "noprobe"})
            log(f"[G0] {key}: FAIL (no-probe items)")
            continue
        pi = torch.as_tensor(inst.pi_gibbs(), dtype=torch.float64, device=dev)
        Ks = torch.as_tensor(inst.K, dtype=torch.float64, device=dev)
        I1 = F.assemble_I1(pi, Ks, inst.t)
        L = F.cholesky_inplace_safe(I1)
        del I1, pi
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        pc = cfg["probes"]
        p8 = PR.select(inst, L, "ex8", pc["n_cand"], pc["n_keep"], pc["max_rank_gap"])
        p16 = PR.select(inst, L, "ex16", pc["n_cand"], pc["n_keep"], pc["max_rank_gap"])
        g0f = G.g0_full(inst, g0, p8["S"], cfg["g0"])
        tried.append({"key": key, "pass": g0f["pass_all"], "stage": "full"})
        log(f"[G0] {key}: {'PASS' if g0f['pass_all'] else 'FAIL'} (lam_min={g0['lambda_min_K']:.4f}, "
            f"omega={g0['omega_psi_X']:.4f}, ex8 dgap={g0f['ex8_min_dgap']:.3g})")
        if g0f["pass_all"]:
            return inst, L, p8, p16, g0f, tried
    raise RuntimeError(f"G0 failed for every geometry key: {tried}")


def p1b_contrasts(inst, L, p8, cfg):
    """8 smallest-margin T_E contrasts on S_ex8 (by min(rho_A, rho_B)) and 8 keyed random entries; per-
    example variances for A (I_1^{-1}) and B (delta method)."""
    tc = cfg["tests"]
    hyp = sorted(p8["hyperplanes"], key=lambda h: min(h[2], h[3]))[: tc["p1b_n_margin"]]
    pairs8 = p8["pairs"]
    out = []
    for (e, f, rA, rB, sA, sB) in hyp:
        out.append({"kind": "TE_contrast", "plus": pairs8[e], "minus": pairs8[f], "var_A": sA ** 2,
                    "var_B": sB ** 2, "rho_A": rA, "rho_B": rB})
    r = C.rng(f"{C.KEY}|p1b|random_entries")
    n = inst.n
    iu = np.triu_indices(n, 1)
    sel = r.choice(len(iu[0]), tc["p1b_n_random"], replace=False)
    pairs = [(int(iu[0][s]), int(iu[1][s])) for s in sel]
    ids, _ = F.entry_ids_pairs(pairs, n)
    SigA = F.cov_A_entries(L, ids)
    _, SigB = F.cov_B_entries(inst, pairs)
    for j, (x, y) in enumerate(pairs):
        out.append({"kind": "entry", "plus": [x, y], "minus": None, "var_A": float(SigA[j, j]),
                    "var_B": float(SigB[j, j])})
    return out


def sd_entry_per_example(inst, L, cfg):
    r = C.rng(f"{C.KEY}|sd_entries")
    n = inst.n
    iu = np.triu_indices(n, 1)
    k = min(cfg["tests"]["sd_entries"], len(iu[0]))
    sel = r.choice(len(iu[0]), k, replace=False)
    pairs = [(int(iu[0][s]), int(iu[1][s])) for s in sel]
    ids, _ = F.entry_ids_pairs(pairs, n)
    SigA = F.cov_A_entries(L, ids)
    return float(np.median(np.sqrt(np.diag(SigA))))


def run(cfg=None, dev=None, write=True, status=True, worker="prestep"):
    cfg = CF.frozen() if cfg is None else cfg
    dev = C.torch_device() if dev is None else dev
    C.verify_frozen_code(strict=True)
    os.makedirs(C.PRESTEP_DIR, exist_ok=True)
    fh = open(os.path.join(C.PRESTEP_DIR, "prestep.log"), "a")
    log = lambda m: _log(m, fh)  # noqa: E731
    T0 = time.time()
    sec = C.GpuSection(worker, "prestep")
    sec.__enter__()
    try:
        inst, L, p8, p16, g0f, tried = build_instance_and_probes(cfg, dev, log)
        S8, S16 = p8["S"], p16["S"]
        probes = {"ex8": (S8, 2), "ex16": (S16, 3)}
        ladder_star = Mt.ladder_deltas(inst.K, inst.KBt)
        omega = I.omega_psi(inst.K, inst.t)
        eps = {"omega_psi": omega, "eps_geo": omega / 2.0, "cka_inf": ladder_star["cka_biased"],
               "one_minus_cka_inf": 1.0 - ladder_star["cka_biased"],
               "eps_cka": (1.0 - ladder_star["cka_biased"]) / 2.0, "Gs_inf": ladder_star["Gs"],
               "Gs_max_inf": ladder_star["Gs_max"]}
        log(f"[eps] eps_geo={eps['eps_geo']:.5f} eps_cka={eps['eps_cka']:.5f} 1-CKA_inf={eps['one_minus_cka_inf']:.5f}")
        # ---- MC curves, N*, grid
        mc = cfg["mc"]
        grid_mc = NS.mc_grid(mc)
        curves = {}
        for name, p, k in (("ex8", p8, 2), ("ex16", p16, 3)):
            kA = np.array([inst.K[a, b] for a, b in p["pairs"]])
            kB = np.array([inst.KBt[a, b] for a, b in p["pairs"]])
            curves[name] = NS.probe_mc(kA, kB, p["SigA"], p["SigB"], len(p["S"]), k, grid_mc, mc["n_draws"],
                                       f"{C.KEY}|mc|{name}", dev, chunk=mc["chunk"])
            log(f"[mc] {name} done ({len(grid_mc)} grid points)")
        anc = NS.anchor(curves["ex8"], mc["level"])
        grid = NS.grid_from_N0(anc["log2N0"], cfg["grid"])
        nstar = dict(anc)
        nstar["Nstar_peer_A"] = NS.peer_nstar(p8["rho_A"])
        nstar["Nstar_peer_B"] = NS.peer_nstar(p8["rho_B"])
        nstar["log2_Nstar_ex16_AB"] = NS.first_reaching(curves["ex16"], "TE_AB", mc["level"])
        log(f"[nstar] log2 N*={anc['log2_Nstar']} log2 N0={anc['log2N0']} grid={grid['grid']}")
        # ---- P1b contrasts, per-entry sd
        contr = p1b_contrasts(inst, L, p8, cfg)
        sd1 = sd_entry_per_example(inst, L, cfg)
        sd_top = sd1 / np.sqrt(2.0 ** grid["log2N_top"])
        # ---- ladder MC and P5 rung rule
        pimc = {}
        for l2 in grid["grid"]:
            pimc[str(l2)] = NS.pi_mc(inst, L, int(l2), mc["ladder_blocks"], f"{C.KEY}|pimc")
            log(f"[pimc] log2N={l2}: " + " ".join(f"{r}:rho={pimc[str(l2)][r]['rho']:.3g}" for r in Mt.ORDER_RUNGS))
        top = pimc[str(grid["log2N_top"])]
        # P5 rung power rule (draft: MC rho > 1.6 -> secondary; R31: also secondary when the MC at N_top is
        # degenerate, i.e. rho undefined or fewer than half of the within pairs of either learner nonzero,
        # since then the block bootstrap cannot put the upper CI of Pi below 0). Falsifiers stay on every rung.
        conf, secd = [], []
        for r_ in Mt.ORDER_RUNGS:
            rho = top[r_]["rho"]
            powered = bool(rho == rho and rho <= cfg["tests"]["p5_rho_power"]
                           and top[r_]["frac_nonzero_AA"] >= 0.5 and top[r_]["frac_nonzero_BB"] >= 0.5)
            (conf if powered else secd).append(r_)
        # ---- lambda_min(I_1) (frozen; Lemma T curvature-derived tolerances)
        lamI = A.lam_min_I1(L, f"{C.KEY}|lanczos|I1")
        log(f"[I1] lambda_min(I_1) >= {lamI['lam_min_I1_lb']:.4e} (Lanczos residual {lamI['res']:.2e}); "
            f"lambda_max(I_1) ~ {lamI['lam_max_I1_est']:.4e}; reference 0.04/(n^2 t^2) = {0.04 / (inst.n ** 2 * inst.t ** 2):.3e}")
        # ---- population limits
        precond = A.FisherPrecond(L, inst.n)
        res, arr = PL.all_limits(inst, precond, cfg, probes, dev, lamI["lam_min_I1_lb"], log=log)
        # ---- G1, G1b population
        g1 = G.g1(inst, arr["0_A"], cfg, sd_top, ladder_star)
        hb0 = res["0"]["hess_bounds_A"]
        g1["g1_fisher_at_Kstar"] = {"lam_min_M": hb0["lam_min_M"], "lam_max_M": hb0["lam_max_M"],
                                    "pass": bool(abs(hb0["lam_min_M"] - 1.0) <= 1e-6 and abs(hb0["lam_max_M"] - 1.0) <= 1e-6)}
        g1["pass_all"] = bool(g1["pass_all"] and g1["g1_fisher_at_Kstar"]["pass"])
        g1["pass_all_with_fallback"] = bool(g1["pass_all_with_fallback"] and g1["g1_fisher_at_Kstar"]["pass"])
        g1b = G.g1b_population(res)
        log(f"[G1] pass_all={g1['pass_all']} with_fallback={g1['pass_all_with_fallback']}; "
            f"G1b pop Rm={g1b['pass_Rm']} (resid {g1b['Rm']['resid']:.2e} tol {g1b['Rm']['tol']:.2e}) "
            f"iv applicable={g1b['iv_lemmaT_applicable']} violation={g1b['iv_tau']['violation']} "
            f"(resid {g1b['iv_tau']['resid']:.2e})")
        frozen = {
            "version": "S2.3 prestep v1.0", "utc": C.utcnow(), "cfg": cfg, "key": C.KEY,
            "instance": {"key": inst.key, "t": inst.t, "Z": inst.Z, "imbalance": inst.imbalance, "meta": inst.meta},
            "g0": g0f, "g0_tried": tried, "probes": {
                "S_ex8": S8, "S_ex16": S16, "ex8": {k: p8[k] for k in ("min_dgap", "rho_A", "rho_B", "top")},
                "ex16": {k: p16[k] for k in ("min_dgap", "rho_A", "rho_B", "top")},
                "ex8_pairs": p8["pairs"], "ex16_pairs": p16["pairs"]},
            "ladder_star": ladder_star, "eps": eps, "nstar": nstar, "grid": grid,
            "mc_curves": curves, "pi_mc": pimc,
            "p5_rungs": {"confirmatory": conf, "secondary": secd},
            "p1b": {"contrasts": contr, "sd_entry_per_example": sd1, "sd_entry_Ntop": sd_top},
            "poplim": res, "g1": g1, "g1b_population": g1b, "lam_min_I1": lamI,
            "gates_pass": bool(g0f["pass_all"] and g1["pass_all_with_fallback"] and g1b["pass"]),
        }
        arrays = {k: np.asarray(v) for k, v in arr.items()}
        arrays.update({"Kstar": inst.K, "KBt": inst.KBt, "W": inst.W, "W2": inst.W2, "z": inst.z,
                       "S_ex8": np.array(S8), "S_ex16": np.array(S16), "SigA_ex8": p8["SigA"],
                       "SigB_ex8": p8["SigB"], "SigA_ex16": p16["SigA"], "SigB_ex16": p16["SigB"]})
    finally:
        sec.__exit__(None, None, None)
    frozen["gpu_hours_prestep"] = sec.seconds / 3600.0
    frozen["wall_s"] = time.time() - T0
    if write:
        np.savez(C.PRESTEP_ARR, **arrays)
        ha = C.sha256_file(C.PRESTEP_ARR)
        frozen["arrays_sha256"] = ha
        h = C.dump_json(frozen, C.FROZEN_PRED)
        os.chmod(C.FROZEN_PRED, 0o444)
        os.chmod(C.PRESTEP_ARR, 0o444)
        log(f"[prestep] frozen_predictions.json sha256 {h}; arrays sha256 {ha}")
        if status:
            C.append_status(
                "run", f"S2.3 Stage A PRE-STEP done (no sample drawn): results/prestep/frozen_predictions.json "
                f"sha256 {h}; prestep_arrays.npz sha256 {ha}; geometry key {inst.key}; G0 pass={g0f['pass_all']}; "
                f"G1 pass={g1['pass_all']} (with fallback {g1['pass_all_with_fallback']}); G1b-pop Rm={g1b['pass_Rm']} "
                f"iv applicable={g1b['iv_lemmaT_applicable']} no-violation={g1b['pass_iv']}; "
                f"lambda_min(I_1) >= {lamI['lam_min_I1_lb']:.3e}; log2 N*={nstar['log2_Nstar']} log2 N0={grid['log2N0']} grid={grid['grid']}; "
                f"P5 confirmatory rungs {conf}, secondary {secd}; eps_geo={eps['eps_geo']:.5f} eps_cka={eps['eps_cka']:.5f}; "
                f"GPU-h this step {frozen['gpu_hours_prestep']:.3f}, ledger total {C.gpu_hours_used():.3f}")
    fh.close()
    return frozen, arrays


if __name__ == "__main__":
    fr, _ = run()
    sys.exit(0 if fr["gates_pass"] else 3)
