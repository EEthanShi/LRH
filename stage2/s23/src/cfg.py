"""Frozen numerical settings of S2.3 Stage A (PREREG_S2_3_v1.0.md; one place, hashed with the code).

FROZEN is the configuration of every Stage A computation. tiny() returns the configuration of the
excluded unit tests (|X| = 12, small N, small Monte Carlo); it is never used by a plan.
"""
from __future__ import annotations

import copy

FROZEN = {
    "t": 0.3,
    "geometry": {"n_super": 4, "n_clu": 4, "n_per": 16, "dim": 8, "sigma": 0.5, "nugget": 0.2,
                 "lv": [1.0, 0.55, 0.30], "n_alt": 10},
    "g0": {"lam_min": 0.05, "max_off": 0.95, "imbalance": 1e-12, "min_gap": 1e-9, "ex8_dgap": 1e-3,
           "omega": 0.2, "twoview_ptp": 0.05},
    "probes": {"n_cand": 20000, "n_keep": 20, "max_rank_gap": 3},
    "mc": {"log2_lo": 20.0, "log2_hi": 46.0, "step": 0.25, "n_draws": 1_000_000, "chunk": 200_000,
           "ladder_blocks": 48, "level": 0.9},
    "grid": {"j_lo": -4, "j_hi": 2, "step_log2": 3, "br_offset": 14, "br_min_log2": 20,
             "dep_offsets": [-12, 0, 6], "pilot_offsets": [-12, 6], "top_offset": 6},
    "runs": {"main": 192, "pilot": 128, "rm": 192, "g1b": 16, "g3": 16, "d2": 16,
             "g1b_offsets": [-12, 0, 6], "d2_log2N": [22, 40], "rm_points": ["br", -12]},
    "solver": {"newton_step_tol": 1e-11, "pcg_rtol": 1e-10, "bm_rel": 1e-6, "bm_max_evals": 20000,
               "bm_stall_chunks": 30, "b_tol_abs": 1e-9},
    "poplim": {"mus": [1, 2, 4, 8, 16, 32], "ranks": [192, 128, 64], "tau_div": 2.5, "n_inits": 8,
               "rank_max_evals": 4000, "polish_iters": 20, "nonconvex_cap_h": 1.5, "accept_relgrad": 1e-9,
               "accept_hess_rel": 1e-8, "accept_spread": 1e-4, "v_rank": 128},
    "tests": {"g1_A_tol": 1e-9, "g1_B_tol": 1e-12, "g1_ncm_tol": 1e-8,
              "g1_exact_tol": 1e-12, "lemmaL_plants": 1000, "lemmaL_m": 24, "iso_plants": 200,
              "p5_rho_power": 1.6, "conv": 0.35, "plat": 0.7, "cka_band": 0.01, "within_cka": 0.01,
              "dep_band": [0.5, 2.0], "dep_hi": 0.01, "dep_lo": 1e-6, "nmatch_target": 0.715,
              "alpha": 0.05, "p1b_n_margin": 8, "p1b_n_random": 8, "p2b_band": 0.15,
              "g3_frac": 0.01, "sd_entries": 512},
    "workers": {"gpu": 2, "cpu": 1},
}


def frozen():
    return copy.deepcopy(FROZEN)


def tiny():
    """Excluded unit-test configuration: |X| = 2*2*3 = 12, cheap Monte Carlo, small grid."""
    c = frozen()
    c["geometry"].update({"n_super": 2, "n_clu": 2, "n_per": 3, "dim": 4, "n_alt": 3})
    c["probes"].update({"n_cand": 200, "n_keep": 5})
    c["mc"].update({"log2_lo": 6.0, "log2_hi": 30.0, "step": 1.0, "n_draws": 4000, "chunk": 4000,
                    "ladder_blocks": 4})
    c["grid"].update({"br_min_log2": 8})
    c["runs"].update({"main": 8, "pilot": 8, "rm": 8, "g1b": 2, "g3": 2, "d2": 2, "d2_log2N": [10, 14]})
    c["poplim"].update({"mus": [1, 4], "ranks": [8, 4], "n_inits": 3, "rank_max_evals": 3000,
                        "polish_iters": 20, "v_rank": 6})
    c["tests"].update({"lemmaL_plants": 50, "iso_plants": 20, "sd_entries": 16})
    return c
