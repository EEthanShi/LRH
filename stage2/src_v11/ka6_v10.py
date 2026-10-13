"""KA6 (PREREG_S2_2_v1.1 section 14, step 3; author decision 2026-10-09: computed BEFORE freeze).

READ-ONLY on the v1.0 S2.2 main cohort (runs/S2_2_main, 1,344 runs; seen data). For every run:
  * the initial head (P0, W0) is recovered EXACTLY from the run's seed key by replaying v1.0 s2lib.reinit
    (v1.0 workers cached one model per process, so the first run of a worker was initialised on CPU and
    later runs on the GPU; both replays are tried and the one matching the run's own stored init
    statistics is kept: meta imb_init.imb_abs and u_init ~ h_init P0^T, y_init ~ u_init W0^T);
  * C0 = P0 P0^T - W0^T W0 and C_end = P P^T - W^T W (paper sign; P, W = stored final float32 head);
  * mu_C = <C0, C_end>_F / (||C0||_F ||C_end||_F), growth = ||C_end - C0||_F / ||C0||_F, rho_end;
  * frac_own_closer exactly as v1.0 a4_mechanism_eps.py (W0 by least squares from u_init, y_init;
    "other" run drawn with default_rng(0) per cell), and the same statistic with the seed-recovered W0.
Per cell (arm x N): median mu_C with a percentile bootstrap (10,000 resamples) giving the two-sided 95% CI,
whose upper end is the one-sided 97.5% upper bound used by C4a; frac_own_closer with a two-sided 95%
Clopper-Pearson interval. Writes results/S2_2_v11/ka6/ka6_v10.json (new path). Nothing in v1.0 is written.

    OMP_NUM_THREADS=1 python src_v11/ka6_v10.py [--device-replay cuda|none]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gaugemath_v11 as GM  # noqa: E402

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
RUNS = os.path.join(ROOT, "runs", "S2_2_main")
OUT = os.path.join(ROOT, "results", "S2_2_v11", "ka6")
NBOOT = 10000


def seed_key_rng(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def replay_init(key, small_init, device):
    """Replay v1.0 s2lib.reinit for the head on CPU or on the GPU (model already on device, channels_last)."""
    import torch
    sys.path.insert(0, os.path.join(ROOT, "src"))  # v1.0 s2lib, imported read-only
    import s2lib as L1
    m = L1.GaugeNet()
    m.small_init = small_init
    if device != "cpu":
        m.to(device).to(memory_format=torch.channels_last)
    L1.reinit(m, key + "|init")
    P0 = m.P.weight.detach().double().cpu().numpy()
    W0 = m.W.weight.detach().double().cpu().numpy()
    return P0, W0


def relerr(A, B):
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


def boot_median_ci(x, key, n=NBOOT):
    x = np.asarray(x, float)
    rng = seed_key_rng(key)
    idx = rng.integers(0, x.size, (n, x.size))
    med = np.median(x[idx], axis=1)
    return [float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))]


def cp95(x, n):
    from scipy.stats import beta
    lo = beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
    return [float(lo), float(hi)]


def frac_own(Mf, Mi):
    """v1.0 a4 definition (per cell; rng reset to default_rng(0) for each cell)."""
    n = len(Mf)
    own = np.array([GM.d_ai(Mf[i], Mi[i]) for i in range(n)])
    rng = np.random.default_rng(0)
    other = np.array([GM.d_ai(Mf[i], Mi[(i + 1 + rng.integers(n - 1)) % n]) for i in range(n)])
    return int(np.sum(own < other)), n, float(own.mean()), float(other.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-replay", default="cuda")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    plan = json.load(open(os.path.join(ROOT, "plans", "S2_2_main.json")))["runs"]
    cells = {}
    for r in plan:
        cells.setdefault(f"{r['arm']}_{r['N']}", []).append(r)
    t0 = time.time()
    per_run, summary = {}, {}
    for cell, rs in cells.items():
        Mf, Mi_ls, Mi_seed, mus, grow, rho_end, c0op = [], [], [], [], [], [], []
        for r in rs:
            d = os.path.join(RUNS, r["run_id"])
            meta = json.load(open(os.path.join(d, "meta.json")))
            z = np.load(os.path.join(d, "feats.npz"))
            P, W = z["P"].astype(np.float64), z["W"].astype(np.float64)
            h0, u0, y0 = (z[k].astype(np.float64) for k in ("h_init", "u_init", "y_init"))
            si = meta["cfg"]["small_init"]
            cands = {"cpu": replay_init(meta["key"], si, "cpu")}
            if a.device_replay != "none":
                cands["cuda"] = replay_init(meta["key"], si, a.device_replay)
            scores = {}
            for dev, (P0, W0) in cands.items():
                e_imb = abs(float(np.linalg.norm(GM.C_of(P0, W0))) - meta["imb_init"]["imb_abs"]) / meta["imb_init"]["imb_abs"]
                scores[dev] = {"imb_abs_relerr": e_imb, "u_init_relerr": relerr(h0 @ P0.T, u0),
                               "y_init_relerr": relerr(u0 @ W0.T, y0)}
            best = min(scores, key=lambda k: scores[k]["imb_abs_relerr"] + scores[k]["u_init_relerr"] + scores[k]["y_init_relerr"])
            sb = scores[best]
            ok = sb["imb_abs_relerr"] < 1e-5 and sb["u_init_relerr"] < 1e-2 and sb["y_init_relerr"] < 1e-2
            P0, W0 = cands[best]
            C0, C1 = GM.C_of(P0, W0), GM.C_of(P, W)
            e_end = abs(float(np.linalg.norm(C1)) - meta["imb_final"]["imb_abs"]) / meta["imb_final"]["imb_abs"]
            mu = GM.mu_C(C0, C1)
            g = float(np.linalg.norm(C1 - C0) / np.linalg.norm(C0))
            Wt, *_ = np.linalg.lstsq(u0, y0, rcond=None)
            W0_ls = Wt.T
            Mf.append(GM.norm_metric(W))
            Mi_ls.append(GM.norm_metric(W0_ls))
            Mi_seed.append(GM.norm_metric(W0))
            mus.append(mu)
            grow.append(g)
            rho_end.append(GM.rho_of(P, W))
            c0op.append(float(np.linalg.norm(C0, 2)))
            per_run[r["run_id"]] = {"replay": best, "replay_ok": ok, **sb, "imb_final_relerr": e_end,
                                    "mu_C": mu, "growth_rel": g, "rho_end": rho_end[-1], "C0_op": c0op[-1],
                                    "C0_fro": float(np.linalg.norm(C0)), "C_end_fro": float(np.linalg.norm(C1)),
                                    "W0_ls_vs_seed_relerr": relerr(W0_ls, W0)}
        x_ls, n, own_ls, oth_ls = frac_own(Mf, Mi_ls)
        x_sd, _, own_sd, oth_sd = frac_own(Mf, Mi_seed)
        mus = np.array(mus)
        ci = boot_median_ci(mus, f"S2.2v11|KA6|mu_C|{cell}")
        gci = boot_median_ci(np.array(grow), f"S2.2v11|KA6|growth|{cell}")
        rr = [per_run[r["run_id"]] for r in rs]
        summary[cell] = {
            "n_runs": len(rs), "replay_ok_all": all(v["replay_ok"] for v in rr),
            "replay_device_counts": {d: sum(v["replay"] == d for v in rr) for d in ("cpu", "cuda")},
            "max_imb_init_relerr": max(v["imb_abs_relerr"] for v in rr),
            "max_imb_final_relerr": max(v["imb_final_relerr"] for v in rr),
            "max_u_init_relerr": max(v["u_init_relerr"] for v in rr), "max_y_init_relerr": max(v["y_init_relerr"] for v in rr),
            "median_W0_ls_vs_seed_relerr": float(np.median([v["W0_ls_vs_seed_relerr"] for v in rr])),
            "mu_C_median": float(np.median(mus)), "mu_C_median_ci95": ci, "mu_C_upper_975_one_sided": ci[1],
            "mu_C_q10_q90": [float(np.quantile(mus, 0.1)), float(np.quantile(mus, 0.9))],
            "mu_C_mean": float(mus.mean()), "frac_mu_C_gt_0p5": float(np.mean(mus > 0.5)),
            "growth_rel_median": float(np.median(grow)), "growth_rel_median_ci95": gci,
            "rho_end_median": float(np.median(rho_end)), "C0_op_median": float(np.median(c0op)),
            "frac_own_closer_v10def": {"x": x_ls, "n": n, "p": x_ls / n, "cp95": cp95(x_ls, n),
                                       "d_own_mean": own_ls, "d_other_mean": oth_ls},
            "frac_own_closer_seedW0": {"x": x_sd, "n": n, "p": x_sd / n, "cp95": cp95(x_sd, n),
                                       "d_own_mean": own_sd, "d_other_mean": oth_sd},
        }
        s = summary[cell]
        print(f"{cell:20s} n={len(rs)} replay_ok={s['replay_ok_all']} dev={s['replay_device_counts']} "
              f"mu_C med {s['mu_C_median']:.3f} CI {ci[0]:.3f}..{ci[1]:.3f} | growth med {s['growth_rel_median']:.2f} | "
              f"rho_end {s['rho_end_median']:.3f} | own-closer v1.0 {x_ls}/{n} seedW0 {x_sd}/{n} ({time.time() - t0:.0f} s)",
              flush=True)
    res = {"label": "KA6, v1.0 S2.2 main cohort, READ-ONLY, seen data (computed before the v1.1 freeze)",
           "sign": "C := P P^T - W^T W (paper); mu_C and growth are sign-free",
           "nboot": NBOOT, "summary": summary, "per_run": per_run,
           "v10_a4_reference": "results/review_final/s22-skeptic/a4_mechanism_eps.json (frac_own_closer)"}
    path = os.path.join(OUT, "ka6_v10.json")
    with open(path + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(path + ".tmp", path)
    # cross-check against the v1.0 a4 values
    a4 = json.load(open(os.path.join(ROOT, "results/review_final/s22-skeptic/a4_mechanism_eps.json")))["mechanism"]
    for cell, s in summary.items():
        if cell in a4:
            print(f"  a4 check {cell}: frac_own_closer v1.0 {a4[cell]['frac_own_closer']:.4f} vs recomputed "
                  f"{s['frac_own_closer_v10def']['p']:.4f}")
    print("wrote", path, hashlib.sha256(open(path, "rb").read()).hexdigest(), f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
