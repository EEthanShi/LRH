"""Summarize a2_pairs.json: per cell, mean over pairs of Gs / 1-CKA / 1-s_N(10) for
yy (function), uu (actual u), common gauge (B's, A's, cell mean), and the split of the u-vs-y excess into
common-transform and mismatch parts. Also checks agreement with the frozen ladder values in s22_summary.json."""
import json, os
import numpy as np
from scipy.stats import spearmanr
ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
OUT = os.path.join(ROOT, "results/review_final/s22-skeptic")
P = json.load(open(os.path.join(OUT, "a2_pairs.json")))
S = json.load(open(os.path.join(ROOT, "results/S2_2/s22_summary.json")))
res = {}
order = ["noWD_25000", "WD_25000", "smallinit_25000", "noWD_250000", "WD_250000", "smallinit_250000", "noWD-shared_25000"]
for cell in order:
    R = sorted([p for p in P if p["cell"] == cell], key=lambda p: p["j"])
    if not R:
        continue
    out = {"n": len(R)}
    # agreement with frozen ladders
    gy = np.array(S["values"][f"{cell}|y"]["Gs"]); gu = np.array(S["values"][f"{cell}|u"]["Gs"])
    j = np.array([p["j"] for p in R])
    out["max_abs_dev_Gs_yy_vs_ladder"] = float(np.max(np.abs(np.array([p["Gs_yy"] for p in R]) - gy[j])))
    out["max_abs_dev_Gs_uu_vs_ladder"] = float(np.max(np.abs(np.array([p["Gs_uu"] for p in R]) - gu[j])))
    for k in ("Gs", "dCKA", "dN10"):
        v = {n: np.array([p[f"{k}_{n}"] for p in R]) for n in ("yy", "uu", "common_B", "common_A", "common_mean")}
        com = (v["common_A"] + v["common_B"]) / 2
        out[k] = {n: float(x.mean()) for n, x in v.items()}
        out[k]["excess_total"] = float((v["uu"] - v["yy"]).mean())
        out[k]["excess_common_pairgauge"] = float((com - v["yy"]).mean())
        out[k]["excess_mismatch"] = float((v["uu"] - com).mean())
        out[k]["frac_pairs_mismatch_pos"] = float(np.mean(v["uu"] > com))
    for k in ("within_Gs_uA_yA", "within_dCKA_uA_yA", "within_dN10_uA_yA", "Gs_tf32_noise", "yWu_relerr_A",
              "gauge_mismatch_AI", "gauge_A_to_mean_AI", "gauge_A_to_I_AI"):
        out[k] = float(np.mean([p[k] for p in R]))
    out["Gs_tf32_noise_max"] = float(np.max([p["Gs_tf32_noise"] for p in R]))
    mm = np.array([p["gauge_mismatch_AI"] for p in R]); ex = np.array([p["Gs_uu"] - (p["Gs_common_A"] + p["Gs_common_B"]) / 2 for p in R])
    r, pv = spearmanr(mm, ex); out["spearman_mismatchAI_vs_Gs_mismatch_excess"] = [float(r), float(pv)]
    res[cell] = out
    print(f"\n{cell} (n={len(R)}): ladder check max|dev| yy {out['max_abs_dev_Gs_yy_vs_ladder']:.1e} uu {out['max_abs_dev_Gs_uu_vs_ladder']:.1e}; "
          f"TF32 noise Gs(u_B, W_B^-1 y_B) mean {out['Gs_tf32_noise']:.1e} max {out['Gs_tf32_noise_max']:.1e}; y=Wu relerr {out['yWu_relerr_A']:.1e}")
    for k in ("Gs", "dCKA", "dN10"):
        o = out[k]
        print(f"  {k:5s} yy {o['yy']:.4f} | common(A,B avg) {(o['common_A']+o['common_B'])/2:.4f} common(mean) {o['common_mean']:.4f} | uu {o['uu']:.4f}"
              f" || excess total {o['excess_total']:+.4f} = common {o['excess_common_pairgauge']:+.4f} + mismatch {o['excess_mismatch']:+.4f} (mismatch>0 in {o['frac_pairs_mismatch_pos']:.2f})")
    print(f"  within-run u vs y: Gs {out['within_Gs_uA_yA']:.4f}, 1-CKA {out['within_dCKA_uA_yA']:.4f}, 1-s_N(10) {out['within_dN10_uA_yA']:.4f}; "
          f"gauge mismatch d_AI {out['gauge_mismatch_AI']:.3f}, to cell mean {out['gauge_A_to_mean_AI']:.3f}, to I {out['gauge_A_to_I_AI']:.3f}; "
          f"spearman(mismatch d_AI, Gs mismatch excess) {r:+.2f} (p={pv:.2g})")
json.dump(res, open(os.path.join(OUT, "a2b_summary.json"), "w"), indent=1)

# ---- decisions at the frozen eps(y) for the counterfactual 'shared gauge' representations
from scipy.stats import beta as _beta
E = json.load(open(os.path.join(ROOT, "prereg/eps_S2_2.json")))["eps"]["y"]
def _dec(x, n):
    lo = _beta.ppf(0.05, x, n - x + 1) if x > 0 else 0.0
    hi = _beta.ppf(0.95, x + 1, n - x) if x < n else 1.0
    return "rel" if lo > 0.9 else ("unr" if hi < 0.8 else "inc")
print("\n=== decisions at frozen eps(y): x/96 for y, actual u, and u re-expressed with ONE shared gauge ===")
res["counterfactual_decisions"] = {}
for cell in order:
    R = [p for p in P if p["cell"] == cell]
    if len(R) < 96:
        continue
    row = {}
    for k, ek in (("Gs", "Gs"), ("dCKA", "dCKA"), ("dN10", "dN_k10")):
        e = E[ek]
        for n in ("yy", "uu", "common_B", "common_A", "common_mean"):
            x = int(np.sum(np.array([p[f"{k}_{n}"] for p in R]) <= e))
            row[f"{k}|{n}"] = f"{_dec(x, 96)} {x}"
    res["counterfactual_decisions"][cell] = row
    print(cell, {kk: vv for kk, vv in row.items() if not kk.startswith("dN10") or kk.endswith(("uu", "common_mean"))})
json.dump(res, open(os.path.join(OUT, "a2b_summary.json"), "w"), indent=1)

# ---- isolate the run-specific (mismatch) part: tolerance from SHARED-gauge u of the same cell (split half), and
# ---- the counterfactual 'mismatch only' Delta = Delta(y) + [Delta(u) - Delta(shared gauge)] judged at the frozen eps(y)
print("\n=== mismatch-isolating tests (exploratory) ===")
res["mismatch_tests"] = {}
for cell in order:
    R = sorted([p for p in P if p["cell"] == cell], key=lambda p: p["j"])
    if len(R) < 96:
        continue
    row = {}
    for k, ek in (("Gs", "Gs"), ("dCKA", "dCKA"), ("dN10", "dN_k10")):
        yy = np.array([p[f"{k}_yy"] for p in R]); uu = np.array([p[f"{k}_uu"] for p in R]); cm = np.array([p[f"{k}_common_mean"] for p in R])
        ev, od = np.arange(0, 96, 2), np.arange(1, 96, 2)
        x_split = int(np.sum(uu[od] <= cm[ev].max()) + np.sum(uu[ev] <= cm[od].max()))
        x_ctrl = int(np.sum(cm[od] <= cm[ev].max()) + np.sum(cm[ev] <= cm[od].max()))
        mo = yy + (uu - cm)
        x_mo = int(np.sum(mo <= E[ek]))
        row[k] = {"u_vs_sharedgauge_tol_splithalf": f"{_dec(x_split, 96)} {x_split}", "sharedgauge_selfcheck": f"{_dec(x_ctrl, 96)} {x_ctrl}",
                  "mismatch_only_at_frozen_eps": f"{_dec(x_mo, 96)} {x_mo}", "mismatch_over_sd_y": float((uu - cm).mean() / yy.std(ddof=1))}
    res["mismatch_tests"][cell] = row
    print(cell, row)
json.dump(res, open(os.path.join(OUT, "a2b_summary.json"), "w"), indent=1)
