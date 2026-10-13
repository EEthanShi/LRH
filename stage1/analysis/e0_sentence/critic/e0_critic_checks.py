"""E0 critic checks (role e0-critic, 2026-10-10). EXPLORATORY; reads only frozen S1.1 outputs.
No sentence-encoder representation exists or is read. Writes analysis/e0_sentence/critic/critic_checks.json.
Single-threaded (BLAS threads set to 1 by the caller)."""
import json, os, hashlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

P = os.path.expanduser("~/hrh_v2_stage1_20261008")
OUT = os.path.join(P, "analysis/e0_sentence/critic/critic_checks.json")
R = {}
c_all = pd.read_parquet(f"{P}/results/S1_1/v11/S1_1_v11_cells_clip.parquet")
S_all = pd.read_parquet(f"{P}/results/S1_1/v11/S1_1_v11_slopes_clip.parquet")
c = c_all[c_all.primary]; S = S_all[S_all.primary]
summ = json.load(open(f"{P}/results/S1_1/v11/S1_1_v11_summary.json"))
el = pd.DataFrame(summ["configs"]["clip"]["secondary"]["primary"]["elasticity_per_pair"]).set_index("pair")

def lvl_slope(k, mref=1024, col="obs", scol="slope_obs"):
    t = c[(c.m == mref) & (c.k == k)].set_index("pair")[["cls", col]].rename(columns={col: "L"})
    return t.join(S[S.k == k].set_index("pair")[[scol]].rename(columns={scol: "b"}))

def ols(x, y):
    X = np.column_stack([np.ones_like(x), x]); beta, *_ = np.linalg.lstsq(X, y, rcond=None); return beta, y - X @ beta

# 1. calibration, LOO, class centroids, within-class slopes
t = lvl_slope(10); x = np.log(t.L.to_numpy()); y = t.b.to_numpy(); cls = t.cls.to_numpy()
beta, r = ols(x, y)
loo = np.array([y[i] - (lambda bb: bb[0] + bb[1] * x[i])(ols(np.delete(x, i), np.delete(y, i))[0]) for i in range(len(x))])
cent = {k_: (float(x[cls == k_].mean()), float(y[cls == k_].mean())) for k_ in ("cross-modal", "cross-vision", "seed-seed")}
cx = np.array([v[0] for v in cent.values()]); cy = np.array([v[1] for v in cent.values()])
cb, cr = ols(cx, cy)
within = {}
for k_ in ("cross-modal", "cross-vision", "seed-seed"):
    m_ = cls == k_; bb, rr = ols(x[m_], y[m_]); within[k_] = {"slope_on_lnL": float(bb[1]), "n": int(m_.sum()),
        "spearman": float(spearmanr(x[m_], y[m_]).correlation)}
# pooled within-class (fixed effects) slope
xd = np.concatenate([x[cls == k_] - x[cls == k_].mean() for k_ in cent]); yd = np.concatenate([y[cls == k_] - y[cls == k_].mean() for k_ in cent])
fe_slope = float((xd @ yd) / (xd @ xd))
R["calibration_k10"] = {"a": float(beta[0]), "b": float(beta[1]), "resid_sd_ddof2": float(r.std(ddof=2)),
    "loo_rmse": float(np.sqrt((loo ** 2).mean())), "loo_max_abs": float(np.abs(loo).max()),
    "loo_by_class_mean": {k_: float(loo[cls == k_].mean()) for k_ in cent},
    "class_centroids_lnL_b": cent, "line_through_centroids": {"a": float(cb[0]), "b": float(cb[1]), "centroid_resid": cr.tolist()},
    "within_class": within, "pooled_within_class_slope_on_lnL": fe_slope}
# VL two-way decomposition (5 ViT x 3 LM) of slope and lnL
vl = t[t.cls == "cross-modal"].copy(); vl["vit"] = [p.split("__")[0] for p in vl.index]; vl["txt"] = [p.split("__")[1] for p in vl.index]
def twoway(v):
    g = v.mean(); a = vl.groupby("vit")[v.name].transform("mean") - g if False else None
for col in ("b",):
    pass
dd = {}
for name, vals in (("b", vl.b), ("lnL", np.log(vl.L))):
    d = pd.DataFrame({"v": vals.to_numpy(), "vit": vl.vit.to_numpy(), "txt": vl.txt.to_numpy()})
    g = d.v.mean(); ve = d.groupby("vit").v.transform("mean") - g; te = d.groupby("txt").v.transform("mean") - g
    res = d.v - g - ve - te; tot = ((d.v - g) ** 2).sum()
    dd[name] = {"frac_vit": float((ve ** 2).sum() / tot), "frac_text": float((te ** 2).sum() / tot), "frac_resid": float((res ** 2).sum() / tot),
                "sd_vit_effect": float(d.groupby("vit").v.mean().std(ddof=1)), "sd_text_effect": float(d.groupby("txt").v.mean().std(ddof=1)),
                "sd_resid": float(np.sqrt((res ** 2).sum() / ((5 - 1) * (3 - 1))))}
R["vl_twoway"] = dd
# 2. zero-parameter anchored prediction and first-order form
b0 = x / np.log(10 / 1024)
R["anchored_powerlaw"] = {"formula": "b0 = ln L / ln(k/m_ref)", "coef": float(1 / np.log(10 / 1024)),
    "resid_sd": float((y - b0).std()), "mean_resid_by_class": {k_: float((y - b0)[cls == k_].mean()) for k_ in cent},
    "median_abs_resid": float(np.median(np.abs(y - b0)))}
f1 = (1 - t.L.to_numpy()) / t.L.to_numpy(); bb1, rr1 = ols(f1, y)
R["first_order_form"] = {"fit": "b = a + c (1-L)/L", "a": float(bb1[0]), "c": float(bb1[1]), "resid_sd": float(rr1.std(ddof=2))}
# 3. N1: its own (level, slope) points vs the obs line
tn = lvl_slope(10, col="N1", scol="slope_N1"); xn = np.log(tn.L.to_numpy()); yn = tn.b.to_numpy()
pred = beta[0] + beta[1] * xn; bn, rn = ols(xn, yn)
R["N1_vs_line"] = {"N1_level_over_obs_median": float(np.median(tn.L.to_numpy() / t.L.to_numpy())),
    "N1_slope_minus_obs_slope_median": float(np.median(yn - y)),
    "N1_points_minus_obsline_median": float(np.median(yn - pred)), "N1_points_minus_obsline_sd": float((yn - pred).std()),
    "N1_own_line": {"a": float(bn[0]), "b": float(bn[1]), "resid_sd": float(rn.std(ddof=2))},
    "N1_points_minus_obsline_by_class": {k_: float((yn - pred)[cls == k_].mean()) for k_ in cent}}
# 4. P1 statistics: sums, ratios, power against m-only, curvature-robust collapse coefficient, k in {10,100} only
cd = c[["pair", "cls", "m", "k", "obs", "eps", "n_anchors"]].copy()
rows = []
for p, g in cd.groupby("pair"):
    s = np.maximum(g.obs.to_numpy(), 0.5 / (5 * g.n_anchors.to_numpy() * g.k.to_numpy()))
    lk = np.log(g.k.to_numpy(float)); lm = np.log(g.m.to_numpy(float)); u = lk - lm; ls = np.log(s)
    X = np.column_stack([np.ones_like(lk), lk, lm]); e = np.linalg.lstsq(X, ls, rcond=None)[0]
    Xq = np.column_stack([np.ones_like(u), u, u ** 2, lk]); q = np.linalg.lstsq(Xq, ls, rcond=None)[0]
    rq = ls - Xq @ q
    m2 = g.k.to_numpy() >= 10; e2 = np.linalg.lstsq(X[m2], ls[m2], rcond=None)[0]
    rows.append({"pair": p, "cls": g.cls.iloc[0], "e_k": e[1], "e_m": e[2], "sum": e[1] + e[2], "ratio_ek_over_minus_em": e[1] / -e[2] if e[2] != 0 else np.nan,
                 "d_kfixed_kOverM_quad": q[3], "quad_rmse": float(np.sqrt((rq ** 2).mean())),
                 "sum_k10_100": e2[1] + e2[2], "e_m_k10_100": e2[2]})
P1 = pd.DataFrame(rows).set_index("pair")
chk = float(np.abs(P1["sum"] - el.loc[P1.index, "sum"]).max())
R["P1"] = {"reproduce_sum_maxabs_diff_vs_summary": chk,
    "by_class": {k_: {"median_sum": float(g["sum"].median()), "max_abs_sum": float(g["sum"].abs().max()),
                      "median_e_m": float(g.e_m.median()), "median_ratio": float(g.ratio_ek_over_minus_em.median()),
                      "median_rel_sum": float((g["sum"] / g.e_m.abs()).median()),
                      "median_d_quad": float(g.d_kfixed_kOverM_quad.median()), "max_abs_d_quad": float(g.d_kfixed_kOverM_quad.abs().max()),
                      "median_sum_k10_100": float(g.sum_k10_100.median()), "max_abs_sum_k10_100": float(g.sum_k10_100.abs().max())}
                 for k_, g in P1.groupby("cls")},
    "n_pairs_abs_em_le_0.15": int((P1.e_m.abs() <= 0.15).sum()),
    "pairs_abs_em_le_0.15": P1.index[P1.e_m.abs() <= 0.15].tolist(),
    "spearman_abs_sum_vs_lnL": float(spearmanr(np.abs(P1.loc[t.index, "sum"]), x).correlation)}
R["P1_per_pair"] = P1.round(5).reset_index().to_dict(orient="records")
# 5. guard: gap between level prediction and modality prediction
bM = float(np.median(vl.b))
gap = {f"{L:.4f}": float(abs(beta[0] + beta[1] * np.log(L) - bM)) for L in (0.0327, 0.05, 0.0981, 0.12, 0.2, 0.3, 0.38)}
R["guard_gap"] = {"beta_M": bM, "gap_by_level": gap, "resid_sd": float(r.std(ddof=2))}
# 6. power simulation for P2b / P2 (draft rules vs cluster-aware alternatives)
rng = np.random.default_rng(20261010)
oldL = vl.L.to_numpy(); oldb = vl.b.to_numpy(); oldvit = vl.vit.to_numpy()
vits = sorted(set(oldvit)); vit_eff_L = vl.groupby("vit").L.apply(lambda s: np.log(s).mean() - np.log(vl.L).mean())
sd_v = dd["b"]["sd_vit_effect"]; sd_res = float(r.std(ddof=2))
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for i, j in enumerate(o): run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1, run)
    return adj
def sim(medL, truth, sd_enc, nsim=400, nperm=1000):
    out = {"draft": {}, "cluster": {}}
    labs_d = []; labs_c = []; p2_lab = []; p2_textlevel = []
    for _ in range(nsim):
        encL = rng.normal(0, 0.25, 3); Lnew = np.array([[np.exp(np.log(medL) + vit_eff_L[v] * 0.5 + encL[e]) for e in range(3)] for v in vits]).ravel()
        Lnew = np.clip(Lnew, 0.01, 0.9)
        pl = beta[0] + beta[1] * np.log(Lnew)
        base = {"level": pl, "modality": np.full(15, bM), "halfway": 0.5 * (pl + bM)}[truth]
        enc = rng.normal(0, sd_enc, 3); vv = rng.normal(0, sd_v, 5)
        bnew = base + np.repeat(vv, 3) + np.tile(enc, 5) + rng.normal(0, sd_res, 15)
        eL = np.abs(bnew - pl); eM = np.abs(bnew - bM)
        # draft P2b (ignoring the guard here; guard reported separately)
        try: pw = wilcoxon(eM - eL, alternative="greater", method="exact").pvalue
        except ValueError: pw = 1.0
        # draft P2 within-ViT permutation
        L30 = np.concatenate([oldL, Lnew]); b30 = np.concatenate([oldb, bnew]); v30 = np.concatenate([oldvit, np.repeat(vits, 3)])
        rho = spearmanr(L30, b30).correlation
        rL = pd.Series(L30).rank().to_numpy(); rb = pd.Series(b30).rank().to_numpy()
        idx_by_v = [np.where(v30 == v)[0] for v in vits]
        perm = np.tile(np.arange(30), (nperm, 1))
        for ii in idx_by_v:
            pr = np.argsort(rng.random((nperm, len(ii))), axis=1); perm[:, ii] = ii[pr]
        rbp = rb[perm]; rLc = rL - rL.mean(); rbpc = rbp - rbp.mean(axis=1, keepdims=True)
        rhos = (rbpc @ rLc) / (np.sqrt((rbpc ** 2).sum(1)) * np.sqrt((rLc ** 2).sum()))
        p2 = (1 + (rhos >= rho).sum()) / (1 + nperm)
        adj = holm(np.array([p2, pw]))
        hold = (np.median(eL) <= 0.5 * np.median(eM)) and (np.median(eL) <= 0.10) and adj[1] < 0.05
        fail = np.median(eL) >= np.median(eM)
        labs_d.append("holds" if hold else ("fails" if fail else "inconclusive"))
        p2_lab.append("holds" if (rho >= 0.5 and adj[0] < 0.05) else ("fails" if rho <= 0.2 else "inconclusive"))
        # cluster-aware alternative: effect-size rule must hold in each encoder (median over its 5 ViTs) and in each leave-one-encoder-out set
        E = (eM - eL).reshape(5, 3); eLm = eL.reshape(5, 3); eMm = eM.reshape(5, 3)
        per_enc = [(np.median(eLm[:, j]) <= 0.5 * np.median(eMm[:, j])) and (np.median(eLm[:, j]) <= 0.10) for j in range(3)]
        hold_c = hold and all(per_enc)
        fail_c = sum(np.median(eLm[:, j]) >= np.median(eMm[:, j]) for j in range(3)) >= 2
        labs_c.append("holds" if hold_c else ("fails" if fail_c else "inconclusive"))
        # text-model-level P2: 6 text models, mean over ViTs, exact over 720 permutations -> rho at 6 units
        tL = np.concatenate([vl.groupby("txt").L.apply(lambda s: np.log(s).mean()).to_numpy(), np.log(Lnew).reshape(5, 3).mean(0)])
        tb = np.concatenate([vl.groupby("txt").b.mean().to_numpy(), bnew.reshape(5, 3).mean(0)])
        p2_textlevel.append(spearmanr(tL, tb).correlation)
    for name, labs in (("draft", labs_d), ("cluster", labs_c)):
        out[name] = {l: labs.count(l) / nsim for l in ("holds", "inconclusive", "fails")}
    out["P2_draft"] = {l: p2_lab.count(l) / nsim for l in ("holds", "inconclusive", "fails")}
    out["P2_textlevel_rho_median"] = float(np.median(p2_textlevel))
    return out
SIM = {}
for medL in (0.07, 0.12, 0.2, 0.3):
    for truth in ("level", "halfway", "modality"):
        for sd_enc in (0.0, 0.05):
            SIM[f"L{medL}_{truth}_encsd{sd_enc}"] = sim(medL, truth, sd_enc)
R["power_sim"] = {"note": "new pairs: 5 ViT x 3 encoders; slope = truth + ViT effect (sd from S1.1 VL two-way) + encoder effect + resid (S1.1 calib resid sd); guard not applied; 400 sims, 1000 within-ViT perms",
                  "sd_vit": sd_v, "sd_resid": sd_res, "results": SIM}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(R, open(OUT, "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if k not in ("P1_per_pair", "power_sim")}, indent=1, default=float))
for k, v in SIM.items(): print(k, v["draft"], v["cluster"], v["P2_draft"], round(v["P2_textlevel_rho_median"], 2))
print(OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest())
