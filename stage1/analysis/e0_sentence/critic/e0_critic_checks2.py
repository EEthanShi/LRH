"""E0 critic checks, part 2 (role e0-critic, 2026-10-10). EXPLORATORY; frozen S1.1 outputs only; no sentence-encoder data.
Fixes the sign of the anchored power-law check of part 1, adds the ViT-specific competitor, the k=1 share of the
elasticity sums, the gap-based guard thresholds and a power simulation of a mixing-weight rule.
Writes analysis/e0_sentence/critic/critic_checks2.json."""
import json, os, hashlib
import numpy as np, pandas as pd
P = os.path.expanduser("~/hrh_v2_stage1_20261008")
OUT = os.path.join(P, "analysis/e0_sentence/critic/critic_checks2.json")
c = pd.read_parquet(f"{P}/results/S1_1/v11/S1_1_v11_cells_clip.parquet"); c = c[c.primary]
S = pd.read_parquet(f"{P}/results/S1_1/v11/S1_1_v11_slopes_clip.parquet"); S = S[S.primary]
t = c[(c.m == 1024) & (c.k == 10)].set_index("pair")[["cls", "obs"]].join(S[S.k == 10].set_index("pair")[["slope_obs"]])
x = np.log(t.obs.to_numpy()); y = t.slope_obs.to_numpy(); cls = t.cls.to_numpy()
a, b = 0.135706, 0.228054; bM = -0.570555
R = {}
b0 = x / np.log(1024 / 10)
R["anchored_powerlaw_fixed"] = {"formula": "b0 = ln L / ln(m_ref/k) (power law through s=1 at k/m=1)", "coef": float(1 / np.log(1024 / 10)),
    "resid_mean_by_class": {k_: float((y - b0)[cls == k_].mean()) for k_ in ("cross-modal", "cross-vision", "seed-seed")},
    "resid_sd": float((y - b0).std()), "fitted_line_coef_for_comparison": b}
vl = t[t.cls == "cross-modal"].copy(); vl["vit"] = [p.split("__")[0] for p in vl.index]
R["vit_specific_vl_slope"] = vl.groupby("vit").slope_obs.mean().round(6).to_dict()
R["vit_specific_vl_level"] = vl.groupby("vit").obs.mean().round(6).to_dict()
# gap-based guard thresholds
sd = 0.05647474462525915
for mult in (1.5, 2.0):
    g = mult * sd
    R[f"guard_L_thresholds_gap_{mult}sd"] = {"upper_L": float(np.exp((bM - a + g) / b)), "lower_L": float(np.exp((bM - a - g) / b))}
R["line_crosses_beta_M_at_L"] = float(np.exp((bM - a) / b))
# k = 1 share of the S1.1 sums: sums with k in {1,10} and {10,100}
rows = []
for p, g in c.groupby("pair"):
    s = np.maximum(g.obs.to_numpy(), 0.5 / (5 * g.n_anchors.to_numpy() * g.k.to_numpy()))
    lk = np.log(g.k.to_numpy(float)); lm = np.log(g.m.to_numpy(float)); ls = np.log(s)
    X = np.column_stack([np.ones_like(lk), lk, lm])
    out = {"pair": p, "cls": g.cls.iloc[0]}
    for name, msk in (("all", np.ones(len(g), bool)), ("k1_10", g.k.to_numpy() <= 10), ("k10_100", g.k.to_numpy() >= 10)):
        e = np.linalg.lstsq(X[msk], ls[msk], rcond=None)[0]; out[f"sum_{name}"] = e[1] + e[2]; out[f"em_{name}"] = e[2]
    rows.append(out)
D = pd.DataFrame(rows)
R["sums_by_k_subset"] = {k_: {col: [float(g[col].median()), float(g[col].abs().max())] for col in ("sum_all", "sum_k1_10", "sum_k10_100", "em_k10_100")}
                         for k_, g in D.groupby("cls")}
R["sums_by_k_subset_note"] = "[median, max abs] per class"
# mixing-weight rule power: b_j = lam*bhat(L_j) + (1-lam)*bM_v + noise ; estimate lam by OLS through origin on (b - bM_v) ~ (bhat - bM_v)
rng = np.random.default_rng(7)
vits = sorted(vl.vit.unique()); bMv = vl.groupby("vit").slope_obs.mean().reindex(vits).to_numpy()
vitL = np.log(vl.groupby("vit").obs.mean().reindex(vits).to_numpy()); vitL = vitL - vitL.mean()
def sim(medL, lam_true, sd_enc, comp, nsim=1000):
    lab = []; lams = []
    for _ in range(nsim):
        L = np.exp(np.log(medL) + 0.5 * vitL[:, None] + rng.normal(0, 0.25, 3)[None, :]).clip(0.01, 0.9)
        bh = a + b * np.log(L)
        Mv = np.repeat(bMv[:, None], 3, 1) if comp == "vit" else np.full((5, 3), bM)
        truthM = np.repeat(bMv[:, None], 3, 1)  # modality truth carries the S1.1 ViT effect
        bo = lam_true * bh + (1 - lam_true) * truthM + rng.normal(0, sd_enc, 3)[None, :] + rng.normal(0, sd, (5, 3))
        u = (bh - Mv).ravel(); v = (bo - Mv).ravel(); lam = (u @ v) / (u @ u)
        # encoder-level replication: lambda per encoder
        lam_e = [((bh - Mv)[:, j] @ (bo - Mv)[:, j]) / ((bh - Mv)[:, j] @ (bh - Mv)[:, j]) for j in range(3)]
        hold = lam >= 0.75 and min(lam_e) > 0.5
        fail = lam <= 0.25 or max(lam_e) < 0.5
        lab.append("holds" if hold else ("fails" if fail else "inconclusive")); lams.append(lam)
    return {l: lab.count(l) / nsim for l in ("holds", "inconclusive", "fails")} | {"lam_median": float(np.median(lams)), "lam_q05_q95": [float(np.quantile(lams, .05)), float(np.quantile(lams, .95))]}
SIM = {}
for medL in (0.12, 0.2, 0.3):
    for lam_true in (1.0, 0.5, 0.0):
        for sd_enc in (0.0, 0.05):
            for comp in ("vit", "const"):
                SIM[f"L{medL}_lam{lam_true}_enc{sd_enc}_{comp}"] = sim(medL, lam_true, sd_enc, comp)
R["mixing_rule_power"] = SIM
json.dump(R, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "mixing_rule_power"}, indent=1))
for k, v in SIM.items(): print(k, v)
print(OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest())
