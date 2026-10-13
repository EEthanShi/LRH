"""E0 critic checks, part 3 (role e0-critic, 2026-10-10). EXPLORATORY; frozen S1.1 outputs only.
Operating characteristics of the proposed P2b mixing-weight rule (ViT-specific competitor, encoder replication,
gap-based guard). Writes analysis/e0_sentence/critic/critic_checks3.json."""
import json, os, hashlib
import numpy as np, pandas as pd
P = os.path.expanduser("~/hrh_v2_stage1_20261008")
OUT = os.path.join(P, "analysis/e0_sentence/critic/critic_checks3.json")
c = pd.read_parquet(f"{P}/results/S1_1/v11/S1_1_v11_cells_clip.parquet"); c = c[c.primary]
S = pd.read_parquet(f"{P}/results/S1_1/v11/S1_1_v11_slopes_clip.parquet"); S = S[S.primary]
t = c[(c.m == 1024) & (c.k == 10)].set_index("pair")[["cls", "obs"]].join(S[S.k == 10].set_index("pair")[["slope_obs"]])
vl = t[t.cls == "cross-modal"].copy(); vl["vit"] = [p.split("__")[0] for p in vl.index]
a, b, sd = 0.135706, 0.228054, 0.05647474462525915
vits = sorted(vl.vit.unique()); bMv = vl.groupby("vit").slope_obs.mean().reindex(vits).to_numpy()
vitL = np.log(vl.groupby("vit").obs.mean().reindex(vits).to_numpy()); vitL -= vitL.mean()
rng = np.random.default_rng(11)
def rule(bh, bo, Mv):
    gap = np.abs(bh - Mv)
    if np.median(gap) < 2 * sd: return "uninformative", np.nan
    u = (bh - Mv); v = (bo - Mv); lam = (u.ravel() @ v.ravel()) / (u.ravel() @ u.ravel())
    lam_e = np.array([(u[:, j] @ v[:, j]) / (u[:, j] @ u[:, j]) for j in range(3)])
    if lam >= 0.75 and lam_e.min() >= 0.5: return "holds", lam
    if lam <= 0.25 and lam_e.max() <= 0.5: return "fails", lam
    return "inconclusive", lam
SIM = {}
for medL in (0.06, 0.09, 0.12, 0.2, 0.3):
    for lam_true in (1.0, 0.75, 0.5, 0.25, 0.0):
        for sd_enc in (0.0, 0.05, 0.10):
            labs = []
            for _ in range(2000):
                L = np.exp(np.log(medL) + 0.5 * vitL[:, None] + rng.normal(0, 0.25, 3)[None, :]).clip(0.01, 0.9)
                bh = a + b * np.log(L); Mv = np.repeat(bMv[:, None], 3, 1)
                bo = lam_true * bh + (1 - lam_true) * Mv + rng.normal(0, sd_enc, 3)[None, :] + rng.normal(0, sd, (5, 3))
                labs.append(rule(bh, bo, Mv)[0])
            SIM[f"L{medL}_lam{lam_true}_enc{sd_enc}"] = {l: labs.count(l) / 2000 for l in ("holds", "inconclusive", "fails", "uninformative")}
json.dump({"rule": "guard: median_j |bhat(L_j)-bM_v(j)| >= 2*0.0565; lam = OLS through origin of (b-bM_v) on (bhat-bM_v); holds: lam>=0.75 and every per-encoder lam_e>=0.5; fails: lam<=0.25 and every lam_e<=0.5",
           "sim": SIM}, open(OUT, "w"), indent=1)
for k, v in SIM.items(): print(k, v)
print(OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest())
