"""Independent re-computation of the S1.1 v1.1 descriptive statistics (plan V5)
directly from the unit JSONs + obs parquet, without importing s11v11_evaluate.
Role s11-run (resume), 2026-10-09. Read-only on results; prints JSON to stdout."""
import glob, json, os, sys
import numpy as np, pandas as pd

P = os.path.expanduser("~/hrh_v2_stage1_20261008/results/S1_1/v11")
cfg = sys.argv[1] if len(sys.argv) > 1 else "clip"
U = "iii-LSlog-EB"

obs = pd.read_parquet(f"{P}/obs/obs_{cfg}.parquet")[["pair", "cls", "primary", "draw", "m", "k", "n_anchors", "chance", "obs"]]
rows = []
for kind in ("nulls", "iii"):
    for f in glob.glob(f"{P}/units/{cfg}/{kind}/*.json"):
        u = json.load(open(f))
        for c in u["cells"]:
            r = {"kind": kind, "pair": u["pair"], "draw": int(u["draw"]), "dir": u["direction"], "m": int(c["m"]), "k": int(c["k"]),
                 "uobs": c.get("obs"), "N1": c.get("N1"), U: c.get(U)}
            rows.append(r)
R = pd.DataFrame(rows)
for c in ("uobs", "N1", U):
    R[c] = pd.to_numeric(R[c], errors="coerce").astype(float)
key = ["pair", "draw", "m", "k"]
# N1: prefer nulls unit, else iii unit (identical where both exist)
n1 =R[R.kind == "nulls"].set_index(key + ["dir"])["N1"].combine_first(R[R.kind == "iii"].set_index(key + ["dir"])["N1"])
n1 = n1.unstack("dir").mean(axis=1, skipna=False).rename("N1")
D = obs.set_index(key).join(n1)
if (R.kind == "iii").any():
    ui = R[R.kind == "iii"].set_index(key + ["dir"])[U].unstack("dir").mean(axis=1, skipna=False).rename(U)
    D = D.join(ui)
else:
    D[U] = np.nan
D = D.reset_index()
fit = D[D.m == 1024].set_index(["pair", "draw", "k"])["obs"].rename("N0")
D = D.join(fit, on=["pair", "draw", "k"])
D["eps"] = 0.5 / (D.n_anchors * D.k)

def slope(x, y):
    return np.polyfit(np.log(x), y, 1)[0]

S = []
for (p, k, d), g in D.groupby(["pair", "k", "draw"]):
    g = g.sort_values("m")
    r = {"pair": p, "k": k, "draw": d, "cls": g.cls.iloc[0], "primary": g.primary.iloc[0]}
    for X in ("obs", "N0", "N1", U):
        y = g[X].to_numpy(float)
        r[X] = slope(g.m.to_numpy(float), np.log(np.maximum(y, g.eps.to_numpy(float)))) if np.isfinite(y).all() and len(y) == 7 else np.nan
    S.append(r)
S = pd.DataFrame(S)
MS = S.groupby(["pair", "k", "cls", "primary"])[["obs", "N0", "N1", U]].mean().reset_index()
MS["sc"] = np.where(MS.obs >= 0.1, "rising", np.where(MS.obs <= -0.1, "falling", "flat"))

C = D.groupby(["pair", "cls", "primary", "m", "k"]).agg(obs=("obs", "mean"), N0=("N0", "mean"), N1=("N1", "mean"),
                                                         X=(U, lambda s: s.mean(skipna=False)), na=("n_anchors", "mean"),
                                                         nd=("draw", "nunique")).reset_index().rename(columns={"X": U})
C["epsb"] = 0.5 / (C.nd * C.na * C.k)

out = {"config": cfg, "slope_classes": {}, "accuracy": {}, "exploratory_signed": {}}
for setname, sel in (("primary", MS.primary), ("all", MS.primary | ~MS.primary)):
    M = MS[sel]
    out["slope_classes"][setname] = {f"{cl}|k={k}": [int((g.sc == "rising").sum()), int((g.sc == "flat").sum()),
                                                      int((g.sc == "falling").sum()), round(float(g.obs.median()), 3)]
                                     for (cl, k), g in M.groupby(["cls", "k"])}
for cl in ("pooled", "seed-seed", "cross-vision", "cross-modal"):
    pc = C[C.primary & ((C.cls == cl) | (cl == "pooled"))]
    for X in ("N0", "N1", U):
        okp = pc.groupby("pair")[X].apply(lambda s: s.notna().all())
        pp = okp[okp].index
        cc = pc[pc.pair.isin(pp) & (pc.m != 1024)]
        el = cc[cc.obs > 2 * cc.k / (cc.m - 1)]
        lr = np.where(el[X] > 0, np.abs(np.log(el[X] / el.obs)), np.inf)
        Mp = MS[MS.pair.isin(pp)]
        nf = Mp[Mp.obs.abs() >= 0.1]
        nfk = set(zip(nf.pair, nf.k))
        sc = cc[[(a, b) in nfk for a, b in zip(cc.pair, cc.k)]]
        msle = float(np.mean((np.log(np.maximum(sc[X], sc.epsb)) - np.log(np.maximum(sc.obs, sc.epsb))) ** 2)) if len(sc) else np.nan
        ps = np.where(np.abs(nf[X]) < 1e-12, 0, np.sign(nf[X]))
        sign = float(np.mean(ps == np.sign(nf.obs))) if len(nf) else np.nan
        out["accuracy"][f"{X}|{cl}"] = {"pairs": len(pp), "median_abs_log": round(float(np.median(lr)), 3), "n_cells": len(el),
                                        "msle_nonflat": round(msle, 3), "n_nonflat": len(nf), "sign": round(sign, 3)}
        if X != "N0":
            out["exploratory_signed"][f"{X}|{cl}"] = {
                "median_signed_log_ratio": round(float(np.median(np.log(el[X] / el.obs)[el[X] > 0])), 3),
                "frac_over": round(float(np.mean(el[X] > el.obs)), 3),
                "median_pred_slope_by_k": {int(k): round(float(g[X].median()), 3) for k, g in Mp.groupby("k")},
                "median_obs_slope_by_k": {int(k): round(float(g.obs.median()), 3) for k, g in Mp.groupby("k")}}
print(json.dumps(out, indent=1))
