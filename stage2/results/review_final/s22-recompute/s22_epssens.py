"""Exploratory eps sensitivity for S2.2 (review s22-recompute): secondary eps_q90 decisions and
a bootstrap of the 24 pilot pairs (eps* = max of a resample) for the knife-edge u cells."""
import json, os
import numpy as np
from scipy.stats import beta as B
OUT = os.path.expanduser("~/hrh_v2_stage2_20261008/results/review_final/s22-recompute")
P = json.load(open(os.path.join(OUT, "recompute_pairs.json")))
E = json.load(open(os.path.expanduser("~/hrh_v2_stage2_20261008/prereg/eps_S2_2.json")))
S = json.load(open(os.path.expanduser("~/hrh_v2_stage2_20261008/results/S2_2/s22_summary.json")))
def dec(x, n=96):
    lo = 0 if x == 0 else B.ppf(.05, x, n - x + 1); hi = 1 if x == n else B.ppf(.95, x + 1, n - x)
    return "rel" if lo > .9 else "unr" if hi < .8 else "inc"
vals = lambda s, tag, k: np.array([P["pairs"][f"{a}__{b}"]["indep"][tag][k] for a, b in P["sets"][s]])
out = {"q90": {}, "boot": {}}
for s in P["sets"]:
    for tag in ("y", "u", "h"):
        for k in ("dN_k10", "Gs", "dCKA", "dPi"):
            e = E["eps_q90"][tag][k]
            x = int(np.sum(vals(s, tag, k) <= e))
            out["q90"][f"{s}|{tag}|{k}"] = {"eps_q90": e, "x": x, "dec": dec(x),
                                            "summary_p_hat_q90": S["p_hat_q90"][f"{s}|{tag}"][k]}
rows = [r for r in E["pilot_rows"] if r["rep"] == "y"]
rng = np.random.default_rng(2026)
for s in ("noWD_250000", "smallinit_250000", "WD_250000"):
    for k in ("Gs", "dCKA", "dN_k10"):
        pv = np.array([r[k] for r in rows]); v = vals(s, "u", k)
        decs = []
        for _ in range(2000):
            e = pv[rng.integers(0, pv.size, pv.size)].max()
            decs.append(dec(int(np.sum(v <= e))))
        loo = [dec(int(np.sum(v <= np.delete(pv, i).max()))) for i in range(pv.size)]
        out["boot"][f"{s}|u|{k}"] = {d: decs.count(d) / 2000 for d in ("rel", "inc", "unr")} | {
            "loo": {d: loo.count(d) for d in ("rel", "inc", "unr")}}
json.dump(out, open(os.path.join(OUT, "eps_sensitivity.json"), "w"), indent=1)
for k, v in out["q90"].items():
    if "250000" in k and k.split("|")[1] in ("y", "u") and k.split("|")[2] in ("dN_k10", "Gs", "dCKA"):
        print(k, v["x"], v["dec"], round(v["eps_q90"], 4), "summary p", round(v["summary_p_hat_q90"], 4), "match", abs(v["x"] / 96 - v["summary_p_hat_q90"]) < 1e-12)
mm = [k for k, v in out["q90"].items() if abs(v["x"] / 96 - v["summary_p_hat_q90"]) > 1e-12]
print("q90 mismatches vs summary:", mm)
for k, v in out["boot"].items(): print(k, v)
