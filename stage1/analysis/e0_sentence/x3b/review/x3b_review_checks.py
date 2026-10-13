"""x3b-review checks (EXPLORATORY, descriptive, post hoc). CPU only; reads X3b outputs, writes one JSON.

1. Re-derive the binding X3b labels from X3B_results.json per_pair with the rules of X3B_READINGS.md sec. 5,
   independently of x3b_report.py's aggregation.
2. Direct matched-k/m check (no power-law fit): ln L(k=100, 10 m') - ln L(k=10, m') for m' in {256, 512, 1024},
   with L(100, .) interpolated linearly in (ln m, ln L) between pool sizes; same for k=1 vs k=10.
3. Level-slope lines of S1.1/E0 applied to the last-block pairs (clip line: ANALYSIS_PLAN_E0.md 7.3;
   ctr line: results/E0_sentence/eval/ctr_calibration.json lines['k10|m1024']); residual = b(10) - min(line, 0).
4. Slope classes at k = 100 (|b| < 0.1 flat), last vs mid.
"""
import hashlib, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/hrh_v2_stage1_20261008")
X = os.path.join(ROOT, "results/E0_sentence/x3b")
OUT = os.path.join(X, "review/X3B_review_checks.json")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


inputs = {
    "X3B_results.json": os.path.join(X, "X3B_results.json"),
    "X3B_levels.parquet": os.path.join(X, "eval/X3B_levels.parquet"),
    "X3B_slopes.parquet": os.path.join(X, "eval/X3B_slopes.parquet"),
    "X3B_pairs.parquet": os.path.join(X, "eval/X3B_pairs.parquet"),
    "ctr_calibration.json": os.path.join(ROOT, "results/E0_sentence/eval/ctr_calibration.json"),
    "X3B_READINGS.md": os.path.join(ROOT, "analysis/e0_sentence/x3b/X3B_READINGS.md"),
}
expect = {"X3B_results.json": "118a49bf7fd15d14d69fdcf0f2b8adadacd573330961d3c8353ed5fec2c1d767",
          "X3B_READINGS.md": "fef078b8058a0bbb19f25697014130f60b5f8d83a8e93d6b8b61900ecb2dd34e"}
hashes = {k: sha(v) for k, v in inputs.items()}
for k, v in expect.items():
    assert hashes[k] == v, (k, hashes[k])

res = json.load(open(inputs["X3B_results.json"]))
lev = pd.read_parquet(inputs["X3B_levels.parquet"])
slo = pd.read_parquet(inputs["X3B_slopes.parquet"])
pairs = pd.read_parquet(inputs["X3B_pairs.parquet"])
cal = json.load(open(inputs["ctr_calibration.json"]))
out = {"role": "x3b-review", "status": "EXPLORATORY, descriptive, post hoc", "inputs_sha256": hashes}

# ---- 1. binding labels re-derived from per-pair records (readings sec. 5) ----
def depth_label(r):
    return ("mostly" if r >= 2 else "partly" if r >= 1.25 else "not") + (" (lowers)" if r < 0.8 else "")

lab = {}
for s, rows in res["per_pair"].items():
    df = pd.DataFrame(rows)
    assert len(df) == 15, s
    medR = float(np.median(df["R_L10_1024"]))
    medb = float(np.median(df["b10"]))
    nstab = int((df["sum"].abs() <= 0.05).sum())
    nrise = int((df["L100_10000"] > df["L100_10000_mid"]).sum())
    nd = df[df["vit"] != "dinov2_b"]
    lab[s] = {"median_R_L10_1024": medR, "depth": depth_label(medR),
              "median_b10_last": medb, "median_b10_mid": float(np.median(df["b10_mid"])),
              "fall": "still falls" if medb < -0.1 else "no longer falls clearly",
              "n_b10_le_-0.1": int((df["b10"] <= -0.1).sum()),
              "n_stable": nstab, "stable": nstab >= 12,
              "max_abs_sum_last": float(df["sum"].abs().max()),
              "n_rise_L100_10000": nrise,
              "median_R_L100_10000_non_dinov2": float(np.median(nd["R_L100_10000"])),
              "n_rise_non_dinov2": int((nd["L100_10000"] > nd["L100_10000_mid"]).sum()),
              "range_R_L10_1024": [float(df["R_L10_1024"].min()), float(df["R_L10_1024"].max())],
              "median_b100_last": float(np.median(df["b100"])), "median_b100_mid": float(np.median(df["b100_mid"])),
              "median_L10_1024_last": float(np.median(df["L10_1024"])),
              "median_L10_1024_mid": float(np.median(df["L10_1024_mid"]))}
b = lab["B"]
lab["Q_b_label"] = ("not specific" if b["n_rise_L100_10000"] >= 12 else
                    "some ViTs only" if b["n_rise_L100_10000"] >= 6 else "largely specific")
lab["Q_b_size"] = ("large" if b["median_R_L100_10000_non_dinov2"] >= 2 else
                   "moderate" if b["median_R_L100_10000_non_dinov2"] >= 1.25 else "little")
lab["matches_report"] = {
    "Qa_depth": res["Q_a"]["depth_label"] == "the level is mostly a depth effect" and lab["A1"]["depth"] == "mostly",
    "Qa_fall": res["Q_a"]["fall_label"] == "still falls with m" and lab["A1"]["fall"] == "still falls",
    "Qa_stab": res["Q_a"]["stability_label"] == "stays close to stable at fixed k/m" and lab["A1"]["stable"],
    "Qb_dir": res["Q_b"]["label"] == "the rise with depth is not specific to DINOv2-B" and lab["Q_b_label"] == "not specific",
    "Qb_size": res["Q_b"]["size_label_descriptive"] == "a large rise, as for DINOv2-B" and lab["Q_b_size"] == "large",
    "Qb_stab": res["Q_b"]["stability_label"] == "stays close to stable at fixed k/m" and lab["B"]["stable"],
}
out["labels_rederived"] = lab

# ---- 2. matched k/m without a fit ----
MS = np.array([256, 512, 1024, 2048, 4096, 8192, 16384], float)
LCOL = [f"L_m{int(m)}" for m in MS]

def depth_of(pair):
    return "last" if pair.split("__")[0].endswith(".r10") else "mid"


def interp_lnL(row, m):
    y = np.log(row[LCOL].to_numpy(float))
    return float(np.interp(np.log(m), np.log(MS), y))


recs = []
for (s, p), g in lev.groupby(["set", "pair"]):
    g = g.set_index("k")
    rec = {"set": s, "pair": p}
    for kl, kh in [(10, 100), (1, 10)]:
        d = []
        for mp in (256, 512, 1024):
            lo = np.log(g.loc[kl, f"L_m{mp}"])
            hi = interp_lnL(g.loc[kh], 10 * mp)
            d.append(hi - lo)
        rec[f"d_{kl}_{kh}"] = d
        rec[f"maxabs_{kl}_{kh}"] = float(np.max(np.abs(d)))
    # how far above chance the matched cells lie (k=100 at m=2560, k=10 at m=256)
    rec["obs_over_chance_k100_m2560"] = float(np.exp(interp_lnL(g.loc[100], 2560)) / (100 / 2559))
    rec["obs_over_chance_k10_m256"] = float(g.loc[10, "L_m256"] / (10 / 255))
    recs.append(rec)
mk = pd.DataFrame(recs)
mk["depth"] = mk["pair"].map(depth_of)
# map reference sets to the set they serve
mk["family"] = mk["set"].str.replace("ref", "", regex=False)
summ = {}
for (fam, dep), g in mk.groupby(["family", "depth"]):
    summ[f"{fam}|{dep}"] = {
        "n": int(len(g)),
        "median_maxabs_10_100": float(g["maxabs_10_100"].median()),
        "max_maxabs_10_100": float(g["maxabs_10_100"].max()),
        "median_signed_10_100": float(np.median(np.concatenate(g["d_10_100"].to_list()))),
        "median_maxabs_1_10": float(g["maxabs_1_10"].median()),
        "max_maxabs_1_10": float(g["maxabs_1_10"].max()),
        "median_obs_over_chance_k100_m2560": float(g["obs_over_chance_k100_m2560"].median()),
        "min_obs_over_chance_k100_m2560": float(g["obs_over_chance_k100_m2560"].min()),
    }
last = mk[mk["depth"] == "last"]
midd = mk[mk["depth"] == "mid"]
summ["ALL|last"] = {"n": int(len(last)), "max_maxabs_10_100": float(last["maxabs_10_100"].max()),
                    "median_maxabs_10_100": float(last["maxabs_10_100"].median()),
                    "n_maxabs_10_100_le_0.10": int((last["maxabs_10_100"] <= 0.10).sum())}
summ["ALL|mid"] = {"n": int(len(midd)), "max_maxabs_10_100": float(midd["maxabs_10_100"].max()),
                   "median_maxabs_10_100": float(midd["maxabs_10_100"].median()),
                   "n_maxabs_10_100_le_0.10": int((midd["maxabs_10_100"] <= 0.10).sum())}
out["matched_km"] = summ
# level scale for comparison: ln of the depth ratio at L(10,1024), last-block pairs
out["matched_km_vs_depth_effect"] = {
    s: float(np.median(np.log(pd.DataFrame(res["per_pair"][s])["R_L10_1024"]))) for s in res["per_pair"]}

# ---- 3. level-slope lines ----
a_clip, b_clip = 0.1357059076472474, 0.22805384255107658  # ANALYSIS_PLAN_E0.md 7.3 (frozen)
a_ctr, b_ctr = cal["lines"]["k10|m1024"]["a"], cal["lines"]["k10|m1024"]["b"]
lines = {}
for s, rows in res["per_pair"].items():
    df = pd.DataFrame(rows)
    pre = df["pre"].iloc[0]
    if pre == "clip":
        a, bb, name = a_clip, b_clip, "clip line (plan 7.3)"
    elif pre == "ctr":
        a, bb, name = a_ctr, b_ctr, "ctr line (ctr_calibration.json k10|m1024)"
    else:
        continue
    for dep, Lc, bc in (("last", "L10_1024", "b10"), ("mid", "L10_1024_mid", "b10_mid")):
        pred = np.minimum(a + bb * np.log(df[Lc].to_numpy(float)), 0.0)
        resid = df[bc].to_numpy(float) - pred
        lines[f"{s}|{dep}"] = {"line": name, "median_pred": float(np.median(pred)),
                               "median_obs": float(np.median(df[bc])), "median_resid": float(np.median(resid)),
                               "range_resid": [float(resid.min()), float(resid.max())],
                               "n_resid_below_-0.1": int((resid < -0.1).sum())}
out["level_slope_lines"] = lines

# ---- 4. slope classes at k = 100 ----
sl = slo[slo["k"] == 100].copy()
sl["depth"] = sl["pair"].map(depth_of)
sl["family"] = sl["set"].str.replace("ref", "", regex=False)
cls = {}
for (fam, dep), g in sl.groupby(["family", "depth"]):
    cls[f"{fam}|{dep}"] = {"n": int(len(g)), "n_flat_k100": int((g["slope_obs"].abs() < 0.1).sum()),
                           "median_b100": float(g["slope_obs"].median()),
                           "frozen_class_counts": g["slope_class"].value_counts().to_dict()}
sl10 = slo[slo["k"] == 10].copy()
sl10["depth"] = sl10["pair"].map(depth_of)
cls["k10_all_last_falling"] = int(((sl10["depth"] == "last") & (sl10["slope_class"] == "falling")).sum())
cls["k10_all_last_n"] = int((sl10["depth"] == "last").sum())
cls["k10_all_mid_falling"] = int(((sl10["depth"] == "mid") & (sl10["slope_class"] == "falling")).sum())
out["slope_classes"] = cls

os.makedirs(os.path.dirname(OUT), exist_ok=False)
with open(OUT, "x") as f:
    json.dump(out, f, indent=1, sort_keys=False)
os.chmod(OUT, 0o444)
print(OUT, sha(OUT))
print(json.dumps({k: out[k] for k in ("matched_km", "level_slope_lines", "slope_classes")}, indent=1))
print(json.dumps(out["labels_rederived"]["matches_report"]))
for s in ("A1", "B"):
    print(s, json.dumps(out["labels_rederived"][s]))
print(json.dumps(out["matched_km_vs_depth_effect"]))
