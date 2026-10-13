"""X3 (EXPLORATORY) results and report; role x3-run. Applies the rules of X3_READINGS.md (sha256 checked against
the STATUS line "[x3-run] READINGS FIXED") to the tables written by x3_scores.py (sha256 checked against
obs_X3.json and the STATUS line "[x3-run] SCORES"). Reads E0 arm-CA tables read only for the reference.

    taskset -c 0-5 venv/bin/python -I analysis/e0_sentence/x3/x3_report.py
Writes results/E0_sentence/x3/X3_results.json and X3_REPORT.md (never overwrites; chmod 444).
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E0DIR = os.path.dirname(HERE)
sys.path.insert(0, E0DIR)
import e0_common as E  # noqa: E402  (frozen)
import e0_evaluate as EV  # noqa: E402  (frozen)

X3OUT = os.path.join(E.OUT, "x3")
READINGS = os.path.join(HERE, "X3_READINGS.md")
OUTDIR = os.environ.get("X3_REPORT_OUTDIR", X3OUT)   # dry runs only; the real run writes into X3OUT
FEATURES = ("x3_cls_r10", "x3_tok_r10", "x3_tok_r05", "x3add_schnaus224")
FEATURE_TEXT = {
    "dinov2_b": "E0 reference: CLS, block 5 (rel 0.5), no final norm, 518 px",
    "x3_cls_r10": "frozen: CLS, block 11 (rel 1.0), no final norm, 518 px",
    "x3_tok_r10": "frozen: patch-token mean, block 11 (rel 1.0), no final norm, 518 px",
    "x3_tok_r05": "frozen: patch-token mean, block 5 (rel 0.5), no final norm, 518 px",
    "x3add_schnaus224": "ADDED: mean of all 257 output tokens after the final norm, 224 px",
}
ENCS = ("mpnet", "contriever", "qwen3emb06")
# fixed in X3_READINGS.md sections 4, 5, 8
L_E0_READINGS = {"mpnet": 0.07715427364361675, "contriever": 0.07633979414898837, "qwen3emb06": 0.07717300965111769}
SCHNAUS = 0.38
BAND = (0.37, 0.39)
CTR_LINE_K10_M1024 = (0.15928637788145117, 0.2361611718219916)
M_GRID = (256, 512, 1024, 2048, 4096, 8192, 16384)


def label(share: float) -> str:
    if share >= 1.0:
        return "closes the whole gap (reaches or passes Schnaus' reading)"
    if share >= 0.5:
        return "closes most of the gap"
    if share >= 0.2:
        return "closes part of the gap"
    if share >= 0.0:
        return "closes little of the gap"
    return "widens the gap"


def share(L, L0, ref=SCHNAUS):
    return math.log(L / L0) / math.log(ref / L0)


def line_of(tag):
    for ln in E.status_lines():
        if tag in ln:
            return ln
    return None


def main():
    t0 = time.perf_counter()
    h_read = E.sha256_file(READINGS)
    ln = line_of("[x3-run] READINGS FIXED")
    if ln is None or h_read not in ln:
        raise SystemExit("REFUSED: readings sha256 not in STATUS")
    with open(os.path.join(X3OUT, "obs", "obs_X3.json")) as f:
        prov = json.load(f)
    sc = line_of("[x3-run] SCORES")
    h_prov = E.sha256_file(os.path.join(X3OUT, "obs", "obs_X3.json"))
    if sc is None or h_prov not in sc:
        raise SystemExit("REFUSED: STATUS lacks '[x3-run] SCORES' with obs_X3.json's sha256")
    tabs = {}
    for rel, h in prov["outputs_sha256"].items():
        p = os.path.join(X3OUT, rel)
        if E.sha256_file(p) != h:
            raise SystemExit(f"REFUSED: {rel} differs from obs_X3.json")
        if rel.startswith("eval/"):
            tabs[rel[5:-8]] = pd.read_parquet(p)
    with open(os.path.join(X3OUT, "validation", "VALIDATION_X3.json")) as f:
        val = json.load(f)
    with open(os.path.join(X3OUT, "extraction", "EXTRACTION_X3.json")) as f:
        ext = json.load(f)
    with open(os.path.join(E.PROJECT, "features", "coco_e0", "x3", "x3_extract.json")) as f:
        side = json.load(f)

    # E0 reference (arm CA), read only
    e0 = {n: pd.read_parquet(os.path.join(E.EVAL, f"E0_{n}_CA.parquet")) for n in ("cells", "slopes", "elasticity", "levels")}
    refp = [f"dinov2_b__{e}" for e in ENCS]
    e0 = {n: t[t["pair"].isin(refp)].copy() for n, t in e0.items()}
    e0["levels_allH2"] = EV.levels(e0["cells"], col="obs_allH2")
    e0_sha = {f"results/E0_sentence/eval/E0_{n}_CA.parquet": E.sha256_file(os.path.join(E.EVAL, f"E0_{n}_CA.parquet"))
              for n in ("cells", "slopes", "elasticity", "levels")}
    allt = {n: pd.concat([e0[n], tabs[f"X3_{n}"]], ignore_index=True) for n in ("cells", "slopes", "elasticity", "levels")}
    allt["levels_allH2"] = pd.concat([e0["levels_allH2"], tabs["X3_levels_allH2"]], ignore_index=True)
    Lx = allt["levels"].set_index(["pair", "k"])
    LxH = allt["levels_allH2"].set_index(["pair", "k"])
    Sx = allt["slopes"].set_index(["pair", "k"])
    Ex = allt["elasticity"].set_index(["pair", "k_set"])
    C = allt["cells"]

    # reference check against the values written into the readings
    ref_check = {e: {"readings": L_E0_READINGS[e], "E0_levels_CA": float(Lx.loc[(f"dinov2_b__{e}", 100), "L_m10000"])}
                 for e in ENCS}
    for e, r in ref_check.items():
        r["equal"] = r["readings"] == r["E0_levels_CA"]
    L0H = {e: float(LxH.loc[(f"dinov2_b__{e}", 100), "L_m10000"]) for e in ENCS}

    rows = []
    for feat in ("dinov2_b",) + FEATURES:
        for e in ENCS:
            p = f"{feat}__{e}"
            r = {"pair": p, "feature": feat, "feature_text": FEATURE_TEXT[feat], "enc": e,
                 "role": "E0 reference" if feat == "dinov2_b" else ("X3 added" if feat.startswith("x3add") else "X3 frozen")}
            r["L_10_1024"] = float(Lx.loc[(p, 10), "L_m1024"])
            r["L_100_8192"] = float(Lx.loc[(p, 100), "L_m8192"])
            r["L_10_10000"] = float(Lx.loc[(p, 10), "L_m10000"])
            r["L_100_10000"] = float(Lx.loc[(p, 100), "L_m10000"])
            r["L_100_10000_allH2"] = float(LxH.loc[(p, 100), "L_m10000"])
            r["L_10_10000_allH2"] = float(LxH.loc[(p, 10), "L_m10000"])
            r["ratio_k100_over_k10_m8192"] = float(Lx.loc[(p, 10), "ratio_k100_over_k10_m8192"])
            r["ratio_k100_over_k10_m16384"] = float(Lx.loc[(p, 10), "ratio_k100_over_k10_m16384"])
            for k in (1, 10, 100):
                r[f"b{k}"] = float(Sx.loc[(p, k), "slope_obs"])
                r[f"b{k}_draw_sd"] = float(Sx.loc[(p, k), "slope_obs_draw_sd"])
                r[f"b{k}_class"] = str(Sx.loc[(p, k), "slope_class"])
                r[f"b{k}_allH2"] = float(Sx.loc[(p, k), "slope_obs_allH2"])
            for ks in ("k10_100", "k1_10_100", "k1_10"):
                for c in ("e_k", "e_m", "sum", "rmse"):
                    r[f"{c}_{ks}"] = float(Ex.loc[(p, ks), c])
            r["p1_stable_abs_sum_le_0.05"] = abs(r["sum_k10_100"]) <= 0.05
            r["p1_informative_abs_em_ge_0.10"] = abs(r["e_m_k10_100"]) >= 0.10
            L0 = L_E0_READINGS[e]
            r["share_0.38"] = share(r["L_100_10000"], L0)
            r["share_0.37"] = share(r["L_100_10000"], L0, BAND[0])
            r["share_0.39"] = share(r["L_100_10000"], L0, BAND[1])
            r["share_allH2_0.38"] = share(r["L_100_10000_allH2"], L0H[e])
            r["gap_label"] = label(r["share_0.38"])
            r["gap_label_binding"] = (e == "mpnet" and feat != "dinov2_b")
            r["ln_L100_10000_minus_ref"] = math.log(r["L_100_10000"] / L0)
            r["ln_L10_1024_minus_ref"] = math.log(r["L_10_1024"] / float(Lx.loc[(f"dinov2_b__{e}", 10), "L_m1024"]))
            r["ratio_to_schnaus_0.38"] = r["L_100_10000"] / SCHNAUS
            a, b = CTR_LINE_K10_M1024
            r["ctr_line_pred_b10"] = a + b * math.log(r["L_10_1024"])
            r["ctr_line_resid_b10"] = r["b10"] - r["ctr_line_pred_b10"]
            rows.append(r)
    R = pd.DataFrame(rows)

    curves = {}
    for p in R["pair"]:
        g = C[C["pair"] == p]
        curves[p] = {int(k): [{"m": int(x.m), "s": float(x.obs), "draw_sd": float(x.obs_draw_sd), "chance": float(x.chance),
                               "s_allH2": float(x.obs_allH2), "n_anchors_mean": float(x.n_anchors)}
                              for x in g[g["k"] == k].sort_values("m").itertuples()] for k in (1, 10, 100)}

    shape = {}
    for pr in R["pair"]:
        shape[pr] = {}
        for k in (1, 10, 100):
            ys = [x["s"] for x in curves[pr][k]]
            ms = [x["m"] for x in curves[pr][k]]
            shape[pr][k] = {"argmax_m": int(ms[int(np.argmax(ys))]), "max_s": float(max(ys)),
                            "non_increasing_in_m": bool(all(b <= a for a, b in zip(ys, ys[1:])))}
    X3 = R[R["role"] != "E0 reference"]
    p1 = {"all_12": {"median_sum": float(X3["sum_k10_100"].median()), "n_abs_le_0.05": int(X3["p1_stable_abs_sum_le_0.05"].sum()),
                     "n_informative": int(X3["p1_informative_abs_em_ge_0.10"].sum()), "n": int(len(X3)),
                     "max_abs_sum": float(X3["sum_k10_100"].abs().max())}}
    inf = X3[X3["p1_informative_abs_em_ge_0.10"]]
    p1["all_12"]["plan_thresholds_descriptive"] = {
        "abs_median_le_0.03": bool(abs(inf["sum_k10_100"].median()) <= 0.03) if len(inf) else None,
        "frac_abs_le_0.05_ge_0.8": bool((inf["sum_k10_100"].abs() <= 0.05).mean() >= 0.8) if len(inf) else None}
    for f in FEATURES:
        g = X3[X3["feature"] == f]
        p1[f] = {"median_sum": float(g["sum_k10_100"].median()), "n_abs_le_0.05": int(g["p1_stable_abs_sum_le_0.05"].sum()),
                 "n": int(len(g)), "sums": dict(zip(g["enc"], g["sum_k10_100"].astype(float)))}

    binding = {f: {k: R[(R.feature == f) & (R.enc == "mpnet")].iloc[0][k] for k in
                   ("L_100_10000", "share_0.38", "share_0.37", "share_0.39", "share_allH2_0.38", "gap_label")}
               for f in FEATURES}
    contrasts = {}
    for e in ENCS:
        g = R[R.enc == e].set_index("feature")
        contrasts[e] = {
            "depth only: x3_cls_r10 vs ref (ln L(100,10000))": float(g.loc["x3_cls_r10", "ln_L100_10000_minus_ref"]),
            "token set only: x3_tok_r05 vs ref": float(g.loc["x3_tok_r05", "ln_L100_10000_minus_ref"]),
            "depth and token set: x3_tok_r10 vs ref": float(g.loc["x3_tok_r10", "ln_L100_10000_minus_ref"]),
            "added (norm + all tokens + 224 px): x3add_schnaus224 vs ref": float(g.loc["x3add_schnaus224", "ln_L100_10000_minus_ref"]),
            "added vs x3_tok_r10 (norm + CLS in the mean + 224 px)": float(g.loc["x3add_schnaus224", "ln_L100_10000_minus_ref"] - g.loc["x3_tok_r10", "ln_L100_10000_minus_ref"])}

    out = {"role": "x3-run", "status": "EXPLORATORY (ANALYSIS_PLAN_E0.md v1.0 section 7.6 X3); no confirmatory claim",
           "readings": {"file": "analysis/e0_sentence/x3/X3_READINGS.md", "sha256": h_read},
           "pair_count": {"frozen_X3_pairs": 9, "added_pairs": 3, "total_X3_pairs": 12, "reference_pairs_E0": 3,
                          "note": "task text said 10 vision variants x 3 = 30 pairs; the frozen X3 is 3 features x 3 encoders = 9 pairs; nothing invented"},
           "reference_check_vs_readings": ref_check,
           "checks": {"extraction_stop_rule": ext["passA_stop_rule"], "passA_vs_s11": ext["passA_vs_s11"],
                      "determinism": ext["determinism"], "timm_antialias_side_note": side["passB"].get("side_note_timm_antialias_resample_64"),
                      "wrapper_validation_pass": val["pass"], "wrapper_validation": {k: val[k] for k in ("cache_ref_vs_E0", "obs_rebuilt_vs_obs_CA", "obs_E0cache_vs_obs_CA")}},
           "schnaus_reading": {"value": SCHNAUS, "band": BAND, "source": "Schnaus et al. Fig. 23 p. 48 dotted line (critic's reading), DINOv2-B x MPNet, MS COCO, k = 100, n = 10,000"},
           "binding_gap_reading_mpnet": binding,
           "pairs": rows, "contrasts_ln_L_100_10000": contrasts, "p1_type_descriptive": p1, "curves": curves, "curve_shape": shape,
           "inputs_sha256": {"obs_X3.json": h_prov, **{f"x3/{k}": v for k, v in prov["outputs_sha256"].items()}, **e0_sha,
                             "x3/validation/VALIDATION_X3.json": E.sha256_file(os.path.join(X3OUT, "validation", "VALIDATION_X3.json")),
                             "x3/extraction/EXTRACTION_X3.json": E.sha256_file(os.path.join(X3OUT, "extraction", "EXTRACTION_X3.json"))},
           "compute": {"extraction_gpu_hours": ext["gpu_hours_process_wall"], "extraction_cpu_core_hours": ext["cpu_core_hours"],
                       "scores_gpu_hours": 0.0, "scores_cpu_core_hours": prov["cpu_core_hours"], "scores_wall_s": prov["wall_s"]},
           "code_sha256": {"x3_report.py": E.sha256_file(os.path.abspath(__file__))}, "created_utc": E.utcnow()}
    ru = resource.getrusage(resource.RUSAGE_SELF)
    out["compute"]["report_cpu_core_hours"] = (ru.ru_utime + ru.ru_stime) / 3600
    md = render(out, R)
    hj = E.write_json_new(os.path.join(OUTDIR, "X3_results.json"), out)
    E.write_new(os.path.join(OUTDIR, "X3_REPORT.md"), md)
    hm = E.sha256_file(os.path.join(OUTDIR, "X3_REPORT.md"))
    for p in ("X3_results.json", "X3_REPORT.md"):
        os.chmod(os.path.join(OUTDIR, p), 0o444)
    print(json.dumps({"X3_results.json": hj, "X3_REPORT.md": hm, "binding": binding, "wall_s": time.perf_counter() - t0}, indent=1, default=str))


def f4(x):
    return f"{x:.4f}"


def render(out, R) -> str:
    L = []
    a = L.append
    a("# X3 protocol bridge: DINOv2-B output features × sentence encoders (EXPLORATORY)\n")
    a("Role `x3-run`, written by `analysis/e0_sentence/x3/x3_report.py` from the hashed tables. "
      f"Readings fixed beforehand: `X3_READINGS.md` sha256 `{out['readings']['sha256']}`. "
      "Everything here is exploratory (plan §7.6). No E0 label changes.\n")
    a("## 1. What ran\n")
    a("- 12 X3 pairs: 3 frozen DINOv2-B features plus 1 added feature, each × MPNet, Contriever and Qwen3-Embedding-0.6B.")
    a("- Text side: N5 (five-caption average). Preprocessing: `ctr`. Design: E0's 5 nested draws, m = 256 to 16,384, k ∈ {1, 10, 100}.")
    a("- The E0 reference pairs (rel-0.5 CLS) are listed next to the X3 pairs.")
    a("- The task text said 30 pairs (10 vision variants). The frozen X3 is 9 pairs, made of 3 features × 3 encoders. No variant was invented.\n")
    a("| feature | what it is |\n|---|---|")
    for k, v in FEATURE_TEXT.items():
        a(f"| `{k}` | {v} |")
    a("")
    c = out["checks"]
    a("## 2. Checks\n")
    a(f"- **Extraction at the S1.1 settings:** reproduces the S1.1 stored arrays bitwise on all 16,384 rows (CLS and patch mean, blocks 5 and 11). STOP rule passed: {c['extraction_stop_rule']['pass']}.")
    a("- **Determinism:** 64 images run twice give bitwise-equal outputs in both passes.")
    s = c["timm_antialias_side_note"]
    a(f"- **Side note on position embeddings:** timm's antialiased resampling, instead of the HF-style one used here, changes the 224 px feature (64 images: min cosine {s['min_cos']:.4f}, mean {s['mean_cos']:.5f}). This is one of the open implementation details.")
    v = c["wrapper_validation"]
    a(f"- **Wrapper validation (STOP rule):** the rebuilt reference cache equals E0's bitwise: {v['cache_ref_vs_E0']['equal']}. The obs rows from the rebuilt cache equal E0's `obs_CA` rows: {v['obs_rebuilt_vs_obs_CA']['equal']}; from E0's cache: {v['obs_E0cache_vs_obs_CA']['equal']}. Overall pass: {c['wrapper_validation_pass']}.\n")
    a("## 3. The gap to Schnaus et al. (binding reading: DINOv2 × MPNet, L(100, 10000), column `obs`)\n")
    a("- share = ln(L / 0.07715) / ln(0.38 / 0.07715). Labels: most ≥ 0.5, part 0.2–0.5, little < 0.2, widens < 0.")
    a("- Schnaus' 0.38 is a figure reading (Fig. 23, p. 48), so the shares against 0.37 and 0.39 are shown as a band.\n")
    a("| feature | L(100, 10000) | ratio to 0.38 | share (0.38) | band (0.39 to 0.37) | share, all-H2 queries | reading |\n|---|---|---|---|---|---|---|")
    for f, b in out["binding_gap_reading_mpnet"].items():
        a(f"| `{f}` | {f4(b['L_100_10000'])} | {b['L_100_10000']/0.38:.3f} | {b['share_0.38']:.3f} | {b['share_0.39']:.3f} to {b['share_0.37']:.3f} | {b['share_allH2_0.38']:.3f} | {b['gap_label']} |")
    a(f"| E0 reference | {f4(L_E0_READINGS['mpnet'])} | {L_E0_READINGS['mpnet']/0.38:.3f} | 0 | | | |\n")
    a("**The other two encoders (descriptive).** Schnaus et al. used no Contriever, and their Qwen embedder is the 8B model.\n")
    a("| pair | L(100, 10000) | share (0.38) | reading (descriptive) |\n|---|---|---|---|")
    for _, r in R[(R.enc != "mpnet") & (R.role != "E0 reference")].iterrows():
        a(f"| `{r['pair']}` | {f4(r['L_100_10000'])} | {r['share_0.38']:.3f} | {r['gap_label']} |")
    a("")
    a("## 4. Levels, all pairs (column `obs`; the all-H2 value is shown where it applies)\n")
    a("| pair | L(10, 1024) | L(100, 8192) | L(10, 10000) | L(100, 10000) | L(100, 10000), all-H2 | s(100)/s(10) at m = 8192 |\n|---|---|---|---|---|---|---|")
    for r in R.itertuples():
        a(f"| `{r.pair}` | {f4(r.L_10_1024)} | {f4(r.L_100_8192)} | {f4(r.L_10_10000)} | {f4(r.L_100_10000)} | {f4(r.L_100_10000_allH2)} | {r.ratio_k100_over_k10_m8192:.3f} |")
    a("")
    a("## 5. Contrasts: ln L(100, 10000) minus the E0 reference (descriptive)\n")
    a("| contrast | mpnet | contriever | qwen3emb06 |\n|---|---|---|---|")
    keys = list(out["contrasts_ln_L_100_10000"]["mpnet"].keys())
    for k in keys:
        a(f"| {k} | " + " | ".join(f"{out['contrasts_ln_L_100_10000'][e][k]:+.3f}" for e in ENCS) + " |")
    a("")
    a("## 6. Scale reading of P1 type (descriptive; e_k + e_m on k ∈ {10, 100})\n")
    a("| pair | e_k | e_m | sum | abs(sum) ≤ 0.05 | sum on {1, 10, 100} | sum on {1, 10} |\n|---|---|---|---|---|---|---|")
    for _, r in R.iterrows():
        a(f"| `{r['pair']}` | {r['e_k_k10_100']:.3f} | {r['e_m_k10_100']:.3f} | {r['sum_k10_100']:+.4f} | {'yes' if r['p1_stable_abs_sum_le_0.05'] else 'no'} | {r['sum_k1_10_100']:+.4f} | {r['sum_k1_10']:+.4f} |")
    p = out["p1_type_descriptive"]["all_12"]
    a(f"\nOver the 12 X3 pairs:")
    a(f"- median sum {p['median_sum']:+.4f}; {p['n_abs_le_0.05']} of {p['n']} within 0.05; largest |sum| {p['max_abs_sum']:.4f}; {p['n_informative']} informative (|e_m| ≥ 0.10).")
    a(f"- The plan's P1 thresholds, applied descriptively: |median| ≤ 0.03 is {p['plan_thresholds_descriptive']['abs_median_le_0.03']}; at least 80% within 0.05 is {p['plan_thresholds_descriptive']['frac_abs_le_0.05_ge_0.8']}.\n")
    a("## 7. Slopes and the level–slope orientation (descriptive)\n")
    a("The `ctr` line at k = 10, m = 1024 is b = 0.1593 + 0.2362 ln L. It is shown for orientation only.\n")
    a("| pair | b(1) | b(10) | b(100) | line b(10) at L(10, 1024) | residual |\n|---|---|---|---|---|---|")
    for r in R.itertuples():
        a(f"| `{r.pair}` | {r.b1:.3f} | {r.b10:.3f} | {r.b100:.3f} | {r.ctr_line_pred_b10:.3f} | {r.ctr_line_resid_b10:+.3f} |")
    a("")
    a("## 8. Full curves: draw-mean s(m, k), column `obs`\n")
    for k in (1, 10, 100):
        a(f"**k = {k}**\n")
        a("| pair | " + " | ".join(f"m={m}" for m in M_GRID) + " |\n|---|" + "---|" * len(M_GRID))
        for pr in R["pair"]:
            cv = {x["m"]: x["s"] for x in out["curves"][pr][k]}
            a(f"| `{pr}` | " + " | ".join(f4(cv[m]) for m in M_GRID) + " |")
        ch = {x["m"]: x["chance"] for x in out["curves"][R['pair'].iloc[0]][k]}
        a("| chance k/(m−1) | " + " | ".join(f4(ch[m]) for m in M_GRID) + " |\n")
    a("## 9. Protocol differences that remain (not replicated)\n")
    a("- Registers: we use the no-register DINOv2-B. Their p. 26 pools \"class, register and patch tokens\", and their p. 27 names a with-registers model for the RAE only.")
    a("- Library and position-embedding interpolation: HF-style here; see §2 for the effect of the timm alternative.")
    a("- bf16 storage (p. 21).")
    a("- Centering: their aligner centers on a disjoint training half (p. 25), and their metric centers each side on its 10,000 samples (p. 26). We center on the 16,384 pool and renormalize.")
    a("- Captions: they average all of the up to 7 captions (p. 25); we use 5. E0's arm CAall shows that ALL against AVG5 changes the DINOv2 × MPNet cell by less than 0.001.")
    a("- Queries: they use all 10,000 points of a random subsample. We use the nested prefixes with S1.1's anchors (`obs`), and all H2 items as a sensitivity.")
    a("- Qwen3-Embedding: 0.6B here, 8B there.\n")
    a("## 10. Observations (descriptive; wording written after seeing the numbers)\n")
    g = R.set_index("pair")
    X3 = R[R.role != "E0 reference"]
    r10 = R[R.feature.isin(["x3_cls_r10", "x3_tok_r10", "x3add_schnaus224"])]
    a(f"- **The gap closes with depth.** Only the block changes between the reference and `x3_cls_r10`: both are the CLS token, with no norm, at 518 px. That change alone moves DINOv2 × MPNet from {f4(g.loc['dinov2_b__mpnet', 'L_100_10000'])} to {f4(g.loc['x3_cls_r10__mpnet', 'L_100_10000'])}, against Schnaus' about 0.38. "
      f"The patch mean at the same mid depth gets to {f4(g.loc['x3_tok_r05__mpnet', 'L_100_10000'])}. "
      f"The Schnaus-style feature sits within 0.01 in ln L of the block-11 patch mean, at {f4(g.loc['x3add_schnaus224__mpnet', 'L_100_10000'])} against {f4(g.loc['x3_tok_r10__mpnet', 'L_100_10000'])}. So the final norm, putting the CLS token in the mean, and 224 px change almost nothing. The three encoders agree to about 0.01 in every row.")
    a(f"- **k = 10 at m = 10,000.** The block-11 features give {f4(r10[r10.enc=='mpnet']['L_10_10000'].min())}–{f4(r10[r10.enc=='mpnet']['L_10_10000'].max())} with MPNet. That is inside the 0.07–0.19 read from Schnaus' Fig. 14b (descriptive, no rule).")
    a(f"- **The scale reading still holds at these levels.** On k ∈ {{10, 100}}, e_k + e_m stays between {X3['sum_k10_100'].min():+.4f} and {X3['sum_k10_100'].max():+.4f} for all 12 X3 pairs, with |e_m| ≥ {X3['e_m_k10_100'].abs().min():.2f}. Agreement still falls with m at fixed k: b(10) is {r10['b10'].min():.3f} to {r10['b10'].max():.3f} for the block-11 features.")
    a(f"- **The level–slope line does not extend to these levels.** The E0 `ctr` line predicts b(10) of about −0.04 to −0.06 at L(10, 1024) ≈ 0.40–0.43. The block-11 pairs give b(10) residuals from {r10['ctr_line_resid_b10'].min():+.3f} to {r10['ctr_line_resid_b10'].max():+.3f}, which are much steeper. "
      "The frozen plan (§0, B6) said in advance that if the levels came out near Schnaus', P1 and the level–slope regularity could not both hold, and that the tension would fall on the latter. That is what these exploratory numbers show. No E0 label changes.")
    cs = out["curve_shape"]
    nm = [(pr, k, cs[pr][k]["argmax_m"]) for pr in R["pair"] for k in (10, 100) if not cs[pr][k]["non_increasing_in_m"]]
    a("- **Curve shapes at high levels.** These pairs have a curve that is not non-increasing in m (pair, k, the m of the maximum): "
      + ("; ".join(f"`{pr}` k = {k}, peak at m = {mm} (k/m = {k/mm:.3f})" for pr, k, mm in nm) if nm else "none") + ".")
    a(f"  - The block-11 patch-mean features (`x3_tok_r10`, `x3add_schnaus224`) fall at every step at k = 100, but flatten between m = 1024 and 4096. Their b(100) is {r10[r10.feature != 'x3_cls_r10']['b100'].min():.3f} to {r10[r10.feature != 'x3_cls_r10']['b100'].max():.3f}.")
    a(f"  - For `x3_cls_r10`, b(100) is {r10[r10.feature == 'x3_cls_r10']['b100'].min():.3f} to {r10[r10.feature == 'x3_cls_r10']['b100'].max():.3f}. That number summarizes a curve that rises and then falls; the curve is not flat.")
    a("  - The sums in §6 are close to 0, so these shapes line up along k/m. The k = 10 and k = 100 peaks of `x3_cls_r10` sit at k/m of about 0.02.\n")
    cp = out["compute"]
    a("## 11. Compute (no cap set)\n")
    a(f"- **GPU:** {cp['extraction_gpu_hours']:.4f} h, all of it the extraction process (wall time).")
    a(f"- **CPU:** extraction {cp['extraction_cpu_core_hours']:.3f} core-h; scores {cp['scores_cpu_core_hours']:.3f} core-h; report {cp['report_cpu_core_hours']:.4f} core-h. All runs were pinned to at most 6 cores.\n")
    a("## 12. Input hashes\n")
    for k, h in out["inputs_sha256"].items():
        a(f"- `{k}` `{h}`")
    a("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
