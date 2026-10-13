"""X3b (EXPLORATORY) report; role x3b-run. Applies analysis/e0_sentence/x3b/X3B_READINGS.md (section 5) unchanged
to the evaluation tables written by x3b_scores.py. Reads only; writes X3B_results.json and X3B_REPORT.md.

    venv/bin/python -I analysis/e0_sentence/x3b/x3b_report.py                      # real run (needs STATUS SCORES)
    venv/bin/python -I analysis/e0_sentence/x3b/x3b_report.py --out-dir /tmp/...   # dry run, same code path
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import e0_common as E  # noqa: E402  (frozen; helpers only)

OUT = os.path.join(E.OUT, "x3b")
EVALDIR = os.path.join(OUT, "eval")
READINGS = os.path.join(HERE, "X3B_READINGS.md")
PROV = os.path.join(OUT, "obs", "obs_X3B.json")
VAL = os.path.join(OUT, "validation", "VALIDATION_X3B.json")
VITS = ("dinov2_b", "clip_b", "mae_b", "augreg_b", "siglip_b")
CELLS = ("L10_1024", "L100_8192", "L100_10000")
REF = {"S1.1 clip VL L(10,1024)": 0.04295406802450704, "E0 CA L(10,1024)": 0.07397754861172179,
       "S1.1 clip vision L(10,1024)": 0.18851616701579738}
LAST_MID = {"A1": ("A1", "A1ref"), "A3": ("A3", "A3ref"), "A4": ("A4", "A4ref"), "B": ("B", "Bref"),
            "A2_r025": ("A2_r025", "A2_r025"), "A2_r075": ("A2_r075", "A2_r075"), "A2_r10": ("A2_r10", "A2_r10")}
SET_TEXT = {"A1": "Q-a binding: clip, ViT block 11 x LM tokens (LM rel 0.5)",
            "A2_r025": "Q-a, LM rel 0.25 (clip), descriptive", "A2_r075": "Q-a, LM rel 0.75 (clip), descriptive",
            "A2_r10": "Q-a, LM rel 1.0 (clip), descriptive", "A3": "Q-a sensitivity: l2", "A4": "Q-a sensitivity: ctr",
            "B": "Q-b binding: ctr, ViT block 11 x sentence encoders (N5)"}


def per_pair(T, pairs):
    Lv, S, el = T["levels"], T["slopes"], T["elasticity"]
    rows = []
    for r in pairs.itertuples(index=False):
        L = Lv[(Lv["set"] == r.set) & (Lv["pair"] == r.pair)].set_index("k")
        s = S[(S["set"] == r.set) & (S["pair"] == r.pair)].set_index("k")["slope_obs"]
        e = el[(el["set"] == r.set) & (el["pair"] == r.pair)].set_index("k_set")
        LH = T["levels_allH2"]
        LH = LH[(LH["set"] == r.set) & (LH["pair"] == r.pair)].set_index("k")
        rows.append({"set": r.set, "pair": r.pair, "pre": r.pre, "vit": r.vit, "rel_v": r.rel_v, "text": r.text, "rel_t": r.rel_t,
                     "L10_1024": float(L.loc[10, "L_m1024"]), "L100_8192": float(L.loc[100, "L_m8192"]),
                     "L100_10000": float(L.loc[100, "L_m10000"]), "L10_10000": float(L.loc[10, "L_m10000"]),
                     "allH2_L10_1024": float(LH.loc[10, "L_m1024"]), "allH2_L100_10000": float(LH.loc[100, "L_m10000"]),
                     "b1": float(s.loc[1]), "b10": float(s.loc[10]), "b100": float(s.loc[100]),
                     "e_k": float(e.loc["k10_100", "e_k"]), "e_m": float(e.loc["k10_100", "e_m"]),
                     "sum": float(e.loc["k10_100", "sum"]), "rmse": float(e.loc["k10_100", "rmse"]),
                     "sum_k1_10_100": float(e.loc["k1_10_100", "sum"]), "sum_k1_10": float(e.loc["k1_10", "sum"])})
    return pd.DataFrame(rows)


def depth_label(med):
    lab = ("the level is mostly a depth effect" if med >= 2 else
           "the level is partly a depth effect" if med >= 1.25 else "the level is not a depth effect")
    if med < 0.8:
        lab += " (the last block lowers the level)"
    return lab


def fall_label(med_b10):
    return "still falls with m" if med_b10 < -0.1 else "no longer falls clearly with m"


def stab(g):
    n = len(g)
    n_in = int((g["sum"].abs() <= 0.05).sum())
    med = float(g["sum"].median())
    return {"n": n, "n_abs_sum_le_0.05": n_in, "frac": n_in / n, "median_sum": med,
            "range_sum": [float(g["sum"].min()), float(g["sum"].max())],
            "n_informative_abs_em_ge_0.10": int((g["e_m"].abs() >= 0.10).sum()),
            "label": "stays close to stable at fixed k/m" if n_in >= 0.8 * n else "not stable at fixed k/m",
            "E0_P1_thresholds_descriptive": bool(abs(med) <= 0.03 and n_in >= 0.8 * n),
            "median_sum_k1_10_100": float(g["sum_k1_10_100"].median()),
            "n_abs_sum_k1_10_100_le_0.05": int((g["sum_k1_10_100"].abs() <= 0.05).sum())}


def last_mid(PP, s):
    a, b = LAST_MID[s]
    if s.startswith("A2"):
        last = PP[(PP["set"] == a) & (PP["rel_v"] == "1.0")]
        mid = PP[(PP["set"] == b) & (PP["rel_v"] == "0.5")]
    else:
        last, mid = PP[PP["set"] == a], PP[PP["set"] == b]
    m = last.merge(mid, on=["vit", "text"], suffixes=("", "_mid"))
    assert len(m) == 15, (s, len(m))
    for c in CELLS:
        m[f"R_{c}"] = m[c] / m[f"{c}_mid"]
    return m


def summarize(m, s):
    out = {"set": s, "what": SET_TEXT[s], "n": int(len(m))}
    for c in CELLS:
        out[f"median_last_{c}"] = float(m[c].median())
        out[f"median_mid_{c}"] = float(m[f"{c}_mid"].median())
        out[f"median_R_{c}"] = float(m[f"R_{c}"].median())
        out[f"range_R_{c}"] = [float(m[f"R_{c}"].min()), float(m[f"R_{c}"].max())]
        out[f"n_last_gt_mid_{c}"] = int((m[c] > m[f"{c}_mid"]).sum())
    out["per_vit_median_R"] = {v: {c: float(m[m["vit"] == v][f"R_{c}"].median()) for c in CELLS} for v in VITS}
    out["per_vit_median_last"] = {v: {c: float(m[m["vit"] == v][c].median()) for c in CELLS} for v in VITS}
    out["median_b"] = {f"b{k}": float(m[f"b{k}"].median()) for k in (1, 10, 100)}
    out["median_b_mid"] = {f"b{k}": float(m[f"b{k}_mid"].median()) for k in (1, 10, 100)}
    out["n_b10_le_-0.1"] = int((m["b10"] <= -0.1).sum())
    out["fall_label"] = fall_label(out["median_b"]["b10"])
    out["stability"] = stab(m)
    out["stability_mid"] = stab(m.rename(columns={"sum": "x", "sum_mid": "sum", "e_m": "y", "e_m_mid": "e_m",
                                                  "sum_k1_10_100": "z", "sum_k1_10_100_mid": "sum_k1_10_100"}))
    out["depth_label_at_L10_1024"] = depth_label(out["median_R_L10_1024"])
    out["median_allH2_R_L10_1024"] = float((m["allH2_L10_1024"] / m["allH2_L10_1024_mid"]).median())
    out["median_allH2_R_L100_10000"] = float((m["allH2_L100_10000"] / m["allH2_L100_10000_mid"]).median())
    return out


def qb_reading(m):
    n = int((m["L100_10000"] > m["L100_10000_mid"]).sum())
    lab = ("the rise with depth is not specific to DINOv2-B" if n >= 12 else
           "the rise with depth holds for some ViTs only" if n >= 6 else "the rise with depth is largely specific to DINOv2-B")
    o = m[m["vit"] != "dinov2_b"]
    medR = float(o["R_L100_10000"].median())
    size = ("a large rise, as for DINOv2-B" if medR >= 2 else "a moderate rise" if medR >= 1.25 else "little or no rise")
    return {"cell": "L(100, 10000)", "n_last_gt_mid": n, "share": n / 15, "label": lab,
            "n_last_gt_mid_non_dinov2": int((o["L100_10000"] > o["L100_10000_mid"]).sum()), "n_non_dinov2": int(len(o)),
            "median_R_non_dinov2": medR, "range_R_non_dinov2": [float(o["R_L100_10000"].min()), float(o["R_L100_10000"].max())],
            "size_label_descriptive": size,
            "descriptive_other_cells": {c: {"n_last_gt_mid": int((m[c] > m[f"{c}_mid"]).sum()),
                                            "n_last_gt_mid_non_dinov2": int((o[c] > o[f"{c}_mid"]).sum()),
                                            "median_R_non_dinov2": float(o[f"R_{c}"].median())} for c in ("L10_1024", "L100_8192")}}


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def report_md(R, M):
    L = ["# X3b report (role `x3b-run`): last-block vision features, CPU only. EXPLORATORY.", "",
         f"Readings `X3B_READINGS.md` sha256 `{R['readings_sha256']}` (STATUS {R['readings_status_utc']}); "
         f"validation `VALIDATION_X3B.json` {R['validation']['sha256'][:16]}... pass = {R['validation']['pass']}; "
         f"scores `obs_X3B.json` {R['provenance_sha256'][:16]}....", "",
         "## Binding readings", "",
         "| question | set | statistic | value | reading |", "|---|---|---|---|---|"]
    a1, b = R["sets"]["A1"], R["sets"]["B"]
    L.append(f"| Q-a level | A1 | median R = L_last/L_mid at L(10,1024) | {fmt(a1['median_R_L10_1024'], 3)} | {a1['depth_label_at_L10_1024']} |")
    L.append(f"| Q-a fall with m | A1 | median b(10) | {fmt(a1['median_b']['b10'], 3)} | {a1['fall_label']} |")
    L.append(f"| Q-a stability | A1 | abs(e_k+e_m) <= 0.05 | {a1['stability']['n_abs_sum_le_0.05']}/15 | {a1['stability']['label']} |")
    q = R["Q_b"]
    L.append(f"| Q-b direction | B | L_last > L_mid at L(100,10000) | {q['n_last_gt_mid']}/15 | {q['label']} |")
    L.append(f"| Q-b stability | B | abs(e_k+e_m) <= 0.05 | {b['stability']['n_abs_sum_le_0.05']}/15 | {b['stability']['label']} |")
    L += ["", "Descriptive companions: Q-b size (12 non-DINOv2 pairs) median R at L(100,10000) "
          f"{fmt(q['median_R_non_dinov2'], 3)} ({q['size_label_descriptive']}); Q-b fall with m: median b(10) "
          f"{fmt(b['median_b']['b10'], 3)} ({b['fall_label']}).", "",
          "## Per set (medians over 15 pairs)", "",
          "| set | what | L_last(10,1024) | L_mid(10,1024) | R(10,1024) | R(100,8192) | R(100,10000) | n last>mid (10,1024 / 100,10000) | b(10) last / mid | stable last / mid | depth label | fall label |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in ("A1", "A2_r025", "A2_r075", "A2_r10", "A3", "A4", "B"):
        x = R["sets"][s]
        L.append(f"| {s} | {x['what']} | {fmt(x['median_last_L10_1024'])} | {fmt(x['median_mid_L10_1024'])} | "
                 f"{fmt(x['median_R_L10_1024'], 3)} | {fmt(x['median_R_L100_8192'], 3)} | {fmt(x['median_R_L100_10000'], 3)} | "
                 f"{x['n_last_gt_mid_L10_1024']} / {x['n_last_gt_mid_L100_10000']} | {fmt(x['median_b']['b10'], 3)} / "
                 f"{fmt(x['median_b_mid']['b10'], 3)} | {x['stability']['n_abs_sum_le_0.05']}/15 / "
                 f"{x['stability_mid']['n_abs_sum_le_0.05']}/15 | {x['depth_label_at_L10_1024']} | {x['fall_label']} |")
    L += ["", "## Per ViT, median R over the 3 text models", "",
          "| set | cell | " + " | ".join(VITS) + " |", "|---|---|" + "---|" * len(VITS)]
    for s in ("A1", "A3", "A4", "B", "A2_r10"):
        for c in ("L10_1024", "L100_10000"):
            L.append(f"| {s} | {c} | " + " | ".join(fmt(R['sets'][s]['per_vit_median_R'][v][c], 2) for v in VITS) + " |")
    for s in ("A1", "B"):
        m = M[s]
        L += ["", f"## Set {s}, per pair", "",
              "| vit | text | L(10,1024) last / mid | R | L(100,10000) last / mid | R | b(1) | b(10) | b(100) | b(10) mid | e_k | e_m | sum | sum mid |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in m.sort_values(["vit", "text"]).itertuples(index=False):
            L.append(f"| {r.vit} | {r.text} | {fmt(r.L10_1024)} / {fmt(r.L10_1024_mid)} | {fmt(r.R_L10_1024, 2)} | "
                     f"{fmt(r.L100_10000)} / {fmt(r.L100_10000_mid)} | {fmt(r.R_L100_10000, 2)} | {fmt(r.b1, 3)} | {fmt(r.b10, 3)} | "
                     f"{fmt(r.b100, 3)} | {fmt(r.b10_mid, 3)} | {fmt(r.e_k, 3)} | {fmt(r.e_m, 3)} | {fmt(r.sum, 4)} | {fmt(r.sum_mid, 4)} |")
    L += ["", "## Cross-reading (descriptive): same ViT features under ctr, LM tokens (A4) vs sentence encoders (B)", "",
          "| vit | A4 R(10,1024) | B R(10,1024) | A4 R(100,10000) | B R(100,10000) |", "|---|---|---|---|---|"]
    for v in VITS:
        a4, bb = R["sets"]["A4"]["per_vit_median_R"][v], R["sets"]["B"]["per_vit_median_R"][v]
        L.append(f"| {v} | {fmt(a4['L10_1024'], 2)} | {fmt(bb['L10_1024'], 2)} | {fmt(a4['L100_10000'], 2)} | {fmt(bb['L100_10000'], 2)} |")
    L += ["", "## Checks", "", f"- Validation (STOP rule): {json.dumps(R['validation']['checks'])}",
          f"- Reference consistency with frozen tables: {json.dumps(R['reference_consistency'])}",
          f"- Compute (scores): CPU {fmt(R['compute']['scores_cpu_core_hours'], 3)} core-h, wall {fmt(R['compute']['scores_wall_s'], 0)} s, "
          f"{R['compute']['workers']} workers x 1 BLAS thread, affinity {R['compute']['cpu_affinity']}; GPU 0 h.",
          "", "All values: column `obs` (S1.1 anchors), E0/X3 interpolation for m = 10,000. EXPLORATORY; no S1.1, E0 or X3 label changes.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or OUT
    hr = E.sha256_file(READINGS)
    hp = E.sha256_file(PROV)
    st = E.status_lines()
    rl = [ln for ln in st if "[x3b-run] READINGS FIXED" in ln and hr in ln]
    if not rl:
        raise SystemExit("REFUSED: no READINGS FIXED line with the readings' sha256")
    if args.out_dir is None and not any("[x3b-run] SCORES" in ln and hp in ln for ln in st):
        raise SystemExit("REFUSED: no '[x3b-run] SCORES' line with obs_X3B.json's sha256")
    with open(PROV) as f:
        prov = json.load(f)
    with open(VAL) as f:
        val = json.load(f)
    if not val["pass"] or E.sha256_file(VAL) != prov["validation_sha256"]:
        raise SystemExit("REFUSED: validation did not pass or its hash differs from the provenance")
    T = {}
    for n in ("levels", "slopes", "elasticity", "levels_allH2", "pairs"):
        p = os.path.join(EVALDIR, f"X3B_{n}.parquet")
        if E.sha256_file(p) != prov["outputs_sha256"][f"eval/X3B_{n}.parquet"]:
            raise SystemExit(f"REFUSED: {p} differs from the provenance")
        T[n] = pd.read_parquet(p)
    PP = per_pair(T, T["pairs"])
    R = {"role": "x3b-run", "status": "EXPLORATORY", "readings_sha256": hr, "readings_status_utc": rl[0][:20],
         "provenance_sha256": hp, "validation": {"sha256": prov["validation_sha256"], "pass": val["pass"],
                                                  "checks": {k: v["equal"] for k, v in val["checks"].items()}},
         "reference_consistency": prov["reference_consistency"], "references_mid_depth": REF,
         "compute": {"scores_cpu_core_hours": prov["cpu_core_hours"], "scores_wall_s": prov["wall_s"],
                     "workers": prov["workers"], "cpu_affinity": prov["cpu_affinity"], "gpu_hours": 0.0},
         "sets": {}, "per_pair": {}}
    M = {}
    for s in LAST_MID:
        m = last_mid(PP, s)
        M[s] = m
        R["sets"][s] = summarize(m, s)
        R["per_pair"][s] = m.to_dict("records")
    R["Q_a"] = {"binding_set": "A1", "depth_label": R["sets"]["A1"]["depth_label_at_L10_1024"],
                "fall_label": R["sets"]["A1"]["fall_label"], "stability_label": R["sets"]["A1"]["stability"]["label"],
                "median_last_L10_1024_vs_0.04295": [R["sets"]["A1"]["median_last_L10_1024"], REF["S1.1 clip VL L(10,1024)"]]}
    R["Q_b"] = qb_reading(M["B"])
    R["Q_b"]["stability_label"] = R["sets"]["B"]["stability"]["label"]
    R["Q_b"]["fall_label_descriptive"] = R["sets"]["B"]["fall_label"]
    R["code_sha256"] = {"x3b_report.py": E.sha256_file(os.path.abspath(__file__))}
    R["created_utc"] = E.utcnow()
    os.makedirs(out_dir, exist_ok=True)
    hj = E.write_json_new(os.path.join(out_dir, "X3B_results.json"), R)
    E.write_new(os.path.join(out_dir, "X3B_REPORT.md"), report_md(R, M))
    print(json.dumps({"X3B_results.json": hj, "X3B_REPORT.md": E.sha256_file(os.path.join(out_dir, "X3B_REPORT.md")),
                      "Q_a": R["Q_a"], "Q_b": {k: R["Q_b"][k] for k in ("n_last_gt_mid", "label", "median_R_non_dinov2",
                                                                           "size_label_descriptive", "stability_label")}},
                     indent=1))


if __name__ == "__main__":
    main()
