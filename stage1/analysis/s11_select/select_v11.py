"""s11-select: apply the pre-stated v1.1 selection rule (PREDICTOR_CHOICE_v1.1.md
Part A; LEAD_DECISIONS_v1.1 D4.3-D4.7) to the plant and S_sel runs.

    python select_v11.py --plants results/S1_1_select/plants --ssel results/S1_1_select/ssel \
        --out results/S1_1_select

Rule
  (a) plant score of a variant = max over the 5 selection classes (P-Gauss,
      P-Aniso, P-Conf, P-Rough, P-Smooth+noise) of the class-mean absolute
      error |pred - obs| over eligible cells (replicate-mean obs in
      [0.05, 0.99], m != m_f), predictions in mode 'd2';
  (b) S_sel eligibility (D4.4): median over eligible S_sel cells (draw-mean
      obs > 2 k/(m-1), m != 512) of |log(pred/obs)|, pred = mean of the two
      directions (pred <= 0 or missing -> +inf); ineligible if > log 1.5;
  (c) null gate (D4.5): class MAE strictly below N0 on P-Conf and P-Aniso and
      strictly below N1 on P-Rough and P-GaussSat (same cells, same mode);
  (d) among variants passing (b) and (c): minimal score; variants within 0.002
      of the minimum are tied and resolved by predict_v11.PREFERENCE;
  (e) if none passes (b) and (c): 'no eligible estimator' (D4.7).
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sel_common as S  # noqa: E402
from hrhv2 import predict_v11 as v  # noqa: E402

TIE = 0.002
NULLS = ("N0", "N0H1", "N1", "N1|mu")
GATE_SENS = {"N0 -> N0H1 (v1.0 N0)": {"N0": "N0H1"}, "N1 -> N1|mu (v1.0 mu-law N1)": {"N1": "N1|mu"}}


def _num(x):
    if x is None:
        return float("nan")
    if isinstance(x, str):
        return float(x)
    return float(x)


# ----------------------------------------------------------------------------
# plants
# ----------------------------------------------------------------------------

def load_plants(d):
    units = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(f).startswith("RUN"):
            continue
        units.append(json.load(open(f)))
    return units


def plant_cells(units):
    groups = {}
    for u in units:
        inf = u["info"]
        plant = inf["plant"]
        c = inf.get("c") if plant == "P-Gauss" else None
        for cell in u["cells"]:
            key = (plant, int(inf["p"]), c, int(cell["m"]), int(cell["k"]))
            groups.setdefault(key, []).append(cell)
    out = []
    for (plant, p, c, m, k), cs in sorted(groups.items(), key=lambda t: (t[0][0], t[0][1], t[0][2] or 0, t[0][3], t[0][4])):
        rec = {"plant": plant, "p": p, "c": c, "m": m, "k": k, "n_reps": len(cs), "is_fit_m": bool(cs[0]["is_fit_m"]),
               "chance": cs[0]["chance"], "n_anchors": float(np.mean([x["n_anchors"] for x in cs]))}
        keys = set().union(*[set(x) for x in cs])
        for kk in keys:
            if kk in ("m", "k", "is_fit_m", "chance", "n_anchors", "N1_defined") or kk.endswith("_s") or kk == "seconds":
                continue
            vals = [_num(x.get(kk)) for x in cs]
            if all(isinstance(x.get(kk), (int, float)) or x.get(kk) is None for x in cs):
                rec[kk] = float(np.mean(vals)) if all(np.isfinite(vals)) else float("nan")
        rec["obs_se"] = float(np.sqrt(np.mean([x["obs_sd"] ** 2 / x["n_anchors"] for x in cs]) / len(cs)))
        out.append(rec)
    return out


def eligible(c):
    return (not c["is_fit_m"]) and S.ELIGIBLE[0] <= c["obs"] <= S.ELIGIBLE[1]


def class_of(c):
    return c["plant"]


def class_mae(cells, pred):
    by = {}
    for c in cells:
        if not eligible(c):
            continue
        by.setdefault(class_of(c), []).append(abs(c.get(pred, float("nan")) - c["obs"]))
    return {cl: {"mae": float(np.mean(e)) if np.all(np.isfinite(e)) else float("nan"),
                 "max": float(np.max(e)) if np.all(np.isfinite(e)) else float("nan"), "n_cells": len(e)}
            for cl, e in by.items()}


def score(mae):
    vals = [mae[cl]["mae"] for cl in S.SELECTION_CLASSES if cl in mae]
    if len(vals) < len(S.SELECTION_CLASSES) or not np.all(np.isfinite(vals)):
        return float("nan"), None
    i = int(np.argmax(vals))
    return float(vals[i]), S.SELECTION_CLASSES[i]


def gate(mae_by_pred, variant, mode_suffix="", null_map=None):
    null_map = null_map or {}
    out = {}
    for cl, null in S.GATE_CLASSES.items():
        nn = null_map.get(null, null)
        if mode_suffix and nn in ("N1",):
            nn = nn + mode_suffix
        a = mae_by_pred[variant + mode_suffix].get(cl, {}).get("mae", float("nan"))
        b = mae_by_pred[nn].get(cl, {}).get("mae", float("nan"))
        out[cl] = {"null": nn, "mae_variant": a, "mae_null": b, "pass": bool(np.isfinite(a) and np.isfinite(b) and a < b)}
    return {"pass": all(x["pass"] for x in out.values()), "by_class": out}


# ----------------------------------------------------------------------------
# S_sel
# ----------------------------------------------------------------------------

def load_ssel(d):
    units = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(f).startswith("RUN"):
            continue
        units.append(json.load(open(f)))
    return units


def ssel_cells(units, preds):
    """Draw-mean cells per (pair, direction, m, k); then the direction mean."""
    g = {}
    for u in units:
        for c in u["cells"]:
            key = (u["cls"], u["pair"], u["direction"], int(c["m"]), int(c["k"]))
            g.setdefault(key, []).append(c)
    perdir = {}
    for (cls, pair, dr, m, k), cs in g.items():
        rec = {"cls": cls, "pair": pair, "direction": dr, "m": m, "k": k, "n_draws": len(cs),
               "is_fit_m": bool(cs[0]["is_fit_m"]), "chance": cs[0]["chance"],
               "obs": float(np.mean([x["obs"] for x in cs])),
               "n_anchors": float(np.mean([x["n_anchors"] for x in cs]))}
        for p in preds:
            vals = [_num(x.get(p)) for x in cs]
            rec[p] = float(np.mean(vals)) if np.all(np.isfinite(vals)) else float("nan")
        perdir[(cls, pair, dr, m, k)] = rec
    both = []
    for (cls, pair, dr, m, k), rec in perdir.items():
        if dr != "AB":
            continue
        rb = perdir.get((cls, pair, "BA", m, k))
        r = {kk: rec[kk] for kk in ("cls", "pair", "m", "k", "is_fit_m", "chance", "obs", "n_anchors", "n_draws")}
        assert rb is not None and abs(rb["obs"] - rec["obs"]) < 1e-12
        for p in preds:
            r[p] = 0.5 * (rec[p] + rb[p])
            r[p + "@AB"] = rec[p]
            r[p + "@BA"] = rb[p]
        both.append(r)
    return sorted(both, key=lambda r: (r["cls"], r["pair"], r["m"], r["k"]))


def ssel_elig_cells(cells):
    return [c for c in cells if (not c["is_fit_m"]) and c["obs"] > 2.0 * c["chance"]]


def abslog(pred, obs):
    if not np.isfinite(pred) or pred <= 0:
        return float("inf")
    return abs(math.log(pred / obs))


def ssel_median(cells, pred, subset=None):
    cc = ssel_elig_cells(cells)
    if subset is not None:
        cc = [c for c in cc if subset(c)]
    if not cc:
        return float("nan"), 0
    return float(np.median([abslog(c[pred], c["obs"]) for c in cc])), len(cc)


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------

def f3(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "pass" if x else "FAIL"
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", required=True)
    ap.add_argument("--ssel", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    PU = load_plants(args.plants)
    cells = plant_cells(PU)
    preds_all = list(v.VARIANTS) + [x + "|mu" for x in v.VARIANTS] + list(NULLS) + ["oracle", "oracle|mu"]
    mae = {p: class_mae(cells, p) for p in preds_all}
    scores = {p: score(mae[p]) for p in preds_all}
    SU = load_ssel(args.ssel)
    sp = list(v.VARIANTS) + [x + "|mu" for x in v.VARIANTS] + ["N0", "N0H1", "N1", "N1|mu"]
    sc = ssel_cells(SU, sp)
    ssel = {}
    for p in sp:
        med, n = ssel_median(sc, p)
        ssel[p] = {"median_abslog": med, "n_cells": n, "eligible": bool(med <= S.SSEL_ELIG_LOG),
                   "by_class": {cl: ssel_median(sc, p, lambda c, cl=cl: c["cls"] == cl)[0] for cl in S.SSEL_PAIRS},
                   "AB_only": ssel_median(sc, p + "@AB")[0], "BA_only": ssel_median(sc, p + "@BA")[0]}
    per_pair = {}
    for c in ssel_elig_cells(sc):
        per_pair.setdefault(c["pair"], []).append(c)
    pair_tab = {pair: {p: float(np.median([abslog(c[p], c["obs"]) for c in cs])) for p in sp} | {"n": len(cs), "cls": cs[0]["cls"]}
                for pair, cs in per_pair.items()}

    # ---- primary rule ---------------------------------------------------------
    res = {}
    for var in v.VARIANTS:
        g = gate(mae, var)
        res[var] = {"score": scores[var][0], "worst_class": scores[var][1], "ssel": ssel[var],
                    "eligible_ssel": ssel[var]["eligible"], "gate": g,
                    "passes": bool(ssel[var]["eligible"] and g["pass"])}
    cand = [x for x in v.VARIANTS if res[x]["passes"] and np.isfinite(res[x]["score"])]
    if cand:
        best = min(res[x]["score"] for x in cand)
        tied = [x for x in cand if res[x]["score"] <= best + TIE]
        chosen = sorted(tied, key=lambda x: v.PREFERENCE.index(x))[0]
        decision = {"outcome": "selected", "chosen": chosen, "candidates": cand, "tied": tied, "best_score": best}
    else:
        decision = {"outcome": "no eligible estimator", "chosen": None, "candidates": []}
    # unconstrained ranking (reported only)
    ranking = sorted([x for x in v.VARIANTS if np.isfinite(res[x]["score"])], key=lambda x: res[x]["score"])

    # ---- sensitivities (reported, never governing) ---------------------------
    sens = {}
    for name, nm in GATE_SENS.items():
        sens[name] = {var: gate(mae, var, null_map=nm)["pass"] for var in v.VARIANTS}
    sens["all predictors in mode 'mu' (SPEC literal)"] = {
        var: {"score": scores[var + "|mu"][0], "gate": gate(mae, var, mode_suffix="|mu")["pass"],
              "ssel_eligible": ssel[var + "|mu"]["eligible"]} for var in v.VARIANTS}
    # D2 diagnostic: exact vs mu at k = 10 (rep 0 only)
    d2diag = []
    for u in PU:
        for c in u["cells"]:
            for var in ("i-LSlog", "iii-LSlog-EB", "ii-IsoXF", "oracle"):
                if f"{var}|exact" in c and c.get(var) is not None:
                    d2diag.append({"unit": u["unit"], "m": c["m"], "k": c["k"], "var": var,
                                   "mu": _num(c[var]), "exact": _num(c[f"{var}|exact"]), "obs": c["obs"]})
    d2_summary = {}
    for var in ("i-LSlog", "iii-LSlog-EB", "ii-IsoXF", "oracle"):
        dd = [abs(x["mu"] - x["exact"]) for x in d2diag if x["var"] == var and np.isfinite(x["mu"]) and np.isfinite(x["exact"])]
        if dd:
            d2_summary[var] = {"n": len(dd), "max_abs_mu_minus_exact": float(np.max(dd)), "mean": float(np.mean(dd))}

    # runtimes
    rt = {"plants_units": len(PU), "plants_unit_s_median": float(np.median([u["total_s"] for u in PU])),
          "plants_unit_s_max": float(np.max([u["total_s"] for u in PU])),
          "plants_cpu_h": float(np.sum([u["total_s"] for u in PU]) / 3600),
          "ssel_units": len(SU), "ssel_cpu_h": float(np.sum([u["total_s"] for u in SU]) / 3600)}
    by_m = {}
    for u in PU:
        for kk, val in u["diag"]["timing"].items():
            by_m.setdefault(kk, []).append(val)
    rt["plants_timing_median_by_stage_s"] = {kk: float(np.median(x)) for kk, x in by_m.items()}

    summary = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "decision": decision, "per_variant": res, "ranking_unconstrained": ranking,
               "class_mae": mae, "scores": {p: {"score": s[0], "worst": s[1]} for p, s in scores.items()},
               "ssel": ssel, "ssel_per_pair": pair_tab, "sensitivities": sens, "d2_diag": d2_summary,
               "runtimes": rt, "n_plant_cells": len(cells),
               "n_plant_cells_eligible": int(sum(eligible(c) for c in cells)),
               "inputs": {"plants": os.path.abspath(args.plants), "ssel": os.path.abspath(args.ssel),
                          "select_v11_sha256": S.file_sha256(os.path.abspath(__file__))}}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "selection_summary.json"), "w") as f:
        json.dump(S.jsonable(summary), f, indent=1)
    import pandas as pd
    pd.DataFrame(cells).to_parquet(os.path.join(args.out, "plant_cells.parquet"), index=False)
    pd.DataFrame(sc).to_parquet(os.path.join(args.out, "ssel_cells.parquet"), index=False)
    pd.DataFrame(d2diag).to_parquet(os.path.join(args.out, "d2_exact_k10_diag.parquet"), index=False)

    # ---- markdown -------------------------------------------------------------
    L = ["# S1.1 estimator re-selection (PREREG v1.1, D4): results", "",
         f"Generated {summary['created_utc']} by `analysis/s11_select/select_v11.py`. Rule: "
         "`hrhv2/PREDICTOR_CHOICE_v1.1.md` Part A. No COCO file was opened.", "",
         f"## Decision: **{decision['outcome']}**" + (f" -> `{decision['chosen']}`" if decision["chosen"] else ""), ""]
    L += ["## Per-variant outcome", "",
          "| variant | plant score (max class MAE) | worst class | S_sel median abs log ratio (<= 0.405) | gate P-Conf (vs N0) | gate P-Aniso (vs N0) | gate P-Rough (vs N1) | gate P-GaussSat (vs N1) | passes (b)+(c) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for var in v.VARIANTS:
        r = res[var]
        gb = r["gate"]["by_class"]
        L.append(f"| `{var}` | {f3(r['score'])} | {r['worst_class']} | {f3(r['ssel']['median_abslog'])} ({'ok' if r['eligible_ssel'] else 'INELIGIBLE'}) | "
                 + " | ".join(f"{f3(gb[cl]['mae_variant'])} vs {f3(gb[cl]['mae_null'])} {'ok' if gb[cl]['pass'] else 'FAIL'}"
                              for cl in ("P-Conf", "P-Aniso", "P-Rough", "P-GaussSat"))
                 + f" | {'yes' if r['passes'] else 'no'} |")
    L += ["", "## Plant table: class-mean absolute error |pred - obs| (eligible cells: replicate-mean obs in [0.05, 0.99], m != 1024)", "",
          "| predictor | " + " | ".join(S.SELECTION_CLASSES) + " | P-GaussSat | max over 5 selection classes |",
          "|---|" + "---|" * (len(S.SELECTION_CLASSES) + 2)]
    for p in preds_all:
        row = [f3(mae[p].get(cl, {}).get("mae")) for cl in S.SELECTION_CLASSES + ("P-GaussSat",)]
        L.append(f"| `{p}` | " + " | ".join(row) + f" | {f3(scores[p][0])} |")
    ncl = {cl: mae["N0"].get(cl, {}).get("n_cells") for cl in S.SELECTION_CLASSES + ("P-GaussSat",)}
    L.append(f"\nEligible cells per class: {ncl}.")
    L += ["", "## S_sel table: median |log(pred/obs)| over eligible cells (draw-mean obs > 2x chance, m in {256, 1023}; fit m = 512)", "",
          "| predictor | pooled (12 pairs) | n cells | ViT-ViT | LM-LM | LM-ViT | A->B only | B->A only |", "|---|---|---|---|---|---|---|---|"]
    for p in sp:
        s = ssel[p]
        L.append(f"| `{p}` | {f3(s['median_abslog'])} | {s['n_cells']} | {f3(s['by_class']['vit-vit'])} | {f3(s['by_class']['lm-lm'])} | "
                 f"{f3(s['by_class']['lm-vit'])} | {f3(s['AB_only'])} | {f3(s['BA_only'])} |")
    L += ["", "Per pair (median |log(pred/obs)|, d2 mode):", "",
          "| pair | class | n cells | " + " | ".join(v.VARIANTS) + " | N0 | N1 |", "|---|---|---|" + "---|" * (len(v.VARIANTS) + 2)]
    for pair, t in sorted(pair_tab.items(), key=lambda x: (x[1]["cls"], x[0])):
        L.append(f"| {pair} | {t['cls']} | {t['n']} | " + " | ".join(f3(t[p]) for p in v.VARIANTS) + f" | {f3(t['N0'])} | {f3(t['N1'])} |")
    L += ["", "S_sel cells (direction mean, draw mean), m != 512:", "",
          "| pair | m | k | obs | chance | " + " | ".join(v.VARIANTS) + " | N0 | N1 |", "|---|---|---|---|---|" + "---|" * (len(v.VARIANTS) + 2)]
    for c in sc:
        if c["is_fit_m"]:
            continue
        L.append(f"| {c['pair']} | {c['m']} | {c['k']} | {f3(c['obs'])} | {f3(c['chance'])} | "
                 + " | ".join(f3(c[p]) for p in v.VARIANTS) + f" | {f3(c['N0'])} | {f3(c['N1'])} |")
    L += ["", "## Sensitivities (reported; the primary rule governs)", ""]
    for name, d in sens.items():
        L.append(f"- {name}: " + json.dumps(S.jsonable(d)))
    L += ["", "## D2 diagnostic: mu-law vs exact discrepancy at k = 10 (rep 0, m in {256, 512, 2048})", "",
          json.dumps(S.jsonable(d2_summary)), "", "## Runtimes", "", json.dumps(S.jsonable(rt), indent=1)]
    with open(os.path.join(args.out, "selection_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(json.dumps(S.jsonable({"decision": decision, "scores": {x: res[x]["score"] for x in v.VARIANTS},
                                 "ssel": {x: ssel[x]["median_abslog"] for x in v.VARIANTS},
                                 "gate": {x: res[x]["gate"]["pass"] for x in v.VARIANTS}}), indent=1))


if __name__ == "__main__":
    main()
