#!/usr/bin/env python
"""OpenLLaMA axis: P-O2b, the features-only flip forecast of the 34 new comparisons (plan section 9 B6 and
P-O2b; VERDICTS_SPEC.md section 7 "Section F"). New file, written at S0b by sub-role PB of oll-axis-extract.

GATED (plan F1(a)): the first action after argument parsing is guard.require_partb_frozen(out_root); before
it, nothing but STATUS.md is read. It exits 3 until the `[oll-axis-predict] Part B FROZEN` STATUS line exists
and the two Part B files hash to the values on it.

Inputs (read-only):
  results/S1_2_layermax_openllama/partB/partB_features_ol.json   boundaries kappa^lo/hi (s_cal), the uncensored
                                                                   boundaries, the h curve, the frozen 3b/7b values
  pair files of the implementation (layer-max = protocol full, biased CKA, both pre):
    E: results/S1_2_layermax_openllama/early/pairs/<pre>/<lm>__<vit>.json   (E summaries: s_obs, c_alpha, s_cal,
                                                                             null_mean, uncensored)
    I: results/S1_2_layermax_openllama/indep/<pre>/pairs/<lm>__<vit>.npz    (obs_cka_b, null_cka_b; layer-max
                                                                             summarised with hrhv2.calibrate.summarize,
                                                                             as the I aggregate does)
  Only open_llama_13b's observed T (and its c_alpha, for the realised offsets) and the 3b/7b reruns' s_cal and
  uncensored (observed orders) are used.

Rule (spec F): for P in {prh_clip+l2, l2}, X in {s_cal, unc}, a in {3b, 7b}, v in the 17 ViTs:
  dec = "below" if T_13b,v < lo_a,v; "above" if T_13b,v > hi_a,v; else "undecided"; for X = s_cal also
        "undecided" if the frozen s_cal of a is 0 [op].
  obs = "below" / "above" / "tie" from X_13b vs X_a (this implementation's reruns); ok = (dec == obs), null if
        undecided. Totals n_decided, x_correct, n_undecided, Clopper-Pearson 95% of x/n;
  verdict "fails" if x < n, else "not evaluable" if n < 10, else "holds". h curve (X = s_cal) from
  partB boundaries.s_cal_h.
  Extras (x|, not compared): the same forecast scored against the FROZEN 3b/7b values (plan B6: reported on both
  if the reruns differ from the frozen values), and a flag whether the reruns equal the frozen values (1e-12 rel).

Outputs (out_root = results/S1_2_layermax_openllama/forecast/<impl>/):
  verdicts_forecast.json  {"schema": "OL-VERDICTS v1", "impl": "E"|"I", "inputs": {path: sha256}, "v": {...}}
                          with every key of verdict_keys.required_keys("F")
  forecast_detail.json    realised offsets c_alpha - Bbar of the 17 new pairs, boundaries used, per-comparison table

RUN (step S8, after the main runs E and I and V-O5; CPU, 1 thread, seconds):
  cd ~/hrh_v2_stage1_20261008
  analysis/s12_layermax_openllama/launch_ol.sh forecast_E 1 -- venv/bin/python analysis/s12_layermax_openllama/forecast_ol.py --impl E
  analysis/s12_layermax_openllama/launch_ol.sh forecast_I 1 -- venv/bin/python analysis/s12_layermax_openllama/forecast_ol.py --impl I
  venv/bin/python analysis/s12_layermax_openllama/compare_verdicts.py results/S1_2_layermax_openllama/forecast/E/verdicts_forecast.json results/S1_2_layermax_openllama/forecast/I/verdicts_forecast.json --sections F

S0B TEST MODE (OL_S0B_TEST=1 before CODE FROZEN only): --test-root DIR replaces the OpenLLaMA pair roots
(DIR/early/pairs/<pre>/ and DIR/indep/<pre>/pairs/), --partb PATH the Part B JSON, --out DIR the output root
(default ~/oll_s0b_scratch/PB/forecast/<impl>); file identity checks (lm/vit/key fields) are skipped because the
stand-in files are relabelled known files. These options are refused outside test mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_openllama")
sys.path.insert(0, PROJ)
sys.path.insert(0, AXIS)

import guard  # noqa: E402

RES_OL = os.path.join(PROJ, "results", "S1_2_layermax_openllama")
OUT_ROOT = os.path.join(RES_OL, "forecast")
PARTB_JSON = os.path.join(RES_OL, "partB", "partB_features_ol.json")
PRES = ("prh_clip+l2", "l2")
XS = ("s_cal", "unc")
SIZES = ("3b", "7b", "13b")
HS = (0.0003, 0.0005, 0.001, 0.0015, 0.002)
OL_SLUGS = ("openlm-research__open_llama_3b", "openlm-research__open_llama_7b", "openlm-research__open_llama_13b")
# pair-file roots of the implementations (plan section 3 and F1 table; VERDICTS_SPEC section 1 "Sources")
ROOTS = {
    "E": {"ol": os.path.join(RES_OL, "early", "pairs", "{pre}"),
          "bloomz": os.path.join(PROJ, "results", "S1_2_layermax", "pairs", "{pre}"),
          "llama": os.path.join(PROJ, "results", "S1_2_layermax_llama", "early", "pairs", "{pre}")},
    "I": {"ol": os.path.join(RES_OL, "indep", "{pre}", "pairs"),
          "bloomz": os.path.join(PROJ, "results", "S1_2_layermax", "indep", "{pre}", "pairs"),
          "llama": os.path.join(PROJ, "results", "S1_2_layermax_llama", "indep", "{pre}", "pairs")},
}
EXT = {"E": ".json", "I": ".npz"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fmt(v: float) -> str:
    return repr(float(v))


def cp95(x: int, n: int):
    """VERDICTS_SPEC section 9 Clopper-Pearson 95%."""
    if n == 0:
        return None, None
    from scipy.stats import beta
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    return lo, hi


def pair_path(impl: str, group: str, pre: str, lm: str, vit: str, test_root: str | None = None) -> str:
    if group == "ol" and test_root is not None:
        base = os.path.join(test_root, "early", "pairs", pre) if impl == "E" else \
            os.path.join(test_root, "indep", pre, "pairs")
    else:
        base = ROOTS[impl][group].format(pre=pre)
    return os.path.join(base, f"{lm}__{vit}{EXT[impl]}")


def layermax(impl: str, path: str, pre: str, lm: str, vit: str, check_identity: bool) -> dict:
    """Layer-max (protocol full) summary for biased CKA: T, c_alpha, s_cal, null_mean, unc."""
    if impl == "E":
        with open(path) as f:
            js = json.load(f)
        if check_identity:
            assert js["lm"] == lm and js["vit"] == vit and js["pre"] == pre, path
        s = js["summaries"]["cka_biased"]["full"]
        return {"T": float(s["s_obs"]), "c_alpha": float(s["c_alpha"]), "s_cal": float(s["s_cal"]),
                "null_mean": float(s["null_mean"]), "unc": float(s["uncensored"])}
    import numpy as np
    from hrhv2.calibrate import summarize
    z = np.load(path, allow_pickle=False)
    if check_identity:
        assert str(z["key"]) == f"S1.2-layermax|{lm}|{vit}", (path, str(z["key"]))
    obs, null = z["obs_cka_b"], z["null_cka_b"]
    K = null.shape[0]
    T = float(obs.max())
    sm = summarize(T, null.reshape(K, -1).max(axis=1))
    return {"T": float(sm["s_obs"]), "c_alpha": float(sm["c_alpha"]), "s_cal": float(sm["s_cal"]),
            "null_mean": float(sm["null_mean"]), "unc": float(sm["uncensored"])}


def resolve(args, what: str) -> tuple[bool, str, str | None, str]:
    """Test mode, out_root, test_root, Part B JSON path. Only STATUS.md is read here (guard.test_mode)."""
    test = guard.test_mode()
    if not test and (args.test_root or args.partb or args.out):
        print("--test-root/--partb/--out are S0b test options (OL_S0B_TEST=1 before CODE FROZEN)", file=sys.stderr)
        sys.exit(2)
    if test:
        out_root = args.out or os.path.join(guard.SCRATCH, "PB", what, args.impl)
        return True, out_root, args.test_root, args.partb or os.path.join(guard.SCRATCH, "PB", "partB",
                                                                           "partB_features_ol.json")
    return False, os.path.join(RES_OL, what, args.impl), None, PARTB_JSON


def obs_order(x13: float, xa: float) -> str:
    return "below" if x13 < xa else ("above" if x13 > xa else "tie")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=("E", "I"), required=True)
    ap.add_argument("--test-root", default=None, help="S0b test mode only")
    ap.add_argument("--partb", default=None, help="S0b test mode only")
    ap.add_argument("--out", default=None, help="S0b test mode only")
    args = ap.parse_args()
    test, out_root, test_root, partb = resolve(args, "forecast")
    gate = guard.require_partb_frozen(out_root, script="forecast_ol.py")      # F1(a): FIRST data-related action
    # ---- nothing involving a pair file, score or Part B value is opened before this line ----
    import numpy as np  # noqa: F401  (after the gate)

    F = json.load(open(partb))
    ols = list(F["ol_lms"])
    vits = list(F["vits"])
    if not test:
        assert ols == list(OL_SLUGS), ols
        import verdict_keys as VK
        assert vits == list(VK.VITS), "ViT order differs from verdict_keys.VITS"
    inputs = {os.path.relpath(partb, PROJ): sha256_file(partb)}
    vals = {}
    for pre in PRES:
        for k, lm in zip(SIZES, ols):
            for v in vits:
                p = pair_path(args.impl, "ol", pre, lm, v, test_root)
                vals[(pre, k, v)] = layermax(args.impl, p, pre, lm, v, check_identity=not test)
                inputs[os.path.relpath(p, PROJ)] = sha256_file(p)
    V = {}
    detail = {"impl": args.impl, "gate": gate, "partB": os.path.relpath(partb, PROJ), "ol_lms": ols, "per_pre": {}}
    for pre in PRES:
        Q = F["per_pre"][pre]
        bnd = Q["boundaries"]
        known = Q["known"]
        known_unc = Q.get("known_uncensored")
        Bbar13 = Q["Bbar"][ols[2]]
        offs = {v: vals[(pre, "13b", v)]["c_alpha"] - Bbar13[v] for v in vits}
        rerun_vs_frozen = {}
        for a in ("3b", "7b"):
            for fld, kf in (("T", "T"), ("c_alpha", "c_alpha"), ("null_mean", "null_mean"), ("s_cal", "s_cal")):
                d = max(abs(vals[(pre, a, v)][fld] - known[kf][a][v]) / max(abs(known[kf][a][v]), 1e-300) for v in vits)
                rerun_vs_frozen[f"{a}|{fld}"] = d
        equal_frozen = all(d <= 1e-12 for d in rerun_vs_frozen.values())
        dp = {"realised_offsets_13b_calpha_minus_Bbar": offs,
              "realised_offsets_median": float(np.median(list(offs.values()))),
              "band": Q.get("band"), "mu_band": Q.get("mu_band"),
              "rerun_vs_frozen_max_rel": rerun_vs_frozen, "rerun_equals_frozen_1e-12": equal_frozen, "X": {}}
        for X in XS:
            rows = []
            nd = xc = nu = 0
            nd_f = xc_f = 0
            for a in ("3b", "7b"):
                for v in vits:
                    T13 = vals[(pre, "13b", v)]["T"]
                    lo, hi = bnd[X]["lo"][a][v], bnd[X]["hi"][a][v]
                    if X == "s_cal" and known["s_cal"][a][v] == 0.0:
                        dec = "undecided"
                    elif T13 < lo:
                        dec = "below"
                    elif T13 > hi:
                        dec = "above"
                    else:
                        dec = "undecided"
                    x13, xa = vals[(pre, "13b", v)][X], vals[(pre, a, v)][X]
                    obs = obs_order(x13, xa)
                    xa_fr = known["s_cal"][a][v] if X == "s_cal" else (
                        known_unc[a][v] if known_unc is not None else known["T"][a][v] - known["null_mean"][a][v])
                    obs_fr = obs_order(x13, xa_fr)
                    ok = None if dec == "undecided" else (dec == obs)
                    ok_fr = None if dec == "undecided" else (dec == obs_fr)
                    if dec == "undecided":
                        nu += 1
                    else:
                        nd += 1
                        xc += int(ok)
                        nd_f += 1
                        xc_f += int(ok_fr)
                    V[f"PO2b|{pre}|{X}|dec|{a}|{v}"] = dec
                    V[f"PO2b|{pre}|{X}|obs|{a}|{v}"] = obs
                    V[f"PO2b|{pre}|{X}|ok|{a}|{v}"] = ok
                    rows.append({"a": a, "vit": v, "T_13b": T13, "lo": lo, "hi": hi, "dec": dec, "X_13b": x13,
                                 "X_a_rerun": xa, "obs": obs, "ok": ok, "X_a_frozen": xa_fr, "obs_frozen": obs_fr,
                                 "ok_frozen": ok_fr, "T_a_rerun": vals[(pre, a, v)]["T"],
                                 "margin_to_nearest_boundary": min(abs(T13 - lo), abs(T13 - hi))})
            lo_, hi_ = cp95(xc, nd)
            verdict = "fails" if xc < nd else ("not evaluable" if nd < 10 else "holds")
            V[f"PO2b|{pre}|{X}|n_decided"] = nd
            V[f"PO2b|{pre}|{X}|x_correct"] = xc
            V[f"PO2b|{pre}|{X}|n_undecided"] = nu
            V[f"PO2b|{pre}|{X}|cp_lo"] = lo_
            V[f"PO2b|{pre}|{X}|cp_hi"] = hi_
            V[f"PO2b|{pre}|{X}|verdict"] = verdict
            vf = "fails" if xc_f < nd_f else ("not evaluable" if nd_f < 10 else "holds")
            V[f"x|PO2b|{pre}|{X}|frozen|x_correct"] = xc_f
            V[f"x|PO2b|{pre}|{X}|frozen|verdict"] = vf
            dp["X"][X] = {"rows": rows, "n_decided": nd, "x_correct": xc, "n_undecided": nu, "cp": [lo_, hi_],
                          "verdict": verdict, "frozen_3b7b": {"x_correct": xc_f, "verdict": vf},
                          "n_forecast_below": sum(r["dec"] == "below" for r in rows),
                          "n_forecast_above": sum(r["dec"] == "above" for r in rows)}
        # h curve (s_cal): band [o - h, o + h]
        hc = {}
        for h in HS:
            bh = bnd["s_cal_h"][fmt(h)]
            nd = xc = 0
            for a in ("3b", "7b"):
                for v in vits:
                    T13 = vals[(pre, "13b", v)]["T"]
                    if known["s_cal"][a][v] == 0.0:
                        continue
                    if T13 < bh["lo"][a][v]:
                        dec = "below"
                    elif T13 > bh["hi"][a][v]:
                        dec = "above"
                    else:
                        continue
                    nd += 1
                    xc += int(dec == obs_order(vals[(pre, "13b", v)]["s_cal"], vals[(pre, a, v)]["s_cal"]))
            V[f"PO2b|{pre}|s_cal|h={fmt(h)}|n_decided"] = nd
            V[f"PO2b|{pre}|s_cal|h={fmt(h)}|x_correct"] = xc
            hc[fmt(h)] = {"n_decided": nd, "x_correct": xc, "p_hat": (xc / nd) if nd else None}
        dp["h_curve_s_cal"] = hc
        V[f"x|PO2b|{pre}|rerun_equals_frozen"] = equal_frozen
        V[f"x|PO2b|{pre}|realised_offset_median"] = dp["realised_offsets_median"]
        detail["per_pre"][pre] = dp
    import verdict_keys as VK
    miss = [k for k in VK.required_keys("F") if k not in V]
    if miss:
        raise SystemExit(f"internal error: {len(miss)} required F keys missing, e.g. {miss[:3]}")
    os.makedirs(out_root, exist_ok=True)
    doc = {"schema": "OL-VERDICTS v1", "impl": args.impl, "inputs": inputs, "v": V}
    with open(os.path.join(out_root, "verdicts_forecast.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    with open(os.path.join(out_root, "forecast_detail.json"), "w") as f:
        json.dump(detail, f, indent=1, default=float)
    for pre in PRES:
        for X in XS:
            print(f"{args.impl} {pre} {X}: n_decided {V[f'PO2b|{pre}|{X}|n_decided']} x_correct "
                  f"{V[f'PO2b|{pre}|{X}|x_correct']} undecided {V[f'PO2b|{pre}|{X}|n_undecided']} -> "
                  f"{V[f'PO2b|{pre}|{X}|verdict']}", flush=True)
    print(f"wrote {out_root}/verdicts_forecast.json, forecast_detail.json", flush=True)


if __name__ == "__main__":
    main()
