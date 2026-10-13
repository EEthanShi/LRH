"""OL-AXIS copy of vl5_e_vs_i.py (implementation E, S0b sub-role E of oll-axis-extract): plan V-O5 from
the E side. GATED (plan F1(a)).

Copied from analysis/s12_layermax_llama/early/vl5_e_vs_i.py (sha256 e31773fd...) and rewritten (every new
line is marked "OL-AXIS"). Kept: the Acc tolerance accumulator, the tolerance form and the pair-level
items. Changed: I's pair files are summarised with E's OWN definitions applied to I's raw arrays (the
I code copy of this axis is not imported, so that E stays independent of it); the mKNN side is compared
in exact rationals; the verdict-level part (compare_verdicts.py, VERDICTS_SPEC section 0) is new.

Order rule (task text of S0b role E): no I output is opened before E's own <RES>/summary.json and
<RES>/verdicts_ol.json exist; their sha256 and mtime and the UTC time at which the first I file is opened
are recorded in the output ("order" block). If either E file is missing the script exits 2 before
opening anything of I.

Part "pairs" (plan section 11 V-O5), all 51 pairs x 2 pre x 3 metrics x 13 subsets + calibrate-then-max:
  * mKNN observed and null counts identical (E mknn.entry_obs*10240 / mknn.null_counts_int16 vs I obs_cnt /
    null_cnt); every exact mKNN summary (T, c_alpha, null mean, s_cal, uncensored, p, argmax, entry c_alpha,
    M, M_unc, TM) identical as Fractions;
  * CKA biased / debiased: E's json summaries vs the same summaries computed here from I's float64 arrays
    (obs_cka_b/d, null_cka_b/d): s_obs, c_alpha, s_cal, uncensored, null mean within
    |a - b| <= 1e-12 max(|a|, |b|) + 1e-15; p exactly equal; argmax and tie counts identical; calibrate-then-max
    M, M_unc, TM, c at argmax within tolerance, argmax identical.
  I pair files: results/S1_2_layermax_openllama/indep/<pre>/pairs/<lm>__<vit>.npz with the LLaMA-axis I keys
  (obs_cnt, null_cnt, obs_cka_b, null_cka_b, obs_cka_d, null_cka_d, key, K, L_A, L_B).
Part "verdicts" (VERDICTS_SPEC section 0): compare_verdicts.compare on
  early/verdicts_ol.json vs indep/verdicts_ol.json (sections EI),
  forecast/E/verdicts_forecast.json vs forecast/I/verdicts_forecast.json (F),
  joint/E/verdicts_joint.json vs joint/I/verdicts_joint.json (J);
  0 mismatches and no missing required key are required. A missing file is reported and fails its item.

Run (step S8, after E's aggregate_ol.py has written summary.json and verdicts_ol.json; the verdict part also
after I's aggregate, forecast_ol.py --impl E/I and joint_summary.py --impl E/I; CPU, 2 threads):
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_openllama/launch_ol.sh ol_E_vo5 2 -- \\
      venv/bin/python analysis/s12_layermax_openllama/early/vl5_e_vs_i.py [--part all|pairs|verdicts]

Output: <RES>/validation/VO5_E_vs_I.json (--part all), VO5_pairs.json or VO5_verdicts.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402  (constants and functions only; reads no data at import)
import guard  # noqa: E402

METS = {"mknn": "mknn", "cka_biased": "cka_b", "cka_debiased": "cka_d"}
SUBSETS = ("full", "fixed_r0.5", "fixed_r1.0", "grid2", "grid3", "grid5", "grid9", "grid17",
           "LMall_ViTlast", "LMlast_ViTall", "noLMemb", "LMdeep", "LMemb_only")
TOL_FIELDS = ("s_obs", "c_alpha", "s_cal", "uncensored", "null_mean")
OIDX = 476
CNT = LC.M_ITEMS * LC.KNN_K


def ok_tol(a, b):
    return abs(a - b) <= 1e-12 * max(abs(a), abs(b)) + 1e-15


def excess(a, b):
    """|a-b| / (1e-12 max(|a|,|b|) + 1e-15): <= 1 passes."""
    return abs(a - b) / (1e-12 * max(abs(a), abs(b)) + 1e-15)


class Acc:
    def __init__(self):
        self.n = 0
        self.n_fail = 0
        self.max_abs = 0.0
        self.max_excess = 0.0
        self.fails = []

    def add(self, a, b, where, tol=True):
        self.n += 1
        if tol:
            a, b = float(a), float(b)
            d = abs(a - b)
            self.max_abs = max(self.max_abs, d)
            e = excess(a, b)
            self.max_excess = max(self.max_excess, e)
            bad = e > 1.0
        else:
            bad = a != b
            try:
                self.max_abs = max(self.max_abs, abs(float(a) - float(b)))
            except (TypeError, ValueError):
                pass
        if bad:
            self.n_fail += 1
            if len(self.fails) < 40:
                self.fails.append({"where": where, "E": str(a), "I": str(b)})

    def out(self):
        return {"n": self.n, "n_fail": self.n_fail, "max_abs": self.max_abs, "max_excess_over_tol": self.max_excess,
                "fails_first40": self.fails}


def sha(path):
    return LC.C12.sha256_file(path)


def cka_summary(obs, null, ia, ib):
    """E's subset summary (run_layermax.subset_summary + hrhv2.calibrate.summarize) on I's float64 arrays."""
    so = obs[np.ix_(ia, ib)]
    flat = so.ravel()
    j = int(np.argmax(flat))
    T = float(flat[j])
    Tk = null[:, ia][:, :, ib].reshape(null.shape[0], -1).max(axis=1)
    c = float(np.sort(np.concatenate(([T], Tk)))[OIDX - 1])
    den = 1.0 - c
    return {"s_obs": T, "c_alpha": c, "s_cal": float(max((T - c) / den, 0.0)) if den > 0 else 0.0,
            "null_mean": float(Tk.mean()), "uncensored": float(T - Tk.mean()),
            "p_value": float((1.0 + np.sum(Tk >= T)) / (Tk.size + 1)),
            "argmax_A": ia[j // len(ib)], "argmax_B": ib[j % len(ib)], "n_tied_max": int(np.sum(flat == flat.max()))}


def cka_ctm(obs, null):
    srt = np.sort(np.concatenate([obs[None], null], axis=0), axis=0)
    c = srt[OIDX - 1]
    nm = null.mean(axis=0)
    den = 1.0 - c
    scal = np.where(den > 0, np.maximum((obs - c) / np.where(den > 0, den, 1.0), 0.0), 0.0)
    unc = obs - nm
    LB = obs.shape[1]
    j = int(np.argmax(scal.ravel()))
    ju = int(np.argmax(unc.ravel()))
    return {"M": float(scal.ravel()[j]), "M_unc": float(unc.ravel()[ju]), "s_obs_at_argmax": float(obs.ravel()[j]),
            "c_alpha_at_argmax": float(c.ravel()[j]), "argmax_A": j // LB, "argmax_B": j % LB,
            "M_unc_argmax_A": ju // LB, "M_unc_argmax_B": ju % LB}


def compare_pairs(AG):
    acc = {}

    def A(k):
        if k not in acc:
            acc[k] = Acc()
        return acc[k]

    n_pairs = 0
    t_first = None
    for pre in LC.PRES:
        for lm, _ in LC.LMS:
            for vit, _ in LC.VITS:
                n_pairs += 1
                where = f"{pre}|{lm}|{vit}"
                base = os.path.join(LC.pairs_dir(pre, lm), f"{lm}__{vit}")
                rec = json.load(open(base + ".json"))
                ze = np.load(base + ".npz")
                if t_first is None:
                    t_first = LC.utc()
                zi = np.load(os.path.join(LC.OL_INDEP_RES, pre, "pairs", f"{lm}__{vit}.npz"))
                A("meta.key_K_shape").add(int(rec["perm_key"] == str(zi["key"]) and rec["K"] == int(zi["K"])
                                              and rec["L_A"] == int(zi["L_A"]) and rec["L_B"] == int(zi["L_B"])), 1,
                                          where, tol=False)
                LA, LB = rec["L_A"], rec["L_B"]
                subs = LC.subsets(LA, LB)
                # ---- mKNN: counts and every exact summary
                cE = np.rint(ze["mknn.entry_obs"] * CNT).astype(np.int64)
                cI = np.asarray(zi["obs_cnt"]).astype(np.int64)
                nE = ze["mknn.null_counts_int16"].astype(np.int64)
                nI = np.asarray(zi["null_cnt"]).astype(np.int64)
                A("mknn.obs_counts_identical").add(int(np.array_equal(cE, cI)), 1, where, tol=False)
                A("mknn.null_counts_identical").add(int(np.array_equal(nE, nI)), 1, where, tol=False)
                xE = AG._mknn_exact(ze, subs, False)
                xI = AG._mknn_exact({"mknn.entry_obs": cI / CNT, "mknn.null_counts_int16": nI}, subs, False)
                for q in SUBSETS:
                    for f in ("T", "c", "mu", "scal", "unc", "p", "c_entry", "mu_entry", "n_tied"):
                        A(f"mknn.exact.{f}").add(xE[q][f], xI[q][f], f"{where}|{q}", tol=False)
                    A("mknn.argmax_identical").add(int(xE[q]["aA"] == xI[q]["aA"] and xE[q]["aB"] == xI[q]["aB"]), 1,
                                                   f"{where}|{q}", tol=False)
                for f in ("M", "M_unc", "TM", "cM", "aA", "aB", "aA_unc", "aB_unc"):
                    A(f"mknn.exact.ctm.{f}").add(xE["ctm"][f], xI["ctm"][f], where, tol=False)
                # ---- CKA: E json vs E's definitions applied to I's float64 arrays
                for met, im in (("cka_biased", "b"), ("cka_debiased", "d")):
                    oI = np.asarray(zi[f"obs_cka_{im}"], dtype=np.float64)
                    nuI = np.asarray(zi[f"null_cka_{im}"], dtype=np.float64)
                    info = acc.setdefault(f"info.{met}.entry_obs_maxabs", {"max_abs": 0.0})   # informational only
                    info["max_abs"] = max(info["max_abs"], float(np.max(np.abs(ze[f"{met}.entry_obs"] - oI))))
                    for q in SUBSETS:
                        e = rec["summaries"][met][q]
                        i = cka_summary(oI, nuI, *subs[q])
                        w2 = f"{where}|{met}|{q}"
                        for f in TOL_FIELDS:
                            A(f"{met}.{f}").add(e[f], i[f], w2)
                        A(f"{met}.p_value_exact").add(float(e["p_value"]), i["p_value"], w2, tol=False)
                        A(f"{met}.argmax_identical").add(int(e["argmax_A"] == i["argmax_A"] and e["argmax_B"] == i["argmax_B"]),
                                                         1, w2, tol=False)
                        A(f"{met}.n_tied_max_identical").add(int(e["n_tied_max"]), i["n_tied_max"], w2, tol=False)
                    ce, ci = rec["calib_then_max"][met], cka_ctm(oI, nuI)
                    w2 = f"{where}|{met}|ctm"
                    for f in ("M", "M_unc", "s_obs_at_argmax", "c_alpha_at_argmax"):
                        A(f"ctm.{met}.{f}").add(ce[f], ci[f], w2)
                    A(f"ctm.{met}.argmax_identical").add(int(all(ce[k] == ci[k] for k in ("argmax_A", "argmax_B",
                                                                                         "M_unc_argmax_A", "M_unc_argmax_B"))),
                                                         1, w2, tol=False)
    out = {"n_pair_pre": n_pairs, "first_I_file_opened_utc": t_first}
    for k, v in sorted(acc.items()):
        out[k] = v.out() if isinstance(v, Acc) else v
    fails = {k: v["n_fail"] for k, v in out.items() if isinstance(v, dict) and "n_fail" in v}
    out["fail_counts"] = fails
    out["plan_items_pass"] = {
        "mknn_counts_identical": fails["mknn.obs_counts_identical"] == 0 and fails["mknn.null_counts_identical"] == 0,
        "mknn_exact_summaries_identical": all(v == 0 for k, v in fails.items() if k.startswith("mknn.exact")),
        "argmax_identical": all(v == 0 for k, v in fails.items() if k.endswith("argmax_identical")),
        "tolerance_fields": all(v == 0 for k, v in fails.items()
                                if any(k == f"{m}.{f}" for m in ("cka_biased", "cka_debiased") for f in TOL_FIELDS)
                                or k.startswith("ctm.") and not k.endswith("argmax_identical")),
        "p_exact": all(fails[f"{m}.p_value_exact"] == 0 for m in ("cka_biased", "cka_debiased")),
        "meta": fails["meta.key_K_shape"] == 0,
    }
    out["pass"] = all(out["plan_items_pass"].values()) and n_pairs == 2 * 3 * len(LC.VITS)
    return out


def compare_verdict_files():
    import compare_verdicts as CV
    import verdict_keys as VK
    items = {"EI": (os.path.join(LC.RES, "verdicts_ol.json"), os.path.join(LC.OL_INDEP_RES, "verdicts_ol.json"), "EI"),
             "F": (os.path.join(LC.OL_FORECAST, "E", "verdicts_forecast.json"),
                   os.path.join(LC.OL_FORECAST, "I", "verdicts_forecast.json"), "F"),
             "J": (os.path.join(LC.OL_JOINT, "E", "verdicts_joint.json"),
                   os.path.join(LC.OL_JOINT, "I", "verdicts_joint.json"), "J")}
    out = {}
    for name, (a, b, sec) in items.items():
        if not (os.path.exists(a) and os.path.exists(b)):
            out[name] = {"a": a, "b": b, "missing": [p for p in (a, b) if not os.path.exists(p)], "pass": False}
            continue
        A_, B_ = json.load(open(a)), json.load(open(b))
        rep = CV.compare(A_["v"], B_["v"], VK.required_keys(sec))
        rep.update({"a": a, "b": b, "a_sha256": sha(a), "b_sha256": sha(b), "sections": sec,
                    "a_impl": A_.get("impl"), "b_impl": B_.get("impl")})
        out[name] = rep
    out["pass"] = all(out[k]["pass"] for k in items)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("all", "pairs", "verdicts"), default="all")
    args = ap.parse_args()
    guard.require_partb_frozen(LC.RES, script="early/vl5_e_vs_i.py")   # plan F1(a): first action
    t0 = time.time()
    # ---- order rule: E's own summary and verdicts must be final before any I output is opened
    order = {"checked_utc": LC.utc()}
    for f in ("summary.json", "verdicts_ol.json"):
        p = os.path.join(LC.RES, f)
        if not os.path.exists(p):
            print(f"[vl5] E's {p} does not exist: V-O5 must run after E's aggregate (no I file opened)", flush=True)
            sys.exit(2)
        order[f] = {"sha256": sha(p), "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(p)))}
    order["rule"] = "E summary.json and verdicts_ol.json existed and were hashed before the first I file was opened"
    import aggregate_ol as AG   # E's exact mKNN code (module import reads no data)
    res = {"order": order, "part": args.part}
    if args.part in ("all", "pairs"):
        res["pairs"] = compare_pairs(AG)
        print("pairs", "PASS" if res["pairs"]["pass"] else "FAIL", res["pairs"]["plan_items_pass"], flush=True)
        for k, v in res["pairs"]["fail_counts"].items():
            if v:
                print("  fail", k, v, flush=True)
    if args.part in ("all", "verdicts"):
        order.setdefault("first_I_file_opened_utc", LC.utc())
        res["verdicts"] = compare_verdict_files()
        for k in ("EI", "F", "J"):
            r = res["verdicts"][k]
            print("verdicts", k, "PASS" if r["pass"] else "FAIL",
                  {kk: r.get(kk) for kk in ("n_required", "n_compared", "n_mismatch", "missing")}, flush=True)
    if "pairs" in res:
        order["first_I_file_opened_utc"] = res["pairs"]["first_I_file_opened_utc"]
    res["pass"] = all(res[k]["pass"] for k in ("pairs", "verdicts") if k in res)
    res["meta"] = {"plan_item": "ANALYSIS_PLAN_OPENLLAMA.md section 11 V-O5 (E side)",
                   "tolerance": "|a-b| <= 1e-12 max(|a|,|b|) + 1e-15; mKNN exact; p, counts, argmax exact",
                   "elapsed_s": time.time() - t0, "finished_utc": LC.utc()}
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    name = {"all": "VO5_E_vs_I.json", "pairs": "VO5_pairs.json", "verdicts": "VO5_verdicts.json"}[args.part]
    with open(os.path.join(LC.RES, "validation", name), "w") as f:
        json.dump(LC.jsonable(res), f, indent=1)
    print("V-O5", "PASS" if res["pass"] else "FAIL", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
