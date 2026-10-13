"""PYT-AXIS copy of vl5_e_vs_i.py (implementation E, S0b sub-role pyt-code-E): plan V-P5 from the E side.
GATED (plan F1(a)).

Copied from analysis/s12_layermax_openllama/early/vl5_e_vs_i.py (sha256 08e6955e...) and rewritten (every new block
is marked "PYT-AXIS"). Kept: the Acc tolerance accumulator, the tolerance form, the pair-level items, and the rule that
I's pair files are summarised with E's OWN definitions applied to I's raw arrays (the I code is not imported, so that
E stays independent of it). Changed:
  * NAMES AND COUNTS ONLY (plan section 11 V-P5, critique B6.4): nothing this script prints or writes contains a value
    of E or of I; a failing item lists only where it failed (pre|lm|vit|metric|subset|field);
  * the verdict level uses the shared comparator analysis/s12_layermax_pythia/compare_verdicts.py (compare(), names
    only) with the shared key lists (verdict_keys.required_keys("EI") on E vs I aggregates, required_keys("FK") on
    forecast/E vs forecast/I), loaded by path;
  * --pre restricts the pair-level part (S7: the primary pre first; plan section 11 order);
  * order rule: no I output is opened before E's own <RES>/verdicts_pyt.json and summary.json exist; their sha256 and
    mtime and the UTC time at which the first I file is opened are recorded (the "order" block); if either is missing
    the script exits 2 before opening anything of I.

Part "pairs" (plan section 11 V-P5), 119 pairs x pre x 3 metrics x 13 subsets + calibrate-then-max:
  * mKNN observed and null counts identical (E mknn.entry_obs*10240 / mknn.null_counts_int16 vs I obs_cnt / null_cnt);
    every exact mKNN summary (T, c_alpha, null mean, s_cal, uncensored, p, argmax, entry c_alpha, M, M_unc, TM)
    identical as Fractions;
  * CKA biased / debiased: E's json summaries vs the same summaries computed here from I's float64 arrays
    (obs_cka_b/d, null_cka_b/d): s_obs, c_alpha, s_cal, uncensored, null mean within |a - b| <= 1e-12 max(|a|, |b|)
    + 1e-15; p exactly equal; argmax and tie counts identical; calibrate-then-max M, M_unc, TM, c at argmax within
    tolerance, argmax identical.
  I pair files: results/S1_2_layermax_pythia/indep/<pre>/pairs/<lm>__<vit>.npz (VERDICTS_SPEC_PYT section 1) with the
  inherited I keys (obs_cnt, null_cnt, obs_cka_b, null_cka_b, obs_cka_d, null_cka_d, key, K, L_A, L_B).
Part "verdicts": compare_verdicts.compare on early/verdicts_pyt.json vs indep/verdicts_pyt.json (EI) and on
  forecast/E/verdicts_forecast.json vs forecast/I/verdicts_forecast.json (FK); 0 mismatches and no missing key are
  required; a missing file fails its item.

Run (S7 after both primary main runs and both primary aggregates: --pre prh_clip+l2; S8 again for both pre and the
forecast files; CPU, 2 threads):
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_pythia/launch_pyt.sh pyt_E_vp5_prh 2 -- \\
      venv/bin/python analysis/s12_layermax_pythia/early/vl5_e_vs_i.py --pre prh_clip+l2 [--part all|pairs|verdicts]

Output: <RES>/validation/VP5_E_vs_I[.<pre>].json (counts and names only)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402  (constants and functions only; reads no data at import)

SUBSETS = ("full", "fixed_r0.5", "fixed_r1.0", "grid2", "grid3", "grid5", "grid9", "grid17",
           "LMall_ViTlast", "LMlast_ViTall", "noLMemb", "LMdeep", "LMemb_only")
TOL_FIELDS = ("s_obs", "c_alpha", "s_cal", "uncensored", "null_mean")
OIDX = 476
CNT = LC.M_ITEMS * LC.KNN_K


def excess(a, b):
    """|a-b| / (1e-12 max(|a|,|b|) + 1e-15): <= 1 passes."""
    return abs(a - b) / (1e-12 * max(abs(a), abs(b)) + 1e-15)


class Acc:  # PYT-AXIS: counts and failure locations only (no values)
    def __init__(self):
        self.n = 0
        self.n_fail = 0
        self.fails = []

    def add(self, a, b, where, tol=True):
        self.n += 1
        if tol:
            bad = excess(float(a), float(b)) > 1.0
        else:
            bad = a != b
        if bad:
            self.n_fail += 1
            if len(self.fails) < 40:
                self.fails.append(where)

    def out(self):
        return {"n": self.n, "n_fail": self.n_fail, "fail_locations_first40": self.fails}


def sha(path):
    return LC.sha256_file(path)


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


def compare_pairs(AG, pres):
    acc = {}

    def A(k):
        if k not in acc:
            acc[k] = Acc()
        return acc[k]

    n_pairs = 0
    t_first = None
    for pre in pres:  # PYT-AXIS
        for lm, _ in LC.LMS:
            for vit, _ in LC.VITS:
                n_pairs += 1
                where = f"{pre}|{lm}|{vit}"
                base = os.path.join(LC.pairs_dir(pre, lm), f"{lm}__{vit}")
                rec = json.load(open(base + ".json"))
                ze = np.load(base + ".npz")
                if t_first is None:
                    t_first = LC.utc()
                zi = np.load(os.path.join(LC.PYT_INDEP_RES, pre, "pairs", f"{lm}__{vit}.npz"))  # PYT-AXIS
                A("meta.key_K_shape").add(int(rec["perm_key"] == str(zi["key"]) and rec["K"] == int(zi["K"])
                                              and rec["L_A"] == int(zi["L_A"]) and rec["L_B"] == int(zi["L_B"])), 1,
                                          where, tol=False)
                LA, LB = rec["L_A"], rec["L_B"]
                subs = LC.subsets(LA, LB)
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
                for f in ("M", "M_unc", "TM", "cM", "aA", "aB", "aA_unc", "aB_unc", "muMunc"):  # PYT-AXIS
                    A(f"mknn.exact.ctm.{f}").add(xE["ctm"][f], xI["ctm"][f], where, tol=False)
                for met, im in (("cka_biased", "b"), ("cka_debiased", "d")):
                    oI = np.asarray(zi[f"obs_cka_{im}"], dtype=np.float64)
                    nuI = np.asarray(zi[f"null_cka_{im}"], dtype=np.float64)
                    for q in SUBSETS:
                        e = rec["summaries"][met][q]
                        i = cka_summary(oI, nuI, *subs[q])
                        w2 = f"{where}|{met}|{q}"
                        for f in TOL_FIELDS:
                            A(f"{met}.{f}").add(e[f], i[f], f"{w2}|{f}")
                        A(f"{met}.p_value_exact").add(float(e["p_value"]), i["p_value"], w2, tol=False)
                        A(f"{met}.argmax_identical").add(int(e["argmax_A"] == i["argmax_A"] and e["argmax_B"] == i["argmax_B"]),
                                                         1, w2, tol=False)
                        A(f"{met}.n_tied_max_identical").add(int(e["n_tied_max"]), i["n_tied_max"], w2, tol=False)
                    ce, ci = rec["calib_then_max"][met], cka_ctm(oI, nuI)
                    w2 = f"{where}|{met}|ctm"
                    for f in ("M", "M_unc", "s_obs_at_argmax", "c_alpha_at_argmax"):
                        A(f"ctm.{met}.{f}").add(ce[f], ci[f], f"{w2}|{f}")
                    A(f"ctm.{met}.argmax_identical").add(int(all(ce[k] == ci[k] for k in ("argmax_A", "argmax_B",
                                                                                         "M_unc_argmax_A", "M_unc_argmax_B"))),
                                                         1, w2, tol=False)
    out = {"n_pair_pre": n_pairs, "first_I_file_opened_utc": t_first}
    for k, v in sorted(acc.items()):
        out[k] = v.out()
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
    out["pass"] = all(out["plan_items_pass"].values()) and n_pairs == len(pres) * len(LC.LMS) * len(LC.VITS)  # PYT-AXIS
    return out


def _load(name):  # PYT-AXIS: shared modules by path (names-only comparator and key lists)
    path = os.path.join(LC.AXIS, name + ".py")
    spec = importlib.util.spec_from_file_location("pyt_shared_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, LC.AXIS)   # compare_verdicts imports verdict_keys
    spec.loader.exec_module(mod)
    return mod, sha(path)


def compare_verdict_files():  # PYT-AXIS
    CV, cv_sha = _load("compare_verdicts")
    VK, vk_sha = _load("verdict_keys")
    items = {"EI": (os.path.join(LC.RES, "verdicts_pyt.json"), os.path.join(LC.PYT_INDEP_RES, "verdicts_pyt.json"), "EI"),
             "FK": (os.path.join(LC.PYT_FORECAST, "E", "verdicts_forecast.json"),
                    os.path.join(LC.PYT_FORECAST, "I", "verdicts_forecast.json"), "FK")}
    out = {"compare_verdicts_sha256": cv_sha, "verdict_keys_sha256": vk_sha}
    for name, (a, b, sec) in items.items():
        if not (os.path.exists(a) and os.path.exists(b)):
            out[name] = {"a": a, "b": b, "missing": [p for p in (a, b) if not os.path.exists(p)], "pass": False}
            continue
        A_, B_ = json.load(open(a)), json.load(open(b))
        rep = CV.compare(A_["v"], B_["v"], VK.required_keys(sec))
        rep = {k: rep[k] for k in ("n_required", "n_compared", "n_missing_in_a", "n_missing_in_b", "n_mismatch",
                                   "mismatching_keys", "missing_in_a", "missing_in_b", "pass") if k in rep}
        rep.update({"a": a, "b": b, "a_sha256": sha(a), "b_sha256": sha(b), "sections": sec,
                    "a_impl": A_.get("impl"), "b_impl": B_.get("impl")})
        out[name] = rep
    out["pass"] = all(out[k]["pass"] for k in items)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("all", "pairs", "verdicts"), default="all")
    ap.add_argument("--pre", nargs="*", default=list(LC.PRES), choices=list(LC.PRES))  # PYT-AXIS
    args = ap.parse_args()
    LC.gate("early/vl5_e_vs_i.py")   # PYT-AXIS: plan F1(a), first action
    pres = tuple(p for p in LC.PRES if p in args.pre)
    t0 = time.time()
    order = {"checked_utc": LC.utc()}
    for f in ("summary.json", "verdicts_pyt.json"):  # PYT-AXIS
        p = os.path.join(LC.RES, f)
        if not os.path.exists(p):
            print(f"[vl5] E's {p} does not exist: V-P5 must run after E's aggregate (no I file opened)", flush=True)
            sys.exit(2)
        order[f] = {"sha256": sha(p), "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(p)))}
    order["rule"] = "E summary.json and verdicts_pyt.json existed and were hashed before the first I file was opened"
    import aggregate_pyt as AG   # E's exact mKNN code (module import reads no data)  PYT-AXIS
    res = {"order": order, "part": args.part, "pres": list(pres)}
    if args.part in ("all", "pairs"):
        res["pairs"] = compare_pairs(AG, pres)
        print("pairs", "PASS" if res["pairs"]["pass"] else "FAIL", res["pairs"]["plan_items_pass"], flush=True)
        for k, v in res["pairs"]["fail_counts"].items():
            if v:
                print("  fail count", k, v, flush=True)
    if args.part in ("all", "verdicts"):
        order.setdefault("first_I_file_opened_utc", LC.utc())
        res["verdicts"] = compare_verdict_files()
        for k in ("EI", "FK"):
            r = res["verdicts"][k]
            print("verdicts", k, "PASS" if r["pass"] else "FAIL",
                  {kk: r.get(kk) for kk in ("n_required", "n_compared", "n_mismatch", "n_missing_in_a",
                                            "n_missing_in_b", "missing")}, flush=True)
            if r.get("mismatching_keys"):
                print("  mismatching keys (names only):", r["mismatching_keys"][:50], flush=True)
    if "pairs" in res:
        order["first_I_file_opened_utc"] = res["pairs"]["first_I_file_opened_utc"]
    res["pass"] = all(res[k]["pass"] for k in ("pairs", "verdicts") if k in res)
    res["meta"] = {"plan_item": "ANALYSIS_PLAN_PYTHIA.md section 11 V-P5 (E side); counts and key names only",
                   "tolerance": "|a-b| <= 1e-12 max(|a|,|b|) + 1e-15; mKNN exact; p, counts, argmax exact",
                   "elapsed_s": time.time() - t0, "finished_utc": LC.utc()}
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    sfx = "" if pres == LC.PRES else "." + "+".join(pres)
    name = {"all": f"VP5_E_vs_I{sfx}.json", "pairs": f"VP5_pairs{sfx}.json", "verdicts": f"VP5_verdicts{sfx}.json"}[args.part]
    with open(os.path.join(LC.RES, "validation", name), "w") as f:
        json.dump(LC.jsonable(res), f, indent=1)
    print("V-P5", "PASS" if res["pass"] else "FAIL", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
