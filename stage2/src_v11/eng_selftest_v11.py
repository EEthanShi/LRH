"""Self-test of the frozen engineering decision rule (eng_v11.decide / projection) on synthetic stage summaries.
EXCLUDED; no training. Writes only results/S2_2_v11/eng_selftest/<case>/ (eng_v11's OUT, PLANS and STATUS are redirected
there, so plans_v11/ and STATUS.md are untouched).

    python src_v11/eng_selftest_v11.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eng_v11 as E  # noqa: E402

BASE = os.path.join(E.ROOT, "results", "S2_2_v11", "eng_selftest")


def ev(t, ok=True, dC=1e-6, fp=0, cfg=None, evals=0.0, th=0.0):
    e = {"n_planned": 8, "n_done": 8, "n_ok": 8 if ok else 7, "all_ok": ok, "delta_C_max": dC, "delta_C_median": dC / 2,
         "fp_fail": fp, "t_mean_s": t, "t_median_s": t, "t_proj_s": t, "t_head_mean_s": th, "evals_mean": evals,
         "substeps_max": 0, "disp_P_median": 0.1, "euler_identity_resid_max": 1e-16, "sharpness": [],
         "by_N": {"12500": {"acc_bank_mean": 0.9, "t_mean_s": t}, "100000": {"acc_bank_mean": 0.9, "t_mean_s": t}},
         "gpu_s": 8 * t}
    if cfg is not None:
        e["cfg"] = cfg
    return e


H64 = {"head_opt": "heun", "K": 64}
H256 = {"head_opt": "heun", "K": 256}
M16 = {"head_opt": "midpoint", "K": 16, "fp_tol": 1e-12, "fp_max_iter": 30}


def case(name, stages, K_min, expect_status, expect_flow=None, expect_reason=None):
    d = os.path.join(BASE, name)
    os.makedirs(d, exist_ok=True)
    E.OUT, E.PLANS, E.STATUS = d, d, os.path.join(d, "STATUS_selftest.md")
    S = {"stages": stages, "order": list(stages), "K_min": K_min}
    c = E.decide(S, 4, run_scale=False)
    ok = c["status"] == expect_status
    if expect_flow is not None:
        ok &= c["flow"] == expect_flow
    if expect_reason is not None:
        ok &= any(r.startswith(expect_reason) for r in c["reasons"])
    marker = os.path.exists(os.path.join(d, "ENG_GO" if expect_status == "GO" else "ENG_STOP"))
    ok &= marker
    print(f"{'PASS' if ok else 'FAIL'} {name}: status {c['status']} flow {c['flow']} reasons {c['reasons']} "
          f"projected {c.get('projection_total_gpu_h', float('nan')):.2f} GPU-h", flush=True)
    return ok, c


def main():
    base = {"base_MOM": ev(12.5), "euler_K1": ev(13.0, evals=488, th=0.2)}
    res = {}
    res["go_heun"] = case("go_heun", {**base, "flow_HEUN-K64": ev(22.0, cfg=H64)}, 1, "GO", H64)
    res["cheapest_wins"] = case("cheapest_wins", {**base, "flow_HEUN-K256": ev(30.0, cfg=H256),
                                                  "flow_MID-K16": ev(25.0, cfg=M16)}, 1, "GO", M16)
    res["stop_a_none"] = case("stop_a_none", {**base, "flow_HEUN-K64": ev(22.0, dC=2e-4, cfg=H64)}, 1, "STOP",
                              expect_reason="STOP-a")
    res["stop_a_fpfail"] = case("stop_a_fpfail", {**base, "flow_MID-K16": ev(22.0, fp=1, cfg=M16)}, 1, "STOP",
                                expect_reason="STOP-a")
    res["stop_a_slow"] = case("stop_a_slow", {**base, "flow_HEUN-K256": ev(45.0, cfg=H256)}, 1, "STOP",
                              expect_reason="STOP-a")
    res["stop_b_budget"] = case("stop_b_budget", {"base_MOM": ev(20.0), "euler_K1": ev(21.0, evals=488, th=0.2),
                                                  "flow_HEUN-K256": ev(39.0, cfg=H256)}, 1, "STOP", expect_reason="STOP-b")
    res["kmin_none"] = case("kmin_none", {"base_MOM": ev(12.5), "euler_K64": ev(30.0, ok=False),
                                          "flow_HEUN-K64": ev(22.0, cfg=H64)}, None, "GO", H64)
    k = res["kmin_none"][1]["projection_rows"]
    ok_drop = all(k[a]["runs"] == 0 for a in ("SGD-K", "KS-4", "KS-16"))
    print(f"{'PASS' if ok_drop else 'FAIL'} kmin_none drops SGD-K and KS from the projection", flush=True)
    g = res["go_heun"][1]
    exp_total = sum(r["gpu_h"] for r in g["projection_rows"].values()) + g["eng_gpu_h"]
    ok_sum = abs(exp_total - g["projection_total_gpu_h"]) < 1e-9
    print(f"{'PASS' if ok_sum else 'FAIL'} projection total = sum of rows + engineering ({g['projection_total_gpu_h']:.2f})")
    allok = all(v[0] for v in res.values()) and ok_drop and ok_sum
    json.dump({"pass": allok, "cases": {n: {"pass": v[0], "status": v[1]["status"], "flow": v[1]["flow"],
                                            "reasons": v[1]["reasons"],
                                            "projection_total_gpu_h": v[1].get("projection_total_gpu_h")}
                                        for n, v in res.items()}},
              open(os.path.join(BASE, "eng_selftest.json"), "w"), indent=1)
    print("ALL PASS" if allok else "SOME FAILED")


if __name__ == "__main__":
    main()
