#!/usr/bin/env python3
"""s23-review-A2: build items_table.tsv and review_A2.json from recompute_A2.json and order_check_A2.json (stdlib only)."""
import json
import os

OUT = os.path.join(os.path.expanduser("~"), "hrh_v2_stage2_20261008", "s23", "results", "stageA", "review_A2")
R = json.load(open(os.path.join(OUT, "recompute_A2.json")))
O = json.load(open(os.path.join(OUT, "order_check_A2.json")))
F0 = R["F0"]
rows = []


def add(item, tag, verdict, review, agree, meaning):
    rows.append({"item": item, "source": tag, "verdict": verdict, "review_numbers": review, "frozen_agrees": agree,
                 "meaning": meaning})


def cmp_ok(k):
    c = R["compare_frozen"][k]
    return c["confirmed_review"] == c["confirmed_frozen"] and c["holds_review"] == c["holds_frozen"]


n_ord = len(O["checks"]); n_ord_fail = O["n_fail"]
add("ORDER", "[Design]", "PASS" if n_ord_fail == 0 else "FAIL",
    f"{n_ord - n_ord_fail}/{n_ord} checks pass; A2 files born 08:22:57-08:31:54Z, A2 FROZEN 08:32:28Z, first new run file "
    f"08:33:34Z, last 09:09:34Z; at the A2 freeze results/stageA held 8 files (gates outputs + pilot caches); first arm-0 cache "
    f"09:09:40Z; A3 files 09:18:54-09:20:36Z (crash 09:10:16Z), N_match 09:21:30Z; A4 files 09:29:33-09:31:24Z (crash 09:25:17Z), "
    f"results 09:31:58Z; A3 cache 7/7 byte-equal; A4: 430 cache files unchanged, only line 705 differs; ledger 12.7063 h",
    "n/a", "A2 frozen before any new item and before any statistic; analyses only after 3456/3456; each fix frozen before its output")
P = F0["P1_A"]
add("F0.P1_A", "[Thm; rate Thm+CLT]", "PASS" if P["confirmed"] else "FAIL",
    f"ratio {P['ratio']:.4f} [{P['ci_lo']:.4f}, {P['ci_hi']:.4f}] < 0.35 (root-N 0.125); p {P['p']:.1e}; Holm thr {P['holm']['thr']:.4f}",
    cmp_ok("P1_A"), "thm:link part 1 (A -> K*) at the root-N rate of thm:peer part 1")
P = F0["P1_B"]
add("F0.P1_B", "[Thm; rate Model]", "PASS" if P["confirmed"] else "FAIL",
    f"ratio {P['ratio']:.4f} [{P['ci_lo']:.4f}, {P['ci_hi']:.4f}] < 0.35; p {P['p']:.1e}", cmp_ok("P1_B"),
    "thm:link part 1 (B -> K*^(t)); no plateau, so the sharp link is not falsified")
P = F0["P2"]
add("F0.P2", "[Thm]", "PASS" if P["confirmed"] else "FAIL",
    f"cross S_ex8 exact T_E {P['x']}/{P['n']}, CP lower {P['cp_lo']:.3f} > 0.9; p {P['p']:.2e}; guard within T_E A 96/96, B 96/96",
    cmp_ok("P2"), "thm:link part 2: exact agreement on T <= T_E reliable at N_top")
P = F0["P3a"]
add("F0.P3a", "[Thm]", "PASS" if P["confirmed"] else "FAIL",
    f"cross Gs_max <= eps_geo 0.1739 in {P['N0']['x']}/96 (N0) and {P['Ntop']['x']}/96 (N_top), CP upper {P['Ntop']['cp_hi']:.3f} < 0.8; "
    f"p {P['p']:.1e}; guards within Gs_max and 1-CKA 96/96 at both N", cmp_ok("P3a"),
    "thm:link part 3: scale-normalized geometry obstructed (cross Gs_max -> omega_psi 0.348)")
P = F0["P3b"]
add("F0.P3b", "[Thm; band Model]", "PASS" if P["confirmed"] else "FAIL",
    f"cross 1-CKA {P['cross_mean']:.6f} (90% CI {P['ci90_cross'][0]:.6f}-{P['ci90_cross'][1]:.6f}) in band {P['band'][0]:.4f}-{P['band'][1]:.4f}; "
    f"within {P['within_A_mean']:.1e} / {P['within_B_mean']:.1e} <= 0.01; p {P['p']:.0e}", cmp_ok("P3b"),
    "thm:link part 3: CKA tends to its limit below one (1-CKA^inf = 0.141355)")
P = F0["P3c"]
add("F0.P3c", "[Thm]", "PASS" if P["confirmed"] else "FAIL",
    f"inverse-link counterfactual median {P['median_c']['N0']:.5f} -> {P['median_c']['Ntop']:.6f}, ratio {P['ratio']:.4f} "
    f"[{P['ci_lo']:.4f}, {P['ci_hi']:.4f}] < 0.35; K_B entries <= 0: 0", cmp_ok("P3c"),
    "thm:link part 3 decomposition: the geometric gap is the link g_t and nothing else")
P = F0["P4a"]
add("F0.P4a", "[Thm]", "PASS" if P["confirmed"] else "FAIL",
    f"cross iso_inf <= eps_geo in {P['x']}/{P['n']}, CP lower {P['cp_lo']:.3f}; p {P['p']:.2e}; guard as P3a at N_top", cmp_ok("P4a"),
    "thm:link part 4: the isotonic discrepancy removes the obstruction")
P = F0["P5_order"]
rr = "; ".join(f"{k}: Pi {v['Pi']:.2e} [{v['ci_lo']:.2e}, {v['ci_hi']:.2e}] rho {v['rho']:.3f}" for k, v in P["rungs"].items())
add("F0.P5_order", "[Model: Thm + Lemma L premise]", "PASS" if P["confirmed"] else "FAIL",
    f"{rr}; no plateau on the 3 confirmatory rungs, plateau on none of 4", cmp_ok("P5_order"),
    "[Model] order on all 256 points: cross disagreement does not exceed within; not a consequence of thm:link")
P = F0["P5_geom"]
rr = "; ".join(f"{k}: Pi {v['Pi']:.5f} [{v['ci_lo']:.5f}, {v['ci_hi']:.5f}] rho {v['rho']:.0f}" for k, v in P["rungs"].items())
add("F0.P5_geom", "[Thm]", "PASS" if P["confirmed"] else "FAIL", rr + "; plateau on both", cmp_ok("P5_geom"),
    "thm:link part 3 on all 256 points: a stable cross-objective geometric gap")
nrej = sum(1 for v in F0.values() if v["holm"]["rejected"])
add("HOLM", "[Design]", "PASS" if nrej == 9 else "FAIL", f"{nrej}/9 rejected; first step p 7.9e-68 vs 0.0056, last p 5e-05 vs 0.05",
    all(R["compare_frozen"][k]["confirmed_review"] == R["compare_frozen"][k]["confirmed_frozen"] for k in F0), "F0 family-wise")
g = R["guards"]
add("GUARDS", "[Design]", "PASS" if g["TE_ok"] and all(v["holds"] for v in g["geo"].values()) else "FAIL",
    "within T_E at N_top A 96/96, B 96/96; within Gs_max and 1-CKA 96/96 for A and B at 2^34 and 2^40", True,
    "thm:link part 4 (same objective agrees); guards do not void any test")
cal = R["calibration"]
ls = all(v["ok"] for k, v in cal.items() if k.startswith("link_self"))
lp = cal[f"link_peer_l40"]
add("CAL.link_self+link_peer+rescale", "[Design]",
    "PASS" if ls and lp["ok"] and cal["rescale_self_l34"]["ok"] and cal["rescale_self_l40"]["ok"] else "FAIL",
    f"link_self 96/96 T_E at all 8 N, |Gs_max - omega_self| max 0; link_peer at N_top: P2 {lp['P2']['x']}/96, P3a {lp['P3a']['x']}/96, "
    f"P4a {lp['P4a']['x']}/96, order Pi < 0 on the 3 confirmatory rungs, no plateau on any, geometry plateau; rescale_self max "
    f"{max(cal['rescale_self_l34']['max_abs'], cal['rescale_self_l40']['max_abs']):.1e}", R["compare_frozen"]["calibration_ok"][1],
    "the pipeline detects an exact link and an exact rescaling")
pl = R["plants"]
add("PLANTS", "[Design]", "PASS" if all(v["match"] for v in pl.values()) else "FAIL",
    "; ".join(f"{k}: broken={v['broken']} (expected {v['expected']}), dEk Pi {v['pi']['dEk']['Pi']:.4f} [{v['pi']['dEk']['ci_lo']:.4f},"
              f" {v['pi']['dEk']['ci_hi']:.4f}] vs dEk_inf {v['delta_inf']['dEk']:.4f}" for k, v in pl.items()),
    R["compare_frozen"]["plants"], "the verdict rule flags a planted break of the order part")
v = R["verdict"]
add("VERDICT", "[Model/Thm]", "PASS" if v["verdict"] == "PRESENT-order" else "FAIL",
    f"{v['verdict']}; L-within ratios " + ", ".join(f"{k} {x['ratio']:.3f}" for k, x in v["L_within"].items()),
    R["compare_frozen"]["verdict"][0] == R["compare_frozen"]["verdict"][1], "order part of the pattern present at N_top")
s = R["stop_rule"]
add("STOP_RULE", "[author decision 2026-10-09]", "PASS" if s["arm0_pattern_present"] else "FAIL",
    f"reasons {s['reasons'] or 'none'}; action: {s['action']}", R["compare_frozen"]["stop_rule"][0] == R["compare_frozen"]["stop_rule"][1],
    "arm 0 inside the premises shows the pattern; Stage B still needs a separate approval")
p7 = R["P7"]
rr = "; ".join(f"{k}: Pi {x['Pi']:.4f} [{x['ci_lo']:.4f}, {x['ci_hi']:.4f}] rho {x['rho']:.3f}" for k, x in p7["Pi"].items())
add("P7", "[Model] secondary", "PASS",
    f"log2 N_match {p7['log2N_match']} (within-A s_N(10) {p7['table_within_A_sN10']['20']:.4f} vs target 0.715: the noisiest grid point, "
    f"less noisy than S2.1); within-B {p7['within_B_sN10']:.4f}, cross {p7['cross_sN10']:.4f}; {rr}; rule: {p7['rule']}",
    R["compare_frozen"]["P7_rule"][0] == R["compare_frozen"]["P7_rule"][1],
    "frozen rule clause 2 applies (S2.1 rho_iso 3.2 vs 1.14 here); noise only approximately matched")
for l2 in ("20", "22"):
    e = R["P8"][l2]
    rr = "; ".join(f"{k}: Pi {x['Pi']:.4f} [{x['ci_lo']:.4f}, {x['ci_hi']:.4f}] rho {x['rho']:.3f}" for k, x in e["Pi"].items())
    add(f"P8.l{l2}", "[Model] secondary, descriptive", "PASS" if all(x["no_plateau"] for x in e["Pi"].values()) else "FAIL",
        f"{rr}; PSD-active A {e['frac_psd_active_A']:.2f}, A' {e['frac_psd_active_Ap']:.2f}; failed A' certificates {e['n_failed_Ap']} "
        f"(reps {e['failed_Ap_reps']}) counted against", True,
        "temperature t vs t/2 where Remark part 1 does not apply (PSD-active): no cross excess on order or scale-normalized geometry")
add("P5b.mc_rho (frozen, not recomputed)", "[Model] secondary", "FAIL",
    "MC rho at N_top vs observed: dN_k10 0.557 vs 0.973 and dMST 0.0 vs 1.111 outside the 1.5x band (dEk, dO within)", True,
    "the 48-block Monte Carlo mispredicted the noise ratio of the two discrete rungs; no theorem claim involved")
add("CERTIFICATES", "[Design] failure handling", "REPORTED",
    "39/192 arm-0 A at 2^25 (no test at 2^25) and 5/192 R- A' at 2^22 (P8, counted against) missed the 1e-6 relative BM "
    "certificate (rel lambda_min(S) 1.0e-6 to 7.8e-5; duality-gap bounds <= 9.1e-7); every other cell 0/192", True,
    "kept and flagged, never dropped (PREREG 5.4)")
with open(os.path.join(OUT, "items_table.tsv"), "w") as f:
    f.write("item\tsource\tverdict\treview_numbers\tfrozen_agrees\tmeaning\n")
    for r in rows:
        f.write("\t".join(str(r[k]) for k in ("item", "source", "verdict", "review_numbers", "frozen_agrees", "meaning")) + "\n")
pcs = R["compare_pairs"]
json.dump({"role": "s23-review-A2", "items": rows, "order_checks": {"n": n_ord, "failed": n_ord_fail},
           "pair_level_comparison": {"pair_sets": len(pcs), "pairs": 96 * len(pcs),
                                     "TE8_mismatches": sum(e.get("TE8_agree_mismatch", 0) for e in pcs.values()),
                                     "F_projection_max_abs_diff_vs_frozen": R["compare_F"]["max_abs_diff"]},
           "certificate_failures": R["certificate_failures"]},
          open(os.path.join(OUT, "review_A2.json"), "w"), indent=1, default=str)
for r in rows:
    print(r["verdict"], r["item"], "|", r["review_numbers"][:300])
