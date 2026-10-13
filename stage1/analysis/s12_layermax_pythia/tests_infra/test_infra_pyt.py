#!/usr/bin/env python
"""S0b tests of the Pythia-axis infrastructure code (role pyt-code-infra, 2026-10-10). NEW FILE (PYT-AXIS).

Covers guard.py, verdict_keys.py, compare_verdicts.py, forecast_pyt.py (P-P2b rule, section K category table, readings,
Holm consistency, end-to-end on the V-P9 stand-in fixture with E- and I-format pair files), partb_pyt.py /
partb_checks_pyt.py outputs of the vp9 stand-in run, feasibility_pyt.py (pattern table) and the F1(a) dry runs.
Amendment A1 (VERDICTS_SPEC_PYT.md section 15, lead rulings R1/R2): t16 (R1 domain table on section K, all 8
categories x both x x both P), t17 (R1 on section F; R2 comparator behaviour), t18 (l2 not run: every l2 FK key null).
Synthetic data and existing non-Pythia files only; no GPU; no Pythia byte; outputs only under
analysis/s12_layermax_pythia/s0b_scratch/PB/tests/<UTC>/ (S0b test mode, PYT_S0B_TEST=1, before CODE FROZEN).
Every test is recorded with pass/fail and details in report.json, failures included.

usage (from the project root, through the launcher):
  PYT_S0B_TEST=1 PYT_LOGDIR=analysis/s12_layermax_pythia/s0b_scratch/PB/logs analysis/s12_layermax_pythia/launch_pyt.sh \
      infra_tests_<n> 2 -- venv/bin/python analysis/s12_layermax_pythia/tests_infra/test_infra_pyt.py [--skip-e2e]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import traceback

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_pythia")
sys.path.insert(0, PROJ)
sys.path.insert(0, AXIS)
PY = os.path.join(PROJ, "venv", "bin", "python")

import guard  # noqa: E402
import verdict_keys as VK  # noqa: E402
import compare_verdicts as CV  # noqa: E402
import forecast_pyt as FC  # noqa: E402

if not guard.test_mode():
    sys.exit("run with PYT_S0B_TEST=1 before CODE FROZEN")
UTC = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
OUT = os.path.join(guard.SCRATCH, "PB", "tests", UTC)
os.makedirs(OUT, exist_ok=True)
RESULTS = []


def test(name):
    def deco(fn):
        def run():
            t0 = time.time()
            try:
                det = fn() or {}
                ok = bool(det.pop("ok", True))
                RESULTS.append({"test": name, "pass": ok, "s": round(time.time() - t0, 2), "details": det})
            except Exception as e:  # noqa: BLE001
                RESULTS.append({"test": name, "pass": False, "s": round(time.time() - t0, 2),
                                "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]})
            print(f"[{'PASS' if RESULTS[-1]['pass'] else 'FAIL'}] {name} ({RESULTS[-1]['s']} s)", flush=True)
        run.__name__ = name
        return run
    return deco


def sub(argv, env_extra=None, drop_test=False):
    env = dict(os.environ)
    if drop_test:
        env.pop("PYT_S0B_TEST", None)
    env.update(env_extra or {})
    p = subprocess.run([PY] + argv, cwd=PROJ, env=env, capture_output=True, text=True, timeout=1800)
    return p.returncode, p.stdout, p.stderr


def listing(roots):
    out = {}
    for root in roots:
        for d, _, fs in os.walk(root):
            for f in fs:
                p = os.path.join(d, f)
                st = os.lstat(p)
                out[p] = (st.st_size, st.st_mtime_ns)
    return out


# ------------------------------------------------------------------------------------------------ keys, comparator
@test("t01_verdict_keys")
def t01():
    ks = VK.required_keys("ALL")
    leg = {s: len(VK.legacy_required_keys(s)) for s in "ABC"}
    det = {"n_all": len(ks), "unique": len(ks) == len(set(ks)), "n_EI": len(VK.required_keys("EI")),
           "n_FK": len(VK.required_keys("FK")), "legacy_counts": leg, "n_comparisons": len(VK.comparisons()),
           "families": {f: sum(1 for v in VK.VITS if VK.VIT_FAMILY4[v] == f) for f in VK.FAMILIES4}}
    det["ok"] = det["unique"] and det["n_comparisons"] == 357 and det["families"] == {"in21k": 4, "MAE": 3, "DINOv2": 4,
                                                                                      "CLIP": 6} and all(leg.values())
    return det


@test("t02_compare_verdicts_names_only")
def t02():
    a = {"schema": "PYT-VERDICTS v1", "v": {"k1": 0.123456789, "k2": "secret-a", "k3": None, "k4": 1.0, "x|e": 5}}
    b = {"schema": "PYT-VERDICTS v1", "v": {"k1": 0.123456789 * (1 + 5e-13), "k2": "secret-b", "k3": None, "k4": 1.0 + 1e-9}}
    pa, pb = os.path.join(OUT, "cmp_a.json"), os.path.join(OUT, "cmp_b.json")
    json.dump(a, open(pa, "w"))
    json.dump(b, open(pb, "w"))
    rc, so, se = sub([os.path.join(AXIS, "compare_verdicts.py"), pa, pb, "--out", os.path.join(OUT, "cmp_rep.json")])
    rep = open(os.path.join(OUT, "cmp_rep.json")).read()
    leak = any(t in so + se + rep for t in ("secret-a", "secret-b", "1.000000001", "0.12345"))
    r = json.load(open(os.path.join(OUT, "cmp_rep.json")))
    # summary sub-path selection (V-P8 form)
    s = {"embedded": {"verdicts": {"E": {"k1": 1.0}}}}
    ps = os.path.join(OUT, "cmp_s.json")
    json.dump(s, open(ps, "w"))
    pc = os.path.join(OUT, "cmp_c.json")
    json.dump({"v": {"k1": 1.0}}, open(pc, "w"))
    rc2, so2, _ = sub([os.path.join(AXIS, "compare_verdicts.py"), pc, ps, "--b-sub", "embedded.verdicts.E"])
    return {"exit": rc, "mismatching_keys": r["mismatching_keys"], "value_leak": leak, "exit_sub": rc2,
            "ok": rc == 1 and sorted(r["mismatching_keys"]) == ["k2", "k4"] and not leak and rc2 == 0}


# ------------------------------------------------------------------------------------------------ guard and dry runs
@test("t03_guard_gate_cases")
def t03():
    g = os.path.join(AXIS, "guard.py")
    scr = os.path.join(guard.SCRATCH, "PB", "gate_probe")
    cases = {
        "no_env_results_early": sub([g, "check", "results/S1_2_layermax_pythia/early"], drop_test=True)[0],
        "no_env_scratch": sub([g, "check", scr], drop_test=True)[0],
        "env_scratch": sub([g, "check", scr])[0],
        "env_results_early": sub([g, "check", "results/S1_2_layermax_pythia/early"])[0],
        "no_env_bad_root": sub([g, "check", "results/S1_2_layermax_pythia/partB"], drop_test=True)[0],
        "audit_no_partB_line": sub([g, "audit"], drop_test=True)[0],
    }
    want = {"no_env_results_early": 3, "no_env_scratch": 3, "env_scratch": 0, "env_results_early": 3,
            "no_env_bad_root": 3, "audit_no_partB_line": 3}
    return {"exit_codes": cases, "want": want, "ok": cases == want}


@test("t04_guard_scan")
def t04():
    rc, so, se = sub([os.path.join(AXIS, "guard.py"), "scan"], drop_test=True)
    rep = json.loads(so)
    return {"exit": rc, "clean": rep["clean"], "hits": rep["find_hits"][:5], "top": rep["results_top_level"],
            "procs": rep["forbidden_processes"][:5], "find_cmd": rep["find_cmd"],
            "ok": rc in (0, 4) and rep["find_hits"] == []}


@test("t05_dry_runs_exit3_no_output")
def t05():
    roots = [os.path.join(PROJ, "results", "S1_2_layermax_pythia"), os.path.join(PROJ, "features", "wit1024_pythia_axis"),
             guard.SCRATCH]
    cmds = {"forecast_E": [os.path.join(AXIS, "forecast_pyt.py"), "--impl", "E"],
            "forecast_I": [os.path.join(AXIS, "forecast_pyt.py"), "--impl", "I"],
            "partb_pyt_real": [os.path.join(AXIS, "partb_pyt.py")],
            "partb_checks_real": [os.path.join(AXIS, "partb_checks_pyt.py")]}
    res = {}
    for k, argv in cmds.items():
        before = listing(roots)
        rc, so, se = sub(argv, drop_test=True)
        after = listing(roots)
        ch = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
        res[k] = {"exit": rc, "n_changed": len(ch), "changed": ch[:5], "stdout_bytes": len(so),
                  "stderr_tail": se.strip().splitlines()[-1:] if se else []}
    return {"runs": res, "ok": all(v["exit"] == 3 and v["n_changed"] == 0 for v in res.values())}


# ------------------------------------------------------------------------------------------------ P-P2b rule
@test("t06_forecast_rule_hand_cases")
def t06():
    S = FC.s_cal
    band = (0.001, 0.003)
    cases = []
    # j (smaller) clearly above j' (larger): j' s_cal max < j s_cal min -> below
    cases.append(("below", FC.forecast_scal(0.50, 0.40, 0.10, 0.12, band, band)))
    # j' clearly above j -> above
    cases.append(("above", FC.forecast_scal(0.40, 0.50, 0.12, 0.10, band, band)))
    # overlapping intervals -> undecided (T equal-ish, offsets overlap)
    cases.append(("undecided", FC.forecast_scal(0.300, 0.3005, 0.10, 0.10, band, band)))
    # both censored at every offset -> tie
    cases.append(("tie", FC.forecast_scal(0.100, 0.1005, 0.10, 0.10, band, band)))
    # j censored at the high end only, j' never censored but small -> undecided (plan: censorable side, unordered)
    cases.append(("undecided", FC.forecast_scal(0.1020, 0.1030, 0.10, 0.10, band, band)))
    # j' censored at every offset, j strictly positive at every offset -> below
    cases.append(("below", FC.forecast_scal(0.20, 0.1005, 0.10, 0.10, band, band)))
    # per-model bands: same T, larger model's band higher -> below
    cases.append(("below", FC.forecast_scal(0.30, 0.30, 0.10, 0.10, (0.001, 0.002), (0.003, 0.004))))
    # boundary: equality of interval ends is not strict -> undecided
    Tj, Bj, Bj2 = 0.30, 0.10, 0.10
    hi_j2_target = S(Tj, Bj + 0.003)  # make S(Tj2; Bj2+0.001) equal to S(Tj; Bj+0.003)
    Tj2 = hi_j2_target * (1 - (Bj2 + 0.001)) + (Bj2 + 0.001)
    cases.append(("undecided-or-below(fp)", FC.forecast_scal(Tj, Tj2, Bj, Bj2, band, band)))
    cases.append(("below", FC.forecast_unc(0.50, 0.40, 0.10, 0.10, (-0.0003, 0.0008))))
    cases.append(("above", FC.forecast_unc(0.40, 0.50, 0.10, 0.10, (-0.0003, 0.0008))))
    cases.append(("undecided", FC.forecast_unc(0.40, 0.4005, 0.10, 0.10, (-0.0003, 0.0008))))
    ok = all(w == g for w, g in cases if not w.startswith("undecided-or"))
    ok = ok and cases[7][1] in ("undecided", "below")
    vb = [FC.verdict_pp2b(29, 0), FC.verdict_pp2b(30, 0), FC.verdict_pp2b(5, 1), FC.verdict_pp2b(100, 1)]
    ok = ok and vb == ["not evaluable", "holds", "fails", "fails"]
    return {"cases": cases, "verdicts": vb, "ok": ok}


# ------------------------------------------------------------------------------------------------ section 7 table
@test("t07_category_table_every_row")
def t07():
    C = FC.category
    rows = {
        "E": C(False, -0.5, 0.001, 0.001, None, "holds", "holds", 0.9, 0.1),
        "D": C(True, 0.10, 0.9, 0.9, False, "holds", "holds", 0.9, 0.95),
        "C_pos": C(True, 0.0999, 0.9, 0.9, False, "holds", "holds", 0.9, 0.95),
        "C_neg": C(True, -0.0999, 0.9, 0.9, False, "holds", "holds", 0.9, 0.85),
        "F_perm": C(True, -0.5, 0.06, 0.001, False, "holds", "holds", 0.9, 0.4),
        "F_mag": C(True, -0.5, 0.001, 0.06, False, "holds", "holds", 0.9, 0.4),
        "F_both": C(True, -0.10, 0.5, 0.5, False, "holds", "holds", 0.9, 0.8),
        "B_pp2a": C(True, -0.5, 0.001, 0.001, True, "fails", "holds", 0.9, 0.4),
        "B_pp2b": C(True, -0.5, 0.001, 0.001, True, "holds", "fails", 0.9, 0.4),
        "A_verified": C(True, -0.6, 0.001, 0.001, True, "holds", "holds", 0.9, 0.30),
        "A_boundary_C040": C(True, -0.6, 0.001, 0.001, True, "holds", "holds", 0.60, 0.40),
        "A_att": C(True, -0.2, 0.001, 0.001, True, "holds", "holds", 0.95, 0.75),
        "A_prime": C(True, -1.2, 0.001, 0.001, True, "holds", "holds", 0.30, -0.9),
        "A_unverified": C(True, -0.6, 0.001, 0.001, True, "not evaluable", "holds", 0.9, 0.3),
        "A_unverified_b": C(True, -0.6, 0.001, 0.001, True, "holds", "not evaluable", 0.9, 0.3),
    }
    want = {"E": ("E", None, None), "D": ("D", None, None), "C_pos": ("C", None, None), "C_neg": ("C", None, None),
            "F_perm": ("F", None, "not consistent across the seven sizes"),
            "F_mag": ("F", None, "size of the drop not established"),
            "F_both": ("F", None, "not consistent across the seven sizes; size of the drop not established"),
            "B_pp2a": ("B", None, None), "B_pp2b": ("B", None, None), "A_verified": ("A", "verified", None),
            "A_boundary_C040": ("A", "verified", None), "A_att": ("A-att", "verified", None),
            "A_prime": ("A-prime", "verified", None), "A_unverified": ("A", "unverified", None),
            "A_unverified_b": ("A", "unverified", None)}
    got = {k: list(v) for k, v in rows.items()}
    reached = sorted({v[0] for v in rows.values()})
    return {"got": got, "reached": reached,
            "ok": all(tuple(rows[k]) == want[k] for k in want) and set(reached) == set(VK.CATEGORIES)}


@test("t08_readings_holm_consistency")
def t08():
    r34 = [FC.reading_h34(0.15, False, True, True), FC.reading_h34(0.0, True, False, None),
           FC.reading_h34(0.05, False, False, None), FC.reading_h34(0.05, False, True, True),
           FC.reading_h34(0.05, False, True, False)]
    w34 = ["moved", "no comparison changed order", "kept, not tested (gate closed)", "equivalence confirmed",
           "kept, equivalence not established"]
    r2 = [FC.reading_h2("fails", "H2a", True, True, -0.5), FC.reading_h2("holds", "H2a", True, True, -0.5),
          FC.reading_h2("holds", "H2b", True, True, 0.01), FC.reading_h2("holds", "H2a", True, False, -0.5),
          FC.reading_h2("not evaluable", "H2b", False, None, 0.2), FC.reading_h2("holds", "H2b", True, False, 0.05)]
    w2 = ["premise failed", "follows the late baselines", "kept as predicted", "drop not confirmed",
          "moved with the late baselines (not tested, gate closed)", "kept"]
    # Holm hand cases: p = (0.01, 0.02, 0.04) -> thresholds 0.0167, 0.025, 0.05 -> all reject; adj (0.03, 0.04, 0.04)
    h1 = FC.holm({"H2": 0.04, "H3": 0.01, "H4": 0.02})
    # p = (0.02, 0.02, 0.5): 0.02 > 0.0167 -> none
    h2 = FC.holm({"H2": 0.02, "H3": 0.02, "H4": 0.5})
    # tie order H2 before H3
    h3 = FC.holm({"H2": 1 / 2001, "H3": 1 / 2001, "H4": 0.03})
    ok_h = (h1["order"] == "H3,H4,H2" and all(h1["reject"].values()) and abs(h1["adj"]["H3"] - 0.03) < 1e-15
            and abs(h1["adj"]["H4"] - 0.04) < 1e-15 and abs(h1["adj"]["H2"] - 0.04) < 1e-15
            and not any(h2["reject"].values()) and h3["order"] == "H2,H3,H4" and all(h3["reject"].values()))
    va = consistent_va("prh_clip+l2", "s_cal")
    c1 = FC.h_consistent(va, "H|prh_clip+l2|s_cal")
    va_bad = dict(va)
    va_bad["H|prh_clip+l2|s_cal|H1|confirmed"] = False
    c2 = FC.h_consistent(va_bad, "H|prh_clip+l2|s_cal")
    return {"r34": r34, "r2": r2, "holm": [h1, h2, h3], "consistent": c1, "inconsistent_detected": not c2,
            "ok": r34 == w34 and r2 == w2 and ok_h and c1 and not c2}


def consistent_va(pre, x, member=True, D=-0.5, p_perm=0.001, p_mag=0.002, branch="H2b", tost_d2=0.01, h4_d=0.02):
    """A hand-made aggregate H block that satisfies the plan's definitions (fixture for K).
    h4_d (A1 test t16): the H4 difference; |h4_d| >= 0.10 gives the reading "moved" and the TOST p 1.0."""
    h = f"H|{pre}|{x}"
    v = {f"{h}|H1|member": member, f"{h}|H1|D": D, f"{h}|H1|p_perm": p_perm, f"{h}|H1|p_mag": p_mag,
         f"{h}|H1|p": max(p_perm, p_mag), f"{h}|H1|nV": 17, f"{h}|H1|R": 0.95, f"{h}|H1|C": 0.95 + D,
         f"{h}|H1|kedge": False}
    conf = (max(p_perm, p_mag) <= 0.05 and D <= -0.10) if member else None
    v[f"{h}|H1|confirmed"] = conf
    gate = "open (H1 confirmed)" if member and conf else ("open (H1 not a member)" if not member else "closed")
    v[f"{h}|gate"] = gate
    v[f"{h}|H2|branch"] = branch
    v[f"{h}|H2|D"] = -0.3 if branch == "H2a" else tost_d2
    v[f"{h}|H2|p_perm"], v[f"{h}|H2|p_mag"] = 0.002, 0.003
    v[f"{h}|H2|p_lo"], v[f"{h}|H2|p_hi"] = 1 / 2001, 2 / 2001
    v[f"TOST|{pre}|{x}|cka_biased|ctm|D"] = tost_d2
    v[f"{h}|H2|p"] = 0.003 if branch == "H2a" else (2 / 2001 if abs(tost_d2) < 0.1 else 1.0)
    for hy, d, deg in (("H3", 0.0, True), ("H4", h4_d, False)):
        v[f"{h}|{hy}|D"], v[f"{h}|{hy}|degenerate"] = d, deg
        v[f"{h}|{hy}|p"] = 1 / 2001 if deg else (0.004 if abs(d) < 0.10 else 1.0)
    tested = gate != "closed"
    ps = {hy: v[f"{h}|{hy}|p"] for hy in VK.HYPS}
    H = FC.holm(ps)
    v[f"{h}|holm_order"] = H["order"] if tested else None
    for hy in VK.HYPS:
        v[f"{h}|{hy}|tested"] = tested
        v[f"{h}|{hy}|p_holm"] = H["adj"][hy] if tested else None
        v[f"{h}|{hy}|holm_reject"] = H["reject"][hy] if tested else None
        cf = (H["reject"][hy] and v[f"{h}|H2|D"] <= -0.10) if (hy == "H2" and branch == "H2a") else H["reject"][hy]
        v[f"{h}|{hy}|confirmed"] = cf if tested else None
    return v


# ------------------------------------------------------------------------------------------------ pattern table
@test("t09_pattern_table_exact")
def t09():
    import feasibility_pyt as FE
    pt = FE.pattern_table()
    return {"table": {k: (d["count"], round(d["dTr"], 4)) for k, d in pt.items()},
            "ok": all(d["equal_plan"] for d in pt.values())}


# ------------------------------------------------------------------------------------------------ Part B outputs (vp9)
PB_DIR = os.path.join(guard.SCRATCH, "PB", "partB_vp9")


@test("t10_partb_vp9_outputs")
def t10():
    F = json.load(open(os.path.join(PB_DIR, "partB_features_pyt.json")))
    req = ["schema", "m", "delta_b", "sizes", "pyt_lms", "vits", "vit_family4", "num_params", "width", "depth_blocks",
           "L_stored", "band_constants", "mu_band", "per_pre"]
    reqp = ["clip_c", "PR", "PR_star", "u_star", "u_star_rel", "Bbar", "Bbar_argmax", "Bfd", "Blate", "gaps", "Tr_Bbar",
            "Tr_Bbar_5size", "Tr_Bfd", "Tr_Blate", "materiality", "material_Bbar", "material_Blate", "material_curve_db",
            "s", "s_flag_zero", "s_late", "s_late_flag_zero", "members", "H2_branch", "Blate_equals_Bbar",
            "n_Blate_equals_Bbar", "H2_same_as_H1", "band", "band_h_curve", "band_flat", "extrapolated", "qualifying"]
    miss = [k for k in req if k not in F] + [f"{p}.{k}" for p in F["per_pre"] for k in reqp if k not in F["per_pre"][p]]
    bad_names = [n for n in os.listdir(PB_DIR) if "EleutherAI__pythia-" in n]
    Q = F["per_pre"]["prh_clip+l2"]
    det = {"missing_fields": miss, "test_mode": F["meta"]["test_mode"], "recheck": F["meta"]["recheck"],
           "members": Q["members"], "s": Q["s"], "Tr_Bbar": Q["Tr_Bbar"], "Tr_Blate": Q["Tr_Blate"],
           "PR_star": [round(Q["PR_star"][k], 2) for k in F["pyt_lms"]], "extrapolated": Q["extrapolated"],
           "qualifying": Q["qualifying"], "n_Blate_equals_Bbar": Q["n_Blate_equals_Bbar"],
           "retro_in_band": Q["B5"]["offset_retrodiction_187"]["n_in_band"],
           "refit": Q["B5"]["offset_retrodiction_187"]["refit"], "files_named_like_pythia": bad_names}
    det["ok"] = not miss and F["meta"]["test_mode"] and F["meta"]["recheck"]["pass"] and not bad_names
    return det


# ------------------------------------------------------------------------------------------------ forecast end to end
def build_fixture(root: str) -> dict:
    """V-P9 stand-in pair roots: synth__pyt-<size> -> known E json / I npz (symlinks; relabelled known files)."""
    import partb_pyt as PB
    src = PB.STANDIN_VP9
    e_dir = {"wit1024": os.path.join(PROJ, "results", "S1_2_layermax", "pairs"),
             "wit1024_llama_axis": os.path.join(PROJ, "results", "S1_2_layermax_llama", "early", "pairs")}
    i_dir = {"wit1024": os.path.join(PROJ, "results", "S1_2_layermax", "indep", "{pre}", "pairs"),
             "wit1024_llama_axis": os.path.join(PROJ, "results", "S1_2_layermax_llama", "indep", "{pre}", "pairs")}
    n = 0
    for pre in VK.PRES:
        for s in VK.SIZES:
            stem, ds = src[s]
            for v in VK.VITS:
                for impl, d, ext in (("E", os.path.join(root, "early", "pairs", pre), ".json"),
                                     ("I", os.path.join(root, "indep", pre, "pairs"), ".npz")):
                    os.makedirs(d, exist_ok=True)
                    tgt = (os.path.join(e_dir[ds], pre, f"{stem}__{v}.json") if impl == "E"
                           else os.path.join(i_dir[ds].format(pre=pre), f"{stem}__{v}.npz"))
                    lnk = os.path.join(d, f"synth__pyt-{s}__{v}{ext}")
                    if not os.path.lexists(lnk):
                        os.symlink(tgt, lnk)
                    n += 1
    return {"n_links": n}


def full_va(F: dict) -> dict:
    """Hand-made aggregate file for the fixture: the H, C, R, G inputs section K reads (values hand-set, consistent)."""
    v = {"G|l2_run": True}
    for pre in VK.PRES:
        Q = F["per_pre"][pre]
        for x in VK.XS:
            v.update(consistent_va(pre, x, member=bool(Q["material_Bbar"]), branch=Q["H2_branch"]))
            v[f"PP2a|{pre}|{x}|verdict"] = "holds"
            v[f"PP2ai|{pre}|{x}|failing"] = ""
            v[f"PP2aii|{pre}|{x}|verdict"] = "holds"
            v[f"PP5a|{pre}|{x}|verdict"] = "holds"
            for s in VK.SIZES:
                v[f"PP2ai|{pre}|{x}|in_band|{s}"] = True
            r = f"R|{pre}|{x}"
            v.update({f"{r}|fragile": False, f"{r}|fragile_sizes": "", f"{r}|grid5|D1": -0.4, f"{r}|grid5|dtr_mknn": 0.0,
                      f"{r}|grid5|dtr_dcka": 0.02, f"{r}|five|carried": False, f"{r}|five|carried_D": False})
        v[f"R|{pre}|n_scal_zero"] = 3
        for s in VK.SIZES:
            v[f"R|{pre}|median_scal|{s}"] = 0.1
    return v


@test("t11_forecast_e2e_E_and_I")
def t11():
    root = os.path.join(guard.SCRATCH, "PB", "fixture_vp9")
    fx = build_fixture(root)
    F = json.load(open(os.path.join(PB_DIR, "partB_features_pyt.json")))
    va = full_va(F)
    res = {"fixture": fx}
    outs = {}
    for impl in ("E", "I"):
        pv = os.path.join(OUT, f"va_{impl}.json")
        json.dump({"schema": "PYT-VERDICTS v1", "impl": f"fixture-{impl}", "inputs": {}, "v": va}, open(pv, "w"))
        od = os.path.join(OUT, "forecast", impl)
        rc, so, se = sub([os.path.join(AXIS, "forecast_pyt.py"), "--impl", impl, "--test-root", root, "--partb",
                          os.path.join(PB_DIR, "partB_features_pyt.json"), "--verdicts", pv, "--out", od])
        res[impl] = {"exit": rc, "stdout": so.strip().splitlines()[-5:], "stderr_tail": se.strip().splitlines()[-3:]}
        outs[impl] = os.path.join(od, "verdicts_forecast.json")
    rc, so, se = sub([os.path.join(AXIS, "compare_verdicts.py"), outs["E"], outs["I"], "--sections", "FK",
                      "--out", os.path.join(OUT, "cmp_forecast_E_vs_I.json")])
    res["compare_FK_E_vs_I"] = json.loads(so.strip().splitlines()[-1]) if so.strip() else {"exit": rc}
    VE = json.load(open(outs["E"]))["v"]
    req = VK.required_keys("FK")
    res["n_required"], res["n_missing"] = len(req), sum(1 for k in req if k not in VE)
    res["primary"] = {k: VE[k] for k in ("PP2b|prh_clip+l2|s_cal|n_decided", "PP2b|prh_clip+l2|s_cal|x_correct",
                                         "PP2b|prh_clip+l2|s_cal|n_undecided", "PP2b|prh_clip+l2|s_cal|n_tie_fc",
                                         "PP2b|prh_clip+l2|s_cal|verdict", "PP2b|prh_clip+l2|s_cal|wrong_sizes",
                                         "PP2b|prh_clip+l2|s_cal|flat|n_decided", "PP2b|prh_clip+l2|s_cal|flat|x_correct",
                                         "K|prh_clip+l2|s_cal|category", "K|prh_clip+l2|s_cal|premise",
                                         "K|prh_clip+l2|s_cal|h_consistent", "K|prh_clip+l2|s_cal|H3_reading",
                                         "K|prh_clip+l2|s_cal|H2_reading")}
    res["label"] = "STAND-IN (BLOOMZ 5 + llama-13b + llama-30b relabelled); hand-made aggregate inputs; not a result"
    res["ok"] = (res["E"]["exit"] == 0 and res["I"]["exit"] == 0 and res["compare_FK_E_vs_I"].get("pass") is True
                 and res["n_missing"] == 0 and VE["K|prh_clip+l2|s_cal|h_consistent"] is True)
    return res


@test("t12_partb_checks_vp9")
def t12():
    rc, so, se = sub([os.path.join(AXIS, "partb_checks_pyt.py")])
    rep = json.load(open(os.path.join(PB_DIR, "partB_retro_checks_pyt.json")))
    k = {pre: {"K1": rep["per_pre"][pre]["K1"]["pass"], "K3": [rep["per_pre"][pre]["K3"]["n_in_band"],
                                                             rep["per_pre"][pre]["K3"]["refit_diff_gt_1e-6"]],
               "K4": {f: (d["x_correct"], d["n_decided"], d["n_undecided"]) for f, d in rep["per_pre"][pre]["K4"].items()
                      if f != "label"}} for pre in VK.PRES}
    return {"exit": rc, "summary": k, "K2": rep["K2"]["pass"], "stdout_tail": so.strip().splitlines()[-3:],
            "ok": rc == 0 and rep["pass"]}


# ------------------------------------------------------------------------------------------------ S0b tools
MINE = ["guard.py", "verdict_keys.py", "compare_verdicts.py", "forecast_pyt.py", "partb_pyt.py", "partb_checks_pyt.py",
        "feasibility_pyt.py", "f1d_outroots.py", "s0b_freeze.py", "tests_infra/test_infra_pyt.py"]


@test("t13_feasibility_test_mode")
def t13():
    rc, so, se = sub([os.path.join(AXIS, "feasibility_pyt.py")])
    fj = json.load(open(os.path.join(guard.SCRATCH, "PB", "feasibility", "feasibility_pyt.json")))
    diffs = [r for r in fj["plan_vs_server"] if not r["equal_at_plan_precision"]]
    s0 = fj["section0"]
    return {"exit": rc, "n_items": len(fj["plan_vs_server"]), "n_diff": len(diffs), "diffs": diffs,
            "fit": {p: {"a": s0[p]["fit11"]["a"], "b": s0[p]["fit11"]["b"], "abs_diff_a": s0[p]["abs_diff_a"],
                        "abs_diff_b": s0[p]["abs_diff_b"]} for p in s0},
            "ok": rc == 0 and all(s0[p]["abs_diff_a"] <= 1e-6 and s0[p]["abs_diff_b"] <= 1e-6 for p in s0)}


@test("t14_f1d_self_test_on_own_files")
def t14():
    out = os.path.join(OUT, "f1d_self_test.json")
    rc, so, se = sub([os.path.join(AXIS, "f1d_outroots.py"), "--only"] + [os.path.join(AXIS, f) for f in MINE] +
                     ["--modules", "extract pyt_axis_common", "--out", out], drop_test=True)
    rep = json.load(open(out))
    viol = rep["violations"]
    return {"exit": rc, "n_files": len(rep["files"]), "imported": rep["imported"], "violations": viol[:20],
            "ok": rc == 6 and not viol}


@test("t15_s0b_freeze_functions")
def t15():
    import s0b_freeze as SF
    inv, apple = SF.inventory()
    orig = SF.check_originals(inv)
    mine_bad = [k for k in orig["bad"] if not k.startswith(("early/", "indep/", "review/", "../../extract/"))
                and k != "launch_ol.sh"]
    spec = [["forecast_E", "analysis/s12_layermax_pythia/forecast_pyt.py", "--impl", "E"],
            ["forecast_I", "analysis/s12_layermax_pythia/forecast_pyt.py", "--impl", "I"],
            ["partb_pyt", "analysis/s12_layermax_pythia/partb_pyt.py"],
            ["partb_checks_pyt", "analysis/s12_layermax_pythia/partb_checks_pyt.py"],
            ["guard_check_early", "analysis/s12_layermax_pythia/guard.py", "check", "results/S1_2_layermax_pythia/early"],
            ["guard_check_indep", "analysis/s12_layermax_pythia/guard.py", "check", "results/S1_2_layermax_pythia/indep"],
            ["guard_check_review", "analysis/s12_layermax_pythia/guard.py", "check", "results/S1_2_layermax_pythia/review"],
            ["guard_check_forecast", "analysis/s12_layermax_pythia/guard.py", "check", "results/S1_2_layermax_pythia/forecast"],
            ["guard_check_cache", "analysis/s12_layermax_pythia/guard.py", "check", "results/S1_2_layermax_pythia/cache"]]
    json.dump(spec, open(os.path.join(OUT, "dryruns_infra.json"), "w"), indent=1)
    saved = os.environ.pop("PYT_S0B_TEST")
    try:
        dr = SF.dry_runs(spec)
    finally:
        os.environ["PYT_S0B_TEST"] = saved
    rec = os.path.join(OUT, "rec_pass.json")
    json.dump({"pass": True}, open(rec, "w"))
    pr = SF.pass_records([rec])
    return {"n_inventory": len(inv), "n_appledouble": len(apple), "appledouble": apple,
            "originals_bad_all": orig["bad"], "originals_bad_mine": mine_bad,
            "dry_runs": [(r["label"], r["exit"], r["n_files_changed"]) for r in dr], "pass_records": pr,
            "ok": not mine_bad and all(r["exit"] == 3 and r["n_files_changed"] == 0 for r in dr) and pr[0]["pass_"]}


# ------------------------------------------------------------------------------------------------ Amendment A1 (R1, R2)
A_FAM = ("A", "A-att", "A-prime")
K_FIELDS = sorted({k.split("|", 3)[3] for k in VK.required_keys("K") if k.startswith("K|prh_clip+l2|s_cal|")})


def k_case(pre, x, cat):
    """Hand-made consistent section-K inputs that reach category `cat` (VERDICTS_SPEC_PYT section 15, test t16)."""
    D = {"E": -0.5, "D": 0.2, "C": 0.0, "F": -0.5, "B": -0.5, "A": -0.6, "A-att": -0.2, "A-prime": -0.5}[cat]
    va = consistent_va(pre, x, member=(cat != "E"), D=D, p_perm=(0.2 if cat == "F" else 0.001),
                       h4_d=(0.15 if cat in ("C", "A") else 0.02))  # H4 reading "moved" in two categories
    h = f"H|{pre}|{x}"
    if cat == "A-prime":
        va[f"{h}|H1|R"], va[f"{h}|H1|C"] = 0.5, 0.0
    va[f"PP2a|{pre}|{x}|verdict"] = "fails" if cat == "B" else "holds"
    va[f"PP2ai|{pre}|{x}|failing"] = "70m" if cat == "B" else ""
    va[f"PP2aii|{pre}|{x}|verdict"] = "holds"
    va[f"PP5a|{pre}|{x}|verdict"] = "holds"
    for s in VK.SIZES:
        va[f"PP2ai|{pre}|{x}|in_band|{s}"] = True
        va[f"R|{pre}|median_scal|{s}"] = 0.0 if s == "70m" else 0.1
    r = f"R|{pre}|{x}"
    va.update({f"{r}|fragile": False, f"{r}|fragile_sizes": "", f"{r}|grid5|D1": -0.4, f"{r}|grid5|dtr_mknn": 0.0,
               f"{r}|grid5|dtr_dcka": 0.02, f"{r}|five|carried": False, f"{r}|five|carried_D": False,
               f"R|{pre}|n_scal_zero": 50})
    vf = {f"PP2b|{pre}|{x}|verdict": "holds", f"PP2b|{pre}|{x}|wrong_all_extrap": None}
    Q = {"extrapolated": {s: s in ("70m", "160m") for s in VK.SIZES}, "H2_same_as_H1": False,
         "material_curve_db": {"Bbar": {FC.fmt(d): True for d in VK.DBS}}}
    return va, vf, Q


@test("t16_R1_domains_section_K")
def t16():
    """A1 R1: every restricted K key is null exactly outside its domain (section 15.1 table); every other K key is
    written in every variant. All 8 categories x both x x both P, hand-made consistent inputs (no data)."""
    bad, reached, n_checked = [], {}, 0
    for pre in VK.PRES:
        for x in VK.XS:
            for cat in VK.CATEGORIES:
                va, vf, Q = k_case(pre, x, cat)
                K = FC.section_K(pre, x, va, vf, Q, list(VK.SIZES))
                c = f"K|{pre}|{x}"
                want_keys = {k for k in VK.required_keys("K") if k.startswith(c + "|")}
                if set(K) != want_keys:
                    bad.append({"case": [pre, x, cat], "key_set_differs": sorted(set(K) ^ want_keys)[:5]})
                    continue
                got = K[f"{c}|category"]
                reached.setdefault(f"{pre}|{x}", []).append(got)
                if got != cat or K[f"{c}|h_consistent"] is not True:
                    bad.append({"case": [pre, x, cat], "category": got, "h_consistent": K[f"{c}|h_consistent"]})
                dom = {"M4": x == "s_cal", "M4_sizes": x == "s_cal", "M4_n_zero": x == "s_cal",
                       "M5_A": cat in A_FAM, "M5_A_sizes": cat in A_FAM, "M5_B": cat == "B", "M2": cat in A_FAM,
                       "premise": cat in A_FAM, "F_reason": cat == "F",
                       "M2_H3": K[f"{c}|H3_reading"] != "moved", "M2_H4": K[f"{c}|H4_reading"] != "moved"}
                for k in sorted(want_keys):
                    f = k[len(c) + 1:]
                    defined = dom.get(f, True)  # unrestricted keys (and the curves) must be written in every variant
                    n_checked += 1
                    if (K[k] is None) == defined:
                        bad.append({"key": k, "category": cat, "want": "value" if defined else "null",
                                    "got_null": K[k] is None})
                if x == "s_cal" and not (isinstance(K[f"{c}|M4"], bool) and isinstance(K[f"{c}|M4_n_zero"], int)
                                         and isinstance(K[f"{c}|M4_sizes"], str)):
                    bad.append({"case": [pre, x, cat], "M4_types": "wrong"})
                if cat == "C" and K[f"{c}|H4_reading"] != "moved":
                    bad.append({"case": [pre, x, cat], "H4_reading_not_moved": K[f"{c}|H4_reading"]})
    return {"n_cases": 2 * 2 * len(VK.CATEGORIES), "n_key_checks": n_checked, "reached": reached,
            "violations": bad[:40], "n_violations": len(bad), "fields": K_FIELDS, "ok": not bad}


@test("t17_R1_domains_section_F_and_R2_comparator")
def t17():
    """A1 R1 on section F (ok null iff undecided; wrong_all_extrap null iff n_wrong = 0; unc: no forecast tie,
    n_tie_fc = 0 not null) on synthetic numbers; A1 R2: compare_verdicts counts null vs bool on G|clip_equal_partB."""
    sizes, vits = list(VK.SIZES), list(VK.VITS)
    pyt = [f"synth__pyt-{s}" for s in sizes]
    bad, det = [], {}
    for case in ("all_correct", "wrong_and_undecided"):
        T = {s: 0.30 + 0.05 * i for i, s in enumerate(sizes)}
        if case == "wrong_and_undecided":
            T["1b"] = T["410m"] + 1e-5  # overlapping intervals -> undecided
        Q = {"extrapolated": {s: False for s in sizes}, "Bbar": {lm: {v: 0.10 for v in vits} for lm in pyt},
             "band": {s: {"lo": 0.001, "hi": 0.002} for s in sizes}, "mu_band": [-0.0003, 0.0008]}
        vals = {}
        for s in sizes:
            sc = FC.s_cal(T[s], 0.1015)
            if case == "wrong_and_undecided" and s == "6.9b":
                sc = 0.0  # observed below every smaller model -> decided forecasts "above" are wrong
            for v in vits:
                vals[(s, v)] = {"T": T[s], "s_cal": sc, "unc": T[s] - 0.1003}
        for x in VK.XS:
            V, _ = FC.section_F("prh_clip+l2", x, vals, Q, sizes, pyt, vits)
            p = f"PP2b|prh_clip+l2|{x}"
            n_ok_bad = sum(1 for a, b, v in VK.comparisons()
                           if (V[f"{p}|ok|{a}|{b}|{v}"] is None) != (V[f"{p}|dec|{a}|{b}|{v}"] == "undecided"))
            nw = V[f"{p}|n_wrong"]
            wae = V[f"{p}|wrong_all_extrap"]
            other_null = [k for k, val in V.items() if val is None and "|ok|" not in k and not k.endswith("wrong_all_extrap")]
            det[f"{case}|{x}"] = {"n_decided": V[f"{p}|n_decided"], "n_undecided": V[f"{p}|n_undecided"], "n_wrong": nw,
                                  "wrong_all_extrap_null": wae is None, "n_tie_fc": V[f"{p}|n_tie_fc"],
                                  "ok_null_rule_violations": n_ok_bad, "other_null_keys": other_null[:5]}
            if n_ok_bad or (wae is None) != (nw == 0) or other_null:
                bad.append(f"{case}|{x}")
            if x == "unc" and (V[f"{p}|n_tie_fc"] != 0 or any(V[f"{p}|dec|{a}|{b}|{v}"] == "tie"
                                                              for a, b, v in VK.comparisons())):
                bad.append(f"{case}|unc ties")
        if case == "wrong_and_undecided" and not (det[f"{case}|s_cal"]["n_wrong"] > 0 and det[f"{case}|s_cal"]["n_undecided"] > 0):
            bad.append(f"{case}: fixture did not reach wrong and undecided")
    pa, pb, pn = (os.path.join(OUT, f"r2_{t}.json") for t in ("bool", "null", "null2"))
    json.dump({"v": {"G|clip_equal_partB": True}}, open(pa, "w"))
    json.dump({"v": {"G|clip_equal_partB": None}}, open(pb, "w"))
    json.dump({"v": {"G|clip_equal_partB": None}}, open(pn, "w"))
    r_bool_null = CV.compare({"G|clip_equal_partB": True}, {"G|clip_equal_partB": None}, ["G|clip_equal_partB"])
    rc1, _, _ = sub([os.path.join(AXIS, "compare_verdicts.py"), pa, pb])
    rc2, _, _ = sub([os.path.join(AXIS, "compare_verdicts.py"), pb, pn])
    det["R2"] = {"bool_vs_null_mismatching": r_bool_null["mismatching_keys"], "exit_bool_vs_null": rc1,
                 "exit_null_vs_null": rc2,
                 "note": "null vs null passes the comparator; in the real run R's own V-P1 (bool) against I/E catches it"}
    if r_bool_null["mismatching_keys"] != ["G|clip_equal_partB"] or rc1 != 1 or rc2 != 0:
        bad.append("R2 comparator")
    return {"details": det, "failed": bad, "ok": not bad}


@test("t18_R1_l2_not_run_e2e")
def t18():
    """A1 R1 / section 9: with G|l2_run false every FK key with P = l2 is null and the prh_clip+l2 keys are as in t11."""
    root = os.path.join(guard.SCRATCH, "PB", "fixture_vp9")
    build_fixture(root)
    F = json.load(open(os.path.join(PB_DIR, "partB_features_pyt.json")))
    va = full_va(F)
    va["G|l2_run"] = False
    pv = os.path.join(OUT, "va_E_nol2.json")
    json.dump({"schema": "PYT-VERDICTS v1", "impl": "fixture-E-nol2", "inputs": {}, "v": va}, open(pv, "w"))
    od = os.path.join(OUT, "forecast_nol2", "E")
    rc, so, se = sub([os.path.join(AXIS, "forecast_pyt.py"), "--impl", "E", "--test-root", root, "--partb",
                      os.path.join(PB_DIR, "partB_features_pyt.json"), "--verdicts", pv, "--out", od])
    V = json.load(open(os.path.join(od, "verdicts_forecast.json")))["v"]
    req = VK.required_keys("FK")
    l2_nonnull = [k for k in req if k.split("|")[1] == "l2" and V.get(k) is not None]
    missing = [k for k in req if k not in V]
    full_E = os.path.join(OUT, "forecast", "E", "verdicts_forecast.json")
    same_prh = None
    if os.path.isfile(full_E):
        VE = json.load(open(full_E))["v"]
        prh = [k for k in req if k.split("|")[1] == "prh_clip+l2"]
        same_prh = CV.compare(VE, V, prh)["n_mismatch"]
    return {"exit": rc, "n_l2_keys": sum(1 for k in req if k.split("|")[1] == "l2"), "n_l2_nonnull": len(l2_nonnull),
            "l2_nonnull": l2_nonnull[:5], "n_missing": len(missing), "prh_mismatch_vs_t11": same_prh,
            "stderr_tail": se.strip().splitlines()[-2:],
            "ok": rc == 0 and not l2_nonnull and not missing and same_prh == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-e2e", action="store_true", help="skip the tests that need the vp9 Part B outputs")
    args = ap.parse_args()
    tests = [t01, t02, t03, t04, t05, t06, t07, t08, t09, t13, t14, t15, t16, t17]
    if not args.skip_e2e:
        tests += [t10, t12, t11, t18]
    for t in tests:
        t()
    rep = {"utc": UTC, "n": len(RESULTS), "n_pass": sum(r["pass"] for r in RESULTS),
           "n_fail": sum(not r["pass"] for r in RESULTS), "results": RESULTS,
           "code_sha256": {f: __import__("hashlib").sha256(open(os.path.join(AXIS, f), "rb").read()).hexdigest()
                           for f in ("guard.py", "verdict_keys.py", "compare_verdicts.py", "forecast_pyt.py", "partb_pyt.py",
                                     "partb_checks_pyt.py", "feasibility_pyt.py", "f1d_outroots.py", "s0b_freeze.py",
                                     "VERDICTS_SPEC_PYT.md", "tests_infra/test_infra_pyt.py")},
           "pass": all(r["pass"] for r in RESULTS)}
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1, default=str)
    print(f"{rep['n_pass']}/{rep['n']} pass; report {os.path.join(OUT, 'report.json')}", flush=True)
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
