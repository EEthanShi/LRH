"""PYT-AXIS (new, implementation E test driver, S0b only): run E's whole S0b test battery once and record every run,
failed runs included, in <final root>/test_log_E.json (VERDICTS_SPEC_PYT section 14 "--test-log"). Runs only in the S0b
test mode; every child writes only under analysis/s12_layermax_pythia/s0b_scratch/E/final_<UTC>/; no GPU is used
(CUDA_VISIBLE_DEVICES is empty; the pair kernel runs with PYT_E_TEST_DEVICE=cpu); every child except the dry runs starts
through launch_pyt.sh (thread tokens, at most 4 threads).

Steps: (1) dry runs of every gated command of early/s0b_dryruns_E.json with PYT_S0B_TEST unset: each must exit 3, print
nothing on stdout and create no file under results/S1_2_layermax_pythia/ or early/; (2) V-P8; (3) V-P9 (synthetic);
(4) legacy outputs compared with the shared compare_verdicts.py; (5) the 7-size stand-in end to end (aggregate in S7 and
S8 modes, V-P7, report, hashes); (6) V-P5 on relabelled LLaMA-axis E/I files, identical and with a planted mismatch;
(7) CPU smoke of the pair-level scripts on synthetic stacks (precompute, main run, no-resume refusal, --restart, V-P3
subset, V-P4 subset).
usage: cd ~/hrh_v2_stage1_20261008 && PYT_S0B_TEST=1 venv/bin/python analysis/s12_layermax_pythia/early/tests/run_e_tests.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import lm_core as LC  # noqa: E402

PROJ = LC.PROJ
EARLY = "analysis/s12_layermax_pythia/early"
LAUNCH = "analysis/s12_layermax_pythia/launch_pyt.sh"
PY = "venv/bin/python"
VIT = "vit_tiny_patch16_224.augreg_in21k"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    if not LC.TEST:
        LC._refuse("run_e_tests.py runs only with PYT_S0B_TEST=1 (S0b test mode)")
    os.chdir(PROJ)
    utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    FR = os.path.join("analysis/s12_layermax_pythia/s0b_scratch/E", f"final_{utc}")
    LOG = os.path.join(FR, "logs")
    os.makedirs(LOG, exist_ok=True)
    base_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", PYT_LOGDIR=LOG)
    runs = []

    def launch(name, threads, root, args, expect_rc=0, extra=None, check=None):
        env = dict(base_env, PYT_S0B_TEST="1", PYT_E_TEST_ROOT=os.path.join(FR, root))
        env.update(extra or {})
        threads = min(threads, 2)   # the axis has 6 thread tokens shared with the other S0b roles
        t0 = time.time()
        tries = 0
        while True:   # launch_pyt.sh exits 75 (nothing started, no marker) while the tokens are taken: wait, retry
            tries += 1
            nm = name if tries == 1 else f"{name}_try{tries}"
            rc = subprocess.run([LAUNCH, nm, str(threads), "--", PY] + args, env=env).returncode
            if rc != 75 or tries >= 40:
                break
            time.sleep(15)
        name = nm
        lp = os.path.join(LOG, name + ".log")
        tail = open(lp).read().splitlines()[-12:] if os.path.exists(lp) else []
        rec = {"name": name, "cmd": " ".join(args), "root": root, "rc": rc, "expected_rc": expect_rc, "tries": tries,
               "seconds": round(time.time() - t0, 1), "log": lp, "log_sha256": sha(lp) if os.path.exists(lp) else None,
               "result_lines": [ln for ln in tail if ("PASS" in ln or "FAIL" in ln or "REFUSED" in ln or "written" in ln
                                                      or "missing" in ln)][-8:]}
        rec["pass"] = rc == expect_rc
        if check is not None:
            try:
                c = check()
            except Exception as e:  # noqa: BLE001
                c = {"pass": False, "error": repr(e)}
            rec["check"] = c
            rec["pass"] = rec["pass"] and bool(c.get("pass"))
        runs.append(rec)
        print(f"[{LC.utc()}] {name}: rc={rc} (expected {expect_rc}) pass={rec['pass']}", flush=True)
        return rec

    def jload(p):
        return json.load(open(os.path.join(FR, p)))

    # ---- (1) dry runs
    marker = os.path.join(FR, "dryrun_marker")
    open(marker, "w").write(utc)
    time.sleep(1.1)
    env0 = {k: v for k, v in base_env.items() if not k.startswith("PYT_S0B_TEST") and not k.startswith("PYT_E_TEST")}
    env0["OMP_NUM_THREADS"] = "1"
    dry = []
    for entry in json.load(open(os.path.join(EARLY, "s0b_dryruns_E.json"))):
        label, script, args = entry[0], entry[1], entry[2:]
        p = subprocess.run([PY, script] + args, env=env0, capture_output=True, text=True)
        dry.append({"label": label, "script": script, "args": args, "rc": p.returncode, "stdout_bytes": len(p.stdout),
                    "stderr_last": p.stderr.strip().splitlines()[-1][:300] if p.stderr.strip() else ""})
    new = subprocess.run(["find", "results/S1_2_layermax_pythia", EARLY, "-newer", marker, "-not", "-path", "*__pycache__*"],
                         capture_output=True, text=True).stdout.split()
    dry_ok = all(d["rc"] == 3 and d["stdout_bytes"] == 0 for d in dry) and not new
    runs.append({"name": "dryruns_E", "n": len(dry), "all_exit3_no_stdout": all(d["rc"] == 3 and d["stdout_bytes"] == 0
                                                                                 for d in dry),
                 "new_files": new, "pass": dry_ok, "runs": dry})
    print(f"[{LC.utc()}] dry runs: {len(dry)} commands, pass={dry_ok}", flush=True)
    # ---- (2) V-P8, (3) V-P9
    launch("pytE_vp8", 4, "vp", [f"{EARLY}/vl8_generalise.py"],
           check=lambda: {"pass": jload("vp/early/validation/VP8_generalisation.json")["pass"]})
    launch("pytE_vp9", 1, "vp", [f"{EARLY}/vl9_known_answers.py"],
           check=lambda: {"pass": jload("vp/early/validation/VP9_known_answers.json")["pass"]})
    # ---- (4) legacy outputs through the shared comparator
    launch("pytE_legacy", 4, "vp", [f"{EARLY}/aggregate_pyt.py", "--legacy"])
    launch("pytE_legacy_common", 4, "vp", [f"{EARLY}/aggregate_pyt.py", "--legacy-common"])
    lg = os.path.join(FR, "vp/early/legacy")
    for impl in ("E", "I"):
        p = subprocess.run([PY, "analysis/s12_layermax_pythia/compare_verdicts.py", f"{lg}/verdicts_ol_legacy.json",
                            "results/S1_2_layermax_openllama/layermax_openllama_summary.json", "--b-sub",
                            f"embedded.verdicts.{impl}", "--legacy", "--sections", "A,B,C",
                            "--out", f"{lg}/compare_vs_OL_{impl}.json"], env=env0, capture_output=True, text=True)
        runs.append({"name": f"compare_legacy_vs_OL_{impl}", "rc": p.returncode, "summary": p.stdout.strip(),
                     "pass": p.returncode == 0})
    p = subprocess.run([PY, "analysis/s12_layermax_pythia/compare_verdicts.py", f"{lg}/verdicts_ol_legacy_common.json",
                        f"{lg}/verdicts_ol_legacy.json", "--legacy", "--sections", "A,B,C"], env=env0,
                       capture_output=True, text=True)
    runs.append({"name": "compare_legacy_common_vs_legacy", "rc": p.returncode, "summary": p.stdout.strip(),
                 "pass": p.returncode == 0})
    # ---- (5) the 7-size stand-in end to end
    fxe = {"PYT_E_TEST_STANDINS": os.path.join(PROJ, FR, "fx", "standins.json")}
    launch("pytE_fx_build", 1, "fx", [f"{EARLY}/tests/build_standin_fixture.py"], extra=fxe)

    def chk_primary():
        d = jload("fx/early/verdicts_pyt.json")["v"]
        nl2 = sum(1 for k, x in d.items() if not k.startswith("x|") and "|l2|" in k + "|" and x is not None)
        return {"missing": len(d["x|check|required_keys_missing"]), "l2_non_null": nl2, "l2_run": d["G|l2_run"],
                "pass": not d["x|check|required_keys_missing"] and nl2 == 0 and d["G|l2_run"] is False}
    launch("pytE_fx_aggregate_primary", 4, "fx", [f"{EARLY}/aggregate_pyt.py", "--pre", "prh_clip+l2"], extra=fxe,
           check=chk_primary)

    def chk_both():
        d = jload("fx/early/verdicts_pyt.json")["v"]
        prev = [f for f in os.listdir(os.path.join(FR, "fx/early")) if ".prev-" in f]
        return {"missing": len(d["x|check|required_keys_missing"]),
                "non_required_non_x": len(d["x|check|non_required_non_x_keys"]), "kept_aside": prev,
                "pass": not d["x|check|required_keys_missing"] and not d["x|check|non_required_non_x_keys"]
                and len(prev) == 2 and d["G|l2_run"] is True}
    launch("pytE_fx_aggregate_both", 4, "fx", [f"{EARLY}/aggregate_pyt.py"], extra=fxe, check=chk_both)
    launch("pytE_fx_vp7", 2, "fx", [f"{EARLY}/vl7_prop5.py"], extra=fxe,
           check=lambda: {"pass": jload("fx/early/validation/VP7_prop5.json")["pass"]})
    launch("pytE_fx_report", 1, "fx", [f"{EARLY}/report_pyt.py"], extra=fxe,
           check=lambda: {"pass": os.path.getsize(os.path.join(FR, "fx/early/REPORT.md")) > 1000})
    launch("pytE_fx_hashes", 1, "fx", [f"{EARLY}/write_hashes.py"], extra=fxe,
           check=lambda: {"pass": len(jload("fx/early/hashes.json")) > 10})
    # ---- (6) V-P5 E side
    for t, extra_args in (("a", []), ("b", ["--perturb", "tr|prh_clip+l2|cka_biased|full|raw"])):
        launch(f"pytE_vp5_fixture_{t}", 1, f"vp5{t}", [f"{EARLY}/tests/build_vp5_fixture.py", os.path.join(FR, "fx")]
               + extra_args)

    def chk5a():
        d = jload("vp5a/early/validation/VP5_E_vs_I.json")
        return {"pairs": d["pairs"]["pass"], "EI_mismatch": d["verdicts"]["EI"]["n_mismatch"],
                "FK_missing_as_expected": "missing" in d["verdicts"]["FK"],
                "pass": d["pairs"]["pass"] and d["verdicts"]["EI"]["n_mismatch"] == 0 and "missing" in d["verdicts"]["FK"]}
    launch("pytE_vp5_a", 2, "vp5a", [f"{EARLY}/vl5_e_vs_i.py"], check=chk5a)

    def chk5b():
        p_ = os.path.join(FR, "vp5b/early/validation/VP5_verdicts.json")
        d = json.load(open(p_))
        txt = open(p_).read()
        planted = jload("fx/early/verdicts_pyt.json")["v"]["tr|prh_clip+l2|cka_biased|full|raw"]
        return {"EI_mismatching_keys": d["verdicts"]["EI"]["mismatching_keys"],
                "value_absent": repr(planted)[:10] not in txt and repr(planted + 0.5)[:10] not in txt,
                "pass": d["verdicts"]["EI"]["mismatching_keys"] == ["tr|prh_clip+l2|cka_biased|full|raw"]
                and repr(planted)[:10] not in txt}
    launch("pytE_vp5_b", 1, "vp5b", [f"{EARLY}/vl5_e_vs_i.py", "--part", "verdicts"], check=chk5b)
    # ---- (7) CPU smoke of the pair-level scripts
    cpe = {"PYT_E_TEST_DEVICE": "cpu", "PYT_E_TEST_STANDINS": os.path.join(PROJ, FR, "cpu", "standins.json")}
    launch("pytE_cpu_synth", 1, "cpu", [f"{EARLY}/tests/make_synth_stacks.py"], extra=cpe)

    def chk_pre():
        s = jload("cpu/early/validation/VP1_VP2_knn_tables.subset.json")["summary"]
        return {"VP1": s["VP1_pass"], "VP2": s["VP2_pass"], "pass": s["VP1_pass"] and s["VP2_pass"]}
    launch("pytE_cpu_precompute", 4, "cpu", [f"{EARLY}/precompute.py", "--models", f"70m,6.9b,{VIT}"], extra=cpe,
           check=chk_pre)
    launch("pytE_cpu_main", 4, "cpu", [f"{EARLY}/run_layermax.py", "--lms", "70m,6.9b", "--vits", VIT], extra=cpe)
    launch("pytE_cpu_main_noresume", 1, "cpu", [f"{EARLY}/run_layermax.py", "--lms", "70m", "--vits", VIT, "--pre", "l2"],
           expect_rc=2, extra=cpe)
    launch("pytE_cpu_main_restart", 2, "cpu", [f"{EARLY}/run_layermax.py", "--lms", "70m", "--vits", VIT, "--pre", "l2",
                                              "--restart"], extra=cpe)

    def chk_vp3():
        s = jload("cpu/early/validation/VP3.subset.json")["summary"]
        return {"n_cases": s["V2_n_cases"], "cka_maxabs": s["V2_max_absdiff_cka"], "mknn_exact": s["V2_mknn_all_exact"],
                "pass": s["V2_cka_pass"] and s["V2_mknn_all_exact"] and s["V2_n_cases"] == 6,
                "note": "subset of 6 of the 24 V-P3 cases (ViT-Ti only), so VP3_pass is false by design"}
    launch("pytE_cpu_vp3", 4, "cpu", [f"{EARLY}/validate.py", "--vits", VIT], extra=cpe, check=chk_vp3)

    def chk_vp4():
        d = jload("cpu/early/validation/VP4_known_pairs.subset.json")
        return {"pass_literal_rule": d["pass"], "bitwise": {p: r["bitwise_all_fields_and_arrays"] for p, r in d["per_pre"].items()},
                "max_rel": {p: r["max_rel"] for p, r in d["per_pre"].items()}, "pass": d["pass"],
                "note": "CPU rerun vs frozen GPU file: bitwise not expected across devices"}
    launch("pytE_cpu_vp4", 4, "cpu", [f"{EARLY}/vl4_known_pyt.py", "--lms", "bigscience__bloomz-560m"], extra=cpe,
           check=chk_vp4)
    code = {f"{EARLY}/{f}": sha(os.path.join(EARLY, f)) for f in sorted(os.listdir(EARLY))
            if os.path.isfile(os.path.join(EARLY, f))}
    code.update({f"{EARLY}/tests/{f}": sha(os.path.join(EARLY, "tests", f)) for f in sorted(os.listdir(os.path.join(EARLY, "tests")))
                 if f.endswith(".py")})
    shared = {f: sha(os.path.join("analysis/s12_layermax_pythia", f)) for f in
              ("VERDICTS_SPEC_PYT.md", "verdict_keys.py", "compare_verdicts.py", "guard.py", "launch_pyt.sh")}
    out = {"what": "implementation E, S0b test battery (VERDICTS_SPEC_PYT section 14 test log)", "utc": utc,
           "final_root": FR, "code_sha256": code, "shared_sha256": shared, "runs": runs,
           "n_runs": len(runs), "n_pass": sum(1 for r in runs if r["pass"]),
           "pass": all(r["pass"] for r in runs),
           "records_for_s0b_freeze": {"VP8": os.path.join(FR, "vp/early/validation/VP8_generalisation.json"),
                                      "VP9": os.path.join(FR, "vp/early/validation/VP9_known_answers.json")}}
    with open(os.path.join(FR, "test_log_E.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[{LC.utc()}] test log {FR}/test_log_E.json: {out['n_pass']}/{out['n_runs']} pass", flush=True)


if __name__ == "__main__":
    main()
