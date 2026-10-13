"""X3b independent check: manifest and verdict (AGREE / DISAGREE).

Usage: python x3bi_manifest.py <outdir> <cpu_core_hours>
EXPLORATORY.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x3bi_common as C  # noqa: E402


def main():
    outdir, cpu_h = sys.argv[1], float(sys.argv[2])
    here = os.path.dirname(os.path.abspath(__file__))
    run = os.path.join(C.ROOT, "results/E0_sentence/x3b")
    oc = json.load(open(os.path.join(outdir, "obs", "x3bi_obs_compare.json")))
    sc = json.load(open(os.path.join(outdir, "stats", "x3bi_compare.json")))
    dchk = json.load(open(os.path.join(outdir, "design_check.json")))
    kc = oc["knn_compare"]
    cj = sc["compare_json_from_my_cells"]
    cr = sc["compare_json_from_run_cells"]
    cells = sc["cells_vs_run_obs"]
    cell_ok = all(v.get("n_gt_tol", 0) == 0 for k, v in cells.items() if isinstance(v, dict)) and \
        cells["rows_matched"] == cells["rows_mine"] == cells["rows_run"]
    only_what_missing = all(p.endswith("/what") for p in cj["missing"])
    checks = {
        "design_equals_S1_1_bitwise": all(dchk.values()),
        "knn_rows_all101_equal": sum(v["rows_all101_equal"] for v in kc.values()) == sum(v["rows"] for v in kc.values()),
        "anchor_overlap_counts_equal": oc["anchor_units_equal"] == oc["anchor_units"] and oc["n_mismatch"] == 0,
        "cells_vs_run_obs_within_1e-12": cell_ok,
        "json_numeric_within_1e-12": cj["n_bad"] == 0 and cj["max_abs_diff_numeric"] <= 1e-12,
        "json_labels_identical": cj["n_bad"] == 0,
        "json_no_field_missing_except_free_text_what": only_what_missing,
        "json_from_run_cells_agrees": cr["n_bad"] == 0,
    }
    verdict = "AGREE" if all(checks.values()) else "DISAGREE"
    code = {os.path.basename(f): C.sha256_file(f) for f in sorted(glob.glob(os.path.join(here, "x3bi_*.py")))}
    outs = {}
    for f in ["design_check.json", "knn_compute.json", "knn.log", "obs.log", "obs/x3bi_cells.parquet",
              "obs/x3bi_obs_compare.json", "obs/x3bi_anchor_counts.npz", "stats/x3bi_stats.json",
              "stats/x3bi_stats_from_run_cells.json", "stats/x3bi_compare.json"]:
        outs[f] = C.sha256_file(os.path.join(outdir, f))
    inputs = {
        "results/E0_sentence/x3b/X3B_results.json": C.sha256_file(os.path.join(run, "X3B_results.json")),
        "results/E0_sentence/x3b/obs/obs_X3B.parquet": C.sha256_file(os.path.join(run, "obs/obs_X3B.parquet")),
        "results/E0_sentence/x3b/obs/obs_X3B_ref.parquet": C.sha256_file(os.path.join(run, "obs/obs_X3B_ref.parquet")),
        "results/E0_sentence/x3b/eval/X3B_pairs.parquet": C.sha256_file(os.path.join(run, "eval/X3B_pairs.parquet")),
        "analysis/e0_sentence/x3b/X3B_READINGS.md": C.sha256_file(os.path.join(C.ROOT, "analysis/e0_sentence/x3b/X3B_READINGS.md")),
    }
    man = {
        "role": "x3b-indep", "status": "EXPLORATORY", "verdict": verdict, "checks": checks,
        "independence": "own numpy code; imports only numpy, pandas, hrhv2.prep, hrhv2.util; did not read or import "
                        "x3b_scores.py, x3b_report.py, e0_*.py, s11copy/, analysis/s11/*.py, x3_*.py or x3/indep/; "
                        "run-side caches, parquets and X3B_results.json read as data only",
        "summary": {
            "models": len(kc), "knn_rows": sum(v["rows"] for v in kc.values()),
            "knn_rows_all101_equal": sum(v["rows_all101_equal"] for v in kc.values()),
            "vals_max_abs_diff_by_pre": {p: max(v["vals_max_abs_diff"] for k, v in kc.items() if k.startswith(p))
                                         for p in ("clip", "l2", "ctr")},
            "median_D_max_abs_diff": max(v["median_D_max_abs_diff"] for v in kc.values()),
            "clip_thresholds_equal": all(v["clip_threshold_mine"] == v["clip_threshold_run"]
                                         for v in kc.values() if "clip_threshold_mine" in v),
            "exact_k_boundary_tie_rows_mine": sum(sum(v["exact_tie_rows_mine"].values()) for v in kc.values()),
            "pairs": oc["n_pairs"], "anchor_units": oc["anchor_units"], "anchor_units_equal": oc["anchor_units_equal"],
            "neartie_clause_used": oc["n_mismatch_accepted_by_neartie_clause"],
            "json_numeric_fields": cj["n_numeric"], "json_label_bool_int_fields": cj["n_label_bool_int"],
            "json_max_abs_diff": cj["max_abs_diff_numeric"], "json_missing": cj["missing"],
            "json_from_run_cells_max_abs_diff": cr["max_abs_diff_numeric"],
            "eval_vs_run": sc["eval_vs_run"], "frozen_refs": sc["frozen_refs"],
        },
        "code_sha256": code, "inputs_sha256": inputs, "outputs_sha256": outs,
        "compute": {"cpu_core_hours": cpu_h, "gpu_hours": 0.0, "cores": "taskset -c 0-5, nice 10, 1 BLAS thread per worker"},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    p = os.path.join(outdir, "INDEP_MANIFEST.json")
    with open(p, "w") as fh:
        json.dump(man, fh, indent=1, default=str)
    os.chmod(p, 0o444)
    print(verdict, C.sha256_file(p))
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
