"""Machine-readable S1.1 state: gate, missing lead decisions, code/plan hashes,
design hashes, and synthetic dry-run timings -> results/S1_1/S1_1_status.json."""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s11_common as C  # noqa: E402

P = C.PROJECT


def main():
    dry = os.path.join(C.RESULTS_DIR, "dryrun_synthetic", "raw", "none")
    timings = {}
    for js in sorted(glob.glob(os.path.join(dry, "*.json"))):
        if ".draw" in js:
            continue
        with open(js) as f:
            d = json.load(f)
        per_draw = {k: v["seconds"] for k, v in d["draw_diag"].items()}
        t16 = {k: v["timing"].get("16384") for k, v in d["draw_diag"].items()}
        timings[d["pair"]] = {"draws": d["draws"], "seconds_total": d["seconds_total"],
                              "seconds_per_draw": per_draw, "m16384_breakdown_draw0": t16.get("0") or next(iter(t16.values())),
                              "a_info": d.get("a_info")}
    st = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "role": "s11",
        "gate_G1": "NOT RUN",
        "reason": ("Verifier verdict for S1.1 is NO with P1 as frozen (results/verification_report.md section 1); "
                   "its decision D4 says not to touch COCO until the lead decides; SPEC section 7 requires G0 to pass "
                   "before real-data analysis and the lead has not recorded D1/D2. No lead decision found in "
                   "STATUS.md, SPEC_CHANGELOG.md or prereg/."),
        "coco_feature_arrays_read": False,
        "coco_metadata_read": ["features/coco_pool/items.parquet (idx, image_id, text)",
                               "features/coco_pool/*.json sidecars"],
        "lead_decisions_needed": ["D1", "D2", "D3", "D4", "D5",
                                  "freeze analysis/s11/ANALYSIS_PLAN_S11.md (sha256 into LEAD_AUTHORIZATION.json)"],
        "authorization_file_present": os.path.exists(os.path.join(C.ANALYSIS_DIR, "LEAD_AUTHORIZATION.json")),
        "plan_draft_sha256": C.file_sha256(os.path.join(C.ANALYSIS_DIR, "ANALYSIS_PLAN_S11.md")),
        "code_sha256": {os.path.basename(f): C.file_sha256(f) for f in sorted(glob.glob(os.path.join(C.ANALYSIS_DIR, "*.py")))},
        "hrhv2_predict_py_sha256": C.file_sha256(os.path.join(P, "hrhv2", "predict.py")),
        "hrhv2_predictor_choice_sha256": C.file_sha256(os.path.join(P, "hrhv2", "PREDICTOR_CHOICE.md")),
        "prereg_sha256": C.file_sha256(os.path.join(P, "prereg", "PREREG_stage1.md")),
        "design": json.load(open(os.path.join(C.RESULTS_DIR, "design", "design.json"))),
        "dry_run_synthetic_timings": timings,
    }
    out = os.path.join(C.RESULTS_DIR, "S1_1_status.json")
    with open(out, "w") as f:
        json.dump(st, f, indent=1, default=str)
    print("wrote", out)
    for k, v in timings.items():
        print(k, round(v["seconds_total"]), {d: round(s) for d, s in v["seconds_per_draw"].items()})


if __name__ == "__main__":
    main()
