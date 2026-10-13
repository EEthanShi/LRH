"""x3-indep step 4: manifest (sha256 of code, inputs, outputs) and compute.

usage: x3i_manifest.py <run_dir>
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x3i_common as C  # noqa: E402


def main():
    run_dir = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    code = {os.path.basename(p): C.sha256_file(p)
            for p in sorted(glob.glob(os.path.join(here, "*.py")) + glob.glob(os.path.join(here, "*.sh")))}
    outs = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "**", "*"), recursive=True)):
        if os.path.isfile(p) and not p.endswith("INDEP_MANIFEST.json"):
            outs[os.path.relpath(p, run_dir)] = C.sha256_file(p)
    obs = json.load(open(os.path.join(run_dir, "obs", "x3i_obs_compare.json")))
    st = json.load(open(os.path.join(run_dir, "stats", "x3i_stats.json")))
    cpu = sum(json.load(open(p))["cpu_s"] for p in glob.glob(os.path.join(run_dir, "knn", "*.json")))
    cpu += obs["cpu_s"] + st["cpu_s"]
    inputs = dict(obs["inputs_sha256"])
    for rel in C.EXPECTED_SHA:
        inputs[rel] = C.sha256_file(C.P(rel))
    inputs["results/S1_1/design/design.npz"] = obs["design_checks"]["design_npz_sha256"]
    inputs["results/E0_sentence/x3/X3_results.json"] = st["X3_results_sha256"]
    man = {"role": "x3-indep", "code_sha256": code, "outputs_sha256": outs, "inputs_sha256": inputs,
           "cpu_core_hours_measured_process_time": cpu / 3600.0, "gpu_hours": 0.0}
    C.write_json_new(os.path.join(run_dir, "INDEP_MANIFEST.json"), man)
    print(C.sha256_file(os.path.join(run_dir, "INDEP_MANIFEST.json")), cpu / 3600.0)


if __name__ == "__main__":
    main()
