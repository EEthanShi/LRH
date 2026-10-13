"""OL-AXIS copy of vl4b_llama7b.py (implementation E, S0b sub-role E of oll-axis-extract): plan V-O4(b).
GATED (plan F1(a)).

Copied from analysis/s12_layermax_llama/early/vl4b_llama7b.py (sha256 361e4c04...). Changes, marked
"OL-AXIS": the LMs compared are open_llama_3b and open_llama_7b (the 34 rerun pairs); the Part B gate is the
first action; the comparison is a function (compare()) that aggregate_ol.py also calls for the section G key
G|rerun_equals_frozen_3b7b; output name.

The 3B/7B layer-max outputs of E (results/S1_2_layermax_openllama/early/pairs/<pre>/
openlm-research__open_llama_{3b,7b}__<vit>.{json,npz}) must reproduce the frozen early outputs
(results/S1_2_layermax/pairs/<pre>/..., read-only): every field of every
subset that both contain (T = s_obs, c_alpha, s_cal, p_value, uncensored,
null_mean, null_sd, argmax, ties, entry fields), and every stored array.

Rule (plan section 11 V-O4(b)): relative difference <= 1e-12 for
CKA; mKNN values, p-values, argmax/tie counts and the mKNN null counts must be
identical. Bitwise equality is expected (same code, same GPU). Any non-identical count invokes the plan
section 3 source rule (aggregate_ol.py then also writes the x|frozen|... extras).

Run (step S7, after run_layermax.py; CPU):
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_openllama/launch_ol.sh ol_E_vo4b 1 -- \\
      venv/bin/python analysis/s12_layermax_openllama/early/vl4b_ol.py

Output: <RES>/validation/VO4b_3b7b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402  (constants and functions only; reads no data at import)
import guard  # noqa: E402

LMS_CMP = (LC.SLUG_3B, LC.SLUG_7B)
EXACT_KEYS = {"p_value", "argmax_A", "argmax_B", "n_tied_max", "n_layer_pairs", "K", "entry_p_at_argmax",
              "layers_A", "layers_B", "alpha"}


def rel(a, b):
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def compare(lms=LMS_CMP, verbose=True):
    out = {"rule": "CKA rel <= 1e-12; mKNN values, p, argmax, ties, null counts identical", "lms": list(lms),
           "per_pre": {}}
    ok_all = True
    for pre in LC.PRES:
        R = {"n_pairs": 0, "n_fields": 0, "max_rel": {"mknn": 0.0, "cka_biased": 0.0, "cka_debiased": 0.0},
             "n_not_bitwise": {"mknn": 0, "cka_biased": 0, "cka_debiased": 0}, "exact_violations": [],
             "arrays": {}, "subsets_compared": None, "missing": []}
        arr_max = {}
        for LM in lms:
            for vit, _ in LC.VITS:
                fm = os.path.join(LC.pairs_dir(pre, LM), f"{LM}__{vit}")
                ff = os.path.join(LC.FROZEN_RES, "pairs", pre, f"{LM}__{vit}")
                if not (os.path.exists(fm + ".json") and os.path.exists(ff + ".json")):
                    R["missing"].append(f"{LM}|{vit}")
                    continue
                jm, jf = json.load(open(fm + ".json")), json.load(open(ff + ".json"))
                R["n_pairs"] += 1
                for met, subs in jf["summaries"].items():
                    R["subsets_compared"] = sorted(subs)
                    for name, sf in subs.items():
                        sm = jm["summaries"][met][name]
                        for k, vf in sf.items():
                            vm = sm[k]
                            R["n_fields"] += 1
                            if k in EXACT_KEYS or met == "mknn":
                                if vm != vf:
                                    R["exact_violations"].append(f"{LM}|{vit}|{met}|{name}|{k}: {vm} vs {vf}")
                            if isinstance(vf, (int, float)) and not isinstance(vf, bool):
                                r = rel(vm, vf)
                                R["max_rel"][met] = max(R["max_rel"][met], r)
                                R["n_not_bitwise"][met] += int(vm != vf)
                zm, zf = np.load(fm + ".npz"), np.load(ff + ".npz")
                for key in zf.files:
                    a, b = zm[key], zf[key]
                    same = bool(a.shape == b.shape and np.array_equal(a, b))
                    d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.shape == b.shape else float("inf")
                    e = arr_max.setdefault(key, {"all_bitwise": True, "max_absdiff": 0.0})
                    e["all_bitwise"] &= same
                    e["max_absdiff"] = max(e["max_absdiff"], d)
                    if key == "mknn.null_counts_int16" and not same:
                        R["exact_violations"].append(f"{LM}|{vit}|{key} not identical")
        R["arrays"] = arr_max
        R["pass"] = bool(R["n_pairs"] == len(lms) * len(LC.VITS) and not R["exact_violations"]
                         and max(R["max_rel"]["cka_biased"], R["max_rel"]["cka_debiased"]) <= 1e-12)
        R["bitwise_all_fields_and_arrays"] = bool(sum(R["n_not_bitwise"].values()) == 0
                                                  and all(v["all_bitwise"] for v in arr_max.values()))
        R["exact_violations"] = R["exact_violations"][:200]
        ok_all &= R["pass"]
        out["per_pre"][pre] = R
        if verbose:
            print(pre, {k: R[k] for k in ("n_pairs", "n_fields", "max_rel", "n_not_bitwise", "pass",
                                          "bitwise_all_fields_and_arrays")}, "exact violations",
                  len(R["exact_violations"]), flush=True)
    out["pass"] = bool(ok_all)
    return out


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    guard.require_partb_frozen(LC.RES, script="early/vl4b_ol.py")   # plan F1(a): first action
    out = compare()
    out["finished_utc"] = LC.utc()
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    with open(os.path.join(LC.RES, "validation", "VO4b_3b7b.json"), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    print("V-O4(b)", "PASS" if out["pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
