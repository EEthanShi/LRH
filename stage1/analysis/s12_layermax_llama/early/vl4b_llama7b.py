"""LLAMA-AXIS (new file, implementation E, role llama-axis-run-early): plan V-L4(b).

The llama-7b layer-max outputs of E (results/S1_2_layermax_llama/early/pairs/<pre>/
huggyllama__llama-7b__<vit>.{json,npz}) must reproduce the frozen early outputs
(results/S1_2_layermax/pairs/<pre>/..., read-only): every field of every
subset that both contain (T = s_obs, c_alpha, s_cal, p_value, uncensored,
null_mean, null_sd, argmax, ties, entry fields), and every stored array.

Rule (plan section 4 V-L4(b), task text): relative difference <= 1e-12 for
CKA; mKNN values, p-values, argmax/tie counts and the mKNN null counts must be
identical. Bitwise equality is expected (same code, same GPU).

Output: results/S1_2_layermax_llama/early/validation/VL4b_llama7b.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402

LM = "huggyllama__llama-7b"
EXACT_KEYS = {"p_value", "argmax_A", "argmax_B", "n_tied_max", "n_layer_pairs", "K", "entry_p_at_argmax",
              "layers_A", "layers_B", "alpha"}


def rel(a, b):
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def main():
    out = {"rule": "CKA rel <= 1e-12; mKNN values, p, argmax, ties, null counts identical", "per_pre": {}}
    ok_all = True
    for pre in LC.PRES:
        R = {"n_pairs": 0, "n_fields": 0, "max_rel": {"mknn": 0.0, "cka_biased": 0.0, "cka_debiased": 0.0},
             "n_not_bitwise": {"mknn": 0, "cka_biased": 0, "cka_debiased": 0}, "exact_violations": [],
             "arrays": {}, "subsets_compared": None, "missing": []}
        arr_max = {}
        for vit, _ in LC.VITS:
            fm = os.path.join(LC.RES, "pairs", pre, f"{LM}__{vit}")
            ff = os.path.join(LC.FROZEN_RES, "pairs", pre, f"{LM}__{vit}")
            if not (os.path.exists(fm + ".json") and os.path.exists(ff + ".json")):
                R["missing"].append(vit)
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
                                R["exact_violations"].append(f"{vit}|{met}|{name}|{k}: {vm} vs {vf}")
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
                    R["exact_violations"].append(f"{vit}|{key} not identical")
        R["arrays"] = arr_max
        R["pass"] = bool(R["n_pairs"] == len(LC.VITS) and not R["exact_violations"]
                         and max(R["max_rel"]["cka_biased"], R["max_rel"]["cka_debiased"]) <= 1e-12)
        R["bitwise_all_fields_and_arrays"] = bool(sum(R["n_not_bitwise"].values()) == 0
                                                  and all(v["all_bitwise"] for v in arr_max.values()))
        ok_all &= R["pass"]
        out["per_pre"][pre] = R
        print(pre, {k: R[k] for k in ("n_pairs", "n_fields", "max_rel", "n_not_bitwise", "pass",
                                      "bitwise_all_fields_and_arrays")}, "exact violations", len(R["exact_violations"]),
              flush=True)
    out["pass"] = bool(ok_all)
    out["finished_utc"] = LC.utc()
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    with open(os.path.join(LC.RES, "validation", "VL4b_llama7b.json"), "w") as f:
        json.dump(LC.jsonable(out), f, indent=1)
    print("V-L4(b)", "PASS" if out["pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
