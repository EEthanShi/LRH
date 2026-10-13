"""PYT-AXIS copy of vl4b_ol.py (implementation E, S0b sub-role pyt-code-E): plan V-P4 (rerun of 4 known pairs).
GATED (plan F1(a)).

Copied from analysis/s12_layermax_openllama/early/vl4b_ol.py (sha256 5c8d5d2c...) and rewritten (every new block is
marked "PYT-AXIS"). Kept: the field/array comparison of compare() (EXACT_KEYS, rel(), the tolerance rule). Changed:
the pairs are the 4 known pairs of plan section 11 V-P4, each rerun here with E's own GPU code (lm_core.ModelGPU,
run_pair, run_layermax.summarize_pair) and compared with its frozen E pair file (each against its own counterpart):

  bigscience__bloomz-560m        x vit_tiny_patch16_224.augreg_in21k            results/S1_2_layermax/pairs/
  openlm-research__open_llama_3b x vit_giant_patch14_dinov2.lvd142m             results/S1_2_layermax/pairs/
  huggyllama__llama-13b          x vit_huge_patch14_224.mae                     results/S1_2_layermax_llama/early/pairs/
  openlm-research__open_llama_13b x vit_huge_patch14_clip_224.laion2b_ft_in12k  results/S1_2_layermax_openllama/early/pairs/

Inputs of the rerun: the LM features of the frozen stacks; the LM clip threshold recomputed with hrhv2 on the float64
all-layer stack (bitwise compared with its frozen reference: S1.2 cache for bloomz-560m and open_llama_3b, the LLaMA
Part B for llama-13b, the OpenLLaMA Part B for open_llama_13b); the LM kNN tables = the frozen knn_final of the LM's
own axis (the GPU tables of this rerun are compared with them and the result reported); the ViT side from E's own
single-model cache (precompute.py; V-P2 checked it against the frozen ViT caches); the permutation key and K read from
the frozen pair file.

Rule (plan section 11 V-P4), both pre: mKNN counts and p identical, CKA <= 1e-12 relative to the frozen pair files.
PYT-AXIS: implemented literally. The integer mKNN counts (rint(mknn.entry_obs * 10240) and mknn.null_counts_int16), every
p value, argmax, tie count and layer list must be identical; every other float field (mKNN and CKA) <= 1e-12 relative.
The OpenLLaMA copy also required the mKNN float fields (count/10240 and what is derived from it) to be bitwise equal; that
stricter outcome is still reported (bitwise_all_fields_and_arrays) but is not the plan's rule: a 1-ulp difference of
count/10240 between devices is not a count difference (the aggregate recomputes mKNN exactly from the counts).
Compared: every field of every subset both files contain (the frozen 8-LM files lack noLMemb/LMdeep/LMemb_only and
calib_then_max), and every stored array both contain. Bitwise equality is expected (same code, same GPU type).

Run (step S6, after precompute.py; GPU, 2 threads):
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_pythia/launch_pyt.sh pyt_E_vp4 2 --gpu 2 -- \\
      venv/bin/python analysis/s12_layermax_pythia/early/vl4_known_pyt.py

Outputs: <RES>/validation/vp4_pairs/<pre>/<lm>__<vit>.{json,npz} (the reruns, run_layermax format)
         <RES>/validation/VP4_known_pairs.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402  (constants and functions only; reads no data at import)

EXACT_KEYS = {"p_value", "argmax_A", "argmax_B", "n_tied_max", "n_layer_pairs", "K", "entry_p_at_argmax",
              "layers_A", "layers_B", "alpha"}


def rel(a, b):
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def frozen_base(pre, lm, vit):  # PYT-AXIS
    return os.path.join(LC.KNOWN_SRC[lm][1], pre, f"{lm}__{vit}")


def rerun_base(pre, lm, vit):  # PYT-AXIS
    return os.path.join(LC.RES, "validation", "vp4_pairs", pre, f"{lm}__{vit}")


def rerun(pres=LC.PRES, pairs=LC.KNOWN4):  # PYT-AXIS
    """GPU rerun of the known pairs with E's code; writes the pair files under <RES>/validation/vp4_pairs/."""
    import torch
    import run_layermax as RL
    LC.setup_torch(threads=2)
    info = {"clip": {}, "lm_gpu_tables_equal_frozen": {}, "timing_s": {}}
    clip = {}
    for lm, _ in pairs:
        c = LC.clip_c(lm)
        ref = LC.s12_clip_c(lm)
        clip[lm] = c
        info["clip"][lm] = {"c": c, "reference": ref, "bitwise_equal": bool(c == ref),
                            "reference_source": LC.KNOWN_SRC[lm][3][0]}
    for pre in pres:
        os.makedirs(os.path.dirname(rerun_base(pre, "x", "y")), exist_ok=True)
        for lm, vit in pairs:
            t0 = time.time()
            fz = json.load(open(frozen_base(pre, lm, vit) + ".json"))
            tab = np.load(os.path.join(LC.KNOWN_SRC[lm][2], f"{lm}__{pre}.npy"))
            A = LC.ModelGPU(lm, pre, "A", clip[lm] if pre == "prh_clip+l2" else None, knn_tables=tab, keep_tables=True)
            info["lm_gpu_tables_equal_frozen"][f"{lm}|{pre}"] = bool(np.array_equal(A.tables, tab))
            vtab = np.load(os.path.join(LC.cache_dir(vit), "knn_final", f"{vit}__{pre}.npy"))
            B = LC.ModelGPU(vit, pre, "B", LC.clip_value(vit) if pre == "prh_clip+l2" else None, knn_tables=vtab)
            R = LC.run_pair(A, B, LC.perms_for(lm, vit, K=int(fz["K"]), key=fz["perm_key"]))
            LC.synchronize()
            js, arrays = RL.summarize_pair(R, A.L, B.L)
            ctm = js.pop("_calib_then_max")
            rec = {"lm": lm, "vit": vit, "pre": pre, "K": int(fz["K"]), "alpha": LC.ALPHA, "perm_key": fz["perm_key"],
                   "L_A": A.L, "L_B": B.L,
                   "vit_layers_relabel_recomputed": [int(x) for x in np.flatnonzero(B.boundary_tie)],
                   "summaries": js, "calib_then_max": ctm, "finished_utc": LC.utc()}
            base = rerun_base(pre, lm, vit)
            np.savez(base + ".tmp.npz", **arrays)
            os.replace(base + ".tmp.npz", base + ".npz")
            with open(base + ".json.tmp", "w") as f:
                json.dump(LC.jsonable(rec), f)
            os.replace(base + ".json.tmp", base + ".json")
            info["timing_s"][f"{lm}|{vit}|{pre}"] = time.time() - t0
            print(f"[{LC.utc()}] V-P4 rerun {pre} {lm} x {vit} ({A.L}x{B.L}) {time.time() - t0:.1f}s done", flush=True)
            del A, B, R
            if LC.DEV == "cuda":
                torch.cuda.empty_cache()
    return info


def compare(pairs=LC.KNOWN4, verbose=True):
    out = {"rule": "mKNN observed and null counts, p, argmax, ties identical; every other field <= 1e-12 relative "  # PYT-AXIS
                   "(plan section 11 V-P4)", "pairs": [list(p) for p in pairs],  # PYT-AXIS
           "per_pre": {}}
    ok_all = True
    for pre in LC.PRES:
        R = {"n_pairs": 0, "n_fields": 0, "max_rel": {"mknn": 0.0, "cka_biased": 0.0, "cka_debiased": 0.0},
             "n_not_bitwise": {"mknn": 0, "cka_biased": 0, "cka_debiased": 0}, "exact_violations": [],
             "arrays": {}, "subsets_compared": {}, "missing": []}  # PYT-AXIS
        arr_max = {}
        for LM, vit in pairs:  # PYT-AXIS
            fm = rerun_base(pre, LM, vit)  # PYT-AXIS
            ff = frozen_base(pre, LM, vit)  # PYT-AXIS
            if not (os.path.exists(fm + ".json") and os.path.exists(ff + ".json")):
                R["missing"].append(f"{LM}|{vit}")
                continue
            jm, jf = json.load(open(fm + ".json")), json.load(open(ff + ".json"))
            R["n_pairs"] += 1
            for met, subs in jf["summaries"].items():
                R["subsets_compared"][f"{LM}|{vit}"] = sorted(subs)  # PYT-AXIS
                for name, sf in subs.items():
                    sm = jm["summaries"][met][name]
                    for k, vf in sf.items():
                        vm = sm[k]
                        R["n_fields"] += 1
                        if k in EXACT_KEYS:  # PYT-AXIS: the plan's identical items (p, argmax, ties, layers)
                            if vm != vf:
                                R["exact_violations"].append(f"{LM}|{vit}|{met}|{name}|{k}: {vm} vs {vf}")
                        if isinstance(vf, (int, float)) and not isinstance(vf, bool):
                            r = rel(vm, vf)
                            R["max_rel"][met] = max(R["max_rel"][met], r)
                            R["n_not_bitwise"][met] += int(vm != vf)
            zm, zf = np.load(fm + ".npz"), np.load(ff + ".npz")
            for key in zf.files:
                if key not in zm.files:  # PYT-AXIS
                    R["exact_violations"].append(f"{LM}|{vit}|{key} missing in the rerun")  # PYT-AXIS
                    continue  # PYT-AXIS
                a, b = zm[key], zf[key]
                same = bool(a.shape == b.shape and np.array_equal(a, b))
                d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.shape == b.shape else float("inf")
                e = arr_max.setdefault(key, {"all_bitwise": True, "max_absdiff": 0.0})
                e["all_bitwise"] &= same
                e["max_absdiff"] = max(e["max_absdiff"], d)
                if key == "mknn.null_counts_int16" and not same:
                    R["exact_violations"].append(f"{LM}|{vit}|{key} not identical")
                if key == "mknn.entry_obs":  # PYT-AXIS: the observed counts, compared as integers
                    ca, cb = np.rint(a * LC.M_ITEMS * LC.KNN_K), np.rint(b * LC.M_ITEMS * LC.KNN_K)
                    if a.shape != b.shape or not np.array_equal(ca, cb):
                        R["exact_violations"].append(f"{LM}|{vit}|mknn observed counts not identical")
        R["arrays"] = arr_max
        R["pass"] = bool(R["n_pairs"] == len(pairs) and not R["exact_violations"]  # PYT-AXIS
                         and max(R["max_rel"].values()) <= 1e-12)  # PYT-AXIS: mKNN float fields too
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare-only", action="store_true", help="compare existing reruns (no GPU)")  # PYT-AXIS
    ap.add_argument("--lms", default=None, help="debug: restrict to the known pairs of these LM stems (comma list); "
                                                "the output then gets the suffix .subset")  # PYT-AXIS
    args = ap.parse_args()
    LC.gate("early/vl4_known_pyt.py")   # PYT-AXIS: plan F1(a), first action
    pairs = LC.KNOWN4  # PYT-AXIS
    if args.lms:  # PYT-AXIS
        want = {x.strip() for x in args.lms.split(",") if x.strip()}
        pairs = tuple(p for p in LC.KNOWN4 if p[0] in want)
    info = None if args.compare_only else rerun(pairs=pairs)  # PYT-AXIS
    out = compare(pairs=pairs)
    if info is not None:  # PYT-AXIS
        out["rerun"] = info
        out["clip_bitwise_equal_reference_all"] = all(v["bitwise_equal"] for v in info["clip"].values())
        out["lm_gpu_tables_equal_frozen_all"] = all(info["lm_gpu_tables_equal_frozen"].values())
        out["pass"] = bool(out["pass"] and out["clip_bitwise_equal_reference_all"])
    out["finished_utc"] = LC.utc()
    os.makedirs(os.path.join(LC.RES, "validation"), exist_ok=True)
    name = "VP4_known_pairs.json" if pairs == LC.KNOWN4 else "VP4_known_pairs.subset.json"  # PYT-AXIS
    with open(os.path.join(LC.RES, "validation", name), "w") as f:  # PYT-AXIS
        json.dump(LC.jsonable(out), f, indent=1)
    print("V-P4", "PASS" if out["pass"] else "FAIL", flush=True)  # PYT-AXIS


if __name__ == "__main__":
    main()
