"""PYT-AXIS (new, implementation E test helper, S0b only): a fixture for E's V-P5 script (vl5_e_vs_i.py), from frozen
non-Pythia files. Runs only in the S0b test mode and writes only under the E test root. Computes no score.

The 7 stand-in slugs synth__pyt-<size> are mapped alternately to huggyllama__llama-13b and huggyllama__llama-30b (the
pair-level part of V-P5 compares E and I on the same known pair, so the size order is irrelevant here):
  early/pairs/<pre>/synth__pyt-<size>__<vit>.{json,npz}   <- results/S1_2_layermax_llama/early/pairs/<pre>/ (E)
  indep/<pre>/pairs/synth__pyt-<size>__<vit>.npz          <- results/S1_2_layermax_llama/indep/<pre>/pairs/ (I)
  early/summary.json, early/verdicts_pyt.json             <- copies of the stand-in fixture's E outputs (argument 1)
  indep/verdicts_pyt.json                                 <- the same E verdicts (identical) or, with --perturb KEY, the
                                                             same with that key's value changed (a planted mismatch)
usage: PYT_S0B_TEST=1 PYT_E_TEST_ROOT=<root> python tests/build_vp5_fixture.py <run_fx root> [--perturb KEY]
"""
from __future__ import annotations

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import lm_core as LC  # noqa: E402

LLAMA_I = os.path.join(LC.PROJ, "results", "S1_2_layermax_llama", "indep")


def main():
    if not LC.TEST:
        LC._refuse("build_vp5_fixture.py runs only with PYT_S0B_TEST=1 (S0b test mode)")
    src_fx = os.path.realpath(sys.argv[1])
    assert LC._inside(src_fx, LC.E_TEST_BASE), src_fx
    perturb = sys.argv[sys.argv.index("--perturb") + 1] if "--perturb" in sys.argv else None
    root = LC.PYT_RES
    assert LC._inside(root, LC.E_TEST_BASE), root
    sizes = list(LC.SIZES)
    mp = {s: ("huggyllama__llama-13b" if i % 2 == 0 else "huggyllama__llama-30b") for i, s in enumerate(sizes)}
    for pre in LC.PRES:
        de = os.path.join(root, "early", "pairs", pre)
        di = os.path.join(root, "indep", pre, "pairs")
        os.makedirs(de, exist_ok=True)
        os.makedirs(di, exist_ok=True)
        for s in sizes:
            lm, sl = mp[s], LC.SLUG_TEST[s]
            for vit, _ in LC.VITS:
                b = os.path.join(LC.LLAMA_E_RES, "pairs", pre, f"{lm}__{vit}")
                rec = json.load(open(b + ".json"))
                rec["lm"], rec["standin_of"] = sl, lm
                json.dump(rec, open(os.path.join(de, f"{sl}__{vit}.json"), "w"))
                shutil.copyfile(b + ".npz", os.path.join(de, f"{sl}__{vit}.npz"))
                shutil.copyfile(os.path.join(LLAMA_I, pre, "pairs", f"{lm}__{vit}.npz"), os.path.join(di, f"{sl}__{vit}.npz"))
    for f in ("summary.json", "verdicts_pyt.json"):
        shutil.copyfile(os.path.join(src_fx, "early", f), os.path.join(root, "early", f))
    V = json.load(open(os.path.join(src_fx, "early", "verdicts_pyt.json")))
    V["impl"] = "I (test copy of E)"
    if perturb:
        V["v"][perturb] = (V["v"][perturb] or 0.0) + 0.5
        V["impl"] += f", planted mismatch at {perturb}"
    json.dump(V, open(os.path.join(root, "indep", "verdicts_pyt.json"), "w"))
    print(f"V-P5 fixture under {root} (perturb={perturb})", flush=True)


if __name__ == "__main__":
    main()
