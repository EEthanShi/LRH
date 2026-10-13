"""PYT-AXIS (new, implementation E test helper, S0b only): small SYNTHETIC feature stacks for the 7 stand-in slugs, for
CPU smoke tests of E's pair-level scripts (precompute.py, run_layermax.py, validate.py). Runs only in the S0b test mode
and writes only under the E test root. Reads no Pythia byte (none exists) and no cross-model score.

Writes (root = $PYT_E_TEST_ROOT): synth_feats/<size>.{npy,json} (float32 [1024, L, d], seeded rng
"S1.2-layermax-pythia|E-test|synth|<size>", L = 3..6 stored layers, d = 16..40), standins.json, and
partB/clip_thresholds_partB_pyt.json with hrhv2.prep.prh_clip_threshold of each synthetic stack (so that V-P1 has a
reference; this is not a Part B).
usage: PYT_S0B_TEST=1 PYT_E_TEST_ROOT=<root> python tests/make_synth_stacks.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402
from hrhv2 import util  # noqa: E402
from hrhv2.prep import prh_clip_threshold  # noqa: E402

SHAPES = {"70m": (3, 16), "160m": (4, 20), "410m": (5, 24), "1b": (4, 28), "1.4b": (5, 28), "2.8b": (6, 32), "6.9b": (6, 40)}


def main():
    if not LC.TEST:
        LC._refuse("make_synth_stacks.py runs only with PYT_S0B_TEST=1 (S0b test mode)")
    root = LC.PYT_RES
    assert LC._inside(root, LC.E_TEST_BASE), root
    out = os.path.join(root, "synth_feats")
    os.makedirs(out, exist_ok=True)
    stand, clip = {}, {}
    for i, (s, (L, d)) in enumerate(SHAPES.items()):
        rng = util.rng_from_key(f"S1.2-layermax-pythia|E-test|synth|{s}")
        base = rng.standard_normal((1024, d))
        X = np.stack([base + 0.5 * rng.standard_normal((1024, d)) * (l + 1) for l in range(L)], axis=1).astype(np.float32)
        b = os.path.join(out, s)
        np.save(b + ".npy", X)
        json.dump({"model_id": f"synthetic stand-in {s}", "num_params": int(10 ** (6 + i)), "shape": [1024, L, d],
                   "dtype": "float32", "label": "synthetic (E S0b test)"}, open(b + ".json", "w"))
        stand[s] = b
        clip[LC.SLUG_TEST[s]] = {"c": float(prh_clip_threshold(X.astype(np.float64)))}
    json.dump(stand, open(os.path.join(root, "standins.json"), "w"), indent=1)
    os.makedirs(os.path.join(root, "partB"), exist_ok=True)
    json.dump({"clip": clip, "label": "synthetic clip reference (E S0b test), not a Part B"},
              open(os.path.join(root, "partB", "clip_thresholds_partB_pyt.json"), "w"), indent=1)
    print(f"synthetic stacks under {out}", flush=True)


if __name__ == "__main__":
    main()
