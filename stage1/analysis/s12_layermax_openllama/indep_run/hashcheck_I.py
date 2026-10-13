#!/usr/bin/env python
"""OpenLLaMA axis, role oll-run-indep (implementation I, plan S6-S8 I side): F6 re-hash of all frozen inputs
at role start and role end. NEW helper (not in the S0b manifest); it reads only frozen inputs and the S0b
code files (no cross-model score), and writes one JSON under results/S1_2_layermax_openllama/indep/.

usage: venv/bin/python -I analysis/s12_layermax_openllama/indep_run/hashcheck_I.py <tag>
       (tag = start | end); exit 0 iff every hash equals its frozen value.
"""
import hashlib
import json
import os
import sys
import time

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
OUT = os.path.join(PROJ, "results", "S1_2_layermax_openllama", "indep")
MAN = "analysis/s12_layermax_openllama/S0B_CODE_MANIFEST.json"
MAN_SHA = "b2b553622a97e042488ace043adde592c1b6f1128bddf252f8a865bc9dafed4d"
# frozen inputs: full sha256 where STATUS gives it, otherwise the logged prefix
FROZEN = {
    "SPEC.md": "e781e44f",
    "prereg/PREREG_stage1.md": "bee0583c",
    "extract/PROTOCOL.md": "e013f98b",
    "extract/registry.py": "5e95fe65",
    "extract/registry_llama_axis.py": "1af88e931897a07a1a64c909733b64ffe15d0e84dfd2efe84fbc5b0a9daa7f95",
    "analysis/s12_layermax/ANALYSIS_PLAN_LAYERMAX.md": "dd1897214d449aaa0e48fb12d5ba43364e015bb16b67508afa221b143431a594",
    "analysis/s12_layermax/indep/PLAN_INDEP.md": "085746a823b23bfe19f9a3c1d85b354a6beec6dd71c6cca802c7cf98a0ab8817",
    "analysis/s12_layermax_llama/ANALYSIS_PLAN_LLAMA.md": "990428e3f29c7ecfba9cf314a45c73c79f0154f86a07520429910d282c0a2fc8",
    "analysis/s12_layermax_llama/PARTB_PREDICTIONS.md": "f9afa919957d01eed8373253e710baa6b7d22c3d613099cabc9f6107f8bec5c4",
    "results/S1_2_layermax_llama/partB/partB_features.json": "ede0127589a02037427b343e9c3593ee427d430e9746d7f4942b15f76a3031bf",
    "results/S1_2_layermax_llama/layermax_llama_summary.json": "db9242f9c5734434c8e5eb0d23204b9e755c4d49cd6636f1d7da58b7e8fc7e7e",
    "results/S1_2_layermax_llama/DONE": "3fb5c0b23689cf9d0d1525316053b3a4f3c46abfd42231443215c47358a80f12",
    "results/S1_2_layermax/indep/layermax_indep_summary.json": "40380e80f417e0da4c88788a7e03ae900c759a2cad589df6b97749f3435651a9",
    "analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md": "4f18d710beda45492b1e191f74580bfc4ef53f6dafe791ad245ff1cf335f3240",
    "analysis/s12_layermax_openllama/PARTB_PREDICTIONS_OL.md": "f3527d09fedf59270180e64af7265d26ff59039031725e7b8d5e38e671208865",
    "results/S1_2_layermax_openllama/partB/partB_features_ol.json": "e1104bd71e5eff9ced65ed402a61c187590bce5dee306c25ee2f64b18fdd7e51",
    "results/S1_2_layermax_openllama/partB/clip_thresholds_partB_ol.json": "f9805138afef0bba302bec863387e611ca3b57e5fe8ae2c8cbcaa91879c0c908",
    "features/wit1024/openlm-research__open_llama_3b.npy": "f6f966ca8bb50301a8e1f9e317cb1bc36bdfa67dbfcc68430de228274b6e9899",
    "features/wit1024/openlm-research__open_llama_7b.npy": "3c7da1c40b319efb1f06034a79756e3c51fe0313cabd34fa0848750fbb60ddd0",
    "features/wit1024_openllama_axis/openlm-research__open_llama_13b.npy": "6132997edd01833490b8ee2f12b5f2dc315c84941e74ae19c54270ee9e8a9c8b",
    MAN: MAN_SHA,
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    tag = sys.argv[1]
    t0 = time.time()
    rows, bad = {}, []
    for rel, ref in FROZEN.items():
        p = os.path.join(PROJ, rel)
        got = sha(p) if os.path.exists(p) else None
        ok = got is not None and got.startswith(ref)
        rows[rel] = {"sha256": got, "frozen": ref, "equal": ok}
        if not ok:
            bad.append(rel)
    man = json.load(open(os.path.join(PROJ, MAN)))
    code_bad = []
    for f in man["files"]:
        p = os.path.join(PROJ, f["path"])
        got = sha(p) if os.path.exists(p) else None
        if got != f["sha256"]:
            code_bad.append([f["path"], got, f["sha256"]])
    res = {"role": "oll-run-indep", "tag": tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "frozen_inputs": rows, "n_frozen": len(rows), "n_frozen_bad": len(bad), "bad": bad,
           "n_code_files": len(man["files"]), "n_code_bad": len(code_bad), "code_bad": code_bad,
           "pass": not bad and not code_bad, "secs": time.time() - t0}
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"role_hashcheck_{tag}.json")
    with open(fn + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(fn + ".tmp", fn)
    print(json.dumps({k: res[k] for k in ("tag", "utc", "n_frozen", "n_frozen_bad", "bad", "n_code_files",
                                           "n_code_bad", "pass")}))
    print("output", fn, sha(fn))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
