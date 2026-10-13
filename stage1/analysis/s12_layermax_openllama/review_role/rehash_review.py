#!/usr/bin/env python
"""OpenLLaMA axis, role oll-review (plan ANALYSIS_PLAN_OPENLLAMA.md 4f18d710, step S8 review part).
F6 re-hash at role start and role end. NEW helper written by the review role (not in the S0b manifest).
Read-only: it hashes frozen inputs, the 106 S0b code files and every E/I/forecast/joint/validation output whose
sha256 was logged in STATUS by an earlier role, and digests the pair trees. It writes one JSON under
results/S1_2_layermax_openllama/review/role/.

usage: venv/bin/python -I analysis/s12_layermax_openllama/review_role/rehash_review.py <start|end>
exit 0 iff every hash equals its STATUS-logged value.
"""
import hashlib
import json
import os
import sys
import time

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
OUT = os.path.join(PROJ, "results", "S1_2_layermax_openllama", "review", "role")
MAN = "analysis/s12_layermax_openllama/S0B_CODE_MANIFEST.json"
R = "results/S1_2_layermax_openllama/"
A = "analysis/s12_layermax_openllama/"
# value = full sha256 when STATUS gives it, else the logged prefix
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
    A + "ANALYSIS_PLAN_OPENLLAMA.md": "4f18d710beda45492b1e191f74580bfc4ef53f6dafe791ad245ff1cf335f3240",
    A + "PARTB_PREDICTIONS_OL.md": "f3527d09fedf59270180e64af7265d26ff59039031725e7b8d5e38e671208865",
    R + "partB/partB_features_ol.json": "e1104bd71e5eff9ced65ed402a61c187590bce5dee306c25ee2f64b18fdd7e51",
    R + "partB/clip_thresholds_partB_ol.json": "f9805138afef0bba302bec863387e611ca3b57e5fe8ae2c8cbcaa91879c0c908",
    R + "partB/partB_retro_checks_ol.json": "d2f59113",
    R + "partB/partB_verify_ol.json": "4cf566d9020911621d3c02470e75ebc5b3259df2f4f88b2a50e05267e5ebfcd8",
    A + "partb_ol.py": "4aab8ee1942cb0e94307306b0ccd9d450102d6bcd41236370f614b3192b8aef6",
    A + "partB_verify/verify_partb_ol.py": "8eca20178be25bc986706384a6711202304f7ee6a89d7f07c2751ea437c32349",
    "features/wit1024/openlm-research__open_llama_3b.npy": "f6f966ca8bb50301a8e1f9e317cb1bc36bdfa67dbfcc68430de228274b6e9899",
    "features/wit1024/openlm-research__open_llama_7b.npy": "3c7da1c40b319efb1f06034a79756e3c51fe0313cabd34fa0848750fbb60ddd0",
    "features/wit1024_openllama_axis/openlm-research__open_llama_13b.npy": "6132997edd01833490b8ee2f12b5f2dc315c84941e74ae19c54270ee9e8a9c8b",
    "features/wit1024_openllama_axis/openlm-research__open_llama_13b.lm_aux.npz": "ac8b79cb37640d4440debf4ca7f995c4314b59c0b54543918157d6333bf46c08",
    "features/wit1024_openllama_axis/openlm-research__open_llama_13b.json": "2b8773f97a37301b785f521ba45f505687450665a1a6b66ab2099d70d5b021a1",
    R + "extraction/EXTRACTION_SUMMARY.json": "a343f3a55ad3165e5ef18fd877943b8a23068040726461735201014835a6ee26",
    R + "feasibility/FEASIBILITY_OL.md": "f4fa89f8c1925b4cd2a5b2f8dad7441115221ece02884b552eef7ce523aeb61a",
    R + "feasibility/feasibility_ol.json": "9654006dbf171bdd2fe26469101f8653303c5a425322ced4eed22d4764819f9e",
    MAN: "b2b553622a97e042488ace043adde592c1b6f1128bddf252f8a865bc9dafed4d",
    # outputs of the run roles (sha256 as logged in STATUS 19:49Z-20:01Z)
    R + "cache/clip_thresholds.json": "da7873faa72a473c7477632dbb563238590450da323e6118f6f46989476d0ff9",
    R + "early/summary.json": "b302119c4fff9ead88f7418fa4e01126f2b621f653691a7cc6cf2e328925890d",
    R + "early/verdicts_ol.json": "0c7f34b59937a6ed6ed154570126c76f726d559f8962c3e2fc1d9aa8f0bf1f8f",
    R + "early/REPORT.md": "85a5f2c83cf02b2d168083dfd3b18141ee043f8822434f61712a09e81a815ed5",
    R + "early/hashes.json": "301271ee0eb1cd8dcbb94942f0014029573828305ebd5e4927789cf653586c36",
    R + "early/validation/VO1_VO2_knn_tables.json": "ad794dce63134f6a361a00537df84393feb83dd56e0c965c2f1c2e8713a2e842",
    R + "early/validation/VO3_VO4a.json": "996c442f6c7f0f98e00533ee151e868d9dd26c69323e08483e693f082600b6be",
    R + "early/validation/VO4b_3b7b.json": "0d680fa310a5d6f4ff1a4ada63d6b331dfaf5ffac6d16f2a9ade595a5368e09f",
    R + "early/validation/VO7_prop5.json": "acdea068b81e6771a97bd668e0d3e7c0f34f47f4be189b2430396b41b5cfed58",
    R + "early/validation/VO5_pairs.json": "db8103696e4f6c95ab0c37ec09621a2fa262b90655559c74ba74d51ff89c57aa",
    R + "early/validation/VO5_verdicts.json": "6c9f9af35dbffa9f9d87ecfe80ee5f8a4be16f1569606321865f7eee6ad5a4c9",
    A + "early/interpretation.txt": "f4be32a45de3617d8d585298ab4e8cedbfd2ff92abc675ce3668cd6c9e6f74be",
    R + "forecast/E/verdicts_forecast.json": "54055939adfc22dd9bf0e89fe3fa138b4b143522a786e5db50ff7bb38ba82263",
    R + "forecast/E/forecast_detail.json": "2ac224c9c1f9e37bcf7f8c74cc86d4fcdc28409b1f96f7bdf39ac78f409e34da",
    R + "forecast/I/verdicts_forecast.json": "87fc0ab8aa80cf6f2620329a8e159524daa836961889f893ec1b294856097b36",
    R + "forecast/I/forecast_detail.json": "5f1b1008",
    R + "joint/E/verdicts_joint.json": "be2e6ee59f5187d7665bae9e52b3248a66ba5a6bada6a8d805831517d888fe5e",
    R + "joint/E/joint_detail.json": "6e1168bbc9c71d24ede66b3762ccc07586311e16fabd96c6d42ec877d0b10124",
    R + "joint/I/verdicts_joint.json": "204ed4acab9ed39c143bd668de8d8a827c6ca08a4b1e2b71060694dc777ec00a",
    R + "joint/I/joint_detail.json": "3a8f645d",
    R + "indep/summary.json": "7b60cf8c8ca646a263da658220f3f3eaf7477dc28b96e18b5df95d6bd7dd0fd9",
    R + "indep/verdicts_ol.json": "58fcc80b6ffca7cecc44cf1951e0bbf6d6e611fe6305b28c4a04ade78ade6967",
    R + "indep/REPORT.md": "4a6bdc4e10114b61125d369e9c6e49a6f2a4793fd0205738a9787815529ba389",
    R + "indep/REPORT.pre_cross.md": "eb2c42ecb8a5aa7617a19cf10cbc0dc8b007bc9641696d1e9c96600744c93c28",
    R + "indep/validation_vl.json": "d15f96af6cfc3e36199e247cf0bb34b0cd6cf2b4e4d73c3dde13d152c86bbb6f",
    R + "indep/validation.json": "e553a37a47646a7db2002a6ba22f179187e2f9c7036593f5ba19d24fb840cf8e",
    R + "indep/validation_vo6_arh.json": "9763236bfa9ceb28ff100461b94520adeb4dedeebd2e236fa716640a357f2f45",
    R + "indep/validation_cross.json": "de95bdb181882fdbddcad1b4eb9331df847f8bc7f9f59176f50ff8063798ae41",
    R + "indep/cross_order.json": "a37bf0a7f4c7fa6cfa13871c278b80fa3cf2d3239f7963888a243d5105d1ae07",
    R + "indep/exact_mknn_check_I.json": "cc2595dcbdcab0d1abad02c14f1e992d3fdb6290ecb85bbc77003881d3f009df",
    R + "indep/exact_mknn_vs_E.json": "b0fbd361596a4bfc81cf9ab57014d883dc62f423b1ea503c5b8bf77f2ac02fbd",
    R + "indep/f1c_audit_at_I_end.json": "ff9153baf982ec9db58db0671f47d42a616f5dbeed66014f86ed411d71c81224",
    A + "indep_run/hashcheck_I.py": "af2fdd7f8da2ba49",
    A + "indep_run/exact_mknn_check_I.py": "4592e8099bc255e8c8e3b4677731044ea34d902489d7c62ba5523ad41dfd43f1",
    A + "indep_run/exact_mknn_vs_E.py": "38561cc34b6e3032d070654a2df08563265db6ac36b535e4e95c0d43fa62b45d",
}
TREES = [R + "early/pairs", R + "indep/prh_clip+l2/pairs", R + "indep/l2/pairs",
         "results/S1_2_layermax/pairs", "results/S1_2_layermax/indep",
         "results/S1_2_layermax_llama/early/pairs", "results/S1_2_layermax_llama/indep"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def tree_digest(rel):
    root = os.path.join(PROJ, rel)
    h = hashlib.sha256()
    n = 0
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            p = os.path.join(dp, f)
            h.update(os.path.relpath(p, root).encode() + b"\0" + sha(p).encode() + b"\n")
            n += 1
    return {"n_files": n, "digest": h.hexdigest()}


def main():
    tag = sys.argv[1]
    t0 = time.time()
    rows, bad = {}, []
    for rel, ref in FROZEN.items():
        p = os.path.join(PROJ, rel)
        got = sha(p) if os.path.exists(p) else None
        ok = got is not None and got.startswith(ref)
        st = os.stat(p) if os.path.exists(p) else None
        rows[rel] = {"sha256": got, "logged": ref, "equal": ok,
                     "mode": oct(st.st_mode & 0o777) if st else None}
        if not ok:
            bad.append(rel)
    man = json.load(open(os.path.join(PROJ, MAN)))
    code_bad = []
    for f in man["files"]:
        p = os.path.join(PROJ, f["path"])
        got = sha(p) if os.path.exists(p) else None
        if got != f["sha256"]:
            code_bad.append([f["path"], got, f["sha256"]])
    trees = {t: tree_digest(t) for t in TREES}
    res = {"role": "oll-review", "tag": tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "files": rows, "n_files": len(rows), "n_bad": len(bad), "bad": bad,
           "n_code_files": len(man["files"]), "n_code_bad": len(code_bad), "code_bad": code_bad,
           "tree_digests": trees, "pass": not bad and not code_bad, "secs": time.time() - t0}
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"rehash_{tag}.json")
    if os.path.exists(fn):
        sys.exit(f"refusing to overwrite {fn}")
    with open(fn + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(fn + ".tmp", fn)
    os.chmod(fn, 0o444)
    print(json.dumps({k: res[k] for k in ("tag", "utc", "n_files", "n_bad", "bad", "n_code_files", "n_code_bad",
                                           "pass")}))
    print(json.dumps({t: v["n_files"] for t, v in trees.items()}))
    print("output", os.path.relpath(fn, PROJ), sha(fn))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
