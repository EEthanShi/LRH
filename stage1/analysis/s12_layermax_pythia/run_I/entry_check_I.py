"""Role pyt-run-I (Pythia axis, implementation I run): read-only entry/exit check (plan F6).

Re-hashes every frozen input this role depends on and prints one JSON object. Writes nothing except the JSON given
with --out (inside results/S1_2_layermax_pythia/logs/). Not part of the frozen code; it computes no score.
"""
import hashlib, json, os, re, stat, sys, time

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
AX = os.path.join(PROJ, "analysis", "s12_layermax_pythia")
PLAN_SHA = "6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b"
MAN_SHA = "ca600c80b94277328da6f9c2ee8e26785776bbfd0d80f8669b0a050098dcb237"
FEAT = {  # STATUS lines 273-279 (S4)
    "70m": "d5b4dfe588fcb515ac318bb463294b6909cb822378d0ebcd536311666b853155",
    "160m": "48f8fd73e31ffacb25e46808f2cac083d2db57bd0428fa49027aee241206cae3",
    "410m": "b15799a0c8c0ac4689dacbc970d578b0fe1605e65cfc856be9f3edaa72545486",
    "1b": "4a01f4c479b05b912e97fbfba60ef0eac32f5aa12c11b6e6f5a66470edadd5b2",
    "1.4b": "196df06dbb2dc2280f24d3a66058b3cb43cba1a18f6fbc029da61fbad58d5707",
    "2.8b": "bedcaa241b2efafa1ee318d4a60a94c224739a80390d491f7a1fe5f60b295742",
    "6.9b": "5ef38085a07d0e3d7809035bae8006519ea104ed1f299d6dab27039e53e5fa07",
}
PARTB = {  # STATUS line 285
    "analysis/s12_layermax_pythia/PARTB_PREDICTIONS_PYT.md": "64a5b83a5ae67413168d612bfe447bcd43e20fd6c5afd2d7e662b8489699d600",
    "results/S1_2_layermax_pythia/partB/partB_features_pyt.json": "a8d6282945d7eeaa64095bf5771100edc21d0441ede18b84a0410f0fd9cd7b6a",
    "analysis/s12_layermax_pythia/partb_pyt.py": "f304677ab9545980bdea8eea1363c77bd1ab5dd920687d7fd8613c742c5ef4de",
    "results/S1_2_layermax_pythia/partB/clip_thresholds_partB_pyt.json": "5de400480d892f9243a78452d239dacf9aa0bd3d6ea7e136ed703dee01375454",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "role": "pyt-run-I"}
    ok = True
    out["plan_sha256"] = sha(os.path.join(AX, "ANALYSIS_PLAN_PYTHIA.md"))
    ok &= out["plan_sha256"] == PLAN_SHA
    man_p = os.path.join(AX, "S0B_CODE_MANIFEST.json")
    out["manifest_sha256"] = sha(man_p)
    ok &= out["manifest_sha256"] == MAN_SHA
    man = json.load(open(man_p))
    bad, notro = [], []
    for e in man["files"]:
        p = os.path.join(PROJ, e["path"])
        if not os.path.isfile(p) or sha(p) != e["sha256"]:
            bad.append(e["path"])
        elif os.stat(p).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            notro.append(e["path"])
    out["manifest_files"] = len(man["files"])
    out["manifest_mismatch"] = bad
    out["manifest_writable"] = notro
    ok &= not bad and not notro
    lines = open(os.path.join(PROJ, "STATUS.md"), encoding="utf-8").read().splitlines()
    ts = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ "
    cf = [i + 1 for i, l in enumerate(lines) if re.match(ts + r"\[pyt-axis-[A-Za-z0-9-]+\] CODE FROZEN", l)]
    pb = [i + 1 for i, l in enumerate(lines) if re.match(ts + re.escape("[pyt-axis-predict] Part B FROZEN"), l)]
    ex = [i + 1 for i, l in enumerate(lines) if re.match(ts + re.escape("[pyt-axis-extract-run] EXTRACTION DONE"), l)]
    out["status_lines"] = len(lines)
    out["status_sha256"] = sha(os.path.join(PROJ, "STATUS.md"))
    out["code_frozen_lines"] = cf
    out["partB_frozen_lines"] = pb
    out["extraction_done_lines"] = ex
    ok &= cf == [247] and pb == [285] and ex == [282]
    out["partB"] = {}
    for rel, want in PARTB.items():
        got = sha(os.path.join(PROJ, rel))
        out["partB"][rel] = got
        ok &= got == want
    out["features"] = {}
    for s, want in FEAT.items():
        got = sha(os.path.join(PROJ, "features", "wit1024_pythia_axis", f"EleutherAI__pythia-{s}.npy"))
        out["features"][s] = got
        ok &= got == want
    out["pass"] = bool(ok)
    s = json.dumps(out, indent=1)
    if len(sys.argv) > 2 and sys.argv[1] == "--out":
        dst = os.path.join(PROJ, sys.argv[2])
        assert dst.startswith(os.path.join(PROJ, "results", "S1_2_layermax_pythia", "logs") + os.sep)
        assert not os.path.exists(dst), dst
        with open(dst, "w") as f:
            f.write(s + "\n")
        print("written", dst, sha(dst))
    print(s if not out["pass"] else json.dumps({k: out[k] for k in ("utc", "pass", "plan_sha256", "manifest_sha256",
          "manifest_files", "status_lines", "status_sha256", "code_frozen_lines", "partB_frozen_lines")}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
