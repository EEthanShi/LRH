import hashlib, json, os, glob, sys
os.chdir(os.path.expanduser("~/hrh_v2_stage1_20261008"))
files = sorted(glob.glob("analysis/s12_layermax_llama/early/*.py") + glob.glob("analysis/s12_layermax_llama/early/*.sh")
               + glob.glob("analysis/s12_layermax_llama/early/*.md") + glob.glob("analysis/s12_layermax_llama/early/*.txt")
               + glob.glob("results/S1_2_layermax_llama/early/validation/*.json") + ["results/S1_2_layermax_llama/early/summary.json",
               "results/S1_2_layermax_llama/early/cache/clip.json", "results/S1_2_layermax_llama/cache/clip_thresholds.json"])
def h(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            d.update(b)
    return d.hexdigest()
out = {p: h(p) for p in files if os.path.exists(p)}
pairs = sorted(glob.glob("results/S1_2_layermax_llama/early/pairs/*/*"))
d = hashlib.sha256()
for p in pairs:
    d.update(p.encode()); d.update(h(p).encode())
out[f"results/S1_2_layermax_llama/early/pairs/ ({len(pairs)} files, sha256 over sorted path+sha256)"] = d.hexdigest()
json.dump(out, open("results/S1_2_layermax_llama/early/hashes.json", "w"), indent=1)
print(json.dumps(out, indent=1))
