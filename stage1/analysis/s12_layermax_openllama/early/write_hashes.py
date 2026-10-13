"""OL-AXIS copy of write_hashes.py (implementation E, S0b sub-role E of oll-axis-extract): sha256 of E's code
and outputs. GATED (plan F1(a); it reads, i.e. hashes, the pair files).

Copied from analysis/s12_layermax_llama/early/write_hashes.py (sha256 2c107c4e...). Changes, marked "OL-AXIS":
paths of this axis (code analysis/s12_layermax_openllama/early/, outputs lm_core.RES); the Part B gate first;
the pair-file digest of E (sorted relative path + sha256, as before) and, for plan section 3 / R.3.7, the
204-file digest of the LLaMA-axis E pair files read by the pooled panel (expected 0d918377...).

Run (step S8, last E step, after report_ol.py; CPU, 1 thread):
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_openllama/launch_ol.sh ol_E_hashes 1 -- \\
      venv/bin/python analysis/s12_layermax_openllama/early/write_hashes.py

Output: <RES>/hashes.json
"""
import argparse, hashlib, json, os, glob, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_core as LC  # noqa: E402
import guard  # noqa: E402

argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
guard.require_partb_frozen(LC.RES, script="early/write_hashes.py")   # plan F1(a): first action
os.chdir(LC.PROJ)
CODE = "analysis/s12_layermax_openllama/early"
RES = os.path.relpath(LC.RES, LC.PROJ)
files = sorted(glob.glob(f"{CODE}/*.py") + glob.glob(f"{CODE}/*.md") + glob.glob(f"{CODE}/*.txt") + glob.glob(f"{CODE}/*.diff")
               + glob.glob(f"{RES}/validation/*.json") + [f"{RES}/summary.json", f"{RES}/verdicts_ol.json", f"{RES}/REPORT.md",
               f"{RES}/cache/clip.json", os.path.relpath(os.path.join(LC.SHARED_CACHE, "clip_thresholds.json"), LC.PROJ)])


def h(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            d.update(b)
    return d.hexdigest()


out = {p: h(p) for p in files if os.path.exists(p)}
for label, pat in (("E", f"{RES}/pairs/*/*"), ("LLaMA-axis E (read by the pooled panel)",
                                              "results/S1_2_layermax_llama/early/pairs/*/*")):
    pairs = sorted(glob.glob(pat))
    d = hashlib.sha256()
    for p in pairs:
        d.update(p.encode()); d.update(h(p).encode())  # noqa: E702
    out[f"{os.path.dirname(os.path.dirname(pat))}/ [{label}] ({len(pairs)} files, sha256 over sorted path+sha256)"] = d.hexdigest()
json.dump(out, open(f"{RES}/hashes.json", "w"), indent=1)
print(json.dumps(out, indent=1))
