"""PYT-AXIS copy of write_hashes.py (implementation E, S0b sub-role pyt-code-E): sha256 of E's code
and outputs. GATED (plan F1(a); it reads, i.e. hashes, the pair files).

Copied from analysis/s12_layermax_openllama/early/write_hashes.py (sha256 a49a206e...). Changes, marked "PYT-AXIS":
paths of this axis (code analysis/s12_layermax_pythia/early/, outputs lm_core.RES); the E gate first; the output
names of this axis (verdicts_pyt.json, validation files VP*); the pair-file digest of E (sorted relative path +
sha256, as before), and the same digest of the frozen E pair trees read by the descriptive cross-family table and the
18-LM panel (results/S1_2_layermax/pairs, results/S1_2_layermax_llama/early/pairs,
results/S1_2_layermax_openllama/early/pairs).

Run (step S8, last E step, after report_pyt.py; CPU, 1 thread):
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_pythia/launch_pyt.sh pyt_E_hashes 1 -- \\
      venv/bin/python analysis/s12_layermax_pythia/early/write_hashes.py

Output: <RES>/hashes.json
"""
import argparse, hashlib, json, os, glob, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_core as LC  # noqa: E402

argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
LC.gate("early/write_hashes.py")   # PYT-AXIS: plan F1(a), first action
os.chdir(LC.PROJ)
CODE = os.path.relpath(LC.EARLY_DIR, LC.PROJ)  # PYT-AXIS
RES = os.path.relpath(LC.RES, LC.PROJ)
files = sorted(glob.glob(f"{CODE}/*.py") + glob.glob(f"{CODE}/*.md") + glob.glob(f"{CODE}/*.txt") + glob.glob(f"{CODE}/*.diff")
               + glob.glob(f"{CODE}/tests/*.py")  # PYT-AXIS
               + glob.glob(f"{RES}/validation/*.json") + [f"{RES}/summary.json", f"{RES}/verdicts_pyt.json", f"{RES}/REPORT.md",  # PYT-AXIS
               f"{RES}/cache/clip.json", os.path.relpath(os.path.join(LC.SHARED_CACHE, "clip_thresholds.json"), LC.PROJ)])


def h(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            d.update(b)
    return d.hexdigest()


out = {p: h(p) for p in files if os.path.exists(p)}
for label, pat in (("E", f"{RES}/pairs/*/*"),  # PYT-AXIS
                   ("frozen 8-LM E (cross-family table)", "results/S1_2_layermax/pairs/*/*"),  # PYT-AXIS
                   ("LLaMA-axis E (cross-family table)", "results/S1_2_layermax_llama/early/pairs/*/*"),  # PYT-AXIS
                   ("OpenLLaMA-axis E (cross-family table)", "results/S1_2_layermax_openllama/early/pairs/*/*")):  # PYT-AXIS
    pairs = sorted(glob.glob(pat))
    d = hashlib.sha256()
    for p in pairs:
        d.update(p.encode()); d.update(h(p).encode())  # noqa: E702
    out[f"{os.path.dirname(os.path.dirname(pat))}/ [{label}] ({len(pairs)} files, sha256 over sorted path+sha256)"] = d.hexdigest()
json.dump(out, open(f"{RES}/hashes.json", "w"), indent=1)
print(json.dumps(out, indent=1))
