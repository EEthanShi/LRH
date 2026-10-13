"""E0 design inputs (role e0-design, 2026-10-10). EXPLORATORY design record, not a frozen artifact.

Reads only: Hugging Face Hub metadata (token=False, no file download to disk), one HTTP HEAD on the
official COCO annotation archive, and the frozen S1.1 v1.1 outputs. Computes no score involving a
sentence encoder (none exists). Writes analysis/e0_sentence/design/design_inputs.json.
"""
import hashlib, json, os, time, urllib.request
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from huggingface_hub import HfApi

P = os.path.expanduser("~/hrh_v2_stage1_20261008")
OUT = os.path.join(P, "analysis", "e0_sentence", "design", "design_inputs.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


rec = {"role": "e0-design", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "note": "exploratory design inputs; the freeze role recomputes every number it adopts"}

# 1. Hub pins (metadata only)
api = HfApi()
need = {
    "sentence-transformers/all-mpnet-base-v2": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.txt", "modules.json", "1_Pooling/config.json", "sentence_bert_config.json",
        "config_sentence_transformers.json", "README.md", "data_config.json"],
    "facebook/contriever": ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.txt", "README.md"],
    "Qwen/Qwen3-Embedding-0.6B": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "modules.json", "1_Pooling/config.json", "config_sentence_transformers.json",
        "generation_config.json", "README.md"],
}
hub = {}
for repo, files in need.items():
    i = api.model_info(repo, token=False, files_metadata=True)
    sib = {s.rfilename: s for s in i.siblings}
    hub[repo] = {"revision": i.sha, "gated": i.gated, "private": i.private, "disabled": i.disabled,
                 "last_modified": str(i.last_modified), "library": i.library_name, "pipeline_tag": i.pipeline_tag,
                 "license": (i.card_data.to_dict().get("license") if i.card_data else None),
                 "files": {f: {"size": sib[f].size, "lfs_sha256": (sib[f].lfs.sha256 if sib[f].lfs else None),
                               "git_blob": sib[f].blob_id} for f in files},
                 "download_bytes": int(sum(sib[f].size for f in files))}
rec["hub_pins"] = hub
try:
    d = api.dataset_info("yerevann/coco-karpathy", token=False, files_metadata=True)
    rec["caption_fallback_hf"] = {"repo": "yerevann/coco-karpathy", "revision": d.sha, "gated": d.gated,
        "files": {s.rfilename: {"size": s.size, "lfs_sha256": (s.lfs.sha256 if s.lfs else None)} for s in d.siblings
                  if s.rfilename.endswith(".parquet")}}
except Exception as e:  # noqa: BLE001
    rec["caption_fallback_hf"] = {"error": repr(e)}

# 2. official COCO caption archive (HEAD only)
u = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
try:
    h = urllib.request.urlopen(urllib.request.Request(u, method="HEAD"), timeout=30)
    rec["caption_source_official"] = {"url": u, "status": h.status, "etag_md5": h.headers.get("ETag", "").strip('"'),
        "content_length": int(h.headers.get("Content-Length")), "last_modified": h.headers.get("Last-Modified"),
        "member": "annotations/captions_val2014.json"}
except Exception as e:  # noqa: BLE001
    rec["caption_source_official"] = {"url": u, "error": repr(e)}

# 3. S1.1 frozen outputs used by the predictions
files = {k: os.path.join(P, v) for k, v in {
    "cells_clip": "results/S1_1/v11/S1_1_v11_cells_clip.parquet", "slopes_clip": "results/S1_1/v11/S1_1_v11_slopes_clip.parquet",
    "cells_l2": "results/S1_1/v11/S1_1_v11_cells_l2.parquet", "slopes_l2": "results/S1_1/v11/S1_1_v11_slopes_l2.parquet",
    "summary": "results/S1_1/v11/S1_1_v11_summary.json", "items": "features/coco_pool/items.parquet"}.items()}
rec["s11_inputs_sha256"] = {k: sha(v) for k, v in files.items()}
summ = json.load(open(files["summary"]))
cal = {}
for cfg in ("clip", "l2"):
    c = pd.read_parquet(files[f"cells_{cfg}"]); S = pd.read_parquet(files[f"slopes_{cfg}"])
    c = c[c.primary]; S = S[S.primary]
    out = {}
    for k in (1, 10, 100):
        t = c[(c.m == 1024) & (c.k == k)].set_index("pair")[["cls", "obs"]].join(S[S.k == k].set_index("pair")[["slope_obs"]])
        x = np.log(t.obs.to_numpy(float)); y = t.slope_obs.to_numpy(float)
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)
        r = y - (beta[0] + beta[1] * x)
        vl = t[t.cls == "cross-modal"]
        out[str(k)] = {"n_pairs": int(len(t)), "a": float(beta[0]), "b": float(beta[1]),
                       "resid_sd_ddof2": float(r.std(ddof=2)), "resid_max_abs": float(np.abs(r).max()),
                       "resid_mean_by_class": {cl: float(r[(t.cls == cl).to_numpy()].mean()) for cl in ("seed-seed", "cross-vision", "cross-modal")},
                       "vl_median_slope": float(vl.slope_obs.median()), "vl_level_min": float(vl.obs.min()), "vl_level_max": float(vl.obs.max()),
                       "vl_level_median": float(vl.obs.median()),
                       "vv_level_median": float(t[t.cls == "cross-vision"].obs.median()), "vv_median_slope": float(t[t.cls == "cross-vision"].slope_obs.median()),
                       "spearman_level_slope_vl15": float(spearmanr(vl.obs, vl.slope_obs).correlation),
                       "spearman_level_slope_all29": float(spearmanr(t.obs, t.slope_obs).correlation),
                       "vl_pairs": {p: {"level_m1024": float(row.obs), "slope": float(row.slope_obs)} for p, row in vl.iterrows()}}
    el = pd.DataFrame(summ["configs"][cfg]["secondary"]["primary"]["elasticity_per_pair"])
    out["elasticity_sum"] = {cl: {"median": float(g["sum"].median()), "median_abs": float(g["sum"].abs().median()),
                                  "max_abs": float(g["sum"].abs().max()), "n": int(len(g))} for cl, g in el.groupby("cls")}
    cal[cfg] = out
rec["s11_level_slope"] = cal
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(rec, f, indent=1)
print(OUT, sha(OUT))
