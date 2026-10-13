"""Conditional check B7 (ANALYSIS_PLAN_E0.md v1.0 §15): exact-string membership of the pool's caption texts in
the MPNet training dataset `sentence-transformers/coco-captions`. Text only; no model; no score.

    venv/bin/python -I extract/e0_sentence/e0_membership.py

Runs only after a STATUS line containing 'E0 B7 APPROVED'. Downloads the dataset's parquet files at the pinned
revision (token=False) into extract/e0_sentence/hf_home and writes results/E0_sentence/extraction/B7_membership.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e0x_common as X  # noqa: E402

os.environ.setdefault("HF_HOME", X.HF_HOME)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import pandas as pd  # noqa: E402


def string_set(parquet_paths) -> tuple[set, dict]:
    s = set()
    cols = {}
    for p in parquet_paths:
        df = pd.read_parquet(p)
        sc = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
        cols[os.path.basename(p)] = {"string_columns": sc, "rows": int(len(df))}
        for c in sc:
            s.update(str(x).strip() for x in df[c].dropna())
    return s, cols


def membership(strings: set, c1_texts, caption_texts) -> dict:
    c1 = [t.strip() for t in c1_texts]
    caps = [t.strip() for t in caption_texts]
    n1 = sum(t in strings for t in c1)
    n2 = sum(t in strings for t in caps)
    return {"c1_in_training_strings": n1, "c1_total": len(c1), "c1_frac": n1 / max(len(c1), 1),
            "captions_in_training_strings": n2, "captions_total": len(caps), "captions_frac": n2 / max(len(caps), 1)}


def main():
    X.require_plan_and_code_frozen()
    if not X.status_has(r"E0 B7 APPROVED"):
        raise SystemExit("REFUSED: B7 needs a STATUS line 'E0 B7 APPROVED' (plan §15)")
    from huggingface_hub import snapshot_download
    repo, rev = X.B7_REPO
    snap = snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=["*.parquet"],
                             cache_dir=X.HUB_DIR, token=False)
    files = sorted(glob.glob(os.path.join(snap, "**", "*.parquet"), recursive=True))
    strings, cols = string_set(files)
    items = pd.read_parquet(X.ITEMS, columns=["idx", "text"])
    index = pd.read_parquet(os.path.join(X.FEAT, "captions_index.parquet"))
    res = membership(strings, items["text"].tolist(), index["text"].tolist())
    res.update({"repo": repo, "revision": rev, "files_sha256": {os.path.relpath(p, snap): X.sha256_file(p) for p in files},
                "columns": cols, "n_distinct_strings": len(strings), "utc": X.utcnow()})
    h = X.write_json_new(os.path.join(X.EXTRACTION, "B7_membership.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("files_sha256", "columns")}, indent=1), h)


if __name__ == "__main__":
    main()
