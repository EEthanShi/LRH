"""E0 captions: pinned official COCO annotations -> caption index of the 16,384 pool images, with checks V-e
(ANALYSIS_PLAN_E0.md v1.0 §3.2, §3.3). Role e0-extract. No model, no score.

    venv/bin/python -I extract/e0_sentence/e0_captions.py

1. Download annotations_trainval2014.zip (https, http on a connection error only) into
   extract/e0_sentence/coco_captions/; accept only size 252,872,794 B and md5 0a379cfc70b0e71301e0f377548639bd.
2. Read only the member annotations/captions_val2014.json (zipfile; nothing executed) and save it.
3. features/coco_e0/captions_index.parquet: cap_row, idx (pool row), image_id, ann_id, rank (0-based by ann id),
   text (.strip()); features/coco_e0/captions_checks.json. Exit 3 (STOP) if check (a) or (b) fails.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e0x_common as X  # noqa: E402


def download(dst: str, urls=X.COCO_ZIP_URLS, size=X.COCO_ZIP_SIZE, md5=X.COCO_ZIP_MD5, opener=urllib.request.urlopen) -> dict:
    if os.path.exists(dst):
        raise FileExistsError(f"{dst} exists; verify it with verify_zip() instead of downloading again")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    if os.path.exists(tmp):
        raise FileExistsError(f"stale partial download {tmp}; inspect it (E0 code deletes nothing)")
    last = None
    for i, url in enumerate(urls):
        try:
            h5, h256, n = hashlib.md5(), hashlib.sha256(), 0
            with opener(url, timeout=120) as r, open(tmp, "wb") as f:
                for b in iter(lambda: r.read(1 << 22), b""):
                    f.write(b)
                    h5.update(b)
                    h256.update(b)
                    n += len(b)
            if n != size or h5.hexdigest() != md5:
                os.replace(tmp, dst + f".rejected_{X.utcnow().replace(':', '')}")
                raise SystemExit(f"STOP: {url}: size {n} / md5 {h5.hexdigest()} do not match the pin")
            os.replace(tmp, dst)
            return {"url": url, "size": n, "md5": h5.hexdigest(), "sha256": h256.hexdigest(), "utc": X.utcnow()}
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            last = e
            if os.path.exists(tmp):
                os.replace(tmp, dst + f".failed_{i}_{X.utcnow().replace(':', '')}")
            continue
    raise SystemExit(f"STOP: official COCO host failed ({last!r}); the mirror needs an addendum (plan Q6)")


def verify_zip(path: str, size=X.COCO_ZIP_SIZE, md5=X.COCO_ZIP_MD5) -> dict:
    h5, h256 = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h5.update(b)
            h256.update(b)
    n = os.path.getsize(path)
    if n != size or h5.hexdigest() != md5:
        raise SystemExit(f"STOP: {path}: size {n} / md5 {h5.hexdigest()} do not match the pin")
    return {"size": n, "md5": h5.hexdigest(), "sha256": h256.hexdigest()}


def extract_member(zpath: str, dst: str, member: str = X.COCO_MEMBER) -> str:
    if os.path.exists(dst):
        raise FileExistsError(dst)
    with zipfile.ZipFile(zpath) as z:
        data = z.read(member)
    X.write_new(dst, data)
    return X.sha256_file(dst)


def build_index(captions_json: dict, items: pd.DataFrame):
    """Caption rows for the pool images, in pool order then ascending annotation id; and the V-e checks."""
    items = items.sort_values("idx").reset_index(drop=True)
    assert np.array_equal(items["idx"].to_numpy(), np.arange(len(items)))
    by_img = {}
    for a in captions_json["annotations"]:
        by_img.setdefault(int(a["image_id"]), []).append((int(a["id"]), a["caption"]))
    rows = []
    missing = []
    eq_any = eq_any_norm = eq_first = 0
    hist = Counter()

    def norm(s):
        return re.sub(r"\s+", " ", s.strip()).lower()
    for r in items.itertuples(index=False):
        caps = sorted(by_img.get(int(r.image_id), []))
        if not caps:
            missing.append(int(r.image_id))
            continue
        hist[len(caps)] += 1
        texts = [c.strip() for _, c in caps]
        eq_any += r.text in texts
        eq_any_norm += norm(r.text) in {norm(t) for t in texts}
        eq_first += r.text == texts[0]
        for rank, ((aid, _), t) in enumerate(zip(caps, texts)):
            rows.append({"idx": int(r.idx), "image_id": int(r.image_id), "ann_id": aid, "rank": rank, "text": t})
    idx = pd.DataFrame(rows)
    idx.insert(0, "cap_row", np.arange(len(idx), dtype=np.int64))
    n = len(items)
    checks = {"n_pool": n, "a_missing_image_ids": missing, "a_pass": not missing,
              "b_frac_c1_equals_an_official_caption": eq_any / n, "b_pass": eq_any / n >= 0.99,
              "b_frac_after_whitespace_and_lowercase": eq_any_norm / n,
              "c_frac_c1_is_lowest_id_annotation": eq_first / n,
              "c_captions_per_image_histogram": {int(k): int(v) for k, v in sorted(hist.items())},
              "n_caption_rows": int(len(idx))}
    return idx, checks


def main():
    X.require_plan_and_code_frozen()
    zpath = os.path.join(X.CAPDIR, "annotations_trainval2014.zip")
    if os.path.exists(zpath):
        dl = {"reused_existing": True, **verify_zip(zpath)}
    else:
        dl = download(zpath)
    jpath = os.path.join(X.CAPDIR, "captions_val2014.json")
    jsha = extract_member(zpath, jpath) if not os.path.exists(jpath) else X.sha256_file(jpath)
    with open(jpath) as f:
        cj = json.load(f)
    items = pd.read_parquet(X.ITEMS, columns=["idx", "image_id", "text"])
    idx, checks = build_index(cj, items)
    os.makedirs(X.FEAT, exist_ok=True)
    out = os.path.join(X.FEAT, "captions_index.parquet")
    if os.path.exists(out):
        raise FileExistsError(out)
    idx.to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    checks.update({"download": dl, "captions_json_sha256": jsha, "index_sha256": X.sha256_file(out),
                   "items_sha256": X.sha256_file(X.ITEMS), "utc": X.utcnow()})
    h = X.write_json_new(os.path.join(X.FEAT, "captions_checks.json"), checks)
    print(json.dumps({k: v for k, v in checks.items() if k != "a_missing_image_ids"}, indent=1))
    print("captions_checks.json", h)
    if not (checks["a_pass"] and checks["b_pass"]):
        print("STOP: caption check (a) or (b) failed (plan §3.3)")
        sys.exit(3)


if __name__ == "__main__":
    main()
