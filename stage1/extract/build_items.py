"""Build the frozen item tables for Stage-1 extraction.

Outputs (all under ~/hrh_v2_stage1_20261008/features/):
  wit1024/items.parquet       S     = minhuh/prh@wit_1024 (snapshot 0220f2b7...), dataset row order
  wit4096_sel/items.parquet   S_sel = minhuh/prh@wit_4096 (snapshot c41c877f...), dataset rows 1025..2048 (1-based)
  coco_pool/items.parquet     first 16384 items of the sha256 permutation of sayakpaul/coco-30-val-2014 image ids
  coco_pool/coco_all_order.parquet  all 30000 items in permutation order (for later extension)
  <ds>/items_manifest.json    provenance + sha256 of the item table

Row order inside each parquet snapshot = sorted shard file names, then row order inside each shard.
This is exactly the order `datasets.load_dataset(...)` yields (verified by verify_wit_order()).
"""
from __future__ import annotations

import glob
import io
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COCO_POOL_N, COCO_REPO, COCO_REV, COCO_SALT, FEATURES, HUB, PRH_REPO, SEL_ROWS,  # noqa: E402
                    WIT1024_REV, WIT4096_REV, append_status, atomic_write_json, sha256_bytes, sha256_file, utcnow)


def snapshot_dir(repo: str, rev: str) -> Path:
    d = HUB / ("datasets--" + repo.replace("/", "--")) / "snapshots" / rev
    if not d.is_dir():
        raise FileNotFoundError(d)
    return d


def shard_files(repo: str, rev: str):
    return sorted(glob.glob(str(snapshot_dir(repo, rev) / "data" / "*.parquet")))


def read_rows(repo: str, rev: str, columns=None):
    """Yield (shard_name, row_in_shard, global_row, record) in dataset order."""
    g = 0
    for f in shard_files(repo, rev):
        t = pq.read_table(f, columns=columns)
        for r, rec in enumerate(t.to_pylist()):
            yield Path(f).name, r, g, rec
            g += 1


def _img_stats(b: bytes):
    import PIL.Image
    import PIL.ExifTags
    from PIL import Image
    im = Image.open(io.BytesIO(b))
    w, h = im.size
    mode = im.mode
    try:
        ori = im.getexif().get(PIL.Image.ExifTags.Base.Orientation)
    except Exception:
        ori = None
    return mode, w, h, (int(ori) if ori is not None else -1)


def add_image_stats(df: pd.DataFrame, blobs: list) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=16) as ex:
        st = list(ex.map(_img_stats, blobs, chunksize=64))
    df["img_mode"] = [s[0] for s in st]
    df["img_w"] = [s[1] for s in st]
    df["img_h"] = [s[2] for s in st]
    df["exif_orientation"] = [s[3] for s in st]
    return df


def write_table(ds: str, df: pd.DataFrame, manifest: dict) -> None:
    out = FEATURES / ds
    out.mkdir(parents=True, exist_ok=True)
    p = out / "items.parquet"
    df.to_parquet(p, index=False)
    manifest.update(items_parquet=str(p), items_sha256=sha256_file(p), m=int(len(df)), created_utc=utcnow(),
                    columns=list(df.columns))
    if "text" in df:
        manifest["texts_sha256"] = sha256_bytes("\n".join(df["text"].tolist()).encode("utf-8"))
    if "caption" in df:
        manifest["captions_sha256"] = sha256_bytes("\n".join(df["caption"].tolist()).encode("utf-8"))
    if "image_sha256" in df:
        manifest["image_hashes_sha256"] = sha256_bytes("\n".join(df["image_sha256"].tolist()).encode())
    manifest["img_mode_counts"] = df["img_mode"].value_counts().to_dict() if "img_mode" in df else None
    if "exif_orientation" in df:
        manifest["exif_orientation_counts"] = {str(k): int(v) for k, v in df["exif_orientation"].value_counts().items()}
    atomic_write_json(out / "items_manifest.json", manifest)
    print(ds, len(df), manifest["items_sha256"])


def build_wit(ds: str, rev: str, rows: tuple | None, s_caps: set | None, s_imgs: set | None):
    recs, blobs = [], []
    for shard, r, g, rec in read_rows(PRH_REPO, rev):
        if rows is not None and not (rows[0] <= g < rows[1]):
            continue
        b = rec["image"]["bytes"]
        texts = rec["text"]
        d = dict(src_shard=shard, src_row_in_shard=r, src_row=g, src_path=rec["image"]["path"],
                 caption=str(texts[0]), n_captions=len(texts), image_sha256=sha256_bytes(b), image_nbytes=len(b),
                 origin=rec.get("origin"))
        if s_caps is not None:
            d["overlap_S_caption0"] = str(texts[0]) in s_caps
            d["overlap_S_anycaption"] = any(str(t) in s_caps for t in texts)
            d["overlap_S_image"] = d["image_sha256"] in s_imgs
        recs.append(d)
        blobs.append(b)
    df = pd.DataFrame(recs)
    df.insert(0, "idx", np.arange(len(df)))
    df["text"] = df["caption"]  # LM input: raw caption (PRH: str(x['text'][caption_idx]))
    df = add_image_stats(df, blobs)
    return df


def build_coco():
    recs, blobs = [], []
    for shard, r, g, rec in read_rows(COCO_REPO, COCO_REV):
        p = rec["image"]["path"]
        m = re.fullmatch(r"COCO_val2014_(\d{12})\.jpg", p)
        assert m, p
        iid = int(m.group(1))
        b = rec["image"]["bytes"]
        key = sha256_bytes(f"{COCO_SALT}:{iid}".encode())
        recs.append(dict(image_id=iid, src_shard=shard, src_row_in_shard=r, src_row=g, src_path=p,
                         caption=str(rec["caption"]), image_sha256=sha256_bytes(b), image_nbytes=len(b), perm_key=key))
        blobs.append(b)
    df = pd.DataFrame(recs)
    n_rows = len(df)
    # dedup by image id: keep the first row (dataset order). coco-30-val-2014 has one row per image already.
    dup = df.duplicated("image_id", keep="first")
    n_dup = int(dup.sum())
    keep_idx = np.flatnonzero(~dup.values)
    df = df.iloc[keep_idx].reset_index(drop=True)
    blobs = [blobs[i] for i in keep_idx]
    order = np.argsort(df["perm_key"].values, kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    blobs = [blobs[i] for i in order]
    df.insert(0, "perm_rank", np.arange(len(df)))
    df["in_pool"] = df["perm_rank"] < COCO_POOL_N
    return df, blobs, n_rows, n_dup


def verify_wit_order(branch: str, rev: str, n_check: int = 2048) -> dict:
    """Compare our row order (parquet shards) with datasets.load_dataset(streaming=True) caption order."""
    try:
        from datasets import load_dataset
        expected = [str(rec["text"][0]) for _, _, g, rec in read_rows(PRH_REPO, rev, columns=["text"]) if g < n_check]
        it = load_dataset(PRH_REPO, revision=branch, split="train", streaming=True).select_columns(["text"])
        caps = []
        for i, row in enumerate(it):
            if i >= len(expected):
                break
            caps.append(str(row["text"][0]))
        return dict(checked=len(expected), equal=bool(caps == expected))
    except Exception as e:
        return dict(error=repr(e))


def main():
    # S
    s = build_wit("wit1024", WIT1024_REV, None, None, None)
    man = dict(dataset="wit1024", repo=PRH_REPO, branch="wit_1024", revision=WIT1024_REV,
               shards=[dict(name=Path(f).name, blob=Path(f).resolve().name) for f in shard_files(PRH_REPO, WIT1024_REV)],
               order="sorted shard names, row order within shard (= datasets.load_dataset order)",
               caption_rule="text[0] (caption_idx=0), no prompt; LM input column 'text' = caption unchanged")
    man["load_dataset_order_check"] = verify_wit_order("wit_1024", WIT1024_REV, 1024)
    write_table("wit1024", s, man)
    # S_sel
    s_caps = set(s["caption"]); s_imgs = set(s["image_sha256"])
    sel = build_wit("wit4096_sel", WIT4096_REV, SEL_ROWS, s_caps, s_imgs)
    sel["dup_caption0_within_sel"] = sel.duplicated("caption", keep=False)
    man = dict(dataset="wit4096_sel", repo=PRH_REPO, branch="wit_4096", revision=WIT4096_REV,
               shards=[dict(name=Path(f).name, blob=Path(f).resolve().name) for f in shard_files(PRH_REPO, WIT4096_REV)],
               rows_0based_halfopen=list(SEL_ROWS),
               order="sorted shard names, row order within shard (= datasets.load_dataset order); rows 1024..2047",
               caption_rule="text[0] (caption_idx=0), no prompt; LM input column 'text' = caption unchanged",
               n_overlap_S_caption0=int(sel["overlap_S_caption0"].sum()),
               n_overlap_S_anycaption=int(sel["overlap_S_anycaption"].sum()),
               n_overlap_S_image=int(sel["overlap_S_image"].sum()),
               overlap_rows=sel.loc[sel["overlap_S_caption0"] | sel["overlap_S_anycaption"] | sel["overlap_S_image"], "idx"].tolist(),
               n_dup_caption0_within_sel=int(sel["dup_caption0_within_sel"].sum()),
               note=("wit_4096 is NOT wit_1024 followed by 3072 new items: about 1024 of its 4096 items carry an S "
                     "caption (most with differently encoded image bytes), scattered over the rows. Rows 1024..2047 "
                     "contain exactly the flagged overlaps listed here; analysis must drop rows with any overlap flag."))
    man["load_dataset_order_check"] = verify_wit_order("wit_4096", WIT4096_REV, 2048)
    write_table("wit4096_sel", sel, man)
    # COCO
    df, blobs, n_rows, n_dup = build_coco()
    out = FEATURES / "coco_pool"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "coco_all_order.parquet", index=False)
    pool = df[df["in_pool"]].reset_index(drop=True).copy()
    pool.insert(0, "idx", np.arange(len(pool)))
    pool["text"] = pool["caption"].str.strip()  # LM input: leading/trailing whitespace removed (WIT has none)
    pool = add_image_stats(pool, blobs[:COCO_POOL_N])
    man = dict(dataset="coco_pool", repo=COCO_REPO, revision=COCO_REV,
               shards=[dict(name=Path(f).name, blob=Path(f).resolve().name) for f in shard_files(COCO_REPO, COCO_REV)],
               n_source_rows=n_rows, n_duplicate_image_rows_dropped=n_dup, n_unique_images=int(len(df)),
               permutation=f"ascending sha256(f'{COCO_SALT}:{{image_id}}') hex digest; image_id = int parsed from COCO_val2014_<12 digits>.jpg",
               salt=COCO_SALT, pool_n=COCO_POOL_N,
               caption_rule=("the single caption stored in the row (coco-30-val-2014 stores one caption per image, "
                             "randomly sampled by the dataset author; it is not necessarily COCO's first annotation)"),
               text_rule=("LM input column 'text' = caption.strip() (leading/trailing whitespace removed; internal "
                          "whitespace kept). Raw caption kept in column 'caption'."),
               n_text_differs_from_caption=int((pool["text"] != pool["caption"]).sum()),
               all_order_parquet=str(out / "coco_all_order.parquet"),
               all_order_sha256=sha256_file(out / "coco_all_order.parquet"),
               nested_prefixes="pool rows 0..m-1 for m in {256,...,16384} (perm_rank order)")
    write_table("coco_pool", pool, man)
    append_status("built item tables wit1024 (m=1024), wit4096_sel (m=1024, see overlap flags), coco_pool (m=16384)",
                  "features/{wit1024,wit4096_sel,coco_pool}/items.parquet")


if __name__ == "__main__":
    main()
