"""Feature extraction core (ViTs via timm, LMs via transformers). fp32 weights/activations, TF32 off.

Primary outputs per (dataset, model):
  <ds>/<slug>.npy          float32 (m, n_layers, d)  RAW pooled features (no clipping / normalisation)
  <ds>/<slug>.tokmean.npy  (ViT only) float32 (m, n_layers, d) mean over patch tokens (prefix tokens excluded)
  <ds>/<slug>.lm_aux.npz   (LM only) n_tokens, nll_mean (nats/token), bpb (bits/byte), last_prenorm (m, d)
  <ds>/<slug>.json         sidecar (provenance, layer names, checks, timings, sha256 of every array)
  <ds>/<slug>.done         marker, written last
"""
from __future__ import annotations

import gc
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from common import (FEATURES, HUB, atomic_save_npy, atomic_save_npz, atomic_write_json, blob_info_for_snapshot_file,
                    code_hashes, env_info, set_strict_fp32, sha256_file, utcnow)
from registry import LM_REVISIONS, REL_DEPTHS, depth_index, slug

DEV = "cuda:0"
DATASET_SOURCES = {"wit1024": ("datasets--minhuh--prh",), "wit4096_sel": ("datasets--minhuh--prh",),
                   "coco_pool": ("datasets--sayakpaul--coco-30-val-2014",)}


# ----------------------------------------------------------------------------------------------------------------- data
def load_items(ds: str, m: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES / ds / "items.parquet")
    return df.iloc[:m].reset_index(drop=True) if m is not None else df


def items_manifest(ds: str) -> dict:
    return json.load(open(FEATURES / ds / "items_manifest.json"))


def load_image_bytes(ds: str, items: pd.DataFrame) -> list:
    man = items_manifest(ds)
    snap = HUB / DATASET_SOURCES[ds][0] / "snapshots" / man["revision"] / "data"
    out = [None] * len(items)
    for shard, grp in items.groupby("src_shard"):
        t = pq.read_table(snap / shard, columns=["image"])
        col = t.column("image")
        for pos, r in zip(grp.index.values, grp["src_row_in_shard"].values):
            out[pos] = col[int(r)]["bytes"].as_py()
    for b, h in zip(out, items["image_sha256"].values):
        assert b is not None
    # verify bytes against the frozen table
    import hashlib
    bad = sum(hashlib.sha256(b).hexdigest() != h for b, h in zip(out, items["image_sha256"].values))
    if bad:
        raise RuntimeError(f"{bad} image byte hashes differ from items table for {ds}")
    return out


class ImageDS(torch.utils.data.Dataset):
    """Decode exactly like datasets.Image().decode_example (PRH/ARH path), then timm eval transform."""

    def __init__(self, blobs, transform):
        self.blobs, self.transform = blobs, transform
        from datasets import Image as HFImage
        self.dec = HFImage()

    def __len__(self):
        return len(self.blobs)

    def __getitem__(self, i):
        im = self.dec.decode_example({"path": None, "bytes": self.blobs[i]})
        conv = 0
        if im.mode != "RGB":  # WIT: never happens (all RGB). COCO: grayscale/CMYK -> RGB.
            im = im.convert("RGB")
            conv = 1
        return self.transform(im), conv


# -------------------------------------------------------------------------------------------------------- downloads
def ensure_timm_weights(model_name: str) -> dict:
    import timm
    from huggingface_hub import hf_hub_download
    cfg = timm.models.get_pretrained_cfg(model_name).to_dict()
    hub = cfg["hf_hub_id"]
    try:
        p = hf_hub_download(hub, "model.safetensors")
    except Exception:
        p = hf_hub_download(hub, cfg.get("hf_hub_filename") or "pytorch_model.bin")
    return dict(hf_hub_id=hub, snapshot=Path(p).parent.name, weights=[blob_info_for_snapshot_file(Path(p))],
                pretrained_cfg={k: cfg.get(k) for k in ["input_size", "crop_pct", "crop_mode", "interpolation",
                                                       "mean", "std", "tag", "architecture", "license"]})


def lm_allow_patterns(model_id: str, rev: str) -> list:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(model_id, revision=rev)
    has_st = any(f.endswith(".safetensors") for f in files)
    pats = ["*.json", "*.model", "tokenizer*", "*.txt"]  # never *.py (no remote code)
    pats += ["*.safetensors"] if has_st else ["pytorch_model*.bin"]
    return pats


def ensure_lm_snapshot(model_id: str) -> Path:
    """snapshot_download at the pinned revision (no-op if complete; waits on locks if prefetch is mid-download)."""
    from huggingface_hub import snapshot_download
    rev = LM_REVISIONS[model_id]
    try:
        p = snapshot_download(model_id, revision=rev, allow_patterns=lm_allow_patterns(model_id, rev))
        snap = Path(p)
    except Exception:
        # offline fallback: accept a complete local snapshot only
        snap = HUB / ("models--" + model_id.replace("/", "--")) / "snapshots" / rev
        ok = snap.is_dir() and (snap / "config.json").exists() and \
            (any(snap.glob("*.safetensors")) or any(snap.glob("pytorch_model*.bin")))
        for idx in list(snap.glob("*.index.json")) if ok else []:
            wm = json.load(open(idx)).get("weight_map", {})
            ok = ok and all((snap / f).exists() for f in set(wm.values()))
        if not ok:
            raise
    assert snap.name == rev, (snap, rev)
    return snap


# --------------------------------------------------------------------------------------------------------- validity
def out_paths(out_root: Path, ds: str, s: str) -> dict:
    d = out_root / ds
    return dict(dir=d, npy=d / f"{s}.npy", tok=d / f"{s}.tokmean.npy", aux=d / f"{s}.lm_aux.npz",
                json=d / f"{s}.json", done=d / f"{s}.done", failed=d / f"{s}.failed")


def is_valid(out_root: Path, ds: str, model_id: str, expect_m: int, check_hash: bool = True) -> tuple[bool, str]:
    p = out_paths(out_root, ds, slug(model_id))
    if not (p["json"].exists() and p["npy"].exists() and p["done"].exists()):
        return False, "missing files"
    try:
        meta = json.load(open(p["json"]))
        if meta.get("status") != "complete":
            return False, "status != complete"
        if meta.get("items_sha256") != items_manifest(ds)["items_sha256"]:
            return False, "items table changed"
        arr = np.load(p["npy"], mmap_mode="r")
        if arr.dtype != np.float32 or tuple(arr.shape) != tuple(meta["shape"]) or arr.shape[0] != expect_m:
            return False, f"shape/dtype mismatch {arr.shape} {arr.dtype}"
        for key, f in meta.get("files_sha256", {}).items():
            fp = p["dir"] / f["name"]
            if not fp.exists():
                return False, f"missing {fp.name}"
            if check_hash and sha256_file(fp) != f["sha256"]:
                return False, f"sha256 mismatch {fp.name}"
        return True, "ok"
    except Exception as e:
        return False, repr(e)


def _basic_array_checks(a: np.ndarray) -> dict:
    a64 = a.astype(np.float64, copy=False)
    finite = bool(np.isfinite(a64).all())
    row_norm = np.linalg.norm(a64, axis=-1)  # (m, L)
    zero_rows = int((row_norm == 0).sum())
    # per-layer number of distinct rows (degenerate features would make every item identical)
    std_over_items = a64.std(axis=0).mean(axis=-1)  # (L,)
    return dict(finite=finite, zero_rows=zero_rows, min_row_norm=float(row_norm.min()),
                max_row_norm=float(row_norm.max()), min_layer_item_std=float(std_over_items.min()),
                max_abs=float(np.abs(a64).max()))


def _write_outputs(out_root, ds, model_id, primary, extras: dict, meta: dict):
    s = slug(model_id)
    p = out_paths(out_root, ds, s)
    p["dir"].mkdir(parents=True, exist_ok=True)
    for f in [p["done"], p["failed"]]:
        if f.exists():
            f.unlink()
    checks = _basic_array_checks(primary)
    if not checks["finite"] or checks["zero_rows"] > 0:
        raise RuntimeError(f"array checks failed for {model_id}/{ds}: {checks}")
    atomic_save_npy(p["npy"], primary)
    files = {"primary": dict(name=p["npy"].name, sha256=sha256_file(p["npy"]), shape=list(primary.shape))}
    if "tokmean" in extras:
        atomic_save_npy(p["tok"], extras["tokmean"])
        files["tokmean"] = dict(name=p["tok"].name, sha256=sha256_file(p["tok"]), shape=list(extras["tokmean"].shape))
    if "lm_aux" in extras:
        atomic_save_npz(p["aux"], **extras["lm_aux"])
        files["lm_aux"] = dict(name=p["aux"].name, sha256=sha256_file(p["aux"]),
                               arrays={k: list(v.shape) for k, v in extras["lm_aux"].items()})
    meta.update(status="complete", shape=list(primary.shape), dtype="float32", files_sha256=files,
                array_checks=checks, finished_utc=utcnow())
    atomic_write_json(p["json"], meta)
    p["done"].write_text(utcnow() + "\n")


# --------------------------------------------------------------------------------------------------------------- ViT
def vit_batch_size(n_params: int, img_size: int) -> int:
    scale = (img_size / 224) ** 2
    if n_params < 120e6:
        bs = 64
    elif n_params < 400e6:
        bs = 32
    elif n_params < 700e6:
        bs = 16
    else:
        bs = 8
    return max(4, int(bs / scale))


def run_vit(model_name: str, datasets: list, out_root: Path, m_limit: int | None, log) -> None:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    fp32_flags = set_strict_fp32()
    t0 = time.time()
    wt = ensure_timm_weights(model_name)
    t_dl = time.time() - t0
    t0 = time.time()
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(DEV).eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    t_load = time.time() - t0
    n_params = int(sum(p.numel() for p in model.parameters()))
    data_cfg = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**data_cfg)
    blocks = model.blocks
    L = len(blocks)
    npt = int(getattr(model, "num_prefix_tokens", 1))
    has_cls = getattr(model, "cls_token", None) is not None
    d = int(model.embed_dim)
    img_size = int(data_cfg["input_size"][-1])
    bs = vit_batch_size(n_params, img_size)
    log(f"{model_name}: L={L} d={d} npt={npt} cls={has_cls} params={n_params/1e6:.1f}M img={img_size} bs={bs} "
        f"load={t_load:.1f}s dl={t_dl:.1f}s")

    fx_done = False
    for ds in datasets:
        items = load_items(ds, m_limit)
        m = len(items)
        ok, why = is_valid(out_root, ds, model_name, m)
        if ok:
            log(f"skip {model_name}/{ds}: valid")
            continue
        t_ds = time.time()
        blobs = load_image_bytes(ds, items)
        loader = torch.utils.data.DataLoader(ImageDS(blobs, transform), batch_size=bs, shuffle=False,
                                             num_workers=12, pin_memory=True, persistent_workers=False)
        is_wit = ds.startswith("wit")
        if is_wit:
            sel = list(range(L))  # all blocks (ARH/PRH: blocks.{i}.add_1 for every i)
            layer_names = [f"blocks.{i}.add_1" for i in sel]
            rel = None
        else:
            ells = [depth_index(r, L) for r in REL_DEPTHS]
            sel = [e - 1 for e in ells]
            layer_names = [f"blocks.{i}.add_1" for i in sel]
            rel = dict(zip([str(r) for r in REL_DEPTHS], ells))
        cls_all = np.empty((m, L, d), np.float32) if has_cls else None
        tok_all = np.empty((m, L, d), np.float32)
        buf = {}

        # (1) FX replication check of the ARH/PRH extractor on the first batch (before hooks are attached)
        fx_check = None
        first_x = None
        if is_wit and not fx_done:
            try:
                from torchvision.models.feature_extraction import create_feature_extractor
                first_x, _ = next(iter(torch.utils.data.DataLoader(ImageDS(blobs[:min(8, m)], transform),
                                                                   batch_size=8, num_workers=0)))
                first_x = first_x.to(DEV)
                fx = create_feature_extractor(model, return_nodes=[f"blocks.{i}.add_1" for i in range(L)])
                with torch.inference_mode():
                    fo = fx(first_x)
                fx_cls = torch.stack([v[:, 0, :] for v in fo.values()], dim=1).double().cpu().numpy()  # (8,L,d)
                fx_check = dict(nodes=list(fo.keys())[:2] + ["..."], ref=fx_cls)
                del fx, fo
            except Exception as e:
                fx_check = dict(error=repr(e))
            fx_done = True

        def mk(i):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, (tuple, list)) else out
                buf[i] = (o[:, 0, :].double() if has_cls else None, o[:, npt:, :].double().mean(dim=1))
            return hook
        hs = [blocks[i].register_forward_hook(mk(i)) for i in range(L)]
        n_conv = 0
        pos = 0
        t_fwd = time.time()
        try:
            with torch.inference_mode():
                for x, conv in loader:
                    n_conv += int(conv.sum())
                    x = x.to(DEV, non_blocking=True)
                    buf.clear()
                    model.forward_features(x)
                    b = x.shape[0]
                    for i in range(L):
                        c, t = buf[i]
                        if has_cls:
                            cls_all[pos:pos + b, i] = c.float().cpu().numpy()
                        tok_all[pos:pos + b, i] = t.float().cpu().numpy()
                    pos += b
        finally:
            for h in hs:
                h.remove()
        assert pos == m
        t_fwd = time.time() - t_fwd
        if fx_check is not None and "ref" in fx_check:
            n = fx_check["ref"].shape[0]
            diff = np.abs(fx_check["ref"] - cls_all[:n].astype(np.float64))
            fx_check = dict(nodes=fx_check["nodes"], n_images=int(n), max_abs_diff=float(diff.max()),
                            max_rel_diff=float(diff.max() / np.abs(fx_check["ref"]).max()))
        # determinism re-check on the first 4 items
        rerun = None
        try:
            xs = torch.stack([ImageDS(blobs[:4], transform)[j][0] for j in range(min(4, m))]).to(DEV)
            hs = [blocks[i].register_forward_hook(mk(i)) for i in range(L)]
            with torch.inference_mode():
                buf.clear()
                model.forward_features(xs)
            for h in hs:
                h.remove()
            src = cls_all if has_cls else tok_all
            again = np.stack([(buf[i][0] if has_cls else buf[i][1]).float().cpu().numpy() for i in range(L)], 1)
            rerun = dict(n=int(xs.shape[0]), batch_size=int(xs.shape[0]),
                         max_abs_diff=float(np.abs(again - src[:xs.shape[0]]).max()))
        except Exception as e:
            rerun = dict(error=repr(e))

        if has_cls:
            primary = cls_all[:, sel, :]
            pooling = "CLS token (index 0) of the residual stream at blocks.{i}.add_1 (block output), no final norm"
        else:
            primary = tok_all[:, sel, :]
            pooling = ("model has NO class token (SigLIP, global_pool=map): primary = mean over all patch tokens of "
                       "blocks.{i}.add_1 output, no final norm, no MAP head")
        extras = dict(tokmean=np.ascontiguousarray(tok_all[:, sel, :]))
        man = items_manifest(ds)
        meta = dict(model_id=model_name, kind="vit", dataset=ds, m=m, m_limit=m_limit, items_sha256=man["items_sha256"],
                    source=dict(timm_name=model_name, **wt), num_params=n_params, n_blocks=L, embed_dim=d,
                    num_prefix_tokens=npt, has_cls_token=has_cls, layer_axis_block_indices=sel,
                    layer_names=layer_names, rel_depths_to_residual_index=rel,
                    depth_rule="ell(r)=floor(r*L+0.5); stored block index = ell-1 (output of block ell)",
                    pooling=pooling, tokmean="mean over tokens [num_prefix_tokens:] (patch tokens only), float64 accumulate",
                    weights_dtype="float32", activations_dtype="float32", fp32_flags=fp32_flags,
                    transform=dict(data_config={k: (list(v) if isinstance(v, tuple) else v) for k, v in data_cfg.items()},
                                   repr=repr(transform)),
                    image_decode="datasets.Image().decode_example (PIL; EXIF transpose if Orientation tag) ; convert('RGB') only if mode!='RGB'",
                    n_converted_to_rgb=n_conv, batch_size=bs, fx_check=fx_check, rerun_check=rerun,
                    timings=dict(download_s=t_dl, load_s=t_load, dataset_total_s=time.time() - t_ds, forward_s=t_fwd,
                                 per_item_ms=1000 * t_fwd / m),
                    env=env_info(), code_sha256=code_hashes(), started_utc=utcnow())
        _write_outputs(out_root, ds, model_name, np.ascontiguousarray(primary), extras, meta)
        log(f"wrote {model_name}/{ds}: {primary.shape} fwd={t_fwd:.1f}s fx={fx_check} rerun={rerun}")
        del blobs, loader, cls_all, tok_all, primary, extras
        gc.collect()
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------- LM
def _find_layers_and_norm(model, L):
    layers = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == L:
            layers = (name, mod)
            break
    base = model.base_model
    norm = None
    for n in ["norm", "ln_f", "final_layer_norm"]:
        if hasattr(base, n):
            norm = (n, getattr(base, n))
            break
    return layers, norm


def _tok_load(path):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(path, use_fast=True), True
    except ValueError:
        return AutoTokenizer.from_pretrained(path, use_fast=False), False


def run_lm(model_id: str, datasets: list, out_root: Path, m_limit: int | None, log) -> None:
    from transformers import AutoModelForCausalLM

    fp32_flags = set_strict_fp32()
    t0 = time.time()
    snap = ensure_lm_snapshot(model_id)
    t_dl = time.time() - t0
    weight_files = sorted(list(snap.glob("*.safetensors")) or list(snap.glob("pytorch_model*.bin")))
    t0 = time.time()
    tok, fast = _tok_load(str(snap))
    model = AutoModelForCausalLM.from_pretrained(str(snap), dtype=torch.float32, local_files_only=True)
    model = model.to(DEV).eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    t_load = time.time() - t0
    n_params = int(sum(p.numel() for p in model.parameters()))
    L = int(model.config.num_hidden_layers)
    d = int(model.config.hidden_size)
    (lname, layers), (nname, fnorm) = _find_layers_and_norm(model, L)
    attn_impl = getattr(model.config, "_attn_implementation", None)
    log(f"{model_id}: L={L} d={d} params={n_params/1e6:.1f}M tok={type(tok).__name__} fast={fast} attn={attn_impl} "
        f"layers={lname} norm={nname} load={t_load:.1f}s dl={t_dl:.1f}s")

    sp = None
    if (snap / "tokenizer.model").exists():
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=str(snap / "tokenizer.model"))
        except Exception:
            sp = None

    pre_buf = {}

    def last_hook(mod, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        pre_buf["last"] = o

    semantics = None
    for ds in datasets:
        items = load_items(ds, m_limit)
        m = len(items)
        ok, why = is_valid(out_root, ds, model_id, m)
        if ok:
            log(f"skip {model_id}/{ds}: valid")
            continue
        texts = items["text"].astype(str).tolist()
        t_ds = time.time()
        feats = np.empty((m, L + 1, d), np.float32)
        prenorm = np.empty((m, d), np.float32)
        n_tok = np.empty(m, np.int64)
        nll = np.full(m, np.nan, np.float64)
        bpb = np.full(m, np.nan, np.float64)
        sp_mismatch = 0
        sp_checked = 0

        # hidden_states semantics check on item 0 (once per model)
        if semantics is None:
            semantics = {}
            cap = {}
            hk = []
            for i, layer in enumerate(layers):
                def mk(i):
                    def h(mod, inp, out):
                        cap[i] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
                    return h
                hk.append(layer.register_forward_hook(mk(i)))

            def pre0(mod, args, kwargs):
                x = args[0] if len(args) else kwargs.get("hidden_states")
                cap["in0"] = x.detach()
            hk.append(layers[0].register_forward_pre_hook(pre0, with_kwargs=True))
            try:
                enc = tok(texts[0], return_tensors="pt")
                with torch.inference_mode():
                    o = model(input_ids=enc["input_ids"].to(DEV), attention_mask=enc["attention_mask"].to(DEV),
                              output_hidden_states=True, use_cache=False)
                    hsx = o.hidden_states
                    semantics["n_hidden_states"] = len(hsx)
                    semantics["hs0_vs_layer0_input"] = float((hsx[0] - cap["in0"]).abs().max())
                    semantics["hs_i_vs_layer_i-1_output_max"] = float(max(
                        (hsx[i + 1] - cap[i]).abs().max() for i in range(L - 1))) if L > 1 else None
                    semantics["hsL_vs_finalnorm(lastlayer)"] = float((hsx[L] - fnorm(cap[L - 1])).abs().max())
                    semantics["hsL_vs_lastlayer_raw"] = float((hsx[L] - cap[L - 1]).abs().max())
            except Exception as e:
                semantics["error"] = repr(e)
            finally:
                for h in hk:
                    h.remove()
            log(f"{model_id} hidden_states semantics: {semantics}")

        hlast = layers[L - 1].register_forward_hook(last_hook)
        t_fwd = time.time()
        try:
            with torch.inference_mode():
                for j, text in enumerate(texts):
                    enc = tok(text, return_tensors="pt")  # batch size 1: no padding, default special tokens
                    ids = enc["input_ids"].to(DEV)
                    am = enc["attention_mask"].to(DEV)
                    assert int(am.min()) == 1
                    out = model(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=False)
                    hsx = out.hidden_states
                    if len(hsx) != L + 1:
                        raise RuntimeError(f"expected {L+1} hidden states, got {len(hsx)}")
                    pooled = torch.stack([h[0].double().mean(dim=0) for h in hsx])  # (L+1, d)
                    feats[j] = pooled.float().cpu().numpy()
                    prenorm[j] = pre_buf["last"][0].double().mean(dim=0).float().cpu().numpy()
                    T = ids.shape[1]
                    n_tok[j] = T
                    if T > 1:
                        lg = out.logits[0, :-1].double()
                        lp = torch.log_softmax(lg, dim=-1)
                        tgt = ids[0, 1:]
                        tok_nll = -lp.gather(1, tgt[:, None])[:, 0]
                        nll[j] = float(tok_nll.mean())
                        bpb[j] = float(tok_nll.sum()) * math.log2(math.e) / len(text.encode("utf-8"))
                    if sp is not None:
                        ref = sp.encode(text)
                        ours = ids[0].tolist()
                        if tok.bos_token_id is not None and ours and ours[0] == tok.bos_token_id:
                            ours = ours[1:]
                        sp_checked += 1
                        sp_mismatch += int(ours != ref)
                    del out, hsx
                    if (j + 1) % 2000 == 0:
                        log(f"  {model_id}/{ds}: {j+1}/{m} ({(time.time()-t_fwd)/(j+1)*1000:.1f} ms/item)")
        finally:
            hlast.remove()
        t_fwd = time.time() - t_fwd
        # determinism re-check: items 0..2 again
        rer = []
        with torch.inference_mode():
            for j in range(min(3, m)):
                enc = tok(texts[j], return_tensors="pt")
                o = model(input_ids=enc["input_ids"].to(DEV), attention_mask=enc["attention_mask"].to(DEV),
                          output_hidden_states=True, use_cache=False)
                p2 = torch.stack([h[0].double().mean(dim=0) for h in o.hidden_states]).float().cpu().numpy()
                rer.append(float(np.abs(p2 - feats[j]).max()))
        rerun = dict(n=len(rer), max_abs_diff=max(rer) if rer else None)

        is_wit = ds.startswith("wit")
        if is_wit:
            sel = list(range(L + 1))
            rel = None
        else:
            sel = [depth_index(r, L) for r in REL_DEPTHS]
            rel = dict(zip([str(r) for r in REL_DEPTHS], sel))
        layer_names = [("embedding_output(hidden_states[0])" if i == 0 else
                        (f"hidden_states[{i}] (=final_norm(output of layer {i-1}); HF convention)" if i == L else
                         f"hidden_states[{i}] (=output of layer {i-1})")) for i in sel]
        primary = np.ascontiguousarray(feats[:, sel, :])
        extras = dict(lm_aux=dict(n_tokens=n_tok, nll_mean=nll, bpb=bpb, last_prenorm=prenorm))
        man = items_manifest(ds)
        meta = dict(model_id=model_id, kind="lm", dataset=ds, m=m, m_limit=m_limit, items_sha256=man["items_sha256"],
                    source=dict(repo=model_id, revision=LM_REVISIONS[model_id], snapshot_dir=str(snap),
                                weights=[blob_info_for_snapshot_file(f) for f in weight_files]),
                    num_params=n_params, n_layers=L, hidden_size=d, layer_axis_hidden_state_indices=sel,
                    layer_names=layer_names, rel_depths_to_residual_index=rel,
                    depth_rule="ell(r)=floor(r*L+0.5); hidden_states[ell] (0 = embedding output)",
                    pooling=("mean over all tokens (batch size 1, attention_mask all ones) of every HF hidden state; "
                             "float64 accumulation; last state is post-final-norm (HF convention, as PRH/ARH); "
                             "pre-norm last-layer mean in lm_aux.last_prenorm"),
                    text_rule=("items column 'text' (WIT: caption_idx=0 string unchanged; COCO: caption.strip()), no prompt, "
                               "tokenizer(text) with default special tokens, no truncation"),
                    tokenizer=dict(cls=type(tok).__name__, use_fast=fast, bos=tok.bos_token_id, eos=tok.eos_token_id,
                                   sp_check=(dict(checked=sp_checked, mismatches=sp_mismatch,
                                                  rule="ids (minus leading BOS) == sentencepiece.encode(text)") if sp else None)),
                    n_tokens=dict(min=int(n_tok.min()), median=float(np.median(n_tok)), max=int(n_tok.max())),
                    nll_mean_over_items=float(np.nanmean(nll)), bpb_mean_over_items=float(np.nanmean(bpb)),
                    weights_dtype="float32", activations_dtype="float32", attn_implementation=attn_impl,
                    fp32_flags=fp32_flags, batch_size=1, hidden_state_semantics=semantics, rerun_check=rerun,
                    timings=dict(download_s=t_dl, load_s=t_load, dataset_total_s=time.time() - t_ds, forward_s=t_fwd,
                                 per_item_ms=1000 * t_fwd / m),
                    env=env_info(), code_sha256=code_hashes(), started_utc=utcnow())
        _write_outputs(out_root, ds, model_id, primary, extras, meta)
        log(f"wrote {model_id}/{ds}: {primary.shape} fwd={t_fwd:.1f}s ({1000*t_fwd/m:.1f} ms/item) rerun={rerun} "
            f"sp={meta['tokenizer']['sp_check']}")
        del feats, primary, extras
        gc.collect()
    del model
    gc.collect()
    torch.cuda.empty_cache()
