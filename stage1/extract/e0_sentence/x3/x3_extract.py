"""X3 (EXPLORATORY, ANALYSIS_PLAN_E0.md v1.0 section 7.6) DINOv2-B feature extraction; role x3-run.

Readings: analysis/e0_sentence/x3/X3_READINGS.md (sha256 in STATUS line "[x3-run] READINGS FIXED").
Reuses the frozen S1.1 extraction helpers in extract/ (imported, never edited): item table, image bytes with
their sha256 check, the PIL decode of ImageDS, strict fp32 (TF32 off), env info.

Pass A (S1.1 settings, check only): timm vit_base_patch14_dinov2.lvd142m, pinned weights loaded from the verified
  local cache file, timm eval transform (518 px, crop_pct 1.0, bicubic), batch 11, hooks on blocks 5 and 11 exactly
  as extract_core.run_vit (CLS token and patch-token mean, float64 accumulate, no final norm). By-product: mean of
  all output tokens of forward_features (after the final LayerNorm) at 518 px (stored, not scored).
Pass B (ADDED variant x3add_schnaus224): same weights, model built at img_size 224 with the position embedding
  resampled 37x37 -> 16x16 by F.interpolate(bicubic, align_corners=False, antialias=False) in float32 (as HF
  Dinov2Model.interpolate_pos_encoding); transform resize short side 256 bicubic, center crop 224, ImageNet
  mean/std; feature = mean over all 257 tokens of forward_features (CLS + 256 patch tokens, after the final norm).
Checks: weight sha256; pass A vs the S1.1 stored arrays (STOP rule of the readings); determinism of 64 images
  run twice; side note: timm's own antialiased pos-embed resample vs the HF-style one on 64 images.

    taskset -c 0-5 venv/bin/python -I extract/e0_sentence/x3/x3_extract.py
Writes features/coco_e0/x3/* (never overwrites) and results/E0_sentence/x3/extraction/EXTRACTION_X3.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

PROJECT = Path.home() / "hrh_v2_stage1_20261008"
EXTRACT = PROJECT / "extract"
sys.path.insert(0, str(EXTRACT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common as XC  # noqa: E402  (S1.1, read only)
import extract_core as XE  # noqa: E402  (S1.1, read only)

MODEL = "vit_base_patch14_dinov2.lvd142m"
SNAP = "4685c99dabffe5affac90bd99dbffd25801ae58d"
WEIGHT_SHA = "55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b"
WEIGHT_SIZE = 346334872
WEIGHT_FILE = Path.home() / ".cache/huggingface/hub/models--timm--vit_base_patch14_dinov2.lvd142m/snapshots" / SNAP / "model.safetensors"
S11_NPY = PROJECT / "features/coco_pool/vit_base_patch14_dinov2.lvd142m.npy"
S11_TOK = PROJECT / "features/coco_pool/vit_base_patch14_dinov2.lvd142m.tokmean.npy"
S11_JSON = PROJECT / "features/coco_pool/vit_base_patch14_dinov2.lvd142m.json"
OUT = PROJECT / "features/coco_e0/x3"
EXTRACTION = PROJECT / "results/E0_sentence/x3/extraction"
BLOCKS = (5, 11)          # rel 0.5 and rel 1.0 (stored S1.1 axes 1 and 3)
S11_AXIS = {5: 1, 11: 3}
DEV = "cuda:0"
NUM_WORKERS = 5           # + main process = 6 cores (taskset -c 0-5)
STOP_REL = 1e-3


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def save_new(path: Path, arr: np.ndarray) -> str:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".tmp.npy")
    if tmp.exists():
        raise FileExistsError(tmp)
    np.save(tmp, arr)
    os.replace(tmp, path)
    return sha256_file(path)


def write_json_new(path: Path, obj) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)
    return sha256_file(path)


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def load_state_dict():
    from safetensors.torch import load_file
    if WEIGHT_FILE.resolve().stat().st_size != WEIGHT_SIZE:
        raise SystemExit("STOP: weight size differs from the S1.1 pin")
    h = sha256_file(WEIGHT_FILE)
    if h != WEIGHT_SHA:
        raise SystemExit(f"STOP: weight sha256 {h} differs from the S1.1 pin {WEIGHT_SHA}")
    return load_file(str(WEIGHT_FILE)), h


def build_518():
    import timm
    m = timm.create_model(MODEL, pretrained=True, num_classes=0,
                          pretrained_cfg_overlay=dict(file=str(WEIGHT_FILE))).to(DEV).eval()
    return m


def hf_style_resample(pos: torch.Tensor, new_grid: int, npt: int = 1) -> torch.Tensor:
    """pos (1, npt + g*g, d) -> (1, npt + n*n, d): bicubic, align_corners False, no antialias, float32."""
    prefix, patch = pos[:, :npt], pos[:, npt:]
    g = int(round(patch.shape[1] ** 0.5))
    assert g * g == patch.shape[1]
    d = patch.shape[-1]
    p = patch.reshape(1, g, g, d).permute(0, 3, 1, 2).to(torch.float32)
    p = F.interpolate(p, size=(new_grid, new_grid), mode="bicubic", align_corners=False, antialias=False)
    p = p.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, d).to(pos.dtype)
    return torch.cat([prefix, p], dim=1)


def build_224(sd):
    import timm
    m = timm.create_model(MODEL, pretrained=False, num_classes=0, img_size=224)
    sd = dict(sd)
    sd["pos_embed"] = hf_style_resample(sd["pos_embed"], m.patch_embed.grid_size[0], m.num_prefix_tokens)
    missing, unexpected = m.load_state_dict(sd, strict=True), None
    return m.to(DEV).eval(), {"load_state_dict": str(missing)}


def build_224_timm_resample():
    import timm
    m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=224,
                          pretrained_cfg_overlay=dict(file=str(WEIGHT_FILE))).to(DEV).eval()
    return m


def transform_224():
    from timm.data.transforms_factory import create_transform
    cfg = dict(input_size=(3, 224, 224), interpolation="bicubic", mean=(0.485, 0.456, 0.406),
               std=(0.229, 0.224, 0.225), crop_pct=0.875, crop_mode="center")
    return create_transform(**cfg), cfg


def transform_518(model):
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    cfg = resolve_data_config(model.pretrained_cfg, model=model)
    return create_transform(**cfg), cfg


def run_pass(model, blobs, transform, bs, hooks: bool):
    """Returns dict of float32 arrays: cls (m, len(BLOCKS), d), tok (same), normmean (m, d)."""
    m = len(blobs)
    d = int(model.embed_dim)
    npt = int(model.num_prefix_tokens)
    out = {"normmean": np.empty((m, d), np.float32)}
    buf = {}
    hs = []
    if hooks:
        out["cls"] = np.empty((m, len(BLOCKS), d), np.float32)
        out["tok"] = np.empty((m, len(BLOCKS), d), np.float32)

        def mk(i):
            def hook(mod, inp, o):
                o = o[0] if isinstance(o, (tuple, list)) else o
                buf[i] = (o[:, 0, :].double(), o[:, npt:, :].double().mean(dim=1))
            return hook
        hs = [model.blocks[i].register_forward_hook(mk(i)) for i in BLOCKS]
    loader = torch.utils.data.DataLoader(XE.ImageDS(blobs, transform), batch_size=bs, shuffle=False,
                                         num_workers=NUM_WORKERS if m > 256 else 0, pin_memory=True,
                                         persistent_workers=False)
    pos, n_conv = 0, 0
    try:
        with torch.inference_mode():
            for x, conv in loader:
                n_conv += int(conv.sum())
                x = x.to(DEV, non_blocking=True)
                buf.clear()
                f = model.forward_features(x)                 # (b, npt + N, d), after the final norm
                b = x.shape[0]
                out["normmean"][pos:pos + b] = f.double().mean(dim=1).float().cpu().numpy()
                if hooks:
                    for j, i in enumerate(BLOCKS):
                        c, t = buf[i]
                        out["cls"][pos:pos + b, j] = c.float().cpu().numpy()
                        out["tok"][pos:pos + b, j] = t.float().cpu().numpy()
                pos += b
    finally:
        for h in hs:
            h.remove()
    assert pos == m
    out["n_converted_to_rgb"] = n_conv
    out["n_tokens"] = int(f.shape[1])
    return out


def maxdiff(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    dd = np.abs(a - b)
    return {"max_abs_diff": float(dd.max()), "max_abs_ref": float(np.abs(b).max()),
            "rel": float(dd.max() / np.abs(b).max()), "n_bitwise_equal_rows": int((dd.reshape(dd.shape[0], -1).max(1) == 0).sum()),
            "n_rows": int(dd.shape[0])}


def cosines(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    c = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return {"min_cos": float(c.min()), "mean_cos": float(c.mean())}


def main():
    t_start = time.time()
    utc_start = XC.utcnow()
    torch.set_num_threads(1)
    aff = sorted(os.sched_getaffinity(0))
    if len(aff) > 6:
        raise SystemExit(f"REFUSED: CPU affinity {aff} has more than 6 cores; run under taskset -c 0-5")
    readings = PROJECT / "analysis/e0_sentence/x3/X3_READINGS.md"
    st = (PROJECT / "STATUS.md").read_text().splitlines()
    rl = [ln for ln in st if "[x3-run] READINGS FIXED" in ln]
    if not rl or sha256_file(readings) not in rl[0]:
        raise SystemExit("REFUSED: STATUS lacks '[x3-run] READINGS FIXED' with the readings' sha256")
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"REFUSED: {OUT} is not empty (never overwrite)")
    OUT.mkdir(parents=True, exist_ok=True)
    fp32 = XC.set_strict_fp32()
    meta = {"role": "x3-run", "plan": "ANALYSIS_PLAN_E0.md v1.0 section 7.6 X3 (EXPLORATORY)",
            "readings_sha256": sha256_file(readings), "started_utc": utc_start, "model": MODEL,
            "snapshot": SNAP, "fp32_flags": fp32, "cpu_affinity": aff, "num_workers": NUM_WORKERS,
            "env": XC.env_info(), "s11_extract_code_sha256": XC.code_hashes()}
    sd, wsha = load_state_dict()
    meta["weights"] = {"file": str(WEIGHT_FILE), "size": WEIGHT_SIZE, "sha256": wsha, "pin_ok": wsha == WEIGHT_SHA}

    items = XE.load_items("coco_pool")
    man = XE.items_manifest("coco_pool")
    meta["items_sha256"] = man["items_sha256"]
    s11meta = json.load(open(S11_JSON))
    if s11meta["items_sha256"] != man["items_sha256"]:
        raise SystemExit("STOP: COCO pool item table differs from the S1.1 sidecar")
    t0 = time.time()
    blobs = XE.load_image_bytes("coco_pool", items)       # verifies every image's sha256 against the table
    meta["image_bytes_s"] = time.time() - t0
    m = len(blobs)

    # ---------------------------------------------------------------- pass A: S1.1 settings (check + by-product)
    model = build_518()
    tf518, cfg518 = transform_518(model)
    bs518 = XE.vit_batch_size(int(sum(p.numel() for p in model.parameters())), int(cfg518["input_size"][-1]))
    meta["passA"] = {"data_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg518.items()},
                     "transform": repr(tf518), "batch_size": bs518, "blocks": list(BLOCKS),
                     "s11_batch_size": s11meta.get("batch_size"), "s11_transform": s11meta["transform"]["repr"]}
    t0 = time.time()
    A = run_pass(model, blobs, tf518, bs518, hooks=True)
    meta["passA"]["forward_s"] = time.time() - t0
    meta["passA"]["n_converted_to_rgb"] = A["n_converted_to_rgb"]
    meta["passA"]["n_tokens"] = A["n_tokens"]
    # determinism: first 64 images twice, same batch size
    d1 = run_pass(model, blobs[:64], tf518, bs518, hooks=True)
    d2 = run_pass(model, blobs[:64], tf518, bs518, hooks=True)
    meta["passA"]["determinism_64"] = {k: maxdiff(d1[k], d2[k]) for k in ("cls", "tok", "normmean")}
    meta["passA"]["check64_vs_main"] = {k: maxdiff(d1[k][:55], A[k][:55]) for k in ("cls", "tok", "normmean")}
    # S1.1 comparison (STOP rule)
    if sha256_file(S11_NPY) != s11meta["files_sha256"]["primary"]["sha256"] or \
            sha256_file(S11_TOK) != s11meta["files_sha256"]["tokmean"]["sha256"]:
        raise SystemExit("STOP: S1.1 stored DINOv2 arrays differ from their sidecar sha256")
    s11 = np.load(S11_NPY, mmap_mode="r")
    s11t = np.load(S11_TOK, mmap_mode="r")
    cmp = {}
    for j, b in enumerate(BLOCKS):
        ax = S11_AXIS[b]
        cmp[f"cls_block{b}_vs_s11_axis{ax}"] = maxdiff(A["cls"][:, j], s11[:, ax])
        cmp[f"tokmean_block{b}_vs_s11_tokmean_axis{ax}"] = maxdiff(A["tok"][:, j], s11t[:, ax])
    meta["passA"]["vs_s11_stored"] = cmp
    frozen_inputs = {"x3_cls_r10": cmp["cls_block11_vs_s11_axis3"], "x3_tok_r10": cmp["tokmean_block11_vs_s11_tokmean_axis3"],
                     "x3_tok_r05": cmp["tokmean_block5_vs_s11_tokmean_axis1"]}
    stop = {k: v["rel"] > STOP_REL for k, v in frozen_inputs.items()}
    meta["passA"]["stop_rule"] = {"threshold_rel": STOP_REL, "fails": stop, "pass": not any(stop.values())}
    files = {}
    files["x3_rerun518_cls.npy"] = save_new(OUT / "x3_rerun518_cls.npy", np.ascontiguousarray(A["cls"]))
    files["x3_rerun518_tokmean.npy"] = save_new(OUT / "x3_rerun518_tokmean.npy", np.ascontiguousarray(A["tok"]))
    files["x3_byproduct518_normmean.npy"] = save_new(OUT / "x3_byproduct518_normmean.npy", A["normmean"])
    del A, d1, d2, model
    torch.cuda.empty_cache()
    if any(stop.values()):
        meta["status"] = "STOPPED: pass A differs from the S1.1 stored arrays beyond the readings' threshold"
        meta["files_sha256"] = files
        write_json_new(OUT / "x3_extract.json", meta)
        raise SystemExit(meta["status"])

    # ---------------------------------------------------------------- pass B: added variant at 224 px
    model, linfo = build_224(sd)
    tf224, cfg224 = transform_224()
    bs224 = XE.vit_batch_size(int(sum(p.numel() for p in model.parameters())), 224)
    meta["passB"] = {"id": "x3add_schnaus224", "data_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg224.items()},
                     "transform": repr(tf224), "batch_size": bs224, "img_size": 224,
                     "pos_embed": "resampled 37x37 -> 16x16, F.interpolate bicubic, align_corners=False, antialias=False, float32 (HF Dinov2 style); prefix (CLS) position kept",
                     "pos_embed_shape": list(model.pos_embed.shape), "num_prefix_tokens": int(model.num_prefix_tokens),
                     "has_register_tokens": bool(getattr(model, "reg_token", None) is not None), **linfo,
                     "feature": "mean over all tokens of forward_features output (after final LayerNorm), float64 accumulate, stored float32"}
    t0 = time.time()
    B = run_pass(model, blobs, tf224, bs224, hooks=False)
    meta["passB"]["forward_s"] = time.time() - t0
    meta["passB"]["n_converted_to_rgb"] = B["n_converted_to_rgb"]
    meta["passB"]["n_tokens"] = B["n_tokens"]
    e1 = run_pass(model, blobs[:64], tf224, bs224, hooks=False)
    e2 = run_pass(model, blobs[:64], tf224, bs224, hooks=False)
    meta["passB"]["determinism_64"] = maxdiff(e1["normmean"], e2["normmean"])
    meta["passB"]["check64_vs_main"] = maxdiff(e1["normmean"], B["normmean"][:64])
    del model
    torch.cuda.empty_cache()
    try:
        mt = build_224_timm_resample()
        t = run_pass(mt, blobs[:64], tf224, bs224, hooks=False)
        meta["passB"]["side_note_timm_antialias_resample_64"] = {**maxdiff(t["normmean"], e1["normmean"]),
                                                               **cosines(t["normmean"], e1["normmean"])}
        del mt
    except Exception as e:
        meta["passB"]["side_note_timm_antialias_resample_64"] = {"error": repr(e)}
    a64 = B["normmean"].astype(np.float64)
    rn = np.linalg.norm(a64, axis=1)
    meta["passB"]["array_checks"] = {"finite": bool(np.isfinite(a64).all()), "zero_rows": int((rn == 0).sum()),
                                     "min_row_norm": float(rn.min()), "max_row_norm": float(rn.max())}
    if not meta["passB"]["array_checks"]["finite"] or meta["passB"]["array_checks"]["zero_rows"]:
        raise SystemExit("STOP: added-variant array has non-finite or zero rows")
    files["x3add_schnaus224.npy"] = save_new(OUT / "x3add_schnaus224.npy", B["normmean"])
    meta["files_sha256"] = files
    meta["shapes"] = {"x3_rerun518_cls.npy": [m, 2, 768], "x3_rerun518_tokmean.npy": [m, 2, 768],
                      "x3_byproduct518_normmean.npy": [m, 768], "x3add_schnaus224.npy": [m, 768]}
    meta["gpu_peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
    meta["wall_s"] = time.time() - t_start
    meta["gpu_hours_process_wall"] = meta["wall_s"] / 3600
    meta["cpu_core_hours"] = cpu_seconds() / 3600
    meta["status"] = "complete"
    meta["finished_utc"] = XC.utcnow()
    meta["code_sha256"] = {"x3_extract.py": sha256_file(__file__)}
    side = write_json_new(OUT / "x3_extract.json", meta)
    (OUT / "x3_extract.done").write_text(meta["finished_utc"] + "\n")
    man_out = {"files_sha256": {f"features/coco_e0/x3/{k}": v for k, v in files.items()},
               "sidecar": {"features/coco_e0/x3/x3_extract.json": side},
               "scoring_inputs": {"x3_cls_r10": "features/coco_pool/vit_base_patch14_dinov2.lvd142m.npy axis 3 (S1.1 stored)",
                                  "x3_tok_r10": "features/coco_pool/vit_base_patch14_dinov2.lvd142m.tokmean.npy axis 3 (S1.1 stored)",
                                  "x3_tok_r05": "features/coco_pool/vit_base_patch14_dinov2.lvd142m.tokmean.npy axis 1 (S1.1 stored)",
                                  "x3add_schnaus224": "features/coco_e0/x3/x3add_schnaus224.npy (new)"},
               "s11_stored_sha256": {"primary": s11meta["files_sha256"]["primary"]["sha256"],
                                     "tokmean": s11meta["files_sha256"]["tokmean"]["sha256"]},
               "passA_stop_rule": meta["passA"]["stop_rule"], "passA_vs_s11": cmp,
               "determinism": {"passA": meta["passA"]["determinism_64"], "passB": meta["passB"]["determinism_64"]},
               "gpu_hours_process_wall": meta["gpu_hours_process_wall"], "cpu_core_hours": meta["cpu_core_hours"],
               "finished_utc": meta["finished_utc"]}
    EXTRACTION.mkdir(parents=True, exist_ok=True)
    h = write_json_new(EXTRACTION / "EXTRACTION_X3.json", man_out)
    for p in list(OUT.iterdir()) + [EXTRACTION / "EXTRACTION_X3.json"]:
        os.chmod(p, 0o444)
    print(json.dumps({"EXTRACTION_X3.json": h, "gpu_h": meta["gpu_hours_process_wall"], "cpu_core_h": meta["cpu_core_hours"],
                      "stop_rule": meta["passA"]["stop_rule"], "passB_det": meta["passB"]["determinism_64"],
                      "side": meta["passB"].get("side_note_timm_antialias_resample_64")}, indent=1, default=str))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)
