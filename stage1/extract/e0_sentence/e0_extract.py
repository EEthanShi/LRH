"""E0 extraction of sentence-encoder card embeddings (ANALYSIS_PLAN_E0.md v1.0 §2.1, §4). Role e0-extract. GPU.

    venv/bin/python -I extract/e0_sentence/e0_extract.py smoke mpnet      # download+verify, checks V-d, 512-text timing
    venv/bin/python -I extract/e0_sentence/e0_extract.py full mpnet       # all C1 texts and all official captions
    venv/bin/python -I extract/e0_sentence/e0_extract.py finalize         # EXTRACTION_E0.json (sha256 -> STATUS)
    venv/bin/python -I extract/e0_sentence/e0_extract.py lm bloomz_1b7    # conditional Q1 only ('E0 Q1 APPROVED')
    venv/bin/python -I extract/e0_sentence/e0_extract.py lm-finalize

Conventions: fp32 (Qwen upcast from bf16), TF32 off, batch size 1, no padding, no truncation, pooling in float64,
stored float32. mpnet: token mean then l2; contriever: token mean, no normalization; qwen3emb06: last token
(must be id 151643) then l2. Every pinned file is verified (LFS sha256 / git blob sha1) before loading.
Exit 3 = STOP (a check failed); nothing is overwritten or deleted.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e0x_common as X  # noqa: E402

os.environ.setdefault("HF_HOME", X.HF_HOME)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

QWEN_TASK = "Given a web search query, retrieve relevant passages that answer the query"
QWEN_QUERIES = ("What is the capital of China?", "Explain gravity")
QWEN_DOCS = ("The capital of China is Beijing.",
             "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and "
             "is responsible for the movement of planets around the sun.")
QWEN_DESIGNER_REF = ((0.7646, 0.1414), (0.1355, 0.6000))
N_CHECK = 64
N_SMOKE = 512
ALLOWED_MISSING = (r"^pooler\.",)
ALLOWED_UNEXPECTED = (r"^lm_head\.", r"^pooler\.", r"(^|\.)position_ids$")


class Stop(Exception):
    """A plan check failed: STOP and report (exit 3)."""


# ---------------------------------------------------------------------------------------------- pure pieces
def pool_hidden(h, mode: str):
    """h: (T, d) tensor of one text -> float64 tensor (d,): token mean or last token (plan §4.1)."""
    import torch
    h64 = h.to(torch.float64)
    if mode == "mean":
        return h64.mean(dim=0)
    if mode == "last":
        return h64[-1]
    raise ValueError(mode)


def card_from_raw(raw, normalize: bool):
    import torch
    if not normalize:
        return raw.clone()
    n = torch.linalg.vector_norm(raw)
    if float(n) == 0.0:
        raise Stop("zero pooled vector")
    return raw / n


def embed_one(model, tok, text: str, spec: dict, device):
    """Batch 1, no padding, no truncation -> (raw float64 np (d,), card float64 np (d,), n_tokens, last id)."""
    import torch
    enc = tok(text, return_tensors="pt", truncation=False, padding=False)
    ids = enc["input_ids"]
    am = torch.ones_like(ids)
    with torch.inference_mode():
        out = model(input_ids=ids.to(device), attention_mask=am.to(device))
    h = out[0][0]
    raw = pool_hidden(h, spec["pooling"])
    card = card_from_raw(raw, spec["normalize"])
    return raw.cpu().numpy(), card.cpu().numpy(), int(ids.shape[1]), int(ids[0, -1])


def embed_many(model, tok, texts, spec, device, label="", every=5000, log=print):
    n = len(texts)
    raw = None
    t0 = time.perf_counter()
    ntok = np.empty(n, np.int64)
    last = np.empty(n, np.int64)
    for i, t in enumerate(texts):
        r, c, nt, li = embed_one(model, tok, t, spec, device)
        if raw is None:
            raw = np.empty((n, r.size), np.float32)
            card = np.empty((n, r.size), np.float32)
        raw[i], card[i], ntok[i], last[i] = r, c, nt, li
        if every and (i + 1) % every == 0:
            log(f"{X.utcnow()} {label} {i + 1}/{n} {(time.perf_counter() - t0) / (i + 1) * 1e3:.2f} ms/text")
    return raw, card, ntok, last, time.perf_counter() - t0


def card_batch(model, tok, texts, spec, device):
    """The card's own padded-batch path (masked mean / left-padded last token), for the checks of plan §4.2."""
    import torch
    import torch.nn.functional as F
    if spec["pooling"] == "last":
        tok.padding_side = "left"
        enc = tok(list(texts), padding=True, truncation=True, max_length=spec["max_tokens"], return_tensors="pt")
    else:
        enc = tok(list(texts), padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc)
    h = out[0]
    m = enc["attention_mask"]
    if spec["pooling"] == "last":
        left = bool(m[:, -1].sum() == m.shape[0])
        if left:
            e = h[:, -1]
        else:
            e = h[torch.arange(h.shape[0], device=h.device), m.sum(dim=1) - 1]
    else:
        mf = m.unsqueeze(-1).to(h.dtype)
        e = (h * mf).sum(dim=1) / torch.clamp(mf.sum(dim=1), min=1e-9)
    if spec["normalize"]:
        e = F.normalize(e, p=2, dim=1)
    return e.to(torch.float64).cpu().numpy()


def parse_card_scores(readme: str):
    """First 2x2 list of floats printed after 'AutoModel.from_pretrained' in the README, or None."""
    i = readme.find("AutoModel.from_pretrained")
    if i < 0:
        return None
    m = re.search(r"\[\[\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*\]\s*,\s*\[\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*\]\]",
                  readme[i:])
    if not m:
        return None
    v = [float(x) for x in m.groups()]
    return ((v[0], v[1]), (v[2], v[3]))


def qwen_example_texts():
    q = [f"Instruct: {QWEN_TASK}\nQuery:{x}" for x in QWEN_QUERIES]
    return q + list(QWEN_DOCS)


def check_loading_info(info: dict) -> dict:
    miss = list(info.get("missing_keys", []))
    unex = list(info.get("unexpected_keys", []))
    bad_m = [k for k in miss if not any(re.search(p, k) for p in ALLOWED_MISSING)]
    bad_u = [k for k in unex if not any(re.search(p, k) for p in ALLOWED_UNEXPECTED)]
    mism = list(info.get("mismatched_keys", []))
    return {"missing": miss, "unexpected": unex, "mismatched": [str(x) for x in mism],
            "ok": not bad_m and not bad_u and not mism}


def c1_caption_equality(c1_card, caps_card, items: pd.DataFrame, index: pd.DataFrame) -> dict:
    """Plan §4.2: for every pool item whose C1 equals an official caption string, both embeddings agree."""
    tx = dict(zip(items["idx"], items["text"]))
    sel = index[[tx[i] == t for i, t in zip(index["idx"], index["text"])]]
    if len(sel) == 0:
        return {"n": 0, "max_abs_diff": None, "ok": True}
    d = np.abs(c1_card[sel["idx"].to_numpy()].astype(np.float64) - caps_card[sel["cap_row"].to_numpy()].astype(np.float64))
    mx = float(d.max())
    return {"n": int(len(sel)), "max_abs_diff": mx, "ok": mx <= 1e-6}


# ---------------------------------------------------------------------------------------------- checks + runs
def run_checks(model, tok, spec, device, c1_texts, readme: str | None, load_bf16=None, log=print) -> dict:
    """Plan §4.2 model checks (card example / padded vs batch-1 / determinism). Raises Stop on failure."""
    res = {}
    texts = list(c1_texts[:N_CHECK])
    b1 = np.stack([embed_one(model, tok, t, spec, device)[1] for t in texts])
    b1b = np.stack([embed_one(model, tok, t, spec, device)[1] for t in texts])
    det = float(np.abs(b1 - b1b).max())
    res["determinism_max_abs"] = det
    if det > 1e-6:
        raise Stop(f"determinism {det} > 1e-6")
    if spec["pooling"] == "last":
        ex = qwen_example_texts()
        if readme is not None:
            missing = [s for s in (QWEN_TASK,) + QWEN_QUERIES + QWEN_DOCS if s not in readme]
            res["card_texts_missing_from_readme"] = missing
            if missing:
                raise Stop(f"card example texts not found in README: {missing}")
        ref = parse_card_scores(readme) if readme else None
        res["card_reference"] = ref if ref else QWEN_DESIGNER_REF
        res["card_reference_source"] = "README" if ref else "designer reading"
        ref = np.asarray(res["card_reference"], float)
        e = card_batch(model, tok, ex, spec, device)
        s32 = e[:2] @ e[2:].T
        eb = np.stack([embed_one(model, tok, t, spec, device)[1] for t in ex])
        sb1 = eb[:2] @ eb[2:].T
        res["card_scores_fp32"] = s32.tolist()
        res["batch1_scores_fp32"] = sb1.tolist()
        res["max_abs_fp32_vs_card"] = float(np.abs(s32 - ref).max())
        res["max_abs_batch1_vs_fp32_card_path"] = float(np.abs(sb1 - s32).max())
        ok_ref = res["max_abs_fp32_vs_card"] <= 5e-3
        if not ok_ref and load_bf16 is not None:
            m16 = load_bf16()
            e16 = card_batch(m16, tok, ex, spec, device)
            s16 = e16[:2] @ e16[2:].T
            res["card_scores_bf16"] = s16.tolist()
            res["max_abs_bf16_vs_card"] = float(np.abs(s16 - ref).max())
            ok_ref = res["max_abs_bf16_vs_card"] <= 5e-3
            del m16
        res["card_example_ok"] = bool(ok_ref)
        if not ok_ref:
            raise Stop("Qwen card example not reproduced within 5e-3 in fp32 or bf16")
        if res["max_abs_batch1_vs_fp32_card_path"] > 1e-4:
            raise Stop("batch-1 path differs from the fp32 card path by more than 1e-4")
        lastids = [embed_one(model, tok, t, spec, device)[3] for t in texts]
        if any(i != spec["last_token_id"] for i in lastids):
            raise Stop("a tokenized input does not end with <|endoftext|> (151643)")
    else:
        e = card_batch(model, tok, texts, spec, device)
        cos = np.sum(e * b1, axis=1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(b1, axis=1))
        res["padded_vs_batch1_min_cosine"] = float(cos.min())
        if cos.min() < 1 - 1e-6:
            raise Stop(f"padded-batch card path vs batch-1: min cosine {cos.min()}")
    return res


def smoke(enc: str, model, tok, device, c1_texts, n_caps: int, readme, load_bf16=None, log=print) -> dict:
    spec = X.ENCODERS[enc]
    res = {"encoder": enc, "checks": run_checks(model, tok, spec, device, c1_texts, readme, load_bf16, log)}
    texts = list(c1_texts[:N_SMOKE])
    _, _, ntok, _, sec = embed_many(model, tok, texts, spec, device, label=f"{enc} smoke", every=0)
    per = sec / len(texts)
    n_total = len(c1_texts) + n_caps
    res.update({"n_smoke": len(texts), "seconds": sec, "ms_per_text": per * 1e3, "n_texts_full": n_total,
                "projected_full_seconds": per * n_total, "projected_full_gpu_hours": per * n_total / 3600,
                "max_tokens_smoke": int(ntok.max()), "note": "estimate only; no cap (author rule)"})
    return res


def full(enc: str, model, tok, device, items: pd.DataFrame, index: pd.DataFrame, feat_dir: str, log=print) -> dict:
    spec = X.ENCODERS[enc]
    items = items.sort_values("idx").reset_index(drop=True)
    c1_texts = items["text"].tolist()
    cap_texts = index.sort_values("cap_row")["text"].tolist()
    paths = {k: os.path.join(feat_dir, f"{enc}.{k}.npy") for k in ("c1", "c1_raw", "caps", "caps_raw")}
    for p in list(paths.values()) + [os.path.join(feat_dir, f"{enc}.done")]:
        if os.path.exists(p):
            raise FileExistsError(p)
    r1, c1, nt1, l1, s1 = embed_many(model, tok, c1_texts, spec, device, label=f"{enc} c1", log=log)
    r2, c2, nt2, l2, s2 = embed_many(model, tok, cap_texts, spec, device, label=f"{enc} caps", log=log)
    checks = {"max_tokens_c1": int(nt1.max()), "max_tokens_caps": int(nt2.max()), "limit": spec["max_tokens"]}
    checks["no_truncation_ok"] = bool(max(nt1.max(), nt2.max()) <= spec["max_tokens"])
    if spec["pooling"] == "last":
        checks["all_end_with_151643"] = bool(np.all(l1 == spec["last_token_id"]) and np.all(l2 == spec["last_token_id"]))
    else:
        checks["all_end_with_151643"] = None
    checks["c1_vs_caption_equality"] = c1_caption_equality(c1, c2, items, index)
    checks["finite"] = bool(np.isfinite(c1).all() and np.isfinite(c2).all())
    ok = checks["no_truncation_ok"] and checks["c1_vs_caption_equality"]["ok"] and checks["finite"] and \
        checks["all_end_with_151643"] in (True, None)
    hashes = {}
    os.makedirs(feat_dir, exist_ok=True)
    for k, arr in (("c1", c1), ("c1_raw", r1), ("caps", c2), ("caps_raw", r2)):
        hashes[os.path.basename(paths[k])] = X.save_npy_new(paths[k], arr)
    nt = os.path.join(feat_dir, f"{enc}.ntok.npz")
    np.savez(nt + ".tmp.npz", c1=nt1, caps=nt2, c1_last_id=l1, caps_last_id=l2)
    os.replace(nt + ".tmp.npz", nt)
    hashes[os.path.basename(nt)] = X.sha256_file(nt)
    return {"encoder": enc, "checks": checks, "ok": bool(ok), "seconds_c1": s1, "seconds_caps": s2,
            "n_c1": len(c1_texts), "n_caps": len(cap_texts), "d": int(c1.shape[1]), "files_sha256": hashes}


# ---------------------------------------------------------------------------------------------- real-model wiring
def device_of():
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare(enc: str) -> tuple[str, dict]:
    from huggingface_hub import snapshot_download
    pins = X.load_pins()
    spec = X.ENCODERS[enc]
    pin = pins[spec["repo"]]
    if pin.get("gated") or pin.get("private") or pin.get("disabled"):
        raise Stop(f"{spec['repo']} is gated/private/disabled at pin time")
    snap = snapshot_download(spec["repo"], revision=pin["revision"], allow_patterns=sorted(pin["files"]),
                             cache_dir=X.HUB_DIR, token=False)
    if os.path.basename(snap.rstrip("/")) != pin["revision"]:
        raise Stop(f"snapshot {snap} is not the pinned revision {pin['revision']}")
    return snap, X.verify_snapshot(snap, pin)


def load_encoder(snap: str, dtype, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    model, info = AutoModel.from_pretrained(snap, dtype=dtype, local_files_only=True, output_loading_info=True)
    model = model.to(device).eval()
    return tok, model, info


def _inputs():
    items = pd.read_parquet(X.ITEMS, columns=["idx", "image_id", "text"]).sort_values("idx").reset_index(drop=True)
    cpath = os.path.join(X.FEAT, "captions_checks.json")
    ipath = os.path.join(X.FEAT, "captions_index.parquet")
    if not (os.path.exists(cpath) and os.path.exists(ipath)):
        raise SystemExit("REFUSED: run e0_captions.py first")
    with open(cpath) as f:
        cc = json.load(f)
    if not (cc["a_pass"] and cc["b_pass"]) or X.sha256_file(ipath) != cc["index_sha256"]:
        raise SystemExit("REFUSED: caption checks failed or the index changed")
    return items, pd.read_parquet(ipath), cc


def cmd_model(enc: str, mode: str):
    import torch
    frozen = X.require_plan_and_code_frozen()
    items, index, cc = _inputs()
    flags = X.set_strict_fp32()
    dev = device_of()
    t0 = time.perf_counter()
    snap, verified = prepare(enc)
    t_dl = time.perf_counter() - t0
    tok, model, info = load_encoder(snap, torch.float32, dev)
    li = check_loading_info(info)
    if not li["ok"]:
        raise Stop(f"loading info outside the allowed patterns: {li}")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise Stop("model is not fp32")
    readme = open(os.path.join(snap, "README.md"), encoding="utf-8").read() if os.path.exists(os.path.join(snap, "README.md")) else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    common = {"encoder": enc, "repo": X.ENCODERS[enc]["repo"], "snapshot": snap, "verified_files": verified,
              "download_seconds": t_dl, "loading_info": li, "fp32_flags": flags, "env": X.env_info(),
              "attn_implementation": getattr(model.config, "_attn_implementation", None), "frozen": frozen,
              "code_sha256": {f: X.sha256_file(os.path.join(HERE, f)) for f in ("e0_extract.py", "e0x_common.py")},
              "captions_index_sha256": cc["index_sha256"], "items_sha256": X.sha256_file(X.ITEMS), "utc": X.utcnow()}
    if mode == "smoke":
        def load_bf16():
            return load_encoder(snap, torch.bfloat16, dev)[1]
        res = smoke(enc, model, tok, dev, items["text"].tolist(), len(index), readme, load_bf16)
        res.update(common)
        res["gpu_peak_mem_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        h = X.write_json_new(os.path.join(X.EXTRACTION, f"smoke_{enc}_{stamp}.json"), res)
        print(f"smoke {enc}: {res['ms_per_text']:.2f} ms/text, projected {res['projected_full_gpu_hours']:.3f} GPU-h; sha256 {h}")
        return
    smokes = sorted(glob.glob(os.path.join(X.EXTRACTION, f"smoke_{enc}_*.json")))
    if not smokes:
        raise SystemExit(f"REFUSED: no smoke file for {enc}; run the smoke first and log its projection in STATUS")
    if not X.status_has(re.escape(os.path.basename(smokes[-1]))):
        raise SystemExit(f"REFUSED: the smoke projection {os.path.basename(smokes[-1])} is not logged in STATUS (plan §4.2)")
    res = full(enc, model, tok, dev, items, index, X.FEAT)
    res.update(common)
    res["smoke_file"] = os.path.relpath(smokes[-1], X.ROOT)
    res["gpu_peak_mem_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    res["pooling"] = X.ENCODERS[enc]
    res["dtype"] = "float32 weights and activations; pooling float64; stored float32"
    X.write_json_new(os.path.join(X.FEAT, f"{enc}.json"), res)
    if not res["ok"]:
        print(json.dumps(X.jsonable(res["checks"]), indent=1))
        raise Stop(f"{enc}: extraction checks failed (no .done written)")
    X.write_new(os.path.join(X.FEAT, f"{enc}.done"), X.utcnow() + "\n")
    print(f"full {enc}: done; c1 {res['seconds_c1']:.0f}s caps {res['seconds_caps']:.0f}s")


def cmd_finalize():
    X.require_plan_and_code_frozen()
    for enc in X.ENCODERS:
        if not os.path.exists(os.path.join(X.FEAT, f"{enc}.done")):
            raise SystemExit(f"REFUSED: {enc} not done")
    with open(os.path.join(X.FEAT, "captions_checks.json")) as f:
        cc = json.load(f)
    files = {}
    for p in sorted(glob.glob(os.path.join(X.FEAT, "*"))):
        if os.path.isfile(p):
            files[os.path.relpath(p, X.ROOT)] = X.sha256_file(p)
    gpu_s = 0.0
    checks = {}
    for enc in X.ENCODERS:
        with open(os.path.join(X.FEAT, f"{enc}.json")) as f:
            sc = json.load(f)
        checks[enc] = {"ok": sc["ok"], **sc["checks"]}
        gpu_s += sc["seconds_c1"] + sc["seconds_caps"]
    smokes = {os.path.relpath(p, X.ROOT): X.sha256_file(p) for p in sorted(glob.glob(os.path.join(X.EXTRACTION, "smoke_*.json")))}
    man = {"plan": "ANALYSIS_PLAN_E0.md v1.0 §4", "utc": X.utcnow(), "files_sha256": files, "captions_checks": cc,
           "encoder_checks": checks, "smoke_files_sha256": smokes, "embedding_gpu_seconds": gpu_s,
           "embedding_gpu_hours": gpu_s / 3600}
    h = X.write_json_new(os.path.join(X.EXTRACTION, "EXTRACTION_E0.json"), man)
    print(f"EXTRACTION_E0.json sha256 {h}; embedding GPU-h {gpu_s / 3600:.3f}")


# ---------------------------------------------------------------------------------------------- conditional Q1 (plan §15)
def lm_snapshot(lmkey: str) -> str:
    repo, rev, _ = X.Q1_LMS[lmkey]
    snap = os.path.join(X.SHARED_HUB, "models--" + repo.replace("/", "--"), "snapshots", rev)
    if not os.path.isdir(snap):
        raise SystemExit(f"REFUSED: local snapshot {snap} missing (Q1 never downloads)")
    return snap


def lm_feature(model, tok, text: str, ell: int, device):
    """S1.1 convention (PROTOCOL §4): batch 1, no padding, token mean of hidden_states[ell] in float64."""
    import torch
    enc = tok(text, return_tensors="pt")
    with torch.inference_mode():
        o = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device),
                  output_hidden_states=True, use_cache=False)
    return o.hidden_states[ell][0].to(torch.float64).mean(dim=0).cpu().numpy()


def lm_run(model, tok, ell, device, cap_texts, c1_ref, items, index, out_dir, slug, log=print) -> dict:
    n = len(cap_texts)
    feats = None
    t0 = time.perf_counter()
    for i, t in enumerate(cap_texts):
        v = lm_feature(model, tok, t, ell, device)
        if feats is None:
            feats = np.empty((n, v.size), np.float32)
        feats[i] = v
        if (i + 1) % 5000 == 0:
            log(f"{X.utcnow()} {slug} {i + 1}/{n}")
    sec = time.perf_counter() - t0
    tx = dict(zip(items["idx"], items["text"]))
    sel = index[[tx[i] == t for i, t in zip(index["idx"], index["text"])]]
    a = feats[sel["cap_row"].to_numpy()].astype(np.float64)
    b = np.asarray(c1_ref[sel["idx"].to_numpy()], np.float64)
    rel = float(np.max(np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-30))) if len(sel) else None
    os.makedirs(out_dir, exist_ok=True)
    h = X.save_npy_new(os.path.join(out_dir, f"{slug}.caps.npy"), feats)
    return {"n_caps": n, "seconds": sec, "c1_check_n": int(len(sel)), "c1_check_max_rel": rel,
            "c1_check_ok": rel is None or rel <= 1e-5, "caps_sha256": h}


def cmd_lm(lmkey: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    X.require_plan_and_code_frozen()
    if not X.status_has(r"E0 Q1 APPROVED"):
        raise SystemExit("REFUSED: conditional arm Q1 needs a STATUS line 'E0 Q1 APPROVED' (plan §15)")
    items, index, cc = _inputs()
    repo, rev, slug = X.Q1_LMS[lmkey]
    meta = json.load(open(os.path.join(X.ROOT, "features", "coco_pool", slug + ".json")))
    ell = int(meta["rel_depths_to_residual_index"]["0.5"])
    ax = meta["layer_axis_hidden_state_indices"].index(ell)
    if ax != 1:
        raise Stop("rel 0.5 is not axis 1 of the S1.1 array")
    npy = os.path.join(X.ROOT, "features", "coco_pool", slug + ".npy")
    if X.sha256_file(npy) != meta["files_sha256"]["primary"]["sha256"]:
        raise Stop("S1.1 LM feature file differs from its sidecar")
    c1_ref = np.load(npy, mmap_mode="r")[:, 1, :]
    flags = X.set_strict_fp32()
    dev = device_of()
    snap = lm_snapshot(lmkey)
    try:
        tok = AutoTokenizer.from_pretrained(snap, use_fast=True)
        fast = True
    except ValueError:
        tok = AutoTokenizer.from_pretrained(snap, use_fast=False)
        fast = False
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32, local_files_only=True).to(dev).eval()
    out_dir = os.path.join(X.FEAT, "lm")
    res = lm_run(model, tok, ell, dev, index.sort_values("cap_row")["text"].tolist(), c1_ref, items, index, out_dir, slug)
    res.update({"lm": lmkey, "repo": repo, "revision": rev, "snapshot": snap, "hidden_state_index": ell, "fast_tokenizer": fast,
                "fp32_flags": flags, "env": X.env_info(), "utc": X.utcnow(),
                "code_sha256": {f: X.sha256_file(os.path.join(HERE, f)) for f in ("e0_extract.py", "e0x_common.py")}})
    X.write_json_new(os.path.join(out_dir, f"{slug}.json"), res)
    if not res["c1_check_ok"]:
        raise Stop(f"{lmkey}: caption features do not reproduce the S1.1 C1 features (max rel {res['c1_check_max_rel']})")
    X.write_new(os.path.join(out_dir, f"{slug}.done"), X.utcnow() + "\n")
    print(f"lm {lmkey}: done, {res['seconds']:.0f}s, C1 check max rel {res['c1_check_max_rel']}")


def cmd_lm_finalize():
    out_dir = os.path.join(X.FEAT, "lm")
    for lm, (_, _, slug) in X.Q1_LMS.items():
        if not os.path.exists(os.path.join(out_dir, f"{slug}.done")):
            raise SystemExit(f"REFUSED: {lm} not done")
    files = {os.path.relpath(p, X.ROOT): X.sha256_file(p) for p in sorted(glob.glob(os.path.join(out_dir, "*"))) if os.path.isfile(p)}
    h = X.write_json_new(os.path.join(X.EXTRACTION, "LMAVG_E0.json"), {"utc": X.utcnow(), "files_sha256": files,
                                                                      "plan": "ANALYSIS_PLAN_E0.md v1.0 §15 Q1"})
    print(f"LMAVG_E0.json sha256 {h}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("smoke", "full", "finalize", "lm", "lm-finalize"))
    ap.add_argument("name", nargs="?")
    a = ap.parse_args(argv)
    try:
        if a.cmd in ("smoke", "full"):
            if a.name not in X.ENCODERS:
                raise SystemExit(f"unknown encoder {a.name}")
            cmd_model(a.name, a.cmd)
        elif a.cmd == "finalize":
            cmd_finalize()
        elif a.cmd == "lm":
            if a.name not in X.Q1_LMS:
                raise SystemExit(f"unknown LM {a.name}")
            cmd_lm(a.name)
        else:
            cmd_lm_finalize()
    except Stop as e:
        print(f"STOP: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
