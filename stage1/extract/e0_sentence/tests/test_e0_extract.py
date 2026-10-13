"""pytest for the E0 extraction code on synthetic data and tiny randomly initialized models (CPU only; nothing is
downloaded; no real encoder, caption file or E0 feature is touched). ANALYSIS_PLAN_E0.md v1.0 §10.

    OMP_NUM_THREADS=4 venv/bin/python -m pytest -q extract/e0_sentence/tests
"""
import hashlib
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import e0_captions as CAP  # noqa: E402
import e0_extract as EX  # noqa: E402
import e0_membership as MEM  # noqa: E402
import e0x_common as X  # noqa: E402

torch.set_num_threads(4)
DEV = torch.device("cpu")


# ---------------------------------------------------------------------------------------------- hashing / pins
def test_git_blob_sha1_known_value(tmp_path):
    p = tmp_path / "h.txt"
    p.write_bytes(b"hello\n")
    assert X.git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_verify_snapshot(tmp_path):
    (tmp_path / "a.json").write_bytes(b"{}\n")
    (tmp_path / "w.bin").write_bytes(b"\x00" * 100)
    pin = {"files": {"a.json": {"size": 3, "lfs_sha256": None, "git_blob": X.git_blob_sha1(tmp_path / "a.json")},
                     "w.bin": {"size": 100, "lfs_sha256": hashlib.sha256(b"\x00" * 100).hexdigest(), "git_blob": "x"}}}
    r = X.verify_snapshot(str(tmp_path), pin)
    assert all(v["ok"] for v in r.values())
    pin["files"]["w.bin"]["lfs_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        X.verify_snapshot(str(tmp_path), pin)


def test_pins_file_and_design_inputs():
    pins = X.load_pins()
    for enc, spec in X.ENCODERS.items():
        pin = pins[spec["repo"]]
        assert spec["weights"] in pin["files"] and pin["files"][spec["weights"]]["lfs_sha256"]
        assert not pin["gated"]


def test_no_write_over(tmp_path):
    p = tmp_path / "x.json"
    X.write_json_new(str(p), {"a": 1})
    with pytest.raises(FileExistsError):
        X.write_json_new(str(p), {"a": 2})


# ---------------------------------------------------------------------------------------------- tiny models
@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    from transformers import BertConfig, BertModel, BertTokenizerFast
    d = tmp_path_factory.mktemp("bert")
    words = "a man dog cat on the of in with two red sitting street table".split()
    (d / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words) + "\n")
    tok = BertTokenizerFast(vocab_file=str(d / "vocab.txt"))
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=5 + len(words), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=64)
    model = BertModel(cfg).eval()
    return model, tok


class FakeQwenTok:
    """Whitespace tokenizer that appends 151643 (as the Qwen3-Embedding post-processor does)."""
    padding_side = "right"

    def _ids(self, t):
        return [10 + (sum(map(ord, w)) % 500) for w in t.split()] + [151643]

    def __call__(self, text, return_tensors="pt", truncation=False, padding=False, max_length=None):
        texts = [text] if isinstance(text, str) else list(text)
        ids = [self._ids(t) for t in texts]
        n = max(map(len, ids))
        out_ids, mask = [], []
        for x in ids:
            pad = [0] * (n - len(x)) if padding else []
            if self.padding_side == "left":
                out_ids.append(pad + x)
                mask.append([0] * len(pad) + [1] * len(x))
            else:
                out_ids.append(x + pad)
                mask.append([1] * len(x) + [0] * len(pad))
        return {"input_ids": torch.tensor(out_ids), "attention_mask": torch.tensor(mask)}


@pytest.fixture(scope="module")
def tiny_qwen():
    from transformers import Qwen3Config, Qwen3Model
    torch.manual_seed(1)
    cfg = Qwen3Config(vocab_size=151700, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, intermediate_size=32, head_dim=8, max_position_embeddings=128)
    return Qwen3Model(cfg).eval(), FakeQwenTok()


def test_pool_and_card():
    h = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    assert torch.allclose(EX.pool_hidden(h, "mean"), torch.tensor([2.0, 4.0], dtype=torch.float64))
    assert torch.allclose(EX.pool_hidden(h, "last"), torch.tensor([3.0, 6.0], dtype=torch.float64))
    c = EX.card_from_raw(torch.tensor([3.0, 4.0], dtype=torch.float64), True)
    assert torch.allclose(c, torch.tensor([0.6, 0.8], dtype=torch.float64))
    assert torch.equal(EX.card_from_raw(torch.tensor([3.0, 4.0]), False), torch.tensor([3.0, 4.0]))


def test_embed_one_matches_manual_mean(tiny_bert):
    model, tok = tiny_bert
    spec = dict(X.ENCODERS["mpnet"])
    raw, card, nt, _ = EX.embed_one(model, tok, "a man on the street", spec, DEV)
    ids = tok("a man on the street", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        h = model(input_ids=ids).last_hidden_state[0].double()
    assert nt == ids.shape[1]
    assert np.allclose(raw, h.mean(0).numpy(), atol=1e-12)
    assert np.allclose(card, raw / np.linalg.norm(raw), atol=1e-12)


def test_run_checks_mean_models(tiny_bert):
    model, tok = tiny_bert
    texts = ["a man on the street", "two red dog", "a cat sitting on the table with a dog",
             "the man of the street"] * 20
    for enc in ("mpnet", "contriever"):
        r = EX.run_checks(model, tok, dict(X.ENCODERS[enc]), DEV, texts, None)
        assert r["padded_vs_batch1_min_cosine"] >= 1 - 1e-6 and r["determinism_max_abs"] <= 1e-6


def test_qwen_checks_stop_and_pass(tiny_qwen):
    model, tok = tiny_qwen
    spec = dict(X.ENCODERS["qwen3emb06"])
    texts = ["a man on the street", "two dogs"] * 40
    readme_bad = "AutoModel.from_pretrained\n" + EX.QWEN_TASK + " ".join(EX.QWEN_QUERIES + EX.QWEN_DOCS) + \
                 "\n# [[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]]"
    with pytest.raises(EX.Stop):
        EX.run_checks(model, tok, spec, DEV, texts, readme_bad)
    ex = EX.qwen_example_texts()
    e = EX.card_batch(model, tok, ex, spec, DEV)
    s = e[:2] @ e[2:].T
    readme_ok = "AutoModel.from_pretrained\n" + EX.QWEN_TASK + " ".join(EX.QWEN_QUERIES + EX.QWEN_DOCS) + \
        f"\n# [[{float(s[0, 0])!r}, {float(s[0, 1])!r}], [{float(s[1, 0])!r}, {float(s[1, 1])!r}]]"
    r = EX.run_checks(model, tok, spec, DEV, texts, readme_ok)
    assert r["card_example_ok"] and r["card_reference_source"] == "README"
    assert r["max_abs_batch1_vs_fp32_card_path"] <= 1e-4


def test_parse_card_scores():
    t = "x AutoModel.from_pretrained(...)\nprint(scores.tolist())\n# [[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]]"
    assert EX.parse_card_scores(t) == ((0.7645568251609802, 0.14142508804798126), (0.13549736142158508, 0.5999549627304077))
    assert EX.parse_card_scores("no model here") is None


def test_check_loading_info():
    assert EX.check_loading_info({"missing_keys": ["pooler.dense.weight"], "unexpected_keys": ["embeddings.position_ids"]})["ok"]
    assert not EX.check_loading_info({"missing_keys": ["encoder.layer.0.attention.self.query.weight"], "unexpected_keys": []})["ok"]


def synthetic_pool(n_items=12):
    words = "a man dog cat on the of in with two red sitting street table".split()
    rng = np.random.default_rng(3)
    items, rows = [], []
    for i in range(n_items):
        caps = [" ".join(rng.choice(words, size=rng.integers(3, 8))) for _ in range(5 + (i % 2))]
        c1 = caps[rng.integers(len(caps))] if i != 3 else "a dog on the table"   # item 3: C1 not official
        items.append({"idx": i, "image_id": 1000 + i, "text": c1})
        for r, t in enumerate(caps):
            rows.append({"idx": i, "image_id": 1000 + i, "ann_id": 50000 + 10 * i + r, "rank": r, "text": t})
    idx = pd.DataFrame(rows)
    idx.insert(0, "cap_row", np.arange(len(idx)))
    return pd.DataFrame(items), idx


def test_full_on_tiny_model(tiny_bert, tmp_path):
    model, tok = tiny_bert
    items, index = synthetic_pool()
    res = EX.full("mpnet", model, tok, DEV, items, index, str(tmp_path), log=lambda *a: None)
    assert res["ok"] and res["checks"]["c1_vs_caption_equality"]["n"] >= 10
    c1 = np.load(tmp_path / "mpnet.c1.npy")
    caps = np.load(tmp_path / "mpnet.caps.npy")
    assert c1.shape == (12, 16) and caps.shape == (len(index), 16) and c1.dtype == np.float32
    assert np.allclose(np.linalg.norm(c1, axis=1), 1, atol=1e-6)
    with pytest.raises(FileExistsError):
        EX.full("mpnet", model, tok, DEV, items, index, str(tmp_path), log=lambda *a: None)


def test_c1_caption_equality_detects_mismatch():
    items, index = synthetic_pool()
    rng = np.random.default_rng(0)
    caps = rng.normal(size=(len(index), 4)).astype(np.float32)
    c1 = np.zeros((12, 4), np.float32)
    tx = dict(zip(items["idx"], items["text"]))
    for r in index.itertuples():
        if tx[r.idx] == r.text:
            c1[r.idx] = caps[r.cap_row]
    assert EX.c1_caption_equality(c1, caps, items, index)["ok"]
    c1[0] += 1e-3
    assert not EX.c1_caption_equality(c1, caps, items, index)["ok"]


def test_smoke_projection(tiny_bert):
    model, tok = tiny_bert
    items, index = synthetic_pool()
    texts = (items["text"].tolist() * 50)[:600]
    r = EX.smoke("contriever", model, tok, DEV, texts, n_caps=1000, readme=None)
    assert r["n_smoke"] == 512 and r["projected_full_seconds"] > 0 and r["n_texts_full"] == 1600


# ---------------------------------------------------------------------------------------------- captions
def coco_json(items, index):
    anns = [{"id": int(r.ann_id), "image_id": int(r.image_id), "caption": " " + r.text + " "} for r in index.itertuples()]
    return {"annotations": anns, "images": [{"id": int(i)} for i in items["image_id"]]}


def test_build_index_checks():
    items, index = synthetic_pool()
    cj = coco_json(items, index)
    idx, chk = CAP.build_index(cj, items)
    assert chk["a_pass"] and len(idx) == len(index)
    assert abs(chk["b_frac_c1_equals_an_official_caption"] - 11 / 12) < 1e-12 and not chk["b_pass"]
    assert list(idx.columns) == ["cap_row", "idx", "image_id", "ann_id", "rank", "text"]
    assert (idx.groupby("idx")["ann_id"].apply(lambda s: s.is_monotonic_increasing)).all()
    cj["annotations"] = [a for a in cj["annotations"] if a["image_id"] != 1005]
    _, chk2 = CAP.build_index(cj, items)
    assert not chk2["a_pass"] and chk2["a_missing_image_ids"] == [1005]


def test_download_accept_reject_and_member(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("annotations/captions_val2014.json", json.dumps({"annotations": []}))
        z.writestr("annotations/other.json", "{}")
    data = buf.getvalue()
    md5 = hashlib.md5(data).hexdigest()

    class R_:
        def __init__(self, b):
            self.b = io.BytesIO(b)

        def read(self, n):
            return self.b.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    dst = str(tmp_path / "a.zip")
    info = CAP.download(dst, urls=("https://x",), size=len(data), md5=md5, opener=lambda u, timeout: R_(data))
    assert info["md5"] == md5 and os.path.exists(dst)
    with pytest.raises(SystemExit):
        CAP.download(str(tmp_path / "b.zip"), urls=("https://x",), size=len(data), md5="0" * 32,
                     opener=lambda u, timeout: R_(data))
    assert any(f.startswith("b.zip.rejected_") for f in os.listdir(tmp_path))

    def boom(u, timeout):
        raise ConnectionError("down")
    with pytest.raises(SystemExit):
        CAP.download(str(tmp_path / "c.zip"), urls=("https://x", "http://x"), size=1, md5="0" * 32, opener=boom)
    h = CAP.extract_member(dst, str(tmp_path / "cap.json"))
    assert h == hashlib.sha256(json.dumps({"annotations": []}).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------- Q1 and B7 pieces
def test_lm_run_reproduces_c1_reference(tmp_path):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(2)
    cfg = GPTNeoXConfig(vocab_size=600, hidden_size=16, num_hidden_layers=4, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=64)
    model = GPTNeoXForCausalLM(cfg).eval()

    class Tok:
        def __call__(self, text, return_tensors="pt"):
            ids = [10 + (sum(map(ord, w)) % 500) for w in text.split()]
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
    tok = Tok()
    items, index = synthetic_pool()
    c1_ref = np.stack([EX.lm_feature(model, tok, t, 2, DEV) for t in items["text"]]).astype(np.float32)
    r = EX.lm_run(model, tok, 2, DEV, index.sort_values("cap_row")["text"].tolist(), c1_ref, items, index,
                  str(tmp_path), "syn", log=lambda *a: None)
    assert r["c1_check_ok"] and r["c1_check_n"] >= 10 and os.path.exists(tmp_path / "syn.caps.npy")


def test_membership_counts(tmp_path):
    df = pd.DataFrame({"caption1": ["a dog", "two cats "], "caption2": ["x", None], "n": [1, 2]})
    p = tmp_path / "t.parquet"
    df.to_parquet(p)
    s, cols = MEM.string_set([str(p)])
    assert s == {"a dog", "two cats", "x"} and cols["t.parquet"]["string_columns"] == ["caption1", "caption2"]
    m = MEM.membership(s, ["a dog", "nope"], [" two cats", "x", "y"])
    assert m["c1_in_training_strings"] == 1 and m["captions_in_training_strings"] == 2
