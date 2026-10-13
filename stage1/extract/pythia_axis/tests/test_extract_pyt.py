"""S0b CPU tests of the Pythia-axis extraction code (role pyt-axis-extract, 2026-10-10). Not citable.

Run only in S0b test mode, through the launcher, CPU only (the GPU is hidden):
  PYT_S0B_TEST=1 analysis/s12_layermax_pythia/launch_pyt.sh pyt_s0b_cputest 4 -- \
      venv/bin/python extract/pythia_axis/tests/test_extract_pyt.py
Outputs only under extract/pythia_axis/s0b_test_out/ (report: test_report_cpu.json). No network, no Pythia byte,
no name of an axis output (EleutherAI__pythia-<size>...). Synthetic data and the existing WIT captions only; the
F-cache test stat-lists the standard hub by name (no content read), exactly as plan section 2 F-cache.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only
os.environ["PYT_DEVICE"] = "cpu"
os.environ["HF_HUB_OFFLINE"] = "1"
assert os.environ.get("PYT_S0B_TEST") == "1", "run with PYT_S0B_TEST=1"
HERE = Path(__file__).resolve().parent
AXIS = HERE.parent
PROJ = AXIS.parents[1]
sys.path[:0] = [str(HERE), str(AXIS), str(AXIS.parent), str(PROJ / "analysis" / "s12_layermax_pythia")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("PYT_THREADS", "4")))
import pyt_axis_common as C  # noqa: E402
import registry_pythia_axis as R  # noqa: E402

assert C.TEST_MODE and str(C.RES_DIR).startswith(str(C.TEST_ROOT))
T0 = time.time()
REPORT = {"started_utc": C.utcnow(), "tests": {}}
VENV_PY = str(PROJ / "venv" / "bin" / "python")


def test(fn):
    name = fn.__name__
    t = time.time()
    try:
        det = fn() or {}
        REPORT["tests"][name] = dict(pass_=True, details=det, s=round(time.time() - t, 2))
        print(f"PASS {name} ({time.time() - t:.1f} s) {json.dumps(det, default=str)[:600]}", flush=True)
    except Exception as e:  # noqa: BLE001
        REPORT["tests"][name] = dict(pass_=False, error=repr(e), tb=traceback.format_exc()[-3000:],
                                     s=round(time.time() - t, 2))
        print(f"FAIL {name}: {e!r}\n{traceback.format_exc()}", flush=True)
    return fn


def texts_all():
    import pandas as pd
    return pd.read_parquet(PROJ / "features" / "wit1024" / "items.parquet")["text"].astype(str).tolist()


TEXTS = texts_all()
SNAPROOT = C.TEST_ROOT / "synthetic_snaps"
RUNID = C.utcnow().replace("-", "").replace(":", "")  # outputs are never overwritten: one directory per run
RUNROOT = C.TEST_ROOT / "runs" / RUNID
TOKJ = None


def snap(variant: str, rev_char: str, seed: int = 0):
    global TOKJ
    import make_synthetic as MS
    if TOKJ is None:
        TOKJ = MS.build_tokenizer(TEXTS)
    p = SNAPROOT / variant / (rev_char * 40)
    if (p / "pytorch_model.bin").exists():
        return p
    return MS.make_snapshot(SNAPROOT / variant, rev_char * 40, TEXTS, variant=variant, seed=seed, tokenizer_json=TOKJ)


# ------------------------------------------------------------------------------------------------------------ T1
@test
def t01_registry():
    assert R.SIZES == ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]
    assert sum(sum(b for _, b, _ in R.WEIGHT_FILES[m]) for m in R.LM_PYT_AXIS) == 26007147158
    assert sum(d["bytes"] for d in R.DOWNLOADS) == 31574762557
    assert [(R.SIZE_OF[d["model"]], d["kind"]) for d in R.DOWNLOADS] == [
        ("70m", "main"), ("160m", "main"), ("410m", "main"), ("1b", "main"), ("1.4b", "main"), ("2.8b", "main"),
        ("2.8b", "step143000"), ("6.9b", "main")]
    vocab = {m: (50432 if m.endswith("6.9b") else 50304) for m in R.LM_PYT_AXIS}
    for m in R.LM_PYT_AXIS:
        e = R.EXPECTED[m]
        cfg = dict(hidden_size=e["hidden_size"], num_hidden_layers=e["n_layers"], vocab_size=vocab[m])
        assert R.expected_num_params_from_config(cfg) == e["num_params"], m
        assert "pytorch_model.bin" in R.ALLOW[m][0] or m.endswith("6.9b")
        assert not any(f.endswith((".safetensors", ".py")) for f in R.ALLOW[m])
    assert abs(R.gpu_need_bytes("EleutherAI/pythia-6.9b") / 1e9 - 29.429208064) < 1e-9
    assert R.PROV_ALLOW == ["pytorch_model.bin", "README.md", "config.json", "special_tokens_map.json", "tokenizer.json",
                            "tokenizer_config.json"]
    assert R.UNEXPECTED_OK_RE.match("gpt_neox.layers.3.attention.rotary_emb.inv_freq")
    assert not R.MISSING_OK_RE.match("gpt_neox.layers.3.attention.rotary_emb.inv_freq")
    assert not R.UNEXPECTED_OK_RE.match("gpt_neox.layers.3.attention.dense.weight")
    return dict(downloads=[(d["model"], d["kind"], d["bytes"]) for d in R.DOWNLOADS])


# ------------------------------------------------------------------------------------------------------------ T2
@test
def t02_mock_s1():
    import download_pyt as D
    from tests_mock_hub import MockApi
    design = PROJ / "analysis" / "s12_layermax_pythia" / "design"
    res = D.resolve_all(MockApi(design), MockApi(design).fetch_config)
    assert res["pass"], res["problems"]
    out = dict(clean=dict(problems=res["problems"], n_models=len(res["models"])))
    expect = dict(gated="160m: gated", lfs="410m: weight pytorch_model.bin", step_differs="1b: main .bin",
                  branches="1.4b: refs", license="2.8b: card licence", converts="70m: converts", f16="6.9b: Hub F16",
                  auth="auth/gating/licence error")
    for fault, sub in expect.items():
        r = D.resolve_all(MockApi(design, fault), MockApi(design).fetch_config)
        assert not r["pass"] and any(sub in p for p in r["problems"]), (fault, r["problems"])
        out[fault] = r["problems"][:2]
    return out


# ------------------------------------------------------------------------------------------------------------ T3
@test
def t03_config_check():
    import download_pyt as D
    import make_synthetic as MS
    out = {}
    for m in R.LM_PYT_AXIS:
        e = R.EXPECTED[m]
        cfg = MS.raw_config(L=e["n_layers"], d=e["hidden_size"], heads=e["n_heads"],
                            vocab=(50432 if m.endswith("6.9b") else 50304))
        cc = D.check_config(m, cfg)
        assert cc["pass_"], (m, cc)
        bad = dict(cfg, num_hidden_layers=e["n_layers"] + 1)
        assert not D.check_config(m, bad)["pass_"]
        out[R.SIZE_OF[m]] = cc["num_params_from_config"]
    return out


# ------------------------------------------------------------------------------------------------------------ T4
@test
def t04_disk_guard():
    import download_pyt as D
    rows = []
    for i, d in enumerate(R.DOWNLOADS):
        later = sum(x["bytes"] for x in R.DOWNLOADS[i + 1:])
        edge = 20 * 10**9 + d["bytes"] + later + 6 * 10**9
        ok = D.disk_check(i, free_bytes=edge)
        bad = D.disk_check(i, free_bytes=edge - 1)
        assert ok["ok"] and not bad["ok"], (i, ok, bad)
        rows.append((R.SIZE_OF[d["model"]], d["kind"], edge))
    first = D.disk_check(0, free_bytes=880_042_905_600)
    assert first["ok"]
    return dict(min_free_needed_bytes=rows, at_freeze_left_gb=first["left_bytes"] / 1e9)


# ------------------------------------------------------------------------------------------------------------ T5
@test
def t05_download_e1_fake():
    import download_pyt as D
    import make_synthetic as MS
    from tests_mock_hub import fake_downloader
    fake = C.TEST_ROOT / "fake_repos"
    items = []
    for name, ch in (("synthetic/neox-a", "a"), ("synthetic/neox-b", "b")):
        p = MS.make_snapshot(fake / name.replace("/", "--"), ch * 40, TEXTS, variant="main", seed=1,
                             tokenizer_json=(TOKJ or MS.build_tokenizer(TEXTS)))
        ids = MS.file_ids(p)
        w = [("pytorch_model.bin", ids["pytorch_model.bin"]["size"], ids["pytorch_model.bin"]["sha256"])]
        sm = sorted((f, v["size"], v["blob"]) for f, v in ids.items() if f != "pytorch_model.bin")
        allow = [f for f, _, _ in w] + [f for f, _, _ in sm]
        items.append(dict(model=name, revision=ch * 40, kind="main", allow=allow, weights=w, small=sm,
                          bytes=sum(x[1] for x in w) + sum(x[1] for x in sm)))
        cfg = json.load(open(p / "config.json"))
        R.EXPECTED[name] = dict(n_layers=3, stored_layers=4, hidden_size=64, n_heads=4,
                                num_params=R.expected_num_params_from_config(cfg))
    saved = R.DOWNLOADS[:]
    R.DOWNLOADS[:] = items
    try:
        dl, redl = fake_downloader(str(fake), C.PYT_HUB)
        corrupt = {"done": False}

        def dl2(item, mw):  # item b arrives corrupted once -> one re-download
            s = dl(item, mw)
            if item["model"].endswith("neox-b") and not corrupt["done"]:
                blob = (s / "pytorch_model.bin").resolve()
                data = bytearray(blob.read_bytes())
                data[-100] ^= 0xFF
                blob.write_bytes(bytes(data))
                corrupt["done"] = True
            return s
        D.download_all(dl2, redl)
        e1a = C.read_json(C.RES_DIR / "E1_synthetic__neox-a.json")
        e1b = C.read_json(C.RES_DIR / "E1_synthetic__neox-b.json")
        assert e1a["E1_pass"] and e1b["E1_pass"], (e1a.get("shards"), e1b.get("shards"))
        assert e1b["retried"] == ["pytorch_model.bin"] and e1a["retried"] == []
        assert e1a["n_buffer_tensors"] == 9 and e1a["non_buffer_all_f16"] and e1a["n_params_diff"] == 0
        D.download_all(dl2, redl)  # second call: both skipped (idempotent resume)
        orig = D.disk_free_bytes
        D.disk_free_bytes = lambda *a: 10**9
        try:
            for mk in C.LOGS_DIR.glob("pyt_download.synthetic__neox-*.done"):
                C.move_marker_aside(mk)
            stopped = False
            try:
                D.download_all(dl2, redl)
            except D.StopRun as e:
                stopped = "disk guard" in str(e)
            assert stopped
        finally:
            D.disk_free_bytes = orig
    finally:
        R.DOWNLOADS[:] = saved
    return dict(e1a_params=e1a["n_params_without_buffers"], e1b_retried=e1b["retried"],
                dtypes=e1a["dtype_counts"], disk_stop=True)


# ------------------------------------------------------------------------------------------------------------ T6
@test
def t06_gate_dry_runs():
    env = {k: v for k, v in os.environ.items() if k not in ("PYT_S0B_TEST", "PYT_DEVICE")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    status_before = (PROJ / "STATUS.md").read_bytes()
    res_root = PROJ / "results" / "S1_2_layermax_pythia"
    before = sorted(str(p) for p in res_root.rglob("*")) if res_root.exists() else []
    cmds = {
        "S1_resolve": ["extract/pythia_axis/download_pyt.py", "resolve"],
        "S3_download": ["extract/pythia_axis/download_pyt.py", "download"],
        "E1_verify": ["extract/pythia_axis/download_pyt.py", "verify", "--model", "EleutherAI/pythia-70m"],
        "fcache": ["extract/pythia_axis/download_pyt.py", "fcache", "--phase", "before_S3"],
        "S2_e2": ["extract/pythia_axis/run_pyt.py", "e2"],
        "S4_primary_70m": ["extract/pythia_axis/run_pyt.py", "primary", "--size", "70m"],
        "S4_primary_6.9b": ["extract/pythia_axis/run_pyt.py", "primary", "--size", "6.9b"],
        "E5": ["extract/pythia_axis/run_pyt.py", "e5"],
        "summary": ["extract/pythia_axis/run_pyt.py", "summary"],
        "ledger": ["extract/pythia_axis/run_pyt.py", "ledger"],
        "diskwatch": ["extract/pythia_axis/run_pyt.py", "diskwatch", "--max-hours", "0"],
        "E1PROV": ["analysis/s12_layermax_pythia/e1prov_pyt.py"],
        "E1PROV_explicit": ["analysis/s12_layermax_pythia/e1prov_pyt.py", "--main", "x", "--step", "y", "--out", "z"],
        "G0_no_smoke_row": ["analysis/s12_layermax_pythia/gpu_gate_pyt.py", "g0", "--smoke-steps", "no_such_step",
                            "--smoke-nominal", "3"],
        "G1_S4": ["analysis/s12_layermax_pythia/gpu_gate_pyt.py", "g1", "--before", "S4", "--last-step", "x",
                  "--last-nominal", "0.5"],
        "memwait": ["analysis/s12_layermax_pythia/gpu_gate_pyt.py", "memwait", "--size", "70m"],
    }
    rcs = {}
    for k, c in cmds.items():
        p = subprocess.run([VENV_PY, "-I"] + c, cwd=str(PROJ), env=env, capture_output=True, text=True, timeout=300)
        rcs[k] = p.returncode
    after = sorted(str(p) for p in res_root.rglob("*")) if res_root.exists() else []
    assert all(v == 3 for v in rcs.values()), rcs
    assert (PROJ / "STATUS.md").read_bytes()[:len(status_before)] == status_before
    new_status = (PROJ / "STATUS.md").read_bytes()[len(status_before):].decode()
    assert "[pyt-axis-extract]" not in new_status, new_status
    assert [p for p in after if p not in before and not p.startswith(str(res_root / "logs"))] == [], after
    return dict(exit_codes=rcs, results_tree_new=[p for p in after if p not in before])


# ------------------------------------------------------------------------------------------------------------ T7
@test
def t07_e1prov():
    import e1prov_pyt as P
    m = snap("main", "1") / "pytorch_model.bin"
    out = {}
    cases = {"step": True, "buffer_diff": True, "perturbed": False, "extra_nonbuf": None, "step_only": False}
    for v, want in cases.items():
        if v == "extra_nonbuf":  # a non-buffer name only in main: compare it AS main against the clean step file
            r = P.compare_bins(snap("extra_nonbuf", "2") / "pytorch_model.bin", snap("step", "3") / "pytorch_model.bin")
            assert not r["pass"] and r["main_only_not_buffer"] == ["gpt_neox.layers.0.attention.extra_weight"], r
            out[v] = dict(pass_=r["pass"], main_only_not_buffer=r["main_only_not_buffer"])
            continue
        other = snap(v, {"step": "3", "buffer_diff": "4", "perturbed": "5", "step_only": "6"}[v]) / "pytorch_model.bin"
        r = P.compare_bins(m, other)
        assert r["pass"] is want, (v, r["pass"], r["nonbuffer_differences"][:2], r["step_only"], r["main_only"])
        out[v] = dict(pass_=r["pass"], main_only=len(r["main_only"]), step_only=r["step_only"],
                      nonbuf_diffs=[d["name"] for d in r["nonbuffer_differences"]],
                      n_buffer_diffs=r["n_buffer_differences"])
    assert out["step"]["main_only"] == 6 and out["buffer_diff"]["n_buffer_diffs"] == 3
    assert out["perturbed"]["nonbuf_diffs"] == ["gpt_neox.layers.1.mlp.dense_h_to_4h.weight"]
    # the CLI in test mode on synthetic files
    p = subprocess.run([VENV_PY, "-I", "analysis/s12_layermax_pythia/e1prov_pyt.py", "--main", str(m), "--step",
                        str(snap("perturbed", "5") / "pytorch_model.bin"), "--out",
                        str(RUNROOT / "results" / "e1prov_synthetic.json")],
                       cwd=str(PROJ), env=dict(os.environ), capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    j = C.read_json(RUNROOT / "results" / "e1prov_synthetic.json")
    assert j["source"] == "step143000" and not j["pass"]
    return out


# ------------------------------------------------------------------------------------------------------------ T8
@test
def t08_synthetic_extraction_cpu():
    import extract_core_pyt as ec
    s = snap("main", "1")
    out_root = RUNROOT / "features" / "synthetic_cpu"
    tok1 = out_root / "_tok1"
    sink = RUNROOT / "results" / "sink_synthetic_cpu.json"
    model_id = "synthetic/neox-tiny-cpu"
    checks = ec.run_lm(model_id, ["wit1024"], out_root, 16, print, snap=s, tok1_dir=tok1, sink_path=sink,
                       gptneox_checks=True)
    stem = out_root / R.slug(model_id)
    meta = json.load(open(str(stem) + ".json"))
    sem = meta["hidden_state_semantics"]
    assert sem["hs0_vs_layer0_input"] == 0.0 and sem["hs_i_vs_layer_i-1_output_max"] == 0.0 \
        and sem["hsL_vs_finalnorm(lastlayer)"] == 0.0, sem
    lc = checks["loading_check"]
    assert lc["pass"], lc
    assert checks["rotary_check"]["ok"], checks["rotary_check"]
    rr = meta["rerun_check"]
    assert rr["max_abs_diff"] == 0.0 and all(rr["primary_bitwise"]) and all(rr["tok1_bitwise"]), rr
    A = np.load(str(stem) + ".npy")
    T1 = np.load(tok1 / f"{R.slug(model_id)}.npy")
    assert A.shape == (16, 4, 64) and T1.shape == (16, 4, 64)
    sj = json.load(open(sink))
    assert len(sj["per_layer"]) == 4 and sj["n_items"] == 16
    # independent reference for item 0 and item 5: plain transformers forward, means in float64
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(s))
    model = AutoModelForCausalLM.from_pretrained(str(s), dtype=torch.float32, local_files_only=True).eval()
    texts = TEXTS[:16]
    for j in (0, 5):
        ids = tok(texts[j], return_tensors="pt")
        with torch.inference_mode():
            hs = model(**ids, output_hidden_states=True, use_cache=False).hidden_states
        full = np.stack([h[0].double().mean(0).float().numpy() for h in hs])
        t1 = np.stack([(h[0, 1:] if h.shape[1] > 1 else h[0]).double().mean(0).float().numpy() for h in hs])
        assert np.array_equal(full, A[j]), ("primary", j, np.abs(full - A[j]).max())
        assert np.array_equal(t1, T1[j]), ("tok1", j, np.abs(t1 - T1[j]).max())
        r0 = np.array([np.linalg.norm(h[0, 0].double().numpy()) / np.linalg.norm(h[0].double().sum(0).numpy()) for h in hs])
        ratio = np.load(sink.with_suffix(".npz"))["ratio"][j]
        assert np.allclose(r0, ratio, rtol=1e-12, atol=0), (r0, ratio)
    # loading-check failures stop before any forward pass
    fails = {}
    for v, ch in (("missing_param", "7"), ("unexpected_param", "8")):
        try:
            ec.run_lm(f"synthetic/neox-{v}", ["wit1024"], RUNROOT / "features" / f"synthetic_{v}", 4, print,
                      snap=snap(v, ch), tok1_dir=RUNROOT / "features" / f"synthetic_{v}" / "_tok1",
                      sink_path=RUNROOT / "results" / f"sink_{v}.json", gptneox_checks=True)
            fails[v] = "NOT RAISED"
        except ec.LoadingCheckError as e:
            fails[v] = json.loads(str(e)) if str(e).startswith("{") else str(e)[:200]
        assert not (RUNROOT / "features" / f"synthetic_{v}" / f"synthetic__neox-{v}.done").exists()
    assert fails["missing_param"] != "NOT RAISED" and fails["unexpected_param"] != "NOT RAISED", fails
    return dict(shape=list(A.shape), semantics=sem, loading=dict(missing=lc["missing_keys"],
                n_unexpected=len(lc["unexpected_keys"]), unexpected_sample=lc["unexpected_keys"][:3],
                head_shared=lc["lm_head_shares_embed_in_storage"], n_values=lc["n_values_sampled"]),
                rotary=checks["rotary_check"], tok1_one_token=sj["n_one_token_captions"],
                ratio_median_layer0=sj["per_layer"][0]["ratio_median"],
                fail_missing=(fails["missing_param"].get("bad_missing") if isinstance(fails["missing_param"], dict) else fails["missing_param"]),
                fail_unexpected=(fails["unexpected_param"].get("bad_unexpected") if isinstance(fails["unexpected_param"], dict) else fails["unexpected_param"]))


# ------------------------------------------------------------------------------------------------------------ T9
@test
def t09_e5():
    import run_pyt as RUN
    import make_synthetic as MS
    a, b = snap("main", "1"), snap("step", "3")
    r = RUN.e5_compare({"synthetic/a": a, "synthetic/b": b}, TEXTS)
    assert r["pass"], r
    other = SNAPROOT / "tok900" / ("9" * 40)
    if not (other / "pytorch_model.bin").exists():
        MS.make_snapshot(SNAPROOT / "tok900", "9" * 40, TEXTS, variant="step", tokenizer_json=MS.build_tokenizer(TEXTS, 900))
    r2 = RUN.e5_compare({"synthetic/a": a, "synthetic/c": other}, TEXTS)
    assert not r2["pass"] and r2["ids_vs_first_model"]["synthetic/c"] > 0
    info = r["tokenizers"]["synthetic/a"]
    return dict(cls=info["cls"], n_tok=info["n_tokens"], bos_first=info["n_first_is_bos"],
                eos_last=info["n_last_is_eos"], neg_case_diff=r2["ids_vs_first_model"])


# ------------------------------------------------------------------------------------------------------------ T10
@test
def t10_status_gates():
    orig = C._real_status_lines
    cases = {
        "empty": ([], None),
        "go": (["2026-10-10T01:00:00Z [pyt-axis-lead] S3 GO: G0 GO and the author's confirmation in chat"], 0),
        "go_without_G0": (["2026-10-10T01:00:00Z [pyt-axis-lead] S3 GO: the author's confirmation"], None),
        "go_without_author": (["2026-10-10T01:00:00Z [pyt-axis-lead] S3 GO: G0 GO"], None),
        "go_then_deferred": (["2026-10-10T01:00:00Z [pyt-axis-lead] S3 GO: G0 GO, author ok",
                              "2026-10-10T01:05:00Z [pyt-axis-extract] DEFERRED (GPU contention): x"], None),
        "deferred_then_go": (["2026-10-10T01:00:00Z [pyt-axis-extract] DEFERRED (GPU contention): x",
                              "2026-10-10T02:00:00Z [pyt-axis-lead] S3 GO: G0 GO, author ok"], 1),
        "other_axis": (["2026-10-10T01:00:00Z [oll-axis-lead] S3 GO: G0 GO, author ok"], None),
        "quoted": (["2026-10-10T01:00:00Z [pyt-axis-freeze] plan says [pyt-axis-x] S3 GO with G0 and author"], None),
    }
    out = {}
    try:
        for k, (lines, want) in cases.items():
            C._real_status_lines = lambda lines=lines: lines
            got = C.s3_go_line()
            exp = None if want is None else lines[want]
            assert got == exp, (k, got)
            out[k] = got is not None
        C._real_status_lines = lambda: ["2026-10-10T01:00:00Z [pyt-axis-s0b] CODE FROZEN (S0b) ..."]
        assert C.code_frozen_line() is not None
        C._real_status_lines = lambda: ["2026-10-10T01:00:00Z [pyt-axis-freeze] Part A FROZEN; next: CODE FROZEN"]
        assert C.code_frozen_line() is None
    finally:
        C._real_status_lines = orig
    assert C.code_frozen_line() is None  # the real STATUS.md has no Pythia CODE FROZEN line at test time
    return out


# ------------------------------------------------------------------------------------------------------------ T11
@test
def t11_gpu_gates():
    import gpu_gate_pyt as G
    assert abs(G.REMAINING_AFTER_SMOKE - 15.4) < 1e-9
    assert {k: round(v, 6) for k, v in G.REMAINING_BEFORE.items()} == {"S4": 14.9, "S6": 10.9, "S7": 7.9}
    r = [G.g0_decision(3.0, 3.0, 3.0), G.g0_decision(9.0, 9.0, 3.0), G.g0_decision(50 - 15.4, 3.0, 3.0),
         G.g0_decision(50 - 15.4 + 1e-9, 3.0, 3.0)]
    assert [x["decision"] for x in r] == ["GO", "DEFERRED", "GO", "DEFERRED"], r
    s = [G.g1_decision("S4", 5.0, 0.5, 0.5), G.g1_decision("S4", 5.0, 0.25, 0.5), G.g1_decision("S7", 40.0, 3.0, 0.5),
         G.g1_decision("S6", 60 - 10.9, 0.5, 0.5)]
    assert [x["decision"] for x in s] == ["GO", "GO", "DEFERRED", "GO"] and s[1]["f"] == 1.0, s
    snap22 = G.s22_snapshot()
    return dict(g0=[(x["f0"], round(x["projection_min"], 3), x["decision"]) for x in r],
                g1=[(x["f"], round(x["projection_min"], 3), x["decision"]) for x in s],
                s22_snapshot=snap22)


# ------------------------------------------------------------------------------------------------------------ T12
@test
def t12_launcher():
    L = str(PROJ / "analysis" / "s12_layermax_pythia" / "launch_pyt.sh")
    logdir = C.TEST_ROOT / "launch_logs"
    env = dict(os.environ, PYT_LOGDIR=str(logdir))
    sfx = "_" + C.utcnow().replace("-", "").replace(":", "")  # markers are never removed: unique names per run
    rc = {}
    rc["threads7"] = subprocess.run(["bash", L, "lt_t7" + sfx, "7", "--", "true"], env=env).returncode
    rc["ok"] = subprocess.run(["bash", L, "lt_ok" + sfx, "1", "--", "true"], env=env).returncode
    rc["marker_exists"] = subprocess.run(["bash", L, "lt_ok" + sfx, "1", "--", "true"], env=env).returncode
    rc["gpu_over_budget"] = subprocess.run(["bash", L, "lt_gpu" + sfx, "1", "--gpu", "61", "--", "true"], env=env).returncode
    # this runner holds PYT_THREADS tokens; hold all but one of the rest, then ask for 2 -> 75
    free = 6 - int(os.environ.get("PYT_THREADS", "4"))
    assert free >= 2, "run the CPU tests with at most 4 thread tokens"
    hold = subprocess.Popen(["bash", L, "lt_hold" + sfx, str(free - 1), "--", "sleep", "6"], env=env)
    time.sleep(2)
    rc["tokens_busy"] = subprocess.run(["bash", L, "lt_busy" + sfx, "2", "--", "true"], env=env).returncode
    hold.wait()
    rc["threads_env"] = subprocess.run(["bash", L, "lt_env" + sfx, "2", "--", "bash", "-c",
                                        "test \"$PYT_THREADS\" = 2 -a \"$OMP_NUM_THREADS\" = 2"], env=env).returncode
    assert rc == {"threads7": 64, "ok": 0, "marker_exists": 65, "gpu_over_budget": 76, "tokens_busy": 75,
                  "threads_env": 0}, rc
    assert (logdir / f"lt_ok{sfx}.done").exists() and not (C.LEDGER.exists() and "lt_gpu" + sfx in C.LEDGER.read_text())
    return rc


# ------------------------------------------------------------------------------------------------------------ T13
@test
def t13_fcache_listing():
    import download_pyt as D
    a = D.fcache_listing()
    b = D.fcache_listing()
    assert a["top_level"] == b["top_level"] and a["polypythia_top_entries"] == b["polypythia_top_entries"]
    assert sorted(a["polypythia_top_entries"]) == sorted(R.POLYPYTHIA_DIRS)
    assert not any(n.startswith("models--EleutherAI--pythia-") and "seed" not in n for n in a["top_level"])
    return dict(n_top=len(a["top_level"]),
                seed_entries={k: [e[0] for e in (v or [])] for k, v in a["polypythia_top_entries"].items()})


REPORT["run_id"] = RUNID
REPORT["code_sha256"] = {str(p.relative_to(PROJ)): C.sha256_file(p) for p in sorted(list(AXIS.glob("*.py")) + list(HERE.glob("*.py"))
                         + list((PROJ / "analysis" / "s12_layermax_pythia").glob("*_pyt.py"))
                         + [PROJ / "analysis" / "s12_layermax_pythia" / "launch_pyt.sh"])}
REPORT["finished_utc"] = C.utcnow()
REPORT["wall_s"] = round(time.time() - T0, 1)
REPORT["n_pass"] = sum(1 for v in REPORT["tests"].values() if v["pass_"])
REPORT["n_fail"] = sum(1 for v in REPORT["tests"].values() if not v["pass_"])
h = C.write_json(RUNROOT / "test_report_cpu.json", REPORT)
(C.TEST_ROOT / "latest_cpu_run.txt").write_text(RUNID + "\n")
print(f"CPU TESTS: {REPORT['n_pass']} pass, {REPORT['n_fail']} fail, {REPORT['wall_s']} s; report sha256 {h}")
sys.exit(0 if REPORT["n_fail"] == 0 else 1)
