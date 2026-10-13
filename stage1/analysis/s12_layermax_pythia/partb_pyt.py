#!/usr/bin/env python
"""S1.2 Pythia axis (fourth size axis), Part B, steps B0-B7 (run by role pyt-axis-predict at step S5).  # PYT-AXIS

Copy of analysis/s12_layermax_openllama/partb_ol.py (sha256 4aab8ee1942cb0e94307306b0ccd9d450102d6bcd41236370f614b3192b8aef6),  # PYT-AXIS
adapted at S0b (role pyt-code-infra, 2026-10-10) to analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md section 9  # PYT-AXIS
(sha256 6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b) and to the contract VERDICTS_SPEC_PYT.md  # PYT-AXIS
section 11. Every changed or added line carries the marker PYT-AXIS; diff: partb_pyt.py.diff.

FEATURES ONLY, plus KNOWN frozen scores of other families. It opens the 7 Pythia stacks, the 7 _tok1 stacks (D7  # PYT-AXIS
only), the 17 ViT stacks, the stacks of the three recheck models (bloomz-560m, open_llama_13b, llama-30b), the frozen  # PYT-AXIS
results/S1_2_layermax_llama/partB/partB_features.json and results/S1_2_layermax_openllama/partB/partB_features_ol.json,  # PYT-AXIS
and the known pair files of the 11 known LMs for the B5 offset retrodiction (results/S1_2_layermax/pairs/,  # PYT-AXIS
results/S1_2_layermax_llama/early/pairs/, results/S1_2_layermax_openllama/early/pairs/). No cross-model score  # PYT-AXIS
involving a Pythia model exists or is computed: every Pythia number is a function of Pythia features alone, or a  # PYT-AXIS
product of two single-model numbers (Proposition 5 baselines). It writes only results/S1_2_layermax_pythia/partB/  # PYT-AXIS
and analysis/s12_layermax_pythia/PARTB_PREDICTIONS_PYT.md (plan B0).  # PYT-AXIS

B0  guard.scan() (plan F1(b)) at start and before writing; refuses (exit 4) unless clean (not in S0b test mode);  # PYT-AXIS
    full scan output logged to partB/guard_scan_{start,end}.json.  # PYT-AXIS
B1  float64 HKH PRs of every stored layer of the 7 Pythia models and the 17 ViTs, both pre; 7 clip thresholds  # PYT-AXIS
    (hrhv2.prep.prh_clip_threshold on the float64 all-layer stack) -> partB/clip_thresholds_partB_pyt.json (V-P1  # PYT-AXIS
    must reproduce them bitwise); recomputation of bloomz-560m, open_llama_13b, llama-30b and the 17 ViTs against  # PYT-AXIS
    the frozen Part B files (PR <= 1e-14 relative, clip bitwise; exit 5 otherwise); PR* of the _tok1 stacks (D7).  # PYT-AXIS
B2  Bbar (+argmax, u*/(L-1)), B_fd (r = 0.5, 1.0 at floor(r(L-1)+0.5)), B_late (LM states with relative depth  # PYT-AXIS
    >= 0.60, all ViT layers) for the 119 pairs, both pre.  # PYT-AXIS
B3  Tr_P(Bbar), Tr_P(B_fd(r)), Tr_P(B_late); per-ViT gaps for all 21 size pairs; the 5-size Tr_P(Bbar).  # PYT-AXIS
B4  materiality: |Tr_P(B)| >= 0.75 and median over ViTs of s_B [B(6.9b, v) - B(70m, v)] >= 3 delta_b = 0.015;  # PYT-AXIS
    the delta_b curve.  # PYT-AXIS
B5  extras (outcomes known, not tests): offset retrodictions on the 187 known pairs under the PR*-indexed band  # PYT-AXIS
    (and the 11-LM fit recomputed), Spearman of PR* against log params, width and depth, PR at fixed depths,  # PYT-AXIS
    runner-up ratios, the B_late = Bbar count, the extrapolation flags.  # PYT-AXIS
B6  s, s_late, H1 membership, H2 branch, ordered member list; band_i = f(PR*_i) +- h with the frozen constants;  # PYT-AXIS
    h curve and flat band; the uncensored band; the P-P2a(ii) and P-P5a counts and needs.  # PYT-AXIS
B7  partB/partB_features_pyt.json (every field of VERDICTS_SPEC_PYT section 11), the DRAFT  # PYT-AXIS
    analysis/s12_layermax_pythia/PARTB_PREDICTIONS_PYT.md (numbers filled in from the JSON) and  # PYT-AXIS
    partB/STATUS_LINE_DRAFT.txt (the freeze line with every hash and value of plan B7 filled in).  # PYT-AXIS

RUN (step S5: after S4 extraction and its checks; nothing else of this axis running; CPU only):  # PYT-AXIS
  cd ~/hrh_v2_stage1_20261008  # PYT-AXIS
  analysis/s12_layermax_pythia/launch_pyt.sh partb_pyt 4 -- venv/bin/python analysis/s12_layermax_pythia/partb_pyt.py  # PYT-AXIS
  then (sanity/retrodiction on known pairs only):  # PYT-AXIS
  analysis/s12_layermax_pythia/launch_pyt.sh partb_checks_pyt 2 -- venv/bin/python analysis/s12_layermax_pythia/partb_checks_pyt.py  # PYT-AXIS
  Exit codes: 0 ok; 3 order gate (no CODE FROZEN line, or S4 incomplete; nothing written); 4 guard scan not clean;  # PYT-AXIS
  5 a recomputation or input check failed (plan hash, frozen input hash,  # PYT-AXIS
  recheck models, extraction incomplete, layer rule); 6 an output already exists (no-delete rule: Part B is never  # PYT-AXIS
  overwritten; a rerun needs a logged deviation and a new output name). Nothing is written to partB/ except the  # PYT-AXIS
  scan logs and the check report when it exits 4 or 5.  # PYT-AXIS

FREEZE (role pyt-axis-predict, plan B7): review the draft PARTB_PREDICTIONS_PYT.md (edit wording only; numbers  # PYT-AXIS
come from the JSON), run `venv/bin/python analysis/s12_layermax_pythia/guard.py scan` (must exit 0; log it),  # PYT-AXIS
chmod 444 both files, then append ONE STATUS line of exactly the form in partB/STATUS_LINE_DRAFT.txt (guard.py parses  # PYT-AXIS
"PARTB_PREDICTIONS_PYT.md sha256 <64 hex>" and "partB_features_pyt.json sha256 <64 hex>"; the sha256 of the markdown  # PYT-AXIS
must be recomputed after the wording review), with <UTC> = date -u +%Y-%m-%dT%H:%M:%S at the time of logging.  # PYT-AXIS

S0b TEST MODE (PYT_S0B_TEST=1, only before the CODE FROZEN line; guard.test_mode()): outputs go to  # PYT-AXIS
analysis/s12_layermax_pythia/s0b_scratch/PB/ (partB_<standin>/ and PARTB_PREDICTIONS_PYT.<standin>.md), a failing scan  # PYT-AXIS
is logged but not refused, and --standin selects the stand-in for the 7 Pythia models (no file named  # PYT-AXIS
EleutherAI__pythia-* is created):  # PYT-AXIS
  vp9   the V-P9 fixture of VERDICTS_SPEC_PYT section 13: bloomz-560m, -1b1, -1b7, -3b, -7b1, llama-13b, llama-30b stand in  # PYT-AXIS
        for 70m .. 6.9b under slugs synth__pyt-<size>; the _tok1 stand-in is the primary stack (labelled)   [default]  # PYT-AXIS
  PYT_S0B_TEST=1 analysis/s12_layermax_pythia/launch_pyt.sh pb_test 2 -- venv/bin/python analysis/s12_layermax_pythia/partb_pyt.py --standin vp9  # PYT-AXIS
"""
from __future__ import annotations

import os

_TH = os.environ.get("PYT_THREADS", "1")  # PYT-AXIS: thread count from launch_pyt.sh (plan F5)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _TH  # PYT-AXIS

import argparse  # noqa: E402  # PYT-AXIS
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402  # PYT-AXIS
import sys  # noqa: E402
import time  # noqa: E402
from fractions import Fraction  # noqa: E402  # PYT-AXIS

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_pythia")  # PYT-AXIS
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "analysis", "s12"))
sys.path.insert(0, AXIS)  # PYT-AXIS

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import common as C  # noqa: E402  (S1.2 registry + preprocess_rows; read-only import)
import guard  # noqa: E402  # PYT-AXIS: F1(b) scan, S0b test mode
import verdict_keys as VK  # noqa: E402  # PYT-AXIS: sizes, ViTs, families, curves (names only)
from hrhv2.prep import prh_clip_threshold, center_gram  # noqa: E402

FEAT = os.path.join(PROJ, "features")
RES_P = os.path.join(PROJ, "results", "S1_2_layermax_pythia")  # PYT-AXIS
OUTDIR_REAL = os.path.join(RES_P, "partB")  # PYT-AXIS
MD_REAL = os.path.join(AXIS, "PARTB_PREDICTIONS_PYT.md")  # PYT-AXIS
SCRATCH_PB = os.path.join(guard.SCRATCH, "PB")  # PYT-AXIS: S0b test outputs only
PLAN = os.path.join(AXIS, "ANALYSIS_PLAN_PYTHIA.md")  # PYT-AXIS
PLAN_SHA = "6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b"  # PYT-AXIS
LLAMA_PARTB = os.path.join(PROJ, "results", "S1_2_layermax_llama", "partB", "partB_features.json")  # PYT-AXIS
LLAMA_PARTB_SHA = "ede0127589a02037427b343e9c3593ee427d430e9746d7f4942b15f76a3031bf"  # plan section 9 B0  # PYT-AXIS
OL_PARTB = os.path.join(PROJ, "results", "S1_2_layermax_openllama", "partB", "partB_features_ol.json")  # PYT-AXIS
OL_PARTB_SHA = "e1104bd71e5eff9ced65ed402a61c187590bce5dee306c25ee2f64b18fdd7e51"  # PYT-AXIS: plan section 9 B0
FROZEN8 = os.path.join(PROJ, "results", "S1_2_layermax", "pairs")  # frozen 8-LM E-format pair files  # PYT-AXIS
LLAMA_E = os.path.join(PROJ, "results", "S1_2_layermax_llama", "early", "pairs")  # LLaMA-axis E pairs  # PYT-AXIS
OL_E = os.path.join(PROJ, "results", "S1_2_layermax_openllama", "early", "pairs")  # PYT-AXIS: OpenLLaMA-axis E pairs

M = 1024
KNN_K = 10
DELTA_B = 0.005
DEPTHS = (0.5, 1.0)
PRES = ("prh_clip+l2", "l2")
LATE_REL = 0.60  # B_late, LM states with relative depth >= 0.60 (plan B2)  # PYT-AXIS
TOL = 1e-9  # plan F2 tolerance  # PYT-AXIS
MAT_TR = 0.75  # PYT-AXIS: plan B4 |Tr_P(B)| >= 0.75
MAT_GAP_MULT = 3  # PYT-AXIS: plan B4 median end-to-end gap >= 3 delta_b
BAND = {  # PYT-AXIS: plan section 9 B6 frozen constants (section 0 fit) and the flat-band sensitivity
    "prh_clip+l2": {"a": 0.001069, "b": 0.012920, "h": 0.0008, "fit_min_PR": 124.5, "fit_max_PR": 359.2,  # PYT-AXIS
                    "flat": [0.0010, 0.0040]},  # PYT-AXIS
    "l2": {"a": 0.001551, "b": 0.006843, "h": 0.0018, "fit_min_PR": 79.6, "fit_max_PR": 180.0,  # PYT-AXIS
           "flat": [0.0010, 0.0045]},  # PYT-AXIS
}  # PYT-AXIS
MU_BAND = (-0.0003, 0.0008)  # PYT-AXIS: plan B6 uncensored band, every model
HS = VK.HS  # PYT-AXIS: plan 5.6 h curve
DBS = VK.DBS  # PYT-AXIS: plan 5.6 delta_b curve
SIZES = list(VK.SIZES)  # PYT-AXIS
FIXED_LAYERS = {0.5: [3, 6, 12, 8, 12, 16, 16], 1.0: [6, 12, 24, 16, 24, 32, 32]}  # PYT-AXIS: plan section 3 table
L_STORED = [7, 13, 25, 17, 25, 33, 33]  # PYT-AXIS: plan section 1
FIVE = ["410m", "1b", "1.4b", "2.8b", "6.9b"]  # PYT-AXIS: plan B3 5-size trend
RECHECK = {  # PYT-AXIS: plan B1 recomputation of three known models (feature dir, frozen Part B)
    "bigscience__bloomz-560m": ("wit1024", "llama"),  # PYT-AXIS
    "openlm-research__open_llama_13b": ("wit1024_openllama_axis", "ol"),  # PYT-AXIS
    "huggyllama__llama-30b": ("wit1024_llama_axis", "llama"),  # PYT-AXIS
}  # PYT-AXIS
KNOWN11 = [  # PYT-AXIS: the 11 known LMs of the band fit (plan section 0) with their frozen E-format pair dirs
    ("bigscience__bloomz-560m", FROZEN8), ("bigscience__bloomz-1b1", FROZEN8), ("bigscience__bloomz-1b7", FROZEN8),  # PYT-AXIS
    ("bigscience__bloomz-3b", FROZEN8), ("bigscience__bloomz-7b1", FROZEN8),  # PYT-AXIS
    ("openlm-research__open_llama_3b", FROZEN8), ("openlm-research__open_llama_7b", FROZEN8),  # PYT-AXIS
    ("openlm-research__open_llama_13b", OL_E), ("huggyllama__llama-7b", FROZEN8),  # PYT-AXIS
    ("huggyllama__llama-13b", LLAMA_E), ("huggyllama__llama-30b", LLAMA_E),  # PYT-AXIS
]
VITS = [(s, f, "wit1024") for s, f in C.VITS]
VIT_STEMS = [s for s, _, _ in VITS]  # PYT-AXIS
STANDIN_VP9 = {  # PYT-AXIS: VERDICTS_SPEC_PYT section 13 (V-P9 7-size stand-in)
    "70m": ("bigscience__bloomz-560m", "wit1024"), "160m": ("bigscience__bloomz-1b1", "wit1024"),  # PYT-AXIS
    "410m": ("bigscience__bloomz-1b7", "wit1024"), "1b": ("bigscience__bloomz-3b", "wit1024"),  # PYT-AXIS
    "1.4b": ("bigscience__bloomz-7b1", "wit1024"), "2.8b": ("huggyllama__llama-13b", "wit1024_llama_axis"),  # PYT-AXIS
    "6.9b": ("huggyllama__llama-30b", "wit1024_llama_axis"),  # PYT-AXIS
}  # PYT-AXIS
CFG: dict = {}  # filled by configure()  # PYT-AXIS


def configure(test: bool, standin: str | None) -> dict:  # PYT-AXIS
    """Model sources, slugs, output paths. Real mode: the 7 Pythia models from features/wit1024_pythia_axis/."""  # PYT-AXIS
    if not test:  # PYT-AXIS
        slugs = {s: VK.PYT_SLUGS[s] for s in SIZES}  # PYT-AXIS
        src = {slugs[s]: (slugs[s], "wit1024_pythia_axis") for s in SIZES}  # PYT-AXIS: slug -> (stem, dir)
        tok1 = {slugs[s]: (slugs[s], os.path.join("wit1024_pythia_axis", "_tok1")) for s in SIZES}  # PYT-AXIS
        outdir, md = OUTDIR_REAL, MD_REAL  # PYT-AXIS
        tok1_label = "token-0-excluded stacks (features/wit1024_pythia_axis/_tok1/)"  # PYT-AXIS
    elif standin == "vp9":  # PYT-AXIS
        slugs = {s: f"synth__pyt-{s}" for s in SIZES}  # PYT-AXIS
        src = {slugs[s]: STANDIN_VP9[s] for s in SIZES}  # PYT-AXIS
        tok1 = dict(src)  # PYT-AXIS: stand-in: the primary stack plays the _tok1 stack (labelled)
        outdir = os.path.join(SCRATCH_PB, "partB_vp9")  # PYT-AXIS
        md = os.path.join(SCRATCH_PB, "PARTB_PREDICTIONS_PYT.vp9.md")  # PYT-AXIS
        tok1_label = "STAND-IN: the primary stack of each stand-in model (no _tok1 stack exists)"  # PYT-AXIS
    else:  # PYT-AXIS
        raise SystemExit(f"unknown --standin {standin}")  # PYT-AXIS
    allsrc = dict(src)  # PYT-AXIS
    allsrc.update({s: (s, ds) for s, _, ds in VITS})  # PYT-AXIS
    allsrc.update({s: (s, d) for s, (d, _) in RECHECK.items()})  # PYT-AXIS
    CFG.clear()  # PYT-AXIS
    CFG.update({"test": test, "standin": standin if test else None, "slugs": slugs, "pyt": [slugs[s] for s in SIZES],  # PYT-AXIS
                "src": allsrc, "tok1": tok1, "tok1_label": tok1_label, "outdir": outdir, "md": md})  # PYT-AXIS
    return CFG  # PYT-AXIS


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sidecar_of(stem: str, ds: str) -> dict:  # PYT-AXIS
    with open(os.path.join(FEAT, ds, stem + ".json")) as f:  # PYT-AXIS
        return json.load(f)


def sidecar(key: str) -> dict:  # PYT-AXIS
    s, ds = CFG["src"][key]  # PYT-AXIS
    return sidecar_of(s, ds)  # PYT-AXIS


def npy_path(key: str, tok1: bool = False) -> str:  # PYT-AXIS
    s, ds = (CFG["tok1"] if tok1 else CFG["src"])[key]  # PYT-AXIS
    return os.path.join(FEAT, ds, s + ".npy")  # PYT-AXIS


def stack(key: str, tok1: bool = False) -> np.ndarray:  # PYT-AXIS
    st = np.load(npy_path(key, tok1), mmap_mode="r")  # PYT-AXIS
    assert st.shape[0] == M and st.ndim == 3, (key, st.shape)  # PYT-AXIS
    assert list(st.shape) == list(sidecar(key)["shape"]), key  # PYT-AXIS: _tok1 has the primary's shape (plan section 2)
    return st


def fixed_layer(L: int, r: float) -> int:
    return int(math.floor(r * (L - 1) + 0.5))


def layer_pr(X: np.ndarray, pre: str, c: float | None) -> tuple[float, float, float]:
    Xp = C.preprocess_rows(X, pre, c)
    K = Xp @ Xp.T
    Kc = center_gram(K)
    tr = float(np.trace(Kc))
    fr = float(np.vdot(Kc.ravel(), Kc.ravel()))
    return tr * tr / fr, tr, fr


def spearman(x, y) -> float:  # PYT-AXIS: VERDICTS_SPEC_PYT section 2 (average ranks, Pearson formula)
    rx, ry = rankdata(x), rankdata(y)  # PYT-AXIS
    a, b = rx - rx.mean(), ry - ry.mean()  # PYT-AXIS
    den = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))  # PYT-AXIS
    return float(np.sum(a * b) / den) if den > 0 else float("nan")  # PYT-AXIS


def spearman_exact_sumd2(values) -> int | None:  # PYT-AXIS: sum d^2 against ranks 1..n when the values have no ties
    r = rankdata(values)  # PYT-AXIS
    if len(set(r.tolist())) != len(r):  # PYT-AXIS
        return None  # PYT-AXIS
    return int(sum((int(ri) - (i + 1)) ** 2 for i, ri in enumerate(r)))  # PYT-AXIS


def jsonable(v):
    return C.jsonable(v)


def sgn(x: float) -> int:  # PYT-AXIS
    return 0 if x == 0 else (1 if x > 0 else -1)  # PYT-AXIS


def ceil_frac(num: int, den: int) -> int:  # PYT-AXIS: exact ceiling (VERDICTS_SPEC_PYT section 12)
    return -((-num) // den)  # PYT-AXIS


def keep_aside(path: str) -> str | None:  # PYT-AXIS: no-delete rule (test mode only: an earlier test output is renamed)
    if not os.path.exists(path):  # PYT-AXIS
        return None  # PYT-AXIS
    dst = path + ".prev-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"  # PYT-AXIS
    os.replace(path, dst)  # PYT-AXIS
    return dst  # PYT-AXIS


def read_pair_json(base: str, pre: str, lm: str, vit: str) -> tuple[dict, str]:  # PYT-AXIS
    p = os.path.join(base, pre, f"{lm}__{vit}.json")  # PYT-AXIS
    with open(p) as f:  # PYT-AXIS
        js = json.load(f)  # PYT-AXIS
    assert js["lm"] == lm and js["vit"] == vit and js["pre"] == pre, p  # PYT-AXIS
    return js, p  # PYT-AXIS


def run_scan(tag: str) -> dict:  # PYT-AXIS
    """Plan F1(b): negative scan, full output logged; refuses unless clean (except S0b test mode)."""  # PYT-AXIS
    try:  # fail closed if the scan itself cannot run  # PYT-AXIS
        rep = guard.scan()  # PYT-AXIS
    except Exception as e:  # noqa: BLE001  # PYT-AXIS
        rep = {"utc": utc(), "clean": False, "find_hits": [], "results_top_level": [],  # PYT-AXIS
               "results_top_level_not_allowed": [], "forbidden_processes": [],  # PYT-AXIS
               "scan_error": f"{type(e).__name__}: {e}"}  # PYT-AXIS
        print(f"[{utc()}] guard.scan() FAILED: {rep['scan_error']} -> treated as NOT clean", flush=True)  # PYT-AXIS
    os.makedirs(CFG["outdir"], exist_ok=True)  # PYT-AXIS
    log = os.path.join(CFG["outdir"], f"guard_scan_{tag}.json")  # PYT-AXIS
    if CFG["test"]:  # PYT-AXIS
        keep_aside(log)  # PYT-AXIS
    elif os.path.exists(log):  # PYT-AXIS: no-delete rule
        raise SystemExit(f"ABORT (6): {log} exists (Part B is never overwritten)")  # PYT-AXIS
    with open(log, "w") as f:  # PYT-AXIS
        json.dump(rep, f, indent=1)  # PYT-AXIS
    print(f"[{utc()}] guard scan ({tag}): clean={rep['clean']} find_hits={len(rep['find_hits'])} "  # PYT-AXIS
          f"top={rep['results_top_level']} not_allowed={rep['results_top_level_not_allowed']} "  # PYT-AXIS
          f"procs={rep['forbidden_processes']}", flush=True)  # PYT-AXIS
    if not rep["clean"] and not CFG["test"]:  # PYT-AXIS
        raise SystemExit(4)  # PYT-AXIS
    return {"utc": rep["utc"], "clean": rep["clean"], "n_find_hits": len(rep["find_hits"]),  # PYT-AXIS
            "results_top_level": rep["results_top_level"], "forbidden_processes": rep["forbidden_processes"],  # PYT-AXIS
            "scan_error": rep.get("scan_error"), "log": log}  # PYT-AXIS


def materiality(G: np.ndarray, sizes_rank: list[int], tr: float, db: float = DELTA_B) -> dict:  # PYT-AXIS: plan B4
    s_B = sgn(tr) if sgn(tr) != 0 else 1  # PYT-AXIS
    end = s_B * (G[-1] - G[0])  # PYT-AXIS: 6.9b minus 70m per ViT (rows in size order)
    med = float(np.median(end))  # PYT-AXIS
    return {"abs_tr": abs(tr), "s_B": s_B, "median_end_gap": med, "threshold_gap": MAT_GAP_MULT * db,  # PYT-AXIS
            "material": bool(abs(tr) >= MAT_TR - TOL and med >= MAT_GAP_MULT * db)}  # PYT-AXIS


def trend_rows(G: np.ndarray) -> dict:  # PYT-AXIS: Spearman per ViT of the 7 rows (size order) and their mean
    rho = [spearman(np.arange(1, G.shape[0] + 1), G[:, j]) for j in range(G.shape[1])]  # PYT-AXIS
    ok = [r for r in rho if np.isfinite(r)]  # PYT-AXIS
    return {"Tr": float(np.mean(ok)) if ok else float("nan"), "rho_per_vit": rho,  # PYT-AXIS
            "rho_common_to_all_vits": bool(len(set(np.round(ok, 12))) <= 1),  # PYT-AXIS
            "n_vit_constant": int(len(rho) - len(ok))}  # PYT-AXIS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # PYT-AXIS
    ap.add_argument("--standin", choices=("vp9",), default="vp9",  # PYT-AXIS
                    help="S0b test mode only: stand-in for the 7 Pythia models (ignored otherwise)")  # PYT-AXIS
    args = ap.parse_args()  # PYT-AXIS
    test = guard.test_mode()  # PYT-AXIS
    configure(test, args.standin)  # PYT-AXIS
    t0 = time.time()
    started = utc()
    OUTDIR = CFG["outdir"]  # PYT-AXIS
    if not CFG["test"]:  # PYT-AXIS: order gate (plan section 11: S5 after CODE FROZEN and S4); exit 3, nothing written
        if guard.code_frozen_line() is None:  # PYT-AXIS
            print("[partb_pyt] REFUSED (exit 3): no '[pyt-axis-<role>] CODE FROZEN' line in STATUS.md", file=sys.stderr)  # PYT-AXIS
            raise SystemExit(3)  # PYT-AXIS
        miss = [p for slug in CFG["pyt"] for p in (os.path.join(FEAT, "wit1024_pythia_axis", slug + ".done"),  # PYT-AXIS
                                                   npy_path(slug, tok1=True)) if not os.path.isfile(p)]  # PYT-AXIS
        if miss:  # PYT-AXIS
            print(f"[partb_pyt] REFUSED (exit 3): S4 not complete, {len(miss)} feature files missing, e.g. {miss[0]}",  # PYT-AXIS
                  file=sys.stderr)  # PYT-AXIS
            raise SystemExit(3)  # PYT-AXIS
    if not CFG["test"]:  # PYT-AXIS: no-delete rule, Part B is written once
        for p in (os.path.join(OUTDIR, "partB_features_pyt.json"), os.path.join(OUTDIR, "clip_thresholds_partB_pyt.json"),  # PYT-AXIS
                  CFG["md"]):  # PYT-AXIS
            if os.path.exists(p):  # PYT-AXIS
                raise SystemExit(f"ABORT (6): {p} exists (Part B is never overwritten)")  # PYT-AXIS
    # ---------------- B0 guard ----------------------------------------------  # PYT-AXIS
    scan_start = run_scan("start")  # PYT-AXIS
    plan_sha = sha256_file(PLAN)
    if plan_sha != PLAN_SHA:
        raise SystemExit(f"ABORT (5): plan hash changed: {plan_sha}")  # PYT-AXIS
    llpb_sha = sha256_file(LLAMA_PARTB)  # PYT-AXIS
    if llpb_sha != LLAMA_PARTB_SHA:  # PYT-AXIS
        raise SystemExit(f"ABORT (5): frozen LLaMA partB_features.json hash changed: {llpb_sha}")  # PYT-AXIS
    olpb_sha = sha256_file(OL_PARTB)  # PYT-AXIS
    if olpb_sha != OL_PARTB_SHA:  # PYT-AXIS
        raise SystemExit(f"ABORT (5): frozen partB_features_ol.json hash changed: {olpb_sha}")  # PYT-AXIS
    LLPB = json.load(open(LLAMA_PARTB))  # PYT-AXIS
    OLPB = json.load(open(OL_PARTB))  # PYT-AXIS
    os.makedirs(OUTDIR, exist_ok=True)  # PYT-AXIS
    PYT = CFG["pyt"]  # PYT-AXIS: 7 slugs in size order
    if not CFG["test"]:  # PYT-AXIS: the 7 extractions must be complete (S4)
        for slug in PYT:  # PYT-AXIS
            for p in (os.path.join(FEAT, "wit1024_pythia_axis", slug + ".done"), npy_path(slug, tok1=True)):  # PYT-AXIS
                if not os.path.isfile(p):  # PYT-AXIS
                    raise SystemExit(f"ABORT (5): {p} missing (S4 not complete)")  # PYT-AXIS

    # ---------------- inputs ------------------------------------------------
    MODELS = PYT + VIT_STEMS + list(RECHECK)  # PYT-AXIS
    inputs, sha_cache = {}, {}  # PYT-AXIS
    for key in MODELS:  # PYT-AXIS
        p = npy_path(key)  # PYT-AXIS
        sc = sidecar(key)  # PYT-AXIS
        if p not in sha_cache:  # PYT-AXIS
            sha_cache[p] = sha256_file(p)  # PYT-AXIS
        inputs[key] = {"path": os.path.relpath(p, PROJ), "sha256": sha_cache[p], "kind": sc["kind"],  # PYT-AXIS
                       "shape": sc["shape"], "num_params": int(sc["num_params"]),  # PYT-AXIS
                       "width": sc.get("hidden_size"), "depth_blocks": sc.get("n_layers"),  # PYT-AXIS
                       "source_stem": CFG["src"][key][0]}  # PYT-AXIS
    tok1_inputs = {}  # PYT-AXIS
    for key in PYT:  # PYT-AXIS
        p = npy_path(key, tok1=True)  # PYT-AXIS
        if p not in sha_cache:  # PYT-AXIS
            sha_cache[p] = sha256_file(p)  # PYT-AXIS
        tok1_inputs[key] = {"path": os.path.relpath(p, PROJ), "sha256": sha_cache[p]}  # PYT-AXIS
    nparams = {k: inputs[k]["num_params"] for k in PYT}  # PYT-AXIS
    if not CFG["test"]:  # PYT-AXIS: plan sections 1 and 2 (registry values; extraction-side registry, read-only)
        sys.path.insert(0, os.path.join(PROJ, "extract", "pythia_axis"))  # PYT-AXIS
        import registry_pythia_axis as REG  # noqa: E402  # PYT-AXIS
        for i, s in enumerate(SIZES):  # PYT-AXIS
            ex = REG.EXPECTED[REG.MODEL_OF[s]]  # PYT-AXIS
            assert inputs[PYT[i]]["shape"] == [M, ex["stored_layers"], ex["hidden_size"]], (s, inputs[PYT[i]]["shape"])  # PYT-AXIS
            assert inputs[PYT[i]]["shape"][1] == L_STORED[i]  # PYT-AXIS
            assert nparams[PYT[i]] == ex["num_params"], (s, nparams[PYT[i]])  # PYT-AXIS
            assert inputs[PYT[i]]["width"] == ex["hidden_size"] and inputs[PYT[i]]["depth_blocks"] == ex["n_layers"]  # PYT-AXIS
    sizes_np = [nparams[k] for k in PYT]  # PYT-AXIS
    if sizes_np != sorted(sizes_np) or len(set(sizes_np)) != len(sizes_np):  # PYT-AXIS
        raise SystemExit(f"ABORT (5): num_params not strictly increasing in size order: {sizes_np}")  # PYT-AXIS
    print(f"[{time.time() - t0:6.0f}s] hashed {len(sha_cache)} feature stacks", flush=True)  # PYT-AXIS

    # ---------------- clip thresholds (all-layer float64 stack) ------------  # PYT-AXIS
    clip, clip_by_src = {}, {}  # PYT-AXIS
    for key in MODELS:  # PYT-AXIS
        p = npy_path(key)  # PYT-AXIS
        if p not in clip_by_src:  # PYT-AXIS
            st64 = np.asarray(stack(key), dtype=np.float64)  # PYT-AXIS
            clip_by_src[p] = prh_clip_threshold(st64)  # PYT-AXIS
            del st64  # PYT-AXIS
        clip[key] = clip_by_src[p]  # PYT-AXIS
    clip_tok1 = {}  # PYT-AXIS: D7 only, each _tok1 stack's own threshold [op]
    for key in PYT:  # PYT-AXIS
        p = npy_path(key, tok1=True)  # PYT-AXIS
        if p not in clip_by_src:  # PYT-AXIS
            st64 = np.asarray(stack(key, tok1=True), dtype=np.float64)  # PYT-AXIS
            clip_by_src[p] = prh_clip_threshold(st64)  # PYT-AXIS
            del st64  # PYT-AXIS
        clip_tok1[key] = clip_by_src[p]  # PYT-AXIS
    print(f"[{time.time() - t0:6.0f}s] clip thresholds: " + ", ".join(f"{k}={clip[k]!r}" for k in PYT), flush=True)  # PYT-AXIS

    # ---------------- B1: per-layer PR --------------------------------------
    PR, TR = {pre: {} for pre in PRES}, {pre: {} for pre in PRES}
    PR_T1 = {pre: {} for pre in PRES}  # PYT-AXIS: D7 _tok1 PRs
    pr_by_src = {}  # PYT-AXIS

    def _prs(key, tok1=False):  # PYT-AXIS
        p = npy_path(key, tok1)  # PYT-AXIS
        if p not in pr_by_src:  # PYT-AXIS
            st = stack(key, tok1)  # PYT-AXIS
            L = st.shape[1]  # PYT-AXIS
            res = {}  # PYT-AXIS
            for pre in PRES:  # PYT-AXIS
                c = (clip_tok1[key] if tok1 else clip[key]) if pre == "prh_clip+l2" else None  # PYT-AXIS
                prs, trs = np.empty(L), np.empty(L)  # PYT-AXIS
                for l in range(L):  # PYT-AXIS
                    prs[l], trs[l], _ = layer_pr(np.asarray(st[:, l, :]), pre, c)  # PYT-AXIS
                res[pre] = (prs, trs)  # PYT-AXIS
            pr_by_src[p] = res  # PYT-AXIS
            del st  # PYT-AXIS
        return pr_by_src[p]  # PYT-AXIS
    for key in MODELS:  # PYT-AXIS
        r_ = _prs(key)  # PYT-AXIS
        for pre in PRES:  # PYT-AXIS
            PR[pre][key], TR[pre][key] = r_[pre]  # PYT-AXIS
        print(f"[{time.time() - t0:6.0f}s] PR {key} L={PR['l2'][key].size} max(prh_clip+l2)={PR['prh_clip+l2'][key].max():.2f} "  # PYT-AXIS
              f"max(l2)={PR['l2'][key].max():.2f}", flush=True)  # PYT-AXIS
    for key in PYT:  # PYT-AXIS
        if CFG["tok1"][key] == CFG["src"][key]:  # PYT-AXIS: stand-in (test mode) uses the primary stack
            for pre in PRES:  # PYT-AXIS
                PR_T1[pre][key] = PR[pre][key]  # PYT-AXIS
            continue  # PYT-AXIS
        r_ = _prs(key, tok1=True)  # PYT-AXIS
        for pre in PRES:  # PYT-AXIS
            PR_T1[pre][key] = r_[pre][0]  # PYT-AXIS

    # ---------------- B1 check: recheck models and the 17 ViTs vs the frozen Part B files --------------  # PYT-AXIS
    recheck = {"tolerance_rel": 1e-14, "per_model": {}}  # PYT-AXIS
    for key in list(RECHECK) + VIT_STEMS:  # PYT-AXIS
        src = RECHECK.get(key, (None, "llama"))[1]  # PYT-AXIS
        F0 = OLPB if src == "ol" else LLPB  # PYT-AXIS
        e = {}  # PYT-AXIS
        for pre in PRES:  # PYT-AXIS
            ref = F0["per_pre"][pre]["lm"].get(key) or F0["per_pre"][pre]["vit"].get(key)  # PYT-AXIS
            a, b = PR[pre][key], np.asarray(ref["PR"], dtype=float)  # PYT-AXIS
            e[pre] = float(np.max(np.abs(a - b) / np.abs(b))) if a.shape == b.shape else float("inf")  # PYT-AXIS
        cref = F0["meta"]["clip_thresholds"].get(key)  # PYT-AXIS
        recheck["per_model"][key] = {"frozen_partB": src, "max_rel_err_PR": e, "clip": clip[key], "clip_frozen": cref,  # PYT-AXIS
                                     "clip_bitwise": (cref is not None and clip[key] == float(cref))}  # PYT-AXIS
    errs = [v for d in recheck["per_model"].values() for v in d["max_rel_err_PR"].values()]  # PYT-AXIS
    recheck["n_models"] = len(recheck["per_model"])  # PYT-AXIS
    recheck["max_rel_err_PR"] = max(errs)  # PYT-AXIS
    recheck["all_clip_bitwise"] = all(d["clip_bitwise"] for d in recheck["per_model"].values())  # PYT-AXIS
    recheck["pass"] = bool(recheck["max_rel_err_PR"] <= 1e-14 and recheck["all_clip_bitwise"]  # PYT-AXIS
                           and recheck["n_models"] == 3 + len(VIT_STEMS))  # PYT-AXIS
    rc_path = os.path.join(OUTDIR, "partB_recompute_check_pyt.json")  # PYT-AXIS
    if CFG["test"]:  # PYT-AXIS
        keep_aside(rc_path)  # PYT-AXIS
    with open(rc_path, "w") as f:  # PYT-AXIS
        json.dump(jsonable(recheck), f, indent=1)  # PYT-AXIS
    print(f"[{time.time() - t0:6.0f}s] B1 recheck of {recheck['n_models']} known models (3 LMs + 17 ViTs): max rel err PR "  # PYT-AXIS
          f"{recheck['max_rel_err_PR']:.3e}, clip bitwise {recheck['all_clip_bitwise']} -> pass {recheck['pass']}", flush=True)  # PYT-AXIS
    if not recheck["pass"]:  # PYT-AXIS
        raise SystemExit("ABORT (5): B1 recomputation of the known models failed (see partB_recompute_check_pyt.json)")  # PYT-AXIS

    # ---------------- B2-B6 --------------------------------------------------  # PYT-AXIS
    used_scores: dict = {}  # frozen known score files opened (path -> sha256)  # PYT-AXIS
    out = {"schema": "PYT-PARTB v1", "m": M, "delta_b": DELTA_B,  # PYT-AXIS: VERDICTS_SPEC_PYT section 11
           "sizes": SIZES, "pyt_lms": PYT, "vits": VIT_STEMS, "vit_family4": dict(VK.VIT_FAMILY4),  # PYT-AXIS
           "num_params": {k: nparams[k] for k in PYT}, "width": {k: inputs[k]["width"] for k in PYT},  # PYT-AXIS
           "depth_blocks": {k: inputs[k]["depth_blocks"] for k in PYT},  # PYT-AXIS
           "L_stored": {s: int(PR["l2"][s].size) for s in PYT + VIT_STEMS},  # PYT-AXIS
           "band_constants": {p: dict(BAND[p]) for p in PRES}, "mu_band": list(MU_BAND),  # PYT-AXIS
           "meta": {"role": "pyt-axis-predict (code: S0b role pyt-code-infra)", "plan": os.path.relpath(PLAN, PROJ),  # PYT-AXIS
                    "plan_sha256": plan_sha,  # PYT-AXIS
                    "script_sha256": sha256_file(os.path.abspath(__file__)), "m": M, "k": KNN_K, "delta_b": DELTA_B,
                    "guard_sha256": sha256_file(guard.__file__),  # PYT-AXIS
                    "verdict_keys_sha256": sha256_file(VK.__file__),  # PYT-AXIS
                    "test_mode": CFG["test"], "standin": CFG["standin"], "size_labels": CFG["slugs"],  # PYT-AXIS
                    "source_stems": {k: CFG["src"][k][0] for k in PYT}, "tok1_label": CFG["tok1_label"],  # PYT-AXIS
                    "threads": _TH, "python": platform.python_version(), "numpy": np.__version__,  # PYT-AXIS
                    "started_utc": started, "guard_scan_start": scan_start, "inputs": inputs, "tok1_inputs": tok1_inputs,  # PYT-AXIS
                    "llama_partB": {"path": os.path.relpath(LLAMA_PARTB, PROJ), "sha256": llpb_sha},  # PYT-AXIS
                    "ol_partB": {"path": os.path.relpath(OL_PARTB, PROJ), "sha256": olpb_sha},  # PYT-AXIS
                    "recheck": {k: recheck[k] for k in ("n_models", "max_rel_err_PR", "all_clip_bitwise", "pass")},  # PYT-AXIS
                    "clip_thresholds": {k: clip[k] for k in PYT}, "clip_thresholds_tok1": clip_tok1,  # PYT-AXIS
                    "mknn_baseline_every_pair": KNN_K / (M - 1), "debiased_cka_baseline_every_pair": 0.0,
                    "note": "features only, plus known frozen scores of BLOOMZ, LLaMA-1 and OpenLLaMA (B5); "  # PYT-AXIS
                            "no cross-model score involving a Pythia model computed or opened"},  # PYT-AXIS
           "per_pre": {}}
    comps = [(a, b, j) for a, b in itertools.combinations(range(len(SIZES)), 2) for j in range(len(VIT_STEMS))]  # PYT-AXIS
    logp = np.log(np.array(sizes_np, dtype=float))  # PYT-AXIS
    for pre in PRES:
        P = PR[pre]
        lm_info, vit_info = {}, {}
        for s in PYT + VIT_STEMS:  # PYT-AXIS
            v = P[s]
            L = v.size
            order = np.argsort(-v, kind="stable")
            rel = np.arange(L) / (L - 1)  # PYT-AXIS
            info = {"L": L, "PR": v.tolist(), "trKc": TR[pre][s].tolist(),
                    "PR_star": float(v[order[0]]), "argmax": int(order[0]), "argmax_rel": float(order[0] / (L - 1)),
                    "PR_second": float(v[order[1]]), "argmax_second": int(order[1]),
                    "PR_star_over_second": float(v[order[0]] / v[order[1]]),
                    "PR_first_layer": float(v[0]), "PR_last_layer": float(v[-1]), "PR_min": float(v.min()),
                    "fixed": {str(r): {"layer": fixed_layer(L, r), "PR": float(v[fixed_layer(L, r)])} for r in DEPTHS}}
            if s in PYT:  # PYT-AXIS: B_late side (LM states with relative depth >= 0.60)
                late = np.flatnonzero(rel >= LATE_REL - 1e-12)  # PYT-AXIS
                ul = int(late[np.argmax(v[late])])  # first maximum  # PYT-AXIS
                info.update({"late_first_layer": int(late[0]), "PR_late_star": float(v[ul]), "u_late": ul,  # PYT-AXIS
                             "u_late_rel": float(ul / (L - 1))})  # PYT-AXIS
            (lm_info if s in PYT else vit_info)[s] = info  # PYT-AXIS
        if not CFG["test"]:  # PYT-AXIS: plan section 3 layer table
            for r in DEPTHS:  # PYT-AXIS
                assert [lm_info[s]["fixed"][str(r)]["layer"] for s in PYT] == FIXED_LAYERS[r], r  # PYT-AXIS
        # B2: Bbar (7 x 17), argmax, fixed depth, late  # PYT-AXIS
        Bbar = np.empty((len(PYT), len(VIT_STEMS)))  # PYT-AXIS
        Bfd = {r: np.empty_like(Bbar) for r in DEPTHS}
        Blate = np.empty_like(Bbar)  # PYT-AXIS
        Bbar_bruteforce_maxdiff = 0.0
        Blate_bruteforce_maxdiff = 0.0  # PYT-AXIS
        for i, a in enumerate(PYT):  # PYT-AXIS
            for j, b in enumerate(VIT_STEMS):  # PYT-AXIS
                Buw = np.sqrt(P[a][:, None] * P[b][None, :]) / (M - 1)
                Bbar[i, j] = math.sqrt(lm_info[a]["PR_star"] * vit_info[b]["PR_star"]) / (M - 1)
                Bbar_bruteforce_maxdiff = max(Bbar_bruteforce_maxdiff, abs(float(Buw.max()) - Bbar[i, j]))
                for r in DEPTHS:
                    Bfd[r][i, j] = Buw[lm_info[a]["fixed"][str(r)]["layer"], vit_info[b]["fixed"][str(r)]["layer"]]
                Blate[i, j] = math.sqrt(lm_info[a]["PR_late_star"] * vit_info[b]["PR_star"]) / (M - 1)  # PYT-AXIS
                Blate_bruteforce_maxdiff = max(Blate_bruteforce_maxdiff,  # PYT-AXIS
                                               abs(float(Buw[lm_info[a]["late_first_layer"]:].max()) - Blate[i, j]))  # PYT-AXIS
        # B3/B4: trends, materiality, gaps  # PYT-AXIS
        grids = {"Bbar": Bbar, "Bfd_0.5": Bfd[0.5], "Bfd_1.0": Bfd[1.0], "Blate": Blate}  # PYT-AXIS
        trends = {k: trend_rows(G) for k, G in grids.items()}  # PYT-AXIS
        mat = {k: materiality(G, list(range(1, 8)), trends[k]["Tr"]) for k, G in grids.items()}  # PYT-AXIS
        mat_curve = {k: {repr(float(db)): materiality(grids[k], list(range(1, 8)), trends[k]["Tr"], db)["material"]  # PYT-AXIS
                         for db in DBS} for k in ("Bbar", "Blate")}  # PYT-AXIS
        gaps = {}  # PYT-AXIS: 21 size pairs, B(larger) - B(smaller) per ViT
        for k, G in grids.items():  # PYT-AXIS
            gaps[k] = {}  # PYT-AXIS
            for a, b in itertools.combinations(range(len(SIZES)), 2):  # PYT-AXIS
                d = G[b] - G[a]  # PYT-AXIS
                gaps[k][f"{SIZES[a]}|{SIZES[b]}"] = {"per_vit": d.tolist(), "median": float(np.median(d)),  # PYT-AXIS
                                                     "median_abs": float(np.median(np.abs(d))),  # PYT-AXIS
                                                     "n_vit_abs_ge_delta_b": int(np.sum(np.abs(d) >= DELTA_B))}  # PYT-AXIS
        i5 = [SIZES.index(s) for s in FIVE]  # PYT-AXIS
        tr5 = trend_rows(Bbar[i5])["Tr"]  # PYT-AXIS
        prstar = np.array([lm_info[s]["PR_star"] for s in PYT])  # PYT-AXIS
        sumd2 = spearman_exact_sumd2(prstar)  # PYT-AXIS: exact check of Tr(Bbar) = Spearman(size, PR*)
        tr_exact = None if sumd2 is None else float(1 - Fraction(6 * sumd2, 7 * 48))  # PYT-AXIS
        # B6: s, s_late, membership  # PYT-AXIS
        Tr_B = float(trends["Bbar"]["Tr"])  # PYT-AXIS
        s_ = sgn(Tr_B) if sgn(Tr_B) != 0 else 1  # PYT-AXIS
        Tr_late = float(trends["Blate"]["Tr"])  # PYT-AXIS
        s_late = sgn(Tr_late) if sgn(Tr_late) != 0 else 1  # PYT-AXIS
        member_H1 = bool(mat["Bbar"]["material"])  # PYT-AXIS
        h2 = "H2a" if mat["Blate"]["material"] else "H2b"  # PYT-AXIS
        members = (["H1"] if member_H1 else []) + [h2, "H3", "H4"]  # PYT-AXIS
        blate_eq = {SIZES[i]: bool(lm_info[s]["PR_late_star"] == lm_info[s]["PR_star"]) for i, s in enumerate(PYT)}  # PYT-AXIS
        n_eq = int(sum(blate_eq.values()))  # PYT-AXIS
        bc = BAND[pre]  # PYT-AXIS
        band, extrap, hcurve = {}, {}, {repr(float(h)): {} for h in HS}  # PYT-AXIS
        for i, s in enumerate(SIZES):  # PYT-AXIS
            ps = lm_info[PYT[i]]["PR_star"]  # PYT-AXIS
            f = bc["a"] + bc["b"] / math.sqrt(ps)  # PYT-AXIS
            band[s] = {"f": f, "lo": f - bc["h"], "hi": f + bc["h"], "PR_star": ps}  # PYT-AXIS
            extrap[s] = bool(ps < bc["fit_min_PR"])  # PYT-AXIS
            for h in HS:  # PYT-AXIS
                hcurve[repr(float(h))][s] = [f - h, f + h]  # PYT-AXIS
        # qualifying comparisons (357, features only)  # PYT-AXIS
        n2 = int(sum(abs(Bbar[b, j] - Bbar[a, j]) >= DELTA_B for a, b, j in comps))  # PYT-AXIS
        n5 = int(sum(abs(Blate[b, j] - Blate[a, j]) >= DELTA_B for a, b, j in comps))  # PYT-AXIS
        quals = {"PP2aii_n": n2, "PP2aii_need": ceil_frac(9 * n2, 10), "PP2aii_need67": ceil_frac(2 * n2, 3),  # PYT-AXIS
                 "PP5a_n": n5, "PP5a_need": ceil_frac(2 * n5, 3),  # PYT-AXIS
                 "PP2aii_n_by_delta_b": {repr(float(db)): int(sum(abs(Bbar[b, j] - Bbar[a, j]) >= db for a, b, j in comps))  # PYT-AXIS
                                         for db in DBS}}  # PYT-AXIS
        # B5: extras (outcomes known, not tests)  # PYT-AXIS
        widths = [inputs[k]["width"] for k in PYT]  # PYT-AXIS
        depths = [inputs[k]["depth_blocks"] for k in PYT]  # PYT-AXIS
        meanB = Bbar.mean(axis=1)  # PYT-AXIS
        b5 = {"label": "extras, outcomes known or features only, not tests",  # PYT-AXIS
              "spearman_PRstar": {"logparams": spearman(prstar, logp), "width": spearman(prstar, widths),  # PYT-AXIS
                                  "depth": spearman(prstar, depths)},  # PYT-AXIS
              "spearman_meanBbar": {"logparams": spearman(meanB, logp), "width": spearman(meanB, widths),  # PYT-AXIS
                                    "depth": spearman(meanB, depths)},  # PYT-AXIS
              "D4_steps_PRstar": {st: float(prstar[SIZES.index(st.split("|")[1])] - prstar[SIZES.index(st.split("|")[0])])  # PYT-AXIS
                                  for st in VK.D4_STEPS},  # PYT-AXIS
              "PR_fixed_depths": {s: {r: lm_info[PYT[i]]["fixed"][r] for r in ("0.5", "1.0")} for i, s in enumerate(SIZES)},  # PYT-AXIS
              "runner_up_ratio": {s: lm_info[PYT[i]]["PR_star_over_second"] for i, s in enumerate(SIZES)},  # PYT-AXIS
              "n_Blate_equals_Bbar": n_eq, "extrapolated": extrap}  # PYT-AXIS
        b5["offset_retrodiction_187"] = known_offsets(pre, LLPB, OLPB, used_scores)  # PYT-AXIS
        # D7 (descriptive): _tok1 PR*, Bbar_tok1, Tr_P(Bbar_tok1) against Tr_P(Bbar)  # PYT-AXIS
        pt1 = {s: float(np.max(PR_T1[pre][PYT[i]])) for i, s in enumerate(SIZES)}  # PYT-AXIS
        Bt1 = np.array([[math.sqrt(pt1[s] * vit_info[v]["PR_star"]) / (M - 1) for v in VIT_STEMS] for s in SIZES])  # PYT-AXIS
        d7 = {"label": "descriptive (plan section 6 D7); " + CFG["tok1_label"], "PR_star_tok1": pt1,  # PYT-AXIS
              "PR_star_tok1_over_PR_star": {s: pt1[s] / lm_info[PYT[i]]["PR_star"] for i, s in enumerate(SIZES)},  # PYT-AXIS
              "argmax_tok1": {s: int(np.argmax(PR_T1[pre][PYT[i]])) for i, s in enumerate(SIZES)},  # PYT-AXIS
              "PR_tok1": {s: np.asarray(PR_T1[pre][PYT[i]]).tolist() for i, s in enumerate(SIZES)},  # PYT-AXIS
              "Tr_Bbar_tok1": trend_rows(Bt1)["Tr"], "Tr_Bbar": Tr_B,  # PYT-AXIS
              "mean_Bbar_tok1": {s: float(Bt1[i].mean()) for i, s in enumerate(SIZES)}}  # PYT-AXIS
        allst = PYT + VIT_STEMS  # PYT-AXIS
        info_all = dict(lm_info, **vit_info)  # PYT-AXIS
        out["per_pre"][pre] = {  # PYT-AXIS: VERDICTS_SPEC_PYT section 11 fields first, then the full detail
            "clip_c": {s: (clip[s] if pre == "prh_clip+l2" else None) for s in allst},  # PYT-AXIS
            "PR": {s: info_all[s]["PR"] for s in allst},  # PYT-AXIS
            "PR_star": {s: info_all[s]["PR_star"] for s in allst},  # PYT-AXIS
            "u_star": {s: info_all[s]["argmax"] for s in allst},  # PYT-AXIS
            "u_star_rel": {s: info_all[s]["argmax_rel"] for s in allst},  # PYT-AXIS
            "Bbar": {a: {b: float(Bbar[i, j]) for j, b in enumerate(VIT_STEMS)} for i, a in enumerate(PYT)},  # PYT-AXIS
            "Bbar_argmax": {a: {b: [lm_info[a]["argmax"], vit_info[b]["argmax"]] for b in VIT_STEMS} for a in PYT},  # PYT-AXIS
            "Bfd": {str(r): {a: {b: float(Bfd[r][i, j]) for j, b in enumerate(VIT_STEMS)}  # PYT-AXIS
                             for i, a in enumerate(PYT)} for r in DEPTHS},  # PYT-AXIS
            "Blate": {a: {b: float(Blate[i, j]) for j, b in enumerate(VIT_STEMS)} for i, a in enumerate(PYT)},  # PYT-AXIS
            "gaps": gaps,  # PYT-AXIS
            "Tr_Bbar": Tr_B, "Tr_Bbar_5size": tr5, "Tr_Bbar_exact_from_PRstar": tr_exact,  # PYT-AXIS
            "Tr_Bfd": {str(r): float(trends[f"Bfd_{r}"]["Tr"]) for r in DEPTHS}, "Tr_Blate": Tr_late,  # PYT-AXIS
            "rho_common_to_all_vits": {k: trends[k]["rho_common_to_all_vits"] for k in grids},  # PYT-AXIS
            "materiality": mat, "material_Bbar": member_H1, "material_Blate": bool(mat["Blate"]["material"]),  # PYT-AXIS
            "material_Bfd": {str(r): bool(mat[f"Bfd_{r}"]["material"]) for r in DEPTHS},  # PYT-AXIS
            "material_curve_db": mat_curve,  # PYT-AXIS
            "s": s_, "s_flag_zero": bool(sgn(Tr_B) == 0), "s_late": s_late, "s_late_flag_zero": bool(sgn(Tr_late) == 0),  # PYT-AXIS
            "members": members, "H2_branch": h2,  # PYT-AXIS
            "Blate_equals_Bbar": blate_eq, "n_Blate_equals_Bbar": n_eq, "H2_same_as_H1": bool(n_eq >= 6),  # PYT-AXIS
            "band": band, "band_h_curve": hcurve, "band_flat": list(bc["flat"]), "extrapolated": extrap,  # PYT-AXIS
            "qualifying": quals,  # PYT-AXIS
            "B5": b5, "D7": d7,  # PYT-AXIS
            "lm": lm_info, "vit": vit_info,  # PYT-AXIS
            "Bbar_factorised_vs_bruteforce_maxabs": Bbar_bruteforce_maxdiff,  # PYT-AXIS
            "Blate_factorised_vs_bruteforce_maxabs": Blate_bruteforce_maxdiff}  # PYT-AXIS
        print(f"== {pre}: Tr(Bbar) {Tr_B:+.4f} (exact {tr_exact}) material {member_H1} s {s_:+d} | Tr(Blate) {Tr_late:+.4f} "  # PYT-AXIS
              f"material {mat['Blate']['material']} -> {h2} | members {members} | Blate=Bbar {n_eq}/7 | "  # PYT-AXIS
              f"PR* {[round(float(x), 2) for x in prstar]} | extrapolated {[s for s in SIZES if extrap[s]]} | "  # PYT-AXIS
              f"PP2aii n {n2} need {quals['PP2aii_need']} | PP5a n {n5} need {quals['PP5a_need']} | "  # PYT-AXIS
              f"retro in band {b5['offset_retrodiction_187']['n_in_band']}/{b5['offset_retrodiction_187']['n']}", flush=True)  # PYT-AXIS
    out["meta"]["score_inputs"] = {p: sha256_file(os.path.join(PROJ, p)) for p in sorted(used_scores)}  # PYT-AXIS
    out["meta"]["n_score_inputs"] = len(used_scores)  # PYT-AXIS
    # ---------------- B0 end scan, then B7 writes ------------------------------  # PYT-AXIS
    scan_end = run_scan("end")  # PYT-AXIS
    out["meta"]["guard_scan_end"] = scan_end  # PYT-AXIS
    out["meta"]["finished_utc"] = utc()  # PYT-AXIS
    paths = {"clip": os.path.join(OUTDIR, "clip_thresholds_partB_pyt.json"),  # PYT-AXIS
             "json": os.path.join(OUTDIR, "partB_features_pyt.json"), "md": CFG["md"],  # PYT-AXIS
             "line": os.path.join(OUTDIR, "STATUS_LINE_DRAFT.txt")}  # PYT-AXIS
    for p in paths.values():  # PYT-AXIS: no-delete rule
        if CFG["test"]:  # PYT-AXIS
            keep_aside(p)  # PYT-AXIS
        elif os.path.exists(p):  # PYT-AXIS
            raise SystemExit(f"ABORT (6): {p} exists (Part B is never overwritten)")  # PYT-AXIS
    with open(paths["clip"], "w") as f:  # PYT-AXIS
        json.dump(jsonable({"note": "hrhv2.prep.prh_clip_threshold on the float64 all-layer stack, q=0.95 (plan B1); "  # PYT-AXIS
                                    "V-P1 must reproduce these bitwise",  # PYT-AXIS
                            "clip": {k: {"c": clip[k]} for k in PYT}, "written_utc": utc()}), f, indent=1)  # PYT-AXIS
    with open(paths["json"] + ".tmp", "w") as f:  # PYT-AXIS
        json.dump(jsonable(out), f, indent=1)  # PYT-AXIS
    os.replace(paths["json"] + ".tmp", paths["json"])  # PYT-AXIS
    print(f"[{time.time() - t0:6.0f}s] wrote {paths['json']}", flush=True)  # PYT-AXIS
    js_sha = sha256_file(paths["json"])  # PYT-AXIS
    F = json.load(open(paths["json"]))  # PYT-AXIS
    with open(paths["md"], "w") as f:  # PYT-AXIS
        f.write("\n".join(build_md(F, js_sha)) + "\n")  # PYT-AXIS
    md_sha = sha256_file(paths["md"])  # PYT-AXIS
    with open(paths["line"], "w") as f:  # PYT-AXIS
        f.write(status_line(F, js_sha, md_sha, sha256_file(paths["clip"])) + "\n")  # PYT-AXIS
    print(f"[{time.time() - t0:6.0f}s] wrote DRAFT {paths['md']} and {paths['line']} (review, chmod 444, STATUS line)",  # PYT-AXIS
          flush=True)  # PYT-AXIS


def known_offsets(pre: str, LLPB: dict, OLPB: dict, used: dict) -> dict:  # PYT-AXIS: plan B5 and section 0
    """c_alpha - Bbar on the 187 known pairs (frozen pair files, cka_biased, full) under the PR*-indexed band; the
    11-LM fit of the per-LM median offset on 1/sqrt(PR*) recomputed (OLS). Outcomes known: not a test."""  # PYT-AXIS
    bc = BAND[pre]  # PYT-AXIS
    vit_pr = {v: LLPB["per_pre"][pre]["vit"][v]["PR_star"] for v in VIT_STEMS}  # PYT-AXIS
    rows, per_lm = [], {}  # PYT-AXIS
    for lm, base in KNOWN11:  # PYT-AXIS
        ref = OLPB["per_pre"][pre]["lm"][lm] if lm == "openlm-research__open_llama_13b" else LLPB["per_pre"][pre]["lm"][lm]  # PYT-AXIS
        ps = float(ref["PR_star"])  # PYT-AXIS
        f = bc["a"] + bc["b"] / math.sqrt(ps)  # PYT-AXIS
        offs = []  # PYT-AXIS
        for v in VIT_STEMS:  # PYT-AXIS
            js, p = read_pair_json(base, pre, lm, v)  # PYT-AXIS
            used[os.path.relpath(p, PROJ)] = None  # PYT-AXIS
            fb = js["summaries"]["cka_biased"]["full"]  # PYT-AXIS
            o = float(fb["c_alpha"]) - math.sqrt(ps * vit_pr[v]) / (M - 1)  # PYT-AXIS
            offs.append(o)  # PYT-AXIS
            rows.append({"lm": lm, "vit": v, "offset": o, "f": f, "in_band": bool(f - bc["h"] <= o <= f + bc["h"]),  # PYT-AXIS
                         "dev_from_fit": o - f,  # PYT-AXIS
                         "null_mean_minus_Bbar": float(fb["null_mean"]) - math.sqrt(ps * vit_pr[v]) / (M - 1),  # PYT-AXIS
                         "null_sd": float(fb.get("null_sd", float("nan")))})  # PYT-AXIS
        per_lm[lm] = {"PR_star": ps, "median_offset": float(np.median(offs)), "min": float(np.min(offs)),  # PYT-AXIS
                      "max": float(np.max(offs)), "f": f, "L_stored": int(ref["L"])}  # PYT-AXIS
    x = np.array([1 / math.sqrt(d["PR_star"]) for d in per_lm.values()])  # PYT-AXIS
    y = np.array([d["median_offset"] for d in per_lm.values()])  # PYT-AXIS
    A = np.column_stack([np.ones_like(x), x])  # PYT-AXIS
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)  # PYT-AXIS
    res = y - A @ coef  # PYT-AXIS
    devs = [r["offset"] - (coef[0] + coef[1] * (1 / math.sqrt(per_lm[r["lm"]]["PR_star"]))) for r in rows]  # PYT-AXIS
    nm = [r["null_mean_minus_Bbar"] for r in rows]  # PYT-AXIS
    sd = [r["null_sd"] for r in rows]  # PYT-AXIS
    return {"label": "retrodiction on known pairs, all inside by construction (plan B5)", "n": len(rows),  # PYT-AXIS
            "n_in_band": int(sum(r["in_band"] for r in rows)),  # PYT-AXIS
            "max_abs_dev_from_frozen_f": float(max(abs(r["dev_from_fit"]) for r in rows)),  # PYT-AXIS
            "refit": {"a": float(coef[0]), "b": float(coef[1]), "residual_sd": float(np.std(res, ddof=2)),  # PYT-AXIS
                      "corr": float(np.corrcoef(x, y)[0, 1]), "max_abs_pair_minus_fit": float(max(abs(d) for d in devs)),  # PYT-AXIS
                      "PR_range": [float(min(d["PR_star"] for d in per_lm.values())),  # PYT-AXIS
                                   float(max(d["PR_star"] for d in per_lm.values()))],  # PYT-AXIS
                      "frozen": {"a": bc["a"], "b": bc["b"]},  # PYT-AXIS
                      "abs_diff_a": abs(float(coef[0]) - bc["a"]), "abs_diff_b": abs(float(coef[1]) - bc["b"])},  # PYT-AXIS
            "null_mean_minus_Bbar_range": [float(min(nm)), float(max(nm))],  # PYT-AXIS
            "null_sd_range": [float(np.nanmin(sd)), float(np.nanmax(sd))],  # PYT-AXIS
            "per_lm": per_lm, "rows": rows}  # PYT-AXIS


# ---------------------------------------------------------------------------------------------------  # PYT-AXIS
# B7 draft of PARTB_PREDICTIONS_PYT.md (numbers from the JSON just written) and the STATUS line draft  # PYT-AXIS
# ---------------------------------------------------------------------------------------------------  # PYT-AXIS

def _f(x, nd=4):  # PYT-AXIS
    return "n/a" if x is None else f"{x:.{nd}f}"  # PYT-AXIS


def _short(s: str) -> str:  # PYT-AXIS
    return s.split("__")[-1]  # PYT-AXIS


def status_line(F: dict, js_sha: str, md_sha: str, clip_sha: str) -> str:  # PYT-AXIS: plan B7 fields
    Q = F["per_pre"]["prh_clip+l2"]  # PYT-AXIS
    m = F["meta"]  # PYT-AXIS
    bands = "; ".join(f"{s} [{Q['band'][s]['lo']:.6f}, {Q['band'][s]['hi']:.6f}]" for s in F["sizes"])  # PYT-AXIS
    ex = ",".join(s for s in F["sizes"] if Q["extrapolated"][s]) or "none"  # PYT-AXIS
    g0, g1 = m["guard_scan_start"], m.get("guard_scan_end", {})  # PYT-AXIS
    return ("<UTC>Z [pyt-axis-predict] Part B FROZEN (S5, plan section 9 B7): "  # PYT-AXIS
            f"analysis/s12_layermax_pythia/PARTB_PREDICTIONS_PYT.md sha256 {md_sha} ; "  # PYT-AXIS
            f"results/S1_2_layermax_pythia/partB/partB_features_pyt.json sha256 {js_sha} ; "  # PYT-AXIS
            f"analysis/s12_layermax_pythia/partb_pyt.py sha256 {m['script_sha256']} ; "  # PYT-AXIS
            f"clip_thresholds_partB_pyt.json sha256 {clip_sha} ; members {','.join(Q['members'])} ; "  # PYT-AXIS
            f"s {Q['s']:+d} (Tr_P(Bbar) {Q['Tr_Bbar']:+.4f}) ; s_late {Q['s_late']:+d} (Tr_P(B_late) {Q['Tr_Blate']:+.4f}) ; "  # PYT-AXIS
            f"H2 branch {Q['H2_branch']} ; B_late = Bbar for {Q['n_Blate_equals_Bbar']}/7 models ; "  # PYT-AXIS
            f"band_i (prh_clip+l2) {bands} ; extrapolated (PR* < 124.5) {ex} ; "  # PYT-AXIS
            f"guard scan clean at start {g0.get('clean')} and end {g1.get('clean')} (recheck at freeze: guard.py scan)"  # PYT-AXIS
            + (" ; S0b TEST OUTPUT, NOT A PART B" if m.get("test_mode") else ""))  # PYT-AXIS


def build_md(F: dict, js_sha: str) -> list[str]:  # PYT-AXIS
    """Plain draft text of PARTB_PREDICTIONS_PYT.md; every number is read from partB_features_pyt.json."""  # PYT-AXIS
    PYT = F["pyt_lms"]  # PYT-AXIS
    SZ = F["sizes"]  # PYT-AXIS
    V = F["vits"]  # PYT-AXIS
    m = F["meta"]  # PYT-AXIS
    L = []  # PYT-AXIS
    a = L.append  # PYT-AXIS
    a("# S1.2 Pythia axis, Part B: instantiated rules from features only (DRAFT v1, generated by partb_pyt.py)")  # PYT-AXIS
    a("")  # PYT-AXIS
    a("Role `pyt-axis-predict`. B7 output of `analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md` (Part A, sha256 "  # PYT-AXIS
      f"`{m['plan_sha256']}`), section 9. Contract: `VERDICTS_SPEC_PYT.md` section 11. **DRAFT: the predict role reviews the "  # PYT-AXIS
      "wording, chmods this file and the JSON 444 and logs the `[pyt-axis-predict] Part B FROZEN` STATUS line "  # PYT-AXIS
      "(template: `partB/STATUS_LINE_DRAFT.txt`; the markdown hash must be recomputed after review).** Every number below is "  # PYT-AXIS
      f"read from `results/S1_2_layermax_pythia/partB/partB_features_pyt.json` (sha256 `{js_sha}` at generation).")  # PYT-AXIS
    if m.get("test_mode"):  # PYT-AXIS
        a("")  # PYT-AXIS
        a(f"**S0b TEST OUTPUT, NOT A PART B.** Stand-in `{m.get('standin')}`: "  # PYT-AXIS
          + ", ".join(f"{k} = {v}" for k, v in m.get("source_stems", {}).items()) + f". {m.get('tok1_label')}.")  # PYT-AXIS
    a("")  # PYT-AXIS
    a("## 0. Freeze conditions and provenance")  # PYT-AXIS
    a("")  # PYT-AXIS
    for tag in ("guard_scan_start", "guard_scan_end"):  # PYT-AXIS
        s = m.get(tag, {})  # PYT-AXIS
        a(f"- **{tag.replace('_', ' ')}** ({s.get('utc')}): clean = {s.get('clean')}; negative scan (plan §4 F1, 7 Pythia "  # PYT-AXIS
          f"slug patterns) hits = {s.get('n_find_hits')}; results top level = {s.get('results_top_level')}; forbidden "  # PYT-AXIS
          f"processes = {len(s.get('forbidden_processes', []))}. Full output: "  # PYT-AXIS
          f"`{os.path.relpath(s.get('log', ''), PROJ) if s.get('log') else ''}`."  # PYT-AXIS
          + (f" **SCAN ERROR: {s.get('scan_error')}**" if s.get("scan_error") else ""))  # PYT-AXIS
    a("- **No cross-model score involving a Pythia model existed or was computed.** Inputs: the feature stacks "  # PYT-AXIS
      f"({len(m['inputs'])} entries + {len(m['tok1_inputs'])} `_tok1` stacks, sha256 in `meta.inputs`), the frozen "  # PYT-AXIS
      f"Part B files `{m['llama_partB']['path']}` (`{m['llama_partB']['sha256'][:8]}…`) and `{m['ol_partB']['path']}` "  # PYT-AXIS
      f"(`{m['ol_partB']['sha256'][:8]}…`), and {m['n_score_inputs']} known pair files of the 11 known LMs (B5 only, "  # PYT-AXIS
      "`meta.score_inputs`).")  # PYT-AXIS
    r = m["recheck"]  # PYT-AXIS
    a(f"- **B1 recomputation of bloomz-560m, open_llama_13b, llama-30b and the 17 ViTs ({r['n_models']} models):** max "  # PYT-AXIS
      f"relative PR error {r['max_rel_err_PR']:.2e} (≤ 1e-14 required); clip thresholds bitwise equal: "  # PYT-AXIS
      f"{r['all_clip_bitwise']}; pass: {r['pass']}.")  # PYT-AXIS
    a("- **Clip thresholds** (`hrhv2.prep.prh_clip_threshold`, float64 all-layer stack), written to "  # PYT-AXIS
      "`clip_thresholds_partB_pyt.json`; V-P1 must reproduce them bitwise: "  # PYT-AXIS
      + ", ".join(f"{s} {m['clip_thresholds'][k]!r}" for s, k in zip(SZ, PYT)) + ".")  # PYT-AXIS
    a(f"- **Code:** `analysis/s12_layermax_pythia/partb_pyt.py` sha256 `{m['script_sha256']}`; threads {m['threads']}; "  # PYT-AXIS
      f"python {m['python']}, numpy {m['numpy']}; run {m['started_utc']} – {m['finished_utc']}.")  # PYT-AXIS
    a("")  # PYT-AXIS
    for pre in ("prh_clip+l2", "l2"):  # PYT-AXIS
        Q = F["per_pre"][pre]  # PYT-AXIS
        prim = pre == "prh_clip+l2"  # PYT-AXIS
        a(f"## {'1' if prim else '2'}. {'Primary' if prim else 'Secondary'} instance: {pre}")  # PYT-AXIS
        a("")  # PYT-AXIS
        a("| size | L_stored | PR* | u* (rel.) | runner-up ratio | PR(emb) | PR(last) | PR at r = 0.5 (layer) | PR_late* (u, rel.) | "  # PYT-AXIS
          "f(PR*) | band_i | extrapolated |")  # PYT-AXIS
        a("|---|---|---|---|---|---|---|---|---|---|---|---|")  # PYT-AXIS
        for s, k in zip(SZ, PYT):  # PYT-AXIS
            q = Q["lm"][k]  # PYT-AXIS
            b = Q["band"][s]  # PYT-AXIS
            a(f"| {s} | {q['L']} | {q['PR_star']:.2f} | {q['argmax']} ({q['argmax_rel']:.3f}) | {q['PR_star_over_second']:.4f} | "  # PYT-AXIS
              f"{q['PR_first_layer']:.2f} | {q['PR_last_layer']:.2f} | {q['fixed']['0.5']['PR']:.2f} ({q['fixed']['0.5']['layer']}) | "  # PYT-AXIS
              f"{q['PR_late_star']:.2f} ({q['u_late']}, {q['u_late_rel']:.3f}) | {b['f']:.6f} | [{b['lo']:.6f}, {b['hi']:.6f}] | "  # PYT-AXIS
              f"{'**yes**' if Q['extrapolated'][s] else 'no'} |")  # PYT-AXIS
        a("")  # PYT-AXIS
        a(f"- **Tr_P(B̄) = Spearman(size, PR*) = {Q['Tr_Bbar']:+.4f}** (exact from PR*: {Q['Tr_Bbar_exact_from_PRstar']}), so "  # PYT-AXIS
          f"**s = {Q['s']:+d}**" + (" (Tr = 0: s := +1, FLAGGED)" if Q["s_flag_zero"] else "") +  # PYT-AXIS
          f"; 5-size Tr_P(B̄) (≥ 410M) {Q['Tr_Bbar_5size']:+.4f}; Tr_P(B_late) {Q['Tr_Blate']:+.4f}, **s_late = {Q['s_late']:+d}**.")  # PYT-AXIS
        a("")  # PYT-AXIS
        a("| quantity | Tr_P | \\|Tr_P\\| ≥ 0.75 | median over ViTs of s_B·[B(6.9b) − B(70m)] (≥ 0.015) | material (B4) |")  # PYT-AXIS
        a("|---|---|---|---|---|")  # PYT-AXIS
        for name, lab in (("Bbar", "B̄"), ("Bfd_0.5", "B_fd(0.5)"), ("Bfd_1.0", "B_fd(1.0)"), ("Blate", "B_late")):  # PYT-AXIS
            mt = Q["materiality"][name]  # PYT-AXIS
            a(f"| {lab} | {mt['abs_tr'] * mt['s_B']:+.4f} | {mt['abs_tr'] >= 0.75 - 1e-9} | {mt['median_end_gap']:+.5f} | "  # PYT-AXIS
              f"**{'MATERIAL' if mt['material'] else 'not material'}** |")  # PYT-AXIS
        a("")  # PYT-AXIS
        a(f"- **Members (plan §5.4):** {', '.join(Q['members'])}; H2 branch {Q['H2_branch']}; B_late = B̄ for "  # PYT-AXIS
          f"{Q['n_Blate_equals_Bbar']}/7 models (" + ", ".join(f"{s} {Q['Blate_equals_Bbar'][s]}" for s in SZ) + ")"  # PYT-AXIS
          + ("; **H2 is not counted separately in the paper (same baselines as H1)**" if Q["H2_same_as_H1"] else "") + ".")  # PYT-AXIS
        a("- δ_b curve (materiality of B̄ / B_late with 3δ_b): " + ", ".join(  # PYT-AXIS
            f"δ_b = {k}: {Q['material_curve_db']['Bbar'][k]} / {Q['material_curve_db']['Blate'][k]}" for k in Q["material_curve_db"]["Bbar"]) + ".")  # PYT-AXIS
        g = Q["gaps"]["Bbar"]  # PYT-AXIS
        a("- B̄ median gaps B(larger) − B(smaller) over the 21 size pairs: " + ", ".join(  # PYT-AXIS
            f"{k} {v['median']:+.4f}" for k, v in g.items()) + ".")  # PYT-AXIS
        qu = Q["qualifying"]  # PYT-AXIS
        a(f"- P-P2a(ii): {qu['PP2aii_n']} comparisons with |B̄ gap| ≥ δ_b, need ⌈0.9n⌉ = {qu['PP2aii_need']} (⌈2n/3⌉ = "  # PYT-AXIS
          f"{qu['PP2aii_need67']} reported)" + (" (n < 6: 'not evaluable')" if qu["PP2aii_n"] < 6 else "") +  # PYT-AXIS
          f"; P-P5a: {qu['PP5a_n']} comparisons with B_late gap ≥ δ_b, need ⌈2n/3⌉ = {qu['PP5a_need']}"  # PYT-AXIS
          + (" (n < 6: 'not evaluable')" if qu["PP5a_n"] < 6 else "") + ".")  # PYT-AXIS
        b5 = Q["B5"]  # PYT-AXIS
        ro = b5["offset_retrodiction_187"]  # PYT-AXIS
        a(f"- **B5 (extras; outcomes known, not tests).** Spearman of PR* with log params {b5['spearman_PRstar']['logparams']:+.3f}, "  # PYT-AXIS
          f"width {b5['spearman_PRstar']['width']:+.3f}, depth {b5['spearman_PRstar']['depth']:+.3f}. Offset retrodiction on the "  # PYT-AXIS
          f"{ro['n']} known pairs: {ro['n_in_band']} inside f(PR*) ± h (all inside by construction; max |offset − f| "  # PYT-AXIS
          f"{ro['max_abs_dev_from_frozen_f']:.6f}). 11-LM refit {ro['refit']['a']:.6f} + {ro['refit']['b']:.6f}/√PR* (frozen "  # PYT-AXIS
          f"{ro['refit']['frozen']['a']} + {ro['refit']['frozen']['b']}/√PR*; |Δa| {ro['refit']['abs_diff_a']:.2e}, |Δb| "  # PYT-AXIS
          f"{ro['refit']['abs_diff_b']:.2e}; the frozen constants apply regardless).")  # PYT-AXIS
        d7 = Q["D7"]  # PYT-AXIS
        a(f"- **D7 (descriptive):** Tr_P(B̄_tok1) {d7['Tr_Bbar_tok1']:+.4f} against Tr_P(B̄) {d7['Tr_Bbar']:+.4f}; PR*_tok1/PR* "  # PYT-AXIS
          + ", ".join(f"{s} {d7['PR_star_tok1_over_PR_star'][s]:.4f}" for s in SZ) + f". {d7['label']}.")  # PYT-AXIS
        if prim:  # PYT-AXIS
            a("")  # PYT-AXIS
            a("Per-ViT B̄ (rows: ViTs; columns: sizes 70m … 6.9b):")  # PYT-AXIS
            a("")  # PYT-AXIS
            a("| ViT | PR* | " + " | ".join(SZ) + " |")  # PYT-AXIS
            a("|---|---|" + "---|" * len(SZ))  # PYT-AXIS
            for v in V:  # PYT-AXIS
                a(f"| {v} | {Q['vit'][v]['PR_star']:.1f} | " + " | ".join(f"{Q['Bbar'][k][v]:.4f}" for k in PYT) + " |")  # PYT-AXIS
        a("")  # PYT-AXIS
    Q = F["per_pre"]["prh_clip+l2"]  # PYT-AXIS
    a("## 3. Instantiated rules (primary prh_clip+l2, biased CKA, layer-max; s_cal primary, uncensored secondary)")  # PYT-AXIS
    a("")  # PYT-AXIS
    if Q["material_Bbar"]:  # PYT-AXIS
        a(f"- **H1 (gatekeeper, α = 0.05) is a member; s = {Q['s']:+d}.** D₁ = s·ΔTr_P(cka_biased, full, s_cal) on V(T, s_cal); "  # PYT-AXIS
          "p_H1 = max(p_perm over 5040 relabellings, p_mag from 2000 ViT-bootstrap draws of H₀: D₁ ≥ −0.10); confirmed iff "  # PYT-AXIS
          "p_H1 ≤ 0.05 and D₁ ≤ −0.10. Then Holm at 0.05 over the three secondary members.")  # PYT-AXIS
    else:  # PYT-AXIS
        a("- **H1 is NOT a member (B̄ not material): category E (not evaluable); Holm at 0.05 over the three secondary "  # PYT-AXIS
          "members directly (plan §5.4).**")  # PYT-AXIS
    if Q["H2_branch"] == "H2a":  # PYT-AXIS
        a(f"- **H2a (B_late material, s_late = {Q['s_late']:+d}):** D₂ = s_late·[Tr_P(M) − Tr_P(raw T, full)] on V(T, M); "  # PYT-AXIS
          "p = max(p_perm, p_mag); confirmed iff Holm rejects and D₂ ≤ −0.10.")  # PYT-AXIS
    else:  # PYT-AXIS
        a("- **H2b (B_late not material):** TOST on Tr_P(M) − Tr_P(raw T, full), margin 0.10; confirmed iff Holm rejects.")  # PYT-AXIS
    a("- **H3 / H4:** TOST on ΔTr_P(mknn, full, s_cal) / ΔTr_P(cka_debiased, full, s_cal), margin 0.10 (degenerate TOST read "  # PYT-AXIS
      "'no comparison changed order', not counted as a confirmed prediction).")  # PYT-AXIS
    a("- **P-P2a(i):** per model, the median over its 17 pairs of c_α − B̄ lies in band_i (table above); holds iff true for "  # PYT-AXIS
      f"≥ 6 of 7 models. **P-P2a(ii)** (consistency check): c_α order = B̄ order in ≥ {Q['qualifying']['PP2aii_need']} of the "  # PYT-AXIS
      f"{Q['qualifying']['PP2aii_n']} comparisons with |B̄ gap| ≥ δ_b.")  # PYT-AXIS
    a("- **P-P2b:** for each comparison (j < j′, v), 'below' iff s_cal(T_j′; B̄_j′ + o_lo,j′) < s_cal(T_j; B̄_j + o_hi,j); "  # PYT-AXIS
      "'above' iff s_cal(T_j′; B̄_j′ + o_hi,j′) > s_cal(T_j; B̄_j + o_lo,j); 'tie' iff both are 0 at every offset in the bands; "  # PYT-AXIS
      "else 'undecided'. Holds iff 0 wrong among ≥ 30 decided. Applied by `forecast_pyt.py` after S7.")  # PYT-AXIS
    a("- Uncensored band (secondary): μ̂ − B̄ ∈ [" + ", ".join(str(x) for x in F["mu_band"]) + "] for every model. Flat-band "  # PYT-AXIS
      f"sensitivity {Q['band_flat']}; h curve {list(Q['band_h_curve'])}.")  # PYT-AXIS
    a("")  # PYT-AXIS
    a("## 4. Next steps allowed by the plan")  # PYT-AXIS
    a("")  # PYT-AXIS
    a("- S6 (single-model precompute, V-P1/V-P2, then the gated V-P3/V-P4/V-P7) and S7 start only after the "  # PYT-AXIS
      "`[pyt-axis-predict] Part B FROZEN` STATUS line exists with the sha256 of this file and of the JSON (guard F1(a)).")  # PYT-AXIS
    return L  # PYT-AXIS


if __name__ == "__main__":
    main()
